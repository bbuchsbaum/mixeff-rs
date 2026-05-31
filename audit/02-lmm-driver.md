# Audit 02/7 — LMM Fit Driver (src/model/linear.rs)

Scope: LinearMixedModel fit/optimizer/PLS/objective paths only. Read-only.
Reference cross-check: MixedModels.jl/src/linearmixedmodel.jl.
Auditor: Auditor 2/7.

## Verdict

The objective and θ→Λ→L→logdet chain is faithful to MixedModels.jl. The
release-risk surface is concentrated in (a) non-convergence being recorded as a
string but never gating `fit()`'s `Ok` return, (b) absence of the Julia
initial-objective rescue path, (c) the scalar golden-section bracketer missing
minima beyond θ0, and (d) the KKT boundary restart having no
"keep-best-of-pre/post" guard. No reachable panic from valid input was found in
the core fit path; `set_theta` lacks a finiteness guard but the optimizers do
not feed it non-finite values in the traced paths.

---

## Objective / parity (commendations)

- `objective_from_components` (linear.rs:2870-2885) matches Julia `objective`
  (linearmixedmodel.jl:826-833) exactly: profiled form
  `logdet + denomdf*(1 + log2π + log(pwrss/denomdf))` and fixed-σ form
  `logdet + denomdf*(2 log σ + log2π) + pwrss/σ²`.
- REML logdet: `logdet_lzz + 2*logdet_lxx` over the X-block diagonal
  (linear.rs:4034-4045) with `denomdf = n - p`; ML uses `logdet_lzz`, `denomdf
  = n`. Matches Julia `ssqdenom` / `logdet` semantics.
- Weighted Jacobian correction `objective_value = profiled - 2 Σ ln(sqrtwts)`
  (linear.rs:2897-2913) matches Julia's `val - 2 sum(log, wts)`.
- `pwrss = last_diag²` (linear.rs:4031-4032) matches Julia
  `abs2(last(last(m.L)))`.
- `finalize_fit_result` always re-runs `set_theta` + `update_l`
  (linear.rs:4438-4439) so `l_blocks`/`reterms` are in sync with the reported
  θ before any `coef`/`vcov`/`loglik` read. The fast-path objective
  (`objective_at_fast_or_generic`) that does not mutate `self` state is only
  used during the optimizer loop; finalize re-syncs. No stale-state read of
  coef/vcov/logLik was found on the normal fit path.
- θ-sign rectification (`rectify_theta_columns`, linear.rs:4472-4505) and the
  zero-snap pass (linear.rs:4443-4458) only accept the snapped point when its
  objective does not regress beyond `ftol_zero_abs`; otherwise it restores the
  pre-snap θ and re-syncs L. Correct.

---

## Findings

### HIGH-1 — `fit()` returns Ok on non-convergence; verdict only in a string
Files: linear.rs:4651-4658, 4701-4708, 5013-5019, 5029-5044, 5541, 5555-5641;
trust_bq.rs:107-112.

`fit()` returns `Ok(self)` regardless of optimizer outcome. Non-convergence is
recorded only as `optsum.return_value` ("MAXEVAL_REACHED", a COBYLA/NLopt fail
label, etc.). In the TrustBQ path the computed
`result.stop_reason.is_acceptable_convergence()` is bound to
`_trust_bq_diagnostics` (linear.rs:5013-5019) and discarded — the
budget-exhaustion verdict never influences control flow. There is no
`converged()` accessor on the fitted-model surface; callers must know to parse
`opt_summary().return_value`.

Reproduction: fit any model with `optsum.max_feval` set low enough to trip the
golden-section / pattern-search budget; `fit()` returns `Ok`, `coef()`/`vcov()`
return numbers computed at a non-optimal θ with no error or distinct status.

Impact: an RC can silently ship a non-optimal fit as if converged. This mirrors
MixedModels.jl's "store, don't throw" contract, but Julia exposes
`optsum.returnvalue` prominently and the R/inference contract here forbids
"silent" surfaces. Severity HIGH for an RC because the non-convergence signal
is effectively invisible at the trait boundary.

Suggested fix: add a `converged()` / `convergence_status()` to the fitted-model
trait derived from `return_value` + `is_acceptable_convergence`, and either make
`fit()` surface a non-fatal warning/typed status or document the required
`opt_summary()` check at the API boundary.

### HIGH-2 — No initial-objective rescue; non-finite `finitial` poisons the loop
Files: linear.rs:5605-5641, 5556-5572; cf. MixedModels.jl
linearmixedmodel.jl:480-491.

Julia detects a failed initial objective and rescales the initial guess before
retrying (`@info "Initial objective evaluation failed, rescaling..."`,
linearmixedmodel.jl:483-491). The Rust `fit()` does:
`self.optsum.finitial = self.objective_at_fast_or_generic(&theta0)?;`
(linear.rs:5607) with no finiteness check and no rescale-retry. If the default
or caller-supplied initial θ produces a non-PD Cholesky, `objective_*` returns
`f64::INFINITY` (or NaN via `log` of a non-finite intermediate). `finitial`
then seeds `best_fmin`:
- Scalar path: `best_fmin = finitial` (linear.rs:4535).
- Pattern search: `best_fmin = ftheta = finitial` (linear.rs:4733-4735).
- NLopt/PRIMA: closures fall back to `invalid_objective = finitial`
  (linear.rs:5224/5400) — the "barrier" is then INFINITY/NaN, not a finite
  reference, weakening trust-region behaviour relative to the comment at
  linear.rs:5499-5501.

With `finitial = NaN`, `obj < best_fmin` is always false (NaN compare), so
every optimizer can fail to update `best_theta` and finalize at the (bad)
initial θ while still returning `Ok`. The scalar bracketer's first decisions
(`fmid >= flo` at linear.rs:4563) also become undefined under NaN.

Reproduction: construct an RE design where θ0 (the lme4-style default) lands on
a non-PD region (degenerate/collinear Z for a vsize≥2 term). Compare against
Julia, which rescales and recovers.

Suggested fix: port the Julia rescue — if `finitial` is non-finite, scale θ0
toward the lower bound (or ×0.5 toward feasibility) and re-evaluate a bounded
number of times before erroring with a typed `Optimization` error rather than
silently proceeding.

### MEDIUM-1 — Scalar golden-section bracketer can miss a minimum beyond θ0
File: linear.rs:4537-4594 (`fit_scalar_single_theta`).

When `theta0 > 0` and `fmid >= flo` (objective at θ0 is not below objective at
0), the code sets `hi = mid = theta0` and the search bracket collapses to
`[a=0, b≈theta0]` (linear.rs:4563, 4590-4594). The expansion loop that probes
θ > θ0 is only entered when `fmid < flo` (linear.rs:4565). For a unimodal
profiled objective whose minimiser lies above θ0 but whose value at θ0 is not
yet below the value at 0 (flat-ish near the origin, common for small variance
components with a shallow basin), the true minimum is never bracketed and the
optimizer converges inside `[0, θ0]`, returning an inflated θ̂ / wrong σ̂.

Reproduction: a single scalar `(1|g)` term with a small-but-nonzero true
variance and a default θ0 chosen below the minimiser; compare θ̂/objective to
the NLopt/Julia result.

Suggested fix: always perform at least one forward expansion probe at
`θ0 + step` (and grow) before fixing `hi`, independent of the `fmid < flo`
test; or seed the bracket with an upper probe and only then run golden section.

### MEDIUM-2 — KKT boundary restart can finalize a worse fit than before restart
Files: linear.rs:2519-2602 (`apply_kkt_guided_boundary_restart`),
2628-2700 (candidates).

The restart *candidate* is only accepted if it improves the objective
(`objective + ftol < best_objective`, linear.rs:2644/2687). But after switching
`optsum.initial` to the candidate and re-running the optimizer
(linear.rs:2547-2582), there is no comparison of the post-restart objective
against the pre-restart `previous_optsum.fmin`. `previous_optsum` is consumed at
linear.rs:2531 to restore tolerances, not retained as a fallback. If the
re-optimization from the (better single point) candidate diverges or stalls at
a worse basin, `fit()` keeps the worse result and labels it
`KKT_BOUNDARY_RESTART(...)`. The restart is gated to scalar / 2×2 terms so blast
radius is bounded, but a "restart made it worse" regression is silent.

Suggested fix: snapshot `(previous_optsum.final_params, previous_optsum.fmin)`
and, after the restart optimizer returns, keep whichever of pre/post has the
lower objective (re-`set_theta`/`update_l` to that point) before stamping the
return value.

### MEDIUM-3 — COBYLA seeds invalid evaluations with INFINITY, not finitial
File: linear.rs:5084 vs 5248 (PRIMA) / 5424 (NLopt).

`fit_cobyla_with_maxeval` maps a failed `profiled_objective_from_parts` to
`f64::INFINITY` (linear.rs:5084), whereas PRIMA (5248) and NLopt (5424) use
`invalid_objective = finitial` as the soft barrier. Feeding INFINITY into
COBYLA's linear-model construction (especially near the start, before a finite
point is found) can produce a degenerate simplex / poor first model and waste
the eval budget or stall at the initial point. Inconsistent with the documented
barrier strategy used elsewhere in the same file.

Suggested fix: use the same `invalid_objective = self.optsum.finitial` (or a
large finite penalty relative to `finitial`) in the COBYLA closure for
consistency and robustness.

### LOW-1 — `set_theta` does not reject non-finite θ
File: linear.rs:2093-2109.

`set_theta` validates only `theta.len()`. A NaN/Inf θ passes through to
`ReMat::set_theta` and into the Cholesky. No traced optimizer path feeds
non-finite θ (bounds projection + finite initial steps), so this is latent, but
a defensive `theta.iter().all(f64::is_finite)` guard returning
`DimensionMismatch`/`Optimization` would harden the public API
(`set_theta` is `pub`).

### LOW-2 — `objective_from_components` does not sanitize NaN result
File: linear.rs:2870-2885.

Only the `fixed_sigma` branch guards `sigma` finiteness. With a NaN `pwrss` or
`logdet` the profiled branch returns NaN. Downstream optimizer comparisons
(`obj + ftol < best`) treat NaN as "not better" (safe for the loop) but a NaN
can still reach `finitial` (see HIGH-2) and `objective_value()` for a finalized
boundary fit. A `debug_assert`/explicit `if !result.is_finite() { INFINITY }`
clamp on the profiled branch would make the failure mode uniform.

### LOW-3 — Constant-response test uses strict `f64::EPSILON`
File: linear.rs:5567-5574.

`y_is_constant` requires `|yi - y0| < f64::EPSILON`. Responses that are
numerically constant up to rounding (e.g. `1.0` vs `1.0 + 3e-16` from upstream
arithmetic) escape the `ConstantResponse` guard and instead drive the optimizer
toward a degenerate Cholesky boundary (PosDefException-style refusal). The
summary-estimate short-circuit (linear.rs:5575-5594) only triggers for
`FixedSamplingVariance`. Consider a relative tolerance
(`<= 1e-12 * (1 + |y0|)`), matching the tolerance philosophy used elsewhere in
this file.

### Note (test-only) — Kenward-Roger parity tolerances loosened
Files: git diff src/model/linear.rs (tests mod, ~18636-18820),
docs/kenward_roger_contract.md, fixtures JSON.

The uncommitted diff is test-only: per-case data dispatch
(`kenward_roger_parity_data`) plus loosened KR assertion tolerances
(`epsilon=1e-6` → `1e-3 + 1e-4*|ref|`; multi-df `<=1.0` →
`<=1.0 + 2e-3*|ref|`). No change to fit-driver code. Not a fit-driver
correctness defect, but flagged for the RC owner: this widens the KR parity
band against pbkrtest and should be a conscious, documented decision (the doc
edit appears to cover it — confirm the contract doc states the new band).

---

## Things checked and found correct

- θ ordering / parmap-driven block layout is consistent between
  `apply_theta_to_reterms`, `set_theta`, `rectify_theta_columns`, and the fast
  paths; deterministic.
- Optimizer dispatch predicates (`use_scalar_single_theta_optimizer`
  3921-3923, `use_nlopt_bobyqa_small_theta_optimizer` 3926-3935,
  `use_large_theta_nlopt_optimizer` 3950-3952) are mutually consistent and
  chosen inside `fit` only, per the project's stated contract; no overlap that
  would non-deterministically pick a different optimizer for the same model.
- `fit()` `AlreadyFitted` guard (linear.rs:5556-5558) and
  `RankSaturatedFixedEffects` guard (5596-5601) are correct early refusals.
- Lower-bound feasibility: diagonal θ ≥ 0, off-diagonal unconstrained
  (`lower_bounds` 2111-2122); COBYLA constraints and bound projection
  (`project_theta_to_bounds` 3954-3960) are consistent with this.
- `fit_nlopt_large_theta` falls back to `fit_cobyla` on NLopt error
  (linear.rs:5547-5550) — graceful, no silent unconverged-as-success there
  beyond HIGH-1's general issue.
- Best-point tracking in NLopt/PRIMA/TrustBQ keeps the logged best vs the
  optimizer's returned point (e.g. linear.rs:5523-5531, 5021-5028) guarding
  against optimizers returning a non-best final iterate.
