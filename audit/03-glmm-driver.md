# Audit 03/7 — GLMM Driver & Model Plumbing

**Auditor:** 3/7 (release-candidate hardening)
**Scope:** `src/model/generalized.rs`, `batch.rs`, `fixed_design.rs`, `data.rs`, `traits.rs`, `summary_estimates.rs`, `mod.rs`
**Method:** real call-path tracing; cross-checked PIRLS/AGQ/loglik against `MixedModels.jl/src/generalizedlinearmixedmodel.jl` and `mixedmodel.jl`.
**Verdict:** RELEASE CANDIDATE UNWISE pending CRITICAL #1 (incorrect AIC/BIC/logLik on the default fit path).

---

## CRITICAL

### C1. `loglikelihood()` / `aic` / `bic` are on the dropped-constant deviance scale for the default fast fit path
**Files:** `src/model/generalized.rs:3219-3221`, `:3209-3217`, `:2141`, `:1080-1082`, `:962-975`; trait defaults `src/model/traits.rs:200-215`.

Call path for the public default fit:
`fit()` (`:1349`) → `fit_with_options(true,1,false)` (`:1402`) → `fit_native_pattern_search`/`fit_native_cobyla` → `finalize_theta_after_optimizer` (`:2128`) → `optsum.fmin = self.deviance(n_agq)` (`:2141`).
For `n_agq <= 1`, `deviance()` returns `laplace_objective()` (`:1080-1082`), which sums `dev_resid_component` (`:962-975`) — i.e. the **deviance-residual scale that drops the response normalizing constants** (for Poisson it drops `−2·Σ ln Γ(y+1)`; for Binomial it drops `−2·Σ ln C(n,k)`).

Then `MixedModelFit::loglikelihood()` is `-self.objective()/2` (`:3219-3221`), and `objective()` returns the cached `optsum.fmin` (`:3209-3217`). The trait `aic`/`aicc`/`bic` (`traits.rs:200-215`) are derived from this. **Result: `loglikelihood`, `aic`, `aicc`, `bic` for every default-path GLMM fit (Poisson, Binomial/Bernoulli) are offset from the true value by the dropped per-observation constant** (`Σ ln(y!)` for Poisson; `Σ ln C(nᵢ,kᵢ)` for Binomial).

Cross-check with the reference: Julia `StatsAPI.loglikelihood(m::GeneralizedLinearMixedModel)` (`generalizedlinearmixedmodel.jl:516-538`) accumulates `GLM.loglik_obs(d, y, mu, wt, ϕ)` (the full normalized log-density) and subtracts `(Σ‖u‖² + logdet)/2`. It is **not** `-deviance/2`. The Rust code already contains a correct full-likelihood helper, `minus_two_loglik_observation` (`:984-1031`) and `laplace_objective_with_response_constants` (`:1058-1064`), but the default fast path never stores it into `fmin`; only the `#[cfg(feature="nlopt")]` joint path uses `deviance_with_response_constants` (`:1562`, `:1628`). So in the default build, AIC/BIC reported by `model_summary`, LRT, and any client (the R layer) are wrong by a fixed additive constant.

Why this blocks RC: AIC/BIC differences are the headline numbers users compare against lme4/`MixedModels.jl`. A Poisson GLMM will silently disagree with lme4 `AIC()` by `2·Σ ln(yᵢ!)` (often hundreds–thousands of units). Model selection (`LRT`/`AIC` comparisons) *within* the crate stays internally consistent only if every compared model uses the identical dropped constant and identical n — true for nested GLMM-vs-GLMM but **not** for GLMM-vs-LMM or cross-engine parity, which the project explicitly co-develops (`CLAUDE.md` parity workflow) and `docs/mixed_model_compiler_inference_contract.md` ("no fake numbers").

**Reproduction:** Fit `y ~ 1 + x + (1|g)` Poisson via `GeneralizedLinearMixedModel::new(...).fit()`. Compare `model.aic()` to `glmer(... , family=poisson)` `AIC()` / Julia `aic(m)`. Expect a discrepancy ≈ `2·Σ ln(yᵢ!)`.

**Suggested fix:** Make the default path store the response-constant objective for likelihood reporting. Either (a) in `finalize_theta_after_optimizer` set `optsum.fmin = self.deviance_with_response_constants(n_agq)` (and shift `finitial` consistently), or (b) override `MixedModelFit::loglikelihood` for the GLMM to compute `-(Σ minus_two_loglik_observation + u_penalty + logdet)/2` directly (mirroring Julia `loglik_obs` + RE penalty) instead of `-objective()/2`, decoupling the optimizer objective from the reported likelihood. Option (b) matches the Julia structure most closely and leaves the optimizer surface untouched.

---

## HIGH

### H1. PIRLS can accept a non-converged conditional mode silently (no non-convergence signal)
**File:** `src/model/generalized.rs:740-852`, `:2993-2995`, `:836-847`.

`pirls()` runs `max_iter = 10` (`:747`) and breaks on `pirls_converged(obj, obj0, tol=1e-5)` (`:836`, `:2993`). If neither convergence nor step-halving collapse occurs within 10 iterations, the loop **just ends and returns `Ok(())`** with whatever the last iterate was — no diagnostic, no flag, no error. Compare Julia `pirls!` (`generalizedlinearmixedmodel.jl:614-669`): same `maxiter=10` default and the same "no hard error if it simply runs out", **but** the Julia step-halving branch throws `ErrorException("number of averaging steps > 10")` when `iter < 2` (`:649-651`). The Rust port’s halving loop instead just `break`s out of the while when `nhalf == max_halvings` (`:817`, `while obj > halving_bound && nhalf < max_halvings`) with **no `iter < 2` hard-failure branch at all** — a diverging first iteration is accepted instead of erroring as in the reference.

Impact: For stiff GLMM surfaces (separation, large random-effect variance) the outer optimizer sees a `obj(θ)` evaluated at a *non-converged* inner mode, biasing θ̂/β̂ and the reported deviance, with no surfaced `PirlsFailure` diagnostic (that diagnostic is only emitted when `update_pirls_at_theta` returns `Err`, `:2133-2136`, which this path never does). This is the exact "convergence criteria that can accept a non-converged conditional mode" failure called out in scope.

**Suggested fix:** Track whether the final iteration actually satisfied `pirls_converged` (or `nhalf` hit `max_halvings`). On non-convergence either (a) port the Julia `iter < 2 → throw` behavior so `update_pirls_at_theta` propagates `Err` (which already records `record_pirls_failure_diagnostic`), or (b) at minimum emit a non-fatal `Diagnostic` (warning) and set a non-convergence flag on the optimizer certificate so the RC does not present a silently non-converged fit as clean.

### H2. `dev_resid_component` Gamma branch can produce NaN/negative deviance from a valid response
**File:** `src/model/generalized.rs:723-729` (and consumed by `laplace_objective` `:966-968`, `deviance` `:1112-1158`).

```
Family::Gamma => {
    if y == 0.0 { 2.0 * (mu.ln()) }
    else { -2.0 * ((y / mu).ln() - (y - mu) / mu) }
}
```
The Gamma deviance residual component is the standard `2[(y-μ)/μ − ln(y/μ)]`, which equals the written `-2*((y/mu).ln() - (y-mu)/mu)`. That is fine. **But** `mu` is *not* bounded here (unlike the Bernoulli branch which clamps). During PIRLS a Gamma+Log or Gamma+Inverse iterate can transiently produce `mu <= 0` (Inverse link `1/eta` with `eta < 0` → negative μ; Log link is safe). `mu.ln()` then yields NaN, `(y/mu).ln()` yields NaN for `mu<0`, and `laplace_objective` propagates NaN into the optimizer. The optimizer guards (`penalized_pirls_deviance_at_theta` maps non-finite → +∞, `:2202-2206`) catch it *at the θ level*, but the AGQ sweep (`deviance`, `:1146-1158`) and `laplace_objective` used inside `pirls()` step-halving (`:812`, `:829`) compare `obj` ordering with NaN — `obj > halving_bound` is `false` for NaN, so step-halving is skipped and a NaN iterate is accepted as "not worse". The `y == 0.0` branch (`2.0*mu.ln()`) is also wrong-signed/degenerate (Gamma has support y>0; for y→0 the deviance term diverges, not `2 ln μ`).

Impact: dispersion families are gated behind a "results not reliable" warning in Julia (`generalizedlinearmixedmodel.jl:395-399`); this crate **does not emit any such warning** (no analogue found) yet still permits Gamma/InverseGaussian fits, and the Gamma deviance can go NaN/negative for `Inverse` link. Combined with H1 (NaN passes the `obj > bound` halving guard), a Gamma+Inverse GLMM can return a fit built on NaN-contaminated inner objectives.

**Suggested fix:** (a) Bound `mu` in the Gamma/InverseGaussian branches of `dev_resid_component` (and `bounded_pirls_mean_and_eta` currently only bounds Bernoulli/Poisson — extend to Inverse-linked dispersion families). (b) In `pirls()` step-halving and convergence, treat non-finite `obj` as "worse" explicitly (`!obj.is_finite() || obj > halving_bound`) so a NaN iterate triggers halving instead of silent acceptance. (c) Port the Julia dispersion-family reliability warning as a `Diagnostic`.

### H3. `is_single_scalar_re` AGQ contract uses `vsize == 1` but does not require a single *grouping* — random-slope-free check is incomplete for crossed scalar REs
**File:** `src/model/generalized.rs:1193-1195`, `:1208-1219`, `:1080-1168`.

`is_single_scalar_re()` = `reterms.len() == 1 && reterms[0].vsize == 1`. `validate_agq` correctly rejects `n_agq>1` when this is false, and `deviance()` has a `debug_assert!` (`:1084`) — **but `debug_assert!` is compiled out in release**. In a release build, if any caller reaches `deviance(n_agq>1)` without `validate_agq` (e.g. a future direct call, or `refit_with_options` path where state changed), the AGQ code indexes `self.u[0]`, `self.lmm.reterms[0]`, `l11_diag()` assuming one scalar term and would produce a silently wrong quadrature rather than panicking. All *current* public entry points (`fit_with_options:1408`, `refit_with_options:1370`) do call `validate_agq` first, so this is HIGH-not-CRITICAL, but the release-mode invariant rests entirely on every caller remembering to preflight; the in-function guard is a no-op in release.

**Suggested fix:** Replace the `debug_assert!` at `:1084` with a hard `if !self.is_single_scalar_re() { return f64::NAN }` or return `Result`, so the AGQ contract is enforced in release builds, not just debug.

---

## MEDIUM

### M1. `LinkFunction::Probit` link/linkinv/mu_eta panic risk on boundary μ
**File:** `src/model/traits.rs:81-85`, `:101-105`, `:121-125`.

`Probit::link(mu)` uses `statrs ... Normal::inverse_cdf(mu)`. The `bounded_pirls_mean_and_eta` clamp (`:2960-2965`) only protects the Bernoulli/Binomial *PIRLS working step* (it clamps μ to `[1e-15, 1-1e-15]` before calling `link`). However, `Probit` is also reachable for `Bernoulli|Binomial` with Probit link via `dev_resid_component`→ no, that uses μ directly; and via `update_eta`→`linkinv` (safe, `cdf` total). The exposed risk: `statrs` `inverse_cdf(0.0)`/`inverse_cdf(1.0)` returns ±inf rather than panicking in current statrs, so this is a numerical (inf-propagation) rather than panic concern, but `Normal::new(0.0,1.0).unwrap()` is constructed on every scalar call (`:83`, `:103`, `:123`) — a perf smell on the hot PIRLS path (per-observation, per-iteration).

**Suggested fix:** Construct the standard normal once (lazy static) and clamp the Probit `link` argument to `[1e-15, 1-1e-15]` to match the Bernoulli deviance clamp; document the inf behavior.

### M2. `Cloglog::mu_eta` returns `exp(eta - exp(eta))` which underflows to 0 for moderately negative eta but is otherwise correct; large negative eta gives 0 weight
**File:** `src/model/traits.rs:126-134`.

`mu_eta = exp(eta − exp(eta))`. For `eta` very negative, `exp(eta)→0`, so `mu_eta→exp(eta)→0`; the PIRLS weight `dmu_deta²/V(μ)` then →0 and `pirls_working_observation` (`:2938-2942`) sets `resid = 0` when `dmu_deta.abs() < 1e-15`. This is defensible (matches GLM behavior) but combined with H1 a whole group going to zero weight can stall PIRLS without a surfaced signal. Tests at `traits.rs:323-335` only check finiteness/nonnegativity at round-tripped η, not the degenerate-weight stall. Lower severity because it is numerically faithful to the reference link; flagged for the convergence interaction.

### M3. `simulate_response` Bernoulli uses `rng.gen::<f64>() < p` — biased vs Binomial(1,p) at p exactly 0/1 boundaries, and Normal/InverseGaussian unsupported
**File:** `src/model/generalized.rs:592-647`.

`Family::Normal | InverseGaussian` return `Unsupported` (acceptable, documented). The `unreachable!()` at `:646` is genuinely unreachable (refused at `:544-553`) — fine. Bernoulli `f64::from(rng.gen::<f64>() < p)` with `p` clamped to `[0,1]`: at `p==1.0`, `gen()<1.0` is true with prob 1 (ok); at `p==0.0`, `gen()<0.0` is false (ok). No bug, but the parametric bootstrap silently cannot cover Normal-as-GLM / InverseGaussian, which `stats::bootstrap` callers must handle. Confirm `stats::bootstrap` surfaces this `Unsupported` rather than treating it as a fit failure. (Outside this audit's file scope; flagged for Auditor owning `stats/`.)

### M4. `dof()` correctly mirrors Julia but `dispersion_parameter` semantics differ for Binomial/Bernoulli
**File:** `src/model/generalized.rs:3169-3173`, `traits.rs:55-60`.

Rust `dof = feterm.rank + parmap.len() + has_dispersion?1:0`, `has_dispersion = Normal|Gamma|InverseGaussian` (`traits.rs:55-60`). Julia `dof = feterm.rank + length(parmap) + dispersion_parameter(m)` where `dispersion_parameter(d)` is false for Bernoulli/Binomial/Poisson, true otherwise — equivalent. **Consistent with reference.** No action; noted as a positive cross-check (the dof itself is correct, which makes C1's logLik error a pure additive-constant error, not compounded by dof).

---

## LOW

### L1. `gh_norm` hard-coded n=2 nodes `{-1, +1}` differ from eigen-derived order but weights symmetric — verify against Julia GHnorm(2)
**File:** `src/types/gauss_hermite.rs:139-153`. Pre-seeded `k=2: z=[-1,1], w=[0.5,0.5]`; `k=3: z=[-√3,0,√3], w=[1/6,2/3,1/6]`. These match `GaussHermiteNormalized` analytic values and Julia’s hard-coded `GHnorm` specials. The general path (`:50-96`) symmetrizes via `(vals - reverse(vals))/2` — robust. No correctness issue found; the AGQ `z==0` Laplace special-case (`:1133-1139`, `mult += w`) correctly mirrors Julia `iszero(z)` branch (`generalizedlinearmixedmodel.jl:96-98`). Positive cross-check.

### L2. `CategoricalColumn::new` first-appearance encoding matches reference; empty input is safe
**File:** `src/model/data.rs:234-256`. Iterates values, assigns `levels.len()` as next id — first-appearance order, matching `CLAUDE.md` ("Categorical levels are encoded by first-appearance order"). Empty `values` → empty `levels`/`refs`, no panic. Degenerate single-level column is handled at the contrast layer (`require_min_levels` errors with ≥2 requirement, `data.rs:194-199`) rather than panicking. Positive cross-check; no reachable panic from valid/degenerate input here.

### L3. Per-call `Normal::new(0.0,1.0).unwrap()` in Probit (traits.rs:83/103/123) and `WaldConfint` (`traits.rs:276`) — perf smell, `unwrap()` on a literally-infallible constructor is acceptable but repeated on hot path.

### L4. `objective()` pre-fit returns `laplace_objective()` (dropped-constant) while post-fit returns `optsum.fmin`; if C1 is fixed via option (a) these two scales must be reconciled or `loglikelihood()` pre-fit will silently differ in scale from post-fit. Mark as a follow-up constraint on the C1 fix.

---

## Positive Observations / Commendations
- AGQ sweep state restoration via `AgqRestoreGuard` (`:2382-2415`) is panic-safe (restores `u`/η/μ on unwind) — good defensive design.
- AGQ math (`deviance`, `:1080-1168`) faithfully mirrors Julia `deviance(m, nAGQ)` including the `z==0` Laplace shortcut, per-group `sd = 1/|L₁₁diag|`, and the `exp((z² + devc0 − devc)/2)` accumulation. Cross-checked line-by-line against `generalizedlinearmixedmodel.jl:84-109`.
- Normal+Identity is correctly rejected at construction (`:309-313`) mirroring Julia’s `ArgumentError` redirect (`generalizedlinearmixedmodel.jl:357-367, 391-393`). Constant-response rejection (`:335-342`) also mirrors Julia (`:273-275`).
- Family/link support matrix is explicitly validated (`validate_supported_glmm_family_link`, `:3031-3059`) and response-domain validated per family (`:3061-3081+`) — stronger input hygiene than the reference.
- The fast-vs-joint divergence IS surfaced, not silently wrong: `uncertified_joint_fallback` (`:2724-2783`) emits an `OptimizerRecovery` warning diagnostic with `scorecard_class="documented_divergence"` and a labelled return code. This directly addresses the project-memory concern; well handled.
- `dof()` correctly matches the Julia reference (see M4), isolating C1 to a clean additive constant.
- Case-weight/offset validation (`:2997-3029`) rejects non-finite/non-positive inputs early.

---

## RC Recommendation
**Do not ship as-is.** C1 (wrong AIC/BIC/logLik on the default fit path) is a release blocker for a crate whose stated purpose is lme4/MixedModels.jl parity and whose inference contract forbids misleading numbers. H1/H2 are correctness hazards on stiff/dispersion-family surfaces that can silently accept bad inner solutions. H3 weakens an AGQ safety invariant in release builds. Address C1 + H1 + H2 before RC; H3/M-items can be fast-followed with tracked diagnostics.
