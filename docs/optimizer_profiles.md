# Optimizer Profiles

`mixeff-rs` has two supported optimizer profiles for downstream users and
packagers.

## Default Profile

The default Cargo feature set enables NLopt:

```sh
cargo build --release
```

This is the performance-oriented profile. LMM fits use NLopt-backed BOBYQA /
NEWUOA where available, while the Rust code still owns the profiled objective,
PLS factorization, diagnostics, and result surface. Use this profile when the
extra native dependency is acceptable and fastest iteration efficiency is the
priority.

## Native TrustBQ Profile

Downstream packages can disable default features:

```sh
cargo build --release --no-default-features
```

or, from another Cargo crate:

```toml
mixeff-rs = { version = "0.1", default-features = false }
```

This selects the native dependency-light LMM path: scalar random-intercept fits
use the scalar native optimizer, and multi-theta LMM fits use TrustBQ. GLMMs use
the native fallback path. No NLopt or CMake dependency is required by this
profile.

TrustBQ is not just a fallback for unavailable NLopt. It gives downstream
projects a stable pure-Rust optimizer option with the same model object,
diagnostics, covariance summaries, and inference surface. That matters for
binary distribution, embedded use, restricted build systems, and package
managers that prefer fewer compiled third-party dependencies.

## Wrapper Feature Matrix

Downstream packages should choose and pin a feature profile explicitly. This
keeps wrapper source builds stable if the Rust crate later changes its default
feature set.

| Consumer profile | Cargo features | Build/dependency posture |
| --- | --- | --- |
| Rust default | `default` (`nlopt`) | Performance-oriented Rust profile; requires the NLopt build surface. |
| Rust fast gemm | `features = ["nlopt", "faer-backend"]` | Opt-in acceleration profile for blocked-Cholesky gemm-heavy workloads. |
| Rust dependency-light | `--no-default-features` | Native TrustBQ/GLMM fallback profile with no NLopt or CMake dependency. |
| R wrapper initial CRAN build | `default-features = false` | Keep the first CRAN path dependency-light and predictable. |
| R wrapper performance builds | explicit `nlopt`; optional `faer-backend` after CI evidence | Suitable for R-universe, GitHub, local, or later CRAN-performance profiles. |
| Future Python wrapper | explicit feature pin chosen by wheel/source-build policy | Do not inherit crate defaults implicitly. |

`faer-backend` is not part of the default profile today. It may be certified as
an opt-in acceleration path independently of any default-promotion decision.
Promoting it into `default` requires separate wrapper packaging evidence:
dependency graph delta, source-build behavior on supported platforms, wrapper
CI coverage, benchmark benefit, and recertified parity fixtures under the
chosen default.

## TrustBQ Policy

The TrustBQ LMM policy is centralized in `trust_bq_model_family_policy` in
`src/model/linear.rs`.

| Model family | Theta dimension | Cross terms | Max evals | Reuse | Stop policy |
| --- | ---: | --- | ---: | --- | --- |
| Small theta / vector RE | `d <= 3` | full quadratic | 1000 | off | numeric `ftol` plus stable theta |
| Moderate theta | `4 <= d < 7` | diagonal only | 1000 | off | numeric `ftol` plus stable theta |
| Crossed / large theta | `d >= 7` | diagonal only | 475 | exact bit-key reuse | statistical stall band |

The current policy is conservative and benchmark-driven. Selective off-diagonal
cross terms were tested and rejected because they destabilized crossed-model
objectives inside the evaluation budget. Exact interpolation-sample reuse is
enabled only where it is safe and useful. The crossed/large default budget is
kept at 475 evaluations because the isolated objective benchmark showed cheap
crossed-small evaluations but the fit benchmark showed that 450 evaluations
missed the objective tolerance while 475 retained `objective_pass=true`.
Certificate-aware stopping is
available for scalar and 2x2 covariance certificates, but it refuses
invalid-boundary and weak-identification classifications.

### Sample Reuse Experiment Policy

Exact interpolation-sample reuse is policy-controlled. The production default
is [`TrustBqSampleReuse::FamilyPolicy`](../src/model/linear/mod.rs), which
keeps the table above unchanged. Two opt-in modes exist for benchmark and
diagnostic runs:

- `TrustBqSampleReuse::AllFamilies` enables exact reuse for every TrustBQ
  model family.
- `TrustBqSampleReuse::Disabled` disables exact reuse for every TrustBQ model
  family, including crossed/large theta.

These modes are audit-recorded through `OptimizerControl` and can change
optimizer traces or evaluation counts. They should not become defaults unless
fit results and certificates are unchanged and benchmark evidence shows a clear
trace or runtime benefit. The benchmark harness exposes the same control with
`MIXEFF_BENCH_TRUST_BQ_SAMPLE_REUSE=all` or `disabled`.

Scope: the override governs every native `LinearMixedModel` TrustBQ solve —
the main theta optimization, the diagonal warm start, and the active-face refit
sub-solve — through the shared `TrustBqSampleReuse::resolve` helper. It does not
reach the GLMM joint Laplace inner solve in `model::generalized`, which fixes
its own reuse policy independently of `OptimizerControl`.

### Start Ladder Policy

TrustBQ keeps the default start simple: it optimizes from the fitted model's
existing initial theta and relies on the centralized policy above for budget,
stall, reuse, cross-term, and certificate-stop behavior.

One ladder is now implemented as an **opt-in** control
(`OptimizerControl::with_trust_bq_start_ladder(TrustBqStartLadder::DiagonalFirst)`,
default `Off`): the diagonal-first / zero-correlation warm start. Stage one
optimizes the covariance with all off-diagonal theta pinned at exactly zero,
on a deliberately coarse budget (`family budget / 8`, clamped to 20–60
evaluations, accepted-step band `ftol_abs 1e-4` / `ftol_rel 1e-6`). Stage two
is the ordinary full-covariance TrustBQ run from the expanded stage-one
optimum with the full family budget, a contracted initial trust radius
(`policy radius / 8`; it re-expands on successful steps), and the unchanged
certificate-stop and boundary-diagnostic behavior. Both stages' evaluations
are counted in `feval` and the fit log, and an opted-in fit is audit-visible
via a `START_LADDER(diagonal_first:<n> evals): <status>` return value.

Benchmark evidence (2026-07-01, `optimizer_bench_harness`, native profile,
`MIXEFF_BENCH_TRUST_BQ_START_LADDER=diagonal_first`):

| Scenario | median ms (single-start → ladder) | status change |
| --- | --- | --- |
| vector_1000 | 1.07 → 0.92 | — |
| vector_10000 | 8.87 → 6.33 | — |
| vector_deep_200x50 | 3.32 → 2.41 | — |
| crossed_small | 6.81 → 6.46 | MAXEVAL_REACHED → FTOL_REACHED |
| crossed_medium | 31.4 → 27.0 | — (total fevals 437 → 516, wall still lower) |
| crossed_large | 110.5 → 76.1 | — |

All rows kept `objective_pass=true`; on a small 24-subject fixture the ladder
also repaired a ~0.52 single-start under-convergence to the NLopt reference
objective (see `test_trust_bq_diagonal_first_ladder_matches_single_start_objective`).
The ladder stays off by default: crossed_medium trades evaluations for wall
time, and default promotion should wait for external-parity refresh evidence
across both compile profiles.

The remaining candidate ladders are documented but not implemented:

| Target family | Candidate warm start | Default status | Promotion requirement |
| --- | --- | --- | --- |
| Crossed scalar intercepts | fit the largest single grouping term, then add remaining scalar terms | not implemented | lower `time_to_certified_fit` on crossed sparse rows with unchanged objective tolerance |
| Badly scaled predictors | internally scaled theta step/radius, not data mutation | not implemented | no ordinary-case slowdown and no coefficient/objective drift |

By default downstream users should continue to treat the TrustBQ profile as
single-start, certificate-aware, and policy-tuned; the diagonal-first ladder
is available for callers who opt in.

### Active-Face Refit (experimental, opt-in)

The "over-specified random slopes" candidate is now implemented as a
post-fit continuation rather than a warm-start ladder:
`OptimizerControl::with_active_face_refit(ActiveFaceRefit::Experimental)`
(default `Off`). After the primary optimizer stops, every fully
parameterized vector term's fitted relative covariance `ΛΛ'` is
eigendecomposed; eigenvalues under the compiler's
`effective_rank_tolerance` (the same cut the effective-covariance summaries
use) mark a lower-rank face. The refit holds the active eigenbasis `U`
(k×r) fixed and re-optimizes only the face factor `C` of `G = U (C Cᵀ) Uᵀ`
— `r(r+1)/2` coordinates instead of `k(k+1)/2` — with native TrustBQ under
the ordinary family policy, expanding each trial to theta through an exact
LQ re-triangularization of `W = U C` (never forming `G`). Detection and
refit iterate while the detected rank keeps shrinking (at most three
rounds), and a round is kept only when it strictly improves the objective.
At the final point every dropped eigendirection is probed by a forward
difference; the audit-visible return value records the outcome:
`ACTIVE_FACE(rank5of8:909 evals:uncertified): FTOL_REACHED` means a
rank-5-of-8 face, 909 face/probe evaluations (all counted in `feval` and
the fit log), and a probe that still found material descent off the face
(`certified` means it did not).

Benchmark evidence (2026-07-02, `active_face_bench`, release, default NLopt
primary path, `singular` fixture row `y ~ 1 + A * B * C + (A * B * C |
group)` REML; lme4 reference objective 766.554 at 4027 evals / 434 ms):

| Method | Objective | fevals | min wall ms | Status |
| --- | --- | --- | --- | --- |
| default | 822.357 | 10000 | 598 | MAXEVAL_REACHED |
| + active face | **764.675** | 10909 | 629 | ACTIVE_FACE(rank5of8:909 evals:uncertified): FTOL_REACHED |
| + active face, `with_max_feval(2000)` primary | 770.935 | 3428 | 156 | ACTIVE_FACE(rank5of8:1428 evals:uncertified): MAXEVAL_REACHED |

The refit turns the crate's worst documented-divergence LMM row from a
budget-exhausted stop 55.8 above the lme4 objective into a converged stop
1.88 *below* it for ~5% wall overhead, and detection is a no-op on
full-rank fits (`test_active_face_refit_noop_on_full_rank_fit` pins
byte-identical results). It stays off by default: the face basis is frozen
at a budget-bound iterate, the `uncertified` probe outcome above is real
(descent off the rank-5 face remains), and promotion should wait for
external-parity refresh evidence plus a policy for re-polishing in the full
space from the face optimum.

TrustBQ stop reasons are also mapped to a stable trace classification inside
`src/optimizer/trust_bq.rs`:

| Trace class | Stop reasons | Interpretation |
| --- | --- | --- |
| `smooth_convergence` | radius, objective, or step tolerance | Standard trust-region convergence. |
| `statistical_stall` | objective stagnation | The model-family policy accepted negligible objective movement inside its statistical stall band. |
| `certificate_accepted` | certified convergence | The caller-owned model certificate accepted the current best point; for LMMs this is where boundary-certified stops enter TrustBQ. |
| `budget_exhaustion` | max evaluations | The optimizer ran out of budget before a convergence or certificate stop. |

KKT-guided invalid-boundary restarts are orchestrated above TrustBQ in the LMM
fit path. They are reported as recovered convergence by the optimizer
certificate and compiler verdict surfaces rather than as a raw TrustBQ stop.

## Practical Tradeoff

NLopt remains the default performance profile because it is usually more
iteration-efficient. TrustBQ is the native profile because it is dependency
light, deterministic, and now competitive enough for serious downstream use.
The benchmark harness in `examples/optimizer_bench_harness.rs` is the current
source for profile comparisons.

For objective/factorization profiling, use
`examples/objective_eval_bench.rs`:

```sh
cargo run --release --features unstable-internals --example objective_eval_bench
cargo run --release --no-default-features --features unstable-internals --example objective_eval_bench
```

This harness builds and fits each model once, then repeatedly evaluates the
public `LinearMixedModel::objective_at` path at the fitted theta. It isolates
steady-state profiled-objective and PLS/factorization cost from model
construction and optimizer search. Because it intentionally uses the public
objective path, it should be read as a factorization-cost probe, not as a
private optimizer-fast-path microbenchmark.
