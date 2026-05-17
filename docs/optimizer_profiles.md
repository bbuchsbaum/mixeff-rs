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

### Start Ladder Policy

TrustBQ currently keeps the default start simple: it optimizes from the fitted
model's existing initial theta and relies on the centralized policy above for
budget, stall, reuse, cross-term, and certificate-stop behavior. A
simple-to-complex start ladder is documented but not enabled by default because
the hard-case benchmark evidence so far points to stopping/budget policy as
the reliable speed win, while unvalidated restarts risk changing objectives or
masking boundary diagnostics.

The candidate ladder for future opt-in experiments is:

| Target family | Candidate warm start | Default status | Promotion requirement |
| --- | --- | --- | --- |
| Crossed scalar intercepts | fit the largest single grouping term, then add remaining scalar terms | off | lower `time_to_certified_fit` on crossed sparse rows with unchanged objective tolerance |
| Random intercept/slope blocks | diagonal/zero-correlation block before the full block | off | fewer fevals on vector-RE rows without losing valid rank-deficient certificates |
| Over-specified random slopes | certified lower-rank face from the covariance KKT certificate | off | active-face benchmark improves fevals or diagnostic stability and records active rank |
| Badly scaled predictors | internally scaled theta step/radius, not data mutation | off | no ordinary-case slowdown and no coefficient/objective drift |

Until one of those ladders has benchmark-backed evidence, downstream users
should treat the current TrustBQ profile as single-start, certificate-aware,
and policy-tuned rather than restart-driven.

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
