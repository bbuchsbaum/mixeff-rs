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

## Practical Tradeoff

NLopt remains the default performance profile because it is usually more
iteration-efficient. TrustBQ is the native profile because it is dependency
light, deterministic, and now competitive enough for serious downstream use.
The benchmark harness in `examples/optimizer_bench_harness.rs` is the current
source for profile comparisons.

For objective/factorization profiling, use
`examples/objective_eval_bench.rs`:

```sh
cargo run --release --example objective_eval_bench
cargo run --release --no-default-features --example objective_eval_bench
```

This harness builds and fits each model once, then repeatedly evaluates the
public `LinearMixedModel::objective_at` path at the fitted theta. It isolates
steady-state profiled-objective and PLS/factorization cost from model
construction and optimizer search. Because it intentionally uses the public
objective path, it should be read as a factorization-cost probe, not as a
private optimizer-fast-path microbenchmark.
