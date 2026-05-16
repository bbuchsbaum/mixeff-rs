# GLMM Support Contract

This document defines the Rust engine contract for generalized linear mixed
models (GLMMs). Downstream bindings, including R wrappers, may translate this
contract for their users, but this repository owns the Rust model semantics,
fit metadata, and artifact diagnostics described here.

## Supported Fits

The default build must fit supported GLMMs without requiring NLopt or any
system optimizer dependency. Supported families and links are:

- Bernoulli with logit link.
- Binomial with logit link, including case weights for grouped-binomial
  proportions.
- Poisson with log link.
- Gamma with log link.

Offsets are fixed linear-predictor offsets. Observation weights are supported
where the family semantics define them, including binomial trial weights.

`fast = true` is the supported fitting mode. `fast = false` is reserved for a
future joint optimizer path and must return an explicit unsupported error
rather than silently selecting another algorithm.

The comparison harness treats the supported GLMM surface as a routine product
path, not an experimental probe. Representative Bernoulli, Binomial, Poisson,
and Gamma rows are emitted by `examples/compare_rust.rs`, compared against
`lme4` by `scripts/compare_lme4.R`, and summarized in `comparison/REPORT.md`.
Stress-tier GLMM fixtures remain opt-in via `MIXEDMODELS_INCLUDE_STRESS=1`.

## Approximation Semantics

`n_agq <= 1` means the Laplace approximation. `n_agq > 1` means adaptive
Gauss-Hermite quadrature (AGQ) and is accepted only for exactly one scalar
random-effects term.

Vector-valued random-effects terms and multiple random-effects terms must reject
`n_agq > 1` before any optimizer evaluations. The artifact records a stable
`invalid_agq_request` diagnostic with the requested `n_agq`, random-effect term
summary, and refusal reason.

REML, Satterthwaite, and Kenward-Roger inference are LMM-only in this contract.
GLMM artifacts must report finite-sample LMM inference as unsupported.

## Fixed-Effect Covariance And Inference

Fitted GLMM artifacts must include the versioned
`mixedmodels.fixed_effect_covariance_matrix` payload. Until the engine certifies
a full model-based GLMM `V_beta` for the fitted approximation, the payload is an
explicit unavailable contract:

- `status = "unavailable"`.
- `reliability = "not_available"`.
- `matrix = null`.
- `reason` explains that GLMM fixed-effect covariance is not certified.
- `details.basis = "user_order"` and `details.rank`/`details.aliased` describe
  the fixed-effect design.

Do not encode `NaN` or `null` cells inside an otherwise numeric matrix. A
future available GLMM covariance payload must be a fully numeric, finite,
symmetric `p x p` matrix ordered exactly like `coef_names`.

Downstream bindings must not reconstruct a dense covariance matrix from
`stderror()` or coefficient-table standard errors. Those summaries may be shown
as provisional model summaries, but workflows requiring `L V L'` need a
certified covariance payload. LMM finite-sample methods, including
Satterthwaite and Kenward-Roger, remain contract-shaped unsupported for GLMMs.

## Optimizer Policy

The CRAN-friendly/default backend is native Rust. The default native optimizer
is COBYLA, and callers may request the in-tree bound-aware PatternSearch
backend. NLopt and PRIMA remain optional parity/development backends, not
required runtime dependencies.

Routine GLMM comparison rows must remain at least as fast as `lme4` by minimum
fit time unless a row is explicitly fenced as stress or non-comparable. The
speed gate also requires optimizer backend, return code, and function-evaluation
counts in the generated artifacts.

Artifacts and summaries must record:

- Family and link.
- Objective approximation boundary (Laplace or AGQ semantics).
- Requested/effective `n_agq`.
- Optimizer name and backend.
- Optimizer certificate status, return code, objective value, function
  evaluations, boundary evidence, and diagnostics.

## Parametric Bootstrap

`parametricbootstrap_glmm()` is a full-model parametric bootstrap for fitted
GLMMs. It simulates from the fitted conditional mean and refits a cloned model
for each replicate, preserving offsets, case weights, optimizer options, and the
effective AGQ setting.

Family draw semantics:

- Bernoulli: draw `y* ~ Bernoulli(mu)`.
- Binomial: draw successes from `Binomial(trials, mu)`, where integer case
  weights are the trial counts, and store the response as `successes / trials`.
- Poisson: draw `y* ~ Poisson(mu)`.
- Gamma: draw `y* ~ Gamma(shape = 1 / phi, scale = mu * phi)`, with
  `phi = dispersion(true)`.

Gamma GLMM bootstrap must never fall back to Gaussian residual simulation.
Families without a certified response simulator return an explicit
`Unsupported` error.

## Diagnostics

GLMM diagnostics must use stable diagnostic codes and formula/user-facing terms
where available. The current contract includes:

- `optimizer_nonconvergence` for fitted optimizer stops that are not acceptable
  convergence criteria, including budget exhaustion.
- `invalid_agq_request` for rejected AGQ shape requests.
- `pirls_failure` for final PIRLS update failures after optimizer selection.
- `boundary_parameter` for theta values on lower bounds.
- `near_unit_random_effect_correlation` for fitted random-effect correlations
  whose absolute value is near one.
- `binomial_separation` for conservative fixed-effect separation diagnostics.

Separation diagnostics must not fire for rare binary predictors that have both
outcomes represented at the rare level.

## Release Checklist

Before shipping a GLMM-facing release, regenerate the comparison artifacts and
run the focused GLMM gates:

```bash
cargo run --release --example compare_rust
Rscript scripts/compare_lme4.R
cargo run --release --example compare_report
cargo test --test glmm_comparison_gates
cargo test --test glmm_speed_parity
cargo test --test glmm_artifact_contract
cargo test --test glmm_diagnostics
cargo test --test parity_agq
cargo test --test parity_gamma_glmm
```

For performance regressions in large crossed GLMM rows, use the focused
profiler:

```bash
MIXEDMODELS_PROFILE_REPEATS=100 cargo run --release --example profile_grouseticks_glmm
```

Any remaining numeric mismatch in `comparison/REPORT.md` must be either an
ordinary tolerance pass or an explicitly classified row with a stable reason.
Do not close a GLMM release while the report contains unclassified numeric
disagreements or a routine speed row slower than `lme4`.

## R Layer Awareness

The R layer is downstream of this Rust contract. R can expose idiomatic errors,
warnings, print methods, and formula terminology, but it should not need to
guess model semantics. Rust artifacts must contain enough structured metadata
for R to explain family/link, approximation, optimizer backend, defaults,
refusals, and pathology diagnostics.
