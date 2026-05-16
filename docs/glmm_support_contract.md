# GLMM Support Contract

This document defines the Rust engine contract for generalized linear mixed
models (GLMMs). Downstream bindings, including R wrappers, may translate this
contract for their users, but this repository owns the Rust model semantics,
fit metadata, and artifact diagnostics described here.

## Supported Fits

The default build enables NLopt, but supported GLMMs must also fit under the
dependency-light `--no-default-features` build without requiring NLopt or any
system optimizer dependency. Supported families and links are:

- Bernoulli with logit link.
- Binomial with logit link, including case weights for grouped-binomial
  proportions.
- Bernoulli and Binomial with probit and complementary log-log links.
- Poisson with log and square-root links.
- Gamma with log link.

`Family::InverseGaussian` (and the Gaussian-GLMM non-identity link paths) are
**implemented but NOT certified for 1.0**: they exist in the engine but are
not validated to the cross-language parity standard the families above are
held to, and their finite-sample inference surface is intentionally
incomplete (e.g. parametric bootstrap explicitly refuses Gamma /
InverseGaussian / Normal). Treat them as experimental; they are not part of
the SemVer-covered GLMM contract for 1.0.

Offsets are fixed linear-predictor offsets. Observation weights are supported
where the family semantics define them, including binomial trial weights.

`fast = true` is the supported fitting mode. `fast = false` is reserved for a
future joint optimizer path and must return an explicit unsupported error
rather than silently selecting another algorithm.

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

## Optimizer Policy

The Rust crate's default release build enables NLopt. Dependency-light
downstream builds can use `--no-default-features`, where the native LMM
optimizer is TrustBQ and GLMMs use the existing native COBYLA / PatternSearch
backends. PRIMA remains an optional development backend, not a required runtime
dependency.

Artifacts and summaries must record:

- Family and link.
- Objective approximation boundary (Laplace or AGQ semantics).
- Requested/effective `n_agq`.
- Optimizer name and backend.
- Optimizer certificate status, return code, objective value, function
  evaluations, boundary evidence, and diagnostics.

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

## R Layer Awareness

The R layer is downstream of this Rust contract. R can expose idiomatic errors,
warnings, print methods, and formula terminology, but it should not need to
guess model semantics. Rust artifacts must contain enough structured metadata
for R to explain family/link, approximation, optimizer backend, defaults,
refusals, and pathology diagnostics.
