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
incomplete (e.g. parametric bootstrap explicitly refuses InverseGaussian /
Normal). Treat them as experimental; they are not part of
the SemVer-covered GLMM contract for 1.0.

Offsets are fixed linear-predictor offsets. Observation weights are supported
where the family semantics define them, including binomial trial weights.

`fast = true` is the supported fitting mode. `fast = false` is reserved for a
future joint optimizer path and must return an explicit unsupported error
rather than silently selecting another algorithm.

## Parity Claim Classes

GLMM release evidence is not a single blanket "`lme4` parity" claim. The
machine-readable source of truth is `comparison/parity_scorecard.toml`, and
the generated comparison report must use the same distinctions:

- `release_blocking_parity`: fitted quantities are within the declared
  tolerance for the stated reference. For GLMM rows, objective values may be
  excluded from this comparison only when the scorecard explicitly records the
  response-constant convention difference.
- `documented_divergence`: the row is fitted and useful evidence, but it is
  not a release-blocking `lme4` parity row. Current examples include
  fast-PIRLS/profiled-objective rows that track the MixedModels.jl `fast=true`
  behavior while differing from `lme4` joint-estimation coefficients.
- `performance_known_slow`: the numerical claim is separately classified, but
  the row remains a release-visible performance issue with a mote id.
- `stress_opt_in`: the row is deliberately excluded from routine comparison
  regeneration unless `MIXEDMODELS_INCLUDE_STRESS=1` is set.
- `unsupported_with_contract`: the row exercises a behavior that Rust rejects
  by design and must carry a stable reason.

Tests must fail if a GLMM documented-divergence row is presented as ordinary
`lme4` parity. `fast=false` remains outside the supported contract until a
certified joint optimizer supplies its own parity fixtures.

The current documented-divergence rows are deliberate release exclusions, not
soft passes:

- `cbpp`, `contraception`, `culcitalogreg`, and `verbagg` are fast-PIRLS /
  profiled-objective rows. Some large rows match the MixedModels.jl
  `fast=true` objective, but they are not `lme4` joint-estimation parity rows.
  The `culcitalogreg` Laplace and AGQ rows have large fixed-effect gaps and
  must remain non-parity until a certified joint GLMM optimizer lands.
- `gopherdat2` has coefficient parity, but Rust records a near-zero covariance
  parameter without lme4's singular flag. This is a diagnostic threshold /
  convention gap plus the normal GLMM objective-constant difference.
- `grouseticks` is tracked separately as `performance_known_slow`: it has a
  MixedModels.jl `fast=true` objective contract and a known lme4 beta gap, but
  its release-visible issue is currently performance.

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
