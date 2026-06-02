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
- Negative-binomial NB2 with log link. The engine supports both a positive
  caller-supplied fixed size parameter `theta`
  (`MASS::negative.binomial(theta)`-style) and explicit glmer.nb-style outer
  estimation of `theta`.
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
Negative-binomial GLMMs use NB2 variance `V(mu) = mu + mu^2 / theta`.
Artifacts and fit summaries record the effective family parameter under
`glmm_fit_metadata.family_parameters.negative_binomial_theta` and its
provenance under
`glmm_fit_metadata.family_parameter_sources.negative_binomial_theta`
(`fixed` or `estimated`). Estimated-theta fits also record the initial theta,
outer iteration count, and convergence flag as family parameters. `dispersion()`
returns the effective `theta` for wrapper compatibility. It is not treated as a
residual scale and must not rescale VarCorr output.

`fast = true` is the supported fitting mode. It is a profiled fast-PIRLS
approximation: the current engine profiles the fixed effects through PIRLS
while optimizing covariance parameters on the profiled GLMM objective. This is
the MixedModels.jl `fast=true` family of behavior, not `lme4::glmer`'s joint
Laplace fit. It is faster, but it can be less accurate for inference when the
profiled approximation is stressed, especially overdispersed Poisson/binomial
models and observation-level random-effect models.

`fast = false` selects a labelled joint attempt: joint Laplace for
`n_agq <= 1`, and joint AGQ for valid single-scalar random-effect models with
`n_agq > 1`. It estimates `[β; θ]` on a joint objective with response
constants retained, records stationarity and covariance evidence when the
joint path certifies, and otherwise returns a labelled fast-PIRLS fallback.
NLopt builds use BOBYQA; dependency-light builds use native TrustBQ so users
still have a documented joint-Laplace route when fast-PIRLS is not adequate.
For dependency-light builds, caller `max_feval` is honored for the joint phase
so downstream wrappers can run bounded slow-audit attempts instead of waiting
for the full default budget.
The prerequisites for promoting any GLMM row — objective convention,
derivative/stationarity evidence, covariance-certificate compatibility,
fallback policy, and lockstep scorecard tests — are specified in
`docs/certified_joint_glmm_optimizer_contract.md`.

Fit-summary payloads and compiler artifacts must expose the effective GLMM
mode rather than requiring wrappers to infer it. The stable summary fields are:

- `estimation_method`: `fast_pirls_profiled`, `joint_laplace`, `joint_agq`, or
  `fallback_fast_pirls`.
- `objective_definition`: `profiled_glmm_deviance` for the supported fast
  path, `joint_glmm_laplace_deviance` for the joint Laplace path, and
  `joint_glmm_agq_deviance` for the joint AGQ path.
- `response_constants`: `dropped` for the supported fast path and labelled
  fallback, `included` for joint objectives.
- `n_agq`: the requested/effective quadrature count.
- `optimizer_convergence_status`: typed optimizer status label such as
  `converged`, `budget_exhausted`, `roundoff_limited`, or `failed`.
- `optimizer_feval`, `optimizer_max_feval`, `optimizer_fit_log_len`: optional
  evaluation-budget instrumentation for wrappers and parity ledgers.
- `fallback_status`: `fallback_fast_pirls` only when an uncertified joint
  attempt returned the deterministic fast-PIRLS fallback.

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
`lme4` parity. `fast=false` parity is row-scoped: only rows that pass the
joint objective/certificate/scorecard gate may be marked
`release_blocking_parity`.

The current documented-divergence rows are deliberate release exclusions, not
soft passes:

- `cbpp`, `contraception`, and `verbagg` are fast-PIRLS /
  profiled-objective rows. Some large rows match the MixedModels.jl
  `fast=true` objective, but they are not `lme4` joint-estimation parity rows.
  The `culcitalogreg` Laplace and AGQ rows are promoted separately through the
  labelled `fast=false` joint gates.
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
optimizer is TrustBQ, GLMM joint Laplace uses native TrustBQ, and remaining
GLMM fallback/profiled paths stay on the native fallbacks. PRIMA remains an
optional development backend, not a required runtime dependency.

Artifacts and summaries must record:

- Family and link.
- Family parameters, when applicable (including fixed or estimated NB2
  `theta` and its source).
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

### Distinguishable Failure Modes

A difficult GLMM result must let a downstream client tell these five
situations apart from the artifact alone. They are different problems with
different correct responses, so they must not collapse into a single "GLMM
did not converge":

| Failure mode | Stable signal | What it means |
| --- | --- | --- |
| Optimizer failure | `optimizer_nonconvergence` (incl. budget exhaustion) on the optimizer certificate / `ConvergenceVerdict` | The optimizer stopped without an acceptable convergence criterion; the point is not a certified optimum. |
| Approximation gap | `documented_divergence` class with the fast-PIRLS / profiled reference, plus `response_constants = dropped` | The fit is on the profiled fast-PIRLS objective family, not the joint Laplace/AGQ deviance; coefficient gaps versus `lme4` are an approximation difference, not an optimizer bug. |
| Weak identification | covariance KKT `WeakIdentification` classification / `FitStatus` not a clean interior or valid boundary | The local information is too weak to certify a clean interior or valid boundary interpretation; the result may still be reportable with caution. |
| Response-constant convention | `response_constants` field (`dropped` vs `included`) | The objective omits/retains response normalising constants; objective values are only comparable when both engines agree on this field. It is a *convention* difference, never reported as an optimizer or identification failure. |
| Separation-like behavior | `binomial_separation` diagnostic and/or `ConvergedPenalised` / `NotIdentifiable` `FitStatus` | The fixed-effect MLE is at or near non-existence (quasi-/complete separation); this is a structural data property, not optimizer noise. |

These signals are independent: a single fit can be, for example, a valid
profiled fast-PIRLS result (approximation gap) on a separation-like dataset
with the `dropped` convention, and each must remain separately readable.
Separation-like behavior must never be silently relabelled as optimizer
nonconvergence, and a `dropped`/`included` convention difference must never be
reported as a fit failure.

## Recovery Policy

There is no default, silent GLMM recovery. The KKT-guided boundary restart
documented in `docs/difficult_model_certification.md` is an LMM
covariance-space mechanism; it is not applied to GLMM fits. Any future GLMM
recovery behavior (joint-optimizer restart, penalised fallback for
separation-like rows, etc.) must be **opt-in or explicitly labelled in the
artifact** and must remain outside the release `lme4` parity gate until it
passes the external-engine parity gates in
`comparison/parity_scorecard.toml` and the lockstep contract tests. A
recovered or fallback GLMM result must record which path produced it and must
not be promoted to `release_blocking_parity` on the strength of the recovery
alone.

## R Layer Awareness

The R layer is downstream of this Rust contract. R can expose idiomatic errors,
warnings, print methods, and formula terminology, but it should not need to
guess model semantics. Rust artifacts must contain enough structured metadata
for R to explain family/link, approximation, optimizer backend, defaults,
refusals, and pathology diagnostics.
