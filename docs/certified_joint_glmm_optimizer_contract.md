# Certified Joint GLMM Optimizer Contract

Status: row-scoped implementation contract. Labelled joint Laplace and joint
AGQ paths are wired to `fit_with_options(fast = false, ...)` when the NLopt
backend is enabled, but rows are certified only when their objective,
stationarity, covariance, fallback, and scorecard evidence pass this gate.

This note specifies what a *certified joint GLMM optimizer* must expose before
any GLMM row may be promoted from `documented_divergence` to
`release_blocking_parity` against `lme4`. It exists so that the current
fast-PIRLS / profiled-objective path stays honestly labelled (see
`docs/glmm_support_contract.md`) and so that the eventual joint path is built
to extend — not bypass — the existing certificate stack
(`docs/difficult_model_certification.md`).

"Joint" here means the optimizer estimates the fixed effects `β` and the
covariance parameters `θ` against the same objective that `lme4`/glmmTMB
optimise (Laplace or AGQ marginal deviance with response constants retained),
rather than the current profiled fast-PIRLS surface that holds `β` at the
inner PIRLS solution and tracks the MixedModels.jl `fast=true` family.

## Why the current path is not certifiable parity

The fast-PIRLS profiled path:

- Optimises a profiled objective whose constant terms differ from the joint
  marginal deviance (the `response_constants = dropped` convention).
- Matches the MixedModels.jl `fast=true` objective family on the large rows
  (`contraception`, `verbagg`) but diverges from `lme4` joint-estimation
  coefficients.
- Has inference-impacting fixed-effect gaps on small-N rows
  (`culcitalogreg`), which is why those rows are explicitly held non-parity.

A certified joint optimizer removes the *objective* and *β-estimation*
mismatch. It must therefore satisfy every requirement below before its rows
re-enter the `lme4` parity gate.

Current evidence: certification is row-scoped. The fixed-beta conditional
PIRLS solve now evaluates the same included-constants joint Laplace objective
as `lme4` at the `cbpp` optimum, and `culcitalogreg` Laplace plus AGQ have
passed the labelled joint-promotion gates. `cbpp` and `contraception` still
remain below the promotion line because their fitted estimates miss row
tolerances.

## Required surface

### 1. Objective

The optimizer must expose, as structured artifact metadata:

- The exact objective it minimises, named and versioned: joint Laplace
  deviance for `n_agq <= 1`, joint AGQ deviance for `n_agq > 1`, with the
  number of quadrature points recorded.
- Whether response normalising constants are **retained** (`response_constants
  = included`). A certified joint row must use the `included` convention so
  that `objective_delta` against `lme4` is meaningful. The
  `response_constants` field must never silently flip between rows of the same
  release class.
- The dispersion / scale handling for Gamma and inverse-Gaussian families,
  stated explicitly (profiled vs jointly estimated), because the objective is
  not comparable across conventions.
- Family, link, offset, and prior-weight handling, sufficient for a downstream
  client to reproduce the objective value.

### 2. Gradients or derivative checks

The optimizer must expose at least one of, and record which:

- Analytic or automatic-differentiation gradients of the joint objective with
  respect to `(β, θ)`, with a finite-difference cross-check residual recorded
  in the certificate; or
- A documented derivative-free certificate with an explicit stationarity
  check (objective-difference probe along feasible directions) of the same
  kind the LMM covariance KKT certificates already use.

A bare "optimizer returned success" code is not acceptable evidence. The
artifact must carry a stationarity / first-order residual and the tolerance it
was judged against, so that "certified" means the same thing it already means
for the LMM path.

### 3. Covariance certificate compatibility

The joint optimizer must reuse the existing certificate vocabulary, not a
parallel one:

- It must populate the existing `compiler::audit::OptimizerCertificate`
  (optimizer stop, objective, parameter-space state, derivative evidence,
  checks, diagnostics) and project to the existing
  `compiler::report::ConvergenceVerdict`.
- Boundary and rank-deficient covariance fits must be classified through the
  existing covariance-cone vocabulary
  (`InteriorConverged`, `ValidZeroVariance`,
  `ValidRankDeficientCovariance`, `InvalidBoundaryStop`,
  `WeakIdentification`). If the GLMM covariance-cone score cannot reuse the
  LMM scalar / 2x2 KKT certificate directly, the joint optimizer must supply a
  GLMM analogue that emits the *same* classification enum and the same
  `FitStatus` leaf, with its objective and approximation scope stated
  explicitly (per integration rule 5 in
  `docs/difficult_model_certification.md`).
- It must not introduce a second "GLMM converged" status, a parallel
  certificate object, or downstream message wording that cannot be derived
  from the existing artifact.

### 4. Fallback policy

The optimizer must define and record a deterministic fallback policy:

- When the joint optimizer fails to certify (non-stationary stop, PIRLS
  failure inside the joint step, indefinite expected information, budget
  exhaustion), it must fall back to the existing fast-PIRLS profiled path and
  **label the result as the fallback path**, not as a certified joint fit. The
  artifact must record which path produced the returned estimates.
- A fallback result keeps the `documented_divergence` class and the existing
  non-`lme4` wording. It must never be silently promoted to parity.
- The joint path is labelled and row-scoped until it passes the external-engine
  parity gates (`comparison/parity_scorecard.toml` plus the
  divergence/scoreboard tests changed in lockstep). `fast = false` may use
  NLopt BOBYQA or the native COBYLA dependency-light path, per
  `docs/glmm_support_contract.md`.
- Promotion of any GLMM row from `documented_divergence` to
  `release_blocking_parity` requires: the `included` objective convention on
  both sides, a recorded stationarity certificate, covariance classification
  through the shared vocabulary, and the scorecard row plus its contract test
  changed together in the same change.

## Acceptance gate summary

A GLMM row is certifiable joint parity only when **all** hold:

1. Objective is the joint Laplace/AGQ deviance with `response_constants =
   included` on both engines.
2. A stationarity / derivative certificate with a recorded residual and
   tolerance is present in the artifact.
3. Covariance state is classified through the existing certificate and
   `FitStatus` vocabulary.
4. The fallback path is recorded and never mislabels a fallback as a certified
   joint fit.
5. `comparison/parity_scorecard.toml` and the corresponding parity/divergence
   contract test are updated together to move the row out of
   `documented_divergence`.

Rows that have not passed this gate keep the honest claim in
`docs/difficult_model_certification.md`: a certified fit *or* a precise
diagnostic, not blanket `lme4` GLMM parity.
