# Difficult Model Certification

Status: internal consolidation plan; not a new public API.

This note records how the difficult-model program should use the certificate
and verdict machinery that already exists in the crate. The goal is to avoid a
second convergence vocabulary while still making hard boundary, singular, and
weak-identification fits easier to certify.

## Existing Surfaces

The source of truth is the current compiler and model certificate stack:

| Layer | Existing surface | Role |
| --- | --- | --- |
| Fit state | `compiler::diagnostics::FitStatus` | Top-level status: interior, boundary, reduced-rank, penalised, not identifiable, not optimized, or not assessed. |
| Optimizer certificate | `compiler::audit::OptimizerCertificate` | Stable evidence bundle for optimizer stop, objective, parameter-space state, derivative evidence, checks, diagnostics, and optional verification. |
| Compact message | `compiler::report::ConvergenceVerdict` | User-facing projection over the optimizer certificate plus structural diagnostics. |
| Covariance KKT checks | `LinearMixedModel::{scalar,two_by_two}_covariance_kkt_certificate` | LMM-only covariance-cone diagnostics for scalar and 2x2 random-effect blocks. |
| TrustBQ stop hook | `TrustBqStopReason::CertifiedConvergence` | Optimizer-level early stop when the model-owned certificate accepts the current point. |
| Recovery | KKT-guided boundary restart in `LinearMixedModel` | Controlled restart from an invalid boundary stop, using the covariance-space descent signal. |
| Release evidence | `comparison/parity_scorecard.toml` plus parity/divergence tests | Rows that are parity, documented divergence, stress opt-in, or diagnostic contracts. |
| Hard-model evidence | `comparison/difficult_model_scoreboard.toml` plus difficult-scoreboard tests | Selected hard rows with pathology axes, certification claims, required metrics, and computed time-to-certified-fit inputs. |

New difficult-model work must extend these surfaces. It must not introduce a
parallel "hard model status", independent certificate object, or downstream
message vocabulary that cannot be derived from the existing artifact.

## Current Coverage

The scalar covariance KKT certificate covers `(1 | group)` blocks. It reports
the fitted theta, variance, directional score, complementarity residual,
tolerances, objective, and a `CovarianceKktClassification`.

The 2x2 covariance KKT certificate covers full random intercept/slope blocks
such as `(1 + x | group)`. It reports the covariance block, reconstructed score
matrix, minimum covariance eigenvalue, minimum score eigenvalue,
complementarity residual, tolerances, objective, and classification.

Both certificates evaluate directional objective differences through the
existing profiled LMM objective. They do not form dense marginal covariance
matrices.

The current classification vocabulary is:

- `InteriorConverged`: covariance block is interior and the covariance-space
  score is near zero.
- `ValidZeroVariance`: scalar variance is zero and the covariance-space score
  supports the boundary.
- `ValidRankDeficientCovariance`: covariance block is singular and the score
  supports the active face.
- `InvalidBoundaryStop`: fitted point is on a boundary but the covariance-space
  score gives a feasible descent direction.
- `WeakIdentification`: the local score is not decisive enough to certify a
  clean interior or valid boundary interpretation.

`WeakIdentification` is the current numeric-uncertainty bucket. Add a separate
`NumericUncertain`-style variant only if an implementation can distinguish
finite-difference noise from genuinely weak statistical information and the
existing status is insufficient.

## Integration Rules

1. A fitted LMM should be called clean only when the optimizer certificate and,
   for supported boundary/singular covariance cases, the covariance KKT
   certificate agree with that interpretation.
2. Boundary and singular covariance fits are not automatic failures. They are
   valid when the covariance-cone KKT certificate supports the boundary or
   lower-rank active face.
3. Invalid boundary stops are optimizer/recovery events. They should flow
   through the existing optimizer certificate, diagnostic codes, TrustBQ status,
   and `ConvergenceVerdict`, not a separate hard-model report.
4. Weak identification should remain visible. It can be acceptable evidence for
   a cautious result, but it is not the same as a certified boundary optimum.
5. GLMM difficult-model rows must not reuse LMM covariance KKT wording unless
   their objective, approximation, and certificate scope are explicitly stated.
6. The parity scorecard is the release gate. A pathology row cannot silently
   become release parity just because it fits; the scorecard and corresponding
   test must change together.

## Gap Audit

| Need | Current state | Next action |
| --- | --- | --- |
| Scalar KKT certificate | Implemented and tested for interior, zero boundary, invalid boundary, and weak identification. | Keep as canonical scalar LMM path. |
| 2x2 KKT certificate | Implemented and tested for valid rank-one boundary and invalid all-zero boundary. | Add more corpus rows before generalizing beyond 2x2. |
| Certificate-aware TrustBQ stop | Implemented via the TrustBQ progress callback and `CertifiedConvergence`. | Keep using the model-owned certificate callback. |
| KKT-guided recovery | Implemented for scalar and 2x2 invalid boundary starts. Recovered fits carry a `KKT_BOUNDARY_RESTART(...)` return code, `OptimizerRecovery` diagnostic, and recovered-convergence verdict wording. | Extend only where a scoreboard row demonstrates value. |
| Active-face POC | Present as LMM prototype tests and `mmtrust_psd_lmm_prototype.md`. | Keep experimental until benchmark evidence supports a default path. |
| Difficult-model corpus | Partly represented by comparison fixtures and pathology tests. | Expand through the corpus/scoreboard mote, not by inventing new statuses. |
| Time-to-certified-fit scoreboard | Implemented as a comparison-backed manifest for selected difficult rows and unit-test recovery cases. | Extend the manifest when adding new pathology classes or comparator workflows. |
| Public docs | Compiler verdict docs explain how covariance KKT certificates, KKT-guided recovery, and recovered-convergence wording feed the existing verdict surface. | Keep public wording aligned with the existing certificate vocabulary as hard-model support expands. |
| Certified joint GLMM optimizer | Labelled joint Laplace and joint AGQ are available for `fast=false` through NLopt BOBYQA when enabled and native COBYLA in dependency-light builds, with labelled fast-PIRLS fallback when certification fails. Certification is row-scoped: `culcitalogreg` Laplace and AGQ are promoted; other difficult binomial rows stay non-parity until they pass objective, derivative/stationarity, covariance-certificate, fallback, and lockstep scorecard gates. | Keep extending the joint path through the existing certificate stack; promote rows only when they pass the gate in `docs/certified_joint_glmm_optimizer_contract.md` plus the scorecard/test change. |

## Release Claim

The release-safe claim is:

`mixeff-rs` aims to return either a certified fit or a precise diagnostic on
difficult mixed models.

That is stronger and safer than claiming universal raw speed or universal
superiority over `lme4` and MixedModels.jl. Ordinary parity remains measured by
the comparison harness and `comparison/parity_scorecard.toml`. Difficult-model
progress is measured by the focused difficult-model scoreboard:
time-to-certified-fit, certificate quality, diagnostic precision, and whether
manual optimizer switching is avoided.

The public wording and release-label contract for that claim lives in
[`difficult_model_release_contract.md`](difficult_model_release_contract.md).
