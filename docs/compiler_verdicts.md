# Compiler Verdicts

This page defines the user-facing fit verdict vocabulary used by compact
print and audit reports.

## Optimizer Stop

The optimizer stop is authoritative for convergence. If the optimizer reports
an acceptable stop, the compact verdict must not reclassify the model as
non-converged because of a later finite-difference inspection. If the optimizer
reports budget exhaustion or an unacceptable return code, the verdict may report
an optimizer failure and should name the code.

## Derivative Inspection

Finite-difference gradient and Hessian evidence is inspection metadata. It can
explain why inference should be treated carefully, but it does not override an
acceptable optimizer stop.

When present, derivative evidence records:

- `test_name`: the check, such as `free_gradient_kkt`.
- `observed`: the measured value, such as max absolute free-gradient.
- `threshold`: the tolerance used by the check when available.
- `regime`: the theta regime in which the check was interpreted.

The finite-difference tolerances are scale-sensitive. Large response or
predictor scales can make absolute gradient checks look worse than the fitted
model warrants. The first next action for this case is to rescale predictors or
the response, then refit or verify optimizer agreement.

The crate exposes the underlying evidence rather than a plug-in convergence
predicate. Downstream layers may choose how to display the evidence, but they
should not replace the verdict by injecting an arbitrary `(gradient, Hessian,
nobs) -> bool` rule.

## Boundary And Singular Fits

Boundary and singular fits are not convergence failures. A theta value on a
variance-component boundary changes the statistical statement: the model is a
boundary or reduced-rank fit, and KKT/gradient checks for an interior optimum
are skipped. The compact surface should say boundary, singular, or reduced-rank,
not non-converged.

For supported LMM covariance blocks, the stronger check is the covariance-cone
KKT certificate rather than an interior theta-gradient rule. Scalar
`(1 | group)` blocks and full 2x2 `(1 + x | group)` blocks can be inspected in
covariance space. The certificate classifies the block as an interior optimum,
a valid zero-variance boundary, a valid rank-deficient covariance, an invalid
boundary stop, or weak identification.

Those classifications feed the existing optimizer certificate and verdict
surfaces. They are not a separate convergence system. A valid zero-variance or
rank-deficient covariance result may support `ConvergedBoundary` or
`ConvergedReducedRank`; an invalid boundary stop should remain an optimizer
failure or recovery event; weak identification should stay visible as cautionary
evidence.

When the solver repairs an invalid boundary stop with a covariance
KKT-guided restart, the optimizer return code is prefixed with
`KKT_BOUNDARY_RESTART(...)` and the verdict headline includes recovered
convergence. This is an acceptable recovered stop when the inner restart
returns an acceptable optimizer code, but it is intentionally not presented as
first-pass clean convergence.

The usual next action is to inspect effective covariance and decide whether the
boundary direction is scientifically central. A simpler random-effect structure,
diagonal covariance, or design-compiled policy can be appropriate when the
direction is unsupported.

The implementation note
[`difficult_model_certification.md`](difficult_model_certification.md) maps the
covariance KKT certificate, TrustBQ certified stop, KKT-guided restart, and
parity scorecard to the existing certificate vocabulary.

## Large Theta Fits

Finite-difference Hessian checks are skipped when the number of free theta
parameters exceeds `convergence_derivative_nparmax` (default 10). Large-theta
models use optimizer-regime evidence and optional verification runs instead of
post-hoc Hessian certification.

## Verification

For LMMs, `verify_convergence()` compares bounded restarts and alternate
optimizer runs. GLMM reports do not recommend that API because GLMM bounded
verification is not exposed in compiler v0; compare tighter GLMM refits or
alternate optimizer fits when optimizer agreement matters. Agreement is
reassuring metadata. Disagreement is reported as fragile or unstable
verification and should be interpreted by comparing objective values, theta,
beta, and effective covariance rank across runs.

## Structural Fit Status

Structural findings are orthogonal to optimizer convergence. Rank-deficient
fixed effects, separation, unsupported random slopes, row-saturated random
effects, and missing repeated-unit dependence paths are design or model
identifiability statements. Optimizer settings cannot fix them, so compact
print keeps them separate from optimizer convergence.

## Not Assessed

Unfitted artifacts report `not assessed`. Fit the model before reading optimizer
or derivative evidence.
