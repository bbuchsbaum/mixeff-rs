# Satterthwaite Scalar Contrast Contract

Status: specified
Owner: Rust fixed-effect inference
Mote issue: `bd-01KQB2DEERRCGJ66ZK8CX856BS`
Parent issue: `bd-01KQATBFPGJ956QAVT26EPJZG3`
Primary local reference: `vendor/lmerTestR`

## Purpose

This note pins the Rust contract for scalar Satterthwaite fixed-effect
contrast tests before numeric p-values are enabled. The implementation target
is the `lmerTestR` approach, translated into Rust-owned model state and
row-level inference payloads.

The initial scope is Gaussian LMM scalar fixed-effect contrasts. Multi-df
Satterthwaite F tests, Kenward-Roger, GLMMs, and bootstrap calibration remain
separate lanes.

## Reference Points

The vendored `lmerTestR` implementation provides the reference shape:

- `vendor/lmerTestR/R/lmer.R`: `as_lmerModLT()` stores `vcov_beta`,
  `vcov_varpar`, and `Jac_list`.
- `vendor/lmerTestR/R/lmer.R`: `devfun_vp()` evaluates the ML or REML
  deviance over `varpar = c(theta, sigma)`.
- `vendor/lmerTestR/R/lmer.R`: `get_covbeta()` evaluates
  `vcov_beta(varpar)` from the current model decomposition.
- `vendor/lmerTestR/R/contest.R`: `contest1D()` computes the scalar
  Satterthwaite denominator df and Student-t p-value.
- `vendor/lmerTestR/pkg_notes/Satterthwaite_for_LMMs.md`: derivation and
  worked example.

Rust should match the mathematics and row-level behavior, not mirror the R
object model.

## Parameterization

Satterthwaite computations use a variance-parameter vector:

```text
varpar = [theta_0, ..., theta_{m-1}, sigma]
```

where `theta` are the fitted relative covariance parameters in optimizer order
and `sigma` is the residual standard deviation. This deliberately differs from
the profiled optimizer objective, which is primarily a function of `theta`.

The fitted point is:

```text
varpar_hat = concat(model.theta(), model.sigma())
```

All derivative and covariance artifacts must declare the parameterization and
ordering they use. The order must match the gradient vector used in the final
df calculation.

## Required Fitted-Side Artifacts

Before `method = satterthwaite` can produce a numeric p-value, the fitted LMM
must expose all of the following with certificates or explicit failure
reasons:

| Artifact | Meaning |
|---|---|
| `vcov_beta` | Fixed-effect covariance at `varpar_hat`, matching the existing fitted `vcov()` on the active coefficient basis. |
| `vcov_varpar` | Asymptotic covariance of `varpar_hat`, estimated as `2 * H^+` where `H` is the Hessian of the ML/REML deviance over `varpar`. |
| `jac_vcov_beta` | List of `k` matrices, each `p x p`, giving `d vcov_beta / d varpar_i` at `varpar_hat`. |
| derivative diagnostics | Step sizes, finite/nonfinite evaluation counts, symmetry checks, and boundary-distance diagnostics. |
| reliability diagnostics | Reasoned grade for low df, boundary-active variance parameters, unstable derivatives, singular Hessian, or rank-deficient fixed effects. |

`k = n_theta + 1` and `p = fixed-effect coefficient count on the active Rust
coefficient surface.`

## Deviance Over `varpar`

Rust needs a side-effect-safe evaluator:

```text
deviance_varpar(varpar, reml) -> Result<f64, reason>
```

The evaluator must:

- validate `varpar.len() == n_theta + 1`
- reject nonfinite values
- reject `sigma <= 0`
- respect lower bounds for `theta`
- update the model decomposition at trial `theta`
- evaluate the ML deviance with the supplied `sigma`
- add the REML fixed-effect determinant adjustment when `reml = true`
- restore the original fitted state after each trial evaluation

The REML and ML formulas must be tested against the existing fitted objective
at `varpar_hat` before derivatives are trusted.

Status: implemented as `LinearMixedModel::deviance_varpar(varpar, reml)` and
covered for scalar ML, vector REML, invalid input rejection, and fitted-state
restoration in `src/model/linear.rs`.

## `vcov_beta(varpar)`

Rust needs a side-effect-safe evaluator:

```text
vcov_beta_at_varpar(varpar) -> Result<DMatrix<f64>, reason>
```

At a trial point, this evaluates the fixed-effect covariance implied by the
trial `theta` decomposition and trial `sigma`. In `lmerTestR` notation this is
the analogue of:

```text
sigma^2 * RXi * RXi'
```

where `RXi` is the inverse triangular factor for the fixed-effect block at the
trial covariance parameters. The Rust implementation can use its existing
`vcov()` machinery if it can evaluate at trial `theta` and override `sigma`
without leaving persistent model state behind.

The result must be symmetric within tolerance and have dimensions `p x p`. A
nonfinite or dimension-mismatched result makes Satterthwaite unavailable for
that fit.

Status: implemented as `LinearMixedModel::vcov_beta_varpar(varpar)` and covered
against the fitted `vcov()` at `varpar_hat` with fitted-state restoration.

## Jacobian

The Jacobian is:

```text
J_i = d vcov_beta(varpar) / d varpar_i
```

for each component of `varpar`. The initial implementation may use numerical
finite differences. It must record enough diagnostics to decide whether the
row is available:

- step size per parameter
- central versus one-sided step choice
- lower-bound clearance for theta parameters
- number of failed trial evaluations
- max asymmetry of each `J_i`
- finite-difference sensitivity if multiple step scales are used

Boundary-active parameters are not automatic hard failures for the whole fit,
but a row must not receive a numeric Satterthwaite p-value if its denominator
df depends on an unstable or unavailable derivative direction.

Status: implemented as `LinearMixedModel::jac_vcov_beta_varpar(varpar)` using
central finite differences with explicit lower-bound stencil rejection. It is
covered for symmetry, fitted-state restoration, and the analytic sigma
derivative `d vcov_beta / d sigma = 2 * vcov_beta / sigma`.

## Covariance of `varpar`

Let `H` be the Hessian of `deviance_varpar(varpar, reml)` at `varpar_hat`.
The reference implementation eigendecomposes `H`, keeps eigenvalues above a
tolerance, and computes:

```text
vcov_varpar = 2 * H^+
```

where `H^+` is the Moore-Penrose inverse over positive eigen-directions.

Rust must expose:

- Hessian method, step policy, and tolerance
- eigenvalues of `H`
- count of positive, near-zero, and negative eigenvalues
- whether the inverse used all directions or a reduced active subspace
- reason when `vcov_varpar` is unavailable or low reliability

Negative or near-zero eigenvalues should normally make reliability `low` or
`not_available`, depending on whether the requested contrast uses those
directions.

Status: implemented as `LinearMixedModel::vcov_varpar(varpar, reml)`, returning
the covariance estimate plus Hessian, eigenvalue counts, tolerance, reduced-rank
flag, reliability grade, and notes. Boundary-active central-difference stencils
return explicit errors instead of silent covariance estimates.

## Scalar Contrast Formula

For a scalar hypothesis `L beta = rhs`, with `L` a row vector of length `p`:

```text
estimate = L beta - rhs
var_con = L vcov_beta L'
se = sqrt(var_con)
t = estimate / se
g_i = L J_i L'
satt_denom = g' vcov_varpar g
denominator_df = 2 * var_con^2 / satt_denom
p_value = 2 * P(T_df >= abs(t))
```

All scalar Satterthwaite rows use:

```text
method = satterthwaite
statistic_name = t
numerator_df = null
denominator_df = denominator_df
```

The row is available only if:

- the contrast is estimable
- `var_con` is finite and strictly positive
- all required `J_i` values are finite
- `vcov_varpar` is available on the active parameterization
- `satt_denom` is finite and strictly positive
- `denominator_df` is finite and positive
- the row-level reliability diagnostic does not mark the result unavailable

If any condition fails, the row keeps `method = satterthwaite` for explicit
Satterthwaite requests, sets `status = not_assessed` or
`p_value_unavailable`, leaves `p_value = null`, and gives a Rust-owned reason.

Status: implemented for explicit scalar LMM fixed-effect contrast requests via
`test_contrast_with_method(..., FixedEffectTestMethod::Satterthwaite)`.
After `lmerTestR` parity certification, the public `auto` policy now tries
Satterthwaite first for eligible scalar Gaussian LMM fixed-effect rows and
falls back to labeled Wald rows with an explanatory note when Satterthwaite
prerequisites fail.

## Auto Method Policy

Current certified behavior is:

```text
auto -> satterthwaite -> asymptotic_wald_z -> not_computed
```

This switch happened only after parity fixtures and row-level reliability
diagnostics were in place.

Explicit `method = satterthwaite` must not silently degrade to Wald. If
Satterthwaite prerequisites fail, it returns a missing p-value with the
Satterthwaite method label and a reason.

## Failure Reasons

The implementation must distinguish at least:

- `satterthwaite_varpar_deviance_unavailable`
- `satterthwaite_vcov_beta_derivative_unavailable`
- `satterthwaite_varpar_covariance_unavailable`
- `satterthwaite_boundary_derivative_unstable`
- `satterthwaite_nonpositive_contrast_variance`
- `satterthwaite_nonpositive_df_denominator`
- `satterthwaite_nonfinite_df`
- `satterthwaite_rank_deficient_contrast`
- `satterthwaite_unvalidated_against_reference`

The user-facing reason may be prose, but tests should assert stable diagnostic
codes or stable reason fragments.

## Reliability Grades

Initial grading:

- `moderate`: all prerequisites pass; Hessian has no negative or near-zero
  active eigenvalues; finite-difference diagnostics are stable; df is not
  pathologically small.
- `low`: numeric result is available but one or more caution conditions apply,
  such as low denominator df, boundary proximity not used strongly by the row,
  or mild derivative sensitivity.
- `not_available`: any prerequisite needed for the row fails.

No `high` Satterthwaite grade is required for the first implementation.

## Parity Fixtures

Before enabling numeric rows, add fixtures that compare Rust output against
`lmerTestR` for:

- `sleepstudy`: random intercept and random intercept/slope variants
- `penicillin` or another crossed random-intercept LMM where supported
- an unbalanced scalar-contrast fixture based on the `ham` example in the
  vendored Satterthwaite note, if the dataset is practical to port
- a rank-deficient fixed-effect case, expected to suppress p-values
- a boundary or singular case, expected to produce low reliability or a
  structured unavailable reason depending on derivative stability

Snapshot rows must include estimate, standard error, t statistic,
denominator df, p-value, method, status, reliability, and reason/notes.

Status: the first pinned lmerTest parity values live in
`tests/fixtures/compiler_contract/satterthwaite_lmer_test_parity_v1.json`.
They cover sleepstudy random-intercept and random-slope scalar Days contrasts,
crossed Penicillin intercept, a deterministic unbalanced sleepstudy variant,
boundary unavailable reasons, and rank-deficient not-estimable reasons.

## Implementation Motes

The Satterthwaite umbrella is intentionally split:

| Mote | Role |
|---|---|
| `bd-01KQB2DEF65WZDK2MQRJ55AFJ2` | Done: implement `deviance_varpar(varpar, reml)`. |
| `bd-01KQB2DEJ7Y9W2NT15R3VZD2KF` | Done: implement `vcov_beta(varpar)` and `jac_vcov_beta`. |
| `bd-01KQB2DWF3X8PAQMEP3J58ZPAD` | Done: estimate `vcov_varpar` from the deviance Hessian. |
| `bd-01KQB2DWEDTD9E00KVKWQ3W2MA` | Done: wire scalar Satterthwaite contrast rows. |
| `bd-01KQB2DWFK0JKH890BA2T1ZEB0` | Done: add `lmerTestR` parity fixtures. |

`bd-01KQATBFPGJ956QAVT26EPJZG3` should close only after those motes establish
numeric row availability and validation. Status: closed after the auto-method
switch and wire-fixture update.
