# Kenward-Roger Fixed-Effect Inference Contract

Status: specified; implementation blocked on method-specific artifacts
Owner: Rust fixed-effect inference
Parent issue: `bd-01KQATBN53CKDTMA5VN5A1YMW3`
Primary local reference: `vendor/lmerTestR`
Primary external reference: `pbkrtest`

## Purpose

This note pins the Rust contract for Kenward-Roger fixed-effect tests before
numeric `method = kenward_roger` p-values are enabled. KR support is in scope,
but only after Rust can produce and certify the adjusted fixed-effect
covariance, denominator degrees of freedom, F scaling, and reference parity
fixtures needed to make the method label true.

The initial scope is Gaussian LMMs fitted by REML with iid Gaussian residuals
and the covariance structures already represented by the Rust LMM engine.
GLMMs, residual covariance structures, random-effect p-values, and R-side
reconstruction are out of scope.

## Reference Points

`lmerTestR` delegates KR to `pbkrtest`:

- `vendor/lmerTestR/R/contest.R`: scalar `contest1D(..., ddf =
  "Kenward-Roger")` calls `pbkrtest::vcovAdj(model)` and
  `pbkrtest::Lb_ddf(L, V0 = vcov(model), Vadj = vcov_beta_adj)`.
- `vendor/lmerTestR/R/contest.R`: multi-df `contestMD(..., ddf =
  "Kenward-Roger")` calls `pbkrtest::KRmodcomp(model, L, betaH = ...)`.
- `vendor/lmerTestR/tests/test_contest1D.R` and
  `vendor/lmerTestR/tests/test_contestMD.R` gate KR parity on `pbkrtest`
  availability and use sleepstudy/cake examples.

`pbkrtest` provides the algorithmic target:

- `vcovAdj()` computes the adjusted fixed-effect covariance `PhiA`.
- `vcovAdj_internal()` builds `Sigma`, `G`, `P`, `Q`, information matrix `IE2`,
  covariance-parameter uncertainty `W`, and `PhiA = Phi + 2 * Gamma`.
- `Lb_ddf(L, V0, Vadj)` computes adjusted denominator df from unadjusted and
  adjusted fixed-effect covariance matrices plus the `P` and `W` attributes
  stored on `Vadj`.
- `KRmodcomp()` produces the multi-df F test, unscaled F statistic, F scaling,
  denominator df, and p-value for restriction-matrix hypotheses.

Rust should match the mathematics and row-level behavior, not mirror the R
object model or rely on live R calls.

## Preconditions

KR rows are available only when all of the following are true:

- the model is a fitted Gaussian LMM
- the fit is REML, or Rust has an explicit certified REML refit/re-evaluation
  path for the KR request
- the requested hypothesis is estimable on the active fixed-effect basis
- the fit intent and post-selection policy permit fixed-effect p-values
- Satterthwaite-level `varpar`, `vcov_beta(varpar)`, `jac_vcov_beta`, and
  `vcov_varpar` diagnostics are available
- KR-specific adjusted covariance and denominator-df artifacts pass reliability
  checks
- parity fixtures against `pbkrtest` pass for the supported model classes

Explicit `method = kenward_roger` requests must not silently degrade to
Satterthwaite or Wald. If any prerequisite fails, Rust returns a
`kenward_roger`-labeled unavailable row with a method-specific reason.

## Required Artifacts

### Sigma/G Decomposition

Rust needs a fitted-side artifact equivalent to `pbkrtest::get_SigmaG()`:

```text
Sigma = covariance matrix of the response y
G_i   = known component matrices whose weighted sum forms Sigma
M     = number of covariance-component matrices
```

The artifact must declare:

- parameterization and ordering of covariance components
- whether the residual variance is included as a component
- dimensions and symmetry checks for `Sigma` and every `G_i`
- positive-definiteness or active-subspace status for `Sigma`
- memory and size limits, since the reference algorithm inverts an `n x n`
  matrix

### Adjusted Fixed-Effect Covariance

Rust needs:

```text
kenward_roger_vcov_beta_adjusted() -> Vadj
```

where `Vadj` contains:

- unadjusted `Phi = vcov_beta`
- adjusted `PhiA`
- `P` matrices
- covariance-parameter uncertainty matrix `W`
- information matrix eigenvalues or condition diagnostics
- notes/reliability grade for generalized inverse use, singular `IE2`,
  boundary-active covariance components, or non-positive adjusted variances

The reference path computes `SigmaInv`, `TT`, `HH`, `OO`, `PP`, `QQ`,
`Ktrace`, `IE2`, `W`, `Gamma`, and `PhiA`. Rust may use equivalent algebra,
but the payload must expose enough diagnostics to explain unavailable or
low-reliability rows.

### Scalar Denominator DF

For scalar `L beta = rhs`, Rust needs:

```text
kenward_roger_lbddf(L, V0, Vadj) -> denominator_df
```

The scalar row uses the adjusted variance:

```text
estimate = L beta - rhs
var_con = L Vadj L'
se = sqrt(var_con)
t = estimate / se
p = 2 * P(T_den_df >= abs(t))
```

Output:

```text
method = kenward_roger
statistic_name = t
numerator_df = null
denominator_df = <KR df>
```

The row is unavailable if `var_con`, `se`, denominator df, or the Student-t
distribution is non-finite or non-positive.

### Multi-DF Hypothesis Rows

For rank `q > 1` hypotheses, Rust needs a `KRmodcomp`-equivalent path:

```text
method = kenward_roger
statistic_name = f
numerator_df = q
denominator_df = <KR df>
statistic = F_scaled
p_value = P(F_q,df >= F_scaled)
```

The payload must also retain method detail for:

- unscaled F statistic
- F scaling
- rank used for `L`
- treatment of non-zero `rhs`
- warning or unavailable reason for rank-deficient `L` with non-zero `rhs`

The default coefficient table remains scalar. Multi-df rows appear only for
explicit term or joint-hypothesis requests.

Implementation note: `bd-01KQDBF2NKE1WQ4DVATG9F808W` adds optional
`FixedEffectInferenceRow.details.contrast_family` and
`details.kenward_roger` metadata. Multi-df KR rows record restriction-row
count, coefficient count, requested/effective rank, rank-deficiency status,
RHS-zero/nonzero status, numerator-df semantics, and the current F scaling
state. The current row statistic remains the unscaled F statistic with
`f_scaling = 1.0`; rows whose pbkrtest reference has non-unit scaling remain
documented as unscaled until scaled-F support is promoted into the row
calculation.

## Reliability and Failure Reasons

KR-specific reasons should distinguish at least:

- `kenward_roger_requires_reml`
- `kenward_roger_sigma_g_unavailable`
- `kenward_roger_sigma_not_positive_definite`
- `kenward_roger_adjusted_vcov_unavailable`
- `kenward_roger_adjusted_vcov_non_positive`
- `kenward_roger_information_singular`
- `kenward_roger_lbddf_unavailable`
- `kenward_roger_nonfinite_df`
- `kenward_roger_f_scaling_unavailable`
- `kenward_roger_unvalidated_against_pbkrtest`

Boundary or reduced-rank covariance states are not automatic hard failures.
They lower reliability or make a row unavailable depending on whether the
adjusted covariance, df, and statistic remain finite and defensible for that
specific hypothesis.

## Auto Policy

KR is opt-in for schema `1.0.0`. The public `auto` ladder remains:

```text
auto -> satterthwaite -> asymptotic_wald_z -> not_computed
```

A later major schema version may choose KR automatically, but only after the
adjusted-covariance and `pbkrtest` parity contracts are stable.

## Work Breakdown

| Issue | Scope |
|---|---|
| `bd-01KQB8C5KJXX2H0D1K5CPQ5R22` | Specify and implement `Sigma/G` and KR component diagnostics. |
| `bd-01KQB8C8TA6DAGNCFC1E3R8NGY` | Implement adjusted fixed-effect covariance with `P`, `W`, condition diagnostics, and reliability notes. |
| `bd-01KQB8CD1HAQPAT21JS7DA0V1P` | Implement scalar and rank-aware `Lb_ddf` denominator-df calculations. |
| `bd-01KQB8CG5TBS4AES1AXGRHNRZA` | Wire explicit KR scalar and multi-df fixed-effect rows without silent fallback. |
| `bd-01KQB8CK9GFQ6VMN49M8B8Y2GW` | Add `pbkrtest` parity fixtures for scalar coefficient rows and multi-df hypothesis rows. |

The parent certification issue `bd-01KQATBN53CKDTMA5VN5A1YMW3` closes once all
five child issues are complete, explicit KR rows are covered by the pbkrtest
fixture, and unavailable paths keep `method = kenward_roger` with Rust-owned
reasons. KR remains opt-in for schema `1.0.0`; default artifact rows continue
to use the documented `auto -> satterthwaite -> asymptotic_wald_z` ladder.

## Implementation Notes

`bd-01KQB8C5KJXX2H0D1K5CPQ5R22` adds
`LinearMixedModel::kenward_roger_sigma_g()`, returning
`KenwardRogerSigmaG`. The current implementation is deliberately narrow:
fitted, unweighted Gaussian LMMs only. It materializes dense response-space
matrices, records the dense-memory requirement, exposes VarCorr covariance
entry weights followed by residual variance, and reports symmetry plus
positive-definiteness diagnostics. Weighted residual models remain unavailable
until a separate KR residual-covariance policy is specified.

`bd-01KQB8C8TA6DAGNCFC1E3R8NGY` adds
`LinearMixedModel::kenward_roger_adjusted_vcov()`, returning
`KenwardRogerAdjustedVcov`. It implements the `pbkrtest::vcovAdj_internal()`
algebra over the active fixed-effect basis: invert `Sigma`, build `P` and `Q`
matrices, form the covariance-parameter information matrix `IE2`, compute
`W = 2 * IE2^-1` or a generalized inverse with an explicit reliability note,
and return `PhiA = Phi + 2 * Phi * U * Phi`. The payload keeps `P`, `Q`, `W`,
`IE2`, eigenvalue diagnostics, and both active-basis and user-order adjusted
covariance matrices for downstream `Lb_ddf` and row-level KR tests.

`bd-01KQB8CD1HAQPAT21JS7DA0V1P` adds
`LinearMixedModel::kenward_roger_lbddf()`, returning `KenwardRogerLbDdf`. It
implements the `pbkrtest::Lb_ddf()` denominator-df formula using active-basis
`V0`, `P`, and `W` from `KenwardRogerAdjustedVcov`. Full user-order contrasts
are mapped through the fixed-effect pivot, redundant restriction rows use the
numerical row rank as numerator df, and singular `L V0 L'` paths use a
generalized inverse with a reliability note instead of silently producing an
unlabeled df.

`bd-01KQB8CG5TBS4AES1AXGRHNRZA` wires explicit
`FixedEffectTestMethod::KenwardRoger` requests into
`LinearMixedModel::test_contrast_with_method()`. Scalar hypotheses return
adjusted-SE t rows with KR denominator df. Multi-df hypotheses return F rows
with numerator df equal to the numerical restriction rank and p-values from the
F distribution. Explicit KR requests do not fall back to Satterthwaite or Wald:
ML fits, unavailable adjusted covariance, and unavailable df return
`method = kenward_roger` with missing p-values and Rust-owned reasons.

`bd-01KQB8CK9GFQ6VMN49M8B8Y2GW` adds the versioned
`tests/fixtures/compiler_contract/kenward_roger_pbkrtest_parity_v1.json`
fixture. It records `lmerTest`/`pbkrtest` reference values for scalar
coefficient rows, scalar RHS contrasts, multi-df restrictions, and row-rank
deficient restriction matrices on supported `sleepstudy` LMMs. The Rust tests
assert scalar KR parity against `contest1D(..., ddf = "Kenward-Roger")` and
assert current multi-df row parity against `pbkrtest::KRmodcomp()`'s unscaled
F statistic/p-value. The same fixture also stores the scaled F statistic,
p-value, and `F.scaling`; rows with `F.scaling != 1` remain explicitly
documented as not yet using pbkrtest's scaled F output in the row payload.
`bd-01KQBDHNVJFZJHSBVB8S15GXEM` fixed the active-basis contrast mapping used
by `Lb_ddf`: full-rank user-order contrasts are now permuted through the
fixed-effect pivot even when all fixed-effect columns are active. This closes
the random-intercept sleepstudy slope df mismatch pinned by the fixture.
