# Summary-Estimate (Meta-Analysis) LMM Front Door

Status: specified
Owner: Rust model layer
Epic mote issue: `bd-01KR7B936N05GQ2TYAPKPSR4JT`
This issue: `bd-01KR7B9NMEDZ527W1NXFEG8TT1`
Related: [`mixed_model_compiler_inference_contract.md`](mixed_model_compiler_inference_contract.md), [`multivariate_shared_theta.md`](multivariate_shared_theta.md)

## Purpose

This contract pins the semantics for a new entry point that fits a
`LinearMixedModel` from first-stage point estimates and their *absolute*
sampling variances, without requiring subject-level data. The fitted model is
mathematically a weighted LMM with the residual scale `sigma` fixed at `1`,
which makes the random-effect variance components absolute (matching the
`tau^2` of `metafor::rma.mv`).

`lme4::lmer` accepts `weights = 1 / V_i` but treats them as *prior weights*
under an estimated residual `sigma`, giving a residual covariance of
`sigma^2 * diag(1 / w_i)`. That is the wrong model class for meta-analysis
unless `sigma` is fixed, which `lmer()` does not expose as a clean fit
option. This crate already has the principled hook
(`OptSummary::sigma: Option<f64>`); the work is exposing it cleanly.

## Scope

In scope:

- A new constructor `LinearMixedModel::from_summary_estimates(...)` that
  takes a column of point estimates `beta_hat_i` and a column of absolute
  sampling variances `V_i`, and produces a fittable `LinearMixedModel`.
- A `SamplingVarianceScale` enum that lets the caller declare whether `V_i`
  is absolute or relative to a known first-stage `sigma`. Inputs are always
  normalized to absolute internally; the fitted model class always carries
  `sigma = 1`.
- A `ResidualSource` marker on `LinearMixedModel` and `VarCorr` that
  distinguishes estimated-`sigma` fits from fixed-sampling-variance fits.
- Reporting tweaks so that `varcorr()` does not display a misleading
  "Residual" row for this fit class.
- Explicit refusal gates for finite-sample inference paths
  (Satterthwaite; Kenward-Roger inherits the existing weighted-model gate).
- A cross-check fixture against `metafor::rma.mv`.

Out of scope (separate epics):

- Multivariate meta-analysis with block sampling covariances `V_i`. This
  lines up with [`multivariate_shared_theta.md`](multivariate_shared_theta.md).
- Bayesian or random-effects-only meta-analysis flavors.
- Auto-detection of "looks like meta-analysis" data on the standard
  `LinearMixedModel::new` path. The new constructor is the only entry point.
- GLMM analogues (logit / Poisson meta-analysis). The fixed-`sigma` hook is
  not currently wired through `GeneralizedLinearMixedModel`; that is its own
  contract.

## Reference Points

Existing code that this contract leans on:

- `src/types/opt_summary.rs:188` — `OptSummary::sigma: Option<f64>`. `None`
  profiles `sigma` out (the usual case); `Some(sigma)` fixes it.
- `src/model/linear.rs:2013` — `objective_from_components` branches on
  `fixed_sigma`. With `Some(1.0)`, the objective becomes
  `logdet + denomdf * (2 * ln(1) + ln(2 pi)) + pwrss / 1`.
- `src/model/linear.rs:3019` — `profiled_objective_from_parts` threads
  `fixed_sigma` through the optimizer hot path. The summary-estimate
  constructor does not need any new optimizer machinery.
- `src/model/linear.rs:7295` — `dof()` already adds the trailing `+1` only
  when `optsum.sigma.is_none()`. Summary-estimate fits therefore correctly
  count `feterm.rank + n_theta` parameters.
- `src/model/linear.rs:4619` — `varest()` returns `sigma` for fixed-`sigma`
  fits (Julia parity). For this fit class that is `1.0`.
- `src/model/linear.rs:2336` — Kenward-Roger is already unavailable for
  weighted models. Summary-estimate fits inherit this refusal.
- `src/model/linear.rs:783-790` — `LinearMixedModel::new` already supports
  `weights`, including `sqrtwts` and `xy_mat.reweight`.

## Statistical Model

For studies `i = 1, ..., n`:

```text
beta_hat_i = X_i beta + Z_i u + e_i
e_i ~ N(0, V_i)            # V_i known, absolute
u ~ N(0, Lambda_theta Lambda_theta')   # absolute, sigma fixed at 1
```

With `weights = 1 / V_i` and `sigma = 1`, the implied residual variance is

```text
Var(e_i) = sigma^2 / w_i = 1 / (1 / V_i) = V_i
```

so the user's declared sampling variances are honored exactly. Random-effect
covariance components are absolute:

```text
Var(u) = sigma^2 * Lambda_theta Lambda_theta' = Lambda_theta Lambda_theta'
```

For a scalar random intercept on a grouping factor, the fitted random-effect
SD is directly `tau` (between-study heterogeneity), and the fitted variance
is `tau^2`. This matches the `metafor::rma.mv` convention.

## Variance Scale Conversion

Inputs:

```rust
pub enum SamplingVarianceScale {
    Absolute,
    Relative { sigma: f64 },
}
```

`Absolute` means `V_i` is already on the correct scale (the typical case for
published meta-analysis inputs and for two-stage fits where the first-stage
residual `sigma_hat` is already folded into the reported variances).

`Relative { sigma }` means the caller has unscaled variances `V_i_rel` (for
example, the diagonal of `(X' X)^{-1}` from a first-stage OLS) and a separate
first-stage residual scale `sigma`. The constructor normalizes to absolute:

```text
absolute_V_i = match scale {
    Absolute            => V_i,
    Relative { sigma }  => sigma * sigma * V_i,
};
weights = 1 / absolute_V_i
optsum.sigma = Some(1.0)
```

The fitted model class **always** carries `sigma = 1`. `Relative` is only an
input conversion; it never appears in the fitted state. This keeps the model
class semantics simple: one scale, one residual source.

Rejection rules:

- `V_i <= 0` for any `i`: error referencing the column name and row index.
- `V_i` non-finite for any `i`: error referencing the column name and row
  index.
- `beta_hat_i` non-finite for any `i`: error referencing the column name and
  row index.
- `Relative { sigma }` with `sigma <= 0` or non-finite: error.

## Public API

```rust
impl LinearMixedModel {
    pub fn from_summary_estimates(
        formula: Formula,
        data: &DataFrame,
        estimate_column: &str,
        sampling_variance_column: &str,
        options: SummaryEstimateOptions,
    ) -> Result<LinearMixedModel>;
}

pub struct SummaryEstimateOptions {
    pub variance_scale: SamplingVarianceScale,  // default Absolute
    pub reml: bool,                             // default true
    pub policy: CompilerPolicy,                 // default CompilerPolicy::default()
}

pub enum SamplingVarianceScale {
    Absolute,
    Relative { sigma: f64 },
}
```

Constructor responsibilities:

1. Resolve `estimate_column` and `sampling_variance_column` from `data`.
   Both must be numeric and equal in length to the response referenced by
   `formula`.
2. Validate `V_i` and `beta_hat_i` per the rejection rules above. Validation
   is the constructor's job, not the type's.
3. Normalize `V_i` to absolute via `SamplingVarianceScale`. Compute
   `weights = 1 / absolute_V_i`.
4. Build a synthetic `DataFrame` (or borrow appropriately) such that the LHS
   of `formula` references `estimate_column`.
5. Call `LinearMixedModel::new_with_policy(formula, &df, Some(&weights),
   policy)`.
6. Set `optsum.sigma = Some(1.0)` and
   `residual_source = ResidualSource::FixedSamplingVariance`. This must
   happen *before* fit; setting it after a fit is a contract violation.
7. Return the unfitted model. Callers invoke `.fit(reml)` as usual.

The constructor itself does **not** fit. This matches the existing
`LinearMixedModel::new` pattern.

## Residual Source Marker

```rust
pub enum ResidualSource {
    EstimatedSigma,           // default, set by LinearMixedModel::new
    FixedSamplingVariance,    // set by from_summary_estimates only
}
```

Carried on `LinearMixedModel` and propagated to `VarCorr`. The existing
fixed-`sigma` test paths (`linear.rs:16245` etc.) use `optsum.sigma = Some(...)`
without this marker; those are *not* summary-estimate fits and continue to
report a residual row labeled with the fixed scale. Only fits constructed
through `from_summary_estimates` set
`residual_source = FixedSamplingVariance`.

## Reporting

| Surface | Default-`EstimatedSigma` behavior | `FixedSamplingVariance` behavior |
|---|---|---|
| `varcorr()` | residual row labeled "Residual" with `self.sigma()` | **omit residual row**; `VarCorr.residual_source = FixedSamplingVariance` |
| `coeftable()` | unchanged | unchanged (Wald SEs use `vcov()` which already respects fixed `sigma`) |
| `varest()` | `sigma()^2` | `1.0` (Julia parity preserved); document the meaning |
| `dof()` | `rank + n_theta + 1` | `rank + n_theta` (already correct via `optsum.sigma.is_some()`) |
| `aic` / `bic` | unchanged math, picks up correct `dof()` | unchanged math, picks up correct `dof()` |
| Satterthwaite | available subject to existing gates | **`InferenceStatus::NotAssessed`** with reason "summary-estimate fit (residual sampling variances fixed); finite-sample methods are undefined when sigma is not estimated" |
| Kenward-Roger | available subject to existing gates | **`MixedModelError::InvalidArgument`** inherited from the weighted-model gate at `kenward_roger_sigma_g`, which rejects all weighted models |
| Wald | available | available |
| Parametric bootstrap | available | available |

Rationale for omitting the residual row in `varcorr()`: a "Residual 1.0"
line invites readers to interpret it as an estimated quantity. For
summary-estimate fits the residual scale is *fixed by user input*, not
estimated, so it does not belong in the variance-components table. The
`residual_source` marker on `VarCorr` lets downstream renderers display
"(fixed sampling scale)" when they choose to.

## Refusal Wording

When Satterthwaite is requested on a summary-estimate fit, the returned
`FixedEffectTest` carries:

```text
InferenceStatus::NotAssessed {
    reason: "summary-estimate fit (residual sampling variances fixed); \
             finite-sample methods are undefined when sigma is not estimated"
        .to_string()
}
```

Implemented at the entry point of `satterthwaite_fixed_effect_test` so the
gate fires before any variance-parameter Hessian computation. The status is
`NotAssessed` (not `Unsupported`) because the existing scalar-contrast
refusal in the same function uses `NotAssessed`; consistency wins over
semantic precision here.

Kenward-Roger inherits the existing weighted-model refusal at
`kenward_roger_sigma_g`:

```text
Err(MixedModelError::InvalidArgument(
    "Kenward-Roger Sigma/G decomposition is currently certified only for \
     unweighted iid Gaussian residual models".to_string()
))
```

Tests must assert the *refusal*, not the literal string — both messages
may be rewritten without changing the contract.

## Acceptance Criteria

- New constructor compiles, validates inputs, and produces a fittable model.
- `cargo test summary_estimate_parity` recovers `metafor::rma.mv` `beta_hat`
  to within `1e-6` and `tau^2` to within `1e-4` on the BCG vaccine fixture
  (or comparable fixed-seed simulation if licensing of `dat.bcg` is
  unclear).
- `varcorr()` on a summary-estimate fit does not contain a "Residual" row.
- Satterthwaite on a summary-estimate fit returns `Unavailable` with the
  documented reason.
- Kenward-Roger on a summary-estimate fit returns `Unavailable` (inherited).
- `cargo test` and `cargo clippy --all-targets` are clean.
- Existing fixtures under `tests/fixtures/compiler_contract/` are
  byte-identical: this contract introduces a new fit class, it does not
  perturb existing ones.

## Worked Example (sketch)

```rust
use mixedmodels::model::{
    LinearMixedModel, SummaryEstimateOptions, SamplingVarianceScale,
};
use mixedmodels::formula::Formula;

let formula = Formula::parse("logrr ~ 1 + (1 | study)")?;
let opts = SummaryEstimateOptions {
    variance_scale: SamplingVarianceScale::Absolute,
    reml: true,
    ..Default::default()
};

let mut model = LinearMixedModel::from_summary_estimates(
    formula,
    &bcg_data,
    "logrr",     // beta_hat per study
    "var_logrr", // V_i (absolute)
    opts,
)?;
model.fit(true)?;

println!("{}", model.coeftable());
println!("{}", model.varcorr()); // <-- no "Residual" row
```

## Notes on Two-Stage Workflows

A common upstream pattern is: fit a per-subject OLS or GLM, harvest
`beta_hat_i` and `vcov(beta_hat_i)`, then run a between-subject mixed model
on the per-subject estimates. This contract supports the diagonal slice of
that workflow (one scalar effect per subject). The full multivariate
generalization (per-subject coefficient *vectors* with full sampling
covariance) is the multivariate-meta-analysis follow-on and is explicitly
out of scope here.
