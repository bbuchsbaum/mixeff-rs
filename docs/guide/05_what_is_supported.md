# What is supported

A concrete inventory of what this crate fits, computes, and deliberately
refuses. Status labels follow the project's no-fake-claims contract:

| Label | Meaning |
|---|---|
| **Stable** | Stable surface, parity-tested, covered by SemVer after 1.0. |
| **Stable, labelled** | Stable surface, but the approximation/path is explicit in the fit metadata — not necessarily the same numerics as another engine. |
| **Refused** | Returns a typed error or a typed inference refusal. |
| **Out of scope** | Deferred to 2.0; not present today. |

## Model classes

| Class | Type | Status |
|---|---|---|
| Linear mixed model | [`LinearMixedModel`](crate::model::LinearMixedModel) | Stable |
| Generalized linear mixed model | [`GeneralizedLinearMixedModel`](crate::model::GeneralizedLinearMixedModel) | Stable for the supported family/link matrix below |

## Families and links

The supported family/link matrix is enforced by
[`GeneralizedLinearMixedModel::new`](crate::model::GeneralizedLinearMixedModel)
— anything outside it returns
[`MixedModelError::UnsupportedFamilyLink`](crate::error::MixedModelError),
except `Family::Normal + LinkFunction::Identity`, which returns an explicit
invalid-argument refusal. That case is **refused on purpose**: it is a linear
mixed model, so use
[`LinearMixedModel`](crate::model::LinearMixedModel).

| Family | Allowed links | Default (canonical) |
|---|---|---|
| `Bernoulli` | `Logit`, `Probit`, `Cloglog` | `Logit` |
| `Binomial` | `Logit`, `Probit`, `Cloglog` | `Logit` |
| `Poisson` | `Log`, `Sqrt` | `Log` |
| `Gamma` | `Log`, `Inverse` | `Inverse` |
| `InverseGaussian` | `Log`, `Inverse` | `Inverse` |
| `Normal` (as GLMM) | `Log`, `Inverse`, `Sqrt` | — (use LMM for Identity) |

The variant lists are intentionally enumerable from the public types, so this
table cannot drift silently:

```rust
use mixeff_rs::model::{Family, LinkFunction};
# fn main() {
let _families = [
    Family::Normal, Family::Bernoulli, Family::Binomial,
    Family::Poisson, Family::Gamma, Family::InverseGaussian,
];
let _links = [
    LinkFunction::Identity, LinkFunction::Log, LinkFunction::Logit,
    LinkFunction::Probit, LinkFunction::Cloglog, LinkFunction::Inverse,
    LinkFunction::Sqrt,
];
# }
```

## Formula DSL

[`parse_formula`](crate::formula::parse_formula) accepts an lme4-style
subset:

| Construct | Meaning | Status |
|---|---|---|
| `y ~ x1 + x2` | Additive fixed effects | Stable |
| `y ~ x1 * x2` | Main effects + interaction | Stable |
| `y ~ x1 : x2` | Interaction only | Stable |
| `y ~ x1 / x2` | Nesting (`x1 + x1:x2`) | Stable |
| `0 + …`, `-1 + …`, `1 + …` | Explicit intercept handling | Stable |
| `(re | g)` | Correlated random effects in group `g` | Stable |
| `(re || g)` | Zero-correlation random effects | Stable |
| `(re | g1 & g2)` | Interaction grouping factor | Stable |
| `(re | g1:g2)` | Cell-level grouping factor | Stable |
| `(re | g1/g2)` | Nested grouping expansion | Stable |
| `(re | g1*g2)` | Main grouping factors plus cell expansion | Stable |
| `I(expr)` and other in-formula transforms | Stateless arithmetic subset | Stable (minimal subset) |
| Full `I()` / model.matrix transformations | — | Out of scope |

## Estimation

| Path | Backend | Status |
|---|---|---|
| LMM, profiled (RE)ML via blocked-Cholesky PLS | Auto-dispatched (`PatternSearch`, `TrustBq`, `NloptBobyqa`, `NloptNewuoa`, `Cobyla`, or `PrimaBobyqa` when the `prima` feature is on) | Stable |
| GLMM, profiled fast-PIRLS (`fast=true`, default) | PIRLS with Laplace/AGQ metadata in the fit summary | Stable, labelled |
| GLMM, joint Laplace (`fast=false`, `n_agq ≤ 1`) | NLopt-backed joint optimization | Stable, labelled (requires the `nlopt` feature) |
| GLMM, adaptive Gauss-Hermite (`fast=false`, `n_agq > 1`) | AGQ for valid single-scalar random-effect GLMMs | Stable, labelled (requires the `nlopt` feature) |

The optimizer choice is made by the fit driver, not by callers. The chosen
optimizer and convergence outcome are always recoverable from
[`MixedModelFit::opt_summary`](crate::model::MixedModelFit::opt_summary). The
GLMM `fast=true` default is **not** the same statistical approximation as
`lme4::glmer`'s joint Laplace fit — see
[the GLMM page](crate::guide::glmms) before reporting inference.

## Inference and post-fit summaries

| Surface | LMM | GLMM | Status |
|---|---|---|---|
| Point estimates ([`coef`](crate::model::MixedModelFit::coef), [`vcov`](crate::model::MixedModelFit::vcov), [`stderror`](crate::model::MixedModelFit::stderror)) | ✓ | ✓ | Stable |
| Random effects ([`ranef`](crate::model::MixedModelFit::ranef)), [`fitted`](crate::model::MixedModelFit::fitted), [`loglikelihood`](crate::model::MixedModelFit::loglikelihood), [`aic`](crate::model::MixedModelFit::aic) / [`bic`](crate::model::MixedModelFit::bic) | ✓ | ✓ | Stable |
| Variance components ([`VarCorr`](crate::stats::VarCorr)) | ✓ | ✓ | Stable |
| Model summary ([`ModelSummary`](crate::stats::ModelSummary)) — markdown / HTML / LaTeX | ✓ | ✓ | Stable |
| Wald CIs ([`CoefTable::wald_confint`](crate::stats::CoefTable::wald_confint)) | ✓ | ✓ | Stable |
| Satterthwaite / Kenward-Roger df rows in [`CoefTable`](crate::stats::CoefTable) | ✓ | — | Stable for Gaussian REML LMMs with iid Gaussian residuals; crossed/nested certification is fixture-driven and expanding |
| Profile-likelihood CIs ([`crate::stats::profile`](mod@crate::stats::profile)) — `σ`, `θ`, ML `β` | ✓ | — | Stable for LMM; GLMM out of scope |
| Parametric bootstrap ([`parametricbootstrap`](crate::model::parametricbootstrap), [`parametricbootstrap_glmm`](crate::stats::bootstrap::parametricbootstrap_glmm)) | ✓ | ✓ | Stable for LMM; stable for Bernoulli, Binomial, Poisson, and Gamma GLMMs. InverseGaussian and Normal-as-GLMM bootstrap are refused |
| Likelihood-ratio tests ([`LikelihoodRatioTest`](crate::stats::LikelihoodRatioTest), [`BoundaryLikelihoodRatioTest`](crate::stats::BoundaryLikelihoodRatioTest), [`ModelComparisonTable`](crate::stats::ModelComparisonTable)) | ✓ | ✓ | Stable, with a typed taxonomy and stable reason codes |

## Refusals

When a quantity is not identifiable, this crate refuses honestly rather than
fabricating it — both at construction time and during inference. See
[when the crate refuses](crate::guide::when_the_crate_refuses) for the full
contract.

## Explicitly out of scope (2.0 candidates)

- Multivariate response (`cbind(y1, y2) ~ …`).
- Profile-likelihood CIs for GLMMs.
- Parametric bootstrap for InverseGaussian and Normal-as-GLMM GLMMs.
- Full `I()` / arbitrary formula-level transformations beyond the minimal
  stateless subset.
- First-class `polars` / `arrow` ingestion (convert into
  [`DataFrame`](crate::model::DataFrame) instead).
- Kenward-Roger beyond the current scalar-test scope.

Back to [getting started](crate::guide::getting_started).
