# Generalized linear mixed models

A GLMM swaps the Gaussian response for a [`Family`](crate::model::Family) and a
[`LinkFunction`](crate::model::LinkFunction).
[`GeneralizedLinearMixedModel`](crate::model::GeneralizedLinearMixedModel)
fits the conditional modes by PIRLS, with an optional adaptive Gauss-Hermite
quadrature refinement. The builder mirrors the LMM one but takes a family:

```rust
use mixeff_rs::prelude::*;
use mixeff_rs::model::{Family, GeneralizedLinearMixedModelBuilder};

# fn main() -> Result<()> {
// Deterministic count data: log-mean 0.8 + 0.1*x + a small group offset.
let offs = [-0.2, 0.1, 0.0, 0.15, -0.05];
let (mut y, mut x, mut g) = (Vec::new(), Vec::new(), Vec::new());
for grp in 0..5 {
    for obs in 0..8 {
        let xv = obs as f64 - 3.5;
        let eta = 0.8 + 0.1 * xv + offs[grp];
        y.push(eta.exp().round().max(1.0));
        x.push(xv);
        g.push(format!("g{}", grp + 1));
    }
}
let mut df = DataFrame::new();
df.add_numeric("y", y)?;
df.add_numeric("x", x)?;
df.add_categorical("g", g)?;

let model = GeneralizedLinearMixedModelBuilder::new(
    parse_formula("y ~ 1 + x + (1 | g)")?,
    &df,
    Family::Poisson, // canonical link (log) chosen automatically
)
.fit()?;

assert_eq!(model.coef().len(), 2);
# Ok(())
# }
```

Pass an explicit link through the lower-level constructor
`GeneralizedLinearMixedModel::new(formula, &df, family, Some(link))` when you
do not want the canonical default. `Family::Normal` with
`LinkFunction::Identity` is **rejected** here on purpose — that is a linear
mixed model, so use [`LinearMixedModel`](crate::model::LinearMixedModel). See
[when the crate refuses][crate::guide::when_the_crate_refuses].

## Estimation semantics — read this before reporting GLMM inference

The default path is `fast=true`: profiled fast-PIRLS with Laplace/AGQ
approximation metadata carried in the fit summary. **It is intentionally not
the same statistical approximation as `lme4::glmer`'s joint Laplace fit**, and
it can be less accurate for inference on overdispersed or
observation-level-random-effect models.

`fast=false` selects a labelled joint path: Laplace for `n_agq <= 1`, and AGQ
for valid single-scalar random-effect GLMMs with `n_agq > 1`. NLopt builds use
BOBYQA; dependency-light builds use the native TrustBQ joint path. Caller
`max_feval` is honored for bounded joint attempts, and the fit artifact records
evaluation counts and typed convergence status. Any joint attempt or
fast-PIRLS fallback is labelled in the optimizer status and diagnostics — it is
never silently presented as ordinary `lme4` parity.

This is the project's no-fake-statistics stance applied to GLMMs: the
approximation actually used is always recoverable from the fit, so a reported
number can be trusted to mean what it says.

Next: [when the crate refuses][crate::guide::when_the_crate_refuses].
