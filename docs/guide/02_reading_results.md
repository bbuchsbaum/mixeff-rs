# Reading the results

A fitted model implements [`MixedModelFit`](crate::model::MixedModelFit), and
the [`stats`](crate::stats) module turns it into the usual summaries. Every
quantity below is computed, never fabricated — see
[when the crate refuses][crate::guide::when_the_crate_refuses] for what happens when
a quantity is not identifiable.

```rust
use mixeff_rs::prelude::*;
use mixeff_rs::model::{FitOptions, LinearMixedModelBuilder};
use mixeff_rs::stats::{CoefTable, ModelSummary};

# fn main() -> Result<()> {
# let group_offsets = [-3.0, -1.5, 0.5, 2.0, -2.0, 1.0, 3.0, -0.5];
# let jitter = [0.12, -0.20, 0.05, 0.17, -0.09, 0.22];
# let (mut y, mut x, mut g) = (Vec::new(), Vec::new(), Vec::new());
# for (gi, off) in group_offsets.iter().enumerate() {
#     for (k, j) in jitter.iter().enumerate() {
#         let xv = k as f64;
#         x.push(xv); y.push(2.0 + 1.5 * xv + off + j); g.push(format!("g{gi}"));
#     }
# }
# let mut df = DataFrame::new();
# df.add_numeric("y", y)?; df.add_numeric("x", x)?; df.add_categorical("g", g)?;
let model = LinearMixedModelBuilder::new(parse_formula("y ~ 1 + x + (1 | g)")?, &df)
    .fit(FitOptions::reml())?;

// --- point estimates and fit criteria (the MixedModelFit trait) ---
let coef = model.coef();              // fixed-effect estimates
let se = model.stderror();            // their standard errors
let theta = model.theta();            // relative covariance parameters
let _ = (model.aic(), model.bic(), model.loglikelihood());
assert_eq!(coef.len(), se.len());
assert!(!theta.is_empty());

// --- variance components ---
let vc = model.varcorr();             // VarCorr: SDs / correlations + residual
println!("{}", vc.to_markdown());

// --- the overall fit summary, ready to render ---
let summary = ModelSummary::from_linear_model(&model);
println!("{}", summary.to_markdown());   // also .to_html(), .to_latex()

// --- large-sample (Wald) confidence intervals ---
let ct = CoefTable::new(
    model.coef_names(),
    coef.iter().copied().collect(),
    se.iter().copied().collect(),
);
for row in ct.wald_confint(0.95) {
    println!("{}: [{:.3}, {:.3}]", row.parameter, row.lower, row.upper);
}
# Ok(())
# }
```

## Which interval to use

| Need | Use |
|------|-----|
| Quick large-sample CI | [`CoefTable::wald_confint`](crate::stats::CoefTable::wald_confint) |
| Small-sample / skewed likelihood | [`profile`](mod@crate::stats::profile) (profile-likelihood CIs) |
| Distribution-free, refit-based | [`parametricbootstrap`](crate::model::parametricbootstrap) |

Wald intervals are symmetric and cheap but degrade near a boundary (a variance
component pinned at zero). Profile and bootstrap intervals are the honest
choice there; both return **typed refusals with stable reason codes** rather
than an extrapolated bound when a side of the interval is not identifiable.

## Random effects and predictions

- [`MixedModelFit::ranef`](crate::model::MixedModelFit::ranef) — conditional
  modes (BLUPs) per grouping factor.
- [`MixedModelFit::fitted`](crate::model::MixedModelFit::fitted) — fitted
  values on the response scale.

Next: [generalized linear mixed models][crate::guide::glmms].
