# When the crate refuses

This crate's defining stance is: **no fake p-values, no hidden model surgery,
explicit identifiability and refusal paths.** When something cannot be
computed honestly, you get a typed error or a typed refusal with a stable
reason code — never a fabricated number and never a silently altered model.

## Refusal as a typed error

Construction and fitting reject ill-posed requests up front.
[`MixedModelError`](crate::error::MixedModelError) is `#[non_exhaustive]`, so
match it with a wildcard arm:

```rust
use mixeff_rs::prelude::*;
use mixeff_rs::model::{Family, GeneralizedLinearMixedModel, LinkFunction};

# fn main() -> Result<()> {
# let mut df = DataFrame::new();
# df.add_numeric("y", vec![1.0, 2.1, 3.0, 4.2, 5.1, 6.0])?;
# df.add_numeric("x", vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0])?;
# df.add_categorical("g", vec!["a","a","b","b","c","c"].into_iter().map(str::to_string).collect())?;
let formula = parse_formula("y ~ 1 + x + (1 | g)")?;

// Normal + Identity is a linear model, not a GLMM. The crate refuses
// rather than quietly fitting a degenerate GLMM.
let attempt = GeneralizedLinearMixedModel::new(
    formula,
    &df,
    Family::Normal,
    Some(LinkFunction::Identity),
);

match attempt {
    Ok(_) => panic!("expected a typed refusal"),
    Err(e) => {
        // A specific, actionable error — here, "use LinearMixedModel".
        let msg = e.to_string();
        assert!(!msg.is_empty());
    }
}
# Ok(())
# }
```

Other up-front refusals include a constant response
(`MixedModelError::ConstantResponse`), a formula with no random effects
(`NoRandomEffects`), rank-saturated or rank-deficient fixed effects
(`RankSaturatedFixedEffects` / `RankDeficient`), and unsupported
family/link pairs (`UnsupportedFamilyLink`). Each carries a stable
[`code()`](crate::error::MixedModelError::code) string for downstream
bindings.

## Refusal as a typed inference result

Inference does not throw; it classifies. Model comparison and likelihood-ratio
tests return a taxonomy, not a bare p-value:

- [`LikelihoodRatioTest`](crate::stats::LikelihoodRatioTest) /
  [`ModelComparisonTable`](crate::stats::ModelComparisonTable) carry a
  [`ModelComparisonClass`](crate::stats::ModelComparisonClass) and a stable
  [`ModelComparisonReasonCode`](crate::stats::ModelComparisonReasonCode).
- A variance-component test on a boundary is reported through
  [`BoundaryLikelihoodRatioTest`](crate::stats::BoundaryLikelihoodRatioTest)
  with an explicit [`BoundaryLrtStatus`](crate::stats::BoundaryLrtStatus) —
  the χ² mixture is named, not silently applied as a plain χ².
- Profile-likelihood CIs return a typed refusal for a side that is not
  identifiable instead of extrapolating a spline past the data.

The practical contract: **a number this crate hands you means what it says.**
If it cannot mean that, you get a refusal you can match on, with a reason code
stable across releases.

Back to [getting started][crate::guide::getting_started].
