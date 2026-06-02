# Getting started

This walks through one linear mixed model end to end: build a data frame,
parse an lme4-style formula, fit, and read the estimates.

The model is `y ~ 1 + x + (1 + x | g)` — a fixed intercept and slope in `x`,
with a per-group random intercept and slope correlated within group `g`.

```rust
use mixeff_rs::prelude::*;
use mixeff_rs::model::{FitOptions, LinearMixedModelBuilder};

# fn main() -> Result<()> {
// Balanced toy data: 8 groups, a clear fixed slope plus group offsets.
let group_offsets = [-3.0, -1.5, 0.5, 2.0, -2.0, 1.0, 3.0, -0.5];
let jitter = [0.12, -0.20, 0.05, 0.17, -0.09, 0.22];

let mut y = Vec::new();
let mut x = Vec::new();
let mut g = Vec::new();
for (gi, off) in group_offsets.iter().enumerate() {
    for (k, j) in jitter.iter().enumerate() {
        let xv = k as f64;
        x.push(xv);
        y.push(2.0 + 1.5 * xv + off + j);
        g.push(format!("g{gi}"));
    }
}

let mut df = DataFrame::new();
df.add_numeric("y", y)?;
df.add_numeric("x", x)?;
df.add_categorical("g", g)?;

// The builder collapses construction and the ML/REML choice into one chain.
let model = LinearMixedModelBuilder::new(parse_formula("y ~ 1 + x + (1 | g)")?, &df)
    .fit(FitOptions::reml())?; // or FitOptions::ml()

let coef = model.coef();          // fixed effects, ~[2.0, 1.5]
assert_eq!(coef.len(), 2);
assert_eq!(model.coef_names(), vec!["(Intercept)".to_string(), "x".to_string()]);
# Ok(())
# }
```

## What just happened

1. [`DataFrame`](crate::model::DataFrame) is a small column-oriented frame.
   Numeric columns are `f64`; categorical columns are encoded by
   first-appearance order. Real callers typically convert from polars/arrow
   into this.
2. [`parse_formula`](crate::formula::parse_formula) accepts R/lme4 syntax:
   `*`, `:`, `/`, `(re | g)`, `(re || g)` (zero-correlation), and
   `(re | g1 & g2)` interactions. `0 +` / `-1` / `1 +` control the intercept.
3. [`LinearMixedModelBuilder`](crate::model::LinearMixedModelBuilder) chooses
   the optimizer automatically based on the θ dimension by default.
   [`FitOptions`](crate::model::FitOptions) carries the ML/REML choice plus a
   narrow, audit-recorded [`OptimizerControl`](crate::model::OptimizerControl)
   escape hatch for recourse, warm starts, and tolerance overrides.

The lower-level form is still available and purely additive:

```rust
# use mixeff_rs::prelude::*;
# use mixeff_rs::model::LinearMixedModel;
# fn main() -> Result<()> {
# let mut df = DataFrame::new();
# df.add_numeric("y", vec![1.0, 2.1, 3.0, 4.2, 5.1, 6.0])?;
# df.add_numeric("x", vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0])?;
# df.add_categorical("g", vec!["a","a","b","b","c","c"].into_iter().map(str::to_string).collect())?;
let formula = parse_formula("y ~ 1 + x + (1 | g)")?;
let mut model = LinearMixedModel::new(formula, &df, None)?;
model.fit(false)?; // false = ML, true = REML
# Ok(())
# }
```

Next: [reading the results][crate::guide::reading_results].
