//! Minimal linear mixed model fit.
//!
//! Run:
//!     cargo run --example quickstart

use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::{DataFrame, LinearMixedModel, MixedModelFit};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Balanced toy data: 8 groups, a clear fixed slope plus group intercepts.
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

    let formula = parse_formula("y ~ 1 + x + (1 | g)")?;
    let mut model = LinearMixedModel::new(formula, &df, None)?;
    model.fit(false)?; // false = ML, true = REML

    println!("fixed effects (intercept, x): {:?}", model.coef());
    println!("objective (-2 log-likelihood): {}", model.objective());
    println!("AIC: {}", model.aic());
    Ok(())
}
