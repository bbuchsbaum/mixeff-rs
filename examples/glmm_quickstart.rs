//! Minimal generalized linear mixed model fit (Bernoulli / logit).
//!
//! Run:
//!     cargo run --example glmm_quickstart

use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::{
    DataFrame, Family, GeneralizedLinearMixedModel, LinkFunction, MixedModelFit,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Deterministic, non-separable binary data: a true logit of
    // -0.5 + 0.8*x + group_offset, thresholded against a pseudo-uniform
    // sequence so the two classes overlap (PIRLS needs non-separation).
    let group_offsets = [-1.2, -0.4, 0.3, 1.0, -0.8, 0.6, 1.3, -0.1, 0.9, -0.5];
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;

    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut g = Vec::new();
    for (gi, off) in group_offsets.iter().enumerate() {
        for k in 0..12 {
            let xv = (k % 5) as f64;
            let logit = -0.5 + 0.8 * xv + off;
            let p = 1.0 / (1.0 + (-logit).exp());
            // xorshift-ish deterministic uniform in [0, 1)
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let u = (seed >> 11) as f64 / (1u64 << 53) as f64;
            x.push(xv);
            y.push(if u < p { 1.0 } else { 0.0 });
            g.push(format!("g{gi}"));
        }
    }

    let mut df = DataFrame::new();
    df.add_numeric("y", y)?;
    df.add_numeric("x", x)?;
    df.add_categorical("g", g)?;

    let formula = parse_formula("y ~ 1 + x + (1 | g)")?;
    let mut model = GeneralizedLinearMixedModel::new(
        formula,
        &df,
        Family::Bernoulli,
        Some(LinkFunction::Logit),
    )?;
    model.fit()?;

    println!("fixed effects (intercept, x): {:?}", model.coef());
    Ok(())
}
