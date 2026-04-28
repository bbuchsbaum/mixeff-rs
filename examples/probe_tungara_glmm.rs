//! Probe the public tungara GLMM stress fixture directly through the GLMM API.
//!
//! This is intentionally separate from `compare_rust`, whose current manifest
//! runner reports GLMMs as not wired yet.

use mixedmodels::datasets;
use mixedmodels::formula::parse_formula;
use mixedmodels::model::traits::MixedModelFit;
use mixedmodels::model::{Family, GeneralizedLinearMixedModel, LinkFunction};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (df, meta) = datasets::load("tungara_single_caller")?;
    let fit = &meta.fits[0];
    let formula = parse_formula(&fit.formula)?;

    let mut model = GeneralizedLinearMixedModel::new(
        formula,
        &df,
        Family::Binomial,
        Some(LinkFunction::Logit),
    )?;

    println!("prefit audit\n{}\n", model.audit_report().to_text());

    match model.fit() {
        Ok(_) => {
            println!("fit status: ok");
            println!("objective: {:.6}", model.objective());
            println!("beta: {:?}", model.coef().as_slice());
            println!("theta: {:?}", model.theta());
            println!("singular: {}", model.is_singular());
            println!("\npostfit audit\n{}", model.audit_report().to_text());
        }
        Err(error) => {
            println!("fit status: error");
            println!("error: {error}");
        }
    }

    Ok(())
}
