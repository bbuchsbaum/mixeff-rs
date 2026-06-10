//! Reproducer for bd-01KT40T6FGVXQQ9N50G2HM0ZZE: trust_bq premature
//! convergence on high-baseline random-intercept binomial GLMMs.
//!
//! Reads a JSON payload (y, x[/x1,x2], sub, item, plus the glmer reference) and
//! fits the joint Laplace GLMM, printing the gap to glmer. Defaults to the
//! committed regression fixture; pass a path to probe a different payload.
//!
//!   cargo run --no-default-features --features unstable-internals \
//!       --example probe_high_baseline_glmm

use std::fs;

use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::traits::MixedModelFit;
use mixeff_rs::model::{DataFrame, Family, GeneralizedLinearMixedModel, LinkFunction};
use serde_json::Value;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let default_fixture = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/regression/glmm_high_baseline_random_intercept.json"
    );
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| default_fixture.to_string());
    let raw = fs::read_to_string(&path)?;
    let v: Value = serde_json::from_str(&raw)?;

    let y: Vec<f64> = v["y"].as_array().unwrap().iter().map(num).collect();
    let two_covariates = v.get("x1").is_some();
    let sub: Vec<String> = v["sub"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| format!("s{}", e.as_i64().unwrap()))
        .collect();
    let item: Vec<String> = v["item"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| format!("i{}", e.as_i64().unwrap()))
        .collect();
    let glmer_loglik = v["glmer_loglik"].as_f64().unwrap();
    let glmer_fixef: Vec<f64> = v["glmer_fixef"]
        .as_array()
        .unwrap()
        .iter()
        .map(num)
        .collect();
    let glmer_theta: Vec<f64> = v["glmer_theta"]
        .as_array()
        .unwrap()
        .iter()
        .map(num)
        .collect();

    let mut data = DataFrame::new();
    data.add_numeric("y", y)?;
    if two_covariates {
        data.add_numeric("x1", v["x1"].as_array().unwrap().iter().map(num).collect())?;
        data.add_numeric("x2", v["x2"].as_array().unwrap().iter().map(num).collect())?;
    } else {
        data.add_numeric("x", v["x"].as_array().unwrap().iter().map(num).collect())?;
    }
    data.add_categorical("sub", sub)?;
    data.add_categorical("item", item)?;

    let formula = if two_covariates {
        parse_formula("y ~ x1 + x2 + (1 | sub) + (1 | item)")?
    } else {
        parse_formula("y ~ x + (1 | sub) + (1 | item)")?
    };
    let mut model = GeneralizedLinearMixedModel::new(
        formula,
        &data,
        Family::Binomial,
        Some(LinkFunction::Logit),
    )?;

    // fast = false -> certified joint Laplace path.
    model.fit_with_options(false, 1, false)?;

    let ll = model.loglikelihood();
    let beta = model.coef();
    let theta = model.theta();
    let optsum = model.lmm().optsum();

    println!("optimizer:     {:?}", optsum.optimizer);
    println!("return_value:  {}", optsum.return_value);
    println!("feval:         {}", optsum.feval);
    println!("max_feval:     {}", optsum.max_feval);
    println!("objective:     {:.6}", model.objective());
    println!();
    println!("mixeff logLik: {:.6}", ll);
    println!("glmer  logLik: {:.6}", glmer_loglik);
    println!("dlogLik:       {:.3e}", ll - glmer_loglik);
    println!();
    let max_dbeta = beta
        .iter()
        .zip(glmer_fixef.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    println!("mixeff fixef:  {:?}", beta.as_slice());
    println!("glmer  fixef:  {glmer_fixef:?}");
    println!("max|dFixef|:   {max_dbeta:.3e}");
    println!();
    println!("mixeff theta:  {:?}", theta);
    println!("glmer  theta:  {glmer_theta:?}");

    if let Some(cert) = model.compiler_artifact().optimizer_certificate.as_ref() {
        println!();
        println!("fit_status:    {:?}", cert.status);
        println!("free_grad:     {:?}", cert.free_gradient_norm);
        for diagnostic in &cert.diagnostics {
            println!(
                "diag {:?} [{:?}]: {}",
                diagnostic.code, diagnostic.severity, diagnostic.message
            );
            println!("  payload: {}", serde_json::json!(diagnostic.payload));
        }
    }

    Ok(())
}

fn num(e: &Value) -> f64 {
    e.as_f64().unwrap()
}
