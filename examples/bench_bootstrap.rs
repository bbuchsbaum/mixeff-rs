//! Bootstrap benchmark harness.
//!
//! Run:
//!
//! ```text
//! cargo run --release --example bench_bootstrap -- 100
//! ```
//!
//! The optional argument is the requested bootstrap replicate count. The output
//! is newline-delimited JSON so results can be archived or compared by scripts.

use mixedmodels::compiler::FixedEffectHypothesis;
use mixedmodels::datasets;
use mixedmodels::formula::parse_formula;
use mixedmodels::model::{
    BootstrapFailedRefitPolicy, FixedEffectBootstrapOptions, LinearMixedModel, MixedModelFit,
};
use serde::Serialize;
use std::time::Instant;

#[derive(Debug, Serialize)]
struct BenchmarkRow {
    benchmark: &'static str,
    dataset: &'static str,
    formula: String,
    requested_replicates: usize,
    completed_replicates: usize,
    successful_replicates: usize,
    finite_statistic_count: Option<usize>,
    failed_refits: usize,
    boundary_count: usize,
    boundary_rate: Option<f64>,
    elapsed_ms: u128,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let requested_replicates = std::env::args()
        .nth(1)
        .map(|arg| arg.parse::<usize>())
        .transpose()?
        .unwrap_or(100);
    let options = FixedEffectBootstrapOptions {
        requested_replicates,
        failed_refit_policy: BootstrapFailedRefitPolicy::Exclude,
        seed: Some(20260506),
    };

    bench_fixed_effect_null(&options)?;
    bench_bootstrap_lrt(&options)?;
    bench_cluster_interval(&options)?;
    Ok(())
}

fn bench_fixed_effect_null(
    options: &FixedEffectBootstrapOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let (data, _) = datasets::load("sleepstudy")?;
    let formula = parse_formula("Reaction ~ 1 + Days + (1 | Subject)")?;
    let mut model = LinearMixedModel::new(formula, &data, None)?;
    model.fit(false)?;
    let days_index = model
        .coef_names()
        .iter()
        .position(|name| name == "Days")
        .ok_or("missing Days coefficient")?;
    let hypothesis = FixedEffectHypothesis::single_coefficient(
        "Days = 0",
        days_index,
        model.coef_names().len(),
    )?;

    let start = Instant::now();
    let row = model.fixed_effect_null_bootstrap_inference_row(
        mixedmodels::compiler::FixedEffectInferenceRowKind::Coefficient,
        hypothesis,
        options,
    );
    let elapsed_ms = start.elapsed().as_millis();
    let bootstrap = row
        .details
        .as_ref()
        .and_then(|details| details.bootstrap.as_ref())
        .ok_or("fixed-effect null row did not return bootstrap details")?;
    println!(
        "{}",
        serde_json::to_string(&BenchmarkRow {
            benchmark: "fixed_effect_null",
            dataset: "sleepstudy",
            formula: model.formula.to_string(),
            requested_replicates: bootstrap.requested_replicates,
            completed_replicates: bootstrap.completed_replicates,
            successful_replicates: bootstrap.successful_replicates,
            finite_statistic_count: bootstrap.finite_statistic_count,
            failed_refits: bootstrap.failed_refits,
            boundary_count: bootstrap.boundary_count,
            boundary_rate: bootstrap.boundary_rate,
            elapsed_ms,
        })?
    );
    Ok(())
}

fn bench_bootstrap_lrt(
    options: &FixedEffectBootstrapOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let (data, _) = datasets::load("sleepstudy")?;
    let formula0 = parse_formula("Reaction ~ 1 + (1 | Subject)")?;
    let formula1 = parse_formula("Reaction ~ 1 + Days + (1 | Subject)")?;
    let mut reduced = LinearMixedModel::new(formula0, &data, None)?;
    let mut alternative = LinearMixedModel::new(formula1, &data, None)?;
    reduced.fit(false)?;
    alternative.fit(false)?;

    let start = Instant::now();
    let result = reduced.bootstrap_likelihood_ratio_test(&alternative, options)?;
    let elapsed_ms = start.elapsed().as_millis();
    let metadata = &result.payload.metadata;
    println!(
        "{}",
        serde_json::to_string(&BenchmarkRow {
            benchmark: "likelihood_ratio",
            dataset: "sleepstudy",
            formula: format!("{} vs {}", reduced.formula, alternative.formula),
            requested_replicates: metadata.requested_replicates,
            completed_replicates: metadata.completed_replicates,
            successful_replicates: metadata.successful_replicates,
            finite_statistic_count: metadata.finite_statistic_count,
            failed_refits: metadata.failed_refits,
            boundary_count: metadata.boundary_count,
            boundary_rate: metadata.boundary_rate,
            elapsed_ms,
        })?
    );
    Ok(())
}

fn bench_cluster_interval(
    options: &FixedEffectBootstrapOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let (data, _) = datasets::load("dyestuff")?;
    let formula = parse_formula("Yield ~ 1 + (1 | Batch)")?;
    let mut model = LinearMixedModel::new(formula, &data, None)?;
    model.fit(false)?;
    let hypothesis =
        FixedEffectHypothesis::single_coefficient("intercept", 0, model.coef_names().len())?;

    let start = Instant::now();
    let payload = model.cluster_resample_full_model_contrast_payload(
        &data,
        "Batch",
        &hypothesis,
        options,
        &[0.95],
    )?;
    let elapsed_ms = start.elapsed().as_millis();
    let metadata = &payload.metadata;
    println!(
        "{}",
        serde_json::to_string(&BenchmarkRow {
            benchmark: "cluster_resample_interval",
            dataset: "dyestuff",
            formula: model.formula.to_string(),
            requested_replicates: metadata.requested_replicates,
            completed_replicates: metadata.completed_replicates,
            successful_replicates: metadata.successful_replicates,
            finite_statistic_count: metadata.finite_statistic_count,
            failed_refits: metadata.failed_refits,
            boundary_count: metadata.boundary_count,
            boundary_rate: metadata.boundary_rate,
            elapsed_ms,
        })?
    );
    Ok(())
}
