//! Fit every recommended model in `datasets/*/meta.toml` with this Rust crate
//! and dump the results to JSON for cross-checking against R lme4.
//!
//! Companion: `scripts/compare_lme4.R` (reads `comparison/manifest.json`,
//! emits `comparison/lme4_results.json`) and `examples/compare_report.rs`
//! (joins both and writes `comparison/REPORT.md`).
//!
//! Run:
//!     cargo run --release --example compare_rust
//!
//! Currently fits only Gaussian/Identity LMMs through `LinearMixedModel`.
//! GLMMs (cbpp, verbagg, grouseticks) are emitted to the manifest with a
//! `status = "not_implemented"` placeholder so the reporter can show them
//! as gaps.

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use serde::Serialize;
use serde_json::json;

use mixedmodels::datasets;
use mixedmodels::formula::parse_formula;
use mixedmodels::model::data::DataFrame;
use mixedmodels::model::linear::LinearMixedModel;
use mixedmodels::model::traits::MixedModelFit;

const TIMING_REPEATS: usize = 3;

#[derive(Serialize, Clone)]
struct ManifestEntry {
    dataset: String,
    formula: String,
    family: String,
    link: String,
    estimator: String,
    weights: Option<String>,
}

#[derive(Serialize)]
struct ResultRecord {
    dataset: String,
    formula: String,
    family: String,
    link: String,
    estimator: String,
    n_obs: usize,
    status: String,
    error: Option<String>,
    beta: Option<Vec<f64>>,
    coef_names: Option<Vec<String>>,
    sigma: Option<f64>,
    theta: Option<Vec<f64>>,
    objective: Option<f64>,
    loglik: Option<f64>,
    aic: Option<f64>,
    bic: Option<f64>,
    is_singular: Option<bool>,
    fit_time_ms: Option<f64>,
    fit_time_ms_min: Option<f64>,
    fit_time_ms_repeats: Option<usize>,
}

fn datasets_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("datasets")
}

fn comparison_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("comparison")
}

fn discover_datasets() -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(datasets_root())
        .expect("read datasets/")
        .filter_map(|e| {
            let e = e.ok()?;
            let path = e.path();
            if !path.is_dir() {
                return None;
            }
            if !path.join("meta.toml").is_file() {
                return None;
            }
            path.file_name()?.to_str().map(|s| s.to_string())
        })
        .collect();
    names.sort();
    names
}

/// Outcome of attempting to fit a single model. We separate the genuine
/// runtime errors (which should bubble up) from "feature gap" cases like
/// categorical predictors in random slopes — those should be reported in
/// the comparison table as known unsupported, not as failures.
enum FitOutcome {
    Ok(LmmFitResult),
    Unsupported(String),
}

fn looks_like_unsupported(msg: &str) -> bool {
    // Surface known feature-gap signatures so the report distinguishes
    // "not implemented yet" from "the optimizer crashed".
    msg.contains("not found or not numeric")
        || msg.contains("interaction") && msg.contains("not supported")
}

fn fit_lmm(
    df: &DataFrame,
    formula_str: &str,
    reml: bool,
) -> Result<FitOutcome, Box<dyn std::error::Error>> {
    // Time the full pipeline (parse → construct → fit) so the result is
    // directly comparable to a single `lmer()` call. Reconstruct the model
    // each repeat — `.fit()` is not idempotent on a fitted model.
    let mut times_ms: Vec<f64> = Vec::with_capacity(TIMING_REPEATS);
    let mut last_model: Option<LinearMixedModel> = None;

    for _ in 0..TIMING_REPEATS {
        let t0 = Instant::now();
        let formula = parse_formula(formula_str)?;
        let mut model = match LinearMixedModel::new(formula, df, None) {
            Ok(m) => m,
            Err(e) => {
                let msg = e.to_string();
                if looks_like_unsupported(&msg) {
                    return Ok(FitOutcome::Unsupported(msg));
                }
                return Err(Box::new(e));
            }
        };
        model.fit(reml)?;
        times_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        last_model = Some(model);
    }

    let model = last_model.unwrap();
    let cold_ms = times_ms[0];
    let min_ms = times_ms.iter().cloned().fold(f64::INFINITY, f64::min);

    let beta: Vec<f64> = MixedModelFit::coef(&model).iter().cloned().collect();
    let names = MixedModelFit::coef_names(&model);
    Ok(FitOutcome::Ok(LmmFitResult {
        beta,
        coef_names: names,
        sigma: model.sigma(),
        theta: model.theta(),
        objective: model.objective_value(),
        loglik: MixedModelFit::loglikelihood(&model),
        aic: MixedModelFit::aic(&model),
        bic: MixedModelFit::bic(&model),
        is_singular: MixedModelFit::is_singular(&model),
        fit_time_ms: cold_ms,
        fit_time_ms_min: min_ms,
        n_obs: MixedModelFit::nobs(&model),
    }))
}

struct LmmFitResult {
    beta: Vec<f64>,
    coef_names: Vec<String>,
    sigma: f64,
    theta: Vec<f64>,
    objective: f64,
    loglik: f64,
    aic: f64,
    bic: f64,
    is_singular: bool,
    fit_time_ms: f64,
    fit_time_ms_min: f64,
    n_obs: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let comparison = comparison_root();
    fs::create_dir_all(&comparison)?;

    let mut manifest = Vec::new();
    let mut results = Vec::new();

    for name in discover_datasets() {
        let (df, meta) = match datasets::load(&name) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("skip {name}: {e}");
                continue;
            }
        };
        let n_obs = df.nrow();

        for fit in &meta.fits {
            let entry = ManifestEntry {
                dataset: name.clone(),
                formula: fit.formula.clone(),
                family: fit.family.clone(),
                link: fit.link.clone(),
                estimator: fit.estimator.clone(),
                weights: fit.weights.clone(),
            };
            manifest.push(entry.clone());

            let mut rec = ResultRecord {
                dataset: name.clone(),
                formula: fit.formula.clone(),
                family: fit.family.clone(),
                link: fit.link.clone(),
                estimator: fit.estimator.clone(),
                n_obs,
                status: "skipped".into(),
                error: None,
                beta: None,
                coef_names: None,
                sigma: None,
                theta: None,
                objective: None,
                loglik: None,
                aic: None,
                bic: None,
                is_singular: None,
                fit_time_ms: None,
                fit_time_ms_min: None,
                fit_time_ms_repeats: None,
            };

            // Only fit Gaussian/Identity LMMs in v1 — GLMMs require additional plumbing.
            let is_gaussian_identity = fit.family.eq_ignore_ascii_case("Gaussian")
                && fit.link.eq_ignore_ascii_case("Identity");
            if !is_gaussian_identity {
                rec.status = "not_implemented".into();
                rec.error = Some(format!(
                    "{}/{} not yet wired into compare_rust",
                    fit.family, fit.link
                ));
                results.push(rec);
                continue;
            }

            let reml = match fit.estimator.to_ascii_uppercase().as_str() {
                "REML" => true,
                "ML" => false,
                other => {
                    rec.status = "error".into();
                    rec.error = Some(format!("unknown estimator `{other}`"));
                    results.push(rec);
                    continue;
                }
            };

            print!("fitting {name} :: {} [{}] ... ", fit.formula, fit.estimator);

            match fit_lmm(&df, &fit.formula, reml) {
                Ok(FitOutcome::Ok(r)) => {
                    println!(
                        "obj={:.4}  σ={:.4}  cold={:.1}ms  min={:.1}ms",
                        r.objective, r.sigma, r.fit_time_ms, r.fit_time_ms_min
                    );
                    rec.status = "ok".into();
                    rec.n_obs = r.n_obs;
                    rec.beta = Some(r.beta);
                    rec.coef_names = Some(r.coef_names);
                    rec.sigma = Some(r.sigma);
                    rec.theta = Some(r.theta);
                    rec.objective = Some(r.objective);
                    rec.loglik = Some(r.loglik);
                    rec.aic = Some(r.aic);
                    rec.bic = Some(r.bic);
                    rec.is_singular = Some(r.is_singular);
                    rec.fit_time_ms = Some(r.fit_time_ms);
                    rec.fit_time_ms_min = Some(r.fit_time_ms_min);
                    rec.fit_time_ms_repeats = Some(TIMING_REPEATS);
                }
                Ok(FitOutcome::Unsupported(msg)) => {
                    println!("UNSUPPORTED: {msg}");
                    rec.status = "unsupported".into();
                    rec.error = Some(msg);
                }
                Err(e) => {
                    println!("ERROR: {e}");
                    rec.status = "error".into();
                    rec.error = Some(e.to_string());
                }
            }
            results.push(rec);
        }
    }

    let manifest_path = comparison.join("manifest.json");
    let results_path = comparison.join("rust_results.json");
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&json!({ "fits": manifest }))?,
    )?;
    fs::write(
        &results_path,
        serde_json::to_string_pretty(&json!({
            "tool": "mixedmodels (rust)",
            "version": env!("CARGO_PKG_VERSION"),
            "results": results,
        }))?,
    )?;
    println!(
        "\nwrote {} entries to {}",
        results.len(),
        results_path.display()
    );
    println!("wrote manifest to {}", manifest_path.display());
    Ok(())
}
