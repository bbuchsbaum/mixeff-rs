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
//! Fits Gaussian/Identity LMMs through `LinearMixedModel` and supported GLMM
//! families through `GeneralizedLinearMixedModel`.

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use serde::Serialize;
use serde_json::json;

use mixeff_rs::datasets;
use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::data::DataFrame;
use mixeff_rs::model::generalized::GeneralizedLinearMixedModel;
use mixeff_rs::model::linear::LinearMixedModel;
use mixeff_rs::model::traits::{Family, LinkFunction, MixedModelFit};

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
    objective_definition: Option<String>,
    response_constants: Option<String>,
    optimizer: Option<String>,
    optimizer_backend: Option<String>,
    optimizer_return_code: Option<String>,
    optimizer_fevals: Option<i64>,
    optimizer_fmin: Option<f64>,
    optimizer_max_fevals: Option<i64>,
    is_singular: Option<bool>,
    fit_time_ms: Option<f64>,
    fit_time_ms_min: Option<f64>,
    fit_time_ms_repeats: Option<usize>,
}

fn comparison_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("comparison")
}

/// Outcome of attempting to fit a single model. We separate the genuine
/// runtime errors (which should bubble up) from "feature gap" cases like
/// categorical predictors in random slopes — those should be reported in
/// the comparison table as known unsupported, not as failures.
enum FitOutcome {
    Ok(Box<FitResult>),
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
    let opt = MixedModelFit::opt_summary(&model);
    Ok(FitOutcome::Ok(Box::new(FitResult {
        beta,
        coef_names: names,
        sigma: model.sigma(),
        theta: model.theta(),
        objective: model.objective_value(),
        loglik: MixedModelFit::loglikelihood(&model),
        aic: MixedModelFit::aic(&model),
        bic: MixedModelFit::bic(&model),
        optimizer: opt.optimizer_name().to_string(),
        optimizer_backend: opt.backend_name().to_string(),
        optimizer_return_code: opt.return_value.clone(),
        optimizer_fevals: opt.feval,
        optimizer_fmin: opt.fmin,
        optimizer_max_fevals: opt.max_feval,
        is_singular: MixedModelFit::is_singular(&model),
        fit_time_ms: cold_ms,
        fit_time_ms_min: min_ms,
        n_obs: MixedModelFit::nobs(&model),
    })))
}

struct FitResult {
    beta: Vec<f64>,
    coef_names: Vec<String>,
    sigma: f64,
    theta: Vec<f64>,
    objective: f64,
    loglik: f64,
    aic: f64,
    bic: f64,
    optimizer: String,
    optimizer_backend: String,
    optimizer_return_code: String,
    optimizer_fevals: i64,
    optimizer_fmin: f64,
    optimizer_max_fevals: i64,
    is_singular: bool,
    fit_time_ms: f64,
    fit_time_ms_min: f64,
    n_obs: usize,
}

struct PreparedGlmmInput {
    data: DataFrame,
    formula: String,
    weights: Option<Vec<f64>>,
    offset: Option<Vec<f64>>,
}

fn parse_glmm_family(label: &str) -> Option<Family> {
    match label.to_ascii_lowercase().as_str() {
        "bernoulli" => Some(Family::Bernoulli),
        "binomial" => Some(Family::Binomial),
        "poisson" => Some(Family::Poisson),
        "gamma" => Some(Family::Gamma),
        _ => None,
    }
}

fn parse_glmm_link(label: &str) -> Option<LinkFunction> {
    match label.to_ascii_lowercase().as_str() {
        "identity" => Some(LinkFunction::Identity),
        "log" => Some(LinkFunction::Log),
        "logit" => Some(LinkFunction::Logit),
        "probit" => Some(LinkFunction::Probit),
        "inverse" => Some(LinkFunction::Inverse),
        "sqrt" => Some(LinkFunction::Sqrt),
        _ => None,
    }
}

fn glmm_n_agq(estimator: &str) -> Option<usize> {
    match estimator.to_ascii_uppercase().as_str() {
        "LAPLACE" => Some(1),
        // Pin a repo-wide convention until metadata grows an explicit n_agq field.
        // This matches the existing cbpp/AGQ parity tests.
        "AGQ" => Some(7),
        _ => None,
    }
}

fn prepared_numeric_column(df: &DataFrame, name: &str) -> Result<Vec<f64>, String> {
    df.numeric(name)
        .map(|values| values.to_vec())
        .ok_or_else(|| format!("column `{name}` is not present or not numeric"))
}

fn remove_formula_term(formula: &str, start: usize, end: usize) -> String {
    let before_trimmed = formula[..start].trim_end();
    let after_trimmed = formula[end..].trim_start();

    let (remove_start, remove_end) = if before_trimmed.ends_with('+') {
        (before_trimmed.len() - 1, end)
    } else if after_trimmed.starts_with('+') {
        let after_ws = formula[end..].len() - after_trimmed.len();
        (start, end + after_ws + 1)
    } else {
        (start, end)
    };

    format!("{}{}", &formula[..remove_start], &formula[remove_end..])
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn extract_log_offset(formula: &str, df: &DataFrame) -> Result<(String, Option<Vec<f64>>), String> {
    let Some(start) = formula.find("offset(log(") else {
        return Ok((formula.to_string(), None));
    };
    let col_start = start + "offset(log(".len();
    let rest = &formula[col_start..];
    let Some(col_len) = rest.find("))") else {
        return Err("only offset(log(column)) terms are supported in compare_rust".to_string());
    };
    let column = rest[..col_len].trim();
    if column.is_empty() {
        return Err("offset(log()) must name a numeric column".to_string());
    }
    let values = prepared_numeric_column(df, column)?;
    let offset = values
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            if *value <= 0.0 || !value.is_finite() {
                Err(format!(
                    "offset column `{column}` must be finite and positive; row {idx} has {value}"
                ))
            } else {
                Ok(value.ln())
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let term_end = col_start + col_len + 2;
    Ok((remove_formula_term(formula, start, term_end), Some(offset)))
}

fn binary_response_column_name(response_name: &str) -> String {
    format!("__{}_binary", response_name)
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect()
}

fn prepare_binomial_response(
    df: &DataFrame,
    formula: &str,
    weights_name: Option<&str>,
) -> Result<(DataFrame, String, Option<Vec<f64>>), String> {
    let Some((lhs, rhs)) = formula.split_once('~') else {
        return Err("formula must contain `~`".to_string());
    };
    let lhs = lhs.trim();
    let rhs = rhs.trim();

    let Some((events_name, trials_name)) = lhs.split_once('/') else {
        if df.numeric(lhs).is_none() {
            let Some(column) = df.categorical(lhs) else {
                return Err(format!("Response `{lhs}` not found or not numeric"));
            };
            if column.levels.len() != 2 {
                return Err(format!(
                    "binomial response `{lhs}` is categorical with {} levels; expected exactly two",
                    column.levels.len()
                ));
            }
            let response_name = binary_response_column_name(lhs);
            let values = column
                .refs
                .iter()
                .map(|&level| f64::from(level == 1))
                .collect::<Vec<_>>();
            let mut data = df.clone();
            data.add_numeric(&response_name, values)
                .map_err(|error| error.to_string())?;
            let weights = weights_name
                .map(|name| prepared_numeric_column(&data, name))
                .transpose()?;
            return Ok((data, format!("{response_name} ~ {rhs}"), weights));
        }
        let weights = weights_name
            .map(|name| prepared_numeric_column(df, name))
            .transpose()?;
        return Ok((df.clone(), formula.to_string(), weights));
    };

    let events_name = events_name.trim();
    let trials_name = trials_name.trim();
    let weights_column = weights_name.unwrap_or(trials_name);
    if weights_column != trials_name {
        return Err(format!(
            "grouped-binomial response denominator `{trials_name}` does not match weights column `{weights_column}`"
        ));
    }

    let events = prepared_numeric_column(df, events_name)?;
    let trials = prepared_numeric_column(df, trials_name)?;
    let proportion = events
        .iter()
        .zip(trials.iter())
        .enumerate()
        .map(|(idx, (&event, &trial))| {
            if !event.is_finite() || !trial.is_finite() || trial <= 0.0 {
                return Err(format!(
                    "grouped-binomial row {idx} must have finite events and positive trials"
                ));
            }
            if event < 0.0 || event > trial {
                return Err(format!(
                    "grouped-binomial row {idx} has events={event} outside [0, {trial}]"
                ));
            }
            Ok(event / trial)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let response_name = format!("__{}_over_{}", events_name, trials_name)
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect::<String>();
    let mut data = df.clone();
    data.add_numeric(&response_name, proportion)
        .map_err(|error| error.to_string())?;
    Ok((data, format!("{response_name} ~ {rhs}"), Some(trials)))
}

fn prepare_glmm_input(
    df: &DataFrame,
    formula: &str,
    family: Family,
    weights_name: Option<&str>,
) -> Result<PreparedGlmmInput, String> {
    let (data, formula, weights) = if matches!(family, Family::Binomial | Family::Bernoulli) {
        prepare_binomial_response(df, formula, weights_name)?
    } else {
        let weights = weights_name
            .map(|name| prepared_numeric_column(df, name))
            .transpose()?;
        (df.clone(), formula.to_string(), weights)
    };
    let (formula, offset) = extract_log_offset(&formula, &data)?;
    Ok(PreparedGlmmInput {
        data,
        formula,
        weights,
        offset,
    })
}

fn fit_glmm(
    df: &DataFrame,
    formula_str: &str,
    family_label: &str,
    link_label: &str,
    estimator: &str,
    weights_name: Option<&str>,
    fast: bool,
) -> Result<FitOutcome, Box<dyn std::error::Error>> {
    let Some(family) = parse_glmm_family(family_label) else {
        return Ok(FitOutcome::Unsupported(format!(
            "GLMM family `{family_label}` is not supported by compare_rust"
        )));
    };
    let Some(link) = parse_glmm_link(link_label) else {
        return Ok(FitOutcome::Unsupported(format!(
            "GLMM link `{link_label}` is not supported by compare_rust"
        )));
    };
    let Some(n_agq) = glmm_n_agq(estimator) else {
        return Ok(FitOutcome::Unsupported(format!(
            "GLMM estimator `{estimator}` is not supported by compare_rust"
        )));
    };

    let prepared = match prepare_glmm_input(df, formula_str, family, weights_name) {
        Ok(prepared) => prepared,
        Err(message) => return Ok(FitOutcome::Unsupported(message)),
    };

    let mut times_ms: Vec<f64> = Vec::with_capacity(TIMING_REPEATS);
    let mut last_model: Option<GeneralizedLinearMixedModel> = None;

    for _ in 0..TIMING_REPEATS {
        let t0 = Instant::now();
        let formula = parse_formula(&prepared.formula)?;
        let model = match (&prepared.weights, &prepared.offset) {
            (Some(weights), Some(offset)) => {
                GeneralizedLinearMixedModel::new_with_weights_and_offset(
                    formula,
                    &prepared.data,
                    family,
                    Some(link),
                    weights.clone(),
                    offset.clone(),
                )
            }
            (Some(weights), None) => GeneralizedLinearMixedModel::new_with_weights(
                formula,
                &prepared.data,
                family,
                Some(link),
                weights.clone(),
            ),
            (None, Some(offset)) => GeneralizedLinearMixedModel::new_with_offset(
                formula,
                &prepared.data,
                family,
                Some(link),
                offset.clone(),
            ),
            (None, None) => {
                GeneralizedLinearMixedModel::new(formula, &prepared.data, family, Some(link))
            }
        };
        let mut model = match model {
            Ok(model) => model,
            Err(error) => {
                let msg = error.to_string();
                if looks_like_unsupported(&msg) {
                    return Ok(FitOutcome::Unsupported(msg));
                }
                return Err(Box::new(error));
            }
        };
        if let Err(error) = model.fit_with_options(fast, n_agq, false) {
            let msg = error.to_string();
            if looks_like_unsupported(&msg) {
                return Ok(FitOutcome::Unsupported(msg));
            }
            return Err(Box::new(error));
        }
        times_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        last_model = Some(model);
    }

    let model = last_model.unwrap();
    let cold_ms = times_ms[0];
    let min_ms = times_ms.iter().cloned().fold(f64::INFINITY, f64::min);
    let opt = MixedModelFit::opt_summary(&model);
    Ok(FitOutcome::Ok(Box::new(FitResult {
        beta: MixedModelFit::coef(&model).iter().cloned().collect(),
        coef_names: MixedModelFit::coef_names(&model),
        sigma: MixedModelFit::dispersion(&model, false),
        theta: model.theta(),
        objective: MixedModelFit::objective(&model),
        loglik: MixedModelFit::loglikelihood(&model),
        aic: MixedModelFit::aic(&model),
        bic: MixedModelFit::bic(&model),
        optimizer: opt.optimizer_name().to_string(),
        optimizer_backend: opt.backend_name().to_string(),
        optimizer_return_code: opt.return_value.clone(),
        optimizer_fevals: opt.feval,
        optimizer_fmin: opt.fmin,
        optimizer_max_fevals: opt.max_feval,
        is_singular: MixedModelFit::is_singular(&model),
        fit_time_ms: cold_ms,
        fit_time_ms_min: min_ms,
        n_obs: MixedModelFit::nobs(&model),
    })))
}

fn use_fast_glmm_comparison_path(family: &str, n_obs: usize) -> bool {
    let _ = (family, n_obs);
    // Current main exposes the profiled-θ PIRLS path for GLMM fitting. The
    // older non-fast joint path from the stash is no longer implemented, so
    // comparison artifacts must classify lme4 differences against explicit
    // objective/constant conventions and fast-oracle fixtures.
    true
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let comparison = comparison_root();
    fs::create_dir_all(&comparison)?;

    let mut manifest = Vec::new();
    let mut results = Vec::new();

    // Stress-tier datasets (kb07, mrk17_exp1, future InstEval) have maximal-RE
    // fits that take many minutes through the in-crate engine. Skip the fit
    // step (but still emit them in the manifest) unless the caller explicitly
    // opts in. The manifest derivation is independent of fit success, so
    // fixture_hygiene::comparison_manifest_matches_registry_derived stays green.
    let include_stress = std::env::var("MIXEDMODELS_INCLUDE_STRESS").is_ok();

    // Drive every (dataset, fit) pair from the catalog. comparison/manifest.json
    // is now a *derived* view of datasets/REGISTRY.md rather than a hand-edited
    // file; the fixture_hygiene test guards against drift.
    for case in datasets::iter_cases() {
        let mixeff_rs::datasets::Case {
            name,
            meta,
            fit,
            fit_index: _,
        } = case;
        let is_stress = meta.tags.difficulty.as_deref() == Some("stress");

        let (df, _) = match datasets::load(&name) {
            Ok(loaded) => loaded,
            Err(e) => {
                eprintln!("skip {name}: {e}");
                continue;
            }
        };
        let n_obs = df.nrow();
        {
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
                objective_definition: None,
                response_constants: None,
                optimizer: None,
                optimizer_backend: None,
                optimizer_return_code: None,
                optimizer_fevals: None,
                optimizer_fmin: None,
                optimizer_max_fevals: None,
                is_singular: None,
                fit_time_ms: None,
                fit_time_ms_min: None,
                fit_time_ms_repeats: None,
            };

            // Skip stress-tier fits unless explicitly opted in. The maximal RE
            // models on kb07 / mrk17_exp1 / InstEval can take 10+ minutes per
            // fit through the in-crate engine — too slow for routine regen.
            // Set MIXEDMODELS_INCLUDE_STRESS=1 to fit them anyway.
            if is_stress && !include_stress {
                rec.status = "skipped_stress".into();
                rec.error = Some("stress fixture; set MIXEDMODELS_INCLUDE_STRESS=1 to fit".into());
                results.push(rec);
                continue;
            }

            let is_gaussian_identity = fit.family.eq_ignore_ascii_case("Gaussian")
                && fit.link.eq_ignore_ascii_case("Identity");
            let fit_result = if is_gaussian_identity {
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
                fit_lmm(&df, &fit.formula, reml)
            } else {
                let fast_glmm = use_fast_glmm_comparison_path(&fit.family, n_obs);
                let glmm_mode = if fast_glmm { "fast" } else { "joint" };
                print!(
                    "fitting {name} :: {} [{} {}/{} {glmm_mode}] ... ",
                    fit.formula, fit.estimator, fit.family, fit.link
                );
                fit_glmm(
                    &df,
                    &fit.formula,
                    &fit.family,
                    &fit.link,
                    &fit.estimator,
                    fit.weights.as_deref(),
                    fast_glmm,
                )
            };

            match fit_result {
                Ok(FitOutcome::Ok(r)) => {
                    if is_gaussian_identity {
                        println!(
                            "obj={:.4}  σ={:.4}  cold={:.1}ms  min={:.1}ms",
                            r.objective, r.sigma, r.fit_time_ms, r.fit_time_ms_min
                        );
                    } else {
                        println!(
                            "obj={:.4}  disp={:.4}  cold={:.1}ms  min={:.1}ms",
                            r.objective, r.sigma, r.fit_time_ms, r.fit_time_ms_min
                        );
                    }
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
                    if is_gaussian_identity {
                        rec.objective_definition =
                            Some(if fit.estimator.eq_ignore_ascii_case("REML") {
                                "restricted_deviance".into()
                            } else {
                                "deviance".into()
                            });
                        rec.response_constants = Some("not_applicable".into());
                    } else {
                        rec.objective_definition = Some("profiled_glmm_deviance".into());
                        rec.response_constants = Some("dropped".into());
                    }
                    rec.optimizer = Some(r.optimizer);
                    rec.optimizer_backend = Some(r.optimizer_backend);
                    rec.optimizer_return_code = Some(r.optimizer_return_code);
                    rec.optimizer_fevals = Some(r.optimizer_fevals);
                    rec.optimizer_fmin = Some(r.optimizer_fmin);
                    rec.optimizer_max_fevals = Some(r.optimizer_max_fevals);
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
            "tool": "mixeff-rs",
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
