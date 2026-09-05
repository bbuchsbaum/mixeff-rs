//! Structured optimizer benchmark harness.
//!
//! Run twice to compare the release optimizer profile with the CRAN-safe
//! native profile:
//!
//! ```sh
//! cargo run --release --example optimizer_bench_harness
//! cargo run --release --no-default-features --example optimizer_bench_harness
//! ```

use std::time::Instant;

use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::data::DataFrame;
use mixeff_rs::model::linear::{
    FitOptions, LinearMixedModel, OptimizerControl, TrustBqSampleReuse, TrustBqStartLadder,
};
use mixeff_rs::model::traits::MixedModelFit;

const DEFAULT_WARMUP: usize = 1;
const DEFAULT_REPS: usize = 5;
const OBJECTIVE_REL_TOL: f64 = 1e-6;

#[derive(Clone, Copy)]
enum ScenarioKind {
    Sleepstudy {
        n_subjects: usize,
        n_obs: usize,
        seed: u64,
    },
    Crossed {
        n_subjects: usize,
        n_items: usize,
        n_sites: usize,
        n_rep: usize,
    },
    /// GLMM on a dataset from the crate registry (`datasets/<name>`), using
    /// that dataset's `fit_index`-th recorded fit spec (formula, family,
    /// link, weights). `fast` selects profiled PIRLS (the default estimator)
    /// or the joint Laplace fit. Needs `--features unstable-internals` for
    /// the registry; without it these rows are skipped with a note.
    Glmm {
        dataset: &'static str,
        fit_index: usize,
        fast: bool,
    },
}

/// Data plus, for GLMM rows, the fit specification resolved from the
/// dataset registry.
struct ScenarioData {
    data: DataFrame,
    total_n: usize,
    seed: u64,
    /// Formula text actually fitted (the scenario's own for LMM rows).
    formula: String,
    glmm: Option<GlmmSpec>,
}

#[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
struct GlmmSpec {
    family: mixeff_rs::model::traits::Family,
    link: Option<mixeff_rs::model::traits::LinkFunction>,
    weights: Option<Vec<f64>>,
    fast: bool,
}

#[derive(Clone, Copy)]
struct Reference {
    objective_best: f64,
    julia_median_ms: Option<f64>,
    julia_feval: Option<f64>,
}

#[derive(Clone, Copy)]
struct Scenario {
    scenario: &'static str,
    family: &'static str,
    formula: &'static str,
    reml: bool,
    kind: ScenarioKind,
    reference: Reference,
}

struct FitRecord {
    wall_ms: f64,
    /// Formula parse + `LinearMixedModel::new`.
    build_ms: f64,
    /// Optimizer search (initial objective + θ search).
    optimizer_ms: f64,
    /// Post-optimizer finalization (restart, certificate, refreshes).
    postfit_ms: f64,
    objective: f64,
    feval: f64,
    optimizer: String,
    backend: String,
    status: String,
    theta: Vec<f64>,
    fixed_effects: Vec<f64>,
    variance_components: Vec<f64>,
    residual_sd: Option<f64>,
    n: usize,
    p: usize,
    q: usize,
    n_reterms: usize,
    d_theta: usize,
}

fn simulate_sleepstudy_like(n_subjects: usize, n_obs_per_subject: usize, seed: u64) -> DataFrame {
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};

    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();

    let beta = [250.0, 10.0];
    let sigma = 25.0;
    let lambda = [[24.0, 0.0], [1.68, 5.23]];

    let total_n = n_subjects * n_obs_per_subject;
    let mut reaction = Vec::with_capacity(total_n);
    let mut days = Vec::with_capacity(total_n);
    let mut subj_labels = Vec::with_capacity(total_n);

    for i in 0..n_subjects {
        let u0 = normal.sample(&mut rng);
        let u1 = normal.sample(&mut rng);
        let b0 = lambda[0][0] * u0;
        let b1 = lambda[1][0] * u0 + lambda[1][1] * u1;
        let label = format!("S{:04}", i + 1);

        for d in 0..n_obs_per_subject {
            let x = d as f64;
            let mu = beta[0] + beta[1] * x + b0 + b1 * x;
            reaction.push(mu + sigma * normal.sample(&mut rng));
            days.push(x);
            subj_labels.push(label.clone());
        }
    }

    let mut df = DataFrame::new();
    df.add_numeric("reaction", reaction).unwrap();
    df.add_numeric("days", days).unwrap();
    df.add_categorical("subj", subj_labels).unwrap();
    df
}

fn centered_mod(value: usize, modulus: usize, center: f64, scale: f64) -> f64 {
    ((value % modulus) as f64 - center) * scale
}

fn simulate_crossed(n_subjects: usize, n_items: usize, n_sites: usize, n_rep: usize) -> DataFrame {
    let beta = [250.0, 9.5];

    let total_n = n_subjects * n_items * n_rep;
    let mut reaction = Vec::with_capacity(total_n);
    let mut days = Vec::with_capacity(total_n);
    let mut subj_labels = Vec::with_capacity(total_n);
    let mut item_labels = Vec::with_capacity(total_n);
    let mut site_labels = Vec::with_capacity(total_n);

    for s in 0..n_subjects {
        let subj_b0 = centered_mod(7 * s + 3, 19, 9.0, 2.4);
        let subj_b1 = centered_mod(11 * s + 5, 17, 8.0, 0.38) + 0.05 * subj_b0;
        let subj_label = format!("S{:03}", s + 1);

        for i in 0..n_items {
            let item_b0 = centered_mod(13 * i + 2, 23, 11.0, 1.6);
            let item_b1 = centered_mod(5 * i + 7, 19, 9.0, 0.27) - 0.04 * item_b0;
            let item_label = format!("I{:03}", i + 1);

            for r in 0..n_rep {
                let site = (5 * s + 3 * i + r) % n_sites;
                let site_b0 = centered_mod(3 * site + 1, 13, 6.0, 1.2);
                let site_b1 = centered_mod(7 * site + 4, 11, 5.0, 0.18) + 0.03 * site_b0;
                let eps = centered_mod(13 * s + 7 * i + 3 * r + 2 * site, 29, 14.0, 0.9);
                let x = r as f64 + (i % 4) as f64 * 0.35 + (s % 3) as f64 * 0.1;

                let mu = beta[0]
                    + beta[1] * x
                    + subj_b0
                    + subj_b1 * x
                    + item_b0
                    + item_b1 * x
                    + site_b0
                    + site_b1 * x;

                reaction.push(mu + eps);
                days.push(x);
                subj_labels.push(subj_label.clone());
                item_labels.push(item_label.clone());
                site_labels.push(format!("K{:03}", site + 1));
            }
        }
    }

    let mut df = DataFrame::new();
    df.add_numeric("reaction", reaction).unwrap();
    df.add_numeric("days", days).unwrap();
    df.add_categorical("subj", subj_labels).unwrap();
    df.add_categorical("item", item_labels).unwrap();
    df.add_categorical("site", site_labels).unwrap();
    df
}

fn scenarios() -> Vec<Scenario> {
    let vector_formula = "reaction ~ 1 + days + (1 + days | subj)";
    let scalar_formula = "reaction ~ 1 + days + (1 | subj)";
    let crossed_formula =
        "reaction ~ 1 + days + (1 + days | subj) + (1 + days | item) + (1 + days | site)";

    vec![
        Scenario {
            scenario: "vector_180",
            family: "vector",
            formula: vector_formula,
            reml: true,
            kind: ScenarioKind::Sleepstudy {
                n_subjects: 18,
                n_obs: 10,
                seed: 42,
            },
            reference: Reference {
                objective_best: 1714.682305,
                julia_median_ms: None,
                julia_feval: None,
            },
        },
        Scenario {
            scenario: "vector_1000",
            family: "vector",
            formula: vector_formula,
            reml: true,
            kind: ScenarioKind::Sleepstudy {
                n_subjects: 100,
                n_obs: 10,
                seed: 42,
            },
            reference: Reference {
                objective_best: 9688.227799,
                julia_median_ms: None,
                julia_feval: None,
            },
        },
        Scenario {
            scenario: "vector_5000",
            family: "vector",
            formula: vector_formula,
            reml: true,
            kind: ScenarioKind::Sleepstudy {
                n_subjects: 500,
                n_obs: 10,
                seed: 42,
            },
            reference: Reference {
                objective_best: 48470.192830,
                julia_median_ms: None,
                julia_feval: None,
            },
        },
        Scenario {
            scenario: "vector_10000",
            family: "vector",
            formula: vector_formula,
            reml: true,
            kind: ScenarioKind::Sleepstudy {
                n_subjects: 1000,
                n_obs: 10,
                seed: 42,
            },
            reference: Reference {
                objective_best: 97060.632680,
                julia_median_ms: None,
                julia_feval: None,
            },
        },
        Scenario {
            scenario: "vector_deep_200x50",
            family: "vector",
            formula: vector_formula,
            reml: true,
            kind: ScenarioKind::Sleepstudy {
                n_subjects: 200,
                n_obs: 50,
                seed: 42,
            },
            reference: Reference {
                objective_best: 94841.024280,
                julia_median_ms: None,
                julia_feval: None,
            },
        },
        Scenario {
            scenario: "scalar_180",
            family: "scalar",
            formula: scalar_formula,
            reml: true,
            kind: ScenarioKind::Sleepstudy {
                n_subjects: 18,
                n_obs: 10,
                seed: 42,
            },
            reference: Reference {
                objective_best: 1738.321580,
                julia_median_ms: None,
                julia_feval: None,
            },
        },
        Scenario {
            scenario: "scalar_1000",
            family: "scalar",
            formula: scalar_formula,
            reml: true,
            kind: ScenarioKind::Sleepstudy {
                n_subjects: 100,
                n_obs: 10,
                seed: 42,
            },
            reference: Reference {
                objective_best: 9861.362625,
                julia_median_ms: None,
                julia_feval: None,
            },
        },
        Scenario {
            scenario: "scalar_5000",
            family: "scalar",
            formula: scalar_formula,
            reml: true,
            kind: ScenarioKind::Sleepstudy {
                n_subjects: 500,
                n_obs: 10,
                seed: 42,
            },
            reference: Reference {
                objective_best: 49709.499919,
                julia_median_ms: None,
                julia_feval: None,
            },
        },
        Scenario {
            scenario: "scalar_10000",
            family: "scalar",
            formula: scalar_formula,
            reml: true,
            kind: ScenarioKind::Sleepstudy {
                n_subjects: 1000,
                n_obs: 10,
                seed: 42,
            },
            reference: Reference {
                objective_best: 99333.616319,
                julia_median_ms: None,
                julia_feval: None,
            },
        },
        Scenario {
            scenario: "scalar_deep_200x50",
            family: "scalar",
            formula: scalar_formula,
            reml: true,
            kind: ScenarioKind::Sleepstudy {
                n_subjects: 200,
                n_obs: 50,
                seed: 42,
            },
            reference: Reference {
                objective_best: 119792.696382,
                julia_median_ms: None,
                julia_feval: None,
            },
        },
        Scenario {
            scenario: "crossed_small",
            family: "crossed",
            formula: crossed_formula,
            reml: true,
            kind: ScenarioKind::Crossed {
                n_subjects: 18,
                n_items: 12,
                n_sites: 6,
                n_rep: 4,
            },
            reference: Reference {
                objective_best: 6177.391766,
                julia_median_ms: None,
                julia_feval: None,
            },
        },
        Scenario {
            scenario: "crossed_medium",
            family: "crossed",
            formula: crossed_formula,
            reml: true,
            kind: ScenarioKind::Crossed {
                n_subjects: 36,
                n_items: 24,
                n_sites: 8,
                n_rep: 4,
            },
            reference: Reference {
                objective_best: 24348.710133,
                julia_median_ms: None,
                julia_feval: None,
            },
        },
        Scenario {
            scenario: "crossed_large",
            family: "crossed",
            formula: crossed_formula,
            reml: true,
            kind: ScenarioKind::Crossed {
                n_subjects: 72,
                n_items: 36,
                n_sites: 12,
                n_rep: 4,
            },
            reference: Reference {
                objective_best: 72320.124564,
                julia_median_ms: None,
                julia_feval: None,
            },
        },
        // GLMM rows (dataset registry; skipped without `unstable-internals`).
        // `objective_best` is the crate's own converged objective for the
        // row's estimator (fast profiled PIRLS unless `fast: false`), so the
        // gate catches regressions of this estimator, not lme4/Julia
        // objective-definition differences.
        glmm_scenario("glmm_cbpp_fast", "cbpp", 0, true, 100.151883),
        glmm_scenario("glmm_cbpp_full", "cbpp", 0, false, 184.052565),
        glmm_scenario("glmm_grouseticks_fast", "grouseticks", 0, true, 851.404637),
        glmm_scenario("glmm_verbagg_fast", "verbagg", 0, true, 8136.170872),
        glmm_scenario(
            "glmm_contra_intercept_fast",
            "contraception",
            0,
            true,
            2413.662637,
        ),
        glmm_scenario(
            "glmm_contra_slope_fast",
            "contraception",
            1,
            true,
            2399.106780,
        ),
        glmm_scenario("glmm_arabidopsis_fast", "arabidopsis", 0, true, 18486.865146),
    ]
}

fn glmm_scenario(
    scenario: &'static str,
    dataset: &'static str,
    fit_index: usize,
    fast: bool,
    objective_best: f64,
) -> Scenario {
    Scenario {
        scenario,
        family: "glmm",
        formula: "",
        reml: false,
        kind: ScenarioKind::Glmm {
            dataset,
            fit_index,
            fast,
        },
        reference: Reference {
            objective_best,
            julia_median_ms: None,
            julia_feval: None,
        },
    }
}

/// Resolve a scenario's data (and, for GLMM rows, its fit spec). `None`
/// means the row cannot run in this build (registry rows without the
/// `unstable-internals` feature) and is skipped with a note on stderr.
fn build_data(scenario: Scenario) -> Option<ScenarioData> {
    match scenario.kind {
        ScenarioKind::Sleepstudy {
            n_subjects,
            n_obs,
            seed,
        } => Some(ScenarioData {
            data: simulate_sleepstudy_like(n_subjects, n_obs, seed),
            total_n: n_subjects * n_obs,
            seed,
            formula: scenario.formula.to_string(),
            glmm: None,
        }),
        ScenarioKind::Crossed {
            n_subjects,
            n_items,
            n_sites,
            n_rep,
        } => Some(ScenarioData {
            data: simulate_crossed(n_subjects, n_items, n_sites, n_rep),
            total_n: n_subjects * n_items * n_rep,
            seed: 0,
            formula: scenario.formula.to_string(),
            glmm: None,
        }),
        ScenarioKind::Glmm {
            dataset,
            fit_index,
            fast,
        } => build_glmm_data(scenario.scenario, dataset, fit_index, fast),
    }
}

#[cfg(feature = "unstable-internals")]
fn build_glmm_data(
    scenario: &str,
    dataset: &str,
    fit_index: usize,
    fast: bool,
) -> Option<ScenarioData> {
    use mixeff_rs::model::traits::{Family, LinkFunction};

    let (data, meta) = match mixeff_rs::datasets::load(dataset) {
        Ok(loaded) => loaded,
        Err(error) => {
            eprintln!("optimizer_bench_harness: skipping {scenario}: dataset '{dataset}': {error}");
            return None;
        }
    };
    let Some(spec) = meta.fits.get(fit_index) else {
        eprintln!(
            "optimizer_bench_harness: skipping {scenario}: dataset '{dataset}' has no fit #{fit_index}"
        );
        return None;
    };
    let family = match spec.family.to_ascii_lowercase().as_str() {
        "bernoulli" => Family::Bernoulli,
        "binomial" => Family::Binomial,
        "poisson" => Family::Poisson,
        other => {
            eprintln!("optimizer_bench_harness: skipping {scenario}: unsupported family '{other}'");
            return None;
        }
    };
    let link = match spec.link.to_ascii_lowercase().as_str() {
        "" | "default" => None,
        "logit" => Some(LinkFunction::Logit),
        "log" => Some(LinkFunction::Log),
        "probit" => Some(LinkFunction::Probit),
        "cloglog" => Some(LinkFunction::Cloglog),
        other => {
            eprintln!("optimizer_bench_harness: skipping {scenario}: unsupported link '{other}'");
            return None;
        }
    };
    let weights = match &spec.weights {
        Some(column) => match data.numeric(column) {
            Some(values) => Some(values.to_vec()),
            None => {
                eprintln!(
                    "optimizer_bench_harness: skipping {scenario}: weights column '{column}' missing"
                );
                return None;
            }
        },
        None => None,
    };
    // Registry formulas use the lme4 spellings for binomial responses:
    // `events / trials ~ rhs` (grouped, trials become the case weights) and
    // a two-level categorical response for Bernoulli. Lower both into the
    // numeric response the crate's parser expects, as compare_rust does.
    let (data, formula, weights) =
        match prepare_binomial_response(&data, &spec.formula, family, weights) {
            Ok(prepared) => prepared,
            Err(message) => {
                eprintln!("optimizer_bench_harness: skipping {scenario}: {message}");
                return None;
            }
        };
    let total_n = data.nrow();
    Some(ScenarioData {
        data,
        total_n,
        seed: 0,
        formula,
        glmm: Some(GlmmSpec {
            family,
            link,
            weights,
            fast,
        }),
    })
}

#[cfg(feature = "unstable-internals")]
fn prepare_binomial_response(
    df: &DataFrame,
    formula: &str,
    family: mixeff_rs::model::traits::Family,
    weights: Option<Vec<f64>>,
) -> Result<(DataFrame, String, Option<Vec<f64>>), String> {
    use mixeff_rs::model::data::Column;
    use mixeff_rs::model::traits::Family;

    if !matches!(family, Family::Binomial | Family::Bernoulli) {
        return Ok((df.clone(), formula.to_string(), weights));
    }
    let Some((lhs, rhs)) = formula.split_once('~') else {
        return Err(format!("formula '{formula}' has no '~'"));
    };
    let (lhs, rhs) = (lhs.trim(), rhs.trim());

    if let Some((events, trials)) = lhs.split_once('/') {
        let (events, trials) = (events.trim(), trials.trim());
        let events = df
            .numeric(events)
            .ok_or_else(|| format!("events column '{events}' missing"))?;
        let trials_values = df
            .numeric(trials)
            .ok_or_else(|| format!("trials column '{trials}' missing"))?;
        let proportion = events
            .iter()
            .zip(trials_values)
            .map(|(&event, &trial)| event / trial)
            .collect::<Vec<_>>();
        let response_name = format!("__{lhs}")
            .chars()
            .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
            .collect::<String>();
        let mut data = df.clone();
        data.add_numeric(&response_name, proportion)
            .map_err(|error| error.to_string())?;
        return Ok((
            data,
            format!("{response_name} ~ {rhs}"),
            Some(trials_values.to_vec()),
        ));
    }

    match df.column(lhs) {
        Some(Column::Categorical(column)) => {
            if column.levels.len() != 2 {
                return Err(format!(
                    "binary response '{lhs}' has {} levels, expected two",
                    column.levels.len()
                ));
            }
            let response_name = format!("__{lhs}_binary");
            let values = column
                .refs
                .iter()
                .map(|&level| f64::from(level == 1))
                .collect::<Vec<_>>();
            let mut data = df.clone();
            data.add_numeric(&response_name, values)
                .map_err(|error| error.to_string())?;
            Ok((data, format!("{response_name} ~ {rhs}"), weights))
        }
        _ => Ok((df.clone(), formula.to_string(), weights)),
    }
}

#[cfg(not(feature = "unstable-internals"))]
fn build_glmm_data(
    scenario: &str,
    dataset: &str,
    _fit_index: usize,
    _fast: bool,
) -> Option<ScenarioData> {
    eprintln!(
        "optimizer_bench_harness: skipping {scenario} ({dataset}): GLMM rows need --features unstable-internals"
    );
    None
}

fn median(values: &[f64]) -> f64 {
    let mut finite: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    finite.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if finite.is_empty() {
        return f64::NAN;
    }
    let mid = finite.len() / 2;
    if finite.len() % 2 == 0 {
        (finite[mid - 1] + finite[mid]) / 2.0
    } else {
        finite[mid]
    }
}

fn mean(values: &[f64]) -> f64 {
    let finite: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if finite.is_empty() {
        return f64::NAN;
    }
    finite.iter().sum::<f64>() / finite.len() as f64
}

fn min_finite(values: &[f64]) -> f64 {
    values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(f64::INFINITY, f64::min)
}

/// Julia MixedModels.jl reference timings keyed by harness scenario name,
/// loaded from `benchmarks/julia_reference.json` (generated by
/// `scripts/bench_julia.jl`; see the file's `provenance` block for host,
/// Julia and MixedModels versions). Missing file or scenario leaves the
/// Julia columns empty rather than failing the run.
fn load_julia_reference() -> std::collections::HashMap<String, (Option<f64>, Option<f64>)> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/benchmarks/julia_reference.json"
    );
    let Ok(text) = std::fs::read_to_string(path) else {
        eprintln!("optimizer_bench_harness: no Julia reference file at {path}");
        return Default::default();
    };
    let Ok(doc) = serde_json::from_str::<serde_json::Value>(&text) else {
        eprintln!("optimizer_bench_harness: could not parse {path}");
        return Default::default();
    };
    if let Some(provenance) = doc.get("provenance") {
        eprintln!(
            "optimizer_bench_harness: Julia reference {} on {} (Julia {}, MixedModels {})",
            provenance["generated"].as_str().unwrap_or("?"),
            provenance["host"].as_str().unwrap_or("?"),
            provenance["julia_version"].as_str().unwrap_or("?"),
            provenance["mixedmodels_version"].as_str().unwrap_or("?"),
        );
    }
    doc.get("scenarios")
        .and_then(|value| value.as_object())
        .map(|scenarios| {
            scenarios
                .iter()
                .map(|(name, entry)| {
                    (
                        name.clone(),
                        (
                            entry.get("median_ms").and_then(|v| v.as_f64()),
                            entry.get("feval").and_then(|v| v.as_f64()),
                        ),
                    )
                })
                .collect()
        })
        .unwrap_or_default()
}

fn compile_profile() -> &'static str {
    if cfg!(feature = "nlopt") {
        "rust_default_nlopt"
    } else {
        "rust_no_default_native"
    }
}

/// Opt-in TrustBQ start-ladder experiment lever for A/B runs:
/// `MIXEFF_BENCH_TRUST_BQ_START_LADDER=diagonal_first`. Unset (the default)
/// keeps the driver's standard single-start behavior.
fn bench_start_ladder() -> TrustBqStartLadder {
    match std::env::var("MIXEFF_BENCH_TRUST_BQ_START_LADDER").as_deref() {
        Ok("diagonal_first") => TrustBqStartLadder::DiagonalFirst,
        _ => TrustBqStartLadder::Off,
    }
}

/// Opt-in TrustBQ exact-sample reuse experiment lever for A/B runs:
/// `MIXEFF_BENCH_TRUST_BQ_SAMPLE_REUSE=all` broadens reuse to every family,
/// `disabled` turns it off for every family, and unset keeps the production
/// family policy.
fn bench_sample_reuse() -> TrustBqSampleReuse {
    match std::env::var("MIXEFF_BENCH_TRUST_BQ_SAMPLE_REUSE").as_deref() {
        Ok("all") | Ok("all_families") => TrustBqSampleReuse::AllFamilies,
        Ok("disabled") | Ok("off") => TrustBqSampleReuse::Disabled,
        _ => TrustBqSampleReuse::FamilyPolicy,
    }
}

fn fit_with_bench_controls(
    model: &mut LinearMixedModel,
    reml: bool,
) -> mixeff_rs::error::Result<()> {
    let ladder = bench_start_ladder();
    let sample_reuse = bench_sample_reuse();
    if ladder == TrustBqStartLadder::Off && sample_reuse == TrustBqSampleReuse::FamilyPolicy {
        model.fit(reml)?;
        return Ok(());
    }
    let options = if reml {
        FitOptions::reml()
    } else {
        FitOptions::ml()
    }
    .with_optimizer_control(
        OptimizerControl::auto()
            .with_trust_bq_start_ladder(ladder)
            .with_trust_bq_sample_reuse(sample_reuse),
    );
    model.fit_with_options(options)?;
    Ok(())
}

fn fit_once(scenario: Scenario, scenario_data: &ScenarioData) -> Result<FitRecord, String> {
    match &scenario_data.glmm {
        Some(spec) => fit_glmm_once(&scenario_data.formula, &scenario_data.data, spec),
        None => fit_lmm_once(scenario, &scenario_data.data),
    }
}

#[cfg(feature = "unstable-internals")]
fn fit_glmm_once(formula: &str, data: &DataFrame, spec: &GlmmSpec) -> Result<FitRecord, String> {
    use mixeff_rs::model::generalized::GeneralizedLinearMixedModel;

    let start = Instant::now();
    let formula = parse_formula(formula).map_err(|err| err.to_string())?;
    let mut model = match &spec.weights {
        Some(weights) => GeneralizedLinearMixedModel::new_with_weights(
            formula,
            data,
            spec.family,
            spec.link,
            weights.clone(),
        ),
        None => GeneralizedLinearMixedModel::new(formula, data, spec.family, spec.link),
    }
    .map_err(|err| err.to_string())?;
    let build_ms = start.elapsed().as_secs_f64() * 1000.0;
    model
        .fit_with_options(spec.fast, 1, false)
        .map_err(|err| err.to_string())?;
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
    let lmm = model.lmm();
    let (n, p, q, n_reterms) = lmm.model_size();
    let theta = model.theta();
    let variance_components = theta.iter().map(|t| t * t).collect();

    Ok(FitRecord {
        wall_ms,
        build_ms,
        optimizer_ms: f64::NAN,
        postfit_ms: f64::NAN,
        objective: model.objective(),
        feval: lmm.optsum().feval as f64,
        optimizer: lmm.optsum().optimizer_name().to_string(),
        backend: lmm.optsum().backend_name().to_string(),
        status: lmm.optsum().return_value.clone(),
        d_theta: theta.len(),
        theta,
        fixed_effects: model.coef().as_slice().to_vec(),
        variance_components,
        residual_sd: None,
        n,
        p,
        q,
        n_reterms,
    })
}

#[cfg(not(feature = "unstable-internals"))]
fn fit_glmm_once(_formula: &str, _data: &DataFrame, _spec: &GlmmSpec) -> Result<FitRecord, String> {
    Err("GLMM rows need --features unstable-internals".to_string())
}

fn fit_lmm_once(scenario: Scenario, data: &DataFrame) -> Result<FitRecord, String> {
    let start = Instant::now();
    let formula = parse_formula(scenario.formula).map_err(|err| err.to_string())?;
    let mut model = LinearMixedModel::new(formula, data, None).map_err(|err| err.to_string())?;
    let build_ms = start.elapsed().as_secs_f64() * 1000.0;
    fit_with_bench_controls(&mut model, scenario.reml).map_err(|err| err.to_string())?;
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
    let (optimizer_ms, postfit_ms) = model
        .fit_phase_timings()
        .map(|timings| {
            (
                timings.optimizer.as_secs_f64() * 1000.0,
                timings.post_fit.as_secs_f64() * 1000.0,
            )
        })
        .unwrap_or((f64::NAN, f64::NAN));
    let (n, p, q, n_reterms) = model.model_size();
    let varcorr = model.varcorr();
    let variance_components = varcorr
        .components
        .iter()
        .flat_map(|component| component.std_dev.iter().map(|sd| sd * sd))
        .collect();

    Ok(FitRecord {
        wall_ms,
        build_ms,
        optimizer_ms,
        postfit_ms,
        objective: model.objective(),
        feval: model.optsum().feval as f64,
        optimizer: model.optsum().optimizer_name().to_string(),
        backend: model.optsum().backend_name().to_string(),
        status: model.optsum().return_value.clone(),
        theta: model.theta(),
        fixed_effects: model.coef().as_slice().to_vec(),
        variance_components,
        residual_sd: varcorr.residual_sd,
        n,
        p,
        q,
        n_reterms,
        d_theta: model.n_theta(),
    })
}

fn csv_field(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn json_numbers(values: &[f64]) -> String {
    serde_json::to_string(values).expect("numeric vectors should serialize")
}

fn option_number(value: Option<f64>) -> String {
    value.map_or_else(String::new, |value| format!("{value:.6}"))
}

fn print_row(fields: Vec<String>) {
    println!("{}", fields.join(","));
}

fn main() {
    println!(
        "method,scenario,family,formula,reml,seed,n,p,q,n_reterms,d_theta,total_n,reps,optimizer,backend,status,median_ms,mean_ms,min_ms,build_ms_median,optimizer_ms_median,postfit_ms_median,objective,reference_objective,objective_gap,objective_tolerance,objective_pass,feval_median,feval_mean,julia_median_ms,julia_feval,julia_speedup,theta,fixed_effects,variance_components,residual_sd"
    );

    let julia_reference = load_julia_reference();
    for mut scenario in scenarios() {
        if let Some((median_ms, feval)) = julia_reference.get(scenario.scenario) {
            scenario.reference.julia_median_ms = *median_ms;
            scenario.reference.julia_feval = *feval;
        }
        let Some(scenario_data) = build_data(scenario) else {
            continue;
        };
        let total_n = scenario_data.total_n;
        let seed = scenario_data.seed;

        for _ in 0..DEFAULT_WARMUP {
            let _ = fit_once(scenario, &scenario_data);
        }

        let mut records = Vec::with_capacity(DEFAULT_REPS);
        let mut errors = Vec::new();
        for _ in 0..DEFAULT_REPS {
            match fit_once(scenario, &scenario_data) {
                Ok(record) => records.push(record),
                Err(err) => errors.push(err),
            }
        }

        if records.is_empty() {
            let status = errors
                .first()
                .cloned()
                .unwrap_or_else(|| "fit failed".to_string());
            print_row(vec![
                compile_profile().to_string(),
                scenario.scenario.to_string(),
                scenario.family.to_string(),
                csv_field(&scenario_data.formula),
                scenario.reml.to_string(),
                seed.to_string(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                total_n.to_string(),
                DEFAULT_REPS.to_string(),
                String::new(),
                String::new(),
                csv_field(&status),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                format!("{:.6}", scenario.reference.objective_best),
                String::new(),
                String::new(),
                "false".to_string(),
                String::new(),
                String::new(),
                option_number(scenario.reference.julia_median_ms),
                option_number(scenario.reference.julia_feval),
                String::new(),
                "[]".to_string(),
                "[]".to_string(),
                "[]".to_string(),
                String::new(),
            ]);
            continue;
        }

        let wall_ms: Vec<f64> = records.iter().map(|record| record.wall_ms).collect();
        let objectives: Vec<f64> = records.iter().map(|record| record.objective).collect();
        let fevals: Vec<f64> = records.iter().map(|record| record.feval).collect();
        let build_ms: Vec<f64> = records.iter().map(|record| record.build_ms).collect();
        let optimizer_ms: Vec<f64> = records.iter().map(|record| record.optimizer_ms).collect();
        let postfit_ms: Vec<f64> = records.iter().map(|record| record.postfit_ms).collect();
        let representative = &records[0];
        let objective = mean(&objectives);
        let objective_gap = objective - scenario.reference.objective_best;
        let objective_tolerance =
            OBJECTIVE_REL_TOL * (1.0 + scenario.reference.objective_best.abs());
        let objective_pass = objective_gap <= objective_tolerance;
        let median_ms = median(&wall_ms);
        let julia_speedup = scenario
            .reference
            .julia_median_ms
            .map(|julia_ms| julia_ms / median_ms);

        print_row(vec![
            compile_profile().to_string(),
            scenario.scenario.to_string(),
            scenario.family.to_string(),
            csv_field(scenario.formula),
            scenario.reml.to_string(),
            seed.to_string(),
            representative.n.to_string(),
            representative.p.to_string(),
            representative.q.to_string(),
            representative.n_reterms.to_string(),
            representative.d_theta.to_string(),
            total_n.to_string(),
            DEFAULT_REPS.to_string(),
            representative.optimizer.clone(),
            representative.backend.clone(),
            csv_field(&representative.status),
            format!("{median_ms:.6}"),
            format!("{:.6}", mean(&wall_ms)),
            format!("{:.6}", min_finite(&wall_ms)),
            format!("{:.6}", median(&build_ms)),
            format!("{:.6}", median(&optimizer_ms)),
            format!("{:.6}", median(&postfit_ms)),
            format!("{objective:.9}"),
            format!("{:.9}", scenario.reference.objective_best),
            format!("{objective_gap:.9}"),
            format!("{objective_tolerance:.9}"),
            objective_pass.to_string(),
            format!("{:.1}", median(&fevals)),
            format!("{:.1}", mean(&fevals)),
            option_number(scenario.reference.julia_median_ms),
            option_number(scenario.reference.julia_feval),
            option_number(julia_speedup),
            csv_field(&json_numbers(&representative.theta)),
            csv_field(&json_numbers(&representative.fixed_effects)),
            csv_field(&json_numbers(&representative.variance_components)),
            option_number(representative.residual_sd),
        ]);
    }
}
