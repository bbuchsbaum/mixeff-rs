//! Tune BOBYQA controls for the 100k sleepstudy-shaped random-slope benchmark.
//!
//! Run:
//!     cargo run --release --example tune_bobyqa_asymptotic

use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::data::DataFrame;
use mixeff_rs::model::linear::LinearMixedModel;
use mixeff_rs::model::traits::MixedModelFit;

const FORMULA: &str = "reaction ~ 1 + days + (1 + days | subj)";
const DEFAULT_N_SUBJECTS: usize = 10_000;
const DEFAULT_N_OBS_PER_SUBJECT: usize = 10;
const DEFAULT_MEASURED_RUNS: usize = 3;

#[derive(Clone, Copy)]
struct Variant {
    name: &'static str,
    ftol_rel: Option<f64>,
    ftol_abs: Option<f64>,
    xtol_abs: Option<f64>,
    initial_step: Option<f64>,
}

#[derive(Debug)]
struct VariantResult {
    name: &'static str,
    median_fit_ms: f64,
    min_fit_ms: f64,
    fevals: i64,
    objective: f64,
    theta: Vec<f64>,
    return_value: String,
}

fn simulate(n_subjects: usize, n_obs_per_subject: usize, seed: u64) -> DataFrame {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();

    let beta = [250.0, 10.0];
    let sigma_resid = 25.0;
    let l11 = 24.0;
    let l21 = 0.07 * 6.0;
    let l22 = (6.0_f64.powi(2) - l21 * l21).sqrt();

    let total_n = n_subjects * n_obs_per_subject;
    let mut reaction = Vec::with_capacity(total_n);
    let mut days = Vec::with_capacity(total_n);
    let mut subj_labels = Vec::with_capacity(total_n);

    for i in 0..n_subjects {
        let u0: f64 = normal.sample(&mut rng);
        let u1: f64 = normal.sample(&mut rng);
        let b0 = l11 * u0;
        let b1 = l21 * u0 + l22 * u1;
        let label = format!("S{:06}", i + 1);
        for d in 0..n_obs_per_subject {
            let x = d as f64;
            let mu = beta[0] + beta[1] * x + b0 + b1 * x;
            let y = mu + sigma_resid * normal.sample(&mut rng);
            reaction.push(y);
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

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn percentile(samples: &mut [f64], q: f64) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((samples.len() - 1) as f64 * q).round() as usize;
    samples[idx]
}

fn configure(model: &mut LinearMixedModel, variant: Variant) {
    let n_theta = model.n_theta();
    if let Some(ftol_rel) = variant.ftol_rel {
        model.optsum_mut().ftol_rel = ftol_rel;
    }
    if let Some(ftol_abs) = variant.ftol_abs {
        model.optsum_mut().ftol_abs = ftol_abs;
    }
    if let Some(xtol_abs) = variant.xtol_abs {
        model.optsum_mut().xtol_abs = vec![xtol_abs; n_theta];
    }
    if let Some(initial_step) = variant.initial_step {
        model.optsum_mut().initial_step = vec![initial_step; n_theta];
    }
}

fn run_variant(
    df: &DataFrame,
    formula: &mixeff_rs::formula::Formula,
    variant: Variant,
    measured_runs: usize,
) -> VariantResult {
    let mut fit_times = Vec::with_capacity(measured_runs);
    let mut last_fevals = -1;
    let mut last_objective = f64::INFINITY;
    let mut last_theta = Vec::new();
    let mut last_return = String::new();

    for _ in 0..measured_runs {
        let mut model = LinearMixedModel::new(formula.clone(), df, None).unwrap();
        configure(&mut model, variant);
        let t0 = Instant::now();
        model.fit(true).unwrap();
        fit_times.push(ms(t0.elapsed()));
        last_fevals = model.opt_summary().feval;
        last_objective = model.opt_summary().fmin;
        last_theta = model.theta();
        last_return = model.opt_summary().return_value.clone();
    }

    let mut sorted = fit_times.clone();
    let median_fit_ms = percentile(&mut sorted, 0.5);
    let min_fit_ms = percentile(&mut fit_times, 0.0);

    VariantResult {
        name: variant.name,
        median_fit_ms,
        min_fit_ms,
        fevals: last_fevals,
        objective: last_objective,
        theta: last_theta,
        return_value: last_return,
    }
}

fn max_abs_delta(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

fn main() {
    let variants = [
        Variant {
            name: "default",
            ftol_rel: None,
            ftol_abs: None,
            xtol_abs: None,
            initial_step: None,
        },
        Variant {
            name: "ftol_rel=1e-10",
            ftol_rel: Some(1e-10),
            ftol_abs: None,
            xtol_abs: None,
            initial_step: None,
        },
        Variant {
            name: "ftol_rel=1e-9",
            ftol_rel: Some(1e-9),
            ftol_abs: None,
            xtol_abs: None,
            initial_step: None,
        },
        Variant {
            name: "ftol_rel=1e-7",
            ftol_rel: Some(1e-7),
            ftol_abs: None,
            xtol_abs: None,
            initial_step: None,
        },
        Variant {
            name: "ftol_abs=1e-6",
            ftol_rel: None,
            ftol_abs: Some(1e-6),
            xtol_abs: None,
            initial_step: None,
        },
        Variant {
            name: "xtol_abs=1e-8",
            ftol_rel: None,
            ftol_abs: None,
            xtol_abs: Some(1e-8),
            initial_step: None,
        },
        Variant {
            name: "xtol_abs=1e-7",
            ftol_rel: None,
            ftol_abs: None,
            xtol_abs: Some(1e-7),
            initial_step: None,
        },
        Variant {
            name: "fr=1e-10 xt=1e-7",
            ftol_rel: Some(1e-10),
            ftol_abs: None,
            xtol_abs: Some(1e-7),
            initial_step: None,
        },
        Variant {
            name: "fr=1e-9 xt=1e-7",
            ftol_rel: Some(1e-9),
            ftol_abs: None,
            xtol_abs: Some(1e-7),
            initial_step: None,
        },
        Variant {
            name: "step=0.25",
            ftol_rel: None,
            ftol_abs: None,
            xtol_abs: None,
            initial_step: Some(0.25),
        },
        Variant {
            name: "step=0.5",
            ftol_rel: None,
            ftol_abs: None,
            xtol_abs: None,
            initial_step: Some(0.5),
        },
        Variant {
            name: "step=1.0",
            ftol_rel: None,
            ftol_abs: None,
            xtol_abs: None,
            initial_step: Some(1.0),
        },
    ];

    let n_subjects = std::env::var("TUNE_SUBJECTS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_N_SUBJECTS);
    let n_obs_per_subject = std::env::var("TUNE_OBS_PER_SUBJECT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_N_OBS_PER_SUBJECT);
    let measured_runs = std::env::var("TUNE_RUNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_MEASURED_RUNS);

    println!(
        "simulating {} rows, formula: {}",
        n_subjects * n_obs_per_subject,
        FORMULA
    );
    let df = simulate(n_subjects, n_obs_per_subject, 42);
    let formula = parse_formula(FORMULA).unwrap();

    let mut results = Vec::with_capacity(variants.len());
    for variant in variants {
        let result = run_variant(&df, &formula, variant, measured_runs);
        println!(
            "{:<16} median_fit={:>7.2} ms min_fit={:>7.2} ms fevals={:>3} obj={:.6} status={}",
            result.name,
            result.median_fit_ms,
            result.min_fit_ms,
            result.fevals,
            result.objective,
            result.return_value
        );
        results.push(result);
    }

    let default = results
        .iter()
        .find(|result| result.name == "default")
        .expect("default variant");
    println!("\nrelative to default:");
    for result in &results {
        println!(
            "{:<16} Δobj={:>12.6e} max|Δθ|={:>12.6e} feval_delta={:>4}",
            result.name,
            result.objective - default.objective,
            max_abs_delta(&result.theta, &default.theta),
            result.fevals - default.fevals
        );
    }
}
