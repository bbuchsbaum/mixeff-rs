//! Asymptotic benchmark — simulate sleepstudy-shaped data at increasing n
//! and time the Rust fit pipeline. The companion R script
//! (`scripts/bench_asymptotic.R`) reads the same `data.csv` files and times
//! `lmer()`. The reporter binary `bench_asymptotic_report.rs` merges both
//! sides into a markdown table.
//!
//! Run:
//!     cargo run --release --example bench_asymptotic
//!     # then:
//!     Rscript scripts/bench_asymptotic.R
//!     cargo run --release --example bench_asymptotic_report
//!
//! By default we benchmark n ∈ {1k, 5k, 20k}. Set `BENCH_ASYMPTOTIC_XL=1`
//! to also include the n=100k case (slow on R, ~minutes).

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use serde::Serialize;
use serde_json::json;

use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::data::DataFrame;
use mixeff_rs::model::linear::LinearMixedModel;
use mixeff_rs::model::traits::MixedModelFit;

const FORMULA: &str = "reaction ~ 1 + days + (1 + days | subj)";
const WARMUP_RUNS: usize = 3;
const MEASURED_RUNS: usize = 5;

#[derive(Serialize)]
struct ScenarioResult {
    label: String,
    n_subjects: usize,
    n_obs_per_subject: usize,
    n_obs: usize,
    formula: String,
    fit_time_ms_min: f64,
    fit_time_ms_median: f64,
    parse_build_ms_median: f64,
    fit_only_ms_median: f64,
    fevals: i64,
    optimizer: String,
    objective: f64,
    sigma: f64,
}

fn comparison_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("comparison")
}

fn simulate(n_subjects: usize, n_obs_per_subject: usize, seed: u64) -> DataFrame {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Sleepstudy-shaped truth: sigma_int = 24, sigma_slope = 6, rho = 0.07,
    // residual = 25, beta = (250, 10).
    let beta = [250.0, 10.0];
    let sigma_resid = 25.0;
    // Cholesky of [[24^2, ρ·24·6], [ρ·24·6, 6^2]]:
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
    df.add_numeric("reaction", reaction);
    df.add_numeric("days", days);
    df.add_categorical("subj", subj_labels);
    df
}

fn write_csv(df: &DataFrame, path: &PathBuf) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "\"reaction\",\"days\",\"subj\"")?;
    let reaction = df.numeric("reaction").unwrap();
    let days = df.numeric("days").unwrap();
    let subj = df.categorical("subj").unwrap();
    for i in 0..df.nrow() {
        writeln!(f, "{},{},\"{}\"", reaction[i], days[i], subj.values[i])?;
    }
    Ok(())
}

fn percentile(samples: &mut [f64], q: f64) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((samples.len() - 1) as f64 * q).round() as usize;
    samples[idx]
}

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn run_scenario(
    label: &str,
    n_subjects: usize,
    n_obs_per: usize,
) -> Result<ScenarioResult, Box<dyn std::error::Error>> {
    println!(
        "\n=== scenario {label}: n_subjects={n_subjects}, n_obs={}",
        n_subjects * n_obs_per
    );

    let df = simulate(n_subjects, n_obs_per, 42);
    let scenario_dir = comparison_root().join("asymptotic").join(label);
    fs::create_dir_all(&scenario_dir)?;
    write_csv(&df, &scenario_dir.join("data.csv"))?;
    println!(
        "  wrote {} rows to {}",
        df.nrow(),
        scenario_dir.join("data.csv").display()
    );

    // Warmup
    for _ in 0..WARMUP_RUNS {
        let f = parse_formula(FORMULA)?;
        let mut m = LinearMixedModel::new(f, &df, None)?;
        m.fit(true)?;
    }

    let mut totals = Vec::with_capacity(MEASURED_RUNS);
    let mut parse_builds = Vec::with_capacity(MEASURED_RUNS);
    let mut fits = Vec::with_capacity(MEASURED_RUNS);
    let mut last_fevals = -1i64;
    let mut last_optimizer = String::new();
    let mut last_objective = 0.0;
    let mut last_sigma = 0.0;

    for k in 0..MEASURED_RUNS {
        let t0 = Instant::now();
        let formula = parse_formula(FORMULA)?;
        let mut model = LinearMixedModel::new(formula, &df, None)?;
        let pb = ms(t0.elapsed());
        let tf = Instant::now();
        model.fit(true)?;
        let fit_only = ms(tf.elapsed());
        let total = ms(t0.elapsed());
        totals.push(total);
        parse_builds.push(pb);
        fits.push(fit_only);
        last_fevals = model.opt_summary().feval;
        last_optimizer = model.opt_summary().optimizer_name().to_string();
        last_objective = model.opt_summary().fmin;
        last_sigma = model.sigma();
        println!(
            "  run {}: total={:.1} ms (parse+build={:.2}, fit={:.1}, fevals={})",
            k + 1,
            total,
            pb,
            fit_only,
            last_fevals
        );
    }

    let mut t_sorted = totals.clone();
    let med_total = percentile(&mut t_sorted, 0.5);
    let min_total = percentile(&mut t_sorted, 0.0);
    let med_pb = percentile(&mut parse_builds.clone(), 0.5);
    let med_fit = percentile(&mut fits.clone(), 0.5);

    Ok(ScenarioResult {
        label: label.to_string(),
        n_subjects,
        n_obs_per_subject: n_obs_per,
        n_obs: n_subjects * n_obs_per,
        formula: FORMULA.to_string(),
        fit_time_ms_min: min_total,
        fit_time_ms_median: med_total,
        parse_build_ms_median: med_pb,
        fit_only_ms_median: med_fit,
        fevals: last_fevals,
        optimizer: last_optimizer,
        objective: last_objective,
        sigma: last_sigma,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut scenarios: Vec<(&str, usize, usize)> = vec![
        ("s", 100, 10),  //  1,000 rows
        ("m", 500, 10),  //  5,000 rows
        ("l", 2000, 10), // 20,000 rows
    ];
    if std::env::var("BENCH_ASYMPTOTIC_XL").is_ok() {
        scenarios.push(("xl", 10_000, 10)); // 100,000 rows
    }

    let mut results = Vec::with_capacity(scenarios.len());
    for (label, ns, no) in &scenarios {
        results.push(run_scenario(label, *ns, *no)?);
    }

    let outpath = comparison_root()
        .join("asymptotic")
        .join("rust_results.json");
    fs::create_dir_all(outpath.parent().unwrap())?;
    fs::write(
        &outpath,
        serde_json::to_string_pretty(&json!({
            "tool": "mixeff-rs",
            "version": env!("CARGO_PKG_VERSION"),
            "results": results,
        }))?,
    )?;
    println!("\nwrote {}", outpath.display());
    Ok(())
}
