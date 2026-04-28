//! Penicillin-style convergence stress benchmark.
//!
//! Run:
//!     cargo run --release --example bench_penicillin_convergence
//!
//! The benchmark resamples the lme4 Penicillin fixture into larger crossed
//! random-intercept grids and records optimizer stop evidence, derivative
//! certificate summaries, theta stability, and runtime.
//!
//! Optional environment controls:
//! - `PENICILLIN_BENCH_WARMUP=2`
//! - `PENICILLIN_BENCH_RUNS=5`
//! - `PENICILLIN_BENCH_VERIFY=0`
//! - `PENICILLIN_BENCH_XL=1`
//! - `PENICILLIN_BENCH_RESPONSE_SCALE=10.0`
//! - `PENICILLIN_BENCH_OUTDIR=/tmp/penicillin_convergence`

use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use serde::Serialize;
use serde_json::json;

use mixedmodels::compiler::{
    ConvergenceVerificationStatus, EvidenceMethod, EvidenceQuality, FitStatus,
};
use mixedmodels::datasets;
use mixedmodels::formula::parse_formula;
use mixedmodels::model::linear::{ConvergenceVerificationOptions, LinearMixedModel};
use mixedmodels::model::traits::MixedModelFit;
use mixedmodels::model::DataFrame;

const FORMULA: &str = "diameter ~ 1 + (1 | plate) + (1 | sample)";

#[derive(Debug, Clone)]
struct PenicillinGrid {
    values: Vec<Vec<f64>>,
    mean: f64,
    n_plate: usize,
    n_sample: usize,
}

#[derive(Debug, Clone)]
struct Scenario {
    label: &'static str,
    n_plate: usize,
    n_sample: usize,
    repeats_per_cell: usize,
    seed: u64,
}

#[derive(Debug, Serialize)]
struct RunResult {
    run_index: usize,
    fit_time_ms: f64,
    verification_time_ms: Option<f64>,
    objective: f64,
    sigma: f64,
    theta: Vec<f64>,
    optimizer: String,
    optimizer_stop: Option<String>,
    acceptable_stop: bool,
    budget_exhausted: bool,
    fevals: Option<usize>,
    fit_status: String,
    raw_gradient_norm: Option<f64>,
    scaled_gradient_norm: Option<f64>,
    hybrid_gradient_norm: Option<f64>,
    hessian_method: String,
    hessian_quality: String,
    hessian_min_eigenvalue: Option<f64>,
    hessian_condition_number: Option<f64>,
    hessian_rank: Option<usize>,
    verification_status: Option<String>,
    verification_runs: usize,
    verification_agreeing_runs: usize,
    verification_max_abs_theta_delta: Option<f64>,
}

#[derive(Debug, Serialize)]
struct ScenarioResult {
    label: String,
    n_plate: usize,
    n_sample: usize,
    repeats_per_cell: usize,
    response_scale: f64,
    n_obs: usize,
    formula: String,
    reml: bool,
    warmup_runs: usize,
    measured_runs: usize,
    fit_time_ms_min: f64,
    fit_time_ms_median: f64,
    verification_time_ms_median: Option<f64>,
    raw_gradient_norm_median: Option<f64>,
    scaled_gradient_norm_median: Option<f64>,
    hybrid_gradient_norm_median: Option<f64>,
    hessian_min_eigenvalue_min: Option<f64>,
    hessian_condition_number_max: Option<f64>,
    theta_repeatability_max_abs: f64,
    theta_verification_max_abs: Option<f64>,
    optimizer_stop_last: Option<String>,
    fit_status_last: String,
    runs: Vec<RunResult>,
}

fn comparison_root() -> PathBuf {
    if let Ok(path) = std::env::var("PENICILLIN_BENCH_OUTDIR") {
        return PathBuf::from(path);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("comparison")
        .join("penicillin_convergence")
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_bool(name: &str, default: bool) -> bool {
    match std::env::var(name).ok().as_deref() {
        Some("0") | Some("false") | Some("FALSE") | Some("no") | Some("NO") => false,
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES") => true,
        _ => default,
    }
}

fn load_penicillin_grid() -> Result<PenicillinGrid, Box<dyn std::error::Error>> {
    let (df, _) = datasets::load("penicillin")?;
    let diameter = df.numeric("diameter").expect("diameter column");
    let plate = df.categorical("plate").expect("plate column");
    let sample = df.categorical("sample").expect("sample column");
    let n_plate = plate.n_levels();
    let n_sample = sample.n_levels();

    let mut values = vec![vec![f64::NAN; n_sample]; n_plate];
    for row in 0..df.nrow() {
        let p = plate.refs[row] as usize;
        let s = sample.refs[row] as usize;
        values[p][s] = diameter[row];
    }

    let mean = diameter.iter().sum::<f64>() / diameter.len() as f64;
    Ok(PenicillinGrid {
        values,
        mean,
        n_plate,
        n_sample,
    })
}

fn simulate_penicillin_like(
    grid: &PenicillinGrid,
    scenario: &Scenario,
    response_scale: f64,
) -> DataFrame {
    let mut rng = StdRng::seed_from_u64(scenario.seed);
    let plate_offset_dist = Normal::new(0.0, 0.25 * response_scale.abs().max(1.0)).unwrap();
    let sample_offset_dist = Normal::new(0.0, 0.45 * response_scale.abs().max(1.0)).unwrap();
    let residual_dist = Normal::new(0.0, 0.15 * response_scale.abs().max(1.0)).unwrap();

    let plate_offsets: Vec<f64> = (0..scenario.n_plate)
        .map(|_| plate_offset_dist.sample(&mut rng))
        .collect();
    let sample_offsets: Vec<f64> = (0..scenario.n_sample)
        .map(|_| sample_offset_dist.sample(&mut rng))
        .collect();

    let n_obs = scenario.n_plate * scenario.n_sample * scenario.repeats_per_cell;
    let mut diameter = Vec::with_capacity(n_obs);
    let mut plate = Vec::with_capacity(n_obs);
    let mut sample = Vec::with_capacity(n_obs);

    for p in 0..scenario.n_plate {
        let p_label = format!("P{:04}", p + 1);
        for s in 0..scenario.n_sample {
            let s_label = format!("S{:04}", s + 1);
            let base = grid.values[p % grid.n_plate][s % grid.n_sample];
            let centered = grid.mean + (base - grid.mean) * response_scale;
            for _ in 0..scenario.repeats_per_cell {
                diameter.push(
                    centered
                        + plate_offsets[p]
                        + sample_offsets[s]
                        + residual_dist.sample(&mut rng),
                );
                plate.push(p_label.clone());
                sample.push(s_label.clone());
            }
        }
    }

    let plate_levels = (0..scenario.n_plate)
        .map(|p| format!("P{:04}", p + 1))
        .collect();
    let sample_levels = (0..scenario.n_sample)
        .map(|s| format!("S{:04}", s + 1))
        .collect();

    let mut df = DataFrame::new();
    df.add_numeric("diameter", diameter);
    df.add_categorical_with_levels("plate", plate, plate_levels);
    df.add_categorical_with_levels("sample", sample, sample_levels);
    df
}

fn verification_options() -> ConvergenceVerificationOptions {
    ConvergenceVerificationOptions {
        restart_from_optimum: true,
        jitter_starts: 1,
        jitter_scale: 1e-4,
        run_optimizer_consensus: true,
        max_function_evaluations: 300,
        objective_tolerance: 1e-5,
        theta_tolerance: 1e-3,
        beta_tolerance: 1e-4,
    }
}

fn run_scenario(
    grid: &PenicillinGrid,
    scenario: &Scenario,
    response_scale: f64,
    warmup_runs: usize,
    measured_runs: usize,
    verify: bool,
) -> Result<ScenarioResult, Box<dyn std::error::Error>> {
    let data = simulate_penicillin_like(grid, scenario, response_scale);
    let formula = parse_formula(FORMULA)?;

    println!(
        "\n=== {}: plates={}, samples={}, repeats={}, n={} ===",
        scenario.label,
        scenario.n_plate,
        scenario.n_sample,
        scenario.repeats_per_cell,
        data.nrow()
    );

    for _ in 0..warmup_runs {
        let mut model = LinearMixedModel::new(formula.clone(), &data, None)?;
        model.fit(true)?;
    }

    let mut runs = Vec::with_capacity(measured_runs);
    for run_index in 1..=measured_runs {
        let mut model = LinearMixedModel::new(formula.clone(), &data, None)?;

        let fit_start = Instant::now();
        model.fit(true)?;
        let fit_time = fit_start.elapsed();

        let (
            verification_time,
            verification_status,
            verification_runs,
            verification_agreeing_runs,
            verification_max_abs_theta_delta,
        ) = if verify {
            let start = Instant::now();
            let verification = model.verify_convergence_with_options(verification_options())?;
            let elapsed = start.elapsed();
            let agreeing = verification.runs.iter().filter(|run| run.agrees).count();
            let max_delta = verification
                .runs
                .iter()
                .filter_map(|run| run.max_abs_theta_delta)
                .fold(None, option_max);
            (
                Some(elapsed),
                Some(convergence_verification_status_label(verification.status).to_string()),
                verification.runs.len(),
                agreeing,
                max_delta,
            )
        } else {
            (None, None, 0, 0, None)
        };

        let certificate = model
            .optimizer_certificate()
            .expect("fit should attach optimizer certificate");
        let gradient = &certificate.evidence.gradient;
        let hessian = &certificate.evidence.hessian;
        let raw_gradient_norm = gradient.raw_gradient_norm;
        let scaled_gradient_norm = gradient.scaled_gradient_norm;
        let hybrid_gradient_norm = raw_gradient_norm
            .zip(scaled_gradient_norm)
            .map(|(raw, scaled)| raw.min(scaled));
        let optimizer_stop = certificate.evidence.optimizer_stop.return_code.clone();

        println!(
            "  run {run_index}: fit={:.2} ms, stop={}, raw_grad={}, scaled_grad={}, hybrid={}, hess_min={}, hess_cond={}, theta_verify_delta={}",
            ms(fit_time),
            optimizer_stop.as_deref().unwrap_or("unknown"),
            fmt_opt(raw_gradient_norm),
            fmt_opt(scaled_gradient_norm),
            fmt_opt(hybrid_gradient_norm),
            fmt_opt(hessian.min_eigenvalue),
            fmt_opt(hessian.condition_number),
            fmt_opt(verification_max_abs_theta_delta),
        );

        runs.push(RunResult {
            run_index,
            fit_time_ms: ms(fit_time),
            verification_time_ms: verification_time.map(ms),
            objective: model.opt_summary().fmin,
            sigma: model.sigma(),
            theta: model.theta(),
            optimizer: model.opt_summary().optimizer_name().to_string(),
            optimizer_stop,
            acceptable_stop: certificate.evidence.optimizer_stop.acceptable_stop,
            budget_exhausted: certificate.evidence.optimizer_stop.budget_exhausted,
            fevals: certificate.evidence.optimizer_stop.function_evaluations,
            fit_status: fit_status_label(certificate.status).to_string(),
            raw_gradient_norm,
            scaled_gradient_norm,
            hybrid_gradient_norm,
            hessian_method: evidence_method_label(&hessian.method),
            hessian_quality: evidence_quality_label(&hessian.quality),
            hessian_min_eigenvalue: hessian.min_eigenvalue,
            hessian_condition_number: hessian.condition_number,
            hessian_rank: hessian.rank,
            verification_status,
            verification_runs,
            verification_agreeing_runs,
            verification_max_abs_theta_delta,
        });
    }

    let fit_times: Vec<f64> = runs.iter().map(|run| run.fit_time_ms).collect();
    let verification_times: Vec<f64> = runs
        .iter()
        .filter_map(|run| run.verification_time_ms)
        .collect();
    let theta_repeatability_max_abs = theta_repeatability_max_abs(&runs);
    let fit_status_last = runs
        .last()
        .map(|run| run.fit_status.clone())
        .unwrap_or_else(|| "unknown".to_string());
    let optimizer_stop_last = runs.last().and_then(|run| run.optimizer_stop.clone());

    Ok(ScenarioResult {
        label: scenario.label.to_string(),
        n_plate: scenario.n_plate,
        n_sample: scenario.n_sample,
        repeats_per_cell: scenario.repeats_per_cell,
        response_scale,
        n_obs: data.nrow(),
        formula: FORMULA.to_string(),
        reml: true,
        warmup_runs,
        measured_runs,
        fit_time_ms_min: min_finite(&fit_times).unwrap_or(f64::NAN),
        fit_time_ms_median: median_finite(&fit_times).unwrap_or(f64::NAN),
        verification_time_ms_median: median_finite(&verification_times),
        raw_gradient_norm_median: median_option(runs.iter().map(|run| run.raw_gradient_norm)),
        scaled_gradient_norm_median: median_option(runs.iter().map(|run| run.scaled_gradient_norm)),
        hybrid_gradient_norm_median: median_option(runs.iter().map(|run| run.hybrid_gradient_norm)),
        hessian_min_eigenvalue_min: min_option(runs.iter().map(|run| run.hessian_min_eigenvalue)),
        hessian_condition_number_max: max_option(
            runs.iter().map(|run| run.hessian_condition_number),
        ),
        theta_repeatability_max_abs,
        theta_verification_max_abs: max_option(
            runs.iter().map(|run| run.verification_max_abs_theta_delta),
        ),
        optimizer_stop_last,
        fit_status_last,
        runs,
    })
}

fn theta_repeatability_max_abs(runs: &[RunResult]) -> f64 {
    let Some(reference) = runs.first().map(|run| run.theta.as_slice()) else {
        return f64::NAN;
    };
    runs.iter()
        .map(|run| max_abs_delta(reference, &run.theta))
        .fold(0.0, f64::max)
}

fn max_abs_delta(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn fmt_opt(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.6e}"))
        .unwrap_or_else(|| "NA".to_string())
}

fn option_max(acc: Option<f64>, value: f64) -> Option<f64> {
    Some(acc.map_or(value, |acc| acc.max(value)))
}

fn min_finite(values: &[f64]) -> Option<f64> {
    values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(None, |acc, value| {
            Some(acc.map_or(value, |acc: f64| acc.min(value)))
        })
}

fn median_finite(values: &[f64]) -> Option<f64> {
    let mut finite: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if finite.is_empty() {
        return None;
    }
    finite.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = finite.len() / 2;
    if finite.len() % 2 == 0 {
        Some((finite[mid - 1] + finite[mid]) / 2.0)
    } else {
        Some(finite[mid])
    }
}

fn median_option(values: impl Iterator<Item = Option<f64>>) -> Option<f64> {
    let finite: Vec<f64> = values.flatten().filter(|value| value.is_finite()).collect();
    median_finite(&finite)
}

fn min_option(values: impl Iterator<Item = Option<f64>>) -> Option<f64> {
    let finite: Vec<f64> = values.flatten().filter(|value| value.is_finite()).collect();
    min_finite(&finite)
}

fn max_option(values: impl Iterator<Item = Option<f64>>) -> Option<f64> {
    values
        .flatten()
        .filter(|value| value.is_finite())
        .fold(None, option_max)
}

fn fit_status_label(status: FitStatus) -> &'static str {
    match status {
        FitStatus::ConvergedInterior => "converged_interior",
        FitStatus::ConvergedBoundary => "converged_boundary",
        FitStatus::ConvergedReducedRank => "converged_reduced_rank",
        FitStatus::ConvergedPenalised => "converged_penalised",
        FitStatus::NotIdentifiable => "not_identifiable",
        FitStatus::NotOptimized => "not_optimized",
        FitStatus::NotAssessed => "not_assessed",
    }
}

fn convergence_verification_status_label(status: ConvergenceVerificationStatus) -> &'static str {
    match status {
        ConvergenceVerificationStatus::NotRun => "not_run",
        ConvergenceVerificationStatus::RestartAgrees => "restart_agrees",
        ConvergenceVerificationStatus::OptimizerConsensus => "optimizer_consensus",
        ConvergenceVerificationStatus::Fragile => "fragile",
        ConvergenceVerificationStatus::Unstable => "unstable",
    }
}

fn evidence_method_label(method: &EvidenceMethod) -> String {
    match method {
        EvidenceMethod::Exact => "exact".to_string(),
        EvidenceMethod::FiniteDifference => "finite_difference".to_string(),
        EvidenceMethod::OptimizerReported => "optimizer_reported".to_string(),
        EvidenceMethod::NotAvailable { reason } => format!("not_available: {reason}"),
        EvidenceMethod::NotAssessed { reason } => format!("not_assessed: {reason}"),
    }
}

fn evidence_quality_label(quality: &EvidenceQuality) -> String {
    match quality {
        EvidenceQuality::Certified => "certified".to_string(),
        EvidenceQuality::Approximate { reason } => format!("approximate: {reason}"),
        EvidenceQuality::Unavailable { reason } => format!("unavailable: {reason}"),
        EvidenceQuality::NotAssessed { reason } => format!("not_assessed: {reason}"),
        EvidenceQuality::Failed { reason } => format!("failed: {reason}"),
    }
}

fn default_scenarios(include_xl: bool) -> Vec<Scenario> {
    let mut scenarios = vec![
        Scenario {
            label: "native_24x6",
            n_plate: 24,
            n_sample: 6,
            repeats_per_cell: 1,
            seed: 120,
        },
        Scenario {
            label: "medium_48x12",
            n_plate: 48,
            n_sample: 12,
            repeats_per_cell: 1,
            seed: 121,
        },
        Scenario {
            label: "large_96x24",
            n_plate: 96,
            n_sample: 24,
            repeats_per_cell: 1,
            seed: 122,
        },
    ];
    if include_xl {
        scenarios.push(Scenario {
            label: "xl_192x48",
            n_plate: 192,
            n_sample: 48,
            repeats_per_cell: 1,
            seed: 123,
        });
    }
    scenarios
}

fn print_summary_csv(results: &[ScenarioResult]) {
    println!("\nscenario,n_obs,fit_ms_median,fit_ms_min,verify_ms_median,raw_grad_median,scaled_grad_median,hybrid_grad_median,hessian_min_eigen_min,hessian_condition_max,theta_repeatability_max_abs,theta_verification_max_abs,optimizer_stop,fit_status");
    for result in results {
        println!(
            "{},{},{:.3},{:.3},{},{},{},{},{},{},{:.6e},{},{},{}",
            result.label,
            result.n_obs,
            result.fit_time_ms_median,
            result.fit_time_ms_min,
            fmt_csv_opt(result.verification_time_ms_median),
            fmt_csv_opt(result.raw_gradient_norm_median),
            fmt_csv_opt(result.scaled_gradient_norm_median),
            fmt_csv_opt(result.hybrid_gradient_norm_median),
            fmt_csv_opt(result.hessian_min_eigenvalue_min),
            fmt_csv_opt(result.hessian_condition_number_max),
            result.theta_repeatability_max_abs,
            fmt_csv_opt(result.theta_verification_max_abs),
            result.optimizer_stop_last.as_deref().unwrap_or("unknown"),
            result.fit_status_last,
        );
    }
}

fn fmt_csv_opt(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.6e}"))
        .unwrap_or_else(|| "NA".to_string())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let warmup_runs = env_usize("PENICILLIN_BENCH_WARMUP", 1);
    let measured_runs = env_usize("PENICILLIN_BENCH_RUNS", 3);
    let verify = env_bool("PENICILLIN_BENCH_VERIFY", true);
    let include_xl = env_bool("PENICILLIN_BENCH_XL", false);
    let response_scale = env_f64("PENICILLIN_BENCH_RESPONSE_SCALE", 1.0);

    let grid = load_penicillin_grid()?;
    let mut results = Vec::new();
    for scenario in default_scenarios(include_xl) {
        results.push(run_scenario(
            &grid,
            &scenario,
            response_scale,
            warmup_runs,
            measured_runs,
            verify,
        )?);
    }

    print_summary_csv(&results);

    let outpath = comparison_root().join("rust_results.json");
    fs::create_dir_all(outpath.parent().expect("output parent"))?;
    fs::write(
        &outpath,
        serde_json::to_string_pretty(&json!({
            "tool": "mixedmodels (rust)",
            "version": env!("CARGO_PKG_VERSION"),
            "formula": FORMULA,
            "benchmark": "penicillin_convergence",
            "response_scale": response_scale,
            "verification_enabled": verify,
            "results": results,
        }))?,
    )?;
    println!("\nwrote {}", outpath.display());

    Ok(())
}
