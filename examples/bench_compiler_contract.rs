//! Compiler-contract benchmark hooks.
//!
//! Run:
//!     cargo run --release --example bench_compiler_contract
//!
//! This benchmark times the compiler/audit surface called out in the compiler
//! contract PRD: formula-to-IR, explanation, design audit, successful
//! parse/build/fit, and design-time failure-path diagnosis/refusal. It writes a
//! JSON baseline so regressions can be compared across commits.
//!
//! Optional environment controls:
//! - `COMPILER_CONTRACT_BENCH_WARMUP=2`
//! - `COMPILER_CONTRACT_BENCH_RUNS=5`
//! - `COMPILER_CONTRACT_BENCH_CROSSED_ROWS=10000`
//! - `COMPILER_CONTRACT_BENCH_FAILURE_ROWS=10000`
//! - `COMPILER_CONTRACT_BENCH_OUTDIR=/tmp/compiler_contract_bench`

use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use serde::Serialize;
use serde_json::json;

use mixeff_rs::compiler::{
    audit_design, compile_formula_ir, explain_model, recommend_policy, CompilerPolicy, Diagnostic,
};
use mixeff_rs::datasets;
use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::data::DataFrame;
use mixeff_rs::model::linear::LinearMixedModel;

const SLEEPSTUDY_FORMULA: &str = "Reaction ~ 1 + Days + (1 + Days | Subject)";
const CROSSED_FORMULA: &str = "y ~ 1 + x + (1 | subject) + (1 | item)";
const FAILURE_FORMULA: &str = "y ~ x + (1 + x | group)";

#[derive(Debug, Serialize)]
struct OperationTiming {
    operation: String,
    scenario: String,
    target_ms: Option<f64>,
    measured_runs: usize,
    min_ms: f64,
    median_ms: f64,
    max_ms: f64,
}

#[derive(Debug, Serialize)]
struct ScenarioInfo {
    label: String,
    formula: String,
    n_obs: usize,
    row_covariance_dense_bytes_if_materialized: u64,
    row_covariance_materialized_by_benchmark: bool,
    note: String,
}

#[derive(Debug, Serialize)]
struct FailurePathSummary {
    scenario: String,
    policy_recommendations: usize,
    diagnostic_codes: Vec<String>,
    policy_actions: Vec<String>,
    refusal_error: String,
}

fn comparison_root() -> PathBuf {
    if let Ok(path) = std::env::var("COMPILER_CONTRACT_BENCH_OUTDIR") {
        return PathBuf::from(path);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("comparison")
        .join("compiler_contract")
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn percentile(samples: &[f64], q: f64) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx]
}

fn simple_error(message: impl Into<String>) -> Box<dyn std::error::Error> {
    Box::new(std::io::Error::other(message.into()))
}

fn time_operation<F>(
    operation: &str,
    scenario: &str,
    target_ms: Option<f64>,
    warmup_runs: usize,
    measured_runs: usize,
    mut run_once: F,
) -> Result<OperationTiming, Box<dyn std::error::Error>>
where
    F: FnMut() -> Result<(), Box<dyn std::error::Error>>,
{
    for _ in 0..warmup_runs {
        run_once()?;
    }

    let mut samples = Vec::with_capacity(measured_runs);
    for _ in 0..measured_runs {
        let start = Instant::now();
        run_once()?;
        samples.push(ms(start.elapsed()));
    }

    Ok(OperationTiming {
        operation: operation.to_string(),
        scenario: scenario.to_string(),
        target_ms,
        measured_runs,
        min_ms: percentile(&samples, 0.0),
        median_ms: percentile(&samples, 0.5),
        max_ms: percentile(&samples, 1.0),
    })
}

fn dense_row_covariance_bytes(n_obs: usize) -> u64 {
    let n = n_obs as u64;
    n.saturating_mul(n)
        .saturating_mul(std::mem::size_of::<f64>() as u64)
}

fn crossed_intercept_data(n_rows: usize) -> DataFrame {
    let n_subjects = 100usize;
    let n_items = (n_rows / n_subjects).max(1);
    let total = n_subjects * n_items;

    let mut y = Vec::with_capacity(total);
    let mut x = Vec::with_capacity(total);
    let mut subject = Vec::with_capacity(total);
    let mut item = Vec::with_capacity(total);

    for subject_idx in 0..n_subjects {
        let subject_shift = subject_idx as f64 * 0.01;
        for item_idx in 0..n_items {
            let item_shift = item_idx as f64 * 0.005;
            let x_value = (item_idx % 10) as f64;
            y.push(10.0 + 0.25 * x_value + subject_shift + item_shift);
            x.push(x_value);
            subject.push(format!("S{:04}", subject_idx + 1));
            item.push(format!("I{:04}", item_idx + 1));
        }
    }

    let mut data = DataFrame::new();
    data.add_numeric("y", y).unwrap();
    data.add_numeric("x", x).unwrap();
    data.add_categorical("subject", subject).unwrap();
    data.add_categorical("item", item).unwrap();
    data
}

fn failure_path_data(n_rows: usize) -> DataFrame {
    let n_groups = 2usize;
    let mut y = Vec::with_capacity(n_rows);
    let mut x = Vec::with_capacity(n_rows);
    let mut group = Vec::with_capacity(n_rows);

    for row in 0..n_rows {
        let group_idx = row % n_groups;
        let x_value = ((row / n_groups) % 100) as f64 / 10.0;
        y.push(5.0 + 0.5 * x_value + group_idx as f64);
        x.push(x_value);
        group.push(format!("g{}", group_idx + 1));
    }

    let mut data = DataFrame::new();
    data.add_numeric("y", y).unwrap();
    data.add_numeric("x", x).unwrap();
    data.add_categorical("group", group).unwrap();
    data
}

fn scenario_info(label: &str, formula: &str, data: &DataFrame, note: &str) -> ScenarioInfo {
    ScenarioInfo {
        label: label.to_string(),
        formula: formula.to_string(),
        n_obs: data.nrow(),
        row_covariance_dense_bytes_if_materialized: dense_row_covariance_bytes(data.nrow()),
        row_covariance_materialized_by_benchmark: false,
        note: note.to_string(),
    }
}

fn summarize_diagnostics(diagnostics: &[Diagnostic]) -> Vec<String> {
    let mut codes = diagnostics
        .iter()
        .map(|diagnostic| format!("{:?}", diagnostic.code))
        .collect::<Vec<_>>();
    codes.sort();
    codes.dedup();
    codes
}

fn failure_summary(
    data: &DataFrame,
    refusal_error: String,
) -> Result<FailurePathSummary, Box<dyn std::error::Error>> {
    let formula = parse_formula(FAILURE_FORMULA)?;
    let semantic = compile_formula_ir(&formula);
    let audit = audit_design(&semantic, data);
    let policy = CompilerPolicy::design_compiled();
    let recommendations = recommend_policy(&semantic, &audit, &policy);

    let mut diagnostics = audit.diagnostics.clone();
    for recommendation in &recommendations {
        diagnostics.extend(recommendation.diagnostics.clone());
    }

    let mut policy_actions = recommendations
        .iter()
        .map(|recommendation| {
            format!(
                "{:?}: {}; inference={}",
                recommendation.action, recommendation.reason, recommendation.inference_consequence
            )
        })
        .collect::<Vec<_>>();
    policy_actions.sort();
    policy_actions.dedup();

    Ok(FailurePathSummary {
        scenario: "two_group_slope_refusal".to_string(),
        policy_recommendations: recommendations.len(),
        diagnostic_codes: summarize_diagnostics(&diagnostics),
        policy_actions,
        refusal_error,
    })
}

fn refusal_error(data: &DataFrame) -> Result<String, Box<dyn std::error::Error>> {
    let formula = parse_formula(FAILURE_FORMULA)?;
    match LinearMixedModel::new_with_compiler_policy(
        formula,
        data,
        None,
        CompilerPolicy::design_compiled(),
    ) {
        Ok(_) => Err(simple_error(
            "expected design_compiled policy to refuse unsupported random-effect distribution",
        )),
        Err(error) => Ok(error.to_string()),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let warmup_runs = env_usize("COMPILER_CONTRACT_BENCH_WARMUP", 2);
    let measured_runs = env_usize("COMPILER_CONTRACT_BENCH_RUNS", 5).max(1);
    let crossed_rows = env_usize("COMPILER_CONTRACT_BENCH_CROSSED_ROWS", 10_000);
    let failure_rows = env_usize("COMPILER_CONTRACT_BENCH_FAILURE_ROWS", 10_000);

    let (sleepstudy, _) = datasets::load("sleepstudy")?;
    let crossed = crossed_intercept_data(crossed_rows);
    let failure = failure_path_data(failure_rows);

    let mut timings = Vec::new();

    timings.push(time_operation(
        "formula_to_ir",
        "sleepstudy",
        Some(10.0),
        warmup_runs,
        measured_runs,
        || {
            let formula = parse_formula(SLEEPSTUDY_FORMULA)?;
            let semantic = compile_formula_ir(&formula);
            black_box(semantic.random_terms.len());
            Ok(())
        },
    )?);

    timings.push(time_operation(
        "explain_model",
        "sleepstudy",
        Some(10.0),
        warmup_runs,
        measured_runs,
        || {
            let formula = parse_formula(SLEEPSTUDY_FORMULA)?;
            let explanation = explain_model(&formula);
            black_box(explanation.sections.len());
            Ok(())
        },
    )?);

    timings.push(time_operation(
        "design_audit",
        "sleepstudy",
        Some(10.0),
        warmup_runs,
        measured_runs,
        || {
            let formula = parse_formula(SLEEPSTUDY_FORMULA)?;
            let semantic = compile_formula_ir(&formula);
            let audit = audit_design(&semantic, &sleepstudy);
            black_box(audit.random_terms.len());
            Ok(())
        },
    )?);

    timings.push(time_operation(
        "model_construct_with_design_audit",
        "sleepstudy",
        None,
        warmup_runs,
        measured_runs,
        || {
            let formula = parse_formula(SLEEPSTUDY_FORMULA)?;
            let model = LinearMixedModel::new(formula, &sleepstudy, None)?;
            black_box(model.design_audit().is_some());
            Ok(())
        },
    )?);

    timings.push(time_operation(
        "parse_construct_fit",
        "sleepstudy",
        Some(50.0),
        warmup_runs,
        measured_runs,
        || {
            let formula = parse_formula(SLEEPSTUDY_FORMULA)?;
            let mut model = LinearMixedModel::new(formula, &sleepstudy, None)?;
            model.fit(false)?;
            black_box(model.objective_value());
            Ok(())
        },
    )?);

    timings.push(time_operation(
        "design_audit",
        "crossed_10k",
        None,
        warmup_runs,
        measured_runs,
        || {
            let formula = parse_formula(CROSSED_FORMULA)?;
            let semantic = compile_formula_ir(&formula);
            let audit = audit_design(&semantic, &crossed);
            black_box(audit.covariance_kernels.kernels.len());
            Ok(())
        },
    )?);

    timings.push(time_operation(
        "failure_path_diagnose",
        "two_group_slope_refusal",
        Some(250.0),
        warmup_runs,
        measured_runs,
        || {
            let formula = parse_formula(FAILURE_FORMULA)?;
            let semantic = compile_formula_ir(&formula);
            let audit = audit_design(&semantic, &failure);
            let recommendations =
                recommend_policy(&semantic, &audit, &CompilerPolicy::design_compiled());
            black_box(recommendations.len());
            Ok(())
        },
    )?);

    timings.push(time_operation(
        "failure_path_refuse",
        "two_group_slope_refusal",
        Some(250.0),
        warmup_runs,
        measured_runs,
        || {
            black_box(refusal_error(&failure)?);
            Ok(())
        },
    )?);

    let refusal = refusal_error(&failure)?;
    let failure_summary = failure_summary(&failure, refusal)?;

    let scenarios = vec![
        scenario_info(
            "sleepstudy",
            SLEEPSTUDY_FORMULA,
            &sleepstudy,
            "sleepstudy-scale parse, explanation, design audit, construction, and fit",
        ),
        scenario_info(
            "crossed_10k",
            CROSSED_FORMULA,
            &crossed,
            "10k-row crossed random-intercept design audit without dense row covariance",
        ),
        scenario_info(
            "two_group_slope_refusal",
            FAILURE_FORMULA,
            &failure,
            "10k-row design-time refusal path with too few grouping levels for random slopes",
        ),
    ];

    let output = json!({
        "tool": "mixeff-rs compiler contract benchmark",
        "version": env!("CARGO_PKG_VERSION"),
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "targets_source": "docs/compiler_contract_v0_prd.md performance budget",
        "row_covariance_materialized_by_default": false,
        "scenarios": scenarios,
        "failure_path": failure_summary,
        "timings": timings,
    });

    let outdir = comparison_root();
    fs::create_dir_all(&outdir)?;
    let outpath = outdir.join("rust_results.json");
    fs::write(&outpath, serde_json::to_string_pretty(&output)?)?;

    println!("wrote {}", outpath.display());
    Ok(())
}
