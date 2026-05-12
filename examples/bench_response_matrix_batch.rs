//! Benchmark amortization for the LMM response-matrix batch API.
//!
//! Companion lme4 loop:
//!
//! ```text
//! Rscript scripts/bench_response_matrix_lme4.R
//! ```
//!
//! Useful local run:
//!
//! ```text
//! MIXEDMODELS_RESPONSE_BATCH_QS=1,4,16,64 cargo run --example bench_response_matrix_batch
//! ```

use std::time::{Duration, Instant};

use nalgebra::DMatrix;

use mixeff_rs::datasets;
use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::{
    BatchOptimizerControl, BatchOptions, BatchWarmStart, LinearMixedModel, LinearMixedModelBatch,
    ResponseBatchMode,
};

const DEFAULT_QS: &[usize] = &[1, 4, 16, 64];

struct Case {
    id: &'static str,
    dataset: &'static str,
    formula: &'static str,
    response: &'static str,
}

const CASES: &[Case] = &[
    Case {
        id: "dyestuff_scalar_re",
        dataset: "dyestuff",
        formula: "Yield ~ 1 + (1 | Batch)",
        response: "Yield",
    },
    Case {
        id: "sleepstudy_slope",
        dataset: "sleepstudy",
        formula: "Reaction ~ 1 + Days + (1 + Days | Subject)",
        response: "Reaction",
    },
    Case {
        id: "penicillin_crossed",
        dataset: "penicillin",
        formula: "diameter ~ 1 + (1 | plate) + (1 | sample)",
        response: "diameter",
    },
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let qs = q_grid();
    println!("engine,case,n,q,mode,total_ms,per_response_ms,success_count,theta_dim");
    for case in CASES {
        let (data, _) = datasets::load(case.dataset)?;
        let formula = parse_formula(case.formula)?;
        let mut template = LinearMixedModel::new(formula, &data, None)?;
        template.fit(true)?;
        let theta = template.theta();
        let batch = LinearMixedModelBatch::from_model_with_options(
            &template,
            BatchOptions {
                chunk_columns: chunk_columns(),
                max_failures: None,
            },
        )?;

        let response = data
            .numeric(case.response)
            .ok_or_else(|| format!("{} response column is not numeric", case.response))?;
        for &q in &qs {
            let responses = response_matrix(response, q);
            let profile = time_it(|| {
                batch.fit_responses(
                    &responses,
                    ResponseBatchMode::ProfileAtTheta {
                        theta: theta.clone(),
                        reml: true,
                    },
                )
            })?;
            print_row(
                case,
                data.nrow(),
                q,
                "profile_at_theta",
                profile.elapsed,
                profile.value.success_count(),
                theta.len(),
            );

            let control = BatchOptimizerControl {
                max_evaluations: max_evaluations(),
                objective_tolerance: 1e-8,
                theta_tolerance: 1e-5,
                initial_step: Some(vec![0.25; theta.len()]),
                options: BatchOptions {
                    chunk_columns: chunk_columns(),
                    max_failures: None,
                },
            };
            let shared = time_it(|| {
                batch.fit_responses(
                    &responses,
                    ResponseBatchMode::OptimizeSharedTheta {
                        reml: true,
                        control: control.clone(),
                    },
                )
            })?;
            print_row(
                case,
                data.nrow(),
                q,
                "optimize_shared_theta",
                shared.elapsed,
                shared.value.success_count(),
                theta.len(),
            );

            if run_per_column() {
                let per_column = time_it(|| {
                    batch.fit_responses(
                        &responses,
                        ResponseBatchMode::OptimizePerColumn {
                            reml: true,
                            warm_start: BatchWarmStart::SharedTheta,
                            control: control.clone(),
                        },
                    )
                })?;
                print_row(
                    case,
                    data.nrow(),
                    q,
                    "optimize_per_column",
                    per_column.elapsed,
                    per_column.value.success_count(),
                    theta.len(),
                );
            }

            if run_scalar_rust_loop() {
                let scalar = time_it(|| scalar_refit_loop(&template, &responses))?;
                print_row(
                    case,
                    data.nrow(),
                    q,
                    "scalar_refit_loop",
                    scalar.elapsed,
                    scalar.value,
                    theta.len(),
                );
            }
        }
    }
    Ok(())
}

struct Timed<T> {
    value: T,
    elapsed: Duration,
}

fn time_it<T, E>(f: impl FnOnce() -> Result<T, E>) -> Result<Timed<T>, E> {
    let start = Instant::now();
    let value = f()?;
    Ok(Timed {
        value,
        elapsed: start.elapsed(),
    })
}

fn scalar_refit_loop(
    template: &LinearMixedModel,
    responses: &DMatrix<f64>,
) -> mixeff_rs::error::Result<usize> {
    let mut successes = 0usize;
    for col in 0..responses.ncols() {
        let mut model = template.clone();
        model.refit(responses.column(col).as_slice())?;
        successes += 1;
    }
    Ok(successes)
}

fn response_matrix(response: &[f64], q: usize) -> DMatrix<f64> {
    let mean = response.iter().sum::<f64>() / response.len() as f64;
    let sd = (response
        .iter()
        .map(|value| {
            let centered = value - mean;
            centered * centered
        })
        .sum::<f64>()
        / response.len() as f64)
        .sqrt()
        .max(1.0);
    DMatrix::from_fn(response.len(), q, |row, col| {
        let scale = 0.75 + 0.5 * ((col % 17) as f64 / 16.0);
        let offset = ((col % 5) as f64 - 2.0) * 0.05 * sd;
        scale * response[row] + offset
    })
}

fn q_grid() -> Vec<usize> {
    std::env::var("MIXEDMODELS_RESPONSE_BATCH_QS")
        .ok()
        .map(|value| {
            value
                .split(',')
                .filter_map(|part| part.trim().parse::<usize>().ok())
                .filter(|value| *value > 0)
                .collect::<Vec<_>>()
        })
        .filter(|values| !values.is_empty())
        .unwrap_or_else(|| DEFAULT_QS.to_vec())
}

fn chunk_columns() -> usize {
    std::env::var("MIXEDMODELS_RESPONSE_BATCH_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(32)
}

fn max_evaluations() -> i64 {
    std::env::var("MIXEDMODELS_RESPONSE_BATCH_MAXEVAL")
        .ok()
        .and_then(|value| value.parse::<i64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(80)
}

fn run_per_column() -> bool {
    env_flag("MIXEDMODELS_RESPONSE_BATCH_PER_COLUMN", false)
}

fn run_scalar_rust_loop() -> bool {
    env_flag("MIXEDMODELS_RESPONSE_BATCH_SCALAR_RUST", false)
}

fn env_flag(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(default)
}

fn print_row(
    case: &Case,
    n: usize,
    q: usize,
    mode: &str,
    elapsed: Duration,
    success_count: usize,
    theta_dim: usize,
) {
    let total_ms = elapsed.as_secs_f64() * 1000.0;
    println!(
        "rust,{},{},{},{},{:.3},{:.6},{},{}",
        case.id,
        n,
        q,
        mode,
        total_ms,
        total_ms / q as f64,
        success_count,
        theta_dim
    );
}
