//! Deterministic performance regression gate against a checked-in baseline.
//!
//! Wall-clock benchmarks are too noisy to gate hard, so this harness splits
//! the gate in two:
//!
//! * **Hard gates** on metrics that are deterministic for a fixed toolchain
//!   and dependency set: fitted objective, optimizer evaluation count,
//!   optimizer/backend identity, and heap allocations (count and bytes) per
//!   profiled-objective evaluation. Any drift here is a real behavior change
//!   — either a regression or an intentional change that warrants
//!   regenerating the baseline.
//! * **Soft gates** on median evaluation wall time, with a wide band
//!   (default 1.5x) that only warns unless `--strict-time` is passed.
//!
//! ```sh
//! # compare against benchmarks/perf_baseline.json (exit 1 on hard failure)
//! cargo run --release --features unstable-internals --example perf_gate
//!
//! # regenerate the baseline after an intentional performance change
//! cargo run --release --features unstable-internals --example perf_gate -- --write-baseline
//! ```
//!
//! Scenarios mirror `objective_eval_bench` (same simulators, same formulas)
//! but use fewer repetitions: timing is only soft-gated, so the gate stays
//! fast enough to run per-change.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::batch::{
    BatchOptimizerControl, BatchWarmStart, LinearMixedModelBatch, ResponseBatchMode,
};
use mixeff_rs::model::data::DataFrame;
use mixeff_rs::model::linear::LinearMixedModel;
use mixeff_rs::model::traits::MixedModelFit;
use nalgebra::DMatrix;

static ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

const SCHEMA_VERSION: u32 = 1;
const TIMING_REPS: usize = 400;
const TIMING_BATCHES: usize = 5;
const ALLOC_EVALS: u64 = 100;
const WARMUP_EVALS: usize = 25;
const OBJECTIVE_REL_TOL: f64 = 1e-9;
const TIME_RATIO_BAND: f64 = 1.5;

#[derive(Debug, Serialize, Deserialize)]
struct Baseline {
    schema_version: u32,
    profile: String,
    scenarios: Vec<ScenarioMetrics>,
    batch_scenarios: Vec<BatchScenarioMetrics>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ScenarioMetrics {
    name: String,
    fit_objective: f64,
    fit_feval: i64,
    optimizer: String,
    backend: String,
    allocs_per_eval: u64,
    bytes_per_eval: u64,
    eval_median_us: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct BatchScenarioMetrics {
    name: String,
    q: usize,
    success_count: usize,
    objective_sum: f64,
    allocs_per_column: u64,
    bytes_per_column: u64,
    total_median_ms: f64,
}

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
}

#[derive(Clone, Copy)]
struct Scenario {
    name: &'static str,
    formula: &'static str,
    reml: bool,
    kind: ScenarioKind,
}

fn scenarios() -> Vec<Scenario> {
    let vector_formula = "reaction ~ 1 + days + (1 + days | subj)";
    let scalar_formula = "reaction ~ 1 + days + (1 | subj)";
    let crossed_formula =
        "reaction ~ 1 + days + (1 + days | subj) + (1 + days | item) + (1 + days | site)";

    vec![
        Scenario {
            name: "scalar_1000",
            formula: scalar_formula,
            reml: true,
            kind: ScenarioKind::Sleepstudy {
                n_subjects: 100,
                n_obs: 10,
                seed: 42,
            },
        },
        Scenario {
            name: "vector_1000",
            formula: vector_formula,
            reml: true,
            kind: ScenarioKind::Sleepstudy {
                n_subjects: 100,
                n_obs: 10,
                seed: 42,
            },
        },
        Scenario {
            name: "vector_10000",
            formula: vector_formula,
            reml: true,
            kind: ScenarioKind::Sleepstudy {
                n_subjects: 1000,
                n_obs: 10,
                seed: 42,
            },
        },
        Scenario {
            name: "crossed_small",
            formula: crossed_formula,
            reml: true,
            kind: ScenarioKind::Crossed {
                n_subjects: 18,
                n_items: 12,
                n_sites: 6,
                n_rep: 4,
            },
        },
        Scenario {
            name: "crossed_medium",
            formula: crossed_formula,
            reml: true,
            kind: ScenarioKind::Crossed {
                n_subjects: 36,
                n_items: 24,
                n_sites: 8,
                n_rep: 4,
            },
        },
    ]
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

fn build_data(kind: ScenarioKind) -> DataFrame {
    match kind {
        ScenarioKind::Sleepstudy {
            n_subjects,
            n_obs,
            seed,
        } => simulate_sleepstudy_like(n_subjects, n_obs, seed),
        ScenarioKind::Crossed {
            n_subjects,
            n_items,
            n_sites,
            n_rep,
        } => simulate_crossed(n_subjects, n_items, n_sites, n_rep),
    }
}

fn compile_profile() -> &'static str {
    if cfg!(feature = "nlopt") {
        "rust_default_nlopt"
    } else {
        "rust_no_default_native"
    }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values[values.len() / 2]
}

fn run_scenario(scenario: Scenario) -> ScenarioMetrics {
    let data = build_data(scenario.kind);
    let formula = parse_formula(scenario.formula).expect("formula parse failed");
    let mut model = LinearMixedModel::new(formula, &data, None).expect("model build failed");
    model.fit(scenario.reml).expect("model fit failed");

    let theta = model.theta();
    let fit_objective = model.objective();
    let fit_feval = model.optsum().feval;
    let optimizer = model.optsum().optimizer_name().to_string();
    let backend = model.optsum().backend_name().to_string();

    for _ in 0..WARMUP_EVALS {
        black_box(model.objective_at(black_box(&theta)).expect("eval failed"));
    }

    let count_before = ALLOC_COUNT.load(Ordering::Relaxed);
    let bytes_before = ALLOC_BYTES.load(Ordering::Relaxed);
    for _ in 0..ALLOC_EVALS {
        black_box(model.objective_at(black_box(&theta)).expect("eval failed"));
    }
    let allocs_per_eval = (ALLOC_COUNT.load(Ordering::Relaxed) - count_before) / ALLOC_EVALS;
    let bytes_per_eval = (ALLOC_BYTES.load(Ordering::Relaxed) - bytes_before) / ALLOC_EVALS;

    let mut batch_us = Vec::with_capacity(TIMING_BATCHES);
    for _ in 0..TIMING_BATCHES {
        let start = Instant::now();
        for _ in 0..TIMING_REPS {
            black_box(model.objective_at(black_box(&theta)).expect("eval failed"));
        }
        batch_us.push(start.elapsed().as_secs_f64() * 1_000_000.0 / TIMING_REPS as f64);
    }

    ScenarioMetrics {
        name: scenario.name.to_string(),
        fit_objective,
        fit_feval,
        optimizer,
        backend,
        allocs_per_eval,
        bytes_per_eval,
        eval_median_us: median(&mut batch_us),
    }
}

/// Deterministic response matrix: column 0 is the simulated response,
/// later columns add bounded arithmetic perturbations.
fn response_matrix(base: &[f64], q: usize) -> DMatrix<f64> {
    DMatrix::from_fn(base.len(), q, |i, j| {
        base[i] + centered_mod(7 * i + 13 * j, 31, 15.0, 0.9)
    })
}

const BATCH_Q: usize = 16;
const BATCH_TIMING_ROUNDS: usize = 5;

fn run_batch_scenarios() -> Vec<BatchScenarioMetrics> {
    let data = simulate_sleepstudy_like(100, 10, 42);
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut template = LinearMixedModel::new(formula, &data, None).expect("template build failed");
    template.fit(true).expect("template fit failed");
    let theta = template.theta();
    let batch = LinearMixedModelBatch::from_model(&template).expect("batch build failed");
    let responses = response_matrix(data.numeric("reaction").unwrap(), BATCH_Q);

    let control = BatchOptimizerControl {
        max_evaluations: 500,
        objective_tolerance: 1e-8,
        theta_tolerance: 1e-5,
        initial_step: Some(vec![0.25; theta.len()]),
        ..BatchOptimizerControl::default()
    };
    let modes: Vec<(&str, ResponseBatchMode)> = vec![
        (
            "batch_profile_at_theta",
            ResponseBatchMode::ProfileAtTheta {
                theta: theta.clone(),
                reml: true,
            },
        ),
        (
            "batch_optimize_per_column",
            ResponseBatchMode::OptimizePerColumn {
                reml: true,
                warm_start: BatchWarmStart::SharedTheta,
                control,
            },
        ),
    ];

    modes
        .into_iter()
        .map(|(name, mode)| {
            eprintln!("running {name} ...");
            // warm-up (also the allocation-count run after the first call
            // has settled any lazily grown buffers)
            let fit = batch
                .fit_responses(&responses, mode.clone())
                .expect("batch warm-up failed");
            let count_before = ALLOC_COUNT.load(Ordering::Relaxed);
            let bytes_before = ALLOC_BYTES.load(Ordering::Relaxed);
            let fit_counted = batch
                .fit_responses(&responses, mode.clone())
                .expect("batch fit failed");
            let allocs = ALLOC_COUNT.load(Ordering::Relaxed) - count_before;
            let bytes = ALLOC_BYTES.load(Ordering::Relaxed) - bytes_before;
            assert_eq!(
                fit.objective, fit_counted.objective,
                "batch not deterministic"
            );

            let mut round_ms = Vec::with_capacity(BATCH_TIMING_ROUNDS);
            for _ in 0..BATCH_TIMING_ROUNDS {
                let start = Instant::now();
                black_box(
                    batch
                        .fit_responses(black_box(&responses), mode.clone())
                        .expect("batch fit failed"),
                );
                round_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
            }

            BatchScenarioMetrics {
                name: name.to_string(),
                q: BATCH_Q,
                success_count: fit_counted.success_count(),
                objective_sum: fit_counted.objective.iter().sum(),
                allocs_per_column: allocs / BATCH_Q as u64,
                bytes_per_column: bytes / BATCH_Q as u64,
                total_median_ms: median(&mut round_ms),
            }
        })
        .collect()
}

fn baseline_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("benchmarks/perf_baseline.json")
}

struct GateReport {
    hard_failures: Vec<String>,
    soft_warnings: Vec<String>,
}

fn compare(baseline: &Baseline, current: &Baseline) -> GateReport {
    let mut report = GateReport {
        hard_failures: Vec::new(),
        soft_warnings: Vec::new(),
    };

    if baseline.schema_version != current.schema_version {
        report.hard_failures.push(format!(
            "schema_version: baseline {} vs current {} — regenerate the baseline",
            baseline.schema_version, current.schema_version
        ));
        return report;
    }
    if baseline.profile != current.profile {
        report.hard_failures.push(format!(
            "compile profile: baseline '{}' vs current '{}' — run with matching features \
             or regenerate the baseline",
            baseline.profile, current.profile
        ));
        return report;
    }

    for base in &baseline.scenarios {
        let Some(cur) = current.scenarios.iter().find(|s| s.name == base.name) else {
            report
                .hard_failures
                .push(format!("{}: scenario missing from current run", base.name));
            continue;
        };

        let obj_tol = OBJECTIVE_REL_TOL * base.fit_objective.abs().max(1.0);
        if (cur.fit_objective - base.fit_objective).abs() > obj_tol {
            report.hard_failures.push(format!(
                "{}: fit_objective drifted {} -> {} (tol {obj_tol:.3e}); numerical behavior \
                 changed — likely surface: theta evaluation or optimizer path",
                base.name, base.fit_objective, cur.fit_objective
            ));
        }
        if cur.fit_feval != base.fit_feval {
            report.hard_failures.push(format!(
                "{}: fit_feval changed {} -> {}; optimizer trajectory changed",
                base.name, base.fit_feval, cur.fit_feval
            ));
        }
        if cur.optimizer != base.optimizer || cur.backend != base.backend {
            report.hard_failures.push(format!(
                "{}: optimizer/backend changed {}/{} -> {}/{}; fit() selection changed",
                base.name, base.optimizer, base.backend, cur.optimizer, cur.backend
            ));
        }
        if cur.allocs_per_eval != base.allocs_per_eval {
            report.hard_failures.push(format!(
                "{}: allocs_per_eval changed {} -> {}; per-evaluation allocation behavior \
                 changed (regenerate the baseline only if this is an intentional change)",
                base.name, base.allocs_per_eval, cur.allocs_per_eval
            ));
        }
        if cur.bytes_per_eval != base.bytes_per_eval {
            report.hard_failures.push(format!(
                "{}: bytes_per_eval changed {} -> {}; per-evaluation allocation behavior \
                 changed (regenerate the baseline only if this is an intentional change)",
                base.name, base.bytes_per_eval, cur.bytes_per_eval
            ));
        }

        let ratio = cur.eval_median_us / base.eval_median_us;
        if ratio > TIME_RATIO_BAND {
            report.soft_warnings.push(format!(
                "{}: eval_median_us {:.3} -> {:.3} ({:.2}x, band {TIME_RATIO_BAND}x)",
                base.name, base.eval_median_us, cur.eval_median_us, ratio
            ));
        } else if ratio < 1.0 / TIME_RATIO_BAND {
            report.soft_warnings.push(format!(
                "{}: eval_median_us improved {:.3} -> {:.3} ({:.2}x) — consider refreshing \
                 the baseline to tighten the gate",
                base.name, base.eval_median_us, cur.eval_median_us, ratio
            ));
        }
    }

    for cur in &current.scenarios {
        if !baseline.scenarios.iter().any(|s| s.name == cur.name) {
            report.hard_failures.push(format!(
                "{}: scenario missing from baseline — regenerate the baseline",
                cur.name
            ));
        }
    }

    for base in &baseline.batch_scenarios {
        let Some(cur) = current.batch_scenarios.iter().find(|s| s.name == base.name) else {
            report.hard_failures.push(format!(
                "{}: batch scenario missing from current run",
                base.name
            ));
            continue;
        };

        if cur.q != base.q || cur.success_count != base.success_count {
            report.hard_failures.push(format!(
                "{}: q/success_count changed {}/{} -> {}/{}; batch outcomes changed",
                base.name, base.q, base.success_count, cur.q, cur.success_count
            ));
        }
        let obj_tol = OBJECTIVE_REL_TOL * base.objective_sum.abs().max(1.0);
        if (cur.objective_sum - base.objective_sum).abs() > obj_tol {
            report.hard_failures.push(format!(
                "{}: objective_sum drifted {} -> {} (tol {obj_tol:.3e}); batch numerical \
                 behavior changed",
                base.name, base.objective_sum, cur.objective_sum
            ));
        }
        if cur.allocs_per_column != base.allocs_per_column
            || cur.bytes_per_column != base.bytes_per_column
        {
            report.hard_failures.push(format!(
                "{}: per-column allocation changed {} allocs/{} bytes -> {} allocs/{} bytes \
                 (regenerate the baseline only if this is an intentional change)",
                base.name,
                base.allocs_per_column,
                base.bytes_per_column,
                cur.allocs_per_column,
                cur.bytes_per_column
            ));
        }
        let ratio = cur.total_median_ms / base.total_median_ms;
        if ratio > TIME_RATIO_BAND {
            report.soft_warnings.push(format!(
                "{}: total_median_ms {:.3} -> {:.3} ({:.2}x, band {TIME_RATIO_BAND}x)",
                base.name, base.total_median_ms, cur.total_median_ms, ratio
            ));
        } else if ratio < 1.0 / TIME_RATIO_BAND {
            report.soft_warnings.push(format!(
                "{}: total_median_ms improved {:.3} -> {:.3} ({:.2}x) — consider refreshing \
                 the baseline to tighten the gate",
                base.name, base.total_median_ms, cur.total_median_ms, ratio
            ));
        }
    }

    for cur in &current.batch_scenarios {
        if !baseline.batch_scenarios.iter().any(|s| s.name == cur.name) {
            report.hard_failures.push(format!(
                "{}: batch scenario missing from baseline — regenerate the baseline",
                cur.name
            ));
        }
    }

    report
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let write_baseline = args.iter().any(|a| a == "--write-baseline");
    let strict_time = args.iter().any(|a| a == "--strict-time");
    if let Some(unknown) = args
        .iter()
        .find(|a| *a != "--write-baseline" && *a != "--strict-time")
    {
        eprintln!("unknown argument '{unknown}' (expected --write-baseline or --strict-time)");
        std::process::exit(2);
    }

    let current = Baseline {
        schema_version: SCHEMA_VERSION,
        profile: compile_profile().to_string(),
        scenarios: scenarios()
            .into_iter()
            .map(|s| {
                eprintln!("running {} ...", s.name);
                run_scenario(s)
            })
            .collect(),
        batch_scenarios: run_batch_scenarios(),
    };

    let path = baseline_path();
    if write_baseline {
        std::fs::create_dir_all(path.parent().unwrap()).expect("create benchmarks/ failed");
        std::fs::write(&path, serde_json::to_string_pretty(&current).unwrap())
            .expect("write baseline failed");
        println!("baseline written to {}", path.display());
        return;
    }

    let baseline_text = std::fs::read_to_string(&path).unwrap_or_else(|err| {
        eprintln!(
            "cannot read baseline {} ({err}); generate one with --write-baseline",
            path.display()
        );
        std::process::exit(2);
    });
    let baseline: Baseline = serde_json::from_str(&baseline_text).expect("baseline parse failed");

    let report = compare(&baseline, &current);

    println!(
        "{}",
        serde_json::to_string_pretty(&current).expect("serialize current metrics")
    );
    for warning in &report.soft_warnings {
        println!("TIME: {warning}");
    }
    for failure in &report.hard_failures {
        println!("FAIL: {failure}");
    }

    let hard_fail =
        !report.hard_failures.is_empty() || (strict_time && !report.soft_warnings.is_empty());
    if hard_fail {
        println!("perf gate: FAILED");
        std::process::exit(1);
    }
    println!(
        "perf gate: OK ({} eval + {} batch scenarios, {} timing notes)",
        current.scenarios.len(),
        current.batch_scenarios.len(),
        report.soft_warnings.len()
    );
}
