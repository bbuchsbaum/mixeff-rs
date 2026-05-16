//! Isolated profiled-objective evaluation benchmark.
//!
//! This harness builds and fits each model once, then repeatedly evaluates the
//! public `LinearMixedModel::objective_at` path at the fitted theta. That keeps
//! model construction and optimizer search out of the timing loop, so the CSV
//! primarily reflects repeated PLS/factorization objective cost.
//!
//! ```sh
//! cargo run --release --example objective_eval_bench
//! cargo run --release --no-default-features --example objective_eval_bench
//! ```

use std::hint::black_box;
use std::time::Instant;

use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::data::DataFrame;
use mixeff_rs::model::linear::LinearMixedModel;
use mixeff_rs::model::traits::MixedModelFit;

const BATCHES: usize = 7;
const WARMUP_EVALS: usize = 25;

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
    scenario: &'static str,
    family: &'static str,
    formula: &'static str,
    reml: bool,
    kind: ScenarioKind,
}

struct EvalStats {
    median_us: f64,
    mean_us: f64,
    min_us: f64,
    objective: f64,
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
            scenario: "vector_1000",
            family: "vector",
            formula: vector_formula,
            reml: true,
            kind: ScenarioKind::Sleepstudy {
                n_subjects: 100,
                n_obs: 10,
                seed: 42,
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
        },
    ]
}

fn build_data(kind: ScenarioKind) -> (DataFrame, usize, u64) {
    match kind {
        ScenarioKind::Sleepstudy {
            n_subjects,
            n_obs,
            seed,
        } => (
            simulate_sleepstudy_like(n_subjects, n_obs, seed),
            n_subjects * n_obs,
            seed,
        ),
        ScenarioKind::Crossed {
            n_subjects,
            n_items,
            n_sites,
            n_rep,
        } => (
            simulate_crossed(n_subjects, n_items, n_sites, n_rep),
            n_subjects * n_items * n_rep,
            0,
        ),
    }
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
    finite[finite.len() / 2]
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

fn compile_profile() -> &'static str {
    if cfg!(feature = "nlopt") {
        "rust_default_nlopt"
    } else {
        "rust_no_default_native"
    }
}

fn eval_reps(family: &str, total_n: usize) -> usize {
    match (family, total_n) {
        ("crossed", n) if n >= 10_000 => 350,
        ("crossed", n) if n >= 3_000 => 700,
        ("crossed", _) => 1_000,
        (_, n) if n >= 10_000 => 1_500,
        _ => 3_000,
    }
}

fn measure_objective_evals(
    model: &mut LinearMixedModel,
    theta: &[f64],
    reps: usize,
) -> Result<EvalStats, String> {
    for _ in 0..WARMUP_EVALS {
        black_box(
            model
                .objective_at(black_box(theta))
                .map_err(|err| err.to_string())?,
        );
    }

    let mut per_eval_us = Vec::with_capacity(BATCHES);
    let mut objective = f64::NAN;
    for _ in 0..BATCHES {
        let start = Instant::now();
        for _ in 0..reps {
            objective = black_box(
                model
                    .objective_at(black_box(theta))
                    .map_err(|err| err.to_string())?,
            );
        }
        per_eval_us.push(start.elapsed().as_secs_f64() * 1_000_000.0 / reps as f64);
    }

    Ok(EvalStats {
        median_us: median(&per_eval_us),
        mean_us: mean(&per_eval_us),
        min_us: min_finite(&per_eval_us),
        objective,
    })
}

fn csv_field(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn print_row(fields: Vec<String>) {
    println!("{}", fields.join(","));
}

fn main() {
    println!(
        "method,scenario,family,formula,reml,seed,n,p,q,n_reterms,d_theta,total_n,eval_path,reps,batches,optimizer,backend,status,fit_ms,fit_feval,fit_objective,eval_objective,objective_gap,eval_median_us,eval_mean_us,eval_min_us,fit_budget_eval_ms,evals_per_sec"
    );

    for scenario in scenarios() {
        let (data, total_n, seed) = build_data(scenario.kind);
        let formula = parse_formula(scenario.formula).expect("formula parse failed");
        let reps = eval_reps(scenario.family, total_n);

        let fit_start = Instant::now();
        let mut model = LinearMixedModel::new(formula, &data, None).expect("model build failed");
        model.fit(scenario.reml).expect("model fit failed");
        let fit_ms = fit_start.elapsed().as_secs_f64() * 1_000.0;

        let theta = model.theta();
        let fit_objective = model.objective();
        let fit_feval = model.optsum.feval as f64;
        let optimizer = model.optsum.optimizer_name().to_string();
        let backend = model.optsum.backend_name().to_string();
        let status = model.optsum.return_value.clone();
        let (n, p, q, n_reterms) = model.model_size();
        let d_theta = model.n_theta();

        let stats =
            measure_objective_evals(&mut model, &theta, reps).expect("objective eval failed");
        let objective_gap = stats.objective - fit_objective;
        let fit_budget_eval_ms = stats.median_us * fit_feval / 1_000.0;
        let evals_per_sec = 1_000_000.0 / stats.median_us;

        print_row(vec![
            compile_profile().to_string(),
            scenario.scenario.to_string(),
            scenario.family.to_string(),
            csv_field(scenario.formula),
            scenario.reml.to_string(),
            seed.to_string(),
            n.to_string(),
            p.to_string(),
            q.to_string(),
            n_reterms.to_string(),
            d_theta.to_string(),
            total_n.to_string(),
            "objective_at_public".to_string(),
            reps.to_string(),
            BATCHES.to_string(),
            optimizer,
            backend,
            csv_field(&status),
            format!("{fit_ms:.6}"),
            format!("{fit_feval:.1}"),
            format!("{fit_objective:.9}"),
            format!("{:.9}", stats.objective),
            format!("{objective_gap:.9}"),
            format!("{:.6}", stats.median_us),
            format!("{:.6}", stats.mean_us),
            format!("{:.6}", stats.min_us),
            format!("{fit_budget_eval_ms:.6}"),
            format!("{evals_per_sec:.3}"),
        ]);
    }
}
