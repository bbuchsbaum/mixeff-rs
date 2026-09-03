//! Experimental CMA-ES probe for the profiled LMM covariance objective.
//!
//! This is deliberately not production code. It asks a narrow empirical
//! question: on the same deterministic objectives used by
//! `optimizer_bench_harness`, can CMA-ES reach the repository's accepted
//! objective band with fewer evaluations than TrustBQ/BOBYQA, and does a
//! CMA-ES -> TrustBQ hybrid help?

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::time::Instant;

use cmaes::{CMAESOptions, DVector};
use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::data::DataFrame;
use mixeff_rs::model::linear::{FitOptions, LinearMixedModel, OptimizerControl};
use mixeff_rs::model::traits::MixedModelFit;
use mixeff_rs::types::Optimizer;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

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
    kind: ScenarioKind,
    reference: f64,
}

fn simulate_sleepstudy_like(n_subjects: usize, n_obs_per_subject: usize, seed: u64) -> DataFrame {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).expect("unit normal");
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

    let mut data = DataFrame::new();
    data.add_numeric("reaction", reaction).unwrap();
    data.add_numeric("days", days).unwrap();
    data.add_categorical("subj", subj_labels).unwrap();
    data
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

    let mut data = DataFrame::new();
    data.add_numeric("reaction", reaction).unwrap();
    data.add_numeric("days", days).unwrap();
    data.add_categorical("subj", subj_labels).unwrap();
    data.add_categorical("item", item_labels).unwrap();
    data.add_categorical("site", site_labels).unwrap();
    data
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

fn scenarios() -> [Scenario; 2] {
    [
        Scenario {
            name: "vector_1000",
            formula: "reaction ~ 1 + days + (1 + days | subj)",
            kind: ScenarioKind::Sleepstudy {
                n_subjects: 100,
                n_obs: 10,
                seed: 42,
            },
            reference: 9688.227799,
        },
        Scenario {
            name: "crossed_small",
            formula: "reaction ~ 1 + days + (1 + days | subj) + (1 + days | item) + (1 + days | site)",
            kind: ScenarioKind::Crossed {
                n_subjects: 18,
                n_items: 12,
                n_sites: 6,
                n_rep: 4,
            },
            reference: 6177.391766,
        },
    ]
}

#[derive(Debug)]
struct CmaOutcome {
    best_theta: Vec<f64>,
    best_objective: f64,
    evaluations: usize,
    first_target_eval: Option<usize>,
    wall_ms: f64,
    reasons: String,
}

fn run_cmaes(
    scenario: Scenario,
    data: &DataFrame,
    seed: u64,
    sigma: f64,
    budget: usize,
) -> CmaOutcome {
    let formula = parse_formula(scenario.formula).unwrap();
    let mut model = LinearMixedModel::new(formula, data, None).unwrap();
    model.optsum_mut().reml = true;

    let initial = model.theta();
    let native_lower = model.lower_bounds();
    // These deterministic benchmark optima are all inside this generous box.
    // Finite bounds keep CMA-ES's rejection sampler well-defined while still
    // allowing correlations/off-diagonal Cholesky elements to change sign.
    let lower = native_lower
        .iter()
        .map(|bound| if bound.is_finite() { *bound } else { -5.0 })
        .collect::<Vec<_>>();
    let upper = vec![5.0; initial.len()];
    let objective_tolerance = 1e-6 * (1.0 + scenario.reference.abs());
    let target = scenario.reference + objective_tolerance;

    let evaluations = Rc::new(Cell::new(0usize));
    let first_target_eval = Rc::new(Cell::new(None::<usize>));
    let best_seen = Rc::new(Cell::new(f64::INFINITY));
    let best_theta = Rc::new(RefCell::new(initial.clone()));

    let eval_counter = Rc::clone(&evaluations);
    let target_counter = Rc::clone(&first_target_eval);
    let best_value = Rc::clone(&best_seen);
    let best_point = Rc::clone(&best_theta);
    let objective = move |point: &DVector<f64>| {
        let n = eval_counter.get() + 1;
        eval_counter.set(n);
        let value = model
            .objective_at(point.as_slice())
            .ok()
            .filter(|value| value.is_finite())
            .unwrap_or(1.0e300);
        if value < best_value.get() {
            best_value.set(value);
            *best_point.borrow_mut() = point.as_slice().to_vec();
        }
        if value <= target && target_counter.get().is_none() {
            target_counter.set(Some(n));
        }
        value
    };

    let start = Instant::now();
    let mut state = CMAESOptions::new(initial, sigma)
        .bounds(lower, upper)
        .max_resamples(Some(1000))
        .max_function_evals(budget)
        .tol_fun(1e-10)
        .tol_fun_hist(1e-10)
        .tol_x(1e-8)
        .seed(seed)
        .build(objective)
        .expect("valid CMA-ES configuration");
    let termination = state.run();
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
    let reasons = termination
        .reasons
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("|");

    let best_theta_value = best_theta.borrow().clone();
    let best_objective = best_seen.get();
    let evaluation_count = evaluations.get();
    let first_target = first_target_eval.get();
    CmaOutcome {
        best_theta: best_theta_value,
        best_objective,
        evaluations: evaluation_count,
        first_target_eval: first_target,
        wall_ms,
        reasons,
    }
}

fn polish_with_trust_bq(
    scenario: Scenario,
    data: &DataFrame,
    theta: Vec<f64>,
    max_feval: usize,
) -> (f64, i64, f64, String) {
    let formula = parse_formula(scenario.formula).unwrap();
    let mut model = LinearMixedModel::new(formula, data, None).unwrap();
    let options = FitOptions::reml().with_optimizer_control(
        OptimizerControl::auto()
            .with_optimizer(Optimizer::TrustBq)
            .with_start_theta(theta)
            .with_max_feval(max_feval),
    );
    let start = Instant::now();
    let fit = model.fit_with_options(options);
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
    match fit {
        Ok(_) => (
            model.objective(),
            model.optsum().feval,
            wall_ms,
            model.optsum().return_value.clone(),
        ),
        Err(error) => (f64::NAN, 0, wall_ms, format!("ERROR:{error}")),
    }
}

fn main() {
    println!("scenario,seed,sigma,cma_budget,cma_evals,cma_first_target_eval,cma_objective,cma_gap,cma_target_pass,cma_wall_ms,cma_stop,hybrid_polish_evals,hybrid_total_evals,hybrid_objective,hybrid_gap,hybrid_target_pass,hybrid_wall_ms,hybrid_stop");

    for scenario in scenarios() {
        let data = build_data(scenario.kind);
        let tolerance = 1e-6 * (1.0 + scenario.reference.abs());
        for sigma in [0.25, 0.5, 1.0] {
            for seed in 1..=5 {
                let budget = if scenario.name == "vector_1000" { 600 } else { 1200 };
                let cma = run_cmaes(scenario, &data, seed, sigma, budget);
                let cma_gap = cma.best_objective - scenario.reference;
                let cma_pass = cma.best_objective <= scenario.reference + tolerance;
                let (hybrid_objective, polish_evals, polish_ms, hybrid_stop) =
                    polish_with_trust_bq(scenario, &data, cma.best_theta.clone(), 500);
                let hybrid_gap = hybrid_objective - scenario.reference;
                let hybrid_pass = hybrid_objective <= scenario.reference + tolerance;
                let hybrid_total = cma.evaluations + polish_evals.max(0) as usize;
                println!(
                    "{},{},{:.2},{},{},{},{:.9},{:.9},{},{:.6},{},{},{},{:.9},{:.9},{},{:.6},{}",
                    scenario.name,
                    seed,
                    sigma,
                    budget,
                    cma.evaluations,
                    cma.first_target_eval.map_or_else(String::new, |value| value.to_string()),
                    cma.best_objective,
                    cma_gap,
                    cma_pass,
                    cma.wall_ms,
                    cma.reasons.replace(',', ";"),
                    polish_evals,
                    hybrid_total,
                    hybrid_objective,
                    hybrid_gap,
                    hybrid_pass,
                    cma.wall_ms + polish_ms,
                    hybrid_stop.replace(',', ";"),
                );
            }
        }
    }
}
