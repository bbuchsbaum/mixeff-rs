//! Parametric-bootstrap refit cost decomposition.
//!
//! Times `parametricbootstrap` end-to-end, then decomposes one replicate into
//! its structural pieces (template clone, response simulation, A-block
//! recomputation, and re-optimization) so amortization work can be checked
//! against measured rebuild cost rather than intuition.
//!
//! ```sh
//! cargo run --release --features unstable-internals --example bootstrap_refit_bench
//! ```

use std::hint::black_box;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;

use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::data::DataFrame;
use mixeff_rs::model::linear::{parametricbootstrap, LinearMixedModel};

const BOOTSTRAP_REPS: usize = 200;
const COMPONENT_REPS: usize = 50;

#[derive(Clone, Copy)]
struct Scenario {
    name: &'static str,
    formula: &'static str,
    n_subjects: usize,
    n_obs: usize,
    seed: u64,
}

fn simulate_sleepstudy_like(n_subjects: usize, n_obs_per_subject: usize, seed: u64) -> DataFrame {
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

fn scenarios() -> Vec<Scenario> {
    vec![
        Scenario {
            name: "vector_1000",
            formula: "reaction ~ 1 + days + (1 + days | subj)",
            n_subjects: 100,
            n_obs: 10,
            seed: 42,
        },
        Scenario {
            name: "scalar_1000",
            formula: "reaction ~ 1 + days + (1 | subj)",
            n_subjects: 100,
            n_obs: 10,
            seed: 42,
        },
        Scenario {
            name: "vector_10000",
            formula: "reaction ~ 1 + days + (1 + days | subj)",
            n_subjects: 1000,
            n_obs: 10,
            seed: 42,
        },
    ]
}

fn time_us<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    let start = Instant::now();
    for _ in 0..reps {
        f();
    }
    start.elapsed().as_secs_f64() * 1_000_000.0 / reps as f64
}

fn main() {
    println!(
        "scenario,n,bootstrap_reps,bootstrap_total_ms,per_replicate_us,clone_us,simulate_us,recompute_a_us,refit_us,fit_feval,eval_us,eval_budget_us"
    );

    for scenario in scenarios() {
        let data = simulate_sleepstudy_like(scenario.n_subjects, scenario.n_obs, scenario.seed);
        let formula = parse_formula(scenario.formula).expect("formula parse failed");
        let mut model = LinearMixedModel::new(formula, &data, None).expect("model build failed");
        model.fit(true).expect("model fit failed");

        // End-to-end bootstrap timing.
        let mut rng = StdRng::seed_from_u64(2026);
        let start = Instant::now();
        let boot = parametricbootstrap(&mut rng, BOOTSTRAP_REPS, &model);
        let bootstrap_total_ms = start.elapsed().as_secs_f64() * 1_000.0;
        assert_eq!(boot.len(), BOOTSTRAP_REPS);
        let per_replicate_us = bootstrap_total_ms * 1_000.0 / BOOTSTRAP_REPS as f64;

        // Component timings.
        let clone_us = time_us(COMPONENT_REPS, || {
            black_box(model.clone());
        });

        let mut rng = StdRng::seed_from_u64(2026);
        let simulate_us = time_us(COMPONENT_REPS, || {
            black_box(model.simulate(&mut rng));
        });

        let mut work = model.clone();
        let recompute_a_us = time_us(COMPONENT_REPS, || {
            work.recompute_a_blocks().expect("recompute failed");
        });

        let mut rng = StdRng::seed_from_u64(2026);
        let y_sim = model.simulate(&mut rng);
        let mut refit_feval = 0.0;
        let refit_us = time_us(COMPONENT_REPS, || {
            let mut work = model.clone();
            work.refit(y_sim.as_slice()).expect("refit failed");
            refit_feval = work.optsum().feval as f64;
        }) - clone_us;

        let theta = model.theta();
        let eval_us = time_us(COMPONENT_REPS, || {
            black_box(model.objective_at(&theta).expect("objective failed"));
        });

        println!(
            "{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.1},{:.3},{:.3}",
            scenario.name,
            scenario.n_subjects * scenario.n_obs,
            BOOTSTRAP_REPS,
            bootstrap_total_ms,
            per_replicate_us,
            clone_us,
            simulate_us,
            recompute_a_us,
            refit_us,
            refit_feval,
            eval_us,
            eval_us * refit_feval,
        );
    }
}
