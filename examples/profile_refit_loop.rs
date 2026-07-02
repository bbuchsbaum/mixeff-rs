//! Loop `refit` on a simulated response for ~15 seconds so an external
//! sampling profiler (e.g. macOS `sample`) can attribute per-replicate cost.
//!
//! ```sh
//! cargo run --release --features unstable-internals --example profile_refit_loop
//! ```

use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::SeedableRng;

use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::data::DataFrame;
use mixeff_rs::model::linear::LinearMixedModel;

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

fn main() {
    let data = simulate_sleepstudy_like(100, 10, 42);
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(true).unwrap();

    let mut rng = StdRng::seed_from_u64(2026);
    let y_sim = model.simulate(&mut rng);

    println!("pid: {}", std::process::id());
    let start = Instant::now();
    let mut reps = 0usize;
    while start.elapsed() < Duration::from_secs(15) {
        let mut work = model.clone();
        work.refit(y_sim.as_slice()).unwrap();
        reps += 1;
    }
    println!("reps: {reps}");
}
