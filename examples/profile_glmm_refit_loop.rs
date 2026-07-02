//! Loop GLMM `refit` on a simulated response for ~15 seconds so an external
//! sampling profiler (e.g. macOS `sample`) can attribute per-replicate cost.
//!
//! ```sh
//! cargo run --release --features unstable-internals --example profile_glmm_refit_loop
//! ```

use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal, Poisson};

use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::data::DataFrame;
use mixeff_rs::model::generalized::GeneralizedLinearMixedModel;
use mixeff_rs::model::traits::{Family, LinkFunction};

fn simulate_poisson(n_groups: usize, n_obs: usize, seed: u64) -> DataFrame {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();

    let total_n = n_groups * n_obs;
    let mut y = Vec::with_capacity(total_n);
    let mut x = Vec::with_capacity(total_n);
    let mut group = Vec::with_capacity(total_n);

    for g in 0..n_groups {
        let b0 = 0.6 * normal.sample(&mut rng);
        let label = format!("G{:04}", g + 1);
        for j in 0..n_obs {
            let xv = (j as f64 - (n_obs as f64 - 1.0) / 2.0) / n_obs as f64;
            let eta = 1.0 + 0.5 * xv + b0;
            let mu = eta.exp();
            let count = Poisson::new(mu.max(1e-8)).unwrap().sample(&mut rng);
            y.push(count);
            x.push(xv);
            group.push(label.clone());
        }
    }

    let mut df = DataFrame::new();
    df.add_numeric("y", y).unwrap();
    df.add_numeric("x", x).unwrap();
    df.add_categorical("group", group).unwrap();
    df
}

fn main() {
    let data = simulate_poisson(100, 10, 42);
    let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();
    let mut model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Poisson, Some(LinkFunction::Log))
            .unwrap();
    model.fit().unwrap();

    let mut rng = StdRng::seed_from_u64(2026);
    let y_sim = model.simulate_response(&mut rng).unwrap();

    println!("pid: {}", std::process::id());
    let fit_start = Instant::now();
    {
        let mut work = model.clone();
        work.refit(y_sim.as_slice()).unwrap();
    }
    println!(
        "one refit: {:.3} ms",
        fit_start.elapsed().as_secs_f64() * 1_000.0
    );

    let start = Instant::now();
    let mut reps = 0usize;
    while start.elapsed() < Duration::from_secs(15) {
        let mut work = model.clone();
        work.refit(y_sim.as_slice()).unwrap();
        reps += 1;
    }
    println!("reps: {reps} ({:.3} ms/refit)", 15_000.0 / reps as f64);
}
