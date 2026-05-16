//! Criterion benches for the two hot paths of the LMM fit.
//!
//! - `lmm_fit_sleepstudy`: end-to-end `fit(false)` (PLS kernel + optimizer)
//!   on a deterministic sleepstudy-shaped dataset.
//! - `theta_objective_loop`: a single `objective_at(theta)` evaluation — the
//!   inner callback the optimizer drives thousands of times per fit.
//!
//! Run with `cargo bench --features unstable-internals`. Output is text-only
//! (criterion is built without plotters), so it is portable and CI-friendly.
//! Compare runs by archiving criterion's own `target/criterion` estimates
//! between commits.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::{DataFrame, LinearMixedModel, MixedModelFit};

/// Deterministic sleepstudy-shaped data: `reaction ~ 1 + days + (1 + days | subj)`.
fn simulate_data(n_subjects: usize, n_obs_per_subject: usize) -> DataFrame {
    let beta = [250.0, 10.0];
    let sigma = 25.0;
    let lambda = [[24.0, 0.0], [1.68, 5.23]];

    let total_n = n_subjects * n_obs_per_subject;
    let mut reaction = Vec::with_capacity(total_n);
    let mut days = Vec::with_capacity(total_n);
    let mut subj = Vec::with_capacity(total_n);

    for i in 0..n_subjects {
        // Deterministic pseudo-normal draws (no rng dep needed for a bench).
        let u0 = ((i as f64 * 12.9898).sin() * 43758.547).fract() - 0.5;
        let u1 = ((i as f64 * 78.233).sin() * 12543.137).fract() - 0.5;
        let b0 = lambda[0][0] * u0;
        let b1 = lambda[1][0] * u0 + lambda[1][1] * u1;

        let label = format!("S{:04}", i + 1);
        for d in 0..n_obs_per_subject {
            let x = d as f64;
            let noise =
                ((((i * n_obs_per_subject + d) as f64) * 31.7).sin() * 9817.21).fract() - 0.5;
            reaction.push(beta[0] + beta[1] * x + b0 + b1 * x + sigma * noise);
            days.push(x);
            subj.push(label.clone());
        }
    }

    let mut df = DataFrame::new();
    df.add_numeric("reaction", reaction).unwrap();
    df.add_numeric("days", days).unwrap();
    df.add_categorical("subj", subj).unwrap();
    df
}

fn bench_fit(c: &mut Criterion) {
    let df = simulate_data(18, 10);
    c.bench_function("lmm_fit_sleepstudy", |b| {
        b.iter(|| {
            let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
            let mut model = LinearMixedModel::new(formula, &df, None).unwrap();
            model.fit(black_box(false)).unwrap();
            black_box(model.objective())
        });
    });
}

fn bench_theta_objective(c: &mut Criterion) {
    let df = simulate_data(18, 10);
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &df, None).unwrap();
    let theta0 = model.theta();

    c.bench_function("theta_objective_loop", |b| {
        b.iter(|| {
            // The optimizer's inner callback: evaluate the profiled objective
            // at a θ. Perturb slightly so the cost is not trivially cached.
            let theta: Vec<f64> = theta0.iter().map(|t| t + 0.01).collect();
            black_box(model.objective_at(black_box(&theta)).unwrap())
        });
    });
}

criterion_group!(benches, bench_fit, bench_theta_objective);
criterion_main!(benches);
