//! Benchmark: fit LMMs on simulated sleepstudy-like data at various sizes.
//! Outputs CSV timing results for comparison with Julia.

use std::time::Instant;

use mixedmodels::formula::parse_formula;
use mixedmodels::model::data::DataFrame;
use mixedmodels::model::linear::LinearMixedModel;
use mixedmodels::model::traits::MixedModelFit;

/// Simulate sleepstudy-like data:
///   y = β₀ + β₁*x + b₀ᵢ + b₁ᵢ*x + ε
fn simulate_data(n_subjects: usize, n_obs_per_subject: usize, seed: u64) -> DataFrame {
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};

    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();

    let beta = [250.0, 10.0];
    let sigma = 25.0;
    // RE Cholesky factor: SD_intercept=24, SD_slope=5.5, corr~0.07
    let lambda = [[24.0, 0.0], [1.68, 5.23]];

    let total_n = n_subjects * n_obs_per_subject;
    let mut reaction = Vec::with_capacity(total_n);
    let mut days = Vec::with_capacity(total_n);
    let mut subj_labels = Vec::with_capacity(total_n);

    for i in 0..n_subjects {
        // Draw random effects for this subject
        let u0 = normal.sample(&mut rng);
        let u1 = normal.sample(&mut rng);
        let b0 = lambda[0][0] * u0;
        let b1 = lambda[1][0] * u0 + lambda[1][1] * u1;

        let label = format!("S{:04}", i + 1);
        for d in 0..n_obs_per_subject {
            let x = d as f64;
            let mu = beta[0] + beta[1] * x + b0 + b1 * x;
            let y = mu + sigma * normal.sample(&mut rng);
            reaction.push(y);
            days.push(x);
            subj_labels.push(label.clone());
        }
    }

    let mut df = DataFrame::new();
    df.add_numeric("reaction", reaction);
    df.add_numeric("days", days);
    df.add_categorical("subj", subj_labels);
    df
}

fn bench_fit(
    data: &DataFrame,
    formula_str: &str,
    n_warmup: usize,
    n_reps: usize,
    reml: bool,
) -> (Vec<f64>, Vec<f64>) {
    let formula = parse_formula(formula_str).expect("formula parse failed");

    // Warmup
    for _ in 0..n_warmup {
        let mut m = LinearMixedModel::new(formula.clone(), data, None).unwrap();
        let _ = m.fit(reml);
    }

    let mut times_ms = Vec::with_capacity(n_reps);
    let mut objectives = Vec::with_capacity(n_reps);

    for _ in 0..n_reps {
        let start = Instant::now();
        let mut m = LinearMixedModel::new(formula.clone(), data, None).unwrap();
        match m.fit(reml) {
            Ok(_) => {
                let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                times_ms.push(elapsed);
                objectives.push(m.objective());
            }
            Err(e) => {
                eprintln!("Fit failed: {}", e);
                times_ms.push(f64::NAN);
                objectives.push(f64::NAN);
            }
        }
    }

    (times_ms, objectives)
}

fn median(v: &[f64]) -> f64 {
    let mut sorted: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if sorted.is_empty() {
        return f64::NAN;
    }
    let n = sorted.len();
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

fn mean(v: &[f64]) -> f64 {
    let finite: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if finite.is_empty() {
        return f64::NAN;
    }
    finite.iter().sum::<f64>() / finite.len() as f64
}

fn main() {
    let scenarios: Vec<(usize, usize, &str)> = vec![
        (18, 10, "sleepstudy_like"),
        (50, 10, "medium_50subj"),
        (100, 10, "medium_100subj"),
        (200, 10, "large_200subj"),
        (500, 10, "large_500subj"),
        (1000, 10, "xlarge_1000subj"),
        (50, 50, "deep_50x50"),
        (100, 50, "deep_100x50"),
        (200, 50, "deep_200x50"),
    ];

    // Vector-valued RE: (1 + days | subj)
    println!("# Vector RE: reaction ~ 1 + days + (1 + days | subj)");
    println!("scenario,n_subjects,n_obs,total_n,median_ms,mean_ms,min_ms,objective");

    for &(n_subj, n_obs, label) in &scenarios {
        let data = simulate_data(n_subj, n_obs, 42);
        let total_n = n_subj * n_obs;

        let (times, objs) = bench_fit(
            &data,
            "reaction ~ 1 + days + (1 + days | subj)",
            2,
            7,
            true,
        );

        let med = median(&times);
        let mn = mean(&times);
        let mi = times
            .iter()
            .copied()
            .filter(|x| x.is_finite())
            .fold(f64::INFINITY, f64::min);
        let obj = mean(&objs);

        println!(
            "{},{},{},{},{:.3},{:.3},{:.3},{:.6}",
            label, n_subj, n_obs, total_n, med, mn, mi, obj
        );
    }

    // Scalar RE: (1 | subj)
    println!("\n# Scalar RE: reaction ~ 1 + days + (1 | subj)");
    println!("scenario,n_subjects,n_obs,total_n,median_ms,mean_ms,min_ms,objective");

    for &(n_subj, n_obs, label) in &scenarios {
        let data = simulate_data(n_subj, n_obs, 42);
        let total_n = n_subj * n_obs;

        let (times, objs) = bench_fit(
            &data,
            "reaction ~ 1 + days + (1 | subj)",
            2,
            7,
            true,
        );

        let med = median(&times);
        let mn = mean(&times);
        let mi = times
            .iter()
            .copied()
            .filter(|x| x.is_finite())
            .fold(f64::INFINITY, f64::min);
        let obj = mean(&objs);

        println!(
            "scalar_{},{},{},{},{:.3},{:.3},{:.3},{:.6}",
            label, n_subj, n_obs, total_n, med, mn, mi, obj
        );
    }
}
