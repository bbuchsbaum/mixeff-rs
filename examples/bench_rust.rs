//! Benchmark: fit LMMs on simulated sleepstudy-like data at various sizes.
//! Outputs CSV timing results for comparison with Julia.

use std::time::Instant;

use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::data::DataFrame;
use mixeff_rs::model::linear::LinearMixedModel;
use mixeff_rs::model::traits::MixedModelFit;

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
    df.add_numeric("reaction", reaction).unwrap();
    df.add_numeric("days", days).unwrap();
    df.add_categorical("subj", subj_labels).unwrap();
    df
}

fn centered_mod(value: usize, modulus: usize, center: f64, scale: f64) -> f64 {
    ((value % modulus) as f64 - center) * scale
}

/// Deterministic crossed-effects benchmark with 9 covariance parameters:
/// (1 + days | subj) + (1 + days | item) + (1 + days | site)
fn simulate_large_theta_data(
    n_subjects: usize,
    n_items: usize,
    n_sites: usize,
    n_rep: usize,
) -> DataFrame {
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

fn bench_fit_with_feval(
    data: &DataFrame,
    formula_str: &str,
    n_warmup: usize,
    n_reps: usize,
    reml: bool,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let formula = parse_formula(formula_str).expect("formula parse failed");

    for _ in 0..n_warmup {
        let mut m = LinearMixedModel::new(formula.clone(), data, None).unwrap();
        let _ = m.fit(reml);
    }

    let mut times_ms = Vec::with_capacity(n_reps);
    let mut objectives = Vec::with_capacity(n_reps);
    let mut fevals = Vec::with_capacity(n_reps);

    for _ in 0..n_reps {
        let start = Instant::now();
        let mut m = LinearMixedModel::new(formula.clone(), data, None).unwrap();
        match m.fit(reml) {
            Ok(_) => {
                let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                times_ms.push(elapsed);
                objectives.push(m.objective());
                fevals.push(m.optsum().feval as f64);
            }
            Err(e) => {
                eprintln!("Fit failed: {}", e);
                times_ms.push(f64::NAN);
                objectives.push(f64::NAN);
                fevals.push(f64::NAN);
            }
        }
    }

    (times_ms, objectives, fevals)
}

#[allow(clippy::type_complexity)]
fn bench_fit_with_breakdown(
    data: &DataFrame,
    formula_str: &str,
    n_warmup: usize,
    n_reps: usize,
    reml: bool,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let formula = parse_formula(formula_str).expect("formula parse failed");

    for _ in 0..n_warmup {
        let mut m = LinearMixedModel::new(formula.clone(), data, None).unwrap();
        let _ = m.fit(reml);
    }

    let mut totals_ms = Vec::with_capacity(n_reps);
    let mut build_ms = Vec::with_capacity(n_reps);
    let mut fit_ms = Vec::with_capacity(n_reps);
    let mut objectives = Vec::with_capacity(n_reps);
    let mut fevals = Vec::with_capacity(n_reps);

    for _ in 0..n_reps {
        let total_start = Instant::now();
        let build_start = Instant::now();
        let mut m = LinearMixedModel::new(formula.clone(), data, None).unwrap();
        let build_elapsed = build_start.elapsed().as_secs_f64() * 1000.0;

        let fit_start = Instant::now();
        match m.fit(reml) {
            Ok(_) => {
                fit_ms.push(fit_start.elapsed().as_secs_f64() * 1000.0);
                totals_ms.push(total_start.elapsed().as_secs_f64() * 1000.0);
                build_ms.push(build_elapsed);
                objectives.push(m.objective());
                fevals.push(m.optsum().feval as f64);
            }
            Err(e) => {
                eprintln!("Fit failed: {}", e);
                fit_ms.push(f64::NAN);
                totals_ms.push(f64::NAN);
                build_ms.push(build_elapsed);
                objectives.push(f64::NAN);
                fevals.push(f64::NAN);
            }
        }
    }

    (totals_ms, build_ms, fit_ms, objectives, fevals)
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

        let (times, objs) = bench_fit(&data, "reaction ~ 1 + days + (1 + days | subj)", 2, 7, true);

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
    println!("scenario,n_subjects,n_obs,total_n,median_ms,mean_ms,min_ms,build_median_ms,fit_median_ms,objective,median_feval,mean_feval");

    for &(n_subj, n_obs, label) in &scenarios {
        let data = simulate_data(n_subj, n_obs, 42);
        let total_n = n_subj * n_obs;

        let (times, builds, fits, objs, fevals) =
            bench_fit_with_breakdown(&data, "reaction ~ 1 + days + (1 | subj)", 2, 7, true);

        let med = median(&times);
        let mn = mean(&times);
        let mi = times
            .iter()
            .copied()
            .filter(|x| x.is_finite())
            .fold(f64::INFINITY, f64::min);
        let obj = mean(&objs);
        let build_med = median(&builds);
        let fit_med = median(&fits);
        let fe_med = median(&fevals);
        let fe_mean = mean(&fevals);

        println!(
            "scalar_{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.6},{:.1},{:.1}",
            label, n_subj, n_obs, total_n, med, mn, mi, build_med, fit_med, obj, fe_med, fe_mean
        );
    }

    let large_theta_scenarios: Vec<(usize, usize, usize, usize, &str)> = vec![
        (18, 12, 6, 4, "crossed_small"),
        (36, 24, 8, 4, "crossed_medium"),
        (72, 36, 12, 4, "crossed_large"),
    ];

    println!(
        "\n# Large-theta RE: reaction ~ 1 + days + (1 + days | subj) + (1 + days | item) + (1 + days | site)"
    );
    println!(
        "scenario,n_subjects,n_items,n_sites,n_rep,total_n,median_ms,mean_ms,min_ms,objective,median_feval,mean_feval"
    );

    for &(n_subj, n_items, n_sites, n_rep, label) in &large_theta_scenarios {
        let data = simulate_large_theta_data(n_subj, n_items, n_sites, n_rep);
        let total_n = n_subj * n_items * n_rep;

        let (times, objs, fevals) = bench_fit_with_feval(
            &data,
            "reaction ~ 1 + days + (1 + days | subj) + (1 + days | item) + (1 + days | site)",
            1,
            5,
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
        let fe_med = median(&fevals);
        let fe_mean = mean(&fevals);

        println!(
            "{},{},{},{},{},{},{:.3},{:.3},{:.3},{:.6},{:.1},{:.1}",
            label, n_subj, n_items, n_sites, n_rep, total_n, med, mn, mi, obj, fe_med, fe_mean
        );
    }
}
