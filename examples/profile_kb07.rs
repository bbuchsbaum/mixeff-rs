//! Instrument the kb07 baseline LMM end-to-end to attribute its ~2 ms cost.
//!
//! Three phases are timed per run:
//!   parse  → `parse_formula`
//!   build  → `LinearMixedModel::new` (FE matrix, RE terms, AL blocks, parmap)
//!   fit    → `.fit(reml=true)` — also captures optimizer feval count
//!
//! We do a few warm-up runs first (so the allocator settles, NLopt's lazy
//! init pays its tax once, etc.) and then report median + min + max + per-
//! evaluation cost over `WARM_RUNS` measured iterations. Median is the
//! robust "typical fit" number; min is the best-case kernel speed.
//!
//! Run:
//!     cargo run --release --example profile_kb07

use std::time::Instant;

use mixedmodels::datasets;
use mixedmodels::formula::parse_formula;
use mixedmodels::model::linear::LinearMixedModel;
use mixedmodels::model::traits::MixedModelFit;

const WARMUP_RUNS: usize = 5;
const WARM_RUNS: usize = 50;

const FORMULA: &str = "rt_trunc ~ 1 + spkr + prec + load + (1 | subj) + (1 | item)";

fn pct(n: f64, total: f64) -> f64 {
    if total > 0.0 {
        100.0 * n / total
    } else {
        0.0
    }
}

fn percentile(samples: &mut [f64], q: f64) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((samples.len() - 1) as f64 * q).round() as usize;
    samples[idx]
}

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (df, _) = datasets::load("kb07")?;
    println!(
        "kb07: n = {}, formula = {FORMULA}\nwarmup = {WARMUP_RUNS}, measured = {WARM_RUNS}\n",
        df.nrow()
    );

    // Warm-up: do not record.
    for _ in 0..WARMUP_RUNS {
        let f = parse_formula(FORMULA)?;
        let mut m = LinearMixedModel::new(f, &df, None)?;
        m.fit(true)?;
    }

    let mut t_parse: Vec<f64> = Vec::with_capacity(WARM_RUNS);
    let mut t_build: Vec<f64> = Vec::with_capacity(WARM_RUNS);
    let mut t_fit: Vec<f64> = Vec::with_capacity(WARM_RUNS);
    let mut t_total: Vec<f64> = Vec::with_capacity(WARM_RUNS);
    let mut fevals: Vec<i64> = Vec::with_capacity(WARM_RUNS);
    let mut last_objective = 0.0;
    let mut last_optimizer = String::new();

    for _ in 0..WARM_RUNS {
        let t0 = Instant::now();

        let t_p = Instant::now();
        let formula = parse_formula(FORMULA)?;
        let parse_ms = ms(t_p.elapsed());

        let t_b = Instant::now();
        let mut model = LinearMixedModel::new(formula, &df, None)?;
        let build_ms = ms(t_b.elapsed());

        let t_f = Instant::now();
        model.fit(true)?;
        let fit_ms = ms(t_f.elapsed());

        let total_ms = ms(t0.elapsed());

        t_parse.push(parse_ms);
        t_build.push(build_ms);
        t_fit.push(fit_ms);
        t_total.push(total_ms);

        let opt = model.opt_summary();
        fevals.push(opt.feval);
        last_objective = opt.fmin;
        last_optimizer = opt.optimizer_name().to_string();
    }

    // Reporting
    let med_total = percentile(&mut t_total.clone(), 0.5);

    let print_phase = |label: &str, mut samples: Vec<f64>| {
        let min = percentile(&mut samples.clone(), 0.0);
        let p50 = percentile(&mut samples, 0.5);
        let max = percentile(&mut samples.clone(), 1.0);
        println!(
            "  {:<10}  median {:>6.3} ms  min {:>6.3}  max {:>6.3}   ({:>5.1}% of total)",
            label,
            p50,
            min,
            max,
            pct(p50, med_total)
        );
    };

    println!("Phase timings (n = {WARM_RUNS}):");
    print_phase("parse", t_parse.clone());
    print_phase("build", t_build.clone());
    print_phase("fit", t_fit.clone());
    println!(
        "  {:<10}  median {:>6.3} ms  (sum of phases above; small overhead from per-call timer ≈ 0)",
        "TOTAL", med_total
    );

    // Optimizer accounting
    let med_feval = {
        let mut v = fevals.iter().map(|&x| x as f64).collect::<Vec<_>>();
        percentile(&mut v, 0.5) as i64
    };
    let med_fit = percentile(&mut t_fit.clone(), 0.5);
    let per_eval_us = if med_feval > 0 {
        med_fit * 1000.0 / med_feval as f64
    } else {
        0.0
    };
    println!(
        "\nOptimizer ({}): median {} fevals  ⇒  ~{:.1} μs per evaluation",
        last_optimizer, med_feval, per_eval_us
    );

    // Sanity-check the fit converged.
    println!(
        "\nFinal objective (last run): {:.4}  (kb07 baseline REML reference: ~28785.85)",
        last_objective
    );

    println!(
        "\nInterpretation hints:
  - parse  : pure string work, should be <0.1 ms
  - build  : FE matrix, RE term construction, AL block gemm, parmap.
             If this dominates, optimizing construction is the lever.
  - fit    : optimizer iterations × per-eval PLS cost.
             If this dominates, the optimizer choice / per-eval kernel is the lever."
    );
    Ok(())
}
