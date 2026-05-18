//! Focused profiler for the known-slow `grouseticks` Poisson GLMM row.
//!
//! This example is intentionally narrow: it uses only the repo-owned
//! `datasets/grouseticks` fixture and times parse -> construct -> fit so the
//! result is comparable to the comparison harness' single-fit timing.
//!
//! ```text
//! MIXEFF_PROFILE_REPEATS=200 \
//!   cargo run --release --features unstable-internals --example profile_grouseticks_glmm
//! ```
//!
//! Set `MIXEFF_PROFILE_PATTERN_SEARCH=1` to force the native PatternSearch
//! backend while profiling.

use std::time::Instant;

use mixeff_rs::datasets;
use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::generalized::GeneralizedLinearMixedModel;
use mixeff_rs::model::traits::{Family, LinkFunction, MixedModelFit};
use mixeff_rs::types::Optimizer;

const FORMULA: &str = "TICKS ~ 1 + YEAR + cHEIGHT + (1 | BROOD) + (1 | INDEX) + (1 | LOCATION)";

fn env_usize(name: &str, fallback: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(fallback)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let repeats = env_usize("MIXEFF_PROFILE_REPEATS", 100);
    let legacy_repeats = env_usize("MIXEDMODELS_PROFILE_REPEATS", repeats);
    let repeats = legacy_repeats.max(1);
    let force_pattern_search = std::env::var("MIXEFF_PROFILE_PATTERN_SEARCH").is_ok()
        || std::env::var("MIXEDMODELS_PROFILE_PATTERN_SEARCH").is_ok();

    let (data, _) = datasets::load("grouseticks")?;
    let mut times = Vec::with_capacity(repeats);
    let mut last_summary = None;

    eprintln!(
        "pid={} dataset=grouseticks n={} repeats={} pattern_search={}",
        std::process::id(),
        data.nrow(),
        repeats,
        force_pattern_search
    );

    for iter in 0..repeats {
        let start = Instant::now();
        let parsed = parse_formula(FORMULA)?;
        let mut model = GeneralizedLinearMixedModel::new(
            parsed,
            &data,
            Family::Poisson,
            Some(LinkFunction::Log),
        )?;
        if force_pattern_search {
            model.lmm_mut().optsum_mut().optimizer = Optimizer::PatternSearch;
        }
        model.fit_with_options(true, 1, false)?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        times.push(elapsed_ms);

        let opt = MixedModelFit::opt_summary(&model);
        if iter == 0 || (iter + 1) % 10 == 0 {
            eprintln!(
                "iter={} ms={:.3} optimizer={} backend={} code={} fevals={} obj={:.6}",
                iter + 1,
                elapsed_ms,
                opt.optimizer_name(),
                opt.backend_name(),
                opt.return_value,
                opt.feval,
                MixedModelFit::objective(&model)
            );
        }

        last_summary = Some((
            MixedModelFit::coef_names(&model),
            MixedModelFit::coef(&model).as_slice().to_vec(),
            MixedModelFit::theta(&model),
            MixedModelFit::objective(&model),
            opt.optimizer_name().to_string(),
            opt.backend_name().to_string(),
            opt.return_value.clone(),
            opt.feval,
        ));
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let min = times.first().copied().unwrap_or(f64::NAN);
    let median = times[times.len() / 2];
    let max = times.last().copied().unwrap_or(f64::NAN);

    println!(
        "grouseticks repeats={} min_ms={:.3} median_ms={:.3} max_ms={:.3}",
        repeats, min, median, max
    );

    if let Some((names, beta, theta, objective, optimizer, backend, code, fevals)) = last_summary {
        println!(
            "optimizer={} backend={} code={} fevals={} objective={:.12}",
            optimizer, backend, code, fevals, objective
        );
        println!("theta={theta:?}");
        println!("coef_names={names:?}");
        println!("beta={beta:?}");
    }

    Ok(())
}
