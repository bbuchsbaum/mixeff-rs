//! Focused profiler harness for the crossed scalar Poisson GLMM parity row.
//!
//! Run a long enough loop to sample with macOS `sample`:
//!
//!     MIXEDMODELS_PROFILE_REPEATS=200 cargo run --release --example profile_grouseticks_glmm

use std::time::Instant;

use mixedmodels::datasets;
use mixedmodels::formula::parse_formula;
use mixedmodels::model::generalized::GeneralizedLinearMixedModel;
use mixedmodels::model::traits::{Family, LinkFunction, MixedModelFit};
use mixedmodels::types::Optimizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let repeats = std::env::var("MIXEDMODELS_PROFILE_REPEATS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(100);
    let (data, _) = datasets::load("grouseticks")?;
    let formula = "TICKS ~ 1 + YEAR + cHEIGHT + (1 | BROOD) + (1 | INDEX) + (1 | LOCATION)";
    let mut times = Vec::with_capacity(repeats);

    eprintln!("pid={} repeats={repeats}", std::process::id());
    for iter in 0..repeats {
        let start = Instant::now();
        let parsed = parse_formula(formula)?;
        let mut model = GeneralizedLinearMixedModel::new(
            parsed,
            &data,
            Family::Poisson,
            Some(LinkFunction::Log),
        )?;
        if std::env::var("MIXEDMODELS_PROFILE_PATTERN_SEARCH").is_ok() {
            model.lmm.optsum.optimizer = Optimizer::PatternSearch;
        }
        model.fit_with_options(true, 1, false)?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        times.push(elapsed_ms);
        if iter == 0 || (iter + 1) % 10 == 0 {
            let opt = MixedModelFit::opt_summary(&model);
            eprintln!(
                "iter={} ms={:.3} fevals={} obj={:.6}",
                iter + 1,
                elapsed_ms,
                opt.feval,
                MixedModelFit::objective(&model)
            );
        }
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let min = times.first().copied().unwrap_or(f64::NAN);
    let median = times[times.len() / 2];
    let max = times.last().copied().unwrap_or(f64::NAN);
    println!("grouseticks repeats={repeats} min_ms={min:.3} median_ms={median:.3} max_ms={max:.3}");
    Ok(())
}
