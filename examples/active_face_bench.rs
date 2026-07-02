//! Benchmark the experimental active-face refit on the checked-in singular
//! fixture rows (K13, bd-01KWFNE630AW0KKP2BPT8ZHETP).
//!
//! Fits the maximal over-specified row `y ~ 1 + A * B * C + (A * B * C |
//! group)` (REML) with and without `ActiveFaceRefit::Experimental` and
//! reports objective, evaluation count, wall time, and the audit status.
//! The lme4 reference for this row is objective 766.554 at 4027 evaluations
//! (`comparison/lme4_results.json`).
//!
//! Run with:
//!
//! ```bash
//! cargo run --release --features unstable-internals --example active_face_bench
//! ```

use std::time::Instant;

use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::{
    ActiveFaceRefit, FitOptions, LinearMixedModel, MixedModelFit, OptimizerControl,
};

const REPS: usize = 5;

fn fit_row(
    formula_text: &str,
    data: &mixeff_rs::model::DataFrame,
    control: OptimizerControl,
) -> (f64, i64, String, f64) {
    let mut best_ms = f64::INFINITY;
    let mut objective = f64::NAN;
    let mut feval = 0;
    let mut status = String::new();
    for _ in 0..REPS {
        let formula = parse_formula(formula_text).expect("formula parse failed");
        let mut model = LinearMixedModel::new(formula, data, None).expect("model build failed");
        let start = Instant::now();
        model
            .fit_with_options(FitOptions::reml().with_optimizer_control(control.clone()))
            .expect("fit failed");
        let elapsed_ms = start.elapsed().as_secs_f64() * 1e3;
        best_ms = best_ms.min(elapsed_ms);
        objective = model.objective();
        feval = model.optsum().feval;
        status = model.optsum().return_value.clone();
    }
    (objective, feval, status, best_ms)
}

fn main() {
    let (data, _) = mixeff_rs::datasets::load("singular").expect("singular fixture load failed");
    let formula = "y ~ 1 + A * B * C + (A * B * C | group)";

    println!("method,objective,feval,min_ms,status");
    for (method, control) in [
        ("default", OptimizerControl::auto()),
        (
            "active_face",
            OptimizerControl::auto().with_active_face_refit(ActiveFaceRefit::Experimental),
        ),
        (
            "active_face_capped_primary",
            OptimizerControl::auto()
                .with_max_feval(2000)
                .with_active_face_refit(ActiveFaceRefit::Experimental),
        ),
    ] {
        let (objective, feval, status, min_ms) = fit_row(formula, &data, control);
        println!("{method},{objective:.6},{feval},{min_ms:.3},\"{status}\"");
    }
}
