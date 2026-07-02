//! Construction-time benchmark for the streamed fixed-design rank/pivot
//! boundary (mote K6, bd-01KWFNE4RXBZ60MHMSN29TDSJY).
//!
//! Builds a high-cardinality categorical fixed design that the auto policy
//! routes to the streamed backend, then times `LinearMixedModel::new`
//! (which is where rank/pivot detection runs). Run under `/usr/bin/time -l`
//! (macOS) or `/usr/bin/time -v` (Linux) to capture peak RSS:
//!
//! ```bash
//! cargo build --release --example streamed_rank_bench
//! /usr/bin/time -l target/release/examples/streamed_rank_bench 1500 20
//! ```
//!
//! Comparing a checkout before and after the K6 seam shows the effect of
//! skipping the dense Householder pass (O(n·p²) flops) and the pivoted
//! copy (one extra n×p dense allocation) on comfortably full-rank designs.

use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::{DataFrame, LinearMixedModel};
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let n_levels: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(1000);
    let obs_per_level: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(20);
    let n_obs = n_levels * obs_per_level;
    let n_groups = 600.min(n_obs);

    let mut y = Vec::with_capacity(n_obs);
    let mut x = Vec::with_capacity(n_obs);
    let mut sku = Vec::with_capacity(n_obs);
    let mut group = Vec::with_capacity(n_obs);
    for level in 0..n_levels {
        for rep in 0..obs_per_level {
            let obs = level * obs_per_level + rep;
            let x_value = rep as f64 - 0.5 + ((level % 5) as f64) * 0.1;
            let sku_effect = ((level % 11) as f64 - 5.0) * 0.07;
            let group_id = (obs * 7) % n_groups;
            let group_effect = ((group_id % 23) as f64 - 11.0) * 0.03;
            y.push(2.0 + 0.8 * x_value + sku_effect + group_effect);
            x.push(x_value);
            sku.push(format!("sku{level:05}"));
            group.push(format!("g{group_id:04}"));
        }
    }
    let mut data = DataFrame::new();
    data.add_numeric("y", y).unwrap();
    data.add_numeric("x", x).unwrap();
    data.add_categorical("sku", sku).unwrap();
    data.add_categorical("group", group).unwrap();

    let formula = parse_formula("y ~ 1 + x + sku + (1 | group)").unwrap();

    let t0 = Instant::now();
    let model = LinearMixedModel::new(formula, &data, None).unwrap();
    let construct_ms = t0.elapsed().as_secs_f64() * 1e3;

    let summary = model.fixed_design_backend_summary();
    println!(
        "n_obs={n_obs} p={} rank={} backend={:?} construct_ms={construct_ms:.1}",
        summary.n_cols,
        model.fixed_effect_rank(),
        summary.storage,
    );
}
