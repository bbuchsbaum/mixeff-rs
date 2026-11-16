//! wasm32 smoke test: a crossed-random-effects LMM must fit end to end in a
//! `wasm32-unknown-unknown` build with default features off (the pure-Rust
//! TrustBQ optimizer path that browser consumers use).
//!
//! `wasm32-unknown-unknown` has no clock and no threads: `std::time::Instant`
//! and `std::thread::spawn` panic at runtime there. This test exercises the
//! full fit path (initial objective, optimizer, boundary restart, optimizer
//! certificate, inference refresh) plus a warm refit, so any reintroduced
//! platform hazard on that path fails CI.
//!
//! Run with (see `.github/workflows/wasm.yml`):
//!
//! ```text
//! CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner \
//!   cargo test --target wasm32-unknown-unknown --no-default-features \
//!   --test wasm_crossed_fit
//! ```
//!
//! On native targets this file compiles to an empty test binary.
#![cfg(target_arch = "wasm32")]

use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::{DataFrame, FitOptions, LinearMixedModelBuilder};
use mixeff_rs::prelude::MixedModelFit;
use wasm_bindgen_test::wasm_bindgen_test;

/// Deterministic, dependency-free pseudo-normal draws (SplitMix64 + sum of
/// uniforms), so the fixture does not depend on an entropy source.
struct Draws(u64);

impl Draws {
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        (0..12).map(|_| self.uniform()).sum::<f64>() - 6.0
    }
}

fn crossed_data(n_participants: usize, n_items: usize, seed: u64) -> DataFrame {
    let mut draws = Draws(seed);
    let p_int: Vec<f64> = (0..n_participants).map(|_| 90.0 * draws.normal()).collect();
    let p_slope: Vec<f64> = (0..n_participants).map(|_| 30.0 * draws.normal()).collect();
    let i_int: Vec<f64> = (0..n_items).map(|_| 40.0 * draws.normal()).collect();
    let i_slope: Vec<f64> = (0..n_items).map(|_| 40.0 * draws.normal()).collect();
    let n = n_participants * n_items;
    let (mut y, mut condition, mut participant, mut item) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for p in 0..n_participants {
        for i in 0..n_items {
            let c = if (p + i) % 2 == 0 { -0.5 } else { 0.5 };
            y.push(
                600.0
                    + p_int[p]
                    + i_int[i]
                    + (25.0 + p_slope[p] + i_slope[i]) * c
                    + 180.0 * draws.normal(),
            );
            condition.push(c);
            participant.push(format!("p{p}"));
            item.push(format!("i{i}"));
        }
    }
    let mut df = DataFrame::new();
    df.add_numeric("y", y).unwrap();
    df.add_numeric("condition", condition).unwrap();
    df.add_categorical("participant", participant).unwrap();
    df.add_categorical("item", item).unwrap();
    df
}

#[wasm_bindgen_test]
fn crossed_lmm_fits_and_refits_under_wasm32() {
    let df = crossed_data(16, 16, 20_260_905);
    let formula =
        parse_formula("y ~ condition + (1 + condition || participant) + (1 + condition || item)")
            .unwrap();
    let mut model = LinearMixedModelBuilder::new(formula, &df)
        .fit(FitOptions::reml())
        .expect("crossed REML fit must succeed under wasm32");

    let beta = model.coef();
    assert_eq!(beta.len(), 2);
    assert!(beta.iter().all(|b| b.is_finite()));
    assert!(model.objective().is_finite());
    assert_eq!(model.theta().len(), 4);
    // No wall clock on wasm32-unknown-unknown: phase timings are absent
    // rather than panicking.
    assert!(model.fit_phase_timings().is_none());

    let table = model.coeftable();
    assert!(table
        .std_errors
        .iter()
        .all(|se| se.is_finite() && *se > 0.0));

    // The compile-once / refit-many path used by power simulation.
    let refit_df = crossed_data(16, 16, 7);
    let y = refit_df.numeric("y").unwrap().to_vec();
    model
        .refit(&y)
        .expect("warm refit must succeed under wasm32");
    assert!(model.objective().is_finite());
}

/// `threads > 1` must not spawn on wasm32: the bootstrap pipeline falls back
/// to the serial path and stays bit-identical to `threads = 1`.
#[wasm_bindgen_test]
fn threaded_bootstrap_runs_serially_under_wasm32() {
    use mixeff_rs::model::linear::{parametricbootstrap_with_options, BootstrapExecutionOptions};
    use rand::SeedableRng;

    let df = crossed_data(8, 8, 11);
    let formula = parse_formula("y ~ condition + (1 | participant) + (1 | item)").unwrap();
    let model = LinearMixedModelBuilder::new(formula, &df)
        .fit(FitOptions::reml())
        .unwrap();
    let run = |threads: usize| {
        let options = BootstrapExecutionOptions { threads };
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        parametricbootstrap_with_options(&mut rng, 4, &model, &options)
            .unwrap()
            .objectives()
    };
    let serial = run(1);
    let threaded = run(3);
    assert_eq!(serial.len(), 4);
    assert_eq!(
        serial.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        threaded.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}

/// The polled worker map behind threaded profiling also runs serially.
#[wasm_bindgen_test]
fn threaded_profile_runs_serially_under_wasm32() {
    use mixeff_rs::stats::profile::{profile_with_options, ProfileOptions};

    let df = crossed_data(8, 8, 5);
    let formula = parse_formula("y ~ condition + (1 | participant)").unwrap();
    let mut model = LinearMixedModelBuilder::new(formula, &df)
        .fit(FitOptions::ml())
        .unwrap();
    let options = ProfileOptions { threads: 2 };
    let profile = profile_with_options(&mut model, &options).expect("profile under wasm32");
    assert!(!profile.tbl.is_empty());
}
