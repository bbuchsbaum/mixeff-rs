//! Crossed-LMM fit-time benchmark core, shared by the native driver
//! (`src/bin/native.rs`) and the wasm32 build driven from Node
//! (`../run_node.mjs`).
//!
//! Model: `y ~ condition + (1 + condition || participant) + (1 + condition || item)`
//! under REML, one observation per participant × item pair, condition
//! counterbalanced as ±0.5 with `condition = (p + i) mod 2`.
//! Truth: condition effect 25, residual SD 180, participant intercept SD 90,
//! item intercept SD 40, participant slope SD 30, item slope SD 40.
//!
//! Each phase is a separate call so the caller can time it with its own
//! clock (`Instant` natively, `performance.now()` under wasm32, which has no
//! `std` clock): `simulate` → `fit` → `satterthwaite`, and
//! `prepare_template` → `simulate` → `refit` for the compile-once /
//! refit-many path a power simulation uses.

use std::cell::RefCell;

use mixeff_rs::compiler::FixedEffectTestMethod;
use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::{DataFrame, FitOptions, LinearMixedModel, LinearMixedModelBuilder};
use mixeff_rs::prelude::MixedModelFit;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

pub const FORMULA: &str =
    "y ~ condition + (1 + condition || participant) + (1 + condition || item)";
pub const DESIGNS: [(usize, usize); 5] = [(16, 16), (24, 24), (32, 32), (40, 40), (64, 64)];

const EFFECT: f64 = 25.0;
const SD_RESID: f64 = 180.0;
const SD_P_INT: f64 = 90.0;
const SD_I_INT: f64 = 40.0;
const SD_P_SLOPE: f64 = 30.0;
const SD_I_SLOPE: f64 = 40.0;

#[derive(Default)]
struct State {
    data: Option<DataFrame>,
    model: Option<LinearMixedModel>,
    template: Option<LinearMixedModel>,
}

thread_local! {
    static STATE: RefCell<State> = RefCell::new(State::default());
}

/// Simulated responses plus the fixed design columns.
pub fn simulate_frame(n_p: usize, n_i: usize, seed: u64) -> DataFrame {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let z = Normal::new(0.0, 1.0).unwrap();
    let mut draw =
        |sd: f64, n: usize| -> Vec<f64> { (0..n).map(|_| sd * z.sample(&mut rng)).collect() };
    let p_int = draw(SD_P_INT, n_p);
    let p_slope = draw(SD_P_SLOPE, n_p);
    let i_int = draw(SD_I_INT, n_i);
    let i_slope = draw(SD_I_SLOPE, n_i);
    let eps = draw(SD_RESID, n_p * n_i);
    let n = n_p * n_i;
    let mut y = Vec::with_capacity(n);
    let mut condition = Vec::with_capacity(n);
    let mut participant = Vec::with_capacity(n);
    let mut item = Vec::with_capacity(n);
    for p in 0..n_p {
        for i in 0..n_i {
            let c = if (p + i) % 2 == 0 { -0.5 } else { 0.5 };
            y.push(p_int[p] + i_int[i] + (EFFECT + p_slope[p] + i_slope[i]) * c + eps[p * n_i + i]);
            condition.push(c);
            participant.push(format!("p{p:03}"));
            item.push(format!("i{i:03}"));
        }
    }
    let mut df = DataFrame::new();
    df.add_numeric("y", y).unwrap();
    df.add_numeric("condition", condition).unwrap();
    df.add_categorical("participant", participant).unwrap();
    df.add_categorical("item", item).unwrap();
    df
}

/// Phase 1: simulate a data set and keep it for the next call.
pub fn simulate(n_p: usize, n_i: usize, seed: u64) {
    let df = simulate_frame(n_p, n_i, seed);
    STATE.with(|s| s.borrow_mut().data = Some(df));
}

/// Phase 2: construct and fit the REML model on the stored data (fresh fit,
/// including model construction). Returns
/// `[beta_condition, se_condition, singular, converged, feval, objective]`.
pub fn fit() -> Vec<f64> {
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        let df = s.data.take().expect("simulate first");
        let model = LinearMixedModelBuilder::new(parse_formula(FORMULA).unwrap(), &df)
            .fit(FitOptions::reml())
            .expect("fit");
        let out = summary(&model);
        s.model = Some(model);
        out
    })
}

fn summary(model: &LinearMixedModel) -> Vec<f64> {
    let table = model.coeftable();
    let k = table.names.iter().position(|n| n == "condition").unwrap();
    vec![
        table.estimates[k],
        table.std_errors[k],
        f64::from(u8::from(model.is_singular())),
        f64::from(u8::from(model.optsum().converged())),
        model.optsum().feval as f64,
        model.objective(),
    ]
}

/// Phase 3: Satterthwaite table for the fitted model. Returns `[df, p]` for
/// `condition` (NaN when unavailable).
pub fn satterthwaite() -> Vec<f64> {
    STATE.with(|s| {
        let s = s.borrow();
        let model = s.model.as_ref().expect("fit first");
        let table = model.coeftable_with_method(FixedEffectTestMethod::Satterthwaite);
        let k = table.names.iter().position(|n| n == "condition").unwrap();
        vec![table.df[k].unwrap_or(f64::NAN), table.p_values[k]]
    })
}

/// Why the Satterthwaite p-value for `condition` is unavailable, if it is,
/// plus the resolved method label (e.g. `"satterthwaite: <reason>"`).
pub fn satterthwaite_reason() -> String {
    STATE.with(|s| {
        let s = s.borrow();
        let model = s.model.as_ref().expect("fit first");
        let table = model.coeftable_with_method(FixedEffectTestMethod::Satterthwaite);
        let k = table.names.iter().position(|n| n == "condition").unwrap();
        format!(
            "{}: {}",
            table.method,
            table.p_value_reasons[k].clone().unwrap_or_default()
        )
    })
}

/// Refit path, setup: fit a REML template once on seed `seed`.
pub fn prepare_template(n_p: usize, n_i: usize, seed: u64) {
    let df = simulate_frame(n_p, n_i, seed);
    let model = LinearMixedModelBuilder::new(parse_formula(FORMULA).unwrap(), &df)
        .fit(FitOptions::reml())
        .expect("template fit");
    STATE.with(|s| s.borrow_mut().template = Some(model));
}

/// Refit path: refit the template's REML criterion on the stored data's
/// response (warm start from the template optimum). Same summary as [`fit`].
pub fn refit() -> Vec<f64> {
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        let df = s.data.take().expect("simulate first");
        let y = df.numeric("y").unwrap().to_vec();
        let template = s.template.as_mut().expect("prepare_template first");
        template.refit(&y).expect("refit");
        summary(template)
    })
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use wasm_bindgen::prelude::wasm_bindgen;

    #[wasm_bindgen]
    pub fn simulate(n_p: u32, n_i: u32, seed: u32) {
        super::simulate(n_p as usize, n_i as usize, u64::from(seed));
    }
    #[wasm_bindgen]
    pub fn fit() -> Vec<f64> {
        super::fit()
    }
    #[wasm_bindgen]
    pub fn satterthwaite() -> Vec<f64> {
        super::satterthwaite()
    }
    #[wasm_bindgen]
    pub fn satterthwaite_reason() -> String {
        super::satterthwaite_reason()
    }
    #[wasm_bindgen]
    pub fn prepare_template(n_p: u32, n_i: u32, seed: u32) {
        super::prepare_template(n_p as usize, n_i as usize, u64::from(seed));
    }
    #[wasm_bindgen]
    pub fn refit() -> Vec<f64> {
        super::refit()
    }
}
