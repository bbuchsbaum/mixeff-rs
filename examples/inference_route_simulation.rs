//! Small inference-route simulation scaffold.
//!
//! This example is intentionally a harness, not a calibration claim. It emits a
//! JSON route table over deterministic fixtures so downstream R code can grow a
//! fuller simulation study without changing the route-status contract.

use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::{DataFrame, LinearMixedModel};
use mixeff_rs::stats::{profile_confint_payload, BoundaryLikelihoodRatioTest, LinearModelFit};
use nalgebra::{DMatrix, DVector};
use serde::Serialize;

#[derive(Debug, Serialize)]
struct RouteRecord {
    scenario: String,
    route: String,
    status: String,
    reason_code: Option<String>,
    finite_output: bool,
}

fn main() {
    let mut records = Vec::new();
    records.extend(run_random_intercept_scenario(
        "interior",
        random_intercept_data(1.25, 8),
    ));
    records.extend(run_random_intercept_scenario(
        "small_group",
        random_intercept_data(0.35, 4),
    ));

    println!("{}", serde_json::to_string_pretty(&records).unwrap());
}

fn run_random_intercept_scenario(name: &str, data: DataFrame) -> Vec<RouteRecord> {
    let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let mut records = Vec::new();
    let fixed_rows = model.fixed_effect_inference_table();
    for row in fixed_rows.rows {
        records.push(RouteRecord {
            scenario: name.to_string(),
            route: format!("wald_or_auto:{}", row.label),
            status: format!("{:?}", row.status),
            reason_code: row.reason.clone(),
            finite_output: row.p_value.is_some_and(|value| value.is_finite()),
        });
    }

    let profile_payload = profile_confint_payload(&mut model, 0.95).unwrap();
    records.push(RouteRecord {
        scenario: name.to_string(),
        route: "profile_ci".to_string(),
        status: "available".to_string(),
        reason_code: None,
        finite_output: profile_payload
            .intervals
            .iter()
            .all(|row| row.lower.is_finite() && row.upper.is_finite()),
    });

    let lm = intercept_slope_lm(&data);
    let boundary = BoundaryLikelihoodRatioTest::variance_component(&lm, &model);
    records.push(RouteRecord {
        scenario: name.to_string(),
        route: "boundary_lrt".to_string(),
        status: format!("{:?}", boundary.status),
        reason_code: boundary.reason_code,
        finite_output: boundary.pvalue.is_some_and(|value| value.is_finite()),
    });

    for route in ["bootstrap", "bootstrap_lrt"] {
        records.push(RouteRecord {
            scenario: name.to_string(),
            route: route.to_string(),
            status: "not_run".to_string(),
            reason_code: Some("simulation_harness_fast_tier_skips_bootstrap".to_string()),
            finite_output: false,
        });
    }

    records
}

fn random_intercept_data(group_sd: f64, n_groups: usize) -> DataFrame {
    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut group = Vec::new();
    for g in 0..n_groups {
        let shift = (g as f64 - (n_groups as f64 - 1.0) / 2.0) * group_sd;
        for i in 0..5 {
            let x_value = i as f64 - 2.0;
            let noise = ((g + i) % 3) as f64 * 0.15;
            y.push(10.0 + 0.7 * x_value + shift + noise);
            x.push(x_value);
            group.push(format!("G{g}"));
        }
    }
    let mut data = DataFrame::new();
    data.add_numeric("y", y).unwrap();
    data.add_numeric("x", x).unwrap();
    data.add_categorical("group", group).unwrap();
    data
}

fn intercept_slope_lm(data: &DataFrame) -> LinearModelFit {
    let y = data.numeric("y").unwrap();
    let x = data.numeric("x").unwrap();
    let response = DVector::from_column_slice(y);
    let mut matrix = DMatrix::zeros(y.len(), 2);
    for row in 0..y.len() {
        matrix[(row, 0)] = 1.0;
        matrix[(row, 1)] = x[row];
    }
    LinearModelFit::fit(response, matrix, Some("y ~ 1 + x".to_string())).unwrap()
}
