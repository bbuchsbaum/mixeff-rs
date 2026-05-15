//! Inference-route simulation harness.
//!
//! This example is a route-status harness, not a calibration claim. It emits a
//! schema-tagged JSON report (`mixedmodels.inference_route_simulation`) over
//! deterministic fixtures so downstream R code can grow a fuller simulation
//! study without changing the route-status contract.
//!
//! Scenario strata, routes, and required checks are specified in
//! `docs/inference_simulation_harness.md`. The harness is built twice under the
//! same fixed seed and the two reports are asserted byte-identical before
//! printing, so a non-deterministic route silently breaking determinism fails
//! the example rather than leaking into the JSON.

use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::{parametricbootstrap, DataFrame, LinearMixedModel};
use mixeff_rs::stats::{
    parametric_bootstrap_lrt, profile_confint_payload, BoundaryLikelihoodRatioTest, LinearModelFit,
};
use nalgebra::{DMatrix, DVector};
use rand::{rngs::StdRng, SeedableRng};
use serde::Serialize;

const SCHEMA_NAME: &str = "mixedmodels.inference_route_simulation";
const SCHEMA_VERSION: &str = "1.0.0";
const SEED: u64 = 20_260_515;
const BOOTSTRAP_REPLICATES: usize = 48;
const PB_LRT_SIMULATIONS: usize = 32;

#[derive(Debug, Clone, PartialEq, Serialize)]
struct RouteRecord {
    scenario: String,
    route: String,
    status: String,
    reason_code: Option<String>,
    finite_output: bool,
    /// Replicates requested for a simulation route (`None` for analytic
    /// routes).
    replicates: Option<usize>,
    /// Replicates whose target statistic was finite.
    finite_statistics: Option<usize>,
    /// Monte Carlo standard error of the bootstrap mean of the slope
    /// coefficient, when a finite replicate distribution is available.
    monte_carlo_se: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct RouteSimulationReport {
    schema_name: String,
    schema_version: String,
    seed: u64,
    records: Vec<RouteRecord>,
}

fn analytic(
    scenario: &str,
    route: &str,
    status: &str,
    reason: Option<String>,
    finite: bool,
) -> RouteRecord {
    RouteRecord {
        scenario: scenario.to_string(),
        route: route.to_string(),
        status: status.to_string(),
        reason_code: reason,
        finite_output: finite,
        replicates: None,
        finite_statistics: None,
        monte_carlo_se: None,
    }
}

fn build_report() -> RouteSimulationReport {
    let mut records = Vec::new();
    // `interior`: full-rank, interior covariance estimate.
    records.extend(run_scenario(
        "interior",
        random_intercept_data(1.25, 8),
        false,
    ));
    // `small_group`: few grouping levels; finite-sample / MC error labelled.
    records.extend(run_scenario(
        "small_group",
        random_intercept_data(0.35, 4),
        false,
    ));
    // `boundary`: the grouping variance collapses to (near) zero.
    records.extend(run_scenario(
        "boundary",
        random_intercept_data(0.0, 6),
        false,
    ));
    // `reduced_rank`: random-effect covariance has unsupported directions
    // (random slope with a within-group-constant predictor over few groups).
    records.extend(run_scenario("reduced_rank", reduced_rank_data(3), true));

    RouteSimulationReport {
        schema_name: SCHEMA_NAME.to_string(),
        schema_version: SCHEMA_VERSION.to_string(),
        seed: SEED,
        records,
    }
}

fn main() {
    // Fixed-seed determinism: two independent builds must agree exactly.
    let report = build_report();
    let replay = build_report();
    assert_eq!(
        report, replay,
        "inference-route simulation harness must be deterministic under a fixed seed"
    );

    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}

fn run_scenario(name: &str, data: DataFrame, random_slope: bool) -> Vec<RouteRecord> {
    let formula_text = if random_slope {
        "y ~ 1 + x + (1 + x | group)"
    } else {
        "y ~ 1 + x + (1 | group)"
    };
    let formula = parse_formula(formula_text).unwrap();
    let mut model = match LinearMixedModel::new(formula, &data, None) {
        Ok(model) => model,
        Err(err) => {
            return vec![analytic(
                name,
                "model_construction",
                "refused",
                Some(format!("construction_failed: {err}")),
                false,
            )];
        }
    };
    if let Err(err) = model.fit(false) {
        return vec![analytic(
            name,
            "model_fit",
            "refused",
            Some(format!("fit_failed: {err}")),
            false,
        )];
    }

    let mut records = Vec::new();

    for row in model.fixed_effect_inference_table().rows {
        records.push(analytic(
            name,
            &format!("wald_or_auto:{}", row.label),
            &format!("{:?}", row.status),
            row.reason.clone(),
            row.p_value.is_some_and(|value| value.is_finite()),
        ));
    }

    match profile_confint_payload(&mut model, 0.95) {
        Ok(payload) => records.push(analytic(
            name,
            "profile_ci",
            "available",
            None,
            payload
                .intervals
                .iter()
                .all(|row| row.lower.is_finite() && row.upper.is_finite()),
        )),
        Err(err) => records.push(analytic(
            name,
            "profile_ci",
            "refused",
            Some(format!("profile_unavailable: {err}")),
            false,
        )),
    }

    let lm = intercept_slope_lm(&data);
    let boundary = BoundaryLikelihoodRatioTest::variance_component(&lm, &model);
    records.push(analytic(
        name,
        "boundary_lrt",
        &format!("{:?}", boundary.status),
        boundary.reason_code.clone(),
        boundary.pvalue.is_some_and(|value| value.is_finite()),
    ));

    // Parametric bootstrap of the fixed effects (real route, with replicate
    // bookkeeping and a Monte Carlo SE for the slope coefficient).
    let mut boot_rng = StdRng::seed_from_u64(SEED);
    let bootstrap = parametricbootstrap(&mut boot_rng, BOOTSTRAP_REPLICATES, &model);
    let slope: Vec<f64> = bootstrap
        .fits
        .iter()
        .filter_map(|fit| fit.beta.as_slice().get(1).copied())
        .filter(|value| value.is_finite())
        .collect();
    let finite_count = slope.len();
    let mc_se = monte_carlo_se(&slope);
    records.push(RouteRecord {
        scenario: name.to_string(),
        route: "parametric_bootstrap".to_string(),
        status: if finite_count > 0 {
            "available"
        } else {
            "degenerate"
        }
        .to_string(),
        reason_code: (finite_count == 0).then(|| "no_finite_bootstrap_statistics".to_string()),
        finite_output: mc_se.is_some_and(f64::is_finite),
        replicates: Some(BOOTSTRAP_REPLICATES),
        finite_statistics: Some(finite_count),
        monte_carlo_se: mc_se,
    });

    // Bootstrap LRT against the no-random-effects-slope null (only
    // meaningful for the random-slope scenarios; the random-intercept
    // scenarios compare against a random-intercept null instead).
    let null_text = "y ~ 1 + x + (1 | group)";
    if formula_text != null_text {
        let mut null_model =
            LinearMixedModel::new(parse_formula(null_text).unwrap(), &data, None).unwrap();
        if null_model.fit(false).is_ok() {
            let mut lrt_rng = StdRng::seed_from_u64(SEED);
            match parametric_bootstrap_lrt(&mut lrt_rng, PB_LRT_SIMULATIONS, &null_model, &model) {
                Ok(pb) => records.push(RouteRecord {
                    scenario: name.to_string(),
                    route: "bootstrap_lrt".to_string(),
                    status: "available".to_string(),
                    reason_code: None,
                    finite_output: pb.p_value.is_finite(),
                    replicates: Some(pb.n_sim_requested),
                    finite_statistics: Some(pb.n_sim_completed),
                    monte_carlo_se: None,
                }),
                Err(err) => records.push(RouteRecord {
                    scenario: name.to_string(),
                    route: "bootstrap_lrt".to_string(),
                    status: "refused".to_string(),
                    reason_code: Some(format!("pb_lrt_unavailable: {err}")),
                    finite_output: false,
                    replicates: Some(PB_LRT_SIMULATIONS),
                    finite_statistics: Some(0),
                    monte_carlo_se: None,
                }),
            }
        }
    } else {
        records.push(RouteRecord {
            scenario: name.to_string(),
            route: "bootstrap_lrt".to_string(),
            status: "not_applicable".to_string(),
            reason_code: Some("random_intercept_scenario_has_no_added_slope".to_string()),
            finite_output: false,
            replicates: None,
            finite_statistics: None,
            monte_carlo_se: None,
        });
    }

    records
}

/// Monte Carlo standard error of the bootstrap mean: `sd / sqrt(R)`.
fn monte_carlo_se(values: &[f64]) -> Option<f64> {
    let n = values.len();
    if n < 2 {
        return None;
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    Some((var / n as f64).sqrt())
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

/// Few groups with a near-degenerate slope signal so the `(1 + x | group)`
/// random-effect covariance has an unsupported (rank-deficient) direction.
fn reduced_rank_data(n_groups: usize) -> DataFrame {
    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut group = Vec::new();
    for g in 0..n_groups {
        let shift = (g as f64) * 0.5;
        for i in 0..6 {
            let x_value = i as f64 - 2.5;
            // Response carries no group-specific slope variation, so the
            // random-slope covariance block is (numerically) rank deficient.
            let noise = ((g * 6 + i) % 4) as f64 * 0.05;
            y.push(4.0 + 0.9 * x_value + shift + noise);
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
