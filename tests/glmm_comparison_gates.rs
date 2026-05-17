//! Repo-wide GLMM comparison gates.
//!
//! These tests make generated comparison artifacts executable evidence: GLMM
//! rows must be wired, routine rows must be classified, rows whose current
//! semantics match lme4 stay numerically gated, and current large-row
//! fast-PIRLS divergences remain tied to an explicit MixedModels.jl oracle.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;

#[cfg(feature = "nlopt")]
use mixeff_rs::compiler::{CertificateCheck, EvidenceMethod, FitStatus};
use mixeff_rs::datasets;
use mixeff_rs::formula::parse_formula;
#[cfg(feature = "nlopt")]
use mixeff_rs::model::data::DataFrame;
use mixeff_rs::model::generalized::GeneralizedLinearMixedModel;
#[cfg(feature = "nlopt")]
use mixeff_rs::model::traits::MixedModelFit;
use mixeff_rs::model::traits::{Family as ModelFamily, LinkFunction};
use serde_json::Value;

const BETA_ABS_TOL: f64 = 1e-3;
const BETA_REL_TOL: f64 = 1e-5;
const SIGMA_ABS_TOL: f64 = 1e-3;
const SIGMA_REL_TOL: f64 = 1e-4;

#[derive(Clone, Copy)]
struct ExpectedGlmmRow {
    dataset: &'static str,
    formula: &'static str,
    family: &'static str,
    link: &'static str,
    estimator: &'static str,
    status: &'static str,
}

#[derive(Clone, Copy)]
struct MatchingGate {
    row: ExpectedGlmmRow,
    beta_abs_tol: f64,
    theta_abs_tol: f64,
    sigma_abs_tol: f64,
}

const ARABIDOPSIS: ExpectedGlmmRow = ExpectedGlmmRow {
    dataset: "arabidopsis",
    formula: "total.fruits ~ amd * nutrient + (1 | reg/popu) + (1 | gen)",
    family: "Poisson",
    link: "Log",
    estimator: "Laplace",
    status: "ok",
};

const CBPP: ExpectedGlmmRow = ExpectedGlmmRow {
    dataset: "cbpp",
    formula: "incidence / size ~ 1 + period + (1 | herd)",
    family: "Binomial",
    link: "Logit",
    estimator: "Laplace",
    status: "ok",
};

const CONTRACEPTION_INTERCEPT: ExpectedGlmmRow = ExpectedGlmmRow {
    dataset: "contraception",
    formula: "use ~ 1 + age + livch + urban + (1 | dist)",
    family: "Binomial",
    link: "Logit",
    estimator: "Laplace",
    status: "ok",
};

const CONTRACEPTION_SLOPE: ExpectedGlmmRow = ExpectedGlmmRow {
    dataset: "contraception",
    formula: "use ~ 1 + age + livch + urban + (1 + urban | dist)",
    family: "Binomial",
    link: "Logit",
    estimator: "Laplace",
    status: "ok",
};

const CULCITA_BINOMIAL_LAPLACE: ExpectedGlmmRow = ExpectedGlmmRow {
    dataset: "culcitalogreg",
    formula: "predation ~ ttt + (1 | block)",
    family: "Binomial",
    link: "Logit",
    estimator: "Laplace",
    status: "ok",
};

const CULCITA_BINOMIAL_AGQ: ExpectedGlmmRow = ExpectedGlmmRow {
    dataset: "culcitalogreg",
    formula: "predation ~ ttt + (1 | block)",
    family: "Binomial",
    link: "Logit",
    estimator: "AGQ",
    status: "ok",
};

const GOPHERDAT2: ExpectedGlmmRow = ExpectedGlmmRow {
    dataset: "gopherdat2",
    formula: "shells ~ year + prev + offset(log(Area)) + (1 | Site)",
    family: "Poisson",
    link: "Log",
    estimator: "Laplace",
    status: "ok",
};

const GROUSETICKS: ExpectedGlmmRow = ExpectedGlmmRow {
    dataset: "grouseticks",
    formula: "TICKS ~ 1 + YEAR + cHEIGHT + (1 | BROOD) + (1 | INDEX) + (1 | LOCATION)",
    family: "Poisson",
    link: "Log",
    estimator: "Laplace",
    status: "ok",
};

const TUNGARA_STRESS: ExpectedGlmmRow = ExpectedGlmmRow {
    dataset: "tungara_single_caller",
    formula: "did_focal_follower_overlap_preceding_call_YN ~ 1 + preceding_caller_mass * distance + (1 | chorus_ID) + (1 + preceding_caller_mass | chorus_ID:focal_toe_clip_number)",
    family: "Binomial",
    link: "Logit",
    estimator: "Laplace",
    status: "skipped_stress",
};

const VERBAGG: ExpectedGlmmRow = ExpectedGlmmRow {
    dataset: "verbagg",
    formula: "r2 ~ 1 + Anger + Gender + btype + situ + mode + (1 | id) + (1 | item)",
    family: "Binomial",
    link: "Logit",
    estimator: "Laplace",
    status: "ok",
};

const EXPECTED_GLMM_ROWS: &[ExpectedGlmmRow] = &[
    ARABIDOPSIS,
    CBPP,
    CONTRACEPTION_INTERCEPT,
    CONTRACEPTION_SLOPE,
    CULCITA_BINOMIAL_LAPLACE,
    CULCITA_BINOMIAL_AGQ,
    GOPHERDAT2,
    GROUSETICKS,
    TUNGARA_STRESS,
    VERBAGG,
];

const MATCHING_GLMM_GATES: &[MatchingGate] = &[
    MatchingGate {
        row: ARABIDOPSIS,
        beta_abs_tol: BETA_ABS_TOL,
        theta_abs_tol: 2e-3,
        sigma_abs_tol: SIGMA_ABS_TOL,
    },
    MatchingGate {
        row: GOPHERDAT2,
        beta_abs_tol: BETA_ABS_TOL,
        theta_abs_tol: 1e-6,
        sigma_abs_tol: SIGMA_ABS_TOL,
    },
];

const FAST_ORACLE_ROWS: &[ExpectedGlmmRow] = &[
    CONTRACEPTION_INTERCEPT,
    CONTRACEPTION_SLOPE,
    GROUSETICKS,
    VERBAGG,
];

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read_json(path: &str) -> Value {
    let path = repo_root().join(path);
    serde_json::from_str(
        &fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}")),
    )
    .unwrap_or_else(|e| panic!("parse {path:?}: {e}"))
}

fn expected_key(dataset: &str, formula: &str, family: &str, link: &str, estimator: &str) -> String {
    format!("{dataset}\n{formula}\n{family}\n{link}\n{estimator}")
}

fn expected_row_key(row: ExpectedGlmmRow) -> String {
    expected_key(
        row.dataset,
        row.formula,
        row.family,
        row.link,
        row.estimator,
    )
}

fn record_key(record: &Value) -> String {
    expected_key(
        record
            .get("dataset")
            .and_then(Value::as_str)
            .unwrap_or("<missing dataset>"),
        record
            .get("formula")
            .and_then(Value::as_str)
            .unwrap_or("<missing formula>"),
        record
            .get("family")
            .and_then(Value::as_str)
            .unwrap_or("<missing family>"),
        record
            .get("link")
            .and_then(Value::as_str)
            .unwrap_or("<missing link>"),
        record
            .get("estimator")
            .and_then(Value::as_str)
            .unwrap_or("<missing estimator>"),
    )
}

fn results_by_key(file: &Value, label: &str) -> BTreeMap<String, Value> {
    let mut by_key = BTreeMap::new();
    for record in file
        .get("results")
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{label}: missing results[]"))
    {
        let key = record_key(record);
        let old = by_key.insert(key.clone(), record.clone());
        assert!(old.is_none(), "{label}: duplicate comparison key {key}");
    }
    by_key
}

fn is_glmm(record: &Value) -> bool {
    record.get("family").and_then(Value::as_str) != Some("Gaussian")
        || record.get("link").and_then(Value::as_str) != Some("Identity")
}

fn status(record: &Value) -> &str {
    record
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("<missing status>")
}

fn field_str<'a>(record: &'a Value, field: &str, key: &str) -> &'a str {
    record
        .get(field)
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("{key}: missing string `{field}`"))
}

fn field_f64(record: &Value, field: &str, key: &str) -> f64 {
    let value = record
        .get(field)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("{key}: missing numeric `{field}`"));
    assert!(value.is_finite(), "{key}: `{field}` is not finite: {value}");
    value
}

fn numeric_array(record: &Value, field: &str, key: &str) -> Vec<f64> {
    let values = record
        .get(field)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{key}: missing numeric array `{field}`"));
    assert!(
        !values.is_empty(),
        "{key}: `{field}` must not be empty for an ok GLMM row"
    );
    values
        .iter()
        .map(|value| {
            let value = value
                .as_f64()
                .unwrap_or_else(|| panic!("{key}: `{field}` contains non-numeric value {value}"));
            assert!(
                value.is_finite(),
                "{key}: `{field}` contains non-finite value {value}"
            );
            value
        })
        .collect()
}

fn optional_numeric_array(record: &Value, field: &str, key: &str) -> Option<Vec<f64>> {
    match record.get(field) {
        Some(Value::Null) | None => None,
        Some(Value::Array(_)) => Some(numeric_array(record, field, key)),
        Some(value) => panic!("{key}: `{field}` must be an array or null; got {value}"),
    }
}

fn max_abs(values: &[f64]) -> f64 {
    values
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max)
}

fn max_abs_delta(actual: &[f64], expected: &[f64], label: &str) -> f64 {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: vector lengths differ"
    );
    actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
}

fn within_tol(delta: f64, reference_scale: f64, abs_tol: f64, rel_tol: f64) -> bool {
    delta <= abs_tol || delta <= reference_scale.abs().max(f64::EPSILON) * rel_tol
}

fn construct_poisson_log_model(row: ExpectedGlmmRow) -> GeneralizedLinearMixedModel {
    assert_eq!(row.family, "Poisson");
    assert_eq!(row.link, "Log");
    assert_eq!(row.estimator, "Laplace");
    let (data, _) = datasets::load(row.dataset).expect("load GLMM fixture dataset");
    let (formula, offset) = if row.dataset == "gopherdat2" {
        let area = data
            .numeric("Area")
            .expect("gopherdat2 must expose numeric Area for offset");
        (
            "shells ~ year + prev + (1 | Site)".to_string(),
            Some(area.iter().map(|value| value.ln()).collect::<Vec<_>>()),
        )
    } else {
        (row.formula.to_string(), None)
    };
    let formula = parse_formula(&formula).expect("parse GLMM fixture formula");
    match offset {
        Some(offset) => GeneralizedLinearMixedModel::new_with_offset(
            formula,
            &data,
            ModelFamily::Poisson,
            Some(LinkFunction::Log),
            offset,
        ),
        None => GeneralizedLinearMixedModel::new(
            formula,
            &data,
            ModelFamily::Poisson,
            Some(LinkFunction::Log),
        ),
    }
    .expect("construct GLMM fixture model")
}

fn fit_poisson_log_fast_path(row: ExpectedGlmmRow) -> GeneralizedLinearMixedModel {
    let mut model = construct_poisson_log_model(row);
    model
        .fit_with_options(true, 1, false)
        .expect("fit GLMM fixture model");
    model
}

#[cfg(feature = "nlopt")]
fn construct_binomial_logit_model(row: ExpectedGlmmRow) -> GeneralizedLinearMixedModel {
    assert_eq!(row.family, "Binomial");
    assert_eq!(row.link, "Logit");
    let (data, _) = datasets::load(row.dataset).expect("load binomial GLMM fixture dataset");
    if row.dataset == "cbpp" {
        let incidence = data.numeric("incidence").expect("cbpp incidence column");
        let size = data.numeric("size").expect("cbpp size column");
        let proportion: Vec<f64> = incidence
            .iter()
            .zip(size.iter())
            .map(|(&y, &n)| y / n)
            .collect();
        let weights = size.to_vec();
        let mut data = data.clone();
        data.add_numeric("proportion", proportion)
            .expect("add cbpp proportion response");
        let formula =
            parse_formula("proportion ~ 1 + period + (1 | herd)").expect("parse cbpp formula");
        return GeneralizedLinearMixedModel::new_with_weights(
            formula,
            &data,
            ModelFamily::Binomial,
            Some(LinkFunction::Logit),
            weights,
        )
        .expect("construct cbpp binomial GLMM");
    }

    let formula = parse_formula(row.formula).expect("parse binomial GLMM formula");
    GeneralizedLinearMixedModel::new(
        formula,
        &data,
        ModelFamily::Binomial,
        Some(LinkFunction::Logit),
    )
    .expect("construct binomial GLMM")
}

#[cfg(feature = "nlopt")]
fn synthetic_overdispersed_poisson_model() -> GeneralizedLinearMixedModel {
    let mut data = DataFrame::new();
    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut group = Vec::new();
    let mut obs = Vec::new();
    let group_effects = [-0.7, 0.2, 0.8, -0.3];
    for (g, effect) in group_effects.iter().enumerate() {
        for j in 0..6 {
            let xv = j as f64 - 2.5;
            let eta = 0.4 + 0.18 * xv + effect;
            let base = eta.exp();
            let overdispersion_bump = if j % 3 == 0 { 2.0 } else { 0.0 };
            y.push((base + overdispersion_bump).round().max(0.0));
            x.push(xv);
            group.push(format!("g{}", g + 1));
            obs.push(format!("o{}_{}", g + 1, j + 1));
        }
    }
    data.add_numeric("y", y).unwrap();
    data.add_numeric("x", x).unwrap();
    data.add_categorical("group", group).unwrap();
    data.add_categorical("obs", obs).unwrap();
    let formula = parse_formula("y ~ 1 + x + (1 | group) + (1 | obs)").unwrap();
    GeneralizedLinearMixedModel::new(
        formula,
        &data,
        ModelFamily::Poisson,
        Some(LinkFunction::Log),
    )
    .expect("construct synthetic Poisson GLMM")
}

fn assert_ok_glmm_payload(record: &Value, label: &str, key: &str) {
    let beta = numeric_array(record, "beta", key);
    let coef_names = record
        .get("coef_names")
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{label}: {key}: missing coef_names array"));
    assert_eq!(
        beta.len(),
        coef_names.len(),
        "{label}: {key}: beta and coef_names length mismatch"
    );
    numeric_array(record, "theta", key);
    for field in [
        "sigma",
        "objective",
        "loglik",
        "aic",
        "bic",
        "fit_time_ms_min",
    ] {
        field_f64(record, field, key);
    }
    assert!(
        record.get("is_singular").and_then(Value::as_bool).is_some(),
        "{label}: {key}: is_singular must be boolean"
    );
    assert!(
        record
            .get("optimizer_fevals")
            .and_then(Value::as_i64)
            .is_some_and(|value| value > 0),
        "{label}: {key}: optimizer_fevals must be positive"
    );
    for field in [
        "objective_definition",
        "response_constants",
        "optimizer",
        "optimizer_backend",
        "optimizer_return_code",
    ] {
        assert!(
            !field_str(record, field, key).is_empty(),
            "{label}: {key}: `{field}` must not be empty"
        );
    }
}

#[test]
fn glmm_comparison_rows_have_expected_status_and_payload_shape() {
    let rust = read_json("comparison/rust_results.json");
    let r = read_json("comparison/lme4_results.json");
    let rust_by_key = results_by_key(&rust, "rust_results.json");
    let r_by_key = results_by_key(&r, "lme4_results.json");
    let expected_keys = EXPECTED_GLMM_ROWS
        .iter()
        .map(|row| expected_row_key(*row))
        .collect::<BTreeSet<_>>();

    for (label, by_key) in [
        ("rust_results.json", &rust_by_key),
        ("lme4_results.json", &r_by_key),
    ] {
        let mut glmm_keys = BTreeSet::new();
        for (key, record) in by_key {
            if !is_glmm(record) {
                continue;
            }
            glmm_keys.insert(key.clone());
            assert_ne!(
                status(record),
                "not_implemented",
                "{label}: GLMM row must be wired or explicitly classified: {key}"
            );
        }
        assert_eq!(
            glmm_keys, expected_keys,
            "{label}: GLMM comparison surface changed; update this gate deliberately"
        );
    }

    for row in EXPECTED_GLMM_ROWS {
        let key = expected_row_key(*row);
        let rust_record = rust_by_key
            .get(&key)
            .unwrap_or_else(|| panic!("rust_results.json missing GLMM row {key}"));
        let r_record = r_by_key
            .get(&key)
            .unwrap_or_else(|| panic!("lme4_results.json missing GLMM row {key}"));

        assert_eq!(status(rust_record), row.status, "rust status for {key}");
        assert_eq!(status(r_record), row.status, "lme4 status for {key}");

        if row.status == "ok" {
            assert_ok_glmm_payload(rust_record, "rust_results.json", &key);
            assert_ok_glmm_payload(r_record, "lme4_results.json", &key);
            assert_eq!(
                rust_record
                    .get("objective_definition")
                    .and_then(Value::as_str),
                Some("profiled_glmm_deviance"),
                "{key}: Rust GLMM objective definition must stay explicit"
            );
            assert_eq!(
                rust_record
                    .get("response_constants")
                    .and_then(Value::as_str),
                Some("dropped"),
                "{key}: Rust GLMM response-constant convention must stay explicit"
            );
            assert_eq!(
                r_record.get("objective_definition").and_then(Value::as_str),
                Some("minus_two_loglik"),
                "{key}: lme4 GLMM objective definition must stay explicit"
            );
            assert_eq!(
                r_record.get("response_constants").and_then(Value::as_str),
                Some("included"),
                "{key}: lme4 GLMM response-constant convention must stay explicit"
            );
        } else {
            let reason = rust_record
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("");
            assert!(
                reason.contains("MIXEDMODELS_INCLUDE_STRESS=1"),
                "{key}: skipped stress row must name opt-in env var"
            );
        }
    }
}

#[test]
fn glmm_lme4_matching_rows_stay_within_numeric_gates() {
    let rust = read_json("comparison/rust_results.json");
    let r = read_json("comparison/lme4_results.json");
    let rust_by_key = results_by_key(&rust, "rust_results.json");
    let r_by_key = results_by_key(&r, "lme4_results.json");

    for gate in MATCHING_GLMM_GATES {
        let key = expected_row_key(gate.row);
        let rust_record = rust_by_key
            .get(&key)
            .unwrap_or_else(|| panic!("rust_results.json missing GLMM row {key}"));
        let r_record = r_by_key
            .get(&key)
            .unwrap_or_else(|| panic!("lme4_results.json missing GLMM row {key}"));

        let rust_beta = numeric_array(rust_record, "beta", &key);
        let r_beta = numeric_array(r_record, "beta", &key);
        let beta_delta = max_abs_delta(&rust_beta, &r_beta, &format!("{key}: beta"));
        assert!(
            within_tol(
                beta_delta,
                max_abs(&r_beta),
                gate.beta_abs_tol,
                BETA_REL_TOL
            ),
            "{key}: beta delta {beta_delta:.6} exceeds GLMM gate tolerance"
        );

        let rust_theta = numeric_array(rust_record, "theta", &key);
        let r_theta = numeric_array(r_record, "theta", &key);
        let theta_delta = max_abs_delta(&rust_theta, &r_theta, &format!("{key}: theta"));
        assert!(
            theta_delta <= gate.theta_abs_tol,
            "{key}: theta delta {theta_delta:.6} exceeds GLMM gate tolerance {}",
            gate.theta_abs_tol
        );

        let sigma_delta =
            (field_f64(rust_record, "sigma", &key) - field_f64(r_record, "sigma", &key)).abs();
        assert!(
            within_tol(
                sigma_delta,
                field_f64(r_record, "sigma", &key),
                gate.sigma_abs_tol,
                SIGMA_REL_TOL,
            ),
            "{key}: sigma delta {sigma_delta:.6} exceeds GLMM gate tolerance"
        );
    }
}

#[test]
fn glmm_included_response_constant_objective_matches_lme4_for_poisson_parity_rows() {
    let r = read_json("comparison/lme4_results.json");
    let r_by_key = results_by_key(&r, "lme4_results.json");

    for row in [ARABIDOPSIS, GOPHERDAT2] {
        let key = expected_row_key(row);
        let mut model = fit_poisson_log_fast_path(row);
        let dropped = model.deviance(1);
        let offset = model.response_constants_offset();
        let included = model.deviance_with_response_constants(1);
        assert!(
            offset > 0.0,
            "{key}: Poisson response-constant offset should be positive"
        );
        assert!(
            (included - (dropped + offset)).abs() <= 1e-8,
            "{key}: included objective must equal dropped objective plus response constants"
        );

        let r_record = r_by_key
            .get(&key)
            .unwrap_or_else(|| panic!("lme4_results.json missing GLMM row {key}"));
        assert_eq!(field_str(r_record, "response_constants", &key), "included");
        let lme4_objective = field_f64(r_record, "objective", &key);
        let delta = (included - lme4_objective).abs();
        assert!(
            delta <= 0.02,
            "{key}: Rust included-constants objective {included:.6} should match lme4 -2logLik {lme4_objective:.6}; delta={delta:.6}"
        );
    }
}

#[test]
fn glmm_fast_path_gaps_match_mixedmodels_jl_fast_oracle() {
    let fixture = read_json("tests/fixtures/parity/glmm_fast_oracles.json");
    assert_eq!(
        fixture.get("schema_version").and_then(Value::as_str),
        Some("1.0.0")
    );
    assert_eq!(
        fixture.get("reference_engine").and_then(Value::as_str),
        Some("MixedModels.jl 5.3.0")
    );
    assert_eq!(
        fixture.get("fit_mode").and_then(Value::as_str),
        Some("fast=true")
    );

    let rust = read_json("comparison/rust_results.json");
    let rust_by_key = results_by_key(&rust, "rust_results.json");
    let rows = fixture
        .get("rows")
        .and_then(Value::as_array)
        .expect("glmm fast oracle fixture must contain rows[]");
    let mut covered = BTreeSet::new();

    for oracle in rows {
        let key = expected_key(
            field_str(oracle, "dataset", "glmm_fast_oracles.json"),
            field_str(oracle, "formula", "glmm_fast_oracles.json"),
            field_str(oracle, "family", "glmm_fast_oracles.json"),
            field_str(oracle, "link", "glmm_fast_oracles.json"),
            field_str(oracle, "estimator", "glmm_fast_oracles.json"),
        );
        covered.insert(key.clone());

        assert!(
            field_str(oracle, "classification", &key).contains("fast-PIRLS"),
            "{key}: oracle classification must explain the lme4 disagreement"
        );
        let rust_record = rust_by_key
            .get(&key)
            .unwrap_or_else(|| panic!("rust_results.json missing GLMM fast-oracle row {key}"));
        assert_eq!(status(rust_record), "ok", "{key}: row must be available");

        let objective_delta = (field_f64(rust_record, "objective", &key)
            - field_f64(oracle, "objective", &key))
        .abs();
        let objective_abs_tol = field_f64(oracle, "objective_abs_tol", &key);
        assert!(
            objective_delta <= objective_abs_tol,
            "{key}: Rust/MixedModels.jl fast objective delta {objective_delta:.8} exceeds {objective_abs_tol}"
        );

        if let Some(expected_beta) = optional_numeric_array(oracle, "rust_comparable_beta", &key) {
            let rust_beta = numeric_array(rust_record, "beta", &key);
            let beta_delta = max_abs_delta(&rust_beta, &expected_beta, &format!("{key}: beta"));
            let beta_abs_tol = field_f64(oracle, "beta_abs_tol", &key);
            assert!(
                beta_delta <= beta_abs_tol,
                "{key}: Rust/MixedModels.jl fast beta delta {beta_delta:.8} exceeds {beta_abs_tol}"
            );
        }

        if let Some(expected_theta_abs) =
            optional_numeric_array(oracle, "rust_comparable_theta_abs", &key)
        {
            let rust_theta_abs = numeric_array(rust_record, "theta", &key)
                .into_iter()
                .map(f64::abs)
                .collect::<Vec<_>>();
            let theta_delta = max_abs_delta(
                &rust_theta_abs,
                &expected_theta_abs,
                &format!("{key}: abs(theta)"),
            );
            let theta_abs_tol = field_f64(oracle, "theta_abs_tol", &key);
            assert!(
                theta_delta <= theta_abs_tol,
                "{key}: Rust/MixedModels.jl fast theta delta {theta_delta:.8} exceeds {theta_abs_tol}"
            );
        }
    }

    let expected_covered = FAST_ORACLE_ROWS
        .iter()
        .map(|row| expected_row_key(*row))
        .collect::<BTreeSet<_>>();
    assert_eq!(
        covered, expected_covered,
        "fast-oracle fixture must cover every fast-PIRLS known gap"
    );
}

#[test]
fn grouseticks_joint_oracle_pins_fast_false_target_without_promoting_current_fast_path() {
    let fixture = read_json("tests/fixtures/parity/glmm_joint_oracles.json");
    assert_eq!(
        fixture.get("schema_version").and_then(Value::as_str),
        Some("1.0.0")
    );
    let rows = fixture
        .get("rows")
        .and_then(Value::as_array)
        .expect("glmm joint oracle fixture must contain rows[]");
    assert_eq!(
        rows.len(),
        1,
        "joint oracle fixture currently pins grouseticks"
    );

    let oracle = &rows[0];
    let key = expected_key(
        field_str(oracle, "dataset", "glmm_joint_oracles.json"),
        field_str(oracle, "formula", "glmm_joint_oracles.json"),
        field_str(oracle, "family", "glmm_joint_oracles.json"),
        field_str(oracle, "link", "glmm_joint_oracles.json"),
        field_str(oracle, "estimator", "glmm_joint_oracles.json"),
    );
    assert_eq!(key, expected_row_key(GROUSETICKS));
    assert!(
        field_str(oracle, "classification", &key).contains("future fast=false"),
        "{key}: fixture classification must keep this as a future joint target"
    );

    let mm_fast_true = oracle
        .get("mixedmodels_jl_fast_true")
        .unwrap_or_else(|| panic!("{key}: missing MixedModels.jl fast=true oracle"));
    let mm_fast_false = oracle
        .get("mixedmodels_jl_fast_false")
        .unwrap_or_else(|| panic!("{key}: missing MixedModels.jl fast=false oracle"));
    let lme4 = oracle
        .get("lme4_glmer")
        .unwrap_or_else(|| panic!("{key}: missing lme4 oracle"));
    let tolerances = oracle
        .get("tolerances")
        .unwrap_or_else(|| panic!("{key}: missing tolerances"));

    assert_eq!(field_str(mm_fast_true, "fit_mode", &key), "fast=true");
    assert_eq!(field_str(mm_fast_false, "fit_mode", &key), "fast=false");
    assert_eq!(field_str(lme4, "fit_mode", &key), "joint");
    assert_eq!(
        field_str(mm_fast_true, "response_constants", &key),
        "dropped"
    );
    assert_eq!(
        field_str(mm_fast_false, "response_constants", &key),
        "dropped"
    );
    assert_eq!(field_str(lme4, "response_constants", &key), "included");

    let rust = read_json("comparison/rust_results.json");
    let rust_by_key = results_by_key(&rust, "rust_results.json");
    let rust_record = rust_by_key
        .get(&key)
        .unwrap_or_else(|| panic!("rust_results.json missing GLMM row {key}"));
    assert_eq!(
        status(rust_record),
        "ok",
        "{key}: current Rust fast path must fit"
    );

    let objective_delta = (field_f64(rust_record, "objective", &key)
        - field_f64(mm_fast_true, "objective", &key))
    .abs();
    assert!(
        objective_delta <= field_f64(tolerances, "rust_fast_true_objective_abs_tol", &key),
        "{key}: Rust fast=true objective must match MixedModels.jl fast=true oracle, got {objective_delta:.8}"
    );

    let rust_beta = numeric_array(rust_record, "beta", &key);
    let fast_true_beta = numeric_array(mm_fast_true, "beta", &key);
    let fast_true_beta_delta = max_abs_delta(
        &rust_beta,
        &fast_true_beta,
        &format!("{key}: fast=true beta"),
    );
    assert!(
        fast_true_beta_delta <= field_f64(tolerances, "rust_fast_true_beta_abs_tol", &key),
        "{key}: Rust fast=true beta must match MixedModels.jl fast=true oracle, got {fast_true_beta_delta:.8}"
    );

    let rust_theta_abs = numeric_array(rust_record, "theta", &key)
        .into_iter()
        .map(f64::abs)
        .collect::<Vec<_>>();
    let fast_true_theta_abs = numeric_array(mm_fast_true, "theta", &key)
        .into_iter()
        .map(f64::abs)
        .collect::<Vec<_>>();
    let fast_true_theta_delta = max_abs_delta(
        &rust_theta_abs,
        &fast_true_theta_abs,
        &format!("{key}: fast=true abs(theta)"),
    );
    assert!(
        fast_true_theta_delta <= field_f64(tolerances, "rust_fast_true_theta_abs_tol", &key),
        "{key}: Rust fast=true theta must match MixedModels.jl fast=true oracle, got {fast_true_theta_delta:.8}"
    );

    let fast_false_beta = numeric_array(mm_fast_false, "beta", &key);
    let lme4_beta = numeric_array(lme4, "beta", &key);
    let fast_false_lme4_beta_delta = max_abs_delta(
        &fast_false_beta,
        &lme4_beta,
        &format!("{key}: fast=false/lme4 beta"),
    );
    assert!(
        fast_false_lme4_beta_delta <= field_f64(tolerances, "fast_false_lme4_beta_abs_tol", &key),
        "{key}: MixedModels.jl fast=false must reproduce the lme4 beta target, got {fast_false_lme4_beta_delta:.8}"
    );

    let fast_false_theta_abs = numeric_array(mm_fast_false, "theta", &key)
        .into_iter()
        .map(f64::abs)
        .collect::<Vec<_>>();
    let lme4_theta_abs = numeric_array(lme4, "theta", &key)
        .into_iter()
        .map(f64::abs)
        .collect::<Vec<_>>();
    let fast_false_lme4_theta_delta = max_abs_delta(
        &fast_false_theta_abs,
        &lme4_theta_abs,
        &format!("{key}: fast=false/lme4 abs(theta)"),
    );
    assert!(
        fast_false_lme4_theta_delta <= field_f64(tolerances, "fast_false_lme4_theta_abs_tol", &key),
        "{key}: MixedModels.jl fast=false must reproduce the lme4 theta scale target, got {fast_false_lme4_theta_delta:.8}"
    );

    let beta_gap_current_to_joint = max_abs_delta(
        &rust_beta,
        &lme4_beta,
        &format!("{key}: current Rust vs joint beta"),
    );
    assert!(
        beta_gap_current_to_joint > 0.1,
        "{key}: fixture must keep the current fast-PIRLS beta gap visible until fast=false lands"
    );
}

#[cfg(feature = "nlopt")]
#[test]
fn experimental_joint_laplace_improves_included_objective_without_changing_public_fast_false() {
    let key = "synthetic overdispersed Poisson GLMM";

    let mut profiled = synthetic_overdispersed_poisson_model();
    profiled
        .fit_with_options(true, 1, false)
        .expect("profiled synthetic fit");
    let profiled_objective = profiled.deviance_with_response_constants(1);

    let mut joint = synthetic_overdispersed_poisson_model();
    joint
        .fit_experimental_joint_laplace_with_response_constants(false)
        .expect("experimental joint fit");
    let joint_objective = joint.deviance_with_response_constants(1);
    assert!(
        joint_objective < profiled_objective - 1.0e-3,
        "{key}: experimental joint fit should reduce included objective vs profiled start; \
         profiled={profiled_objective:.6}, joint={joint_objective:.6}"
    );
    assert!(
        joint
            .opt_summary()
            .return_value
            .contains("EXPERIMENTAL_JOINT"),
        "{key}: opt summary must label the path experimental"
    );

    assert!(
        MixedModelFit::coef(&joint)
            .iter()
            .all(|value| value.is_finite()),
        "{key}: experimental joint beta must stay finite"
    );
    let certificate = joint
        .compiler_artifact()
        .optimizer_certificate
        .as_ref()
        .expect("experimental joint fit must record an optimizer certificate");
    assert!(
        matches!(
            certificate.status,
            FitStatus::ConvergedInterior | FitStatus::ConvergedBoundary
        ),
        "{key}: experimental joint fit should classify covariance state through FitStatus, got {:?}",
        certificate.status
    );
    assert!(
        matches!(
            certificate.evidence.gradient.method,
            EvidenceMethod::FiniteDifference
        ),
        "{key}: experimental joint certificate must record finite-difference stationarity evidence"
    );
    let residual = certificate
        .free_gradient_norm
        .expect("experimental joint certificate must record a first-order residual");
    let tolerance = certificate
        .checks
        .iter()
        .find_map(|check| match check {
            CertificateCheck::FreeGradientOk { tolerance, .. } => Some(*tolerance),
            _ => None,
        })
        .expect("experimental joint certificate must record the stationarity tolerance");
    assert!(
        residual <= tolerance,
        "{key}: stationarity residual {residual:.6e} should be <= tolerance {tolerance:.6e}"
    );

    let mut public_fast_false = synthetic_overdispersed_poisson_model();
    let error = public_fast_false
        .fit_with_options(false, 1, false)
        .expect_err("stable fast=false must remain unsupported");
    assert!(
        error.to_string().contains("fast = false") && error.to_string().contains("not implemented"),
        "{key}: public fast=false must remain an explicit unsupported path, got {error}"
    );
}

#[cfg(feature = "nlopt")]
#[test]
fn experimental_joint_binomial_rows_stay_below_promotion_gate() {
    let rows = [
        CBPP,
        CULCITA_BINOMIAL_LAPLACE,
        CONTRACEPTION_INTERCEPT,
        CONTRACEPTION_SLOPE,
    ];
    let lme4 = read_json("comparison/lme4_results.json");
    let lme4_by_key = results_by_key(&lme4, "lme4_results.json");
    let mut passed = Vec::new();
    let mut missed_objective_gate = Vec::new();
    for row in rows {
        let key = expected_row_key(row);
        let lme4_record = lme4_by_key
            .get(&key)
            .unwrap_or_else(|| panic!("lme4_results.json missing GLMM row {key}"));

        let mut joint = construct_binomial_logit_model(row);
        joint
            .fit_experimental_joint_laplace_with_response_constants(false)
            .unwrap_or_else(|err| panic!("{key}: fit experimental joint failed: {err}"));

        let objective = joint.deviance_with_response_constants(1);
        let lme4_objective = field_f64(lme4_record, "objective", &key);
        let beta = MixedModelFit::coef(&joint);
        let theta = joint.theta();
        let lme4_beta = numeric_array(lme4_record, "beta", &key);
        let lme4_theta = numeric_array(lme4_record, "theta", &key);
        let objective_delta = (objective - lme4_objective).abs();
        let beta_delta = max_abs_delta(beta.as_slice(), &lme4_beta, &format!("{key}: beta"));
        let theta_delta = max_abs_delta(&theta, &lme4_theta, &format!("{key}: theta"));
        let status = joint.opt_summary().return_value.clone();
        let pass = objective_delta <= 1e-4 && beta_delta <= 1e-3 && theta_delta <= 2e-3;

        println!(
            "promotion probe {key}: pass={pass}; status={status}; objective={objective:.9}; lme4={lme4_objective:.9}; objective_delta={objective_delta:.9}; beta_delta={beta_delta:.9}; theta_delta={theta_delta:.9}; theta={theta:?}; beta={:?}",
            beta.as_slice()
        );
        if pass {
            passed.push(key);
        } else if objective_delta > 1e-4 {
            missed_objective_gate.push(key);
        }
    }

    assert!(
        passed.is_empty(),
        "binomial GLMM rows now satisfy experimental joint promotion tolerance; promote these rows through the scorecard gate: {passed:?}"
    );
    assert!(
        !missed_objective_gate.is_empty(),
        "at least one probed binomial row should record why phase 6 is still blocked"
    );
}

#[test]
fn glmm_report_contains_expected_numeric_classifications() {
    let report =
        fs::read_to_string(repo_root().join("comparison").join("REPORT.md")).expect("read report");
    assert!(
        !report.contains("numeric_disagreement"),
        "GLMM numeric disagreements must be classified in comparison/REPORT.md"
    );
    for required in [
        "small Binomial/Logit row uses the current fast-PIRLS profiled path",
        "Binomial/AGQ row uses the current fast-PIRLS profiled path with AGQ quadrature",
        "Poisson/Log multi-random-intercept row matches MixedModels.jl 5.3.0 fast=true",
        "large crossed Binomial/Logit row matches MixedModels.jl 5.3.0 fast=true",
        "large Binomial/Logit random-intercept row matches MixedModels.jl 5.3.0 fast=true",
        "large Binomial/Logit random-slope row matches MixedModels.jl 5.3.0 fast=true",
    ] {
        assert!(
            report.contains(required),
            "comparison/REPORT.md missing GLMM numeric classification: {required}"
        );
    }
}
