//! Repo-wide GLMM comparison gates.
//!
//! These tests make generated comparison artifacts executable evidence: GLMM
//! rows must be wired, routine rows must be classified, rows whose current
//! semantics match lme4 stay numerically gated, and current large-row
//! fast-PIRLS divergences remain tied to an explicit MixedModels.jl oracle.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;

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
