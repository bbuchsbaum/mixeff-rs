//! Repo-wide GLMM comparison gates.
//!
//! These tests make the generated comparison artifacts executable evidence:
//! GLMM rows must be wired, every routine row must be classified, rows whose
//! current semantics match lme4 must stay within tolerance, and current numeric
//! gaps must remain explicit until the parity-fix bead removes them.

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

#[derive(Clone, Copy)]
struct KnownGap {
    row: ExpectedGlmmRow,
    reason: &'static str,
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

const CBPP_AGQ: ExpectedGlmmRow = ExpectedGlmmRow {
    dataset: "cbpp",
    formula: "incidence / size ~ 1 + period + (1 | herd)",
    family: "Binomial",
    link: "Logit",
    estimator: "AGQ",
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

const CULCITA_BERNOULLI_LAPLACE: ExpectedGlmmRow = ExpectedGlmmRow {
    dataset: "culcitalogreg",
    formula: "predation ~ ttt + (1 | block)",
    family: "Bernoulli",
    link: "Logit",
    estimator: "Laplace",
    status: "ok",
};

const ERGOSTOOL_GAMMA: ExpectedGlmmRow = ExpectedGlmmRow {
    dataset: "ergostool",
    formula: "effort ~ 1 + Type + (1 | Subject)",
    family: "Gamma",
    link: "Log",
    estimator: "Laplace",
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
    CBPP_AGQ,
    CONTRACEPTION_INTERCEPT,
    CONTRACEPTION_SLOPE,
    CULCITA_BINOMIAL_LAPLACE,
    CULCITA_BINOMIAL_AGQ,
    CULCITA_BERNOULLI_LAPLACE,
    ERGOSTOOL_GAMMA,
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
        row: CBPP,
        beta_abs_tol: BETA_ABS_TOL,
        theta_abs_tol: BETA_ABS_TOL,
        sigma_abs_tol: SIGMA_ABS_TOL,
    },
    MatchingGate {
        row: CBPP_AGQ,
        beta_abs_tol: BETA_ABS_TOL,
        theta_abs_tol: BETA_ABS_TOL,
        sigma_abs_tol: SIGMA_ABS_TOL,
    },
    MatchingGate {
        row: CULCITA_BINOMIAL_LAPLACE,
        beta_abs_tol: BETA_ABS_TOL,
        theta_abs_tol: BETA_ABS_TOL,
        sigma_abs_tol: SIGMA_ABS_TOL,
    },
    MatchingGate {
        row: CULCITA_BINOMIAL_AGQ,
        // The comparison artifact records lme4 values rounded to four
        // decimals; the unrounded AGQ fit is at the same optimum.
        beta_abs_tol: 2e-3,
        theta_abs_tol: BETA_ABS_TOL,
        sigma_abs_tol: SIGMA_ABS_TOL,
    },
    MatchingGate {
        row: CULCITA_BERNOULLI_LAPLACE,
        beta_abs_tol: BETA_ABS_TOL,
        theta_abs_tol: BETA_ABS_TOL,
        sigma_abs_tol: SIGMA_ABS_TOL,
    },
    MatchingGate {
        row: GOPHERDAT2,
        beta_abs_tol: BETA_ABS_TOL,
        theta_abs_tol: 1e-6,
        sigma_abs_tol: SIGMA_ABS_TOL,
    },
];

const KNOWN_NUMERIC_GAPS: &[KnownGap] = &[
    KnownGap {
        row: CONTRACEPTION_INTERCEPT,
        reason: "large Binomial/Logit row matches MixedModels.jl 5.3.0 fast=true profiled objective; lme4 beta gap is fast-PIRLS versus joint-estimate divergence tracked by bd-01KR6TEJ7VEJDSXWKEV35Y76NR",
    },
    KnownGap {
        row: CONTRACEPTION_SLOPE,
        reason: "large Binomial/Logit random-slope row matches MixedModels.jl 5.3.0 fast=true profiled objective; lme4 beta gap is fast-PIRLS versus joint-estimate divergence tracked by bd-01KR6TEJ7VEJDSXWKEV35Y76NR",
    },
    KnownGap {
        row: ERGOSTOOL_GAMMA,
        reason: "Gamma/Log dispersion and theta conventions are not an lme4-only oracle",
    },
    KnownGap {
        row: GROUSETICKS,
        reason: "Poisson/Log multi-random-intercept row matches MixedModels.jl 5.3.0 fast=true profiled objective; lme4 beta gap is fast-PIRLS versus joint-estimate divergence tracked by bd-01KR6TEJ7VEJDSXWKEV35Y76NR",
    },
    KnownGap {
        row: VERBAGG,
        reason: "large crossed Binomial/Logit row matches MixedModels.jl 5.3.0 fast=true profiled objective; lme4 beta gap is fast-PIRLS versus joint-estimate divergence tracked by bd-01KR6TEJ7VEJDSXWKEV35Y76NR",
    },
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

fn expected_row_key(row: ExpectedGlmmRow) -> String {
    expected_key(
        row.dataset,
        row.formula,
        row.family,
        row.link,
        row.estimator,
    )
}

fn expected_key(dataset: &str, formula: &str, family: &str, link: &str, estimator: &str) -> String {
    format!("{dataset}\n{formula}\n{family}\n{link}\n{estimator}")
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

fn nested_object<'a>(record: &'a Value, field: &str, key: &str) -> &'a Value {
    let value = record
        .get(field)
        .unwrap_or_else(|| panic!("{key}: missing object `{field}`"));
    assert!(value.is_object(), "{key}: `{field}` must be an object");
    value
}

fn optional_numeric_array(record: &Value, field: &str, key: &str) -> Option<Vec<f64>> {
    match record.get(field) {
        Some(Value::Null) | None => None,
        Some(Value::Array(_)) => Some(numeric_array(record, field, key)),
        Some(value) => panic!("{key}: `{field}` must be an array or null; got {value}"),
    }
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
        field_f64(record, "fit_time_ms_min", key) > 0.0,
        "{label}: {key}: fit_time_ms_min must be positive"
    );
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

fn assert_glmm_objective_constants_fenced(rust: &Value, r: &Value, key: &str) {
    assert_eq!(
        rust.get("objective_definition").and_then(Value::as_str),
        Some("profiled_glmm_deviance"),
        "{key}: Rust GLMM objective definition must stay explicit"
    );
    assert_eq!(
        rust.get("response_constants").and_then(Value::as_str),
        Some("dropped"),
        "{key}: Rust GLMM response-constant convention must stay explicit"
    );
    assert_eq!(
        r.get("objective_definition").and_then(Value::as_str),
        Some("minus_two_loglik"),
        "{key}: lme4 GLMM objective definition must stay explicit"
    );
    assert_eq!(
        r.get("response_constants").and_then(Value::as_str),
        Some("included"),
        "{key}: lme4 GLMM response-constant convention must stay explicit"
    );
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
        let mut families = BTreeSet::new();
        for (key, record) in by_key {
            if !is_glmm(record) {
                continue;
            }
            glmm_keys.insert(key.clone());
            families.insert((
                field_str(record, "family", key).to_owned(),
                field_str(record, "link", key).to_owned(),
            ));
            assert_ne!(
                status(record),
                "not_implemented",
                "{label}: GLMM row must be wired or explicitly classified: {key}"
            );
        }
        assert_eq!(
            glmm_keys, expected_keys,
            "{label}: GLMM comparison surface changed; update this gate and the GLMM finish beads deliberately"
        );
        for required in [
            ("Bernoulli".to_string(), "Logit".to_string()),
            ("Binomial".to_string(), "Logit".to_string()),
            ("Gamma".to_string(), "Log".to_string()),
            ("Poisson".to_string(), "Log".to_string()),
        ] {
            assert!(
                families.contains(&required),
                "{label}: missing GLMM family/link coverage for {required:?}"
            );
        }
    }

    for row in EXPECTED_GLMM_ROWS {
        let key = expected_row_key(*row);
        let rust_record = rust_by_key
            .get(&key)
            .unwrap_or_else(|| panic!("rust_results.json missing GLMM row {key}"));
        let r_record = r_by_key
            .get(&key)
            .unwrap_or_else(|| panic!("lme4_results.json missing GLMM row {key}"));

        assert_eq!(
            status(rust_record),
            row.status,
            "rust_results.json: unexpected status for {key}"
        );
        assert_eq!(
            status(r_record),
            row.status,
            "lme4_results.json: unexpected status for {key}"
        );

        if row.status == "ok" {
            assert_ok_glmm_payload(rust_record, "rust_results.json", &key);
            assert_ok_glmm_payload(r_record, "lme4_results.json", &key);
            assert_glmm_objective_constants_fenced(rust_record, r_record, &key);
        } else {
            let rust_reason = rust_record
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("");
            let r_reason = r_record.get("error").and_then(Value::as_str).unwrap_or("");
            assert!(
                rust_reason.contains("MIXEDMODELS_INCLUDE_STRESS=1"),
                "rust_results.json: skipped stress row must name opt-in env var: {key}"
            );
            assert!(
                r_reason.contains("MIXEDMODELS_INCLUDE_STRESS=1"),
                "lme4_results.json: skipped stress row must name opt-in env var: {key}"
            );
        }
    }
}

#[test]
fn glmm_lme4_numeric_agreement_is_gated_or_explicitly_allowlisted() {
    let rust = read_json("comparison/rust_results.json");
    let r = read_json("comparison/lme4_results.json");
    let rust_by_key = results_by_key(&rust, "rust_results.json");
    let r_by_key = results_by_key(&r, "lme4_results.json");

    let matching = MATCHING_GLMM_GATES
        .iter()
        .map(|gate| expected_row_key(gate.row))
        .collect::<BTreeSet<_>>();
    let known_gaps = KNOWN_NUMERIC_GAPS
        .iter()
        .map(|gap| expected_row_key(gap.row))
        .collect::<BTreeSet<_>>();
    let ok_keys = EXPECTED_GLMM_ROWS
        .iter()
        .filter(|row| row.status == "ok")
        .map(|row| expected_row_key(*row))
        .collect::<BTreeSet<_>>();
    let classified = matching
        .union(&known_gaps)
        .cloned()
        .collect::<BTreeSet<_>>();
    assert_eq!(
        ok_keys, classified,
        "every ok GLMM row must be either a numeric gate or an explicit known gap"
    );

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

    for gap in KNOWN_NUMERIC_GAPS {
        let key = expected_row_key(gap.row);
        assert!(
            !gap.reason.is_empty(),
            "{key}: known GLMM numeric gap must carry a reason"
        );
        let rust_record = rust_by_key
            .get(&key)
            .unwrap_or_else(|| panic!("rust_results.json missing GLMM row {key}"));
        let r_record = r_by_key
            .get(&key)
            .unwrap_or_else(|| panic!("lme4_results.json missing GLMM row {key}"));

        let rust_beta = numeric_array(rust_record, "beta", &key);
        let r_beta = numeric_array(r_record, "beta", &key);
        let beta_delta = max_abs_delta(&rust_beta, &r_beta, &format!("{key}: beta"));
        let sigma_delta =
            (field_f64(rust_record, "sigma", &key) - field_f64(r_record, "sigma", &key)).abs();
        let beta_passes = within_tol(beta_delta, max_abs(&r_beta), BETA_ABS_TOL, BETA_REL_TOL);
        let sigma_passes = within_tol(
            sigma_delta,
            field_f64(r_record, "sigma", &key),
            SIGMA_ABS_TOL,
            SIGMA_REL_TOL,
        );
        assert!(
            !(beta_passes && sigma_passes),
            "{key}: known gap now passes beta/sigma tolerances; remove it from KNOWN_NUMERIC_GAPS and promote it to MATCHING_GLMM_GATES"
        );
    }

    let key = expected_row_key(GOPHERDAT2);
    let rust_record = rust_by_key
        .get(&key)
        .unwrap_or_else(|| panic!("rust_results.json missing GLMM row {key}"));
    let r_record = r_by_key
        .get(&key)
        .unwrap_or_else(|| panic!("lme4_results.json missing GLMM row {key}"));
    assert_eq!(
        rust_record.get("is_singular").and_then(Value::as_bool),
        Some(false),
        "{key}: Rust singular flag should remain explicit"
    );
    assert_eq!(
        r_record.get("is_singular").and_then(Value::as_bool),
        Some(true),
        "{key}: lme4 singular flag should remain explicit until the boundary convention is reconciled"
    );
}

#[test]
fn glmm_known_fast_path_gaps_match_mixedmodels_jl_fast_oracle() {
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
        assert_eq!(
            status(rust_record),
            "ok",
            "{key}: fast-oracle row must be available in rust_results.json"
        );
        assert_eq!(
            rust_record
                .get("objective_definition")
                .and_then(Value::as_str),
            Some("profiled_glmm_deviance"),
            "{key}: Rust objective must be the profiled GLMM deviance used by the fast oracle"
        );

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
                "{key}: Rust/MixedModels.jl fast comparable beta delta {beta_delta:.8} exceeds {beta_abs_tol}"
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
                "{key}: Rust/MixedModels.jl fast comparable theta delta {theta_delta:.8} exceeds {theta_abs_tol}"
            );
        }
    }

    let expected_covered = [
        expected_row_key(CONTRACEPTION_INTERCEPT),
        expected_row_key(CONTRACEPTION_SLOPE),
        expected_row_key(GROUSETICKS),
        expected_row_key(VERBAGG),
    ]
    .into_iter()
    .collect::<BTreeSet<_>>();
    assert_eq!(
        covered, expected_covered,
        "fast-oracle fixture must cover the GLMM known gaps attributed to MixedModels.jl fast=true semantics"
    );
}

#[test]
fn glmm_report_contains_no_unclassified_numeric_disagreements() {
    let report =
        fs::read_to_string(repo_root().join("comparison").join("REPORT.md")).expect("read report");
    assert!(
        !report.contains("numeric_disagreement"),
        "GLMM numeric disagreements must be classified in comparison/REPORT.md"
    );
    for required in [
        "culcitalogreg Binomial/AGQ is accepted by the row-specific 2e-3 beta gate",
        "Gamma/Log dispersion and theta conventions are not treated as an lme4-only oracle",
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

#[test]
fn gamma_mixedmodels_jl_fixture_remains_the_gamma_oracle() {
    let fixture = read_json("tests/fixtures/parity/gamma_glmm_engines.json");
    assert_eq!(
        fixture.get("schema_version").and_then(Value::as_str),
        Some("1.0.0")
    );
    assert!(
        fixture
            .get("source")
            .and_then(Value::as_str)
            .unwrap_or("")
            .contains("MixedModels.jl"),
        "Gamma GLMM fixture must identify the MixedModels.jl reference source"
    );

    let engines = fixture
        .get("engines")
        .and_then(Value::as_array)
        .expect("gamma fixture must record engine references");
    let mixedmodels = engines
        .iter()
        .find(|engine| engine.get("engine").and_then(Value::as_str) == Some("MixedModels.jl"))
        .expect("gamma fixture must include MixedModels.jl");
    assert_eq!(
        mixedmodels.get("status").and_then(Value::as_str),
        Some("fit")
    );
    assert_eq!(
        mixedmodels.get("verdict").and_then(Value::as_str),
        Some("parity_reference")
    );
    let rust_reference = nested_object(&fixture, "rust_reference", "gamma_glmm_engines.json");
    let rust_beta = numeric_array(rust_reference, "beta", "gamma fixture rust_reference");
    let mm_beta = numeric_array(mixedmodels, "beta", "gamma fixture MixedModels.jl");
    let beta_delta = max_abs_delta(&rust_beta, &mm_beta, "gamma fixture Rust/MixedModels beta");
    assert!(
        beta_delta <= 2e-5,
        "gamma fixture Rust/MixedModels.jl beta delta {beta_delta:.8} exceeds oracle tolerance"
    );

    let rust_theta = numeric_array(rust_reference, "theta", "gamma fixture rust_reference");
    let mm_theta = numeric_array(mixedmodels, "theta", "gamma fixture MixedModels.jl");
    let theta_delta = max_abs_delta(
        &rust_theta,
        &mm_theta,
        "gamma fixture Rust/MixedModels theta",
    );
    assert!(
        theta_delta <= 1e-7,
        "gamma fixture Rust/MixedModels.jl theta delta {theta_delta:.8} exceeds oracle tolerance"
    );

    let objective_delta = (field_f64(rust_reference, "objective", "gamma fixture rust_reference")
        - field_f64(mixedmodels, "objective", "gamma fixture MixedModels.jl"))
    .abs();
    assert!(
        objective_delta <= 1e-7,
        "gamma fixture Rust/MixedModels.jl objective delta {objective_delta:.8} exceeds oracle tolerance"
    );

    let lme4 = engines
        .iter()
        .find(|engine| engine.get("engine").and_then(Value::as_str) == Some("lme4::glmer"))
        .expect("gamma fixture must include lme4::glmer");
    assert_eq!(lme4.get("status").and_then(Value::as_str), Some("fit"));
    assert_eq!(
        lme4.get("verdict").and_then(Value::as_str),
        Some("documented_divergence"),
        "Gamma lme4 result must stay recorded as a non-oracle comparison point"
    );
    assert!(
        lme4.get("note")
            .and_then(Value::as_str)
            .unwrap_or("")
            .contains("not as the sole oracle"),
        "Gamma lme4 divergence note must explain why it is not the sole oracle"
    );
    let notes = fixture
        .get("notes")
        .and_then(Value::as_array)
        .expect("gamma fixture must record notes");
    assert!(
        notes
            .iter()
            .any(|note| note.as_str().unwrap_or("").contains("not be promoted")),
        "Gamma fixture notes must fence glmer as a drift sentinel rather than the oracle"
    );
}
