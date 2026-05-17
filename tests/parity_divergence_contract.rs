#![cfg(feature = "unstable-internals")]

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;

use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RowKey {
    dataset: &'static str,
    formula: &'static str,
    family: &'static str,
    link: &'static str,
    estimator: &'static str,
}

#[derive(Debug, Deserialize)]
struct Scorecard {
    row: Vec<ScorecardRow>,
}

#[derive(Debug, Deserialize)]
struct ScorecardRow {
    dataset: String,
    formula: String,
    family: String,
    link: String,
    estimator: String,
    #[serde(rename = "class")]
    class_name: String,
    reference: String,
    #[serde(default)]
    reason: Option<String>,
    #[serde(default)]
    issue_id: Option<String>,
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn key(dataset: &str, formula: &str, family: &str, link: &str, estimator: &str) -> String {
    format!("{dataset}\n{formula}\n{family}\n{link}\n{estimator}")
}

fn row_key(row: RowKey) -> String {
    key(
        row.dataset,
        row.formula,
        row.family,
        row.link,
        row.estimator,
    )
}

fn record_key(record: &Value) -> String {
    key(
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

fn scorecard_key(row: &ScorecardRow) -> String {
    key(
        &row.dataset,
        &row.formula,
        &row.family,
        &row.link,
        &row.estimator,
    )
}

fn read_json(path: &str) -> Value {
    let path = repo_root().join(path);
    serde_json::from_str(
        &fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {path:?}: {error}")),
    )
    .unwrap_or_else(|error| panic!("parse {path:?}: {error}"))
}

fn scorecard_by_key() -> BTreeMap<String, ScorecardRow> {
    let path = repo_root().join("comparison/parity_scorecard.toml");
    let scorecard: Scorecard = toml::from_str(
        &fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {path:?}: {error}")),
    )
    .unwrap_or_else(|error| panic!("parse {path:?}: {error}"));
    scorecard
        .row
        .into_iter()
        .map(|row| (scorecard_key(&row), row))
        .collect()
}

fn results_by_key(path: &str) -> BTreeMap<String, Value> {
    let json = read_json(path);
    json.get("results")
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{path}: missing results[]"))
        .iter()
        .map(|record| (record_key(record), record.clone()))
        .collect()
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

fn field_bool(record: &Value, field: &str, key: &str) -> bool {
    record
        .get(field)
        .and_then(Value::as_bool)
        .unwrap_or_else(|| panic!("{key}: missing boolean `{field}`"))
}

fn numeric_array(record: &Value, field: &str, key: &str) -> Vec<f64> {
    record
        .get(field)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{key}: missing array `{field}`"))
        .iter()
        .map(|value| {
            value
                .as_f64()
                .unwrap_or_else(|| panic!("{key}: `{field}` contains non-numeric {value}"))
        })
        .collect()
}

fn max_abs_delta(left: &[f64], right: &[f64], label: &str) -> f64 {
    assert_eq!(left.len(), right.len(), "{label}: vector length mismatch");
    left.iter()
        .zip(right.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max)
}

fn scorecard_row(scorecard: &BTreeMap<String, ScorecardRow>, row: RowKey) -> &ScorecardRow {
    let key = row_key(row);
    scorecard
        .get(&key)
        .unwrap_or_else(|| panic!("scorecard missing row {key}"))
}

fn comparison_row<'a>(rows: &'a BTreeMap<String, Value>, row: RowKey, label: &str) -> &'a Value {
    let key = row_key(row);
    rows.get(&key)
        .unwrap_or_else(|| panic!("{label} missing row {key}"))
}

const CBPP: RowKey = RowKey {
    dataset: "cbpp",
    formula: "incidence / size ~ 1 + period + (1 | herd)",
    family: "Binomial",
    link: "Logit",
    estimator: "Laplace",
};

const CONTRACEPTION_INTERCEPT: RowKey = RowKey {
    dataset: "contraception",
    formula: "use ~ 1 + age + livch + urban + (1 | dist)",
    family: "Binomial",
    link: "Logit",
    estimator: "Laplace",
};

const CONTRACEPTION_SLOPE: RowKey = RowKey {
    dataset: "contraception",
    formula: "use ~ 1 + age + livch + urban + (1 + urban | dist)",
    family: "Binomial",
    link: "Logit",
    estimator: "Laplace",
};

const CULCITA_LAPLACE: RowKey = RowKey {
    dataset: "culcitalogreg",
    formula: "predation ~ ttt + (1 | block)",
    family: "Binomial",
    link: "Logit",
    estimator: "Laplace",
};

const CULCITA_AGQ: RowKey = RowKey {
    dataset: "culcitalogreg",
    formula: "predation ~ ttt + (1 | block)",
    family: "Binomial",
    link: "Logit",
    estimator: "AGQ",
};

const GOPHERDAT2: RowKey = RowKey {
    dataset: "gopherdat2",
    formula: "shells ~ year + prev + offset(log(Area)) + (1 | Site)",
    family: "Poisson",
    link: "Log",
    estimator: "Laplace",
};

const VERBAGG: RowKey = RowKey {
    dataset: "verbagg",
    formula: "r2 ~ 1 + Anger + Gender + btype + situ + mode + (1 | id) + (1 | item)",
    family: "Binomial",
    link: "Logit",
    estimator: "Laplace",
};

const NESTED_EXPLICIT: RowKey = RowKey {
    dataset: "nested_constant_response",
    formula: "logterrisize ~ 1 + spm + (1 | studyarea) + (1 | studyarea:teriid)",
    family: "Gaussian",
    link: "Identity",
    estimator: "REML",
};

const NESTED_SLASH: RowKey = RowKey {
    dataset: "nested_constant_response",
    formula: "logterrisize ~ 1 + spm + (1 | studyarea/teriid)",
    family: "Gaussian",
    link: "Identity",
    estimator: "REML",
};

const SINGULAR_MAXIMAL: RowKey = RowKey {
    dataset: "singular",
    formula: "y ~ 1 + A * B * C + (A * B * C | group)",
    family: "Gaussian",
    link: "Identity",
    estimator: "REML",
};

const SINGULAR_DOUBLE_BAR: RowKey = RowKey {
    dataset: "singular",
    formula: "y ~ 1 + A * B * C + (A * B * C || group)",
    family: "Gaussian",
    link: "Identity",
    estimator: "REML",
};

#[test]
fn glmm_fast_pirls_divergences_are_quantified_and_kept_non_lme4() {
    let scorecard = scorecard_by_key();
    let rust = results_by_key("comparison/rust_results.json");
    let lme4 = results_by_key("comparison/lme4_results.json");

    let expectations = [
        (CBPP, 0.03, 0.05, "fast_pirls_profiled_glmm"),
        (
            CONTRACEPTION_INTERCEPT,
            0.02,
            0.03,
            "MixedModels.jl fast=true",
        ),
        (CONTRACEPTION_SLOPE, 0.03, 0.04, "MixedModels.jl fast=true"),
        (CULCITA_AGQ, 0.8, 1.0, "fast_pirls_profiled_glmm_agq"),
        (VERBAGG, 0.08, 0.09, "MixedModels.jl fast=true"),
    ];

    for (row, min_beta_delta, max_beta_delta, reference) in expectations {
        let key = row_key(row);
        let score = scorecard_row(&scorecard, row);
        assert_eq!(score.class_name, "documented_divergence", "{key}");
        assert_eq!(score.reference, reference, "{key}");
        assert_eq!(
            score.issue_id.as_deref(),
            Some("bd-01KRV8F0C4X8W7S5XGQYFP8P48"),
            "{key}: GLMM fast-PIRLS divergence should route to the GLMM child mote"
        );
        let reason = score.reason.as_deref().unwrap_or("");
        assert!(
            reason.contains("fast-PIRLS") && reason.contains("lme4"),
            "{key}: reason must name fast-PIRLS and lme4 divergence"
        );

        let rust_row = comparison_row(&rust, row, "rust_results.json");
        let lme4_row = comparison_row(&lme4, row, "lme4_results.json");
        assert_eq!(field_str(rust_row, "status", &key), "ok", "{key}");
        assert_eq!(field_str(lme4_row, "status", &key), "ok", "{key}");
        assert_eq!(
            field_str(rust_row, "response_constants", &key),
            "dropped",
            "{key}: Rust GLMM objective convention"
        );
        assert_eq!(
            field_str(lme4_row, "response_constants", &key),
            "included",
            "{key}: lme4 GLMM objective convention"
        );
        assert!(
            field_str(rust_row, "optimizer_return_code", &key).contains("REACHED")
                || field_str(rust_row, "optimizer_return_code", &key).contains("SUCCESS"),
            "{key}: Rust optimizer status must be recorded"
        );

        let beta_delta = max_abs_delta(
            &numeric_array(rust_row, "beta", &key),
            &numeric_array(lme4_row, "beta", &key),
            &format!("{key}: beta"),
        );
        assert!(
            beta_delta >= min_beta_delta && beta_delta <= max_beta_delta,
            "{key}: expected documented lme4 beta gap in [{min_beta_delta}, {max_beta_delta}], got {beta_delta}"
        );

        if row == CULCITA_AGQ {
            assert!(
                beta_delta > 0.8 && reason.contains("large beta gap"),
                "{key}: culcitalogreg AGQ must keep its large-gap diagnosis"
            );
            assert!(
                reason.contains("inference-impacting") && reason.contains("non-parity"),
                "{key}: culcitalogreg AGQ must be explicitly marked inference-impacting non-parity, not a soft pass"
            );
        }
    }
}

#[test]
fn culcitalogreg_laplace_is_promoted_to_joint_laplace_parity() {
    let scorecard = scorecard_by_key();
    let key = row_key(CULCITA_LAPLACE);
    let score = scorecard_row(&scorecard, CULCITA_LAPLACE);
    assert_eq!(score.class_name, "release_blocking_parity", "{key}");
    assert_eq!(score.reference, "lme4_joint_laplace", "{key}");
    assert_eq!(
        score.issue_id.as_deref(),
        Some("bd-01KRVGT0H37JYNYB5FA2EZD5CW")
    );
    let reason = score.reason.as_deref().unwrap_or("");
    assert!(
        reason.contains("fast=false") && reason.contains("objective"),
        "{key}: promoted GLMM row must name the certified joint path and objective evidence"
    );
}

#[test]
fn mixedmodels_fast_oracle_scope_is_explicit_for_large_profiled_rows() {
    let fixture = read_json("tests/fixtures/parity/glmm_fast_oracles.json");
    assert_eq!(
        fixture.get("reference_engine").and_then(Value::as_str),
        Some("MixedModels.jl 5.3.0")
    );
    assert_eq!(
        fixture.get("fit_mode").and_then(Value::as_str),
        Some("fast=true")
    );

    let rows = fixture
        .get("rows")
        .and_then(Value::as_array)
        .expect("glmm_fast_oracles.json rows[]");
    let covered = rows.iter().map(record_key).collect::<BTreeSet<_>>();
    let expected = [CONTRACEPTION_INTERCEPT, CONTRACEPTION_SLOPE, VERBAGG]
        .into_iter()
        .map(row_key)
        .collect::<BTreeSet<_>>();
    assert!(
        expected.is_subset(&covered),
        "large GLMM fast-profiled divergence rows should carry MixedModels.jl fast=true evidence"
    );
    assert!(
        !covered.contains(&row_key(CBPP)),
        "cbpp remains outside the MixedModels.jl fast-oracle fixture scope"
    );
}

#[test]
fn gopherdat2_keeps_coefficient_parity_but_documents_near_boundary_singularity_gap() {
    let scorecard = scorecard_by_key();
    let rust = results_by_key("comparison/rust_results.json");
    let lme4 = results_by_key("comparison/lme4_results.json");
    let key = row_key(GOPHERDAT2);
    let score = scorecard_row(&scorecard, GOPHERDAT2);
    assert_eq!(score.class_name, "documented_divergence");
    assert_eq!(score.reference, "lme4_numeric_without_objective_constants");
    assert_eq!(
        score.issue_id.as_deref(),
        Some("bd-01KRV8F8QDB9MVN08TQM5DK82P")
    );
    let reason = score.reason.as_deref().unwrap_or("");
    assert!(
        reason.contains("near-zero theta") && reason.contains("singular flag"),
        "{key}: scorecard reason should explain the diagnostic-threshold decision"
    );

    let rust_row = comparison_row(&rust, GOPHERDAT2, "rust_results.json");
    let lme4_row = comparison_row(&lme4, GOPHERDAT2, "lme4_results.json");
    let beta_delta = max_abs_delta(
        &numeric_array(rust_row, "beta", &key),
        &numeric_array(lme4_row, "beta", &key),
        &format!("{key}: beta"),
    );
    assert!(
        beta_delta <= 1e-3,
        "{key}: gopherdat2 should keep coefficient parity, got beta delta {beta_delta}"
    );
    assert!(!field_bool(rust_row, "is_singular", &key));
    assert!(field_bool(lme4_row, "is_singular", &key));
    assert!(
        numeric_array(rust_row, "theta", &key)[0] <= 1e-6,
        "{key}: Rust theta should be near the boundary even when the singular flag differs"
    );
    assert_eq!(field_str(rust_row, "response_constants", &key), "dropped");
    assert_eq!(field_str(lme4_row, "response_constants", &key), "included");
}

/// AC4 (issue bd-01KRVA2201SY7W2TSZEANCERG5): for GLMM rows where the
/// coefficients match but the objective constants differ, coefficient parity
/// must be pinned separately from objective comparability. Objective values
/// are explicitly *not* compared because the response-constant convention
/// differs; coefficient parity is asserted on its own with a tight tolerance.
#[test]
fn glmm_coefficient_parity_is_pinned_separately_from_objective_constants() {
    let scorecard = scorecard_by_key();
    let rust = results_by_key("comparison/rust_results.json");
    let lme4 = results_by_key("comparison/lme4_results.json");

    let mut checked = 0;
    for (key, row) in &scorecard {
        if row.reference != "lme4_numeric_without_objective_constants" {
            continue;
        }
        checked += 1;

        let rust_row = rust
            .get(key)
            .unwrap_or_else(|| panic!("rust_results.json missing {key}"));
        let lme4_row = lme4
            .get(key)
            .unwrap_or_else(|| panic!("lme4_results.json missing {key}"));

        // The objective convention genuinely differs, so any objective
        // comparison would be meaningless. We assert the divergence and
        // deliberately do NOT compare objective values.
        assert_eq!(
            field_str(rust_row, "response_constants", key),
            "dropped",
            "{key}: Rust GLMM objective drops response constants"
        );
        assert_eq!(
            field_str(lme4_row, "response_constants", key),
            "included",
            "{key}: lme4 GLMM objective includes response constants"
        );
        let objective_gap =
            (field_f64(rust_row, "objective", key) - field_f64(lme4_row, "objective", key)).abs();
        assert!(
            objective_gap > 1.0,
            "{key}: objective constants must be demonstrably non-comparable, gap {objective_gap}"
        );

        // Coefficient parity is pinned independently and tightly.
        let beta_delta = max_abs_delta(
            &numeric_array(rust_row, "beta", key),
            &numeric_array(lme4_row, "beta", key),
            &format!("{key}: beta"),
        );
        assert!(
            beta_delta <= 1e-3,
            "{key}: coefficient parity must hold independently of the objective, got {beta_delta}"
        );

        // The scorecard reason must record that objective constants are the
        // excluded dimension, not the coefficients.
        let reason = scorecard
            .get(key)
            .and_then(|r| r.reason.as_deref())
            .unwrap_or("");
        assert!(
            reason.contains("objective constants") || reason.contains("objective"),
            "{key}: scorecard must record that objective constants are the excluded dimension"
        );
    }

    assert!(
        checked >= 2,
        "expected the numeric-without-objective-constants contract to cover \
         both the release-blocking and documented-divergence GLMM cases"
    );
}

#[test]
fn boundary_pathology_lmm_divergences_stay_diagnostic_not_parity() {
    let scorecard = scorecard_by_key();
    let rust = results_by_key("comparison/rust_results.json");
    let lme4 = results_by_key("comparison/lme4_results.json");

    for row in [
        NESTED_EXPLICIT,
        NESTED_SLASH,
        SINGULAR_MAXIMAL,
        SINGULAR_DOUBLE_BAR,
    ] {
        let key = row_key(row);
        let score = scorecard_row(&scorecard, row);
        assert_eq!(score.class_name, "documented_divergence", "{key}");
        assert_eq!(
            score.issue_id.as_deref(),
            Some("bd-01KRV8FJ10V9HS3RX183X88B6A"),
            "{key}: LMM pathology divergence should route to the LMM child mote"
        );
        assert!(
            score.reason.as_deref().unwrap_or("").contains("diagnostic")
                || score.reason.as_deref().unwrap_or("").contains("pathology")
                || score.reason.as_deref().unwrap_or("").contains("boundary"),
            "{key}: LMM pathology reason must name the diagnostic/pathology contract"
        );
    }

    let nested_explicit_r = comparison_row(&lme4, NESTED_EXPLICIT, "lme4_results.json");
    assert_eq!(
        field_str(nested_explicit_r, "status", &row_key(NESTED_EXPLICIT)),
        "error"
    );

    let nested_slash_rust = comparison_row(&rust, NESTED_SLASH, "rust_results.json");
    let nested_slash_lme4 = comparison_row(&lme4, NESTED_SLASH, "lme4_results.json");
    let nested_delta = (field_f64(nested_slash_rust, "objective", &row_key(NESTED_SLASH))
        - field_f64(nested_slash_lme4, "objective", &row_key(NESTED_SLASH)))
    .abs();
    assert!(
        nested_delta > 40.0,
        "nested shorthand should remain a clearly documented constant-response divergence"
    );

    let singular_max = comparison_row(&rust, SINGULAR_MAXIMAL, "rust_results.json");
    assert_eq!(
        field_str(
            singular_max,
            "optimizer_return_code",
            &row_key(SINGULAR_MAXIMAL)
        ),
        "MAXEVAL_REACHED"
    );
    let singular_max_delta = max_abs_delta(
        &numeric_array(singular_max, "beta", &row_key(SINGULAR_MAXIMAL)),
        &numeric_array(
            comparison_row(&lme4, SINGULAR_MAXIMAL, "lme4_results.json"),
            "beta",
            &row_key(SINGULAR_MAXIMAL),
        ),
        "singular maximal beta",
    );
    assert!(singular_max_delta > 10.0);

    let singular_double = comparison_row(&rust, SINGULAR_DOUBLE_BAR, "rust_results.json");
    let singular_double_lme4 = comparison_row(&lme4, SINGULAR_DOUBLE_BAR, "lme4_results.json");
    assert!(field_bool(
        singular_double,
        "is_singular",
        &row_key(SINGULAR_DOUBLE_BAR)
    ));
    assert!(field_bool(
        singular_double_lme4,
        "is_singular",
        &row_key(SINGULAR_DOUBLE_BAR)
    ));
    assert!(
        numeric_array(singular_double, "theta", &row_key(SINGULAR_DOUBLE_BAR))
            .iter()
            .any(|theta| theta.abs() < 1e-4),
        "singular double-bar row should expose near-zero covariance directions"
    );
}
