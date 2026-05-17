#![cfg(feature = "unstable-internals")]

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;

use serde::Deserialize;
use serde_json::Value;

use mixeff_rs::datasets;

const OBJECTIVE_ABS_TOL: f64 = 1e-2;
const OBJECTIVE_REL_TOL: f64 = 1e-5;
const BETA_ABS_TOL: f64 = 1e-3;
const BETA_REL_TOL: f64 = 1e-5;
const SIGMA_ABS_TOL: f64 = 1e-3;
const SIGMA_REL_TOL: f64 = 1e-4;

#[derive(Debug, Deserialize)]
struct Scorecard {
    schema_version: String,
    classes: Vec<String>,
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

fn scorecard() -> Scorecard {
    let path = repo_root().join("comparison/parity_scorecard.toml");
    toml::from_str(&fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}")))
        .unwrap_or_else(|e| panic!("parse {path:?}: {e}"))
}

fn read_json(path: &str) -> Value {
    let path = repo_root().join(path);
    serde_json::from_str(
        &fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}")),
    )
    .unwrap_or_else(|e| panic!("parse {path:?}: {e}"))
}

fn key(dataset: &str, formula: &str, family: &str, link: &str, estimator: &str) -> String {
    format!("{dataset}\n{formula}\n{family}\n{link}\n{estimator}")
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

fn field_f64(record: &Value, field: &str, key: &str) -> f64 {
    let value = record
        .get(field)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("{key}: missing numeric `{field}`"));
    assert!(value.is_finite(), "{key}: `{field}` is not finite: {value}");
    value
}

fn numeric_array(record: &Value, field: &str, key: &str) -> Vec<f64> {
    record
        .get(field)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{key}: missing numeric array `{field}`"))
        .iter()
        .map(|value| {
            let value = value
                .as_f64()
                .unwrap_or_else(|| panic!("{key}: `{field}` contains non-numeric {value}"));
            assert!(
                value.is_finite(),
                "{key}: `{field}` contains non-finite {value}"
            );
            value
        })
        .collect()
}

fn beta_by_name(record: &Value, key: &str) -> BTreeMap<String, f64> {
    let names = record
        .get("coef_names")
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{key}: missing coef_names"))
        .iter()
        .map(|value| {
            value
                .as_str()
                .unwrap_or_else(|| panic!("{key}: non-string coef name {value}"))
                .replace(": ", "")
        })
        .collect::<Vec<_>>();
    let beta = numeric_array(record, "beta", key);
    assert_eq!(names.len(), beta.len(), "{key}: beta/name length mismatch");
    names.into_iter().zip(beta).collect()
}

fn within_tol(delta: f64, reference_scale: f64, abs_tol: f64, rel_tol: f64) -> bool {
    delta <= abs_tol || delta <= reference_scale.abs().max(f64::EPSILON) * rel_tol
}

#[test]
fn parity_scorecard_covers_every_dataset_fit_once() {
    let scorecard = scorecard();
    assert_eq!(scorecard.schema_version, "1.0.0");
    let declared_classes = scorecard.classes.iter().cloned().collect::<BTreeSet<_>>();
    assert_eq!(
        declared_classes,
        [
            "documented_divergence",
            "performance_known_slow",
            "release_blocking_parity",
            "stress_opt_in",
            "unsupported_with_contract",
        ]
        .into_iter()
        .map(str::to_string)
        .collect::<BTreeSet<_>>()
    );

    let mut expected = BTreeSet::new();
    let mut release_blocking_without_expected = Vec::new();
    for case in datasets::iter_cases() {
        let case_key = key(
            &case.name,
            &case.fit.formula,
            &case.fit.family,
            &case.fit.link,
            &case.fit.estimator,
        );
        if case.fit.expected.is_none() {
            release_blocking_without_expected.push(case_key.clone());
        }
        expected.insert(case_key);
    }

    let mut actual = BTreeSet::new();
    for row in &scorecard.row {
        assert!(
            declared_classes.contains(&row.class_name),
            "{}: unknown scorecard class `{}`",
            scorecard_key(row),
            row.class_name
        );
        assert!(
            !row.reference.trim().is_empty(),
            "{}: reference must be explicit",
            scorecard_key(row)
        );
        let inserted = actual.insert(scorecard_key(row));
        assert!(inserted, "duplicate scorecard row: {}", scorecard_key(row));
    }

    assert_eq!(
        actual, expected,
        "comparison/parity_scorecard.toml must classify every datasets/* fit exactly once"
    );

    let release_keys = scorecard
        .row
        .iter()
        .filter(|row| row.class_name == "release_blocking_parity")
        .map(scorecard_key)
        .collect::<BTreeSet<_>>();
    release_blocking_without_expected.retain(|key| release_keys.contains(key));
    assert!(
        release_blocking_without_expected.is_empty(),
        "release-blocking parity rows need pinned expected values: {release_blocking_without_expected:#?}"
    );
}

#[test]
fn release_blocking_scorecard_rows_pass_checked_in_comparison_artifacts() {
    let scorecard = scorecard();
    let rust = read_json("comparison/rust_results.json");
    let r = read_json("comparison/lme4_results.json");
    let rust_by_key = results_by_key(&rust, "rust_results.json");
    let r_by_key = results_by_key(&r, "lme4_results.json");

    for row in scorecard
        .row
        .iter()
        .filter(|row| row.class_name == "release_blocking_parity")
    {
        let key = scorecard_key(row);
        let rust_row = rust_by_key
            .get(&key)
            .unwrap_or_else(|| panic!("rust_results.json missing release row {key}"));
        let r_row = r_by_key
            .get(&key)
            .unwrap_or_else(|| panic!("lme4_results.json missing release row {key}"));
        if row.reference == "lme4_joint_laplace" {
            let reason = row.reason.as_deref().unwrap_or("");
            assert!(
                row.issue_id.as_deref() == Some("bd-01KRVGT0H37JYNYB5FA2EZD5CW")
                    && reason.contains("fast=false")
                    && reason.contains("objective"),
                "{key}: joint GLMM release rows must name the certified fast=false evidence and phase-6 issue"
            );
            assert_eq!(
                rust_row.get("objective_definition").and_then(Value::as_str),
                Some("joint_glmm_laplace_deviance"),
                "{key}: joint GLMM release row must use the joint objective artifact"
            );
            assert_eq!(
                rust_row.get("response_constants").and_then(Value::as_str),
                Some("included"),
                "{key}: joint GLMM release row must retain response constants"
            );
            assert!(
                rust_row
                    .get("optimizer_return_code")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .starts_with("JOINT_LAPLACE:"),
                "{key}: joint GLMM release row must carry the labelled joint optimizer status"
            );
        }
        assert_eq!(
            rust_row.get("status").and_then(Value::as_str),
            Some("ok"),
            "{key}: Rust release row must be ok"
        );
        assert_eq!(
            r_row.get("status").and_then(Value::as_str),
            Some("ok"),
            "{key}: lme4 release row must be ok"
        );

        let rust_beta = beta_by_name(rust_row, &key);
        let r_beta = beta_by_name(r_row, &key);
        assert_eq!(
            rust_beta.keys().collect::<Vec<_>>(),
            r_beta.keys().collect::<Vec<_>>(),
            "{key}: coefficient-name sets differ"
        );
        let beta_delta = rust_beta
            .iter()
            .map(|(name, actual)| {
                (actual
                    - r_beta
                        .get(name)
                        .unwrap_or_else(|| panic!("{key}: missing beta for {name}")))
                .abs()
            })
            .fold(0.0_f64, f64::max);
        assert!(
            within_tol(
                beta_delta,
                r_beta.values().map(|v| v.abs()).fold(0.0, f64::max),
                BETA_ABS_TOL,
                BETA_REL_TOL,
            ),
            "{key}: beta delta {beta_delta:.6} exceeds release tolerance"
        );

        let sigma_delta =
            (field_f64(rust_row, "sigma", &key) - field_f64(r_row, "sigma", &key)).abs();
        assert!(
            within_tol(
                sigma_delta,
                field_f64(r_row, "sigma", &key),
                SIGMA_ABS_TOL,
                SIGMA_REL_TOL,
            ),
            "{key}: sigma delta {sigma_delta:.6} exceeds release tolerance"
        );

        let same_objective_convention = rust_row.get("response_constants").and_then(Value::as_str)
            == r_row.get("response_constants").and_then(Value::as_str);
        if same_objective_convention {
            let objective_delta = (field_f64(rust_row, "objective", &key)
                - field_f64(r_row, "objective", &key))
            .abs();
            assert!(
                within_tol(
                    objective_delta,
                    field_f64(r_row, "objective", &key),
                    OBJECTIVE_ABS_TOL,
                    OBJECTIVE_REL_TOL,
                ),
                "{key}: objective delta {objective_delta:.6} exceeds release tolerance"
            );
        } else {
            assert!(
                row.reference.contains("without_objective_constants"),
                "{key}: non-comparable objective conventions require an explicit scorecard reference"
            );
        }
    }
}

#[test]
fn non_release_scorecard_rows_are_explained_and_not_presented_as_lme4_parity() {
    let scorecard = scorecard();
    let report =
        fs::read_to_string(repo_root().join("comparison/REPORT.md")).expect("read REPORT.md");

    for row in scorecard
        .row
        .iter()
        .filter(|row| row.class_name != "release_blocking_parity")
    {
        let key = scorecard_key(row);
        let reason = row
            .reason
            .as_deref()
            .unwrap_or_else(|| panic!("{key}: non-release rows require a reason"));
        assert!(!reason.trim().is_empty(), "{key}: empty reason");

        if matches!(
            row.class_name.as_str(),
            "documented_divergence" | "unsupported_with_contract" | "performance_known_slow"
        ) {
            let issue = row
                .issue_id
                .as_deref()
                .unwrap_or_else(|| panic!("{key}: class `{}` requires issue_id", row.class_name));
            assert!(
                issue.starts_with("bd-"),
                "{key}: issue_id must be a mote id, got {issue}"
            );
        }

        assert!(
            report.contains(&format!("`{}`", row.dataset)),
            "{key}: comparison/REPORT.md must mention non-release row dataset"
        );

        if row.family != "Gaussian" {
            assert!(
                row.reference != "lme4" || row.class_name == "stress_opt_in",
                "{key}: GLMM non-release row must not be presented as ordinary lme4 parity"
            );
        }
    }
}
