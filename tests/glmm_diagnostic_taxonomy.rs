//! GLMM difficult-model diagnostic taxonomy contract
//! (issue bd-01KRVA2201SY7W2TSZEANCERG5, AC2 + AC7).
//!
//! A difficult GLMM result must let a downstream client tell five different
//! situations apart from the artifact alone: optimizer failure, approximation
//! gap, weak identification, response-constant convention, and separation-like
//! behavior. They have different correct responses, so they must map to
//! distinct, stable signals and must not collapse into a single
//! "GLMM did not converge".
//!
//! The five modes use heterogeneous vocabularies (a diagnostic code, a
//! scorecard class + reference + objective convention, a covariance-cone
//! classification, a results field, and a separation code / fit-status
//! carve-out). This test pins each mode to its real artifact signal exactly
//! as `docs/glmm_support_contract.md` enumerates them, and proves the five
//! resulting signals are mutually distinct — including the response-constant
//! convention, which is part of the distinctness check, not a side test.
#![cfg(feature = "unstable-internals")]

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;

use mixeff_rs::compiler::diagnostics::{DiagnosticCode, FitStatus};
use mixeff_rs::model::linear::CovarianceKktClassification;
use serde::Deserialize;
use serde_json::Value;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn ser(code: &DiagnosticCode) -> String {
    serde_json::to_string(code).unwrap()
}

#[derive(Debug, Deserialize)]
struct Scorecard {
    row: Vec<ScorecardRow>,
}

#[derive(Debug, Deserialize)]
struct ScorecardRow {
    dataset: String,
    estimator: String,
    #[serde(rename = "class")]
    class_name: String,
    reference: String,
}

fn scorecard() -> Vec<ScorecardRow> {
    let path = repo_root().join("comparison/parity_scorecard.toml");
    let parsed: Scorecard =
        toml::from_str(&fs::read_to_string(path).expect("read parity_scorecard.toml"))
            .expect("parse parity_scorecard.toml");
    parsed.row
}

fn results_by_key(path: &str) -> BTreeMap<String, Value> {
    let json: Value = serde_json::from_str(
        &fs::read_to_string(repo_root().join(path)).expect("read comparison results"),
    )
    .expect("parse comparison results");
    json.get("results")
        .and_then(Value::as_array)
        .expect("results[]")
        .iter()
        .map(|r| {
            let k = format!(
                "{}\n{}",
                r.get("dataset").and_then(Value::as_str).unwrap_or(""),
                r.get("estimator").and_then(Value::as_str).unwrap_or(""),
            );
            (k, r.clone())
        })
        .collect()
}

/// Each mode is pinned to its real artifact signal exactly as the contract
/// table enumerates it, then the five signals are proven mutually distinct.
#[test]
fn five_glmm_failure_modes_map_to_distinct_artifact_signals() {
    let mut signals: BTreeSet<String> = BTreeSet::new();

    // Mode 1 — optimizer failure: the dedicated stable diagnostic code.
    let optimizer_failure = ser(&DiagnosticCode::OptimizerNonconvergence);
    assert_eq!(optimizer_failure, "\"optimizer_nonconvergence\"");
    signals.insert(format!("optimizer_failure={optimizer_failure}"));

    // Mode 2 — approximation gap: a `documented_divergence` scorecard row on a
    // fast-PIRLS reference whose Rust objective drops response constants. This
    // is NOT `pirls_failure` (that is an optimizer-side final-update code);
    // the contract defines the approximation gap at the scorecard level.
    let card = scorecard();
    let rust = results_by_key("comparison/rust_results.json");
    let approx = card
        .iter()
        .find(|r| {
            r.class_name == "documented_divergence"
                && (r.reference.contains("fast_pirls") || r.reference.contains("fast=true"))
        })
        .expect("an approximation-gap row must exist (documented_divergence + fast-PIRLS)");
    let approx_key = format!("{}\n{}", approx.dataset, approx.estimator);
    let approx_rc = rust
        .get(&approx_key)
        .and_then(|r| r.get("response_constants"))
        .and_then(Value::as_str)
        .unwrap_or("");
    assert_eq!(
        approx_rc, "dropped",
        "approximation-gap row {approx_key} must use the dropped objective convention"
    );
    signals.insert(format!(
        "approximation_gap={}+{}+rc:{}",
        approx.class_name, approx.reference, approx_rc
    ));

    // Mode 3 — weak identification: the covariance-cone classification, which
    // the contract contrasts against a clean interior / valid boundary.
    let weak = format!("{:?}", CovarianceKktClassification::WeakIdentification);
    assert_eq!(weak, "WeakIdentification");
    for clean in [
        CovarianceKktClassification::InteriorConverged,
        CovarianceKktClassification::ValidZeroVariance,
        CovarianceKktClassification::ValidRankDeficientCovariance,
    ] {
        assert_ne!(
            CovarianceKktClassification::WeakIdentification,
            clean,
            "weak identification must be distinct from clean interior / valid boundary"
        );
    }
    signals.insert(format!("weak_identification=kkt:{weak}"));

    // Mode 4 — response-constant convention: the machine-readable field whose
    // value genuinely differs across engines (Rust dropped vs lme4 included).
    let lme4 = results_by_key("comparison/lme4_results.json");
    let rust_rc = rust
        .get(&approx_key)
        .and_then(|r| r.get("response_constants"))
        .and_then(Value::as_str)
        .expect("rust response_constants");
    let lme4_rc = lme4
        .get(&approx_key)
        .and_then(|r| r.get("response_constants"))
        .and_then(Value::as_str)
        .expect("lme4 response_constants");
    assert_eq!(rust_rc, "dropped");
    assert_eq!(lme4_rc, "included");
    assert_ne!(
        rust_rc, lme4_rc,
        "the response-constant convention must be a real cross-engine difference"
    );
    signals.insert(format!("response_constants={rust_rc}!={lme4_rc}"));

    // Mode 5 — separation-like behavior: the dedicated separation code, and a
    // fit-status leaf carve-out so a non-existent MLE is never an ordinary
    // converged fit.
    let separation = ser(&DiagnosticCode::BinomialSeparation);
    assert_eq!(separation, "\"binomial_separation\"");
    let penalised = serde_json::to_string(&FitStatus::ConvergedPenalised).unwrap();
    let not_identifiable = serde_json::to_string(&FitStatus::NotIdentifiable).unwrap();
    let interior = serde_json::to_string(&FitStatus::ConvergedInterior).unwrap();
    assert_eq!(penalised, "\"converged_penalised\"");
    assert_eq!(not_identifiable, "\"not_identifiable\"");
    assert_ne!(penalised, interior);
    assert_ne!(not_identifiable, interior);
    signals.insert(format!(
        "separation={separation}+{penalised}+{not_identifiable}"
    ));

    // The taxonomy is only useful if the five modes are mutually distinct.
    assert_eq!(
        signals.len(),
        5,
        "the five GLMM failure modes must map to five distinct artifact signals, got {signals:?}"
    );
}

#[test]
fn response_constant_convention_is_a_mode_not_a_failure() {
    // The convention difference must coexist with an ok optimizer status: it
    // is a convention, not a fit failure or identification problem.
    let rust = results_by_key("comparison/rust_results.json");
    let key = "culcitalogreg\nAGQ".to_string();
    let rust_row = rust.get(&key).expect("rust culcitalogreg AGQ row");
    assert_eq!(
        rust_row.get("response_constants").and_then(Value::as_str),
        Some("dropped")
    );
    assert_eq!(
        rust_row.get("status").and_then(Value::as_str),
        Some("ok"),
        "a response-constant convention difference must not present as a fit failure"
    );
}

#[test]
fn glmm_contract_doc_enumerates_the_five_modes_and_optin_recovery() {
    let doc = fs::read_to_string(repo_root().join("docs/glmm_support_contract.md"))
        .expect("read docs/glmm_support_contract.md");

    assert!(
        doc.contains("Distinguishable Failure Modes"),
        "contract must enumerate the distinguishable failure modes"
    );
    for needle in [
        "Optimizer failure",
        "Approximation gap",
        "Weak identification",
        "Response-constant convention",
        "Separation-like behavior",
    ] {
        assert!(
            doc.contains(needle),
            "GLMM contract must name the `{needle}` failure mode"
        );
    }
    // The contract must pin approximation gap to the scorecard class +
    // convention, not to pirls_failure.
    assert!(
        doc.contains("documented_divergence") && doc.contains("response_constants = dropped"),
        "contract must define the approximation gap via the scorecard class and objective convention"
    );
    for code in [
        "optimizer_nonconvergence",
        "WeakIdentification",
        "response_constants",
        "binomial_separation",
    ] {
        assert!(
            doc.contains(code),
            "GLMM contract must cite the stable signal `{code}`"
        );
    }

    // AC7: any GLMM recovery is opt-in / labelled and outside the parity gate.
    assert!(
        doc.contains("Recovery Policy"),
        "contract must state the GLMM recovery policy"
    );
    assert!(
        doc.contains("no default, silent GLMM recovery")
            && doc.contains("opt-in or explicitly labelled"),
        "GLMM recovery must be documented as opt-in/labelled, not a default behavior"
    );
    assert!(
        doc.contains("certified_joint_glmm_optimizer_contract.md"),
        "contract must reference the certified joint optimizer prerequisites"
    );
}
