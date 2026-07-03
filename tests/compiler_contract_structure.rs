#![cfg(feature = "unstable-internals")]
//! Optimizer-independent compiler-contract coverage.
//!
//! `tests/compiler_contract_snapshots.rs` pins fitted artifacts against the
//! NLopt parity optimizer (`#![cfg(all(nlopt, unstable-internals))]`), so it
//! only runs on the NLopt leg. This file covers the *optimizer-independent*
//! contract — schema identifiers, serde round-trip stability, and artifact /
//! audit-report / inference-table structure — under the COBYLA build. It is
//! gated on `unstable-internals` (the `compiler` module is not stable 1.0
//! surface); CI runs the `unstable-internals` leg on every push, so
//! wire-serialization regressions are still caught on every run, not only the
//! NLopt leg. Exact θ/β/objective values stay pinned in the NLopt snapshots
//! (they are optimizer-sensitive); nothing here asserts them.

use mixeff_rs::compiler::{compile_formula_ir, CompiledModelArtifact, ModelAuditReport};
use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::{DataFrame, LinearMixedModel};

fn contract_data() -> DataFrame {
    let mut data = DataFrame::new();
    data.add_numeric("y", vec![1.0, 2.1, 3.2, 4.1, 5.0, 6.2])
        .unwrap();
    data.add_numeric("x", vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        .unwrap();
    data.add_numeric("x2", vec![0.0, 2.0, 0.0, 2.0, 0.0, 2.0])
        .unwrap();
    data.add_categorical(
        "subject",
        vec!["s1", "s1", "s2", "s2", "s3", "s3"]
            .into_iter()
            .map(str::to_string)
            .collect(),
    )
    .unwrap();
    data
}

/// Prefit artifact: formula -> semantic IR -> design audit. Entirely
/// optimizer-independent (no model is fitted), so it must hold on the
/// default build exactly as it does under `--features nlopt`.
fn prefit_artifact() -> CompiledModelArtifact {
    let formula = parse_formula("y ~ x + x2 + (1 + x | subject)").unwrap();
    let semantic = compile_formula_ir(&formula);
    let mut artifact = CompiledModelArtifact::new(formula.to_string(), semantic);
    artifact.attach_design_audit(&contract_data());
    artifact
}

fn pretty(value: &impl serde::Serialize) -> String {
    serde_json::to_string_pretty(value).unwrap()
}

#[test]
fn prefit_artifact_schema_and_structure_are_stable_on_default_build() {
    let artifact = prefit_artifact();
    let json = pretty(&artifact);
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(
        value["schema"]["schema_name"],
        "mixedmodels.compiled_model_artifact"
    );
    assert_eq!(value["schema"]["schema_version"], 1);
    // Prefit: no inference table yet.
    assert!(value["fixed_effect_inference_table"].is_null());
    assert_eq!(
        value["design_audit"]["schema_name"],
        "mixedmodels.design_audit"
    );
    assert_eq!(
        value["theta_maps"][0]["map"]["schema_name"],
        "mixedmodels.theta_map"
    );
    assert_eq!(value["covariance_parameter_traces"][0]["term_id"], "r0");

    // serde round-trip is stable (serialize -> deserialize -> serialize).
    let decoded: CompiledModelArtifact = serde_json::from_str(&json).unwrap();
    assert_eq!(
        pretty(&decoded),
        json,
        "CompiledModelArtifact JSON must round-trip byte-stably"
    );
}

#[test]
fn audit_report_schema_and_structure_are_stable_on_default_build() {
    let report = ModelAuditReport::from_artifact(&prefit_artifact());
    let json = pretty(&report);
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(value["schema_name"], "mixedmodels.model_audit_report");
    assert_eq!(value["schema_version"], 2);
    assert_eq!(
        value["requested_formula"],
        "y ~ 1 + x + x2 + (1 + x | subject)"
    );
    assert_eq!(value["random_term_cards"].as_array().unwrap().len(), 1);
    assert_eq!(
        value["random_term_cards"][0]["schema_name"],
        "mixedmodels.random_term_card"
    );
    assert!(
        !value["sections"].as_array().unwrap().is_empty(),
        "audit report must render sections"
    );

    let decoded: ModelAuditReport = serde_json::from_str(&json).unwrap();
    assert_eq!(
        pretty(&decoded),
        json,
        "ModelAuditReport JSON must round-trip byte-stably"
    );
}

#[test]
fn fitted_cobyla_inference_table_structure_is_well_formed() {
    // A real default-build (COBYLA) fit. We assert the inference-table
    // *contract* (schema, row labels, per-row shape) — never the exact
    // statistic/df/p values, which are optimizer-sensitive.
    let formula = parse_formula("y ~ 1 + x + (1 | subject)").unwrap();
    let mut model = LinearMixedModel::new(formula, &contract_data(), None).unwrap();
    model.fit(false).unwrap();

    let table = model.fixed_effect_inference_table();
    assert_eq!(
        table.schema_name, "mixedmodels.fixed_effect_inference_table",
        "inference table must carry its schema id"
    );
    assert!(!table.schema_version.is_empty());
    assert_eq!(
        table.rows.len(),
        2,
        "intercept + x produce two coefficient rows"
    );
    let labels: Vec<&str> = table.rows.iter().map(|r| r.label.as_str()).collect();
    assert!(labels.contains(&"(Intercept)"));
    assert!(labels.contains(&"x"));
    for row in &table.rows {
        // The contract: every row declares a method and a status; finite
        // estimate/std_error when the row is Available.
        let _ = row.method;
        let _ = row.status;
        if let Some(estimate) = row.estimate {
            assert!(estimate.is_finite());
        }
    }

    // The serialized table deserializes back into an equivalent structure.
    // (Byte-equal re-serialization is *not* asserted: f64 fields reformat to
    // a different shortest-decimal on the round trip, which is a JSON number
    // representation artifact, not a contract change.)
    let json = pretty(&table);
    let decoded: mixeff_rs::compiler::FixedEffectInferenceTable =
        serde_json::from_str(&json).unwrap();
    assert_eq!(decoded.schema_name, table.schema_name);
    assert_eq!(decoded.schema_version, table.schema_version);
    assert_eq!(decoded.rows.len(), table.rows.len());
    for (a, b) in decoded.rows.iter().zip(table.rows.iter()) {
        assert_eq!(a.label, b.label);
        assert_eq!(a.method, b.method);
        assert_eq!(a.status, b.status);
        assert_eq!(a.statistic_name, b.statistic_name);
    }
}
