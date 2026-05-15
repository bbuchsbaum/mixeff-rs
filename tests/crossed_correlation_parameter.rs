#![cfg(feature = "unstable-internals")]

use mixeff_rs::compiler::{DiagnosticCode, VarCorrEntryKind};
use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::{DataFrame, LinearMixedModel};

// toy: 8 rows of crossed person × firm earnings; tests the
// crossed-correlation diagnostic path, not numerical fit accuracy.
fn earnings_data() -> DataFrame {
    let mut data = DataFrame::new();
    data.add_numeric(
        "earnings",
        vec![40.0, 42.0, 47.0, 45.0, 53.0, 50.0, 55.0, 59.0],
    )
    .unwrap();
    data.add_numeric("experience", vec![1.0, 2.0, 4.0, 3.0, 6.0, 5.0, 7.0, 8.0])
        .unwrap();
    data.add_categorical(
        "person",
        vec!["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4"]
            .into_iter()
            .map(str::to_string)
            .collect(),
    )
    .unwrap();
    data.add_categorical(
        "firm",
        vec!["f1", "f2", "f1", "f3", "f2", "f3", "f1", "f2"]
            .into_iter()
            .map(str::to_string)
            .collect(),
    )
    .unwrap();
    data
}

#[test]
fn crossed_scalar_random_intercepts_report_absent_cross_block_correlation() {
    let data = earnings_data();
    let formula = parse_formula("earnings ~ 1 + experience + (1 | person) + (1 | firm)").unwrap();
    let model = LinearMixedModel::new(formula, &data, None).unwrap();
    let artifact = model.compiler_artifact();

    assert!(artifact
        .covariance_parameter_traces
        .iter()
        .flat_map(|trace| trace.varcorr_entries.iter())
        .all(|entry| entry.kind == VarCorrEntryKind::StandardDeviation));

    let diagnostic = artifact
        .diagnostics
        .iter()
        .find(|diagnostic| {
            diagnostic.code == DiagnosticCode::CovarianceAssumption
                && diagnostic.message.contains("person")
                && diagnostic.message.contains("firm")
                && diagnostic.message.contains("no correlation parameter")
        })
        .expect("separate crossed scalar random intercepts should explain absent correlation");
    assert_eq!(
        diagnostic
            .payload
            .get("correlation_parameter")
            .and_then(|value| value.as_str()),
        Some("not_estimated")
    );
    assert!(diagnostic
        .suggested_actions
        .iter()
        .any(|action| action.contains("cross-block correlation is fixed absent")));
}

#[test]
fn vector_random_effect_reports_within_block_correlation_parameter() {
    let data = earnings_data();
    let formula = parse_formula("earnings ~ 1 + experience + (1 + experience | person)").unwrap();
    let model = LinearMixedModel::new(formula, &data, None).unwrap();
    let artifact = model.compiler_artifact();

    assert!(artifact
        .covariance_parameter_traces
        .iter()
        .flat_map(|trace| trace.varcorr_entries.iter())
        .any(|entry| {
            entry.kind == VarCorrEntryKind::Correlation
                && entry.label == "corr(experience,intercept)"
        }));
}

#[test]
fn nested_scalar_random_intercepts_do_not_claim_crossed_correlation() {
    let data = earnings_data();
    let formula =
        parse_formula("earnings ~ 1 + experience + (1 | firm) + (1 | firm:person)").unwrap();
    let model = LinearMixedModel::new(formula, &data, None).unwrap();

    assert!(!model
        .compiler_artifact()
        .diagnostics
        .iter()
        .any(|diagnostic| {
            diagnostic.code == DiagnosticCode::CovarianceAssumption
                && diagnostic.message.contains("no correlation parameter")
        }));
}
