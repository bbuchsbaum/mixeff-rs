use mixedmodels::compiler::{ConvergenceLevel, ConvergenceSource, ConvergenceVerdict};
use mixedmodels::compiler::{DiagnosticCode, DiagnosticSeverity};
use mixedmodels::datasets;
use mixedmodels::formula::parse_formula;
use mixedmodels::model::LinearMixedModel;

fn nested_formula() -> &'static str {
    "logterrisize ~ 1 + spm + (1 | studyarea) + (1 | studyarea:teriid)"
}

#[test]
fn nested_constant_response_prefit_message_points_to_observation_unit() {
    let (data, _) = datasets::load("nested_constant_response").unwrap();
    let formula = parse_formula(nested_formula()).unwrap();
    let model = LinearMixedModel::new(formula, &data, None).unwrap();

    let diagnostic = model
        .compiler_artifact()
        .diagnostics
        .iter()
        .find(|diagnostic| {
            diagnostic.code == DiagnosticCode::NotIdentifiable
                && diagnostic
                    .message
                    .contains("response 'logterrisize' is constant")
                && diagnostic.message.contains("studyarea:teriid")
        })
        .expect("constant response within nested unit should be diagnosed before fitting");

    assert_eq!(diagnostic.severity, DiagnosticSeverity::Error);
    assert_eq!(
        diagnostic
            .payload
            .get("constant_response_levels")
            .and_then(|value| value.as_u64()),
        Some(12)
    );
    assert_eq!(
        diagnostic
            .payload
            .get("repeated_levels")
            .and_then(|value| value.as_u64()),
        Some(12)
    );
    assert!(diagnostic
        .payload
        .get("varying_numeric_columns")
        .and_then(|value| value.as_array())
        .unwrap()
        .iter()
        .any(|value| value.as_str() == Some("spm")));
    assert!(diagnostic
        .suggested_actions
        .iter()
        .any(|action| action.contains("aggregate to one row per lower-level unit")));
    assert!(diagnostic
        .suggested_actions
        .iter()
        .any(|action| action.contains("before changing optimizers")));

    let report = model.audit_report().to_text();
    assert!(report.contains("response 'logterrisize' is constant"));
    assert!(report.contains("studyarea:teriid"));
    assert!(report.contains("observation"));
    assert!(report.contains("optimizer"));
}

#[test]
fn nested_slash_formula_expands_to_same_diagnostic_target() {
    let (data, meta) = datasets::load("nested_constant_response").unwrap();
    let slash_formula = meta
        .fits
        .iter()
        .find(|fit| fit.formula.contains("studyarea/teriid"))
        .expect("slash formula is recorded in metadata");
    let formula = parse_formula(&slash_formula.formula).unwrap();
    let model = LinearMixedModel::new(formula, &data, None).unwrap();

    assert!(model
        .compiler_artifact()
        .diagnostics
        .iter()
        .any(|diagnostic| {
            diagnostic.code == DiagnosticCode::NotIdentifiable
                && diagnostic.message.contains("studyarea:teriid")
                && diagnostic.message.contains("numeric predictor")
        }));
}

#[test]
fn nested_constant_response_fit_attempt_keeps_structural_explanation() {
    let (data, _) = datasets::load("nested_constant_response").unwrap();
    let formula = parse_formula(nested_formula()).unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

    model.fit(true).unwrap();

    let verdict = ConvergenceVerdict::for_artifact(model.compiler_artifact());
    assert_eq!(verdict.level, ConvergenceLevel::Failed);
    assert!(matches!(
        verdict.source,
        ConvergenceSource::Structural | ConvergenceSource::Mixed
    ));

    let report = model.audit_report().to_text();
    assert!(report.contains("response 'logterrisize' is constant"));
    assert!(report.contains("aggregate to one row per lower-level unit"));
    assert!(report.contains("optimizer"));
}
