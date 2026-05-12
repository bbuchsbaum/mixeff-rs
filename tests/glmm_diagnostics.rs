#![cfg(not(feature = "nlopt"))]

use mixeff_rs::compiler::{DiagnosticCode, DiagnosticSeverity, FitStatus};
use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::{DataFrame, Family, GeneralizedLinearMixedModel, LinkFunction};
use mixeff_rs::types::Optimizer;

// toy: 5 groups × 5 obs of Gamma/Log; covers PIRLS, AGQ-refusal,
// boundary-θ, and near-unity-correlation diagnostic paths.
fn gamma_diagnostic_fixture() -> DataFrame {
    let group_effects = [-0.2, 0.0, 0.25, -0.05];
    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut group = Vec::new();

    for (g, group_effect) in group_effects.iter().enumerate() {
        for obs in 0..5 {
            let xv = obs as f64 - 2.0;
            let eta = 0.7 + 0.18 * xv + group_effect;
            let wiggle = 0.9 + 0.03 * ((g + obs) % 4) as f64;
            y.push(eta.exp() * wiggle);
            x.push(xv);
            group.push(format!("g{}", g + 1));
        }
    }

    let mut data = DataFrame::new();
    data.add_numeric("y", y).unwrap();
    data.add_numeric("x", x).unwrap();
    data.add_categorical("group", group).unwrap();
    data
}

#[test]
fn glmm_maxeval_stop_records_optimizer_nonconvergence_diagnostic() {
    let data = gamma_diagnostic_fixture();
    let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();
    let mut model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Gamma, Some(LinkFunction::Log))
            .unwrap();
    model.lmm.optsum.max_feval = 1;

    model.fit_with_options(true, 1, false).unwrap();

    let certificate = model
        .compiler_artifact()
        .optimizer_certificate
        .as_ref()
        .expect("fitted GLMM should record an optimizer certificate");
    assert_eq!(certificate.status, FitStatus::NotOptimized);
    assert_eq!(model.lmm.optsum.return_value, "MAXEVAL_REACHED");
    assert_eq!(
        certificate.evidence.optimizer_stop.return_code.as_deref(),
        Some("MAXEVAL_REACHED")
    );
    assert!(certificate.evidence.optimizer_stop.budget_exhausted);

    let diagnostic = certificate
        .diagnostics
        .iter()
        .find(|diagnostic| diagnostic.code == DiagnosticCode::OptimizerNonconvergence)
        .expect("budget exhaustion should be reported as optimizer nonconvergence");
    assert_eq!(diagnostic.severity, DiagnosticSeverity::Warning);
    assert!(diagnostic.message.contains("MAXEVAL_REACHED"));
    assert_eq!(
        diagnostic.payload.get("return_code"),
        Some(&serde_json::json!("MAXEVAL_REACHED"))
    );
    assert_eq!(
        diagnostic.payload.get("budget_exhausted"),
        Some(&serde_json::json!(true))
    );
    assert!(
        !certificate
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == DiagnosticCode::OptimizerNotAssessed),
        "a fitted optimizer failure should not reuse the pre-fit not-assessed diagnostic code"
    );
}

#[test]
fn glmm_invalid_agq_request_records_stable_artifact_diagnostic() {
    let data = gamma_diagnostic_fixture();
    let formula = parse_formula("y ~ 1 + x + (1 + x | group)").unwrap();
    let mut model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Gamma, Some(LinkFunction::Log))
            .unwrap();
    let feval_before = model.lmm.optsum.feval;

    let err = model
        .fit_with_options(true, 7, false)
        .expect_err("AGQ should be refused for vector-valued random effects");

    assert!(err.to_string().contains("n_agq = 7"));
    assert_eq!(
        model.lmm.optsum.feval, feval_before,
        "invalid AGQ request should fail before optimizer evaluations"
    );
    let diagnostic = model
        .compiler_artifact()
        .diagnostics
        .iter()
        .find(|diagnostic| diagnostic.code == DiagnosticCode::InvalidAgqRequest)
        .expect("invalid AGQ request should be recorded on the artifact");
    assert_eq!(diagnostic.severity, DiagnosticSeverity::Error);
    assert!(diagnostic.message.contains("n_agq = 7"));
    assert_eq!(diagnostic.affected_terms, vec!["(1 + x | group)"]);
    assert_eq!(diagnostic.payload.get("n_agq"), Some(&serde_json::json!(7)));
    assert_eq!(
        diagnostic.payload.get("random_effect_term_count"),
        Some(&serde_json::json!(1))
    );
    assert!(diagnostic
        .payload
        .get("reason")
        .and_then(|value| value.as_str())
        .is_some_and(|reason| reason.contains("requires exactly one scalar random-effects term")));
}

#[test]
fn glmm_final_pirls_failure_records_stable_artifact_diagnostic() {
    let data = gamma_diagnostic_fixture();
    let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();
    let mut model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Gamma, Some(LinkFunction::Log))
            .unwrap();
    model.lmm.optsum.optimizer = Optimizer::PatternSearch;
    model.lmm.optsum.initial = vec![f64::NAN];
    model.lmm.optsum.max_feval = 1;

    let err = model
        .fit_with_options(true, 1, false)
        .expect_err("final PIRLS update should reject nonfinite theta");

    assert!(err.to_string().contains("theta"));
    let diagnostic = model
        .compiler_artifact()
        .diagnostics
        .iter()
        .find(|diagnostic| diagnostic.code == DiagnosticCode::PirlsFailure)
        .expect("final PIRLS failure should be recorded on the artifact");
    assert_eq!(diagnostic.severity, DiagnosticSeverity::Error);
    assert!(diagnostic.message.contains("PIRLS failed"));
    assert_eq!(
        diagnostic.payload.get("theta_len"),
        Some(&serde_json::json!(1))
    );
    assert_eq!(
        diagnostic.payload.get("nonfinite_theta_indices"),
        Some(&serde_json::json!([0]))
    );
    assert!(diagnostic
        .payload
        .get("reason")
        .and_then(|value| value.as_str())
        .is_some_and(|reason| reason.contains("theta")));
}

#[test]
fn glmm_boundary_theta_records_boundary_parameter_diagnostic() {
    let data = gamma_diagnostic_fixture();
    let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();
    let mut model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Gamma, Some(LinkFunction::Log))
            .unwrap();
    model.lmm.optsum.optimizer = Optimizer::PatternSearch;
    model.lmm.optsum.initial = vec![0.0];
    model.lmm.optsum.max_feval = 1;

    model.fit_with_options(true, 1, false).unwrap();

    let certificate = model
        .compiler_artifact()
        .optimizer_certificate
        .as_ref()
        .expect("fitted GLMM should record an optimizer certificate");
    let diagnostic = certificate
        .diagnostics
        .iter()
        .find(|diagnostic| diagnostic.code == DiagnosticCode::BoundaryParameter)
        .expect("theta on its lower bound should be diagnosed as a boundary parameter");
    assert_eq!(diagnostic.severity, DiagnosticSeverity::Info);
    assert_eq!(diagnostic.affected_terms, vec!["covariance parameter 1"]);
    assert!(diagnostic.message.contains("covariance parameter 1"));
    assert!(!diagnostic.message.contains("theta[0]"));
    assert_eq!(
        diagnostic.payload.get("theta_index"),
        Some(&serde_json::json!(0))
    );
    assert_eq!(
        diagnostic.payload.get("lower_bound"),
        Some(&serde_json::json!(0.0))
    );
}

#[test]
fn glmm_near_unit_random_effect_correlation_records_diagnostic() {
    let data = gamma_diagnostic_fixture();
    let formula = parse_formula("y ~ 1 + x + (1 + x | group)").unwrap();
    let mut model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Gamma, Some(LinkFunction::Log))
            .unwrap();
    model.lmm.optsum.optimizer = Optimizer::PatternSearch;
    model.lmm.optsum.initial = vec![1.0, 1000.0, 0.001];
    model.lmm.optsum.max_feval = 1;

    model.fit_with_options(true, 1, false).unwrap();

    let diagnostic = model
        .compiler_artifact()
        .diagnostics
        .iter()
        .find(|diagnostic| diagnostic.code == DiagnosticCode::NearUnitRandomEffectCorrelation)
        .expect("near-unit random-effect correlation should be diagnosed");
    assert_eq!(diagnostic.severity, DiagnosticSeverity::Warning);
    assert_eq!(diagnostic.affected_terms, vec!["group"]);
    assert!(diagnostic.message.contains("group"));
    assert!(
        diagnostic
            .payload
            .get("correlation")
            .and_then(|value| value.as_f64())
            .is_some_and(|correlation| correlation.abs() >= 0.99),
        "near-unit correlation payload should record an absolute correlation above threshold"
    );
}
