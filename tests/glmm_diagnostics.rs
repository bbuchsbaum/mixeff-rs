#![cfg(not(feature = "nlopt"))]

use mixedmodels::compiler::{DiagnosticCode, DiagnosticSeverity, FitStatus};
use mixedmodels::formula::parse_formula;
use mixedmodels::model::{DataFrame, Family, GeneralizedLinearMixedModel, LinkFunction};

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
