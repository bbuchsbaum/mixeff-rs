#![cfg(not(feature = "nlopt"))]

use mixedmodels::compiler::{
    DiagnosticCode, InferenceAvailability, ModelKind, ObjectiveApproximation,
};
use mixedmodels::formula::parse_formula;
use mixedmodels::model::{DataFrame, Family, GeneralizedLinearMixedModel, LinkFunction};

// toy: 4 groups × 5 obs of Gamma/Log GLMM data; tests the artifact
// metadata surface, not numerical fit accuracy.
fn gamma_log_contract_data() -> DataFrame {
    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut group = Vec::new();
    for g in 0..4 {
        for obs in 0..5 {
            let xv = obs as f64 - 2.0;
            let eta = 0.5 + 0.2 * xv + (g as f64 - 1.5) * 0.08;
            y.push(eta.exp() * (0.95 + 0.02 * ((g + obs) % 3) as f64));
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
fn native_glmm_artifact_records_support_contract_metadata() {
    let data = gamma_log_contract_data();
    let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();
    let mut model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Gamma, Some(LinkFunction::Log))
            .unwrap();

    model.fit_with_options(true, 1, false).unwrap();

    let artifact = model.compiler_artifact();
    assert_eq!(
        artifact.model_boundary.model_kind,
        ModelKind::GeneralizedLinearMixedModel
    );
    assert_eq!(artifact.model_boundary.response_distribution, "gamma");
    assert_eq!(artifact.model_boundary.link, "log");
    assert!(matches!(
        artifact.model_boundary.objective_approximation,
        ObjectiveApproximation::Laplace { .. }
    ));
    assert!(matches!(
        artifact.model_boundary.inference_availability,
        InferenceAvailability::Unsupported { .. }
    ));

    assert_eq!(model.lmm.optsum.n_agq, 1);
    assert_eq!(model.lmm.optsum.backend.label(), "native");
    assert_eq!(model.lmm.optsum.optimizer_name(), "cobyla");

    let certificate = artifact
        .optimizer_certificate
        .as_ref()
        .expect("fitted GLMM should carry optimizer certificate metadata");
    assert_eq!(certificate.optimizer_name.as_deref(), Some("cobyla"));
    assert!(certificate.objective_value.is_some_and(f64::is_finite));
    assert_eq!(certificate.evidence.parameter_space.n_theta, 1);
    assert_eq!(certificate.evidence.sample_size.n_observations, Some(20));
    assert!(certificate
        .diagnostics
        .iter()
        .all(|diagnostic| diagnostic.code != DiagnosticCode::InvalidAgqRequest));
}
