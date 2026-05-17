#![cfg(not(feature = "nlopt"))]
#![cfg(feature = "unstable-internals")]

use mixeff_rs::compiler::{
    DiagnosticCode, InferenceAvailability, ModelKind, ObjectiveApproximation,
};
use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::{DataFrame, Family, GeneralizedLinearMixedModel, LinkFunction};
use mixeff_rs::stats::FitSummaryPayload;

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

    assert_eq!(model.lmm().optsum.n_agq, 1);
    assert_eq!(model.lmm().optsum.backend.label(), "native");
    assert_eq!(model.lmm().optsum.optimizer_name(), "cobyla");

    let metadata = artifact
        .glmm_fit_metadata
        .as_ref()
        .expect("fitted GLMM artifact should expose fit-mode metadata");
    assert_eq!(metadata.estimation_method, "fast_pirls_profiled");
    assert_eq!(metadata.objective_definition, "profiled_glmm_deviance");
    assert_eq!(metadata.response_constants, "dropped");
    assert_eq!(metadata.n_agq, 1);
    assert_eq!(metadata.optimizer_backend, "native");
    assert_eq!(metadata.optimizer, "cobyla");
    assert_eq!(metadata.fallback_status, None);

    let summary = FitSummaryPayload::from_generalized_model(&model);
    assert_eq!(
        summary.estimation_method.as_deref(),
        Some("fast_pirls_profiled")
    );
    assert_eq!(
        summary.objective_definition.as_deref(),
        Some("profiled_glmm_deviance")
    );
    assert_eq!(summary.response_constants.as_deref(), Some("dropped"));
    assert_eq!(summary.n_agq, Some(1));
    assert_eq!(summary.optimizer_backend, "native");
    assert_eq!(summary.fallback_status, None);

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
