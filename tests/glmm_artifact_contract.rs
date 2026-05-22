#![cfg(not(feature = "nlopt"))]
#![cfg(feature = "unstable-internals")]

use mixeff_rs::compiler::{
    CompiledModelArtifact, DiagnosticCode, FixedEffectCovarianceMethod,
    FixedEffectCovarianceStatus, InferenceAvailability, ModelKind, ObjectiveApproximation,
    ReliabilityGrade,
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

    assert_eq!(model.lmm().optsum().n_agq, 1);
    assert_eq!(model.lmm().optsum().backend.label(), "native");
    assert_eq!(model.lmm().optsum().optimizer_name(), "cobyla");

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

    let parity_scope = artifact
        .diagnostics
        .iter()
        .find(|diagnostic| {
            diagnostic.code == DiagnosticCode::SupportNote
                && diagnostic
                    .payload
                    .get("glmm_parity_scope")
                    .and_then(serde_json::Value::as_str)
                    == Some("fast_pirls_not_lme4_joint_parity")
        })
        .expect("fast-PIRLS GLMM artifact should state its lme4 parity scope");
    assert_eq!(
        parity_scope.severity,
        mixeff_rs::compiler::DiagnosticSeverity::Info
    );
    assert_eq!(
        parity_scope.payload["scorecard_class"],
        serde_json::json!("documented_divergence")
    );
    assert_eq!(
        parity_scope.payload["external_engine_parity"],
        serde_json::json!("not_certified")
    );
    assert_eq!(
        parity_scope.payload["reference_gate"],
        serde_json::json!("lme4_joint_laplace")
    );
    assert_eq!(
        parity_scope.payload["response_constants"],
        serde_json::json!("dropped")
    );

    let covariance = artifact
        .fixed_effect_covariance_matrix
        .as_ref()
        .expect("fitted GLMM artifact should carry fixed-effect covariance geometry");
    assert_eq!(covariance.status, FixedEffectCovarianceStatus::Available);
    assert_eq!(
        covariance.method,
        FixedEffectCovarianceMethod::PirlsLaplaceWorkingHessian
    );
    assert_eq!(covariance.reliability, ReliabilityGrade::Moderate);
    assert_eq!(
        covariance.coef_names,
        vec!["(Intercept)".to_string(), "x".to_string()]
    );
    assert_eq!(covariance.details.matrix_rows, 2);
    assert_eq!(covariance.details.matrix_cols, 2);
    assert_eq!(covariance.details.finite, Some(true));
    assert_eq!(covariance.details.symmetric, Some(true));
    let matrix = covariance
        .matrix
        .as_ref()
        .expect("available covariance payload should include matrix values");
    assert_eq!(matrix.len(), 2);
    assert!(matrix.iter().all(|row| row.len() == 2));
    assert!(matrix.iter().flatten().all(|value| value.is_finite()));
    assert!(covariance
        .notes
        .iter()
        .any(|note| note.contains("PIRLS/Laplace working-Hessian")));

    let value = serde_json::to_value(artifact).unwrap();
    assert_eq!(
        value["fixed_effect_covariance_matrix"]["method"],
        "pirls_laplace_working_hessian"
    );
    assert!(value["fixed_effect_covariance_matrix"]["matrix"].is_array());
    let json = serde_json::to_string(artifact).unwrap();
    let decoded: CompiledModelArtifact = serde_json::from_str(&json).unwrap();
    assert_eq!(&decoded, artifact);
}
