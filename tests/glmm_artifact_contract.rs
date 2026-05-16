#![cfg(not(feature = "nlopt"))]

use mixedmodels::compiler::{
    DiagnosticCode, FixedEffectCovarianceStatus, InferenceAvailability, ModelKind,
    ObjectiveApproximation,
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
    let inference_reason = match &artifact.model_boundary.inference_availability {
        InferenceAvailability::Unsupported { reason } => reason,
        other => panic!("expected GLMM inference unsupported boundary, got {other:?}"),
    };
    assert!(inference_reason.contains("Satterthwaite/KR"));

    assert_eq!(model.lmm.optsum.n_agq, 1);
    assert_eq!(model.lmm.optsum.backend.label(), "native");
    assert_eq!(model.lmm.optsum.optimizer_name(), "cobyla");

    let covariance = artifact
        .fixed_effect_covariance_matrix
        .as_ref()
        .expect("fitted GLMM artifact must carry a covariance payload");
    assert_eq!(covariance.status, FixedEffectCovarianceStatus::Unavailable);
    assert!(
        covariance.matrix.is_none(),
        "unavailable GLMM covariance must be matrix=null, not partially numeric"
    );
    assert_eq!(covariance.details.basis, "user_order");
    assert_eq!(covariance.details.rank, model.lmm.feterm.rank);
    assert_eq!(covariance.details.symmetric, None);
    assert_eq!(covariance.details.positive_semidefinite, None);
    assert_eq!(covariance.details.condition_number, None);
    assert!(covariance
        .reason
        .as_deref()
        .unwrap_or("")
        .contains("GLMM fixed-effect covariance matrix is unavailable"));
    assert!(covariance
        .notes
        .iter()
        .any(|note| note.contains("do not reconstruct a dense covariance matrix")));

    let covariance_json = artifact
        .table_by_name("fixed_effect_covariance_matrix")
        .unwrap()
        .expect("fitted GLMM artifact must expose covariance table by name");
    assert_eq!(covariance_json["schema_version"], "1.0.0");
    assert_eq!(covariance_json["status"], "unavailable");
    assert!(
        covariance_json["matrix"].is_null(),
        "serialized unavailable GLMM covariance must use matrix=null"
    );

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
