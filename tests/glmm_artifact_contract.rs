#![cfg(not(feature = "nlopt"))]
#![cfg(feature = "unstable-internals")]

use mixeff_rs::compiler::{
    CompiledModelArtifact, DiagnosticCode, FixedEffectCovarianceMethod,
    FixedEffectCovarianceStatus, FixedEffectInferenceMethod, FixedEffectInferenceStatus,
    InferenceAvailability, ModelKind, ObjectiveApproximation, ReliabilityGrade,
};
use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::{
    DataFrame, Family, GeneralizedLinearMixedModel, LinkFunction, MixedModelFit,
};
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

fn poisson_correlated_slope_contract_data() -> DataFrame {
    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut g1 = Vec::new();
    let mut g2 = Vec::new();

    for group1 in 0..16 {
        for group2 in 0..15 {
            for obs in 0..6 {
                let xv = (obs as f64 - 2.5) / 1.8 + (group2 as f64 % 4.0 - 1.5) * 0.08;
                let u1 = 0.35 * ((group1 + 1) as f64 * 1.1).sin();
                let v1 = 0.42 * ((group1 + 2) as f64 * 0.9).cos();
                let u2 = 0.28 * ((group2 + 1) as f64 * 0.8).cos();
                let v2 = 0.27 * ((group2 + 3) as f64 * 1.0).sin();
                let eta = 1.35 + 0.36 * xv + u1 + v1 * xv + u2 + v2 * xv;
                let noise = 0.88 + 0.06 * ((group1 * 13 + group2 * 7 + obs * 5) % 5) as f64;

                y.push((eta.exp() * noise).round().max(0.0));
                x.push(xv);
                g1.push(format!("g{}", group1 + 1));
                g2.push(format!("h{}", group2 + 1));
            }
        }
    }

    let mut data = DataFrame::new();
    data.add_numeric("y", y).unwrap();
    data.add_numeric("x", x).unwrap();
    data.add_categorical("g1", g1).unwrap();
    data.add_categorical("g2", g2).unwrap();
    data
}

fn osf_willingness_study1b_wait_data() -> DataFrame {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/parity/osf_willingness_to_wait_study1b.csv");
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(&path)
        .unwrap_or_else(|error| panic!("read {path:?}: {error}"));

    let headers = rdr
        .headers()
        .unwrap_or_else(|error| panic!("read headers from {path:?}: {error}"))
        .clone();
    let index = |name: &str| -> usize {
        headers
            .iter()
            .position(|candidate| candidate == name)
            .unwrap_or_else(|| panic!("{path:?} is missing required column `{name}`"))
    };
    let id_idx = index("ID");
    let title_idx = index("Title");
    let wait_idx = index("wait_choice");
    let enjoyment_idx = index("Enjoyment");

    let mut wait_choice = Vec::new();
    let mut enjoyment = Vec::new();
    let mut id = Vec::new();
    let mut title = Vec::new();
    for record in rdr.records() {
        let record = record.unwrap_or_else(|error| panic!("read record from {path:?}: {error}"));
        wait_choice.push(
            record[wait_idx]
                .parse::<f64>()
                .unwrap_or_else(|error| panic!("parse wait_choice in {path:?}: {error}")),
        );
        enjoyment.push(
            record[enjoyment_idx]
                .parse::<f64>()
                .unwrap_or_else(|error| panic!("parse Enjoyment in {path:?}: {error}")),
        );
        id.push(record[id_idx].to_string());
        title.push(record[title_idx].to_string());
    }

    let mean_enjoyment = enjoyment.iter().sum::<f64>() / enjoyment.len() as f64;
    let enjoyment_centered = enjoyment
        .into_iter()
        .map(|value| value - mean_enjoyment)
        .collect::<Vec<_>>();

    let mut data = DataFrame::new();
    data.add_numeric("wait_choice", wait_choice).unwrap();
    data.add_numeric("Enjoyment_centered", enjoyment_centered)
        .unwrap();
    data.add_categorical("ID", id).unwrap();
    data.add_categorical("Title", title).unwrap();
    data
}

fn bernoulli_separation_contract_data() -> DataFrame {
    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut x2 = Vec::new();
    let mut group = Vec::new();

    for g in 0..5 {
        for obs in 0..6 {
            let row = g * 6 + obs;
            let separated = row == 29;
            y.push(if separated {
                0.0
            } else if (row * 7 + 3) % 11 < 6 {
                1.0
            } else {
                0.0
            });
            x.push(obs as f64 - 2.5);
            x2.push(if separated { 1.0 } else { 0.0 });
            group.push(format!("g{}", g + 1));
        }
    }

    let mut data = DataFrame::new();
    data.add_numeric("y", y).unwrap();
    data.add_numeric("x", x).unwrap();
    data.add_numeric("x2", x2).unwrap();
    data.add_categorical("group", group).unwrap();
    data
}

fn poisson_boundary_random_intercept_contract_data() -> DataFrame {
    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut group = Vec::new();

    for g in 0..5 {
        for obs in 0..6 {
            let xv = obs as f64 - 2.5;
            let eta = 0.8 + 0.15 * xv;
            y.push(eta.exp().round().max(0.0));
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
    assert_eq!(summary.coefficients.method, "not-computed");
    assert!(summary
        .coefficients
        .std_errors
        .iter()
        .all(|value| value.is_nan()));
    assert!(summary
        .coefficients
        .z_values
        .iter()
        .all(|value| value.is_nan()));
    assert!(summary
        .coefficients
        .p_values
        .iter()
        .all(|value| value.is_nan()));
    assert!(summary
        .coefficients
        .p_value_reasons
        .iter()
        .all(|reason| reason
            .as_deref()
            .unwrap_or("")
            .contains("certified GLMM fixed-effect Wald inference is not implemented")));
    for row in summary
        .summary
        .rows
        .iter()
        .filter(|row| row.label == "(Intercept)" || row.label == "x")
    {
        assert!(row.estimate.is_some());
        assert_eq!(row.std_error, None);
        assert_eq!(row.z_stat, None);
        assert_eq!(row.pvalue, None);
    }
    assert!(model.stderror().iter().all(|value| value.is_nan()));
    for row in model.wald_confint(0.95) {
        assert!(row.estimate.is_finite());
        assert!(row.lower.is_nan());
        assert!(row.upper.is_nan());
    }

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

    let inference = artifact
        .fixed_effect_inference_table
        .as_ref()
        .expect("fitted GLMM artifact should carry explicit inference refusal rows");
    assert_eq!(inference.rows.len(), 2);
    for row in &inference.rows {
        assert_eq!(row.method, FixedEffectInferenceMethod::NotComputed);
        assert_eq!(row.status, FixedEffectInferenceStatus::Unsupported);
        assert_eq!(row.reliability, ReliabilityGrade::NotAvailable);
        assert!(row.std_error.is_none());
        assert!(row.statistic.is_none());
        assert!(row.p_value.is_none());
        assert!(row
            .reason
            .as_deref()
            .unwrap_or("")
            .contains("certified GLMM fixed-effect Wald inference is not implemented"));
        assert!(row
            .reason
            .as_deref()
            .unwrap_or("")
            .contains("certified active-subspace Hessian"));
    }

    let value = serde_json::to_value(artifact).unwrap();
    assert_eq!(
        value["fixed_effect_covariance_matrix"]["method"],
        "pirls_laplace_working_hessian"
    );
    assert_eq!(
        value["fixed_effect_inference_table"]["rows"][0]["method"],
        "not_computed"
    );
    assert!(value["fixed_effect_covariance_matrix"]["matrix"].is_array());
    let json = serde_json::to_string(artifact).unwrap();
    let decoded: CompiledModelArtifact = serde_json::from_str(&json).unwrap();
    assert_eq!(&decoded, artifact);
}

#[test]
fn joint_laplace_glmm_artifact_reports_certified_wald_rows_when_hessian_passes() {
    let data = gamma_log_contract_data();
    let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();
    let mut model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Gamma, Some(LinkFunction::Log))
            .unwrap();

    model.fit_with_options(false, 1, false).unwrap();

    let artifact = model.compiler_artifact();
    let metadata = artifact
        .glmm_fit_metadata
        .as_ref()
        .expect("joint-laplace GLMM artifact should expose fit metadata");
    assert_eq!(metadata.estimation_method, "joint_laplace");
    assert_eq!(metadata.objective_definition, "joint_glmm_laplace_deviance");
    assert_eq!(metadata.response_constants, "included");
    assert!(matches!(
        artifact.model_boundary.inference_availability,
        InferenceAvailability::Available { ref method }
            if method == "asymptotic_wald_z_joint_laplace_active_hessian"
    ));

    let covariance = artifact
        .fixed_effect_covariance_matrix
        .as_ref()
        .expect("joint-laplace GLMM artifact should carry certified covariance");
    assert_eq!(
        covariance.method,
        FixedEffectCovarianceMethod::JointLaplaceActiveHessian
    );
    assert_eq!(covariance.status, FixedEffectCovarianceStatus::Available);
    assert_eq!(covariance.reliability, ReliabilityGrade::Moderate);
    assert_eq!(covariance.details.matrix_rows, 2);
    assert_eq!(covariance.details.matrix_cols, 2);
    assert_eq!(covariance.details.finite, Some(true));
    assert_eq!(covariance.details.symmetric, Some(true));
    let matrix = covariance
        .matrix
        .as_ref()
        .expect("joint-laplace covariance should carry matrix values");
    assert_eq!(matrix.len(), 2);
    assert!(matrix.iter().all(|row| row.len() == 2));
    assert!(matrix[0][0] > 0.0);
    assert!(matrix[1][1] > 0.0);
    assert!((matrix[0][1] - matrix[1][0]).abs() < 1.0e-8);
    assert!(covariance
        .notes
        .iter()
        .any(|note| note.contains("inverse finite-difference Hessian")));

    let inference = artifact
        .fixed_effect_inference_table
        .as_ref()
        .expect("joint-laplace GLMM artifact should carry Wald inference rows");
    assert_eq!(inference.rows.len(), 2);
    // lme4 2.0.1 reference:
    // glmer(y ~ 1 + x + (1 | group), Gamma(log), nAGQ = 1,
    //       control = glmerControl(optimizer = "bobyqa"))
    let lme4_reference = [
        (
            "(Intercept)",
            0.4680676199782735,
            0.06290563672667525,
            7.440789797773219,
        ),
        (
            "x",
            0.2005181662023690,
            0.002706648896103688,
            74.08355272492919,
        ),
    ];
    for row in &inference.rows {
        assert_eq!(row.method, FixedEffectInferenceMethod::AsymptoticWaldZ);
        assert_eq!(row.status, FixedEffectInferenceStatus::Available);
        assert_eq!(row.reliability, ReliabilityGrade::Moderate);
        assert!(row.estimate.unwrap().is_finite());
        assert!(row.std_error.unwrap() > 0.0);
        assert!(row.statistic.unwrap().is_finite());
        let p_value = row.p_value.unwrap();
        assert!((0.0..=1.0).contains(&p_value));
        assert_eq!(row.reason, None);
        assert!(row
            .notes
            .iter()
            .any(|note| note.contains("inverse finite-difference Hessian")));
        assert!(row
            .notes
            .iter()
            .any(|note| note.contains("joint Hessian certificate")));

        let (_, expected_estimate, expected_se, expected_z) = lme4_reference
            .iter()
            .find(|(label, _, _, _)| *label == row.label)
            .copied()
            .expect("reference row should exist");
        assert!(
            (row.estimate.unwrap() - expected_estimate).abs() <= 2.0e-3,
            "{} estimate diverged from lme4 reference: observed {}, expected {}",
            row.label,
            row.estimate.unwrap(),
            expected_estimate
        );
        assert!(
            (row.std_error.unwrap() - expected_se).abs() <= 5.0e-5,
            "{} SE diverged from lme4 reference: observed {}, expected {}",
            row.label,
            row.std_error.unwrap(),
            expected_se
        );
        assert!(
            (row.statistic.unwrap() - expected_z).abs() <= 2.0e-2,
            "{} Wald statistic diverged from lme4 reference: observed {}, expected {}",
            row.label,
            row.statistic.unwrap(),
            expected_z
        );
    }
    let se = model.stderror();
    assert_eq!(se.len(), lme4_reference.len());
    let wald_ci = model.wald_confint(0.95);
    assert_eq!(wald_ci.len(), lme4_reference.len());
    let z95 = 1.959963984540054;
    for (idx, (label, _expected_estimate, expected_se, _expected_z)) in
        lme4_reference.iter().enumerate()
    {
        assert!(
            (se[idx] - expected_se).abs() <= 5.0e-5,
            "{label} stderror() diverged from lme4 reference: observed {}, expected {}",
            se[idx],
            expected_se
        );
        let estimate = model.coef()[idx];
        assert_eq!(wald_ci[idx].parameter, *label);
        assert!((wald_ci[idx].estimate - estimate).abs() <= 1.0e-12);
        assert!(
            (wald_ci[idx].lower - (estimate - z95 * se[idx])).abs() <= 1.0e-10,
            "{label} lower Wald CI should use certified SE"
        );
        assert!(
            (wald_ci[idx].upper - (estimate + z95 * se[idx])).abs() <= 1.0e-10,
            "{label} upper Wald CI should use certified SE"
        );
    }

    let summary = FitSummaryPayload::from_generalized_model(&model);
    assert_eq!(summary.coefficients.method, "wald-z");
    assert!(summary
        .coefficients
        .std_errors
        .iter()
        .all(|value| value.is_finite() && *value > 0.0));
    assert!(summary
        .coefficients
        .z_values
        .iter()
        .all(|value| value.is_finite()));
    assert!(summary
        .coefficients
        .p_values
        .iter()
        .all(|value| value.is_finite() && (0.0..=1.0).contains(value)));
    assert!(summary
        .coefficients
        .p_value_reasons
        .iter()
        .all(Option::is_none));
    for row in summary
        .summary
        .rows
        .iter()
        .filter(|row| row.label == "(Intercept)" || row.label == "x")
    {
        assert!(row.std_error.unwrap() > 0.0);
        assert!(row.z_stat.unwrap().is_finite());
        assert!((0.0..=1.0).contains(&row.pvalue.unwrap()));
    }
}

#[test]
fn joint_laplace_glmm_wald_rows_match_glmer_on_correlated_random_slopes() {
    let data = poisson_correlated_slope_contract_data();
    let formula = parse_formula("y ~ 1 + x + (1 + x | g1) + (1 + x | g2)").unwrap();
    let mut model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Poisson, Some(LinkFunction::Log))
            .unwrap();

    model.fit_with_options(false, 1, false).unwrap();

    let artifact = model.compiler_artifact();
    let metadata = artifact
        .glmm_fit_metadata
        .as_ref()
        .expect("joint-laplace GLMM artifact should expose fit metadata");
    assert_eq!(metadata.estimation_method, "joint_laplace");
    assert_eq!(metadata.objective_definition, "joint_glmm_laplace_deviance");
    assert_eq!(metadata.response_constants, "included");
    assert!(matches!(
        artifact.model_boundary.inference_availability,
        InferenceAvailability::Available { ref method }
            if method == "asymptotic_wald_z_joint_laplace_active_hessian"
    ));

    let covariance = artifact
        .fixed_effect_covariance_matrix
        .as_ref()
        .expect("joint-laplace correlated-slope GLMM should carry certified covariance");
    assert_eq!(
        covariance.method,
        FixedEffectCovarianceMethod::JointLaplaceActiveHessian
    );
    assert_eq!(covariance.status, FixedEffectCovarianceStatus::Available);
    assert_eq!(covariance.reliability, ReliabilityGrade::Moderate);
    assert_eq!(covariance.details.matrix_rows, 2);
    assert_eq!(covariance.details.matrix_cols, 2);

    let inference = artifact
        .fixed_effect_inference_table
        .as_ref()
        .expect("joint-laplace correlated-slope GLMM should carry Wald rows");
    assert_eq!(inference.rows.len(), 2);
    assert_eq!(
        model.theta().len(),
        6,
        "two correlated random-slope terms should exercise six covariance parameters"
    );

    // lme4 2.0.1 reference:
    // glmer(y ~ 1 + x + (1 + x | g1) + (1 + x | g2), poisson(log),
    //       nAGQ = 1, control = glmerControl(optimizer = "bobyqa"))
    let lme4_reference = [
        (
            "(Intercept)",
            1.3403253125885581,
            0.08232432213402913,
            16.281036731847301,
        ),
        (
            "x",
            0.3110704756038575,
            0.08791135327173999,
            3.538456229223516,
        ),
    ];
    for row in &inference.rows {
        assert_eq!(row.method, FixedEffectInferenceMethod::AsymptoticWaldZ);
        assert_eq!(row.status, FixedEffectInferenceStatus::Available);
        assert_eq!(row.reliability, ReliabilityGrade::Moderate);

        let (_, expected_estimate, expected_se, expected_z) = lme4_reference
            .iter()
            .find(|(label, _, _, _)| *label == row.label)
            .copied()
            .expect("reference row should exist");
        let estimate = row.estimate.unwrap();
        let std_error = row.std_error.unwrap();
        let statistic = row.statistic.unwrap();
        assert!(
            (estimate - expected_estimate).abs() <= 2.0e-3,
            "{} estimate diverged from lme4 reference: observed {}, expected {}",
            row.label,
            estimate,
            expected_estimate
        );
        assert!(
            (std_error - expected_se).abs() <= 2.0e-3,
            "{} SE diverged from lme4 reference: observed {}, expected {}",
            row.label,
            std_error,
            expected_se
        );
        assert!(
            (statistic - expected_z).abs() <= 5.0e-2,
            "{} Wald statistic diverged from lme4 reference: observed {}, expected {}",
            row.label,
            statistic,
            expected_z
        );
    }
    let se = model.stderror();
    assert_eq!(se.len(), lme4_reference.len());
    for (idx, (_label, _expected_estimate, expected_se, _expected_z)) in
        lme4_reference.iter().enumerate()
    {
        assert!(
            (se[idx] - expected_se).abs() <= 2.0e-3,
            "stderror() should use certified correlated-slope GLMM SE rows"
        );
    }
    assert!(model
        .wald_confint(0.95)
        .iter()
        .all(|row| row.lower.is_finite() && row.upper.is_finite() && row.lower < row.upper));
}

#[test]
fn joint_laplace_glmm_wald_rows_match_glmer_on_osf_study1b_correlated_slopes() {
    if std::env::var_os("MIXEFF_RUN_OSF_WALD_PARITY").is_none() {
        return;
    }

    let data = osf_willingness_study1b_wait_data();
    let formula = parse_formula(
        "wait_choice ~ 1 + Enjoyment_centered + \
         (1 + Enjoyment_centered | ID) + (1 + Enjoyment_centered | Title)",
    )
    .unwrap();
    let mut model = GeneralizedLinearMixedModel::new(
        formula,
        &data,
        Family::Bernoulli,
        Some(LinkFunction::Logit),
    )
    .unwrap();

    model.fit_with_options(false, 1, false).unwrap();

    let artifact = model.compiler_artifact();
    let metadata = artifact
        .glmm_fit_metadata
        .as_ref()
        .expect("joint-laplace OSF GLMM artifact should expose fit metadata");
    assert_eq!(metadata.estimation_method, "joint_laplace");
    assert_eq!(metadata.objective_definition, "joint_glmm_laplace_deviance");
    assert_eq!(metadata.response_constants, "included");
    let availability = &artifact.model_boundary.inference_availability;
    let first_reason = artifact
        .fixed_effect_inference_table
        .as_ref()
        .and_then(|table| table.rows.first())
        .and_then(|row| row.reason.as_deref());
    assert!(
        matches!(
            availability,
            InferenceAvailability::Available { ref method }
            if method == "asymptotic_wald_z_joint_laplace_active_hessian"
        ),
        "OSF study1b correlated-slope GLMM Wald rows were not certified: availability={availability:?}, first_reason={first_reason:?}"
    );
    assert_eq!(
        model.theta().len(),
        6,
        "two correlated random-slope terms should exercise six covariance parameters"
    );

    // lme4 2.0.1 reference:
    // glmer(wait_choice ~ 1 + Enjoyment_centered
    //       + (1 + Enjoyment_centered | ID)
    //       + (1 + Enjoyment_centered | Title),
    //       study1b, binomial(logit), nAGQ = 1,
    //       control = glmerControl(optimizer = "bobyqa"))
    let lme4_reference = [
        (
            "(Intercept)",
            -1.6946010298834349,
            0.37647010448086149,
            -4.5012897696623950,
        ),
        (
            "Enjoyment_centered",
            1.0296271533640451,
            0.10953315541246519,
            9.4001414410715540,
        ),
    ];
    let inference = artifact
        .fixed_effect_inference_table
        .as_ref()
        .expect("joint-laplace OSF GLMM should carry Wald rows");
    assert_eq!(inference.rows.len(), lme4_reference.len());
    for row in &inference.rows {
        assert_eq!(row.method, FixedEffectInferenceMethod::AsymptoticWaldZ);
        assert_eq!(row.status, FixedEffectInferenceStatus::Available);
        assert_eq!(row.reliability, ReliabilityGrade::Moderate);

        let (_, expected_estimate, expected_se, expected_z) = lme4_reference
            .iter()
            .find(|(label, _, _, _)| *label == row.label)
            .copied()
            .expect("reference row should exist");
        let estimate = row.estimate.unwrap();
        let std_error = row.std_error.unwrap();
        let statistic = row.statistic.unwrap();
        assert!(
            (estimate - expected_estimate).abs() <= 5.0e-3,
            "{} estimate diverged from lme4 reference: observed {}, expected {}",
            row.label,
            estimate,
            expected_estimate
        );
        assert!(
            (std_error - expected_se).abs() <= 5.0e-3,
            "{} SE diverged from lme4 reference: observed {}, expected {}",
            row.label,
            std_error,
            expected_se
        );
        assert!(
            (statistic - expected_z).abs() <= 5.0e-2,
            "{} Wald statistic diverged from lme4 reference: observed {}, expected {}",
            row.label,
            statistic,
            expected_z
        );
    }
}

#[test]
fn binomial_separation_keeps_glmm_wald_rows_unavailable_with_reason() {
    let data = bernoulli_separation_contract_data();
    let formula = parse_formula("y ~ 1 + x + x2 + (1 | group)").unwrap();
    let mut model = GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None)
        .expect("separation fixture should construct");

    model
        .fit_with_options(false, 1, false)
        .expect("separation fixture should fit without fabricating inference");

    let artifact = model.compiler_artifact();
    assert!(artifact
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == DiagnosticCode::BinomialSeparation));
    assert!(matches!(
        artifact.model_boundary.inference_availability,
        InferenceAvailability::NotAssessed { ref reason }
            if reason.contains("binomial separation diagnostics")
    ));

    let inference = artifact
        .fixed_effect_inference_table
        .as_ref()
        .expect("fitted separated GLMM should still carry explicit inference rows");
    assert_eq!(inference.rows.len(), 3);
    for row in &inference.rows {
        assert_eq!(row.method, FixedEffectInferenceMethod::NotComputed);
        assert_eq!(row.status, FixedEffectInferenceStatus::NotAssessed);
        assert_eq!(row.reliability, ReliabilityGrade::NotAvailable);
        assert!(row.std_error.is_none());
        assert!(row.statistic.is_none());
        assert!(row.p_value.is_none());
        assert!(row
            .reason
            .as_deref()
            .unwrap_or("")
            .contains("binomial separation diagnostics"));
        assert!(row
            .notes
            .iter()
            .any(|note| note.contains("separation-robust inference backend")));
    }

    let summary = FitSummaryPayload::from_generalized_model(&model);
    assert_eq!(summary.coefficients.method, "not-computed");
    assert!(summary
        .coefficients
        .std_errors
        .iter()
        .all(|value| value.is_nan()));
    assert!(summary
        .coefficients
        .z_values
        .iter()
        .all(|value| value.is_nan()));
    assert!(summary
        .coefficients
        .p_values
        .iter()
        .all(|value| value.is_nan()));
    assert!(summary
        .coefficients
        .p_value_reasons
        .iter()
        .all(|reason| reason
            .as_deref()
            .unwrap_or("")
            .contains("binomial separation diagnostics")));
    assert!(model.stderror().iter().all(|value| value.is_nan()));
    assert!(model
        .wald_confint(0.95)
        .iter()
        .all(|row| row.lower.is_nan() && row.upper.is_nan()));
}

#[test]
fn joint_laplace_glmm_boundary_theta_still_certifies_fixed_effect_rows() {
    let data = poisson_boundary_random_intercept_contract_data();
    let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();
    let mut model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Poisson, Some(LinkFunction::Log))
            .expect("boundary fixture should construct");

    model
        .fit_with_options(false, 1, false)
        .expect("boundary fixture should fit without fabricating inference");

    let theta = model.theta();
    assert!(
        theta.iter().any(|value| value.abs() <= 1.0e-4),
        "boundary fixture should pin at least one covariance scale near zero, got {theta:?}"
    );

    let artifact = model.compiler_artifact();
    assert!(matches!(
        artifact.model_boundary.inference_availability,
        InferenceAvailability::Available { ref method }
            if method == "asymptotic_wald_z_joint_laplace_active_hessian"
    ));

    let covariance = artifact
        .fixed_effect_covariance_matrix
        .as_ref()
        .expect("boundary joint-laplace GLMM should carry certified fixed covariance");
    assert_eq!(
        covariance.method,
        FixedEffectCovarianceMethod::JointLaplaceActiveHessian
    );
    assert_eq!(covariance.status, FixedEffectCovarianceStatus::Available);
    assert!(covariance
        .notes
        .iter()
        .any(|note| note.contains("omitted from the active Hessian") && note.contains("theta 1")));

    let inference = artifact
        .fixed_effect_inference_table
        .as_ref()
        .expect("boundary joint-laplace GLMM should carry fixed-effect Wald rows");
    assert_eq!(inference.rows.len(), 2);
    for row in &inference.rows {
        assert_eq!(row.method, FixedEffectInferenceMethod::AsymptoticWaldZ);
        assert_eq!(row.status, FixedEffectInferenceStatus::Available);
        assert_eq!(row.reliability, ReliabilityGrade::Moderate);
        assert!(row
            .std_error
            .is_some_and(|value| value.is_finite() && value > 0.0));
        assert!(row.statistic.is_some_and(f64::is_finite));
        assert!(row.reason.is_none());
        assert!(row.notes.iter().any(
            |note| note.contains("omitted from the active Hessian") && note.contains("theta 1")
        ));
    }

    assert!(model
        .stderror()
        .iter()
        .all(|value| value.is_finite() && *value > 0.0));
    let summary = FitSummaryPayload::from_generalized_model(&model);
    assert_eq!(summary.coefficients.method, "wald-z");
    assert!(summary
        .coefficients
        .std_errors
        .iter()
        .all(|value| value.is_finite() && *value > 0.0));
}
