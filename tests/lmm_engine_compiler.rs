#![cfg(feature = "unstable-internals")]

// Engine-level compiler tests migrated from src/model/linear/tests.rs
// (ranked-audit M3). Public-API only; internals-bound tests stay inline.

mod common;
#[allow(unused_imports)]
use common::*;

use approx::assert_relative_eq;
use mixeff_rs::compiler::{
    CertificateCheck, CompilerPolicy, ConvergenceVerificationStatus, DiagnosticCode,
    EffectiveRankStatus, EstimabilityAssessment, EstimabilityStatus, EvidenceMethod,
    EvidenceQuality, FitIntent, FitStatus, FixedEffectCovarianceMethod,
    FixedEffectCovarianceStatus, FixedEffectHypothesis, FixedEffectInferenceMethod,
    FixedEffectInferenceRowKind, FixedEffectInferenceStatus, FixedEffectReliabilityReason,
    FixedEffectStatisticName, FixedEffectTermTestType, FixedEffectTestMethod, InferenceMethod,
    InferenceStatus, RankStatus, ReliabilityGrade,
};
#[allow(unused_imports)]
use mixeff_rs::error::*;
use mixeff_rs::formula::parse_formula;
#[allow(unused_imports)]
use mixeff_rs::model::data::{Column, DataFrame};
#[allow(unused_imports)]
use mixeff_rs::model::fixed_design::*;
#[allow(unused_imports)]
use mixeff_rs::model::linear::*;
#[allow(unused_imports)]
use mixeff_rs::model::traits::MixedModelFit;
#[allow(unused_imports)]
use mixeff_rs::stats::*;
#[allow(unused_imports)]
use mixeff_rs::types::*;
use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::SeedableRng;

#[test]
fn test_lmm_carries_compiler_artifact_design_audit() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let model = LinearMixedModel::new(formula.clone(), &data, None).unwrap();

    let artifact = model.compiler_artifact();
    assert_eq!(artifact.requested_formula, formula.to_string());
    assert_eq!(artifact.semantic_model.random_terms.len(), 1);
    assert_eq!(artifact.theta_maps.len(), 1);

    let audit = model.design_audit().expect("design audit should attach");
    assert_eq!(audit.fixed_effect_rank.status, RankStatus::FullRank);
    assert_eq!(audit.fixed_effect_rank.rank, Some(2));
    assert_eq!(audit.random_terms[0].group.name, "subj");
    assert_eq!(audit.random_terms[0].group.n_levels, Some(18));
    assert_eq!(audit.random_terms[0].requested_covariance_parameters, 3);
}

// Rank-detection depends on the optimizer landing in the reduced-rank
// region of the theta surface; the native no-default-features path can
// converge full-rank on this fit, so the assertion only holds with NLopt.
#[cfg(feature = "nlopt")]
#[test]
fn test_singular_fixture_zcp_fit_exposes_reduced_effective_rank() {
    let (data, _) = mixeff_rs::datasets::load("singular").unwrap();
    let formula = parse_formula("y ~ 1 + A * B * C + (A * B * C || group)").unwrap();
    let mut model = LinearMixedModel::new(formula.clone(), &data, None).unwrap();

    model.fit(false).unwrap();

    let summary = &model.compiler_artifact().effective_covariance[0];
    assert_eq!(summary.requested_rank, 8);
    assert!(summary.supported_rank < summary.requested_rank);
    assert_eq!(summary.status, EffectiveRankStatus::ReducedRank);
    assert_eq!(
        model.optimizer_certificate().unwrap().status,
        FitStatus::ConvergedReducedRank
    );
}

#[test]
fn test_lmm_compiler_artifact_records_rank_deficient_fixed_effects() {
    let mut data = DataFrame::new();
    data.add_numeric("y", vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    data.add_numeric("x", vec![0.0, 1.0, 0.0, 1.0]).unwrap();
    data.add_numeric("x2", vec![0.0, 2.0, 0.0, 2.0]).unwrap();
    data.add_categorical(
        "z",
        vec!["a", "a", "b", "b"]
            .into_iter()
            .map(str::to_string)
            .collect(),
    )
    .unwrap();

    let formula = parse_formula("y ~ x + x2 + (1 | z)").unwrap();
    let model = LinearMixedModel::new(formula, &data, None).unwrap();
    let audit = model.design_audit().expect("design audit should attach");

    assert_eq!(audit.fixed_effect_rank.status, RankStatus::RankDeficient);
    assert_eq!(audit.fixed_effect_rank.rank, Some(2));
    assert_eq!(audit.fixed_effect_rank.expected, Some(3));
    assert!(model
        .compiler_artifact()
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == DiagnosticCode::FixedEffectRankDeficient));
}

#[test]
fn test_lmm_optimizer_certificate_records_interior_fit() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula.clone(), &data, None).unwrap();

    assert!(model.optimizer_certificate().is_none());
    model.fit(false).unwrap();

    let certificate = model
        .optimizer_certificate()
        .expect("optimizer certificate should attach after fit");
    assert_eq!(certificate.status, FitStatus::ConvergedInterior);
    assert_eq!(
        certificate.optimizer_name.as_deref(),
        Some("pattern_search")
    );
    assert!(certificate.objective_value.is_some());
    assert!(certificate.evidence.optimizer_stop.acceptable_stop);
    assert!(!certificate.evidence.optimizer_stop.budget_exhausted);
    assert_eq!(certificate.evidence.parameter_space.n_theta, 1);
    assert_eq!(certificate.evidence.parameter_space.n_boundary, 0);
    assert_eq!(certificate.evidence.sample_size.n_observations, Some(180));
    assert_eq!(certificate.evidence.sample_size.n_theta, 1);
    assert!(matches!(
        certificate.evidence.certification_quality,
        EvidenceQuality::Approximate { .. }
    ));
    assert!(matches!(
        certificate.evidence.gradient.method,
        EvidenceMethod::FiniteDifference
    ));
    assert!(certificate.evidence.gradient.raw_gradient_norm.is_some());
    assert!(certificate.evidence.gradient.free_gradient_norm.is_some());
    assert!(certificate
        .evidence
        .gradient
        .projected_gradient_norm
        .is_some());
    assert!(matches!(
        certificate.evidence.hessian.method,
        EvidenceMethod::FiniteDifference
    ));
    assert!(certificate.evidence.hessian.min_eigenvalue.is_some());
    assert_eq!(certificate.evidence.hessian.rank, Some(1));
    assert!(certificate
        .checks
        .iter()
        .any(|check| matches!(check, CertificateCheck::FreeGradientOk { .. })));
    assert!(certificate
        .checks
        .iter()
        .any(|check| matches!(check, CertificateCheck::HessianPsdOnActiveSubspace { .. })));
    assert!(!certificate
        .checks
        .iter()
        .any(|check| matches!(check, CertificateCheck::NotAssessed { .. })));

    let verification = model.verify_convergence().unwrap();
    assert!(matches!(
        verification.status,
        ConvergenceVerificationStatus::RestartAgrees
            | ConvergenceVerificationStatus::OptimizerConsensus
    ));
    assert!(!verification.runs.is_empty());
    assert!(verification.runs.iter().all(|run| run.agrees));
    assert!(model
        .optimizer_certificate()
        .unwrap()
        .verification
        .is_some());

    let trace = &model.compiler_artifact().covariance_parameter_traces[0];
    assert!(trace.theta.value.is_some());
    assert!(trace.lambda.value.is_some());
    assert_eq!(trace.varcorr_entries[0].label, "sd(intercept)");
    assert!(trace.varcorr_entries[0].value.is_some());
}

#[test]
fn test_lmm_convergence_verification_is_not_run_before_fit() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

    let verification = model.verify_convergence().unwrap();

    assert_eq!(verification.status, ConvergenceVerificationStatus::NotRun);
    assert!(verification.runs.is_empty());
    assert_eq!(verification.message, "model has not been fitted");
}

#[test]
fn test_lmm_optimizer_certificate_records_boundary_fit() {
    let data = singular_re_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

    model.fit(false).unwrap();

    let certificate = model
        .optimizer_certificate()
        .expect("optimizer certificate should attach after fit");
    assert_eq!(certificate.status, FitStatus::ConvergedReducedRank);
    assert_eq!(certificate.evidence.parameter_space.n_boundary, 1);
    assert_eq!(
        certificate.evidence.parameter_space.boundary_indices,
        vec![0]
    );
    assert!(certificate
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == DiagnosticCode::BoundaryParameter));
    assert!(certificate.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == DiagnosticCode::BoundaryParameter
            && diagnostic
                .suggested_actions
                .iter()
                .any(|action| action.contains("valid fitted boundary"))
    }));
    let boundary_diagnostic = certificate
        .diagnostics
        .iter()
        .find(|diagnostic| diagnostic.code == DiagnosticCode::BoundaryParameter)
        .expect("boundary parameter diagnostic");
    assert_eq!(boundary_diagnostic.affected_terms, vec!["(1 | batch)"]);
    assert!(boundary_diagnostic
        .message
        .contains("standard deviation for intercept in (1 | batch)"));
    assert!(!boundary_diagnostic.message.contains("theta[0]"));
    assert_eq!(
        boundary_diagnostic.payload.get("theta_index"),
        Some(&serde_json::json!(0))
    );
    assert_eq!(
        boundary_diagnostic.payload.get("term_id"),
        Some(&serde_json::json!("r0"))
    );
    assert!(matches!(
        &certificate.evidence.gradient.method,
        EvidenceMethod::NotAssessed { reason } if reason.contains("variance-component boundary")
    ));
    assert!(certificate
        .evidence
        .gradient
        .kkt_boundary_gradient_max
        .is_none());
    assert!(matches!(
        &certificate.evidence.hessian.quality,
        EvidenceQuality::NotAssessed { reason } if reason.contains("variance-component boundary")
    ));
    assert_eq!(certificate.evidence.hessian.rank, None);
    assert!(certificate.checks.iter().any(|check| matches!(
        check,
        CertificateCheck::NotAssessed { reason }
            if reason.contains("boundary-gradient KKT check skipped")
    )));
    assert!(certificate
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == DiagnosticCode::CovarianceReduced));
    let covariance_diagnostic = certificate
        .diagnostics
        .iter()
        .find(|diagnostic| diagnostic.code == DiagnosticCode::CovarianceReduced)
        .expect("covariance reduced diagnostic");
    assert_eq!(covariance_diagnostic.affected_terms, vec!["(1 | batch)"]);
    assert!(covariance_diagnostic
        .message
        .contains("fitted covariance for (1 | batch)"));
    assert!(!covariance_diagnostic.message.contains("r0"));
    assert_eq!(
        covariance_diagnostic.payload.get("term_id"),
        Some(&serde_json::json!("r0"))
    );
    assert!(model
        .compiler_artifact()
        .reductions
        .iter()
        .all(|reduction| reduction.diagnostics.is_empty()));
    assert!(!model
        .compiler_artifact()
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == DiagnosticCode::CovarianceReduced));
    assert_eq!(
        model.compiler_artifact().effective_covariance[0].supported_rank,
        0
    );
}

#[test]
fn test_two_by_two_covariance_kkt_certificate_valid_rank_one_rho_one() {
    let data = rank_one_rho_one_random_slope_fixture();
    let formula = parse_formula("y ~ 1 + x + (1 + x | group)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

    model.fit(false).unwrap();
    let fitted_theta = model.theta();
    let rank_one_theta = [fitted_theta[0], fitted_theta[1], 0.0];
    model.set_theta(&rank_one_theta).unwrap();

    let certificate = model.two_by_two_covariance_kkt_certificate().unwrap();
    assert_eq!(certificate.blocks.len(), 1);
    let block = &certificate.blocks[0];
    assert_eq!(
        block.classification,
        CovarianceKktClassification::ValidRankDeficientCovariance
    );
    assert!(block.min_eig_g <= certificate.covariance_tolerance);
    assert!(block.min_eig_score >= -certificate.score_tolerance);
    assert!(block.complementarity <= certificate.complementarity_tolerance);
    assert!(block.residual.is_finite());
}

#[test]
fn test_two_by_two_covariance_kkt_certificate_flags_invalid_boundary_stop() {
    let data = rank_one_rho_one_random_slope_fixture();
    let formula = parse_formula("y ~ 1 + x + (1 + x | group)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

    model.fit(false).unwrap();
    model.set_theta(&[0.0, 0.0, 0.0]).unwrap();

    let certificate = model.two_by_two_covariance_kkt_certificate().unwrap();
    assert_eq!(certificate.blocks.len(), 1);
    let block = &certificate.blocks[0];
    assert_eq!(
        block.classification,
        CovarianceKktClassification::InvalidBoundaryStop
    );
    assert!(block.min_eig_g <= certificate.covariance_tolerance);
    assert!(block.min_eig_score < -certificate.score_tolerance);
    assert!(block.residual.is_finite());
}

#[test]
fn test_effective_covariance_rank_uses_policy_thresholds() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    let mut policy = CompilerPolicy::maximal_feasible();
    policy.thresholds.effective_rank_relative_tolerance = 2.0;
    model.set_compiler_policy(policy).unwrap();

    model.fit(false).unwrap();

    let summary = &model.compiler_artifact().effective_covariance[0];
    assert_eq!(summary.status, EffectiveRankStatus::ReducedRank);
    assert_eq!(summary.supported_rank, 0);
    assert!(model
        .compiler_artifact()
        .reproducibility
        .thresholds
        .iter()
        .any(|(name, value)| name == "effective_rank_relative_tolerance" && value == "2"));
}

#[test]
fn test_lmm_new_with_compiler_policy_applies_policy_before_fit() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut policy = CompilerPolicy::as_specified();
    policy.thresholds.effective_rank_relative_tolerance = 0.25;

    let model = LinearMixedModel::new_with_compiler_policy(formula, &data, None, policy).unwrap();

    assert_eq!(
        model.compiler_policy().random_strategy,
        mixeff_rs::compiler::RandomStrategy::AsSpecified
    );
    assert!(model
        .compiler_artifact()
        .reproducibility
        .thresholds
        .iter()
        .any(|(name, value)| name == "effective_rank_relative_tolerance" && value == "0.25"));
}

#[test]
fn test_lmm_design_compiled_refuses_unsupported_random_distribution() {
    let data = grouped_slope_data(2);
    let formula = parse_formula("y ~ x + (1 + x | group)").unwrap();

    let result = LinearMixedModel::new_with_compiler_policy(
        formula,
        &data,
        None,
        CompilerPolicy::design_compiled(),
    );

    assert!(result.is_err());
    assert!(result
        .err()
        .unwrap()
        .to_string()
        .contains("design_compiled refused"));
}

#[test]
fn test_lmm_design_compiled_refuses_row_saturated_random_effect() {
    let data = grouped_slope_data(100);
    let formula = parse_formula("y ~ x + (1 + x | group)").unwrap();

    let err = LinearMixedModel::new_with_compiler_policy(
        formula,
        &data,
        None,
        CompilerPolicy::design_compiled(),
    )
    .expect_err("row-saturated random-effect terms should be refused");
    let message = err.to_string();

    assert!(message.contains("number of observations (200)"));
    assert!(message.contains("random coefficients (200)"));
    assert!(message.contains("residual scale"));
}

#[test]
fn test_lmm_set_compiler_policy_rejects_after_fit() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(true).unwrap();

    let error = model
        .set_compiler_policy(CompilerPolicy::as_specified())
        .expect_err("fitted models must reject policy mutation");

    assert!(matches!(error, MixedModelError::AlreadyFitted));
}

#[test]
fn test_lmm_fit_with_compiler_policy_applies_policy_then_fits() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    let mut policy = CompilerPolicy::as_specified();
    policy.thresholds.effective_rank_relative_tolerance = 0.5;

    model.fit_with_compiler_policy(false, policy).unwrap();

    assert_eq!(
        model.compiler_policy().random_strategy,
        mixeff_rs::compiler::RandomStrategy::AsSpecified
    );
    assert!(model.optimizer_certificate().is_some());
    assert!(model
        .compiler_artifact()
        .reproducibility
        .thresholds
        .iter()
        .any(|(name, value)| name == "effective_rank_relative_tolerance" && value == "0.5"));
}

#[test]
fn test_objective_at_reuses_work_blocks_without_drift() {
    let data = simulate_sleepstudy_like(8, 6, 7);
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

    let theta_a = [1.3, -0.15, 0.8];
    let theta_b = [0.7, 0.25, 1.4];

    let obj_a1 = model.objective_at(&theta_a).unwrap();
    let _obj_b = model.objective_at(&theta_b).unwrap();
    let obj_a2 = model.objective_at(&theta_a).unwrap();

    assert_relative_eq!(obj_a1, obj_a2, epsilon = 1e-10, max_relative = 1e-10);
}

#[test]
fn test_scalar_objective_matches_julia_on_shared_fixture() {
    let data = shared_julia_parity_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

    let julia_theta = [0.6273717260668661];
    let julia_objective = 223.742_068_488_410_9;

    let rust_objective = model.objective_at(&julia_theta).unwrap();

    assert_relative_eq!(
        rust_objective,
        julia_objective,
        epsilon = 1e-8,
        max_relative = 1e-8
    );

    model.fit(true).unwrap();
    assert_relative_eq!(
        model.objective_value(),
        julia_objective,
        epsilon = 1e-5,
        max_relative = 1e-5
    );
    assert_relative_eq!(
        model.sigma(),
        30.23875724370832,
        epsilon = 1e-5,
        max_relative = 1e-5
    );
}

#[cfg(feature = "nlopt")]
#[test]
fn test_vector_objective_matches_julia_on_shared_fixture() {
    let data = shared_julia_parity_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

    let julia_theta = [0.6565437822843008, -0.019160976185379253, 0.0];
    let julia_objective = 223.73509351902135;

    let rust_objective = model.objective_at(&julia_theta).unwrap();

    assert_relative_eq!(
        rust_objective,
        julia_objective,
        epsilon = 1e-8,
        max_relative = 1e-8
    );

    model.fit(true).unwrap();
    assert_relative_eq!(
        model.objective_value(),
        julia_objective,
        epsilon = 1e-4,
        max_relative = 1e-4
    );
    assert_relative_eq!(
        model.sigma(),
        30.22863368533761,
        epsilon = 1e-4,
        max_relative = 1e-4
    );
}

#[test]
fn test_coeftable_with_method_surfaces_satterthwaite_df() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(true).unwrap();

    // Default table is asymptotic Wald-z with no df.
    let wald = model.coeftable();
    assert_eq!(wald.method, "wald-z");
    assert_eq!(wald.statistic_name, "z");
    assert!(wald.df.iter().all(|d| d.is_none()));

    // Satterthwaite table self-identifies and carries finite df.
    let satt = model.coeftable_with_method(FixedEffectTestMethod::Satterthwaite);
    assert_eq!(satt.method, "satterthwaite");
    assert_eq!(satt.statistic_name, "t");
    assert_eq!(satt.names, wald.names);
    assert_eq!(satt.estimates.len(), wald.estimates.len());
    for (e_s, e_w) in satt.estimates.iter().zip(wald.estimates.iter()) {
        assert_relative_eq!(e_s, e_w, epsilon = 1e-9);
    }
    assert!(
        satt.df
            .iter()
            .any(|d| d.map(|v| v.is_finite()).unwrap_or(false)),
        "Satterthwaite table must carry finite denominator df"
    );
    // The rendered table announces the method (no longer misleading).
    let rendered = format!("{satt}");
    assert!(rendered.contains("Method: satterthwaite"));
    assert!(rendered.contains("t value"));
    assert!(mixeff_rs::stats::coeftable_to_markdown(&satt).contains("*Method: satterthwaite*"));
}

#[test]
fn test_vcov_beta_varpar_matches_fitted_vcov_and_restores_state() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(true).unwrap();

    let theta_before = model.theta();
    let objective_before = model.objective_value();
    let vcov_before = model.vcov();
    let varpar = fitted_varpar(&model);

    let vcov = model.vcov_beta_varpar(&varpar).unwrap();

    assert_matrix_relative_eq(&vcov, &vcov_before, 1e-10);
    assert_eq!(model.theta(), theta_before);
    assert_relative_eq!(model.objective_value(), objective_before, epsilon = 1e-10);
    assert_matrix_relative_eq(&model.vcov(), &vcov_before, 1e-10);
}

#[test]
fn test_jac_vcov_beta_varpar_rejects_boundary_stencil_without_changing_state() {
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let theta_before = model.theta();
    let objective_before = model.objective_value();
    let mut varpar = fitted_varpar(&model);
    varpar[0] = 0.0;

    let err = model.jac_vcov_beta_varpar(&varpar).unwrap_err();

    assert!(err.to_string().contains("lower bound"));
    assert_eq!(model.theta(), theta_before);
    assert_relative_eq!(model.objective_value(), objective_before, epsilon = 1e-10);
}

#[test]
fn test_kenward_roger_adjusted_vcov_rejects_unweighted_prerequisite_gap() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let weights = vec![1.0; data.nrow()];
    let mut model = LinearMixedModel::new(formula, &data, Some(&weights)).unwrap();
    model.fit(true).unwrap();

    let err = model.kenward_roger_adjusted_vcov().unwrap_err();

    assert!(err
        .to_string()
        .contains("unweighted iid Gaussian residual models"));
}

#[test]
fn test_kenward_roger_lbddf_scalar_contrast_matches_expected_scale() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(true).unwrap();

    let l = DMatrix::from_row_slice(1, model.coef_names().len(), &[0.0, 1.0]);
    let ddf = model.kenward_roger_lbddf(&l).unwrap();

    assert_eq!(ddf.restriction_rank, 1);
    assert_relative_eq!(ddf.numerator_df, 1.0, epsilon = 1e-12);
    assert!(ddf.denominator_df.is_finite());
    assert!(
        (15.0..=20.0).contains(&ddf.denominator_df),
        "pbkrtest sleepstudy days df is expected near 17, got {}",
        ddf.denominator_df
    );
    assert!(ddf.a1.is_finite());
    assert!(ddf.a2.is_finite());
    assert!(ddf.b.is_finite());
    assert!(ddf.g.is_finite());
    assert!(ddf.rho.is_finite());
    assert!(matches!(
        ddf.reliability,
        ReliabilityGrade::Moderate | ReliabilityGrade::Low
    ));
}

#[test]
fn test_kenward_roger_lbddf_handles_rank_deficient_restriction_matrix() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(true).unwrap();

    let l = DMatrix::from_row_slice(
        2,
        model.coef_names().len(),
        &[
            0.0, 1.0, //
            0.0, 1.0,
        ],
    );
    let ddf = model.kenward_roger_lbddf(&l).unwrap();

    assert_eq!(ddf.restriction_rank, 1);
    assert_relative_eq!(ddf.numerator_df, 1.0, epsilon = 1e-12);
    assert!(ddf.used_generalized_inverse);
    assert!(ddf
        .notes
        .iter()
        .any(|note| note.contains("row rank 1 is lower")));
    assert!(ddf.denominator_df.is_finite());
    assert!(ddf.denominator_df > 0.0);
}

#[test]
fn test_kenward_roger_lbddf_multi_df_contrast_returns_rank_df() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(true).unwrap();

    let l = DMatrix::identity(model.coef_names().len(), model.coef_names().len());
    let ddf = model.kenward_roger_lbddf(&l).unwrap();

    assert_eq!(ddf.restriction_rank, 2);
    assert_relative_eq!(ddf.numerator_df, 2.0, epsilon = 1e-12);
    assert!(ddf.denominator_df.is_finite());
    assert!(ddf.denominator_df > 0.0);
}

#[test]
fn test_lmm_explicit_kenward_roger_scalar_request_returns_t_test() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(true).unwrap();

    let hypothesis =
        FixedEffectHypothesis::single_coefficient("days = 0", 1, model.coef_names().len()).unwrap();
    let test = model.test_contrast_with_method(hypothesis, FixedEffectTestMethod::KenwardRoger);

    assert_eq!(test.method, InferenceMethod::KenwardRoger);
    assert_eq!(test.status, InferenceStatus::Available);
    assert!(test.numerator_df.is_none());
    assert!(test.denominator_df.unwrap().is_finite());
    assert!((15.0..=20.0).contains(&test.denominator_df.unwrap()));
    assert!(test.standard_errors[0].unwrap().is_finite());
    assert!(test.statistics[0].unwrap().is_finite());
    assert!(test.p_values[0].unwrap().is_finite());
    assert!((0.0..=1.0).contains(&test.p_values[0].unwrap()));
    assert!(test.notes.iter().any(|note| note.contains("Kenward-Roger")));
}

#[test]
fn test_fixed_effect_h0_simulation_smoke_for_analytic_p_values() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula.clone(), &data, None).unwrap();
    model.fit(true).unwrap();

    let days_index = model
        .coef_names()
        .iter()
        .position(|name| name == "days")
        .unwrap();
    let hypothesis =
        FixedEffectHypothesis::single_coefficient("days = 0", days_index, model.coef_names().len())
            .unwrap();
    let target = model
        .fixed_effect_null_bootstrap_target(&hypothesis)
        .unwrap();

    let mut rng = StdRng::seed_from_u64(20260501);
    let mut wald_p_values = Vec::new();
    let mut satterthwaite_p_values = Vec::new();
    let mut kenward_roger_p_values = Vec::new();

    for _ in 0..8 {
        let y_sim = model.simulate_fixed_effect_null(&mut rng, &target).unwrap();
        let mut sim_data = DataFrame::new();
        sim_data
            .add_numeric("reaction", y_sim.iter().copied().collect())
            .unwrap();
        sim_data
            .add_numeric("days", data.numeric("days").unwrap().to_vec())
            .unwrap();
        let subj = data.categorical("subj").unwrap();
        sim_data
            .add_categorical_with_levels("subj", subj.values.clone(), subj.levels.clone())
            .unwrap();
        let mut work = LinearMixedModel::new(formula.clone(), &sim_data, None).unwrap();
        work.fit(true).unwrap();

        let wald = work
            .test_contrast_with_method(hypothesis.clone(), FixedEffectTestMethod::AsymptoticWaldZ);
        let satterthwaite = work
            .test_contrast_with_method(hypothesis.clone(), FixedEffectTestMethod::Satterthwaite);
        let kenward_roger =
            work.test_contrast_with_method(hypothesis.clone(), FixedEffectTestMethod::KenwardRoger);

        assert_eq!(wald.status, InferenceStatus::Available);
        assert_eq!(satterthwaite.status, InferenceStatus::Available);
        assert_eq!(kenward_roger.status, InferenceStatus::Available);
        wald_p_values.push(wald.p_values[0].unwrap());
        satterthwaite_p_values.push(satterthwaite.p_values[0].unwrap());
        kenward_roger_p_values.push(kenward_roger.p_values[0].unwrap());
    }

    for (label, values) in [
        ("Wald", &wald_p_values),
        ("Satterthwaite", &satterthwaite_p_values),
        ("Kenward-Roger", &kenward_roger_p_values),
    ] {
        assert_eq!(values.len(), 8, "{label} should produce all p-values");
        assert!(
            values
                .iter()
                .all(|p| p.is_finite() && (0.0..=1.0).contains(p)),
            "{label} p-values should be finite probabilities: {values:?}"
        );
        let tiny = values.iter().filter(|&&p| p < 0.01).count();
        assert!(
            tiny <= 2,
            "{label} produced too many tiny p-values under the simulated null: {values:?}"
        );
    }
}

#[test]
fn test_lmm_test_contrast_returns_labeled_asymptotic_result() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let days_index = model
        .coef_names()
        .iter()
        .position(|name| name == "days")
        .unwrap();
    let hypothesis =
        FixedEffectHypothesis::single_coefficient("days = 0", days_index, model.coef_names().len())
            .unwrap();
    let test = model.test_contrast_with_method(hypothesis, FixedEffectTestMethod::AsymptoticWaldZ);

    assert!(matches!(test.status, InferenceStatus::Available));
    assert_eq!(test.p_values.len(), 1);
    assert!(test.p_values[0].unwrap() < 0.01);
    assert_eq!(test.estimability.status, EstimabilityStatus::Estimable);
    assert!(test
        .notes
        .iter()
        .any(|note| note.contains("asymptotic Wald z")));
}

#[test]
fn test_lmm_explicit_satterthwaite_request_returns_scalar_t_test() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let days_index = model
        .coef_names()
        .iter()
        .position(|name| name == "days")
        .unwrap();
    let hypothesis =
        FixedEffectHypothesis::single_coefficient("days = 0", days_index, model.coef_names().len())
            .unwrap();

    let test = model.test_contrast_with_method(hypothesis, FixedEffectTestMethod::Satterthwaite);

    assert_eq!(test.method, InferenceMethod::Satterthwaite);
    assert_eq!(test.status, InferenceStatus::Available);
    assert_eq!(test.reliability, ReliabilityGrade::Moderate);
    assert!(test.denominator_df.unwrap().is_finite());
    assert!(test.denominator_df.unwrap() > 0.0);
    assert!(test.p_values[0].unwrap().is_finite());
    assert!((0.0..=1.0).contains(&test.p_values[0].unwrap()));
    assert!(test.statistics[0].unwrap().is_finite());
    assert!(test
        .notes
        .iter()
        .any(|note| note.contains("Satterthwaite denominator df computed")));
}

#[test]
fn test_lmm_satterthwaite_boundary_and_rank_deficient_cases_return_reasons() {
    let data = dyestuff2_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();
    let hypothesis =
        FixedEffectHypothesis::single_coefficient("(Intercept) = 0", 0, model.coef_names().len())
            .unwrap();
    let test = model.test_contrast_with_method(hypothesis, FixedEffectTestMethod::Satterthwaite);

    assert_eq!(test.method, InferenceMethod::Satterthwaite);
    assert!(
        matches!(test.status, InferenceStatus::NotAssessed { ref reason }
            if reason.contains("lower bound"))
    );
    assert_eq!(test.p_values, vec![None]);

    let n = 30usize;
    let x: Vec<f64> = (0..n).map(|i| (i % 5) as f64).collect();
    let x2: Vec<f64> = x.iter().map(|&v| 2.0 * v).collect();
    let y: Vec<f64> = (0..n).map(|i| (i % 7) as f64 + 1.0).collect();
    let z: Vec<String> = (0..n).map(|i| format!("G{}", i % 6)).collect();

    let mut df = DataFrame::new();
    df.add_numeric("y", y).unwrap();
    df.add_numeric("x", x).unwrap();
    df.add_numeric("x2", x2).unwrap();
    df.add_categorical("z", z).unwrap();

    let formula = parse_formula("y ~ 1 + x + x2 + (1 | z)").unwrap();
    let mut model = LinearMixedModel::new(formula, &df, None).unwrap();
    model.fit(false).unwrap();
    let dropped_label = model
        .fixed_effect_inference_table()
        .rows
        .into_iter()
        .find(|row| row.status == FixedEffectInferenceStatus::NotEstimable)
        .expect("rank-deficient fit should mark one coefficient not estimable")
        .label;
    let dropped_index = model
        .coef_names()
        .iter()
        .position(|name| name == &dropped_label)
        .unwrap();
    let hypothesis = FixedEffectHypothesis::single_coefficient(
        format!("{dropped_label} = 0"),
        dropped_index,
        model.coef_names().len(),
    )
    .unwrap();
    let test = model.test_contrast_with_method(hypothesis, FixedEffectTestMethod::Satterthwaite);

    assert!(
        matches!(test.status, InferenceStatus::NotEstimable { ref reason }
            if reason.contains("aliased") || reason.contains("non-finite"))
    );
    assert_eq!(test.p_values, vec![None]);
}

#[test]
fn test_lmm_fixed_effect_inference_table_returns_ordered_satterthwaite_rows() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let table = model.fixed_effect_inference_table();
    let names = model.coef_names();

    assert_eq!(table.rows.len(), names.len());
    assert_eq!(
        table
            .rows
            .iter()
            .map(|row| row.label.clone())
            .collect::<Vec<_>>(),
        names
    );
    for row in &table.rows {
        assert_eq!(row.kind, FixedEffectInferenceRowKind::Coefficient);
        assert_eq!(row.method, FixedEffectInferenceMethod::Satterthwaite);
        assert_eq!(row.status, FixedEffectInferenceStatus::Available);
        assert_eq!(row.reliability, ReliabilityGrade::Moderate);
        assert_eq!(
            row.reliability_reason,
            Some(FixedEffectReliabilityReason::SatterthwaiteFiniteDifferenceApproximation)
        );
        assert_eq!(row.statistic_name, Some(FixedEffectStatisticName::T));
        assert!(row.estimate.is_some());
        assert!(row.std_error.is_some());
        assert!(row.statistic.is_some());
        assert!(row.p_value.is_some());
        assert!(row.numerator_df.is_none());
        assert!(row.denominator_df.is_some());
        assert!(row.reason.is_none());
        assert!(matches!(
            row.estimability,
            EstimabilityAssessment::FixedContrast(_)
        ));
        assert!(row
            .notes
            .iter()
            .any(|note| note.contains("Satterthwaite denominator df")));
    }
    let artifact_table = model
        .compiler_artifact()
        .fixed_effect_inference_table
        .as_ref()
        .expect("fitted artifact should carry cheap fixed-effect rows");
    assert_eq!(artifact_table.rows.len(), table.rows.len());
    for row in &artifact_table.rows {
        assert_eq!(row.kind, FixedEffectInferenceRowKind::Coefficient);
        assert_eq!(row.method, FixedEffectInferenceMethod::AsymptoticWaldZ);
        assert_eq!(
            row.reliability_reason,
            Some(FixedEffectReliabilityReason::AsymptoticWaldZFallback)
        );
        assert_eq!(row.statistic_name, Some(FixedEffectStatisticName::Z));
        assert!(row.denominator_df.is_none());
    }
}

#[test]
fn test_lmm_fixed_effect_covariance_matrix_unavailable_for_rank_deficient_fit() {
    let n = 30usize;
    let x: Vec<f64> = (0..n).map(|i| (i % 5) as f64).collect();
    let x2: Vec<f64> = x.iter().map(|&v| 2.0 * v).collect();
    let y: Vec<f64> = (0..n).map(|i| (i % 7) as f64 + 1.0).collect();
    let z: Vec<String> = (0..n).map(|i| format!("G{}", i % 6)).collect();

    let mut df = DataFrame::new();
    df.add_numeric("y", y).unwrap();
    df.add_numeric("x", x).unwrap();
    df.add_numeric("x2", x2).unwrap();
    df.add_categorical("z", z).unwrap();

    let formula = parse_formula("y ~ 1 + x + x2 + (1 | z)").unwrap();
    let mut model = LinearMixedModel::new(formula, &df, None).unwrap();
    model.fit(false).unwrap();

    let payload = model.fixed_effect_covariance_matrix();

    assert_eq!(payload.status, FixedEffectCovarianceStatus::Unavailable);
    assert_eq!(payload.method, FixedEffectCovarianceMethod::Unavailable);
    assert_eq!(payload.reliability, ReliabilityGrade::NotAvailable);
    assert_eq!(
        payload.reason.as_deref(),
        Some("rank_deficient_fixed_effects")
    );
    assert_eq!(payload.matrix, None);
    assert_eq!(payload.details.rank, Some(2));
    assert_eq!(payload.details.expected_rank, Some(3));
    assert_eq!(payload.details.aliased.len(), 1);
    assert!(payload.details.aliased[0] == "x" || payload.details.aliased[0] == "x2");
    assert_eq!(payload.details.finite, Some(false));
    assert_eq!(
        model
            .compiler_artifact()
            .fixed_effect_covariance_matrix
            .as_ref(),
        Some(&payload)
    );
}

#[test]
fn test_lmm_fixed_effect_inference_table_marks_aliased_column_not_estimable() {
    let n = 30usize;
    let x: Vec<f64> = (0..n).map(|i| (i % 5) as f64).collect();
    let x2: Vec<f64> = x.iter().map(|&v| 2.0 * v).collect();
    let y: Vec<f64> = (0..n).map(|i| (i % 7) as f64 + 1.0).collect();
    let z: Vec<String> = (0..n).map(|i| format!("G{}", i % 6)).collect();

    let mut df = DataFrame::new();
    df.add_numeric("y", y).unwrap();
    df.add_numeric("x", x).unwrap();
    df.add_numeric("x2", x2).unwrap();
    df.add_categorical("z", z).unwrap();

    let formula = parse_formula("y ~ 1 + x + x2 + (1 | z)").unwrap();
    let mut model = LinearMixedModel::new(formula, &df, None).unwrap();
    model.fit(false).unwrap();

    let table = model.fixed_effect_inference_table();
    let dropped = table
        .rows
        .iter()
        .find(|row| row.status == FixedEffectInferenceStatus::NotEstimable)
        .expect("one aliased coefficient should be marked not estimable");

    assert_eq!(dropped.method, FixedEffectInferenceMethod::NotComputed);
    assert_eq!(dropped.reliability, ReliabilityGrade::NotAvailable);
    assert!(dropped.p_value.is_none());
    assert!(dropped.reason.as_deref().unwrap().contains("aliased"));
}

#[test]
fn test_lmm_fixed_effect_inference_table_omits_p_values_for_predictive_fit_intent() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new_with_compiler_policy(
        formula,
        &data,
        None,
        CompilerPolicy::predictive(),
    )
    .unwrap();

    model.fit(false).unwrap();

    assert_eq!(
        model.compiler_artifact().reproducibility.fit_intent,
        FitIntent::Predictive
    );
    let table = model
        .compiler_artifact()
        .fixed_effect_inference_table
        .as_ref()
        .expect("fitted artifacts should carry fixed-effect inference rows");
    assert!(table.rows.iter().all(|row| {
        row.status == FixedEffectInferenceStatus::PValueUnavailable
            && row.method == FixedEffectInferenceMethod::NotComputed
            && row.p_value.is_none()
            && row
                .reason
                .as_deref()
                .unwrap()
                .contains("predictive fit intent")
    }));
}

#[test]
fn test_lmm_test_contrast_marks_aliased_column_not_estimable() {
    let n = 30usize;
    let x: Vec<f64> = (0..n).map(|i| (i % 5) as f64).collect();
    let x2: Vec<f64> = x.iter().map(|&v| 2.0 * v).collect();
    let y: Vec<f64> = (0..n).map(|i| (i % 7) as f64 + 1.0).collect();
    let z: Vec<String> = (0..n).map(|i| format!("G{}", i % 6)).collect();

    let mut df = DataFrame::new();
    df.add_numeric("y", y).unwrap();
    df.add_numeric("x", x).unwrap();
    df.add_numeric("x2", x2).unwrap();
    df.add_categorical("z", z).unwrap();

    let formula = parse_formula("y ~ 1 + x + x2 + (1 | z)").unwrap();
    let mut model = LinearMixedModel::new(formula, &df, None).unwrap();
    model.fit(false).unwrap();
    let ct = model.coeftable();
    let dropped = ct
        .std_errors
        .iter()
        .position(|se| se.is_nan())
        .expect("one fixed-effect column should be dropped");

    let hypothesis =
        FixedEffectHypothesis::single_coefficient("dropped coefficient", dropped, ct.len())
            .unwrap();
    let test = model.test_contrast(hypothesis);

    assert!(matches!(test.status, InferenceStatus::NotEstimable { .. }));
    assert_eq!(test.estimability.status, EstimabilityStatus::NotEstimable);
    assert_eq!(test.p_values, vec![None]);
}

#[test]
fn test_lmm_fixed_effect_term_rows_are_rust_owned() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let hypotheses = model.fixed_effect_term_hypotheses();
    assert!(hypotheses
        .iter()
        .any(|hypothesis| hypothesis.label == "days"));

    let table = model.fixed_effect_term_inference_table(FixedEffectTestMethod::Auto);
    let days = table
        .rows
        .iter()
        .find(|row| row.label == "days")
        .expect("days term row should be exposed");
    assert_eq!(days.kind, FixedEffectInferenceRowKind::Term);
    let family = days
        .details
        .as_ref()
        .and_then(|details| details.contrast_family.as_ref())
        .expect("term row should carry contrast-family details");
    assert_eq!(family.family_label, "days");
    assert_eq!(family.restriction_rows, 1);
    assert_eq!(family.coefficient_count, model.coef_names().len());
}

#[test]
fn test_lmm_fixed_effect_term_hypotheses_have_explicit_type_semantics() {
    let model = typed_term_test_fixture();
    let names = model.coef_names();
    let x_index = names.iter().position(|name| name == "x").unwrap();

    let type_i = model.fixed_effect_term_hypotheses_for_type(FixedEffectTermTestType::TypeI);
    let type_ii = model.fixed_effect_term_hypotheses_for_type(FixedEffectTermTestType::TypeII);
    let type_iii = model.fixed_effect_term_hypotheses_for_type(FixedEffectTermTestType::TypeIII);

    let x_type_i = hypothesis_by_label(&type_i, "x");
    let x_type_ii = hypothesis_by_label(&type_ii, "x");
    let x_type_iii = hypothesis_by_label(&type_iii, "x");
    let interaction_type_ii = hypothesis_by_label(&type_ii, "x:z");

    assert_eq!(x_type_iii.l.values.nrows(), 1);
    assert_eq!(x_type_iii.l.values.ncols(), names.len());
    for col in 0..names.len() {
        let expected = if col == x_index { 1.0 } else { 0.0 };
        assert_relative_eq!(x_type_iii.l.values[(0, col)], expected, epsilon = 1.0e-12);
    }

    assert!(
            matrices_differ(&x_type_i.l.values, &x_type_iii.l.values, 1.0e-9),
            "Type I x hypothesis should not collapse to the Type III coefficient block in the interaction fixture"
        );
    assert!(
            matrices_differ(&x_type_ii.l.values, &x_type_iii.l.values, 1.0e-9),
            "Type II x hypothesis should not collapse to the Type III coefficient block in the interaction fixture"
        );
    assert_eq!(interaction_type_ii.l.values.nrows(), 1);
    assert_eq!(interaction_type_ii.l.values.ncols(), names.len());

    let table = model.fixed_effect_term_inference_table_for_type(
        FixedEffectTestMethod::Satterthwaite,
        FixedEffectTermTestType::TypeII,
    );
    let x_row = table
        .rows
        .iter()
        .find(|row| row.label == "x")
        .expect("Type II table should include x term row");
    assert_eq!(x_row.kind, FixedEffectInferenceRowKind::Term);
    assert!(x_row
        .notes
        .iter()
        .any(|note| note.contains("fixed-effect term test type: type_ii")));
}

#[test]
fn test_lmm_fixed_effect_contrast_table_is_rust_owned() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let days_index = model
        .coef_names()
        .iter()
        .position(|name| name == "days")
        .unwrap();
    let hypothesis =
        FixedEffectHypothesis::single_coefficient("days = 0", days_index, model.coef_names().len())
            .unwrap();
    let table =
        model.fixed_effect_contrast_inference_table(vec![hypothesis], FixedEffectTestMethod::Auto);

    assert_eq!(
        table.schema_name,
        mixeff_rs::compiler::FIXED_EFFECT_INFERENCE_TABLE_SCHEMA
    );
    assert_eq!(table.rows.len(), 1);
    let row = &table.rows[0];
    assert_eq!(row.kind, FixedEffectInferenceRowKind::Contrast);
    assert_eq!(row.label, "days = 0");
    assert_eq!(row.status, FixedEffectInferenceStatus::Available);
    let family = row
        .details
        .as_ref()
        .and_then(|details| details.contrast_family.as_ref())
        .expect("contrast row should carry contrast-family details");
    assert_eq!(family.family_label, "days = 0");
    assert_eq!(
        family.numerator_df_semantics,
        "scalar_contrast_no_numerator_df"
    );
}

#[test]
fn test_fixed_effect_null_bootstrap_target_projects_beta_and_simulates() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let days_index = model
        .coef_names()
        .iter()
        .position(|name| name == "days")
        .unwrap();
    let hypothesis =
        FixedEffectHypothesis::single_coefficient("days = 0", days_index, model.coef_names().len())
            .unwrap();

    let target = model
        .fixed_effect_null_bootstrap_target(&hypothesis)
        .unwrap();
    let fitted_contrast = (&hypothesis.l.values * &target.beta_fitted)[0];
    let null_contrast = (&hypothesis.l.values * &target.beta_null)[0];

    assert_eq!(target.target.kind, BootstrapTargetKind::FixedEffectNull);
    assert_eq!(
        target.covariance_policy,
        FixedEffectNullCovariancePolicy::ReuseFittedCovariance
    );
    assert!(fitted_contrast.abs() > 1.0);
    assert_relative_eq!(null_contrast, 0.0, epsilon = 1e-8);
    assert_eq!(target.theta, model.theta());
    assert_relative_eq!(target.sigma, model.sigma(), epsilon = 1e-12);
    assert!(target
        .notes
        .iter()
        .any(|note| note.contains("reuses fitted covariance")));

    let mut rng = StdRng::seed_from_u64(20260429);
    let y_sim = model.simulate_fixed_effect_null(&mut rng, &target).unwrap();
    assert_eq!(y_sim.len(), model.nobs());

    let mut mismatched = target.clone();
    mismatched.sigma *= 1.01;
    assert!(matches!(
        model.simulate_fixed_effect_null(&mut rng, &mismatched),
        Err(MixedModelError::InvalidArgument(_))
    ));
}

#[test]
fn test_fixed_effect_null_bootstrap_table_callable_returns_inference_table() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let days_index = model
        .coef_names()
        .iter()
        .position(|name| name == "days")
        .unwrap();
    let hypothesis =
        FixedEffectHypothesis::single_coefficient("days = 0", days_index, model.coef_names().len())
            .unwrap();
    let table = model.fixed_effect_null_bootstrap_inference_table(
        vec![hypothesis],
        FixedEffectBootstrapOptions {
            requested_replicates: 2,
            failed_refit_policy: BootstrapFailedRefitPolicy::Exclude,
            seed: Some(20260503),
        },
    );

    assert_eq!(
        table.schema_name,
        mixeff_rs::compiler::FIXED_EFFECT_INFERENCE_TABLE_SCHEMA
    );
    assert_eq!(table.rows.len(), 1);
    let row = &table.rows[0];
    assert_eq!(row.method, FixedEffectInferenceMethod::Bootstrap);
    assert_eq!(row.kind, FixedEffectInferenceRowKind::Contrast);
    assert!(matches!(
        row.status,
        FixedEffectInferenceStatus::Available | FixedEffectInferenceStatus::NotAssessed
    ));
    let bootstrap = row
        .details
        .as_ref()
        .and_then(|details| details.bootstrap.as_ref())
        .expect("bridge row should carry bootstrap details");
    assert_eq!(bootstrap.requested_replicates, 2);
    assert_eq!(bootstrap.seed, Some(20260503));
    assert!(bootstrap.null_target.is_some());
}

#[test]
fn test_fixed_effect_null_bootstrap_multi_df_term_returns_joint_f_row() {
    let (model, hypothesis) = three_level_condition_fixture();

    let row = model.fixed_effect_null_bootstrap_inference_row(
        FixedEffectInferenceRowKind::Term,
        hypothesis,
        &FixedEffectBootstrapOptions {
            requested_replicates: 35,
            failed_refit_policy: BootstrapFailedRefitPolicy::Exclude,
            seed: Some(20260512),
        },
    );

    assert_eq!(row.kind, FixedEffectInferenceRowKind::Term);
    assert_eq!(row.method, FixedEffectInferenceMethod::Bootstrap);
    assert_eq!(row.status, FixedEffectInferenceStatus::Available);
    assert_eq!(row.statistic_name, Some(FixedEffectStatisticName::F));
    assert_eq!(row.numerator_df, Some(2.0));
    assert!(row.denominator_df.is_none());
    assert!(row.statistic.unwrap().is_finite());
    assert!(row.p_value.unwrap().is_finite());
    assert!(row
        .notes
        .iter()
        .any(|note| note.contains("statistic=joint_wald_f")));

    let details = row.details.expect("term row should carry details");
    let bootstrap = details.bootstrap.expect("bootstrap metadata");
    assert_eq!(bootstrap.target_kind, "fixed_effect_null");
    assert_eq!(bootstrap.requested_replicates, 35);
    assert_eq!(bootstrap.finite_statistic_count, Some(35));
    let family = details.contrast_family.expect("contrast-family metadata");
    assert_eq!(family.restriction_rows, 2);
    assert_eq!(family.effective_rank, Some(2));
    assert_eq!(family.numerator_df, Some(2.0));
    assert_eq!(family.numerator_df_semantics, "effective_restriction_rank");
}

#[test]
fn test_cluster_resample_full_model_contrast_payload_returns_intervals() {
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();
    let hypothesis =
        FixedEffectHypothesis::single_coefficient("intercept", 0, model.coef_names().len())
            .unwrap();

    let payload = model
        .cluster_resample_full_model_contrast_payload(
            &data,
            "batch",
            &hypothesis,
            &FixedEffectBootstrapOptions {
                requested_replicates: 3,
                failed_refit_policy: BootstrapFailedRefitPolicy::Exclude,
                seed: Some(20260517),
            },
            &[0.95],
        )
        .unwrap();

    assert_eq!(
        payload.metadata.target.kind,
        BootstrapTargetKind::ClusterResample
    );
    assert_eq!(payload.metadata.requested_replicates, 3);
    assert_eq!(payload.metadata.completed_replicates, 3);
    assert_eq!(payload.metadata.finite_statistic_count, Some(3));
    assert!(payload.metadata.mcse.is_none());
    assert_eq!(payload.replicate_statistics.as_ref().map(Vec::len), Some(3));
    let intervals = payload.intervals.as_ref().expect("intervals");
    assert_eq!(intervals.len(), 1);
    assert_eq!(intervals[0].parameter, "intercept");
    assert_eq!(intervals[0].method, BootstrapIntervalMethod::Percentile);
    assert!(payload
        .metadata
        .notes
        .iter()
        .any(|note| note.contains("estimator-distribution target")));
}
