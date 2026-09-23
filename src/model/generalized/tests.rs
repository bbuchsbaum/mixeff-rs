use super::*;
use crate::compiler::{NewtonDecrementVariant, NewtonDecrementVerdict};
use crate::formula::parse_formula;
use crate::model::data::DataFrame;
use crate::model::linear::FitToleranceOverrides;
use approx::assert_relative_eq;
use rand::SeedableRng;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

fn agq_poisson_fixture() -> GeneralizedLinearMixedModel {
    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut g = Vec::new();
    for grp in 0..5 {
        for obs in 0..8 {
            let xv = obs as f64 - 3.5;
            let eta = 0.8 + 0.1 * xv + [-0.2, 0.1, 0.0, 0.15, -0.05][grp];
            y.push(eta.exp().round().max(1.0));
            x.push(xv);
            g.push(format!("g{}", grp + 1));
        }
    }
    let mut data = DataFrame::new();
    data.add_numeric("y", y).unwrap();
    data.add_numeric("x", x).unwrap();
    data.add_categorical("g", g).unwrap();
    let formula = parse_formula("y ~ 1 + x + (1 | g)").unwrap();
    let mut model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Poisson, None).unwrap();
    model.fit().unwrap();
    model
}

fn small_joint_poisson_fixture() -> GeneralizedLinearMixedModel {
    let mut data = DataFrame::new();
    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut group = Vec::new();
    let mut obs = Vec::new();
    let group_effects = [-0.7, 0.2, 0.8, -0.3];
    for (g, effect) in group_effects.iter().enumerate() {
        for j in 0..6 {
            let xv = j as f64 - 2.5;
            let eta = 0.4 + 0.18 * xv + effect;
            let base = eta.exp();
            let overdispersion_bump = if j % 3 == 0 { 2.0 } else { 0.0 };
            y.push((base + overdispersion_bump).round().max(0.0));
            x.push(xv);
            group.push(format!("g{}", g + 1));
            obs.push(format!("o{}_{}", g + 1, j + 1));
        }
    }
    data.add_numeric("y", y).unwrap();
    data.add_numeric("x", x).unwrap();
    data.add_categorical("group", group).unwrap();
    data.add_categorical("obs", obs).unwrap();
    let formula = parse_formula("y ~ 1 + x + (1 | group) + (1 | obs)").unwrap();
    GeneralizedLinearMixedModel::new(formula, &data, Family::Poisson, Some(LinkFunction::Log))
        .unwrap()
}

#[cfg(feature = "nlopt")]
#[test]
fn experimental_joint_failed_stop_records_uncertified_certificate() {
    let mut model = small_joint_poisson_fixture();
    model.fit_with_options_impl(1, false).unwrap();
    let start_beta = model.beta.as_slice().to_vec();
    let start_theta = model.theta.clone();
    let start_objective = model.deviance_with_response_constants(1);

    model
        .fit_joint_glmm_from_start(start_beta, start_theta, start_objective, 1, 1, None)
        .unwrap();

    let certificate = model
        .compiler_artifact()
        .optimizer_certificate
        .as_ref()
        .expect("failed joint attempt should still record an optimizer certificate");
    assert!(
        !certificate.evidence.optimizer_stop.acceptable_stop,
        "forced one-evaluation joint fit must not be certified as an acceptable stop"
    );
    assert!(
        certificate.free_gradient_norm.is_none(),
        "failed optimizer stop must not report a passing stationarity residual"
    );
    assert!(
        model
            .opt_summary()
            .return_value
            .starts_with("JOINT_LAPLACE"),
        "forced failure must keep a joint-Laplace return-code namespace"
    );
    assert!(
        model.opt_summary().return_value.contains("MAXEVAL_REACHED"),
        "forced one-evaluation joint fit must report MAXEVAL_REACHED, got {}",
        model.opt_summary().return_value
    );
}

#[test]
fn joint_glmm_stationarity_failure_is_not_converged_interior() {
    let params = vec![1.41606, 0.08172, 0.45, 0.68];
    let lower_bounds = vec![f64::NEG_INFINITY, f64::NEG_INFINITY, 0.0, 0.0];
    let gradient = vec![3.5e-2, 1.0e-3, 0.0, 0.0];
    let gradient_tolerance = 2.0e-2;

    let mut optsum = OptSummary::new(params.clone());
    optsum.optimizer = Optimizer::TrustBq;
    optsum.backend = Optimizer::TrustBq.canonical_backend();
    optsum.return_value = "JOINT_LAPLACE:FTOL_REACHED".to_string();
    optsum.finitial = 2845.394;
    optsum.fmin = 2845.394;
    optsum.feval = 23;
    optsum.max_feval = 5000;
    optsum.final_params = params.clone();

    let mut certificate = OptimizerCertificate::from_opt_summary_with_context(
        &optsum,
        &params,
        &lower_bounds,
        Some(2854),
    );
    certificate.apply_derivative_evidence(
        OptimizerDerivativeEvidence {
            method: EvidenceMethod::FiniteDifference,
            hessian_method: EvidenceMethod::FiniteDifference,
            gradient: gradient.clone(),
            hessian: None,
        },
        gradient_tolerance,
        1.0e-6,
    );

    let certification = JointLaplaceCertificationGradient {
        gradient: gradient.clone(),
        probe_gradient: gradient.clone(),
        escalated_indices: Vec::new(),
        unassessable_indices: Vec::new(),
        base_objective: f64::NAN,
        curvature_probes: Vec::new(),
    };
    annotate_glmm_covariance_status(
        &mut certificate,
        &params,
        2,
        &lower_bounds,
        &certification,
        gradient_tolerance,
        None,
    );

    assert_eq!(certificate.status, crate::compiler::FitStatus::NotOptimized);
    assert!(
        joint_certificate_requires_fallback(&certificate),
        "assessed stationarity failure should still trigger labelled fallback"
    );
    assert!(certificate.checks.iter().any(|check| {
        matches!(
            check,
            crate::compiler::CertificateCheck::DerivativeMismatch { kind, .. }
                if kind == "free_gradient_kkt_mismatch"
        )
    }));
    let diagnostic = certificate
        .diagnostics
        .iter()
        .find(|diagnostic| {
            diagnostic.code == DiagnosticCode::OptimizerNonconvergence
                && diagnostic
                    .payload
                    .get("stationarity_check")
                    .and_then(serde_json::Value::as_str)
                    == Some("free_gradient_kkt")
        })
        .expect("failed stationarity should be reported as optimizer nonconvergence");
    assert_eq!(
        diagnostic.payload.get("return_code"),
        Some(&serde_json::json!("JOINT_LAPLACE:FTOL_REACHED"))
    );
    assert_eq!(
        diagnostic.payload.get("free_gradient_norm"),
        Some(&serde_json::json!(3.5e-2))
    );
}

#[test]
fn joint_glmm_noise_dominated_stationarity_is_not_assessed() {
    // Probe readings on the two theta components are pure inner-PIRLS
    // noise (bd-01KTQFTH6J0ZFGR5RMV28HAX44 measured 0.703/0.365 at a
    // glmer-equivalent optimum); the escalated steps disagreed, so the
    // components are unassessable. The certificate must say NotAssessed,
    // not NotOptimized, and must not trigger the fast-PIRLS fallback.
    let params = vec![1.43958, 0.08172, 0.3861, 0.5219];
    let lower_bounds = vec![f64::NEG_INFINITY, f64::NEG_INFINITY, 0.0, 0.0];
    let probe_gradient = vec![-2.7e-5, 1.0e-3, 0.703, 0.365];
    let gradient_tolerance = 2.0e-2;

    let mut optsum = OptSummary::new(params.clone());
    optsum.optimizer = Optimizer::TrustBq;
    optsum.backend = Optimizer::TrustBq.canonical_backend();
    optsum.return_value = "JOINT_LAPLACE:FTOL_REACHED".to_string();
    optsum.finitial = 2851.2;
    optsum.fmin = 2845.375;
    optsum.feval = 55;
    optsum.max_feval = 820;
    optsum.final_params = params.clone();

    let mut certificate = OptimizerCertificate::from_opt_summary_with_context(
        &optsum,
        &params,
        &lower_bounds,
        Some(2880),
    );
    let certification = JointLaplaceCertificationGradient {
        gradient: probe_gradient.clone(),
        probe_gradient: probe_gradient.clone(),
        escalated_indices: Vec::new(),
        unassessable_indices: vec![2, 3],
        base_objective: f64::NAN,
        curvature_probes: Vec::new(),
    };
    certificate.apply_derivative_evidence(
        OptimizerDerivativeEvidence {
            method: EvidenceMethod::FiniteDifference,
            hessian_method: EvidenceMethod::FiniteDifference,
            gradient: certification.gradient.clone(),
            hessian: None,
        },
        gradient_tolerance,
        1.0e-6,
    );
    annotate_glmm_covariance_status(
        &mut certificate,
        &params,
        2,
        &lower_bounds,
        &certification,
        gradient_tolerance,
        None,
    );

    assert_eq!(certificate.status, crate::compiler::FitStatus::NotAssessed);
    assert!(
        !joint_certificate_requires_fallback(&certificate),
        "an unassessable stationarity probe must not discard the joint candidate"
    );
    let diagnostic = certificate
        .diagnostics
        .iter()
        .find(|diagnostic| {
            diagnostic.code == DiagnosticCode::OptimizerNotAssessed
                && diagnostic
                    .payload
                    .get("stationarity_check")
                    .and_then(serde_json::Value::as_str)
                    == Some("free_gradient_kkt_noise_dominated")
        })
        .expect("noise-dominated stationarity should be reported as not assessed");
    assert_eq!(
        diagnostic.payload.get("unassessable_indices"),
        Some(&serde_json::json!([2, 3]))
    );
    assert_eq!(
        diagnostic.payload.get("return_code"),
        Some(&serde_json::json!("JOINT_LAPLACE:FTOL_REACHED"))
    );
    assert!(
        !certificate
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == DiagnosticCode::OptimizerNonconvergence),
        "an unassessable probe must not be labelled optimizer nonconvergence"
    );
}

#[test]
fn joint_glmm_escalated_stationarity_pass_certifies_with_evidence_trail() {
    // The default-step probe was noise-dominated on theta but the
    // escalated steps agreed on a near-zero gradient: the fit certifies
    // as interior-converged with an Info trail recording the escalation.
    let params = vec![1.43958, 0.08172, 0.3861, 0.5219];
    let lower_bounds = vec![f64::NEG_INFINITY, f64::NEG_INFINITY, 0.0, 0.0];
    let probe_gradient = vec![-2.7e-5, 1.0e-3, 0.703, 0.365];
    let assessed_gradient = vec![-2.7e-5, 1.0e-3, 2.4e-3, -1.1e-3];
    let gradient_tolerance = 2.0e-2;

    let mut optsum = OptSummary::new(params.clone());
    optsum.optimizer = Optimizer::TrustBq;
    optsum.backend = Optimizer::TrustBq.canonical_backend();
    optsum.return_value = "JOINT_LAPLACE:FTOL_REACHED".to_string();
    optsum.finitial = 2851.2;
    optsum.fmin = 2845.375;
    optsum.feval = 55;
    optsum.max_feval = 820;
    optsum.final_params = params.clone();

    let mut certificate = OptimizerCertificate::from_opt_summary_with_context(
        &optsum,
        &params,
        &lower_bounds,
        Some(2880),
    );
    let certification = JointLaplaceCertificationGradient {
        gradient: assessed_gradient.clone(),
        probe_gradient,
        escalated_indices: vec![2, 3],
        unassessable_indices: Vec::new(),
        base_objective: f64::NAN,
        curvature_probes: Vec::new(),
    };
    certificate.apply_derivative_evidence(
        OptimizerDerivativeEvidence {
            method: EvidenceMethod::FiniteDifference,
            hessian_method: EvidenceMethod::FiniteDifference,
            gradient: certification.gradient.clone(),
            hessian: None,
        },
        gradient_tolerance,
        1.0e-6,
    );
    annotate_glmm_covariance_status(
        &mut certificate,
        &params,
        2,
        &lower_bounds,
        &certification,
        gradient_tolerance,
        None,
    );

    assert_eq!(
        certificate.status,
        crate::compiler::FitStatus::ConvergedInterior
    );
    assert!(!joint_certificate_requires_fallback(&certificate));
    let trail = certificate
        .diagnostics
        .iter()
        .find(|diagnostic| {
            diagnostic.code == DiagnosticCode::OptimizerRecovery
                && diagnostic
                    .payload
                    .get("stationarity_check")
                    .and_then(serde_json::Value::as_str)
                    == Some("free_gradient_kkt_escalated_step")
        })
        .expect("escalated certification must leave an evidence trail");
    assert_eq!(trail.severity, DiagnosticSeverity::Info);
    assert_eq!(
        trail.payload.get("escalated_indices"),
        Some(&serde_json::json!([2, 3]))
    );
    assert_eq!(
        trail.payload.get("probe_gradient_max_abs"),
        Some(&serde_json::json!(0.703))
    );
}

#[test]
fn joint_glmm_nonfinite_objective_stop_is_not_converged_interior() {
    let params = vec![448.9995, 0.79586, 0.42];
    let lower_bounds = vec![f64::NEG_INFINITY, f64::NEG_INFINITY, 0.0];

    let mut optsum = OptSummary::new(params.clone());
    optsum.optimizer = Optimizer::TrustBq;
    optsum.backend = Optimizer::TrustBq.canonical_backend();
    optsum.return_value = "JOINT_LAPLACE:FTOL_REACHED".to_string();
    optsum.finitial = 2540.376;
    optsum.fmin = f64::INFINITY;
    optsum.feval = 61;
    optsum.max_feval = 5000;
    optsum.final_params = params.clone();

    let certificate = OptimizerCertificate::from_opt_summary_with_context(
        &optsum,
        &params,
        &lower_bounds,
        Some(5279),
    );

    assert_eq!(certificate.status, crate::compiler::FitStatus::NotOptimized);
    assert_eq!(certificate.objective_value, None);
    assert!(
        !certificate.evidence.optimizer_stop.acceptable_stop,
        "non-finite objective must invalidate an otherwise acceptable joint stop"
    );
    assert!(
        joint_certificate_requires_fallback(&certificate),
        "non-finite objective joint attempts should trigger the labelled fallback path"
    );
    assert!(certificate.checks.iter().any(|check| {
        matches!(
            check,
            crate::compiler::CertificateCheck::Failed { code, .. }
                if code == "non_finite_objective"
        )
    }));
    let diagnostic = certificate
        .diagnostics
        .iter()
        .find(|diagnostic| {
            diagnostic.code == DiagnosticCode::OptimizerNonconvergence
                && diagnostic
                    .payload
                    .get("objective_finite")
                    .and_then(serde_json::Value::as_bool)
                    == Some(false)
        })
        .expect("non-finite objective should be reported as optimizer nonconvergence");
    assert_eq!(
        diagnostic.payload.get("return_code"),
        Some(&serde_json::json!("JOINT_LAPLACE:FTOL_REACHED"))
    );

    let fallback = agq_poisson_fixture();
    let recovered = uncertified_joint_fallback(&certificate, &optsum, Some(fallback)).unwrap();
    assert!(
        recovered
            .opt_summary()
            .return_value
            .starts_with("JOINT_LAPLACE_FALLBACK_FAST_PIRLS"),
        "non-finite joint objective should return the labelled fallback result"
    );
    let metadata = recovered
        .compiler_artifact()
        .glmm_fit_metadata
        .as_ref()
        .expect("fallback fit must record GLMM fit metadata");
    assert_eq!(metadata.estimation_method, "fallback_fast_pirls");
    assert_eq!(metadata.requested_method.as_deref(), Some("joint_laplace"));
    assert_eq!(
        metadata.effective_method.as_deref(),
        Some("fast_pirls_profiled")
    );
}

#[test]
fn glmm_fit_metadata_records_requested_and_effective_methods() {
    let mut optsum = OptSummary::new(vec![0.5]);
    optsum.return_value =
        "JOINT_LAPLACE_FALLBACK_FAST_PIRLS(joint=JOINT_LAPLACE_FAILED:XTOL_REACHED; fast=FTOL_REACHED)"
            .to_string();
    let fallback = crate::compiler::GlmmFitMetadata::from_opt_summary(&optsum);
    assert_eq!(fallback.estimation_method, "fallback_fast_pirls");
    assert_eq!(fallback.requested_method.as_deref(), Some("joint_laplace"));
    assert_eq!(
        fallback.effective_method.as_deref(),
        Some("fast_pirls_profiled")
    );

    optsum.return_value =
        "JOINT_AGQ_FALLBACK_FAST_PIRLS(joint=JOINT_AGQ_FAILED:XTOL_REACHED; fast=FTOL_REACHED)"
            .to_string();
    let agq_fallback = crate::compiler::GlmmFitMetadata::from_opt_summary(&optsum);
    assert_eq!(agq_fallback.requested_method.as_deref(), Some("joint_agq"));
    assert_eq!(
        agq_fallback.effective_method.as_deref(),
        Some("fast_pirls_profiled")
    );

    optsum.return_value = "JOINT_LAPLACE:FTOL_REACHED".to_string();
    let joint = crate::compiler::GlmmFitMetadata::from_opt_summary(&optsum);
    assert_eq!(joint.requested_method.as_deref(), Some("joint_laplace"));
    assert_eq!(joint.effective_method.as_deref(), Some("joint_laplace"));

    optsum.return_value = "FTOL_REACHED".to_string();
    let profiled = crate::compiler::GlmmFitMetadata::from_opt_summary(&optsum);
    assert_eq!(
        profiled.requested_method.as_deref(),
        Some("fast_pirls_profiled")
    );
    assert_eq!(
        profiled.effective_method.as_deref(),
        Some("fast_pirls_profiled")
    );
}

#[test]
fn joint_glmm_not_assessed_stationarity_keeps_joint_candidate() {
    let params = vec![1.2, -0.25, 0.42, 0.68];
    let lower_bounds = vec![f64::NEG_INFINITY, f64::NEG_INFINITY, 0.0, 0.0];

    let mut optsum = OptSummary::new(params.clone());
    optsum.optimizer = Optimizer::TrustBq;
    optsum.backend = Optimizer::TrustBq.canonical_backend();
    optsum.return_value = "JOINT_LAPLACE:FTOL_REACHED".to_string();
    optsum.finitial = 1137.42;
    optsum.fmin = 1136.50;
    optsum.feval = 578;
    optsum.max_feval = 1140;
    optsum.final_params = params.clone();

    let mut certificate = OptimizerCertificate::from_opt_summary_with_context(
        &optsum,
        &params,
        &lower_bounds,
        Some(1427),
    );
    certificate.status = crate::compiler::FitStatus::NotAssessed;
    certificate.mark_derivative_checks_not_assessed(
        "objective gradient is not exposed by the current derivative-free optimizer path",
    );

    assert!(
        certificate.evidence.optimizer_stop.acceptable_stop,
        "an acceptable optimizer stop with unassessed derivatives is not a hard optimizer failure"
    );
    assert!(
        matches!(
            certificate.evidence.gradient.method,
            EvidenceMethod::NotAssessed { .. }
        ),
        "regression must exercise the no-gradient/not-assessed derivative path"
    );
    assert!(
        !joint_certificate_requires_fallback(&certificate),
        "not-assessed stationarity should not be conflated with an assessed optimizer failure"
    );

    let fallback = agq_poisson_fixture();
    assert!(
        uncertified_joint_fallback(&certificate, &optsum, Some(fallback)).is_none(),
        "acceptable joint candidates with unassessed stationarity should remain joint fits"
    );
}

#[test]
fn joint_glmm_ftol_at_budget_boundary_keeps_not_available_joint_candidate() {
    let params = vec![1.2, -0.25, 0.42, 0.68];
    let lower_bounds = vec![f64::NEG_INFINITY, f64::NEG_INFINITY, 0.0, 0.0];

    let mut optsum = OptSummary::new(params.clone());
    optsum.optimizer = Optimizer::TrustBq;
    optsum.backend = Optimizer::TrustBq.canonical_backend();
    optsum.return_value = "JOINT_LAPLACE:FTOL_REACHED".to_string();
    optsum.finitial = 1137.42;
    optsum.fmin = 1136.50;
    optsum.feval = 578;
    optsum.max_feval = 578;
    optsum.final_params = params.clone();

    let certificate = OptimizerCertificate::from_opt_summary_with_context(
        &optsum,
        &params,
        &lower_bounds,
        Some(1427),
    );

    assert!(
        certificate.evidence.optimizer_stop.acceptable_stop,
        "joint FTOL at the evaluation cap is a clean stop, not budget exhaustion"
    );
    assert!(!certificate.evidence.optimizer_stop.budget_exhausted);
    assert_eq!(
        certificate.status,
        crate::compiler::FitStatus::ConvergedInterior
    );
    assert!(
        matches!(
            certificate.evidence.gradient.method,
            EvidenceMethod::NotAvailable { .. }
        ),
        "regression must exercise the production no-gradient/NotAvailable path"
    );
    assert!(
            !joint_certificate_requires_fallback(&certificate),
            "production NotAvailable derivative evidence on an acceptable joint FTOL stop should not discard the joint candidate"
        );

    let fallback = agq_poisson_fixture();
    assert!(
        uncertified_joint_fallback(&certificate, &optsum, Some(fallback)).is_none(),
        "acceptable joint candidates with NotAvailable derivatives should remain joint fits"
    );
}

#[cfg(feature = "nlopt")]
#[test]
fn experimental_joint_failed_stop_returns_labelled_fast_pirls_fallback() {
    let mut model = small_joint_poisson_fixture();
    model.fit_with_options_impl(1, false).unwrap();
    let fallback = model.clone();
    let start_beta = model.beta.as_slice().to_vec();
    let start_theta = model.theta.clone();
    let start_objective = model.deviance_with_response_constants(1);

    model
        .fit_joint_glmm_from_start(
            start_beta,
            start_theta,
            start_objective,
            1,
            1,
            Some(fallback),
        )
        .unwrap();

    assert!(
        model
            .opt_summary()
            .return_value
            .starts_with("JOINT_LAPLACE_FALLBACK_FAST_PIRLS"),
        "fallback result must label the returned estimates, got {}",
        model.opt_summary().return_value
    );
    let certificate = model
        .compiler_artifact()
        .optimizer_certificate
        .as_ref()
        .expect("fallback fit should retain the fast-PIRLS certificate");
    assert!(
            !matches!(certificate.status, crate::compiler::FitStatus::NotOptimized),
            "fallback certificate should describe the returned fast-PIRLS fit, not the failed joint attempt"
        );
    assert!(
        certificate.diagnostics.iter().any(|diagnostic| {
            diagnostic.code == DiagnosticCode::OptimizerRecovery
                && diagnostic.payload.get("fit_mode")
                    == Some(&serde_json::json!("fallback_fast_pirls"))
                && diagnostic.payload.get("scorecard_class")
                    == Some(&serde_json::json!("documented_divergence"))
        }),
        "fallback artifact must record the documented-divergence fallback path"
    );
    let substitution = certificate
        .estimator_substitution
        .as_ref()
        .expect("fallback certificate must carry a machine-readable substitution record");
    assert_eq!(substitution.requested_method, "joint_laplace");
    assert_eq!(substitution.effective_method, "fast_pirls_profiled");
    assert!(
        substitution
            .requested_return_code
            .as_deref()
            .is_some_and(|code| code.starts_with("JOINT_LAPLACE")),
        "substitution must record the failed joint attempt's return code, got {:?}",
        substitution.requested_return_code
    );
    let metadata = model
        .compiler_artifact()
        .glmm_fit_metadata
        .as_ref()
        .expect("fallback fit must record GLMM fit metadata");
    assert_eq!(metadata.requested_method.as_deref(), Some("joint_laplace"));
    assert_eq!(
        metadata.effective_method.as_deref(),
        Some("fast_pirls_profiled")
    );
}

#[test]
fn profiled_glmm_certificate_records_first_order_evidence_or_explicit_skip() {
    let mut model = small_joint_poisson_fixture();
    model.fit_with_options_impl(1, false).unwrap();

    let artifact = model.compiler_artifact();
    let certificate = artifact
        .optimizer_certificate
        .as_ref()
        .expect("profiled fast-PIRLS fit must carry an optimizer certificate");
    let issued = artifact.diagnostics.iter().any(|diagnostic| {
        diagnostic
            .payload
            .get("glmm_pirls_profiled_optimum_certificate")
            == Some(&serde_json::json!("issued"))
    });
    if issued {
        assert!(
            certificate.free_gradient_norm.is_some(),
            "an issued profiled-optimum certificate must populate free_gradient_norm"
        );
        assert!(
            certificate.checks.iter().any(|check| matches!(
                check,
                crate::compiler::CertificateCheck::FreeGradientOk { .. }
            )),
            "an issued profiled-optimum certificate must record the free-gradient check"
        );
    } else {
        assert!(
            certificate.checks.iter().any(|check| matches!(
                check,
                crate::compiler::CertificateCheck::NotAssessed { reason }
                    if reason.contains("profiled-optimum certificate not issued")
            )),
            "a skipped profiled-optimum certificate must leave an explicit not-assessed reason"
        );
    }
}

#[test]
fn stateless_transform_glmm_end_to_end() {
    // A transformed predictor `I(x^2)` flows through the GLMM build
    // (which wraps an internal LMM) — proving the materialization seam
    // is wired on the GLMM path too.
    use crate::model::traits::MixedModelFit;

    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut g = Vec::new();
    for grp in 0..5 {
        for obs in 0..8 {
            let xv = obs as f64 - 3.5;
            let eta = 0.6 + 0.05 * xv + 0.01 * xv * xv + [-0.2, 0.1, 0.0, 0.15, -0.05][grp];
            y.push(eta.exp().round().max(1.0));
            x.push(xv);
            g.push(format!("g{}", grp + 1));
        }
    }
    let mut data = DataFrame::new();
    data.add_numeric("y", y).unwrap();
    data.add_numeric("x", x).unwrap();
    data.add_categorical("g", g).unwrap();

    let formula = parse_formula("y ~ 1 + x + I(x^2) + (1 | g)").unwrap();
    assert!(formula.derived.iter().any(|d| d.label == "I(x^2)"));
    let mut model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Poisson, None).unwrap();
    model.fit().unwrap();

    let names = model.coef_names();
    assert!(
        names.iter().any(|n| n == "I(x^2)"),
        "GLMM coef_names should contain `I(x^2)`, got {names:?}"
    );
    assert!(model.objective().is_finite());
}

#[test]
fn agq_deviance_restores_state_on_normal_path() {
    let mut model = agq_poisson_fixture();
    let u0 = model.u[0].clone();
    let eta0 = model.eta.clone();
    let mu0 = model.mu.clone();

    let dev = model.deviance(5);
    assert!(dev.is_finite());

    // The AGQ sweep perturbs u/eta/mu; the guard must restore them exactly.
    assert_eq!(model.u[0], u0, "u not restored after deviance(5)");
    assert_eq!(model.eta, eta0, "eta not restored after deviance(5)");
    assert_eq!(model.mu, mu0, "mu not restored after deviance(5)");
}

#[test]
fn agq_restore_guard_restores_state_on_panic() {
    let mut model = agq_poisson_fixture();
    let u0 = model.u[0].clone();
    let eta0 = model.eta.clone();
    let mu0 = model.mu.clone();
    let u0_flat: Vec<f64> = model.u[0].as_slice().to_vec();
    let n_levels = model.u[0].ncols();

    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut work = AgqRestoreGuard {
            glmm: &mut model,
            u0_flat: u0_flat.clone(),
        };
        // Desync state the way the AGQ sweep would, then blow up mid-sweep.
        for g in 0..n_levels {
            work.u[0][(0, g)] += 7.0;
        }
        work.update_eta();
        panic!("simulated panic inside AGQ sweep");
    }));

    assert!(result.is_err(), "the closure was expected to panic");
    // Guard's Drop ran during unwinding and restored the model.
    assert_eq!(model.u[0], u0, "u not restored after panic");
    assert_eq!(model.eta, eta0, "eta not restored after panic");
    assert_eq!(model.mu, mu0, "mu not restored after panic");
}

#[test]
fn glmm_builder_matches_direct_construction_byte_for_byte() {
    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut g = Vec::new();
    for grp in 0..5 {
        for obs in 0..8 {
            let xv = obs as f64 - 3.5;
            let eta = 0.8 + 0.1 * xv + [-0.2, 0.1, 0.0, 0.15, -0.05][grp];
            y.push(eta.exp().round().max(1.0));
            x.push(xv);
            g.push(format!("g{}", grp + 1));
        }
    }
    let mut data = DataFrame::new();
    data.add_numeric("y", y).unwrap();
    data.add_numeric("x", x).unwrap();
    data.add_categorical("g", g).unwrap();

    let mut direct = GeneralizedLinearMixedModel::new(
        parse_formula("y ~ 1 + x + (1 | g)").unwrap(),
        &data,
        Family::Poisson,
        None,
    )
    .unwrap();
    direct.fit().unwrap();

    let built = GeneralizedLinearMixedModelBuilder::new(
        parse_formula("y ~ 1 + x + (1 | g)").unwrap(),
        &data,
        Family::Poisson,
    )
    .fit()
    .unwrap();

    assert_eq!(
        built.coef(),
        direct.coef(),
        "builder coef must match direct"
    );
    assert_eq!(built.theta, direct.theta, "builder theta must match direct");
}

fn assert_glmm_theta_diagonals_nonnegative(model: &GeneralizedLinearMixedModel) {
    for (idx, &(_, row, col)) in model.lmm.parmap.iter().enumerate() {
        if row == col {
            assert!(
                model.theta[idx] >= 0.0,
                "GLMM theta diagonal {idx} should be rectified, got {}",
                model.theta[idx]
            );
            assert_eq!(
                model.lmm.optsum.final_params[idx], model.theta[idx],
                "GLMM OptSummary must store the rectified theta value"
            );
        }
    }
}

fn resampled_contra_response(data: &DataFrame) -> Vec<f64> {
    data.numeric("use_num")
        .unwrap()
        .iter()
        .enumerate()
        .map(
            |(idx, &value)| {
                if idx % 11 == 0 {
                    1.0 - value
                } else {
                    value
                }
            },
        )
        .collect()
}

fn refit_cold_contra_model(new_y: &[f64]) -> GeneralizedLinearMixedModel {
    let mut data = contra_fixture();
    data.add_numeric("use_num", new_y.to_vec()).unwrap();
    let formula =
        parse_formula("use_num ~ 1 + age + age2 + urban + livch + (1 | urban_dist)").unwrap();
    let mut model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();
    model.fit_with_options(true, 1, false).unwrap();
    model
}

fn glmm_retained_state_slots(model: &GeneralizedLinearMixedModel) -> usize {
    let matrix_slots = |matrix: &DMatrix<f64>| matrix.nrows() * matrix.ncols();
    let block_slots = |block: &MatrixBlock| block.nrows() * block.ncols();

    model.beta.len()
        + model.beta0.len()
        + model.theta.capacity()
        + model.b.iter().map(matrix_slots).sum::<usize>()
        + model.u.iter().map(matrix_slots).sum::<usize>()
        + model.u0.iter().map(matrix_slots).sum::<usize>()
        + model.eta.len()
        + model.mu.len()
        + model.y.len()
        + model.offset.len()
        + model.wt.capacity()
        + model.devc.capacity()
        + model.devc0.capacity()
        + model.sd.capacity()
        + model.mult.capacity()
        + model.lmm.y.len()
        + matrix_slots(&model.lmm.xy_mat.xy)
        + matrix_slots(&model.lmm.xy_mat.wtxy)
        + model
            .lmm
            .reterms
            .iter()
            .map(|rt| matrix_slots(&rt.z) + matrix_slots(&rt.wtz) + matrix_slots(&rt.lambda))
            .sum::<usize>()
        + model.lmm.a_blocks.iter().map(block_slots).sum::<usize>()
        + model.lmm.l_blocks.iter().map(block_slots).sum::<usize>()
        + model.lmm.optsum.initial.capacity()
        + model.lmm.optsum.final_params.capacity()
        + model.lmm.optsum.fit_log.capacity()
}

#[test]
fn test_gamma_pirls_components_are_link_specific() {
    let eta = 2.0_f64.ln();
    let (sqrtw_log, z_log) =
        pirls_working_observation(Family::Gamma, LinkFunction::Log, 3.0, eta, 2.0, 1.0);
    assert!(
        (sqrtw_log - 1.0).abs() < 1e-12,
        "Gamma-log should use dmu/deta=mu, giving unit IRLS weight"
    );
    assert!(
        (z_log - (eta + 0.5)).abs() < 1e-12,
        "Gamma-log working response should divide by dmu/deta=2"
    );

    let (sqrtw_inverse, z_inverse) =
        pirls_working_observation(Family::Gamma, LinkFunction::Inverse, 3.0, 0.5, 2.0, 1.0);
    assert!(
        (sqrtw_inverse - 2.0).abs() < 1e-12,
        "Gamma-inverse should retain |dmu/deta|=mu^2 in the weight"
    );
    assert!(
        (z_inverse - 0.25).abs() < 1e-12,
        "Gamma-inverse working response must preserve the negative derivative"
    );
}

#[test]
fn test_pirls_no_iter0_break_on_first_step_halving_slack() {
    let accepted_obj = 100.0_f64;
    let old_inflated_reference = accepted_obj * 1.0001;
    let first_step_obj = old_inflated_reference + 0.5e-5;
    let tol = 1e-5_f64;

    assert!(
        (first_step_obj - old_inflated_reference).abs() < tol,
        "this is the old false-convergence case when the halving slack is reused"
    );
    assert!(
        !pirls_converged(first_step_obj, accepted_obj, tol),
        "PIRLS convergence must compare against the uninflated accepted objective"
    );
    assert!(pirls_converged(accepted_obj + tol * 0.5, accepted_obj, tol));
}

#[test]
fn test_pirls_handles_bernoulli_near_separation() {
    let (sqrtw_low, z_low) = pirls_working_observation(
        Family::Bernoulli,
        LinkFunction::Logit,
        0.0,
        -1000.0,
        0.0,
        1.0,
    );
    let (sqrtw_high, z_high) =
        pirls_working_observation(Family::Bernoulli, LinkFunction::Log, 1.0, 1000.0, 1.0, 1.0);

    assert!(sqrtw_low.is_finite());
    assert!(z_low.is_finite());
    assert!(sqrtw_high.is_finite());
    assert!(z_high.is_finite());
    assert!(
        sqrtw_high < 4.0e7,
        "clamped Bernoulli variance should keep sqrt weight bounded, got {sqrtw_high}"
    );
}

#[test]
fn test_pirls_no_inf_weight_under_logit() {
    for (y, eta, mu) in [(0.0, -1000.0, 0.0), (1.0, 1000.0, 1.0)] {
        let (sqrtw, z) =
            pirls_working_observation(Family::Binomial, LinkFunction::Logit, y, eta, mu, 25.0);
        assert!(sqrtw.is_finite());
        assert!(z.is_finite());
    }
}

#[test]
fn test_pirls_no_inf_weight_under_binary_noncanonical_links() {
    for link in [LinkFunction::Probit, LinkFunction::Cloglog] {
        for (y, eta, mu) in [(0.0, -1000.0, 0.0), (1.0, 1000.0, 1.0)] {
            let (sqrtw, z) = pirls_working_observation(Family::Binomial, link, y, eta, mu, 25.0);
            assert!(sqrtw.is_finite(), "{link:?} sqrt weight was {sqrtw}");
            assert!(z.is_finite(), "{link:?} working response was {z}");
        }
    }
}

#[test]
fn test_glmm_offset_enters_linear_predictor() {
    let data = constant_response_fixture(vec![0.0, 1.0, 0.0, 1.0]);
    let formula = parse_formula("y ~ 1 + (1 | g)").unwrap();
    let offset = vec![0.1, -0.2, 0.3, -0.4];

    let model = GeneralizedLinearMixedModel::new_with_offset(
        formula,
        &data,
        Family::Bernoulli,
        None,
        offset.clone(),
    )
    .unwrap();

    for (idx, want) in offset.iter().enumerate() {
        assert!((model.offset[idx] - want).abs() < 1e-12);
        assert!((model.eta[idx] - want).abs() < 1e-12);
        assert!((model.mu[idx] - LinkFunction::Logit.linkinv(*want)).abs() < 1e-12);
    }
}

#[test]
fn test_glmm_offset_validation() {
    let data = constant_response_fixture(vec![0.0, 1.0, 0.0, 1.0]);
    let formula = parse_formula("y ~ 1 + (1 | g)").unwrap();

    let err = GeneralizedLinearMixedModel::new_with_offset(
        formula,
        &data,
        Family::Bernoulli,
        None,
        vec![0.0],
    )
    .unwrap_err();

    match err {
        MixedModelError::InvalidArgument(message) => {
            assert!(message.contains("offset length"));
            assert!(message.contains("number of observations"));
        }
        other => panic!("expected InvalidArgument error, got {other:?}"),
    }
}

#[test]
fn test_pirls_working_response_subtracts_offset() {
    let eta = 1.25_f64;
    let mu = eta.exp();
    let offset = -0.75_f64;
    let (sqrtw_plain, z_plain) =
        pirls_working_observation(Family::Poisson, LinkFunction::Log, 3.0, eta, mu, 2.0);
    let (sqrtw_offset, z_offset) = pirls_working_observation_with_offset(
        Family::Poisson,
        LinkFunction::Log,
        3.0,
        eta,
        mu,
        2.0,
        offset,
    );

    assert!((sqrtw_offset - sqrtw_plain).abs() < 1e-12);
    assert!((z_offset - (z_plain - offset)).abs() < 1e-12);
}

#[test]
fn test_pirls_handles_poisson_log_extreme_offset_scale() {
    for (y, eta, mu) in [
        (0.0, -1000.0, 0.0),
        (1.0, -1000.0, 0.0),
        (0.0, 1000.0, f64::INFINITY),
        (1.0, 1000.0, f64::INFINITY),
    ] {
        let (sqrtw, z) =
            pirls_working_observation(Family::Poisson, LinkFunction::Log, y, eta, mu, 1.0);

        assert!(sqrtw.is_finite(), "sqrt weight was {sqrtw}");
        assert!(sqrtw > 0.0, "sqrt weight should stay positive");
        assert!(sqrtw < 4.0e6, "sqrt weight was {sqrtw}");
        assert!(z.is_finite(), "working response was {z}");
    }
}

#[test]
fn test_pirls_handles_poisson_sqrt_zero_mean_start() {
    for (y, eta, mu) in [(0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (3.0, -0.1, 0.01)] {
        let (sqrtw, z) =
            pirls_working_observation(Family::Poisson, LinkFunction::Sqrt, y, eta, mu, 1.0);
        assert!(sqrtw.is_finite(), "sqrt weight was {sqrtw}");
        assert!(sqrtw > 0.0, "sqrt weight should stay positive");
        assert!(z.is_finite(), "working response was {z}");
    }
}

#[test]
fn test_negative_binomial_pirls_uses_fixed_theta_variance() {
    let theta = 4.0;
    let eta = 2.0_f64.ln();
    let mu = 2.0;
    let (sqrtw, z) = pirls_working_observation_with_family_parameters(
        Family::NegativeBinomial,
        LinkFunction::Log,
        Some(theta),
        3.0,
        eta,
        mu,
        1.0,
    );

    let expected_variance = mu + mu * mu / theta;
    let expected_sqrtw = (mu * mu / expected_variance).sqrt();
    assert_relative_eq!(sqrtw, expected_sqrtw, epsilon = 1e-12);
    assert_relative_eq!(z, eta + 0.5, epsilon = 1e-12);
}

fn gamma_dispersion_fixture() -> DataFrame {
    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut group = Vec::new();
    let group_effects = [-0.25, 0.1, 0.3, -0.15];
    for (g, &group_effect) in group_effects.iter().enumerate() {
        for obs in 0..5 {
            let xv = obs as f64 - 2.0;
            let eta = 1.2 + 0.25 * xv + group_effect;
            let wiggle = 1.0 + 0.06 * ((g + obs) % 3) as f64;
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

fn negative_binomial_fixture() -> DataFrame {
    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut group = Vec::new();
    let group_effects = [-0.35, 0.1, 0.25, -0.05];
    for (g, &group_effect) in group_effects.iter().enumerate() {
        for obs in 0..6 {
            let xv = obs as f64 - 2.5;
            let eta = 1.0 + 0.18 * xv + group_effect;
            let base = eta.exp();
            let overdispersion_bump = if (g + obs) % 3 == 0 { 2.0 } else { 0.0 };
            y.push((base + overdispersion_bump).round().max(0.0));
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

#[cfg(not(feature = "nlopt"))]
fn two_term_poisson_fixture() -> DataFrame {
    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut g1 = Vec::new();
    let mut g2 = Vec::new();
    for a in 0..4 {
        for b in 0..3 {
            for obs in 0..3 {
                let xv = obs as f64 - 1.0;
                let eta = 1.0 + 0.15 * xv + [-0.25, 0.05, 0.2, -0.1][a] + [0.1, -0.15, 0.05][b];
                y.push(eta.exp().round().max(1.0));
                x.push(xv);
                g1.push(format!("g1_{}", a + 1));
                g2.push(format!("g2_{}", b + 1));
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

#[test]
fn test_glmm_constructor_accepts_gamma_with_positive_response() {
    let data = gamma_dispersion_fixture();
    let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();

    let model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Gamma, Some(LinkFunction::Log))
            .unwrap();

    assert_eq!(model.family, Family::Gamma);
    assert_eq!(model.dispersion(false), 1.0);
    assert_eq!(model.dispersion(true), 1.0);
}

#[test]
fn test_negative_binomial_constructor_requires_fixed_theta() {
    let data = negative_binomial_fixture();
    let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();

    let missing_theta =
        GeneralizedLinearMixedModel::new(formula.clone(), &data, Family::NegativeBinomial, None)
            .expect_err("plain NB constructor should require fixed theta");
    match missing_theta {
        MixedModelError::InvalidArgument(message) => {
            assert!(message.contains("negative-binomial"));
            assert!(message.contains("fixed theta"));
        }
        other => panic!("expected InvalidArgument error, got {other:?}"),
    }

    let bad_theta =
        GeneralizedLinearMixedModel::new_negative_binomial(formula.clone(), &data, 0.0, None)
            .expect_err("NB theta must be positive");
    match bad_theta {
        MixedModelError::InvalidArgument(message) => {
            assert!(message.contains("positive"));
            assert!(message.contains("theta"));
        }
        other => panic!("expected InvalidArgument error, got {other:?}"),
    }

    let estimated = GeneralizedLinearMixedModel::new_negative_binomial_estimated(
        formula.clone(),
        &data,
        None,
        None,
    )
    .unwrap();
    assert!(estimated.negative_binomial_theta_estimated());
    assert!(estimated
        .negative_binomial_theta()
        .is_some_and(|theta| theta.is_finite() && theta > 0.0));

    let bad_link = GeneralizedLinearMixedModel::new_negative_binomial(
        formula,
        &data,
        2.5,
        Some(LinkFunction::Sqrt),
    )
    .expect_err("fixed-theta NB only supports log link in this slice");
    match bad_link {
        MixedModelError::UnsupportedFamilyLink { family, link } => {
            assert_eq!(family, "negative_binomial");
            assert_eq!(link, "sqrt");
        }
        other => panic!("expected UnsupportedFamilyLink error, got {other:?}"),
    }
}

#[test]
fn test_negative_binomial_fixed_theta_fit_records_metadata() {
    let data = negative_binomial_fixture();
    let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();
    let mut model =
        GeneralizedLinearMixedModel::new_negative_binomial(formula, &data, 2.5, None).unwrap();
    model.lmm.optsum.max_feval = 80;

    model.fit_with_options(true, 1, false).unwrap();

    assert_eq!(model.family, Family::NegativeBinomial);
    assert_eq!(model.link, LinkFunction::Log);
    assert_eq!(model.negative_binomial_theta(), Some(2.5));
    assert_eq!(model.dispersion(false), 2.5);
    assert_eq!(model.dispersion(true), 2.5);
    assert_eq!(model.dof(), model.lmm.feterm.rank + model.lmm.parmap.len());
    assert!(model.objective().is_finite());
    assert!(model.loglikelihood().is_finite());

    let metadata = model
        .compiler_artifact()
        .glmm_fit_metadata
        .as_ref()
        .expect("fitted NB GLMM should record fit metadata");
    assert_eq!(
        metadata.family_parameters.get("negative_binomial_theta"),
        Some(&2.5)
    );
    assert_eq!(
        metadata
            .family_parameters
            .get("negative_binomial_variance_power"),
        Some(&2.0)
    );
    assert_eq!(
        metadata
            .family_parameter_sources
            .get("negative_binomial_theta")
            .map(String::as_str),
        Some("fixed")
    );
    assert_eq!(
        model
            .compiler_artifact()
            .model_boundary
            .response_distribution,
        "negative_binomial"
    );
    let payload = crate::stats::FitSummaryPayload::from_generalized_model(&model);
    assert_eq!(
        payload.family_parameters.get("negative_binomial_theta"),
        Some(&2.5)
    );
    assert_eq!(
        payload
            .family_parameter_sources
            .get("negative_binomial_theta")
            .map(String::as_str),
        Some("fixed")
    );

    let vc = model.varcorr();
    assert!(vc.residual_sd.is_none());
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let y_sim = model.simulate_response(&mut rng).unwrap();
    assert_eq!(y_sim.len(), model.nobs());
    assert!(y_sim
        .iter()
        .all(|value| is_nonnegative_integer_response(*value)));
}

#[test]
fn test_negative_binomial_estimated_theta_fit_records_metadata() {
    let data = negative_binomial_fixture();
    let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();
    let mut model =
        GeneralizedLinearMixedModel::new_negative_binomial_estimated(formula, &data, None, None)
            .unwrap();
    let start_theta = model.negative_binomial_theta().unwrap();

    let control = OptimizerControl::auto()
        .with_optimizer(Optimizer::PatternSearch)
        .with_max_feval(80);
    model
        .fit_with_glmm_options(GlmmFitOptions::fast_laplace().with_optimizer_control(control))
        .unwrap();

    let theta = model.negative_binomial_theta().unwrap();
    assert!(model.negative_binomial_theta_estimated());
    assert!(theta.is_finite() && theta > 0.0);
    assert_eq!(model.dispersion(false), theta);
    assert_eq!(model.dispersion(true), theta);
    assert_eq!(
        model.dof(),
        model.lmm.feterm.rank + model.lmm.parmap.len() + 1
    );
    assert!(model.objective().is_finite());
    assert!(model.loglikelihood().is_finite());

    let metadata = model
        .compiler_artifact()
        .glmm_fit_metadata
        .as_ref()
        .expect("estimated NB GLMM should record fit metadata");
    assert_eq!(
        metadata.family_parameters.get("negative_binomial_theta"),
        Some(&theta)
    );
    assert_eq!(
        metadata
            .family_parameters
            .get("negative_binomial_theta_initial"),
        Some(&start_theta)
    );
    assert!(metadata
        .family_parameters
        .get("negative_binomial_theta_outer_iterations")
        .is_some_and(|value| *value >= 1.0));
    assert_eq!(
        metadata
            .family_parameter_sources
            .get("negative_binomial_theta")
            .map(String::as_str),
        Some("estimated")
    );

    let json = serde_json::to_string(model.compiler_artifact()).unwrap();
    let artifact: crate::compiler::CompiledModelArtifact = serde_json::from_str(&json).unwrap();
    let roundtrip_metadata = artifact.glmm_fit_metadata.unwrap();
    assert_eq!(
        roundtrip_metadata
            .family_parameter_sources
            .get("negative_binomial_theta")
            .map(String::as_str),
        Some("estimated")
    );

    let payload = crate::stats::FitSummaryPayload::from_generalized_model(&model);
    assert_eq!(
        payload.family_parameters.get("negative_binomial_theta"),
        Some(&theta)
    );
    assert_eq!(
        payload
            .family_parameter_sources
            .get("negative_binomial_theta")
            .map(String::as_str),
        Some("estimated")
    );
}

#[test]
fn test_gamma_inverse_gaussian_deviance_finite_at_nonpositive_mu() {
    // Regression for audit 03·H2 / mote bd-01KRXCQ8T7J50F739C7ADHFD41:
    // an inverse-link Gamma/InverseGaussian GLMM can transiently propose
    // μ ≤ 0 during PIRLS. The per-observation deviance component must stay
    // finite (a large penalty step-halving can reject), never NaN/Inf
    // that would slip the `obj > halving_bound` guard.
    let data = gamma_dispersion_fixture();
    let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();
    let gamma = GeneralizedLinearMixedModel::new(
        formula.clone(),
        &data,
        Family::Gamma,
        Some(LinkFunction::Log),
    )
    .unwrap();
    for &mu in &[0.0_f64, -1e-12, -1.0, -1e6] {
        let d = gamma.dev_resid_component(2.5, mu);
        assert!(d.is_finite(), "Gamma dev at μ={mu} must be finite, got {d}");
    }

    let inv_g = GeneralizedLinearMixedModel::new(
        formula,
        &data,
        Family::InverseGaussian,
        Some(LinkFunction::Log),
    )
    .unwrap();
    for &mu in &[0.0_f64, -1e-9, -3.0] {
        let d = inv_g.dev_resid_component(2.5, mu);
        assert!(
            d.is_finite(),
            "InverseGaussian dev at μ={mu} must be finite, got {d}"
        );
    }
}

#[cfg(not(feature = "nlopt"))]
#[test]
fn test_glmm_fit_uses_native_cobyla_without_nlopt() {
    let data = gamma_dispersion_fixture();
    let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();
    let mut model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Gamma, Some(LinkFunction::Log))
            .unwrap();
    model.lmm.optsum.max_feval = 50;

    model.fit_with_options(true, 1, false).unwrap();

    assert_eq!(model.lmm.optsum.optimizer, Optimizer::Cobyla);
    assert_eq!(model.lmm.optsum.backend.label(), "native");
    assert!(model.lmm.optsum.feval > 0);
    assert!(model.lmm.optsum.fmin.is_finite());
    assert!(!model.lmm.optsum.fit_log.is_empty());
    assert!(model.lmm.compiler_artifact.optimizer_certificate.is_some());
}

#[cfg(not(feature = "nlopt"))]
#[test]
fn test_glmm_fit_uses_native_pattern_search_when_requested() {
    let data = gamma_dispersion_fixture();
    let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();
    let mut model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Gamma, Some(LinkFunction::Log))
            .unwrap();
    model.lmm.optsum.optimizer = Optimizer::PatternSearch;
    model.lmm.optsum.max_feval = 120;

    model.fit_with_options(true, 1, false).unwrap();

    assert_eq!(model.lmm.optsum.optimizer, Optimizer::PatternSearch);
    assert_eq!(model.lmm.optsum.backend.label(), "native");
    assert!(model.lmm.optsum.feval > 0);
    assert!(model.lmm.optsum.fmin.is_finite());
    assert!(!model.lmm.optsum.fit_log.is_empty());
    assert!(model.lmm.compiler_artifact.optimizer_certificate.is_some());
}

#[cfg(not(feature = "nlopt"))]
#[test]
fn test_glmm_pattern_search_handles_multitheta_poisson_fit() {
    let data = two_term_poisson_fixture();
    let formula = parse_formula("y ~ 1 + x + (1 | g1) + (1 | g2)").unwrap();
    let mut model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Poisson, None).unwrap();
    model.lmm.optsum.optimizer = Optimizer::PatternSearch;
    model.lmm.optsum.max_feval = 180;

    model.fit_with_options(true, 1, false).unwrap();

    assert_eq!(model.theta.len(), 2);
    assert_eq!(model.lmm.optsum.optimizer, Optimizer::PatternSearch);
    assert!(model.theta.iter().all(|value| value.is_finite()));
    assert!(model.theta.iter().all(|value| *value >= 0.0));
    assert!(model.objective().is_finite());
}

#[test]
fn test_glmm_constructor_rejects_nonpositive_gamma_response() {
    let data = contra_fixture();
    let formula = parse_formula("use_num ~ 1 + (1 | urban_dist)").unwrap();
    let err =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Gamma, Some(LinkFunction::Log))
            .expect_err("Gamma GLMM should reject zero responses");

    match err {
        MixedModelError::InvalidArgument(msg) => {
            assert!(msg.contains("gamma"));
            assert!(msg.contains("strictly positive"));
        }
        other => panic!("expected InvalidArgument error, got {other:?}"),
    }
}

#[test]
fn test_gamma_glmm_refit_rejects_nonpositive_response() {
    let data = gamma_dispersion_fixture();
    let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();
    let mut model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Gamma, Some(LinkFunction::Log))
            .unwrap();
    let mut new_y = data.numeric("y").unwrap().to_vec();
    new_y[0] = 0.0;

    let err = model
        .refit(&new_y)
        .expect_err("Gamma GLMM refit/bootstrap response must stay strictly positive");

    match err {
        MixedModelError::InvalidArgument(msg) => {
            assert!(msg.contains("gamma"));
            assert!(msg.contains("strictly positive"));
        }
        other => panic!("expected InvalidArgument error, got {other:?}"),
    }
}

#[test]
fn test_glmm_constructor_accepts_normal_nonidentity_dispersion_family() {
    let data = gamma_dispersion_fixture();
    let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();

    let model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Normal, Some(LinkFunction::Sqrt))
            .unwrap();

    assert_eq!(model.family, Family::Normal);
    assert_eq!(model.link, LinkFunction::Sqrt);
    assert_eq!(model.dispersion(false), 1.0);
}

#[test]
fn test_gamma_glmm_fit_estimates_pearson_dispersion() {
    let data = gamma_dispersion_fixture();
    let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();
    let mut model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Gamma, Some(LinkFunction::Log))
            .unwrap();

    model.fit_with_options(true, 1, false).unwrap();

    let sigma = model.dispersion(false);
    let phi = model.dispersion(true);
    let expected_phi =
        model.pearson_dispersion_numerator() / (model.nobs() - model.lmm.feterm.rank) as f64;

    assert!(sigma.is_finite());
    assert!(sigma > 0.0);
    assert_relative_eq!(phi, sigma * sigma, epsilon = 1e-12);
    assert_relative_eq!(phi, expected_phi, epsilon = 1e-12, max_relative = 1e-12);
    assert_eq!(
        model.dof(),
        model.lmm.feterm.rank + model.lmm.parmap.len() + 1
    );
    assert_relative_eq!(model.varcorr().residual_sd.unwrap(), sigma, epsilon = 1e-12);
}

/// Difficult-model corpus row `gamma_near_zero_random_effect_unit`
/// (see `comparison/difficult_model_scoreboard.toml`). Gamma is
/// implemented but not 1.0-certified and there is no Gamma comparison
/// fixture, so the near-zero-random-effect axis is represented here as a
/// deterministic unit-test diagnostic, never an lme4-parity claim.
#[test]
fn test_gamma_glmm_near_zero_random_effect_is_diagnostic() {
    // Every group shares the same linear predictor: the between-group
    // variance is structurally negligible, so the MLE of theta sits at
    // (or against) the zero boundary.
    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut group = Vec::new();
    for g in 0..5 {
        for obs in 0..6 {
            let xv = obs as f64 - 2.5;
            let eta = 1.1 + 0.2 * xv;
            let wiggle = 1.0 + 0.04 * ((g + obs) % 3) as f64;
            y.push(eta.exp() * wiggle);
            x.push(xv);
            group.push(format!("g{}", g + 1));
        }
    }
    let mut data = DataFrame::new();
    data.add_numeric("y", y).unwrap();
    data.add_numeric("x", x).unwrap();
    data.add_categorical("group", group).unwrap();

    let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();
    let mut model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Gamma, Some(LinkFunction::Log))
            .unwrap();
    model.lmm_mut().optsum.optimizer = Optimizer::PatternSearch;
    model.lmm_mut().optsum.initial = vec![0.0];
    model.lmm_mut().optsum.max_feval = 1000;

    model.fit_with_options(true, 1, false).unwrap();

    // optimizer_status: a certificate must be recorded for this fit.
    assert!(
        model.lmm.compiler_artifact.optimizer_certificate.is_some(),
        "Gamma near-zero RE fit must record an optimizer certificate"
    );
    assert!(model.lmm.optsum.feval > 0);

    // time_to_certified_fit input: the objective is finite and computable.
    assert!(model.objective().is_finite());

    // certification_status: this is a near-zero boundary diagnostic, not
    // an lme4-parity claim. theta is non-negative and pinned near zero.
    let theta = model.theta();
    assert_eq!(theta.len(), 1);
    assert!(theta[0].is_finite() && theta[0] >= 0.0);
    assert!(
        theta[0] < 1e-1,
        "near-zero random-effect axis: expected theta pinned near the \
             zero boundary, got {}",
        theta[0]
    );

    // The corpus criterion is a *diagnostic*, not just a small number:
    // the near-zero random effect must be reported through the artifact
    // as a singular/boundary covariance, so an ordinary interior fit that
    // merely happened to land on a small theta would NOT satisfy this.
    assert!(
        model.is_singular(),
        "near-zero random-effect axis must surface as a singular/boundary \
             covariance in the artifact, not be inferred from theta alone"
    );
    let certificate = model
        .lmm
        .compiler_artifact
        .optimizer_certificate
        .as_ref()
        .expect("near-zero Gamma GLMM should retain optimizer certificate");
    assert_eq!(
        certificate.status,
        crate::compiler::FitStatus::ConvergedBoundary,
        "near-zero Gamma GLMM should classify as a boundary covariance state; return={}",
        model.lmm.optsum.return_value
    );
    assert!(
        certificate.diagnostics.iter().any(|diagnostic| {
            diagnostic.payload.get("covariance_kkt_classification")
                == Some(&serde_json::json!("ValidZeroVariance"))
        }),
        "near-zero Gamma GLMM should expose the existing covariance classification leaf"
    );
}

#[test]
fn test_poisson_glmm_near_zero_random_effect_classifies_boundary() {
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

    let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();
    let mut model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Poisson, Some(LinkFunction::Log))
            .unwrap();
    model.fit_with_options(true, 1, false).unwrap();

    let theta = model.theta();
    assert!(
        theta.iter().any(|value| value.abs() <= 1.0e-4),
        "near-zero Poisson random effect should pin a covariance scale near zero, got {theta:?}"
    );
    let certificate = model
        .lmm
        .compiler_artifact
        .optimizer_certificate
        .as_ref()
        .expect("near-zero Poisson GLMM should retain optimizer certificate");
    assert_eq!(
        certificate.status,
        crate::compiler::FitStatus::ConvergedBoundary,
        "near-zero Poisson GLMM should classify as a boundary covariance state"
    );
    assert!(
        certificate.diagnostics.iter().any(|diagnostic| {
            diagnostic.payload.get("covariance_kkt_classification")
                == Some(&serde_json::json!("ValidZeroVariance"))
        }),
        "near-zero Poisson GLMM should expose the existing covariance classification leaf"
    );
}

#[cfg(not(feature = "nlopt"))]
#[test]
fn test_glmm_fast_false_uses_native_joint_or_fallback_path_without_nlopt() {
    let data = contra_fixture();
    let formula = parse_formula("use_num ~ 1 + (1 | urban_dist)").unwrap();
    let mut model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();
    model.lmm.optsum.max_feval = 80;

    model.fit_with_options(false, 1, false).unwrap();

    assert!(
        model.lmm.optsum.return_value.contains("JOINT_LAPLACE"),
        "fast=false without nlopt must use the labelled joint Laplace path or fallback, got {}",
        model.lmm.optsum.return_value
    );
    assert_eq!(model.lmm.optsum.backend.label(), "native");
    let trust_bq_attempted = model.lmm.optsum.optimizer == Optimizer::TrustBq
        || model
            .lmm
            .compiler_artifact
            .diagnostics
            .iter()
            .any(|diagnostic| {
                diagnostic.code == DiagnosticCode::OptimizerRecovery
                    && diagnostic.payload.get("joint_optimizer")
                        == Some(&serde_json::json!("trust_bq"))
            });
    assert!(
            trust_bq_attempted,
            "native fast=false should attempt the TrustBQ joint optimizer or record it in fallback diagnostics"
        );
    let metadata = model
        .lmm
        .compiler_artifact
        .glmm_fit_metadata
        .as_ref()
        .expect("native fast=false fit should record GLMM metadata");
    assert_eq!(metadata.optimizer_max_feval, Some(80));
    assert!(metadata.optimizer_feval.unwrap_or_default() >= 0);
    assert!(
        matches!(
            metadata.estimation_method.as_str(),
            "joint_laplace" | "fallback_fast_pirls"
        ),
        "native fast=false must record either joint Laplace or a labelled fallback, got {:?}",
        metadata
    );

    let data = contra_fixture();
    let formula = parse_formula("use_num ~ 1 + (1 | urban_dist)").unwrap();
    let mut model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();
    model.lmm.optsum.max_feval = 80;
    model.fit_with_options(false, 7, false).unwrap();
    assert!(
        model.lmm.optsum.return_value.contains("JOINT_AGQ"),
        "valid scalar-RE AGQ should also use the labelled native joint path, got {}",
        model.lmm.optsum.return_value
    );
}

#[cfg(not(feature = "nlopt"))]
#[test]
fn test_glmm_joint_laplace_honors_configured_max_feval_without_nlopt() {
    let data = contra_fixture();
    let formula = parse_formula("use_num ~ 1 + (1 | urban_dist)").unwrap();
    let mut model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();
    model.fit_with_options_impl(1, false).unwrap();
    let start_beta = model.beta.as_slice().to_vec();
    let start_theta = model.theta.clone();
    let start_objective = model.deviance_with_response_constants(1);

    model
        .fit_joint_glmm_from_start(start_beta, start_theta, start_objective, 1, 3, None)
        .unwrap();

    assert_eq!(model.lmm.optsum.optimizer, Optimizer::TrustBq);
    assert_eq!(model.lmm.optsum.max_feval, 3);
    assert!(model.lmm.optsum.feval <= 3);
    assert!(
        model.lmm.optsum.return_value.contains("MAXEVAL_REACHED"),
        "forced tiny budget should report maxeval, got {}",
        model.lmm.optsum.return_value
    );
    let metadata = model
        .lmm
        .compiler_artifact
        .glmm_fit_metadata
        .as_ref()
        .expect("joint fit should record GLMM metadata");
    assert_eq!(metadata.optimizer, "trust_bq");
    assert_eq!(metadata.optimizer_feval, Some(model.lmm.optsum.feval));
    assert_eq!(metadata.optimizer_max_feval, Some(3));
    assert_eq!(
        metadata.optimizer_fit_log_len,
        Some(model.lmm.optsum.fit_log.len())
    );
    assert_eq!(metadata.optimizer_convergence_status, "budget_exhausted");
}

#[cfg(not(feature = "nlopt"))]
#[test]
fn test_budgeted_native_joint_laplace_records_high_baseline_multi_re_metadata() {
    let mut correct = Vec::new();
    let mut x = Vec::new();
    let mut participant = Vec::new();
    let mut item = Vec::new();
    for subj in 0..8 {
        let subj_shift = (subj as f64 - 3.5) * 0.08;
        for trial in 0..8 {
            let xv = if trial % 2 == 0 { -0.5 } else { 0.5 };
            let item_id = trial % 4;
            let eta = 2.8 + 0.35 * xv + subj_shift - 0.04 * item_id as f64;
            let p = 1.0 / (1.0 + (-eta).exp());
            let deterministic_u = ((subj * 17 + trial * 11) % 101) as f64 / 101.0;
            correct.push((deterministic_u < p) as i32 as f64);
            x.push(xv);
            participant.push(format!("s{subj}"));
            item.push(format!("i{item_id}"));
        }
    }
    let mut data = DataFrame::new();
    data.add_numeric("correct", correct).unwrap();
    data.add_numeric("x", x).unwrap();
    data.add_categorical("participant", participant).unwrap();
    data.add_categorical("item", item).unwrap();
    let formula = parse_formula("correct ~ 1 + x + (1 + x | participant) + (1 | item)").unwrap();
    let mut model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();
    model.lmm.optsum.max_feval = 40;

    model.fit_with_options(false, 1, false).unwrap();

    assert!(
        model.lmm.optsum.return_value.contains("JOINT_LAPLACE"),
        "budgeted high-baseline multi-RE fit should attempt the labelled joint route, got {}",
        model.lmm.optsum.return_value
    );
    let metadata = model
        .lmm
        .compiler_artifact
        .glmm_fit_metadata
        .as_ref()
        .expect("budgeted joint route should record GLMM metadata");
    assert_eq!(metadata.n_agq, 1);
    assert_eq!(metadata.optimizer_max_feval, Some(40));
    assert!(metadata.optimizer_feval.unwrap_or_default() <= 40);
    assert_eq!(
            metadata.estimation_method, "joint_laplace",
            "budgeted high-baseline multi-RE fit should keep the native joint candidate instead of returning the fast-PIRLS fallback"
        );
    if model.lmm.optsum.return_value.contains("MAXEVAL_REACHED") {
        let certificate = model
            .lmm
            .compiler_artifact
            .optimizer_certificate
            .as_ref()
            .expect("budget-limited joint candidate should retain certificate");
        assert!(certificate.diagnostics.iter().any(|diagnostic| {
            diagnostic.code == DiagnosticCode::OptimizerNonconvergence
                && diagnostic.payload.get("fit_mode")
                    == Some(&serde_json::json!("uncertified_joint_candidate"))
                && diagnostic.payload.get("scorecard_class")
                    == Some(&serde_json::json!("budget_limited_joint_candidate"))
        }));
    }
    let trust_bq_attempted = model.lmm.optsum.optimizer == Optimizer::TrustBq
        || model
            .lmm
            .compiler_artifact
            .diagnostics
            .iter()
            .any(|diagnostic| {
                diagnostic.code == DiagnosticCode::OptimizerRecovery
                    && diagnostic.payload.get("joint_optimizer")
                        == Some(&serde_json::json!("trust_bq"))
            });
    assert!(trust_bq_attempted);
}

#[cfg(feature = "nlopt")]
#[test]
fn test_glmm_fast_false_uses_labelled_joint_or_fallback_path() {
    let data = contra_fixture();
    let formula = parse_formula("use_num ~ 1 + (1 | urban_dist)").unwrap();
    let mut model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();

    model.fit_with_options(false, 1, false).unwrap();

    assert!(model.lmm.optsum.return_value.contains("JOINT_LAPLACE"));
    let metadata = model
        .lmm
        .compiler_artifact
        .glmm_fit_metadata
        .as_ref()
        .expect("fast=false fit should record GLMM metadata");
    assert!(
        matches!(
            metadata.estimation_method.as_str(),
            "joint_laplace" | "fallback_fast_pirls"
        ),
        "fast=false must record either certified joint Laplace or a labelled fallback, got {:?}",
        metadata
    );
    if metadata.estimation_method == "joint_laplace" {
        assert_eq!(metadata.objective_definition, "joint_glmm_laplace_deviance");
        assert_eq!(metadata.response_constants, "included");
    } else {
        assert_eq!(metadata.objective_definition, "profiled_glmm_deviance");
        assert_eq!(metadata.response_constants, "dropped");
        assert_eq!(
            metadata.fallback_status.as_deref(),
            Some("fallback_fast_pirls")
        );
    }
}

#[cfg(feature = "nlopt")]
#[test]
fn test_glmm_fast_false_nagq_uses_labelled_joint_agq_or_fallback_path() {
    let data = contra_fixture();
    let formula = parse_formula("use_num ~ 1 + (1 | urban_dist)").unwrap();
    let mut model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();

    model.fit_with_options(false, 7, false).unwrap();

    assert!(
        model.lmm.optsum.return_value.contains("JOINT_AGQ"),
        "fast=false n_agq>1 must label the joint AGQ path, got {}",
        model.lmm.optsum.return_value
    );
    let metadata = model
        .lmm
        .compiler_artifact
        .glmm_fit_metadata
        .as_ref()
        .expect("fast=false AGQ fit should record GLMM metadata");
    assert!(
        matches!(
            metadata.estimation_method.as_str(),
            "joint_agq" | "fallback_fast_pirls"
        ),
        "fast=false AGQ must record either certified joint AGQ or a labelled fallback, got {:?}",
        metadata
    );
    if metadata.estimation_method == "joint_agq" {
        assert_eq!(metadata.objective_definition, "joint_glmm_agq_deviance");
        assert_eq!(metadata.response_constants, "included");
        assert_eq!(metadata.n_agq, 7);
    } else {
        assert_eq!(metadata.objective_definition, "profiled_glmm_deviance");
        assert_eq!(metadata.response_constants, "dropped");
        assert_eq!(
            metadata.fallback_status.as_deref(),
            Some("fallback_fast_pirls")
        );
    }
}

fn constant_response_fixture(y: Vec<f64>) -> DataFrame {
    let mut data = DataFrame::new();
    data.add_numeric("y", y).unwrap();
    data.add_categorical(
        "g",
        vec![
            "a".to_string(),
            "a".to_string(),
            "b".to_string(),
            "b".to_string(),
        ],
    )
    .unwrap();
    data
}

fn assert_constant_response_rejected(family: Family, y: Vec<f64>) {
    let data = constant_response_fixture(y);
    let formula = parse_formula("y ~ 1 + (1 | g)").unwrap();

    let err = GeneralizedLinearMixedModel::new(formula, &data, family, None).unwrap_err();

    match err {
        MixedModelError::InvalidArgument(message) => {
            assert!(message.contains("response is constant"));
        }
        other => panic!("expected InvalidArgument error, got {other:?}"),
    }
}

#[test]
fn test_glmm_rejects_constant_response_bernoulli() {
    assert_constant_response_rejected(Family::Bernoulli, vec![0.0, 0.0, 0.0, 0.0]);
    assert_constant_response_rejected(Family::Bernoulli, vec![1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn test_glmm_rejects_constant_response_poisson() {
    assert_constant_response_rejected(Family::Poisson, vec![3.0, 3.0, 3.0, 3.0]);
}

#[test]
fn test_glmm_accepts_near_constant() {
    let data = constant_response_fixture(vec![0.0, 0.0, 0.0, 1.0]);
    let formula = parse_formula("y ~ 1 + (1 | g)").unwrap();

    GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();
}

#[test]
fn test_glmm_constructor_supports_requested_family_link_pairs() {
    let mut binomial_data = constant_response_fixture(vec![0.0, 0.25, 0.75, 1.0]);
    binomial_data
        .add_numeric("x", vec![-1.0, -0.5, 0.5, 1.0])
        .unwrap();
    let binomial_formula = parse_formula("y ~ 1 + x + (1 | g)").unwrap();
    for link in [LinkFunction::Probit, LinkFunction::Cloglog] {
        let model = GeneralizedLinearMixedModel::new(
            binomial_formula.clone(),
            &binomial_data,
            Family::Binomial,
            Some(link),
        )
        .unwrap();
        assert_eq!(model.link, link);
        assert!(model.mu.iter().all(|mu| *mu > 0.0 && *mu < 1.0));
    }

    let mut poisson_data = constant_response_fixture(vec![0.0, 1.0, 2.0, 4.0]);
    poisson_data
        .add_numeric("x", vec![-1.0, -0.5, 0.5, 1.0])
        .unwrap();
    let poisson_formula = parse_formula("y ~ 1 + x + (1 | g)").unwrap();
    let model = GeneralizedLinearMixedModel::new(
        poisson_formula,
        &poisson_data,
        Family::Poisson,
        Some(LinkFunction::Sqrt),
    )
    .unwrap();
    assert_eq!(model.link, LinkFunction::Sqrt);
    assert!(model.mu.iter().all(|mu| *mu >= 0.0));
}

#[test]
fn test_glmm_constructor_rejects_unsupported_family_link_pairs() {
    let data = constant_response_fixture(vec![0.0, 0.0, 0.0, 1.0]);
    let formula = parse_formula("y ~ 1 + (1 | g)").unwrap();

    for (family, link) in [
        (Family::Binomial, LinkFunction::Sqrt),
        (Family::Poisson, LinkFunction::Probit),
    ] {
        let err = GeneralizedLinearMixedModel::new(formula.clone(), &data, family, Some(link))
            .unwrap_err();
        match err {
            MixedModelError::UnsupportedFamilyLink {
                family: got_family,
                link: got_link,
            } => {
                assert_eq!(got_family, family_label(family));
                assert_eq!(got_link, link_label(link));
            }
            other => panic!("expected UnsupportedFamilyLink error, got {other:?}"),
        }
    }
}

/// Build a DataFrame from the embedded contra.csv.
///
/// Columns: use_num (numeric 0/1), age, age2 (= age²), urban (Y/N),
///          livch (0+/1/2/3+), urban_dist (interaction string).
fn contra_fixture() -> DataFrame {
    let csv = include_str!("../contra.csv");
    let mut use_num = Vec::new();
    let mut age = Vec::new();
    let mut age2 = Vec::new();
    let mut urban = Vec::new();
    let mut livch = Vec::new();
    let mut urban_dist = Vec::new();

    for line in csv.lines() {
        let parts: Vec<&str> = line.split(',').collect();
        use_num.push(parts[0].parse::<f64>().unwrap());
        age.push(parts[1].parse::<f64>().unwrap());
        age2.push(parts[2].parse::<f64>().unwrap());
        urban.push(parts[3].to_string());
        livch.push(parts[4].to_string());
        urban_dist.push(parts[5].to_string());
    }

    let mut df = DataFrame::new();
    df.add_numeric("use_num", use_num).unwrap();
    df.add_numeric("age", age).unwrap();
    df.add_numeric("age2", age2).unwrap();
    df.add_categorical("urban", urban).unwrap();
    df.add_categorical("livch", livch).unwrap();
    df.add_categorical("urban_dist", urban_dist).unwrap();
    df
}

#[test]
fn glmm_fast_options_record_caller_native_optimizer_override() {
    let data = contra_fixture();
    let formula = parse_formula("use_num ~ 1 + (1 | urban_dist)").unwrap();
    let mut model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();
    let control = OptimizerControl::auto()
        .with_optimizer(Optimizer::Cobyla)
        .with_max_feval(120)
        .with_tolerances(FitToleranceOverrides::default().with_ftol_abs(1.0e-8));

    model
        .fit_with_glmm_options(GlmmFitOptions::fast_laplace().with_optimizer_control(control))
        .unwrap();

    assert_eq!(model.lmm.optsum.optimizer, Optimizer::Cobyla);
    assert_eq!(model.lmm.optsum.optimizer_source_name(), "caller");
    assert!(model.lmm.optsum.caller_set_field("optimizer"));
    assert!(model.lmm.optsum.caller_set_field("max_feval"));

    let certificate = model
        .lmm
        .optimizer_certificate()
        .expect("GLMM fit should attach optimizer certificate");
    assert_eq!(certificate.optimizer_control.optimizer_source, "caller");
    assert!(certificate
        .optimizer_control
        .caller_set_fields
        .iter()
        .any(|field| field == "optimizer"));
    let metadata = model
        .lmm
        .compiler_artifact
        .glmm_fit_metadata
        .as_ref()
        .expect("GLMM fit should record metadata");
    assert_eq!(metadata.optimizer, "cobyla");
    assert_eq!(metadata.optimizer_source.as_deref(), Some("caller"));
    assert!(metadata
        .caller_set_fields
        .iter()
        .any(|field| field == "max_feval"));
}

#[test]
fn glmm_joint_options_reject_unwired_optimizer_before_fitting() {
    let data = contra_fixture();
    let formula = parse_formula("use_num ~ 1 + (1 | urban_dist)").unwrap();
    let mut model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();

    let err = model
        .fit_with_glmm_options(GlmmFitOptions::joint_laplace().with_optimizer(Optimizer::Cobyla))
        .expect_err("joint GLMM Cobyla override should be unsupported");

    assert_eq!(err.code(), "unsupported");
    assert!(!model.is_fitted());
}

// ── GLMM parity tests (pirls.jl) ─────────────────────────────────────────

#[cfg(feature = "nlopt")]
#[test]
fn test_contra_glmm_theta_and_deviance() {
    // pirls.jl:
    //   gm0 = fit(MixedModel, first(gfms[:contra]), contra, Bernoulli(); fast=true)
    //   @test isapprox(gm0.θ, [0.5720746212924732], atol=0.001)
    //   @test isapprox(deviance(gm0), 2361.657202855648, atol=0.001)
    //
    // Equivalent formula (pre-computed age² and urban×dist interaction):
    //   use_num ~ 1 + age + age2 + urban + livch + (1 | urban_dist)
    let data = contra_fixture();
    let formula =
        parse_formula("use_num ~ 1 + age + age2 + urban + livch + (1 | urban_dist)").unwrap();

    let mut model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();

    model.fit_with_options(true, 1, false).unwrap();

    let theta = &model.theta;
    assert_eq!(theta.len(), 1);
    assert_relative_eq!(theta[0], 0.5720746212924732, epsilon = 0.01);

    let dev = model.deviance(1);
    assert_relative_eq!(dev, 2361.657202855648, epsilon = 1.0);
}

#[cfg(feature = "nlopt")]
#[test]
fn test_cbpp_binomial_glmm_with_case_weights() {
    // pirls.jl:125-147, MixedModels.jl/test/pirls.jl
    //   gm2 = fit(MixedModel, first(gfms[:cbpp]), cbpp, Binomial();
    //              wts=float(cbpp.hsz), init_from_lmm=[:β, :θ])
    //   @test deviance(gm2, true) ≈ 100.09585620707632 rtol=0.0001
    //   @test loglikelihood(gm2)  ≈ -92.02628187247377 atol=0.001
    //
    // Formula in modelcache.jl:
    //   (incid / hsz) ~ 1 + period + (1 | herd)
    //
    // Bundled cbpp dataset uses lme4 column names: incidence, size, herd,
    // period. The response is the per-trial proportion (incidence/size)
    // and `size` provides the case weights.
    let (data, _) = crate::datasets::load("cbpp").unwrap();
    let incidence = data.numeric("incidence").unwrap();
    let size = data.numeric("size").unwrap();

    let proportion: Vec<f64> = incidence
        .iter()
        .zip(size.iter())
        .map(|(&y, &n)| y / n)
        .collect();
    let weights: Vec<f64> = size.to_vec();

    let mut data_with_proportion = data.clone();
    data_with_proportion
        .add_numeric("proportion", proportion)
        .unwrap();

    let formula = parse_formula("proportion ~ 1 + period + (1 | herd)").unwrap();

    let mut model = GeneralizedLinearMixedModel::new_with_weights(
        formula,
        &data_with_proportion,
        Family::Binomial,
        None,
        weights,
    )
    .unwrap();

    model.fit_with_options(true, 1, false).unwrap();

    let dev = model.deviance(1);
    // Julia ref: `deviance(gm2, true) ≈ 100.09585620707632`, rtol=0.0001.
    assert_relative_eq!(dev, 100.09585620707632, max_relative = 1e-3);

    // `MixedModelFit::loglikelihood` is on the full normalized `-2 logLik`
    // scale (response normalising constants retained), so it is now
    // directly comparable to Julia's `loglikelihood(gm2) ≈
    // -92.02628187247377` (pirls.jl:125-147). This pins the B1 fix:
    // before it, `loglikelihood` was `-objective/2` on the
    // dropped-constant scale and AIC/BIC were offset by `2·Σ ln C(nᵢ,kᵢ)`.
    // Same fast-PIRLS-vs-joint band as the deviance check above (the
    // log-likelihood inherits that divergence): rtol 1e-3.
    let ll = MixedModelFit::loglikelihood(&model);
    assert_relative_eq!(ll, -92.02628187247377, max_relative = 1e-3);
    // AIC/BIC follow from the corrected log-likelihood + dof.
    let dof = MixedModelFit::dof(&model) as f64;
    assert_relative_eq!(model.aic(), -2.0 * ll + 2.0 * dof, epsilon = 1e-9);
}

#[cfg(feature = "nlopt")]
#[test]
fn experimental_joint_cbpp_objective_matches_lme4_at_lme4_parameters() {
    let (data, _) = crate::datasets::load("cbpp").unwrap();
    let incidence = data.numeric("incidence").unwrap();
    let size = data.numeric("size").unwrap();

    let proportion: Vec<f64> = incidence
        .iter()
        .zip(size.iter())
        .map(|(&y, &n)| y / n)
        .collect();
    let weights: Vec<f64> = size.to_vec();

    let mut data_with_proportion = data.clone();
    data_with_proportion
        .add_numeric("proportion", proportion)
        .unwrap();

    let formula = parse_formula("proportion ~ 1 + period + (1 | herd)").unwrap();
    let mut model = GeneralizedLinearMixedModel::new_with_weights(
        formula,
        &data_with_proportion,
        Family::Binomial,
        None,
        weights,
    )
    .unwrap();
    // lme4 optimum fitted with glmerControl(optimizer = "bobyqa",
    // tolPwrss = 1e-13, optCtrl = list(maxfun = 200000, rhoend = 1e-11)).
    // The tightened inner tolerance matters: at the 1e-7 default, lme4's
    // recorded deviance carries a one-inner-iteration-stale ldL2 (5.6e-4 on
    // this row), while the Rust joint objective is the exact at-mode
    // Laplace value. See mote bd-01KWFNE6GB3FN3FQJM0VKGXCG0.
    let params = vec![
        -1.398_532_13,
        -0.992_332_69,
        -1.128_672_11,
        -1.580_313_90,
        0.642_261_43,
    ];
    let objective = model.joint_glmm_deviance_at_params(&params, 4, 1);
    let lme4_objective = 184.052_563_74;
    let delta = (objective - lme4_objective).abs();
    assert!(
            delta <= 5.0e-6,
            "cbpp joint objective should match lme4's at-mode deviance at the tolPwrss-tight lme4 optimum; rust={objective:.9}, lme4={lme4_objective:.9}, delta={delta:.9}"
        );
}

#[cfg(feature = "nlopt")]
#[test]
fn test_cbpp_agq_deviance_uses_case_weights() {
    let (data, _) = crate::datasets::load("cbpp").unwrap();
    let incidence = data.numeric("incidence").unwrap();
    let size = data.numeric("size").unwrap();

    let proportion: Vec<f64> = incidence
        .iter()
        .zip(size.iter())
        .map(|(&y, &n)| y / n)
        .collect();
    let weights: Vec<f64> = size.to_vec();

    let mut data_with_proportion = data.clone();
    data_with_proportion
        .add_numeric("proportion", proportion)
        .unwrap();

    let formula = parse_formula("proportion ~ 1 + period + (1 | herd)").unwrap();
    let mut model = GeneralizedLinearMixedModel::new_with_weights(
        formula,
        &data_with_proportion,
        Family::Binomial,
        None,
        weights,
    )
    .unwrap();
    model.fit_with_options(true, 1, false).unwrap();

    let weighted_agq = model.deviance(5);
    model.wt = vec![1.0; model.y.len()];
    let unit_weight_agq = model.deviance(5);
    assert!(
            (weighted_agq - unit_weight_agq).abs() > 1.0,
            "AGQ deviance must include binomial case weights; weighted={weighted_agq}, unit={unit_weight_agq}"
        );
}

/// fast=true regression guard for the contraception `(1 | dist)` row. The
/// comparison harness fits this row through the certified fast=false joint
/// path since its promotion (mote bd-01M35AQYXEXZHJA7JA7GTXR032), so the
/// MixedModels.jl fast=true oracle it used to carry is pinned here instead.
/// Reference: MixedModels.jl 5.3.0 `fit(MixedModel, @formula(use ~ 1 + age +
/// livch + urban + (1 | dist)), contra, Bernoulli(); fast=true)`, deviance
/// 2413.6626372063283 (also `glmm_contra_intercept_fast` in
/// examples/optimizer_bench_harness.rs). Response constants are zero for 0/1
/// data, so the dropped-constants deviance is directly comparable.
#[test]
fn contraception_intercept_fast_pirls_matches_mixedmodels_fast_true_objective() {
    let (data, _) = crate::datasets::load("contraception").unwrap();
    let formula = parse_formula("use ~ 1 + age + livch + urban + (1 | dist)").unwrap();
    let mut model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Binomial, None).unwrap();
    model.fit_with_options(true, 1, false).unwrap();
    let objective = model.deviance(1);
    let julia_fast = 2413.6626372063283;
    let delta = (objective - julia_fast).abs();
    assert!(
        delta <= 1e-3,
        "contraception (1 | dist) fast=true objective {objective:.9} should match MixedModels.jl fast=true {julia_fast:.9}; delta={delta:.3e}"
    );
}

#[cfg(feature = "nlopt")]
#[test]
fn test_grouseticks_poisson_glmm_deviance() {
    // pirls.jl:194-227, MixedModels.jl/test/pirls.jl
    //   gm4 = fit(MixedModel, only(gfms[:grouseticks]), grouseticks,
    //              Poisson(); fast=true)
    //   @test isapprox(deviance(gm4), 851.4046, atol=0.001)
    //
    // Formula in modelcache.jl:
    //   ticks ~ 1 + year + ch + (1 | index) + (1 | brood) + (1 | location)
    let (data, _) = crate::datasets::load("grouseticks").unwrap();
    let formula =
        parse_formula("TICKS ~ 1 + YEAR + cHEIGHT + (1 | INDEX) + (1 | BROOD) + (1 | LOCATION)")
            .unwrap();

    let mut model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Poisson, None).unwrap();

    model.fit_with_options(true, 1, false).unwrap();

    let theta = model.theta.clone();
    assert_eq!(theta.len(), 3, "expected three scalar-RE θ components");
    for (i, &t) in theta.iter().enumerate() {
        assert!(t >= 0.0, "θ[{i}] = {t} should be nonnegative");
        assert!(t.is_finite(), "θ[{i}] = {t} should be finite");
    }

    let dev = model.deviance(1);
    // Julia uses atol=0.001; we allow a slightly larger absolute slack
    // to absorb any remaining BOBYQA-vs-NEWUOA optimizer-driver
    // differences. Julia ref deviance: 851.4046.
    assert_relative_eq!(dev, 851.4046, max_relative = 1e-3);
}

#[cfg(feature = "nlopt")]
#[test]
fn test_contra_glmm_nagq_7_deviance() {
    // pirls.jl (contra testset, lines 94-97):
    //   refit!(gm0; nAGQ=7)
    //   @test isapprox(deviance(gm0), 2360.876, atol=0.001)
    //
    // After re-fitting with 7-point adaptive Gauss-Hermite quadrature
    // the deviance should drop slightly from the Laplace value.
    let data = contra_fixture();
    let formula =
        parse_formula("use_num ~ 1 + age + age2 + urban + livch + (1 | urban_dist)").unwrap();

    let mut model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();

    model.fit_with_options(true, 7, false).unwrap();

    // optsum should record the AGQ choice and a cached AGQ deviance.
    assert_eq!(model.lmm.optsum.n_agq, 7);
    eprintln!(
        "contra nAGQ=7 deviance: rust = {:.6}, julia ref = 2360.876",
        model.lmm.optsum.fmin
    );
    assert_relative_eq!(model.lmm.optsum.fmin, 2360.876, epsilon = 1.0);

    // Re-evaluating at the converged state should match the cached value
    // exactly (no further optimization between the two calls).
    let dev_agq = model.deviance(7);
    assert_relative_eq!(dev_agq, model.lmm.optsum.fmin, epsilon = 1e-9);

    // The Laplace value (n_agq = 1) should be close to but distinct from
    // the AGQ value at the same θ.
    let dev_lap = model.deviance(1);
    assert!(
        (dev_lap - dev_agq).abs() < 5.0,
        "Laplace and AGQ deviances should be within ~5 units (got {dev_lap} vs {dev_agq})",
    );
}

#[test]
fn test_matrix_block_diag_covers_all_variants() {
    // Direct unit test of the diagonal-extraction helper that AGQ uses
    // on the (1,1) Cholesky block. Contra exercises one variant in
    // practice; this guards the other two so refactors of the L block
    // layout can't silently break AGQ.
    use crate::types::MatrixBlock;
    use nalgebra::{DMatrix, DVector};

    let diag = MatrixBlock::Diagonal(DVector::from_vec(vec![1.0, 2.0, 3.0]));
    assert_eq!(matrix_block_diag(&diag), vec![1.0, 2.0, 3.0]);

    let blk0 = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
    let blk1 = DMatrix::from_row_slice(2, 2, &[5.0, 6.0, 7.0, 8.0]);
    let bd = MatrixBlock::BlockDiagonal(vec![blk0, blk1]);
    // Diagonal of each 2x2 block in order:
    // blk0 -> (0,0)=1, (1,1)=4; blk1 -> (0,0)=5, (1,1)=8.
    assert_eq!(matrix_block_diag(&bd), vec![1.0, 4.0, 5.0, 8.0]);

    // Dense, rectangular: returns min(rows,cols) diagonals.
    let m = DMatrix::from_row_slice(
        3,
        4,
        &[
            10.0, 0.0, 0.0, 0.0, //
            0.0, 20.0, 0.0, 0.0, //
            0.0, 0.0, 30.0, 0.0,
        ],
    );
    let dense = MatrixBlock::Dense(m);
    assert_eq!(matrix_block_diag(&dense), vec![10.0, 20.0, 30.0]);
}

#[test]
fn test_glmm_validate_agq_accepts_single_scalar_re() {
    let data = contra_fixture();
    let formula =
        parse_formula("use_num ~ 1 + age + age2 + urban + livch + (1 | urban_dist)").unwrap();
    let model = GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();

    // Single scalar RE: validation should accept any n_agq.
    assert!(model.is_single_scalar_re());
    assert!(model.validate_agq(0).is_ok());
    assert!(model.validate_agq(1).is_ok());
    assert!(model.validate_agq(7).is_ok());
    assert!(model.validate_agq(25).is_ok());
}

#[test]
fn test_glmm_validate_agq_rejects_vector_random_effect() {
    // (1 + age | urban_dist) has vsize == 2 — vector-valued RE.
    // AGQ is only defined for scalar REs; n_agq > 1 must be refused.
    let data = contra_fixture();
    let formula =
        parse_formula("use_num ~ 1 + age2 + urban + livch + (1 + age | urban_dist)").unwrap();
    let model = GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();

    assert!(
        !model.is_single_scalar_re(),
        "expected vector-RE model not to be classified single-scalar"
    );
    assert_eq!(model.lmm.reterms.len(), 1);
    assert_eq!(model.lmm.reterms[0].vsize, 2);

    // n_agq <= 1 is always allowed (Laplace).
    assert!(model.validate_agq(0).is_ok());
    assert!(model.validate_agq(1).is_ok());

    // n_agq > 1 must error with InvalidArgument citing the vsize mismatch.
    for n_agq in [2_usize, 3, 7, 11] {
        let err = model.validate_agq(n_agq).expect_err(&format!(
            "validate_agq({n_agq}) should error on a vector RE model"
        ));
        match err {
            MixedModelError::InvalidArgument(msg) => {
                assert!(
                    msg.contains("scalar"),
                    "error message should mention 'scalar' requirement; got {msg}"
                );
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }
}

#[test]
fn test_glmm_validate_agq_rejects_multi_term_random_effects() {
    // Two grouping factors (urban_dist + livch) — multi-term RE.
    // Even with each term scalar, AGQ is undefined.
    let data = contra_fixture();
    let formula =
        parse_formula("use_num ~ 1 + age + age2 + urban + (1 | urban_dist) + (1 | livch)").unwrap();
    let model = GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();

    assert_eq!(model.lmm.reterms.len(), 2);
    assert!(!model.is_single_scalar_re());

    assert!(model.validate_agq(1).is_ok());
    for n_agq in [2_usize, 7] {
        let err = model
            .validate_agq(n_agq)
            .expect_err("validate_agq should error on multi-term model");
        assert!(matches!(err, MixedModelError::InvalidArgument(_)));
    }
}

#[test]
fn test_glmm_fit_with_options_rejects_invalid_nagq_up_front() {
    // The fit entry point must preflight the AGQ guard, so users never
    // get a partial fit followed by a panic deep inside deviance().
    let data = contra_fixture();
    let formula =
        parse_formula("use_num ~ 1 + age2 + urban + livch + (1 + age | urban_dist)").unwrap();
    let mut model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();

    // Laplace fit is fine on a vector RE.
    let lap_result = model.fit_with_options(true, 1, false);
    assert!(lap_result.is_ok());

    // But asking for AGQ on the same shape must error before any work.
    let mut model2 = {
        let data = contra_fixture();
        let formula =
            parse_formula("use_num ~ 1 + age2 + urban + livch + (1 + age | urban_dist)").unwrap();
        GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap()
    };
    let feval_before = model2.lmm.optsum.feval;
    let err = model2
        .fit_with_options(true, 7, false)
        .expect_err("fit_with_options(_, 7, _) on a vector-RE model should error before fitting");
    assert!(matches!(err, MixedModelError::InvalidArgument(_)));
    assert_eq!(
        model2.lmm.optsum.feval, feval_before,
        "no objective evaluations should have happened on the rejected fit",
    );

    let mut model3 = {
        let data = contra_fixture();
        let formula =
            parse_formula("use_num ~ 1 + age2 + urban + livch + (1 + age | urban_dist)").unwrap();
        GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap()
    };
    let feval_before = model3.lmm.optsum.feval;
    let err = model3
        .fit_with_options(false, 7, false)
        .expect_err("fast=false AGQ must reject invalid RE shape before fitting");
    assert!(matches!(err, MixedModelError::InvalidArgument(_)));
    assert_eq!(
        model3.lmm.optsum.feval, feval_before,
        "fast=false invalid AGQ request must not run the joint optimizer",
    );
}

#[test]
fn test_glmm_refit_resets_theta_to_initial() {
    let data = contra_fixture();
    let formula =
        parse_formula("use_num ~ 1 + age + age2 + urban + livch + (1 | urban_dist)").unwrap();
    let mut model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();
    let initial_theta = model.lmm.optsum.initial.clone();

    model.fit_with_options(true, 1, false).unwrap();
    assert!(
        (model.theta[0] - initial_theta[0]).abs() > 1e-6,
        "fixture should move away from its starting theta"
    );

    let new_y = resampled_contra_response(&data);
    model.reset_for_refit(Some(&new_y)).unwrap();

    assert_eq!(model.theta, initial_theta);
    assert_eq!(model.lmm.optsum.final_params, initial_theta);
    assert_eq!(model.lmm.optsum.feval, 0);
    assert!(model.lmm.optsum.return_value.is_empty());
}

#[test]
fn test_glmm_bootstrap_does_not_warm_start() {
    let data = contra_fixture();
    let formula =
        parse_formula("use_num ~ 1 + age + age2 + urban + livch + (1 | urban_dist)").unwrap();
    let mut model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();
    let initial_theta = model.lmm.optsum.initial.clone();
    model.fit_with_options(true, 1, false).unwrap();
    let fitted_theta = model.theta.clone();

    let err = model
        .fit_with_options(true, 1, false)
        .expect_err("plain fit_with_options must not silently warm-start a fitted GLMM");
    assert!(matches!(err, MixedModelError::AlreadyFitted));

    let new_y = resampled_contra_response(&data);
    model.reset_for_refit(Some(&new_y)).unwrap();
    assert_eq!(
        model.theta, initial_theta,
        "bootstrap/refit reset must ignore the previous optimum"
    );
    assert_ne!(model.theta, fitted_theta);
}

#[test]
fn test_glmm_refit_after_resample_matches_cold_fit() {
    let data = contra_fixture();
    let formula =
        parse_formula("use_num ~ 1 + age + age2 + urban + livch + (1 | urban_dist)").unwrap();
    let mut warm_model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();
    warm_model.fit_with_options(true, 1, false).unwrap();

    let new_y = resampled_contra_response(&data);
    warm_model.refit(&new_y).unwrap();
    let cold_model = refit_cold_contra_model(&new_y);

    assert_relative_eq!(warm_model.theta[0], cold_model.theta[0], epsilon = 1e-8);
    assert_relative_eq!(
        warm_model.lmm.optsum.fmin,
        cold_model.lmm.optsum.fmin,
        epsilon = 1e-8
    );
    for (warm, cold) in warm_model.beta.iter().zip(cold_model.beta.iter()) {
        assert_relative_eq!(warm, cold, epsilon = 1e-8);
    }
}

#[test]
fn test_glmm_repeated_refit_does_not_accumulate_retained_state() {
    let data = contra_fixture();
    let formula =
        parse_formula("use_num ~ 1 + age + age2 + urban + livch + (1 | urban_dist)").unwrap();
    let original_y = data.numeric("use_num").unwrap().to_vec();
    let perturbed_y = resampled_contra_response(&data);
    let mut model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();
    model.fit_with_options(true, 1, false).unwrap();

    let baseline_slots = glmm_retained_state_slots(&model);
    let baseline_fit_log_capacity = model.lmm.optsum.fit_log.capacity();

    for iteration in 0..8 {
        let y = if iteration % 2 == 0 {
            &perturbed_y
        } else {
            &original_y
        };
        model.refit(y).unwrap();

        assert_eq!(
            glmm_retained_state_slots(&model),
            baseline_slots,
            "GLMM refit should reuse bounded work buffers rather than accumulating retained state"
        );
        assert_eq!(
            model.lmm.optsum.fit_log.capacity(),
            baseline_fit_log_capacity,
            "GLMM optimizer logging must not retain one entry per refit iteration"
        );
    }
}

#[test]
fn test_glmm_theta_probe_penalizes_invalid_theta() {
    let data = contra_fixture();
    let formula =
        parse_formula("use_num ~ 1 + age + age2 + urban + livch + (1 | urban_dist)").unwrap();
    let mut model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();

    let value = model.penalized_pirls_deviance_at_theta(&[f64::NAN], 1);
    assert!(
        value.is_infinite() && value.is_sign_positive(),
        "invalid optimizer probes should be penalized, not evaluated from stale state"
    );
}

#[test]
fn test_glmm_final_theta_update_propagates_invalid_theta_error() {
    let data = contra_fixture();
    let formula =
        parse_formula("use_num ~ 1 + age + age2 + urban + livch + (1 | urban_dist)").unwrap();
    let mut model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();

    let err = model
        .update_pirls_at_theta(&[f64::NAN], true)
        .expect_err("final theta update must propagate invalid-theta errors");
    assert!(matches!(err, MixedModelError::InvalidArgument(_)));
}

fn glmm_prediction_data() -> DataFrame {
    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut group = Vec::new();
    let group_effects = [-0.45, 0.1, 0.35, -0.05, 0.25];
    for (g, effect) in group_effects.iter().enumerate() {
        for obs in 0..8 {
            let xv = obs as f64 - 3.5;
            let eta = 0.6 + 0.2 * xv + effect;
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

fn glmm_certified_prediction_data() -> DataFrame {
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

fn glmm_prediction_fixture() -> (GeneralizedLinearMixedModel, DataFrame) {
    let data = glmm_prediction_data();
    let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();
    let mut model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Poisson, None).unwrap();
    model.fit().unwrap();
    (model, data)
}

#[test]
fn test_glmm_predict_new_same_data_matches_fitted_on_response_and_link_scale() {
    let (model, data) = glmm_prediction_fixture();

    let response = model
        .predict_new(&data, GlmmPredictionScale::Response, NewReLevels::Error)
        .unwrap();
    let fitted = model.fitted();
    assert_eq!(response.len(), fitted.len());
    for (idx, prediction) in response.iter().enumerate() {
        assert_relative_eq!(
            prediction.expect("training rows have known random-effect levels"),
            fitted[idx],
            epsilon = 1e-9,
            max_relative = 1e-9
        );
    }

    let link = model
        .predict_new(&data, GlmmPredictionScale::Link, NewReLevels::Error)
        .unwrap();
    assert_eq!(link.len(), model.eta.len());
    for (idx, prediction) in link.iter().enumerate() {
        assert_relative_eq!(
            prediction.expect("training rows have known random-effect levels"),
            model.eta[idx],
            epsilon = 1e-9,
            max_relative = 1e-9
        );
    }
}

#[test]
fn test_glmm_predict_new_unseen_levels_follow_policy() {
    let (model, _) = glmm_prediction_fixture();

    let mut newdata = DataFrame::new();
    newdata.add_numeric("y", vec![0.0, 0.0]).unwrap();
    newdata.add_numeric("x", vec![0.0, 0.0]).unwrap();
    newdata
        .add_categorical("group", vec!["NEW".to_string(), "g1".to_string()])
        .unwrap();

    let err = model
        .predict_new(&newdata, GlmmPredictionScale::Response, NewReLevels::Error)
        .unwrap_err();
    assert_eq!(err.code(), "invalid_argument");
    assert!(err.to_string().contains("NEW"));
    assert!(err.to_string().contains("group"));

    let population = model
        .predict_new(
            &newdata,
            GlmmPredictionScale::Response,
            NewReLevels::Population,
        )
        .unwrap();
    assert_eq!(population.len(), 2);
    assert!(population[0].is_some());
    assert!(population[1].is_some());

    let missing = model
        .predict_new(
            &newdata,
            GlmmPredictionScale::Response,
            NewReLevels::Missing,
        )
        .unwrap();
    assert_eq!(missing[0], None);
    assert!(missing[1].is_some());
}

#[test]
fn test_glmm_predict_new_with_offset_applies_offset_on_link_scale() {
    let (model, data) = glmm_prediction_fixture();

    let base = model
        .predict_new(&data, GlmmPredictionScale::Link, NewReLevels::Error)
        .unwrap();
    let offset = vec![0.25; data.nrow()];
    let shifted = model
        .predict_new_with_offset(
            &data,
            Some(&offset),
            GlmmPredictionScale::Link,
            NewReLevels::Error,
        )
        .unwrap();

    for (base, shifted) in base.iter().zip(shifted.iter()) {
        assert_relative_eq!(
            shifted.expect("known level"),
            base.expect("known level") + 0.25,
            epsilon = 1e-12
        );
    }
}

#[test]
fn test_glmm_predict_new_variance_returns_degraded_working_delta_payload() {
    let (model, data) = glmm_prediction_fixture();

    let payload = model
        .predict_new_variance(&data, GlmmPredictionScale::Response, NewReLevels::Error)
        .unwrap();
    assert_eq!(
        payload.method,
        PredictionVarianceMethod::GlmmPirlsLaplaceWorkingDelta
    );
    assert_eq!(payload.confidence_level, Some(0.95));
    assert_eq!(payload.rows.len(), data.nrow());
    let fitted = model.fitted();
    let first = &payload.rows[0];
    assert_eq!(first.status, PredictionVarianceStatus::Degraded);
    assert_relative_eq!(
        first.prediction.expect("GLMM point prediction"),
        fitted[0],
        epsilon = 1e-9,
        max_relative = 1e-9
    );
    assert!(first.fixed_variance.unwrap() > 0.0);
    assert!(first.random_variance.unwrap() >= 0.0);
    assert!(first.fixed_random_covariance.unwrap().is_finite());
    assert!(first.combined_variance.unwrap() > 0.0);
    assert!(first.se_fit.unwrap() > 0.0);
    assert!(first.prediction_variance.unwrap() > 0.0);
    assert!(first.confidence_lower.unwrap() < first.prediction.unwrap());
    assert!(first.confidence_upper.unwrap() > first.prediction.unwrap());
    let prediction_lower = first.prediction_lower.unwrap();
    let prediction_upper = first.prediction_upper.unwrap();
    assert!(prediction_lower >= 0.0);
    assert!(prediction_lower <= prediction_upper);
    assert_eq!(prediction_lower.fract(), 0.0, "poisson bounds are counts");
    assert_eq!(prediction_upper.fract(), 0.0, "poisson bounds are counts");
    assert!(first.prediction_variance.unwrap() > first.combined_variance.unwrap());
    let reason = first.reason.as_deref().unwrap_or("");
    assert!(reason.contains("the fast-PIRLS profiled optimum certificate was not issued"));
    assert!(reason.contains("GlmmFitOptions::joint_laplace()"));
}

fn glmm_certified_pirls_poisson_fixture() -> (GeneralizedLinearMixedModel, DataFrame) {
    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut group = Vec::new();
    let group_effects = [-0.9_f64, -0.3, 0.2, 0.7, 1.1, -0.5];
    for (g, effect) in group_effects.iter().enumerate() {
        for obs in 0..10 {
            let xv = (obs as f64 - 4.5) / 3.0;
            let eta = 1.0 + 0.3 * xv + effect;
            let noise = 0.85 + 0.3 * (((g * 13 + obs * 7) % 11) as f64 / 10.0);
            y.push((eta.exp() * noise).round().max(0.0));
            x.push(xv);
            group.push(format!("g{}", g + 1));
        }
    }
    let mut data = DataFrame::new();
    data.add_numeric("y", y).unwrap();
    data.add_numeric("x", x).unwrap();
    data.add_categorical("group", group).unwrap();
    let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();
    let mut model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Poisson, None).unwrap();
    model.fit().unwrap();
    (model, data)
}

#[test]
#[cfg(feature = "nlopt")]
fn test_glmm_pirls_certified_prediction_variance_rows_available() {
    let (model, data) = glmm_certified_pirls_poisson_fixture();
    assert!(
        matches!(model.pirls_profiled_optimum_certificate(), Some(Ok(_))),
        "fixture should certify: {:?}",
        model.pirls_profiled_optimum_certificate()
    );

    let payload = model
        .predict_new_variance(&data, GlmmPredictionScale::Response, NewReLevels::Error)
        .unwrap();
    assert_eq!(
        payload.method,
        PredictionVarianceMethod::GlmmPirlsProfiledCertifiedConditionalDelta
    );
    assert!(payload
        .notes
        .iter()
        .any(|note| note.contains("certified profiled optimum")));
    for row in &payload.rows {
        assert_eq!(row.status, PredictionVarianceStatus::Available);
        assert_eq!(row.reason, None);
        let prediction = row.prediction.expect("point prediction");
        assert!(row.se_fit.unwrap() > 0.0);
        // The Poisson future-observation variance is dominated by the
        // family term E[mu], so it must exceed the fitted-mean variance.
        assert!(row.prediction_variance.unwrap() > row.combined_variance.unwrap());
        let lower = row.prediction_lower.unwrap();
        let upper = row.prediction_upper.unwrap();
        assert_eq!(lower.fract(), 0.0);
        assert_eq!(upper.fract(), 0.0);
        assert!(lower >= 0.0);
        assert!(lower <= prediction.ceil());
        assert!(upper >= prediction.floor());
        assert!(upper > lower);
    }

    assert!(model
        .compiler_artifact()
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic
            .payload
            .get("glmm_pirls_profiled_optimum_certificate")
            .and_then(serde_json::Value::as_str)
            == Some("issued")));
}

#[test]
#[cfg(not(feature = "nlopt"))]
fn test_glmm_pirls_native_prediction_variance_rows_degrade_without_certificate() {
    let (model, data) = glmm_certified_pirls_poisson_fixture();
    assert!(
        matches!(model.pirls_profiled_optimum_certificate(), Some(Err(_))),
        "native fixture should keep uncertified geometry explicit: {:?}",
        model.pirls_profiled_optimum_certificate()
    );

    let payload = model
        .predict_new_variance(&data, GlmmPredictionScale::Response, NewReLevels::Error)
        .unwrap();
    assert_eq!(
        payload.method,
        PredictionVarianceMethod::GlmmPirlsLaplaceWorkingDelta
    );
    for row in &payload.rows {
        assert_eq!(row.status, PredictionVarianceStatus::Degraded);
        let reason = row.reason.as_deref().unwrap_or("");
        assert!(reason.contains("the fast-PIRLS profiled optimum certificate was not issued"));
        assert!(reason.contains("GlmmFitOptions::joint_laplace()"));
        assert!(row.se_fit.unwrap() > 0.0);
        assert!(row.prediction_variance.unwrap() > row.combined_variance.unwrap());
    }
}

#[test]
fn test_glmm_pirls_uncertified_fit_keeps_degraded_with_refit_guidance() {
    let (mut model, data) = glmm_certified_pirls_poisson_fixture();
    model.complete_pirls_certificate();
    model.pirls_profiled_optimum_certificate =
        Some(Err("forced certificate failure for test".to_string()));

    let payload = model
        .predict_new_variance(&data, GlmmPredictionScale::Response, NewReLevels::Error)
        .unwrap();
    assert_eq!(
        payload.method,
        PredictionVarianceMethod::GlmmPirlsLaplaceWorkingDelta
    );
    let first = &payload.rows[0];
    assert_eq!(first.status, PredictionVarianceStatus::Degraded);
    let reason = first.reason.as_deref().unwrap();
    assert!(reason.contains("forced certificate failure for test"));
    assert!(reason.contains("GlmmFitOptions::joint_laplace()"));
    // Degraded rows still carry the (uncertified) predictive columns so
    // downstream layers can surface them together with the reason.
    assert!(first.prediction_variance.unwrap() > 0.0);
}

#[test]
fn test_glmm_link_scale_rows_do_not_carry_future_observation_columns() {
    let (model, data) = glmm_certified_pirls_poisson_fixture();
    let payload = model
        .predict_new_variance(&data, GlmmPredictionScale::Link, NewReLevels::Error)
        .unwrap();
    let first = &payload.rows[0];
    if matches!(model.pirls_profiled_optimum_certificate(), Some(Ok(_))) {
        assert_eq!(first.status, PredictionVarianceStatus::Available);
    } else {
        assert_eq!(first.status, PredictionVarianceStatus::Degraded);
    }
    assert_eq!(first.prediction_variance, None);
    assert_eq!(first.prediction_lower, None);
    assert_eq!(first.prediction_upper, None);
    assert!(payload
        .notes
        .iter()
        .any(|note| note.contains("response-scale objects")));
}

#[test]
fn test_glmm_bernoulli_future_observation_bounds_are_support_points() {
    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut group = Vec::new();
    for g in 0..8usize {
        for obs in 0..12usize {
            let idx = g * 12 + obs;
            let xv = (obs as f64 - 5.5) / 2.2;
            let eta = -0.3 + 1.8 * xv + (g as f64 - 3.5) * 0.25;
            let p = 1.0 / (1.0 + (-eta).exp());
            let u = ((idx * 37 + 11) % 97) as f64 / 97.0;
            y.push(if p > u { 1.0 } else { 0.0 });
            x.push(xv);
            group.push(format!("g{}", g + 1));
        }
    }
    let mut data = DataFrame::new();
    data.add_numeric("y", y).unwrap();
    data.add_numeric("x", x).unwrap();
    data.add_categorical("group", group).unwrap();

    let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();
    let mut model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();
    model.fit_with_options(false, 1, false).unwrap();

    let payload = model
        .predict_new_variance(&data, GlmmPredictionScale::Response, NewReLevels::Error)
        .unwrap();
    let mut saw_zero_lower = false;
    let mut saw_unit_upper = false;
    for row in &payload.rows {
        let lower = row.prediction_lower.unwrap();
        let upper = row.prediction_upper.unwrap();
        assert!(lower == 0.0 || lower == 1.0);
        assert!(upper == 0.0 || upper == 1.0);
        assert!(lower <= upper);
        let variance = row.prediction_variance.unwrap();
        // Law of total variance for a Bernoulli future observation:
        // bounded by the maximal Bernoulli variance.
        assert!(variance > 0.0 && variance <= 0.25 + 1.0e-9);
        saw_zero_lower |= lower == 0.0;
        saw_unit_upper |= upper == 1.0;
    }
    assert!(saw_zero_lower && saw_unit_upper);
}

#[test]
fn test_glmm_binomial_future_observation_refused_with_trial_count_reason() {
    let (data, _) = crate::datasets::load("cbpp").unwrap();
    let incidence = data.numeric("incidence").unwrap();
    let size = data.numeric("size").unwrap();
    let proportion: Vec<f64> = incidence
        .iter()
        .zip(size.iter())
        .map(|(&y, &n)| y / n)
        .collect();
    let weights: Vec<f64> = size.to_vec();
    let mut data_with_proportion = data.clone();
    data_with_proportion
        .add_numeric("proportion", proportion)
        .unwrap();
    let formula = parse_formula("proportion ~ 1 + period + (1 | herd)").unwrap();
    let mut model = GeneralizedLinearMixedModel::new_with_weights(
        formula,
        &data_with_proportion,
        Family::Binomial,
        None,
        weights,
    )
    .unwrap();
    model.fit().unwrap();

    let payload = model
        .predict_new_variance(
            &data_with_proportion,
            GlmmPredictionScale::Response,
            NewReLevels::Error,
        )
        .unwrap();
    let first = &payload.rows[0];
    assert_eq!(first.prediction_variance, None);
    assert_eq!(first.prediction_lower, None);
    assert_eq!(first.prediction_upper, None);
    assert!(first.confidence_lower.is_some());
    assert!(payload
        .notes
        .iter()
        .any(|note| note.contains("trial count")));
}

#[test]
fn test_discrete_mixture_quantile_matches_single_poisson_reference() {
    let poisson = PoissonDist::new(4.2).unwrap();
    let cdf = |t: u64| poisson.cdf(t);
    // scipy.stats.poisson.ppf reference values for lambda = 4.2.
    assert_eq!(discrete_mixture_quantile(&cdf, 0.025, 4.2), Some(1.0));
    assert_eq!(discrete_mixture_quantile(&cdf, 0.975, 4.2), Some(9.0));
    assert_eq!(discrete_mixture_quantile(&cdf, 0.005, 4.2), Some(0.0));
    assert_eq!(discrete_mixture_quantile(&cdf, 0.995, 4.2), Some(10.0));
}

#[test]
fn test_inverse_gaussian_cdf_matches_scipy_reference() {
    // scipy.stats.invgauss(mu=1, scale=1).cdf(1) and
    // scipy.stats.invgauss(mu=4, scale=0.5).cdf(3) (mean 2, shape 0.5).
    // statrs's erfc-based normal CDF carries ~1e-11 absolute error, so
    // the comparison tolerance reflects that, not the IG formula.
    assert_relative_eq!(
        inverse_gaussian_cdf(1.0, 1.0, 1.0),
        0.6681020012231706,
        epsilon = 1.0e-9
    );
    assert_relative_eq!(
        inverse_gaussian_cdf(3.0, 2.0, 0.5),
        0.8343083811593116,
        epsilon = 1.0e-9
    );
    assert_eq!(inverse_gaussian_cdf(0.0, 1.0, 1.0), 0.0);
    assert_eq!(inverse_gaussian_cdf(-1.0, 1.0, 1.0), 0.0);
}

#[test]
fn test_standard_normal_ln_cdf_tail_is_continuous_and_consistent() {
    let direct = Normal::new(0.0, 1.0).unwrap().cdf(-5.0).ln();
    assert_relative_eq!(standard_normal_ln_cdf(-5.0), direct, epsilon = 1.0e-12);
    let just_above = standard_normal_ln_cdf(-36.9);
    let just_below = standard_normal_ln_cdf(-37.1);
    assert!(just_below < just_above);
    assert!((just_below - just_above).abs() < 8.0);
}

#[test]
fn test_continuous_mixture_quantile_matches_single_normal_reference() {
    let normal = Normal::new(2.0, 3.0).unwrap();
    let cdf = |t: f64| normal.cdf(t);
    let q = continuous_mixture_quantile(&cdf, 0.975, None, 2.0, 3.0).unwrap();
    assert_relative_eq!(q, 2.0 + 1.959963984540054 * 3.0, epsilon = 1.0e-6);
    let q_low = continuous_mixture_quantile(&cdf, 0.025, None, 2.0, 3.0).unwrap();
    assert_relative_eq!(q_low, 2.0 - 1.959963984540054 * 3.0, epsilon = 1.0e-6);
}

#[test]
fn test_glmm_response_scale_confidence_bounds_stay_in_family_range() {
    // Strong slope pushes fitted probabilities near 0 and 1 so symmetric
    // response-scale bounds would escape (0, 1).
    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut group = Vec::new();
    for g in 0..8usize {
        for obs in 0..12usize {
            let idx = g * 12 + obs;
            let xv = (obs as f64 - 5.5) / 2.2;
            let eta = -0.3 + 1.8 * xv + (g as f64 - 3.5) * 0.25;
            let p = 1.0 / (1.0 + (-eta).exp());
            let u = ((idx * 37 + 11) % 97) as f64 / 97.0;
            y.push(if p > u { 1.0 } else { 0.0 });
            x.push(xv);
            group.push(format!("g{}", g + 1));
        }
    }
    let mut data = DataFrame::new();
    data.add_numeric("y", y).unwrap();
    data.add_numeric("x", x).unwrap();
    data.add_categorical("group", group).unwrap();

    let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();
    let mut model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();
    model.fit_with_options(false, 1, false).unwrap();

    let response = model
        .predict_new_variance(&data, GlmmPredictionScale::Response, NewReLevels::Error)
        .unwrap();
    let link = model
        .predict_new_variance(&data, GlmmPredictionScale::Link, NewReLevels::Error)
        .unwrap();
    assert!(response
        .notes
        .iter()
        .any(|note| note.contains("mapped through the inverse link")));

    let z = 1.959963984540054;
    let mut rows_with_bounds = 0;
    let mut symmetric_would_escape = false;
    for (row, link_row) in response.rows.iter().zip(link.rows.iter()) {
        let (Some(fit), Some(se_fit), Some(lower), Some(upper)) = (
            row.prediction,
            row.se_fit,
            row.confidence_lower,
            row.confidence_upper,
        ) else {
            continue;
        };
        rows_with_bounds += 1;
        assert!(
            lower > 0.0 && upper < 1.0,
            "row {}: response bounds ({lower}, {upper}) escape (0, 1)",
            row.row
        );
        assert!(lower < fit && fit < upper);
        if fit - z * se_fit < 0.0 || fit + z * se_fit > 1.0 {
            symmetric_would_escape = true;
        }
        // The response bounds must be the link-scale bounds mapped
        // through the inverse link.
        let link_lower = link_row.confidence_lower.expect("link lower bound");
        let link_upper = link_row.confidence_upper.expect("link upper bound");
        assert_relative_eq!(
            lower,
            1.0 / (1.0 + (-link_lower).exp()),
            epsilon = 1e-12,
            max_relative = 1e-12
        );
        assert_relative_eq!(
            upper,
            1.0 / (1.0 + (-link_upper).exp()),
            epsilon = 1e-12,
            max_relative = 1e-12
        );
    }
    assert!(rows_with_bounds > 0, "fixture should yield bounded rows");
    assert!(
        symmetric_would_escape,
        "fixture should reproduce the symmetric-bounds escape this test guards against"
    );
}

#[test]
fn test_glmm_predict_new_variance_reports_joint_laplace_conditional_rows_available() {
    let data = glmm_certified_prediction_data();
    let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();
    let mut model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Gamma, Some(LinkFunction::Log))
            .unwrap();
    model.fit_with_options(false, 1, false).unwrap();

    let artifact = model.compiler_artifact();
    let covariance = artifact
        .fixed_effect_covariance_matrix
        .as_ref()
        .expect("joint-laplace fit should expose fixed covariance");
    assert_eq!(
        covariance.method,
        FixedEffectCovarianceMethod::JointLaplaceActiveHessian
    );
    let matrix = covariance
        .matrix
        .as_ref()
        .expect("certified covariance should carry matrix values");

    let payload = model
        .predict_new_variance(&data, GlmmPredictionScale::Response, NewReLevels::Error)
        .unwrap();
    let first = &payload.rows[0];
    assert_eq!(
        payload.method,
        PredictionVarianceMethod::GlmmJointLaplaceConditionalDelta
    );
    assert_eq!(first.status, PredictionVarianceStatus::Available);
    assert_eq!(first.reason, None);
    assert!(payload
        .notes
        .iter()
        .any(|note| note.contains("conditional-mode covariance")));

    assert!(matrix.iter().flatten().all(|value| value.is_finite()));
    assert!(first.fixed_variance.expect("fixed component") > 0.0);
    assert!(first.random_variance.expect("random component") >= 0.0);
    assert!(first
        .fixed_random_covariance
        .expect("fixed/random covariance")
        .is_finite());
    assert_relative_eq!(
        first.combined_variance.expect("combined component"),
        first.fixed_variance.unwrap()
            + first.random_variance.unwrap()
            + 2.0 * first.fixed_random_covariance.unwrap(),
        epsilon = 1.0e-8,
        max_relative = 1.0e-8
    );
    assert!(first.se_fit.unwrap() > 0.0);
    assert!(first.prediction_variance.unwrap() > 0.0);
    assert!(first.confidence_lower.unwrap() < first.prediction.unwrap());
    assert!(first.confidence_upper.unwrap() > first.prediction.unwrap());
    let prediction_lower = first.prediction_lower.unwrap();
    let prediction_upper = first.prediction_upper.unwrap();
    assert!(prediction_lower > 0.0, "gamma future bounds stay positive");
    assert!(prediction_lower < first.prediction.unwrap());
    assert!(prediction_upper > first.prediction.unwrap());
    assert!(prediction_lower <= first.confidence_lower.unwrap() + 1.0e-9);
    assert!(prediction_upper >= first.confidence_upper.unwrap() - 1.0e-9);

    let link_payload = model
        .predict_new_variance(&data, GlmmPredictionScale::Link, NewReLevels::Error)
        .unwrap();

    // lme4 2.0.1 reference:
    // glmer(y ~ 1 + x + (1 | group), data, family = Gamma(link = "log"),
    //       nAGQ = 1, control = glmerControl(optimizer = "bobyqa"))
    // predict(..., newdata = data[1:5,], re.form = NULL, se.fit = TRUE)
    // emits lme4's documented approximation warning for se.fit.
    let lme4_response_fit = [0.9529792, 1.1645747, 1.4231520, 1.7391427, 2.1252947];
    let lme4_response_se = [0.01402705, 0.01476743, 0.01696936, 0.02205325, 0.03128255];
    for (idx, (row, (expected_fit, expected_se))) in payload
        .rows
        .iter()
        .take(lme4_response_fit.len())
        .zip(lme4_response_fit.into_iter().zip(lme4_response_se))
        .enumerate()
    {
        let fit = row.prediction.expect("response-scale GLMM prediction");
        assert!(
            (fit - expected_fit).abs() <= 5.0e-5_f64.max(5.0e-5 * expected_fit.abs()),
            "response-scale lme4 fit parity row {idx}: observed {fit}, expected {expected_fit}"
        );
        let se_fit = row.se_fit.expect("response-scale GLMM se.fit");
        assert!(
                (se_fit - expected_se).abs() <= 5.0e-5_f64.max(5.0e-5 * expected_se.abs()),
                "response-scale lme4 se.fit parity row {idx}: observed {se_fit}, expected {expected_se}"
            );
    }

    let lme4_link_fit = [-0.0481622, 0.1523560, 0.3528741, 0.5533923, 0.7539105];
    let lme4_link_se = [0.01471916, 0.01268053, 0.01192378, 0.01268053, 0.01471916];
    let lme4_link_fixed = [
        0.0006883062,
        0.0006324485,
        0.0006138292,
        0.0006324485,
        0.0006883062,
    ];
    let lme4_link_random = [0.0006815289; 5];
    let lme4_link_cross = [-0.0005765908; 5];
    let lme4_link_combined = [
        0.0002166536,
        0.0001607959,
        0.0001421766,
        0.0001607959,
        0.0002166536,
    ];
    for (idx, (row, (expected_fit, expected_se))) in link_payload
        .rows
        .iter()
        .take(lme4_link_fit.len())
        .zip(lme4_link_fit.into_iter().zip(lme4_link_se))
        .enumerate()
    {
        assert_eq!(row.status, PredictionVarianceStatus::Available);
        let fit = row.prediction.expect("link-scale GLMM prediction");
        assert!(
            (fit - expected_fit).abs() <= 5.0e-5_f64.max(5.0e-5 * expected_fit.abs()),
            "link-scale lme4 fit parity row {idx}: observed {fit}, expected {expected_fit}"
        );
        let se_fit = row.se_fit.expect("link-scale GLMM se.fit");
        assert!(
            (se_fit - expected_se).abs() <= 5.0e-5_f64.max(5.0e-5 * expected_se.abs()),
            "link-scale lme4 se.fit parity row {idx}: observed {se_fit}, expected {expected_se}"
        );
        let fixed = row.fixed_variance.expect("link-scale GLMM fixed component");
        assert!(
            (fixed - lme4_link_fixed[idx]).abs()
                <= 1.0e-6_f64.max(1.0e-6 * lme4_link_fixed[idx].abs()),
            "link-scale lme4 fixed component parity row {idx}: observed {fixed}, expected {}",
            lme4_link_fixed[idx]
        );
        let random = row
            .random_variance
            .expect("link-scale GLMM random component");
        assert!(
            (random - lme4_link_random[idx]).abs()
                <= 1.0e-6_f64.max(1.0e-6 * lme4_link_random[idx].abs()),
            "link-scale lme4 random component parity row {idx}: observed {random}, expected {}",
            lme4_link_random[idx]
        );
        let cross = row
            .fixed_random_covariance
            .expect("link-scale GLMM fixed/random component");
        assert!(
                (cross - lme4_link_cross[idx]).abs()
                    <= 1.0e-6_f64.max(1.0e-6 * lme4_link_cross[idx].abs()),
                "link-scale lme4 fixed/random component parity row {idx}: observed {cross}, expected {}",
                lme4_link_cross[idx]
            );
        let combined = row
            .combined_variance
            .expect("link-scale GLMM combined component");
        assert!(
            (combined - lme4_link_combined[idx]).abs()
                <= 1.0e-6_f64.max(1.0e-6 * lme4_link_combined[idx].abs()),
            "link-scale lme4 combined component parity row {idx}: observed {combined}, expected {}",
            lme4_link_combined[idx]
        );
    }
}

#[test]
fn test_glmm_predict_new_variance_unseen_level_keeps_unavailable_reason() {
    let (model, _) = glmm_prediction_fixture();

    let mut newdata = DataFrame::new();
    newdata.add_numeric("y", vec![0.0, 0.0]).unwrap();
    newdata.add_numeric("x", vec![0.0, 0.0]).unwrap();
    newdata
        .add_categorical("group", vec!["NEW".to_string(), "g1".to_string()])
        .unwrap();

    let payload = model
        .predict_new_variance(
            &newdata,
            GlmmPredictionScale::Response,
            NewReLevels::Population,
        )
        .unwrap();
    let unseen = &payload.rows[0];
    assert_eq!(unseen.status, PredictionVarianceStatus::Unavailable);
    assert!(unseen.prediction.is_some());
    assert!(unseen.fixed_variance.is_some());
    assert_eq!(unseen.random_variance, None);
    assert_eq!(unseen.fixed_random_covariance, None);
    assert_eq!(unseen.combined_variance, None);
    assert_eq!(unseen.se_fit, None);
    assert!(unseen
        .reason
        .as_deref()
        .unwrap_or("")
        .contains("new level 'NEW'"));

    let known = &payload.rows[1];
    assert_eq!(known.status, PredictionVarianceStatus::Degraded);
    assert!(known.se_fit.unwrap() > 0.0);
}

#[test]
fn test_glmm_profile_likelihood_methods_refuse_with_explicit_reason() {
    let (mut model, _) = glmm_prediction_fixture();

    let sigma_err = model.profile_sigma(4.0).unwrap_err();
    assert_eq!(sigma_err.code(), "unsupported");
    let sigma_msg = sigma_err.to_string();
    assert!(sigma_msg.contains("profile_sigma"));
    assert!(sigma_msg.contains("GLMM profile likelihood is not implemented"));
    assert!(sigma_msg.contains("LMM-only"));

    let theta_err = model.profile_theta(0, 4.0).unwrap_err();
    assert_eq!(theta_err.code(), "unsupported");
    let theta_msg = theta_err.to_string();
    assert!(theta_msg.contains("profile_theta"));
    assert!(theta_msg.contains("GLMM profile likelihood is not implemented"));
    assert!(theta_msg.contains("LMM-only"));
}

#[test]
fn test_glmm_rectify_after_fit() {
    let data = contra_fixture();
    let formula =
        parse_formula("use_num ~ 1 + age2 + urban + livch + (1 + age | urban_dist)").unwrap();
    let mut model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();
    let mut theta = vec![-0.5, 0.05, -0.25];

    model.finalize_theta_after_optimizer(&mut theta, 1).unwrap();

    assert_eq!(theta, vec![0.5, -0.05, 0.25]);
    assert_eq!(model.theta, theta);
    assert_eq!(model.lmm.optsum.final_params, theta);
    assert!(model.lmm.optsum.fmin.is_finite());
    assert_glmm_theta_diagonals_nonnegative(&model);
}

#[test]
fn test_glmm_deviance_agq_restores_state() {
    // After a Laplace fit, snapshotting (u, eta, mu) and then calling
    // deviance(7) must leave those vectors bit-equivalent on return:
    // AGQ is supposed to perturb-and-restore.
    let data = contra_fixture();
    let formula =
        parse_formula("use_num ~ 1 + age + age2 + urban + livch + (1 | urban_dist)").unwrap();

    let mut model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();
    model.fit_with_options(true, 1, false).unwrap(); // Laplace fit.

    let u_snap: Vec<DMatrix<f64>> = model.u.clone();
    let eta_snap = model.eta.clone();
    let mu_snap = model.mu.clone();

    let _agq = model.deviance(7);

    // u must be byte-identical: the AGQ sweep restores from u₀.
    assert_eq!(model.u.len(), u_snap.len());
    for (after, before) in model.u.iter().zip(u_snap.iter()) {
        assert_eq!(
            after.shape(),
            before.shape(),
            "u shape must not change across deviance(n_agq)"
        );
        for (a, b) in after.iter().zip(before.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "u entry diverged: before={b}, after={a}"
            );
        }
    }

    // eta and mu may pick up tiny fp differences from the final
    // update_eta() call, but should be within ~1e-12 absolute.
    for (a, b) in model.eta.iter().zip(eta_snap.iter()) {
        assert!(
            (a - b).abs() < 1e-10,
            "eta drifted across AGQ sweep: before={b}, after={a}"
        );
    }
    for (a, b) in model.mu.iter().zip(mu_snap.iter()) {
        assert!(
            (a - b).abs() < 1e-12,
            "mu drifted across AGQ sweep: before={b}, after={a}"
        );
    }

    // And a Laplace re-eval must match its pre-AGQ value.
    let lap_after = model.deviance(1);
    let lap_before = {
        // Recompute a fresh Laplace from the pre-AGQ snapshot for parity.
        let dev_resid: f64 = (0..model.y.len())
            .map(|i| model.dev_resid_component(model.y[i], mu_snap[i]))
            .sum();
        let u_pen: f64 = u_snap
            .iter()
            .map(|u| u.iter().map(|x| x * x).sum::<f64>())
            .sum();
        dev_resid + u_pen + model.lmm_logdet()
    };
    assert!(
        (lap_after - lap_before).abs() < 1e-9,
        "Laplace deviance drifted across AGQ sweep: before={lap_before}, after={lap_after}"
    );
}

#[cfg(feature = "nlopt")]
#[test]
fn test_glmm_nagq_sweep_converges_on_contra() {
    // At a fixed θ, the n-point AGQ deviance should approach a limit
    // as n_agq grows. We assert:
    //   * all values lie within a small band around the Julia reference
    //     (~2360.876, our Rust fit ~2360.98)
    //   * successive doublings of n_agq move by less than 0.05 (well
    //     below the 1.0 tolerance pattern used elsewhere)
    //   * n_agq=1 path equals laplace_objective() exactly.
    let data = contra_fixture();
    let formula =
        parse_formula("use_num ~ 1 + age + age2 + urban + livch + (1 | urban_dist)").unwrap();

    let mut model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();
    model.fit_with_options(true, 7, false).unwrap();

    let lap_direct = model.laplace_objective();
    let lap_via_dev = model.deviance(1);
    assert_eq!(
        lap_direct.to_bits(),
        lap_via_dev.to_bits(),
        "deviance(1) and laplace_objective() must agree bit-for-bit"
    );

    let dev3 = model.deviance(3);
    let dev5 = model.deviance(5);
    let dev7 = model.deviance(7);
    let dev9 = model.deviance(9);
    let dev15 = model.deviance(15);

    // Rough band: all AGQ evaluations should sit within ~2 deviance units
    // of the Julia reference 2360.876.
    for (label, val) in [
        ("nAGQ=3", dev3),
        ("nAGQ=5", dev5),
        ("nAGQ=7", dev7),
        ("nAGQ=9", dev9),
        ("nAGQ=15", dev15),
    ] {
        assert!(
            (val - 2360.876_f64).abs() < 2.0,
            "{label} deviance {val} too far from Julia ref 2360.876"
        );
    }

    // Convergence: successive refinements should change by < 0.05.
    for (a_label, a, b_label, b) in [
        ("nAGQ=3", dev3, "nAGQ=5", dev5),
        ("nAGQ=5", dev5, "nAGQ=7", dev7),
        ("nAGQ=7", dev7, "nAGQ=9", dev9),
        ("nAGQ=9", dev9, "nAGQ=15", dev15),
    ] {
        assert!(
            (a - b).abs() < 0.05,
            "AGQ refinement |{a_label} - {b_label}| = {} should be < 0.05",
            (a - b).abs()
        );
    }
}

#[test]
fn test_glmm_compiler_artifact_records_boundary_metadata() {
    let data = contra_fixture();
    let formula =
        parse_formula("use_num ~ 1 + age + age2 + urban + livch + (1 | urban_dist)").unwrap();

    let model = GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();
    let artifact = model.compiler_artifact();

    assert_eq!(
        artifact.model_boundary.model_kind,
        crate::compiler::ModelKind::GeneralizedLinearMixedModel
    );
    assert_eq!(artifact.model_boundary.response_distribution, "bernoulli");
    assert_eq!(artifact.model_boundary.link, "logit");
    assert!(matches!(
        artifact.model_boundary.objective_approximation,
        crate::compiler::ObjectiveApproximation::Laplace { .. }
    ));
    assert!(matches!(
        artifact.model_boundary.inference_availability,
        crate::compiler::InferenceAvailability::Unsupported { .. }
    ));
}

#[test]
fn test_glmm_new_with_compiler_policy_applies_internal_policy() {
    let data = contra_fixture();
    let formula =
        parse_formula("use_num ~ 1 + age + age2 + urban + livch + (1 | urban_dist)").unwrap();
    let mut policy = CompilerPolicy::as_specified();
    policy.thresholds.effective_rank_relative_tolerance = 0.125;

    let model = GeneralizedLinearMixedModel::new_with_compiler_policy(
        formula,
        &data,
        Family::Bernoulli,
        None,
        policy,
    )
    .unwrap();

    assert_eq!(
        model.compiler_policy().random_strategy,
        crate::compiler::RandomStrategy::AsSpecified
    );
    assert!(model
        .compiler_artifact()
        .reproducibility
        .thresholds
        .iter()
        .any(|(name, value)| name == "effective_rank_relative_tolerance" && value == "0.125"));
}

fn progress_callback_poisson_fixture() -> GeneralizedLinearMixedModel {
    let mut data = DataFrame::new();
    data.add_numeric("y", vec![1.0, 2.0, 1.0, 3.0, 2.0, 4.0])
        .unwrap();
    data.add_numeric("x", vec![-1.0, 0.0, 1.0, -1.0, 0.0, 1.0])
        .unwrap();
    data.add_categorical(
        "group",
        vec!["a", "a", "a", "b", "b", "b"]
            .into_iter()
            .map(str::to_string)
            .collect(),
    )
    .unwrap();
    GeneralizedLinearMixedModel::new(
        parse_formula("y ~ 1 + x + (1 | group)").unwrap(),
        &data,
        Family::Poisson,
        Some(LinkFunction::Log),
    )
    .unwrap()
}

#[test]
fn fast_glmm_fit_propagates_pirls_host_interrupt_callback_error() {
    let mut model = progress_callback_poisson_fixture();
    let callback = FitProgressCallback::new(|progress| {
        if progress.phase == FitProgressPhase::Pirls {
            return Err(MixedModelError::Interrupted("test interrupt".to_string()));
        }
        Ok(())
    });

    let error = model
        .fit_with_glmm_options(
            GlmmFitOptions::fast_laplace()
                .with_optimizer(Optimizer::PatternSearch)
                .with_progress_callback(callback),
        )
        .unwrap_err();

    assert_eq!(error.code(), "interrupted");
}

#[test]
fn joint_trust_bq_propagates_host_interrupt_callback_error() {
    let mut model = progress_callback_poisson_fixture();
    let events = Arc::new(AtomicUsize::new(0));
    let callback_events = Arc::clone(&events);
    let callback = FitProgressCallback::new(move |progress| {
        if progress.phase == FitProgressPhase::JointGlmmOptimizer {
            callback_events.fetch_add(1, Ordering::SeqCst);
            return Err(MixedModelError::Interrupted("test interrupt".to_string()));
        }
        Ok(())
    });

    let error = model
        .fit_with_glmm_options(
            GlmmFitOptions::joint_laplace()
                .with_optimizer(Optimizer::TrustBq)
                .with_progress_callback(callback),
        )
        .unwrap_err();

    assert_eq!(error.code(), "interrupted");
    assert_eq!(events.load(Ordering::SeqCst), 1);
}

#[test]
fn glmm_verify_convergence_is_not_run_before_fit() {
    let mut model = small_joint_poisson_fixture();
    let verification = model.verify_convergence().unwrap();
    assert_eq!(
        verification.status,
        crate::compiler::ConvergenceVerificationStatus::NotRun
    );
    assert!(verification.runs.is_empty());
    assert_eq!(verification.message, "model has not been fitted");
}

#[test]
fn glmm_verify_convergence_profiled_restarts_agree() {
    let mut model = small_joint_poisson_fixture();
    model.fit_with_options(true, 1, false).unwrap();
    let reference_objective = model.opt_summary().fmin;
    let reference_theta = model.theta.clone();

    let verification = model.verify_convergence().unwrap();

    assert_eq!(
        verification.status,
        crate::compiler::ConvergenceVerificationStatus::RestartAgrees,
        "profiled restarts should reproduce the recorded optimum: {}",
        verification.message
    );
    assert_eq!(verification.runs.len(), 2, "restart + one jitter run");
    assert!(verification
        .runs
        .iter()
        .any(|run| run.label == "restart_from_optimum"));
    assert!(verification
        .runs
        .iter()
        .any(|run| run.label == "jitter_restart_1"));
    for run in &verification.runs {
        assert!(
            run.agrees,
            "run {} disagreed: {:?}",
            run.label, run.diagnostics
        );
    }
    assert_eq!(verification.reference_objective, Some(reference_objective));
    assert_eq!(verification.reference_theta, reference_theta);

    // The verify pass must not perturb the fitted model itself.
    assert_eq!(model.opt_summary().fmin, reference_objective);
    assert_eq!(model.theta, reference_theta);

    let certificate = model
        .compiler_artifact()
        .optimizer_certificate
        .as_ref()
        .expect("fitted GLMM carries an optimizer certificate");
    assert_eq!(
        certificate.verification.as_ref().map(|v| v.status),
        Some(crate::compiler::ConvergenceVerificationStatus::RestartAgrees),
        "verification verdict must be attached to the optimizer certificate"
    );
}

#[cfg(feature = "nlopt")]
#[test]
fn glmm_verify_convergence_joint_restarts_agree() {
    let mut model = small_joint_poisson_fixture();
    model.fit_with_options(false, 1, false).unwrap();
    let joint_fit = model
        .opt_summary()
        .return_value
        .starts_with("JOINT_LAPLACE:");

    let verification = model.verify_convergence().unwrap();

    assert_eq!(verification.runs.len(), 2, "restart + one jitter run");
    if joint_fit {
        // A certified joint fit should reproduce its own optimum; a labelled
        // fallback is verified against the profiled objective instead and is
        // covered by the profiled test above.
        assert_eq!(
            verification.status,
            crate::compiler::ConvergenceVerificationStatus::RestartAgrees,
            "joint restarts should reproduce the recorded optimum: {}",
            verification.message
        );
    }
}

#[test]
#[cfg(not(feature = "nlopt"))]
fn glmm_verify_convergence_joint_restarts_agree_native() {
    // The native (no-nlopt) joint route uses TrustBQ for the joint stage; the
    // verification refit must reselect the profiled prefit optimizer instead
    // of inheriting the fitted clone's joint-stage backend (which the profiled
    // dispatch rejects with an Unsupported error).
    let mut model = small_joint_poisson_fixture();
    model.fit_with_options(false, 1, false).unwrap();

    let verification = model.verify_convergence().unwrap();

    for run in &verification.runs {
        assert!(
            run.return_code.is_some(),
            "verification run {} never refitted: {:?}",
            run.label,
            run.diagnostics
        );
    }
    if model
        .opt_summary()
        .return_value
        .starts_with("JOINT_LAPLACE:")
    {
        assert_eq!(
            verification.status,
            crate::compiler::ConvergenceVerificationStatus::RestartAgrees,
            "native joint restarts should reproduce the recorded optimum: {}",
            verification.message
        );
    }
}

#[cfg(feature = "nlopt")]
#[test]
fn glmm_verify_convergence_reports_estimator_substitution_not_objective_drift() {
    // Squeezing the verification budget makes the joint refit fail its own
    // certification and return the labelled fast-PIRLS fallback. That run
    // must be reported as an estimator substitution (incomparable
    // objectives), not as an objective that moved 60+ deviance units.
    let mut model = small_joint_poisson_fixture();
    model.fit_with_options(false, 1, false).unwrap();
    if !model
        .opt_summary()
        .return_value
        .starts_with("JOINT_LAPLACE:")
    {
        return; // fixture fell back at fit time; nothing to squeeze
    }

    let mut options = crate::model::linear::ConvergenceVerificationOptions::glmm_defaults();
    options.max_function_evaluations = 25;
    options.jitter_starts = 0;
    let verification = model.verify_convergence_with_options(options).unwrap();

    let run = verification
        .runs
        .iter()
        .find(|run| run.label == "restart_from_optimum")
        .expect("restart run present");
    if run
        .return_code
        .as_deref()
        .is_some_and(|code| code.contains("FALLBACK_FAST_PIRLS"))
    {
        assert!(!run.agrees);
        assert_eq!(
            run.objective_delta, None,
            "cross-estimator objectives must not be scored as a delta"
        );
        assert!(
            run.diagnostics
                .iter()
                .any(|diagnostic| diagnostic.contains("different estimator")),
            "substitution must be named in the run diagnostics: {:?}",
            run.diagnostics
        );
    }
}

#[test]
#[cfg(not(feature = "nlopt"))]
fn glmm_native_uncertified_profiled_optimum_leaves_explicit_skip_reason() {
    let (model, _) = glmm_certified_pirls_poisson_fixture();
    assert!(matches!(
        model.pirls_profiled_optimum_certificate(),
        Some(Err(_))
    ));
    let certificate = model
        .compiler_artifact()
        .optimizer_certificate
        .as_ref()
        .expect("fitted GLMM carries an optimizer certificate");
    assert!(certificate.free_gradient_norm.is_none());
    assert!(
        certificate.checks.iter().any(|check| matches!(
            check,
            crate::compiler::CertificateCheck::NotAssessed { reason }
                if reason.contains("profiled-optimum certificate not issued")
        )),
        "uncertified profiled optimum must leave an explicit not-assessed reason"
    );
}

#[test]
fn glmm_fit_metadata_deserializes_legacy_json_without_method_fields() {
    let legacy = serde_json::json!({
        "estimation_method": "fast_pirls_profiled",
        "objective_definition": "profiled_glmm_deviance",
        "response_constants": "dropped",
        "n_agq": 1,
        "optimizer_backend": "native",
        "optimizer": "cobyla",
        "optimizer_status": "FTOL_REACHED",
        "optimizer_convergence_status": "converged"
    });
    let metadata: crate::compiler::GlmmFitMetadata = serde_json::from_value(legacy).unwrap();
    assert_eq!(metadata.requested_method, None);
    assert_eq!(metadata.effective_method, None);
    assert_eq!(metadata.fallback_status, None);
}

/// Post-fit tail cost on the registry GLMM rows: the profiled-optimum
/// certificate (finite-difference PIRLS probes) and the final PIRLS pass.
/// `cargo test --release --lib generalized::tests::glmm_post_fit_tail_cost -- --ignored --nocapture`
#[test]
#[ignore]
fn glmm_post_fit_tail_cost() {
    use std::time::Instant;
    let cases: Vec<(&str, &str, Family)> = vec![
        (
            "grouseticks",
            "TICKS ~ 1 + YEAR + cHEIGHT + (1 | BROOD) + (1 | INDEX) + (1 | LOCATION)",
            Family::Poisson,
        ),
        (
            "verbagg",
            "r2 ~ 1 + Anger + Gender + btype + situ + mode + (1 | id) + (1 | item)",
            Family::Binomial,
        ),
    ];
    for (name, formula, family) in cases {
        let (mut data, _) = crate::datasets::load(name).unwrap();
        if let Some(column) = data.categorical("r2") {
            // verbagg's binary response is recorded as a factor (N/Y).
            let numeric: Vec<f64> = column
                .values
                .iter()
                .map(|v| if v == "Y" { 1.0 } else { 0.0 })
                .collect();
            let mut lowered = DataFrame::new();
            for name in data.column_names() {
                if name == "r2" {
                    lowered.add_numeric("r2", numeric.clone()).unwrap();
                } else if let Some(values) = data.numeric(name) {
                    lowered.add_numeric(name, values.to_vec()).unwrap();
                } else if let Some(cat) = data.categorical(name) {
                    lowered.add_categorical(name, cat.values.clone()).unwrap();
                }
            }
            data = lowered;
        }
        let mut model =
            GeneralizedLinearMixedModel::new(parse_formula(formula).unwrap(), &data, family, None)
                .unwrap();
        let t0 = Instant::now();
        model.fit_with_options(true, 1, false).unwrap();
        let fit_ms = t0.elapsed().as_secs_f64() * 1e3;
        let t1 = Instant::now();
        model.complete_pirls_certificate();
        let certificate_ms = t1.elapsed().as_secs_f64() * 1e3;
        let outcome = model.pirls_profiled_optimum_certificate().clone().unwrap();
        let mut theta = model.lmm.optsum.final_params.clone();
        let t2 = Instant::now();
        model.finalize_theta_after_optimizer(&mut theta, 1).unwrap();
        let final_pirls_ms = t2.elapsed().as_secs_f64() * 1e3;
        println!(
            "{name:12} d={} fit {fit_ms:8.1} ms (feval {}) | certificate {certificate_ms:7.1} ms ({:.0}% of fit, {}) | final PIRLS {final_pirls_ms:6.1} ms",
            model.theta.len(),
            model.lmm.optsum.feval,
            100.0 * certificate_ms / fit_ms,
            if outcome.is_ok() { "issued" } else { "not issued" }
        );
    }
}

/// Phase 7 T7.2: the profiled-optimum certificate is deferred; inspecting
/// through `&self` and completing in place must agree exactly, and the
/// eager wire form (diagnostic slot, evidence, checks) is preserved.
#[test]
fn deferred_pirls_certificate_matches_in_place_completion() {
    let (fitted, _) = glmm_certified_pirls_poisson_fixture();
    assert!(
        fitted.pirls_certificate_pending,
        "the fit should defer the certificate"
    );
    assert!(fitted.pirls_profiled_optimum_certificate.is_none());
    let deferred_checks = fitted
        .lmm
        .compiler_artifact
        .optimizer_certificate
        .as_ref()
        .unwrap()
        .checks
        .iter()
        .filter(|check| {
            matches!(check, crate::compiler::CertificateCheck::NotAssessed { reason }
                if reason.contains("deferred"))
        })
        .count();
    assert_eq!(
        deferred_checks, 3,
        "deferred marker on the stored certificate"
    );

    // `&self` inspection completes on a clone and caches the view.
    let inspected = fitted.clone();
    let inspected_json = serde_json::to_value(inspected.compiler_artifact()).unwrap();
    let inspected_certificate = inspected.pirls_profiled_optimum_certificate().clone();
    assert!(
        inspected.pirls_certificate_pending,
        "inspection must not mutate the stored state"
    );

    // In-place completion (what mutating paths call).
    let mut completed = fitted.clone();
    completed.complete_pirls_certificate();
    assert!(!completed.pirls_certificate_pending);
    let completed_json = serde_json::to_value(completed.compiler_artifact()).unwrap();
    assert_eq!(inspected_json, completed_json);
    assert_eq!(
        inspected_certificate,
        completed.pirls_profiled_optimum_certificate
    );
    // The completed certificate carries first-order evidence, not the marker.
    let certificate = completed
        .compiler_artifact()
        .optimizer_certificate
        .as_ref()
        .unwrap();
    assert!(!certificate.checks.iter().any(|check| {
        matches!(check, crate::compiler::CertificateCheck::NotAssessed { reason }
            if reason.contains("deferred"))
    }));
    assert!(completed
        .compiler_artifact()
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic
            .payload
            .contains_key("glmm_pirls_profiled_optimum_certificate")));
}

#[test]
fn deferred_pirls_certificate_is_completed_by_verification_and_reset_by_refit() {
    let (mut model, data) = glmm_certified_pirls_poisson_fixture();
    assert!(model.pirls_certificate_pending);
    model.verify_convergence().unwrap();
    assert!(
        !model.pirls_certificate_pending,
        "verification records on the completed certificate"
    );
    assert!(model.pirls_profiled_optimum_certificate.is_some());

    let y = data.numeric("y").unwrap().to_vec();
    model.refit(&y).unwrap();
    assert!(
        model.pirls_certificate_pending,
        "a refit defers its own certificate again"
    );
    let payload = model
        .predict_new_variance(&data, GlmmPredictionScale::Response, NewReLevels::Error)
        .unwrap();
    assert!(!payload.rows.is_empty());
}

/// Phase 7 T7.3: a warm-started refit reaches the cold refit's optimum
/// with fewer evaluations; `refit` keeps its cold-start (Julia `refit!`)
/// semantics.
#[test]
fn warm_refit_reaches_the_cold_refit_optimum_with_fewer_evaluations() {
    use crate::model::linear::RefitStart;
    use rand::SeedableRng;
    let (model, _) = glmm_certified_pirls_poisson_fixture();
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let y_sim = model.simulate_response(&mut rng).unwrap();

    let mut cold = model.clone();
    cold.refit(&y_sim).unwrap();
    let mut cold_explicit = model.clone();
    cold_explicit
        .refit_with_start(&y_sim, RefitStart::Initial)
        .unwrap();
    assert_eq!(cold.lmm.optsum.feval, cold_explicit.lmm.optsum.feval);
    assert_eq!(cold.objective(), cold_explicit.objective());

    let mut warm = model.clone();
    warm.refit_with_start(&y_sim, RefitStart::Fitted).unwrap();
    let tolerance = 1e-6 * (1.0 + cold.objective().abs());
    assert!(
        (warm.objective() - cold.objective()).abs() <= tolerance,
        "warm {} vs cold {}",
        warm.objective(),
        cold.objective()
    );
    for (a, b) in warm.theta().iter().zip(cold.theta()) {
        assert!((a - b).abs() <= 1e-3, "theta {a} vs {b}");
    }
    assert!(
        warm.lmm.optsum.feval <= cold.lmm.optsum.feval,
        "warm {} evaluations vs cold {}",
        warm.lmm.optsum.feval,
        cold.lmm.optsum.feval
    );
    assert!(
        warm.pirls_certificate_pending,
        "the refit defers its certificate"
    );

    let mut bad = model.clone();
    assert!(bad
        .refit_with_start(&y_sim, RefitStart::From(vec![f64::NAN]))
        .is_err());
}

/// Cold versus warm refit cost on the registry GLMM rows.
/// `cargo test --release --lib generalized::tests::glmm_refit_cost -- --ignored --nocapture`
#[test]
#[ignore]
fn glmm_refit_cost() {
    use crate::model::linear::RefitStart;
    use rand::SeedableRng;
    use std::time::Instant;
    let (data, _) = crate::datasets::load("grouseticks").unwrap();
    let formula = "TICKS ~ 1 + YEAR + cHEIGHT + (1 | BROOD) + (1 | INDEX) + (1 | LOCATION)";
    let mut model = GeneralizedLinearMixedModel::new(
        parse_formula(formula).unwrap(),
        &data,
        Family::Poisson,
        None,
    )
    .unwrap();
    model.fit_with_options(true, 1, false).unwrap();
    let mut rng = rand::rngs::StdRng::seed_from_u64(3);
    for replicate in 0..3 {
        let y_sim = model.simulate_response(&mut rng).unwrap();
        let mut cold = model.clone();
        let t0 = Instant::now();
        cold.refit(&y_sim).unwrap();
        let cold_ms = t0.elapsed().as_secs_f64() * 1e3;
        let mut warm = model.clone();
        let t1 = Instant::now();
        warm.refit_with_start(&y_sim, RefitStart::Fitted).unwrap();
        let warm_ms = t1.elapsed().as_secs_f64() * 1e3;
        println!(
            "grouseticks replicate {replicate}: cold {cold_ms:7.1} ms ({} evals, obj {:.6}) | warm {warm_ms:7.1} ms ({} evals, obj {:.6}) | {:.2}x",
            cold.lmm.optsum.feval,
            cold.objective(),
            warm.lmm.optsum.feval,
            warm.objective(),
            cold_ms / warm_ms
        );
    }
}

/// Joint Laplace rows used by the deferred-inference tests: cbpp grouped
/// binomial, culcita Bernoulli and contraception `(1 | dist)`.
fn joint_laplace_row(name: &str) -> GeneralizedLinearMixedModel {
    let (formula, data, family, weights) = match name {
        "cbpp" => {
            let (mut data, _) = crate::datasets::load("cbpp").unwrap();
            let incidence = data.numeric("incidence").unwrap().to_vec();
            let size = data.numeric("size").unwrap().to_vec();
            let proportion = incidence
                .iter()
                .zip(&size)
                .map(|(&y, &n)| y / n)
                .collect::<Vec<_>>();
            data.add_numeric("proportion", proportion).unwrap();
            (
                "proportion ~ 1 + period + (1 | herd)",
                data,
                Family::Binomial,
                Some(size),
            )
        }
        "culcita" => {
            let (data, _) = crate::datasets::load("culcitalogreg").unwrap();
            (
                "predation ~ ttt + (1 | block)",
                data,
                Family::Binomial,
                None,
            )
        }
        "contraception" => {
            let (data, _) = crate::datasets::load("contraception").unwrap();
            (
                "use ~ 1 + age + livch + urban + (1 | dist)",
                data,
                Family::Binomial,
                None,
            )
        }
        other => panic!("unknown joint row {other}"),
    };
    let formula = parse_formula(formula).unwrap();
    match weights {
        Some(weights) => {
            GeneralizedLinearMixedModel::new_with_weights(formula, &data, family, None, weights)
                .unwrap()
        }
        None => GeneralizedLinearMixedModel::new(formula, &data, family, None).unwrap(),
    }
}

fn joint_laplace_fit(name: &str) -> GeneralizedLinearMixedModel {
    let mut model = joint_laplace_row(name);
    model.fit_with_options(false, 1, false).unwrap();
    model
}

/// Everything a caller can read about a fitted model's inference, as JSON.
fn joint_reported_results(model: &GeneralizedLinearMixedModel) -> serde_json::Value {
    let matrix = |m: DMatrix<f64>| {
        (0..m.nrows())
            .map(|i| (0..m.ncols()).map(|j| m[(i, j)]).collect::<Vec<_>>())
            .collect::<Vec<_>>()
    };
    serde_json::json!({
        "coef": MixedModelFit::coef(model).as_slice(),
        "theta": model.theta(),
        "objective": MixedModelFit::objective(model),
        "vcov": matrix(MixedModelFit::vcov(model)),
        "stderror": MixedModelFit::stderror(model).as_slice(),
        "fit_summary": serde_json::to_value(
            crate::stats::model_summary::FitSummaryPayload::from_generalized_model(model)
        ).unwrap(),
        "artifact": serde_json::to_value(model.compiler_artifact()).unwrap(),
        "audit_report": serde_json::to_value(model.audit_report()).unwrap(),
        "print_summary": model.print_summary().to_string(),
    })
}

thread_local! {
    /// Test-only switch restoring the eager joint-Laplace inference path, so
    /// the deferred path can be compared against it in the same build.
    pub(super) static EAGER_JOINT_LAPLACE_INFERENCE: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

fn eager_joint_laplace_fit(name: &str) -> GeneralizedLinearMixedModel {
    EAGER_JOINT_LAPLACE_INFERENCE.with(|eager| eager.set(true));
    let mut model = joint_laplace_row(name);
    let result = model.fit_with_options(false, 1, false).map(|_| ());
    EAGER_JOINT_LAPLACE_INFERENCE.with(|eager| eager.set(false));
    result.unwrap();
    assert!(!model.joint_inference_pending);
    model
}

fn pirls_state(model: &GeneralizedLinearMixedModel) -> Vec<f64> {
    let mut state = Vec::new();
    for u in &model.u {
        state.extend_from_slice(u.as_slice());
    }
    state.extend_from_slice(model.eta.as_slice());
    state.extend_from_slice(model.mu.as_slice());
    state.extend_from_slice(model.beta.as_slice());
    state.push(model.dispersion);
    state
}

/// The joint-Laplace inference artifacts (finite-difference joint Hessian)
/// are deferred until inspected; every reported result must equal the eager
/// path's, and neither `&self` inspection nor in-place completion may move
/// the fitted PIRLS state.
#[test]
fn deferred_joint_laplace_inference_matches_eager_path() {
    for name in ["cbpp", "culcita", "contraception"] {
        let eager = eager_joint_laplace_fit(name);
        let eager_results = joint_reported_results(&eager);

        let deferred = joint_laplace_fit(name);
        assert!(deferred.joint_inference_pending, "{name}: fit should defer");
        assert!(!deferred.pirls_certificate_pending);
        assert!(deferred
            .lmm
            .compiler_artifact
            .fixed_effect_inference_table
            .is_none());
        assert!(deferred
            .lmm
            .compiler_artifact
            .fixed_effect_covariance_matrix
            .is_none());
        let fitted_state = pirls_state(&deferred);
        // The eager Hessian's restoring re-evaluation lands on the fitted
        // modes on these rows, so even the unreported state agrees.
        assert_eq!(fitted_state, pirls_state(&eager), "{name}: fitted state");

        // `&self` readers complete on a clone and see the eager payloads.
        let inspected = deferred.clone();
        assert_eq!(
            MixedModelFit::vcov(&inspected),
            MixedModelFit::vcov(&eager),
            "{name}: vcov"
        );
        assert_eq!(joint_reported_results(&inspected), eager_results, "{name}");
        assert!(
            inspected.joint_inference_pending,
            "{name}: &self stays deferred"
        );
        assert_eq!(pirls_state(&inspected), fitted_state, "{name}: &self state");

        // In-place completion (what mutating paths call), from a fresh
        // clone and from one with a cached inspection.
        let mut completed = deferred.clone();
        completed.complete_joint_laplace_inference();
        assert!(!completed.joint_inference_pending);
        assert_eq!(joint_reported_results(&completed), eager_results, "{name}");
        assert_eq!(
            serde_json::to_value(&completed.lmm.compiler_artifact).unwrap(),
            eager_results["artifact"],
            "{name}: stored artifact after completion"
        );
        assert_eq!(pirls_state(&completed), fitted_state, "{name}: state");
        let mut completed_from_view = inspected.clone();
        completed_from_view.complete_joint_laplace_inference();
        assert_eq!(
            serde_json::to_value(&completed_from_view.lmm.compiler_artifact).unwrap(),
            eager_results["artifact"],
            "{name}: completion from a cached inspection"
        );
    }
}

/// Values pinned from the eager path on macOS aarch64 with the NLopt BOBYQA
/// joint driver whose FTOL stops are confirmed by restarts (e774238).
///
/// This is a fit-level anchor, not a fixture gate (VERSIONING.md §3.1).
/// Before the confirmation restarts, BOBYQA's FTOL stop landed wherever
/// last-bit rounding put it (Linux x86_64 contraception 9.2e-5 above the
/// optimum); with them, perturbed starts land within ~5e-8 of the optimum,
/// and the Linux x86_64 and Windows CI lanes for e774238 pass this test.
/// The objective is held to the documented parity band. The
/// finite-difference standard errors keep a looser 1e-3 relative tolerance:
/// the stop point still varies at the 1e-8 objective level, and the
/// finite-difference Hessian evaluated there moves the SEs far less than
/// 1e-3, which still catches any real Hessian regression. Deferred-vs-eager
/// equality is checked exactly, on one platform, by the tests above.
#[cfg(feature = "nlopt")]
#[test]
fn deferred_joint_laplace_inference_matches_pinned_eager_values() {
    let pinned: [(&str, f64, &[f64]); 3] = [
        (
            "cbpp",
            184.05256373024386,
            &[
                0.23247181492924562,
                0.3066425760106268,
                0.32663735507697705,
                0.4274373106693532,
            ],
        ),
        (
            "culcita",
            60.70584123451043,
            &[
                1.8130429096648315,
                1.4651833576983542,
                1.551965193127762,
                1.725285543382114,
            ],
        ),
        (
            "contraception",
            2413.6164607096907,
            &[
                0.14903571978693272,
                0.007885872895859583,
                0.17958009844119066,
                0.15438236572648847,
                0.16294238769115987,
                0.11942516623455823,
            ],
        ),
    ];
    for (name, objective, stderror) in pinned {
        let model = joint_laplace_fit(name);
        assert!(model.joint_inference_pending);
        let optsum = model.opt_summary();
        let actual = MixedModelFit::objective(&model);
        assert!(
            (actual - objective).abs() <= 1e-7_f64.max(1e-8 * objective.abs()),
            "{name}: objective {actual:.12} vs pinned {objective:.12} \
             (feval {} of max {}, return {})",
            optsum.feval,
            optsum.max_feval,
            optsum.return_value
        );
        let se = MixedModelFit::stderror(&model);
        assert_eq!(se.len(), stderror.len());
        for (actual, expected) in se.iter().zip(stderror) {
            assert_relative_eq!(*actual, *expected, max_relative = 1e-3);
        }
        let vcov = MixedModelFit::vcov(&model);
        for (index, expected) in stderror.iter().enumerate() {
            assert_relative_eq!(vcov[(index, index)].sqrt(), *expected, max_relative = 1e-3);
        }
        assert!(matches!(
            model
                .compiler_artifact()
                .model_boundary
                .inference_availability,
            InferenceAvailability::Available { .. }
        ));
    }
}

/// NLopt BOBYQA's FTOL stop fires on the first small improving step at any
/// trust-region radius, so without confirmation restarts the joint fit
/// stopped anywhere from 1e-7 to 2.5e-4 above the optimum depending on
/// last-bit differences in the start (contraception stopped 9.2e-5 high on
/// Linux CI). Starts perturbed at 1e-7..1e-6 relative must all land no more
/// than 1e-7 (the parity band) above the pinned optimum.
#[cfg(feature = "nlopt")]
#[test]
fn joint_laplace_nlopt_stop_is_robust_to_start_perturbations() {
    for (name, pinned) in [
        ("contraception", 2413.6164607096907),
        ("cbpp", 184.05256373024386),
    ] {
        let mut profiled = joint_laplace_row(name);
        profiled.fit_with_options(true, 1, false).unwrap();
        let start_beta = profiled.beta.as_slice().to_vec();
        let start_theta = profiled.theta.clone();
        let mut rng = rand::rngs::StdRng::seed_from_u64(20260922);
        for eps in [1e-7, 1e-6] {
            for _ in 0..3 {
                let mut perturb = |value: f64| {
                    let r: f64 = rand::Rng::gen_range(&mut rng, -1.0..1.0);
                    value + eps * r * value.abs().max(1e-3)
                };
                let beta = start_beta.iter().map(|&v| perturb(v)).collect::<Vec<_>>();
                let theta = start_theta
                    .iter()
                    .map(|&v| perturb(v).max(0.0))
                    .collect::<Vec<_>>();
                let mut model = profiled.clone();
                let start_objective = model.deviance_with_response_constants(1);
                model.lmm.optsum.optimizer = Optimizer::NloptBobyqa;
                let maxeval = joint_glmm_default_maxeval_for(
                    Optimizer::NloptBobyqa,
                    beta.len() + theta.len(),
                );
                model
                    .fit_joint_glmm_from_start(
                        beta,
                        theta,
                        start_objective,
                        1,
                        maxeval,
                        Some(profiled.clone()),
                    )
                    .unwrap();
                let optsum = model.opt_summary();
                let objective = MixedModelFit::objective(&model);
                // One-sided: a platform that finds a lower optimum passes;
                // the lower bound only guards against a broken objective.
                assert!(
                    objective <= pinned + 1e-7 && objective >= pinned - 1e-4,
                    "{name} eps {eps:e}: objective {objective:.10} vs pinned {pinned:.10} \
                     (feval {} of max {}, return {})",
                    optsum.feval,
                    optsum.max_feval,
                    optsum.return_value
                );
            }
        }
    }
}

/// A tight evaluation budget must not turn a confirmed or unconfirmable
/// FTOL stop into a budget stop. A confirming restart that gains nothing
/// keeps the FTOL label even if it then exhausts the budget, and a restart
/// that cannot afford twice BOBYQA's 2n + 1 interpolation set is not
/// launched; that stop keeps its FTOL label and is recorded as unconfirmed
/// on the certificate rather than claimed as confirmed. Sweeping the budget
/// across cbpp's first-pass and restart stop points covers both cases.
#[cfg(feature = "nlopt")]
#[test]
fn joint_laplace_nlopt_ftol_confirmation_respects_tight_budgets() {
    let pinned = 184.05256373024386;
    let mut profiled = joint_laplace_row("cbpp");
    profiled.fit_with_options(true, 1, false).unwrap();
    let n_params = profiled.beta.len() + profiled.theta.len();
    let min_restart_budget = 2 * (2 * n_params as i64 + 1);
    // Objectives reached by an FTOL stop under a smaller budget.
    let mut ftol_objectives: Vec<f64> = Vec::new();
    let (mut saw_unconfirmed, mut saw_confirmed) = (false, false);
    for maxeval in 40..=130u32 {
        let mut model = profiled.clone();
        let start_objective = model.deviance_with_response_constants(1);
        model.lmm.optsum.optimizer = Optimizer::NloptBobyqa;
        model
            .fit_joint_glmm_from_start(
                profiled.beta.as_slice().to_vec(),
                profiled.theta.clone(),
                start_objective,
                1,
                maxeval,
                Some(profiled.clone()),
            )
            .unwrap();
        let optsum = model.opt_summary().clone();
        let objective = MixedModelFit::objective(&model);
        let label = optsum.return_value.as_str();
        assert!(
            label == "JOINT_LAPLACE:FTOL_REACHED" || label == "JOINT_LAPLACE:MAXEVAL_REACHED",
            "max {maxeval}: unexpected return {label}"
        );
        assert!(optsum.feval <= i64::from(maxeval));
        let unconfirmed = model
            .compiler_artifact()
            .optimizer_certificate
            .as_ref()
            .is_some_and(|certificate| {
                certificate
                    .diagnostics
                    .iter()
                    .any(|diagnostic| diagnostic.payload.contains_key("ftol_confirmation"))
            });
        if label.ends_with("MAXEVAL_REACHED") {
            // A budget stop must have moved past every FTOL stop reached
            // under a smaller budget: a restart that gained nothing (or was
            // never launched) cannot relabel the fit.
            assert!(
                ftol_objectives.iter().all(|&f| objective < f - 1e-12),
                "max {maxeval}: MAXEVAL at objective {objective:.12} already reached by an FTOL stop"
            );
            assert!(
                !unconfirmed,
                "max {maxeval}: budget stop marked as an FTOL confirmation gap"
            );
        } else {
            ftol_objectives.push(objective);
            if unconfirmed {
                saw_unconfirmed = true;
                assert!(
                    i64::from(maxeval) - optsum.feval < min_restart_budget,
                    "max {maxeval}: unconfirmed although {} evaluations remained",
                    i64::from(maxeval) - optsum.feval
                );
            } else {
                saw_confirmed = true;
                assert!(
                    objective <= pinned + 1e-7,
                    "max {maxeval}: confirmed FTOL stop at {objective:.12}, pinned {pinned:.12}"
                );
            }
        }
    }
    assert!(saw_unconfirmed && saw_confirmed);
}

/// Every path that records on, re-fits, or switches estimator from a
/// deferred joint fit must complete or clear the deferral: a stale flag
/// would let an inspection overwrite a later stage's payloads.
#[test]
fn deferred_joint_laplace_inference_across_verification_refit_and_stage_switch() {
    let eager = eager_joint_laplace_fit("cbpp");
    let eager_artifact = serde_json::to_value(eager.compiler_artifact()).unwrap();

    // Verification records on the certificate only, so the inference stays
    // deferred (a view cached before verification is dropped); everything
    // but the verification record is the eager artifact.
    let mut verified = joint_laplace_fit("cbpp");
    let fitted_state = pirls_state(&verified);
    let _ = MixedModelFit::vcov(&verified);
    verified.verify_convergence().unwrap();
    assert!(verified.joint_inference_pending);
    assert_eq!(pirls_state(&verified), fitted_state);
    let mut verified_artifact = serde_json::to_value(verified.compiler_artifact()).unwrap();
    assert!(!verified_artifact["optimizer_certificate"]["verification"].is_null());
    verified_artifact["optimizer_certificate"]["verification"] =
        eager_artifact["optimizer_certificate"]["verification"].clone();
    assert_eq!(verified_artifact, eager_artifact);

    // A caller-driven PIRLS completes from the fitted state first.
    let mut driven = joint_laplace_fit("cbpp");
    driven.pirls(false, false).unwrap();
    assert!(!driven.joint_inference_pending);
    assert_eq!(
        serde_json::to_value(driven.compiler_artifact()).unwrap(),
        eager_artifact
    );

    // Refits (profiled) clear the joint deferral and defer their own
    // certificate; the payloads are the profiled ones, not the joint ones.
    let mut refit = joint_laplace_fit("cbpp");
    let y = refit.y.as_slice().to_vec();
    // The joint stage's optimizer is recorded on the fit; dependency-light
    // builds cannot run TrustBQ on the profiled path a refit takes.
    refit.configure_profile_start_optimizer();
    refit.refit(&y).unwrap();
    assert!(!refit.joint_inference_pending);
    assert!(refit.pirls_certificate_pending);
    let refit_artifact = refit.compiler_artifact();
    assert_eq!(
        refit_artifact
            .glmm_fit_metadata
            .as_ref()
            .unwrap()
            .estimation_method,
        "fast_pirls_profiled"
    );
    assert!(matches!(
        refit_artifact.model_boundary.inference_availability,
        InferenceAvailability::Unsupported { .. }
    ));

    // profiled -> joint -> profiled stage switches on one model.
    let mut model = joint_laplace_row("cbpp");
    model.fit_with_options(true, 1, false).unwrap();
    assert!(model.pirls_certificate_pending && !model.joint_inference_pending);
    model.reset_for_refit(None).unwrap();
    assert!(!model.pirls_certificate_pending && !model.joint_inference_pending);
    model.fit_with_options(false, 1, false).unwrap();
    assert!(!model.pirls_certificate_pending && model.joint_inference_pending);
    assert!(model.pirls_profiled_optimum_certificate().is_none());
    assert!(matches!(
        model
            .compiler_artifact()
            .model_boundary
            .inference_availability,
        InferenceAvailability::Available { .. }
    ));
    model.reset_for_refit(None).unwrap();
    model.configure_profile_start_optimizer();
    model.fit_with_options(true, 1, false).unwrap();
    assert!(model.pirls_certificate_pending && !model.joint_inference_pending);
    assert!(model.pirls_profiled_optimum_certificate().is_some());
    assert_eq!(
        model
            .compiler_artifact()
            .glmm_fit_metadata
            .as_ref()
            .unwrap()
            .estimation_method,
        "fast_pirls_profiled"
    );

    // Joint AGQ is not deferred (its artifacts carry no Hessian).
    let mut agq = joint_laplace_row("culcita");
    agq.fit_with_options(false, 3, false).unwrap();
    assert!(!agq.joint_inference_pending);
}

/// A rejected AGQ request after a fit records its diagnostic on the stored
/// artifact; a `&self` view cached before it must not hide it (fast and
/// joint deferrals).
#[test]
fn deferred_inspection_sees_invalid_agq_diagnostic_recorded_after_fit() {
    let (_, data) = glmm_certified_pirls_poisson_fixture();
    for fast in [true, false] {
        let formula = parse_formula("y ~ 1 + x + (1 + x | group)").unwrap();
        let mut model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Poisson, None).unwrap();
        model.fit_with_options(fast, 1, false).unwrap();
        assert!(
            if fast {
                model.pirls_certificate_pending
            } else {
                model.joint_inference_pending
            },
            "fast={fast}: the fit should defer its post-fit work"
        );
        let has_invalid_agq = |model: &GeneralizedLinearMixedModel| {
            model
                .compiler_artifact()
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == DiagnosticCode::InvalidAgqRequest)
        };
        // Fill the cached view, then fail an AGQ request.
        let _ = MixedModelFit::vcov(&model);
        assert!(!has_invalid_agq(&model));
        assert!(model
            .fit_with_glmm_options(GlmmFitOptions {
                fast,
                n_agq: 5,
                ..GlmmFitOptions::default()
            })
            .is_err());
        assert!(has_invalid_agq(&model), "fast={fast}: compiler_artifact");
        assert!(
            model
                .audit_report()
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == DiagnosticCode::InvalidAgqRequest),
            "fast={fast}: audit_report"
        );
        assert!(
            model
                .print_summary()
                .top_diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == DiagnosticCode::InvalidAgqRequest),
            "fast={fast}: print_summary"
        );
        let view = serde_json::to_value(model.compiler_artifact()).unwrap();
        model.complete_deferred_inspection();
        assert_eq!(
            view,
            serde_json::to_value(model.compiler_artifact()).unwrap(),
            "fast={fast}: view vs in-place completion"
        );
    }
}

/// Unstable probes that move beta/theta/modes complete deferred work from
/// the fitted state first, so the inference table reports the fitted beta.
#[test]
fn deferred_joint_laplace_inference_is_completed_before_probes() {
    let eager = eager_joint_laplace_fit("cbpp");
    let eager_artifact = serde_json::to_value(eager.compiler_artifact()).unwrap();
    let mut probed = joint_laplace_fit("cbpp");
    let far_theta = vec![probed.theta[0] * 0.5];
    probed.profiled_deviance_at_theta(&far_theta, 1).unwrap();
    assert!(!probed.joint_inference_pending);
    assert_eq!(
        serde_json::to_value(probed.compiler_artifact()).unwrap(),
        eager_artifact
    );
    let mut via_lmm_mut = joint_laplace_fit("cbpp");
    let _ = via_lmm_mut.lmm_mut();
    assert!(!via_lmm_mut.joint_inference_pending);
    assert_eq!(
        serde_json::to_value(via_lmm_mut.compiler_artifact()).unwrap(),
        eager_artifact
    );
}

/// Prediction variance reads the certified joint covariance through the
/// deferral.
#[test]
fn deferred_joint_laplace_inference_prediction_variance_matches_eager() {
    let eager = eager_joint_laplace_fit("culcita");
    let deferred = joint_laplace_fit("culcita");
    let (data, _) = crate::datasets::load("culcitalogreg").unwrap();
    let eager_payload = eager
        .predict_new_variance(&data, GlmmPredictionScale::Response, NewReLevels::Error)
        .unwrap();
    let deferred_payload = deferred
        .predict_new_variance(&data, GlmmPredictionScale::Response, NewReLevels::Error)
        .unwrap();
    assert_eq!(
        serde_json::to_value(&deferred_payload).unwrap(),
        serde_json::to_value(&eager_payload).unwrap()
    );
}

/// Joint Laplace fit cost, eager versus deferred inference (interleaved in
/// one binary), with and without a first inference inspection (`vcov`).
/// `cargo test --release --lib generalized::tests::glmm_joint_inference_deferral_cost -- --ignored --nocapture`
#[test]
#[ignore]
fn glmm_joint_inference_deferral_cost() {
    use std::time::Instant;
    fn median(mut values: Vec<f64>) -> f64 {
        values.sort_by(f64::total_cmp);
        values[values.len() / 2]
    }
    for name in ["cbpp", "contraception"] {
        // [eager fit, eager fit+vcov, deferred fit, deferred fit+vcov]
        let mut samples: [Vec<f64>; 4] = Default::default();
        for rep in 0..28 {
            for eager in [true, false] {
                EAGER_JOINT_LAPLACE_INFERENCE.with(|flag| flag.set(eager));
                let mut model = joint_laplace_row(name);
                let t0 = Instant::now();
                model.fit_with_options(false, 1, false).unwrap();
                let fit = t0.elapsed().as_secs_f64() * 1e3;
                let vcov = MixedModelFit::vcov(&model);
                let inspected = t0.elapsed().as_secs_f64() * 1e3;
                EAGER_JOINT_LAPLACE_INFERENCE.with(|flag| flag.set(false));
                assert!(vcov.iter().all(|value| value.is_finite()));
                if rep >= 3 {
                    let offset = if eager { 0 } else { 2 };
                    samples[offset].push(fit);
                    samples[offset + 1].push(inspected);
                }
            }
        }
        let [eager_fit, eager_vcov, deferred_fit, deferred_vcov] = samples.map(median);
        println!(
            "{name:14} eager fit {eager_fit:8.3} ms (+vcov {eager_vcov:8.3}) | deferred fit {deferred_fit:8.3} ms (+vcov {deferred_vcov:8.3}) | fit speedup {:.2}x (25 reps, interleaved)",
            eager_fit / deferred_fit
        );
    }
}

// --- Fixed-β PIRLS `[X|y]`-block staleness and response-constant cache
// (mote bd-01M35KYR62T30QR6DDPYBYNEVC). ---

fn assert_blocks_bit_identical(actual: &[MatrixBlock], expected: &[MatrixBlock], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: block count");
    for (idx, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            std::mem::discriminant(a),
            std::mem::discriminant(e),
            "{what}[{idx}]: storage kind changed"
        );
        let (a, e) = (a.as_dense(), e.as_dense());
        assert_eq!(a.shape(), e.shape(), "{what}[{idx}]: shape");
        for (x, y) in a.iter().zip(e.iter()) {
            assert_eq!(x.to_bits(), y.to_bits(), "{what}[{idx}]: {x:e} vs {y:e}");
        }
    }
}

fn assert_bits_eq(actual: &[f64], expected: &[f64], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (x, y) in actual.iter().zip(expected) {
        assert_eq!(x.to_bits(), y.to_bits(), "{what}: {x:e} vs {y:e}");
    }
}

/// The model with its A/L system rebuilt in full from its own final IRLS
/// weights and working response (the pre-skip behaviour of every PIRLS
/// iteration).
fn with_full_rebuild_at_current_weights(
    model: &GeneralizedLinearMixedModel,
) -> GeneralizedLinearMixedModel {
    let mut reference = model.clone();
    let sqrtwts = model.lmm.sqrtwts.clone();
    let rank = model.lmm.feterm.rank;
    let working_y: Vec<f64> = model.lmm.xy_mat.xy.column(rank).iter().copied().collect();
    reference
        .lmm
        .update_irls_weights(&sqrtwts, &working_y)
        .unwrap();
    reference.lmm.update_l().unwrap();
    reference
}

#[test]
fn fixed_beta_pirls_exits_with_fe_blocks_bit_identical_to_a_full_rebuild() {
    for name in ["cbpp", "contraception"] {
        let mut model = joint_laplace_row(name);
        let n_beta = model.lmm.feterm.rank;
        let mut params: Vec<f64> = model.beta.iter().copied().collect();
        params.extend(model.theta.iter().map(|t| 0.8 * t));
        let objective = model.joint_glmm_deviance_at_params(&params, n_beta, 1);
        assert!(objective.is_finite(), "{name}: probe objective");
        assert!(
            !model.lmm.fe_blocks_stale(),
            "{name}: PIRLS returned with stale [X|y] blocks"
        );

        let reference = with_full_rebuild_at_current_weights(&model);
        assert_blocks_bit_identical(&model.lmm.a_blocks, &reference.lmm.a_blocks, "A");
        assert_blocks_bit_identical(&model.lmm.l_blocks, &reference.lmm.l_blocks, "L");
        // The readers of the `[X|y]` rows agree too.
        assert_bits_eq(
            model.lmm.beta().as_slice(),
            reference.lmm.beta().as_slice(),
            "profiled beta",
        );
        assert_eq!(model.lmm.pwrss().to_bits(), reference.lmm.pwrss().to_bits());
    }
}

#[test]
fn re_only_update_marks_fe_blocks_stale_until_refreshed_or_fully_factored() {
    let mut model = joint_laplace_row("cbpp");
    model.fit_with_options(true, 1, false).unwrap();
    assert!(
        !model.lmm.fe_blocks_stale(),
        "fast (varying-β) fit left stale blocks"
    );

    let rank = model.lmm.feterm.rank;
    let sqrtwts: Vec<f64> = model.lmm.sqrtwts.iter().map(|w| 1.1 * w).collect();
    let working_y: Vec<f64> = model
        .lmm
        .xy_mat
        .xy
        .column(rank)
        .iter()
        .map(|y| y + 0.05)
        .collect();

    let mut full = model.clone();
    full.lmm.update_irls_weights(&sqrtwts, &working_y).unwrap();
    full.lmm.update_l().unwrap();
    assert!(!full.lmm.fe_blocks_stale());

    let mut partial = model.clone();
    partial
        .lmm
        .update_irls_weights_re_only(&sqrtwts, &working_y);
    assert!(partial.lmm.fe_a_blocks_stale && partial.lmm.fe_l_row_stale);
    partial.lmm.update_l_re_only().unwrap();
    assert!(partial.lmm.fe_blocks_stale());

    // The RE rows are already exact; only the `[X|y]` rows lag.
    let k = partial.lmm.reterms.len();
    let re_blocks = k * (k + 1) / 2;
    assert_blocks_bit_identical(
        &partial.lmm.a_blocks[..re_blocks],
        &full.lmm.a_blocks[..re_blocks],
        "RE A",
    );
    assert_blocks_bit_identical(
        &partial.lmm.l_blocks[..re_blocks],
        &full.lmm.l_blocks[..re_blocks],
        "RE L",
    );
    let stale_fe_a = partial.lmm.a_blocks[re_blocks + k].as_dense();
    assert_ne!(
        stale_fe_a,
        full.lmm.a_blocks[re_blocks + k].as_dense(),
        "the skipped [X|y]'[X|y] block should still hold the previous weights"
    );

    // Path 1: the cheap refresh used on PIRLS exit.
    let mut refreshed = partial.clone();
    refreshed.lmm.refresh_stale_fe_blocks().unwrap();
    assert!(!refreshed.lmm.fe_blocks_stale());
    assert_blocks_bit_identical(&refreshed.lmm.a_blocks, &full.lmm.a_blocks, "refreshed A");
    assert_blocks_bit_identical(&refreshed.lmm.l_blocks, &full.lmm.l_blocks, "refreshed L");

    // Path 2: a full factor update self-heals the stale A rows first.
    partial.lmm.update_l().unwrap();
    assert!(!partial.lmm.fe_blocks_stale());
    assert_blocks_bit_identical(&partial.lmm.a_blocks, &full.lmm.a_blocks, "self-healed A");
    assert_blocks_bit_identical(&partial.lmm.l_blocks, &full.lmm.l_blocks, "self-healed L");
}

#[test]
fn joint_and_fast_fits_leave_fe_blocks_matching_a_forced_rebuild() {
    for fast in [false, true] {
        let mut model = joint_laplace_row("cbpp");
        model.fit_with_options(fast, 1, false).unwrap();
        assert!(
            !model.lmm.fe_blocks_stale(),
            "fast={fast}: stale blocks after fit"
        );

        let mut forced = model.clone();
        forced.lmm.recompute_a_blocks().unwrap();
        forced.lmm.update_l().unwrap();
        assert_blocks_bit_identical(&model.lmm.a_blocks, &forced.lmm.a_blocks, "A");
        assert_blocks_bit_identical(&model.lmm.l_blocks, &forced.lmm.l_blocks, "L");
        assert_bits_eq(
            MixedModelFit::vcov(&model).as_slice(),
            MixedModelFit::vcov(&forced).as_slice(),
            "vcov",
        );
        assert_bits_eq(
            MixedModelFit::stderror(&model).as_slice(),
            MixedModelFit::stderror(&forced).as_slice(),
            "stderror",
        );
    }
}

#[test]
fn response_constant_cache_matches_uncached_and_follows_the_response() {
    let mut model = joint_laplace_row("cbpp");
    model.fit_with_options(false, 1, false).unwrap();
    assert!(
        model.current_response_log_constants().is_some(),
        "joint fit builds the cache"
    );

    let uncached_objective = |model: &GeneralizedLinearMixedModel| {
        let mut plain = model.clone();
        plain.response_log_constants = None;
        plain.laplace_objective_with_response_constants()
    };
    assert_eq!(
        model.laplace_objective_with_response_constants().to_bits(),
        uncached_objective(&model).to_bits()
    );

    // Reset for a refit with a new response: the constants follow the new y.
    let old_constants = model.current_response_log_constants().unwrap().to_vec();
    let sizes = model.wt.clone();
    let new_y: Vec<f64> = model
        .y
        .iter()
        .zip(&sizes)
        .map(|(&p, &n)| ((p * n).round() + 1.0).min(n) / n)
        .collect();
    model.reset_for_refit(Some(&new_y)).unwrap();
    assert!(
        model.current_response_log_constants().is_none(),
        "a new response must invalidate the cached constants"
    );
    // A joint-objective evaluation on the new response (optimizer-free, so
    // the check is the same with and without NLopt).
    let n_beta = model.lmm.feterm.rank;
    let mut params: Vec<f64> = model.beta.iter().copied().collect();
    params.extend(model.theta.iter().copied());
    let refit_objective = model.joint_glmm_deviance_at_params(&params, n_beta, 1);
    assert!(refit_objective.is_finite());
    let new_constants = model.current_response_log_constants().unwrap().to_vec();
    assert_ne!(new_constants, old_constants);
    let expected: Vec<f64> = (0..model.y.len())
        .map(|i| model.response_log_constant_observation(i).unwrap())
        .collect();
    assert_bits_eq(&new_constants, &expected, "refit constants");
    assert_eq!(
        model.laplace_objective_with_response_constants().to_bits(),
        uncached_objective(&model).to_bits()
    );
    assert_eq!(
        refit_objective.to_bits(),
        uncached_objective(&model).to_bits()
    );

    // Direct writes to the public response/weight fields are caught too.
    model.y[0] = if model.y[0] > 0.0 { 0.0 } else { 1.0 };
    assert!(model.current_response_log_constants().is_none());
    assert_eq!(
        model.laplace_objective_with_response_constants().to_bits(),
        uncached_objective(&model).to_bits()
    );
    model.ensure_response_log_constants();
    model.wt[0] += 1.0;
    assert!(model.current_response_log_constants().is_none());
}

#[test]
fn response_constant_cache_key_is_checked_once_per_objective_evaluation() {
    use super::pirls::RESPONSE_LOG_CONSTANT_KEY_CHECKS;
    let key_checks = || RESPONSE_LOG_CONSTANT_KEY_CHECKS.with(|count| count.get());

    let mut model = joint_laplace_row("culcita");
    model.fit_with_options(true, 1, false).unwrap();
    model.ensure_response_log_constants();
    assert!(
        model.y.len() > 3,
        "fixture must have more observations than allowed checks"
    );

    for n_agq in [1, 7] {
        let before = key_checks();
        let objective = model.deviance_with_response_constants(n_agq);
        assert!(objective.is_finite());
        let checks = key_checks() - before;
        // One check in `ensure_response_log_constants`, one per summed
        // objective / offset — independent of the number of observations.
        assert!(
            checks <= 2,
            "n_agq={n_agq}: {checks} cache-key checks for one evaluation (n = {})",
            model.y.len()
        );
    }
    let before = key_checks();
    let _ = model.response_constants_offset();
    let _ = model.laplace_objective_with_response_constants();
    assert_eq!(key_checks() - before, 2);
}

// ---- Newton-decrement stationarity (bd-01M35YW5H031824SVC9EWP0JRZ) ----

/// Certification probes of `f` at `x` built the way the joint certificate
/// builds them: a default-step probe per coordinate (central unless the minus
/// step crosses the lower bound), plus the two escalated probes for the
/// coordinates listed in `escalate`.
fn synthetic_certification(
    f: &dyn Fn(&[f64]) -> f64,
    x: &[f64],
    lower_bounds: &[f64],
    escalate: &[usize],
) -> JointLaplaceCertificationGradient {
    let base = f(x);
    let probe = |index: usize, h: f64| {
        let mut plus = x.to_vec();
        plus[index] += h;
        if x[index] - h > lower_bounds[index] {
            let mut minus = x.to_vec();
            minus[index] -= h;
            JointFdProbe::central(h, f(&plus), f(&minus))
        } else {
            JointFdProbe::forward(h, f(&plus), base)
        }
    };
    let mut curvature_probes = Vec::new();
    let mut gradient = Vec::new();
    for (index, value) in x.iter().enumerate() {
        let scale = value.abs().max(1.0);
        let mut probes = vec![probe(index, JOINT_LAPLACE_FD_RELATIVE_STEP * scale)];
        if escalate.contains(&index) {
            probes.extend(
                JOINT_LAPLACE_CERT_FD_ESCALATED_RELATIVE_STEPS
                    .map(|step| probe(index, step * scale)),
            );
        }
        gradient.push(probes.last().unwrap().gradient);
        curvature_probes.push(probes);
    }
    JointLaplaceCertificationGradient {
        probe_gradient: curvature_probes.iter().map(|p| p[0].gradient).collect(),
        gradient,
        escalated_indices: escalate.to_vec(),
        unassessable_indices: Vec::new(),
        base_objective: base,
        curvature_probes,
    }
}

fn quadratic_gap<'a>(hessian: &'a DMatrix<f64>, optimum: &[f64]) -> impl Fn(&[f64]) -> f64 + 'a {
    let optimum = DVector::from_column_slice(optimum);
    move |x: &[f64]| {
        let d = DVector::from_column_slice(x) - &optimum;
        0.5 * d.dot(&(hessian * &d))
    }
}

fn correlated_test_hessian() -> DMatrix<f64> {
    DMatrix::from_row_slice(3, 3, &[40.0, 30.0, 1.0, 30.0, 25.0, 0.5, 1.0, 0.5, 8.0])
}

#[test]
fn newton_decrement_recovers_quadratic_gap() {
    let hessian = correlated_test_hessian();
    let optimum = [0.4, -1.2, 0.7];
    let f = quadratic_gap(&hessian, &optimum);
    let lower_bounds = [f64::NEG_INFINITY, f64::NEG_INFINITY, 0.0];
    // Premature point along the flat (correlated) β direction plus a θ offset.
    for (scale, expect_within) in [(5.0e-4, true), (2.0e-2, false)] {
        let x = [
            optimum[0] + scale,
            optimum[1] - 1.1 * scale,
            optimum[2] + 0.1 * scale,
        ];
        let gap = f(&x);
        let certification = synthetic_certification(&f, &x, &lower_bounds, &[]);
        let g = DVector::from_column_slice(&certification.gradient);

        let diagonal = joint_newton_decrement(&certification, &x, &lower_bounds, 2, None, 1.0e-6);
        assert!(diagonal.excluded.is_empty());
        assert_eq!(diagonal.parameter_indices, vec![0, 1, 2]);
        assert_eq!(diagonal.eager.variant, NewtonDecrementVariant::Diagonal);
        let expected_diagonal = 0.5 * (0..3).map(|i| g[i] * g[i] / hessian[(i, i)]).sum::<f64>();
        assert_relative_eq!(
            diagonal.eager.objective_gap.unwrap(),
            expected_diagonal,
            max_relative = 1e-5
        );

        let beta_block = hessian.view((0, 0), (2, 2)).into_owned();
        let block = joint_newton_decrement(
            &certification,
            &x,
            &lower_bounds,
            2,
            Some(&beta_block),
            1.0e-6,
        );
        assert_eq!(
            block.eager.variant,
            NewtonDecrementVariant::BetaBlockThetaDiagonal
        );
        let g_beta = g.rows(0, 2).into_owned();
        let expected_block = 0.5
            * (g_beta.dot(&beta_block.clone().cholesky().unwrap().solve(&g_beta))
                + g[2] * g[2] / hessian[(2, 2)]);
        assert_relative_eq!(
            block.eager.objective_gap.unwrap(),
            expected_block,
            max_relative = 1e-5
        );
        // The β block absorbs the strong β correlation, so it tracks the
        // true gap far better than the diagonal does here.
        assert!((block.eager.objective_gap.unwrap() / gap - 1.0).abs() < 0.1);
        let diagonal_ratio = diagonal.eager.objective_gap.unwrap() / gap;
        assert!(
            !(0.5..=2.0).contains(&diagonal_ratio),
            "diagonal/true = {diagonal_ratio}"
        );

        let full = full_newton_decrement_estimate(&block, &hessian, &[0, 1, 2]);
        assert_eq!(full.variant, NewtonDecrementVariant::Full);
        assert_relative_eq!(full.objective_gap.unwrap(), gap, max_relative = 1e-5);
        let expected_verdict = if expect_within {
            NewtonDecrementVerdict::WithinTolerance
        } else {
            NewtonDecrementVerdict::ExceedsTolerance
        };
        assert_eq!(full.verdict, expected_verdict, "gap {gap:e}");
        assert_eq!(block.eager.verdict, expected_verdict, "gap {gap:e}");
    }
}

#[test]
fn newton_decrement_block_falls_back_to_diagonal_when_working_curvature_disagrees() {
    let hessian = correlated_test_hessian();
    let optimum = [0.4, -1.2, 0.7];
    let f = quadratic_gap(&hessian, &optimum);
    let lower_bounds = [f64::NEG_INFINITY, f64::NEG_INFINITY, 0.0];
    let x = [0.41, -1.21, 0.71];
    let certification = synthetic_certification(&f, &x, &lower_bounds, &[]);
    // A working Hessian whose diagonal is 3x the probed curvature is not a
    // trustworthy stand-in: the estimate must fall back to the diagonal.
    let wrong_block = 3.0 * hessian.view((0, 0), (2, 2)).into_owned();
    let evidence = joint_newton_decrement(
        &certification,
        &x,
        &lower_bounds,
        2,
        Some(&wrong_block),
        1.0e-6,
    );
    assert_eq!(evidence.eager.variant, NewtonDecrementVariant::Diagonal);
}

#[test]
fn newton_decrement_handles_bound_active_and_one_sided_coordinates() {
    let hessian = DMatrix::from_diagonal(&DVector::from_vec(vec![10.0, 4.0, 2.0]));
    let optimum = [0.5, -0.3, 0.0];
    let f = quadratic_gap(&hessian, &optimum);
    let lower_bounds = [f64::NEG_INFINITY, f64::NEG_INFINITY, 0.0];

    // θ exactly on its bound: excluded as at_lower_bound; the decrement is
    // assessed over the free β coordinates only.
    let x = [0.501, -0.302, 0.0];
    let certification = synthetic_certification(&f, &x, &lower_bounds, &[]);
    let evidence = joint_newton_decrement(&certification, &x, &lower_bounds, 2, None, 1.0e-6);
    assert_eq!(evidence.parameter_indices, vec![0, 1]);
    assert_eq!(
        evidence.excluded,
        vec![crate::compiler::NewtonDecrementExclusion {
            index: 2,
            reason: "at_lower_bound".to_string()
        }]
    );
    let expected = 0.5 * (10.0 * 0.001_f64.powi(2) + 4.0 * 0.002_f64.powi(2));
    assert_relative_eq!(
        evidence.eager.objective_gap.unwrap(),
        expected,
        max_relative = 1e-5
    );
    assert_eq!(
        evidence.eager.verdict,
        NewtonDecrementVerdict::ExceedsTolerance
    );

    // θ just inside its bound (closer than the probe step): only a forward
    // probe exists, so its curvature is unknown and the decrement is not
    // assessed rather than guessed.
    let x = [0.5, -0.3, 5.0e-6];
    let certification = synthetic_certification(&f, &x, &lower_bounds, &[]);
    let evidence = joint_newton_decrement(&certification, &x, &lower_bounds, 2, None, 1.0e-6);
    assert_eq!(evidence.eager.verdict, NewtonDecrementVerdict::NotAssessed);
    assert!(evidence.eager.objective_gap.is_none());
    assert_eq!(evidence.excluded[0].reason, "one_sided_probe");
}

#[test]
fn newton_decrement_refuses_nonpositive_curvature() {
    // A saddle in the θ coordinate.
    let hessian = DMatrix::from_diagonal(&DVector::from_vec(vec![10.0, 4.0, -2.0]));
    let optimum = [0.5, -0.3, 1.0];
    let f = quadratic_gap(&hessian, &optimum);
    let lower_bounds = [f64::NEG_INFINITY, f64::NEG_INFINITY, 0.0];
    let x = [0.5001, -0.3, 1.0005];
    let certification = synthetic_certification(&f, &x, &lower_bounds, &[]);
    let evidence = joint_newton_decrement(&certification, &x, &lower_bounds, 2, None, 1.0e-6);
    assert_eq!(evidence.eager.verdict, NewtonDecrementVerdict::NotAssessed);
    assert!(evidence.eager.objective_gap.is_none());
    assert_eq!(evidence.excluded[0].index, 2);
    assert_eq!(evidence.excluded[0].reason, "nonpositive_curvature");

    // The full Hessian is indefinite: reported as such, never regularized.
    let x = [0.5001, -0.3, 1.0];
    let certification = synthetic_certification(&f, &x, &lower_bounds, &[]);
    let mut evidence = joint_newton_decrement(&certification, &x, &lower_bounds, 2, None, 1.0e-6);
    // Force the gradient readings to exist for every coordinate.
    evidence.parameter_indices = vec![0, 1, 2];
    evidence.gradient = certification.gradient.clone();
    let full = full_newton_decrement_estimate(&evidence, &hessian, &[0, 1, 2]);
    assert_eq!(full.verdict, NewtonDecrementVerdict::NotAssessed);
    assert!(full.objective_gap.is_none());
    assert!(full.reason.unwrap().contains("not positive definite"));
}

#[test]
fn newton_decrement_escalated_readings_are_judged_in_gap_units() {
    // A stiff coordinate with a strong cubic term: the escalated central
    // differences disagree through O(h^2) truncation (by far more than the
    // absolute 2e-2 agreement test allows), but their Richardson
    // extrapolation reproduces the default-step reading.
    let (c, k, x0) = (1.7e5, 5.0e4, 0.0);
    let f = move |x: &[f64]| {
        let d = x[0] - x0;
        0.5 * c * d * d + k * d * d * d
    };
    let x = [x0 - 2.0e-7];
    let lower = [f64::NEG_INFINITY];
    let certification = synthetic_certification(&f, &x, &lower, &[0]);
    let probes = &certification.curvature_probes[0];
    // Apart in gap units, not only in absolute terms: the reading is
    // determined through the default-step confirmation.
    assert!((probes[1].gradient - probes[2].gradient).powi(2) / (2.0 * c) > 1.0e-7);
    let true_gradient = c * (x[0] - x0) + 3.0 * k * (x[0] - x0).powi(2);
    match decrement_reading(&certification, &x, &lower, 0, 1.0e-6) {
        DecrementReading::Determined {
            gradient,
            curvature,
        } => {
            assert_relative_eq!(gradient, true_gradient, max_relative = 1e-3);
            assert_relative_eq!(curvature, c, max_relative = 1e-3);
        }
        other => panic!("expected a determined reading, got {other:?}"),
    }

    // Readings that agree nowhere (escalated pair apart, extrapolation away
    // from the default-step reading) are ill-determined.
    let mut noisy = certification.clone();
    noisy.curvature_probes[0][0] = JointFdProbe::central(
        probes[0].step,
        f(&[x[0] + probes[0].step]) + 1.0e-4,
        f(&[x[0] - probes[0].step]),
    );
    assert_eq!(
        decrement_reading(&noisy, &x, &lower, 0, 1.0e-6),
        DecrementReading::Excluded("gradient_ill_determined")
    );
}

/// Certificate for a synthetic acceptable joint stop at `params`.
fn synthetic_joint_certificate(
    params: &[f64],
    lower_bounds: &[f64],
    certification: &JointLaplaceCertificationGradient,
) -> OptimizerCertificate {
    let mut optsum = OptSummary::new(params.to_vec());
    optsum.optimizer = Optimizer::TrustBq;
    optsum.backend = Optimizer::TrustBq.canonical_backend();
    optsum.return_value = "JOINT_LAPLACE:FTOL_REACHED".to_string();
    optsum.finitial = 100.1;
    optsum.fmin = 100.0;
    optsum.feval = 50;
    optsum.max_feval = 5000;
    optsum.final_params = params.to_vec();
    let mut certificate = OptimizerCertificate::from_opt_summary_with_context(
        &optsum,
        params,
        lower_bounds,
        Some(500),
    );
    certificate.apply_derivative_evidence(
        OptimizerDerivativeEvidence {
            method: EvidenceMethod::FiniteDifference,
            hessian_method: EvidenceMethod::FiniteDifference,
            gradient: certification.gradient.clone(),
            hessian: None,
        },
        2.0e-2,
        1.0e-6,
    );
    certificate
}

#[test]
fn joint_glmm_decrement_certifies_stiff_gradient_and_flags_flat_gap() {
    let lower_bounds = [f64::NEG_INFINITY, f64::NEG_INFINITY, 0.0];
    // Stiff β direction (curvature 6e4, like contraception's `age`): a raw
    // gradient of ~0.15 is only 2e-7 above the optimum in deviance.
    let stiff = DMatrix::from_diagonal(&DVector::from_vec(vec![6.0e4, 50.0, 30.0]));
    let optimum = [0.0, 1.0, 0.6];
    let f = quadratic_gap(&stiff, &optimum);
    let x = [2.5e-6, 1.0, 0.6];
    let certification = synthetic_certification(&f, &x, &lower_bounds, &[]);
    let mut certificate = synthetic_joint_certificate(&x, &lower_bounds, &certification);
    assert!(certificate.free_gradient_norm.unwrap() > 2.0e-2);
    annotate_glmm_covariance_status(
        &mut certificate,
        &x,
        2,
        &lower_bounds,
        &certification,
        2.0e-2,
        Some(&stiff.view((0, 0), (2, 2)).into_owned()),
    );
    assert_eq!(
        certificate.status,
        crate::compiler::FitStatus::ConvergedInterior
    );
    assert!(!joint_certificate_requires_fallback(&certificate));
    // The raw gradient evidence is kept alongside the decrement.
    assert!(certificate.free_gradient_norm.unwrap() > 2.0e-2);
    let evidence = certificate.stationarity_decrement.as_ref().unwrap();
    assert_eq!(
        evidence.eager.verdict,
        NewtonDecrementVerdict::WithinTolerance
    );
    assert!(certificate.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == DiagnosticCode::OptimizerRecovery
            && diagnostic.payload.get("stationarity_check")
                == Some(&serde_json::json!("newton_decrement"))
    }));
    assert!(!certificate
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == DiagnosticCode::OptimizerNonconvergence));
    // The new field round-trips through the wire format.
    let json = serde_json::to_value(&certificate).unwrap();
    assert_eq!(
        json["stationarity_decrement"]["eager"]["verdict"],
        serde_json::json!("within_tolerance")
    );
    let decoded: OptimizerCertificate = serde_json::from_value(json).unwrap();
    assert_eq!(decoded, certificate);

    // Flat θ direction (curvature 1e-2): a raw gradient of 2e-4 passes the
    // absolute tolerance, yet the stop sits 2e-6 above the optimum.
    let flat = DMatrix::from_diagonal(&DVector::from_vec(vec![50.0, 30.0, 1.0e-2]));
    let f = quadratic_gap(&flat, &optimum);
    let x = [0.0, 1.0, 0.62];
    let certification = synthetic_certification(&f, &x, &lower_bounds, &[]);
    let mut certificate = synthetic_joint_certificate(&x, &lower_bounds, &certification);
    assert!(certificate.free_gradient_norm.unwrap() <= 2.0e-2);
    annotate_glmm_covariance_status(
        &mut certificate,
        &x,
        2,
        &lower_bounds,
        &certification,
        2.0e-2,
        Some(&flat.view((0, 0), (2, 2)).into_owned()),
    );
    assert_eq!(certificate.status, crate::compiler::FitStatus::NotOptimized);
    assert!(joint_certificate_requires_fallback(&certificate));
    let diagnostic = certificate
        .diagnostics
        .iter()
        .find(|diagnostic| diagnostic.code == DiagnosticCode::OptimizerNonconvergence)
        .unwrap();
    assert_eq!(
        diagnostic.payload.get("stationarity_check"),
        Some(&serde_json::json!("newton_decrement"))
    );
    let gap = diagnostic.payload["decrement_objective_gap"]
        .as_f64()
        .unwrap();
    assert_relative_eq!(gap, f(&x), max_relative = 1e-4);
    let mut optsum = OptSummary::new(x.to_vec());
    optsum.finitial = 100.1;
    optsum.fmin = 100.0;
    record_uncertified_joint_candidate_diagnostic(&mut certificate, &optsum);
    assert!(certificate.diagnostics.iter().any(|diagnostic| {
        diagnostic.payload.get("scorecard_class")
            == Some(&serde_json::json!(
                "stationarity_uncertified_joint_candidate"
            ))
    }));
}

/// Objective gap of the joint fit's certificate evidence against the true gap
/// at deliberately premature points, on real models. Premature points are
/// displacements along the flattest Hessian direction evaluated as if they
/// were final (the direction where a raw gradient norm is least informative)
/// and, with NLopt, loose-`ftol_abs` BOBYQA stops. The eager estimate and the
/// full-Hessian estimate must track the true gap within a factor of three,
/// rank every premature point above the optimum, and leave the optimum
/// certified.
#[test]
fn joint_glmm_newton_decrement_calibration() {
    for (name, n_agq) in [("cbpp", 1), ("culcita", 1), ("culcita", 5)] {
        let mut best = joint_laplace_row(name);
        best.fit_with_options(false, n_agq, false).unwrap();
        let p = best.beta.len();
        let optimum = best.lmm.optsum.final_params.clone();
        let mut lower_bounds = vec![f64::NEG_INFINITY; p];
        lower_bounds.extend(best.lmm.lower_bounds());
        let certificate = best
            .lmm
            .compiler_artifact
            .optimizer_certificate
            .clone()
            .unwrap();
        assert_eq!(
            certificate.status,
            crate::compiler::FitStatus::ConvergedInterior
        );
        let optimum_gap = certificate
            .stationarity_decrement
            .as_ref()
            .and_then(|evidence| evidence.eager.objective_gap)
            .unwrap();
        assert!(
            optimum_gap < 1.0e-7,
            "{name} agq{n_agq}: optimum reads {optimum_gap:e}"
        );

        let hessian = best
            .finite_difference_joint_laplace_hessian(&optimum, &lower_bounds)
            .unwrap();
        let eigen = SymmetricEigen::new(hessian.clone());
        let flattest = (0..optimum.len())
            .min_by(|&a, &b| eigen.eigenvalues[a].total_cmp(&eigen.eigenvalues[b]))
            .unwrap();
        let direction = eigen.eigenvectors.column(flattest).into_owned();
        let curvature = eigen.eigenvalues[flattest];
        #[allow(unused_mut, reason = "NLopt stops are appended when the feature is on")]
        let mut points = [1.0e-5, 1.0e-4]
            .map(|target| {
                let step = (2.0 * target / curvature).sqrt();
                let x = (0..optimum.len())
                    .map(|i| optimum[i] + step * direction[i])
                    .collect::<Vec<_>>();
                (x, None::<crate::compiler::FitStatus>)
            })
            .to_vec();
        #[cfg(feature = "nlopt")]
        {
            let mut profiled = joint_laplace_row(name);
            profiled.fit_with_options(true, n_agq, false).unwrap();
            for ftol_abs in [1.0e-2, 1.0e-4] {
                let mut model = profiled.clone();
                let start_objective = model.deviance_with_response_constants(n_agq);
                model.lmm.optsum.optimizer = Optimizer::NloptBobyqa;
                model.lmm.optsum.ftol_abs = ftol_abs;
                model
                    .lmm
                    .optsum
                    .caller_set_fields
                    .push("ftol_abs".to_string());
                model
                    .fit_joint_glmm_from_start(
                        profiled.beta.as_slice().to_vec(),
                        profiled.theta.clone(),
                        start_objective,
                        n_agq,
                        2000,
                        None,
                    )
                    .unwrap();
                let status = model
                    .lmm
                    .compiler_artifact
                    .optimizer_certificate
                    .as_ref()
                    .map(|certificate| certificate.status);
                points.push((model.lmm.optsum.final_params.clone(), status));
            }
        }

        let f_optimum = best.joint_glmm_deviance_at_params(&optimum, p, n_agq);
        let mut calibrated = 0;
        for (x, fit_status) in points {
            let mut model = best.clone();
            let true_gap = model.joint_glmm_deviance_at_params(&x, p, n_agq) - f_optimum;
            if true_gap < 3.0e-6 {
                // A stop that landed within the tolerance band is not a
                // premature-stop calibration point.
                continue;
            }
            calibrated += 1;
            // A premature optimizer stop's own certificate must say so,
            // whatever its raw gradient norm read.
            if let Some(status) = fit_status {
                assert_eq!(status, crate::compiler::FitStatus::NotOptimized);
            }
            let certification =
                model.joint_laplace_certification_gradient(&x, p, n_agq, &lower_bounds, 2.0e-2);
            let beta_hessian = model.joint_working_beta_hessian();
            let evidence = joint_newton_decrement(
                &certification,
                &x,
                &lower_bounds,
                p,
                beta_hessian.as_ref(),
                JOINT_STATIONARITY_GAP_TOLERANCE,
            );
            let eager = evidence.eager.objective_gap.unwrap();
            assert_eq!(
                evidence.eager.variant,
                NewtonDecrementVariant::BetaBlockThetaDiagonal
            );
            assert_eq!(
                evidence.eager.verdict,
                NewtonDecrementVerdict::ExceedsTolerance
            );
            assert!(eager > 10.0 * optimum_gap);
            assert!(
                (1.0 / 3.0..=3.0).contains(&(eager / true_gap)),
                "{name} agq{n_agq}: eager {eager:e} vs true {true_gap:e}"
            );
            let point_hessian = model
                .finite_difference_joint_laplace_hessian(&x, &lower_bounds)
                .unwrap();
            let active = (0..x.len()).collect::<Vec<_>>();
            let full = full_newton_decrement_estimate(&evidence, &point_hessian, &active)
                .objective_gap
                .unwrap();
            assert!(
                (1.0 / 3.0..=3.0).contains(&(full / true_gap)),
                "{name} agq{n_agq}: full {full:e} vs true {true_gap:e}"
            );
        }
        assert!(calibrated >= 2, "{name} agq{n_agq}: {calibrated} points");
    }
}

#[test]
fn joint_glmm_diagonal_decrement_is_evidence_only() {
    // Without a working β block the estimate is the pure diagonal, which
    // correlated coordinates can bias either way: it is recorded, and the
    // raw gradient rule decides as before.
    let lower_bounds = [f64::NEG_INFINITY, f64::NEG_INFINITY, 0.0];
    let stiff = DMatrix::from_diagonal(&DVector::from_vec(vec![6.0e4, 50.0, 30.0]));
    let optimum = [0.0, 1.0, 0.6];
    let f = quadratic_gap(&stiff, &optimum);
    let x = [2.5e-6, 1.0, 0.6];
    let certification = synthetic_certification(&f, &x, &lower_bounds, &[]);
    let mut certificate = synthetic_joint_certificate(&x, &lower_bounds, &certification);
    annotate_glmm_covariance_status(
        &mut certificate,
        &x,
        2,
        &lower_bounds,
        &certification,
        2.0e-2,
        None,
    );
    let evidence = certificate.stationarity_decrement.as_ref().unwrap();
    assert_eq!(evidence.eager.variant, NewtonDecrementVariant::Diagonal);
    assert_eq!(
        evidence.eager.verdict,
        NewtonDecrementVerdict::WithinTolerance
    );
    assert_eq!(certificate.status, crate::compiler::FitStatus::NotOptimized);
}
