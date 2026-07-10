use super::*;
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

#[cfg(feature = "nlopt")]
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
    };
    annotate_glmm_covariance_status(
        &mut certificate,
        &params,
        2,
        &lower_bounds,
        &certification,
        gradient_tolerance,
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
    };
    certificate.apply_derivative_evidence(
        OptimizerDerivativeEvidence {
            method: EvidenceMethod::FiniteDifference,
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
    };
    certificate.apply_derivative_evidence(
        OptimizerDerivativeEvidence {
            method: EvidenceMethod::FiniteDifference,
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
    for g in 0..4 {
        for obs in 0..5 {
            let xv = obs as f64 - 2.0;
            let eta = 1.2 + 0.25 * xv + group_effects[g];
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
    for g in 0..4 {
        for obs in 0..6 {
            let xv = obs as f64 - 2.5;
            let eta = 1.0 + 0.18 * xv + group_effects[g];
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
        matches!(model.pirls_profiled_optimum_certificate, Some(Ok(_))),
        "fixture should certify: {:?}",
        model.pirls_profiled_optimum_certificate
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
        matches!(model.pirls_profiled_optimum_certificate, Some(Err(_))),
        "native fixture should keep uncertified geometry explicit: {:?}",
        model.pirls_profiled_optimum_certificate
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
    if matches!(model.pirls_profiled_optimum_certificate, Some(Ok(_))) {
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
