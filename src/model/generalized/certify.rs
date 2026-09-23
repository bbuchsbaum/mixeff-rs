//! GLMM covariance certification, fallback annotation, and joint-Hessian helpers.
//!
//! Moved verbatim from the former single-file `generalized.rs` during the
//! module split (bd-01KWG1BKEWB91RXAXC0350SFMK). No logic changes.

use super::*;
use crate::compiler::{
    NewtonDecrementEstimate, NewtonDecrementEvidence, NewtonDecrementExclusion,
    NewtonDecrementVariant, NewtonDecrementVerdict,
};

pub(crate) fn joint_glmm_status_prefix(n_agq: usize) -> &'static str {
    if n_agq <= 1 {
        "JOINT_LAPLACE"
    } else {
        "JOINT_AGQ"
    }
}

pub(crate) fn glmm_objective_includes_response_constants(return_value: &str) -> bool {
    return_value.starts_with("JOINT_LAPLACE:")
        || return_value.starts_with("JOINT_LAPLACE_FAILED:")
        || return_value.starts_with("JOINT_AGQ:")
        || return_value.starts_with("JOINT_AGQ_FAILED:")
        || return_value.starts_with("EXPERIMENTAL_JOINT:")
        || return_value.starts_with("EXPERIMENTAL_JOINT_FAILED:")
}

/// Classify a joint GLMM stop from its certification probes.
///
/// Stationarity is judged by the Newton-decrement estimate of the objective
/// gap (see [`joint_newton_decrement`]) whenever its β-block variant is
/// assessed (the pure-diagonal fallback is recorded but does not decide):
/// a gap within [`JOINT_STATIONARITY_GAP_TOLERANCE`] certifies the free
/// coordinates even when the raw gradient norm exceeds `gradient_tolerance`
/// (gradient magnitudes depend on parameter scaling; the gap does not), and a
/// gap above it is an assessed non-stationarity even when the raw gradient
/// passes. When the decrement is not assessed the raw noise-aware gradient
/// rule decides, as before. The raw gradient evidence is always kept on the
/// certificate. Bound-held covariance parameters keep the one-sided KKT
/// check on the raw gradient.
pub(crate) fn annotate_glmm_covariance_status(
    certificate: &mut OptimizerCertificate,
    params: &[f64],
    n_beta: usize,
    lower_bounds: &[f64],
    certification: &JointLaplaceCertificationGradient,
    gradient_tolerance: f64,
    beta_hessian: Option<&DMatrix<f64>>,
) {
    if !certificate.evidence.optimizer_stop.acceptable_stop || params.len() <= n_beta {
        return;
    }
    let boundary_tolerance = 1.0e-8;
    let gradient = certification.gradient.as_slice();

    let decrement = joint_newton_decrement(
        certification,
        params,
        lower_bounds,
        n_beta,
        beta_hessian,
        JOINT_STATIONARITY_GAP_TOLERANCE,
    );
    // Only the β-block estimate decides. The pure diagonal can miss the gap
    // by the inverse of `1 - ρ` for correlated coordinates in either
    // direction (it is invariant to rescaling the parameters, not to
    // rotating them), so it is recorded as evidence and the raw gradient
    // rule decides, as it did before.
    let decrement_verdict = match decrement.eager.variant {
        NewtonDecrementVariant::BetaBlockThetaDiagonal => decrement.eager.verdict,
        _ => NewtonDecrementVerdict::NotAssessed,
    };
    let decrement_gap = decrement.eager.objective_gap;
    let decrement_variant = decrement.eager.variant;
    certificate.stationarity_decrement = Some(decrement);
    let raw_gradient_fails = certificate
        .free_gradient_norm
        .is_some_and(|norm| !norm.is_finite() || norm > gradient_tolerance);
    let insert_decrement_payload = |diagnostic: &mut Diagnostic| {
        diagnostic.payload.insert(
            "decrement_objective_gap".to_string(),
            serde_json::json!(decrement_gap),
        );
        diagnostic.payload.insert(
            "decrement_variant".to_string(),
            serde_json::json!(decrement_variant),
        );
        diagnostic.payload.insert(
            "decrement_gap_tolerance".to_string(),
            serde_json::json!(JOINT_STATIONARITY_GAP_TOLERANCE),
        );
    };
    match decrement_verdict {
        NewtonDecrementVerdict::WithinTolerance if raw_gradient_fails => {
            // The raw gradient exceeds its absolute tolerance, but the
            // estimated objective gap is negligible: the large components lie
            // along stiff directions where a small step already changes the
            // objective a lot. Replace the generic derivative-failure
            // diagnostic with the decrement's evidence and certify.
            certificate.diagnostics.retain(|diagnostic| {
                !(diagnostic.code == DiagnosticCode::OptimizerNonconvergence
                    && diagnostic.payload.contains_key("derivative_failures"))
            });
            let free_gradient_norm = certificate.free_gradient_norm;
            let mut diagnostic = Diagnostic::new(
                DiagnosticCode::OptimizerRecovery,
                DiagnosticSeverity::Info,
                DiagnosticStage::Certification,
                "GLMM joint stationarity certified by the Newton-decrement objective gap; the raw free-gradient norm exceeds its absolute tolerance only along stiff directions",
            )
            .with_suggested_actions(vec![
                "read decrement_objective_gap (deviance units) as the stationarity evidence; the raw gradient norm depends on parameter scaling".to_string(),
            ]);
            diagnostic
                .payload
                .insert("fit_mode".to_string(), serde_json::json!("joint_glmm"));
            diagnostic.payload.insert(
                "stationarity_check".to_string(),
                serde_json::json!("newton_decrement"),
            );
            diagnostic.payload.insert(
                "free_gradient_norm".to_string(),
                serde_json::json!(free_gradient_norm),
            );
            diagnostic.payload.insert(
                "gradient_tolerance".to_string(),
                serde_json::json!(gradient_tolerance),
            );
            insert_decrement_payload(&mut diagnostic);
            insert_certification_gradient_payload(&mut diagnostic, certification);
            certificate.diagnostics.push(diagnostic);
            for check in &mut certificate.checks {
                if let crate::compiler::CertificateCheck::DerivativeMismatch {
                    kind, message, ..
                } = check
                {
                    if kind == "free_gradient_kkt_mismatch" {
                        message.push_str(
                            "; superseded by Newton-decrement stationarity (see stationarity_decrement)",
                        );
                    }
                }
            }
            certificate.evidence.certification_quality = EvidenceQuality::Approximate {
                reason: format!(
                    "finite-difference Newton-decrement stationarity passed (estimated objective gap {:.3e} <= {JOINT_STATIONARITY_GAP_TOLERANCE:.1e}); raw free-gradient norm {:.6e} exceeds the absolute tolerance {gradient_tolerance:.6e}",
                    decrement_gap.unwrap_or(f64::NAN),
                    free_gradient_norm.unwrap_or(f64::NAN),
                ),
            };
        }
        NewtonDecrementVerdict::ExceedsTolerance if !raw_gradient_fails => {
            // The raw gradient passes its absolute tolerance, but the
            // estimated objective gap does not: the stop rests along a flat
            // direction where small gradients still leave the objective
            // materially above its optimum.
            certificate.diagnostics.retain(|diagnostic| {
                !(diagnostic.code == DiagnosticCode::OptimizerNonconvergence
                    && diagnostic.payload.contains_key("derivative_failures"))
            });
            certificate.status = crate::compiler::FitStatus::NotOptimized;
            let mut diagnostic = Diagnostic::new(
                DiagnosticCode::OptimizerNonconvergence,
                DiagnosticSeverity::Warning,
                DiagnosticStage::Certification,
                "GLMM joint optimizer stop failed Newton-decrement stationarity: the estimated objective gap exceeds its tolerance although the raw gradient passes; convergence is not certified",
            )
            .with_suggested_actions(vec![
                "treat this joint GLMM result as not optimized until a tighter run or alternate optimizer certifies stationarity".to_string(),
                "fall back to the labelled fast-PIRLS GLMM result when available rather than reporting a silent interior convergence".to_string(),
            ]);
            diagnostic
                .payload
                .insert("fit_mode".to_string(), serde_json::json!("joint_glmm"));
            diagnostic.payload.insert(
                "stationarity_check".to_string(),
                serde_json::json!("newton_decrement"),
            );
            diagnostic.payload.insert(
                "free_gradient_norm".to_string(),
                serde_json::json!(certificate.free_gradient_norm),
            );
            diagnostic.payload.insert(
                "gradient_tolerance".to_string(),
                serde_json::json!(gradient_tolerance),
            );
            insert_decrement_payload(&mut diagnostic);
            insert_certification_gradient_payload(&mut diagnostic, certification);
            if let Some(return_code) = &certificate.evidence.optimizer_stop.return_code {
                diagnostic
                    .payload
                    .insert("return_code".to_string(), serde_json::json!(return_code));
            }
            certificate.diagnostics.push(diagnostic);
            certificate.evidence.certification_quality = EvidenceQuality::Approximate {
                reason: format!(
                    "finite-difference Newton-decrement stationarity failed: estimated objective gap {:.3e} exceeds {JOINT_STATIONARITY_GAP_TOLERANCE:.1e}",
                    decrement_gap.unwrap_or(f64::NAN),
                ),
            };
            return;
        }
        _ => {}
    }

    if let Some(free_gradient_norm) = certificate.free_gradient_norm {
        if decrement_verdict != NewtonDecrementVerdict::WithinTolerance
            && (!free_gradient_norm.is_finite() || free_gradient_norm > gradient_tolerance)
        {
            // `apply_derivative_evidence` emits a generic convergence
            // diagnostic before this GLMM-specific, noise-aware pass. Replace
            // it here: the assembled reading may be an assessed failure or an
            // explicitly unassessable noise-dominated probe, and reporting
            // both would contradict the latter verdict.
            certificate.diagnostics.retain(|diagnostic| {
                !(diagnostic.code == DiagnosticCode::OptimizerNonconvergence
                    && diagnostic.payload.contains_key("derivative_failures"))
            });
            // The assembled gradient failed the free-component KKT check. A
            // failure is only an *assessed* non-stationarity when some
            // failing free component carries a trusted reading; if every
            // failing component is one the noise-aware probe could not
            // assess, the honest verdict is "not assessable", not "not
            // optimized".
            // An assessed decrement above tolerance is itself an assessed
            // failure, whatever the per-component noise verdicts.
            let assessed_failure = decrement_verdict == NewtonDecrementVerdict::ExceedsTolerance
                || certification_gradient_assessed_free_failure(
                    certification,
                    params,
                    lower_bounds,
                    gradient_tolerance,
                );
            if assessed_failure {
                certificate.status = crate::compiler::FitStatus::NotOptimized;
                if !certificate.diagnostics.iter().any(|diagnostic| {
                    diagnostic.code == DiagnosticCode::OptimizerNonconvergence
                        && diagnostic
                            .payload
                            .get("stationarity_check")
                            .and_then(serde_json::Value::as_str)
                            == Some("free_gradient_kkt")
                }) {
                    let mut diagnostic = Diagnostic::new(
                        DiagnosticCode::OptimizerNonconvergence,
                        DiagnosticSeverity::Warning,
                        DiagnosticStage::Certification,
                        "GLMM joint optimizer stop failed finite-difference stationarity; convergence is not certified",
                    )
                    .with_suggested_actions(vec![
                        "treat this joint GLMM result as not optimized until a tighter run or alternate optimizer certifies stationarity".to_string(),
                        "fall back to the labelled fast-PIRLS GLMM result when available rather than reporting a silent interior convergence".to_string(),
                    ]);
                    diagnostic
                        .payload
                        .insert("fit_mode".to_string(), serde_json::json!("joint_glmm"));
                    diagnostic.payload.insert(
                        "stationarity_check".to_string(),
                        serde_json::json!("free_gradient_kkt"),
                    );
                    diagnostic.payload.insert(
                        "free_gradient_norm".to_string(),
                        serde_json::json!(free_gradient_norm),
                    );
                    diagnostic.payload.insert(
                        "gradient_tolerance".to_string(),
                        serde_json::json!(gradient_tolerance),
                    );
                    if decrement_verdict == NewtonDecrementVerdict::ExceedsTolerance {
                        insert_decrement_payload(&mut diagnostic);
                    }
                    insert_certification_gradient_payload(&mut diagnostic, certification);
                    if let Some(return_code) = &certificate.evidence.optimizer_stop.return_code {
                        diagnostic
                            .payload
                            .insert("return_code".to_string(), serde_json::json!(return_code));
                    }
                    certificate.diagnostics.push(diagnostic);
                }
            } else {
                certificate.status = crate::compiler::FitStatus::NotAssessed;
                let mut diagnostic = Diagnostic::new(
                    DiagnosticCode::OptimizerNotAssessed,
                    DiagnosticSeverity::Warning,
                    DiagnosticStage::Certification,
                    "GLMM joint stationarity could not be assessed: the finite-difference probe is noise-dominated on a flat deviance direction even at escalated steps",
                )
                .with_suggested_actions(vec![
                    "treat this fit as an acceptable optimizer stop whose stationarity is unverifiable, not as an assessed optimization failure".to_string(),
                    "certify externally (reference fit or refit with a tighter inner PIRLS tolerance) before promoting this row to strict parity".to_string(),
                ]);
                diagnostic
                    .payload
                    .insert("fit_mode".to_string(), serde_json::json!("joint_glmm"));
                diagnostic.payload.insert(
                    "stationarity_check".to_string(),
                    serde_json::json!("free_gradient_kkt_noise_dominated"),
                );
                diagnostic.payload.insert(
                    "free_gradient_norm".to_string(),
                    serde_json::json!(free_gradient_norm),
                );
                diagnostic.payload.insert(
                    "gradient_tolerance".to_string(),
                    serde_json::json!(gradient_tolerance),
                );
                insert_certification_gradient_payload(&mut diagnostic, certification);
                if let Some(return_code) = &certificate.evidence.optimizer_stop.return_code {
                    diagnostic
                        .payload
                        .insert("return_code".to_string(), serde_json::json!(return_code));
                }
                certificate.diagnostics.push(diagnostic);
            }
            return;
        }
    }
    let theta_params = &params[n_beta..];
    let theta_lower = lower_bounds.get(n_beta..).unwrap_or(&[]);
    let theta_gradient = gradient.get(n_beta..).unwrap_or(&[]);
    let boundary_indices = theta_params
        .iter()
        .zip(theta_lower.iter())
        .enumerate()
        .filter_map(|(index, (value, lower))| {
            lower
                .is_finite()
                .then_some(())
                .filter(|_| *value <= *lower + boundary_tolerance)
                .map(|_| index)
        })
        .collect::<Vec<_>>();

    if boundary_indices.is_empty() {
        certificate.status = crate::compiler::FitStatus::ConvergedInterior;
        record_certification_gradient_escalation(certificate, certification, gradient_tolerance);
        return;
    }

    // A boundary KKT violation must be *proven* on an assessed reading; an
    // unassessable component cannot demote a boundary stop.
    let invalid_boundary = boundary_indices.iter().any(|&index| {
        !certification
            .unassessable_indices
            .contains(&(index + n_beta))
            && theta_gradient
                .get(index)
                .is_some_and(|value| *value < -gradient_tolerance)
    });
    let classification = if invalid_boundary {
        certificate.status = crate::compiler::FitStatus::NotOptimized;
        CovarianceKktClassification::InvalidBoundaryStop
    } else {
        certificate.status = crate::compiler::FitStatus::ConvergedBoundary;
        CovarianceKktClassification::ValidZeroVariance
    };

    let mut diagnostic = Diagnostic::new(
        DiagnosticCode::BoundaryParameter,
        DiagnosticSeverity::Info,
        DiagnosticStage::Certification,
        format!("GLMM joint covariance state classified as {classification:?}"),
    )
    .with_suggested_actions(vec![
        "interpret zero random-effect scales as boundary covariance estimates, not missing optimizer metadata".to_string(),
        "use the recorded stationarity residual and scorecard class before promoting a GLMM row to parity".to_string(),
    ]);
    diagnostic.payload.insert(
        "covariance_kkt_classification".to_string(),
        serde_json::json!(format!("{classification:?}")),
    );
    diagnostic.payload.insert(
        "boundary_theta_indices".to_string(),
        serde_json::json!(boundary_indices),
    );
    diagnostic.payload.insert(
        "gradient_tolerance".to_string(),
        serde_json::json!(gradient_tolerance),
    );
    insert_certification_gradient_payload(&mut diagnostic, certification);
    certificate.diagnostics.push(diagnostic);
    record_certification_gradient_escalation(certificate, certification, gradient_tolerance);
}

/// True when some free (non-boundary) component fails the stationarity
/// tolerance on an *assessed* reading — i.e. the failure is a proven
/// non-stationarity rather than a noise-dominated probe artifact.
pub(crate) fn certification_gradient_assessed_free_failure(
    certification: &JointLaplaceCertificationGradient,
    params: &[f64],
    lower_bounds: &[f64],
    gradient_tolerance: f64,
) -> bool {
    let boundary_tolerance = 1.0e-8;
    certification
        .gradient
        .iter()
        .enumerate()
        .any(|(index, value)| {
            let at_bound = lower_bounds.get(index).copied().is_some_and(|lower| {
                lower.is_finite()
                    && params.get(index).copied().unwrap_or(f64::NAN) <= lower + boundary_tolerance
            });
            !at_bound
                && (!value.is_finite() || value.abs() > gradient_tolerance)
                && !certification.unassessable_indices.contains(&index)
        })
}

/// Records the noise-aware probe context on a certification diagnostic so the
/// evidence trail shows which components were assessed at escalated
/// finite-difference steps (and which could not be assessed at all).
pub(crate) fn insert_certification_gradient_payload(
    diagnostic: &mut Diagnostic,
    certification: &JointLaplaceCertificationGradient,
) {
    if !certification.was_escalated() {
        return;
    }
    let max_abs = |values: &[f64]| {
        values
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max)
    };
    diagnostic.payload.insert(
        "probe_gradient_max_abs".to_string(),
        serde_json::json!(max_abs(&certification.probe_gradient)),
    );
    diagnostic.payload.insert(
        "assessed_gradient_max_abs".to_string(),
        serde_json::json!(max_abs(&certification.gradient)),
    );
    diagnostic.payload.insert(
        "escalated_indices".to_string(),
        serde_json::json!(certification.escalated_indices),
    );
    diagnostic.payload.insert(
        "unassessable_indices".to_string(),
        serde_json::json!(certification.unassessable_indices),
    );
    diagnostic.payload.insert(
        "escalated_relative_steps".to_string(),
        serde_json::json!(JOINT_LAPLACE_CERT_FD_ESCALATED_RELATIVE_STEPS),
    );
}

/// Leaves an Info-severity evidence trail on certificates whose stationarity
/// verdict relied on escalated finite-difference steps, so a passing status
/// never hides that the default-step probe was noise-dominated.
pub(crate) fn record_certification_gradient_escalation(
    certificate: &mut OptimizerCertificate,
    certification: &JointLaplaceCertificationGradient,
    gradient_tolerance: f64,
) {
    if !certification.was_escalated() {
        return;
    }
    let mut diagnostic = Diagnostic::new(
        DiagnosticCode::OptimizerRecovery,
        DiagnosticSeverity::Info,
        DiagnosticStage::Certification,
        "GLMM joint stationarity was assessed with escalated finite-difference steps; the default-step probe was dominated by inner-PIRLS deviance noise",
    )
    .with_suggested_actions(vec![
        "read the recorded probe and assessed gradient norms before applying strict external-engine tolerances".to_string(),
    ]);
    diagnostic
        .payload
        .insert("fit_mode".to_string(), serde_json::json!("joint_glmm"));
    diagnostic.payload.insert(
        "stationarity_check".to_string(),
        serde_json::json!("free_gradient_kkt_escalated_step"),
    );
    diagnostic.payload.insert(
        "gradient_tolerance".to_string(),
        serde_json::json!(gradient_tolerance),
    );
    insert_certification_gradient_payload(&mut diagnostic, certification);
    certificate.diagnostics.push(diagnostic);
}

pub(crate) fn annotate_glmm_singular_covariance_status(
    certificate: &mut OptimizerCertificate,
    theta: &[f64],
    is_singular: bool,
) {
    let near_zero_theta = theta
        .iter()
        .any(|value| value.is_finite() && value.abs() <= 1.0e-4);
    if !(is_singular || near_zero_theta) {
        return;
    }
    let boundary_roundoff = certificate
        .evidence
        .optimizer_stop
        .return_code
        .as_deref()
        .is_some_and(|code| code == "ROUNDOFF_LIMITED" || code == "FAILED:ROUNDOFF_LIMITED");
    if !certificate.evidence.optimizer_stop.acceptable_stop && !boundary_roundoff {
        return;
    }
    if boundary_roundoff {
        certificate.evidence.optimizer_stop.acceptable_stop = true;
        certificate.evidence.certification_quality = EvidenceQuality::Approximate {
            reason: "roundoff-limited optimizer stop accepted only for a near-zero GLMM covariance boundary"
                .to_string(),
        };
        certificate
            .checks
            .retain(|check| !matches!(check, crate::compiler::CertificateCheck::Failed { .. }));
        certificate
            .diagnostics
            .retain(|diagnostic| diagnostic.code != DiagnosticCode::OptimizerNonconvergence);
    }
    if matches!(
        certificate.status,
        crate::compiler::FitStatus::ConvergedInterior | crate::compiler::FitStatus::NotOptimized
    ) {
        certificate.status = crate::compiler::FitStatus::ConvergedBoundary;
    }
    let classification = CovarianceKktClassification::ValidZeroVariance;
    let mut diagnostic = Diagnostic::new(
        DiagnosticCode::BoundaryParameter,
        DiagnosticSeverity::Info,
        DiagnosticStage::Certification,
        format!("GLMM covariance state classified as {classification:?}"),
    )
    .with_suggested_actions(vec![
        "treat the singular random-effect covariance as an explicit boundary state in downstream summaries".to_string(),
        "do not promote the GLMM row to external-engine parity without the joint optimizer scorecard gate".to_string(),
    ]);
    diagnostic.payload.insert(
        "covariance_kkt_classification".to_string(),
        serde_json::json!(format!("{classification:?}")),
    );
    diagnostic
        .payload
        .insert("is_singular".to_string(), serde_json::json!(true));
    certificate.diagnostics.push(diagnostic);
}

pub(crate) fn uncertified_joint_fallback(
    joint_certificate: &OptimizerCertificate,
    joint_optsum: &OptSummary,
    fallback_fast_pirls: Option<GeneralizedLinearMixedModel>,
) -> Option<GeneralizedLinearMixedModel> {
    if !joint_certificate_requires_fallback(joint_certificate) {
        return None;
    }
    let mut fallback = fallback_fast_pirls?;
    let joint_return_code = joint_optsum.return_value.clone();
    let fast_return_code = fallback.lmm.optsum.return_value.clone();
    let fallback_prefix = if joint_return_code.starts_with("JOINT_AGQ") {
        "JOINT_AGQ_FALLBACK_FAST_PIRLS"
    } else {
        "JOINT_LAPLACE_FALLBACK_FAST_PIRLS"
    };
    fallback.lmm.optsum.return_value =
        format!("{fallback_prefix}(joint={joint_return_code}; fast={fast_return_code})");
    let mut diagnostic = Diagnostic::new(
        DiagnosticCode::OptimizerRecovery,
        DiagnosticSeverity::Warning,
        DiagnosticStage::Certification,
        "joint GLMM did not certify; returning labelled fast-PIRLS fallback",
    )
    .with_suggested_actions(vec![
        "treat this as a documented-divergence fast-PIRLS GLMM result, not a certified joint fit"
            .to_string(),
        "inspect the joint optimizer return code before promoting this row to parity".to_string(),
    ]);
    diagnostic.payload.insert(
        "fit_mode".to_string(),
        serde_json::json!("fallback_fast_pirls"),
    );
    diagnostic.payload.insert(
        "scorecard_class".to_string(),
        serde_json::json!("documented_divergence"),
    );
    diagnostic.payload.insert(
        "joint_return_code".to_string(),
        serde_json::json!(joint_return_code),
    );
    diagnostic.payload.insert(
        "joint_fit_status".to_string(),
        serde_json::json!(format!("{:?}", joint_certificate.status)),
    );
    if let Some(free_gradient_norm) = joint_certificate.free_gradient_norm {
        diagnostic.payload.insert(
            "joint_free_gradient_norm".to_string(),
            serde_json::json!(free_gradient_norm),
        );
    }
    diagnostic.payload.insert(
        "fast_pirls_return_code".to_string(),
        serde_json::json!(fast_return_code),
    );
    diagnostic.payload.insert(
        "joint_optimizer".to_string(),
        serde_json::json!(joint_optsum.optimizer_name()),
    );
    diagnostic.payload.insert(
        "joint_optimizer_backend".to_string(),
        serde_json::json!(joint_optsum.backend_name()),
    );
    diagnostic.payload.insert(
        "joint_feval".to_string(),
        serde_json::json!(joint_optsum.feval),
    );
    diagnostic.payload.insert(
        "joint_max_feval".to_string(),
        serde_json::json!(joint_optsum.max_feval),
    );
    diagnostic.payload.insert(
        "joint_fmin".to_string(),
        serde_json::json!(joint_optsum.fmin),
    );
    fallback
        .lmm
        .compiler_artifact
        .diagnostics
        .push(diagnostic.clone());
    let requested_method = if joint_return_code.starts_with("JOINT_AGQ") {
        "joint_agq"
    } else {
        "joint_laplace"
    };
    // Every fast-PIRLS fit driver installs a certificate before recording
    // metadata, so the substitution record always has a home; a missing
    // certificate here would silently drop the machine-readable substitution.
    debug_assert!(
        fallback.lmm.compiler_artifact.optimizer_certificate.is_some(),
        "fast-PIRLS fallback fit is missing its optimizer certificate; the estimator substitution record would be dropped"
    );
    if let Some(certificate) = &mut fallback.lmm.compiler_artifact.optimizer_certificate {
        certificate.diagnostics.push(diagnostic);
        certificate.estimator_substitution = Some(crate::compiler::EstimatorSubstitution {
            requested_method: requested_method.to_string(),
            effective_method: "fast_pirls_profiled".to_string(),
            requested_fit_status: joint_certificate.status,
            requested_return_code: Some(joint_return_code.clone()),
            requested_free_gradient_norm: joint_certificate.free_gradient_norm,
            reason: "joint GLMM did not certify; returning labelled fast-PIRLS fallback"
                .to_string(),
        });
    }
    fallback.record_glmm_fit_metadata();
    Some(fallback)
}

pub(crate) fn joint_certificate_requires_fallback(
    joint_certificate: &OptimizerCertificate,
) -> bool {
    !joint_certificate.evidence.optimizer_stop.acceptable_stop
        || matches!(
            joint_certificate.status,
            crate::compiler::FitStatus::NotOptimized
        )
}

pub(crate) fn joint_candidate_materially_improves_profiled_start(optsum: &OptSummary) -> bool {
    let (Some(initial), Some(final_value)) = (
        optsum.finitial.is_finite().then_some(optsum.finitial),
        optsum.fmin.is_finite().then_some(optsum.fmin),
    ) else {
        return false;
    };
    if final_value >= initial {
        return false;
    }
    let scale = initial.abs().max(final_value.abs()).max(1.0);
    let tolerance =
        (optsum.ftol_abs.max(1.0e-8) + optsum.ftol_rel.max(1.0e-10) * scale).max(1.0e-8) * 10.0;
    initial - final_value > tolerance
}

pub(crate) fn record_uncertified_joint_candidate_diagnostic(
    certificate: &mut OptimizerCertificate,
    optsum: &OptSummary,
) {
    let objective_delta = optsum.finitial - optsum.fmin;
    let budget_limited = certificate.evidence.optimizer_stop.budget_exhausted
        || optsum.return_value.contains("MAXEVAL_REACHED")
        || optsum.return_value.contains("MAXTIME_REACHED");
    let stationarity_uncertified = certificate.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == DiagnosticCode::OptimizerNonconvergence
            && diagnostic
                .payload
                .get("stationarity_check")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|check| check == "free_gradient_kkt" || check == "newton_decrement")
    });
    let (message, scorecard_class, certification_gap, first_action) = if budget_limited {
        (
            "returning improved joint GLMM candidate after budget exhaustion; convergence is not certified",
            "budget_limited_joint_candidate",
            "budget_exhausted",
            "treat fixed effects and log-likelihood as a budget-limited joint-Laplace candidate, not a certified optimizer convergence",
        )
    } else if stationarity_uncertified {
        (
            "returning improved joint GLMM candidate with uncertified stationarity; convergence is not certified",
            "stationarity_uncertified_joint_candidate",
            "stationarity_uncertified",
            "treat fixed effects and log-likelihood as an uncertified joint-Laplace candidate, not a certified optimizer convergence",
        )
    } else {
        (
            "returning improved joint GLMM candidate without full optimizer certification",
            "uncertified_joint_candidate",
            "uncertified",
            "treat fixed effects and log-likelihood as an uncertified joint-Laplace candidate until an external reference verifies it",
        )
    };
    let mut diagnostic = Diagnostic::new(
        DiagnosticCode::OptimizerNonconvergence,
        DiagnosticSeverity::Warning,
        DiagnosticStage::Certification,
        message,
    )
    .with_suggested_actions(vec![
        first_action.to_string(),
        "increase max_feval or compare against an external joint-Laplace reference before promoting this row to strict parity".to_string(),
    ]);
    diagnostic.payload.insert(
        "fit_mode".to_string(),
        serde_json::json!("uncertified_joint_candidate"),
    );
    diagnostic.payload.insert(
        "scorecard_class".to_string(),
        serde_json::json!(scorecard_class),
    );
    diagnostic.payload.insert(
        "certification_gap".to_string(),
        serde_json::json!(certification_gap),
    );
    diagnostic.payload.insert(
        "objective_delta".to_string(),
        serde_json::json!(objective_delta),
    );
    diagnostic
        .payload
        .insert("joint_fmin".to_string(), serde_json::json!(optsum.fmin));
    diagnostic
        .payload
        .insert("joint_feval".to_string(), serde_json::json!(optsum.feval));
    diagnostic.payload.insert(
        "joint_max_feval".to_string(),
        serde_json::json!(optsum.max_feval),
    );
    certificate.diagnostics.push(diagnostic);
}

pub(crate) fn glmm_profile_likelihood_unsupported_reason(operation: &str) -> String {
    format!(
        "{operation}: GLMM profile likelihood is not implemented in this release; \
         profile_sigma/profile_theta remain LMM-only, so GLMM callers must use \
         certified Wald intervals when available or an explicit bootstrap/profile \
         implementation rather than fabricated profile-likelihood intervals"
    )
}

pub(crate) struct GlmmJointHessianCertification {
    pub(crate) inverse: DMatrix<f64>,
    pub(crate) min_eigenvalue: f64,
    pub(crate) condition_number: f64,
    pub(crate) rank: usize,
}

pub(crate) struct GlmmFixedEffectInferenceArtifacts {
    pub(crate) table: FixedEffectInferenceTable,
    pub(crate) covariance: Option<FixedEffectCovarianceMatrix>,
}

pub(crate) const GLMM_PIRLS_MAX_ITER: usize = 10;

pub(crate) const GLMM_HESSIAN_PIRLS_MAX_ITER: usize = 50;

/// Default relative finite-difference step for joint-Laplace gradients.
pub(crate) const JOINT_LAPLACE_FD_RELATIVE_STEP: f64 = 1.0e-5;

/// Escalated relative steps for the stationarity certification gradient.
///
/// The inner PIRLS stopping rule leaves an O(1e-5) absolute error in the
/// deviance, so a central difference at relative step `h` carries a noise
/// term of roughly `1e-5 / h` in the gradient. Against the 2e-2 stationarity
/// tolerance the default 1e-5 step is useless on flat directions (noise
/// O(1)); these two steps put the noise term at roughly tolerance/2 and
/// tolerance/8 while keeping the central-difference truncation error far
/// below tolerance, and disagreement between them flags a component whose
/// surface is too rough to assess at any trusted step.
pub(crate) const JOINT_LAPLACE_CERT_FD_ESCALATED_RELATIVE_STEPS: [f64; 2] = [1.0e-3, 4.0e-3];

/// Stationarity tolerance for the post-fit profiled fast-PIRLS optimum
/// certificate; matches the joint-Laplace fit-time certification tolerance.
pub(crate) const PIRLS_PROFILED_CERTIFICATE_GRADIENT_TOLERANCE: f64 = 2.0e-2;

/// Theta-dimension budget for the post-fit profiled-optimum certificate. The
/// curvature probe costs ~2k^2 PIRLS solves; beyond this the certificate is
/// skipped with an explicit reason rather than silently slowing every fit.
pub(crate) const PIRLS_PROFILED_CERTIFICATE_MAX_THETA: usize = 12;

/// Result of the noise-aware stationarity gradient probe used by the
/// joint-Laplace optimizer certificate.
#[derive(Debug, Clone)]
pub(crate) struct JointLaplaceCertificationGradient {
    /// Assessed gradient: default-step readings where those already pass the
    /// tolerance, escalated-step readings where the default step was
    /// noise-dominated but the larger steps agreed, and the raw default-step
    /// readings for unassessable components.
    pub(crate) gradient: Vec<f64>,
    /// Raw default-step readings, kept for the certificate evidence trail.
    pub(crate) probe_gradient: Vec<f64>,
    /// Components certified (or honestly failed) via escalated steps.
    pub(crate) escalated_indices: Vec<usize>,
    /// Components whose escalated-step readings disagreed: the probe cannot
    /// distinguish noise from signal, so stationarity is not assessable there.
    pub(crate) unassessable_indices: Vec<usize>,
    /// Objective at the certified point (the base of every central probe).
    pub(crate) base_objective: f64,
    /// Per-component probes: the default-step probe, followed by both
    /// escalated probes (smaller step first) for escalated or unassessable
    /// components.
    pub(crate) curvature_probes: Vec<Vec<JointFdProbe>>,
}

impl JointLaplaceCertificationGradient {
    fn was_escalated(&self) -> bool {
        !self.escalated_indices.is_empty() || !self.unassessable_indices.is_empty()
    }
}

/// One finite-difference probe of a single coordinate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct JointFdProbe {
    /// Absolute step.
    pub(crate) step: f64,
    /// Gradient estimate: central `(f+ - f-) / 2h`, or forward `(f+ - f0) / h`.
    pub(crate) gradient: f64,
    f_plus: f64,
    /// `None` for a forward probe (the minus side would cross a lower bound).
    f_minus: Option<f64>,
}

impl JointFdProbe {
    pub(crate) fn central(step: f64, f_plus: f64, f_minus: f64) -> Self {
        Self {
            step,
            gradient: (f_plus - f_minus) / (2.0 * step),
            f_plus,
            f_minus: Some(f_minus),
        }
    }

    pub(crate) fn forward(step: f64, f_plus: f64, base: f64) -> Self {
        Self {
            step,
            gradient: (f_plus - base) / step,
            f_plus,
            f_minus: None,
        }
    }

    /// Central second difference `(f+ + f- - 2 f0) / h^2`; `None` for a
    /// forward probe or a non-finite reading.
    pub(crate) fn curvature(&self, base: f64) -> Option<f64> {
        let f_minus = self.f_minus?;
        let value = (self.f_plus + f_minus - 2.0 * base) / (self.step * self.step);
        value.is_finite().then_some(value)
    }
}

/// Evidence that a profiled fast-PIRLS fit sits at a certified optimum of its
/// own objective: assessed stationarity over theta plus positive-definite,
/// well-conditioned curvature over the interior theta coordinates. Beta is
/// exactly minimized by the penalized least-squares step at every probed
/// theta, so no separate beta-direction evidence is required.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PirlsProfiledOptimumCertificate {
    /// Largest assessed absolute gradient component over theta.
    pub(crate) gradient_max_abs: f64,
    /// Assessed theta gradient of the profiled objective (interior components
    /// central-difference, boundary components one-sided).
    pub(crate) gradient: Vec<f64>,
    /// Smallest eigenvalue of the interior-theta profiled Hessian.
    pub(crate) min_eigenvalue: f64,
    /// Condition number of the interior-theta profiled Hessian.
    pub(crate) condition_number: f64,
    /// One-based theta indices whose gradient needed escalated FD steps.
    pub(crate) escalated_theta_indices: Vec<usize>,
    /// One-based theta indices held at their lower bounds (one-sided
    /// stationarity check; omitted from the curvature probe).
    pub(crate) boundary_theta_indices: Vec<usize>,
}

pub(crate) fn glmm_hessian_step(value: f64) -> f64 {
    1.0e-4 * value.abs().max(1.0)
}

pub(crate) fn certify_glmm_joint_hessian(
    hessian: &DMatrix<f64>,
    context: &str,
) -> std::result::Result<GlmmJointHessianCertification, String> {
    const CONDITION_NUMBER_MAX: f64 = 1.0e10;

    if hessian.nrows() == 0 || hessian.nrows() != hessian.ncols() {
        return Err(format!(
            "{context} has shape {}x{}",
            hessian.nrows(),
            hessian.ncols()
        ));
    }
    if !matrix_is_finite_local(hessian) {
        return Err(format!("{context} contains non-finite entries"));
    }

    let symmetric = 0.5 * (hessian + hessian.transpose());
    let diagonal_scale = (0..symmetric.nrows())
        .map(|index| symmetric[(index, index)].abs())
        .fold(1.0_f64, f64::max);
    let eigen_tolerance = 1.0e-7 * diagonal_scale;
    let eigen = SymmetricEigen::new(symmetric.clone());
    let mut min_eigenvalue = f64::INFINITY;
    let mut max_eigenvalue = 0.0_f64;
    let mut rank = 0usize;
    for value in eigen.eigenvalues.iter().copied() {
        min_eigenvalue = min_eigenvalue.min(value);
        max_eigenvalue = max_eigenvalue.max(value);
        if value > eigen_tolerance {
            rank += 1;
        }
    }

    if min_eigenvalue <= eigen_tolerance {
        return Err(format!(
            "{context} is not positive definite on the active parameter space: min eigenvalue {min_eigenvalue:.6e} <= tolerance {eigen_tolerance:.6e}"
        ));
    }
    if rank != symmetric.nrows() {
        return Err(format!(
            "{context} rank {rank} is below expected rank {}",
            symmetric.nrows()
        ));
    }

    let condition_number = max_eigenvalue / min_eigenvalue;
    if !condition_number.is_finite() || condition_number > CONDITION_NUMBER_MAX {
        return Err(format!(
            "{context} condition number {condition_number:.6e} exceeds certification threshold {CONDITION_NUMBER_MAX:.6e}"
        ));
    }

    let cholesky = symmetric
        .cholesky()
        .ok_or_else(|| format!("{context} Cholesky factorization failed"))?;
    let inverse = cholesky.inverse();
    if !matrix_is_finite_local(&inverse) {
        return Err(format!("{context} inverse contains non-finite entries"));
    }

    Ok(GlmmJointHessianCertification {
        inverse,
        min_eigenvalue,
        condition_number,
        rank,
    })
}

pub(crate) fn matrix_is_finite_local(matrix: &DMatrix<f64>) -> bool {
    matrix.iter().all(|value| value.is_finite())
}

pub(crate) fn matrix_rows_local(matrix: &DMatrix<f64>) -> Vec<Vec<f64>> {
    (0..matrix.nrows())
        .map(|row| (0..matrix.ncols()).map(|col| matrix[(row, col)]).collect())
        .collect()
}

pub(crate) fn matrix_max_asymmetry_local(matrix: &DMatrix<f64>) -> f64 {
    if matrix.nrows() != matrix.ncols() {
        return f64::INFINITY;
    }
    let mut max_asymmetry = 0.0_f64;
    for row in 0..matrix.nrows() {
        for col in 0..row {
            max_asymmetry = max_asymmetry.max((matrix[(row, col)] - matrix[(col, row)]).abs());
        }
    }
    max_asymmetry
}

pub(crate) fn unpivot_glmm_fixed_effect_covariance(
    active_covariance: &DMatrix<f64>,
    pivot: &[usize],
    full_p: usize,
) -> DMatrix<f64> {
    let mut result = DMatrix::zeros(full_p, full_p);
    for active_row in 0..active_covariance.nrows() {
        let full_row = pivot[active_row];
        for active_col in 0..active_covariance.ncols() {
            let full_col = pivot[active_col];
            result[(full_row, full_col)] = active_covariance[(active_row, active_col)];
        }
    }
    result
}

pub(crate) fn glmm_joint_laplace_fixed_effect_covariance_matrix(
    coef_names: Vec<String>,
    covariance: &DMatrix<f64>,
    rank: usize,
    certification: &GlmmJointHessianCertification,
    omitted_boundary_theta_indices: &[usize],
) -> std::result::Result<FixedEffectCovarianceMatrix, String> {
    let finite = matrix_is_finite_local(covariance);
    let symmetric = finite && matrix_max_asymmetry_local(covariance) <= 1.0e-8;
    let details = FixedEffectCovarianceDetails {
        rank: Some(rank),
        expected_rank: Some(coef_names.len()),
        aliased: Vec::new(),
        matrix_rows: covariance.nrows(),
        matrix_cols: covariance.ncols(),
        finite: Some(finite),
        symmetric: Some(symmetric),
    };

    if !finite {
        return Err(
            "joint-laplace GLMM active-Hessian covariance contains non-finite entries".to_string(),
        );
    }
    if !symmetric {
        return Err(
            "joint-laplace GLMM active-Hessian covariance failed symmetry validation".to_string(),
        );
    }

    Ok(FixedEffectCovarianceMatrix::joint_laplace_active_hessian(
        coef_names,
        matrix_rows_local(covariance),
        details,
        glmm_joint_laplace_hessian_notes(certification, omitted_boundary_theta_indices),
    ))
}

pub(crate) fn glmm_joint_laplace_hessian_notes(
    certification: &GlmmJointHessianCertification,
    omitted_boundary_theta_indices: &[usize],
) -> Vec<String> {
    let mut notes = vec![
        "fixed-effect covariance derived from the beta block of the inverse finite-difference Hessian over joint-laplace beta plus interior theta parameters"
            .to_string(),
        format!(
            "joint Hessian certificate: min eigenvalue {:.6e}, condition number {:.6e}, rank {}",
            certification.min_eigenvalue,
            certification.condition_number,
            certification.rank
        ),
    ];
    if !omitted_boundary_theta_indices.is_empty() {
        let labels = omitted_boundary_theta_indices
            .iter()
            .map(|index| index.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        notes.push(format!(
            "boundary covariance parameters held fixed at their lower bounds and omitted from the active Hessian: theta {labels}"
        ));
    }
    notes
}

pub(crate) fn glmm_inference_availability_for_table(
    metadata: &GlmmFitMetadata,
    table: &FixedEffectInferenceTable,
) -> InferenceAvailability {
    if !table.rows.is_empty()
        && table
            .rows
            .iter()
            .all(|row| row.status == FixedEffectInferenceStatus::Available)
    {
        return InferenceAvailability::Available {
            method: "asymptotic_wald_z_joint_laplace_active_hessian".to_string(),
        };
    }

    if metadata.estimation_method == "joint_laplace" {
        return InferenceAvailability::NotAssessed {
            reason: table
                .rows
                .first()
                .and_then(|row| row.reason.clone())
                .unwrap_or_else(|| {
                    "joint-laplace GLMM fixed-effect Hessian certificate did not pass quality gates"
                        .to_string()
                }),
        };
    }

    InferenceAvailability::Unsupported {
        reason: glmm_fixed_effect_inference_unsupported_reason(&metadata.estimation_method),
    }
}

pub(crate) fn glmm_fixed_effect_inference_unsupported_reason(estimation_method: &str) -> String {
    format!(
        "certified GLMM fixed-effect Wald inference is not implemented for {estimation_method}; \
         fast-PIRLS/profiled covariance geometry remains a working-Hessian payload, while only \
         joint-laplace fits with a passing certified active-subspace Hessian over active beta plus \
         interior theta parameters can report Wald SE/z/p/confint"
    )
}

/// Largest estimated objective gap, in deviance units, that the joint GLMM
/// certificate counts as stationary.
///
/// A deviance gap of 1e-6 moves a likelihood-ratio statistic by 1e-6 and
/// displaces the estimates by about 1e-3 standard errors (a gap `t` along a
/// direction is `(δ/SE)^2`): statistically invisible. It sits ten times above
/// the 1e-7 objective band the joint-fit parity tests hold, so an estimate
/// off by the factor of 1.5 the β-block curvature shows on the calibration
/// set (see `joint_glmm_newton_decrement_calibration`) cannot flag a fit
/// inside that band, and it sits more than an order of magnitude above the
/// estimate read at certified joint optima (<= 4e-8 across contraception,
/// cbpp, culcita, grouseticks, tungara and the rare-event Bernoulli set,
/// Laplace and AGQ), which is the finite-difference noise floor.
pub(crate) const JOINT_STATIONARITY_GAP_TOLERANCE: f64 = 1.0e-6;

/// Bound tolerance for treating a covariance parameter as held at its lower
/// bound (matches the certificate's boundary classification).
const DECREMENT_BOUNDARY_TOLERANCE: f64 = 1.0e-8;

/// Largest ratio between the two escalated-step curvature readings (or
/// between the working and finite-difference fixed-effect curvatures) for the
/// curvature to count as determined.
const DECREMENT_CURVATURE_AGREEMENT_RATIO: f64 = 2.0;

/// Largest disagreement between the two escalated gradient readings of a
/// coordinate, in gap units `(g_a - g_b)^2 / (2 H_ii)`, as a fraction of the
/// gap tolerance, for the reading to count as determined.
const DECREMENT_GRADIENT_AGREEMENT_FRACTION: f64 = 0.1;

/// Gradient and curvature reading of one free coordinate, or why it has none.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum DecrementReading {
    Determined { gradient: f64, curvature: f64 },
    Excluded(&'static str),
}

/// Reading of coordinate `index` for the Newton decrement.
///
/// Rules (conservative: a coordinate that cannot be read is excluded and the
/// decrement is then not assessed, rather than guessed):
/// - a covariance parameter within 1e-8 of its lower bound is excluded
///   (`at_lower_bound`): its stationarity is the one-sided KKT condition the
///   boundary check applies;
/// - a coordinate probed only on one side (the minus step would cross the
///   bound) has no curvature (`one_sided_probe`);
/// - a default-step coordinate uses its central gradient and second
///   difference `(f+ + f- - 2 f0) / h^2`, which must be positive
///   (`nonpositive_curvature`);
/// - an escalated coordinate uses the two escalated probes: both second
///   differences must be positive and agree within a factor of two
///   (`curvature_ill_determined`). The gradient is the Richardson
///   extrapolation `g_R = g_a + (g_a - g_b) / 15` of the two central
///   differences (steps in ratio 1:4), which cancels their O(h^2) truncation
///   error; the curvature is the larger-step reading, the less
///   noise-sensitive one. The reading counts as determined when, in gap
///   units `(Δg)^2 / (2 H)`, either the two escalated readings agree or `g_R`
///   agrees with the independent default-step reading, to within
///   `0.1 * gap_tolerance` (`gradient_ill_determined` otherwise).
///
/// The gap-unit agreement test replaces the absolute 2e-2 agreement test of
/// the gradient certificate. On a stiff coordinate the escalated steps
/// disagree through truncation alone (grouseticks' `cHEIGHT`, curvature
/// 1.7e5: 0.016 at h = 1e-3 and 0.874 at 4e-3), while their Richardson
/// extrapolation (-0.0412) reproduces the default-step reading (-0.0410) to
/// 1e-13 in objective units. On a flat, noise-dominated coordinate the
/// default-step reading is the unreliable one and the escalated readings
/// agree.
pub(crate) fn decrement_reading(
    certification: &JointLaplaceCertificationGradient,
    params: &[f64],
    lower_bounds: &[f64],
    index: usize,
    gap_tolerance: f64,
) -> DecrementReading {
    let lower = lower_bounds
        .get(index)
        .copied()
        .unwrap_or(f64::NEG_INFINITY);
    if lower.is_finite()
        && params.get(index).copied().unwrap_or(f64::NAN) <= lower + DECREMENT_BOUNDARY_TOLERANCE
    {
        return DecrementReading::Excluded("at_lower_bound");
    }
    let base = certification.base_objective;
    let Some(probes) = certification.curvature_probes.get(index) else {
        return DecrementReading::Excluded("non_finite_reading");
    };
    if !base.is_finite() {
        return DecrementReading::Excluded("non_finite_reading");
    }
    match probes.as_slice() {
        [probe] => {
            if !probe.gradient.is_finite() {
                return DecrementReading::Excluded("non_finite_reading");
            }
            match probe.curvature(base) {
                None if probe.f_minus.is_none() => DecrementReading::Excluded("one_sided_probe"),
                None => DecrementReading::Excluded("non_finite_reading"),
                Some(curvature) if curvature <= 0.0 => {
                    DecrementReading::Excluded("nonpositive_curvature")
                }
                Some(curvature) => DecrementReading::Determined {
                    gradient: probe.gradient,
                    curvature,
                },
            }
        }
        [default, small, large] => {
            if small.f_minus.is_none() || large.f_minus.is_none() {
                return DecrementReading::Excluded("one_sided_probe");
            }
            if !(small.gradient.is_finite() && large.gradient.is_finite()) {
                return DecrementReading::Excluded("non_finite_reading");
            }
            let (Some(c_small), Some(c_large)) = (small.curvature(base), large.curvature(base))
            else {
                return DecrementReading::Excluded("non_finite_reading");
            };
            if c_small <= 0.0 || c_large <= 0.0 {
                return DecrementReading::Excluded("nonpositive_curvature");
            }
            if c_small.max(c_large) > DECREMENT_CURVATURE_AGREEMENT_RATIO * c_small.min(c_large) {
                return DecrementReading::Excluded("curvature_ill_determined");
            }
            let extrapolated = small.gradient + (small.gradient - large.gradient) / 15.0;
            let gap_units = |a: f64, b: f64| (a - b).powi(2) / (2.0 * c_large);
            let agreement = DECREMENT_GRADIENT_AGREEMENT_FRACTION * gap_tolerance;
            let escalated_agree = gap_units(small.gradient, large.gradient) <= agreement;
            let default_confirms = default.gradient.is_finite()
                && gap_units(extrapolated, default.gradient) <= agreement;
            if !(escalated_agree || default_confirms) {
                return DecrementReading::Excluded("gradient_ill_determined");
            }
            DecrementReading::Determined {
                gradient: extrapolated,
                curvature: c_large,
            }
        }
        _ => DecrementReading::Excluded("non_finite_reading"),
    }
}

/// Eager Newton-decrement stationarity evidence for a joint GLMM stop.
///
/// Uses only the certification gradient's own probes (no extra objective
/// evaluations) plus, when supplied, the fixed-effect block of the working
/// penalized-least-squares Hessian at the fitted point (`beta_hessian`, in
/// the optimizer's β order). With that block the estimate is
/// `½ (g_βᵀ H_ββ⁻¹ g_β + Σ_θ g_i² / H_ii)`; without it (or when its diagonal
/// disagrees with the finite-difference curvature by more than a factor of
/// two, or it is not positive definite) the estimate is the diagonal
/// `½ Σ g_i² / H_ii`. If any free coordinate has no determined reading the
/// estimate is not assessed.
pub(crate) fn joint_newton_decrement(
    certification: &JointLaplaceCertificationGradient,
    params: &[f64],
    lower_bounds: &[f64],
    n_beta: usize,
    beta_hessian: Option<&DMatrix<f64>>,
    gap_tolerance: f64,
) -> NewtonDecrementEvidence {
    let mut parameter_indices = Vec::new();
    let mut gradient = Vec::new();
    let mut curvature = Vec::new();
    let mut excluded = Vec::new();
    let mut undetermined = Vec::new();
    for index in 0..params.len() {
        match decrement_reading(certification, params, lower_bounds, index, gap_tolerance) {
            DecrementReading::Determined {
                gradient: g,
                curvature: c,
            } => {
                parameter_indices.push(index);
                gradient.push(g);
                curvature.push(c);
            }
            DecrementReading::Excluded(reason) => {
                if reason != "at_lower_bound" {
                    undetermined.push(index);
                }
                excluded.push(NewtonDecrementExclusion {
                    index,
                    reason: reason.to_string(),
                });
            }
        }
    }

    let eager = if !undetermined.is_empty() {
        NewtonDecrementEstimate {
            variant: NewtonDecrementVariant::Diagonal,
            verdict: NewtonDecrementVerdict::NotAssessed,
            objective_gap: None,
            reason: Some(format!(
                "no determined gradient/curvature reading for free parameter(s) {}",
                undetermined
                    .iter()
                    .map(|index| index.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
        }
    } else {
        let diagonal_term = |position: usize| gradient[position].powi(2) / curvature[position];
        let block = beta_hessian.and_then(|hessian| {
            beta_block_decrement(hessian, &parameter_indices, &gradient, &curvature, n_beta)
        });
        let (variant, decrement) = match block {
            Some(beta_part) => {
                let theta_part = (0..parameter_indices.len())
                    .filter(|&position| parameter_indices[position] >= n_beta)
                    .map(diagonal_term)
                    .sum::<f64>();
                (
                    NewtonDecrementVariant::BetaBlockThetaDiagonal,
                    beta_part + theta_part,
                )
            }
            None => (
                NewtonDecrementVariant::Diagonal,
                (0..parameter_indices.len()).map(diagonal_term).sum::<f64>(),
            ),
        };
        newton_decrement_estimate(variant, decrement, gap_tolerance)
    };

    NewtonDecrementEvidence {
        gap_tolerance,
        parameter_indices,
        gradient,
        excluded,
        eager,
        full: None,
    }
}

/// `g_βᵀ H_ββ⁻¹ g_β` from the working fixed-effect Hessian, or `None` when
/// the block cannot be used: a fixed effect is missing a determined reading,
/// the block is not positive definite, or its diagonal disagrees with the
/// finite-difference curvature by more than
/// [`DECREMENT_CURVATURE_AGREEMENT_RATIO`] (the working Hessian is then not a
/// trustworthy stand-in for the objective's curvature).
fn beta_block_decrement(
    hessian: &DMatrix<f64>,
    parameter_indices: &[usize],
    gradient: &[f64],
    curvature: &[f64],
    n_beta: usize,
) -> Option<f64> {
    if n_beta == 0 || hessian.nrows() != n_beta || hessian.ncols() != n_beta {
        return None;
    }
    if parameter_indices.len() < n_beta
        || parameter_indices[..n_beta] != (0..n_beta).collect::<Vec<_>>()[..]
    {
        return None;
    }
    for index in 0..n_beta {
        let working = hessian[(index, index)];
        let probed = curvature[index];
        if !(working.is_finite() && working > 0.0)
            || working.max(probed) > DECREMENT_CURVATURE_AGREEMENT_RATIO * working.min(probed)
        {
            return None;
        }
    }
    let symmetric = 0.5 * (hessian + hessian.transpose());
    let cholesky = symmetric.cholesky()?;
    let g_beta = DVector::from_column_slice(&gradient[..n_beta]);
    let solved = cholesky.solve(&g_beta);
    let value = g_beta.dot(&solved);
    (value.is_finite() && value >= 0.0).then_some(value)
}

/// Estimate record for a decrement `λ²` (the estimate is `λ²/2`).
pub(crate) fn newton_decrement_estimate(
    variant: NewtonDecrementVariant,
    decrement: f64,
    gap_tolerance: f64,
) -> NewtonDecrementEstimate {
    let gap = 0.5 * decrement;
    if !gap.is_finite() || gap < 0.0 {
        return NewtonDecrementEstimate {
            variant,
            verdict: NewtonDecrementVerdict::NotAssessed,
            objective_gap: None,
            reason: Some(format!(
                "decrement is not a finite non-negative number ({decrement})"
            )),
        };
    }
    NewtonDecrementEstimate {
        variant,
        verdict: if gap <= gap_tolerance {
            NewtonDecrementVerdict::WithinTolerance
        } else {
            NewtonDecrementVerdict::ExceedsTolerance
        },
        objective_gap: Some(gap),
        reason: None,
    }
}

/// Full Newton decrement `gᵀH⁻¹g` over the active indices of a
/// finite-difference Hessian, from the eager decrement's gradient readings.
/// Not assessed when an active coordinate has no determined gradient reading
/// or the Hessian is not positive definite (reported, never regularized).
pub(crate) fn full_newton_decrement_estimate(
    evidence: &NewtonDecrementEvidence,
    hessian: &DMatrix<f64>,
    active_indices: &[usize],
) -> NewtonDecrementEstimate {
    let not_assessed = |reason: String| NewtonDecrementEstimate {
        variant: NewtonDecrementVariant::Full,
        verdict: NewtonDecrementVerdict::NotAssessed,
        objective_gap: None,
        reason: Some(reason),
    };
    if hessian.nrows() != active_indices.len() || hessian.ncols() != active_indices.len() {
        return not_assessed(format!(
            "Hessian shape {}x{} does not match {} active parameters",
            hessian.nrows(),
            hessian.ncols(),
            active_indices.len()
        ));
    }
    let mut g = DVector::zeros(active_indices.len());
    for (position, index) in active_indices.iter().enumerate() {
        match evidence
            .parameter_indices
            .iter()
            .position(|candidate| candidate == index)
        {
            Some(slot) => g[position] = evidence.gradient[slot],
            None => {
                return not_assessed(format!(
                    "no determined gradient reading for active parameter {index}"
                ))
            }
        }
    }
    if !matrix_is_finite_local(hessian) {
        return not_assessed("Hessian contains non-finite entries".to_string());
    }
    let symmetric = 0.5 * (hessian + hessian.transpose());
    let min_eigenvalue = SymmetricEigen::new(symmetric.clone())
        .eigenvalues
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let Some(cholesky) = (min_eigenvalue > 0.0)
        .then(|| symmetric.cholesky())
        .flatten()
    else {
        return not_assessed(format!(
            "Hessian is not positive definite (min eigenvalue {min_eigenvalue:.6e}); the decrement is undefined"
        ));
    };
    let solved = cholesky.solve(&g);
    newton_decrement_estimate(
        NewtonDecrementVariant::Full,
        g.dot(&solved),
        evidence.gap_tolerance,
    )
}
