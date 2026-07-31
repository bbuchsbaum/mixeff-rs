//! GLMM convergence re-verification: bounded multi-start refits of the same
//! estimator with an engine-owned agreement verdict, mirroring the LMM
//! `verify_convergence` contract (bd-01KYW1WQ38JDZ3SGFDC43V87AE).

use super::*;
use crate::compiler::{ConvergenceVerification, ConvergenceVerificationRun};
use crate::model::linear::optimizer::{
    build_verification_run, jittered_theta, verification_message, verification_status,
};
use crate::model::linear::{ConvergenceVerificationOptions, OptimizerChoice};

impl GeneralizedLinearMixedModel {
    /// Re-verify the fitted GLMM optimum with bounded restarts.
    ///
    /// Mirrors `LinearMixedModel::verify_convergence`: refits the same
    /// estimator that produced the returned fit (profiled fast-PIRLS, or
    /// labelled joint Laplace/AGQ) from the fitted theta and from
    /// deterministically jittered starts, then returns an engine-owned
    /// agreement verdict over objective, theta, and beta. The verification is
    /// also attached to the optimizer certificate on the compiled artifact.
    ///
    /// For a labelled fast-PIRLS fallback fit the route verifies the
    /// effective estimator's own optimum (the profiled objective the returned
    /// numbers came from), not the failed joint attempt.
    ///
    /// Joint-route caveat: the joint driver always re-derives its start from
    /// a fresh profiled fast-PIRLS prefit, so a jittered theta perturbs that
    /// prefit's start rather than the joint stage directly; with the default
    /// small jitter the prefit re-converges and the jitter run is close to a
    /// second restart-from-optimum run.
    pub fn verify_convergence(&mut self) -> Result<ConvergenceVerification> {
        self.verify_convergence_with_options(ConvergenceVerificationOptions::glmm_defaults())
    }

    /// Run GLMM convergence verification with explicit controls.
    ///
    /// `run_optimizer_consensus` is not implemented for GLMMs and is ignored:
    /// the GLMM fit drivers own optimizer selection, so the run families are
    /// `restart_from_optimum` and `jitter_restart_*` only. Effective
    /// covariance rank agreement is scored only when the artifact records
    /// rank summaries (GLMM fits currently do not, so that clause is
    /// trivially satisfied).
    pub fn verify_convergence_with_options(
        &mut self,
        options: ConvergenceVerificationOptions,
    ) -> Result<ConvergenceVerification> {
        if self.lmm.optsum.feval <= 0 {
            let verification = ConvergenceVerification::not_run("model has not been fitted");
            if let Some(certificate) = &mut self.lmm.compiler_artifact.optimizer_certificate {
                certificate.verification = Some(verification.clone());
            }
            return Ok(verification);
        }

        // The returned fit's own objective family: joint deviances include
        // response constants and carry a JOINT_* return-code prefix; the
        // profiled fast-PIRLS deviance (including labelled fallback fits)
        // does not.
        let fast = !glmm_objective_includes_response_constants(&self.lmm.optsum.return_value);
        let n_agq = self.lmm.optsum.n_agq.max(1);

        let reference_theta = self.theta.clone();
        let reference_beta = self.beta.as_slice().to_vec();
        let reference_objective = self
            .lmm
            .optsum
            .fmin
            .is_finite()
            .then_some(self.lmm.optsum.fmin);
        let reference_effective_ranks = self
            .lmm
            .compiler_artifact
            .effective_covariance
            .iter()
            .map(|summary| summary.supported_rank)
            .collect::<Vec<_>>();

        let mut runs = Vec::new();
        if options.restart_from_optimum {
            runs.push(self.glmm_convergence_verification_run(
                "restart_from_optimum",
                fast,
                n_agq,
                &reference_theta,
                &reference_theta,
                &reference_beta,
                reference_objective,
                &reference_effective_ranks,
                &options,
            ));
        }

        for jitter_index in 0..options.jitter_starts {
            let start = jittered_theta(
                &reference_theta,
                &self.lmm.lower_bounds(),
                options.jitter_scale,
                jitter_index,
            );
            runs.push(self.glmm_convergence_verification_run(
                &format!("jitter_restart_{}", jitter_index + 1),
                fast,
                n_agq,
                &start,
                &reference_theta,
                &reference_beta,
                reference_objective,
                &reference_effective_ranks,
                &options,
            ));
        }

        let status = verification_status(&runs, &options);
        let message = verification_message(status, &runs);
        let verification = ConvergenceVerification {
            status,
            objective_tolerance: options.objective_tolerance,
            theta_tolerance: options.theta_tolerance,
            beta_tolerance: options.beta_tolerance,
            reference_objective,
            reference_theta,
            reference_beta,
            reference_effective_ranks,
            runs,
            message,
        };

        if let Some(certificate) = &mut self.lmm.compiler_artifact.optimizer_certificate {
            certificate.verification = Some(verification.clone());
        }
        Ok(verification)
    }

    #[allow(clippy::too_many_arguments)]
    fn glmm_convergence_verification_run(
        &self,
        label: &str,
        fast: bool,
        n_agq: usize,
        start_theta: &[f64],
        reference_theta: &[f64],
        reference_beta: &[f64],
        reference_objective: Option<f64>,
        reference_effective_ranks: &[usize],
        options: &ConvergenceVerificationOptions,
    ) -> ConvergenceVerificationRun {
        // Reproduce the original optimizer selection: a caller-forced
        // optimizer is re-requested explicitly; an auto-selected fit must
        // start the refit from the default profiled optimizer, because the
        // fitted clone's `optsum.optimizer` records the *final* stage's
        // optimizer (for a joint fit, the joint-stage backend), which the
        // profiled prefit dispatch cannot run.
        let caller_optimizer = self.lmm.optsum.caller_selected_optimizer();
        let mut candidate = self.clone();
        let result = candidate.reset_for_refit(None).and_then(|_| {
            if caller_optimizer.is_none() {
                candidate.configure_profile_start_optimizer();
            }
            candidate
                .fit_with_glmm_options(GlmmFitOptions {
                    fast,
                    n_agq,
                    verbose: false,
                    optimizer_control: OptimizerControl {
                        optimizer: caller_optimizer
                            .map(OptimizerChoice::Named)
                            .unwrap_or(OptimizerChoice::Auto),
                        start_theta: Some(start_theta.to_vec()),
                        max_feval: Some(options.max_function_evaluations),
                        ..OptimizerControl::default()
                    },
                    progress_callback: None,
                })
                .map(|_| ())
        });

        match result {
            Ok(()) => {
                // A joint verification refit can itself fail certification and
                // return the labelled fast-PIRLS fallback. Its profiled
                // deviance drops the response constants the joint reference
                // includes, so the objectives are in different units; scoring
                // the delta would misreport an estimator substitution as a
                // reproducibility failure.
                let candidate_joint =
                    glmm_objective_includes_response_constants(&candidate.lmm.optsum.return_value);
                if candidate_joint == fast {
                    return ConvergenceVerificationRun {
                        label: label.to_string(),
                        optimizer_name: Some(candidate.lmm.optsum.optimizer_name().to_string()),
                        return_code: Some(candidate.lmm.optsum.return_value.clone()),
                        objective_value: None,
                        objective_delta: None,
                        max_abs_theta_delta: None,
                        max_abs_beta_delta: None,
                        effective_ranks: Vec::new(),
                        agrees: false,
                        diagnostics: vec![
                            "verification refit returned a different estimator than the reference \
                             fit (joint certification failed and the labelled fast-PIRLS fallback \
                             was substituted); objectives are not comparable across estimators, so \
                             the joint optimum could not be re-verified within this budget"
                                .to_string(),
                        ],
                    };
                }

                build_verification_run(
                    label,
                    Some(candidate.lmm.optsum.optimizer_name().to_string()),
                    Some(candidate.lmm.optsum.return_value.clone()),
                    candidate
                        .lmm
                        .optsum
                        .fmin
                        .is_finite()
                        .then_some(candidate.lmm.optsum.fmin),
                    candidate.theta.clone(),
                    candidate.beta.as_slice().to_vec(),
                    candidate
                        .lmm
                        .compiler_artifact
                        .effective_covariance
                        .iter()
                        .map(|summary| summary.supported_rank)
                        .collect::<Vec<_>>(),
                    reference_theta,
                    reference_beta,
                    reference_objective,
                    reference_effective_ranks,
                    options,
                )
            }
            Err(error) => ConvergenceVerificationRun {
                label: label.to_string(),
                optimizer_name: Some(self.lmm.optsum.optimizer_name().to_string()),
                return_code: None,
                objective_value: None,
                objective_delta: None,
                max_abs_theta_delta: None,
                max_abs_beta_delta: None,
                effective_ranks: Vec::new(),
                agrees: false,
                diagnostics: vec![error.to_string()],
            },
        }
    }
}
