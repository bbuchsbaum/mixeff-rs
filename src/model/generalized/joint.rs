//! Joint Laplace/AGQ GLMM optimizer, its certification gradients/Hessians,
//! and joint-mode fixed-effect inference artifacts.
//!
//! Moved verbatim from `generalized/mod.rs` during the module split
//! (bd-01KWHYQSTWK60P6HA4S4B2K99P). No logic changes.

use super::*;

impl GeneralizedLinearMixedModel {
    /// Labelled joint GLMM Laplace fit.
    ///
    /// This path optimizes `[β; θ]` against the included-response-constants
    /// Laplace objective. The public `fast = false` path delegates here for
    /// `n_agq <= 1` when NLopt is enabled, while summaries keep it distinct
    /// from the fast-PIRLS profiled path and from labelled fallback results.
    #[cfg(feature = "nlopt")]
    pub fn fit_experimental_joint_laplace_with_response_constants(
        &mut self,
        verbose: bool,
    ) -> Result<&mut Self> {
        self.fit_joint_glmm_with_response_constants(1, verbose)
    }

    /// Labelled joint GLMM fit with response constants retained.
    ///
    /// For `n_agq <= 1` this is joint Laplace; for `n_agq > 1` this is joint
    /// AGQ and is accepted only for the scalar random-effect shapes permitted
    /// by [`validate_agq`](Self::validate_agq).
    pub fn fit_joint_glmm_with_response_constants(
        &mut self,
        n_agq: usize,
        verbose: bool,
    ) -> Result<&mut Self> {
        if self.lmm.optsum.feval > 0 {
            return Err(MixedModelError::AlreadyFitted);
        }
        self.validate_agq(n_agq)?;
        let joint_optimizer = self
            .lmm
            .optsum
            .caller_selected_optimizer()
            .unwrap_or_else(default_joint_glmm_optimizer);
        validate_joint_glmm_optimizer(joint_optimizer)?;
        let saved_optimizer_source = self.lmm.optsum.optimizer_source;
        let saved_caller_set_fields = self.lmm.optsum.caller_set_fields.clone();

        // Use the supported fast path as the deterministic start. This keeps
        // the joint optimizer focused on whether [β; θ] can improve the same
        // included-constants objective for the requested approximation.
        if self.lmm.optsum.caller_selected_optimizer().is_some() {
            self.configure_profile_start_optimizer();
        }
        self.fit_with_options_impl(n_agq, verbose)?;
        let fallback_fast_pirls = self.clone();
        let start_beta = self.beta.as_slice().to_vec();
        let start_theta = self.theta.clone();
        let profiled_start_objective = self.deviance_with_response_constants(n_agq);
        let n_joint_params = start_beta.len() + start_theta.len();
        self.lmm.optsum.optimizer = joint_optimizer;
        self.lmm.optsum.backend = joint_optimizer.canonical_backend();
        self.lmm.optsum.optimizer_source = saved_optimizer_source;
        self.lmm.optsum.caller_set_fields = saved_caller_set_fields;
        let maxeval =
            joint_glmm_configured_maxeval_for(&self.lmm.optsum, n_joint_params, joint_optimizer);
        self.fit_joint_glmm_from_start(
            start_beta,
            start_theta,
            profiled_start_objective,
            n_agq,
            maxeval,
            Some(fallback_fast_pirls),
        )
    }

    pub(super) fn fit_joint_glmm_from_start(
        &mut self,
        start_beta: Vec<f64>,
        start_theta: Vec<f64>,
        profiled_start_objective: f64,
        n_agq: usize,
        maxeval: u32,
        fallback_fast_pirls: Option<Self>,
    ) -> Result<&mut Self> {
        let optimizer = self
            .lmm
            .optsum
            .caller_selected_optimizer()
            .unwrap_or_else(|| match self.lmm.optsum.optimizer {
                Optimizer::TrustBq | Optimizer::NloptBobyqa => self.lmm.optsum.optimizer,
                _ => default_joint_glmm_optimizer(),
            });
        self.lmm.optsum.optimizer = optimizer;
        self.lmm.optsum.backend = optimizer.canonical_backend();
        match optimizer {
            Optimizer::TrustBq => self.fit_joint_glmm_from_start_trust_bq(
                start_beta,
                start_theta,
                profiled_start_objective,
                n_agq,
                maxeval,
                fallback_fast_pirls,
            ),
            Optimizer::NloptBobyqa => {
                #[cfg(feature = "nlopt")]
                {
                    self.fit_joint_glmm_from_start_nlopt_bobyqa(
                        start_beta,
                        start_theta,
                        profiled_start_objective,
                        n_agq,
                        maxeval,
                        fallback_fast_pirls,
                    )
                }
                #[cfg(not(feature = "nlopt"))]
                {
                    let _ = (
                        start_beta,
                        start_theta,
                        profiled_start_objective,
                        n_agq,
                        maxeval,
                        fallback_fast_pirls,
                    );
                    Err(MixedModelError::Unsupported(
                        "joint GLMM NloptBobyqa requires the `nlopt` feature; rebuild with `--features nlopt` or pick TrustBq"
                            .to_string(),
                    ))
                }
            }
            optimizer => Err(MixedModelError::Unsupported(format!(
                "Optimizer::{optimizer:?} is not wired for joint GLMM fits; pick TrustBq or NloptBobyqa where available"
            ))),
        }
    }

    #[cfg(feature = "nlopt")]
    fn fit_joint_glmm_from_start_nlopt_bobyqa(
        &mut self,
        start_beta: Vec<f64>,
        start_theta: Vec<f64>,
        profiled_start_objective: f64,
        n_agq: usize,
        maxeval: u32,
        fallback_fast_pirls: Option<Self>,
    ) -> Result<&mut Self> {
        use nlopt::{Algorithm as NloptAlgorithm, Nlopt, Target as NloptTarget};

        let n_beta = self.beta.len();
        let n_theta = self.theta.len();
        let n_params = n_beta + n_theta;
        let mut initial = start_beta;
        initial.extend(start_theta);
        debug_assert_eq!(initial.len(), n_params);

        let mut lower_bounds = vec![f64::NEG_INFINITY; n_beta];
        lower_bounds.extend(self.lmm.lower_bounds());
        self.lmm.optsum.optimizer = Optimizer::NloptBobyqa;
        self.lmm.optsum.backend = Optimizer::NloptBobyqa.canonical_backend();
        self.lmm.optsum.finitial = profiled_start_objective;
        let ftol_rel = if self.lmm.optsum.caller_set_field("ftol_rel") {
            self.lmm.optsum.ftol_rel
        } else {
            1e-10
        };
        let ftol_abs = if self.lmm.optsum.caller_set_field("ftol_abs") {
            self.lmm.optsum.ftol_abs
        } else {
            1e-7
        };
        let xtol_rel = self
            .lmm
            .optsum
            .caller_set_field("xtol_rel")
            .then_some(self.lmm.optsum.xtol_rel);
        let mut initial_step = vec![0.1; n_beta];
        if self.lmm.optsum.caller_set_field("initial_step") {
            initial_step.extend(self.lmm.optsum.initial_step.clone());
        } else {
            initial_step.extend(vec![0.5; n_theta]);
        }

        let feval_count = std::cell::Cell::new(0i64);
        let fit_log: Rc<RefCell<Vec<FitLogEntry>>> = Rc::new(RefCell::new(Vec::new()));
        let model = std::cell::RefCell::new(self);
        let obj_fn = |params: &[f64], _grad: Option<&mut [f64]>, _data: &mut ()| -> f64 {
            feval_count.set(feval_count.get() + 1);
            let objective = model
                .borrow_mut()
                .joint_glmm_deviance_at_params(params, n_beta, n_agq);
            fit_log.borrow_mut().push(FitLogEntry {
                theta: params.to_vec(),
                objective,
            });
            objective
        };

        let mut optimizer = Nlopt::new(
            NloptAlgorithm::Bobyqa,
            n_params,
            obj_fn,
            NloptTarget::Minimize,
            (),
        );
        optimizer.set_lower_bounds(&lower_bounds).ok();
        optimizer.set_ftol_rel(ftol_rel).ok();
        optimizer.set_ftol_abs(ftol_abs).ok();
        if let Some(xtol_rel) = xtol_rel {
            optimizer.set_xtol_rel(xtol_rel).ok();
        }
        optimizer.set_maxeval(maxeval).ok();
        optimizer.set_initial_step(&initial_step).ok();

        let mut params = initial;
        let nlopt_result = optimizer.optimize(&mut params);
        drop(optimizer);

        let me = model.into_inner();
        let final_objective = me.joint_glmm_deviance_at_params(&params, n_beta, n_agq);
        me.refresh_dispersion();
        let status_prefix = joint_glmm_status_prefix(n_agq);
        let status_label = match &nlopt_result {
            Ok((status, _fmin)) => {
                format!(
                    "{status_prefix}:{}",
                    experimental_nlopt_status_label(&format!("{status:?}"))
                )
            }
            Err((status, _fmin)) => {
                format!(
                    "{status_prefix}_FAILED:{}",
                    experimental_nlopt_status_label(&format!("{status:?}"))
                )
            }
        };
        me.lmm.optsum.return_value = status_label;
        me.lmm.optsum.n_agq = n_agq;
        me.lmm.optsum.feval = feval_count.get();
        me.lmm.optsum.max_feval = maxeval as i64;
        me.lmm.optsum.fit_log = rc_refcell_into_inner_or_clone(fit_log);
        me.lmm.optsum.fmin = final_objective;
        me.lmm.optsum.final_params = params;
        let mut lower_bounds = vec![f64::NEG_INFINITY; n_beta];
        lower_bounds.extend(me.lmm.lower_bounds());
        let mut certificate = OptimizerCertificate::from_opt_summary_with_context(
            &me.lmm.optsum,
            &me.lmm.optsum.final_params,
            &lower_bounds,
            Some(me.lmm.dims.n),
        );
        let certification_gradient = me.joint_laplace_certification_gradient(
            &me.lmm.optsum.final_params.clone(),
            n_beta,
            n_agq,
            &lower_bounds,
            2.0e-2,
        );
        certificate.apply_derivative_evidence(
            OptimizerDerivativeEvidence {
                method: EvidenceMethod::FiniteDifference,
                gradient: certification_gradient.gradient.clone(),
                hessian: None,
            },
            2.0e-2,
            1.0e-6,
        );
        annotate_glmm_covariance_status(
            &mut certificate,
            &me.lmm.optsum.final_params,
            n_beta,
            &lower_bounds,
            &certification_gradient,
            2.0e-2,
        );
        if joint_certificate_requires_fallback(&certificate)
            && joint_candidate_materially_improves_profiled_start(&me.lmm.optsum)
        {
            record_uncertified_joint_candidate_diagnostic(&mut certificate, &me.lmm.optsum);
            me.lmm.compiler_artifact.optimizer_certificate = Some(certificate);
            me.record_glmm_fit_metadata();
            me.refresh_binomial_separation_diagnostics();
            me.refresh_near_unit_random_effect_correlation_diagnostics();
            return Ok(me);
        }
        if let Some(fallback) =
            uncertified_joint_fallback(&certificate, &me.lmm.optsum, fallback_fast_pirls)
        {
            *me = fallback;
            me.refresh_binomial_separation_diagnostics();
            me.refresh_near_unit_random_effect_correlation_diagnostics();
            return Ok(me);
        }
        me.lmm.compiler_artifact.optimizer_certificate = Some(certificate);
        me.record_glmm_fit_metadata();
        me.refresh_binomial_separation_diagnostics();
        me.refresh_near_unit_random_effect_correlation_diagnostics();
        Ok(me)
    }

    fn fit_joint_glmm_from_start_trust_bq(
        &mut self,
        start_beta: Vec<f64>,
        start_theta: Vec<f64>,
        profiled_start_objective: f64,
        n_agq: usize,
        maxeval: u32,
        fallback_fast_pirls: Option<Self>,
    ) -> Result<&mut Self> {
        let mut fallback_fast_pirls = fallback_fast_pirls;
        let n_beta = self.beta.len();
        let n_theta = self.theta.len();
        let n_params = n_beta + n_theta;
        let mut initial = start_beta;
        initial.extend(start_theta);
        debug_assert_eq!(initial.len(), n_params);

        let mut lower_bounds = vec![f64::NEG_INFINITY; n_beta];
        lower_bounds.extend(self.lmm.lower_bounds());
        let upper_bounds = vec![f64::INFINITY; n_params];
        self.lmm.optsum.optimizer = Optimizer::TrustBq;
        self.lmm.optsum.backend = Optimizer::TrustBq.canonical_backend();
        self.lmm.optsum.finitial = profiled_start_objective;
        self.lmm.optsum.max_feval = maxeval as i64;
        let ftol_abs = self.lmm.optsum.ftol_abs.max(1.0e-7);
        let ftol_rel = self.lmm.optsum.ftol_rel.max(1.0e-10);
        let initial_radius = joint_glmm_trust_bq_initial_radius(&initial, n_beta);
        let compact_joint_space = (5..=8).contains(&n_params);

        let invalid_objective = profiled_start_objective.abs().max(1.0)
            + 1.0e6 * (1.0 + profiled_start_objective.abs());
        let best_params = RefCell::new(initial.clone());
        let best_fmin = Cell::new(profiled_start_objective);
        let fit_log: Rc<RefCell<Vec<FitLogEntry>>> = Rc::new(RefCell::new(Vec::new()));

        let progress_callback = self.lmm.progress_callback.clone();
        let model = std::cell::RefCell::new(self);
        let mut objective_fn = |params: &[f64]| -> Result<f64> {
            let raw_objective = model
                .borrow_mut()
                .joint_glmm_deviance_at_params(params, n_beta, n_agq);
            let objective = if raw_objective.is_finite() {
                raw_objective
            } else {
                invalid_objective
            };
            fit_log.borrow_mut().push(FitLogEntry {
                theta: params.to_vec(),
                objective,
            });
            if raw_objective.is_finite() && objective < best_fmin.get() {
                best_fmin.set(objective);
                *best_params.borrow_mut() = params.to_vec();
            }
            Ok(objective)
        };
        let mut last_progress = 0usize;
        let mut progress_fn = |progress: &TrustBqProgress<'_>| -> Result<bool> {
            if let Some(callback) = &progress_callback {
                callback.report_if_due(
                    FitProgressPhase::JointGlmmOptimizer,
                    progress.fevals,
                    Some(maxeval.max(1) as usize),
                    &mut last_progress,
                )?;
            }
            Ok(false)
        };

        let result = minimize_trust_bq_with_progress(
            &initial,
            &lower_bounds,
            &upper_bounds,
            TrustBqOptions {
                initial_radius,
                final_radius: 1.0e-5,
                max_evaluations: maxeval.max(1) as usize,
                ftol_abs,
                ftol_rel,
                ftol_requires_local_radius: true,
                max_cross_terms: if compact_joint_space { usize::MAX } else { 0 },
                stall_iterations: if compact_joint_space { 4 } else { 3 },
                stall_ftol_abs: if compact_joint_space { -1.0 } else { 1.0e-6 },
                stall_ftol_rel: if compact_joint_space { -1.0 } else { 1.0e-8 },
                stall_requires_stable_x: compact_joint_space,
                reuse_samples: true,
                ..TrustBqOptions::default()
            },
            &mut objective_fn,
            &mut progress_fn,
        )?;

        let logged_best_params = best_params.into_inner();
        let logged_best_fmin = best_fmin.get();
        let (mut params, candidate_objective) =
            if logged_best_fmin.is_finite() && logged_best_fmin <= result.fmin {
                (logged_best_params, logged_best_fmin)
            } else {
                (result.x, result.fmin)
            };
        let me = model.into_inner();
        let status_prefix = joint_glmm_status_prefix(n_agq);
        me.lmm.optsum.return_value = format!(
            "{status_prefix}:{}",
            trust_bq_status_label(result.stop_reason)
        );
        me.lmm.optsum.n_agq = n_agq;
        me.lmm.optsum.feval = result.fevals as i64;
        me.lmm.optsum.max_feval = maxeval as i64;
        me.lmm.optsum.fit_log = rc_refcell_into_inner_or_clone(fit_log);
        me.lmm.optsum.fmin = candidate_objective;
        me.lmm.optsum.final_trust_radius = Some(result.final_radius);
        me.lmm.optsum.final_params = params.clone();

        let mut lower_bounds = vec![f64::NEG_INFINITY; n_beta];
        lower_bounds.extend(me.lmm.lower_bounds());
        let mut certificate = OptimizerCertificate::from_opt_summary_with_context(
            &me.lmm.optsum,
            &me.lmm.optsum.final_params,
            &lower_bounds,
            Some(me.lmm.dims.n),
        );
        let optimizer_stop_requires_fallback = !certificate.evidence.optimizer_stop.acceptable_stop;
        if optimizer_stop_requires_fallback
            && joint_candidate_materially_improves_profiled_start(&me.lmm.optsum)
        {
            let final_objective = me.joint_glmm_deviance_at_params(&params, n_beta, n_agq);
            me.refresh_dispersion();
            me.lmm.optsum.fmin = final_objective;
            me.lmm.optsum.final_params = std::mem::take(&mut params);
            certificate = OptimizerCertificate::from_opt_summary_with_context(
                &me.lmm.optsum,
                &me.lmm.optsum.final_params,
                &lower_bounds,
                Some(me.lmm.dims.n),
            );
            record_uncertified_joint_candidate_diagnostic(&mut certificate, &me.lmm.optsum);
            me.lmm.compiler_artifact.optimizer_certificate = Some(certificate);
            me.record_glmm_fit_metadata();
            me.refresh_binomial_separation_diagnostics();
            me.refresh_near_unit_random_effect_correlation_diagnostics();
            return Ok(me);
        }
        if fallback_fast_pirls.is_some() && optimizer_stop_requires_fallback {
            if let Some(fallback) =
                uncertified_joint_fallback(&certificate, &me.lmm.optsum, fallback_fast_pirls.take())
            {
                *me = fallback;
                me.refresh_binomial_separation_diagnostics();
                me.refresh_near_unit_random_effect_correlation_diagnostics();
                return Ok(me);
            }
        }

        if compact_joint_space {
            if let Some(polished) =
                me.polish_joint_laplace_stationarity(&params, &lower_bounds, 4, 2.0e-2)
            {
                params = polished;
            }
        }

        let final_objective = me.joint_glmm_deviance_at_params(&params, n_beta, n_agq);
        me.refresh_dispersion();
        me.lmm.optsum.fmin = final_objective;
        me.lmm.optsum.final_params = std::mem::take(&mut params);
        certificate = OptimizerCertificate::from_opt_summary_with_context(
            &me.lmm.optsum,
            &me.lmm.optsum.final_params,
            &lower_bounds,
            Some(me.lmm.dims.n),
        );
        let mut certification_gradient = me.joint_laplace_certification_gradient(
            &me.lmm.optsum.final_params.clone(),
            n_beta,
            n_agq,
            &lower_bounds,
            2.0e-2,
        );
        // trust_bq's derivative-free ftol stop can rest a steep, narrow
        // valley's width (~1e-3 deviance) short of the stationary point, where
        // the *assessed* gradient is genuinely above tolerance even though the
        // fit is reference-equivalent to several decimals. That failure is
        // polishable: take damped Newton steps to the stationary point and
        // re-certify, instead of surfacing fit_status=not_optimized on a fit
        // the polish can finish.
        if certificate.evidence.optimizer_stop.acceptable_stop
            && certification_gradient_assessed_free_failure(
                &certification_gradient,
                &me.lmm.optsum.final_params,
                &lower_bounds,
                2.0e-2,
            )
        {
            if let Some(polished) = me.polish_joint_laplace_stationarity(
                &me.lmm.optsum.final_params.clone(),
                &lower_bounds,
                4,
                2.0e-2,
            ) {
                let polished_objective = me.joint_glmm_deviance_at_params(&polished, n_beta, n_agq);
                if polished_objective.is_finite() && polished_objective <= me.lmm.optsum.fmin {
                    me.refresh_dispersion();
                    me.lmm.optsum.fmin = polished_objective;
                    me.lmm.optsum.final_params = polished;
                    certificate = OptimizerCertificate::from_opt_summary_with_context(
                        &me.lmm.optsum,
                        &me.lmm.optsum.final_params,
                        &lower_bounds,
                        Some(me.lmm.dims.n),
                    );
                    certification_gradient = me.joint_laplace_certification_gradient(
                        &me.lmm.optsum.final_params.clone(),
                        n_beta,
                        n_agq,
                        &lower_bounds,
                        2.0e-2,
                    );
                }
            }
        }
        certificate.apply_derivative_evidence(
            OptimizerDerivativeEvidence {
                method: EvidenceMethod::FiniteDifference,
                gradient: certification_gradient.gradient.clone(),
                hessian: None,
            },
            2.0e-2,
            1.0e-6,
        );
        annotate_glmm_covariance_status(
            &mut certificate,
            &me.lmm.optsum.final_params,
            n_beta,
            &lower_bounds,
            &certification_gradient,
            2.0e-2,
        );
        if joint_certificate_requires_fallback(&certificate)
            && joint_candidate_materially_improves_profiled_start(&me.lmm.optsum)
        {
            record_uncertified_joint_candidate_diagnostic(&mut certificate, &me.lmm.optsum);
            me.lmm.compiler_artifact.optimizer_certificate = Some(certificate);
            me.record_glmm_fit_metadata();
            me.refresh_binomial_separation_diagnostics();
            me.refresh_near_unit_random_effect_correlation_diagnostics();
            return Ok(me);
        }
        if let Some(fallback) =
            uncertified_joint_fallback(&certificate, &me.lmm.optsum, fallback_fast_pirls)
        {
            *me = fallback;
            me.refresh_binomial_separation_diagnostics();
            me.refresh_near_unit_random_effect_correlation_diagnostics();
            return Ok(me);
        }
        me.lmm.compiler_artifact.optimizer_certificate = Some(certificate);
        me.record_glmm_fit_metadata();
        me.refresh_binomial_separation_diagnostics();
        me.refresh_near_unit_random_effect_correlation_diagnostics();
        Ok(me)
    }

    pub(super) fn joint_glmm_deviance_at_params(
        &mut self,
        params: &[f64],
        n_beta: usize,
        n_agq: usize,
    ) -> f64 {
        if params.len() != n_beta + self.theta.len()
            || !params.iter().all(|value| value.is_finite())
        {
            return f64::INFINITY;
        }
        self.beta = DVector::from_column_slice(&params[..n_beta]);
        let theta = &params[n_beta..];
        match self.update_pirls_at_theta(theta, false) {
            Ok(_) => {
                let deviance = self.deviance_with_response_constants(n_agq);
                if deviance.is_finite() {
                    deviance
                } else {
                    f64::INFINITY
                }
            }
            Err(_) => f64::INFINITY,
        }
    }

    fn joint_glmm_deviance_at_params_for_hessian(
        &mut self,
        params: &[f64],
        n_beta: usize,
        n_agq: usize,
    ) -> std::result::Result<f64, String> {
        if params.len() != n_beta + self.theta.len() {
            return Err(format!(
                "parameter vector has length {}, expected {} fixed effects plus {} covariance parameters",
                params.len(),
                n_beta,
                self.theta.len()
            ));
        }
        if !params.iter().all(|value| value.is_finite()) {
            return Err("parameter vector contains non-finite values".to_string());
        }

        self.beta = DVector::from_column_slice(&params[..n_beta]);
        let theta = &params[n_beta..];
        self.update_pirls_at_theta_with_options(theta, false, GLMM_HESSIAN_PIRLS_MAX_ITER, true)
            .map_err(|error| format!("conditional-mode PIRLS probe failed: {error}"))?;

        let deviance = self.deviance_with_response_constants(n_agq);
        if deviance.is_finite() {
            Ok(deviance)
        } else {
            Err("probe objective is non-finite after the certification PIRLS probe".to_string())
        }
    }

    fn joint_laplace_finite_difference_gradient(
        &mut self,
        params: &[f64],
        n_beta: usize,
        n_agq: usize,
        lower_bounds: &[f64],
    ) -> Vec<f64> {
        let gradient = (0..params.len())
            .map(|index| {
                let h = JOINT_LAPLACE_FD_RELATIVE_STEP * params[index].abs().max(1.0);
                self.joint_laplace_fd_gradient_component(
                    params,
                    index,
                    h,
                    n_beta,
                    n_agq,
                    lower_bounds,
                )
            })
            .collect();
        let _ = self.joint_glmm_deviance_at_params(params, n_beta, n_agq);
        gradient
    }

    fn joint_laplace_fd_gradient_component(
        &mut self,
        params: &[f64],
        index: usize,
        h: f64,
        n_beta: usize,
        n_agq: usize,
        lower_bounds: &[f64],
    ) -> f64 {
        let value = params[index];
        let lower = lower_bounds
            .get(index)
            .copied()
            .unwrap_or(f64::NEG_INFINITY);
        let mut plus = params.to_vec();
        plus[index] = value + h;
        let fp = self.joint_glmm_deviance_at_params(&plus, n_beta, n_agq);
        if value - h > lower {
            let mut minus = params.to_vec();
            minus[index] = value - h;
            let fm = self.joint_glmm_deviance_at_params(&minus, n_beta, n_agq);
            (fp - fm) / (2.0 * h)
        } else {
            let base = self.joint_glmm_deviance_at_params(params, n_beta, n_agq);
            (fp - base) / h
        }
    }

    /// Stationarity gradient for the joint-Laplace certificate, robust to the
    /// inner-PIRLS deviance noise floor.
    ///
    /// The deviance returned by a PIRLS solve carries an O(1e-5) absolute
    /// error from its own stopping rule, so a finite difference at the default
    /// step `1e-5 * scale` amplifies that error to an O(0.1-1) gradient
    /// reading in directions where the surface is nearly flat — exactly the
    /// directions a converged fit produces. Components whose default-step
    /// reading exceeds the tolerance are therefore re-probed at two larger
    /// steps where the deviance signal dominates the PIRLS noise. If the two
    /// large-step estimates agree, that estimate is the assessed gradient
    /// (which may still fail the tolerance — a genuine non-stationarity). If
    /// they disagree, the component cannot be assessed at any trusted step and
    /// is reported as such rather than as a failure.
    fn joint_laplace_certification_gradient(
        &mut self,
        params: &[f64],
        n_beta: usize,
        n_agq: usize,
        lower_bounds: &[f64],
        gradient_tolerance: f64,
    ) -> JointLaplaceCertificationGradient {
        let probe_gradient =
            self.joint_laplace_finite_difference_gradient(params, n_beta, n_agq, lower_bounds);
        let mut gradient = probe_gradient.clone();
        let mut escalated_indices = Vec::new();
        let mut unassessable_indices = Vec::new();
        for (index, &value) in params.iter().enumerate() {
            let raw = probe_gradient[index];
            if raw.is_finite() && raw.abs() <= gradient_tolerance {
                continue;
            }
            let scale = value.abs().max(1.0);
            let estimates = JOINT_LAPLACE_CERT_FD_ESCALATED_RELATIVE_STEPS.map(|step| {
                self.joint_laplace_fd_gradient_component(
                    params,
                    index,
                    step * scale,
                    n_beta,
                    n_agq,
                    lower_bounds,
                )
            });
            let consistent = estimates.iter().all(|estimate| estimate.is_finite())
                && (estimates[0] - estimates[1]).abs() <= gradient_tolerance;
            if consistent {
                gradient[index] = estimates[1];
                escalated_indices.push(index);
            } else {
                unassessable_indices.push(index);
            }
        }
        if !(escalated_indices.is_empty() && unassessable_indices.is_empty()) {
            let _ = self.joint_glmm_deviance_at_params(params, n_beta, n_agq);
        }
        JointLaplaceCertificationGradient {
            gradient,
            probe_gradient,
            escalated_indices,
            unassessable_indices,
        }
    }

    pub(super) fn glmm_joint_laplace_fixed_effect_inference_artifacts(
        &mut self,
    ) -> std::result::Result<GlmmFixedEffectInferenceArtifacts, String> {
        let p = self.beta.len();
        let n_theta = self.theta.len();
        let full_coef_names = self.coef_names();
        if self.lmm.feterm.rank != full_coef_names.len() {
            return Err(
                "joint-laplace GLMM Wald inference is unavailable for rank-deficient fixed effects"
                    .to_string(),
            );
        }

        let params = self.lmm.optsum.final_params.clone();
        if params.len() != p + n_theta {
            return Err(format!(
                "joint-laplace GLMM final parameter vector has length {}, expected {} fixed effects plus {} covariance parameters",
                params.len(),
                p,
                n_theta
            ));
        }

        let mut lower_bounds = vec![f64::NEG_INFINITY; p];
        lower_bounds.extend(self.lmm.lower_bounds());
        let mut active_indices = (0..p).collect::<Vec<_>>();
        let mut omitted_boundary_theta_indices = Vec::new();
        for index in p..params.len() {
            let lower = lower_bounds[index];
            if lower.is_finite() && params[index] <= lower + glmm_hessian_step(params[index]) {
                omitted_boundary_theta_indices.push(index - p + 1);
            } else {
                active_indices.push(index);
            }
        }

        let hessian = self.finite_difference_joint_laplace_hessian_for_indices(
            &params,
            &lower_bounds,
            &active_indices,
            true,
        )?;
        let certification = certify_glmm_joint_hessian(&hessian, "joint-laplace GLMM Hessian")?;
        let beta_covariance = 2.0 * certification.inverse.view((0, 0), (p, p)).into_owned();
        if !matrix_is_finite_local(&beta_covariance) {
            return Err(
                "joint-laplace GLMM fixed-effect covariance contains non-finite entries"
                    .to_string(),
            );
        }
        let full_covariance = unpivot_glmm_fixed_effect_covariance(
            &beta_covariance,
            &self.lmm.feterm.piv,
            full_coef_names.len(),
        );
        let covariance_payload = glmm_joint_laplace_fixed_effect_covariance_matrix(
            full_coef_names.clone(),
            &full_covariance,
            self.lmm.feterm.rank,
            &certification,
            &omitted_boundary_theta_indices,
        )?;
        let inference_notes =
            glmm_joint_laplace_hessian_notes(&certification, &omitted_boundary_theta_indices);

        let normal = Normal::new(0.0, 1.0)
            .map_err(|err| format!("normal reference distribution unavailable: {err}"))?;
        let estimates = self.coef();
        let mut std_errors = vec![f64::NAN; full_coef_names.len()];
        for full_index in 0..full_coef_names.len() {
            let variance = full_covariance[(full_index, full_index)];
            if !variance.is_finite() || variance <= 0.0 {
                return Err(format!(
                    "joint-laplace GLMM fixed-effect covariance has invalid variance for coefficient {}",
                    full_coef_names
                        .get(full_index)
                        .cloned()
                        .unwrap_or_else(|| full_index.to_string())
                ));
            }
            std_errors[full_index] = variance.sqrt();
        }

        let rows = full_coef_names
            .into_iter()
            .enumerate()
            .map(|(index, label)| {
                let estimate = estimates
                    .get(index)
                    .copied()
                    .filter(|value| value.is_finite());
                let std_error = std_errors
                    .get(index)
                    .copied()
                    .filter(|value| value.is_finite() && *value > 0.0);
                let statistic = estimate.zip(std_error).map(|(estimate, se)| estimate / se);
                let p_value = statistic.map(|z| 2.0 * (1.0 - normal.cdf(z.abs())));
                FixedEffectInferenceRow {
                    label: label.clone(),
                    kind: FixedEffectInferenceRowKind::Coefficient,
                    estimate,
                    std_error,
                    numerator_df: None,
                    denominator_df: None,
                    statistic,
                    statistic_name: Some(crate::compiler::FixedEffectStatisticName::Z),
                    p_value,
                    method: FixedEffectInferenceMethod::AsymptoticWaldZ,
                    status: FixedEffectInferenceStatus::Available,
                    reliability: ReliabilityGrade::Moderate,
                    reliability_reason: Some(
                        FixedEffectReliabilityReason::GlmmJointLaplaceActiveHessianWald,
                    ),
                    estimability: EstimabilityAssessment::FixedContrast(
                        FixedContrastEstimability::estimable(label, 1, 1),
                    ),
                    reason: None,
                    details: None,
                    notes: inference_notes.clone(),
                }
            })
            .collect();

        Ok(GlmmFixedEffectInferenceArtifacts {
            table: FixedEffectInferenceTable::new(rows),
            covariance: Some(covariance_payload),
        })
    }

    fn finite_difference_joint_laplace_hessian(
        &mut self,
        params: &[f64],
        lower_bounds: &[f64],
    ) -> std::result::Result<DMatrix<f64>, String> {
        let active_indices = (0..params.len()).collect::<Vec<_>>();
        self.finite_difference_joint_laplace_hessian_for_indices(
            params,
            lower_bounds,
            &active_indices,
            false,
        )
    }

    fn finite_difference_joint_laplace_hessian_for_indices(
        &mut self,
        params: &[f64],
        lower_bounds: &[f64],
        active_indices: &[usize],
        use_hessian_certification_probe: bool,
    ) -> std::result::Result<DMatrix<f64>, String> {
        let n = active_indices.len();
        let p = self.beta.len();
        let n_agq = self.lmm.optsum.n_agq;

        macro_rules! eval_hessian_probe {
            ($probe:expr, $context:expr) => {
                if use_hessian_certification_probe {
                    match self.joint_glmm_deviance_at_params_for_hessian($probe, p, n_agq) {
                        Ok(value) => value,
                        Err(reason) => {
                            let _ = self.joint_glmm_deviance_at_params(params, p, n_agq);
                            return Err(format!("{}: {}", $context, reason));
                        }
                    }
                } else {
                    let value = self.joint_glmm_deviance_at_params($probe, p, n_agq);
                    if value.is_finite() {
                        value
                    } else {
                        let _ = self.joint_glmm_deviance_at_params(params, p, n_agq);
                        return Err(format!("{} is non-finite", $context));
                    }
                }
            };
        }

        let base = eval_hessian_probe!(
            params,
            "joint-laplace GLMM Hessian certificate base objective"
        );

        let mut steps = Vec::with_capacity(n);
        for &index in active_indices {
            let value = *params.get(index).ok_or_else(|| {
                format!(
                    "joint-laplace GLMM Hessian active parameter index {} is out of range",
                    index + 1
                )
            })?;
            let h = glmm_hessian_step(value);
            let lower = lower_bounds
                .get(index)
                .copied()
                .unwrap_or(f64::NEG_INFINITY);
            if lower.is_finite() && value - h <= lower {
                let _ = self.joint_glmm_deviance_at_params(params, p, n_agq);
                return Err(format!(
                    "joint-laplace GLMM Hessian central difference step for parameter {} would cross its lower bound",
                    index + 1
                ));
            }
            steps.push(h);
        }

        let mut hessian = DMatrix::zeros(n, n);
        for active_i in 0..n {
            let i = active_indices[active_i];
            let hi = steps[active_i];
            let mut plus = params.to_vec();
            plus[i] += hi;
            let f_plus = eval_hessian_probe!(
                &plus,
                format!(
                    "joint-laplace GLMM Hessian diagonal plus probe for parameter {}",
                    i + 1
                )
            );
            let mut minus = params.to_vec();
            minus[i] -= hi;
            let f_minus = eval_hessian_probe!(
                &minus,
                format!(
                    "joint-laplace GLMM Hessian diagonal minus probe for parameter {}",
                    i + 1
                )
            );
            hessian[(active_i, active_i)] = (f_plus - 2.0 * base + f_minus) / (hi * hi);

            for active_j in 0..active_i {
                let j = active_indices[active_j];
                let hj = steps[active_j];
                let mut pp = params.to_vec();
                pp[i] += hi;
                pp[j] += hj;
                let f_pp = eval_hessian_probe!(
                    &pp,
                    format!(
                        "joint-laplace GLMM Hessian off-diagonal ++ probe for parameters {} and {}",
                        i + 1,
                        j + 1
                    )
                );

                let mut pm = params.to_vec();
                pm[i] += hi;
                pm[j] -= hj;
                let f_pm = eval_hessian_probe!(
                    &pm,
                    format!(
                        "joint-laplace GLMM Hessian off-diagonal +- probe for parameters {} and {}",
                        i + 1,
                        j + 1
                    )
                );

                let mut mp = params.to_vec();
                mp[i] -= hi;
                mp[j] += hj;
                let f_mp = eval_hessian_probe!(
                    &mp,
                    format!(
                        "joint-laplace GLMM Hessian off-diagonal -+ probe for parameters {} and {}",
                        i + 1,
                        j + 1
                    )
                );

                let mut mm = params.to_vec();
                mm[i] -= hi;
                mm[j] -= hj;
                let f_mm = eval_hessian_probe!(
                    &mm,
                    format!(
                        "joint-laplace GLMM Hessian off-diagonal -- probe for parameters {} and {}",
                        i + 1,
                        j + 1
                    )
                );
                let value = (f_pp - f_pm - f_mp + f_mm) / (4.0 * hi * hj);
                hessian[(active_i, active_j)] = value;
                hessian[(active_j, active_i)] = value;
            }
        }
        let _ = self.joint_glmm_deviance_at_params(params, p, n_agq);

        Ok(hessian)
    }

    fn polish_joint_laplace_stationarity(
        &mut self,
        params: &[f64],
        lower_bounds: &[f64],
        max_iterations: usize,
        gradient_tolerance: f64,
    ) -> Option<Vec<f64>> {
        let p = self.beta.len();
        let n_agq = self.lmm.optsum.n_agq;
        let mut current = params.to_vec();
        let mut current_objective = self.joint_glmm_deviance_at_params(&current, p, n_agq);
        if !current_objective.is_finite() {
            return None;
        }

        for _ in 0..max_iterations {
            let certification = self.joint_laplace_certification_gradient(
                &current,
                p,
                n_agq,
                lower_bounds,
                gradient_tolerance,
            );
            // Polish only on assessed gradient signal: components the
            // noise-aware probe could not assess carry no usable descent
            // direction, and Newton steps on probe noise just burn a full
            // finite-difference Hessian before the line search rejects them.
            let mut gradient = certification.gradient;
            for &index in &certification.unassessable_indices {
                gradient[index] = 0.0;
            }
            let free_gradient_norm = gradient
                .iter()
                .map(|value| value.abs())
                .fold(0.0_f64, f64::max);
            if !free_gradient_norm.is_finite() || free_gradient_norm <= gradient_tolerance {
                break;
            }

            let hessian = self
                .finite_difference_joint_laplace_hessian(&current, lower_bounds)
                .ok()?;
            let step = hessian
                .cholesky()
                .map(|cholesky| cholesky.solve(&DVector::from_column_slice(&gradient)))?;
            if !step.iter().all(|value| value.is_finite()) {
                break;
            }
            let step_norm = step.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);
            if step_norm <= 1.0e-8 {
                break;
            }

            let mut accepted = None;
            for damping in [1.0, 0.5, 0.25, 0.125, 0.0625] {
                let mut trial = current.clone();
                for (index, value) in trial.iter_mut().enumerate() {
                    *value -= damping * step[index];
                    let lower = lower_bounds
                        .get(index)
                        .copied()
                        .unwrap_or(f64::NEG_INFINITY);
                    if lower.is_finite() && *value <= lower {
                        *value = lower + 1.0e-8;
                    }
                }
                if !trial.iter().all(|value| value.is_finite()) {
                    continue;
                }
                let trial_objective = self.joint_glmm_deviance_at_params(&trial, p, n_agq);
                if trial_objective.is_finite()
                    && trial_objective
                        < current_objective
                            - (1.0e-9 * current_objective.abs().max(1.0)).max(1.0e-9)
                {
                    accepted = Some((trial, trial_objective));
                    break;
                }
            }

            let Some((trial, trial_objective)) = accepted else {
                break;
            };
            current = trial;
            current_objective = trial_objective;
        }

        let _ = self.joint_glmm_deviance_at_params(&current, p, n_agq);
        Some(current)
    }

    pub(super) fn certified_joint_laplace_fixed_covariance(&self) -> Option<DMatrix<f64>> {
        let covariance = self
            .lmm
            .compiler_artifact
            .fixed_effect_covariance_matrix
            .as_ref()?;
        if covariance.status != FixedEffectCovarianceStatus::Available
            || covariance.method != FixedEffectCovarianceMethod::JointLaplaceActiveHessian
        {
            return None;
        }
        let matrix = covariance.matrix.as_ref()?;
        let p = self.lmm.feterm.rank;
        if matrix.len() != p || matrix.iter().any(|row| row.len() != p) {
            return None;
        }
        let values = matrix
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect::<Vec<_>>();
        let dense = DMatrix::from_row_slice(p, p, &values);
        matrix_is_finite_local(&dense).then_some(dense)
    }
}

pub(crate) fn default_joint_glmm_optimizer() -> Optimizer {
    #[cfg(feature = "nlopt")]
    {
        Optimizer::NloptBobyqa
    }
    #[cfg(not(feature = "nlopt"))]
    {
        Optimizer::TrustBq
    }
}

pub(crate) fn trust_bq_joint_glmm_default_maxeval(n_params: usize) -> u32 {
    // Native TrustBQ uses local quadratic models, so it should need far fewer
    // objective calls than the previous COBYLA fallback while still leaving
    // enough budget for mixed beta/theta scales on large-intercept Bernoulli
    // models.
    (500usize + 80usize * n_params.max(1)).min(8_000) as u32
}

pub(crate) fn joint_glmm_default_maxeval_for(optimizer: Optimizer, n_params: usize) -> u32 {
    match optimizer {
        Optimizer::TrustBq => trust_bq_joint_glmm_default_maxeval(n_params),
        Optimizer::NloptBobyqa => 200,
        _ => trust_bq_joint_glmm_default_maxeval(n_params),
    }
}

pub(crate) fn joint_glmm_configured_maxeval_for(
    optsum: &OptSummary,
    n_params: usize,
    optimizer: Optimizer,
) -> u32 {
    if optsum.max_feval > 0 {
        optsum.max_feval.min(u32::MAX as i64).max(1) as u32
    } else {
        joint_glmm_default_maxeval_for(optimizer, n_params)
    }
}

pub(crate) fn validate_joint_glmm_optimizer(optimizer: Optimizer) -> Result<()> {
    match optimizer {
        Optimizer::TrustBq => Ok(()),
        Optimizer::NloptBobyqa => {
            #[cfg(feature = "nlopt")]
            {
                Ok(())
            }
            #[cfg(not(feature = "nlopt"))]
            {
                Err(MixedModelError::Unsupported(
                    "joint GLMM NloptBobyqa requires the `nlopt` feature; rebuild with `--features nlopt` or pick TrustBq"
                        .to_string(),
                ))
            }
        }
        other => Err(MixedModelError::Unsupported(format!(
            "Optimizer::{other:?} is not wired for joint GLMM fits; pick TrustBq or NloptBobyqa where available"
        ))),
    }
}

pub(crate) fn joint_glmm_trust_bq_initial_radius(initial: &[f64], n_beta: usize) -> f64 {
    let beta_scale = initial
        .iter()
        .take(n_beta)
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    // Keep beta moves large enough to repair high-baseline intercept starts,
    // but do not let one large coefficient make theta probes excessive.
    (0.25 * beta_scale).clamp(0.25, 1.0)
}

pub(crate) fn trust_bq_status_label(status: TrustBqStopReason) -> &'static str {
    match status {
        TrustBqStopReason::RadiusBelowTolerance => "RADIUS_REACHED",
        TrustBqStopReason::ObjectiveTolerance => "FTOL_REACHED",
        TrustBqStopReason::MaxEvaluations => "MAXEVAL_REACHED",
        TrustBqStopReason::StepBelowTolerance => "XTOL_REACHED",
        TrustBqStopReason::ObjectiveStagnation => "FTOL_REACHED",
        TrustBqStopReason::CertifiedConvergence => "FTOL_REACHED",
    }
}

pub(crate) fn glmm_block_index(row: usize, col: usize) -> usize {
    debug_assert!(row >= col);
    row * (row + 1) / 2 + col
}

pub(crate) fn solve_dense_lower_against_rhs(l: &DMatrix<f64>, rhs: &mut [f64]) {
    for i in 0..rhs.len() {
        let mut sum = rhs[i];
        for j in 0..i {
            sum -= l[(i, j)] * rhs[j];
        }
        rhs[i] = sum / l[(i, i)];
    }
}

pub(crate) fn solve_dense_upper_from_lower_transpose_against_rhs(
    l: &DMatrix<f64>,
    rhs: &mut [f64],
) {
    for i in (0..rhs.len()).rev() {
        let mut sum = rhs[i];
        for j in (i + 1)..rhs.len() {
            sum -= l[(j, i)] * rhs[j];
        }
        rhs[i] = sum / l[(i, i)];
    }
}
