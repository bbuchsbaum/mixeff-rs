//! Profiled/fast-PIRLS GLMM fit drivers and optimizer plumbing (native
//! COBYLA, pattern search, and the cfg-gated NLopt alternates).
//!
//! Moved verbatim from `generalized/mod.rs` during the module split
//! (bd-01KWHYQSTWK60P6HA4S4B2K99P). No logic changes.

use super::*;

impl GeneralizedLinearMixedModel {
    /// Fit the GLMM.
    pub fn fit(&mut self) -> Result<&mut Self> {
        self.fit_with_glmm_options(GlmmFitOptions::default())
    }

    /// Refit the GLMM to a new response vector from the recorded initial θ.
    ///
    /// This mirrors Julia's `refit!` semantics for bootstrap and simulation
    /// workflows: the optimizer starts from `optsum.initial`, not from the
    /// previous optimum.
    pub fn refit(&mut self, new_y: &[f64]) -> Result<&mut Self> {
        let n_agq = self.lmm.optsum.n_agq.max(1);
        self.refit_with_options(new_y, n_agq, false)
    }

    /// Refit the GLMM to a new response vector with an explicit AGQ setting.
    pub fn refit_with_options(
        &mut self,
        new_y: &[f64],
        n_agq: usize,
        verbose: bool,
    ) -> Result<&mut Self> {
        if let Err(error) = self.validate_agq(n_agq) {
            self.record_invalid_agq_diagnostic(n_agq, &error.to_string());
            return Err(error);
        }
        self.reset_for_refit(Some(new_y))?;
        self.fit_with_options(true, n_agq, verbose)
    }

    /// Fit after first applying a compiler policy.
    pub fn fit_with_compiler_policy(
        &mut self,
        compiler_policy: CompilerPolicy,
    ) -> Result<&mut Self> {
        self.set_compiler_policy(compiler_policy)?;
        self.fit()
    }

    /// Fit with options.
    ///
    /// `fast` selects the MixedModels.jl-style fast path, which profiles over
    /// θ and updates β through PIRLS. `fast = false` selects the certified
    /// joint path: joint Laplace for `n_agq <= 1`, and joint AGQ for valid
    /// single-scalar random-effect models with `n_agq > 1`. NLopt builds use
    /// BOBYQA; dependency-light builds use the native TrustBQ joint path.
    ///
    /// `n_agq` selects the deviance approximation: `1` (or `0`) means the
    /// Laplace approximation; values `>= 2` request `n_agq`-point adaptive
    /// Gauss-Hermite quadrature, which is only valid for models with a
    /// single scalar random-effects term and is rejected up front
    /// otherwise.
    pub fn fit_with_options(
        &mut self,
        fast: bool,
        n_agq: usize,
        verbose: bool,
    ) -> Result<&mut Self> {
        self.fit_with_glmm_options(GlmmFitOptions {
            fast,
            n_agq,
            verbose,
            optimizer_control: OptimizerControl::default(),
            progress_callback: None,
        })
    }

    /// Fit with explicit GLMM options.
    pub fn fit_with_glmm_options(&mut self, options: GlmmFitOptions) -> Result<&mut Self> {
        let GlmmFitOptions {
            fast,
            n_agq,
            verbose,
            optimizer_control,
            progress_callback,
        } = options;
        if let Err(error) = self.validate_agq(n_agq) {
            self.record_invalid_agq_diagnostic(n_agq, &error.to_string());
            return Err(error);
        }
        if self.lmm.optsum.feval > 0 {
            return Err(MixedModelError::AlreadyFitted);
        }
        self.pending_progress_error = None;
        self.lmm.progress_callback = progress_callback;
        self.lmm.apply_optimizer_control(&optimizer_control)?;
        if let Some(start_theta) = &optimizer_control.start_theta {
            self.theta = start_theta.clone();
        }
        if self.family == Family::NegativeBinomial && self.negative_binomial_estimate_theta {
            return self.fit_negative_binomial_estimated_theta(fast, n_agq, verbose);
        }
        if !fast {
            return self.fit_joint_glmm_with_response_constants(n_agq, verbose);
        }
        self.fit_with_options_impl(n_agq, verbose)
    }

    fn fit_negative_binomial_estimated_theta(
        &mut self,
        fast: bool,
        n_agq: usize,
        verbose: bool,
    ) -> Result<&mut Self> {
        let initial_theta = self.require_negative_binomial_theta()?;
        let mut current_theta = clamp_negative_binomial_theta(initial_theta);
        let mut last_fit_theta = f64::NAN;
        let mut update_iterations = 0usize;
        let mut converged = false;

        for iteration in 0..NEGATIVE_BINOMIAL_THETA_MAX_ITERS {
            if self.lmm.optsum.feval > 0 {
                self.reset_for_refit(None)?;
            }
            self.negative_binomial_theta = Some(current_theta);
            self.fit_negative_binomial_conditional(fast, n_agq, verbose)?;
            last_fit_theta = current_theta;

            let next_theta = self.estimate_negative_binomial_theta_given_fit()?;
            update_iterations = iteration + 1;
            let relative_change = relative_theta_change(current_theta, next_theta);
            if verbose {
                eprintln!(
                    "  NB theta outer iter {}: theta = {:.6}, updated = {:.6}, rel_change = {:.3e}",
                    iteration + 1,
                    current_theta,
                    next_theta,
                    relative_change
                );
            }
            current_theta = next_theta;
            if relative_change <= NEGATIVE_BINOMIAL_THETA_TOL {
                converged = true;
                break;
            }
        }

        if relative_theta_change(last_fit_theta, current_theta)
            > NEGATIVE_BINOMIAL_THETA_FINAL_REFIT_TOL
        {
            self.reset_for_refit(None)?;
            self.negative_binomial_theta = Some(current_theta);
            self.fit_negative_binomial_conditional(fast, n_agq, verbose)?;
            last_fit_theta = current_theta;
        }

        self.negative_binomial_theta = Some(last_fit_theta);
        self.refresh_dispersion();
        self.record_negative_binomial_theta_estimation_metadata(
            initial_theta,
            last_fit_theta,
            update_iterations,
            converged,
        );
        Ok(self)
    }

    fn fit_negative_binomial_conditional(
        &mut self,
        fast: bool,
        n_agq: usize,
        verbose: bool,
    ) -> Result<()> {
        if fast {
            self.fit_with_options_impl(n_agq, verbose)?;
        } else {
            self.fit_joint_glmm_with_response_constants(n_agq, verbose)?;
        }
        Ok(())
    }

    pub(super) fn configure_profile_start_optimizer(&mut self) {
        let optimizer = default_fast_glmm_optimizer();
        self.lmm.optsum.optimizer = optimizer;
        self.lmm.optsum.backend = optimizer.canonical_backend();
        self.lmm.optsum.optimizer_source = crate::types::OptimizerSource::Auto;
        self.lmm
            .optsum
            .caller_set_fields
            .retain(|field| field != "optimizer");
    }

    #[cfg(not(feature = "nlopt"))]
    pub(super) fn fit_with_options_impl(
        &mut self,
        n_agq: usize,
        _verbose: bool,
    ) -> Result<&mut Self> {
        match self.lmm.optsum.optimizer {
            Optimizer::PatternSearch => self.fit_native_pattern_search(n_agq),
            Optimizer::Cobyla => self.fit_native_cobyla(n_agq),
            Optimizer::TrustBq => Err(MixedModelError::Unsupported(
                "TrustBQ is reserved for the dependency-light fast=false joint GLMM path; pick COBYLA or pattern_search for fast-PIRLS GLMMs"
                    .to_string(),
            )),
            Optimizer::NloptBobyqa | Optimizer::NloptNewuoa => Err(MixedModelError::Unsupported(
                "NLopt GLMM optimizers require the `nlopt` feature; rebuild with `--features nlopt` or pick a native optimizer"
                    .to_string(),
            )),
            Optimizer::PrimaBobyqa
            | Optimizer::PrimaCobyla
            | Optimizer::PrimaLincoa
            | Optimizer::PrimaNewuoa => Err(MixedModelError::Unsupported(
                "PRIMA GLMM optimizers are not wired; pick a native optimizer".to_string(),
            )),
        }
    }

    #[cfg(feature = "nlopt")]
    pub(super) fn fit_with_options_impl(
        &mut self,
        n_agq: usize,
        _verbose: bool,
    ) -> Result<&mut Self> {
        use nlopt::{Algorithm as NloptAlgorithm, Nlopt, Target as NloptTarget};

        match self.lmm.optsum.caller_selected_optimizer() {
            Some(Optimizer::PatternSearch) => return self.fit_native_pattern_search(n_agq),
            Some(Optimizer::Cobyla) => return self.fit_native_cobyla(n_agq),
            Some(Optimizer::NloptBobyqa) | None => {}
            Some(Optimizer::TrustBq) => {
                return Err(MixedModelError::Unsupported(
                    "TrustBQ is reserved for fast=false joint GLMM fits; pick Cobyla, pattern_search, or NloptBobyqa for fast-PIRLS GLMMs"
                        .to_string(),
                ));
            }
            Some(Optimizer::NloptNewuoa) => {
                return Err(MixedModelError::Unsupported(
                    "NloptNewuoa is unconstrained and is not wired for bounded fast-PIRLS GLMM theta optimization; pick NloptBobyqa"
                        .to_string(),
                ));
            }
            Some(
                Optimizer::PrimaBobyqa
                | Optimizer::PrimaCobyla
                | Optimizer::PrimaLincoa
                | Optimizer::PrimaNewuoa,
            ) => {
                return Err(MixedModelError::Unsupported(
                    "PRIMA GLMM optimizers are not wired; pick Cobyla, pattern_search, or NloptBobyqa"
                        .to_string(),
                ));
            }
        }

        let n_theta = self.theta.len();
        let lb = self.lmm.lower_bounds();
        let initial_theta = self.lmm.optsum.initial.clone();
        self.lmm.optsum.optimizer = Optimizer::NloptBobyqa;
        self.lmm.optsum.backend = Optimizer::NloptBobyqa.canonical_backend();
        let ftol_rel = if self.lmm.optsum.caller_set_field("ftol_rel") {
            self.lmm.optsum.ftol_rel
        } else {
            1e-12
        };
        let ftol_abs = if self.lmm.optsum.caller_set_field("ftol_abs") {
            self.lmm.optsum.ftol_abs
        } else {
            1e-8
        };
        let xtol_rel = self
            .lmm
            .optsum
            .caller_set_field("xtol_rel")
            .then_some(self.lmm.optsum.xtol_rel);
        let xtol_abs = self
            .lmm
            .optsum
            .caller_set_field("xtol_abs")
            .then(|| self.lmm.optsum.xtol_abs.clone());
        let maxeval = if self.lmm.optsum.max_feval > 0 {
            self.lmm.optsum.max_feval.min(u32::MAX as i64).max(1) as u32
        } else {
            500
        };
        let initial_step = if self.lmm.optsum.caller_set_field("initial_step") {
            self.lmm.optsum.initial_step.clone()
        } else {
            vec![0.75; n_theta]
        };

        let feval_count = std::cell::Cell::new(0i64);
        let fit_log: Rc<RefCell<Vec<FitLogEntry>>> = Rc::new(RefCell::new(Vec::with_capacity(
            self.lmm.optsum.fit_log.capacity(),
        )));

        // Hand the model to the BOBYQA callback through a RefCell instead of a
        // raw `*mut Self`. The callback is the only borrower while the optimizer
        // is alive; `model.into_inner()` recovers `&mut self` once the optimizer
        // (and thus the closure) has been dropped.
        let model = std::cell::RefCell::new(self);
        let obj_fn = |theta: &[f64], _grad: Option<&mut [f64]>, _data: &mut ()| -> f64 {
            feval_count.set(feval_count.get() + 1);
            let objective = model
                .borrow_mut()
                .penalized_pirls_deviance_at_theta(theta, n_agq);
            fit_log.borrow_mut().push(FitLogEntry {
                theta: theta.to_vec(),
                objective,
            });
            objective
        };

        let mut optimizer = Nlopt::new(
            NloptAlgorithm::Bobyqa,
            n_theta,
            obj_fn,
            NloptTarget::Minimize,
            (),
        );
        optimizer.set_lower_bounds(&lb).ok();
        // Match MixedModels.jl OptSummary defaults: ftol_rel=1e-12,
        // ftol_abs=1e-8, xtol_rel=0. Setting xtol_rel=1e-8 here previously
        // forced BOBYQA to shrink its trust region (ρ_beg → ρ_end) all the
        // way to 1e-8 before exploring multi-dim moves, which on multi-RE
        // GLMM surfaces (e.g. grouseticks Poisson) caused premature
        // termination at the initial θ with status `XtolReached`.
        optimizer.set_ftol_rel(ftol_rel).ok();
        optimizer.set_ftol_abs(ftol_abs).ok();
        if let Some(xtol_rel) = xtol_rel {
            optimizer.set_xtol_rel(xtol_rel).ok();
        }
        if let Some(xtol_abs) = &xtol_abs {
            optimizer.set_xtol_abs(xtol_abs).ok();
        }
        optimizer.set_maxeval(maxeval).ok();
        // Mirror the LMM cobyla initial step default; without an explicit
        // initial step BOBYQA falls back to per-axis defaults that may be
        // too small for parameters near the lower bound.
        optimizer.set_initial_step(&initial_step).ok();

        let mut theta = initial_theta;
        let nlopt_result = optimizer.optimize(&mut theta);
        drop(optimizer);

        // Optimizer (and its closure) dropped: reclaim exclusive `&mut self`.
        let me = model.into_inner();
        if let Some(message) = me.pending_progress_error.take() {
            return Err(MixedModelError::Interrupted(message));
        }
        me.finalize_theta_after_optimizer(&mut theta, n_agq)?;
        me.lmm.optsum.return_value = match nlopt_result {
            Ok((status, _fmin)) => experimental_nlopt_status_label(&format!("{status:?}")),
            Err((status, _fmin)) => {
                format!(
                    "FAILED:{}",
                    experimental_nlopt_status_label(&format!("{status:?}"))
                )
            }
        };
        me.lmm.optsum.feval = feval_count.get();
        me.lmm.optsum.fit_log = rc_refcell_into_inner_or_clone(fit_log);
        let mut certificate = OptimizerCertificate::from_opt_summary_with_context(
            &me.lmm.optsum,
            &me.theta,
            &me.lmm.lower_bounds(),
            Some(me.lmm.dims.n),
        );
        annotate_glmm_singular_covariance_status(&mut certificate, &me.theta, me.lmm.is_singular());
        me.lmm.compiler_artifact.optimizer_certificate = Some(certificate);
        me.record_glmm_fit_metadata();
        me.refresh_binomial_separation_diagnostics();
        me.refresh_near_unit_random_effect_correlation_diagnostics();

        Ok(me)
    }

    fn fit_native_cobyla(&mut self, n_agq: usize) -> Result<&mut Self> {
        let lb = self.lmm.lower_bounds();
        let initial_theta = self.lmm.optsum.initial.clone();
        self.lmm.optsum.optimizer = Optimizer::Cobyla;
        self.lmm.optsum.backend = Optimizer::Cobyla.canonical_backend();

        let best_theta: Rc<RefCell<Vec<f64>>> = Rc::new(RefCell::new(initial_theta.clone()));
        let best_fmin: Rc<Cell<f64>> = Rc::new(Cell::new(f64::INFINITY));
        let feval_count: Rc<Cell<i64>> = Rc::new(Cell::new(0i64));
        let fit_log: Rc<RefCell<Vec<FitLogEntry>>> = Rc::new(RefCell::new(Vec::with_capacity(
            self.lmm.optsum.fit_log.capacity(),
        )));

        // Compute every `self`-dependent input before handing the model to the
        // optimizer callback, so `self` is free to move into the RefCell.
        let bounds: Vec<(f64, f64)> = lb.iter().map(|&lo| (lo, f64::INFINITY)).collect();
        let constraint_fns: Vec<Box<dyn cobyla::Func<()>>> = lb
            .iter()
            .enumerate()
            .filter(|(_, &lo)| lo > f64::NEG_INFINITY)
            .map(|(i, &lo)| {
                Box::new(move |x: &[f64], _: &mut ()| -> f64 { x[i] - lo })
                    as Box<dyn cobyla::Func<()>>
            })
            .collect();
        let cons_refs: Vec<&dyn cobyla::Func<()>> =
            constraint_fns.iter().map(|f| f.as_ref()).collect();
        let maxeval = if self.lmm.optsum.max_feval > 0 {
            self.lmm.optsum.max_feval as usize
        } else {
            500
        };
        let stop_tol = cobyla::StopTols {
            ftol_rel: self.lmm.optsum.ftol_rel,
            ftol_abs: self.lmm.optsum.ftol_abs,
            xtol_rel: self.lmm.optsum.xtol_rel,
            xtol_abs: self.lmm.optsum.xtol_abs.clone(),
        };

        // Hand the model to the COBYLA callback through a RefCell instead of a
        // raw `*mut Self`. The callback is the only borrower while the optimizer
        // is alive; `model.into_inner()` recovers `&mut self` afterwards.
        let model = std::cell::RefCell::new(self);
        let objective_fn = |theta: &[f64], _data: &mut ()| -> f64 {
            feval_count.set(feval_count.get() + 1);
            let objective = model
                .borrow_mut()
                .penalized_pirls_deviance_at_theta(theta, n_agq);
            fit_log.borrow_mut().push(FitLogEntry {
                theta: theta.to_vec(),
                objective,
            });
            if objective < best_fmin.get() {
                best_fmin.set(objective);
                *best_theta.borrow_mut() = theta.to_vec();
            }
            objective
        };

        let result = cobyla::minimize(
            objective_fn,
            &initial_theta,
            &bounds,
            &cons_refs,
            (),
            maxeval,
            cobyla::RhoBeg::All(0.75),
            Some(stop_tol),
        );

        let (mut theta, return_value) = match result {
            Ok((status, x_opt, fmin)) if fmin.is_finite() => {
                (x_opt, Self::cobyla_success_status_label(status))
            }
            Ok((status, _x_opt, _fmin)) => (
                best_theta.borrow().clone(),
                Self::cobyla_success_status_label(status),
            ),
            Err((status @ cobyla::FailStatus::RoundoffLimited, _x_opt, _fmin)) => (
                best_theta.borrow().clone(),
                Self::cobyla_fail_status_label(status),
            ),
            Err((status, x_opt, fmin)) if fmin.is_finite() => {
                (x_opt, Self::cobyla_fail_status_label(status))
            }
            Err((status, _x_opt, _fmin)) if best_fmin.get().is_finite() => (
                best_theta.borrow().clone(),
                Self::cobyla_fail_status_label(status),
            ),
            Err((_status, _x_opt, _fmin)) => {
                return Err(MixedModelError::Optimization(
                    "COBYLA optimization failed while fitting GLMM".to_string(),
                ));
            }
        };

        // Optimizer finished and consumed its closure: reclaim `&mut self`.
        let me = model.into_inner();
        if let Some(message) = me.pending_progress_error.take() {
            return Err(MixedModelError::Interrupted(message));
        }
        me.finalize_theta_after_optimizer(&mut theta, n_agq)?;
        me.lmm.optsum.return_value = return_value;
        me.lmm.optsum.feval = feval_count.get();
        me.lmm.optsum.fit_log = rc_refcell_into_inner_or_clone(fit_log);
        let mut certificate = OptimizerCertificate::from_opt_summary_with_context(
            &me.lmm.optsum,
            &me.theta,
            &me.lmm.lower_bounds(),
            Some(me.lmm.dims.n),
        );
        annotate_glmm_singular_covariance_status(&mut certificate, &me.theta, me.lmm.is_singular());
        me.lmm.compiler_artifact.optimizer_certificate = Some(certificate);
        me.record_glmm_fit_metadata();
        me.refresh_binomial_separation_diagnostics();
        me.refresh_near_unit_random_effect_correlation_diagnostics();

        Ok(me)
    }

    fn fit_native_pattern_search(&mut self, n_agq: usize) -> Result<&mut Self> {
        let lower_bounds = self.lmm.lower_bounds();
        let n_theta = self.theta.len();
        let maxeval = if self.lmm.optsum.max_feval > 0 {
            self.lmm.optsum.max_feval
        } else {
            500
        };
        let mut step_tol = self.lmm.optsum.xtol_abs.clone();
        if step_tol.len() != n_theta {
            step_tol = vec![1e-5; n_theta];
        }
        for tol in &mut step_tol {
            *tol = tol.max(1e-5);
        }
        let mut step = self.lmm.optsum.initial_step.clone();
        if step.len() != n_theta {
            step = vec![0.75; n_theta];
        }
        for (value, tol) in step.iter_mut().zip(step_tol.iter()) {
            *value = value.abs().max(*tol);
        }

        let mut theta = self.lmm.optsum.initial.clone();
        project_theta_to_bounds(&mut theta, &lower_bounds);
        let mut best_theta = theta.clone();
        let mut best_fmin = f64::INFINITY;
        let mut feval_count = 0i64;
        let mut fit_log = Vec::with_capacity(self.lmm.optsum.fit_log.capacity());
        let mut preferred_sign = vec![-1.0; n_theta];
        for (idx, lower) in lower_bounds.iter().enumerate() {
            if !lower.is_finite() {
                preferred_sign[idx] = 1.0;
            }
        }

        let mut current_f = record_pattern_search_eval(
            self,
            &theta,
            n_agq,
            &mut feval_count,
            &mut fit_log,
            &mut best_theta,
            &mut best_fmin,
        )?;
        self.lmm.optsum.finitial = current_f;

        while feval_count < maxeval && !steps_are_small(&step, &step_tol) {
            let base_theta = theta.clone();
            let base_f = current_f;
            let mut moved = false;

            for idx in 0..n_theta {
                let mut accepted = false;
                for dir in [preferred_sign[idx], -preferred_sign[idx]] {
                    let mut trial = theta.clone();
                    trial[idx] += dir * step[idx];
                    project_theta_to_bounds(&mut trial, &lower_bounds);
                    if (trial[idx] - theta[idx]).abs() <= step_tol[idx] * 0.5 {
                        continue;
                    }
                    let ftrial = record_pattern_search_eval(
                        self,
                        &trial,
                        n_agq,
                        &mut feval_count,
                        &mut fit_log,
                        &mut best_theta,
                        &mut best_fmin,
                    )?;
                    if ftrial + self.lmm.optsum.ftol_abs < current_f {
                        theta = trial;
                        current_f = ftrial;
                        preferred_sign[idx] = dir;
                        step[idx] = (step[idx] * 1.1).max(step_tol[idx]);
                        moved = true;
                        accepted = true;
                        break;
                    }
                    if feval_count >= maxeval {
                        break;
                    }
                }
                if !accepted {
                    preferred_sign[idx] = -preferred_sign[idx];
                    step[idx] *= 0.5;
                }
                if feval_count >= maxeval {
                    break;
                }
            }

            if moved && feval_count < maxeval {
                let mut pattern = theta.clone();
                for idx in 0..n_theta {
                    pattern[idx] += theta[idx] - base_theta[idx];
                }
                project_theta_to_bounds(&mut pattern, &lower_bounds);
                if pattern != theta {
                    let fpattern = record_pattern_search_eval(
                        self,
                        &pattern,
                        n_agq,
                        &mut feval_count,
                        &mut fit_log,
                        &mut best_theta,
                        &mut best_fmin,
                    )?;
                    if fpattern + self.lmm.optsum.ftol_abs < current_f {
                        theta = pattern;
                        current_f = fpattern;
                    }
                }
            }

            if !moved {
                for value in &mut step {
                    *value *= 0.5;
                }
            }
            if (base_f - current_f).abs() <= self.lmm.optsum.ftol_abs
                && steps_are_small(&step, &step_tol)
            {
                break;
            }
        }

        self.lmm.optsum.optimizer = Optimizer::PatternSearch;
        self.lmm.optsum.backend = Optimizer::PatternSearch.canonical_backend();
        self.finalize_theta_after_optimizer(&mut best_theta, n_agq)?;
        self.lmm.optsum.return_value = if feval_count >= maxeval {
            "MAXEVAL_REACHED".to_string()
        } else {
            "SUCCESS".to_string()
        };
        self.lmm.optsum.feval = feval_count;
        self.lmm.optsum.fit_log = fit_log;
        let mut certificate = OptimizerCertificate::from_opt_summary_with_context(
            &self.lmm.optsum,
            &self.theta,
            &self.lmm.lower_bounds(),
            Some(self.lmm.dims.n),
        );
        annotate_glmm_singular_covariance_status(
            &mut certificate,
            &self.theta,
            self.lmm.is_singular(),
        );
        self.lmm.compiler_artifact.optimizer_certificate = Some(certificate);
        self.record_glmm_fit_metadata();
        self.refresh_binomial_separation_diagnostics();
        self.refresh_near_unit_random_effect_correlation_diagnostics();

        Ok(self)
    }

    pub(super) fn finalize_theta_after_optimizer(
        &mut self,
        theta: &mut [f64],
        n_agq: usize,
    ) -> Result<()> {
        LinearMixedModel::rectify_theta_columns(theta, &self.lmm.parmap, self.lmm.reterms.len());

        // Final PIRLS at optimal θ, after matching MixedModels.jl's
        // post-optimizer sign convention for Cholesky columns.
        let pirls_converged = match self.update_pirls_at_theta(theta, true) {
            Ok(converged) => converged,
            Err(error) => {
                self.record_pirls_failure_diagnostic(theta, &error.to_string());
                return Err(error);
            }
        };
        if !pirls_converged {
            // Not a hard failure (Julia also returns a model here), but the
            // unverified modes must be observable rather than silently
            // accepted as a good fit (audit 03·H1).
            self.record_pirls_nonconvergence_diagnostic(theta);
        }
        self.beta = self.lmm.beta();
        self.refresh_dispersion();

        self.lmm.optsum.n_agq = n_agq;
        self.lmm.optsum.fmin = self.deviance(n_agq);
        self.lmm.optsum.final_params = theta.to_vec();
        Ok(())
    }

    fn cobyla_success_status_label(status: cobyla::SuccessStatus) -> String {
        match status {
            cobyla::SuccessStatus::Success => "SUCCESS".to_string(),
            cobyla::SuccessStatus::StopValReached => "STOPVAL_REACHED".to_string(),
            cobyla::SuccessStatus::FtolReached => "FTOL_REACHED".to_string(),
            cobyla::SuccessStatus::XtolReached => "XTOL_REACHED".to_string(),
            cobyla::SuccessStatus::MaxEvalReached => "MAXEVAL_REACHED".to_string(),
            cobyla::SuccessStatus::MaxTimeReached => "MAXTIME_REACHED".to_string(),
        }
    }

    fn cobyla_fail_status_label(status: cobyla::FailStatus) -> String {
        match status {
            cobyla::FailStatus::Failure => "FAILURE".to_string(),
            cobyla::FailStatus::InvalidArgs => "INVALID_ARGS".to_string(),
            cobyla::FailStatus::OutOfMemory => "OUT_OF_MEMORY".to_string(),
            cobyla::FailStatus::RoundoffLimited => "ROUNDOFF_LIMITED".to_string(),
            cobyla::FailStatus::ForcedStop => "FORCED_STOP".to_string(),
            cobyla::FailStatus::UnexpectedError => "UNEXPECTED_ERROR".to_string(),
        }
    }
}

pub(crate) fn project_theta_to_bounds(theta: &mut [f64], lower_bounds: &[f64]) {
    for (value, lower) in theta.iter_mut().zip(lower_bounds.iter()) {
        if lower.is_finite() && *value < *lower {
            *value = *lower;
        }
    }
}

pub(crate) fn steps_are_small(step: &[f64], step_tol: &[f64]) -> bool {
    step.iter()
        .zip(step_tol.iter())
        .all(|(step, tol)| *step <= *tol)
}

pub(crate) fn record_pattern_search_eval(
    model: &mut GeneralizedLinearMixedModel,
    theta: &[f64],
    n_agq: usize,
    feval_count: &mut i64,
    fit_log: &mut Vec<FitLogEntry>,
    best_theta: &mut Vec<f64>,
    best_fmin: &mut f64,
) -> Result<f64> {
    *feval_count += 1;
    let objective = model.penalized_pirls_deviance_at_theta(theta, n_agq);
    fit_log.push(FitLogEntry {
        theta: theta.to_vec(),
        objective,
    });
    if objective < *best_fmin {
        *best_fmin = objective;
        *best_theta = theta.to_vec();
    }
    if let Some(message) = model.pending_progress_error.take() {
        return Err(MixedModelError::Interrupted(message));
    }
    Ok(objective)
}

pub(crate) fn lower_triangle_pair(offset: usize) -> (usize, usize) {
    let mut row = 1usize;
    let mut remaining = offset;
    while remaining >= row {
        remaining -= row;
        row += 1;
    }
    (row, remaining)
}

pub(crate) fn default_fast_glmm_optimizer() -> Optimizer {
    #[cfg(feature = "nlopt")]
    {
        Optimizer::NloptBobyqa
    }
    #[cfg(not(feature = "nlopt"))]
    {
        Optimizer::Cobyla
    }
}

#[cfg(feature = "nlopt")]
pub(crate) fn experimental_nlopt_status_label(name: &str) -> String {
    match name {
        "Success" => "SUCCESS".to_string(),
        "StopValReached" => "STOPVAL_REACHED".to_string(),
        "FtolReached" => "FTOL_REACHED".to_string(),
        "XtolReached" => "XTOL_REACHED".to_string(),
        "MaxEvalReached" => "MAXEVAL_REACHED".to_string(),
        "MaxTimeReached" => "MAXTIME_REACHED".to_string(),
        "RoundoffLimited" => "ROUNDOFF_LIMITED".to_string(),
        "InvalidArgs" => "INVALID_ARGS".to_string(),
        "OutOfMemory" => "OUT_OF_MEMORY".to_string(),
        "ForcedStop" => "FORCED_STOP".to_string(),
        "Failure" => "FAILURE".to_string(),
        other => other.to_ascii_uppercase(),
    }
}
