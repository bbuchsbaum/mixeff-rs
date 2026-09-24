//! Joint Laplace/AGQ GLMM optimizer, its certification gradients/Hessians,
//! and joint-mode fixed-effect inference artifacts.
//!
//! Moved verbatim from `generalized/mod.rs` during the module split
//! (bd-01KWHYQSTWK60P6HA4S4B2K99P). No logic changes.

use super::*;
use crate::optimizer::trust_bq::TrustBqResult;

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
        let fitted = match optimizer {
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
        };
        // Evaluations after the optimizer (polish, certification probes) hold
        // a host interrupt on the model rather than failing; return it.
        let me = fitted?;
        me.take_pending_interrupt()?;
        Ok(me)
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
        self.pending_progress_error = None;
        self.lmm.optsum.optimizer = Optimizer::NloptBobyqa;
        self.lmm.optsum.backend = Optimizer::NloptBobyqa.canonical_backend();
        self.lmm.optsum.finitial = profiled_start_objective;
        // Stopping tolerances follow MixedModels.jl's `OptSummary` defaults
        // for the fast = false joint fit (ftol_rel 1e-12, ftol_abs 1e-8; no
        // xtol_abs, which Julia only applies when its length matches θ).
        let (ftol_abs, ftol_rel) = self.joint_ftol();
        let xtol_rel = self
            .lmm
            .optsum
            .caller_set_field("xtol_rel")
            .then_some(self.lmm.optsum.xtol_rel);
        // NLopt's BOBYQA rescales the parameters so the initial steps are
        // equal, so the β steps set the conditioning of the β block. One
        // profiled (fast-PIRLS) standard error per coefficient makes that
        // block roughly a correlation matrix; a flat 0.1 step leaves it as
        // ill-conditioned as the covariates' scales (contraception's `age`
        // coefficient has ~1/19 the SE of the intercept).
        let mut initial_step = self.joint_beta_initial_steps();
        if self.lmm.optsum.caller_set_field("initial_step") {
            initial_step.extend(self.lmm.optsum.initial_step.clone());
        } else {
            initial_step.extend(vec![JOINT_THETA_STEP; n_theta]);
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
        let run_bobyqa = |params: &mut Vec<f64>, steps: &[f64], budget: u32| {
            let mut optimizer = Nlopt::new(
                NloptAlgorithm::Bobyqa,
                n_params,
                &obj_fn,
                NloptTarget::Minimize,
                (),
            );
            optimizer.set_lower_bounds(&lower_bounds).ok();
            optimizer.set_ftol_rel(ftol_rel).ok();
            optimizer.set_ftol_abs(ftol_abs).ok();
            if let Some(xtol_rel) = xtol_rel {
                optimizer.set_xtol_rel(xtol_rel).ok();
            }
            optimizer.set_maxeval(budget.max(1)).ok();
            optimizer.set_initial_step(steps).ok();
            optimizer.optimize(params)
        };

        // A host interrupt raised inside an evaluation's PIRLS is held on the
        // model (see `joint_glmm_deviance_at_params`); NLopt cannot be
        // stopped from the objective, so it is returned after each run.
        let take_interrupt = || model.borrow_mut().pending_progress_error.take();
        let mut params = initial;
        let mut nlopt_result = run_bobyqa(&mut params, &initial_step, maxeval);
        if let Some(message) = take_interrupt() {
            return Err(MixedModelError::Interrupted(message));
        }
        // NLopt's BOBYQA declares FTOL_REACHED on the first accepted step
        // whose improvement is below the tolerance, whatever the trust-region
        // radius; it is not a contraction test (the analogue of TrustBQ's
        // `ftol_requires_local_radius`). A step that happens to gain little
        // on a still-large radius therefore stops the fit well short of the
        // optimum, and which step does so is decided by last-bit rounding
        // (contraception stopped 9.2e-5 above the optimum on Linux, 1e-6 on
        // macOS). Confirm every FTOL stop by restarting from the incumbent
        // with a fresh, 10x smaller interpolation set: the stop stands only
        // once a restart gains no more than the tolerance.
        //
        // A restart needs 2n + 1 evaluations just to build its interpolation
        // model, and a confirming restart then takes a few trust-region steps
        // at the reduced radius; one launched with less than twice the model
        // size cannot finish either, so it is not started and the stop is
        // recorded as unconfirmed.
        let min_restart_budget = 2 * (2 * n_params as u32 + 1);
        let mut confirmation = JointFtolConfirmation::NotNeeded;
        let mut restarts = 0usize;
        while matches!(nlopt_result, Ok((nlopt::SuccessState::FtolReached, _))) {
            let incumbent = match &nlopt_result {
                Ok((_, fmin)) | Err((_, fmin)) => *fmin,
            };
            if restarts == JOINT_MAX_CONFIRMATION_RESTARTS {
                confirmation = JointFtolConfirmation::Unconfirmed {
                    reason: "restart_cap_reached",
                    restarts,
                    last_gain: None,
                };
                break;
            }
            let used = u32::try_from(feval_count.get()).unwrap_or(u32::MAX);
            let remaining = maxeval.saturating_sub(used);
            if remaining < min_restart_budget {
                confirmation = JointFtolConfirmation::Unconfirmed {
                    reason: "insufficient_evaluation_budget",
                    restarts,
                    last_gain: None,
                };
                break;
            }
            let steps = joint_restart_steps(&initial_step, &params, &lower_bounds);
            let mut candidate = params.clone();
            let result = run_bobyqa(&mut candidate, &steps, remaining);
            if let Some(message) = take_interrupt() {
                return Err(MixedModelError::Interrupted(message));
            }
            restarts += 1;
            let (restart_ok, best) = match &result {
                Ok((_, fmin)) => (true, *fmin),
                Err((_, fmin)) => (false, *fmin),
            };
            // The restart starts at the incumbent (its steps keep NLopt from
            // shifting it off a bound), so a best value above the incumbent
            // means the incumbent was not reproduced.
            if !best.is_finite()
                || best > incumbent
                || !candidate.iter().all(|value| value.is_finite())
            {
                confirmation = JointFtolConfirmation::Unconfirmed {
                    reason: "restart_did_not_reproduce_incumbent",
                    restarts,
                    last_gain: None,
                };
                break;
            }
            let gain = incumbent - best;
            params = candidate;
            if gain <= ftol_abs.max(ftol_rel * incumbent.abs()) {
                // Confirmed: the stop stands with its FTOL label whatever
                // code the confirming restart itself returned (a budget or
                // roundoff stop after gaining nothing does not unconfirm it).
                nlopt_result = Ok((nlopt::SuccessState::FtolReached, best));
                confirmation = JointFtolConfirmation::Confirmed;
                break;
            }
            if !restart_ok {
                // Keep the better point, but a restart that failed after a
                // material gain confirms nothing; the label stays the first
                // pass's FTOL stop and the gap is recorded.
                nlopt_result = Ok((nlopt::SuccessState::FtolReached, best));
                confirmation = JointFtolConfirmation::Unconfirmed {
                    reason: "restart_failed_after_material_gain",
                    restarts,
                    last_gain: Some(gain),
                };
                break;
            }
            // A material gain: the restart's own stop is the new incumbent
            // status (FTOL loops to be confirmed again; XTOL/SUCCESS are
            // BOBYQA's native radius convergence; MAXEVAL is an honest
            // budget stop).
            nlopt_result = result;
            if !matches!(nlopt_result, Ok((nlopt::SuccessState::FtolReached, _))) {
                confirmation = JointFtolConfirmation::NotNeeded;
            }
        }

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
        let mut certification_gradient = me.joint_laplace_certification_gradient(
            &me.lmm.optsum.final_params.clone(),
            n_beta,
            n_agq,
            &lower_bounds,
            2.0e-2,
        );
        certificate.apply_derivative_evidence(
            OptimizerDerivativeEvidence {
                method: EvidenceMethod::FiniteDifference,
                hessian_method: EvidenceMethod::FiniteDifference,
                gradient: certification_gradient.gradient.clone(),
                hessian: None,
            },
            2.0e-2,
            1.0e-6,
        );
        let beta_hessian = me.joint_working_beta_hessian();
        me.confirm_decrement_curvature(
            &mut certification_gradient,
            &me.lmm.optsum.final_params.clone(),
            n_beta,
            n_agq,
            &lower_bounds,
            beta_hessian.as_ref(),
        );
        annotate_glmm_covariance_status(
            &mut certificate,
            &me.lmm.optsum.final_params,
            n_beta,
            &lower_bounds,
            &certification_gradient,
            2.0e-2,
            beta_hessian.as_ref(),
        );
        record_joint_ftol_confirmation(&mut certificate, &confirmation, "NLopt BOBYQA");
        me.take_pending_interrupt()?;
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
        self.pending_progress_error = None;
        self.lmm.optsum.optimizer = Optimizer::TrustBq;
        self.lmm.optsum.backend = Optimizer::TrustBq.canonical_backend();
        self.lmm.optsum.finitial = profiled_start_objective;
        self.lmm.optsum.max_feval = maxeval as i64;
        // MixedModels.jl's joint-fit tolerances. The former floors (ftol_abs
        // 1e-7, ftol_rel 1e-10: 3.4e-7 per accepted step at contraception's
        // deviance) let the local-radius FTOL stop land several 1e-7 high.
        let (ftol_abs, ftol_rel) = self.joint_ftol();
        // Up to twelve parameters the model carries every cross term (at most
        // 66 extra samples, a 90-sample model against a default budget of at
        // least 1460), the stall stop uses the FTOL band with stable
        // parameters, and the stop is polished. These once applied only to
        // five to eight parameters; with a diagonal model and a 1e-6
        // statistical stall band the three-parameter rare-event Bernoulli
        // AGQ5 fit stopped 5.4e-7, and the nine-parameter contraception
        // random-slope fit 1.9e-6, above the optimum.
        let compact_joint_space = n_params <= 12;

        // TrustBQ's trust region is a Euclidean ball and its interpolation
        // stencil samples every axis at the radius, so it works in
        // transformed coordinates `x = x0 + T z` (see
        // `joint_trust_bq_transform`): a unit radius is one profiled standard
        // error along every fixed-effect direction, the scaling NLopt's
        // BOBYQA gets from its initial steps, plus decorrelation. In raw
        // coordinates a radius that is local for the intercept
        // (contraception SE 0.149) is 2.6 standard errors of `age` (SE
        // 0.0079); the axis stencil's truncation error there hid an `age`
        // gradient of 2.6 and the local-radius FTOL stop fired 1.05e-4 above
        // the optimum. The reported final trust radius is in `z` units.
        let transform = self.joint_trust_bq_transform(n_theta);
        let to_x = |z: &[f64]| {
            let offset = &transform * DVector::from_column_slice(z);
            initial
                .iter()
                .zip(offset.iter())
                .map(|(start, delta)| start + delta)
                .collect::<Vec<_>>()
        };
        let initial_z = vec![0.0; n_params];
        // The transform is block diagonal with a diagonal θ block, so each
        // bounded coordinate keeps a per-coordinate bound.
        let lower_bounds_z = (0..n_params)
            .map(|index| (lower_bounds[index] - initial[index]) / transform[(index, index)])
            .collect::<Vec<_>>();

        let invalid_objective = profiled_start_objective.abs().max(1.0)
            + 1.0e6 * (1.0 + profiled_start_objective.abs());
        let best_params = RefCell::new(initial.clone());
        let best_fmin = Cell::new(profiled_start_objective);
        let fit_log: Rc<RefCell<Vec<FitLogEntry>>> = Rc::new(RefCell::new(Vec::new()));

        let progress_callback = self.lmm.progress_callback.clone();
        let model = std::cell::RefCell::new(self);
        // Objective evaluations across all passes, including a restart that
        // fails part-way (its own count is lost with its error).
        let evaluations = Cell::new(0usize);
        let mut objective_fn = |z: &[f64]| -> Result<f64> {
            evaluations.set(evaluations.get() + 1);
            let params = to_x(z);
            let params = params.as_slice();
            let raw_objective = model
                .borrow_mut()
                .joint_glmm_deviance_at_params(params, n_beta, n_agq);
            if let Some(message) = model.borrow_mut().pending_progress_error.take() {
                return Err(MixedModelError::Interrupted(message));
            }
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
        // Evaluations spent by earlier TrustBQ passes, so progress reports
        // count the whole fit across confirmation restarts.
        let spent_before_pass = Cell::new(0usize);
        let mut progress_fn = |progress: &TrustBqProgress<'_>| -> Result<bool> {
            if let Some(callback) = &progress_callback {
                callback.report_if_due(
                    FitProgressPhase::JointGlmmOptimizer,
                    spent_before_pass.get() + progress.fevals,
                    Some(maxeval.max(1) as usize),
                    &mut last_progress,
                )?;
            }
            Ok(false)
        };
        let pass_options =
            |initial_radius: f64, final_radius: f64, max_evaluations: usize| TrustBqOptions {
                initial_radius,
                final_radius,
                max_evaluations: max_evaluations.max(1),
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
            };

        let result = minimize_trust_bq_with_progress(
            &initial_z,
            &lower_bounds_z,
            &upper_bounds,
            pass_options(
                JOINT_TRUST_BQ_INITIAL_RADIUS,
                JOINT_TRUST_BQ_FINAL_RADIUS,
                maxeval as usize,
            ),
            &mut objective_fn,
            &mut progress_fn,
        )?;
        // An objective-tolerance or stagnation stop rests on one small
        // accepted step (or a few stalled iterations) of one interpolation
        // model, which a poorly resolved direction can fake even at a
        // contracted radius. As for NLopt BOBYQA, confirm it by restarting
        // from the incumbent with a fresh model at a tenth of the initial
        // radius (see `confirm_joint_trust_bq_stop`).
        let model_size = 2 * n_params
            + if compact_joint_space {
                n_params * n_params.saturating_sub(1) / 2
            } else {
                0
            };
        let plan =
            JointRestartPlan::for_model_size(model_size, maxeval as usize, ftol_abs, ftol_rel);
        let restart_radius = JOINT_RESTART_STEP_FACTOR * JOINT_TRUST_BQ_INITIAL_RADIUS;
        let (result, fevals, confirmation) = confirm_joint_trust_bq_stop(
            result,
            evaluations.get(),
            &plan,
            |start, budget, spent| {
                spent_before_pass.set(spent);
                let restart = minimize_trust_bq_with_progress(
                    start,
                    &lower_bounds_z,
                    &upper_bounds,
                    pass_options(restart_radius, JOINT_TRUST_BQ_FINAL_RADIUS, budget),
                    &mut objective_fn,
                    &mut progress_fn,
                );
                (restart, evaluations.get())
            },
        )?;

        let logged_best_params = best_params.into_inner();
        let logged_best_fmin = best_fmin.get();
        let (mut params, candidate_objective) =
            if logged_best_fmin.is_finite() && logged_best_fmin <= result.fmin {
                (logged_best_params, logged_best_fmin)
            } else {
                (to_x(&result.x), result.fmin)
            };
        let me = model.into_inner();
        let status_prefix = joint_glmm_status_prefix(n_agq);
        me.lmm.optsum.return_value = format!(
            "{status_prefix}:{}",
            trust_bq_status_label(result.stop_reason)
        );
        me.lmm.optsum.n_agq = n_agq;
        me.lmm.optsum.feval = fevals as i64;
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
        // Read at the certified point: the probes' last evaluation restores
        // it, and a rejected polish below would leave the state elsewhere.
        let mut beta_hessian = me.joint_working_beta_hessian();
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
                    beta_hessian = me.joint_working_beta_hessian();
                }
            }
        }
        certificate.apply_derivative_evidence(
            OptimizerDerivativeEvidence {
                method: EvidenceMethod::FiniteDifference,
                hessian_method: EvidenceMethod::FiniteDifference,
                gradient: certification_gradient.gradient.clone(),
                hessian: None,
            },
            2.0e-2,
            1.0e-6,
        );
        me.confirm_decrement_curvature(
            &mut certification_gradient,
            &me.lmm.optsum.final_params.clone(),
            n_beta,
            n_agq,
            &lower_bounds,
            beta_hessian.as_ref(),
        );
        annotate_glmm_covariance_status(
            &mut certificate,
            &me.lmm.optsum.final_params,
            n_beta,
            &lower_bounds,
            &certification_gradient,
            2.0e-2,
            beta_hessian.as_ref(),
        );
        record_joint_ftol_confirmation(&mut certificate, &confirmation, "TrustBQ");
        me.take_pending_interrupt()?;
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

    /// Initial steps (NLopt BOBYQA) and coordinate scales (TrustBQ) for the
    /// joint β block: one profiled (fast-PIRLS) standard error per
    /// coefficient where that covariance is available and finite, else a flat
    /// default step.
    ///
    /// Each step is capped at `max(1, |β_j|)`. An SE above that means the
    /// profiled Hessian is nearly singular in that coordinate (typically
    /// near-separation in a binomial model, SE 1e2–1e4); a step that size
    /// would put BOBYQA's first interpolation points deep in a saturated
    /// logistic tail where the deviance is flat and carries no curvature
    /// information. `|β_j|` is NLopt's own default step (what MixedModels.jl
    /// uses), and the floor of 1 keeps near-zero coefficients movable.
    fn joint_beta_initial_steps(&self) -> Vec<f64> {
        let n_beta = self.beta.len();
        let mut steps = vec![JOINT_DEFAULT_BETA_STEP; n_beta];
        // The profiled covariance is in the unpivoted coefficient order;
        // active coefficient `i` is full coefficient `piv[i]`.
        let piv = &self.lmm.feterm.piv;
        if let Some(covariance) = self.profiled_glmm_fixed_effect_covariance() {
            for (index, (step, &full)) in steps.iter_mut().zip(piv).enumerate() {
                if full < covariance.nrows() && full < covariance.ncols() {
                    let se = covariance[(full, full)].sqrt();
                    if se.is_finite() && se > 0.0 {
                        *step = se.min(self.beta[index].abs().max(1.0));
                    }
                }
            }
        }
        steps
    }

    /// Joint-fit `(ftol_abs, ftol_rel)`: the caller's values where set, else
    /// MixedModels.jl's `OptSummary` defaults.
    fn joint_ftol(&self) -> (f64, f64) {
        let optsum = &self.lmm.optsum;
        let ftol_abs = if optsum.caller_set_field("ftol_abs") {
            optsum.ftol_abs
        } else {
            JOINT_FTOL_ABS
        };
        let ftol_rel = if optsum.caller_set_field("ftol_rel") {
            optsum.ftol_rel
        } else {
            JOINT_FTOL_REL
        };
        (ftol_abs, ftol_rel)
    }

    /// Coordinate transform for the TrustBQ joint driver, `x = x0 + T z`.
    /// The β block is `D L`, for `D` the joint β steps and `L` the Cholesky
    /// factor of the profiled fixed-effect correlation matrix, so that
    /// `T Tᵀ` is the profiled covariance wherever no step is capped: a unit
    /// ball in `z` is the profiled confidence ellipsoid and the working β
    /// Hessian is near `2 I`. Without a usable covariance, or when the
    /// correlation is nearly singular (see
    /// [`well_conditioned_correlation_factor`]), the block is `D`. Each
    /// covariance parameter is scaled by the caller's `initial_step` where
    /// one of the right length is set (as for NLopt), else by
    /// [`JOINT_THETA_STEP`].
    pub(super) fn joint_trust_bq_transform(&self, n_theta: usize) -> DMatrix<f64> {
        let steps = self.joint_beta_initial_steps();
        let n_beta = steps.len();
        let n = n_beta + n_theta;
        let mut transform = DMatrix::zeros(n, n);
        let beta_block = self
            .joint_beta_correlation_factor()
            .and_then(well_conditioned_correlation_factor)
            .unwrap_or_else(|| DMatrix::identity(n_beta, n_beta));
        for i in 0..n_beta {
            for j in 0..=i {
                transform[(i, j)] = steps[i] * beta_block[(i, j)];
            }
        }
        let caller_theta_steps = self
            .lmm
            .optsum
            .caller_set_field("initial_step")
            .then_some(&self.lmm.optsum.initial_step)
            .filter(|steps| {
                steps.len() == n_theta && steps.iter().all(|step| step.is_finite() && *step > 0.0)
            });
        for (offset, index) in (n_beta..n).enumerate() {
            transform[(index, index)] =
                caller_theta_steps.map_or(JOINT_THETA_STEP, |steps| steps[offset]);
        }
        transform
    }

    /// Lower Cholesky factor of the profiled fixed-effect correlation matrix
    /// in the optimizer's β order, or `None` when it is unavailable.
    pub(super) fn joint_beta_correlation_factor(&self) -> Option<DMatrix<f64>> {
        let covariance = self.profiled_glmm_fixed_effect_covariance()?;
        let piv = &self.lmm.feterm.piv;
        let p = self.beta.len();
        if piv.len() < p {
            return None;
        }
        let mut correlation = DMatrix::zeros(p, p);
        for i in 0..p {
            for j in 0..p {
                let cij = *covariance.get((piv[i], piv[j]))?;
                let cii = *covariance.get((piv[i], piv[i]))?;
                let cjj = *covariance.get((piv[j], piv[j]))?;
                if !(cii > 0.0 && cjj > 0.0) {
                    return None;
                }
                correlation[(i, j)] = cij / (cii * cjj).sqrt();
            }
        }
        let factor = correlation.cholesky()?.l();
        matrix_is_finite_local(&factor).then_some(factor)
    }

    /// Returns a host interrupt held by `joint_glmm_deviance_at_params`.
    fn take_pending_interrupt(&mut self) -> Result<()> {
        match self.pending_progress_error.take() {
            Some(message) => Err(MixedModelError::Interrupted(message)),
            None => Ok(()),
        }
    }

    pub(super) fn joint_glmm_deviance_at_params(
        &mut self,
        params: &[f64],
        n_beta: usize,
        n_agq: usize,
    ) -> f64 {
        if self.pending_progress_error.is_some()
            || params.len() != n_beta + self.theta.len()
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
            // A host interrupt raised by the inner PIRLS progress report is
            // held for the joint driver to return; later evaluations
            // short-circuit.
            Err(MixedModelError::Interrupted(message)) => {
                self.pending_progress_error = Some(message);
                f64::INFINITY
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

    /// Default-step central-difference probes of every coordinate, plus the
    /// objective at `params` itself (evaluated last, which also restores the
    /// fitted PIRLS state). The base value is what turns each central pair
    /// into a diagonal curvature reading at no extra cost.
    fn joint_laplace_finite_difference_probes(
        &mut self,
        params: &[f64],
        n_beta: usize,
        n_agq: usize,
        lower_bounds: &[f64],
    ) -> (Vec<JointFdProbe>, f64) {
        let probes = (0..params.len())
            .map(|index| {
                let h = JOINT_LAPLACE_FD_RELATIVE_STEP * params[index].abs().max(1.0);
                self.joint_laplace_fd_probe(params, index, h, n_beta, n_agq, lower_bounds)
            })
            .collect();
        let base = self.joint_glmm_deviance_at_params(params, n_beta, n_agq);
        (probes, base)
    }

    /// One finite-difference probe of coordinate `index` at step `h`:
    /// central when `value - h` stays above the lower bound, forward
    /// otherwise. A forward probe needs the objective at `params` and
    /// evaluates it; a central probe does not.
    fn joint_laplace_fd_probe(
        &mut self,
        params: &[f64],
        index: usize,
        h: f64,
        n_beta: usize,
        n_agq: usize,
        lower_bounds: &[f64],
    ) -> JointFdProbe {
        let value = params[index];
        let lower = lower_bounds
            .get(index)
            .copied()
            .unwrap_or(f64::NEG_INFINITY);
        let mut plus = params.to_vec();
        plus[index] = value + h;
        let f_plus = self.joint_glmm_deviance_at_params(&plus, n_beta, n_agq);
        if value - h > lower {
            let mut minus = params.to_vec();
            minus[index] = value - h;
            let f_minus = self.joint_glmm_deviance_at_params(&minus, n_beta, n_agq);
            JointFdProbe::central(h, f_plus, f_minus)
        } else {
            let base = self.joint_glmm_deviance_at_params(params, n_beta, n_agq);
            JointFdProbe::forward(h, f_plus, base)
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
    ///
    /// The central probes also carry diagonal curvature readings, from which
    /// the result's [`JointLaplaceCertificationGradient::newton_decrement`]
    /// estimates the objective gap without further evaluations.
    pub(super) fn joint_laplace_certification_gradient(
        &mut self,
        params: &[f64],
        n_beta: usize,
        n_agq: usize,
        lower_bounds: &[f64],
        gradient_tolerance: f64,
    ) -> JointLaplaceCertificationGradient {
        let (probes, base_objective) =
            self.joint_laplace_finite_difference_probes(params, n_beta, n_agq, lower_bounds);
        let probe_gradient = probes
            .iter()
            .map(|probe| probe.gradient)
            .collect::<Vec<_>>();
        let mut gradient = probe_gradient.clone();
        let mut curvature_probes = probes.iter().map(|probe| vec![*probe]).collect::<Vec<_>>();
        let mut escalated_indices = Vec::new();
        let mut unassessable_indices = Vec::new();
        for (index, &value) in params.iter().enumerate() {
            let raw = probe_gradient[index];
            if raw.is_finite() && raw.abs() <= gradient_tolerance {
                continue;
            }
            let scale = value.abs().max(1.0);
            let estimates = JOINT_LAPLACE_CERT_FD_ESCALATED_RELATIVE_STEPS.map(|step| {
                self.joint_laplace_fd_probe(
                    params,
                    index,
                    step * scale,
                    n_beta,
                    n_agq,
                    lower_bounds,
                )
            });
            curvature_probes[index].extend(estimates);
            let consistent = estimates
                .iter()
                .all(|estimate| estimate.gradient.is_finite())
                && (estimates[0].gradient - estimates[1].gradient).abs() <= gradient_tolerance;
            if consistent {
                gradient[index] = estimates[1].gradient;
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
            base_objective,
            curvature_probes,
        }
    }

    /// Before a decrement above tolerance can demote a stop, confirm the
    /// curvature of every interior covariance parameter read from a single
    /// default-step probe: its second difference `(f+ + f- - 2 f0) / h^2`
    /// divides PIRLS stopping noise by `h^2 = 1e-10`, so a spuriously small
    /// positive reading would inflate `g^2 / H`. The escalated probes are
    /// added to `curvature_probes` only (the reported gradient and its
    /// escalation record are untouched), so [`decrement_reading`] then needs
    /// agreeing escalated curvatures and a gap-unit-consistent gradient.
    /// Fixed effects need no confirmation: the β-block estimate takes their
    /// curvature from the working Hessian after checking it against the
    /// probes. Costs four evaluations per such parameter, only on stops the
    /// eager estimate would reject; fitted state is restored.
    fn confirm_decrement_curvature(
        &mut self,
        certification: &mut JointLaplaceCertificationGradient,
        params: &[f64],
        n_beta: usize,
        n_agq: usize,
        lower_bounds: &[f64],
        beta_hessian: Option<&DMatrix<f64>>,
    ) {
        let preliminary = joint_newton_decrement(
            certification,
            params,
            lower_bounds,
            n_beta,
            beta_hessian,
            JOINT_STATIONARITY_GAP_TOLERANCE,
        );
        if preliminary.eager.verdict != crate::compiler::NewtonDecrementVerdict::ExceedsTolerance {
            return;
        }
        let mut probed = false;
        for &index in &preliminary.parameter_indices {
            if index < n_beta || certification.curvature_probes[index].len() != 1 {
                continue;
            }
            let scale = params[index].abs().max(1.0);
            let probes = JOINT_LAPLACE_CERT_FD_ESCALATED_RELATIVE_STEPS.map(|step| {
                self.joint_laplace_fd_probe(
                    params,
                    index,
                    step * scale,
                    n_beta,
                    n_agq,
                    lower_bounds,
                )
            });
            certification.curvature_probes[index].extend(probes);
            probed = true;
        }
        if probed {
            let _ = self.joint_glmm_deviance_at_params(params, n_beta, n_agq);
        }
    }

    /// Record the full Newton decrement `gᵀH⁻¹g` on the optimizer
    /// certificate's stationarity evidence once the joint Hessian exists
    /// (or why it could not be formed). Evidence only: the fit status stays
    /// the one the eager decrement decided at fit time, so it cannot depend
    /// on whether or when inference was inspected.
    fn record_full_newton_decrement(
        &mut self,
        hessian: std::result::Result<&DMatrix<f64>, &String>,
        active_indices: &[usize],
    ) {
        let Some(evidence) = self
            .lmm
            .compiler_artifact
            .optimizer_certificate
            .as_mut()
            .and_then(|certificate| certificate.stationarity_decrement.as_mut())
        else {
            return;
        };
        evidence.full = Some(match hessian {
            Ok(hessian) => full_newton_decrement_estimate(evidence, hessian, active_indices),
            Err(reason) => crate::compiler::NewtonDecrementEstimate {
                variant: crate::compiler::NewtonDecrementVariant::Full,
                verdict: crate::compiler::NewtonDecrementVerdict::NotAssessed,
                objective_gap: None,
                reason: Some(format!("joint Hessian unavailable: {reason}")),
            },
        });
    }

    /// Fixed-effect block of the working penalized-least-squares Hessian of
    /// the deviance at the current PIRLS state, in the optimizer's β order:
    /// `2 Cov⁻¹` for the working fixed-effect covariance (the deviance is
    /// `-2 log L`). It is the β-β curvature of the Laplace deviance up to the
    /// β-dependence of the log-determinant term, and costs no objective
    /// evaluations. PIRLS weights are the expected (Fisher) weights, so for a
    /// non-canonical link this is the expected rather than the observed
    /// curvature, and for `n_agq > 1` it is the Laplace rather than the AGQ
    /// curvature; the decrement uses it only when its diagonal agrees with
    /// the probed curvature within a factor of two. `None` when the
    /// covariance is unavailable or singular.
    pub(super) fn joint_working_beta_hessian(&self) -> Option<DMatrix<f64>> {
        let covariance = self.profiled_glmm_fixed_effect_covariance()?;
        let piv = &self.lmm.feterm.piv;
        let p = self.beta.len();
        if piv.len() < p {
            return None;
        }
        let mut active = DMatrix::zeros(p, p);
        for i in 0..p {
            for j in 0..p {
                active[(i, j)] = *covariance.get((piv[i], piv[j]))?;
            }
        }
        let inverse = active.cholesky()?.inverse();
        matrix_is_finite_local(&inverse).then(|| 2.0 * inverse)
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
        );
        self.record_full_newton_decrement(hessian.as_ref(), &active_indices);
        let hessian = hessian?;
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

    pub(super) fn finite_difference_joint_laplace_hessian(
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
            .inference_artifact()
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

/// MixedModels.jl `OptSummary` default `ftol_rel`, used by its joint
/// (`fast = false`) GLMM fit.
const JOINT_FTOL_REL: f64 = 1.0e-12;
/// MixedModels.jl `OptSummary` default `ftol_abs`.
const JOINT_FTOL_ABS: f64 = 1.0e-8;
/// Initial-step (NLopt BOBYQA) or initial-radius (TrustBQ) factor for the
/// restarts that confirm an FTOL stop, relative to the first pass's.
const JOINT_RESTART_STEP_FACTOR: f64 = 0.1;
/// Cap on FTOL-confirmation restarts; each is also bounded by the remaining
/// evaluation budget.
const JOINT_MAX_CONFIRMATION_RESTARTS: usize = 20;
/// Outcome of confirming a joint optimizer's FTOL stop by restarting.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum JointFtolConfirmation {
    /// The final stop was not an FTOL stop (native radius convergence, a
    /// budget stop, or a failure), so there was nothing to confirm.
    NotNeeded,
    /// A restart from the incumbent gained no more than the tolerance.
    Confirmed,
    /// Confirmation could not be completed; the fit keeps its best point and
    /// FTOL label, and this is recorded on the certificate.
    Unconfirmed {
        reason: &'static str,
        restarts: usize,
        last_gain: Option<f64>,
    },
}

/// Budget and tolerance rules for confirming a TrustBQ joint FTOL stop.
#[derive(Debug, Clone, Copy)]
pub(super) struct JointRestartPlan {
    /// The fit's total evaluation budget.
    pub(super) maxeval: usize,
    /// A restart is launched only with at least this many evaluations left.
    pub(super) min_restart_budget: usize,
    /// Evaluation allowance of one restart.
    pub(super) restart_budget: usize,
    pub(super) max_restarts: usize,
    pub(super) ftol_abs: f64,
    pub(super) ftol_rel: f64,
}

impl JointRestartPlan {
    /// A restart needs one model to take a step and a second to stop, so one
    /// that cannot afford two models is not launched. At an optimum no
    /// restart step is accepted, so neither the FTOL nor the stagnation stop
    /// can fire (stagnation needs a descent first) and a restart would
    /// contract to the final radius, a fresh model per halving: four fresh
    /// models (the halvings that make a radius local) settle whether anything
    /// material is left, and a restart that gains is itself confirmed by the
    /// next one.
    pub(super) fn for_model_size(
        model_size: usize,
        maxeval: usize,
        ftol_abs: f64,
        ftol_rel: f64,
    ) -> Self {
        Self {
            maxeval,
            min_restart_budget: 2 * (model_size + 1),
            restart_budget: 4 * (model_size + 1),
            max_restarts: JOINT_MAX_CONFIRMATION_RESTARTS,
            ftol_abs,
            ftol_rel,
        }
    }
}

/// Confirms a TrustBQ joint FTOL or stagnation stop by restarts from the
/// incumbent: the stop stands once a restart gains no more than the
/// tolerance, keeping its label whatever the confirming restart returned.
/// A material gain moves the incumbent, which is then confirmed in turn,
/// unless the fit's budget is spent (then an honest `MaxEvaluations` stop).
/// A restart that exhausted only its own allowance is not a stop. The stop is
/// recorded as unconfirmed when the restart cap is reached, when fewer than
/// `min_restart_budget` evaluations remain, or when a restart fails; a host
/// interrupt is propagated. Radius and step stops are TrustBQ's native
/// convergence and need no confirmation.
///
/// `run_restart(start, budget, spent)` runs one restart from `start` with
/// `budget` evaluations, `spent` having been used so far, and returns its
/// result with the total evaluations spent afterwards (counted even when it
/// fails). Returns the final result, total evaluations and the outcome.
pub(super) fn confirm_joint_trust_bq_stop<R>(
    mut result: TrustBqResult,
    mut fevals: usize,
    plan: &JointRestartPlan,
    mut run_restart: R,
) -> Result<(TrustBqResult, usize, JointFtolConfirmation)>
where
    R: FnMut(&[f64], usize, usize) -> (Result<TrustBqResult>, usize),
{
    let mut confirmation = JointFtolConfirmation::NotNeeded;
    let mut restarts = 0usize;
    let confirmed_stop_reason = result.stop_reason;
    while matches!(
        result.stop_reason,
        TrustBqStopReason::ObjectiveTolerance | TrustBqStopReason::ObjectiveStagnation
    ) {
        let incumbent = result.fmin;
        if restarts == plan.max_restarts {
            confirmation = JointFtolConfirmation::Unconfirmed {
                reason: "restart_cap_reached",
                restarts,
                last_gain: None,
            };
            break;
        }
        let remaining = plan.maxeval.saturating_sub(fevals);
        if remaining < plan.min_restart_budget {
            confirmation = JointFtolConfirmation::Unconfirmed {
                reason: "insufficient_evaluation_budget",
                restarts,
                last_gain: None,
            };
            break;
        }
        let (restart, spent) = run_restart(&result.x, remaining.min(plan.restart_budget), fevals);
        restarts += 1;
        fevals = spent;
        let restart = match restart {
            Ok(restart) => restart,
            // A host interrupt stops the fit, as it does in the first pass;
            // only a genuine optimizer failure leaves the stop unconfirmed.
            Err(error @ MixedModelError::Interrupted(_)) => return Err(error),
            Err(_) => {
                confirmation = JointFtolConfirmation::Unconfirmed {
                    reason: "restart_failed",
                    restarts,
                    last_gain: None,
                };
                break;
            }
        };
        // The restart starts at the incumbent and reports its best point, so
        // it can only match or improve on it.
        let gain = incumbent - restart.fmin;
        result = restart;
        if gain <= plan.ftol_abs.max(plan.ftol_rel * incumbent.abs()) {
            result.stop_reason = confirmed_stop_reason;
            confirmation = JointFtolConfirmation::Confirmed;
            break;
        }
        confirmation = JointFtolConfirmation::NotNeeded;
        if fevals >= plan.maxeval {
            result.stop_reason = TrustBqStopReason::MaxEvaluations;
            break;
        }
        if result.stop_reason == TrustBqStopReason::MaxEvaluations {
            result.stop_reason = confirmed_stop_reason;
        }
    }
    Ok((result, fevals, confirmation))
}

/// Initial steps for a confirming restart: the first pass's steps scaled by
/// [`JOINT_RESTART_STEP_FACTOR`], shrunk for any component that
/// lies strictly inside its lower bound by less than the step. NLopt's
/// BOBYQA moves such a start to `lb + step` (bobyqa.c, "components of X
/// that become within distance RHOBEG from their bounds"), which would start
/// the restart away from the incumbent it is meant to confirm; half the
/// distance to the bound keeps the start in place. A component exactly on
/// its bound is left on it by BOBYQA and keeps the scaled step.
#[cfg(feature = "nlopt")]
fn joint_restart_steps(initial_step: &[f64], params: &[f64], lower_bounds: &[f64]) -> Vec<f64> {
    initial_step
        .iter()
        .zip(params)
        .zip(lower_bounds)
        .map(|((&step, &value), &lower)| {
            let step = step * JOINT_RESTART_STEP_FACTOR;
            let gap = value - lower;
            if lower.is_finite() && gap > 0.0 && gap <= step && 0.5 * gap > 0.0 {
                0.5 * gap
            } else {
                step
            }
        })
        .collect()
}

/// Records an FTOL stop whose confirmation could not be completed, so the
/// certificate never implies a confirmation that did not happen.
fn record_joint_ftol_confirmation(
    certificate: &mut OptimizerCertificate,
    confirmation: &JointFtolConfirmation,
    optimizer: &str,
) {
    let JointFtolConfirmation::Unconfirmed {
        reason,
        restarts,
        last_gain,
    } = confirmation
    else {
        return;
    };
    let mut diagnostic = Diagnostic::new(
        DiagnosticCode::OptimizerRecovery,
        DiagnosticSeverity::Warning,
        DiagnosticStage::Certification,
        format!(
            "joint GLMM {optimizer} FTOL stop could not be confirmed by a restart from the incumbent; the reported optimum may be short of the true optimum"
        ),
    )
    .with_suggested_actions(vec![
        "raise max_feval (or leave it at the default) so the FTOL stop can be confirmed".to_string(),
    ]);
    diagnostic
        .payload
        .insert("fit_mode".to_string(), serde_json::json!("joint_glmm"));
    diagnostic.payload.insert(
        "ftol_confirmation".to_string(),
        serde_json::json!("unconfirmed"),
    );
    diagnostic
        .payload
        .insert("reason".to_string(), serde_json::json!(reason));
    diagnostic
        .payload
        .insert("restarts".to_string(), serde_json::json!(restarts));
    if let Some(gain) = last_gain {
        diagnostic
            .payload
            .insert("last_restart_gain".to_string(), serde_json::json!(gain));
    }
    certificate.diagnostics.push(diagnostic);
}

/// Smallest diagonal entry of the Cholesky factor of the profiled β
/// correlation matrix for which the TrustBQ joint transform decorrelates.
/// `L_kk² = 1 − R²_k`, the unexplained fraction of coefficient `k`'s
/// variance given the coefficients before it, so the bound is a variance
/// inflation factor of 1e6 (R² = 1 − 1e-6). Beyond it the working covariance
/// is near-singular (collinear covariates, or quasi-separation where the SEs
/// are also capped), its thin directions are set by rounding in the working
/// Hessian rather than by the likelihood, and the unit ball in `z` would be a
/// 1e-3-thin sliver along them. The SE scaling alone is kept instead.
const JOINT_MIN_CORRELATION_FACTOR_DIAGONAL: f64 = 1.0e-3;

/// `factor` when its smallest diagonal entry is at least
/// [`JOINT_MIN_CORRELATION_FACTOR_DIAGONAL`], else `None`.
pub(super) fn well_conditioned_correlation_factor(factor: DMatrix<f64>) -> Option<DMatrix<f64>> {
    let min_diagonal = factor
        .diagonal()
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    (min_diagonal >= JOINT_MIN_CORRELATION_FACTOR_DIAGONAL).then_some(factor)
}

/// β initial step when no profiled standard error is available.
const JOINT_DEFAULT_BETA_STEP: f64 = 0.1;
/// Initial step (NLopt BOBYQA) and coordinate scale (TrustBQ) for each
/// covariance parameter of the joint fit.
const JOINT_THETA_STEP: f64 = 0.5;
/// TrustBQ joint initial radius in scaled coordinates: one scale unit, i.e.
/// the NLopt BOBYQA initial steps.
const JOINT_TRUST_BQ_INITIAL_RADIUS: f64 = 1.0;
/// TrustBQ joint final radius in scaled coordinates.
const JOINT_TRUST_BQ_FINAL_RADIUS: f64 = 1.0e-5;

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
        // MixedModels.jl runs its joint fit without an evaluation cap; NLopt
        // BOBYQA's FTOL-confirmation restarts need the same room as TrustBQ
        // rather than a single-pass budget.
        Optimizer::TrustBq | Optimizer::NloptBobyqa => {
            trust_bq_joint_glmm_default_maxeval(n_params)
        }
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

pub(crate) fn trust_bq_status_label(status: TrustBqStopReason) -> &'static str {
    match status {
        TrustBqStopReason::RadiusBelowTolerance => "RADIUS_REACHED",
        TrustBqStopReason::ObjectiveTolerance => "FTOL_REACHED",
        TrustBqStopReason::MaxEvaluations => "MAXEVAL_REACHED",
        TrustBqStopReason::StepBelowTolerance => "XTOL_REACHED",
        TrustBqStopReason::ObjectiveStagnation => "FTOL_REACHED",
        TrustBqStopReason::CertifiedConvergence => "FTOL_REACHED",
        TrustBqStopReason::GradientBelowTolerance => "GTOL_REACHED",
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
