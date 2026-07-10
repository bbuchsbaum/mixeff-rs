//! Optimizer drivers and convergence machinery for the linear mixed model:
//! convergence verification, the covariance-cone KKT certificates and
//! KKT-guided boundary restart, and the family of fit drivers (scalar,
//! pattern-search, trust-BQ, NLopt/COBYLA/PRIMA) with their status labels and
//! finite-difference helpers. Moved verbatim from the former single-file
//! `linear.rs`.

use super::*;

#[derive(Debug, Clone)]
struct KktBoundaryRestartCandidate {
    theta: Vec<f64>,
    objective: f64,
    reason: String,
}

fn record_pattern_eval<F>(
    objective: &mut F,
    theta: &[f64],
    feval_count: &mut i64,
    fit_log: &mut Vec<FitLogEntry>,
    best_theta: &mut Vec<f64>,
    best_fmin: &mut f64,
) -> Result<f64>
where
    F: FnMut(&[f64]) -> Result<f64>,
{
    let obj = objective(theta)?;
    *feval_count += 1;
    fit_log.push(FitLogEntry {
        theta: theta.to_vec(),
        objective: obj,
    });
    if obj < *best_fmin {
        *best_fmin = obj;
        *best_theta = theta.to_vec();
    }
    Ok(obj)
}

impl LinearMixedModel {
    /// Run bounded convergence verification and attach the result to the
    /// optimizer certificate.
    ///
    /// Refits the model from the current optimum (and from one or more
    /// jittered starts, and via alternate optimizers when consensus is
    /// requested) and reports whether the runs agree on θ, β, and the
    /// objective. The result is stored on
    /// `compiler_artifact.optimizer_certificate.verification` so the
    /// audit report and the convergence verdict can pick it up. lme4's
    /// analogue is `allFit()`.
    ///
    /// # When to call
    ///
    /// Run this after [`fit`](Self::fit) when the compact print shows
    /// `convergence: caution` or `convergence: ok` with a
    /// `next: verify convergence` hint — that is, when the
    /// optimizer stopped acceptably but gradient/Hessian evidence is
    /// weak or unavailable, or when the model is at a boundary or
    /// reduced-rank optimum and you want optimizer-agreement
    /// reassurance. It is **not** the right tool for structural design
    /// failures (row-saturated random effects, separation,
    /// rank-deficient fixed effects); the verdict's `next:` line
    /// already excludes optimizer tinkering when the source is
    /// structural.
    ///
    /// Use [`verify_convergence_with_options`](Self::verify_convergence_with_options)
    /// when you need finer-grained control over jitter scale, alternate
    /// optimizer choice, or agreement tolerances.
    pub fn verify_convergence(&mut self) -> Result<ConvergenceVerification> {
        self.verify_convergence_with_options(ConvergenceVerificationOptions::default())
    }

    /// Run convergence verification with explicit controls.
    pub fn verify_convergence_with_options(
        &mut self,
        options: ConvergenceVerificationOptions,
    ) -> Result<ConvergenceVerification> {
        if self.optsum.feval <= 0 {
            let verification = ConvergenceVerification::not_run("model has not been fitted");
            if let Some(certificate) = &mut self.compiler_artifact.optimizer_certificate {
                certificate.verification = Some(verification.clone());
            }
            return Ok(verification);
        }

        let reference_theta = self.theta();
        let reference_beta = self.beta().iter().copied().collect::<Vec<_>>();
        let reference_objective = self.optsum.fmin.is_finite().then_some(self.optsum.fmin);
        let reference_effective_ranks = self
            .compiler_artifact
            .effective_covariance
            .iter()
            .map(|summary| summary.supported_rank)
            .collect::<Vec<_>>();

        let mut runs = Vec::new();
        if options.restart_from_optimum {
            runs.push(self.convergence_verification_run(
                "restart_from_optimum",
                self.optsum.optimizer,
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
                &self.lower_bounds(),
                options.jitter_scale,
                jitter_index,
            );
            runs.push(self.convergence_verification_run(
                &format!("jitter_restart_{}", jitter_index + 1),
                self.optsum.optimizer,
                &start,
                &reference_theta,
                &reference_beta,
                reference_objective,
                &reference_effective_ranks,
                &options,
            ));
        }

        if options.run_optimizer_consensus {
            for optimizer in self.alternate_verification_optimizers() {
                runs.push(self.convergence_verification_run(
                    &format!("optimizer_consensus_{}", optimizer_name(optimizer)),
                    optimizer,
                    &reference_theta,
                    &reference_theta,
                    &reference_beta,
                    reference_objective,
                    &reference_effective_ranks,
                    &options,
                ));
            }
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

        if let Some(certificate) = &mut self.compiler_artifact.optimizer_certificate {
            certificate.verification = Some(verification.clone());
        }
        Ok(verification)
    }

    fn convergence_verification_run(
        &self,
        label: &str,
        optimizer: Optimizer,
        start_theta: &[f64],
        reference_theta: &[f64],
        reference_beta: &[f64],
        reference_objective: Option<f64>,
        reference_effective_ranks: &[usize],
        options: &ConvergenceVerificationOptions,
    ) -> ConvergenceVerificationRun {
        let mut candidate = self.clone();
        let result = candidate
            .reset_for_convergence_verification(start_theta, options.max_function_evaluations)
            .and_then(|_| candidate.fit_with_forced_optimizer(self.optsum.reml, optimizer));

        match result {
            Ok(()) => {
                let objective_value = candidate
                    .optsum
                    .fmin
                    .is_finite()
                    .then_some(candidate.optsum.fmin);
                let theta = candidate.theta();
                let beta = candidate.beta().iter().copied().collect::<Vec<_>>();
                let effective_ranks = candidate
                    .compiler_artifact
                    .effective_covariance
                    .iter()
                    .map(|summary| summary.supported_rank)
                    .collect::<Vec<_>>();
                let objective_delta = objective_value
                    .zip(reference_objective)
                    .map(|(value, reference)| (value - reference).abs());
                let max_abs_theta_delta = max_abs_delta(&theta, reference_theta);
                let max_abs_beta_delta = max_abs_delta(&beta, reference_beta);
                let ranks_agree = effective_ranks == reference_effective_ranks;
                let mut diagnostics = Vec::new();
                if objective_delta
                    .map(|delta| delta > options.objective_tolerance)
                    .unwrap_or(true)
                {
                    diagnostics.push("objective changed beyond tolerance".to_string());
                }
                if max_abs_theta_delta
                    .map(|delta| delta > options.theta_tolerance)
                    .unwrap_or(true)
                {
                    diagnostics.push("theta parameterization changed beyond tolerance".to_string());
                }
                if max_abs_beta_delta
                    .map(|delta| delta > options.beta_tolerance)
                    .unwrap_or(true)
                {
                    diagnostics.push("fixed-effect estimates changed beyond tolerance".to_string());
                }
                if !ranks_agree {
                    diagnostics
                        .push("effective covariance ranks changed during verification".to_string());
                }
                let agrees = objective_delta
                    .map(|delta| delta <= options.objective_tolerance)
                    .unwrap_or(false)
                    && max_abs_theta_delta
                        .map(|delta| delta <= options.theta_tolerance)
                        .unwrap_or(false)
                    && max_abs_beta_delta
                        .map(|delta| delta <= options.beta_tolerance)
                        .unwrap_or(false)
                    && ranks_agree;

                ConvergenceVerificationRun {
                    label: label.to_string(),
                    optimizer_name: Some(optimizer_name(optimizer).to_string()),
                    return_code: Some(candidate.optsum.return_value.clone()),
                    objective_value,
                    objective_delta,
                    max_abs_theta_delta,
                    max_abs_beta_delta,
                    effective_ranks,
                    agrees,
                    diagnostics,
                }
            }
            Err(error) => ConvergenceVerificationRun {
                label: label.to_string(),
                optimizer_name: Some(optimizer_name(optimizer).to_string()),
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

    fn reset_for_convergence_verification(
        &mut self,
        start_theta: &[f64],
        max_function_evaluations: usize,
    ) -> Result<()> {
        let previous = self.optsum.clone();
        let mut optsum = OptSummary::new(start_theta.to_vec());
        optsum.xtol_zero_abs = previous.xtol_zero_abs;
        optsum.ftol_zero_abs = previous.ftol_zero_abs;
        optsum.ftol_rel = previous.ftol_rel;
        optsum.ftol_abs = previous.ftol_abs;
        optsum.xtol_rel = previous.xtol_rel;
        optsum.xtol_abs = previous.xtol_abs;
        optsum.initial_step = previous.initial_step;
        optsum.max_feval = max_function_evaluations as i64;
        optsum.max_time = previous.max_time;
        optsum.optimizer = previous.optimizer;
        optsum.backend = previous.backend;
        optsum.optimizer_source = previous.optimizer_source;
        optsum.caller_set_fields = previous.caller_set_fields;
        optsum.rhobeg = previous.rhobeg;
        optsum.rhoend = previous.rhoend;
        optsum.reml = previous.reml;
        optsum.n_agq = previous.n_agq;
        optsum.sigma = previous.sigma;
        self.optsum = optsum;
        self.set_theta(start_theta)?;
        self.update_l()
    }

    pub(super) fn fit_with_forced_optimizer(
        &mut self,
        reml: bool,
        optimizer: Optimizer,
    ) -> Result<()> {
        self.optsum.reml = reml;
        self.set_initial_objective_with_rescue()?;
        match optimizer {
            Optimizer::PatternSearch => {
                if self.n_theta() == 1 {
                    self.fit_scalar_single_theta()?;
                } else {
                    self.fit_multivariate_pattern_search()?;
                }
            }
            Optimizer::Cobyla => {
                self.fit_cobyla(reml)?;
            }
            Optimizer::TrustBq => {
                let maxeval = (self.optsum.max_feval > 0).then_some(self.optsum.max_feval as usize);
                self.fit_trust_bq_with_maxeval(reml, maxeval)?;
            }
            Optimizer::NloptBobyqa => {
                #[cfg(feature = "nlopt")]
                self.fit_nlopt_small_theta_with_maxeval(
                    reml,
                    Some(self.optsum.max_feval.max(1) as usize),
                )?;
                #[cfg(not(feature = "nlopt"))]
                return Err(MixedModelError::Unsupported(
                    "Optimizer::NloptBobyqa requires the `nlopt` feature; \
                     rebuild with `--features nlopt` or pick a non-NLopt optimizer"
                        .to_string(),
                ));
            }
            Optimizer::NloptNewuoa => {
                #[cfg(feature = "nlopt")]
                self.fit_nlopt_large_theta_with_maxeval(
                    reml,
                    Some(self.optsum.max_feval.max(1) as usize),
                )?;
                #[cfg(not(feature = "nlopt"))]
                return Err(MixedModelError::Unsupported(
                    "Optimizer::NloptNewuoa requires the `nlopt` feature; \
                     rebuild with `--features nlopt` or pick a non-NLopt optimizer"
                        .to_string(),
                ));
            }
            Optimizer::PrimaBobyqa => {
                #[cfg(feature = "prima")]
                self.fit_prima_bobyqa_with_maxeval(
                    reml,
                    Some(self.optsum.max_feval.max(1) as usize),
                )?;
                #[cfg(not(feature = "prima"))]
                return Err(MixedModelError::Unsupported(
                    "Optimizer::PrimaBobyqa requires the `prima` feature and a system \
                     PRIMA C library (`libprimac`); rebuild with `--features prima` \
                     or pick a non-PRIMA optimizer"
                        .to_string(),
                ));
            }
            Optimizer::PrimaCobyla | Optimizer::PrimaLincoa | Optimizer::PrimaNewuoa => {
                return Err(MixedModelError::Unsupported(
                    "Only Optimizer::PrimaBobyqa is wired to the LMM optimizer path; \
                     PRIMA COBYLA, LINCOA, and NEWUOA are reserved for later backend parity work"
                        .to_string(),
                ));
            }
        }
        self.apply_kkt_guided_boundary_restart(reml)?;
        self.apply_active_face_refit()?;
        self.refresh_optimizer_certificate();
        self.refresh_effective_covariance_summaries();
        self.refresh_covariance_parameter_traces();
        self.refresh_fixed_effect_covariance_matrix();
        self.refresh_fixed_effect_inference_table();
        Ok(())
    }

    fn alternate_verification_optimizers(&self) -> Vec<Optimizer> {
        let current = self.optsum.optimizer;
        let alternate = if current != Optimizer::PatternSearch {
            Optimizer::PatternSearch
        } else if self.n_theta() == 1 {
            Optimizer::Cobyla
        } else if self.n_theta() <= 6 {
            Optimizer::NloptBobyqa
        } else {
            Optimizer::Cobyla
        };
        vec![alternate]
    }

    pub(super) fn refresh_optimizer_certificate(&mut self) {
        let theta = self.theta();
        let lower_bounds = self.lower_bounds();
        let mut certificate = OptimizerCertificate::from_opt_summary_with_context(
            &self.optsum,
            &theta,
            &lower_bounds,
            Some(self.dims.n),
        );
        if certificate.evidence.optimizer_stop.acceptable_stop {
            if let Some(reason) = self.derivative_certificate_skip_reason(&certificate) {
                certificate.mark_derivative_checks_not_assessed(reason);
            } else if let Some(derivatives) =
                self.finite_difference_optimizer_derivatives(&theta, &lower_bounds)
            {
                let (gradient_tolerance, hessian_tolerance) =
                    self.derivative_certificate_tolerances(certificate.objective_value);
                certificate.apply_derivative_evidence(
                    derivatives,
                    gradient_tolerance,
                    hessian_tolerance,
                );
            }
        }
        self.reword_optimizer_certificate_diagnostics(&mut certificate);
        self.compiler_artifact.optimizer_certificate = Some(certificate);
    }

    fn reword_optimizer_certificate_diagnostics(&self, certificate: &mut OptimizerCertificate) {
        for diagnostic in &mut certificate.diagnostics {
            if diagnostic.code != DiagnosticCode::BoundaryParameter {
                continue;
            }
            let Some(theta_index) = diagnostic
                .payload
                .get("theta_index")
                .and_then(serde_json::Value::as_u64)
                .map(|value| value as usize)
            else {
                continue;
            };
            let Some((term_id, source_syntax, parameter_role)) =
                self.covariance_parameter_context(theta_index)
            else {
                continue;
            };

            diagnostic.message =
                format!("{parameter_role} in {source_syntax} is on its lower bound");
            diagnostic.affected_terms = vec![source_syntax.clone()];
            diagnostic
                .payload
                .insert("term_id".to_string(), serde_json::json!(term_id));
            diagnostic.payload.insert(
                "source_syntax".to_string(),
                serde_json::json!(source_syntax),
            );
            diagnostic.payload.insert(
                "parameter_role".to_string(),
                serde_json::json!(parameter_role),
            );
        }
    }

    fn covariance_parameter_context(&self, theta_index: usize) -> Option<(String, String, String)> {
        for theta_map in &self.compiler_artifact.theta_maps {
            let block = theta_map.block();
            let Some(slot) = block
                .theta_slots
                .iter()
                .find(|slot| slot.global_index == Some(theta_index))
            else {
                continue;
            };
            let row_basis = block
                .optimizer_basis
                .get(slot.lambda_row)
                .cloned()
                .unwrap_or_else(|| format!("basis {}", slot.lambda_row + 1));
            let col_basis = block
                .optimizer_basis
                .get(slot.lambda_col)
                .cloned()
                .unwrap_or_else(|| format!("basis {}", slot.lambda_col + 1));
            let parameter_role = if slot.lambda_row == slot.lambda_col {
                format!("standard deviation for {row_basis}")
            } else {
                format!("covariance link between {row_basis} and {col_basis}")
            };
            let source_syntax = self
                .compiler_artifact
                .semantic_model
                .random_terms
                .iter()
                .find(|term| term.id == block.term_id)
                .map(|term| term.source_syntax.text.clone())
                .unwrap_or_else(|| format!("random-effect term for {}", block.group));
            return Some((block.term_id.clone(), source_syntax, parameter_role));
        }

        None
    }

    fn derivative_certificate_skip_reason(
        &self,
        certificate: &OptimizerCertificate,
    ) -> Option<String> {
        if self.suppress_derivative_diagnostics {
            return Some(
                "finite-difference derivative diagnostics are skipped for internal bootstrap-replicate refits"
                    .to_string(),
            );
        }

        let n_theta = certificate.evidence.parameter_space.n_theta;
        if n_theta == 0 {
            return Some(
                "there are no theta parameters, so derivative KKT/Hessian checks are not applicable"
                    .to_string(),
            );
        }

        if certificate.evidence.parameter_space.n_boundary > 0 {
            return Some(format!(
                "one or more covariance parameters are on a variance-component boundary (parameter indices: {}); boundary fits are reported as singular/boundary, not non-converged",
                certificate
                    .evidence
                    .parameter_space
                    .boundary_indices
                    .iter()
                    .map(|index| index.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }

        let nparmax = self
            .compiler_artifact
            .compiler_policy
            .thresholds
            .convergence_derivative_nparmax;
        if n_theta > nparmax {
            return Some(format!(
                "theta dimension {n_theta} exceeds convergence_derivative_nparmax {nparmax}; finite-difference KKT/Hessian checks are skipped for large-theta optimizer regimes"
            ));
        }

        None
    }

    fn derivative_certificate_tolerances(&self, objective_value: Option<f64>) -> (f64, f64) {
        let objective_scale = objective_value.unwrap_or(self.optsum.fmin).abs().max(1.0);
        let objective_tolerance = self
            .optsum
            .ftol_abs
            .max(self.optsum.ftol_zero_abs)
            .max(self.optsum.ftol_rel.max(0.0) * objective_scale);
        let gradient_tolerance = objective_tolerance.sqrt().max(1e-4);
        let hessian_tolerance = 1e-5_f64.max(gradient_tolerance * 1e-2);
        (gradient_tolerance, hessian_tolerance)
    }

    fn finite_difference_optimizer_derivatives(
        &self,
        theta: &[f64],
        lower_bounds: &[f64],
    ) -> Option<OptimizerDerivativeEvidence> {
        let n_theta = theta.len();
        if n_theta == 0
            || n_theta
                > self
                    .compiler_artifact
                    .compiler_policy
                    .thresholds
                    .convergence_derivative_nparmax
        {
            return None;
        }

        let weight_logdet_correction = self.weight_logdet_correction();
        let mut evaluator: Option<LinearMixedModel> = None;
        let mut objective = |trial: &[f64]| {
            if let Some(value) = self.profiled_objective_fast(trial) {
                Some(value - weight_logdet_correction)
            } else {
                let evaluator = evaluator.get_or_insert_with(|| self.clone());
                evaluator.objective_at_fast_or_generic(trial).ok()
            }
        };

        let f0 = objective(theta)?;
        if !f0.is_finite() {
            return None;
        }

        let boundary_tolerance = self.optsum.xtol_zero_abs.max(1e-12) * 10.0;
        let boundary_mask = theta
            .iter()
            .zip(lower_bounds.iter())
            .map(|(&value, &lower)| {
                lower.is_finite() && (value - lower).abs() <= boundary_tolerance
            })
            .collect::<Vec<_>>();
        let gradient_steps = finite_difference_steps(theta, lower_bounds, 1e-5);
        let hessian_steps = finite_difference_steps(theta, lower_bounds, 1e-4);

        let mut gradient = vec![0.0; n_theta];
        for index in 0..n_theta {
            gradient[index] = finite_difference_gradient_coordinate(
                &mut objective,
                theta,
                lower_bounds,
                f0,
                index,
                gradient_steps[index],
            )?;
        }

        let free_indices = boundary_mask
            .iter()
            .enumerate()
            .filter_map(|(index, is_boundary)| (!*is_boundary).then_some(index))
            .collect::<Vec<_>>();
        let mut hessian = DMatrix::zeros(n_theta, n_theta);
        for &row in &free_indices {
            let row_step =
                feasible_central_step(theta[row], lower_bounds[row], hessian_steps[row])?;
            let mut plus = theta.to_vec();
            let mut minus = theta.to_vec();
            plus[row] += row_step;
            minus[row] -= row_step;
            let f_plus = objective(&plus)?;
            let f_minus = objective(&minus)?;
            if !f_plus.is_finite() || !f_minus.is_finite() {
                return None;
            }
            hessian[(row, row)] = (f_plus - 2.0 * f0 + f_minus) / (row_step * row_step);

            for &col in free_indices.iter().filter(|&&col| col > row) {
                let col_step =
                    feasible_central_step(theta[col], lower_bounds[col], hessian_steps[col])?;
                let f_pp = finite_difference_objective_2d(
                    &mut objective,
                    theta,
                    row,
                    row_step,
                    col,
                    col_step,
                )?;
                let f_pm = finite_difference_objective_2d(
                    &mut objective,
                    theta,
                    row,
                    row_step,
                    col,
                    -col_step,
                )?;
                let f_mp = finite_difference_objective_2d(
                    &mut objective,
                    theta,
                    row,
                    -row_step,
                    col,
                    col_step,
                )?;
                let f_mm = finite_difference_objective_2d(
                    &mut objective,
                    theta,
                    row,
                    -row_step,
                    col,
                    -col_step,
                )?;
                let value = (f_pp - f_pm - f_mp + f_mm) / (4.0 * row_step * col_step);
                hessian[(row, col)] = value;
                hessian[(col, row)] = value;
            }
        }

        Some(OptimizerDerivativeEvidence {
            method: EvidenceMethod::FiniteDifference,
            gradient,
            hessian: Some(hessian),
        })
    }

    pub(super) fn refresh_covariance_parameter_traces(&mut self) {
        let lambdas = self
            .reterms
            .iter()
            .map(|reterm| matrix_rows(&reterm.lambda))
            .collect::<Vec<_>>();
        let sd_scale = if self.optsum.feval >= 0 {
            Some(self.sigma())
        } else {
            None
        };
        self.compiler_artifact.refresh_covariance_parameter_traces(
            Some(&lambdas),
            sd_scale,
            &self.parmap,
        );
    }

    pub(super) fn refresh_effective_covariance_summaries(&mut self) {
        let Some(certificate) = &self.compiler_artifact.optimizer_certificate else {
            return;
        };
        // ConvergedPenalised fits still expose well-defined Λ matrices, so
        // their effective-covariance summaries are meaningful and should be
        // refreshed alongside the standard converged statuses. The
        // *promotion* path below stays narrower (only Interior/Boundary
        // promote to ReducedRank) — ConvergedPenalised is a contractual
        // leaf and must not be silently rewritten.
        if !matches!(
            certificate.status,
            crate::compiler::FitStatus::ConvergedInterior
                | crate::compiler::FitStatus::ConvergedBoundary
                | crate::compiler::FitStatus::ConvergedReducedRank
                | crate::compiler::FitStatus::ConvergedPenalised
        ) {
            self.compiler_artifact.effective_covariance.clear();
            return;
        }

        let thresholds = self.compiler_artifact.compiler_policy.thresholds.clone();
        let sigma_sq = self.sigma().powi(2);
        let mut summaries = Vec::with_capacity(self.reterms.len());
        let mut reductions = Vec::new();
        let mut transitions = Vec::new();
        let mut diagnostics = Vec::new();

        for (term_index, reterm) in self.reterms.iter().enumerate() {
            let theta_map = self.compiler_artifact.theta_maps.get(term_index);
            let term_id = theta_map
                .map(|map| map.block().term_id.clone())
                .unwrap_or_else(|| format!("r{term_index}"));
            let source_syntax = self
                .compiler_artifact
                .semantic_model
                .random_terms
                .iter()
                .find(|term| term.id == term_id)
                .map(|term| term.source_syntax.text.clone())
                .unwrap_or_else(|| format!("random-effect term for {}", reterm.grouping_name));
            let requested_basis = theta_map
                .map(|map| map.block().optimizer_basis.clone())
                .filter(|basis| basis.len() == reterm.vsize)
                .unwrap_or_else(|| {
                    reterm
                        .cnames
                        .iter()
                        .map(|name| user_basis_label(name))
                        .collect()
                });
            let requested_rank = reterm.vsize;
            let covariance = sigma_sq * (&reterm.lambda * reterm.lambda.transpose());
            let eig = SymmetricEigen::new(covariance);
            let mut pairs = (0..reterm.vsize)
                .map(|idx| {
                    (
                        eig.eigenvalues[idx],
                        eig.eigenvectors
                            .column(idx)
                            .iter()
                            .copied()
                            .collect::<Vec<_>>(),
                    )
                })
                .collect::<Vec<_>>();
            pairs.sort_by(|left, right| {
                right
                    .0
                    .partial_cmp(&left.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let max_eigenvalue = pairs
                .first()
                .map(|(value, _)| value.max(0.0))
                .unwrap_or(0.0);
            let rank_tolerance = thresholds.effective_rank_tolerance(max_eigenvalue);
            let total_positive: f64 = pairs.iter().map(|(value, _)| value.max(0.0)).sum();
            let pairs_snapshot = pairs.clone();
            let mut directions = Vec::new();
            let mut unsupported_directions = Vec::new();

            for (pc_index, (eigenvalue, vector)) in pairs.into_iter().enumerate() {
                let oriented = orient_eigenvector(vector);
                let loadings = requested_basis
                    .iter()
                    .cloned()
                    .zip(oriented)
                    .map(|(basis, loading)| BasisLoading { basis, loading })
                    .collect::<Vec<_>>();
                let nonnegative_eigenvalue = eigenvalue.max(0.0);
                let direction = SupportedCovarianceDirection {
                    label: format!("PC{}", pc_index + 1),
                    loadings,
                    eigenvalue: Some(if nonnegative_eigenvalue <= rank_tolerance {
                        0.0
                    } else {
                        nonnegative_eigenvalue
                    }),
                    variance_explained: if total_positive > 0.0 {
                        Some(nonnegative_eigenvalue / total_positive)
                    } else {
                        Some(0.0)
                    },
                    user_scale_summary: String::new(),
                };
                let mut direction = direction;
                direction.user_scale_summary = format_loading_summary(&direction.loadings);
                if nonnegative_eigenvalue > rank_tolerance {
                    directions.push(direction);
                } else {
                    unsupported_directions.push(direction);
                }
            }

            let supported_rank = directions.len();
            let status = if supported_rank == requested_rank {
                EffectiveRankStatus::FullRank
            } else {
                EffectiveRankStatus::ReducedRank
            };
            let inference_consequence = if status == EffectiveRankStatus::ReducedRank {
                "fixed-effect inference is conditional on a certificate-time reduced-rank covariance summary; unsupported directions are not evidence of zero population variance".to_string()
            } else {
                "fixed-effect inference can condition on the fitted full-rank covariance for this term".to_string()
            };

            let interpretable_submodel = if status == EffectiveRankStatus::ReducedRank {
                detect_interpretable_submodel(
                    &pairs_snapshot,
                    &requested_basis,
                    requested_rank,
                    rank_tolerance,
                    sigma_sq,
                    &self.compiler_artifact.semantic_model.random_terms,
                    &term_id,
                )
            } else {
                None
            };

            summaries.push(EffectiveCovarianceSummary {
                term_id: term_id.clone(),
                source_syntax: source_syntax_for_term(
                    &self.compiler_artifact.semantic_model.random_terms,
                    &term_id,
                ),
                requested_basis: requested_basis.clone(),
                requested_rank,
                supported_rank,
                status,
                directions,
                unsupported_directions,
                inference_consequence: inference_consequence.clone(),
                interpretable_submodel: interpretable_submodel.clone(),
            });

            if status == EffectiveRankStatus::ReducedRank {
                let mut suggested_actions = vec![
                    "treat unsupported covariance directions as unsupported by this fit, not as proof of zero population variance".to_string(),
                ];
                if let Some(submodel) = &interpretable_submodel {
                    suggested_actions.push(format!(
                        "consider refitting the simpler random-effect term {}; this fitted model was not silently refit",
                        submodel.suggested_formula
                    ));
                }
                let mut diagnostic = Diagnostic::new(
                    DiagnosticCode::CovarianceReduced,
                    DiagnosticSeverity::Info,
                    DiagnosticStage::Certification,
                    format!(
                        "fitted covariance for {source_syntax} has effective rank {supported_rank} of requested rank {requested_rank}"
                    ),
                )
                .with_affected_terms(vec![source_syntax.clone()])
                .with_suggested_actions(suggested_actions);
                diagnostic
                    .payload
                    .insert("term_id".to_string(), serde_json::json!(term_id.clone()));
                diagnostic.payload.insert(
                    "source_syntax".to_string(),
                    serde_json::json!(source_syntax.clone()),
                );
                diagnostic.payload.insert(
                    "rank_tolerance".to_string(),
                    serde_json::json!(rank_tolerance),
                );
                diagnostic.payload.insert(
                    "effective_rank_relative_tolerance".to_string(),
                    serde_json::json!(thresholds.effective_rank_relative_tolerance),
                );
                diagnostic.payload.insert(
                    "effective_rank_absolute_tolerance".to_string(),
                    serde_json::json!(thresholds.effective_rank_absolute_tolerance),
                );
                diagnostic.payload.insert(
                    "requested_rank".to_string(),
                    serde_json::json!(requested_rank),
                );
                diagnostic.payload.insert(
                    "supported_rank".to_string(),
                    serde_json::json!(supported_rank),
                );
                if let Some(submodel) = &interpretable_submodel {
                    if let Ok(payload) = serde_json::to_value(submodel) {
                        diagnostic
                            .payload
                            .insert("interpretable_submodel".to_string(), payload);
                    }
                }

                diagnostics.push(diagnostic.clone());
                reductions.push(ReductionRecord {
                    trigger: ReductionTrigger::CertificateTimeBoundary,
                    phase: "fit-time effective covariance rank".to_string(),
                    reason: format!(
                        "effective covariance rank {supported_rank} is below requested rank {requested_rank}"
                    ),
                    affected_term: term_id.clone(),
                    replacement_term: Some(format!(
                        "reduced_rank({}, basis = {}, rank = {})",
                        reterm.grouping_name,
                        requested_basis.join(" + "),
                        supported_rank
                    )),
                    inference_consequence: inference_consequence.clone(),
                    diagnostics: Vec::new(),
                });

                if let Some(theta_map) = theta_map {
                    transitions.push(CovarianceFamilyTransition {
                        from: theta_map.family(),
                        to: CovarianceFamily::ReducedRank {
                            rank: Some(supported_rank),
                        },
                        trigger: ReductionTrigger::CertificateTimeBoundary,
                        affected_term: term_id,
                        dropped_or_reparameterized_slots: Vec::new(),
                        inference_consequence,
                    });
                }
            }
        }

        self.compiler_artifact.effective_covariance = summaries;
        self.compiler_artifact.reductions.extend(reductions);
        self.compiler_artifact
            .covariance_transitions
            .extend(transitions);

        if !diagnostics.is_empty() {
            if let Some(certificate) = &mut self.compiler_artifact.optimizer_certificate {
                if matches!(
                    certificate.status,
                    crate::compiler::FitStatus::ConvergedInterior
                        | crate::compiler::FitStatus::ConvergedBoundary
                ) {
                    certificate.status = crate::compiler::FitStatus::ConvergedReducedRank;
                }
                certificate.diagnostics.extend(diagnostics);
            }
        }
    }
    /// Post-fit covariance-cone KKT diagnostic for scalar random-effect terms.
    ///
    /// This first certificate works in covariance space for terms of the form
    /// `(1 | group)`. It estimates `dF/dv` for `v = theta^2` by directional
    /// objective differences through the existing profiled LMM objective. No
    /// dense marginal covariance matrix is formed.
    pub fn scalar_covariance_kkt_certificate(&self) -> Result<ScalarCovarianceKktCertificate> {
        if !self.optsum.is_fitted() {
            return Err(MixedModelError::NotFitted);
        }
        if self.reterms.is_empty() {
            return Err(MixedModelError::NoRandomEffects);
        }
        if self
            .reterms
            .iter()
            .any(|term| term.vsize != 1 || term.n_theta() != 1)
        {
            return Err(MixedModelError::Unsupported(
                "scalar covariance KKT certificate currently supports only scalar random-effect terms"
                    .to_string(),
            ));
        }

        let theta = self.theta();
        let mut evaluator = self.clone();
        let objective = evaluator.objective_at(&theta)?;
        let variance_tolerance = SCALAR_KKT_VARIANCE_TOLERANCE;
        let score_tolerance = (1e-5 * (1.0 + objective.abs())).max(1e-6);
        let mut blocks = Vec::with_capacity(self.reterms.len());

        for (term_index, term) in self.reterms.iter().enumerate() {
            let theta_index = term_index;
            let theta_value = theta[theta_index].max(0.0);
            let variance = theta_value * theta_value;
            let score =
                Self::scalar_covariance_score(&mut evaluator, theta_index, &theta, objective)?;
            let complementarity = (variance * score).abs() / (1.0 + variance.abs() * score.abs());
            let classification = classify_scalar_covariance_kkt(
                variance,
                score,
                variance_tolerance,
                score_tolerance,
            );
            let residual = scalar_covariance_kkt_residual(
                variance,
                score,
                complementarity,
                variance_tolerance,
            );
            let term = self
                .covariance_parameter_context(theta_index)
                .map(|(_, source_syntax, _)| source_syntax)
                .unwrap_or_else(|| format!("(1 | {})", term.grouping_name));

            blocks.push(ScalarCovarianceKktBlock {
                term_index,
                theta_index,
                term,
                theta: theta_value,
                variance,
                score,
                complementarity,
                residual,
                classification,
            });
        }

        let residual = blocks
            .iter()
            .map(|block| block.residual)
            .fold(0.0, f64::max);

        Ok(ScalarCovarianceKktCertificate {
            blocks,
            residual,
            variance_tolerance,
            score_tolerance,
            objective,
        })
    }

    /// Evaluate the profiled objective at `theta` without mutating `self`.
    ///
    /// Clones a fresh evaluator per call, so it is a convenience for one-off
    /// probes (primarily tests); repeated-probe paths such as the KKT
    /// certificates share a single cloned evaluator instead.
    #[cfg(test)]
    pub(super) fn objective_at_theta_for_certificate(&self, theta: &[f64]) -> Result<f64> {
        let mut evaluator = self.clone();
        evaluator.objective_at(theta)
    }

    fn scalar_covariance_score(
        evaluator: &mut LinearMixedModel,
        theta_index: usize,
        theta: &[f64],
        objective: f64,
    ) -> Result<f64> {
        let variance = theta[theta_index].max(0.0).powi(2);
        let mut step = scalar_covariance_variance_step(variance);

        for _ in 0..8 {
            let plus =
                Self::objective_at_scalar_variance(evaluator, theta, theta_index, variance + step);
            if variance > 1.5 * step {
                let minus_variance = variance - step;
                if let (Ok(f_plus), Ok(f_minus)) = (
                    plus,
                    Self::objective_at_scalar_variance(
                        evaluator,
                        theta,
                        theta_index,
                        minus_variance,
                    ),
                ) {
                    if f_plus.is_finite() && f_minus.is_finite() {
                        return Ok((f_plus - f_minus) / (2.0 * step));
                    }
                }
            } else if let Ok(f_plus) = plus {
                if f_plus.is_finite() && objective.is_finite() {
                    return Ok((f_plus - objective) / step);
                }
            }
            step *= 0.25;
        }

        Err(MixedModelError::Optimization(format!(
            "failed to compute scalar covariance score for theta[{theta_index}]"
        )))
    }

    fn objective_at_scalar_variance(
        evaluator: &mut LinearMixedModel,
        theta: &[f64],
        theta_index: usize,
        variance: f64,
    ) -> Result<f64> {
        let mut trial = theta.to_vec();
        trial[theta_index] = variance.max(0.0).sqrt();
        evaluator.objective_at(&trial)
    }

    unstable_internal_method! {
    /// Post-fit covariance-cone KKT diagnostic for 2x2 random-effect terms.
    ///
    /// This certificate works in covariance space for full `(1 + x | group)`
    /// style blocks. It estimates directional derivatives `dF(G + t uu')/dt`
    /// through the existing profiled LMM objective and reconstructs the 2x2
    /// covariance score matrix. No dense marginal covariance matrix is formed.
    ///
    /// Unstable internal surface: `pub` only with the `unstable-internals`
    /// feature; otherwise `pub(crate)`.
    unstable_vis fn two_by_two_covariance_kkt_certificate(
        &self,
    ) -> Result<TwoByTwoCovarianceKktCertificate> {
        if !self.optsum.is_fitted() {
            return Err(MixedModelError::NotFitted);
        }
        if self.reterms.is_empty() {
            return Err(MixedModelError::NoRandomEffects);
        }
        if self
            .reterms
            .iter()
            .any(|term| term.vsize != 2 || term.n_theta() != 3)
        {
            return Err(MixedModelError::Unsupported(
                "2x2 covariance KKT certificate currently supports only full 2x2 random-effect terms"
                    .to_string(),
            ));
        }

        let theta = self.theta();
        let mut evaluator = self.clone();
        let objective = evaluator.objective_at(&theta)?;
        let covariance_tolerance = TWO_BY_TWO_KKT_COVARIANCE_TOLERANCE;
        let score_tolerance = (1e-5 * (1.0 + objective.abs())).max(1e-6);
        let complementarity_tolerance = 1e-4;
        let mut blocks = Vec::with_capacity(self.reterms.len());

        let mut theta_start_index = 0;
        for (term_index, term) in self.reterms.iter().enumerate() {
            let n_theta = term.n_theta();
            let theta_block = [
                theta[theta_start_index],
                theta[theta_start_index + 1],
                theta[theta_start_index + 2],
            ];
            let covariance = two_by_two_covariance_from_theta(theta_block);
            let score = Self::two_by_two_covariance_score(
                &mut evaluator,
                theta_start_index,
                &theta,
                objective,
                covariance,
            )?;
            let (min_eig_g, max_eig_g) = symmetric_2x2_eigenvalues(covariance);
            let (min_eig_score, _) = symmetric_2x2_eigenvalues(score);
            let complementarity = two_by_two_complementarity(covariance, score);
            let residual =
                two_by_two_covariance_kkt_residual(min_eig_g, min_eig_score, complementarity);
            let classification = classify_two_by_two_covariance_kkt(
                min_eig_g,
                max_eig_g,
                min_eig_score,
                two_by_two_frobenius_norm(score),
                complementarity,
                covariance_tolerance,
                score_tolerance,
                complementarity_tolerance,
            );
            let term = self
                .covariance_parameter_context(theta_start_index)
                .map(|(_, source_syntax, _)| source_syntax)
                .unwrap_or_else(|| format!("(2x2 | {})", term.grouping_name));

            blocks.push(TwoByTwoCovarianceKktBlock {
                term_index,
                theta_start_index,
                term,
                theta: theta_block,
                covariance,
                score,
                min_eig_g,
                min_eig_score,
                complementarity,
                residual,
                classification,
            });

            theta_start_index += n_theta;
        }

        let residual = blocks
            .iter()
            .map(|block| block.residual)
            .fold(0.0, f64::max);

        Ok(TwoByTwoCovarianceKktCertificate {
            blocks,
            residual,
            covariance_tolerance,
            score_tolerance,
            complementarity_tolerance,
            objective,
        })
    }
    }

    fn two_by_two_covariance_score(
        evaluator: &mut LinearMixedModel,
        theta_start_index: usize,
        theta: &[f64],
        objective: f64,
        covariance: [[f64; 2]; 2],
    ) -> Result<[[f64; 2]; 2]> {
        let e1 = [[1.0, 0.0], [0.0, 0.0]];
        let e2 = [[0.0, 0.0], [0.0, 1.0]];
        let plus = [[0.5, 0.5], [0.5, 0.5]];
        let minus = [[0.5, -0.5], [-0.5, 0.5]];

        let s00 = Self::two_by_two_directional_covariance_score(
            evaluator,
            theta_start_index,
            theta,
            objective,
            covariance,
            e1,
        )?;
        let s11 = Self::two_by_two_directional_covariance_score(
            evaluator,
            theta_start_index,
            theta,
            objective,
            covariance,
            e2,
        )?;
        let d_plus = Self::two_by_two_directional_covariance_score(
            evaluator,
            theta_start_index,
            theta,
            objective,
            covariance,
            plus,
        )?;
        let d_minus = Self::two_by_two_directional_covariance_score(
            evaluator,
            theta_start_index,
            theta,
            objective,
            covariance,
            minus,
        )?;

        let s01_from_plus = d_plus - 0.5 * (s00 + s11);
        let s01_from_minus = 0.5 * (s00 + s11) - d_minus;
        let s01 = 0.5 * (s01_from_plus + s01_from_minus);

        Ok([[s00, s01], [s01, s11]])
    }

    fn two_by_two_directional_covariance_score(
        evaluator: &mut LinearMixedModel,
        theta_start_index: usize,
        theta: &[f64],
        objective: f64,
        covariance: [[f64; 2]; 2],
        direction: [[f64; 2]; 2],
    ) -> Result<f64> {
        let mut step = two_by_two_covariance_step(covariance);

        for _ in 0..8 {
            let plus_cov = two_by_two_add_direction(covariance, direction, step);
            let plus = Self::objective_at_two_by_two_covariance(
                evaluator,
                theta,
                theta_start_index,
                plus_cov,
            );
            let minus_cov = two_by_two_add_direction(covariance, direction, -step);

            if two_by_two_theta_from_covariance(minus_cov).is_some() {
                if let (Ok(f_plus), Ok(f_minus)) = (
                    plus,
                    Self::objective_at_two_by_two_covariance(
                        evaluator,
                        theta,
                        theta_start_index,
                        minus_cov,
                    ),
                ) {
                    if f_plus.is_finite() && f_minus.is_finite() {
                        return Ok((f_plus - f_minus) / (2.0 * step));
                    }
                }
            } else if let Ok(f_plus) = plus {
                if f_plus.is_finite() && objective.is_finite() {
                    return Ok((f_plus - objective) / step);
                }
            }

            step *= 0.25;
        }

        Err(MixedModelError::Optimization(format!(
            "failed to compute 2x2 covariance score for theta block starting at {theta_start_index}"
        )))
    }

    fn objective_at_two_by_two_covariance(
        evaluator: &mut LinearMixedModel,
        theta: &[f64],
        theta_start_index: usize,
        covariance: [[f64; 2]; 2],
    ) -> Result<f64> {
        let theta_block = two_by_two_theta_from_covariance(covariance).ok_or_else(|| {
            MixedModelError::Optimization(
                "2x2 covariance perturbation is not positive semidefinite".to_string(),
            )
        })?;
        let mut trial = theta.to_vec();
        trial[theta_start_index..theta_start_index + 3].copy_from_slice(&theta_block);
        evaluator.objective_at(&trial)
    }

    pub(super) fn trust_bq_covariance_kkt_certifies_theta(
        &mut self,
        theta: &[f64],
        objective: f64,
        fevals: i64,
        reml: bool,
    ) -> Result<bool> {
        if !objective.is_finite() {
            return Ok(false);
        }
        let supported_scalar = self
            .reterms
            .iter()
            .all(|term| term.vsize == 1 && term.n_theta() == 1);
        let supported_two_by_two = self
            .reterms
            .iter()
            .all(|term| term.vsize == 2 && term.n_theta() == 3);
        if !supported_scalar && !supported_two_by_two {
            return Ok(false);
        }

        let previous_optsum = self.optsum.clone();
        let previous_theta = self.theta();
        let certified = (|| -> Result<bool> {
            self.set_theta(theta)?;
            self.optsum.reml = reml;
            self.optsum.optimizer = Optimizer::TrustBq;
            self.optsum.backend = Optimizer::TrustBq.canonical_backend();
            self.optsum.final_params = theta.to_vec();
            self.optsum.fmin = objective;
            self.optsum.feval = fevals.max(1);
            self.optsum.return_value = "FTOL_REACHED".to_string();

            if supported_scalar {
                let certificate = self.scalar_covariance_kkt_certificate()?;
                Ok(certificate.blocks.iter().all(|block| {
                    matches!(
                        block.classification,
                        CovarianceKktClassification::InteriorConverged
                            | CovarianceKktClassification::ValidZeroVariance
                    )
                }))
            } else {
                let certificate = self.two_by_two_covariance_kkt_certificate()?;
                Ok(certificate.blocks.iter().all(|block| {
                    matches!(
                        block.classification,
                        CovarianceKktClassification::InteriorConverged
                            | CovarianceKktClassification::ValidZeroVariance
                            | CovarianceKktClassification::ValidRankDeficientCovariance
                    )
                }))
            }
        })();

        let restore_result = self.set_theta(&previous_theta);
        self.optsum = previous_optsum;
        restore_result?;
        // The certificate is purely an early-stop accelerator. On degenerate
        // surfaces (e.g. a response constant within nested grouping levels)
        // the finite-difference score probes can fail outright; that must
        // read as "not certified, keep optimizing", not abort the fit.
        Ok(certified.unwrap_or(false))
    }

    pub(super) fn apply_kkt_guided_boundary_restart(&mut self, reml: bool) -> Result<bool> {
        // The KKT-guided restart is a best-effort post-fit improvement probe.
        // On degenerate surfaces (e.g. a response constant within nested
        // grouping levels) the certificate's finite-difference score probes
        // can fail to evaluate; that must skip the restart, not turn an
        // otherwise completed fit into an error.
        let Some(candidate) = self.kkt_boundary_restart_candidate().unwrap_or(None) else {
            return Ok(false);
        };

        let previous_optsum = self.optsum.clone();
        let previous_feval = previous_optsum.feval.max(0);
        let previous_max_feval = previous_optsum.max_feval.max(0);
        let previous_fit_log = previous_optsum.fit_log.clone();
        let optimizer = previous_optsum.optimizer;
        let n_theta = self.n_theta();

        self.optsum = previous_optsum;
        self.optsum.initial = candidate.theta.clone();
        self.optsum.final_params = candidate.theta.clone();
        self.optsum.finitial = candidate.objective;
        self.optsum.fmin = f64::INFINITY;
        self.optsum.feval = -1;
        self.optsum.fit_log.clear();
        if self.optsum.max_feval > 0 {
            self.optsum.max_feval = self
                .optsum
                .max_feval
                .max(if n_theta == 1 { 100 } else { 500 });
        }
        self.set_theta(&candidate.theta)?;
        self.update_l()?;

        match optimizer {
            Optimizer::PatternSearch => {
                if self.n_theta() == 1 {
                    self.fit_scalar_single_theta()?;
                } else {
                    self.fit_multivariate_pattern_search()?;
                }
            }
            Optimizer::Cobyla => {
                self.fit_cobyla(reml)?;
            }
            Optimizer::TrustBq => {
                self.fit_trust_bq_with_maxeval(reml, None)?;
            }
            Optimizer::NloptBobyqa => {
                #[cfg(feature = "nlopt")]
                self.fit_nlopt_small_theta(reml)?;
                #[cfg(not(feature = "nlopt"))]
                return Ok(false);
            }
            Optimizer::NloptNewuoa => {
                #[cfg(feature = "nlopt")]
                self.fit_nlopt_large_theta(reml)?;
                #[cfg(not(feature = "nlopt"))]
                return Ok(false);
            }
            Optimizer::PrimaBobyqa => {
                #[cfg(feature = "prima")]
                self.fit_prima_bobyqa_with_maxeval(reml, None)?;
                #[cfg(not(feature = "prima"))]
                return Ok(false);
            }
            Optimizer::PrimaCobyla | Optimizer::PrimaLincoa | Optimizer::PrimaNewuoa => {
                return Ok(false);
            }
        }

        let restart_return = self.optsum.return_value.clone();
        if previous_feval > 0 {
            self.optsum.feval += previous_feval;
        }
        if self.optsum.max_feval > 0 && previous_max_feval > 0 {
            self.optsum.max_feval += previous_max_feval;
        }
        if !previous_fit_log.is_empty() {
            let mut fit_log = previous_fit_log;
            fit_log.extend(self.optsum.fit_log.clone());
            self.optsum.fit_log = fit_log;
        }
        self.optsum.return_value = format!(
            "KKT_BOUNDARY_RESTART({}): {restart_return}",
            candidate.reason
        );

        Ok(true)
    }

    fn kkt_boundary_restart_candidate(&self) -> Result<Option<KktBoundaryRestartCandidate>> {
        if self.reterms.is_empty() || !self.optsum.is_fitted() {
            return Ok(None);
        }

        if self
            .reterms
            .iter()
            .all(|term| term.vsize == 1 && term.n_theta() == 1)
        {
            return self.scalar_kkt_boundary_restart_candidate();
        }

        if self
            .reterms
            .iter()
            .all(|term| term.vsize == 2 && term.n_theta() == 3)
        {
            return self.two_by_two_kkt_boundary_restart_candidate();
        }

        Ok(None)
    }

    fn scalar_kkt_boundary_restart_candidate(&self) -> Result<Option<KktBoundaryRestartCandidate>> {
        let base_theta = self.theta();
        // An InvalidBoundaryStop classification requires a block variance at
        // or below the certificate's variance tolerance, so a fit with every
        // scalar variance strictly interior can never produce a restart
        // candidate. Checking that in theta space skips the certificate's
        // finite-difference probes entirely on non-boundary fits.
        if base_theta
            .iter()
            .all(|&value| value.max(0.0).powi(2) > SCALAR_KKT_VARIANCE_TOLERANCE)
        {
            return Ok(None);
        }

        let certificate = self.scalar_covariance_kkt_certificate()?;
        let base_objective = certificate.objective;
        let mut best_theta = base_theta.clone();
        let mut best_objective = base_objective;
        let mut reason = None;
        let mut evaluator = self.clone();

        for block in certificate.blocks.iter().filter(|block| {
            block.classification == CovarianceKktClassification::InvalidBoundaryStop
        }) {
            let scale = 1.0 + block.variance.abs().max((-block.score).max(0.0));
            for delta in kkt_restart_delta_grid(scale) {
                let mut trial = base_theta.clone();
                trial[block.theta_index] = delta.sqrt();
                let objective = evaluator.objective_at(&trial)?;
                if objective + self.optsum.ftol_abs.max(1e-10) < best_objective {
                    best_objective = objective;
                    best_theta = trial;
                    reason = Some(format!("scalar theta[{}]", block.theta_index));
                }
            }
        }

        Ok(reason.map(|reason| KktBoundaryRestartCandidate {
            theta: best_theta,
            objective: best_objective,
            reason,
        }))
    }

    fn two_by_two_kkt_boundary_restart_candidate(
        &self,
    ) -> Result<Option<KktBoundaryRestartCandidate>> {
        let base_theta = self.theta();
        // An InvalidBoundaryStop classification requires a block covariance
        // whose smallest eigenvalue is at or below the certificate's
        // covariance tolerance, so a fit with every 2x2 block strictly inside
        // the PSD cone can never produce a restart candidate. Checking that
        // in theta space skips the certificate's finite-difference probes
        // entirely on non-boundary fits.
        let any_block_on_psd_boundary = base_theta.chunks_exact(3).any(|block| {
            let covariance = two_by_two_covariance_from_theta([block[0], block[1], block[2]]);
            symmetric_2x2_eigenvalues(covariance).0 <= TWO_BY_TWO_KKT_COVARIANCE_TOLERANCE
        });
        if !any_block_on_psd_boundary {
            return Ok(None);
        }

        let certificate = self.two_by_two_covariance_kkt_certificate()?;
        let base_objective = certificate.objective;
        let mut best_theta = base_theta.clone();
        let mut best_objective = base_objective;
        let mut reason = None;
        let mut evaluator = self.clone();

        for block in certificate.blocks.iter().filter(|block| {
            block.classification == CovarianceKktClassification::InvalidBoundaryStop
        }) {
            let direction = symmetric_2x2_min_eigenvector(block.score);
            let outer = [
                [direction[0] * direction[0], direction[0] * direction[1]],
                [direction[1] * direction[0], direction[1] * direction[1]],
            ];
            let scale = 1.0 + two_by_two_frobenius_norm(block.covariance);
            for delta in kkt_restart_delta_grid(scale) {
                let covariance = two_by_two_add_direction(block.covariance, outer, delta);
                let Some(theta_block) = two_by_two_theta_from_covariance(covariance) else {
                    continue;
                };
                let mut trial = base_theta.clone();
                trial[block.theta_start_index..block.theta_start_index + 3]
                    .copy_from_slice(&theta_block);
                let objective = evaluator.objective_at(&trial)?;
                if objective + self.optsum.ftol_abs.max(1e-10) < best_objective {
                    best_objective = objective;
                    best_theta = trial;
                    reason = Some(format!("2x2 block {}", block.term_index));
                }
            }
        }

        Ok(reason.map(|reason| KktBoundaryRestartCandidate {
            theta: best_theta,
            objective: best_objective,
            reason,
        }))
    }

    pub(super) fn theta_at_lower_bound(&self) -> bool {
        let theta = self.theta();
        let lb = self.lower_bounds();
        let boundary_tolerance = self.optsum.xtol_zero_abs.max(1e-12) * 10.0;
        theta.iter().zip(lb.iter()).any(|(&value, &lower)| {
            lower.is_finite() && (value - lower).abs() <= boundary_tolerance
        })
    }

    pub(super) fn optimizer_certificate_reports_boundary(&self) -> bool {
        self.compiler_artifact
            .optimizer_certificate
            .as_ref()
            .is_some_and(|certificate| certificate.evidence.parameter_space.n_boundary > 0)
    }

    pub(super) fn has_reduced_effective_covariance(&self) -> bool {
        self.compiler_artifact
            .effective_covariance
            .iter()
            .any(|summary| summary.status == EffectiveRankStatus::ReducedRank)
    }
    fn use_scalar_single_theta_optimizer(&self) -> bool {
        self.reterms.len() == 1 && self.reterms[0].vsize == 1 && self.n_theta() == 1
    }

    #[cfg(feature = "nlopt")]
    fn use_nlopt_bobyqa_small_theta_optimizer(&self) -> bool {
        // Smooth, low-dimensional problems benefit substantially from
        // BOBYQA's trust-region modelling vs. pattern_search (~3× fewer
        // evals on profiled kb07-class fits). Pattern search remains the
        // automatic fallback if BOBYQA fails to converge. Gated to the
        // `nlopt` feature; without it the auto-fit dispatch uses the native
        // scalar pattern-search or multi-theta TrustBQ paths.
        let n_theta = self.n_theta();
        n_theta > 1 && n_theta <= 6
    }

    #[cfg(feature = "nlopt")]
    fn use_large_single_vsize2_optimizer_tuning(&self) -> bool {
        self.reterms.len() == 1
            && self.reterms[0].vsize == 2
            && self.n_theta() == 3
            && self.reterms[0].n_ranef() >= 512
            && self.a_blocks.len() == 3
            && matches!(self.a_blocks[0], MatrixBlock::BlockDiagonal(_))
            && matches!(self.a_blocks[1], MatrixBlock::Dense(_))
            && matches!(self.a_blocks[2], MatrixBlock::Dense(_))
    }

    #[cfg(feature = "nlopt")]
    fn use_large_theta_nlopt_optimizer(&self) -> bool {
        self.n_theta() > 6
    }

    fn project_theta_to_bounds(theta: &mut [f64], lower_bounds: &[f64]) {
        for (value, &lower) in theta.iter_mut().zip(lower_bounds.iter()) {
            if lower.is_finite() && *value < lower {
                *value = lower;
            }
        }
    }

    fn steps_are_small(step: &[f64], step_tol: &[f64]) -> bool {
        step.iter()
            .zip(step_tol.iter())
            .all(|(&value, &tol)| value <= tol)
    }

    pub(super) fn apply_theta_to_reterms(reterms: &mut [ReMat], theta: &[f64]) -> Option<()> {
        let mut offset = 0;
        for rt in reterms {
            let nt = rt.n_theta();
            if offset + nt > theta.len() {
                return None;
            }
            rt.set_theta(&theta[offset..offset + nt]).ok()?;
            offset += nt;
        }
        (offset == theta.len()).then_some(())
    }

    fn profiled_objective_from_parts(
        a_blocks: &[MatrixBlock],
        l_blocks: &mut [MatrixBlock],
        reterms: &mut [ReMat],
        theta: &[f64],
        dims: ModelDims,
        is_reml: bool,
        fixed_sigma: Option<f64>,
        cholesky_zero_pad_tolerance: f64,
    ) -> Option<f64> {
        if let Some(obj) = Self::profiled_objective_one_vsize1_fast(
            a_blocks,
            reterms,
            theta,
            dims,
            is_reml,
            fixed_sigma,
            cholesky_zero_pad_tolerance,
        ) {
            return Some(obj);
        }

        if let Some(obj) = Self::profiled_objective_one_vsize2_fast(
            a_blocks,
            reterms,
            theta,
            dims,
            is_reml,
            fixed_sigma,
            cholesky_zero_pad_tolerance,
        ) {
            return Some(obj);
        }

        Self::apply_theta_to_reterms(reterms, theta)?;
        if update_l_from_parts(a_blocks, l_blocks, reterms, cholesky_zero_pad_tolerance).is_err() {
            return None;
        }

        let k = reterms.len();
        let n = dims.n as f64;
        let p = dims.p as f64;

        let mut logdet_lzz = 0.0;
        for j in 0..k {
            logdet_lzz += logdet_block(&l_blocks[block_index(j, j)]);
        }

        let l_last = l_blocks[block_index(k, k)].as_dense();
        let pp1 = l_last.nrows();
        let last_diag = l_last[(pp1 - 1, pp1 - 1)];
        let pwrss = last_diag * last_diag;

        let logdet = if is_reml {
            let mut logdet_lxx = 0.0;
            for i in 0..(pp1 - 1) {
                let d = l_last[(i, i)];
                if d > 0.0 {
                    logdet_lxx += d.ln();
                }
            }
            logdet_lzz + 2.0 * logdet_lxx
        } else {
            logdet_lzz
        };

        let denomdf = if is_reml { n - p } else { n };
        Some(Self::objective_from_components(
            logdet,
            pwrss,
            denomdf,
            fixed_sigma,
        ))
    }

    pub(super) fn cholesky_last_and_logdet_one_vsize1_fast(
        a_blocks: &[MatrixBlock],
        reterms: &[ReMat],
        theta: &[f64],
        cholesky_zero_pad_tolerance: f64,
    ) -> Option<(DMatrix<f64>, f64)> {
        if reterms.len() != 1 || reterms[0].vsize != 1 || theta.len() != 1 || a_blocks.len() != 3 {
            return None;
        }

        let MatrixBlock::Diagonal(a00_diag) = &a_blocks[0] else {
            return None;
        };
        let MatrixBlock::Dense(a10) = &a_blocks[1] else {
            return None;
        };
        let MatrixBlock::Dense(a11) = &a_blocks[2] else {
            return None;
        };

        if a00_diag.is_empty() {
            return None;
        }
        if a10.ncols() != a00_diag.len() || a11.nrows() != a11.ncols() || a11.nrows() != a10.nrows()
        {
            return None;
        }

        let pp1 = a11.nrows();
        let lambda = theta[0];
        let mut l_last = a11.clone();
        let mut logdet_lzz = 0.0;
        let mut solved_by_row = if pp1 == 3 { Vec::new() } else { vec![0.0; pp1] };

        for (level, &src_diag) in a00_diag.iter().enumerate() {
            let mut l00 = lambda * lambda * src_diag + 1.0;
            let pivot_tolerance =
                cholesky_zero_pad_abs_tolerance(l00.abs(), cholesky_zero_pad_tolerance);

            if l00 <= 0.0 {
                if l00 < -pivot_tolerance {
                    return None;
                }
                l00 = 0.0;
            } else {
                l00 = l00.sqrt();
            }

            if l00 > 0.0 {
                logdet_lzz += 2.0 * l00.ln();
            }

            if pp1 == 3 {
                let z0 = solve_scaled_vsize1_row(a10, 0, level, lambda, l00);
                let z1 = solve_scaled_vsize1_row(a10, 1, level, lambda, l00);
                let z2 = solve_scaled_vsize1_row(a10, 2, level, lambda, l00);

                l_last[(0, 0)] -= z0 * z0;
                l_last[(1, 0)] -= z1 * z0;
                l_last[(1, 1)] -= z1 * z1;
                l_last[(2, 0)] -= z2 * z0;
                l_last[(2, 1)] -= z2 * z1;
                l_last[(2, 2)] -= z2 * z2;
            } else {
                for row in 0..pp1 {
                    solved_by_row[row] = solve_scaled_vsize1_row(a10, row, level, lambda, l00);
                }
                for row in 0..pp1 {
                    for col in 0..=row {
                        l_last[(row, col)] -= solved_by_row[row] * solved_by_row[col];
                    }
                }
            }
        }

        let mut l_last_block = MatrixBlock::Dense(l_last);
        if cholesky_block_with_tolerance(&mut l_last_block, cholesky_zero_pad_tolerance).is_err() {
            return None;
        }
        let MatrixBlock::Dense(l_last) = l_last_block else {
            unreachable!();
        };
        Some((l_last, logdet_lzz))
    }

    pub(super) fn profiled_objective_one_vsize1_fast(
        a_blocks: &[MatrixBlock],
        reterms: &[ReMat],
        theta: &[f64],
        dims: ModelDims,
        is_reml: bool,
        fixed_sigma: Option<f64>,
        cholesky_zero_pad_tolerance: f64,
    ) -> Option<f64> {
        let (l_last, logdet_lzz) = Self::cholesky_last_and_logdet_one_vsize1_fast(
            a_blocks,
            reterms,
            theta,
            cholesky_zero_pad_tolerance,
        )?;
        let pp1 = l_last.nrows();

        let last_diag = l_last[(pp1 - 1, pp1 - 1)];
        let pwrss = last_diag * last_diag;
        let logdet = if is_reml {
            let mut logdet_lxx = 0.0;
            for i in 0..(pp1 - 1) {
                let d = l_last[(i, i)];
                if d > 0.0 {
                    logdet_lxx += d.ln();
                }
            }
            logdet_lzz + 2.0 * logdet_lxx
        } else {
            logdet_lzz
        };

        let n = dims.n as f64;
        let p = dims.p as f64;
        let denomdf = if is_reml { n - p } else { n };
        Some(Self::objective_from_components(
            logdet,
            pwrss,
            denomdf,
            fixed_sigma,
        ))
    }

    pub(super) fn profiled_objective_one_vsize2_fast(
        a_blocks: &[MatrixBlock],
        reterms: &[ReMat],
        theta: &[f64],
        dims: ModelDims,
        is_reml: bool,
        fixed_sigma: Option<f64>,
        cholesky_zero_pad_tolerance: f64,
    ) -> Option<f64> {
        if reterms.len() != 1 || reterms[0].vsize != 2 || theta.len() != 3 || a_blocks.len() != 3 {
            return None;
        }

        let MatrixBlock::BlockDiagonal(a00_blocks) = &a_blocks[0] else {
            return None;
        };
        let MatrixBlock::Dense(a10) = &a_blocks[1] else {
            return None;
        };
        let MatrixBlock::Dense(a11) = &a_blocks[2] else {
            return None;
        };

        if a00_blocks.is_empty()
            || !a00_blocks
                .iter()
                .all(|block| block.nrows() == 2 && block.ncols() == 2)
        {
            return None;
        }
        if a10.ncols() != 2 * a00_blocks.len()
            || a10.ncols() < 512
            || a11.nrows() != a11.ncols()
            || a11.nrows() != a10.nrows()
        {
            return None;
        }

        let pp1 = a11.nrows();
        let lam00 = theta[0];
        let lam10 = theta[1];
        let lam11 = theta[2];
        let mut l_last = a11.clone();
        let mut logdet_lzz = 0.0;
        let mut solved0_by_row = if pp1 == 3 { Vec::new() } else { vec![0.0; pp1] };
        let mut solved1_by_row = if pp1 == 3 { Vec::new() } else { vec![0.0; pp1] };

        for (level, src_blk) in a00_blocks.iter().enumerate() {
            let s00 = src_blk[(0, 0)];
            let s01 = src_blk[(0, 1)];
            let s10 = src_blk[(1, 0)];
            let s11 = src_blk[(1, 1)];

            let t00 = s00 * lam00 + s01 * lam10;
            let t10 = s10 * lam00 + s11 * lam10;
            let t11 = s11 * lam11;

            let mut l00 = lam00 * t00 + lam10 * t10 + 1.0;
            let mut l10 = lam11 * t10;
            let mut l11 = lam11 * t11 + 1.0;
            let pivot_tolerance = cholesky_zero_pad_abs_tolerance(
                l00.abs().max(l11.abs()),
                cholesky_zero_pad_tolerance,
            );

            if l00 <= 0.0 {
                if l00 < -pivot_tolerance {
                    return None;
                }
                l00 = 0.0;
                l10 = 0.0;
            } else {
                l00 = l00.sqrt();
                l10 /= l00;
            }

            l11 -= l10 * l10;
            if l11 <= 0.0 {
                if l11 < -pivot_tolerance {
                    return None;
                }
                l11 = 0.0;
            } else {
                l11 = l11.sqrt();
            }

            if l00 > 0.0 {
                logdet_lzz += l00.ln();
            }
            if l11 > 0.0 {
                logdet_lzz += l11.ln();
            }

            let col0 = 2 * level;
            let col1 = col0 + 1;
            if pp1 == 3 {
                let (z00, z01) =
                    solve_scaled_vsize2_row(a10, 0, col0, col1, lam00, lam10, lam11, l00, l10, l11);
                let (z10, z11) =
                    solve_scaled_vsize2_row(a10, 1, col0, col1, lam00, lam10, lam11, l00, l10, l11);
                let (z20, z21) =
                    solve_scaled_vsize2_row(a10, 2, col0, col1, lam00, lam10, lam11, l00, l10, l11);

                l_last[(0, 0)] -= z00 * z00 + z01 * z01;
                l_last[(1, 0)] -= z10 * z00 + z11 * z01;
                l_last[(1, 1)] -= z10 * z10 + z11 * z11;
                l_last[(2, 0)] -= z20 * z00 + z21 * z01;
                l_last[(2, 1)] -= z20 * z10 + z21 * z11;
                l_last[(2, 2)] -= z20 * z20 + z21 * z21;
            } else {
                for row in 0..pp1 {
                    let (solved0, solved1) = solve_scaled_vsize2_row(
                        a10, row, col0, col1, lam00, lam10, lam11, l00, l10, l11,
                    );
                    solved0_by_row[row] = solved0;
                    solved1_by_row[row] = solved1;
                }

                for row in 0..pp1 {
                    for col in 0..=row {
                        l_last[(row, col)] -= solved0_by_row[row] * solved0_by_row[col]
                            + solved1_by_row[row] * solved1_by_row[col];
                    }
                }
            }
        }
        logdet_lzz *= 2.0;

        let mut l_last_block = MatrixBlock::Dense(l_last);
        if cholesky_block_with_tolerance(&mut l_last_block, cholesky_zero_pad_tolerance).is_err() {
            return None;
        }
        let MatrixBlock::Dense(l_last) = l_last_block else {
            unreachable!();
        };

        let last_diag = l_last[(pp1 - 1, pp1 - 1)];
        let pwrss = last_diag * last_diag;
        let logdet = if is_reml {
            let mut logdet_lxx = 0.0;
            for i in 0..(pp1 - 1) {
                let d = l_last[(i, i)];
                if d > 0.0 {
                    logdet_lxx += d.ln();
                }
            }
            logdet_lzz + 2.0 * logdet_lxx
        } else {
            logdet_lzz
        };

        let n = dims.n as f64;
        let p = dims.p as f64;
        let denomdf = if is_reml { n - p } else { n };
        Some(Self::objective_from_components(
            logdet,
            pwrss,
            denomdf,
            fixed_sigma,
        ))
    }

    #[cfg(feature = "nlopt")]
    fn nlopt_ok(
        result: std::result::Result<nlopt::SuccessState, NloptFailState>,
        action: &str,
    ) -> Result<()> {
        result.map(|_| ()).map_err(|status| {
            MixedModelError::Optimization(format!("NLopt {action} failed: {status:?}"))
        })
    }

    #[cfg(feature = "nlopt")]
    pub(super) fn nlopt_status_label(name: &str) -> String {
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

    pub(super) fn cobyla_success_status_label(status: cobyla::SuccessStatus) -> String {
        match status {
            cobyla::SuccessStatus::Success => "SUCCESS".to_string(),
            cobyla::SuccessStatus::StopValReached => "STOPVAL_REACHED".to_string(),
            cobyla::SuccessStatus::FtolReached => "FTOL_REACHED".to_string(),
            cobyla::SuccessStatus::XtolReached => "XTOL_REACHED".to_string(),
            cobyla::SuccessStatus::MaxEvalReached => "MAXEVAL_REACHED".to_string(),
            cobyla::SuccessStatus::MaxTimeReached => "MAXTIME_REACHED".to_string(),
        }
    }

    pub(super) fn cobyla_fail_status_label(status: cobyla::FailStatus) -> String {
        match status {
            cobyla::FailStatus::Failure => "FAILURE".to_string(),
            cobyla::FailStatus::InvalidArgs => "INVALID_ARGS".to_string(),
            cobyla::FailStatus::OutOfMemory => "OUT_OF_MEMORY".to_string(),
            cobyla::FailStatus::RoundoffLimited => "ROUNDOFF_LIMITED".to_string(),
            cobyla::FailStatus::ForcedStop => "FORCED_STOP".to_string(),
            cobyla::FailStatus::UnexpectedError => "UNEXPECTED_ERROR".to_string(),
        }
    }

    pub(super) fn trust_bq_status_label(status: TrustBqStopReason) -> String {
        match status {
            TrustBqStopReason::RadiusBelowTolerance => "RADIUS_REACHED".to_string(),
            TrustBqStopReason::ObjectiveTolerance => "FTOL_REACHED".to_string(),
            TrustBqStopReason::MaxEvaluations => "MAXEVAL_REACHED".to_string(),
            TrustBqStopReason::StepBelowTolerance => "XTOL_REACHED".to_string(),
            TrustBqStopReason::ObjectiveStagnation => "FTOL_REACHED".to_string(),
            TrustBqStopReason::CertifiedConvergence => "FTOL_REACHED".to_string(),
        }
    }

    fn record_scalar_eval(
        &mut self,
        theta: f64,
        feval_count: &mut i64,
        fit_log: &mut Vec<FitLogEntry>,
        best_theta: &mut f64,
        best_fmin: &mut f64,
    ) -> Result<f64> {
        let obj = self.objective_at_fast_or_generic(&[theta])?;
        *feval_count += 1;
        fit_log.push(FitLogEntry {
            theta: vec![theta],
            objective: obj,
        });
        if obj < *best_fmin {
            *best_fmin = obj;
            *best_theta = theta;
        }
        Ok(obj)
    }

    pub(super) fn finalize_fit_result(
        &mut self,
        mut best_theta_val: Vec<f64>,
        mut best_fmin_val: f64,
        feval_count: i64,
        fit_log: Vec<FitLogEntry>,
        optimizer: Optimizer,
        return_value: Option<String>,
    ) -> Result<&mut Self> {
        Self::rectify_theta_columns(&mut best_theta_val, &self.parmap, self.reterms.len());
        self.set_theta(&best_theta_val)?;
        self.update_l()?;

        let mut xmin = best_theta_val.clone();
        let mut modified = false;
        for (i, (_, row, col)) in self.parmap.iter().enumerate() {
            if row == col && xmin[i] > 0.0 && xmin[i] < self.optsum.xtol_zero_abs {
                xmin[i] = 0.0;
                modified = true;
            }
        }
        if modified {
            let zero_obj = self.objective_at(&xmin)?;
            if zero_obj <= best_fmin_val + self.optsum.ftol_zero_abs {
                best_fmin_val = zero_obj;
                best_theta_val = xmin;
            } else {
                self.set_theta(&best_theta_val)?;
                self.update_l()?;
            }
        }

        self.optsum.optimizer = optimizer;
        self.optsum.backend = optimizer.canonical_backend();
        self.optsum.final_params = best_theta_val;
        self.optsum.fmin = best_fmin_val;
        self.optsum.feval = feval_count;
        self.optsum.return_value = return_value.unwrap_or_else(|| "SUCCESS".to_string());
        self.optsum.fit_log = fit_log;
        self.optsum.final_trust_radius = None;

        Ok(self)
    }

    pub(crate) fn rectify_theta_columns(
        theta: &mut [f64],
        parmap: &[(usize, usize, usize)],
        n_terms: usize,
    ) {
        for block in 0..n_terms {
            let max_col = parmap
                .iter()
                .filter(|&&(term, _, _)| term == block)
                .map(|&(_, _, col)| col)
                .max();

            let Some(max_col) = max_col else {
                continue;
            };

            for col in 0..=max_col {
                let diag_idx = parmap.iter().position(|&(term, row, col_idx)| {
                    term == block && row == col && col_idx == col
                });
                let Some(diag_idx) = diag_idx else {
                    continue;
                };

                if theta[diag_idx] < 0.0 {
                    for (idx, &(term, _, col_idx)) in parmap.iter().enumerate() {
                        if term == block && col_idx == col {
                            theta[idx] = -theta[idx];
                        }
                    }
                }
            }
        }
    }

    fn fit_scalar_single_theta(&mut self) -> Result<&mut Self> {
        const INVPHI: f64 = 0.6180339887498949;

        let maxeval = if self.optsum.max_feval > 0 {
            self.optsum.max_feval
        } else {
            10000
        };
        let xtol = self
            .optsum
            .xtol_abs
            .first()
            .copied()
            .unwrap_or(1e-8)
            .max(1e-4);
        let mut step = self
            .optsum
            .initial_step
            .first()
            .copied()
            .unwrap_or(0.75)
            .abs()
            .max(1e-6);
        let theta0 = self.optsum.initial[0].max(0.0);

        let mut feval_count = 0i64;
        let mut fit_log = Vec::new();
        let mut best_theta = theta0;
        let mut best_fmin = self.optsum.finitial;

        let mut lo = 0.0;
        let flo = if theta0 > 0.0 {
            self.record_scalar_eval(
                0.0,
                &mut feval_count,
                &mut fit_log,
                &mut best_theta,
                &mut best_fmin,
            )?
        } else {
            self.optsum.finitial
        };

        let mut mid = if theta0 > 0.0 { theta0 } else { step };
        let mut fmid = if theta0 > 0.0 {
            self.optsum.finitial
        } else {
            self.record_scalar_eval(
                mid,
                &mut feval_count,
                &mut fit_log,
                &mut best_theta,
                &mut best_fmin,
            )?
        };

        let mut hi = if fmid >= flo { mid } else { mid + step };

        if fmid < flo {
            let mut fhi = self.record_scalar_eval(
                hi,
                &mut feval_count,
                &mut fit_log,
                &mut best_theta,
                &mut best_fmin,
            )?;

            while feval_count < maxeval && fhi < fmid {
                lo = mid;
                mid = hi;
                fmid = fhi;
                step *= 2.0;
                hi = mid + step;
                fhi = self.record_scalar_eval(
                    hi,
                    &mut feval_count,
                    &mut fit_log,
                    &mut best_theta,
                    &mut best_fmin,
                )?;
            }
        }

        let mut a = lo;
        let mut b = hi.max(mid).max(step);
        if b <= a {
            b = a + step;
        }

        let mut c = b - (b - a) * INVPHI;
        let mut d = a + (b - a) * INVPHI;
        let mut fc = if (c - theta0).abs() <= xtol {
            self.optsum.finitial
        } else {
            self.record_scalar_eval(
                c,
                &mut feval_count,
                &mut fit_log,
                &mut best_theta,
                &mut best_fmin,
            )?
        };
        let mut fd = if (d - theta0).abs() <= xtol {
            self.optsum.finitial
        } else if (d - c).abs() <= xtol {
            fc
        } else {
            self.record_scalar_eval(
                d,
                &mut feval_count,
                &mut fit_log,
                &mut best_theta,
                &mut best_fmin,
            )?
        };

        while feval_count < maxeval && (b - a) > xtol * (1.0 + a.abs().max(b.abs())) {
            if fc <= fd {
                b = d;
                d = c;
                fd = fc;
                c = b - (b - a) * INVPHI;
                fc = self.record_scalar_eval(
                    c,
                    &mut feval_count,
                    &mut fit_log,
                    &mut best_theta,
                    &mut best_fmin,
                )?;
            } else {
                a = c;
                c = d;
                fc = fd;
                d = a + (b - a) * INVPHI;
                fd = self.record_scalar_eval(
                    d,
                    &mut feval_count,
                    &mut fit_log,
                    &mut best_theta,
                    &mut best_fmin,
                )?;
            }
        }

        self.finalize_fit_result(
            vec![best_theta],
            best_fmin,
            feval_count,
            fit_log,
            Optimizer::PatternSearch,
            (feval_count >= maxeval).then(|| "MAXEVAL_REACHED".to_string()),
        )
    }

    fn fit_multivariate_pattern_search(&mut self) -> Result<&mut Self> {
        let n_theta = self.n_theta();
        let maxeval = if self.optsum.max_feval > 0 {
            self.optsum.max_feval
        } else {
            10000
        };
        let lower_bounds = self.lower_bounds();
        let mut step_tol: Vec<f64> = self
            .optsum
            .xtol_abs
            .iter()
            .map(|&tol| tol.max(1e-5))
            .collect();
        if step_tol.len() != n_theta {
            step_tol = vec![1e-5; n_theta];
        }

        let mut step: Vec<f64> = self
            .optsum
            .initial_step
            .iter()
            .zip(step_tol.iter())
            .map(|(&initial, &tol)| initial.abs().max(tol))
            .collect();
        if step.len() != n_theta {
            step = vec![0.5; n_theta];
        }

        let outcome = Self::run_multivariate_pattern_search(
            self.optsum.initial.clone(),
            self.optsum.finitial,
            &lower_bounds,
            step,
            &step_tol,
            maxeval,
            self.optsum.ftol_abs,
            |theta| self.objective_at(theta),
        )?;

        self.finalize_fit_result(
            outcome.best_theta,
            outcome.best_fmin,
            outcome.feval_count,
            outcome.fit_log,
            Optimizer::PatternSearch,
            (outcome.feval_count >= maxeval).then(|| "MAXEVAL_REACHED".to_string()),
        )
    }

    pub(crate) fn run_multivariate_pattern_search<F>(
        initial: Vec<f64>,
        finitial: f64,
        lower_bounds: &[f64],
        mut step: Vec<f64>,
        step_tol: &[f64],
        maxeval: i64,
        ftol_abs: f64,
        mut objective: F,
    ) -> Result<PatternSearchOutcome>
    where
        F: FnMut(&[f64]) -> Result<f64>,
    {
        let n_theta = initial.len();
        let mut preferred_sign = vec![-1.0; n_theta];
        for (i, &lower) in lower_bounds.iter().enumerate() {
            if !lower.is_finite() {
                preferred_sign[i] = 1.0;
            }
        }

        let mut theta = initial;
        let mut ftheta = finitial;
        let mut best_theta = theta.clone();
        let mut best_fmin = ftheta;
        let mut feval_count = 0i64;
        let mut fit_log = Vec::new();

        while feval_count < maxeval && !Self::steps_are_small(&step, step_tol) {
            let base_theta = theta.clone();
            let base_f = ftheta;
            let mut moved = vec![false; n_theta];
            let mut exploratory_direction = vec![0.0; n_theta];

            for i in 0..n_theta {
                let mut chosen_theta = theta[i];
                let mut chosen_f = ftheta;
                let mut chosen_sign = 0.0;
                exploratory_direction[i] = preferred_sign[i];

                for dir in [preferred_sign[i], -preferred_sign[i]] {
                    let mut trial = theta.clone();
                    trial[i] += dir * step[i];
                    Self::project_theta_to_bounds(&mut trial, lower_bounds);
                    if (trial[i] - theta[i]).abs() <= step_tol[i] * 0.5 {
                        continue;
                    }

                    let ftrial = record_pattern_eval(
                        &mut objective,
                        &trial,
                        &mut feval_count,
                        &mut fit_log,
                        &mut best_theta,
                        &mut best_fmin,
                    )?;
                    if ftrial + ftol_abs < ftheta {
                        chosen_theta = trial[i];
                        chosen_f = ftrial;
                        chosen_sign = dir;
                        break;
                    }
                    if feval_count >= maxeval {
                        break;
                    }
                }

                if chosen_f < ftheta {
                    theta[i] = chosen_theta;
                    ftheta = chosen_f;
                    moved[i] = true;
                    preferred_sign[i] = chosen_sign;
                } else {
                    preferred_sign[i] = -preferred_sign[i];
                }

                if feval_count >= maxeval {
                    break;
                }
            }

            let mut any_moved = moved.iter().any(|&m| m);
            if feval_count < maxeval {
                let mut pattern_candidates = Vec::with_capacity(if any_moved { 1 } else { 2 });
                if any_moved {
                    let mut pattern = theta.clone();
                    for i in 0..n_theta {
                        pattern[i] += theta[i] - base_theta[i];
                    }
                    Self::project_theta_to_bounds(&mut pattern, lower_bounds);
                    pattern_candidates.push(pattern);
                } else {
                    let mut push_candidate = |pattern: Vec<f64>| {
                        if pattern != theta && !pattern_candidates.contains(&pattern) {
                            pattern_candidates.push(pattern);
                        }
                    };

                    for direction_sign in [1.0, -1.0] {
                        let mut pattern = base_theta.clone();
                        for i in 0..n_theta {
                            pattern[i] += direction_sign * exploratory_direction[i] * step[i];
                        }
                        Self::project_theta_to_bounds(&mut pattern, lower_bounds);
                        push_candidate(pattern);
                    }

                    for i in 0..n_theta {
                        for j in (i + 1)..n_theta {
                            for dir_i in [exploratory_direction[i], -exploratory_direction[i]] {
                                for dir_j in [exploratory_direction[j], -exploratory_direction[j]] {
                                    let mut pattern = base_theta.clone();
                                    pattern[i] += dir_i * step[i];
                                    pattern[j] += dir_j * step[j];
                                    Self::project_theta_to_bounds(&mut pattern, lower_bounds);
                                    push_candidate(pattern);
                                }
                            }
                        }
                    }
                }

                for pattern in pattern_candidates {
                    if feval_count >= maxeval {
                        break;
                    }
                    let fpattern = record_pattern_eval(
                        &mut objective,
                        &pattern,
                        &mut feval_count,
                        &mut fit_log,
                        &mut best_theta,
                        &mut best_fmin,
                    )?;
                    if fpattern + ftol_abs < ftheta {
                        for i in 0..n_theta {
                            if (pattern[i] - theta[i]).abs() > 0.0 {
                                preferred_sign[i] = (pattern[i] - theta[i]).signum();
                                moved[i] = true;
                            }
                        }
                        theta = pattern;
                        ftheta = fpattern;
                        any_moved = true;
                        break;
                    }
                }
            }

            if !any_moved {
                for value in &mut step {
                    *value *= 0.5;
                }
                continue;
            }

            for i in 0..n_theta {
                if moved[i] {
                    step[i] = (step[i] * 1.1).max(step_tol[i]);
                } else {
                    step[i] *= 0.5;
                }
            }

            if (base_f - ftheta).abs() <= ftol_abs && Self::steps_are_small(&step, step_tol) {
                break;
            }
        }

        #[cfg(test)]
        let exit_reason = if feval_count >= maxeval {
            "maxeval"
        } else if Self::steps_are_small(&step, step_tol) {
            "step_tolerance"
        } else {
            "ftol_or_no_progress"
        };

        Ok(PatternSearchOutcome {
            best_theta,
            best_fmin,
            feval_count,
            fit_log,
            #[cfg(test)]
            trace_label: None,
            #[cfg(test)]
            active_rank: None,
            #[cfg(test)]
            inactive_directions: None,
            #[cfg(test)]
            exit_reason: exit_reason.to_string(),
        })
    }

    fn fit_trust_bq_with_maxeval(
        &mut self,
        reml: bool,
        maxeval_override: Option<usize>,
    ) -> Result<&mut Self> {
        self.optsum.optimizer = Optimizer::TrustBq;
        self.optsum.backend = Optimizer::TrustBq.canonical_backend();

        let a_blocks = self.a_blocks.clone();
        let l_blocks_template = self.l_blocks.clone();
        let reterms_template = self.reterms.clone();
        let dims = self.dims;
        let is_reml = reml;
        let fixed_sigma = self.optsum.sigma;
        let sample_reuse_control = self.trust_bq_sample_reuse;
        let cholesky_zero_pad_tolerance = self
            .compiler_policy()
            .thresholds
            .cholesky_zero_pad_tolerance;
        let invalid_objective =
            self.optsum.finitial.abs().max(1.0) + 1.0e6 * (1.0 + self.optsum.finitial.abs());
        let best_theta = std::cell::RefCell::new(self.optsum.initial.clone());
        let best_fmin = std::cell::Cell::new(self.optsum.finitial);
        let fit_log: std::cell::RefCell<Vec<FitLogEntry>> = std::cell::RefCell::new(Vec::new());

        let reterms_work = std::cell::RefCell::new(reterms_template.clone());
        let l_blocks_work = std::cell::RefCell::new(l_blocks_template);

        let mut objective_fn = |theta: &[f64]| -> Result<f64> {
            let obj = {
                let mut rw = reterms_work.borrow_mut();
                let mut lw = l_blocks_work.borrow_mut();
                Self::profiled_objective_from_parts(
                    &a_blocks,
                    &mut lw,
                    &mut rw,
                    theta,
                    dims,
                    is_reml,
                    fixed_sigma,
                    cholesky_zero_pad_tolerance,
                )
                .unwrap_or(invalid_objective)
            };
            // TrustBQ requires every objective value to be finite, unlike the
            // NLopt/COBYLA backends which tolerate ±inf trial values. A
            // degenerate theta (e.g. a response constant within nested
            // grouping levels driving the profiled deviance to -inf/NaN) is
            // mapped to the same finite penalty as a hard evaluation error so
            // the trust region steps away from it instead of aborting the
            // fit; it also keeps the best-theta tracker below from latching
            // onto a non-finite "optimum".
            let obj = if obj.is_finite() {
                obj
            } else {
                invalid_objective
            };

            fit_log.borrow_mut().push(FitLogEntry {
                theta: theta.to_vec(),
                objective: obj,
            });
            if obj + 1e-12 < best_fmin.get() {
                best_fmin.set(obj);
                *best_theta.borrow_mut() = theta.to_vec();
            }

            Ok(obj)
        };

        let n_theta = self.n_theta();
        let policy = trust_bq_model_family_policy(
            n_theta,
            maxeval_override,
            &self.optsum.initial_step,
            &self.optsum.xtol_abs,
            self.optsum.max_feval,
            self.optsum.ftol_abs,
            self.optsum.ftol_rel,
        );
        let resolve_reuse_samples =
            |family_policy_reuse: bool| sample_reuse_control.resolve(family_policy_reuse);
        let mut trust_bq_initial = self.optsum.initial.clone();
        let lower_bounds = self.lower_bounds();
        let upper_bounds = vec![f64::INFINITY; n_theta];

        // Opt-in diagonal-first warm-start ladder: optimize the
        // zero-correlation covariance (off-diagonal theta pinned at exactly
        // zero) on a coarse budget, then hand the expanded optimum to the
        // full-covariance stage below. The two stages share one evaluation
        // budget and one fit log; the full stage still runs to its own
        // convergence/certificate stop, so boundary diagnostics and KKT
        // certificates are computed at the final full-covariance optimum
        // exactly as in the single-start path.
        let mut ladder_fevals = 0usize;
        let mut ladder_label: Option<String> = None;
        if self.trust_bq_start_ladder == TrustBqStartLadder::DiagonalFirst {
            let diagonal_indices: Vec<usize> = self
                .parmap
                .iter()
                .enumerate()
                .filter(|(_, (_, row, col))| row == col)
                .map(|(index, _)| index)
                .collect();
            if !diagonal_indices.is_empty() && diagonal_indices.len() < n_theta {
                let reduced_initial: Vec<f64> = diagonal_indices
                    .iter()
                    .map(|&index| trust_bq_initial[index])
                    .collect();
                let reduced_lower = vec![0.0_f64; diagonal_indices.len()];
                let reduced_upper = vec![f64::INFINITY; diagonal_indices.len()];
                let reduced_step: Vec<f64> = diagonal_indices
                    .iter()
                    .map(|&index| self.optsum.initial_step.get(index).copied().unwrap_or(0.75))
                    .collect();
                let reduced_xtol: Vec<f64> = diagonal_indices
                    .iter()
                    .map(|&index| self.optsum.xtol_abs.get(index).copied().unwrap_or(1e-10))
                    .collect();
                // A warm start needs the right neighborhood, not a certified
                // optimum. Deliberately bypass the family policy's tolerance
                // mapping (the small family clamps ftol to parity-grade
                // bands, which makes the stage polish its constrained
                // optimum and burn the shared budget): stop on a coarse
                // accepted-step band and a short stall window instead.
                let stage_budget = (policy.max_evaluations / 8).clamp(20, 60);
                let mut expanded = trust_bq_initial.clone();
                for (index, value) in expanded.iter_mut().enumerate() {
                    if !diagonal_indices.contains(&index) {
                        *value = 0.0;
                    }
                }
                let stage_result = {
                    let progress_callback = self.progress_callback.clone();
                    let mut last_progress = 0usize;
                    let mut stage_objective = |reduced: &[f64]| -> Result<f64> {
                        let mut full = expanded.clone();
                        for (slot, &index) in diagonal_indices.iter().enumerate() {
                            full[index] = reduced[slot];
                        }
                        objective_fn(&full)
                    };
                    minimize_trust_bq_with_progress(
                        &reduced_initial,
                        &reduced_lower,
                        &reduced_upper,
                        TrustBqOptions {
                            initial_radius: trust_bq_initial_radius(
                                &reduced_step,
                                diagonal_indices.len(),
                            ),
                            final_radius: trust_bq_final_radius(
                                &reduced_xtol,
                                diagonal_indices.len(),
                            )
                            .max(1e-3),
                            max_evaluations: stage_budget,
                            ftol_abs: 1e-4,
                            ftol_rel: 1e-6,
                            max_cross_terms: if diagonal_indices.len() <= 3 {
                                usize::MAX
                            } else {
                                0
                            },
                            reuse_samples: resolve_reuse_samples(diagonal_indices.len() >= 7),
                            stall_iterations: 3,
                            stall_ftol_rel: 1e-6,
                            stall_ftol_abs: 1e-8,
                            stall_requires_stable_x: false,
                            ..TrustBqOptions::default()
                        },
                        &mut stage_objective,
                        |progress| {
                            if let Some(callback) = &progress_callback {
                                callback.report_if_due(
                                    FitProgressPhase::LmmOptimizer,
                                    progress.fevals,
                                    Some(stage_budget),
                                    &mut last_progress,
                                )?;
                            }
                            Ok(false)
                        },
                    )
                };
                if let Ok(stage_result) = stage_result {
                    ladder_fevals = stage_result.fevals;
                    if stage_result.fmin.is_finite() {
                        for (slot, &index) in diagonal_indices.iter().enumerate() {
                            expanded[index] = stage_result.x[slot];
                        }
                        trust_bq_initial = expanded;
                        ladder_label = Some(format!("diagonal_first:{ladder_fevals} evals"));
                    }
                }
            }
        }
        // The full stage keeps the family's whole evaluation budget: the
        // coarse warm-start stage is bounded overhead (opted into by the
        // caller), and starving the full stage below the family budget was
        // observed to trade certified FTOL stops for budget exhaustion on
        // crossed rows.
        let full_stage_max_evaluations = policy.max_evaluations;

        let mut certificate_stop = TrustBqCertificateStopState::new(
            n_theta,
            full_stage_max_evaluations,
            policy.certificate_ftol_abs,
            policy.certificate_ftol_rel,
        );
        let progress_callback = self.progress_callback.clone();
        let mut last_progress = 0usize;
        let mut certificate_progress = |progress: &TrustBqProgress<'_>| -> Result<bool> {
            if let Some(callback) = &progress_callback {
                callback.report_if_due(
                    FitProgressPhase::LmmOptimizer,
                    progress.fevals,
                    Some(full_stage_max_evaluations),
                    &mut last_progress,
                )?;
            }
            if !certificate_stop.should_check(progress) {
                return Ok(false);
            }
            self.trust_bq_covariance_kkt_certifies_theta(
                progress.x,
                progress.fmin,
                progress.fevals as i64,
                reml,
            )
        };
        let result = minimize_trust_bq_with_progress(
            &trust_bq_initial,
            &lower_bounds,
            &upper_bounds,
            TrustBqOptions {
                // From a ladder warm start the optimum is expected nearby, so
                // begin with a contracted trust region (it re-expands on
                // successful steps); a cold start keeps the policy radius.
                initial_radius: if ladder_label.is_some() {
                    (policy.initial_radius / 8.0).max(policy.final_radius * 10.0)
                } else {
                    policy.initial_radius
                },
                final_radius: policy.final_radius,
                max_evaluations: full_stage_max_evaluations,
                ftol_abs: policy.ftol_abs,
                ftol_rel: policy.ftol_rel,
                ftol_requires_local_radius: true,
                max_cross_terms: policy.max_cross_terms,
                reuse_samples: resolve_reuse_samples(policy.reuse_samples),
                stall_iterations: policy.stall_iterations,
                stall_ftol_rel: policy.stall_ftol_rel,
                stall_ftol_abs: policy.stall_ftol_abs,
                stall_requires_stable_x: policy.stall_requires_stable_x,
                ..TrustBqOptions::default()
            },
            &mut objective_fn,
            &mut certificate_progress,
        )?;
        let trace_classification = result.trace_classification();
        let _trust_bq_diagnostics = (
            result.iterations,
            result.final_radius,
            result.last_model_sample_count,
            trace_classification.as_str(),
            result.stop_reason.is_acceptable_convergence(),
        );

        let logged_best_theta = best_theta.into_inner();
        let logged_best_fmin = best_fmin.get();
        let (final_theta, final_fmin) =
            if logged_best_fmin.is_finite() && logged_best_fmin <= result.fmin {
                (logged_best_theta, logged_best_fmin)
            } else {
                (result.x, result.fmin)
            };
        let base_status = Self::trust_bq_status_label(result.stop_reason);
        let return_value = Some(match &ladder_label {
            Some(label) => format!("START_LADDER({label}): {base_status}"),
            None => base_status,
        });

        self.finalize_fit_result(
            final_theta,
            final_fmin,
            (result.fevals + ladder_fevals) as i64,
            fit_log.into_inner(),
            Optimizer::TrustBq,
            return_value,
        )?;
        self.optsum.final_trust_radius = Some(result.final_radius);
        Ok(self)
    }

    pub(super) fn fit_cobyla_with_maxeval(
        &mut self,
        reml: bool,
        maxeval_override: Option<usize>,
    ) -> Result<&mut Self> {
        let lb = self.lower_bounds();
        self.optsum.optimizer = Optimizer::Cobyla;

        let a_blocks = self.a_blocks.clone();
        let l_blocks_template = self.l_blocks.clone();
        let reterms_template = self.reterms.clone();
        let dims = self.dims;
        let is_reml = reml;
        let fixed_sigma = self.optsum.sigma;
        let cholesky_zero_pad_tolerance = self
            .compiler_policy()
            .thresholds
            .cholesky_zero_pad_tolerance;
        let best_theta = std::cell::RefCell::new(self.optsum.initial.clone());
        let best_fmin = std::cell::Cell::new(f64::INFINITY);
        let feval_count = std::cell::Cell::new(0i64);
        let fit_log: std::cell::RefCell<Vec<(Vec<f64>, f64)>> = std::cell::RefCell::new(Vec::new());

        let reterms_work = std::cell::RefCell::new(reterms_template.clone());
        let l_blocks_work = std::cell::RefCell::new(l_blocks_template);

        let objective_fn = |theta: &[f64], _data: &mut ()| -> f64 {
            feval_count.set(feval_count.get() + 1);
            let obj = {
                let mut rw = reterms_work.borrow_mut();
                let mut lw = l_blocks_work.borrow_mut();
                Self::profiled_objective_from_parts(
                    &a_blocks,
                    &mut lw,
                    &mut rw,
                    theta,
                    dims,
                    is_reml,
                    fixed_sigma,
                    cholesky_zero_pad_tolerance,
                )
                .unwrap_or(f64::INFINITY)
            };

            fit_log.borrow_mut().push((theta.to_vec(), obj));
            if obj < best_fmin.get() {
                best_fmin.set(obj);
                *best_theta.borrow_mut() = theta.to_vec();
            }

            obj
        };

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

        let maxeval = maxeval_override.unwrap_or({
            if self.optsum.max_feval > 0 {
                self.optsum.max_feval as usize
            } else {
                10000
            }
        });

        let stop_tol = cobyla::StopTols {
            ftol_rel: self.optsum.ftol_rel,
            ftol_abs: self.optsum.ftol_abs,
            xtol_rel: self.optsum.xtol_rel,
            xtol_abs: self.optsum.xtol_abs.clone(),
        };
        let rhobeg = match self.optsum.initial_step.len() {
            0 => cobyla::RhoBeg::All(0.75),
            len if len == self.n_theta() => {
                if self
                    .optsum
                    .initial_step
                    .iter()
                    .all(|step| step.is_finite() && *step > 0.0)
                {
                    cobyla::RhoBeg::Set(self.optsum.initial_step.clone())
                } else {
                    return Err(MixedModelError::Optimization(
                        "COBYLA initial_step values must be finite and positive".to_string(),
                    ));
                }
            }
            len => {
                return Err(MixedModelError::Optimization(format!(
                    "COBYLA initial_step length {len} does not match theta length {}",
                    self.n_theta()
                )));
            }
        };

        let result = cobyla::minimize(
            objective_fn,
            &self.optsum.initial,
            &bounds,
            &cons_refs,
            (),
            maxeval,
            rhobeg,
            Some(stop_tol),
        );

        let (best_theta_val, best_fmin_val, return_value);

        match result {
            Ok((status, x_opt, fmin)) => {
                best_fmin_val = fmin;
                best_theta_val = x_opt;
                return_value = Some(Self::cobyla_success_status_label(status));
            }
            Err((status @ cobyla::FailStatus::RoundoffLimited, x_opt, _)) => {
                best_theta_val = x_opt;
                best_fmin_val = best_fmin.get();
                return_value = Some(Self::cobyla_fail_status_label(status));
            }
            Err((status, x_opt, fmin)) => {
                if fmin.is_finite() {
                    best_fmin_val = fmin;
                    best_theta_val = x_opt;
                    return_value = Some(Self::cobyla_fail_status_label(status));
                } else {
                    return Err(MixedModelError::Optimization(
                        "COBYLA optimization failed".to_string(),
                    ));
                }
            }
        }

        self.finalize_fit_result(
            best_theta_val,
            best_fmin_val,
            feval_count.get(),
            fit_log
                .into_inner()
                .into_iter()
                .map(|(theta, obj)| FitLogEntry {
                    theta,
                    objective: obj,
                })
                .collect(),
            Optimizer::Cobyla,
            return_value,
        )
    }

    fn fit_cobyla(&mut self, reml: bool) -> Result<&mut Self> {
        self.fit_cobyla_with_maxeval(reml, None)
    }

    #[cfg(feature = "prima")]
    fn fit_prima_bobyqa_with_maxeval(
        &mut self,
        reml: bool,
        maxeval_override: Option<usize>,
    ) -> Result<&mut Self> {
        self.optsum.optimizer = Optimizer::PrimaBobyqa;
        self.optsum.backend = OptimizerBackend::Prima;

        let a_blocks = self.a_blocks.clone();
        let l_blocks_template = self.l_blocks.clone();
        let reterms_template = self.reterms.clone();
        let dims = self.dims;
        let is_reml = reml;
        let fixed_sigma = self.optsum.sigma;
        let cholesky_zero_pad_tolerance = self
            .compiler_policy()
            .thresholds
            .cholesky_zero_pad_tolerance;
        let invalid_objective = self.optsum.finitial;
        let best_theta = std::cell::RefCell::new(self.optsum.initial.clone());
        let best_fmin = std::cell::Cell::new(self.optsum.finitial);
        let feval_count = std::cell::Cell::new(0i64);
        let fit_log: std::cell::RefCell<Vec<FitLogEntry>> = std::cell::RefCell::new(Vec::new());

        let reterms_work = std::cell::RefCell::new(reterms_template.clone());
        let l_blocks_work = std::cell::RefCell::new(l_blocks_template);

        let mut objective_fn = |theta: &[f64]| -> f64 {
            feval_count.set(feval_count.get() + 1);
            let obj = {
                let mut rw = reterms_work.borrow_mut();
                let mut lw = l_blocks_work.borrow_mut();
                Self::profiled_objective_from_parts(
                    &a_blocks,
                    &mut lw,
                    &mut rw,
                    theta,
                    dims,
                    is_reml,
                    fixed_sigma,
                    cholesky_zero_pad_tolerance,
                )
                .unwrap_or(invalid_objective)
            };

            fit_log.borrow_mut().push(FitLogEntry {
                theta: theta.to_vec(),
                objective: obj,
            });
            if obj + 1e-12 < best_fmin.get() {
                best_fmin.set(obj);
                *best_theta.borrow_mut() = theta.to_vec();
            }

            obj
        };

        let maxfun = maxeval_override.unwrap_or_else(|| {
            if self.optsum.max_feval > 0 {
                self.optsum.max_feval as usize
            } else {
                10000
            }
        });

        let lower_bounds = self.lower_bounds();
        let upper_bounds = vec![f64::INFINITY; self.n_theta()];
        let result = minimize_bobyqa(
            &self.optsum.initial,
            &lower_bounds,
            &upper_bounds,
            PrimaBobyqaOptions {
                rhobeg: self.optsum.rhobeg,
                rhoend: self.optsum.rhoend,
                maxfun,
            },
            &mut objective_fn,
        )?;

        let logged_best_theta = best_theta.into_inner();
        let logged_best_fmin = best_fmin.get();
        let (final_theta, final_fmin) =
            if logged_best_fmin.is_finite() && logged_best_fmin <= result.fmin {
                (logged_best_theta, logged_best_fmin)
            } else {
                (result.x, result.fmin)
            };

        self.finalize_fit_result(
            final_theta,
            final_fmin,
            if result.feval > 0 {
                result.feval
            } else {
                feval_count.get()
            },
            fit_log.into_inner(),
            Optimizer::PrimaBobyqa,
            None,
        )?;
        self.optsum.return_value = result.return_code;
        self.optsum.backend = OptimizerBackend::Prima;

        Ok(self)
    }

    #[cfg(feature = "nlopt")]
    fn fit_nlopt_large_theta_with_maxeval(
        &mut self,
        reml: bool,
        maxeval_override: Option<usize>,
    ) -> Result<&mut Self> {
        // NEWUOA is unconstrained — no lower-bound enforcement, so the soft
        // barrier (objective returns finitial outside the feasible region)
        // is what keeps θ ≥ 0. This has been the behaviour for n_theta > 6
        // since the original port and is preserved.
        self.fit_nlopt_with_algorithm(
            NloptAlgorithm::Newuoa,
            Optimizer::NloptNewuoa,
            reml,
            maxeval_override,
            /*use_lower_bounds=*/ false,
        )
    }

    /// Small-θ path (n_theta ∈ 2..=6). Uses BOBYQA, which is bounded — we
    /// pass `θ_lower` from `lower_bounds()` so the optimizer never walks
    /// off the feasible region. On smooth, well-conditioned problems
    /// (most LMMs in this regime) BOBYQA converges in roughly half the
    /// evaluations of the pattern-search fallback; profiling kb07 (n_theta
    /// = 2) showed pattern_search using ~84 evaluations for what BOBYQA
    /// typically does in ~25.
    #[cfg(feature = "nlopt")]
    fn fit_nlopt_small_theta_with_maxeval(
        &mut self,
        reml: bool,
        maxeval_override: Option<usize>,
    ) -> Result<&mut Self> {
        self.fit_nlopt_with_algorithm(
            NloptAlgorithm::Bobyqa,
            Optimizer::NloptBobyqa,
            reml,
            maxeval_override,
            /*use_lower_bounds=*/ true,
        )
    }

    #[cfg(feature = "nlopt")]
    fn fit_nlopt_small_theta(&mut self, reml: bool) -> Result<&mut Self> {
        // Mirror the large-θ fallback pattern: if BOBYQA fails to converge
        // (rare on this problem class but possible on near-singular fits),
        // retry with the robust pattern-search optimizer rather than
        // bubbling the error up.
        if self.fit_nlopt_small_theta_with_maxeval(reml, None).is_err() {
            // Reset so pattern_search starts from the recorded initial θ
            // rather than wherever BOBYQA bailed out.
            self.optsum.feval = -1;
            self.optsum.fmin = f64::INFINITY;
            self.optsum.fit_log.clear();
            return self.fit_multivariate_pattern_search();
        }
        Ok(self)
    }

    #[cfg(feature = "nlopt")]
    fn fit_nlopt_with_algorithm(
        &mut self,
        algorithm: NloptAlgorithm,
        optimizer: Optimizer,
        reml: bool,
        maxeval_override: Option<usize>,
        use_lower_bounds: bool,
    ) -> Result<&mut Self> {
        const NLOPT_FTOL_REL_DEFAULT: f64 = 1e-10;
        const NLOPT_FTOL_ABS_DEFAULT: f64 = 1e-8;
        const RUST_FTOL_REL_DEFAULT: f64 = 1e-8;
        const RUST_FTOL_ABS_DEFAULT: f64 = 1e-12;
        const RUST_INITIAL_STEP_DEFAULT: f64 = 0.75;
        const LARGE_VSIZE2_BOBYQA_FTOL_REL_DEFAULT: f64 = 1e-10;

        self.optsum.optimizer = optimizer;
        let use_large_vsize2_tuning =
            optimizer == Optimizer::NloptBobyqa && self.use_large_single_vsize2_optimizer_tuning();

        let a_blocks = self.a_blocks.clone();
        let l_blocks_template = self.l_blocks.clone();
        let reterms_template = self.reterms.clone();
        let dims = self.dims;
        let is_reml = reml;
        let fixed_sigma = self.optsum.sigma;
        let cholesky_zero_pad_tolerance = self
            .compiler_policy()
            .thresholds
            .cholesky_zero_pad_tolerance;
        let invalid_objective = self.optsum.finitial;
        let best_theta = std::cell::RefCell::new(self.optsum.initial.clone());
        let best_fmin = std::cell::Cell::new(self.optsum.finitial);
        let feval_count = std::cell::Cell::new(0i64);
        let fit_log: std::cell::RefCell<Vec<FitLogEntry>> = std::cell::RefCell::new(Vec::new());

        let reterms_work = std::cell::RefCell::new(reterms_template.clone());
        let l_blocks_work = std::cell::RefCell::new(l_blocks_template);

        let objective_fn = |theta: &[f64], _gradient: Option<&mut [f64]>, _data: &mut ()| -> f64 {
            feval_count.set(feval_count.get() + 1);
            let obj = {
                let mut rw = reterms_work.borrow_mut();
                let mut lw = l_blocks_work.borrow_mut();
                Self::profiled_objective_from_parts(
                    &a_blocks,
                    &mut lw,
                    &mut rw,
                    theta,
                    dims,
                    is_reml,
                    fixed_sigma,
                    cholesky_zero_pad_tolerance,
                )
                .unwrap_or(invalid_objective)
            };

            fit_log.borrow_mut().push(FitLogEntry {
                theta: theta.to_vec(),
                objective: obj,
            });
            if obj + 1e-12 < best_fmin.get() {
                best_fmin.set(obj);
                *best_theta.borrow_mut() = theta.to_vec();
            }

            obj
        };

        let maxeval = maxeval_override.unwrap_or({
            if self.optsum.max_feval > 0 {
                self.optsum.max_feval as usize
            } else {
                10000
            }
        });

        let n_theta = self.n_theta();
        let mut opt = Nlopt::new(algorithm, n_theta, objective_fn, NloptTarget::Minimize, ());
        let ftol_rel = if (self.optsum.ftol_rel - RUST_FTOL_REL_DEFAULT).abs() <= f64::EPSILON {
            if use_large_vsize2_tuning {
                // The large one-term random-slope fast path can spend many
                // extra BOBYQA evaluations polishing below the numerical
                // scale that changes the fitted model. The global NLopt
                // default below is already the parity/performance compromise
                // for the other model classes.
                LARGE_VSIZE2_BOBYQA_FTOL_REL_DEFAULT
            } else {
                NLOPT_FTOL_REL_DEFAULT
            }
        } else {
            self.optsum.ftol_rel
        };
        let ftol_abs = if (self.optsum.ftol_abs - RUST_FTOL_ABS_DEFAULT).abs() <= f64::EPSILON {
            NLOPT_FTOL_ABS_DEFAULT
        } else {
            self.optsum.ftol_abs
        };
        if ftol_rel > 0.0 {
            Self::nlopt_ok(opt.set_ftol_rel(ftol_rel), "set_ftol_rel")?;
        }
        if ftol_abs > 0.0 {
            Self::nlopt_ok(opt.set_ftol_abs(ftol_abs), "set_ftol_abs")?;
        }
        if self.optsum.xtol_rel > 0.0 {
            Self::nlopt_ok(opt.set_xtol_rel(self.optsum.xtol_rel), "set_xtol_rel")?;
        }
        if !self.optsum.xtol_abs.is_empty() {
            Self::nlopt_ok(opt.set_xtol_abs(&self.optsum.xtol_abs), "set_xtol_abs")?;
        }
        let use_nlopt_default_initial_step = self.optsum.initial_step.len() == n_theta
            && self
                .optsum
                .initial_step
                .iter()
                .all(|&step| (step - RUST_INITIAL_STEP_DEFAULT).abs() <= f64::EPSILON);
        if !self.optsum.initial_step.is_empty() && !use_nlopt_default_initial_step {
            Self::nlopt_ok(
                opt.set_initial_step(&self.optsum.initial_step),
                "set_initial_step",
            )?;
        }
        if maxeval > 0 {
            // `maxeval` derives from a caller-settable `max_feval`;
            // `nlopt::set_maxeval` takes u32. A plain `as u32` wraps silently
            // on 64-bit, so e.g. 2^32+5 would stop the optimizer after 5
            // evaluations while `fit()` still returns Ok — non-convergence
            // masquerading as a fit. Saturate instead (mirrors the PRIMA
            // path's explicit bound): a value at/above u32::MAX simply means
            // "effectively unlimited".
            let maxeval_u32 = maxeval.min(u32::MAX as usize) as u32;
            Self::nlopt_ok(opt.set_maxeval(maxeval_u32), "set_maxeval")?;
        }
        if self.optsum.max_time > 0.0 {
            Self::nlopt_ok(opt.set_maxtime(self.optsum.max_time), "set_maxtime")?;
        }
        if use_lower_bounds {
            // BOBYQA is bounded — let NLopt enforce θ ≥ θ_lower instead of
            // relying on the soft "objective returns finitial when invalid"
            // barrier, which can confuse the trust-region update step.
            let lb = self.lower_bounds();
            Self::nlopt_ok(opt.set_lower_bounds(&lb), "set_lower_bounds")?;
        }

        let mut theta = self.optsum.initial.clone();
        let optimize_result = opt.optimize(&mut theta);
        let status_label = match &optimize_result {
            Ok((status, _)) => Self::nlopt_status_label(&format!("{status:?}")),
            Err((status, _)) => Self::nlopt_status_label(&format!("{status:?}")),
        };

        let (candidate_theta, candidate_fmin) = match optimize_result {
            Ok((_, fmin)) => (theta.clone(), fmin),
            Err((NloptFailState::RoundoffLimited, fmin)) => (theta.clone(), fmin),
            Err((status, _)) => {
                return Err(MixedModelError::Optimization(format!(
                    "NLopt large-theta optimization failed: {status:?}"
                )));
            }
        };

        let logged_best_theta = best_theta.into_inner();
        let logged_best_fmin = best_fmin.get();
        let (final_theta, final_fmin) = if logged_best_fmin.is_finite()
            && (!candidate_fmin.is_finite() || logged_best_fmin <= candidate_fmin)
        {
            (logged_best_theta, logged_best_fmin)
        } else {
            (candidate_theta, candidate_fmin)
        };

        self.finalize_fit_result(
            final_theta,
            final_fmin,
            feval_count.get(),
            fit_log.into_inner(),
            optimizer,
            None,
        )?;
        self.optsum.return_value = status_label;

        Ok(self)
    }

    #[cfg(feature = "nlopt")]
    fn fit_nlopt_large_theta(&mut self, reml: bool) -> Result<&mut Self> {
        if self.fit_nlopt_large_theta_with_maxeval(reml, None).is_err() {
            return self.fit_cobyla(reml);
        }
        Ok(self)
    }

    /// Fit the model by optimizing θ to minimize the objective.
    pub fn fit(&mut self, reml: bool) -> Result<&mut Self> {
        let options = if reml {
            FitOptions::reml()
        } else {
            FitOptions::ml()
        };
        self.fit_with_options(options)
    }

    /// Fit the model with explicit options.
    pub fn fit_with_options(&mut self, options: FitOptions) -> Result<&mut Self> {
        if self.optsum.feval > 0 {
            return Err(MixedModelError::AlreadyFitted);
        }
        self.progress_callback = options.progress_callback.clone();
        let reml = options.criterion.is_reml();

        // Check for constant response. Skipped for summary-estimate fits:
        // identical first-stage point estimates with different sampling
        // variances are a well-defined meta-analysis case (tau -> 0,
        // beta_hat = common value, weights set the residual variance per
        // study). See docs/summary_estimates_meta_analysis.md.
        let summary_estimate_fit = self.residual_source
            == crate::model::summary_estimates::ResidualSource::FixedSamplingVariance;
        let y_is_constant = {
            let y = self.y();
            let y0 = y[0];
            y.iter().all(|&yi| (yi - y0).abs() < f64::EPSILON)
        };
        if y_is_constant && !summary_estimate_fit {
            return Err(MixedModelError::ConstantResponse);
        }
        if y_is_constant && summary_estimate_fit {
            // Analytical short-circuit: with identical first-stage
            // estimates the meta-analysis fit collapses to tau -> 0 and
            // beta = common value. Running the optimizer here hits a
            // degenerate Cholesky boundary and surfaces as
            // PosDefException, so fix theta at the lower bound directly
            // and finalize.
            self.optsum.reml = reml;
            let theta_zero = vec![0.0_f64; self.optsum.initial.len()];
            let obj_zero = self.objective_at(&theta_zero)?;
            self.optsum.finitial = obj_zero;
            return self.finalize_fit_result(
                theta_zero,
                obj_zero,
                1,
                Vec::new(),
                Optimizer::PatternSearch,
                Some("CONSTANT_RESPONSE_SHORTCIRCUIT".to_string()),
            );
        }

        if self.feterm.rank >= self.dims.n {
            return Err(MixedModelError::RankSaturatedFixedEffects {
                rank: self.feterm.rank,
                nobs: self.dims.n,
            });
        }

        self.apply_optimizer_control(&options.optimizer_control)?;
        self.optsum.reml = reml;

        if let Some(optimizer) = options.optimizer_control.optimizer.named() {
            self.fit_with_forced_optimizer(reml, optimizer)?;
            return Ok(self);
        }

        // Initial objective evaluation (with one rescaling retry on a
        // non-finite value — see set_initial_objective_with_rescue).
        self.set_initial_objective_with_rescue()?;

        if self.use_scalar_single_theta_optimizer() {
            self.fit_scalar_single_theta()?;
        } else {
            // The `use_*_nlopt_*` predicates always return `false` when
            // the `nlopt` feature is disabled, so the no-feature build
            // never reaches the nlopt arms even if they appear in the
            // source. Cfg-gating the call sites lets the no-feature
            // build still type-check (the methods themselves are gated
            // out below).
            #[cfg(feature = "nlopt")]
            {
                if self.use_nlopt_bobyqa_small_theta_optimizer() {
                    self.fit_nlopt_small_theta(reml)?;
                } else if self.use_large_theta_nlopt_optimizer() {
                    self.fit_nlopt_large_theta(reml)?;
                } else {
                    self.fit_cobyla(reml)?;
                }
            }
            #[cfg(not(feature = "nlopt"))]
            {
                self.configure_native_auto_crossed_large_recourse();
                self.fit_trust_bq_with_maxeval(reml, None)?;
            }
        }

        self.apply_kkt_guided_boundary_restart(reml)?;
        self.apply_active_face_refit()?;
        self.refresh_optimizer_certificate();
        self.refresh_effective_covariance_summaries();
        self.refresh_covariance_parameter_traces();
        self.refresh_fixed_effect_covariance_matrix();
        self.refresh_fixed_effect_inference_table();
        Ok(self)
    }

    #[cfg(any(test, not(feature = "nlopt")))]
    pub(super) fn configure_native_auto_crossed_large_recourse(&mut self) {
        if !self.should_auto_use_native_crossed_large_ladder() {
            return;
        }

        self.trust_bq_start_ladder = TrustBqStartLadder::DiagonalFirst;
        if self.optsum.max_feval <= 0 {
            self.optsum.max_feval = NATIVE_AUTO_CROSSED_LARGE_MAX_FEVAL;
        }
    }

    #[cfg(any(test, not(feature = "nlopt")))]
    pub(super) fn should_auto_use_native_crossed_large_ladder(&self) -> bool {
        self.optsum.optimizer_source == OptimizerSource::Auto
            && self.trust_bq_start_ladder == TrustBqStartLadder::Off
            && self.n_theta() >= 7
            && self.has_crossed_full_cholesky_vector_term()
    }

    #[cfg(any(test, not(feature = "nlopt")))]
    fn has_crossed_full_cholesky_vector_term(&self) -> bool {
        self.reterms.iter().any(|vector_term| {
            full_cholesky_vector_term(vector_term)
                && self.reterms.iter().any(|other| {
                    !std::ptr::eq(vector_term, other)
                        && random_terms_are_crossed(vector_term, other)
                })
        })
    }
}

/// Variance tolerance shared by the scalar covariance KKT certificate and the
/// theta-space pre-check that gates the KKT-guided boundary restart.
const SCALAR_KKT_VARIANCE_TOLERANCE: f64 = 1e-8;

/// Covariance-eigenvalue tolerance shared by the 2x2 covariance KKT
/// certificate and the theta-space pre-check that gates the KKT-guided
/// boundary restart.
const TWO_BY_TWO_KKT_COVARIANCE_TOLERANCE: f64 = 1e-8;

#[cfg(any(test, not(feature = "nlopt")))]
pub(super) const NATIVE_AUTO_CROSSED_LARGE_MAX_FEVAL: i64 = 2_000;

#[cfg(any(test, not(feature = "nlopt")))]
fn full_cholesky_vector_term(term: &ReMat) -> bool {
    term.vsize >= 2 && term.n_theta() == term.vsize * (term.vsize + 1) / 2
}

#[cfg(any(test, not(feature = "nlopt")))]
fn random_terms_are_crossed(left: &ReMat, right: &ReMat) -> bool {
    !refs_nested_within(&left.refs, &right.refs) && !refs_nested_within(&right.refs, &left.refs)
}

#[cfg(any(test, not(feature = "nlopt")))]
fn refs_nested_within(child: &[u32], parent: &[u32]) -> bool {
    if child.len() != parent.len() {
        return false;
    }

    let mut mapping = std::collections::BTreeMap::new();
    for (&child_ref, &parent_ref) in child.iter().zip(parent) {
        if let Some(previous) = mapping.insert(child_ref, parent_ref) {
            if previous != parent_ref {
                return false;
            }
        }
    }
    true
}

fn classify_scalar_covariance_kkt(
    variance: f64,
    score: f64,
    variance_tolerance: f64,
    score_tolerance: f64,
) -> CovarianceKktClassification {
    if variance <= variance_tolerance {
        if score < -score_tolerance {
            CovarianceKktClassification::InvalidBoundaryStop
        } else {
            CovarianceKktClassification::ValidZeroVariance
        }
    } else if score.abs() <= score_tolerance {
        CovarianceKktClassification::InteriorConverged
    } else {
        CovarianceKktClassification::WeakIdentification
    }
}

fn scalar_covariance_kkt_residual(
    variance: f64,
    score: f64,
    complementarity: f64,
    variance_tolerance: f64,
) -> f64 {
    if variance <= variance_tolerance {
        (-score).max(0.0).max(complementarity)
    } else {
        score.abs().max(complementarity)
    }
}

pub(super) fn two_by_two_covariance_from_theta(theta: [f64; 3]) -> [[f64; 2]; 2] {
    let l00 = theta[0].max(0.0);
    let l10 = theta[1];
    let l11 = theta[2].max(0.0);
    [[l00 * l00, l00 * l10], [l00 * l10, l10 * l10 + l11 * l11]]
}

pub(super) fn two_by_two_theta_from_covariance(covariance: [[f64; 2]; 2]) -> Option<[f64; 3]> {
    let a = covariance[0][0];
    let b = 0.5 * (covariance[0][1] + covariance[1][0]);
    let c = covariance[1][1];
    let scale = a.abs().max(b.abs()).max(c.abs()).max(1.0);
    let tolerance = 1e-10 * scale;
    let (min_eig, _) = symmetric_2x2_eigenvalues([[a, b], [b, c]]);
    if min_eig < -tolerance || a < -tolerance || c < -tolerance {
        return None;
    }

    if a <= tolerance {
        if b.abs() > 10.0 * tolerance {
            return None;
        }
        return Some([0.0, 0.0, c.max(0.0).sqrt()]);
    }

    let l00 = a.max(0.0).sqrt();
    let l10 = b / l00;
    let schur = c - l10 * l10;
    if schur < -10.0 * tolerance {
        return None;
    }
    Some([l00, l10, schur.max(0.0).sqrt()])
}

fn two_by_two_covariance_step(covariance: [[f64; 2]; 2]) -> f64 {
    (1e-5 * (1.0 + two_by_two_frobenius_norm(covariance))).max(1e-8)
}

fn two_by_two_add_direction(
    covariance: [[f64; 2]; 2],
    direction: [[f64; 2]; 2],
    step: f64,
) -> [[f64; 2]; 2] {
    [
        [
            covariance[0][0] + step * direction[0][0],
            covariance[0][1] + step * direction[0][1],
        ],
        [
            covariance[1][0] + step * direction[1][0],
            covariance[1][1] + step * direction[1][1],
        ],
    ]
}

pub(super) fn symmetric_2x2_eigenvalues(matrix: [[f64; 2]; 2]) -> (f64, f64) {
    let a = matrix[0][0];
    let b = 0.5 * (matrix[0][1] + matrix[1][0]);
    let c = matrix[1][1];
    let center = 0.5 * (a + c);
    let radius = (0.5 * (a - c)).hypot(b);
    (center - radius, center + radius)
}

fn symmetric_2x2_min_eigenvector(matrix: [[f64; 2]; 2]) -> [f64; 2] {
    let a = matrix[0][0];
    let b = 0.5 * (matrix[0][1] + matrix[1][0]);
    let c = matrix[1][1];
    let (lambda, _) = symmetric_2x2_eigenvalues([[a, b], [b, c]]);
    let mut vector = if b.abs() > 1e-14 {
        [b, lambda - a]
    } else if a <= c {
        [1.0, 0.0]
    } else {
        [0.0, 1.0]
    };
    let norm = vector[0].hypot(vector[1]);
    if norm > 0.0 && norm.is_finite() {
        vector[0] /= norm;
        vector[1] /= norm;
    }
    vector
}

pub(super) fn two_by_two_frobenius_norm(matrix: [[f64; 2]; 2]) -> f64 {
    (matrix[0][0] * matrix[0][0]
        + matrix[0][1] * matrix[0][1]
        + matrix[1][0] * matrix[1][0]
        + matrix[1][1] * matrix[1][1])
        .sqrt()
}

fn two_by_two_multiply(left: [[f64; 2]; 2], right: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [
        [
            left[0][0] * right[0][0] + left[0][1] * right[1][0],
            left[0][0] * right[0][1] + left[0][1] * right[1][1],
        ],
        [
            left[1][0] * right[0][0] + left[1][1] * right[1][0],
            left[1][0] * right[0][1] + left[1][1] * right[1][1],
        ],
    ]
}

fn two_by_two_complementarity(covariance: [[f64; 2]; 2], score: [[f64; 2]; 2]) -> f64 {
    let product = two_by_two_multiply(score, covariance);
    two_by_two_frobenius_norm(product)
        / (1.0 + two_by_two_frobenius_norm(score) * two_by_two_frobenius_norm(covariance))
}

fn two_by_two_covariance_kkt_residual(
    min_eig_g: f64,
    min_eig_score: f64,
    complementarity: f64,
) -> f64 {
    (-min_eig_g)
        .max(0.0)
        .max((-min_eig_score).max(0.0))
        .max(complementarity)
}

fn classify_two_by_two_covariance_kkt(
    min_eig_g: f64,
    max_eig_g: f64,
    min_eig_score: f64,
    score_norm: f64,
    complementarity: f64,
    covariance_tolerance: f64,
    score_tolerance: f64,
    complementarity_tolerance: f64,
) -> CovarianceKktClassification {
    if min_eig_g > covariance_tolerance {
        if score_norm <= score_tolerance {
            CovarianceKktClassification::InteriorConverged
        } else {
            CovarianceKktClassification::WeakIdentification
        }
    } else if min_eig_score < -score_tolerance {
        CovarianceKktClassification::InvalidBoundaryStop
    } else if complementarity <= complementarity_tolerance {
        if max_eig_g <= covariance_tolerance {
            CovarianceKktClassification::ValidZeroVariance
        } else {
            CovarianceKktClassification::ValidRankDeficientCovariance
        }
    } else {
        CovarianceKktClassification::WeakIdentification
    }
}

fn kkt_restart_delta_grid(scale: f64) -> [f64; 6] {
    let base = (1e-4 * scale.max(1.0)).max(1e-8);
    [
        base,
        10.0 * base,
        100.0 * base,
        1_000.0 * base,
        10_000.0 * base,
        100_000.0 * base,
    ]
}

pub(super) fn finite_difference_steps(
    theta: &[f64],
    lower_bounds: &[f64],
    relative_scale: f64,
) -> Vec<f64> {
    theta
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            let lower = lower_bounds
                .get(index)
                .copied()
                .unwrap_or(f64::NEG_INFINITY);
            let scale = if lower.is_finite() {
                value.abs().max(lower.abs()).max(1.0)
            } else {
                value.abs().max(1.0)
            };
            (relative_scale * scale).max(1e-8)
        })
        .collect()
}

pub(super) fn feasible_central_step(value: f64, lower: f64, requested_step: f64) -> Option<f64> {
    let scale = value.abs().max(1.0);
    let min_step = 1e-10 * scale;
    let mut step = requested_step.abs().max(min_step);
    if lower.is_finite() {
        let clearance = value - lower;
        if clearance <= min_step {
            return None;
        }
        step = step.min(0.5 * clearance);
    }
    step.is_finite().then_some(step).filter(|step| *step > 0.0)
}

fn finite_difference_gradient_coordinate<F>(
    objective: &mut F,
    theta: &[f64],
    lower_bounds: &[f64],
    f0: f64,
    index: usize,
    step: f64,
) -> Option<f64>
where
    F: FnMut(&[f64]) -> Option<f64>,
{
    let lower = lower_bounds
        .get(index)
        .copied()
        .unwrap_or(f64::NEG_INFINITY);
    if !lower.is_finite() || theta[index] - step >= lower {
        let mut plus = theta.to_vec();
        let mut minus = theta.to_vec();
        plus[index] += step;
        minus[index] -= step;
        let f_plus = objective(&plus)?;
        let f_minus = objective(&minus)?;
        if f_plus.is_finite() && f_minus.is_finite() {
            return Some((f_plus - f_minus) / (2.0 * step));
        }
    }

    let mut plus = theta.to_vec();
    let mut plus2 = theta.to_vec();
    plus[index] += step;
    plus2[index] += 2.0 * step;
    let f_plus = objective(&plus)?;
    let f_plus2 = objective(&plus2)?;
    if f_plus.is_finite() && f_plus2.is_finite() {
        Some((-3.0 * f0 + 4.0 * f_plus - f_plus2) / (2.0 * step))
    } else {
        None
    }
}

fn finite_difference_objective_2d<F>(
    objective: &mut F,
    theta: &[f64],
    row: usize,
    row_delta: f64,
    col: usize,
    col_delta: f64,
) -> Option<f64>
where
    F: FnMut(&[f64]) -> Option<f64>,
{
    let mut trial = theta.to_vec();
    trial[row] += row_delta;
    trial[col] += col_delta;
    objective(&trial).filter(|value| value.is_finite())
}

pub(super) fn finite_difference_deviance_varpar(
    evaluator: &mut LinearMixedModel,
    varpar: &[f64],
    index: usize,
    delta: f64,
    reml: bool,
) -> Result<f64> {
    let mut trial = varpar.to_vec();
    trial[index] += delta;
    evaluator.deviance_varpar(&trial, reml).and_then(|value| {
        value.is_finite().then_some(value).ok_or_else(|| {
            MixedModelError::Optimization(
                "finite-difference deviance_varpar evaluation is non-finite".to_string(),
            )
        })
    })
}

pub(super) fn finite_difference_deviance_varpar_2d(
    evaluator: &mut LinearMixedModel,
    varpar: &[f64],
    row: usize,
    row_delta: f64,
    col: usize,
    col_delta: f64,
    reml: bool,
) -> Result<f64> {
    let mut trial = varpar.to_vec();
    trial[row] += row_delta;
    trial[col] += col_delta;
    evaluator.deviance_varpar(&trial, reml).and_then(|value| {
        value.is_finite().then_some(value).ok_or_else(|| {
            MixedModelError::Optimization(
                "finite-difference deviance_varpar evaluation is non-finite".to_string(),
            )
        })
    })
}

fn jittered_theta(
    theta: &[f64],
    lower_bounds: &[f64],
    jitter_scale: f64,
    jitter_index: usize,
) -> Vec<f64> {
    let mut jittered = theta
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            let direction = ((index + 1 + jitter_index * 17) as f64).sin();
            let scale = value.abs().max(1.0);
            value + direction * jitter_scale * scale
        })
        .collect::<Vec<_>>();
    LinearMixedModel::project_theta_to_bounds(&mut jittered, lower_bounds);
    jittered
}

fn optimizer_name(optimizer: Optimizer) -> &'static str {
    match optimizer {
        Optimizer::Cobyla => "cobyla",
        Optimizer::PatternSearch => "pattern_search",
        Optimizer::TrustBq => "trust_bq",
        Optimizer::NloptNewuoa => "newuoa",
        Optimizer::NloptBobyqa => "bobyqa",
        Optimizer::PrimaBobyqa => "bobyqa",
        Optimizer::PrimaCobyla => "cobyla",
        Optimizer::PrimaLincoa => "lincoa",
        Optimizer::PrimaNewuoa => "newuoa",
    }
}

pub(super) fn trust_bq_initial_radius(initial_step: &[f64], n_theta: usize) -> f64 {
    if initial_step.len() == n_theta
        && initial_step
            .iter()
            .all(|step| step.is_finite() && *step > 0.0)
    {
        initial_step.iter().copied().fold(0.0, f64::max)
    } else {
        0.75
    }
}

pub(super) fn trust_bq_final_radius(xtol_abs: &[f64], n_theta: usize) -> f64 {
    if xtol_abs.len() == n_theta
        && xtol_abs
            .iter()
            .all(|tolerance| tolerance.is_finite() && *tolerance > 0.0)
    {
        xtol_abs.iter().copied().fold(1e-5, f64::max).max(1e-5)
    } else {
        1e-5
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrustBqModelFamily {
    Small,
    Moderate,
    CrossedLarge,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct TrustBqModelFamilyPolicy {
    pub(super) initial_radius: f64,
    pub(super) final_radius: f64,
    pub(super) max_evaluations: usize,
    pub(super) ftol_abs: f64,
    pub(super) ftol_rel: f64,
    pub(super) max_cross_terms: usize,
    pub(super) reuse_samples: bool,
    pub(super) stall_iterations: usize,
    pub(super) stall_ftol_rel: f64,
    pub(super) stall_ftol_abs: f64,
    pub(super) stall_requires_stable_x: bool,
    pub(super) certificate_ftol_abs: f64,
    pub(super) certificate_ftol_rel: f64,
}

/// Central TrustBQ tuning matrix for the profiled-LMM theta objective.
///
/// Evidence summary:
/// - small theta (`d <= 3`) keeps full quadratic cross terms; vector RE and
///   other compact blocks are stable with the richer interpolation model.
/// - moderate theta (`4 <= d < 7`) uses the diagonal model but keeps numeric
///   stall tolerances; there is not enough benchmark evidence to loosen stops.
/// - crossed/large theta (`d >= 7`) uses the diagonal model, a 475-evaluation
///   default budget, exact sample reuse, and the statistical stall band from
///   bd-01KRPK18RJDG76E7E5Q01AR73J. Selective cross terms were rejected by
///   bd-01KRPK18T967WA61XD6KSA043W, exact reuse was safe but marginal in
///   bd-01KRPK18TNRMXYBST852KZN5TX, and certificate-aware stop remains
///   conservative after bd-01KRPK18SMAKTTZCGY94HN6C7Y.
pub(super) fn trust_bq_model_family_policy(
    n_theta: usize,
    maxeval_override: Option<usize>,
    initial_step: &[f64],
    xtol_abs: &[f64],
    configured_max_feval: i64,
    configured_ftol_abs: f64,
    configured_ftol_rel: f64,
) -> TrustBqModelFamilyPolicy {
    let family = if n_theta >= 7 {
        TrustBqModelFamily::CrossedLarge
    } else if n_theta <= 3 {
        TrustBqModelFamily::Small
    } else {
        TrustBqModelFamily::Moderate
    };
    // Map the configured (NLopt-style) tolerances onto TrustBQ's
    // accepted-step stop band. For the small family the default
    // `ftol_rel = 1e-8` is far too loose as an accepted-step criterion: at
    // |f| ~ 2e3 it stops on any accepted reduction below ~2e-5, which on the
    // flat ridge of e.g. the sleepstudy full-covariance ML surface leaves
    // theta ~1e-4 short of the optimum — a ~6e-4 sigma / ~3e-3 fitted-value
    // error, outside the 5e-4 absolute band the cross-engine parity fixtures
    // certify. Small problems re-probe cheaply (full cross-term model, few
    // axes), so capping the relative band there buys parity-grade endpoints
    // for a few dozen extra evaluations. Explicitly *tighter* configured
    // values are still honored; moderate/crossed families keep the previous
    // floors unchanged.
    let (ftol_abs, ftol_rel) = match family {
        TrustBqModelFamily::Small => (
            configured_ftol_abs.max(1e-10),
            configured_ftol_rel.clamp(1e-12, 1e-11),
        ),
        TrustBqModelFamily::Moderate | TrustBqModelFamily::CrossedLarge => (
            configured_ftol_abs.max(1e-8),
            configured_ftol_rel.max(1e-10),
        ),
    };
    let max_evaluations = maxeval_override.unwrap_or_else(|| {
        if configured_max_feval > 0 {
            configured_max_feval as usize
        } else if family == TrustBqModelFamily::CrossedLarge {
            475
        } else {
            1000
        }
    });

    let (stall_iterations, stall_ftol_rel, stall_ftol_abs, stall_requires_stable_x) = match family {
        TrustBqModelFamily::CrossedLarge => (3, 1e-6, 1e-8, false),
        TrustBqModelFamily::Small | TrustBqModelFamily::Moderate => (4, -1.0, -1.0, true),
    };
    let certificate_ftol_abs = if stall_ftol_abs >= 0.0 {
        stall_ftol_abs
    } else {
        ftol_abs
    };
    let certificate_ftol_rel = if stall_ftol_rel >= 0.0 {
        stall_ftol_rel
    } else {
        ftol_rel
    };

    TrustBqModelFamilyPolicy {
        initial_radius: trust_bq_initial_radius(initial_step, n_theta),
        final_radius: trust_bq_final_radius(xtol_abs, n_theta),
        max_evaluations,
        ftol_abs,
        ftol_rel,
        max_cross_terms: if family == TrustBqModelFamily::Small {
            usize::MAX
        } else {
            0
        },
        reuse_samples: family == TrustBqModelFamily::CrossedLarge,
        stall_iterations,
        stall_ftol_rel,
        stall_ftol_abs,
        stall_requires_stable_x,
        certificate_ftol_abs,
        certificate_ftol_rel,
    }
}

#[derive(Debug, Clone)]
struct TrustBqCertificateStopState {
    best_f: f64,
    best_x: Vec<f64>,
    last_meaningful_feval: usize,
    objective_tolerance_abs: f64,
    objective_tolerance_rel: f64,
    theta_tolerance: f64,
    min_fevals: usize,
    min_tail_fevals: usize,
}

impl TrustBqCertificateStopState {
    fn new(n_theta: usize, maxeval: usize, ftol_abs: f64, ftol_rel: f64) -> Self {
        let model_eval_floor = (2 * n_theta + 2).max(8);
        let min_tail_fevals = model_eval_floor
            .max(24)
            .min(maxeval.saturating_sub(1).max(1));
        let min_fevals = (3 * model_eval_floor)
            .max(50)
            .min(maxeval.saturating_sub(1).max(1));
        Self {
            best_f: f64::INFINITY,
            best_x: Vec::new(),
            last_meaningful_feval: 0,
            objective_tolerance_abs: ftol_abs.max(1e-8),
            objective_tolerance_rel: ftol_rel.max(1e-10),
            theta_tolerance: 1e-5,
            min_fevals,
            min_tail_fevals,
        }
    }

    fn should_check(&mut self, progress: &TrustBqProgress<'_>) -> bool {
        if !progress.fmin.is_finite() {
            return false;
        }
        let scaled_objective_tolerance = self.objective_tolerance_abs
            + self.objective_tolerance_rel * progress.fmin.abs().max(1.0);

        if self.best_x.is_empty() || (self.best_f - progress.fmin) > scaled_objective_tolerance {
            self.best_f = progress.fmin;
            self.best_x = progress.x.to_vec();
            self.last_meaningful_feval = progress.fevals;
            return false;
        }

        let theta_move = progress
            .x
            .iter()
            .zip(self.best_x.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let stable_theta = theta_move <= self.theta_tolerance;
        let enough_total = progress.fevals >= self.min_fevals;
        let enough_tail =
            progress.fevals.saturating_sub(self.last_meaningful_feval) >= self.min_tail_fevals;
        let contracted = progress.radius < 0.75;

        enough_total && enough_tail && stable_theta && contracted
    }
}

fn verification_status(
    runs: &[ConvergenceVerificationRun],
    options: &ConvergenceVerificationOptions,
) -> ConvergenceVerificationStatus {
    if runs.is_empty() {
        return ConvergenceVerificationStatus::NotRun;
    }

    let all_agree = runs.iter().all(|run| run.agrees);
    if all_agree
        && runs
            .iter()
            .any(|run| run.label.starts_with("optimizer_consensus_"))
    {
        ConvergenceVerificationStatus::OptimizerConsensus
    } else if all_agree {
        ConvergenceVerificationStatus::RestartAgrees
    } else if runs
        .iter()
        .any(|run| run.label == "restart_from_optimum" && core_verification_failed(run, options))
    {
        ConvergenceVerificationStatus::Unstable
    } else {
        ConvergenceVerificationStatus::Fragile
    }
}

fn core_verification_failed(
    run: &ConvergenceVerificationRun,
    options: &ConvergenceVerificationOptions,
) -> bool {
    let objective_failed = run
        .objective_delta
        .map(|delta| delta > options.objective_tolerance)
        .unwrap_or(true);
    let beta_failed = run
        .max_abs_beta_delta
        .map(|delta| delta > options.beta_tolerance)
        .unwrap_or(true);
    let rank_failed = run
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.contains("effective covariance ranks changed"));
    objective_failed || beta_failed || rank_failed
}

fn verification_message(
    status: ConvergenceVerificationStatus,
    runs: &[ConvergenceVerificationRun],
) -> String {
    match status {
        ConvergenceVerificationStatus::NotRun => "convergence verification was not run".to_string(),
        ConvergenceVerificationStatus::RestartAgrees => {
            "restart from fitted theta agrees with the recorded optimum".to_string()
        }
        ConvergenceVerificationStatus::OptimizerConsensus => {
            "restart and alternate optimizer checks agree with the recorded optimum".to_string()
        }
        ConvergenceVerificationStatus::Fragile => {
            let failed = runs
                .iter()
                .filter(|run| !run.agrees)
                .map(|run| run.label.clone())
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                "objective, fixed effects, and rank are stable, but parameterization checks are fragile: {failed}"
            )
        }
        ConvergenceVerificationStatus::Unstable => {
            "restart from fitted theta did not reproduce the recorded optimum".to_string()
        }
    }
}
