//! GLMM fit metadata, diagnostic recording, and fallback labelling.
//!
//! Moved verbatim from `generalized/mod.rs` during the module split
//! (bd-01KWHYQSTWK60P6HA4S4B2K99P). No logic changes.

use super::*;

impl GeneralizedLinearMixedModel {
    /// Fixed-effect covariance from the final PIRLS working Hessian, rescaled
    /// from the inner working-LMM residual convention to the GLMM dispersion
    /// convention. For Bernoulli/Poisson families the target scale is exactly
    /// one; scaled families use the same ML `sqrt(pwrss / n)` convention as
    /// GLMM prediction covariance.
    pub(super) fn profiled_glmm_fixed_effect_covariance(&self) -> Option<DMatrix<f64>> {
        let inner_scale = self.lmm.sigma();
        let glmm_scale = self.glmm_conditional_prediction_covariance_scale()?;
        if !inner_scale.is_finite()
            || inner_scale <= 0.0
            || !glmm_scale.is_finite()
            || glmm_scale <= 0.0
        {
            return None;
        }
        let multiplier = (glmm_scale / inner_scale).powi(2);
        let covariance = self.lmm.vcov() * multiplier;
        covariance
            .iter()
            .all(|value| value.is_finite())
            .then_some(covariance)
    }

    pub(super) fn profiled_glmm_fixed_effect_covariance_matrix(
        &self,
    ) -> FixedEffectCovarianceMatrix {
        let mut payload = self.lmm.glmm_fixed_effect_covariance_matrix();
        let Some(covariance) = self.profiled_glmm_fixed_effect_covariance() else {
            return FixedEffectCovarianceMatrix::unavailable(
                payload.coef_names,
                "glmm_fixed_effect_covariance_scale_unavailable",
                payload.details,
                vec![
                    "PIRLS/Laplace fixed-effect covariance could not be rescaled from the inner working-LMM residual convention to the GLMM dispersion convention"
                        .to_string(),
                ],
            );
        };
        payload.matrix = Some(matrix_rows_local(&covariance));
        let inner_scale = self.lmm.sigma();
        let glmm_scale = self
            .glmm_conditional_prediction_covariance_scale()
            .expect("profiled covariance already validated the GLMM scale");
        payload.notes = vec![format!(
            "PIRLS/Laplace working-Hessian fixed-effect covariance rescaled from inner working-LMM sigma {inner_scale:.9} to GLMM dispersion scale {glmm_scale:.9}; inference claims remain on fixed_effect_inference_table rows"
        )];
        payload
    }

    pub(super) fn recorded_fixed_effect_covariance(&self) -> Option<DMatrix<f64>> {
        let payload = self
            .lmm
            .compiler_artifact
            .fixed_effect_covariance_matrix
            .as_ref()?;
        if payload.status != FixedEffectCovarianceStatus::Available {
            return None;
        }
        let rows = payload.matrix.as_ref()?;
        let nrows = rows.len();
        let ncols = rows.first().map(Vec::len).unwrap_or(0);
        if nrows == 0 || ncols == 0 || rows.iter().any(|row| row.len() != ncols) || nrows != ncols {
            return None;
        }
        let values = rows.iter().flatten().copied().collect::<Vec<_>>();
        values
            .iter()
            .all(|value| value.is_finite())
            .then(|| DMatrix::from_row_slice(nrows, ncols, &values))
    }

    pub(super) fn record_invalid_agq_diagnostic(&mut self, n_agq: usize, reason: &str) {
        self.lmm
            .compiler_artifact
            .diagnostics
            .retain(|diagnostic| diagnostic.code != DiagnosticCode::InvalidAgqRequest);

        let affected_terms = self
            .lmm
            .reterms
            .iter()
            .map(random_effect_term_label)
            .collect::<Vec<_>>();
        let term_summaries = self
            .lmm
            .reterms
            .iter()
            .map(|term| {
                serde_json::json!({
                    "group": &term.grouping_name,
                    "columns": &term.cnames,
                    "basis_dimension": term.vsize,
                })
            })
            .collect::<Vec<_>>();
        let mut diagnostic = Diagnostic::new(
            DiagnosticCode::InvalidAgqRequest,
            DiagnosticSeverity::Error,
            DiagnosticStage::Optimization,
            format!(
                "Invalid adaptive Gauss-Hermite quadrature request: n_agq = {n_agq} requires exactly one scalar random-effects term. Use n_agq = 1 for the Laplace approximation or simplify the random-effects structure."
            ),
        )
        .with_affected_terms(affected_terms)
        .with_suggested_actions(vec![
            "use n_agq = 1 for Laplace approximation on this random-effects structure"
                .to_string(),
            "fit AGQ only for a model with exactly one scalar random-effects term".to_string(),
        ]);
        diagnostic
            .payload
            .insert("n_agq".to_string(), serde_json::json!(n_agq));
        diagnostic
            .payload
            .insert("reason".to_string(), serde_json::json!(reason));
        diagnostic.payload.insert(
            "random_effect_term_count".to_string(),
            serde_json::json!(self.lmm.reterms.len()),
        );
        diagnostic.payload.insert(
            "random_effect_terms".to_string(),
            serde_json::json!(term_summaries),
        );
        self.lmm.compiler_artifact.diagnostics.push(diagnostic);
    }

    pub(super) fn record_pirls_failure_diagnostic(&mut self, theta: &[f64], reason: &str) {
        self.lmm
            .compiler_artifact
            .diagnostics
            .retain(|diagnostic| diagnostic.code != DiagnosticCode::PirlsFailure);

        let affected_terms = self
            .lmm
            .reterms
            .iter()
            .map(random_effect_term_label)
            .collect::<Vec<_>>();
        let nonfinite_theta_indices = theta
            .iter()
            .enumerate()
            .filter_map(|(index, value)| (!value.is_finite()).then_some(index))
            .collect::<Vec<_>>();
        let mut diagnostic = Diagnostic::new(
            DiagnosticCode::PirlsFailure,
            DiagnosticSeverity::Error,
            DiagnosticStage::Optimization,
            "PIRLS failed while evaluating the final optimizer parameters for the GLMM; the fit was not completed.",
        )
        .with_affected_terms(affected_terms)
        .with_suggested_actions(vec![
            "inspect the optimizer return code and theta values before using this fit".to_string(),
            "try a different starting value, a simpler random-effects structure, or a lower optimizer step budget to localize the failure".to_string(),
            "check response domain, offsets, weights, and predictor scaling for invalid values".to_string(),
        ]);
        diagnostic
            .payload
            .insert("reason".to_string(), serde_json::json!(reason));
        diagnostic
            .payload
            .insert("theta_len".to_string(), serde_json::json!(theta.len()));
        diagnostic.payload.insert(
            "nonfinite_theta_indices".to_string(),
            serde_json::json!(nonfinite_theta_indices),
        );
        self.lmm.compiler_artifact.diagnostics.push(diagnostic);
    }

    /// Record a Warning when the inner PIRLS at the *final* optimizer θ did
    /// not reach its convergence tolerance within the iteration budget.
    ///
    /// The fit is still returned (mirroring MixedModels.jl, which also
    /// returns a model after a bounded PIRLS), but the non-convergence must
    /// not be *silent* (audit 03·H1): a downstream consumer can see this
    /// diagnostic instead of unknowingly trusting unverified conditional
    /// modes. Distinct from [`Self::record_pirls_failure_diagnostic`], which
    /// flags a hard PIRLS/linear-algebra failure that aborts the fit.
    pub(super) fn record_pirls_nonconvergence_diagnostic(&mut self, theta: &[f64]) {
        self.lmm
            .compiler_artifact
            .diagnostics
            .retain(|d| d.code != DiagnosticCode::OptimizerNonconvergence);

        let affected_terms = self
            .lmm
            .reterms
            .iter()
            .map(random_effect_term_label)
            .collect::<Vec<_>>();
        let mut diagnostic = Diagnostic::new(
            DiagnosticCode::OptimizerNonconvergence,
            DiagnosticSeverity::Warning,
            DiagnosticStage::Optimization,
            "the inner PIRLS conditional-mode solve did not reach its \
             convergence tolerance within the iteration budget at the final \
             optimizer parameters; the random-effect modes (and therefore the \
             Laplace/AGQ objective) are the best seen but unverified.",
        )
        .with_affected_terms(affected_terms)
        .with_suggested_actions(vec![
            "treat the conditional modes and objective as provisional and \
             cross-check against an alternate starting value"
                .to_string(),
            "simplify the random-effects structure or rescale predictors if \
             the GLMM surface is ill-conditioned near the optimum"
                .to_string(),
        ]);
        diagnostic
            .payload
            .insert("theta_len".to_string(), serde_json::json!(theta.len()));
        diagnostic
            .payload
            .insert("stage".to_string(), serde_json::json!("final_pirls"));
        self.lmm.compiler_artifact.diagnostics.push(diagnostic);
    }

    pub(super) fn reset_for_refit(&mut self, new_y: Option<&[f64]>) -> Result<()> {
        if let Some(new_y) = new_y {
            if new_y.len() != self.y.len() {
                return Err(MixedModelError::InvalidArgument(format!(
                    "Response length {} does not match model ({} observations)",
                    new_y.len(),
                    self.y.len()
                )));
            }
            validate_glmm_response_domain(self.family, self.link, new_y)?;
            let y_max = new_y.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let y_min = new_y.iter().copied().fold(f64::INFINITY, f64::min);
            if (y_max - y_min) < f64::EPSILON {
                return Err(MixedModelError::InvalidArgument(
                    "response is constant; GLMM refit requires variation in the response"
                        .to_string(),
                ));
            }

            let p = self.lmm.feterm.rank;
            for obs in 0..self.y.len() {
                let sw = if self.lmm.sqrtwts.is_empty() {
                    1.0
                } else {
                    self.lmm.sqrtwts[obs]
                };
                self.y[obs] = new_y[obs];
                self.lmm.y[obs] = new_y[obs];
                self.lmm.xy_mat.xy[(obs, p)] = new_y[obs];
                self.lmm.xy_mat.wtxy[(obs, p)] = sw * new_y[obs];
            }
            self.lmm.recompute_a_blocks()?;
        }

        let initial_theta = self.lmm.optsum.initial.clone();
        self.lmm.set_theta(&initial_theta)?;
        self.lmm.update_l()?;
        self.theta = initial_theta.clone();

        self.beta = DVector::zeros(self.lmm.feterm.rank);
        self.beta0 = self.beta.clone();
        for u in &mut self.u {
            u.fill(0.0);
        }
        for u0 in &mut self.u0 {
            u0.fill(0.0);
        }
        for b in &mut self.b {
            b.fill(0.0);
        }
        self.eta.fill(0.0);
        self.mu.fill(0.0);
        self.dispersion = 1.0;
        self.update_eta();

        self.lmm.optsum.finitial = f64::INFINITY;
        self.lmm.optsum.final_params = initial_theta;
        self.lmm.optsum.fmin = f64::INFINITY;
        self.lmm.optsum.feval = 0;
        self.lmm.optsum.return_value.clear();
        self.lmm.optsum.fit_log.clear();
        self.lmm.compiler_artifact.optimizer_certificate = None;
        self.lmm.compiler_artifact.glmm_fit_metadata = None;
        self.lmm.compiler_artifact.fixed_effect_covariance_matrix = None;
        self.lmm.compiler_artifact.effective_covariance.clear();
        self.pirls_profiled_optimum_certificate = None;
        Ok(())
    }

    pub(super) fn record_glmm_fit_metadata(&mut self) {
        let mut metadata = GlmmFitMetadata::from_opt_summary(&self.lmm.optsum);
        if let Some(theta) = self.negative_binomial_theta {
            metadata
                .family_parameters
                .insert("negative_binomial_theta".to_string(), theta);
            metadata
                .family_parameters
                .insert("negative_binomial_variance_power".to_string(), 2.0);
            metadata.family_parameter_sources.insert(
                "negative_binomial_theta".to_string(),
                if self.negative_binomial_estimate_theta {
                    "estimated".to_string()
                } else {
                    "fixed".to_string()
                },
            );
            metadata.family_parameter_sources.insert(
                "negative_binomial_variance_power".to_string(),
                "fixed".to_string(),
            );
        }
        self.record_fast_pirls_parity_scope_diagnostic(&metadata);
        self.record_pirls_profiled_optimum_certificate(&metadata);
        let inference_artifacts = self.glmm_fixed_effect_inference_artifacts(&metadata);
        let inference_availability =
            glmm_inference_availability_for_table(&metadata, &inference_artifacts.table);
        let covariance = inference_artifacts
            .covariance
            .unwrap_or_else(|| self.profiled_glmm_fixed_effect_covariance_matrix());
        self.lmm
            .compiler_artifact
            .model_boundary
            .inference_availability = inference_availability;
        self.lmm.compiler_artifact.glmm_fit_metadata = Some(metadata);
        self.lmm.compiler_artifact.fixed_effect_covariance_matrix = Some(covariance);
        self.lmm.compiler_artifact.fixed_effect_inference_table = Some(inference_artifacts.table);
    }

    /// Run the post-fit profiled-optimum certificate for profiled fast-PIRLS
    /// fits, store the outcome for prediction-variance gating, and record a
    /// provenance diagnostic either way.
    fn record_pirls_profiled_optimum_certificate(&mut self, metadata: &GlmmFitMetadata) {
        if !matches!(
            metadata.estimation_method.as_str(),
            "fast_pirls_profiled" | "fallback_fast_pirls"
        ) {
            self.pirls_profiled_optimum_certificate = None;
            return;
        }
        // Fit drivers can record metadata more than once for the same final
        // fit (e.g. a joint fallback re-labelling a profiled fit); the
        // certificate and its diagnostic are per-fit, not per-recording.
        if self.pirls_profiled_optimum_certificate.is_some() {
            return;
        }
        let outcome = self.certify_pirls_profiled_optimum();

        let mut diagnostic = Diagnostic::new(
            DiagnosticCode::SupportNote,
            DiagnosticSeverity::Info,
            DiagnosticStage::Certification,
            match &outcome {
                Ok(_) => "Fast-PIRLS profiled optimum certificate issued",
                Err(_) => "Fast-PIRLS profiled optimum certificate not issued",
            },
        );
        diagnostic.payload.insert(
            "glmm_pirls_profiled_optimum_certificate".to_string(),
            serde_json::json!(if outcome.is_ok() {
                "issued"
            } else {
                "not_issued"
            }),
        );
        diagnostic.payload.insert(
            "estimation_method".to_string(),
            serde_json::json!(metadata.estimation_method.as_str()),
        );
        match &outcome {
            Ok(certificate) => {
                diagnostic.payload.insert(
                    "gradient_max_abs".to_string(),
                    serde_json::json!(certificate.gradient_max_abs),
                );
                diagnostic.payload.insert(
                    "min_eigenvalue".to_string(),
                    serde_json::json!(certificate.min_eigenvalue),
                );
                diagnostic.payload.insert(
                    "condition_number".to_string(),
                    serde_json::json!(certificate.condition_number),
                );
                if !certificate.escalated_theta_indices.is_empty() {
                    diagnostic.payload.insert(
                        "escalated_theta_indices".to_string(),
                        serde_json::json!(certificate.escalated_theta_indices),
                    );
                }
                if !certificate.boundary_theta_indices.is_empty() {
                    diagnostic.payload.insert(
                        "boundary_theta_indices".to_string(),
                        serde_json::json!(certificate.boundary_theta_indices),
                    );
                }
            }
            Err(reason) => {
                diagnostic
                    .payload
                    .insert("reason".to_string(), serde_json::json!(reason));
            }
        }
        self.lmm.compiler_artifact.diagnostics.push(diagnostic);
        self.pirls_profiled_optimum_certificate = Some(outcome);
    }

    pub(super) fn record_negative_binomial_theta_estimation_metadata(
        &mut self,
        initial_theta: f64,
        final_theta: f64,
        update_iterations: usize,
        converged: bool,
    ) {
        if let Some(metadata) = &mut self.lmm.compiler_artifact.glmm_fit_metadata {
            metadata
                .family_parameters
                .insert("negative_binomial_theta_initial".to_string(), initial_theta);
            metadata.family_parameters.insert(
                "negative_binomial_theta_outer_iterations".to_string(),
                update_iterations as f64,
            );
            metadata.family_parameters.insert(
                "negative_binomial_theta_outer_converged".to_string(),
                if converged { 1.0 } else { 0.0 },
            );
            metadata
                .family_parameters
                .insert("negative_binomial_theta".to_string(), final_theta);
            metadata.family_parameter_sources.insert(
                "negative_binomial_theta".to_string(),
                "estimated".to_string(),
            );
            metadata.family_parameter_sources.insert(
                "negative_binomial_theta_initial".to_string(),
                "method_of_moments_or_caller_start".to_string(),
            );
            metadata.family_parameter_sources.insert(
                "negative_binomial_theta_outer_iterations".to_string(),
                "estimated".to_string(),
            );
            metadata.family_parameter_sources.insert(
                "negative_binomial_theta_outer_converged".to_string(),
                "estimated".to_string(),
            );
        }
    }

    fn record_fast_pirls_parity_scope_diagnostic(&mut self, metadata: &GlmmFitMetadata) {
        if metadata.estimation_method != "fast_pirls_profiled" {
            return;
        }
        let scope = "fast_pirls_not_lme4_joint_parity";
        if self
            .lmm
            .compiler_artifact
            .diagnostics
            .iter()
            .any(|diagnostic| {
                diagnostic.code == DiagnosticCode::SupportNote
                    && diagnostic
                        .payload
                        .get("glmm_parity_scope")
                        .and_then(serde_json::Value::as_str)
                        == Some(scope)
            })
        {
            return;
        }

        let mut diagnostic = Diagnostic::new(
            DiagnosticCode::SupportNote,
            DiagnosticSeverity::Info,
            DiagnosticStage::Certification,
            "Fast-PIRLS GLMM fit is not certified as lme4 joint-Laplace parity",
        )
        .with_suggested_actions(vec![
            "treat this fit as profiled fast-PIRLS evidence, not an lme4 joint-Laplace parity row"
                .to_string(),
            "consult the parity scorecard or downstream mismatch ledger before applying strict lme4 tolerances"
                .to_string(),
        ]);
        diagnostic
            .payload
            .insert("glmm_parity_scope".to_string(), serde_json::json!(scope));
        diagnostic.payload.insert(
            "scorecard_class".to_string(),
            serde_json::json!("documented_divergence"),
        );
        diagnostic.payload.insert(
            "external_engine_parity".to_string(),
            serde_json::json!("not_certified"),
        );
        diagnostic.payload.insert(
            "reference_gate".to_string(),
            serde_json::json!("lme4_joint_laplace"),
        );
        diagnostic.payload.insert(
            "estimation_method".to_string(),
            serde_json::json!(metadata.estimation_method.as_str()),
        );
        diagnostic.payload.insert(
            "objective_definition".to_string(),
            serde_json::json!(metadata.objective_definition.as_str()),
        );
        diagnostic.payload.insert(
            "response_constants".to_string(),
            serde_json::json!(metadata.response_constants.as_str()),
        );
        self.lmm.compiler_artifact.diagnostics.push(diagnostic);
    }

    pub(super) fn refresh_near_unit_random_effect_correlation_diagnostics(&mut self) {
        const NEAR_UNIT_CORR_THRESHOLD: f64 = 0.99;

        let varcorr = self.varcorr();
        let mut diagnostics = Vec::new();
        for component in &varcorr.components {
            for (offset, &corr) in component.correlations.iter().enumerate() {
                if corr.abs() < NEAR_UNIT_CORR_THRESHOLD {
                    continue;
                }
                let (row, col) = lower_triangle_pair(offset);
                let row_name = component
                    .names
                    .get(row)
                    .cloned()
                    .unwrap_or_else(|| format!("basis[{row}]"));
                let col_name = component
                    .names
                    .get(col)
                    .cloned()
                    .unwrap_or_else(|| format!("basis[{col}]"));
                let mut diagnostic = Diagnostic::new(
                    DiagnosticCode::NearUnitRandomEffectCorrelation,
                    DiagnosticSeverity::Warning,
                    DiagnosticStage::Certification,
                    format!(
                        "random-effect correlation for group {} between {} and {} is {:.3}; the fitted covariance is nearly one-dimensional",
                        component.group, col_name, row_name, corr
                    ),
                )
                .with_affected_terms(vec![component.group.clone()])
                .with_suggested_actions(vec![
                    "consider a zero-correlation (`||`) or reduced-rank random-effect structure".to_string(),
                    "treat correlation estimates and Hessian-based standard errors cautiously".to_string(),
                ]);
                diagnostic
                    .payload
                    .insert("group".to_string(), serde_json::json!(component.group));
                diagnostic
                    .payload
                    .insert("correlation".to_string(), serde_json::json!(corr));
                diagnostic.payload.insert(
                    "threshold".to_string(),
                    serde_json::json!(NEAR_UNIT_CORR_THRESHOLD),
                );
                diagnostics.push(diagnostic);
            }
        }

        if diagnostics.is_empty() {
            return;
        }

        self.lmm
            .compiler_artifact
            .diagnostics
            .extend(diagnostics.clone());
        if let Some(certificate) = &mut self.lmm.compiler_artifact.optimizer_certificate {
            certificate.diagnostics.extend(diagnostics);
        }
    }

    pub(super) fn refresh_binomial_separation_diagnostics(&mut self) {
        self.lmm
            .compiler_artifact
            .diagnostics
            .retain(|diagnostic| diagnostic.code != DiagnosticCode::BinomialSeparation);
        if let Some(certificate) = &mut self.lmm.compiler_artifact.optimizer_certificate {
            certificate
                .diagnostics
                .retain(|diagnostic| diagnostic.code != DiagnosticCode::BinomialSeparation);
        }

        let diagnostics = self.conservative_binomial_separation_diagnostics();
        if diagnostics.is_empty() {
            return;
        }

        self.lmm
            .compiler_artifact
            .diagnostics
            .extend(diagnostics.clone());
        if let Some(certificate) = &mut self.lmm.compiler_artifact.optimizer_certificate {
            certificate.diagnostics.extend(diagnostics);
        }
    }

    pub(super) fn conservative_binomial_separation_diagnostics(&self) -> Vec<Diagnostic> {
        if !matches!(self.family, Family::Bernoulli | Family::Binomial)
            || !self.y.iter().all(|value| is_binary_response(*value))
        {
            return Vec::new();
        }

        let mut diagnostics = Vec::new();
        for column_index in 0..self.lmm.feterm.rank {
            let column_name = self
                .lmm
                .feterm
                .cnames
                .get(column_index)
                .cloned()
                .unwrap_or_else(|| format!("fixed_effect[{column_index}]"));
            if is_intercept_column(&column_name) {
                continue;
            }

            let column_values = self.lmm.feterm.x.column(column_index);
            let Some(split) = binary_column_split(column_values.iter().copied()) else {
                continue;
            };

            let low_counts = outcome_counts_for_value(
                column_values.iter().copied(),
                self.y.iter().copied(),
                split.low,
            );
            let high_counts = outcome_counts_for_value(
                column_values.iter().copied(),
                self.y.iter().copied(),
                split.high,
            );

            if let Some(diagnostic) =
                separation_diagnostic_for_side(&column_name, split.low, low_counts, high_counts)
            {
                diagnostics.push(diagnostic);
            }
            if let Some(diagnostic) =
                separation_diagnostic_for_side(&column_name, split.high, high_counts, low_counts)
            {
                diagnostics.push(diagnostic);
            }
        }

        diagnostics
    }
}
