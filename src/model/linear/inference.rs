//! Fixed-effect inference for the linear mixed model: the coefficient table,
//! contrast/hypothesis tests (Wald, Satterthwaite, Kenward-Roger, bootstrap),
//! the fixed-effect covariance matrix and inference tables, and Cook's
//! distance. Moved verbatim from the former single-file `linear.rs`.

use super::*;

impl LinearMixedModel {
    /// Coefficient table for the fixed effects.
    ///
    /// Returns a [`CoefTable`] with one row per fixed-effects term (in the
    /// original, unpivoted column order) containing:
    /// - the estimate (`β`)
    /// - the standard error
    /// - the Wald z-statistic (`β / SE`)
    /// - the two-sided p-value from the standard normal distribution
    ///
    /// Mirrors `coeftable(m)` in MixedModels.jl / StatsModels.jl.  As in
    /// Julia, p-values use the z-distribution (large-sample approximation).
    pub fn coeftable(&self) -> CoefTable {
        let names = self.coef_names();
        let estimates: Vec<f64> = MixedModelFit::coef(self).iter().cloned().collect();
        let std_errors: Vec<f64> = self.stderror().iter().cloned().collect();
        CoefTable::new_with_p_value_policy(
            names,
            estimates,
            std_errors,
            self.fixed_effect_p_value_policy(),
        )
    }

    /// Coefficient table using a degrees-of-freedom-based inference method
    /// (Satterthwaite or Kenward-Roger) instead of the asymptotic Wald-z of
    /// [`coeftable`](Self::coeftable).
    ///
    /// Each row carries the method's statistic, its t-distribution
    /// denominator df, the t p-value, and a `method`/`statistic_name` label
    /// so downstream clients can see the table is not asymptotic Wald-z.
    /// `FixedEffectTestMethod::Auto` resolves the model's policy-preferred
    /// method; `AsymptoticWaldZ` is accepted and yields a Wald-z table with
    /// no df (equivalent in content to [`coeftable`](Self::coeftable)).
    pub fn coeftable_with_method(&self, method: FixedEffectTestMethod) -> CoefTable {
        let table =
            self.fixed_effect_contrast_inference_table(self.coefficient_hypotheses(), method);

        let n = table.rows.len();
        let mut names = Vec::with_capacity(n);
        let mut estimates = Vec::with_capacity(n);
        let mut std_errors = Vec::with_capacity(n);
        let mut statistics = Vec::with_capacity(n);
        let mut p_values = Vec::with_capacity(n);
        let mut p_value_reasons = Vec::with_capacity(n);
        let mut df = Vec::with_capacity(n);

        let mut resolved_method = FixedEffectInferenceMethod::NotComputed;
        let mut resolved_stat = FixedEffectStatisticName::T;

        for row in &table.rows {
            names.push(row.label.clone());
            estimates.push(row.estimate.unwrap_or(f64::NAN));
            std_errors.push(row.std_error.unwrap_or(f64::NAN));
            statistics.push(row.statistic.unwrap_or(f64::NAN));
            df.push(row.denominator_df);
            match row.p_value {
                Some(p) => {
                    p_values.push(p);
                    p_value_reasons.push(None);
                }
                None => {
                    p_values.push(f64::NAN);
                    p_value_reasons.push(Some(
                        row.reason
                            .clone()
                            .unwrap_or_else(|| "p-value unavailable".to_string()),
                    ));
                }
            }
            resolved_method = row.method;
            if let Some(name) = row.statistic_name {
                resolved_stat = name;
            }
        }

        CoefTable::from_df_inference(
            names,
            estimates,
            std_errors,
            statistics,
            p_values,
            p_value_reasons,
            df,
            fixed_effect_statistic_name_label(resolved_stat),
            fixed_effect_inference_method_label(resolved_method),
        )
    }

    /// Build one zero-valued single-coefficient hypothesis per fixed effect.
    pub fn coefficient_hypotheses(&self) -> Vec<FixedEffectHypothesis> {
        let names = self.coef_names();
        names
            .iter()
            .enumerate()
            .filter_map(|(index, name)| {
                FixedEffectHypothesis::single_coefficient(name.clone(), index, names.len()).ok()
            })
            .collect()
    }

    /// Test a fixed-effect contrast with the model's default method policy.
    pub fn test_contrast(&self, hypothesis: FixedEffectHypothesis) -> FixedEffectTest {
        self.test_contrast_with_method(hypothesis, FixedEffectTestMethod::Auto)
    }

    /// Test a fixed-effect contrast with an explicitly requested method.
    pub fn test_contrast_with_method(
        &self,
        hypothesis: FixedEffectHypothesis,
        requested_method: FixedEffectTestMethod,
    ) -> FixedEffectTest {
        let label = hypothesis.label.clone();
        let n_coefficients = self.coef_names().len();
        if hypothesis.n_coefficients() != n_coefficients {
            let reason = format!(
                "contrast has {} coefficient column(s), but the model has {n_coefficients}",
                hypothesis.n_coefficients()
            );
            return fixed_effect_test_unavailable(
                hypothesis,
                FixedContrastEstimability::not_assessed(label),
                InferenceStatus::Unsupported { reason },
            );
        }

        let beta = self.coef();
        let vcov = self.vcov();
        let estimates = (&hypothesis.l.values * &beta - &hypothesis.rhs.values)
            .iter()
            .copied()
            .collect::<Vec<_>>();
        let standard_errors = contrast_standard_errors(&hypothesis.l.values, &vcov);
        let statistics = estimates
            .iter()
            .zip(standard_errors.iter())
            .map(|(&estimate, se)| {
                se.and_then(|se| {
                    (se > 0.0 && se.is_finite() && estimate.is_finite()).then_some(estimate / se)
                })
            })
            .collect::<Vec<_>>();

        let estimability = assess_fixed_contrast_estimability(&hypothesis, &beta, &vcov);
        if estimability.status == EstimabilityStatus::NotEstimable {
            return FixedEffectTest {
                hypothesis,
                estimates,
                standard_errors,
                statistics,
                numerator_df: Some(1.0),
                denominator_df: None,
                p_values: vec![None; estimability.requested_rank.unwrap_or(1)],
                method: InferenceMethod::NotComputed {
                    reason: "contrast is not estimable under the fitted fixed-effect design"
                        .to_string(),
                },
                reliability: ReliabilityGrade::NotAvailable,
                status: InferenceStatus::NotEstimable {
                    reason: "contrast touches aliased or non-finite coefficient directions"
                        .to_string(),
                },
                estimability,
                notes: Vec::new(),
            };
        }

        if hypothesis.n_contrasts() != 1
            && matches!(requested_method, FixedEffectTestMethod::AsymptoticWaldZ)
        {
            let reason =
                "multi-df asymptotic Wald contrast tests are not implemented in this scaffold"
                    .to_string();
            return FixedEffectTest {
                hypothesis,
                estimates,
                standard_errors,
                statistics,
                numerator_df: Some(estimability.requested_rank.unwrap_or(0) as f64),
                denominator_df: None,
                p_values: vec![None; estimability.requested_rank.unwrap_or(0)],
                method: InferenceMethod::NotComputed {
                    reason: reason.clone(),
                },
                reliability: ReliabilityGrade::NotAvailable,
                status: InferenceStatus::Unsupported { reason },
                estimability,
                notes: Vec::new(),
            };
        }

        match requested_method {
            FixedEffectTestMethod::Auto => match self.fixed_effect_p_value_policy() {
                CoefTablePValuePolicy::AsymptoticWaldZ => {
                    let satterthwaite = self.satterthwaite_fixed_effect_test(
                        hypothesis.clone(),
                        estimates.clone(),
                        standard_errors.clone(),
                        statistics.clone(),
                        estimability.clone(),
                    );
                    if satterthwaite.status == InferenceStatus::Available
                        || satterthwaite.hypothesis.n_contrasts() != 1
                    {
                        satterthwaite
                    } else {
                        let mut wald = fixed_effect_test_asymptotic_wald_z(
                            hypothesis,
                            estimates,
                            standard_errors,
                            statistics,
                            estimability,
                        );
                        if let Some(reason) = fixed_effect_inference_reason(&satterthwaite) {
                            wald.notes
                                .push(format!("auto Satterthwaite unavailable: {reason}"));
                        }
                        wald
                    }
                }
                CoefTablePValuePolicy::Unavailable { reason } => {
                    fixed_effect_test_p_value_unavailable(
                        hypothesis,
                        estimates,
                        standard_errors,
                        statistics,
                        estimability,
                        reason,
                    )
                }
            },
            FixedEffectTestMethod::AsymptoticWaldZ => match self.fixed_effect_p_value_policy() {
                CoefTablePValuePolicy::AsymptoticWaldZ => fixed_effect_test_asymptotic_wald_z(
                    hypothesis,
                    estimates,
                    standard_errors,
                    statistics,
                    estimability,
                ),
                CoefTablePValuePolicy::Unavailable { reason } => {
                    fixed_effect_test_p_value_unavailable(
                        hypothesis,
                        estimates,
                        standard_errors,
                        statistics,
                        estimability,
                        reason,
                    )
                }
            },
            FixedEffectTestMethod::Satterthwaite => self.satterthwaite_fixed_effect_test(
                hypothesis,
                estimates,
                standard_errors,
                statistics,
                estimability,
            ),
            FixedEffectTestMethod::KenwardRoger => self.kenward_roger_fixed_effect_test(
                hypothesis,
                estimates,
                standard_errors,
                statistics,
                estimability,
            ),
            FixedEffectTestMethod::ParametricBootstrap => fixed_effect_test_not_assessed_with_method(
                hypothesis,
                estimates,
                standard_errors,
                statistics,
                InferenceMethod::ParametricBootstrap,
                estimability,
                "parametric bootstrap fixed-effect inference requires a certified fixed_effect_null bootstrap payload; call test_contrast_with_bootstrap_payload with replicate accounting, failed-refit policy, Monte Carlo uncertainty, and reproducibility state"
                    .to_string(),
            ),
        }
    }

    /// Test a fixed-effect contrast using a certified bootstrap payload.
    pub fn test_contrast_with_bootstrap_payload(
        &self,
        hypothesis: FixedEffectHypothesis,
        payload: &BootstrapRunPayload,
    ) -> FixedEffectTest {
        let label = hypothesis.label.clone();
        let n_coefficients = self.coef_names().len();
        if hypothesis.n_coefficients() != n_coefficients {
            let reason = format!(
                "contrast has {} coefficient column(s), but the model has {n_coefficients}",
                hypothesis.n_coefficients()
            );
            return fixed_effect_test_unavailable(
                hypothesis,
                FixedContrastEstimability::not_assessed(label),
                InferenceStatus::Unsupported { reason },
            );
        }

        let beta = self.coef();
        let vcov = self.vcov();
        let estimates = (&hypothesis.l.values * &beta - &hypothesis.rhs.values)
            .iter()
            .copied()
            .collect::<Vec<_>>();
        let standard_errors = contrast_standard_errors(&hypothesis.l.values, &vcov);
        let statistics = estimates
            .iter()
            .zip(standard_errors.iter())
            .map(|(&estimate, se)| {
                se.and_then(|se| {
                    (se > 0.0 && se.is_finite() && estimate.is_finite()).then_some(estimate / se)
                })
            })
            .collect::<Vec<_>>();

        let estimability = assess_fixed_contrast_estimability(&hypothesis, &beta, &vcov);
        if estimability.status == EstimabilityStatus::NotEstimable {
            return FixedEffectTest {
                hypothesis,
                estimates,
                standard_errors,
                statistics,
                numerator_df: None,
                denominator_df: None,
                p_values: vec![None; estimability.requested_rank.unwrap_or(1)],
                method: InferenceMethod::ParametricBootstrap,
                reliability: ReliabilityGrade::NotAvailable,
                status: InferenceStatus::NotEstimable {
                    reason: "bootstrap fixed-effect inference requires an estimable contrast"
                        .to_string(),
                },
                estimability,
                notes: Vec::new(),
            };
        }

        self.bootstrap_fixed_effect_test_from_payload(
            hypothesis,
            estimates,
            standard_errors,
            statistics,
            estimability,
            payload,
        )
    }

    /// Build one fixed-effect inference row from a bootstrap payload.
    pub fn fixed_effect_bootstrap_inference_row(
        &self,
        kind: FixedEffectInferenceRowKind,
        hypothesis: FixedEffectHypothesis,
        payload: &BootstrapRunPayload,
    ) -> FixedEffectInferenceRow {
        let mut row = fixed_effect_test_to_inference_row(
            kind,
            self.test_contrast_with_bootstrap_payload(hypothesis, payload),
        );
        attach_bootstrap_details(&mut row, payload, None);
        row
    }

    /// Build an inference table for user-supplied fixed-effect hypotheses.
    pub fn fixed_effect_contrast_inference_table(
        &self,
        hypotheses: Vec<FixedEffectHypothesis>,
        method: FixedEffectTestMethod,
    ) -> FixedEffectInferenceTable {
        let rows = hypotheses
            .into_iter()
            .map(|hypothesis| {
                self.fixed_effect_contrast_inference_row(
                    FixedEffectInferenceRowKind::Contrast,
                    hypothesis,
                    method,
                )
            })
            .collect();
        FixedEffectInferenceTable::new(rows)
    }

    /// Build one inference row for a user-supplied fixed-effect hypothesis.
    pub fn fixed_effect_contrast_inference_row(
        &self,
        kind: FixedEffectInferenceRowKind,
        hypothesis: FixedEffectHypothesis,
        method: FixedEffectTestMethod,
    ) -> FixedEffectInferenceRow {
        fixed_effect_test_to_inference_row(kind, self.test_contrast_with_method(hypothesis, method))
    }

    /// Run fixed-effect null bootstrap inference for a set of hypotheses.
    pub fn fixed_effect_null_bootstrap_inference_table(
        &self,
        hypotheses: Vec<FixedEffectHypothesis>,
        options: FixedEffectBootstrapOptions,
    ) -> FixedEffectInferenceTable {
        let rows = hypotheses
            .into_iter()
            .map(|hypothesis| {
                self.fixed_effect_null_bootstrap_inference_row(
                    FixedEffectInferenceRowKind::Contrast,
                    hypothesis,
                    &options,
                )
            })
            .collect();
        FixedEffectInferenceTable::new(rows)
    }

    /// Run fixed-effect null bootstrap inference for one hypothesis.
    pub fn fixed_effect_null_bootstrap_inference_row(
        &self,
        kind: FixedEffectInferenceRowKind,
        hypothesis: FixedEffectHypothesis,
        options: &FixedEffectBootstrapOptions,
    ) -> FixedEffectInferenceRow {
        let target = match self.fixed_effect_null_bootstrap_target(&hypothesis) {
            Ok(target) => target,
            Err(error) => {
                let mut test = self.test_contrast_with_method(
                    hypothesis,
                    FixedEffectTestMethod::ParametricBootstrap,
                );
                test.status = InferenceStatus::NotAssessed {
                    reason: format!("bootstrap_null_target_unavailable: {error}"),
                };
                return fixed_effect_test_to_inference_row(kind, test);
            }
        };

        match self.fixed_effect_null_bootstrap_payload(&hypothesis, &target, options) {
            Ok(payload) => {
                let mut row = self.fixed_effect_bootstrap_inference_row(kind, hypothesis, &payload);
                attach_bootstrap_details(&mut row, &payload, Some(&target));
                row
            }
            Err(error) => {
                let mut test = self.test_contrast_with_method(
                    hypothesis,
                    FixedEffectTestMethod::ParametricBootstrap,
                );
                test.status = InferenceStatus::NotAssessed {
                    reason: format!("bootstrap_replicate_accounting_unavailable: {error}"),
                };
                fixed_effect_test_to_inference_row(kind, test)
            }
        }
    }

    fn fixed_effect_null_bootstrap_payload(
        &self,
        hypothesis: &FixedEffectHypothesis,
        target: &FixedEffectNullBootstrapTarget,
        options: &FixedEffectBootstrapOptions,
    ) -> Result<BootstrapRunPayload> {
        let mut rng = match options.seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => rand::rngs::StdRng::from_entropy(),
        };
        let mut fits = Vec::with_capacity(options.requested_replicates);
        let mut statistics = Vec::with_capacity(options.requested_replicates);
        let mut last_progress = 0usize;

        for replicate in 0..options.requested_replicates {
            if let Some(callback) = &self.progress_callback {
                callback.report_if_due(
                    FitProgressPhase::Bootstrap,
                    replicate + 1,
                    Some(options.requested_replicates),
                    &mut last_progress,
                )?;
            }
            let y_sim = self.simulate_fixed_effect_null(&mut rng, target)?;
            let mut work = self.clone();
            work.suppress_derivative_diagnostics = true;
            match work.refit(y_sim.as_slice()) {
                Ok(()) => {
                    statistics.push(
                        fixed_effect_bootstrap_statistic(&work, hypothesis)
                            .map(|statistic| statistic.value)
                            .unwrap_or(f64::NAN),
                    );
                    fits.push(BootstrapReplicate {
                        objective: work.objective(),
                        sigma: work.sigma(),
                        beta: work.beta(),
                        se: work.stderror(),
                        theta: work.theta(),
                    });
                }
                Err(error @ MixedModelError::Interrupted(_)) => return Err(error),
                Err(_) => {
                    let beta = work.beta();
                    statistics.push(f64::NAN);
                    fits.push(BootstrapReplicate {
                        objective: f64::NAN,
                        sigma: f64::NAN,
                        se: DVector::from_element(beta.len(), f64::NAN),
                        beta,
                        theta: work.theta(),
                    });
                    if options.failed_refit_policy == BootstrapFailedRefitPolicy::Abort {
                        break;
                    }
                }
            }
        }

        let bootstrap = MixedModelBootstrap { fits };
        let p_value = fixed_effect_bootstrap_statistic(self, hypothesis).and_then(|observed| {
            let finite = statistics
                .iter()
                .copied()
                .filter(|value| value.is_finite())
                .collect::<Vec<_>>();
            (!finite.is_empty()).then(|| {
                let extreme = finite
                    .iter()
                    .filter(|&&value| value >= observed.value)
                    .count();
                (extreme as f64 + 1.0) / (finite.len() as f64 + 1.0)
            })
        });
        let seed_record = options
            .seed
            .map(BootstrapSeedRecord::std_rng)
            .unwrap_or_else(BootstrapSeedRecord::unspecified);
        let metadata = bootstrap.run_metadata_for_model(
            self,
            target.target.clone(),
            options.requested_replicates,
            options.failed_refit_policy,
            seed_record,
            BootstrapRefitOptions::from_model(self),
            Some(hypothesis.label.clone()),
            Some(&statistics),
            p_value,
        );
        Ok(bootstrap.into_run_payload_with_statistics(metadata, statistics))
    }

    /// Build a cluster-resampling full-model bootstrap payload for one contrast.
    pub fn cluster_resample_full_model_contrast_payload(
        &self,
        data: &DataFrame,
        group: &str,
        hypothesis: &FixedEffectHypothesis,
        options: &FixedEffectBootstrapOptions,
        levels: &[f64],
    ) -> Result<BootstrapRunPayload> {
        if data.nrow() != self.nobs() {
            return Err(MixedModelError::InvalidArgument(format!(
                "cluster bootstrap data has {} rows, but the fitted model has {} observations",
                data.nrow(),
                self.nobs()
            )));
        }
        if !self.reterms.iter().any(|term| term.grouping_name == group) {
            return Err(MixedModelError::InvalidArgument(format!(
                "cluster bootstrap group `{group}` is not a random-effect grouping factor in the fitted model"
            )));
        }
        let n_coefficients = self.coef_names().len();
        if hypothesis.n_coefficients() != n_coefficients {
            return Err(MixedModelError::DimensionMismatch(format!(
                "cluster bootstrap contrast has {} coefficient column(s), but the model has {n_coefficients}",
                hypothesis.n_coefficients()
            )));
        }
        if hypothesis.n_contrasts() != 1 {
            return Err(MixedModelError::InvalidArgument(
                "cluster bootstrap intervals are currently certified only for scalar contrasts"
                    .to_string(),
            ));
        }
        if levels.is_empty() {
            return Err(MixedModelError::InvalidArgument(
                "cluster bootstrap intervals require at least one confidence level".to_string(),
            ));
        }

        let mut rng = match options.seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => rand::rngs::StdRng::from_entropy(),
        };
        let mut fits = Vec::with_capacity(options.requested_replicates);
        let mut statistics = Vec::with_capacity(options.requested_replicates);
        let mut distinct_counts = Vec::with_capacity(options.requested_replicates);
        let mut duplicate_counts = Vec::with_capacity(options.requested_replicates);
        let mut last_progress = 0usize;

        for replicate in 0..options.requested_replicates {
            if let Some(callback) = &self.progress_callback {
                callback.report_if_due(
                    FitProgressPhase::Bootstrap,
                    replicate + 1,
                    Some(options.requested_replicates),
                    &mut last_progress,
                )?;
            }
            let (resampled, draw) = data.cluster_resample(group, &mut rng)?;
            distinct_counts.push(draw.distinct_sampled_level_count);
            duplicate_counts.push(draw.duplicate_count);

            let mut work = match LinearMixedModel::new(self.formula.clone(), &resampled, None) {
                Ok(model) => model,
                Err(_) => {
                    statistics.push(f64::NAN);
                    fits.push(failed_bootstrap_replicate_like(self));
                    if options.failed_refit_policy == BootstrapFailedRefitPolicy::Abort {
                        break;
                    }
                    continue;
                }
            };
            work.progress_callback = self.progress_callback.clone();

            match work.fit(self.optsum.reml) {
                Ok(_) => {
                    statistics
                        .push(scalar_contrast_estimate(&work, hypothesis).unwrap_or(f64::NAN));
                    fits.push(BootstrapReplicate {
                        objective: work.objective(),
                        sigma: work.sigma(),
                        beta: work.beta(),
                        se: work.stderror(),
                        theta: work.theta(),
                    });
                }
                Err(error @ MixedModelError::Interrupted(_)) => return Err(error),
                Err(_) => {
                    statistics.push(f64::NAN);
                    fits.push(failed_bootstrap_replicate_like(self));
                    if options.failed_refit_policy == BootstrapFailedRefitPolicy::Abort {
                        break;
                    }
                }
            }
        }

        let observed = scalar_contrast_estimate(self, hypothesis).ok_or_else(|| {
            MixedModelError::InvalidArgument(
                "cluster bootstrap intervals require a finite observed scalar contrast".to_string(),
            )
        })?;
        let intervals = bootstrap_scalar_percentile_intervals(
            &hypothesis.label,
            &statistics,
            observed,
            levels,
        )?;
        let bootstrap = MixedModelBootstrap { fits };
        let seed_record = options
            .seed
            .map(BootstrapSeedRecord::std_rng)
            .unwrap_or_else(BootstrapSeedRecord::unspecified);
        let mut metadata = bootstrap.run_metadata_for_model(
            self,
            BootstrapTarget::cluster_resample(format!(
                "{} cluster resample by {group}",
                hypothesis.label
            )),
            options.requested_replicates,
            options.failed_refit_policy,
            seed_record,
            BootstrapRefitOptions::from_model(self),
            Some(hypothesis.label.clone()),
            Some(&statistics),
            None,
        );
        metadata.notes.push(
            "cluster_resample is an estimator-distribution target; it does not certify fixed-effect hypothesis-test p-values"
                .to_string(),
        );
        metadata.notes.push(format!(
            "cluster_resample group={group}, relabeling_policy=replicate_local_unique_levels"
        ));
        if let (Some(min_distinct), Some(max_duplicates)) =
            (distinct_counts.iter().min(), duplicate_counts.iter().max())
        {
            metadata.notes.push(format!(
                "cluster_resample draw summary: min_distinct_sampled_levels={min_distinct}, max_duplicate_count={max_duplicates}"
            ));
        }

        Ok(bootstrap
            .into_run_payload_with_statistics_and_intervals(metadata, statistics, intervals))
    }

    /// Build one fixed-effect term hypothesis per compiler-audited term.
    pub fn fixed_effect_term_hypotheses(&self) -> Vec<FixedEffectHypothesis> {
        self.fixed_effect_term_hypotheses_for_type(FixedEffectTermTestType::TypeIII)
    }

    /// Build fixed-effect term hypotheses with explicit ANOVA-style term semantics.
    ///
    /// Type III preserves the existing coefficient-block hypothesis for each
    /// term. Type I and Type II use the fitted model matrix cross-product to
    /// build sequential and marginal contrast bases, respectively, following
    /// the Doolittle contrast construction used by lmerTest.
    pub fn fixed_effect_term_hypotheses_for_type(
        &self,
        term_test_type: FixedEffectTermTestType,
    ) -> Vec<FixedEffectHypothesis> {
        let term_indices = self.fixed_effect_term_index_sets();
        if term_indices.is_empty() {
            return Vec::new();
        }
        match term_test_type {
            FixedEffectTermTestType::TypeI => {
                self.fixed_effect_type_i_term_hypotheses(&term_indices)
            }
            FixedEffectTermTestType::TypeII => {
                self.fixed_effect_type_ii_term_hypotheses(&term_indices)
            }
            FixedEffectTermTestType::TypeIII => term_indices
                .iter()
                .filter_map(|(term, indices)| {
                    fixed_effect_identity_hypothesis(term, indices, self.coef_names().len())
                })
                .collect(),
        }
    }

    fn fixed_effect_term_index_sets(&self) -> Vec<(String, Vec<usize>)> {
        let names = self.coef_names();
        let Some(audit) = self.compiler_artifact.design_audit.as_ref() else {
            return Vec::new();
        };
        audit
            .fixed_effects
            .terms
            .iter()
            .filter_map(|term| {
                let indices = audit
                    .fixed_effects
                    .columns
                    .iter()
                    .filter(|column| column.source_term == term.term)
                    .filter_map(|column| names.iter().position(|name| name == &column.name))
                    .collect::<Vec<_>>();
                if indices.is_empty() {
                    return None;
                }
                Some((term.term.clone(), indices))
            })
            .collect()
    }

    fn fixed_effect_type_i_term_hypotheses(
        &self,
        term_indices: &[(String, Vec<usize>)],
    ) -> Vec<FixedEffectHypothesis> {
        let p = self.coef_names().len();
        if self.feterm.x.ncols() != p || p == 0 {
            return Vec::new();
        }
        let basis = doolittle_contrast_basis(&self.feterm.x);
        term_indices
            .iter()
            .filter_map(|(term, indices)| fixed_effect_basis_hypothesis(term, indices, &basis))
            .collect()
    }

    fn fixed_effect_type_ii_term_hypotheses(
        &self,
        term_indices: &[(String, Vec<usize>)],
    ) -> Vec<FixedEffectHypothesis> {
        let p = self.coef_names().len();
        if self.feterm.x.ncols() != p || p == 0 {
            return Vec::new();
        }
        let mut col_terms = vec![String::new(); p];
        for (term, indices) in term_indices {
            for &index in indices {
                if index < p {
                    col_terms[index] = term.clone();
                }
            }
        }
        term_indices
            .iter()
            .filter_map(|(term, _indices)| {
                let contained_terms = fixed_effect_terms_containing(term, term_indices);
                fixed_effect_type_ii_hypothesis(term, &self.feterm.x, &col_terms, &contained_terms)
            })
            .collect()
    }

    /// Build an inference table for compiler-audited fixed-effect terms.
    pub fn fixed_effect_term_inference_table(
        &self,
        method: FixedEffectTestMethod,
    ) -> FixedEffectInferenceTable {
        self.fixed_effect_term_inference_table_for_type(method, FixedEffectTermTestType::TypeIII)
    }

    /// Build an inference table for compiler-audited fixed-effect terms with
    /// explicit Type I, Type II, or Type III term-test semantics.
    pub fn fixed_effect_term_inference_table_for_type(
        &self,
        method: FixedEffectTestMethod,
        term_test_type: FixedEffectTermTestType,
    ) -> FixedEffectInferenceTable {
        let rows = self
            .fixed_effect_term_hypotheses_for_type(term_test_type)
            .into_iter()
            .map(|hypothesis| {
                let mut row = fixed_effect_test_to_inference_row(
                    FixedEffectInferenceRowKind::Term,
                    self.test_contrast_with_method(hypothesis, method),
                );
                row.notes.push(format!(
                    "fixed-effect term test type: {}",
                    fixed_effect_term_test_type_label(term_test_type)
                ));
                row
            })
            .collect();
        FixedEffectInferenceTable::new(rows)
    }

    fn satterthwaite_fixed_effect_reliability(&self, denominator_df: f64) -> ReliabilityGrade {
        if !denominator_df.is_finite() || denominator_df <= 2.0 {
            return ReliabilityGrade::Low;
        }

        let Some(certificate) = &self.compiler_artifact.optimizer_certificate else {
            return ReliabilityGrade::Low;
        };

        let clean_interior = certificate.status == FitStatus::ConvergedInterior
            && certificate.evidence.optimizer_stop.acceptable_stop
            && certificate.evidence.parameter_space.n_boundary == 0
            && !self.theta_at_lower_bound()
            && !self.has_reduced_effective_covariance();
        let finite_difference_diagnostics = matches!(
            certificate.evidence.gradient.method,
            EvidenceMethod::Exact | EvidenceMethod::FiniteDifference
        ) && matches!(
            certificate.evidence.hessian.method,
            EvidenceMethod::Exact | EvidenceMethod::FiniteDifference
        );
        let hessian_positive_on_active_space = certificate
            .evidence
            .hessian
            .min_eigenvalue
            .is_some_and(|value| value.is_finite() && value > 0.0)
            && certificate.evidence.hessian.rank
                == Some(certificate.evidence.parameter_space.n_free);
        let no_failed_checks = certificate.checks.iter().all(|check| {
            !matches!(
                check,
                CertificateCheck::DerivativeMismatch { .. } | CertificateCheck::Failed { .. }
            )
        });

        if clean_interior
            && finite_difference_diagnostics
            && hessian_positive_on_active_space
            && no_failed_checks
        {
            ReliabilityGrade::Moderate
        } else {
            ReliabilityGrade::Low
        }
    }

    fn satterthwaite_fixed_effect_test(
        &self,
        hypothesis: FixedEffectHypothesis,
        estimates: Vec<f64>,
        standard_errors: Vec<Option<f64>>,
        statistics: Vec<Option<f64>>,
        estimability: FixedContrastEstimability,
    ) -> FixedEffectTest {
        use statrs::distribution::{ContinuousCDF, FisherSnedecor, StudentsT};

        let method = InferenceMethod::Satterthwaite;
        if self.residual_source
            == crate::model::summary_estimates::ResidualSource::FixedSamplingVariance
        {
            return fixed_effect_test_not_assessed_with_method(
                hypothesis,
                estimates,
                standard_errors,
                statistics,
                method,
                estimability,
                "summary-estimate fit (residual sampling variances fixed); \
                 finite-sample methods are undefined when sigma is not estimated"
                    .to_string(),
            );
        }

        let mut varpar = self.theta();
        varpar.push(self.sigma());
        let mut evaluator = self.clone();
        let jacobian = match evaluator.jac_vcov_beta_varpar(&varpar) {
            Ok(jacobian) => jacobian,
            Err(error) => {
                return fixed_effect_test_not_assessed_with_method(
                    hypothesis,
                    estimates,
                    standard_errors,
                    statistics,
                    method,
                    estimability,
                    format!("Satterthwaite fixed-effect inference could not compute vcov_beta derivatives: {error}"),
                );
            }
        };
        let vcov_varpar = match evaluator.vcov_varpar(&varpar, self.optsum.reml) {
            Ok(estimate) => estimate,
            Err(error) => {
                return fixed_effect_test_not_assessed_with_method(
                    hypothesis,
                    estimates,
                    standard_errors,
                    statistics,
                    method,
                    estimability,
                    format!("Satterthwaite fixed-effect inference could not estimate vcov_varpar: {error}"),
                );
            }
        };

        if hypothesis.n_contrasts() != 1 {
            let vcov = self.vcov();
            let contrast_cov = symmetrize_matrix(
                &(&hypothesis.l.values * &vcov * hypothesis.l.values.transpose()),
            );
            if !matrix_is_finite(&contrast_cov) {
                return fixed_effect_test_not_assessed_with_method(
                    hypothesis,
                    estimates,
                    standard_errors,
                    statistics,
                    method,
                    estimability,
                    "Satterthwaite fixed-effect inference produced a non-finite contrast covariance"
                        .to_string(),
                );
            }
            let eig = SymmetricEigen::new(contrast_cov.clone());
            let max_eigen = eig
                .eigenvalues
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max)
                .max(0.0);
            let tolerance = (1.0e-8 * max_eigen).max(0.0);
            let positive = eig
                .eigenvalues
                .iter()
                .enumerate()
                .filter_map(|(index, &value)| (value > tolerance).then_some((index, value)))
                .collect::<Vec<_>>();
            let q = positive.len();
            if q == 0 {
                return fixed_effect_test_not_assessed_with_method(
                    hypothesis,
                    estimates,
                    standard_errors,
                    statistics,
                    method,
                    estimability,
                    "Satterthwaite fixed-effect inference found zero positive contrast-covariance directions"
                        .to_string(),
                );
            }

            let estimate_vector = DVector::from_column_slice(&estimates);
            let mut f_numerator = 0.0;
            let mut direction_dfs = Vec::with_capacity(q);
            for (eig_index, eig_value) in positive {
                let eigen_direction = eig.eigenvectors.column(eig_index).transpose();
                let contrast_direction = &eigen_direction * &hypothesis.l.values;
                let rotated_estimate = (&eigen_direction * &estimate_vector)[0];
                f_numerator += rotated_estimate * rotated_estimate / eig_value;
                let gradient = jacobian
                    .iter()
                    .map(|derivative| {
                        let value =
                            &contrast_direction * derivative * contrast_direction.transpose();
                        value[(0, 0)]
                    })
                    .collect::<Vec<_>>();
                if gradient.iter().any(|value| !value.is_finite()) {
                    return fixed_effect_test_not_assessed_with_method(
                        hypothesis,
                        estimates,
                        standard_errors,
                        statistics,
                        method,
                        estimability,
                        "Satterthwaite fixed-effect inference produced a non-finite multi-df variance-gradient component"
                            .to_string(),
                    );
                }
                let gradient = DVector::from_vec(gradient);
                let denom = (gradient.transpose() * &vcov_varpar.covariance * &gradient)[(0, 0)];
                if !denom.is_finite() || denom <= 0.0 {
                    return fixed_effect_test_not_assessed_with_method(
                        hypothesis,
                        estimates,
                        standard_errors,
                        statistics,
                        method,
                        estimability,
                        "Satterthwaite fixed-effect inference requires finite positive denominator variance for every multi-df direction"
                            .to_string(),
                    );
                }
                let df = 2.0 * eig_value * eig_value / denom;
                if !df.is_finite() || df <= 0.0 {
                    return fixed_effect_test_not_assessed_with_method(
                        hypothesis,
                        estimates,
                        standard_errors,
                        statistics,
                        method,
                        estimability,
                        "Satterthwaite fixed-effect inference produced a non-finite multi-df denominator component"
                            .to_string(),
                    );
                }
                direction_dfs.push(df);
            }
            let denominator_df = match satterthwaite_f_denominator_df(&direction_dfs, 1.0e-8) {
                Some(df) => df,
                None => {
                    return fixed_effect_test_not_assessed_with_method(
                        hypothesis,
                        estimates,
                        standard_errors,
                        statistics,
                        method,
                        estimability,
                        "Satterthwaite fixed-effect inference could not combine multi-df denominator df components"
                            .to_string(),
                    );
                }
            };
            let f_statistic = f_numerator / q as f64;
            if !f_statistic.is_finite() || f_statistic < 0.0 {
                return fixed_effect_test_not_assessed_with_method(
                    hypothesis,
                    estimates,
                    standard_errors,
                    statistics,
                    method,
                    estimability,
                    "Satterthwaite fixed-effect inference produced a non-finite F statistic"
                        .to_string(),
                );
            }
            let p_value = match FisherSnedecor::new(q as f64, denominator_df) {
                Ok(f_dist) => Some(1.0 - f_dist.cdf(f_statistic)),
                Err(error) => {
                    return fixed_effect_test_not_assessed_with_method(
                        hypothesis,
                        estimates,
                        standard_errors,
                        statistics,
                        method,
                        estimability,
                        format!("Satterthwaite fixed-effect inference could not construct F distribution: {error}"),
                    );
                }
            };

            let mut notes = vec![
                "Satterthwaite multi-df F row computed from eigen-directions of L V_beta L' and finite-difference vcov_beta Jacobian over varpar"
                    .to_string(),
            ];
            if q < hypothesis.n_contrasts() {
                notes.push(format!(
                    "Satterthwaite restriction matrix effective rank {q} is lower than {} submitted row(s)",
                    hypothesis.n_contrasts()
                ));
            }
            notes.extend(vcov_varpar.notes);

            return FixedEffectTest {
                hypothesis,
                estimates,
                standard_errors,
                statistics: vec![Some(f_statistic)],
                numerator_df: Some(q as f64),
                denominator_df: Some(denominator_df),
                p_values: vec![p_value],
                method,
                reliability: self.satterthwaite_fixed_effect_reliability(denominator_df),
                status: InferenceStatus::Available,
                estimability,
                notes,
            };
        }

        let Some(std_error) = standard_errors.first().copied().flatten() else {
            return fixed_effect_test_not_assessed_with_method(
                hypothesis,
                estimates,
                standard_errors,
                statistics,
                method,
                estimability,
                "Satterthwaite fixed-effect inference requires an available fixed-effect standard error"
                    .to_string(),
            );
        };
        let var_con = std_error * std_error;
        if !var_con.is_finite() || var_con <= 0.0 {
            return fixed_effect_test_not_assessed_with_method(
                hypothesis,
                estimates,
                standard_errors,
                statistics,
                method,
                estimability,
                "Satterthwaite fixed-effect inference requires a finite positive contrast variance"
                    .to_string(),
            );
        }

        let gradient = jacobian
            .iter()
            .map(|derivative| contrast_row_quadratic_form(&hypothesis.l.values, 0, derivative))
            .collect::<Vec<_>>();
        if gradient.iter().any(|value| !value.is_finite()) {
            return fixed_effect_test_not_assessed_with_method(
                hypothesis,
                estimates,
                standard_errors,
                statistics,
                method,
                estimability,
                "Satterthwaite fixed-effect inference produced a non-finite variance-gradient component"
                    .to_string(),
            );
        }

        let gradient = DVector::from_vec(gradient);
        let satt_denom = (gradient.transpose() * &vcov_varpar.covariance * &gradient)[(0, 0)];
        if !satt_denom.is_finite() || satt_denom <= 0.0 {
            return fixed_effect_test_not_assessed_with_method(
                hypothesis,
                estimates,
                standard_errors,
                statistics,
                method,
                estimability,
                "Satterthwaite fixed-effect inference requires a finite positive denominator variance"
                    .to_string(),
            );
        }

        let denominator_df = 2.0 * var_con * var_con / satt_denom;
        if !denominator_df.is_finite() || denominator_df <= 0.0 {
            return fixed_effect_test_not_assessed_with_method(
                hypothesis,
                estimates,
                standard_errors,
                statistics,
                method,
                estimability,
                "Satterthwaite fixed-effect inference produced a non-finite denominator df"
                    .to_string(),
            );
        }

        let statistic = estimates[0] / std_error;
        let p_value = match StudentsT::new(0.0, 1.0, denominator_df) {
            Ok(t_dist) => Some(2.0 * (1.0 - t_dist.cdf(statistic.abs()))),
            Err(error) => {
                return fixed_effect_test_not_assessed_with_method(
                    hypothesis,
                    estimates,
                    standard_errors,
                    statistics,
                    method,
                    estimability,
                    format!("Satterthwaite fixed-effect inference could not construct Student-t distribution: {error}"),
                );
            }
        };

        let mut notes = vec![
            "Satterthwaite denominator df computed from finite-difference vcov_beta Jacobian and deviance Hessian over varpar"
                .to_string(),
        ];
        notes.extend(vcov_varpar.notes);

        FixedEffectTest {
            hypothesis,
            estimates,
            standard_errors,
            statistics: vec![Some(statistic)],
            numerator_df: None,
            denominator_df: Some(denominator_df),
            p_values: vec![p_value],
            method,
            reliability: self.satterthwaite_fixed_effect_reliability(denominator_df),
            status: InferenceStatus::Available,
            estimability,
            notes,
        }
    }

    fn kenward_roger_fixed_effect_test(
        &self,
        hypothesis: FixedEffectHypothesis,
        estimates: Vec<f64>,
        standard_errors: Vec<Option<f64>>,
        statistics: Vec<Option<f64>>,
        estimability: FixedContrastEstimability,
    ) -> FixedEffectTest {
        use statrs::distribution::{ContinuousCDF, FisherSnedecor, StudentsT};

        let method = InferenceMethod::KenwardRoger;
        if !self.optsum.reml {
            return fixed_effect_test_not_assessed_with_method(
                hypothesis,
                estimates,
                standard_errors,
                statistics,
                method,
                estimability,
                "Kenward-Roger fixed-effect inference is certified only for REML LMM fits"
                    .to_string(),
            );
        }

        let adjusted = match self.kenward_roger_adjusted_vcov() {
            Ok(adjusted) => adjusted,
            Err(error) => {
                return fixed_effect_test_not_assessed_with_method(
                    hypothesis,
                    estimates,
                    standard_errors,
                    statistics,
                    method,
                    estimability,
                    format!(
                        "Kenward-Roger fixed-effect inference could not compute adjusted vcov: {error}"
                    ),
                );
            }
        };
        let lbddf = match self.kenward_roger_lbddf_with_adjusted(&hypothesis.l.values, &adjusted) {
            Ok(lbddf) => lbddf,
            Err(error) => {
                return fixed_effect_test_not_assessed_with_method(
                    hypothesis,
                    estimates,
                    standard_errors,
                    statistics,
                    method,
                    estimability,
                    format!(
                        "Kenward-Roger fixed-effect inference could not compute denominator df: {error}"
                    ),
                );
            }
        };

        let adjusted_standard_errors =
            contrast_standard_errors(&hypothesis.l.values, &adjusted.adjusted_vcov);
        let estimate_vector = DVector::from_column_slice(&estimates);
        let contrast_cov = symmetrize_matrix(
            &(&hypothesis.l.values * &adjusted.adjusted_vcov * hypothesis.l.values.transpose()),
        );
        if !matrix_is_finite(&contrast_cov) {
            return fixed_effect_test_not_assessed_with_method(
                hypothesis,
                estimates,
                adjusted_standard_errors,
                statistics,
                method,
                estimability,
                "Kenward-Roger fixed-effect inference produced a non-finite adjusted contrast covariance"
                    .to_string(),
            );
        }

        let mut notes = vec![
            "Kenward-Roger adjusted covariance and denominator df computed from response-space Sigma/G components"
                .to_string(),
        ];
        notes.extend(adjusted.notes);
        notes.extend(lbddf.notes);

        if hypothesis.n_contrasts() == 1 {
            let Some(std_error) = adjusted_standard_errors.first().copied().flatten() else {
                return fixed_effect_test_not_assessed_with_method(
                    hypothesis,
                    estimates,
                    adjusted_standard_errors,
                    statistics,
                    method,
                    estimability,
                    "Kenward-Roger fixed-effect inference requires an available adjusted standard error"
                        .to_string(),
                );
            };
            let var_con = std_error * std_error;
            if !var_con.is_finite() || var_con <= 0.0 {
                return fixed_effect_test_not_assessed_with_method(
                    hypothesis,
                    estimates,
                    adjusted_standard_errors,
                    statistics,
                    method,
                    estimability,
                    "Kenward-Roger fixed-effect inference requires a finite positive adjusted contrast variance"
                        .to_string(),
                );
            }
            let statistic = estimates[0] / std_error;
            let p_value = match StudentsT::new(0.0, 1.0, lbddf.denominator_df) {
                Ok(t_dist) => Some(2.0 * (1.0 - t_dist.cdf(statistic.abs()))),
                Err(error) => {
                    return fixed_effect_test_not_assessed_with_method(
                        hypothesis,
                        estimates,
                        adjusted_standard_errors,
                        statistics,
                        method,
                        estimability,
                        format!(
                            "Kenward-Roger fixed-effect inference could not construct Student-t distribution: {error}"
                        ),
                    );
                }
            };
            return FixedEffectTest {
                hypothesis,
                estimates,
                standard_errors: adjusted_standard_errors,
                statistics: vec![Some(statistic)],
                numerator_df: None,
                denominator_df: Some(lbddf.denominator_df),
                p_values: vec![p_value],
                method,
                reliability: lbddf.reliability,
                status: InferenceStatus::Available,
                estimability,
                notes,
            };
        }

        let q = lbddf.restriction_rank;
        let contrast_cov_inverse = symmetric_pseudoinverse(&contrast_cov, 1e-10);
        let quadratic =
            (estimate_vector.transpose() * contrast_cov_inverse * &estimate_vector)[(0, 0)];
        if !quadratic.is_finite() || quadratic < 0.0 {
            return fixed_effect_test_not_assessed_with_method(
                hypothesis,
                estimates,
                adjusted_standard_errors,
                statistics,
                method,
                estimability,
                "Kenward-Roger fixed-effect inference produced a non-finite F quadratic form"
                    .to_string(),
            );
        }
        let f_statistic = quadratic / q as f64;
        if !f_statistic.is_finite() || f_statistic < 0.0 {
            return fixed_effect_test_not_assessed_with_method(
                hypothesis,
                estimates,
                adjusted_standard_errors,
                statistics,
                method,
                estimability,
                "Kenward-Roger fixed-effect inference produced a non-finite F statistic"
                    .to_string(),
            );
        }
        let p_value = match FisherSnedecor::new(q as f64, lbddf.denominator_df) {
            Ok(f_dist) => Some(1.0 - f_dist.cdf(f_statistic)),
            Err(error) => {
                return fixed_effect_test_not_assessed_with_method(
                    hypothesis,
                    estimates,
                    adjusted_standard_errors,
                    statistics,
                    method,
                    estimability,
                    format!(
                        "Kenward-Roger fixed-effect inference could not construct F distribution: {error}"
                    ),
                );
            }
        };
        notes.push(
            "Kenward-Roger multi-df F row uses F scaling = 1.0 in the current row payload"
                .to_string(),
        );

        FixedEffectTest {
            hypothesis,
            estimates,
            standard_errors: adjusted_standard_errors,
            statistics: vec![Some(f_statistic)],
            numerator_df: Some(q as f64),
            denominator_df: Some(lbddf.denominator_df),
            p_values: vec![p_value],
            method,
            reliability: lbddf.reliability,
            status: InferenceStatus::Available,
            estimability,
            notes,
        }
    }

    fn bootstrap_fixed_effect_test_from_payload(
        &self,
        hypothesis: FixedEffectHypothesis,
        estimates: Vec<f64>,
        standard_errors: Vec<Option<f64>>,
        statistics: Vec<Option<f64>>,
        estimability: FixedContrastEstimability,
        payload: &BootstrapRunPayload,
    ) -> FixedEffectTest {
        const MIN_SUCCESSFUL_REPLICATES: usize = 30;
        const MODERATE_SUCCESSFUL_REPLICATES: usize = 999;
        const MODERATE_MAX_MCSE: f64 = 0.02;
        const MODERATE_MAX_FAILED_REFIT_RATE: f64 = 0.01;
        const MODERATE_MAX_BOUNDARY_RATE: f64 = 0.05;
        const CONTINUITY_CORRECTION: f64 = 1.0;

        let method = InferenceMethod::ParametricBootstrap;

        if payload.metadata.schema_name != BOOTSTRAP_RUN_SCHEMA
            || payload.metadata.schema_version != BOOTSTRAP_RUN_SCHEMA_VERSION
        {
            return fixed_effect_test_not_assessed_with_method(
                hypothesis,
                estimates,
                standard_errors,
                statistics,
                method,
                estimability,
                format!(
                    "bootstrap_replicate_accounting_unavailable: expected {BOOTSTRAP_RUN_SCHEMA} {BOOTSTRAP_RUN_SCHEMA_VERSION}, got {} {}",
                    payload.metadata.schema_name, payload.metadata.schema_version
                ),
            );
        }

        if payload.metadata.target.kind != BootstrapTargetKind::FixedEffectNull {
            return fixed_effect_test_not_assessed_with_method(
                hypothesis,
                estimates,
                standard_errors,
                statistics,
                method,
                estimability,
                "bootstrap_null_target_unavailable: payload target is not fixed_effect_null"
                    .to_string(),
            );
        }

        if payload.metadata.target.contrast_label.as_deref() != Some(hypothesis.label.as_str()) {
            return fixed_effect_test_not_assessed_with_method(
                hypothesis,
                estimates,
                standard_errors,
                statistics,
                method,
                estimability,
                "bootstrap_null_target_unavailable: payload contrast label does not match requested hypothesis"
                    .to_string(),
            );
        }

        if let Err(error) = payload.replicates.validate_for_model(self) {
            return fixed_effect_test_not_assessed_with_method(
                hypothesis,
                estimates,
                standard_errors,
                statistics,
                method,
                estimability,
                format!("bootstrap_replicate_accounting_unavailable: {error}"),
            );
        }

        if payload.metadata.completed_replicates != payload.replicates.len() {
            return fixed_effect_test_not_assessed_with_method(
                hypothesis,
                estimates,
                standard_errors,
                statistics,
                method,
                estimability,
                "bootstrap_replicate_accounting_unavailable: completed_replicates does not match replicate count"
                    .to_string(),
            );
        }

        let actual_successful = payload
            .replicates
            .fits
            .iter()
            .filter(|fit| fit.is_successful())
            .count();
        if payload.metadata.successful_replicates != actual_successful {
            return fixed_effect_test_not_assessed_with_method(
                hypothesis,
                estimates,
                standard_errors,
                statistics,
                method,
                estimability,
                "bootstrap_replicate_accounting_unavailable: successful_replicates does not match successful refit count"
                    .to_string(),
            );
        }

        if payload.metadata.failed_refit_policy != BootstrapFailedRefitPolicy::Exclude {
            return fixed_effect_test_not_assessed_with_method(
                hypothesis,
                estimates,
                standard_errors,
                statistics,
                method,
                estimability,
                "bootstrap_failed_refit_policy_unavailable: only exclude failed-refit policy is certified for fixed-effect bootstrap rows"
                    .to_string(),
            );
        }

        let Some(observed) = fixed_effect_bootstrap_statistic(self, &hypothesis) else {
            return fixed_effect_test_not_assessed_with_method(
                hypothesis,
                estimates,
                standard_errors,
                statistics,
                method,
                estimability,
                "bootstrap_observed_statistic_nonfinite: observed fixed-effect statistic is unavailable"
                    .to_string(),
            );
        };
        let observed_statistic = observed.value;

        let replicate_statistics = match payload.replicate_statistics.as_deref() {
            Some(values) => {
                if values.len() != payload.replicates.len() {
                    return fixed_effect_test_not_assessed_with_method(
                        hypothesis,
                        estimates,
                        standard_errors,
                        statistics,
                        method,
                        estimability,
                        "bootstrap_replicate_accounting_unavailable: replicate_statistics length does not match replicate count"
                            .to_string(),
                    );
                }
                values.iter().map(|value| value.abs()).collect::<Vec<_>>()
            }
            None => {
                match self.bootstrap_coefficient_statistics_from_replicates(&hypothesis, payload) {
                    Ok(values) => values,
                    Err(error) => {
                        return fixed_effect_test_not_assessed_with_method(
                            hypothesis,
                            estimates,
                            standard_errors,
                            statistics,
                            method,
                            estimability,
                            format!("bootstrap_replicate_accounting_unavailable: {error}"),
                        );
                    }
                }
            }
        };

        let finite_statistics = replicate_statistics
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        if let Some(recorded) = payload.metadata.finite_statistic_count {
            if recorded != finite_statistics.len() {
                return fixed_effect_test_not_assessed_with_method(
                    hypothesis,
                    estimates,
                    standard_errors,
                    statistics,
                    method,
                    estimability,
                    "bootstrap_replicate_accounting_unavailable: finite_statistic_count does not match finite replicate statistics"
                        .to_string(),
                );
            }
        }

        if finite_statistics.len() < MIN_SUCCESSFUL_REPLICATES {
            return fixed_effect_test_not_assessed_with_method(
                hypothesis,
                estimates,
                standard_errors,
                statistics,
                method,
                estimability,
                format!(
                    "bootstrap_successful_replicates_too_few: {} finite replicate statistic(s), need at least {MIN_SUCCESSFUL_REPLICATES}",
                    finite_statistics.len()
                ),
            );
        }

        let extreme = finite_statistics
            .iter()
            .filter(|&&value| value >= observed_statistic)
            .count();
        let denominator = finite_statistics.len() as f64 + CONTINUITY_CORRECTION;
        let p_value = (extreme as f64 + CONTINUITY_CORRECTION) / denominator;
        let mcse = (p_value * (1.0 - p_value) / finite_statistics.len() as f64).sqrt();
        if !p_value.is_finite() || !mcse.is_finite() {
            return fixed_effect_test_not_assessed_with_method(
                hypothesis,
                estimates,
                standard_errors,
                statistics,
                method,
                estimability,
                "bootstrap_mcse_unavailable: bootstrap p-value or Monte Carlo standard error is non-finite"
                    .to_string(),
            );
        }

        let failed_refit_rate = if payload.metadata.completed_replicates > 0 {
            payload.metadata.failed_refits as f64 / payload.metadata.completed_replicates as f64
        } else {
            1.0
        };
        let boundary_rate = payload.metadata.boundary_rate.unwrap_or(0.0);
        let reliability = if finite_statistics.len() >= MODERATE_SUCCESSFUL_REPLICATES
            && mcse <= MODERATE_MAX_MCSE
            && failed_refit_rate <= MODERATE_MAX_FAILED_REFIT_RATE
            && boundary_rate <= MODERATE_MAX_BOUNDARY_RATE
        {
            ReliabilityGrade::Moderate
        } else {
            ReliabilityGrade::Low
        };

        let mut notes = vec![
            format!(
                "bootstrap fixed-effect row computed from fixed_effect_null target `{}`",
                payload.metadata.target.label
            ),
            format!("bootstrap fixed-effect statistic={}", observed.label),
            format!(
                "requested_replicates={}, completed_replicates={}, successful_replicates={}, finite_statistics={}",
                payload.metadata.requested_replicates,
                payload.metadata.completed_replicates,
                payload.metadata.successful_replicates,
                finite_statistics.len()
            ),
            format!(
                "failed_refit_policy={:?}, failed_refits={}, boundary_rate={:.6}, mcse={:.6}",
                payload.metadata.failed_refit_policy,
                payload.metadata.failed_refits,
                boundary_rate,
                mcse
            ),
        ];
        notes.extend(payload.metadata.notes.clone());

        FixedEffectTest {
            hypothesis,
            estimates,
            standard_errors,
            statistics: vec![Some(observed_statistic)],
            numerator_df: observed.numerator_df,
            denominator_df: None,
            p_values: vec![Some(p_value)],
            method,
            reliability,
            status: InferenceStatus::Available,
            estimability,
            notes,
        }
    }

    fn bootstrap_coefficient_statistics_from_replicates(
        &self,
        hypothesis: &FixedEffectHypothesis,
        payload: &BootstrapRunPayload,
    ) -> Result<Vec<f64>> {
        let (coefficient_index, coefficient_weight) =
            scalar_single_coefficient_contrast(&hypothesis.l.values).ok_or_else(|| {
                MixedModelError::InvalidArgument(
                    "replicate_statistics are required for non-coefficient bootstrap contrasts"
                        .to_string(),
                )
            })?;
        let rhs = hypothesis.rhs.values[0];
        let mut values = Vec::new();
        for fit in &payload.replicates.fits {
            if !fit.is_successful() {
                values.push(f64::NAN);
                continue;
            }
            let beta = self.fixed_effect_active_vector_to_user_basis(&fit.beta, "beta")?;
            let se = self.fixed_effect_active_vector_to_user_basis(&fit.se, "standard error")?;
            let estimate = coefficient_weight * beta[coefficient_index] - rhs;
            let standard_error = coefficient_weight.abs() * se[coefficient_index];
            let statistic =
                if standard_error.is_finite() && standard_error > 0.0 && estimate.is_finite() {
                    (estimate / standard_error).abs()
                } else {
                    f64::NAN
                };
            values.push(statistic);
        }
        Ok(values)
    }

    /// Build the default fixed-effect coefficient inference table.
    pub fn fixed_effect_inference_table(&self) -> FixedEffectInferenceTable {
        self.fixed_effect_inference_table_with_method(FixedEffectTestMethod::Auto)
    }

    fn fixed_effect_inference_table_with_method(
        &self,
        method: FixedEffectTestMethod,
    ) -> FixedEffectInferenceTable {
        let rows = self
            .coefficient_hypotheses()
            .into_iter()
            .map(|hypothesis| {
                fixed_effect_test_to_inference_row(
                    FixedEffectInferenceRowKind::Coefficient,
                    self.test_contrast_with_method(hypothesis, method),
                )
            })
            .collect();
        FixedEffectInferenceTable::new(rows)
    }

    /// Return the fixed-effect covariance matrix with compiler-audit metadata.
    pub fn fixed_effect_covariance_matrix(&self) -> FixedEffectCovarianceMatrix {
        self.fixed_effect_covariance_matrix_with_available_method(
            FixedEffectCovarianceMethod::ModelBased,
            vec![
                "model-based fixed-effect covariance geometry; inference claims remain on fixed_effect_inference_table rows"
                    .to_string(),
            ],
        )
    }

    pub(crate) fn glmm_fixed_effect_covariance_matrix(&self) -> FixedEffectCovarianceMatrix {
        self.fixed_effect_covariance_matrix_with_available_method(
            FixedEffectCovarianceMethod::PirlsLaplaceWorkingHessian,
            vec![
                "PIRLS/Laplace working-Hessian fixed-effect covariance geometry; inference claims remain on fixed_effect_inference_table rows"
                    .to_string(),
            ],
        )
    }

    fn fixed_effect_covariance_matrix_with_available_method(
        &self,
        method: FixedEffectCovarianceMethod,
        notes: Vec<String>,
    ) -> FixedEffectCovarianceMatrix {
        let coef_names = self.coef_names();
        let vcov = self.vcov();
        let expected_rank = coef_names.len();
        let rank = self.feterm.rank;
        let aliased = aliased_fixed_effect_names(&coef_names, &self.feterm.piv, rank);
        let finite = matrix_is_finite(&vcov);
        let symmetric = finite && matrix_max_asymmetry(&vcov) <= 1e-8;
        let details = FixedEffectCovarianceDetails {
            rank: Some(rank),
            expected_rank: Some(expected_rank),
            aliased,
            matrix_rows: vcov.nrows(),
            matrix_cols: vcov.ncols(),
            finite: Some(finite),
            symmetric: Some(symmetric),
        };

        if rank < expected_rank {
            return FixedEffectCovarianceMatrix::unavailable(
                coef_names,
                "rank_deficient_fixed_effects",
                details,
                vec![
                    "fixed-effect covariance matrix is unavailable because the fixed-effect design is rank deficient"
                        .to_string(),
                ],
            );
        }

        if !finite {
            return FixedEffectCovarianceMatrix::unavailable(
                coef_names,
                "fixed_effect_covariance_nonfinite",
                details,
                vec!["fixed-effect covariance matrix contains non-finite entries".to_string()],
            );
        }

        if !symmetric {
            return FixedEffectCovarianceMatrix::unavailable(
                coef_names,
                "fixed_effect_covariance_not_symmetric",
                details,
                vec!["fixed-effect covariance matrix failed symmetry validation".to_string()],
            );
        }

        match method {
            FixedEffectCovarianceMethod::ModelBased => FixedEffectCovarianceMatrix::model_based(
                coef_names,
                matrix_rows(&vcov),
                details,
                notes,
            ),
            FixedEffectCovarianceMethod::PirlsLaplaceWorkingHessian => {
                FixedEffectCovarianceMatrix::pirls_laplace_working_hessian(
                    coef_names,
                    matrix_rows(&vcov),
                    details,
                    notes,
                )
            }
            FixedEffectCovarianceMethod::JointLaplaceActiveHessian => {
                FixedEffectCovarianceMatrix::joint_laplace_active_hessian(
                    coef_names,
                    matrix_rows(&vcov),
                    details,
                    notes,
                )
            }
            FixedEffectCovarianceMethod::Unavailable => unreachable!(
                "available covariance constructor should not be called with unavailable method"
            ),
        }
    }

    pub(super) fn refresh_fixed_effect_covariance_matrix(&mut self) {
        self.compiler_artifact.fixed_effect_covariance_matrix =
            Some(self.fixed_effect_covariance_matrix());
    }

    pub(super) fn refresh_fixed_effect_inference_table(&mut self) {
        // Keep ordinary fit() comparable to MixedModels.jl: fitting records
        // cheap coefficient rows, while explicit inference calls compute
        // finite-sample Satterthwaite/KR rows on demand.
        self.compiler_artifact.fixed_effect_inference_table = Some(
            self.fixed_effect_inference_table_with_method(FixedEffectTestMethod::AsymptoticWaldZ),
        );
    }

    fn fixed_effect_p_value_policy(&self) -> CoefTablePValuePolicy {
        if self
            .compiler_artifact
            .reductions
            .iter()
            .any(|record| record.trigger == ReductionTrigger::SelectionTime)
        {
            return CoefTablePValuePolicy::Unavailable {
                reason: "ordinary fixed-effect p-values are unavailable after selection-time model changes"
                    .to_string(),
            };
        }

        if let Some(reason) = self
            .compiler_artifact
            .reproducibility
            .fit_intent
            .p_value_unavailable_reason()
        {
            CoefTablePValuePolicy::Unavailable { reason }
        } else {
            CoefTablePValuePolicy::AsymptoticWaldZ
        }
    }

    /// Cook's distance for each observation.
    ///
    /// Measures the influence of each observation on the fixed-effects
    /// estimates.  The formula mirrors `cooksdistance(model)` in Julia's
    /// MixedModels.jl (linearmixedmodel.jl line 420):
    ///
    /// ```text
    /// D_i = (r_i / (1 - h_i))^2 * h_i / (σ² * p)
    /// ```
    ///
    /// where `r_i` is the i-th residual, `h_i` is the i-th leverage,
    /// `σ²` is the variance estimate, and `p` is the rank of the
    /// fixed-effects matrix.
    pub fn cooks_distance(&self) -> DVector<f64> {
        let r = self.residuals();
        let h = self.leverage();
        let mse = self.varest();
        let p = self.feterm.rank as f64;
        let n = self.dims.n;

        let mut d = DVector::zeros(n);
        for i in 0..n {
            let denom = 1.0 - h[i];
            if denom.abs() > f64::EPSILON {
                d[i] = (r[i] / denom).powi(2) * h[i] / (mse * p);
            }
        }
        d
    }
}

pub(super) fn kenward_roger_covariance_component_count(reterm: &ReMat) -> usize {
    reterm.inds.len()
}

pub(super) fn kenward_roger_covariance_component_indices(reterm: &ReMat) -> Vec<(usize, usize)> {
    reterm
        .inds
        .iter()
        .map(|&index| {
            let col = index / reterm.vsize;
            let row = index % reterm.vsize;
            (row, col)
        })
        .collect()
}

pub(super) fn kenward_roger_response_component(
    reterm: &ReMat,
    row: usize,
    col: usize,
    n_observations: usize,
) -> Result<DMatrix<f64>> {
    if row >= reterm.vsize || col >= reterm.vsize {
        return Err(MixedModelError::DimensionMismatch(format!(
            "KR covariance component ({row}, {col}) is outside random-effect vector size {}",
            reterm.vsize
        )));
    }
    if reterm.n_obs() != n_observations {
        return Err(MixedModelError::DimensionMismatch(format!(
            "KR random-effect term '{}' has {} observations, expected {n_observations}",
            reterm.grouping_name,
            reterm.n_obs()
        )));
    }

    let mut component = DMatrix::zeros(n_observations, n_observations);
    for obs_i in 0..n_observations {
        let level_i = reterm.refs[obs_i];
        for obs_j in 0..=obs_i {
            if level_i != reterm.refs[obs_j] {
                continue;
            }
            let value = if row == col {
                reterm.z[(row, obs_i)] * reterm.z[(row, obs_j)]
            } else {
                reterm.z[(row, obs_i)] * reterm.z[(col, obs_j)]
                    + reterm.z[(col, obs_i)] * reterm.z[(row, obs_j)]
            };
            component[(obs_i, obs_j)] = value;
            component[(obs_j, obs_i)] = value;
        }
    }
    Ok(component)
}

fn contrast_standard_errors(l: &DMatrix<f64>, vcov: &DMatrix<f64>) -> Vec<Option<f64>> {
    (0..l.nrows())
        .map(|row| {
            let mut variance = 0.0;
            for i in 0..l.ncols() {
                for j in 0..l.ncols() {
                    variance += l[(row, i)] * vcov[(i, j)] * l[(row, j)];
                }
            }
            (variance.is_finite() && variance >= 0.0).then_some(variance.max(0.0).sqrt())
        })
        .collect()
}

fn contrast_row_quadratic_form(l: &DMatrix<f64>, row: usize, matrix: &DMatrix<f64>) -> f64 {
    let mut value = 0.0;
    for i in 0..l.ncols() {
        for j in 0..l.ncols() {
            value += l[(row, i)] * matrix[(i, j)] * l[(row, j)];
        }
    }
    value
}

pub(super) fn assess_fixed_contrast_estimability(
    hypothesis: &FixedEffectHypothesis,
    beta: &DVector<f64>,
    vcov: &DMatrix<f64>,
) -> FixedContrastEstimability {
    let mut estimable_rows = 0usize;
    for row in 0..hypothesis.l.values.nrows() {
        let row_estimable = (0..hypothesis.l.values.ncols()).all(|col| {
            let weight = hypothesis.l.values[(row, col)];
            weight.abs() <= 1e-12 || (beta[col].is_finite() && vcov[(col, col)].is_finite())
        });
        if row_estimable {
            estimable_rows += 1;
        }
    }

    let requested = hypothesis.n_contrasts();
    if estimable_rows == requested {
        FixedContrastEstimability::estimable(hypothesis.label.clone(), estimable_rows, requested)
    } else if estimable_rows == 0 {
        FixedContrastEstimability::not_estimable(hypothesis.label.clone(), requested, Vec::new())
    } else {
        FixedContrastEstimability::partially_estimable(
            hypothesis.label.clone(),
            estimable_rows,
            requested,
            Vec::new(),
        )
    }
}

fn scalar_single_coefficient_contrast(l: &DMatrix<f64>) -> Option<(usize, f64)> {
    if l.nrows() != 1 {
        return None;
    }
    let mut found = None;
    for col in 0..l.ncols() {
        let value = l[(0, col)];
        if value.abs() <= 1e-12 {
            continue;
        }
        if found.is_some() {
            return None;
        }
        found = Some((col, value));
    }
    found
}

fn scalar_contrast_abs_studentized(
    model: &LinearMixedModel,
    hypothesis: &FixedEffectHypothesis,
) -> Option<f64> {
    if hypothesis.n_contrasts() != 1 || hypothesis.n_coefficients() != model.coef_names().len() {
        return None;
    }
    let beta = model.coef();
    let vcov = model.vcov();
    let estimate = (&hypothesis.l.values * beta - &hypothesis.rhs.values)[0];
    let se = contrast_standard_errors(&hypothesis.l.values, &vcov)
        .into_iter()
        .next()
        .flatten()?;
    (estimate.is_finite() && se.is_finite() && se > 0.0).then_some((estimate / se).abs())
}

pub(super) struct FixedEffectBootstrapStatistic {
    pub(super) value: f64,
    pub(super) numerator_df: Option<f64>,
    pub(super) label: &'static str,
}

pub(super) fn fixed_effect_bootstrap_statistic(
    model: &LinearMixedModel,
    hypothesis: &FixedEffectHypothesis,
) -> Option<FixedEffectBootstrapStatistic> {
    if hypothesis.n_contrasts() == 1 {
        return scalar_contrast_abs_studentized(model, hypothesis).map(|value| {
            FixedEffectBootstrapStatistic {
                value,
                numerator_df: None,
                label: "studentized_scalar_t",
            }
        });
    }

    if hypothesis.n_coefficients() != model.coef_names().len() || hypothesis.n_contrasts() == 0 {
        return None;
    }

    let beta = model.coef();
    let vcov = model.vcov();
    if !matrix_is_finite(&vcov) {
        return None;
    }

    let delta = &hypothesis.l.values * beta - &hypothesis.rhs.values;
    if !delta.iter().all(|value| value.is_finite()) {
        return None;
    }

    let middle =
        symmetrize_matrix(&(&hypothesis.l.values * vcov * hypothesis.l.values.transpose()));
    if !matrix_is_finite(&middle) {
        return None;
    }

    let eig = SymmetricEigen::new(middle.clone());
    let max_abs = eig
        .eigenvalues
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    let tolerance = (1e-10 * max_abs.max(1.0)).max(1e-12);
    let min_eigen = eig
        .eigenvalues
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    if min_eigen < -tolerance {
        return None;
    }

    let effective_rank = eig
        .eigenvalues
        .iter()
        .filter(|value| value.abs() > tolerance)
        .count();
    if effective_rank == 0 {
        return None;
    }

    let min_abs = eig
        .eigenvalues
        .iter()
        .map(|value| value.abs())
        .fold(f64::INFINITY, f64::min);
    let middle_inverse = if min_abs <= tolerance {
        symmetric_pseudoinverse(&middle, tolerance)
    } else {
        invert_spd_matrix(&middle, "fixed-effect bootstrap L V L' matrix").ok()?
    };
    let quadratic = (delta.transpose() * middle_inverse * delta)[(0, 0)];
    let statistic = quadratic / effective_rank as f64;
    (statistic.is_finite() && statistic >= 0.0).then_some(FixedEffectBootstrapStatistic {
        value: statistic,
        numerator_df: Some(effective_rank as f64),
        label: "joint_wald_f",
    })
}

fn scalar_contrast_estimate(
    model: &LinearMixedModel,
    hypothesis: &FixedEffectHypothesis,
) -> Option<f64> {
    if hypothesis.n_contrasts() != 1 || hypothesis.n_coefficients() != model.coef_names().len() {
        return None;
    }
    let estimate = (&hypothesis.l.values * model.coef() - &hypothesis.rhs.values)[0];
    estimate.is_finite().then_some(estimate)
}

fn bootstrap_scalar_percentile_intervals(
    label: &str,
    statistics: &[f64],
    observed: f64,
    levels: &[f64],
) -> Result<Vec<BootstrapInterval>> {
    if !observed.is_finite() {
        return Err(MixedModelError::InvalidArgument(
            "bootstrap intervals require a finite observed statistic".to_string(),
        ));
    }
    let mut finite = statistics
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return Err(MixedModelError::InvalidArgument(
            "bootstrap intervals require at least one finite replicate statistic".to_string(),
        ));
    }
    finite.sort_by(|a, b| a.partial_cmp(b).unwrap());

    levels
        .iter()
        .map(|&level| {
            validate_level(level)?;
            let alpha = (1.0 - level) / 2.0;
            Ok(BootstrapInterval {
                parameter: label.to_string(),
                level,
                lower: quantile_sorted(&finite, alpha),
                upper: quantile_sorted(&finite, 1.0 - alpha),
                n: finite.len(),
                method: BootstrapIntervalMethod::Percentile,
            })
        })
        .collect()
}

fn failed_bootstrap_replicate_like(model: &LinearMixedModel) -> BootstrapReplicate {
    BootstrapReplicate {
        objective: f64::NAN,
        sigma: f64::NAN,
        beta: model.beta(),
        se: DVector::from_element(model.feterm.rank, f64::NAN),
        theta: model.theta(),
    }
}

fn fixed_effect_statistic_name_label(name: FixedEffectStatisticName) -> &'static str {
    match name {
        FixedEffectStatisticName::Z => "z",
        FixedEffectStatisticName::T => "t",
        FixedEffectStatisticName::F => "F",
        FixedEffectStatisticName::ChiSquare => "chisq",
    }
}

fn fixed_effect_inference_method_label(method: FixedEffectInferenceMethod) -> &'static str {
    match method {
        FixedEffectInferenceMethod::AsymptoticWaldZ => "wald-z",
        FixedEffectInferenceMethod::Satterthwaite => "satterthwaite",
        FixedEffectInferenceMethod::KenwardRoger => "kenward-roger",
        FixedEffectInferenceMethod::Bootstrap => "bootstrap",
        FixedEffectInferenceMethod::NotComputed => "not-computed",
    }
}

fn fixed_effect_term_test_type_label(term_test_type: FixedEffectTermTestType) -> &'static str {
    match term_test_type {
        FixedEffectTermTestType::TypeI => "type_i",
        FixedEffectTermTestType::TypeII => "type_ii",
        FixedEffectTermTestType::TypeIII => "type_iii",
    }
}

fn fixed_effect_identity_hypothesis(
    term: &str,
    indices: &[usize],
    n_coefficients: usize,
) -> Option<FixedEffectHypothesis> {
    if indices.is_empty() || n_coefficients == 0 {
        return None;
    }
    let mut l = DMatrix::zeros(indices.len(), n_coefficients);
    for (row, &index) in indices.iter().enumerate() {
        if index >= n_coefficients {
            return None;
        }
        l[(row, index)] = 1.0;
    }
    Some(FixedEffectHypothesis::zero_rhs(
        term.to_string(),
        crate::compiler::ContrastMatrix::new(l).ok()?,
    ))
}

fn fixed_effect_basis_hypothesis(
    term: &str,
    row_indices: &[usize],
    basis: &DMatrix<f64>,
) -> Option<FixedEffectHypothesis> {
    if row_indices.is_empty() || basis.ncols() == 0 {
        return None;
    }
    let mut l = DMatrix::zeros(row_indices.len(), basis.ncols());
    for (row, &source_row) in row_indices.iter().enumerate() {
        if source_row >= basis.nrows() {
            return None;
        }
        for col in 0..basis.ncols() {
            l[(row, col)] = basis[(source_row, col)];
        }
    }
    Some(FixedEffectHypothesis::zero_rhs(
        term.to_string(),
        crate::compiler::ContrastMatrix::new(l).ok()?,
    ))
}

fn fixed_effect_type_ii_hypothesis(
    term: &str,
    x: &DMatrix<f64>,
    col_terms: &[String],
    contained_terms: &[String],
) -> Option<FixedEffectHypothesis> {
    let p = x.ncols();
    if p == 0 || col_terms.len() != p {
        return None;
    }
    let mut moved = Vec::new();
    for (index, col_term) in col_terms.iter().enumerate() {
        if col_term == term
            || contained_terms
                .iter()
                .any(|contained| contained == col_term)
        {
            moved.push(index);
        }
    }
    let row_positions = moved
        .iter()
        .enumerate()
        .filter_map(|(position, &original)| (col_terms[original] == term).then_some(position))
        .collect::<Vec<_>>();
    if row_positions.is_empty() {
        return None;
    }
    let moved_len = moved.len();
    let mut permutation = (0..p)
        .filter(|index| !moved.contains(index))
        .collect::<Vec<_>>();
    permutation.extend(moved);
    let x_new = select_matrix_columns(x, &permutation);
    let basis_new = doolittle_contrast_basis(&x_new);
    let moved_start = p - moved_len;
    let mut l = DMatrix::zeros(row_positions.len(), p);
    for (out_row, relative_row) in row_positions.into_iter().enumerate() {
        let source_row = moved_start + relative_row;
        for (new_col, &original_col) in permutation.iter().enumerate() {
            l[(out_row, original_col)] = basis_new[(source_row, new_col)];
        }
    }
    Some(FixedEffectHypothesis::zero_rhs(
        term.to_string(),
        crate::compiler::ContrastMatrix::new(l).ok()?,
    ))
}

fn select_matrix_columns(x: &DMatrix<f64>, columns: &[usize]) -> DMatrix<f64> {
    let mut out = DMatrix::zeros(x.nrows(), columns.len());
    for (new_col, &old_col) in columns.iter().enumerate() {
        for row in 0..x.nrows() {
            out[(row, new_col)] = x[(row, old_col)];
        }
    }
    out
}

fn doolittle_contrast_basis(x: &DMatrix<f64>) -> DMatrix<f64> {
    if x.ncols() == 0 {
        return DMatrix::zeros(0, 0);
    }
    let crossprod = x.transpose() * x;
    doolittle_lower(&crossprod, 1.0e-6).transpose()
}

fn doolittle_lower(x: &DMatrix<f64>, eps: f64) -> DMatrix<f64> {
    let n = x.nrows();
    debug_assert_eq!(n, x.ncols());
    let mut lower = DMatrix::zeros(n, n);
    let mut upper = DMatrix::zeros(n, n);
    for i in 0..n {
        lower[(i, i)] = 1.0;
    }
    for i in 0..n {
        for j in 0..n {
            let mut value = x[(i, j)];
            for k in 0..i {
                value -= lower[(i, k)] * upper[(k, j)];
            }
            upper[(i, j)] = if value.abs() < eps { 0.0 } else { value };
        }
        for j in (i + 1)..n {
            let mut value = x[(j, i)];
            for k in 0..i {
                value -= lower[(j, k)] * upper[(k, i)];
            }
            lower[(j, i)] = if upper[(i, i)].abs() < eps {
                0.0
            } else {
                value / upper[(i, i)]
            };
            if lower[(j, i)].abs() < eps {
                lower[(j, i)] = 0.0;
            }
        }
    }
    lower
}

fn fixed_effect_terms_containing(term: &str, term_indices: &[(String, Vec<usize>)]) -> Vec<String> {
    term_indices
        .iter()
        .filter_map(|(candidate, _)| {
            fixed_effect_term_contains(candidate, term).then_some(candidate.clone())
        })
        .collect()
}

fn fixed_effect_term_contains(candidate: &str, term: &str) -> bool {
    let term_parts = fixed_effect_term_parts(term);
    let candidate_parts = fixed_effect_term_parts(candidate);
    !term_parts.is_empty()
        && candidate_parts.len() > term_parts.len()
        && term_parts
            .iter()
            .all(|part| candidate_parts.iter().any(|candidate| candidate == part))
}

fn fixed_effect_term_parts(term: &str) -> Vec<&str> {
    if term == "1" || term == "(Intercept)" {
        return Vec::new();
    }
    term.split(':')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .collect()
}

pub(super) fn satterthwaite_f_denominator_df(direction_dfs: &[f64], tolerance: f64) -> Option<f64> {
    if direction_dfs.is_empty() || direction_dfs.iter().any(|df| !df.is_finite() || *df <= 0.0) {
        return None;
    }
    if direction_dfs.len() == 1 {
        return Some(direction_dfs[0]);
    }
    if direction_dfs
        .windows(2)
        .all(|pair| (pair[1] - pair[0]).abs() < tolerance)
    {
        return Some(direction_dfs.iter().sum::<f64>() / direction_dfs.len() as f64);
    }
    if direction_dfs.iter().any(|df| *df <= 2.0) {
        return Some(2.0);
    }
    let expected = direction_dfs.iter().map(|df| df / (df - 2.0)).sum::<f64>();
    let denom = expected - direction_dfs.len() as f64;
    (denom.is_finite() && denom > 0.0).then_some(2.0 * expected / denom)
}

pub(super) fn fixed_effect_test_to_inference_row(
    kind: FixedEffectInferenceRowKind,
    test: FixedEffectTest,
) -> FixedEffectInferenceRow {
    let statistic_name = fixed_effect_statistic_name(&test);
    let reason = fixed_effect_inference_reason(&test);
    let reliability_reason = fixed_effect_reliability_reason(&test);
    let details = fixed_effect_details_for_test(kind, &test, statistic_name);
    FixedEffectInferenceRow {
        label: test.hypothesis.label.clone(),
        kind,
        estimate: finite_option(test.estimates.first().copied()),
        std_error: finite_option(test.standard_errors.first().copied().flatten()),
        numerator_df: fixed_effect_row_numerator_df(&test, statistic_name),
        denominator_df: test.denominator_df,
        statistic: finite_option(test.statistics.first().copied().flatten()),
        statistic_name,
        p_value: finite_option(test.p_values.first().copied().flatten()),
        method: fixed_effect_inference_method(&test.method),
        status: fixed_effect_inference_status(&test.status),
        reliability: test.reliability,
        reliability_reason,
        estimability: EstimabilityAssessment::FixedContrast(test.estimability),
        reason,
        details,
        notes: test.notes,
    }
}

fn fixed_effect_details_for_test(
    kind: FixedEffectInferenceRowKind,
    test: &FixedEffectTest,
    statistic_name: Option<FixedEffectStatisticName>,
) -> Option<FixedEffectInferenceDetails> {
    let contrast_family = (kind != FixedEffectInferenceRowKind::Coefficient
        || test.hypothesis.n_contrasts() > 1)
        .then(|| contrast_family_details(kind, test, statistic_name));
    let kenward_roger =
        (test.method == InferenceMethod::KenwardRoger).then(|| KenwardRogerInferenceDetails {
            restriction_rank: test.estimability.rank,
            f_scaling: (statistic_name == Some(FixedEffectStatisticName::F)).then_some(1.0),
            statistic_scale: (statistic_name == Some(FixedEffectStatisticName::F))
                .then(|| "unscaled".to_string()),
        });
    let details = FixedEffectInferenceDetails {
        bootstrap: None,
        contrast_family,
        kenward_roger,
    };
    (!details.is_empty()).then_some(details)
}

fn contrast_family_details(
    kind: FixedEffectInferenceRowKind,
    test: &FixedEffectTest,
    statistic_name: Option<FixedEffectStatisticName>,
) -> ContrastFamilyDetails {
    let requested_rank = test.estimability.requested_rank;
    let effective_rank = test.estimability.rank;
    let rank_deficient = match (effective_rank, requested_rank) {
        (Some(rank), Some(requested)) => Some(rank < requested),
        _ => None,
    };
    let numerator_df_semantics = match (kind, statistic_name) {
        (_, Some(FixedEffectStatisticName::F)) => "effective_restriction_rank",
        (FixedEffectInferenceRowKind::Term, _) => "term_scalar_or_unavailable",
        _ => "scalar_contrast_no_numerator_df",
    }
    .to_string();
    ContrastFamilyDetails {
        family_id: test.hypothesis.label.clone(),
        family_label: test.hypothesis.label.clone(),
        restriction_rows: test.hypothesis.n_contrasts(),
        coefficient_count: test.hypothesis.n_coefficients(),
        requested_rank,
        effective_rank,
        rank_deficient,
        rhs_nonzero: test
            .hypothesis
            .rhs
            .values
            .iter()
            .any(|value| value.abs() > 0.0),
        numerator_df: fixed_effect_row_numerator_df(test, statistic_name),
        numerator_df_semantics,
    }
}

fn attach_bootstrap_details(
    row: &mut FixedEffectInferenceRow,
    payload: &BootstrapRunPayload,
    null_target: Option<&FixedEffectNullBootstrapTarget>,
) {
    let details = row.details.get_or_insert(FixedEffectInferenceDetails {
        bootstrap: None,
        contrast_family: None,
        kenward_roger: None,
    });
    details.bootstrap = Some(BootstrapInferenceDetails {
        target_kind: bootstrap_target_kind_label(payload.metadata.target.kind).to_string(),
        target_label: payload.metadata.target.label.clone(),
        contrast_label: payload.metadata.target.contrast_label.clone(),
        requested_replicates: payload.metadata.requested_replicates,
        completed_replicates: payload.metadata.completed_replicates,
        successful_replicates: payload.metadata.successful_replicates,
        failed_refits: payload.metadata.failed_refits,
        failed_refit_policy: bootstrap_failed_refit_policy_label(
            payload.metadata.failed_refit_policy,
        )
        .to_string(),
        boundary_count: payload.metadata.boundary_count,
        boundary_rate: payload.metadata.boundary_rate,
        seed_rng: payload.metadata.seed_record.rng.clone(),
        seed: payload.metadata.seed_record.seed,
        finite_statistic_count: payload.metadata.finite_statistic_count,
        mcse: payload.metadata.mcse,
        null_target: null_target.map(|target| FixedEffectNullTargetSummary {
            covariance_policy: fixed_effect_null_covariance_policy_label(target.covariance_policy)
                .to_string(),
            coefficient_count: target.coefficient_names.len(),
            theta_count: target.theta.len(),
            sigma: target.sigma.is_finite().then_some(target.sigma),
            reml: target.reml,
        }),
    });
}

fn bootstrap_target_kind_label(kind: BootstrapTargetKind) -> &'static str {
    match kind {
        BootstrapTargetKind::FullModelDistribution => "full_model_distribution",
        BootstrapTargetKind::FixedEffectNull => "fixed_effect_null",
        BootstrapTargetKind::ClusterResample => "cluster_resample",
    }
}

fn bootstrap_failed_refit_policy_label(policy: BootstrapFailedRefitPolicy) -> &'static str {
    match policy {
        BootstrapFailedRefitPolicy::Exclude => "exclude",
        BootstrapFailedRefitPolicy::CountExtreme => "count_extreme",
        BootstrapFailedRefitPolicy::Abort => "abort",
    }
}

fn fixed_effect_null_covariance_policy_label(
    policy: FixedEffectNullCovariancePolicy,
) -> &'static str {
    match policy {
        FixedEffectNullCovariancePolicy::ReuseFittedCovariance => "reuse_fitted_covariance",
    }
}

fn fixed_effect_inference_method(method: &InferenceMethod) -> FixedEffectInferenceMethod {
    match method {
        InferenceMethod::AsymptoticWaldZ => FixedEffectInferenceMethod::AsymptoticWaldZ,
        InferenceMethod::Satterthwaite => FixedEffectInferenceMethod::Satterthwaite,
        InferenceMethod::KenwardRoger => FixedEffectInferenceMethod::KenwardRoger,
        InferenceMethod::ParametricBootstrap => FixedEffectInferenceMethod::Bootstrap,
        InferenceMethod::NotComputed { .. } => FixedEffectInferenceMethod::NotComputed,
    }
}

fn fixed_effect_inference_status(status: &InferenceStatus) -> FixedEffectInferenceStatus {
    match status {
        InferenceStatus::Available => FixedEffectInferenceStatus::Available,
        InferenceStatus::PValueUnavailable { .. } => FixedEffectInferenceStatus::PValueUnavailable,
        InferenceStatus::NotEstimable { .. } => FixedEffectInferenceStatus::NotEstimable,
        InferenceStatus::NotAssessed { .. } => FixedEffectInferenceStatus::NotAssessed,
        InferenceStatus::Unsupported { .. } => FixedEffectInferenceStatus::Unsupported,
    }
}

fn fixed_effect_reliability_reason(test: &FixedEffectTest) -> Option<FixedEffectReliabilityReason> {
    if test.reliability == ReliabilityGrade::NotAvailable {
        return None;
    }
    match test.method {
        InferenceMethod::AsymptoticWaldZ => {
            Some(FixedEffectReliabilityReason::AsymptoticWaldZFallback)
        }
        InferenceMethod::Satterthwaite => {
            Some(FixedEffectReliabilityReason::SatterthwaiteFiniteDifferenceApproximation)
        }
        InferenceMethod::KenwardRoger => {
            Some(FixedEffectReliabilityReason::KenwardRogerApproximation)
        }
        InferenceMethod::ParametricBootstrap => {
            Some(FixedEffectReliabilityReason::ParametricBootstrapMonteCarlo)
        }
        InferenceMethod::NotComputed { .. } => None,
    }
}

fn fixed_effect_statistic_name(test: &FixedEffectTest) -> Option<FixedEffectStatisticName> {
    match test.method {
        InferenceMethod::AsymptoticWaldZ => Some(FixedEffectStatisticName::Z),
        InferenceMethod::Satterthwaite if test.hypothesis.n_contrasts() > 1 => {
            Some(FixedEffectStatisticName::F)
        }
        InferenceMethod::Satterthwaite => Some(FixedEffectStatisticName::T),
        InferenceMethod::KenwardRoger if test.hypothesis.n_contrasts() > 1 => {
            Some(FixedEffectStatisticName::F)
        }
        InferenceMethod::KenwardRoger => Some(FixedEffectStatisticName::T),
        InferenceMethod::ParametricBootstrap if test.hypothesis.n_contrasts() > 1 => {
            Some(FixedEffectStatisticName::F)
        }
        InferenceMethod::ParametricBootstrap => Some(FixedEffectStatisticName::T),
        InferenceMethod::NotComputed { .. } => None,
    }
}

fn fixed_effect_row_numerator_df(
    test: &FixedEffectTest,
    statistic_name: Option<FixedEffectStatisticName>,
) -> Option<f64> {
    match statistic_name {
        Some(FixedEffectStatisticName::F) => test.numerator_df,
        _ => None,
    }
}

pub(super) fn fixed_effect_inference_reason(test: &FixedEffectTest) -> Option<String> {
    match &test.status {
        InferenceStatus::Available => match &test.method {
            InferenceMethod::NotComputed { reason } => Some(reason.clone()),
            _ => None,
        },
        InferenceStatus::PValueUnavailable { reason }
        | InferenceStatus::NotEstimable { reason }
        | InferenceStatus::NotAssessed { reason }
        | InferenceStatus::Unsupported { reason } => Some(reason.clone()),
    }
}

fn finite_option(value: Option<f64>) -> Option<f64> {
    value.filter(|value| value.is_finite())
}

fn fixed_effect_test_asymptotic_wald_z(
    hypothesis: FixedEffectHypothesis,
    estimates: Vec<f64>,
    standard_errors: Vec<Option<f64>>,
    statistics: Vec<Option<f64>>,
    estimability: FixedContrastEstimability,
) -> FixedEffectTest {
    use statrs::distribution::{ContinuousCDF, Normal};

    let normal = Normal::new(0.0, 1.0).unwrap();
    let p_values = statistics
        .iter()
        .map(|stat| stat.map(|z| 2.0 * (1.0 - normal.cdf(z.abs()))))
        .collect::<Vec<_>>();
    let p_value_available = p_values.iter().all(Option::is_some);
    FixedEffectTest {
        hypothesis,
        estimates,
        standard_errors,
        statistics,
        numerator_df: Some(1.0),
        denominator_df: None,
        p_values,
        method: InferenceMethod::AsymptoticWaldZ,
        reliability: ReliabilityGrade::Low,
        status: if p_value_available {
            InferenceStatus::Available
        } else {
            InferenceStatus::PValueUnavailable {
                reason: "standard error is unavailable, so the Wald z p-value is unavailable"
                    .to_string(),
            }
        },
        estimability,
        notes: vec![
            "asymptotic Wald z is a labeled fallback, not a finite-sample correction".to_string(),
        ],
    }
}

fn fixed_effect_test_p_value_unavailable(
    hypothesis: FixedEffectHypothesis,
    estimates: Vec<f64>,
    standard_errors: Vec<Option<f64>>,
    statistics: Vec<Option<f64>>,
    estimability: FixedContrastEstimability,
    reason: String,
) -> FixedEffectTest {
    FixedEffectTest {
        hypothesis,
        estimates,
        standard_errors,
        statistics,
        numerator_df: Some(1.0),
        denominator_df: None,
        p_values: vec![None],
        method: InferenceMethod::NotComputed {
            reason: reason.clone(),
        },
        reliability: ReliabilityGrade::NotAvailable,
        status: InferenceStatus::PValueUnavailable { reason },
        estimability,
        notes: Vec::new(),
    }
}

fn fixed_effect_test_not_assessed_with_method(
    hypothesis: FixedEffectHypothesis,
    estimates: Vec<f64>,
    standard_errors: Vec<Option<f64>>,
    statistics: Vec<Option<f64>>,
    method: InferenceMethod,
    estimability: FixedContrastEstimability,
    reason: String,
) -> FixedEffectTest {
    let n = hypothesis.n_contrasts();
    FixedEffectTest {
        hypothesis,
        estimates,
        standard_errors,
        statistics,
        numerator_df: Some(1.0),
        denominator_df: None,
        p_values: vec![None; n],
        method,
        reliability: ReliabilityGrade::NotAvailable,
        status: InferenceStatus::NotAssessed {
            reason: reason.clone(),
        },
        estimability,
        notes: vec![reason],
    }
}

fn fixed_effect_test_unavailable(
    hypothesis: FixedEffectHypothesis,
    estimability: FixedContrastEstimability,
    status: InferenceStatus,
) -> FixedEffectTest {
    let n = hypothesis.n_contrasts();
    let reason = match &status {
        InferenceStatus::Available => "fixed-effect test unavailable".to_string(),
        InferenceStatus::PValueUnavailable { reason }
        | InferenceStatus::NotEstimable { reason }
        | InferenceStatus::NotAssessed { reason }
        | InferenceStatus::Unsupported { reason } => reason.clone(),
    };
    FixedEffectTest {
        hypothesis,
        estimates: vec![f64::NAN; n],
        standard_errors: vec![None; n],
        statistics: vec![None; n],
        numerator_df: None,
        denominator_df: None,
        p_values: vec![None; n],
        method: InferenceMethod::NotComputed { reason },
        reliability: ReliabilityGrade::NotAvailable,
        status,
        estimability,
        notes: Vec::new(),
    }
}
