//! Prediction on training and new data for the linear mixed model: fixed-effect
//! predictors, alignment of newdata to the training factor encoding, and
//! prediction-variance decomposition. Moved verbatim from the former
//! single-file `linear.rs`.

use super::*;

impl LinearMixedModel {
    /// Predictions on the training data (identical to `fitted()`).
    pub fn predict(&self) -> DVector<f64> {
        self.fitted()
    }

    /// Population-level (fixed-effects-only) fitted values on the training
    /// data: the marginal linear predictor `Xβ`, **excluding** the random
    /// effects contribution that [`Self::predict`]/`fitted` include.
    ///
    /// Equivalent to `lme4`'s `predict(model, re.form = NA)` on the training
    /// frame, using the full-rank fixed-effects design.
    ///
    /// Stable public API for downstream frontends that need lme4-compatible
    /// population predictions for the training frame without exposing the
    /// internal fixed-effect design storage.
    pub fn fixed_effect_fitted(&self) -> DVector<f64> {
        self.feterm.full_rank_x() * &self.beta()
    }

    /// Names of categorical columns that participate in the *fixed-effects*
    /// design (directly or via an interaction). Only these need training-time
    /// realignment; grouping-only categoricals are handled training-anchored by
    /// the random-effects path and may legitimately carry unseen levels.
    fn fixed_effect_predictor_names(&self) -> std::collections::HashSet<String> {
        use crate::formula::FixedTerm;
        let mut names = std::collections::HashSet::new();
        for term in &self.formula.fixed_terms {
            match term {
                FixedTerm::Column(name) => {
                    names.insert(name.clone());
                }
                FixedTerm::Interaction(vars) => {
                    for v in vars {
                        names.insert(v.clone());
                    }
                }
                FixedTerm::Intercept | FixedTerm::NoIntercept => {}
            }
        }
        names
    }

    /// Rebuild `newdata` so every fixed-effect categorical column reuses the
    /// training-time level order (and explicit contrast). This makes the
    /// predict-time fixed-effects encoding identical to training regardless of
    /// observation order in `newdata`. A categorical value absent from the
    /// training levels is rejected here rather than silently absorbed into the
    /// reference cell. Categorical columns that are *not* fixed-effect
    /// predictors (e.g. RE grouping factors) are passed through unchanged so
    /// the `NewReLevels` policy still governs unseen grouping levels.
    fn align_newdata_to_training(&self, newdata: &DataFrame) -> Result<DataFrame> {
        let fe_predictors = self.fixed_effect_predictor_names();
        let names: Vec<String> = newdata
            .column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        let mut aligned = DataFrame::new();
        for name in names {
            match newdata.column(&name) {
                Some(Column::Numeric(v)) => {
                    aligned.add_numeric_unchecked(&name, v.clone())?;
                }
                Some(Column::Categorical(cat)) => {
                    let snap = if fe_predictors.contains(&name) {
                        self.training_categorical.get(&name)
                    } else {
                        None
                    };
                    match snap {
                        Some(snap) => {
                            if let Some(contrast) = &snap.contrast {
                                aligned.add_categorical_with_contrast(
                                    &name,
                                    cat.values.clone(),
                                    snap.levels.clone(),
                                    contrast.clone(),
                                )?;
                            } else {
                                aligned.add_categorical_with_levels(
                                    &name,
                                    cat.values.clone(),
                                    snap.levels.clone(),
                                )?;
                            }
                        }
                        None => {
                            aligned.add_categorical_with_levels(
                                &name,
                                cat.values.clone(),
                                cat.levels.clone(),
                            )?;
                        }
                    }
                }
                None => unreachable!("column name came from this frame"),
            }
        }
        Ok(aligned)
    }

    /// Predictions for new data with configurable handling of unseen RE levels.
    pub fn predict_new(
        &self,
        newdata: &DataFrame,
        new_re_levels: NewReLevels,
    ) -> Result<Vec<Option<f64>>> {
        let beta = self.beta();
        let b_list = self.ranef_b();
        self.linear_predict_new_with_state(newdata, &beta, &b_list, new_re_levels)
    }

    pub(crate) fn predict_new_design(
        &self,
        newdata: &DataFrame,
    ) -> Result<(
        DataFrame,
        DMatrix<f64>,
        std::collections::HashMap<String, usize>,
    )> {
        // Re-run the stateless transform evaluator on `newdata`. Correct by
        // construction: each transform is a pure pointwise recipe, so there
        // is no stored basis to diverge from — prediction simply re-evaluates
        // the same expression. See `docs/formula_transform_seam.md`.
        let materialized = self.formula.materialize(newdata)?;

        // Realign categorical columns to the training factor encoding so that
        // newdata's own observation order cannot reorder/drop dummy columns.
        let aligned = self.align_newdata_to_training(&materialized)?;
        let (raw_x, raw_names) = build_fixed_effects_matrix(&self.formula, &aligned)?;

        let name_to_col = raw_names
            .into_iter()
            .enumerate()
            .map(|(i, n)| (n, i))
            .collect();

        Ok((materialized, raw_x, name_to_col))
    }

    /// New-data linear predictor using caller-supplied fixed/random effects.
    ///
    /// GLMMs share the LMM formula lowering, training-anchored categorical
    /// encoding, and random-effect level policy, but their fitted β and
    /// conditional modes live on the GLMM wrapper rather than this inner LMM.
    pub(crate) fn linear_predict_new_with_state(
        &self,
        newdata: &DataFrame,
        beta: &DVector<f64>,
        b_list: &[DMatrix<f64>],
        new_re_levels: NewReLevels,
    ) -> Result<Vec<Option<f64>>> {
        let n_new = newdata.nrow();
        if beta.len() != self.feterm.rank {
            return Err(MixedModelError::DimensionMismatch(format!(
                "prediction beta length {} does not match fixed-effect rank {}",
                beta.len(),
                self.feterm.rank
            )));
        }
        if b_list.len() != self.reterms.len() {
            return Err(MixedModelError::DimensionMismatch(format!(
                "prediction random-effect term count {} does not match fitted term count {}",
                b_list.len(),
                self.reterms.len()
            )));
        }
        for (term_idx, (rt, b)) in self.reterms.iter().zip(b_list.iter()).enumerate() {
            if b.nrows() != rt.vsize || b.ncols() != rt.n_levels() {
                return Err(MixedModelError::DimensionMismatch(format!(
                    "prediction random-effect matrix for term {term_idx} has shape {}x{}, expected {}x{}",
                    b.nrows(),
                    b.ncols(),
                    rt.vsize,
                    rt.n_levels()
                )));
            }
        }

        let (materialized, raw_x, name_to_col) = self.predict_new_design(newdata)?;
        let newdata = &materialized;

        let p = self.feterm.rank;
        let mut fe_pred = vec![0.0f64; n_new];

        for new_col in 0..p {
            // feterm.cnames[new_col] is the column name at pivot position new_col
            let name = &self.feterm.cnames[new_col];
            if let Some(&raw_col) = name_to_col.get(name) {
                for obs in 0..n_new {
                    fe_pred[obs] += raw_x[(obs, raw_col)] * beta[new_col];
                }
            }
            // Column absent from newdata → treat as 0 contribution
        }

        // --- Random-effects part ---
        // Build level-name → index maps for each RE term (training levels)
        let level_maps: Vec<std::collections::HashMap<&str, usize>> = self
            .reterms
            .iter()
            .map(|rt| {
                rt.levels
                    .iter()
                    .enumerate()
                    .map(|(i, s)| (s.as_str(), i))
                    .collect()
            })
            .collect();

        let mut result: Vec<Option<f64>> = fe_pred.into_iter().map(Some).collect();

        for (term_idx, rt) in self.reterms.iter().enumerate() {
            let b = &b_list[term_idx];
            let level_map = &level_maps[term_idx];

            let new_level_names = self.get_new_grouping_levels(rt, newdata)?;

            for obs in 0..n_new {
                if result[obs].is_none() {
                    continue;
                }
                let level_name = &new_level_names[obs];
                match level_map.get(level_name.as_str()) {
                    Some(&level_idx) => {
                        let z_obs = self.get_z_for_obs(rt, newdata, obs)?;
                        let re_contrib: f64 =
                            (0..rt.vsize).map(|s| z_obs[s] * b[(s, level_idx)]).sum();
                        *result[obs].as_mut().unwrap() += re_contrib;
                    }
                    None => match new_re_levels {
                        NewReLevels::Error => {
                            return Err(MixedModelError::InvalidArgument(format!(
                                "New level '{}' in grouping factor '{}'. \
                                 Use NewReLevels::Population or ::Missing to allow this.",
                                level_name, rt.grouping_name
                            )));
                        }
                        NewReLevels::Population => {} // zero RE, nothing to add
                        NewReLevels::Missing => {
                            result[obs] = None;
                        }
                    },
                }
            }
        }

        Ok(result)
    }

    /// Prediction variance for new data, including fixed-effect and
    /// conditional random-effect uncertainty on the LMM identity-link scale.
    ///
    /// Rows with unseen grouping levels under [`NewReLevels::Population`] or
    /// [`NewReLevels::Missing`] return the point-prediction policy result but
    /// mark the combined variance unavailable with a reason. This keeps the
    /// no-fake-certainty contract: the engine does not substitute zero random
    /// uncertainty for a level whose conditional covariance is unavailable.
    pub fn predict_new_variance(
        &self,
        newdata: &DataFrame,
        new_re_levels: NewReLevels,
    ) -> Result<PredictionVariancePayload> {
        self.predict_new_variance_with_level(newdata, new_re_levels, 0.95)
    }

    /// Prediction variance and intervals for new data at the requested
    /// confidence level.
    pub fn predict_new_variance_with_level(
        &self,
        newdata: &DataFrame,
        new_re_levels: NewReLevels,
        level: f64,
    ) -> Result<PredictionVariancePayload> {
        if self.optsum.feval <= 0 {
            return Err(MixedModelError::NotFitted);
        }
        let z = prediction_interval_cutoff(level)?;

        let predictions = self.predict_new(newdata, new_re_levels)?;
        let n_new = newdata.nrow();
        let (materialized, raw_x, name_to_col) = self.predict_new_design(newdata)?;
        let newdata = &materialized;
        let sigma_sq = self.sigma().powi(2);
        let offsets = self.prediction_system_offsets();

        let level_maps: Vec<std::collections::HashMap<&str, usize>> = self
            .reterms
            .iter()
            .map(|rt| {
                rt.levels
                    .iter()
                    .enumerate()
                    .map(|(i, s)| (s.as_str(), i))
                    .collect()
            })
            .collect();
        let level_names_by_term = self
            .reterms
            .iter()
            .map(|rt| self.get_new_grouping_levels(rt, newdata))
            .collect::<Result<Vec<_>>>()?;

        let mut rows = Vec::with_capacity(n_new);
        for obs in 0..n_new {
            let mut reason: Option<String> = None;

            for (term_idx, rt) in self.reterms.iter().enumerate() {
                let level_name = &level_names_by_term[term_idx][obs];
                match level_maps[term_idx].get(level_name.as_str()) {
                    Some(_) => {}
                    None => match new_re_levels {
                        NewReLevels::Error => {
                            return Err(MixedModelError::InvalidArgument(format!(
                                "New level '{}' in grouping factor '{}'. \
                                 Use NewReLevels::Population or ::Missing to allow this.",
                                level_name, rt.grouping_name
                            )));
                        }
                        NewReLevels::Population | NewReLevels::Missing => {
                            reason.get_or_insert_with(|| {
                                format!(
                                    "prediction variance unavailable for new level '{}' in grouping factor '{}'",
                                    level_name, rt.grouping_name
                                )
                            });
                        }
                    },
                }
            }

            let mut fixed_variance =
                clean_prediction_variance_component(self.prediction_fixed_variance_for_obs(
                    obs,
                    &raw_x,
                    &name_to_col,
                    &offsets,
                    sigma_sq,
                )?);
            if fixed_variance.is_none() && reason.is_none() {
                reason.get_or_insert_with(|| {
                    "fixed-effect prediction variance is non-finite or negative".to_string()
                });
            }
            let mut random_variance = None;
            let mut fixed_random_covariance = None;
            let mut combined_variance = None;

            if reason.is_none() {
                let components = self.prediction_variance_components_for_obs(
                    obs,
                    newdata,
                    &raw_x,
                    &name_to_col,
                    &level_maps,
                    &level_names_by_term,
                    &offsets,
                    sigma_sq,
                )?;

                fixed_variance = clean_prediction_variance_component(components.fixed_variance);
                if fixed_variance.is_none() {
                    reason.get_or_insert_with(|| {
                        "fixed-effect prediction variance is non-finite or negative".to_string()
                    });
                }

                random_variance = clean_prediction_variance_component(components.random_variance);
                if random_variance.is_none() {
                    reason.get_or_insert_with(|| {
                        "random-effect prediction variance is non-finite or negative".to_string()
                    });
                }

                fixed_random_covariance =
                    clean_prediction_covariance_component(components.fixed_random_covariance);
                if fixed_random_covariance.is_none() {
                    reason.get_or_insert_with(|| {
                        "fixed/random prediction covariance is non-finite".to_string()
                    });
                }

                combined_variance =
                    clean_prediction_variance_component(components.combined_variance);
                if combined_variance.is_none() {
                    reason.get_or_insert_with(|| {
                        "combined prediction variance is non-finite or negative".to_string()
                    });
                }
            }
            let se_fit = combined_variance.map(f64::sqrt);
            let prediction_variance = if reason.is_none() {
                combined_variance
                    .and_then(|combined| clean_prediction_variance_component(combined + sigma_sq))
            } else {
                None
            };
            let (confidence_lower, confidence_upper, prediction_lower, prediction_upper) = match (
                predictions[obs],
                se_fit,
                prediction_variance.map(f64::sqrt),
                reason.is_none(),
            ) {
                (Some(prediction), Some(se_fit), Some(prediction_se), true) => (
                    Some(prediction - z * se_fit),
                    Some(prediction + z * se_fit),
                    Some(prediction - z * prediction_se),
                    Some(prediction + z * prediction_se),
                ),
                _ => (None, None, None, None),
            };
            let status = if reason.is_none() {
                PredictionVarianceStatus::Available
            } else {
                PredictionVarianceStatus::Unavailable
            };

            rows.push(PredictionVarianceRow {
                row: obs,
                prediction: predictions[obs],
                fixed_variance,
                random_variance,
                fixed_random_covariance,
                combined_variance,
                se_fit,
                prediction_variance,
                confidence_lower,
                confidence_upper,
                prediction_lower,
                prediction_upper,
                status,
                reason,
            });
        }

        Ok(PredictionVariancePayload::new(
            PredictionVarianceMethod::LmmConditionalModeCovariance,
            rows,
            Some(level),
            vec![
                "fixed component is x V_beta x' on the fitted LMM identity-link scale".to_string(),
                "random component is the random-effect-only row variance from the joint penalized Cholesky solve"
                    .to_string(),
                "combined fitted-mean variance includes the fixed/random cross covariance term"
                    .to_string(),
                "confidence intervals use the combined fitted-mean variance; prediction intervals additionally include residual variance"
                    .to_string(),
            ],
        ))
    }

    pub(crate) fn fixed_prediction_design_for_obs(
        &self,
        obs: usize,
        raw_x: &DMatrix<f64>,
        name_to_col: &std::collections::HashMap<String, usize>,
    ) -> DVector<f64> {
        let p = self.feterm.rank;
        let mut x = DVector::zeros(p);
        for active_col in 0..p {
            let name = &self.feterm.cnames[active_col];
            if let Some(&raw_col) = name_to_col.get(name) {
                x[active_col] = raw_x[(obs, raw_col)];
            }
        }
        x
    }

    fn prediction_system_offsets(&self) -> Vec<usize> {
        let k = self.reterms.len();
        let mut offsets = vec![0usize; k + 1];
        for j in 0..k {
            offsets[j + 1] = offsets[j] + self.reterms[j].n_ranef();
        }
        offsets
    }

    fn prediction_variance_components_for_obs(
        &self,
        obs: usize,
        newdata: &DataFrame,
        raw_x: &DMatrix<f64>,
        name_to_col: &std::collections::HashMap<String, usize>,
        level_maps: &[std::collections::HashMap<&str, usize>],
        level_names_by_term: &[Vec<String>],
        offsets: &[usize],
        sigma_sq: f64,
    ) -> Result<PredictionVarianceComponents> {
        let k = self.reterms.len();
        let p = self.feterm.rank;
        let pp1 = p + 1;
        let nranef_total = offsets[k];
        let len = nranef_total + pp1;
        let mut fixed = vec![0.0; len];
        let mut random = vec![0.0; len];

        let x = self.fixed_prediction_design_for_obs(obs, raw_x, name_to_col);
        for col in 0..p {
            fixed[nranef_total + col] = x[col];
        }

        for (term_idx, rt) in self.reterms.iter().enumerate() {
            let level_name = &level_names_by_term[term_idx][obs];
            let Some(&level_idx) = level_maps[term_idx].get(level_name.as_str()) else {
                continue;
            };
            let z_obs = self.get_z_for_obs(rt, newdata, obs)?;
            let offset = offsets[term_idx] + level_idx * rt.vsize;
            for col in 0..rt.vsize {
                let mut value = 0.0;
                for row in col..rt.vsize {
                    value += rt.lambda[(row, col)] * z_obs[row];
                }
                random[offset + col] = value;
            }
        }

        let mut combined = fixed.clone();
        for (dst, src) in combined.iter_mut().zip(random.iter()) {
            *dst += *src;
        }

        let h_fixed = self.prediction_design_norm_sq(&fixed, offsets)?;
        let h_random = self.prediction_design_norm_sq(&random, offsets)?;
        let h_combined = self.prediction_design_norm_sq(&combined, offsets)?;
        let fixed_variance = sigma_sq * h_fixed;
        let random_variance = sigma_sq * h_random;
        let combined_variance = sigma_sq * h_combined;
        let fixed_random_covariance = 0.5 * (combined_variance - fixed_variance - random_variance);

        Ok(PredictionVarianceComponents {
            fixed_variance,
            random_variance,
            fixed_random_covariance,
            combined_variance,
        })
    }

    fn prediction_fixed_variance_for_obs(
        &self,
        obs: usize,
        raw_x: &DMatrix<f64>,
        name_to_col: &std::collections::HashMap<String, usize>,
        offsets: &[usize],
        sigma_sq: f64,
    ) -> Result<f64> {
        let k = self.reterms.len();
        let p = self.feterm.rank;
        let pp1 = p + 1;
        let nranef_total = offsets[k];
        let mut fixed = vec![0.0; nranef_total + pp1];
        let x = self.fixed_prediction_design_for_obs(obs, raw_x, name_to_col);
        for col in 0..p {
            fixed[nranef_total + col] = x[col];
        }
        Ok(sigma_sq * self.prediction_design_norm_sq(&fixed, offsets)?)
    }

    fn prediction_design_norm_sq(&self, v: &[f64], offsets: &[usize]) -> Result<f64> {
        let k = self.reterms.len();
        let p = self.feterm.rank;
        let pp1 = p + 1;
        let nranef_total = offsets[k];
        let expected_len = nranef_total + pp1;
        if v.len() != expected_len {
            return Err(MixedModelError::DimensionMismatch(format!(
                "prediction variance row has length {}, expected {}",
                v.len(),
                expected_len
            )));
        }

        let mut w = vec![0.0f64; expected_len];

        for j in 0..k {
            let nranef_j = self.reterms[j].n_ranef();
            let mut rhs = vec![0.0f64; nranef_j];
            for idx in 0..nranef_j {
                rhs[idx] = v[offsets[j] + idx];
            }
            for m in 0..j {
                let l_jm = self.l_blocks[block_index(j, m)].as_dense();
                let nranef_m = self.reterms[m].n_ranef();
                for row in 0..nranef_j {
                    let mut dot = 0.0;
                    for col in 0..nranef_m {
                        dot += l_jm[(row, col)] * w[offsets[m] + col];
                    }
                    rhs[row] -= dot;
                }
            }

            solve_lower_block_against_rhs(&self.l_blocks[block_index(j, j)], &mut rhs);
            for idx in 0..nranef_j {
                w[offsets[j] + idx] = rhs[idx];
            }
        }

        let mut rhs_k = vec![0.0f64; pp1];
        rhs_k.copy_from_slice(&v[nranef_total..nranef_total + pp1]);
        for j in 0..k {
            let l_kj = self.l_blocks[block_index(k, j)].as_dense();
            let nranef_j = self.reterms[j].n_ranef();
            for row in 0..pp1 {
                let mut dot = 0.0;
                for col in 0..nranef_j {
                    dot += l_kj[(row, col)] * w[offsets[j] + col];
                }
                rhs_k[row] -= dot;
            }
        }

        let l_kk = self.l_blocks[block_index(k, k)].as_dense();
        solve_lower_block_against_rhs(&MatrixBlock::Dense(l_kk), &mut rhs_k);

        let sum_sq: f64 = w[..nranef_total]
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            + rhs_k[..p].iter().map(|value| value * value).sum::<f64>();
        Ok(sum_sq)
    }

    /// Collect the grouping-factor level string for each observation in `newdata`.
    fn get_new_grouping_levels(&self, rt: &ReMat, newdata: &DataFrame) -> Result<Vec<String>> {
        use crate::formula::GroupingFactor;

        for re_term in &self.formula.random_terms {
            if random_term_grouping_name(re_term) != rt.grouping_name {
                continue;
            }
            return match &re_term.grouping {
                GroupingFactor::Single(name) => {
                    let cat = newdata.categorical(name).ok_or_else(|| {
                        MixedModelError::InvalidArgument(format!(
                            "Grouping factor '{}' not found in newdata",
                            name
                        ))
                    })?;
                    Ok(cat.values.clone())
                }
                GroupingFactor::Interaction(names) => {
                    let cats: Vec<_> = names
                        .iter()
                        .map(|n| {
                            newdata.categorical(n).ok_or_else(|| {
                                MixedModelError::InvalidArgument(format!(
                                    "Grouping factor '{}' not found in newdata",
                                    n
                                ))
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let levels = (0..newdata.nrow())
                        .map(|i| {
                            cats.iter()
                                .map(|c| c.values[i].clone())
                                .collect::<Vec<_>>()
                                .join("_")
                        })
                        .collect();
                    Ok(levels)
                }
                GroupingFactor::Cell(names) => {
                    let cats: Vec<_> = names
                        .iter()
                        .map(|n| {
                            newdata.categorical(n).ok_or_else(|| {
                                MixedModelError::InvalidArgument(format!(
                                    "Grouping factor '{}' not found in newdata",
                                    n
                                ))
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let levels = (0..newdata.nrow())
                        .map(|i| {
                            cats.iter()
                                .map(|c| c.values[i].clone())
                                .collect::<Vec<_>>()
                                .join("_")
                        })
                        .collect();
                    Ok(levels)
                }
            };
        }
        Err(MixedModelError::InvalidArgument(format!(
            "RE term '{}' not found in formula",
            rt.grouping_name
        )))
    }

    /// Build the z covariate vector for observation `obs` from `newdata`.
    fn get_z_for_obs(&self, rt: &ReMat, newdata: &DataFrame, obs: usize) -> Result<Vec<f64>> {
        for re_term in &self.formula.random_terms {
            if random_term_grouping_name(re_term) != rt.grouping_name {
                continue;
            }
            let (z, cnames) = random_term_z_for_obs(re_term, newdata, obs)?;
            if cnames == rt.cnames {
                return Ok(z);
            }
        }
        Err(MixedModelError::InvalidArgument(format!(
            "RE term '{}' with basis [{}] not found in formula",
            rt.grouping_name,
            rt.cnames.join(", ")
        )))
    }
}

fn random_term_grouping_name(rt: &crate::formula::RandomTerm) -> String {
    use crate::formula::GroupingFactor;

    match &rt.grouping {
        GroupingFactor::Single(name) => name.clone(),
        GroupingFactor::Interaction(names) | GroupingFactor::Cell(names) => names.join(" & "),
    }
}

struct PredictionVarianceComponents {
    fixed_variance: f64,
    random_variance: f64,
    fixed_random_covariance: f64,
    combined_variance: f64,
}

fn clean_prediction_variance_component(value: f64) -> Option<f64> {
    if !value.is_finite() || value < -1.0e-10 {
        None
    } else {
        Some(value.max(0.0))
    }
}

fn clean_prediction_covariance_component(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

fn random_term_z_for_obs(
    rt: &crate::formula::RandomTerm,
    data: &DataFrame,
    obs: usize,
) -> Result<(Vec<f64>, Vec<String>)> {
    use crate::formula::FixedTerm;

    let mut z = Vec::new();
    let mut cnames = Vec::new();
    let has_intercept =
        rt.terms.iter().any(|t| matches!(t, FixedTerm::Intercept)) || rt.terms.is_empty();
    if has_intercept {
        z.push(1.0);
        cnames.push("(Intercept)".to_string());
    }

    let basis_coding = random_effect_basis_coding(rt);
    for term in &rt.terms {
        for (col, name) in random_effect_basis_columns(term, data, data.nrow(), basis_coding)? {
            z.push(col[obs]);
            cnames.push(name);
        }
    }

    Ok((z, cnames))
}

/// Build the fixed-effects model matrix from formula and data.
/// Capture the canonical level order (and explicit contrast, if present) of
/// every categorical column in the training frame. Numeric columns carry no
/// encoding contract and are skipped.
pub(super) fn snapshot_training_categorical(
    data: &DataFrame,
) -> std::collections::HashMap<String, TrainingCategoricalLevels> {
    let mut map = std::collections::HashMap::new();
    let names: Vec<String> = data.column_names().iter().map(|s| s.to_string()).collect();
    for name in names {
        if let Some(Column::Categorical(cat)) = data.column(&name) {
            map.insert(
                name,
                TrainingCategoricalLevels {
                    levels: cat.levels.clone(),
                    contrast: cat.contrast.clone(),
                },
            );
        }
    }
    map
}
