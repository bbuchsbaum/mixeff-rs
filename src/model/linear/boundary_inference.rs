//! Explicit, conditional finite-sample inference on a selected covariance face.
//!
//! This module is deliberately opt-in.  It conditions on the fitted active
//! face and does not account for the selection of that face.

use super::*;

const BOUNDARY_NORMALIZED_SD_TOLERANCE: f64 = 1.0e-7;

/// Policy for finite-sample fixed-effect inference at a covariance boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryInferencePolicy {
    /// Preserve the ordinary inference path.
    Default,
    /// Condition on covariance components certified to be exactly zero.
    ActiveSet,
}

/// Fixed-effect inference plus the boundary policy and selected covariance face.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundaryFixedEffectTest {
    pub test: FixedEffectTest,
    pub requested_method: FixedEffectTestMethod,
    pub policy: BoundaryInferencePolicy,
    /// True only when inference was actually conditioned on a selected face.
    #[serde(default)]
    pub conditioning_applied: bool,
    pub fixed_theta_indices: Vec<usize>,
    pub fixed_components: Vec<String>,
}

impl LinearMixedModel {
    /// Test a contrast, optionally conditioning on a KKT-certified covariance face.
    ///
    /// `ActiveSet` requires an explicit Satterthwaite or Kenward–Roger method.
    /// At a boundary it supports unweighted REML models with estimated residual
    /// sigma and independent scalar or diagonal random-effect covariances.
    /// Selected zero components are held fixed before the information matrix
    /// is inverted; residual sigma remains a free parameter.
    ///
    /// Successful conditional results are graded `Low`: they do not include
    /// covariance-structure selection uncertainty. Invalid optimizer stops,
    /// inadequate restricted curvature and existing p-value prohibitions are
    /// not overridden. Interior fits and the `Default` policy retain ordinary
    /// inference. This method does not mutate the fitted model.
    /// See the [boundary inference guide](crate::guide::boundary_inference)
    /// for usage and numerical qualification.
    pub fn test_contrast_with_boundary_policy(
        &self,
        hypothesis: FixedEffectHypothesis,
        method: FixedEffectTestMethod,
        policy: BoundaryInferencePolicy,
    ) -> BoundaryFixedEffectTest {
        if policy == BoundaryInferencePolicy::Default {
            return BoundaryFixedEffectTest {
                test: self.test_contrast_with_method(hypothesis, method),
                requested_method: method,
                policy,
                conditioning_applied: false,
                fixed_theta_indices: Vec::new(),
                fixed_components: Vec::new(),
            };
        }

        let baseline = self.test_contrast_with_method(hypothesis.clone(), method);
        // Retain the ordinary design/contrast guards before any matrix
        // multiplication in the conditional path.
        if hypothesis.n_coefficients() != self.coef_names().len()
            || matches!(baseline.status, InferenceStatus::NotEstimable { .. })
        {
            return BoundaryFixedEffectTest {
                test: baseline,
                requested_method: method,
                policy,
                conditioning_applied: false,
                fixed_theta_indices: Vec::new(),
                fixed_components: Vec::new(),
            };
        }
        let refused =
            |reason: String, fixed_theta_indices: Vec<usize>, fixed_components: Vec<String>| {
                BoundaryFixedEffectTest {
                    test: boundary_refusal(baseline.clone(), reason),
                    requested_method: method,
                    policy,
                    conditioning_applied: false,
                    fixed_theta_indices,
                    fixed_components,
                }
            };

        if !matches!(
            method,
            FixedEffectTestMethod::Satterthwaite | FixedEffectTestMethod::KenwardRoger
        ) {
            return refused(
                "boundary active-set inference requires an explicit Satterthwaite or KenwardRoger method".to_string(),
                Vec::new(), Vec::new(),
            );
        }
        if matches!(
            self.inspection_artifact()
                .optimizer_certificate
                .as_ref()
                .map(|certificate| certificate.status),
            Some(FitStatus::ConvergedInterior)
        ) {
            return BoundaryFixedEffectTest {
                test: baseline,
                requested_method: method,
                policy,
                conditioning_applied: false,
                fixed_theta_indices: Vec::new(),
                fixed_components: Vec::new(),
            };
        }
        if !self.optsum.is_fitted() || !self.optsum.reml {
            return refused(
                "boundary active-set inference requires a fitted REML model".to_string(),
                Vec::new(),
                Vec::new(),
            );
        }
        if !self.sqrtwts.is_empty()
            || self.residual_source
                == crate::model::summary_estimates::ResidualSource::FixedSamplingVariance
        {
            return refused("boundary active-set inference requires an unweighted model with estimated residual sigma".to_string(), Vec::new(), Vec::new());
        }
        if self.optsum.sigma.is_some() || !self.sigma().is_finite() || self.sigma() <= 0.0 {
            return refused(
                "boundary active-set inference requires a finite estimated residual sigma"
                    .to_string(),
                Vec::new(),
                Vec::new(),
            );
        }
        if !self.independent_variance_parameters() {
            return refused("boundary active-set inference supports only independent scalar or diagonal covariance parameters".to_string(), Vec::new(), Vec::new());
        }
        let Some(certificate) = &self.inspection_artifact().optimizer_certificate else {
            return refused(
                "boundary active-set inference requires an optimizer certificate".to_string(),
                Vec::new(),
                Vec::new(),
            );
        };
        if !certificate.evidence.optimizer_stop.acceptable_stop
            || !matches!(
                certificate.status,
                FitStatus::ConvergedBoundary | FitStatus::ConvergedReducedRank
            )
        {
            return refused("boundary active-set inference requires an acceptable converged boundary optimizer stop".to_string(), Vec::new(), Vec::new());
        }
        if baseline.estimability.status == EstimabilityStatus::NotEstimable {
            return refused(
                "boundary active-set inference requires an estimable contrast".to_string(),
                Vec::new(),
                Vec::new(),
            );
        }
        if let CoefTablePValuePolicy::Unavailable { reason } = self.fixed_effect_p_value_policy() {
            return refused(format!("boundary active-set inference cannot override the model p-value policy: {reason}"), Vec::new(), Vec::new());
        }

        let kkt = match self.scalar_covariance_kkt_certificate() {
            Ok(value) => value,
            Err(error) => return refused(format!("boundary active-set inference could not obtain covariance KKT certificate: {error}"), Vec::new(), Vec::new()),
        };
        let mut fixed_theta_indices = Vec::new();
        let mut fixed_components = Vec::new();
        for block in &kkt.blocks {
            if block.normalized_variance <= BOUNDARY_NORMALIZED_SD_TOLERANCE.powi(2)
                && block.classification == CovarianceKktClassification::ValidZeroVariance
            {
                fixed_theta_indices.push(block.theta_index);
                let (term, row, _) = self.parmap[block.theta_index];
                let re = &self.reterms[term];
                fixed_components.push(format!("{term}:{}[{}]", re.grouping_name, re.cnames[row]));
            } else if block.classification == CovarianceKktClassification::ValidZeroVariance {
                return refused(
                    format!("boundary active-set inference requires '{}' to be closer to zero before fixing it (normalized variance {})", block.term, block.normalized_variance),
                    fixed_theta_indices, fixed_components,
                );
            } else if block.classification != CovarianceKktClassification::InteriorConverged {
                return refused(
                    format!("boundary active-set inference requires KKT stationarity for every free covariance component; '{}' is {:?}", block.term, block.classification),
                    fixed_theta_indices, fixed_components,
                );
            }
        }
        if fixed_theta_indices.is_empty() {
            return refused(
                "boundary active-set inference found no KKT-certified zero covariance components"
                    .to_string(),
                fixed_theta_indices,
                fixed_components,
            );
        }

        let theta = self.theta();
        let mut snapped = theta.clone();
        for &index in &fixed_theta_indices {
            snapped[index] = 0.0;
        }
        let mut model = self.clone();
        let (base_logdet, base_pwrss) = model.determinant_term_and_pwrss();
        let snapped_objective = match model.objective_at(&snapped) {
            Ok(value) => value,
            Err(error) => return refused(format!("boundary active-set inference could not evaluate selected covariance face: {error}"), fixed_theta_indices, fixed_components),
        };
        let residual_df = self.dof_residual().max(1) as f64;
        let (snapped_logdet, snapped_pwrss) = model.determinant_term_and_pwrss();
        // Difference likelihood components so additive constants from changing
        // response units do not enter either the value or its tolerance.
        let snap_difference = snapped_logdet - base_logdet
            + residual_df * ((snapped_pwrss - base_pwrss) / base_pwrss).ln_1p();
        let snap_tolerance = 1.0e-7 * residual_df;
        if !snapped_objective.is_finite()
            || !snap_difference.is_finite()
            || snap_difference.abs() > snap_tolerance
        {
            return refused(format!("boundary active-set inference refused to snap selected covariance components: objective changed by {} (tolerance {snap_tolerance})", snap_difference.abs()), fixed_theta_indices, fixed_components);
        }

        let mut varpar = snapped;
        varpar.push(model.sigma());
        let mut coordinate_scales = (0..theta.len())
            .map(|i| self.independent_variance_coordinate_scale(i).sqrt())
            .collect::<Vec<_>>();
        coordinate_scales.push(model.sigma());
        let free_theta = (0..theta.len())
            .filter(|index| !fixed_theta_indices.contains(index))
            .collect::<Vec<_>>();
        let mut free_varpar = free_theta.clone();
        free_varpar.push(theta.len());
        let hessian = match model.hessian_deviance_varpar_scaled_on_indices(
            &varpar,
            true,
            &free_varpar,
            &coordinate_scales,
        ) {
            Ok(value) => value,
            Err(error) => {
                return refused(
                    format!(
                    "boundary active-set inference could not compute restricted Hessian: {error}"
                ),
                    fixed_theta_indices,
                    fixed_components,
                )
            }
        };
        let vcov = match restricted_varpar_covariance(hessian) {
            Ok(value) => value,
            Err(error) => return refused(error.to_string(), fixed_theta_indices, fixed_components),
        };
        let gradient = match model.gradient_deviance_varpar_checked_unrestored(&varpar, true) {
            Ok(value) => value,
            Err(error) => {
                return refused(
                    format!(
                        "boundary active-set inference could not evaluate stationarity: {error}"
                    ),
                    fixed_theta_indices,
                    fixed_components,
                )
            }
        };
        let gradient_tolerance = 1.0e-5 * residual_df;
        if free_varpar.iter().any(|&index| {
            let normalized = gradient[index] * coordinate_scales[index];
            !normalized.is_finite() || normalized.abs() > gradient_tolerance
        }) {
            return refused("boundary active-set inference requires gradient stationarity over free covariance parameters and residual sigma".to_string(), fixed_theta_indices, fixed_components);
        }

        let jacobian = match model.jac_vcov_beta_varpar_analytic(&varpar) {
            Ok(all) => free_varpar.iter().map(|&index| &all[index] * coordinate_scales[index]).collect::<Vec<_>>(),
            Err(error) => return refused(format!("boundary active-set inference could not compute restricted vcov_beta Jacobian: {error}"), fixed_theta_indices, fixed_components),
        };
        let ingredients = std::sync::Arc::new(Ok((jacobian, vcov)));
        let test = match method {
            FixedEffectTestMethod::Satterthwaite => model
                .satterthwaite_fixed_effect_test_with_ingredients(
                    hypothesis,
                    baseline.estimates,
                    baseline.standard_errors,
                    baseline.statistics,
                    baseline.estimability,
                    Some(ingredients),
                ),
            FixedEffectTestMethod::KenwardRoger => {
                let adjusted = restricted_kenward_roger_adjustment(&model, &free_theta);
                match adjusted {
                    Ok(adjusted) if !adjusted.used_generalized_inverse => conditional_kenward_roger_test(&model, baseline, adjusted),
                    Ok(_) => boundary_refusal(baseline, "boundary active-set inference refuses a Kenward-Roger generalized inverse".to_string()),
                    Err(error) => boundary_refusal(baseline, format!("boundary active-set inference could not compute restricted Kenward-Roger adjustment: {error}")),
                }
            }
            _ => unreachable!(),
        };
        let mut test = test;
        let conditioning_applied = test.status == InferenceStatus::Available;
        if conditioning_applied {
            test.reliability = ReliabilityGrade::Low;
            test.notes.push("boundary_active_set; conditional on selected covariance structure; selection uncertainty not included".to_string());
        }
        BoundaryFixedEffectTest {
            test,
            requested_method: method,
            policy,
            conditioning_applied,
            fixed_theta_indices,
            fixed_components,
        }
    }
}

fn restricted_varpar_covariance(hessian: DMatrix<f64>) -> Result<VcovVarparEstimate> {
    let eigen = SymmetricEigen::new(symmetrize_matrix(&hessian));
    let min_eigen = eigen
        .eigenvalues
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let scale = eigen
        .eigenvalues
        .iter()
        .map(|v| v.abs())
        .fold(1.0_f64, f64::max);
    if !min_eigen.is_finite() || min_eigen <= 1e-10 * scale {
        return Err(MixedModelError::Optimization(
            "boundary active-set inference requires a positive-definite restricted Hessian"
                .to_string(),
        ));
    }
    let covariance = LinearMixedModel::vcov_varpar_from_hessian(hessian)?;
    if covariance.used_reduced_rank || covariance.negative_eigenvalues > 0 {
        return Err(MixedModelError::Optimization(
            "boundary active-set inference refuses reduced-rank or indefinite restricted Hessian covariance".to_string(),
        ));
    }
    Ok(covariance)
}

fn boundary_refusal(mut test: FixedEffectTest, reason: String) -> FixedEffectTest {
    test.status = InferenceStatus::Unsupported {
        reason: reason.clone(),
    };
    test.p_values = vec![None; test.p_values.len().max(1)];
    test.denominator_df = None;
    test.reliability = ReliabilityGrade::NotAvailable;
    test.notes.push(reason);
    test
}

fn restricted_kenward_roger_adjustment(
    model: &LinearMixedModel,
    free_theta: &[usize],
) -> Result<KenwardRogerAdjustedVcov> {
    let (phi, x) = model.kenward_roger_phi_and_x()?;
    let all = model.kenward_roger_ingredients_with_response_unit(&x, model.sigma())?;
    let ingredients = select_kenward_roger_components(model, all, free_theta)?;
    model.kenward_roger_adjusted_from_ingredients(phi, ingredients)
}

fn select_kenward_roger_components(
    model: &LinearMixedModel,
    all: kenward_roger::KenwardRogerIngredients,
    free_theta: &[usize],
) -> Result<kenward_roger::KenwardRogerIngredients> {
    let n = model.n_theta() + 1;
    let mut labels = model
        .parmap
        .iter()
        .map(|&(term, row, col)| {
            let re = &model.reterms[term];
            format!(
                "{term}:{}[{},{}]",
                re.grouping_name, re.cnames[row], re.cnames[col]
            )
        })
        .collect::<Vec<_>>();
    labels.push("residual".to_string());
    if all.p_matrices.len() != n
        || all.q_matrices.len() != n * (n + 1) / 2
        || all.ktrace.shape() != (n, n)
        || all.component_labels != labels
        || free_theta.iter().any(|&i| i >= model.n_theta())
    {
        return Err(MixedModelError::InvalidArgument(
            "boundary active-set KR component order or dimensions do not match theta plus residual"
                .to_string(),
        ));
    }
    let mut selected = free_theta.to_vec();
    selected.push(model.n_theta());
    // Absolute variance components a_j use fixed response/design units.
    // Transform P, Q and K together before the information matrix is inverted.
    let sigma_sq = model.sigma().powi(2);
    let scales = selected
        .iter()
        .map(|&i| {
            if i == model.n_theta() {
                sigma_sq
            } else {
                sigma_sq * model.independent_variance_coordinate_scale(i)
            }
        })
        .collect::<Vec<_>>();
    let mut q_matrices = Vec::with_capacity(selected.len() * (selected.len() + 1) / 2);
    for (r, &old_r) in selected.iter().enumerate() {
        for (s, &old_s) in selected.iter().enumerate().skip(r) {
            q_matrices.push(
                &all.q_matrices[symmetric_pair_index(old_r, old_s, n)] * (scales[r] * scales[s]),
            );
        }
    }
    let mut ktrace = DMatrix::zeros(selected.len(), selected.len());
    for (r, &old_r) in selected.iter().enumerate() {
        for (s, &old_s) in selected.iter().enumerate() {
            ktrace[(r, s)] = all.ktrace[(old_r, old_s)] * scales[r] * scales[s];
        }
    }
    Ok(kenward_roger::KenwardRogerIngredients {
        p_matrices: selected
            .iter()
            .enumerate()
            .map(|(i, &index)| &all.p_matrices[index] * scales[i])
            .collect(),
        q_matrices,
        ktrace,
        component_labels: selected
            .iter()
            .map(|&index| all.component_labels[index].clone())
            .collect(),
    })
}

// Equilibrate contrast rows for KR's rank and covariance solves, then return
// estimates/SEs and the hypothesis in the caller's original units.
fn conditional_kenward_roger_test(
    model: &LinearMixedModel,
    baseline: FixedEffectTest,
    adjusted: KenwardRogerAdjustedVcov,
) -> FixedEffectTest {
    let mut hypothesis = baseline.hypothesis.clone();
    let row_scales = baseline
        .standard_errors
        .iter()
        .map(|se| {
            se.filter(|value| value.is_finite() && *value > 0.0)
                .unwrap_or(1.0)
        })
        .collect::<Vec<_>>();
    for (row, &scale) in row_scales.iter().enumerate() {
        hypothesis.l.values.row_mut(row).scale_mut(1.0 / scale);
        hypothesis.rhs.values[row] /= scale;
    }
    let estimates = baseline
        .estimates
        .iter()
        .zip(&row_scales)
        .map(|(v, s)| v / s)
        .collect();
    let standard_errors = baseline
        .standard_errors
        .iter()
        .zip(&row_scales)
        .map(|(v, s)| v.map(|v| v / s))
        .collect();
    let mut result = model.kenward_roger_fixed_effect_test_with_adjusted(
        hypothesis,
        estimates,
        standard_errors,
        baseline.statistics,
        baseline.estimability,
        Ok(adjusted),
    );
    result.hypothesis = baseline.hypothesis;
    result.estimates = baseline.estimates;
    for (error, scale) in result.standard_errors.iter_mut().zip(row_scales) {
        *error = error.map(|value| value * scale);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kr_component_selection_rejects_dimension_and_order_drift() {
        let model = strong_intercept_fit();
        let (_, x) = model.kenward_roger_phi_and_x().unwrap();
        let mut all = model.kenward_roger_ingredients(&x).unwrap();
        all.component_labels.swap(0, 1);
        assert!(select_kenward_roger_components(&model, all, &[0]).is_err());
        let mut all = model.kenward_roger_ingredients(&x).unwrap();
        all.p_matrices.pop();
        assert!(select_kenward_roger_components(&model, all, &[0]).is_err());
    }

    fn strong_intercept_fit() -> LinearMixedModel {
        let mut data = DataFrame::new();
        data.add_numeric(
            "y",
            (0..48)
                .map(|i| (i / 6) as f64 + [0.1, -0.2, 0.3, -0.1, 0.2, -0.3][i % 6])
                .collect(),
        )
        .unwrap();
        data.add_categorical("g", (0..48).map(|i| format!("g{}", i / 6)).collect())
            .unwrap();
        let mut model = LinearMixedModel::new(
            crate::formula::parse_formula("y ~ 1 + (1 | g)").unwrap(),
            &data,
            None,
        )
        .unwrap();
        model.fit(true).unwrap();
        model
    }

    #[test]
    fn invalid_boundary_is_refused_even_with_a_stale_acceptable_certificate() {
        let mut model = strong_intercept_fit();
        model.objective_at(&[0.0]).unwrap();
        // Pretend the backend accepted the bad boundary point. The inference
        // route must independently check covariance scores, not theta gradients.
        let certificate = model
            .compiler_artifact
            .optimizer_certificate
            .as_mut()
            .unwrap();
        certificate.status = FitStatus::ConvergedBoundary;
        model.inspection_artifact = std::sync::OnceLock::new();
        model.derivative_evidence_pending = false;
        for method in [
            FixedEffectTestMethod::Satterthwaite,
            FixedEffectTestMethod::KenwardRoger,
        ] {
            let result = model.test_contrast_with_boundary_policy(
                model.coefficient_hypotheses().remove(0),
                method,
                BoundaryInferencePolicy::ActiveSet,
            );
            assert!(!result.conditioning_applied);
            assert!(result.test.p_values.iter().all(Option::is_none));
            assert!(
                result
                    .test
                    .p_value_unavailable_reason()
                    .unwrap()
                    .contains("KKT"),
                "{result:?}"
            );
        }
    }

    #[test]
    fn restricted_curvature_refuses_rank_deficiency_and_negative_directions() {
        for diagonal in [[2.0, 0.0], [2.0, -1.0], [2.0, 1e-12]] {
            let hessian = DMatrix::from_diagonal(&DVector::from_vec(diagonal.to_vec()));
            assert!(restricted_varpar_covariance(hessian).is_err());
        }
        // Independent quadratic with curvatures 2 and 4 has 2 H^-1 = (1, .5).
        let hessian = DMatrix::from_diagonal(&DVector::from_vec(vec![2.0, 4.0]));
        let result = restricted_varpar_covariance(hessian).unwrap();
        assert!((result.covariance[(0, 0)] - 1.0).abs() < 1e-12);
        assert!((result.covariance[(1, 1)] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn active_policy_cannot_override_selection_time_refusal() {
        let mut data = DataFrame::new();
        data.add_numeric(
            "y",
            (0..48).map(|i| [-2.0, -1.0, 1.0, 2.0][i % 4]).collect(),
        )
        .unwrap();
        data.add_categorical("g", (0..48).map(|i| format!("g{}", i / 4)).collect())
            .unwrap();
        let mut model = LinearMixedModel::new(
            crate::formula::parse_formula("y ~ 1 + (1 | g)").unwrap(),
            &data,
            None,
        )
        .unwrap();
        model.fit(true).unwrap();
        model.compiler_artifact.reductions.push(ReductionRecord {
            trigger: ReductionTrigger::SelectionTime,
            phase: "selection".into(),
            reason: "test selection".into(),
            affected_term: "g".into(),
            replacement_term: None,
            inference_consequence: "unavailable".into(),
            diagnostics: Vec::new(),
        });
        for method in [
            FixedEffectTestMethod::Satterthwaite,
            FixedEffectTestMethod::KenwardRoger,
        ] {
            let result = model.test_contrast_with_boundary_policy(
                model.coefficient_hypotheses().remove(0),
                method,
                BoundaryInferencePolicy::ActiveSet,
            );
            assert!(!result.conditioning_applied);
            assert!(result.test.p_values.iter().all(Option::is_none));
            assert!(
                result
                    .test
                    .p_value_unavailable_reason()
                    .unwrap()
                    .contains("p-value policy"),
                "{result:?}"
            );
        }
    }

    #[test]
    fn active_policy_preserves_interior_results() {
        let model = strong_intercept_fit();
        for method in [
            FixedEffectTestMethod::Satterthwaite,
            FixedEffectTestMethod::KenwardRoger,
        ] {
            let h = model.coefficient_hypotheses().remove(0);
            let baseline = model.test_contrast_with_method(h.clone(), method);
            let result = model.test_contrast_with_boundary_policy(
                h,
                method,
                BoundaryInferencePolicy::ActiveSet,
            );
            assert!(!result.conditioning_applied);
            assert_eq!(result.test, baseline);
        }
    }
}
