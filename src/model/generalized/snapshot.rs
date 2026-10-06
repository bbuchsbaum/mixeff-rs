use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::error::{MixedModelError, Result};
use crate::model::linear::LinearMixedModel;
use crate::model::snapshot::{decode_snapshot, encode_snapshot};
use crate::model::traits::{Family, LinkFunction};

use super::{GeneralizedLinearMixedModel, PirlsProfiledOptimumCertificate};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FittedStateSeal {
    y: Vec<f64>,
    wt: Vec<f64>,
    offset: Vec<f64>,
    beta: Vec<f64>,
    theta: Vec<f64>,
    u: Vec<Vec<f64>>,
    dispersion: f64,
    nb_theta: Option<f64>,
    family: Family,
    link: LinkFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GlmmSnapshot {
    lmm_json: String,
    beta: Vec<f64>,
    theta: Vec<f64>,
    u: Vec<MatrixSnapshot>,
    y: Vec<f64>,
    wt: Vec<f64>,
    offset: Vec<f64>,
    dispersion: f64,
    negative_binomial_theta: Option<f64>,
    negative_binomial_estimate_theta: bool,
    family: Family,
    link: LinkFunction,
    eta: Vec<f64>,
    mu: Vec<f64>,
    objective_witness: f64,
    pirls_certificate: Option<std::result::Result<PirlsProfiledOptimumCertificate, String>>,
    artifact: crate::compiler::CompiledModelArtifact,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MatrixSnapshot {
    rows: usize,
    cols: usize,
    values: Vec<f64>,
}

impl GeneralizedLinearMixedModel {
    pub(crate) fn seal_fitted_state(&mut self) {
        self.fitted_state_seal = Some(FittedStateSeal {
            y: self.y.as_slice().to_vec(),
            wt: self.wt.clone(),
            offset: self.offset.as_slice().to_vec(),
            beta: self.beta.as_slice().to_vec(),
            theta: self.theta.clone(),
            u: self.u.iter().map(|m| m.as_slice().to_vec()).collect(),
            dispersion: self.dispersion,
            nb_theta: self.negative_binomial_theta,
            family: self.family,
            link: self.link,
        });
    }

    /// Serialize a fitted GLMM with its exact last PIRLS working LMM state.
    pub fn snapshot_json(&self) -> Result<String> {
        let seal = self.fitted_state_seal.as_ref().ok_or_else(|| {
            MixedModelError::InvalidArgument(
                "cannot snapshot an unsealed or stale fitted GLMM".to_string(),
            )
        })?;
        if !seal_matches(self, seal) {
            return Err(MixedModelError::InvalidArgument(
                "cannot snapshot GLMM after public fitted-state fields changed".to_string(),
            ));
        }
        validate_glmm_metadata(self)?;
        let fixed_state_objective = fixed_state_deviance(self)?;
        if !same_scalar(fixed_state_objective, self.lmm.optsum.fmin) {
            return Err(MixedModelError::InvalidArgument(
                "GLMM snapshot fixed-state deviance contradicts optimizer fmin".to_string(),
            ));
        }
        let payload = GlmmSnapshot {
            lmm_json: self.lmm.snapshot_json_glmm()?,
            beta: self.beta.as_slice().to_vec(),
            theta: self.theta.clone(),
            u: self.u.iter().map(matrix_snapshot).collect(),
            y: self.y.as_slice().to_vec(),
            wt: self.wt.clone(),
            offset: self.offset.as_slice().to_vec(),
            dispersion: self.dispersion,
            negative_binomial_theta: self.negative_binomial_theta,
            negative_binomial_estimate_theta: self.negative_binomial_estimate_theta,
            family: self.family,
            link: self.link,
            eta: self.eta.as_slice().to_vec(),
            mu: self.mu.as_slice().to_vec(),
            objective_witness: fixed_state_objective,
            pirls_certificate: self.pirls_profiled_optimum_certificate().clone(),
            artifact: self.compiler_artifact().clone(),
        };
        validate_glmm(&payload)?;
        encode_snapshot(&payload)
    }

    /// Restore a GLMM at the recorded working response and weights, without
    /// PIRLS or covariance-parameter optimization.
    pub fn restore_json(json: &str) -> Result<Self> {
        let snapshot: GlmmSnapshot = decode_snapshot(json)?;
        validate_glmm(&snapshot)?;
        let lmm = LinearMixedModel::restore_json_glmm(&snapshot.lmm_json)?;
        if snapshot.theta != lmm.theta()
            || snapshot.beta.len() != lmm.feterm.rank
            || snapshot.y.len() != lmm.dims.n
            || snapshot.y.as_slice() != lmm.y.as_slice()
        {
            return Err(MixedModelError::InvalidArgument(
                "GLMM snapshot parameters or observed response contradict its restored LMM basis"
                    .to_string(),
            ));
        }
        let mut u = Vec::with_capacity(snapshot.u.len());
        if snapshot.u.len() != lmm.reterms.len() {
            return Err(MixedModelError::InvalidArgument(
                "GLMM random-effect count contradicts LMM basis".to_string(),
            ));
        }
        for (matrix, term) in snapshot.u.iter().zip(&lmm.reterms) {
            if matrix.rows != term.vsize
                || matrix.cols != term.n_levels()
                || matrix.values.len() != matrix.rows * matrix.cols
            {
                return Err(MixedModelError::InvalidArgument(
                    "GLMM random-effect shape contradicts LMM basis".to_string(),
                ));
            }
            u.push(DMatrix::from_column_slice(
                matrix.rows,
                matrix.cols,
                &matrix.values,
            ));
        }
        let mut model = GeneralizedLinearMixedModel {
            fitted_state_seal: None,
            lmm,
            beta: DVector::from_vec(snapshot.beta.clone()),
            beta0: DVector::from_vec(snapshot.beta),
            theta: snapshot.theta,
            b: u.iter()
                .map(|m| DMatrix::zeros(m.nrows(), m.ncols()))
                .collect(),
            u: u.clone(),
            u0: u,
            eta: DVector::zeros(snapshot.y.len()),
            mu: DVector::zeros(snapshot.y.len()),
            y: DVector::from_vec(snapshot.y),
            offset: DVector::from_vec(snapshot.offset),
            wt: snapshot.wt,
            dispersion: snapshot.dispersion,
            negative_binomial_theta: snapshot.negative_binomial_theta,
            negative_binomial_estimate_theta: snapshot.negative_binomial_estimate_theta,
            family: snapshot.family,
            link: snapshot.link,
            devc: vec![],
            devc0: vec![],
            sd: vec![],
            mult: vec![],
            pirls_profiled_optimum_certificate: snapshot.pirls_certificate,
            pirls_certificate_pending: false,
            pirls_certificate_diagnostic_slot: 0,
            joint_inference_pending: false,
            inspection: std::sync::OnceLock::new(),
            warm_refit_step: None,
            pending_progress_error: None,
            response_log_constants: None,
        };
        model.lmm.compiler_artifact = snapshot.artifact;
        validate_glmm_metadata(&model)?;
        if !same_scalar(model.lmm.optsum.fmin, snapshot.objective_witness) {
            return Err(MixedModelError::InvalidArgument(
                "GLMM snapshot objective witness contradicts optimizer fmin".to_string(),
            ));
        }
        model.update_eta();
        if !same_vector(model.eta.as_slice(), &snapshot.eta)
            || !same_vector(model.mu.as_slice(), &snapshot.mu)
            || !same_scalar(fixed_state_deviance(&model)?, snapshot.objective_witness)
        {
            return Err(MixedModelError::InvalidArgument(
                "GLMM snapshot derived-state witness does not match restored parameters"
                    .to_string(),
            ));
        }
        model.seal_fitted_state();
        Ok(model)
    }
}

fn matrix_snapshot(matrix: &DMatrix<f64>) -> MatrixSnapshot {
    MatrixSnapshot {
        rows: matrix.nrows(),
        cols: matrix.ncols(),
        values: matrix.as_slice().to_vec(),
    }
}
fn seal_matches(model: &GeneralizedLinearMixedModel, seal: &FittedStateSeal) -> bool {
    seal.y == model.y.as_slice()
        && seal.wt == model.wt
        && seal.offset == model.offset.as_slice()
        && seal.beta == model.beta.as_slice()
        && seal.theta == model.theta
        && seal.u.iter().zip(&model.u).all(|(a, b)| a == b.as_slice())
        && seal.u.len() == model.u.len()
        && seal.dispersion.to_bits() == model.dispersion.to_bits()
        && seal.nb_theta.map(f64::to_bits) == model.negative_binomial_theta.map(f64::to_bits)
        && seal.family == model.family
        && seal.link == model.link
}
fn validate_glmm(snapshot: &GlmmSnapshot) -> Result<()> {
    let finite = snapshot
        .beta
        .iter()
        .chain(&snapshot.theta)
        .chain(&snapshot.y)
        .chain(&snapshot.wt)
        .chain(&snapshot.offset)
        .chain(&snapshot.eta)
        .chain(&snapshot.mu)
        .all(|v| v.is_finite())
        && snapshot.dispersion.is_finite()
        && snapshot.dispersion > 0.0
        && snapshot
            .u
            .iter()
            .all(|m| m.values.iter().all(|v| v.is_finite()));
    if !finite
        || snapshot.wt.iter().any(|v| *v < 0.0)
        || snapshot.offset.len() != snapshot.y.len()
        || (!snapshot.wt.is_empty() && snapshot.wt.len() != snapshot.y.len())
    {
        return Err(MixedModelError::InvalidArgument(
            "GLMM snapshot contains non-finite or contradictory fitted state".to_string(),
        ));
    }
    if snapshot.family == Family::NegativeBinomial
        && snapshot
            .negative_binomial_theta
            .filter(|v| *v > 0.0 && v.is_finite())
            .is_none()
    {
        return Err(MixedModelError::InvalidArgument(
            "negative-binomial snapshot is missing a positive theta".to_string(),
        ));
    }
    if snapshot.family != Family::NegativeBinomial
        && (snapshot.negative_binomial_theta.is_some() || snapshot.negative_binomial_estimate_theta)
    {
        return Err(MixedModelError::InvalidArgument(
            "non-negative-binomial snapshot carries negative-binomial state".to_string(),
        ));
    }
    super::validate_supported_glmm_family_link(snapshot.family, snapshot.link)?;
    super::validate_glmm_response_domain(snapshot.family, snapshot.link, &snapshot.y)?;
    if !snapshot.wt.is_empty() {
        super::validate_case_weights(&snapshot.wt, snapshot.y.len())?;
    }
    Ok(())
}
fn fixed_state_deviance(model: &GeneralizedLinearMixedModel) -> Result<f64> {
    let n_agq = model.lmm.optsum.n_agq;
    model.validate_agq(n_agq)?;
    let mut probe = model.clone();
    let value = if super::glmm_objective_includes_response_constants(&probe.lmm.optsum.return_value)
    {
        probe.deviance_with_response_constants(n_agq)
    } else {
        probe.deviance(n_agq)
    };
    if !value.is_finite() {
        return Err(MixedModelError::InvalidArgument(
            "GLMM snapshot fixed-state deviance is non-finite".to_string(),
        ));
    }
    Ok(value)
}
fn validate_glmm_metadata(model: &GeneralizedLinearMixedModel) -> Result<()> {
    if model.lmm.compiler_artifact.optimizer_certificate.is_none() {
        return Err(MixedModelError::InvalidArgument(
            "GLMM snapshot is missing its optimizer certificate".to_string(),
        ));
    }
    let boundary = &model.lmm.compiler_artifact.model_boundary;
    if boundary.model_kind != crate::compiler::ModelKind::GeneralizedLinearMixedModel
        || boundary.response_distribution != super::family_label(model.family)
        || boundary.link != super::link_label(model.link)
    {
        return Err(MixedModelError::InvalidArgument(
            "GLMM family or link contradicts its fit evidence".to_string(),
        ));
    }
    let metadata = model
        .lmm
        .compiler_artifact
        .glmm_fit_metadata
        .as_ref()
        .ok_or_else(|| {
            MixedModelError::InvalidArgument(
                "GLMM snapshot artifact lacks fit metadata".to_string(),
            )
        })?;
    let expected = crate::compiler::GlmmFitMetadata::from_opt_summary(&model.lmm.optsum);
    if metadata.estimation_method != expected.estimation_method
        || metadata.requested_method != expected.requested_method
        || metadata.effective_method != expected.effective_method
        || metadata.objective_definition != expected.objective_definition
        || metadata.response_constants != expected.response_constants
        || metadata.fallback_status != expected.fallback_status
    {
        return Err(MixedModelError::InvalidArgument(
            "GLMM estimator metadata contradicts its recorded fit".to_string(),
        ));
    }
    if metadata.n_agq != model.lmm.optsum.n_agq
        || metadata.optimizer_status != model.lmm.optsum.return_value
        || metadata.estimation_method.is_empty()
        || metadata
            .effective_method
            .as_deref()
            .is_some_and(str::is_empty)
        || metadata
            .requested_method
            .as_deref()
            .is_some_and(str::is_empty)
    {
        return Err(MixedModelError::InvalidArgument(
            "GLMM snapshot metadata contradicts its optimizer state".to_string(),
        ));
    }
    let effective = metadata
        .effective_method
        .as_deref()
        .unwrap_or(&metadata.estimation_method);
    let expected_parameter_count = match effective {
        "fast_pirls_profiled" | "fallback_fast_pirls" => model.theta.len(),
        "joint_laplace" | "joint_agq" => model.beta.len() + model.theta.len(),
        _ => {
            return Err(MixedModelError::InvalidArgument(
                "GLMM snapshot has an unknown effective estimator".to_string(),
            ))
        }
    };
    if model.lmm.optsum.final_params.len() != expected_parameter_count {
        return Err(MixedModelError::InvalidArgument(
            "GLMM optimizer parameter layout contradicts its effective estimator".to_string(),
        ));
    }
    if expected_parameter_count > model.theta.len()
        && !same_vector(
            &model.lmm.optsum.final_params[..model.beta.len()],
            model.beta.as_slice(),
        )
    {
        return Err(MixedModelError::InvalidArgument(
            "GLMM coefficients contradict their optimizer result".to_string(),
        ));
    }
    let status = model.lmm.optsum.return_value.as_str();
    let status_matches = match effective {
        "fast_pirls_profiled" => !super::glmm_objective_includes_response_constants(status),
        "fallback_fast_pirls" => status.contains("FALLBACK_FAST_PIRLS"),
        "joint_laplace" => {
            super::glmm_objective_includes_response_constants(status) && metadata.n_agq <= 1
        }
        "joint_agq" => {
            super::glmm_objective_includes_response_constants(status) && metadata.n_agq > 1
        }
        _ => false,
    };
    if !status_matches {
        return Err(MixedModelError::InvalidArgument(
            "GLMM effective estimator contradicts optimizer return status".to_string(),
        ));
    }
    match model.family {
        Family::NegativeBinomial => {
            let theta = model
                .negative_binomial_theta
                .expect("validated before metadata check");
            if metadata
                .family_parameters
                .get("negative_binomial_theta")
                .copied()
                != Some(theta)
                || metadata
                    .family_parameter_sources
                    .get("negative_binomial_theta")
                    .map(String::as_str)
                    != Some(if model.negative_binomial_estimate_theta {
                        "estimated"
                    } else {
                        "fixed"
                    })
            {
                return Err(MixedModelError::InvalidArgument(
                    "GLMM negative-binomial metadata contradicts fitted state".to_string(),
                ));
            }
        }
        _ if !metadata.family_parameters.is_empty() => {
            return Err(MixedModelError::InvalidArgument(
                "non-NB GLMM metadata contains family parameters".to_string(),
            ))
        }
        _ => {}
    }
    Ok(())
}
fn same_scalar(lhs: f64, rhs: f64) -> bool {
    lhs.is_finite()
        && rhs.is_finite()
        && (lhs - rhs).abs() <= 1e-10 * lhs.abs().max(rhs.abs()).max(1.0)
}
fn same_vector(lhs: &[f64], rhs: &[f64]) -> bool {
    lhs.len() == rhs.len() && lhs.iter().zip(rhs).all(|(&a, &b)| same_scalar(a, b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formula::parse_formula;
    use crate::model::data::DataFrame;
    use crate::model::snapshot::OptimizerEntryGuard;
    use crate::model::traits::MixedModelFit;

    fn weighted_offset_poisson() -> GeneralizedLinearMixedModel {
        weighted_offset_poisson_with_options(super::super::GlmmFitOptions::joint_laplace())
    }

    fn weighted_offset_poisson_with_options(
        options: super::super::GlmmFitOptions,
    ) -> GeneralizedLinearMixedModel {
        let mut frame = DataFrame::new();
        let mut y = Vec::new();
        let mut x = Vec::new();
        let mut group = Vec::new();
        for g in 0..4 {
            for row in 0..5 {
                x.push(row as f64 - 2.0);
                y.push((2 + g + row % 3) as f64);
                group.push(format!("g{g}"));
            }
        }
        frame.add_numeric("y", y).unwrap();
        frame.add_numeric("x", x).unwrap();
        frame.add_categorical("g", group).unwrap();
        let weights = (0..20).map(|i| 1.0 + (i % 2) as f64).collect();
        let offset = (0..20).map(|i| (i % 3) as f64 * 0.01).collect();
        let mut model = super::super::GeneralizedLinearMixedModelBuilder::new(
            parse_formula("y ~ 1 + x + (1 | g)").unwrap(),
            &frame,
            Family::Poisson,
        )
        .weights(weights)
        .offset(offset)
        .build()
        .unwrap();
        model.fit_with_glmm_options(options).unwrap();
        model
    }

    #[test]
    fn snapshot_preserves_glmm_working_optimizer_strategies_and_refits() {
        use super::super::GlmmFitOptions;
        use crate::model::linear::{
            ActiveFaceRefit, OptimizerControl, TrustBqGradientOracle, TrustBqSampleReuse,
            TrustBqStartLadder,
        };

        for options in [
            GlmmFitOptions::fast_laplace(),
            GlmmFitOptions::joint_laplace(),
        ] {
            let control = OptimizerControl::default()
                .with_trust_bq_start_ladder(TrustBqStartLadder::DiagonalFirst)
                .with_trust_bq_sample_reuse(TrustBqSampleReuse::Disabled)
                .with_trust_bq_gradient_oracle(TrustBqGradientOracle::Disabled)
                .with_active_face_refit(ActiveFaceRefit::Experimental);
            let mut original = weighted_offset_poisson_with_options(
                options.with_optimizer_control(control.clone()),
            );
            let json = original.snapshot_json().unwrap();
            let guard = OptimizerEntryGuard::expect_none();
            let mut restored = GeneralizedLinearMixedModel::restore_json(&json).unwrap();
            assert_eq!(guard.entries_since(), 0);
            for model in [&original, &restored] {
                assert_eq!(
                    model.lmm.trust_bq_start_ladder,
                    control.trust_bq_start_ladder
                );
                assert_eq!(
                    model.lmm.trust_bq_sample_reuse,
                    control.trust_bq_sample_reuse
                );
                assert_eq!(
                    model.lmm.trust_bq_gradient_oracle,
                    control.trust_bq_gradient_oracle
                );
                assert_eq!(model.lmm.active_face_refit, control.active_face_refit);
            }
            assert_eq!(
                original.lmm.optsum.caller_set_fields,
                restored.lmm.optsum.caller_set_fields
            );
            let response: Vec<_> = original
                .y
                .iter()
                .enumerate()
                .map(|(i, y)| y + (i % 2) as f64)
                .collect();
            original.refit(&response).unwrap();
            restored.refit(&response).unwrap();
            assert!(same_scalar(original.objective(), restored.objective()));
            assert!(same_vector(&original.theta, &restored.theta));
            assert!(same_vector(
                original.beta.as_slice(),
                restored.beta.as_slice()
            ));
            assert_eq!(original.lmm.optsum.feval, restored.lmm.optsum.feval);
            assert_eq!(
                original.lmm.optsum.return_value,
                restored.lmm.optsum.return_value
            );
            assert_eq!(
                original.lmm.optsum.caller_set_fields,
                restored.lmm.optsum.caller_set_fields
            );
            assert_eq!(
                restored.lmm.trust_bq_start_ladder,
                control.trust_bq_start_ladder
            );
            assert_eq!(
                restored.lmm.trust_bq_sample_reuse,
                control.trust_bq_sample_reuse
            );
            assert_eq!(
                restored.lmm.trust_bq_gradient_oracle,
                control.trust_bq_gradient_oracle
            );
            assert_eq!(restored.lmm.active_face_refit, control.active_face_refit);
        }
    }

    #[test]
    fn snapshot_restores_weighted_offset_glmm_without_optimizer() {
        let model = weighted_offset_poisson();
        let json = model.snapshot_json().unwrap();
        let guard = OptimizerEntryGuard::expect_none();
        let restored = GeneralizedLinearMixedModel::restore_json(&json).unwrap();
        assert_eq!(guard.entries_since(), 0);
        assert_eq!(model.compiler_artifact(), restored.compiler_artifact());
        assert_eq!(model.theta, restored.theta);
        assert!(same_vector(model.beta.as_slice(), restored.beta.as_slice()));
        assert!(same_vector(model.eta.as_slice(), restored.eta.as_slice()));
    }

    #[test]
    fn snapshot_refuses_public_glmm_response_mutation() {
        let mut model = weighted_offset_poisson();
        model.y[0] += 1.0;
        assert!(model.snapshot_json().is_err());
    }

    #[test]
    fn snapshot_restores_joint_agq_without_optimizer() {
        let frame = {
            let mut frame = DataFrame::new();
            frame
                .add_numeric("y", (0..20).map(|i| (2 + i % 3) as f64).collect())
                .unwrap();
            frame
                .add_numeric("x", (0..20).map(|i| i as f64 / 10.0).collect())
                .unwrap();
            frame
                .add_categorical("g", (0..20).map(|i| format!("g{}", i / 5)).collect())
                .unwrap();
            frame
        };
        let mut model = super::super::GeneralizedLinearMixedModelBuilder::new(
            parse_formula("y ~ 1 + x + (1 | g)").unwrap(),
            &frame,
            Family::Poisson,
        )
        .build()
        .unwrap();
        model.fit_with_options(false, 7, false).unwrap();
        assert!(model.lmm.optsum.return_value.starts_with("JOINT_AGQ:"));
        let expected_artifact = model.compiler_artifact().clone();
        let expected_vcov = model.vcov();
        let json = model.snapshot_json().unwrap();
        let guard = OptimizerEntryGuard::expect_none();
        let restored = GeneralizedLinearMixedModel::restore_json(&json).unwrap();
        assert_eq!(guard.entries_since(), 0);
        assert_eq!(restored.compiler_artifact(), &expected_artifact);
        assert!(same_vector(
            restored.vcov().as_slice(),
            expected_vcov.as_slice()
        ));
    }

    #[test]
    fn snapshot_restores_fixed_negative_binomial_state_without_optimizer() {
        let mut frame = DataFrame::new();
        frame
            .add_numeric("y", (0..20).map(|i| (1 + i % 5) as f64).collect())
            .unwrap();
        frame
            .add_numeric("x", (0..20).map(|i| i as f64 / 10.0).collect())
            .unwrap();
        frame
            .add_categorical("g", (0..20).map(|i| format!("g{}", i / 5)).collect())
            .unwrap();
        let mut model = super::super::GeneralizedLinearMixedModelBuilder::new(
            parse_formula("y ~ 1 + x + (1 | g)").unwrap(),
            &frame,
            Family::NegativeBinomial,
        )
        .negative_binomial_theta(2.0)
        .build()
        .unwrap();
        model.fit_with_options(false, 1, false).unwrap();
        #[cfg(feature = "nlopt")]
        assert!(model
            .lmm
            .optsum
            .return_value
            .contains("FALLBACK_FAST_PIRLS"));
        let expected_artifact = model.compiler_artifact().clone();
        let expected_certificate = model.pirls_profiled_optimum_certificate().clone();
        let expected_vcov = model.vcov();
        let json = model.snapshot_json().unwrap();
        let guard = OptimizerEntryGuard::expect_none();
        let restored = GeneralizedLinearMixedModel::restore_json(&json).unwrap();
        assert_eq!(guard.entries_since(), 0);
        assert_eq!(restored.compiler_artifact(), &expected_artifact);
        assert_eq!(
            restored.pirls_profiled_optimum_certificate(),
            &expected_certificate
        );
        assert!(same_vector(
            restored.vcov().as_slice(),
            expected_vcov.as_slice()
        ));
        assert_eq!(restored.negative_binomial_theta(), Some(2.0));
        assert!(!restored.negative_binomial_theta_estimated());
    }

    #[test]
    fn snapshot_restores_estimated_negative_binomial_state_without_optimizer() {
        let mut frame = DataFrame::new();
        frame
            .add_numeric("y", (0..24).map(|i| (1 + i % 6) as f64).collect())
            .unwrap();
        frame
            .add_numeric("x", (0..24).map(|i| i as f64 / 12.0).collect())
            .unwrap();
        frame
            .add_categorical("g", (0..24).map(|i| format!("g{}", i / 6)).collect())
            .unwrap();
        let mut model = super::super::GeneralizedLinearMixedModelBuilder::new(
            parse_formula("y ~ 1 + x + (1 | g)").unwrap(),
            &frame,
            Family::NegativeBinomial,
        )
        .estimate_negative_binomial_theta(Some(1.5))
        .build()
        .unwrap();
        model.fit_with_options(false, 1, false).unwrap();
        let expected_artifact = model.compiler_artifact().clone();
        let expected_certificate = model.pirls_profiled_optimum_certificate().clone();
        let expected_vcov = model.vcov();
        let expected_fitted = model.fitted();
        let expected_objective = model.objective();
        let json = model.snapshot_json().unwrap();
        let guard = OptimizerEntryGuard::expect_none();
        let mut restored = GeneralizedLinearMixedModel::restore_json(&json).unwrap();
        assert_eq!(guard.entries_since(), 0);
        assert_eq!(restored.compiler_artifact(), &expected_artifact);
        assert_eq!(
            restored.pirls_profiled_optimum_certificate(),
            &expected_certificate
        );
        assert!(same_vector(
            restored.vcov().as_slice(),
            expected_vcov.as_slice()
        ));
        assert!(same_vector(
            restored.fitted().as_slice(),
            expected_fitted.as_slice()
        ));
        assert!(same_scalar(restored.objective(), expected_objective));
        let effective = restored
            .compiler_artifact()
            .glmm_fit_metadata
            .as_ref()
            .unwrap()
            .effective_method
            .clone();
        let response = restored.y.as_slice().to_vec();
        restored.refit(&response).unwrap();
        assert_eq!(
            restored
                .compiler_artifact()
                .glmm_fit_metadata
                .as_ref()
                .unwrap()
                .effective_method,
            effective
        );
    }

    #[test]
    fn restore_rejects_typed_glmm_parameter_and_metadata_corruption() {
        let model = weighted_offset_poisson();
        let mut snapshot: GlmmSnapshot =
            crate::model::snapshot::decode_snapshot(&model.snapshot_json().unwrap()).unwrap();
        snapshot.theta[0] += 0.1;
        let json = crate::model::snapshot::encode_snapshot(&snapshot).unwrap();
        assert!(GeneralizedLinearMixedModel::restore_json(&json).is_err());

        let mut snapshot: GlmmSnapshot =
            crate::model::snapshot::decode_snapshot(&model.snapshot_json().unwrap()).unwrap();
        snapshot.artifact.glmm_fit_metadata.as_mut().unwrap().n_agq += 1;
        let json = crate::model::snapshot::encode_snapshot(&snapshot).unwrap();
        assert!(GeneralizedLinearMixedModel::restore_json(&json).is_err());
    }

    #[test]
    fn glmm_working_state_cannot_be_restored_as_a_standalone_lmm() {
        let model = weighted_offset_poisson();
        assert!(model.lmm.snapshot_json().is_err());
        let payload: GlmmSnapshot = decode_snapshot(&model.snapshot_json().unwrap()).unwrap();
        assert!(LinearMixedModel::restore_json(&payload.lmm_json).is_err());
    }

    #[test]
    fn snapshot_preserves_fast_pirls_and_gamma_dispersion_queries() {
        for family in [Family::Poisson, Family::Gamma] {
            let mut frame = DataFrame::new();
            let mut y = Vec::new();
            let mut x = Vec::new();
            let mut group = Vec::new();
            for (g, effect) in [-0.25, 0.1, 0.3, -0.15].into_iter().enumerate() {
                for observation in 0..5 {
                    let value = observation as f64 - 2.0;
                    let mean = (1.2 + 0.25 * value + effect).exp();
                    x.push(value);
                    y.push(if family == Family::Gamma {
                        mean * (1.0 + 0.06 * ((g + observation) % 3) as f64)
                    } else {
                        mean.round()
                    });
                    group.push(format!("g{g}"));
                }
            }
            frame.add_numeric("y", y).unwrap();
            frame.add_numeric("x", x).unwrap();
            frame.add_categorical("g", group).unwrap();
            let mut model = GeneralizedLinearMixedModel::new(
                parse_formula("y ~ x + (1 | g)").unwrap(),
                &frame,
                family,
                Some(LinkFunction::Log),
            )
            .unwrap();
            model
                .fit_with_options(family == Family::Poisson, 1, false)
                .unwrap();
            if family == Family::Gamma {
                assert_ne!(model.dispersion, 1.0);
            }
            let artifact = model.compiler_artifact().clone();
            let certificate = model.pirls_profiled_optimum_certificate().clone();
            let scale = super::super::GlmmPredictionScale::Response;
            let levels = crate::model::linear::NewReLevels::Error;
            let expected_prediction = model.predict_new(&frame, scale, levels).unwrap();
            let expected_uncertainty = model.predict_new_variance(&frame, scale, levels).unwrap();
            let sigma = model.dispersion;
            let json = model.snapshot_json().unwrap();
            drop(model);
            let guard = OptimizerEntryGuard::expect_none();
            let restored = GeneralizedLinearMixedModel::restore_json(&json).unwrap();
            assert_eq!(&artifact, restored.compiler_artifact());
            assert_eq!(&certificate, restored.pirls_profiled_optimum_certificate());
            assert_eq!(restored.dispersion, sigma);
            let prediction = restored.predict_new(&frame, scale, levels).unwrap();
            assert_eq!(prediction.len(), expected_prediction.len());
            for (actual, expected) in prediction.iter().zip(&expected_prediction) {
                assert!(same_scalar(
                    actual.expect("training prediction"),
                    expected.expect("training prediction")
                ));
            }
            let uncertainty = restored
                .predict_new_variance(&frame, scale, levels)
                .unwrap();
            assert_eq!(uncertainty.rows.len(), expected_uncertainty.rows.len());
            for (actual, expected) in uncertainty.rows.iter().zip(&expected_uncertainty.rows) {
                assert_eq!(actual.status, expected.status);
                match (actual.se_fit, expected.se_fit) {
                    (Some(a), Some(b)) => assert!(same_scalar(a, b)),
                    (a, b) => assert_eq!(a, b),
                }
            }
            assert_eq!(
                guard.entries_since(),
                0,
                "restore and ordinary queries must not optimize"
            );
        }
    }
}
