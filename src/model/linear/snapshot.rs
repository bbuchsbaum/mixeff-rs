use serde::{Deserialize, Serialize};

use crate::error::{MixedModelError, Result};
use crate::model::snapshot::{decode_snapshot, encode_snapshot, TrainingRecipe};
use crate::model::summary_estimates::ResidualSource;
use crate::model::traits::MixedModelFit;
use crate::types::OptSummary;

use super::{
    ActiveFaceRefit, LinearMixedModel, TrustBqGradientOracle, TrustBqSampleReuse,
    TrustBqStartLadder,
};

// Keep these encodings private to the opaque, source-bound snapshot format.
// Remote derives validate the enum variants without adding public serde APIs.
#[derive(Serialize, Deserialize)]
#[serde(remote = "TrustBqStartLadder", rename_all = "snake_case")]
enum SnapshotStartLadder {
    Off,
    DiagonalFirst,
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "TrustBqSampleReuse", rename_all = "snake_case")]
enum SnapshotSampleReuse {
    FamilyPolicy,
    Disabled,
    AllFamilies,
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "TrustBqGradientOracle", rename_all = "snake_case")]
enum SnapshotGradientOracle {
    FamilyPolicy,
    Disabled,
    AllFamilies,
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "ActiveFaceRefit", rename_all = "snake_case")]
enum SnapshotActiveFaceRefit {
    Off,
    Experimental,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LinearSnapshot {
    kind: LinearSnapshotKind,
    recipe: TrainingRecipe,
    response: Vec<f64>,
    working_response: Vec<f64>,
    theta: Vec<f64>,
    working_sqrtwts: Vec<f64>,
    optsum: OptSummary,
    #[serde(with = "SnapshotStartLadder")]
    trust_bq_start_ladder: TrustBqStartLadder,
    #[serde(with = "SnapshotSampleReuse")]
    trust_bq_sample_reuse: TrustBqSampleReuse,
    #[serde(with = "SnapshotGradientOracle")]
    trust_bq_gradient_oracle: TrustBqGradientOracle,
    #[serde(with = "SnapshotActiveFaceRefit")]
    active_face_refit: ActiveFaceRefit,
    artifact: crate::compiler::CompiledModelArtifact,
    residual_source: ResidualSource,
    beta_witness: Vec<f64>,
    objective_witness: f64,
    basis: BasisWitness,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum LinearSnapshotKind {
    StandaloneLmm,
    GlmmWorkingState,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct BasisWitness {
    effective_formula: String,
    fixed_rows: usize,
    fixed_cols: usize,
    fixed: Vec<f64>,
    piv: Vec<usize>,
    rank: usize,
    names: Vec<String>,
    parmap: Vec<(usize, usize, usize)>,
    random: Vec<RandomBasisWitness>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct RandomBasisWitness {
    group: String,
    levels: Vec<String>,
    refs: Vec<u32>,
    names: Vec<String>,
    rows: usize,
    cols: usize,
    z: Vec<f64>,
}

impl LinearMixedModel {
    /// Serialize a completed LMM as an opaque snapshot for a matching engine
    /// source/build configuration.
    /// It stores construction inputs, current response, and fitted state; it
    /// is not a cross-version interchange format.
    pub fn snapshot_json(&self) -> Result<String> {
        self.snapshot_json_for(LinearSnapshotKind::StandaloneLmm)
    }

    pub(crate) fn snapshot_json_glmm(&self) -> Result<String> {
        self.snapshot_json_for(LinearSnapshotKind::GlmmWorkingState)
    }

    fn snapshot_json_for(&self, kind: LinearSnapshotKind) -> Result<String> {
        if !self.is_fitted() {
            return Err(MixedModelError::InvalidArgument(
                "cannot snapshot an unfitted model".to_string(),
            ));
        }
        let optsum = self.optsum.clone();
        validate_optsum_json(&optsum)?;
        let response = self.y.as_slice().to_vec();
        let theta = self.theta();
        if response.iter().any(|v| !v.is_finite())
            || theta.iter().any(|v| !v.is_finite())
            || self.sqrtwts.iter().any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(MixedModelError::InvalidArgument(
                "cannot snapshot non-finite fitted state".to_string(),
            ));
        }
        let payload = LinearSnapshot {
            kind,
            recipe: (*self.training_recipe).clone(),
            response,
            working_response: self.xy_mat.xy.column(self.feterm.rank).as_slice().to_vec(),
            theta,
            working_sqrtwts: self.sqrtwts.clone(),
            optsum,
            trust_bq_start_ladder: self.trust_bq_start_ladder,
            trust_bq_sample_reuse: self.trust_bq_sample_reuse,
            trust_bq_gradient_oracle: self.trust_bq_gradient_oracle,
            active_face_refit: self.active_face_refit,
            artifact: self.compiler_artifact().clone(),
            residual_source: self.residual_source,
            beta_witness: self.beta().as_slice().to_vec(),
            objective_witness: self.objective(),
            basis: basis_witness(self),
        };
        validate_artifact(&payload)?;
        validate_fitted_state(&payload)?;
        encode_snapshot(&payload)
    }

    /// Restore a snapshot by rebuilding the basis then installing fixed state.
    /// No optimizer search is entered.
    pub fn restore_json(json: &str) -> Result<Self> {
        Self::restore_json_for(json, LinearSnapshotKind::StandaloneLmm)
    }

    pub(crate) fn restore_json_glmm(json: &str) -> Result<Self> {
        Self::restore_json_for(json, LinearSnapshotKind::GlmmWorkingState)
    }

    fn restore_json_for(json: &str, kind: LinearSnapshotKind) -> Result<Self> {
        let snapshot: LinearSnapshot = decode_snapshot(json)?;
        if snapshot.kind != kind {
            return Err(MixedModelError::InvalidArgument(
                "a GLMM working state is not a standalone fitted LMM".to_string(),
            ));
        }
        snapshot.recipe.validate()?;
        validate_optsum_json(&snapshot.optsum)?;
        validate_artifact(&snapshot)?;
        validate_fitted_state(&snapshot)?;
        let frame = crate::model::data::DataFrame::from_snapshot_frame(&snapshot.recipe.frame)?;
        let mut restored = LinearMixedModel::new_with_policy_internal(
            snapshot.recipe.formula.clone(),
            &frame,
            snapshot.recipe.weights.as_deref(),
            snapshot.recipe.compiler_policy.clone(),
        )?;
        if basis_witness(&restored) != snapshot.basis {
            return Err(MixedModelError::InvalidArgument("snapshot design basis, pivot, random levels, or parameter map does not match its construction recipe".to_string()));
        }
        restored.restore_fixed_state(
            &snapshot.response,
            &snapshot.working_response,
            &snapshot.theta,
            &snapshot.working_sqrtwts,
            snapshot.optsum,
            snapshot.artifact,
            snapshot.residual_source,
        )?;
        restored.trust_bq_start_ladder = snapshot.trust_bq_start_ladder;
        restored.trust_bq_sample_reuse = snapshot.trust_bq_sample_reuse;
        restored.trust_bq_gradient_oracle = snapshot.trust_bq_gradient_oracle;
        restored.active_face_refit = snapshot.active_face_refit;
        if !same_finite_vector(restored.beta().as_slice(), &snapshot.beta_witness, 1e-10)
            || !same_finite(restored.objective(), snapshot.objective_witness, 1e-10)
        {
            return Err(MixedModelError::InvalidArgument(
                "snapshot fitted-state witness does not match reconstructed basis".to_string(),
            ));
        }
        Ok(restored)
    }
}

fn validate_optsum_json(summary: &OptSummary) -> Result<()> {
    let finite = [
        summary.fmin,
        summary.xtol_zero_abs,
        summary.ftol_zero_abs,
        summary.ftol_rel,
        summary.ftol_abs,
        summary.xtol_rel,
        summary.max_time,
        summary.rhobeg,
        summary.rhoend,
    ]
    .into_iter()
    .all(f64::is_finite)
        && summary.final_trust_radius.is_none_or(f64::is_finite)
        && summary
            .initial
            .iter()
            .chain(&summary.final_params)
            .chain(&summary.xtol_abs)
            .chain(&summary.initial_step)
            .all(|v| v.is_finite())
        && summary
            .sigma
            .is_none_or(|value| value.is_finite() && value > 0.0)
        && summary
            .fit_log
            .iter()
            .all(|entry| entry.theta.iter().all(|v| v.is_finite()));
    if !finite {
        return Err(MixedModelError::Unsupported(
            "snapshot requires finite optimizer parameters, controls, and final objective"
                .to_string(),
        ));
    }
    Ok(())
}

fn same_finite(lhs: f64, rhs: f64, tolerance: f64) -> bool {
    lhs.is_finite()
        && rhs.is_finite()
        && (lhs - rhs).abs() <= tolerance * lhs.abs().max(rhs.abs()).max(1.0)
}
fn same_finite_vector(lhs: &[f64], rhs: &[f64], tolerance: f64) -> bool {
    lhs.len() == rhs.len()
        && lhs
            .iter()
            .zip(rhs)
            .all(|(&a, &b)| same_finite(a, b, tolerance))
}

fn basis_witness(model: &LinearMixedModel) -> BasisWitness {
    BasisWitness {
        effective_formula: model.formula.to_string(),
        fixed_rows: model.feterm.x.nrows(),
        fixed_cols: model.feterm.x.ncols(),
        fixed: model.feterm.x.as_slice().to_vec(),
        piv: model.feterm.piv.clone(),
        rank: model.feterm.rank,
        names: model.feterm.cnames.clone(),
        parmap: model.parmap.clone(),
        random: model
            .reterms
            .iter()
            .map(|term| RandomBasisWitness {
                group: term.grouping_name.clone(),
                levels: term.levels.clone(),
                refs: term.refs.clone(),
                names: term.cnames.clone(),
                rows: term.z.nrows(),
                cols: term.z.ncols(),
                z: term.z.as_slice().to_vec(),
            })
            .collect(),
    }
}

fn validate_artifact(snapshot: &LinearSnapshot) -> Result<()> {
    if snapshot.artifact.requested_formula != snapshot.recipe.formula.to_string() {
        return Err(MixedModelError::InvalidArgument(
            "snapshot compiler artifact contradicts its original formula".to_string(),
        ));
    }
    let effective = snapshot
        .artifact
        .effective_formula
        .as_deref()
        .unwrap_or(&snapshot.artifact.requested_formula);
    if effective != snapshot.basis.effective_formula {
        return Err(MixedModelError::InvalidArgument(
            "snapshot compiler artifact contradicts its effective design basis".to_string(),
        ));
    }
    if snapshot.residual_source == ResidualSource::FixedSamplingVariance
        && snapshot.optsum.sigma != Some(1.0)
    {
        return Err(MixedModelError::InvalidArgument(
            "fixed sampling-variance snapshot must record sigma = 1".to_string(),
        ));
    }
    Ok(())
}

fn validate_fitted_state(snapshot: &LinearSnapshot) -> Result<()> {
    if snapshot.optsum.feval <= 0 {
        return Err(MixedModelError::InvalidArgument(
            "snapshot optimizer summary is not fitted".to_string(),
        ));
    }
    let glmm_inner = snapshot.kind == LinearSnapshotKind::GlmmWorkingState;
    let expected_kind = if glmm_inner {
        crate::compiler::ModelKind::GeneralizedLinearMixedModel
    } else {
        crate::compiler::ModelKind::LinearMixedModel
    };
    if snapshot.artifact.model_boundary.model_kind != expected_kind
        || snapshot.artifact.glmm_fit_metadata.is_some() != glmm_inner
    {
        return Err(MixedModelError::InvalidArgument(
            "snapshot model kind contradicts its fit evidence".to_string(),
        ));
    }
    if !glmm_inner {
        if snapshot.working_response != snapshot.response {
            return Err(MixedModelError::InvalidArgument(
                "ordinary LMM snapshot has a mutated working response".to_string(),
            ));
        }
        let expected_weights = snapshot
            .recipe
            .weights
            .as_ref()
            .map(|weights| {
                weights
                    .iter()
                    .map(|weight| weight.sqrt())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if snapshot.working_sqrtwts != expected_weights {
            return Err(MixedModelError::InvalidArgument(
                "ordinary LMM snapshot working weights differ from construction weights"
                    .to_string(),
            ));
        }
    }
    if glmm_inner {
        if snapshot.optsum.final_params.len() < snapshot.theta.len()
            || !snapshot.optsum.final_params.ends_with(&snapshot.theta)
        {
            return Err(MixedModelError::InvalidArgument(
                "GLMM inner snapshot parameters do not end in the recorded theta".to_string(),
            ));
        }
    } else {
        if snapshot.optsum.final_params != snapshot.theta {
            return Err(MixedModelError::InvalidArgument(
                "LMM snapshot theta contradicts its optimizer summary".to_string(),
            ));
        }
        if !same_finite(snapshot.optsum.fmin, snapshot.objective_witness, 1e-10) {
            return Err(MixedModelError::InvalidArgument(
                "LMM snapshot objective contradicts optimizer fmin".to_string(),
            ));
        }
    }
    {
        let certificate = snapshot
            .artifact
            .optimizer_certificate
            .as_ref()
            .ok_or_else(|| {
                MixedModelError::InvalidArgument(
                    "fitted snapshot is missing its optimizer certificate".to_string(),
                )
            })?;
        let labelled_fallback =
            glmm_inner && snapshot.optsum.return_value.contains("FALLBACK_FAST_PIRLS");
        if (!labelled_fallback
            && (certificate.evidence.optimizer_stop.return_code.as_deref()
                != Some(&snapshot.optsum.return_value)
                || certificate.evidence.optimizer_stop.function_evaluations
                    != usize::try_from(snapshot.optsum.feval).ok()))
            || certificate
                .objective_value
                .is_some_and(|value| !same_finite(value, snapshot.optsum.fmin, 1e-10))
        {
            return Err(MixedModelError::InvalidArgument(
                "snapshot optimizer certificate contradicts its optimizer summary".to_string(),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formula::parse_formula;
    use crate::model::data::DataFrame;
    use crate::model::linear::{
        ActiveFaceRefit, FitOptions, OptimizerControl, TrustBqGradientOracle, TrustBqSampleReuse,
        TrustBqStartLadder,
    };
    use crate::model::snapshot::OptimizerEntryGuard;

    #[test]
    fn review_snapshot_preserves_optimizer_strategy_controls() {
        let (data, meta) = crate::datasets::load("dyestuff").unwrap();
        let mut model =
            LinearMixedModel::new(parse_formula(&meta.fits[0].formula).unwrap(), &data, None)
                .unwrap();
        let control = OptimizerControl::default()
            .with_trust_bq_start_ladder(TrustBqStartLadder::DiagonalFirst)
            .with_trust_bq_sample_reuse(TrustBqSampleReuse::Disabled)
            .with_trust_bq_gradient_oracle(TrustBqGradientOracle::Disabled)
            .with_active_face_refit(ActiveFaceRefit::Experimental);
        model
            .fit_with_options(FitOptions::ml().with_optimizer_control(control))
            .unwrap();
        let expected = (
            model.trust_bq_start_ladder,
            model.trust_bq_sample_reuse,
            model.trust_bq_gradient_oracle,
            model.active_face_refit,
        );
        let original_audit = model.optsum.caller_set_fields.clone();
        let guard = OptimizerEntryGuard::expect_none();
        let restored = LinearMixedModel::restore_json(&model.snapshot_json().unwrap()).unwrap();
        assert_eq!(guard.entries_since(), 0);
        assert_eq!(restored.optsum.caller_set_fields, original_audit);
        let actual = (
            restored.trust_bq_start_ladder,
            restored.trust_bq_sample_reuse,
            restored.trust_bq_gradient_oracle,
            restored.active_face_refit,
        );
        assert_eq!(
            actual, expected,
            "restored strategies must match the caller settings retained in the audit"
        );
    }

    #[test]
    fn snapshot_strategy_controls_survive_native_refits() {
        let data = varied_data();
        for control in [
            OptimizerControl::default(),
            OptimizerControl::default()
                .with_trust_bq_start_ladder(TrustBqStartLadder::DiagonalFirst)
                .with_trust_bq_sample_reuse(TrustBqSampleReuse::Disabled)
                .with_trust_bq_gradient_oracle(TrustBqGradientOracle::Disabled)
                .with_active_face_refit(ActiveFaceRefit::Experimental),
            OptimizerControl::default()
                .with_trust_bq_sample_reuse(TrustBqSampleReuse::AllFamilies)
                .with_trust_bq_gradient_oracle(TrustBqGradientOracle::AllFamilies),
        ] {
            let mut original =
                LinearMixedModel::new(parse_formula("y ~ x + (1 + x | g)").unwrap(), &data, None)
                    .unwrap();
            original
                .fit_with_options(
                    FitOptions::ml().with_optimizer_control(
                        control
                            .clone()
                            .with_optimizer(crate::types::Optimizer::TrustBq),
                    ),
                )
                .unwrap();
            let original_audit = original.optsum.caller_set_fields.clone();
            let json = original.snapshot_json().unwrap();
            let guard = OptimizerEntryGuard::expect_none();
            let mut restored = LinearMixedModel::restore_json(&json).unwrap();
            assert_eq!(guard.entries_since(), 0);
            assert_eq!(
                restored.trust_bq_start_ladder,
                control.trust_bq_start_ladder
            );
            assert_eq!(
                restored.trust_bq_sample_reuse,
                control.trust_bq_sample_reuse
            );
            assert_eq!(
                restored.trust_bq_gradient_oracle,
                control.trust_bq_gradient_oracle
            );
            assert_eq!(restored.active_face_refit, control.active_face_refit);
            let response: Vec<_> = original
                .y
                .iter()
                .enumerate()
                .map(|(i, y)| y + 0.1 * (i as f64).cos())
                .collect();
            original.refit(&response).unwrap();
            restored.refit(&response).unwrap();
            assert!(same_finite(
                original.objective(),
                restored.objective(),
                1e-10
            ));
            assert!(same_finite_vector(
                &original.theta(),
                &restored.theta(),
                1e-10
            ));
            assert_eq!(original.optsum.feval, restored.optsum.feval);
            assert_eq!(original.optsum.return_value, restored.optsum.return_value);
            assert_eq!(original.optsum.caller_set_fields, original_audit);
            assert_eq!(restored.optsum.caller_set_fields, original_audit);
            assert_eq!(
                restored.optsum.caller_selected_optimizer(),
                Some(crate::types::Optimizer::TrustBq)
            );
            assert_eq!(
                restored.trust_bq_start_ladder,
                control.trust_bq_start_ladder
            );
            assert_eq!(
                restored.trust_bq_sample_reuse,
                control.trust_bq_sample_reuse
            );
            assert_eq!(
                restored.trust_bq_gradient_oracle,
                control.trust_bq_gradient_oracle
            );
            assert_eq!(restored.active_face_refit, control.active_face_refit);
        }
    }

    #[test]
    fn snapshot_requires_known_optimizer_strategy_controls() {
        let model = transformed_weighted_model();
        let payload: LinearSnapshot = decode_snapshot(&model.snapshot_json().unwrap()).unwrap();
        let value = serde_json::to_value(payload).unwrap();
        for field in [
            "trust_bq_start_ladder",
            "trust_bq_sample_reuse",
            "trust_bq_gradient_oracle",
            "active_face_refit",
        ] {
            let mut invalid = value.clone();
            invalid[field] = "unknown_strategy".into();
            assert!(serde_json::from_value::<LinearSnapshot>(invalid)
                .unwrap_err()
                .to_string()
                .contains("unknown variant"));
            let mut missing = value.clone();
            missing.as_object_mut().unwrap().remove(field);
            assert!(serde_json::from_value::<LinearSnapshot>(missing)
                .unwrap_err()
                .to_string()
                .contains(&format!("missing field `{field}`")));
        }
    }

    fn query_values(model: &LinearMixedModel, data: &DataFrame) -> serde_json::Value {
        serde_json::json!({
            "beta": model.beta(), "theta": model.theta(), "objective": model.objective(),
            "vcov": model.vcov(), "cond_var": model.cond_var(),
            "prediction": model.predict_new(data, super::super::NewReLevels::Error).unwrap(),
            "uncertainty": model.predict_new_variance(data, super::super::NewReLevels::Error).unwrap(),
            "contrasts": model.fixed_effect_contrast_inference_table(model.coefficient_hypotheses(), crate::model::linear::FixedEffectTestMethod::Satterthwaite),
        })
    }

    fn assert_values_close(actual: &serde_json::Value, expected: &serde_json::Value) {
        match (actual, expected) {
            (serde_json::Value::Number(a), serde_json::Value::Number(b)) => {
                assert!(
                    same_finite(a.as_f64().unwrap(), b.as_f64().unwrap(), 1e-9),
                    "{a} != {b}"
                );
            }
            (serde_json::Value::Array(a), serde_json::Value::Array(b)) => {
                assert_eq!(a.len(), b.len());
                for (a, b) in a.iter().zip(b) {
                    assert_values_close(a, b);
                }
            }
            (serde_json::Value::Object(a), serde_json::Value::Object(b)) => {
                assert_eq!(a.len(), b.len());
                for (key, b) in b {
                    assert_values_close(a.get(key).unwrap(), b);
                }
            }
            _ => assert_eq!(actual, expected),
        }
    }

    fn assert_queries_survive(model: LinearMixedModel, data: &DataFrame) {
        let expected = query_values(&model, data);
        let artifact = model.compiler_artifact().clone();
        let json = model.snapshot_json().unwrap();
        drop(model);
        let guard = OptimizerEntryGuard::expect_none();
        let restored = LinearMixedModel::restore_json(&json).unwrap();
        assert_eq!(&artifact, restored.compiler_artifact());
        assert_values_close(&query_values(&restored, data), &expected);
        assert_eq!(
            guard.entries_since(),
            0,
            "restoration and ordinary queries must not optimize"
        );
    }

    #[test]
    fn snapshot_preserves_boundary_fit_and_conditional_queries() {
        let (data, meta) = crate::datasets::load("dyestuff2").unwrap();
        let mut model =
            LinearMixedModel::new(parse_formula(&meta.fits[0].formula).unwrap(), &data, None)
                .unwrap();
        model.fit(false).unwrap();
        assert!(model.theta().contains(&0.0));
        assert_queries_survive(model, &data);
    }

    fn varied_data() -> DataFrame {
        let mut data = DataFrame::new();
        data.add_numeric(
            "y",
            (0..48)
                .map(|i| 2.0 + 0.3 * (i / 6) as f64 + 0.2 * (i % 6) as f64 + (i as f64).sin())
                .collect(),
        )
        .unwrap();
        data.add_numeric("x", (0..48).map(|i| (i % 6) as f64 / 3.0).collect())
            .unwrap();
        data.add_numeric("v", (0..48).map(|i| 0.5 + (i % 3) as f64 / 10.0).collect())
            .unwrap();
        data.add_categorical("g", (0..48).map(|i| format!("g{}", i / 6)).collect())
            .unwrap();
        data
    }

    #[test]
    fn snapshot_preserves_ordered_and_unordered_custom_contrasts() {
        use crate::model::data::{CategoricalContrast, ContrastSource};
        for ordered in [false, true] {
            let mut data = varied_data();
            let levels = vec!["medium".to_string(), "low".to_string(), "high".to_string()];
            let q = 1.0 / 3.0_f64.sqrt();
            let contrast = CategoricalContrast::new(
                levels.clone(),
                nalgebra::DMatrix::from_row_slice(3, 2, &[q, 0.12345678901234568, -q, q, 0.0, -q]),
                vec!["c1".into(), "c2".into()],
                ordered,
                ContrastSource::Custom,
            )
            .unwrap();
            data.add_categorical_with_contrast(
                "condition",
                (0..48).map(|i| levels[i % 3].clone()).collect(),
                levels,
                contrast,
            )
            .unwrap();
            let mut model = LinearMixedModel::new(
                parse_formula("y ~ x + condition + (1 | g)").unwrap(),
                &data,
                None,
            )
            .unwrap();
            model.fit(true).unwrap();
            assert_queries_survive(model, &data);
        }
    }

    #[test]
    fn snapshot_replays_reduced_construction_and_final_factorization_policy() {
        let mut data = varied_data();
        data.add_numeric("between", (0..48).map(|i| (i / 6) as f64).collect())
            .unwrap();
        let mut model = LinearMixedModel::new_with_compiler_policy(
            parse_formula("y ~ x + between + (1 + between | g)").unwrap(),
            &data,
            None,
            crate::compiler::CompilerPolicy::design_compiled(),
        )
        .unwrap();
        assert!(!model.compiler_artifact().reductions.is_empty());
        let mut final_policy = crate::compiler::CompilerPolicy::as_specified();
        final_policy.thresholds.cholesky_zero_pad_tolerance = 1e-12;
        model.set_compiler_policy(final_policy).unwrap();
        model.fit(true).unwrap();
        assert_queries_survive(model, &data);
    }

    #[test]
    fn snapshot_preserves_fixed_sampling_variance_semantics() {
        let data = varied_data();
        let mut model = LinearMixedModel::from_summary_estimates(
            parse_formula("y ~ x + (1 | g)").unwrap(),
            &data,
            "y",
            "v",
            Default::default(),
        )
        .unwrap();
        model.fit(true).unwrap();
        assert_eq!(model.residual_source, ResidualSource::FixedSamplingVariance);
        assert_queries_survive(model, &data);
    }

    fn transformed_weighted_model() -> LinearMixedModel {
        let mut frame = DataFrame::new();
        frame
            .add_numeric("y", (0..18).map(|i| 2.0 + i as f64 / 3.0).collect())
            .unwrap();
        frame
            .add_numeric("x", (0..18).map(|i| i as f64 / 7.0).collect())
            .unwrap();
        frame
            .add_categorical("g", (0..18).map(|i| format!("g{}", i / 3)).collect())
            .unwrap();
        let weights = (0..18)
            .map(|i| 1.0 + (i % 3) as f64 / 4.0)
            .collect::<Vec<_>>();
        let mut model = LinearMixedModel::new_with_policy_internal(
            parse_formula("log(y) ~ 1 + x + (1 | g)").unwrap(),
            &frame,
            Some(&weights),
            Default::default(),
        )
        .unwrap();
        model.fit(false).unwrap();
        let refit = model.y.iter().map(|value| value + 0.01).collect::<Vec<_>>();
        model.refit(&refit).unwrap();
        model
    }

    #[test]
    fn snapshot_restores_weighted_transformed_refit_without_optimizer() {
        let model = transformed_weighted_model();
        let json = model.snapshot_json().unwrap();
        let guard = OptimizerEntryGuard::expect_none();
        let restored = LinearMixedModel::restore_json(&json).unwrap();
        assert_eq!(guard.entries_since(), 0);
        assert_eq!(model.compiler_artifact(), restored.compiler_artifact());
        assert_eq!(model.theta(), restored.theta());
        assert_eq!(model.response(), restored.response());
        assert!(same_finite_vector(
            model.beta().as_slice(),
            restored.beta().as_slice(),
            1e-12
        ));
    }

    #[test]
    fn ordinary_lmm_rejects_mutated_working_state() {
        let model = transformed_weighted_model();
        let mut snapshot = LinearSnapshot {
            kind: LinearSnapshotKind::StandaloneLmm,
            recipe: (*model.training_recipe).clone(),
            response: model.y.as_slice().to_vec(),
            working_response: model
                .xy_mat
                .xy
                .column(model.feterm.rank)
                .as_slice()
                .to_vec(),
            theta: model.theta(),
            working_sqrtwts: model.sqrtwts.clone(),
            optsum: model.optsum.clone(),
            trust_bq_start_ladder: model.trust_bq_start_ladder,
            trust_bq_sample_reuse: model.trust_bq_sample_reuse,
            trust_bq_gradient_oracle: model.trust_bq_gradient_oracle,
            active_face_refit: model.active_face_refit,
            artifact: model.compiler_artifact().clone(),
            residual_source: model.residual_source,
            beta_witness: model.beta().as_slice().to_vec(),
            objective_witness: model.objective(),
            basis: basis_witness(&model),
        };
        snapshot.working_response[0] += 0.5;
        assert!(validate_fitted_state(&snapshot).is_err());
        let json = crate::model::snapshot::encode_snapshot(&snapshot).unwrap();
        assert!(LinearMixedModel::restore_json(&json).is_err());
    }

    #[test]
    fn snapshot_refuses_missing_or_contradictory_fit_evidence() {
        let model = transformed_weighted_model();
        let json = model.snapshot_json().unwrap();
        for defect in 0..4 {
            let mut payload: LinearSnapshot = decode_snapshot(&json).unwrap();
            match defect {
                0 => payload.optsum.final_params.clear(),
                1 => payload.optsum.feval = 0,
                2 => payload.optsum.fmin += 1.0,
                _ => payload.artifact.optimizer_certificate = None,
            }
            let invalid = encode_snapshot(&payload).unwrap();
            assert!(
                LinearMixedModel::restore_json(&invalid).is_err(),
                "defect {defect}"
            );
        }
    }
}
