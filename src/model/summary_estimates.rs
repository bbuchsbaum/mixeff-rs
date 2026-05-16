//! Types and constructor for the summary-estimate (meta-analysis) LMM
//! front door.
//!
//! See `docs/summary_estimates_meta_analysis.md` for the full contract.

use crate::compiler::policy::CompilerPolicy;
use crate::error::{MixedModelError, Result};
use crate::formula::Formula;
use crate::model::data::DataFrame;
use crate::model::linear::LinearMixedModel;
use serde::{Deserialize, Serialize};

/// How the caller scaled the input sampling variances.
///
/// Inputs are normalized to `Absolute` internally; the fitted
/// `LinearMixedModel` always carries `optsum.sigma == Some(1.0)` for
/// summary-estimate fits.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
#[derive(Default)]
pub enum SamplingVarianceScale {
    /// `V_i` is on the absolute scale. The implied residual variance for
    /// observation `i` is exactly `V_i`. This is the typical case for
    /// published meta-analysis inputs.
    #[default]
    Absolute,
    /// `V_i` is unscaled (for example, the diagonal of `(X' X)^{-1}` from a
    /// first-stage OLS) and the caller supplies a separate first-stage
    /// residual standard deviation `sigma`. Internally normalized to
    /// `absolute_V_i = sigma * sigma * V_i`.
    Relative {
        /// First-stage residual standard deviation. Must be finite and
        /// strictly positive.
        sigma: f64,
    },
}

/// Options for [`LinearMixedModel::from_summary_estimates`].
///
/// Construct with `Default::default()` and override fields as needed.
#[derive(Debug, Clone)]
pub struct SummaryEstimateOptions {
    /// How the caller's `V_i` is scaled. Default: `Absolute`.
    pub variance_scale: SamplingVarianceScale,
    /// Compiler policy applied during model construction.
    pub policy: CompilerPolicy,
}

impl Default for SummaryEstimateOptions {
    fn default() -> Self {
        SummaryEstimateOptions {
            variance_scale: SamplingVarianceScale::Absolute,
            policy: CompilerPolicy::default(),
        }
    }
}

/// Marks how the residual scale `sigma` is determined for a fitted model.
///
/// Carried on `LinearMixedModel` and propagated to `VarCorr` so renderers can
/// decide whether to display a "Residual" row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
#[derive(Default)]
pub enum ResidualSource {
    /// `sigma` is estimated from the data (the usual LMM case) or pinned
    /// for parity testing without changing model class semantics. Default.
    #[default]
    EstimatedSigma,
    /// `sigma` is fixed at `1.0` because per-observation sampling variances
    /// were supplied through `LinearMixedModel::from_summary_estimates`.
    /// In this regime, random-effect variances are absolute (between-study
    /// `tau^2`) and `varcorr()` should not display a residual row.
    FixedSamplingVariance,
}

impl LinearMixedModel {
    /// Build a summary-estimate (meta-analysis) `LinearMixedModel` from
    /// first-stage point estimates and absolute sampling variances.
    ///
    /// `data` must contain a numeric column named `estimate_column` (the
    /// `beta_hat_i`) and a numeric column named `sampling_variance_column`
    /// (`V_i`). The `formula` LHS must reference `estimate_column`. The
    /// resulting model is unfitted; call [`LinearMixedModel::fit`] as usual.
    ///
    /// Internally the constructor sets `weights = 1 / V_i` (after any
    /// `Relative` -> `Absolute` normalization) and pins
    /// `optsum.sigma = Some(1.0)`. The model is tagged
    /// [`ResidualSource::FixedSamplingVariance`] so downstream consumers
    /// (`varcorr`, finite-sample inference) can render or refuse correctly.
    ///
    /// See `docs/summary_estimates_meta_analysis.md` for the full contract.
    pub fn from_summary_estimates(
        formula: Formula,
        data: &DataFrame,
        estimate_column: &str,
        sampling_variance_column: &str,
        options: SummaryEstimateOptions,
    ) -> Result<LinearMixedModel> {
        if formula.response != estimate_column {
            return Err(MixedModelError::InvalidArgument(format!(
                "from_summary_estimates: formula response '{}' must match estimate_column '{}'",
                formula.response, estimate_column
            )));
        }

        if estimate_column == sampling_variance_column {
            return Err(MixedModelError::InvalidArgument(format!(
                "from_summary_estimates: estimate_column and sampling_variance_column \
                 must differ; got '{estimate_column}'"
            )));
        }

        let scale_factor_sq = match options.variance_scale {
            SamplingVarianceScale::Absolute => 1.0,
            SamplingVarianceScale::Relative { sigma } => {
                if !sigma.is_finite() || sigma <= 0.0 {
                    return Err(MixedModelError::InvalidArgument(format!(
                        "from_summary_estimates: Relative variance scale requires a \
                         finite positive sigma; got {sigma}"
                    )));
                }
                let scale_factor_sq = sigma * sigma;
                if !scale_factor_sq.is_finite() || scale_factor_sq <= 0.0 {
                    return Err(MixedModelError::InvalidArgument(format!(
                        "from_summary_estimates: Relative variance scale produced a \
                         non-finite squared sigma; got sigma={sigma}"
                    )));
                }
                scale_factor_sq
            }
        };

        let beta_hat = data.numeric(estimate_column).ok_or_else(|| {
            MixedModelError::InvalidArgument(format!(
                "from_summary_estimates: estimate column '{estimate_column}' \
                 not found or not numeric"
            ))
        })?;
        let v_raw = data.numeric(sampling_variance_column).ok_or_else(|| {
            MixedModelError::InvalidArgument(format!(
                "from_summary_estimates: sampling variance column \
                 '{sampling_variance_column}' not found or not numeric"
            ))
        })?;

        if beta_hat.len() != v_raw.len() {
            return Err(MixedModelError::InvalidArgument(format!(
                "from_summary_estimates: estimate column '{}' has {} rows but \
                 sampling variance column '{}' has {} rows",
                estimate_column,
                beta_hat.len(),
                sampling_variance_column,
                v_raw.len()
            )));
        }

        let mut weights = Vec::with_capacity(v_raw.len());
        for (i, (b, v)) in beta_hat.iter().zip(v_raw.iter()).enumerate() {
            if !b.is_finite() {
                return Err(MixedModelError::InvalidArgument(format!(
                    "from_summary_estimates: estimate column '{estimate_column}' \
                     contains a non-finite value at row {i}"
                )));
            }
            if !v.is_finite() {
                return Err(MixedModelError::InvalidArgument(format!(
                    "from_summary_estimates: sampling variance column \
                     '{sampling_variance_column}' contains a non-finite value at row {i}"
                )));
            }
            if *v <= 0.0 {
                return Err(MixedModelError::InvalidArgument(format!(
                    "from_summary_estimates: sampling variance column \
                     '{sampling_variance_column}' contains a non-positive value at row {i}"
                )));
            }
            let absolute_v = scale_factor_sq * *v;
            if !absolute_v.is_finite() || absolute_v <= 0.0 {
                return Err(MixedModelError::InvalidArgument(format!(
                    "from_summary_estimates: normalized sampling variance for column \
                     '{sampling_variance_column}' is not finite and positive at row {i}"
                )));
            }
            let weight = 1.0 / absolute_v;
            if !weight.is_finite() || weight <= 0.0 {
                return Err(MixedModelError::InvalidArgument(format!(
                    "from_summary_estimates: derived weight from sampling variance column \
                     '{sampling_variance_column}' is not finite and positive at row {i}"
                )));
            }
            weights.push(weight);
        }

        let mut model = LinearMixedModel::new_with_compiler_policy(
            formula,
            data,
            Some(&weights),
            options.policy.clone(),
        )?;

        model.optsum.sigma = Some(1.0);
        model.residual_source = ResidualSource::FixedSamplingVariance;

        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formula::parse_formula;
    use crate::model::data::DataFrame;

    fn three_study_frame() -> DataFrame {
        let mut df = DataFrame::new();
        df.add_numeric("logrr", vec![0.10, -0.20, 0.05]).unwrap();
        df.add_numeric("v_logrr", vec![0.04, 0.09, 0.01]).unwrap();
        df.add_categorical(
            "study",
            vec!["A".to_string(), "B".to_string(), "C".to_string()],
        )
        .unwrap();
        df
    }

    #[test]
    fn defaults_match_contract() {
        let opts = SummaryEstimateOptions::default();
        assert_eq!(opts.variance_scale, SamplingVarianceScale::Absolute);
        assert_eq!(ResidualSource::default(), ResidualSource::EstimatedSigma);
        assert_eq!(
            SamplingVarianceScale::default(),
            SamplingVarianceScale::Absolute
        );
    }

    #[test]
    fn relative_scale_carries_sigma() {
        let scale = SamplingVarianceScale::Relative { sigma: 2.5 };
        match scale {
            SamplingVarianceScale::Relative { sigma } => assert_eq!(sigma, 2.5),
            SamplingVarianceScale::Absolute => panic!("expected Relative"),
        }
    }

    #[test]
    fn constructs_unfitted_model_with_fixed_sigma_and_marker() {
        let formula = parse_formula("logrr ~ 1 + (1 | study)").unwrap();
        let model = LinearMixedModel::from_summary_estimates(
            formula,
            &three_study_frame(),
            "logrr",
            "v_logrr",
            SummaryEstimateOptions::default(),
        )
        .expect("constructor should succeed");
        assert_eq!(model.residual_source, ResidualSource::FixedSamplingVariance);
        assert_eq!(model.optsum.sigma, Some(1.0));
        assert_eq!(model.optsum.feval, -1, "model must be unfitted");
        assert_eq!(model.dims.n, 3);
    }

    #[test]
    fn relative_scale_matches_absolute_when_sigma_one() {
        let formula = parse_formula("logrr ~ 1 + (1 | study)").unwrap();
        let abs_model = LinearMixedModel::from_summary_estimates(
            formula.clone(),
            &three_study_frame(),
            "logrr",
            "v_logrr",
            SummaryEstimateOptions::default(),
        )
        .unwrap();
        let rel_model = LinearMixedModel::from_summary_estimates(
            formula,
            &three_study_frame(),
            "logrr",
            "v_logrr",
            SummaryEstimateOptions {
                variance_scale: SamplingVarianceScale::Relative { sigma: 1.0 },
                ..SummaryEstimateOptions::default()
            },
        )
        .unwrap();
        assert_eq!(abs_model.sqrtwts.len(), rel_model.sqrtwts.len());
        for (a, b) in abs_model.sqrtwts.iter().zip(rel_model.sqrtwts.iter()) {
            assert!(
                (a - b).abs() < 1e-15,
                "sqrtwts must match between Absolute and Relative{{sigma=1}}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn rejects_non_positive_variance() {
        // Frame with a zero variance at row 1.
        let mut df = DataFrame::new();
        df.add_numeric("logrr", vec![0.10, -0.20, 0.05]).unwrap();
        df.add_numeric("v_logrr", vec![0.04, 0.0, 0.01]).unwrap();
        df.add_categorical(
            "study",
            vec!["A".to_string(), "B".to_string(), "C".to_string()],
        )
        .unwrap();
        let formula = parse_formula("logrr ~ 1 + (1 | study)").unwrap();
        let err = LinearMixedModel::from_summary_estimates(
            formula,
            &df,
            "logrr",
            "v_logrr",
            SummaryEstimateOptions::default(),
        )
        .unwrap_err();
        match err {
            MixedModelError::InvalidArgument(msg) => {
                assert!(msg.contains("non-positive"), "got: {msg}");
                assert!(msg.contains("row 1"), "got: {msg}");
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn rejects_non_finite_variance() {
        let mut df = DataFrame::new();
        df.add_numeric("logrr", vec![0.1, 0.2, 0.3]).unwrap();
        // Inject the non-finite sentinel past the add_numeric boundary guard
        // so the from_summary_estimates defense-in-depth check is exercised.
        df.add_numeric_unchecked("v_logrr", vec![0.04, f64::NAN, 0.01])
            .unwrap();
        df.add_categorical(
            "study",
            vec!["A".to_string(), "B".to_string(), "C".to_string()],
        )
        .unwrap();
        let formula = parse_formula("logrr ~ 1 + (1 | study)").unwrap();
        let err = LinearMixedModel::from_summary_estimates(
            formula,
            &df,
            "logrr",
            "v_logrr",
            SummaryEstimateOptions::default(),
        )
        .unwrap_err();
        match err {
            MixedModelError::InvalidArgument(msg) => {
                assert!(msg.contains("non-finite"), "got: {msg}");
                assert!(msg.contains("v_logrr"), "got: {msg}");
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn rejects_non_finite_estimate() {
        let mut df = DataFrame::new();
        // Inject the non-finite sentinel past the add_numeric boundary guard
        // so the from_summary_estimates defense-in-depth check is exercised.
        df.add_numeric_unchecked("logrr", vec![0.1, f64::INFINITY, 0.3])
            .unwrap();
        df.add_numeric("v_logrr", vec![0.04, 0.09, 0.01]).unwrap();
        df.add_categorical(
            "study",
            vec!["A".to_string(), "B".to_string(), "C".to_string()],
        )
        .unwrap();
        let formula = parse_formula("logrr ~ 1 + (1 | study)").unwrap();
        let err = LinearMixedModel::from_summary_estimates(
            formula,
            &df,
            "logrr",
            "v_logrr",
            SummaryEstimateOptions::default(),
        )
        .unwrap_err();
        match err {
            MixedModelError::InvalidArgument(msg) => {
                assert!(msg.contains("non-finite"), "got: {msg}");
                assert!(msg.contains("logrr"), "got: {msg}");
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn rejects_relative_with_non_positive_sigma() {
        let formula = parse_formula("logrr ~ 1 + (1 | study)").unwrap();
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let err = LinearMixedModel::from_summary_estimates(
                formula.clone(),
                &three_study_frame(),
                "logrr",
                "v_logrr",
                SummaryEstimateOptions {
                    variance_scale: SamplingVarianceScale::Relative { sigma: bad },
                    ..SummaryEstimateOptions::default()
                },
            )
            .unwrap_err();
            match err {
                MixedModelError::InvalidArgument(msg) => {
                    assert!(msg.contains("Relative"), "for sigma={bad}: {msg}");
                }
                other => panic!("expected InvalidArgument for sigma={bad}, got {other:?}"),
            }
        }
    }

    #[test]
    fn rejects_relative_with_overflowing_sigma_square() {
        let formula = parse_formula("logrr ~ 1 + (1 | study)").unwrap();
        let err = LinearMixedModel::from_summary_estimates(
            formula,
            &three_study_frame(),
            "logrr",
            "v_logrr",
            SummaryEstimateOptions {
                variance_scale: SamplingVarianceScale::Relative { sigma: f64::MAX },
                ..SummaryEstimateOptions::default()
            },
        )
        .unwrap_err();
        match err {
            MixedModelError::InvalidArgument(msg) => {
                assert!(msg.contains("squared sigma"), "got: {msg}");
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn rejects_normalized_variance_overflow() {
        let mut df = DataFrame::new();
        df.add_numeric("logrr", vec![0.1, 0.2, 0.3]).unwrap();
        df.add_numeric("v_logrr", vec![2.0, 1.0, 1.0]).unwrap();
        df.add_categorical(
            "study",
            vec!["A".to_string(), "B".to_string(), "C".to_string()],
        )
        .unwrap();
        let formula = parse_formula("logrr ~ 1 + (1 | study)").unwrap();
        let err = LinearMixedModel::from_summary_estimates(
            formula,
            &df,
            "logrr",
            "v_logrr",
            SummaryEstimateOptions {
                variance_scale: SamplingVarianceScale::Relative { sigma: 1e154 },
                ..SummaryEstimateOptions::default()
            },
        )
        .unwrap_err();
        match err {
            MixedModelError::InvalidArgument(msg) => {
                assert!(msg.contains("normalized sampling variance"), "got: {msg}");
                assert!(msg.contains("row 0"), "got: {msg}");
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn rejects_weight_overflow_from_tiny_variance() {
        let mut df = DataFrame::new();
        df.add_numeric("logrr", vec![0.1, 0.2, 0.3]).unwrap();
        df.add_numeric("v_logrr", vec![f64::MIN_POSITIVE / 8.0, 0.09, 0.01])
            .unwrap();
        df.add_categorical(
            "study",
            vec!["A".to_string(), "B".to_string(), "C".to_string()],
        )
        .unwrap();
        let formula = parse_formula("logrr ~ 1 + (1 | study)").unwrap();
        let err = LinearMixedModel::from_summary_estimates(
            formula,
            &df,
            "logrr",
            "v_logrr",
            SummaryEstimateOptions::default(),
        )
        .unwrap_err();
        match err {
            MixedModelError::InvalidArgument(msg) => {
                assert!(msg.contains("derived weight"), "got: {msg}");
                assert!(msg.contains("row 0"), "got: {msg}");
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn rejects_formula_response_mismatch() {
        let formula = parse_formula("v_logrr ~ 1 + (1 | study)").unwrap();
        let err = LinearMixedModel::from_summary_estimates(
            formula,
            &three_study_frame(),
            "logrr",
            "v_logrr",
            SummaryEstimateOptions::default(),
        )
        .unwrap_err();
        match err {
            MixedModelError::InvalidArgument(msg) => {
                assert!(msg.contains("formula response"), "got: {msg}");
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn rejects_same_estimate_and_variance_columns() {
        let formula = parse_formula("logrr ~ 1 + (1 | study)").unwrap();
        let err = LinearMixedModel::from_summary_estimates(
            formula,
            &three_study_frame(),
            "logrr",
            "logrr",
            SummaryEstimateOptions::default(),
        )
        .unwrap_err();
        match err {
            MixedModelError::InvalidArgument(msg) => {
                assert!(msg.contains("must differ"), "got: {msg}");
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    fn paired_study_frame() -> DataFrame {
        // Two observations per study so the (1|study) RE is identifiable
        // when fit unweighted — used for gate tests that need a successful
        // fit to traverse test_contrast_with_method.
        let mut df = DataFrame::new();
        df.add_numeric("y", vec![1.0, 1.5, 2.0, 0.8, 1.4, 2.1])
            .unwrap();
        df.add_categorical(
            "study",
            vec![
                "A".to_string(),
                "A".to_string(),
                "B".to_string(),
                "B".to_string(),
                "C".to_string(),
                "C".to_string(),
            ],
        )
        .unwrap();
        df
    }

    #[test]
    fn varest_returns_one_for_summary_estimate_fit() {
        let formula = parse_formula("logrr ~ 1 + (1 | study)").unwrap();
        let model = LinearMixedModel::from_summary_estimates(
            formula,
            &three_study_frame(),
            "logrr",
            "v_logrr",
            SummaryEstimateOptions::default(),
        )
        .unwrap();
        assert_eq!(model.varest(), 1.0);
    }

    #[test]
    fn kenward_roger_inherits_weighted_model_refusal() {
        let formula = parse_formula("logrr ~ 1 + (1 | study)").unwrap();
        let mut model = LinearMixedModel::from_summary_estimates(
            formula,
            &three_study_frame(),
            "logrr",
            "v_logrr",
            SummaryEstimateOptions::default(),
        )
        .unwrap();
        model
            .fit(true)
            .expect("summary-estimate fit should complete before KR refusal");
        // Summary-estimate fits always carry weights, so the existing
        // weighted-model gate at the entry of kenward_roger_sigma_g fires.
        let err = model.kenward_roger_sigma_g().unwrap_err();
        match err {
            MixedModelError::InvalidArgument(msg) => {
                assert!(
                    msg.contains("unweighted") || msg.contains("Kenward-Roger"),
                    "expected weighted-model refusal, got: {msg}"
                );
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn satterthwaite_refuses_summary_estimate_fit() {
        use crate::compiler::estimability::{
            FixedEffectHypothesis, FixedEffectTestMethod, InferenceStatus,
        };
        use crate::model::traits::MixedModelFit;

        // Use a paired-study frame so the unweighted LMM fit converges
        // cleanly. Then artificially mark the model as a summary-estimate
        // fit to drive the Satterthwaite gate without exercising the
        // meta-analysis fit-time path (which is covered by the metafor
        // parity issue).
        let formula = parse_formula("y ~ 1 + (1 | study)").unwrap();
        let mut model = LinearMixedModel::new(formula, &paired_study_frame(), None).unwrap();
        model.fit(true).expect("paired-study fit should converge");
        model.residual_source = ResidualSource::FixedSamplingVariance;

        let n_coef = model.coef_names().len();
        let hypothesis =
            FixedEffectHypothesis::single_coefficient("(Intercept) = 0", 0, n_coef).unwrap();
        let test =
            model.test_contrast_with_method(hypothesis, FixedEffectTestMethod::Satterthwaite);
        match test.status {
            InferenceStatus::NotAssessed { reason } => {
                assert!(
                    reason.contains("summary-estimate"),
                    "reason should name the model class: {reason}"
                );
                assert!(
                    reason.contains("sigma"),
                    "reason should explain the sigma constraint: {reason}"
                );
            }
            other => panic!("expected NotAssessed, got: {other:?}"),
        }
    }

    #[test]
    fn varcorr_omits_residual_row_for_summary_estimate_fit() {
        let formula = parse_formula("logrr ~ 1 + (1 | study)").unwrap();
        let model = LinearMixedModel::from_summary_estimates(
            formula,
            &three_study_frame(),
            "logrr",
            "v_logrr",
            SummaryEstimateOptions::default(),
        )
        .unwrap();
        let vc = model.varcorr();
        assert_eq!(vc.residual_source, ResidualSource::FixedSamplingVariance);
        let md = vc.to_markdown();
        assert!(
            !md.contains("Residual"),
            "markdown should omit residual: {md}"
        );
        let display = format!("{vc}");
        assert!(
            !display.contains("Residual"),
            "Display should omit residual: {display}"
        );
    }

    #[test]
    fn rejects_missing_variance_column() {
        let formula = parse_formula("logrr ~ 1 + (1 | study)").unwrap();
        let err = LinearMixedModel::from_summary_estimates(
            formula,
            &three_study_frame(),
            "logrr",
            "no_such_column",
            SummaryEstimateOptions::default(),
        )
        .unwrap_err();
        match err {
            MixedModelError::InvalidArgument(msg) => {
                assert!(msg.contains("not found"), "got: {msg}");
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    // ── Phase A of issue 6: Rust-only fit-path smoke test. ───────────────
    // The metafor::rma.mv cross-check lives in a separate child issue with
    // its own R-bridge harness. Here we just confirm the fit path runs
    // end-to-end on a real meta-analysis fixture.

    #[test]
    fn fit_smoke_three_study_frame_does_not_panic() {
        use crate::model::traits::MixedModelFit;

        let formula = parse_formula("logrr ~ 1 + (1 | study)").unwrap();
        let mut model = LinearMixedModel::from_summary_estimates(
            formula,
            &three_study_frame(),
            "logrr",
            "v_logrr",
            SummaryEstimateOptions::default(),
        )
        .unwrap();
        model
            .fit(true)
            .expect("REML fit on three_study_frame should not error");
        assert!(model.optsum.feval > 0, "fit must record evaluations");
        let beta = model.coef();
        assert!(
            beta.iter().all(|b| b.is_finite()),
            "fitted beta must be finite: {beta:?}"
        );
        let tau_sd = model.varcorr().components[0].std_dev[0];
        assert!(
            tau_sd >= 0.0,
            "tau (random-effect SD) must be non-negative: {tau_sd}"
        );
        // The fit class must remain a summary-estimate fit after fitting.
        assert_eq!(model.residual_source, ResidualSource::FixedSamplingVariance);
        assert_eq!(model.optsum.sigma, Some(1.0));
    }

    #[test]
    fn fit_recovers_intercept_with_zero_heterogeneity() {
        use crate::model::traits::MixedModelFit;

        // Five identical estimates with varying sampling variances —
        // a valid meta-analysis input. The fit() short-circuit detects
        // this case and returns the analytical answer: beta = common
        // value, tau = 0, sidestepping the optimizer's degenerate
        // Cholesky path.
        let common = 0.123_456;
        let mut df = DataFrame::new();
        df.add_numeric("logrr", vec![common; 5]).unwrap();
        df.add_numeric("v_logrr", vec![0.04, 0.09, 0.01, 0.16, 0.025])
            .unwrap();
        df.add_categorical(
            "study",
            vec![
                "A".to_string(),
                "B".to_string(),
                "C".to_string(),
                "D".to_string(),
                "E".to_string(),
            ],
        )
        .unwrap();
        let formula = parse_formula("logrr ~ 1 + (1 | study)").unwrap();
        let mut model = LinearMixedModel::from_summary_estimates(
            formula,
            &df,
            "logrr",
            "v_logrr",
            SummaryEstimateOptions::default(),
        )
        .unwrap();
        model
            .fit(true)
            .expect("constant-response summary-estimate fit should short-circuit cleanly");

        let beta = model.coef();
        assert_eq!(beta.len(), 1, "intercept-only model");
        assert!(
            (beta[0] - common).abs() < 1e-12,
            "fitted intercept {} should equal common value {} for constant-response shortcut",
            beta[0],
            common
        );

        let tau_sd = model.varcorr().components[0].std_dev[0];
        assert!(
            tau_sd < 1e-12,
            "tau should be 0 (theta fixed at lower bound) for constant-response shortcut, got {tau_sd}"
        );

        assert_eq!(model.optsum.return_value, "CONSTANT_RESPONSE_SHORTCIRCUIT");
        assert_eq!(model.optsum.feval, 1);
    }

    #[test]
    fn ordinary_lmm_still_rejects_constant_response() {
        // The relaxation must scope strictly to FixedSamplingVariance
        // fits. Ordinary unweighted LMMs continue to reject a constant
        // response, since for those it really is a degenerate fit.
        let mut df = DataFrame::new();
        df.add_numeric("y", vec![1.5; 6]).unwrap();
        df.add_categorical(
            "g",
            vec![
                "A".to_string(),
                "A".to_string(),
                "B".to_string(),
                "B".to_string(),
                "C".to_string(),
                "C".to_string(),
            ],
        )
        .unwrap();
        let formula = parse_formula("y ~ 1 + (1 | g)").unwrap();
        let mut model = LinearMixedModel::new(formula, &df, None).unwrap();
        let err = model.fit(true).unwrap_err();
        match err {
            MixedModelError::ConstantResponse => {}
            other => panic!("expected ConstantResponse, got {other:?}"),
        }
    }
}
