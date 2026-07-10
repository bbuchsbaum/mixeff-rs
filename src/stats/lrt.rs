//! Likelihood ratio tests for nested mixed models.

use std::fmt;

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::linalg::stats_rank;
use crate::model::linear::LinearMixedModel;
use crate::model::traits::{MixedModelFit, RandomEffectTermInfo};
use crate::types::OptSummary;

const LOG_LIK_TOL: f64 = 1.0e-10;
/// Stable schema name for boundary-aware LRT payloads.
pub const BOUNDARY_LRT_SCHEMA: &str = "mixedmodels.boundary_lrt";
/// Stable schema version for boundary-aware LRT payloads.
pub const BOUNDARY_LRT_SCHEMA_VERSION: &str = "1.0.0";
/// Stable schema name for parametric-bootstrap LRT payloads.
pub const PARAMETRIC_BOOTSTRAP_LRT_SCHEMA: &str = "mixedmodels.parametric_bootstrap_lrt";
/// Stable schema version for parametric-bootstrap LRT payloads.
pub const PARAMETRIC_BOOTSTRAP_LRT_SCHEMA_VERSION: &str = "1.0.0";

/// Ordinary Gaussian linear-model fit for comparison with mixed models.
///
/// This is the no-random-effects case used by likelihood-ratio tests. It lets
/// callers compare `lm`-style fits with mixed models through the same
/// [`MixedModelFit`] interface while retaining structural nestedness checks.
#[derive(Debug, Clone)]
pub struct LinearModelFit {
    response: DVector<f64>,
    model_matrix: DMatrix<f64>,
    coefficients: DVector<f64>,
    fitted_values: DVector<f64>,
    residual_values: DVector<f64>,
    covariance: DMatrix<f64>,
    standard_errors: DVector<f64>,
    sigma: f64,
    loglik: f64,
    rank: usize,
    formula: Option<String>,
    optsum: OptSummary,
}

impl LinearModelFit {
    /// Fit an ordinary Gaussian linear model by least squares.
    pub fn fit(
        response: DVector<f64>,
        model_matrix: DMatrix<f64>,
        formula: Option<String>,
    ) -> Result<Self, String> {
        if model_matrix.nrows() != response.len() {
            return Err("response length must match model matrix rows".to_string());
        }
        if response.is_empty() {
            return Err("linear model requires at least one observation".to_string());
        }

        let rank = stats_rank(&model_matrix).0;
        if rank != model_matrix.ncols() {
            return Err(
                "linear model comparison currently requires a full-rank model matrix".to_string(),
            );
        }
        if rank >= response.len() {
            return Err(
                "linear model requires residual degrees of freedom for variance estimation"
                    .to_string(),
            );
        }

        let xtx = model_matrix.transpose() * &model_matrix;
        let xty = model_matrix.transpose() * &response;
        let coefficients = xtx
            .clone()
            .lu()
            .solve(&xty)
            .ok_or_else(|| "linear model least-squares solve failed".to_string())?;
        let fitted_values = &model_matrix * &coefficients;
        let residual_values = &response - &fitted_values;
        let rss = residual_values.dot(&residual_values);
        let n = response.len() as f64;
        let sigma_sq_mle = rss / n;
        let sigma = sigma_sq_mle.sqrt();
        let loglik = if sigma_sq_mle > 0.0 {
            -0.5 * n * ((2.0 * std::f64::consts::PI).ln() + 1.0 + sigma_sq_mle.ln())
        } else {
            f64::INFINITY
        };

        let xtx_inv = xtx
            .try_inverse()
            .ok_or_else(|| "linear model covariance solve failed".to_string())?;
        let sigma_sq_unbiased = rss / (response.len() - rank) as f64;
        let covariance = xtx_inv * sigma_sq_unbiased;
        let standard_errors = DVector::from_iterator(
            covariance.ncols(),
            (0..covariance.ncols()).map(|idx| covariance[(idx, idx)].sqrt()),
        );

        let mut optsum = OptSummary::new(Vec::new());
        optsum.reml = false;

        Ok(Self {
            response,
            model_matrix,
            coefficients,
            fitted_values,
            residual_values,
            covariance,
            standard_errors,
            sigma,
            loglik,
            rank,
            formula,
            optsum,
        })
    }
}

impl MixedModelFit for LinearModelFit {
    fn nobs(&self) -> usize {
        self.response.len()
    }

    fn dof(&self) -> usize {
        self.rank + 1
    }

    fn coef(&self) -> DVector<f64> {
        self.coefficients.clone()
    }

    fn fixef(&self) -> DVector<f64> {
        self.coefficients.clone()
    }

    fn coef_names(&self) -> Vec<String> {
        (0..self.coefficients.len())
            .map(|idx| format!("x{idx}"))
            .collect()
    }

    fn vcov(&self) -> DMatrix<f64> {
        self.covariance.clone()
    }

    fn stderror(&self) -> DVector<f64> {
        self.standard_errors.clone()
    }

    fn fitted(&self) -> DVector<f64> {
        self.fitted_values.clone()
    }

    fn residuals(&self) -> DVector<f64> {
        self.residual_values.clone()
    }

    fn response(&self) -> &DVector<f64> {
        &self.response
    }

    fn model_matrix(&self) -> &DMatrix<f64> {
        &self.model_matrix
    }

    fn objective(&self) -> f64 {
        -2.0 * self.loglik
    }

    fn loglikelihood(&self) -> f64 {
        self.loglik
    }

    fn formula_label(&self) -> Option<String> {
        self.formula.clone()
    }

    fn is_fitted(&self) -> bool {
        true
    }

    fn is_singular(&self) -> bool {
        false
    }

    fn opt_summary(&self) -> &OptSummary {
        &self.optsum
    }

    fn theta(&self) -> Vec<f64> {
        Vec::new()
    }

    fn dispersion(&self, sqr: bool) -> f64 {
        if sqr {
            self.sigma * self.sigma
        } else {
            self.sigma
        }
    }

    fn ranef(&self) -> Vec<DMatrix<f64>> {
        Vec::new()
    }
}

/// Result of a likelihood ratio test comparing nested models.
///
/// # Boundary caveat (variance-component comparisons)
///
/// This is the classical χ²(Δdf) test. When the models differ by one or
/// more **random-effects variance components**, the null places that
/// parameter on the boundary of its space (σ² = 0), so the naive χ²(Δdf)
/// reference distribution is **anti-conservative** (p-values too small;
/// the correct null is a ½:½ χ̄² mixture for one boundary parameter). This
/// struct intentionally reports the unadjusted statistic — matching
/// `lme4`/MixedModels.jl behaviour — and does not itself detect or correct
/// the boundary case. For variance-component comparisons use
/// [`BoundaryLikelihoodRatioTest`] (certified χ̄² mixture for the
/// one-added-boundary-parameter case) or [`parametric_bootstrap_lrt`],
/// which are statistically honest at the boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LikelihoodRatioTest {
    /// Number of observations (must be equal across models).
    pub nobs: usize,
    /// Canonical formula label for each model, when available.
    pub formulas: Vec<String>,
    /// Degrees of freedom for each model.
    pub dof: Vec<usize>,
    /// Log-likelihood for each model.
    pub loglik: Vec<f64>,
    /// Deviance (-2 * loglik) for each model.
    pub deviance: Vec<f64>,
    /// Chi-squared statistics (between successive models).
    pub chisq: Vec<f64>,
    /// Degrees of freedom for each chi-squared test.
    pub chisq_dof: Vec<usize>,
    /// P-values for each test.
    pub pvalues: Vec<f64>,
    /// Whether the larger model had a lower log-likelihood within optimizer tolerance.
    pub loglik_within_optimizer_tol: Vec<bool>,
}

/// High-level structural class for comparing two fitted model objects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ModelComparisonClass {
    /// Same response, family/link, fit criterion, fixed-effect space, random
    /// terms, and model degrees of freedom.
    SameModelSpace,
    /// Fixed-effect space grows while random-effect terms are unchanged.
    NestedFixedEffects,
    /// Random-effect terms grow while fixed-effect space is unchanged.
    NestedRandomEffects,
    /// Both fixed-effect space and random-effect terms grow.
    NestedFixedAndRandomEffects,
    /// Fixed effects and random-effect columns are unchanged, but degrees of
    /// freedom differ. This is the shape of covariance-parameter comparisons
    /// such as diagonal versus full covariance for the same random basis.
    SameFixedEffectsCovarianceDifference,
    /// Fixed-effect spaces are incompatible for a nested-model LRT.
    NonNestedFixedEffects,
    /// Random-effect terms are incompatible for a nested-model LRT.
    NonNestedRandomEffects,
    /// Models were not fitted to the same response values.
    DifferentResponse,
    /// Conditional response families differ.
    DifferentFamily,
    /// Link functions differ.
    DifferentLink,
    /// REML-fitted and ML-fitted models were mixed.
    MixedFitCriterion,
    /// Models have equal or decreasing degrees of freedom in the requested
    /// order, even though their spaces otherwise look nested.
    InvalidModelOrder,
}

/// Coarse fixed-effect relation used by [`ModelComparisonAssessment`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum FixedEffectComparison {
    /// Fixed-effect column spaces are equal.
    Same,
    /// The first model's fixed-effect space is nested in the second.
    Nested,
    /// The second model's fixed-effect space is nested in the first.
    ReverseNested,
    /// Fixed-effect spaces are not nested.
    NonNested,
}

/// Coarse random-effect relation used by [`ModelComparisonAssessment`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum RandomEffectComparison {
    /// Random-effect terms are equal.
    Same,
    /// The first model's random-effect terms are nested in the second.
    Nested,
    /// The second model's random-effect terms are nested in the first.
    ReverseNested,
    /// Random-effect terms are not nested.
    NonNested,
}

/// Suggested valid comparison route when the requested LRT is unavailable or
/// not the best default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ModelComparisonAlternative {
    /// Refit both models with ML and run an ordinary LRT.
    MlRefitLikelihoodRatio,
    /// Use the REML likelihood comparison when fixed-effect spaces match.
    RemlLikelihoodRatio,
    /// Compare AIC/BIC instead of an LRT p-value.
    InformationCriteria,
    /// Test a fixed-effect contrast in a single fitted model.
    FixedEffectContrastTest,
    /// Use a simulation-calibrated LRT reference distribution.
    ParametricBootstrap,
    /// Compare predictive performance with held-out data.
    CrossValidation,
    /// Reverse model order before comparing.
    ReorderModels,
    /// Refit with a common likelihood criterion before comparing.
    RefitWithCommonCriterion,
}

/// Structured preflight assessment for comparing two fitted models.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelComparisonAssessment {
    /// Structural comparison class for the pair.
    pub class: ModelComparisonClass,
    /// Fixed-effect column-space relation.
    pub fixed_effects: FixedEffectComparison,
    /// Random-effect term relation.
    pub random_effects: RandomEffectComparison,
    /// Whether an ordinary LRT is valid for the pair as supplied.
    pub lrt_available: bool,
    /// Human-readable reason when `lrt_available` is false.
    pub lrt_reason: Option<String>,
    /// Whether AIC/BIC-style comparison is meaningful for the pair.
    pub information_criteria_available: bool,
    /// Human-readable reason when information criteria are unavailable.
    pub information_criteria_reason: Option<String>,
    /// Whether ML refitting would be required for the requested comparison.
    pub ml_refit_required: bool,
    /// Human-readable reason for ML refit requirement.
    pub ml_refit_reason: Option<String>,
    /// Suggested comparison routes that remain valid.
    pub valid_alternatives: Vec<ModelComparisonAlternative>,
}

/// Requested comparison mode for [`ModelComparisonTable`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ModelComparisonMethod {
    /// Use likelihood-ratio columns for adjacent comparisons where they are
    /// valid; otherwise leave them empty and report the reason.
    Auto,
    /// Require likelihood-ratio comparisons unless a non-refitting policy asks
    /// the table to report why refitting is needed.
    LikelihoodRatio,
    /// Report information-criteria rows without likelihood-ratio statistics.
    InformationCriteria,
}

/// Policy for comparisons that would require ML refits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ModelComparisonRefitPolicy {
    /// Return an error when a requested comparison would require ML refits.
    Error,
    /// Mark the row as requiring ML refits. The Rust layer does not refit.
    Ml,
    /// Never refit; mark the row and leave likelihood-ratio columns empty.
    Never,
}

/// Stable reason code for a model-comparison table row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ModelComparisonReasonCode {
    /// Information-criteria table requested no LRT.
    InformationCriteriaRequested,
    /// Models are structurally non-nested for LRT.
    NonNestedModelsLrtInvalid,
    /// REML comparison would require ML refits.
    MlRefitRequired,
    /// Larger model had a lower log-likelihood beyond tolerance.
    LowerLoglikelihoodLrtInvalid,
    /// Models occupy the same parameter space, so no LRT increment exists.
    SameModelSpaceLrtInvalid,
    /// Models were fit to different response values.
    DifferentResponse,
    /// Conditional response families differ.
    DifferentFamily,
    /// Link functions differ.
    DifferentLink,
    /// REML and ML fits were mixed.
    MixedFitCriterion,
    /// Models are ordered from larger to smaller or have non-increasing dof.
    InvalidModelOrder,
    /// LRT is unavailable for another classified reason.
    LrtUnavailable,
}

impl ModelComparisonReasonCode {
    /// Stable snake-case reason code.
    pub fn as_str(&self) -> &'static str {
        match self {
            ModelComparisonReasonCode::InformationCriteriaRequested => {
                "information_criteria_requested"
            }
            ModelComparisonReasonCode::NonNestedModelsLrtInvalid => "non_nested_models_lrt_invalid",
            ModelComparisonReasonCode::MlRefitRequired => "ml_refit_required",
            ModelComparisonReasonCode::LowerLoglikelihoodLrtInvalid => {
                "lower_loglikelihood_lrt_invalid"
            }
            ModelComparisonReasonCode::SameModelSpaceLrtInvalid => "same_model_space_lrt_invalid",
            ModelComparisonReasonCode::DifferentResponse => "different_response",
            ModelComparisonReasonCode::DifferentFamily => "different_family",
            ModelComparisonReasonCode::DifferentLink => "different_link",
            ModelComparisonReasonCode::MixedFitCriterion => "mixed_fit_criterion",
            ModelComparisonReasonCode::InvalidModelOrder => "invalid_model_order",
            ModelComparisonReasonCode::LrtUnavailable => "lrt_unavailable",
        }
    }
}

/// Options for building a [`ModelComparisonTable`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelComparisonOptions {
    /// Table construction method.
    pub method: ModelComparisonMethod,
    /// Refitting behavior when ML refits would be needed.
    pub refit_policy: ModelComparisonRefitPolicy,
}

impl Default for ModelComparisonOptions {
    fn default() -> Self {
        Self {
            method: ModelComparisonMethod::Auto,
            refit_policy: ModelComparisonRefitPolicy::Never,
        }
    }
}

/// One display row in a model comparison table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelComparisonRow {
    /// Model label, usually the formula.
    pub label: String,
    /// Number of observations.
    pub nobs: usize,
    /// Model degrees of freedom.
    pub dof: usize,
    /// Model log-likelihood.
    pub loglik: f64,
    /// Model deviance, `-2 * loglik`.
    pub deviance: f64,
    /// Akaike information criterion.
    pub aic: f64,
    /// Bayesian information criterion.
    pub bic: f64,
    /// Difference from the minimum AIC in the table.
    pub delta_aic: f64,
    /// Difference from the minimum BIC in the table.
    pub delta_bic: f64,
    /// Adjacent likelihood-ratio statistic when available.
    pub chisq: Option<f64>,
    /// Degrees of freedom for `chisq`.
    pub chisq_dof: Option<usize>,
    /// LRT p-value when available.
    pub pvalue: Option<f64>,
    /// Whether a non-positive log-likelihood difference was within optimizer tolerance.
    pub loglik_within_optimizer_tol: Option<bool>,
    /// Structural class for the adjacent comparison.
    pub comparison_class: Option<ModelComparisonClass>,
    /// Whether LRT columns are valid for this row.
    pub lrt_available: bool,
    /// Whether information criteria are valid for this row.
    pub information_criteria_available: bool,
    /// Whether this row would require ML refitting for LRT.
    pub requires_ml_refit: bool,
    /// Stable reason code when LRT columns are unavailable.
    pub reason_code: Option<String>,
    /// Human-readable reason when LRT columns are unavailable.
    pub reason: Option<String>,
}

/// Information-criteria table with optional valid LRT columns.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelComparisonTable {
    /// Requested comparison method.
    pub method: ModelComparisonMethod,
    /// Refitting policy used to build the table.
    pub refit_policy: ModelComparisonRefitPolicy,
    /// Display rows, one per model.
    pub rows: Vec<ModelComparisonRow>,
    /// Pairwise assessments for adjacent model pairs.
    pub assessments: Vec<ModelComparisonAssessment>,
    /// Present for API honesty: automatic refitting is not performed here.
    pub refit_performed: bool,
}

/// Status for a boundary-aware variance-component LRT route.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum BoundaryLrtStatus {
    /// Boundary-aware reference distribution was certified and evaluated.
    Available,
    /// The comparison shape was plausible but not certified by this route.
    NotAssessed,
    /// Boundary-aware LRT is unsupported for this comparison class.
    Unsupported,
}

/// One component of a chi-square mixture reference distribution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundaryLrtMixtureComponent {
    /// Mixture weight for this reference-distribution component.
    pub weight: f64,
    /// Chi-square degrees of freedom, or `None` for a point mass.
    pub chisq_df: Option<usize>,
    /// Point-mass location when this component is degenerate.
    pub point_mass_at: Option<f64>,
}

/// Boundary-aware LRT payload for variance-component comparisons.
///
/// This deliberately does not represent a fixed-effect p-value route. The
/// certified v1 route is the classic one-added-boundary-parameter comparison
/// with the 50:50 mixture of a point mass at zero and chi-square(1).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundaryLikelihoodRatioTest {
    /// Stable schema name.
    pub schema_name: String,
    /// Stable schema version.
    pub schema_version: String,
    /// Boundary LRT availability status.
    pub status: BoundaryLrtStatus,
    /// Stable reason code when unavailable.
    pub reason_code: Option<String>,
    /// Human-readable reason when unavailable.
    pub reason: Option<String>,
    /// Structural comparison class when it could be assessed.
    pub comparison_class: Option<ModelComparisonClass>,
    /// Observed likelihood-ratio statistic.
    pub statistic: Option<f64>,
    /// Ordinary chi-square degrees of freedom before boundary adjustment.
    pub ordinary_chisq_dof: Option<usize>,
    /// Boundary-adjusted p-value when available.
    pub pvalue: Option<f64>,
    /// Whether a non-positive log-likelihood difference was within optimizer tolerance.
    pub loglik_within_optimizer_tol: Option<bool>,
    /// Certified mixture reference distribution.
    pub mixture: Vec<BoundaryLrtMixtureComponent>,
    /// Literature references supporting the route.
    pub references: Vec<String>,
    /// Reader-facing caveats and interpretation notes.
    pub notes: Vec<String>,
}

impl BoundaryLikelihoodRatioTest {
    /// Assess and, when certified, compute a boundary-aware variance-component LRT.
    pub fn variance_component(smaller: &dyn MixedModelFit, larger: &dyn MixedModelFit) -> Self {
        let assessment = ModelComparisonAssessment::assess(smaller, larger);
        let comparison_class = Some(assessment.class);
        let references = vec![
            "Self and Liang (1987), JASA 82:605-610".to_string(),
            "Stram and Lee (1994), Biometrics 50:1171-1177".to_string(),
        ];

        if !matches!(
            assessment.class,
            ModelComparisonClass::NestedRandomEffects
                | ModelComparisonClass::SameFixedEffectsCovarianceDifference
        ) {
            return Self::refusal(
                BoundaryLrtStatus::Unsupported,
                "boundary_lrt_requires_variance_component_comparison",
                "boundary_lrt is only certified for nested random-effect or covariance-parameter comparisons with identical fixed effects",
                comparison_class,
                references,
            );
        }

        if assessment.fixed_effects != FixedEffectComparison::Same {
            return Self::refusal(
                BoundaryLrtStatus::Unsupported,
                "boundary_lrt_not_fixed_effect_method",
                "boundary_lrt is a variance-component route, not a fixed-effect p-value method",
                comparison_class,
                references,
            );
        }

        if !assessment.lrt_available {
            return Self::refusal(
                BoundaryLrtStatus::NotAssessed,
                "boundary_lrt_likelihood_comparison_unavailable",
                assessment
                    .lrt_reason
                    .as_deref()
                    .unwrap_or("ordinary likelihood comparison is unavailable for this model pair"),
                comparison_class,
                references,
            );
        }

        let values = match likelihood_ratio_values(smaller, larger) {
            Ok(values) => values,
            Err(reason) => {
                return Self::refusal(
                    BoundaryLrtStatus::NotAssessed,
                    "boundary_lrt_likelihood_comparison_unavailable",
                    &reason,
                    comparison_class,
                    references,
                );
            }
        };

        if values.chisq_dof != 1 {
            return Self::refusal(
                BoundaryLrtStatus::NotAssessed,
                "boundary_lrt_mixture_weights_not_certified",
                "boundary_lrt v1 certifies only one added boundary variance/covariance parameter; for higher-dimensional boundaries use the boundary-robust parametric-bootstrap LRT (stats::parametric_bootstrap_lrt) or a simulation-calibrated mixture",
                comparison_class,
                references,
            );
        }

        let pvalue = self_liang_one_parameter_pvalue(values.chisq);
        Self {
            schema_name: BOUNDARY_LRT_SCHEMA.to_string(),
            schema_version: BOUNDARY_LRT_SCHEMA_VERSION.to_string(),
            status: BoundaryLrtStatus::Available,
            reason_code: None,
            reason: None,
            comparison_class,
            statistic: Some(values.chisq),
            ordinary_chisq_dof: Some(values.chisq_dof),
            pvalue: Some(pvalue),
            loglik_within_optimizer_tol: Some(values.loglik_within_optimizer_tol),
            mixture: self_liang_one_parameter_mixture(),
            references,
            notes: vec![
                "p-value uses a 50:50 mixture of point mass at zero and chi-square(1) for a single boundary variance/covariance parameter".to_string(),
                "this route is for variance-component comparisons and must not be surfaced as fixed-effect inference".to_string(),
            ],
        }
    }

    fn refusal(
        status: BoundaryLrtStatus,
        reason_code: &str,
        reason: &str,
        comparison_class: Option<ModelComparisonClass>,
        references: Vec<String>,
    ) -> Self {
        Self {
            schema_name: BOUNDARY_LRT_SCHEMA.to_string(),
            schema_version: BOUNDARY_LRT_SCHEMA_VERSION.to_string(),
            status,
            reason_code: Some(reason_code.to_string()),
            reason: Some(reason.to_string()),
            comparison_class,
            statistic: None,
            ordinary_chisq_dof: None,
            pvalue: None,
            loglik_within_optimizer_tol: None,
            mixture: Vec::new(),
            references,
            notes: vec![
                "boundary_lrt is intentionally restricted until the comparison geometry certifies the reference distribution".to_string(),
            ],
        }
    }
}

impl ModelComparisonTable {
    /// Build a comparison table using automatic method selection and no refits.
    pub fn compare(models: &[&dyn MixedModelFit]) -> Result<Self, String> {
        Self::compare_with_options(models, ModelComparisonOptions::default())
    }

    /// Build a comparison table with an explicit method and refit policy.
    pub fn compare_with_options(
        models: &[&dyn MixedModelFit],
        options: ModelComparisonOptions,
    ) -> Result<Self, String> {
        if models.len() < 2 {
            return Err("At least two models are needed".to_string());
        }

        let assessments = assess_model_comparison_sequence(models)?;
        validate_model_comparison_options(&assessments, options)?;

        let aic: Vec<f64> = models.iter().map(|model| model.aic()).collect();
        let bic: Vec<f64> = models.iter().map(|model| model.bic()).collect();
        let min_aic = finite_min(&aic);
        let min_bic = finite_min(&bic);
        let mut rows = Vec::with_capacity(models.len());

        for (idx, model) in models.iter().enumerate() {
            let assessment = idx.checked_sub(1).map(|previous| &assessments[previous]);
            let lrt_values = assessment.and_then(|assessment| {
                if options.method == ModelComparisonMethod::InformationCriteria {
                    None
                } else if assessment.lrt_available {
                    Some(likelihood_ratio_values(models[idx - 1], *model))
                } else {
                    None
                }
            });

            let (chisq, chisq_dof, pvalue, within_tol, lrt_reason) = match lrt_values {
                Some(Ok(values)) => (
                    Some(values.chisq),
                    Some(values.chisq_dof),
                    Some(values.pvalue),
                    Some(values.loglik_within_optimizer_tol),
                    None,
                ),
                Some(Err(reason)) => (None, None, None, None, Some(reason)),
                None => (None, None, None, None, None),
            };

            let reason = comparison_row_reason(assessment, options.method, lrt_reason);
            let reason_code =
                comparison_row_reason_code(assessment, options.method, reason.as_deref())
                    .map(|code| code.as_str().to_string());
            let lrt_available = chisq.is_some();
            rows.push(ModelComparisonRow {
                label: model
                    .formula_label()
                    .unwrap_or_else(|| format!("model{}", idx + 1)),
                nobs: model.nobs(),
                dof: model.dof(),
                loglik: model.loglikelihood(),
                deviance: -2.0 * model.loglikelihood(),
                aic: aic[idx],
                bic: bic[idx],
                delta_aic: delta_from_min(aic[idx], min_aic),
                delta_bic: delta_from_min(bic[idx], min_bic),
                chisq,
                chisq_dof,
                pvalue,
                loglik_within_optimizer_tol: within_tol,
                comparison_class: assessment.map(|assessment| assessment.class),
                lrt_available,
                information_criteria_available: assessment
                    .map(|assessment| assessment.information_criteria_available)
                    .unwrap_or(true),
                requires_ml_refit: assessment
                    .map(|assessment| assessment.ml_refit_required)
                    .unwrap_or(false),
                reason_code,
                reason,
            });
        }

        Ok(Self {
            method: options.method,
            refit_policy: options.refit_policy,
            rows,
            assessments,
            refit_performed: false,
        })
    }
}

impl ModelComparisonAssessment {
    /// Assess a pair of models in the supplied order.
    ///
    /// The order is meaningful for LRT availability: the first model is treated
    /// as the smaller/null model and the second as the larger/alternative model.
    pub fn assess(smaller: &dyn MixedModelFit, larger: &dyn MixedModelFit) -> Self {
        assess_model_pair(smaller, larger)
    }

    /// Whether an ordinary likelihood-ratio test is valid for this pair.
    pub fn lrt_is_available(&self) -> bool {
        self.lrt_available
    }

    /// Whether AIC/BIC-style information criteria are valid for this pair.
    pub fn information_criteria_are_available(&self) -> bool {
        self.information_criteria_available
    }
}

/// Assess each adjacent pair in a model sequence.
pub fn assess_model_comparison_sequence(
    models: &[&dyn MixedModelFit],
) -> Result<Vec<ModelComparisonAssessment>, String> {
    if models.len() < 2 {
        return Err("At least two models are needed".to_string());
    }
    Ok(models
        .windows(2)
        .map(|pair| ModelComparisonAssessment::assess(pair[0], pair[1]))
        .collect())
}

impl LikelihoodRatioTest {
    /// Perform a likelihood ratio test on two or more nested models.
    ///
    /// Models should be provided in order from smallest to largest.
    pub fn test(models: &[&dyn MixedModelFit]) -> Result<Self, String> {
        let formulas = models
            .iter()
            .map(|m| m.formula_label().unwrap_or_else(|| "NA".to_string()))
            .collect();
        Self::test_with_formulas(models, formulas)
    }

    /// Perform a likelihood ratio test with explicit formula labels.
    pub fn test_with_formulas(
        models: &[&dyn MixedModelFit],
        formulas: Vec<String>,
    ) -> Result<Self, String> {
        if models.len() < 2 {
            return Err("At least two models are needed".to_string());
        }
        if formulas.len() != models.len() {
            return Err("Formula labels must match the number of models".to_string());
        }

        let assessments = assess_model_comparison_sequence(models)?;
        for assessment in &assessments {
            if !assessment.lrt_available {
                return Err(assessment.lrt_reason.clone().unwrap_or_else(|| {
                    "ordinary likelihood-ratio test is unavailable for this comparison".to_string()
                }));
            }
        }

        let nobs = models[0].nobs();
        let dof: Vec<usize> = models.iter().map(|m| m.dof()).collect();
        let loglik: Vec<f64> = models.iter().map(|m| m.loglikelihood()).collect();
        let deviance: Vec<f64> = loglik.iter().map(|ll| -2.0 * ll).collect();

        let mut chisq = Vec::new();
        let mut chisq_dof = Vec::new();
        let mut pvalues = Vec::new();
        let mut loglik_within_optimizer_tol = Vec::new();

        for i in 1..models.len() {
            debug_assert!(
                dof[i] > dof[i - 1],
                "comparison classifier should reject non-increasing degrees of freedom"
            );
            let loglik_diff = loglik[i] - loglik[i - 1];
            if loglik_diff < -LOG_LIK_TOL {
                return Err(
                    "Log-likelihood must not be lower in models with more degrees of freedom"
                        .to_string(),
                );
            }

            let within_optimizer_tol = loglik_diff <= 0.0;
            let chi = if within_optimizer_tol {
                0.0
            } else {
                2.0 * loglik_diff
            };
            let ddof = dof[i] - dof[i - 1];
            use statrs::distribution::{ChiSquared, ContinuousCDF};
            let dist = ChiSquared::new(ddof as f64).unwrap();
            let pval = 1.0 - dist.cdf(chi);
            chisq.push(chi);
            chisq_dof.push(ddof);
            pvalues.push(pval);
            loglik_within_optimizer_tol.push(within_optimizer_tol);
        }

        Ok(LikelihoodRatioTest {
            nobs,
            formulas,
            dof,
            loglik,
            deviance,
            chisq,
            chisq_dof,
            pvalues,
            loglik_within_optimizer_tol,
        })
    }

    /// Extract the p-value when exactly one comparison is present.
    pub fn pvalue(&self) -> Result<f64, String> {
        match self.pvalues.as_slice() {
            [pvalue] => Ok(*pvalue),
            _ => Err("Cannot extract only one p-value from a multiple test result.".to_string()),
        }
    }

    /// Render the likelihood-ratio test as a markdown table.
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str("|                                          | model-dof | -2 logLik |  χ² | χ²-dof | P(>χ²) |\n");
        out.push_str("|:---------------------------------------- | ---------:| ---------:| ---:| ------:|:------ |\n");

        out.push_str(&format!(
            "| {:<40} | {:>9} | {:>9} | {:>3} | {:>6} | {:<6} |\n",
            escape_markdown_pipes(&self.formulas[0]),
            self.dof[0],
            (2.0 * self.loglik[0]).round() as i64,
            "",
            "",
            ""
        ));

        for i in 1..self.formulas.len() {
            out.push_str(&format!(
                "| {:<40} | {:>9} | {:>9} | {:>3} | {:>6} | {:<6} |\n",
                escape_markdown_pipes(&self.formulas[i]),
                self.dof[i],
                (2.0 * self.loglik[i]).round() as i64,
                self.chisq[i - 1].round() as i64,
                self.chisq_dof[i - 1],
                format_pvalue(self.pvalues[i - 1])
            ));
        }

        out
    }

    /// Render the likelihood-ratio test as an HTML table.
    pub fn to_html(&self) -> String {
        let rows = self.html_rows();
        let mut out = String::from("<table><tr>");

        for cell in &rows[0] {
            out.push_str(&format!("<th align=\"right\">{cell}</th>"));
        }
        out.push_str("</tr>");

        for row in rows.iter().skip(1) {
            out.push_str("<tr>");
            for (idx, cell) in row.iter().enumerate() {
                let align = if idx == 0 { "left" } else { "right" };
                out.push_str(&format!("<td align=\"{align}\">{cell}</td>"));
            }
            out.push_str("</tr>");
        }

        out.push_str("</table>\n");
        out
    }

    /// Render the likelihood-ratio test as a LaTeX table.
    ///
    /// Mirrors the column spec from `MixedModels.jl/test/mime.jl`:
    /// `{l | r | r | r | r | l}` with χ² rendered as `$\chi^2$`.
    pub fn to_latex(&self) -> String {
        let rows = self.latex_rows();
        let mut out = String::new();

        out.push_str("\\begin{tabular}\n");
        out.push_str("{l | r | r | r | r | l}\n");
        out.push_str(&rows[0].join(" & "));
        out.push_str(" \\\\\n\\hline\n");

        for row in rows.iter().skip(1) {
            out.push_str(&row.join(" & "));
            out.push_str(" \\\\\n");
        }

        out.push_str("\\end{tabular}\n");
        out
    }

    fn html_rows(&self) -> Vec<Vec<String>> {
        let mut rows = vec![vec![
            String::new(),
            "model-dof".to_string(),
            "-2 logLik".to_string(),
            "χ²".to_string(),
            "χ²-dof".to_string(),
            "P(&gt;χ²)".to_string(),
        ]];

        rows.push(vec![
            self.formulas[0].clone(),
            self.dof[0].to_string(),
            ((2.0 * self.loglik[0]).round() as i64).to_string(),
            String::new(),
            String::new(),
            String::new(),
        ]);

        for i in 1..self.formulas.len() {
            rows.push(vec![
                self.formulas[i].clone(),
                self.dof[i].to_string(),
                ((2.0 * self.loglik[i]).round() as i64).to_string(),
                (self.chisq[i - 1].round() as i64).to_string(),
                self.chisq_dof[i - 1].to_string(),
                format_pvalue(self.pvalues[i - 1]),
            ]);
        }

        rows
    }

    fn latex_rows(&self) -> Vec<Vec<String>> {
        let mut rows = vec![vec![
            String::new(),
            "model-dof".to_string(),
            "-2 logLik".to_string(),
            "$\\chi^2$".to_string(),
            "$\\chi^2$-dof".to_string(),
            "P(>$\\chi^2$)".to_string(),
        ]];

        rows.push(vec![
            self.formulas[0].clone(),
            self.dof[0].to_string(),
            ((2.0 * self.loglik[0]).round() as i64).to_string(),
            String::new(),
            String::new(),
            String::new(),
        ]);

        for i in 1..self.formulas.len() {
            rows.push(vec![
                self.formulas[i].clone(),
                self.dof[i].to_string(),
                ((2.0 * self.loglik[i]).round() as i64).to_string(),
                (self.chisq[i - 1].round() as i64).to_string(),
                self.chisq_dof[i - 1].to_string(),
                format_pvalue(self.pvalues[i - 1]),
            ]);
        }

        rows
    }
}

impl fmt::Display for LikelihoodRatioTest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Likelihood-ratio test: {} models fitted on {} observations",
            self.formulas.len(),
            self.nobs
        )?;
        writeln!(f, "Model Formulae")?;
        for (idx, formula) in self.formulas.iter().enumerate() {
            writeln!(f, "{}: {}", idx + 1, formula)?;
        }

        let rows = self.plaintext_rows();
        let widths = column_widths(&rows);
        let rule_len = widths.iter().sum::<usize>() + 2 * (widths.len() - 1);
        let rule = "─".repeat(rule_len);
        writeln!(f, "{rule}")?;

        for (row_idx, row) in rows.iter().enumerate() {
            for (col_idx, cell) in row.iter().enumerate() {
                if col_idx > 0 {
                    write!(f, "  ")?;
                }
                if col_idx == 0 {
                    write!(f, "{cell:<width$}", width = widths[col_idx])?;
                } else {
                    write!(f, "{cell:>width$}", width = widths[col_idx])?;
                }
            }
            if row_idx == 0 {
                writeln!(f)?;
                writeln!(f, "{rule}")?;
            } else if row_idx + 1 < rows.len() {
                writeln!(f)?;
            }
        }

        write!(f, "\n{rule}")
    }
}

impl LikelihoodRatioTest {
    fn plaintext_rows(&self) -> Vec<Vec<String>> {
        let mut rows = vec![vec![
            "".to_string(),
            "DoF".to_string(),
            "-2 logLik".to_string(),
            "χ²".to_string(),
            "χ²-dof".to_string(),
            "P(>χ²)".to_string(),
        ]];

        rows.push(vec![
            "[1]".to_string(),
            self.dof[0].to_string(),
            format!("{:.4}", self.deviance[0]),
            String::new(),
            String::new(),
            String::new(),
        ]);

        for i in 1..self.formulas.len() {
            rows.push(vec![
                format!("[{}]", i + 1),
                self.dof[i].to_string(),
                format!("{:.4}", self.deviance[i]),
                format!("{:.4}", self.chisq[i - 1]),
                self.chisq_dof[i - 1].to_string(),
                format_pvalue(self.pvalues[i - 1]),
            ]);
        }

        rows
    }
}

fn fixed_effect_space_is_nested(smaller: &DMatrix<f64>, larger: &DMatrix<f64>) -> bool {
    if smaller.nrows() != larger.nrows() {
        return false;
    }
    if smaller.ncols() == 0 {
        return true;
    }

    let larger_rank = stats_rank(larger).0;
    let mut combined = DMatrix::zeros(larger.nrows(), larger.ncols() + smaller.ncols());
    for row in 0..larger.nrows() {
        for col in 0..larger.ncols() {
            combined[(row, col)] = larger[(row, col)];
        }
        for col in 0..smaller.ncols() {
            combined[(row, larger.ncols() + col)] = smaller[(row, col)];
        }
    }

    stats_rank(&combined).0 == larger_rank
}

fn fixed_effect_spaces_are_equal(lhs: &DMatrix<f64>, rhs: &DMatrix<f64>) -> bool {
    if lhs.nrows() != rhs.nrows() {
        return false;
    }

    let lhs_rank = stats_rank(lhs).0;
    let rhs_rank = stats_rank(rhs).0;
    if lhs_rank != rhs_rank {
        return false;
    }

    let mut combined = DMatrix::zeros(lhs.nrows(), lhs.ncols() + rhs.ncols());
    for row in 0..lhs.nrows() {
        for col in 0..lhs.ncols() {
            combined[(row, col)] = lhs[(row, col)];
        }
        for col in 0..rhs.ncols() {
            combined[(row, lhs.ncols() + col)] = rhs[(row, col)];
        }
    }

    stats_rank(&combined).0 == lhs_rank
}

fn random_effect_terms_are_nested(
    smaller: &[RandomEffectTermInfo],
    larger: &[RandomEffectTermInfo],
) -> bool {
    smaller.iter().all(|small| {
        larger
            .iter()
            .any(|large| random_effect_term_is_nested(small, large))
    })
}

fn random_effect_term_is_nested(
    smaller: &RandomEffectTermInfo,
    larger: &RandomEffectTermInfo,
) -> bool {
    smaller.group == larger.group
        && smaller
            .columns
            .iter()
            .all(|column| larger.columns.iter().any(|candidate| candidate == column))
}

fn assess_model_pair(
    smaller: &dyn MixedModelFit,
    larger: &dyn MixedModelFit,
) -> ModelComparisonAssessment {
    let same_response =
        smaller.nobs() == larger.nobs() && vectors_equal(smaller.response(), larger.response());
    let same_family = smaller.family_kind() == larger.family_kind();
    let same_link = smaller.link_kind() == larger.link_kind();
    let same_fit_criterion = smaller.opt_summary().reml == larger.opt_summary().reml;
    let fixed_effects = fixed_effect_comparison(smaller.model_matrix(), larger.model_matrix());
    let random_effects = random_effect_comparison(
        &smaller.random_effect_terms(),
        &larger.random_effect_terms(),
    );
    let dof_increases = larger.dof() > smaller.dof();
    let same_model_space = fixed_effects == FixedEffectComparison::Same
        && random_effects == RandomEffectComparison::Same
        && smaller.dof() == larger.dof();

    let class = if !same_response {
        ModelComparisonClass::DifferentResponse
    } else if !same_family {
        ModelComparisonClass::DifferentFamily
    } else if !same_link {
        ModelComparisonClass::DifferentLink
    } else if !same_fit_criterion {
        ModelComparisonClass::MixedFitCriterion
    } else if same_model_space {
        ModelComparisonClass::SameModelSpace
    } else if fixed_effects == FixedEffectComparison::Same
        && random_effects == RandomEffectComparison::Same
        && dof_increases
    {
        ModelComparisonClass::SameFixedEffectsCovarianceDifference
    } else if fixed_effects == FixedEffectComparison::Nested
        && random_effects == RandomEffectComparison::Same
    {
        ModelComparisonClass::NestedFixedEffects
    } else if fixed_effects == FixedEffectComparison::Same
        && random_effects == RandomEffectComparison::Nested
    {
        ModelComparisonClass::NestedRandomEffects
    } else if fixed_effects == FixedEffectComparison::Nested
        && random_effects == RandomEffectComparison::Nested
    {
        ModelComparisonClass::NestedFixedAndRandomEffects
    } else if matches!(fixed_effects, FixedEffectComparison::ReverseNested)
        || matches!(random_effects, RandomEffectComparison::ReverseNested)
    {
        ModelComparisonClass::InvalidModelOrder
    } else if fixed_effects == FixedEffectComparison::NonNested {
        ModelComparisonClass::NonNestedFixedEffects
    } else if random_effects == RandomEffectComparison::NonNested {
        ModelComparisonClass::NonNestedRandomEffects
    } else {
        ModelComparisonClass::InvalidModelOrder
    };

    let reml = smaller.opt_summary().reml;
    let fixed_effects_match = fixed_effects == FixedEffectComparison::Same;
    let structurally_nested = matches!(
        class,
        ModelComparisonClass::NestedFixedEffects
            | ModelComparisonClass::NestedRandomEffects
            | ModelComparisonClass::NestedFixedAndRandomEffects
            | ModelComparisonClass::SameFixedEffectsCovarianceDifference
    );
    let lrt_available = structurally_nested
        && dof_increases
        && same_response
        && same_family
        && same_link
        && same_fit_criterion
        && (!reml || fixed_effects_match);
    let lrt_reason = if lrt_available {
        None
    } else {
        Some(lrt_unavailable_reason(
            class,
            reml,
            fixed_effects_match,
            dof_increases,
        ))
    };

    let ml_refit_required = same_response
        && same_family
        && same_link
        && same_fit_criterion
        && reml
        && fixed_effects != FixedEffectComparison::Same;
    let ml_refit_reason = ml_refit_required.then(|| {
        "models differ in fixed effects but were fitted by REML; refit with ML for fixed-effect likelihood comparisons".to_string()
    });

    let information_criteria_available = same_response
        && same_family
        && same_link
        && same_fit_criterion
        && (!reml || fixed_effects_match);
    let information_criteria_reason = if information_criteria_available {
        None
    } else {
        Some(information_criteria_unavailable_reason(
            same_response,
            same_family,
            same_link,
            same_fit_criterion,
            reml,
            fixed_effects_match,
        ))
    };

    let valid_alternatives = comparison_alternatives(
        class,
        lrt_available,
        information_criteria_available,
        ml_refit_required,
        reml,
        fixed_effects,
        random_effects,
    );

    ModelComparisonAssessment {
        class,
        fixed_effects,
        random_effects,
        lrt_available,
        lrt_reason,
        information_criteria_available,
        information_criteria_reason,
        ml_refit_required,
        ml_refit_reason,
        valid_alternatives,
    }
}

fn fixed_effect_comparison(smaller: &DMatrix<f64>, larger: &DMatrix<f64>) -> FixedEffectComparison {
    if fixed_effect_spaces_are_equal(smaller, larger) {
        FixedEffectComparison::Same
    } else if fixed_effect_space_is_nested(smaller, larger) {
        FixedEffectComparison::Nested
    } else if fixed_effect_space_is_nested(larger, smaller) {
        FixedEffectComparison::ReverseNested
    } else {
        FixedEffectComparison::NonNested
    }
}

fn random_effect_comparison(
    smaller: &[RandomEffectTermInfo],
    larger: &[RandomEffectTermInfo],
) -> RandomEffectComparison {
    if random_effect_terms_are_equal(smaller, larger) {
        RandomEffectComparison::Same
    } else if random_effect_terms_are_nested(smaller, larger) {
        RandomEffectComparison::Nested
    } else if random_effect_terms_are_nested(larger, smaller) {
        RandomEffectComparison::ReverseNested
    } else {
        RandomEffectComparison::NonNested
    }
}

fn random_effect_terms_are_equal(
    lhs: &[RandomEffectTermInfo],
    rhs: &[RandomEffectTermInfo],
) -> bool {
    lhs.len() == rhs.len()
        && lhs.iter().all(|left| {
            rhs.iter()
                .any(|right| random_effect_term_matches(left, right))
        })
}

fn random_effect_term_matches(lhs: &RandomEffectTermInfo, rhs: &RandomEffectTermInfo) -> bool {
    lhs.group == rhs.group
        && lhs.columns.len() == rhs.columns.len()
        && lhs
            .columns
            .iter()
            .all(|column| rhs.columns.iter().any(|candidate| candidate == column))
}

fn lrt_unavailable_reason(
    class: ModelComparisonClass,
    reml: bool,
    fixed_effects_match: bool,
    dof_increases: bool,
) -> String {
    match class {
        ModelComparisonClass::DifferentResponse => {
            "models were not fitted to the same response values".to_string()
        }
        ModelComparisonClass::DifferentFamily => {
            "models have different conditional response families".to_string()
        }
        ModelComparisonClass::DifferentLink => "models have different link functions".to_string(),
        ModelComparisonClass::MixedFitCriterion => {
            "models mix REML and ML fit criteria; refit with a common criterion".to_string()
        }
        ModelComparisonClass::SameModelSpace => "models describe the same model space".to_string(),
        ModelComparisonClass::InvalidModelOrder if !dof_increases => {
            "larger model must have more degrees of freedom than the smaller model".to_string()
        }
        ModelComparisonClass::InvalidModelOrder => {
            "models appear nested only in the reverse order".to_string()
        }
        ModelComparisonClass::NonNestedFixedEffects => {
            "fixed-effect column spaces are not nested".to_string()
        }
        ModelComparisonClass::NonNestedRandomEffects => {
            "random-effect term structures are not nested".to_string()
        }
        _ if reml && !fixed_effects_match => {
            "REML likelihood-ratio tests require identical fixed effects; refit with ML".to_string()
        }
        _ if !dof_increases => {
            "larger model must have more degrees of freedom than the smaller model".to_string()
        }
        _ => "ordinary likelihood-ratio test is unavailable for this comparison".to_string(),
    }
}

fn information_criteria_unavailable_reason(
    same_response: bool,
    same_family: bool,
    same_link: bool,
    same_fit_criterion: bool,
    reml: bool,
    fixed_effects_match: bool,
) -> String {
    if !same_response {
        "information criteria require models fit to the same response values".to_string()
    } else if !same_family {
        "information criteria require a common conditional response family".to_string()
    } else if !same_link {
        "information criteria require a common link function".to_string()
    } else if !same_fit_criterion {
        "information criteria require a common REML/ML fit criterion".to_string()
    } else if reml && !fixed_effects_match {
        "REML information criteria are not comparable across different fixed effects".to_string()
    } else {
        "information criteria are unavailable for this comparison".to_string()
    }
}

fn comparison_alternatives(
    class: ModelComparisonClass,
    lrt_available: bool,
    information_criteria_available: bool,
    ml_refit_required: bool,
    reml: bool,
    fixed_effects: FixedEffectComparison,
    random_effects: RandomEffectComparison,
) -> Vec<ModelComparisonAlternative> {
    let mut alternatives = Vec::new();
    if lrt_available {
        alternatives.push(if reml {
            ModelComparisonAlternative::RemlLikelihoodRatio
        } else {
            ModelComparisonAlternative::MlRefitLikelihoodRatio
        });
    }
    if ml_refit_required {
        alternatives.push(ModelComparisonAlternative::MlRefitLikelihoodRatio);
        alternatives.push(ModelComparisonAlternative::RefitWithCommonCriterion);
    }
    if class == ModelComparisonClass::MixedFitCriterion {
        alternatives.push(ModelComparisonAlternative::RefitWithCommonCriterion);
    }
    if information_criteria_available {
        alternatives.push(ModelComparisonAlternative::InformationCriteria);
    }
    if matches!(
        fixed_effects,
        FixedEffectComparison::Nested | FixedEffectComparison::NonNested
    ) {
        alternatives.push(ModelComparisonAlternative::FixedEffectContrastTest);
    }
    if matches!(
        random_effects,
        RandomEffectComparison::Nested | RandomEffectComparison::NonNested
    ) || class == ModelComparisonClass::SameFixedEffectsCovarianceDifference
    {
        alternatives.push(ModelComparisonAlternative::ParametricBootstrap);
    }
    if class == ModelComparisonClass::InvalidModelOrder {
        alternatives.push(ModelComparisonAlternative::ReorderModels);
    }
    if matches!(
        class,
        ModelComparisonClass::NonNestedFixedEffects | ModelComparisonClass::NonNestedRandomEffects
    ) {
        alternatives.push(ModelComparisonAlternative::CrossValidation);
    }
    alternatives.dedup();
    alternatives
}

struct LrtValues {
    chisq: f64,
    chisq_dof: usize,
    pvalue: f64,
    loglik_within_optimizer_tol: bool,
}

fn validate_model_comparison_options(
    assessments: &[ModelComparisonAssessment],
    options: ModelComparisonOptions,
) -> Result<(), String> {
    for assessment in assessments {
        if options.refit_policy == ModelComparisonRefitPolicy::Error && assessment.ml_refit_required
        {
            return Err(assessment
                .ml_refit_reason
                .clone()
                .unwrap_or_else(|| "comparison requires ML refits".to_string()));
        }

        match options.method {
            ModelComparisonMethod::Auto => {}
            ModelComparisonMethod::LikelihoodRatio => {
                if !assessment.lrt_available
                    && (!assessment.ml_refit_required
                        || options.refit_policy == ModelComparisonRefitPolicy::Error)
                {
                    return Err(assessment.lrt_reason.clone().unwrap_or_else(|| {
                        "ordinary likelihood-ratio test is unavailable for this comparison"
                            .to_string()
                    }));
                }
            }
            ModelComparisonMethod::InformationCriteria => {
                if !assessment.information_criteria_available {
                    return Err(assessment
                        .information_criteria_reason
                        .clone()
                        .unwrap_or_else(|| {
                            "information criteria are unavailable for this comparison".to_string()
                        }));
                }
            }
        }
    }

    Ok(())
}

fn likelihood_ratio_values(
    smaller: &dyn MixedModelFit,
    larger: &dyn MixedModelFit,
) -> Result<LrtValues, String> {
    let loglik_diff = larger.loglikelihood() - smaller.loglikelihood();
    if loglik_diff < -LOG_LIK_TOL {
        return Err(
            "larger model has lower log-likelihood; likelihood-ratio columns omitted".to_string(),
        );
    }

    let loglik_within_optimizer_tol = loglik_diff <= 0.0;
    let chisq = if loglik_within_optimizer_tol {
        0.0
    } else {
        2.0 * loglik_diff
    };
    let chisq_dof = larger.dof() - smaller.dof();
    use statrs::distribution::{ChiSquared, ContinuousCDF};
    let dist = ChiSquared::new(chisq_dof as f64).unwrap();
    let pvalue = 1.0 - dist.cdf(chisq);

    Ok(LrtValues {
        chisq,
        chisq_dof,
        pvalue,
        loglik_within_optimizer_tol,
    })
}

fn self_liang_one_parameter_mixture() -> Vec<BoundaryLrtMixtureComponent> {
    vec![
        BoundaryLrtMixtureComponent {
            weight: 0.5,
            chisq_df: None,
            point_mass_at: Some(0.0),
        },
        BoundaryLrtMixtureComponent {
            weight: 0.5,
            chisq_df: Some(1),
            point_mass_at: None,
        },
    ]
}

fn self_liang_one_parameter_pvalue(chisq: f64) -> f64 {
    if chisq <= 0.0 {
        return 1.0;
    }
    use statrs::distribution::{ChiSquared, ContinuousCDF};
    let dist = ChiSquared::new(1.0).unwrap();
    0.5 * (1.0 - dist.cdf(chisq))
}

fn comparison_row_reason(
    assessment: Option<&ModelComparisonAssessment>,
    method: ModelComparisonMethod,
    lrt_reason: Option<String>,
) -> Option<String> {
    if let Some(reason) = lrt_reason {
        return Some(reason);
    }

    let assessment = assessment?;
    if assessment.ml_refit_required {
        return assessment.ml_refit_reason.clone();
    }
    if method == ModelComparisonMethod::InformationCriteria {
        return Some(
            "information-criteria comparison requested; likelihood-ratio columns omitted"
                .to_string(),
        );
    }
    if !assessment.lrt_available {
        return assessment.lrt_reason.clone();
    }
    None
}

fn comparison_row_reason_code(
    assessment: Option<&ModelComparisonAssessment>,
    method: ModelComparisonMethod,
    reason: Option<&str>,
) -> Option<ModelComparisonReasonCode> {
    let assessment = assessment?;
    if reason == Some("larger model has lower log-likelihood; likelihood-ratio columns omitted") {
        return Some(ModelComparisonReasonCode::LowerLoglikelihoodLrtInvalid);
    }
    if assessment.ml_refit_required {
        return Some(ModelComparisonReasonCode::MlRefitRequired);
    }
    if method == ModelComparisonMethod::InformationCriteria {
        return Some(ModelComparisonReasonCode::InformationCriteriaRequested);
    }
    if assessment.lrt_available && reason.is_none() {
        return None;
    }

    Some(match assessment.class {
        ModelComparisonClass::NonNestedFixedEffects
        | ModelComparisonClass::NonNestedRandomEffects => {
            ModelComparisonReasonCode::NonNestedModelsLrtInvalid
        }
        ModelComparisonClass::SameModelSpace => ModelComparisonReasonCode::SameModelSpaceLrtInvalid,
        ModelComparisonClass::DifferentResponse => ModelComparisonReasonCode::DifferentResponse,
        ModelComparisonClass::DifferentFamily => ModelComparisonReasonCode::DifferentFamily,
        ModelComparisonClass::DifferentLink => ModelComparisonReasonCode::DifferentLink,
        ModelComparisonClass::MixedFitCriterion => ModelComparisonReasonCode::MixedFitCriterion,
        ModelComparisonClass::InvalidModelOrder => ModelComparisonReasonCode::InvalidModelOrder,
        _ => ModelComparisonReasonCode::LrtUnavailable,
    })
}

fn finite_min(values: &[f64]) -> Option<f64> {
    values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .min_by(|left, right| left.total_cmp(right))
}

fn delta_from_min(value: f64, minimum: Option<f64>) -> f64 {
    match minimum {
        Some(minimum) if value.is_finite() => value - minimum,
        _ => f64::NAN,
    }
}

fn vectors_equal(lhs: &DVector<f64>, rhs: &DVector<f64>) -> bool {
    lhs.len() == rhs.len()
        && lhs
            .iter()
            .zip(rhs.iter())
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

fn column_widths(rows: &[Vec<String>]) -> Vec<usize> {
    (0..rows[0].len())
        .map(|col| rows.iter().map(|row| row[col].len()).max().unwrap_or(0))
        .collect()
}

fn escape_markdown_pipes(label: &str) -> String {
    label.replace('|', "\\|")
}

fn format_pvalue(pvalue: f64) -> String {
    if !pvalue.is_finite() {
        return String::new();
    }
    if pvalue <= 0.0 {
        return "<1e-99".to_string();
    }
    if pvalue < 1.0e-4 {
        let exponent = (-pvalue.log10()).floor().max(1.0) as i32;
        return format!("<1e-{exponent:02}");
    }
    format!("{pvalue:.4}")
}

/// Result of a parametric-bootstrap likelihood-ratio test.
///
/// The bootstrap p-value is boundary-robust: it makes no chi-square (or
/// 50:50-mixture) assumption, so it is valid when the added parameter sits
/// on the boundary of its space — exactly the case the analytic
/// [`BoundaryLikelihoodRatioTest`] refuses for more than one added
/// variance/covariance parameter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParametricBootstrapLrt {
    /// Stable schema name.
    pub schema_name: String,
    /// Stable schema version.
    pub schema_version: String,
    /// Observed statistic `2·(ℓ_larger − ℓ_smaller)` (clamped at 0 within
    /// optimizer tolerance, matching the ordinary LRT convention).
    pub observed_statistic: f64,
    /// Nominal `larger.dof − smaller.dof`. Informational only — the
    /// p-value does **not** use a chi-square reference with this dof.
    pub chisq_dof: usize,
    /// Number of bootstrap simulations requested by the caller.
    pub n_sim_requested: usize,
    /// Replicates where the null simulation refit succeeded for *both*
    /// the smaller and larger models.
    pub n_sim_completed: usize,
    /// Replicates discarded because a refit failed numerically.
    pub n_refit_failures: usize,
    /// `(1 + #{T* ≥ T_obs}) / (n_sim_completed + 1)`.
    pub p_value: f64,
}

/// Parametric-bootstrap likelihood-ratio test of a nested LMM pair.
///
/// Simulates `n_sim` responses from the fitted **null** (`smaller`) model,
/// refits both models to each, and compares the resulting LR statistics to
/// the observed one. This is the boundary-robust route the analytic
/// [`BoundaryLikelihoodRatioTest`] points to when more than one added
/// variance/covariance parameter prevents a certified mixture.
///
/// Both models must be ML-fitted to the *same* observations and properly
/// nested (`smaller` ⊂ `larger`, `larger.dof > smaller.dof`). The caller
/// owns RNG seeding for reproducibility.
pub fn parametric_bootstrap_lrt<R: rand::Rng>(
    rng: &mut R,
    n_sim: usize,
    smaller: &LinearMixedModel,
    larger: &LinearMixedModel,
) -> crate::error::Result<ParametricBootstrapLrt> {
    use crate::error::MixedModelError;

    if n_sim == 0 {
        return Err(MixedModelError::InvalidArgument(
            "parametric-bootstrap LRT requires n_sim >= 1".to_string(),
        ));
    }
    if smaller.nobs() != larger.nobs() {
        return Err(MixedModelError::InvalidArgument(
            "smaller and larger models must be fit to the same number of observations".to_string(),
        ));
    }
    if larger.dof() <= smaller.dof() {
        return Err(MixedModelError::InvalidArgument(
            "larger model must have strictly more parameters than smaller (proper nesting)"
                .to_string(),
        ));
    }

    let observed = likelihood_ratio_values(smaller, larger).map_err(|reason| {
        MixedModelError::InvalidArgument(format!(
            "observed likelihood-ratio statistic is unavailable: {reason}"
        ))
    })?;
    let t_obs = observed.chisq;

    let mut completed = 0usize;
    let mut failures = 0usize;
    let mut at_least_as_extreme = 0usize;
    let mut last_progress = 0usize;

    for replicate in 0..n_sim {
        if let Some(callback) = &smaller.progress_callback {
            callback.report_if_due(
                crate::model::linear::FitProgressPhase::Bootstrap,
                replicate + 1,
                Some(n_sim),
                &mut last_progress,
            )?;
        }
        let y_star = smaller.simulate(rng);
        let mut null_fit = smaller.clone();
        let mut alt_fit = larger.clone();
        // Replicates only contribute a likelihood-ratio statistic, so skip
        // the optimizer certificate's finite-difference derivative
        // diagnostics on these internal refits.
        null_fit.suppress_derivative_diagnostics = true;
        alt_fit.suppress_derivative_diagnostics = true;
        match (
            null_fit.refit(y_star.as_slice()),
            alt_fit.refit(y_star.as_slice()),
        ) {
            (Ok(()), Ok(())) => {
                let diff = alt_fit.loglikelihood() - null_fit.loglikelihood();
                let t_star = if diff <= 0.0 { 0.0 } else { 2.0 * diff };
                if t_star >= t_obs {
                    at_least_as_extreme += 1;
                }
                completed += 1;
            }
            (Err(error @ MixedModelError::Interrupted(_)), _)
            | (_, Err(error @ MixedModelError::Interrupted(_))) => return Err(error),
            _ => failures += 1,
        }
    }

    if completed == 0 {
        return Err(MixedModelError::InvalidArgument(
            "every parametric-bootstrap refit failed; no bootstrap p-value can be formed"
                .to_string(),
        ));
    }

    let p_value = (1.0 + at_least_as_extreme as f64) / (completed as f64 + 1.0);

    Ok(ParametricBootstrapLrt {
        schema_name: PARAMETRIC_BOOTSTRAP_LRT_SCHEMA.to_string(),
        schema_version: PARAMETRIC_BOOTSTRAP_LRT_SCHEMA_VERSION.to_string(),
        observed_statistic: t_obs,
        chisq_dof: observed.chisq_dof,
        n_sim_requested: n_sim,
        n_sim_completed: completed,
        n_refit_failures: failures,
        p_value,
    })
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::{DMatrix, DVector};

    use super::*;
    use crate::types::OptSummary;

    #[derive(Clone)]
    struct DummyFit {
        nobs: usize,
        dof: usize,
        loglik: f64,
        formula: Option<String>,
        response: DVector<f64>,
        model_matrix: DMatrix<f64>,
        optsum: OptSummary,
        family: Option<crate::model::traits::Family>,
        link: Option<crate::model::traits::LinkFunction>,
        random_terms: Vec<RandomEffectTermInfo>,
    }

    impl DummyFit {
        fn new(nobs: usize, dof: usize, loglik: f64, formula: Option<&str>) -> Self {
            Self {
                nobs,
                dof,
                loglik,
                formula: formula.map(str::to_string),
                response: DVector::zeros(nobs),
                model_matrix: DMatrix::zeros(nobs, 0),
                optsum: OptSummary::new(Vec::new()),
                family: None,
                link: None,
                random_terms: Vec::new(),
            }
        }

        fn with_reml(mut self, reml: bool) -> Self {
            self.optsum.reml = reml;
            self
        }

        fn with_family(
            mut self,
            family: crate::model::traits::Family,
            link: crate::model::traits::LinkFunction,
        ) -> Self {
            self.family = Some(family);
            self.link = Some(link);
            self
        }

        fn with_model_matrix(mut self, model_matrix: DMatrix<f64>) -> Self {
            self.model_matrix = model_matrix;
            self
        }

        fn with_response(mut self, response: DVector<f64>) -> Self {
            self.nobs = response.len();
            self.response = response;
            self
        }

        fn with_random_terms(mut self, random_terms: Vec<RandomEffectTermInfo>) -> Self {
            self.random_terms = random_terms;
            self
        }
    }

    impl MixedModelFit for DummyFit {
        fn nobs(&self) -> usize {
            self.nobs
        }

        fn dof(&self) -> usize {
            self.dof
        }

        fn coef(&self) -> DVector<f64> {
            DVector::zeros(0)
        }

        fn fixef(&self) -> DVector<f64> {
            DVector::zeros(0)
        }

        fn coef_names(&self) -> Vec<String> {
            Vec::new()
        }

        fn vcov(&self) -> DMatrix<f64> {
            DMatrix::zeros(0, 0)
        }

        fn stderror(&self) -> DVector<f64> {
            DVector::zeros(0)
        }

        fn fitted(&self) -> DVector<f64> {
            DVector::zeros(self.nobs)
        }

        fn residuals(&self) -> DVector<f64> {
            DVector::zeros(self.nobs)
        }

        fn response(&self) -> &DVector<f64> {
            &self.response
        }

        fn model_matrix(&self) -> &DMatrix<f64> {
            &self.model_matrix
        }

        fn objective(&self) -> f64 {
            -2.0 * self.loglik
        }

        fn loglikelihood(&self) -> f64 {
            self.loglik
        }

        fn formula_label(&self) -> Option<String> {
            self.formula.clone()
        }

        fn is_fitted(&self) -> bool {
            true
        }

        fn is_singular(&self) -> bool {
            false
        }

        fn opt_summary(&self) -> &OptSummary {
            &self.optsum
        }

        fn theta(&self) -> Vec<f64> {
            Vec::new()
        }

        fn dispersion(&self, _sqr: bool) -> f64 {
            1.0
        }

        fn ranef(&self) -> Vec<DMatrix<f64>> {
            Vec::new()
        }

        fn random_effect_terms(&self) -> Vec<RandomEffectTermInfo> {
            self.random_terms.clone()
        }

        fn family_kind(&self) -> Option<crate::model::traits::Family> {
            self.family
        }

        fn link_kind(&self) -> Option<crate::model::traits::LinkFunction> {
            self.link
        }
    }

    fn intercept_x() -> DMatrix<f64> {
        DMatrix::from_row_slice(
            4,
            2,
            &[
                1.0, 0.0, //
                1.0, 1.0, //
                1.0, 2.0, //
                1.0, 3.0,
            ],
        )
    }

    fn intercept_z() -> DMatrix<f64> {
        DMatrix::from_row_slice(
            4,
            2,
            &[
                1.0, 0.0, //
                1.0, 1.0, //
                1.0, 0.0, //
                1.0, 1.0,
            ],
        )
    }

    fn subject_intercept() -> RandomEffectTermInfo {
        RandomEffectTermInfo {
            group: "subject".to_string(),
            columns: vec!["(Intercept)".to_string()],
        }
    }

    fn subject_intercept_slope() -> RandomEffectTermInfo {
        RandomEffectTermInfo {
            group: "subject".to_string(),
            columns: vec!["(Intercept)".to_string(), "x".to_string()],
        }
    }

    fn item_intercept() -> RandomEffectTermInfo {
        RandomEffectTermInfo {
            group: "item".to_string(),
            columns: vec!["(Intercept)".to_string()],
        }
    }

    fn lrt_refusal_from_assessment(
        smaller: &dyn MixedModelFit,
        larger: &dyn MixedModelFit,
    ) -> String {
        let assessment = ModelComparisonAssessment::assess(smaller, larger);
        assert!(!assessment.lrt_available);

        let err = LikelihoodRatioTest::test(&[smaller, larger]).unwrap_err();

        assert_eq!(Some(err.as_str()), assessment.lrt_reason.as_deref());
        err
    }

    #[test]
    fn test_model_comparison_table_reports_valid_lrt_columns_for_nested_models() {
        let small_x = DMatrix::from_element(4, 1, 1.0);
        let large_x = intercept_x();
        let m0 = DummyFit::new(4, 2, -10.0, Some("y ~ 1"))
            .with_reml(false)
            .with_model_matrix(small_x);
        let m1 = DummyFit::new(4, 3, -7.0, Some("y ~ 1 + x"))
            .with_reml(false)
            .with_model_matrix(large_x);

        let table = ModelComparisonTable::compare(&[&m0, &m1]).unwrap();

        assert_eq!(table.method, ModelComparisonMethod::Auto);
        assert!(!table.refit_performed);
        assert_eq!(table.rows.len(), 2);
        assert_eq!(table.rows[0].label, "y ~ 1");
        assert_eq!(table.rows[1].label, "y ~ 1 + x");
        assert_eq!(
            table.rows[1].comparison_class,
            Some(ModelComparisonClass::NestedFixedEffects)
        );
        assert!(table.rows[1].lrt_available);
        assert_eq!(table.rows[1].chisq_dof, Some(1));
        assert_relative_eq!(table.rows[1].chisq.unwrap(), 6.0);
        assert!(table.rows[1].pvalue.unwrap() < 0.05);
        assert!(table.rows[1].reason.is_none());
    }

    #[test]
    fn test_model_comparison_table_keeps_ic_rows_for_non_nested_models() {
        let m0 = DummyFit::new(4, 3, -10.0, Some("y ~ 1 + x"))
            .with_reml(false)
            .with_model_matrix(intercept_x());
        let m1 = DummyFit::new(4, 4, -8.0, Some("y ~ 1 + z"))
            .with_reml(false)
            .with_model_matrix(intercept_z());

        let table = ModelComparisonTable::compare(&[&m0, &m1]).unwrap();

        assert_eq!(table.rows.len(), 2);
        assert_relative_eq!(table.rows[0].aic, 26.0);
        assert_relative_eq!(table.rows[1].aic, 24.0);
        assert_relative_eq!(table.rows[0].delta_aic, 2.0);
        assert_relative_eq!(table.rows[1].delta_aic, 0.0);
        assert!(!table.rows[1].lrt_available);
        assert_eq!(table.rows[1].chisq, None);
        assert_eq!(table.rows[1].chisq_dof, None);
        assert_eq!(table.rows[1].pvalue, None);
        assert_eq!(
            table.rows[1].reason.as_deref(),
            Some("fixed-effect column spaces are not nested")
        );
        assert_eq!(
            table.rows[1].reason_code.as_deref(),
            Some("non_nested_models_lrt_invalid")
        );
        assert!(table.rows[1].information_criteria_available);
    }

    #[test]
    fn test_model_comparison_table_reports_ml_refit_required_without_refitting() {
        let small_x = DMatrix::from_element(4, 1, 1.0);
        let large_x = intercept_x();
        let m0 = DummyFit::new(4, 2, -10.0, Some("y ~ 1"))
            .with_reml(true)
            .with_model_matrix(small_x);
        let m1 = DummyFit::new(4, 3, -7.0, Some("y ~ 1 + x"))
            .with_reml(true)
            .with_model_matrix(large_x);

        let table = ModelComparisonTable::compare(&[&m0, &m1]).unwrap();

        assert!(!table.refit_performed);
        assert!(!table.rows[1].lrt_available);
        assert!(table.rows[1].requires_ml_refit);
        assert_eq!(
            table.rows[1].reason.as_deref(),
            Some(
                "models differ in fixed effects but were fitted by REML; refit with ML for fixed-effect likelihood comparisons"
            )
        );
        assert_eq!(
            table.rows[1].reason_code.as_deref(),
            Some("ml_refit_required")
        );

        let err = ModelComparisonTable::compare_with_options(
            &[&m0, &m1],
            ModelComparisonOptions {
                method: ModelComparisonMethod::Auto,
                refit_policy: ModelComparisonRefitPolicy::Error,
            },
        )
        .unwrap_err();
        assert_eq!(
            err,
            "models differ in fixed effects but were fitted by REML; refit with ML for fixed-effect likelihood comparisons"
        );
    }

    #[test]
    fn test_model_comparison_table_can_omit_lrt_when_ic_requested() {
        let small_x = DMatrix::from_element(4, 1, 1.0);
        let large_x = intercept_x();
        let m0 = DummyFit::new(4, 2, -10.0, Some("y ~ 1"))
            .with_reml(false)
            .with_model_matrix(small_x);
        let m1 = DummyFit::new(4, 3, -7.0, Some("y ~ 1 + x"))
            .with_reml(false)
            .with_model_matrix(large_x);

        let table = ModelComparisonTable::compare_with_options(
            &[&m0, &m1],
            ModelComparisonOptions {
                method: ModelComparisonMethod::InformationCriteria,
                refit_policy: ModelComparisonRefitPolicy::Never,
            },
        )
        .unwrap();

        assert!(!table.rows[1].lrt_available);
        assert_eq!(table.rows[1].chisq, None);
        assert_eq!(
            table.rows[1].reason.as_deref(),
            Some("information-criteria comparison requested; likelihood-ratio columns omitted")
        );
        assert_eq!(
            table.rows[1].reason_code.as_deref(),
            Some("information_criteria_requested")
        );
    }

    #[test]
    fn test_model_comparison_table_marks_incompatible_pairs() {
        use crate::model::traits::{Family, LinkFunction};

        let m0 = DummyFit::new(4, 2, -10.0, Some("y ~ 1"))
            .with_response(DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]));
        let m_different_response = DummyFit::new(4, 3, -9.0, Some("z ~ 1 + x"))
            .with_response(DVector::from_vec(vec![1.0, 2.0, 3.0, 5.0]));

        let table = ModelComparisonTable::compare(&[&m0, &m_different_response]).unwrap();
        assert!(!table.rows[1].lrt_available);
        assert!(!table.rows[1].information_criteria_available);
        assert_eq!(table.rows[1].pvalue, None);
        assert_eq!(
            table.rows[1].reason_code.as_deref(),
            Some("different_response")
        );

        let err = ModelComparisonTable::compare_with_options(
            &[&m0, &m_different_response],
            ModelComparisonOptions {
                method: ModelComparisonMethod::InformationCriteria,
                refit_policy: ModelComparisonRefitPolicy::Never,
            },
        )
        .unwrap_err();
        assert_eq!(
            err,
            "information criteria require models fit to the same response values"
        );

        let bernoulli = DummyFit::new(4, 2, -10.0, Some("bernoulli"))
            .with_family(Family::Bernoulli, LinkFunction::Logit);
        let poisson = DummyFit::new(4, 3, -9.0, Some("poisson"))
            .with_family(Family::Poisson, LinkFunction::Log);
        let table = ModelComparisonTable::compare(&[&bernoulli, &poisson]).unwrap();
        assert_eq!(
            table.rows[1].reason_code.as_deref(),
            Some("different_family")
        );
        assert!(!table.rows[1].information_criteria_available);
        assert_eq!(table.rows[1].pvalue, None);

        let logit = DummyFit::new(4, 2, -10.0, Some("logit"))
            .with_family(Family::Bernoulli, LinkFunction::Logit);
        let probit = DummyFit::new(4, 3, -9.0, Some("probit"))
            .with_family(Family::Bernoulli, LinkFunction::Probit);
        let table = ModelComparisonTable::compare(&[&logit, &probit]).unwrap();
        assert_eq!(table.rows[1].reason_code.as_deref(), Some("different_link"));
        assert!(!table.rows[1].information_criteria_available);
        assert_eq!(table.rows[1].pvalue, None);
    }

    #[test]
    fn test_model_comparison_table_lrt_method_rejects_non_nested_fixed_effects() {
        let m0 = DummyFit::new(4, 3, -10.0, Some("y ~ 1 + x"))
            .with_reml(false)
            .with_model_matrix(intercept_x());
        let m1 = DummyFit::new(4, 4, -8.0, Some("y ~ 1 + z"))
            .with_reml(false)
            .with_model_matrix(intercept_z());

        let err = ModelComparisonTable::compare_with_options(
            &[&m0, &m1],
            ModelComparisonOptions {
                method: ModelComparisonMethod::LikelihoodRatio,
                refit_policy: ModelComparisonRefitPolicy::Never,
            },
        )
        .unwrap_err();

        assert_eq!(err, "fixed-effect column spaces are not nested");
    }

    #[test]
    fn test_model_comparison_assessment_classifies_same_model_space() {
        let x = intercept_x();
        let m0 = DummyFit::new(4, 3, -10.0, Some("m0")).with_model_matrix(x.clone());
        let m1 = DummyFit::new(4, 3, -10.0, Some("m1")).with_model_matrix(x);

        let assessment = ModelComparisonAssessment::assess(&m0, &m1);

        assert_eq!(assessment.class, ModelComparisonClass::SameModelSpace);
        assert_eq!(assessment.fixed_effects, FixedEffectComparison::Same);
        assert_eq!(assessment.random_effects, RandomEffectComparison::Same);
        assert!(!assessment.lrt_available);
        assert_eq!(
            assessment.lrt_reason.as_deref(),
            Some("models describe the same model space")
        );
        assert!(assessment.information_criteria_available);
        assert!(!assessment.ml_refit_required);
    }

    #[test]
    fn test_model_comparison_assessment_classifies_nested_fixed_effects() {
        let intercept = DMatrix::from_element(4, 1, 1.0);
        let m0 = DummyFit::new(4, 2, -10.0, Some("m0"))
            .with_reml(false)
            .with_model_matrix(intercept);
        let m1 = DummyFit::new(4, 3, -9.0, Some("m1"))
            .with_reml(false)
            .with_model_matrix(intercept_x());

        let assessment = ModelComparisonAssessment::assess(&m0, &m1);

        assert_eq!(assessment.class, ModelComparisonClass::NestedFixedEffects);
        assert_eq!(assessment.fixed_effects, FixedEffectComparison::Nested);
        assert_eq!(assessment.random_effects, RandomEffectComparison::Same);
        assert!(assessment.lrt_available);
        assert!(assessment.information_criteria_available);
        assert!(assessment
            .valid_alternatives
            .contains(&ModelComparisonAlternative::FixedEffectContrastTest));
    }

    #[test]
    fn test_model_comparison_assessment_classifies_nested_random_effects() {
        let x = intercept_x();
        let m0 = DummyFit::new(4, 3, -10.0, Some("m0"))
            .with_model_matrix(x.clone())
            .with_random_terms(vec![subject_intercept()]);
        let m1 = DummyFit::new(4, 5, -9.0, Some("m1"))
            .with_model_matrix(x)
            .with_random_terms(vec![subject_intercept_slope()]);

        let assessment = ModelComparisonAssessment::assess(&m0, &m1);

        assert_eq!(assessment.class, ModelComparisonClass::NestedRandomEffects);
        assert_eq!(assessment.fixed_effects, FixedEffectComparison::Same);
        assert_eq!(assessment.random_effects, RandomEffectComparison::Nested);
        assert!(assessment.lrt_available);
        assert!(assessment
            .valid_alternatives
            .contains(&ModelComparisonAlternative::ParametricBootstrap));
    }

    #[test]
    fn test_model_comparison_assessment_classifies_covariance_difference() {
        let x = intercept_x();
        let random = subject_intercept_slope();
        let m0 = DummyFit::new(4, 4, -10.0, Some("diagonal"))
            .with_model_matrix(x.clone())
            .with_random_terms(vec![random.clone()]);
        let m1 = DummyFit::new(4, 5, -9.0, Some("full"))
            .with_model_matrix(x)
            .with_random_terms(vec![random]);

        let assessment = ModelComparisonAssessment::assess(&m0, &m1);

        assert_eq!(
            assessment.class,
            ModelComparisonClass::SameFixedEffectsCovarianceDifference
        );
        assert_eq!(assessment.fixed_effects, FixedEffectComparison::Same);
        assert_eq!(assessment.random_effects, RandomEffectComparison::Same);
        assert!(assessment.lrt_available);
        assert!(assessment
            .valid_alternatives
            .contains(&ModelComparisonAlternative::ParametricBootstrap));
    }

    #[test]
    fn test_model_comparison_assessment_classifies_non_nested_fixed_effects() {
        let m0 = DummyFit::new(4, 3, -10.0, Some("m0"))
            .with_reml(false)
            .with_model_matrix(intercept_x());
        let m1 = DummyFit::new(4, 3, -9.0, Some("m1"))
            .with_reml(false)
            .with_model_matrix(intercept_z());

        let assessment = ModelComparisonAssessment::assess(&m0, &m1);

        assert_eq!(
            assessment.class,
            ModelComparisonClass::NonNestedFixedEffects
        );
        assert_eq!(assessment.fixed_effects, FixedEffectComparison::NonNested);
        assert!(!assessment.lrt_available);
        assert_eq!(
            assessment.lrt_reason.as_deref(),
            Some("fixed-effect column spaces are not nested")
        );
        assert!(assessment.information_criteria_available);
        assert!(assessment
            .valid_alternatives
            .contains(&ModelComparisonAlternative::CrossValidation));
    }

    #[test]
    fn test_model_comparison_assessment_classifies_non_nested_random_effects() {
        let x = intercept_x();
        let m0 = DummyFit::new(4, 4, -10.0, Some("m0"))
            .with_model_matrix(x.clone())
            .with_random_terms(vec![subject_intercept()]);
        let m1 = DummyFit::new(4, 5, -9.0, Some("m1"))
            .with_model_matrix(x)
            .with_random_terms(vec![item_intercept()]);

        let assessment = ModelComparisonAssessment::assess(&m0, &m1);

        assert_eq!(
            assessment.class,
            ModelComparisonClass::NonNestedRandomEffects
        );
        assert_eq!(assessment.random_effects, RandomEffectComparison::NonNested);
        assert!(!assessment.lrt_available);
        assert_eq!(
            assessment.lrt_reason.as_deref(),
            Some("random-effect term structures are not nested")
        );
        assert!(assessment.information_criteria_available);
    }

    #[test]
    fn test_model_comparison_assessment_classifies_incompatible_inputs() {
        use crate::model::traits::{Family, LinkFunction};

        let m0 = DummyFit::new(4, 2, -10.0, Some("m0"))
            .with_response(DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]));
        let m_different_response = DummyFit::new(4, 3, -9.0, Some("m1"))
            .with_response(DVector::from_vec(vec![1.0, 2.0, 3.0, 5.0]));
        assert_eq!(
            ModelComparisonAssessment::assess(&m0, &m_different_response).class,
            ModelComparisonClass::DifferentResponse
        );

        let bernoulli = DummyFit::new(4, 2, -10.0, Some("bernoulli"))
            .with_family(Family::Bernoulli, LinkFunction::Logit);
        let poisson = DummyFit::new(4, 3, -9.0, Some("poisson"))
            .with_family(Family::Poisson, LinkFunction::Log);
        assert_eq!(
            ModelComparisonAssessment::assess(&bernoulli, &poisson).class,
            ModelComparisonClass::DifferentFamily
        );

        let logit = DummyFit::new(4, 2, -10.0, Some("logit"))
            .with_family(Family::Bernoulli, LinkFunction::Logit);
        let probit = DummyFit::new(4, 3, -9.0, Some("probit"))
            .with_family(Family::Bernoulli, LinkFunction::Probit);
        assert_eq!(
            ModelComparisonAssessment::assess(&logit, &probit).class,
            ModelComparisonClass::DifferentLink
        );

        let ml = DummyFit::new(4, 2, -10.0, Some("ml")).with_reml(false);
        let reml = DummyFit::new(4, 3, -9.0, Some("reml")).with_reml(true);
        let assessment = ModelComparisonAssessment::assess(&ml, &reml);
        assert_eq!(assessment.class, ModelComparisonClass::MixedFitCriterion);
        assert!(!assessment.information_criteria_available);
        assert!(assessment
            .valid_alternatives
            .contains(&ModelComparisonAlternative::RefitWithCommonCriterion));
    }

    #[test]
    fn test_model_comparison_assessment_marks_ml_refit_for_reml_fixed_effect_change() {
        let intercept = DMatrix::from_element(4, 1, 1.0);
        let m0 = DummyFit::new(4, 2, -10.0, Some("m0"))
            .with_reml(true)
            .with_model_matrix(intercept);
        let m1 = DummyFit::new(4, 3, -9.0, Some("m1"))
            .with_reml(true)
            .with_model_matrix(intercept_x());

        let assessment = ModelComparisonAssessment::assess(&m0, &m1);

        assert_eq!(assessment.class, ModelComparisonClass::NestedFixedEffects);
        assert!(!assessment.lrt_available);
        assert!(assessment.ml_refit_required);
        assert_eq!(
            assessment.ml_refit_reason.as_deref(),
            Some("models differ in fixed effects but were fitted by REML; refit with ML for fixed-effect likelihood comparisons")
        );
        assert!(!assessment.information_criteria_available);
    }

    #[test]
    fn test_model_comparison_sequence_assesses_adjacent_pairs() {
        let intercept = DMatrix::from_element(4, 1, 1.0);
        let m0 = DummyFit::new(4, 2, -10.0, Some("m0")).with_model_matrix(intercept);
        let m1 = DummyFit::new(4, 3, -9.0, Some("m1")).with_model_matrix(intercept_x());
        let m2 = DummyFit::new(4, 4, -8.0, Some("m2"))
            .with_model_matrix(intercept_x())
            .with_random_terms(vec![subject_intercept()]);

        let assessments = assess_model_comparison_sequence(&[&m0, &m1, &m2]).unwrap();

        assert_eq!(assessments.len(), 2);
        assert_eq!(
            assessments[0].class,
            ModelComparisonClass::NestedFixedEffects
        );
        assert_eq!(
            assessments[1].class,
            ModelComparisonClass::NestedRandomEffects
        );
    }

    #[test]
    fn test_lrt_rejects_non_increasing_dof() {
        let m0 = DummyFit::new(180, 6, -876.0, Some("m0"));
        let m1 = DummyFit::new(180, 4, -875.0, Some("m1"));
        let err = lrt_refusal_from_assessment(&m0, &m1);

        assert_eq!(
            err,
            "larger model must have more degrees of freedom than the smaller model"
        );
    }

    #[test]
    fn test_lrt_rejects_decreasing_loglikelihood() {
        let m0 = DummyFit::new(180, 4, -876.0, Some("m0"));
        let m1 = DummyFit::new(180, 6, -877.0, Some("m1"));
        let err = LikelihoodRatioTest::test(&[&m0, &m1]).unwrap_err();

        assert_eq!(
            err,
            "Log-likelihood must not be lower in models with more degrees of freedom"
        );
    }

    #[test]
    fn test_lrt_flags_loglik_within_tol() {
        let m0 = DummyFit::new(180, 4, -100.0, Some("m0"));
        let m1 = DummyFit::new(180, 5, m0.loglikelihood() - LOG_LIK_TOL * 0.5, Some("m1"));

        let lrt = LikelihoodRatioTest::test(&[&m0, &m1]).unwrap();

        assert_eq!(lrt.loglik_within_optimizer_tol, vec![true]);
        assert_relative_eq!(lrt.chisq[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(lrt.pvalues[0], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_lrt_strict_violation_still_rejects() {
        let m0 = DummyFit::new(180, 4, -100.0, Some("m0"));
        let m1 = DummyFit::new(180, 5, m0.loglikelihood() - LOG_LIK_TOL * 2.0, Some("m1"));

        let err = LikelihoodRatioTest::test(&[&m0, &m1]).unwrap_err();

        assert_eq!(
            err,
            "Log-likelihood must not be lower in models with more degrees of freedom"
        );
    }

    #[test]
    fn test_lrt_chi_nonneg_invariant() {
        let m0 = DummyFit::new(180, 4, -100.0, Some("m0"));
        let m1 = DummyFit::new(180, 5, m0.loglikelihood() - LOG_LIK_TOL * 0.5, Some("m1"));
        let m2 = DummyFit::new(180, 6, -99.5, Some("m2"));

        let lrt = LikelihoodRatioTest::test(&[&m0, &m1, &m2]).unwrap();

        assert!(lrt.chisq.iter().all(|chi| *chi >= 0.0));
        assert_eq!(lrt.loglik_within_optimizer_tol, vec![true, false]);
        assert_relative_eq!(lrt.chisq[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(lrt.chisq[1], 1.0, epsilon = 1e-9);
    }

    #[test]
    fn test_lrt_accepts_distinct_fits_with_identical_response_values() {
        let response = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let m0 = DummyFit::new(4, 2, -10.0, Some("y ~ 1")).with_response(response.clone());
        let m1 = DummyFit::new(4, 3, -9.0, Some("y ~ 1 + x")).with_response(response);

        let lrt = LikelihoodRatioTest::test(&[&m0, &m1]).unwrap();

        assert_eq!(lrt.nobs, 4);
        assert_relative_eq!(lrt.chisq[0], 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_lrt_rejects_same_nobs_with_different_response_values() {
        let m0 = DummyFit::new(4, 2, -10.0, Some("y ~ 1"))
            .with_response(DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]));
        let m1 = DummyFit::new(4, 3, -9.0, Some("z ~ 1 + x"))
            .with_response(DVector::from_vec(vec![1.0, 2.0, 3.0, 5.0]));

        let err = lrt_refusal_from_assessment(&m0, &m1);

        assert_eq!(err, "models were not fitted to the same response values");
    }

    #[test]
    fn test_lrt_rejects_fixed_effect_non_nested_column_space() {
        let small_x = DMatrix::from_row_slice(
            4,
            2,
            &[
                1.0, 0.0, //
                1.0, 1.0, //
                1.0, 2.0, //
                1.0, 3.0,
            ],
        );
        let large_x = DMatrix::from_row_slice(
            4,
            2,
            &[
                1.0, 0.0, //
                1.0, 1.0, //
                1.0, 0.0, //
                1.0, 1.0,
            ],
        );
        let m0 = DummyFit::new(4, 2, -10.0, Some("y ~ x"))
            .with_reml(false)
            .with_model_matrix(small_x);
        let m1 = DummyFit::new(4, 3, -9.0, Some("y ~ z"))
            .with_reml(false)
            .with_model_matrix(large_x);

        let err = lrt_refusal_from_assessment(&m0, &m1);

        assert_eq!(err, "fixed-effect column spaces are not nested");
    }

    #[test]
    fn test_lrt_rejects_random_effect_non_nested_terms() {
        let subject_intercept = RandomEffectTermInfo {
            group: "subject".to_string(),
            columns: vec!["(Intercept)".to_string()],
        };
        let item_intercept = RandomEffectTermInfo {
            group: "item".to_string(),
            columns: vec!["(Intercept)".to_string()],
        };
        let m0 = DummyFit::new(10, 2, -10.0, Some("y ~ 1 + (1 | subject)"))
            .with_random_terms(vec![subject_intercept]);
        let m1 = DummyFit::new(10, 3, -9.0, Some("y ~ 1 + (1 | item)"))
            .with_random_terms(vec![item_intercept]);

        let err = lrt_refusal_from_assessment(&m0, &m1);

        assert_eq!(err, "random-effect term structures are not nested");
    }

    #[test]
    fn test_lrt_accepts_random_intercept_nested_in_random_slope_same_group() {
        let subject_intercept = RandomEffectTermInfo {
            group: "subject".to_string(),
            columns: vec!["(Intercept)".to_string()],
        };
        let subject_intercept_slope = RandomEffectTermInfo {
            group: "subject".to_string(),
            columns: vec!["(Intercept)".to_string(), "x".to_string()],
        };
        let m0 = DummyFit::new(10, 2, -10.0, Some("y ~ 1 + (1 | subject)"))
            .with_random_terms(vec![subject_intercept]);
        let m1 = DummyFit::new(10, 4, -9.0, Some("y ~ x + (1 + x | subject)"))
            .with_random_terms(vec![subject_intercept_slope]);

        let lrt = LikelihoodRatioTest::test(&[&m0, &m1]).unwrap();

        assert_eq!(lrt.chisq_dof, vec![2]);
        assert_relative_eq!(lrt.chisq[0], 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_linear_model_fit_compares_with_mixed_model_like_fit() {
        let y = DVector::from_vec(vec![1.0, 2.1, 2.9, 4.2, 5.1]);
        let intercept = DMatrix::from_element(5, 1, 1.0);
        let lm0 =
            LinearModelFit::fit(y.clone(), intercept.clone(), Some("y ~ 1".to_string())).unwrap();
        let mixed_like = DummyFit::new(
            5,
            lm0.dof() + 1,
            lm0.loglikelihood() + 1.5,
            Some("y ~ 1 + (1 | g)"),
        )
        .with_model_matrix(intercept)
        .with_response(y)
        .with_reml(false)
        .with_random_terms(vec![RandomEffectTermInfo {
            group: "g".to_string(),
            columns: vec!["(Intercept)".to_string()],
        }]);

        let lrt = LikelihoodRatioTest::test(&[&lm0, &mixed_like]).unwrap();

        assert_eq!(lrt.formulas, vec!["y ~ 1", "y ~ 1 + (1 | g)"]);
        assert_eq!(lrt.chisq_dof, vec![1]);
        assert_relative_eq!(lrt.chisq[0], 3.0, epsilon = 1e-12);
    }

    #[test]
    fn test_linear_model_fit_rejects_non_nested_mixed_comparison() {
        let y = DVector::from_vec(vec![1.0, 2.1, 2.9, 4.2, 5.1]);
        let x = DMatrix::from_row_slice(
            5,
            2,
            &[
                1.0, 0.0, //
                1.0, 1.0, //
                1.0, 2.0, //
                1.0, 3.0, //
                1.0, 4.0,
            ],
        );
        let lm1 = LinearModelFit::fit(y.clone(), x, Some("y ~ x".to_string())).unwrap();
        let mixed_intercept = DummyFit::new(
            5,
            lm1.dof() + 1,
            lm1.loglikelihood() + 1.0,
            Some("y ~ 1 + (1 | g)"),
        )
        .with_model_matrix(DMatrix::from_element(5, 1, 1.0))
        .with_response(y)
        .with_reml(false)
        .with_random_terms(vec![RandomEffectTermInfo {
            group: "g".to_string(),
            columns: vec!["(Intercept)".to_string()],
        }]);

        let err = lrt_refusal_from_assessment(&lm1, &mixed_intercept);

        assert_eq!(err, "models appear nested only in the reverse order");
    }

    #[test]
    fn test_pvalue_requires_a_single_comparison() {
        let m0 = DummyFit::new(180, 4, -897.0, Some("m0"));
        let m1 = DummyFit::new(180, 5, -890.0, Some("m1"));
        let m2 = DummyFit::new(180, 6, -876.0, Some("m2"));

        let lrt_single = LikelihoodRatioTest::test(&[&m0, &m1]).unwrap();
        assert!(lrt_single.pvalue().unwrap() < 0.01);

        let lrt_multiple = LikelihoodRatioTest::test(&[&m0, &m1, &m2]).unwrap();
        assert_eq!(
            lrt_multiple.pvalue().unwrap_err(),
            "Cannot extract only one p-value from a multiple test result."
        );
    }

    #[test]
    fn test_lrt_display_includes_formula_table() {
        let m0 = DummyFit::new(180, 4, -897.0, Some("reaction ~ 1 + days + (1 | subj)"));
        let m1 = DummyFit::new(
            180,
            6,
            -876.0,
            Some("reaction ~ 1 + days + (1 + days | subj)"),
        );
        let lrt = LikelihoodRatioTest::test(&[&m0, &m1]).unwrap();
        let out = lrt.to_string();

        assert!(out.contains("Likelihood-ratio test: 2 models fitted on 180 observations"));
        assert!(out.contains("1: reaction ~ 1 + days + (1 | subj)"));
        assert!(out.contains("2: reaction ~ 1 + days + (1 + days | subj)"));
        assert!(out.contains("[2]"));
        assert!(out.contains("1752.0000"));
        assert!(out.contains("42.0000"));
        assert!(out.contains("<1e-09"));
    }

    #[test]
    fn test_lrt_markdown_matches_julia_style_table() {
        let m0 = DummyFit::new(180, 4, -897.0, Some("reaction ~ 1 + days + (1 | subj)"));
        let m1 = DummyFit::new(
            180,
            6,
            -876.0,
            Some("reaction ~ 1 + days + (1 + days | subj)"),
        );
        let lrt = LikelihoodRatioTest::test(&[&m0, &m1]).unwrap();

        assert_eq!(
            lrt.to_markdown(),
            concat!(
                "|                                          | model-dof | -2 logLik |  χ² | χ²-dof | P(>χ²) |\n",
                "|:---------------------------------------- | ---------:| ---------:| ---:| ------:|:------ |\n",
                "| reaction ~ 1 + days + (1 \\| subj)        |         4 |     -1794 |     |        |        |\n",
                "| reaction ~ 1 + days + (1 + days \\| subj) |         6 |     -1752 |  42 |      2 | <1e-09 |\n"
            )
        );
    }

    #[test]
    fn test_lrt_latex_matches_julia_header() {
        let m0 = DummyFit::new(180, 4, -897.0, Some("reaction ~ 1 + days + (1 | subj)"));
        let m1 = DummyFit::new(
            180,
            6,
            -876.0,
            Some("reaction ~ 1 + days + (1 + days | subj)"),
        );
        let lrt = LikelihoodRatioTest::test(&[&m0, &m1]).unwrap();
        let out = lrt.to_latex();

        // mime.jl asserts via startswith on this exact header.
        assert!(out.starts_with(concat!(
            "\\begin{tabular}\n",
            "{l | r | r | r | r | l}\n",
            " & model-dof & -2 logLik & $\\chi^2$ & $\\chi^2$-dof & P(>$\\chi^2$) \\\\",
        )));
        assert!(out.contains("reaction ~ 1 + days + (1 | subj) & 4 & -1794"));
        assert!(out.contains("& 42 & 2 & <1e-09"));
        assert!(out.ends_with("\\end{tabular}\n"));
    }

    #[test]
    fn test_lrt_rejects_mixing_reml_and_ml() {
        let m_ml = DummyFit::new(180, 4, -897.0, Some("m0")).with_reml(false);
        let m_reml = DummyFit::new(180, 6, -876.0, Some("m1")).with_reml(true);

        let err = lrt_refusal_from_assessment(&m_ml, &m_reml);
        assert_eq!(
            err,
            "models mix REML and ML fit criteria; refit with a common criterion"
        );
    }

    #[test]
    fn test_lrt_rejects_reml_with_fe_mismatch() {
        let intercept = DMatrix::from_element(4, 1, 1.0);
        let intercept_slope = DMatrix::from_row_slice(
            4,
            2,
            &[
                1.0, 0.0, //
                1.0, 1.0, //
                1.0, 2.0, //
                1.0, 3.0,
            ],
        );
        let m0 = DummyFit::new(4, 2, -10.0, Some("y ~ 1"))
            .with_reml(true)
            .with_model_matrix(intercept);
        let m1 = DummyFit::new(4, 3, -9.0, Some("y ~ 1 + x"))
            .with_reml(true)
            .with_model_matrix(intercept_slope);

        let err = lrt_refusal_from_assessment(&m0, &m1);

        assert_eq!(
            err,
            "REML likelihood-ratio tests require identical fixed effects; refit with ML"
        );
    }

    #[test]
    fn test_lrt_accepts_reml_with_identical_fe() {
        let x = DMatrix::from_row_slice(
            4,
            2,
            &[
                1.0, 0.0, //
                1.0, 1.0, //
                1.0, 2.0, //
                1.0, 3.0,
            ],
        );
        let reordered_x = DMatrix::from_row_slice(
            4,
            2,
            &[
                0.0, 1.0, //
                1.0, 1.0, //
                2.0, 1.0, //
                3.0, 1.0,
            ],
        );
        let m0 = DummyFit::new(4, 3, -10.0, Some("y ~ 1 + x"))
            .with_reml(true)
            .with_model_matrix(x);
        let m1 = DummyFit::new(4, 4, -9.0, Some("y ~ x + 1 + (1 | g)"))
            .with_reml(true)
            .with_model_matrix(reordered_x)
            .with_random_terms(vec![RandomEffectTermInfo {
                group: "g".to_string(),
                columns: vec!["(Intercept)".to_string()],
            }]);

        let lrt = LikelihoodRatioTest::test(&[&m0, &m1]).unwrap();

        assert_eq!(lrt.chisq_dof, vec![1]);
        assert_relative_eq!(lrt.chisq[0], 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_lrt_ml_with_nested_fe_still_works() {
        let intercept = DMatrix::from_element(4, 1, 1.0);
        let intercept_slope = DMatrix::from_row_slice(
            4,
            2,
            &[
                1.0, 0.0, //
                1.0, 1.0, //
                1.0, 2.0, //
                1.0, 3.0,
            ],
        );
        let m0 = DummyFit::new(4, 2, -10.0, Some("y ~ 1"))
            .with_reml(false)
            .with_model_matrix(intercept);
        let m1 = DummyFit::new(4, 3, -9.0, Some("y ~ 1 + x"))
            .with_reml(false)
            .with_model_matrix(intercept_slope);

        let lrt = LikelihoodRatioTest::test(&[&m0, &m1]).unwrap();

        assert_eq!(lrt.chisq_dof, vec![1]);
        assert_relative_eq!(lrt.chisq[0], 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_lrt_rejects_mixing_families() {
        use crate::model::traits::{Family, LinkFunction};
        let m_bernoulli = DummyFit::new(180, 4, -897.0, Some("m_bernoulli"))
            .with_family(Family::Bernoulli, LinkFunction::Logit);
        let m_poisson = DummyFit::new(180, 6, -876.0, Some("m_poisson"))
            .with_family(Family::Poisson, LinkFunction::Log);

        let err = lrt_refusal_from_assessment(&m_bernoulli, &m_poisson);
        assert_eq!(err, "models have different conditional response families");
    }

    #[test]
    fn test_lrt_rejects_mixing_links() {
        use crate::model::traits::{Family, LinkFunction};
        let m_logit = DummyFit::new(180, 4, -897.0, Some("m_logit"))
            .with_family(Family::Bernoulli, LinkFunction::Logit);
        let m_probit = DummyFit::new(180, 6, -876.0, Some("m_probit"))
            .with_family(Family::Bernoulli, LinkFunction::Probit);

        let err = lrt_refusal_from_assessment(&m_logit, &m_probit);
        assert_eq!(err, "models have different link functions");
    }

    #[test]
    fn test_lrt_rejects_glmm_vs_lmm_family_mix() {
        use crate::model::traits::{Family, LinkFunction};
        let m_lmm = DummyFit::new(180, 4, -897.0, Some("m_lmm"));
        let m_glmm = DummyFit::new(180, 6, -876.0, Some("m_glmm"))
            .with_family(Family::Bernoulli, LinkFunction::Logit);

        let err = lrt_refusal_from_assessment(&m_lmm, &m_glmm);
        assert_eq!(err, "models have different conditional response families");
    }

    #[test]
    fn test_lrt_html_includes_table_markup() {
        let m0 = DummyFit::new(180, 4, -897.0, Some("reaction ~ 1 + days + (1 | subj)"));
        let m1 = DummyFit::new(
            180,
            6,
            -876.0,
            Some("reaction ~ 1 + days + (1 + days | subj)"),
        );
        let lrt = LikelihoodRatioTest::test(&[&m0, &m1]).unwrap();
        let out = lrt.to_html();

        assert!(out.starts_with("<table><tr>"));
        assert!(out.contains("<th align=\"right\">model-dof</th>"));
        // χ² is left literal in HTML (no MathJax escaping required).
        assert!(out.contains("<th align=\"right\">χ²</th>"));
        assert!(out.contains("<th align=\"right\">P(&gt;χ²)</th>"));
        assert!(out.contains(
            "<td align=\"left\">reaction ~ 1 + days + (1 | subj)</td><td align=\"right\">4</td>"
        ));
        assert!(out.contains("<td align=\"right\"><1e-09</td>"));
        assert!(out.ends_with("</table>\n"));
    }

    fn pb_lrt_fixture() -> crate::model::data::DataFrame {
        use crate::model::data::DataFrame;
        let mut y = Vec::new();
        let mut x = Vec::new();
        let mut g = Vec::new();
        for subj in 0..12 {
            let intercept = (subj as f64 - 5.5) * 4.0;
            let slope = (subj as f64 - 5.5) * 0.8;
            for day in 0..8 {
                let xv = day as f64;
                let noise = ((subj * 8 + day) as f64 * 12.9898).sin() * 7.0;
                y.push(250.0 + intercept + (10.0 + slope) * xv + noise);
                x.push(xv);
                g.push(format!("s{subj}"));
            }
        }
        let mut data = DataFrame::new();
        data.add_numeric("y", y).unwrap();
        data.add_numeric("x", x).unwrap();
        data.add_categorical("g", g).unwrap();
        data
    }

    #[test]
    fn test_parametric_bootstrap_lrt_runs_and_is_seed_deterministic() {
        use crate::formula::parse_formula;
        use crate::model::linear::LinearMixedModel;
        use rand::{rngs::StdRng, SeedableRng};

        let data = pb_lrt_fixture();
        let mut smaller =
            LinearMixedModel::new(parse_formula("y ~ 1 + x + (1 | g)").unwrap(), &data, None)
                .unwrap();
        smaller.fit(false).unwrap(); // ML
        let mut larger = LinearMixedModel::new(
            parse_formula("y ~ 1 + x + (1 + x | g)").unwrap(),
            &data,
            None,
        )
        .unwrap();
        larger.fit(false).unwrap();

        let mut rng = StdRng::seed_from_u64(20260515);
        let result = parametric_bootstrap_lrt(&mut rng, 20, &smaller, &larger).unwrap();

        assert_eq!(result.schema_name, PARAMETRIC_BOOTSTRAP_LRT_SCHEMA);
        assert_eq!(result.n_sim_requested, 20);
        assert_eq!(
            result.n_sim_completed + result.n_refit_failures,
            20,
            "every replicate is either completed or a recorded failure"
        );
        assert!(result.n_sim_completed > 0);
        assert!(result.observed_statistic >= 0.0);
        assert!(
            result.p_value > 0.0 && result.p_value <= 1.0,
            "bootstrap p-value must be in (0, 1], got {}",
            result.p_value
        );
        assert_eq!(result.chisq_dof, larger.dof() - smaller.dof());

        // Same seed → identical result.
        let mut rng2 = StdRng::seed_from_u64(20260515);
        let result2 = parametric_bootstrap_lrt(&mut rng2, 20, &smaller, &larger).unwrap();
        assert_eq!(result, result2);
    }

    #[test]
    fn test_parametric_bootstrap_lrt_rejects_bad_arguments() {
        use crate::formula::parse_formula;
        use crate::model::linear::LinearMixedModel;
        use rand::{rngs::StdRng, SeedableRng};

        let data = pb_lrt_fixture();
        let mut smaller =
            LinearMixedModel::new(parse_formula("y ~ 1 + x + (1 | g)").unwrap(), &data, None)
                .unwrap();
        smaller.fit(false).unwrap();
        let mut larger = LinearMixedModel::new(
            parse_formula("y ~ 1 + x + (1 + x | g)").unwrap(),
            &data,
            None,
        )
        .unwrap();
        larger.fit(false).unwrap();

        let mut rng = StdRng::seed_from_u64(1);
        assert!(parametric_bootstrap_lrt(&mut rng, 0, &smaller, &larger).is_err());
        // Swapped nesting (smaller has more dof than "larger") is rejected.
        assert!(parametric_bootstrap_lrt(&mut rng, 5, &larger, &smaller).is_err());
    }
}
