//! Linear mixed-effects model (LMM).
//!
//! Implements the penalized least squares (PLS) algorithm for fitting
//! linear mixed models via profile likelihood optimization.
//!
//! The model is: y = Xβ + Zb + ε, where b ~ N(0, σ²Λθ Λθ') and ε ~ N(0, σ²I).
//!
//! The θ parameters control the relative covariance factor Λ. The objective
//! function (deviance or REML criterion) is profiled over β and σ², leaving
//! only θ to be optimized numerically.

use nalgebra::{DMatrix, DVector, SymmetricEigen};
use nalgebra_sparse::{coo::CooMatrix, csc::CscMatrix};
#[cfg(feature = "nlopt")]
use nlopt::{
    Algorithm as NloptAlgorithm, FailState as NloptFailState, Nlopt, Target as NloptTarget,
};
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::compiler::{
    compile_formula_ir, BasisLoading, BootstrapInferenceDetails, CertificateCheck,
    CompiledModelArtifact, CompilerPolicy, ContrastFamilyDetails, ConvergenceVerification,
    ConvergenceVerificationRun, ConvergenceVerificationStatus, CovarianceFamily,
    CovarianceFamilyTransition, DesignAudit, Diagnostic, DiagnosticCode, DiagnosticSeverity,
    DiagnosticStage, DominantLoading, EffectiveCovarianceSummary, EffectiveRankStatus,
    EstimabilityAssessment, EstimabilityStatus, EvidenceMethod, FitStatus,
    FixedContrastEstimability, FixedEffectCovarianceDetails, FixedEffectCovarianceMatrix,
    FixedEffectCovarianceMethod, FixedEffectHypothesis, FixedEffectInferenceDetails,
    FixedEffectInferenceMethod, FixedEffectInferenceRow, FixedEffectInferenceRowKind,
    FixedEffectInferenceStatus, FixedEffectInferenceTable, FixedEffectNullTargetSummary,
    FixedEffectReliabilityReason, FixedEffectStatisticName, FixedEffectTermTestType,
    FixedEffectTest, FixedEffectTestMethod, InferenceMethod, InferenceStatus,
    InterpretableSubmodel, KenwardRogerInferenceDetails, ModelAuditReport, ModelStateChange,
    ModelStateSummary, OptimizerCertificate, OptimizerDerivativeEvidence, PolicyAction,
    PolicyRecommendation, ReductionRecord, ReductionTrigger, ReliabilityGrade,
    SupportedCovarianceDirection, DOMINANT_LOADING_THRESHOLD, INTERPRETABLE_GAP_TOLERANCE,
};
use crate::error::{MixedModelError, Result};
use crate::formula::{FixedTerm, Formula, RandomCovariance, RandomTerm};
use crate::model::data::{CategoricalCoding, Column, DataFrame};
use crate::model::fixed_design::{
    DenseFixedDesign, FixedDesign, FixedDesignBackend, FixedDesignBackendPreference,
    FixedDesignBuildPolicy, FixedDesignStorage, FixedDesignSummary,
};
use crate::model::traits::MixedModelFit;
#[cfg(feature = "prima")]
use crate::optimizer::prima::{minimize_bobyqa, PrimaBobyqaOptions};
use crate::optimizer::trust_bq::{
    minimize_with_progress as minimize_trust_bq_with_progress, TrustBqOptions, TrustBqProgress,
    TrustBqStopReason,
};
use crate::stats::{BlockDescription, CoefTable, CoefTablePValuePolicy, ModelSummary, VarCorr};
use crate::types::matrix_block::{
    block_index, with_block_pair_mut, with_block_triple, with_dense_block, MatrixBlock,
};
#[cfg(feature = "prima")]
use crate::types::opt_summary::OptimizerBackend;
use crate::types::{FeMat, FeTerm, FitLogEntry, OptSummary, Optimizer, OptimizerSource, ReMat};
use crate::unstable_internal_method;

mod blocks;
pub(crate) use blocks::*;

mod bootstrap;
pub use bootstrap::{
    parametricbootstrap, BootstrapFailedRefitPolicy, BootstrapInterval, BootstrapIntervalMethod,
    BootstrapQuantile, BootstrapRefitOptions, BootstrapReplicate, BootstrapRunMetadata,
    BootstrapRunPayload, BootstrapSeedRecord, BootstrapTarget, BootstrapTargetKind,
    FixedEffectBootstrapOptions, FixedEffectNullBootstrapTarget, FixedEffectNullCovariancePolicy,
    MixedModelBootstrap, BOOTSTRAP_RUN_SCHEMA, BOOTSTRAP_RUN_SCHEMA_VERSION,
};
use bootstrap::{quantile_sorted, validate_level};

/// A fitted (or constructed but unfitted) linear mixed-effects model.
///
/// Corresponds to `LinearMixedModel{T}` in MixedModels.jl.
///
/// # Fields
/// - `formula`: the parsed model formula
/// - `reterms`: random-effects model matrices, sorted by decreasing nranef
/// - `xy_mat`: the fixed-effects model matrix concatenated with y, with optional weighting
/// - `feterm`: the fixed-effects model matrix with rank/pivot info
/// - `sqrtwts`: square roots of case weights (empty if unweighted)
/// - `parmap`: mapping from θ indices to (block, row, col) in λ
/// - `dims`: model dimensions (n, p, nretrms)
/// - `a_blocks`: lower triangle of [Z X y]'[Z X y] in blocked storage
/// - `l_blocks`: blocked lower Cholesky factor of Λ'AΛ + I
/// - `optsum`: optimization summary
/// - `compiler_artifact`: semantic compiler/audit metadata for the requested model
/// - `residual_source`: whether `sigma` is estimated (default) or fixed by
///   user-supplied sampling variances (set only by
///   `LinearMixedModel::from_summary_estimates`). See
///   `docs/summary_estimates_meta_analysis.md`.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct LinearMixedModel {
    pub(crate) formula: Formula,
    pub(crate) reterms: Vec<ReMat>,
    pub(crate) xy_mat: FeMat,
    pub(crate) y: DVector<f64>,
    pub(crate) feterm: FeTerm,
    pub(crate) fixed_design: FixedDesign,
    pub(crate) sqrtwts: Vec<f64>,
    pub(crate) parmap: Vec<(usize, usize, usize)>, // (block, row, col)
    pub(crate) dims: ModelDims,
    pub(crate) a_blocks: Vec<MatrixBlock>,
    pub(crate) l_blocks: Vec<MatrixBlock>,
    pub(crate) optsum: OptSummary,
    pub(crate) compiler_artifact: CompiledModelArtifact,
    pub(crate) residual_source: crate::model::summary_estimates::ResidualSource,
    /// Training-time categorical level order (and explicit contrast, if any)
    /// for every categorical column in the fitting frame, keyed by column
    /// name. Used by [`LinearMixedModel::predict_new`] to rebuild the
    /// fixed-effects design against the *training* factor encoding rather than
    /// newdata's own first-appearance order, so that differing observation
    /// order or a missing level in newdata cannot silently reorder or drop
    /// dummy columns.
    pub(crate) training_categorical: std::collections::HashMap<String, TrainingCategoricalLevels>,
}

/// Snapshot of a training categorical column's encoding contract: the
/// canonical level order plus any explicit contrast basis. Stored on the
/// fitted model so prediction can realign newdata to the training encoding.
#[derive(Debug, Clone)]
pub(crate) struct TrainingCategoricalLevels {
    pub(crate) levels: Vec<String>,
    pub(crate) contrast: Option<crate::model::data::CategoricalContrast>,
}

/// Model dimensions.
#[derive(Debug, Clone, Copy)]
pub struct ModelDims {
    /// Number of observations.
    pub n: usize,
    /// Rank of the fixed-effects matrix.
    pub p: usize,
    /// Number of random-effects terms.
    pub nretrms: usize,
}

/// How to handle random-effects levels not seen during training.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum NewReLevels {
    /// Return an error if any unseen levels are encountered.
    Error,
    /// Use zero random effects for unseen levels (population-level prediction).
    Population,
    /// Return `None` for observations that have unseen levels.
    Missing,
}

/// Availability status for one prediction-variance row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum PredictionVarianceStatus {
    /// Fixed-effect, random-effect, and combined variance components are available.
    Available,
    /// Variance components are computed but the method is approximate or not
    /// certified for the requested model family.
    Degraded,
    /// At least one required variance component is unavailable; see `reason`.
    Unavailable,
}

/// Engine method used to construct a prediction-variance payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum PredictionVarianceMethod {
    /// LMM identity-link approximation using model-based fixed-effect
    /// covariance plus conditional-mode random-effect covariance blocks.
    LmmConditionalModeCovariance,
    /// GLMM working-Hessian delta-method approximation. This is useful
    /// geometry, but not the certified active-subspace Hessian requested for
    /// full GLMM Wald/parity claims.
    GlmmPirlsLaplaceWorkingDelta,
    /// GLMM joint-Laplace fitted-mean delta-method approximation using the
    /// final PIRLS/Laplace conditional-mode covariance over fixed and random
    /// effects. The fitted-mean variance includes fixed, random, and cross
    /// terms; future-observation columns are plug-in predictive summaries on
    /// the response scale.
    GlmmJointLaplaceConditionalDelta,
    /// GLMM fast-PIRLS profiled fitted-mean delta-method approximation whose
    /// conditional-mode covariance geometry is certified by a post-fit
    /// stationarity-plus-curvature certificate of the profiled optimum. Same
    /// component structure as [`Self::GlmmJointLaplaceConditionalDelta`], for
    /// the profiled fast-PIRLS estimator's own objective.
    GlmmPirlsProfiledCertifiedConditionalDelta,
    /// No certified prediction-variance method is available for this model.
    Unavailable,
}

/// Row-level prediction variance for a new-data prediction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct PredictionVarianceRow {
    /// Zero-based row index in the supplied new data.
    pub row: usize,
    /// Point prediction on the model's prediction scale, when available.
    pub prediction: Option<f64>,
    /// Fixed-effect contribution `x V_beta x'`.
    pub fixed_variance: Option<f64>,
    /// Random-effect contribution `z V_b z'`.
    pub random_variance: Option<f64>,
    /// Covariance between the fixed-effect and random-effect prediction
    /// contributions. `combined_variance` includes `2 * fixed_random_covariance`.
    pub fixed_random_covariance: Option<f64>,
    /// Combined fitted-mean variance.
    pub combined_variance: Option<f64>,
    /// Square root of `combined_variance`.
    pub se_fit: Option<f64>,
    /// Prediction variance for a future observation. For LMMs this is
    /// `combined_variance + sigma^2`; for GLMM response-scale rows it is the
    /// law-of-total-variance moment of the plug-in predictive distribution.
    pub prediction_variance: Option<f64>,
    /// Lower confidence bound for the fitted mean on the prediction scale.
    pub confidence_lower: Option<f64>,
    /// Upper confidence bound for the fitted mean on the prediction scale.
    pub confidence_upper: Option<f64>,
    /// Lower prediction bound for a future observation on the prediction scale.
    pub prediction_lower: Option<f64>,
    /// Upper prediction bound for a future observation on the prediction scale.
    pub prediction_upper: Option<f64>,
    /// Availability status for the combined variance.
    pub status: PredictionVarianceStatus,
    /// Stable human-readable reason when unavailable or degraded.
    pub reason: Option<String>,
}

/// Engine prediction-variance payload with row-level provenance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct PredictionVariancePayload {
    /// Versioned schema name for downstream JSON consumers.
    pub schema_name: String,
    /// Version of this payload schema.
    pub schema_version: String,
    /// Crate version that produced the payload.
    pub crate_version: Option<String>,
    /// Method used to compute the variance rows.
    pub method: PredictionVarianceMethod,
    /// Confidence level used for interval columns, if requested.
    pub confidence_level: Option<f64>,
    /// One row per prediction row.
    pub rows: Vec<PredictionVarianceRow>,
    /// Payload-level notes describing scope and assumptions.
    pub notes: Vec<String>,
}

impl PredictionVariancePayload {
    pub(crate) fn new(
        method: PredictionVarianceMethod,
        rows: Vec<PredictionVarianceRow>,
        confidence_level: Option<f64>,
        notes: Vec<String>,
    ) -> Self {
        Self {
            schema_name: "mixedmodels.prediction_variance".to_string(),
            schema_version: "3".to_string(),
            crate_version: Some(env!("CARGO_PKG_VERSION").to_string()),
            method,
            confidence_level,
            rows,
            notes,
        }
    }
}

/// Profiled quantities for a batch of response columns sharing the same
/// fixed-effects design, random-effects structure, and theta.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct ResponseMatrixProfile {
    /// Fixed-effects solutions for each response column, shape `p x q`.
    pub beta: DMatrix<f64>,
    /// Profiled residual scales for each response column, length `q`.
    pub sigma: DVector<f64>,
    /// Penalized weighted residual sum of squares for each response column.
    pub pwrss: DVector<f64>,
    /// Profiled objective contribution for each response column.
    pub objectives: DVector<f64>,
    /// Sum of profiled objective contributions across all columns.
    pub total_objective: f64,
    /// Shared random-effects log-determinant term.
    pub logdet_re: f64,
    /// Shared fixed-effects log-determinant term used by REML.
    pub logdet_xx: f64,
}

#[derive(Debug)]
pub(crate) struct PatternSearchOutcome {
    pub(crate) best_theta: Vec<f64>,
    pub(crate) best_fmin: f64,
    pub(crate) feval_count: i64,
    pub(crate) fit_log: Vec<FitLogEntry>,
    #[cfg(test)]
    pub(crate) trace_label: Option<String>,
    #[cfg(test)]
    pub(crate) active_rank: Option<usize>,
    #[cfg(test)]
    pub(crate) inactive_directions: Option<usize>,
    #[cfg(test)]
    pub(crate) exit_reason: String,
}

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

/// Covariance estimate for `varpar = c(theta, sigma)` plus Hessian diagnostics.
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq)]
pub struct VcovVarparEstimate {
    pub covariance: DMatrix<f64>,
    pub hessian: DMatrix<f64>,
    pub eigenvalues: Vec<f64>,
    pub tolerance: f64,
    pub positive_eigenvalues: usize,
    pub near_zero_eigenvalues: usize,
    pub negative_eigenvalues: usize,
    pub used_reduced_rank: bool,
    pub reliability: ReliabilityGrade,
    pub notes: Vec<String>,
}

/// First-order covariance-cone classification for covariance blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CovarianceKktClassification {
    /// The covariance block is interior and the covariance-space score is near zero.
    InteriorConverged,
    /// The variance is at zero and the covariance-space score supports the boundary.
    ValidZeroVariance,
    /// The covariance block is singular and the score supports the active face.
    ValidRankDeficientCovariance,
    /// The variance is at zero but the score indicates a feasible covariance increase.
    InvalidBoundaryStop,
    /// The covariance block is positive or near-boundary, but the local score is not decisive.
    WeakIdentification,
}

/// Per-term scalar covariance-cone KKT diagnostic.
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarCovarianceKktBlock {
    pub term_index: usize,
    pub theta_index: usize,
    pub term: String,
    pub theta: f64,
    pub variance: f64,
    pub score: f64,
    pub complementarity: f64,
    pub residual: f64,
    pub classification: CovarianceKktClassification,
}

/// Scalar covariance-cone KKT certificate for fitted LMM covariance blocks.
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarCovarianceKktCertificate {
    pub blocks: Vec<ScalarCovarianceKktBlock>,
    pub residual: f64,
    pub variance_tolerance: f64,
    pub score_tolerance: f64,
    pub objective: f64,
}

/// Per-term 2x2 covariance-cone KKT diagnostic.
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq)]
pub struct TwoByTwoCovarianceKktBlock {
    pub term_index: usize,
    pub theta_start_index: usize,
    pub term: String,
    pub theta: [f64; 3],
    pub covariance: [[f64; 2]; 2],
    pub score: [[f64; 2]; 2],
    pub min_eig_g: f64,
    pub min_eig_score: f64,
    pub complementarity: f64,
    pub residual: f64,
    pub classification: CovarianceKktClassification,
}

/// 2x2 covariance-cone KKT certificate for fitted LMM covariance blocks.
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq)]
pub struct TwoByTwoCovarianceKktCertificate {
    pub blocks: Vec<TwoByTwoCovarianceKktBlock>,
    pub residual: f64,
    pub covariance_tolerance: f64,
    pub score_tolerance: f64,
    pub complementarity_tolerance: f64,
    pub objective: f64,
}

/// Kenward-Roger response-covariance decomposition.
///
/// This is the Rust analogue of `pbkrtest::get_SigmaG()`: `sigma` is the
/// fitted marginal response covariance and each component matrix is a known
/// `G_i` such that `sigma = sum_i weights[i] * components[i]`.
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq)]
pub struct KenwardRogerSigmaG {
    pub sigma: DMatrix<f64>,
    pub components: Vec<DMatrix<f64>>,
    pub component_weights: Vec<f64>,
    pub component_labels: Vec<String>,
    pub residual_component_index: usize,
    pub covariance_parameterization: String,
    pub includes_residual_variance: bool,
    pub n_observations: usize,
    pub dense_bytes: u128,
    pub sigma_min_eigenvalue: f64,
    pub sigma_max_eigenvalue: f64,
    pub sigma_positive_definite: bool,
    pub max_component_asymmetry: f64,
    pub reliability: ReliabilityGrade,
    pub notes: Vec<String>,
}

/// Kenward-Roger adjusted fixed-effect covariance payload.
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq)]
pub struct KenwardRogerAdjustedVcov {
    pub unadjusted_vcov_active: DMatrix<f64>,
    pub adjusted_vcov_active: DMatrix<f64>,
    pub adjusted_vcov: DMatrix<f64>,
    pub p_matrices: Vec<DMatrix<f64>>,
    pub q_matrices: Vec<DMatrix<f64>>,
    pub w: DMatrix<f64>,
    pub information_matrix: DMatrix<f64>,
    pub information_eigenvalues: Vec<f64>,
    pub condition_min_abs_eigenvalue: f64,
    pub used_generalized_inverse: bool,
    pub component_labels: Vec<String>,
    pub reliability: ReliabilityGrade,
    pub notes: Vec<String>,
}

/// Kenward-Roger denominator degrees-of-freedom result for `L beta = rhs`.
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq)]
pub struct KenwardRogerLbDdf {
    pub denominator_df: f64,
    pub numerator_df: f64,
    pub restriction_rank: usize,
    pub a1: f64,
    pub a2: f64,
    pub b: f64,
    pub g: f64,
    pub rho: f64,
    pub used_generalized_inverse: bool,
    pub reliability: ReliabilityGrade,
    pub notes: Vec<String>,
}

/// Controls the bounded verification workflow run after a fitted model.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct ConvergenceVerificationOptions {
    pub restart_from_optimum: bool,
    pub jitter_starts: usize,
    pub jitter_scale: f64,
    pub run_optimizer_consensus: bool,
    pub max_function_evaluations: usize,
    pub objective_tolerance: f64,
    pub theta_tolerance: f64,
    pub beta_tolerance: f64,
}

impl Default for ConvergenceVerificationOptions {
    fn default() -> Self {
        Self {
            restart_from_optimum: true,
            jitter_starts: 1,
            jitter_scale: 1e-4,
            run_optimizer_consensus: true,
            max_function_evaluations: 500,
            objective_tolerance: 1e-5,
            theta_tolerance: 1e-3,
            beta_tolerance: 1e-4,
        }
    }
}

/// Estimation criterion for a linear mixed model fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ModelCriterion {
    /// Maximum likelihood (the default; equivalent to `fit(false)`).
    #[default]
    Ml,
    /// Restricted maximum likelihood (equivalent to `fit(true)`).
    Reml,
}

impl ModelCriterion {
    /// `true` for [`ModelCriterion::Reml`].
    pub fn is_reml(self) -> bool {
        matches!(self, ModelCriterion::Reml)
    }
}

/// Optional caller choice for the optimizer used by a fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum OptimizerChoice {
    /// Keep the fit driver's automatic optimizer selection.
    #[default]
    Auto,
    /// Request a specific optimizer. Unsupported choices return a typed error
    /// instead of silently falling back.
    Named(Optimizer),
}

impl OptimizerChoice {
    fn named(self) -> Option<Optimizer> {
        match self {
            OptimizerChoice::Auto => None,
            OptimizerChoice::Named(optimizer) => Some(optimizer),
        }
    }
}

/// Optional convergence-tolerance overrides for a fit.
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct FitToleranceOverrides {
    /// Relative tolerance on the objective.
    pub ftol_rel: Option<f64>,
    /// Absolute tolerance on the objective.
    pub ftol_abs: Option<f64>,
    /// Relative tolerance on optimizer parameters.
    pub xtol_rel: Option<f64>,
    /// Per-parameter absolute tolerance on optimizer parameters.
    pub xtol_abs: Option<Vec<f64>>,
    /// Per-parameter initial optimizer step size.
    pub initial_step: Option<Vec<f64>>,
}

impl FitToleranceOverrides {
    /// Set the relative objective tolerance.
    pub fn with_ftol_rel(mut self, value: f64) -> Self {
        self.ftol_rel = Some(value);
        self
    }

    /// Set the absolute objective tolerance.
    pub fn with_ftol_abs(mut self, value: f64) -> Self {
        self.ftol_abs = Some(value);
        self
    }

    /// Set the relative parameter tolerance.
    pub fn with_xtol_rel(mut self, value: f64) -> Self {
        self.xtol_rel = Some(value);
        self
    }

    /// Set per-parameter absolute tolerances.
    pub fn with_xtol_abs(mut self, values: Vec<f64>) -> Self {
        self.xtol_abs = Some(values);
        self
    }

    /// Set per-parameter initial optimizer steps.
    pub fn with_initial_step(mut self, values: Vec<f64>) -> Self {
        self.initial_step = Some(values);
        self
    }
}

/// Narrow, audit-recorded caller control over optimizer setup.
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct OptimizerControl {
    /// Optimizer selection. Defaults to [`OptimizerChoice::Auto`].
    pub optimizer: OptimizerChoice,
    /// Optional convergence-tolerance overrides.
    pub tolerances: FitToleranceOverrides,
    /// Optional warm-start theta vector.
    pub start_theta: Option<Vec<f64>>,
    /// Optional maximum number of optimizer function evaluations.
    pub max_feval: Option<usize>,
}

impl OptimizerControl {
    /// Driver-selected optimizer and default tolerances.
    pub fn auto() -> Self {
        Self::default()
    }

    /// Request a specific optimizer.
    pub fn with_optimizer(mut self, optimizer: Optimizer) -> Self {
        self.optimizer = OptimizerChoice::Named(optimizer);
        self
    }

    /// Attach tolerance overrides.
    pub fn with_tolerances(mut self, tolerances: FitToleranceOverrides) -> Self {
        self.tolerances = tolerances;
        self
    }

    /// Set the warm-start theta vector.
    pub fn with_start_theta(mut self, start_theta: Vec<f64>) -> Self {
        self.start_theta = Some(start_theta);
        self
    }

    /// Set the maximum number of optimizer function evaluations.
    pub fn with_max_feval(mut self, max_feval: usize) -> Self {
        self.max_feval = Some(max_feval);
        self
    }

    fn caller_set_fields(&self) -> Vec<String> {
        let mut fields = Vec::new();
        if self.optimizer.named().is_some() {
            fields.push("optimizer".to_string());
        }
        if self.tolerances.ftol_rel.is_some() {
            fields.push("ftol_rel".to_string());
        }
        if self.tolerances.ftol_abs.is_some() {
            fields.push("ftol_abs".to_string());
        }
        if self.tolerances.xtol_rel.is_some() {
            fields.push("xtol_rel".to_string());
        }
        if self.tolerances.xtol_abs.is_some() {
            fields.push("xtol_abs".to_string());
        }
        if self.tolerances.initial_step.is_some() {
            fields.push("initial_step".to_string());
        }
        if self.start_theta.is_some() {
            fields.push("start_theta".to_string());
        }
        if self.max_feval.is_some() {
            fields.push("max_feval".to_string());
        }
        fields
    }
}

/// Options controlling how a model is fit.
///
/// By default the fit driver chooses the optimizer. [`OptimizerControl`] is a
/// narrow opt-in escape hatch for recourse, warm starts, and explicit
/// tolerance overrides; any supplied field is recorded in the optimizer
/// certificate.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct FitOptions {
    /// ML or REML. Defaults to [`ModelCriterion::Ml`].
    pub criterion: ModelCriterion,
    /// Optional audit-recorded optimizer controls.
    pub optimizer_control: OptimizerControl,
}

impl FitOptions {
    /// Options requesting a maximum-likelihood fit.
    pub fn ml() -> Self {
        Self {
            criterion: ModelCriterion::Ml,
            optimizer_control: OptimizerControl::default(),
        }
    }

    /// Options requesting a restricted-maximum-likelihood fit.
    pub fn reml() -> Self {
        Self {
            criterion: ModelCriterion::Reml,
            optimizer_control: OptimizerControl::default(),
        }
    }

    /// Attach optimizer controls to these fit options.
    pub fn with_optimizer_control(mut self, control: OptimizerControl) -> Self {
        self.optimizer_control = control;
        self
    }

    /// Request a specific optimizer while leaving other controls at default.
    pub fn with_optimizer(mut self, optimizer: Optimizer) -> Self {
        self.optimizer_control = self.optimizer_control.with_optimizer(optimizer);
        self
    }
}

fn validate_positive_control_value(name: &str, value: f64) -> Result<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(MixedModelError::InvalidArgument(format!(
            "{name} override must be finite and positive"
        )))
    }
}

fn validate_control_vector(name: &str, values: &[f64], n_theta: usize) -> Result<()> {
    if values.len() != n_theta {
        return Err(MixedModelError::InvalidArgument(format!(
            "{name} length {} does not match theta length {n_theta}",
            values.len()
        )));
    }
    if values.iter().all(|value| value.is_finite() && *value > 0.0) {
        Ok(())
    } else {
        Err(MixedModelError::InvalidArgument(format!(
            "{name} values must be finite and positive"
        )))
    }
}

/// Fluent builder for [`LinearMixedModel`].
///
/// Collapses construction (`new` / weights / compiler policy) and the
/// `fit(reml: bool)` boolean into a single chained surface:
///
/// ```
/// use mixeff_rs::formula::parse_formula;
/// use mixeff_rs::model::{DataFrame, FitOptions, LinearMixedModelBuilder, MixedModelFit};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut df = DataFrame::new();
/// df.add_numeric("y", vec![1.0, 2.1, 3.0, 4.2, 5.1, 6.0])?;
/// df.add_numeric("x", vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0])?;
/// df.add_categorical(
///     "g",
///     vec!["a", "a", "b", "b", "c", "c"].into_iter().map(str::to_string).collect(),
/// )?;
///
/// let model = LinearMixedModelBuilder::new(parse_formula("y ~ 1 + x + (1 | g)")?, &df)
///     .fit(FitOptions::reml())?;
/// assert_eq!(model.coef().len(), 2);
/// # Ok(())
/// # }
/// ```
pub struct LinearMixedModelBuilder<'a> {
    formula: Formula,
    data: &'a DataFrame,
    weights: Option<Vec<f64>>,
    compiler_policy: Option<CompilerPolicy>,
}

impl<'a> LinearMixedModelBuilder<'a> {
    /// Start a builder for `formula` over `data`.
    pub fn new(formula: Formula, data: &'a DataFrame) -> Self {
        Self {
            formula,
            data,
            weights: None,
            compiler_policy: None,
        }
    }

    /// Attach per-observation weights.
    pub fn weights(mut self, weights: Vec<f64>) -> Self {
        self.weights = Some(weights);
        self
    }

    /// Attach a compiler policy applied to the internal compiled artifact.
    pub fn compiler_policy(mut self, compiler_policy: CompilerPolicy) -> Self {
        self.compiler_policy = Some(compiler_policy);
        self
    }

    /// Construct the (unfitted) model.
    pub fn build(self) -> Result<LinearMixedModel> {
        let mut model = LinearMixedModel::new(self.formula, self.data, self.weights.as_deref())?;
        if let Some(policy) = self.compiler_policy {
            model.set_compiler_policy(policy)?;
        }
        Ok(model)
    }

    /// Construct and fit the model in one step.
    pub fn fit(self, options: FitOptions) -> Result<LinearMixedModel> {
        let mut model = self.build()?;
        model.fit_with_options(options)?;
        Ok(model)
    }
}

impl LinearMixedModel {
    /// Construct a LinearMixedModel from a formula, data, and optional weights.
    pub fn new(formula: Formula, data: &DataFrame, weights: Option<&[f64]>) -> Result<Self> {
        Self::new_with_policy_internal(formula, data, weights, CompilerPolicy::default())
    }

    fn new_with_policy_internal(
        formula: Formula,
        data: &DataFrame,
        weights: Option<&[f64]>,
        compiler_policy: CompilerPolicy,
    ) -> Result<Self> {
        Self::new_with_policies_internal(
            formula,
            data,
            weights,
            compiler_policy,
            FixedDesignBuildPolicy::default(),
        )
    }

    #[cfg(test)]
    fn new_with_fixed_design_policy(
        formula: Formula,
        data: &DataFrame,
        weights: Option<&[f64]>,
        fixed_design_policy: FixedDesignBuildPolicy,
    ) -> Result<Self> {
        Self::new_with_policies_internal(
            formula,
            data,
            weights,
            CompilerPolicy::default(),
            fixed_design_policy,
        )
    }

    fn new_with_policies_internal(
        formula: Formula,
        data: &DataFrame,
        weights: Option<&[f64]>,
        compiler_policy: CompilerPolicy,
        fixed_design_policy: FixedDesignBuildPolicy,
    ) -> Result<Self> {
        if formula.random_terms.is_empty() {
            return Err(MixedModelError::NoRandomEffects);
        }

        // Data-boundary seam: lower the stateless in-formula transforms
        // (`I(days^2)`, `log(reaction)`, …) into synthetic numeric columns
        // before any design construction. Above this point everything keeps
        // seeing "a column by name"; the formula's term/response references
        // already carry the canonical labels. See
        // `docs/formula_transform_seam.md`.
        let materialized = formula.materialize(data)?;
        let data = &materialized;

        let semantic_model = compile_formula_ir(&formula);
        let mut compiler_artifact = CompiledModelArtifact::new_with_policy(
            formula.to_string(),
            semantic_model,
            compiler_policy,
        );
        compiler_artifact.attach_design_audit(data);
        let mut effective_formula = formula.clone();
        if compiler_artifact
            .compiler_policy
            .apply_design_time_reductions
        {
            let reductions = apply_design_compiled_policy(
                &mut effective_formula,
                &compiler_artifact.policy_recommendations,
            )?;
            if !reductions.is_empty() {
                let effective_semantic_model = compile_formula_ir(&effective_formula);
                compiler_artifact.set_effective_model(
                    effective_formula.to_string(),
                    effective_semantic_model,
                    reductions,
                );
            }
        }
        if effective_formula.random_terms.is_empty() {
            return Err(MixedModelError::NoRandomEffects);
        }
        refuse_unsupported_random_covariance(&effective_formula)?;

        let n = data.nrow();

        // Build the response vector
        let y_data = data.numeric(&effective_formula.response).ok_or_else(|| {
            MixedModelError::InvalidArgument(format!(
                "Response '{}' not found or not numeric",
                effective_formula.response
            ))
        })?;
        let y = DVector::from_column_slice(y_data);

        // Build the fixed-effects design through the backend-selection policy.
        // FeTerm still owns rank/pivot metadata; the selected full-rank backend
        // is used below for solver cross-products.
        let raw_fixed_design =
            if use_direct_dense_fixed_design(&effective_formula, data, fixed_design_policy) {
                FixedDesign::Dense(build_fixed_effects_design(&effective_formula, data)?)
            } else {
                crate::model::fixed_design::build_fixed_effects_design_with_policy(
                    &effective_formula,
                    data,
                    fixed_design_policy,
                )?
            };
        let feterm = FeTerm::new(
            raw_fixed_design.materialize_dense(),
            raw_fixed_design.column_names().to_vec(),
        );
        let fixed_design = raw_fixed_design.select_columns(&feterm.piv[..feterm.rank])?;
        if fixed_design.storage() == FixedDesignStorage::Streamed {
            compiler_artifact
                .diagnostics
                .push(fixed_design_backend_diagnostic(&fixed_design));
        }

        // Build the random-effects terms
        let mut ordered_reterms = Vec::new();
        for (semantic_index, rt) in effective_formula.random_terms.iter().enumerate() {
            let remat = build_re_mat(rt, data, n)?;
            ordered_reterms.push((semantic_index, remat));
        }

        // Sort by decreasing nranef (matches Julia behavior)
        ordered_reterms.sort_by(|a, b| b.1.n_ranef().cmp(&a.1.n_ranef()));
        let optimizer_order = ordered_reterms
            .iter()
            .map(|(semantic_index, _)| *semantic_index)
            .collect::<Vec<_>>();
        let optimizer_basis = ordered_reterms
            .iter()
            .map(|(_, remat)| {
                remat
                    .cnames
                    .iter()
                    .map(|name| user_basis_label(name))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        compiler_artifact
            .rebuild_theta_maps_for_optimizer_order_with_basis(&optimizer_order, &optimizer_basis);
        let mut reterms = ordered_reterms
            .into_iter()
            .map(|(_, remat)| remat)
            .collect::<Vec<_>>();

        // Build FeMat = [full_rank_X | y]
        let mut xy_mat = FeMat::new(&feterm, &y);

        // Apply weights: scale each row of X, Z, and y by sqrt(w_i).
        let mut sqrtwts_dvec = None;
        let sqrtwts = if let Some(wts) = weights {
            let sw: Vec<f64> = wts.iter().map(|w| w.sqrt()).collect();
            let sw_dvec = DVector::from_vec(sw.clone());
            xy_mat.reweight(&sw_dvec);
            for rt in &mut reterms {
                rt.reweight(&sw_dvec);
            }
            sqrtwts_dvec = Some(sw_dvec);
            sw
        } else {
            vec![]
        };

        // Create cross-product blocks A and Cholesky blocks L
        let (a_blocks, l_blocks) =
            create_al_from_fixed_design(&reterms, &fixed_design, &y, sqrtwts_dvec.as_ref())?;

        // Build theta vector from all reterms
        let theta: Vec<f64> = reterms.iter().flat_map(|rt| rt.get_theta()).collect();

        // Build parmap: mapping from θ index to (re_term_index, row, col) in lambda
        let parmap = build_parmap(&reterms);

        let dims = ModelDims {
            n,
            p: feterm.rank,
            nretrms: reterms.len(),
        };

        let optsum = OptSummary::new(theta);

        let training_categorical = snapshot_training_categorical(data);

        let mut model = LinearMixedModel {
            formula: effective_formula,
            reterms,
            xy_mat,
            y,
            feterm,
            fixed_design,
            sqrtwts,
            parmap,
            dims,
            a_blocks,
            l_blocks,
            optsum,
            compiler_artifact,
            residual_source: crate::model::summary_estimates::ResidualSource::EstimatedSigma,
            training_categorical,
        };
        debug_assert_eq!(
            model.dims.p, model.feterm.rank,
            "ModelDims::p must track the active fixed-effect rank"
        );
        model.refresh_covariance_parameter_traces();
        Ok(model)
    }

    /// Construct a model and apply a compiler policy before any fitting or
    /// certification occurs.
    pub fn new_with_compiler_policy(
        formula: Formula,
        data: &DataFrame,
        weights: Option<&[f64]>,
        compiler_policy: CompilerPolicy,
    ) -> Result<Self> {
        Self::new_with_policy_internal(formula, data, weights, compiler_policy)
    }

    /// Apply a compiler policy before fitting.
    ///
    /// Policies affect advisory recommendations, reproducibility metadata, and
    /// fit-time certification such as effective covariance rank. Changing the
    /// policy after a fit would make the certificate ambiguous, so fitted models
    /// reject this operation.
    pub fn set_compiler_policy(&mut self, compiler_policy: CompilerPolicy) -> Result<&mut Self> {
        if self.optsum.feval > 0 {
            return Err(MixedModelError::AlreadyFitted);
        }
        self.compiler_artifact.set_compiler_policy(compiler_policy);
        Ok(self)
    }

    /// Return a copy of this model with a compiler policy applied.
    pub fn with_compiler_policy(mut self, compiler_policy: CompilerPolicy) -> Result<Self> {
        self.set_compiler_policy(compiler_policy)?;
        Ok(self)
    }

    /// Fit after first applying a compiler policy.
    pub fn fit_with_compiler_policy(
        &mut self,
        reml: bool,
        compiler_policy: CompilerPolicy,
    ) -> Result<&mut Self> {
        self.set_compiler_policy(compiler_policy)?;
        self.fit(reml)
    }

    /// Round-trippable compiler artifact attached at construction time.
    pub fn compiler_artifact(&self) -> &CompiledModelArtifact {
        &self.compiler_artifact
    }

    unstable_internal_method! {
    /// Mutable compiler artifact (model-IR / audit metadata).
    ///
    /// Unstable internal surface: `pub` only with the `unstable-internals`
    /// feature; otherwise `pub(crate)`. The compiler/IR is still in flux and
    /// is not part of the stable 1.0 API.
    #[allow(dead_code)]
    unstable_vis fn compiler_artifact_mut(&mut self) -> &mut CompiledModelArtifact {
        &mut self.compiler_artifact
    }
    }

    /// Compiler policy attached to this model.
    pub fn compiler_policy(&self) -> &CompilerPolicy {
        &self.compiler_artifact.compiler_policy
    }

    /// Map from each θ index to its `(block, row, col)` slot in the relative
    /// covariance factor.
    pub fn parmap(&self) -> &[(usize, usize, usize)] {
        &self.parmap
    }

    /// Active fixed-effect rank after pivoting and rank detection.
    pub fn fixed_effect_rank(&self) -> usize {
        self.feterm.rank
    }

    unstable_internal_method! {
    /// Lower triangle of `[Z X y]'[Z X y]` in blocked storage (raw PLS
    /// solver state).
    ///
    /// Unstable internal surface: `pub` only with the `unstable-internals`
    /// feature; otherwise `pub(crate)`. Leaks the blocked-Cholesky layout and
    /// is not part of the stable 1.0 API.
    #[allow(dead_code)]
    unstable_vis fn a_blocks(&self) -> &[MatrixBlock] {
        &self.a_blocks
    }
    }

    unstable_internal_method! {
    /// Mutable lower triangle of `[Z X y]'[Z X y]` (raw PLS solver state).
    ///
    /// Unstable internal surface: `pub` only with the `unstable-internals`
    /// feature; otherwise `pub(crate)`. Not part of the stable 1.0 API.
    #[allow(dead_code)]
    unstable_vis fn a_blocks_mut(&mut self) -> &mut [MatrixBlock] {
        &mut self.a_blocks
    }
    }

    unstable_internal_method! {
    /// Blocked lower Cholesky factor of `Λ'AΛ + I` (raw PLS solver state).
    ///
    /// Unstable internal surface: `pub` only with the `unstable-internals`
    /// feature; otherwise `pub(crate)`. Leaks the blocked-Cholesky layout and
    /// is not part of the stable 1.0 API.
    #[allow(dead_code)]
    unstable_vis fn l_blocks(&self) -> &[MatrixBlock] {
        &self.l_blocks
    }
    }

    unstable_internal_method! {
    /// Mutable blocked lower Cholesky factor `Λ'AΛ + I` (raw PLS solver
    /// state).
    ///
    /// Unstable internal surface: `pub` only with the `unstable-internals`
    /// feature; otherwise `pub(crate)`. Not part of the stable 1.0 API.
    #[allow(dead_code)]
    unstable_vis fn l_blocks_mut(&mut self) -> &mut [MatrixBlock] {
        &mut self.l_blocks
    }
    }

    unstable_internal_method! {
    /// Disjoint mutable `l_blocks` and immutable `a_blocks` borrows (raw PLS
    /// solver state), so kernels can update `L` in place from `A` without
    /// fighting the borrow checker.
    ///
    /// Unstable internal surface: `pub` only with the `unstable-internals`
    /// feature; otherwise `pub(crate)`. Not part of the stable 1.0 API.
    #[allow(dead_code)]
    unstable_vis fn l_blocks_mut_a_blocks(&mut self) -> (&mut [MatrixBlock], &[MatrixBlock]) {
        (&mut self.l_blocks, &self.a_blocks)
    }
    }

    /// Runtime summary for the selected fixed-effect design backend.
    pub fn fixed_design_backend_summary(&self) -> FixedDesignSummary {
        self.fixed_design.summary()
    }

    /// Number of active fixed-effect entries stored by the selected backend.
    ///
    /// Dense designs report `n * p`; streamed designs report the actual
    /// non-zero row entries stored after rank/pivot column selection.
    pub fn fixed_design_active_entries(&self) -> usize {
        fixed_design_active_entries(&self.fixed_design)
    }

    /// Active-entry density of the selected fixed-effect backend.
    pub fn fixed_design_density(&self) -> f64 {
        fixed_design_density(&self.fixed_design)
    }

    /// Prefit design audit attached to the compiler artifact, if available.
    pub fn design_audit(&self) -> Option<&DesignAudit> {
        self.compiler_artifact.design_audit.as_ref()
    }

    /// Fit-time optimizer certificate attached to the compiler artifact, if available.
    pub fn optimizer_certificate(&self) -> Option<&OptimizerCertificate> {
        self.compiler_artifact.optimizer_certificate.as_ref()
    }

    /// Stable user-facing audit report derived from the compiler artifact.
    pub fn audit_report(&self) -> ModelAuditReport {
        self.compiler_artifact.audit_report()
    }

    /// Compact default print summary (PRD § 15).
    pub fn print_summary(&self) -> crate::compiler::ModelPrint {
        self.compiler_artifact.print_summary()
    }

    /// Source-to-fitted parameterization drilldown (PRD § 15).
    pub fn parameterization(&self) -> crate::compiler::ParameterizationDrilldown {
        self.compiler_artifact.parameterization()
    }

    /// Requested, semantic, supported, and fitted model-state view.
    pub fn model_state_summary(&self) -> ModelStateSummary {
        self.compiler_artifact.model_state_summary()
    }

    /// Recorded or recommended requested-to-fitted model changes.
    pub fn changes(&self) -> Vec<ModelStateChange> {
        self.compiler_artifact.changes()
    }

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
    /// `next: run verify_convergence()` hint — that is, when the
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

    fn fit_with_forced_optimizer(&mut self, reml: bool, optimizer: Optimizer) -> Result<()> {
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

    fn refresh_optimizer_certificate(&mut self) {
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

    fn refresh_covariance_parameter_traces(&mut self) {
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

    fn refresh_effective_covariance_summaries(&mut self) {
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
                    .zip(oriented.into_iter())
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

    /// Get the response vector y (last column of xy_mat).
    pub fn y(&self) -> DVector<f64> {
        self.y.clone()
    }

    /// Borrow the response vector `y` without cloning.
    ///
    /// Read-only view of the same data returned (by value) from
    /// [`LinearMixedModel::y`]; prefer this when an owned copy is unnecessary.
    pub fn y_ref(&self) -> &DVector<f64> {
        &self.y
    }

    /// Borrow the model formula.
    ///
    /// The fitted model owns the parsed [`Formula`]; it is exposed read-only
    /// because mutating it post-fit would silently desynchronize every derived
    /// quantity (design matrices, β, vcov, …).
    pub fn formula(&self) -> &Formula {
        &self.formula
    }

    /// Borrow the random-effects terms.
    ///
    /// Read-only: the per-term Λ_θ/Z/refs are part of the fitted state and must
    /// not be mutated externally without re-fitting.
    pub fn reterms(&self) -> &[ReMat] {
        &self.reterms
    }

    /// Borrow the model dimensions (`n`, `p`, `nretrms`).
    pub fn dims(&self) -> &ModelDims {
        &self.dims
    }

    /// Borrow the optimization summary.
    ///
    /// Read-only mirror of [`MixedModelFit::opt_summary`]; mutating optimizer
    /// state after a fit invalidates convergence diagnostics.
    pub fn optsum(&self) -> &OptSummary {
        &self.optsum
    }

    // The method inside the macro carries its own docs when it is public.
    unstable_internal_method! {
    /// Mutable optimizer summary.
    ///
    /// **Unstable internal surface:** `pub` only with the
    /// `unstable-internals` feature; otherwise `pub(crate)`. This escape
    /// hatch exists solely so in-repo benchmarks and tuning harnesses can
    /// configure optimizer tolerances and the initial θ *before* calling
    /// [`LinearMixedModel::fit`]. Mutating `optsum` *after* a fit silently
    /// desynchronizes convergence diagnostics; there is no supported reason
    /// to do so. Not part of the stable 1.0 API and exempt from SemVer.
    #[allow(dead_code)]
    unstable_vis fn optsum_mut(&mut self) -> &mut OptSummary {
        &mut self.optsum
    }
    }

    pub(crate) fn apply_optimizer_control(&mut self, control: &OptimizerControl) -> Result<()> {
        let n_theta = self.n_theta();

        if let Some(value) = control.tolerances.ftol_rel {
            validate_positive_control_value("ftol_rel", value)?;
            self.optsum.ftol_rel = value;
        }
        if let Some(value) = control.tolerances.ftol_abs {
            validate_positive_control_value("ftol_abs", value)?;
            self.optsum.ftol_abs = value;
        }
        if let Some(value) = control.tolerances.xtol_rel {
            validate_positive_control_value("xtol_rel", value)?;
            self.optsum.xtol_rel = value;
        }
        if let Some(values) = &control.tolerances.xtol_abs {
            validate_control_vector("xtol_abs", values, n_theta)?;
            self.optsum.xtol_abs = values.clone();
        }
        if let Some(values) = &control.tolerances.initial_step {
            validate_control_vector("initial_step", values, n_theta)?;
            self.optsum.initial_step = values.clone();
        }
        if let Some(max_feval) = control.max_feval {
            if max_feval == 0 {
                return Err(MixedModelError::InvalidArgument(
                    "max_feval override must be positive".to_string(),
                ));
            }
            self.optsum.max_feval = i64::try_from(max_feval).map_err(|_| {
                MixedModelError::InvalidArgument(
                    "max_feval override exceeds the supported integer range".to_string(),
                )
            })?;
        }
        if let Some(start_theta) = &control.start_theta {
            if start_theta.len() != n_theta {
                return Err(MixedModelError::InvalidArgument(format!(
                    "start_theta length {} does not match theta length {n_theta}",
                    start_theta.len()
                )));
            }
            if !start_theta.iter().all(|value| value.is_finite()) {
                return Err(MixedModelError::InvalidArgument(
                    "start_theta values must be finite".to_string(),
                ));
            }
            for (index, (&value, &lower)) in start_theta
                .iter()
                .zip(self.lower_bounds().iter())
                .enumerate()
            {
                if lower.is_finite() && value < lower {
                    return Err(MixedModelError::InvalidArgument(format!(
                        "start_theta[{index}] = {value} is below lower bound {lower}"
                    )));
                }
            }
            self.optsum.initial = start_theta.clone();
            self.optsum.final_params = start_theta.clone();
            self.set_theta(start_theta)?;
            self.update_l()?;
        }

        if let Some(optimizer) = control.optimizer.named() {
            self.optsum.optimizer = optimizer;
            self.optsum.backend = optimizer.canonical_backend();
            self.optsum.optimizer_source = OptimizerSource::Caller;
        } else {
            self.optsum.optimizer_source = OptimizerSource::Auto;
        }
        self.optsum.caller_set_fields = control.caller_set_fields();
        Ok(())
    }

    /// Get the current θ parameter vector.
    pub fn theta(&self) -> Vec<f64> {
        self.reterms.iter().flat_map(|rt| rt.get_theta()).collect()
    }

    /// Set the θ parameter vector, distributing values to each ReMat's λ.
    pub fn set_theta(&mut self, theta: &[f64]) -> Result<()> {
        let expected = self.n_theta();
        if theta.len() != expected {
            return Err(MixedModelError::DimensionMismatch(format!(
                "theta vector length mismatch: expected {expected}, got {}",
                theta.len()
            )));
        }

        let mut offset = 0;
        for rt in &mut self.reterms {
            let n = rt.n_theta();
            rt.set_theta(&theta[offset..offset + n])?;
            offset += n;
        }
        Ok(())
    }

    /// Lower bounds on θ. Diagonal elements of λ are ≥ 0, off-diagonal are unconstrained.
    pub fn lower_bounds(&self) -> Vec<f64> {
        let mut lb = Vec::new();
        for (_, row, col) in &self.parmap {
            if row == col {
                lb.push(0.0); // diagonal elements are non-negative
            } else {
                lb.push(f64::NEG_INFINITY);
            }
        }
        lb
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
        let objective = self.objective_at_theta_for_certificate(&theta)?;
        let variance_tolerance = 1e-8;
        let score_tolerance = (1e-5 * (1.0 + objective.abs())).max(1e-6);
        let mut blocks = Vec::with_capacity(self.reterms.len());

        for (term_index, term) in self.reterms.iter().enumerate() {
            let theta_index = term_index;
            let theta_value = theta[theta_index].max(0.0);
            let variance = theta_value * theta_value;
            let score = self.scalar_covariance_score(theta_index, &theta, objective)?;
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

    fn objective_at_theta_for_certificate(&self, theta: &[f64]) -> Result<f64> {
        let mut evaluator = self.clone();
        evaluator.objective_at(theta)
    }

    fn scalar_covariance_score(
        &self,
        theta_index: usize,
        theta: &[f64],
        objective: f64,
    ) -> Result<f64> {
        let variance = theta[theta_index].max(0.0).powi(2);
        let mut step = scalar_covariance_variance_step(variance);

        for _ in 0..8 {
            let plus = self.objective_at_scalar_variance(theta, theta_index, variance + step);
            if variance > 1.5 * step {
                let minus_variance = variance - step;
                if let (Ok(f_plus), Ok(f_minus)) = (
                    plus,
                    self.objective_at_scalar_variance(theta, theta_index, minus_variance),
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
        &self,
        theta: &[f64],
        theta_index: usize,
        variance: f64,
    ) -> Result<f64> {
        let mut trial = theta.to_vec();
        trial[theta_index] = variance.max(0.0).sqrt();
        self.objective_at_theta_for_certificate(&trial)
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
        let objective = self.objective_at_theta_for_certificate(&theta)?;
        let covariance_tolerance = 1e-8;
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
            let score =
                self.two_by_two_covariance_score(theta_start_index, &theta, objective, covariance)?;
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
        &self,
        theta_start_index: usize,
        theta: &[f64],
        objective: f64,
        covariance: [[f64; 2]; 2],
    ) -> Result<[[f64; 2]; 2]> {
        let e1 = [[1.0, 0.0], [0.0, 0.0]];
        let e2 = [[0.0, 0.0], [0.0, 1.0]];
        let plus = [[0.5, 0.5], [0.5, 0.5]];
        let minus = [[0.5, -0.5], [-0.5, 0.5]];

        let s00 = self.two_by_two_directional_covariance_score(
            theta_start_index,
            theta,
            objective,
            covariance,
            e1,
        )?;
        let s11 = self.two_by_two_directional_covariance_score(
            theta_start_index,
            theta,
            objective,
            covariance,
            e2,
        )?;
        let d_plus = self.two_by_two_directional_covariance_score(
            theta_start_index,
            theta,
            objective,
            covariance,
            plus,
        )?;
        let d_minus = self.two_by_two_directional_covariance_score(
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
        &self,
        theta_start_index: usize,
        theta: &[f64],
        objective: f64,
        covariance: [[f64; 2]; 2],
        direction: [[f64; 2]; 2],
    ) -> Result<f64> {
        let mut step = two_by_two_covariance_step(covariance);

        for _ in 0..8 {
            let plus_cov = two_by_two_add_direction(covariance, direction, step);
            let plus = self.objective_at_two_by_two_covariance(theta, theta_start_index, plus_cov);
            let minus_cov = two_by_two_add_direction(covariance, direction, -step);

            if two_by_two_theta_from_covariance(minus_cov).is_some() {
                if let (Ok(f_plus), Ok(f_minus)) = (
                    plus,
                    self.objective_at_two_by_two_covariance(theta, theta_start_index, minus_cov),
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
        &self,
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
        self.objective_at_theta_for_certificate(&trial)
    }

    fn trust_bq_covariance_kkt_certifies_theta(
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

    fn apply_kkt_guided_boundary_restart(&mut self, reml: bool) -> Result<bool> {
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
        let certificate = self.scalar_covariance_kkt_certificate()?;
        let base_theta = self.theta();
        let base_objective = certificate.objective;
        let mut best_theta = base_theta.clone();
        let mut best_objective = base_objective;
        let mut reason = None;

        for block in certificate.blocks.iter().filter(|block| {
            block.classification == CovarianceKktClassification::InvalidBoundaryStop
        }) {
            let scale = 1.0 + block.variance.abs().max((-block.score).max(0.0));
            for delta in kkt_restart_delta_grid(scale) {
                let mut trial = base_theta.clone();
                trial[block.theta_index] = delta.sqrt();
                let objective = self.objective_at_theta_for_certificate(&trial)?;
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
        let certificate = self.two_by_two_covariance_kkt_certificate()?;
        let base_theta = self.theta();
        let base_objective = certificate.objective;
        let mut best_theta = base_theta.clone();
        let mut best_objective = base_objective;
        let mut reason = None;

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
                let objective = self.objective_at_theta_for_certificate(&trial)?;
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

    fn theta_at_lower_bound(&self) -> bool {
        let theta = self.theta();
        let lb = self.lower_bounds();
        let boundary_tolerance = self.optsum.xtol_zero_abs.max(1e-12) * 10.0;
        theta.iter().zip(lb.iter()).any(|(&value, &lower)| {
            lower.is_finite() && (value - lower).abs() <= boundary_tolerance
        })
    }

    fn optimizer_certificate_reports_boundary(&self) -> bool {
        self.compiler_artifact
            .optimizer_certificate
            .as_ref()
            .is_some_and(|certificate| certificate.evidence.parameter_space.n_boundary > 0)
    }

    fn has_reduced_effective_covariance(&self) -> bool {
        self.compiler_artifact
            .effective_covariance
            .iter()
            .any(|summary| summary.status == EffectiveRankStatus::ReducedRank)
    }

    unstable_internal_method! {
    /// Update the blocked Cholesky factor L from A and current λ values.
    ///
    /// This is the core operation: L = cholesky(Λ'AΛ + I).
    /// The blocks of L are updated in-place.
    ///
    /// Unstable internal surface: `pub` only with the `unstable-internals`
    /// feature; otherwise `pub(crate)`. Not covered by the 1.0 SemVer
    /// guarantee.
    unstable_vis fn update_l(&mut self) -> Result<()> {
        let cholesky_zero_pad_tolerance = self
            .compiler_policy()
            .thresholds
            .cholesky_zero_pad_tolerance;
        update_l_from_parts(
            &self.a_blocks,
            &mut self.l_blocks,
            &self.reterms,
            cholesky_zero_pad_tolerance,
        )
    }
    }

    /// Update IRLS weights and working response, then rebuild A blocks.
    /// Called at each PIRLS iteration of a GLMM.
    ///
    /// * `sqrtwts` - square-root of the IRLS weights (length n)
    /// * `working_y` - working response values (length n)
    pub fn update_irls_weights(&mut self, sqrtwts: &[f64], working_y: &[f64]) -> Result<()> {
        let n = self.dims.n;
        debug_assert_eq!(sqrtwts.len(), n);
        debug_assert_eq!(working_y.len(), n);

        self.sqrtwts = sqrtwts.to_vec();

        // Update wtz for every RE term: wtz[s, obs] = sqrtwts[obs] * z[s, obs]
        for rt in &mut self.reterms {
            let vsize = rt.vsize;
            for obs in 0..n {
                for s in 0..vsize {
                    rt.wtz[(s, obs)] = sqrtwts[obs] * rt.z[(s, obs)];
                }
            }
        }

        // Update wtxy: first `rank` columns from X, last column from working_y
        let rank = self.feterm.rank;
        for obs in 0..n {
            let sw = sqrtwts[obs];
            for col in 0..rank {
                self.xy_mat.wtxy[(obs, col)] = sw * self.feterm.x[(obs, col)];
            }
            // y column (last)
            self.xy_mat.wtxy[(obs, rank)] = sw * working_y[obs];
            self.xy_mat.xy[(obs, rank)] = working_y[obs];
        }

        // Rebuild A blocks
        self.recompute_a_blocks()?;
        Ok(())
    }

    unstable_internal_method! {
    /// Recompute all A-block cross products from the current wtz / wtxy.
    /// Does NOT rebuild L — call `update_l()` after this.
    ///
    /// Unstable internal surface: `pub` only with the `unstable-internals`
    /// feature; otherwise `pub(crate)`.
    unstable_vis fn recompute_a_blocks(&mut self) -> Result<()> {
        let k = self.reterms.len();
        let mut idx = 0;
        let sqrtwts = if self.sqrtwts.is_empty() {
            None
        } else {
            Some(DVector::from_column_slice(&self.sqrtwts))
        };
        let weighted_fixed_design =
            weighted_fixed_design_for_solver(&self.fixed_design, sqrtwts.as_ref())?;
        let weighted_response = self.xy_mat.wtxy.column(self.feterm.rank).into_owned();

        // RE × RE blocks
        for i in 0..k {
            for j in 0..=i {
                let block = if i == j {
                    compute_re_cross_product(&self.reterms[i], &self.reterms[i])
                } else {
                    compute_re_cross_product(&self.reterms[i], &self.reterms[j])
                };
                self.a_blocks[idx] = block;
                idx += 1;
            }
        }

        // FE × RE blocks: [X|y]' Z_j
        for j in 0..k {
            let block = compute_fixed_response_re_cross_product(
                &weighted_fixed_design,
                &weighted_response,
                &self.reterms[j],
            )?;
            self.a_blocks[idx] = block;
            idx += 1;
        }

        // FE × FE block: [X|y]' [X|y]
        self.a_blocks[idx] = MatrixBlock::Dense(compute_fixed_response_cross_product(
            &weighted_fixed_design,
            &weighted_response,
        )?);

        Ok(())
    }
    }

    fn determinant_term_and_pwrss_for_reml(&self, reml: bool) -> (f64, f64) {
        let k = self.reterms.len();

        let mut logdet = 0.0;
        for j in 0..k {
            logdet += logdet_block(&self.l_blocks[block_index(j, j)]);
        }

        let l_dense = self.l_blocks[block_index(k, k)].as_dense();
        let pp1 = l_dense.nrows();
        let last_diag = l_dense[(pp1 - 1, pp1 - 1)];
        let pwrss = last_diag * last_diag;

        if reml {
            let mut logdet_lxx = 0.0;
            for i in 0..(pp1 - 1) {
                let d = l_dense[(i, i)];
                if d > 0.0 {
                    logdet_lxx += d.ln();
                }
            }
            logdet += 2.0 * logdet_lxx;
        }

        (logdet, pwrss)
    }

    fn determinant_term_and_pwrss(&self) -> (f64, f64) {
        self.determinant_term_and_pwrss_for_reml(self.optsum.reml)
    }

    fn objective_from_components(
        logdet: f64,
        pwrss: f64,
        denomdf: f64,
        fixed_sigma: Option<f64>,
    ) -> f64 {
        let log2pi = (2.0 * std::f64::consts::PI).ln();
        if let Some(sigma) = fixed_sigma {
            if !sigma.is_finite() || sigma <= 0.0 {
                return f64::INFINITY;
            }
            logdet + denomdf * (2.0 * sigma.ln() + log2pi) + pwrss / (sigma * sigma)
        } else {
            logdet + denomdf * (1.0 + (2.0 * std::f64::consts::PI * pwrss / denomdf).ln())
        }
    }

    fn profiled_objective_value(&self) -> f64 {
        let denomdf = if self.optsum.reml {
            (self.dims.n - self.dims.p) as f64
        } else {
            self.dims.n as f64
        };
        let (logdet, pwrss) = self.determinant_term_and_pwrss();
        Self::objective_from_components(logdet, pwrss, denomdf, self.optsum.sigma)
    }

    fn weight_logdet_correction(&self) -> f64 {
        if self.sqrtwts.is_empty() {
            0.0
        } else {
            2.0 * self.sqrtwts.iter().map(|sqrtwt| sqrtwt.ln()).sum::<f64>()
        }
    }

    /// Compute the user-facing deviance or REML criterion for the current θ.
    ///
    /// Weighted LMMs subtract the log-Jacobian term for the row scaling,
    /// matching MixedModels.jl's `objective(::LinearMixedModel)`. The optimizer
    /// hot path remains the internal `profiled_objective_from_parts`, whose
    /// target omits this θ-constant correction.
    pub fn objective_value(&self) -> f64 {
        self.profiled_objective_value() - self.weight_logdet_correction()
    }

    unstable_internal_method! {
    /// Set θ, update L, and return the objective value.
    ///
    /// Unstable internal surface: `pub` only with the `unstable-internals`
    /// feature; otherwise `pub(crate)`.
    unstable_vis fn objective_at(&mut self, theta: &[f64]) -> Result<f64> {
        self.set_theta(theta)?;
        self.update_l()?;
        Ok(self.objective_value())
    }
    }

    fn objective_at_fast_or_generic(&mut self, theta: &[f64]) -> Result<f64> {
        if let Some(objective) = self.profiled_objective_fast(theta) {
            return Ok(objective - self.weight_logdet_correction());
        }

        self.objective_at(theta)
    }

    /// Evaluate the objective at `optsum.initial` and store it as
    /// `optsum.finitial`, rescaling the initial guess once if the first
    /// evaluation is non-finite.
    ///
    /// Port of MixedModels.jl `linearmixedmodel.jl:478-491`. Without this, a
    /// non-finite initial objective (e.g. a non-PD θ₀ on a poorly scaled
    /// model) seeds the optimizer's `best_fmin` with `NaN/Inf`; every
    /// `obj < best_fmin` test is then false, so the model finalizes at the
    /// bad initial θ yet `fit()` returns `Ok` — a silent non-fit. If the
    /// rescaled retry is still non-finite we refuse instead of proceeding.
    fn set_initial_objective_with_rescue(&mut self) -> Result<()> {
        let theta0 = self.optsum.initial.clone();
        let finitial = self.objective_at_fast_or_generic(&theta0)?;
        if finitial.is_finite() {
            self.optsum.finitial = finitial;
            return Ok(());
        }

        // Julia: optsum.initial ./= (max(sqrtwts)^2 or 1) * max(response).
        let wt_scale = if self.sqrtwts.is_empty() {
            1.0
        } else {
            self.sqrtwts
                .iter()
                .copied()
                .fold(f64::MIN, f64::max)
                .powi(2)
        };
        let resp_max = self.y.iter().copied().fold(f64::MIN, f64::max);
        let denom = wt_scale * resp_max;
        let rescaled: Vec<f64> = if denom.is_finite() && denom.abs() > 0.0 {
            theta0.iter().map(|t| t / denom).collect()
        } else {
            theta0.clone()
        };

        let retried = self.objective_at_fast_or_generic(&rescaled)?;
        if !retried.is_finite() {
            return Err(MixedModelError::Optimization(
                "initial objective is non-finite even after rescaling the \
                 initial guess; the model is likely misspecified or too \
                 poorly scaled for the data to be fit reliably"
                    .to_string(),
            ));
        }
        // Subsequent optimization starts from the rescaled guess (Julia
        // mutates optsum.initial in place before optimize!).
        self.optsum.initial = rescaled;
        self.optsum.finitial = retried;
        Ok(())
    }

    fn profiled_objective_fast(&self, theta: &[f64]) -> Option<f64> {
        self.profiled_objective_fast_at(theta, self.optsum.reml, self.optsum.sigma)
    }

    fn profiled_objective_fast_at(
        &self,
        theta: &[f64],
        is_reml: bool,
        fixed_sigma: Option<f64>,
    ) -> Option<f64> {
        let tolerance = self
            .compiler_policy()
            .thresholds
            .cholesky_zero_pad_tolerance;
        if let Some(objective) = Self::profiled_objective_one_vsize1_fast(
            &self.a_blocks,
            &self.reterms,
            theta,
            self.dims,
            is_reml,
            fixed_sigma,
            tolerance,
        ) {
            return Some(objective);
        }

        Self::profiled_objective_one_vsize2_fast(
            &self.a_blocks,
            &self.reterms,
            theta,
            self.dims,
            is_reml,
            fixed_sigma,
            tolerance,
        )
    }

    fn vcov_active_from_l_last(l_last: &DMatrix<f64>, sigma: f64) -> DMatrix<f64> {
        let pp1 = l_last.nrows();
        let p = pp1 - 1;

        if p == 0 {
            return DMatrix::zeros(0, 0);
        }

        let l_xx = l_last.view((0, 0), (p, p)).clone_owned();
        let mut l_inv = DMatrix::<f64>::identity(p, p);
        for j in 0..p {
            for i in j..p {
                let mut s = l_inv[(i, j)];
                for k2 in j..i {
                    s -= l_xx[(i, k2)] * l_inv[(k2, j)];
                }
                l_inv[(i, j)] = s / l_xx[(i, i)];
            }
        }

        let sigma_sq = sigma * sigma;
        sigma_sq * (&l_inv.transpose() * &l_inv)
    }

    fn vcov_active_with_sigma(&self, sigma: f64) -> DMatrix<f64> {
        let k = self.reterms.len();
        let l_last = self.l_blocks[block_index(k, k)].as_dense();
        Self::vcov_active_from_l_last(&l_last, sigma)
    }

    fn unpivot_fixed_effect_covariance(&self, active_vcov: &DMatrix<f64>) -> DMatrix<f64> {
        // Unpivot
        let full_p = self.feterm.piv.len();
        let p = active_vcov.nrows();
        if p == full_p {
            let mut result = DMatrix::zeros(full_p, full_p);
            for i in 0..full_p {
                for j in 0..full_p {
                    result[(self.feterm.piv[i], self.feterm.piv[j])] = active_vcov[(i, j)];
                }
            }
            result
        } else {
            let mut result = DMatrix::from_element(full_p, full_p, f64::NAN);
            for i in 0..p {
                for j in 0..p {
                    result[(self.feterm.piv[i], self.feterm.piv[j])] = active_vcov[(i, j)];
                }
            }
            result
        }
    }

    fn vcov_with_sigma(&self, sigma: f64) -> DMatrix<f64> {
        let active = self.vcov_active_with_sigma(sigma);
        self.unpivot_fixed_effect_covariance(&active)
    }

    /// Evaluate the ML or REML deviance over `varpar = c(theta, sigma)`.
    ///
    /// This is the Rust analogue of `lmerTestR::devfun_vp`: it evaluates the
    /// unprofiled criterion at trial covariance parameters and a trial residual
    /// standard deviation, then restores the fitted model state.
    pub fn deviance_varpar(&mut self, varpar: &[f64], reml: bool) -> Result<f64> {
        self.validate_varpar(varpar)?;
        let n_theta = self.n_theta();
        let theta = &varpar[..n_theta];
        let sigma = varpar[n_theta];

        if let Some(deviance) = self.profiled_objective_fast_at(theta, reml, Some(sigma)) {
            return Ok(deviance);
        }

        let original_theta = self.theta();
        let original_l_blocks = self.l_blocks.clone();

        let result = (|| {
            self.set_theta(theta)?;
            self.update_l()?;

            let denomdf = if reml {
                (self.dims.n - self.dims.p) as f64
            } else {
                self.dims.n as f64
            };
            let (logdet, pwrss) = self.determinant_term_and_pwrss_for_reml(reml);
            let deviance = Self::objective_from_components(logdet, pwrss, denomdf, Some(sigma));
            if deviance.is_finite() {
                Ok(deviance)
            } else {
                Err(MixedModelError::Optimization(
                    "deviance over variance parameters is non-finite".to_string(),
                ))
            }
        })();

        self.set_theta(&original_theta)?;
        self.l_blocks = original_l_blocks;

        result
    }

    unstable_internal_method! {
    /// Evaluate the fixed-effect covariance matrix at `varpar = c(theta, sigma)`.
    ///
    /// This is the Rust analogue of `lmerTestR::get_covbeta`: at a trial
    /// covariance parameter point it returns `sigma^2 * RXi * RXi'`, then
    /// restores the fitted model state.
    ///
    /// Unstable internal surface: `pub` only with the `unstable-internals`
    /// feature; otherwise `pub(crate)`.
    unstable_vis fn vcov_beta_varpar(&mut self, varpar: &[f64]) -> Result<DMatrix<f64>> {
        self.validate_varpar(varpar)?;
        if let Some(vcov) = self.vcov_beta_varpar_fast(varpar) {
            return Ok(vcov);
        }

        let n_theta = self.n_theta();
        let theta = &varpar[..n_theta];
        let sigma = varpar[n_theta];

        let original_theta = self.theta();
        let original_l_blocks = self.l_blocks.clone();

        let result = (|| {
            self.set_theta(theta)?;
            self.update_l()?;

            let vcov = self.vcov_with_sigma(sigma);
            if matrix_is_finite(&vcov) {
                Ok(vcov)
            } else {
                Err(MixedModelError::InvalidArgument(
                    "vcov_beta(varpar) contains non-finite entries".to_string(),
                ))
            }
        })();

        self.set_theta(&original_theta)?;
        self.l_blocks = original_l_blocks;

        result
    }
    }

    fn vcov_beta_varpar_fast(&self, varpar: &[f64]) -> Option<DMatrix<f64>> {
        let n_theta = self.n_theta();
        let theta = &varpar[..n_theta];
        let sigma = varpar[n_theta];
        let tolerance = self
            .compiler_policy()
            .thresholds
            .cholesky_zero_pad_tolerance;
        let (l_last, _) = Self::cholesky_last_and_logdet_one_vsize1_fast(
            &self.a_blocks,
            &self.reterms,
            theta,
            tolerance,
        )?;
        let active = Self::vcov_active_from_l_last(&l_last, sigma);
        Some(self.unpivot_fixed_effect_covariance(&active))
    }

    unstable_internal_method! {
    /// Numerically differentiate `vcov_beta_varpar` with respect to `varpar`.
    ///
    /// Returns one `p x p` matrix per `varpar` component. The first
    /// implementation intentionally requires a feasible central-difference
    /// stencil; boundary-active parameters return an explicit unavailable
    /// reason instead of silently producing one-sided derivatives.
    ///
    /// Unstable internal surface: `pub` only with the `unstable-internals`
    /// feature; otherwise `pub(crate)`.
    unstable_vis fn jac_vcov_beta_varpar(&mut self, varpar: &[f64]) -> Result<Vec<DMatrix<f64>>> {
        self.validate_varpar(varpar)?;

        let lower_bounds = self.varpar_lower_bounds();
        let steps = finite_difference_steps(varpar, &lower_bounds, 1e-5);
        let mut jacobian = Vec::with_capacity(varpar.len());

        for index in 0..varpar.len() {
            let lower = lower_bounds
                .get(index)
                .copied()
                .unwrap_or(f64::NEG_INFINITY);
            let step =
                feasible_central_step(varpar[index], lower, steps[index]).ok_or_else(|| {
                    MixedModelError::InvalidArgument(format!(
                        "cannot compute central finite-difference derivative for varpar[{index}]: \
                     value is at or too near lower bound {lower}"
                    ))
                })?;

            let mut plus = varpar.to_vec();
            let mut minus = varpar.to_vec();
            plus[index] += step;
            minus[index] -= step;

            let vcov_plus = self.vcov_beta_varpar(&plus)?;
            let vcov_minus = self.vcov_beta_varpar(&minus)?;
            let derivative = (&vcov_plus - &vcov_minus) * (0.5 / step);
            if !matrix_is_finite(&derivative) {
                return Err(MixedModelError::InvalidArgument(format!(
                    "jac_vcov_beta derivative for varpar[{index}] contains non-finite entries"
                )));
            }
            jacobian.push(symmetrize_matrix(&derivative));
        }

        Ok(jacobian)
    }
    }

    /// Estimate `vcov(varpar)` from the Hessian of `deviance_varpar`.
    ///
    /// This mirrors the lmerTest convention `2 * H^+`, where `H^+` is the
    /// Moore-Penrose inverse of the Hessian over positive eigen-directions.
    pub fn vcov_varpar(&mut self, varpar: &[f64], reml: bool) -> Result<VcovVarparEstimate> {
        let hessian = self.hessian_deviance_varpar(varpar, reml)?;
        let hessian = symmetrize_matrix(&hessian);
        let eig = SymmetricEigen::new(hessian.clone());
        let eigenvalues = eig.eigenvalues.as_slice().to_vec();
        let max_abs_eigenvalue = eigenvalues
            .iter()
            .map(|value| value.abs())
            .fold(0.0, f64::max);
        let tolerance = (1e-8 * max_abs_eigenvalue.max(1.0)).max(1e-10);

        let positive_eigenvalues = eigenvalues
            .iter()
            .filter(|value| **value > tolerance)
            .count();
        let near_zero_eigenvalues = eigenvalues
            .iter()
            .filter(|value| value.abs() <= tolerance)
            .count();
        let negative_eigenvalues = eigenvalues
            .iter()
            .filter(|value| **value < -tolerance)
            .count();

        if positive_eigenvalues == 0 {
            return Err(MixedModelError::Optimization(
                "vcov_varpar unavailable: deviance Hessian has no positive eigen-directions"
                    .to_string(),
            ));
        }

        let mut inverse = DMatrix::zeros(varpar.len(), varpar.len());
        for (index, &eigenvalue) in eigenvalues.iter().enumerate() {
            if eigenvalue > tolerance {
                let column = eig.eigenvectors.column(index);
                inverse += (column * column.transpose()) * (1.0 / eigenvalue);
            }
        }

        let covariance = symmetrize_matrix(&(2.0 * inverse));
        if !matrix_is_finite(&covariance) {
            return Err(MixedModelError::Optimization(
                "vcov_varpar unavailable: covariance estimate contains non-finite entries"
                    .to_string(),
            ));
        }

        let used_reduced_rank = positive_eigenvalues < varpar.len();
        let mut notes = Vec::new();
        if near_zero_eigenvalues > 0 {
            notes.push(format!(
                "deviance Hessian has {near_zero_eigenvalues} near-zero eigenvalue(s)"
            ));
        }
        if negative_eigenvalues > 0 {
            notes.push(format!(
                "deviance Hessian has {negative_eigenvalues} negative eigenvalue(s)"
            ));
        }
        if used_reduced_rank {
            notes.push(
                "vcov_varpar used the positive-eigenvalue subspace of the Hessian".to_string(),
            );
        }

        Ok(VcovVarparEstimate {
            covariance,
            hessian,
            eigenvalues,
            tolerance,
            positive_eigenvalues,
            near_zero_eigenvalues,
            negative_eigenvalues,
            used_reduced_rank,
            reliability: if used_reduced_rank {
                ReliabilityGrade::Low
            } else {
                ReliabilityGrade::Moderate
            },
            notes,
        })
    }

    unstable_internal_method! {
    /// Build the Kenward-Roger response-covariance component decomposition.
    ///
    /// The returned matrices follow the `pbkrtest::get_SigmaG()` convention:
    /// fitted marginal response covariance is represented as a weighted sum of
    /// known component matrices. Random-effect component weights are fitted
    /// VarCorr covariance entries (`sigma^2 * Lambda Lambda'`); the final
    /// component is the residual variance multiplying the identity matrix.
    ///
    /// Unstable internal surface: `pub` only with the `unstable-internals`
    /// feature; otherwise `pub(crate)`.
    unstable_vis fn kenward_roger_sigma_g(&self) -> Result<KenwardRogerSigmaG> {
        if self.optsum.feval <= 0 {
            return Err(MixedModelError::NotFitted);
        }
        if !self.sqrtwts.is_empty() {
            return Err(MixedModelError::InvalidArgument(
                "Kenward-Roger Sigma/G decomposition is currently certified only for unweighted iid Gaussian residual models"
                    .to_string(),
            ));
        }

        let n = self.dims.n;
        let n_components: usize = self
            .reterms
            .iter()
            .map(kenward_roger_covariance_component_count)
            .sum::<usize>()
            + 1;
        let dense_bytes = dense_block_bytes(n, n).saturating_mul((n_components + 1) as u128);
        let limit = dense_block_limit_bytes();
        if dense_bytes > limit {
            return Err(MixedModelError::ProblemTooLarge(format!(
                "Kenward-Roger Sigma/G would materialize {} dense {} x {} f64 matrices ({:.2} GiB), above the configured limit ({:.2} GiB)",
                n_components + 1,
                n,
                n,
                dense_bytes as f64 / 1024.0_f64.powi(3),
                limit as f64 / 1024.0_f64.powi(3)
            )));
        }

        let sigma = self.sigma();
        if !sigma.is_finite() || sigma <= 0.0 {
            return Err(MixedModelError::InvalidArgument(
                "Kenward-Roger Sigma/G requires a finite positive residual sigma".to_string(),
            ));
        }
        let sigma_sq = sigma * sigma;

        let mut components = Vec::with_capacity(n_components);
        let mut component_weights = Vec::with_capacity(n_components);
        let mut component_labels = Vec::with_capacity(n_components);

        for (term_index, reterm) in self.reterms.iter().enumerate() {
            let covariance = sigma_sq * (&reterm.lambda * reterm.lambda.transpose());
            for (row, col) in kenward_roger_covariance_component_indices(reterm) {
                let component = kenward_roger_response_component(reterm, row, col, n)?;
                let label = format!(
                    "{}:{}[{},{}]",
                    term_index, reterm.grouping_name, reterm.cnames[row], reterm.cnames[col]
                );
                components.push(component);
                component_weights.push(covariance[(row, col)]);
                component_labels.push(label);
            }
        }

        let residual_component_index = components.len();
        components.push(DMatrix::identity(n, n));
        component_weights.push(sigma_sq);
        component_labels.push("residual".to_string());

        let mut response_covariance = DMatrix::zeros(n, n);
        for (component, &weight) in components.iter().zip(component_weights.iter()) {
            if !weight.is_finite() {
                return Err(MixedModelError::InvalidArgument(
                    "Kenward-Roger Sigma/G component weight is non-finite".to_string(),
                ));
            }
            if !matrix_is_finite(component) {
                return Err(MixedModelError::InvalidArgument(
                    "Kenward-Roger Sigma/G component contains non-finite entries".to_string(),
                ));
            }
            response_covariance += component * weight;
        }
        let response_covariance = symmetrize_matrix(&response_covariance);

        if !matrix_is_finite(&response_covariance) {
            return Err(MixedModelError::InvalidArgument(
                "Kenward-Roger Sigma/G response covariance contains non-finite entries".to_string(),
            ));
        }

        let max_component_asymmetry = components
            .iter()
            .map(matrix_max_asymmetry)
            .fold(0.0, f64::max)
            .max(matrix_max_asymmetry(&response_covariance));
        let eig = SymmetricEigen::new(response_covariance.clone());
        let sigma_min_eigenvalue = eig
            .eigenvalues
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let sigma_max_eigenvalue = eig
            .eigenvalues
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let eigen_tolerance = (1e-10 * sigma_max_eigenvalue.abs().max(1.0)).max(1e-12);
        let sigma_positive_definite = sigma_min_eigenvalue > eigen_tolerance;
        let mut notes = Vec::new();
        if !sigma_positive_definite {
            notes.push(format!(
                "response covariance is not positive definite at tolerance {eigen_tolerance}"
            ));
        }

        Ok(KenwardRogerSigmaG {
            sigma: response_covariance,
            components,
            component_weights,
            component_labels,
            residual_component_index,
            covariance_parameterization: "VarCorr covariance entries followed by residual variance"
                .to_string(),
            includes_residual_variance: true,
            n_observations: n,
            dense_bytes,
            sigma_min_eigenvalue,
            sigma_max_eigenvalue,
            sigma_positive_definite,
            max_component_asymmetry,
            reliability: if sigma_positive_definite {
                ReliabilityGrade::Moderate
            } else {
                ReliabilityGrade::NotAvailable
            },
            notes,
        })
    }
    }

    unstable_internal_method! {
    /// Compute the Kenward-Roger adjusted fixed-effect covariance.
    ///
    /// This is the Rust analogue of `pbkrtest::vcovAdj_internal()`. It uses the
    /// active fixed-effect basis internally and exposes an unpivoted
    /// `adjusted_vcov` for the user-facing coefficient surface.
    ///
    /// Unstable internal surface: `pub` only with the `unstable-internals`
    /// feature; otherwise `pub(crate)`.
    unstable_vis fn kenward_roger_adjusted_vcov(&self) -> Result<KenwardRogerAdjustedVcov> {
        let sigma_g = self.kenward_roger_sigma_g()?;
        if !sigma_g.sigma_positive_definite {
            return Err(MixedModelError::Singular(
                "Kenward-Roger adjusted covariance requires a positive-definite response covariance"
                    .to_string(),
            ));
        }

        let phi = self.vcov_active_with_sigma(self.sigma());
        if !matrix_is_finite(&phi) {
            return Err(MixedModelError::InvalidArgument(
                "Kenward-Roger adjusted covariance requires finite active fixed-effect covariance"
                    .to_string(),
            ));
        }
        let x = self.feterm.full_rank_x().into_owned();
        if x.ncols() != phi.ncols() {
            return Err(MixedModelError::DimensionMismatch(format!(
                "Kenward-Roger active fixed-effect covariance has {} columns, but X has {}",
                phi.ncols(),
                x.ncols()
            )));
        }

        let sigma_inv = invert_spd_matrix(&sigma_g.sigma, "Kenward-Roger response covariance")?;
        let tt = &sigma_inv * &x;
        let n_components = sigma_g.components.len();
        let p = phi.ncols();

        let mut hh = Vec::with_capacity(n_components);
        let mut oo = Vec::with_capacity(n_components);
        let mut p_matrices = Vec::with_capacity(n_components);
        for component in &sigma_g.components {
            let h = component * &sigma_inv;
            let o = &h * &x;
            let p_matrix = symmetrize_matrix(&(-o.transpose() * &tt));
            hh.push(h);
            oo.push(o);
            p_matrices.push(p_matrix);
        }

        let mut q_matrices = Vec::with_capacity(n_components.saturating_mul(n_components + 1) / 2);
        let mut information_matrix = DMatrix::zeros(n_components, n_components);
        for rr in 0..n_components {
            for ss in rr..n_components {
                let q_matrix = oo[rr].transpose() * &sigma_inv * &oo[ss];
                let q_index = q_matrices.len();
                q_matrices.push(q_matrix);

                let ktrace = matrix_elementwise_dot(&hh[rr].transpose(), &hh[ss]);
                let phi_q = matrix_elementwise_dot(&phi, &q_matrices[q_index]);
                let phi_p_rr = &phi * &p_matrices[rr];
                let pp_term = matrix_elementwise_dot(&phi_p_rr, &(&p_matrices[ss] * &phi));
                let value = ktrace - 2.0 * phi_q + pp_term;
                information_matrix[(rr, ss)] = value;
                information_matrix[(ss, rr)] = value;
            }
        }
        let information_matrix = symmetrize_matrix(&information_matrix);
        if !matrix_is_finite(&information_matrix) {
            return Err(MixedModelError::InvalidArgument(
                "Kenward-Roger information matrix contains non-finite entries".to_string(),
            ));
        }

        let information_eigen = SymmetricEigen::new(information_matrix.clone());
        let information_eigenvalues = information_eigen.eigenvalues.as_slice().to_vec();
        let condition_min_abs_eigenvalue = information_eigenvalues
            .iter()
            .map(|value| value.abs())
            .fold(f64::INFINITY, f64::min);
        let max_abs_eigenvalue = information_eigenvalues
            .iter()
            .map(|value| value.abs())
            .fold(0.0, f64::max);
        let generalized_inverse_tolerance = (1e-10 * max_abs_eigenvalue.max(1.0)).max(1e-12);
        let used_generalized_inverse =
            condition_min_abs_eigenvalue <= generalized_inverse_tolerance;
        let w = if used_generalized_inverse {
            2.0 * symmetric_pseudoinverse(&information_matrix, generalized_inverse_tolerance)
        } else {
            2.0 * invert_spd_matrix(
                &information_matrix,
                "Kenward-Roger covariance-parameter information matrix",
            )?
        };
        let w = symmetrize_matrix(&w);
        if !matrix_is_finite(&w) {
            return Err(MixedModelError::InvalidArgument(
                "Kenward-Roger covariance-parameter uncertainty matrix contains non-finite entries"
                    .to_string(),
            ));
        }

        let mut uu = DMatrix::zeros(p, p);
        for rr in 0..n_components {
            for ss in (rr + 1)..n_components {
                let q_index = symmetric_pair_index(rr, ss, n_components);
                uu +=
                    w[(rr, ss)] * (&q_matrices[q_index] - &p_matrices[rr] * &phi * &p_matrices[ss]);
            }
        }
        uu = &uu + uu.transpose();
        for rr in 0..n_components {
            let q_index = symmetric_pair_index(rr, rr, n_components);
            uu += w[(rr, rr)] * (&q_matrices[q_index] - &p_matrices[rr] * &phi * &p_matrices[rr]);
        }

        let gamma = &phi * uu * &phi;
        let adjusted_active = symmetrize_matrix(&(&phi + 2.0 * gamma));
        if !matrix_is_finite(&adjusted_active) {
            return Err(MixedModelError::InvalidArgument(
                "Kenward-Roger adjusted fixed-effect covariance contains non-finite entries"
                    .to_string(),
            ));
        }

        let mut notes = Vec::new();
        if used_generalized_inverse {
            notes.push(format!(
                "Kenward-Roger information matrix used a generalized inverse at tolerance {generalized_inverse_tolerance}"
            ));
        }
        if sigma_g.reliability != ReliabilityGrade::Moderate {
            notes.extend(sigma_g.notes.clone());
        }

        let reliability = if used_generalized_inverse {
            ReliabilityGrade::Low
        } else {
            ReliabilityGrade::Moderate
        };

        Ok(KenwardRogerAdjustedVcov {
            unadjusted_vcov_active: phi,
            adjusted_vcov: self.unpivot_fixed_effect_covariance(&adjusted_active),
            adjusted_vcov_active: adjusted_active,
            p_matrices,
            q_matrices,
            w,
            information_matrix,
            information_eigenvalues,
            condition_min_abs_eigenvalue,
            used_generalized_inverse,
            component_labels: sigma_g.component_labels,
            reliability,
            notes,
        })
    }
    }

    unstable_internal_method! {
    #[allow(dead_code)]
    /// Compute Kenward-Roger denominator df for `L beta = rhs`.
    ///
    /// This follows `pbkrtest::Lb_ddf(L, V0, Vadj)` using the active fixed-effect
    /// covariance basis. User-order full-rank contrasts are accepted and mapped
    /// onto the active basis.
    ///
    /// Unstable internal surface: `pub` only with the `unstable-internals`
    /// feature; otherwise `pub(crate)`.
    unstable_vis fn kenward_roger_lbddf(&self, l: &DMatrix<f64>) -> Result<KenwardRogerLbDdf> {
        let adjusted = self.kenward_roger_adjusted_vcov()?;
        self.kenward_roger_lbddf_with_adjusted(l, &adjusted)
    }
    }

    unstable_internal_method! {
    /// Unstable internal surface: `pub` only with the `unstable-internals`
    /// feature; otherwise `pub(crate)`.
    unstable_vis fn kenward_roger_lbddf_with_adjusted(
        &self,
        l: &DMatrix<f64>,
        adjusted: &KenwardRogerAdjustedVcov,
    ) -> Result<KenwardRogerLbDdf> {
        let l_active = self.fixed_effect_contrast_to_active_basis(l)?;
        if l_active.nrows() == 0 {
            return Err(MixedModelError::InvalidArgument(
                "Kenward-Roger Lb_ddf requires at least one restriction row".to_string(),
            ));
        }
        if l_active.ncols() != adjusted.unadjusted_vcov_active.ncols() {
            return Err(MixedModelError::DimensionMismatch(format!(
                "Kenward-Roger Lb_ddf contrast has {} active columns, but V0 has {}",
                l_active.ncols(),
                adjusted.unadjusted_vcov_active.ncols()
            )));
        }

        let rank_tolerance = 1e-10;
        let restriction_rank = matrix_rank(&l_active, rank_tolerance);
        if restriction_rank == 0 {
            return Err(MixedModelError::InvalidArgument(
                "Kenward-Roger Lb_ddf contrast has zero numerical rank".to_string(),
            ));
        }

        let v0 = &adjusted.unadjusted_vcov_active;
        let middle = symmetrize_matrix(&(&l_active * v0 * l_active.transpose()));
        let middle_eig = SymmetricEigen::new(middle.clone());
        let middle_max_abs = middle_eig
            .eigenvalues
            .iter()
            .map(|value| value.abs())
            .fold(0.0, f64::max);
        let middle_tol = (1e-10 * middle_max_abs.max(1.0)).max(1e-12);
        let middle_min_abs = middle_eig
            .eigenvalues
            .iter()
            .map(|value| value.abs())
            .fold(f64::INFINITY, f64::min);
        let used_middle_generalized_inverse = middle_min_abs <= middle_tol;
        let middle_inverse = if used_middle_generalized_inverse {
            symmetric_pseudoinverse(&middle, middle_tol)
        } else {
            invert_spd_matrix(&middle, "Kenward-Roger L V0 L' matrix")?
        };
        let theta = l_active.transpose() * middle_inverse * &l_active;
        let theta_v0 = &theta * v0;

        let mut a1 = 0.0;
        let mut a2 = 0.0;
        let n_components = adjusted.p_matrices.len();
        if adjusted.w.shape() != (n_components, n_components) {
            return Err(MixedModelError::DimensionMismatch(format!(
                "Kenward-Roger W is {} x {}, expected {n_components} x {n_components}",
                adjusted.w.nrows(),
                adjusted.w.ncols()
            )));
        }
        for ii in 0..n_components {
            for jj in ii..n_components {
                let e = if ii == jj { 1.0 } else { 2.0 };
                let ui = &theta_v0 * &adjusted.p_matrices[ii] * v0;
                let uj = &theta_v0 * &adjusted.p_matrices[jj] * v0;
                a1 += e * adjusted.w[(ii, jj)] * matrix_trace(&ui) * matrix_trace(&uj);
                a2 += e * adjusted.w[(ii, jj)] * matrix_trace_product(&ui, &uj);
            }
        }

        let q = restriction_rank as f64;
        let b = (a1 + 6.0 * a2) / (2.0 * q);
        let g_denom = (q + 2.0) * a2;
        if !g_denom.is_finite() || g_denom.abs() <= 1e-14 {
            return Err(MixedModelError::InvalidArgument(
                "Kenward-Roger Lb_ddf has non-finite or zero g denominator".to_string(),
            ));
        }
        let g = ((q + 1.0) * a1 - (q + 4.0) * a2) / g_denom;
        let c_denom = 3.0 * q + 2.0 * (1.0 - g);
        if !c_denom.is_finite() || c_denom.abs() <= 1e-14 {
            return Err(MixedModelError::InvalidArgument(
                "Kenward-Roger Lb_ddf has non-finite or zero correction denominator".to_string(),
            ));
        }
        let c1 = g / c_denom;
        let c2 = (q - g) / c_denom;
        let c3 = (q + 2.0 - g) / c_denom;
        let mut v0_correction = 1.0 + c1 * b;
        let v1 = 1.0 - c2 * b;
        let v2 = 1.0 - c3 * b;
        if v0_correction.abs() < 1e-10 {
            v0_correction = 0.0;
        }
        let rho = (1.0 / q) * div_zero(1.0 - a2 / q, v1, 1e-14).powi(2) * v0_correction / v2;
        let denominator = q * rho - 1.0;
        if !denominator.is_finite() || denominator.abs() <= 1e-14 {
            return Err(MixedModelError::InvalidArgument(
                "Kenward-Roger Lb_ddf has non-finite or zero final denominator".to_string(),
            ));
        }
        let denominator_df = 4.0 + (q + 2.0) / denominator;
        if !denominator_df.is_finite() || denominator_df <= 0.0 {
            return Err(MixedModelError::InvalidArgument(
                "Kenward-Roger Lb_ddf produced a non-finite or non-positive denominator df"
                    .to_string(),
            ));
        }

        let mut notes = adjusted.notes.clone();
        let used_generalized_inverse =
            adjusted.used_generalized_inverse || used_middle_generalized_inverse;
        if used_middle_generalized_inverse {
            notes.push(format!(
                "Kenward-Roger L V0 L' used a generalized inverse at tolerance {middle_tol}"
            ));
        }
        if restriction_rank < l_active.nrows() {
            notes.push(format!(
                "Kenward-Roger restriction matrix row rank {restriction_rank} is lower than {} submitted row(s)",
                l_active.nrows()
            ));
        }

        Ok(KenwardRogerLbDdf {
            denominator_df,
            numerator_df: q,
            restriction_rank,
            a1,
            a2,
            b,
            g,
            rho,
            used_generalized_inverse,
            reliability: if used_generalized_inverse {
                ReliabilityGrade::Low
            } else {
                adjusted.reliability
            },
            notes,
        })
    }
    }

    fn fixed_effect_contrast_to_active_basis(&self, l: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let active_p = self.feterm.rank;
        let full_p = self.feterm.piv.len();
        if l.ncols() != full_p && l.ncols() != active_p {
            return Err(MixedModelError::DimensionMismatch(format!(
                "fixed-effect contrast has {} column(s), expected active {active_p} or full {full_p}",
                l.ncols()
            )));
        }
        if l.ncols() == active_p && l.ncols() != full_p {
            return Ok(l.clone());
        }
        for dropped_position in active_p..full_p {
            let original_col = self.feterm.piv[dropped_position];
            for row in 0..l.nrows() {
                if l[(row, original_col)].abs() > 1e-12 {
                    return Err(MixedModelError::InvalidArgument(format!(
                        "Kenward-Roger contrast touches dropped fixed-effect column {original_col}"
                    )));
                }
            }
        }
        let mut active = DMatrix::zeros(l.nrows(), active_p);
        for active_col in 0..active_p {
            let original_col = self.feterm.piv[active_col];
            for row in 0..l.nrows() {
                active[(row, active_col)] = l[(row, original_col)];
            }
        }
        Ok(active)
    }

    fn fixed_effect_user_beta_to_active_basis(&self, beta: &DVector<f64>) -> Result<DVector<f64>> {
        let active_p = self.feterm.rank;
        let full_p = self.feterm.piv.len();
        if beta.len() == active_p && beta.len() != full_p {
            return Ok(beta.clone());
        }
        if beta.len() != full_p {
            return Err(MixedModelError::DimensionMismatch(format!(
                "fixed-effect beta has length {}, expected active {active_p} or full {full_p}",
                beta.len()
            )));
        }
        let mut active = DVector::zeros(active_p);
        for active_col in 0..active_p {
            active[active_col] = beta[self.feterm.piv[active_col]];
        }
        Ok(active)
    }

    fn fixed_effect_active_vector_to_user_basis(
        &self,
        values: &DVector<f64>,
        label: &str,
    ) -> Result<DVector<f64>> {
        let active_p = self.feterm.rank;
        let full_p = self.feterm.piv.len();
        if values.len() == full_p {
            return Ok(values.clone());
        }
        if values.len() != active_p {
            return Err(MixedModelError::DimensionMismatch(format!(
                "fixed-effect {label} vector has length {}, expected active {active_p} or full {full_p}",
                values.len()
            )));
        }
        let mut full = DVector::from_element(full_p, f64::NAN);
        for active_col in 0..active_p {
            full[self.feterm.piv[active_col]] = values[active_col];
        }
        Ok(full)
    }

    fn hessian_deviance_varpar(&mut self, varpar: &[f64], reml: bool) -> Result<DMatrix<f64>> {
        self.validate_varpar(varpar)?;
        let lower_bounds = self.varpar_lower_bounds();
        let steps = finite_difference_steps(varpar, &lower_bounds, 1e-4);
        let f0 = self.deviance_varpar(varpar, reml)?;
        if !f0.is_finite() {
            return Err(MixedModelError::Optimization(
                "deviance_varpar at fitted varpar is non-finite".to_string(),
            ));
        }

        let mut central_steps = Vec::with_capacity(varpar.len());
        for index in 0..varpar.len() {
            let lower = lower_bounds
                .get(index)
                .copied()
                .unwrap_or(f64::NEG_INFINITY);
            let step =
                feasible_central_step(varpar[index], lower, steps[index]).ok_or_else(|| {
                    MixedModelError::InvalidArgument(format!(
                        "cannot compute central finite-difference Hessian for varpar[{index}]: \
                     value is at or too near lower bound {lower}"
                    ))
                })?;
            central_steps.push(step);
        }

        let mut hessian = DMatrix::zeros(varpar.len(), varpar.len());
        for row in 0..varpar.len() {
            let h_row = central_steps[row];
            let f_plus = finite_difference_deviance_varpar(self, varpar, row, h_row, reml)?;
            let f_minus = finite_difference_deviance_varpar(self, varpar, row, -h_row, reml)?;
            hessian[(row, row)] = (f_plus - 2.0 * f0 + f_minus) / (h_row * h_row);

            for col in 0..row {
                let h_col = central_steps[col];
                let f_pp = finite_difference_deviance_varpar_2d(
                    self, varpar, row, h_row, col, h_col, reml,
                )?;
                let f_pm = finite_difference_deviance_varpar_2d(
                    self, varpar, row, h_row, col, -h_col, reml,
                )?;
                let f_mp = finite_difference_deviance_varpar_2d(
                    self, varpar, row, -h_row, col, h_col, reml,
                )?;
                let f_mm = finite_difference_deviance_varpar_2d(
                    self, varpar, row, -h_row, col, -h_col, reml,
                )?;
                let value = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h_row * h_col);
                hessian[(row, col)] = value;
                hessian[(col, row)] = value;
            }
        }

        if matrix_is_finite(&hessian) {
            Ok(hessian)
        } else {
            Err(MixedModelError::Optimization(
                "deviance_varpar Hessian contains non-finite entries".to_string(),
            ))
        }
    }

    fn validate_varpar(&self, varpar: &[f64]) -> Result<()> {
        let n_theta = self.n_theta();
        if varpar.len() != n_theta + 1 {
            return Err(MixedModelError::DimensionMismatch(format!(
                "varpar has length {}, expected {} theta parameter(s) plus sigma",
                varpar.len(),
                n_theta
            )));
        }
        if varpar.iter().any(|value| !value.is_finite()) {
            return Err(MixedModelError::InvalidArgument(
                "varpar contains a non-finite value".to_string(),
            ));
        }

        let sigma = varpar[n_theta];
        if sigma <= 0.0 {
            return Err(MixedModelError::InvalidArgument(
                "varpar sigma must be positive".to_string(),
            ));
        }

        let lower_bounds = self.lower_bounds();
        if let Some((index, (&value, &lower))) = varpar[..n_theta]
            .iter()
            .zip(lower_bounds.iter())
            .enumerate()
            .find(|(_, (&value, &lower))| lower.is_finite() && value < lower)
        {
            return Err(MixedModelError::InvalidArgument(format!(
                "theta[{index}] = {value} is below lower bound {lower}"
            )));
        }

        Ok(())
    }

    fn varpar_lower_bounds(&self) -> Vec<f64> {
        let mut lower_bounds = self.lower_bounds();
        lower_bounds.push(0.0);
        lower_bounds
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

    fn apply_theta_to_reterms(reterms: &mut [ReMat], theta: &[f64]) -> Option<()> {
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

    fn cholesky_last_and_logdet_one_vsize1_fast(
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

    fn profiled_objective_one_vsize1_fast(
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

    fn profiled_objective_one_vsize2_fast(
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
    fn nlopt_status_label(name: &str) -> String {
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

    fn cobyla_success_status_label(status: cobyla::SuccessStatus) -> String {
        match status {
            cobyla::SuccessStatus::Success => "SUCCESS".to_string(),
            cobyla::SuccessStatus::StopValReached => "STOPVAL_REACHED".to_string(),
            cobyla::SuccessStatus::FtolReached => "FTOL_REACHED".to_string(),
            cobyla::SuccessStatus::XtolReached => "XTOL_REACHED".to_string(),
            cobyla::SuccessStatus::MaxEvalReached => "MAXEVAL_REACHED".to_string(),
            cobyla::SuccessStatus::MaxTimeReached => "MAXTIME_REACHED".to_string(),
        }
    }

    fn cobyla_fail_status_label(status: cobyla::FailStatus) -> String {
        match status {
            cobyla::FailStatus::Failure => "FAILURE".to_string(),
            cobyla::FailStatus::InvalidArgs => "INVALID_ARGS".to_string(),
            cobyla::FailStatus::OutOfMemory => "OUT_OF_MEMORY".to_string(),
            cobyla::FailStatus::RoundoffLimited => "ROUNDOFF_LIMITED".to_string(),
            cobyla::FailStatus::ForcedStop => "FORCED_STOP".to_string(),
            cobyla::FailStatus::UnexpectedError => "UNEXPECTED_ERROR".to_string(),
        }
    }

    fn trust_bq_status_label(status: TrustBqStopReason) -> String {
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

    fn finalize_fit_result(
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
        let trust_bq_initial = self.optsum.initial.clone();
        let mut certificate_stop = TrustBqCertificateStopState::new(
            n_theta,
            policy.max_evaluations,
            policy.certificate_ftol_abs,
            policy.certificate_ftol_rel,
        );
        let lower_bounds = self.lower_bounds();
        let upper_bounds = vec![f64::INFINITY; n_theta];
        let mut certificate_progress = |progress: &TrustBqProgress<'_>| -> Result<bool> {
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
                initial_radius: policy.initial_radius,
                final_radius: policy.final_radius,
                max_evaluations: policy.max_evaluations,
                ftol_abs: policy.ftol_abs,
                ftol_rel: policy.ftol_rel,
                max_cross_terms: policy.max_cross_terms,
                reuse_samples: policy.reuse_samples,
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
        let return_value = Some(Self::trust_bq_status_label(result.stop_reason));

        self.finalize_fit_result(
            final_theta,
            final_fmin,
            result.fevals as i64,
            fit_log.into_inner(),
            Optimizer::TrustBq,
            return_value,
        )?;
        self.optsum.final_trust_radius = Some(result.final_radius);
        Ok(self)
    }

    fn fit_cobyla_with_maxeval(
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
                self.fit_trust_bq_with_maxeval(reml, None)?;
            }
        }

        self.apply_kkt_guided_boundary_restart(reml)?;
        self.refresh_optimizer_certificate();
        self.refresh_effective_covariance_summaries();
        self.refresh_covariance_parameter_traces();
        self.refresh_fixed_effect_covariance_matrix();
        self.refresh_fixed_effect_inference_table();
        Ok(self)
    }

    /// Extract the fixed-effects coefficients β from the Cholesky factor.
    pub fn beta(&self) -> DVector<f64> {
        let k = self.reterms.len();
        let l_last = self.l_blocks[block_index(k, k)].as_dense();
        let pp1 = l_last.nrows();
        let p = pp1 - 1;

        if p == 0 {
            return DVector::zeros(0);
        }

        let l_xx = l_last.view((0, 0), (p, p));
        let mut beta = DVector::zeros(p);
        for j in 0..p {
            beta[j] = l_last[(pp1 - 1, j)];
        }

        for i in (0..p).rev() {
            let mut s = beta[i];
            for j in (i + 1)..p {
                s -= l_xx[(j, i)] * beta[j];
            }
            beta[i] = s / l_xx[(i, i)];
        }

        beta
    }

    /// Standard deviation estimate (σ).
    pub fn sigma(&self) -> f64 {
        if let Some(sigma) = self.optsum.sigma {
            return sigma;
        }
        let k = self.reterms.len();
        let l_last = self.l_blocks[block_index(k, k)].as_dense();
        let pp1 = l_last.nrows();
        let last_diag = l_last[(pp1 - 1, pp1 - 1)].abs();

        let denom = if self.optsum.reml {
            (self.dims.n - self.dims.p) as f64
        } else {
            self.dims.n as f64
        };

        last_diag / denom.sqrt()
    }

    /// Penalized weighted residual sum of squares.
    pub fn pwrss(&self) -> f64 {
        let k = self.reterms.len();
        let l_last = self.l_blocks[block_index(k, k)].as_dense();
        let pp1 = l_last.nrows();
        let last_diag = l_last[(pp1 - 1, pp1 - 1)];
        last_diag * last_diag
    }

    /// Profile one or more response columns at the current theta.
    ///
    /// Each response column shares the current model's fixed-effects design,
    /// random-effects structure, and theta, but gets its own profiled beta
    /// and sigma.
    pub fn profile_response_matrix(
        &self,
        responses: &DMatrix<f64>,
        reml: bool,
    ) -> Result<ResponseMatrixProfile> {
        if responses.nrows() != self.dims.n {
            return Err(MixedModelError::DimensionMismatch(format!(
                "response matrix has {} rows, expected {}",
                responses.nrows(),
                self.dims.n
            )));
        }

        let kernel = crate::model::kernel::LmmObjectiveKernel::from_model(self)?;
        let mut workspace = kernel.workspace();
        // The workspace reterms carry this model's current λ; only the
        // factorization needs to be brought up to date before profiling.
        workspace.update_l()?;
        workspace.profile(responses, reml)
    }

    /// Log-determinant of the RE Cholesky blocks.
    pub fn logdet_re(&self) -> f64 {
        let k = self.reterms.len();
        let mut ld = 0.0;
        for j in 0..k {
            ld += logdet_block(&self.l_blocks[block_index(j, j)]);
        }
        ld
    }

    /// Conditional modes of the random effects (the "u" vectors, on the spherical scale).
    ///
    /// Solves the blocked lower-triangular system `L * u = c` where:
    ///   - `c_j = Λ_j' Z_j' wr`  (weighted residuals projected onto RE term j)
    ///   - `wr = W^{1/2}(y - Xβ)`  (weighted residuals in observation space)
    ///   - `L` is the blocked Cholesky factor stored in `self.l_blocks`
    ///
    /// Returns one matrix per RE term with shape `vsize × n_levels`.
    pub fn ranef_u(&self) -> Vec<DMatrix<f64>> {
        let k = self.reterms.len();
        let p = self.dims.p;
        let n = self.dims.n;
        let beta = self.beta();
        let wtxy = &self.xy_mat.wtxy;

        // Step 1: weighted residuals wr[obs] = wy[obs] - wX[obs,:]*beta
        let mut wr = vec![0.0f64; n];
        for obs in 0..n {
            let mut val = wtxy[(obs, p)]; // weighted y (last column)
            for q in 0..p {
                val -= wtxy[(obs, q)] * beta[q];
            }
            wr[obs] = val;
        }

        // Step 2: c_j = Λ_j' Z_j' wr  for each RE term j
        let mut c_vecs: Vec<DVector<f64>> = Vec::with_capacity(k);
        for j in 0..k {
            let re = &self.reterms[j];
            let vs = re.vsize;
            let nranef = re.n_ranef();
            let n_levels = re.n_levels();

            // Accumulate Z_j' wr (wtz already incorporates sqrtwts)
            let mut c = vec![0.0f64; nranef];
            for obs in 0..n {
                let r = re.refs[obs] as usize;
                for s in 0..vs {
                    c[r * vs + s] += re.wtz[(s, obs)] * wr[obs];
                }
            }

            // Apply Λ_j' per level block: c_scaled[lev,i] = Σ_{row>=i} Λ[row,i] * c[lev,row]
            let lambda = &re.lambda;
            let mut c_scaled = vec![0.0f64; nranef];
            for lev in 0..n_levels {
                for i in 0..vs {
                    let mut val = 0.0;
                    // Λ' is upper triangular of Λ stored as lower, so Λ'[i,row] = Λ[row,i]
                    for row in i..vs {
                        val += lambda[(row, i)] * c[lev * vs + row];
                    }
                    c_scaled[lev * vs + i] = val;
                }
            }

            c_vecs.push(DVector::from_vec(c_scaled));
        }

        // Step 3: blocked solve  (L L') u = c  via forward then backward pass.

        // Forward pass: solve L * v = c  (lower-triangular blocked forward substitution)
        let mut v_vecs: Vec<DVector<f64>> = Vec::with_capacity(k);
        for j in 0..k {
            let nranef_j = self.reterms[j].n_ranef();

            let mut rhs = c_vecs[j].clone();

            // rhs -= L[j,m] * v_m  for all already-solved m < j
            for m in 0..j {
                let l_jm = self.l_blocks[block_index(j, m)].as_dense();
                let v_m = &v_vecs[m];
                for row in 0..nranef_j {
                    let mut dot = 0.0;
                    for col in 0..v_m.len() {
                        dot += l_jm[(row, col)] * v_m[col];
                    }
                    rhs[row] -= dot;
                }
            }

            // Solve L[j,j] * v_j = rhs  (forward substitution)
            let mut v_j = rhs.as_slice().to_vec();
            solve_lower_block_against_rhs(&self.l_blocks[block_index(j, j)], &mut v_j);
            let v_j = DVector::from_vec(v_j);
            v_vecs.push(v_j);
        }

        // Backward pass: solve L' * u = v  (upper-triangular blocked back-substitution)
        let mut u_vecs: Vec<DVector<f64>> = vec![DVector::zeros(0); k];
        for j in (0..k).rev() {
            let nranef_j = self.reterms[j].n_ranef();

            let mut rhs = v_vecs[j].clone();

            // rhs -= L[m,j]' * u_m  for all already-solved m > j
            for m in (j + 1)..k {
                let l_mj = self.l_blocks[block_index(m, j)].as_dense();
                let u_m = &u_vecs[m];
                // L[m,j]' is nranef_j × nranef_m: rhs[row] -= sum_col l_mj[(col,row)] * u_m[col]
                for row in 0..nranef_j {
                    let mut dot = 0.0;
                    for col in 0..u_m.len() {
                        dot += l_mj[(col, row)] * u_m[col];
                    }
                    rhs[row] -= dot;
                }
            }

            // Solve L[j,j]' * u_j = rhs  (backward substitution with L')
            let mut u_j = rhs.as_slice().to_vec();
            solve_upper_block_from_lower_transpose_against_rhs(
                &self.l_blocks[block_index(j, j)],
                &mut u_j,
            );
            let u_j = DVector::from_vec(u_j);
            u_vecs[j] = u_j;
        }

        // Step 4: reshape u vectors into vsize × n_levels matrices
        self.reterms
            .iter()
            .zip(u_vecs)
            .map(|(rt, u)| {
                let vs = rt.vsize;
                let nl = rt.n_levels();
                DMatrix::from_column_slice(vs, nl, u.as_slice())
            })
            .collect()
    }

    /// Conditional modes on the original scale: b = Λ * u
    pub fn ranef_b(&self) -> Vec<DMatrix<f64>> {
        self.ranef_u()
            .into_iter()
            .zip(&self.reterms)
            .map(|(u, rt)| &rt.lambda * &u)
            .collect()
    }

    /// Grouping factor names.
    pub fn fnames(&self) -> Vec<String> {
        self.reterms
            .iter()
            .map(|rt| rt.grouping_name.clone())
            .collect()
    }

    /// Variance-covariance summary for the fitted random effects.
    pub fn varcorr(&self) -> VarCorr {
        VarCorr::from_model(&self.reterms, self.sigma()).with_residual_source(self.residual_source)
    }

    /// Condition number of each RE Lambda factor.
    ///
    /// Mirrors `cond(fm)` in Julia's MixedModels.jl. For a scalar RE, the
    /// condition number is always 1.0. For a vector RE, it is the ratio of the
    /// largest to smallest singular value of the lower-triangular Cholesky factor.
    pub fn cond(&self) -> Vec<f64> {
        self.reterms
            .iter()
            .map(|rt| {
                let s = rt.vsize;
                if s <= 1 {
                    1.0
                } else {
                    let svd = rt.lambda.clone().svd(false, false);
                    let sv = &svd.singular_values;
                    let smax = sv.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let smin = sv.iter().cloned().fold(f64::INFINITY, f64::min);
                    if smin < f64::EPSILON {
                        f64::INFINITY
                    } else {
                        smax / smin
                    }
                }
            })
            .collect()
    }

    /// Residual degrees of freedom: `nobs - rank(X)`.
    ///
    /// Mirrors `dof_residual(fm)` in Julia's MixedModels.jl.
    pub fn dof_residual(&self) -> usize {
        self.nobs().saturating_sub(self.feterm.rank)
    }

    /// Residual scale reported by Julia's `varest(fm)`.
    ///
    /// For estimated-σ fits this is σ². For fixed-σ fits, MixedModels.jl
    /// reports the fixed σ itself, not σ².
    ///
    /// For summary-estimate fits constructed via
    /// [`LinearMixedModel::from_summary_estimates`] this returns `1.0` —
    /// σ is fixed at 1, so the Julia-parity convention reports the SD,
    /// not σ². The "residual variance" of a summary-estimate fit is the
    /// user-supplied sampling variance `V_i`, not an estimated quantity.
    /// See `docs/summary_estimates_meta_analysis.md`.
    pub fn varest(&self) -> f64 {
        if let Some(sigma) = self.optsum.sigma {
            return sigma;
        }
        let s = self.sigma();
        s * s
    }

    /// Log-determinant of the RE blocks of the Cholesky factor L.
    ///
    /// Mirrors `logdet(fm)` in Julia's MixedModels.jl.
    pub fn logdet(&self) -> f64 {
        self.logdet_re()
    }

    /// Model dimensions as `(n, p, total_nranef, nretrms)`.
    ///
    /// Mirrors `size(fm)` in Julia's MixedModels.jl where the four
    /// elements are:
    /// - `n`: number of observations
    /// - `p`: rank of the fixed-effects matrix
    /// - `total_nranef`: total number of random-effects columns (`Σ vsize_j * n_levels_j`)
    /// - `nretrms`: number of RE grouping factors
    pub fn model_size(&self) -> (usize, usize, usize, usize) {
        let total_nranef: usize = self.reterms.iter().map(|rt| rt.n_ranef()).sum();
        (self.dims.n, self.dims.p, total_nranef, self.dims.nretrms)
    }

    /// Standard deviations of each random-effects term plus the residual σ.
    ///
    /// Returns one `Vec<f64>` per RE term (with one entry per random-effects
    /// component), followed by `vec![sigma]` for the residual.
    ///
    /// Mirrors `std(fm)` in Julia's MixedModels.jl.
    pub fn std_devs(&self) -> Vec<Vec<f64>> {
        let sigma = self.sigma();
        let mut out: Vec<Vec<f64>> = self
            .reterms
            .iter()
            .map(|rt| {
                (0..rt.vsize)
                    .map(|i| {
                        let sq: f64 = (0..=i).map(|j| rt.lambda[(i, j)].powi(2)).sum();
                        sigma * sq.sqrt()
                    })
                    .collect()
            })
            .collect();
        out.push(vec![sigma]);
        out
    }

    /// Simulate a new response vector from the fitted model.
    ///
    /// Draws `u_j ~ N(0, I)` for each RE term, computes `b_j = σ Λ_j u_j`,
    /// adds fixed-effects `Xβ`, RE contribution `Σ Z_j b_j`, and i.i.d.
    /// residual noise `ε ~ N(0, σ²)`.
    ///
    /// Mirrors `simulate(fm)` in Julia's MixedModels.jl.
    pub fn simulate<R: rand::Rng>(&self, rng: &mut R) -> DVector<f64> {
        let beta = self.beta();
        self.simulate_with_active_beta(rng, &beta)
            .expect("fitted beta should match active fixed-effect design")
    }

    fn simulate_with_active_beta<R: rand::Rng>(
        &self,
        rng: &mut R,
        beta: &DVector<f64>,
    ) -> Result<DVector<f64>> {
        use rand_distr::{Distribution, Normal};

        let n = self.dims.n;
        let sigma = self.sigma();

        // Fixed-effects prediction: Xβ
        let x = self.feterm.full_rank_x();
        if beta.len() != x.ncols() {
            return Err(MixedModelError::DimensionMismatch(format!(
                "simulation beta has length {}, but active fixed-effect design has {} column(s)",
                beta.len(),
                x.ncols()
            )));
        }
        let mut y_new: DVector<f64> = x * beta;

        // Random-effects contribution
        let normal01 = Normal::new(0.0, 1.0).unwrap();
        for rt in &self.reterms {
            let n_levels = rt.n_levels();
            // u ~ N(0, I)
            let u = DMatrix::from_fn(rt.vsize, n_levels, |_, _| normal01.sample(rng));
            // b = sigma * Λ * u
            let b = sigma * &rt.lambda * &u;
            let bvec = DVector::from_column_slice(b.as_slice());
            for (obs, &ref_idx) in rt.refs.iter().enumerate() {
                let r = ref_idx as usize;
                for s in 0..rt.vsize {
                    y_new[obs] += rt.z[(s, obs)] * bvec[r * rt.vsize + s];
                }
            }
        }

        // Residual noise ε ~ N(0, σ²)
        let eps_dist = Normal::new(0.0, sigma).unwrap();
        for i in 0..n {
            y_new[i] += eps_dist.sample(rng);
        }

        Ok(y_new)
    }

    /// Build the null data-generating state for a fixed-effect bootstrap test.
    pub fn fixed_effect_null_bootstrap_target(
        &self,
        hypothesis: &FixedEffectHypothesis,
    ) -> Result<FixedEffectNullBootstrapTarget> {
        let n_coefficients = self.coef_names().len();
        if hypothesis.n_coefficients() != n_coefficients {
            return Err(MixedModelError::DimensionMismatch(format!(
                "fixed-effect null target contrast has {} coefficient column(s), but the model has {n_coefficients}",
                hypothesis.n_coefficients()
            )));
        }

        let beta_fitted = self.coef();
        let vcov = self.vcov();
        let estimability = assess_fixed_contrast_estimability(hypothesis, &beta_fitted, &vcov);
        if estimability.status != EstimabilityStatus::Estimable {
            return Err(MixedModelError::InvalidArgument(
                "fixed-effect null bootstrap target requires an estimable contrast".to_string(),
            ));
        }
        if !matrix_is_finite(&vcov) {
            return Err(MixedModelError::InvalidArgument(
                "fixed-effect null bootstrap target requires finite fixed-effect covariance"
                    .to_string(),
            ));
        }

        let middle =
            symmetrize_matrix(&(&hypothesis.l.values * &vcov * hypothesis.l.values.transpose()));
        if !matrix_is_finite(&middle) {
            return Err(MixedModelError::InvalidArgument(
                "fixed-effect null bootstrap target produced non-finite L V L'".to_string(),
            ));
        }
        let middle_eig = SymmetricEigen::new(middle.clone());
        let middle_max_abs = middle_eig
            .eigenvalues
            .iter()
            .map(|value| value.abs())
            .fold(0.0, f64::max);
        let middle_tol = (1e-10 * middle_max_abs.max(1.0)).max(1e-12);
        let middle_min_abs = middle_eig
            .eigenvalues
            .iter()
            .map(|value| value.abs())
            .fold(f64::INFINITY, f64::min);
        let used_generalized_inverse = middle_min_abs <= middle_tol;
        let middle_inverse = if used_generalized_inverse {
            symmetric_pseudoinverse(&middle, middle_tol)
        } else {
            invert_spd_matrix(&middle, "fixed-effect null bootstrap L V L' matrix")?
        };

        let fitted_contrast = &hypothesis.l.values * &beta_fitted - &hypothesis.rhs.values;
        let beta_null = &beta_fitted
            - &vcov * hypothesis.l.values.transpose() * middle_inverse * fitted_contrast;
        let _beta_null_active = self.fixed_effect_user_beta_to_active_basis(&beta_null)?;

        let mut notes = vec![
            "fixed-effect null target reuses fitted covariance parameters; variance re-estimation under the null is not yet implemented"
                .to_string(),
        ];
        if used_generalized_inverse {
            notes.push(format!(
                "fixed-effect null target used a generalized inverse for L V L' at tolerance {middle_tol}"
            ));
        }

        Ok(FixedEffectNullBootstrapTarget {
            target: BootstrapTarget::fixed_effect_null(
                format!("{} fixed-effect null", hypothesis.label),
                hypothesis.label.clone(),
            ),
            covariance_policy: FixedEffectNullCovariancePolicy::ReuseFittedCovariance,
            coefficient_names: self.coef_names(),
            beta_fitted,
            beta_null,
            theta: self.theta(),
            sigma: self.sigma(),
            reml: self.optsum.reml,
            notes,
        })
    }

    /// Simulate one response vector under a fixed-effect null target.
    pub fn simulate_fixed_effect_null<R: rand::Rng>(
        &self,
        rng: &mut R,
        target: &FixedEffectNullBootstrapTarget,
    ) -> Result<DVector<f64>> {
        if target.covariance_policy != FixedEffectNullCovariancePolicy::ReuseFittedCovariance {
            return Err(MixedModelError::InvalidArgument(
                "unsupported fixed-effect null bootstrap covariance policy".to_string(),
            ));
        }
        if target.theta.len() != self.n_theta()
            || target
                .theta
                .iter()
                .zip(self.theta().iter())
                .any(|(lhs, rhs)| (*lhs - *rhs).abs() > 1e-10)
            || (target.sigma - self.sigma()).abs() > 1e-10
        {
            return Err(MixedModelError::InvalidArgument(
                "fixed-effect null bootstrap target does not match the fitted covariance state"
                    .to_string(),
            ));
        }
        let beta_active = self.fixed_effect_user_beta_to_active_basis(&target.beta_null)?;
        self.simulate_with_active_beta(rng, &beta_active)
    }

    /// Refit the model with a new response vector.
    ///
    /// Replaces the response, rebuilds the cross-product matrices, and
    /// re-runs the full optimization from the original initial parameters.
    ///
    /// Mirrors `refit!(fm, new_y)` in Julia's MixedModels.jl.
    pub fn refit(&mut self, new_y: &[f64]) -> Result<()> {
        if new_y.len() != self.dims.n {
            return Err(MixedModelError::InvalidArgument(format!(
                "Response length {} does not match model ({} observations)",
                new_y.len(),
                self.dims.n
            )));
        }

        let y_max = new_y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let y_min = new_y.iter().cloned().fold(f64::INFINITY, f64::min);
        if (y_max - y_min) < f64::EPSILON {
            return Err(MixedModelError::InvalidArgument(
                "The response is constant and thus model fitting has failed".to_string(),
            ));
        }

        let p = self.feterm.rank;
        for obs in 0..self.dims.n {
            let sw = if self.sqrtwts.is_empty() {
                1.0
            } else {
                self.sqrtwts[obs]
            };
            self.y[obs] = new_y[obs];
            self.xy_mat.xy[(obs, p)] = new_y[obs];
            self.xy_mat.wtxy[(obs, p)] = sw * new_y[obs];
        }

        self.recompute_a_blocks()?;

        // Reset fit state so fit() doesn't reject as AlreadyFitted
        let reml = self.optsum.reml;
        self.optsum.feval = 0;

        // Re-optimize from initial θ
        let initial = self.optsum.initial.clone();
        self.set_theta(&initial)?;
        self.fit(reml)?;
        Ok(())
    }

    /// Hat matrix diagonal (leverage values) for each observation.
    ///
    /// Computes `h_i = ||L⁻¹ vᵢ||²` where `vᵢ` is the i-th column of
    /// the (weighted) design matrix `[ΛZ | X]'`.  The sum equals the
    /// model degrees of freedom (rank of X + RE θ parameters).
    ///
    /// Mirrors `leverage(fm)` in Julia's MixedModels.jl.
    pub fn leverage(&self) -> DVector<f64> {
        let k = self.reterms.len();
        let p = self.dims.p;
        let n = self.dims.n;
        let wtxy = &self.xy_mat.wtxy;
        let pp1 = p + 1; // p fixed effects + 1 response (y slot kept at 0)

        // Cumulative column offsets into the stacked RE vector
        let mut offsets = vec![0usize; k + 1];
        for j in 0..k {
            offsets[j + 1] = offsets[j] + self.reterms[j].n_ranef();
        }
        let nranef_total = offsets[k];

        let mut h = DVector::zeros(n);

        for obs in 0..n {
            // Build vᵢ: weighted design column [Λⱼ' wtzⱼ[:,obs]; ...; wtxy[obs,0..p]; 0]
            let mut v = vec![0.0f64; nranef_total + pp1];

            for j in 0..k {
                let re = &self.reterms[j];
                let vs = re.vsize;
                let r = re.refs[obs] as usize;
                let lambda = &re.lambda;
                let offset = offsets[j] + r * vs;
                // (Λⱼ')_{i,row} = Λⱼ[row,i];  Λ is lower-triangular → row ≥ i
                for i in 0..vs {
                    let mut val = 0.0;
                    for row in i..vs {
                        val += lambda[(row, i)] * re.wtz[(row, obs)];
                    }
                    v[offset + i] = val;
                }
            }
            for q in 0..p {
                v[nranef_total + q] = wtxy[(obs, q)];
            }
            // v[nranef_total + p] = 0  (y slot excluded from leverage)

            // Forward solve L * w = v  (lower-triangular blocked)
            let mut w = vec![0.0f64; nranef_total + pp1];

            // RE blocks j = 0..k
            for j in 0..k {
                let re_j = &self.reterms[j];
                let nranef_j = re_j.n_ranef();

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

            // FE block (k-th block): forward solve L[k,k] * w_k = rhs_k
            let l_kk = self.l_blocks[block_index(k, k)].as_dense();
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
            let mut w_k = vec![0.0f64; pp1];
            w_k.copy_from_slice(&rhs_k);
            solve_lower_block_against_rhs(&MatrixBlock::Dense(l_kk), &mut w_k);

            // h_obs = ||w_RE||² + ||w_FE||²  (exclude w_k[p] = y component)
            let sum_sq: f64 = w[..nranef_total].iter().map(|x| x * x).sum::<f64>()
                + w_k[..p].iter().map(|x| x * x).sum::<f64>();
            h[obs] = sum_sq;
        }

        h
    }

    /// Conditional variance matrices of the random effects.
    ///
    /// Returns one `Vec<DMatrix<f64>>` per RE term.  Each inner vector has one
    /// `vsize × vsize` positive-semi-definite matrix per level of the grouping
    /// factor.  The matrices are the diagonal blocks of `σ² Λ(Λ'Z'ZΛ+I)⁻¹Λ'`.
    ///
    /// Mirrors `condVar(m)` in Julia's MixedModels.jl.
    pub fn cond_var(&self) -> Vec<Vec<DMatrix<f64>>> {
        let k = self.reterms.len();
        let sigma = self.sigma();
        let mut result = Vec::with_capacity(k);

        for j in 0..k {
            let re_j = &self.reterms[j];
            let vs_j = re_j.vsize;
            let n_levels_j = re_j.n_levels();

            // λt = σ * Λ_j'  (vs_j × vs_j)
            let lambda_j = &re_j.lambda;
            let mut lambda_t = DMatrix::zeros(vs_j, vs_j);
            for row in 0..vs_j {
                for col in 0..vs_j {
                    // Λ'[row,col] = Λ[col,row]
                    lambda_t[(row, col)] = sigma * lambda_j[(col, row)];
                }
            }

            // Local row offsets within the sub-L starting at term j
            // Sub-L includes RE terms j..k-1 (no FE block)
            let mut local_off = vec![0usize; k - j + 1];
            for m in 0..(k - j) {
                local_off[m + 1] = local_off[m] + self.reterms[j + m].n_ranef();
            }
            let q_j = local_off[k - j]; // total rows in sub-L

            let mut condvars = Vec::with_capacity(n_levels_j);

            for b in 0..n_levels_j {
                // scratch = zeros(q_j, vs_j); set block at level b to λt
                let mut scratch = DMatrix::zeros(q_j, vs_j);
                for row in 0..vs_j {
                    for col in 0..vs_j {
                        scratch[(b * vs_j + row, col)] = lambda_t[(row, col)];
                    }
                }

                // Forward solve: for each sub-block i (term j+i) in order
                for i in 0..(k - j) {
                    let blk_i = j + i;
                    let nranef_i = self.reterms[blk_i].n_ranef();
                    let off_i = local_off[i];

                    // Subtract cross-block contributions: L[blk_i, blk_prev] * scratch[prev]
                    for prev in 0..i {
                        let blk_prev = j + prev;
                        let nranef_prev = self.reterms[blk_prev].n_ranef();
                        let off_prev = local_off[prev];
                        let l_cross = self.l_blocks[block_index(blk_i, blk_prev)].as_dense();
                        for col in 0..vs_j {
                            for row in 0..nranef_i {
                                let mut dot = 0.0;
                                for c in 0..nranef_prev {
                                    dot += l_cross[(row, c)] * scratch[(off_prev + c, col)];
                                }
                                scratch[(off_i + row, col)] -= dot;
                            }
                        }
                    }

                    // Solve L[blk_i, blk_i] * scratch[i_part] = scratch[i_part]
                    for col in 0..vs_j {
                        let mut rhs: Vec<f64> = (0..nranef_i)
                            .map(|idx| scratch[(off_i + idx, col)])
                            .collect();
                        solve_lower_block_against_rhs(
                            &self.l_blocks[block_index(blk_i, blk_i)],
                            &mut rhs,
                        );
                        for idx in 0..nranef_i {
                            scratch[(off_i + idx, col)] = rhs[idx];
                        }
                    }
                }

                // condvar_b = scratch' * scratch  (vs_j × vs_j)
                condvars.push(scratch.transpose() * &scratch);
            }

            result.push(condvars);
        }

        result
    }

    /// Structural summary of the blocked `A`/`L` system.
    pub fn block_description(&self) -> BlockDescription {
        BlockDescription::from_linear_model(self)
    }

    /// Fixed/random-effects summary table.
    pub fn summary(&self) -> ModelSummary {
        ModelSummary::from_linear_model(self)
    }

    /// Render the model summary as markdown.
    pub fn summary_markdown(&self) -> String {
        self.summary().to_markdown()
    }

    /// Render the model summary as HTML.
    pub fn summary_html(&self) -> String {
        self.summary().to_html()
    }

    /// Render the model summary as LaTeX.
    pub fn summary_latex(&self) -> String {
        self.summary().to_latex()
    }

    /// Number of θ parameters.
    pub fn n_theta(&self) -> usize {
        self.reterms.iter().map(|rt| rt.n_theta()).sum()
    }

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

        for _ in 0..options.requested_replicates {
            let y_sim = self.simulate_fixed_effect_null(&mut rng, target)?;
            let mut work = self.clone();
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

        for _ in 0..options.requested_replicates {
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

    fn refresh_fixed_effect_covariance_matrix(&mut self) {
        self.compiler_artifact.fixed_effect_covariance_matrix =
            Some(self.fixed_effect_covariance_matrix());
    }

    fn refresh_fixed_effect_inference_table(&mut self) {
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

impl std::fmt::Display for LinearMixedModel {
    /// Default print: the compact `ModelPrint` summary (PRD § 15).
    /// Heavier reports stay one explicit method call away
    /// (`audit_report`, `parameterization`, `changes`, `explain_model`).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.print_summary(), f)
    }
}

impl MixedModelFit for LinearMixedModel {
    fn nobs(&self) -> usize {
        self.dims.n
    }

    fn dof(&self) -> usize {
        self.feterm.rank + self.n_theta() + usize::from(self.optsum.sigma.is_none())
    }

    fn coef(&self) -> DVector<f64> {
        let beta = self.fixef();
        let mut full = DVector::from_element(self.feterm.piv.len(), 0.0);
        for (i, &val) in beta.iter().enumerate() {
            if i < self.feterm.piv.len() {
                full[self.feterm.piv[i]] = val;
            }
        }
        full
    }

    fn fixef(&self) -> DVector<f64> {
        self.beta()
    }

    fn coef_names(&self) -> Vec<String> {
        let mut names = self.feterm.cnames.clone();
        // Unpivot
        let mut result = vec![String::new(); names.len()];
        for (i, name) in names.drain(..).enumerate() {
            if i < self.feterm.piv.len() {
                result[self.feterm.piv[i]] = name;
            }
        }
        result
    }

    fn vcov(&self) -> DMatrix<f64> {
        self.vcov_with_sigma(self.sigma())
    }

    fn stderror(&self) -> DVector<f64> {
        let vc = self.vcov();
        DVector::from_iterator(vc.nrows(), (0..vc.nrows()).map(|i| vc[(i, i)].sqrt()))
    }

    fn fitted(&self) -> DVector<f64> {
        let beta = self.beta();
        let x = self.feterm.full_rank_x();
        let mut yhat = x * &beta;

        // Add random effects contribution
        for (rt, b) in self.reterms.iter().zip(self.ranef_b()) {
            // y += Z * b (using sparse multiplication via refs)
            let bvec = DVector::from_column_slice(b.as_slice());
            for (obs, &ref_idx) in rt.refs.iter().enumerate() {
                let r = ref_idx as usize;
                for s in 0..rt.vsize {
                    yhat[obs] += rt.z[(s, obs)] * bvec[r * rt.vsize + s];
                }
            }
        }

        yhat
    }

    fn residuals(&self) -> DVector<f64> {
        let y = self.y();
        let yhat = self.fitted();
        y - yhat
    }

    fn response(&self) -> &DVector<f64> {
        &self.y
    }

    fn model_matrix(&self) -> &DMatrix<f64> {
        &self.feterm.x
    }

    fn objective(&self) -> f64 {
        self.objective_value()
    }

    fn loglikelihood(&self) -> f64 {
        -self.objective_value() / 2.0
    }

    fn formula_label(&self) -> Option<String> {
        Some(self.formula.to_string())
    }

    fn is_fitted(&self) -> bool {
        self.optsum.feval > 0
    }

    fn is_singular(&self) -> bool {
        self.theta_at_lower_bound()
            || self.optimizer_certificate_reports_boundary()
            || self.has_reduced_effective_covariance()
    }

    fn opt_summary(&self) -> &OptSummary {
        &self.optsum
    }

    fn theta(&self) -> Vec<f64> {
        LinearMixedModel::theta(self)
    }

    fn dispersion(&self, sqr: bool) -> f64 {
        let s = self.sigma();
        if sqr && self.optsum.sigma.is_none() {
            s * s
        } else {
            s
        }
    }

    fn ranef(&self) -> Vec<DMatrix<f64>> {
        self.ranef_b()
    }
}

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

pub(crate) fn prediction_interval_cutoff(level: f64) -> Result<f64> {
    use statrs::distribution::{ContinuousCDF, Normal};

    if !(level > 0.0 && level < 1.0) {
        return Err(MixedModelError::InvalidArgument(format!(
            "prediction interval level must be in (0,1); got {level}"
        )));
    }
    Ok(Normal::new(0.0, 1.0)
        .unwrap()
        .inverse_cdf(1.0 - (1.0 - level) / 2.0))
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

// === Helper functions for model construction ===

/// Build the fixed-effects model matrix from formula and data.
/// Capture the canonical level order (and explicit contrast, if present) of
/// every categorical column in the training frame. Numeric columns carry no
/// encoding contract and are skipped.
fn snapshot_training_categorical(
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

fn use_direct_dense_fixed_design(
    formula: &Formula,
    data: &DataFrame,
    policy: FixedDesignBuildPolicy,
) -> bool {
    match policy.preference {
        FixedDesignBackendPreference::Dense => true,
        FixedDesignBackendPreference::Streamed => false,
        FixedDesignBackendPreference::Auto => fixed_terms_are_numeric_only(formula, data),
    }
}

fn fixed_terms_are_numeric_only(formula: &Formula, data: &DataFrame) -> bool {
    formula.fixed_terms.iter().all(|term| match term {
        FixedTerm::Intercept | FixedTerm::NoIntercept => true,
        FixedTerm::Column(name) => data.numeric(name).is_some(),
        FixedTerm::Interaction(vars) => vars.iter().all(|name| data.numeric(name).is_some()),
    })
}

fn build_fixed_effects_matrix(
    formula: &Formula,
    data: &DataFrame,
) -> Result<(DMatrix<f64>, Vec<String>)> {
    Ok(build_fixed_effects_design(formula, data)?.into_parts())
}

fn build_fixed_effects_design(formula: &Formula, data: &DataFrame) -> Result<DenseFixedDesign> {
    use crate::formula::FixedTerm;

    let n = data.nrow();
    let mut columns: Vec<DVector<f64>> = Vec::new();
    let mut names: Vec<String> = Vec::new();

    // Check if we have an intercept
    let has_intercept = formula.has_intercept();

    if has_intercept {
        columns.push(DVector::from_element(n, 1.0));
        names.push("(Intercept)".to_string());
    }

    for term in &formula.fixed_terms {
        match term {
            FixedTerm::Intercept | FixedTerm::NoIntercept => {
                // Already handled
            }
            FixedTerm::Column(name) => match data.column(name) {
                Some(Column::Numeric(v)) => {
                    columns.push(DVector::from_column_slice(v));
                    names.push(name.clone());
                }
                Some(Column::Categorical(cat)) => {
                    for encoded in cat.encoded_columns(name, CategoricalCoding::Treatment) {
                        columns.push(DVector::from_column_slice(&encoded.values));
                        names.push(encoded.name);
                    }
                }
                None => {
                    return Err(MixedModelError::InvalidArgument(format!(
                        "Column '{}' not found in data",
                        name
                    )));
                }
            },
            FixedTerm::Interaction(vars) => {
                // N-way interaction. Each variable contributes a list of
                // (column, label) pairs: numeric → 1 pair (the column itself),
                // categorical(L) → L-1 dummy pairs (skipping the reference
                // level). The interaction is the Cartesian product, with
                // columns multiplied element-wise and labels joined by `:`.
                let per_var = expand_interaction_factors(vars, data, n)?;
                for (col, name) in cartesian_interaction(&per_var, n) {
                    columns.push(col);
                    names.push(name);
                }
            }
        }
    }

    if columns.is_empty() {
        // No fixed effects at all — create an empty matrix
        return DenseFixedDesign::new(DMatrix::zeros(n, 0), vec![]);
    }

    let p = columns.len();
    let mut x = DMatrix::zeros(n, p);
    for (j, col) in columns.iter().enumerate() {
        x.set_column(j, col);
    }

    DenseFixedDesign::new(x, names)
}

/// Per-variable expansion used by interaction terms: numeric → one column,
/// categorical(L) → L-1 dummy columns (skip reference level). Returns one
/// `Vec<(column, label)>` per input variable, in the order they were given.
fn expand_interaction_factors(
    vars: &[String],
    data: &DataFrame,
    n: usize,
) -> Result<Vec<Vec<(DVector<f64>, String)>>> {
    expand_interaction_factors_with_coding(vars, data, n, BasisCoding::Treatment)
}

fn expand_interaction_factors_with_coding(
    vars: &[String],
    data: &DataFrame,
    n: usize,
    coding: BasisCoding,
) -> Result<Vec<Vec<(DVector<f64>, String)>>> {
    let mut per_var: Vec<Vec<(DVector<f64>, String)>> = Vec::with_capacity(vars.len());
    for v in vars {
        per_var.push(expand_factor_columns_with_coding(
            v,
            data,
            "interaction term",
            coding,
        )?);
    }
    let _ = n; // n only used by callers for sanity checks
    Ok(per_var)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BasisCoding {
    Treatment,
    CellMeans,
}

fn categorical_coding(coding: BasisCoding) -> CategoricalCoding {
    match coding {
        BasisCoding::Treatment => CategoricalCoding::Treatment,
        BasisCoding::CellMeans => CategoricalCoding::CellMeans,
    }
}

fn expand_factor_columns_with_coding(
    name: &str,
    data: &DataFrame,
    context: &str,
    coding: BasisCoding,
) -> Result<Vec<(DVector<f64>, String)>> {
    match data.column(name) {
        Some(Column::Numeric(arr)) => Ok(vec![(DVector::from_column_slice(arr), name.to_string())]),
        Some(Column::Categorical(cat)) => Ok(cat
            .encoded_columns(name, categorical_coding(coding))
            .into_iter()
            .map(|column| (DVector::from_column_slice(&column.values), column.name))
            .collect()),
        None => Err(MixedModelError::InvalidArgument(format!(
            "Column '{name}' not found in data ({context})"
        ))),
    }
}

/// Cartesian product of expanded interaction factors. Iterates the FIRST
/// variable's columns slowest (outermost), matching how the Rust crate
/// emits main effects elsewhere. lme4 uses the opposite ordering — column
/// space is identical, but β positions differ; the cross-impl reporter
/// matches by normalized coefficient name to handle this.
fn cartesian_interaction(
    per_var: &[Vec<(DVector<f64>, String)>],
    n: usize,
) -> Vec<(DVector<f64>, String)> {
    let mut acc: Vec<(DVector<f64>, String)> = vec![(DVector::from_element(n, 1.0), String::new())];
    for cols in per_var {
        let mut next = Vec::with_capacity(acc.len() * cols.len());
        for (acc_col, acc_name) in &acc {
            for (c, name) in cols {
                let prod = acc_col.component_mul(c);
                let new_name = if acc_name.is_empty() {
                    name.clone()
                } else {
                    format!("{acc_name}:{name}")
                };
                next.push((prod, new_name));
            }
        }
        acc = next;
    }
    // Drop the seed row when the input was empty (no factors at all).
    if per_var.is_empty() {
        return Vec::new();
    }
    acc
}

fn random_effect_basis_columns(
    term: &crate::formula::FixedTerm,
    data: &DataFrame,
    n: usize,
    coding: BasisCoding,
) -> Result<Vec<(DVector<f64>, String)>> {
    use crate::formula::FixedTerm;

    match term {
        FixedTerm::Intercept | FixedTerm::NoIntercept => Ok(Vec::new()),
        FixedTerm::Column(name) => {
            expand_factor_columns_with_coding(name, data, "random-effect basis", coding)
        }
        FixedTerm::Interaction(vars) => {
            let per_var = expand_interaction_factors_with_coding(vars, data, n, coding)?;
            Ok(cartesian_interaction(&per_var, n))
        }
    }
}

fn random_effect_basis_coding(rt: &crate::formula::RandomTerm) -> BasisCoding {
    if rt
        .terms
        .iter()
        .any(|term| matches!(term, crate::formula::FixedTerm::NoIntercept))
    {
        BasisCoding::CellMeans
    } else {
        BasisCoding::Treatment
    }
}

fn refuse_unsupported_random_covariance(formula: &Formula) -> Result<()> {
    for term in &formula.random_terms {
        if !term.covariance.is_supported_for_fit() {
            return Err(MixedModelError::Unsupported(format!(
                "structured random-effect covariance family `{}` is parsed but not fitted in v1.0; use `|`, `||`, or `diag(...)` for fitted models",
                term.covariance.label()
            )));
        }
    }
    Ok(())
}

/// Build a ReMat from a random term specification and data.
fn build_re_mat(rt: &crate::formula::RandomTerm, data: &DataFrame, n: usize) -> Result<ReMat> {
    use crate::formula::{FixedTerm, GroupingFactor};

    // Get grouping factor
    let (group_name, refs, levels) = match &rt.grouping {
        GroupingFactor::Single(name) => {
            let cat = data.categorical(name).ok_or_else(|| {
                MixedModelError::InvalidArgument(format!(
                    "Grouping factor '{}' not found or not categorical",
                    name
                ))
            })?;
            (name.clone(), cat.refs.clone(), cat.levels.clone())
        }
        GroupingFactor::Interaction(names) | GroupingFactor::Cell(names) => {
            // Create interaction levels
            let cats: Vec<&crate::model::data::CategoricalColumn> = names
                .iter()
                .map(|name| {
                    data.categorical(name).ok_or_else(|| {
                        MixedModelError::InvalidArgument(format!(
                            "Grouping factor '{}' not found",
                            name
                        ))
                    })
                })
                .collect::<Result<Vec<_>>>()?;

            let group_name = names.join(" & ");
            let mut level_map = indexmap::IndexMap::new();
            let mut refs = Vec::with_capacity(n);

            for obs in 0..n {
                let key: String = cats
                    .iter()
                    .map(|c| c.levels[c.refs[obs] as usize].clone())
                    .collect::<Vec<_>>()
                    .join("_");
                let idx = level_map.len();
                let idx = *level_map.entry(key.clone()).or_insert(idx);
                refs.push(idx as u32);
            }

            let levels: Vec<String> = level_map.keys().cloned().collect();
            (group_name, refs, levels)
        }
    };

    // Build the Z matrix (transposed: s × n)
    let mut z_rows: Vec<DVector<f64>> = Vec::new();
    let mut cnames: Vec<String> = Vec::new();

    let has_re_intercept =
        rt.terms.iter().any(|t| matches!(t, FixedTerm::Intercept)) || rt.terms.is_empty();

    if has_re_intercept {
        z_rows.push(DVector::from_element(n, 1.0));
        cnames.push("(Intercept)".to_string());
    }

    let basis_coding = random_effect_basis_coding(rt);
    for term in &rt.terms {
        for (col, name) in random_effect_basis_columns(term, data, n, basis_coding)? {
            z_rows.push(col);
            cnames.push(name);
        }
    }

    let vsize = z_rows.len();
    let mut z = DMatrix::zeros(vsize, n);
    for (i, row) in z_rows.iter().enumerate() {
        z.set_row(i, &row.transpose());
    }

    let mut remat = ReMat::new(group_name, refs, levels, cnames, z);

    if rt.zerocorr || matches!(rt.covariance, RandomCovariance::Diagonal) {
        remat.zerocorr();
    }

    Ok(remat)
}

/// Build the parameter map: Vec<(block_idx, row, col)> for each θ element.
fn build_parmap(reterms: &[ReMat]) -> Vec<(usize, usize, usize)> {
    let mut parmap = Vec::new();
    for (block, rt) in reterms.iter().enumerate() {
        for &ind in &rt.inds {
            let s = rt.vsize;
            let col = ind / s;
            let row = ind % s;
            parmap.push((block, row, col));
        }
    }
    parmap
}

fn kenward_roger_covariance_component_count(reterm: &ReMat) -> usize {
    reterm.inds.len()
}

fn kenward_roger_covariance_component_indices(reterm: &ReMat) -> Vec<(usize, usize)> {
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

fn kenward_roger_response_component(
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

fn matrix_rows(matrix: &DMatrix<f64>) -> Vec<Vec<f64>> {
    (0..matrix.nrows())
        .map(|row| {
            (0..matrix.ncols())
                .map(|col| matrix[(row, col)])
                .collect::<Vec<_>>()
        })
        .collect()
}

fn aliased_fixed_effect_names(coef_names: &[String], pivot: &[usize], rank: usize) -> Vec<String> {
    pivot
        .iter()
        .skip(rank)
        .filter_map(|&index| coef_names.get(index).cloned())
        .collect()
}

fn max_abs_delta(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() != right.len() {
        return None;
    }
    Some(
        left.iter()
            .zip(right.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max),
    )
}

fn matrix_is_finite(matrix: &DMatrix<f64>) -> bool {
    matrix.iter().all(|value| value.is_finite())
}

fn matrix_elementwise_dot(left: &DMatrix<f64>, right: &DMatrix<f64>) -> f64 {
    if left.shape() != right.shape() {
        return f64::NAN;
    }
    left.iter()
        .zip(right.iter())
        .map(|(lhs, rhs)| lhs * rhs)
        .sum()
}

fn matrix_trace(matrix: &DMatrix<f64>) -> f64 {
    let n = matrix.nrows().min(matrix.ncols());
    (0..n).map(|idx| matrix[(idx, idx)]).sum()
}

fn matrix_trace_product(left: &DMatrix<f64>, right: &DMatrix<f64>) -> f64 {
    if left.ncols() != right.nrows() {
        return f64::NAN;
    }
    let mut trace = 0.0;
    let n = left.nrows().min(right.ncols());
    for row in 0..n {
        for col in 0..left.ncols() {
            trace += left[(row, col)] * right[(col, row)];
        }
    }
    trace
}

fn matrix_max_asymmetry(matrix: &DMatrix<f64>) -> f64 {
    if matrix.nrows() != matrix.ncols() {
        return f64::INFINITY;
    }
    let mut max_delta = 0.0_f64;
    for row in 0..matrix.nrows() {
        for col in 0..row {
            max_delta = max_delta.max((matrix[(row, col)] - matrix[(col, row)]).abs());
        }
    }
    max_delta
}

fn invert_spd_matrix(matrix: &DMatrix<f64>, context: &str) -> Result<DMatrix<f64>> {
    if matrix.nrows() != matrix.ncols() {
        return Err(MixedModelError::DimensionMismatch(format!(
            "{context} is {} x {}, expected square",
            matrix.nrows(),
            matrix.ncols()
        )));
    }
    matrix
        .clone()
        .cholesky()
        .map(|chol| chol.inverse())
        .ok_or(MixedModelError::LinAlg(
            crate::error::LinAlgError::NotPositiveDefinite,
        ))
}

fn symmetric_pseudoinverse(matrix: &DMatrix<f64>, tolerance: f64) -> DMatrix<f64> {
    let matrix = symmetrize_matrix(matrix);
    let eig = SymmetricEigen::new(matrix);
    let mut inverse = DMatrix::zeros(eig.eigenvectors.nrows(), eig.eigenvectors.ncols());
    for (index, &eigenvalue) in eig.eigenvalues.iter().enumerate() {
        if eigenvalue.abs() > tolerance {
            let column = eig.eigenvectors.column(index);
            inverse += (column * column.transpose()) * (1.0 / eigenvalue);
        }
    }
    symmetrize_matrix(&inverse)
}

fn matrix_rank(matrix: &DMatrix<f64>, relative_tolerance: f64) -> usize {
    let svd = matrix.clone().svd(false, false);
    let max_singular = svd.singular_values.iter().copied().fold(0.0, f64::max);
    let tolerance = (relative_tolerance * max_singular.max(1.0)).max(1e-12);
    svd.singular_values
        .iter()
        .filter(|value| **value > tolerance)
        .count()
}

fn symmetric_pair_index(row: usize, col: usize, dimension: usize) -> usize {
    debug_assert!(row <= col);
    debug_assert!(col < dimension);
    row * dimension - row.saturating_mul(row.saturating_sub(1)) / 2 + (col - row)
}

fn div_zero(numerator: f64, denominator: f64, tolerance: f64) -> f64 {
    if numerator.abs() < tolerance && denominator.abs() < tolerance {
        1.0
    } else {
        numerator / denominator
    }
}

fn scalar_covariance_variance_step(variance: f64) -> f64 {
    (1e-5 * (1.0 + variance.abs())).max(1e-8)
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

fn two_by_two_covariance_from_theta(theta: [f64; 3]) -> [[f64; 2]; 2] {
    let l00 = theta[0].max(0.0);
    let l10 = theta[1];
    let l11 = theta[2].max(0.0);
    [[l00 * l00, l00 * l10], [l00 * l10, l10 * l10 + l11 * l11]]
}

fn two_by_two_theta_from_covariance(covariance: [[f64; 2]; 2]) -> Option<[f64; 3]> {
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

fn symmetric_2x2_eigenvalues(matrix: [[f64; 2]; 2]) -> (f64, f64) {
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

fn two_by_two_frobenius_norm(matrix: [[f64; 2]; 2]) -> f64 {
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

fn symmetrize_matrix(matrix: &DMatrix<f64>) -> DMatrix<f64> {
    let mut result = matrix.clone();
    for row in 0..matrix.nrows() {
        for col in 0..row {
            let value = 0.5 * (matrix[(row, col)] + matrix[(col, row)]);
            result[(row, col)] = value;
            result[(col, row)] = value;
        }
    }
    result
}

fn finite_difference_steps(theta: &[f64], lower_bounds: &[f64], relative_scale: f64) -> Vec<f64> {
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

fn feasible_central_step(value: f64, lower: f64, requested_step: f64) -> Option<f64> {
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

fn finite_difference_deviance_varpar(
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

fn finite_difference_deviance_varpar_2d(
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

fn assess_fixed_contrast_estimability(
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

struct FixedEffectBootstrapStatistic {
    value: f64,
    numerator_df: Option<f64>,
    label: &'static str,
}

fn fixed_effect_bootstrap_statistic(
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

fn satterthwaite_f_denominator_df(direction_dfs: &[f64], tolerance: f64) -> Option<f64> {
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

fn fixed_effect_test_to_inference_row(
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

fn fixed_effect_inference_reason(test: &FixedEffectTest) -> Option<String> {
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

fn trust_bq_initial_radius(initial_step: &[f64], n_theta: usize) -> f64 {
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

fn trust_bq_final_radius(xtol_abs: &[f64], n_theta: usize) -> f64 {
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
struct TrustBqModelFamilyPolicy {
    initial_radius: f64,
    final_radius: f64,
    max_evaluations: usize,
    ftol_abs: f64,
    ftol_rel: f64,
    max_cross_terms: usize,
    reuse_samples: bool,
    stall_iterations: usize,
    stall_ftol_rel: f64,
    stall_ftol_abs: f64,
    stall_requires_stable_x: bool,
    certificate_ftol_abs: f64,
    certificate_ftol_rel: f64,
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
fn trust_bq_model_family_policy(
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

fn apply_design_compiled_policy(
    formula: &mut Formula,
    recommendations: &[PolicyRecommendation],
) -> Result<Vec<ReductionRecord>> {
    let mut reductions = Vec::new();

    for recommendation in recommendations {
        let Some(term_index) = term_index_from_id(&recommendation.term_id) else {
            return Err(MixedModelError::InvalidArgument(format!(
                "policy recommendation references unknown random term '{}'",
                recommendation.term_id
            )));
        };
        let Some(term) = formula.random_terms.get_mut(term_index) else {
            return Err(MixedModelError::InvalidArgument(format!(
                "policy recommendation references missing random term '{}'",
                recommendation.term_id
            )));
        };

        match recommendation.action {
            PolicyAction::ReduceCovariance => {
                term.zerocorr = true;
                reductions.push(reduction_from_recommendation(
                    recommendation,
                    Some(term.to_string()),
                ));
            }
            PolicyAction::DropUnsupportedBasis => {
                let unsupported = unsupported_basis_from_recommendation(recommendation);
                if unsupported.is_empty() {
                    return Err(MixedModelError::InvalidArgument(format!(
                        "cannot apply unsupported-basis reduction for '{}' without basis payload",
                        recommendation.term_id
                    )));
                }
                let removed = drop_unsupported_basis_terms(term, &unsupported)?;
                if removed.is_empty() {
                    return Err(MixedModelError::InvalidArgument(format!(
                        "unsupported basis for '{}' could not be mapped to formula terms: {}",
                        recommendation.term_id,
                        unsupported.join(", ")
                    )));
                }
                reductions.push(reduction_from_recommendation(
                    recommendation,
                    Some(term.to_string()),
                ));
            }
            PolicyAction::RefuseRandomTermDistribution | PolicyAction::MarkNotAssessable => {
                return Err(MixedModelError::InvalidArgument(format!(
                    "design_compiled refused {}: {}",
                    recommendation.source_syntax, recommendation.reason
                )));
            }
        }
    }

    Ok(reductions)
}

fn term_index_from_id(term_id: &str) -> Option<usize> {
    term_id.strip_prefix('r')?.parse().ok()
}

fn reduction_from_recommendation(
    recommendation: &PolicyRecommendation,
    replacement_term: Option<String>,
) -> ReductionRecord {
    ReductionRecord {
        trigger: ReductionTrigger::DesignTime,
        phase: "design_compiled".to_string(),
        reason: recommendation.reason.clone(),
        affected_term: recommendation.term_id.clone(),
        replacement_term,
        inference_consequence: recommendation.inference_consequence.clone(),
        diagnostics: recommendation.diagnostics.clone(),
    }
}

fn unsupported_basis_from_recommendation(recommendation: &PolicyRecommendation) -> Vec<String> {
    recommendation
        .diagnostics
        .first()
        .and_then(|diagnostic| diagnostic.payload.get("unsupported_basis"))
        .and_then(|value| value.as_array())
        .map(|values| {
            values
                .iter()
                .filter_map(|value| value.as_str().map(str::to_string))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn drop_unsupported_basis_terms(
    term: &mut RandomTerm,
    unsupported_basis: &[String],
) -> Result<Vec<String>> {
    let mut removed = Vec::new();
    term.terms.retain(|fixed_term| {
        if matches!(fixed_term, FixedTerm::Intercept | FixedTerm::NoIntercept) {
            return true;
        }
        let label = fixed_term.to_string();
        if unsupported_basis.iter().any(|basis| basis == &label) {
            removed.push(label);
            false
        } else {
            true
        }
    });

    let has_intercept = term
        .terms
        .iter()
        .any(|fixed_term| matches!(fixed_term, FixedTerm::Intercept))
        || term.terms.is_empty();
    let has_basis = term
        .terms
        .iter()
        .any(|fixed_term| !matches!(fixed_term, FixedTerm::Intercept | FixedTerm::NoIntercept));
    if !has_intercept && !has_basis {
        return Err(MixedModelError::InvalidArgument(
            "design_compiled would remove every random-effect basis direction".to_string(),
        ));
    }

    Ok(removed)
}

fn user_basis_label(name: &str) -> String {
    if name == "(Intercept)" {
        "intercept".to_string()
    } else {
        name.to_string()
    }
}

fn orient_eigenvector(mut vector: Vec<f64>) -> Vec<f64> {
    let pivot = vector
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| {
            left.abs()
                .partial_cmp(&right.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, _)| idx);

    if let Some(idx) = pivot {
        if vector[idx] < 0.0 {
            for value in &mut vector {
                *value = -*value;
            }
        }
    }

    vector
}

fn format_loading_summary(loadings: &[BasisLoading]) -> String {
    let mut parts = String::new();
    for (idx, loading) in loadings.iter().enumerate() {
        let value = if loading.loading.abs() < 5e-13 {
            0.0
        } else {
            loading.loading
        };
        if idx == 0 {
            parts.push_str(&format!("{value:.3}*{}", loading.basis));
        } else if value < 0.0 {
            parts.push_str(&format!(" - {:.3}*{}", value.abs(), loading.basis));
        } else {
            parts.push_str(&format!(" + {value:.3}*{}", loading.basis));
        }
    }
    parts
}

fn source_syntax_for_term(terms: &[crate::compiler::RandomTermIr], term_id: &str) -> String {
    terms
        .iter()
        .find(|term| term.id == term_id)
        .map(|term| term.source_syntax.text.clone())
        .unwrap_or_else(|| term_id.to_string())
}

/// Build a "drop the off-axis column" rewrite for a rank-2 random term.
///
/// Returns `None` if the basis is not exactly two columns or the kept column
/// cannot be addressed by the simple `(1 | g)` / `(0 + x | g)` template.
fn suggest_drop_off_axis(
    grouping: &str,
    basis_names: &[String],
    keep_idx: usize,
) -> Option<String> {
    if basis_names.len() != 2 || keep_idx >= basis_names.len() {
        return None;
    }
    let kept = &basis_names[keep_idx];
    if kept.eq_ignore_ascii_case("intercept") || kept == "(Intercept)" {
        Some(format!("(1 | {grouping})"))
    } else {
        Some(format!("(0 + {kept} | {grouping})"))
    }
}

/// Detect whether a reduced-rank random-effect term has a single supported
/// direction that loads almost entirely on one user-facing basis column.
///
/// Returns a structured `InterpretableSubmodel` suggestion if so, or `None`
/// when the rank gate, dominance threshold, or formula rewrite are not met.
/// Never refits the model: the suggestion is metadata only.
// TODO(bd-01KQ8FSZPCBTWWS2Q11WWMQ2VY-followup): generalise to requested_rank > 2
// once the rewrite spec for higher-rank submodels exists.
fn detect_interpretable_submodel(
    pairs: &[(f64, Vec<f64>)],
    requested_basis: &[String],
    requested_rank: usize,
    rank_tolerance: f64,
    sigma_sq: f64,
    semantic_terms: &[crate::compiler::RandomTermIr],
    term_id: &str,
) -> Option<InterpretableSubmodel> {
    if requested_rank != 2 {
        return None;
    }

    let supported: Vec<&(f64, Vec<f64>)> = pairs
        .iter()
        .filter(|(eig, _)| eig.max(0.0) > rank_tolerance)
        .collect();
    if supported.len() != 1 {
        return None;
    }
    let supported_pair = supported[0];
    if supported_pair.1.len() != requested_basis.len() {
        return None;
    }

    let oriented = orient_eigenvector(supported_pair.1.clone());
    let (keep_idx, dominant_abs) = oriented
        .iter()
        .enumerate()
        .map(|(idx, value)| (idx, value.abs()))
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))?;
    if dominant_abs < DOMINANT_LOADING_THRESHOLD {
        return None;
    }

    let term = semantic_terms.iter().find(|term| term.id == term_id)?;
    if !matches!(term.covariance, crate::compiler::CovarianceForm::Full) {
        return None;
    }
    let basis_names: Vec<String> = term.basis.iter().map(|coef| coef.name.clone()).collect();
    if basis_names.len() != requested_basis.len() {
        return None;
    }
    let grouping_label = term.group.label();
    let suggested_formula = suggest_drop_off_axis(&grouping_label, &basis_names, keep_idx)?;

    let mut loadings_dominant = oriented
        .iter()
        .zip(requested_basis.iter())
        .map(|(loading, basis)| DominantLoading {
            basis: basis.clone(),
            loading: if loading.abs() < 5e-13 { 0.0 } else { *loading },
        })
        .collect::<Vec<_>>();
    loadings_dominant.sort_by(|a, b| {
        b.loading
            .abs()
            .partial_cmp(&a.loading.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let unsupported_eigenvalue = pairs
        .iter()
        .map(|(eig, _)| eig.max(0.0))
        .filter(|eig| *eig <= rank_tolerance)
        .fold(0.0_f64, f64::max);
    let safe_sigma_sq = sigma_sq.max(f64::EPSILON);
    let objective_gap = (1.0 + unsupported_eigenvalue / safe_sigma_sq).ln().max(0.0);
    let within_tolerance =
        objective_gap.is_finite() && objective_gap <= INTERPRETABLE_GAP_TOLERANCE;

    Some(InterpretableSubmodel {
        suggested_formula,
        loadings_dominant,
        objective_gap,
        within_tolerance,
    })
}

#[cfg(test)]
mod tests;
