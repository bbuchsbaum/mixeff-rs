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
use std::sync::Arc;

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

mod active_face;

mod blocks;
pub(crate) use blocks::*;

mod bootstrap;
pub use bootstrap::{
    parametricbootstrap, try_parametricbootstrap, BootstrapFailedRefitPolicy, BootstrapInterval,
    BootstrapIntervalMethod, BootstrapQuantile, BootstrapRefitOptions, BootstrapReplicate,
    BootstrapRunMetadata, BootstrapRunPayload, BootstrapSeedRecord, BootstrapTarget,
    BootstrapTargetKind, FixedEffectBootstrapOptions, FixedEffectNullBootstrapTarget,
    FixedEffectNullCovariancePolicy, MixedModelBootstrap, BOOTSTRAP_RUN_SCHEMA,
    BOOTSTRAP_RUN_SCHEMA_VERSION,
};
use bootstrap::{quantile_sorted, validate_level};

mod predict;

mod optimizer;
use optimizer::*;

mod inference;
use inference::*;

/// Long-running engine phase reported to a host progress callback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FitProgressPhase {
    /// Profiled LMM covariance-parameter optimization.
    LmmOptimizer,
    /// Joint GLMM fixed-effect/covariance-parameter optimization.
    JointGlmmOptimizer,
    /// A PIRLS conditional-mode solve.
    Pirls,
    /// A parametric or resampling bootstrap replicate loop.
    Bootstrap,
}

/// One throttled progress event emitted by a long-running fit loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct FitProgress {
    /// Engine phase emitting this event.
    pub phase: FitProgressPhase,
    /// Current evaluation, iteration, or replicate in this phase.
    pub current: usize,
    /// Known phase total, when the driver has one.
    pub total: Option<usize>,
}

/// Cloneable host callback used for progress reporting and interruption.
///
/// Returning an error stops the active engine loop immediately. Hosts can use
/// this to translate their native interrupt mechanism (for example,
/// `R_CheckUserInterrupt`) into [`MixedModelError::Interrupted`]. The callback
/// is invoked at most once per `every` units of progress, and when a known
/// total is reached.
#[derive(Clone)]
pub struct FitProgressCallback {
    callback: Arc<dyn Fn(FitProgress) -> Result<()> + Send + Sync + 'static>,
    every: usize,
}

impl std::fmt::Debug for FitProgressCallback {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("FitProgressCallback")
            .field("every", &self.every)
            .finish_non_exhaustive()
    }
}

impl FitProgressCallback {
    /// Create an unthrottled callback (one event per driver progress unit).
    pub fn new<F>(callback: F) -> Self
    where
        F: Fn(FitProgress) -> Result<()> + Send + Sync + 'static,
    {
        Self {
            callback: Arc::new(callback),
            every: 1,
        }
    }

    /// Invoke the callback at most once per `every` units of progress.
    pub fn with_interval(mut self, every: usize) -> Self {
        self.every = every.max(1);
        self
    }

    pub(crate) fn report_if_due(
        &self,
        phase: FitProgressPhase,
        current: usize,
        total: Option<usize>,
        last_reported: &mut usize,
    ) -> Result<()> {
        let final_event = total.is_some_and(|total| current >= total);
        if current.saturating_sub(*last_reported) < self.every && !final_event {
            return Ok(());
        }
        (self.callback)(FitProgress {
            phase,
            current,
            total,
        })
        .map_err(|error| match error {
            MixedModelError::Interrupted(_) => error,
            other => MixedModelError::Interrupted(other.to_string()),
        })?;
        *last_reported = current;
        Ok(())
    }
}

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
    /// Skip the optimizer certificate's finite-difference derivative checks
    /// after fitting. Set only on internal bootstrap-replicate refits, where
    /// per-fit derivative diagnostics are never read; the certificate then
    /// records the checks explicitly as not assessed.
    pub(crate) suppress_derivative_diagnostics: bool,
    /// Opt-in TrustBQ warm-start ladder, carried from
    /// [`OptimizerControl::trust_bq_start_ladder`]. Defaults to `Off`.
    pub(crate) trust_bq_start_ladder: TrustBqStartLadder,
    /// Opt-in TrustBQ exact-sample reuse policy override, carried from
    /// [`OptimizerControl::trust_bq_sample_reuse`]. Defaults to the family
    /// policy.
    pub(crate) trust_bq_sample_reuse: TrustBqSampleReuse,
    /// Opt-in post-fit active-face refit for singular vector blocks, carried
    /// from [`OptimizerControl::active_face_refit`]. Defaults to `Off`.
    pub(crate) active_face_refit: ActiveFaceRefit,
    /// Optional host progress/interrupt callback inherited by refits.
    pub(crate) progress_callback: Option<FitProgressCallback>,
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

/// Opt-in TrustBQ warm-start ladder strategy.
///
/// Experimental and benchmark-gated: the default is
/// [`TrustBqStartLadder::Off`], which keeps the single-start TrustBQ
/// behavior documented in `docs/optimizer_profiles.md`. Ladders only apply
/// to the native TrustBQ LMM path; other optimizers ignore this control.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum TrustBqStartLadder {
    /// Single-start TrustBQ (the default).
    #[default]
    Off,
    /// Optimize the zero-correlation (diagonal-only) covariance first, then
    /// use that optimum as the warm start for the full-covariance
    /// optimization. The two stages share one evaluation budget and both
    /// stages' evaluations are counted in `feval`.
    DiagonalFirst,
}

/// Opt-in exact interpolation-sample reuse policy for the native TrustBQ path.
///
/// Experimental and benchmark-gated: the default is
/// [`TrustBqSampleReuse::FamilyPolicy`], which keeps the central TrustBQ model
/// family policy unchanged. Use the other modes only for A/B benchmark runs or
/// tightly scoped diagnostics; they can alter optimizer traces and reported
/// evaluation counts even when the final fit is numerically unchanged.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum TrustBqSampleReuse {
    /// Use the model-family policy in `trust_bq_model_family_policy` (the
    /// default). Today that means exact sample reuse only for crossed/large
    /// theta families.
    #[default]
    FamilyPolicy,
    /// Disable exact sample reuse for every TrustBQ family.
    Disabled,
    /// Enable exact sample reuse for every TrustBQ family.
    AllFamilies,
}

impl TrustBqSampleReuse {
    /// Resolve the effective `reuse_samples` flag for a single TrustBQ solve,
    /// given the value the model-family policy would otherwise use.
    ///
    /// This is the single source of truth for every native-TrustBQ option
    /// site (main θ-optimization, diagonal warm start, and the active-face
    /// refit sub-solve) so the override cannot silently miss a path.
    pub(crate) fn resolve(self, family_policy_reuse: bool) -> bool {
        match self {
            TrustBqSampleReuse::FamilyPolicy => family_policy_reuse,
            TrustBqSampleReuse::Disabled => false,
            TrustBqSampleReuse::AllFamilies => true,
        }
    }
}

/// Opt-in post-fit active-face refit for singular vector random-effect
/// blocks.
///
/// Experimental and benchmark-gated: the default is [`ActiveFaceRefit::Off`],
/// which leaves fits untouched. When enabled, a fitted vector block whose
/// covariance eigendecomposition shows a lower-rank face (by the same
/// `effective_rank_tolerance` the effective-covariance summaries use) is
/// re-optimized on that face — `r(r+1)/2` face coordinates instead of the
/// term's `k(k+1)/2` theta coordinates — and the refit is kept only when the
/// objective strictly improves. The refit is audit-visible through an
/// `ACTIVE_FACE(<rank>:<evals>:<certified|uncertified>): <status>` return
/// value; `certified` means a finite-difference probe of every dropped
/// direction found no material descent off the face.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum ActiveFaceRefit {
    /// No active-face continuation (the default).
    #[default]
    Off,
    /// Detect lower-rank faces after the primary optimizer stops and
    /// re-optimize on them (best-effort, improvement-gated).
    Experimental,
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
    /// Opt-in TrustBQ warm-start ladder. Defaults to
    /// [`TrustBqStartLadder::Off`].
    pub trust_bq_start_ladder: TrustBqStartLadder,
    /// Opt-in exact-sample reuse override for native TrustBQ. Defaults to
    /// [`TrustBqSampleReuse::FamilyPolicy`].
    pub trust_bq_sample_reuse: TrustBqSampleReuse,
    /// Opt-in post-fit active-face refit for singular vector blocks.
    /// Defaults to [`ActiveFaceRefit::Off`].
    pub active_face_refit: ActiveFaceRefit,
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

    /// Opt into a TrustBQ warm-start ladder.
    pub fn with_trust_bq_start_ladder(mut self, ladder: TrustBqStartLadder) -> Self {
        self.trust_bq_start_ladder = ladder;
        self
    }

    /// Override exact interpolation-sample reuse for native TrustBQ.
    pub fn with_trust_bq_sample_reuse(mut self, reuse: TrustBqSampleReuse) -> Self {
        self.trust_bq_sample_reuse = reuse;
        self
    }

    /// Opt into the experimental post-fit active-face refit.
    pub fn with_active_face_refit(mut self, refit: ActiveFaceRefit) -> Self {
        self.active_face_refit = refit;
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
        if self.trust_bq_start_ladder != TrustBqStartLadder::Off {
            fields.push("trust_bq_start_ladder".to_string());
        }
        if self.trust_bq_sample_reuse != TrustBqSampleReuse::FamilyPolicy {
            fields.push("trust_bq_sample_reuse".to_string());
        }
        if self.active_face_refit != ActiveFaceRefit::Off {
            fields.push("active_face_refit".to_string());
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
    /// Optional throttled host progress/interrupt callback.
    pub progress_callback: Option<FitProgressCallback>,
}

impl FitOptions {
    /// Options requesting a maximum-likelihood fit.
    pub fn ml() -> Self {
        Self {
            criterion: ModelCriterion::Ml,
            optimizer_control: OptimizerControl::default(),
            progress_callback: None,
        }
    }

    /// Options requesting a restricted-maximum-likelihood fit.
    pub fn reml() -> Self {
        Self {
            criterion: ModelCriterion::Reml,
            optimizer_control: OptimizerControl::default(),
            progress_callback: None,
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

    /// Attach a host progress/interrupt callback to this fit and later refits.
    pub fn with_progress_callback(mut self, callback: FitProgressCallback) -> Self {
        self.progress_callback = Some(callback);
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
        let feterm = feterm_for_fixed_design(
            &raw_fixed_design,
            &mut compiler_artifact,
            fixed_design_policy,
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
        ordered_reterms.sort_by_key(|(_, remat)| std::cmp::Reverse(remat.n_ranef()));
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

        let training_categorical = predict::snapshot_training_categorical(data);

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
            suppress_derivative_diagnostics: false,
            trust_bq_start_ladder: TrustBqStartLadder::default(),
            trust_bq_sample_reuse: TrustBqSampleReuse::default(),
            active_face_refit: ActiveFaceRefit::default(),
            progress_callback: None,
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

        self.trust_bq_start_ladder = control.trust_bq_start_ladder;
        self.trust_bq_sample_reuse = control.trust_bq_sample_reuse;
        self.active_face_refit = control.active_face_refit;

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
            self.a_blocks[idx] = finalize_fixed_re_block(block, k);
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

// === Helper functions for model construction ===

/// Rank tolerance for fixed-effects rank/pivot detection. Must match the
/// default used by [`crate::linalg::stats_rank`] so the streamed Gram
/// certificate and the dense Householder fallback test the same boundary.
const FIXED_EFFECTS_RANK_TOLERANCE: f64 = 1e-8;

/// Rank/pivot seam for the streamed fixed-design backend.
///
/// Dense backends keep the exact pivoted-QR path unchanged. Streamed
/// backends first try to certify full column rank from the (never
/// densified) Gram matrix `X'X`; a certified design skips the dense
/// Householder pass and the pivoted copy entirely — the result is
/// byte-identical to `FeTerm::new`'s full-rank early return. When the
/// certificate is ambiguous (possible rank deficiency or conditioning
/// beyond the Gram safety margin) the exact dense `stats_rank` path runs
/// as before, so Householder pivot parity is preserved; the taken path
/// and its cost are recorded as a construction diagnostic either way.
fn feterm_for_fixed_design(
    raw_fixed_design: &FixedDesign,
    compiler_artifact: &mut CompiledModelArtifact,
    policy: FixedDesignBuildPolicy,
) -> FeTerm {
    if raw_fixed_design.storage() != FixedDesignStorage::Streamed {
        return FeTerm::new(
            raw_fixed_design.materialize_dense(),
            raw_fixed_design.column_names().to_vec(),
        );
    }

    let certificate = crate::linalg::gram_full_rank_certificate(
        &raw_fixed_design.xtx(),
        FIXED_EFFECTS_RANK_TOLERANCE,
        crate::linalg::GRAM_CERTIFICATE_SAFETY_FACTOR,
    );
    compiler_artifact
        .diagnostics
        .push(streamed_rank_path_diagnostic(
            &certificate,
            raw_fixed_design,
            policy,
        ));
    if certificate.is_certified() {
        FeTerm::with_certified_full_rank(
            raw_fixed_design.materialize_dense(),
            raw_fixed_design.column_names().to_vec(),
        )
    } else {
        FeTerm::new(
            raw_fixed_design.materialize_dense(),
            raw_fixed_design.column_names().to_vec(),
        )
    }
}

/// Diagnostic recording which rank/pivot path a streamed fixed design
/// took at construction, and — for the dense fallback — whether the
/// materialized pass exceeded the policy's dense-bytes bound.
fn streamed_rank_path_diagnostic(
    certificate: &crate::linalg::GramRankCertificate,
    raw_fixed_design: &FixedDesign,
    policy: FixedDesignBuildPolicy,
) -> Diagnostic {
    let dense_bytes = raw_fixed_design.dense_bytes();
    let over_dense_bound = dense_bytes > policy.max_dense_bytes;
    let (severity, message, actions) = if certificate.is_certified() {
        (
            DiagnosticSeverity::Info,
            format!(
                "streamed fixed-effect rank/pivot: Gram certificate established full rank \
                 (min diagonal ratio {:.3e}); dense Householder pass and pivoted copy skipped",
                certificate.min_ratio()
            ),
            vec![
                "no action required; rank detection stayed on the streamed path".to_string(),
                "the model matrix itself is still materialized once for downstream surfaces"
                    .to_string(),
            ],
        )
    } else if over_dense_bound {
        (
            DiagnosticSeverity::Warning,
            format!(
                "streamed fixed-effect rank/pivot: Gram certificate ambiguous \
                 (min diagonal ratio {:.3e}); exact dense Householder pass materialized \
                 {dense_bytes} bytes, exceeding the backend policy bound of {} bytes",
                certificate.min_ratio(),
                policy.max_dense_bytes
            ),
            vec![
                "the design may be rank-deficient or ill-conditioned; the exact dense pivot \
                 was computed for correctness"
                    .to_string(),
                "if construction memory is a concern, simplify or re-parameterize the fixed \
                 effects so the design is comfortably full rank"
                    .to_string(),
            ],
        )
    } else {
        (
            DiagnosticSeverity::Info,
            format!(
                "streamed fixed-effect rank/pivot: Gram certificate ambiguous \
                 (min diagonal ratio {:.3e}); fell back to the exact dense Householder pass",
                certificate.min_ratio()
            ),
            vec![
                "no action required; rank and pivot are exact (Householder parity preserved)"
                    .to_string(),
            ],
        )
    };
    let mut diagnostic = Diagnostic::new(
        DiagnosticCode::SupportNote,
        severity,
        DiagnosticStage::DesignAudit,
        message,
    )
    .with_suggested_actions(actions);
    diagnostic.payload.insert(
        "diagnostic_kind".to_string(),
        serde_json::json!("fixed_design_rank_path"),
    );
    diagnostic.payload.insert(
        "rank_path".to_string(),
        serde_json::json!(if certificate.is_certified() {
            "streamed_gram_certified"
        } else {
            "dense_householder_fallback"
        }),
    );
    diagnostic.payload.insert(
        "gram_min_diagonal_ratio".to_string(),
        serde_json::json!(certificate.min_ratio()),
    );
    diagnostic
        .payload
        .insert("dense_bytes".to_string(), serde_json::json!(dense_bytes));
    diagnostic.payload.insert(
        "policy_max_dense_bytes".to_string(),
        serde_json::json!(policy.max_dense_bytes),
    );
    diagnostic
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
