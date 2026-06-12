use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::model::data::DataFrame;
use crate::types::{ConvergenceStatus, OptSummary};

use super::audit::{audit_design, DesignAudit, OptimizerCertificate};
use super::diagnostics::{Diagnostic, DiagnosticCode};
use super::estimability::{EstimabilityAssessment, ReliabilityGrade};
use super::ir::{CovarianceSupportStatus, SemanticModel};
use super::policy::{
    recommend_policy, CompilerPolicy, CompilerThresholds, PolicyAction, PolicyRecommendation,
    RandomStrategy,
};
use super::report::ModelAuditReport;
use super::theta_map::{
    CovarianceFamily, CovarianceFamilyTransition, ParameterConstraint, ParameterStatus, ThetaMap,
    ThetaSlot,
};

pub const COMPILED_ARTIFACT_SCHEMA: &str = "mixedmodels.compiled_model_artifact";
pub const COMPILED_ARTIFACT_SCHEMA_VERSION: u32 = 1;
pub const MODEL_STATE_SUMMARY_SCHEMA: &str = "mixedmodels.model_state_summary";
pub const MODEL_STATE_SUMMARY_SCHEMA_VERSION: u32 = 1;
pub const FIXED_EFFECT_INFERENCE_TABLE_SCHEMA: &str = "mixedmodels.fixed_effect_inference_table";
pub const FIXED_EFFECT_INFERENCE_TABLE_SCHEMA_VERSION: &str = "1.0.0";
pub const FIXED_EFFECT_INFERENCE_TABLE_NAME: &str = "fixed_effect_inference";
pub const FIXED_EFFECT_COVARIANCE_MATRIX_SCHEMA: &str =
    "mixedmodels.fixed_effect_covariance_matrix";
pub const FIXED_EFFECT_COVARIANCE_MATRIX_SCHEMA_VERSION: &str = "1.0.0";
pub const FIXED_EFFECT_COVARIANCE_MATRIX_NAME: &str = "fixed_effect_covariance_matrix";

/// Threshold above which a single basis column is treated as carrying the
/// full mass of a reduced-rank covariance direction. A direction with one
/// loading whose absolute value meets or exceeds this cutoff is "dominant"
/// on that column, and the simpler formula that drops every other column
/// from that random-effect term is reported as a candidate submodel.
///
/// 0.95 is conservative: the squared loading explains >= 90% of the
/// direction's variance, leaving <= 10% for the remaining columns combined.
pub const DOMINANT_LOADING_THRESHOLD: f64 = 0.95;

/// Tolerance on the approximate `-2 * log-likelihood` gap below which an
/// `InterpretableSubmodel` is reported as practically equivalent to the
/// fitted reduced-rank model.
pub const INTERPRETABLE_GAP_TOLERANCE: f64 = 0.5;

/// Schema metadata included in serializable compiler artifacts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchemaMetadata {
    pub schema_name: String,
    pub schema_version: u32,
    pub crate_version: Option<String>,
}

impl SchemaMetadata {
    pub fn compiled_model_artifact() -> Self {
        Self {
            schema_name: COMPILED_ARTIFACT_SCHEMA.to_string(),
            schema_version: COMPILED_ARTIFACT_SCHEMA_VERSION,
            crate_version: Some(env!("CARGO_PKG_VERSION").to_string()),
        }
    }
}

/// Reproducibility record for deterministic compiler/fit artifacts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReproducibilityRecord {
    pub fit_intent: FitIntent,
    pub seed: Option<u64>,
    pub random_state_used: bool,
    pub thresholds: Vec<(String, String)>,
}

impl Default for ReproducibilityRecord {
    fn default() -> Self {
        Self {
            fit_intent: FitIntent::default(),
            seed: None,
            random_state_used: false,
            thresholds: default_thresholds(),
        }
    }
}

impl ReproducibilityRecord {
    pub fn with_thresholds(thresholds: &CompilerThresholds) -> Self {
        Self {
            thresholds: thresholds.reproducibility_entries(),
            ..Self::default()
        }
    }

    pub fn with_policy(policy: &CompilerPolicy) -> Self {
        let mut thresholds = policy.thresholds.reproducibility_entries();
        thresholds.push((
            "apply_design_time_reductions".to_string(),
            policy.apply_design_time_reductions.to_string(),
        ));
        Self {
            fit_intent: fit_intent_for_policy(policy),
            seed: None,
            random_state_used: false,
            thresholds,
        }
    }
}

/// Model-family boundary metadata shared by LMM and GLMM artifacts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelBoundary {
    pub model_kind: ModelKind,
    pub response_distribution: String,
    pub link: String,
    pub objective_approximation: ObjectiveApproximation,
    pub optimizer_certificate_scope: OptimizerCertificateScope,
    pub covariance_derivatives: DerivativeAvailability,
    pub inference_availability: InferenceAvailability,
}

impl ModelBoundary {
    pub fn lmm() -> Self {
        Self {
            model_kind: ModelKind::LinearMixedModel,
            response_distribution: "gaussian".to_string(),
            link: "identity".to_string(),
            objective_approximation: ObjectiveApproximation::ExactGaussian,
            optimizer_certificate_scope: OptimizerCertificateScope::ExactObjective,
            covariance_derivatives: DerivativeAvailability::NotAvailable {
                reason: "compiler v0 does not expose covariance derivative certificates"
                    .to_string(),
            },
            inference_availability: InferenceAvailability::NotAssessed {
                reason: "finite-sample inference is not implemented in compiler v0".to_string(),
            },
        }
    }

    pub fn glmm(
        family: impl Into<String>,
        link: impl Into<String>,
        approximation: ObjectiveApproximation,
    ) -> Self {
        Self {
            model_kind: ModelKind::GeneralizedLinearMixedModel,
            response_distribution: family.into(),
            link: link.into(),
            objective_approximation: approximation,
            optimizer_certificate_scope: OptimizerCertificateScope::ApproximatedObjective,
            covariance_derivatives: DerivativeAvailability::NotAvailable {
                reason:
                    "GLMM covariance derivatives are not certified for the objective approximation"
                        .to_string(),
            },
            inference_availability: InferenceAvailability::Unsupported {
                reason:
                    "LMM finite-sample methods such as Satterthwaite/KR are unsupported for GLMMs in compiler v0"
                        .to_string(),
            },
        }
    }
}

/// Post-fit GLMM estimation metadata for downstream wrappers.
///
/// This is intentionally separate from optimizer certificates: it records which
/// GLMM objective family was actually returned, even when a failed joint
/// attempt falls back to the supported fast-PIRLS path.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GlmmFitMetadata {
    pub estimation_method: String,
    pub objective_definition: String,
    pub response_constants: String,
    pub n_agq: usize,
    pub optimizer_backend: String,
    pub optimizer: String,
    pub optimizer_status: String,
    pub optimizer_convergence_status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub optimizer_feval: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub optimizer_max_feval: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub optimizer_fit_log_len: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub optimizer_source: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub caller_set_fields: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fallback_status: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub family_parameters: BTreeMap<String, f64>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub family_parameter_sources: BTreeMap<String, String>,
}

impl GlmmFitMetadata {
    pub fn from_opt_summary(opt: &OptSummary) -> Self {
        let status = opt.return_value.as_str();
        let is_fallback = status.starts_with("JOINT_LAPLACE_FALLBACK_FAST_PIRLS")
            || status.starts_with("JOINT_AGQ_FALLBACK_FAST_PIRLS")
            || status.starts_with("EXPERIMENTAL_JOINT_FALLBACK_FAST_PIRLS");
        let is_joint = status.starts_with("JOINT_LAPLACE:")
            || status.starts_with("JOINT_LAPLACE_FAILED:")
            || status.starts_with("JOINT_AGQ:")
            || status.starts_with("JOINT_AGQ_FAILED:")
            || status.starts_with("EXPERIMENTAL_JOINT:")
            || status.starts_with("EXPERIMENTAL_JOINT_FAILED:");
        let (estimation_method, objective_definition, response_constants, fallback_status) =
            if is_fallback {
                (
                    "fallback_fast_pirls",
                    "profiled_glmm_deviance",
                    "dropped",
                    Some("fallback_fast_pirls"),
                )
            } else if status.starts_with("JOINT_AGQ:") || status.starts_with("JOINT_AGQ_FAILED:") {
                ("joint_agq", "joint_glmm_agq_deviance", "included", None)
            } else if is_joint {
                (
                    "joint_laplace",
                    "joint_glmm_laplace_deviance",
                    "included",
                    None,
                )
            } else {
                (
                    "fast_pirls_profiled",
                    "profiled_glmm_deviance",
                    "dropped",
                    None,
                )
            };

        Self {
            estimation_method: estimation_method.to_string(),
            objective_definition: objective_definition.to_string(),
            response_constants: response_constants.to_string(),
            n_agq: opt.n_agq,
            optimizer_backend: opt.backend_name().to_string(),
            optimizer: opt.optimizer_name().to_string(),
            optimizer_status: status.to_string(),
            optimizer_convergence_status: convergence_status_label(opt.convergence_status())
                .to_string(),
            optimizer_feval: (opt.feval >= 0).then_some(opt.feval),
            optimizer_max_feval: (opt.max_feval > 0).then_some(opt.max_feval),
            optimizer_fit_log_len: (!opt.fit_log.is_empty()).then_some(opt.fit_log.len()),
            optimizer_source: (!opt.caller_set_fields.is_empty()
                || opt.optimizer_source_name() != "auto")
                .then(|| opt.optimizer_source_name().to_string()),
            caller_set_fields: opt.caller_set_fields.clone(),
            fallback_status: fallback_status.map(str::to_string),
            family_parameters: BTreeMap::new(),
            family_parameter_sources: BTreeMap::new(),
        }
    }
}

fn convergence_status_label(status: ConvergenceStatus) -> &'static str {
    match status {
        ConvergenceStatus::Converged => "converged",
        ConvergenceStatus::BudgetExhausted => "budget_exhausted",
        ConvergenceStatus::RoundoffLimited => "roundoff_limited",
        ConvergenceStatus::Failed => "failed",
        ConvergenceStatus::NotRun => "not_run",
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    LinearMixedModel,
    GeneralizedLinearMixedModel,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectiveApproximation {
    ExactGaussian,
    Pirls,
    Laplace { inner: String },
    AdaptiveGaussHermite { n_points: Option<usize> },
    NotAssessed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerCertificateScope {
    ExactObjective,
    ApproximatedObjective,
    NotAssessed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DerivativeAvailability {
    Available,
    NotAvailable { reason: String },
    NotAssessed { reason: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InferenceAvailability {
    Available { method: String },
    Unsupported { reason: String },
    NotAssessed { reason: String },
}

/// Versioned fixed-effect inference table stored on fitted artifacts.
///
/// The table is the durable row-level inference payload consumed by R and other
/// clients. `CompiledModelArtifact::fixed_effect_inference_table` is `None` for
/// prefit artifacts and becomes `Some(_)` after coefficient or contrast
/// inference has been computed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FixedEffectInferenceTable {
    pub schema_name: String,
    pub schema_version: String,
    pub crate_version: Option<String>,
    pub rows: Vec<FixedEffectInferenceRow>,
}

impl FixedEffectInferenceTable {
    pub fn new(rows: Vec<FixedEffectInferenceRow>) -> Self {
        Self {
            schema_name: FIXED_EFFECT_INFERENCE_TABLE_SCHEMA.to_string(),
            schema_version: FIXED_EFFECT_INFERENCE_TABLE_SCHEMA_VERSION.to_string(),
            crate_version: Some(env!("CARGO_PKG_VERSION").to_string()),
            rows,
        }
    }
}

/// Versioned fixed-effect covariance matrix stored on fitted artifacts.
///
/// This payload is reusable covariance geometry for consumers such as `vcov()`,
/// `emmeans`, and marginal-quantity tools. It is deliberately not the
/// inferential claim surface: p-values, degrees of freedom, method labels, and
/// reliability reasons live on [`FixedEffectInferenceTable`] rows.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FixedEffectCovarianceMatrix {
    pub schema_name: String,
    pub schema_version: String,
    pub crate_version: Option<String>,
    pub coef_names: Vec<String>,
    pub matrix: Option<Vec<Vec<f64>>>,
    pub method: FixedEffectCovarianceMethod,
    pub status: FixedEffectCovarianceStatus,
    pub reliability: ReliabilityGrade,
    pub reason: Option<String>,
    pub details: FixedEffectCovarianceDetails,
    pub notes: Vec<String>,
}

impl FixedEffectCovarianceMatrix {
    fn available_with_method(
        coef_names: Vec<String>,
        matrix: Vec<Vec<f64>>,
        method: FixedEffectCovarianceMethod,
        reliability: ReliabilityGrade,
        details: FixedEffectCovarianceDetails,
        notes: Vec<String>,
    ) -> Self {
        Self {
            schema_name: FIXED_EFFECT_COVARIANCE_MATRIX_SCHEMA.to_string(),
            schema_version: FIXED_EFFECT_COVARIANCE_MATRIX_SCHEMA_VERSION.to_string(),
            crate_version: Some(env!("CARGO_PKG_VERSION").to_string()),
            coef_names,
            matrix: Some(matrix),
            method,
            status: FixedEffectCovarianceStatus::Available,
            reliability,
            reason: None,
            details,
            notes,
        }
    }

    pub fn model_based(
        coef_names: Vec<String>,
        matrix: Vec<Vec<f64>>,
        details: FixedEffectCovarianceDetails,
        notes: Vec<String>,
    ) -> Self {
        Self::available_with_method(
            coef_names,
            matrix,
            FixedEffectCovarianceMethod::ModelBased,
            ReliabilityGrade::High,
            details,
            notes,
        )
    }

    pub fn pirls_laplace_working_hessian(
        coef_names: Vec<String>,
        matrix: Vec<Vec<f64>>,
        details: FixedEffectCovarianceDetails,
        notes: Vec<String>,
    ) -> Self {
        Self::available_with_method(
            coef_names,
            matrix,
            FixedEffectCovarianceMethod::PirlsLaplaceWorkingHessian,
            ReliabilityGrade::Moderate,
            details,
            notes,
        )
    }

    pub fn joint_laplace_active_hessian(
        coef_names: Vec<String>,
        matrix: Vec<Vec<f64>>,
        details: FixedEffectCovarianceDetails,
        notes: Vec<String>,
    ) -> Self {
        Self::available_with_method(
            coef_names,
            matrix,
            FixedEffectCovarianceMethod::JointLaplaceActiveHessian,
            ReliabilityGrade::Moderate,
            details,
            notes,
        )
    }

    pub fn unavailable(
        coef_names: Vec<String>,
        reason: impl Into<String>,
        details: FixedEffectCovarianceDetails,
        notes: Vec<String>,
    ) -> Self {
        Self {
            schema_name: FIXED_EFFECT_COVARIANCE_MATRIX_SCHEMA.to_string(),
            schema_version: FIXED_EFFECT_COVARIANCE_MATRIX_SCHEMA_VERSION.to_string(),
            crate_version: Some(env!("CARGO_PKG_VERSION").to_string()),
            coef_names,
            matrix: None,
            method: FixedEffectCovarianceMethod::Unavailable,
            status: FixedEffectCovarianceStatus::Unavailable,
            reliability: ReliabilityGrade::NotAvailable,
            reason: Some(reason.into()),
            details,
            notes,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FixedEffectCovarianceMethod {
    ModelBased,
    PirlsLaplaceWorkingHessian,
    JointLaplaceActiveHessian,
    Unavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FixedEffectCovarianceStatus {
    Available,
    Unavailable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FixedEffectCovarianceDetails {
    pub rank: Option<usize>,
    pub expected_rank: Option<usize>,
    pub aliased: Vec<String>,
    pub matrix_rows: usize,
    pub matrix_cols: usize,
    pub finite: Option<bool>,
    pub symmetric: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FixedEffectInferenceRow {
    pub label: String,
    pub kind: FixedEffectInferenceRowKind,
    pub estimate: Option<f64>,
    pub std_error: Option<f64>,
    pub numerator_df: Option<f64>,
    pub denominator_df: Option<f64>,
    pub statistic: Option<f64>,
    pub statistic_name: Option<FixedEffectStatisticName>,
    pub p_value: Option<f64>,
    pub method: FixedEffectInferenceMethod,
    pub status: FixedEffectInferenceStatus,
    pub reliability: ReliabilityGrade,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reliability_reason: Option<FixedEffectReliabilityReason>,
    pub estimability: EstimabilityAssessment,
    pub reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<FixedEffectInferenceDetails>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FixedEffectReliabilityReason {
    AsymptoticWaldZFallback,
    SatterthwaiteFiniteDifferenceApproximation,
    KenwardRogerApproximation,
    ParametricBootstrapMonteCarlo,
    GlmmJointLaplaceActiveHessianWald,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FixedEffectInferenceDetails {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bootstrap: Option<BootstrapInferenceDetails>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contrast_family: Option<ContrastFamilyDetails>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kenward_roger: Option<KenwardRogerInferenceDetails>,
}

impl FixedEffectInferenceDetails {
    pub fn is_empty(&self) -> bool {
        self.bootstrap.is_none() && self.contrast_family.is_none() && self.kenward_roger.is_none()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BootstrapInferenceDetails {
    pub target_kind: String,
    pub target_label: String,
    pub contrast_label: Option<String>,
    pub requested_replicates: usize,
    pub completed_replicates: usize,
    pub successful_replicates: usize,
    pub failed_refits: usize,
    pub failed_refit_policy: String,
    pub boundary_count: usize,
    pub boundary_rate: Option<f64>,
    pub seed_rng: String,
    pub seed: Option<u64>,
    pub finite_statistic_count: Option<usize>,
    pub mcse: Option<f64>,
    pub null_target: Option<FixedEffectNullTargetSummary>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FixedEffectNullTargetSummary {
    pub covariance_policy: String,
    pub coefficient_count: usize,
    pub theta_count: usize,
    pub sigma: Option<f64>,
    pub reml: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContrastFamilyDetails {
    pub family_id: String,
    pub family_label: String,
    pub restriction_rows: usize,
    pub coefficient_count: usize,
    pub requested_rank: Option<usize>,
    pub effective_rank: Option<usize>,
    pub rank_deficient: Option<bool>,
    pub rhs_nonzero: bool,
    pub numerator_df: Option<f64>,
    pub numerator_df_semantics: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KenwardRogerInferenceDetails {
    pub restriction_rank: Option<usize>,
    pub f_scaling: Option<f64>,
    pub statistic_scale: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FixedEffectInferenceRowKind {
    Coefficient,
    Contrast,
    Term,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FixedEffectStatisticName {
    Z,
    T,
    F,
    ChiSquare,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FixedEffectInferenceMethod {
    AsymptoticWaldZ,
    Satterthwaite,
    KenwardRoger,
    Bootstrap,
    NotComputed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FixedEffectInferenceStatus {
    Available,
    PValueUnavailable,
    NotEstimable,
    NotAssessed,
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactTable {
    FixedEffectInference,
    FixedEffectCovariance,
}

impl ArtifactTable {
    pub fn as_str(self) -> &'static str {
        match self {
            ArtifactTable::FixedEffectInference => FIXED_EFFECT_INFERENCE_TABLE_NAME,
            ArtifactTable::FixedEffectCovariance => FIXED_EFFECT_COVARIANCE_MATRIX_NAME,
        }
    }
}

impl std::str::FromStr for ArtifactTable {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            FIXED_EFFECT_INFERENCE_TABLE_NAME | "fixed_effect_inference_table" => {
                Ok(ArtifactTable::FixedEffectInference)
            }
            FIXED_EFFECT_COVARIANCE_MATRIX_NAME => Ok(ArtifactTable::FixedEffectCovariance),
            other => Err(format!(
                "unsupported artifact table `{other}`; supported tables: {FIXED_EFFECT_INFERENCE_TABLE_NAME}, {FIXED_EFFECT_COVARIANCE_MATRIX_NAME}"
            )),
        }
    }
}

/// Fit intent taxonomy from the compiler v0 contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FitMode {
    Confirmatory,
    Exploratory,
    Predictive,
}

/// Specific compiler policy within the fit mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FitIntent {
    #[default]
    ConfirmatoryAsSpecified,
    ConfirmatoryDesignCompiled,
    Exploratory,
    Predictive,
}

impl FitIntent {
    pub fn mode(self) -> FitMode {
        match self {
            FitIntent::ConfirmatoryAsSpecified | FitIntent::ConfirmatoryDesignCompiled => {
                FitMode::Confirmatory
            }
            FitIntent::Exploratory => FitMode::Exploratory,
            FitIntent::Predictive => FitMode::Predictive,
        }
    }

    pub fn allows_design_time_reduction(self) -> bool {
        matches!(self, FitIntent::ConfirmatoryDesignCompiled)
    }

    pub fn allows_confirmatory_p_values(self) -> bool {
        matches!(
            self,
            FitIntent::ConfirmatoryAsSpecified | FitIntent::ConfirmatoryDesignCompiled
        )
    }

    pub fn p_value_unavailable_reason(self) -> Option<String> {
        match self {
            FitIntent::ConfirmatoryAsSpecified | FitIntent::ConfirmatoryDesignCompiled => None,
            FitIntent::Exploratory => Some(
                "ordinary fixed-effect p-values are unavailable for exploratory fit intent"
                    .to_string(),
            ),
            FitIntent::Predictive => Some(
                "ordinary fixed-effect p-values are unavailable for predictive fit intent"
                    .to_string(),
            ),
        }
    }
}

fn fit_intent_for_policy(policy: &CompilerPolicy) -> FitIntent {
    match policy.random_strategy {
        RandomStrategy::AsSpecified => FitIntent::ConfirmatoryAsSpecified,
        RandomStrategy::MaximalFeasible if policy.apply_design_time_reductions => {
            FitIntent::ConfirmatoryDesignCompiled
        }
        RandomStrategy::MaximalFeasible => FitIntent::ConfirmatoryAsSpecified,
        RandomStrategy::Regularized => FitIntent::Exploratory,
        RandomStrategy::Predictive => FitIntent::Predictive,
    }
}

/// Why a requested model changed or was refused.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReductionTrigger {
    DesignTime,
    CertificateTimeBoundary,
    SelectionTime,
    NotAReduction,
}

/// One requested-to-effective model change.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReductionRecord {
    pub trigger: ReductionTrigger,
    pub phase: String,
    pub reason: String,
    pub affected_term: String,
    pub replacement_term: Option<String>,
    pub inference_consequence: String,
    pub diagnostics: Vec<Diagnostic>,
}

/// Requested/supported/fitted model-state view derived from an artifact.
///
/// This is intentionally computed from the current artifact instead of stored
/// as a mutable field. It gives R and other clients one stable view of what was
/// requested, what the design supports, what was fitted, and what changed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelStateSummary {
    pub schema_name: String,
    pub schema_version: u32,
    pub requested: ModelStageState,
    pub semantic: ModelStageState,
    pub supported: ModelStageState,
    pub fitted: ModelStageState,
    pub changes: Vec<ModelStateChange>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelStageState {
    pub stage: ModelStateStage,
    pub status: ModelStateStatus,
    pub formula: String,
    pub fixed_terms: Vec<String>,
    pub random_terms: Vec<ModelRandomTermState>,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelStateStage {
    Requested,
    Semantic,
    Supported,
    Fitted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelStateStatus {
    Requested,
    Canonical,
    Supported,
    AdvisoryChanges,
    Refused,
    Fitted,
    Reduced,
    NotAssessed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelRandomTermState {
    pub term_id: String,
    pub source_syntax: String,
    pub group: String,
    pub semantic_basis: Vec<String>,
    pub optimizer_basis: Vec<String>,
    pub covariance: String,
    pub covariance_support: CovarianceSupportStatus,
    pub basis_dimension: usize,
    pub covariance_parameters: Option<usize>,
    pub information_status: Option<String>,
    pub requested_rank: Option<usize>,
    pub supported_rank: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelStateChange {
    pub status: ModelChangeStatus,
    pub trigger: ReductionTrigger,
    pub from_stage: ModelStateStage,
    pub to_stage: ModelStateStage,
    pub affected_term: String,
    pub reason: String,
    pub replacement_term: Option<String>,
    pub inference_consequence: String,
    pub diagnostics: Vec<Diagnostic>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelChangeStatus {
    Applied,
    Recommended,
    Diagnostic,
}

/// Post-fit or analysis-time summary of the effective covariance rank for one
/// random-effect term. This records the user-scale meaning of supported and
/// unsupported random-effect directions without changing the requested model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EffectiveCovarianceSummary {
    pub term_id: String,
    pub source_syntax: String,
    pub requested_basis: Vec<String>,
    pub requested_rank: usize,
    pub supported_rank: usize,
    pub status: EffectiveRankStatus,
    pub directions: Vec<SupportedCovarianceDirection>,
    pub unsupported_directions: Vec<SupportedCovarianceDirection>,
    pub inference_consequence: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interpretable_submodel: Option<InterpretableSubmodel>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EffectiveRankStatus {
    FullRank,
    ReducedRank,
    NotAssessed,
}

/// One covariance eigen-direction expressed in the user-facing random-effect
/// basis. Loadings are kept structured so clients can render basis-stable or
/// user-scale explanations without parsing display text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SupportedCovarianceDirection {
    pub label: String,
    pub loadings: Vec<BasisLoading>,
    pub eigenvalue: Option<f64>,
    pub variance_explained: Option<f64>,
    pub user_scale_summary: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BasisLoading {
    pub basis: String,
    pub loading: f64,
}

/// One basis column's contribution to a dominant covariance direction.
///
/// `loading` is the *signed* component of the oriented eigenvector for the
/// supported direction along the basis axis named by `basis`. Sorted by
/// `|loading|` descending so callers can read the dominant component first.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DominantLoading {
    pub basis: String,
    pub loading: f64,
}

/// Suggestion that a simpler formula would fit nearly as well as the fitted
/// reduced-rank model.
///
/// Surfaced when the supported eigenvector of a reduced-rank covariance
/// direction loads almost entirely on one basis column. Never produced by
/// silent refitting: the user opts in by editing their formula to match
/// `suggested_formula` and re-running the fit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InterpretableSubmodel {
    /// Rewritten random-effect formula text (replacement for the term's
    /// `source_syntax`) that drops the off-direction column(s).
    pub suggested_formula: String,
    /// Loadings of the dominant supported direction expressed in the
    /// user-facing random-effect basis. Sorted by `|loading|` descending.
    pub loadings_dominant: Vec<DominantLoading>,
    /// Approximate gap in REML/ML objective (on the `-2 * log-likelihood`
    /// scale) between the fitted reduced-rank model and the suggested
    /// submodel. Lower-bounded above zero.
    pub objective_gap: f64,
    /// `true` when `objective_gap <= INTERPRETABLE_GAP_TOLERANCE`.
    pub within_tolerance: bool,
}

/// End-to-end trace for one optimizer covariance parameter.
///
/// A trace records the path from source formula syntax through semantic and
/// optimizer basis columns into a concrete `theta`/`Lambda` slot and the
/// user-facing VarCorr entries associated with that slot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CovarianceParameterTrace {
    pub term_id: String,
    pub source_syntax: String,
    pub semantic_term_index: usize,
    pub optimizer_term_index: usize,
    pub group: String,
    pub covariance_family: CovarianceFamily,
    pub covariance_support: CovarianceSupportStatus,
    pub user_basis: Vec<String>,
    pub optimizer_basis: Vec<String>,
    pub theta: ThetaSlotTrace,
    pub lambda: LambdaSlotTrace,
    pub parmap_entry: Option<ParmapTrace>,
    pub varcorr_entries: Vec<VarCorrEntryTrace>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThetaSlotTrace {
    pub global_index: Option<usize>,
    pub local_index: usize,
    pub name: String,
    pub constraint: ParameterConstraint,
    pub status: ParameterStatus,
    pub value: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LambdaSlotTrace {
    pub row: usize,
    pub col: usize,
    pub row_basis: String,
    pub col_basis: String,
    pub value: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParmapTrace {
    pub theta_index: usize,
    pub term_index: usize,
    pub lambda_row: usize,
    pub lambda_col: usize,
    pub matches_theta_map: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VarCorrEntryTrace {
    pub kind: VarCorrEntryKind,
    pub label: String,
    pub basis: Vec<String>,
    pub value: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VarCorrEntryKind {
    StandardDeviation,
    Correlation,
}

/// Round-trippable compiler artifact. Fitting extends this artifact rather than
/// reconstructing semantic meaning from formula strings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompiledModelArtifact {
    pub schema: SchemaMetadata,
    pub model_boundary: ModelBoundary,
    #[serde(default)]
    pub fixed_effect_inference_table: Option<FixedEffectInferenceTable>,
    #[serde(default)]
    pub fixed_effect_covariance_matrix: Option<FixedEffectCovarianceMatrix>,
    pub requested_formula: String,
    pub semantic_model: SemanticModel,
    pub effective_formula: Option<String>,
    pub effective_semantic_model: Option<SemanticModel>,
    pub theta_maps: Vec<ThetaMap>,
    pub design_audit: Option<DesignAudit>,
    pub compiler_policy: CompilerPolicy,
    pub policy_recommendations: Vec<PolicyRecommendation>,
    pub effective_covariance: Vec<EffectiveCovarianceSummary>,
    pub reductions: Vec<ReductionRecord>,
    pub covariance_transitions: Vec<CovarianceFamilyTransition>,
    pub covariance_parameter_traces: Vec<CovarianceParameterTrace>,
    pub reproducibility: ReproducibilityRecord,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub glmm_fit_metadata: Option<GlmmFitMetadata>,
    pub optimizer_certificate: Option<OptimizerCertificate>,
    pub diagnostics: Vec<Diagnostic>,
}

impl CompiledModelArtifact {
    pub fn new(requested_formula: impl Into<String>, semantic_model: SemanticModel) -> Self {
        Self::new_with_policy(requested_formula, semantic_model, CompilerPolicy::default())
    }

    pub fn new_with_policy(
        requested_formula: impl Into<String>,
        semantic_model: SemanticModel,
        compiler_policy: CompilerPolicy,
    ) -> Self {
        let mut global_start = 0;
        let mut theta_maps = Vec::with_capacity(semantic_model.random_terms.len());
        for (term_index, term) in semantic_model.random_terms.iter().enumerate() {
            let map = ThetaMap::from_random_term(term_index, term, global_start);
            global_start += map.n_free();
            theta_maps.push(map);
        }

        let reproducibility = ReproducibilityRecord::with_policy(&compiler_policy);

        let mut artifact = Self {
            schema: SchemaMetadata::compiled_model_artifact(),
            model_boundary: ModelBoundary::lmm(),
            fixed_effect_inference_table: None,
            fixed_effect_covariance_matrix: None,
            requested_formula: requested_formula.into(),
            diagnostics: semantic_model.diagnostics.clone(),
            semantic_model,
            effective_formula: None,
            effective_semantic_model: None,
            theta_maps,
            design_audit: None,
            compiler_policy,
            policy_recommendations: Vec::new(),
            effective_covariance: Vec::new(),
            reductions: Vec::new(),
            covariance_transitions: Vec::new(),
            covariance_parameter_traces: Vec::new(),
            reproducibility,
            glmm_fit_metadata: None,
            optimizer_certificate: None,
        };
        artifact.refresh_covariance_parameter_traces(None, None, &[]);
        artifact
    }

    pub fn attach_design_audit(&mut self, data: &DataFrame) {
        let audit = audit_design(&self.semantic_model, data);
        self.diagnostics.extend(audit.diagnostics.clone());
        self.policy_recommendations =
            recommend_policy(&self.semantic_model, &audit, &self.compiler_policy);
        self.design_audit = Some(audit);
    }

    pub fn set_compiler_policy(&mut self, policy: CompilerPolicy) {
        self.reproducibility = ReproducibilityRecord::with_policy(&policy);
        self.compiler_policy = policy;
        if let Some(audit) = &self.design_audit {
            self.policy_recommendations =
                recommend_policy(&self.semantic_model, audit, &self.compiler_policy);
        }
    }

    pub fn set_model_boundary(&mut self, boundary: ModelBoundary) {
        self.model_boundary = boundary;
    }

    pub fn record_effective_covariance_summary(&mut self, summary: EffectiveCovarianceSummary) {
        self.effective_covariance.push(summary);
    }

    pub fn set_effective_model(
        &mut self,
        effective_formula: impl Into<String>,
        effective_semantic_model: SemanticModel,
        reductions: Vec<ReductionRecord>,
    ) {
        self.effective_formula = Some(effective_formula.into());
        self.diagnostics
            .extend(effective_semantic_model.diagnostics.clone());
        self.effective_semantic_model = Some(effective_semantic_model);
        self.reductions.extend(reductions);
    }

    fn active_semantic_model(&self) -> &SemanticModel {
        self.effective_semantic_model
            .as_ref()
            .unwrap_or(&self.semantic_model)
    }

    /// Build a stable user-facing audit report from the current artifact state.
    /// Compact default-print summary of the artifact (PRD § 15).
    ///
    /// Suitable for `Display`; consumers wanting structured access can
    /// keep the returned [`super::print::ModelPrint`] around and
    /// inspect its public fields. Heavier reports
    /// (`audit_report`, `explain_model`, `parameterization`,
    /// `changes`) stay one explicit method call away.
    pub fn print_summary(&self) -> super::print::ModelPrint {
        super::print::ModelPrint::from_artifact(self)
    }

    /// Source-to-fitted parameterization drilldown (PRD § 15).
    ///
    /// Wraps the artifact's `covariance_parameter_traces` so callers
    /// can render the per-(term, theta-slot) trace through source
    /// syntax, `theta`, `Lambda`, `parmap`, and VarCorr entries
    /// without flattening it manually. See [`super::print::ParameterizationDrilldown`].
    pub fn parameterization(&self) -> super::print::ParameterizationDrilldown {
        super::print::ParameterizationDrilldown::from_artifact(self)
    }

    pub fn audit_report(&self) -> ModelAuditReport {
        ModelAuditReport::from_artifact(self)
    }

    pub fn table(&self, table: ArtifactTable) -> Option<serde_json::Value> {
        match table {
            ArtifactTable::FixedEffectInference => self
                .fixed_effect_inference_table
                .as_ref()
                .map(|table| serde_json::to_value(table).expect("inference table serializes")),
            ArtifactTable::FixedEffectCovariance => self
                .fixed_effect_covariance_matrix
                .as_ref()
                .map(|matrix| serde_json::to_value(matrix).expect("covariance matrix serializes")),
        }
    }

    pub fn table_by_name(
        &self,
        table: &str,
    ) -> std::result::Result<Option<serde_json::Value>, String> {
        Ok(self.table(table.parse()?))
    }

    /// Build a stable requested -> semantic -> supported -> fitted state view.
    pub fn model_state_summary(&self) -> ModelStateSummary {
        ModelStateSummary::from_artifact(self)
    }

    /// Return all recorded or recommended model-state changes.
    pub fn changes(&self) -> Vec<ModelStateChange> {
        self.model_state_summary().changes
    }

    /// Rebuild theta maps in optimizer term order while preserving semantic
    /// term ids. This is needed because the numerical engine may reorder random
    /// terms for sparse factorization efficiency.
    pub fn rebuild_theta_maps_for_optimizer_order(&mut self, semantic_order: &[usize]) {
        let optimizer_basis = semantic_order
            .iter()
            .map(|&semantic_index| {
                self.semantic_model
                    .random_terms
                    .get(semantic_index)
                    .map(|term| {
                        term.basis
                            .iter()
                            .map(|basis| basis.name.clone())
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default()
            })
            .collect::<Vec<_>>();
        self.rebuild_theta_maps_for_optimizer_order_with_basis(semantic_order, &optimizer_basis);
    }

    /// Rebuild theta maps with optimizer-basis names from materialized ReMat
    /// columns. This records data-dependent basis expansion, such as
    /// categorical random slopes, while preserving the semantic user basis.
    pub fn rebuild_theta_maps_for_optimizer_order_with_basis(
        &mut self,
        semantic_order: &[usize],
        optimizer_basis: &[Vec<String>],
    ) {
        let source_groups = optimizer_source_groups(self.active_semantic_model());
        if semantic_order.len() != source_groups.len() {
            self.diagnostics.push(Diagnostic::new(
                super::diagnostics::DiagnosticCode::Unsupported,
                super::diagnostics::DiagnosticSeverity::Error,
                super::diagnostics::DiagnosticStage::Parameterization,
                "theta-map optimizer order does not cover all optimizer source random terms",
            ));
            return;
        }
        if optimizer_basis.len() != semantic_order.len() {
            self.diagnostics.push(Diagnostic::new(
                super::diagnostics::DiagnosticCode::Unsupported,
                super::diagnostics::DiagnosticSeverity::Error,
                super::diagnostics::DiagnosticStage::Parameterization,
                "theta-map optimizer basis does not cover all semantic random terms",
            ));
            return;
        }

        let mut global_start = 0;
        let mut theta_maps = Vec::new();
        for (optimizer_index, &source_group_index) in semantic_order.iter().enumerate() {
            let Some(source_group) = source_groups.get(source_group_index) else {
                self.diagnostics.push(Diagnostic::new(
                    super::diagnostics::DiagnosticCode::Unsupported,
                    super::diagnostics::DiagnosticSeverity::Error,
                    super::diagnostics::DiagnosticStage::Parameterization,
                    format!(
                        "optimizer source random-term index {source_group_index} is out of range"
                    ),
                ));
                return;
            };
            let mut split_lambda_cursor = 0;
            for &semantic_index in source_group {
                let Some(term) = self
                    .active_semantic_model()
                    .random_terms
                    .get(semantic_index)
                else {
                    self.diagnostics.push(Diagnostic::new(
                        super::diagnostics::DiagnosticCode::Unsupported,
                        super::diagnostics::DiagnosticSeverity::Error,
                        super::diagnostics::DiagnosticStage::Parameterization,
                        format!("semantic random-term index {semantic_index} is out of range"),
                    ));
                    return;
                };
                let map = if source_group.len() > 1 {
                    let lambda_indices = split_term_optimizer_basis_range(
                        term,
                        &optimizer_basis[optimizer_index],
                        split_lambda_cursor,
                    );
                    let Some(lambda_indices) = lambda_indices else {
                        self.diagnostics.push(Diagnostic::new(
                            super::diagnostics::DiagnosticCode::Unsupported,
                            super::diagnostics::DiagnosticSeverity::Error,
                            super::diagnostics::DiagnosticStage::Parameterization,
                            format!(
                                "split random-term '{}' is missing from optimizer basis",
                                term.id
                            ),
                        ));
                        return;
                    };
                    split_lambda_cursor = lambda_indices.end;
                    ThetaMap::from_split_random_term_with_optimizer_basis(
                        optimizer_index,
                        term,
                        global_start,
                        optimizer_basis[optimizer_index].clone(),
                        lambda_indices,
                    )
                } else {
                    ThetaMap::from_random_term_with_optimizer_basis(
                        optimizer_index,
                        term,
                        global_start,
                        optimizer_basis[optimizer_index].clone(),
                    )
                };
                global_start += map.n_free();
                theta_maps.push(map);
            }
            if source_group.len() > 1
                && split_lambda_cursor != optimizer_basis[optimizer_index].len()
            {
                self.diagnostics.push(Diagnostic::new(
                    super::diagnostics::DiagnosticCode::Unsupported,
                    super::diagnostics::DiagnosticSeverity::Error,
                    super::diagnostics::DiagnosticStage::Parameterization,
                    format!(
                        "split random-terms cover {split_lambda_cursor} of {} optimizer-basis columns",
                        optimizer_basis[optimizer_index].len()
                    ),
                ));
                return;
            }
        }

        self.theta_maps = theta_maps;
        self.refresh_covariance_parameter_traces(None, None, &[]);
    }

    /// Rebuild end-to-end covariance parameter traces from the current
    /// theta-map state. When fitted `Lambda` values and an optimizer parmap are
    /// supplied, traces include concrete theta/Lambda/VarCorr values and
    /// parmap alignment checks; otherwise they remain a compiler skeleton.
    pub fn refresh_covariance_parameter_traces(
        &mut self,
        lambdas: Option<&[Vec<Vec<f64>>]>,
        sd_scale: Option<f64>,
        parmap: &[(usize, usize, usize)],
    ) {
        self.covariance_parameter_traces =
            covariance_parameter_traces(self, lambdas, sd_scale, parmap);
    }
}

fn covariance_parameter_traces(
    artifact: &CompiledModelArtifact,
    lambdas: Option<&[Vec<Vec<f64>>]>,
    sd_scale: Option<f64>,
    parmap: &[(usize, usize, usize)],
) -> Vec<CovarianceParameterTrace> {
    let active_model = artifact.active_semantic_model();
    let mut traces = Vec::new();

    for theta_map in &artifact.theta_maps {
        let block = theta_map.block();
        let optimizer_term_index = block.term_index;
        let semantic_term_index = active_model
            .random_terms
            .iter()
            .position(|term| term.id == block.term_id)
            .unwrap_or(block.term_index);
        let source_syntax = active_model
            .random_terms
            .get(semantic_term_index)
            .map(|term| term.source_syntax.text.clone())
            .unwrap_or_else(|| block.term_id.clone());
        let lambda = lambdas.and_then(|values| values.get(optimizer_term_index));

        for slot in &block.theta_slots {
            traces.push(parameter_trace_for_slot(
                slot,
                theta_map.family(),
                block,
                source_syntax.clone(),
                semantic_term_index,
                optimizer_term_index,
                lambda,
                sd_scale,
                parmap,
            ));
        }
    }

    traces
}

fn optimizer_source_groups(semantic_model: &SemanticModel) -> Vec<Vec<usize>> {
    let mut groups = Vec::new();
    let mut index = 0;
    while index < semantic_model.random_terms.len() {
        let term = &semantic_model.random_terms[index];
        if let Some(block_group) = &term.block_group {
            let mut group = Vec::new();
            while index < semantic_model.random_terms.len()
                && semantic_model.random_terms[index].block_group.as_ref() == Some(block_group)
            {
                group.push(index);
                index += 1;
            }
            groups.push(group);
        } else {
            groups.push(vec![index]);
            index += 1;
        }
    }
    groups
}

/// The contiguous run of materialized optimizer-basis columns belonging to one
/// split (`||`) random term, starting at `cursor`. Materialization preserves
/// semantic coefficient order, so split terms claim consecutive runs: a
/// numeric coefficient claims its verbatim column, a factor coefficient claims
/// one level-coded column per contrast.
fn split_term_optimizer_basis_range(
    term: &super::ir::RandomTermIr,
    optimizer_basis: &[String],
    cursor: usize,
) -> Option<std::ops::Range<usize>> {
    let basis = term.basis.first()?;
    let mut end = cursor;
    while end < optimizer_basis.len()
        && optimizer_basis_column_materializes(&optimizer_basis[end], &basis.name)
    {
        end += 1;
    }
    (end > cursor).then_some(cursor..end)
}

/// Whether a materialized optimizer-basis column name is an expansion of a
/// semantic basis coefficient. Numeric coefficients materialize verbatim
/// ("x"), factor coefficients expand to level-coded columns ("f: b"), and
/// interaction coefficients ("x:f") expand each factor component in place
/// ("x:f: b", "f: a:x").
fn optimizer_basis_column_materializes(column: &str, coefficient: &str) -> bool {
    let components: Vec<&str> = coefficient.split(':').collect();
    column_matches_coefficient_components(column, &components)
}

fn column_matches_coefficient_components(column: &str, components: &[&str]) -> bool {
    let Some((component, rest_components)) = components.split_first() else {
        return column.is_empty();
    };
    let Some(rest) = column.strip_prefix(component) else {
        return false;
    };
    if rest_components.is_empty() {
        // Final component: a verbatim numeric column or a level-coded factor
        // expansion ("f: b").
        return rest.is_empty() || rest.starts_with(": ");
    }
    // Numeric component followed by the ":" interaction separator.
    if let Some(next) = rest.strip_prefix(':') {
        if column_matches_coefficient_components(next, rest_components) {
            return true;
        }
    }
    // Factor component expanded to ": <level>" before the interaction
    // separator; the level itself may contain ':' so try every cut point.
    if let Some(level_rest) = rest.strip_prefix(": ") {
        let mut search = level_rest;
        while let Some(position) = search.find(':') {
            if column_matches_coefficient_components(&search[position + 1..], rest_components) {
                return true;
            }
            search = &search[position + 1..];
        }
    }
    false
}

#[allow(clippy::too_many_arguments)]
fn parameter_trace_for_slot(
    slot: &ThetaSlot,
    covariance_family: CovarianceFamily,
    block: &super::theta_map::ThetaMapBlock,
    source_syntax: String,
    semantic_term_index: usize,
    optimizer_term_index: usize,
    lambda: Option<&Vec<Vec<f64>>>,
    sd_scale: Option<f64>,
    parmap: &[(usize, usize, usize)],
) -> CovarianceParameterTrace {
    let row_basis = basis_label(&block.optimizer_basis, slot.lambda_row);
    let col_basis = basis_label(&block.optimizer_basis, slot.lambda_col);
    let lambda_value = lambda_value(lambda, slot.lambda_row, slot.lambda_col);
    let parmap_entry = parmap_entry(slot, parmap);
    let varcorr_entries = varcorr_entries_for_slot(
        slot,
        &block.optimizer_basis,
        covariance_family.clone(),
        lambda,
        sd_scale,
    );

    CovarianceParameterTrace {
        term_id: block.term_id.clone(),
        source_syntax,
        semantic_term_index,
        optimizer_term_index,
        group: block.group.clone(),
        covariance_support: covariance_family.support_status(),
        covariance_family,
        user_basis: block.user_basis.clone(),
        optimizer_basis: block.optimizer_basis.clone(),
        theta: ThetaSlotTrace {
            global_index: slot.global_index,
            local_index: slot.local_index,
            name: slot.name.clone(),
            constraint: slot.constraint.clone(),
            status: slot.status,
            value: lambda_value,
        },
        lambda: LambdaSlotTrace {
            row: slot.lambda_row,
            col: slot.lambda_col,
            row_basis,
            col_basis,
            value: lambda_value,
        },
        parmap_entry,
        varcorr_entries,
    }
}

fn basis_label(basis: &[String], index: usize) -> String {
    basis
        .get(index)
        .cloned()
        .unwrap_or_else(|| format!("basis[{index}]"))
}

fn lambda_value(lambda: Option<&Vec<Vec<f64>>>, row: usize, col: usize) -> Option<f64> {
    lambda
        .and_then(|matrix| matrix.get(row))
        .and_then(|values| values.get(col))
        .copied()
}

fn parmap_entry(slot: &ThetaSlot, parmap: &[(usize, usize, usize)]) -> Option<ParmapTrace> {
    let theta_index = slot.global_index?;
    let &(term_index, lambda_row, lambda_col) = parmap.get(theta_index)?;
    Some(ParmapTrace {
        theta_index,
        term_index,
        lambda_row,
        lambda_col,
        matches_theta_map: term_index == slot.term_index
            && lambda_row == slot.lambda_row
            && lambda_col == slot.lambda_col,
    })
}

fn varcorr_entries_for_slot(
    slot: &ThetaSlot,
    basis: &[String],
    covariance_family: CovarianceFamily,
    lambda: Option<&Vec<Vec<f64>>>,
    sd_scale: Option<f64>,
) -> Vec<VarCorrEntryTrace> {
    if let CovarianceFamily::Structured { kind } = covariance_family {
        return structured_varcorr_entries_for_slot(slot, basis, &kind);
    }

    let mut entries = Vec::new();
    let row_basis = basis_label(basis, slot.lambda_row);
    entries.push(VarCorrEntryTrace {
        kind: VarCorrEntryKind::StandardDeviation,
        label: format!("sd({row_basis})"),
        basis: vec![row_basis.clone()],
        value: sd_scale.and_then(|scale| row_std_dev(lambda, slot.lambda_row, scale)),
    });

    if matches!(covariance_family, CovarianceFamily::FullCholesky) {
        for previous in 0..slot.lambda_row {
            if slot.lambda_col <= previous || slot.lambda_col == slot.lambda_row {
                let previous_basis = basis_label(basis, previous);
                entries.push(VarCorrEntryTrace {
                    kind: VarCorrEntryKind::Correlation,
                    label: format!("corr({row_basis},{previous_basis})"),
                    basis: vec![row_basis.clone(), previous_basis],
                    value: sd_scale.and_then(|_| correlation(lambda, slot.lambda_row, previous)),
                });
            }
        }
    }

    entries
}

fn structured_varcorr_entries_for_slot(
    slot: &ThetaSlot,
    basis: &[String],
    kind: &str,
) -> Vec<VarCorrEntryTrace> {
    match slot.local_index {
        0 => vec![VarCorrEntryTrace {
            kind: VarCorrEntryKind::StandardDeviation,
            label: format!("sd({kind}.scale)"),
            basis: basis.to_vec(),
            value: None,
        }],
        _ => vec![VarCorrEntryTrace {
            kind: VarCorrEntryKind::Correlation,
            label: format!("corr({kind}.structure)"),
            basis: basis.to_vec(),
            value: None,
        }],
    }
}

fn row_std_dev(lambda: Option<&Vec<Vec<f64>>>, row: usize, sd_scale: f64) -> Option<f64> {
    let matrix = lambda?;
    let row_values = matrix.get(row)?;
    let norm_sq = (0..=row)
        .map(|col| row_values.get(col).copied().unwrap_or(0.0).powi(2))
        .sum::<f64>();
    Some(sd_scale * norm_sq.sqrt())
}

fn correlation(lambda: Option<&Vec<Vec<f64>>>, row: usize, col: usize) -> Option<f64> {
    let matrix = lambda?;
    let row_values = matrix.get(row)?;
    let col_values = matrix.get(col)?;
    let row_norm = (0..=row)
        .map(|idx| row_values.get(idx).copied().unwrap_or(0.0).powi(2))
        .sum::<f64>()
        .sqrt();
    let col_norm = (0..=col)
        .map(|idx| col_values.get(idx).copied().unwrap_or(0.0).powi(2))
        .sum::<f64>()
        .sqrt();
    if row_norm <= 0.0 || col_norm <= 0.0 {
        return None;
    }
    let dot = (0..=col)
        .map(|idx| {
            row_values.get(idx).copied().unwrap_or(0.0)
                * col_values.get(idx).copied().unwrap_or(0.0)
        })
        .sum::<f64>();
    Some(dot / (row_norm * col_norm))
}

impl ModelStateSummary {
    pub fn from_artifact(artifact: &CompiledModelArtifact) -> Self {
        let changes = model_state_changes(artifact);
        Self {
            schema_name: MODEL_STATE_SUMMARY_SCHEMA.to_string(),
            schema_version: MODEL_STATE_SUMMARY_SCHEMA_VERSION,
            requested: ModelStageState {
                stage: ModelStateStage::Requested,
                status: ModelStateStatus::Requested,
                formula: artifact.requested_formula.clone(),
                fixed_terms: artifact.semantic_model.fixed_terms.clone(),
                random_terms: random_term_states(artifact, ModelStateStage::Requested),
                reason: Some("formula as requested by the caller".to_string()),
            },
            semantic: ModelStageState {
                stage: ModelStateStage::Semantic,
                status: ModelStateStatus::Canonical,
                formula: semantic_formula(&artifact.semantic_model),
                fixed_terms: artifact.semantic_model.fixed_terms.clone(),
                random_terms: random_term_states(artifact, ModelStateStage::Semantic),
                reason: Some("formula compiled into semantic IR".to_string()),
            },
            supported: supported_stage_state(artifact),
            fitted: fitted_stage_state(artifact),
            changes,
        }
    }
}

fn supported_stage_state(artifact: &CompiledModelArtifact) -> ModelStageState {
    let has_design_reductions = artifact
        .reductions
        .iter()
        .any(|reduction| reduction.trigger == ReductionTrigger::DesignTime);
    let (status, reason) = if artifact.design_audit.is_none() {
        (
            ModelStateStatus::NotAssessed,
            Some("design audit has not been attached".to_string()),
        )
    } else if has_design_reductions {
        (
            ModelStateStatus::Reduced,
            Some("design_compiled applied deterministic design-time model changes".to_string()),
        )
    } else if artifact.policy_recommendations.is_empty() {
        (
            ModelStateStatus::Supported,
            Some("design audit did not recommend design-time model changes".to_string()),
        )
    } else if artifact
        .policy_recommendations
        .iter()
        .any(|recommendation| {
            matches!(
                recommendation.action,
                PolicyAction::RefuseRandomTermDistribution | PolicyAction::MarkNotAssessable
            )
        })
    {
        (
            ModelStateStatus::Refused,
            Some(
                "design audit found at least one unsupported random-effect distribution"
                    .to_string(),
            ),
        )
    } else {
        (
            ModelStateStatus::AdvisoryChanges,
            Some("design audit recommended explicit design-time model changes".to_string()),
        )
    };

    ModelStageState {
        stage: ModelStateStage::Supported,
        status,
        formula: artifact
            .effective_formula
            .clone()
            .unwrap_or_else(|| semantic_formula(&artifact.semantic_model)),
        fixed_terms: artifact.semantic_model.fixed_terms.clone(),
        random_terms: random_term_states(artifact, ModelStateStage::Supported),
        reason,
    }
}

fn fitted_stage_state(artifact: &CompiledModelArtifact) -> ModelStageState {
    let (status, reason) = match &artifact.optimizer_certificate {
        None => (
            ModelStateStatus::NotAssessed,
            Some("model has not been fitted".to_string()),
        ),
        Some(certificate) => {
            let has_fit_reductions = artifact
                .reductions
                .iter()
                .any(|reduction| reduction.trigger != ReductionTrigger::DesignTime);
            if artifact
                .effective_covariance
                .iter()
                .any(|summary| summary.status == EffectiveRankStatus::ReducedRank)
                || has_fit_reductions
            {
                (
                    ModelStateStatus::Reduced,
                    Some(format!(
                        "fit completed with {:?} and recorded fitted-state reductions",
                        certificate.status
                    )),
                )
            } else {
                (
                    ModelStateStatus::Fitted,
                    Some(format!("fit completed with {:?}", certificate.status)),
                )
            }
        }
    };

    ModelStageState {
        stage: ModelStateStage::Fitted,
        status,
        formula: artifact
            .effective_formula
            .clone()
            .unwrap_or_else(|| semantic_formula(&artifact.semantic_model)),
        fixed_terms: artifact.semantic_model.fixed_terms.clone(),
        random_terms: random_term_states(artifact, ModelStateStage::Fitted),
        reason,
    }
}

fn random_term_states(
    artifact: &CompiledModelArtifact,
    stage: ModelStateStage,
) -> Vec<ModelRandomTermState> {
    let model = if matches!(stage, ModelStateStage::Supported | ModelStateStage::Fitted) {
        artifact.active_semantic_model()
    } else {
        &artifact.semantic_model
    };

    model
        .random_terms
        .iter()
        .map(|term| {
            let semantic_basis = term
                .basis
                .iter()
                .map(|basis| basis.name.clone())
                .collect::<Vec<_>>();
            let theta_map = artifact
                .theta_maps
                .iter()
                .find(|map| map.block().term_id == term.id);
            let audit = artifact.design_audit.as_ref().and_then(|audit| {
                audit
                    .random_terms
                    .iter()
                    .find(|random| random.term_id == term.id)
            });
            let effective = artifact
                .effective_covariance
                .iter()
                .find(|summary| summary.term_id == term.id);
            let optimizer_basis =
                if matches!(stage, ModelStateStage::Supported | ModelStateStage::Fitted) {
                    theta_map
                        .map(|map| map.block().optimizer_basis.clone())
                        .unwrap_or_else(|| semantic_basis.clone())
                } else {
                    semantic_basis.clone()
                };
            let basis_dimension = match stage {
                ModelStateStage::Requested | ModelStateStage::Semantic => semantic_basis.len(),
                ModelStateStage::Supported | ModelStateStage::Fitted => audit
                    .map(|audit| audit.information_budget.basis_dimension)
                    .unwrap_or_else(|| optimizer_basis.len()),
            };

            ModelRandomTermState {
                term_id: term.id.clone(),
                source_syntax: term.source_syntax.text.clone(),
                group: term.group.label(),
                semantic_basis,
                optimizer_basis,
                covariance: covariance_label(&term.covariance),
                covariance_support: term.covariance_support,
                basis_dimension,
                covariance_parameters: audit
                    .map(|audit| audit.requested_covariance_parameters)
                    .or_else(|| theta_map.map(|map| map.n_free())),
                information_status: audit
                    .map(|audit| information_status_label(audit.information_budget.status)),
                requested_rank: effective.map(|summary| summary.requested_rank),
                supported_rank: effective.map(|summary| summary.supported_rank),
            }
        })
        .collect()
}

fn model_state_changes(artifact: &CompiledModelArtifact) -> Vec<ModelStateChange> {
    let mut changes = Vec::new();

    changes.extend(
        artifact
            .semantic_model
            .diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.code == DiagnosticCode::FormulaCanonicalized)
            .map(|diagnostic| ModelStateChange {
                status: ModelChangeStatus::Diagnostic,
                trigger: ReductionTrigger::NotAReduction,
                from_stage: ModelStateStage::Requested,
                to_stage: ModelStateStage::Semantic,
                affected_term: diagnostic
                    .affected_terms
                    .first()
                    .cloned()
                    .unwrap_or_else(|| artifact.requested_formula.clone()),
                reason: diagnostic.message.clone(),
                replacement_term: diagnostic
                    .payload
                    .get("canonical_terms")
                    .map(ToString::to_string),
                inference_consequence:
                    "formula canonicalization changes representation, not the declared estimand"
                        .to_string(),
                diagnostics: vec![diagnostic.clone()],
            }),
    );

    let has_applied_design_reductions = artifact
        .reductions
        .iter()
        .any(|reduction| reduction.trigger == ReductionTrigger::DesignTime);
    if !has_applied_design_reductions {
        changes.extend(
            artifact
                .policy_recommendations
                .iter()
                .map(|recommendation| ModelStateChange {
                    status: ModelChangeStatus::Recommended,
                    trigger: ReductionTrigger::DesignTime,
                    from_stage: ModelStateStage::Semantic,
                    to_stage: ModelStateStage::Supported,
                    affected_term: recommendation.term_id.clone(),
                    reason: recommendation.reason.clone(),
                    replacement_term: recommended_replacement(recommendation),
                    inference_consequence: recommendation.inference_consequence.clone(),
                    diagnostics: recommendation.diagnostics.clone(),
                }),
        );
    }

    changes.extend(
        artifact
            .reductions
            .iter()
            .map(|reduction| ModelStateChange {
                status: ModelChangeStatus::Applied,
                trigger: reduction.trigger.clone(),
                from_stage: if reduction.trigger == ReductionTrigger::DesignTime {
                    ModelStateStage::Semantic
                } else {
                    ModelStateStage::Supported
                },
                to_stage: if reduction.trigger == ReductionTrigger::DesignTime {
                    ModelStateStage::Supported
                } else {
                    ModelStateStage::Fitted
                },
                affected_term: reduction.affected_term.clone(),
                reason: reduction.reason.clone(),
                replacement_term: reduction.replacement_term.clone(),
                inference_consequence: reduction.inference_consequence.clone(),
                diagnostics: reduction.diagnostics.clone(),
            }),
    );

    changes
}

fn recommended_replacement(recommendation: &PolicyRecommendation) -> Option<String> {
    match recommendation.action {
        PolicyAction::DropUnsupportedBasis => {
            Some("drop unsupported random-effect basis direction(s)".to_string())
        }
        PolicyAction::ReduceCovariance => recommendation
            .recommended_covariance
            .as_ref()
            .map(|covariance| format!("covariance={covariance}")),
        PolicyAction::RefuseRandomTermDistribution => {
            Some("refuse random-effect distribution".to_string())
        }
        PolicyAction::MarkNotAssessable => Some("mark term as not assessable".to_string()),
    }
}

fn semantic_formula(semantic_model: &SemanticModel) -> String {
    let mut rhs = semantic_model.fixed_terms.clone();
    rhs.extend(
        semantic_model
            .random_terms
            .iter()
            .map(|term| term.source_syntax.text.clone()),
    );
    let rhs = if rhs.is_empty() {
        "1".to_string()
    } else {
        rhs.join(" + ")
    };
    format!("{} ~ {}", semantic_model.response, rhs)
}

fn covariance_label(covariance: &super::ir::CovarianceForm) -> String {
    match covariance {
        super::ir::CovarianceForm::Scalar => "scalar".to_string(),
        super::ir::CovarianceForm::Diagonal => "diagonal".to_string(),
        super::ir::CovarianceForm::Full => "full".to_string(),
        super::ir::CovarianceForm::Structured { kind } => format!("structured:{kind}"),
        super::ir::CovarianceForm::ReducedRank { rank } => match rank {
            Some(rank) => format!("reduced_rank:{rank}"),
            None => "reduced_rank".to_string(),
        },
        super::ir::CovarianceForm::Unsupported { reason } => {
            format!("unsupported:{reason}")
        }
    }
}

fn information_status_label(status: super::audit::InformationBudgetStatus) -> String {
    match status {
        super::audit::InformationBudgetStatus::Sufficient => "sufficient".to_string(),
        super::audit::InformationBudgetStatus::WeaklySupported => "weakly_supported".to_string(),
        super::audit::InformationBudgetStatus::TooRich => "too_rich".to_string(),
        super::audit::InformationBudgetStatus::NotAssessable => "not_assessable".to_string(),
    }
}

fn default_thresholds() -> Vec<(String, String)> {
    CompilerThresholds::default().reproducibility_entries()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::compile_formula_ir;
    use crate::formula::parse_formula;
    use crate::model::data::DataFrame;

    #[test]
    fn compiled_artifact_builds_theta_maps() {
        let formula = parse_formula("y ~ x + (1 + x | subject)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let artifact = CompiledModelArtifact::new(formula.to_string(), semantic);

        assert_eq!(artifact.theta_maps.len(), 1);
        assert_eq!(artifact.theta_maps[0].n_free(), 3);
        assert_eq!(
            artifact.reproducibility.fit_intent,
            FitIntent::ConfirmatoryAsSpecified
        );
    }

    #[test]
    fn compiled_artifact_rebuilds_theta_maps_with_optimizer_basis() {
        let formula = parse_formula("y ~ cond + (0 + cond | subject)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let mut artifact = CompiledModelArtifact::new(formula.to_string(), semantic);
        let optimizer_basis = vec![vec![
            "cond: A".to_string(),
            "cond: B".to_string(),
            "cond: C".to_string(),
        ]];

        artifact.rebuild_theta_maps_for_optimizer_order_with_basis(&[0], &optimizer_basis);

        let block = artifact.theta_maps[0].block();
        assert_eq!(block.user_basis, vec!["cond".to_string()]);
        assert_eq!(block.optimizer_basis, optimizer_basis[0].clone());
        assert_eq!(artifact.theta_maps[0].n_free(), 6);
    }

    #[test]
    fn split_zerocorr_factor_terms_map_to_expanded_optimizer_basis() {
        let formula = parse_formula("y ~ x + f + (1 + f + x || g)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let mut artifact = CompiledModelArtifact::new(formula.to_string(), semantic);
        // A three-level factor f expands to two level-contrast columns in the
        // shared diagonal optimizer block.
        let optimizer_basis = vec![vec![
            "intercept".to_string(),
            "f: b".to_string(),
            "f: c".to_string(),
            "x".to_string(),
        ]];

        artifact.rebuild_theta_maps_for_optimizer_order_with_basis(&[0], &optimizer_basis);

        let errors = artifact
            .diagnostics
            .iter()
            .filter(|diag| diag.severity == crate::compiler::diagnostics::DiagnosticSeverity::Error)
            .map(|diag| diag.message.clone())
            .collect::<Vec<_>>();
        assert!(
            errors.is_empty(),
            "split factor rebuild should not record error diagnostics: {errors:?}"
        );

        assert_eq!(artifact.theta_maps.len(), 3);
        let intercept_map = &artifact.theta_maps[0];
        assert_eq!(intercept_map.n_free(), 1);
        assert!(matches!(intercept_map, ThetaMap::Scalar(_)));
        assert_eq!(intercept_map.block().theta_slots[0].lambda_row, 0);

        let factor_map = &artifact.theta_maps[1];
        assert_eq!(factor_map.block().user_basis, vec!["f".to_string()]);
        assert_eq!(factor_map.n_free(), 2);
        assert!(matches!(factor_map, ThetaMap::Diagonal(_)));
        let factor_slots = &factor_map.block().theta_slots;
        assert_eq!(
            factor_slots
                .iter()
                .map(|slot| (slot.lambda_row, slot.lambda_col, slot.global_index))
                .collect::<Vec<_>>(),
            vec![(1, 1, Some(1)), (2, 2, Some(2))]
        );
        assert_eq!(factor_map.block().optimizer_basis, optimizer_basis[0]);

        let slope_map = &artifact.theta_maps[2];
        assert_eq!(slope_map.n_free(), 1);
        assert_eq!(slope_map.block().theta_slots[0].lambda_row, 3);
        assert_eq!(slope_map.block().theta_slots[0].global_index, Some(3));

        assert_eq!(artifact.covariance_parameter_traces.len(), 4);
    }

    #[test]
    fn optimizer_basis_column_materialization_handles_factor_expansions() {
        assert!(optimizer_basis_column_materializes("x", "x"));
        assert!(optimizer_basis_column_materializes(
            "intercept",
            "intercept"
        ));
        assert!(optimizer_basis_column_materializes("f: b", "f"));
        assert!(optimizer_basis_column_materializes("x:f: b", "x:f"));
        assert!(optimizer_basis_column_materializes("f: a:x", "f:x"));
        assert!(optimizer_basis_column_materializes("f: a:g: b", "f:g"));

        assert!(!optimizer_basis_column_materializes("x2", "x"));
        assert!(!optimizer_basis_column_materializes("g: a", "f"));
        assert!(!optimizer_basis_column_materializes("x", "f"));
        assert!(!optimizer_basis_column_materializes("f", "f:x"));
        assert!(!optimizer_basis_column_materializes("f: a", "f:x"));
    }

    #[test]
    fn compiled_artifact_traces_structured_covariance_placeholders() {
        let formula = parse_formula("y ~ x + ar1(0 + x | subject)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let artifact = CompiledModelArtifact::new(formula.to_string(), semantic);
        let value = serde_json::to_value(&artifact).unwrap();

        assert_eq!(
            value["semantic_model"]["random_terms"][0]["covariance_support"],
            "parsed_refused"
        );
        assert_eq!(value["theta_maps"][0]["family"], "structured");
        assert_eq!(
            value["theta_maps"][0]["map"]["covariance_family"],
            serde_json::json!({"structured": {"kind": "ar1"}})
        );
        assert_eq!(
            value["theta_maps"][0]["map"]["support_status"],
            "parsed_refused"
        );
        assert_eq!(artifact.theta_maps[0].n_free(), 0);
        assert_eq!(artifact.covariance_parameter_traces.len(), 1);
        assert_eq!(
            value["covariance_parameter_traces"][0]["covariance_support"],
            "parsed_refused"
        );
        assert_eq!(
            value["covariance_parameter_traces"][0]["theta"]["status"],
            "not_assessed"
        );
        assert!(value["covariance_parameter_traces"][0]["theta"]["global_index"].is_null());
        assert_eq!(
            value["covariance_parameter_traces"][0]["varcorr_entries"][0]["value"],
            serde_json::Value::Null
        );
    }

    #[test]
    fn compiled_artifact_round_trips_json() {
        let formula = parse_formula("y ~ x + (1 + x || subject)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let artifact = CompiledModelArtifact::new(formula.to_string(), semantic);

        let json = serde_json::to_string(&artifact).unwrap();
        let decoded: CompiledModelArtifact = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, artifact);
    }

    #[test]
    fn compiled_artifact_serializes_prefit_inference_table_as_null() {
        let formula = parse_formula("y ~ x + (1 | subject)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let artifact = CompiledModelArtifact::new(formula.to_string(), semantic);

        let value = serde_json::to_value(&artifact).unwrap();

        assert!(value["fixed_effect_inference_table"].is_null());
        assert!(value["fixed_effect_covariance_matrix"].is_null());
        assert!(artifact
            .table(ArtifactTable::FixedEffectInference)
            .is_none());
        assert!(artifact
            .table(ArtifactTable::FixedEffectCovariance)
            .is_none());
        assert!(artifact
            .table_by_name(FIXED_EFFECT_INFERENCE_TABLE_NAME)
            .unwrap()
            .is_none());
        assert!(artifact
            .table_by_name(FIXED_EFFECT_COVARIANCE_MATRIX_NAME)
            .unwrap()
            .is_none());
    }

    #[test]
    fn compiled_artifact_deserializes_missing_inference_table_as_none() {
        let formula = parse_formula("y ~ x + (1 | subject)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let artifact = CompiledModelArtifact::new(formula.to_string(), semantic);
        let mut value = serde_json::to_value(&artifact).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .remove("fixed_effect_inference_table");
        value
            .as_object_mut()
            .unwrap()
            .remove("fixed_effect_covariance_matrix");

        let decoded: CompiledModelArtifact = serde_json::from_value(value).unwrap();

        assert!(decoded.fixed_effect_inference_table.is_none());
        assert!(decoded.fixed_effect_covariance_matrix.is_none());
    }

    #[test]
    fn fixed_effect_inference_table_round_trips_on_artifact() {
        let formula = parse_formula("y ~ x + (1 | subject)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let mut artifact = CompiledModelArtifact::new(formula.to_string(), semantic);
        artifact.fixed_effect_inference_table = Some(FixedEffectInferenceTable::new(vec![
            FixedEffectInferenceRow {
                label: "x".to_string(),
                kind: FixedEffectInferenceRowKind::Coefficient,
                estimate: Some(1.25),
                std_error: Some(0.5),
                numerator_df: None,
                denominator_df: None,
                statistic: Some(2.5),
                statistic_name: Some(FixedEffectStatisticName::Z),
                p_value: Some(0.0124),
                method: FixedEffectInferenceMethod::AsymptoticWaldZ,
                status: FixedEffectInferenceStatus::Available,
                reliability: ReliabilityGrade::Low,
                reliability_reason: Some(FixedEffectReliabilityReason::AsymptoticWaldZFallback),
                estimability: EstimabilityAssessment::FixedContrast(
                    super::super::estimability::FixedContrastEstimability::estimable("x", 1, 1),
                ),
                reason: None,
                details: None,
                notes: vec![
                    "asymptotic Wald z is a labeled fallback, not a finite-sample correction"
                        .to_string(),
                ],
            },
        ]));

        let json = serde_json::to_string(&artifact).unwrap();
        let decoded: CompiledModelArtifact = serde_json::from_str(&json).unwrap();
        let table = decoded
            .fixed_effect_inference_table
            .as_ref()
            .expect("table should round-trip");

        assert_eq!(table.schema_name, FIXED_EFFECT_INFERENCE_TABLE_SCHEMA);
        assert_eq!(
            table.schema_version,
            FIXED_EFFECT_INFERENCE_TABLE_SCHEMA_VERSION
        );
        assert_eq!(table.rows.len(), 1);
        assert_eq!(table.rows[0].label, "x");
        assert_eq!(table.rows[0].numerator_df, None);
        assert_eq!(table.rows[0].denominator_df, None);
        assert_eq!(decoded, artifact);

        let table_value = artifact
            .table(ArtifactTable::FixedEffectInference)
            .expect("bridge table accessor should expose fitted inference table");
        assert_eq!(
            table_value["schema_name"],
            FIXED_EFFECT_INFERENCE_TABLE_SCHEMA
        );
        assert_eq!(
            table_value["schema_version"],
            FIXED_EFFECT_INFERENCE_TABLE_SCHEMA_VERSION
        );
        assert_eq!(table_value["rows"][0]["label"], "x");

        let named = artifact
            .table_by_name(FIXED_EFFECT_INFERENCE_TABLE_NAME)
            .unwrap()
            .expect("named bridge table accessor should expose inference table");
        assert_eq!(named, table_value);
        let alias = artifact
            .table_by_name("fixed_effect_inference_table")
            .unwrap()
            .expect("legacy descriptive table name should remain accepted");
        assert_eq!(alias, table_value);
        let err = artifact.table_by_name("coeftable").unwrap_err();
        assert!(err.contains("unsupported artifact table"));
    }

    #[test]
    fn fixed_effect_covariance_matrix_round_trips_on_artifact() {
        let formula = parse_formula("y ~ x + (1 | subject)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let mut artifact = CompiledModelArtifact::new(formula.to_string(), semantic);
        let details = FixedEffectCovarianceDetails {
            rank: Some(2),
            expected_rank: Some(2),
            aliased: Vec::new(),
            matrix_rows: 2,
            matrix_cols: 2,
            finite: Some(true),
            symmetric: Some(true),
        };
        artifact.fixed_effect_covariance_matrix = Some(FixedEffectCovarianceMatrix::model_based(
            vec!["(Intercept)".to_string(), "x".to_string()],
            vec![vec![4.0, 0.25], vec![0.25, 1.0]],
            details,
            vec![
                "model-based fixed-effect covariance geometry; inference claims remain on fixed_effect_inference_table rows"
                    .to_string(),
            ],
        ));

        let json = serde_json::to_string(&artifact).unwrap();
        let decoded: CompiledModelArtifact = serde_json::from_str(&json).unwrap();
        let covariance = decoded
            .fixed_effect_covariance_matrix
            .as_ref()
            .expect("covariance matrix should round-trip");

        assert_eq!(
            covariance.schema_name,
            FIXED_EFFECT_COVARIANCE_MATRIX_SCHEMA
        );
        assert_eq!(
            covariance.schema_version,
            FIXED_EFFECT_COVARIANCE_MATRIX_SCHEMA_VERSION
        );
        assert_eq!(covariance.status, FixedEffectCovarianceStatus::Available);
        assert_eq!(covariance.method, FixedEffectCovarianceMethod::ModelBased);
        assert_eq!(covariance.matrix.as_ref().unwrap()[1][1], 1.0);
        assert_eq!(decoded, artifact);

        let table_value = artifact
            .table(ArtifactTable::FixedEffectCovariance)
            .expect("bridge table accessor should expose fitted covariance matrix");
        assert_eq!(
            table_value["schema_name"],
            FIXED_EFFECT_COVARIANCE_MATRIX_SCHEMA
        );
        assert_eq!(table_value["matrix"][0][1], 0.25);

        let named = artifact
            .table_by_name(FIXED_EFFECT_COVARIANCE_MATRIX_NAME)
            .unwrap()
            .expect("named bridge table accessor should expose covariance matrix");
        assert_eq!(named, table_value);
    }

    #[test]
    fn compiled_artifact_can_attach_design_audit() {
        let mut data = DataFrame::new();
        data.add_numeric("y", vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        data.add_numeric("x", vec![0.0, 1.0, 0.0, 1.0]).unwrap();
        data.add_categorical(
            "subject",
            vec!["s1", "s1", "s2", "s2"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        )
        .unwrap();

        let formula = parse_formula("y ~ x + (1 + x | subject)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let mut artifact = CompiledModelArtifact::new(formula.to_string(), semantic);
        artifact.attach_design_audit(&data);

        let audit = artifact.design_audit.as_ref().expect("audit should attach");
        assert_eq!(audit.random_terms[0].group.n_levels, Some(2));
        assert!(!artifact.policy_recommendations.is_empty());
        assert!(artifact.reductions.is_empty());
    }

    #[test]
    fn compiled_artifact_exposes_requested_supported_fitted_model_state() {
        let mut data = DataFrame::new();
        data.add_numeric("y", vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        data.add_numeric("x", vec![0.0, 1.0, 0.0, 1.0]).unwrap();
        data.add_categorical(
            "subject",
            vec!["s1", "s1", "s2", "s2"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        )
        .unwrap();

        let formula = parse_formula("y ~ x + (1 + x | subject)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let mut artifact = CompiledModelArtifact::new(formula.to_string(), semantic);
        artifact.attach_design_audit(&data);

        let state = artifact.model_state_summary();
        assert_eq!(state.schema_name, MODEL_STATE_SUMMARY_SCHEMA);
        assert_eq!(state.requested.status, ModelStateStatus::Requested);
        assert_eq!(state.semantic.status, ModelStateStatus::Canonical);
        assert_eq!(state.supported.status, ModelStateStatus::Refused);
        assert_eq!(state.fitted.status, ModelStateStatus::NotAssessed);
        assert_eq!(state.supported.random_terms[0].basis_dimension, 2);
        assert_eq!(
            state.supported.random_terms[0]
                .information_status
                .as_deref(),
            Some("too_rich")
        );
        assert!(state
            .changes
            .iter()
            .any(|change| change.status == ModelChangeStatus::Recommended
                && change.trigger == ReductionTrigger::DesignTime));
        assert_eq!(artifact.changes(), state.changes);
    }

    #[test]
    fn compiled_artifact_can_disable_policy_recommendations() {
        let mut data = DataFrame::new();
        data.add_numeric("y", vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        data.add_numeric("x", vec![0.0, 1.0, 0.0, 1.0]).unwrap();
        data.add_categorical(
            "subject",
            vec!["s1", "s1", "s2", "s2"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        )
        .unwrap();

        let formula = parse_formula("y ~ x + (1 + x | subject)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let mut artifact = CompiledModelArtifact::new(formula.to_string(), semantic);
        artifact.set_compiler_policy(super::CompilerPolicy::as_specified());
        artifact.attach_design_audit(&data);

        assert!(artifact.policy_recommendations.is_empty());
    }

    #[test]
    fn compiled_artifact_policy_updates_reproducibility_thresholds() {
        let formula = parse_formula("y ~ x + (1 + x | subject)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let mut artifact = CompiledModelArtifact::new(formula.to_string(), semantic);
        let mut policy = CompilerPolicy::maximal_feasible();
        policy.thresholds.effective_rank_relative_tolerance = 0.25;

        artifact.set_compiler_policy(policy);

        assert!(artifact
            .reproducibility
            .thresholds
            .iter()
            .any(|(name, value)| name == "effective_rank_relative_tolerance" && value == "0.25"));
    }

    #[test]
    fn compiled_artifact_records_effective_covariance_summary() {
        let formula = parse_formula("y ~ x + (1 + x | subject)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let mut artifact = CompiledModelArtifact::new(formula.to_string(), semantic);

        artifact.record_effective_covariance_summary(EffectiveCovarianceSummary {
            term_id: "r0".to_string(),
            source_syntax: "(1 + x | subject)".to_string(),
            requested_basis: vec!["intercept".to_string(), "x".to_string()],
            requested_rank: 2,
            supported_rank: 1,
            status: EffectiveRankStatus::ReducedRank,
            directions: vec![SupportedCovarianceDirection {
                label: "PC1".to_string(),
                loadings: vec![
                    BasisLoading {
                        basis: "intercept".to_string(),
                        loading: 0.7,
                    },
                    BasisLoading {
                        basis: "x".to_string(),
                        loading: 0.3,
                    },
                ],
                eigenvalue: Some(1.0),
                variance_explained: Some(0.95),
                user_scale_summary: "0.700*intercept + 0.300*x".to_string(),
            }],
            unsupported_directions: Vec::new(),
            inference_consequence: "conditional on effective covariance rank".to_string(),
            interpretable_submodel: None,
        });

        let json = serde_json::to_string(&artifact).unwrap();
        let decoded: CompiledModelArtifact = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.effective_covariance.len(), 1);
        assert_eq!(
            decoded.effective_covariance[0].status,
            EffectiveRankStatus::ReducedRank
        );
        assert_eq!(
            decoded.effective_covariance[0].directions[0].loadings[0].basis,
            "intercept"
        );
    }

    #[test]
    fn compiled_artifact_builds_audit_report() {
        let mut data = DataFrame::new();
        data.add_numeric("y", vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        data.add_numeric("x", vec![0.0, 1.0, 0.0, 1.0]).unwrap();
        data.add_categorical(
            "subject",
            vec!["s1", "s1", "s2", "s2"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        )
        .unwrap();

        let formula = parse_formula("y ~ x + (1 + x | subject)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let mut artifact = CompiledModelArtifact::new(formula.to_string(), semantic);
        artifact.attach_design_audit(&data);

        let report = artifact.audit_report();
        let text = report.to_text();

        assert_eq!(report.requested_formula, formula.to_string());
        assert!(text.contains("Random Effects"));
        assert!(text.contains("Optimizer"));
        assert!(text.contains("model has not been fitted"));
    }
}
