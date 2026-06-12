// This module is the `unstable-internals` public surface. On a default build
// it is `pub(crate)` and many types / re-exports here have no in-crate
// consumer (they were only used by external test/example crates, which are
// now feature-gated), so they read as `unused_imports` / `dead_code`.
// Suppress those lints ONLY when the feature is off; with `unstable-internals`
// enabled this is real public API and full linting stays in force (verified:
// clippy --features unstable-internals -D warnings is clean).
#![cfg_attr(not(feature = "unstable-internals"), allow(unused_imports, dead_code))]
//! Compiler-contract layer for mixed model specifications.
//!
//! This module is intentionally additive. It records semantic model meaning,
//! diagnostics, parameterization maps, and explanation artifacts without
//! changing the existing numerical fitting path.

pub mod artifact;
pub mod audit;
pub mod diagnostics;
pub mod estimability;
pub mod explain;
pub mod ir;
pub mod policy;
pub mod print;
pub mod random_term_card;
pub mod report;
pub mod theta_map;

pub use artifact::{
    ArtifactTable, BasisLoading, BootstrapInferenceDetails, CompiledModelArtifact,
    ContrastFamilyDetails, CovarianceParameterTrace, DerivativeAvailability, DominantLoading,
    EffectiveCovarianceSummary, EffectiveRankStatus, FitIntent, FitMode,
    FixedEffectCovarianceDetails, FixedEffectCovarianceMatrix, FixedEffectCovarianceMethod,
    FixedEffectCovarianceStatus, FixedEffectInferenceDetails, FixedEffectInferenceMethod,
    FixedEffectInferenceRow, FixedEffectInferenceRowKind, FixedEffectInferenceStatus,
    FixedEffectInferenceTable, FixedEffectNullTargetSummary, FixedEffectReliabilityReason,
    FixedEffectStatisticName, GlmmFitMetadata, InferenceAvailability, InterpretableSubmodel,
    KenwardRogerInferenceDetails, LambdaSlotTrace, ModelBoundary, ModelChangeStatus, ModelKind,
    ModelRandomTermState, ModelStageState, ModelStateChange, ModelStateStage, ModelStateStatus,
    ModelStateSummary, ObjectiveApproximation, OptimizerCertificateScope, ParmapTrace,
    ReductionRecord, ReductionTrigger, ReproducibilityRecord, SchemaMetadata,
    SupportedCovarianceDirection, ThetaSlotTrace, VarCorrEntryKind, VarCorrEntryTrace,
    DOMINANT_LOADING_THRESHOLD, FIXED_EFFECT_COVARIANCE_MATRIX_NAME,
    FIXED_EFFECT_COVARIANCE_MATRIX_SCHEMA, FIXED_EFFECT_COVARIANCE_MATRIX_SCHEMA_VERSION,
    FIXED_EFFECT_INFERENCE_TABLE_NAME, FIXED_EFFECT_INFERENCE_TABLE_SCHEMA,
    FIXED_EFFECT_INFERENCE_TABLE_SCHEMA_VERSION, INTERPRETABLE_GAP_TOLERANCE,
};
pub use audit::{
    audit_design, BasisAudit, CertificateCheck, ConvergenceEvidence, ConvergenceVerification,
    ConvergenceVerificationRun, ConvergenceVerificationStatus, CovarianceKernelAudit,
    CovarianceKernelGraphAudit, DependencePathAudit, DependencePathKind, DesignAudit,
    EmptyCellAudit, EvidenceMethod, EvidenceQuality, FitAudit, FixedEffectAudit,
    FixedEffectColumnAudit, FixedEffectColumnKind, FixedEffectTermAudit, FixedEffectTermStatus,
    GradientEvidence, GroupingAudit, HessianEvidence, InformationBudgetStatus,
    MissingDependencePathAudit, OptimizerCertificate, OptimizerDerivativeEvidence,
    OptimizerStopEvidence, ParameterSpaceEvidence, RandomEffectEffectiveNReport,
    RandomEffectInformationBudget, RandomTermAudit, RankAssessment, RankStatus, SampleSizeContext,
};
pub use diagnostics::{Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticStage, FitStatus};
pub use estimability::{
    ContrastMatrix, ContrastRhs, EstimabilityAssessment, EstimabilityStatus,
    FixedContrastEstimability, FixedEffectHypothesis, FixedEffectTermTestType, FixedEffectTest,
    FixedEffectTestMethod, FixedTermEstimability, InferenceMethod, InferenceStatus,
    KernelPathEstimability, RandomCovarianceEstimability, RandomVarianceEstimability,
    ReliabilityGrade,
};
pub use explain::{explain_model, ModelExplanation};
pub use ir::{
    compile_formula_ir, CovarianceForm, CovarianceStory, CovarianceSupportStatus, GroupingFactorIr,
    GroupingRole, InterceptPolicy, RandomCoefficient, RandomCoefficientKind, RandomTermIr,
    SemanticModel, SourceSyntax, StructuredCovarianceKind,
};
pub use policy::{
    recommend_policy, CompilerPolicy, CompilerThresholds, PolicyAction, PolicyRecommendation,
    RandomStrategy,
};
pub use print::{ModelPrint, ParameterizationDrilldown, MODEL_PRINT_TOP_DIAGNOSTICS};
pub use random_term_card::{
    CrossCardConstraint, DesignSupport, ImpliedConstraint, ImpliedConstraintKind, RandomTermBlock,
    RandomTermCard, RoleOrigin, WithinGroupVariation, RANDOM_TERM_CARD_SCHEMA,
    RANDOM_TERM_CARD_SCHEMA_VERSION,
};
pub use report::{
    AuditReportLine, AuditReportSection, AuditReportStatus, ConvergenceLevel, ConvergenceSource,
    ConvergenceVerdict, ModelAuditReport,
};
pub use theta_map::{
    CovarianceFamily, CovarianceFamilyTransition, ParameterConstraint, ParameterStatus, ThetaMap,
    ThetaMapBlock, ThetaSlot,
};
