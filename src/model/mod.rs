//! Mixed model types and fitting algorithms.
//!
//! This is the high-level entry point. Most callers build a
//! [`DataFrame`], parse a formula with
//! [`parse_formula`](crate::formula::parse_formula), and fit through one of
//! the builders:
//!
//! - [`LinearMixedModelBuilder`] / [`LinearMixedModel`] — profiled (RE)ML via
//!   a blocked Cholesky PLS step, with automatic optimizer selection.
//!   [`FitOptions`] carries the ML/REML choice and tolerances.
//! - [`GeneralizedLinearMixedModelBuilder`] /
//!   [`GeneralizedLinearMixedModel`] — PIRLS for the conditional modes with
//!   optional adaptive Gauss-Hermite quadrature; pick a [`Family`] and
//!   [`LinkFunction`].
//!
//! Fitted models implement [`MixedModelFit`] (`coef`, `vcov`, `fitted`,
//! `aic`/`bic`, `theta`, `ranef`, …). Post-fit summaries — variance
//! components, coefficient tables, likelihood-ratio tests, profile and
//! bootstrap CIs — live in [`crate::stats`].

pub mod batch;
pub mod data;
pub mod fixed_design;
pub mod generalized;
pub mod linear;
pub mod summary_estimates;
pub mod traits;

pub use batch::{
    BatchOptimizerControl, BatchOptions, BatchThetaGrouping, BatchWarmStart, LinearMixedModelBatch,
    ResponseBatchFit, ResponseBatchMode, ResponseColumnDiagnostic, ResponseDiagnosticReason,
    ResponseFitStatus, ThetaBatch,
};
pub use data::{
    CategoricalCoding, CategoricalColumn, CategoricalContrast, Column, ContrastSource, DataFrame,
    EncodedCategoricalColumn,
};
pub use fixed_design::{
    CompiledMixedModelDesign, DenseFixedDesign, FixedDesign, FixedDesignBackend,
    FixedDesignBackendPreference, FixedDesignBuildPolicy, FixedDesignStorage, FixedDesignSummary,
    StreamedFixedDesign,
};
pub use generalized::{GeneralizedLinearMixedModel, GeneralizedLinearMixedModelBuilder};
pub use linear::{
    parametricbootstrap, BootstrapFailedRefitPolicy, BootstrapInterval, BootstrapIntervalMethod,
    BootstrapQuantile, BootstrapRefitOptions, BootstrapReplicate, BootstrapRunMetadata,
    BootstrapRunPayload, BootstrapSeedRecord, BootstrapTarget, BootstrapTargetKind,
    ConvergenceVerificationOptions, FitOptions, FixedEffectBootstrapOptions,
    FixedEffectNullBootstrapTarget, FixedEffectNullCovariancePolicy, KenwardRogerAdjustedVcov,
    KenwardRogerLbDdf, KenwardRogerSigmaG, LinearMixedModel, LinearMixedModelBuilder,
    MixedModelBootstrap, ModelCriterion, ModelDims, NewReLevels, ResponseMatrixProfile,
    VcovVarparEstimate, BOOTSTRAP_RUN_SCHEMA, BOOTSTRAP_RUN_SCHEMA_VERSION,
};
pub use summary_estimates::{ResidualSource, SamplingVarianceScale, SummaryEstimateOptions};
pub use traits::{Family, LinkFunction, MixedModelFit, RandomEffectTermInfo, WaldConfintRow};
