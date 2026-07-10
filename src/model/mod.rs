//! Mixed model types and fitting algorithms.
//!
//! This is the high-level entry point. Most callers build a
//! [`DataFrame`], parse a formula with
//! [`parse_formula`](crate::formula::parse_formula), and fit through one of
//! the builders:
//!
//! - [`LinearMixedModelBuilder`] / [`LinearMixedModel`] — profiled (RE)ML via
//!   a blocked Cholesky PLS step, with automatic optimizer selection.
//!   [`FitOptions`] carries the ML/REML choice plus optional audit-recorded
//!   optimizer controls.
//! - [`GeneralizedLinearMixedModelBuilder`] /
//!   [`GeneralizedLinearMixedModel`] — PIRLS for the conditional modes with
//!   optional adaptive Gauss-Hermite quadrature; pick a [`Family`] and
//!   [`LinkFunction`].
//!
//! Fitted models implement [`MixedModelFit`] (`coef`, `vcov`, `fitted`,
//! `aic`/`bic`, `theta`, `ranef`, …). Post-fit summaries — variance
//! components, coefficient tables, likelihood-ratio tests, profile and
//! bootstrap CIs — live in [`crate::stats`].

#[doc(hidden)]
pub mod batch;
pub mod data;
#[doc(hidden)]
pub mod fixed_design;
pub mod generalized;
pub(crate) mod kernel;
pub mod linear;
pub mod summary_estimates;
pub mod traits;

pub use data::{
    CategoricalCoding, CategoricalColumn, CategoricalContrast, Column, ContrastSource, DataFrame,
    EncodedCategoricalColumn,
};
pub use generalized::{
    GeneralizedLinearMixedModel, GeneralizedLinearMixedModelBuilder, GlmmFitOptions,
    GlmmPredictionScale,
};
pub use linear::{
    parametricbootstrap, try_parametricbootstrap, ActiveFaceRefit, BootstrapFailedRefitPolicy,
    BootstrapInterval, BootstrapIntervalMethod, BootstrapQuantile, BootstrapRefitOptions,
    BootstrapReplicate, BootstrapRunMetadata, BootstrapRunPayload, BootstrapSeedRecord,
    BootstrapTarget, BootstrapTargetKind, FitOptions, FitProgress, FitProgressCallback,
    FitProgressPhase, FitToleranceOverrides, FixedEffectBootstrapOptions,
    FixedEffectNullBootstrapTarget, FixedEffectNullCovariancePolicy, LinearMixedModel,
    LinearMixedModelBuilder, MixedModelBootstrap, ModelCriterion, NewReLevels, OptimizerChoice,
    OptimizerControl, PredictionVarianceMethod, PredictionVariancePayload, PredictionVarianceRow,
    PredictionVarianceStatus, TrustBqSampleReuse, TrustBqStartLadder, BOOTSTRAP_RUN_SCHEMA,
    BOOTSTRAP_RUN_SCHEMA_VERSION,
};
pub use summary_estimates::{ResidualSource, SamplingVarianceScale, SummaryEstimateOptions};
pub use traits::{Family, LinkFunction, MixedModelFit, RandomEffectTermInfo, WaldConfintRow};
