//! Mixed model types and fitting algorithms.

pub mod data;
pub mod fixed_design;
pub mod generalized;
pub mod linear;
pub mod traits;

pub use data::{CategoricalColumn, Column, DataFrame};
pub use fixed_design::{
    CompiledMixedModelDesign, DenseFixedDesign, FixedDesign, FixedDesignBackend,
    FixedDesignBackendPreference, FixedDesignBuildPolicy, FixedDesignStorage, FixedDesignSummary,
    StreamedFixedDesign,
};
pub use generalized::GeneralizedLinearMixedModel;
pub use linear::{
    parametricbootstrap, BootstrapFailedRefitPolicy, BootstrapInterval, BootstrapIntervalMethod,
    BootstrapQuantile, BootstrapRefitOptions, BootstrapReplicate, BootstrapRunMetadata,
    BootstrapRunPayload, BootstrapSeedRecord, BootstrapTarget, BootstrapTargetKind,
    ConvergenceVerificationOptions, FixedEffectBootstrapOptions, FixedEffectNullBootstrapTarget,
    FixedEffectNullCovariancePolicy, KenwardRogerAdjustedVcov, KenwardRogerLbDdf,
    KenwardRogerSigmaG, LinearMixedModel, MixedModelBootstrap, ModelDims, NewReLevels,
    ResponseMatrixProfile, VcovVarparEstimate, BOOTSTRAP_RUN_SCHEMA, BOOTSTRAP_RUN_SCHEMA_VERSION,
};
pub use traits::{Family, LinkFunction, MixedModelFit, RandomEffectTermInfo};
