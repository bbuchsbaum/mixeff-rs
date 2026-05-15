//! Statistical methods: VarCorr, bootstrap, likelihood ratio tests.

pub mod block_description;
pub mod bootstrap;
pub mod coeftable;
pub mod lrt;
pub mod model_summary;
pub mod profile;
pub mod spline;
pub mod varcorr;

pub use block_description::BlockDescription;
pub use bootstrap::{
    restore_replicates, restorereplicates, save_replicates, savereplicates, shortest_cov_int,
};
pub use coeftable::{coeftable_to_markdown, CoefTable, CoefTablePValuePolicy};
pub use lrt::{
    assess_model_comparison_sequence, BoundaryLikelihoodRatioTest, BoundaryLrtMixtureComponent,
    BoundaryLrtStatus, FixedEffectComparison, LikelihoodRatioTest, LinearModelFit,
    ModelComparisonAlternative, ModelComparisonAssessment, ModelComparisonClass,
    ModelComparisonMethod, ModelComparisonOptions, ModelComparisonReasonCode,
    ModelComparisonRefitPolicy, ModelComparisonRow, ModelComparisonTable, RandomEffectComparison,
    BOUNDARY_LRT_SCHEMA, BOUNDARY_LRT_SCHEMA_VERSION,
};
pub use model_summary::{ModelSummary, ModelSummaryRow};
pub use profile::{
    profile, profile_beta, profile_betas, profile_confint_payload, profile_sigma, profile_theta,
    profile_theta_scalar, ConfintRow, MixedModelProfile, ProfileLikelihoodCiPayload,
    ProfileLikelihoodCiRow, ProfileRow, PROFILE_LIKELIHOOD_CI_SCHEMA,
    PROFILE_LIKELIHOOD_CI_SCHEMA_VERSION,
};
pub use varcorr::{VarCorr, VarCorrComponent};
