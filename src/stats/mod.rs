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
    assess_model_comparison_sequence, FixedEffectComparison, LikelihoodRatioTest, LinearModelFit,
    ModelComparisonAlternative, ModelComparisonAssessment, ModelComparisonClass,
    ModelComparisonMethod, ModelComparisonOptions, ModelComparisonRefitPolicy, ModelComparisonRow,
    ModelComparisonTable, RandomEffectComparison,
};
pub use model_summary::{ModelSummary, ModelSummaryRow};
pub use profile::{
    profile, profile_beta, profile_betas, profile_sigma, profile_theta, profile_theta_scalar,
    ConfintRow, MixedModelProfile, ProfileRow,
};
pub use varcorr::{VarCorr, VarCorrComponent};
