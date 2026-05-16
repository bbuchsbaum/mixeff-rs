//! Post-fit statistical summaries and inference.
//!
//! These operate on a fitted model from [`crate::model`]:
//!
//! - [`VarCorr`] — variance components / correlations (residual-source aware).
//! - [`CoefTable`] / [`ModelSummary`] — fixed-effect estimates and the overall
//!   fit summary; [`CoefTable`] surfaces Wald and, when available,
//!   Satterthwaite / Kenward-Roger degrees-of-freedom rows.
//! - [`LikelihoodRatioTest`] / [`ModelComparisonTable`] /
//!   [`BoundaryLikelihoodRatioTest`] — model comparison with a typed taxonomy
//!   and stable reason codes.
//! - [`profile()`] and [`parametricbootstrap`](crate::model::parametricbootstrap)
//!   — profile-likelihood and parametric-bootstrap confidence intervals.
//!
//! Inference follows the project's no-fake-statistics stance: unavailable
//! quantities are returned as explicit, typed refusals (with stable reason
//! codes and versioned JSON schemas) rather than fabricated numbers.

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
    assess_model_comparison_sequence, parametric_bootstrap_lrt, BoundaryLikelihoodRatioTest,
    BoundaryLrtMixtureComponent, BoundaryLrtStatus, FixedEffectComparison, LikelihoodRatioTest,
    LinearModelFit, ModelComparisonAlternative, ModelComparisonAssessment, ModelComparisonClass,
    ModelComparisonMethod, ModelComparisonOptions, ModelComparisonReasonCode,
    ModelComparisonRefitPolicy, ModelComparisonRow, ModelComparisonTable, ParametricBootstrapLrt,
    RandomEffectComparison, BOUNDARY_LRT_SCHEMA, BOUNDARY_LRT_SCHEMA_VERSION,
    PARAMETRIC_BOOTSTRAP_LRT_SCHEMA, PARAMETRIC_BOOTSTRAP_LRT_SCHEMA_VERSION,
};
pub use model_summary::{
    FitSummaryPayload, ModelSummary, ModelSummaryRow, FIT_SUMMARY_SCHEMA,
    FIT_SUMMARY_SCHEMA_VERSION,
};
pub use profile::{
    profile, profile_beta, profile_betas, profile_confint_payload, profile_sigma, profile_theta,
    profile_theta_scalar, ConfintRow, MixedModelProfile, ProfileLikelihoodCiPayload,
    ProfileLikelihoodCiRow, ProfileRow, PROFILE_LIKELIHOOD_CI_SCHEMA,
    PROFILE_LIKELIHOOD_CI_SCHEMA_VERSION,
};
pub use varcorr::{VarCorr, VarCorrComponent};
