//! Error types for the mixeff-rs crate.

use thiserror::Error;

/// Top-level error type for all mixed model operations.
#[derive(Error, Debug)]
pub enum MixedModelError {
    #[error("Formula error: {0}")]
    Formula(#[from] crate::formula::FormulaError),

    #[error("Linear algebra error: {0}")]
    LinAlg(#[from] LinAlgError),

    #[error("Optimization error: {0}")]
    Optimization(String),

    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    #[error("Model not fitted: call fit() first")]
    NotFitted,

    #[error("Model already fitted: use refit() instead")]
    AlreadyFitted,

    #[error("Constant response: model fitting failed")]
    ConstantResponse,

    #[error("No random effects in formula: this is not a mixed model")]
    NoRandomEffects,

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("Unsupported model: {0}")]
    Unsupported(String),

    #[error("Unsupported family/link combination: {family}/{link}")]
    UnsupportedFamilyLink { family: String, link: String },

    #[error("Problem too large: {0}")]
    ProblemTooLarge(String),

    #[error("Singular model: {0}")]
    Singular(String),

    #[error("Fixed-effect design is rank-saturated: rank(X) = {rank} and n = {nobs}, leaving zero residual degrees of freedom. Ordinary unpenalized LMM fitting is not identifiable; use fewer fixed effects or an explicit penalized/MAP fixed-effect prior.")]
    RankSaturatedFixedEffects { rank: usize, nobs: usize },

    #[error("Positive definite exception during Cholesky")]
    PosDefException,
}

/// Error type for linear algebra operations.
#[derive(Error, Debug)]
pub enum LinAlgError {
    #[error("Matrix is not positive definite")]
    NotPositiveDefinite,

    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    #[error("Singular matrix")]
    Singular,

    #[error("Rank deficient matrix (rank {rank}, expected {expected})")]
    RankDeficient { rank: usize, expected: usize },
}

pub type Result<T> = std::result::Result<T, MixedModelError>;
