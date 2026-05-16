//! Error types for the mixeff-rs crate.

use thiserror::Error;

/// Top-level error type for all mixed model operations.
#[derive(Error, Debug)]
#[non_exhaustive]
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
#[non_exhaustive]
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

impl MixedModelError {
    /// Stable machine-readable error code for downstream bindings.
    ///
    /// These strings are part of the public error contract. The display text
    /// may improve over time, but callers that need branching behavior should
    /// use this code instead of parsing [`std::fmt::Display`] output.
    pub fn code(&self) -> &'static str {
        match self {
            MixedModelError::Formula(_) => "formula",
            MixedModelError::LinAlg(_) => "linalg",
            MixedModelError::Optimization(_) => "optimization",
            MixedModelError::DimensionMismatch(_) => "dimension_mismatch",
            MixedModelError::NotFitted => "not_fitted",
            MixedModelError::AlreadyFitted => "already_fitted",
            MixedModelError::ConstantResponse => "constant_response",
            MixedModelError::NoRandomEffects => "no_random_effects",
            MixedModelError::InvalidArgument(_) => "invalid_argument",
            MixedModelError::Unsupported(_) => "unsupported",
            MixedModelError::UnsupportedFamilyLink { .. } => "unsupported_family_link",
            MixedModelError::ProblemTooLarge(_) => "problem_too_large",
            MixedModelError::Singular(_) => "singular_model",
            MixedModelError::RankSaturatedFixedEffects { .. } => "rank_saturated_fixed_effects",
            MixedModelError::PosDefException => "positive_definite_exception",
        }
    }
}

impl LinAlgError {
    /// Stable machine-readable code for linear algebra errors.
    pub fn code(&self) -> &'static str {
        match self {
            LinAlgError::NotPositiveDefinite => "matrix_not_positive_definite",
            LinAlgError::DimensionMismatch(_) => "dimension_mismatch",
            LinAlgError::Singular => "singular_matrix",
            LinAlgError::RankDeficient { .. } => "rank_deficient",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_model_error_codes_are_stable_machine_strings() {
        let err = MixedModelError::InvalidArgument("bad input".to_string());
        assert_eq!(err.code(), "invalid_argument");

        let linalg = LinAlgError::RankDeficient {
            rank: 2,
            expected: 3,
        };
        assert_eq!(linalg.code(), "rank_deficient");
        assert_eq!(MixedModelError::LinAlg(linalg).code(), "linalg");
    }
}
