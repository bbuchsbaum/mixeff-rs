//! Error types for the mixeff-rs crate.

use thiserror::Error;

/// Top-level error type for all mixed model operations.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum MixedModelError {
    /// Formula parsing or formula-lowering failed.
    #[error("Formula error: {0}")]
    Formula(#[from] crate::formula::FormulaError),

    /// Linear algebra backend reported a numerical failure.
    #[error("Linear algebra error: {0}")]
    LinAlg(#[from] LinAlgError),

    /// Optimizer failed, exhausted its budget, or returned an unusable state.
    #[error("Optimization error: {0}")]
    Optimization(String),

    /// Input arrays or matrices had incompatible dimensions.
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    /// Operation requires fitted model state, but the model has not been fit.
    #[error("Model not fitted: call fit() first")]
    NotFitted,

    /// Operation is only valid before the first fit.
    #[error("Model already fitted: use refit() instead")]
    AlreadyFitted,

    /// A host callback requested that the current fit or inference loop stop.
    #[error("Operation interrupted: {0}")]
    Interrupted(String),

    /// The response is constant and cannot identify the requested model.
    #[error("Constant response: model fitting failed")]
    ConstantResponse,

    /// The formula has no random-effects terms.
    #[error("No random effects in formula: this is not a mixed model")]
    NoRandomEffects,

    /// Caller supplied an invalid argument value.
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    /// Requested model or operation is outside the implemented support matrix.
    #[error("Unsupported model: {0}")]
    Unsupported(String),

    /// Requested GLMM family/link pair is unsupported.
    #[error("Unsupported family/link combination: {family}/{link}")]
    UnsupportedFamilyLink {
        /// Distribution family label.
        family: String,
        /// Link-function label.
        link: String,
    },

    /// Requested design would exceed configured memory or size limits.
    #[error("Problem too large: {0}")]
    ProblemTooLarge(String),

    /// Model fit or covariance structure is singular in a context that
    /// requires full rank.
    #[error("Singular model: {0}")]
    Singular(String),

    /// Fixed-effect rank leaves no residual degrees of freedom.
    #[error("Fixed-effect design is rank-saturated: rank(X) = {rank} and n = {nobs}, leaving zero residual degrees of freedom. Ordinary unpenalized LMM fitting is not identifiable; use fewer fixed effects or an explicit penalized/MAP fixed-effect prior.")]
    RankSaturatedFixedEffects {
        /// Numerical rank of the fixed-effect design.
        rank: usize,
        /// Number of observations.
        nobs: usize,
    },

    /// Cholesky factorization encountered a non-positive-definite matrix.
    #[error("Positive definite exception during Cholesky")]
    PosDefException,
}

/// Error type for linear algebra operations.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum LinAlgError {
    /// Matrix was expected to be positive definite but was not.
    #[error("Matrix is not positive definite")]
    NotPositiveDefinite,

    /// Matrix/vector dimensions are incompatible.
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    /// Matrix is singular.
    #[error("Singular matrix")]
    Singular,

    /// Matrix rank was lower than required.
    #[error("Rank deficient matrix (rank {rank}, expected {expected})")]
    RankDeficient {
        /// Observed numerical rank.
        rank: usize,
        /// Required rank for the operation.
        expected: usize,
    },
}

/// Crate-wide result type using [`MixedModelError`].
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
            MixedModelError::Interrupted(_) => "interrupted",
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
