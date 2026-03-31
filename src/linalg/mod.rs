//! Linear algebra operations for mixed models.
//!
//! This module contains the core linear algebra routines ported from
//! Julia's MixedModels.jl, including rank updates, unblocked Cholesky
//! factorization, pivoted QR, log-determinant computation, and block
//! matrix operations used in building and updating the blocked lower
//! Cholesky factor.

// Re-export LinAlgError so submodules can use `crate::linalg::LinAlgError`
pub use crate::error::LinAlgError;

pub mod block_ops;
pub mod chol_unblocked;
pub mod logdet;
pub mod pivot;
pub mod rank_update;

pub use block_ops::*;
pub use chol_unblocked::*;
pub use logdet::*;
pub use pivot::*;
pub use rank_update::*;
