//! Linear algebra operations for mixed models.
//!
//! This module contains the core linear algebra routines ported from
//! Julia's MixedModels.jl, including rank updates, unblocked Cholesky
//! factorization, pivoted QR, log-determinant computation, and block
//! matrix operations used in building and updating the blocked lower
//! Cholesky factor.

// Re-export LinAlgError so submodules can use `crate::linalg::LinAlgError`
pub use crate::error::LinAlgError;

// These are faithful MixedModels.jl numerical-primitive ports kept with full
// per-function unit-test coverage for parity. Not all are wired into the
// active fit path — `model::linear` carries its own blocked Cholesky/PLS
// routines — so several primitives have no non-test caller. Demoting `linalg`
// to `pub(crate)` (v1.0 API trim) made that visible as dead_code. Retaining
// the tested ports is deliberate; whether to wire them in or remove them is
// tracked as a follow-up rather than silently suppressed here. See
// `docs/linalg_primitive_audit.md` for the current caller map and post-1.0
// decision boundary.
#[allow(dead_code)]
pub mod block_ops;
#[allow(dead_code)]
pub mod chol_unblocked;
#[allow(dead_code)]
pub mod logdet;
#[allow(dead_code)]
pub mod pivot;
#[allow(dead_code)]
pub mod rank_update;

// `stats_rank` is the only primitive consumed through this facade
// (`stats::lrt`); everything else is reached via submodule paths, so no
// blanket re-export is warranted.
pub use pivot::stats_rank;
