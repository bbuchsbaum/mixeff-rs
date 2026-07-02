//! Linear algebra operations for mixed models.
//!
//! This module contains the core linear algebra routines ported from
//! Julia's MixedModels.jl, including rank updates, unblocked Cholesky
//! factorization, pivoted QR, log-determinant computation, and block
//! matrix operations used in building and updating the blocked lower
//! Cholesky factor.

// Re-export LinAlgError so submodules can use `crate::linalg::LinAlgError`
pub use crate::error::LinAlgError;

// `block_ops`, `chol_unblocked`, `logdet`, and `rank_update` are faithful
// MixedModels.jl numerical-primitive ports kept with full per-function
// unit-test coverage for parity. They are not wired into the active fit path
// — `model::linear` carries its own blocked Cholesky/PLS routines — so they
// have no non-test caller. Demoting `linalg` to `pub(crate)` (v1.0 API trim)
// made that visible as dead_code. Retaining the tested ports is deliberate;
// whether to wire them in or remove them is tracked as a follow-up rather than
// silently suppressed here. See `docs/linalg_primitive_audit.md` for the
// caller map and post-1.0 decision boundary.
//
// `pivot` is NOT in that group: it is live infrastructure (`compiler::audit`,
// `stats::lrt`, and `types::fe_term` depend on `pivoted_qr_with_tol`,
// `stats_rank_with_tol`, and `stats_rank`). Only the no-tol `pivoted_qr`
// convenience wrapper is currently uncalled, so the dead_code allowance is
// scoped to that single function in `pivot.rs` instead of the whole module.
#[allow(dead_code)]
pub mod block_ops;
#[allow(dead_code)]
pub mod chol_unblocked;
#[allow(dead_code)]
pub mod logdet;
pub mod pivot;
#[allow(dead_code)]
pub mod rank_update;

// `stats_rank` is the only primitive consumed through this facade
// (`stats::lrt`); everything else is reached via submodule paths, so no
// blanket re-export is warranted.
pub use pivot::stats_rank;
pub use pivot::{gram_full_rank_certificate, GramRankCertificate, GRAM_CERTIFICATE_SAFETY_FACTOR};
