//! Typed support values for advanced callers and tests.
//!
//! The intentionally stable surface here is small — optimization state and a
//! generic matrix block, exposed for inspecting fits and writing tests:
//!
//! * [`OptSummary`] — optimization state, tolerances, and fit log
//! * [`FitLogEntry`] — one objective evaluation in the fit log
//! * [`Optimizer`] / [`OptimizerSource`] / [`ConvergenceStatus`] — optimizer
//!   choice, provenance, and outcome
//! * [`MatrixBlock`] — a generic dense/diagonal matrix block
//!
//! The remaining containers — [`UniformBlockDiagonal`], [`BlockedSparse`],
//! [`RaggedArray`], [`GaussHermiteNormalized`] / [`gh_norm`], [`FeTerm`],
//! [`FeMat`], [`ReMat`] — are storage backing the fit path. They are visible
//! for in-tree benches and staged downstream work but are implementation
//! detail, not part of the stable surface unless asserted in
//! `tests/public_api.rs`. See `docs/semver_policy.md`.

mod blocked_sparse;
pub mod fe_mat;
pub mod fe_term;
mod gauss_hermite;
pub mod matrix_block;
pub mod opt_summary;
mod ragged_array;
pub mod re_mat;
mod uniform_block_diagonal;

pub use blocked_sparse::BlockedSparse;
pub use fe_mat::FeMat;
pub use fe_term::FeTerm;
pub use gauss_hermite::{gh_norm, GaussHermiteNormalized};
pub use matrix_block::MatrixBlock;
pub use opt_summary::{ConvergenceStatus, FitLogEntry, OptSummary, Optimizer, OptimizerSource};
pub use ragged_array::RaggedArray;
pub use re_mat::ReMat;
pub use uniform_block_diagonal::UniformBlockDiagonal;
