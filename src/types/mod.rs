//! Core types for the mixed-effects models library.
//!
//! This module re-exports the primary data structures used throughout the
//! crate:
//!
//! * [`UniformBlockDiagonal`] -- homogeneous block diagonal matrix
//! * [`BlockedSparse`] -- sparse matrix with block structure
//! * [`RaggedArray`] -- jagged array for group-wise accumulation
//! * [`GaussHermiteNormalized`] and [`gh_norm`] -- Gauss-Hermite quadrature
//! * [`FeTerm`] -- fixed-effects model matrix with pivoted QR rank detection
//! * [`FeMat`] -- concatenated `[X | y]` matrix with optional weighting
//! * [`ReMat`] -- random-effects model matrix for one grouping factor
//! * [`OptSummary`] -- optimisation state, tolerances, and fit log

mod blocked_sparse;
mod gauss_hermite;
mod ragged_array;
mod uniform_block_diagonal;
pub mod fe_term;
pub mod fe_mat;
pub mod re_mat;
pub mod opt_summary;

pub use blocked_sparse::BlockedSparse;
pub use gauss_hermite::{gh_norm, GaussHermiteNormalized};
pub use ragged_array::RaggedArray;
pub use uniform_block_diagonal::UniformBlockDiagonal;
pub use fe_term::FeTerm;
pub use fe_mat::FeMat;
pub use re_mat::ReMat;
pub use opt_summary::{OptSummary, FitLogEntry, Optimizer};
