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
pub use opt_summary::{FitLogEntry, OptSummary, Optimizer};
pub use ragged_array::RaggedArray;
pub use re_mat::ReMat;
pub use uniform_block_diagonal::UniformBlockDiagonal;
