//! Blocked matrix storage used by the mixed-model PLS system.

use nalgebra::{DMatrix, DVector};
use nalgebra_sparse::csc::CscMatrix;

/// A block in the lower-triangular blocked matrix system.
///
/// The blocked system stores the lower triangle of `[Z1 Z2 ... X y]'[Z1 Z2 ... X y]`.
/// Blocks can be dense, diagonal, block-diagonal, or sparse.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum MatrixBlock {
    /// Dense rectangular block stored as a full matrix.
    Dense(DMatrix<f64>),
    /// Sparse rectangular block stored in compressed sparse column format.
    Sparse(CscMatrix<f64>),
    /// Square diagonal block stored by its diagonal entries.
    Diagonal(DVector<f64>),
    /// Uniform block diagonal: `nlevels` blocks each of size `vsize x vsize`.
    /// Total matrix is `(nlevels * vsize) x (nlevels * vsize)`.
    BlockDiagonal(Vec<DMatrix<f64>>),
}

impl MatrixBlock {
    /// Number of rows represented by this block.
    pub fn nrows(&self) -> usize {
        match self {
            MatrixBlock::Dense(m) => m.nrows(),
            MatrixBlock::Sparse(m) => m.nrows(),
            MatrixBlock::Diagonal(v) => v.len(),
            MatrixBlock::BlockDiagonal(blocks) => blocks.iter().map(|b| b.nrows()).sum(),
        }
    }

    /// Number of columns represented by this block.
    pub fn ncols(&self) -> usize {
        match self {
            MatrixBlock::Dense(m) => m.ncols(),
            MatrixBlock::Sparse(m) => m.ncols(),
            MatrixBlock::Diagonal(v) => v.len(),
            MatrixBlock::BlockDiagonal(blocks) => blocks.iter().map(|b| b.ncols()).sum(),
        }
    }

    /// Materialize this block as a dense matrix.
    pub fn as_dense(&self) -> DMatrix<f64> {
        match self {
            MatrixBlock::Dense(m) => m.clone(),
            MatrixBlock::Sparse(m) => {
                let mut result = DMatrix::zeros(m.nrows(), m.ncols());
                for (row, col, value) in m.triplet_iter() {
                    result[(row, col)] += *value;
                }
                result
            }
            MatrixBlock::Diagonal(v) => DMatrix::from_diagonal(v),
            MatrixBlock::BlockDiagonal(blocks) => {
                let total_rows = blocks.iter().map(|b| b.nrows()).sum();
                let total_cols = blocks.iter().map(|b| b.ncols()).sum();
                let mut result = DMatrix::zeros(total_rows, total_cols);
                let mut row_offset = 0;
                let mut col_offset = 0;
                for blk in blocks {
                    for i in 0..blk.nrows() {
                        for j in 0..blk.ncols() {
                            result[(row_offset + i, col_offset + j)] = blk[(i, j)];
                        }
                    }
                    row_offset += blk.nrows();
                    col_offset += blk.ncols();
                }
                result
            }
        }
    }

    /// Borrow the underlying dense matrix when this block is dense.
    pub fn as_dense_ref(&self) -> Option<&DMatrix<f64>> {
        match self {
            MatrixBlock::Dense(m) => Some(m),
            MatrixBlock::Sparse(_) => None,
            _ => None,
        }
    }

    /// Mutably borrow the underlying dense matrix when this block is dense.
    pub fn as_dense_mut(&mut self) -> Option<&mut DMatrix<f64>> {
        match self {
            MatrixBlock::Dense(m) => Some(m),
            MatrixBlock::Sparse(_) => None,
            _ => None,
        }
    }

    /// Borrow the diagonal vector when this block is diagonal.
    pub fn as_diag_ref(&self) -> Option<&DVector<f64>> {
        match self {
            MatrixBlock::Diagonal(v) => Some(v),
            _ => None,
        }
    }

    /// Mutably borrow the diagonal vector when this block is diagonal.
    pub fn as_diag_mut(&mut self) -> Option<&mut DVector<f64>> {
        match self {
            MatrixBlock::Diagonal(v) => Some(v),
            _ => None,
        }
    }
}

/// Convert lower-triangle block coordinates to the packed block index.
pub(crate) fn block_index(i: usize, j: usize) -> usize {
    debug_assert!(i >= j);
    i * (i + 1) / 2 + j
}

pub(crate) fn with_block_pair_mut<T, F>(
    blocks: &mut [MatrixBlock],
    dst_idx: usize,
    src_idx: usize,
    f: F,
) -> T
where
    F: FnOnce(&mut MatrixBlock, &MatrixBlock) -> T,
{
    debug_assert_ne!(dst_idx, src_idx);

    if dst_idx < src_idx {
        let (left, right) = blocks.split_at_mut(src_idx);
        f(&mut left[dst_idx], &right[0])
    } else {
        let (left, right) = blocks.split_at_mut(dst_idx);
        f(&mut right[0], &left[src_idx])
    }
}

pub(crate) fn with_block_triple<T, F>(
    blocks: &mut [MatrixBlock],
    target_idx: usize,
    src_a_idx: usize,
    src_b_idx: usize,
    f: F,
) -> crate::error::Result<T>
where
    F: FnOnce(&mut MatrixBlock, &MatrixBlock, &MatrixBlock) -> T,
{
    debug_assert_ne!(target_idx, src_a_idx);
    debug_assert_ne!(target_idx, src_b_idx);
    debug_assert_ne!(src_a_idx, src_b_idx);

    let n_blocks = blocks.len();
    if target_idx >= n_blocks {
        return Err(crate::error::MixedModelError::DimensionMismatch(format!(
            "blocked Cholesky target index {target_idx} is out of bounds for {n_blocks} blocks"
        )));
    }

    let (before, target_and_after) = blocks.split_at_mut(target_idx);
    let (target, after) = target_and_after
        .split_first_mut()
        .expect("target_idx < n_blocks checked above");

    let get_src = |idx: usize| -> &MatrixBlock {
        if idx < target_idx {
            &before[idx]
        } else {
            &after[idx - target_idx - 1]
        }
    };

    Ok(f(target, get_src(src_a_idx), get_src(src_b_idx)))
}

pub(crate) fn with_dense_block<T, F>(block: &MatrixBlock, f: F) -> T
where
    F: FnOnce(&DMatrix<f64>) -> T,
{
    match block {
        MatrixBlock::Dense(mat) => f(mat),
        _ => {
            let dense = block.as_dense();
            f(&dense)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_block_unit_solves() {
        let diag = MatrixBlock::Diagonal(DVector::from_vec(vec![2.0, 3.0]));
        assert_eq!(diag.nrows(), 2);
        assert_eq!(diag.ncols(), 2);
        assert_eq!(
            diag.as_dense(),
            DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 3.0])
        );

        let block = MatrixBlock::BlockDiagonal(vec![
            DMatrix::from_row_slice(1, 1, &[4.0]),
            DMatrix::from_row_slice(2, 2, &[5.0, 6.0, 7.0, 8.0]),
        ]);
        assert_eq!(block.nrows(), 3);
        assert_eq!(block.ncols(), 3);
        assert_eq!(
            block.as_dense(),
            DMatrix::from_row_slice(3, 3, &[4.0, 0.0, 0.0, 0.0, 5.0, 6.0, 0.0, 7.0, 8.0])
        );

        assert_eq!(block_index(0, 0), 0);
        assert_eq!(block_index(1, 0), 1);
        assert_eq!(block_index(2, 1), 4);
    }
}
