//! Blocked sparse matrices.
//!
//! A `BlockedSparse` wraps a CSC sparse matrix whose nonzero entries form
//! rectangular blocks of rows and/or columns. The nonzero values are also
//! accessible as a dense matrix (`nzs_as_mat`) for efficient block-level
//! operations.
//!
//! This is a Rust port of `BlockedSparse{T,S,P}` from MixedModels.jl.

use nalgebra::DMatrix;
use nalgebra_sparse::CscMatrix;

/// A sparse matrix whose nonzeros form blocks.
///
/// In the Julia source `BlockedSparse{T,S,P}`, the type parameters `S` and `P`
/// denote the row-block size and column-block size respectively. In this Rust
/// port they are stored as plain `usize` fields.
///
/// # Fields
///
/// * `cscmat` - the underlying sparse matrix in Compressed Sparse Column format
/// * `nzs_as_mat` - the nonzero values reshaped as a dense matrix, providing a
///   block-oriented view of the same data
/// * `col_block_ptr` - indices into the columns that delimit blocks of columns
/// * `row_block_size` - the row block size (corresponds to `S` in Julia)
/// * `col_block_size` - the column block size (corresponds to `P` in Julia)
#[derive(Debug, Clone)]
pub struct BlockedSparse {
    /// CSC representation for general sparse-matrix calculations.
    pub cscmat: CscMatrix<f64>,
    /// Nonzero values of `cscmat` reshaped as a dense matrix.
    pub nzs_as_mat: DMatrix<f64>,
    /// Pattern of blocks of columns (0-based indices).
    pub col_block_ptr: Vec<usize>,
    /// Row block size (the `S` type parameter in the Julia version).
    pub row_block_size: usize,
    /// Column block size (the `P` type parameter in the Julia version).
    pub col_block_size: usize,
}

impl BlockedSparse {
    /// Create a new `BlockedSparse`.
    ///
    /// # Arguments
    ///
    /// * `cscmat` - the CSC sparse matrix
    /// * `nzs_as_mat` - nonzeros reshaped as a dense matrix
    /// * `col_block_ptr` - column-block boundary indices
    /// * `row_block_size` - row block size
    /// * `col_block_size` - column block size
    pub fn new(
        cscmat: CscMatrix<f64>,
        nzs_as_mat: DMatrix<f64>,
        col_block_ptr: Vec<usize>,
        row_block_size: usize,
        col_block_size: usize,
    ) -> Self {
        Self {
            cscmat,
            nzs_as_mat,
            col_block_ptr,
            row_block_size,
            col_block_size,
        }
    }

    /// Dimensions of the sparse matrix `(nrows, ncols)`.
    #[inline]
    pub fn size(&self) -> (usize, usize) {
        (self.cscmat.nrows(), self.cscmat.ncols())
    }

    /// Number of rows.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.cscmat.nrows()
    }

    /// Number of columns.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.cscmat.ncols()
    }

    /// Number of stored (structural) nonzeros.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.cscmat.nnz()
    }

    /// Element access `(i, j)` with zero-based indices.
    ///
    /// Delegates to the underlying CSC matrix. Returns `0.0` for structural
    /// zeros.
    pub fn get(&self, i: usize, j: usize) -> f64 {
        // nalgebra_sparse CscMatrix does not have a direct (i,j) accessor that
        // returns 0 for structural zeros, so we search the column manually.
        let col = self.cscmat.col(j);
        let row_indices = col.row_indices();
        let values = col.values();
        match row_indices.binary_search(&i) {
            Ok(pos) => values[pos],
            Err(_) => 0.0,
        }
    }

    /// Convert to a full dense matrix.
    pub fn to_dense(&self) -> DMatrix<f64> {
        let (nrows, ncols) = self.size();
        let mut dense = DMatrix::zeros(nrows, ncols);
        // Iterate over columns of the CSC matrix
        for j in 0..ncols {
            let col = self.cscmat.col(j);
            for (&row, &val) in col.row_indices().iter().zip(col.values().iter()) {
                dense[(row, j)] = val;
            }
        }
        dense
    }

    /// Density of the matrix (fraction of nonzero entries).
    pub fn density(&self) -> f64 {
        let (m, n) = self.size();
        let total = m * n;
        if total == 0 {
            0.0
        } else {
            self.nnz() as f64 / total as f64
        }
    }

    /// Return a dense matrix if density exceeds `threshold`, otherwise
    /// return `None`.
    ///
    /// Mirrors the Julia `densify` function.
    pub fn densify(&self, threshold: f64) -> Option<DMatrix<f64>> {
        if self.density() > threshold {
            Some(self.to_dense())
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra_sparse::CooMatrix;

    /// Helper: build a small blocked-sparse matrix for testing.
    fn example_blocked_sparse() -> BlockedSparse {
        // Build a 4x4 sparse matrix with a 2x2 block in the top-left and
        // bottom-right.
        let mut coo = CooMatrix::new(4, 4);
        // Top-left block
        coo.push(0, 0, 1.0);
        coo.push(0, 1, 2.0);
        coo.push(1, 0, 3.0);
        coo.push(1, 1, 4.0);
        // Bottom-right block
        coo.push(2, 2, 5.0);
        coo.push(2, 3, 6.0);
        coo.push(3, 2, 7.0);
        coo.push(3, 3, 8.0);

        let csc = CscMatrix::from(&coo);
        let nnz = csc.nnz();

        // nzs_as_mat: nonzeros laid out as a (4, 2) matrix (4 nnz per block, 2 blocks)
        // This is illustrative; the exact reshape depends on usage context.
        let nzs = DMatrix::from_column_slice(nnz / 2, 2, csc.values());

        let col_block_ptr = vec![0, 2, 4];

        BlockedSparse::new(csc, nzs, col_block_ptr, 2, 2)
    }

    #[test]
    fn test_size() {
        let bs = example_blocked_sparse();
        assert_eq!(bs.size(), (4, 4));
        assert_eq!(bs.nrows(), 4);
        assert_eq!(bs.ncols(), 4);
    }

    #[test]
    fn test_nnz() {
        let bs = example_blocked_sparse();
        assert_eq!(bs.nnz(), 8);
    }

    #[test]
    fn test_element_access() {
        let bs = example_blocked_sparse();
        assert_relative_eq!(bs.get(0, 0), 1.0);
        assert_relative_eq!(bs.get(1, 1), 4.0);
        assert_relative_eq!(bs.get(3, 3), 8.0);
        // Structural zero
        assert_relative_eq!(bs.get(0, 2), 0.0);
        assert_relative_eq!(bs.get(2, 0), 0.0);
    }

    #[test]
    fn test_to_dense() {
        let bs = example_blocked_sparse();
        let dense = bs.to_dense();
        assert_eq!((dense.nrows(), dense.ncols()), (4, 4));
        assert_relative_eq!(dense[(0, 0)], 1.0);
        assert_relative_eq!(dense[(2, 3)], 6.0);
        assert_relative_eq!(dense[(0, 3)], 0.0);
    }

    #[test]
    fn test_density() {
        let bs = example_blocked_sparse();
        assert_relative_eq!(bs.density(), 0.5);
    }

    #[test]
    fn test_densify_below_threshold() {
        let bs = example_blocked_sparse();
        // density is 0.5, threshold 0.6 → should densify
        assert!(bs.densify(0.4).is_some());
        // threshold 0.6 → density 0.5 is not > 0.6
        assert!(bs.densify(0.6).is_none());
    }
}
