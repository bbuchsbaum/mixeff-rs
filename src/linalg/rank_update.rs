//! Rank-k update operations for building the blocked Cholesky factor.
//!
//! These are ports of the `rankUpdate!` family from Julia's MixedModels.jl
//! (`linalg/rankUpdate.jl`). The key operation is `C := α * A' * A + β * C`
//! for symmetric `C` stored as a full dense matrix (lower triangle is
//! authoritative).

use nalgebra::{DMatrix, DVector};
use nalgebra_sparse::CscMatrix;

use crate::linalg::LinAlgError;

/// Symmetric rank-k update for a dense matrix: `C := α * A' * A + β * C`.
///
/// The result is written into the **lower triangle** of `C` (the upper
/// triangle is not touched). This mirrors the BLAS `dsyrk` convention
/// used in Julia's `rankUpdate!` for `HermOrSym{T, StridedMatrix}`.
///
/// # Errors
///
/// Returns [`LinAlgError::DimensionMismatch`] when `C` is not square or
/// `C.nrows() != A.ncols()`.
pub fn rank_update_dense(
    c: &mut DMatrix<f64>,
    a: &DMatrix<f64>,
    alpha: f64,
    beta: f64,
) -> Result<(), LinAlgError> {
    let (m, k) = (a.nrows(), a.ncols());
    if c.nrows() != k || c.ncols() != k {
        return Err(LinAlgError::DimensionMismatch(format!(
            "C is {}x{} but A'A would be {}x{}",
            c.nrows(),
            c.ncols(),
            k,
            k
        )));
    }

    // Scale lower triangle of C by beta.
    if beta != 1.0 {
        for j in 0..k {
            for i in j..k {
                c[(i, j)] *= beta;
            }
        }
    }

    // Accumulate α * A' * A into the lower triangle.
    // C[i,j] += α * Σ_l A[l,i] * A[l,j]  for i >= j
    for j in 0..k {
        for i in j..k {
            let mut dot = 0.0;
            for l in 0..m {
                dot += a[(l, i)] * a[(l, j)];
            }
            c[(i, j)] += alpha * dot;
        }
    }

    Ok(())
}

/// Symmetric rank-k update for a diagonal target stored as a vector:
/// `c_diag[i] := α * Σ_j A[i,j]² + β * c_diag[i]`.
///
/// This is the specialisation for `Hermitian{Diagonal}` in Julia, where `C`
/// is diagonal (stored as a vector) and only the diagonal of `A' * A` is
/// needed.
///
/// # Errors
///
/// Returns [`LinAlgError::DimensionMismatch`] when `c_diag.len() != A.ncols()`.
pub fn rank_update_diag(
    c_diag: &mut DVector<f64>,
    a: &DMatrix<f64>,
    alpha: f64,
    beta: f64,
) -> Result<(), LinAlgError> {
    let (_m, k) = (a.nrows(), a.ncols());
    if c_diag.len() != k {
        return Err(LinAlgError::DimensionMismatch(format!(
            "c_diag has length {} but A has {} columns",
            c_diag.len(),
            k
        )));
    }

    for i in 0..k {
        let row_sumsq: f64 = a.column(i).iter().map(|&v| v * v).sum();
        c_diag[i] = alpha * row_sumsq + beta * c_diag[i];
    }

    Ok(())
}

/// General rank update with sparse `A` and dense `B`:
/// `C := α * A' * B + β * C`.
///
/// This computes the product of the transpose of a sparse CSC matrix with
/// a dense matrix and accumulates the result into `C`. It is used for
/// off-diagonal blocks in the blocked system where one factor is sparse
/// (the Z matrix of a random-effect term).
///
/// # Errors
///
/// Returns [`LinAlgError::DimensionMismatch`] on incompatible dimensions.
pub fn rank_update_sparse_dense(
    c: &mut DMatrix<f64>,
    a: &CscMatrix<f64>,
    b: &DMatrix<f64>,
    alpha: f64,
    beta: f64,
) -> Result<(), LinAlgError> {
    let (a_rows, a_cols) = (a.nrows(), a.ncols());
    let (b_rows, b_cols) = (b.nrows(), b.ncols());

    if a_rows != b_rows {
        return Err(LinAlgError::DimensionMismatch(format!(
            "A has {} rows but B has {} rows; they must match for A'*B",
            a_rows, b_rows
        )));
    }
    if c.nrows() != a_cols || c.ncols() != b_cols {
        return Err(LinAlgError::DimensionMismatch(format!(
            "C is {}x{} but A'*B would be {}x{}",
            c.nrows(),
            c.ncols(),
            a_cols,
            b_cols
        )));
    }

    // Scale C by beta.
    if beta != 1.0 {
        c.scale_mut(beta);
    }

    // Accumulate α * A' * B.
    // For each column j of A (in CSC), iterate over its nonzeros.
    // A'[j, row] = A[row, j] = val, so C[j, :] += α * val * B[row, :].
    let col_offsets = a.col_offsets();
    let row_indices = a.row_indices();
    let values = a.values();

    for col_a in 0..a_cols {
        let start = col_offsets[col_a];
        let end = col_offsets[col_a + 1];
        for idx in start..end {
            let row = row_indices[idx];
            let a_val = alpha * values[idx];
            for col_b in 0..b_cols {
                c[(col_a, col_b)] += a_val * b[(row, col_b)];
            }
        }
    }

    Ok(())
}

/// Symmetric rank update from a sparse matrix: `C := α * A' * A + β * C`.
///
/// Only the **lower triangle** of `C` is updated. This is the sparse
/// analogue of [`rank_update_dense`] and ports the
/// `rankUpdate!(C::HermOrSym, A::SparseMatrixCSC, α, β)` method from Julia.
///
/// # Errors
///
/// Returns [`LinAlgError::DimensionMismatch`] when `C` is not
/// `A.nrows() x A.nrows()`.
pub fn rank_update_sparse(
    c: &mut DMatrix<f64>,
    a: &CscMatrix<f64>,
    alpha: f64,
    beta: f64,
) -> Result<(), LinAlgError> {
    let (m, _n) = (a.nrows(), a.ncols());
    if c.nrows() != m || c.ncols() != m {
        return Err(LinAlgError::DimensionMismatch(format!(
            "C is {}x{} but A'A (lower) requires {}x{}",
            c.nrows(),
            c.ncols(),
            m,
            m
        )));
    }

    // Scale lower triangle of C by beta.
    if beta != 1.0 {
        for j in 0..m {
            for i in j..m {
                c[(i, j)] *= beta;
            }
        }
    }

    // For each column of A, do a symmetric outer-product update on the
    // nonzero rows. This mirrors the Julia inner loop that iterates over
    // nzrange(A, jj) and updates Cd[rv[kk], rvj].
    let col_offsets = a.col_offsets();
    let row_indices = a.row_indices();
    let values = a.values();

    for col in 0..a.ncols() {
        let start = col_offsets[col];
        let end = col_offsets[col + 1];

        // For each pair (k, j) where k >= j in this column's nonzeros,
        // update C[row_k, row_j] += α * val_k * val_j.
        for jj in start..end {
            let row_j = row_indices[jj];
            let a_val_j = alpha * values[jj];
            for kk in jj..end {
                let row_k = row_indices[kk];
                // Ensure lower triangle: row_k >= row_j
                if row_k >= row_j {
                    c[(row_k, row_j)] += values[kk] * a_val_j;
                } else {
                    c[(row_j, row_k)] += values[kk] * a_val_j;
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Helper: build a CscMatrix from triplets using CooMatrix (nalgebra-sparse 0.10 API).
    fn csc_from_triplets(
        nrows: usize,
        ncols: usize,
        row_indices: &[usize],
        col_indices: &[usize],
        values: &[f64],
    ) -> CscMatrix<f64> {
        use nalgebra_sparse::CooMatrix;
        let mut coo = CooMatrix::new(nrows, ncols);
        for i in 0..row_indices.len() {
            coo.push(row_indices[i], col_indices[i], values[i]);
        }
        CscMatrix::from(&coo)
    }

    #[test]
    fn test_rank_update_dense_identity() {
        // A = I(2), so A'A = I. With alpha=1, beta=0, C should become I.
        let a = DMatrix::identity(2, 2);
        let mut c = DMatrix::zeros(2, 2);
        rank_update_dense(&mut c, &a, 1.0, 0.0).unwrap();
        assert_relative_eq!(c[(0, 0)], 1.0, epsilon = 1e-12);
        assert_relative_eq!(c[(1, 1)], 1.0, epsilon = 1e-12);
        assert_relative_eq!(c[(1, 0)], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_rank_update_dense_known_values() {
        // A = [[1, 2], [3, 4]], A'A = [[10, 14], [14, 20]]
        // With alpha=2, beta=3, C_init = [[1,0],[0,1]]:
        // C_new = 2*A'A + 3*I = [[23, 28], [28, 43]]
        // (lower triangle only)
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let mut c = DMatrix::identity(2, 2);
        rank_update_dense(&mut c, &a, 2.0, 3.0).unwrap();
        assert_relative_eq!(c[(0, 0)], 23.0, epsilon = 1e-12);
        assert_relative_eq!(c[(1, 0)], 28.0, epsilon = 1e-12);
        assert_relative_eq!(c[(1, 1)], 43.0, epsilon = 1e-12);
    }

    #[test]
    fn test_rank_update_dense_rectangular() {
        // A is 3x2, so A'A is 2x2.
        // A = [[1,0],[0,1],[1,1]], A'A = [[2,1],[1,2]]
        let a = DMatrix::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let mut c = DMatrix::zeros(2, 2);
        rank_update_dense(&mut c, &a, 1.0, 0.0).unwrap();
        assert_relative_eq!(c[(0, 0)], 2.0, epsilon = 1e-12);
        assert_relative_eq!(c[(1, 0)], 1.0, epsilon = 1e-12);
        assert_relative_eq!(c[(1, 1)], 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_rank_update_dense_dimension_mismatch() {
        let a = DMatrix::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let mut c = DMatrix::zeros(3, 3);
        let result = rank_update_dense(&mut c, &a, 1.0, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_rank_update_diag() {
        // A = [[1,2],[3,4],[5,6]], columns: col0=[1,3,5], col1=[2,4,6]
        // diag(A'A) = [1+9+25, 4+16+36] = [35, 56]
        let a = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut c = DVector::from_vec(vec![1.0, 1.0]);
        rank_update_diag(&mut c, &a, 1.0, 0.0).unwrap();
        assert_relative_eq!(c[0], 35.0, epsilon = 1e-12);
        assert_relative_eq!(c[1], 56.0, epsilon = 1e-12);
    }

    #[test]
    fn test_rank_update_diag_with_scaling() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let mut c = DVector::from_vec(vec![2.0, 3.0]);
        rank_update_diag(&mut c, &a, 2.0, 0.5).unwrap();
        // c[0] = 2.0 * 1.0 + 0.5 * 2.0 = 3.0
        // c[1] = 2.0 * 1.0 + 0.5 * 3.0 = 3.5
        assert_relative_eq!(c[0], 3.0, epsilon = 1e-12);
        assert_relative_eq!(c[1], 3.5, epsilon = 1e-12);
    }

    #[test]
    fn test_rank_update_sparse_dense() {
        // A (sparse, 3x2): [[1,0],[0,1],[1,1]]
        // B (dense, 3x2): [[2,0],[0,3],[1,1]]
        // A'B = [[3,1],[1,4]]
        let a = csc_from_triplets(3, 2, &[0, 2, 1, 2], &[0, 0, 1, 1], &[1.0, 1.0, 1.0, 1.0]);
        let b = DMatrix::from_row_slice(3, 2, &[2.0, 0.0, 0.0, 3.0, 1.0, 1.0]);
        let mut c = DMatrix::zeros(2, 2);
        rank_update_sparse_dense(&mut c, &a, &b, 1.0, 0.0).unwrap();
        assert_relative_eq!(c[(0, 0)], 3.0, epsilon = 1e-12);
        assert_relative_eq!(c[(0, 1)], 1.0, epsilon = 1e-12);
        assert_relative_eq!(c[(1, 0)], 1.0, epsilon = 1e-12);
        assert_relative_eq!(c[(1, 1)], 4.0, epsilon = 1e-12);
    }

    #[test]
    fn test_rank_update_sparse_dense_with_scaling() {
        // A sparse 2x2 identity, B dense 2x2 = [[1,2],[3,4]]
        // A'B = B. With alpha=2, beta=3, C_init=I:
        // C = 2*B + 3*I = [[5,4],[6,7]]
        let a = csc_from_triplets(2, 2, &[0, 1], &[0, 1], &[1.0, 1.0]);
        let b = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let mut c = DMatrix::identity(2, 2);
        rank_update_sparse_dense(&mut c, &a, &b, 2.0, 3.0).unwrap();
        assert_relative_eq!(c[(0, 0)], 5.0, epsilon = 1e-12);
        assert_relative_eq!(c[(0, 1)], 4.0, epsilon = 1e-12);
        assert_relative_eq!(c[(1, 0)], 6.0, epsilon = 1e-12);
        assert_relative_eq!(c[(1, 1)], 11.0, epsilon = 1e-12);
    }

    #[test]
    fn test_rank_update_sparse_symmetric() {
        // Square sparse A so C = A'A has the same shape as A (the
        // implementation requires C to be a.nrows() x a.nrows()).
        let a2 = csc_from_triplets(2, 2, &[0, 1, 0, 1], &[0, 0, 1, 1], &[1.0, 2.0, 3.0, 4.0]);
        // A = [[1,3],[2,4]]
        // A'A (lower) via column-wise outer product:
        // col 0: rows [0,1] vals [1,2] -> outer = [[1,2],[2,4]]
        // col 1: rows [0,1] vals [3,4] -> outer = [[9,12],[12,16]]
        // sum = [[10,14],[14,20]]
        let mut c = DMatrix::zeros(2, 2);
        rank_update_sparse(&mut c, &a2, 1.0, 0.0).unwrap();
        assert_relative_eq!(c[(0, 0)], 10.0, epsilon = 1e-12);
        assert_relative_eq!(c[(1, 0)], 14.0, epsilon = 1e-12);
        assert_relative_eq!(c[(1, 1)], 20.0, epsilon = 1e-12);
    }

    #[test]
    fn test_rank_update_sparse_dense_dimension_mismatch() {
        let a = csc_from_triplets(3, 2, &[0, 1], &[0, 1], &[1.0, 1.0]);
        let b = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let mut c = DMatrix::zeros(2, 2);
        let result = rank_update_sparse_dense(&mut c, &a, &b, 1.0, 0.0);
        assert!(result.is_err());
    }
}
