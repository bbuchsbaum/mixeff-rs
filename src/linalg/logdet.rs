//! Log-determinant computation for triangular and block-structured matrices.
//!
//! These routines port the `LD` function from Julia's MixedModels.jl
//! (`linalg/logdet.jl`). The log-determinant of a lower triangular
//! factor `L` is `sum(log(L[i,i]))`, and for a block diagonal it is the
//! sum over blocks.

use nalgebra::{DMatrix, DVector};

/// Log-determinant of a lower triangular matrix.
///
/// Computes `sum_{i} ln(L[i,i])`. This is `log(det(L))` when `L` is
/// triangular (since `det(L) = prod(L[i,i])`).
///
/// The matrix need not be stored as a `LowerTriangular` wrapper; only
/// the diagonal elements are read.
///
/// # Panics
///
/// Panics if `l` is not square.
pub fn logdet_triangular(l: &DMatrix<f64>) -> f64 {
    assert_eq!(
        l.nrows(),
        l.ncols(),
        "logdet_triangular: matrix must be square (got {}x{})",
        l.nrows(),
        l.ncols()
    );
    let n = l.nrows();
    let mut s = 0.0;
    for i in 0..n {
        s += l[(i, i)].ln();
    }
    s
}

/// Log-determinant of a diagonal matrix stored as a vector.
///
/// Computes `sum_{i} ln(d[i])`.
pub fn logdet_diag(d: &DVector<f64>) -> f64 {
    d.iter().map(|&v| v.ln()).sum()
}

/// Log-determinant of a uniform block diagonal matrix's Cholesky factor.
///
/// Each block is assumed to be a lower triangular Cholesky factor. The
/// log-determinant of the full block diagonal is the sum of the
/// log-determinants of the individual blocks.
///
/// This mirrors `LD(d::UniformBlockDiagonal)` from Julia, which sums
/// `log(dat[j, j, k])` over all diagonal elements of all blocks.
///
/// # Panics
///
/// Panics if any block is not square.
pub fn logdet_block_diagonal(blocks: &[DMatrix<f64>]) -> f64 {
    blocks.iter().map(logdet_triangular).sum()
}

/// Log-determinant of a symmetric positive-definite matrix via its
/// Cholesky factor.
///
/// If `L` is the Cholesky factor of `A = L * L'`, then
/// `log(det(A)) = 2 * sum(log(diag(L)))`. This function returns that
/// value.
///
/// # Panics
///
/// Panics if `l` is not square.
pub fn logdet_from_chol(l: &DMatrix<f64>) -> f64 {
    2.0 * logdet_triangular(l)
}

/// Log-determinant of a block-diagonal symmetric PD matrix from its
/// Cholesky blocks.
///
/// Returns `2 * logdet_block_diagonal(blocks)`.
pub fn logdet_block_diagonal_from_chol(blocks: &[DMatrix<f64>]) -> f64 {
    2.0 * logdet_block_diagonal(blocks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_logdet_triangular_identity() {
        let l = DMatrix::identity(3, 3);
        assert_relative_eq!(logdet_triangular(&l), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_logdet_triangular_known() {
        // L = [[2, 0], [1, 3]]
        // det(L) = 6, log(det(L)) = log(2) + log(3)
        let mut l = DMatrix::zeros(2, 2);
        l[(0, 0)] = 2.0;
        l[(1, 0)] = 1.0;
        l[(1, 1)] = 3.0;
        let expected = 2.0_f64.ln() + 3.0_f64.ln();
        assert_relative_eq!(logdet_triangular(&l), expected, epsilon = 1e-12);
    }

    #[test]
    fn test_logdet_triangular_1x1() {
        let l = DMatrix::from_row_slice(1, 1, &[5.0]);
        assert_relative_eq!(logdet_triangular(&l), 5.0_f64.ln(), epsilon = 1e-12);
    }

    #[test]
    fn test_logdet_diag() {
        let d = DVector::from_vec(vec![2.0, 3.0, 5.0]);
        let expected = 2.0_f64.ln() + 3.0_f64.ln() + 5.0_f64.ln();
        assert_relative_eq!(logdet_diag(&d), expected, epsilon = 1e-12);
    }

    #[test]
    fn test_logdet_diag_single() {
        let d = DVector::from_vec(vec![7.0]);
        assert_relative_eq!(logdet_diag(&d), 7.0_f64.ln(), epsilon = 1e-12);
    }

    #[test]
    fn test_logdet_block_diagonal() {
        // Block 0: [[2, 0], [0, 3]] -> logdet = ln(2) + ln(3)
        // Block 1: [[5, 0], [0, 7]] -> logdet = ln(5) + ln(7)
        let b0 = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 3.0]);
        let b1 = DMatrix::from_row_slice(2, 2, &[5.0, 0.0, 0.0, 7.0]);
        let expected = 2.0_f64.ln() + 3.0_f64.ln() + 5.0_f64.ln() + 7.0_f64.ln();
        assert_relative_eq!(logdet_block_diagonal(&[b0, b1]), expected, epsilon = 1e-12);
    }

    #[test]
    fn test_logdet_block_diagonal_single_block() {
        let b = DMatrix::identity(3, 3);
        assert_relative_eq!(logdet_block_diagonal(&[b]), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_logdet_block_diagonal_empty() {
        let blocks: &[DMatrix<f64>] = &[];
        assert_relative_eq!(logdet_block_diagonal(blocks), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_logdet_from_chol() {
        // A = [[4, 2], [2, 5]], L = [[2, 0], [1, 2]]
        // det(A) = 16, log(det(A)) = log(16) = 2 * (log(2) + log(2))
        let mut l = DMatrix::zeros(2, 2);
        l[(0, 0)] = 2.0;
        l[(1, 0)] = 1.0;
        l[(1, 1)] = 2.0;
        let expected = 16.0_f64.ln();
        assert_relative_eq!(logdet_from_chol(&l), expected, epsilon = 1e-12);
    }

    #[test]
    fn test_logdet_block_diagonal_from_chol() {
        let b0 = DMatrix::from_row_slice(1, 1, &[3.0]);
        let b1 = DMatrix::from_row_slice(1, 1, &[4.0]);
        // det = 3*4 = 12 (per block diagonal), but these are chol factors
        // so the actual matrix determinant = 3^2 * 4^2 = 144
        // log(144) = 2 * (ln(3) + ln(4))
        let expected = 2.0 * (3.0_f64.ln() + 4.0_f64.ln());
        assert_relative_eq!(
            logdet_block_diagonal_from_chol(&[b0, b1]),
            expected,
            epsilon = 1e-12
        );
    }

    #[test]
    #[should_panic]
    fn test_logdet_triangular_not_square() {
        let l = DMatrix::zeros(2, 3);
        logdet_triangular(&l);
    }
}
