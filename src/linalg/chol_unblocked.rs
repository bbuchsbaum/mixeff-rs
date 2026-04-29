//! Unblocked (column-by-column) Cholesky factorization.
//!
//! This is a port of `cholUnblocked!` from Julia's MixedModels.jl
//! (`linalg/cholUnblocked.jl`). The key difference from a standard LAPACK
//! `potrf` is that a **zero diagonal element does not cause a failure**;
//! instead the entire row of the factor is set to zero. This is needed
//! because singular random-effects covariance structures can produce
//! zero diagonal elements in the blocked system.

use nalgebra::{DMatrix, DVector};

use crate::linalg::LinAlgError;

/// In-place lower Cholesky factor of a positive-semidefinite matrix.
///
/// Overwrites the **lower triangle** of `a` with its Cholesky factor `L`
/// such that `A = L * L'`. The upper triangle is **not** zeroed.
///
/// Unlike the standard LAPACK `potrf`, when a diagonal element is zero
/// (or negative, which would indicate a non-positive-semidefinite matrix
/// in exact arithmetic but can arise from rounding), the entire row of
/// `L` is set to zero and factorization continues. This handles singular
/// covariance structures that appear in mixed models when a variance
/// component is estimated at zero.
///
/// # Errors
///
/// Returns [`LinAlgError::NotPositiveDefinite`] if a diagonal element is
/// **strictly negative** (below `-tol` where `tol = n * eps * max_diag`),
/// indicating the matrix is not positive semidefinite.
///
/// Returns [`LinAlgError::DimensionMismatch`] if `a` is not square.
pub fn chol_unblocked(a: &mut DMatrix<f64>) -> Result<(), LinAlgError> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinAlgError::DimensionMismatch(format!(
            "Matrix is {}x{}, must be square",
            a.nrows(),
            a.ncols()
        )));
    }

    if n == 0 {
        return Ok(());
    }

    // Tolerance for treating a diagonal as zero.
    let max_diag = (0..n).map(|i| a[(i, i)].abs()).fold(0.0_f64, f64::max);
    let tol = (n as f64) * f64::EPSILON * max_diag;

    for j in 0..n {
        // Subtract contributions from previously computed columns.
        // a[j,j] -= sum_{k<j} L[j,k]^2
        let mut d = a[(j, j)];
        for k in 0..j {
            d -= a[(j, k)] * a[(j, k)];
        }

        if !d.is_finite() {
            return Err(LinAlgError::NotPositiveDefinite);
        }

        if d <= tol {
            if d < -tol {
                return Err(LinAlgError::NotPositiveDefinite);
            }
            // Singular: set row j and the not-yet-computed column tail to zero.
            for k in 0..=j {
                a[(j, k)] = 0.0;
            }
            for i in (j + 1)..n {
                a[(i, j)] = 0.0;
            }
        } else {
            let ljj = d.sqrt();
            a[(j, j)] = ljj;

            // Update the sub-diagonal entries in column j:
            // L[i,j] = (A[i,j] - sum_{k<j} L[i,k]*L[j,k]) / L[j,j]
            for i in (j + 1)..n {
                let mut s = a[(i, j)];
                for k in 0..j {
                    s -= a[(i, k)] * a[(j, k)];
                }
                a[(i, j)] = s / ljj;
            }
        }
    }

    Ok(())
}

/// In-place Cholesky factorization for a diagonal matrix stored as a vector.
///
/// Each diagonal element is replaced by its square root. This mirrors
/// `cholUnblocked!(D::Diagonal, Val{:L})` from Julia.
///
/// # Errors
///
/// Returns [`LinAlgError::NotPositiveDefinite`] if any diagonal element
/// is negative.
pub fn chol_unblocked_diag(d: &mut DVector<f64>) -> Result<(), LinAlgError> {
    for i in 0..d.len() {
        let val = d[i];
        if !(val.is_finite() && val >= 0.0) {
            return Err(LinAlgError::NotPositiveDefinite);
        }
        d[i] = val.sqrt();
    }
    Ok(())
}

/// In-place Cholesky factorization for a uniform block diagonal.
///
/// Factorizes each block independently via [`chol_unblocked`]. This mirrors
/// `cholUnblocked!(D::UniformBlockDiagonal, Val{:L})` from Julia.
///
/// # Errors
///
/// Returns [`LinAlgError::NotPositiveDefinite`] if any block is not
/// positive semidefinite.
pub fn chol_unblocked_blocks(blocks: &mut [DMatrix<f64>]) -> Result<(), LinAlgError> {
    for block in blocks.iter_mut() {
        chol_unblocked(block)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_chol_1x1() {
        let mut a = DMatrix::from_row_slice(1, 1, &[4.0]);
        chol_unblocked(&mut a).unwrap();
        assert_relative_eq!(a[(0, 0)], 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_chol_2x2() {
        // A = [[4, 2], [2, 5]], L = [[2, 0], [1, 2]]
        let mut a = DMatrix::from_row_slice(2, 2, &[4.0, 2.0, 2.0, 5.0]);
        chol_unblocked(&mut a).unwrap();
        assert_relative_eq!(a[(0, 0)], 2.0, epsilon = 1e-12);
        assert_relative_eq!(a[(1, 0)], 1.0, epsilon = 1e-12);
        assert_relative_eq!(a[(1, 1)], 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_chol_3x3() {
        // A = [[4, 2, 1], [2, 5, 3], [1, 3, 6]]
        // L should satisfy L*L' = A
        let mut a = DMatrix::from_row_slice(3, 3, &[4.0, 2.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 6.0]);
        let a_orig = a.clone();
        chol_unblocked(&mut a).unwrap();

        // Verify L * L' = A (using lower triangle of a)
        let n = 3;
        let mut l = DMatrix::zeros(n, n);
        for j in 0..n {
            for i in j..n {
                l[(i, j)] = a[(i, j)];
            }
        }
        let product = &l * l.transpose();
        for i in 0..n {
            for j in 0..n {
                assert_relative_eq!(product[(i, j)], a_orig[(i, j)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_chol_identity() {
        let mut a = DMatrix::identity(4, 4);
        chol_unblocked(&mut a).unwrap();
        for i in 0..4 {
            for j in 0..=i {
                if i == j {
                    assert_relative_eq!(a[(i, j)], 1.0, epsilon = 1e-12);
                } else {
                    assert_relative_eq!(a[(i, j)], 0.0, epsilon = 1e-12);
                }
            }
        }
    }

    #[test]
    fn test_chol_singular_zero_diagonal() {
        // A singular PSD matrix: [[1, 1], [1, 1]]
        // Eigenvalues: 0 and 2, so it's PSD.
        //
        // Our chol_unblocked handles singularity by zeroing the entire
        // row when the diagonal becomes zero. So L[0,0]=1 is computed,
        // then d = A[1,1] - L[1,0]^2 = 1-1 = 0, triggering the
        // singular-row path which zeros row 1 entirely: L = [[1,0],[0,0]].
        // This is the MixedModels.jl convention for singular covariance.
        let mut a = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, 1.0]);
        chol_unblocked(&mut a).unwrap();
        assert_relative_eq!(a[(0, 0)], 1.0, epsilon = 1e-10);
        // Row 1 is zeroed because diagonal went to zero
        assert_relative_eq!(a[(1, 0)], 0.0, epsilon = 1e-10);
        assert_relative_eq!(a[(1, 1)], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_chol_singular_middle_column_clears_column_tail() {
        // The second pivot collapses to zero, but the third column remains
        // informative. The singular column tail must be cleared so later
        // updates do not read stale pre-factorization values.
        let mut a = DMatrix::from_row_slice(
            3,
            3,
            &[
                1.0, 1.0, 1.0, //
                1.0, 1.0, 1.0, //
                1.0, 1.0, 2.0,
            ],
        );
        chol_unblocked(&mut a).unwrap();

        assert_relative_eq!(a[(0, 0)], 1.0, epsilon = 1e-12);
        assert_relative_eq!(a[(1, 0)], 0.0, epsilon = 1e-12);
        assert_relative_eq!(a[(1, 1)], 0.0, epsilon = 1e-12);
        assert_relative_eq!(a[(2, 0)], 1.0, epsilon = 1e-12);
        assert_relative_eq!(a[(2, 1)], 0.0, epsilon = 1e-12);
        assert_relative_eq!(a[(2, 2)], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_chol_rejects_nan_diagonal() {
        let mut a = DMatrix::from_row_slice(2, 2, &[f64::NAN, 0.0, 0.0, 1.0]);
        assert!(matches!(
            chol_unblocked(&mut a),
            Err(LinAlgError::NotPositiveDefinite)
        ));
    }

    #[test]
    fn test_chol_rejects_inf_diagonal() {
        let mut a = DMatrix::from_row_slice(2, 2, &[f64::INFINITY, 0.0, 0.0, 1.0]);
        assert!(matches!(
            chol_unblocked(&mut a),
            Err(LinAlgError::NotPositiveDefinite)
        ));
    }

    #[test]
    fn test_chol_zero_matrix() {
        // A zero matrix is PSD. All elements of L should be zero.
        let mut a = DMatrix::zeros(3, 3);
        chol_unblocked(&mut a).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(a[(i, j)], 0.0, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_chol_diag() {
        let mut d = DVector::from_vec(vec![4.0, 9.0, 16.0]);
        chol_unblocked_diag(&mut d).unwrap();
        assert_relative_eq!(d[0], 2.0, epsilon = 1e-12);
        assert_relative_eq!(d[1], 3.0, epsilon = 1e-12);
        assert_relative_eq!(d[2], 4.0, epsilon = 1e-12);
    }

    #[test]
    fn test_chol_diag_with_zero() {
        let mut d = DVector::from_vec(vec![4.0, 0.0, 9.0]);
        chol_unblocked_diag(&mut d).unwrap();
        assert_relative_eq!(d[0], 2.0, epsilon = 1e-12);
        assert_relative_eq!(d[1], 0.0, epsilon = 1e-12);
        assert_relative_eq!(d[2], 3.0, epsilon = 1e-12);
    }

    #[test]
    fn test_chol_diag_negative() {
        let mut d = DVector::from_vec(vec![4.0, -1.0, 9.0]);
        let result = chol_unblocked_diag(&mut d);
        assert!(result.is_err());
    }

    #[test]
    fn test_chol_diag_rejects_nan_entry() {
        let mut d = DVector::from_vec(vec![4.0, f64::NAN, 9.0]);
        assert!(matches!(
            chol_unblocked_diag(&mut d),
            Err(LinAlgError::NotPositiveDefinite)
        ));
    }

    #[test]
    fn test_chol_blocks() {
        let b1 = DMatrix::from_row_slice(2, 2, &[4.0, 2.0, 2.0, 5.0]);
        let b2 = DMatrix::identity(2, 2);
        let mut blocks = vec![b1, b2];
        chol_unblocked_blocks(&mut blocks).unwrap();

        // Block 0: L = [[2, 0], [1, 2]]
        assert_relative_eq!(blocks[0][(0, 0)], 2.0, epsilon = 1e-12);
        assert_relative_eq!(blocks[0][(1, 0)], 1.0, epsilon = 1e-12);
        assert_relative_eq!(blocks[0][(1, 1)], 2.0, epsilon = 1e-12);

        // Block 1: L = I
        assert_relative_eq!(blocks[1][(0, 0)], 1.0, epsilon = 1e-12);
        assert_relative_eq!(blocks[1][(1, 0)], 0.0, epsilon = 1e-12);
        assert_relative_eq!(blocks[1][(1, 1)], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_chol_not_square() {
        let mut a = DMatrix::zeros(2, 3);
        let result = chol_unblocked(&mut a);
        assert!(result.is_err());
    }

    #[test]
    fn test_chol_empty() {
        let mut a = DMatrix::zeros(0, 0);
        chol_unblocked(&mut a).unwrap();
    }
}
