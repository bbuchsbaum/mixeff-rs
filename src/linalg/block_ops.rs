//! Block matrix operations for the blocked Cholesky factor.
//!
//! These operations are used in `updateL!` for scaling blocks by Λ
//! and inflating diagonal blocks with the identity.

use nalgebra::DMatrix;

/// Copy and scale a diagonal block: `L_jj = Λ_j' * A_jj * Λ_j + I`
///
/// For a scalar random effect (lambda is 1×1), this simplifies to
/// `L_jj[i] = λ² * A_jj[i] + 1`.
pub fn copy_scale_inflate(l: &mut DMatrix<f64>, a: &DMatrix<f64>, lambda: &DMatrix<f64>) {
    let temp = a * lambda;
    *l = lambda.transpose() * &temp;
    let n = l.nrows().min(l.ncols());
    for i in 0..n {
        l[(i, i)] += 1.0;
    }
}

/// Scale an off-diagonal block: `L_ij = Λ_i' * A_ij * Λ_j`
pub fn copy_scale_offdiag(
    l: &mut DMatrix<f64>,
    a: &DMatrix<f64>,
    lambda_i: &DMatrix<f64>,
    lambda_j: &DMatrix<f64>,
) {
    *l = lambda_i.transpose() * a * lambda_j;
}

/// Right-multiply by Λ: `L_kj = A_kj * Λ_j`
pub fn copy_rmul_lambda(l: &mut DMatrix<f64>, a: &DMatrix<f64>, lambda_j: &DMatrix<f64>) {
    *l = a * lambda_j;
}

/// Left-multiply by Λ': `B = Λ' * B`
pub fn lmul_lambda_transpose(lambda: &DMatrix<f64>, b: &mut DMatrix<f64>) {
    let result = lambda.transpose() * &*b;
    b.copy_from(&result);
}

/// Right-multiply by Λ: `A = A * Λ`
pub fn rmul_lambda(a: &mut DMatrix<f64>, lambda: &DMatrix<f64>) {
    let result = &*a * lambda;
    a.copy_from(&result);
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_copy_scale_inflate_identity_lambda() {
        // With λ = I, result = A + I
        let mut l = DMatrix::zeros(2, 2);
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        copy_scale_inflate(&mut l, &a, &DMatrix::identity(2, 2));
        assert_relative_eq!(l[(0, 0)], 2.0, epsilon = 1e-12); // 1 + 1
        assert_relative_eq!(l[(0, 1)], 2.0, epsilon = 1e-12);
        assert_relative_eq!(l[(1, 0)], 3.0, epsilon = 1e-12);
        assert_relative_eq!(l[(1, 1)], 5.0, epsilon = 1e-12); // 4 + 1
    }

    #[test]
    fn test_copy_scale_inflate_scalar_lambda() {
        // λ = [[2.0]] → L = 2*4*2 + 1 = 17
        let mut l = DMatrix::zeros(1, 1);
        let a = DMatrix::from_element(1, 1, 4.0);
        let lambda = DMatrix::from_element(1, 1, 2.0);
        copy_scale_inflate(&mut l, &a, &lambda);
        assert_relative_eq!(l[(0, 0)], 17.0, epsilon = 1e-12);
    }

    #[test]
    fn test_copy_scale_inflate_zero_lambda() {
        // λ = 0 → L = 0'*A*0 + I = I
        let mut l = DMatrix::zeros(2, 2);
        let a = DMatrix::from_row_slice(2, 2, &[5.0, 6.0, 7.0, 8.0]);
        copy_scale_inflate(&mut l, &a, &DMatrix::zeros(2, 2));
        assert_relative_eq!(l[(0, 0)], 1.0, epsilon = 1e-12);
        assert_relative_eq!(l[(0, 1)], 0.0, epsilon = 1e-12);
        assert_relative_eq!(l[(1, 0)], 0.0, epsilon = 1e-12);
        assert_relative_eq!(l[(1, 1)], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_copy_scale_offdiag() {
        // L = λ_i' * A * λ_j where λ_i = diag(2,3), λ_j = diag(1,2)
        let mut l = DMatrix::zeros(2, 2);
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let lambda_i = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 3.0]);
        let lambda_j = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 2.0]);
        copy_scale_offdiag(&mut l, &a, &lambda_i, &lambda_j);
        // diag(2,3)' * [[1,2],[3,4]] * diag(1,2) = [[2,4],[9,12]] * diag(1,2) = [[2,8],[9,24]]
        assert_relative_eq!(l[(0, 0)], 2.0, epsilon = 1e-12);
        assert_relative_eq!(l[(0, 1)], 8.0, epsilon = 1e-12);
        assert_relative_eq!(l[(1, 0)], 9.0, epsilon = 1e-12);
        assert_relative_eq!(l[(1, 1)], 24.0, epsilon = 1e-12);
    }

    #[test]
    fn test_copy_rmul_lambda() {
        // L = A * λ
        let mut l = DMatrix::zeros(2, 2);
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let lambda = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        copy_rmul_lambda(&mut l, &a, &lambda);
        // [[1,2],[3,4]] * [[1,2],[3,4]] = [[7,10],[15,22]]
        assert_relative_eq!(l[(0, 0)], 7.0, epsilon = 1e-12);
        assert_relative_eq!(l[(0, 1)], 10.0, epsilon = 1e-12);
        assert_relative_eq!(l[(1, 0)], 15.0, epsilon = 1e-12);
        assert_relative_eq!(l[(1, 1)], 22.0, epsilon = 1e-12);
    }

    #[test]
    fn test_lmul_lambda_transpose() {
        // B ← λ' * B
        let lambda = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let mut b = DMatrix::from_row_slice(2, 2, &[5.0, 6.0, 7.0, 8.0]);
        lmul_lambda_transpose(&lambda, &mut b);
        // [[1,3],[2,4]] * [[5,6],[7,8]] = [[26,30],[38,44]]
        assert_relative_eq!(b[(0, 0)], 26.0, epsilon = 1e-12);
        assert_relative_eq!(b[(0, 1)], 30.0, epsilon = 1e-12);
        assert_relative_eq!(b[(1, 0)], 38.0, epsilon = 1e-12);
        assert_relative_eq!(b[(1, 1)], 44.0, epsilon = 1e-12);
    }

    #[test]
    fn test_lmul_lambda_transpose_identity() {
        // I' * B = B (unchanged)
        let lambda = DMatrix::identity(3, 3);
        let mut b = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b_orig = b.clone();
        lmul_lambda_transpose(&lambda, &mut b);
        assert_relative_eq!(b, b_orig, epsilon = 1e-12);
    }

    #[test]
    fn test_rmul_lambda() {
        // A ← A * λ
        let lambda = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let mut a = DMatrix::from_row_slice(2, 2, &[5.0, 6.0, 7.0, 8.0]);
        rmul_lambda(&mut a, &lambda);
        // [[5,6],[7,8]] * [[1,2],[3,4]] = [[23,34],[31,46]]
        assert_relative_eq!(a[(0, 0)], 23.0, epsilon = 1e-12);
        assert_relative_eq!(a[(0, 1)], 34.0, epsilon = 1e-12);
        assert_relative_eq!(a[(1, 0)], 31.0, epsilon = 1e-12);
        assert_relative_eq!(a[(1, 1)], 46.0, epsilon = 1e-12);
    }

    #[test]
    fn test_rmul_lambda_identity() {
        // A * I = A (unchanged)
        let mut a = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let a_orig = a.clone();
        rmul_lambda(&mut a, &DMatrix::identity(3, 3));
        assert_relative_eq!(a, a_orig, epsilon = 1e-12);
    }
}
