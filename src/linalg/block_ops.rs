//! Block matrix operations for the blocked Cholesky factor.
//!
//! These operations are used in `updateL!` for scaling blocks by Λ
//! and inflating diagonal blocks with the identity.

use nalgebra::DMatrix;

/// Copy and scale a diagonal block: `L_jj = Λ_j' * A_jj * Λ_j + I`
///
/// For a scalar random effect (lambda is 1×1), this simplifies to
/// `L_jj[i] = λ² * A_jj[i] + 1`.
pub fn copy_scale_inflate(
    l: &mut DMatrix<f64>,
    a: &DMatrix<f64>,
    lambda: &DMatrix<f64>,
) {
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
pub fn copy_rmul_lambda(
    l: &mut DMatrix<f64>,
    a: &DMatrix<f64>,
    lambda_j: &DMatrix<f64>,
) {
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
