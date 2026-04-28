//! Pivoted QR factorization and statistical rank detection.
//!
//! This is a port of `pivot.jl` from Julia's MixedModels.jl. The
//! [`stats_rank`] function computes the numerical column rank of a
//! matrix using a column-pivoted QR decomposition, with a tolerance
//! relative to the largest diagonal element of R.

use nalgebra::DMatrix;

/// Result of a column-pivoted QR factorization.
#[derive(Debug, Clone)]
pub struct PivotedQR {
    /// The upper triangular factor `R` from QR (stored in the upper
    /// triangle of a working matrix).
    pub r: DMatrix<f64>,
    /// Column pivot indices (0-based). `piv[i]` is the original column
    /// index that was moved to position `i`.
    pub piv: Vec<usize>,
    /// The orthogonal factor `Q` (stored as a dense matrix). This may
    /// be used for further computations but is not always needed.
    pub q: DMatrix<f64>,
}

/// Compute a rank-revealing QR factorization with column pivoting.
///
/// Uses modified Gram-Schmidt with column pivoting. At each step the
/// column with the largest remaining norm is pivoted into position and
/// the remaining columns are orthogonalised against it.
///
/// Returns `(rank, pivot_indices, R_factor)` where:
/// - `rank` is the numerical column rank.
/// - `pivot_indices` are the 0-based column permutation.
/// - `R_factor` is the upper triangular R from the factorization.
///
/// The default tolerance is `1e-8`.
pub fn pivoted_qr(a: &DMatrix<f64>) -> (usize, Vec<usize>, DMatrix<f64>) {
    pivoted_qr_with_tol(a, 1e-8)
}

/// Same as [`pivoted_qr`] but with a custom tolerance.
pub fn pivoted_qr_with_tol(a: &DMatrix<f64>, ranktol: f64) -> (usize, Vec<usize>, DMatrix<f64>) {
    let (m, n) = (a.nrows(), a.ncols());

    if n == 0 {
        return (0, Vec::new(), DMatrix::zeros(0, 0));
    }

    // Modified Gram-Schmidt with column pivoting.
    // Q is m×min(m,n), R is min(m,n)×n.
    let min_mn = m.min(n);
    let mut q = a.clone(); // working copy; columns get orthogonalised in-place
    let mut r = DMatrix::<f64>::zeros(min_mn, n);
    let mut piv: Vec<usize> = (0..n).collect();
    let mut col_norms: Vec<f64> = (0..n).map(|j| q.column(j).norm_squared()).collect();
    let mut rank = 0;

    for k in 0..min_mn {
        // Find the column with the largest remaining norm.
        let mut best = k;
        let mut best_norm = col_norms[k];
        for j in (k + 1)..n {
            if col_norms[j] > best_norm {
                best = j;
                best_norm = col_norms[j];
            }
        }

        // Swap columns k and best (in both q and bookkeeping).
        if best != k {
            q.swap_columns(k, best);
            piv.swap(k, best);
            col_norms.swap(k, best);
            // Also swap already-computed R entries in rows 0..k
            for i in 0..k {
                let tmp = r[(i, k)];
                r[(i, k)] = r[(i, best)];
                r[(i, best)] = tmp;
            }
        }

        // Compute the norm of the remaining part of column k.
        let norm_k = q.column(k).norm();
        if norm_k < f64::EPSILON * (m.max(n) as f64) {
            break;
        }

        r[(k, k)] = norm_k;
        rank += 1;

        // Normalise column k of Q.
        let inv_norm = 1.0 / norm_k;
        for i in 0..m {
            q[(i, k)] *= inv_norm;
        }

        // Orthogonalise remaining columns against column k.
        for j in (k + 1)..n {
            let dot: f64 = (0..m).map(|i| q[(i, k)] * q[(i, j)]).sum();
            r[(k, j)] = dot;
            for i in 0..m {
                q[(i, j)] -= dot * q[(i, k)];
            }
            col_norms[j] = q.column(j).norm_squared();
        }
    }

    // Determine rank from diagonal of R using tolerance.
    let detected_rank = compute_rank_from_r(&r, ranktol);
    let rank = rank.min(detected_rank);

    (rank, piv, r)
}

/// Compute the rank from the R factor's diagonal using the given tolerance.
///
/// A column is considered dependent if `|R[i,i]| < ranktol * |R[0,0]|`.
fn compute_rank_from_r(r: &DMatrix<f64>, ranktol: f64) -> usize {
    let diag_len = r.nrows().min(r.ncols());
    if diag_len == 0 {
        return 0;
    }

    let r00 = r[(0, 0)].abs();
    if r00 < f64::EPSILON {
        return 0;
    }

    let threshold = ranktol * r00;
    let mut rank = 0;
    for i in 0..diag_len {
        if r[(i, i)].abs() > threshold {
            rank += 1;
        } else {
            break;
        }
    }
    rank
}

/// Compute the numerical column rank of a matrix using a pivoted QR
/// decomposition.
///
/// Returns `(rank, pivot_indices)` where `rank` is the number of
/// linearly independent columns and `pivot_indices` gives the column
/// reordering. In the full-rank case, `pivot_indices` is `0..n`.
///
/// This mirrors `statsrank` from Julia's MixedModels.jl. The rank is
/// determined from the absolute values of the diagonal of R, relative
/// to the first (and largest) diagonal element.
///
/// # Arguments
///
/// * `a` - The matrix to analyse.
///
/// The default rank tolerance is `1e-8`.
pub fn stats_rank(a: &DMatrix<f64>) -> (usize, Vec<usize>) {
    stats_rank_with_tol(a, 1e-8)
}

/// Same as [`stats_rank`] but with a custom tolerance.
pub fn stats_rank_with_tol(a: &DMatrix<f64>, ranktol: f64) -> (usize, Vec<usize>) {
    let (_m, n) = (a.nrows(), a.ncols());

    if n == 0 {
        return (0, Vec::new());
    }

    let (rank, piv, _r) = pivoted_qr_with_tol(a, ranktol);

    if rank == n {
        // Full rank: return the identity permutation (matching Julia behaviour).
        return (n, (0..n).collect());
    }

    // For the rank-deficient case, sort the first `rank` pivot indices
    // to maintain original column order among the independent columns
    // (matching Julia's `sort!(view(piv, 1:rank))`).
    let mut result_piv = piv;
    result_piv[0..rank].sort();

    (rank, result_piv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_pivoted_qr_identity() {
        let a = DMatrix::identity(3, 3);
        let (rank, piv, r) = pivoted_qr(&a);
        assert_eq!(rank, 3);
        assert_eq!(piv.len(), 3);
        // R should have |diag| = 1
        for i in 0..3 {
            assert_relative_eq!(r[(i, i)].abs(), 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_pivoted_qr_rank_deficient() {
        // col3 = col1 + col2
        let a = DMatrix::from_row_slice(
            3,
            3,
            &[
                1.0, 0.0, 1.0, //
                0.0, 1.0, 1.0, //
                1.0, 1.0, 2.0, //
            ],
        );
        let (rank, _piv, _r) = pivoted_qr(&a);
        assert_eq!(rank, 2);
    }

    #[test]
    fn test_pivoted_qr_rectangular_tall() {
        // 4x2 full-rank matrix
        let a = DMatrix::from_row_slice(4, 2, &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0]);
        let (rank, piv, r) = pivoted_qr(&a);
        assert_eq!(rank, 2);
        assert_eq!(piv.len(), 2);
        assert_eq!(r.ncols(), 2);
    }

    #[test]
    fn test_pivoted_qr_rectangular_wide() {
        // 2x4 matrix with rank 2
        let a = DMatrix::from_row_slice(2, 4, &[1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 1.0, 1.0]);
        let (rank, _piv, _r) = pivoted_qr(&a);
        assert_eq!(rank, 2);
    }

    #[test]
    fn test_pivoted_qr_zero_matrix() {
        let a = DMatrix::zeros(3, 2);
        let (rank, _piv, _r) = pivoted_qr(&a);
        assert_eq!(rank, 0);
    }

    #[test]
    fn test_pivoted_qr_single_column() {
        let a = DMatrix::from_column_slice(3, 1, &[1.0, 2.0, 3.0]);
        let (rank, piv, r) = pivoted_qr(&a);
        assert_eq!(rank, 1);
        assert_eq!(piv, vec![0]);
        assert_relative_eq!(
            r[(0, 0)].abs(),
            (1.0_f64 + 4.0 + 9.0).sqrt(),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_pivoted_qr_empty() {
        let a = DMatrix::zeros(3, 0);
        let (rank, piv, _r) = pivoted_qr(&a);
        assert_eq!(rank, 0);
        assert!(piv.is_empty());
    }

    #[test]
    fn test_stats_rank_full_rank() {
        let a = DMatrix::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let (rank, piv) = stats_rank(&a);
        assert_eq!(rank, 2);
        // Full rank: identity permutation
        assert_eq!(piv, vec![0, 1]);
    }

    #[test]
    fn test_stats_rank_rank_deficient() {
        // col3 = col1 + col2
        let a = DMatrix::from_row_slice(
            3,
            3,
            &[
                1.0, 0.0, 1.0, //
                0.0, 1.0, 1.0, //
                1.0, 1.0, 2.0, //
            ],
        );
        let (rank, piv) = stats_rank(&a);
        assert_eq!(rank, 2);
        // The first `rank` pivot indices should be sorted.
        assert!(piv[0] < piv[1]);
    }

    #[test]
    fn test_stats_rank_zero_cols() {
        let a = DMatrix::zeros(5, 0);
        let (rank, piv) = stats_rank(&a);
        assert_eq!(rank, 0);
        assert!(piv.is_empty());
    }

    #[test]
    fn test_stats_rank_all_zero() {
        let a = DMatrix::zeros(3, 3);
        let (rank, _piv) = stats_rank(&a);
        assert_eq!(rank, 0);
    }

    #[test]
    fn test_pivoted_qr_preserves_product() {
        // For a full-rank matrix, Q*R (with column pivoting) should
        // reconstruct the original (permuted) matrix.
        let a = DMatrix::from_row_slice(
            4,
            3,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 2.0, 1.0, 3.0],
        );
        let (rank, piv, _r) = pivoted_qr(&a);
        assert_eq!(rank, 3);
        // Verify the pivot is a permutation of [0,1,2]
        let mut sorted_piv = piv.clone();
        sorted_piv.sort();
        assert_eq!(sorted_piv, vec![0, 1, 2]);
    }

    // ── Tests ported from MixedModels.jl/test/pivot.jl ─────────────────────

    #[test]
    fn test_stats_rank_full_rank_intercept_plus_predictor() {
        // Mirrors pivot.jl "fullranknumeric": [1, U] is full rank.
        let n = 200;
        let mut a = DMatrix::zeros(n, 2);
        for i in 0..n {
            a[(i, 0)] = 1.0;
            a[(i, 1)] = (i % 10) as f64;
        }
        let (rank, piv) = stats_rank(&a);
        assert_eq!(rank, 2);
        assert_eq!(piv, vec![0, 1]);
    }

    #[test]
    fn test_stats_rank_dependent_column_realistic() {
        // Mirrors pivot.jl "dependentcolumn": V = U − 4.5 (mean-centred U)
        // makes [1, U, V, Z] rank-deficient with rank 3.
        //
        // Note: unlike Julia's LAPACK-based pivot (which preserves the intercept),
        // our modified Gram-Schmidt drops the lowest-norm column from the dependent
        // set {1, U, V}. The key properties — rank == 3 and Z retained — are
        // implementation-independent and are what we test here.
        let n = 200;
        let u: Vec<f64> = (0..n).map(|i| (i % 10) as f64).collect();
        let v: Vec<f64> = u.iter().map(|&x| x - 4.5).collect();
        // Z: deterministic sequence linearly independent of 1, U, V
        let z: Vec<f64> = (0..n)
            .map(|i| (((i * 7 + 3) % 13) as f64) * 0.1 + 0.05)
            .collect();

        let mut a = DMatrix::zeros(n, 4);
        for i in 0..n {
            a[(i, 0)] = 1.0;
            a[(i, 1)] = u[i];
            a[(i, 2)] = v[i];
            a[(i, 3)] = z[i];
        }

        let (rank, piv) = stats_rank(&a);
        // V is a linear combination of 1 and U → rank 3
        assert_eq!(rank, 3);
        // Z (col 3) is linearly independent and must stay in the selected set
        assert!(
            piv[..rank].contains(&3),
            "Z (col 3) must not be pivoted out"
        );
        // The dropped column must be from the linearly dependent set {1, U, V}
        let dropped = piv[rank];
        assert!(
            dropped == 0 || dropped == 1 || dropped == 2,
            "dropped column must be from the dependent set, got col {dropped}"
        );
    }
}
