//! Pivoted QR factorization and statistical rank detection.
//!
//! This is a port of `pivot.jl` from Julia's MixedModels.jl. The
//! [`stats_rank`] function computes the numerical column rank of a
//! matrix using a column-pivoted QR decomposition, with a tolerance
//! relative to the largest diagonal element of R.
//!
//! The factorization is Householder QR with Businger-Golub column
//! pivoting, matching the LAPACK `xGEQP3`/`xLAQPS` algorithm that Julia
//! uses through `LinearAlgebra.qr(x, ColumnNorm())`. Modified
//! Gram-Schmidt was previously used here; it loses orthogonality on
//! near-rank-deficient designs and selected a *different* set of kept
//! columns than LAPACK, which broke cross-language parity for
//! rank-deficient fixed-effects designs. Householder reflectors with
//! the LAPACK partial-norm downdating recurrence (including the
//! reorthogonalization safeguard) restore that parity.

use nalgebra::DMatrix;

/// Compute a rank-revealing QR factorization with column pivoting.
///
/// Uses Householder reflectors with Businger-Golub column pivoting
/// (LAPACK `xGEQP3` semantics): at each step the column with the
/// largest remaining 2-norm is pivoted into position, a Householder
/// reflector eliminates the sub-diagonal entries, and the trailing
/// partial column norms are updated with the LAPACK downdating
/// recurrence (recomputed exactly when cancellation is detected).
///
/// Returns `(rank, pivot_indices, R_factor)` where:
/// - `rank` is the numerical column rank (from the R diagonal).
/// - `pivot_indices` is the full 0-based column permutation (length `n`).
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

    let min_mn = m.min(n);

    // `work` holds the matrix being reduced. Householder vectors are
    // written into the sub-diagonal part but we never reconstruct Q;
    // only the upper-triangular R is read back.
    let mut work = a.clone();
    let mut piv: Vec<usize> = (0..n).collect();

    // LAPACK xLAQPS norm bookkeeping:
    //   vn1[j] = current partial 2-norm of the active part of column j
    //   vn2[j] = reference copy used by the cancellation safeguard
    let mut vn1: Vec<f64> = (0..n).map(|j| work.column(j).norm()).collect();
    let mut vn2 = vn1.clone();
    let tol3z = f64::EPSILON.sqrt();

    let mut r = DMatrix::<f64>::zeros(min_mn, n);

    for k in 0..min_mn {
        // --- Businger-Golub pivot: largest remaining partial norm. ---
        let mut best = k;
        let mut best_norm = vn1[k];
        for j in (k + 1)..n {
            if vn1[j] > best_norm {
                best = j;
                best_norm = vn1[j];
            }
        }
        if best != k {
            work.swap_columns(k, best);
            piv.swap(k, best);
            vn1.swap(k, best);
            vn2.swap(k, best);
            for i in 0..k {
                r.swap((i, k), (i, best));
            }
        }

        // --- Householder reflector for work[k.., k] (LAPACK dlarfg). ---
        let alpha = work[(k, k)];
        let mut xnorm_sq = 0.0;
        for i in (k + 1)..m {
            xnorm_sq += work[(i, k)] * work[(i, k)];
        }

        let (beta, tau) = if xnorm_sq == 0.0 {
            // Column already in upper-triangular form: no reflection.
            (alpha, 0.0)
        } else {
            let norm = (alpha * alpha + xnorm_sq).sqrt();
            let beta = if alpha >= 0.0 { -norm } else { norm };
            let tau = (beta - alpha) / beta;
            let inv = 1.0 / (alpha - beta);
            for i in (k + 1)..m {
                work[(i, k)] *= inv;
            }
            (beta, tau)
        };

        r[(k, k)] = beta;

        // --- Apply reflector to trailing columns and write R row k. ---
        for j in (k + 1)..n {
            if tau != 0.0 {
                let mut dot = work[(k, j)]; // implicit v[k] == 1
                for i in (k + 1)..m {
                    dot += work[(i, k)] * work[(i, j)];
                }
                let w = tau * dot;
                work[(k, j)] -= w;
                for i in (k + 1)..m {
                    work[(i, j)] -= w * work[(i, k)];
                }
            }
            r[(k, j)] = work[(k, j)];

            // --- LAPACK partial-norm downdate with safeguard. ---
            if vn1[j] != 0.0 {
                let mut temp = (work[(k, j)].abs() / vn1[j]).powi(2);
                temp = (1.0 - temp).max(0.0);
                let ratio = vn1[j] / vn2[j];
                let temp2 = temp * ratio * ratio;
                if temp2 <= tol3z {
                    // Cancellation: recompute the exact trailing norm.
                    let mut s = 0.0;
                    for i in (k + 1)..m {
                        s += work[(i, j)] * work[(i, j)];
                    }
                    let exact = s.sqrt();
                    vn1[j] = exact;
                    vn2[j] = exact;
                } else {
                    vn1[j] *= temp.sqrt();
                }
            }
        }
    }

    let rank = compute_rank_from_r(&r, ranktol);
    (rank, piv, r)
}

/// Compute the rank from the R factor's diagonal using the given tolerance.
///
/// A column is considered dependent if `|R[i,i]| <= ranktol * |R[0,0]|`.
/// With Businger-Golub pivoting the diagonal magnitudes are
/// non-increasing, so this matches Julia's
/// `searchsortedlast(abs.(diag(R)), fdv*ranktol; rev=true)`.
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
/// This mirrors `statsrank` from Julia's MixedModels.jl, including the
/// intercept-preservation trick: when the first column is the all-ones
/// intercept and column pivoting would otherwise move it out of leading
/// position, the column is temporarily inflated and the factorization
/// re-run so the intercept stays in the retained set (matching LAPACK +
/// the Julia reference). The rank is determined from the absolute
/// values of the diagonal of R, relative to the first (and largest)
/// diagonal element.
///
/// The default rank tolerance is `1e-8`.
pub fn stats_rank(a: &DMatrix<f64>) -> (usize, Vec<usize>) {
    stats_rank_with_tol(a, 1e-8)
}

/// Same as [`stats_rank`] but with a custom tolerance.
pub fn stats_rank_with_tol(a: &DMatrix<f64>, ranktol: f64) -> (usize, Vec<usize>) {
    let (m, n) = (a.nrows(), a.ncols());

    if n == 0 {
        return (0, Vec::new());
    }

    if let Some(full_rank) = quick_full_rank_identity_pivot(a, ranktol) {
        return full_rank;
    }

    let (_rank, piv, r) = pivoted_qr_with_tol(a, ranktol);

    let diag_len = r.nrows().min(r.ncols());
    let dvec: Vec<f64> = (0..diag_len).map(|i| r[(i, i)].abs()).collect();
    let fdv = dvec.first().copied().unwrap_or(0.0);
    let cmp = fdv * ranktol;

    // Full rank (and the diagonal is long enough to cover every column):
    // identity permutation, matching Julia's `collect(axes(x, 2))`.
    if diag_len == n && dvec.last().map(|&d| d > cmp).unwrap_or(false) {
        return (n, (0..n).collect());
    }

    // Rank = count of diagonal entries above the relative threshold.
    let rank = dvec.iter().filter(|&&d| d > cmp).count();
    if rank == n {
        return (n, (0..n).collect());
    }

    let mut piv = piv;

    // Intercept-preservation: if the first column is all-ones and the
    // pivot moved it out of the leading slot, inflate it and re-run so
    // LAPACK keeps it (Julia pivot.jl lines 27-34).
    let first_col_all_ones = m > 0 && (0..m).all(|i| a[(i, 0)] == 1.0);
    if first_col_all_ones && piv.first() != Some(&0) {
        let mut inflated = a.clone();
        let scale = (fdv + 1.0) / (m as f64).sqrt();
        for i in 0..m {
            inflated[(i, 0)] = scale;
        }
        let (_r2, piv2, _rr2) = pivoted_qr_with_tol(&inflated, ranktol);
        piv = piv2;
    }

    // Maintain original column order among the linearly independent
    // columns (Julia's `sort!(view(piv, 1:rank))`).
    piv[0..rank].sort_unstable();

    (rank, piv)
}

fn quick_full_rank_identity_pivot(a: &DMatrix<f64>, ranktol: f64) -> Option<(usize, Vec<usize>)> {
    let (m, n) = (a.nrows(), a.ncols());
    if m == 0 || n == 0 || n > 2 {
        return None;
    }

    if n == 1 {
        let norm = a.column(0).norm();
        return (norm > f64::EPSILON).then(|| (1, vec![0]));
    }

    let mut g00 = 0.0;
    let mut g01 = 0.0;
    let mut g11 = 0.0;
    for row in 0..m {
        let x0 = a[(row, 0)];
        let x1 = a[(row, 1)];
        g00 += x0 * x0;
        g01 += x0 * x1;
        g11 += x1 * x1;
    }

    let trace = g00 + g11;
    if trace <= f64::EPSILON {
        return None;
    }
    let diff = g00 - g11;
    let discriminant = (diff * diff + 4.0 * g01 * g01).sqrt();
    let lambda_max = 0.5 * (trace + discriminant);
    let lambda_min = (0.5 * (trace - discriminant)).max(0.0);
    if lambda_max <= f64::EPSILON {
        return None;
    }

    let sigma_max = lambda_max.sqrt();
    let sigma_min = lambda_min.sqrt();
    let full_rank_margin = (ranktol * sigma_max * 100.0).max(f64::EPSILON.sqrt());
    (sigma_min > full_rank_margin).then(|| (2, vec![0, 1]))
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
    fn test_pivoted_qr_factorization_reconstructs_permuted_matrix() {
        // R must satisfy: for the retained columns, |R[k,k]| equals the
        // norm of the residual after projecting out earlier columns, and
        // the diagonal is non-increasing (Businger-Golub property).
        let a = DMatrix::from_row_slice(
            4,
            3,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 2.0, 1.0, 3.0],
        );
        let (rank, piv, r) = pivoted_qr(&a);
        assert_eq!(rank, 3);
        // Diagonal magnitudes are non-increasing under column pivoting.
        for i in 0..2 {
            assert!(
                r[(i, i)].abs() >= r[(i + 1, i + 1)].abs() - 1e-12,
                "pivoted R diagonal must be non-increasing"
            );
        }
        // The Frobenius norm of R equals that of A (orthogonal Q).
        let a_fro = a.norm();
        let mut r_fro = 0.0;
        for i in 0..r.nrows() {
            for j in i..r.ncols() {
                r_fro += r[(i, j)] * r[(i, j)];
            }
        }
        assert_relative_eq!(r_fro.sqrt(), a_fro, epsilon = 1e-9);
        let mut sorted_piv = piv.clone();
        sorted_piv.sort();
        assert_eq!(sorted_piv, vec![0, 1, 2]);
    }

    #[test]
    fn test_pivoted_qr_rectangular_tall() {
        let a = DMatrix::from_row_slice(4, 2, &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0]);
        let (rank, piv, r) = pivoted_qr(&a);
        assert_eq!(rank, 2);
        assert_eq!(piv.len(), 2);
        assert_eq!(r.ncols(), 2);
    }

    #[test]
    fn test_pivoted_qr_rectangular_wide() {
        let a = DMatrix::from_row_slice(2, 4, &[1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 1.0, 1.0]);
        let (rank, piv, _r) = pivoted_qr(&a);
        assert_eq!(rank, 2);
        assert_eq!(piv.len(), 4);
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

    // ── Tests ported from MixedModels.jl/test/pivot.jl ─────────────────────

    #[test]
    fn test_stats_rank_full_rank_intercept_plus_predictor() {
        // pivot.jl "fullranknumeric": [1, U] is full rank, pivot == 1:2.
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
    fn test_stats_rank_dependent_column_matches_lapack_julia() {
        // pivot.jl "dependentcolumn": V = U − 4.5 (mean-centred U) makes
        // [1, U, V, Z] rank-deficient with rank 3. Julia/LAPACK asserts:
        //   pivot[1] == 1   (intercept retained, first position)
        //   pivot[3] == 4   (Z, col index 3 here, not pivoted out)
        //   pivot[4] in {2,3} (either U or V dropped)
        let n = 200;
        let u: Vec<f64> = (0..n).map(|i| (i % 10) as f64).collect();
        let v: Vec<f64> = u.iter().map(|&x| x - 4.5).collect();
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
        assert_eq!(rank, 3, "V is a linear combo of 1 and U → rank 3");
        // Intercept retained in the leading position (LAPACK parity that
        // the old MGS port could not satisfy).
        assert_eq!(piv[0], 0, "intercept (col 0) must remain first");
        // Z (col 3) is independent and must stay in the retained set.
        assert!(
            piv[..rank].contains(&3),
            "Z (col 3) must not be pivoted out, got piv={piv:?}"
        );
        // The dropped column is U or V (the dependent pair).
        let dropped = piv[rank];
        assert!(
            dropped == 1 || dropped == 2,
            "dropped column must be U or V, got col {dropped}"
        );
        // Independent columns keep ascending (original) order.
        assert!(piv[..rank].windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn test_stats_rank_intercept_preserved_when_pivot_would_move_it() {
        // The intercept column has a smaller raw norm than a large-scale
        // predictor, so naive Businger-Golub pivots it out of leading
        // position. The intercept-preservation trick must restore it.
        let n = 50;
        let mut a = DMatrix::zeros(n, 3);
        for i in 0..n {
            a[(i, 0)] = 1.0; // intercept (norm sqrt(50))
            a[(i, 1)] = 100.0 * ((i % 7) as f64 + 1.0); // large-scale predictor
            a[(i, 2)] = (i % 3) as f64 - 1.0;
        }
        let (rank, piv) = stats_rank(&a);
        assert_eq!(rank, 3);
        assert_eq!(piv, vec![0, 1, 2], "intercept must be preserved first");
    }

    #[test]
    fn test_stats_rank_qr_missing_cells_relative_order() {
        // pivot.jl "qr missing cells": independent columns preserve their
        // relative (sorted) order and the rank is detected correctly on a
        // rank-deficient categorical-style design.
        let n = 60;
        // Build [1, A, B, A:B-collinear] where the last column duplicates A.
        let mut a = DMatrix::zeros(n, 4);
        for i in 0..n {
            a[(i, 0)] = 1.0;
            a[(i, 1)] = (i % 5) as f64;
            a[(i, 2)] = (i % 4) as f64;
            a[(i, 3)] = (i % 5) as f64; // exact duplicate of col 1
        }
        let (rank, piv) = stats_rank(&a);
        assert_eq!(rank, 3);
        let kept = &piv[..rank];
        assert!(
            kept.windows(2).all(|w| w[0] < w[1]),
            "independent columns must keep relative order, got {kept:?}"
        );
    }
}
