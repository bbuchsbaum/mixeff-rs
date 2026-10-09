//! Cache-friendly dense kernels for the blocked PLS factorization.
//!
//! The dense arms of the blocked Cholesky (`cholesky_block_with_tolerance`),
//! the off-diagonal triangular solve (`rdiv_lower_transpose`) and the
//! diagonal-block downdate (`rank_k_downdate`) used to be textbook triple
//! loops whose innermost index walked a *row* of a column-major matrix
//! (stride `n`). On the large dense blocks of crossed designs (thousands of
//! second-factor levels) that made every objective evaluation memory bound:
//! a 2000 × 2000 block took ~7.5 s to factor.
//!
//! The kernels here keep the same arithmetic but reorder it:
//!
//! * Up to [`DENSE_BLOCKED_MIN_DIM`] the Cholesky and the triangular solve
//!   run a left-looking *column* (axpy) sweep. Every output element is
//!   accumulated over the same `k` sequence and with the same products as
//!   the old dot-product loops, so the result is bit-identical to them; only
//!   the traversal is contiguous.
//! * Above it they switch to a blocked algorithm: panels of
//!   [`DENSE_PANEL_WIDTH`] columns are factored (solved) with the column
//!   sweep, and the trailing (leading) update is a gemm through
//!   nalgebra/matrixmultiply. The sums are associated differently, so the
//!   results agree with the unblocked kernel to rounding (~1e-13 relative on
//!   well-conditioned input), not bit for bit.
//!
//! Zero-pivot semantics are unchanged in both regimes: a pivot `s <= 0`
//! with `s >= -tol` zeroes the column (singular random effect), a pivot
//! below `-tol` is a `PosDefException`, and `|L[j,j]| < 1e-30` zeroes the
//! corresponding column of the right-divided block. The strictly upper
//! triangle of a factored block is always returned as exact zeros (the old
//! kernel left the input's upper entries in place for zero-pivot columns).

use nalgebra::{DMatrix, DMatrixView, DMatrixViewMut, Dyn};

use crate::error::{MixedModelError, Result};

use super::blocks::BLOCK_TRIANGULAR_SOLVE_ZERO_TOLERANCE;

/// Blocks with at most this many columns use the unblocked column sweep,
/// which is bit-identical to the historical kernels.
pub(super) const DENSE_BLOCKED_MIN_DIM: usize = 256;

/// Panel width of the blocked kernels.
pub(super) const DENSE_PANEL_WIDTH: usize = 96;

/// Minimum inner dimension at which the symmetric (lower-triangle) blocked
/// downdate is used instead of one full gemm.
const SYRK_MIN_INNER_DIM: usize = 16;

/// In-place lower Cholesky of a dense symmetric block (only the lower
/// triangle is read). See the module docs for the zero-pivot contract.
pub(super) fn dense_cholesky_lower_in_place(mat: &mut DMatrix<f64>, tol: f64) -> Result<()> {
    let n = mat.nrows();
    debug_assert_eq!(n, mat.ncols());
    if n <= DENSE_BLOCKED_MIN_DIM {
        cholesky_panel(mat, 0, n, tol)?;
    } else {
        let mut j0 = 0;
        while j0 < n {
            let j1 = (j0 + DENSE_PANEL_WIDTH).min(n);
            cholesky_panel(mat, j0, j1, tol)?;
            if j1 < n {
                trailing_symmetric_update(mat, j0, j1);
            }
            j0 = j1;
        }
    }
    zero_strict_upper(mat);
    Ok(())
}

/// Left-looking column sweep over columns `j0..j1`, assuming every column
/// `< j0` has already been applied to columns `>= j0` (rows `>= j0`).
fn cholesky_panel(mat: &mut DMatrix<f64>, j0: usize, j1: usize, tol: f64) -> Result<()> {
    let n = mat.nrows();
    let data = mat.as_mut_slice();
    for j in j0..j1 {
        let (left, right) = data.split_at_mut(j * n);
        let col_j = &mut right[j..n];
        for k in j0..j {
            let ljk = left[k * n + j];
            if ljk != 0.0 {
                let col_k = &left[k * n + j..k * n + n];
                for (dst, &src) in col_j.iter_mut().zip(col_k) {
                    *dst -= src * ljk;
                }
            }
        }
        let s = col_j[0];
        if s <= 0.0 {
            if s < -tol {
                return Err(MixedModelError::PosDefException);
            }
            // Zero column (singular random effect).
            col_j.fill(0.0);
            continue;
        }
        let d = s.sqrt();
        col_j[0] = d;
        for value in &mut col_j[1..] {
            *value /= d;
        }
    }
    Ok(())
}

/// `A[j1.., j1..] -= P P'` (lower triangle only) with `P = L[j1.., j0..j1]`.
fn trailing_symmetric_update(mat: &mut DMatrix<f64>, j0: usize, j1: usize) {
    let n = mat.nrows();
    let w = j1 - j0;
    let (left, right) = mat.as_mut_slice().split_at_mut(j1 * n);
    let mut c0 = j1;
    while c0 < n {
        let c1 = (c0 + DENSE_PANEL_WIDTH).min(n);
        // P[c0..n, :]
        let p_rows: DMatrixView<'_, f64, Dyn, Dyn> = DMatrixView::from_slice_with_strides_generic(
            &left[j0 * n + c0..],
            Dyn(n - c0),
            Dyn(w),
            Dyn(1),
            Dyn(n),
        );
        // P[c0..c1, :]'
        let p_cols_t: DMatrixView<'_, f64, Dyn, Dyn> = DMatrixView::from_slice_with_strides_generic(
            &left[j0 * n + c0..],
            Dyn(w),
            Dyn(c1 - c0),
            Dyn(n),
            Dyn(1),
        );
        // A[c0..n, c0..c1]
        let mut target: DMatrixViewMut<'_, f64, Dyn, Dyn> =
            DMatrixViewMut::from_slice_with_strides_generic(
                &mut right[(c0 - j1) * n + c0..],
                Dyn(n - c0),
                Dyn(c1 - c0),
                Dyn(1),
                Dyn(n),
            );
        target.gemm(-1.0, &p_rows, &p_cols_t, 1.0);
        c0 = c1;
    }
}

fn zero_strict_upper(mat: &mut DMatrix<f64>) {
    let n = mat.nrows();
    let data = mat.as_mut_slice();
    for j in 1..n {
        data[j * n..j * n + j].fill(0.0);
    }
}

/// `A = A L^{-T}` for a dense `A` (`m × n`) and dense lower-triangular `L`
/// (`n × n`, only the lower triangle is read). Columns whose pivot
/// `|L[j,j]|` is below the solve tolerance are zeroed.
pub(super) fn dense_rdiv_lower_transpose_in_place(a: &mut DMatrix<f64>, l: &DMatrix<f64>) {
    let n = l.nrows();
    debug_assert_eq!(a.ncols(), n);
    if n <= DENSE_BLOCKED_MIN_DIM || a.nrows() == 0 {
        rdiv_panel(a, l, 0, n);
        return;
    }
    let m = a.nrows();
    let mut j0 = 0;
    while j0 < n {
        let j1 = (j0 + DENSE_PANEL_WIDTH).min(n);
        if j0 > 0 {
            // A[:, j0..j1] -= A[:, 0..j0] * L[j0..j1, 0..j0]'
            let (done, rest) = a.as_mut_slice().split_at_mut(j0 * m);
            let solved: DMatrixView<'_, f64, Dyn, Dyn> =
                DMatrixView::from_slice_with_strides_generic(done, Dyn(m), Dyn(j0), Dyn(1), Dyn(m));
            let l_rows_t: DMatrixView<'_, f64, Dyn, Dyn> =
                DMatrixView::from_slice_with_strides_generic(
                    &l.as_slice()[j0..],
                    Dyn(j0),
                    Dyn(j1 - j0),
                    Dyn(n),
                    Dyn(1),
                );
            let mut panel: DMatrixViewMut<'_, f64, Dyn, Dyn> =
                DMatrixViewMut::from_slice_with_strides_generic(
                    rest,
                    Dyn(m),
                    Dyn(j1 - j0),
                    Dyn(1),
                    Dyn(m),
                );
            panel.gemm(-1.0, &solved, &l_rows_t, 1.0);
        }
        rdiv_panel(a, l, j0, j1);
        j0 = j1;
    }
}

/// Column sweep of `A L^{-T}` over columns `j0..j1`, assuming columns
/// `< j0` of `A` are final and already applied to columns `>= j0`.
fn rdiv_panel(a: &mut DMatrix<f64>, l: &DMatrix<f64>, j0: usize, j1: usize) {
    let m = a.nrows();
    let data = a.as_mut_slice();
    for j in j0..j1 {
        let (left, right) = data.split_at_mut(j * m);
        let col_j = &mut right[..m];
        let ljj = l[(j, j)];
        if ljj.abs() < BLOCK_TRIANGULAR_SOLVE_ZERO_TOLERANCE {
            col_j.fill(0.0);
            continue;
        }
        for k in j0..j {
            let ljk = l[(j, k)];
            if ljk != 0.0 {
                let col_k = &left[k * m..(k + 1) * m];
                for (dst, &src) in col_j.iter_mut().zip(col_k) {
                    *dst -= src * ljk;
                }
            }
        }
        for value in col_j.iter_mut() {
            *value /= ljj;
        }
    }
}

/// Whether [`symmetric_rank_k_downdate_lower`] should replace a full gemm
/// for `C (n × n) -= A A'` with `A` of shape `n × k`.
pub(super) fn use_symmetric_downdate(n: usize, k: usize) -> bool {
    n > DENSE_BLOCKED_MIN_DIM && k >= SYRK_MIN_INNER_DIM
}

/// `C -= A A'` on the lower triangle (diagonal included) of `C` only.
///
/// The strictly upper triangle of `C` is unspecified afterwards (entries
/// inside the diagonal panels are downdated, the rest are left stale): the
/// only consumer of a downdated diagonal block is the dense Cholesky, which
/// reads the lower triangle and returns an exactly-zero upper triangle.
pub(super) fn symmetric_rank_k_downdate_lower(c: &mut DMatrix<f64>, a: &DMatrix<f64>) {
    let n = c.nrows();
    debug_assert_eq!(n, c.ncols());
    debug_assert_eq!(n, a.nrows());
    let k = a.ncols();
    if n == 0 || k == 0 {
        return;
    }
    let a_data = a.as_slice();
    let c_data = c.as_mut_slice();
    let mut c0 = 0;
    while c0 < n {
        let c1 = (c0 + DENSE_PANEL_WIDTH).min(n);
        let a_rows: DMatrixView<'_, f64, Dyn, Dyn> = DMatrixView::from_slice_with_strides_generic(
            &a_data[c0..],
            Dyn(n - c0),
            Dyn(k),
            Dyn(1),
            Dyn(n),
        );
        let a_cols_t: DMatrixView<'_, f64, Dyn, Dyn> = DMatrixView::from_slice_with_strides_generic(
            &a_data[c0..],
            Dyn(k),
            Dyn(c1 - c0),
            Dyn(n),
            Dyn(1),
        );
        let mut target: DMatrixViewMut<'_, f64, Dyn, Dyn> =
            DMatrixViewMut::from_slice_with_strides_generic(
                &mut c_data[c0 * n + c0..],
                Dyn(n - c0),
                Dyn(c1 - c0),
                Dyn(1),
                Dyn(n),
            );
        target.gemm(-1.0, &a_rows, &a_cols_t, 1.0);
        c0 = c1;
    }
}

#[cfg(test)]
pub(super) mod reference {
    //! The historical unblocked kernels, kept verbatim as test oracles.
    use super::*;

    pub(in crate::model::linear) fn old_dense_cholesky(
        mat: &mut DMatrix<f64>,
        tol: f64,
    ) -> Result<()> {
        let n = mat.nrows();
        for j in 0..n {
            let mut s = mat[(j, j)];
            for k in 0..j {
                s -= mat[(j, k)] * mat[(j, k)];
            }
            if s <= 0.0 {
                if s < -tol {
                    return Err(MixedModelError::PosDefException);
                }
                for i in j..n {
                    mat[(i, j)] = 0.0;
                }
                continue;
            }
            mat[(j, j)] = s.sqrt();
            for i in (j + 1)..n {
                let mut s = mat[(i, j)];
                for k in 0..j {
                    s -= mat[(i, k)] * mat[(j, k)];
                }
                mat[(i, j)] = s / mat[(j, j)];
            }
            for i in 0..j {
                mat[(i, j)] = 0.0;
            }
        }
        Ok(())
    }

    pub(in crate::model::linear) fn old_dense_rdiv(
        a_mat: &mut DMatrix<f64>,
        l_dense: &DMatrix<f64>,
    ) {
        let n = l_dense.nrows();
        for j in 0..n {
            if l_dense[(j, j)].abs() < BLOCK_TRIANGULAR_SOLVE_ZERO_TOLERANCE {
                for i in 0..a_mat.nrows() {
                    a_mat[(i, j)] = 0.0;
                }
                continue;
            }
            for i in 0..a_mat.nrows() {
                let mut s = a_mat[(i, j)];
                for k in 0..j {
                    s -= a_mat[(i, k)] * l_dense[(j, k)];
                }
                a_mat[(i, j)] = s / l_dense[(j, j)];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::reference::{old_dense_cholesky, old_dense_rdiv};
    use super::*;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    fn random_spd(n: usize, rank: usize, seed: u64) -> DMatrix<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        let b = DMatrix::<f64>::from_fn(n, rank, |_, _| rng.gen::<f64>() - 0.5);
        let mut a = &b * b.transpose();
        if rank >= n {
            for i in 0..n {
                a[(i, i)] += n as f64 * 0.05;
            }
        }
        a
    }

    /// Symmetric PSD matrix with exactly-zero rows/columns, the structure a
    /// boundary (θ = 0) random effect produces: `Λ'AΛ` with zeroed Λ columns
    /// and no `+ I` inflation for the zeroed coordinates.
    fn spd_with_zero_columns(n: usize, zero_every: usize, seed: u64) -> DMatrix<f64> {
        let mut a = random_spd(n, n, seed);
        for z in (0..n).step_by(zero_every) {
            for i in 0..n {
                a[(z, i)] = 0.0;
                a[(i, z)] = 0.0;
            }
        }
        a
    }

    fn max_rel_lower_diff(a: &DMatrix<f64>, b: &DMatrix<f64>) -> f64 {
        let n = a.nrows();
        let scale = a.iter().fold(0.0_f64, |m, v| m.max(v.abs())).max(1e-300);
        let mut worst = 0.0_f64;
        for j in 0..a.ncols() {
            for i in j..n {
                worst = worst.max((a[(i, j)] - b[(i, j)]).abs() / scale);
            }
        }
        worst
    }

    fn assert_upper_zero(m: &DMatrix<f64>) {
        for j in 0..m.ncols() {
            for i in 0..j.min(m.nrows()) {
                assert_eq!(m[(i, j)], 0.0, "upper entry ({i},{j}) not zeroed");
            }
        }
    }

    #[test]
    fn unblocked_cholesky_is_bit_identical_to_old_kernel() {
        for (n, seed) in [
            (1, 1),
            (7, 2),
            (64, 3),
            (200, 4),
            (DENSE_BLOCKED_MIN_DIM, 5),
        ] {
            for input in [random_spd(n, n, seed), spd_with_zero_columns(n, 5, seed)] {
                let mut new = input.clone();
                let mut old = input.clone();
                dense_cholesky_lower_in_place(&mut new, 1e-10).unwrap();
                old_dense_cholesky(&mut old, 1e-10).unwrap();
                for j in 0..n {
                    for i in j..n {
                        assert_eq!(
                            new[(i, j)].to_bits(),
                            old[(i, j)].to_bits(),
                            "n={n} ({i},{j})"
                        );
                    }
                }
                assert_upper_zero(&new);
            }
        }
    }

    #[test]
    fn blocked_cholesky_matches_old_kernel_on_spd_and_singular_inputs() {
        for (n, seed) in [(257, 11), (300, 12), (513, 13)] {
            let inputs = [
                random_spd(n, n, seed),
                spd_with_zero_columns(n, 7, seed),
                spd_with_zero_columns(n, 2, seed + 100),
            ];
            for input in inputs {
                let mut new = input.clone();
                let mut old = input.clone();
                dense_cholesky_lower_in_place(&mut new, 1e-10).unwrap();
                old_dense_cholesky(&mut old, 1e-10).unwrap();
                let diff = max_rel_lower_diff(&old, &new);
                assert!(diff < 1e-12, "n={n}: max relative diff {diff:e}");
                // Exact zero columns stay exact zeros.
                for j in 0..n {
                    if old[(j, j)] == 0.0 {
                        assert!((j..n).all(|i| new[(i, j)] == 0.0), "column {j}");
                    }
                }
                assert_upper_zero(&new);
            }
        }
    }

    #[test]
    fn blocked_cholesky_reconstructs_rank_deficient_input() {
        // Rank-deficient PSD: trailing pivots are rounding noise, so the two
        // kernels may zero different columns. Both must reconstruct A.
        let n = 300;
        let a = random_spd(n, 40, 21);
        let scale = a.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let mut new = a.clone();
        dense_cholesky_lower_in_place(&mut new, 1e-8).unwrap();
        let rebuilt = &new * new.transpose();
        let err = (&rebuilt - &a).amax() / scale;
        assert!(err < 1e-10, "reconstruction error {err:e}");
    }

    #[test]
    fn cholesky_negative_pivot_beyond_tolerance_is_posdef_exception_in_both_regimes() {
        for n in [10, 400] {
            let mut a = random_spd(n, n, 31);
            let j = n - 3;
            // Make the Schur complement at pivot j clearly negative.
            a[(j, j)] = -10.0 * n as f64;
            let mut new = a.clone();
            let mut old = a.clone();
            assert!(matches!(
                dense_cholesky_lower_in_place(&mut new, 1e-10),
                Err(MixedModelError::PosDefException)
            ));
            assert!(matches!(
                old_dense_cholesky(&mut old, 1e-10),
                Err(MixedModelError::PosDefException)
            ));
        }
    }

    #[test]
    fn rdiv_matches_old_kernel() {
        for (m, n, seed, exact) in [
            (37, 50, 41, true),
            (500, DENSE_BLOCKED_MIN_DIM, 42, true),
            (300, 400, 43, false),
            (1, 700, 44, false),
        ] {
            let mut l = spd_with_zero_columns(n, 9, seed);
            dense_cholesky_lower_in_place(&mut l, 1e-10).unwrap();
            let mut rng = StdRng::seed_from_u64(seed + 7);
            let a = DMatrix::<f64>::from_fn(m, n, |_, _| rng.gen::<f64>() - 0.5);
            let mut new = a.clone();
            let mut old = a.clone();
            dense_rdiv_lower_transpose_in_place(&mut new, &l);
            old_dense_rdiv(&mut old, &l);
            if exact {
                assert!(new
                    .iter()
                    .zip(old.iter())
                    .all(|(x, y)| x.to_bits() == y.to_bits()));
            } else {
                let scale = old.amax().max(1e-300);
                let diff = (&new - &old).amax() / scale;
                assert!(diff < 1e-12, "m={m} n={n}: diff {diff:e}");
            }
        }
    }

    #[test]
    fn symmetric_downdate_matches_full_gemm_on_lower_triangle() {
        let n = 300;
        let mut rng = StdRng::seed_from_u64(51);
        let a = DMatrix::<f64>::from_fn(n, 40, |_, _| rng.gen::<f64>() - 0.5);
        let c = random_spd(n, n, 52);
        let mut full = c.clone();
        full.gemm(-1.0, &a, &a.transpose(), 1.0);
        let mut lower = c.clone();
        symmetric_rank_k_downdate_lower(&mut lower, &a);
        assert!(max_rel_lower_diff(&full, &lower) < 1e-14);
    }
}
