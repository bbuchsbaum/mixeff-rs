//! Blocked A/L Cholesky construction and response-matrix profiling for the
//! LMM penalized-least-squares (PLS) step. Moved verbatim from the former
//! single-file `linear.rs`; see [`super`] for the model driver.

use super::*;

pub(super) const DEFAULT_DENSE_BLOCK_LIMIT_BYTES: u128 = 16 * 1024 * 1024 * 1024;

pub(super) fn dense_block_limit_bytes() -> u128 {
    std::env::var("MIXEDMODELS_MAX_DENSE_BLOCK_BYTES")
        .ok()
        .and_then(|value| value.parse::<u128>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_DENSE_BLOCK_LIMIT_BYTES)
}

pub(super) fn dense_block_bytes(nrows: usize, ncols: usize) -> u128 {
    (nrows as u128)
        .saturating_mul(ncols as u128)
        .saturating_mul(std::mem::size_of::<f64>() as u128)
}

pub(super) fn ensure_dense_block_within_limit(
    nrows: usize,
    ncols: usize,
    context: impl Into<String>,
) -> Result<()> {
    ensure_dense_block_within_explicit_limit(nrows, ncols, context, dense_block_limit_bytes())
}

pub(super) fn ensure_dense_block_within_explicit_limit(
    nrows: usize,
    ncols: usize,
    context: impl Into<String>,
    limit: u128,
) -> Result<()> {
    let bytes = dense_block_bytes(nrows, ncols);
    if bytes > limit {
        return Err(MixedModelError::ProblemTooLarge(format!(
            "{} would require a dense {} x {} f64 block ({:.2} GiB), above the configured limit ({:.2} GiB). \
             For large partially crossed random effects, use a more storage-aware formulation or raise MIXEDMODELS_MAX_DENSE_BLOCK_BYTES only if this allocation is intentional.",
            context.into(),
            nrows,
            ncols,
            bytes as f64 / 1024.0_f64.powi(3),
            limit as f64 / 1024.0_f64.powi(3)
        )));
    }
    Ok(())
}

pub(super) fn validate_dense_block_plan(
    reterms: &[ReMat],
    fixed_response_cols: usize,
) -> Result<()> {
    for i in 0..reterms.len() {
        let ri = reterms[i].n_ranef();
        ensure_dense_block_within_limit(
            fixed_response_cols,
            ri,
            format!(
                "[X|y]'Z block for grouping factor '{}'",
                reterms[i].grouping_name
            ),
        )?;

        for j in 0..i {
            if reterms[i].vsize != 1 || reterms[j].vsize != 1 {
                let rj = reterms[j].n_ranef();
                ensure_dense_block_within_limit(
                    ri,
                    rj,
                    format!(
                        "off-diagonal random-effects cross-product block '{}' x '{}'",
                        reterms[i].grouping_name, reterms[j].grouping_name
                    ),
                )?;
            }
        }

        if (0..i).any(|j| !is_nested(&reterms[j], &reterms[i])) {
            for row in i..reterms.len() {
                ensure_dense_block_within_limit(
                    reterms[row].n_ranef(),
                    ri,
                    format!(
                        "crossed random-effects fill-in block '{}' x '{}'",
                        reterms[row].grouping_name, reterms[i].grouping_name
                    ),
                )?;
            }
        }
    }
    Ok(())
}

/// Single absolute floor for triangular-solve denominators (back/forward
/// substitution against the blocked Cholesky factor). This is intentionally
/// distinct from the policy-controlled Cholesky *padding* tolerance
/// (`cholesky_zero_pad_abs_tolerance`, a *relative* tolerance scaled by the
/// diagonal magnitude): padding decides whether a near-zero pivot is regularized
/// during factorization, whereas this guards a division in the solve step. All
/// solve-step zero guards reference this one constant so they cannot drift apart
/// (the prior hazard was the same magic literal copied across ~16 sites).
/// Threading the configured policy tolerance into the solve floor as well is a
/// numerics change with Julia-parity implications and is tracked separately.
pub(super) const BLOCK_TRIANGULAR_SOLVE_ZERO_TOLERANCE: f64 = 1e-30;

pub(super) fn copy_block(dst: &mut MatrixBlock, src: &MatrixBlock) {
    match (dst, src) {
        (MatrixBlock::Dense(dst_mat), MatrixBlock::Dense(src_mat)) => {
            if dst_mat.shape() == src_mat.shape() {
                dst_mat.copy_from(src_mat);
            } else {
                *dst_mat = src_mat.clone();
            }
        }
        (MatrixBlock::Diagonal(dst_diag), MatrixBlock::Diagonal(src_diag)) => {
            if dst_diag.len() == src_diag.len() {
                dst_diag.copy_from(src_diag);
            } else {
                *dst_diag = src_diag.clone();
            }
        }
        (MatrixBlock::BlockDiagonal(dst_blocks), MatrixBlock::BlockDiagonal(src_blocks))
            if dst_blocks.len() == src_blocks.len() =>
        {
            for (dst_blk, src_blk) in dst_blocks.iter_mut().zip(src_blocks.iter()) {
                if dst_blk.shape() == src_blk.shape() {
                    dst_blk.copy_from(src_blk);
                } else {
                    *dst_blk = src_blk.clone();
                }
            }
        }
        (MatrixBlock::Sparse(dst_mat), MatrixBlock::Sparse(src_mat)) => {
            if dst_mat.nrows() == src_mat.nrows()
                && dst_mat.ncols() == src_mat.ncols()
                && dst_mat.nnz() == src_mat.nnz()
                && dst_mat.col_offsets() == src_mat.col_offsets()
                && dst_mat.row_indices() == src_mat.row_indices()
            {
                dst_mat.values_mut().copy_from_slice(src_mat.values());
            } else {
                *dst_mat = src_mat.clone();
            }
        }
        // A dense destination that the factorization promoted earlier keeps
        // its dense buffer: scatter the source into it instead of reverting
        // to the source's variant (which the next factorization would only
        // promote again, one allocation per evaluation).
        (MatrixBlock::Dense(dst_mat), src) if dst_mat.shape() == (src.nrows(), src.ncols()) => {
            fill_dense_from_block(dst_mat, src, 1.0);
        }
        (dst_block, src_block) => {
            *dst_block = src_block.clone();
        }
    }
}

/// Write `scale * src` into `dst` (same shape), zeroing entries `src` does
/// not carry. Produces exactly the numbers `src.as_dense()` scaled by
/// `scale` would, without allocating.
pub(super) fn fill_dense_from_block(dst: &mut DMatrix<f64>, src: &MatrixBlock, scale: f64) {
    match src {
        MatrixBlock::Dense(src_mat) => {
            if scale == 1.0 {
                dst.copy_from(src_mat);
            } else {
                for (d, &s) in dst.as_mut_slice().iter_mut().zip(src_mat.as_slice()) {
                    *d = s * scale;
                }
            }
        }
        MatrixBlock::Diagonal(diag) => {
            dst.fill(0.0);
            for (i, &value) in diag.iter().enumerate() {
                dst[(i, i)] = if scale == 1.0 { value } else { value * scale };
            }
        }
        MatrixBlock::BlockDiagonal(blocks) => {
            dst.fill(0.0);
            let mut row_offset = 0;
            let mut col_offset = 0;
            for block in blocks {
                for col in 0..block.ncols() {
                    for row in 0..block.nrows() {
                        let value = block[(row, col)];
                        dst[(row_offset + row, col_offset + col)] =
                            if scale == 1.0 { value } else { value * scale };
                    }
                }
                row_offset += block.nrows();
                col_offset += block.ncols();
            }
        }
        MatrixBlock::Sparse(csc) => {
            dst.fill(0.0);
            for (row, col, &value) in csc.triplet_iter() {
                dst[(row, col)] = if scale == 1.0 { value } else { value * scale };
            }
        }
    }
}

/// `C -= A * Bᵀ` over typed blocks.
///
/// Sparse operands are consumed in CSC form (MixedModels.jl `mul!` for
/// `BlockedSparse` operands in `src/linalg.jl`) instead of being densified
/// on every factorization. A sparse `C` stays sparse, with its values
/// updated in place, when its pattern covers every entry of the product --
/// the nested-grouping case, where the product pattern is the `A` block
/// pattern the target was built from. Otherwise `C` is promoted to dense
/// once, as before. Dense·dense (and any other operand combination) keeps
/// the original densify-and-gemm path, so fits without sparse off-diagonal
/// blocks are bit-identical to it.
pub(super) fn subtract_product_from_blocks(c: &mut MatrixBlock, a: &MatrixBlock, b: &MatrixBlock) {
    match (a, b) {
        (MatrixBlock::Sparse(a_sp), MatrixBlock::Sparse(b_sp)) => {
            if let MatrixBlock::Sparse(c_sp) = c {
                if subtract_sparse_sparse_t_in_pattern(c_sp, a_sp, b_sp) {
                    return;
                }
            }
            subtract_sparse_sparse_t_into_dense(dense_target(c), a_sp, b_sp);
        }
        (MatrixBlock::Dense(a_d), MatrixBlock::Sparse(b_sp)) => {
            subtract_dense_sparse_t_into_dense(dense_target(c), a_d, b_sp);
        }
        (MatrixBlock::Sparse(a_sp), MatrixBlock::Dense(b_d)) => {
            subtract_sparse_dense_t_into_dense(dense_target(c), a_sp, b_d);
        }
        _ => with_dense_block(a, |a_dense| {
            with_dense_block(b, |b_dense| {
                subtract_product(c, a_dense, b_dense);
            })
        }),
    }
}

/// Promote a non-dense target to dense (the same promotion
/// `subtract_product` performs) and return its dense storage.
fn dense_target(c: &mut MatrixBlock) -> &mut DMatrix<f64> {
    if !matches!(c, MatrixBlock::Dense(_)) {
        *c = MatrixBlock::Dense(c.as_dense());
    }
    match c {
        MatrixBlock::Dense(mat) => mat,
        _ => unreachable!("target was just promoted to dense"),
    }
}

/// `C -= A * Bᵀ` with all three sparse, updating `C`'s stored values in
/// place. Returns `false` -- leaving `C` untouched -- when some entry of
/// the product falls outside `C`'s pattern.
pub(super) fn subtract_sparse_sparse_t_in_pattern(
    c: &mut CscMatrix<f64>,
    a: &CscMatrix<f64>,
    b: &CscMatrix<f64>,
) -> bool {
    debug_assert_eq!(a.ncols(), b.ncols());
    debug_assert_eq!(c.nrows(), a.nrows());
    debug_assert_eq!(c.ncols(), b.nrows());
    let (a_off, a_rows, a_vals) = a.csc_data();
    let (b_off, b_rows, b_vals) = b.csc_data();

    // Coverage pass: every (A row, B row) pair sharing an inner index must
    // be a stored entry of C. Row indices are sorted within each column.
    {
        let (c_off, c_rows, _) = c.csc_data();
        for t in 0..a.ncols() {
            let a_range = a_off[t]..a_off[t + 1];
            if a_range.is_empty() {
                continue;
            }
            for &jj in &b_rows[b_off[t]..b_off[t + 1]] {
                let col_rows = &c_rows[c_off[jj]..c_off[jj + 1]];
                if a_rows[a_range.clone()]
                    .iter()
                    .any(|row| col_rows.binary_search(row).is_err())
                {
                    return false;
                }
            }
        }
    }

    let (c_off, c_rows, c_vals) = c.csc_data_mut();
    for t in 0..a.ncols() {
        let a_range = a_off[t]..a_off[t + 1];
        if a_range.is_empty() {
            continue;
        }
        for ib in b_off[t]..b_off[t + 1] {
            let jj = b_rows[ib];
            let bv = b_vals[ib];
            let start = c_off[jj];
            let col_rows = &c_rows[start..c_off[jj + 1]];
            for ia in a_range.clone() {
                let pos = start
                    + col_rows
                        .binary_search(&a_rows[ia])
                        .expect("coverage checked above");
                c_vals[pos] -= a_vals[ia] * bv;
            }
        }
    }
    true
}

/// `C -= A * Bᵀ` into a dense `C` with `A` and `B` sparse.
pub(super) fn subtract_sparse_sparse_t_into_dense(
    c: &mut DMatrix<f64>,
    a: &CscMatrix<f64>,
    b: &CscMatrix<f64>,
) {
    debug_assert_eq!(a.ncols(), b.ncols());
    debug_assert_eq!(c.shape(), (a.nrows(), b.nrows()));
    let (a_off, a_rows, a_vals) = a.csc_data();
    let (b_off, b_rows, b_vals) = b.csc_data();
    let m = c.nrows();
    let c_data = c.as_mut_slice();
    for t in 0..a.ncols() {
        for ib in b_off[t]..b_off[t + 1] {
            let bv = b_vals[ib];
            let c_col = &mut c_data[b_rows[ib] * m..(b_rows[ib] + 1) * m];
            for ia in a_off[t]..a_off[t + 1] {
                c_col[a_rows[ia]] -= a_vals[ia] * bv;
            }
        }
    }
}

/// `C -= A * Bᵀ` into a dense `C` with `A` dense and `B` sparse: one column
/// axpy of `A[:, t]` per stored entry of `B[:, t]`.
pub(super) fn subtract_dense_sparse_t_into_dense(
    c: &mut DMatrix<f64>,
    a: &DMatrix<f64>,
    b: &CscMatrix<f64>,
) {
    debug_assert_eq!(a.ncols(), b.ncols());
    debug_assert_eq!(c.shape(), (a.nrows(), b.nrows()));
    let (b_off, b_rows, b_vals) = b.csc_data();
    let m = c.nrows();
    let a_data = a.as_slice();
    let c_data = c.as_mut_slice();
    for t in 0..b.ncols() {
        let a_col = &a_data[t * m..(t + 1) * m];
        for ib in b_off[t]..b_off[t + 1] {
            let bv = b_vals[ib];
            let c_col = &mut c_data[b_rows[ib] * m..(b_rows[ib] + 1) * m];
            for (dst, &av) in c_col.iter_mut().zip(a_col) {
                *dst -= av * bv;
            }
        }
    }
}

/// `C -= A * Bᵀ` into a dense `C` with `A` sparse and `B` dense.
pub(super) fn subtract_sparse_dense_t_into_dense(
    c: &mut DMatrix<f64>,
    a: &CscMatrix<f64>,
    b: &DMatrix<f64>,
) {
    debug_assert_eq!(a.ncols(), b.ncols());
    debug_assert_eq!(c.shape(), (a.nrows(), b.nrows()));
    // Column-major walk over C: for each output column jj, scatter
    // `A[:, t] * B[jj, t]` down that column (contiguous writes).
    let (a_off, a_rows, a_vals) = a.csc_data();
    let m = c.nrows();
    let c_data = c.as_mut_slice();
    for jj in 0..b.nrows() {
        let c_col = &mut c_data[jj * m..(jj + 1) * m];
        for t in 0..a.ncols() {
            let bv = b[(jj, t)];
            for ia in a_off[t]..a_off[t + 1] {
                c_col[a_rows[ia]] -= a_vals[ia] * bv;
            }
        }
    }
}

/// `dst -= L * v` for a typed block `L`, accumulating each row's dot
/// product in column order before subtracting -- the same arithmetic as
/// the dense row-dot loop over `L.as_dense()`. For finite `v` the results
/// are identical to that loop (skipped entries are exact zeros); a
/// non-finite `v` entry differs, since the dense loop forms `0 * Inf = NaN`
/// for entries the typed kernels never visit.
pub(crate) fn subtract_block_matvec(dst: &mut [f64], l: &MatrixBlock, v: &[f64]) {
    debug_assert_eq!(dst.len(), l.nrows());
    debug_assert_eq!(v.len(), l.ncols());
    match l {
        MatrixBlock::Dense(mat) => {
            for (row, dst_row) in dst.iter_mut().enumerate() {
                let mut dot = 0.0;
                for (col, &v_col) in v.iter().enumerate() {
                    dot += mat[(row, col)] * v_col;
                }
                *dst_row -= dot;
            }
        }
        MatrixBlock::Sparse(csc) => {
            let mut dots = vec![0.0; dst.len()];
            let (off, rows, vals) = csc.csc_data();
            for (col, &v_col) in v.iter().enumerate() {
                for idx in off[col]..off[col + 1] {
                    dots[rows[idx]] += vals[idx] * v_col;
                }
            }
            for (dst_row, dot) in dst.iter_mut().zip(dots) {
                *dst_row -= dot;
            }
        }
        MatrixBlock::Diagonal(diag) => {
            for ((dst_row, &d), &v_row) in dst.iter_mut().zip(diag.iter()).zip(v) {
                let dot = 0.0 + d * v_row;
                *dst_row -= dot;
            }
        }
        MatrixBlock::BlockDiagonal(blocks) => {
            // One offset serves rows and columns: sub-blocks are square
            // (the uniform block-diagonal RE layout).
            let mut offset = 0;
            for block in blocks {
                debug_assert_eq!(
                    block.nrows(),
                    block.ncols(),
                    "non-square diagonal sub-block"
                );
                for row in 0..block.nrows() {
                    let mut dot = 0.0;
                    for col in 0..block.ncols() {
                        dot += block[(row, col)] * v[offset + col];
                    }
                    dst[offset + row] -= dot;
                }
                offset += block.nrows();
            }
        }
    }
}

/// `dst -= Lᵀ * u` for a typed block `L`, with the same accumulation
/// contract as [`subtract_block_matvec`] (identical for finite `u`).
pub(crate) fn subtract_block_transpose_matvec(dst: &mut [f64], l: &MatrixBlock, u: &[f64]) {
    debug_assert_eq!(dst.len(), l.ncols());
    debug_assert_eq!(u.len(), l.nrows());
    match l {
        MatrixBlock::Dense(mat) => {
            for (row, dst_row) in dst.iter_mut().enumerate() {
                let mut dot = 0.0;
                for (col, &u_col) in u.iter().enumerate() {
                    dot += mat[(col, row)] * u_col;
                }
                *dst_row -= dot;
            }
        }
        MatrixBlock::Sparse(csc) => {
            let (off, rows, vals) = csc.csc_data();
            for (row, dst_row) in dst.iter_mut().enumerate() {
                let mut dot = 0.0;
                for idx in off[row]..off[row + 1] {
                    dot += vals[idx] * u[rows[idx]];
                }
                *dst_row -= dot;
            }
        }
        MatrixBlock::Diagonal(diag) => {
            for ((dst_row, &d), &u_row) in dst.iter_mut().zip(diag.iter()).zip(u) {
                let dot = 0.0 + d * u_row;
                *dst_row -= dot;
            }
        }
        MatrixBlock::BlockDiagonal(blocks) => {
            // One offset serves rows and columns: sub-blocks are square.
            let mut offset = 0;
            for block in blocks {
                debug_assert_eq!(
                    block.nrows(),
                    block.ncols(),
                    "non-square diagonal sub-block"
                );
                for row in 0..block.ncols() {
                    let mut dot = 0.0;
                    for col in 0..block.nrows() {
                        dot += block[(col, row)] * u[offset + col];
                    }
                    dst[offset + row] -= dot;
                }
                offset += block.nrows();
            }
        }
    }
}

#[inline]
pub(super) fn solve_scaled_vsize2_row(
    a10: &DMatrix<f64>,
    row: usize,
    col0: usize,
    col1: usize,
    lam00: f64,
    lam10: f64,
    lam11: f64,
    l00: f64,
    l10: f64,
    l11: f64,
) -> (f64, f64) {
    let x0 = a10[(row, col0)];
    let x1 = a10[(row, col1)];
    let mut solved0 = x0 * lam00 + x1 * lam10;
    let mut solved1 = x1 * lam11;

    solved0 = if l00.abs() < BLOCK_TRIANGULAR_SOLVE_ZERO_TOLERANCE {
        0.0
    } else {
        solved0 / l00
    };
    solved1 = if l11.abs() < BLOCK_TRIANGULAR_SOLVE_ZERO_TOLERANCE {
        0.0
    } else {
        (solved1 - solved0 * l10) / l11
    };

    (solved0, solved1)
}

#[inline]
pub(super) fn solve_scaled_vsize1_row(
    a10: &DMatrix<f64>,
    row: usize,
    col: usize,
    lambda: f64,
    l00: f64,
) -> f64 {
    if l00.abs() < BLOCK_TRIANGULAR_SOLVE_ZERO_TOLERANCE {
        0.0
    } else {
        a10[(row, col)] * lambda / l00
    }
}

pub(crate) fn update_l_from_parts(
    a_blocks: &[MatrixBlock],
    l_blocks: &mut [MatrixBlock],
    reterms: &[ReMat],
    cholesky_zero_pad_tolerance: f64,
) -> Result<()> {
    update_l_re_rows_from_parts(a_blocks, l_blocks, reterms, cholesky_zero_pad_tolerance)?;
    update_l_fe_row_from_parts(a_blocks, l_blocks, reterms, cholesky_zero_pad_tolerance)
}

/// Random-effects rows of the blocked Cholesky: every `L[i, j]` with
/// `i < k`, where `k` is the number of RE terms.
///
/// The blocked factorization is left-looking, so the RE rows never read the
/// trailing `[X|y]` row. Running the RE rows to completion first and the
/// `[X|y]` row second ([`update_l_fe_row_from_parts`]) applies exactly the
/// same floating-point operations to every block, in the same per-block
/// order, as the interleaved sweep; [`update_l_from_parts`] is the two
/// halves back to back. Conditional-mode PIRLS with fixed β reads only the
/// RE rows, so it runs this half alone and marks the `[X|y]` row stale.
#[allow(
    clippy::needless_range_loop,
    reason = "blocked Cholesky indexes packed A/L blocks and covariance terms with the same Julia-compatible block coordinates"
)]
pub(crate) fn update_l_re_rows_from_parts(
    a_blocks: &[MatrixBlock],
    l_blocks: &mut [MatrixBlock],
    reterms: &[ReMat],
    cholesky_zero_pad_tolerance: f64,
) -> Result<()> {
    let k = reterms.len(); // number of RE terms

    // Copy A to L, scaling by Λ
    // For diagonal blocks L[j,j] = Λ_j' A[j,j] Λ_j + I
    for j in 0..k {
        let idx_jj = block_index(j, j);
        copy_scale_inflate(&mut l_blocks[idx_jj], &a_blocks[idx_jj], &reterms[j]);
    }

    // For off-diagonal RE blocks L[i,j] = Λ_i' A[i,j] Λ_j, i > j
    for i in 1..k {
        for j in 0..i {
            let idx_ij = block_index(i, j);
            copy_and_scale_offdiag(
                &mut l_blocks[idx_ij],
                &a_blocks[idx_ij],
                &reterms[i],
                &reterms[j],
            );
        }
    }

    // Blocked Cholesky factorization of the RE rows
    for j in 0..k {
        let diag_idx = block_index(j, j);

        // Update L[j,j] by subtracting L[j,0..j] * L[j,0..j]'
        downdate_diagonal_block(l_blocks, j);

        // Cholesky of diagonal block
        cholesky_block_with_tolerance(&mut l_blocks[diag_idx], cholesky_zero_pad_tolerance)?;

        // Solve for off-diagonal RE blocks: L[i,j] for j < i < k
        for i in (j + 1)..k {
            solve_off_diagonal_block(l_blocks, i, j)?;
        }
    }

    Ok(())
}

/// Trailing `[X|y]` row of the blocked Cholesky (`L[k, 0..=k]`), given RE
/// rows already factored for the current `A` and Λ by
/// [`update_l_re_rows_from_parts`].
#[allow(
    clippy::needless_range_loop,
    reason = "blocked Cholesky indexes packed A/L blocks and covariance terms with the same Julia-compatible block coordinates"
)]
pub(crate) fn update_l_fe_row_from_parts(
    a_blocks: &[MatrixBlock],
    l_blocks: &mut [MatrixBlock],
    reterms: &[ReMat],
    cholesky_zero_pad_tolerance: f64,
) -> Result<()> {
    let k = reterms.len(); // number of RE terms

    // For FE-RE blocks L[k,j] = A[k,j] Λ_j (no Λ on left for FeMat)
    for j in 0..k {
        let idx_kj = block_index(k, j);
        copy_and_rmul_lambda(&mut l_blocks[idx_kj], &a_blocks[idx_kj], &reterms[j]);
    }

    // Copy the [X|y]'[X|y] block unchanged
    let idx_kk = block_index(k, k);
    copy_block(&mut l_blocks[idx_kk], &a_blocks[idx_kk]);

    // L[k,j] for j < k, in the same column order as the interleaved sweep
    for j in 0..k {
        solve_off_diagonal_block(l_blocks, k, j)?;
    }

    // L[k,k]: downdate by the finished row, then factor
    downdate_diagonal_block(l_blocks, k);
    cholesky_block_with_tolerance(&mut l_blocks[idx_kk], cholesky_zero_pad_tolerance)
}

/// `L[j,j] -= Σ_{jj<j} L[j,jj] L[j,jj]'`.
fn downdate_diagonal_block(l_blocks: &mut [MatrixBlock], j: usize) {
    let diag_idx = block_index(j, j);
    for jj in 0..j {
        let off_idx = block_index(j, jj);
        with_block_pair_mut(l_blocks, diag_idx, off_idx, |diag, off| match off {
            MatrixBlock::Sparse(off_sparse) => rank_k_downdate_sparse(diag, off_sparse),
            _ => {
                if let Some(off_dense) = off.as_dense_ref() {
                    rank_k_downdate(diag, off_dense);
                } else {
                    let off_dense = off.as_dense();
                    rank_k_downdate(diag, &off_dense);
                }
            }
        });
    }
}

/// `L[i,j] = (L[i,j] - Σ_{jj<j} L[i,jj] L[j,jj]') L[j,j]^{-T}` for `i > j`,
/// with `L[j,j]` already factored.
fn solve_off_diagonal_block(l_blocks: &mut [MatrixBlock], i: usize, j: usize) -> Result<()> {
    let target_idx = block_index(i, j);
    let diag_idx = block_index(j, j);

    // L[i,j] -= sum_{jj<j} L[i,jj] * L[j,jj]'
    for jj in 0..j {
        with_block_triple(
            l_blocks,
            target_idx,
            block_index(i, jj),
            block_index(j, jj),
            subtract_product_from_blocks,
        )?;
    }

    // L[i,j] = L[i,j] * L[j,j]^{-T}
    with_block_pair_mut(l_blocks, target_idx, diag_idx, |target, diag| {
        rdiv_lower_transpose(target, diag);
    });
    Ok(())
}

pub(super) fn is_nested(a: &ReMat, b: &ReMat) -> bool {
    if a.refs.len() != b.refs.len() {
        return false;
    }

    let mut bins = vec![None; a.n_levels()];
    for (&aref, &bref) in a.refs.iter().zip(b.refs.iter()) {
        let slot = &mut bins[aref as usize];
        match slot {
            Some(prev) if *prev != bref => return false,
            Some(_) => {}
            None => *slot = Some(bref),
        }
    }
    true
}

pub(super) fn promote_crossed_fill_in_blocks(l: &mut [MatrixBlock], reterms: &[ReMat]) {
    let k = reterms.len();
    for i in 1..k {
        if (0..i).any(|j| !is_nested(&reterms[j], &reterms[i])) {
            for row in i..k {
                let idx = block_index(row, i);
                if !matches!(l[idx], MatrixBlock::Dense(_)) {
                    l[idx] = MatrixBlock::Dense(l[idx].as_dense());
                }
            }
        }
    }
}

/// Create the A (cross-product) and L (Cholesky) block arrays.
#[cfg(test)]
pub(super) fn create_al(
    reterms: &[ReMat],
    xy: &FeMat,
) -> Result<(Vec<MatrixBlock>, Vec<MatrixBlock>)> {
    validate_dense_block_plan(reterms, xy.wtxy.ncols())?;

    if reterms.len() == 1 && reterms[0].vsize == 2 && reterms[0].n_ranef() >= 512 {
        return Ok(create_al_single_vsize2(&reterms[0], xy));
    }

    let k = reterms.len();
    let total = k + 1;
    let n_blocks = total * (total + 1) / 2;
    let mut a = Vec::with_capacity(n_blocks);
    let mut l = Vec::with_capacity(n_blocks);

    // RE × RE blocks
    for i in 0..k {
        for j in 0..=i {
            let block = if i == j {
                // Diagonal block: Z_i' Z_i
                compute_re_cross_product(&reterms[i], &reterms[i])
            } else {
                // Off-diagonal: Z_i' Z_j
                compute_re_cross_product(&reterms[i], &reterms[j])
            };
            a.push(block.clone());
            l.push(block);
        }
    }

    // FE × RE blocks: [X|y]' Z_j
    for reterm in reterms {
        let block = compute_fe_re_cross_product(xy, reterm);
        a.push(block.clone());
        l.push(block);
    }

    // FE × FE block: [X|y]' [X|y]
    let wtxy = &xy.wtxy;
    let feblock = MatrixBlock::Dense(wtxy.transpose() * wtxy);
    a.push(feblock.clone());
    l.push(feblock);

    promote_crossed_fill_in_blocks(&mut l, reterms);

    Ok((a, l))
}

/// Create A/L blocks using fixed-design backend cross-products for the fixed
/// side of the system.
pub(super) fn create_al_from_fixed_design(
    reterms: &[ReMat],
    fixed_design: &FixedDesign,
    y: &DVector<f64>,
    sqrtwts: Option<&DVector<f64>>,
) -> Result<(Vec<MatrixBlock>, Vec<MatrixBlock>)> {
    validate_dense_block_plan(reterms, fixed_design.n_cols() + 1)?;
    let weighted_fixed_design = weighted_fixed_design_for_solver(fixed_design, sqrtwts)?;
    let weighted_y = weighted_response_for_solver(y, sqrtwts)?;

    let k = reterms.len();
    let total = k + 1;
    let n_blocks = total * (total + 1) / 2;
    let mut a = Vec::with_capacity(n_blocks);
    let mut l = Vec::with_capacity(n_blocks);

    for i in 0..k {
        for j in 0..=i {
            let block = if i == j {
                compute_re_cross_product(&reterms[i], &reterms[i])
            } else {
                compute_re_cross_product(&reterms[i], &reterms[j])
            };
            a.push(block.clone());
            l.push(block);
        }
    }

    for re in reterms {
        let block =
            compute_fixed_response_re_cross_product(&weighted_fixed_design, &weighted_y, re)?;
        let block = finalize_fixed_re_block(block, k);
        a.push(block.clone());
        l.push(block);
    }

    let block = MatrixBlock::Dense(compute_fixed_response_cross_product(
        &weighted_fixed_design,
        &weighted_y,
    )?);
    a.push(block.clone());
    l.push(block);

    promote_crossed_fill_in_blocks(&mut l, reterms);

    Ok((a, l))
}

/// Sparse `[X|y]'Z` blocks are only kept with a single random-effect term:
/// with multiple terms the blocked factorization's off-diagonal updates
/// (`subtract_product`) would re-materialize them on every θ evaluation, so
/// promote them to dense once at construction instead.
pub(super) fn finalize_fixed_re_block(block: MatrixBlock, n_reterms: usize) -> MatrixBlock {
    if n_reterms > 1 && matches!(block, MatrixBlock::Sparse(_)) {
        MatrixBlock::Dense(block.as_dense())
    } else {
        block
    }
}

/// The solver-side fixed design: the caller's design borrowed as is when
/// there are no observation weights, or a row-weighted copy otherwise.
pub(super) fn weighted_fixed_design_for_solver<'a>(
    fixed_design: &'a FixedDesign,
    sqrtwts: Option<&DVector<f64>>,
) -> Result<std::borrow::Cow<'a, FixedDesign>> {
    match sqrtwts {
        Some(weights) => Ok(std::borrow::Cow::Owned(
            fixed_design.with_sqrt_weights(weights)?,
        )),
        None => Ok(std::borrow::Cow::Borrowed(fixed_design)),
    }
}

/// The solver-side response: borrowed when unweighted, a weighted copy
/// otherwise.
pub(super) fn weighted_response_for_solver<'a>(
    y: &'a DVector<f64>,
    sqrtwts: Option<&DVector<f64>>,
) -> Result<std::borrow::Cow<'a, DVector<f64>>> {
    if let Some(weights) = sqrtwts {
        if weights.len() != y.len() {
            return Err(MixedModelError::DimensionMismatch(format!(
                "response has {} rows but sqrt weights have {}",
                y.len(),
                weights.len()
            )));
        }
        Ok(std::borrow::Cow::Owned(y.component_mul(weights)))
    } else {
        Ok(std::borrow::Cow::Borrowed(y))
    }
}

pub(super) fn fixed_design_backend_diagnostic(fixed_design: &FixedDesign) -> Diagnostic {
    let summary = fixed_design.summary();
    let active_entries = fixed_design_active_entries(fixed_design);
    let density = fixed_design_density(fixed_design);
    let mut diagnostic = Diagnostic::new(
        DiagnosticCode::SupportNote,
        DiagnosticSeverity::Info,
        DiagnosticStage::DesignAudit,
        format!(
            "fixed-effect design backend selected: {}; n={}, p={}, dense_if_materialized={} bytes, active_entries={}, density={:.6}",
            fixed_design_storage_label(summary.storage),
            summary.n_obs,
            summary.n_cols,
            summary.dense_bytes,
            active_entries,
            density
        ),
    )
    .with_suggested_actions(vec![
        "no action required; streamed fixed effects avoid materializing dense X for solver cross-products".to_string(),
        "rank/pivot detection uses a streamed Gram certificate when the design is comfortably full rank; ambiguous designs fall back to an exact dense Householder pass (see the fixed_design_rank_path diagnostic)".to_string(),
    ]);
    diagnostic.payload.insert(
        "diagnostic_kind".to_string(),
        serde_json::json!("fixed_design_backend"),
    );
    diagnostic.payload.insert(
        "storage".to_string(),
        serde_json::json!(fixed_design_storage_label(summary.storage)),
    );
    diagnostic
        .payload
        .insert("n_obs".to_string(), serde_json::json!(summary.n_obs));
    diagnostic
        .payload
        .insert("n_cols".to_string(), serde_json::json!(summary.n_cols));
    diagnostic.payload.insert(
        "dense_bytes".to_string(),
        serde_json::json!(summary.dense_bytes.to_string()),
    );
    diagnostic.payload.insert(
        "active_entries".to_string(),
        serde_json::json!(active_entries),
    );
    diagnostic
        .payload
        .insert("density".to_string(), serde_json::json!(density));
    diagnostic
}

pub(super) fn fixed_design_active_entries(fixed_design: &FixedDesign) -> usize {
    match fixed_design {
        FixedDesign::Dense(design) => design.n_obs() * design.n_cols(),
        FixedDesign::Streamed(design) => design.active_entries(),
    }
}

pub(super) fn fixed_design_density(fixed_design: &FixedDesign) -> f64 {
    match fixed_design {
        FixedDesign::Dense(design) => {
            if design.n_obs() == 0 || design.n_cols() == 0 {
                0.0
            } else {
                1.0
            }
        }
        FixedDesign::Streamed(design) => design.density(),
    }
}

pub(super) fn fixed_design_storage_label(storage: FixedDesignStorage) -> &'static str {
    match storage {
        FixedDesignStorage::Dense => "dense",
        FixedDesignStorage::Streamed => "streamed",
        FixedDesignStorage::Sparse => "sparse",
    }
}

#[cfg(test)]
pub(super) fn create_al_single_vsize2(
    re: &ReMat,
    xy: &FeMat,
) -> (Vec<MatrixBlock>, Vec<MatrixBlock>) {
    let nlevels = re.n_levels();
    let pp1 = xy.wtxy.ncols();
    let mut re_re_blocks: Vec<DMatrix<f64>> = (0..nlevels).map(|_| DMatrix::zeros(2, 2)).collect();
    let mut fe_re = DMatrix::zeros(pp1, re.n_ranef());
    let mut fe_fe = DMatrix::zeros(pp1, pp1);

    for obs in 0..re.n_obs() {
        let level = re.refs[obs] as usize;
        let col0 = 2 * level;
        let col1 = col0 + 1;
        let z0 = re.wtz[(0, obs)];
        let z1 = re.wtz[(1, obs)];

        let block = &mut re_re_blocks[level];
        block[(0, 0)] += z0 * z0;
        block[(0, 1)] += z0 * z1;
        block[(1, 0)] += z1 * z0;
        block[(1, 1)] += z1 * z1;

        for row in 0..pp1 {
            let x = xy.wtxy[(obs, row)];
            fe_re[(row, col0)] += x * z0;
            fe_re[(row, col1)] += x * z1;
            for col in 0..=row {
                fe_fe[(row, col)] += x * xy.wtxy[(obs, col)];
            }
        }
    }

    for row in 0..pp1 {
        for col in 0..row {
            fe_fe[(col, row)] = fe_fe[(row, col)];
        }
    }

    let a = vec![
        MatrixBlock::BlockDiagonal(re_re_blocks),
        MatrixBlock::Dense(fe_re),
        MatrixBlock::Dense(fe_fe),
    ];
    let l = a.clone();
    (a, l)
}

/// Create the structural A and L block arrays for `[Z X]' [Z X]`.
pub(crate) fn create_structural_al(
    reterms: &[ReMat],
    x: &DMatrix<f64>,
) -> Result<(Vec<MatrixBlock>, Vec<MatrixBlock>)> {
    validate_dense_block_plan(reterms, x.ncols())?;

    let k = reterms.len();
    let total = k + 1;
    let n_blocks = total * (total + 1) / 2;
    let mut a = Vec::with_capacity(n_blocks);
    let mut l = Vec::with_capacity(n_blocks);

    for i in 0..k {
        for j in 0..=i {
            let block = if i == j {
                compute_re_cross_product(&reterms[i], &reterms[i])
            } else {
                compute_re_cross_product(&reterms[i], &reterms[j])
            };
            a.push(block.clone());
            l.push(block);
        }
    }

    for reterm in reterms {
        let block = compute_x_re_cross_product(x, reterm);
        a.push(block.clone());
        l.push(block);
    }

    let xblock = MatrixBlock::Dense(x.transpose() * x);
    a.push(xblock.clone());
    l.push(xblock);

    promote_crossed_fill_in_blocks(&mut l, reterms);

    Ok((a, l))
}

/// Compute Z_i' Z_j for two random effects terms.
pub(super) fn compute_re_cross_product(a: &ReMat, b: &ReMat) -> MatrixBlock {
    let nranef_a = a.n_ranef();
    let nranef_b = b.n_ranef();

    // `wtz` is `vsize × n_obs` column-major, so observation `obs` occupies
    // the contiguous run `[obs * vsize, (obs + 1) * vsize)`. Every branch
    // below accumulates each output cell in increasing observation order,
    // exactly as the element-indexed loops it replaces, so the sums are
    // bit-identical; only the bounds-checked matrix indexing is gone.
    let a_wtz = a.wtz.as_slice();
    let b_wtz = b.wtz.as_slice();

    if std::ptr::eq(a, b) && a.vsize == 1 {
        // Scalar RE: diagonal result
        let n_levels = a.n_levels();
        let mut diag = DVector::zeros(n_levels);
        let out = diag.as_mut_slice();
        for (&ref_idx, &z) in a.refs.iter().zip(a_wtz) {
            out[ref_idx as usize] += z * z;
        }
        MatrixBlock::Diagonal(diag)
    } else if std::ptr::eq(a, b) && a.vsize > 1 {
        // Vector RE, same term: block-diagonal result
        // Each level k gets a vsize × vsize block: sum_{obs with ref==k} wtz[:,obs] * wtz[:,obs]'
        let s = a.vsize;
        let n_levels = a.n_levels();
        let mut blocks: Vec<DMatrix<f64>> = (0..n_levels).map(|_| DMatrix::zeros(s, s)).collect();

        for (obs, &ref_idx) in a.refs.iter().enumerate() {
            let col = &a_wtz[obs * s..(obs + 1) * s];
            // `s × s` column-major: (si, sj) -> sj * s + si.
            let blk = blocks[ref_idx as usize].as_mut_slice();
            for si in 0..s {
                let wtz_si = col[si];
                for sj in 0..s {
                    blk[sj * s + si] += wtz_si * col[sj];
                }
            }
        }
        MatrixBlock::BlockDiagonal(blocks)
    } else if a.vsize == 1 && b.vsize == 1 {
        // Keep every scalar-intercept off-diagonal cross-product sparse, not
        // just the truly-crossed ones. The raw Z_a' Z_b cross-product of two
        // scalar-intercept terms always has at most n_obs structural nonzeros
        // (one per observation) regardless of nesting, while its dense shape
        // (nlevels_a x nlevels_b) can be enormous. Reverse-ordered nested
        // terms hit this too: with reterms sorted by decreasing nranef, a
        // finer factor (e.g. observation-level INDEX, 403 levels) precedes a
        // coarser one (BROOD, 118 levels), so A[BROOD x INDEX] is a 118x403
        // block with only 403 nonzeros. Materializing it dense forced dense
        // gemm/downdate through the blocked Cholesky; the sparse form lets the
        // factorization use rank_k_downdate_sparse and keeps the diagonal
        // INDEX block diagonal. Numerically transparent (objective/theta/beta
        // unchanged to ~1e-8, identical optimizer trajectory) but ~25% faster
        // on the grouseticks Poisson GLMM (bd-01KRSQYRHF8VK627HZ6Z23CP93).
        let mut entries = BTreeMap::<(usize, usize), f64>::new();

        for (obs, (&ri, &rj)) in a.refs.iter().zip(b.refs.iter()).enumerate() {
            let value = a_wtz[obs] * b_wtz[obs];
            if value != 0.0 {
                *entries.entry((ri as usize, rj as usize)).or_insert(0.0) += value;
            }
        }
        let mut result = CooMatrix::new(nranef_a, nranef_b);
        for ((row, col), value) in entries {
            if value != 0.0 {
                result.push(row, col, value);
            }
        }
        MatrixBlock::Sparse(CscMatrix::from(&result))
    } else {
        // General case: dense result. This includes reverse-ordered nested
        // scalar terms, where preserving the previous dense algebra keeps the
        // optimizer path stable.
        let va = a.vsize;
        let vb = b.vsize;
        let mut result = DMatrix::zeros(nranef_a, nranef_b);
        // `nranef_a × nranef_b` column-major: (row, col) -> col * nranef_a + row.
        let out = result.as_mut_slice();

        for (obs, (&ri, &rj)) in a.refs.iter().zip(b.refs.iter()).enumerate() {
            let row0 = ri as usize * va;
            let col0 = rj as usize * vb;
            let ca = &a_wtz[obs * va..(obs + 1) * va];
            let cb = &b_wtz[obs * vb..(obs + 1) * vb];
            for si in 0..va {
                let wtz_si = ca[si];
                for sj in 0..vb {
                    out[(col0 + sj) * nranef_a + row0 + si] += wtz_si * cb[sj];
                }
            }
        }
        MatrixBlock::Dense(result)
    }
}

/// Compute [X|y]' Z_j.
#[cfg(test)]
pub(super) fn compute_fe_re_cross_product(xy: &FeMat, re: &ReMat) -> MatrixBlock {
    let pp1 = xy.wtxy.ncols(); // p + 1
    let nranef = re.n_ranef();
    let n = re.refs.len();

    let mut result = DMatrix::zeros(pp1, nranef);
    let wtxy = &xy.wtxy;

    for obs in 0..n {
        let r = re.refs[obs] as usize;
        for col in 0..pp1 {
            for s in 0..re.vsize {
                result[(col, r * re.vsize + s)] += wtxy[(obs, col)] * re.wtz[(s, obs)];
            }
        }
    }

    MatrixBlock::Dense(result)
}

/// Compute `[X|y]' Z_j` using fixed-design backend cross-products.
pub(super) fn compute_fixed_response_re_cross_product(
    fixed_design: &FixedDesign,
    y: &DVector<f64>,
    re: &ReMat,
) -> Result<MatrixBlock> {
    if y.len() != fixed_design.n_obs() {
        return Err(MixedModelError::DimensionMismatch(format!(
            "fixed-effect design has {} rows but response has {}",
            fixed_design.n_obs(),
            y.len()
        )));
    }
    if re.n_obs() != fixed_design.n_obs() {
        return Err(MixedModelError::DimensionMismatch(format!(
            "fixed-effect design has {} rows but random term '{}' has {} rows",
            fixed_design.n_obs(),
            re.grouping_name,
            re.n_obs()
        )));
    }

    let fixed_re = fixed_design.xt_reterm(re)?;
    let response_re = response_re_cross_product_vec(y.as_slice(), re);
    let pp1 = fixed_design.n_cols() + 1;

    // A sparse X'Z (streamed high-cardinality designs) stays sparse in the
    // combined [X|y]'Z block; the y'Z row appended below is structurally
    // dense but adds only n_ranef nonzeros.
    if let MatrixBlock::Sparse(xt_sparse) = &fixed_re {
        let mut coo = CooMatrix::new(pp1, re.n_ranef());
        for (row, col, value) in xt_sparse.triplet_iter() {
            coo.push(row, col, *value);
        }
        for (col, &value) in response_re.iter().enumerate() {
            if value != 0.0 {
                coo.push(pp1 - 1, col, value);
            }
        }
        return Ok(MatrixBlock::Sparse(CscMatrix::from(&coo)));
    }

    let fixed_re = match fixed_re {
        MatrixBlock::Dense(dense) => dense,
        other => other.as_dense(),
    };
    let p = fixed_design.n_cols();
    let nranef = re.n_ranef();
    let mut result = DMatrix::zeros(pp1, nranef);
    // Stack `X'Z` (p rows) over `y'Z` (one row), column by column.
    let src = fixed_re.as_slice();
    let out = result.as_mut_slice();
    let yz = response_re.as_slice();
    for col in 0..nranef {
        out[col * pp1..col * pp1 + p].copy_from_slice(&src[col * p..(col + 1) * p]);
        out[col * pp1 + p] = yz[col];
    }
    Ok(MatrixBlock::Dense(result))
}

/// Structural sparsity pattern of the scalar-intercept cross product
/// `Z_a' Z_b` (one potential nonzero per distinct `(level_a, level_b)` pair
/// that co-occurs in an observation), plus, for every observation, the
/// position of its pair in the CSC value array. The pattern depends only
/// on the two reference vectors, so the weighted rebuilds that PIRLS runs
/// every iteration can refresh the values in place instead of re-deriving
/// the pattern through a map of `(row, col)` keys each time.
#[derive(Debug, Clone)]
pub(crate) struct ScalarCrossPattern {
    pub(super) csc: CscMatrix<f64>,
    /// `entry_of_obs[obs]` indexes `csc.values()`.
    pub(super) entry_of_obs: Vec<u32>,
}

impl ScalarCrossPattern {
    pub(super) fn new(a: &ReMat, b: &ReMat) -> Self {
        debug_assert_eq!(a.vsize, 1);
        debug_assert_eq!(b.vsize, 1);
        // Unique (col = level_b, row = level_a) pairs in CSC order.
        let mut pairs: Vec<(u32, u32)> = a
            .refs
            .iter()
            .zip(&b.refs)
            .map(|(&ri, &rj)| (rj, ri))
            .collect();
        pairs.sort_unstable();
        pairs.dedup();
        let n_cols = b.n_ranef();
        let mut col_offsets = Vec::with_capacity(n_cols + 1);
        let mut row_indices = Vec::with_capacity(pairs.len());
        col_offsets.push(0usize);
        let mut current_col = 0usize;
        for &(col, row) in &pairs {
            while current_col < col as usize {
                col_offsets.push(row_indices.len());
                current_col += 1;
            }
            row_indices.push(row as usize);
        }
        while col_offsets.len() < n_cols + 1 {
            col_offsets.push(row_indices.len());
        }
        let values = vec![0.0; pairs.len()];
        let csc =
            CscMatrix::try_from_csc_data(a.n_ranef(), n_cols, col_offsets, row_indices, values)
                .expect("structural scalar cross pattern is sorted CSC data");
        let entry_of_obs = a
            .refs
            .iter()
            .zip(&b.refs)
            .map(|(&ri, &rj)| {
                pairs
                    .binary_search(&(rj, ri))
                    .expect("every observation pair is in the pattern") as u32
            })
            .collect();
        Self { csc, entry_of_obs }
    }

    /// Refresh the values for the current weighted `Z` columns: each entry
    /// accumulates its observations' products in increasing observation
    /// order, exactly as `compute_re_cross_product`'s map-based arm does.
    pub(super) fn refresh(&mut self, a: &ReMat, b: &ReMat) -> MatrixBlock {
        let a_wtz = a.wtz.as_slice();
        let b_wtz = b.wtz.as_slice();
        let values = self.csc.values_mut();
        values.fill(0.0);
        for ((&entry, &za), &zb) in self.entry_of_obs.iter().zip(a_wtz).zip(b_wtz) {
            let value = za * zb;
            if value != 0.0 {
                values[entry as usize] += value;
            }
        }
        MatrixBlock::Sparse(self.csc.clone())
    }
}

/// `[X|y]' Z_j` straight from the already-weighted `[X|y]` matrix
/// (`FeMat::wtxy`, `n × (p+1)`) and the weighted `Z_j` (`ReMat::wtz`).
///
/// Used by the weighted A-block rebuild (PIRLS iterations, weighted
/// refits), where the solver-side weighted design is exactly `wtxy` and
/// materializing a weighted `FixedDesign` copy every iteration was the
/// dominant per-iteration cost. Each output cell is accumulated in
/// increasing observation order, and the products `sw·x · wtz` are the
/// same ones the `FixedDesign` route forms, so the block is bit-identical
/// to `compute_fixed_response_re_cross_product` on the weighted design.
pub(super) fn compute_wtxy_re_cross_product(wtxy: &DMatrix<f64>, re: &ReMat) -> DMatrix<f64> {
    let pp1 = wtxy.ncols();
    let nranef = re.n_ranef();
    let vsize = re.vsize;
    let mut result = DMatrix::zeros(pp1, nranef);
    // `pp1 × nranef` column-major: (col, k) -> k * pp1 + col.
    let out = result.as_mut_slice();
    let wtz = re.wtz.as_slice();
    let x = wtxy.as_slice();
    let n = wtxy.nrows();
    // Observation-outer: when rows are sorted by group, a column-outer pass
    // hits the same output cell on consecutive iterations and serializes on
    // the load/store chain. Each cell still sums in increasing observation
    // order, so the result is unchanged.
    for (obs, (&ref_idx, z)) in re.refs.iter().zip(wtz.chunks_exact(vsize)).enumerate() {
        let level_base = ref_idx as usize * vsize * pp1;
        for col in 0..pp1 {
            let x_value = x[col * n + obs];
            let base = level_base + col;
            for (s, &z_value) in z.iter().enumerate() {
                out[base + s * pp1] += x_value * z_value;
            }
        }
    }
    result
}

/// `[X|y]' [X|y]` straight from the already-weighted `[X|y]` matrix.
///
/// Reproduces `compute_fixed_response_cross_product` on the weighted dense
/// design bit for bit: the `X'X` corner uses sequential column dots up to
/// `NALGEBRA_SEQUENTIAL_GEMM_MAX_DIM` columns and nalgebra's product above
/// it (on column views with the same memory layout as the owned weighted
/// matrix), `X'y` sequential dots, and `y'y` nalgebra's `dot`.
pub(super) fn compute_wtxy_cross_product(wtxy: &DMatrix<f64>) -> DMatrix<f64> {
    let pp1 = wtxy.ncols();
    let p = pp1.saturating_sub(1);
    let mut result = DMatrix::zeros(pp1, pp1);
    if pp1 == 0 {
        return result;
    }
    if p > crate::model::fixed_design::NALGEBRA_SEQUENTIAL_GEMM_MAX_DIM {
        let x = wtxy.columns(0, p);
        let xtx = x.transpose() * x;
        for row in 0..p {
            for col in 0..p {
                result[(row, col)] = xtx[(row, col)];
            }
        }
    } else {
        for i in 0..p {
            let xi = wtxy.column(i);
            let xi = xi.as_slice();
            for j in 0..p {
                let xj = wtxy.column(j);
                let xj = xj.as_slice();
                let mut acc = 0.0;
                for (&a, &b) in xi.iter().zip(xj) {
                    acc += a * b;
                }
                result[(i, j)] = acc;
            }
        }
    }
    let y = wtxy.column(p);
    let y_slice = y.as_slice();
    for col in 0..p {
        let x_col = wtxy.column(col);
        let mut acc = 0.0;
        for (&a, &b) in x_col.as_slice().iter().zip(y_slice) {
            acc += a * b;
        }
        result[(col, p)] = acc;
        result[(p, col)] = acc;
    }
    result[(p, p)] = y.dot(&y);
    result
}

/// `y'Z_j` for a single response vector: length `n_ranef`, accumulated in
/// observation order per cell (bit-identical to the matrix form
/// `compute_response_re_cross_product` with one column).
pub(super) fn response_re_cross_product_vec(y: &[f64], re: &ReMat) -> DVector<f64> {
    let vsize = re.vsize;
    let mut result = DVector::zeros(re.n_ranef());
    let out = result.as_mut_slice();
    let wtz = re.wtz.as_slice();
    for ((&ref_idx, &response), z) in re.refs.iter().zip(y).zip(wtz.chunks_exact(vsize)) {
        let base = ref_idx as usize * vsize;
        for (s, &z_value) in z.iter().enumerate() {
            out[base + s] += z_value * response;
        }
    }
    result
}

/// Compute `[X|y]' [X|y]` using fixed-design backend cross-products.
pub(super) fn compute_fixed_response_cross_product(
    fixed_design: &FixedDesign,
    y: &DVector<f64>,
) -> Result<DMatrix<f64>> {
    if y.len() != fixed_design.n_obs() {
        return Err(MixedModelError::DimensionMismatch(format!(
            "fixed-effect design has {} rows but response has {}",
            fixed_design.n_obs(),
            y.len()
        )));
    }

    let p = fixed_design.n_cols();
    let xtx = fixed_design.xtx();
    let xty = fixed_design.xty(y)?;
    let mut result = DMatrix::zeros(p + 1, p + 1);
    for row in 0..p {
        for col in 0..p {
            result[(row, col)] = xtx[(row, col)];
        }
        result[(row, p)] = xty[row];
        result[(p, row)] = xty[row];
    }
    result[(p, p)] = y.dot(y);
    Ok(result)
}

/// Compute X' Z_j.
pub(super) fn compute_x_re_cross_product(x: &DMatrix<f64>, re: &ReMat) -> MatrixBlock {
    let p = x.ncols();
    let nranef = re.n_ranef();
    let vsize = re.vsize;

    let mut result = DMatrix::zeros(p, nranef);
    // `p × nranef` column-major: (col, k) -> k * p + col. Each cell is
    // accumulated in increasing observation order, as before.
    let out = result.as_mut_slice();
    let wtz = re.wtz.as_slice();
    for col in 0..p {
        let x_col = x.column(col);
        let x_col = x_col.as_slice();
        for ((&ref_idx, &x_value), z) in re.refs.iter().zip(x_col).zip(wtz.chunks_exact(vsize)) {
            let base = ref_idx as usize * vsize * p + col;
            for (s, &z_value) in z.iter().enumerate() {
                out[base + s * p] += x_value * z_value;
            }
        }
    }

    MatrixBlock::Dense(result)
}

pub(super) fn apply_lambda_transpose_to_rhs(rhs: &mut DMatrix<f64>, re: &ReMat) {
    let s = re.vsize;
    let nlevels = re.n_levels();
    let q = rhs.ncols();

    if s == 1 {
        let lam = re.lambda[(0, 0)];
        for row in 0..rhs.nrows() {
            for col in 0..q {
                rhs[(row, col)] *= lam;
            }
        }
        return;
    }

    if s == 2 {
        let l00 = re.lambda[(0, 0)];
        let l10 = re.lambda[(1, 0)];
        let l11 = re.lambda[(1, 1)];
        for level in 0..nlevels {
            let row0 = level * 2;
            let row1 = row0 + 1;
            for col in 0..q {
                let x0 = rhs[(row0, col)];
                let x1 = rhs[(row1, col)];
                rhs[(row0, col)] = l00 * x0 + l10 * x1;
                rhs[(row1, col)] = l11 * x1;
            }
        }
        return;
    }

    for level in 0..nlevels {
        let offset = level * s;
        let mut temp = vec![0.0; s];
        for col in 0..q {
            for (row, temp_row) in temp.iter_mut().enumerate() {
                let mut sum = 0.0;
                for inner in row..s {
                    sum += re.lambda[(inner, row)] * rhs[(offset + inner, col)];
                }
                *temp_row = sum;
            }
            for row in 0..s {
                rhs[(offset + row, col)] = temp[row];
            }
        }
    }
}

pub(super) fn subtract_left_block_product(
    dst: &mut DMatrix<f64>,
    lhs: &MatrixBlock,
    rhs: &DMatrix<f64>,
) {
    match lhs {
        MatrixBlock::Diagonal(diag) => {
            for row in 0..diag.len() {
                let scale = diag[row];
                for col in 0..rhs.ncols() {
                    dst[(row, col)] -= scale * rhs[(row, col)];
                }
            }
        }
        MatrixBlock::BlockDiagonal(blocks) => {
            let mut row_offset = 0;
            for block in blocks {
                let s = block.nrows();
                for row in 0..s {
                    for col in 0..rhs.ncols() {
                        let mut sum = 0.0;
                        for inner in 0..s {
                            sum += block[(row, inner)] * rhs[(row_offset + inner, col)];
                        }
                        dst[(row_offset + row, col)] -= sum;
                    }
                }
                row_offset += s;
            }
        }
        MatrixBlock::Sparse(mat) => {
            for (row, inner, value) in mat.triplet_iter() {
                for col in 0..rhs.ncols() {
                    dst[(row, col)] -= value * rhs[(inner, col)];
                }
            }
        }
        MatrixBlock::Dense(mat) => {
            for row in 0..mat.nrows() {
                for col in 0..rhs.ncols() {
                    let mut sum = 0.0;
                    for inner in 0..mat.ncols() {
                        sum += mat[(row, inner)] * rhs[(inner, col)];
                    }
                    dst[(row, col)] -= sum;
                }
            }
        }
    }
}

pub(super) fn solve_lower_block_against_rhs(l: &MatrixBlock, rhs: &mut [f64]) {
    debug_assert_eq!(l.nrows(), rhs.len());
    debug_assert_eq!(l.ncols(), rhs.len());

    match l {
        MatrixBlock::Diagonal(diag) => {
            for row in 0..diag.len() {
                let denom = diag[row];
                if denom.abs() < BLOCK_TRIANGULAR_SOLVE_ZERO_TOLERANCE {
                    rhs[row] = 0.0;
                    continue;
                }
                rhs[row] /= denom;
            }
        }
        MatrixBlock::BlockDiagonal(blocks) => {
            let mut row_offset = 0;
            for block in blocks {
                let s = block.nrows();
                for row in 0..s {
                    let diag = block[(row, row)];
                    if diag.abs() < BLOCK_TRIANGULAR_SOLVE_ZERO_TOLERANCE {
                        rhs[row_offset + row] = 0.0;
                        continue;
                    }
                    let mut sum = rhs[row_offset + row];
                    for inner in 0..row {
                        sum -= block[(row, inner)] * rhs[row_offset + inner];
                    }
                    rhs[row_offset + row] = sum / diag;
                }
                row_offset += s;
            }
        }
        MatrixBlock::Dense(mat) => {
            for row in 0..mat.nrows() {
                let diag = mat[(row, row)];
                if diag.abs() < BLOCK_TRIANGULAR_SOLVE_ZERO_TOLERANCE {
                    rhs[row] = 0.0;
                    continue;
                }
                let mut sum = rhs[row];
                for inner in 0..row {
                    sum -= mat[(row, inner)] * rhs[inner];
                }
                rhs[row] = sum / diag;
            }
        }
        MatrixBlock::Sparse(_) => {
            let dense = l.as_dense();
            solve_lower_block_against_rhs(&MatrixBlock::Dense(dense), rhs);
        }
    }
}

pub(super) fn solve_upper_block_from_lower_transpose_against_rhs(l: &MatrixBlock, rhs: &mut [f64]) {
    debug_assert_eq!(l.nrows(), rhs.len());
    debug_assert_eq!(l.ncols(), rhs.len());

    match l {
        MatrixBlock::Diagonal(diag) => {
            for row in (0..diag.len()).rev() {
                let denom = diag[row];
                if denom.abs() < BLOCK_TRIANGULAR_SOLVE_ZERO_TOLERANCE {
                    rhs[row] = 0.0;
                    continue;
                }
                rhs[row] /= denom;
            }
        }
        MatrixBlock::BlockDiagonal(blocks) => {
            let mut row_offset = 0;
            for block in blocks {
                let s = block.nrows();
                for row in (0..s).rev() {
                    let diag = block[(row, row)];
                    if diag.abs() < BLOCK_TRIANGULAR_SOLVE_ZERO_TOLERANCE {
                        rhs[row_offset + row] = 0.0;
                        continue;
                    }
                    let mut sum = rhs[row_offset + row];
                    for inner in (row + 1)..s {
                        sum -= block[(inner, row)] * rhs[row_offset + inner];
                    }
                    rhs[row_offset + row] = sum / diag;
                }
                row_offset += s;
            }
        }
        MatrixBlock::Dense(mat) => {
            for row in (0..mat.nrows()).rev() {
                let diag = mat[(row, row)];
                if diag.abs() < BLOCK_TRIANGULAR_SOLVE_ZERO_TOLERANCE {
                    rhs[row] = 0.0;
                    continue;
                }
                let mut sum = rhs[row];
                for inner in (row + 1)..mat.nrows() {
                    sum -= mat[(inner, row)] * rhs[inner];
                }
                rhs[row] = sum / diag;
            }
        }
        MatrixBlock::Sparse(_) => {
            let dense = l.as_dense();
            solve_upper_block_from_lower_transpose_against_rhs(&MatrixBlock::Dense(dense), rhs);
        }
    }
}

pub(super) fn solve_lower_block_rhs(rhs: &mut DMatrix<f64>, l: &MatrixBlock) {
    debug_assert_eq!(rhs.nrows(), l.nrows());

    let mut column_rhs = vec![0.0; rhs.nrows()];
    for col in 0..rhs.ncols() {
        for row in 0..rhs.nrows() {
            column_rhs[row] = rhs[(row, col)];
        }
        solve_lower_block_against_rhs(l, &mut column_rhs);
        for row in 0..rhs.nrows() {
            rhs[(row, col)] = column_rhs[row];
        }
    }
}

pub(super) fn solve_upper_from_lower_transpose(
    l: &DMatrix<f64>,
    rhs: &DMatrix<f64>,
) -> DMatrix<f64> {
    let p = l.nrows();
    let q = rhs.ncols();
    let mut result = rhs.clone();

    for col in 0..q {
        for row in (0..p).rev() {
            let mut sum = result[(row, col)];
            for inner in (row + 1)..p {
                sum -= l[(inner, row)] * result[(inner, col)];
            }
            result[(row, col)] = sum / l[(row, row)];
        }
    }

    result
}

pub(super) fn response_column_sums_of_squares(y: &DMatrix<f64>) -> DVector<f64> {
    let mut sums = DVector::zeros(y.ncols());
    for col in 0..y.ncols() {
        let mut sum = 0.0;
        for row in 0..y.nrows() {
            let value = y[(row, col)];
            sum += value * value;
        }
        sums[col] = sum;
    }
    sums
}

/// Reusable buffers for `profile_response_matrix_with_scratch`: the
/// per-term response right-hand sides, the unscaled `Z_jᵀY` copies (kept for
/// the batch gradient oracle), the `XᵀY` block, and the column buffer of
/// the blocked forward solve. Sized on first use and regrown only when a
/// wider chunk arrives, so a θ evaluation over a chunk allocates nothing
/// beyond its output vectors.
#[derive(Debug, Clone, Default)]
pub(crate) struct ProfileScratch {
    rhs: Vec<DMatrix<f64>>,
    zty: Vec<DMatrix<f64>>,
    xty: DMatrix<f64>,
    column: Vec<f64>,
}

impl ProfileScratch {
    /// Unscaled `Z_jᵀY` blocks from the last profile call (per term).
    pub(crate) fn response_re_cross_products(&self) -> &[DMatrix<f64>] {
        &self.zty
    }

    fn prepare(&mut self, reterms: &[ReMat], p: usize, q: usize) {
        let k = reterms.len();
        if self.rhs.len() != k + 1 || self.zty.len() != k {
            self.rhs = reterms
                .iter()
                .map(|re| DMatrix::zeros(re.n_ranef(), q))
                .chain(std::iter::once(DMatrix::zeros(p, q)))
                .collect();
            self.zty = reterms
                .iter()
                .map(|re| DMatrix::zeros(re.n_ranef(), q))
                .collect();
        } else {
            for (block, re) in self.rhs.iter_mut().zip(reterms) {
                if block.nrows() != re.n_ranef() || block.ncols() != q {
                    *block = DMatrix::zeros(re.n_ranef(), q);
                }
            }
            let last = &mut self.rhs[k];
            if last.nrows() != p || last.ncols() != q {
                *last = DMatrix::zeros(p, q);
            }
            for (block, re) in self.zty.iter_mut().zip(reterms) {
                if block.nrows() != re.n_ranef() || block.ncols() != q {
                    *block = DMatrix::zeros(re.n_ranef(), q);
                }
            }
        }
        if self.xty.nrows() != p || self.xty.ncols() != q {
            self.xty = DMatrix::zeros(p, q);
        }
        let longest = reterms
            .iter()
            .map(|re| re.n_ranef())
            .max()
            .unwrap_or(0)
            .max(p);
        if self.column.len() < longest {
            self.column.resize(longest, 0.0);
        }
    }
}

/// `dst = Zᵀ Y` for one term, written in place (same accumulation order as
/// `compute_response_re_cross_product`).
pub(super) fn compute_response_re_cross_product_into(
    dst: &mut DMatrix<f64>,
    y: &DMatrix<f64>,
    re: &ReMat,
) {
    let q = y.ncols();
    let n = re.refs.len();
    dst.fill(0.0);
    for obs in 0..n {
        let r = re.refs[obs] as usize;
        for s in 0..re.vsize {
            let row = r * re.vsize + s;
            let weight = re.wtz[(s, obs)];
            for col in 0..q {
                dst[(row, col)] += weight * y[(obs, col)];
            }
        }
    }
}

fn solve_lower_block_rhs_with_column(rhs: &mut DMatrix<f64>, l: &MatrixBlock, column: &mut [f64]) {
    let rows = rhs.nrows();
    let column = &mut column[..rows];
    for col in 0..rhs.ncols() {
        for row in 0..rows {
            column[row] = rhs[(row, col)];
        }
        solve_lower_block_against_rhs(l, column);
        for row in 0..rows {
            rhs[(row, col)] = column[row];
        }
    }
}

/// `profile_response_matrix_with_l_blocks` on caller-owned scratch.
#[allow(clippy::too_many_arguments)]
pub(crate) fn profile_response_matrix_with_scratch(
    reterms: &[ReMat],
    x: &DMatrix<f64>,
    responses: &DMatrix<f64>,
    l_blocks: &[MatrixBlock],
    reml: bool,
    n: usize,
    p: usize,
    scratch: &mut ProfileScratch,
) -> Result<ResponseMatrixProfile> {
    if responses.nrows() != x.nrows() {
        return Err(MixedModelError::DimensionMismatch(format!(
            "response matrix has {} rows, but design matrix has {}",
            responses.nrows(),
            x.nrows()
        )));
    }
    let k = reterms.len();
    let q = responses.ncols();
    let total = k + 1;
    let expected_blocks = total * (total + 1) / 2;
    if l_blocks.len() != expected_blocks {
        return Err(MixedModelError::DimensionMismatch(format!(
            "blocked factor has {} blocks, expected {}",
            l_blocks.len(),
            expected_blocks
        )));
    }

    scratch.prepare(reterms, p, q);
    for (j, re) in reterms.iter().enumerate() {
        compute_response_re_cross_product_into(&mut scratch.zty[j], responses, re);
        scratch.rhs[j].copy_from(&scratch.zty[j]);
        apply_lambda_transpose_to_rhs(&mut scratch.rhs[j], re);
    }
    x.tr_mul_to(responses, &mut scratch.xty);
    scratch.rhs[k].copy_from(&scratch.xty);

    // Blocked forward solve on the shared column buffer.
    for row_block in 0..=k {
        let (solved, current_and_after) = scratch.rhs.split_at_mut(row_block);
        let current = &mut current_and_after[0];
        for (prev, solved_prev) in solved.iter().enumerate() {
            subtract_left_block_product(
                current,
                &l_blocks[block_index(row_block, prev)],
                solved_prev,
            );
        }
        solve_lower_block_rhs_with_column(
            current,
            &l_blocks[block_index(row_block, row_block)],
            &mut scratch.column,
        );
    }

    let mut solved_norm_sq = DVector::<f64>::zeros(q);
    for block in &scratch.rhs {
        for col in 0..q {
            let mut sum = 0.0;
            for row in 0..block.nrows() {
                let value = block[(row, col)];
                sum += value * value;
            }
            solved_norm_sq[col] += sum;
        }
    }

    let response_ss = response_column_sums_of_squares(responses);
    let mut pwrss = DVector::zeros(q);
    for col in 0..q {
        let residual = response_ss[col] - solved_norm_sq[col];
        pwrss[col] = if residual < 0.0 && residual > -1e-10 {
            0.0
        } else {
            residual
        };
    }

    let x_block = &l_blocks[block_index(k, k)];
    let beta = match x_block {
        MatrixBlock::Dense(l_xx) => solve_upper_from_lower_transpose(l_xx, &scratch.rhs[k]),
        _ => {
            let l_xx = x_block.as_dense();
            solve_upper_from_lower_transpose(&l_xx, &scratch.rhs[k])
        }
    };

    let mut logdet_re = 0.0;
    for j in 0..k {
        logdet_re += logdet_block(&l_blocks[block_index(j, j)]);
    }
    let logdet_xx = logdet_block(x_block);

    let denom = if reml {
        n.checked_sub(p).ok_or_else(|| {
            MixedModelError::DimensionMismatch(format!(
                "REML requires n >= p, got n={} and p={}",
                n, p
            ))
        })?
    } else {
        n
    };
    if denom == 0 {
        return Err(MixedModelError::DimensionMismatch(
            "profile denominator must be positive".to_string(),
        ));
    }
    let denom_f = denom as f64;
    let constant = 2.0 * std::f64::consts::PI / denom_f;

    let mut sigma = DVector::zeros(q);
    let mut objectives = DVector::zeros(q);
    let mut total_objective = 0.0;
    for col in 0..q {
        sigma[col] = (pwrss[col] / denom_f).sqrt();
        let mut objective = logdet_re + denom_f * (1.0 + (constant * pwrss[col]).ln());
        if reml {
            objective += logdet_xx;
        }
        objectives[col] = objective;
        total_objective += objective;
    }

    Ok(ResponseMatrixProfile {
        beta,
        sigma,
        pwrss,
        objectives,
        total_objective,
        logdet_re,
        logdet_xx,
    })
}

// === Block Cholesky helper functions ===

/// Copy A to L and scale blockwise: L_jj = Λ_j' A_jj Λ_j + I
///
/// A is (nranef × nranef) where nranef = vsize * nlevels.
/// Λ is (vsize × vsize). Scaling is applied to each (vsize × vsize)
/// sub-block of A independently.
pub(super) fn copy_scale_inflate(l: &mut MatrixBlock, a: &MatrixBlock, re: &ReMat) {
    let s = re.vsize;

    if s == 1 {
        // Scalar RE
        let lam = re.lambda[(0, 0)];
        let lam_sq = lam * lam;
        match (l, a) {
            (MatrixBlock::Diagonal(l_diag), MatrixBlock::Diagonal(a_diag)) => {
                for i in 0..l_diag.len() {
                    l_diag[i] = lam_sq * a_diag[i] + 1.0;
                }
            }
            (l_block, _) => with_dense_block(a, |a_dense| {
                let n = a_dense.nrows();
                let result = match l_block {
                    MatrixBlock::Dense(result) if result.shape() == (n, n) => result,
                    _ => {
                        *l_block = MatrixBlock::Dense(DMatrix::zeros(n, n));
                        match l_block {
                            MatrixBlock::Dense(result) => result,
                            _ => unreachable!(),
                        }
                    }
                };
                for i in 0..n {
                    for j in 0..n {
                        result[(i, j)] = lam_sq * a_dense[(i, j)];
                    }
                    result[(i, i)] += 1.0;
                }
            }),
        }
    } else {
        // Vector RE: apply Λ blockwise
        let lambda = &re.lambda;

        match a {
            MatrixBlock::BlockDiagonal(a_blocks) => {
                if matches!(l, MatrixBlock::Dense(_)) {
                    let nranef = a_blocks.iter().map(|blk| blk.nrows()).sum();
                    let result = match l {
                        MatrixBlock::Dense(result) if result.shape() == (nranef, nranef) => result,
                        _ => {
                            *l = MatrixBlock::Dense(DMatrix::zeros(nranef, nranef));
                            match l {
                                MatrixBlock::Dense(result) => result,
                                _ => unreachable!(),
                            }
                        }
                    };

                    if s == 2 {
                        let l00 = lambda[(0, 0)];
                        let l01 = lambda[(0, 1)];
                        let l10 = lambda[(1, 0)];
                        let l11 = lambda[(1, 1)];

                        result.fill(0.0);
                        for (level, src_blk) in a_blocks.iter().enumerate() {
                            let row0 = level * 2;
                            let row1 = row0 + 1;

                            let s00 = src_blk[(0, 0)];
                            let s01 = src_blk[(0, 1)];
                            let s10 = src_blk[(1, 0)];
                            let s11 = src_blk[(1, 1)];

                            let t00 = s00 * l00 + s01 * l10;
                            let t01 = s00 * l01 + s01 * l11;
                            let t10 = s10 * l00 + s11 * l10;
                            let t11 = s10 * l01 + s11 * l11;

                            result[(row0, row0)] = l00 * t00 + l10 * t10 + 1.0;
                            result[(row0, row1)] = l00 * t01 + l10 * t11;
                            result[(row1, row0)] = l01 * t00 + l11 * t10;
                            result[(row1, row1)] = l01 * t01 + l11 * t11 + 1.0;
                        }
                        return;
                    }

                    result.fill(0.0);
                    for (level, src_blk) in a_blocks.iter().enumerate() {
                        for row in 0..s {
                            for col in 0..s {
                                let mut sum = 0.0;
                                for inner_row in 0..s {
                                    for inner_col in 0..s {
                                        sum += lambda[(inner_row, row)]
                                            * src_blk[(inner_row, inner_col)]
                                            * lambda[(inner_col, col)];
                                    }
                                }
                                result[(level * s + row, level * s + col)] = sum;
                            }
                            result[(level * s + row, level * s + row)] += 1.0;
                        }
                    }
                    return;
                }

                let l_blocks = match l {
                    MatrixBlock::BlockDiagonal(l_blocks) => {
                        let shapes_match = l_blocks.len() == a_blocks.len()
                            && l_blocks
                                .iter()
                                .zip(a_blocks.iter())
                                .all(|(dst, src)| dst.shape() == src.shape());
                        if !shapes_match {
                            *l_blocks = a_blocks
                                .iter()
                                .map(|blk| DMatrix::zeros(blk.nrows(), blk.ncols()))
                                .collect();
                        }
                        l_blocks
                    }
                    _ => {
                        *l = MatrixBlock::BlockDiagonal(
                            a_blocks
                                .iter()
                                .map(|blk| DMatrix::zeros(blk.nrows(), blk.ncols()))
                                .collect(),
                        );
                        match l {
                            MatrixBlock::BlockDiagonal(l_blocks) => l_blocks,
                            _ => unreachable!(),
                        }
                    }
                };

                if s == 2 {
                    let l00 = lambda[(0, 0)];
                    let l01 = lambda[(0, 1)];
                    let l10 = lambda[(1, 0)];
                    let l11 = lambda[(1, 1)];

                    for (dst_blk, src_blk) in l_blocks.iter_mut().zip(a_blocks.iter()) {
                        let s00 = src_blk[(0, 0)];
                        let s01 = src_blk[(0, 1)];
                        let s10 = src_blk[(1, 0)];
                        let s11 = src_blk[(1, 1)];

                        let t00 = s00 * l00 + s01 * l10;
                        let t01 = s00 * l01 + s01 * l11;
                        let t10 = s10 * l00 + s11 * l10;
                        let t11 = s10 * l01 + s11 * l11;

                        dst_blk[(0, 0)] = l00 * t00 + l10 * t10 + 1.0;
                        dst_blk[(0, 1)] = l00 * t01 + l10 * t11;
                        dst_blk[(1, 0)] = l01 * t00 + l11 * t10;
                        dst_blk[(1, 1)] = l01 * t01 + l11 * t11 + 1.0;
                    }
                    return;
                }

                for (dst_blk, src_blk) in l_blocks.iter_mut().zip(a_blocks.iter()) {
                    for row in 0..s {
                        for col in 0..s {
                            let mut sum = 0.0;
                            for inner_row in 0..s {
                                for inner_col in 0..s {
                                    sum += lambda[(inner_row, row)]
                                        * src_blk[(inner_row, inner_col)]
                                        * lambda[(inner_col, col)];
                                }
                            }
                            dst_blk[(row, col)] = sum;
                        }
                        dst_blk[(row, row)] += 1.0;
                    }
                }
            }
            _ => {
                // Dense fallback: apply Λ blockwise to each (s×s) sub-block
                with_dense_block(a, |a_dense| {
                    let nranef = a_dense.nrows();
                    let nlevels = nranef / s;
                    let result = match l {
                        MatrixBlock::Dense(result) if result.shape() == (nranef, nranef) => result,
                        _ => {
                            *l = MatrixBlock::Dense(DMatrix::zeros(nranef, nranef));
                            match l {
                                MatrixBlock::Dense(result) => result,
                                _ => unreachable!(),
                            }
                        }
                    };

                    for bk in 0..nlevels {
                        for bl in 0..nlevels {
                            for row in 0..s {
                                for col in 0..s {
                                    let mut sum = 0.0;
                                    for inner_row in 0..s {
                                        for inner_col in 0..s {
                                            sum += lambda[(inner_row, row)]
                                                * a_dense[(bk * s + inner_row, bl * s + inner_col)]
                                                * lambda[(inner_col, col)];
                                        }
                                    }
                                    result[(bk * s + row, bl * s + col)] = sum;
                                }
                            }
                        }
                    }
                    for i in 0..nranef {
                        result[(i, i)] += 1.0;
                    }
                })
            }
        }
    }
}

/// Copy off-diagonal block and scale blockwise: L_ij = Λ_i' A_ij Λ_j
///
/// A is (nranef_i × nranef_j). Λ_i is (vsize_i × vsize_i), Λ_j is (vsize_j × vsize_j).
pub(super) fn copy_and_scale_offdiag(
    l: &mut MatrixBlock,
    a: &MatrixBlock,
    re_i: &ReMat,
    re_j: &ReMat,
) {
    let si = re_i.vsize;
    let sj = re_j.vsize;

    if si == 1 && sj == 1 {
        let scale = re_i.lambda[(0, 0)] * re_j.lambda[(0, 0)];
        if let MatrixBlock::Sparse(a_sparse) = a {
            // An L block the factorization promoted to dense stays dense:
            // scatter the scaled sparse values into its buffer rather than
            // reverting it to sparse (and re-promoting it) every evaluation.
            if let MatrixBlock::Dense(dense) = l {
                if dense.shape() == (a_sparse.nrows(), a_sparse.ncols()) {
                    fill_dense_from_block(dense, a, scale);
                    return;
                }
            }
            let result = match l {
                MatrixBlock::Sparse(result)
                    if result.nrows() == a_sparse.nrows()
                        && result.ncols() == a_sparse.ncols()
                        && result.nnz() == a_sparse.nnz()
                        && result.col_offsets() == a_sparse.col_offsets()
                        && result.row_indices() == a_sparse.row_indices() =>
                {
                    result
                }
                _ => {
                    *l = MatrixBlock::Sparse(a_sparse.clone());
                    match l {
                        MatrixBlock::Sparse(result) => result,
                        _ => unreachable!(),
                    }
                }
            };
            result.values_mut().copy_from_slice(a_sparse.values());
            for value in result.values_mut() {
                *value *= scale;
            }
            return;
        }
    }

    with_dense_block(a, |a_dense| {
        let nranef_i = a_dense.nrows();
        let nranef_j = a_dense.ncols();
        let nlevels_i = nranef_i / si;
        let nlevels_j = nranef_j / sj;
        let lambda_j = &re_j.lambda;
        let result = match l {
            MatrixBlock::Dense(result) if result.shape() == (nranef_i, nranef_j) => result,
            _ => {
                *l = MatrixBlock::Dense(DMatrix::zeros(nranef_i, nranef_j));
                match l {
                    MatrixBlock::Dense(result) => result,
                    _ => unreachable!(),
                }
            }
        };

        if si == 2 && sj == 2 {
            let li00 = re_i.lambda[(0, 0)];
            let li01 = re_i.lambda[(0, 1)];
            let li10 = re_i.lambda[(1, 0)];
            let li11 = re_i.lambda[(1, 1)];
            let lj00 = lambda_j[(0, 0)];
            let lj01 = lambda_j[(0, 1)];
            let lj10 = lambda_j[(1, 0)];
            let lj11 = lambda_j[(1, 1)];

            for bi in 0..nlevels_i {
                let row0 = bi * 2;
                let row1 = row0 + 1;
                for bj in 0..nlevels_j {
                    let col0 = bj * 2;
                    let col1 = col0 + 1;
                    let a00 = a_dense[(row0, col0)];
                    let a01 = a_dense[(row0, col1)];
                    let a10 = a_dense[(row1, col0)];
                    let a11 = a_dense[(row1, col1)];

                    let t00 = a00 * lj00 + a01 * lj10;
                    let t01 = a00 * lj01 + a01 * lj11;
                    let t10 = a10 * lj00 + a11 * lj10;
                    let t11 = a10 * lj01 + a11 * lj11;

                    result[(row0, col0)] = li00 * t00 + li10 * t10;
                    result[(row0, col1)] = li00 * t01 + li10 * t11;
                    result[(row1, col0)] = li01 * t00 + li11 * t10;
                    result[(row1, col1)] = li01 * t01 + li11 * t11;
                }
            }
            return;
        }

        for bi in 0..nlevels_i {
            for bj in 0..nlevels_j {
                for row in 0..si {
                    for col in 0..sj {
                        let mut sum = 0.0;
                        for inner_row in 0..si {
                            for inner_col in 0..sj {
                                sum += re_i.lambda[(inner_row, row)]
                                    * a_dense[(bi * si + inner_row, bj * sj + inner_col)]
                                    * lambda_j[(inner_col, col)];
                            }
                        }
                        result[(bi * si + row, bj * sj + col)] = sum;
                    }
                }
            }
        }
    });
}

/// Copy and right-multiply blockwise by Λ: L_kj = A_kj Λ_j
///
/// A is (pp1 × nranef_j). Λ_j is (vsize_j × vsize_j).
/// Each column-block of size vsize_j gets right-multiplied by Λ_j.
pub(super) fn copy_and_rmul_lambda(l: &mut MatrixBlock, a: &MatrixBlock, re_j: &ReMat) {
    let sj = re_j.vsize;
    if sj == 1 {
        let lam = re_j.lambda[(0, 0)];
        match a {
            MatrixBlock::Dense(a_dense) => {
                let nrows = a_dense.nrows();
                let ncols = a_dense.ncols();
                let result = match l {
                    MatrixBlock::Dense(result) if result.shape() == (nrows, ncols) => result,
                    _ => {
                        *l = MatrixBlock::Dense(DMatrix::zeros(nrows, ncols));
                        match l {
                            MatrixBlock::Dense(result) => result,
                            _ => unreachable!(),
                        }
                    }
                };

                for i in 0..nrows {
                    for j in 0..ncols {
                        result[(i, j)] = a_dense[(i, j)] * lam;
                    }
                }
                return;
            }
            MatrixBlock::Sparse(a_sparse) => {
                // Scalar λ scale of a sparse [X|y]'Z block: keep the sparse
                // structure and reuse the L buffer when it already matches.
                // A dense L buffer (promoted by an earlier factorization)
                // is filled in place instead of reverted to sparse.
                if let MatrixBlock::Dense(dense) = l {
                    if dense.shape() == (a_sparse.nrows(), a_sparse.ncols()) {
                        fill_dense_from_block(dense, a, lam);
                        return;
                    }
                }
                let result = match l {
                    MatrixBlock::Sparse(result)
                        if result.nrows() == a_sparse.nrows()
                            && result.ncols() == a_sparse.ncols()
                            && result.nnz() == a_sparse.nnz()
                            && result.col_offsets() == a_sparse.col_offsets()
                            && result.row_indices() == a_sparse.row_indices() =>
                    {
                        result
                    }
                    _ => {
                        *l = MatrixBlock::Sparse(a_sparse.clone());
                        match l {
                            MatrixBlock::Sparse(result) => result,
                            _ => unreachable!(),
                        }
                    }
                };
                for (dst, src) in result.values_mut().iter_mut().zip(a_sparse.values()) {
                    *dst = src * lam;
                }
                return;
            }
            _ => {
                let a_dense = a.as_dense();
                let nrows = a_dense.nrows();
                let ncols = a_dense.ncols();
                let result = match l {
                    MatrixBlock::Dense(result) if result.shape() == (nrows, ncols) => result,
                    _ => {
                        *l = MatrixBlock::Dense(DMatrix::zeros(nrows, ncols));
                        match l {
                            MatrixBlock::Dense(result) => result,
                            _ => unreachable!(),
                        }
                    }
                };

                for i in 0..nrows {
                    for j in 0..ncols {
                        result[(i, j)] = a_dense[(i, j)] * lam;
                    }
                }
                return;
            }
        }
    }

    with_dense_block(a, |a_dense| {
        let nrows = a_dense.nrows();
        let ncols = a_dense.ncols();
        let nblocks = ncols / sj;
        let lambda_j = &re_j.lambda;
        let result = match l {
            MatrixBlock::Dense(result) if result.shape() == (nrows, ncols) => result,
            _ => {
                *l = MatrixBlock::Dense(DMatrix::zeros(nrows, ncols));
                match l {
                    MatrixBlock::Dense(result) => result,
                    _ => unreachable!(),
                }
            }
        };

        if sj == 2 {
            let l00 = lambda_j[(0, 0)];
            let l01 = lambda_j[(0, 1)];
            let l10 = lambda_j[(1, 0)];
            let l11 = lambda_j[(1, 1)];

            for b in 0..nblocks {
                let col0 = b * 2;
                let col1 = col0 + 1;
                for i in 0..nrows {
                    let x0 = a_dense[(i, col0)];
                    let x1 = a_dense[(i, col1)];
                    result[(i, col0)] = x0 * l00 + x1 * l10;
                    result[(i, col1)] = x0 * l01 + x1 * l11;
                }
            }
            return;
        }

        for b in 0..nblocks {
            for i in 0..nrows {
                for j in 0..sj {
                    let mut sum = 0.0;
                    for inner in 0..sj {
                        sum += a_dense[(i, b * sj + inner)] * lambda_j[(inner, j)];
                    }
                    result[(i, b * sj + j)] = sum;
                }
            }
        }
    });
}

/// Zero-copy transposed view of a dense column-major matrix, so gemm
/// operands of the form `A * Bᵀ` need not materialize `Bᵀ` on every
/// objective evaluation.
#[cfg(not(feature = "faer-backend"))]
pub(super) fn transposed_view(m: &DMatrix<f64>) -> nalgebra::DMatrixView<'_, f64, nalgebra::Dyn> {
    let (nrows, ncols) = m.shape();
    nalgebra::DMatrixView::from_slice_with_strides_generic(
        m.as_slice(),
        nalgebra::Dyn(ncols),
        nalgebra::Dyn(nrows),
        nalgebra::Dyn(nrows),
        nalgebra::Dyn(1),
    )
}

/// Zero-copy transpose of `m.rows(row_offset, block_rows)`.
#[cfg(not(feature = "faer-backend"))]
pub(super) fn transposed_rows_view(
    m: &DMatrix<f64>,
    row_offset: usize,
    block_rows: usize,
) -> nalgebra::DMatrixView<'_, f64, nalgebra::Dyn> {
    debug_assert!(row_offset + block_rows <= m.nrows());
    nalgebra::DMatrixView::from_slice_with_strides_generic(
        &m.as_slice()[row_offset..],
        nalgebra::Dyn(m.ncols()),
        nalgebra::Dyn(block_rows),
        nalgebra::Dyn(m.nrows()),
        nalgebra::Dyn(1),
    )
}

/// `C -= A * Bᵀ`, dispatched to the compiled gemm backend. The default
/// backend is nalgebra (matrixmultiply); the experimental `faer-backend`
/// feature routes the product through faer's matmul instead.
#[cfg(not(feature = "faer-backend"))]
pub(super) fn gemm_sub_abt(c: &mut DMatrix<f64>, a: &DMatrix<f64>, b: &DMatrix<f64>) {
    c.gemm(-1.0, a, &transposed_view(b), 1.0);
}

/// `C -= A * Bᵀ` through faer's sequential matmul over zero-copy views of
/// the nalgebra column-major storage.
#[cfg(feature = "faer-backend")]
pub(super) fn gemm_sub_abt(c: &mut DMatrix<f64>, a: &DMatrix<f64>, b: &DMatrix<f64>) {
    let (m, k) = a.shape();
    let n = b.nrows();
    debug_assert_eq!(b.ncols(), k);
    debug_assert_eq!(c.shape(), (m, n));
    let a_ref = faer::MatRef::from_column_major_slice(a.as_slice(), m, k);
    let b_ref = faer::MatRef::from_column_major_slice(b.as_slice(), n, k);
    let c_mut = faer::MatMut::from_column_major_slice_mut(c.as_mut_slice(), m, n);
    faer::linalg::matmul::matmul(
        c_mut,
        faer::Accum::Add,
        a_ref,
        b_ref.transpose(),
        -1.0,
        faer::Par::Seq,
    );
}

/// `C -= A[rows] * A[rows]ᵀ` over `block_rows` rows starting at `row_offset`.
#[cfg(not(feature = "faer-backend"))]
pub(super) fn gemm_sub_rows_aat(
    c: &mut DMatrix<f64>,
    a: &DMatrix<f64>,
    row_offset: usize,
    block_rows: usize,
) {
    let a_block = a.rows(row_offset, block_rows);
    c.gemm(
        -1.0,
        &a_block,
        &transposed_rows_view(a, row_offset, block_rows),
        1.0,
    );
}

#[cfg(feature = "faer-backend")]
pub(super) fn gemm_sub_rows_aat(
    c: &mut DMatrix<f64>,
    a: &DMatrix<f64>,
    row_offset: usize,
    block_rows: usize,
) {
    debug_assert!(row_offset + block_rows <= a.nrows());
    if a.ncols() == 0 {
        // Zero-column downdate is a no-op; the offset slice below would
        // panic on an empty backing slice where nalgebra's gemm no-ops.
        return;
    }
    let a_ref = faer::MatRef::from_column_major_slice_with_stride(
        &a.as_slice()[row_offset..],
        block_rows,
        a.ncols(),
        a.nrows(),
    );
    let (m, n) = (c.nrows(), c.ncols());
    let c_mut = faer::MatMut::from_column_major_slice_mut(c.as_mut_slice(), m, n);
    faer::linalg::matmul::matmul(
        c_mut,
        faer::Accum::Add,
        a_ref,
        a_ref.transpose(),
        -1.0,
        faer::Par::Seq,
    );
}

/// Rank-k downdate: C -= A * A' (modifies diagonal block)
pub(super) fn rank_k_downdate(c: &mut MatrixBlock, a: &DMatrix<f64>) {
    match c {
        MatrixBlock::Dense(c_mat) => {
            if c_mat.nrows() == c_mat.ncols()
                && c_mat.nrows() == a.nrows()
                && c_mat.nrows() <= 4
                && a.ncols() >= 512
            {
                rank_k_downdate_small_dense(c_mat, a);
            } else {
                gemm_sub_abt(c_mat, a, a);
            }
        }
        MatrixBlock::Diagonal(c_diag) => {
            // A * A' diagonal entries
            for i in 0..c_diag.len() {
                let row = a.row(i);
                c_diag[i] -= row.dot(&row);
            }
        }
        MatrixBlock::BlockDiagonal(blocks) => {
            if a.ncols() >= 512
                && a.nrows() == blocks.len() * 2
                && blocks.iter().all(|blk| blk.shape() == (2, 2))
            {
                rank_k_downdate_vsize2_blocks(blocks, a);
                return;
            }

            // For each block k, downdate by the corresponding rows of A
            let mut row_offset = 0;
            for blk in blocks.iter_mut() {
                let s = blk.nrows();
                gemm_sub_rows_aat(blk, a, row_offset, s);
                row_offset += s;
            }
        }
        MatrixBlock::Sparse(_) => {
            let mut dense = c.as_dense();
            gemm_sub_abt(&mut dense, a, a);
            *c = MatrixBlock::Dense(dense);
        }
    }
}

fn rank_k_downdate_small_dense(c: &mut DMatrix<f64>, a: &DMatrix<f64>) {
    debug_assert_eq!(c.nrows(), c.ncols());
    debug_assert_eq!(c.nrows(), a.nrows());
    debug_assert!(c.nrows() <= 4);

    let n = c.nrows();
    let mut sums = [0.0; 16];
    for k in 0..a.ncols() {
        for row in 0..n {
            let row_val = a[(row, k)];
            for col in 0..=row {
                sums[row * 4 + col] += row_val * a[(col, k)];
            }
        }
    }
    for row in 0..n {
        for col in 0..=row {
            let sum = sums[row * 4 + col];
            c[(row, col)] -= sum;
            if row != col {
                c[(col, row)] -= sum;
            }
        }
    }
}

fn rank_k_downdate_vsize2_blocks(blocks: &mut [DMatrix<f64>], a: &DMatrix<f64>) {
    debug_assert_eq!(a.nrows(), blocks.len() * 2);

    let mut row_offset = 0;
    for blk in blocks.iter_mut() {
        debug_assert_eq!(blk.shape(), (2, 2));
        let mut s00 = 0.0;
        let mut s10 = 0.0;
        let mut s11 = 0.0;
        for col in 0..a.ncols() {
            let a0 = a[(row_offset, col)];
            let a1 = a[(row_offset + 1, col)];
            s00 += a0 * a0;
            s10 += a1 * a0;
            s11 += a1 * a1;
        }
        blk[(0, 0)] -= s00;
        blk[(1, 0)] -= s10;
        blk[(0, 1)] -= s10;
        blk[(1, 1)] -= s11;
        row_offset += 2;
    }
}

/// Rank-k downdate from a sparse block: C -= A * A'.
pub(super) fn rank_k_downdate_sparse(c: &mut MatrixBlock, a: &CscMatrix<f64>) {
    match c {
        MatrixBlock::Dense(c_mat) => {
            for col_idx in 0..a.ncols() {
                let col = a.col(col_idx);
                let rows = col.row_indices();
                let values = col.values();
                for left in 0..rows.len() {
                    let row_i = rows[left];
                    let value_i = values[left];
                    for right in 0..rows.len() {
                        let row_j = rows[right];
                        c_mat[(row_i, row_j)] -= value_i * values[right];
                    }
                }
            }
        }
        MatrixBlock::Diagonal(c_diag) => {
            for (row, _, value) in a.triplet_iter() {
                c_diag[row] -= value * value;
            }
        }
        _ => {
            let mut dense = c.as_dense();
            for col_idx in 0..a.ncols() {
                let col = a.col(col_idx);
                let rows = col.row_indices();
                let values = col.values();
                for left in 0..rows.len() {
                    let row_i = rows[left];
                    let value_i = values[left];
                    for right in 0..rows.len() {
                        let row_j = rows[right];
                        dense[(row_i, row_j)] -= value_i * values[right];
                    }
                }
            }
            *c = MatrixBlock::Dense(dense);
        }
    }
}

/// Subtract product: C -= A * B'
pub(super) fn subtract_product(c: &mut MatrixBlock, a: &DMatrix<f64>, b: &DMatrix<f64>) {
    match c {
        MatrixBlock::Dense(c_mat) => {
            gemm_sub_abt(c_mat, a, b);
        }
        MatrixBlock::BlockDiagonal(_) => {
            // Promote to dense — off-diagonal updates destroy block-diagonal structure
            let mut c_dense = c.as_dense();
            gemm_sub_abt(&mut c_dense, a, b);
            *c = MatrixBlock::Dense(c_dense);
        }
        MatrixBlock::Sparse(_) => {
            let mut c_dense = c.as_dense();
            gemm_sub_abt(&mut c_dense, a, b);
            *c = MatrixBlock::Dense(c_dense);
        }
        _ => {
            let mut c_dense = c.as_dense();
            gemm_sub_abt(&mut c_dense, a, b);
            *c = MatrixBlock::Dense(c_dense);
        }
    }
}

/// In-place Cholesky of a block (handles zero diagonal gracefully).
#[cfg(test)]
pub(super) fn cholesky_block(block: &mut MatrixBlock) -> Result<()> {
    cholesky_block_with_tolerance(
        block,
        crate::compiler::policy::DEFAULT_CHOLESKY_ZERO_PAD_TOLERANCE,
    )
}

pub(super) fn cholesky_zero_pad_abs_tolerance(diagonal_scale: f64, relative_tolerance: f64) -> f64 {
    if !diagonal_scale.is_finite() || !relative_tolerance.is_finite() {
        return 0.0;
    }
    relative_tolerance.max(0.0) * diagonal_scale.max(0.0)
}

pub(super) fn diagonal_abs_max_matrix(mat: &DMatrix<f64>) -> f64 {
    (0..mat.nrows().min(mat.ncols()))
        .map(|idx| mat[(idx, idx)].abs())
        .fold(0.0_f64, f64::max)
}

pub(super) fn cholesky_block_with_tolerance(
    block: &mut MatrixBlock,
    cholesky_zero_pad_tolerance: f64,
) -> Result<()> {
    match block {
        MatrixBlock::Diagonal(diag) => {
            let tol = cholesky_zero_pad_abs_tolerance(
                diag.iter().map(|value| value.abs()).fold(0.0_f64, f64::max),
                cholesky_zero_pad_tolerance,
            );
            for i in 0..diag.len() {
                if diag[i] <= 0.0 {
                    if diag[i] < -tol {
                        return Err(MixedModelError::PosDefException);
                    }
                    diag[i] = 0.0;
                } else {
                    diag[i] = diag[i].sqrt();
                }
            }
            Ok(())
        }
        MatrixBlock::BlockDiagonal(blocks) => {
            // Cholesky each small block independently: O(nlevels * s³)
            if blocks.first().is_some_and(|blk| blk.nrows() == 2) {
                for blk in blocks.iter_mut() {
                    let tol = cholesky_zero_pad_abs_tolerance(
                        diagonal_abs_max_matrix(blk),
                        cholesky_zero_pad_tolerance,
                    );
                    let d00 = blk[(0, 0)];
                    if d00 <= 0.0 {
                        if d00 < -tol {
                            return Err(MixedModelError::PosDefException);
                        }
                        blk[(0, 0)] = 0.0;
                        blk[(1, 0)] = 0.0;
                    } else {
                        blk[(0, 0)] = d00.sqrt();
                        blk[(1, 0)] /= blk[(0, 0)];
                    }

                    let d11 = blk[(1, 1)] - blk[(1, 0)] * blk[(1, 0)];
                    if d11 <= 0.0 {
                        if d11 < -tol {
                            return Err(MixedModelError::PosDefException);
                        }
                        blk[(1, 1)] = 0.0;
                    } else {
                        blk[(1, 1)] = d11.sqrt();
                    }
                    blk[(0, 1)] = 0.0;
                }
                return Ok(());
            }

            for blk in blocks.iter_mut() {
                let n = blk.nrows();
                let tol = cholesky_zero_pad_abs_tolerance(
                    diagonal_abs_max_matrix(blk),
                    cholesky_zero_pad_tolerance,
                );
                for j in 0..n {
                    let mut s = blk[(j, j)];
                    for k in 0..j {
                        s -= blk[(j, k)] * blk[(j, k)];
                    }
                    if s <= 0.0 {
                        if s < -tol {
                            return Err(MixedModelError::PosDefException);
                        }
                        for i in j..n {
                            blk[(i, j)] = 0.0;
                        }
                        continue;
                    }
                    blk[(j, j)] = s.sqrt();
                    for i in (j + 1)..n {
                        let mut s = blk[(i, j)];
                        for k in 0..j {
                            s -= blk[(i, k)] * blk[(j, k)];
                        }
                        blk[(i, j)] = s / blk[(j, j)];
                    }
                    for i in 0..j {
                        blk[(i, j)] = 0.0;
                    }
                }
            }
            Ok(())
        }
        MatrixBlock::Dense(mat) => {
            let n = mat.nrows();
            let tol = cholesky_zero_pad_abs_tolerance(
                diagonal_abs_max_matrix(mat),
                cholesky_zero_pad_tolerance,
            );
            for j in 0..n {
                // Compute L[j,j]
                let mut s = mat[(j, j)];
                for k in 0..j {
                    s -= mat[(j, k)] * mat[(j, k)];
                }
                if s <= 0.0 {
                    if s < -tol {
                        return Err(MixedModelError::PosDefException);
                    }
                    // Zero row (singular RE)
                    for i in j..n {
                        mat[(i, j)] = 0.0;
                    }
                    continue;
                }
                mat[(j, j)] = s.sqrt();

                // Compute L[i,j] for i > j
                for i in (j + 1)..n {
                    let mut s = mat[(i, j)];
                    for k in 0..j {
                        s -= mat[(i, k)] * mat[(j, k)];
                    }
                    mat[(i, j)] = s / mat[(j, j)];
                }

                // Zero out upper triangle
                for i in 0..j {
                    mat[(i, j)] = 0.0;
                }
            }
            Ok(())
        }
        MatrixBlock::Sparse(_) => {
            let dense = block.as_dense();
            *block = MatrixBlock::Dense(dense);
            cholesky_block_with_tolerance(block, cholesky_zero_pad_tolerance)
        }
    }
}

/// Right-divide by lower triangular transpose: A = A * L^{-T}
pub(super) fn rdiv_lower_transpose(a: &mut MatrixBlock, l: &MatrixBlock) {
    match l {
        MatrixBlock::Diagonal(l_diag) => match a {
            MatrixBlock::Dense(a_mat) => {
                for j in 0..l_diag.len() {
                    let denom = l_diag[j];
                    if denom.abs() < BLOCK_TRIANGULAR_SOLVE_ZERO_TOLERANCE {
                        for i in 0..a_mat.nrows() {
                            a_mat[(i, j)] = 0.0;
                        }
                        continue;
                    }
                    for i in 0..a_mat.nrows() {
                        a_mat[(i, j)] /= denom;
                    }
                }
            }
            MatrixBlock::Sparse(a_sparse) => {
                for j in 0..a_sparse.ncols() {
                    let denom = l_diag[j];
                    let mut col = a_sparse.col_mut(j);
                    if denom.abs() < BLOCK_TRIANGULAR_SOLVE_ZERO_TOLERANCE {
                        for value in col.values_mut() {
                            *value = 0.0;
                        }
                    } else {
                        for value in col.values_mut() {
                            *value /= denom;
                        }
                    }
                }
            }
            MatrixBlock::Diagonal(a_diag) => {
                for i in 0..a_diag.len() {
                    let denom = l_diag[i];
                    if denom.abs() > BLOCK_TRIANGULAR_SOLVE_ZERO_TOLERANCE {
                        a_diag[i] /= denom;
                    } else {
                        a_diag[i] = 0.0;
                    }
                }
            }
            MatrixBlock::BlockDiagonal(_) => {
                let mut a_dense = a.as_dense();
                for j in 0..l_diag.len() {
                    let denom = l_diag[j];
                    if denom.abs() < BLOCK_TRIANGULAR_SOLVE_ZERO_TOLERANCE {
                        for i in 0..a_dense.nrows() {
                            a_dense[(i, j)] = 0.0;
                        }
                        continue;
                    }
                    for i in 0..a_dense.nrows() {
                        a_dense[(i, j)] /= denom;
                    }
                }
                *a = MatrixBlock::Dense(a_dense);
            }
        },
        MatrixBlock::BlockDiagonal(l_blocks) => {
            // L is block-diagonal: solve each column-block independently
            // A[:,block_k] = A[:,block_k] * L_k^{-T}
            match a {
                MatrixBlock::Dense(a_mat) => {
                    let mut col_offset = 0;
                    for l_blk in l_blocks {
                        let s = l_blk.nrows();
                        if s == 2 {
                            let c0 = col_offset;
                            let c1 = col_offset + 1;
                            let l00 = l_blk[(0, 0)];
                            let l10 = l_blk[(1, 0)];
                            let l11 = l_blk[(1, 1)];

                            for i in 0..a_mat.nrows() {
                                let x0 = a_mat[(i, c0)];
                                if l00.abs() < BLOCK_TRIANGULAR_SOLVE_ZERO_TOLERANCE {
                                    a_mat[(i, c0)] = 0.0;
                                } else {
                                    a_mat[(i, c0)] = x0 / l00;
                                }

                                if l11.abs() < BLOCK_TRIANGULAR_SOLVE_ZERO_TOLERANCE {
                                    a_mat[(i, c1)] = 0.0;
                                } else {
                                    a_mat[(i, c1)] = (a_mat[(i, c1)] - a_mat[(i, c0)] * l10) / l11;
                                }
                            }
                            col_offset += s;
                            continue;
                        }

                        // Solve the s-column slice of A against L_k
                        for j in 0..s {
                            let cj = col_offset + j;
                            if l_blk[(j, j)].abs() < BLOCK_TRIANGULAR_SOLVE_ZERO_TOLERANCE {
                                for i in 0..a_mat.nrows() {
                                    a_mat[(i, cj)] = 0.0;
                                }
                                continue;
                            }
                            for i in 0..a_mat.nrows() {
                                let mut val = a_mat[(i, cj)];
                                for k in 0..j {
                                    val -= a_mat[(i, col_offset + k)] * l_blk[(j, k)];
                                }
                                a_mat[(i, cj)] = val / l_blk[(j, j)];
                            }
                        }
                        col_offset += s;
                    }
                }
                MatrixBlock::BlockDiagonal(_) | MatrixBlock::Sparse(_) => {
                    // Both block-diagonal: promote A to dense, then solve
                    let mut a_dense = a.as_dense();
                    let mut col_offset = 0;
                    for l_blk in l_blocks {
                        let s = l_blk.nrows();
                        for j in 0..s {
                            let cj = col_offset + j;
                            if l_blk[(j, j)].abs() < BLOCK_TRIANGULAR_SOLVE_ZERO_TOLERANCE {
                                for i in 0..a_dense.nrows() {
                                    a_dense[(i, cj)] = 0.0;
                                }
                                continue;
                            }
                            for i in 0..a_dense.nrows() {
                                let mut val = a_dense[(i, cj)];
                                for k in 0..j {
                                    val -= a_dense[(i, col_offset + k)] * l_blk[(j, k)];
                                }
                                a_dense[(i, cj)] = val / l_blk[(j, j)];
                            }
                        }
                        col_offset += s;
                    }
                    *a = MatrixBlock::Dense(a_dense);
                }
                MatrixBlock::Diagonal(a_diag) => {
                    // Diagonal A, BlockDiagonal L: promote to dense
                    let mut a_dense = DMatrix::from_diagonal(a_diag);
                    let mut col_offset = 0;
                    for l_blk in l_blocks {
                        let s = l_blk.nrows();
                        for j in 0..s {
                            let cj = col_offset + j;
                            if l_blk[(j, j)].abs() < BLOCK_TRIANGULAR_SOLVE_ZERO_TOLERANCE {
                                for i in 0..a_dense.nrows() {
                                    a_dense[(i, cj)] = 0.0;
                                }
                                continue;
                            }
                            for i in 0..a_dense.nrows() {
                                let mut val = a_dense[(i, cj)];
                                for k in 0..j {
                                    val -= a_dense[(i, col_offset + k)] * l_blk[(j, k)];
                                }
                                a_dense[(i, cj)] = val / l_blk[(j, j)];
                            }
                        }
                        col_offset += s;
                    }
                    *a = MatrixBlock::Dense(a_dense);
                }
            }
        }
        _ => {
            // L is Dense or Diagonal — original logic. Borrow a dense L
            // directly; this solve only reads it, and cloning the whole
            // diagonal block here was one full copy per off-diagonal block
            // per objective evaluation.
            let l_owned;
            let l_dense: &DMatrix<f64> = match l.as_dense_ref() {
                Some(dense) => dense,
                None => {
                    l_owned = l.as_dense();
                    &l_owned
                }
            };
            let n = l_dense.nrows();

            match a {
                MatrixBlock::Dense(a_mat) => {
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
                MatrixBlock::Diagonal(a_diag) => match l {
                    MatrixBlock::Diagonal(l_diag) => {
                        for i in 0..a_diag.len() {
                            if l_diag[i].abs() > BLOCK_TRIANGULAR_SOLVE_ZERO_TOLERANCE {
                                a_diag[i] /= l_diag[i];
                            } else {
                                a_diag[i] = 0.0;
                            }
                        }
                    }
                    _ => {
                        let mut a_dense = DMatrix::from_diagonal(a_diag);
                        for j in 0..n {
                            if l_dense[(j, j)].abs() < BLOCK_TRIANGULAR_SOLVE_ZERO_TOLERANCE {
                                for i in 0..a_dense.nrows() {
                                    a_dense[(i, j)] = 0.0;
                                }
                                continue;
                            }
                            for i in 0..a_dense.nrows() {
                                let mut s = a_dense[(i, j)];
                                for k in 0..j {
                                    s -= a_dense[(i, k)] * l_dense[(j, k)];
                                }
                                a_dense[(i, j)] = s / l_dense[(j, j)];
                            }
                        }
                        *a = MatrixBlock::Dense(a_dense);
                    }
                },
                MatrixBlock::BlockDiagonal(_) | MatrixBlock::Sparse(_) => {
                    // Promote to dense and solve
                    let mut a_dense = a.as_dense();
                    for j in 0..n {
                        if l_dense[(j, j)].abs() < BLOCK_TRIANGULAR_SOLVE_ZERO_TOLERANCE {
                            for i in 0..a_dense.nrows() {
                                a_dense[(i, j)] = 0.0;
                            }
                            continue;
                        }
                        for i in 0..a_dense.nrows() {
                            let mut s = a_dense[(i, j)];
                            for k in 0..j {
                                s -= a_dense[(i, k)] * l_dense[(j, k)];
                            }
                            a_dense[(i, j)] = s / l_dense[(j, j)];
                        }
                    }
                    *a = MatrixBlock::Dense(a_dense);
                }
            }
        }
    }
}

/// Log-determinant of a Cholesky block (sum of log of diagonal elements).
pub(super) fn logdet_block(block: &MatrixBlock) -> f64 {
    match block {
        MatrixBlock::Diagonal(diag) => {
            diag.iter()
                .filter(|&&d| d > 0.0)
                .map(|d| d.ln())
                .sum::<f64>()
                * 2.0
        }
        MatrixBlock::BlockDiagonal(blocks) => {
            if blocks.first().is_some_and(|blk| blk.nrows() == 2) {
                let mut ld = 0.0;
                for blk in blocks {
                    let d0 = blk[(0, 0)];
                    let d1 = blk[(1, 1)];
                    if d0 > 0.0 {
                        ld += d0.ln();
                    }
                    if d1 > 0.0 {
                        ld += d1.ln();
                    }
                }
                return ld * 2.0;
            }

            let mut ld = 0.0;
            for blk in blocks {
                let n = blk.nrows();
                for i in 0..n {
                    let d = blk[(i, i)];
                    if d > 0.0 {
                        ld += d.ln();
                    }
                }
            }
            ld * 2.0
        }
        MatrixBlock::Dense(mat) => {
            let n = mat.nrows().min(mat.ncols());
            let mut ld = 0.0;
            for i in 0..n {
                let d = mat[(i, i)];
                if d > 0.0 {
                    ld += d.ln();
                }
            }
            ld * 2.0
        }
        MatrixBlock::Sparse(mat) => {
            let dense = MatrixBlock::Sparse(mat.clone()).as_dense();
            logdet_block(&MatrixBlock::Dense(dense))
        }
    }
}

/// The A-block builders replace nalgebra products with explicit sequential
/// slice loops where (and only where) nalgebra itself sums sequentially, so
/// every `A` entry, and therefore every objective, is bit-identical to the
/// previous construction. These tests pin that assumption against the
/// pinned nalgebra version: if an upgrade changes the small-dimension gemm
/// threshold or the gemv accumulation order, they fail here rather than in
/// a parity fixture.
#[cfg(test)]
mod nalgebra_summation_order {
    use nalgebra::{DMatrix, DVector};

    use crate::model::fixed_design::NALGEBRA_SEQUENTIAL_GEMM_MAX_DIM;

    fn design(n: usize, p: usize) -> (DMatrix<f64>, DVector<f64>) {
        let x = DMatrix::from_fn(n, p, |i, j| ((i * 7 + j * 3) as f64 * 0.37).sin() + 0.1);
        let y = DVector::from_fn(n, |i, _| ((i * 11) as f64 * 0.19).cos());
        (x, y)
    }

    fn sequential_xtx(x: &DMatrix<f64>) -> DMatrix<f64> {
        let p = x.ncols();
        DMatrix::from_fn(p, p, |i, j| {
            (0..x.nrows()).fold(0.0, |acc, k| acc + x[(k, i)] * x[(k, j)])
        })
    }

    fn sequential_xty(x: &DMatrix<f64>, y: &DVector<f64>) -> DVector<f64> {
        DVector::from_fn(x.ncols(), |i, _| {
            (0..x.nrows()).fold(0.0, |acc, k| acc + x[(k, i)] * y[k])
        })
    }

    #[test]
    fn skinny_xtx_is_a_sequential_dot_per_cell_up_to_the_pinned_width() {
        for p in 1..=NALGEBRA_SEQUENTIAL_GEMM_MAX_DIM {
            for n in [7usize, 997, 10_000] {
                let (x, _) = design(n, p);
                assert_eq!(
                    x.transpose() * &x,
                    sequential_xtx(&x),
                    "nalgebra x'x is no longer sequential at p={p}, n={n}; \
                     DenseFixedDesign::xtx would change every objective"
                );
            }
        }
    }

    #[test]
    fn wide_xtx_takes_the_blocked_gemm_path_above_the_pinned_width() {
        // Documents why the explicit loop is not used for wide designs: the
        // blocked product sums in a different order.
        let (x, _) = design(4096, NALGEBRA_SEQUENTIAL_GEMM_MAX_DIM + 3);
        assert_ne!(x.transpose() * &x, sequential_xtx(&x));
    }

    #[test]
    fn xty_gemv_accumulates_in_observation_order_for_any_width() {
        for p in [1usize, 2, 5, 8, 16] {
            for n in [7usize, 997, 10_000] {
                let (x, y) = design(n, p);
                assert_eq!(
                    x.transpose() * &y,
                    sequential_xty(&x, &y),
                    "nalgebra gemv order changed at p={p}, n={n}"
                );
            }
        }
    }
}

#[cfg(test)]
mod sparse_product_paths {
    use nalgebra::DMatrix;
    use nalgebra_sparse::{coo::CooMatrix, csc::CscMatrix};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use super::{
        block_index, subtract_block_matvec, subtract_block_transpose_matvec, subtract_product,
        subtract_product_from_blocks,
    };
    use crate::types::MatrixBlock;

    const TOL: f64 = 1e-12;

    fn random_sparse(rng: &mut StdRng, nrows: usize, ncols: usize, density: f64) -> CscMatrix<f64> {
        let mut coo = CooMatrix::new(nrows, ncols);
        for col in 0..ncols {
            for row in 0..nrows {
                if rng.gen::<f64>() < density {
                    coo.push(row, col, rng.gen_range(-2.0..2.0));
                }
            }
        }
        CscMatrix::from(&coo)
    }

    /// One stored entry per column: the nested-grouping indicator shape.
    fn random_indicator(rng: &mut StdRng, nrows: usize, ncols: usize) -> CscMatrix<f64> {
        let mut coo = CooMatrix::new(nrows, ncols);
        for col in 0..ncols {
            coo.push(rng.gen_range(0..nrows), col, rng.gen_range(0.1..2.0));
        }
        CscMatrix::from(&coo)
    }

    fn random_dense(rng: &mut StdRng, nrows: usize, ncols: usize) -> DMatrix<f64> {
        DMatrix::from_fn(nrows, ncols, |_, _| rng.gen_range(-2.0..2.0))
    }

    fn dense_of(block: &MatrixBlock) -> DMatrix<f64> {
        block.as_dense()
    }

    /// Sparse target whose pattern is `pattern_of` plus random extra entries
    /// (and optionally minus one product entry), with random values.
    fn sparse_target(
        rng: &mut StdRng,
        product: &DMatrix<f64>,
        extra_density: f64,
        drop_one: bool,
    ) -> (CscMatrix<f64>, bool) {
        let (m, n) = product.shape();
        let mut coo = CooMatrix::new(m, n);
        let mut dropped = false;
        for col in 0..n {
            for row in 0..m {
                let in_product = product[(row, col)] != 0.0;
                if in_product && drop_one && !dropped {
                    dropped = true;
                    continue;
                }
                if in_product || rng.gen::<f64>() < extra_density {
                    coo.push(row, col, rng.gen_range(-2.0..2.0));
                }
            }
        }
        (CscMatrix::from(&coo), dropped)
    }

    fn assert_close(actual: &DMatrix<f64>, expected: &DMatrix<f64>, what: &str) {
        assert_eq!(actual.shape(), expected.shape(), "{what}: shape");
        let diff = (actual - expected).abs().max();
        assert!(diff <= TOL, "{what}: max abs diff {diff:e}");
    }

    fn expected_after(c0: &DMatrix<f64>, a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
        c0 - a * b.transpose()
    }

    #[test]
    fn sparse_operands_into_dense_target_match_dense_gemm() {
        let mut rng = StdRng::seed_from_u64(0x5eed_0001);
        for trial in 0..40 {
            let m = rng.gen_range(1..12);
            let n = rng.gen_range(1..12);
            let k = rng.gen_range(1..30);
            let density = [0.05, 0.2, 0.5, 1.0][trial % 4];
            let a_sp = random_sparse(&mut rng, m, k, density);
            let b_sp = random_sparse(&mut rng, n, k, density);
            let a_d = random_dense(&mut rng, m, k);
            let b_d = random_dense(&mut rng, n, k);
            let c0 = random_dense(&mut rng, m, n);
            let a_sp_d = dense_of(&MatrixBlock::Sparse(a_sp.clone()));
            let b_sp_d = dense_of(&MatrixBlock::Sparse(b_sp.clone()));

            let cases: [(MatrixBlock, MatrixBlock, &DMatrix<f64>, &DMatrix<f64>, &str); 3] = [
                (
                    MatrixBlock::Sparse(a_sp.clone()),
                    MatrixBlock::Sparse(b_sp.clone()),
                    &a_sp_d,
                    &b_sp_d,
                    "sparse*sparse'",
                ),
                (
                    MatrixBlock::Dense(a_d.clone()),
                    MatrixBlock::Sparse(b_sp.clone()),
                    &a_d,
                    &b_sp_d,
                    "dense*sparse'",
                ),
                (
                    MatrixBlock::Sparse(a_sp.clone()),
                    MatrixBlock::Dense(b_d.clone()),
                    &a_sp_d,
                    &b_d,
                    "sparse*dense'",
                ),
            ];
            for (a, b, a_ref, b_ref, what) in cases {
                let expected = expected_after(&c0, a_ref, b_ref);
                let mut c = MatrixBlock::Dense(c0.clone());
                subtract_product_from_blocks(&mut c, &a, &b);
                let MatrixBlock::Dense(got) = &c else {
                    panic!("{what}: dense target changed variant");
                };
                assert_close(got, &expected, what);

                // A diagonal target is promoted to dense, as before.
                if m == n {
                    let diag0 = c0.diagonal();
                    let mut c = MatrixBlock::Diagonal(diag0.clone());
                    subtract_product_from_blocks(&mut c, &a, &b);
                    let expected = expected_after(&DMatrix::from_diagonal(&diag0), a_ref, b_ref);
                    assert!(matches!(c, MatrixBlock::Dense(_)), "{what}: diag promotion");
                    assert_close(&c.as_dense(), &expected, what);
                }
            }
        }
    }

    #[test]
    fn sparse_target_covering_the_product_stays_sparse_and_matches_dense() {
        let mut rng = StdRng::seed_from_u64(0x5eed_0002);
        for trial in 0..60 {
            let m = rng.gen_range(1..10);
            let n = rng.gen_range(1..14);
            let k = rng.gen_range(1..40);
            let (a_sp, b_sp) = if trial % 2 == 0 {
                (
                    random_indicator(&mut rng, m, k),
                    random_indicator(&mut rng, n, k),
                )
            } else {
                (
                    random_sparse(&mut rng, m, k, 0.15),
                    random_sparse(&mut rng, n, k, 0.15),
                )
            };
            let a_ref = dense_of(&MatrixBlock::Sparse(a_sp.clone()));
            let b_ref = dense_of(&MatrixBlock::Sparse(b_sp.clone()));
            // Structural product pattern (entries with a shared inner index).
            let pattern = a_ref.map(|x| (x != 0.0) as u8 as f64)
                * b_ref.map(|x| (x != 0.0) as u8 as f64).transpose();
            let (c_sp, _) = sparse_target(&mut rng, &pattern, 0.2, false);
            let c0 = dense_of(&MatrixBlock::Sparse(c_sp.clone()));
            let expected = expected_after(&c0, &a_ref, &b_ref);

            let mut c = MatrixBlock::Sparse(c_sp.clone());
            subtract_product_from_blocks(
                &mut c,
                &MatrixBlock::Sparse(a_sp.clone()),
                &MatrixBlock::Sparse(b_sp.clone()),
            );
            let MatrixBlock::Sparse(got) = &c else {
                panic!("covered sparse target was densified (trial {trial})");
            };
            assert_eq!(got.col_offsets(), c_sp.col_offsets(), "pattern changed");
            assert_eq!(got.row_indices(), c_sp.row_indices(), "pattern changed");
            assert_close(&c.as_dense(), &expected, "covered sparse target");
        }
    }

    #[test]
    fn sparse_target_missing_a_product_entry_falls_back_to_dense() {
        let mut rng = StdRng::seed_from_u64(0x5eed_0003);
        let mut exercised = 0;
        for trial in 0..60 {
            let m = rng.gen_range(1..10);
            let n = rng.gen_range(1..14);
            let k = rng.gen_range(1..40);
            let a_sp = random_sparse(&mut rng, m, k, 0.2);
            let b_sp = random_sparse(&mut rng, n, k, 0.2);
            let a_ref = dense_of(&MatrixBlock::Sparse(a_sp.clone()));
            let b_ref = dense_of(&MatrixBlock::Sparse(b_sp.clone()));
            let pattern = a_ref.map(|x| (x != 0.0) as u8 as f64)
                * b_ref.map(|x| (x != 0.0) as u8 as f64).transpose();
            let (c_sp, dropped) = sparse_target(&mut rng, &pattern, 0.1, true);
            if !dropped {
                continue;
            }
            exercised += 1;
            let c0 = dense_of(&MatrixBlock::Sparse(c_sp.clone()));
            let expected = expected_after(&c0, &a_ref, &b_ref);
            let mut c = MatrixBlock::Sparse(c_sp);
            subtract_product_from_blocks(
                &mut c,
                &MatrixBlock::Sparse(a_sp),
                &MatrixBlock::Sparse(b_sp),
            );
            assert!(
                matches!(c, MatrixBlock::Dense(_)),
                "uncovered sparse target must be promoted (trial {trial})"
            );
            assert_close(&c.as_dense(), &expected, "uncovered sparse target");
        }
        assert!(exercised >= 30, "too few uncovered trials: {exercised}");
    }

    #[test]
    fn dense_dense_path_is_unchanged_bitwise() {
        let mut rng = StdRng::seed_from_u64(0x5eed_0004);
        for _ in 0..20 {
            let (m, n, k) = (
                rng.gen_range(1..20),
                rng.gen_range(1..20),
                rng.gen_range(1..40),
            );
            let a = random_dense(&mut rng, m, k);
            let b = random_dense(&mut rng, n, k);
            let c0 = random_dense(&mut rng, m, n);
            let mut via_blocks = MatrixBlock::Dense(c0.clone());
            subtract_product_from_blocks(
                &mut via_blocks,
                &MatrixBlock::Dense(a.clone()),
                &MatrixBlock::Dense(b.clone()),
            );
            let mut direct = MatrixBlock::Dense(c0);
            subtract_product(&mut direct, &a, &b);
            assert_eq!(via_blocks.as_dense(), direct.as_dense());
        }
    }

    #[test]
    fn block_matvecs_are_bit_identical_to_the_dense_loops() {
        let mut rng = StdRng::seed_from_u64(0x5eed_0005);
        let dense_matvec = |l: &DMatrix<f64>, v: &[f64], dst: &mut [f64]| {
            for row in 0..l.nrows() {
                let mut dot = 0.0;
                for col in 0..v.len() {
                    dot += l[(row, col)] * v[col];
                }
                dst[row] -= dot;
            }
        };
        let dense_tmatvec = |l: &DMatrix<f64>, u: &[f64], dst: &mut [f64]| {
            for row in 0..l.ncols() {
                let mut dot = 0.0;
                for col in 0..u.len() {
                    dot += l[(col, row)] * u[col];
                }
                dst[row] -= dot;
            }
        };
        for trial in 0..40 {
            let (m, n) = (rng.gen_range(1..25), rng.gen_range(1..25));
            let blocks = [
                MatrixBlock::Sparse(random_sparse(&mut rng, m, n, 0.3)),
                MatrixBlock::Dense(random_dense(&mut rng, m, n)),
                MatrixBlock::Diagonal(random_dense(&mut rng, m, 1).column(0).into_owned()),
                MatrixBlock::BlockDiagonal(vec![
                    random_dense(&mut rng, 2, 2),
                    random_dense(&mut rng, 1, 1),
                    random_dense(&mut rng, 3, 3),
                ]),
            ];
            for block in &blocks {
                let dense = block.as_dense();
                let (r, c) = dense.shape();
                let v: Vec<f64> = (0..c).map(|_| rng.gen_range(-3.0..3.0)).collect();
                let u: Vec<f64> = (0..r).map(|_| rng.gen_range(-3.0..3.0)).collect();
                let base_r: Vec<f64> = (0..r).map(|_| rng.gen_range(-3.0..3.0)).collect();
                let base_c: Vec<f64> = (0..c).map(|_| rng.gen_range(-3.0..3.0)).collect();

                let (mut got, mut want) = (base_r.clone(), base_r.clone());
                subtract_block_matvec(&mut got, block, &v);
                dense_matvec(&dense, &v, &mut want);
                assert_eq!(got, want, "matvec trial {trial}");

                let (mut got, mut want) = (base_c.clone(), base_c.clone());
                subtract_block_transpose_matvec(&mut got, block, &u);
                dense_tmatvec(&dense, &u, &mut want);
                assert_eq!(got, want, "transpose matvec trial {trial}");
            }
        }
    }

    /// Assert the blocked factor held in `model.l_blocks` equals a dense
    /// Cholesky of the assembled `Λ'AΛ + I` system (general per-term Λ,
    /// including vector-valued terms).
    fn assert_blocked_l_matches_dense_cholesky(
        model: &crate::model::linear::LinearMixedModel,
        label: &str,
    ) {
        let k = model.reterms.len();
        let sizes: Vec<usize> = (0..=k)
            .map(|b| model.a_blocks[block_index(b, b)].nrows())
            .collect();
        let mut offsets = vec![0; k + 1];
        for b in 1..=k {
            offsets[b] = offsets[b - 1] + sizes[b - 1];
        }
        let total: usize = sizes.iter().sum();
        let mut a_full = DMatrix::<f64>::zeros(total, total);
        let mut lambda = DMatrix::<f64>::identity(total, total);
        let mut l_full = DMatrix::<f64>::zeros(total, total);
        for (j, rt) in model.reterms.iter().enumerate() {
            let s = rt.vsize;
            for level in 0..sizes[j] / s {
                let base = offsets[j] + level * s;
                for r in 0..s {
                    for c in 0..s {
                        lambda[(base + r, base + c)] = rt.lambda[(r, c)];
                    }
                }
            }
        }
        for i in 0..=k {
            for j in 0..=i {
                let a = model.a_blocks[block_index(i, j)].as_dense();
                let l = model.l_blocks[block_index(i, j)].as_dense();
                for r in 0..sizes[i] {
                    for c in 0..sizes[j] {
                        let (gr, gc) = (offsets[i] + r, offsets[j] + c);
                        a_full[(gr, gc)] = a[(r, c)];
                        a_full[(gc, gr)] = a[(r, c)];
                        if i != j || c <= r {
                            l_full[(gr, gc)] = l[(r, c)];
                        }
                    }
                }
            }
        }
        let mut system = lambda.transpose() * a_full * &lambda;
        for d in 0..offsets[k] {
            system[(d, d)] += 1.0;
        }
        let reference = nalgebra::Cholesky::new(system)
            .unwrap_or_else(|| panic!("{label}: reference system not positive definite"))
            .l();
        let diff = (&l_full - &reference).abs().max();
        let magnitude = reference.abs().max().max(1.0);
        assert!(
            diff <= 1e-10 * magnitude,
            "{label}: blocked L differs from dense Cholesky by {diff:e}"
        );
    }

    /// End to end: three nested scalar terms keep every RE off-diagonal L
    /// block sparse, and the blocked factor equals a dense Cholesky of the
    /// assembled `Λ'AΛ + I` system.
    #[test]
    fn nested_scalar_terms_keep_sparse_l_and_match_dense_cholesky() {
        use crate::formula::parse_formula;
        use crate::model::linear::LinearMixedModel;

        let (mut data, _) = crate::datasets::load("grouseticks").unwrap();
        let ticks: Vec<f64> = data
            .numeric("TICKS")
            .unwrap()
            .iter()
            .map(|t| (t + 1.0).ln())
            .collect();
        data.add_numeric("LT", ticks).unwrap();
        let formula =
            parse_formula("LT ~ 1 + YEAR + cHEIGHT + (1 | BROOD) + (1 | INDEX) + (1 | LOCATION)")
                .unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        let k = model.reterms.len();
        assert_eq!(k, 3);

        for theta in [[1.0, 1.0, 1.0], [0.3, 1.7, 0.6], [2.5, 0.05, 1.1]] {
            model.set_theta(&theta).unwrap();
            model.update_l().unwrap();
            for i in 1..k {
                for j in 0..i {
                    assert!(
                        matches!(model.l_blocks[block_index(i, j)], MatrixBlock::Sparse(_)),
                        "L[{i},{j}] was densified"
                    );
                }
            }
            assert_blocked_l_matches_dense_cholesky(&model, &format!("grouseticks {theta:?}"));
        }
    }

    /// Partially nested, crossed, zero-valued-covariate and vector-valued
    /// layouts, each refactored over a θ path (including θ = 0 and a return
    /// to 1) that reuses the same L buffers across evaluations, so a block
    /// promoted to dense or kept sparse by one evaluation must stay correct
    /// on the next.
    #[test]
    fn mixed_block_layouts_match_dense_cholesky_across_theta_path() {
        use crate::formula::parse_formula;
        use crate::model::data::DataFrame;
        use crate::model::linear::LinearMixedModel;

        let mut rng = StdRng::seed_from_u64(7);
        let (mut g0, mut g1, mut g2, mut g3, mut x, mut y) =
            (vec![], vec![], vec![], vec![], vec![], vec![]);
        // g0 and g1 are crossed with each other, both nested in g2; g3 is
        // crossed with everything. 30% of x is exactly zero.
        for r in 0..8 {
            for a in 0..6 {
                for b in 0..3 {
                    for _ in 0..rng.gen_range(1..4) {
                        g2.push(format!("r{r}"));
                        g0.push(format!("s{r}_{a}"));
                        g1.push(format!("b{r}_{b}"));
                        g3.push(format!("c{}", rng.gen_range(0..7)));
                        x.push(if rng.gen::<f64>() < 0.3 {
                            0.0
                        } else {
                            rng.gen_range(-1.0..1.0)
                        });
                        y.push(rng.gen_range(-1.0..1.0));
                    }
                }
            }
        }
        let mut df = DataFrame::new();
        df.add_categorical("g0", g0).unwrap();
        df.add_categorical("g1", g1).unwrap();
        df.add_categorical("g2", g2).unwrap();
        df.add_categorical("g3", g3).unwrap();
        df.add_numeric("x", x).unwrap();
        df.add_numeric("y", y).unwrap();

        let formulas = [
            "y ~ 1 + x + (1 | g0) + (1 | g1) + (1 | g2)",
            "y ~ 1 + x + (1 | g0) + (1 | g1) + (1 | g2) + (1 | g3)",
            "y ~ 1 + x + (1 | g0) + (0 + x | g2) + (1 | g2)",
            "y ~ 1 + x + (1 + x | g0) + (1 | g2)",
            "y ~ 1 + x + (1 | g0) + (1 + x | g2)",
            "y ~ 1 + x + (1 | g1) + (1 + x | g0) + (1 | g2)",
        ];
        let mut rng = StdRng::seed_from_u64(11);
        for formula in formulas {
            let mut model =
                LinearMixedModel::new(parse_formula(formula).unwrap(), &df, None).unwrap();
            let nt = model.theta().len();
            let mut thetas = vec![vec![1.0; nt]];
            for _ in 0..3 {
                thetas.push((0..nt).map(|_| rng.gen_range(0.05..2.0)).collect());
            }
            thetas.push(vec![0.0; nt]);
            thetas.push(vec![1.0; nt]);
            for theta in &thetas {
                model.set_theta(theta).unwrap();
                model.update_l().unwrap();
                assert_blocked_l_matches_dense_cholesky(&model, &format!("{formula} {theta:?}"));
            }
        }
    }
}

#[cfg(test)]
mod wtxy_re_cross_product {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use super::{compute_fe_re_cross_product, compute_wtxy_re_cross_product};
    use crate::formula::parse_formula;
    use crate::model::data::DataFrame;
    use crate::model::linear::LinearMixedModel;
    use crate::types::MatrixBlock;

    /// The observation-outer kernel must equal the reference `[X|y]'Z_j`
    /// bit for bit, for group-sorted and interleaved rows and for scalar and
    /// vector-valued terms.
    #[test]
    fn matches_reference_bitwise() {
        for sorted in [true, false] {
            let mut rng = StdRng::seed_from_u64(if sorted { 3 } else { 4 });
            let (mut g, mut h, mut x, mut y) = (vec![], vec![], vec![], vec![]);
            for i in 0..400 {
                let level = if sorted { i / 20 } else { rng.gen_range(0..20) };
                g.push(format!("g{level}"));
                h.push(format!("h{}", rng.gen_range(0..7)));
                x.push(rng.gen_range(-2.0..2.0));
                y.push(rng.gen_range(-1.0..1.0));
            }
            let mut df = DataFrame::new();
            df.add_categorical("g", g).unwrap();
            df.add_categorical("h", h).unwrap();
            df.add_numeric("x", x).unwrap();
            df.add_numeric("y", y).unwrap();
            let model = LinearMixedModel::new(
                parse_formula("y ~ 1 + x + (1 + x | g) + (1 | h)").unwrap(),
                &df,
                None,
            )
            .unwrap();
            for re in &model.reterms {
                let fast = compute_wtxy_re_cross_product(&model.xy_mat.wtxy, re);
                let MatrixBlock::Dense(reference) = compute_fe_re_cross_product(&model.xy_mat, re)
                else {
                    panic!("reference block is dense");
                };
                assert_eq!(fast.shape(), reference.shape());
                for (a, b) in fast.iter().zip(reference.iter()) {
                    assert_eq!(a.to_bits(), b.to_bits(), "sorted={sorted}");
                }
            }
        }
    }
}
