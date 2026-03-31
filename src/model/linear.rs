//! Linear mixed-effects model (LMM).
//!
//! Implements the penalized least squares (PLS) algorithm for fitting
//! linear mixed models via profile likelihood optimization.
//!
//! The model is: y = Xβ + Zb + ε, where b ~ N(0, σ²Λθ Λθ') and ε ~ N(0, σ²I).
//!
//! The θ parameters control the relative covariance factor Λ. The objective
//! function (deviance or REML criterion) is profiled over β and σ², leaving
//! only θ to be optimized numerically.

use nalgebra::{DMatrix, DVector};

use crate::error::{MixedModelError, Result};
use crate::formula::Formula;
use crate::model::data::{Column, DataFrame};
use crate::model::traits::MixedModelFit;
use crate::types::{FeMat, FeTerm, OptSummary, Optimizer, ReMat};

/// A fitted (or constructed but unfitted) linear mixed-effects model.
///
/// Corresponds to `LinearMixedModel{T}` in MixedModels.jl.
///
/// # Fields
/// - `formula`: the parsed model formula
/// - `reterms`: random-effects model matrices, sorted by decreasing nranef
/// - `xy_mat`: the fixed-effects model matrix concatenated with y, with optional weighting
/// - `feterm`: the fixed-effects model matrix with rank/pivot info
/// - `sqrtwts`: square roots of case weights (empty if unweighted)
/// - `parmap`: mapping from θ indices to (block, row, col) in λ
/// - `dims`: model dimensions (n, p, nretrms)
/// - `a_blocks`: lower triangle of [Z X y]'[Z X y] in blocked storage
/// - `l_blocks`: blocked lower Cholesky factor of Λ'AΛ + I
/// - `optsum`: optimization summary
#[derive(Debug, Clone)]
pub struct LinearMixedModel {
    pub formula: Formula,
    pub reterms: Vec<ReMat>,
    pub xy_mat: FeMat,
    pub feterm: FeTerm,
    pub sqrtwts: Vec<f64>,
    pub parmap: Vec<(usize, usize, usize)>, // (block, row, col)
    pub dims: ModelDims,
    pub a_blocks: Vec<MatrixBlock>,
    pub l_blocks: Vec<MatrixBlock>,
    pub optsum: OptSummary,
}

/// Model dimensions.
#[derive(Debug, Clone, Copy)]
pub struct ModelDims {
    pub n: usize,      // number of observations
    pub p: usize,      // rank of fixed-effects matrix
    pub nretrms: usize, // number of random-effects terms
}

/// A block in the lower-triangular blocked matrix system.
///
/// The blocked system stores the lower triangle of [Z₁ Z₂ ... X y]'[Z₁ Z₂ ... X y].
/// Blocks can be dense, diagonal, block-diagonal, or (in the L factor) lower triangular.
#[derive(Debug, Clone)]
pub enum MatrixBlock {
    Dense(DMatrix<f64>),
    Diagonal(DVector<f64>),
    /// Uniform block diagonal: `nlevels` blocks each of size `vsize × vsize`.
    /// Total matrix is `(nlevels * vsize) × (nlevels * vsize)`.
    BlockDiagonal(Vec<DMatrix<f64>>),
}

impl MatrixBlock {
    pub fn nrows(&self) -> usize {
        match self {
            MatrixBlock::Dense(m) => m.nrows(),
            MatrixBlock::Diagonal(v) => v.len(),
            MatrixBlock::BlockDiagonal(blocks) => {
                blocks.iter().map(|b| b.nrows()).sum()
            }
        }
    }

    pub fn ncols(&self) -> usize {
        match self {
            MatrixBlock::Dense(m) => m.ncols(),
            MatrixBlock::Diagonal(v) => v.len(),
            MatrixBlock::BlockDiagonal(blocks) => {
                blocks.iter().map(|b| b.ncols()).sum()
            }
        }
    }

    pub fn as_dense(&self) -> DMatrix<f64> {
        match self {
            MatrixBlock::Dense(m) => m.clone(),
            MatrixBlock::Diagonal(v) => DMatrix::from_diagonal(v),
            MatrixBlock::BlockDiagonal(blocks) => {
                let total = blocks.iter().map(|b| b.nrows()).sum();
                let mut result = DMatrix::zeros(total, total);
                let mut offset = 0;
                for blk in blocks {
                    let s = blk.nrows();
                    for i in 0..s {
                        for j in 0..s {
                            result[(offset + i, offset + j)] = blk[(i, j)];
                        }
                    }
                    offset += s;
                }
                result
            }
        }
    }

    pub fn as_dense_ref(&self) -> Option<&DMatrix<f64>> {
        match self {
            MatrixBlock::Dense(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_dense_mut(&mut self) -> Option<&mut DMatrix<f64>> {
        match self {
            MatrixBlock::Dense(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_diag_mut(&mut self) -> Option<&mut DVector<f64>> {
        match self {
            MatrixBlock::Diagonal(v) => Some(v),
            _ => None,
        }
    }
}

/// Convert row-major lower triangle index to linear index.
/// For a system with k random effects terms + 1 (for [X|y]),
/// block (i, j) where i >= j maps to index i*(i+1)/2 + j.
#[inline]
fn block_index(i: usize, j: usize) -> usize {
    debug_assert!(i >= j);
    i * (i + 1) / 2 + j
}

impl LinearMixedModel {
    /// Construct a LinearMixedModel from a formula, data, and optional weights.
    pub fn new(
        formula: Formula,
        data: &DataFrame,
        weights: Option<&[f64]>,
    ) -> Result<Self> {
        if formula.random_terms.is_empty() {
            return Err(MixedModelError::NoRandomEffects);
        }

        let n = data.nrow();

        // Build the response vector
        let y_data = data
            .numeric(&formula.response)
            .ok_or_else(|| {
                MixedModelError::InvalidArgument(format!(
                    "Response '{}' not found or not numeric",
                    formula.response
                ))
            })?;
        let y = DVector::from_column_slice(y_data);

        // Build the fixed-effects model matrix
        let (x_mat, fe_names) = build_fixed_effects_matrix(&formula, data)?;
        let feterm = FeTerm::new(x_mat, fe_names);

        // Build the random-effects terms
        let mut reterms = Vec::new();
        for rt in &formula.random_terms {
            let remat = build_re_mat(rt, data, n)?;
            reterms.push(remat);
        }

        // Sort by decreasing nranef (matches Julia behavior)
        reterms.sort_by(|a, b| b.n_ranef().cmp(&a.n_ranef()));

        // Build FeMat = [full_rank_X | y]
        let xy_mat = FeMat::new(&feterm, &y);

        // Apply weights
        let sqrtwts = if let Some(wts) = weights {
            let sw: Vec<f64> = wts.iter().map(|w| w.sqrt()).collect();
            // TODO: reweight reterms and xy_mat
            sw
        } else {
            vec![]
        };

        // Create cross-product blocks A and Cholesky blocks L
        let (a_blocks, l_blocks) = create_al(&reterms, &xy_mat);

        // Build theta vector from all reterms
        let theta: Vec<f64> = reterms.iter().flat_map(|rt| rt.get_theta()).collect();

        // Build parmap: mapping from θ index to (re_term_index, row, col) in lambda
        let parmap = build_parmap(&reterms);

        let dims = ModelDims {
            n,
            p: feterm.rank,
            nretrms: reterms.len(),
        };

        let optsum = OptSummary::new(theta);

        Ok(LinearMixedModel {
            formula: formula.clone(),
            reterms,
            xy_mat,
            feterm,
            sqrtwts,
            parmap,
            dims,
            a_blocks,
            l_blocks,
            optsum,
        })
    }

    /// Get the response vector y (last column of xy_mat).
    pub fn y(&self) -> DVector<f64> {
        let xy = &self.xy_mat.xy;
        xy.column(xy.ncols() - 1).into()
    }

    /// Get the current θ parameter vector.
    pub fn theta(&self) -> Vec<f64> {
        self.reterms.iter().flat_map(|rt| rt.get_theta()).collect()
    }

    /// Set the θ parameter vector, distributing values to each ReMat's λ.
    pub fn set_theta(&mut self, theta: &[f64]) -> Result<()> {
        let mut offset = 0;
        for rt in &mut self.reterms {
            let n = rt.n_theta();
            if offset + n > theta.len() {
                return Err(MixedModelError::DimensionMismatch(
                    "theta vector too short".into(),
                ));
            }
            rt.set_theta(&theta[offset..offset + n]);
            offset += n;
        }
        Ok(())
    }

    /// Lower bounds on θ. Diagonal elements of λ are ≥ 0, off-diagonal are unconstrained.
    pub fn lower_bounds(&self) -> Vec<f64> {
        let mut lb = Vec::new();
        for (_, row, col) in &self.parmap {
            if row == col {
                lb.push(0.0); // diagonal elements are non-negative
            } else {
                lb.push(f64::NEG_INFINITY);
            }
        }
        lb
    }

    /// Update the blocked Cholesky factor L from A and current λ values.
    ///
    /// This is the core operation: L = cholesky(Λ'AΛ + I).
    /// The blocks of L are updated in-place.
    pub fn update_l(&mut self) -> Result<()> {
        let k = self.reterms.len(); // number of RE terms
        let total = k + 1; // +1 for the [X|y] block

        // Copy A to L, scaling by Λ
        // For diagonal blocks L[j,j] = Λ_j' A[j,j] Λ_j + I
        for j in 0..k {
            let idx_jj = block_index(j, j);
            copy_scale_inflate(
                &mut self.l_blocks[idx_jj],
                &self.a_blocks[idx_jj],
                &self.reterms[j],
            );
        }

        // For off-diagonal RE blocks L[i,j] = Λ_i' A[i,j] Λ_j, i > j
        for i in 1..k {
            for j in 0..i {
                let idx_ij = block_index(i, j);
                copy_and_scale_offdiag(
                    &mut self.l_blocks[idx_ij],
                    &self.a_blocks[idx_ij],
                    &self.reterms[i],
                    &self.reterms[j],
                );
            }
        }

        // For FE-RE blocks L[k,j] = A[k,j] Λ_j (no Λ on left for FeMat)
        for j in 0..k {
            let idx_kj = block_index(k, j);
            copy_and_rmul_lambda(
                &mut self.l_blocks[idx_kj],
                &self.a_blocks[idx_kj],
                &self.reterms[j],
            );
        }

        // Copy the [X|y]'[X|y] block unchanged
        let idx_kk = block_index(k, k);
        self.l_blocks[idx_kk] = self.a_blocks[idx_kk].clone();

        // Blocked Cholesky factorization
        for j in 0..total {
            // Update L[j,j] by subtracting L[j,0..j] * L[j,0..j]'
            for jj in 0..j {
                let l_j_jj = self.l_blocks[block_index(j, jj)].as_dense();
                rank_k_downdate(&mut self.l_blocks[block_index(j, j)], &l_j_jj);
            }

            // Cholesky of diagonal block
            cholesky_block(&mut self.l_blocks[block_index(j, j)])?;

            // Solve for off-diagonal blocks: L[i,j] for i > j
            for i in (j + 1)..total {
                // L[i,j] -= sum_{jj<j} L[i,jj] * L[j,jj]'
                for jj in 0..j {
                    let l_i_jj = self.l_blocks[block_index(i, jj)].as_dense();
                    let l_j_jj = self.l_blocks[block_index(j, jj)].as_dense();
                    subtract_product(
                        &mut self.l_blocks[block_index(i, j)],
                        &l_i_jj,
                        &l_j_jj,
                    );
                }
                // L[i,j] = L[i,j] * L[j,j]^{-T}
                let l_jj = self.l_blocks[block_index(j, j)].clone();
                rdiv_lower_transpose(
                    &mut self.l_blocks[block_index(i, j)],
                    &l_jj,
                );
            }
        }

        Ok(())
    }

    /// Compute the profiled deviance or REML criterion for the current θ.
    pub fn objective_value(&self) -> f64 {
        let n = self.dims.n as f64;
        let p = self.dims.p as f64;
        let k = self.reterms.len();

        // log|L_ZZ| = sum of logdet of RE diagonal blocks
        let mut logdet_lzz = 0.0;
        for j in 0..k {
            logdet_lzz += logdet_block(&self.l_blocks[block_index(j, j)]);
        }

        // The last block L[k,k] is (p+1) × (p+1).
        // Its last diagonal element squared is the profiled σ² * n (or n-p for REML).
        let l_last = &self.l_blocks[block_index(k, k)];
        let l_dense = l_last.as_dense();
        let pp1 = l_dense.nrows();
        let last_diag = l_dense[(pp1 - 1, pp1 - 1)];
        let pwrss = last_diag * last_diag; // penalized weighted residual sum of squares

        if self.optsum.reml {
            // REML criterion
            // logdet of FE Cholesky factor
            let mut logdet_lxx = 0.0;
            for i in 0..(pp1 - 1) {
                let d = l_dense[(i, i)];
                if d > 0.0 {
                    logdet_lxx += d.ln();
                }
            }
            logdet_lzz + 2.0 * logdet_lxx + (n - p) * (1.0 + (2.0 * std::f64::consts::PI * pwrss / (n - p)).ln())
        } else {
            // ML deviance
            logdet_lzz + n * (1.0 + (2.0 * std::f64::consts::PI * pwrss / n).ln())
        }
    }

    /// Set θ, update L, and return the objective value.
    pub fn objective_at(&mut self, theta: &[f64]) -> Result<f64> {
        self.set_theta(theta)?;
        self.update_l()?;
        Ok(self.objective_value())
    }

    /// Fit the model by optimizing θ to minimize the objective.
    pub fn fit(&mut self, reml: bool) -> Result<&mut Self> {
        if self.optsum.feval > 0 {
            return Err(MixedModelError::AlreadyFitted);
        }

        // Check for constant response
        let y = self.y();
        let y0 = y[0];
        if y.iter().all(|&yi| (yi - y0).abs() < f64::EPSILON) {
            return Err(MixedModelError::ConstantResponse);
        }

        self.optsum.reml = reml;

        // Initial objective evaluation
        let theta0 = self.optsum.initial.clone();
        self.optsum.finitial = self.objective_at(&theta0)?;

        // Set up COBYLA optimizer
        let n_theta = theta0.len();
        let lb = self.lower_bounds();
        self.optsum.optimizer = Optimizer::Cobyla;

        // We need a closure that captures a mutable reference to self.
        // Since cobyla expects Fn, we use interior mutability.
        let a_blocks = self.a_blocks.clone();
        let l_blocks_template = self.l_blocks.clone();
        let reterms_template = self.reterms.clone();
        let dims = self.dims;
        let is_reml = reml;
        let parmap = self.parmap.clone();

        let best_theta = std::cell::RefCell::new(theta0.clone());
        let best_fmin = std::cell::Cell::new(f64::INFINITY);
        let feval_count = std::cell::Cell::new(0i64);
        let fit_log: std::cell::RefCell<Vec<(Vec<f64>, f64)>> = std::cell::RefCell::new(Vec::new());

        // Create mutable state for the closure
        let reterms_work = std::cell::RefCell::new(reterms_template.clone());
        let l_blocks_work = std::cell::RefCell::new(l_blocks_template);

        let objective_fn = |theta: &[f64], _data: &mut ()| -> f64 {
            feval_count.set(feval_count.get() + 1);

            // Set theta in working reterms
            let mut offset = 0;
            for rt in &mut *reterms_work.borrow_mut() {
                let nt = rt.n_theta();
                rt.set_theta(&theta[offset..offset + nt]);
                offset += nt;
            }

            // Update L from A
            let mut rw = reterms_work.borrow_mut();
            let mut lw = l_blocks_work.borrow_mut();
            let k = rw.len();
            let total = k + 1;

            // Copy and scale
            for j in 0..k {
                let idx = block_index(j, j);
                copy_scale_inflate(&mut lw[idx], &a_blocks[idx], &rw[j]);
            }
            for i in 1..k {
                for j in 0..i {
                    let idx = block_index(i, j);
                    copy_and_scale_offdiag(
                        &mut lw[idx],
                        &a_blocks[idx],
                        &rw[i],
                        &rw[j],
                    );
                }
            }
            for j in 0..k {
                let idx = block_index(k, j);
                copy_and_rmul_lambda(&mut lw[idx], &a_blocks[idx], &rw[j]);
            }
            let idx_kk = block_index(k, k);
            lw[idx_kk] = a_blocks[idx_kk].clone();

            // Blocked Cholesky
            for j in 0..total {
                for jj in 0..j {
                    let l_j_jj = lw[block_index(j, jj)].as_dense();
                    rank_k_downdate(&mut lw[block_index(j, j)], &l_j_jj);
                }
                if cholesky_block(&mut lw[block_index(j, j)]).is_err() {
                    return f64::INFINITY;
                }
                for i in (j + 1)..total {
                    for jj in 0..j {
                        let l_i_jj = lw[block_index(i, jj)].as_dense();
                        let l_j_jj = lw[block_index(j, jj)].as_dense();
                        subtract_product(&mut lw[block_index(i, j)], &l_i_jj, &l_j_jj);
                    }
                    let l_jj_clone = lw[block_index(j, j)].clone();
                    rdiv_lower_transpose(
                        &mut lw[block_index(i, j)],
                        &l_jj_clone,
                    );
                }
            }

            // Compute objective
            let n = dims.n as f64;
            let p = dims.p as f64;

            let mut logdet_lzz = 0.0;
            for j in 0..k {
                logdet_lzz += logdet_block(&lw[block_index(j, j)]);
            }

            let l_last = lw[block_index(k, k)].as_dense();
            let pp1 = l_last.nrows();
            let last_diag = l_last[(pp1 - 1, pp1 - 1)];
            let pwrss = last_diag * last_diag;

            let obj = if is_reml {
                let mut logdet_lxx = 0.0;
                for i in 0..(pp1 - 1) {
                    let d = l_last[(i, i)];
                    if d > 0.0 {
                        logdet_lxx += d.ln();
                    }
                }
                logdet_lzz + 2.0 * logdet_lxx
                    + (n - p) * (1.0 + (2.0 * std::f64::consts::PI * pwrss / (n - p)).ln())
            } else {
                logdet_lzz + n * (1.0 + (2.0 * std::f64::consts::PI * pwrss / n).ln())
            };

            drop(rw);
            drop(lw);

            fit_log.borrow_mut().push((theta.to_vec(), obj));
            if obj < best_fmin.get() {
                best_fmin.set(obj);
                *best_theta.borrow_mut() = theta.to_vec();
            }

            obj
        };

        // Build bounds for cobyla: each parameter gets (lb[i], +INF)
        let bounds: Vec<(f64, f64)> = lb
            .iter()
            .map(|&lo| (lo, f64::INFINITY))
            .collect();

        // Build constraint functions for cobyla.
        // For each parameter with a finite lower bound, add constraint: x[i] - lb[i] >= 0.
        // cobyla treats constraints as "should become non-negative".
        let constraint_fns: Vec<Box<dyn cobyla::Func<()>>> = lb
            .iter()
            .enumerate()
            .filter(|(_, &lo)| lo > f64::NEG_INFINITY)
            .map(|(i, &lo)| {
                Box::new(move |x: &[f64], _: &mut ()| -> f64 { x[i] - lo })
                    as Box<dyn cobyla::Func<()>>
            })
            .collect();
        let cons_refs: Vec<&dyn cobyla::Func<()>> = constraint_fns
            .iter()
            .map(|f| f.as_ref())
            .collect();

        // Determine max evaluations
        let maxeval = if self.optsum.max_feval > 0 {
            self.optsum.max_feval as usize
        } else {
            10000
        };

        // Set up stop tolerances
        let stop_tol = cobyla::StopTols {
            ftol_rel: self.optsum.ftol_rel,
            ftol_abs: self.optsum.ftol_abs,
            xtol_rel: self.optsum.xtol_rel,
            xtol_abs: self.optsum.xtol_abs.clone(),
            ..cobyla::StopTols::default()
        };

        // Run COBYLA optimizer
        let result = cobyla::minimize(
            objective_fn,
            &theta0,
            &bounds,
            &cons_refs,
            (),
            maxeval,
            cobyla::RhoBeg::All(0.75),
            Some(stop_tol),
        );

        // Extract values from Cell/RefCell after optimizer is done
        let mut best_theta_val = best_theta.borrow().clone();
        let mut best_fmin_val = best_fmin.get();

        match result {
            Ok((_, x_opt, fmin)) => {
                best_fmin_val = fmin;
                best_theta_val = x_opt;
            }
            Err((cobyla::FailStatus::RoundoffLimited, x_opt, _)) => {
                // Acceptable termination — use tracked best
                best_theta_val = x_opt;
            }
            Err((_, x_opt, fmin)) => {
                // For other "failures" that still produced a value,
                // accept if we got a finite result
                if fmin.is_finite() {
                    best_fmin_val = fmin;
                    best_theta_val = x_opt;
                } else {
                    return Err(MixedModelError::Optimization(
                        "COBYLA optimization failed".to_string(),
                    ));
                }
            }
        }

        // Install the optimal θ
        self.set_theta(&best_theta_val)?;
        self.update_l()?;

        // Check for near-zero parameters that can be set to exactly zero
        let mut xmin = best_theta_val.clone();
        let mut modified = false;
        for (i, (_, row, col)) in self.parmap.iter().enumerate() {
            if row == col && xmin[i] > 0.0 && xmin[i] < self.optsum.xtol_zero_abs {
                xmin[i] = 0.0;
                modified = true;
            }
        }
        if modified {
            let zero_obj = self.objective_at(&xmin)?;
            if zero_obj <= best_fmin_val + self.optsum.ftol_zero_abs {
                best_fmin_val = zero_obj;
                best_theta_val = xmin;
            } else {
                // Revert
                self.set_theta(&best_theta_val)?;
                self.update_l()?;
            }
        }

        // Finalize
        self.optsum.final_params = best_theta_val;
        self.optsum.fmin = best_fmin_val;
        self.optsum.feval = feval_count.get();
        self.optsum.return_value = "SUCCESS".to_string();
        self.optsum.fit_log = fit_log
            .into_inner()
            .into_iter()
            .map(|(theta, obj)| crate::types::FitLogEntry {
                theta,
                objective: obj,
            })
            .collect();

        Ok(self)
    }

    /// Extract the fixed-effects coefficients β from the Cholesky factor.
    pub fn beta(&self) -> DVector<f64> {
        let k = self.reterms.len();
        let l_last = self.l_blocks[block_index(k, k)].as_dense();
        let pp1 = l_last.nrows();
        let p = pp1 - 1;

        if p == 0 {
            return DVector::zeros(0);
        }

        // L_XX is the lower-left p×p submatrix of the last block
        let l_xx = l_last.view((0, 0), (p, p));

        // The last row of L_last (excluding the diagonal) contains L'u
        // β = L_XX^{-T} * L_last[p, 0..p]
        let mut beta = DVector::zeros(p);
        for j in 0..p {
            beta[j] = l_last[(pp1 - 1, j)];
        }

        // Solve L_XX' β = rhs (forward substitution with transpose)
        // L_XX is lower triangular, so L_XX' is upper triangular
        for i in (0..p).rev() {
            let mut s = beta[i];
            for j in (i + 1)..p {
                s -= l_xx[(j, i)] * beta[j];
            }
            beta[i] = s / l_xx[(i, i)];
        }

        beta
    }

    /// Standard deviation estimate (σ).
    pub fn sigma(&self) -> f64 {
        let k = self.reterms.len();
        let l_last = self.l_blocks[block_index(k, k)].as_dense();
        let pp1 = l_last.nrows();
        let last_diag = l_last[(pp1 - 1, pp1 - 1)].abs();

        let denom = if self.optsum.reml {
            (self.dims.n - self.dims.p) as f64
        } else {
            self.dims.n as f64
        };

        last_diag / denom.sqrt()
    }

    /// Penalized weighted residual sum of squares.
    pub fn pwrss(&self) -> f64 {
        let k = self.reterms.len();
        let l_last = self.l_blocks[block_index(k, k)].as_dense();
        let pp1 = l_last.nrows();
        let last_diag = l_last[(pp1 - 1, pp1 - 1)];
        last_diag * last_diag
    }

    /// Log-determinant of the RE Cholesky blocks.
    pub fn logdet_re(&self) -> f64 {
        let k = self.reterms.len();
        let mut ld = 0.0;
        for j in 0..k {
            ld += logdet_block(&self.l_blocks[block_index(j, j)]);
        }
        ld
    }

    /// Conditional modes of the random effects (the "u" vectors, on the spherical scale).
    pub fn ranef_u(&self) -> Vec<DMatrix<f64>> {
        let k = self.reterms.len();
        let beta = self.beta();

        let mut u_vecs: Vec<DVector<f64>> = Vec::new();
        for j in 0..k {
            let l_jj = self.l_blocks[block_index(j, j)].as_dense();
            let nranef_j = l_jj.nrows();

            // u_j = L_jj^{-1} * (rhs_j - sum_{i<j} L[j,i] * u_i - L[j,k] * beta_augmented)
            let idx_kj = block_index(k, j);
            let l_kj = self.l_blocks[idx_kj].as_dense();

            // rhs from L[k+1,j] row (the XY block row)
            // Actually we need to solve the blocked triangular system.
            // For now, extract from the last column of the L factor blocks.
            let mut rhs = DVector::zeros(nranef_j);

            // The RHS comes from L[k,j]' * beta + last_col contributions
            // This is a simplification - full implementation would do blocked solve
            // For the last column: L_last_col[j] = L[j, end_col]
            // TODO: implement full blocked back-solve for ranef

            u_vecs.push(rhs);
        }

        // Reshape u vectors into matrices (vsize × nlevels)
        self.reterms
            .iter()
            .zip(u_vecs)
            .map(|(rt, u)| {
                let vs = rt.vsize;
                let nl = rt.n_levels();
                DMatrix::from_column_slice(vs, nl, u.as_slice())
            })
            .collect()
    }

    /// Conditional modes on the original scale: b = Λ * u
    pub fn ranef_b(&self) -> Vec<DMatrix<f64>> {
        self.ranef_u()
            .into_iter()
            .zip(&self.reterms)
            .map(|(u, rt)| &rt.lambda * &u)
            .collect()
    }

    /// Grouping factor names.
    pub fn fnames(&self) -> Vec<String> {
        self.reterms.iter().map(|rt| rt.grouping_name.clone()).collect()
    }

    /// Number of θ parameters.
    pub fn n_theta(&self) -> usize {
        self.reterms.iter().map(|rt| rt.n_theta()).sum()
    }
}

impl MixedModelFit for LinearMixedModel {
    fn nobs(&self) -> usize {
        self.dims.n
    }

    fn dof(&self) -> usize {
        self.dims.p + self.n_theta() + 1 // +1 for σ
    }

    fn coef(&self) -> DVector<f64> {
        let beta = self.fixef();
        let mut full = DVector::from_element(self.feterm.piv.len(), 0.0);
        for (i, &val) in beta.iter().enumerate() {
            if i < self.feterm.piv.len() {
                full[self.feterm.piv[i]] = val;
            }
        }
        full
    }

    fn fixef(&self) -> DVector<f64> {
        self.beta()
    }

    fn coef_names(&self) -> Vec<String> {
        let mut names = self.feterm.cnames.clone();
        // Unpivot
        let mut result = vec![String::new(); names.len()];
        for (i, name) in names.drain(..).enumerate() {
            if i < self.feterm.piv.len() {
                result[self.feterm.piv[i]] = name;
            }
        }
        result
    }

    fn vcov(&self) -> DMatrix<f64> {
        let k = self.reterms.len();
        let l_last = self.l_blocks[block_index(k, k)].as_dense();
        let pp1 = l_last.nrows();
        let p = pp1 - 1;

        if p == 0 {
            return DMatrix::zeros(0, 0);
        }

        let l_xx = l_last.view((0, 0), (p, p)).clone_owned();

        // L_inv = L_XX^{-1}
        let mut l_inv = DMatrix::<f64>::identity(p, p);
        // Forward solve: L_XX * L_inv = I
        for j in 0..p {
            for i in j..p {
                let mut s = l_inv[(i, j)];
                for k2 in j..i {
                    s -= l_xx[(i, k2)] * l_inv[(k2, j)];
                }
                l_inv[(i, j)] = s / l_xx[(i, i)];
            }
        }

        let sigma_sq = self.dispersion(true);
        let vcov_perm = sigma_sq * (&l_inv.transpose() * &l_inv);

        // Unpivot
        let full_p = self.feterm.piv.len();
        if p == full_p {
            let mut result = DMatrix::zeros(full_p, full_p);
            for i in 0..full_p {
                for j in 0..full_p {
                    result[(self.feterm.piv[i], self.feterm.piv[j])] = vcov_perm[(i, j)];
                }
            }
            result
        } else {
            let nan = 0.0_f64 / 0.0_f64;
            let mut result = DMatrix::from_element(full_p, full_p, nan);
            for i in 0..p {
                for j in 0..p {
                    result[(self.feterm.piv[i], self.feterm.piv[j])] = vcov_perm[(i, j)];
                }
            }
            result
        }
    }

    fn stderror(&self) -> DVector<f64> {
        let vc = self.vcov();
        DVector::from_iterator(vc.nrows(), (0..vc.nrows()).map(|i| vc[(i, i)].sqrt()))
    }

    fn fitted(&self) -> DVector<f64> {
        let beta = self.beta();
        let x = self.feterm.full_rank_x();
        let mut yhat = x * &beta;

        // Add random effects contribution
        for (rt, b) in self.reterms.iter().zip(self.ranef_b()) {
            // y += Z * b (using sparse multiplication via refs)
            let bvec = DVector::from_column_slice(b.as_slice());
            for (obs, &ref_idx) in rt.refs.iter().enumerate() {
                let r = ref_idx as usize;
                for s in 0..rt.vsize {
                    yhat[obs] += rt.z[(s, obs)] * bvec[r * rt.vsize + s];
                }
            }
        }

        yhat
    }

    fn residuals(&self) -> DVector<f64> {
        let y = self.y();
        let yhat = self.fitted();
        y - yhat
    }

    fn response(&self) -> &DVector<f64> {
        // Return a reference to the last column of xy
        // Since we can't return a column view with the right lifetime,
        // this is a limitation. For now, we store y separately.
        // TODO: store y as a separate field
        unimplemented!("Use y() method instead")
    }

    fn model_matrix(&self) -> &DMatrix<f64> {
        &self.feterm.x
    }

    fn objective(&self) -> f64 {
        self.objective_value()
    }

    fn loglikelihood(&self) -> f64 {
        let obj = self.objective_value();
        let n = self.dims.n as f64;
        if self.optsum.reml {
            // REML log-likelihood
            let p = self.dims.p as f64;
            -0.5 * (obj - (n - p) * (2.0 * std::f64::consts::PI).ln())
        } else {
            // ML log-likelihood
            -0.5 * (obj - n * (2.0 * std::f64::consts::PI).ln())
        }
    }

    fn is_fitted(&self) -> bool {
        self.optsum.feval > 0
    }

    fn is_singular(&self) -> bool {
        let theta = self.theta();
        let lb = self.lower_bounds();
        theta.iter().zip(lb.iter()).any(|(&t, &l)| {
            l >= 0.0 && (t - l).abs() < f64::EPSILON
        })
    }

    fn opt_summary(&self) -> &OptSummary {
        &self.optsum
    }

    fn theta(&self) -> Vec<f64> {
        LinearMixedModel::theta(self)
    }

    fn dispersion(&self, sqr: bool) -> f64 {
        let s = self.sigma();
        if sqr { s * s } else { s }
    }

    fn ranef(&self) -> Vec<DMatrix<f64>> {
        self.ranef_b()
    }
}

// === Helper functions for model construction ===

/// Build the fixed-effects model matrix from formula and data.
fn build_fixed_effects_matrix(
    formula: &Formula,
    data: &DataFrame,
) -> Result<(DMatrix<f64>, Vec<String>)> {
    use crate::formula::FixedTerm;

    let n = data.nrow();
    let mut columns: Vec<DVector<f64>> = Vec::new();
    let mut names: Vec<String> = Vec::new();

    // Check if we have an intercept
    let has_intercept = formula.has_intercept();

    if has_intercept {
        columns.push(DVector::from_element(n, 1.0));
        names.push("(Intercept)".to_string());
    }

    for term in &formula.fixed_terms {
        match term {
            FixedTerm::Intercept | FixedTerm::NoIntercept => {
                // Already handled
            }
            FixedTerm::Column(name) => {
                match data.column(name) {
                    Some(Column::Numeric(v)) => {
                        columns.push(DVector::from_column_slice(v));
                        names.push(name.clone());
                    }
                    Some(Column::Categorical(cat)) => {
                        // Dummy coding (treatment/reference coding)
                        // Skip the first level (reference)
                        for (lvl_idx, lvl) in cat.levels.iter().enumerate().skip(1) {
                            let col: Vec<f64> = cat
                                .refs
                                .iter()
                                .map(|&r| if r as usize == lvl_idx { 1.0 } else { 0.0 })
                                .collect();
                            columns.push(DVector::from_column_slice(&col));
                            names.push(format!("{}: {}", name, lvl));
                        }
                    }
                    None => {
                        return Err(MixedModelError::InvalidArgument(format!(
                            "Column '{}' not found in data",
                            name
                        )));
                    }
                }
            }
            FixedTerm::Interaction(vars) => {
                // For now, support interaction of two numeric variables
                if vars.len() == 2 {
                    if let (Some(Column::Numeric(a)), Some(Column::Numeric(b))) =
                        (data.column(&vars[0]), data.column(&vars[1]))
                    {
                        let col: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x * y).collect();
                        columns.push(DVector::from_column_slice(&col));
                        names.push(format!("{}:{}", vars[0], vars[1]));
                    }
                }
            }
            FixedTerm::Nested(_) => {
                // Nesting is expanded into main effect + interaction during parsing
            }
        }
    }

    if columns.is_empty() {
        // No fixed effects at all — create an empty matrix
        return Ok((DMatrix::zeros(n, 0), vec![]));
    }

    let p = columns.len();
    let mut x = DMatrix::zeros(n, p);
    for (j, col) in columns.iter().enumerate() {
        x.set_column(j, col);
    }

    Ok((x, names))
}

/// Build a ReMat from a random term specification and data.
fn build_re_mat(
    rt: &crate::formula::RandomTerm,
    data: &DataFrame,
    n: usize,
) -> Result<ReMat> {
    use crate::formula::{FixedTerm, GroupingFactor};

    // Get grouping factor
    let (group_name, refs, levels) = match &rt.grouping {
        GroupingFactor::Single(name) => {
            let cat = data.categorical(name).ok_or_else(|| {
                MixedModelError::InvalidArgument(format!(
                    "Grouping factor '{}' not found or not categorical",
                    name
                ))
            })?;
            (name.clone(), cat.refs.clone(), cat.levels.clone())
        }
        GroupingFactor::Interaction(names) => {
            // Create interaction levels
            let cats: Vec<&crate::model::data::CategoricalColumn> = names
                .iter()
                .map(|name| {
                    data.categorical(name).ok_or_else(|| {
                        MixedModelError::InvalidArgument(format!(
                            "Grouping factor '{}' not found",
                            name
                        ))
                    })
                })
                .collect::<Result<Vec<_>>>()?;

            let group_name = names.join(" & ");
            let mut level_map = indexmap::IndexMap::new();
            let mut refs = Vec::with_capacity(n);

            for obs in 0..n {
                let key: String = cats
                    .iter()
                    .map(|c| c.levels[c.refs[obs] as usize].clone())
                    .collect::<Vec<_>>()
                    .join("_");
                let idx = level_map.len();
                let idx = *level_map.entry(key.clone()).or_insert(idx);
                refs.push(idx as u32);
            }

            let levels: Vec<String> = level_map.keys().cloned().collect();
            (group_name, refs, levels)
        }
    };

    // Build the Z matrix (transposed: s × n)
    let mut z_rows: Vec<DVector<f64>> = Vec::new();
    let mut cnames: Vec<String> = Vec::new();

    let has_re_intercept = rt.terms.iter().any(|t| matches!(t, FixedTerm::Intercept))
        || rt.terms.is_empty();

    if has_re_intercept {
        z_rows.push(DVector::from_element(n, 1.0));
        cnames.push("(Intercept)".to_string());
    }

    for term in &rt.terms {
        match term {
            FixedTerm::Intercept | FixedTerm::NoIntercept => {}
            FixedTerm::Column(name) => {
                let col = data.numeric(name).ok_or_else(|| {
                    MixedModelError::InvalidArgument(format!(
                        "Random effect variable '{}' not found or not numeric",
                        name
                    ))
                })?;
                z_rows.push(DVector::from_column_slice(col));
                cnames.push(name.clone());
            }
            _ => {}
        }
    }

    let vsize = z_rows.len();
    let mut z = DMatrix::zeros(vsize, n);
    for (i, row) in z_rows.iter().enumerate() {
        z.set_row(i, &row.transpose());
    }

    let mut remat = ReMat::new(group_name, refs, levels, cnames, z);

    if rt.zerocorr {
        remat.zerocorr();
    }

    Ok(remat)
}

/// Build the parameter map: Vec<(block_idx, row, col)> for each θ element.
fn build_parmap(reterms: &[ReMat]) -> Vec<(usize, usize, usize)> {
    let mut parmap = Vec::new();
    for (block, rt) in reterms.iter().enumerate() {
        for &ind in &rt.inds {
            let s = rt.vsize;
            let col = ind / s;
            let row = ind % s;
            parmap.push((block, row, col));
        }
    }
    parmap
}

/// Create the A (cross-product) and L (Cholesky) block arrays.
fn create_al(reterms: &[ReMat], xy: &FeMat) -> (Vec<MatrixBlock>, Vec<MatrixBlock>) {
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
    for j in 0..k {
        let block = compute_fe_re_cross_product(xy, &reterms[j]);
        a.push(block.clone());
        l.push(block);
    }

    // FE × FE block: [X|y]' [X|y]
    let wtxy = &xy.wtxy;
    let feblock = MatrixBlock::Dense(wtxy.transpose() * wtxy);
    a.push(feblock.clone());
    l.push(feblock);

    (a, l)
}

/// Compute Z_i' Z_j for two random effects terms.
fn compute_re_cross_product(a: &ReMat, b: &ReMat) -> MatrixBlock {
    let nranef_a = a.n_ranef();
    let nranef_b = b.n_ranef();

    if std::ptr::eq(a, b) && a.vsize == 1 {
        // Scalar RE: diagonal result
        let n_levels = a.n_levels();
        let mut diag = DVector::zeros(n_levels);
        for (obs, &ref_idx) in a.refs.iter().enumerate() {
            let r = ref_idx as usize;
            diag[r] += a.wtz[(0, obs)] * a.wtz[(0, obs)];
        }
        MatrixBlock::Diagonal(diag)
    } else if std::ptr::eq(a, b) && a.vsize > 1 {
        // Vector RE, same term: block-diagonal result
        // Each level k gets a vsize × vsize block: sum_{obs with ref==k} wtz[:,obs] * wtz[:,obs]'
        let s = a.vsize;
        let n_levels = a.n_levels();
        let mut blocks: Vec<DMatrix<f64>> = (0..n_levels)
            .map(|_| DMatrix::zeros(s, s))
            .collect();

        for (obs, &ref_idx) in a.refs.iter().enumerate() {
            let k = ref_idx as usize;
            let blk = &mut blocks[k];
            for si in 0..s {
                let wtz_si = a.wtz[(si, obs)];
                for sj in 0..s {
                    blk[(si, sj)] += wtz_si * a.wtz[(sj, obs)];
                }
            }
        }
        MatrixBlock::BlockDiagonal(blocks)
    } else {
        // General case: dense result (different terms)
        let mut result = DMatrix::zeros(nranef_a, nranef_b);
        let n = a.refs.len();

        for obs in 0..n {
            let ri = a.refs[obs] as usize;
            let rj = b.refs[obs] as usize;
            for si in 0..a.vsize {
                for sj in 0..b.vsize {
                    result[(ri * a.vsize + si, rj * b.vsize + sj)] +=
                        a.wtz[(si, obs)] * b.wtz[(sj, obs)];
                }
            }
        }
        MatrixBlock::Dense(result)
    }
}

/// Compute [X|y]' Z_j.
fn compute_fe_re_cross_product(xy: &FeMat, re: &ReMat) -> MatrixBlock {
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

// === Block Cholesky helper functions ===

/// Copy A to L and scale blockwise: L_jj = Λ_j' A_jj Λ_j + I
///
/// A is (nranef × nranef) where nranef = vsize * nlevels.
/// Λ is (vsize × vsize). Scaling is applied to each (vsize × vsize)
/// sub-block of A independently.
fn copy_scale_inflate(l: &mut MatrixBlock, a: &MatrixBlock, re: &ReMat) {
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
            (l_block, _) => {
                let a_dense = a.as_dense();
                let n = a_dense.nrows();
                let mut result = DMatrix::zeros(n, n);
                for i in 0..n {
                    for j in 0..n {
                        result[(i, j)] = lam_sq * a_dense[(i, j)];
                    }
                    result[(i, i)] += 1.0;
                }
                *l_block = MatrixBlock::Dense(result);
            }
        }
    } else {
        // Vector RE: apply Λ blockwise
        let lambda = &re.lambda;
        let lambda_t = lambda.transpose();

        match a {
            MatrixBlock::BlockDiagonal(a_blocks) => {
                // BlockDiagonal path: O(nlevels * s³) — only process diagonal blocks
                let nlevels = a_blocks.len();
                let mut l_blocks = Vec::with_capacity(nlevels);
                for k in 0..nlevels {
                    // Λ' * A_block_k * Λ + I
                    let scaled = &lambda_t * &a_blocks[k] * lambda;
                    let mut blk = scaled;
                    for i in 0..s {
                        blk[(i, i)] += 1.0;
                    }
                    l_blocks.push(blk);
                }
                *l = MatrixBlock::BlockDiagonal(l_blocks);
            }
            _ => {
                // Dense fallback: apply Λ blockwise to each (s×s) sub-block
                let a_dense = a.as_dense();
                let nranef = a_dense.nrows();
                let nlevels = nranef / s;
                let mut result = DMatrix::zeros(nranef, nranef);

                for bk in 0..nlevels {
                    for bl in 0..nlevels {
                        let mut a_block = DMatrix::zeros(s, s);
                        for i in 0..s {
                            for j in 0..s {
                                a_block[(i, j)] = a_dense[(bk * s + i, bl * s + j)];
                            }
                        }
                        let scaled = &lambda_t * &a_block * lambda;
                        for i in 0..s {
                            for j in 0..s {
                                result[(bk * s + i, bl * s + j)] = scaled[(i, j)];
                            }
                        }
                    }
                }
                for i in 0..nranef {
                    result[(i, i)] += 1.0;
                }
                *l = MatrixBlock::Dense(result);
            }
        }
    }
}

/// Copy off-diagonal block and scale blockwise: L_ij = Λ_i' A_ij Λ_j
///
/// A is (nranef_i × nranef_j). Λ_i is (vsize_i × vsize_i), Λ_j is (vsize_j × vsize_j).
fn copy_and_scale_offdiag(l: &mut MatrixBlock, a: &MatrixBlock, re_i: &ReMat, re_j: &ReMat) {
    let si = re_i.vsize;
    let sj = re_j.vsize;
    let a_dense = a.as_dense();
    let nranef_i = a_dense.nrows();
    let nranef_j = a_dense.ncols();
    let nlevels_i = nranef_i / si;
    let nlevels_j = nranef_j / sj;
    let lambda_i_t = re_i.lambda.transpose();
    let lambda_j = &re_j.lambda;

    let mut result = DMatrix::zeros(nranef_i, nranef_j);

    for bi in 0..nlevels_i {
        for bj in 0..nlevels_j {
            let mut a_block = DMatrix::zeros(si, sj);
            for i in 0..si {
                for j in 0..sj {
                    a_block[(i, j)] = a_dense[(bi * si + i, bj * sj + j)];
                }
            }
            let scaled = &lambda_i_t * &a_block * lambda_j;
            for i in 0..si {
                for j in 0..sj {
                    result[(bi * si + i, bj * sj + j)] = scaled[(i, j)];
                }
            }
        }
    }
    *l = MatrixBlock::Dense(result);
}

/// Copy and right-multiply blockwise by Λ: L_kj = A_kj Λ_j
///
/// A is (pp1 × nranef_j). Λ_j is (vsize_j × vsize_j).
/// Each column-block of size vsize_j gets right-multiplied by Λ_j.
fn copy_and_rmul_lambda(l: &mut MatrixBlock, a: &MatrixBlock, re_j: &ReMat) {
    let sj = re_j.vsize;
    let a_dense = a.as_dense();
    let nrows = a_dense.nrows();
    let ncols = a_dense.ncols();
    let nblocks = ncols / sj;
    let lambda_j = &re_j.lambda;

    let mut result = DMatrix::zeros(nrows, ncols);

    for b in 0..nblocks {
        // Extract the (nrows × sj) column block
        let mut block = DMatrix::zeros(nrows, sj);
        for i in 0..nrows {
            for j in 0..sj {
                block[(i, j)] = a_dense[(i, b * sj + j)];
            }
        }
        // Right-multiply by Λ_j
        let scaled = &block * lambda_j;
        for i in 0..nrows {
            for j in 0..sj {
                result[(i, b * sj + j)] = scaled[(i, j)];
            }
        }
    }
    *l = MatrixBlock::Dense(result);
}

/// Rank-k downdate: C -= A * A' (modifies diagonal block)
fn rank_k_downdate(c: &mut MatrixBlock, a: &DMatrix<f64>) {
    match c {
        MatrixBlock::Dense(c_mat) => {
            *c_mat -= a * a.transpose();
        }
        MatrixBlock::Diagonal(c_diag) => {
            // A * A' diagonal entries
            for i in 0..c_diag.len() {
                let row = a.row(i);
                c_diag[i] -= row.dot(&row);
            }
        }
        MatrixBlock::BlockDiagonal(blocks) => {
            // For each block k, downdate by the corresponding rows of A
            let mut row_offset = 0;
            for blk in blocks.iter_mut() {
                let s = blk.nrows();
                // Extract the s rows from A starting at row_offset
                let a_sub = a.rows(row_offset, s);
                // blk -= a_sub * a_sub'
                *blk -= &a_sub * a_sub.transpose();
                row_offset += s;
            }
        }
    }
}

/// Subtract product: C -= A * B'
fn subtract_product(c: &mut MatrixBlock, a: &DMatrix<f64>, b: &DMatrix<f64>) {
    match c {
        MatrixBlock::Dense(c_mat) => {
            *c_mat -= a * b.transpose();
        }
        MatrixBlock::BlockDiagonal(_) => {
            // Promote to dense — off-diagonal updates destroy block-diagonal structure
            let mut c_dense = c.as_dense();
            c_dense -= a * b.transpose();
            *c = MatrixBlock::Dense(c_dense);
        }
        _ => {
            let mut c_dense = c.as_dense();
            c_dense -= a * b.transpose();
            *c = MatrixBlock::Dense(c_dense);
        }
    }
}

/// In-place Cholesky of a block (handles zero diagonal gracefully).
fn cholesky_block(block: &mut MatrixBlock) -> Result<()> {
    match block {
        MatrixBlock::Diagonal(diag) => {
            for i in 0..diag.len() {
                if diag[i] <= 0.0 {
                    if diag[i] < -1e-8 {
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
            for blk in blocks.iter_mut() {
                let n = blk.nrows();
                for j in 0..n {
                    let mut s = blk[(j, j)];
                    for k in 0..j {
                        s -= blk[(j, k)] * blk[(j, k)];
                    }
                    if s <= 0.0 {
                        if s < -1e-8 {
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
            for j in 0..n {
                // Compute L[j,j]
                let mut s = mat[(j, j)];
                for k in 0..j {
                    s -= mat[(j, k)] * mat[(j, k)];
                }
                if s <= 0.0 {
                    if s < -1e-8 {
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
    }
}

/// Right-divide by lower triangular transpose: A = A * L^{-T}
fn rdiv_lower_transpose(a: &mut MatrixBlock, l: &MatrixBlock) {
    match l {
        MatrixBlock::BlockDiagonal(l_blocks) => {
            // L is block-diagonal: solve each column-block independently
            // A[:,block_k] = A[:,block_k] * L_k^{-T}
            match a {
                MatrixBlock::Dense(a_mat) => {
                    let mut col_offset = 0;
                    for l_blk in l_blocks {
                        let s = l_blk.nrows();
                        // Solve the s-column slice of A against L_k
                        for j in 0..s {
                            let cj = col_offset + j;
                            if l_blk[(j, j)].abs() < 1e-30 {
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
                MatrixBlock::BlockDiagonal(_) => {
                    // Both block-diagonal: promote A to dense, then solve
                    let mut a_dense = a.as_dense();
                    let mut col_offset = 0;
                    for l_blk in l_blocks {
                        let s = l_blk.nrows();
                        for j in 0..s {
                            let cj = col_offset + j;
                            if l_blk[(j, j)].abs() < 1e-30 {
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
                            if l_blk[(j, j)].abs() < 1e-30 {
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
            // L is Dense or Diagonal — original logic
            let l_dense = l.as_dense();
            let n = l_dense.nrows();

            match a {
                MatrixBlock::Dense(a_mat) => {
                    for j in 0..n {
                        if l_dense[(j, j)].abs() < 1e-30 {
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
                MatrixBlock::Diagonal(a_diag) => {
                    match l {
                        MatrixBlock::Diagonal(l_diag) => {
                            for i in 0..a_diag.len() {
                                if l_diag[i].abs() > 1e-30 {
                                    a_diag[i] /= l_diag[i];
                                } else {
                                    a_diag[i] = 0.0;
                                }
                            }
                        }
                        _ => {
                            let mut a_dense = DMatrix::from_diagonal(a_diag);
                            for j in 0..n {
                                if l_dense[(j, j)].abs() < 1e-30 {
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
                MatrixBlock::BlockDiagonal(_) => {
                    // Promote to dense and solve
                    let mut a_dense = a.as_dense();
                    for j in 0..n {
                        if l_dense[(j, j)].abs() < 1e-30 {
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
fn logdet_block(block: &MatrixBlock) -> f64 {
    match block {
        MatrixBlock::Diagonal(diag) => {
            diag.iter().filter(|&&d| d > 0.0).map(|d| d.ln()).sum::<f64>() * 2.0
        }
        MatrixBlock::BlockDiagonal(blocks) => {
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_index() {
        assert_eq!(block_index(0, 0), 0);
        assert_eq!(block_index(1, 0), 1);
        assert_eq!(block_index(1, 1), 2);
        assert_eq!(block_index(2, 0), 3);
        assert_eq!(block_index(2, 1), 4);
        assert_eq!(block_index(2, 2), 5);
    }

    #[test]
    fn test_cholesky_block_diagonal() {
        let mut block = MatrixBlock::Diagonal(DVector::from_vec(vec![4.0, 9.0, 16.0]));
        cholesky_block(&mut block).unwrap();
        if let MatrixBlock::Diagonal(d) = &block {
            assert!((d[0] - 2.0).abs() < 1e-10);
            assert!((d[1] - 3.0).abs() < 1e-10);
            assert!((d[2] - 4.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_cholesky_block_dense() {
        // [[4, 2], [2, 5]] → L = [[2, 0], [1, 2]]
        let mut block = MatrixBlock::Dense(DMatrix::from_row_slice(2, 2, &[4.0, 2.0, 2.0, 5.0]));
        cholesky_block(&mut block).unwrap();
        if let MatrixBlock::Dense(m) = &block {
            assert!((m[(0, 0)] - 2.0).abs() < 1e-10);
            assert!((m[(1, 0)] - 1.0).abs() < 1e-10);
            assert!((m[(1, 1)] - 2.0).abs() < 1e-10);
            assert!(m[(0, 1)].abs() < 1e-10);
        }
    }

    #[test]
    fn test_logdet_block() {
        let block = MatrixBlock::Diagonal(DVector::from_vec(vec![2.0, 3.0]));
        let ld = logdet_block(&block);
        // logdet = 2 * (ln(2) + ln(3)) = 2 * ln(6)
        assert!((ld - 2.0 * 6.0_f64.ln()).abs() < 1e-10);
    }
}
