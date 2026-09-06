//! Dense reference implementation of the profiled (RE)ML deviance gradient
//! with respect to θ. See `docs/profiled_deviance_gradient.md` for the
//! derivation.
//!
//! This is an O(q³) oracle for small models: it materializes `Z`, `Λ`, `M`
//! and their inverses densely and is used to validate the formulas (and,
//! later, the blocked production gradient) against the finite-difference
//! derivatives the optimizer certificate already computes. It is not on
//! any fit path.

use super::*;

/// Objective and gradient produced by the dense reference.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct DenseReferenceGradient {
    /// Profiled deviance at `theta` (without the observation-weight
    /// constant), on the same scale as `objective_at`.
    pub(crate) objective: f64,
    /// `∂f/∂θ_k` in the model's θ order.
    pub(crate) gradient: Vec<f64>,
}

// The dense reference is a test oracle for the blocked gradient.
#[cfg_attr(not(test), allow(dead_code))]
impl LinearMixedModel {
    /// Dense reference objective and gradient at `theta` for the given
    /// criterion and the model's current fixed-σ setting.
    pub(crate) fn dense_reference_gradient(
        &self,
        theta: &[f64],
        reml: bool,
    ) -> Result<DenseReferenceGradient> {
        let n = self.dims.n;
        let p = self.feterm.rank;
        let n_theta = self.n_theta();
        if theta.len() != n_theta {
            return Err(MixedModelError::DimensionMismatch(format!(
                "theta length {} does not match {n_theta}",
                theta.len()
            )));
        }
        let sqrtwts: Vec<f64> = if self.sqrtwts.is_empty() {
            vec![1.0; n]
        } else {
            self.sqrtwts.clone()
        };

        // Weighted design pieces.
        let x_full = self.feterm.full_rank_x();
        let mut x_w = DMatrix::zeros(n, p);
        for col in 0..p {
            for obs in 0..n {
                x_w[(obs, col)] = sqrtwts[obs] * x_full[(obs, col)];
            }
        }
        let y_w = DVector::from_iterator(n, self.y.iter().zip(&sqrtwts).map(|(y, sw)| sw * y));

        // Term offsets, weighted Z, and Λ(θ).
        let mut offsets = Vec::with_capacity(self.reterms.len());
        let mut q = 0usize;
        for re in &self.reterms {
            offsets.push(q);
            q += re.n_ranef();
        }
        let mut z_w = DMatrix::zeros(n, q);
        let mut lambda = DMatrix::zeros(q, q);
        let mut theta_offset = 0usize;
        let mut lambda_blocks = Vec::with_capacity(self.reterms.len());
        for (j, re) in self.reterms.iter().enumerate() {
            let s = re.vsize;
            let mut lam = DMatrix::zeros(s, s);
            for (m, &idx) in re.inds.iter().enumerate() {
                lam[idx] = theta[theta_offset + m];
            }
            theta_offset += re.inds.len();
            for (obs, &level) in re.refs.iter().enumerate() {
                let base = offsets[j] + level as usize * s;
                for c in 0..s {
                    z_w[(obs, base + c)] = sqrtwts[obs] * re.z[(c, obs)];
                }
            }
            for level in 0..re.n_levels() {
                let base = offsets[j] + level * s;
                for r in 0..s {
                    for c in 0..s {
                        lambda[(base + r, base + c)] = lam[(r, c)];
                    }
                }
            }
            lambda_blocks.push(lam);
        }

        let z_t = &z_w * &lambda;
        let a_zz = z_w.transpose() * &z_w;
        let m = z_t.transpose() * &z_t + DMatrix::identity(q, q);

        // Penalized least squares for (û, β̂).
        let r = z_t.transpose() * &x_w; // q × p
        let xtx = x_w.transpose() * &x_w;
        let mut k = DMatrix::zeros(q + p, q + p);
        k.view_mut((0, 0), (q, q)).copy_from(&m);
        k.view_mut((0, q), (q, p)).copy_from(&r);
        k.view_mut((q, 0), (p, q)).copy_from(&r.transpose());
        k.view_mut((q, q), (p, p)).copy_from(&xtx);
        let mut rhs = DVector::zeros(q + p);
        rhs.rows_mut(0, q).copy_from(&(z_t.transpose() * &y_w));
        rhs.rows_mut(q, p).copy_from(&(x_w.transpose() * &y_w));
        let solution = k
            .clone()
            .cholesky()
            .map(|chol| chol.solve(&rhs))
            .or_else(|| k.lu().solve(&rhs))
            .ok_or_else(|| {
                MixedModelError::Optimization(
                    "dense reference: singular penalized least squares system".to_string(),
                )
            })?;
        let u_hat = solution.rows(0, q).into_owned();
        let beta_hat = solution.rows(q, p).into_owned();
        let residual = &y_w - &x_w * &beta_hat - &z_t * &u_hat;
        let pwrss = residual.dot(&residual) + u_hat.dot(&u_hat);

        let m_chol = m.clone().cholesky().ok_or_else(|| {
            MixedModelError::Optimization("dense reference: M is not positive definite".to_string())
        })?;
        let logdet_m = 2.0 * m_chol.l().diagonal().iter().map(|d| d.ln()).sum::<f64>();
        let m_inv = m_chol.inverse();

        // REML profiled fixed-effects cross product and its inverse.
        let (logdet_c, c_inv, q_mat) = if reml {
            let q_mat = &m_inv * &r; // q × p
            let c = &xtx - r.transpose() * &q_mat;
            let c_chol = c.clone().cholesky().ok_or_else(|| {
                MixedModelError::Optimization(
                    "dense reference: profiled X'X is not positive definite".to_string(),
                )
            })?;
            let logdet_c = 2.0 * c_chol.l().diagonal().iter().map(|d| d.ln()).sum::<f64>();
            (logdet_c, Some(c_chol.inverse()), Some(q_mat))
        } else {
            (0.0, None, None)
        };

        let denomdf = if reml { (n - p) as f64 } else { n as f64 };
        let logdet = logdet_m + logdet_c;
        let objective = Self::objective_from_components(logdet, pwrss, denomdf, self.optsum.sigma);

        // Shared pieces for the gradient.
        let p_mat = &m_inv * lambda.transpose() * &a_zz; // M⁻¹ Λᵀ A_ZZ
        let g = z_w.transpose() * &residual; // Z_wᵀ r̂
        let t_mat = z_w.transpose() * &x_w; // Z_wᵀ X_w (q × p)
        let v_mat = q_mat.as_ref().map(|q_mat| &a_zz * &lambda * q_mat); // A_ZZ Λ M⁻¹ R

        let mut gradient = Vec::with_capacity(n_theta);
        let mut theta_offset = 0usize;
        for (j, re) in self.reterms.iter().enumerate() {
            let s = re.vsize;
            let n_levels = re.n_levels();
            for &idx in &re.inds {
                let a = idx % s;
                let b = idx / s;
                let mut d_logdet_m = 0.0;
                let mut d_pwrss = 0.0;
                let mut d_logdet_c = 0.0;
                for level in 0..n_levels {
                    let row_a = offsets[j] + level * s + a;
                    let row_b = offsets[j] + level * s + b;
                    d_logdet_m += 2.0 * p_mat[(row_b, row_a)];
                    d_pwrss -= 2.0 * g[row_a] * u_hat[row_b];
                    if let (Some(c_inv), Some(q_mat), Some(v_mat)) =
                        (c_inv.as_ref(), q_mat.as_ref(), v_mat.as_ref())
                    {
                        let t_row = t_mat.row(row_a);
                        let q_row = q_mat.row(row_b);
                        let v_row = v_mat.row(row_a);
                        let qct = (q_row * c_inv * t_row.transpose())[(0, 0)];
                        let vcq = (v_row * c_inv * q_row.transpose())[(0, 0)];
                        d_logdet_c += -2.0 * qct + 2.0 * vcq;
                    }
                }
                let d_logdet = d_logdet_m + d_logdet_c;
                let value = match self.optsum.sigma {
                    Some(sigma) => d_logdet + d_pwrss / (sigma * sigma),
                    None => d_logdet + denomdf * d_pwrss / pwrss,
                };
                gradient.push(value);
            }
            theta_offset += re.inds.len();
        }
        debug_assert_eq!(theta_offset, n_theta);

        Ok(DenseReferenceGradient {
            objective,
            gradient,
        })
    }
}

/// Largest number of dense entries the block selected inverse may hold
/// before the blocked gradient declines (callers then fall back to finite
/// differences). Crossed designs with a few hundred random effects per
/// term stay far below this.
const BLOCKED_GRADIENT_MAX_DENSE_ENTRIES: usize = 4_000_000;

/// Triangular-solve guard, matching the block solve helpers.
const GRADIENT_SOLVE_ZERO_TOLERANCE: f64 = 1e-30;

#[inline]
fn guarded_div(numerator: f64, denominator: f64) -> f64 {
    if denominator.abs() < GRADIENT_SOLVE_ZERO_TOLERANCE {
        0.0
    } else {
        numerator / denominator
    }
}

// Model-level entry points: the production consumers (optimizer oracle,
// certificate evidence, Kenward-Roger Hessian) run on cloned parts through
// `ProfiledGradientInputs`; these remain as the test oracles.
#[cfg_attr(not(test), allow(dead_code))]
impl LinearMixedModel {
    /// Objective and analytic gradient at `theta` (sets θ and refactorizes,
    /// like `objective_at`). See `docs/profiled_deviance_gradient.md`.
    pub(crate) fn objective_and_gradient_at(&mut self, theta: &[f64]) -> Result<(f64, Vec<f64>)> {
        self.set_theta(theta)?;
        self.update_l()?;
        // Same scale as `objective_at` (observation-weight constant applied).
        let objective = self.objective_value();
        let gradient = self.profiled_gradient_at_current_theta()?;
        Ok((objective, gradient))
    }

    /// Analytic gradient of the profiled deviance at the model's current θ
    /// from the current blocked factor (see
    /// [`ProfiledGradientInputs::profiled_gradient`]).
    pub(crate) fn profiled_gradient_at_current_theta(&self) -> Result<Vec<f64>> {
        ProfiledGradientInputs {
            a_blocks: &self.a_blocks,
            l_blocks: &self.l_blocks,
            reterms: &self.reterms,
            dims: self.dims,
            reml: self.optsum.reml,
            sigma: self.optsum.sigma,
        }
        .profiled_gradient()
    }
}

/// What the analytic gradient reads: the θ-invariant `A` blocks, the factor
/// `L` at the current θ (as `update_l_from_parts` leaves it), the
/// random-effects terms carrying Λ, and the objective's settings. Built from
/// a `LinearMixedModel` or from the optimizer's cloned work blocks.
pub(super) struct ProfiledGradientInputs<'a> {
    pub(super) a_blocks: &'a [MatrixBlock],
    pub(super) l_blocks: &'a [MatrixBlock],
    pub(super) reterms: &'a [ReMat],
    pub(super) dims: ModelDims,
    pub(super) reml: bool,
    pub(super) sigma: Option<f64>,
}

impl ProfiledGradientInputs<'_> {
    /// Analytic gradient of the profiled deviance at the factor's θ, from
    /// the blocked factor and the θ-invariant `A` blocks only (no pass over
    /// the observations).
    ///
    /// Single-term models run one allocation-free pass over the levels:
    /// per level, one `s × s` solve with `p + 1` right-hand sides, the
    /// level's `(L_ℓ L_ℓᵀ)⁻¹`, and `O(s²(p + 1))` further flops. Several
    /// terms use the block Takahashi recursion for the selected inverse of
    /// `M = ΛᵀAΛ + I` on the factor's block pattern (only the per-level
    /// diagonal blocks are formed for the leading level-structured term)
    /// plus one blocked solve with `p + 1` right-hand sides.
    pub(super) fn profiled_gradient(&self) -> Result<Vec<f64>> {
        let reml = self.reml;
        let k = self.reterms.len();
        if k == 0 {
            return Ok(Vec::new());
        }
        let pwrss_scale = self.pwrss_gradient_scale(reml);
        let (beta, c_inv) = self.beta_and_c_inverse(reml);

        let single_term = k == 1
            && LevelBlocks::is_level_structured(&self.a_blocks[block_index(0, 0)])
            && LevelBlocks::is_level_structured(&self.l_blocks[block_index(0, 0)]);
        let parts = if single_term {
            vec![self.single_term_gradient_parts(reml, &beta, c_inv.as_deref())]
        } else {
            self.multi_term_gradient_parts(reml, &beta, c_inv.as_deref())?
        };

        let n_theta: usize = self.reterms.iter().map(|re| re.inds.len()).sum();
        let mut gradient = Vec::with_capacity(n_theta);
        for (re, term) in self.reterms.iter().zip(&parts) {
            let s = re.vsize;
            for &idx in &re.inds {
                let slot = (idx % s) * s + idx / s;
                gradient.push(term.d_logdet[slot] + pwrss_scale * term.d_pwrss[slot]);
            }
        }
        Ok(gradient)
    }

    /// Multiplier of `∂pwrss/∂θ` in the objective gradient: `denomdf / pwrss`
    /// for the profiled objective, `1 / σ²` when σ is fixed.
    fn pwrss_gradient_scale(&self, reml: bool) -> f64 {
        match self.sigma {
            Some(sigma) => 1.0 / (sigma * sigma),
            None => {
                let k = self.reterms.len();
                let pwrss = with_dense_block(&self.l_blocks[block_index(k, k)], |l_last| {
                    let pp1 = l_last.nrows();
                    let last_diag = l_last[(pp1 - 1, pp1 - 1)];
                    last_diag * last_diag
                });
                let denomdf = if reml {
                    (self.dims.n - self.dims.p) as f64
                } else {
                    self.dims.n as f64
                };
                denomdf / pwrss
            }
        }
    }

    /// β from the trailing factor block (as `beta()`, without cloning the
    /// block) and, for REML, `C⁻¹ = (L_xx L_xxᵀ)⁻¹` row-major.
    fn beta_and_c_inverse(&self, reml: bool) -> (Vec<f64>, Option<Vec<f64>>) {
        let k = self.reterms.len();
        with_dense_block(&self.l_blocks[block_index(k, k)], |l_last| {
            let pp1 = l_last.nrows();
            let p = pp1 - 1;
            let mut beta = vec![0.0; p];
            for j in 0..p {
                beta[j] = l_last[(p, j)];
            }
            for i in (0..p).rev() {
                let mut sum = beta[i];
                for j in (i + 1)..p {
                    sum -= l_last[(j, i)] * beta[j];
                }
                beta[i] = sum / l_last[(i, i)];
            }
            let c_inv = reml.then(|| {
                let mut l_inv = vec![0.0; p * p];
                lower_inverse_flat(|r, c| l_last[(r, c)], p, &mut l_inv);
                let mut c_inv = vec![0.0; p * p];
                for r in 0..p {
                    for c in 0..p {
                        let mut sum = 0.0;
                        for m in r.max(c)..p {
                            sum += l_inv[m * p + r] * l_inv[m * p + c];
                        }
                        c_inv[r * p + c] = sum;
                    }
                }
                c_inv
            });
            (beta, c_inv)
        })
    }

    /// One pass over the levels of a single (block-)diagonal term.
    fn single_term_gradient_parts(
        &self,
        reml: bool,
        beta: &[f64],
        c_inv: Option<&[f64]>,
    ) -> TermGradientParts {
        match self.reterms[0].vsize {
            1 => self.scalar_term_gradient_parts(reml, beta, c_inv),
            2 => self.vsize2_term_gradient_parts(reml, beta, c_inv),
            _ => self.vector_term_gradient_parts(reml, beta, c_inv),
        }
    }

    /// Scalar random-effect term (`s = 1`): every per-level quantity is a
    /// scalar, so the level loop is a handful of flops and one division.
    fn scalar_term_gradient_parts(
        &self,
        reml: bool,
        beta: &[f64],
        c_inv: Option<&[f64]>,
    ) -> TermGradientParts {
        let re = &self.reterms[0];
        let n_levels = re.n_levels();
        let p = beta.len();
        let pp1 = p + 1;
        let lam = re.lambda[(0, 0)];
        let a00 = LevelBlocks::of(&self.a_blocks[block_index(0, 0)]);
        let l00 = LevelBlocks::of(&self.l_blocks[block_index(0, 0)]);
        let a10 = FeCrossBlock::of(&self.a_blocks[block_index(1, 0)]);
        let c_inv = if reml { c_inv } else { None };

        let mut b = vec![0.0; pp1];
        let mut a = [0.0];
        let mut l = [0.0];
        let mut wq = vec![0.0; p];
        let mut d_logdet = 0.0;
        let mut d_pwrss = 0.0;
        for level in 0..n_levels {
            a10.fill(level, 1, pp1, &mut b);
            a00.fill(level, 1, &mut a);
            l00.fill(level, 1, &mut l);
            let (a, l) = (a[0], l[0]);
            let mut e = b[p];
            for (beta_r, b_r) in beta.iter().zip(&b) {
                e -= beta_r * b_r;
            }
            let l_inv = guarded_div(1.0, l);
            let m_inv = l_inv * l_inv;
            let u = lam * e * m_inv;
            let g = e - a * lam * u;
            d_pwrss -= 2.0 * g * u;
            d_logdet += 2.0 * m_inv * lam * a;
            if let Some(c_inv) = c_inv {
                // Q = m_inv λ T, V = a λ Q; contribution 2 (V − T) C⁻¹ Qᵀ.
                let q_scale = m_inv * lam;
                let v_scale = a * lam;
                for c in 0..p {
                    let mut value = 0.0;
                    for m in 0..p {
                        value += b[m] * c_inv[m * p + c];
                    }
                    wq[c] = q_scale * value;
                }
                let mut total = 0.0;
                for c in 0..p {
                    let t = b[c];
                    total += (v_scale * q_scale * t - t) * wq[c];
                }
                d_logdet += 2.0 * total;
            }
        }
        TermGradientParts {
            d_logdet: vec![d_logdet],
            d_pwrss: vec![d_pwrss],
        }
    }

    /// Two-dimensional random-effect term (`s = 2`), unrolled like the
    /// objective's `profiled_objective_one_vsize2_fast`.
    fn vsize2_term_gradient_parts(
        &self,
        reml: bool,
        beta: &[f64],
        c_inv: Option<&[f64]>,
    ) -> TermGradientParts {
        let re = &self.reterms[0];
        let n_levels = re.n_levels();
        let p = beta.len();
        let pp1 = p + 1;
        let (m00, m10, m11) = (re.lambda[(0, 0)], re.lambda[(1, 0)], re.lambda[(1, 1)]);
        let a00 = LevelBlocks::of(&self.a_blocks[block_index(0, 0)]);
        let l00 = LevelBlocks::of(&self.l_blocks[block_index(0, 0)]);
        let a10 = FeCrossBlock::of(&self.a_blocks[block_index(1, 0)]);
        let c_inv = if reml { c_inv } else { None };

        let mut b = vec![0.0; pp1 * 2]; // b[r * 2 + c] = ([X|y]ᵀ Z_ℓ)[r, c]
        let mut a = [0.0; 4];
        let mut l = [0.0; 4];
        let mut q = vec![0.0; 2 * p]; // Q_ℓ row-major
        let mut v = vec![0.0; 2 * p]; // V_ℓ row-major
        let mut wq = vec![0.0; 2 * p]; // Q_ℓ C⁻¹
        let mut d_logdet = [0.0; 4];
        let mut d_pwrss = [0.0; 4];

        for level in 0..n_levels {
            a10.fill(level, 2, pp1, &mut b);
            a00.fill(level, 2, &mut a);
            l00.fill(level, 2, &mut l);
            let (a00v, a01v, a11v) = (a[0], a[1], a[3]);
            let (l00v, l10v, l11v) = (l[0], l[2], l[3]);
            let r0 = guarded_div(1.0, l00v);
            let r1 = guarded_div(1.0, l11v);

            // (L Lᵀ)⁻¹ y.
            let solve = |y0: f64, y1: f64| -> (f64, f64) {
                let t0 = y0 * r0;
                let t1 = (y1 - l10v * t0) * r1;
                let x1 = t1 * r1;
                let x0 = (t0 - l10v * x1) * r0;
                (x0, x1)
            };
            // Λᵀ y, Λ y, A y.
            let lambda_t = |y0: f64, y1: f64| (m00 * y0 + m10 * y1, m11 * y1);
            let lambda = |y0: f64, y1: f64| (m00 * y0, m10 * y0 + m11 * y1);
            let a_mul = |y0: f64, y1: f64| (a00v * y0 + a01v * y1, a01v * y0 + a11v * y1);

            let mut e0 = b[p * 2];
            let mut e1 = b[p * 2 + 1];
            for (r, beta_r) in beta.iter().enumerate() {
                e0 -= beta_r * b[r * 2];
                e1 -= beta_r * b[r * 2 + 1];
            }

            // pwrss: u = M⁻¹ Λᵀ e, g = e − A Λ u.
            let (c0, c1) = lambda_t(e0, e1);
            let (u0, u1) = solve(c0, c1);
            let (lu0, lu1) = lambda(u0, u1);
            let (w0, w1) = a_mul(lu0, lu1);
            let (g0, g1) = (e0 - w0, e1 - w1);
            d_pwrss[0] -= 2.0 * g0 * u0;
            d_pwrss[1] -= 2.0 * g0 * u1;
            d_pwrss[2] -= 2.0 * g1 * u0;
            d_pwrss[3] -= 2.0 * g1 * u1;

            // logdet M: 2 (Z Λᵀ A)[b, a] with Z = (L Lᵀ)⁻¹.
            let z01 = -l10v * r0 * r1 * r1;
            let z00 = r0 * r0 + (l10v * r0 * r1) * (l10v * r0 * r1);
            let z11 = r1 * r1;
            let lta00 = m00 * a00v + m10 * a01v;
            let lta01 = m00 * a01v + m10 * a11v;
            let lta10 = m11 * a01v;
            let lta11 = m11 * a11v;
            let d00 = z00 * lta00 + z01 * lta10;
            let d01 = z00 * lta01 + z01 * lta11;
            let d10 = z01 * lta00 + z11 * lta10;
            let d11 = z01 * lta01 + z11 * lta11;
            d_logdet[0] += 2.0 * d00; // slot (a=0,b=0): d[b][a] = d00
            d_logdet[1] += 2.0 * d10; // slot (a=0,b=1): d[1][0]
            d_logdet[2] += 2.0 * d01; // slot (a=1,b=0): d[0][1]
            d_logdet[3] += 2.0 * d11; // slot (a=1,b=1)

            // REML: 2 ((V − T) C⁻¹ Qᵀ)[a, b], T[·, c] = b[c, ·].
            if let Some(c_inv) = c_inv {
                for c in 0..p {
                    let (t0, t1) = (b[c * 2], b[c * 2 + 1]);
                    let (rc0, rc1) = lambda_t(t0, t1);
                    let (q0, q1) = solve(rc0, rc1);
                    let (lq0, lq1) = lambda(q0, q1);
                    let (v0, v1) = a_mul(lq0, lq1);
                    q[c] = q0;
                    q[p + c] = q1;
                    v[c] = v0 - t0;
                    v[p + c] = v1 - t1;
                }
                for r in 0..2 {
                    for c in 0..p {
                        let mut value = 0.0;
                        for m in 0..p {
                            value += q[r * p + m] * c_inv[m * p + c];
                        }
                        wq[r * p + c] = value;
                    }
                }
                for aa in 0..2 {
                    for bb in 0..2 {
                        let mut total = 0.0;
                        for c in 0..p {
                            total += v[aa * p + c] * wq[bb * p + c];
                        }
                        d_logdet[aa * 2 + bb] += 2.0 * total;
                    }
                }
            }
        }
        TermGradientParts {
            d_logdet: d_logdet.to_vec(),
            d_pwrss: d_pwrss.to_vec(),
        }
    }

    /// Vector random-effect term (`s ≥ 3`) on flat row-major scratch.
    #[allow(
        clippy::needless_range_loop,
        reason = "small dense kernels indexed by row and column into flat scratch buffers"
    )]
    fn vector_term_gradient_parts(
        &self,
        reml: bool,
        beta: &[f64],
        c_inv: Option<&[f64]>,
    ) -> TermGradientParts {
        let re = &self.reterms[0];
        let s = re.vsize;
        let n_levels = re.n_levels();
        let p = beta.len();
        let pp1 = p + 1;
        let cols = if reml { pp1 } else { 1 };
        let uc = cols - 1;
        let lambda = &re.lambda;
        let a00 = LevelBlocks::of(&self.a_blocks[block_index(0, 0)]);
        let l00 = LevelBlocks::of(&self.l_blocks[block_index(0, 0)]);
        let a10 = FeCrossBlock::of(&self.a_blocks[block_index(1, 0)]);
        let c_inv = if reml { c_inv } else { None };

        let ss = s * s;
        // Scratch, all row-major.
        let mut b = vec![0.0; pp1 * s]; // [X|y]ᵀ Z_ℓ
        let mut a = vec![0.0; ss]; // Z_ℓᵀ Z_ℓ
        let mut l = vec![0.0; ss]; // L_ℓ
        let mut rd = vec![0.0; s]; // 1 / diag(L_ℓ)
        let mut e = vec![0.0; s]; // Z_ℓᵀ (y − Xβ)
        let mut x = vec![0.0; s * cols]; // [Q_ℓ | u_ℓ]
        let mut lx = vec![0.0; s * cols]; // Λ [Q_ℓ | u_ℓ]
        let mut w = vec![0.0; s * cols]; // A_ℓ Λ [Q_ℓ | u_ℓ] = [V_ℓ | ·]
        let mut l_inv = vec![0.0; ss];
        let mut z = vec![0.0; ss]; // (L_ℓ L_ℓᵀ)⁻¹
        let mut lta = vec![0.0; ss]; // Λᵀ A_ℓ
        let mut d = vec![0.0; ss]; // Z_ℓ Λᵀ A_ℓ
        let mut wq = vec![0.0; s * p]; // Q_ℓ C⁻¹
        let mut parts = TermGradientParts::new(s);

        for level in 0..n_levels {
            a10.fill(level, s, pp1, &mut b);
            a00.fill(level, s, &mut a);
            l00.fill(level, s, &mut l);
            for r in 0..s {
                rd[r] = guarded_div(1.0, l[r * s + r]);
            }

            for c in 0..s {
                let mut value = b[p * s + c];
                for r in 0..p {
                    value -= beta[r] * b[r * s + c];
                }
                e[c] = value;
            }

            // x ← Λᵀ [T_ℓ | e_ℓ], with T_ℓ[c, r] = b[r, c].
            for j in 0..cols {
                for c in 0..s {
                    let mut value = 0.0;
                    for i in c..s {
                        let y = if j == uc { e[i] } else { b[j * s + i] };
                        value += lambda[(i, c)] * y;
                    }
                    x[c * cols + j] = value;
                }
            }
            // x ← (L_ℓ L_ℓᵀ)⁻¹ x.
            for j in 0..cols {
                for r in 0..s {
                    let mut value = x[r * cols + j];
                    for i in 0..r {
                        value -= l[r * s + i] * x[i * cols + j];
                    }
                    x[r * cols + j] = value * rd[r];
                }
                for r in (0..s).rev() {
                    let mut value = x[r * cols + j];
                    for i in (r + 1)..s {
                        value -= l[i * s + r] * x[i * cols + j];
                    }
                    x[r * cols + j] = value * rd[r];
                }
            }
            // lx = Λ x; w = A_ℓ lx.
            for j in 0..cols {
                for r in 0..s {
                    let mut value = 0.0;
                    for c in 0..=r {
                        value += lambda[(r, c)] * x[c * cols + j];
                    }
                    lx[r * cols + j] = value;
                }
            }
            for r in 0..s {
                for j in 0..cols {
                    let mut value = 0.0;
                    for c in 0..s {
                        value += a[r * s + c] * lx[c * cols + j];
                    }
                    w[r * cols + j] = value;
                }
            }

            // pwrss: −2 g_ℓ[a] u_ℓ[b] with g_ℓ = e_ℓ − A_ℓ Λ u_ℓ.
            for aa in 0..s {
                let g = e[aa] - w[aa * cols + uc];
                for bb in 0..s {
                    parts.d_pwrss[aa * s + bb] -= 2.0 * g * x[bb * cols + uc];
                }
            }

            // logdet M: 2 (Z_ℓ Λᵀ A_ℓ)[b, a].
            for col in 0..s {
                for row in 0..col {
                    l_inv[row * s + col] = 0.0;
                }
                l_inv[col * s + col] = rd[col];
                for row in (col + 1)..s {
                    let mut sum = 0.0;
                    for m in col..row {
                        sum -= l[row * s + m] * l_inv[m * s + col];
                    }
                    l_inv[row * s + col] = sum * rd[row];
                }
            }
            for r in 0..s {
                for c in 0..s {
                    let mut value = 0.0;
                    for m in r.max(c)..s {
                        value += l_inv[m * s + r] * l_inv[m * s + c];
                    }
                    z[r * s + c] = value;
                }
            }
            for r in 0..s {
                for c in 0..s {
                    let mut value = 0.0;
                    for i in r..s {
                        value += lambda[(i, r)] * a[i * s + c];
                    }
                    lta[r * s + c] = value;
                }
            }
            for r in 0..s {
                for c in 0..s {
                    let mut value = 0.0;
                    for m in 0..s {
                        value += z[r * s + m] * lta[m * s + c];
                    }
                    d[r * s + c] = value;
                }
            }
            for aa in 0..s {
                for bb in 0..s {
                    parts.d_logdet[aa * s + bb] += 2.0 * d[bb * s + aa];
                }
            }

            // REML: logdet C, 2 ((V_ℓ − T_ℓ) C⁻¹ Q_ℓᵀ)[a, b] with
            // Q_ℓ = x[:, 0..p], V_ℓ = w[:, 0..p], T_ℓ[a, c] = b[c, a].
            if let Some(c_inv) = c_inv {
                for r in 0..s {
                    for c in 0..p {
                        let mut value = 0.0;
                        for m in 0..p {
                            value += x[r * cols + m] * c_inv[m * p + c];
                        }
                        wq[r * p + c] = value;
                    }
                }
                for aa in 0..s {
                    for bb in 0..s {
                        let mut total = 0.0;
                        for c in 0..p {
                            total += (w[aa * cols + c] - b[c * s + aa]) * wq[bb * p + c];
                        }
                        parts.d_logdet[aa * s + bb] += 2.0 * total;
                    }
                }
            }
        }
        parts
    }

    /// General path: dense selected-inverse blocks off the leading term.
    fn multi_term_gradient_parts(
        &self,
        reml: bool,
        beta: &[f64],
        c_inv: Option<&[f64]>,
    ) -> Result<Vec<TermGradientParts>> {
        let k = self.reterms.len();
        let p = beta.len();
        let cols = if reml { p + 1 } else { 1 };
        let uc = cols - 1;
        let sizes: Vec<usize> = self.reterms.iter().map(|re| re.n_ranef()).collect();

        // Right-hand sides Λ_jᵀ [T_j | e_j] from the fixed-effects cross
        // blocks; T_j = Z_jᵀ X_w (kept for REML), e_j = Z_jᵀ (y − Xβ).
        let mut t_blocks: Vec<DMatrix<f64>> = Vec::with_capacity(k);
        let mut e_vecs: Vec<Vec<f64>> = Vec::with_capacity(k);
        let mut rhs: Vec<DMatrix<f64>> = Vec::with_capacity(k);
        for (j, re) in self.reterms.iter().enumerate() {
            let q = sizes[j];
            let mut t = DMatrix::zeros(q, if reml { p } else { 0 });
            let mut e = vec![0.0; q];
            for_each_nonzero(&self.a_blocks[block_index(k, j)], |r, c, v| {
                if r < p {
                    e[c] -= beta[r] * v;
                    if reml {
                        t[(c, r)] = v;
                    }
                } else {
                    e[c] += v;
                }
            });
            let mut block = DMatrix::zeros(q, cols);
            if reml {
                block.columns_mut(0, p).copy_from(&t);
            }
            for (row, &value) in e.iter().enumerate() {
                block[(row, uc)] = value;
            }
            apply_lambda_transpose_to_rhs(&mut block, re);
            t_blocks.push(t);
            e_vecs.push(e);
            rhs.push(block);
        }

        // X = M⁻¹ Λᵀ [T | e] = [Q | u] by blocked forward and backward solves
        // on the random-effects part of the factor.
        for j in 0..k {
            let (solved, rest) = rhs.split_at_mut(j);
            let current = &mut rest[0];
            for (m, solved_m) in solved.iter().enumerate() {
                sub_block_times_dense(current, &self.l_blocks[block_index(j, m)], solved_m);
            }
            solve_lower_block_columns(current, &self.l_blocks[block_index(j, j)]);
        }
        for j in (0..k).rev() {
            let (head, tail) = rhs.split_at_mut(j + 1);
            let current = &mut head[j];
            for (offset, solved) in tail.iter().enumerate() {
                let i = j + 1 + offset;
                subtract_left_block_transpose_product(
                    current,
                    &self.l_blocks[block_index(i, j)],
                    solved,
                );
            }
            solve_upper_block_columns(current, &self.l_blocks[block_index(j, j)]);
        }
        let x = rhs;

        // W_j = Σ_i S_ji Λ_i X_i, accumulated negated: −W = [−V | A Λ u − ·].
        let lx: Vec<DMatrix<f64>> = x
            .iter()
            .zip(self.reterms.iter())
            .map(|(xj, re)| {
                let mut m = xj.clone();
                apply_lambda_to_rhs(&mut m, re);
                m
            })
            .collect();
        let mut neg_w: Vec<DMatrix<f64>> = sizes.iter().map(|&q| DMatrix::zeros(q, cols)).collect();
        for (j, neg_w_j) in neg_w.iter_mut().enumerate() {
            for (i, lx_i) in lx.iter().enumerate() {
                if i <= j {
                    sub_block_times_dense(neg_w_j, &self.a_blocks[block_index(j, i)], lx_i);
                } else {
                    subtract_left_block_transpose_product(
                        neg_w_j,
                        &self.a_blocks[block_index(i, j)],
                        lx_i,
                    );
                }
            }
        }

        let selected = self.selected_inverse(&sizes)?;
        let d_blocks = self.logdet_m_level_blocks(&selected);

        let c_inv_matrix = c_inv.map(|c| DMatrix::from_row_slice(p, p, c));
        let mut parts = Vec::with_capacity(k);
        for (j, re) in self.reterms.iter().enumerate() {
            let s = re.vsize;
            let n_levels = re.n_levels();
            let mut term = TermGradientParts::new(s);
            let x_j = &x[j];
            let neg_w_j = &neg_w[j];
            let e_j = &e_vecs[j];
            let d_j = &d_blocks[j];
            for level in 0..n_levels {
                let base = level * s;
                for aa in 0..s {
                    let g = e_j[base + aa] + neg_w_j[(base + aa, uc)];
                    for bb in 0..s {
                        term.d_pwrss[aa * s + bb] -= 2.0 * g * x_j[(base + bb, uc)];
                        term.d_logdet[aa * s + bb] += 2.0 * d_j[level * s * s + bb * s + aa];
                    }
                }
            }
            if let Some(c_inv) = &c_inv_matrix {
                let wq = x_j.columns(0, p) * c_inv; // q_j × p
                let t_j = &t_blocks[j];
                for level in 0..n_levels {
                    let base = level * s;
                    for aa in 0..s {
                        for bb in 0..s {
                            let mut total = 0.0;
                            for c in 0..p {
                                total -= (neg_w_j[(base + aa, c)] + t_j[(base + aa, c)])
                                    * wq[(base + bb, c)];
                            }
                            term.d_logdet[aa * s + bb] += 2.0 * total;
                        }
                    }
                }
            }
            parts.push(term);
        }
        Ok(parts)
    }

    /// `M⁻¹` on the factor's block pattern by the block Takahashi
    /// recursion on the lower factor, processed from the last term
    /// backwards: `Z_ij = ([i = j] L_jjᵀ⁻¹ − Σ_{m>j} Z_im L_mj) L_jj⁻¹`.
    /// When `L_00` is level-structured only the per-level diagonal blocks
    /// of `Z_00` are formed (nothing else reads its off-diagonal blocks).
    #[allow(
        clippy::needless_range_loop,
        reason = "the recursion indexes the packed block triangle by term pair"
    )]
    fn selected_inverse(&self, sizes: &[usize]) -> Result<SelectedInverse> {
        let k = self.reterms.len();
        let leading_level_structured = LevelBlocks::is_level_structured(&self.l_blocks[0]);
        let total: usize = (0..k)
            .flat_map(|i| (0..=i).map(move |j| (i, j)))
            .filter(|&(i, j)| !(i == 0 && j == 0 && leading_level_structured))
            .map(|(i, j)| sizes[i] * sizes[j])
            .sum();
        if total > BLOCKED_GRADIENT_MAX_DENSE_ENTRIES {
            return Err(MixedModelError::Optimization(format!(
                "blocked gradient: selected inverse would hold {total} dense entries"
            )));
        }

        let mut dense: Vec<Vec<DMatrix<f64>>> = (0..k)
            .map(|i| (0..=i).map(|_| DMatrix::zeros(0, 0)).collect())
            .collect();
        let mut leading_levels: Option<Vec<f64>> = None;

        for j in (0..k).rev() {
            let l_jj = &self.l_blocks[block_index(j, j)];
            // The leading level-structured block never needs its dense
            // inverse (only its level-diagonal selected-inverse blocks are
            // formed); every other diagonal block does, whatever its storage
            // (a nested design keeps later blocks (block-)diagonal too).
            let l_jj_inv = if j == 0 && leading_level_structured {
                None
            } else {
                Some(dense_lower_inverse(l_jj))
            };
            for i in (j..k).rev() {
                if i == 0 && j == 0 && leading_level_structured {
                    leading_levels = Some(self.leading_selected_inverse_levels(&dense, sizes));
                    continue;
                }
                let mut w = DMatrix::zeros(sizes[i], sizes[j]);
                for m in (j + 1)..k {
                    let l_mj = &self.l_blocks[block_index(m, j)];
                    if i >= m {
                        add_dense_times_block(&mut w, &dense[i][m], l_mj);
                    } else {
                        add_dense_transpose_times_block(&mut w, &dense[m][i], l_mj);
                    }
                }
                let z_ij = if i == j {
                    // Z_jj = (L_jjᵀ⁻¹ − W) L_jj⁻¹, one product.
                    let l_inv = l_jj_inv
                        .as_ref()
                        .expect("dense diagonal block has an inverse");
                    let mut r = l_inv.transpose();
                    r -= &w;
                    r * l_inv
                } else {
                    w.neg_mut();
                    right_divide_by_lower(&mut w, l_jj, l_jj_inv.as_ref());
                    w
                };
                dense[i][j] = z_ij;
            }
        }
        Ok(SelectedInverse {
            dense,
            leading_levels,
        })
    }

    /// Per-level `s × s` diagonal blocks of `Z_00` for a level-structured
    /// leading term: `Z_ℓ = (L_ℓᵀ⁻¹ − W_ℓ) L_ℓ⁻¹` with
    /// `W_ℓ = Σ_{m>0} (Z_m0ᵀ L_m0)_ℓℓ`, row-major, `n_levels × s²`.
    #[allow(
        clippy::needless_range_loop,
        reason = "small dense kernels indexed by row and column into flat scratch buffers"
    )]
    fn leading_selected_inverse_levels(
        &self,
        dense: &[Vec<DMatrix<f64>>],
        sizes: &[usize],
    ) -> Vec<f64> {
        let k = self.reterms.len();
        let re = &self.reterms[0];
        let s = re.vsize;
        let ss = s * s;
        let n_levels = re.n_levels();
        let mut w = vec![0.0; n_levels * ss];
        for m in 1..k {
            debug_assert_eq!(dense[m][0].ncols(), sizes[0]);
            add_diag_level_blocks_of_transpose_product(
                &mut w,
                &dense[m][0],
                &self.l_blocks[block_index(m, 0)],
                s,
            );
        }
        let l00 = LevelBlocks::of(&self.l_blocks[0]);
        let mut l = vec![0.0; ss];
        let mut l_inv = vec![0.0; ss];
        let mut rhs = vec![0.0; ss];
        let mut out = vec![0.0; n_levels * ss];
        for level in 0..n_levels {
            l00.fill(level, s, &mut l);
            lower_inverse_flat(|r, c| l[r * s + c], s, &mut l_inv);
            // rhs = L_ℓᵀ⁻¹ − W_ℓ  (L_ℓᵀ⁻¹ = (L_ℓ⁻¹)ᵀ).
            let w_level = &w[level * ss..(level + 1) * ss];
            for r in 0..s {
                for c in 0..s {
                    rhs[r * s + c] = l_inv[c * s + r] - w_level[r * s + c];
                }
            }
            let z_level = &mut out[level * ss..(level + 1) * ss];
            for r in 0..s {
                for c in 0..s {
                    let mut value = 0.0;
                    for m in 0..s {
                        value += rhs[r * s + m] * l_inv[m * s + c];
                    }
                    z_level[r * s + c] = value;
                }
            }
        }
        out
    }

    /// For each term `j` and level `ℓ`, the `s_j × s_j` block
    /// `(M⁻¹ Λᵀ A_ZZ)[(j,ℓ,·),(j,ℓ,·)] = Σ_i (Z_ji Λ_iᵀ S_ij)_ℓℓ`, row-major
    /// `n_levels_j × s_j²` per term.
    fn logdet_m_level_blocks(&self, selected: &SelectedInverse) -> Vec<Vec<f64>> {
        let k = self.reterms.len();
        let mut result = Vec::with_capacity(k);
        for (j, re_j) in self.reterms.iter().enumerate() {
            let s = re_j.vsize;
            let ss = s * s;
            let n_levels = re_j.n_levels();
            let mut d = vec![0.0; n_levels * ss];

            // Own term: A_jj is (block-)diagonal, so only the level-diagonal
            // blocks of Z_jj enter: D_ℓ += Z_jj,ℓℓ Λ_jᵀ A_jj,ℓ.
            let a_jj = LevelBlocks::of(&self.a_blocks[block_index(j, j)]);
            let owned_levels;
            let z_levels: &[f64] = match (&selected.leading_levels, j) {
                (Some(levels), 0) => levels,
                _ => {
                    owned_levels = diag_level_blocks_flat(&selected.dense[j][j], s);
                    &owned_levels
                }
            };
            add_diag_logdet_blocks(&mut d, z_levels, &a_jj, &re_j.lambda, s);

            // Other terms: D_ℓ += Z_ji[ℓ rows, ·] (Λ_iᵀ S_ij)[·, ℓ cols].
            let mut block = DMatrix::zeros(s, s);
            for (i, re_i) in self.reterms.iter().enumerate() {
                if i == j {
                    continue;
                }
                let mut b = if i > j {
                    self.a_blocks[block_index(i, j)].as_dense()
                } else {
                    self.a_blocks[block_index(j, i)].as_dense().transpose()
                };
                apply_lambda_transpose_to_rhs(&mut b, re_i);
                for level in 0..n_levels {
                    let cols = b.columns(level * s, s);
                    if j > i {
                        selected.dense[j][i]
                            .rows(level * s, s)
                            .mul_to(&cols, &mut block);
                    } else {
                        selected.dense[i][j]
                            .columns(level * s, s)
                            .tr_mul_to(&cols, &mut block);
                    }
                    let dst = &mut d[level * ss..(level + 1) * ss];
                    for r in 0..s {
                        for c in 0..s {
                            dst[r * s + c] += block[(r, c)];
                        }
                    }
                }
            }
            result.push(d);
        }
        result
    }
}

/// Per-term accumulators over the `s × s` slots of Λ, row-major `[a * s + b]`
/// for the slot `Λ[a, b]`.
struct TermGradientParts {
    d_logdet: Vec<f64>,
    d_pwrss: Vec<f64>,
}

impl TermGradientParts {
    fn new(s: usize) -> Self {
        Self {
            d_logdet: vec![0.0; s * s],
            d_pwrss: vec![0.0; s * s],
        }
    }
}

/// `M⁻¹` restricted to what the gradient needs.
struct SelectedInverse {
    /// `dense[i][j]` for `j ≤ i`; `dense[0][0]` is empty when
    /// `leading_levels` is set.
    dense: Vec<Vec<DMatrix<f64>>>,
    /// Per-level diagonal blocks of `Z_00`, row-major `n_levels × s²`.
    leading_levels: Option<Vec<f64>>,
}

/// Read-only per-level view of a (block-)diagonal `A` or `L` block.
enum LevelBlocks<'a> {
    Diagonal(&'a DVector<f64>),
    BlockDiagonal(&'a [DMatrix<f64>]),
    Dense(&'a DMatrix<f64>),
    Owned(DMatrix<f64>),
}

impl LevelBlocks<'_> {
    fn is_level_structured(block: &MatrixBlock) -> bool {
        matches!(
            block,
            MatrixBlock::Diagonal(_) | MatrixBlock::BlockDiagonal(_)
        )
    }

    fn of(block: &MatrixBlock) -> LevelBlocks<'_> {
        match block {
            MatrixBlock::Diagonal(diag) => LevelBlocks::Diagonal(diag),
            MatrixBlock::BlockDiagonal(blocks) => LevelBlocks::BlockDiagonal(blocks),
            MatrixBlock::Dense(dense) => LevelBlocks::Dense(dense),
            MatrixBlock::Sparse(_) => LevelBlocks::Owned(block.as_dense()),
        }
    }

    /// The level's `s × s` block, row-major.
    fn fill(&self, level: usize, s: usize, out: &mut [f64]) {
        match self {
            LevelBlocks::Diagonal(diag) => {
                for r in 0..s {
                    for c in 0..s {
                        out[r * s + c] = if r == c { diag[level * s + r] } else { 0.0 };
                    }
                }
            }
            LevelBlocks::BlockDiagonal(blocks) => {
                let block = &blocks[level];
                for r in 0..s {
                    for c in 0..s {
                        out[r * s + c] = block[(r, c)];
                    }
                }
            }
            LevelBlocks::Dense(dense) => Self::fill_from_dense(dense, level, s, out),
            LevelBlocks::Owned(dense) => Self::fill_from_dense(dense, level, s, out),
        }
    }

    fn fill_from_dense(dense: &DMatrix<f64>, level: usize, s: usize, out: &mut [f64]) {
        let offset = level * s;
        for r in 0..s {
            for c in 0..s {
                out[r * s + c] = dense[(offset + r, offset + c)];
            }
        }
    }
}

/// Read-only view of a `(p + 1) × q` fixed-effects cross block.
enum FeCrossBlock<'a> {
    Dense(&'a DMatrix<f64>),
    Sparse(&'a nalgebra_sparse::CscMatrix<f64>),
    Owned(DMatrix<f64>),
}

impl FeCrossBlock<'_> {
    fn of(block: &MatrixBlock) -> FeCrossBlock<'_> {
        match block {
            MatrixBlock::Dense(dense) => FeCrossBlock::Dense(dense),
            MatrixBlock::Sparse(csc) => FeCrossBlock::Sparse(csc),
            _ => FeCrossBlock::Owned(block.as_dense()),
        }
    }

    /// The level's `pp1 × s` column block, row-major.
    fn fill(&self, level: usize, s: usize, pp1: usize, out: &mut [f64]) {
        match self {
            FeCrossBlock::Dense(dense) => Self::fill_from_dense(dense, level, s, pp1, out),
            FeCrossBlock::Owned(dense) => Self::fill_from_dense(dense, level, s, pp1, out),
            FeCrossBlock::Sparse(csc) => {
                out.iter_mut().for_each(|v| *v = 0.0);
                let offsets = csc.col_offsets();
                let rows = csc.row_indices();
                let values = csc.values();
                for c in 0..s {
                    let col = level * s + c;
                    for idx in offsets[col]..offsets[col + 1] {
                        out[rows[idx] * s + c] = values[idx];
                    }
                }
            }
        }
    }
}

impl FeCrossBlock<'_> {
    fn fill_from_dense(dense: &DMatrix<f64>, level: usize, s: usize, pp1: usize, out: &mut [f64]) {
        for c in 0..s {
            let col = dense.column(level * s + c);
            for r in 0..pp1 {
                out[r * s + c] = col[r];
            }
        }
    }
}

/// Inverse of an `n × n` lower-triangular matrix read through `get`, written
/// row-major (zero pivots invert to zero columns, as the solve helpers do).
fn lower_inverse_flat(get: impl Fn(usize, usize) -> f64, n: usize, out: &mut [f64]) {
    for col in 0..n {
        for row in 0..col {
            out[row * n + col] = 0.0;
        }
        for row in col..n {
            let mut sum = if row == col { 1.0 } else { 0.0 };
            for m in col..row {
                sum -= get(row, m) * out[m * n + col];
            }
            out[row * n + col] = guarded_div(sum, get(row, row));
        }
    }
}

/// Dense inverse of a lower-triangular block, one column at a time by
/// forward substitution along the factor's contiguous columns.
fn dense_lower_inverse(block: &MatrixBlock) -> DMatrix<f64> {
    with_dense_block(block, |l| {
        let n = l.nrows();
        let mut inv = DMatrix::<f64>::zeros(n, n);
        for col in 0..n {
            let mut x = inv.column_mut(col);
            x[col] = 1.0;
            for m in col..n {
                let scale = guarded_div(x[m], l[(m, m)]);
                x[m] = scale;
                if scale != 0.0 && m + 1 < n {
                    let l_col = l.column(m);
                    let mut tail = x.rows_range_mut((m + 1)..);
                    tail.axpy(-scale, &l_col.rows_range((m + 1)..), 1.0);
                }
            }
        }
        inv
    })
}

/// Visit every stored entry of a block as `(row, col, value)`.
fn for_each_nonzero(block: &MatrixBlock, mut f: impl FnMut(usize, usize, f64)) {
    match block {
        MatrixBlock::Dense(dense) => {
            for c in 0..dense.ncols() {
                let col = dense.column(c);
                for r in 0..dense.nrows() {
                    f(r, c, col[r]);
                }
            }
        }
        MatrixBlock::Sparse(csc) => {
            for (r, c, v) in csc.triplet_iter() {
                f(r, c, *v);
            }
        }
        MatrixBlock::Diagonal(diag) => {
            for (i, &v) in diag.iter().enumerate() {
                f(i, i, v);
            }
        }
        MatrixBlock::BlockDiagonal(blocks) => {
            let mut offset = 0;
            for block in blocks {
                let s = block.nrows();
                for c in 0..s {
                    for r in 0..s {
                        f(offset + r, offset + c, block[(r, c)]);
                    }
                }
                offset += s;
            }
        }
    }
}

/// `dst -= lhsᵀ rhs` for a block `lhs` whose row count matches `rhs`.
fn subtract_left_block_transpose_product(
    dst: &mut DMatrix<f64>,
    lhs: &MatrixBlock,
    rhs: &DMatrix<f64>,
) {
    if let MatrixBlock::Dense(mat) = lhs {
        dst.gemm_tr(-1.0, mat, rhs, 1.0);
        return;
    }
    let cols = rhs.ncols();
    for_each_nonzero(lhs, |r, c, v| {
        for col in 0..cols {
            dst[(c, col)] -= v * rhs[(r, col)];
        }
    });
}

/// Sparse blocks at or above this fill fraction are multiplied densely.
const DENSE_PRODUCT_FILL_THRESHOLD: f64 = 0.25;

/// `dst += lhs · rhs` for a block `rhs`.
fn add_dense_times_block(dst: &mut DMatrix<f64>, lhs: &DMatrix<f64>, rhs: &MatrixBlock) {
    match rhs {
        MatrixBlock::Dense(mat) => dst.gemm(1.0, lhs, mat, 1.0),
        MatrixBlock::Sparse(csc) => {
            let entries = (csc.nrows() * csc.ncols()) as f64;
            if entries > 0.0 && csc.nnz() as f64 >= DENSE_PRODUCT_FILL_THRESHOLD * entries {
                dst.gemm(1.0, lhs, &rhs.as_dense(), 1.0);
                return;
            }
            let offsets = csc.col_offsets();
            let rows = csc.row_indices();
            let values = csc.values();
            for c in 0..csc.ncols() {
                let mut column = dst.column_mut(c);
                for idx in offsets[c]..offsets[c + 1] {
                    column.axpy(values[idx], &lhs.column(rows[idx]), 1.0);
                }
            }
        }
        _ => {
            let rows = lhs.nrows();
            for_each_nonzero(rhs, |r, c, v| {
                for row in 0..rows {
                    dst[(row, c)] += v * lhs[(row, r)];
                }
            });
        }
    }
}

/// `dst += lhsᵀ · rhs` for a block `rhs` sharing `lhs`'s row count.
fn add_dense_transpose_times_block(dst: &mut DMatrix<f64>, lhs: &DMatrix<f64>, rhs: &MatrixBlock) {
    match rhs {
        MatrixBlock::Dense(mat) => dst.gemm_tr(1.0, lhs, mat, 1.0),
        MatrixBlock::Sparse(_) => {
            let lhs_t = lhs.transpose();
            add_dense_times_block(dst, &lhs_t, rhs);
        }
        _ => {
            let cols = lhs.ncols();
            for_each_nonzero(rhs, |r, c, v| {
                for row in 0..cols {
                    dst[(row, c)] += v * lhs[(r, row)];
                }
            });
        }
    }
}

/// `w ← w · L⁻¹` for a lower-triangular block `L` (dense blocks use the
/// precomputed inverse).
fn right_divide_by_lower(w: &mut DMatrix<f64>, l: &MatrixBlock, l_inv: Option<&DMatrix<f64>>) {
    match l {
        MatrixBlock::Diagonal(diag) => {
            for (c, &d) in diag.iter().enumerate() {
                let inv = guarded_div(1.0, d);
                w.column_mut(c).scale_mut(inv);
            }
        }
        MatrixBlock::BlockDiagonal(blocks) => {
            let mut offset = 0;
            let mut reciprocals = Vec::new();
            for block in blocks {
                let s = block.nrows();
                reciprocals.clear();
                reciprocals.extend((0..s).map(|c| guarded_div(1.0, block[(c, c)])));
                // Per row: solve L_ℓᵀ xᵀ = w_rowᵀ on the level's columns.
                for row in 0..w.nrows() {
                    for c in (0..s).rev() {
                        let mut value = w[(row, offset + c)];
                        for m in (c + 1)..s {
                            value -= block[(m, c)] * w[(row, offset + m)];
                        }
                        w[(row, offset + c)] = value * reciprocals[c];
                    }
                }
                offset += s;
            }
        }
        MatrixBlock::Dense(_) | MatrixBlock::Sparse(_) => {
            let inv = l_inv.expect("dense diagonal block has an inverse");
            let product = &*w * inv;
            *w = product;
        }
    }
}

/// `out_ℓ += (lhsᵀ rhs)_ℓℓ` for every level: `lhs` and `rhs` share their row
/// count and column count `n_levels · s`; `out` is row-major `n_levels × s²`.
fn add_diag_level_blocks_of_transpose_product(
    out: &mut [f64],
    lhs: &DMatrix<f64>,
    rhs: &MatrixBlock,
    s: usize,
) {
    for_each_nonzero(rhs, |r, c, v| {
        let level = c / s;
        let cc = c % s;
        let base = level * s * s;
        for rr in 0..s {
            out[base + rr * s + cc] += v * lhs[(r, level * s + rr)];
        }
    });
}

/// Own-term contribution `Z_ℓ Λᵀ A_ℓ` per level.
#[allow(
    clippy::needless_range_loop,
    reason = "small dense kernels indexed by row and column into flat scratch buffers"
)]
fn add_diag_logdet_blocks(
    out: &mut [f64],
    z_levels: &[f64],
    a00: &LevelBlocks<'_>,
    lambda: &DMatrix<f64>,
    s: usize,
) {
    let ss = s * s;
    let n_levels = z_levels.len() / ss;
    let mut a = vec![0.0; ss];
    let mut lta = vec![0.0; ss];
    for level in 0..n_levels {
        a00.fill(level, s, &mut a);
        for r in 0..s {
            for c in 0..s {
                let mut value = 0.0;
                for i in r..s {
                    value += lambda[(i, r)] * a[i * s + c];
                }
                lta[r * s + c] = value;
            }
        }
        let z = &z_levels[level * ss..(level + 1) * ss];
        let d = &mut out[level * ss..(level + 1) * ss];
        for r in 0..s {
            for c in 0..s {
                let mut value = 0.0;
                for m in 0..s {
                    value += z[r * s + m] * lta[m * s + c];
                }
                d[r * s + c] += value;
            }
        }
    }
}

/// `rhs ← Λ rhs` per level (lower-triangular Λ).
fn apply_lambda_to_rhs(rhs: &mut DMatrix<f64>, re: &ReMat) {
    let s = re.vsize;
    let cols = rhs.ncols();
    if s == 1 {
        rhs.scale_mut(re.lambda[(0, 0)]);
        return;
    }
    let lambda = &re.lambda;
    let mut scratch = vec![0.0; s];
    for level in 0..re.n_levels() {
        let offset = level * s;
        for col in 0..cols {
            for r in 0..s {
                let mut value = 0.0;
                for c in 0..=r {
                    value += lambda[(r, c)] * rhs[(offset + c, col)];
                }
                scratch[r] = value;
            }
            for r in 0..s {
                rhs[(offset + r, col)] = scratch[r];
            }
        }
    }
}

/// Per-level diagonal `s × s` blocks of a dense square matrix, row-major.
fn diag_level_blocks_flat(m: &DMatrix<f64>, s: usize) -> Vec<f64> {
    let n_levels = m.nrows() / s;
    let mut out = vec![0.0; n_levels * s * s];
    for level in 0..n_levels {
        let base = level * s;
        for r in 0..s {
            for c in 0..s {
                out[level * s * s + r * s + c] = m[(base + r, base + c)];
            }
        }
    }
    out
}

/// `dst -= block · rhs`.
fn sub_block_times_dense(dst: &mut DMatrix<f64>, block: &MatrixBlock, rhs: &DMatrix<f64>) {
    match block {
        MatrixBlock::Dense(mat) => dst.gemm(-1.0, mat, rhs, 1.0),
        MatrixBlock::Sparse(_) => {
            let cols = rhs.ncols();
            for_each_nonzero(block, |r, c, v| {
                for col in 0..cols {
                    dst[(r, col)] -= v * rhs[(c, col)];
                }
            });
        }
        _ => subtract_left_block_product(dst, block, rhs),
    }
}

/// Solve `L X = X` in place for a lower-triangular block.
fn solve_lower_block_columns(x: &mut DMatrix<f64>, l: &MatrixBlock) {
    if let MatrixBlock::Dense(mat) = l {
        if mat.solve_lower_triangular_mut(x) {
            return;
        }
    }
    solve_lower_block_rhs(x, l);
}

/// Solve `Lᵀ X = X` in place for a lower-triangular block.
fn solve_upper_block_columns(x: &mut DMatrix<f64>, l: &MatrixBlock) {
    if let MatrixBlock::Dense(mat) = l {
        if mat.tr_solve_lower_triangular_mut(x) {
            return;
        }
    }
    solve_upper_block_rhs(x, l);
}

/// Solve `Lᵀ X = rhs` column by column for a lower-triangular block `L`.
fn solve_upper_block_rhs(rhs: &mut DMatrix<f64>, l: &MatrixBlock) {
    let mut column = vec![0.0; rhs.nrows()];
    for col in 0..rhs.ncols() {
        column.copy_from_slice(rhs.column(col).as_slice());
        solve_upper_block_from_lower_transpose_against_rhs(l, &mut column);
        rhs.column_mut(col).copy_from_slice(&column);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formula::parse_formula;
    use crate::model::data::DataFrame;

    pub(super) fn sleepstudy_like(n_subjects: usize, n_obs: usize, seed: u64) -> DataFrame {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        use rand_distr::{Distribution, Normal};
        let mut rng = StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let (mut reaction, mut days, mut subj) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..n_subjects {
            let u0 = normal.sample(&mut rng);
            let u1 = normal.sample(&mut rng);
            let b0 = 24.0 * u0;
            let b1 = 1.68 * u0 + 5.23 * u1;
            for d in 0..n_obs {
                let x = d as f64;
                reaction.push(250.0 + 10.0 * x + b0 + b1 * x + 25.0 * normal.sample(&mut rng));
                days.push(x);
                subj.push(format!("S{i:03}"));
            }
        }
        let mut df = DataFrame::new();
        df.add_numeric("reaction", reaction).unwrap();
        let curvature: Vec<f64> = days.iter().map(|d| (d - 4.5) * (d - 4.5) / 10.0).collect();
        df.add_numeric("days", days).unwrap();
        df.add_numeric("curvature", curvature).unwrap();
        df.add_categorical("subj", subj).unwrap();
        df
    }

    pub(super) fn crossed_like(
        n_subjects: usize,
        n_items: usize,
        n_sites: usize,
        n_rep: usize,
    ) -> DataFrame {
        let cm = |value: usize, modulus: usize, center: f64, scale: f64| {
            ((value % modulus) as f64 - center) * scale
        };
        let (mut reaction, mut days, mut subj, mut item, mut site) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for s in 0..n_subjects {
            let sb0 = cm(7 * s + 3, 19, 9.0, 2.4);
            let sb1 = cm(11 * s + 5, 17, 8.0, 0.38) + 0.05 * sb0;
            for i in 0..n_items {
                let ib0 = cm(13 * i + 2, 23, 11.0, 1.6);
                let ib1 = cm(5 * i + 7, 19, 9.0, 0.27) - 0.04 * ib0;
                for r in 0..n_rep {
                    let k = (5 * s + 3 * i + r) % n_sites;
                    let kb0 = cm(3 * k + 1, 13, 6.0, 1.2);
                    let kb1 = cm(7 * k + 4, 11, 5.0, 0.18) + 0.03 * kb0;
                    let eps = cm(13 * s + 7 * i + 3 * r + 2 * k, 29, 14.0, 0.9);
                    let x = r as f64 + (i % 4) as f64 * 0.35 + (s % 3) as f64 * 0.1;
                    reaction.push(
                        250.0 + 9.5 * x + sb0 + sb1 * x + ib0 + ib1 * x + kb0 + kb1 * x + eps,
                    );
                    days.push(x);
                    subj.push(format!("S{s:03}"));
                    item.push(format!("I{i:03}"));
                    site.push(format!("K{k:03}"));
                }
            }
        }
        let mut df = DataFrame::new();
        df.add_numeric("reaction", reaction).unwrap();
        df.add_numeric("days", days).unwrap();
        df.add_categorical("subj", subj).unwrap();
        df.add_categorical("item", item).unwrap();
        df.add_categorical("site", site).unwrap();
        df
    }

    /// Compare the dense reference against the certificate's finite
    /// differences at `theta`, and its objective against the fit path.
    fn check_at(model: &LinearMixedModel, theta: &[f64], reml: bool, tol: f64, label: &str) {
        let mut probe = model.clone();
        probe.optsum.reml = reml;
        let reference = probe
            .dense_reference_gradient(theta, reml)
            .expect("dense reference");
        // The fit path reports the weighted deviance minus the
        // observation-weight constant (`weight_logdet_correction`); the
        // reference carries the weighted deviance itself.
        let fitted_objective = probe.objective_at(theta).expect("objective at theta");
        let fit_path = fitted_objective + probe.weight_logdet_correction();
        assert!(
            (reference.objective - fit_path).abs() <= 1e-8 * fit_path.abs().max(1.0),
            "{label}: reference objective {} vs fit path {fit_path}",
            reference.objective
        );
        let lower = probe.lower_bounds();
        let fd = probe
            .finite_difference_optimizer_derivatives(theta, &lower)
            .expect("finite differences");
        assert_eq!(fd.gradient.len(), reference.gradient.len());
        for (k, (analytic, numeric)) in reference.gradient.iter().zip(&fd.gradient).enumerate() {
            let scale = numeric.abs().max(1.0);
            assert!(
                (analytic - numeric).abs() <= tol * scale,
                "{label}: gradient[{k}] analytic {analytic} vs finite-difference {numeric} (theta {theta:?})"
            );
        }
    }

    pub(super) fn fitted(formula: &str, data: &DataFrame, reml: bool) -> LinearMixedModel {
        let mut model = LinearMixedModel::new(parse_formula(formula).unwrap(), data, None).unwrap();
        model.fit(reml).unwrap();
        model
    }

    #[test]
    fn dense_reference_matches_finite_differences_on_vector_sleepstudy() {
        let data = sleepstudy_like(18, 10, 42);
        for reml in [true, false] {
            let model = fitted("reaction ~ 1 + days + (1 + days | subj)", &data, reml);
            let theta = model.theta();
            check_at(&model, &theta, reml, 1e-5, "vector optimum");
            let away: Vec<f64> = theta.iter().map(|t| t + 0.15).collect();
            check_at(&model, &away, reml, 1e-5, "vector interior");
        }
    }

    #[test]
    fn dense_reference_matches_finite_differences_on_scalar_and_weighted_fits() {
        let data = sleepstudy_like(24, 8, 7);
        let model = fitted("reaction ~ 1 + days + (1 | subj)", &data, true);
        check_at(&model, &model.theta(), true, 1e-5, "scalar optimum");
        check_at(&model, &[0.3], true, 1e-5, "scalar interior");

        let weights: Vec<f64> = (0..data.nrow())
            .map(|i| 0.5 + (i % 3) as f64 * 0.75)
            .collect();
        let mut weighted = LinearMixedModel::new(
            parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap(),
            &data,
            Some(&weights),
        )
        .unwrap();
        weighted.fit(false).unwrap();
        check_at(
            &weighted,
            &weighted.theta(),
            false,
            1e-5,
            "weighted ML optimum",
        );
        let away: Vec<f64> = weighted.theta().iter().map(|t| t + 0.2).collect();
        check_at(&weighted, &away, false, 1e-5, "weighted ML interior");
    }

    #[test]
    fn dense_reference_matches_finite_differences_on_crossed_vector_terms() {
        let data = crossed_like(10, 6, 4, 3);
        let model = fitted(
            "reaction ~ 1 + days + (1 + days | subj) + (1 + days | item) + (1 + days | site)",
            &data,
            true,
        );
        check_at(&model, &model.theta(), true, 1e-5, "crossed optimum");
        let away: Vec<f64> = model
            .theta()
            .iter()
            .enumerate()
            .map(|(k, t)| t + 0.1 + 0.02 * k as f64)
            .collect();
        check_at(&model, &away, true, 1e-5, "crossed interior");
    }

    /// The blocked gradient must reproduce the dense reference to rounding.
    fn check_blocked_matches_dense(model: &LinearMixedModel, theta: &[f64], label: &str) {
        let reml = model.optsum.reml;
        let reference = model
            .dense_reference_gradient(theta, reml)
            .expect("dense reference");
        let mut probe = model.clone();
        let (objective, blocked) = probe
            .objective_and_gradient_at(theta)
            .expect("blocked gradient");
        let fit_path_objective = objective + probe.weight_logdet_correction();
        assert!(
            (reference.objective - fit_path_objective).abs()
                <= 1e-8 * fit_path_objective.abs().max(1.0),
            "{label}: objective {fit_path_objective} vs reference {}",
            reference.objective
        );
        assert_eq!(blocked.len(), reference.gradient.len());
        for (k, (b, r)) in blocked.iter().zip(&reference.gradient).enumerate() {
            let scale = r.abs().max(1.0);
            assert!(
                (b - r).abs() <= 1e-9 * scale,
                "{label}: gradient[{k}] blocked {b} vs dense reference {r} (theta {theta:?})"
            );
        }
    }

    #[test]
    fn blocked_gradient_matches_dense_reference_single_term() {
        let data = sleepstudy_like(18, 10, 42);
        for (formula, reml) in [
            ("reaction ~ 1 + days + (1 + days | subj)", true),
            ("reaction ~ 1 + days + (1 + days | subj)", false),
            ("reaction ~ 1 + days + (1 | subj)", true),
            ("reaction ~ 1 + days + (1 | subj)", false),
            // Three-dimensional term: the generic (s ≥ 3) kernel.
            ("reaction ~ 1 + days + (1 + days + curvature | subj)", true),
            (
                "reaction ~ 1 + days + curvature + (1 + days + curvature | subj)",
                false,
            ),
        ] {
            let model = fitted(formula, &data, reml);
            let theta = model.theta();
            check_blocked_matches_dense(&model, &theta, formula);
            let away: Vec<f64> = theta.iter().map(|t| t + 0.15).collect();
            check_blocked_matches_dense(&model, &away, formula);
            let mut boundary = theta.clone();
            boundary[0] = 0.0;
            check_blocked_matches_dense(&model, &boundary, formula);
        }
    }

    #[test]
    fn blocked_gradient_matches_dense_reference_weighted() {
        let data = sleepstudy_like(24, 8, 7);
        let weights: Vec<f64> = (0..data.nrow())
            .map(|i| 0.5 + (i % 3) as f64 * 0.75)
            .collect();
        let mut weighted = LinearMixedModel::new(
            parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap(),
            &data,
            Some(&weights),
        )
        .unwrap();
        weighted.fit(true).unwrap();
        check_blocked_matches_dense(&weighted, &weighted.theta(), "weighted REML");
        let away: Vec<f64> = weighted.theta().iter().map(|t| t + 0.2).collect();
        check_blocked_matches_dense(&weighted, &away, "weighted REML interior");
    }

    #[test]
    fn blocked_gradient_matches_dense_reference_crossed_and_nested_terms() {
        let data = crossed_like(10, 6, 4, 3);
        for reml in [true, false] {
            let model = fitted(
                "reaction ~ 1 + days + (1 + days | subj) + (1 + days | item) + (1 + days | site)",
                &data,
                reml,
            );
            check_blocked_matches_dense(&model, &model.theta(), "crossed vector");
            let away: Vec<f64> = model
                .theta()
                .iter()
                .enumerate()
                .map(|(k, t)| t + 0.1 + 0.02 * k as f64)
                .collect();
            check_blocked_matches_dense(&model, &away, "crossed vector interior");
            // Two scalar crossed terms (kb07-shaped).
            let scalar = fitted("reaction ~ 1 + days + (1 | subj) + (1 | item)", &data, reml);
            check_blocked_matches_dense(&scalar, &scalar.theta(), "crossed scalar");
            check_blocked_matches_dense(&scalar, &[0.4, 0.0], "crossed scalar boundary");
        }
    }

    #[test]
    fn dense_reference_matches_one_sided_finite_differences_at_the_boundary() {
        let data = sleepstudy_like(18, 10, 42);
        let model = fitted("reaction ~ 1 + days + (1 + days | subj)", &data, true);
        // Put the slope variance on its lower bound; the certificate uses
        // one-sided differences there, so the tolerance is O(h) looser.
        let mut theta = model.theta();
        theta[2] = 0.0;
        check_at(&model, &theta, true, 1e-3, "boundary slope");
        let zero = vec![0.0; theta.len()];
        check_at(&model, &zero, true, 1e-3, "all boundary");
    }
}

#[cfg(test)]
mod oracle_fit {
    //! The native TrustBQ path driven by the analytic gradient must reach
    //! the interpolation path's optimum with fewer evaluations.
    use super::tests::{crossed_like, sleepstudy_like};
    use super::*;
    use crate::formula::parse_formula;
    use crate::model::data::DataFrame;

    fn fit_trust_bq(
        formula: &str,
        data: &DataFrame,
        reml: bool,
        oracle: TrustBqGradientOracle,
    ) -> LinearMixedModel {
        let mut model = LinearMixedModel::new(parse_formula(formula).unwrap(), data, None).unwrap();
        let options = if reml {
            FitOptions::reml()
        } else {
            FitOptions::ml()
        }
        .with_optimizer_control(
            OptimizerControl::auto()
                .with_optimizer(Optimizer::TrustBq)
                .with_trust_bq_gradient_oracle(oracle),
        );
        model.fit_with_options(options).unwrap();
        model
    }

    fn assert_same_optimum(
        oracle: &LinearMixedModel,
        interpolation: &LinearMixedModel,
        label: &str,
    ) {
        let reference = interpolation.optsum.fmin;
        let tolerance = 1e-6 * (1.0 + reference.abs());
        assert!(
            oracle.optsum.fmin <= reference + tolerance,
            "{label}: oracle objective {} vs interpolation {reference} (tolerance {tolerance})",
            oracle.optsum.fmin
        );
        assert!(
            oracle.optsum.return_value.starts_with("GRADIENT_ORACLE:"),
            "{label}: {}",
            oracle.optsum.return_value
        );
        assert!(
            !interpolation
                .optsum
                .return_value
                .contains("GRADIENT_ORACLE"),
            "{label}: {}",
            interpolation.optsum.return_value
        );
        assert_eq!(
            oracle.optsum.convergence_status(),
            crate::types::opt_summary::ConvergenceStatus::Converged,
            "{label}: oracle stop {}",
            oracle.optsum.return_value
        );
    }

    #[test]
    fn gradient_oracle_reaches_the_interpolation_optimum_on_crossed_terms_with_fewer_evaluations() {
        let data = crossed_like(30, 20, 8, 3);
        let formula = "reaction ~ 1 + days + (1 + days | subj) + (1 + days | item) + (1 | site)";
        for reml in [true, false] {
            // d = 7: the family policy already selects the oracle.
            let oracle = fit_trust_bq(formula, &data, reml, TrustBqGradientOracle::FamilyPolicy);
            let interpolation = fit_trust_bq(formula, &data, reml, TrustBqGradientOracle::Disabled);
            assert_same_optimum(&oracle, &interpolation, &format!("crossed reml={reml}"));
            assert!(
                oracle.optsum.feval * 2 < interpolation.optsum.feval,
                "reml={reml}: oracle {} evaluations vs interpolation {}",
                oracle.optsum.feval,
                interpolation.optsum.feval
            );
            for (a, b) in oracle.theta().iter().zip(interpolation.theta()) {
                assert!((a - b).abs() <= 2e-3, "reml={reml}: theta {a} vs {b}");
            }
        }
    }

    #[test]
    fn gradient_oracle_matches_the_interpolation_optimum_on_a_single_vector_term() {
        let data = sleepstudy_like(18, 10, 42);
        let formula = "reaction ~ 1 + days + (1 + days | subj)";
        for reml in [true, false] {
            let oracle = fit_trust_bq(formula, &data, reml, TrustBqGradientOracle::FamilyPolicy);
            let interpolation = fit_trust_bq(formula, &data, reml, TrustBqGradientOracle::Disabled);
            assert_same_optimum(&oracle, &interpolation, &format!("vector reml={reml}"));
            assert!(
                oracle.optsum.feval * 2 < interpolation.optsum.feval,
                "reml={reml}: oracle {} evaluations vs interpolation {}",
                oracle.optsum.feval,
                interpolation.optsum.feval
            );
            for (a, b) in oracle.theta().iter().zip(interpolation.theta()) {
                assert!((a - b).abs() <= 1e-3, "reml={reml}: theta {a} vs {b}");
            }
        }
    }

    #[test]
    fn gradient_oracle_stops_cleanly_at_a_zero_variance_boundary() {
        // No subject effect in the data: the intercept variance optimum is
        // on the θ = 0 boundary, where the gradient points outward.
        let data = {
            use rand::rngs::StdRng;
            use rand::SeedableRng;
            use rand_distr::{Distribution, Normal};
            let mut rng = StdRng::seed_from_u64(11);
            let normal = Normal::new(0.0, 1.0).unwrap();
            let (mut reaction, mut days, mut subj) = (Vec::new(), Vec::new(), Vec::new());
            for i in 0..24 {
                for d in 0..10 {
                    reaction.push(250.0 + 10.0 * d as f64 + 25.0 * normal.sample(&mut rng));
                    days.push(d as f64);
                    subj.push(format!("S{i:03}"));
                }
            }
            let mut df = DataFrame::new();
            df.add_numeric("reaction", reaction).unwrap();
            df.add_numeric("days", days).unwrap();
            df.add_categorical("subj", subj).unwrap();
            df
        };
        let formula = "reaction ~ 1 + days + (1 | subj)";
        let oracle = fit_trust_bq(formula, &data, true, TrustBqGradientOracle::AllFamilies);
        let interpolation = fit_trust_bq(formula, &data, true, TrustBqGradientOracle::Disabled);
        assert_same_optimum(&oracle, &interpolation, "boundary");
        assert!(oracle.theta()[0] <= 1e-3, "theta {:?}", oracle.theta());
    }
}

#[cfg(test)]
mod certificate {
    //! S5.4: the certificate's derivative evidence on the analytic gradient.
    use super::tests::{crossed_like, fitted, sleepstudy_like};
    use super::*;
    use crate::compiler::audit::{CertificateCheck, EvidenceMethod};
    use crate::model::data::DataFrame;

    fn assert_close(label: &str, analytic: f64, numeric: f64, tolerance: f64) {
        let scale = numeric.abs().max(1.0);
        assert!(
            (analytic - numeric).abs() <= tolerance * scale,
            "{label}: analytic {analytic} vs finite-difference {numeric}"
        );
    }

    /// A fit whose θ are all clearly interior: objective differences across
    /// a near-singular Λ (pivots inside the Cholesky zero-pad band) are not
    /// smooth, so the finite-difference Hessian is only meaningful there.
    fn interior_fit(formula: &str, data: &DataFrame, reml: bool) -> Option<LinearMixedModel> {
        let model = fitted(formula, data, reml);
        model.theta().iter().all(|t| *t > 0.05).then_some(model)
    }

    #[test]
    fn analytic_derivative_evidence_matches_the_finite_difference_evidence() {
        let mut checked = 0usize;
        for reml in [true, false] {
            let sleep = (1..12u64).find_map(|seed| {
                interior_fit(
                    "reaction ~ 1 + days + (1 + days | subj)",
                    &sleepstudy_like(18, 10, seed),
                    reml,
                )
            });
            let crossed = interior_fit(
                "reaction ~ 1 + days + (1 + days | subj) + (1 + days | item) + (1 | site)",
                &crossed_like(30, 20, 8, 3),
                reml,
            );
            for model in [sleep, crossed].into_iter().flatten() {
                checked += 1;
                let formula = "fit";
                let theta = model.theta();
                let lower = model.lower_bounds();
                let analytic = model
                    .analytic_optimizer_derivatives(&theta, &lower)
                    .expect("analytic evidence");
                let numeric = model
                    .finite_difference_optimizer_derivatives(&theta, &lower)
                    .expect("finite-difference evidence");
                assert_eq!(analytic.method, EvidenceMethod::Exact);
                assert_eq!(analytic.hessian_method, EvidenceMethod::FiniteDifference);
                let label = format!("{formula} reml={reml}");
                // At an optimum both gradients are tiny and the objective
                // differences are rounding-limited (about 1e-4 absolute on
                // objectives of order 1e4); the formula itself is validated
                // away from the optimum by `tests::check_at`.
                for (k, (a, n)) in analytic.gradient.iter().zip(&numeric.gradient).enumerate() {
                    assert_close(&format!("{label} gradient[{k}]"), *a, *n, 1e-3);
                }
                let (ha, hn) = (analytic.hessian.unwrap(), numeric.hessian.unwrap());
                let scale = hn.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
                for i in 0..ha.nrows() {
                    for j in 0..ha.ncols() {
                        assert!(
                            (ha[(i, j)] - hn[(i, j)]).abs() <= 1e-3 * scale,
                            "{label} hessian[{i},{j}]: analytic-gradient {} vs objective differences {}",
                            ha[(i, j)],
                            hn[(i, j)]
                        );
                    }
                }
            }
        }
        assert!(
            checked >= 2,
            "expected interior fits to compare, got {checked}"
        );
    }

    #[test]
    fn inspected_certificate_reports_the_exact_gradient() {
        // Derivative checks are skipped for boundary fits by design, so take
        // the first seed whose optimum is interior.
        let certificate = (1..12u64)
            .find_map(|seed| {
                let data = sleepstudy_like(18, 10, seed);
                let model = fitted("reaction ~ 1 + days + (1 + days | subj)", &data, true);
                let certificate = model.optimizer_certificate()?.clone();
                (certificate.evidence.parameter_space.n_boundary == 0).then_some(certificate)
            })
            .expect("an interior sleepstudy-like fit");
        assert_eq!(certificate.evidence.gradient.method, EvidenceMethod::Exact);
        assert_eq!(
            certificate.evidence.hessian.method,
            EvidenceMethod::FiniteDifference
        );
        assert!(
            certificate
                .checks
                .iter()
                .any(|check| matches!(check, CertificateCheck::FreeGradientOk { .. })),
            "{:?}",
            certificate.checks
        );
        assert!(
            certificate
                .checks
                .iter()
                .any(|check| matches!(check, CertificateCheck::HessianPsdOnActiveSubspace { .. })),
            "{:?}",
            certificate.checks
        );
    }
}

#[cfg(test)]
mod cost {
    //! Gradient cost relative to one objective evaluation. Ignored by
    //! default; run with
    //! `cargo test --release --lib gradient::cost -- --ignored --nocapture`.
    use super::tests::{crossed_like, fitted, sleepstudy_like};
    use super::{
        block_index, FitOptions, LinearMixedModel, MatrixBlock, Optimizer, OptimizerControl,
        ProfiledGradientInputs, TrustBqGradientOracle,
    };
    use crate::formula::parse_formula;
    use std::time::Instant;

    /// Trajectory of the maximal singular fixture under each optimizer path.
    #[test]
    #[ignore]
    fn singular_maximal_trajectories() {
        let (data, _) = crate::datasets::load("singular").unwrap();
        let formula = "y ~ 1 + A * B * C + (A * B * C | group)";
        let variants: Vec<(&str, OptimizerControl)> = vec![
            ("auto", OptimizerControl::auto()),
            (
                "trust_bq oracle",
                OptimizerControl::auto().with_optimizer(Optimizer::TrustBq),
            ),
            (
                "trust_bq interpolation",
                OptimizerControl::auto()
                    .with_optimizer(Optimizer::TrustBq)
                    .with_trust_bq_gradient_oracle(TrustBqGradientOracle::Disabled),
            ),
            #[cfg(feature = "nlopt")]
            (
                "nlopt newuoa",
                OptimizerControl::auto().with_optimizer(Optimizer::NloptNewuoa),
            ),
        ];
        for (label, control) in variants {
            let mut model =
                LinearMixedModel::new(parse_formula(formula).unwrap(), &data, None).unwrap();
            let t0 = Instant::now();
            model
                .fit_with_options(FitOptions::reml().with_optimizer_control(control))
                .unwrap();
            let theta = model.theta();
            let at_bound = theta.iter().filter(|t| t.abs() <= 1e-10).count();
            println!(
                "{label:24} objective {:.6} feval {} ms {:.1} status {} theta_at_zero {}/{}",
                model.objective_value(),
                model.optsum.feval,
                t0.elapsed().as_secs_f64() * 1e3,
                model.optsum.return_value,
                at_bound,
                theta.len()
            );
            let log = &model.optsum.fit_log;
            let mut best = f64::INFINITY;
            for (i, entry) in log.iter().enumerate() {
                best = best.min(entry.objective);
                if i % 25 == 0 || i + 1 == log.len() {
                    println!("    eval {i:4} best {best:.6}");
                }
            }
        }
    }

    /// Per-iteration dynamics of the gradient loop on the singular fixture
    /// (`MIXEFF_TRACE_CASE=crossed_ml` switches to the crossed ML case).
    #[test]
    #[ignore]
    fn singular_maximal_gradient_loop_trace() {
        use crate::optimizer::trust_bq::{minimize_with_gradient_and_progress, TrustBqOptions};
        let crossed_case = std::env::var("MIXEFF_TRACE_CASE").as_deref() == Ok("crossed_ml");
        let (data, formula, reml) = if crossed_case {
            (
                crossed_like(30, 20, 8, 3),
                "reaction ~ 1 + days + (1 + days | subj) + (1 + days | item) + (1 | site)",
                false,
            )
        } else {
            (
                crate::datasets::load("singular").unwrap().0,
                "y ~ 1 + A * B * C + (A * B * C | group)",
                true,
            )
        };
        let mut model =
            LinearMixedModel::new(parse_formula(formula).unwrap(), &data, None).unwrap();
        model.optsum.reml = reml;
        let initial = model.optsum.initial.clone();
        let lower = model.lower_bounds();
        let upper = vec![f64::INFINITY; initial.len()];
        let mut calls = 0usize;
        let mut oracle = |theta: &[f64]| -> crate::error::Result<(f64, Vec<f64>)> {
            calls += 1;
            match model.objective_and_gradient_at(theta) {
                Ok(pair) => Ok(pair),
                Err(_) => Ok((1e9, vec![f64::NAN; theta.len()])),
            }
        };
        let mut last_fevals = 0usize;
        let mut iteration = 0usize;
        let result = minimize_with_gradient_and_progress(
            &initial,
            &lower,
            &upper,
            TrustBqOptions {
                initial_radius: 0.75,
                final_radius: 1e-6,
                max_evaluations: 475,
                ftol_abs: 1e-8,
                ftol_rel: 1e-10,
                ftol_requires_local_radius: true,
                stall_iterations: 3,
                stall_ftol_rel: 1e-6,
                stall_ftol_abs: 1e-8,
                stall_requires_stable_x: false,
                gradient_tolerance_rel: 1e-9,
                ..TrustBqOptions::default()
            },
            &mut oracle,
            |progress| {
                iteration += 1;
                let jump = progress.fevals - last_fevals;
                last_fevals = progress.fevals;
                if iteration <= 60 || iteration % 10 == 0 || jump > 2 {
                    println!(
                        "iter {iteration:4} fevals {:4} (+{jump:2}) fmin {:.6} radius {:.3e}",
                        progress.fevals, progress.fmin, progress.radius
                    );
                }
                Ok(false)
            },
        )
        .unwrap();
        println!("{result:?}");
    }

    #[test]
    #[ignore]
    fn certificate_evidence_cost() {
        let cases = [
            (
                "vector_10000",
                "reaction ~ 1 + days + (1 + days | subj)",
                sleepstudy_like(1000, 10, 42),
            ),
            (
                "crossed d=7",
                "reaction ~ 1 + days + (1 + days | subj) + (1 + days | item) + (1 | site)",
                crossed_like(60, 40, 12, 3),
            ),
        ];
        for (label, formula, data) in cases {
            let model = fitted(formula, &data, true);
            let theta = model.theta();
            let lower = model.lower_bounds();
            let t0 = Instant::now();
            let analytic = model
                .analytic_optimizer_derivatives(&theta, &lower)
                .unwrap();
            let analytic_ms = t0.elapsed().as_secs_f64() * 1e3;
            let t1 = Instant::now();
            let numeric = model
                .finite_difference_optimizer_derivatives(&theta, &lower)
                .unwrap();
            let numeric_ms = t1.elapsed().as_secs_f64() * 1e3;
            println!(
                "{label:14} d={} analytic {analytic_ms:8.3} ms  finite-difference {numeric_ms:8.3} ms  ratio {:.1}x  |grad| {:.2e} vs {:.2e}",
                theta.len(),
                numeric_ms / analytic_ms,
                analytic.gradient.iter().fold(0.0_f64, |m, v| m.max(v.abs())),
                numeric.gradient.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
            );
        }
    }

    #[test]
    #[ignore]
    fn crossed_gradient_phase_costs() {
        let crossed = crossed_like(60, 40, 12, 3);
        let mut model = fitted(
            "reaction ~ 1 + days + (1 + days | subj) + (1 + days | item) + (1 | site)",
            &crossed,
            true,
        );
        let theta = model.theta();
        model.objective_at(&theta).unwrap();
        let sizes: Vec<usize> = model.reterms.iter().map(|re| re.n_ranef()).collect();
        let inputs = ProfiledGradientInputs {
            a_blocks: &model.a_blocks,
            l_blocks: &model.l_blocks,
            reterms: &model.reterms,
            dims: model.dims,
            reml: true,
            sigma: None,
        };
        let reps = 20;
        let time = |label: &str, mut f: Box<dyn FnMut() + '_>| {
            for _ in 0..3 {
                f();
            }
            let t0 = Instant::now();
            for _ in 0..reps {
                f();
            }
            println!(
                "{label:32} {:8.1} us",
                t0.elapsed().as_secs_f64() * 1e6 / reps as f64
            );
        };
        let g = &inputs;
        time(
            "beta_and_c_inverse",
            Box::new(move || {
                std::hint::black_box(g.beta_and_c_inverse(true));
            }),
        );
        let s = sizes.clone();
        time(
            "selected_inverse",
            Box::new(move || {
                std::hint::black_box(g.selected_inverse(&s).unwrap());
            }),
        );
        let selected = inputs.selected_inverse(&sizes).unwrap();
        let sel = &selected;
        time(
            "logdet_m_level_blocks",
            Box::new(move || {
                std::hint::black_box(g.logdet_m_level_blocks(sel));
            }),
        );
        let (beta, c_inv) = inputs.beta_and_c_inverse(true);
        let (b, c) = (&beta, &c_inv);
        time(
            "multi_term_gradient_parts",
            Box::new(move || {
                std::hint::black_box(g.multi_term_gradient_parts(true, b, c.as_deref()).unwrap());
            }),
        );
        let m = &model;
        time(
            "dense_lower_inverse(L_11)",
            Box::new(move || {
                std::hint::black_box(super::dense_lower_inverse(&m.l_blocks[block_index(1, 1)]));
            }),
        );
        for (i, j) in [(1usize, 0usize), (2, 0), (2, 1), (1, 1), (2, 2)] {
            let b = &model.l_blocks[block_index(i, j)];
            let kind = match b {
                MatrixBlock::Dense(_) => "Dense",
                MatrixBlock::Sparse(c) => {
                    println!("  L({i},{j}) nnz {} of {}", c.nnz(), c.nrows() * c.ncols());
                    "Sparse"
                }
                MatrixBlock::Diagonal(_) => "Diagonal",
                MatrixBlock::BlockDiagonal(_) => "BlockDiagonal",
            };
            println!("  L({i},{j}) {kind} {}x{}", b.nrows(), b.ncols());
        }
    }

    #[test]
    #[ignore]
    fn gradient_cost_in_evaluation_equivalents() {
        let sleep = sleepstudy_like(1000, 10, 42);
        let crossed = crossed_like(60, 40, 12, 3);
        let cases: [(&str, &crate::model::data::DataFrame, bool); 5] = [
            ("scalar_10000 ML", &sleep, false),
            ("scalar_10000 REML", &sleep, true),
            ("vector_10000 ML", &sleep, false),
            ("vector_10000 REML", &sleep, true),
            ("crossed REML", &crossed, true),
        ];
        let formulas = [
            "reaction ~ 1 + days + (1 | subj)",
            "reaction ~ 1 + days + (1 | subj)",
            "reaction ~ 1 + days + (1 + days | subj)",
            "reaction ~ 1 + days + (1 + days | subj)",
            "reaction ~ 1 + days + (1 + days | subj) + (1 + days | item) + (1 | site)",
        ];
        println!(
            "{:24} {:>3} {:>8} {:>8} {:>8} {:>6} {:>8}",
            "case", "d", "fast_us", "eval_us", "grad_us", "ratio", "vs_fast"
        );
        for ((label, data, reml), formula) in cases.iter().zip(formulas) {
            let mut model = fitted(formula, data, *reml);
            let theta: Vec<f64> = model.theta();
            let reps = 50;
            // Warm up.
            for _ in 0..3 {
                model.objective_at(&theta).unwrap();
                model.objective_and_gradient_at(&theta).unwrap();
            }
            let t0 = Instant::now();
            for _ in 0..reps {
                std::hint::black_box(model.objective_at(&theta).unwrap());
            }
            let eval = t0.elapsed().as_secs_f64() * 1e6 / reps as f64;
            let tf = Instant::now();
            for _ in 0..reps {
                std::hint::black_box(model.objective_at_fast_or_generic(&theta).unwrap());
            }
            let fast = tf.elapsed().as_secs_f64() * 1e6 / reps as f64;
            let t1 = Instant::now();
            for _ in 0..reps {
                std::hint::black_box(model.objective_and_gradient_at(&theta).unwrap());
            }
            let both = t1.elapsed().as_secs_f64() * 1e6 / reps as f64;
            let grad = both - eval;
            println!(
                "{:24} {:>3} {:>8.1} {:>8.1} {:>8.1} {:>6.2} {:>8.2}",
                label,
                theta.len(),
                fast,
                eval,
                grad,
                grad / eval,
                both / fast
            );
        }
    }
}
