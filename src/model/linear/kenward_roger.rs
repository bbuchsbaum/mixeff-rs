//! Kenward-Roger adjusted fixed-effect covariance without `n × n` matrices.
//!
//! `pbkrtest::vcovAdj_internal()` (and this crate's original port of it)
//! builds the marginal response covariance `Σ = σ²(I + ZΛΛ'Z')`, one dense
//! `n × n` matrix per covariance component `G_a`, and `Σ⁻¹` by dense
//! inversion: O(n³) time and O(n² · #components) memory.
//!
//! Every quantity the adjustment needs is a contraction of `G_a Σ⁻¹`
//! against `X` or against another `G_b Σ⁻¹`. Each random-effect component
//! is a short sum of outer products of per-coefficient designs,
//! `G_a = Σ_{(U,V)} Z_U Z_V'` (`Z_U` is the `n × L_t` design of one
//! random-effect coefficient `U` of term `t`, one column per level), so
//! with the `q × q` matrix `C = Z'Σ⁻¹Z` and the `q × p` matrix
//! `M = Z'Σ⁻¹X` (`M_U`, `C[U,U']` their coefficient sub-blocks):
//!
//! * `P_a = -X'Σ⁻¹G_aΣ⁻¹X = -Σ_{(U,V)} M_U' M_V`
//! * `Q_ab = X'Σ⁻¹G_aΣ⁻¹G_bΣ⁻¹X = Σ_{(U,V)} Σ_{(U',V')} M_V' C[U,U'] M_V'`
//! * `tr(G_aΣ⁻¹G_bΣ⁻¹) = Σ_{(U,V)} Σ_{(U',V')} tr(C[V,U'] C[V',U])`
//!
//! The residual component (`G = I`) needs the same contractions one power
//! of `Σ⁻¹` higher: `C₂ = Z'Σ⁻²Z`, `M₂ = Z'Σ⁻²X`, `X'Σ⁻²X`, `X'Σ⁻³X` and
//! `tr(Σ⁻²)`. With `S = Z'Z`, the Cholesky factor `L L' = Λ'SΛ + I`,
//! `N = (L L')⁻¹`, `Y = L⁻¹Λ'S`, `T = L⁻¹Λ'Z'X`, `V = L⁻ᵀY` and
//! `T₂ = L⁻ᵀT`, the Woodbury identity gives
//!
//! * `C = σ⁻²(S − Y'Y)`, `M = σ⁻²(Z'X − Y'T)`
//! * `C₂ = σ⁻²C − σ⁻⁴V'V`, `M₂ = σ⁻²M − σ⁻⁴V'T₂`
//! * `Σ⁻¹X = σ⁻²(X − ZΛT₂)` (formed observation-wise, `n × p`)
//! * `tr(Σ⁻²) = σ⁻⁴(n − q + ‖N‖²_F)`
//!
//! Memory is O(q² + n·p); nothing of size `n × n` is formed. The response
//! covariance is positive definite by construction (`λ_min(Σ) ≥ σ²`); its
//! extreme eigenvalues come from the `q × q` problem when the cheap bound
//! cannot decide the historical tolerance test.

use super::*;

/// One random-effect coefficient's columns in the stacked `Z`
/// (`offset + level * vsize + coefficient` for every level).
#[derive(Debug, Clone, Copy)]
struct CoefficientColumns {
    offset: usize,
    vsize: usize,
    coefficient: usize,
    n_levels: usize,
}

impl CoefficientColumns {
    #[inline]
    fn index(&self, level: usize) -> usize {
        self.offset + level * self.vsize + self.coefficient
    }
}

/// `G_a = Σ Z_U Z_V'` as its `(U, V)` outer-product terms.
type ComponentTerms = Vec<(CoefficientColumns, CoefficientColumns)>;

/// The pieces of the Kenward-Roger adjustment that depend on the
/// response-covariance components; everything downstream is `p × p`.
#[derive(Debug, Clone)]
pub(super) struct KenwardRogerIngredients {
    /// `P_a`, one per component (random-effect entries, then residual).
    pub(super) p_matrices: Vec<DMatrix<f64>>,
    /// Upper-triangle packed `Q_ab`, `a <= b` (`symmetric_pair_index`).
    pub(super) q_matrices: Vec<DMatrix<f64>>,
    /// `tr(G_aΣ⁻¹G_bΣ⁻¹)`, symmetric `n_components × n_components`.
    pub(super) ktrace: DMatrix<f64>,
    pub(super) component_labels: Vec<String>,
}

impl LinearMixedModel {
    /// Gates shared by the Kenward-Roger entry points (same refusals, in
    /// the same order, as `kenward_roger_sigma_g`).
    fn kenward_roger_prerequisites(&self) -> Result<f64> {
        if self.optsum.feval <= 0 {
            return Err(MixedModelError::NotFitted);
        }
        if !self.sqrtwts.is_empty() {
            return Err(MixedModelError::InvalidArgument(
                "Kenward-Roger Sigma/G decomposition is currently certified only for unweighted iid Gaussian residual models"
                    .to_string(),
            ));
        }
        let sigma = self.sigma();
        if !sigma.is_finite() || sigma <= 0.0 {
            return Err(MixedModelError::InvalidArgument(
                "Kenward-Roger Sigma/G requires a finite positive residual sigma".to_string(),
            ));
        }
        Ok(sigma)
    }

    /// Low-rank (Woodbury) Kenward-Roger ingredients for the active fixed
    /// design `x`; see the module docs.
    pub(super) fn kenward_roger_ingredients(
        &self,
        x: &DMatrix<f64>,
    ) -> Result<KenwardRogerIngredients> {
        let sigma = self.kenward_roger_prerequisites()?;
        let sigma_sq = sigma * sigma;
        let inv_sigma_sq = 1.0 / sigma_sq;
        let n = self.dims.n;
        let p = x.ncols();

        let mut offsets = Vec::with_capacity(self.reterms.len());
        let mut q = 0usize;
        for re in &self.reterms {
            if re.n_obs() != n {
                return Err(MixedModelError::DimensionMismatch(format!(
                    "KR random-effect term '{}' has {} observations, expected {n}",
                    re.grouping_name,
                    re.n_obs()
                )));
            }
            offsets.push(q);
            q += re.n_ranef();
        }
        // About six live `q × q` matrices at the peak.
        let dense_bytes = dense_block_bytes(q, q).saturating_mul(6);
        let limit = dense_block_limit_bytes();
        if dense_bytes > limit {
            return Err(MixedModelError::ProblemTooLarge(format!(
                "Kenward-Roger adjustment would materialize dense {q} x {q} f64 matrices ({:.2} GiB), above the configured limit ({:.2} GiB)",
                dense_bytes as f64 / 1024.0_f64.powi(3),
                limit as f64 / 1024.0_f64.powi(3)
            )));
        }

        // Components and labels in the `kenward_roger_sigma_g` order.
        let mut components: Vec<ComponentTerms> = Vec::new();
        let mut component_labels = Vec::new();
        for (term_index, re) in self.reterms.iter().enumerate() {
            let covariance = sigma_sq * (&re.lambda * re.lambda.transpose());
            let columns = |coefficient: usize| CoefficientColumns {
                offset: offsets[term_index],
                vsize: re.vsize,
                coefficient,
                n_levels: re.n_levels(),
            };
            for (row, col) in kenward_roger_covariance_component_indices(re) {
                if row >= re.vsize || col >= re.vsize {
                    return Err(MixedModelError::DimensionMismatch(format!(
                        "KR covariance component ({row}, {col}) is outside random-effect vector size {}",
                        re.vsize
                    )));
                }
                if !covariance[(row, col)].is_finite() {
                    return Err(MixedModelError::InvalidArgument(
                        "Kenward-Roger Sigma/G component weight is non-finite".to_string(),
                    ));
                }
                components.push(if row == col {
                    vec![(columns(row), columns(row))]
                } else {
                    vec![(columns(row), columns(col)), (columns(col), columns(row))]
                });
                component_labels.push(format!(
                    "{}:{}[{},{}]",
                    term_index, re.grouping_name, re.cnames[row], re.cnames[col]
                ));
            }
        }
        component_labels.push("residual".to_string());
        let n_re_components = components.len();
        let n_components = n_re_components + 1;

        // S = Z'Z and Z'X from the grouping references.
        let mut s = DMatrix::<f64>::zeros(q, q);
        let mut ztx = DMatrix::<f64>::zeros(q, p);
        let mut row_entries: Vec<(usize, f64)> = Vec::new();
        for obs in 0..n {
            row_entries.clear();
            for (re, &offset) in self.reterms.iter().zip(&offsets) {
                let base = offset + re.refs[obs] as usize * re.vsize;
                for c in 0..re.vsize {
                    let value = re.z[(c, obs)];
                    if value != 0.0 {
                        row_entries.push((base + c, value));
                    }
                }
            }
            for &(j, zj) in &row_entries {
                for &(i, zi) in &row_entries {
                    s[(i, j)] += zi * zj;
                }
                for col in 0..p {
                    ztx[(j, col)] += zj * x[(obs, col)];
                }
            }
        }

        // F = Λ'S, LL' = Λ'SΛ + I = Λ'F' + I.
        let mut f = s.clone();
        self.kr_apply_lambda_transpose_left(&offsets, &mut f);
        let mut factor = f.transpose();
        self.kr_apply_lambda_transpose_left(&offsets, &mut factor);
        for i in 0..q {
            factor[(i, i)] += 1.0;
        }
        if !matrix_is_finite(&factor) {
            return Err(MixedModelError::InvalidArgument(
                "Kenward-Roger Sigma/G component weight is non-finite".to_string(),
            ));
        }
        let factor = symmetrize_matrix(&factor);
        let mut l = factor.clone();
        // Λ'SΛ + I ⪰ I: every pivot is at least one.
        super::dense_kernels::dense_cholesky_lower_in_place(&mut l, 0.0)?;

        // Historical PD gate on the n × n response covariance:
        // λ_min(Σ) > max(1e-10 · max(|λ_max(Σ)|, 1), 1e-12). Here
        // λ_min(Σ) ≥ σ² and λ_max(Σ) = σ² λ_max(LL') ≤ σ² tr(LL'), so the
        // bound decides it unless the margin is tiny.
        let trace_bound: f64 = (0..q).map(|i| factor[(i, i)]).sum::<f64>().max(1.0);
        let bound_tolerance = (1e-10 * (sigma_sq * trace_bound).max(1.0)).max(1e-12);
        if !(sigma_sq > bound_tolerance) {
            Self::kr_exact_positive_definite_gate(&factor, sigma_sq, n)?;
        }
        drop(factor);

        // Y' = (SΛ) L⁻ᵀ = F' L⁻ᵀ; T' = (Λ'Z'X)' L⁻ᵀ.
        let mut y_t = f.transpose();
        drop(f);
        super::dense_kernels::dense_rdiv_lower_transpose_in_place(&mut y_t, &l);
        let mut lt_ztx = ztx.clone();
        self.kr_apply_lambda_transpose_left(&offsets, &mut lt_ztx);
        let mut t_t = lt_ztx.transpose();
        super::dense_kernels::dense_rdiv_lower_transpose_in_place(&mut t_t, &l);
        let t = t_t.transpose();

        // C = σ⁻²(S − Y'Y), M = σ⁻²(Z'X − Y'T).
        let mut c = s;
        c.gemm(-1.0, &y_t, &y_t.transpose(), 1.0);
        c *= inv_sigma_sq;
        let c = symmetrize_matrix(&c);
        let mut m = ztx;
        m.gemm(-1.0, &y_t, &t, 1.0);
        m *= inv_sigma_sq;

        // U = L⁻ᵀ (upper), N = U U', V = U Y, T₂ = U T, Z₃ = U'T₂.
        let mut u = DMatrix::<f64>::identity(q, q);
        super::dense_kernels::dense_rdiv_lower_transpose_in_place(&mut u, &l);
        drop(l);
        let n_frobenius_sq = (&u * u.transpose()).norm_squared();
        let v = &u * y_t.transpose();
        drop(y_t);
        let t2 = &u * &t;
        let z3 = u.transpose() * &t2;
        drop(u);

        // C₂ = σ⁻²C − σ⁻⁴V'V, M₂ = σ⁻²M − σ⁻⁴V'T₂.
        let inv_sigma_4 = inv_sigma_sq * inv_sigma_sq;
        let v_t = v.transpose();
        let mut c2 = &c * inv_sigma_sq;
        c2.gemm(-inv_sigma_4, &v_t, &v, 1.0);
        drop(v);
        let mut m2 = &m * inv_sigma_sq;
        m2.gemm(-inv_sigma_4, &v_t, &t2, 1.0);
        drop(v_t);

        // W = Σ⁻¹X = σ⁻²(X − ZΛT₂), observation-wise.
        let mut lambda_t2 = t2;
        self.kr_apply_lambda_left(&offsets, &mut lambda_t2);
        let mut w = x.clone();
        for obs in 0..n {
            for (re, &offset) in self.reterms.iter().zip(&offsets) {
                let base = offset + re.refs[obs] as usize * re.vsize;
                for coefficient in 0..re.vsize {
                    let z = re.z[(coefficient, obs)];
                    if z != 0.0 {
                        for col in 0..p {
                            w[(obs, col)] -= z * lambda_t2[(base + coefficient, col)];
                        }
                    }
                }
            }
        }
        w *= inv_sigma_sq;
        // X'Σ⁻²X = W'W; X'Σ⁻³X = W'Σ⁻¹W = σ⁻²(W'W − (Λ'M)'N(Λ'M)) with
        // Λ'M = σ⁻²NΛ'Z'X = σ⁻²T₂, so (Λ'M)'N(Λ'M) = σ⁻⁴ Z₃'Z₃.
        let wtw = w.transpose() * &w;
        let xt_sigma3_x = (&wtw - (z3.transpose() * &z3) * inv_sigma_4) * inv_sigma_sq;
        let trace_sigma_inv_sq = inv_sigma_4 * ((n as f64) - (q as f64) + n_frobenius_sq);

        let gather_rows = |mat: &DMatrix<f64>, cols: &CoefficientColumns| -> DMatrix<f64> {
            DMatrix::from_fn(cols.n_levels, mat.ncols(), |level, col| {
                mat[(cols.index(level), col)]
            })
        };
        let gather_block =
            |mat: &DMatrix<f64>, rows: &CoefficientColumns, cols: &CoefficientColumns| {
                DMatrix::from_fn(rows.n_levels, cols.n_levels, |a, b| {
                    mat[(rows.index(a), cols.index(b))]
                })
            };
        // tr(C[A,B] C[D,E]) with `C[A,B]` (|A| × |B|) and `C[D,E]` (|D| × |E|),
        // |B| = |D|, |A| = |E|.
        let trace_of_product = |a: &CoefficientColumns,
                                b: &CoefficientColumns,
                                d: &CoefficientColumns,
                                e: &CoefficientColumns| {
            let mut total = 0.0;
            for i in 0..a.n_levels {
                for j in 0..b.n_levels {
                    total += c[(a.index(i), b.index(j))] * c[(d.index(j), e.index(i))];
                }
            }
            total
        };

        let m_rows: Vec<Vec<(DMatrix<f64>, DMatrix<f64>)>> = components
            .iter()
            .map(|terms| {
                terms
                    .iter()
                    .map(|(u_cols, v_cols)| (gather_rows(&m, u_cols), gather_rows(&m, v_cols)))
                    .collect()
            })
            .collect();

        let mut p_matrices = Vec::with_capacity(n_components);
        for terms in &m_rows {
            let mut p_a = DMatrix::<f64>::zeros(p, p);
            for (m_u, m_v) in terms {
                p_a.gemm_tr(-1.0, m_u, m_v, 1.0);
            }
            p_matrices.push(symmetrize_matrix(&p_a));
        }
        p_matrices.push(symmetrize_matrix(&(-&wtw)));

        let mut q_matrices = Vec::with_capacity(n_components * (n_components + 1) / 2);
        let mut ktrace = DMatrix::<f64>::zeros(n_components, n_components);
        for rr in 0..n_components {
            for ss in rr..n_components {
                let (q_matrix, trace) = if ss == n_re_components && rr == n_re_components {
                    (xt_sigma3_x.clone(), trace_sigma_inv_sq)
                } else if ss == n_re_components {
                    let mut q_ab = DMatrix::<f64>::zeros(p, p);
                    let mut trace = 0.0;
                    for ((u_cols, v_cols), (_, m_v)) in components[rr].iter().zip(&m_rows[rr]) {
                        q_ab.gemm_tr(1.0, m_v, &gather_rows(&m2, u_cols), 1.0);
                        for level in 0..u_cols.n_levels {
                            trace += c2[(v_cols.index(level), u_cols.index(level))];
                        }
                    }
                    (q_ab, trace)
                } else {
                    let mut q_ab = DMatrix::<f64>::zeros(p, p);
                    let mut trace = 0.0;
                    for ((u_a, v_a), (_, m_va)) in components[rr].iter().zip(&m_rows[rr]) {
                        for ((u_b, v_b), (_, m_vb)) in components[ss].iter().zip(&m_rows[ss]) {
                            let c_block = gather_block(&c, u_a, u_b);
                            let middle = &c_block * m_vb;
                            q_ab.gemm_tr(1.0, m_va, &middle, 1.0);
                            trace += trace_of_product(v_a, u_b, v_b, u_a);
                        }
                    }
                    (q_ab, trace)
                };
                q_matrices.push(q_matrix);
                ktrace[(rr, ss)] = trace;
                ktrace[(ss, rr)] = trace;
            }
        }

        Ok(KenwardRogerIngredients {
            p_matrices,
            q_matrices,
            ktrace,
            component_labels,
        })
    }

    /// `mat ← Λ' mat` with the block-diagonal Λ of the random-effect terms.
    fn kr_apply_lambda_transpose_left(&self, offsets: &[usize], mat: &mut DMatrix<f64>) {
        let mut scratch = Vec::new();
        for (re, &offset) in self.reterms.iter().zip(offsets) {
            let v = re.vsize;
            scratch.resize(v, 0.0);
            for col in 0..mat.ncols() {
                for level in 0..re.n_levels() {
                    let base = offset + level * v;
                    for (r, slot) in scratch.iter_mut().enumerate() {
                        *slot = (r..v)
                            .map(|c| re.lambda[(c, r)] * mat[(base + c, col)])
                            .sum();
                    }
                    for (r, &value) in scratch.iter().enumerate() {
                        mat[(base + r, col)] = value;
                    }
                }
            }
        }
    }

    /// `mat ← Λ mat` with the block-diagonal Λ of the random-effect terms.
    fn kr_apply_lambda_left(&self, offsets: &[usize], mat: &mut DMatrix<f64>) {
        let mut scratch = Vec::new();
        for (re, &offset) in self.reterms.iter().zip(offsets) {
            let v = re.vsize;
            scratch.resize(v, 0.0);
            for col in 0..mat.ncols() {
                for level in 0..re.n_levels() {
                    let base = offset + level * v;
                    for (r, slot) in scratch.iter_mut().enumerate() {
                        *slot = (0..=r)
                            .map(|c| re.lambda[(r, c)] * mat[(base + c, col)])
                            .sum();
                    }
                    for (r, &value) in scratch.iter().enumerate() {
                        mat[(base + r, col)] = value;
                    }
                }
            }
        }
    }

    /// Exact version of the historical positive-definiteness gate on
    /// `Σ = σ²(I + ZΛΛ'Z')`, from the eigenvalues of `factor = Λ'SΛ + I`:
    /// the spectrum of `Σ/σ²` is the largest `min(n, q)` of them plus
    /// `max(n − q, 0)` ones.
    fn kr_exact_positive_definite_gate(
        factor: &DMatrix<f64>,
        sigma_sq: f64,
        n: usize,
    ) -> Result<()> {
        let q = factor.nrows();
        let mut eigenvalues: Vec<f64> = SymmetricEigen::new(factor.clone())
            .eigenvalues
            .iter()
            .copied()
            .collect();
        eigenvalues.sort_by(|a, b| b.total_cmp(a));
        let largest = eigenvalues.first().copied().unwrap_or(1.0).max(1.0) * sigma_sq;
        let smallest = if n > q || q == 0 {
            sigma_sq
        } else {
            eigenvalues[n.saturating_sub(1)] * sigma_sq
        };
        let tolerance = (1e-10 * largest.abs().max(1.0)).max(1e-12);
        if smallest > tolerance {
            Ok(())
        } else {
            Err(MixedModelError::Singular(
                "Kenward-Roger adjusted covariance requires a positive-definite response covariance"
                    .to_string(),
            ))
        }
    }
}
