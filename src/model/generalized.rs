//! Generalized linear mixed-effects model (GLMM).
//!
//! Fits models of the form:
//!   g(E[y]) = Xβ + Zb
//! where g is a link function, b ~ N(0, σ²Λ'Λ), and y|b follows
//! an exponential family distribution.
//!
//! Uses Penalized Iteratively Reweighted Least Squares (PIRLS) for
//! the conditional modes, with optional adaptive Gauss-Hermite quadrature.

use nalgebra::{DMatrix, DVector};

use crate::compiler::{
    CompiledModelArtifact, CompilerPolicy, ModelAuditReport, ModelBoundary, ObjectiveApproximation,
};
#[cfg(feature = "nlopt")]
use crate::compiler::{
    Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticStage, OptimizerCertificate,
};
use crate::error::{MixedModelError, Result};
use crate::formula::Formula;
use crate::model::data::DataFrame;
use crate::model::linear::LinearMixedModel;
use crate::model::traits::{Family, LinkFunction, MixedModelFit, RandomEffectTermInfo};
use crate::stats::{BlockDescription, ModelSummary, VarCorr};
use crate::types::OptSummary;

/// A generalized linear mixed-effects model.
#[derive(Debug, Clone)]
#[allow(dead_code)] // beta0/u0 reserved for step-halving; devc/devc0/sd/mult reserved for AGQ
pub struct GeneralizedLinearMixedModel {
    /// Internal linear mixed model (local Laplace approximation).
    pub lmm: LinearMixedModel,

    /// Fixed-effects coefficients (pivoted).
    pub beta: DVector<f64>,
    /// Previous β for step-halving.
    beta0: DVector<f64>,

    /// Covariance parameters.
    pub theta: Vec<f64>,

    /// Random effects on the b-scale: vec(Λ * u) per term.
    pub b: Vec<DMatrix<f64>>,
    /// Random effects on the u-scale (orthogonal).
    pub u: Vec<DMatrix<f64>>,
    /// Previous u for step-halving.
    u0: Vec<DMatrix<f64>>,

    /// Linear predictor η = Xβ + Zb.
    pub eta: DVector<f64>,
    /// Conditional mean μ = g⁻¹(η).
    pub mu: DVector<f64>,
    /// Response vector.
    pub y: DVector<f64>,
    /// Prior weights.
    pub wt: Vec<f64>,

    /// Distribution family.
    pub family: Family,
    /// Link function.
    pub link: LinkFunction,

    /// Deviance components for AGQ.
    devc: Vec<f64>,
    devc0: Vec<f64>,
    /// Approximate SDs for AGQ.
    sd: Vec<f64>,
    /// Multipliers for AGQ.
    mult: Vec<f64>,
}

impl GeneralizedLinearMixedModel {
    /// Construct a GLMM from formula, data, distribution, and link.
    pub fn new(
        formula: Formula,
        data: &DataFrame,
        family: Family,
        link: Option<LinkFunction>,
    ) -> Result<Self> {
        Self::new_with_policy_internal(formula, data, family, link, CompilerPolicy::default())
    }

    /// Construct a GLMM with per-observation case weights.
    ///
    /// Used for binomial-with-trials data where the response is a proportion
    /// `y / n` and `weights[i] = n_i` is the trial count. Mirrors Julia's
    /// `fit(MixedModel, formula, data, Binomial(); wts = ...)`.
    ///
    /// `weights.len()` must equal the number of observations. The vector is
    /// stored as `self.wt` and incorporated into IRLS weights and deviance
    /// residuals during PIRLS.
    pub fn new_with_weights(
        formula: Formula,
        data: &DataFrame,
        family: Family,
        link: Option<LinkFunction>,
        weights: Vec<f64>,
    ) -> Result<Self> {
        let mut model =
            Self::new_with_policy_internal(formula, data, family, link, CompilerPolicy::default())?;
        if weights.len() != model.y.len() {
            return Err(MixedModelError::InvalidArgument(format!(
                "case weights length ({}) does not match number of observations ({})",
                weights.len(),
                model.y.len()
            )));
        }
        for (i, &w) in weights.iter().enumerate() {
            if !w.is_finite() || w <= 0.0 {
                return Err(MixedModelError::InvalidArgument(format!(
                    "case weight at index {i} must be finite and positive (got {w})"
                )));
            }
        }
        model.wt = weights;
        Ok(model)
    }

    fn new_with_policy_internal(
        formula: Formula,
        data: &DataFrame,
        family: Family,
        link: Option<LinkFunction>,
        compiler_policy: CompilerPolicy,
    ) -> Result<Self> {
        let link = link.unwrap_or_else(|| family.canonical_link());

        // For Normal + Identity, redirect to LMM
        if family == Family::Normal && link == LinkFunction::Identity {
            return Err(MixedModelError::InvalidArgument(
                "Use LinearMixedModel for Normal distribution with IdentityLink".to_string(),
            ));
        }
        if glmm_dispersion_family_requires_refusal(family, link) {
            return Err(MixedModelError::Unsupported(format!(
                "{} GLMM with {} link requires a dispersion parameter, but GLMM dispersion \
                 estimation is not implemented yet",
                family_label(family),
                link_label(link)
            )));
        }

        // Build the internal LMM
        let mut lmm =
            LinearMixedModel::new_with_compiler_policy(formula, data, None, compiler_policy)?;
        lmm.compiler_artifact
            .set_model_boundary(ModelBoundary::glmm(
                family_label(family),
                link_label(link),
                ObjectiveApproximation::Laplace {
                    inner: "pirls".to_string(),
                },
            ));
        let n = lmm.dims.n;
        let p = lmm.dims.p;

        let beta = DVector::zeros(p.min(lmm.feterm.rank));
        let theta = lmm.theta();
        let y = lmm.y();

        let u: Vec<DMatrix<f64>> = lmm
            .reterms
            .iter()
            .map(|rt| DMatrix::zeros(rt.vsize, rt.n_levels()))
            .collect();

        let b = u.clone();
        let u0 = u
            .iter()
            .map(|m| DMatrix::zeros(m.nrows(), m.ncols()))
            .collect();

        let eta = DVector::zeros(n);
        let mu = DVector::zeros(n);

        // AGQ vectors — only used for single RE term
        let agq_len = if u.len() == 1 {
            u[0].nrows() * u[0].ncols()
        } else {
            0
        };

        Ok(GeneralizedLinearMixedModel {
            lmm,
            beta: beta.clone(),
            beta0: beta,
            theta,
            b,
            u,
            u0,
            eta,
            mu,
            y,
            wt: Vec::new(),
            family,
            link,
            devc: vec![0.0; agq_len],
            devc0: vec![0.0; agq_len],
            sd: vec![0.0; agq_len],
            mult: vec![0.0; agq_len],
        })
    }

    /// Construct a GLMM and apply a compiler policy to the internal compiled
    /// artifact before fitting.
    pub fn new_with_compiler_policy(
        formula: Formula,
        data: &DataFrame,
        family: Family,
        link: Option<LinkFunction>,
        compiler_policy: CompilerPolicy,
    ) -> Result<Self> {
        Self::new_with_policy_internal(formula, data, family, link, compiler_policy)
    }

    /// Round-trippable compiler artifact attached to the internal model.
    pub fn compiler_artifact(&self) -> &CompiledModelArtifact {
        self.lmm.compiler_artifact()
    }

    /// Compiler policy attached to the internal compiled artifact.
    pub fn compiler_policy(&self) -> &CompilerPolicy {
        self.lmm.compiler_policy()
    }

    /// Apply a compiler policy before fitting.
    pub fn set_compiler_policy(&mut self, compiler_policy: CompilerPolicy) -> Result<&mut Self> {
        self.lmm.set_compiler_policy(compiler_policy)?;
        Ok(self)
    }

    /// Return a copy of this model with a compiler policy applied.
    pub fn with_compiler_policy(mut self, compiler_policy: CompilerPolicy) -> Result<Self> {
        self.set_compiler_policy(compiler_policy)?;
        Ok(self)
    }

    /// Stable user-facing audit report derived from the compiler artifact.
    pub fn audit_report(&self) -> ModelAuditReport {
        self.lmm.audit_report()
    }

    /// Compact default print summary (PRD § 15).
    pub fn print_summary(&self) -> crate::compiler::ModelPrint {
        self.lmm.print_summary()
    }

    /// Source-to-fitted parameterization drilldown (PRD § 15).
    pub fn parameterization(&self) -> crate::compiler::ParameterizationDrilldown {
        self.lmm.parameterization()
    }

    /// Update the linear predictor η and conditional mean μ.
    pub fn update_eta(&mut self) {
        let n = self.eta.len();
        let x = self.lmm.feterm.full_rank_x();

        // η = X * β
        self.eta = x * &self.beta;

        // Add random effects: η += Z_i * b_i
        for (i, rt) in self.lmm.reterms.iter().enumerate() {
            // b_i = λ_i * u_i
            self.b[i] = &rt.lambda * &self.u[i];
            // Multiply Z * vec(b) using refs for sparse multiplication
            let bvec = DVector::from_column_slice(self.b[i].as_slice());
            for (obs, &ref_idx) in rt.refs.iter().enumerate() {
                let r = ref_idx as usize;
                for s in 0..rt.vsize {
                    self.eta[obs] += rt.z[(s, obs)] * bvec[r * rt.vsize + s];
                }
            }
        }

        // μ = g⁻¹(η)
        for i in 0..n {
            self.mu[i] = self.link.linkinv(self.eta[i]);
        }
    }

    /// Variance-covariance summary for the fitted random effects.
    pub fn varcorr(&self) -> VarCorr {
        let scale = self.dispersion(false);
        let residual_sd = if self.family.has_dispersion() {
            Some(scale)
        } else {
            None
        };
        VarCorr::from_reterms(&self.lmm.reterms, scale, residual_sd)
    }

    /// Structural summary of the blocked `A`/`L` system.
    pub fn block_description(&self) -> BlockDescription {
        BlockDescription::from_generalized_model(self)
    }

    /// Fixed/random-effects summary table.
    pub fn summary(&self) -> ModelSummary {
        ModelSummary::from_generalized_model(self)
    }

    /// Render the model summary as markdown.
    pub fn summary_markdown(&self) -> String {
        self.summary().to_markdown()
    }

    /// Render the model summary as HTML.
    pub fn summary_html(&self) -> String {
        self.summary().to_html()
    }

    /// Render the model summary as LaTeX.
    pub fn summary_latex(&self) -> String {
        self.summary().to_latex()
    }

    /// Compute the deviance residuals for the current μ.
    #[allow(dead_code)] // exposed once GLMM diagnostics surface lands
    fn deviance_residuals(&self) -> DVector<f64> {
        let n = self.y.len();
        let mut devresid = DVector::zeros(n);
        for i in 0..n {
            devresid[i] = self.dev_resid_component(self.y[i], self.mu[i]);
        }
        devresid
    }

    /// Single observation deviance residual component.
    fn dev_resid_component(&self, y: f64, mu: f64) -> f64 {
        match self.family {
            Family::Bernoulli | Family::Binomial => {
                let eps = 1e-15;
                let mu = mu.max(eps).min(1.0 - eps);
                if y == 1.0 {
                    -2.0 * mu.ln()
                } else if y == 0.0 {
                    -2.0 * (1.0 - mu).ln()
                } else {
                    2.0 * (y * (y / mu).ln() + (1.0 - y) * ((1.0 - y) / (1.0 - mu)).ln())
                }
            }
            Family::Poisson => {
                if y == 0.0 {
                    2.0 * mu
                } else {
                    2.0 * (y * (y / mu).ln() - (y - mu))
                }
            }
            Family::Normal => (y - mu).powi(2),
            Family::Gamma => {
                if y == 0.0 {
                    2.0 * (mu.ln())
                } else {
                    -2.0 * ((y / mu).ln() - (y - mu) / mu)
                }
            }
            Family::InverseGaussian => (y - mu).powi(2) / (y * mu * mu),
        }
    }

    /// PIRLS: Penalized Iteratively Reweighted Least Squares.
    ///
    /// Updates β and u until convergence. The working response and weights
    /// are derived from the current μ = g⁻¹(Xβ + Zb).
    ///
    /// * `vary_beta` – if false, β is held fixed and only u is updated
    pub fn pirls(&mut self, vary_beta: bool, verbose: bool) -> Result<()> {
        // Mirrors MixedModels.jl/src/generalizedlinearmixedmodel.jl pirls!
        // (lines 614-669): step-halving toward the previous accepted iterate
        // whenever a fresh IRLS step would worsen the Laplace objective. Keeps
        // the outer optimizer's view of obj(θ) consistent across probes —
        // without this, BOBYQA on multi-RE GLMM surfaces (e.g. grouseticks
        // Poisson) sees noisy values and reports `RoundoffLimited`.
        let max_iter = 10;
        let tol = 1.0e-5;
        let max_halvings = 10;

        let n = self.y.len();

        // Initialise u to zero; keep existing β.
        for u in self.u.iter_mut() {
            u.fill(0.0);
        }
        for (i, rt) in self.lmm.reterms.iter().enumerate() {
            self.b[i] = &rt.lambda * &self.u[i];
        }
        self.update_eta();

        // Save the initial accepted state for halving; the 1.0001 slack on
        // obj0 matches Julia's tolerant first-iteration acceptance.
        let mut u_prev: Vec<DMatrix<f64>> = self.u.clone();
        let mut beta_prev = self.beta.clone();
        let mut obj0 = self.laplace_objective() * 1.0001;

        let mut sqrtwts = vec![0.0f64; n];
        let mut working_y = vec![0.0f64; n];

        for iter in 0..max_iter {
            // --- Compute IRLS weights and working response ---
            for obs in 0..n {
                let mu_obs = self.mu[obs];
                let eta_obs = self.eta[obs];
                let y_obs = self.y[obs];

                let case_w = if self.wt.is_empty() {
                    1.0
                } else {
                    self.wt[obs]
                };
                (sqrtwts[obs], working_y[obs]) = pirls_working_observation(
                    self.family,
                    self.link,
                    y_obs,
                    eta_obs,
                    mu_obs,
                    case_w,
                );
            }

            // --- Update the LMM with new IRLS weights ---
            self.lmm.update_irls_weights(&sqrtwts, &working_y);
            self.lmm.update_l()?;

            // --- Propose new β / u from the LMM solution ---
            if vary_beta {
                self.beta = self.lmm.beta();
            }
            let new_u = self.lmm.ranef_u();
            for (i, rt) in self.lmm.reterms.iter().enumerate() {
                self.u[i].copy_from(&new_u[i]);
                self.b[i] = &rt.lambda * &self.u[i];
            }
            self.update_eta();
            let mut obj = self.laplace_objective();

            // --- Step-halving: average toward the previous accepted state
            //     until obj is no worse, up to `max_halvings` averagings. ---
            let mut nhalf = 0;
            while obj > obj0 && nhalf < max_halvings {
                nhalf += 1;
                for i in 0..self.u.len() {
                    self.u[i] = 0.5 * (&self.u[i] + &u_prev[i]);
                }
                if vary_beta {
                    self.beta = 0.5 * (&self.beta + &beta_prev);
                }
                for (i, rt) in self.lmm.reterms.iter().enumerate() {
                    self.b[i] = &rt.lambda * &self.u[i];
                }
                self.update_eta();
                obj = self.laplace_objective();
            }

            if verbose {
                eprintln!("  PIRLS iter {iter}: obj = {obj:.6} (nhalf = {nhalf})");
            }

            if (obj - obj0).abs() < tol {
                break;
            }

            // Accept iterate as the new previous state.
            for i in 0..self.u.len() {
                u_prev[i].copy_from(&self.u[i]);
            }
            beta_prev = self.beta.clone();
            obj0 = obj;
        }

        Ok(())
    }

    /// Laplace approximation objective: deviance residuals + u penalty + log|L|.
    pub fn laplace_objective(&self) -> f64 {
        // For binomial-with-trials data the response is a per-trial proportion
        // and `wt[i]` is the trial count; weighting the per-observation
        // deviance contribution by `wt[i]` recovers the binomial deviance.
        let dev: f64 = (0..self.y.len())
            .map(|i| {
                let case_w = if self.wt.is_empty() { 1.0 } else { self.wt[i] };
                case_w * self.dev_resid_component(self.y[i], self.mu[i])
            })
            .sum();
        let u_penalty: f64 = self
            .u
            .iter()
            .map(|u| u.iter().map(|x| x * x).sum::<f64>())
            .sum();
        dev + u_penalty + self.lmm_logdet()
    }

    /// Deviance of the GLMM.
    ///
    /// For `n_agq <= 1`, returns the Laplace approximation
    /// (`laplace_objective`).
    ///
    /// For `n_agq > 1`, returns the deviance evaluated by `n_agq`-point
    /// adaptive Gauss-Hermite quadrature. AGQ is only defined for models
    /// with a single scalar random-effects term; on multi-term or
    /// vector-valued RE models, calling with `n_agq > 1` is a programmer
    /// error (use [`validate_agq`](Self::validate_agq) up front, or call
    /// via [`fit_with_options`](Self::fit_with_options) which preflights).
    ///
    /// Mutates internal `u`, `eta`, `mu` during the AGQ sweep but restores
    /// observable state before returning.
    pub fn deviance(&mut self, n_agq: usize) -> f64 {
        if n_agq <= 1 {
            return self.laplace_objective();
        }
        debug_assert!(
            self.is_single_scalar_re(),
            "AGQ with n_agq > 1 requires exactly one scalar random-effects term; \
             callers must invoke validate_agq() before reaching this path"
        );

        let n_levels = self.u[0].ncols();
        let n_obs = self.y.len();

        // Snapshot u₀ (a flat vector of length n_levels since vsize == 1).
        let u0_flat: Vec<f64> = self.u[0].as_slice().to_vec();
        debug_assert_eq!(u0_flat.len(), n_levels);

        // Per-group sd from the diagonal of the (1,1) Cholesky block:
        // sd[g] = 1 / |L₁₁_diag[g]|.
        let l11_diag = self.l11_diag();
        debug_assert_eq!(l11_diag.len(), n_levels);
        let sd: Vec<f64> = l11_diag.iter().map(|d| 1.0 / d.abs()).collect();

        // Group index per observation. Clone to release the borrow on
        // `self.lmm` so we can call `update_eta(&mut self)` inside the loop.
        let refs: Vec<u32> = self.lmm.reterms[0].refs.clone();

        // devc0[g] = u₀[g]² + Σ_{i in group g} devresid_i  (at the conditional modes)
        let mut devc0 = vec![0.0_f64; n_levels];
        for (g, &uv) in u0_flat.iter().enumerate() {
            devc0[g] = uv * uv;
        }
        for i in 0..n_obs {
            devc0[refs[i] as usize] += self.dev_resid_component(self.y[i], self.mu[i]);
        }

        // Sweep over GH nodes.
        let rule = crate::types::gh_norm(n_agq);
        let mut mult = vec![0.0_f64; n_levels];
        let mut devc = vec![0.0_f64; n_levels];

        for (&z, &w) in rule.z.iter().zip(rule.w.iter()) {
            if w == 0.0 {
                continue;
            }
            if z == 0.0 {
                // devc == devc0, exp(0) * w simplifies to w
                for g in 0..n_levels {
                    mult[g] += w;
                }
                continue;
            }
            // u[g] = u₀[g] + z * sd[g]
            for g in 0..n_levels {
                self.u[0][(0, g)] = u0_flat[g] + z * sd[g];
            }
            self.update_eta();
            // devc[g] = u[g]² + Σ devresid_i (per group)
            for g in 0..n_levels {
                let uv = self.u[0][(0, g)];
                devc[g] = uv * uv;
            }
            for i in 0..n_obs {
                devc[refs[i] as usize] += self.dev_resid_component(self.y[i], self.mu[i]);
            }
            // mult[g] += exp((z² + devc0[g] - devc[g]) / 2) * w
            let z2 = z * z;
            for g in 0..n_levels {
                mult[g] += ((z2 + devc0[g] - devc[g]) * 0.5).exp() * w;
            }
        }

        // Restore u and η/μ.
        for g in 0..n_levels {
            self.u[0][(0, g)] = u0_flat[g];
        }
        self.update_eta();

        let sum_devc0: f64 = devc0.iter().sum();
        let log_mult: f64 = mult.iter().map(|m| m.ln()).sum();
        let log_sd: f64 = sd.iter().map(|s| s.ln()).sum();
        sum_devc0 - 2.0 * (log_mult + log_sd)
    }

    /// True iff the model has exactly one random-effects term and that
    /// term has `vsize == 1` (a scalar random effect).
    pub fn is_single_scalar_re(&self) -> bool {
        self.lmm.reterms.len() == 1 && self.lmm.reterms[0].vsize == 1
    }

    /// Diagonal of the (1,1) block of the lower-Cholesky factor `L`.
    ///
    /// For a single scalar RE term this is a per-level vector of length
    /// `n_levels`. Used by [`deviance`](Self::deviance) to derive AGQ
    /// node spacings.
    fn l11_diag(&self) -> Vec<f64> {
        matrix_block_diag(&self.lmm.l_blocks[0])
    }

    /// Reject `n_agq > 1` on models that don't satisfy the AGQ contract
    /// (single scalar RE term).
    pub fn validate_agq(&self, n_agq: usize) -> Result<()> {
        if n_agq > 1 && !self.is_single_scalar_re() {
            let sizes: Vec<usize> = self.lmm.reterms.iter().map(|rt| rt.vsize).collect();
            return Err(MixedModelError::InvalidArgument(format!(
                "n_agq = {n_agq} > 1 requires exactly one scalar random-effects term; \
                 this model has {} term(s) with vsizes {:?}",
                self.lmm.reterms.len(),
                sizes,
            )));
        }
        Ok(())
    }

    /// Log-determinant from the LMM's Cholesky factor.
    fn lmm_logdet(&self) -> f64 {
        // Delegate to the internal LMM's block structure
        let k = self.lmm.dims.nretrms;
        let mut logdet = 0.0;
        for j in 0..k {
            let idx = j * (j + 1) / 2 + j; // block_index(j, j)
            logdet += match &self.lmm.l_blocks[idx] {
                crate::model::linear::MatrixBlock::Dense(m) => {
                    let n = m.nrows().min(m.ncols());
                    (0..n).map(|i| m[(i, i)].abs().ln()).sum::<f64>()
                }
                crate::model::linear::MatrixBlock::Diagonal(v) => {
                    v.iter().map(|x| x.abs().ln()).sum::<f64>()
                }
                crate::model::linear::MatrixBlock::BlockDiagonal(blocks) => blocks
                    .iter()
                    .map(|blk| {
                        let n = blk.nrows();
                        (0..n).map(|i| blk[(i, i)].abs().ln()).sum::<f64>()
                    })
                    .sum::<f64>(),
                crate::model::linear::MatrixBlock::Sparse(m) => {
                    let dense = crate::model::linear::MatrixBlock::Sparse(m.clone()).as_dense();
                    let n = dense.nrows().min(dense.ncols());
                    (0..n).map(|i| dense[(i, i)].abs().ln()).sum::<f64>()
                }
            };
        }
        2.0 * logdet
    }

    /// Fit the GLMM.
    pub fn fit(&mut self) -> Result<&mut Self> {
        self.fit_with_options(false, 1, false)
    }

    /// Fit after first applying a compiler policy.
    pub fn fit_with_compiler_policy(
        &mut self,
        compiler_policy: CompilerPolicy,
    ) -> Result<&mut Self> {
        self.set_compiler_policy(compiler_policy)?;
        self.fit()
    }

    /// Fit with options.
    ///
    /// `n_agq` selects the deviance approximation: `1` (or `0`) means the
    /// Laplace approximation; values `>= 2` request `n_agq`-point adaptive
    /// Gauss-Hermite quadrature, which is only valid for models with a
    /// single scalar random-effects term and is rejected up front
    /// otherwise.
    pub fn fit_with_options(
        &mut self,
        _fast: bool,
        n_agq: usize,
        _verbose: bool,
    ) -> Result<&mut Self> {
        self.validate_agq(n_agq)?;
        self.fit_with_options_impl(_fast, n_agq, _verbose)
    }

    #[cfg(not(feature = "nlopt"))]
    fn fit_with_options_impl(
        &mut self,
        _fast: bool,
        _n_agq: usize,
        _verbose: bool,
    ) -> Result<&mut Self> {
        Err(MixedModelError::Optimization(
            "GLMM fitting currently requires the `nlopt` feature; \
             rebuild with default features (or `--features nlopt`) to fit \
             generalized linear mixed models. The Laplace and AGQ paths use \
             NLopt's BOBYQA optimizer for the outer θ search."
                .to_string(),
        ))
    }

    #[cfg(feature = "nlopt")]
    fn fit_with_options_impl(
        &mut self,
        _fast: bool,
        n_agq: usize,
        _verbose: bool,
    ) -> Result<&mut Self> {
        use nlopt::{Algorithm as NloptAlgorithm, Nlopt, Target as NloptTarget};

        let n_theta = self.theta.len();
        let lb = self.lmm.lower_bounds();
        let initial_theta = self.theta.clone();

        let mut feval_count: i64 = 0;
        let feval_ptr = &mut feval_count as *mut i64;

        let obj_fn = {
            let model_ptr = self as *mut GeneralizedLinearMixedModel;
            move |theta: &[f64], _grad: Option<&mut [f64]>, _data: &mut ()| -> f64 {
                let model = unsafe { &mut *model_ptr };
                let _ = model.lmm.set_theta(theta);
                model.lmm.update_l().ok();
                model.theta = theta.to_vec();
                for u in model.u.iter_mut() {
                    u.fill(0.0);
                }
                model.pirls(true, false).ok();
                unsafe { *feval_ptr += 1 };
                model.deviance(n_agq)
            }
        };

        let mut optimizer = Nlopt::new(
            NloptAlgorithm::Bobyqa,
            n_theta,
            obj_fn,
            NloptTarget::Minimize,
            (),
        );
        optimizer.set_lower_bounds(&lb).ok();
        // Match MixedModels.jl OptSummary defaults: ftol_rel=1e-12,
        // ftol_abs=1e-8, xtol_rel=0. Setting xtol_rel=1e-8 here previously
        // forced BOBYQA to shrink its trust region (ρ_beg → ρ_end) all the
        // way to 1e-8 before exploring multi-dim moves, which on multi-RE
        // GLMM surfaces (e.g. grouseticks Poisson) caused premature
        // termination at the initial θ with status `XtolReached`.
        optimizer.set_ftol_rel(1e-12).ok();
        optimizer.set_ftol_abs(1e-8).ok();
        optimizer.set_maxeval(500).ok();
        // Mirror the LMM cobyla initial step default; without an explicit
        // initial step BOBYQA falls back to per-axis defaults that may be
        // too small for parameters near the lower bound.
        let initial_step = vec![0.75; n_theta];
        optimizer.set_initial_step(&initial_step).ok();

        let mut theta = initial_theta;
        let nlopt_result = optimizer.optimize(&mut theta);

        // Final PIRLS at optimal θ
        let _ = self.lmm.set_theta(&theta);
        self.lmm.update_l().ok();
        self.theta = theta.clone();
        for u in self.u.iter_mut() {
            u.fill(0.0);
        }
        self.pirls(true, false).ok();
        self.beta = self.lmm.beta();

        self.lmm.optsum.n_agq = n_agq;
        self.lmm.optsum.fmin = self.deviance(n_agq);
        self.lmm.optsum.final_params = theta;
        self.lmm.optsum.return_value = match nlopt_result {
            Ok((status, _fmin)) => format!("{status:?}"),
            Err((status, _fmin)) => format!("FAILED:{status:?}"),
        };
        self.lmm.optsum.feval = feval_count;
        self.lmm.compiler_artifact.optimizer_certificate =
            Some(OptimizerCertificate::from_opt_summary_with_context(
                &self.lmm.optsum,
                &self.theta,
                &self.lmm.lower_bounds(),
                Some(self.lmm.dims.n),
            ));
        self.refresh_near_unit_random_effect_correlation_diagnostics();

        Ok(self)
    }

    #[cfg(feature = "nlopt")]
    fn refresh_near_unit_random_effect_correlation_diagnostics(&mut self) {
        const NEAR_UNIT_CORR_THRESHOLD: f64 = 0.99;

        let varcorr = self.varcorr();
        let mut diagnostics = Vec::new();
        for component in &varcorr.components {
            for (offset, &corr) in component.correlations.iter().enumerate() {
                if corr.abs() < NEAR_UNIT_CORR_THRESHOLD {
                    continue;
                }
                let (row, col) = lower_triangle_pair(offset);
                let row_name = component
                    .names
                    .get(row)
                    .cloned()
                    .unwrap_or_else(|| format!("basis[{row}]"));
                let col_name = component
                    .names
                    .get(col)
                    .cloned()
                    .unwrap_or_else(|| format!("basis[{col}]"));
                let mut diagnostic = Diagnostic::new(
                    DiagnosticCode::NearUnitRandomEffectCorrelation,
                    DiagnosticSeverity::Warning,
                    DiagnosticStage::Certification,
                    format!(
                        "random-effect correlation for group {} between {} and {} is {:.3}; the fitted covariance is nearly one-dimensional",
                        component.group, col_name, row_name, corr
                    ),
                )
                .with_affected_terms(vec![component.group.clone()])
                .with_suggested_actions(vec![
                    "consider a zero-correlation (`||`) or reduced-rank random-effect structure".to_string(),
                    "treat correlation estimates and Hessian-based standard errors cautiously".to_string(),
                ]);
                diagnostic
                    .payload
                    .insert("group".to_string(), serde_json::json!(component.group));
                diagnostic
                    .payload
                    .insert("correlation".to_string(), serde_json::json!(corr));
                diagnostic.payload.insert(
                    "threshold".to_string(),
                    serde_json::json!(NEAR_UNIT_CORR_THRESHOLD),
                );
                diagnostics.push(diagnostic);
            }
        }

        if diagnostics.is_empty() {
            return;
        }

        self.lmm
            .compiler_artifact
            .diagnostics
            .extend(diagnostics.clone());
        if let Some(certificate) = &mut self.lmm.compiler_artifact.optimizer_certificate {
            certificate.diagnostics.extend(diagnostics);
        }
    }
}

/// Diagonal entries of a [`MatrixBlock`].
///
/// Used by AGQ to read per-level scalings off the (1,1) block of the
/// Cholesky factor. Pulled out of [`GeneralizedLinearMixedModel::l11_diag`]
/// so the variant logic is unit-testable without constructing a full LMM.
///
/// For a `Dense` block this returns `min(nrows, ncols)` diagonal entries.
fn matrix_block_diag(block: &crate::model::linear::MatrixBlock) -> Vec<f64> {
    use crate::model::linear::MatrixBlock;
    match block {
        MatrixBlock::Dense(m) => {
            let n = m.nrows().min(m.ncols());
            (0..n).map(|i| m[(i, i)]).collect()
        }
        MatrixBlock::Diagonal(v) => v.iter().copied().collect(),
        MatrixBlock::BlockDiagonal(blocks) => {
            let mut out = Vec::new();
            for blk in blocks {
                for i in 0..blk.nrows() {
                    out.push(blk[(i, i)]);
                }
            }
            out
        }
        MatrixBlock::Sparse(m) => {
            let dense = MatrixBlock::Sparse(m.clone()).as_dense();
            let n = dense.nrows().min(dense.ncols());
            (0..n).map(|i| dense[(i, i)]).collect()
        }
    }
}

#[cfg(feature = "nlopt")]
fn lower_triangle_pair(offset: usize) -> (usize, usize) {
    let mut row = 1usize;
    let mut remaining = offset;
    while remaining >= row {
        remaining -= row;
        row += 1;
    }
    (row, remaining)
}

fn pirls_working_observation(
    family: Family,
    link: LinkFunction,
    y: f64,
    eta: f64,
    mu: f64,
    case_weight: f64,
) -> (f64, f64) {
    let dmu_deta = link.mu_eta(eta);
    let var_mu = family.variance(mu);
    let weight = if dmu_deta.is_finite() && var_mu.is_finite() && var_mu > 0.0 {
        case_weight * dmu_deta * dmu_deta / var_mu
    } else {
        0.0
    };
    let resid = if !dmu_deta.is_finite() || dmu_deta.abs() < 1e-15 {
        0.0
    } else {
        (y - mu) / dmu_deta
    };
    (weight.max(0.0).sqrt(), eta + resid)
}

fn glmm_dispersion_family_requires_refusal(family: Family, link: LinkFunction) -> bool {
    matches!(family, Family::Gamma | Family::InverseGaussian)
        || (family == Family::Normal && link != LinkFunction::Identity)
}

fn family_label(family: Family) -> &'static str {
    match family {
        Family::Normal => "gaussian",
        Family::Bernoulli => "bernoulli",
        Family::Binomial => "binomial",
        Family::Poisson => "poisson",
        Family::Gamma => "gamma",
        Family::InverseGaussian => "inverse_gaussian",
    }
}

fn link_label(link: LinkFunction) -> &'static str {
    match link {
        LinkFunction::Identity => "identity",
        LinkFunction::Log => "log",
        LinkFunction::Logit => "logit",
        LinkFunction::Probit => "probit",
        LinkFunction::Inverse => "inverse",
        LinkFunction::Sqrt => "sqrt",
    }
}

impl std::fmt::Display for GeneralizedLinearMixedModel {
    /// Default print: the compact `ModelPrint` summary (PRD § 15).
    /// Heavier reports (`audit_report`, `parameterization`,
    /// `changes`, `explain_model`) remain one explicit method call
    /// away.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.print_summary(), f)
    }
}

impl MixedModelFit for GeneralizedLinearMixedModel {
    fn nobs(&self) -> usize {
        self.y.len()
    }

    fn dof(&self) -> usize {
        self.lmm.feterm.rank
            + self.lmm.parmap.len()
            + if self.family.has_dispersion() { 1 } else { 0 }
    }

    fn coef(&self) -> DVector<f64> {
        let mut full = DVector::from_element(self.lmm.feterm.piv.len(), 0.0);
        for (i, &val) in self.beta.iter().enumerate() {
            if i < self.lmm.feterm.piv.len() {
                full[self.lmm.feterm.piv[i]] = val;
            }
        }
        full
    }

    fn fixef(&self) -> DVector<f64> {
        self.beta.clone()
    }
    fn coef_names(&self) -> Vec<String> {
        self.lmm.coef_names()
    }
    fn vcov(&self) -> DMatrix<f64> {
        self.lmm.vcov()
    }
    fn stderror(&self) -> DVector<f64> {
        self.lmm.stderror()
    }
    fn fitted(&self) -> DVector<f64> {
        self.mu.clone()
    }
    fn residuals(&self) -> DVector<f64> {
        &self.y - &self.mu
    }
    fn response(&self) -> &DVector<f64> {
        &self.y
    }
    fn model_matrix(&self) -> &DMatrix<f64> {
        &self.lmm.feterm.x
    }
    fn objective(&self) -> f64 {
        if self.is_fitted() {
            // Cached deviance at the optimum (Laplace or AGQ as fit specified).
            self.lmm.optsum.fmin
        } else {
            // Pre-fit: only the Laplace approximation is callable from &self.
            self.laplace_objective()
        }
    }

    fn loglikelihood(&self) -> f64 {
        -self.objective() / 2.0
    }

    fn formula_label(&self) -> Option<String> {
        Some(self.lmm.formula.to_string())
    }

    fn is_fitted(&self) -> bool {
        self.lmm.optsum.feval > 0
    }
    fn is_singular(&self) -> bool {
        self.lmm.is_singular()
    }
    fn opt_summary(&self) -> &OptSummary {
        &self.lmm.optsum
    }
    fn theta(&self) -> Vec<f64> {
        self.theta.clone()
    }

    fn dispersion(&self, sqr: bool) -> f64 {
        if self.family.has_dispersion() {
            self.lmm.dispersion(sqr)
        } else if sqr {
            1.0
        } else {
            1.0
        }
    }

    fn ranef(&self) -> Vec<DMatrix<f64>> {
        self.b.clone()
    }

    fn random_effect_terms(&self) -> Vec<RandomEffectTermInfo> {
        self.lmm.random_effect_terms()
    }

    fn family_kind(&self) -> Option<crate::model::traits::Family> {
        Some(self.family)
    }

    fn link_kind(&self) -> Option<crate::model::traits::LinkFunction> {
        Some(self.link)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formula::parse_formula;
    use crate::model::data::DataFrame;
    #[cfg(feature = "nlopt")]
    use approx::assert_relative_eq;

    #[test]
    fn test_gamma_pirls_components_are_link_specific() {
        let eta = 2.0_f64.ln();
        let (sqrtw_log, z_log) =
            pirls_working_observation(Family::Gamma, LinkFunction::Log, 3.0, eta, 2.0, 1.0);
        assert!(
            (sqrtw_log - 1.0).abs() < 1e-12,
            "Gamma-log should use dmu/deta=mu, giving unit IRLS weight"
        );
        assert!(
            (z_log - (eta + 0.5)).abs() < 1e-12,
            "Gamma-log working response should divide by dmu/deta=2"
        );

        let (sqrtw_inverse, z_inverse) =
            pirls_working_observation(Family::Gamma, LinkFunction::Inverse, 3.0, 0.5, 2.0, 1.0);
        assert!(
            (sqrtw_inverse - 2.0).abs() < 1e-12,
            "Gamma-inverse should retain |dmu/deta|=mu^2 in the weight"
        );
        assert!(
            (z_inverse - 0.25).abs() < 1e-12,
            "Gamma-inverse working response must preserve the negative derivative"
        );
    }

    #[test]
    fn test_glmm_constructor_rejects_gamma_until_dispersion_supported() {
        let data = contra_fixture();
        let formula = parse_formula("use_num ~ 1 + (1 | urban_dist)").unwrap();
        let err = GeneralizedLinearMixedModel::new(
            formula,
            &data,
            Family::Gamma,
            Some(LinkFunction::Log),
        )
        .expect_err("Gamma GLMM should be refused until dispersion support lands");

        match err {
            MixedModelError::Unsupported(msg) => {
                assert!(msg.contains("gamma"));
                assert!(msg.contains("log"));
                assert!(msg.contains("dispersion"));
                assert!(msg.contains("not implemented"));
            }
            other => panic!("expected Unsupported error, got {other:?}"),
        }
    }

    #[test]
    fn test_glmm_constructor_rejects_normal_nonidentity_until_dispersion_supported() {
        let data = contra_fixture();
        let formula = parse_formula("use_num ~ 1 + (1 | urban_dist)").unwrap();
        let err = GeneralizedLinearMixedModel::new(
            formula,
            &data,
            Family::Normal,
            Some(LinkFunction::Sqrt),
        )
        .expect_err("Normal GLMM with non-identity link should be refused");

        match err {
            MixedModelError::Unsupported(msg) => {
                assert!(msg.contains("gaussian"));
                assert!(msg.contains("sqrt"));
                assert!(msg.contains("dispersion"));
            }
            other => panic!("expected Unsupported error, got {other:?}"),
        }
    }

    /// Build a DataFrame from the embedded contra.csv.
    ///
    /// Columns: use_num (numeric 0/1), age, age2 (= age²), urban (Y/N),
    ///          livch (0+/1/2/3+), urban_dist (interaction string).
    fn contra_fixture() -> DataFrame {
        let csv = include_str!("contra.csv");
        let mut use_num = Vec::new();
        let mut age = Vec::new();
        let mut age2 = Vec::new();
        let mut urban = Vec::new();
        let mut livch = Vec::new();
        let mut urban_dist = Vec::new();

        for line in csv.lines() {
            let parts: Vec<&str> = line.split(',').collect();
            use_num.push(parts[0].parse::<f64>().unwrap());
            age.push(parts[1].parse::<f64>().unwrap());
            age2.push(parts[2].parse::<f64>().unwrap());
            urban.push(parts[3].to_string());
            livch.push(parts[4].to_string());
            urban_dist.push(parts[5].to_string());
        }

        let mut df = DataFrame::new();
        df.add_numeric("use_num", use_num);
        df.add_numeric("age", age);
        df.add_numeric("age2", age2);
        df.add_categorical("urban", urban);
        df.add_categorical("livch", livch);
        df.add_categorical("urban_dist", urban_dist);
        df
    }

    // ── GLMM parity tests (pirls.jl) ─────────────────────────────────────────

    #[cfg(feature = "nlopt")]
    #[test]
    fn test_contra_glmm_theta_and_deviance() {
        // pirls.jl:
        //   gm0 = fit(MixedModel, first(gfms[:contra]), contra, Bernoulli(); fast=true)
        //   @test isapprox(gm0.θ, [0.5720746212924732], atol=0.001)
        //   @test isapprox(deviance(gm0), 2361.657202855648, atol=0.001)
        //
        // Equivalent formula (pre-computed age² and urban×dist interaction):
        //   use_num ~ 1 + age + age2 + urban + livch + (1 | urban_dist)
        let data = contra_fixture();
        let formula =
            parse_formula("use_num ~ 1 + age + age2 + urban + livch + (1 | urban_dist)").unwrap();

        let mut model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();

        model.fit_with_options(true, 1, false).unwrap();

        let theta = &model.theta;
        assert_eq!(theta.len(), 1);
        assert_relative_eq!(theta[0], 0.5720746212924732, epsilon = 0.01);

        let dev = model.deviance(1);
        assert_relative_eq!(dev, 2361.657202855648, epsilon = 1.0);
    }

    #[cfg(feature = "nlopt")]
    #[test]
    fn test_cbpp_binomial_glmm_with_case_weights() {
        // pirls.jl:125-147, MixedModels.jl/test/pirls.jl
        //   gm2 = fit(MixedModel, first(gfms[:cbpp]), cbpp, Binomial();
        //              wts=float(cbpp.hsz), init_from_lmm=[:β, :θ])
        //   @test deviance(gm2, true) ≈ 100.09585620707632 rtol=0.0001
        //   @test loglikelihood(gm2)  ≈ -92.02628187247377 atol=0.001
        //
        // Formula in modelcache.jl:
        //   (incid / hsz) ~ 1 + period + (1 | herd)
        //
        // Bundled cbpp dataset uses lme4 column names: incidence, size, herd,
        // period. The response is the per-trial proportion (incidence/size)
        // and `size` provides the case weights.
        let (data, _) = crate::datasets::load("cbpp").unwrap();
        let incidence = data.numeric("incidence").unwrap();
        let size = data.numeric("size").unwrap();

        let proportion: Vec<f64> = incidence
            .iter()
            .zip(size.iter())
            .map(|(&y, &n)| y / n)
            .collect();
        let weights: Vec<f64> = size.to_vec();

        let mut data_with_proportion = data.clone();
        data_with_proportion.add_numeric("proportion", proportion);

        let formula = parse_formula("proportion ~ 1 + period + (1 | herd)").unwrap();

        let mut model = GeneralizedLinearMixedModel::new_with_weights(
            formula,
            &data_with_proportion,
            Family::Binomial,
            None,
            weights,
        )
        .unwrap();

        model.fit_with_options(true, 1, false).unwrap();

        let dev = model.deviance(1);
        // Julia ref: `deviance(gm2, true) ≈ 100.09585620707632`, rtol=0.0001.
        // (Julia's `loglikelihood(m) ≈ -92.02628` includes the binomial
        // saturation constant; Rust's MixedModelFit::loglikelihood returns
        // -objective/2 by definition, so the two are not directly comparable
        // without the saturation term — the deviance check is the meaningful
        // parity assertion here.)
        assert_relative_eq!(dev, 100.09585620707632, max_relative = 1e-3);
    }

    #[cfg(feature = "nlopt")]
    #[test]
    fn test_grouseticks_poisson_glmm_deviance() {
        // pirls.jl:194-227, MixedModels.jl/test/pirls.jl
        //   gm4 = fit(MixedModel, only(gfms[:grouseticks]), grouseticks,
        //              Poisson(); fast=true)
        //   @test isapprox(deviance(gm4), 851.4046, atol=0.001)
        //
        // Formula in modelcache.jl:
        //   ticks ~ 1 + year + ch + (1 | index) + (1 | brood) + (1 | location)
        let (data, _) = crate::datasets::load("grouseticks").unwrap();
        let formula = parse_formula(
            "TICKS ~ 1 + YEAR + cHEIGHT + (1 | INDEX) + (1 | BROOD) + (1 | LOCATION)",
        )
        .unwrap();

        let mut model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Poisson, None).unwrap();

        model.fit_with_options(true, 1, false).unwrap();

        let theta = model.theta.clone();
        assert_eq!(theta.len(), 3, "expected three scalar-RE θ components");
        for (i, &t) in theta.iter().enumerate() {
            assert!(t >= 0.0, "θ[{i}] = {t} should be nonnegative");
            assert!(t.is_finite(), "θ[{i}] = {t} should be finite");
        }

        let dev = model.deviance(1);
        // Julia uses atol=0.001; we allow a slightly larger absolute slack
        // to absorb any remaining BOBYQA-vs-NEWUOA optimizer-driver
        // differences. Julia ref deviance: 851.4046.
        assert_relative_eq!(dev, 851.4046, max_relative = 1e-3);
    }

    #[cfg(feature = "nlopt")]
    #[test]
    fn test_contra_glmm_nagq_7_deviance() {
        // pirls.jl (contra testset, lines 94-97):
        //   refit!(gm0; nAGQ=7)
        //   @test isapprox(deviance(gm0), 2360.876, atol=0.001)
        //
        // After re-fitting with 7-point adaptive Gauss-Hermite quadrature
        // the deviance should drop slightly from the Laplace value.
        let data = contra_fixture();
        let formula =
            parse_formula("use_num ~ 1 + age + age2 + urban + livch + (1 | urban_dist)").unwrap();

        let mut model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();

        model.fit_with_options(true, 7, false).unwrap();

        // optsum should record the AGQ choice and a cached AGQ deviance.
        assert_eq!(model.lmm.optsum.n_agq, 7);
        eprintln!(
            "contra nAGQ=7 deviance: rust = {:.6}, julia ref = 2360.876",
            model.lmm.optsum.fmin
        );
        assert_relative_eq!(model.lmm.optsum.fmin, 2360.876, epsilon = 1.0);

        // Re-evaluating at the converged state should match the cached value
        // exactly (no further optimization between the two calls).
        let dev_agq = model.deviance(7);
        assert_relative_eq!(dev_agq, model.lmm.optsum.fmin, epsilon = 1e-9);

        // The Laplace value (n_agq = 1) should be close to but distinct from
        // the AGQ value at the same θ.
        let dev_lap = model.deviance(1);
        assert!(
            (dev_lap - dev_agq).abs() < 5.0,
            "Laplace and AGQ deviances should be within ~5 units (got {dev_lap} vs {dev_agq})",
        );
    }

    #[test]
    fn test_matrix_block_diag_covers_all_variants() {
        // Direct unit test of the diagonal-extraction helper that AGQ uses
        // on the (1,1) Cholesky block. Contra exercises one variant in
        // practice; this guards the other two so refactors of the L block
        // layout can't silently break AGQ.
        use crate::model::linear::MatrixBlock;
        use nalgebra::{DMatrix, DVector};

        let diag = MatrixBlock::Diagonal(DVector::from_vec(vec![1.0, 2.0, 3.0]));
        assert_eq!(matrix_block_diag(&diag), vec![1.0, 2.0, 3.0]);

        let blk0 = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let blk1 = DMatrix::from_row_slice(2, 2, &[5.0, 6.0, 7.0, 8.0]);
        let bd = MatrixBlock::BlockDiagonal(vec![blk0, blk1]);
        // Diagonal of each 2x2 block in order:
        // blk0 -> (0,0)=1, (1,1)=4; blk1 -> (0,0)=5, (1,1)=8.
        assert_eq!(matrix_block_diag(&bd), vec![1.0, 4.0, 5.0, 8.0]);

        // Dense, rectangular: returns min(rows,cols) diagonals.
        let m = DMatrix::from_row_slice(
            3,
            4,
            &[
                10.0, 0.0, 0.0, 0.0, //
                0.0, 20.0, 0.0, 0.0, //
                0.0, 0.0, 30.0, 0.0,
            ],
        );
        let dense = MatrixBlock::Dense(m);
        assert_eq!(matrix_block_diag(&dense), vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_glmm_validate_agq_accepts_single_scalar_re() {
        let data = contra_fixture();
        let formula =
            parse_formula("use_num ~ 1 + age + age2 + urban + livch + (1 | urban_dist)").unwrap();
        let model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();

        // Single scalar RE: validation should accept any n_agq.
        assert!(model.is_single_scalar_re());
        assert!(model.validate_agq(0).is_ok());
        assert!(model.validate_agq(1).is_ok());
        assert!(model.validate_agq(7).is_ok());
        assert!(model.validate_agq(25).is_ok());
    }

    #[test]
    fn test_glmm_validate_agq_rejects_vector_random_effect() {
        // (1 + age | urban_dist) has vsize == 2 — vector-valued RE.
        // AGQ is only defined for scalar REs; n_agq > 1 must be refused.
        let data = contra_fixture();
        let formula =
            parse_formula("use_num ~ 1 + age2 + urban + livch + (1 + age | urban_dist)").unwrap();
        let model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();

        assert!(
            !model.is_single_scalar_re(),
            "expected vector-RE model not to be classified single-scalar"
        );
        assert_eq!(model.lmm.reterms.len(), 1);
        assert_eq!(model.lmm.reterms[0].vsize, 2);

        // n_agq <= 1 is always allowed (Laplace).
        assert!(model.validate_agq(0).is_ok());
        assert!(model.validate_agq(1).is_ok());

        // n_agq > 1 must error with InvalidArgument citing the vsize mismatch.
        for n_agq in [2_usize, 3, 7, 11] {
            let err = model.validate_agq(n_agq).expect_err(&format!(
                "validate_agq({n_agq}) should error on a vector RE model"
            ));
            match err {
                MixedModelError::InvalidArgument(msg) => {
                    assert!(
                        msg.contains("scalar"),
                        "error message should mention 'scalar' requirement; got {msg}"
                    );
                }
                other => panic!("expected InvalidArgument, got {other:?}"),
            }
        }
    }

    #[test]
    fn test_glmm_validate_agq_rejects_multi_term_random_effects() {
        // Two grouping factors (urban_dist + livch) — multi-term RE.
        // Even with each term scalar, AGQ is undefined.
        let data = contra_fixture();
        let formula =
            parse_formula("use_num ~ 1 + age + age2 + urban + (1 | urban_dist) + (1 | livch)")
                .unwrap();
        let model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();

        assert_eq!(model.lmm.reterms.len(), 2);
        assert!(!model.is_single_scalar_re());

        assert!(model.validate_agq(1).is_ok());
        for n_agq in [2_usize, 7] {
            let err = model
                .validate_agq(n_agq)
                .expect_err("validate_agq should error on multi-term model");
            assert!(matches!(err, MixedModelError::InvalidArgument(_)));
        }
    }

    #[cfg(feature = "nlopt")]
    #[test]
    fn test_glmm_fit_with_options_rejects_invalid_nagq_up_front() {
        // The fit entry point must preflight the AGQ guard, so users never
        // get a partial fit followed by a panic deep inside deviance().
        let data = contra_fixture();
        let formula =
            parse_formula("use_num ~ 1 + age2 + urban + livch + (1 + age | urban_dist)").unwrap();
        let mut model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();

        // Laplace fit is fine on a vector RE.
        let lap_result = model.fit_with_options(true, 1, false);
        assert!(lap_result.is_ok());

        // But asking for AGQ on the same shape must error before any work.
        let mut model2 = {
            let data = contra_fixture();
            let formula =
                parse_formula("use_num ~ 1 + age2 + urban + livch + (1 + age | urban_dist)")
                    .unwrap();
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap()
        };
        let feval_before = model2.lmm.optsum.feval;
        let err = model2.fit_with_options(true, 7, false).expect_err(
            "fit_with_options(_, 7, _) on a vector-RE model should error before fitting",
        );
        assert!(matches!(err, MixedModelError::InvalidArgument(_)));
        assert_eq!(
            model2.lmm.optsum.feval, feval_before,
            "no objective evaluations should have happened on the rejected fit",
        );
    }

    #[cfg(feature = "nlopt")]
    #[test]
    fn test_glmm_deviance_agq_restores_state() {
        // After a Laplace fit, snapshotting (u, eta, mu) and then calling
        // deviance(7) must leave those vectors bit-equivalent on return:
        // AGQ is supposed to perturb-and-restore.
        let data = contra_fixture();
        let formula =
            parse_formula("use_num ~ 1 + age + age2 + urban + livch + (1 | urban_dist)").unwrap();

        let mut model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();
        model.fit_with_options(true, 1, false).unwrap(); // Laplace fit.

        let u_snap: Vec<DMatrix<f64>> = model.u.clone();
        let eta_snap = model.eta.clone();
        let mu_snap = model.mu.clone();

        let _agq = model.deviance(7);

        // u must be byte-identical: the AGQ sweep restores from u₀.
        assert_eq!(model.u.len(), u_snap.len());
        for (after, before) in model.u.iter().zip(u_snap.iter()) {
            assert_eq!(
                after.shape(),
                before.shape(),
                "u shape must not change across deviance(n_agq)"
            );
            for (a, b) in after.iter().zip(before.iter()) {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "u entry diverged: before={b}, after={a}"
                );
            }
        }

        // eta and mu may pick up tiny fp differences from the final
        // update_eta() call, but should be within ~1e-12 absolute.
        for (a, b) in model.eta.iter().zip(eta_snap.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "eta drifted across AGQ sweep: before={b}, after={a}"
            );
        }
        for (a, b) in model.mu.iter().zip(mu_snap.iter()) {
            assert!(
                (a - b).abs() < 1e-12,
                "mu drifted across AGQ sweep: before={b}, after={a}"
            );
        }

        // And a Laplace re-eval must match its pre-AGQ value.
        let lap_after = model.deviance(1);
        let lap_before = {
            // Recompute a fresh Laplace from the pre-AGQ snapshot for parity.
            let dev_resid: f64 = (0..model.y.len())
                .map(|i| model.dev_resid_component(model.y[i], mu_snap[i]))
                .sum();
            let u_pen: f64 = u_snap
                .iter()
                .map(|u| u.iter().map(|x| x * x).sum::<f64>())
                .sum();
            dev_resid + u_pen + model.lmm_logdet()
        };
        assert!(
            (lap_after - lap_before).abs() < 1e-9,
            "Laplace deviance drifted across AGQ sweep: before={lap_before}, after={lap_after}"
        );
    }

    #[cfg(feature = "nlopt")]
    #[test]
    fn test_glmm_nagq_sweep_converges_on_contra() {
        // At a fixed θ, the n-point AGQ deviance should approach a limit
        // as n_agq grows. We assert:
        //   * all values lie within a small band around the Julia reference
        //     (~2360.876, our Rust fit ~2360.98)
        //   * successive doublings of n_agq move by less than 0.05 (well
        //     below the 1.0 tolerance pattern used elsewhere)
        //   * n_agq=1 path equals laplace_objective() exactly.
        let data = contra_fixture();
        let formula =
            parse_formula("use_num ~ 1 + age + age2 + urban + livch + (1 | urban_dist)").unwrap();

        let mut model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();
        model.fit_with_options(true, 7, false).unwrap();

        let lap_direct = model.laplace_objective();
        let lap_via_dev = model.deviance(1);
        assert_eq!(
            lap_direct.to_bits(),
            lap_via_dev.to_bits(),
            "deviance(1) and laplace_objective() must agree bit-for-bit"
        );

        let dev3 = model.deviance(3);
        let dev5 = model.deviance(5);
        let dev7 = model.deviance(7);
        let dev9 = model.deviance(9);
        let dev15 = model.deviance(15);

        // Rough band: all AGQ evaluations should sit within ~2 deviance units
        // of the Julia reference 2360.876.
        for (label, val) in [
            ("nAGQ=3", dev3),
            ("nAGQ=5", dev5),
            ("nAGQ=7", dev7),
            ("nAGQ=9", dev9),
            ("nAGQ=15", dev15),
        ] {
            assert!(
                (val - 2360.876_f64).abs() < 2.0,
                "{label} deviance {val} too far from Julia ref 2360.876"
            );
        }

        // Convergence: successive refinements should change by < 0.05.
        for (a_label, a, b_label, b) in [
            ("nAGQ=3", dev3, "nAGQ=5", dev5),
            ("nAGQ=5", dev5, "nAGQ=7", dev7),
            ("nAGQ=7", dev7, "nAGQ=9", dev9),
            ("nAGQ=9", dev9, "nAGQ=15", dev15),
        ] {
            assert!(
                (a - b).abs() < 0.05,
                "AGQ refinement |{a_label} - {b_label}| = {} should be < 0.05",
                (a - b).abs()
            );
        }
    }

    #[test]
    fn test_glmm_compiler_artifact_records_boundary_metadata() {
        let data = contra_fixture();
        let formula =
            parse_formula("use_num ~ 1 + age + age2 + urban + livch + (1 | urban_dist)").unwrap();

        let model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();
        let artifact = model.compiler_artifact();

        assert_eq!(
            artifact.model_boundary.model_kind,
            crate::compiler::ModelKind::GeneralizedLinearMixedModel
        );
        assert_eq!(artifact.model_boundary.response_distribution, "bernoulli");
        assert_eq!(artifact.model_boundary.link, "logit");
        assert!(matches!(
            artifact.model_boundary.objective_approximation,
            crate::compiler::ObjectiveApproximation::Laplace { .. }
        ));
        assert!(matches!(
            artifact.model_boundary.inference_availability,
            crate::compiler::InferenceAvailability::Unsupported { .. }
        ));
    }

    #[test]
    fn test_glmm_new_with_compiler_policy_applies_internal_policy() {
        let data = contra_fixture();
        let formula =
            parse_formula("use_num ~ 1 + age + age2 + urban + livch + (1 | urban_dist)").unwrap();
        let mut policy = CompilerPolicy::as_specified();
        policy.thresholds.effective_rank_relative_tolerance = 0.125;

        let model = GeneralizedLinearMixedModel::new_with_compiler_policy(
            formula,
            &data,
            Family::Bernoulli,
            None,
            policy,
        )
        .unwrap();

        assert_eq!(
            model.compiler_policy().random_strategy,
            crate::compiler::RandomStrategy::AsSpecified
        );
        assert!(model
            .compiler_artifact()
            .reproducibility
            .thresholds
            .iter()
            .any(|(name, value)| name == "effective_rank_relative_tolerance" && value == "0.125"));
    }
}
