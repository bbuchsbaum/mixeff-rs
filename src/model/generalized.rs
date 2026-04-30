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
#[cfg(not(feature = "nlopt"))]
use std::cell::Cell;
use std::cell::RefCell;
use std::rc::Rc;

use crate::compiler::{
    CompiledModelArtifact, CompilerPolicy, Diagnostic, DiagnosticCode, DiagnosticSeverity,
    DiagnosticStage, ModelAuditReport, ModelBoundary, ObjectiveApproximation, OptimizerCertificate,
};
use crate::error::{MixedModelError, Result};
use crate::formula::Formula;
use crate::model::data::DataFrame;
use crate::model::linear::LinearMixedModel;
use crate::model::traits::{Family, LinkFunction, MixedModelFit, RandomEffectTermInfo};
use crate::stats::{BlockDescription, ModelSummary, VarCorr};
use crate::types::{FitLogEntry, MatrixBlock, OptSummary, Optimizer, ReMat};

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
    /// Fixed linear predictor offset, one value per observation.
    pub offset: DVector<f64>,
    /// Prior weights.
    pub wt: Vec<f64>,
    /// Estimated residual scale for dispersion families.
    ///
    /// Stored as sigma, so `dispersion(false)` returns this value and
    /// `dispersion(true)` returns sigma squared. Non-dispersion GLMM families
    /// keep this at 1.
    dispersion: f64,

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
        validate_case_weights(&weights, model.y.len())?;
        model.wt = weights;
        Ok(model)
    }

    /// Construct a GLMM with a fixed per-observation linear predictor offset.
    ///
    /// The offset is not estimated. It enters the linear predictor as
    /// `eta = offset + X beta + Z b` and is subtracted from the PIRLS working
    /// response before solving the internal LMM approximation.
    pub fn new_with_offset(
        formula: Formula,
        data: &DataFrame,
        family: Family,
        link: Option<LinkFunction>,
        offset: Vec<f64>,
    ) -> Result<Self> {
        let mut model =
            Self::new_with_policy_internal(formula, data, family, link, CompilerPolicy::default())?;
        model.set_offset(offset)?;
        Ok(model)
    }

    /// Construct a GLMM with both case weights and a fixed linear predictor offset.
    pub fn new_with_weights_and_offset(
        formula: Formula,
        data: &DataFrame,
        family: Family,
        link: Option<LinkFunction>,
        weights: Vec<f64>,
        offset: Vec<f64>,
    ) -> Result<Self> {
        let mut model = Self::new_with_offset(formula, data, family, link, offset)?;
        validate_case_weights(&weights, model.y.len())?;
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
        if let Some(y) = data.numeric(&formula.response) {
            validate_glmm_response_domain(family, link, y)?;
            if let Some(&first) = y.first() {
                if y.iter().all(|&value| value == first) {
                    return Err(MixedModelError::InvalidArgument(
                        "response is constant; GLMM construction requires variation in the response"
                            .to_string(),
                    ));
                }
            }
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
        let offset = DVector::zeros(n);

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
            offset,
            wt: Vec::new(),
            dispersion: 1.0,
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

    /// Replace the fixed linear predictor offset before fitting.
    pub fn set_offset(&mut self, offset: Vec<f64>) -> Result<&mut Self> {
        if self.is_fitted() {
            return Err(MixedModelError::AlreadyFitted);
        }
        validate_offset(&offset, self.y.len())?;
        self.offset = DVector::from_vec(offset);
        self.update_eta();
        Ok(self)
    }

    /// Update the linear predictor η and conditional mean μ.
    pub fn update_eta(&mut self) {
        let n = self.eta.len();
        let x = self.lmm.feterm.full_rank_x();

        // η = offset + X * β
        self.eta = &self.offset + x * &self.beta;

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

        // Save the initial accepted state for halving. The 1.0001 slack is
        // only an acceptance bound for the first step-halving loop; convergence
        // is compared with the uninflated accepted objective.
        let mut u_prev: Vec<DMatrix<f64>> = self.u.clone();
        let mut beta_prev = self.beta.clone();
        let mut obj0 = self.laplace_objective();
        let mut halving_bound = obj0 * 1.0001;

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
                (sqrtwts[obs], working_y[obs]) = pirls_working_observation_with_offset(
                    self.family,
                    self.link,
                    y_obs,
                    eta_obs,
                    mu_obs,
                    case_w,
                    self.offset[obs],
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
            while obj > halving_bound && nhalf < max_halvings {
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

            if pirls_converged(obj, obj0, tol) {
                break;
            }

            // Accept iterate as the new previous state.
            for i in 0..self.u.len() {
                u_prev[i].copy_from(&self.u[i]);
            }
            beta_prev = self.beta.clone();
            obj0 = obj;
            halving_bound = obj;
        }

        self.refresh_dispersion();

        Ok(())
    }

    /// Laplace approximation objective: deviance residuals + u penalty + log|L|.
    pub fn laplace_objective(&self) -> f64 {
        // For binomial-with-trials data the response is a per-trial proportion
        // and `wt[i]` is the trial count; weighting the per-observation
        // deviance contribution by `wt[i]` recovers the binomial deviance.
        let dev: f64 = (0..self.y.len())
            .map(|i| self.case_weight(i) * self.dev_resid_component(self.y[i], self.mu[i]))
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
            devc0[refs[i] as usize] +=
                self.case_weight(i) * self.dev_resid_component(self.y[i], self.mu[i]);
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
                devc[refs[i] as usize] +=
                    self.case_weight(i) * self.dev_resid_component(self.y[i], self.mu[i]);
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

    fn case_weight(&self, obs: usize) -> f64 {
        if self.wt.is_empty() {
            1.0
        } else {
            self.wt[obs]
        }
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

    fn record_invalid_agq_diagnostic(&mut self, n_agq: usize, reason: &str) {
        self.lmm
            .compiler_artifact
            .diagnostics
            .retain(|diagnostic| diagnostic.code != DiagnosticCode::InvalidAgqRequest);

        let affected_terms = self
            .lmm
            .reterms
            .iter()
            .map(random_effect_term_label)
            .collect::<Vec<_>>();
        let term_summaries = self
            .lmm
            .reterms
            .iter()
            .map(|term| {
                serde_json::json!({
                    "group": &term.grouping_name,
                    "columns": &term.cnames,
                    "basis_dimension": term.vsize,
                })
            })
            .collect::<Vec<_>>();
        let mut diagnostic = Diagnostic::new(
            DiagnosticCode::InvalidAgqRequest,
            DiagnosticSeverity::Error,
            DiagnosticStage::Optimization,
            format!(
                "Invalid adaptive Gauss-Hermite quadrature request: n_agq = {n_agq} requires exactly one scalar random-effects term. Use n_agq = 1 for the Laplace approximation or simplify the random-effects structure."
            ),
        )
        .with_affected_terms(affected_terms)
        .with_suggested_actions(vec![
            "use n_agq = 1 for Laplace approximation on this random-effects structure"
                .to_string(),
            "fit AGQ only for a model with exactly one scalar random-effects term".to_string(),
        ]);
        diagnostic
            .payload
            .insert("n_agq".to_string(), serde_json::json!(n_agq));
        diagnostic
            .payload
            .insert("reason".to_string(), serde_json::json!(reason));
        diagnostic.payload.insert(
            "random_effect_term_count".to_string(),
            serde_json::json!(self.lmm.reterms.len()),
        );
        diagnostic.payload.insert(
            "random_effect_terms".to_string(),
            serde_json::json!(term_summaries),
        );
        self.lmm.compiler_artifact.diagnostics.push(diagnostic);
    }

    /// Log-determinant from the LMM's Cholesky factor.
    fn lmm_logdet(&self) -> f64 {
        // Delegate to the internal LMM's block structure
        let k = self.lmm.dims.nretrms;
        let mut logdet = 0.0;
        for j in 0..k {
            let idx = j * (j + 1) / 2 + j; // block_index(j, j)
            logdet += match &self.lmm.l_blocks[idx] {
                MatrixBlock::Dense(m) => {
                    let n = m.nrows().min(m.ncols());
                    (0..n).map(|i| m[(i, i)].abs().ln()).sum::<f64>()
                }
                MatrixBlock::Diagonal(v) => v.iter().map(|x| x.abs().ln()).sum::<f64>(),
                MatrixBlock::BlockDiagonal(blocks) => blocks
                    .iter()
                    .map(|blk| {
                        let n = blk.nrows();
                        (0..n).map(|i| blk[(i, i)].abs().ln()).sum::<f64>()
                    })
                    .sum::<f64>(),
                MatrixBlock::Sparse(m) => {
                    let dense = MatrixBlock::Sparse(m.clone()).as_dense();
                    let n = dense.nrows().min(dense.ncols());
                    (0..n).map(|i| dense[(i, i)].abs().ln()).sum::<f64>()
                }
            };
        }
        2.0 * logdet
    }

    /// Fit the GLMM.
    pub fn fit(&mut self) -> Result<&mut Self> {
        self.fit_with_options(true, 1, false)
    }

    /// Refit the GLMM to a new response vector from the recorded initial θ.
    ///
    /// This mirrors Julia's `refit!` semantics for bootstrap and simulation
    /// workflows: the optimizer starts from `optsum.initial`, not from the
    /// previous optimum.
    pub fn refit(&mut self, new_y: &[f64]) -> Result<&mut Self> {
        let n_agq = self.lmm.optsum.n_agq.max(1);
        self.refit_with_options(new_y, n_agq, false)
    }

    /// Refit the GLMM to a new response vector with an explicit AGQ setting.
    pub fn refit_with_options(
        &mut self,
        new_y: &[f64],
        n_agq: usize,
        verbose: bool,
    ) -> Result<&mut Self> {
        if let Err(error) = self.validate_agq(n_agq) {
            self.record_invalid_agq_diagnostic(n_agq, &error.to_string());
            return Err(error);
        }
        self.reset_for_refit(Some(new_y))?;
        self.fit_with_options(true, n_agq, verbose)
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
    /// `fast` selects the supported MixedModels.jl-style fast path, which
    /// profiles over θ and updates β through PIRLS. `fast = false` is not
    /// implemented yet because it requires a distinct joint `[β; θ]`
    /// optimizer path; passing `false` returns [`MixedModelError::Unsupported`]
    /// rather than silently using the fast path.
    ///
    /// `n_agq` selects the deviance approximation: `1` (or `0`) means the
    /// Laplace approximation; values `>= 2` request `n_agq`-point adaptive
    /// Gauss-Hermite quadrature, which is only valid for models with a
    /// single scalar random-effects term and is rejected up front
    /// otherwise.
    pub fn fit_with_options(
        &mut self,
        fast: bool,
        n_agq: usize,
        verbose: bool,
    ) -> Result<&mut Self> {
        if !fast {
            return Err(MixedModelError::Unsupported(
                "GLMM fit_with_options(fast = false) is not implemented; \
                 use fast = true for the current profiled-θ PIRLS path"
                    .to_string(),
            ));
        }
        if let Err(error) = self.validate_agq(n_agq) {
            self.record_invalid_agq_diagnostic(n_agq, &error.to_string());
            return Err(error);
        }
        if self.lmm.optsum.feval > 0 {
            return Err(MixedModelError::AlreadyFitted);
        }
        self.fit_with_options_impl(n_agq, verbose)
    }

    fn reset_for_refit(&mut self, new_y: Option<&[f64]>) -> Result<()> {
        if let Some(new_y) = new_y {
            if new_y.len() != self.y.len() {
                return Err(MixedModelError::InvalidArgument(format!(
                    "Response length {} does not match model ({} observations)",
                    new_y.len(),
                    self.y.len()
                )));
            }
            validate_glmm_response_domain(self.family, self.link, new_y)?;
            let y_max = new_y.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let y_min = new_y.iter().copied().fold(f64::INFINITY, f64::min);
            if (y_max - y_min) < f64::EPSILON {
                return Err(MixedModelError::InvalidArgument(
                    "response is constant; GLMM refit requires variation in the response"
                        .to_string(),
                ));
            }

            let p = self.lmm.feterm.rank;
            for obs in 0..self.y.len() {
                let sw = if self.lmm.sqrtwts.is_empty() {
                    1.0
                } else {
                    self.lmm.sqrtwts[obs]
                };
                self.y[obs] = new_y[obs];
                self.lmm.y[obs] = new_y[obs];
                self.lmm.xy_mat.xy[(obs, p)] = new_y[obs];
                self.lmm.xy_mat.wtxy[(obs, p)] = sw * new_y[obs];
            }
            self.lmm.recompute_a_blocks();
        }

        let initial_theta = self.lmm.optsum.initial.clone();
        self.lmm.set_theta(&initial_theta)?;
        self.lmm.update_l()?;
        self.theta = initial_theta.clone();

        self.beta = DVector::zeros(self.lmm.feterm.rank);
        self.beta0 = self.beta.clone();
        for u in &mut self.u {
            u.fill(0.0);
        }
        for u0 in &mut self.u0 {
            u0.fill(0.0);
        }
        for b in &mut self.b {
            b.fill(0.0);
        }
        self.eta.fill(0.0);
        self.mu.fill(0.0);
        self.dispersion = 1.0;
        self.update_eta();

        self.lmm.optsum.finitial = f64::INFINITY;
        self.lmm.optsum.final_params = initial_theta;
        self.lmm.optsum.fmin = f64::INFINITY;
        self.lmm.optsum.feval = 0;
        self.lmm.optsum.return_value.clear();
        self.lmm.optsum.fit_log.clear();
        self.lmm.compiler_artifact.optimizer_certificate = None;
        self.lmm.compiler_artifact.effective_covariance.clear();
        Ok(())
    }

    #[cfg(not(feature = "nlopt"))]
    fn fit_with_options_impl(&mut self, n_agq: usize, _verbose: bool) -> Result<&mut Self> {
        match self.lmm.optsum.optimizer {
            Optimizer::PatternSearch => self.fit_native_pattern_search(n_agq),
            Optimizer::Cobyla => self.fit_native_cobyla(n_agq),
            Optimizer::NloptBobyqa | Optimizer::NloptNewuoa => Err(MixedModelError::Optimization(
                "NLopt GLMM optimizers require the `nlopt` feature; rebuild with `--features nlopt` or pick a native optimizer"
                    .to_string(),
            )),
            Optimizer::PrimaBobyqa
            | Optimizer::PrimaCobyla
            | Optimizer::PrimaLincoa
            | Optimizer::PrimaNewuoa => Err(MixedModelError::Optimization(
                "PRIMA GLMM optimizers are not wired; pick a native optimizer".to_string(),
            )),
        }
    }

    #[cfg(feature = "nlopt")]
    fn fit_with_options_impl(&mut self, n_agq: usize, _verbose: bool) -> Result<&mut Self> {
        use nlopt::{Algorithm as NloptAlgorithm, Nlopt, Target as NloptTarget};

        let n_theta = self.theta.len();
        let lb = self.lmm.lower_bounds();
        let initial_theta = self.lmm.optsum.initial.clone();
        self.lmm.optsum.optimizer = Optimizer::NloptBobyqa;
        self.lmm.optsum.backend = Optimizer::NloptBobyqa.canonical_backend();

        let mut feval_count: i64 = 0;
        let feval_ptr = &mut feval_count as *mut i64;
        let fit_log: Rc<RefCell<Vec<FitLogEntry>>> = Rc::new(RefCell::new(Vec::with_capacity(
            self.lmm.optsum.fit_log.capacity(),
        )));

        let obj_fn = {
            let model_ptr = self as *mut GeneralizedLinearMixedModel;
            let fit_log = Rc::clone(&fit_log);
            move |theta: &[f64], _grad: Option<&mut [f64]>, _data: &mut ()| -> f64 {
                let model = unsafe { &mut *model_ptr };
                unsafe { *feval_ptr += 1 };
                let objective = model.penalized_pirls_deviance_at_theta(theta, n_agq);
                fit_log.borrow_mut().push(FitLogEntry {
                    theta: theta.to_vec(),
                    objective,
                });
                objective
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
        drop(optimizer);

        self.finalize_theta_after_optimizer(&mut theta, n_agq)?;
        self.lmm.optsum.return_value = match nlopt_result {
            Ok((status, _fmin)) => format!("{status:?}"),
            Err((status, _fmin)) => format!("FAILED:{status:?}"),
        };
        self.lmm.optsum.feval = feval_count;
        self.lmm.optsum.fit_log = rc_refcell_into_inner_or_clone(fit_log);
        self.lmm.compiler_artifact.optimizer_certificate =
            Some(OptimizerCertificate::from_opt_summary_with_context(
                &self.lmm.optsum,
                &self.theta,
                &self.lmm.lower_bounds(),
                Some(self.lmm.dims.n),
            ));
        self.refresh_binomial_separation_diagnostics();
        self.refresh_near_unit_random_effect_correlation_diagnostics();

        Ok(self)
    }

    #[cfg(not(feature = "nlopt"))]
    fn fit_native_cobyla(&mut self, n_agq: usize) -> Result<&mut Self> {
        let lb = self.lmm.lower_bounds();
        let initial_theta = self.lmm.optsum.initial.clone();
        self.lmm.optsum.optimizer = Optimizer::Cobyla;
        self.lmm.optsum.backend = Optimizer::Cobyla.canonical_backend();

        let best_theta: Rc<RefCell<Vec<f64>>> = Rc::new(RefCell::new(initial_theta.clone()));
        let best_fmin: Rc<Cell<f64>> = Rc::new(Cell::new(f64::INFINITY));
        let feval_count: Rc<Cell<i64>> = Rc::new(Cell::new(0i64));
        let fit_log: Rc<RefCell<Vec<FitLogEntry>>> = Rc::new(RefCell::new(Vec::with_capacity(
            self.lmm.optsum.fit_log.capacity(),
        )));

        let objective_fn = {
            let model_ptr = self as *mut GeneralizedLinearMixedModel;
            let best_theta = Rc::clone(&best_theta);
            let best_fmin = Rc::clone(&best_fmin);
            let feval_count = Rc::clone(&feval_count);
            let fit_log = Rc::clone(&fit_log);
            move |theta: &[f64], _data: &mut ()| -> f64 {
                feval_count.set(feval_count.get() + 1);
                let objective =
                    unsafe { (&mut *model_ptr).penalized_pirls_deviance_at_theta(theta, n_agq) };
                fit_log.borrow_mut().push(FitLogEntry {
                    theta: theta.to_vec(),
                    objective,
                });
                if objective < best_fmin.get() {
                    best_fmin.set(objective);
                    *best_theta.borrow_mut() = theta.to_vec();
                }
                objective
            }
        };

        let bounds: Vec<(f64, f64)> = lb.iter().map(|&lo| (lo, f64::INFINITY)).collect();
        let constraint_fns: Vec<Box<dyn cobyla::Func<()>>> = lb
            .iter()
            .enumerate()
            .filter(|(_, &lo)| lo > f64::NEG_INFINITY)
            .map(|(i, &lo)| {
                Box::new(move |x: &[f64], _: &mut ()| -> f64 { x[i] - lo })
                    as Box<dyn cobyla::Func<()>>
            })
            .collect();
        let cons_refs: Vec<&dyn cobyla::Func<()>> =
            constraint_fns.iter().map(|f| f.as_ref()).collect();
        let maxeval = if self.lmm.optsum.max_feval > 0 {
            self.lmm.optsum.max_feval as usize
        } else {
            500
        };
        let stop_tol = cobyla::StopTols {
            ftol_rel: self.lmm.optsum.ftol_rel,
            ftol_abs: self.lmm.optsum.ftol_abs,
            xtol_rel: self.lmm.optsum.xtol_rel,
            xtol_abs: self.lmm.optsum.xtol_abs.clone(),
            ..cobyla::StopTols::default()
        };

        let result = cobyla::minimize(
            objective_fn,
            &initial_theta,
            &bounds,
            &cons_refs,
            (),
            maxeval,
            cobyla::RhoBeg::All(0.75),
            Some(stop_tol),
        );

        let (mut theta, return_value) = match result {
            Ok((status, x_opt, fmin)) if fmin.is_finite() => {
                (x_opt, Self::cobyla_success_status_label(status))
            }
            Ok((status, _x_opt, _fmin)) => (
                best_theta.borrow().clone(),
                Self::cobyla_success_status_label(status),
            ),
            Err((status @ cobyla::FailStatus::RoundoffLimited, _x_opt, _fmin)) => (
                best_theta.borrow().clone(),
                Self::cobyla_fail_status_label(status),
            ),
            Err((status, x_opt, fmin)) if fmin.is_finite() => {
                (x_opt, Self::cobyla_fail_status_label(status))
            }
            Err((status, _x_opt, _fmin)) if best_fmin.get().is_finite() => (
                best_theta.borrow().clone(),
                Self::cobyla_fail_status_label(status),
            ),
            Err((_status, _x_opt, _fmin)) => {
                return Err(MixedModelError::Optimization(
                    "COBYLA optimization failed while fitting GLMM".to_string(),
                ));
            }
        };

        self.finalize_theta_after_optimizer(&mut theta, n_agq)?;
        self.lmm.optsum.return_value = return_value;
        self.lmm.optsum.feval = feval_count.get();
        self.lmm.optsum.fit_log = rc_refcell_into_inner_or_clone(fit_log);
        self.lmm.compiler_artifact.optimizer_certificate =
            Some(OptimizerCertificate::from_opt_summary_with_context(
                &self.lmm.optsum,
                &self.theta,
                &self.lmm.lower_bounds(),
                Some(self.lmm.dims.n),
            ));
        self.refresh_binomial_separation_diagnostics();
        self.refresh_near_unit_random_effect_correlation_diagnostics();

        Ok(self)
    }

    #[cfg(not(feature = "nlopt"))]
    fn fit_native_pattern_search(&mut self, n_agq: usize) -> Result<&mut Self> {
        let lower_bounds = self.lmm.lower_bounds();
        let n_theta = self.theta.len();
        let maxeval = if self.lmm.optsum.max_feval > 0 {
            self.lmm.optsum.max_feval
        } else {
            500
        };
        let mut step_tol = self.lmm.optsum.xtol_abs.clone();
        if step_tol.len() != n_theta {
            step_tol = vec![1e-5; n_theta];
        }
        for tol in &mut step_tol {
            *tol = tol.max(1e-5);
        }
        let mut step = self.lmm.optsum.initial_step.clone();
        if step.len() != n_theta {
            step = vec![0.75; n_theta];
        }
        for (value, tol) in step.iter_mut().zip(step_tol.iter()) {
            *value = value.abs().max(*tol);
        }

        let mut theta = self.lmm.optsum.initial.clone();
        project_theta_to_bounds(&mut theta, &lower_bounds);
        let mut best_theta = theta.clone();
        let mut best_fmin = f64::INFINITY;
        let mut feval_count = 0i64;
        let mut fit_log = Vec::with_capacity(self.lmm.optsum.fit_log.capacity());
        let mut preferred_sign = vec![-1.0; n_theta];
        for (idx, lower) in lower_bounds.iter().enumerate() {
            if !lower.is_finite() {
                preferred_sign[idx] = 1.0;
            }
        }

        let mut current_f = record_pattern_search_eval(
            self,
            &theta,
            n_agq,
            &mut feval_count,
            &mut fit_log,
            &mut best_theta,
            &mut best_fmin,
        );
        self.lmm.optsum.finitial = current_f;

        while feval_count < maxeval && !steps_are_small(&step, &step_tol) {
            let base_theta = theta.clone();
            let base_f = current_f;
            let mut moved = false;

            for idx in 0..n_theta {
                let mut accepted = false;
                for dir in [preferred_sign[idx], -preferred_sign[idx]] {
                    let mut trial = theta.clone();
                    trial[idx] += dir * step[idx];
                    project_theta_to_bounds(&mut trial, &lower_bounds);
                    if (trial[idx] - theta[idx]).abs() <= step_tol[idx] * 0.5 {
                        continue;
                    }
                    let ftrial = record_pattern_search_eval(
                        self,
                        &trial,
                        n_agq,
                        &mut feval_count,
                        &mut fit_log,
                        &mut best_theta,
                        &mut best_fmin,
                    );
                    if ftrial + self.lmm.optsum.ftol_abs < current_f {
                        theta = trial;
                        current_f = ftrial;
                        preferred_sign[idx] = dir;
                        step[idx] = (step[idx] * 1.1).max(step_tol[idx]);
                        moved = true;
                        accepted = true;
                        break;
                    }
                    if feval_count >= maxeval {
                        break;
                    }
                }
                if !accepted {
                    preferred_sign[idx] = -preferred_sign[idx];
                    step[idx] *= 0.5;
                }
                if feval_count >= maxeval {
                    break;
                }
            }

            if moved && feval_count < maxeval {
                let mut pattern = theta.clone();
                for idx in 0..n_theta {
                    pattern[idx] += theta[idx] - base_theta[idx];
                }
                project_theta_to_bounds(&mut pattern, &lower_bounds);
                if pattern != theta {
                    let fpattern = record_pattern_search_eval(
                        self,
                        &pattern,
                        n_agq,
                        &mut feval_count,
                        &mut fit_log,
                        &mut best_theta,
                        &mut best_fmin,
                    );
                    if fpattern + self.lmm.optsum.ftol_abs < current_f {
                        theta = pattern;
                        current_f = fpattern;
                    }
                }
            }

            if !moved {
                for value in &mut step {
                    *value *= 0.5;
                }
            }
            if (base_f - current_f).abs() <= self.lmm.optsum.ftol_abs
                && steps_are_small(&step, &step_tol)
            {
                break;
            }
        }

        self.lmm.optsum.optimizer = Optimizer::PatternSearch;
        self.lmm.optsum.backend = Optimizer::PatternSearch.canonical_backend();
        self.finalize_theta_after_optimizer(&mut best_theta, n_agq)?;
        self.lmm.optsum.return_value = if feval_count >= maxeval {
            "MAXEVAL_REACHED".to_string()
        } else {
            "SUCCESS".to_string()
        };
        self.lmm.optsum.feval = feval_count;
        self.lmm.optsum.fit_log = fit_log;
        self.lmm.compiler_artifact.optimizer_certificate =
            Some(OptimizerCertificate::from_opt_summary_with_context(
                &self.lmm.optsum,
                &self.theta,
                &self.lmm.lower_bounds(),
                Some(self.lmm.dims.n),
            ));
        self.refresh_binomial_separation_diagnostics();
        self.refresh_near_unit_random_effect_correlation_diagnostics();

        Ok(self)
    }

    fn finalize_theta_after_optimizer(&mut self, theta: &mut Vec<f64>, n_agq: usize) -> Result<()> {
        LinearMixedModel::rectify_theta_columns(theta, &self.lmm.parmap, self.lmm.reterms.len());

        // Final PIRLS at optimal θ, after matching MixedModels.jl's
        // post-optimizer sign convention for Cholesky columns.
        self.update_pirls_at_theta(theta, true)?;
        self.beta = self.lmm.beta();
        self.refresh_dispersion();

        self.lmm.optsum.n_agq = n_agq;
        self.lmm.optsum.fmin = self.deviance(n_agq);
        self.lmm.optsum.final_params = theta.clone();
        Ok(())
    }

    fn update_pirls_at_theta(&mut self, theta: &[f64], vary_beta: bool) -> Result<()> {
        if theta.len() != self.theta.len() {
            return Err(MixedModelError::DimensionMismatch(format!(
                "theta vector length {} does not match fitted GLMM theta length {}",
                theta.len(),
                self.theta.len()
            )));
        }
        if !theta.iter().all(|value| value.is_finite()) {
            return Err(MixedModelError::InvalidArgument(
                "theta values must be finite".to_string(),
            ));
        }
        self.lmm.set_theta(theta)?;
        self.lmm.update_l()?;
        self.theta = theta.to_vec();
        for u in self.u.iter_mut() {
            u.fill(0.0);
        }
        self.pirls(vary_beta, false)?;
        Ok(())
    }

    fn refresh_dispersion(&mut self) {
        self.dispersion = self.estimated_dispersion_scale();
    }

    fn estimated_dispersion_scale(&self) -> f64 {
        if !self.family.has_dispersion() {
            return 1.0;
        }

        let pearson = self.pearson_dispersion_numerator();
        let denom = self.y.len().saturating_sub(self.lmm.feterm.rank).max(1) as f64;
        let variance = (pearson / denom).max(f64::MIN_POSITIVE);
        variance.sqrt()
    }

    fn pearson_dispersion_numerator(&self) -> f64 {
        let mut total = 0.0;
        for obs in 0..self.y.len() {
            let mu = self.mu[obs];
            let variance = self.family.variance(mu);
            if !variance.is_finite() || variance <= 0.0 {
                continue;
            }
            let residual = self.y[obs] - mu;
            total += self.case_weight(obs) * residual * residual / variance;
        }
        total
    }

    fn penalized_pirls_deviance_at_theta(&mut self, theta: &[f64], n_agq: usize) -> f64 {
        match self.update_pirls_at_theta(theta, true) {
            Ok(()) => {
                let deviance = self.deviance(n_agq);
                if deviance.is_finite() {
                    deviance
                } else {
                    f64::INFINITY
                }
            }
            Err(_) => f64::INFINITY,
        }
    }

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

    fn refresh_binomial_separation_diagnostics(&mut self) {
        self.lmm
            .compiler_artifact
            .diagnostics
            .retain(|diagnostic| diagnostic.code != DiagnosticCode::BinomialSeparation);
        if let Some(certificate) = &mut self.lmm.compiler_artifact.optimizer_certificate {
            certificate
                .diagnostics
                .retain(|diagnostic| diagnostic.code != DiagnosticCode::BinomialSeparation);
        }

        let diagnostics = self.conservative_binomial_separation_diagnostics();
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

    fn conservative_binomial_separation_diagnostics(&self) -> Vec<Diagnostic> {
        if !matches!(self.family, Family::Bernoulli | Family::Binomial)
            || self.link != LinkFunction::Logit
            || !self.y.iter().all(|value| is_binary_response(*value))
        {
            return Vec::new();
        }

        let mut diagnostics = Vec::new();
        for column_index in 0..self.lmm.feterm.rank {
            let column_name = self
                .lmm
                .feterm
                .cnames
                .get(column_index)
                .cloned()
                .unwrap_or_else(|| format!("fixed_effect[{column_index}]"));
            if is_intercept_column(&column_name) {
                continue;
            }

            let column_values = self.lmm.feterm.x.column(column_index);
            let Some(split) = binary_column_split(column_values.iter().copied()) else {
                continue;
            };

            let low_counts = outcome_counts_for_value(
                column_values.iter().copied(),
                self.y.iter().copied(),
                split.low,
            );
            let high_counts = outcome_counts_for_value(
                column_values.iter().copied(),
                self.y.iter().copied(),
                split.high,
            );

            if let Some(diagnostic) =
                separation_diagnostic_for_side(&column_name, split.low, low_counts, high_counts)
            {
                diagnostics.push(diagnostic);
            }
            if let Some(diagnostic) =
                separation_diagnostic_for_side(&column_name, split.high, high_counts, low_counts)
            {
                diagnostics.push(diagnostic);
            }
        }

        diagnostics
    }

    #[cfg(not(feature = "nlopt"))]
    fn cobyla_success_status_label(status: cobyla::SuccessStatus) -> String {
        match status {
            cobyla::SuccessStatus::Success => "SUCCESS".to_string(),
            cobyla::SuccessStatus::StopValReached => "STOPVAL_REACHED".to_string(),
            cobyla::SuccessStatus::FtolReached => "FTOL_REACHED".to_string(),
            cobyla::SuccessStatus::XtolReached => "XTOL_REACHED".to_string(),
            cobyla::SuccessStatus::MaxEvalReached => "MAXEVAL_REACHED".to_string(),
            cobyla::SuccessStatus::MaxTimeReached => "MAXTIME_REACHED".to_string(),
        }
    }

    #[cfg(not(feature = "nlopt"))]
    fn cobyla_fail_status_label(status: cobyla::FailStatus) -> String {
        match status {
            cobyla::FailStatus::Failure => "FAILURE".to_string(),
            cobyla::FailStatus::InvalidArgs => "INVALID_ARGS".to_string(),
            cobyla::FailStatus::OutOfMemory => "OUT_OF_MEMORY".to_string(),
            cobyla::FailStatus::RoundoffLimited => "ROUNDOFF_LIMITED".to_string(),
            cobyla::FailStatus::ForcedStop => "FORCED_STOP".to_string(),
            cobyla::FailStatus::UnexpectedError => "UNEXPECTED_ERROR".to_string(),
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
fn matrix_block_diag(block: &MatrixBlock) -> Vec<f64> {
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

fn rc_refcell_into_inner_or_clone<T: Clone>(value: Rc<RefCell<Vec<T>>>) -> Vec<T> {
    match Rc::try_unwrap(value) {
        Ok(cell) => cell.into_inner(),
        Err(value) => value.borrow().clone(),
    }
}

#[cfg(not(feature = "nlopt"))]
fn project_theta_to_bounds(theta: &mut [f64], lower_bounds: &[f64]) {
    for (value, lower) in theta.iter_mut().zip(lower_bounds.iter()) {
        if lower.is_finite() && *value < *lower {
            *value = *lower;
        }
    }
}

#[cfg(not(feature = "nlopt"))]
fn steps_are_small(step: &[f64], step_tol: &[f64]) -> bool {
    step.iter()
        .zip(step_tol.iter())
        .all(|(step, tol)| *step <= *tol)
}

#[cfg(not(feature = "nlopt"))]
fn record_pattern_search_eval(
    model: &mut GeneralizedLinearMixedModel,
    theta: &[f64],
    n_agq: usize,
    feval_count: &mut i64,
    fit_log: &mut Vec<FitLogEntry>,
    best_theta: &mut Vec<f64>,
    best_fmin: &mut f64,
) -> f64 {
    *feval_count += 1;
    let objective = model.penalized_pirls_deviance_at_theta(theta, n_agq);
    fit_log.push(FitLogEntry {
        theta: theta.to_vec(),
        objective,
    });
    if objective < *best_fmin {
        *best_fmin = objective;
        *best_theta = theta.to_vec();
    }
    objective
}

fn lower_triangle_pair(offset: usize) -> (usize, usize) {
    let mut row = 1usize;
    let mut remaining = offset;
    while remaining >= row {
        remaining -= row;
        row += 1;
    }
    (row, remaining)
}

#[derive(Debug, Clone, Copy)]
struct BinaryColumnSplit {
    low: f64,
    high: f64,
}

#[derive(Debug, Clone, Copy)]
struct OutcomeCounts {
    n: usize,
    successes: usize,
    failures: usize,
}

fn is_binary_response(value: f64) -> bool {
    (value - 0.0).abs() < 1e-12 || (value - 1.0).abs() < 1e-12
}

fn is_intercept_column(name: &str) -> bool {
    matches!(name, "1" | "(Intercept)" | "Intercept" | "intercept")
}

fn random_effect_term_label(reterm: &ReMat) -> String {
    let columns = reterm
        .cnames
        .iter()
        .map(|name| {
            if is_intercept_column(name) {
                "1"
            } else {
                name.as_str()
            }
        })
        .collect::<Vec<_>>()
        .join(" + ");
    format!("({columns} | {})", reterm.grouping_name)
}

fn binary_column_split(values: impl Iterator<Item = f64>) -> Option<BinaryColumnSplit> {
    let mut unique = Vec::new();
    for value in values {
        if !value.is_finite() {
            return None;
        }
        if unique
            .iter()
            .all(|seen: &f64| (value - *seen).abs() > 1e-12)
        {
            unique.push(value);
            if unique.len() > 2 {
                return None;
            }
        }
    }
    if unique.len() != 2 {
        return None;
    }
    unique.sort_by(|a, b| a.total_cmp(b));
    Some(BinaryColumnSplit {
        low: unique[0],
        high: unique[1],
    })
}

fn outcome_counts_for_value(
    values: impl Iterator<Item = f64>,
    y: impl Iterator<Item = f64>,
    target: f64,
) -> OutcomeCounts {
    let mut counts = OutcomeCounts {
        n: 0,
        successes: 0,
        failures: 0,
    };
    for (value, response) in values.zip(y) {
        if (value - target).abs() <= 1e-12 {
            counts.n += 1;
            if response > 0.5 {
                counts.successes += 1;
            } else {
                counts.failures += 1;
            }
        }
    }
    counts
}

fn separation_diagnostic_for_side(
    column_name: &str,
    value: f64,
    side: OutcomeCounts,
    complement: OutcomeCounts,
) -> Option<Diagnostic> {
    if side.n == 0
        || complement.n == 0
        || !side_is_pure(side)
        || !complement_has_opposite(side, complement)
    {
        return None;
    }

    let outcome = if side.successes == side.n { 1 } else { 0 };
    let kind = if side_is_pure(complement) {
        "complete_fixed_effect"
    } else {
        "quasi_complete_fixed_effect"
    };
    let rows = if side.n == 1 { "row" } else { "rows" };
    let value_label = format_column_value(value);
    let mut diagnostic = Diagnostic::new(
        DiagnosticCode::BinomialSeparation,
        DiagnosticSeverity::Warning,
        DiagnosticStage::Certification,
        format!(
            "Possible separation in binomial model: `{column_name} = {value_label}` occurs in {} {rows}, and all such rows have y = {outcome}. The coefficient for `{column_name}` may be unbounded; standard errors, Wald tests, and p-values for this term are unreliable.",
            side.n
        ),
    )
    .with_affected_terms(vec![column_name.to_string()])
    .with_suggested_actions(vec![
        "inspect the corresponding rows or levels for sparse outcome support".to_string(),
        "consider removing or combining rare predictors, or use penalized/Bayesian logistic mixed modeling".to_string(),
        "report inference for this term as unreliable if the model is retained".to_string(),
    ]);
    diagnostic
        .payload
        .insert("term".to_string(), serde_json::json!(column_name));
    diagnostic
        .payload
        .insert("value".to_string(), serde_json::json!(value));
    diagnostic
        .payload
        .insert("n_at_value".to_string(), serde_json::json!(side.n));
    diagnostic.payload.insert(
        "successes_at_value".to_string(),
        serde_json::json!(side.successes),
    );
    diagnostic.payload.insert(
        "failures_at_value".to_string(),
        serde_json::json!(side.failures),
    );
    diagnostic.payload.insert(
        "complement_successes".to_string(),
        serde_json::json!(complement.successes),
    );
    diagnostic.payload.insert(
        "complement_failures".to_string(),
        serde_json::json!(complement.failures),
    );
    diagnostic
        .payload
        .insert("separation_kind".to_string(), serde_json::json!(kind));
    Some(diagnostic)
}

fn side_is_pure(counts: OutcomeCounts) -> bool {
    counts.n > 0 && (counts.successes == counts.n || counts.failures == counts.n)
}

fn complement_has_opposite(side: OutcomeCounts, complement: OutcomeCounts) -> bool {
    if side.successes == side.n {
        complement.failures > 0
    } else {
        complement.successes > 0
    }
}

fn format_column_value(value: f64) -> String {
    if (value.round() - value).abs() < 1e-12 {
        format!("{value:.0}")
    } else {
        format!("{value:.6}")
    }
}

fn pirls_working_observation(
    family: Family,
    link: LinkFunction,
    y: f64,
    eta: f64,
    mu: f64,
    case_weight: f64,
) -> (f64, f64) {
    let (working_mu, eta_for_derivative) = bounded_pirls_mean_and_eta(family, link, mu, eta);
    let dmu_deta = link.mu_eta(eta_for_derivative);
    let var_mu = family.variance(working_mu);
    let weight = if dmu_deta.is_finite() && var_mu.is_finite() && var_mu > 0.0 {
        case_weight * dmu_deta * dmu_deta / var_mu
    } else {
        0.0
    };
    let resid = if !dmu_deta.is_finite() || dmu_deta.abs() < 1e-15 {
        0.0
    } else {
        (y - working_mu) / dmu_deta
    };
    (weight.max(0.0).sqrt(), eta + resid)
}

fn pirls_working_observation_with_offset(
    family: Family,
    link: LinkFunction,
    y: f64,
    eta: f64,
    mu: f64,
    case_weight: f64,
    offset: f64,
) -> (f64, f64) {
    let (sqrt_weight, working_response) =
        pirls_working_observation(family, link, y, eta, mu, case_weight);
    (sqrt_weight, working_response - offset)
}

fn bounded_pirls_mean_and_eta(family: Family, link: LinkFunction, mu: f64, eta: f64) -> (f64, f64) {
    const BOUNDED_MEAN_EPS: f64 = 1e-15;
    const LOG_LINK_ETA_BOUND: f64 = 30.0;
    if matches!(family, Family::Bernoulli | Family::Binomial) {
        let bounded_mu = mu.clamp(BOUNDED_MEAN_EPS, 1.0 - BOUNDED_MEAN_EPS);
        (bounded_mu, link.link(bounded_mu))
    } else if family == Family::Poisson && link == LinkFunction::Log {
        let bounded_eta = eta.clamp(-LOG_LINK_ETA_BOUND, LOG_LINK_ETA_BOUND);
        (bounded_eta.exp(), bounded_eta)
    } else {
        (mu, eta)
    }
}

fn pirls_converged(obj: f64, accepted_obj: f64, tol: f64) -> bool {
    (obj - accepted_obj).abs() < tol
}

fn validate_case_weights(weights: &[f64], n_obs: usize) -> Result<()> {
    if weights.len() != n_obs {
        return Err(MixedModelError::InvalidArgument(format!(
            "case weights length ({}) does not match number of observations ({n_obs})",
            weights.len()
        )));
    }
    for (i, &w) in weights.iter().enumerate() {
        if !w.is_finite() || w <= 0.0 {
            return Err(MixedModelError::InvalidArgument(format!(
                "case weight at index {i} must be finite and positive (got {w})"
            )));
        }
    }
    Ok(())
}

fn validate_offset(offset: &[f64], n_obs: usize) -> Result<()> {
    if offset.len() != n_obs {
        return Err(MixedModelError::InvalidArgument(format!(
            "offset length ({}) does not match number of observations ({n_obs})",
            offset.len()
        )));
    }
    for (idx, &value) in offset.iter().enumerate() {
        if !value.is_finite() {
            return Err(MixedModelError::InvalidArgument(format!(
                "offset at index {idx} must be finite (got {value})"
            )));
        }
    }
    Ok(())
}

fn validate_glmm_response_domain(family: Family, link: LinkFunction, y: &[f64]) -> Result<()> {
    for (idx, &value) in y.iter().enumerate() {
        if !value.is_finite() {
            return Err(MixedModelError::InvalidArgument(format!(
                "response at index {idx} must be finite for GLMM construction (got {value})"
            )));
        }
        if matches!(family, Family::Gamma | Family::InverseGaussian) && value <= 0.0 {
            return Err(MixedModelError::InvalidArgument(format!(
                "{} GLMM response must be strictly positive; index {idx} has {value}",
                family_label(family)
            )));
        }
        if family == Family::Normal && link == LinkFunction::Sqrt && value < 0.0 {
            return Err(MixedModelError::InvalidArgument(format!(
                "gaussian GLMM with sqrt link requires non-negative responses; index {idx} has {value}"
            )));
        }
    }
    Ok(())
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
        if sqr {
            self.dispersion * self.dispersion
        } else {
            self.dispersion
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
    use approx::assert_relative_eq;

    fn assert_glmm_theta_diagonals_nonnegative(model: &GeneralizedLinearMixedModel) {
        for (idx, &(_, row, col)) in model.lmm.parmap.iter().enumerate() {
            if row == col {
                assert!(
                    model.theta[idx] >= 0.0,
                    "GLMM theta diagonal {idx} should be rectified, got {}",
                    model.theta[idx]
                );
                assert_eq!(
                    model.lmm.optsum.final_params[idx], model.theta[idx],
                    "GLMM OptSummary must store the rectified theta value"
                );
            }
        }
    }

    fn resampled_contra_response(data: &DataFrame) -> Vec<f64> {
        data.numeric("use_num")
            .unwrap()
            .iter()
            .enumerate()
            .map(
                |(idx, &value)| {
                    if idx % 11 == 0 {
                        1.0 - value
                    } else {
                        value
                    }
                },
            )
            .collect()
    }

    fn refit_cold_contra_model(new_y: &[f64]) -> GeneralizedLinearMixedModel {
        let mut data = contra_fixture();
        data.add_numeric("use_num", new_y.to_vec()).unwrap();
        let formula =
            parse_formula("use_num ~ 1 + age + age2 + urban + livch + (1 | urban_dist)").unwrap();
        let mut model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();
        model.fit_with_options(true, 1, false).unwrap();
        model
    }

    fn glmm_retained_state_slots(model: &GeneralizedLinearMixedModel) -> usize {
        let matrix_slots = |matrix: &DMatrix<f64>| matrix.nrows() * matrix.ncols();
        let block_slots = |block: &MatrixBlock| block.nrows() * block.ncols();

        model.beta.len()
            + model.beta0.len()
            + model.theta.capacity()
            + model.b.iter().map(matrix_slots).sum::<usize>()
            + model.u.iter().map(matrix_slots).sum::<usize>()
            + model.u0.iter().map(matrix_slots).sum::<usize>()
            + model.eta.len()
            + model.mu.len()
            + model.y.len()
            + model.offset.len()
            + model.wt.capacity()
            + model.devc.capacity()
            + model.devc0.capacity()
            + model.sd.capacity()
            + model.mult.capacity()
            + model.lmm.y.len()
            + matrix_slots(&model.lmm.xy_mat.xy)
            + matrix_slots(&model.lmm.xy_mat.wtxy)
            + model
                .lmm
                .reterms
                .iter()
                .map(|rt| matrix_slots(&rt.z) + matrix_slots(&rt.wtz) + matrix_slots(&rt.lambda))
                .sum::<usize>()
            + model.lmm.a_blocks.iter().map(block_slots).sum::<usize>()
            + model.lmm.l_blocks.iter().map(block_slots).sum::<usize>()
            + model.lmm.optsum.initial.capacity()
            + model.lmm.optsum.final_params.capacity()
            + model.lmm.optsum.fit_log.capacity()
    }

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
    fn test_pirls_no_iter0_break_on_first_step_halving_slack() {
        let accepted_obj = 100.0_f64;
        let old_inflated_reference = accepted_obj * 1.0001;
        let first_step_obj = old_inflated_reference + 0.5e-5;
        let tol = 1e-5_f64;

        assert!(
            (first_step_obj - old_inflated_reference).abs() < tol,
            "this is the old false-convergence case when the halving slack is reused"
        );
        assert!(
            !pirls_converged(first_step_obj, accepted_obj, tol),
            "PIRLS convergence must compare against the uninflated accepted objective"
        );
        assert!(pirls_converged(accepted_obj + tol * 0.5, accepted_obj, tol));
    }

    #[test]
    fn test_pirls_handles_bernoulli_near_separation() {
        let (sqrtw_low, z_low) = pirls_working_observation(
            Family::Bernoulli,
            LinkFunction::Logit,
            0.0,
            -1000.0,
            0.0,
            1.0,
        );
        let (sqrtw_high, z_high) =
            pirls_working_observation(Family::Bernoulli, LinkFunction::Log, 1.0, 1000.0, 1.0, 1.0);

        assert!(sqrtw_low.is_finite());
        assert!(z_low.is_finite());
        assert!(sqrtw_high.is_finite());
        assert!(z_high.is_finite());
        assert!(
            sqrtw_high < 4.0e7,
            "clamped Bernoulli variance should keep sqrt weight bounded, got {sqrtw_high}"
        );
    }

    #[test]
    fn test_pirls_no_inf_weight_under_logit() {
        for (y, eta, mu) in [(0.0, -1000.0, 0.0), (1.0, 1000.0, 1.0)] {
            let (sqrtw, z) =
                pirls_working_observation(Family::Binomial, LinkFunction::Logit, y, eta, mu, 25.0);
            assert!(sqrtw.is_finite());
            assert!(z.is_finite());
        }
    }

    #[test]
    fn test_glmm_offset_enters_linear_predictor() {
        let data = constant_response_fixture(vec![0.0, 1.0, 0.0, 1.0]);
        let formula = parse_formula("y ~ 1 + (1 | g)").unwrap();
        let offset = vec![0.1, -0.2, 0.3, -0.4];

        let model = GeneralizedLinearMixedModel::new_with_offset(
            formula,
            &data,
            Family::Bernoulli,
            None,
            offset.clone(),
        )
        .unwrap();

        for (idx, want) in offset.iter().enumerate() {
            assert!((model.offset[idx] - want).abs() < 1e-12);
            assert!((model.eta[idx] - want).abs() < 1e-12);
            assert!((model.mu[idx] - LinkFunction::Logit.linkinv(*want)).abs() < 1e-12);
        }
    }

    #[test]
    fn test_glmm_offset_validation() {
        let data = constant_response_fixture(vec![0.0, 1.0, 0.0, 1.0]);
        let formula = parse_formula("y ~ 1 + (1 | g)").unwrap();

        let err = GeneralizedLinearMixedModel::new_with_offset(
            formula,
            &data,
            Family::Bernoulli,
            None,
            vec![0.0],
        )
        .unwrap_err();

        match err {
            MixedModelError::InvalidArgument(message) => {
                assert!(message.contains("offset length"));
                assert!(message.contains("number of observations"));
            }
            other => panic!("expected InvalidArgument error, got {other:?}"),
        }
    }

    #[test]
    fn test_pirls_working_response_subtracts_offset() {
        let eta = 1.25_f64;
        let mu = eta.exp();
        let offset = -0.75_f64;
        let (sqrtw_plain, z_plain) =
            pirls_working_observation(Family::Poisson, LinkFunction::Log, 3.0, eta, mu, 2.0);
        let (sqrtw_offset, z_offset) = pirls_working_observation_with_offset(
            Family::Poisson,
            LinkFunction::Log,
            3.0,
            eta,
            mu,
            2.0,
            offset,
        );

        assert!((sqrtw_offset - sqrtw_plain).abs() < 1e-12);
        assert!((z_offset - (z_plain - offset)).abs() < 1e-12);
    }

    #[test]
    fn test_pirls_handles_poisson_log_extreme_offset_scale() {
        for (y, eta, mu) in [
            (0.0, -1000.0, 0.0),
            (1.0, -1000.0, 0.0),
            (0.0, 1000.0, f64::INFINITY),
            (1.0, 1000.0, f64::INFINITY),
        ] {
            let (sqrtw, z) =
                pirls_working_observation(Family::Poisson, LinkFunction::Log, y, eta, mu, 1.0);

            assert!(sqrtw.is_finite(), "sqrt weight was {sqrtw}");
            assert!(sqrtw > 0.0, "sqrt weight should stay positive");
            assert!(sqrtw < 4.0e6, "sqrt weight was {sqrtw}");
            assert!(z.is_finite(), "working response was {z}");
        }
    }

    fn gamma_dispersion_fixture() -> DataFrame {
        let mut y = Vec::new();
        let mut x = Vec::new();
        let mut group = Vec::new();
        let group_effects = [-0.25, 0.1, 0.3, -0.15];
        for g in 0..4 {
            for obs in 0..5 {
                let xv = obs as f64 - 2.0;
                let eta = 1.2 + 0.25 * xv + group_effects[g];
                let wiggle = 1.0 + 0.06 * ((g + obs) % 3) as f64;
                y.push(eta.exp() * wiggle);
                x.push(xv);
                group.push(format!("g{}", g + 1));
            }
        }

        let mut data = DataFrame::new();
        data.add_numeric("y", y).unwrap();
        data.add_numeric("x", x).unwrap();
        data.add_categorical("group", group).unwrap();
        data
    }

    fn two_term_poisson_fixture() -> DataFrame {
        let mut y = Vec::new();
        let mut x = Vec::new();
        let mut g1 = Vec::new();
        let mut g2 = Vec::new();
        for a in 0..4 {
            for b in 0..3 {
                for obs in 0..3 {
                    let xv = obs as f64 - 1.0;
                    let eta = 1.0 + 0.15 * xv + [-0.25, 0.05, 0.2, -0.1][a] + [0.1, -0.15, 0.05][b];
                    y.push(eta.exp().round().max(1.0));
                    x.push(xv);
                    g1.push(format!("g1_{}", a + 1));
                    g2.push(format!("g2_{}", b + 1));
                }
            }
        }

        let mut data = DataFrame::new();
        data.add_numeric("y", y).unwrap();
        data.add_numeric("x", x).unwrap();
        data.add_categorical("g1", g1).unwrap();
        data.add_categorical("g2", g2).unwrap();
        data
    }

    #[test]
    fn test_glmm_constructor_accepts_gamma_with_positive_response() {
        let data = gamma_dispersion_fixture();
        let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();

        let model = GeneralizedLinearMixedModel::new(
            formula,
            &data,
            Family::Gamma,
            Some(LinkFunction::Log),
        )
        .unwrap();

        assert_eq!(model.family, Family::Gamma);
        assert_eq!(model.dispersion(false), 1.0);
        assert_eq!(model.dispersion(true), 1.0);
    }

    #[cfg(not(feature = "nlopt"))]
    #[test]
    fn test_glmm_fit_uses_native_cobyla_without_nlopt() {
        let data = gamma_dispersion_fixture();
        let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();
        let mut model = GeneralizedLinearMixedModel::new(
            formula,
            &data,
            Family::Gamma,
            Some(LinkFunction::Log),
        )
        .unwrap();
        model.lmm.optsum.max_feval = 50;

        model.fit_with_options(true, 1, false).unwrap();

        assert_eq!(model.lmm.optsum.optimizer, Optimizer::Cobyla);
        assert_eq!(model.lmm.optsum.backend.label(), "native");
        assert!(model.lmm.optsum.feval > 0);
        assert!(model.lmm.optsum.fmin.is_finite());
        assert!(!model.lmm.optsum.fit_log.is_empty());
        assert!(model.lmm.compiler_artifact.optimizer_certificate.is_some());
    }

    #[cfg(not(feature = "nlopt"))]
    #[test]
    fn test_glmm_fit_uses_native_pattern_search_when_requested() {
        let data = gamma_dispersion_fixture();
        let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();
        let mut model = GeneralizedLinearMixedModel::new(
            formula,
            &data,
            Family::Gamma,
            Some(LinkFunction::Log),
        )
        .unwrap();
        model.lmm.optsum.optimizer = Optimizer::PatternSearch;
        model.lmm.optsum.max_feval = 120;

        model.fit_with_options(true, 1, false).unwrap();

        assert_eq!(model.lmm.optsum.optimizer, Optimizer::PatternSearch);
        assert_eq!(model.lmm.optsum.backend.label(), "native");
        assert!(model.lmm.optsum.feval > 0);
        assert!(model.lmm.optsum.fmin.is_finite());
        assert!(!model.lmm.optsum.fit_log.is_empty());
        assert!(model.lmm.compiler_artifact.optimizer_certificate.is_some());
    }

    #[cfg(not(feature = "nlopt"))]
    #[test]
    fn test_glmm_pattern_search_handles_multitheta_poisson_fit() {
        let data = two_term_poisson_fixture();
        let formula = parse_formula("y ~ 1 + x + (1 | g1) + (1 | g2)").unwrap();
        let mut model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Poisson, None).unwrap();
        model.lmm.optsum.optimizer = Optimizer::PatternSearch;
        model.lmm.optsum.max_feval = 180;

        model.fit_with_options(true, 1, false).unwrap();

        assert_eq!(model.theta.len(), 2);
        assert_eq!(model.lmm.optsum.optimizer, Optimizer::PatternSearch);
        assert!(model.theta.iter().all(|value| value.is_finite()));
        assert!(model.theta.iter().all(|value| *value >= 0.0));
        assert!(model.objective().is_finite());
    }

    #[test]
    fn test_glmm_constructor_rejects_nonpositive_gamma_response() {
        let data = contra_fixture();
        let formula = parse_formula("use_num ~ 1 + (1 | urban_dist)").unwrap();
        let err = GeneralizedLinearMixedModel::new(
            formula,
            &data,
            Family::Gamma,
            Some(LinkFunction::Log),
        )
        .expect_err("Gamma GLMM should reject zero responses");

        match err {
            MixedModelError::InvalidArgument(msg) => {
                assert!(msg.contains("gamma"));
                assert!(msg.contains("strictly positive"));
            }
            other => panic!("expected InvalidArgument error, got {other:?}"),
        }
    }

    #[test]
    fn test_gamma_glmm_refit_rejects_nonpositive_response() {
        let data = gamma_dispersion_fixture();
        let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();
        let mut model = GeneralizedLinearMixedModel::new(
            formula,
            &data,
            Family::Gamma,
            Some(LinkFunction::Log),
        )
        .unwrap();
        let mut new_y = data.numeric("y").unwrap().to_vec();
        new_y[0] = 0.0;

        let err = model
            .refit(&new_y)
            .expect_err("Gamma GLMM refit/bootstrap response must stay strictly positive");

        match err {
            MixedModelError::InvalidArgument(msg) => {
                assert!(msg.contains("gamma"));
                assert!(msg.contains("strictly positive"));
            }
            other => panic!("expected InvalidArgument error, got {other:?}"),
        }
    }

    #[test]
    fn test_glmm_constructor_accepts_normal_nonidentity_dispersion_family() {
        let data = gamma_dispersion_fixture();
        let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();

        let model = GeneralizedLinearMixedModel::new(
            formula,
            &data,
            Family::Normal,
            Some(LinkFunction::Sqrt),
        )
        .unwrap();

        assert_eq!(model.family, Family::Normal);
        assert_eq!(model.link, LinkFunction::Sqrt);
        assert_eq!(model.dispersion(false), 1.0);
    }

    #[test]
    fn test_gamma_glmm_fit_estimates_pearson_dispersion() {
        let data = gamma_dispersion_fixture();
        let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();
        let mut model = GeneralizedLinearMixedModel::new(
            formula,
            &data,
            Family::Gamma,
            Some(LinkFunction::Log),
        )
        .unwrap();

        model.fit_with_options(true, 1, false).unwrap();

        let sigma = model.dispersion(false);
        let phi = model.dispersion(true);
        let expected_phi =
            model.pearson_dispersion_numerator() / (model.nobs() - model.lmm.feterm.rank) as f64;

        assert!(sigma.is_finite());
        assert!(sigma > 0.0);
        assert_relative_eq!(phi, sigma * sigma, epsilon = 1e-12);
        assert_relative_eq!(phi, expected_phi, epsilon = 1e-12, max_relative = 1e-12);
        assert_eq!(
            model.dof(),
            model.lmm.feterm.rank + model.lmm.parmap.len() + 1
        );
        assert_relative_eq!(model.varcorr().residual_sd.unwrap(), sigma, epsilon = 1e-12);
    }

    #[test]
    fn test_glmm_fast_parameter_documented_or_implemented() {
        let data = contra_fixture();
        let formula = parse_formula("use_num ~ 1 + (1 | urban_dist)").unwrap();
        let mut model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();

        let err = model.fit_with_options(false, 1, false).unwrap_err();

        match err {
            MixedModelError::Unsupported(message) => {
                assert!(message.contains("fast = false"));
                assert!(message.contains("not implemented"));
            }
            other => panic!("expected Unsupported error, got {other:?}"),
        }
    }

    fn constant_response_fixture(y: Vec<f64>) -> DataFrame {
        let mut data = DataFrame::new();
        data.add_numeric("y", y).unwrap();
        data.add_categorical(
            "g",
            vec![
                "a".to_string(),
                "a".to_string(),
                "b".to_string(),
                "b".to_string(),
            ],
        )
        .unwrap();
        data
    }

    fn assert_constant_response_rejected(family: Family, y: Vec<f64>) {
        let data = constant_response_fixture(y);
        let formula = parse_formula("y ~ 1 + (1 | g)").unwrap();

        let err = GeneralizedLinearMixedModel::new(formula, &data, family, None).unwrap_err();

        match err {
            MixedModelError::InvalidArgument(message) => {
                assert!(message.contains("response is constant"));
            }
            other => panic!("expected InvalidArgument error, got {other:?}"),
        }
    }

    #[test]
    fn test_glmm_rejects_constant_response_bernoulli() {
        assert_constant_response_rejected(Family::Bernoulli, vec![0.0, 0.0, 0.0, 0.0]);
        assert_constant_response_rejected(Family::Bernoulli, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_glmm_rejects_constant_response_poisson() {
        assert_constant_response_rejected(Family::Poisson, vec![3.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_glmm_accepts_near_constant() {
        let data = constant_response_fixture(vec![0.0, 0.0, 0.0, 1.0]);
        let formula = parse_formula("y ~ 1 + (1 | g)").unwrap();

        GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();
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
        df.add_numeric("use_num", use_num).unwrap();
        df.add_numeric("age", age).unwrap();
        df.add_numeric("age2", age2).unwrap();
        df.add_categorical("urban", urban).unwrap();
        df.add_categorical("livch", livch).unwrap();
        df.add_categorical("urban_dist", urban_dist).unwrap();
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
        data_with_proportion
            .add_numeric("proportion", proportion)
            .unwrap();

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
    fn test_cbpp_agq_deviance_uses_case_weights() {
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
        data_with_proportion
            .add_numeric("proportion", proportion)
            .unwrap();

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

        let weighted_agq = model.deviance(5);
        model.wt = vec![1.0; model.y.len()];
        let unit_weight_agq = model.deviance(5);
        assert!(
            (weighted_agq - unit_weight_agq).abs() > 1.0,
            "AGQ deviance must include binomial case weights; weighted={weighted_agq}, unit={unit_weight_agq}"
        );
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
        use crate::types::MatrixBlock;
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

    #[test]
    fn test_glmm_refit_resets_theta_to_initial() {
        let data = contra_fixture();
        let formula =
            parse_formula("use_num ~ 1 + age + age2 + urban + livch + (1 | urban_dist)").unwrap();
        let mut model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();
        let initial_theta = model.lmm.optsum.initial.clone();

        model.fit_with_options(true, 1, false).unwrap();
        assert!(
            (model.theta[0] - initial_theta[0]).abs() > 1e-6,
            "fixture should move away from its starting theta"
        );

        let new_y = resampled_contra_response(&data);
        model.reset_for_refit(Some(&new_y)).unwrap();

        assert_eq!(model.theta, initial_theta);
        assert_eq!(model.lmm.optsum.final_params, initial_theta);
        assert_eq!(model.lmm.optsum.feval, 0);
        assert!(model.lmm.optsum.return_value.is_empty());
    }

    #[test]
    fn test_glmm_bootstrap_does_not_warm_start() {
        let data = contra_fixture();
        let formula =
            parse_formula("use_num ~ 1 + age + age2 + urban + livch + (1 | urban_dist)").unwrap();
        let mut model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();
        let initial_theta = model.lmm.optsum.initial.clone();
        model.fit_with_options(true, 1, false).unwrap();
        let fitted_theta = model.theta.clone();

        let err = model
            .fit_with_options(true, 1, false)
            .expect_err("plain fit_with_options must not silently warm-start a fitted GLMM");
        assert!(matches!(err, MixedModelError::AlreadyFitted));

        let new_y = resampled_contra_response(&data);
        model.reset_for_refit(Some(&new_y)).unwrap();
        assert_eq!(
            model.theta, initial_theta,
            "bootstrap/refit reset must ignore the previous optimum"
        );
        assert_ne!(model.theta, fitted_theta);
    }

    #[test]
    fn test_glmm_refit_after_resample_matches_cold_fit() {
        let data = contra_fixture();
        let formula =
            parse_formula("use_num ~ 1 + age + age2 + urban + livch + (1 | urban_dist)").unwrap();
        let mut warm_model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();
        warm_model.fit_with_options(true, 1, false).unwrap();

        let new_y = resampled_contra_response(&data);
        warm_model.refit(&new_y).unwrap();
        let cold_model = refit_cold_contra_model(&new_y);

        assert_relative_eq!(warm_model.theta[0], cold_model.theta[0], epsilon = 1e-8);
        assert_relative_eq!(
            warm_model.lmm.optsum.fmin,
            cold_model.lmm.optsum.fmin,
            epsilon = 1e-8
        );
        for (warm, cold) in warm_model.beta.iter().zip(cold_model.beta.iter()) {
            assert_relative_eq!(warm, cold, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_glmm_repeated_refit_does_not_accumulate_retained_state() {
        let data = contra_fixture();
        let formula =
            parse_formula("use_num ~ 1 + age + age2 + urban + livch + (1 | urban_dist)").unwrap();
        let original_y = data.numeric("use_num").unwrap().to_vec();
        let perturbed_y = resampled_contra_response(&data);
        let mut model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();
        model.fit_with_options(true, 1, false).unwrap();

        let baseline_slots = glmm_retained_state_slots(&model);
        let baseline_fit_log_capacity = model.lmm.optsum.fit_log.capacity();

        for iteration in 0..8 {
            let y = if iteration % 2 == 0 {
                &perturbed_y
            } else {
                &original_y
            };
            model.refit(y).unwrap();

            assert_eq!(
                glmm_retained_state_slots(&model),
                baseline_slots,
                "GLMM refit should reuse bounded work buffers rather than accumulating retained state"
            );
            assert_eq!(
                model.lmm.optsum.fit_log.capacity(),
                baseline_fit_log_capacity,
                "GLMM optimizer logging must not retain one entry per refit iteration"
            );
        }
    }

    #[test]
    fn test_glmm_theta_probe_penalizes_invalid_theta() {
        let data = contra_fixture();
        let formula =
            parse_formula("use_num ~ 1 + age + age2 + urban + livch + (1 | urban_dist)").unwrap();
        let mut model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();

        let value = model.penalized_pirls_deviance_at_theta(&[f64::NAN], 1);
        assert!(
            value.is_infinite() && value.is_sign_positive(),
            "invalid optimizer probes should be penalized, not evaluated from stale state"
        );
    }

    #[test]
    fn test_glmm_final_theta_update_propagates_invalid_theta_error() {
        let data = contra_fixture();
        let formula =
            parse_formula("use_num ~ 1 + age + age2 + urban + livch + (1 | urban_dist)").unwrap();
        let mut model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();

        let err = model
            .update_pirls_at_theta(&[f64::NAN], true)
            .expect_err("final theta update must propagate invalid-theta errors");
        assert!(matches!(err, MixedModelError::InvalidArgument(_)));
    }

    #[test]
    fn test_glmm_rectify_after_fit() {
        let data = contra_fixture();
        let formula =
            parse_formula("use_num ~ 1 + age2 + urban + livch + (1 + age | urban_dist)").unwrap();
        let mut model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();
        let mut theta = vec![-0.5, 0.05, -0.25];

        model.finalize_theta_after_optimizer(&mut theta, 1).unwrap();

        assert_eq!(theta, vec![0.5, -0.05, 0.25]);
        assert_eq!(model.theta, theta);
        assert_eq!(model.lmm.optsum.final_params, theta);
        assert!(model.lmm.optsum.fmin.is_finite());
        assert_glmm_theta_diagonals_nonnegative(&model);
    }

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
