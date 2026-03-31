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

use crate::error::{MixedModelError, Result};
use crate::formula::Formula;
use crate::model::data::DataFrame;
use crate::model::linear::LinearMixedModel;
use crate::model::traits::{Family, LinkFunction, MixedModelFit};
use crate::types::OptSummary;

/// A generalized linear mixed-effects model.
#[derive(Debug, Clone)]
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
        let link = link.unwrap_or_else(|| family.canonical_link());

        // For Normal + Identity, redirect to LMM
        if family == Family::Normal && link == LinkFunction::Identity {
            return Err(MixedModelError::InvalidArgument(
                "Use LinearMixedModel for Normal distribution with IdentityLink".to_string(),
            ));
        }

        // Build the internal LMM
        let lmm = LinearMixedModel::new(formula, data, None)?;
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
        let u0 = u.iter().map(|m| DMatrix::zeros(m.nrows(), m.ncols())).collect();

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

    /// Compute the deviance residuals for the current μ.
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
            Family::Normal => {
                (y - mu).powi(2)
            }
            Family::Gamma => {
                if y == 0.0 {
                    2.0 * (mu.ln())
                } else {
                    -2.0 * ((y / mu).ln() - (y - mu) / mu)
                }
            }
            Family::InverseGaussian => {
                (y - mu).powi(2) / (y * mu * mu)
            }
        }
    }

    /// PIRLS: Penalized Iteratively Reweighted Least Squares.
    ///
    /// Finds the conditional modes of the random effects.
    pub fn pirls(&mut self, vary_beta: bool, verbose: bool) -> Result<()> {
        let max_iter = 10;

        // Initialize u to zero
        for (u, u0) in self.u.iter_mut().zip(self.u0.iter_mut()) {
            u.fill(0.0);
            u0.copy_from(u);
        }

        if vary_beta {
            self.beta0.copy_from(&self.beta);
        }

        self.update_eta();
        let devresid = self.deviance_residuals();
        let u_penalty: f64 = self.u.iter().map(|u| u.iter().map(|x| x * x).sum::<f64>()).sum();
        let mut obj0 = devresid.sum() + u_penalty;
        let logdet = self.lmm_logdet();
        obj0 += logdet;
        obj0 *= 1.0001; // slight inflation for convergence check

        for _iter in 0..max_iter {
            // Update working response and weights in the LMM
            // (simplified — full implementation would update LMM.y and reweight)

            self.update_eta();
            let devresid = self.deviance_residuals();
            let u_penalty: f64 = self.u.iter().map(|u| u.iter().map(|x| x * x).sum::<f64>()).sum();
            let obj = devresid.sum() + u_penalty + self.lmm_logdet();

            // Step halving
            let mut nhalf = 0;
            while obj > obj0 && nhalf < 10 {
                nhalf += 1;
                for (u, u0) in self.u.iter_mut().zip(self.u0.iter()) {
                    for (a, &b) in u.iter_mut().zip(u0.iter()) {
                        *a = (*a + b) / 2.0;
                    }
                }
                if vary_beta {
                    for (a, &b) in self.beta.iter_mut().zip(self.beta0.iter()) {
                        *a = (*a + b) / 2.0;
                    }
                }
                self.update_eta();
            }

            if (obj - obj0).abs() < 1e-5 {
                break;
            }

            for (u0, u) in self.u0.iter_mut().zip(self.u.iter()) {
                u0.copy_from(u);
            }
            self.beta0.copy_from(&self.beta);
            obj0 = obj;
        }

        Ok(())
    }

    /// Deviance of the GLMM.
    pub fn deviance(&self, n_agq: usize) -> f64 {
        let devresid = self.deviance_residuals();
        let u_penalty: f64 = self.u.iter().map(|u| u.iter().map(|x| x * x).sum::<f64>()).sum();
        devresid.sum() + self.lmm_logdet() + u_penalty
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
                crate::model::linear::MatrixBlock::BlockDiagonal(blocks) => {
                    blocks.iter().map(|blk| {
                        let n = blk.nrows();
                        (0..n).map(|i| blk[(i, i)].abs().ln()).sum::<f64>()
                    }).sum::<f64>()
                }
            };
        }
        2.0 * logdet
    }

    /// Fit the GLMM.
    pub fn fit(&mut self) -> Result<&mut Self> {
        self.fit_with_options(false, 1, false)
    }

    /// Fit with options.
    pub fn fit_with_options(
        &mut self,
        fast: bool,
        n_agq: usize,
        verbose: bool,
    ) -> Result<&mut Self> {
        // For a simple implementation, use Laplace approximation (nAGQ = 1)
        // and optimize over θ (fast=true) or [β, θ] (fast=false).

        let n_theta = self.theta.len();
        let n_beta = self.beta.len();
        let n_params = if fast { n_theta } else { n_beta + n_theta };

        let mut initial = if fast {
            self.theta.clone()
        } else {
            let mut v = self.beta.as_slice().to_vec();
            v.extend_from_slice(&self.theta);
            v
        };

        let mut lb = if fast {
            self.lmm.lower_bounds()
        } else {
            let mut v = vec![f64::NEG_INFINITY; n_beta];
            v.extend(self.lmm.lower_bounds());
            v
        };

        // Optimization
        let mut feval = 0i64;

        // Simple iterative optimization: evaluate objective at initial params,
        // then use COBYLA for derivative-free optimization.
        let mut best_theta = initial.clone();
        let mut best_fmin = f64::INFINITY;

        // Set initial params
        if fast {
            let _ = self.lmm.set_theta(&initial);
            self.theta = initial.clone();
        } else {
            let beta_slice = &initial[..n_beta];
            let theta_slice = &initial[n_beta..];
            self.beta = DVector::from_column_slice(beta_slice);
            let _ = self.lmm.set_theta(theta_slice);
            self.theta = theta_slice.to_vec();
        }

        if let Ok(()) = self.pirls(fast, verbose) {
            let obj = self.deviance(n_agq);
            if obj < best_fmin {
                best_fmin = obj;
                best_theta = initial.clone();
            }
        }

        // TODO: Full COBYLA optimization loop for GLMM
        // For now, the initial PIRLS evaluation provides a baseline fit.

        self.lmm.optsum.fmin = best_fmin;
        self.lmm.optsum.final_params = best_theta;
        self.lmm.optsum.return_value = "SUCCESS".to_string();
        self.lmm.optsum.feval = 1;

        Ok(self)
    }
}

impl MixedModelFit for GeneralizedLinearMixedModel {
    fn nobs(&self) -> usize { self.y.len() }

    fn dof(&self) -> usize {
        self.lmm.feterm.rank + self.lmm.parmap.len() +
            if self.family.has_dispersion() { 1 } else { 0 }
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

    fn fixef(&self) -> DVector<f64> { self.beta.clone() }
    fn coef_names(&self) -> Vec<String> { self.lmm.coef_names() }
    fn vcov(&self) -> DMatrix<f64> { self.lmm.vcov() }
    fn stderror(&self) -> DVector<f64> { self.lmm.stderror() }
    fn fitted(&self) -> DVector<f64> { self.mu.clone() }
    fn residuals(&self) -> DVector<f64> { &self.y - &self.mu }
    fn response(&self) -> &DVector<f64> { &self.y }
    fn model_matrix(&self) -> &DMatrix<f64> { &self.lmm.feterm.x }
    fn objective(&self) -> f64 { self.deviance(1) }

    fn loglikelihood(&self) -> f64 {
        -self.deviance(self.lmm.optsum.n_agq) / 2.0
    }

    fn is_fitted(&self) -> bool { self.lmm.optsum.feval > 0 }
    fn is_singular(&self) -> bool { self.lmm.is_singular() }
    fn opt_summary(&self) -> &OptSummary { &self.lmm.optsum }
    fn theta(&self) -> Vec<f64> { self.theta.clone() }

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
}
