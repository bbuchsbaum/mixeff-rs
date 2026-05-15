//! Trait definitions for mixed models.

use nalgebra::{DMatrix, DVector};

use crate::types::OptSummary;

/// Structural summary of one random-effects term.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RandomEffectTermInfo {
    /// Grouping factor name.
    pub group: String,
    /// Random-effect basis columns for this grouping factor.
    pub columns: Vec<String>,
}

/// Distribution families for GLMMs.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum Family {
    Normal,
    Bernoulli,
    Binomial,
    Poisson,
    Gamma,
    InverseGaussian,
}

/// Link functions for GLMMs.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum LinkFunction {
    Identity,
    Log,
    Logit,
    Probit,
    Cloglog,
    Inverse,
    Sqrt,
}

impl Family {
    /// Return the canonical link function for this family.
    pub fn canonical_link(&self) -> LinkFunction {
        match self {
            Family::Normal => LinkFunction::Identity,
            Family::Bernoulli => LinkFunction::Logit,
            Family::Binomial => LinkFunction::Logit,
            Family::Poisson => LinkFunction::Log,
            Family::Gamma => LinkFunction::Inverse,
            Family::InverseGaussian => LinkFunction::Inverse,
        }
    }

    /// Whether this family has a dispersion parameter.
    pub fn has_dispersion(&self) -> bool {
        matches!(
            self,
            Family::Normal | Family::Gamma | Family::InverseGaussian
        )
    }

    /// GLM variance function V(μ) for this conditional family.
    pub fn variance(&self, mu: f64) -> f64 {
        match self {
            Family::Normal => 1.0,
            Family::Bernoulli | Family::Binomial => mu * (1.0 - mu),
            Family::Poisson => mu,
            Family::Gamma => mu * mu,
            Family::InverseGaussian => mu * mu * mu,
        }
    }
}

impl LinkFunction {
    /// Apply the link function: η = g(μ)
    pub fn link(&self, mu: f64) -> f64 {
        match self {
            LinkFunction::Identity => mu,
            LinkFunction::Log => mu.ln(),
            LinkFunction::Logit => (mu / (1.0 - mu)).ln(),
            LinkFunction::Probit => {
                use statrs::distribution::{ContinuousCDF, Normal};
                let n = Normal::new(0.0, 1.0).unwrap();
                n.inverse_cdf(mu)
            }
            LinkFunction::Cloglog => (-(-mu).ln_1p()).ln(),
            LinkFunction::Inverse => 1.0 / mu,
            LinkFunction::Sqrt => mu.sqrt(),
        }
    }

    /// Apply the inverse link function: μ = g⁻¹(η)
    pub fn linkinv(&self, eta: f64) -> f64 {
        match self {
            LinkFunction::Identity => eta,
            LinkFunction::Log => eta.exp(),
            LinkFunction::Logit => {
                let e = eta.exp();
                e / (1.0 + e)
            }
            LinkFunction::Probit => {
                use statrs::distribution::{ContinuousCDF, Normal};
                let n = Normal::new(0.0, 1.0).unwrap();
                n.cdf(eta)
            }
            LinkFunction::Cloglog => -(-eta.exp()).exp_m1(),
            LinkFunction::Inverse => 1.0 / eta,
            LinkFunction::Sqrt => eta * eta,
        }
    }

    /// Derivative of the inverse link: dμ/dη
    pub fn mu_eta(&self, eta: f64) -> f64 {
        match self {
            LinkFunction::Identity => 1.0,
            LinkFunction::Log => eta.exp(),
            LinkFunction::Logit => {
                let e = eta.exp();
                e / (1.0 + e).powi(2)
            }
            LinkFunction::Probit => {
                use statrs::distribution::{Continuous, Normal};
                let n = Normal::new(0.0, 1.0).unwrap();
                n.pdf(eta)
            }
            LinkFunction::Cloglog => {
                if eta == f64::INFINITY {
                    return 0.0;
                }
                let exp_eta = eta.exp();
                (eta - exp_eta).exp()
            }
            LinkFunction::Inverse => -1.0 / (eta * eta),
            LinkFunction::Sqrt => 2.0 * eta,
        }
    }
}

/// Common interface for fitted mixed models.
pub trait MixedModelFit {
    /// Number of observations.
    fn nobs(&self) -> usize;

    /// Degrees of freedom (number of estimated parameters).
    fn dof(&self) -> usize;

    /// Fixed-effects coefficient vector (unpivoted).
    fn coef(&self) -> DVector<f64>;

    /// Fixed-effects coefficient vector (pivoted, possibly truncated).
    fn fixef(&self) -> DVector<f64>;

    /// Names of coefficients.
    fn coef_names(&self) -> Vec<String>;

    /// Variance-covariance matrix of fixed effects.
    fn vcov(&self) -> DMatrix<f64>;

    /// Standard errors of fixed effects.
    fn stderror(&self) -> DVector<f64>;

    /// Fitted values.
    fn fitted(&self) -> DVector<f64>;

    /// Residuals.
    fn residuals(&self) -> DVector<f64>;

    /// Response vector.
    fn response(&self) -> &DVector<f64>;

    /// Model matrix for fixed effects.
    fn model_matrix(&self) -> &DMatrix<f64>;

    /// The objective function value (deviance or REML criterion).
    fn objective(&self) -> f64;

    /// Log-likelihood.
    fn loglikelihood(&self) -> f64;

    /// Canonical formula label used in summaries, if available.
    fn formula_label(&self) -> Option<String> {
        None
    }

    /// AIC.
    fn aic(&self) -> f64 {
        -2.0 * self.loglikelihood() + 2.0 * self.dof() as f64
    }

    /// AICc (corrected AIC).
    fn aicc(&self) -> f64 {
        let n = self.nobs() as f64;
        let k = self.dof() as f64;
        self.aic() + 2.0 * k * (k + 1.0) / (n - k - 1.0)
    }

    /// BIC.
    fn bic(&self) -> f64 {
        let n = self.nobs() as f64;
        -2.0 * self.loglikelihood() + self.dof() as f64 * n.ln()
    }

    /// Whether the model has been fitted.
    fn is_fitted(&self) -> bool;

    /// Whether the fit is singular (any θ at its lower bound).
    fn is_singular(&self) -> bool;

    /// Optimization summary.
    fn opt_summary(&self) -> &OptSummary;

    /// The θ parameter vector.
    fn theta(&self) -> Vec<f64>;

    /// Dispersion parameter estimate (σ² for LMM).
    fn dispersion(&self, sqr: bool) -> f64;

    /// Random effects (conditional modes), one matrix per grouping factor.
    fn ranef(&self) -> Vec<DMatrix<f64>>;

    /// Random-effects term structure, used by model-comparison helpers to
    /// reject obviously non-nested comparisons before computing LRT statistics.
    fn random_effect_terms(&self) -> Vec<RandomEffectTermInfo> {
        Vec::new()
    }

    /// Conditional distribution family. `None` for ordinary `LinearMixedModel`s
    /// (Gaussian by construction); `Some(_)` for GLMMs.
    ///
    /// Used by `LikelihoodRatioTest` to refuse comparisons across families,
    /// matching `MixedModels._samefamily` in the Julia reference.
    fn family_kind(&self) -> Option<Family> {
        None
    }

    /// Link function. `None` for ordinary `LinearMixedModel`s (identity by
    /// construction); `Some(_)` for GLMMs.
    fn link_kind(&self) -> Option<LinkFunction> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::{Family, LinkFunction};

    #[test]
    fn test_family_variance_functions() {
        assert_eq!(Family::Normal.variance(2.0), 1.0);
        assert_eq!(Family::Poisson.variance(2.0), 2.0);
        assert_eq!(Family::Gamma.variance(2.0), 4.0);
        assert_eq!(Family::InverseGaussian.variance(2.0), 8.0);
        assert_eq!(Family::Bernoulli.variance(0.25), 0.1875);
        assert_eq!(Family::Binomial.variance(0.25), 0.1875);
    }

    #[test]
    fn test_inverse_link_mu_eta_preserves_sign() {
        assert_eq!(LinkFunction::Inverse.mu_eta(0.5), -4.0);
        assert_eq!(LinkFunction::Log.mu_eta(0.5), 0.5_f64.exp());
    }

    #[test]
    fn test_cloglog_link_round_trips_and_handles_extremes() {
        for mu in [1e-12, 0.01, 0.25, 0.75, 1.0 - 1e-12] {
            let eta = LinkFunction::Cloglog.link(mu);
            let roundtrip = LinkFunction::Cloglog.linkinv(eta);
            assert!((roundtrip - mu).abs() <= 2e-15_f64.max(1e-12 * mu.abs()));
            assert!(LinkFunction::Cloglog.mu_eta(eta).is_finite());
            assert!(LinkFunction::Cloglog.mu_eta(eta) >= 0.0);
        }

        assert_eq!(LinkFunction::Cloglog.linkinv(f64::NEG_INFINITY), 0.0);
        assert_eq!(LinkFunction::Cloglog.linkinv(f64::INFINITY), 1.0);
        assert_eq!(LinkFunction::Cloglog.mu_eta(f64::INFINITY), 0.0);
    }
}
