//! Likelihood ratio tests for nested mixed models.

use crate::model::traits::MixedModelFit;

/// Result of a likelihood ratio test comparing nested models.
#[derive(Debug, Clone)]
pub struct LikelihoodRatioTest {
    /// Number of observations (must be equal across models).
    pub nobs: usize,
    /// Degrees of freedom for each model.
    pub dof: Vec<usize>,
    /// Log-likelihood for each model.
    pub loglik: Vec<f64>,
    /// Deviance (-2 * loglik) for each model.
    pub deviance: Vec<f64>,
    /// Chi-squared statistics (between successive models).
    pub chisq: Vec<f64>,
    /// Degrees of freedom for each chi-squared test.
    pub chisq_dof: Vec<usize>,
    /// P-values for each test.
    pub pvalues: Vec<f64>,
}

impl LikelihoodRatioTest {
    /// Perform a likelihood ratio test on two or more nested models.
    ///
    /// Models should be provided in order from smallest to largest.
    pub fn test(models: &[&dyn MixedModelFit]) -> Result<Self, String> {
        if models.len() < 2 {
            return Err("At least two models are needed".to_string());
        }

        let nobs = models[0].nobs();
        for m in models {
            if m.nobs() != nobs {
                return Err("All models must have the same number of observations".to_string());
            }
        }

        let dof: Vec<usize> = models.iter().map(|m| m.dof()).collect();
        let loglik: Vec<f64> = models.iter().map(|m| m.loglikelihood()).collect();
        let deviance: Vec<f64> = loglik.iter().map(|ll| -2.0 * ll).collect();

        let mut chisq = Vec::new();
        let mut chisq_dof = Vec::new();
        let mut pvalues = Vec::new();

        for i in 1..models.len() {
            let chi = 2.0 * (loglik[i] - loglik[i - 1]).abs();
            let ddof = dof[i].saturating_sub(dof[i - 1]);
            let pval = if ddof > 0 {
                use statrs::distribution::{ChiSquared, ContinuousCDF};
                let dist = ChiSquared::new(ddof as f64).unwrap();
                1.0 - dist.cdf(chi)
            } else {
                1.0
            };
            chisq.push(chi);
            chisq_dof.push(ddof);
            pvalues.push(pval);
        }

        Ok(LikelihoodRatioTest {
            nobs,
            dof,
            loglik,
            deviance,
            chisq,
            chisq_dof,
            pvalues,
        })
    }
}
