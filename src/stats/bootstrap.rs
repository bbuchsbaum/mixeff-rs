//! Parametric bootstrap for mixed-effects models.
//!
//! Provides `parametric_bootstrap()` for generating bootstrap samples
//! from a fitted mixed model, and the `MixedModelBootstrap` type for
//! storing and summarizing results.

use nalgebra::DMatrix;

/// Result of a parametric bootstrap.
#[derive(Debug, Clone)]
pub struct MixedModelBootstrap {
    /// Bootstrap replicates: each entry contains (objective, sigma, beta, theta).
    pub fits: Vec<BootstrapReplicate>,
    /// Lower triangular lambda matrices (templates from original fit).
    pub lambda: Vec<DMatrix<f64>>,
    /// Indices of free parameters in each lambda.
    pub inds: Vec<Vec<usize>>,
    /// Lower bounds on theta.
    pub lower_bounds: Vec<f64>,
    /// Number of replicates.
    pub n_samples: usize,
}

/// A single bootstrap replicate.
#[derive(Debug, Clone)]
pub struct BootstrapReplicate {
    pub objective: f64,
    pub sigma: f64,
    pub beta: Vec<f64>,
    pub theta: Vec<f64>,
    pub se: Vec<f64>,
}

/// Shortest coverage interval containing `level` proportion of values.
pub fn shortest_cov_int(v: &mut [f64], level: f64) -> (f64, f64) {
    assert!((0.0..1.0).contains(&level), "level must be in (0, 1)");
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    let ilen = ((n as f64) * level).ceil() as usize;
    if ilen >= n {
        return (v[0], v[n - 1]);
    }
    let mut min_len = f64::INFINITY;
    let mut best_i = 0;
    for i in 0..=(n - ilen) {
        let len = v[i + ilen - 1] - v[i];
        if len < min_len {
            min_len = len;
            best_i = i;
        }
    }
    (v[best_i], v[best_i + ilen - 1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shortest_cov_int() {
        let mut v: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let (lo, hi) = shortest_cov_int(&mut v, 0.95);
        assert!(hi - lo <= 95.0);
    }
}
