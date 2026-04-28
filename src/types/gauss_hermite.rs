//! Gauss-Hermite quadrature on the normalized scale.
//!
//! `GaussHermiteNormalized` provides abscissae and weights for Gauss-Hermite
//! quadrature, normalized so that the weights sum to unity. This allows
//! evaluation of expectations with respect to a normal density:
//!
//! ```text
//! E[h(X)] ≈ Σ_i  h(σ * z_i + μ) * w_i
//! ```
//!
//! where `X ~ N(μ, σ)`.
//!
//! This is a Rust port of `GaussHermiteNormalized{K}` and the memoized
//! `GHnorm` function from MixedModels.jl.

use nalgebra::DMatrix;
use std::collections::HashMap;
use std::sync::Mutex;

/// Gauss-Hermite quadrature rule on the normalized (Z) scale.
///
/// # Fields
///
/// * `z` - abscissae (quadrature nodes)
/// * `w` - weights, normalized to sum to 1
#[derive(Debug, Clone)]
pub struct GaussHermiteNormalized {
    /// Abscissae for the quadrature rule.
    pub z: Vec<f64>,
    /// Weights normalized to sum to unity.
    pub w: Vec<f64>,
}

impl GaussHermiteNormalized {
    /// Compute a `k`-point Gauss-Hermite quadrature rule on the normalized
    /// scale.
    ///
    /// The construction mirrors the Julia implementation:
    ///
    /// 1. Form the symmetric tridiagonal matrix with zero diagonal and
    ///    sub/super-diagonal `[sqrt(1), sqrt(2), ..., sqrt(k-1)]`.
    /// 2. Compute its eigendecomposition.
    /// 3. The eigenvalues give the abscissae; the squared first-row entries
    ///    of the eigenvector matrix give (unnormalized) weights.
    /// 4. Symmetrize both vectors and normalize the weights to sum to 1.
    ///
    /// # Panics
    ///
    /// Panics if `k == 0`.
    pub fn new(k: usize) -> Self {
        assert!(k > 0, "number of quadrature points must be positive");

        if k == 1 {
            return Self {
                z: vec![0.0],
                w: vec![1.0],
            };
        }

        // Build the symmetric tridiagonal matrix as a full DMatrix.
        // Diagonal = 0, sub/super-diagonal[i] = sqrt(i+1) for i in 0..k-1.
        let mut mat = DMatrix::zeros(k, k);
        for i in 0..(k - 1) {
            let val = ((i + 1) as f64).sqrt();
            mat[(i, i + 1)] = val;
            mat[(i + 1, i)] = val;
        }

        // Eigendecomposition of the symmetric matrix.
        let eig = mat.symmetric_eigen();
        let eigenvalues = &eig.eigenvalues;
        let eigenvectors = &eig.eigenvectors;

        // Sort eigenvalues (and corresponding eigenvectors) in ascending order.
        let mut indices: Vec<usize> = (0..k).collect();
        indices.sort_by(|&a, &b| eigenvalues[a].partial_cmp(&eigenvalues[b]).unwrap());

        let sorted_vals: Vec<f64> = indices.iter().map(|&i| eigenvalues[i]).collect();
        let first_row: Vec<f64> = indices.iter().map(|&i| eigenvectors[(0, i)]).collect();

        // Symmetrize abscissae: z = (vals - reverse(vals)) / 2
        let n = sorted_vals.len();
        let z: Vec<f64> = (0..n)
            .map(|i| (sorted_vals[i] - sorted_vals[n - 1 - i]) / 2.0)
            .collect();

        // Weights: squared first-row entries, symmetrized and normalized to sum to 1.
        let raw_w: Vec<f64> = first_row.iter().map(|v| v * v).collect();
        let sym_w: Vec<f64> = (0..n)
            .map(|i| (raw_w[i] + raw_w[n - 1 - i]) / 2.0)
            .collect();
        let w_sum: f64 = sym_w.iter().sum();
        let w: Vec<f64> = sym_w.iter().map(|v| v / w_sum).collect();

        Self { z, w }
    }

    /// Number of quadrature points.
    #[inline]
    pub fn len(&self) -> usize {
        self.z.len()
    }

    /// Whether the quadrature rule is empty (always false after construction).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.z.is_empty()
    }

    /// Iterator over `(z, w)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (f64, f64)> + '_ {
        self.z.iter().copied().zip(self.w.iter().copied())
    }
}

// ---------------------------------------------------------------------------
// Memoized global cache — mirrors Julia's `GHnormd` Dict
// ---------------------------------------------------------------------------

// We use a `Mutex<HashMap>` rather than a more exotic concurrent map since
// quadrature rules are constructed very infrequently and contention is
// negligible.

/// Global memoization cache for `GaussHermiteNormalized` instances.
///
/// The first call to `gh_norm(k)` computes the rule; subsequent calls for the
/// same `k` return a clone from the cache.
static GH_CACHE: std::sync::LazyLock<Mutex<HashMap<usize, GaussHermiteNormalized>>> =
    std::sync::LazyLock::new(|| {
        let mut m = HashMap::new();
        // Pre-populate the trivial cases that Julia hard-codes.
        m.insert(
            1,
            GaussHermiteNormalized {
                z: vec![0.0],
                w: vec![1.0],
            },
        );
        m.insert(
            2,
            GaussHermiteNormalized {
                z: vec![-1.0, 1.0],
                w: vec![0.5, 0.5],
            },
        );
        let sqrt3 = 3.0_f64.sqrt();
        m.insert(
            3,
            GaussHermiteNormalized {
                z: vec![-sqrt3, 0.0, sqrt3],
                w: vec![1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
            },
        );
        Mutex::new(m)
    });

/// Return a (possibly cached) `k`-point Gauss-Hermite quadrature rule on the
/// normalized scale.
///
/// This is the Rust equivalent of the Julia `GHnorm(k)` function. The result
/// is memoized so that repeated calls with the same `k` avoid redundant
/// eigendecompositions.
pub fn gh_norm(k: usize) -> GaussHermiteNormalized {
    let mut cache = GH_CACHE.lock().unwrap();
    cache
        .entry(k)
        .or_insert_with(|| GaussHermiteNormalized::new(k))
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_gh2() {
        let gh = gh_norm(2);
        assert_eq!(gh.z, vec![-1.0, 1.0]);
        assert_eq!(gh.w, vec![0.5, 0.5]);
    }

    #[test]
    fn test_gh2_memoized() {
        let a = gh_norm(2);
        let b = gh_norm(2);
        assert_eq!(a.z, b.z);
        assert_eq!(a.w, b.w);
    }

    #[test]
    fn test_gh9_weights_sum_to_one() {
        let gh = gh_norm(9);
        let wsum: f64 = gh.w.iter().sum();
        assert_relative_eq!(wsum, 1.0, epsilon = 1e-14);
        assert_eq!(gh.len(), 9);
    }

    #[test]
    fn test_gh1() {
        let gh = gh_norm(1);
        assert_eq!(gh.z, vec![0.0]);
        assert_eq!(gh.w, vec![1.0]);
    }

    #[test]
    fn test_gh3() {
        let gh = gh_norm(3);
        assert_eq!(gh.len(), 3);
        let sqrt3 = 3.0_f64.sqrt();
        assert_relative_eq!(gh.z[0], -sqrt3, epsilon = 1e-14);
        assert_relative_eq!(gh.z[1], 0.0, epsilon = 1e-14);
        assert_relative_eq!(gh.z[2], sqrt3, epsilon = 1e-14);
        assert_relative_eq!(gh.w[0], 1.0 / 6.0, epsilon = 1e-14);
        assert_relative_eq!(gh.w[1], 2.0 / 3.0, epsilon = 1e-14);
        assert_relative_eq!(gh.w[2], 1.0 / 6.0, epsilon = 1e-14);
    }

    #[test]
    fn test_symmetry() {
        let gh = gh_norm(7);
        let n = gh.len();
        for i in 0..n {
            // z should be antisymmetric
            assert_relative_eq!(gh.z[i], -gh.z[n - 1 - i], epsilon = 1e-12);
            // w should be symmetric
            assert_relative_eq!(gh.w[i], gh.w[n - 1 - i], epsilon = 1e-14);
        }
    }

    #[test]
    fn test_iter() {
        let gh = gh_norm(3);
        let pairs: Vec<(f64, f64)> = gh.iter().collect();
        assert_eq!(pairs.len(), 3);
        assert_relative_eq!(pairs[0].0, gh.z[0]);
        assert_relative_eq!(pairs[0].1, gh.w[0]);
    }
}
