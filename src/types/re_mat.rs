//! Random-effects model matrix for one grouping factor.
//!
//! This is a port of the random-effects term representation from
//! Julia's MixedModels.jl. Each `ReMat` corresponds to a single
//! grouping factor (e.g. `(1 + x | subject)`) and stores the
//! transposed model matrix, the relative covariance factor `Λ`,
//! and the sparse structure needed for the blocked Cholesky.

use crate::error::{MixedModelError, Result};
use nalgebra::{DMatrix, DVector};
use nalgebra_sparse::csc::CscMatrix;

/// Random-effects model matrix for one grouping factor.
///
/// Represents a term like `(1 + x | subject)` in a mixed-effects
/// formula. The model matrix `Z` is stored in transposed form
/// (`s × n`) for efficiency, where `s` is the vector size of the
/// random effect and `n` is the number of observations.
///
/// The relative covariance factor `Λ` is a lower-triangular `s × s`
/// matrix. The full random-effects covariance is `σ² Λ Λ'`, and the
/// random-effects design in the pseudo-data formulation is `Z Λ`.
#[derive(Debug, Clone)]
pub struct ReMat {
    /// Name of the grouping factor (e.g. "subject").
    pub grouping_name: String,

    /// Reference indices into `levels`, one per observation (0-based).
    /// Length equals the number of observations.
    pub refs: Vec<u32>,

    /// Unique level labels for the grouping factor.
    pub levels: Vec<String>,

    /// Column names for the random-effects terms
    /// (e.g. `["(Intercept)", "x"]`).
    pub cnames: Vec<String>,

    /// Transpose of the model matrix: dimension `s × n`, where
    /// `s = vsize` and `n = n_obs`.
    pub z: DMatrix<f64>,

    /// Weighted copy of `z`. Equal to `z` when there are no
    /// observation weights.
    pub wtz: DMatrix<f64>,

    /// Lower-triangular relative covariance factor, dimension `s × s`.
    /// For a scalar random effect (`vsize == 1`) this is a 1×1 matrix.
    pub lambda: DMatrix<f64>,

    /// Linear indices (column-major) of the free parameters within
    /// `lambda`. For a full lower-triangular `s × s` matrix these are
    /// the `s(s+1)/2` positions on and below the diagonal.
    pub inds: Vec<usize>,

    /// The transpose of the model matrix stored as a sparse CSC matrix
    /// of dimension `(vsize * n_levels) × n_obs`, suitable for forming
    /// cross-products in the blocked Cholesky.
    pub adj_a: CscMatrix<f64>,

    /// Scratch space for intermediate computations, same dimensions as `z`.
    pub scratch: DMatrix<f64>,

    /// Vector size of the random effect: 1 for a scalar random
    /// intercept, >1 for vector-valued (e.g. random intercept + slope).
    pub vsize: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReMatCovarianceKernel {
    FullCholesky,
    Diagonal,
}

impl ReMatCovarianceKernel {
    fn theta_indices(self, vsize: usize) -> Vec<usize> {
        match self {
            ReMatCovarianceKernel::FullCholesky => lower_triangular_indices(vsize),
            ReMatCovarianceKernel::Diagonal => diagonal_indices(vsize),
        }
    }

    fn zero_inactive_lambda(self, lambda: &mut DMatrix<f64>) {
        if self == ReMatCovarianceKernel::Diagonal {
            let n = lambda.nrows();
            for row in 0..n {
                for col in 0..n {
                    if row != col {
                        lambda[(row, col)] = 0.0;
                    }
                }
            }
        }
    }
}

impl ReMat {
    /// Construct a new `ReMat` with identity `Λ`.
    ///
    /// # Arguments
    ///
    /// * `grouping_name` - Name of the grouping factor.
    /// * `refs` - Per-observation indices into `levels` (0-based).
    /// * `levels` - Unique level labels.
    /// * `cnames` - Column names of the random-effects terms.
    /// * `z` - Transposed model matrix of dimension `vsize × n_obs`.
    ///
    /// # Panics
    ///
    /// * If `refs` contains an index >= `levels.len()`.
    /// * If `z.nrows() != cnames.len()`.
    /// * If `z.ncols() != refs.len()`.
    pub fn new(
        grouping_name: String,
        refs: Vec<u32>,
        levels: Vec<String>,
        cnames: Vec<String>,
        z: DMatrix<f64>,
    ) -> Self {
        let vsize = cnames.len();
        let n_obs = refs.len();
        let n_lvl = levels.len();

        assert_eq!(
            z.nrows(),
            vsize,
            "ReMat::new: z.nrows() ({}) must equal vsize (cnames.len() = {})",
            z.nrows(),
            vsize
        );
        assert_eq!(
            z.ncols(),
            n_obs,
            "ReMat::new: z.ncols() ({}) must equal n_obs (refs.len() = {})",
            z.ncols(),
            n_obs
        );
        for (i, &r) in refs.iter().enumerate() {
            assert!(
                (r as usize) < n_lvl,
                "ReMat::new: refs[{}] = {} is out of bounds for {} levels",
                i,
                r,
                n_lvl
            );
        }

        // Identity lambda (s × s)
        let lambda = DMatrix::identity(vsize, vsize);

        let inds = ReMatCovarianceKernel::FullCholesky.theta_indices(vsize);

        // Build the sparse adjoint matrix.
        let adj_a = build_sparse_adjoint(&refs, &z, vsize, n_lvl, n_obs);

        let wtz = z.clone();
        let scratch = DMatrix::zeros(vsize, n_obs);

        ReMat {
            grouping_name,
            refs,
            levels,
            cnames,
            z,
            wtz,
            lambda,
            inds,
            adj_a,
            scratch,
            vsize,
        }
    }

    /// Number of grouping-factor levels.
    pub fn n_levels(&self) -> usize {
        self.levels.len()
    }

    /// Total number of random-effect coefficients: `vsize × n_levels`.
    pub fn n_ranef(&self) -> usize {
        self.vsize * self.levels.len()
    }

    /// Number of free parameters in `Λ` (the θ vector for this term).
    pub fn n_theta(&self) -> usize {
        self.inds.len()
    }

    /// Number of observations.
    pub fn n_obs(&self) -> usize {
        self.refs.len()
    }

    /// Extract the free parameters from `Λ` into a vector.
    pub fn get_theta(&self) -> Vec<f64> {
        self.inds
            .iter()
            .map(|&idx| {
                let (row, col) = linear_to_subscript(idx, self.vsize);
                self.lambda[(row, col)]
            })
            .collect()
    }

    /// Install parameter values into `Λ`.
    pub fn set_theta(&mut self, v: &[f64]) -> Result<()> {
        let expected = self.n_theta();
        if v.len() != expected {
            return Err(MixedModelError::DimensionMismatch(format!(
                "ReMat::set_theta expected {expected} values, got {}",
                v.len()
            )));
        }
        for (k, &idx) in self.inds.iter().enumerate() {
            let (row, col) = linear_to_subscript(idx, self.vsize);
            self.lambda[(row, col)] = v[k];
        }
        Ok(())
    }

    /// Lower bounds on the θ parameters.
    ///
    /// Diagonal elements of `Λ` must be ≥ 0; off-diagonal elements are
    /// unconstrained (−∞). This matches Julia's MixedModels.jl convention.
    pub fn lower_bound(&self) -> Vec<f64> {
        self.inds
            .iter()
            .map(|&idx| {
                let (row, col) = linear_to_subscript(idx, self.vsize);
                if row == col {
                    0.0
                } else {
                    f64::NEG_INFINITY
                }
            })
            .collect()
    }

    /// Name of the grouping factor.
    pub fn fname(&self) -> &str {
        &self.grouping_name
    }

    /// Left-multiply `b` by `Λ'` (transpose of lambda) in place.
    ///
    /// Computes `b ← Λ' * b` where `Λ` is `vsize × vsize` and `b`
    /// has `vsize` rows. Operates on each `vsize`-row block.
    pub fn lmul_lambda(&self, b: &mut DMatrix<f64>) {
        if self.vsize == 1 {
            // Scalar case: just scale by lambda[0,0].
            *b *= self.lambda[(0, 0)];
            return;
        }

        // General case: b ← Λ' * b
        let lambda_t = self.lambda.transpose();
        let result = &lambda_t * &*b;
        b.copy_from(&result);
    }

    /// Right-multiply `a` by `Λ` in place.
    ///
    /// Computes `a ← a * Λ` where `Λ` is `vsize × vsize` and `a`
    /// has `vsize` columns. Operates on each `vsize`-column block.
    pub fn rmul_lambda(&self, a: &mut DMatrix<f64>) {
        if self.vsize == 1 {
            // Scalar case: just scale.
            *a *= self.lambda[(0, 0)];
            return;
        }

        // General case: a ← a * Λ
        let result = &*a * &self.lambda;
        a.copy_from(&result);
    }

    /// Return the full sparse representation of `Λ' Z'`.
    ///
    /// Dimensions: `n_ranef × n_obs` stored as CSC.
    pub fn sparse(&self) -> CscMatrix<f64> {
        // Rebuild the sparse adjoint with Λ applied.
        let n_obs = self.n_obs();
        let total_re = self.n_ranef();
        let s = self.vsize;

        let mut triplets: Vec<(usize, usize, f64)> = Vec::new();

        for obs in 0..n_obs {
            let lvl = self.refs[obs] as usize;
            let row_start = lvl * s;
            // The s × 1 slice of wtz for this observation.
            let z_col = self.wtz.column(obs);
            // Multiply by Λ' to get the s × 1 result.
            let lam_t_z = self.lambda.transpose() * z_col;

            for r in 0..s {
                let val = lam_t_z[r];
                if val != 0.0 {
                    triplets.push((row_start + r, obs, val));
                }
            }
        }

        triplet_to_csc(total_re, n_obs, &triplets)
    }

    /// Re-weight the model matrix by square-root observation weights.
    ///
    /// Sets `wtz[:, i] = z[:, i] * sqrtwts[i]` for every observation.
    ///
    /// # Panics
    ///
    /// Panics if `sqrtwts.len() != self.n_obs()`.
    pub fn reweight(&mut self, sqrtwts: &DVector<f64>) {
        let n = self.n_obs();
        assert_eq!(
            sqrtwts.len(),
            n,
            "ReMat::reweight: sqrtwts length ({}) must match n_obs ({})",
            sqrtwts.len(),
            n
        );

        self.wtz = self.z.clone();
        for j in 0..n {
            let w = sqrtwts[j];
            for i in 0..self.vsize {
                self.wtz[(i, j)] *= w;
            }
        }
    }

    /// Restrict `Λ` to diagonal (zero-correlation model).
    ///
    /// Sets all off-diagonal elements of `Λ` to zero and updates `inds`
    /// to contain only the diagonal positions.
    pub fn zerocorr(&mut self) {
        self.set_covariance_kernel(ReMatCovarianceKernel::Diagonal);
    }

    fn set_covariance_kernel(&mut self, kernel: ReMatCovarianceKernel) {
        kernel.zero_inactive_lambda(&mut self.lambda);
        self.inds = kernel.theta_indices(self.vsize);
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Return column-major linear indices of the lower-triangular elements
/// of an `s × s` matrix (including diagonal), ordered by column.
fn lower_triangular_indices(s: usize) -> Vec<usize> {
    let mut inds = Vec::with_capacity(s * (s + 1) / 2);
    for col in 0..s {
        for row in col..s {
            inds.push(col * s + row); // column-major index
        }
    }
    inds
}

fn diagonal_indices(s: usize) -> Vec<usize> {
    (0..s).map(|k| k * s + k).collect()
}

/// Convert a column-major linear index to (row, col) subscripts for
/// an `nrows`-row matrix.
fn linear_to_subscript(idx: usize, nrows: usize) -> (usize, usize) {
    let col = idx / nrows;
    let row = idx % nrows;
    (row, col)
}

/// Build the sparse CSC adjoint matrix `Z'` of dimension
/// `(vsize * n_levels) × n_obs`.
///
/// Each observation contributes a `vsize`-tall column segment placed
/// at the row block corresponding to its grouping-factor level.
fn build_sparse_adjoint(
    refs: &[u32],
    z: &DMatrix<f64>,
    vsize: usize,
    n_levels: usize,
    n_obs: usize,
) -> CscMatrix<f64> {
    let nrows = vsize * n_levels;
    let mut col_offsets = Vec::with_capacity(n_obs + 1);
    let mut row_indices = Vec::with_capacity(vsize * n_obs);
    let mut values = Vec::with_capacity(vsize * n_obs);

    col_offsets.push(0);
    for (obs, &lvl) in refs.iter().enumerate() {
        let row_start = (lvl as usize) * vsize;
        for r in 0..vsize {
            let val = z[(r, obs)];
            if val != 0.0 {
                row_indices.push(row_start + r);
                values.push(val);
            }
        }
        col_offsets.push(row_indices.len());
    }

    CscMatrix::try_from_csc_data(nrows, n_obs, col_offsets, row_indices, values)
        .expect("ReMat::new constructs sorted CSC adjoint data")
}

/// Build a CSC matrix from triplets. Duplicate entries at the same
/// position are summed.
fn triplet_to_csc(nrows: usize, ncols: usize, triplets: &[(usize, usize, f64)]) -> CscMatrix<f64> {
    use nalgebra_sparse::CooMatrix;

    let mut coo = CooMatrix::new(nrows, ncols);
    for &(r, c, v) in triplets {
        coo.push(r, c, v);
    }
    CscMatrix::from(&coo)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    /// Helper: build a simple scalar ReMat with 3 levels and 6 observations.
    fn make_scalar_remat() -> ReMat {
        // 6 observations, 3 levels, scalar random intercept
        let refs = vec![0, 0, 1, 1, 2, 2];
        let levels = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let cnames = vec!["(Intercept)".to_string()];
        // z is 1 × 6 (all ones for intercept)
        let z = DMatrix::from_row_slice(1, 6, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        ReMat::new("group".to_string(), refs, levels, cnames, z)
    }

    /// Helper: build a vector-valued ReMat (intercept + slope) with 2 levels.
    fn make_vector_remat() -> ReMat {
        // 4 observations, 2 levels, vsize = 2
        let refs = vec![0, 0, 1, 1];
        let levels = vec!["s1".to_string(), "s2".to_string()];
        let cnames = vec!["(Intercept)".to_string(), "x".to_string()];
        // z is 2 × 4
        let z = DMatrix::from_row_slice(
            2,
            4,
            &[
                1.0, 1.0, 1.0, 1.0, // intercept row
                0.1, 0.2, 0.3, 0.4, // slope row
            ],
        );
        ReMat::new("subject".to_string(), refs, levels, cnames, z)
    }

    #[test]
    fn test_scalar_construction() {
        let re = make_scalar_remat();
        assert_eq!(re.vsize, 1);
        assert_eq!(re.n_levels(), 3);
        assert_eq!(re.n_ranef(), 3);
        assert_eq!(re.n_theta(), 1);
        assert_eq!(re.n_obs(), 6);
        assert_eq!(re.fname(), "group");
    }

    #[test]
    fn test_vector_construction() {
        let re = make_vector_remat();
        assert_eq!(re.vsize, 2);
        assert_eq!(re.n_levels(), 2);
        assert_eq!(re.n_ranef(), 4);
        assert_eq!(re.n_theta(), 3); // 2×2 lower tri = 3
        assert_eq!(re.n_obs(), 4);
    }

    #[test]
    fn test_get_set_theta() {
        let mut re = make_vector_remat();
        // Initial lambda is identity, so theta = [1, 0, 1]
        let theta = re.get_theta();
        assert_eq!(theta.len(), 3);
        assert!((theta[0] - 1.0).abs() < 1e-12); // lambda[0,0]
        assert!((theta[1] - 0.0).abs() < 1e-12); // lambda[1,0]
        assert!((theta[2] - 1.0).abs() < 1e-12); // lambda[1,1]

        // Set new theta
        re.set_theta(&[2.0, 0.5, 3.0]).unwrap();
        let theta2 = re.get_theta();
        assert!((theta2[0] - 2.0).abs() < 1e-12);
        assert!((theta2[1] - 0.5).abs() < 1e-12);
        assert!((theta2[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_lower_bound() {
        let re = make_vector_remat();
        let lb = re.lower_bound();
        assert_eq!(lb.len(), 3);
        assert!((lb[0] - 0.0).abs() < 1e-12); // diagonal
        assert!(lb[1] == f64::NEG_INFINITY); // off-diagonal
        assert!((lb[2] - 0.0).abs() < 1e-12); // diagonal
    }

    #[test]
    fn test_reweight() {
        let mut re = make_scalar_remat();
        let wts = DVector::from_column_slice(&[0.5, 1.0, 1.5, 2.0, 2.5, 3.0]);
        re.reweight(&wts);
        // Check first observation: wtz[0,0] = z[0,0] * 0.5 = 0.5
        assert!((re.wtz[(0, 0)] - 0.5).abs() < 1e-12);
        // Check last observation: wtz[0,5] = z[0,5] * 3.0 = 3.0
        assert!((re.wtz[(0, 5)] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_zerocorr() {
        let mut re = make_vector_remat();
        re.set_theta(&[2.0, 0.5, 3.0]).unwrap();
        re.zerocorr();
        // Off-diagonal should be zero
        assert!((re.lambda[(1, 0)] - 0.0).abs() < 1e-12);
        // Only diagonal indices remain
        assert_eq!(re.n_theta(), 2);
        let theta = re.get_theta();
        assert!((theta[0] - 2.0).abs() < 1e-12);
        assert!((theta[1] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_covariance_kernel_patterns_own_theta_indices() {
        assert_eq!(
            ReMatCovarianceKernel::FullCholesky.theta_indices(3),
            vec![0, 1, 2, 4, 5, 8]
        );
        assert_eq!(
            ReMatCovarianceKernel::Diagonal.theta_indices(3),
            vec![0, 4, 8]
        );
    }

    #[test]
    fn test_diagonal_kernel_preserves_diagonal_theta_values() {
        let mut re = make_vector_remat();
        re.set_theta(&[2.0, 0.5, 3.0]).unwrap();

        re.set_covariance_kernel(ReMatCovarianceKernel::Diagonal);

        assert_eq!(re.inds, vec![0, 3]);
        assert_eq!(re.get_theta(), vec![2.0, 3.0]);
        assert_eq!(re.lower_bound(), vec![0.0, 0.0]);
        assert_eq!(re.lambda[(1, 0)], 0.0);
    }

    #[test]
    fn test_remat_set_theta_returns_err_on_length_mismatch() {
        let mut re = make_vector_remat();
        let before = re.get_theta();

        let err = re.set_theta(&[2.0, 0.5]).unwrap_err();

        assert!(matches!(err, MixedModelError::DimensionMismatch(_)));
        assert_eq!(re.get_theta(), before);
    }

    #[test]
    fn test_lmul_lambda_scalar() {
        let re = make_scalar_remat();
        let mut b = DMatrix::from_row_slice(1, 3, &[1.0, 2.0, 3.0]);
        re.lmul_lambda(&mut b);
        // lambda is identity (1×1 with value 1.0), so b unchanged
        assert!((b[(0, 0)] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_sparse_dimensions() {
        let re = make_scalar_remat();
        let sp = re.sparse();
        assert_eq!(sp.nrows(), 3); // n_ranef = 3
        assert_eq!(sp.ncols(), 6); // n_obs = 6
    }
}
