//! Fixed-effects model matrix concatenated with the response vector.
//!
//! This is a port of `FeMat{T,S}` from Julia's MixedModels.jl.
//! It stores the horizontal concatenation `[X | y]` (where `X` is the
//! full-rank portion of the design matrix and `y` is the response)
//! together with a weighted copy used during IRLS or weighted
//! least-squares.

use nalgebra::{DMatrix, DVector};

use super::fe_term::FeTerm;

/// Fixed-effects model matrix `[X | y]` with optional weighting.
///
/// The `xy` field holds the original concatenation of the full-rank
/// columns of the design matrix with the response vector. The `wtxy`
/// field holds a copy that has been row-scaled by observation weights;
/// when there are no weights `wtxy` is identical to `xy`.
#[derive(Debug, Clone)]
pub struct FeMat {
    /// Original `[X | y]` matrix of dimension n_obs × (rank + 1).
    pub xy: DMatrix<f64>,

    /// Weighted copy of `xy`. Each row `i` is scaled by `sqrtwts[i]`.
    /// Equals `xy` when no weights have been applied.
    pub wtxy: DMatrix<f64>,
}

impl FeMat {
    /// Create a new `FeMat` by concatenating the full-rank columns of
    /// `feterm` with the response vector `y`.
    ///
    /// # Arguments
    ///
    /// * `feterm` - The fixed-effects term (after pivoted QR rank detection).
    /// * `y` - The response vector, length must equal `feterm.n_obs()`.
    ///
    /// # Panics
    ///
    /// Panics if `y.len() != feterm.n_obs()`.
    pub fn new(feterm: &FeTerm, y: &DVector<f64>) -> Self {
        let n = feterm.n_obs();
        assert_eq!(
            y.len(),
            n,
            "FeMat::new: y length ({}) must match n_obs ({})",
            y.len(),
            n
        );

        let x_full = feterm.full_rank_x();
        let p = x_full.ncols();

        // Build [X | y] with dimensions n × (p + 1).
        let mut xy = DMatrix::zeros(n, p + 1);
        for j in 0..p {
            xy.set_column(j, &x_full.column(j));
        }
        xy.set_column(p, &y.column(0));

        let wtxy = xy.clone();

        FeMat { xy, wtxy }
    }

    /// Re-weight the model matrix by square-root observation weights.
    ///
    /// Sets `wtxy[i, :] = sqrtwts[i] * xy[i, :]` for every row `i`.
    ///
    /// # Arguments
    ///
    /// * `sqrtwts` - Square roots of the observation weights, length n_obs.
    ///
    /// # Panics
    ///
    /// Panics if `sqrtwts.len() != self.n_obs()`.
    pub fn reweight(&mut self, sqrtwts: &DVector<f64>) {
        let n = self.xy.nrows();
        assert_eq!(
            sqrtwts.len(),
            n,
            "FeMat::reweight: sqrtwts length ({}) must match n_obs ({})",
            sqrtwts.len(),
            n
        );

        self.wtxy = self.xy.clone();
        for i in 0..n {
            let w = sqrtwts[i];
            for j in 0..self.wtxy.ncols() {
                self.wtxy[(i, j)] *= w;
            }
        }
    }

    /// Compute the cross-product matrix `wtxy' * wtxy`.
    ///
    /// This yields a (rank+1) × (rank+1) symmetric matrix whose
    /// upper-left block is `X'X`, the last row/column involves `X'y`
    /// and `y'y`.
    pub fn cross_product(&self) -> DMatrix<f64> {
        self.wtxy.transpose() * &self.wtxy
    }

    /// Number of observations (rows).
    pub fn n_obs(&self) -> usize {
        self.xy.nrows()
    }

    /// Number of columns, including the response (rank + 1).
    pub fn n_cols(&self) -> usize {
        self.xy.ncols()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    fn make_test_feterm() -> FeTerm {
        let x = DMatrix::from_row_slice(4, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let cnames = vec!["x1".to_string(), "x2".to_string()];
        FeTerm::new(x, cnames)
    }

    #[test]
    fn test_new() {
        let fe = make_test_feterm();
        let y = DVector::from_column_slice(&[10.0, 20.0, 30.0, 40.0]);
        let femat = FeMat::new(&fe, &y);
        assert_eq!(femat.n_obs(), 4);
        // rank should be 2, so ncols = 3 (2 + response)
        assert_eq!(femat.n_cols(), fe.rank + 1);
    }

    #[test]
    fn test_cross_product_unweighted() {
        let fe = make_test_feterm();
        let y = DVector::from_column_slice(&[1.0, 1.0, 1.0, 1.0]);
        let femat = FeMat::new(&fe, &y);
        let cp = femat.cross_product();
        // cross product should be symmetric
        assert_eq!(cp.nrows(), cp.ncols());
        for i in 0..cp.nrows() {
            for j in 0..cp.ncols() {
                assert!(
                    (cp[(i, j)] - cp[(j, i)]).abs() < 1e-12,
                    "cross product not symmetric at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_reweight() {
        let fe = make_test_feterm();
        let y = DVector::from_column_slice(&[1.0, 2.0, 3.0, 4.0]);
        let mut femat = FeMat::new(&fe, &y);

        let wts = DVector::from_column_slice(&[0.5, 1.0, 1.5, 2.0]);
        femat.reweight(&wts);

        // Check that first row is scaled by 0.5
        for j in 0..femat.n_cols() {
            assert!(
                (femat.wtxy[(0, j)] - 0.5 * femat.xy[(0, j)]).abs() < 1e-12,
                "reweight incorrect at row 0, col {}",
                j
            );
        }
    }
}
