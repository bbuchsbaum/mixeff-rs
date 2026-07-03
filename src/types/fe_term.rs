//! Fixed-effects model matrix with pivoted QR rank detection.
//!
//! This is a port of `FeTerm{T,S}` from Julia's MixedModels.jl.
//! The pivoted QR decomposition is used to detect rank deficiency
//! and reorder columns so that the first `rank` columns form a
//! full-rank submatrix.

use nalgebra::DMatrix;

use crate::linalg::pivot::stats_rank;

/// Fixed-effects model matrix with column pivoting for rank detection.
///
/// Stores the (possibly reordered) model matrix along with pivot
/// information from a rank-revealing QR factorization. When the
/// original design matrix is rank-deficient the redundant columns
/// are moved to the end so that `full_rank_x()` returns only the
/// linearly independent columns.
#[derive(Debug, Clone)]
pub struct FeTerm {
    /// The (possibly pivoted) model matrix, with columns reordered
    /// so that the first `rank` columns are linearly independent.
    pub x: DMatrix<f64>,

    /// Pivot indices from the rank-revealing QR decomposition.
    /// `piv[i]` gives the original column index of the i-th column
    /// in the pivoted matrix.
    pub piv: Vec<usize>,

    /// Computational rank of the model matrix, i.e. the number of
    /// linearly independent columns detected by the pivoted QR.
    pub rank: usize,

    /// Column names in pivoted order. The first `rank` names
    /// correspond to the linearly independent columns.
    pub cnames: Vec<String>,
}

impl FeTerm {
    /// Create a new `FeTerm` from a design matrix and column names.
    ///
    /// Performs a column-pivoted QR decomposition to detect the
    /// computational rank. Columns are reordered according to the
    /// pivot so that the first `rank` columns are linearly independent.
    ///
    /// # Arguments
    ///
    /// * `x` - The fixed-effects design matrix (n_obs × p).
    /// * `cnames` - Column names for the design matrix. Must have
    ///   length equal to `x.ncols()`.
    ///
    /// # Panics
    ///
    /// Panics if `cnames.len() != x.ncols()`.
    pub fn new(x: DMatrix<f64>, cnames: Vec<String>) -> Self {
        assert_eq!(
            cnames.len(),
            x.ncols(),
            "FeTerm::new: cnames length ({}) must match number of columns ({})",
            cnames.len(),
            x.ncols()
        );

        let (n, p) = (x.nrows(), x.ncols());

        if p == 0 {
            return FeTerm {
                x,
                piv: Vec::new(),
                rank: 0,
                cnames,
            };
        }

        let (rank, piv) = stats_rank(&x);

        // Build the pivoted X (take columns from the original x in pivot order).
        let mut pivoted_x = DMatrix::zeros(n, p);
        for (new_j, &orig_j) in piv.iter().enumerate() {
            pivoted_x.set_column(new_j, &x.column(orig_j));
        }

        // Reorder column names according to pivot.
        let pivoted_cnames: Vec<String> = piv.iter().map(|&j| cnames[j].clone()).collect();

        FeTerm {
            x: pivoted_x,
            piv,
            rank,
            cnames: pivoted_cnames,
        }
    }

    /// Create a `FeTerm` from a design matrix already certified to be
    /// full column rank, skipping the pivoted-QR pass entirely.
    ///
    /// This is the streamed fixed-design seam: when a
    /// `crate::linalg::pivot::gram_full_rank_certificate` run on the
    /// (never-densified) Gram matrix certifies full rank, the result is
    /// byte-identical to what [`FeTerm::new`] would produce — the
    /// full-rank early return of `stats_rank` uses the identity
    /// permutation and leaves the matrix untouched — so no rank
    /// detection and no pivoted copy are needed. Callers must NOT use
    /// this constructor without such a certificate; an undetected rank
    /// deficiency would silently corrupt downstream inference.
    ///
    /// # Panics
    ///
    /// Panics if `cnames.len() != x.ncols()`.
    pub fn with_certified_full_rank(x: DMatrix<f64>, cnames: Vec<String>) -> Self {
        assert_eq!(
            cnames.len(),
            x.ncols(),
            "FeTerm::with_certified_full_rank: cnames length ({}) must match number of columns ({})",
            cnames.len(),
            x.ncols()
        );

        let p = x.ncols();
        FeTerm {
            x,
            piv: (0..p).collect(),
            rank: p,
            cnames,
        }
    }

    /// Return a view of the first `rank` (linearly independent) columns.
    ///
    /// This is the submatrix used in fitting; redundant columns are excluded.
    pub fn full_rank_x(&self) -> nalgebra::DMatrixView<'_, f64> {
        self.x.columns(0, self.rank)
    }

    /// Whether the original design matrix was full column-rank.
    pub fn is_full_rank(&self) -> bool {
        self.rank == self.x.ncols()
    }

    /// Number of observations (rows).
    pub fn n_obs(&self) -> usize {
        self.x.nrows()
    }

    /// Total number of columns (before rank truncation).
    pub fn n_cols(&self) -> usize {
        self.x.ncols()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    #[test]
    fn test_full_rank() {
        // 3×2 full-rank matrix
        let x = DMatrix::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let cnames = vec!["a".to_string(), "b".to_string()];
        let fe = FeTerm::new(x, cnames);
        assert_eq!(fe.rank, 2);
        assert!(fe.is_full_rank());
        assert_eq!(fe.n_obs(), 3);
        assert_eq!(fe.n_cols(), 2);
    }

    #[test]
    fn test_rank_deficient() {
        // 3×3 matrix where col3 = col1 + col2
        let x = DMatrix::from_row_slice(
            3,
            3,
            &[
                1.0, 0.0, 1.0, //
                0.0, 1.0, 1.0, //
                1.0, 1.0, 2.0, //
            ],
        );
        let cnames = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let fe = FeTerm::new(x, cnames);
        assert_eq!(fe.rank, 2);
        assert!(!fe.is_full_rank());
        assert_eq!(fe.full_rank_x().ncols(), 2);
    }

    #[test]
    fn test_feterm_rank_matches_stats_rank() {
        let x = DMatrix::from_row_slice(
            4,
            3,
            &[
                1.0, 0.0, 1.0, //
                1.0, 1.0, 2.0, //
                1.0, 2.0, 3.0, //
                1.0, 3.0, 4.0, //
            ],
        );
        let cnames = vec![
            "(Intercept)".to_string(),
            "x".to_string(),
            "intercept_plus_x".to_string(),
        ];
        let (rank, piv) = stats_rank(&x);

        let fe = FeTerm::new(x, cnames);

        assert_eq!(fe.rank, rank);
        assert_eq!(fe.piv, piv);
    }

    #[test]
    fn test_feterm_rank_intercept_preserving() {
        let x = DMatrix::from_row_slice(
            4,
            3,
            &[
                1.0, 0.0, 1.0, //
                1.0, 1.0, 2.0, //
                1.0, 2.0, 3.0, //
                1.0, 3.0, 4.0, //
            ],
        );
        let cnames = vec![
            "(Intercept)".to_string(),
            "x".to_string(),
            "intercept_plus_x".to_string(),
        ];

        let fe = FeTerm::new(x, cnames);

        assert_eq!(fe.rank, 2);
        assert_eq!(fe.piv[0], 0);
        assert_eq!(fe.cnames[0], "(Intercept)");
        assert!(fe.piv[0..fe.rank].windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn certified_full_rank_matches_qr_constructor_on_full_rank_input() {
        let x = DMatrix::from_row_slice(
            4,
            3,
            &[
                1.0, 0.0, 2.0, //
                1.0, 1.0, 0.5, //
                1.0, 2.0, -1.0, //
                1.0, 3.0, 4.0, //
            ],
        );
        let cnames = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let via_qr = FeTerm::new(x.clone(), cnames.clone());
        let via_cert = FeTerm::with_certified_full_rank(x, cnames);
        assert_eq!(via_cert.rank, via_qr.rank);
        assert_eq!(via_cert.piv, via_qr.piv);
        assert_eq!(via_cert.cnames, via_qr.cnames);
        assert_eq!(via_cert.x, via_qr.x);
    }

    #[test]
    fn test_zero_columns() {
        let x = DMatrix::zeros(5, 0);
        let cnames: Vec<String> = Vec::new();
        let fe = FeTerm::new(x, cnames);
        assert_eq!(fe.rank, 0);
        assert!(fe.is_full_rank()); // 0 == 0
    }
}
