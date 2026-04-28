//! Fixed-effects model matrix with pivoted QR rank detection.
//!
//! This is a port of `FeTerm{T,S}` from Julia's MixedModels.jl.
//! The pivoted QR decomposition is used to detect rank deficiency
//! and reorder columns so that the first `rank` columns form a
//! full-rank submatrix.

use nalgebra::DMatrix;

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

        // --- Column-pivoted QR via modified Gram-Schmidt with pivoting ---
        //
        // At each step we pick the column with the largest remaining norm,
        // swap it into position, and orthogonalise the remaining columns
        // against it. The rank is determined by checking when the residual
        // norm drops below a tolerance.

        let tol = (n.max(p) as f64) * f64::EPSILON * {
            // Estimate the largest column norm of the original matrix.
            (0..p).map(|j| x.column(j).norm()).fold(0.0_f64, f64::max)
        };

        // Work on a mutable copy.
        let mut qr = x.clone();
        let mut piv: Vec<usize> = (0..p).collect();
        let mut col_norms: Vec<f64> = (0..p).map(|j| qr.column(j).norm_squared()).collect();
        let mut rank = 0;

        for k in 0..p.min(n) {
            // Find the column with the largest remaining norm.
            let mut best = k;
            let mut best_norm = col_norms[k];
            for j in (k + 1)..p {
                if col_norms[j] > best_norm {
                    best = j;
                    best_norm = col_norms[j];
                }
            }

            // Check if the remaining norm is below tolerance.
            if best_norm.sqrt() <= tol {
                break;
            }

            // Swap columns k and best.
            if best != k {
                qr.swap_columns(k, best);
                piv.swap(k, best);
                col_norms.swap(k, best);
            }

            // Normalise the pivot column.
            let pivot_norm = qr.column(k).norm();
            if pivot_norm <= tol {
                break;
            }

            rank += 1;

            // Orthogonalise remaining columns against column k.
            // Extract the pivot column first.
            let pivot_col = qr.column(k).clone_owned();
            let norm_sq = pivot_col.norm_squared();

            for j in (k + 1)..p {
                let coeff = qr.column(j).dot(&pivot_col) / norm_sq;
                // qr.column_mut(j) -= coeff * pivot_col
                for i in 0..n {
                    qr[(i, j)] -= coeff * pivot_col[i];
                }
                // Update the running column norm.
                col_norms[j] = qr.column(j).norm_squared();
            }
        }

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
    fn test_zero_columns() {
        let x = DMatrix::zeros(5, 0);
        let cnames: Vec<String> = Vec::new();
        let fe = FeTerm::new(x, cnames);
        assert_eq!(fe.rank, 0);
        assert!(fe.is_full_rank()); // 0 == 0
    }
}
