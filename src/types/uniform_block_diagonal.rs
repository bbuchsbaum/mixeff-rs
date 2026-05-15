//! Uniform (homogeneous) block diagonal matrices.
//!
//! A `UniformBlockDiagonal` is a block diagonal matrix where every diagonal
//! block has the same dimensions `m x m`. It is stored as a `Vec` of `k`
//! dense `m x m` matrices rather than a full `(m*k) x (m*k)` matrix.
//!
//! This is a Rust port of `UniformBlockDiagonal{T}` from MixedModels.jl.

use nalgebra::DMatrix;

/// A homogeneous block diagonal matrix.
///
/// `k` diagonal blocks each of size `block_size x block_size`.
/// Off-diagonal blocks are implicitly zero.
///
/// # Fields
///
/// * `blocks` - the `k` square diagonal blocks stored as dense matrices
/// * `block_size` - the row/column dimension of each block (`m`)
#[derive(Debug, Clone)]
pub struct UniformBlockDiagonal {
    /// The diagonal blocks, each of size `block_size x block_size`.
    pub blocks: Vec<DMatrix<f64>>,
    /// Row and column dimension of each block.
    pub block_size: usize,
}

impl UniformBlockDiagonal {
    /// Create a `UniformBlockDiagonal` from a vector of identically-sized
    /// square matrices.
    ///
    /// # Panics
    ///
    /// Panics if `blocks` is empty, any block is not square, or blocks differ
    /// in size.
    pub fn new(blocks: Vec<DMatrix<f64>>) -> Self {
        assert!(!blocks.is_empty(), "blocks must be non-empty");
        let m = blocks[0].nrows();
        assert_eq!(blocks[0].ncols(), m, "each block must be square");
        for (i, blk) in blocks.iter().enumerate().skip(1) {
            assert_eq!(
                (blk.nrows(), blk.ncols()),
                (m, m),
                "block {i} has inconsistent dimensions"
            );
        }
        Self {
            blocks,
            block_size: m,
        }
    }

    /// Create a `UniformBlockDiagonal` from a flat column-major slice
    /// interpreted as a 3-D array of dimensions `(m, m, k)`.
    ///
    /// The slice length must equal `m * m * k`.
    pub fn from_3d_slice(data: &[f64], m: usize, k: usize) -> Self {
        assert_eq!(data.len(), m * m * k, "data length must equal m * m * k");
        let blocks: Vec<DMatrix<f64>> = (0..k)
            .map(|blk_idx| {
                let offset = blk_idx * m * m;
                DMatrix::from_column_slice(m, m, &data[offset..offset + m * m])
            })
            .collect();
        Self {
            blocks,
            block_size: m,
        }
    }

    /// Number of diagonal blocks.
    #[inline]
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Dimensions of the full (virtual) matrix: `(m*k, m*k)`.
    #[inline]
    pub fn size(&self) -> (usize, usize) {
        let n = self.block_size * self.num_blocks();
        (n, n)
    }

    /// Element access with zero-based `(i, j)` indices.
    ///
    /// Returns `0.0` for any position that falls outside a diagonal block.
    ///
    /// # Panics
    ///
    /// Panics if `(i, j)` is out of bounds.
    pub fn get(&self, i: usize, j: usize) -> f64 {
        let (rows, cols) = self.size();
        assert!(
            i < rows && j < cols,
            "index ({i}, {j}) out of bounds for {}x{} matrix",
            rows,
            cols
        );
        let m = self.block_size;
        let i_blk = i / m;
        let j_blk = j / m;
        if i_blk == j_blk {
            let i_off = i % m;
            let j_off = j % m;
            self.blocks[i_blk][(i_off, j_off)]
        } else {
            0.0
        }
    }

    /// Convert to a full dense matrix.
    pub fn to_dense(&self) -> DMatrix<f64> {
        let (rows, cols) = self.size();
        let mut out = DMatrix::zeros(rows, cols);
        self.copy_to_dense(&mut out);
        out
    }

    /// Copy into a pre-allocated dense matrix, zeroing off-diagonal blocks.
    ///
    /// # Panics
    ///
    /// Panics if `dest` does not have the correct dimensions.
    pub fn copy_to_dense(&self, dest: &mut DMatrix<f64>) {
        let (rows, cols) = self.size();
        assert_eq!(
            (dest.nrows(), dest.ncols()),
            (rows, cols),
            "destination matrix dimension mismatch"
        );
        dest.fill(0.0);
        let m = self.block_size;
        for (k, blk) in self.blocks.iter().enumerate() {
            let row_off = k * m;
            let col_off = k * m;
            for j in 0..m {
                for i in 0..m {
                    dest[(row_off + i, col_off + j)] = blk[(i, j)];
                }
            }
        }
    }

    /// Create a deep copy, optionally converting element values via a
    /// mapping function.
    ///
    /// Since we only support `f64`, this is effectively a clone, but the
    /// signature mirrors `copy_oftype` from the Julia source and allows
    /// in-place element transformation (e.g., rounding).
    pub fn copy_with<F: Fn(f64) -> f64>(&self, f: F) -> Self {
        let blocks = self.blocks.iter().map(|blk| blk.map(&f)).collect();
        Self {
            blocks,
            block_size: self.block_size,
        }
    }
}

impl std::ops::Index<(usize, usize)> for UniformBlockDiagonal {
    type Output = f64;

    /// Index with zero-based `(row, col)`.
    ///
    /// **Note:** because off-block elements are always `0.0`, this returns a
    /// reference to a static zero for those positions. It returns a reference
    /// into the block storage for on-block elements.
    fn index(&self, (i, j): (usize, usize)) -> &f64 {
        let m = self.block_size;
        let i_blk = i / m;
        let j_blk = j / m;
        if i_blk == j_blk {
            let i_off = i % m;
            let j_off = j % m;
            &self.blocks[i_blk][(i_off, j_off)]
        } else {
            // Return a reference to a static zero for off-block positions.
            static ZERO: f64 = 0.0;
            &ZERO
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Build the example from the Julia tests:
    /// reshape(1.0:12.0, (2, 2, 3))
    /// In Julia column-major order that gives:
    ///   block 0: [[1,2],[3,4]]
    ///   block 1: [[5,6],[7,8]]
    ///   block 2: [[9,10],[11,12]]
    fn example_2x2x3() -> UniformBlockDiagonal {
        let data: Vec<f64> = (1..=12).map(|v| v as f64).collect();
        UniformBlockDiagonal::from_3d_slice(&data, 2, 3)
    }

    #[test]
    fn test_size() {
        let ubd = example_2x2x3();
        assert_eq!(ubd.size(), (6, 6));
        assert_eq!(ubd.block_size, 2);
        assert_eq!(ubd.num_blocks(), 3);
    }

    #[test]
    fn test_element_access() {
        let ubd = example_2x2x3();
        // block 0 occupies rows 0..2, cols 0..2
        // In column-major: data[0]=1, data[1]=2, data[2]=3, data[3]=4
        // So block 0 = [[1,3],[2,4]]  (nalgebra column-major)
        assert_relative_eq!(ubd.get(0, 0), 1.0);
        assert_relative_eq!(ubd.get(1, 0), 2.0);
        assert_relative_eq!(ubd.get(0, 1), 3.0);
        assert_relative_eq!(ubd.get(1, 1), 4.0);

        // Off-block element
        assert_relative_eq!(ubd.get(2, 0), 0.0);

        // block 1
        assert_relative_eq!(ubd.get(2, 2), 5.0);

        // block 2, element (4,5) maps to block 2, local (0,1) = 11
        assert_relative_eq!(ubd.get(4, 5), 11.0);
    }

    #[test]
    fn test_to_dense() {
        let ubd = example_2x2x3();
        let dense = ubd.to_dense();
        assert_eq!(dense.nrows(), 6);
        assert_eq!(dense.ncols(), 6);

        // Check a known off-block zero
        assert_relative_eq!(dense[(2, 0)], 0.0);

        // Check a known on-block value
        assert_relative_eq!(dense[(0, 0)], 1.0);
        assert_relative_eq!(dense[(4, 5)], 11.0);
    }

    #[test]
    fn test_copy_to_dense() {
        let ubd = example_2x2x3();
        let mut dest = DMatrix::from_element(6, 6, 999.0);
        ubd.copy_to_dense(&mut dest);
        // Off-block should be zeroed
        assert_relative_eq!(dest[(3, 0)], 0.0);
        assert_relative_eq!(dest[(0, 0)], 1.0);
    }

    #[test]
    fn test_index_trait() {
        let ubd = example_2x2x3();
        assert_relative_eq!(ubd[(0, 0)], 1.0);
        assert_relative_eq!(ubd[(2, 0)], 0.0);
    }

    #[test]
    fn test_copy_with_identity() {
        let ubd = example_2x2x3();
        let ubd2 = ubd.copy_with(|x| x);
        assert_eq!(ubd2.size(), ubd.size());
        assert_relative_eq!(ubd2.get(0, 0), ubd.get(0, 0));
    }

    #[test]
    fn test_new_from_blocks() {
        let b1 = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let b2 = DMatrix::from_row_slice(2, 2, &[5.0, 6.0, 7.0, 8.0]);
        let ubd = UniformBlockDiagonal::new(vec![b1, b2]);
        assert_eq!(ubd.size(), (4, 4));
        assert_relative_eq!(ubd.get(0, 0), 1.0);
        assert_relative_eq!(ubd.get(2, 2), 5.0);
        assert_relative_eq!(ubd.get(0, 2), 0.0);
    }

    #[test]
    #[should_panic]
    fn test_out_of_bounds() {
        let ubd = example_2x2x3();
        let _ = ubd.get(6, 0);
    }
}
