//! Ragged (jagged) array for accumulating values by group.
//!
//! A `RaggedArray` pairs a flat data vector with a parallel vector of group
//! indices. It is used in MixedModels.jl to accumulate deviance components
//! across levels of a grouping factor.
//!
//! This is a Rust port of `RaggedArray{T,I}` from MixedModels.jl.

/// A ragged array: a flat vector of values with a parallel vector of group
/// indices.
///
/// Each element `data[i]` belongs to the group identified by `refs[i]`.
/// Group indices are zero-based and must satisfy `refs[i] < n_groups`.
///
/// # Fields
///
/// * `data` - the flat vector of values
/// * `refs` - group index for each value (zero-based)
/// * `n_groups` - total number of groups
#[derive(Debug, Clone)]
pub struct RaggedArray {
    /// The flat data vector.
    pub data: Vec<f64>,
    /// Group index for each element (zero-based).
    pub refs: Vec<usize>,
    /// Total number of groups.
    pub n_groups: usize,
}

impl RaggedArray {
    /// Create a new `RaggedArray`.
    ///
    /// # Panics
    ///
    /// Panics if `data` and `refs` have different lengths, or if any index
    /// in `refs` is `>= n_groups`.
    pub fn new(data: Vec<f64>, refs: Vec<usize>, n_groups: usize) -> Self {
        assert_eq!(
            data.len(),
            refs.len(),
            "data and refs must have the same length"
        );
        for (pos, &r) in refs.iter().enumerate() {
            assert!(
                r < n_groups,
                "refs[{pos}] = {r} is out of range for {n_groups} groups"
            );
        }
        Self {
            data,
            refs,
            n_groups,
        }
    }

    /// Number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the array is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Sum the data values by group, returning a vector of length `n_groups`.
    ///
    /// Equivalent to the Julia `sum!(zeros(n_groups), ragged)`.
    pub fn sum_by_group(&self) -> Vec<f64> {
        let mut sums = vec![0.0; self.n_groups];
        for (&val, &grp) in self.data.iter().zip(self.refs.iter()) {
            sums[grp] += val;
        }
        sums
    }

    /// Accumulate data values into a pre-allocated vector.
    ///
    /// Values are **added** to the existing contents of `dest`; the caller
    /// should zero the vector first if a fresh accumulation is desired.
    ///
    /// # Panics
    ///
    /// Panics if `dest.len() < self.n_groups`.
    pub fn sum_into(&self, dest: &mut [f64]) {
        assert!(dest.len() >= self.n_groups, "destination slice too short");
        for (&val, &grp) in self.data.iter().zip(self.refs.iter()) {
            dest[grp] += val;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_sum_by_group() {
        // 3 groups, 6 elements
        let ra = RaggedArray::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![0, 1, 2, 0, 1, 2],
            3,
        );
        let sums = ra.sum_by_group();
        assert_eq!(sums.len(), 3);
        assert_relative_eq!(sums[0], 5.0); // 1 + 4
        assert_relative_eq!(sums[1], 7.0); // 2 + 5
        assert_relative_eq!(sums[2], 9.0); // 3 + 6
    }

    #[test]
    fn test_sum_into() {
        let ra = RaggedArray::new(vec![1.0, 2.0, 3.0], vec![0, 0, 1], 2);
        let mut dest = vec![10.0, 20.0];
        ra.sum_into(&mut dest);
        assert_relative_eq!(dest[0], 13.0); // 10 + 1 + 2
        assert_relative_eq!(dest[1], 23.0); // 20 + 3
    }

    #[test]
    fn test_len_and_empty() {
        let ra = RaggedArray::new(vec![1.0], vec![0], 1);
        assert_eq!(ra.len(), 1);
        assert!(!ra.is_empty());

        let empty = RaggedArray::new(vec![], vec![], 1);
        assert!(empty.is_empty());
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn test_invalid_ref() {
        RaggedArray::new(vec![1.0], vec![5], 3);
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn test_mismatched_lengths() {
        RaggedArray::new(vec![1.0, 2.0], vec![0], 1);
    }
}
