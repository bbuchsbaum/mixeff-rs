//! Data frame abstraction for passing tabular data to model constructors.

use crate::error::{MixedModelError, Result};
use indexmap::IndexMap;
use nalgebra::DMatrix;
use std::collections::HashMap;

/// A simple column-oriented table for feeding data to mixed models.
///
/// Columns can be numeric (`f64`) or categorical (`String`).
/// This is intentionally minimal — real applications may want to
/// convert from polars/arrow DataFrames into this representation.
#[derive(Debug, Clone)]
pub struct DataFrame {
    /// Ordered mapping from column name → column data.
    columns: IndexMap<String, Column>,
    n_rows: usize,
}

/// A single column of data.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Column {
    /// Numeric `f64` values.
    Numeric(Vec<f64>),
    /// Categorical values encoded as integer references into a level table.
    Categorical(CategoricalColumn),
}

/// A categorical (factor) column with level encoding.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CategoricalColumn {
    /// The unique levels in order of first appearance.
    pub levels: Vec<String>,
    /// Index into `levels` for each row (0-based).
    pub refs: Vec<u32>,
    /// Original string values
    pub values: Vec<String>,
    /// Optional explicit contrast basis supplied by a frontend.
    pub contrast: Option<CategoricalContrast>,
}

/// Stable source label for a categorical contrast basis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ContrastSource {
    /// Treatment/dummy coding.
    Treatment,
    /// Sum-to-zero/deviation coding.
    Sum,
    /// Helmert coding.
    Helmert,
    /// Orthonormal polynomial coding for ordered factors.
    Polynomial,
    /// User-supplied contrast matrix.
    Custom,
    /// Contrast source was not supplied by the frontend.
    Unknown,
}

impl ContrastSource {
    /// Stable lowercase label for serialization and diagnostics.
    pub fn as_str(self) -> &'static str {
        match self {
            ContrastSource::Treatment => "treatment",
            ContrastSource::Sum => "sum",
            ContrastSource::Helmert => "helmert",
            ContrastSource::Polynomial => "polynomial",
            ContrastSource::Custom => "custom",
            ContrastSource::Unknown => "unknown",
        }
    }
}

/// Categorical coding mode used when no explicit contrast basis is supplied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CategoricalCoding {
    /// Drop the first level as a reference category.
    Treatment,
    /// Encode every level as its own indicator column.
    CellMeans,
}

/// Explicit categorical contrast basis.
///
/// Rows are in categorical level order and columns are the encoded basis
/// columns used in fixed-effect and random-effect design construction.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct CategoricalContrast {
    /// Level order corresponding to matrix rows.
    pub levels: Vec<String>,
    /// Contrast basis with one row per level.
    pub matrix: DMatrix<f64>,
    /// Names for encoded columns, one per matrix column.
    pub column_names: Vec<String>,
    /// Whether the contrast treats the factor as ordered.
    pub ordered: bool,
    /// Provenance label for the contrast basis.
    pub source: ContrastSource,
}

impl CategoricalContrast {
    /// Construct a contrast basis after validating shape and column names.
    pub fn new(
        levels: Vec<String>,
        matrix: DMatrix<f64>,
        column_names: Vec<String>,
        ordered: bool,
        source: ContrastSource,
    ) -> Result<Self> {
        validate_contrast_shape(&levels, &matrix, &column_names)?;
        Ok(Self {
            levels,
            matrix,
            column_names,
            ordered,
            source,
        })
    }

    /// Treatment (dummy) coding with the first level as the reference.
    ///
    /// `k` levels produce a `k × (k-1)` matrix whose reference row is all
    /// zeros; column `j` is the indicator for `levels[j + 1]`.
    pub fn treatment(levels: Vec<String>) -> Result<Self> {
        let k = Self::require_min_levels(&levels)?;
        let mut matrix = DMatrix::zeros(k, k - 1);
        for j in 0..k - 1 {
            matrix[(j + 1, j)] = 1.0;
        }
        let column_names = levels[1..].to_vec();
        Self::new(
            levels,
            matrix,
            column_names,
            false,
            ContrastSource::Treatment,
        )
    }

    /// Sum-to-zero (deviation) coding.
    ///
    /// `k` levels produce a `k × (k-1)` matrix where column `j` is `1` for
    /// `levels[j]`, `-1` for the last level, and `0` otherwise.
    pub fn sum(levels: Vec<String>) -> Result<Self> {
        let k = Self::require_min_levels(&levels)?;
        let mut matrix = DMatrix::zeros(k, k - 1);
        for j in 0..k - 1 {
            matrix[(j, j)] = 1.0;
            matrix[(k - 1, j)] = -1.0;
        }
        let column_names = levels[..k - 1].to_vec();
        Self::new(levels, matrix, column_names, false, ContrastSource::Sum)
    }

    /// Helmert coding: each column contrasts a level against the mean of all
    /// preceding levels (matches R's `contr.helmert`).
    pub fn helmert(levels: Vec<String>) -> Result<Self> {
        let k = Self::require_min_levels(&levels)?;
        let mut matrix = DMatrix::zeros(k, k - 1);
        for c in 0..k - 1 {
            for row in 0..=c {
                matrix[(row, c)] = -1.0;
            }
            matrix[(c + 1, c)] = (c + 1) as f64;
        }
        let column_names = levels[1..].to_vec();
        Self::new(levels, matrix, column_names, false, ContrastSource::Helmert)
    }

    /// Orthonormal polynomial contrasts over equally-spaced scores (matches
    /// R's `contr.poly`). Columns are the linear, quadratic, … trends; the
    /// contrast is marked `ordered`.
    pub fn polynomial(levels: Vec<String>) -> Result<Self> {
        let k = Self::require_min_levels(&levels)?;
        // Centered, equally-spaced scores keep the Vandermonde well-conditioned.
        let center = (k - 1) as f64 / 2.0;
        let mut vander = DMatrix::zeros(k, k);
        for i in 0..k {
            let x = i as f64 - center;
            let mut acc = 1.0;
            for p in 0..k {
                vander[(i, p)] = acc;
                acc *= x;
            }
        }
        let qr = vander.qr();
        let q = qr.q();
        let r = qr.r();
        let mut matrix = DMatrix::zeros(k, k - 1);
        for deg in 1..k {
            // R normalizes so the QR diagonal is positive (fixes column sign).
            let sign = if r[(deg, deg)] < 0.0 { -1.0 } else { 1.0 };
            for row in 0..k {
                matrix[(row, deg - 1)] = sign * q[(row, deg)];
            }
        }
        let column_names = (1..k).map(polynomial_column_name).collect();
        Self::new(
            levels,
            matrix,
            column_names,
            true,
            ContrastSource::Polynomial,
        )
    }

    fn require_min_levels(levels: &[String]) -> Result<usize> {
        let k = levels.len();
        if k < 2 {
            return Err(MixedModelError::InvalidArgument(format!(
                "a categorical contrast needs at least 2 levels, got {k}"
            )));
        }
        Ok(k)
    }
}

/// R-style polynomial contrast column label: `.L`, `.Q`, `.C`, then `^4`, …
fn polynomial_column_name(degree: usize) -> String {
    match degree {
        1 => ".L".to_string(),
        2 => ".Q".to_string(),
        3 => ".C".to_string(),
        d => format!("^{d}"),
    }
}

/// One encoded numeric column derived from a categorical predictor.
#[derive(Debug, Clone, PartialEq)]
pub struct EncodedCategoricalColumn {
    /// Display/stable column name.
    pub name: String,
    /// Encoded numeric values, one per input row.
    pub values: Vec<f64>,
    /// Whether the values came from an explicit contrast matrix.
    pub explicit_contrast: bool,
}

/// Audit record for one cluster-resampling draw.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClusterResampleDraw {
    /// Grouping column used for cluster resampling.
    pub group: String,
    /// Number of observed levels in the original grouping factor.
    pub original_level_count: usize,
    /// Sampled original levels in draw order.
    pub sampled_levels: Vec<String>,
    /// Number of distinct original levels sampled at least once.
    pub distinct_sampled_level_count: usize,
    /// Number of duplicate level draws.
    pub duplicate_count: usize,
    /// Number of rows in the resampled data frame.
    pub output_rows: usize,
    /// Stable label for the relabeling rule used for duplicate clusters.
    pub relabeling_policy: String,
}

impl CategoricalColumn {
    /// Construct a categorical column using first-appearance level order.
    pub fn new(values: Vec<String>) -> Self {
        let mut levels = Vec::new();
        let mut level_map: HashMap<String, u32> = HashMap::new();
        let mut refs = Vec::with_capacity(values.len());
        for v in &values {
            let idx = if let Some(&idx) = level_map.get(v) {
                idx
            } else {
                let idx = levels.len() as u32;
                levels.push(v.clone());
                level_map.insert(v.clone(), idx);
                idx
            };
            refs.push(idx);
        }
        CategoricalColumn {
            levels,
            refs,
            values,
            contrast: None,
        }
    }

    /// Number of canonical levels.
    pub fn n_levels(&self) -> usize {
        self.levels.len()
    }

    /// Construct from values together with an explicit canonical level order.
    ///
    /// Returns an error if any observed value is not in `levels`. Use this when
    /// the level order matters (e.g. matching a reference implementation's
    /// factor encoding) rather than first-appearance order.
    pub fn with_levels(values: Vec<String>, levels: Vec<String>) -> Result<Self> {
        let level_map: HashMap<&String, u32> = levels
            .iter()
            .enumerate()
            .map(|(i, s)| (s, i as u32))
            .collect();
        let mut refs = Vec::with_capacity(values.len());
        for (row, v) in values.iter().enumerate() {
            let Some(&idx) = level_map.get(v) else {
                return Err(MixedModelError::InvalidArgument(format!(
                    "categorical value `{v}` at row {row} is not present in canonical levels"
                )));
            };
            refs.push(idx);
        }
        Ok(CategoricalColumn {
            levels,
            refs,
            values,
            contrast: None,
        })
    }

    /// Construct a categorical column with explicit level order and contrast.
    pub fn with_levels_and_contrast(
        values: Vec<String>,
        levels: Vec<String>,
        contrast: CategoricalContrast,
    ) -> Result<Self> {
        let mut cat = Self::with_levels(values, levels)?;
        cat.set_contrast(contrast)?;
        Ok(cat)
    }

    /// Attach an explicit contrast basis to this categorical column.
    pub fn set_contrast(&mut self, contrast: CategoricalContrast) -> Result<()> {
        if contrast.levels != self.levels {
            return Err(MixedModelError::InvalidArgument(
                "categorical contrast levels must match canonical levels exactly".to_string(),
            ));
        }
        validate_contrast_shape(&contrast.levels, &contrast.matrix, &contrast.column_names)?;
        self.contrast = Some(contrast);
        Ok(())
    }

    /// Encode this categorical column for fixed-effect or random-effect design construction.
    pub fn encoded_columns(
        &self,
        variable: &str,
        coding: CategoricalCoding,
    ) -> Vec<EncodedCategoricalColumn> {
        if coding == CategoricalCoding::Treatment {
            if let Some(contrast) = &self.contrast {
                return (0..contrast.matrix.ncols())
                    .map(|column| EncodedCategoricalColumn {
                        name: format!("{variable}: {}", contrast.column_names[column]),
                        values: self
                            .refs
                            .iter()
                            .map(|&reference| contrast.matrix[(reference as usize, column)])
                            .collect(),
                        explicit_contrast: true,
                    })
                    .collect();
            }
        }

        let skip_reference = usize::from(coding == CategoricalCoding::Treatment);
        self.levels
            .iter()
            .enumerate()
            .skip(skip_reference)
            .map(|(level_index, level)| EncodedCategoricalColumn {
                name: format!("{variable}: {level}"),
                values: self
                    .refs
                    .iter()
                    .map(|&reference| f64::from(reference as usize == level_index))
                    .collect(),
                explicit_contrast: false,
            })
            .collect()
    }
}

impl DataFrame {
    /// Create a new empty DataFrame.
    pub fn new() -> Self {
        DataFrame {
            columns: IndexMap::new(),
            n_rows: 0,
        }
    }

    /// Number of rows.
    pub fn nrow(&self) -> usize {
        self.n_rows
    }

    /// Number of columns.
    pub fn ncol(&self) -> usize {
        self.columns.len()
    }

    /// Column names.
    pub fn column_names(&self) -> Vec<&str> {
        self.columns.keys().map(|s| s.as_str()).collect()
    }

    fn validate_new_column_len(&mut self, name: &str, len: usize) -> Result<()> {
        if self.columns.is_empty() {
            self.n_rows = len;
            return Ok(());
        }

        if len != self.n_rows {
            return Err(MixedModelError::DimensionMismatch(format!(
                "column `{name}` has length {len}, expected {}",
                self.n_rows
            )));
        }
        Ok(())
    }

    /// Add a numeric column.
    ///
    /// Rejects non-finite values (`NaN`, `+Inf`, `-Inf`): they would otherwise
    /// propagate silently into the cross-product and surface only as an opaque
    /// Cholesky failure. Use [`DataFrame::add_numeric_unchecked`] to bypass this
    /// check when non-finite sentinels are intentional.
    pub fn add_numeric(&mut self, name: &str, data: Vec<f64>) -> Result<&mut Self> {
        if let Some(pos) = data.iter().position(|v| !v.is_finite()) {
            return Err(MixedModelError::InvalidArgument(format!(
                "numeric column `{name}` contains a non-finite value ({}) at index {pos}; \
                 reject NaN/Inf before fitting or use add_numeric_unchecked",
                data[pos]
            )));
        }
        self.validate_new_column_len(name, data.len())?;
        self.columns.insert(name.to_string(), Column::Numeric(data));
        Ok(self)
    }

    /// Add a numeric column without rejecting non-finite values.
    ///
    /// Escape hatch for callers that deliberately encode `NaN`/`Inf` sentinels.
    /// Non-finite values propagate unchecked into model fitting and will
    /// typically surface as a Cholesky/positive-definite failure.
    pub fn add_numeric_unchecked(&mut self, name: &str, data: Vec<f64>) -> Result<&mut Self> {
        self.validate_new_column_len(name, data.len())?;
        self.columns.insert(name.to_string(), Column::Numeric(data));
        Ok(self)
    }

    /// Add a categorical column from string values.
    pub fn add_categorical(&mut self, name: &str, data: Vec<String>) -> Result<&mut Self> {
        self.validate_new_column_len(name, data.len())?;
        let cat = CategoricalColumn::new(data);
        self.columns
            .insert(name.to_string(), Column::Categorical(cat));
        Ok(self)
    }

    /// Add a categorical column with an explicit canonical level order.
    pub fn add_categorical_with_levels(
        &mut self,
        name: &str,
        data: Vec<String>,
        levels: Vec<String>,
    ) -> Result<&mut Self> {
        let cat = CategoricalColumn::with_levels(data, levels)?;
        self.validate_new_column_len(name, cat.values.len())?;
        self.columns
            .insert(name.to_string(), Column::Categorical(cat));
        Ok(self)
    }

    /// Add a categorical column with explicit level order and contrast basis.
    pub fn add_categorical_with_contrast(
        &mut self,
        name: &str,
        data: Vec<String>,
        levels: Vec<String>,
        contrast: CategoricalContrast,
    ) -> Result<&mut Self> {
        let cat = CategoricalColumn::with_levels_and_contrast(data, levels, contrast)?;
        self.validate_new_column_len(name, cat.values.len())?;
        self.columns
            .insert(name.to_string(), Column::Categorical(cat));
        Ok(self)
    }

    /// Get a numeric column by name.
    pub fn numeric(&self, name: &str) -> Option<&[f64]> {
        match self.columns.get(name)? {
            Column::Numeric(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    /// Get a categorical column by name.
    pub fn categorical(&self, name: &str) -> Option<&CategoricalColumn> {
        match self.columns.get(name)? {
            Column::Categorical(c) => Some(c),
            _ => None,
        }
    }

    /// Get a column (either type) by name.
    pub fn column(&self, name: &str) -> Option<&Column> {
        self.columns.get(name)
    }

    /// Check if a column exists.
    pub fn has_column(&self, name: &str) -> bool {
        self.columns.contains_key(name)
    }

    /// Return a row-selected data frame while preserving categorical level order.
    pub fn select_rows(&self, rows: &[usize]) -> Result<Self> {
        let mut out = DataFrame::new();
        for (name, column) in &self.columns {
            match column {
                Column::Numeric(values) => {
                    let selected = rows
                        .iter()
                        .map(|&row| {
                            values.get(row).copied().ok_or_else(|| {
                                MixedModelError::InvalidArgument(format!(
                                    "row index {row} is out of bounds for {} rows",
                                    self.n_rows
                                ))
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    out.add_numeric(name, selected)?;
                }
                Column::Categorical(cat) => {
                    let selected = rows
                        .iter()
                        .map(|&row| {
                            cat.values.get(row).cloned().ok_or_else(|| {
                                MixedModelError::InvalidArgument(format!(
                                    "row index {row} is out of bounds for {} rows",
                                    self.n_rows
                                ))
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let mut selected_cat =
                        CategoricalColumn::with_levels(selected, cat.levels.clone())?;
                    if let Some(contrast) = &cat.contrast {
                        selected_cat.set_contrast(contrast.clone())?;
                    }
                    out.validate_new_column_len(name, selected_cat.values.len())?;
                    out.columns
                        .insert(name.clone(), Column::Categorical(selected_cat));
                }
            }
        }
        Ok(out)
    }

    /// Resample a categorical grouping factor by cluster with replacement.
    ///
    /// Duplicate sampled clusters are relabeled with draw-local unique levels
    /// so refitted random effects remain independent bootstrap clusters.
    pub fn cluster_resample<R: rand::Rng>(
        &self,
        group: &str,
        rng: &mut R,
    ) -> Result<(Self, ClusterResampleDraw)> {
        let group_column = self.categorical(group).ok_or_else(|| {
            MixedModelError::InvalidArgument(format!(
                "cluster resampling group `{group}` must be an observed categorical column"
            ))
        })?;
        if group_column.levels.is_empty() {
            return Err(MixedModelError::InvalidArgument(format!(
                "cluster resampling group `{group}` has no levels"
            )));
        }

        let mut rows_by_level = vec![Vec::new(); group_column.levels.len()];
        for (row, &reference) in group_column.refs.iter().enumerate() {
            let level = reference as usize;
            if let Some(rows) = rows_by_level.get_mut(level) {
                rows.push(row);
            }
        }

        let mut sampled_indices = Vec::with_capacity(group_column.levels.len());
        for _ in 0..group_column.levels.len() {
            sampled_indices.push(rng.gen_range(0..group_column.levels.len()));
        }
        let sampled_levels = sampled_indices
            .iter()
            .map(|&level| group_column.levels[level].clone())
            .collect::<Vec<_>>();
        let mut counts = vec![0usize; group_column.levels.len()];
        for &level in &sampled_indices {
            counts[level] += 1;
        }
        let distinct_sampled_level_count = counts.iter().filter(|&&count| count > 0).count();
        let duplicate_count = sampled_indices
            .len()
            .saturating_sub(distinct_sampled_level_count);

        let mut row_indices = Vec::new();
        let mut resampled_group_values = Vec::new();
        for (draw_index, &level) in sampled_indices.iter().enumerate() {
            let relabeled = format!("{}__boot{}", group_column.levels[level], draw_index + 1);
            for &row in &rows_by_level[level] {
                row_indices.push(row);
                resampled_group_values.push(relabeled.clone());
            }
        }

        let mut out = self.select_rows(&row_indices)?;
        out.columns.insert(
            group.to_string(),
            Column::Categorical(CategoricalColumn::new(resampled_group_values)),
        );
        let draw = ClusterResampleDraw {
            group: group.to_string(),
            original_level_count: group_column.levels.len(),
            sampled_levels,
            distinct_sampled_level_count,
            duplicate_count,
            output_rows: out.nrow(),
            relabeling_policy: "replicate_local_unique_levels".to_string(),
        };
        Ok((out, draw))
    }
}

fn validate_contrast_shape(
    levels: &[String],
    matrix: &DMatrix<f64>,
    column_names: &[String],
) -> Result<()> {
    if matrix.nrows() != levels.len() {
        return Err(MixedModelError::DimensionMismatch(format!(
            "categorical contrast matrix has {} row(s), expected {} level(s)",
            matrix.nrows(),
            levels.len()
        )));
    }
    if matrix.ncols() != column_names.len() {
        return Err(MixedModelError::DimensionMismatch(format!(
            "categorical contrast matrix has {} column(s), but {} contrast column name(s) were supplied",
            matrix.ncols(),
            column_names.len()
        )));
    }
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(MixedModelError::InvalidArgument(
            "categorical contrast matrix entries must be finite".to_string(),
        ));
    }
    let mut seen = std::collections::BTreeSet::new();
    for name in column_names {
        if name.is_empty() {
            return Err(MixedModelError::InvalidArgument(
                "categorical contrast column names must be non-empty".to_string(),
            ));
        }
        if !seen.insert(name) {
            return Err(MixedModelError::InvalidArgument(format!(
                "duplicate categorical contrast column name `{name}`"
            )));
        }
    }
    Ok(())
}

impl Default for DataFrame {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn levels(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("L{i}")).collect()
    }

    #[test]
    fn contrast_constructors_reject_under_two_levels() {
        for ctor in [
            CategoricalContrast::treatment,
            CategoricalContrast::sum,
            CategoricalContrast::helmert,
            CategoricalContrast::polynomial,
        ] {
            assert!(ctor(levels(1)).is_err());
            assert!(ctor(levels(2)).is_ok());
        }
    }

    #[test]
    fn treatment_contrast_has_zero_reference_row() {
        let c = CategoricalContrast::treatment(levels(4)).unwrap();
        assert_eq!(c.source, ContrastSource::Treatment);
        assert_eq!(c.matrix.shape(), (4, 3));
        assert_eq!(c.column_names, vec!["L1", "L2", "L3"]);
        for j in 0..3 {
            assert_eq!(c.matrix[(0, j)], 0.0);
            assert_eq!(c.matrix[(j + 1, j)], 1.0);
        }
    }

    #[test]
    fn sum_contrast_last_row_is_minus_one() {
        let c = CategoricalContrast::sum(levels(3)).unwrap();
        assert_eq!(c.source, ContrastSource::Sum);
        assert_eq!(c.matrix.shape(), (3, 2));
        for j in 0..2 {
            assert_eq!(c.matrix[(j, j)], 1.0);
            assert_eq!(c.matrix[(2, j)], -1.0);
        }
        // Columns sum to zero (the defining deviation-coding property).
        for j in 0..2 {
            let s: f64 = (0..3).map(|i| c.matrix[(i, j)]).sum();
            assert!(s.abs() < 1e-12);
        }
    }

    #[test]
    fn helmert_contrast_matches_r_contr_helmert_4() {
        // R's contr.helmert(4):
        //   [,1] [,2] [,3]
        //   -1   -1   -1
        //    1   -1   -1
        //    0    2   -1
        //    0    0    3
        let c = CategoricalContrast::helmert(levels(4)).unwrap();
        assert_eq!(c.source, ContrastSource::Helmert);
        let expected = [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [0.0, 2.0, -1.0],
            [0.0, 0.0, 3.0],
        ];
        for i in 0..4 {
            for j in 0..3 {
                assert_eq!(c.matrix[(i, j)], expected[i][j], "({i},{j})");
            }
        }
    }

    #[test]
    fn polynomial_contrast_is_orthonormal_and_ordered() {
        let c = CategoricalContrast::polynomial(levels(4)).unwrap();
        assert_eq!(c.source, ContrastSource::Polynomial);
        assert!(c.ordered);
        assert_eq!(c.column_names, vec![".L", ".Q", ".C"]);
        let m = &c.matrix;
        // Orthonormal columns: M'M == I_{k-1}.
        for a in 0..3 {
            for b in 0..3 {
                let dot: f64 = (0..4).map(|i| m[(i, a)] * m[(i, b)]).sum();
                let want = if a == b { 1.0 } else { 0.0 };
                assert!((dot - want).abs() < 1e-10, "<{a},{b}> = {dot}");
            }
        }
        // Linear column increases monotonically across levels.
        for i in 0..3 {
            assert!(m[(i + 1, 0)] > m[(i, 0)]);
        }
    }

    #[test]
    fn polynomial_contrast_matches_r_contr_poly_4() {
        // R's contr.poly(4) in closed form:
        //   .L = c(-3, -1, 1, 3) / sqrt(20)
        //   .Q = c(1, -1, -1, 1) / 2
        //   .C = c(-1, 3, -3, 1) / sqrt(20)
        let c = CategoricalContrast::polynomial(levels(4)).unwrap();
        let s20 = 20.0_f64.sqrt();
        let expected = [
            [-3.0 / s20, 0.5, -1.0 / s20],
            [-1.0 / s20, -0.5, 3.0 / s20],
            [1.0 / s20, -0.5, -3.0 / s20],
            [3.0 / s20, 0.5, 1.0 / s20],
        ];
        for (i, row) in expected.iter().enumerate() {
            for (j, want) in row.iter().enumerate() {
                assert!(
                    (c.matrix[(i, j)] - want).abs() < 1e-12,
                    "({i},{j}): got {}, want {want}",
                    c.matrix[(i, j)]
                );
            }
        }
    }

    #[test]
    fn add_numeric_rejects_non_finite() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut df = DataFrame::new();
            let err = df.add_numeric("x", vec![1.0, bad, 3.0]).unwrap_err();
            match err {
                MixedModelError::InvalidArgument(msg) => {
                    assert!(msg.contains("non-finite"), "got: {msg}");
                    assert!(msg.contains("`x`"), "got: {msg}");
                    assert!(msg.contains("index 1"), "got: {msg}");
                }
                other => panic!("expected InvalidArgument, got {other:?}"),
            }
        }
    }

    #[test]
    fn add_numeric_unchecked_allows_non_finite() {
        let mut df = DataFrame::new();
        df.add_numeric_unchecked("x", vec![1.0, f64::NAN, 3.0])
            .unwrap();
        assert_eq!(df.numeric("x").map(|c| c.len()), Some(3));
    }

    #[test]
    fn test_add_categorical_with_levels_unknown_value_returns_err() {
        let mut df = DataFrame::new();

        let err = df
            .add_categorical_with_levels(
                "group",
                vec!["a".to_string(), "missing".to_string()],
                vec!["a".to_string(), "b".to_string()],
            )
            .unwrap_err();

        match err {
            MixedModelError::InvalidArgument(message) => {
                assert!(message.contains("missing"));
                assert!(message.contains("row 1"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
        assert_eq!(df.nrow(), 0);
        assert!(!df.has_column("group"));
    }

    #[test]
    fn test_add_categorical_with_contrast_validates_shape_and_names() {
        let mut df = DataFrame::new();
        let contrast = CategoricalContrast::new(
            vec!["A".to_string(), "B".to_string()],
            DMatrix::from_row_slice(2, 1, &[0.5, -0.5]),
            vec!["A_vs_B".to_string()],
            false,
            ContrastSource::Custom,
        )
        .unwrap();
        df.add_categorical_with_contrast(
            "anchor",
            vec!["A".to_string(), "B".to_string()],
            vec!["A".to_string(), "B".to_string()],
            contrast,
        )
        .unwrap();

        let encoded = df
            .categorical("anchor")
            .unwrap()
            .encoded_columns("anchor", CategoricalCoding::Treatment);
        assert_eq!(encoded[0].name, "anchor: A_vs_B");
        assert_eq!(encoded[0].values, vec![0.5, -0.5]);

        let err = CategoricalContrast::new(
            vec!["A".to_string(), "B".to_string()],
            DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]),
            vec!["dup".to_string(), "dup".to_string()],
            false,
            ContrastSource::Custom,
        )
        .unwrap_err();
        assert!(matches!(err, MixedModelError::InvalidArgument(_)));
    }

    #[test]
    fn test_add_column_length_mismatch_returns_err() {
        let mut df = DataFrame::new();
        df.add_numeric("y", vec![1.0, 2.0, 3.0]).unwrap();

        let err = df
            .add_categorical("group", vec!["a".to_string()])
            .unwrap_err();

        assert!(matches!(err, MixedModelError::DimensionMismatch(_)));
        assert!(!df.has_column("group"));
    }

    #[test]
    fn test_cluster_resample_relabels_duplicate_clusters() {
        use rand::SeedableRng;

        let mut df = DataFrame::new();
        df.add_numeric("y", vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        df.add_categorical(
            "group",
            vec!["a".into(), "a".into(), "b".into(), "b".into()],
        )
        .unwrap();

        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let (resampled, draw) = df.cluster_resample("group", &mut rng).unwrap();

        assert_eq!(draw.original_level_count, 2);
        assert_eq!(draw.sampled_levels.len(), 2);
        assert_eq!(draw.output_rows, 4);
        assert_eq!(draw.relabeling_policy, "replicate_local_unique_levels");
        let group = resampled.categorical("group").unwrap();
        assert_eq!(group.values.len(), 4);
        assert_eq!(group.n_levels(), 2);
        assert!(group.values.iter().all(|value| value.contains("__boot")));
    }
}
