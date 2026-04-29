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
pub enum Column {
    Numeric(Vec<f64>),
    Categorical(CategoricalColumn),
}

/// A categorical (factor) column with level encoding.
#[derive(Debug, Clone)]
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
pub enum ContrastSource {
    Treatment,
    Sum,
    Helmert,
    Polynomial,
    Custom,
    Unknown,
}

impl ContrastSource {
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
pub enum CategoricalCoding {
    Treatment,
    CellMeans,
}

/// Explicit categorical contrast basis.
///
/// Rows are in categorical level order and columns are the encoded basis
/// columns used in fixed-effect and random-effect design construction.
#[derive(Debug, Clone, PartialEq)]
pub struct CategoricalContrast {
    pub levels: Vec<String>,
    pub matrix: DMatrix<f64>,
    pub column_names: Vec<String>,
    pub ordered: bool,
    pub source: ContrastSource,
}

impl CategoricalContrast {
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
}

#[derive(Debug, Clone, PartialEq)]
pub struct EncodedCategoricalColumn {
    pub name: String,
    pub values: Vec<f64>,
    pub explicit_contrast: bool,
}

impl CategoricalColumn {
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

    pub fn with_levels_and_contrast(
        values: Vec<String>,
        levels: Vec<String>,
        contrast: CategoricalContrast,
    ) -> Result<Self> {
        let mut cat = Self::with_levels(values, levels)?;
        cat.set_contrast(contrast)?;
        Ok(cat)
    }

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
    pub fn add_numeric(&mut self, name: &str, data: Vec<f64>) -> Result<&mut Self> {
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

#[cfg(test)]
mod tests {
    use super::*;

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
}

impl Default for DataFrame {
    fn default() -> Self {
        Self::new()
    }
}
