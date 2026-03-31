//! Data frame abstraction for passing tabular data to model constructors.

use std::collections::HashMap;
use indexmap::IndexMap;

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
        CategoricalColumn { levels, refs, values }
    }

    pub fn n_levels(&self) -> usize {
        self.levels.len()
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

    /// Add a numeric column.
    pub fn add_numeric(&mut self, name: &str, data: Vec<f64>) -> &mut Self {
        if self.columns.is_empty() {
            self.n_rows = data.len();
        } else {
            assert_eq!(data.len(), self.n_rows, "Column length mismatch");
        }
        self.columns.insert(name.to_string(), Column::Numeric(data));
        self
    }

    /// Add a categorical column from string values.
    pub fn add_categorical(&mut self, name: &str, data: Vec<String>) -> &mut Self {
        if self.columns.is_empty() {
            self.n_rows = data.len();
        } else {
            assert_eq!(data.len(), self.n_rows, "Column length mismatch");
        }
        let cat = CategoricalColumn::new(data);
        self.columns.insert(name.to_string(), Column::Categorical(cat));
        self
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

impl Default for DataFrame {
    fn default() -> Self {
        Self::new()
    }
}
