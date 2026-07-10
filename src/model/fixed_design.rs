//! Fixed-effect design backends.
//!
//! Fitting an LMM needs fixed-effect cross-products such as `X'X`,
//! `X'y`, and `X'Z`. It does not always need a materialized dense
//! `n x p` model matrix. This module makes that boundary explicit so the
//! current dense implementation can coexist with future streamed/sparse
//! high-cardinality fixed-effect backends.

use nalgebra::{DMatrix, DVector};
use nalgebra_sparse::{coo::CooMatrix, csc::CscMatrix};

use crate::error::{MixedModelError, Result};
use crate::formula::{FixedTerm, Formula};
use crate::model::data::{CategoricalCoding, Column, DataFrame};
use crate::types::{MatrixBlock, ReMat};

/// Storage strategy used by a fixed-effect design backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FixedDesignStorage {
    /// Materialized dense `n x p` matrix.
    Dense,
    /// Structured terms/refs with streamed cross-products.
    Streamed,
    /// Sparse matrix representation.
    Sparse,
}

/// Lightweight dimensions and allocation estimate for a fixed-effect design.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedDesignSummary {
    pub storage: FixedDesignStorage,
    pub n_obs: usize,
    pub n_cols: usize,
    pub dense_bytes: u128,
}

/// Backend preference used when compiling a formula fixed-effect design.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FixedDesignBackendPreference {
    /// Use a conservative policy based on dense allocation size and streamed
    /// row density.
    Auto,
    /// Force a materialized dense fixed-effect design.
    Dense,
    /// Force a streamed fixed-effect design.
    Streamed,
}

/// Selection policy for formula-to-fixed-design compilation.
///
/// The formula parser and term expansion are shared across backends. This
/// policy only decides whether the expanded row representation should remain
/// streamed or be materialized into the existing dense `FeTerm` path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixedDesignBuildPolicy {
    pub preference: FixedDesignBackendPreference,
    pub max_dense_bytes: u128,
    pub min_streamed_cols: usize,
    pub max_streamed_density: f64,
}

impl FixedDesignBuildPolicy {
    /// Default automatic policy.
    ///
    /// This keeps ordinary low-dimensional designs dense, but selects the
    /// streamed backend for wide sparse designs such as high-cardinality
    /// treatment-coded factors.
    pub fn auto() -> Self {
        Self {
            preference: FixedDesignBackendPreference::Auto,
            max_dense_bytes: 256 * 1024 * 1024,
            min_streamed_cols: 128,
            max_streamed_density: 0.25,
        }
    }

    pub fn dense() -> Self {
        Self {
            preference: FixedDesignBackendPreference::Dense,
            ..Self::auto()
        }
    }

    pub fn streamed() -> Self {
        Self {
            preference: FixedDesignBackendPreference::Streamed,
            ..Self::auto()
        }
    }

    pub fn with_max_dense_bytes(mut self, max_dense_bytes: u128) -> Self {
        self.max_dense_bytes = max_dense_bytes;
        self
    }

    pub fn with_min_streamed_cols(mut self, min_streamed_cols: usize) -> Self {
        self.min_streamed_cols = min_streamed_cols;
        self
    }

    pub fn with_max_streamed_density(mut self, max_streamed_density: f64) -> Self {
        self.max_streamed_density = max_streamed_density;
        self
    }

    fn select_storage(&self, design: &StreamedFixedDesign) -> Result<FixedDesignStorage> {
        self.validate()?;
        match self.preference {
            FixedDesignBackendPreference::Dense => Ok(FixedDesignStorage::Dense),
            FixedDesignBackendPreference::Streamed => Ok(FixedDesignStorage::Streamed),
            FixedDesignBackendPreference::Auto => {
                let summary = design.summary();
                let should_stream = summary.dense_bytes > self.max_dense_bytes
                    || (summary.n_cols >= self.min_streamed_cols
                        && design.density() <= self.max_streamed_density);
                Ok(if should_stream {
                    FixedDesignStorage::Streamed
                } else {
                    FixedDesignStorage::Dense
                })
            }
        }
    }

    fn validate(&self) -> Result<()> {
        if !self.max_streamed_density.is_finite()
            || self.max_streamed_density < 0.0
            || self.max_streamed_density > 1.0
        {
            return Err(MixedModelError::InvalidArgument(format!(
                "max streamed fixed-design density must be in [0, 1], got {}",
                self.max_streamed_density
            )));
        }
        Ok(())
    }
}

impl Default for FixedDesignBuildPolicy {
    fn default() -> Self {
        Self::auto()
    }
}

/// Backend interface for fixed-effect design operations required by the
/// profiled least-squares solver.
///
/// The default methods are intentionally dense fallbacks. Future streamed or
/// sparse backends should override the cross-product methods so categorical
/// dummy columns do not have to be materialized.
pub trait FixedDesignBackend {
    fn storage(&self) -> FixedDesignStorage;
    fn n_obs(&self) -> usize;
    fn n_cols(&self) -> usize;
    fn column_names(&self) -> &[String];
    fn materialize_dense(&self) -> DMatrix<f64>;

    fn dense_bytes(&self) -> u128 {
        dense_bytes(self.n_obs(), self.n_cols())
    }

    fn summary(&self) -> FixedDesignSummary {
        FixedDesignSummary {
            storage: self.storage(),
            n_obs: self.n_obs(),
            n_cols: self.n_cols(),
            dense_bytes: self.dense_bytes(),
        }
    }

    fn xtx(&self) -> DMatrix<f64> {
        let x = self.materialize_dense();
        x.transpose() * x
    }

    fn xty(&self, y: &DVector<f64>) -> Result<DVector<f64>> {
        if y.len() != self.n_obs() {
            return Err(MixedModelError::DimensionMismatch(format!(
                "fixed-effect design has {} rows but response has {}",
                self.n_obs(),
                y.len()
            )));
        }
        let x = self.materialize_dense();
        Ok(x.transpose() * y)
    }

    fn xt_reterm(&self, re: &ReMat) -> Result<MatrixBlock> {
        if re.n_obs() != self.n_obs() {
            return Err(MixedModelError::DimensionMismatch(format!(
                "fixed-effect design has {} rows but random term '{}' has {} rows",
                self.n_obs(),
                re.grouping_name,
                re.n_obs()
            )));
        }

        let x = self.materialize_dense();
        let p = x.ncols();
        let nranef = re.n_ranef();
        let mut result = DMatrix::zeros(p, nranef);

        for obs in 0..re.n_obs() {
            let level = re.refs[obs] as usize;
            for fixed_col in 0..p {
                for basis_row in 0..re.vsize {
                    result[(fixed_col, level * re.vsize + basis_row)] +=
                        x[(obs, fixed_col)] * re.wtz[(basis_row, obs)];
                }
            }
        }

        Ok(MatrixBlock::Dense(result))
    }

    fn row_dot_beta(&self, row: usize, beta: &DVector<f64>) -> Result<f64> {
        if row >= self.n_obs() {
            return Err(MixedModelError::InvalidArgument(format!(
                "fixed-effect row {row} is out of bounds for {} rows",
                self.n_obs()
            )));
        }
        if beta.len() != self.n_cols() {
            return Err(MixedModelError::DimensionMismatch(format!(
                "fixed-effect design has {} columns but beta has {} entries",
                self.n_cols(),
                beta.len()
            )));
        }

        let x = self.materialize_dense();
        Ok(x.row(row).dot(beta))
    }
}

/// Owned fixed-effect design dispatch.
///
/// Frontends that compile formulas can choose the representation that best
/// matches their transformed columns, while the fitting engine consumes a
/// single design type.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum FixedDesign {
    /// Materialized dense `n x p` fixed-effect design.
    Dense(DenseFixedDesign),
    /// Row-streamed fixed-effect design.
    Streamed(StreamedFixedDesign),
}

impl FixedDesign {
    pub fn dense(x: DMatrix<f64>, column_names: Vec<String>) -> Result<Self> {
        Ok(Self::Dense(DenseFixedDesign::new(x, column_names)?))
    }

    pub fn streamed(
        n_obs: usize,
        column_names: Vec<String>,
        rows: Vec<Vec<(usize, f64)>>,
    ) -> Result<Self> {
        Ok(Self::Streamed(StreamedFixedDesign::new(
            n_obs,
            column_names,
            rows,
        )?))
    }

    pub fn as_dense(&self) -> Option<&DenseFixedDesign> {
        match self {
            Self::Dense(design) => Some(design),
            Self::Streamed(_) => None,
        }
    }

    pub fn as_streamed(&self) -> Option<&StreamedFixedDesign> {
        match self {
            Self::Dense(_) => None,
            Self::Streamed(design) => Some(design),
        }
    }

    /// Return a design with columns selected in the requested order.
    pub fn select_columns(&self, columns: &[usize]) -> Result<Self> {
        match self {
            Self::Dense(design) => Ok(Self::Dense(design.select_columns(columns)?)),
            Self::Streamed(design) => Ok(Self::Streamed(design.select_columns(columns)?)),
        }
    }

    /// Return a row-weighted design where row `i` is scaled by
    /// `sqrt_weights[i]`.
    pub fn with_sqrt_weights(&self, sqrt_weights: &DVector<f64>) -> Result<Self> {
        match self {
            Self::Dense(design) => Ok(Self::Dense(design.with_sqrt_weights(sqrt_weights)?)),
            Self::Streamed(design) => Ok(Self::Streamed(design.with_sqrt_weights(sqrt_weights)?)),
        }
    }
}

impl From<DenseFixedDesign> for FixedDesign {
    fn from(value: DenseFixedDesign) -> Self {
        Self::Dense(value)
    }
}

impl From<StreamedFixedDesign> for FixedDesign {
    fn from(value: StreamedFixedDesign) -> Self {
        Self::Streamed(value)
    }
}

impl FixedDesignBackend for FixedDesign {
    fn storage(&self) -> FixedDesignStorage {
        match self {
            Self::Dense(design) => design.storage(),
            Self::Streamed(design) => design.storage(),
        }
    }

    fn n_obs(&self) -> usize {
        match self {
            Self::Dense(design) => design.n_obs(),
            Self::Streamed(design) => design.n_obs(),
        }
    }

    fn n_cols(&self) -> usize {
        match self {
            Self::Dense(design) => design.n_cols(),
            Self::Streamed(design) => design.n_cols(),
        }
    }

    fn column_names(&self) -> &[String] {
        match self {
            Self::Dense(design) => design.column_names(),
            Self::Streamed(design) => design.column_names(),
        }
    }

    fn materialize_dense(&self) -> DMatrix<f64> {
        match self {
            Self::Dense(design) => design.materialize_dense(),
            Self::Streamed(design) => design.materialize_dense(),
        }
    }

    fn xtx(&self) -> DMatrix<f64> {
        match self {
            Self::Dense(design) => design.xtx(),
            Self::Streamed(design) => design.xtx(),
        }
    }

    fn xty(&self, y: &DVector<f64>) -> Result<DVector<f64>> {
        match self {
            Self::Dense(design) => design.xty(y),
            Self::Streamed(design) => design.xty(y),
        }
    }

    fn xt_reterm(&self, re: &ReMat) -> Result<MatrixBlock> {
        match self {
            Self::Dense(design) => design.xt_reterm(re),
            Self::Streamed(design) => design.xt_reterm(re),
        }
    }

    fn row_dot_beta(&self, row: usize, beta: &DVector<f64>) -> Result<f64> {
        match self {
            Self::Dense(design) => design.row_dot_beta(row, beta),
            Self::Streamed(design) => design.row_dot_beta(row, beta),
        }
    }
}

/// Engine-level model design supplied after formula/transformation work.
///
/// This type is intentionally language-neutral: frontends can evaluate
/// transforms such as splines and offsets, then pass the resulting numeric
/// design to the engine without the engine knowing where those columns came
/// from.
#[derive(Debug, Clone)]
pub struct CompiledMixedModelDesign {
    response_name: Option<String>,
    response: DVector<f64>,
    fixed: FixedDesign,
    random_terms: Vec<ReMat>,
    offset: DVector<f64>,
    case_weights: Option<DVector<f64>>,
}

impl CompiledMixedModelDesign {
    /// Create a compiled design from already-transformed response, fixed
    /// effects, and random effects.
    pub fn new(
        response: DVector<f64>,
        fixed: impl Into<FixedDesign>,
        random_terms: Vec<ReMat>,
    ) -> Result<Self> {
        let n_obs = response.len();
        let fixed = fixed.into();
        let offset = DVector::zeros(n_obs);
        let design = Self {
            response_name: None,
            response,
            fixed,
            random_terms,
            offset,
            case_weights: None,
        };
        design.validate()?;
        Ok(design)
    }

    /// Convenience constructor for a materialized dense fixed-effect design.
    pub fn from_dense(
        response: DVector<f64>,
        x: DMatrix<f64>,
        column_names: Vec<String>,
        random_terms: Vec<ReMat>,
    ) -> Result<Self> {
        Self::new(
            response,
            DenseFixedDesign::new(x, column_names)?,
            random_terms,
        )
    }

    pub fn with_response_name(mut self, response_name: impl Into<String>) -> Self {
        self.response_name = Some(response_name.into());
        self
    }

    pub fn with_offset(mut self, offset: DVector<f64>) -> Result<Self> {
        self.set_offset(offset)?;
        Ok(self)
    }

    pub fn with_case_weights(mut self, case_weights: DVector<f64>) -> Result<Self> {
        self.set_case_weights(case_weights)?;
        Ok(self)
    }

    pub fn set_offset(&mut self, offset: DVector<f64>) -> Result<&mut Self> {
        validate_design_vector("offset", &offset, self.n_obs(), true)?;
        self.offset = offset;
        Ok(self)
    }

    pub fn set_case_weights(&mut self, case_weights: DVector<f64>) -> Result<&mut Self> {
        validate_design_vector("case weights", &case_weights, self.n_obs(), false)?;
        self.case_weights = Some(case_weights);
        Ok(self)
    }

    pub fn clear_case_weights(&mut self) -> &mut Self {
        self.case_weights = None;
        self
    }

    pub fn response_name(&self) -> Option<&str> {
        self.response_name.as_deref()
    }

    pub fn response(&self) -> &DVector<f64> {
        &self.response
    }

    pub fn fixed_design(&self) -> &FixedDesign {
        &self.fixed
    }

    pub fn random_terms(&self) -> &[ReMat] {
        &self.random_terms
    }

    pub fn offset(&self) -> &DVector<f64> {
        &self.offset
    }

    pub fn case_weights(&self) -> Option<&DVector<f64>> {
        self.case_weights.as_ref()
    }

    pub fn n_obs(&self) -> usize {
        self.response.len()
    }

    pub fn n_fixed_cols(&self) -> usize {
        self.fixed.n_cols()
    }

    /// Evaluate `offset + X beta` without materializing `X` for streamed
    /// fixed-effect designs.
    pub fn fixed_linear_predictor(&self, beta: &DVector<f64>) -> Result<DVector<f64>> {
        if beta.len() != self.fixed.n_cols() {
            return Err(MixedModelError::DimensionMismatch(format!(
                "fixed-effect design has {} columns but beta has {} entries",
                self.fixed.n_cols(),
                beta.len()
            )));
        }

        let mut eta = self.offset.clone();
        for row in 0..self.n_obs() {
            eta[row] += self.fixed.row_dot_beta(row, beta)?;
        }
        Ok(eta)
    }

    pub fn validate(&self) -> Result<()> {
        if self.random_terms.is_empty() {
            return Err(MixedModelError::NoRandomEffects);
        }
        validate_design_vector("response", &self.response, self.response.len(), true)?;
        if self.fixed.n_obs() != self.n_obs() {
            return Err(MixedModelError::DimensionMismatch(format!(
                "fixed-effect design has {} rows but response has {}",
                self.fixed.n_obs(),
                self.n_obs()
            )));
        }
        validate_design_vector("offset", &self.offset, self.n_obs(), true)?;
        if let Some(case_weights) = &self.case_weights {
            validate_design_vector("case weights", case_weights, self.n_obs(), false)?;
        }
        for (term_index, term) in self.random_terms.iter().enumerate() {
            if term.n_obs() != self.n_obs() {
                return Err(MixedModelError::DimensionMismatch(format!(
                    "random term {term_index} ('{}') has {} rows but response has {}",
                    term.grouping_name,
                    term.n_obs(),
                    self.n_obs()
                )));
            }
        }
        Ok(())
    }
}

/// Materialized dense fixed-effect design.
#[derive(Debug, Clone)]
pub struct DenseFixedDesign {
    x: DMatrix<f64>,
    column_names: Vec<String>,
}

impl DenseFixedDesign {
    pub fn new(x: DMatrix<f64>, column_names: Vec<String>) -> Result<Self> {
        if x.ncols() != column_names.len() {
            return Err(MixedModelError::DimensionMismatch(format!(
                "fixed-effect design has {} columns but {} column names",
                x.ncols(),
                column_names.len()
            )));
        }
        Ok(Self { x, column_names })
    }

    pub fn matrix(&self) -> &DMatrix<f64> {
        &self.x
    }

    pub fn into_parts(self) -> (DMatrix<f64>, Vec<String>) {
        (self.x, self.column_names)
    }

    pub fn with_sqrt_weights(&self, sqrt_weights: &DVector<f64>) -> Result<Self> {
        validate_sqrt_weights(sqrt_weights, self.x.nrows())?;
        let mut weighted = self.x.clone();
        for row in 0..weighted.nrows() {
            let scale = sqrt_weights[row];
            for column in 0..weighted.ncols() {
                weighted[(row, column)] *= scale;
            }
        }
        Self::new(weighted, self.column_names.clone())
    }

    pub fn select_columns(&self, columns: &[usize]) -> Result<Self> {
        validate_column_selection(columns, self.n_cols())?;
        let mut selected = DMatrix::zeros(self.n_obs(), columns.len());
        let mut selected_names = Vec::with_capacity(columns.len());
        for (new_col, &old_col) in columns.iter().enumerate() {
            selected.set_column(new_col, &self.x.column(old_col));
            selected_names.push(self.column_names[old_col].clone());
        }
        Self::new(selected, selected_names)
    }
}

impl FixedDesignBackend for DenseFixedDesign {
    fn storage(&self) -> FixedDesignStorage {
        FixedDesignStorage::Dense
    }

    fn n_obs(&self) -> usize {
        self.x.nrows()
    }

    fn n_cols(&self) -> usize {
        self.x.ncols()
    }

    fn column_names(&self) -> &[String] {
        &self.column_names
    }

    fn materialize_dense(&self) -> DMatrix<f64> {
        self.x.clone()
    }

    fn xtx(&self) -> DMatrix<f64> {
        self.x.transpose() * &self.x
    }

    fn xty(&self, y: &DVector<f64>) -> Result<DVector<f64>> {
        if y.len() != self.x.nrows() {
            return Err(MixedModelError::DimensionMismatch(format!(
                "fixed-effect design has {} rows but response has {}",
                self.x.nrows(),
                y.len()
            )));
        }
        Ok(self.x.transpose() * y)
    }

    fn row_dot_beta(&self, row: usize, beta: &DVector<f64>) -> Result<f64> {
        if row >= self.x.nrows() {
            return Err(MixedModelError::InvalidArgument(format!(
                "fixed-effect row {row} is out of bounds for {} rows",
                self.x.nrows()
            )));
        }
        if beta.len() != self.x.ncols() {
            return Err(MixedModelError::DimensionMismatch(format!(
                "fixed-effect design has {} columns but beta has {} entries",
                self.x.ncols(),
                beta.len()
            )));
        }
        Ok(self.x.row(row).dot(beta))
    }
}

/// Streamed fixed-effect design represented as active entries per row.
///
/// This is the backend shape needed for high-cardinality categorical fixed
/// effects. For example, an intercept plus one treatment-coded factor with
/// many levels has only two active entries per row: the intercept and the
/// observed non-reference level, if any. Cross-products can therefore be
/// accumulated from row entries without allocating the dense `n x p` matrix.
#[derive(Debug, Clone)]
pub struct StreamedFixedDesign {
    n_obs: usize,
    column_names: Vec<String>,
    rows: Vec<Vec<(usize, f64)>>,
}

impl StreamedFixedDesign {
    /// Create a streamed design from row-wise active entries.
    ///
    /// Each inner vector stores `(column_index, value)` pairs for one
    /// observation. Duplicate column entries within a row are summed so the
    /// stored representation has canonical row semantics.
    pub fn new(
        n_obs: usize,
        column_names: Vec<String>,
        rows: Vec<Vec<(usize, f64)>>,
    ) -> Result<Self> {
        if rows.len() != n_obs {
            return Err(MixedModelError::DimensionMismatch(format!(
                "streamed fixed-effect design expected {n_obs} row(s), got {}",
                rows.len()
            )));
        }

        let n_cols = column_names.len();
        let mut canonical_rows = Vec::with_capacity(rows.len());
        for (row_index, row) in rows.into_iter().enumerate() {
            let mut values = std::collections::BTreeMap::<usize, f64>::new();
            for (column, value) in row {
                if column >= n_cols {
                    return Err(MixedModelError::DimensionMismatch(format!(
                        "streamed fixed-effect row {row_index} references column {column}, but design has {n_cols} column(s)"
                    )));
                }
                if !value.is_finite() {
                    return Err(MixedModelError::InvalidArgument(format!(
                        "streamed fixed-effect row {row_index}, column {column} is non-finite"
                    )));
                }
                if value != 0.0 {
                    *values.entry(column).or_insert(0.0) += value;
                }
            }
            canonical_rows.push(
                values
                    .into_iter()
                    .filter(|(_, value)| *value != 0.0)
                    .collect::<Vec<_>>(),
            );
        }

        Ok(Self {
            n_obs,
            column_names,
            rows: canonical_rows,
        })
    }

    pub fn rows(&self) -> &[Vec<(usize, f64)>] {
        &self.rows
    }

    pub fn active_entries(&self) -> usize {
        self.rows.iter().map(Vec::len).sum()
    }

    pub fn density(&self) -> f64 {
        if self.n_obs == 0 || self.n_cols() == 0 {
            return 0.0;
        }
        self.active_entries() as f64 / (self.n_obs as f64 * self.n_cols() as f64)
    }

    pub fn from_dense(x: &DMatrix<f64>, column_names: Vec<String>) -> Result<Self> {
        if x.ncols() != column_names.len() {
            return Err(MixedModelError::DimensionMismatch(format!(
                "fixed-effect design has {} columns but {} column names",
                x.ncols(),
                column_names.len()
            )));
        }

        let rows = (0..x.nrows())
            .map(|row| {
                (0..x.ncols())
                    .filter_map(|column| {
                        let value = x[(row, column)];
                        (value != 0.0).then_some((column, value))
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        Self::new(x.nrows(), column_names, rows)
    }

    pub fn with_sqrt_weights(&self, sqrt_weights: &DVector<f64>) -> Result<Self> {
        validate_sqrt_weights(sqrt_weights, self.n_obs)?;
        let rows = self
            .rows
            .iter()
            .enumerate()
            .map(|(row_index, row)| {
                let scale = sqrt_weights[row_index];
                row.iter()
                    .map(|&(column, value)| (column, value * scale))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        Self::new(self.n_obs, self.column_names.clone(), rows)
    }

    pub fn select_columns(&self, columns: &[usize]) -> Result<Self> {
        validate_column_selection(columns, self.n_cols())?;
        let mut old_to_new = vec![None; self.n_cols()];
        for (new_col, &old_col) in columns.iter().enumerate() {
            old_to_new[old_col] = Some(new_col);
        }
        let selected_names = columns
            .iter()
            .map(|&column| self.column_names[column].clone())
            .collect::<Vec<_>>();
        let selected_rows = self
            .rows
            .iter()
            .map(|row| {
                row.iter()
                    .filter_map(|&(old_col, value)| {
                        old_to_new[old_col].map(|new_col| (new_col, value))
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        Self::new(self.n_obs, selected_names, selected_rows)
    }

    fn validate_response_len(&self, len: usize, context: &str) -> Result<()> {
        if len != self.n_obs {
            return Err(MixedModelError::DimensionMismatch(format!(
                "streamed fixed-effect design has {} rows but {context} has {len}",
                self.n_obs
            )));
        }
        Ok(())
    }
}

/// Build a fixed-effect design using the default backend policy.
pub fn build_fixed_effects_design(formula: &Formula, data: &DataFrame) -> Result<FixedDesign> {
    build_fixed_effects_design_with_policy(formula, data, FixedDesignBuildPolicy::default())
}

/// Build a fixed-effect design using an explicit backend selection policy.
pub fn build_fixed_effects_design_with_policy(
    formula: &Formula,
    data: &DataFrame,
    policy: FixedDesignBuildPolicy,
) -> Result<FixedDesign> {
    let streamed = build_streamed_fixed_effects_design(formula, data)?;
    match policy.select_storage(&streamed)? {
        FixedDesignStorage::Dense => FixedDesign::dense(
            streamed.materialize_dense(),
            streamed.column_names().to_vec(),
        ),
        FixedDesignStorage::Streamed => Ok(FixedDesign::Streamed(streamed)),
        FixedDesignStorage::Sparse => Err(MixedModelError::InvalidArgument(
            "sparse fixed-effect backend is not implemented".to_string(),
        )),
    }
}

/// Build a streamed fixed-effect design from a parsed formula and model data.
///
/// This mirrors the current dense fixed-effect treatment coding:
/// intercepts are explicit, numeric columns contribute one column, categorical
/// columns use treatment coding with the first observed/canonical level as the
/// reference, and interactions are products of the expanded component columns.
pub fn build_streamed_fixed_effects_design(
    formula: &Formula,
    data: &DataFrame,
) -> Result<StreamedFixedDesign> {
    let n = data.nrow();
    let mut column_names = Vec::new();
    let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];

    if formula.has_intercept() {
        let column = push_column_name(&mut column_names, "(Intercept)".to_string());
        for row in &mut rows {
            row.push((column, 1.0));
        }
    }

    for term in &formula.fixed_terms {
        match term {
            FixedTerm::Intercept | FixedTerm::NoIntercept => {}
            FixedTerm::Column(name) => {
                append_streamed_main_effect(name, data, &mut column_names, &mut rows)?;
            }
            FixedTerm::Interaction(vars) => {
                append_streamed_interaction(vars, formula, data, &mut column_names, &mut rows)?;
            }
        }
    }

    StreamedFixedDesign::new(n, column_names, rows)
}

fn push_column_name(column_names: &mut Vec<String>, name: String) -> usize {
    let index = column_names.len();
    column_names.push(name);
    index
}

fn append_streamed_main_effect(
    name: &str,
    data: &DataFrame,
    column_names: &mut Vec<String>,
    rows: &mut [Vec<(usize, f64)>],
) -> Result<()> {
    match data.column(name) {
        Some(Column::Numeric(values)) => {
            let column = push_column_name(column_names, name.to_string());
            for (row, &value) in values.iter().enumerate() {
                if value != 0.0 {
                    rows[row].push((column, value));
                }
            }
            Ok(())
        }
        Some(Column::Categorical(cat)) => {
            let encoded = cat.encoded_columns(name, CategoricalCoding::Treatment);
            let level_columns = encoded
                .iter()
                .map(|column| push_column_name(column_names, column.name.clone()))
                .collect::<Vec<_>>();

            for (column_index, encoded_column) in encoded.iter().enumerate() {
                for (row, &value) in encoded_column.values.iter().enumerate() {
                    if value != 0.0 {
                        rows[row].push((level_columns[column_index], value));
                    }
                }
            }
            Ok(())
        }
        None => Err(MixedModelError::InvalidArgument(format!(
            "Column '{name}' not found in data"
        ))),
    }
}

#[derive(Debug, Clone)]
struct StreamedFactorColumn {
    label: String,
    values: Vec<f64>,
}

fn append_streamed_interaction(
    vars: &[String],
    formula: &Formula,
    data: &DataFrame,
    column_names: &mut Vec<String>,
    rows: &mut [Vec<(usize, f64)>],
) -> Result<()> {
    let n = data.nrow();
    let treatment_variables = interaction_treatment_variables(formula, vars);
    let global_order = fixed_effect_variable_order(formula);
    let ordered_vars = global_order
        .into_iter()
        .filter(|name| vars.iter().any(|var| var == name))
        .collect::<Vec<_>>();
    let factors = ordered_vars
        .iter()
        .map(|name| {
            let coding = if treatment_variables.contains(*name) {
                CategoricalCoding::Treatment
            } else {
                CategoricalCoding::CellMeans
            };
            streamed_factor_columns(name, data, n, coding)
        })
        .collect::<Result<Vec<_>>>()?;

    let mut current = Vec::with_capacity(factors.len());
    append_streamed_interaction_products(&factors, 0, &mut current, column_names, rows);
    Ok(())
}

fn streamed_factor_columns(
    name: &str,
    data: &DataFrame,
    n: usize,
    coding: CategoricalCoding,
) -> Result<Vec<StreamedFactorColumn>> {
    match data.column(name) {
        Some(Column::Numeric(values)) => Ok(vec![StreamedFactorColumn {
            label: name.to_string(),
            values: values.clone(),
        }]),
        Some(Column::Categorical(cat)) => Ok(cat
            .encoded_columns(name, coding)
            .into_iter()
            .map(|column| StreamedFactorColumn {
                label: column.name,
                values: column.values,
            })
            .collect()),
        None => Err(MixedModelError::InvalidArgument(format!(
            "Column '{name}' not found in data"
        ))),
    }
    .and_then(|columns| {
        if columns.iter().all(|column| column.values.len() == n) {
            Ok(columns)
        } else {
            Err(MixedModelError::DimensionMismatch(format!(
                "interaction term column '{name}' has inconsistent row count"
            )))
        }
    })
}

/// R's terms machinery assigns contrast code 1 (ordinary contrasts) or 2
/// (full indicators) to each factor occurrence in an interaction. The code is
/// determined by marginality: the interaction term contributes the ANOVA
/// components not already spanned by lower-order fixed terms. Variables that
/// occur in every still-uncovered component use ordinary contrasts; the
/// others need full indicators. For `b + a:b`, for example, the uncovered
/// components are `{a}` and `{a,b}`, so `a` is treatment-coded while `b` is
/// full-coded — exactly R's `contrasts=1/2` assignment.
fn interaction_treatment_variables(
    formula: &Formula,
    vars: &[String],
) -> std::collections::BTreeSet<String> {
    let lower_terms = formula
        .fixed_terms
        .iter()
        .filter_map(fixed_term_variables)
        .filter(|term| {
            term.len() < vars.len() && term.iter().all(|name| vars.iter().any(|var| var == name))
        })
        .collect::<Vec<_>>();

    let mut intersection: Option<std::collections::BTreeSet<String>> = None;
    let mut current = Vec::new();
    visit_interaction_components(vars, 0, &mut current, &lower_terms, &mut intersection);
    intersection.unwrap_or_default()
}

fn fixed_term_variables(term: &FixedTerm) -> Option<Vec<String>> {
    match term {
        FixedTerm::Column(name) => Some(vec![name.clone()]),
        FixedTerm::Interaction(vars) => Some(vars.clone()),
        FixedTerm::Intercept | FixedTerm::NoIntercept => None,
    }
}

fn fixed_effect_variable_order(formula: &Formula) -> Vec<&str> {
    let mut order = Vec::new();
    for term in &formula.fixed_terms {
        match term {
            FixedTerm::Column(name) => {
                if !order.contains(&name.as_str()) {
                    order.push(name.as_str());
                }
            }
            FixedTerm::Interaction(vars) => {
                for name in vars {
                    if !order.contains(&name.as_str()) {
                        order.push(name.as_str());
                    }
                }
            }
            FixedTerm::Intercept | FixedTerm::NoIntercept => {}
        }
    }
    order
}

fn visit_interaction_components(
    vars: &[String],
    index: usize,
    current: &mut Vec<String>,
    lower_terms: &[Vec<String>],
    intersection: &mut Option<std::collections::BTreeSet<String>>,
) {
    if index == vars.len() {
        if current.is_empty()
            || lower_terms.iter().any(|term| {
                current
                    .iter()
                    .all(|name| term.iter().any(|candidate| candidate == name))
            })
        {
            return;
        }
        let component = current
            .iter()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>();
        match intersection {
            Some(existing) => existing.retain(|name| component.contains(name)),
            None => *intersection = Some(component),
        }
        return;
    }

    visit_interaction_components(vars, index + 1, current, lower_terms, intersection);
    current.push(vars[index].clone());
    visit_interaction_components(vars, index + 1, current, lower_terms, intersection);
    current.pop();
}

fn append_streamed_interaction_products(
    factors: &[Vec<StreamedFactorColumn>],
    index: usize,
    current: &mut Vec<StreamedFactorColumn>,
    column_names: &mut Vec<String>,
    rows: &mut [Vec<(usize, f64)>],
) {
    if index == factors.len() {
        if current.is_empty() {
            return;
        }
        let column = push_column_name(
            column_names,
            current
                .iter()
                .map(|factor| factor.label.as_str())
                .collect::<Vec<_>>()
                .join(":"),
        );

        for row in 0..rows.len() {
            let value = current
                .iter()
                .map(|factor| factor.values[row])
                .product::<f64>();
            if value != 0.0 {
                rows[row].push((column, value));
            }
        }
        return;
    }

    for factor in &factors[index] {
        current.push(factor.clone());
        append_streamed_interaction_products(factors, index + 1, current, column_names, rows);
        current.pop();
    }
}

impl FixedDesignBackend for StreamedFixedDesign {
    fn storage(&self) -> FixedDesignStorage {
        FixedDesignStorage::Streamed
    }

    fn n_obs(&self) -> usize {
        self.n_obs
    }

    fn n_cols(&self) -> usize {
        self.column_names.len()
    }

    fn column_names(&self) -> &[String] {
        &self.column_names
    }

    fn materialize_dense(&self) -> DMatrix<f64> {
        let mut x = DMatrix::zeros(self.n_obs, self.n_cols());
        for (row_index, row) in self.rows.iter().enumerate() {
            for &(column, value) in row {
                x[(row_index, column)] += value;
            }
        }
        x
    }

    fn xtx(&self) -> DMatrix<f64> {
        let p = self.n_cols();
        let mut xtx = DMatrix::zeros(p, p);
        for row in &self.rows {
            for &(col_i, value_i) in row {
                for &(col_j, value_j) in row {
                    xtx[(col_i, col_j)] += value_i * value_j;
                }
            }
        }
        xtx
    }

    fn xty(&self, y: &DVector<f64>) -> Result<DVector<f64>> {
        self.validate_response_len(y.len(), "response")?;
        let mut xty = DVector::zeros(self.n_cols());
        for (row_index, row) in self.rows.iter().enumerate() {
            let y_value = y[row_index];
            for &(column, value) in row {
                xty[column] += value * y_value;
            }
        }
        Ok(xty)
    }

    fn xt_reterm(&self, re: &ReMat) -> Result<MatrixBlock> {
        self.validate_response_len(re.n_obs(), &format!("random term '{}'", re.grouping_name))?;

        if re.vsize == 1 && xt_reterm_sparse_worthwhile(self.n_cols(), re.n_ranef()) {
            // Accumulate the structural nonzeros only. Per-cell addition
            // order matches the dense loop (observation-major), so the
            // resulting values are bit-identical to the dense path.
            let mut entries = std::collections::BTreeMap::<(usize, usize), f64>::new();
            let mut structural_nnz = 0usize;
            for (obs, row) in self.rows.iter().enumerate() {
                let level = re.refs[obs] as usize;
                let wtz = re.wtz[(0, obs)];
                for &(fixed_col, fixed_value) in row {
                    entries
                        .entry((fixed_col, level))
                        .and_modify(|value| *value += fixed_value * wtz)
                        .or_insert_with(|| {
                            structural_nnz += 1;
                            fixed_value * wtz
                        });
                }
            }
            let density =
                structural_nnz as f64 / ((self.n_cols() as f64) * (re.n_ranef() as f64)).max(1.0);
            if density <= XT_RETERM_SPARSE_MAX_DENSITY {
                let mut coo = CooMatrix::new(self.n_cols(), re.n_ranef());
                for ((row, col), value) in entries {
                    if value != 0.0 {
                        coo.push(row, col, value);
                    }
                }
                return Ok(MatrixBlock::Sparse(CscMatrix::from(&coo)));
            }
            // Fall through to the dense accumulation so the emitted block is
            // built exactly the way the dense backend builds it.
        }

        let mut result = DMatrix::zeros(self.n_cols(), re.n_ranef());
        for (obs, row) in self.rows.iter().enumerate() {
            let level = re.refs[obs] as usize;
            for &(fixed_col, fixed_value) in row {
                for basis_row in 0..re.vsize {
                    result[(fixed_col, level * re.vsize + basis_row)] +=
                        fixed_value * re.wtz[(basis_row, obs)];
                }
            }
        }
        Ok(MatrixBlock::Dense(result))
    }

    fn row_dot_beta(&self, row: usize, beta: &DVector<f64>) -> Result<f64> {
        if row >= self.n_obs {
            return Err(MixedModelError::InvalidArgument(format!(
                "fixed-effect row {row} is out of bounds for {} rows",
                self.n_obs
            )));
        }
        if beta.len() != self.n_cols() {
            return Err(MixedModelError::DimensionMismatch(format!(
                "fixed-effect design has {} columns but beta has {} entries",
                self.n_cols(),
                beta.len()
            )));
        }

        Ok(self.rows[row]
            .iter()
            .map(|&(column, value)| value * beta[column])
            .sum())
    }
}

fn dense_bytes(n_rows: usize, n_cols: usize) -> u128 {
    (n_rows as u128)
        .saturating_mul(n_cols as u128)
        .saturating_mul(std::mem::size_of::<f64>() as u128)
}

/// A sparse `X'Z` cross-product only pays off once the dense block is large
/// enough to matter; below this size the dense block is cheap and avoids
/// sparse overhead in the blocked factorization. Density above the cap means
/// the block is effectively dense and the sparse representation would only
/// add indirection.
const XT_RETERM_SPARSE_MIN_DENSE_BYTES: u128 = 512 * 1024;
const XT_RETERM_SPARSE_MAX_DENSITY: f64 = 0.25;

fn xt_reterm_sparse_worthwhile(n_cols: usize, n_ranef: usize) -> bool {
    dense_bytes(n_cols, n_ranef) >= XT_RETERM_SPARSE_MIN_DENSE_BYTES
}

fn validate_sqrt_weights(sqrt_weights: &DVector<f64>, n_obs: usize) -> Result<()> {
    validate_design_vector("sqrt weights", sqrt_weights, n_obs, false)
}

fn validate_column_selection(columns: &[usize], n_cols: usize) -> Result<()> {
    let mut seen = vec![false; n_cols];
    for &column in columns {
        if column >= n_cols {
            return Err(MixedModelError::DimensionMismatch(format!(
                "fixed-effect column selection references column {column}, but design has {n_cols} column(s)"
            )));
        }
        if seen[column] {
            return Err(MixedModelError::InvalidArgument(format!(
                "fixed-effect column selection contains duplicate column {column}"
            )));
        }
        seen[column] = true;
    }
    Ok(())
}

fn validate_design_vector(
    label: &str,
    values: &DVector<f64>,
    n_obs: usize,
    allow_negative: bool,
) -> Result<()> {
    if values.len() != n_obs {
        return Err(MixedModelError::DimensionMismatch(format!(
            "{label} length ({}) does not match number of observations ({n_obs})",
            values.len()
        )));
    }
    for (idx, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(MixedModelError::InvalidArgument(format!(
                "{label} at index {idx} must be finite (got {value})"
            )));
        }
        if !allow_negative && value < 0.0 {
            return Err(MixedModelError::InvalidArgument(format!(
                "{label} at index {idx} must be non-negative (got {value})"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formula::parse_formula;
    use crate::model::data::CategoricalContrast;
    use crate::types::FeTerm;
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn dense_summary_reports_materialized_size() {
        let design = DenseFixedDesign::new(
            DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 1.0, 3.0, 1.0, 4.0]),
            vec!["intercept".to_string(), "x".to_string()],
        )
        .unwrap();

        assert_eq!(design.summary().storage, FixedDesignStorage::Dense);
        assert_eq!(design.summary().dense_bytes, 3 * 2 * 8);
    }

    #[test]
    fn dense_backend_cross_products_match_matrix_algebra() {
        let x = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);
        let design =
            DenseFixedDesign::new(x.clone(), vec!["intercept".into(), "x".into()]).unwrap();
        let y = DVector::from_column_slice(&[10.0, 20.0, 30.0]);

        assert_eq!(design.xtx(), x.transpose() * &x);
        assert_eq!(design.xty(&y).unwrap(), x.transpose() * y);
    }

    #[test]
    fn dense_backend_validates_dimensions() {
        let err = DenseFixedDesign::new(DMatrix::zeros(2, 2), vec!["x".to_string()]).unwrap_err();
        assert!(matches!(err, MixedModelError::DimensionMismatch(_)));

        let design = DenseFixedDesign::new(DMatrix::zeros(2, 1), vec!["x".to_string()]).unwrap();
        let err = design
            .xty(&DVector::from_column_slice(&[1.0, 2.0, 3.0]))
            .unwrap_err();
        assert!(matches!(err, MixedModelError::DimensionMismatch(_)));
    }

    #[test]
    fn dense_backend_selects_columns_in_requested_order() {
        let design = DenseFixedDesign::new(
            DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            vec!["a".into(), "b".into(), "c".into()],
        )
        .unwrap();

        let selected = design.select_columns(&[2, 0]).unwrap();

        assert_eq!(selected.column_names(), &["c".to_string(), "a".to_string()]);
        assert_eq!(
            selected.matrix(),
            &DMatrix::from_row_slice(2, 2, &[3.0, 1.0, 6.0, 4.0])
        );
    }

    fn toy_random_intercept(n_obs: usize) -> ReMat {
        ReMat::new(
            "subject".to_string(),
            (0..n_obs).map(|obs| (obs % 2) as u32).collect(),
            vec!["s0".to_string(), "s1".to_string()],
            vec!["(Intercept)".to_string()],
            DMatrix::from_element(1, n_obs, 1.0),
        )
    }

    #[test]
    fn streamed_backend_matches_dense_cross_products() {
        let x = DMatrix::from_row_slice(
            5,
            4,
            &[
                1.0, 0.0, 0.0, 2.0, //
                1.0, 1.0, 0.0, 3.0, //
                1.0, 0.0, 1.0, 4.0, //
                1.0, 0.0, 0.0, 5.0, //
                1.0, 1.0, 0.0, 6.0, //
            ],
        );
        let names = vec![
            "(Intercept)".to_string(),
            "sku: A".to_string(),
            "sku: B".to_string(),
            "x".to_string(),
        ];
        let dense = DenseFixedDesign::new(x.clone(), names.clone()).unwrap();
        let streamed = StreamedFixedDesign::from_dense(&x, names).unwrap();
        let y = DVector::from_column_slice(&[10.0, 20.0, 30.0, 40.0, 50.0]);

        assert_eq!(streamed.summary().storage, FixedDesignStorage::Streamed);
        assert_eq!(streamed.materialize_dense(), x);
        assert_eq!(streamed.xtx(), dense.xtx());
        assert_eq!(streamed.xty(&y).unwrap(), dense.xty(&y).unwrap());
    }

    #[test]
    fn streamed_backend_crosses_random_terms_without_dense_x() {
        let rows = vec![
            vec![(0, 1.0), (3, 2.0)],
            vec![(0, 1.0), (1, 1.0), (3, 3.0)],
            vec![(0, 1.0), (2, 1.0), (3, 4.0)],
            vec![(0, 1.0), (3, 5.0)],
            vec![(0, 1.0), (1, 1.0), (3, 6.0)],
        ];
        let names = vec![
            "(Intercept)".to_string(),
            "sku: A".to_string(),
            "sku: B".to_string(),
            "x".to_string(),
        ];
        let streamed = StreamedFixedDesign::new(5, names.clone(), rows).unwrap();
        let dense = DenseFixedDesign::new(streamed.materialize_dense(), names).unwrap();
        let re = ReMat::new(
            "purchaser".to_string(),
            vec![0, 0, 1, 1, 1],
            vec!["p0".to_string(), "p1".to_string()],
            vec!["(Intercept)".to_string(), "year".to_string()],
            DMatrix::from_row_slice(
                2,
                5,
                &[
                    1.0, 1.0, 1.0, 1.0, 1.0, //
                    0.0, 1.0, 0.0, 1.0, 2.0, //
                ],
            ),
        );

        assert_eq!(
            streamed.xt_reterm(&re).unwrap().as_dense(),
            dense.xt_reterm(&re).unwrap().as_dense()
        );
    }

    #[test]
    fn streamed_backend_row_dot_beta_uses_active_entries_only() {
        let design = StreamedFixedDesign::new(
            3,
            vec!["intercept".into(), "a".into(), "b".into()],
            vec![
                vec![(0, 1.0), (1, 2.0)],
                vec![(0, 1.0), (2, 3.0)],
                vec![(0, 1.0)],
            ],
        )
        .unwrap();
        let beta = DVector::from_column_slice(&[10.0, 2.0, 4.0]);

        assert_eq!(design.row_dot_beta(0, &beta).unwrap(), 14.0);
        assert_eq!(design.row_dot_beta(1, &beta).unwrap(), 22.0);
        assert_eq!(design.row_dot_beta(2, &beta).unwrap(), 10.0);
    }

    #[test]
    fn streamed_backend_validates_rows_and_canonicalizes_duplicates() {
        let err =
            StreamedFixedDesign::new(1, vec!["x".to_string()], vec![vec![(1, 1.0)]]).unwrap_err();
        assert!(matches!(err, MixedModelError::DimensionMismatch(_)));

        let design = StreamedFixedDesign::new(
            1,
            vec!["x".to_string()],
            vec![vec![(0, 1.0), (0, 2.0), (0, -3.0)]],
        )
        .unwrap();
        assert!(design.rows()[0].is_empty());
    }

    #[test]
    fn streamed_backend_selects_columns_without_materializing_dense_x() {
        let design = StreamedFixedDesign::new(
            3,
            vec!["intercept".into(), "a".into(), "b".into(), "x".into()],
            vec![
                vec![(0, 1.0), (3, 2.0)],
                vec![(0, 1.0), (1, 1.0), (3, 3.0)],
                vec![(0, 1.0), (2, 1.0), (3, 4.0)],
            ],
        )
        .unwrap();

        let selected = design.select_columns(&[3, 0]).unwrap();

        assert_eq!(
            selected.column_names(),
            &["x".to_string(), "intercept".to_string()]
        );
        assert_eq!(
            selected.rows(),
            &[
                vec![(0, 2.0), (1, 1.0)],
                vec![(0, 3.0), (1, 1.0)],
                vec![(0, 4.0), (1, 1.0)],
            ]
        );
    }

    #[test]
    fn fixed_design_rejects_invalid_column_selection() {
        let design = FixedDesign::streamed(
            1,
            vec!["a".into(), "b".into()],
            vec![vec![(0, 1.0), (1, 2.0)]],
        )
        .unwrap();

        let out_of_bounds = design.select_columns(&[2]).unwrap_err();
        assert!(matches!(
            out_of_bounds,
            MixedModelError::DimensionMismatch(_)
        ));

        let duplicate = design.select_columns(&[1, 1]).unwrap_err();
        assert!(matches!(duplicate, MixedModelError::InvalidArgument(_)));
    }

    #[test]
    fn streamed_backend_reports_active_entry_density() {
        let design = StreamedFixedDesign::new(
            3,
            vec!["intercept".into(), "a".into(), "b".into(), "x".into()],
            vec![
                vec![(0, 1.0), (3, 2.0)],
                vec![(0, 1.0), (1, 1.0), (3, 3.0)],
                vec![(0, 1.0)],
            ],
        )
        .unwrap();

        assert_eq!(design.active_entries(), 6);
        assert_eq!(design.density(), 0.5);
    }

    #[test]
    fn fixed_design_policy_forces_dense_backend() {
        let formula = parse_formula("y ~ 1 + x + sku").unwrap();
        let mut data = DataFrame::new();
        data.add_numeric("y", vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        data.add_numeric("x", vec![2.0, 0.0, 3.0, 4.0]).unwrap();
        data.add_categorical(
            "sku",
            vec!["ref".into(), "a".into(), "b".into(), "a".into()],
        )
        .unwrap();

        let design = build_fixed_effects_design_with_policy(
            &formula,
            &data,
            FixedDesignBuildPolicy::dense(),
        )
        .unwrap();

        assert_eq!(design.storage(), FixedDesignStorage::Dense);
        assert_eq!(design.column_names().len(), 4);
        assert!(design.as_dense().is_some());
    }

    #[test]
    fn fixed_design_policy_forces_streamed_backend() {
        let formula = parse_formula("y ~ 1 + x").unwrap();
        let mut data = DataFrame::new();
        data.add_numeric("y", vec![1.0, 2.0, 3.0]).unwrap();
        data.add_numeric("x", vec![2.0, 3.0, 4.0]).unwrap();

        let design = build_fixed_effects_design_with_policy(
            &formula,
            &data,
            FixedDesignBuildPolicy::streamed(),
        )
        .unwrap();

        assert_eq!(design.storage(), FixedDesignStorage::Streamed);
        assert!(design.as_streamed().is_some());
    }

    #[test]
    fn fixed_design_auto_policy_keeps_small_numeric_design_dense() {
        let formula = parse_formula("y ~ 1 + x").unwrap();
        let mut data = DataFrame::new();
        data.add_numeric("y", vec![1.0, 2.0, 3.0]).unwrap();
        data.add_numeric("x", vec![2.0, 3.0, 4.0]).unwrap();

        let design = build_fixed_effects_design(&formula, &data).unwrap();

        assert_eq!(design.storage(), FixedDesignStorage::Dense);
        assert!(design.as_dense().is_some());
    }

    #[test]
    fn fixed_design_auto_policy_streams_high_cardinality_factor() {
        let n_levels = 256usize;
        let n_obs = 512usize;
        let formula = parse_formula("y ~ 1 + sku").unwrap();
        let mut data = DataFrame::new();
        data.add_numeric("y", (0..n_obs).map(|idx| idx as f64).collect())
            .unwrap();
        data.add_categorical(
            "sku",
            (0..n_obs)
                .map(|idx| format!("sku{}", idx % n_levels))
                .collect(),
        )
        .unwrap();

        let design = build_fixed_effects_design(&formula, &data).unwrap();

        assert_eq!(design.storage(), FixedDesignStorage::Streamed);
        let streamed = design.as_streamed().unwrap();
        assert_eq!(streamed.n_cols(), n_levels);
        assert!(streamed.density() < FixedDesignBuildPolicy::auto().max_streamed_density);
        assert!(streamed.rows().iter().all(|row| row.len() <= 2));
    }

    #[test]
    fn fixed_design_auto_policy_streams_when_dense_bytes_exceed_limit() {
        let formula = parse_formula("y ~ 1 + x").unwrap();
        let mut data = DataFrame::new();
        data.add_numeric("y", vec![1.0, 2.0, 3.0]).unwrap();
        data.add_numeric("x", vec![2.0, 3.0, 4.0]).unwrap();

        let policy = FixedDesignBuildPolicy::auto().with_max_dense_bytes(8);
        let design = build_fixed_effects_design_with_policy(&formula, &data, policy).unwrap();

        assert_eq!(design.storage(), FixedDesignStorage::Streamed);
    }

    #[test]
    fn fixed_design_policy_validates_density_threshold() {
        let formula = parse_formula("y ~ 1 + x").unwrap();
        let mut data = DataFrame::new();
        data.add_numeric("y", vec![1.0, 2.0]).unwrap();
        data.add_numeric("x", vec![2.0, 3.0]).unwrap();
        let policy = FixedDesignBuildPolicy::auto().with_max_streamed_density(1.5);

        let err = build_fixed_effects_design_with_policy(&formula, &data, policy).unwrap_err();

        assert!(matches!(err, MixedModelError::InvalidArgument(_)));
    }

    #[test]
    fn streamed_builder_matches_dense_main_effect_treatment_coding() {
        let formula = parse_formula("y ~ 1 + x + sku").unwrap();
        let mut data = DataFrame::new();
        data.add_numeric("y", vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        data.add_numeric("x", vec![2.0, 0.0, 3.0, 4.0]).unwrap();
        data.add_categorical(
            "sku",
            vec!["ref".into(), "a".into(), "b".into(), "a".into()],
        )
        .unwrap();

        let streamed = build_streamed_fixed_effects_design(&formula, &data).unwrap();
        let expected = DMatrix::from_row_slice(
            4,
            4,
            &[
                1.0, 2.0, 0.0, 0.0, //
                1.0, 0.0, 1.0, 0.0, //
                1.0, 3.0, 0.0, 1.0, //
                1.0, 4.0, 1.0, 0.0, //
            ],
        );

        assert_eq!(
            streamed.column_names(),
            &[
                "(Intercept)".to_string(),
                "x".to_string(),
                "sku: a".to_string(),
                "sku: b".to_string(),
            ]
        );
        assert_eq!(streamed.materialize_dense(), expected);
    }

    #[test]
    fn streamed_builder_matches_dense_interaction_order_and_values() {
        let formula = parse_formula("y ~ 0 + x:sku").unwrap();
        let mut data = DataFrame::new();
        data.add_numeric("y", vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        data.add_numeric("x", vec![2.0, 0.0, 3.0, 4.0]).unwrap();
        data.add_categorical(
            "sku",
            vec!["ref".into(), "a".into(), "b".into(), "a".into()],
        )
        .unwrap();

        let streamed = build_streamed_fixed_effects_design(&formula, &data).unwrap();
        let expected = DMatrix::from_row_slice(
            4,
            3,
            &[
                2.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, //
                0.0, 0.0, 3.0, //
                0.0, 4.0, 0.0, //
            ],
        );

        assert_eq!(
            streamed.column_names(),
            &[
                "x:sku: ref".to_string(),
                "x:sku: a".to_string(),
                "x:sku: b".to_string(),
            ]
        );
        assert_eq!(streamed.materialize_dense(), expected);
    }

    #[test]
    fn non_marginal_interaction_matches_r_full_dummy_expansion() {
        let temperature_levels = ["175", "185", "195", "205", "215", "225"]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        let recipe_levels = ["A", "B", "C"]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        let mut temperature = Vec::new();
        let mut recipe = Vec::new();
        for recipe_level in &recipe_levels {
            for temperature_level in &temperature_levels {
                temperature.push(temperature_level.clone());
                recipe.push(recipe_level.clone());
            }
        }
        let mut data = DataFrame::new();
        data.add_numeric("angle", vec![0.0; temperature.len()])
            .unwrap();
        data.add_categorical_with_contrast(
            "temperature",
            temperature.clone(),
            temperature_levels.clone(),
            CategoricalContrast::polynomial(temperature_levels.clone()).unwrap(),
        )
        .unwrap();
        data.add_categorical_with_levels("recipe", recipe.clone(), recipe_levels.clone())
            .unwrap();

        let non_marginal = build_streamed_fixed_effects_design(
            &parse_formula("angle ~ temperature + recipe:temperature").unwrap(),
            &data,
        )
        .unwrap();
        assert_eq!(non_marginal.n_cols(), 18);
        assert_eq!(
            &non_marginal.column_names()[..6],
            &[
                "(Intercept)".to_string(),
                "temperature: .L".to_string(),
                "temperature: .Q".to_string(),
                "temperature: .C".to_string(),
                "temperature: ^4".to_string(),
                "temperature: ^5".to_string(),
            ]
        );
        let expected_interaction_names = temperature_levels
            .iter()
            .flat_map(|temperature| {
                ["B", "C"]
                    .into_iter()
                    .map(move |recipe| format!("temperature: {temperature}:recipe: {recipe}"))
            })
            .collect::<Vec<_>>();
        assert_eq!(
            &non_marginal.column_names()[6..],
            expected_interaction_names
        );

        // Independent dummy-product oracle: every B/C row activates exactly
        // the column for its full temperature level and treatment-coded
        // recipe; A rows activate none of the 12 interaction columns.
        let x = non_marginal.materialize_dense();
        for row in 0..temperature.len() {
            let active = (6..18)
                .filter(|&column| x[(row, column)] == 1.0)
                .collect::<Vec<_>>();
            if recipe[row] == "A" {
                assert!(active.is_empty());
            } else {
                let temperature_index = temperature_levels
                    .iter()
                    .position(|level| level == &temperature[row])
                    .unwrap();
                let recipe_index = usize::from(recipe[row] == "C");
                assert_eq!(active, vec![6 + 2 * temperature_index + recipe_index]);
            }
        }

        let marginal = build_streamed_fixed_effects_design(
            &parse_formula("angle ~ temperature + recipe + recipe:temperature").unwrap(),
            &data,
        )
        .unwrap();
        assert_eq!(marginal.n_cols(), 18);
        assert_eq!(
            marginal
                .column_names()
                .iter()
                .filter(|name| name.contains(":recipe:"))
                .count(),
            10,
            "with both main effects present, R uses reduced contrasts in the interaction"
        );
    }

    #[test]
    fn streamed_builder_cross_products_match_materialized_formula_design() {
        let formula = parse_formula("y ~ 1 + x + sku + x:sku").unwrap();
        let mut data = DataFrame::new();
        data.add_numeric("y", vec![1.0, 2.0, 3.0, 4.0, 5.0])
            .unwrap();
        data.add_numeric("x", vec![2.0, 0.0, 3.0, 4.0, 5.0])
            .unwrap();
        data.add_categorical(
            "sku",
            vec!["ref".into(), "a".into(), "b".into(), "a".into(), "b".into()],
        )
        .unwrap();

        let streamed = build_streamed_fixed_effects_design(&formula, &data).unwrap();
        let dense = DenseFixedDesign::new(
            streamed.materialize_dense(),
            streamed.column_names().to_vec(),
        )
        .unwrap();
        let y = DVector::from_column_slice(data.numeric("y").unwrap());
        let re = toy_random_intercept(data.nrow());

        assert_eq!(streamed.xtx(), dense.xtx());
        assert_eq!(streamed.xty(&y).unwrap(), dense.xty(&y).unwrap());
        assert_eq!(
            streamed.xt_reterm(&re).unwrap().as_dense(),
            dense.xt_reterm(&re).unwrap().as_dense()
        );
    }

    #[test]
    fn streamed_weighted_cross_products_match_dense_weighted_design() {
        let formula = parse_formula("y ~ 1 + x + sku").unwrap();
        let mut data = DataFrame::new();
        data.add_numeric("y", vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        data.add_numeric("x", vec![2.0, 0.0, 3.0, 4.0]).unwrap();
        data.add_categorical(
            "sku",
            vec!["ref".into(), "a".into(), "b".into(), "a".into()],
        )
        .unwrap();

        let sqrt_weights = DVector::from_column_slice(&[1.0, 2.0, 0.5, 3.0]);
        let y = DVector::from_column_slice(data.numeric("y").unwrap());
        let y_weighted = y.component_mul(&sqrt_weights);

        let streamed = build_streamed_fixed_effects_design(&formula, &data).unwrap();
        let dense = DenseFixedDesign::new(
            streamed.materialize_dense(),
            streamed.column_names().to_vec(),
        )
        .unwrap();
        let streamed_weighted = streamed.with_sqrt_weights(&sqrt_weights).unwrap();
        let dense_weighted = dense.with_sqrt_weights(&sqrt_weights).unwrap();

        assert_eq!(streamed_weighted.xtx(), dense_weighted.xtx());
        assert_eq!(
            streamed_weighted.xty(&y_weighted).unwrap(),
            dense_weighted.xty(&y_weighted).unwrap()
        );
    }

    #[test]
    fn streamed_weighted_xtz_matches_dense_weighted_design() {
        let formula = parse_formula("y ~ 1 + x + sku").unwrap();
        let mut data = DataFrame::new();
        data.add_numeric("y", vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        data.add_numeric("x", vec![2.0, 0.0, 3.0, 4.0]).unwrap();
        data.add_categorical(
            "sku",
            vec!["ref".into(), "a".into(), "b".into(), "a".into()],
        )
        .unwrap();

        let sqrt_weights = DVector::from_column_slice(&[1.0, 2.0, 0.5, 3.0]);
        let streamed = build_streamed_fixed_effects_design(&formula, &data).unwrap();
        let dense = DenseFixedDesign::new(
            streamed.materialize_dense(),
            streamed.column_names().to_vec(),
        )
        .unwrap();
        let streamed_weighted = streamed.with_sqrt_weights(&sqrt_weights).unwrap();
        let dense_weighted = dense.with_sqrt_weights(&sqrt_weights).unwrap();

        let mut re = ReMat::new(
            "subject".to_string(),
            vec![0, 0, 1, 1],
            vec!["s0".to_string(), "s1".to_string()],
            vec!["(Intercept)".to_string(), "year".to_string()],
            DMatrix::from_row_slice(
                2,
                4,
                &[
                    1.0, 1.0, 1.0, 1.0, //
                    0.0, 1.0, 0.0, 2.0, //
                ],
            ),
        );
        re.reweight(&sqrt_weights);

        assert_eq!(
            streamed_weighted.xt_reterm(&re).unwrap().as_dense(),
            dense_weighted.xt_reterm(&re).unwrap().as_dense()
        );
    }

    #[test]
    fn streamed_weighting_validates_sqrt_weights() {
        let design = StreamedFixedDesign::new(
            2,
            vec!["intercept".into()],
            vec![vec![(0, 1.0)], vec![(0, 1.0)]],
        )
        .unwrap();

        let wrong_len = design
            .with_sqrt_weights(&DVector::from_column_slice(&[1.0]))
            .unwrap_err();
        assert!(matches!(wrong_len, MixedModelError::DimensionMismatch(_)));

        let negative = design
            .with_sqrt_weights(&DVector::from_column_slice(&[1.0, -1.0]))
            .unwrap_err();
        assert!(matches!(negative, MixedModelError::InvalidArgument(_)));
    }

    #[test]
    fn streamed_builder_handles_high_cardinality_factor_without_dense_x() {
        let n_levels = 2_000usize;
        let n_obs = 4_000usize;
        let formula = parse_formula("y ~ 1 + sku").unwrap();
        let mut data = DataFrame::new();
        data.add_numeric("y", (0..n_obs).map(|idx| idx as f64).collect())
            .unwrap();
        data.add_categorical(
            "sku",
            (0..n_obs)
                .map(|idx| format!("sku{}", idx % n_levels))
                .collect(),
        )
        .unwrap();

        let streamed = build_streamed_fixed_effects_design(&formula, &data).unwrap();
        let summary = streamed.summary();

        assert_eq!(summary.storage, FixedDesignStorage::Streamed);
        assert_eq!(summary.n_obs, n_obs);
        assert_eq!(summary.n_cols, n_levels);
        assert_eq!(summary.dense_bytes, n_obs as u128 * n_levels as u128 * 8);

        // Intercept plus at most one non-reference treatment dummy per row.
        assert!(streamed.rows().iter().all(|row| row.len() <= 2));
        assert_eq!(streamed.rows()[0], vec![(0, 1.0)]);
        assert_eq!(streamed.rows()[1], vec![(0, 1.0), (1, 1.0)]);
    }

    #[test]
    fn streamed_builder_preserves_feterm_rank_and_pivot_order() {
        let formula = parse_formula("y ~ 1 + x + x_dup + sku").unwrap();
        let mut data = DataFrame::new();
        data.add_numeric("y", vec![1.0, 2.0, 3.0, 4.0, 5.0])
            .unwrap();
        data.add_numeric("x", vec![1.0, 2.0, 3.0, 4.0, 5.0])
            .unwrap();
        data.add_numeric("x_dup", vec![1.0, 2.0, 3.0, 4.0, 5.0])
            .unwrap();
        data.add_categorical(
            "sku",
            vec!["ref".into(), "a".into(), "b".into(), "a".into(), "b".into()],
        )
        .unwrap();

        let streamed = build_streamed_fixed_effects_design(&formula, &data).unwrap();
        let dense = DenseFixedDesign::new(
            streamed.materialize_dense(),
            streamed.column_names().to_vec(),
        )
        .unwrap();

        let streamed_fe = FeTerm::new(
            streamed.materialize_dense(),
            streamed.column_names().to_vec(),
        );
        let dense_fe = FeTerm::new(dense.matrix().clone(), dense.column_names().to_vec());

        assert_eq!(streamed_fe.rank, dense_fe.rank);
        assert_eq!(streamed_fe.piv, dense_fe.piv);
        assert_eq!(streamed_fe.cnames, dense_fe.cnames);
        assert!(!streamed_fe.is_full_rank());
    }

    #[test]
    fn fixed_design_enum_dispatches_to_streamed_backend() {
        let design = FixedDesign::streamed(
            2,
            vec!["intercept".into(), "x".into()],
            vec![vec![(0, 1.0), (1, 2.0)], vec![(0, 1.0), (1, 3.0)]],
        )
        .unwrap();
        let beta = DVector::from_column_slice(&[10.0, 2.0]);

        assert_eq!(design.storage(), FixedDesignStorage::Streamed);
        assert_eq!(design.row_dot_beta(1, &beta).unwrap(), 16.0);
        assert!(design.as_streamed().is_some());
    }

    #[test]
    fn compiled_design_validates_shared_observation_count() {
        let response = DVector::from_column_slice(&[1.0, 2.0, 3.0]);
        let err = CompiledMixedModelDesign::from_dense(
            response,
            DMatrix::from_row_slice(2, 1, &[1.0, 1.0]),
            vec!["(Intercept)".to_string()],
            vec![toy_random_intercept(3)],
        )
        .unwrap_err();

        assert!(matches!(err, MixedModelError::DimensionMismatch(_)));
    }

    #[test]
    fn compiled_design_accepts_frontend_supplied_offset_and_weights() {
        let response = DVector::from_column_slice(&[1.0, 2.0, 3.0]);
        let design = CompiledMixedModelDesign::from_dense(
            response,
            DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 1.0, 3.0, 1.0, 4.0]),
            vec!["(Intercept)".to_string(), "x".to_string()],
            vec![toy_random_intercept(3)],
        )
        .unwrap()
        .with_response_name("y")
        .with_offset(DVector::from_column_slice(&[0.1, 0.2, 0.3]))
        .unwrap()
        .with_case_weights(DVector::from_column_slice(&[1.0, 2.0, 3.0]))
        .unwrap();

        assert_eq!(design.response_name(), Some("y"));
        assert_eq!(design.n_obs(), 3);
        assert_eq!(design.n_fixed_cols(), 2);
        assert_eq!(design.offset()[2], 0.3);
        assert_eq!(design.case_weights().unwrap()[1], 2.0);
    }

    #[test]
    fn compiled_design_fixed_eta_includes_offset_without_dense_materialization() {
        let fixed = FixedDesign::streamed(
            3,
            vec!["intercept".into(), "x".into()],
            vec![
                vec![(0, 1.0), (1, 2.0)],
                vec![(0, 1.0), (1, 3.0)],
                vec![(0, 1.0), (1, 4.0)],
            ],
        )
        .unwrap();
        let design = CompiledMixedModelDesign::new(
            DVector::from_column_slice(&[1.0, 2.0, 3.0]),
            fixed,
            vec![toy_random_intercept(3)],
        )
        .unwrap()
        .with_offset(DVector::from_column_slice(&[10.0, 20.0, 30.0]))
        .unwrap();
        let eta = design
            .fixed_linear_predictor(&DVector::from_column_slice(&[1.0, 2.0]))
            .unwrap();

        assert_eq!(eta, DVector::from_column_slice(&[15.0, 27.0, 39.0]));
    }
}
