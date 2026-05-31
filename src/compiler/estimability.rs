use serde::{Deserialize, Serialize};

use nalgebra::{DMatrix, DVector};

use super::diagnostics::Diagnostic;

/// Context-specific estimability assessment. The variants avoid representing
/// nonsensical free products such as random-effect basis dependence on a fixed
/// contrast.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "estimand", content = "assessment", rename_all = "snake_case")]
pub enum EstimabilityAssessment {
    FixedContrast(FixedContrastEstimability),
    FixedTerm(FixedTermEstimability),
    RandomVarianceDirection(RandomVarianceEstimability),
    RandomCovarianceParameter(RandomCovarianceEstimability),
    KernelPath(KernelPathEstimability),
}

/// Shared status vocabulary. Not all statuses are valid for all assessment
/// variants; constructors should enforce context-specific use as the module
/// matures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EstimabilityStatus {
    Estimable,
    PartiallyEstimable,
    WeaklyEstimable,
    NotEstimable,
    BasisDependent,
    NotAssessed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FixedContrastEstimability {
    pub label: String,
    pub status: EstimabilityStatus,
    pub rank: Option<usize>,
    pub requested_rank: Option<usize>,
    pub diagnostics: Vec<Diagnostic>,
}

impl FixedContrastEstimability {
    pub fn estimable(label: impl Into<String>, rank: usize, requested_rank: usize) -> Self {
        Self {
            label: label.into(),
            status: EstimabilityStatus::Estimable,
            rank: Some(rank),
            requested_rank: Some(requested_rank),
            diagnostics: Vec::new(),
        }
    }

    pub fn partially_estimable(
        label: impl Into<String>,
        rank: usize,
        requested_rank: usize,
        diagnostics: Vec<Diagnostic>,
    ) -> Self {
        Self {
            label: label.into(),
            status: EstimabilityStatus::PartiallyEstimable,
            rank: Some(rank),
            requested_rank: Some(requested_rank),
            diagnostics,
        }
    }

    pub fn weakly_estimable(
        label: impl Into<String>,
        rank: usize,
        requested_rank: usize,
        diagnostics: Vec<Diagnostic>,
    ) -> Self {
        Self {
            label: label.into(),
            status: EstimabilityStatus::WeaklyEstimable,
            rank: Some(rank),
            requested_rank: Some(requested_rank),
            diagnostics,
        }
    }

    pub fn not_estimable(
        label: impl Into<String>,
        requested_rank: usize,
        diagnostics: Vec<Diagnostic>,
    ) -> Self {
        Self {
            label: label.into(),
            status: EstimabilityStatus::NotEstimable,
            rank: Some(0),
            requested_rank: Some(requested_rank),
            diagnostics,
        }
    }

    pub fn not_assessed(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            status: EstimabilityStatus::NotAssessed,
            rank: None,
            requested_rank: None,
            diagnostics: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FixedTermEstimability {
    pub term: String,
    pub status: EstimabilityStatus,
    pub aliased_columns: Vec<String>,
    pub diagnostics: Vec<Diagnostic>,
}

impl FixedTermEstimability {
    pub fn estimable(term: impl Into<String>) -> Self {
        Self {
            term: term.into(),
            status: EstimabilityStatus::Estimable,
            aliased_columns: Vec::new(),
            diagnostics: Vec::new(),
        }
    }

    pub fn partially_estimable(
        term: impl Into<String>,
        aliased_columns: Vec<String>,
        diagnostics: Vec<Diagnostic>,
    ) -> Self {
        Self {
            term: term.into(),
            status: EstimabilityStatus::PartiallyEstimable,
            aliased_columns,
            diagnostics,
        }
    }

    pub fn not_estimable(
        term: impl Into<String>,
        aliased_columns: Vec<String>,
        diagnostics: Vec<Diagnostic>,
    ) -> Self {
        Self {
            term: term.into(),
            status: EstimabilityStatus::NotEstimable,
            aliased_columns,
            diagnostics,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RandomVarianceEstimability {
    pub term_id: String,
    pub basis_direction: String,
    pub status: EstimabilityStatus,
    pub diagnostics: Vec<Diagnostic>,
}

impl RandomVarianceEstimability {
    pub fn estimable(term_id: impl Into<String>, basis_direction: impl Into<String>) -> Self {
        Self {
            term_id: term_id.into(),
            basis_direction: basis_direction.into(),
            status: EstimabilityStatus::Estimable,
            diagnostics: Vec::new(),
        }
    }

    pub fn weakly_estimable(
        term_id: impl Into<String>,
        basis_direction: impl Into<String>,
        diagnostics: Vec<Diagnostic>,
    ) -> Self {
        Self {
            term_id: term_id.into(),
            basis_direction: basis_direction.into(),
            status: EstimabilityStatus::WeaklyEstimable,
            diagnostics,
        }
    }

    pub fn basis_dependent(
        term_id: impl Into<String>,
        basis_direction: impl Into<String>,
        diagnostics: Vec<Diagnostic>,
    ) -> Self {
        Self {
            term_id: term_id.into(),
            basis_direction: basis_direction.into(),
            status: EstimabilityStatus::BasisDependent,
            diagnostics,
        }
    }

    pub fn not_estimable(
        term_id: impl Into<String>,
        basis_direction: impl Into<String>,
        diagnostics: Vec<Diagnostic>,
    ) -> Self {
        Self {
            term_id: term_id.into(),
            basis_direction: basis_direction.into(),
            status: EstimabilityStatus::NotEstimable,
            diagnostics,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RandomCovarianceEstimability {
    pub term_id: String,
    pub parameter: String,
    pub status: EstimabilityStatus,
    pub diagnostics: Vec<Diagnostic>,
}

impl RandomCovarianceEstimability {
    pub fn estimable(term_id: impl Into<String>, parameter: impl Into<String>) -> Self {
        Self {
            term_id: term_id.into(),
            parameter: parameter.into(),
            status: EstimabilityStatus::Estimable,
            diagnostics: Vec::new(),
        }
    }

    pub fn basis_dependent(
        term_id: impl Into<String>,
        parameter: impl Into<String>,
        diagnostics: Vec<Diagnostic>,
    ) -> Self {
        Self {
            term_id: term_id.into(),
            parameter: parameter.into(),
            status: EstimabilityStatus::BasisDependent,
            diagnostics,
        }
    }

    pub fn not_estimable(
        term_id: impl Into<String>,
        parameter: impl Into<String>,
        diagnostics: Vec<Diagnostic>,
    ) -> Self {
        Self {
            term_id: term_id.into(),
            parameter: parameter.into(),
            status: EstimabilityStatus::NotEstimable,
            diagnostics,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KernelPathEstimability {
    pub path: String,
    pub status: EstimabilityStatus,
    pub diagnostics: Vec<Diagnostic>,
}

impl KernelPathEstimability {
    pub fn estimable(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            status: EstimabilityStatus::Estimable,
            diagnostics: Vec::new(),
        }
    }

    pub fn not_estimable(path: impl Into<String>, diagnostics: Vec<Diagnostic>) -> Self {
        Self {
            path: path.into(),
            status: EstimabilityStatus::NotEstimable,
            diagnostics,
        }
    }

    pub fn not_assessed(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            status: EstimabilityStatus::NotAssessed,
            diagnostics: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContrastMatrix {
    pub values: DMatrix<f64>,
}

impl ContrastMatrix {
    pub fn new(values: DMatrix<f64>) -> Result<Self, String> {
        if values.nrows() == 0 || values.ncols() == 0 {
            return Err("contrast matrix must have at least one row and one column".to_string());
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err("contrast matrix contains a non-finite value".to_string());
        }
        Ok(Self { values })
    }

    pub fn single_coefficient(index: usize, n_coefficients: usize) -> Result<Self, String> {
        if index >= n_coefficients {
            return Err(format!(
                "coefficient index {index} is out of bounds for {n_coefficients} coefficients"
            ));
        }
        let mut values = DMatrix::zeros(1, n_coefficients);
        values[(0, index)] = 1.0;
        Ok(Self { values })
    }

    pub fn n_contrasts(&self) -> usize {
        self.values.nrows()
    }

    pub fn n_coefficients(&self) -> usize {
        self.values.ncols()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContrastRhs {
    pub values: DVector<f64>,
}

impl ContrastRhs {
    pub fn new(values: DVector<f64>) -> Result<Self, String> {
        if values.iter().any(|value| !value.is_finite()) {
            return Err("contrast right-hand side contains a non-finite value".to_string());
        }
        Ok(Self { values })
    }

    pub fn zeros(n_contrasts: usize) -> Self {
        Self {
            values: DVector::zeros(n_contrasts),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FixedEffectHypothesis {
    pub label: String,
    pub l: ContrastMatrix,
    pub rhs: ContrastRhs,
}

impl FixedEffectHypothesis {
    pub fn new(
        label: impl Into<String>,
        l: ContrastMatrix,
        rhs: ContrastRhs,
    ) -> Result<Self, String> {
        if l.n_contrasts() != rhs.values.len() {
            return Err(format!(
                "contrast matrix has {} row(s), but rhs has length {}",
                l.n_contrasts(),
                rhs.values.len()
            ));
        }
        Ok(Self {
            label: label.into(),
            l,
            rhs,
        })
    }

    pub fn zero_rhs(label: impl Into<String>, l: ContrastMatrix) -> Self {
        let rhs = ContrastRhs::zeros(l.n_contrasts());
        Self {
            label: label.into(),
            l,
            rhs,
        }
    }

    pub fn single_coefficient(
        label: impl Into<String>,
        index: usize,
        n_coefficients: usize,
    ) -> Result<Self, String> {
        Ok(Self::zero_rhs(
            label,
            ContrastMatrix::single_coefficient(index, n_coefficients)?,
        ))
    }

    pub fn n_contrasts(&self) -> usize {
        self.l.n_contrasts()
    }

    pub fn n_coefficients(&self) -> usize {
        self.l.n_coefficients()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InferenceMethod {
    AsymptoticWaldZ,
    Satterthwaite,
    KenwardRoger,
    ParametricBootstrap,
    NotComputed { reason: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FixedEffectTestMethod {
    Auto,
    AsymptoticWaldZ,
    Satterthwaite,
    KenwardRoger,
    ParametricBootstrap,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FixedEffectTermTestType {
    TypeI,
    TypeII,
    TypeIII,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReliabilityGrade {
    High,
    Moderate,
    Low,
    NotAvailable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InferenceStatus {
    Available,
    PValueUnavailable { reason: String },
    NotEstimable { reason: String },
    NotAssessed { reason: String },
    Unsupported { reason: String },
}

impl InferenceStatus {
    pub fn p_value_available(&self) -> bool {
        matches!(self, InferenceStatus::Available)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FixedEffectTest {
    pub hypothesis: FixedEffectHypothesis,
    pub estimates: Vec<f64>,
    pub standard_errors: Vec<Option<f64>>,
    pub statistics: Vec<Option<f64>>,
    pub numerator_df: Option<f64>,
    pub denominator_df: Option<f64>,
    pub p_values: Vec<Option<f64>>,
    pub method: InferenceMethod,
    pub reliability: ReliabilityGrade,
    pub status: InferenceStatus,
    pub estimability: FixedContrastEstimability,
    pub notes: Vec<String>,
}

impl FixedEffectTest {
    pub fn p_value_unavailable_reason(&self) -> Option<&str> {
        match &self.status {
            InferenceStatus::PValueUnavailable { reason }
            | InferenceStatus::NotEstimable { reason }
            | InferenceStatus::NotAssessed { reason }
            | InferenceStatus::Unsupported { reason } => Some(reason.as_str()),
            InferenceStatus::Available => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_effect_hypothesis_validates_dimensions() {
        let l = ContrastMatrix::new(DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0])).unwrap();
        let rhs = ContrastRhs::zeros(1);

        let err = FixedEffectHypothesis::new("bad", l, rhs).unwrap_err();

        assert!(err.contains("rhs has length 1"));
    }

    #[test]
    fn single_coefficient_hypothesis_builds_l_beta_equals_zero() {
        let hypothesis = FixedEffectHypothesis::single_coefficient("x", 1, 3).unwrap();

        assert_eq!(hypothesis.l.values.nrows(), 1);
        assert_eq!(hypothesis.l.values.ncols(), 3);
        assert_eq!(hypothesis.l.values[(0, 1)], 1.0);
        assert_eq!(hypothesis.rhs.values[0], 0.0);
    }

    #[test]
    fn typed_estimability_constructors_keep_context_specific_statuses() {
        let fixed = FixedContrastEstimability::estimable("x", 1, 1);
        let random = RandomVarianceEstimability::basis_dependent("r0", "slope", Vec::new());

        assert_eq!(fixed.status, EstimabilityStatus::Estimable);
        assert_eq!(random.status, EstimabilityStatus::BasisDependent);
    }
}
