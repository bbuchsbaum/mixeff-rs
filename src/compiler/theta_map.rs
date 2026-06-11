use serde::{Deserialize, Serialize};

use super::artifact::ReductionTrigger;
use super::ir::{CovarianceForm, CovarianceSupportStatus, RandomTermIr};

pub const THETA_MAP_SCHEMA: &str = "mixedmodels.theta_map";
pub const THETA_MAP_SCHEMA_VERSION: u32 = 1;

/// Optimizer-facing covariance family.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CovarianceFamily {
    Scalar,
    Diagonal,
    FullCholesky,
    Structured { kind: String },
    ReducedRank { rank: Option<usize> },
    Unsupported { reason: String },
}

/// Sum-typed theta map. Each variant represents a distinct optimization
/// manifold rather than a full covariance with hidden active zeros.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "family", content = "map", rename_all = "snake_case")]
pub enum ThetaMap {
    Scalar(ThetaMapBlock),
    Diagonal(ThetaMapBlock),
    FullCholesky(ThetaMapBlock),
    Structured(ThetaMapBlock),
    ReducedRank(ThetaMapBlock),
    Unsupported(ThetaMapBlock),
}

/// Shared payload for a theta-map variant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThetaMapBlock {
    pub schema_name: String,
    pub schema_version: u32,
    pub term_id: String,
    pub term_index: usize,
    pub group: String,
    pub covariance_family: CovarianceFamily,
    pub support_status: CovarianceSupportStatus,
    pub user_basis: Vec<String>,
    pub optimizer_basis: Vec<String>,
    pub theta_slots: Vec<ThetaSlot>,
    pub source_parmap: Vec<(usize, usize, usize)>,
}

/// One free or inactive optimizer parameter slot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThetaSlot {
    pub global_index: Option<usize>,
    pub local_index: usize,
    pub term_index: usize,
    pub basis_row: usize,
    pub basis_col: usize,
    pub lambda_row: usize,
    pub lambda_col: usize,
    pub name: String,
    pub constraint: ParameterConstraint,
    pub status: ParameterStatus,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParameterConstraint {
    LowerBound { lower: f64 },
    Interval { lower: f64, upper: f64 },
    Unconstrained,
    Fixed { value: f64 },
    NotAssessed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParameterStatus {
    Free,
    Boundary,
    Inactive,
    NotAssessed,
}

/// Explicit covariance-family transition record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CovarianceFamilyTransition {
    pub from: CovarianceFamily,
    pub to: CovarianceFamily,
    pub trigger: ReductionTrigger,
    pub affected_term: String,
    pub dropped_or_reparameterized_slots: Vec<ThetaSlot>,
    pub inference_consequence: String,
}

impl ThetaMap {
    pub fn from_random_term(term_index: usize, term: &RandomTermIr, global_start: usize) -> Self {
        let basis = term
            .basis
            .iter()
            .map(|b| b.name.clone())
            .collect::<Vec<_>>();
        Self::from_random_term_with_optimizer_basis(term_index, term, global_start, basis)
    }

    pub fn from_random_term_with_optimizer_basis(
        term_index: usize,
        term: &RandomTermIr,
        global_start: usize,
        optimizer_basis: Vec<String>,
    ) -> Self {
        let family =
            CovarianceFamily::from_covariance_and_basis(&term.covariance, optimizer_basis.len());
        let block = ThetaMapBlock::from_random_term(
            term_index,
            term,
            global_start,
            &family,
            optimizer_basis,
        );
        match family {
            CovarianceFamily::Scalar => ThetaMap::Scalar(block),
            CovarianceFamily::Diagonal => ThetaMap::Diagonal(block),
            CovarianceFamily::FullCholesky => ThetaMap::FullCholesky(block),
            CovarianceFamily::Structured { .. } => ThetaMap::Structured(block),
            CovarianceFamily::ReducedRank { .. } => ThetaMap::ReducedRank(block),
            CovarianceFamily::Unsupported { .. } => ThetaMap::Unsupported(block),
        }
    }

    /// Theta map for one split (`||`) random term that materialized into the
    /// contiguous `lambda_indices` columns of the shared optimizer block. A
    /// numeric coefficient occupies one column (Scalar); a factor coefficient
    /// expands to one column per level contrast, each with an independent
    /// variance (Diagonal).
    pub fn from_split_random_term_with_optimizer_basis(
        term_index: usize,
        term: &RandomTermIr,
        global_start: usize,
        optimizer_basis: Vec<String>,
        lambda_indices: std::ops::Range<usize>,
    ) -> Self {
        let scalar = lambda_indices.len() == 1;
        let block = ThetaMapBlock::from_split_random_term(
            term_index,
            term,
            global_start,
            optimizer_basis,
            lambda_indices,
        );
        if scalar {
            ThetaMap::Scalar(block)
        } else {
            ThetaMap::Diagonal(block)
        }
    }

    pub fn block(&self) -> &ThetaMapBlock {
        match self {
            ThetaMap::Scalar(block)
            | ThetaMap::Diagonal(block)
            | ThetaMap::FullCholesky(block)
            | ThetaMap::Structured(block)
            | ThetaMap::ReducedRank(block)
            | ThetaMap::Unsupported(block) => block,
        }
    }

    pub fn family(&self) -> CovarianceFamily {
        self.block().covariance_family.clone()
    }

    pub fn n_free(&self) -> usize {
        self.block()
            .theta_slots
            .iter()
            .filter(|slot| slot.status == ParameterStatus::Free)
            .count()
    }
}

impl ThetaMapBlock {
    fn from_random_term(
        term_index: usize,
        term: &RandomTermIr,
        global_start: usize,
        family: &CovarianceFamily,
        optimizer_basis: Vec<String>,
    ) -> Self {
        let user_basis: Vec<String> = term.basis.iter().map(|b| b.name.clone()).collect();
        let theta_slots =
            theta_slots_for_family(term_index, &optimizer_basis, family, global_start);
        let source_parmap = theta_slots
            .iter()
            .filter_map(|slot| {
                slot.global_index
                    .map(|_| (term_index, slot.lambda_row, slot.lambda_col))
            })
            .collect();

        Self {
            schema_name: THETA_MAP_SCHEMA.to_string(),
            schema_version: THETA_MAP_SCHEMA_VERSION,
            term_id: term.id.clone(),
            term_index,
            group: term.group.label(),
            covariance_family: family.clone(),
            support_status: family.support_status(),
            user_basis,
            optimizer_basis,
            theta_slots,
            source_parmap,
        }
    }

    fn from_split_random_term(
        term_index: usize,
        term: &RandomTermIr,
        global_start: usize,
        optimizer_basis: Vec<String>,
        lambda_indices: std::ops::Range<usize>,
    ) -> Self {
        let user_basis: Vec<String> = term.basis.iter().map(|b| b.name.clone()).collect();
        let covariance_family = if lambda_indices.len() == 1 {
            CovarianceFamily::Scalar
        } else {
            CovarianceFamily::Diagonal
        };
        let theta_slots = lambda_indices
            .clone()
            .enumerate()
            .map(|(local, lambda_index)| {
                make_slot_with_lambda(
                    term_index,
                    local,
                    lambda_index,
                    lambda_index,
                    lambda_index,
                    lambda_index,
                    global_start + local,
                    &optimizer_basis,
                )
            })
            .collect();
        let source_parmap = lambda_indices
            .map(|lambda_index| (term_index, lambda_index, lambda_index))
            .collect();

        Self {
            schema_name: THETA_MAP_SCHEMA.to_string(),
            schema_version: THETA_MAP_SCHEMA_VERSION,
            term_id: term.id.clone(),
            term_index,
            group: term.group.label(),
            support_status: covariance_family.support_status(),
            covariance_family,
            user_basis,
            optimizer_basis,
            theta_slots,
            source_parmap,
        }
    }
}

impl From<&CovarianceForm> for CovarianceFamily {
    fn from(value: &CovarianceForm) -> Self {
        match value {
            CovarianceForm::Scalar => CovarianceFamily::Scalar,
            CovarianceForm::Diagonal => CovarianceFamily::Diagonal,
            CovarianceForm::Full => CovarianceFamily::FullCholesky,
            CovarianceForm::Structured { kind } => CovarianceFamily::Structured {
                kind: kind.label().to_string(),
            },
            CovarianceForm::ReducedRank { rank } => CovarianceFamily::ReducedRank { rank: *rank },
            CovarianceForm::Unsupported { reason } => CovarianceFamily::Unsupported {
                reason: reason.clone(),
            },
        }
    }
}

impl CovarianceFamily {
    fn from_covariance_and_basis(covariance: &CovarianceForm, basis_dimension: usize) -> Self {
        match covariance {
            CovarianceForm::Scalar if basis_dimension > 1 => CovarianceFamily::FullCholesky,
            other => CovarianceFamily::from(other),
        }
    }

    pub fn support_status(&self) -> CovarianceSupportStatus {
        match self {
            CovarianceFamily::Scalar
            | CovarianceFamily::Diagonal
            | CovarianceFamily::FullCholesky => CovarianceSupportStatus::Supported,
            CovarianceFamily::Structured { .. } => CovarianceSupportStatus::ParsedRefused,
            CovarianceFamily::ReducedRank { .. } => CovarianceSupportStatus::Future,
            CovarianceFamily::Unsupported { .. } => CovarianceSupportStatus::Unsupported,
        }
    }
}

fn theta_slots_for_family(
    term_index: usize,
    basis: &[String],
    family: &CovarianceFamily,
    global_start: usize,
) -> Vec<ThetaSlot> {
    match family {
        CovarianceFamily::Scalar => scalar_slots(term_index, basis, global_start),
        CovarianceFamily::Diagonal => diagonal_slots(term_index, basis, global_start),
        CovarianceFamily::FullCholesky => full_cholesky_slots(term_index, basis, global_start),
        CovarianceFamily::Structured { kind } => structured_slots(term_index, basis, kind),
        CovarianceFamily::ReducedRank { .. } | CovarianceFamily::Unsupported { .. } => Vec::new(),
    }
}

fn scalar_slots(term_index: usize, basis: &[String], global_start: usize) -> Vec<ThetaSlot> {
    if basis.is_empty() {
        Vec::new()
    } else {
        vec![make_slot(term_index, 0, 0, 0, global_start, basis)]
    }
}

fn diagonal_slots(term_index: usize, basis: &[String], global_start: usize) -> Vec<ThetaSlot> {
    basis
        .iter()
        .enumerate()
        .map(|(local, _)| make_slot(term_index, local, local, local, global_start + local, basis))
        .collect()
}

fn full_cholesky_slots(term_index: usize, basis: &[String], global_start: usize) -> Vec<ThetaSlot> {
    let mut slots = Vec::new();
    let mut local = 0;
    for col in 0..basis.len() {
        for row in col..basis.len() {
            slots.push(make_slot(
                term_index,
                local,
                row,
                col,
                global_start + local,
                basis,
            ));
            local += 1;
        }
    }
    slots
}

fn structured_slots(term_index: usize, basis: &[String], kind: &str) -> Vec<ThetaSlot> {
    if basis.is_empty() {
        return Vec::new();
    }

    let mut slots = vec![ThetaSlot {
        global_index: None,
        local_index: 0,
        term_index,
        basis_row: 0,
        basis_col: 0,
        lambda_row: 0,
        lambda_col: 0,
        name: format!("theta[{term_index}:{kind}.sd]"),
        constraint: ParameterConstraint::LowerBound { lower: 0.0 },
        status: ParameterStatus::NotAssessed,
    }];

    if basis.len() > 1 {
        slots.push(ThetaSlot {
            global_index: None,
            local_index: 1,
            term_index,
            basis_row: 1,
            basis_col: 0,
            lambda_row: 1,
            lambda_col: 0,
            name: format!("theta[{term_index}:{kind}.correlation]"),
            constraint: ParameterConstraint::Interval {
                lower: -1.0,
                upper: 1.0,
            },
            status: ParameterStatus::NotAssessed,
        });
    }

    slots
}

fn make_slot(
    term_index: usize,
    local_index: usize,
    row: usize,
    col: usize,
    global_index: usize,
    basis: &[String],
) -> ThetaSlot {
    make_slot_with_lambda(
        term_index,
        local_index,
        row,
        col,
        row,
        col,
        global_index,
        basis,
    )
}

fn make_slot_with_lambda(
    term_index: usize,
    local_index: usize,
    row: usize,
    col: usize,
    lambda_row: usize,
    lambda_col: usize,
    global_index: usize,
    basis: &[String],
) -> ThetaSlot {
    let row_name = basis.get(row).map(String::as_str).unwrap_or("unknown");
    let col_name = basis.get(col).map(String::as_str).unwrap_or("unknown");
    ThetaSlot {
        global_index: Some(global_index),
        local_index,
        term_index,
        basis_row: row,
        basis_col: col,
        lambda_row,
        lambda_col,
        name: format!("theta[{term_index}:{row_name},{col_name}]"),
        constraint: if row == col {
            ParameterConstraint::LowerBound { lower: 0.0 }
        } else {
            ParameterConstraint::Unconstrained
        },
        status: ParameterStatus::Free,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::compile_formula_ir;
    use crate::formula::parse_formula;

    #[test]
    fn full_cholesky_map_uses_column_major_lower_triangle_order() {
        let formula = parse_formula("y ~ x + (1 + x | subject)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let map = ThetaMap::from_random_term(0, &semantic.random_terms[0], 0);

        let slots = &map.block().theta_slots;
        assert_eq!(slots.len(), 3);
        assert_eq!((slots[0].lambda_row, slots[0].lambda_col), (0, 0));
        assert_eq!((slots[1].lambda_row, slots[1].lambda_col), (1, 0));
        assert_eq!((slots[2].lambda_row, slots[2].lambda_col), (1, 1));
    }

    #[test]
    fn split_double_bar_maps_are_scalar_slots() {
        let formula = parse_formula("y ~ x + (1 + x || subject)").unwrap();
        let semantic = compile_formula_ir(&formula);
        assert_eq!(semantic.random_terms.len(), 2);

        let intercept_map = ThetaMap::from_random_term(0, &semantic.random_terms[0], 0);
        assert!(matches!(intercept_map, ThetaMap::Scalar(_)));
        let intercept_slots = &intercept_map.block().theta_slots;
        assert_eq!(intercept_slots.len(), 1);
        assert_eq!(
            (intercept_slots[0].lambda_row, intercept_slots[0].lambda_col),
            (0, 0)
        );

        let slope_map = ThetaMap::from_random_term(1, &semantic.random_terms[1], 1);
        assert!(matches!(slope_map, ThetaMap::Scalar(_)));
        let slope_slots = &slope_map.block().theta_slots;
        assert_eq!(slope_slots.len(), 1);
        assert_eq!(
            (slope_slots[0].lambda_row, slope_slots[0].lambda_col),
            (0, 0)
        );
    }

    #[test]
    fn expanded_categorical_basis_uses_optimizer_columns_for_full_map() {
        let formula = parse_formula("y ~ cond + (0 + cond | subject)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let optimizer_basis = vec![
            "cond: A".to_string(),
            "cond: B".to_string(),
            "cond: C".to_string(),
        ];
        let map = ThetaMap::from_random_term_with_optimizer_basis(
            0,
            &semantic.random_terms[0],
            0,
            optimizer_basis.clone(),
        );

        assert!(matches!(map, ThetaMap::FullCholesky(_)));
        assert_eq!(map.n_free(), 6);
        assert_eq!(map.block().user_basis, vec!["cond".to_string()]);
        assert_eq!(map.block().optimizer_basis, optimizer_basis);
        assert_eq!(map.block().theta_slots[0].name, "theta[0:cond: A,cond: A]");
        assert_eq!(map.block().theta_slots[5].name, "theta[0:cond: C,cond: C]");
    }

    #[test]
    fn expanded_categorical_basis_preserves_zero_correlation_diagonal_map() {
        let formula = parse_formula("y ~ cond + (0 + cond || subject)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let optimizer_basis = vec![
            "cond: A".to_string(),
            "cond: B".to_string(),
            "cond: C".to_string(),
        ];
        let map = ThetaMap::from_random_term_with_optimizer_basis(
            0,
            &semantic.random_terms[0],
            0,
            optimizer_basis.clone(),
        );

        assert!(matches!(map, ThetaMap::Diagonal(_)));
        assert_eq!(map.n_free(), 3);
        assert_eq!(map.block().user_basis, vec!["cond".to_string()]);
        assert_eq!(map.block().optimizer_basis, optimizer_basis);
        assert!(map
            .block()
            .theta_slots
            .iter()
            .all(|slot| slot.lambda_row == slot.lambda_col));
    }

    #[test]
    fn structured_maps_carry_inactive_placeholder_slots() {
        let formula = parse_formula("y ~ x + cs(1 + x | subject)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let map = ThetaMap::from_random_term(0, &semantic.random_terms[0], 0);

        assert!(matches!(map, ThetaMap::Structured(_)));
        assert_eq!(
            map.family(),
            CovarianceFamily::Structured {
                kind: "compound_symmetry".to_string()
            }
        );
        assert_eq!(
            map.block().support_status,
            CovarianceSupportStatus::ParsedRefused
        );
        assert_eq!(map.n_free(), 0);
        assert_eq!(map.block().theta_slots.len(), 2);
        assert!(
            map.block()
                .theta_slots
                .iter()
                .all(|slot| slot.global_index.is_none()
                    && slot.status == ParameterStatus::NotAssessed)
        );
        assert_eq!(
            map.block().theta_slots[1].constraint,
            ParameterConstraint::Interval {
                lower: -1.0,
                upper: 1.0
            }
        );
    }

    #[test]
    fn theta_map_round_trips_json() {
        let formula = parse_formula("y ~ x + (1 + x || subject)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let map = ThetaMap::from_random_term(0, &semantic.random_terms[0], 0);
        let json = serde_json::to_string(&map).unwrap();
        let decoded: ThetaMap = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, map);
    }
}
