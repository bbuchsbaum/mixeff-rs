use serde::{Deserialize, Serialize};

use super::artifact::ReductionTrigger;
use super::ir::{CovarianceForm, RandomTermIr};

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
        match self {
            ThetaMap::Scalar(_) => CovarianceFamily::Scalar,
            ThetaMap::Diagonal(_) => CovarianceFamily::Diagonal,
            ThetaMap::FullCholesky(_) => CovarianceFamily::FullCholesky,
            ThetaMap::Structured(_) => CovarianceFamily::Structured {
                kind: "structured".to_string(),
            },
            ThetaMap::ReducedRank(_) => CovarianceFamily::ReducedRank { rank: None },
            ThetaMap::Unsupported(_) => CovarianceFamily::Unsupported {
                reason: "unsupported theta map".to_string(),
            },
        }
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
            CovarianceForm::Structured { kind } => {
                CovarianceFamily::Structured { kind: kind.clone() }
            }
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
        CovarianceFamily::Structured { .. }
        | CovarianceFamily::ReducedRank { .. }
        | CovarianceFamily::Unsupported { .. } => Vec::new(),
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

fn make_slot(
    term_index: usize,
    local_index: usize,
    row: usize,
    col: usize,
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
        lambda_row: row,
        lambda_col: col,
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
    fn diagonal_map_has_only_diagonal_slots() {
        let formula = parse_formula("y ~ x + (1 + x || subject)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let map = ThetaMap::from_random_term(0, &semantic.random_terms[0], 0);

        assert!(matches!(map, ThetaMap::Diagonal(_)));
        let slots = &map.block().theta_slots;
        assert_eq!(slots.len(), 2);
        assert_eq!((slots[0].lambda_row, slots[0].lambda_col), (0, 0));
        assert_eq!((slots[1].lambda_row, slots[1].lambda_col), (1, 1));
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
    fn theta_map_round_trips_json() {
        let formula = parse_formula("y ~ x + (1 + x || subject)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let map = ThetaMap::from_random_term(0, &semantic.random_terms[0], 0);
        let json = serde_json::to_string(&map).unwrap();
        let decoded: ThetaMap = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, map);
    }
}
