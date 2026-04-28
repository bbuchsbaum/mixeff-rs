use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::audit::InformationBudgetStatus;
use super::ir::{CovarianceForm, GroupingFactorIr, GroupingRole};

pub const RANDOM_TERM_CARD_SCHEMA: &str = "mixedmodels.random_term_card";
pub const RANDOM_TERM_CARD_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RandomTermCard {
    pub schema_name: String,
    pub schema_version: u32,
    pub term_id: String,
    pub original_fragment: String,
    pub canonical_fragment: String,
    pub group: GroupingFactorIr,
    pub blocks: Vec<RandomTermBlock>,
    pub implied_constraints: Vec<ImpliedConstraint>,
    pub design_support: DesignSupport,
    pub role_origin: RoleOrigin,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RandomTermBlock {
    pub basis: Vec<String>,
    pub intercept: bool,
    pub slopes: Vec<String>,
    pub covariance: CovarianceForm,
    pub theta_parameters: usize,
    pub english: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImpliedConstraint {
    #[serde(rename = "type")]
    pub kind: ImpliedConstraintKind,
    pub between: Vec<String>,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImpliedConstraintKind {
    ZeroCovariance,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DesignSupport {
    pub group_levels: Option<usize>,
    pub min_rows_per_group: Option<usize>,
    pub median_rows_per_group: Option<usize>,
    pub within_group_variation: BTreeMap<String, WithinGroupVariation>,
    pub status: InformationBudgetStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WithinGroupVariation {
    Present,
    Absent,
    Constant,
    NotAssessed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoleOrigin {
    pub declared_by_user: bool,
    pub observed_from_data: bool,
    pub role: GroupingRole,
}

impl RoleOrigin {
    pub fn observed(role: GroupingRole) -> Self {
        Self {
            declared_by_user: false,
            observed_from_data: true,
            role,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CrossCardConstraint {
    #[serde(rename = "type")]
    pub kind: ImpliedConstraintKind,
    pub between_cards: Vec<String>,
    pub between_basis: Vec<String>,
    pub reason: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_term_card_round_trips_json() {
        let mut within_group_variation = BTreeMap::new();
        within_group_variation.insert("Intercept".to_string(), WithinGroupVariation::Present);
        within_group_variation.insert("Days".to_string(), WithinGroupVariation::Present);

        let card = RandomTermCard {
            schema_name: RANDOM_TERM_CARD_SCHEMA.to_string(),
            schema_version: RANDOM_TERM_CARD_SCHEMA_VERSION,
            term_id: "r0".to_string(),
            original_fragment: "(Days | Subject)".to_string(),
            canonical_fragment: "(1 + Days | Subject)".to_string(),
            group: GroupingFactorIr::Single {
                name: "Subject".to_string(),
            },
            blocks: vec![RandomTermBlock {
                basis: vec!["Intercept".to_string(), "Days".to_string()],
                intercept: true,
                slopes: vec!["Days".to_string()],
                covariance: CovarianceForm::Full,
                theta_parameters: 3,
                english:
                    "`Subject` units differ in baseline and `Days` slope; the model estimates whether these are associated."
                        .to_string(),
            }],
            implied_constraints: vec![ImpliedConstraint {
                kind: ImpliedConstraintKind::ZeroCovariance,
                between: vec!["Intercept".to_string(), "Days".to_string()],
                reason:
                    "The double-bar syntax fixes the covariance between `Intercept` and `Days` to zero."
                        .to_string(),
            }],
            design_support: DesignSupport {
                group_levels: Some(18),
                min_rows_per_group: Some(10),
                median_rows_per_group: Some(10),
                within_group_variation,
                status: InformationBudgetStatus::Sufficient,
            },
            role_origin: RoleOrigin::observed(GroupingRole::SampledUnit),
        };

        let json = serde_json::to_string(&card).unwrap();
        let decoded: RandomTermCard = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, card);
    }

    #[test]
    fn cross_card_constraint_round_trips_json() {
        let constraint = CrossCardConstraint {
            kind: ImpliedConstraintKind::ZeroCovariance,
            between_cards: vec!["r0".to_string(), "r1".to_string()],
            between_basis: vec!["Intercept".to_string(), "Days".to_string()],
            reason:
                "Separate random-effect blocks fix the covariance between `Intercept` and `Days` to zero."
                    .to_string(),
        };

        let json = serde_json::to_string(&constraint).unwrap();
        let decoded: CrossCardConstraint = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, constraint);
    }
}
