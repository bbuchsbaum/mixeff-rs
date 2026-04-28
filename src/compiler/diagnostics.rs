use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Top-level statistical fit status used by compiler and optimizer artifacts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FitStatus {
    ConvergedInterior,
    ConvergedBoundary,
    ConvergedReducedRank,
    NotIdentifiable,
    NotOptimized,
    NotAssessed,
}

/// Severity of a structured diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticSeverity {
    Info,
    Warning,
    Error,
}

/// Compiler/fitting stage that produced a diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticStage {
    FormulaParsing,
    SemanticIr,
    DesignAudit,
    Estimability,
    Parameterization,
    Optimization,
    Certification,
    Inference,
    Serialization,
    NotAssessed,
}

/// Stable diagnostic code namespace for compiler-contract artifacts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticCode {
    FormulaCanonicalized,
    FormulaCanonicalizationUnsupported,
    DuplicateRandomTerm,
    ConflictingCovariance,
    CrossingLikelyUnintended,
    FixedEffectColumnMissing,
    FixedEffectRankDeficient,
    FixedEffectEmptyCell,
    RandomSlopeWithoutIntercept,
    FixedRandomRedundant,
    RepeatedUnitUnmodeled,
    RandomSlopeUnsupported,
    RandomEffectFewLevels,
    CovarianceTooRich,
    CovarianceReduced,
    BoundaryParameter,
    NearUnitRandomEffectCorrelation,
    NotIdentifiable,
    OptimizerNotAssessed,
    InferenceUnavailable,
    SerializationNotAssessed,
    Unsupported,
}

/// Machine-readable diagnostic with optional structured payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Diagnostic {
    pub code: DiagnosticCode,
    pub severity: DiagnosticSeverity,
    pub stage: DiagnosticStage,
    pub message: String,
    pub affected_terms: Vec<String>,
    pub suggested_actions: Vec<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub payload: BTreeMap<String, serde_json::Value>,
}

impl Diagnostic {
    pub fn new(
        code: DiagnosticCode,
        severity: DiagnosticSeverity,
        stage: DiagnosticStage,
        message: impl Into<String>,
    ) -> Self {
        Self {
            code,
            severity,
            stage,
            message: message.into(),
            affected_terms: Vec::new(),
            suggested_actions: Vec::new(),
            payload: BTreeMap::new(),
        }
    }

    pub fn with_affected_terms(mut self, terms: Vec<String>) -> Self {
        self.affected_terms = terms;
        self
    }

    pub fn with_suggested_actions(mut self, actions: Vec<String>) -> Self {
        self.suggested_actions = actions;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostic_round_trips_json() {
        let mut diagnostic = Diagnostic::new(
            DiagnosticCode::CovarianceTooRich,
            DiagnosticSeverity::Warning,
            DiagnosticStage::DesignAudit,
            "covariance structure is too rich for the observed design",
        )
        .with_affected_terms(vec!["(1 + x | subject)".to_string()])
        .with_suggested_actions(vec!["use diagonal covariance".to_string()]);
        diagnostic
            .payload
            .insert("n_levels".to_string(), serde_json::json!(3));

        let json = serde_json::to_string(&diagnostic).unwrap();
        let decoded: Diagnostic = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded, diagnostic);
    }
}
