use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Top-level statistical fit status used by compiler and optimizer artifacts.
///
/// `ConvergedPenalised` is for fits whose maximum-likelihood estimate does
/// **not** exist (likelihood unbounded; e.g. fixed-effect or conditional
/// separation in a logistic GLMM) but for which a well-defined penalised
/// estimate (Firth, ridge, weakly-informative prior) does exist. Calling
/// such a fit `Converged*` would be dishonest — it is not an MLE — so the
/// contract carves out a dedicated leaf status. Refusal/`NotIdentifiable`
/// remains the right answer when no penalty is applied.
///
/// The ML-non-existence reason and the penalty method belong on the
/// artifact (alongside the optimizer certificate), not on this enum, so
/// the variant stays unit-shaped and `Copy + Eq`. See
/// `docs/mixed_model_compiler_inference_contract.md` for the
/// Refusal-vs-ConvergedPenalised decision tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FitStatus {
    ConvergedInterior,
    ConvergedBoundary,
    ConvergedReducedRank,
    ConvergedPenalised,
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
    BinomialSeparation,
    NotIdentifiable,
    InvalidAgqRequest,
    PirlsFailure,
    OptimizerNotAssessed,
    OptimizerNonconvergence,
    OptimizerRecovery,
    InferenceUnavailable,
    SerializationNotAssessed,
    Unsupported,
    // Pedagogical taxonomy - append-only ordering. Do not alphabetize.
    ScopeNote,
    SupportNote,
    SyntaxExpansion,
    CovarianceAssumption,
    StructuralRefusal,
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

    #[test]
    fn pedagogical_diagnostic_codes_use_stable_snake_case_names() {
        let cases = [
            (DiagnosticCode::ScopeNote, "\"scope_note\""),
            (DiagnosticCode::SupportNote, "\"support_note\""),
            (DiagnosticCode::SyntaxExpansion, "\"syntax_expansion\""),
            (
                DiagnosticCode::CovarianceAssumption,
                "\"covariance_assumption\"",
            ),
            (DiagnosticCode::StructuralRefusal, "\"structural_refusal\""),
            (DiagnosticCode::InvalidAgqRequest, "\"invalid_agq_request\""),
            (DiagnosticCode::PirlsFailure, "\"pirls_failure\""),
            (
                DiagnosticCode::OptimizerNonconvergence,
                "\"optimizer_nonconvergence\"",
            ),
            (DiagnosticCode::OptimizerRecovery, "\"optimizer_recovery\""),
        ];

        for (code, expected) in cases {
            assert_eq!(serde_json::to_string(&code).unwrap(), expected);
        }
    }
}
