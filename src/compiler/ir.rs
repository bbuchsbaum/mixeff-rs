use serde::{Deserialize, Serialize};

use std::collections::{BTreeMap, BTreeSet};

use crate::formula::{FixedTerm, Formula, GroupingFactor, RandomTerm, RandomTermExpansion};

use super::diagnostics::{Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticStage};

pub const SEMANTIC_MODEL_SCHEMA: &str = "mixedmodels.semantic_model";
pub const SEMANTIC_MODEL_SCHEMA_VERSION: u32 = 1;

/// Semantic model compiled from formula syntax.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticModel {
    pub schema_name: String,
    pub schema_version: u32,
    pub response: String,
    pub fixed_terms: Vec<String>,
    pub random_terms: Vec<RandomTermIr>,
    pub diagnostics: Vec<Diagnostic>,
}

/// Semantic random-effect term.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RandomTermIr {
    pub id: String,
    pub group: GroupingFactorIr,
    pub basis: Vec<RandomCoefficient>,
    pub covariance: CovarianceForm,
    pub intercept: InterceptPolicy,
    pub role: GroupingRole,
    pub source_syntax: SourceSyntax,
    pub covariance_story: CovarianceStory,
}

/// Grouping factor represented by the semantic compiler.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GroupingFactorIr {
    Single { name: String },
    Interaction { names: Vec<String> },
    Cell { names: Vec<String> },
}

impl GroupingFactorIr {
    pub fn label(&self) -> String {
        match self {
            GroupingFactorIr::Single { name } => name.clone(),
            GroupingFactorIr::Interaction { names } => names.join(":"),
            GroupingFactorIr::Cell { names } => names.join(":"),
        }
    }

    pub fn is_cell(&self) -> bool {
        matches!(self, GroupingFactorIr::Cell { .. })
    }
}

/// Random coefficient basis column.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RandomCoefficient {
    pub name: String,
    pub kind: RandomCoefficientKind,
    pub source: String,
}

/// Basis-column role inside a random-effect term.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RandomCoefficientKind {
    Intercept,
    Slope,
    Interaction,
    Unsupported,
}

/// Covariance-family requested by the semantic random-effect term.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CovarianceForm {
    Scalar,
    Diagonal,
    Full,
    Structured { kind: String },
    ReducedRank { rank: Option<usize> },
    Unsupported { reason: String },
}

/// Whether the random-coefficient basis includes an intercept.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterceptPolicy {
    Included,
    Omitted,
}

/// Declared or inferred scientific role for a grouping factor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GroupingRole {
    SampledUnit,
    Item,
    Site,
    Batch,
    Block,
    Treatment,
    RepeatedUnit,
    Unknown,
}

/// Source syntax preserved for round-tripping and diagnostics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceSyntax {
    /// Canonical display text for the term consumed by the compiler.
    pub text: String,
    /// Source text as written by the user, when it differs from canonical text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub written: Option<String>,
}

impl SourceSyntax {
    pub fn new(canonical: impl Into<String>, written: Option<String>) -> Self {
        let canonical = canonical.into();
        let written = written.filter(|text| text != &canonical);
        Self {
            text: canonical,
            written,
        }
    }

    pub fn user_text(&self) -> &str {
        self.written.as_deref().unwrap_or(&self.text)
    }
}

/// User-facing covariance interpretation generated from semantic IR.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CovarianceStory {
    pub summary: String,
    pub assumptions: Vec<String>,
    pub dependence: Vec<String>,
}

pub fn compile_formula_ir(formula: &Formula) -> SemanticModel {
    SemanticModel::from_formula(formula)
}

impl SemanticModel {
    pub fn from_formula(formula: &Formula) -> Self {
        let mut diagnostics = Vec::new();
        let random_terms = formula
            .random_terms
            .iter()
            .enumerate()
            .map(|(idx, term)| RandomTermIr::from_formula_term(idx, term, &mut diagnostics))
            .collect::<Vec<_>>();

        emit_formula_canonicalization_diagnostics(
            &formula.random_terms,
            &random_terms,
            &mut diagnostics,
        );
        emit_duplicate_and_conflicting_covariance_diagnostics(
            &formula.random_terms,
            &mut diagnostics,
        );

        Self {
            schema_name: SEMANTIC_MODEL_SCHEMA.to_string(),
            schema_version: SEMANTIC_MODEL_SCHEMA_VERSION,
            response: formula.response.clone(),
            fixed_terms: formula
                .fixed_terms
                .iter()
                .map(ToString::to_string)
                .collect(),
            random_terms,
            diagnostics,
        }
    }
}

impl RandomTermIr {
    fn from_formula_term(
        index: usize,
        term: &RandomTerm,
        diagnostics: &mut Vec<Diagnostic>,
    ) -> Self {
        let source = term.to_string();
        let written = term.source.as_ref().map(|source| source.written.clone());
        let group = grouping_ir(&term.grouping);
        let intercept = intercept_policy(term);
        let mut basis = Vec::new();

        if intercept == InterceptPolicy::Included {
            basis.push(RandomCoefficient {
                name: "intercept".to_string(),
                kind: RandomCoefficientKind::Intercept,
                source: "1".to_string(),
            });
        }

        for fixed in &term.terms {
            match fixed {
                FixedTerm::Intercept | FixedTerm::NoIntercept => {}
                FixedTerm::Column(name) => basis.push(RandomCoefficient {
                    name: name.clone(),
                    kind: RandomCoefficientKind::Slope,
                    source: name.clone(),
                }),
                FixedTerm::Interaction(names) => basis.push(RandomCoefficient {
                    name: names.join(":"),
                    kind: RandomCoefficientKind::Interaction,
                    source: names.join(":"),
                }),
                FixedTerm::Nested(names) => {
                    let label = names.join("/");
                    diagnostics.push(
                        Diagnostic::new(
                            DiagnosticCode::FormulaCanonicalizationUnsupported,
                            DiagnosticSeverity::Warning,
                            DiagnosticStage::SemanticIr,
                            "nested random coefficient bases are not canonicalized in compiler v0",
                        )
                        .with_affected_terms(vec![source.clone()]),
                    );
                    basis.push(RandomCoefficient {
                        name: label.clone(),
                        kind: RandomCoefficientKind::Unsupported,
                        source: label,
                    });
                }
            }
        }

        if basis.is_empty() {
            diagnostics.push(
                Diagnostic::new(
                    DiagnosticCode::NotIdentifiable,
                    DiagnosticSeverity::Error,
                    DiagnosticStage::SemanticIr,
                    "random-effect term has an empty compiled basis",
                )
                .with_affected_terms(vec![source.clone()]),
            );
        }

        if intercept == InterceptPolicy::Omitted
            && basis
                .iter()
                .any(|b| b.kind != RandomCoefficientKind::Intercept)
        {
            diagnostics.push(
                Diagnostic::new(
                    DiagnosticCode::RandomSlopeWithoutIntercept,
                    DiagnosticSeverity::Info,
                    DiagnosticStage::SemanticIr,
                    "random slope term omits a random intercept; this leaves baseline grouping dependence unmodeled unless represented elsewhere",
                )
                .with_affected_terms(vec![source.clone()])
                .with_suggested_actions(vec![
                    "consider adding a random intercept if baseline grouping dependence is expected".to_string(),
                ]),
            );
        }

        let covariance = covariance_form(term.zerocorr, basis.len());
        let story = covariance_story(&group, &basis, &covariance, intercept);

        Self {
            id: format!("r{index}"),
            group,
            basis,
            covariance,
            intercept,
            role: GroupingRole::Unknown,
            source_syntax: SourceSyntax::new(source, written),
            covariance_story: story,
        }
    }
}

fn emit_formula_canonicalization_diagnostics(
    formula_terms: &[RandomTerm],
    ir_terms: &[RandomTermIr],
    diagnostics: &mut Vec<Diagnostic>,
) {
    let mut by_source: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut expansions: BTreeMap<String, RandomTermExpansion> = BTreeMap::new();

    for (formula_term, ir_term) in formula_terms.iter().zip(ir_terms.iter()) {
        let Some(source) = &formula_term.source else {
            continue;
        };

        if !canonical_text_equivalent(&source.written, &ir_term.source_syntax.text)
            || source.expansion.is_some()
        {
            by_source
                .entry(source.written.clone())
                .or_default()
                .push(ir_term.source_syntax.text.clone());
        }

        if let Some(expansion) = source.expansion {
            expansions
                .entry(source.written.clone())
                .or_insert(expansion);
        }
    }

    for (written, canonical_terms) in by_source {
        let canonical = canonical_terms.join(" + ");
        let mut diagnostic = Diagnostic::new(
            DiagnosticCode::FormulaCanonicalized,
            DiagnosticSeverity::Info,
            DiagnosticStage::SemanticIr,
            format!("random-effect term was canonicalized as {canonical}"),
        )
        .with_affected_terms(vec![written.clone()]);
        diagnostic.payload.insert(
            "canonical_terms".to_string(),
            serde_json::json!(canonical_terms),
        );
        diagnostics.push(diagnostic);
    }

    for (written, expansion) in expansions {
        if expansion == RandomTermExpansion::CrossedGrouping {
            diagnostics.push(
                Diagnostic::new(
                    DiagnosticCode::CrossingLikelyUnintended,
                    DiagnosticSeverity::Info,
                    DiagnosticStage::SemanticIr,
                    "crossed grouping shorthand expands to main grouping effects plus a cell effect; confirm this is intended",
                )
                .with_affected_terms(vec![written])
                .with_suggested_actions(vec![
                    "use separate terms like (1 | a) + (1 | b) for crossed main effects only".to_string(),
                    "use a cell term like (1 | a:b) if only same-cell dependence is intended".to_string(),
                ]),
            );
        }
    }
}

fn canonical_text_equivalent(lhs: &str, rhs: &str) -> bool {
    lhs.split_whitespace().collect::<String>() == rhs.split_whitespace().collect::<String>()
}

fn emit_duplicate_and_conflicting_covariance_diagnostics(
    terms: &[RandomTerm],
    diagnostics: &mut Vec<Diagnostic>,
) {
    let mut exact_terms: BTreeMap<String, String> = BTreeMap::new();
    let mut covariance_terms: BTreeMap<String, (bool, String)> = BTreeMap::new();

    for term in terms {
        let source = term_user_source(term);
        let exact_key = random_term_identity_key(term, true);
        if let Some(first_source) = exact_terms.get(&exact_key) {
            diagnostics.push(
                Diagnostic::new(
                    DiagnosticCode::DuplicateRandomTerm,
                    DiagnosticSeverity::Warning,
                    DiagnosticStage::SemanticIr,
                    "duplicate random-effect term requests the same grouping, basis, and covariance structure twice",
                )
                .with_affected_terms(vec![first_source.clone(), source.clone()])
                .with_suggested_actions(vec![
                    "remove the duplicate random-effect term".to_string(),
                ]),
            );
        } else {
            exact_terms.insert(exact_key, source.clone());
        }

        let covariance_key = random_term_identity_key(term, false);
        if let Some((first_zerocorr, first_source)) = covariance_terms.get(&covariance_key) {
            if *first_zerocorr != term.zerocorr {
                diagnostics.push(
                    Diagnostic::new(
                        DiagnosticCode::ConflictingCovariance,
                        DiagnosticSeverity::Error,
                        DiagnosticStage::SemanticIr,
                        "same random-effect basis is requested with both correlated and zero-correlation covariance",
                    )
                    .with_affected_terms(vec![first_source.clone(), source.clone()])
                    .with_suggested_actions(vec![
                        "choose either correlated | syntax or zero-correlation || syntax for this grouping and basis".to_string(),
                    ]),
                );
            }
        } else {
            covariance_terms.insert(covariance_key, (term.zerocorr, source));
        }
    }
}

fn random_term_identity_key(term: &RandomTerm, include_covariance: bool) -> String {
    let mut basis = BTreeSet::new();
    let mut intercept = intercept_policy(term);
    for fixed in &term.terms {
        match fixed {
            FixedTerm::Intercept => {
                intercept = InterceptPolicy::Included;
            }
            FixedTerm::NoIntercept => {
                intercept = InterceptPolicy::Omitted;
            }
            FixedTerm::Column(name) => {
                basis.insert(format!("column:{name}"));
            }
            FixedTerm::Interaction(names) => {
                basis.insert(format!("interaction:{}", names.join(":")));
            }
            FixedTerm::Nested(names) => {
                basis.insert(format!("nested:{}", names.join("/")));
            }
        }
    }

    let covariance = if include_covariance {
        format!(";zerocorr={}", term.zerocorr)
    } else {
        String::new()
    };

    format!(
        "group={};intercept={:?};basis={}{covariance}",
        grouping_key(&term.grouping),
        intercept,
        basis.into_iter().collect::<Vec<_>>().join(",")
    )
}

fn grouping_key(grouping: &GroupingFactor) -> String {
    match grouping {
        GroupingFactor::Single(name) => format!("single:{name}"),
        GroupingFactor::Interaction(names) => format!("interaction:{}", names.join("&")),
        GroupingFactor::Cell(names) => format!("cell:{}", names.join(":")),
    }
}

fn term_user_source(term: &RandomTerm) -> String {
    term.source
        .as_ref()
        .map(|source| source.written.clone())
        .unwrap_or_else(|| term.to_string())
}

fn grouping_ir(grouping: &GroupingFactor) -> GroupingFactorIr {
    match grouping {
        GroupingFactor::Single(name) => GroupingFactorIr::Single { name: name.clone() },
        GroupingFactor::Interaction(names) => GroupingFactorIr::Interaction {
            names: names.clone(),
        },
        GroupingFactor::Cell(names) => GroupingFactorIr::Cell {
            names: names.clone(),
        },
    }
}

fn intercept_policy(term: &RandomTerm) -> InterceptPolicy {
    if term
        .terms
        .iter()
        .any(|t| matches!(t, FixedTerm::NoIntercept))
    {
        InterceptPolicy::Omitted
    } else if term.terms.iter().any(|t| matches!(t, FixedTerm::Intercept)) {
        InterceptPolicy::Included
    } else {
        InterceptPolicy::Omitted
    }
}

fn covariance_form(zerocorr: bool, basis_len: usize) -> CovarianceForm {
    match (zerocorr, basis_len) {
        (_, 0) => CovarianceForm::Unsupported {
            reason: "empty basis".to_string(),
        },
        (true, _) => CovarianceForm::Diagonal,
        (_, 1) => CovarianceForm::Scalar,
        (false, _) => CovarianceForm::Full,
    }
}

fn covariance_story(
    group: &GroupingFactorIr,
    basis: &[RandomCoefficient],
    covariance: &CovarianceForm,
    intercept: InterceptPolicy,
) -> CovarianceStory {
    let group_label = group.label();
    let basis_names = basis
        .iter()
        .map(|b| b.name.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    let summary = format!("{group_label}: varying coefficients [{basis_names}]");

    let mut assumptions = Vec::new();
    if intercept == InterceptPolicy::Included {
        assumptions.push(format!(
            "{group_label} levels may differ in baseline response"
        ));
    } else {
        assumptions.push(format!(
            "{group_label} levels are not given a shared baseline offset by this term"
        ));
    }

    match covariance {
        CovarianceForm::Scalar => {
            assumptions.push("single variance direction is estimated".to_string());
        }
        CovarianceForm::Diagonal => {
            assumptions
                .push("random coefficients vary independently in the compiled basis".to_string());
        }
        CovarianceForm::Full => {
            assumptions.push("random coefficients may covary in the compiled basis".to_string());
        }
        CovarianceForm::Structured { kind } => {
            assumptions.push(format!("structured covariance family: {kind}"));
        }
        CovarianceForm::ReducedRank { rank } => {
            assumptions.push(format!(
                "reduced-rank covariance requested with rank {rank:?}"
            ));
        }
        CovarianceForm::Unsupported { reason } => {
            assumptions.push(format!("covariance unsupported: {reason}"));
        }
    }

    let dependence = if group.is_cell() {
        vec![format!(
            "only observations sharing the same {group_label} cell are correlated through this term"
        )]
    } else {
        vec![format!(
            "observations sharing {group_label} are correlated through this term's random coefficients"
        )]
    };

    CovarianceStory {
        summary,
        assumptions,
        dependence,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formula::parse_formula;

    #[test]
    fn compiles_full_intercept_slope_term() {
        let formula = parse_formula("y ~ x + (1 + x | subject)").unwrap();
        let model = compile_formula_ir(&formula);

        let term = &model.random_terms[0];
        assert_eq!(term.group.label(), "subject");
        assert_eq!(term.intercept, InterceptPolicy::Included);
        assert_eq!(term.covariance, CovarianceForm::Full);
        assert_eq!(term.basis.len(), 2);
        assert_eq!(term.basis[0].kind, RandomCoefficientKind::Intercept);
        assert_eq!(term.basis[1].name, "x");
    }

    #[test]
    fn compiles_zero_correlation_as_diagonal() {
        let formula = parse_formula("y ~ x + (1 + x || subject)").unwrap();
        let model = compile_formula_ir(&formula);
        assert_eq!(model.random_terms[0].covariance, CovarianceForm::Diagonal);
    }

    #[test]
    fn records_slope_only_intercept_omission() {
        let formula = parse_formula("y ~ x + (0 + x | subject)").unwrap();
        let model = compile_formula_ir(&formula);
        let term = &model.random_terms[0];

        assert_eq!(term.intercept, InterceptPolicy::Omitted);
        assert_eq!(term.covariance, CovarianceForm::Scalar);
        assert!(model
            .diagnostics
            .iter()
            .any(|d| d.code == DiagnosticCode::RandomSlopeWithoutIntercept));
    }

    #[test]
    fn semantic_model_round_trips_json() {
        let formula = parse_formula("y ~ x + (1 + x || subject)").unwrap();
        let model = compile_formula_ir(&formula);
        let json = serde_json::to_string(&model).unwrap();
        let decoded: SemanticModel = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, model);
    }

    #[test]
    fn cell_grouping_has_cell_dependence_story() {
        let formula = parse_formula("y ~ x + (1 | subject:item)").unwrap();
        let model = compile_formula_ir(&formula);
        let term = &model.random_terms[0];

        assert_eq!(term.group.label(), "subject:item");
        assert!(matches!(term.group, GroupingFactorIr::Cell { .. }));
        assert!(term
            .covariance_story
            .dependence
            .iter()
            .any(|line| line.contains("same subject:item cell")));
    }

    #[test]
    fn crossed_star_grouping_compiles_expanded_terms() {
        let formula = parse_formula("y ~ x + (1 | subject*item)").unwrap();
        let model = compile_formula_ir(&formula);
        let labels: Vec<_> = model.random_terms.iter().map(|t| t.group.label()).collect();

        assert_eq!(labels, vec!["subject", "item", "subject:item"]);
    }

    #[test]
    fn preserves_written_source_syntax_when_parser_canonicalizes() {
        let formula = parse_formula("y ~ x + (x | subject)").unwrap();
        let model = compile_formula_ir(&formula);
        let source = &model.random_terms[0].source_syntax;

        assert_eq!(source.text, "(1 + x | subject)");
        assert_eq!(source.written.as_deref(), Some("(x | subject)"));
        assert_eq!(source.user_text(), "(x | subject)");
        assert!(model
            .diagnostics
            .iter()
            .any(|d| d.code == DiagnosticCode::FormulaCanonicalized));
    }

    #[test]
    fn emits_crossed_grouping_canonicalization_diagnostics() {
        let formula = parse_formula("y ~ x + (1 | subject*item)").unwrap();
        let model = compile_formula_ir(&formula);

        assert!(model
            .diagnostics
            .iter()
            .any(|d| d.code == DiagnosticCode::FormulaCanonicalized
                && d.affected_terms == vec!["(1 | subject*item)"]));
        assert!(model
            .diagnostics
            .iter()
            .any(|d| d.code == DiagnosticCode::CrossingLikelyUnintended
                && d.affected_terms == vec!["(1 | subject*item)"]));
    }

    #[test]
    fn emits_duplicate_random_term_diagnostic() {
        let formula = parse_formula("y ~ x + (1 + x | subject) + (x + 1 | subject)").unwrap();
        let model = compile_formula_ir(&formula);

        assert!(model
            .diagnostics
            .iter()
            .any(|d| d.code == DiagnosticCode::DuplicateRandomTerm
                && d.severity == DiagnosticSeverity::Warning));
    }

    #[test]
    fn emits_conflicting_covariance_diagnostic() {
        let formula = parse_formula("y ~ x + (1 + x | subject) + (1 + x || subject)").unwrap();
        let model = compile_formula_ir(&formula);

        assert!(model
            .diagnostics
            .iter()
            .any(|d| d.code == DiagnosticCode::ConflictingCovariance
                && d.severity == DiagnosticSeverity::Error));
    }
}
