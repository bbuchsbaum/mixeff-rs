use serde::{Deserialize, Serialize};

use std::collections::{BTreeMap, BTreeSet};

use crate::formula::{
    FixedTerm, Formula, GroupingFactor, RandomCovariance, RandomTerm, RandomTermExpansion,
};

use super::diagnostics::{Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticStage};
use super::random_term_card::RoleOrigin;

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
    #[serde(default)]
    pub role_origins: BTreeMap<String, RoleOrigin>,
    pub diagnostics: Vec<Diagnostic>,
}

/// Semantic random-effect term.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RandomTermIr {
    pub id: String,
    pub group: GroupingFactorIr,
    pub basis: Vec<RandomCoefficient>,
    pub covariance: CovarianceForm,
    pub covariance_support: CovarianceSupportStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_group: Option<String>,
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
    Structured { kind: StructuredCovarianceKind },
    ReducedRank { rank: Option<usize> },
    Unsupported { reason: String },
}

impl CovarianceForm {
    pub fn support_status(&self) -> CovarianceSupportStatus {
        match self {
            CovarianceForm::Scalar | CovarianceForm::Diagonal | CovarianceForm::Full => {
                CovarianceSupportStatus::Supported
            }
            CovarianceForm::Structured { .. } => CovarianceSupportStatus::ParsedRefused,
            CovarianceForm::ReducedRank { .. } => CovarianceSupportStatus::Future,
            CovarianceForm::Unsupported { .. } => CovarianceSupportStatus::Unsupported,
        }
    }
}

/// v1 support status for a random-effect covariance family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CovarianceSupportStatus {
    /// The family is fitted by the current engine.
    Supported,
    /// The syntax compiles into typed IR but fitting refuses before optimization.
    ParsedRefused,
    /// The artifact vocabulary is reserved for future fitted support.
    Future,
    /// The request is outside the compiler/fitting contract.
    Unsupported,
}

/// Structured random-effect covariance families parsed for the v1.0
/// covariance-pluggability contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuredCovarianceKind {
    CompoundSymmetry,
    Ar1,
}

impl StructuredCovarianceKind {
    pub fn label(self) -> &'static str {
        match self {
            StructuredCovarianceKind::CompoundSymmetry => "compound_symmetry",
            StructuredCovarianceKind::Ar1 => "ar1",
        }
    }
}

impl std::fmt::Display for StructuredCovarianceKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
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
        let mut random_terms = Vec::new();
        for (source_index, term) in formula.random_terms.iter().enumerate() {
            let compiled_terms =
                RandomTermIr::from_formula_term(source_index, term, &mut diagnostics);
            for mut compiled in compiled_terms {
                compiled.id = format!("r{}", random_terms.len());
                random_terms.push(compiled);
            }
        }

        emit_formula_canonicalization_diagnostics(
            &formula.random_terms,
            &random_terms,
            &mut diagnostics,
        );
        emit_duplicate_and_conflicting_covariance_diagnostics(
            &formula.random_terms,
            &mut diagnostics,
        );
        emit_covariance_assumption_diagnostics(&random_terms, &mut diagnostics);
        let role_origins = random_terms
            .iter()
            .map(|term| (term.id.clone(), RoleOrigin::observed(term.role)))
            .collect();

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
            role_origins,
            diagnostics,
        }
    }
}

impl RandomTermIr {
    fn from_formula_term(
        source_index: usize,
        term: &RandomTerm,
        diagnostics: &mut Vec<Diagnostic>,
    ) -> Vec<Self> {
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

        if term.zerocorr && basis.len() > 1 {
            let block_group = Some(format!("bg{source_index}"));
            return basis
                .into_iter()
                .map(|coefficient| {
                    let split_intercept = if coefficient.kind == RandomCoefficientKind::Intercept {
                        InterceptPolicy::Included
                    } else {
                        InterceptPolicy::Omitted
                    };
                    let split_basis = vec![coefficient];
                    let covariance = CovarianceForm::Scalar;
                    let covariance_support = covariance.support_status();
                    let split_source = random_term_text(&split_basis, &group, split_intercept);
                    let story =
                        covariance_story(&group, &split_basis, &covariance, split_intercept);
                    Self {
                        id: String::new(),
                        group: group.clone(),
                        basis: split_basis,
                        covariance,
                        covariance_support,
                        block_group: block_group.clone(),
                        intercept: split_intercept,
                        role: GroupingRole::Unknown,
                        source_syntax: SourceSyntax::new(split_source, written.clone()),
                        covariance_story: story,
                    }
                })
                .collect();
        }

        let covariance = covariance_form(term.covariance, basis.len());
        let covariance_support = covariance.support_status();
        let story = covariance_story(&group, &basis, &covariance, intercept);

        vec![Self {
            id: String::new(),
            group,
            basis,
            covariance,
            covariance_support,
            block_group: None,
            intercept,
            role: GroupingRole::Unknown,
            source_syntax: SourceSyntax::new(source, written),
            covariance_story: story,
        }]
    }
}

fn emit_formula_canonicalization_diagnostics(
    formula_terms: &[RandomTerm],
    ir_terms: &[RandomTermIr],
    diagnostics: &mut Vec<Diagnostic>,
) {
    let mut by_source: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut expansions: BTreeMap<String, RandomTermExpansion> = BTreeMap::new();

    for ir_term in ir_terms {
        let Some(written) = &ir_term.source_syntax.written else {
            continue;
        };
        by_source
            .entry(written.clone())
            .or_default()
            .push(ir_term.source_syntax.text.clone());
    }

    for formula_term in formula_terms {
        let Some(source) = &formula_term.source else {
            continue;
        };

        if let Some(expansion) = source.expansion {
            expansions
                .entry(source.written.clone())
                .or_insert(expansion);
        }
    }

    let canonical_by_source = by_source.clone();

    for (written, canonical_terms) in by_source {
        if canonical_terms.len() == 1 && canonical_text_equivalent(&written, &canonical_terms[0]) {
            continue;
        }
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
        if let Some(canonical_terms) = canonical_by_source.get(&written) {
            let canonical = canonical_terms.join(" + ");
            let mut diagnostic = Diagnostic::new(
                DiagnosticCode::SyntaxExpansion,
                DiagnosticSeverity::Info,
                DiagnosticStage::SemanticIr,
                format!("random-effect shorthand expands to {canonical}"),
            )
            .with_affected_terms(vec![written.clone()])
            .with_suggested_actions(vec![format!(
                "`{written}` expands to `{canonical}` - the canonical form."
            )]);
            diagnostic
                .payload
                .insert("written".to_string(), serde_json::json!(written.clone()));
            diagnostic
                .payload
                .insert("canonical".to_string(), serde_json::json!(canonical));
            diagnostic.payload.insert(
                "expansion_kind".to_string(),
                serde_json::json!(expansion_kind_label(expansion)),
            );
            diagnostics.push(diagnostic);
        }

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

fn expansion_kind_label(expansion: RandomTermExpansion) -> &'static str {
    match expansion {
        RandomTermExpansion::NestedGrouping => "nested",
        RandomTermExpansion::CrossedGrouping => "crossed_with_cell",
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

fn emit_covariance_assumption_diagnostics(
    ir_terms: &[RandomTermIr],
    diagnostics: &mut Vec<Diagnostic>,
) {
    let mut by_block_group: BTreeMap<&str, Vec<&RandomTermIr>> = BTreeMap::new();
    for term in ir_terms {
        if let Some(block_group) = &term.block_group {
            by_block_group
                .entry(block_group.as_str())
                .or_default()
                .push(term);
        }
    }

    for terms in by_block_group.values() {
        for left_index in 0..terms.len() {
            for right_index in (left_index + 1)..terms.len() {
                let left = terms[left_index];
                let right = terms[right_index];
                let Some(left_basis) = left.basis.first() else {
                    continue;
                };
                let Some(right_basis) = right.basis.first() else {
                    continue;
                };
                diagnostics.push(covariance_assumption_diagnostic(
                    left.group.label(),
                    [
                        basis_display_name(left_basis),
                        basis_display_name(right_basis),
                    ],
                    "double_bar_syntax",
                    vec![left.source_syntax.user_text().to_string()],
                ));
            }
        }
    }

    for left_index in 0..ir_terms.len() {
        for right_index in (left_index + 1)..ir_terms.len() {
            let left_term = &ir_terms[left_index];
            let right_term = &ir_terms[right_index];
            if left_term.group != right_term.group {
                continue;
            }
            if left_term.block_group.is_some() && left_term.block_group == right_term.block_group {
                continue;
            }

            for left_basis in &left_term.basis {
                for right_basis in &right_term.basis {
                    let left = basis_display_name(left_basis);
                    let right = basis_display_name(right_basis);
                    if left == right {
                        continue;
                    }
                    diagnostics.push(covariance_assumption_diagnostic(
                        left_term.group.label(),
                        [left, right],
                        "separate_random_effect_blocks",
                        vec![
                            left_term.source_syntax.user_text().to_string(),
                            right_term.source_syntax.user_text().to_string(),
                        ],
                    ));
                }
            }
        }
    }
}

fn basis_display_name(basis: &RandomCoefficient) -> String {
    if basis.kind == RandomCoefficientKind::Intercept {
        "Intercept".to_string()
    } else {
        basis.name.clone()
    }
}

fn covariance_assumption_diagnostic(
    group: String,
    between: [String; 2],
    reason: &'static str,
    affected_terms: Vec<String>,
) -> Diagnostic {
    let message = match reason {
        "double_bar_syntax" => format!(
            "the covariance between '{}' and '{}' is fixed at zero by || syntax",
            between[0], between[1]
        ),
        "separate_random_effect_blocks" => format!(
            "the covariance between '{}' and '{}' is fixed at zero by separate random-effect blocks",
            between[0], between[1]
        ),
        _ => "a covariance is fixed at zero by formula syntax".to_string(),
    };
    let mut diagnostic = Diagnostic::new(
        DiagnosticCode::CovarianceAssumption,
        DiagnosticSeverity::Info,
        DiagnosticStage::SemanticIr,
        message.clone(),
    )
    .with_affected_terms(affected_terms)
    .with_suggested_actions(vec![message]);
    diagnostic
        .payload
        .insert("group".to_string(), serde_json::json!(group));
    diagnostic
        .payload
        .insert("between".to_string(), serde_json::json!(between));
    diagnostic
        .payload
        .insert("reason".to_string(), serde_json::json!(reason));
    diagnostic
}

fn random_term_text(
    basis: &[RandomCoefficient],
    group: &GroupingFactorIr,
    intercept: InterceptPolicy,
) -> String {
    let lhs = if basis.len() == 1 && basis[0].kind == RandomCoefficientKind::Intercept {
        "1".to_string()
    } else {
        let mut parts = Vec::new();
        if intercept == InterceptPolicy::Omitted {
            parts.push("0".to_string());
        }
        parts.extend(basis.iter().map(|basis| basis.source.clone()));
        parts.join(" + ")
    };
    format!("({lhs} | {})", group.label())
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

fn covariance_form(covariance: RandomCovariance, basis_len: usize) -> CovarianceForm {
    match (covariance, basis_len) {
        (_, 0) => CovarianceForm::Unsupported {
            reason: "empty basis".to_string(),
        },
        (RandomCovariance::Full, 1) => CovarianceForm::Scalar,
        (RandomCovariance::Full, _) => CovarianceForm::Full,
        (RandomCovariance::Diagonal, _) => CovarianceForm::Diagonal,
        (RandomCovariance::CompoundSymmetry, _) => CovarianceForm::Structured {
            kind: StructuredCovarianceKind::CompoundSymmetry,
        },
        (RandomCovariance::Ar1, _) => CovarianceForm::Structured {
            kind: StructuredCovarianceKind::Ar1,
        },
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
        assert_eq!(term.covariance_support, CovarianceSupportStatus::Supported);
        assert_eq!(term.basis.len(), 2);
        assert_eq!(term.basis[0].kind, RandomCoefficientKind::Intercept);
        assert_eq!(term.basis[1].name, "x");
    }

    #[test]
    fn compiles_zero_correlation_as_scalar_block_group() {
        let formula = parse_formula("y ~ x + (1 + x || subject)").unwrap();
        let model = compile_formula_ir(&formula);
        assert_eq!(model.random_terms.len(), 2);
        assert_eq!(model.random_terms[0].covariance, CovarianceForm::Scalar);
        assert_eq!(model.random_terms[1].covariance, CovarianceForm::Scalar);
        assert_eq!(model.random_terms[0].block_group.as_deref(), Some("bg0"));
        assert_eq!(model.random_terms[1].block_group.as_deref(), Some("bg0"));
        assert_eq!(model.random_terms[0].source_syntax.text, "(1 | subject)");
        assert_eq!(
            model.random_terms[1].source_syntax.text,
            "(0 + x | subject)"
        );
        assert_eq!(
            model.random_terms[0].source_syntax.written.as_deref(),
            Some("(1 + x || subject)")
        );
    }

    #[test]
    fn compiles_diag_wrapper_as_single_diagonal_block() {
        let formula = parse_formula("y ~ x + diag(1 + x | subject)").unwrap();
        let model = compile_formula_ir(&formula);
        assert_eq!(model.random_terms.len(), 1);
        let term = &model.random_terms[0];
        assert_eq!(term.covariance, CovarianceForm::Diagonal);
        assert_eq!(term.source_syntax.text, "diag(1 + x | subject)");
        assert_eq!(term.block_group, None);
    }

    #[test]
    fn compiles_structured_wrappers_as_typed_covariance_families() {
        let cases = [
            (
                "y ~ x + cs(1 + x | subject)",
                StructuredCovarianceKind::CompoundSymmetry,
            ),
            (
                "y ~ x + ar1(0 + x | subject)",
                StructuredCovarianceKind::Ar1,
            ),
        ];

        for (source, kind) in cases {
            let formula = parse_formula(source).unwrap();
            let model = compile_formula_ir(&formula);
            assert_eq!(model.random_terms.len(), 1);
            assert_eq!(
                model.random_terms[0].covariance,
                CovarianceForm::Structured { kind }
            );
            assert_eq!(
                model.random_terms[0].covariance_support,
                CovarianceSupportStatus::ParsedRefused
            );
        }
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
        assert_eq!(model.role_origins.len(), 2);
        assert_eq!(
            model.role_origins.get("r0"),
            Some(&RoleOrigin::observed(GroupingRole::Unknown))
        );
        assert_eq!(
            model.role_origins.get("r1"),
            Some(&RoleOrigin::observed(GroupingRole::Unknown))
        );
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
    fn emits_syntax_expansion_diagnostic_for_nested_grouping() {
        let formula = parse_formula("y ~ x + (1 | school/class)").unwrap();
        let model = compile_formula_ir(&formula);

        let diagnostic = model
            .diagnostics
            .iter()
            .find(|d| d.code == DiagnosticCode::SyntaxExpansion)
            .expect("nested grouping expansion should be diagnosed");
        assert_eq!(diagnostic.severity, DiagnosticSeverity::Info);
        assert_eq!(diagnostic.stage, DiagnosticStage::SemanticIr);
        assert_eq!(
            diagnostic.payload.get("written"),
            Some(&serde_json::json!("(1 | school/class)"))
        );
        assert_eq!(
            diagnostic.payload.get("canonical"),
            Some(&serde_json::json!("(1 | school) + (1 | school:class)"))
        );
        assert_eq!(
            diagnostic.payload.get("expansion_kind"),
            Some(&serde_json::json!("nested"))
        );
    }

    #[test]
    fn emits_syntax_expansion_diagnostic_for_crossed_grouping() {
        let formula = parse_formula("y ~ x + (1 | subject*item)").unwrap();
        let model = compile_formula_ir(&formula);

        let diagnostic = model
            .diagnostics
            .iter()
            .find(|d| d.code == DiagnosticCode::SyntaxExpansion)
            .expect("crossed grouping expansion should be diagnosed");
        assert_eq!(
            diagnostic.payload.get("canonical"),
            Some(&serde_json::json!(
                "(1 | subject) + (1 | item) + (1 | subject:item)"
            ))
        );
        assert_eq!(
            diagnostic.payload.get("expansion_kind"),
            Some(&serde_json::json!("crossed_with_cell"))
        );
    }

    #[test]
    fn emits_covariance_assumption_for_double_bar_syntax() {
        let formula = parse_formula("y ~ x + (1 + x || subject)").unwrap();
        let model = compile_formula_ir(&formula);

        let diagnostic = model
            .diagnostics
            .iter()
            .find(|d| d.code == DiagnosticCode::CovarianceAssumption)
            .expect("double-bar syntax should be diagnosed");
        assert_eq!(diagnostic.severity, DiagnosticSeverity::Info);
        assert_eq!(diagnostic.stage, DiagnosticStage::SemanticIr);
        assert_eq!(
            diagnostic.payload.get("group"),
            Some(&serde_json::json!("subject"))
        );
        assert_eq!(
            diagnostic.payload.get("between"),
            Some(&serde_json::json!(["Intercept", "x"]))
        );
        assert_eq!(
            diagnostic.payload.get("reason"),
            Some(&serde_json::json!("double_bar_syntax"))
        );
    }

    #[test]
    fn emits_covariance_assumption_for_split_random_effect_blocks() {
        let formula = parse_formula("y ~ x + (1 | subject) + (0 + x | subject)").unwrap();
        let model = compile_formula_ir(&formula);

        let diagnostic = model
            .diagnostics
            .iter()
            .find(|d| {
                d.code == DiagnosticCode::CovarianceAssumption
                    && d.payload.get("reason")
                        == Some(&serde_json::json!("separate_random_effect_blocks"))
            })
            .expect("split random-effect blocks should be diagnosed");
        assert_eq!(
            diagnostic.affected_terms,
            vec!["(1 | subject)".to_string(), "(0 + x | subject)".to_string()]
        );
        assert_eq!(
            diagnostic.payload.get("between"),
            Some(&serde_json::json!(["Intercept", "x"]))
        );
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
