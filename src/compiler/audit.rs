use serde::{Deserialize, Serialize};

use nalgebra::{DMatrix, DVector, SymmetricEigen};

use crate::linalg::pivot::{pivoted_qr_with_tol, stats_rank_with_tol};
use crate::model::data::{CategoricalCoding, Column, ContrastSource, DataFrame};
use crate::types::opt_summary::optimizer_final_status_code;
use crate::types::OptSummary;

use super::diagnostics::{
    Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticStage, FitStatus,
};
use super::ir::{
    CovarianceForm, CovarianceSupportStatus, GroupingFactorIr, InterceptPolicy, RandomCoefficient,
    RandomCoefficientKind, RandomTermIr, SemanticModel,
};

pub const DESIGN_AUDIT_SCHEMA: &str = "mixedmodels.design_audit";
pub const DESIGN_AUDIT_SCHEMA_VERSION: u32 = 1;

/// Rank diagnostic for a design matrix.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RankAssessment {
    pub rank: Option<usize>,
    pub expected: Option<usize>,
    pub status: RankStatus,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RankStatus {
    FullRank,
    RankDeficient,
    NotAssessed,
}

/// Fixed-effect design audit. This is the compiler-level view of X: it records
/// which columns were constructed, whether the matrix is rank deficient, and
/// which factor combinations are absent before any optimizer is run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FixedEffectAudit {
    pub n_rows: usize,
    pub n_columns: usize,
    pub rank: RankAssessment,
    pub columns: Vec<FixedEffectColumnAudit>,
    pub terms: Vec<FixedEffectTermAudit>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub contrast_bases: Vec<CategoricalContrastAudit>,
    pub aliased_columns: Vec<String>,
    pub empty_cells: Vec<EmptyCellAudit>,
    pub diagnostics: Vec<Diagnostic>,
}

impl FixedEffectAudit {
    pub fn not_assessed() -> Self {
        Self {
            n_rows: 0,
            n_columns: 0,
            rank: RankAssessment {
                rank: None,
                expected: None,
                status: RankStatus::NotAssessed,
                reason: Some("fixed-effect audit not run".to_string()),
            },
            columns: Vec::new(),
            terms: Vec::new(),
            contrast_bases: Vec::new(),
            aliased_columns: Vec::new(),
            empty_cells: Vec::new(),
            diagnostics: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FixedEffectColumnAudit {
    pub name: String,
    pub source_term: String,
    pub kind: FixedEffectColumnKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FixedEffectColumnKind {
    Intercept,
    Numeric,
    CategoricalDummy,
    CategoricalContrast,
    Interaction,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CategoricalContrastAudit {
    pub variable: String,
    pub levels: Vec<String>,
    pub column_names: Vec<String>,
    pub source: String,
    pub ordered: bool,
    pub explicit: bool,
    pub contrast_matrix: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FixedEffectTermAudit {
    pub term: String,
    pub n_columns: usize,
    pub status: FixedEffectTermStatus,
    pub aliased_columns: Vec<String>,
    pub diagnostics: Vec<Diagnostic>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FixedEffectTermStatus {
    Estimable,
    PartiallyEstimable,
    NotEstimable,
    NotAssessed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmptyCellAudit {
    pub term: String,
    pub factors: Vec<String>,
    pub levels: Vec<String>,
    pub reason: String,
}

/// Grouping-factor summary used by design audits.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GroupingAudit {
    pub name: String,
    pub n_observations: Option<usize>,
    pub n_levels: Option<usize>,
    pub min_obs_per_level: Option<usize>,
    #[serde(default)]
    pub median_obs_per_level: Option<usize>,
    pub max_obs_per_level: Option<usize>,
    pub repeated: Option<bool>,
    pub reason: Option<String>,
}

/// Compact covariance-kernel/dependence-path graph for a compiled design.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CovarianceKernelGraphAudit {
    pub kernels: Vec<CovarianceKernelAudit>,
    pub repeated_units: Vec<DependencePathAudit>,
    pub missing_dependence_paths: Vec<MissingDependencePathAudit>,
}

impl CovarianceKernelGraphAudit {
    pub fn not_assessed() -> Self {
        Self {
            kernels: Vec::new(),
            repeated_units: Vec::new(),
            missing_dependence_paths: Vec::new(),
        }
    }
}

/// One random-effect covariance kernel requested by the semantic model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CovarianceKernelAudit {
    pub term_id: String,
    pub group: String,
    pub parts: Vec<String>,
    pub path: DependencePathKind,
    pub has_intercept: bool,
    pub basis: Vec<String>,
    pub covariance_family: String,
    pub support_status: CovarianceSupportStatus,
    pub expected_parameter_count: usize,
    pub source_syntax: String,
}

/// Whether a dependence path is marginal for one factor or scoped to a cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DependencePathKind {
    Marginal,
    Cell,
    Interaction,
}

/// Observed repetition and model coverage for one dependence path.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DependencePathAudit {
    pub unit: String,
    pub parts: Vec<String>,
    pub path: DependencePathKind,
    pub n_observations: usize,
    pub n_levels: usize,
    pub min_obs_per_level: usize,
    pub max_obs_per_level: usize,
    pub repeated: bool,
    pub covered_by_terms: Vec<String>,
}

/// Repeated dependence path left independent by the requested random effects.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MissingDependencePathAudit {
    pub unit: String,
    pub parts: Vec<String>,
    pub path: DependencePathKind,
    pub n_levels: usize,
    pub max_obs_per_level: usize,
    pub reason: String,
    pub suggested_random_term: String,
}

/// Random-effect term audit placeholder for v0.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RandomTermAudit {
    pub term_id: String,
    pub group: GroupingAudit,
    pub basis_size: usize,
    pub requested_covariance_parameters: usize,
    pub information_budget: RandomEffectInformationBudget,
    pub basis: Vec<BasisAudit>,
    pub diagnostics: Vec<Diagnostic>,
}

/// Deterministic v0 information-budget check for one random-effect term.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RandomEffectInformationBudget {
    pub n_levels: Option<usize>,
    pub basis_dimension: usize,
    pub covariance_family: String,
    pub requested_covariance_parameters: usize,
    pub min_levels_variance: usize,
    pub min_levels_full_covariance: Option<usize>,
    pub effective_n: RandomEffectEffectiveNReport,
    pub status: InformationBudgetStatus,
    pub reason: Option<String>,
}

/// Grouping-level effective-n report for random-effect covariance estimation.
///
/// Rows are useful for fixed-effect precision, but random-effect distributions
/// are primarily informed by independent grouping levels. This struct makes
/// that accounting explicit so a model with many rows but few groups does not
/// look better supported than it is.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RandomEffectEffectiveNReport {
    pub n_rows: Option<usize>,
    pub n_levels: Option<usize>,
    pub min_obs_per_level: Option<usize>,
    pub max_obs_per_level: Option<usize>,
    pub basis_dimension: usize,
    pub covariance_parameters: usize,
    pub levels_per_basis_direction: Option<f64>,
    pub levels_per_covariance_parameter: Option<f64>,
    pub rows_per_covariance_parameter: Option<f64>,
    pub total_rows_can_mislead: bool,
    pub explanation: String,
    pub recommendation: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InformationBudgetStatus {
    Sufficient,
    WeaklySupported,
    TooRich,
    NotAssessable,
}

/// Per-basis-column design support check.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BasisAudit {
    pub name: String,
    pub kind: String,
    pub min_within_group_sd: Option<f64>,
    pub max_within_group_sd: Option<f64>,
    pub supported: Option<bool>,
    pub reason: Option<String>,
}

/// Prefit design audit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DesignAudit {
    pub schema_name: String,
    pub schema_version: u32,
    pub fixed_effect_rank: RankAssessment,
    pub fixed_effects: FixedEffectAudit,
    pub random_terms: Vec<RandomTermAudit>,
    #[serde(default = "CovarianceKernelGraphAudit::not_assessed")]
    pub covariance_kernels: CovarianceKernelGraphAudit,
    pub diagnostics: Vec<Diagnostic>,
}

impl DesignAudit {
    pub fn not_assessed() -> Self {
        Self {
            schema_name: DESIGN_AUDIT_SCHEMA.to_string(),
            schema_version: DESIGN_AUDIT_SCHEMA_VERSION,
            fixed_effect_rank: RankAssessment {
                rank: None,
                expected: None,
                status: RankStatus::NotAssessed,
                reason: Some("design audit not run".to_string()),
            },
            fixed_effects: FixedEffectAudit::not_assessed(),
            random_terms: Vec::new(),
            covariance_kernels: CovarianceKernelGraphAudit::not_assessed(),
            diagnostics: Vec::new(),
        }
    }
}

/// Run the v0 prefit design audit.
pub fn audit_design(semantic_model: &SemanticModel, data: &DataFrame) -> DesignAudit {
    let mut diagnostics = Vec::new();
    let fixed_effects = audit_fixed_effects(semantic_model, data);
    diagnostics.extend(fixed_effects.diagnostics.clone());
    let mut random_terms = semantic_model
        .random_terms
        .iter()
        .map(|term| audit_random_term(term, data, &semantic_model.response, &mut diagnostics))
        .collect::<Vec<_>>();

    for (term_id, diagnostic) in
        fixed_random_redundancy_diagnostics(semantic_model, data, &fixed_effects)
    {
        if let Some(term_audit) = random_terms
            .iter_mut()
            .find(|term_audit| term_audit.term_id == term_id)
        {
            term_audit.diagnostics.push(diagnostic.clone());
        }
        diagnostics.push(diagnostic);
    }

    for (term_id, diagnostic) in scope_note_diagnostics(semantic_model, data) {
        if let Some(term_audit) = random_terms
            .iter_mut()
            .find(|term_audit| term_audit.term_id == term_id)
        {
            term_audit.diagnostics.push(diagnostic.clone());
        }
        diagnostics.push(diagnostic);
    }

    for (term_id, diagnostic) in zerocorr_factor_decorrelation_diagnostics(semantic_model, data) {
        if let Some(term_audit) = random_terms
            .iter_mut()
            .find(|term_audit| term_audit.term_id == term_id)
        {
            term_audit.diagnostics.push(diagnostic.clone());
        }
        diagnostics.push(diagnostic);
    }

    diagnostics.extend(crossed_scalar_correlation_absence_diagnostics(
        semantic_model,
    ));

    let covariance_kernels = audit_covariance_kernels(semantic_model, data);
    diagnostics.extend(
        covariance_kernels
            .missing_dependence_paths
            .iter()
            .map(repeated_unit_unmodeled_diagnostic),
    );

    DesignAudit {
        schema_name: DESIGN_AUDIT_SCHEMA.to_string(),
        schema_version: DESIGN_AUDIT_SCHEMA_VERSION,
        fixed_effect_rank: fixed_effects.rank.clone(),
        fixed_effects,
        random_terms,
        covariance_kernels,
        diagnostics,
    }
}

fn fixed_random_redundancy_diagnostics(
    semantic_model: &SemanticModel,
    data: &DataFrame,
    fixed_effects: &FixedEffectAudit,
) -> Vec<(String, Diagnostic)> {
    let has_fixed_intercept = semantic_model.fixed_terms.iter().any(|term| term == "1");
    if !has_fixed_intercept {
        return Vec::new();
    }

    semantic_model
        .random_terms
        .iter()
        .filter_map(|term| {
            if !term
                .basis
                .iter()
                .any(|basis| basis.kind == RandomCoefficientKind::Intercept)
            {
                return None;
            }

            let GroupingFactorIr::Single { name } = &term.group else {
                return None;
            };
            let fixed_term = fixed_effects.terms.iter().find(|audit| audit.term == *name)?;
            data.categorical(name)?;
            if fixed_term.status == FixedEffectTermStatus::NotEstimable {
                return None;
            }

            let mut diagnostic = Diagnostic::new(
                DiagnosticCode::FixedRandomRedundant,
                DiagnosticSeverity::Warning,
                DiagnosticStage::DesignAudit,
                format!(
                    "random intercept for '{name}' is redundant with fixed-effect term '{name}'"
                ),
            )
            .with_affected_terms(vec![term.source_syntax.text.clone(), name.clone()])
            .with_suggested_actions(vec![
                format!("drop the random intercept term '{}'", term.source_syntax.text),
                format!(
                    "or remove fixed-effect term '{name}' if '{name}' should be modeled as a sampled random unit"
                ),
            ]);
            diagnostic
                .payload
                .insert("group".to_string(), serde_json::json!(name));
            diagnostic.payload.insert(
                "random_term".to_string(),
                serde_json::json!(term.source_syntax.text),
            );
            diagnostic.payload.insert(
                "fixed_term".to_string(),
                serde_json::json!(fixed_term.term),
            );
            if let Some(cat) = data.categorical(name) {
                diagnostic
                    .payload
                    .insert("n_levels".to_string(), serde_json::json!(cat.n_levels()));
            }
            Some((term.id.clone(), diagnostic))
        })
        .collect()
}

fn scope_note_diagnostics(
    semantic_model: &SemanticModel,
    data: &DataFrame,
) -> Vec<(String, Diagnostic)> {
    let fixed_numeric_terms = semantic_model
        .fixed_terms
        .iter()
        .map(String::as_str)
        .filter(|term| *term != "1" && !term.contains(':'))
        .filter(|term| data.numeric(term).is_some())
        .collect::<Vec<_>>();

    let mut diagnostics = Vec::new();
    for term in &semantic_model.random_terms {
        let Some(refs) = grouping_refs(&term.group, data) else {
            continue;
        };
        for &fixed_effect in &fixed_numeric_terms {
            if group_has_random_slope(semantic_model, &term.group, fixed_effect) {
                continue;
            }
            let Some(values) = data.numeric(fixed_effect) else {
                continue;
            };
            let refs = Some(refs.clone());
            let basis = audit_values(fixed_effect, "numeric", &refs, values.iter().copied());
            if basis.supported != Some(true) {
                continue;
            }

            let group = term.group.label();
            let mut diagnostic = Diagnostic::new(
                DiagnosticCode::ScopeNote,
                DiagnosticSeverity::Info,
                DiagnosticStage::DesignAudit,
                format!(
                    "`{}` varies within `{group}`, so a `{group}`-level slope is structurally possible",
                    fixed_effect
                ),
            )
            .with_affected_terms(vec![term.id.clone()])
            .with_suggested_actions(vec![format!(
                "`{}` varies within `{group}`, so a `{group}`-level slope is structurally possible.",
                fixed_effect
            )]);
            diagnostic
                .payload
                .insert("group".to_string(), serde_json::json!(group));
            diagnostic
                .payload
                .insert("fixed_effect".to_string(), serde_json::json!(fixed_effect));
            diagnostic
                .payload
                .insert("varies_within_group".to_string(), serde_json::json!(true));
            diagnostics.push((term.id.clone(), diagnostic));
        }
    }

    diagnostics
}

fn group_has_random_slope(
    semantic_model: &SemanticModel,
    group: &GroupingFactorIr,
    fixed_effect: &str,
) -> bool {
    semantic_model.random_terms.iter().any(|term| {
        term.group == *group && term.basis.iter().any(|basis| basis.name == fixed_effect)
    })
}

fn audit_covariance_kernels(
    semantic_model: &SemanticModel,
    data: &DataFrame,
) -> CovarianceKernelGraphAudit {
    let kernels = semantic_model
        .random_terms
        .iter()
        .map(covariance_kernel_audit)
        .collect::<Vec<_>>();
    let fixed_categorical = fixed_categorical_terms(semantic_model, data);
    let mut candidates = std::collections::BTreeMap::new();

    for term in &semantic_model.random_terms {
        let (parts, path) = dependence_parts_and_path(&term.group);
        insert_dependence_candidate(&mut candidates, parts.clone(), path);
        if path != DependencePathKind::Marginal {
            for part in parts {
                insert_dependence_candidate(
                    &mut candidates,
                    vec![part],
                    DependencePathKind::Marginal,
                );
            }
        }
    }

    for name in data.column_names() {
        if name == semantic_model.response {
            continue;
        }
        if data.categorical(name).is_none() || fixed_categorical.contains(name) {
            continue;
        }
        if is_grouping_like_name(name) {
            insert_dependence_candidate(
                &mut candidates,
                vec![name.to_string()],
                DependencePathKind::Marginal,
            );
        }
    }

    let repeated_units = candidates
        .into_values()
        .filter_map(|candidate| dependence_path_audit(candidate, data, &kernels))
        .filter(|unit| unit.repeated)
        .collect::<Vec<_>>();
    let missing_dependence_paths = repeated_units
        .iter()
        .filter(|unit| {
            unit.covered_by_terms.is_empty()
                && !marginal_fixed_effect_covers_unit(unit, &fixed_categorical)
        })
        .map(|unit| missing_dependence_path_audit(unit, semantic_model))
        .collect::<Vec<_>>();

    CovarianceKernelGraphAudit {
        kernels,
        repeated_units,
        missing_dependence_paths,
    }
}

fn covariance_kernel_audit(term: &RandomTermIr) -> CovarianceKernelAudit {
    let (parts, path) = dependence_parts_and_path(&term.group);
    let basis = term
        .basis
        .iter()
        .map(|basis| basis.name.clone())
        .collect::<Vec<_>>();
    let has_intercept = term
        .basis
        .iter()
        .any(|basis| basis.kind == RandomCoefficientKind::Intercept);
    let effective_covariance = effective_covariance_form(&term.covariance, term.basis.len());

    CovarianceKernelAudit {
        term_id: term.id.clone(),
        group: term.group.label(),
        parts,
        path,
        has_intercept,
        basis,
        covariance_family: covariance_family_label(&effective_covariance),
        support_status: effective_covariance.support_status(),
        expected_parameter_count: requested_covariance_parameters(
            &effective_covariance,
            term.basis.len(),
        ),
        source_syntax: term.source_syntax.text.clone(),
    }
}

#[derive(Debug, Clone)]
struct DependencePathCandidate {
    unit: String,
    parts: Vec<String>,
    path: DependencePathKind,
}

fn insert_dependence_candidate(
    candidates: &mut std::collections::BTreeMap<
        (DependencePathKind, String),
        DependencePathCandidate,
    >,
    parts: Vec<String>,
    path: DependencePathKind,
) {
    if parts.is_empty() {
        return;
    }
    let unit = parts.join(":");
    candidates
        .entry((path, unit.clone()))
        .or_insert(DependencePathCandidate { unit, parts, path });
}

fn dependence_path_audit(
    candidate: DependencePathCandidate,
    data: &DataFrame,
    kernels: &[CovarianceKernelAudit],
) -> Option<DependencePathAudit> {
    let counts = match candidate.path {
        DependencePathKind::Marginal => single_grouping_counts(candidate.parts.first()?, data)?,
        DependencePathKind::Cell | DependencePathKind::Interaction => {
            composite_grouping_counts(&candidate.parts, data)?
        }
    };
    let n_observations = counts.iter().sum::<usize>();
    let min_obs_per_level = counts.iter().copied().min().unwrap_or(0);
    let max_obs_per_level = counts.iter().copied().max().unwrap_or(0);
    let covered_by_terms = kernels
        .iter()
        .filter(|kernel| kernel_covers_dependence_path(kernel, &candidate.parts, candidate.path))
        .map(|kernel| kernel.term_id.clone())
        .collect::<Vec<_>>();

    Some(DependencePathAudit {
        unit: candidate.unit,
        parts: candidate.parts,
        path: candidate.path,
        n_observations,
        n_levels: counts.len(),
        min_obs_per_level,
        max_obs_per_level,
        repeated: max_obs_per_level >= 2,
        covered_by_terms,
    })
}

fn kernel_covers_dependence_path(
    kernel: &CovarianceKernelAudit,
    parts: &[String],
    path: DependencePathKind,
) -> bool {
    kernel.has_intercept && kernel.path == path && kernel.parts == parts
}

fn missing_dependence_path_audit(
    unit: &DependencePathAudit,
    semantic_model: &SemanticModel,
) -> MissingDependencePathAudit {
    MissingDependencePathAudit {
        unit: unit.unit.clone(),
        parts: unit.parts.clone(),
        path: unit.path,
        n_levels: unit.n_levels,
        max_obs_per_level: unit.max_obs_per_level,
        reason: missing_dependence_reason(unit, semantic_model),
        suggested_random_term: suggested_random_intercept(unit),
    }
}

fn missing_dependence_reason(unit: &DependencePathAudit, semantic_model: &SemanticModel) -> String {
    let exact_slope_only = semantic_model.random_terms.iter().any(|term| {
        let (parts, path) = dependence_parts_and_path(&term.group);
        parts == unit.parts
            && path == unit.path
            && !term
                .basis
                .iter()
                .any(|basis| basis.kind == RandomCoefficientKind::Intercept)
    });
    if exact_slope_only {
        return format!(
            "repeated {} path '{}' has random coefficients but no random intercept",
            dependence_path_label(unit.path),
            unit.unit
        );
    }

    let appears_inside_composite = unit.path == DependencePathKind::Marginal
        && semantic_model.random_terms.iter().any(|term| {
            let (parts, path) = dependence_parts_and_path(&term.group);
            path != DependencePathKind::Marginal && parts.iter().any(|part| part == &unit.unit)
        });
    if appears_inside_composite {
        return format!(
            "'{}' appears inside a composite grouping, but cell-level kernels do not model marginal {}-wide dependence",
            unit.unit, unit.unit
        );
    }

    format!(
        "repeated {} path '{}' has no random-intercept covariance kernel",
        dependence_path_label(unit.path),
        unit.unit
    )
}

fn suggested_random_intercept(unit: &DependencePathAudit) -> String {
    match unit.path {
        DependencePathKind::Marginal => format!("(1 | {})", unit.unit),
        DependencePathKind::Cell | DependencePathKind::Interaction => {
            format!("(1 | {})", unit.parts.join(":"))
        }
    }
}

fn repeated_unit_unmodeled_diagnostic(missing: &MissingDependencePathAudit) -> Diagnostic {
    let mut diagnostic = Diagnostic::new(
        DiagnosticCode::RepeatedUnitUnmodeled,
        DiagnosticSeverity::Warning,
        DiagnosticStage::DesignAudit,
        format!(
            "repeated {} unit '{}' is not covered by a random-intercept dependence path",
            dependence_path_label(missing.path),
            missing.unit
        ),
    )
    .with_affected_terms(vec![missing.unit.clone()])
    .with_suggested_actions(vec![
        format!(
            "add a random intercept term such as {} if '{}' is a sampled or repeated unit",
            missing.suggested_random_term, missing.unit
        ),
        format!(
            "model '{}' as a fixed effect only when its levels are designed or directly compared",
            missing.unit
        ),
    ]);
    diagnostic
        .payload
        .insert("unit".to_string(), serde_json::json!(missing.unit));
    diagnostic
        .payload
        .insert("parts".to_string(), serde_json::json!(missing.parts));
    diagnostic.payload.insert(
        "path".to_string(),
        serde_json::json!(dependence_path_label(missing.path)),
    );
    diagnostic
        .payload
        .insert("n_levels".to_string(), serde_json::json!(missing.n_levels));
    diagnostic.payload.insert(
        "max_obs_per_level".to_string(),
        serde_json::json!(missing.max_obs_per_level),
    );
    diagnostic.payload.insert(
        "suggested_random_term".to_string(),
        serde_json::json!(missing.suggested_random_term),
    );
    diagnostic
        .payload
        .insert("reason".to_string(), serde_json::json!(missing.reason));
    diagnostic
}

fn dependence_parts_and_path(group: &GroupingFactorIr) -> (Vec<String>, DependencePathKind) {
    match group {
        GroupingFactorIr::Single { name } => (vec![name.clone()], DependencePathKind::Marginal),
        GroupingFactorIr::Interaction { names } => (names.clone(), DependencePathKind::Interaction),
        GroupingFactorIr::Cell { names } => (names.clone(), DependencePathKind::Cell),
    }
}

fn dependence_path_label(path: DependencePathKind) -> &'static str {
    match path {
        DependencePathKind::Marginal => "marginal",
        DependencePathKind::Cell => "cell",
        DependencePathKind::Interaction => "interaction",
    }
}

fn fixed_categorical_terms(
    semantic_model: &SemanticModel,
    data: &DataFrame,
) -> std::collections::BTreeSet<String> {
    let mut fixed = std::collections::BTreeSet::new();
    for term in &semantic_model.fixed_terms {
        if term == "1" {
            continue;
        }
        for part in term.split(':') {
            if data.categorical(part).is_some() {
                fixed.insert(part.to_string());
            }
        }
    }
    fixed
}

fn marginal_fixed_effect_covers_unit(
    unit: &DependencePathAudit,
    fixed_categorical: &std::collections::BTreeSet<String>,
) -> bool {
    unit.path == DependencePathKind::Marginal && fixed_categorical.contains(&unit.unit)
}

fn single_grouping_counts(name: &str, data: &DataFrame) -> Option<Vec<usize>> {
    let cat = data.categorical(name)?;
    let refs = cat.refs.iter().map(|&r| r as usize).collect::<Vec<_>>();
    Some(counts_from_refs(cat.n_levels(), &refs))
}

fn grouping_refs(group: &GroupingFactorIr, data: &DataFrame) -> Option<Vec<usize>> {
    match group {
        GroupingFactorIr::Single { name } => data
            .categorical(name)
            .map(|cat| cat.refs.iter().map(|&r| r as usize).collect()),
        GroupingFactorIr::Interaction { names } | GroupingFactorIr::Cell { names } => {
            composite_grouping_refs(names, data)
        }
    }
}

fn composite_grouping_refs(names: &[String], data: &DataFrame) -> Option<Vec<usize>> {
    let cats = names
        .iter()
        .map(|name| data.categorical(name))
        .collect::<Option<Vec<_>>>()?;
    let mut level_map = std::collections::BTreeMap::new();
    let mut refs = Vec::with_capacity(data.nrow());
    for row in 0..data.nrow() {
        let key = cats
            .iter()
            .map(|cat| cat.values[row].clone())
            .collect::<Vec<_>>();
        let next = level_map.len();
        let idx = *level_map.entry(key).or_insert(next);
        refs.push(idx);
    }
    Some(refs)
}

fn composite_grouping_counts(names: &[String], data: &DataFrame) -> Option<Vec<usize>> {
    let cats = names
        .iter()
        .map(|name| data.categorical(name))
        .collect::<Option<Vec<_>>>()?;
    let mut level_map = std::collections::BTreeMap::new();
    let mut refs = Vec::with_capacity(data.nrow());
    for row in 0..data.nrow() {
        let key = cats
            .iter()
            .map(|cat| cat.values[row].clone())
            .collect::<Vec<_>>();
        let next = level_map.len();
        let idx = *level_map.entry(key).or_insert(next);
        refs.push(idx);
    }
    Some(counts_from_refs(level_map.len(), &refs))
}

fn is_grouping_like_name(name: &str) -> bool {
    let normalized = name
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect::<String>();
    const UNIT_NAMES: &[&str] = &[
        "animal",
        "batch",
        "block",
        "class",
        "classroom",
        "cluster",
        "dyad",
        "family",
        "firm",
        "g",
        "grp",
        "group",
        "herd",
        "hospital",
        "id",
        "item",
        "lab",
        "mouse",
        "observer",
        "pair",
        "participant",
        "patient",
        "person",
        "plate",
        "plot",
        "rater",
        "rat",
        "sample",
        "school",
        "site",
        "store",
        "subj",
        "subject",
        "unit",
        "worker",
    ];

    UNIT_NAMES.contains(&normalized.as_str())
        || normalized.ends_with("_id")
        || normalized.ends_with("_ids")
        || normalized.ends_with("_unit")
}

fn audit_fixed_effects(semantic_model: &SemanticModel, data: &DataFrame) -> FixedEffectAudit {
    let mut builder = FixedDesignBuilder::new(data);
    for term in &semantic_model.fixed_terms {
        builder.push_term(term);
    }

    let FixedDesignBuild {
        matrix,
        columns,
        term_ranges,
        empty_cells,
        mut diagnostics,
    } = builder.finish();

    let rank = fixed_rank_assessment(&matrix);
    let aliased_columns = if rank.status == RankStatus::RankDeficient {
        aliased_columns(&matrix, rank.rank.unwrap_or(0), &columns)
    } else {
        Vec::new()
    };
    if rank.status == RankStatus::RankDeficient {
        let observed_rank = rank.rank.unwrap_or(0);
        let requested_columns = rank.expected.unwrap_or(0);
        let mut diagnostic = Diagnostic::new(
            DiagnosticCode::FixedEffectRankDeficient,
            DiagnosticSeverity::Warning,
            DiagnosticStage::DesignAudit,
            format!(
                "fixed-effect formula is rank-deficient (rank {observed_rank} of {requested_columns}); some requested coefficients are not separately estimable from the observed data"
            ),
        )
        .with_affected_terms(aliased_columns.clone())
        .with_suggested_actions(vec![
            "drop redundant fixed-effect terms, combine sparse factor levels, or test only estimable contrasts"
                .to_string(),
        ]);
        diagnostic
            .payload
            .insert("rank".to_string(), serde_json::json!(observed_rank));
        diagnostic.payload.insert(
            "requested_columns".to_string(),
            serde_json::json!(requested_columns),
        );
        diagnostic.payload.insert(
            "aliased_columns".to_string(),
            serde_json::json!(aliased_columns.clone()),
        );
        diagnostics.push(diagnostic);
    }

    for empty_cell in &empty_cells {
        let cell = format_factor_level_assignment(&empty_cell.factors, &empty_cell.levels);
        let mut diagnostic = Diagnostic::new(
            DiagnosticCode::FixedEffectEmptyCell,
            DiagnosticSeverity::Warning,
            DiagnosticStage::DesignAudit,
            format!(
                "interaction '{}' has no observations for {}; effects that depend on this cell are not estimable",
                empty_cell.term, cell
            ),
        )
        .with_affected_terms(vec![empty_cell.term.clone()])
        .with_suggested_actions(vec![
            "test estimable contrasts over observed cells or simplify the unsupported interaction"
                .to_string(),
        ]);
        diagnostic.payload.insert(
            "term".to_string(),
            serde_json::json!(empty_cell.term.clone()),
        );
        diagnostic.payload.insert(
            "factors".to_string(),
            serde_json::json!(empty_cell.factors.clone()),
        );
        diagnostic.payload.insert(
            "levels".to_string(),
            serde_json::json!(empty_cell.levels.clone()),
        );
        diagnostic
            .payload
            .insert("cell".to_string(), serde_json::json!(cell));
        diagnostics.push(diagnostic);
    }

    let aliased_set = aliased_columns
        .iter()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    let terms = term_ranges
        .into_iter()
        .map(|range| {
            let aliased = columns[range.start..range.end]
                .iter()
                .filter(|column| aliased_set.contains(&column.name))
                .map(|column| column.name.clone())
                .collect::<Vec<_>>();
            let status = fixed_term_status(range.end - range.start, aliased.len());
            let mut term_diagnostics = Vec::new();
            if status != FixedEffectTermStatus::Estimable {
                term_diagnostics.push(
                    Diagnostic::new(
                        DiagnosticCode::FixedEffectRankDeficient,
                        DiagnosticSeverity::Warning,
                        DiagnosticStage::Estimability,
                        format!(
                            "fixed-effect term '{}' is not fully estimable from the compiled design",
                            range.term
                        ),
                    )
                    .with_affected_terms(vec![range.term.clone()]),
                );
            }

            FixedEffectTermAudit {
                term: range.term,
                n_columns: range.end - range.start,
                status,
                aliased_columns: aliased,
                diagnostics: term_diagnostics,
            }
        })
        .collect::<Vec<_>>();

    FixedEffectAudit {
        n_rows: data.nrow(),
        n_columns: columns.len(),
        rank,
        columns,
        terms,
        contrast_bases: categorical_contrast_audits_for_terms(data, &semantic_model.fixed_terms),
        aliased_columns,
        empty_cells,
        diagnostics,
    }
}

fn format_factor_level_assignment(factors: &[String], levels: &[String]) -> String {
    factors
        .iter()
        .zip(levels.iter())
        .map(|(factor, level)| format!("{factor}={level}"))
        .collect::<Vec<_>>()
        .join(", ")
}

struct FixedDesignBuilder<'a> {
    data: &'a DataFrame,
    columns: Vec<DesignColumn>,
    term_ranges: Vec<TermColumnRange>,
    empty_cells: Vec<EmptyCellAudit>,
    diagnostics: Vec<Diagnostic>,
}

struct FixedDesignBuild {
    matrix: DMatrix<f64>,
    columns: Vec<FixedEffectColumnAudit>,
    term_ranges: Vec<TermColumnRange>,
    empty_cells: Vec<EmptyCellAudit>,
    diagnostics: Vec<Diagnostic>,
}

#[derive(Debug, Clone)]
struct DesignColumn {
    audit: FixedEffectColumnAudit,
    values: DVector<f64>,
}

#[derive(Debug, Clone)]
struct TermColumnRange {
    term: String,
    start: usize,
    end: usize,
}

impl<'a> FixedDesignBuilder<'a> {
    fn new(data: &'a DataFrame) -> Self {
        Self {
            data,
            columns: Vec::new(),
            term_ranges: Vec::new(),
            empty_cells: Vec::new(),
            diagnostics: Vec::new(),
        }
    }

    fn push_term(&mut self, term: &str) {
        let start = self.columns.len();
        if term == "1" {
            self.columns.push(DesignColumn {
                audit: FixedEffectColumnAudit {
                    name: "(Intercept)".to_string(),
                    source_term: term.to_string(),
                    kind: FixedEffectColumnKind::Intercept,
                },
                values: DVector::from_element(self.data.nrow(), 1.0),
            });
        } else if term.contains(':') {
            self.push_interaction_term(term);
        } else {
            self.push_main_effect(term);
        }

        let end = self.columns.len();
        self.term_ranges.push(TermColumnRange {
            term: term.to_string(),
            start,
            end,
        });
    }

    fn push_main_effect(&mut self, name: &str) {
        match self.data.column(name) {
            Some(Column::Numeric(values)) => self.columns.push(DesignColumn {
                audit: FixedEffectColumnAudit {
                    name: name.to_string(),
                    source_term: name.to_string(),
                    kind: FixedEffectColumnKind::Numeric,
                },
                values: DVector::from_column_slice(values),
            }),
            Some(Column::Categorical(cat)) => {
                for encoded in cat.encoded_columns(name, CategoricalCoding::Treatment) {
                    self.columns.push(DesignColumn {
                        audit: FixedEffectColumnAudit {
                            name: encoded.name,
                            source_term: name.to_string(),
                            kind: if encoded.explicit_contrast {
                                FixedEffectColumnKind::CategoricalContrast
                            } else {
                                FixedEffectColumnKind::CategoricalDummy
                            },
                        },
                        values: DVector::from_column_slice(&encoded.values),
                    });
                }
            }
            None => self.missing_fixed_column(name, name),
        }
    }

    fn push_interaction_term(&mut self, term: &str) {
        let factors = term.split(':').map(str::to_string).collect::<Vec<_>>();
        self.empty_cells
            .extend(empty_cells_for_interaction(term, &factors, self.data));

        let mut factor_columns = Vec::new();
        for factor in &factors {
            match design_columns_for_factor(factor, term, self.data) {
                Some(columns) => factor_columns.push(columns),
                None => {
                    self.missing_fixed_column(factor, term);
                    return;
                }
            }
        }

        let mut current = Vec::new();
        append_interaction_products(&factor_columns, 0, &mut current, &mut self.columns);
    }

    fn missing_fixed_column(&mut self, name: &str, term: &str) {
        self.diagnostics.push(
            Diagnostic::new(
                DiagnosticCode::FixedEffectColumnMissing,
                DiagnosticSeverity::Error,
                DiagnosticStage::DesignAudit,
                format!("fixed-effect column '{name}' is missing from data"),
            )
            .with_affected_terms(vec![term.to_string()]),
        );
    }

    fn finish(self) -> FixedDesignBuild {
        let audit_columns = self
            .columns
            .iter()
            .map(|column| column.audit.clone())
            .collect::<Vec<_>>();
        let matrix = design_matrix_from_columns(self.data.nrow(), &self.columns);
        FixedDesignBuild {
            matrix,
            columns: audit_columns,
            term_ranges: self.term_ranges,
            empty_cells: self.empty_cells,
            diagnostics: self.diagnostics,
        }
    }
}

fn design_columns_for_factor(
    factor: &str,
    source_term: &str,
    data: &DataFrame,
) -> Option<Vec<DesignColumn>> {
    match data.column(factor)? {
        Column::Numeric(values) => Some(vec![DesignColumn {
            audit: FixedEffectColumnAudit {
                name: factor.to_string(),
                source_term: source_term.to_string(),
                kind: FixedEffectColumnKind::Numeric,
            },
            values: DVector::from_column_slice(values),
        }]),
        Column::Categorical(cat) => {
            let columns = cat
                .encoded_columns(factor, CategoricalCoding::Treatment)
                .into_iter()
                .map(|encoded| DesignColumn {
                    audit: FixedEffectColumnAudit {
                        name: encoded.name,
                        source_term: source_term.to_string(),
                        kind: if encoded.explicit_contrast {
                            FixedEffectColumnKind::CategoricalContrast
                        } else {
                            FixedEffectColumnKind::CategoricalDummy
                        },
                    },
                    values: DVector::from_column_slice(&encoded.values),
                })
                .collect();
            Some(columns)
        }
    }
}

fn append_interaction_products(
    factors: &[Vec<DesignColumn>],
    index: usize,
    current: &mut Vec<DesignColumn>,
    out: &mut Vec<DesignColumn>,
) {
    if index == factors.len() {
        if current.is_empty() {
            return;
        }
        let n = current[0].values.len();
        let mut values = DVector::from_element(n, 1.0);
        let mut names = Vec::with_capacity(current.len());
        let source_term = current[0].audit.source_term.clone();
        for column in current.iter() {
            values.component_mul_assign(&column.values);
            names.push(column.audit.name.clone());
        }
        out.push(DesignColumn {
            audit: FixedEffectColumnAudit {
                name: names.join(":"),
                source_term,
                kind: FixedEffectColumnKind::Interaction,
            },
            values,
        });
        return;
    }

    for column in &factors[index] {
        current.push(column.clone());
        append_interaction_products(factors, index + 1, current, out);
        current.pop();
    }
}

fn design_matrix_from_columns(n_rows: usize, columns: &[DesignColumn]) -> DMatrix<f64> {
    if columns.is_empty() {
        return DMatrix::zeros(n_rows, 0);
    }
    let mut matrix = DMatrix::zeros(n_rows, columns.len());
    for (index, column) in columns.iter().enumerate() {
        matrix.set_column(index, &column.values);
    }
    matrix
}

/// Column count above which the audit tries the Gram full-rank
/// certificate before the O(n·p²) Householder pass. Wide (typically
/// high-cardinality categorical) designs are almost always comfortably
/// full rank, and the certificate answers that case from a p×p
/// factorization; ambiguous designs still take the exact QR below, so
/// the reported rank is unchanged in every case.
const GRAM_RANK_FAST_PATH_MIN_COLS: usize = 32;

fn fixed_rank_assessment(matrix: &DMatrix<f64>) -> RankAssessment {
    let expected = matrix.ncols();
    let rank = if expected == 0 {
        0
    } else if expected >= GRAM_RANK_FAST_PATH_MIN_COLS
        && crate::linalg::gram_full_rank_certificate(
            &matrix.tr_mul(matrix),
            1e-8,
            crate::linalg::GRAM_CERTIFICATE_SAFETY_FACTOR,
        )
        .is_certified()
    {
        expected
    } else {
        let (rank, _pivots) = stats_rank_with_tol(matrix, 1e-8);
        rank
    };
    RankAssessment {
        rank: Some(rank),
        expected: Some(expected),
        status: if rank == expected {
            RankStatus::FullRank
        } else {
            RankStatus::RankDeficient
        },
        reason: if rank == expected {
            None
        } else {
            Some("fixed-effect columns are linearly dependent under tolerance 1e-8".to_string())
        },
    }
}

fn aliased_columns(
    matrix: &DMatrix<f64>,
    rank: usize,
    columns: &[FixedEffectColumnAudit],
) -> Vec<String> {
    if rank >= columns.len() {
        return Vec::new();
    }
    let (_rank, pivots, _r) = pivoted_qr_with_tol(matrix, 1e-8);
    let mut aliased = pivots
        .into_iter()
        .skip(rank)
        .filter_map(|index| columns.get(index))
        .map(|column| column.name.clone())
        .collect::<Vec<_>>();
    aliased.sort();
    aliased
}

fn fixed_term_status(n_columns: usize, n_aliased: usize) -> FixedEffectTermStatus {
    if n_columns == 0 {
        FixedEffectTermStatus::NotEstimable
    } else if n_aliased == 0 {
        FixedEffectTermStatus::Estimable
    } else if n_aliased < n_columns {
        FixedEffectTermStatus::PartiallyEstimable
    } else {
        FixedEffectTermStatus::NotEstimable
    }
}

fn empty_cells_for_interaction(
    term: &str,
    factors: &[String],
    data: &DataFrame,
) -> Vec<EmptyCellAudit> {
    let categorical = factors
        .iter()
        .filter_map(|factor| data.categorical(factor).map(|cat| (factor, cat)))
        .collect::<Vec<_>>();
    if categorical.len() < 2 {
        return Vec::new();
    }

    let mut observed = std::collections::BTreeSet::new();
    for row in 0..data.nrow() {
        let key = categorical
            .iter()
            .map(|(_factor, cat)| cat.values[row].clone())
            .collect::<Vec<_>>();
        observed.insert(key);
    }

    let mut expected = Vec::new();
    append_level_products(&categorical, 0, &mut Vec::new(), &mut expected);

    expected
        .into_iter()
        .filter(|levels| !observed.contains(levels))
        .map(|levels| EmptyCellAudit {
            term: term.to_string(),
            factors: categorical
                .iter()
                .map(|(factor, _cat)| (*factor).clone())
                .collect(),
            levels,
            reason: "no observations exist for this factor combination".to_string(),
        })
        .collect()
}

fn append_level_products(
    factors: &[(&String, &crate::model::data::CategoricalColumn)],
    index: usize,
    current: &mut Vec<String>,
    out: &mut Vec<Vec<String>>,
) {
    if index == factors.len() {
        out.push(current.clone());
        return;
    }

    for level in &factors[index].1.levels {
        current.push(level.clone());
        append_level_products(factors, index + 1, current, out);
        current.pop();
    }
}

fn audit_random_term(
    term: &RandomTermIr,
    data: &DataFrame,
    response: &str,
    global_diagnostics: &mut Vec<Diagnostic>,
) -> RandomTermAudit {
    let (group, refs) = grouping_audit(&term.group, data, global_diagnostics);
    let mut diagnostics = Vec::new();
    let basis = expanded_basis_audit(term, &refs, data);
    if basis_was_expanded(term, &basis) {
        let diagnostic = expanded_basis_diagnostic(term, &basis, data);
        global_diagnostics.push(diagnostic.clone());
        diagnostics.push(diagnostic);
    }

    let basis_dimension = basis.len();
    let effective_covariance = effective_covariance_form(&term.covariance, basis_dimension);
    let requested_covariance_parameters =
        requested_covariance_parameters(&effective_covariance, basis_dimension);
    let information_budget = information_budget(
        &group,
        &effective_covariance,
        basis_dimension,
        requested_covariance_parameters,
    );
    if let Some(diagnostic) = unsupported_structured_covariance_diagnostic(term) {
        global_diagnostics.push(diagnostic.clone());
        diagnostics.push(diagnostic);
    }

    for basis_audit in &basis {
        if basis_audit.supported == Some(false) {
            let unsupported = Diagnostic::new(
                DiagnosticCode::RandomSlopeUnsupported,
                DiagnosticSeverity::Warning,
                DiagnosticStage::DesignAudit,
                format!(
                    "random slope '{}' has no detectable within-group variation",
                    basis_audit.name
                ),
            )
            .with_affected_terms(vec![term.source_syntax.text.clone()]);
            diagnostics.push(unsupported.clone());

            let structural = structural_refusal_diagnostic(term, basis_audit);
            global_diagnostics.push(structural.clone());
            diagnostics.push(structural);
        }
    }
    if let Some(diagnostic) = response_constant_within_group_diagnostic(term, data, response, &refs)
    {
        global_diagnostics.push(diagnostic.clone());
        diagnostics.push(diagnostic);
    }
    if information_budget.status == InformationBudgetStatus::WeaklySupported
        || information_budget.status == InformationBudgetStatus::TooRich
    {
        let diagnostic = information_budget_diagnostic(term, &information_budget);
        global_diagnostics.push(diagnostic.clone());
        diagnostics.push(diagnostic);
    }
    if information_budget.status == InformationBudgetStatus::WeaklySupported {
        let diagnostic = support_note_diagnostic(term, &information_budget);
        global_diagnostics.push(diagnostic.clone());
        diagnostics.push(diagnostic);
    }

    RandomTermAudit {
        term_id: term.id.clone(),
        group,
        basis_size: basis_dimension,
        requested_covariance_parameters,
        information_budget,
        basis,
        diagnostics,
    }
}

fn unsupported_structured_covariance_diagnostic(term: &RandomTermIr) -> Option<Diagnostic> {
    let CovarianceForm::Structured { kind } = &term.covariance else {
        return None;
    };
    let family = kind.label();
    let mut diagnostic = Diagnostic::new(
        DiagnosticCode::Unsupported,
        DiagnosticSeverity::Error,
        DiagnosticStage::DesignAudit,
        format!(
            "structured random-effect covariance family `{family}` is parsed but not fitted in v1.0"
        ),
    )
    .with_affected_terms(vec![term.source_syntax.user_text().to_string()])
    .with_suggested_actions(vec![
        "use an unstructured `|` random-effect term or diagonal `diag(...)` / `||` term for fitted v1.0 models".to_string(),
        "keep the structured term as an expected-refuse fixture until fitted covariance kernels land".to_string(),
    ]);
    diagnostic
        .payload
        .insert("covariance_family".to_string(), serde_json::json!(family));
    diagnostic.payload.insert(
        "support_status".to_string(),
        serde_json::json!("parsed_refused"),
    );
    Some(diagnostic)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RandomBasisCoding {
    Treatment,
    CellMeans,
}

fn expanded_basis_audit(
    term: &RandomTermIr,
    refs: &Option<Vec<usize>>,
    data: &DataFrame,
) -> Vec<BasisAudit> {
    let coding = random_basis_coding(term);
    let mut basis = Vec::new();

    for coefficient in &term.basis {
        match coefficient.kind {
            RandomCoefficientKind::Intercept => basis.push(BasisAudit {
                name: coefficient.name.clone(),
                kind: "intercept".to_string(),
                min_within_group_sd: None,
                max_within_group_sd: None,
                supported: Some(true),
                reason: None,
            }),
            RandomCoefficientKind::Slope => {
                basis.extend(expand_single_basis_audit(coefficient, refs, data, coding));
            }
            RandomCoefficientKind::Interaction => {
                basis.extend(expand_interaction_basis_audit(
                    coefficient,
                    refs,
                    data,
                    coding,
                ));
            }
            RandomCoefficientKind::Unsupported => basis.push(BasisAudit {
                name: coefficient.name.clone(),
                kind: "unsupported".to_string(),
                min_within_group_sd: None,
                max_within_group_sd: None,
                supported: None,
                reason: Some("basis column is unsupported by semantic compiler".to_string()),
            }),
        }
    }

    basis
}

fn expand_single_basis_audit(
    coefficient: &RandomCoefficient,
    refs: &Option<Vec<usize>>,
    data: &DataFrame,
    coding: RandomBasisCoding,
) -> Vec<BasisAudit> {
    match data.column(&coefficient.source) {
        Some(Column::Numeric(values)) => vec![audit_values(
            &coefficient.name,
            "slope",
            refs,
            values.iter().copied(),
        )],
        Some(Column::Categorical(cat)) => {
            categorical_basis_audits(&coefficient.source, cat, refs, coding)
        }
        None => vec![BasisAudit {
            name: coefficient.name.clone(),
            kind: "slope".to_string(),
            min_within_group_sd: None,
            max_within_group_sd: None,
            supported: None,
            reason: Some("slope column is missing".to_string()),
        }],
    }
}

fn expand_interaction_basis_audit(
    coefficient: &RandomCoefficient,
    refs: &Option<Vec<usize>>,
    data: &DataFrame,
    coding: RandomBasisCoding,
) -> Vec<BasisAudit> {
    let vars = coefficient
        .source
        .split(':')
        .map(str::to_string)
        .collect::<Vec<_>>();
    let Ok(per_var) = expanded_interaction_columns(&vars, data, coding) else {
        return vec![BasisAudit {
            name: coefficient.name.clone(),
            kind: "interaction".to_string(),
            min_within_group_sd: None,
            max_within_group_sd: None,
            supported: None,
            reason: Some("one or more interaction columns are missing".to_string()),
        }];
    };

    cartesian_audit_columns(&per_var)
        .into_iter()
        .map(|(name, values)| audit_values(&name, "interaction", refs, values))
        .collect()
}

fn categorical_basis_audits(
    name: &str,
    cat: &crate::model::data::CategoricalColumn,
    refs: &Option<Vec<usize>>,
    coding: RandomBasisCoding,
) -> Vec<BasisAudit> {
    cat.encoded_columns(name, categorical_coding(coding))
        .into_iter()
        .map(|encoded| {
            audit_values(
                &encoded.name,
                if encoded.explicit_contrast {
                    "categorical_contrast"
                } else if coding == RandomBasisCoding::Treatment {
                    "categorical_dummy"
                } else {
                    "categorical_cell"
                },
                refs,
                encoded.values,
            )
        })
        .collect()
}

fn categorical_coding(coding: RandomBasisCoding) -> CategoricalCoding {
    match coding {
        RandomBasisCoding::Treatment => CategoricalCoding::Treatment,
        RandomBasisCoding::CellMeans => CategoricalCoding::CellMeans,
    }
}

fn categorical_contrast_audits_for_terms(
    data: &DataFrame,
    fixed_terms: &[String],
) -> Vec<CategoricalContrastAudit> {
    let mut variables = Vec::new();
    for term in fixed_terms {
        if term == "1" || term == "0" {
            continue;
        }
        for variable in term.split(':') {
            if data.categorical(variable).is_some()
                && !variables
                    .iter()
                    .any(|existing: &String| existing == variable)
            {
                variables.push(variable.to_string());
            }
        }
    }

    variables
        .into_iter()
        .filter_map(|variable| {
            let cat = data.categorical(&variable)?;
            let explicit = cat.contrast.is_some();
            let (column_names, source, ordered, contrast_matrix) = match &cat.contrast {
                Some(contrast) => (
                    contrast.column_names.clone(),
                    contrast.source.as_str().to_string(),
                    contrast.ordered,
                    matrix_rows(&contrast.matrix),
                ),
                None => {
                    let column_names = cat
                        .levels
                        .iter()
                        .skip(1)
                        .map(|level| format!("{variable}: {level}"))
                        .collect::<Vec<_>>();
                    (
                        column_names,
                        ContrastSource::Treatment.as_str().to_string(),
                        false,
                        default_treatment_matrix(cat.levels.len()),
                    )
                }
            };
            Some(CategoricalContrastAudit {
                variable,
                levels: cat.levels.clone(),
                column_names,
                source,
                ordered,
                explicit,
                contrast_matrix,
            })
        })
        .collect()
}

fn matrix_rows(matrix: &DMatrix<f64>) -> Vec<Vec<f64>> {
    (0..matrix.nrows())
        .map(|row| {
            (0..matrix.ncols())
                .map(|col| matrix[(row, col)])
                .collect::<Vec<_>>()
        })
        .collect()
}

fn default_treatment_matrix(n_levels: usize) -> Vec<Vec<f64>> {
    (0..n_levels)
        .map(|row| {
            (1..n_levels)
                .map(|col_level| f64::from(row == col_level))
                .collect::<Vec<_>>()
        })
        .collect()
}

fn expanded_interaction_columns(
    vars: &[String],
    data: &DataFrame,
    coding: RandomBasisCoding,
) -> Result<Vec<Vec<(String, Vec<f64>)>>, ()> {
    vars.iter()
        .map(|name| match data.column(name) {
            Some(Column::Numeric(values)) => Ok(vec![(name.clone(), values.clone())]),
            Some(Column::Categorical(cat)) => Ok(cat
                .encoded_columns(name, categorical_coding(coding))
                .into_iter()
                .map(|column| (column.name, column.values))
                .collect()),
            None => Err(()),
        })
        .collect()
}

fn cartesian_audit_columns(per_var: &[Vec<(String, Vec<f64>)>]) -> Vec<(String, Vec<f64>)> {
    if per_var.is_empty() {
        return Vec::new();
    }

    let n = per_var
        .iter()
        .find_map(|cols| cols.first().map(|(_, values)| values.len()))
        .unwrap_or(0);
    let mut acc = vec![(String::new(), vec![1.0; n])];
    for cols in per_var {
        let mut next = Vec::with_capacity(acc.len() * cols.len());
        for (acc_name, acc_values) in &acc {
            for (name, values) in cols {
                let product = acc_values
                    .iter()
                    .zip(values.iter())
                    .map(|(left, right)| left * right)
                    .collect::<Vec<_>>();
                let joined = if acc_name.is_empty() {
                    name.clone()
                } else {
                    format!("{acc_name}:{name}")
                };
                next.push((joined, product));
            }
        }
        acc = next;
    }
    acc
}

fn basis_was_expanded(term: &RandomTermIr, basis: &[BasisAudit]) -> bool {
    let semantic = term
        .basis
        .iter()
        .map(|basis| basis.name.as_str())
        .collect::<Vec<_>>();
    let expanded = basis
        .iter()
        .map(|basis| basis.name.as_str())
        .collect::<Vec<_>>();
    semantic != expanded
}

fn expanded_basis_diagnostic(
    term: &RandomTermIr,
    basis: &[BasisAudit],
    data: &DataFrame,
) -> Diagnostic {
    let coding = random_basis_coding(term);
    let explicit_contrast_variables = explicit_contrast_variables(term, data);
    let contrast_available = !explicit_contrast_variables.is_empty();
    let contrast_used = contrast_available && coding == RandomBasisCoding::Treatment;
    let message = if contrast_available && coding == RandomBasisCoding::CellMeans {
        format!(
            "random-effect term '{}' uses cell-means coding by no-intercept categorical formula semantics; supplied contrast basis for {} was not used for this term",
            term.source_syntax.text,
            explicit_contrast_variables.join(", ")
        )
    } else {
        "random-effect basis was expanded into optimizer columns".to_string()
    };
    let mut diagnostic = Diagnostic::new(
        DiagnosticCode::FormulaCanonicalized,
        DiagnosticSeverity::Info,
        DiagnosticStage::DesignAudit,
        message,
    )
    .with_affected_terms(vec![term.source_syntax.text.clone()]);
    diagnostic.payload.insert(
        "semantic_basis".to_string(),
        serde_json::json!(term
            .basis
            .iter()
            .map(|basis| basis.name.clone())
            .collect::<Vec<_>>()),
    );
    diagnostic.payload.insert(
        "expanded_basis".to_string(),
        serde_json::json!(basis
            .iter()
            .map(|basis| basis.name.clone())
            .collect::<Vec<_>>()),
    );
    if contrast_available {
        diagnostic.payload.insert(
            "basis_coding".to_string(),
            serde_json::json!(random_basis_coding_label(coding)),
        );
        diagnostic
            .payload
            .insert("contrast_available".to_string(), serde_json::json!(true));
        diagnostic.payload.insert(
            "contrast_used".to_string(),
            serde_json::json!(contrast_used),
        );
        diagnostic.payload.insert(
            "contrast_variables".to_string(),
            serde_json::json!(explicit_contrast_variables),
        );
    }
    if contrast_available && coding == RandomBasisCoding::CellMeans {
        diagnostic.payload.insert(
            "reason".to_string(),
            serde_json::json!("no_intercept_categorical_formula_semantics"),
        );
    }
    diagnostic
}

fn random_basis_coding(term: &RandomTermIr) -> RandomBasisCoding {
    if term.intercept == InterceptPolicy::Omitted {
        RandomBasisCoding::CellMeans
    } else {
        RandomBasisCoding::Treatment
    }
}

fn random_basis_coding_label(coding: RandomBasisCoding) -> &'static str {
    match coding {
        RandomBasisCoding::Treatment => "treatment",
        RandomBasisCoding::CellMeans => "cell_means",
    }
}

fn explicit_contrast_variables(term: &RandomTermIr, data: &DataFrame) -> Vec<String> {
    let mut variables = std::collections::BTreeSet::new();
    for coefficient in &term.basis {
        for variable in coefficient.source.split(':') {
            if data
                .categorical(variable)
                .and_then(|cat| cat.contrast.as_ref())
                .is_some()
            {
                variables.insert(variable.to_string());
            }
        }
    }
    variables.into_iter().collect()
}

fn grouping_audit(
    group: &GroupingFactorIr,
    data: &DataFrame,
    diagnostics: &mut Vec<Diagnostic>,
) -> (GroupingAudit, Option<Vec<usize>>) {
    match group {
        GroupingFactorIr::Single { name } => match data.categorical(name) {
            Some(cat) => {
                let refs = cat.refs.iter().map(|&r| r as usize).collect::<Vec<_>>();
                let counts = counts_from_refs(cat.n_levels(), &refs);
                (audit_from_counts(name.clone(), counts, None), Some(refs))
            }
            None => {
                diagnostics.push(
                    Diagnostic::new(
                        DiagnosticCode::NotIdentifiable,
                        DiagnosticSeverity::Error,
                        DiagnosticStage::DesignAudit,
                        format!("grouping factor '{name}' is missing or not categorical"),
                    )
                    .with_affected_terms(vec![name.clone()]),
                );
                (
                    GroupingAudit {
                        name: name.clone(),
                        n_observations: None,
                        n_levels: None,
                        min_obs_per_level: None,
                        median_obs_per_level: None,
                        max_obs_per_level: None,
                        repeated: None,
                        reason: Some("missing or non-categorical grouping factor".to_string()),
                    },
                    None,
                )
            }
        },
        GroupingFactorIr::Interaction { names } | GroupingFactorIr::Cell { names } => {
            interaction_grouping_audit(names, data, diagnostics)
        }
    }
}

fn interaction_grouping_audit(
    names: &[String],
    data: &DataFrame,
    diagnostics: &mut Vec<Diagnostic>,
) -> (GroupingAudit, Option<Vec<usize>>) {
    let label = names.join(":");
    let mut cats = Vec::new();
    for name in names {
        match data.categorical(name) {
            Some(cat) => cats.push(cat),
            None => {
                diagnostics.push(
                    Diagnostic::new(
                        DiagnosticCode::NotIdentifiable,
                        DiagnosticSeverity::Error,
                        DiagnosticStage::DesignAudit,
                        format!("grouping factor '{name}' is missing or not categorical"),
                    )
                    .with_affected_terms(vec![name.clone()]),
                );
                return (
                    GroupingAudit {
                        name: label,
                        n_observations: None,
                        n_levels: None,
                        min_obs_per_level: None,
                        median_obs_per_level: None,
                        max_obs_per_level: None,
                        repeated: None,
                        reason: Some("one or more grouping factors are missing".to_string()),
                    },
                    None,
                );
            }
        }
    }

    let mut level_map = std::collections::BTreeMap::new();
    let mut refs = Vec::with_capacity(data.nrow());
    for row in 0..data.nrow() {
        let key = cats
            .iter()
            .map(|cat| cat.values[row].as_str())
            .collect::<Vec<_>>()
            .join(":");
        let next = level_map.len();
        let idx = *level_map.entry(key).or_insert(next);
        refs.push(idx);
    }

    let counts = counts_from_refs(level_map.len(), &refs);
    (audit_from_counts(label, counts, None), Some(refs))
}

fn audit_from_counts(name: String, counts: Vec<usize>, reason: Option<String>) -> GroupingAudit {
    let n_observations = counts.iter().sum::<usize>();
    let min_obs_per_level = counts.iter().copied().min();
    let median_obs_per_level = median_count(&counts);
    let max_obs_per_level = counts.iter().copied().max();
    GroupingAudit {
        name,
        n_observations: Some(n_observations),
        n_levels: Some(counts.len()),
        min_obs_per_level,
        median_obs_per_level,
        max_obs_per_level,
        repeated: max_obs_per_level.map(|max| max >= 2),
        reason,
    }
}

fn median_count(counts: &[usize]) -> Option<usize> {
    if counts.is_empty() {
        return None;
    }

    let mut sorted = counts.to_vec();
    sorted.sort_unstable();
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        Some(sorted[mid])
    } else {
        Some(sorted[mid - 1] + (sorted[mid] - sorted[mid - 1]) / 2)
    }
}

fn counts_from_refs(n_levels: usize, refs: &[usize]) -> Vec<usize> {
    let mut counts = vec![0; n_levels];
    for &idx in refs {
        if let Some(count) = counts.get_mut(idx) {
            *count += 1;
        }
    }
    counts
}

fn audit_values<I>(name: &str, kind: &str, refs: &Option<Vec<usize>>, values: I) -> BasisAudit
where
    I: IntoIterator<Item = f64>,
{
    let Some(refs) = refs else {
        return BasisAudit {
            name: name.to_string(),
            kind: kind.to_string(),
            min_within_group_sd: None,
            max_within_group_sd: None,
            supported: None,
            reason: Some("grouping refs unavailable".to_string()),
        };
    };

    let n_levels = refs.iter().copied().max().map(|m| m + 1).unwrap_or(0);
    let mut by_level = vec![Vec::new(); n_levels];
    for (&group, value) in refs.iter().zip(values) {
        by_level[group].push(value);
    }

    let sds = by_level
        .iter()
        .filter(|vals| vals.len() >= 2)
        .map(|vals| sample_sd(vals))
        .collect::<Vec<_>>();
    let min_sd = sds.iter().copied().reduce(f64::min);
    let max_sd = sds.iter().copied().reduce(f64::max);
    let supported = max_sd.map(|sd| sd > 1e-8);

    BasisAudit {
        name: name.to_string(),
        kind: kind.to_string(),
        min_within_group_sd: min_sd,
        max_within_group_sd: max_sd,
        supported,
        reason: if supported == Some(false) {
            Some("all repeated groups have within-group sd <= 1e-8".to_string())
        } else {
            None
        },
    }
}

fn sample_sd(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let var = values
        .iter()
        .map(|value| {
            let centered = value - mean;
            centered * centered
        })
        .sum::<f64>()
        / (values.len() - 1) as f64;
    var.sqrt()
}

fn response_constant_within_group_diagnostic(
    term: &RandomTermIr,
    data: &DataFrame,
    response: &str,
    refs: &Option<Vec<usize>>,
) -> Option<Diagnostic> {
    if !term
        .basis
        .iter()
        .any(|basis| basis.kind == RandomCoefficientKind::Intercept)
    {
        return None;
    }
    let refs = refs.as_ref()?;
    let y = data.numeric(response)?;
    if refs.len() != y.len() {
        return None;
    }

    let n_levels = refs.iter().copied().max().map(|max| max + 1).unwrap_or(0);
    if n_levels == 0 {
        return None;
    }
    let mut y_by_level = vec![Vec::new(); n_levels];
    for (&group, &value) in refs.iter().zip(y.iter()) {
        y_by_level[group].push(value);
    }

    let repeated_levels = y_by_level.iter().filter(|values| values.len() >= 2).count();
    if repeated_levels == 0 {
        return None;
    }
    let constant_response_levels = y_by_level
        .iter()
        .filter(|values| values.len() >= 2 && sample_sd(values) <= 1e-8)
        .count();
    if constant_response_levels == 0 {
        return None;
    }

    let constant_fraction = constant_response_levels as f64 / repeated_levels as f64;
    if constant_fraction < 0.8 {
        return None;
    }

    let varying_numeric_columns = data
        .column_names()
        .into_iter()
        .filter(|name| *name != response)
        .filter_map(|name| {
            let values = data.numeric(name)?;
            numeric_varies_within_constant_response_group(refs, &y_by_level, values)
                .then(|| name.to_string())
        })
        .collect::<Vec<_>>();
    if varying_numeric_columns.is_empty() {
        return None;
    }

    let group_name = grouping_factor_label(&term.group);
    let mut diagnostic = Diagnostic::new(
        DiagnosticCode::NotIdentifiable,
        DiagnosticSeverity::Error,
        DiagnosticStage::DesignAudit,
        format!(
            "response '{response}' is constant within {constant_response_levels} of \
             {repeated_levels} repeated levels of random-intercept group '{group_name}', \
             while numeric predictor(s) vary within those levels"
        ),
    )
    .with_affected_terms(vec![term.source_syntax.text.clone()])
    .with_suggested_actions(vec![
        format!(
            "Check whether '{response}' is measured once per '{group_name}' level and then repeated across rows; if so, aggregate to one row per lower-level unit or move predictors to the correct observation level."
        ),
        "Revise the random-effect/observation-unit structure before changing optimizers; optimizer changes cannot identify within-unit effects when the response is duplicated.".to_string(),
    ]);
    diagnostic
        .payload
        .insert("response".to_string(), serde_json::json!(response));
    diagnostic
        .payload
        .insert("group".to_string(), serde_json::json!(group_name));
    diagnostic.payload.insert(
        "repeated_levels".to_string(),
        serde_json::json!(repeated_levels),
    );
    diagnostic.payload.insert(
        "constant_response_levels".to_string(),
        serde_json::json!(constant_response_levels),
    );
    diagnostic.payload.insert(
        "varying_numeric_columns".to_string(),
        serde_json::json!(varying_numeric_columns),
    );
    Some(diagnostic)
}

fn numeric_varies_within_constant_response_group(
    refs: &[usize],
    y_by_level: &[Vec<f64>],
    values: &[f64],
) -> bool {
    if refs.len() != values.len() {
        return false;
    }

    let mut x_by_level = vec![Vec::new(); y_by_level.len()];
    for (&group, &value) in refs.iter().zip(values.iter()) {
        x_by_level[group].push(value);
    }

    y_by_level
        .iter()
        .zip(x_by_level.iter())
        .any(|(y_values, x_values)| {
            y_values.len() >= 2 && sample_sd(y_values) <= 1e-8 && sample_sd(x_values) > 1e-8
        })
}

fn grouping_factor_label(group: &GroupingFactorIr) -> String {
    match group {
        GroupingFactorIr::Single { name } => name.clone(),
        GroupingFactorIr::Interaction { names } | GroupingFactorIr::Cell { names } => {
            names.join(":")
        }
    }
}

/// A factor inside a zero-correlation (`||`) random term is fully
/// decorrelated here: its treatment-coded level contrasts each get an
/// independent variance and no within-factor level covariances are estimated.
/// That is a deliberate, strictly smaller covariance family than giving the
/// factor its own correlated cell-means block, and zero-correlation
/// expansions of factor terms are known to differ across mixed-model
/// implementations — so say it out loud at design-audit time, where the data
/// reveals which slope coefficients are factors.
fn zerocorr_factor_decorrelation_diagnostics(
    semantic_model: &SemanticModel,
    data: &DataFrame,
) -> Vec<(String, Diagnostic)> {
    semantic_model
        .random_terms
        .iter()
        .filter(|term| term.block_group.is_some())
        .filter_map(|term| {
            let basis = term.basis.first()?;
            if basis.kind != RandomCoefficientKind::Slope {
                return None;
            }
            let factor = data.categorical(&basis.name)?;
            if factor.n_levels() < 2 {
                return None;
            }
            let group = grouping_factor_label(&term.group);
            let name = &basis.name;
            let mut diagnostic = Diagnostic::new(
                DiagnosticCode::CovarianceAssumption,
                DiagnosticSeverity::Info,
                DiagnosticStage::DesignAudit,
                format!(
                    "zero-correlation syntax fully decorrelates factor '{name}' within '{group}': \
                     each treatment-coded level contrast of '{name}' receives an independent \
                     variance and no within-factor level covariances are estimated"
                ),
            )
            .with_affected_terms(vec![term.source_syntax.user_text().to_string()])
            .with_suggested_actions(vec![
                format!(
                    "to estimate within-factor level covariances for '{name}', give it its own \
                     correlated cell-means block `(0 + {name} | {group})` and keep the remaining \
                     coefficients under zero-correlation terms"
                ),
                "zero-correlation expansions of factor terms differ across mixed-model \
                 implementations; when matching an external fit, write the intended expansion \
                 explicitly instead of relying on `||`"
                    .to_string(),
            ]);
            diagnostic
                .payload
                .insert("group".to_string(), serde_json::json!(group));
            diagnostic
                .payload
                .insert("factor".to_string(), serde_json::json!(name));
            diagnostic
                .payload
                .insert("n_levels".to_string(), serde_json::json!(factor.n_levels()));
            diagnostic.payload.insert(
                "reason".to_string(),
                serde_json::json!("double_bar_factor_term"),
            );
            diagnostic.payload.insert(
                "dropped".to_string(),
                serde_json::json!("within_factor_level_covariances"),
            );
            diagnostic.payload.insert(
                "correlated_block_equivalent".to_string(),
                serde_json::json!(format!("(0 + {name} | {group})")),
            );
            Some((term.id.clone(), diagnostic))
        })
        .collect()
}

fn crossed_scalar_correlation_absence_diagnostics(
    semantic_model: &SemanticModel,
) -> Vec<Diagnostic> {
    let scalar_intercepts = semantic_model
        .random_terms
        .iter()
        .filter(|term| is_scalar_random_intercept(term))
        .collect::<Vec<_>>();
    let mut diagnostics = Vec::new();

    for i in 0..scalar_intercepts.len() {
        for j in (i + 1)..scalar_intercepts.len() {
            let left = scalar_intercepts[i];
            let right = scalar_intercepts[j];
            let left_factors = grouping_factor_names(&left.group);
            let right_factors = grouping_factor_names(&right.group);
            if !disjoint_factor_sets(&left_factors, &right_factors) {
                continue;
            }

            let left_group = grouping_factor_label(&left.group);
            let right_group = grouping_factor_label(&right.group);
            let mut diagnostic = Diagnostic::new(
                DiagnosticCode::CovarianceAssumption,
                DiagnosticSeverity::Info,
                DiagnosticStage::DesignAudit,
                format!(
                    "no correlation parameter is estimated between random-intercept groups \
                     '{left_group}' and '{right_group}'; separate scalar random-effect terms \
                     define independent covariance blocks"
                ),
            )
            .with_affected_terms(vec![
                left.source_syntax.user_text().to_string(),
                right.source_syntax.user_text().to_string(),
            ])
            .with_suggested_actions(vec![
                format!(
                    "Report standard deviations for '{left_group}' and '{right_group}' separately; their cross-block correlation is fixed absent by this parameterization."
                ),
                "A nonzero cross-block correlation would require an explicit coupled covariance structure, which is not represented by separate `(1 | group)` terms.".to_string(),
            ]);
            diagnostic
                .payload
                .insert("left_group".to_string(), serde_json::json!(left_group));
            diagnostic
                .payload
                .insert("right_group".to_string(), serde_json::json!(right_group));
            diagnostic.payload.insert(
                "correlation_parameter".to_string(),
                serde_json::json!("not_estimated"),
            );
            diagnostic.payload.insert(
                "covariance_blocks".to_string(),
                serde_json::json!("independent"),
            );
            diagnostics.push(diagnostic);
        }
    }

    diagnostics
}

fn is_scalar_random_intercept(term: &RandomTermIr) -> bool {
    matches!(term.covariance, CovarianceForm::Scalar)
        && term.basis.len() == 1
        && term.basis[0].kind == RandomCoefficientKind::Intercept
}

fn grouping_factor_names(group: &GroupingFactorIr) -> Vec<&str> {
    match group {
        GroupingFactorIr::Single { name } => vec![name.as_str()],
        GroupingFactorIr::Interaction { names } | GroupingFactorIr::Cell { names } => {
            names.iter().map(String::as_str).collect()
        }
    }
}

fn disjoint_factor_sets(left: &[&str], right: &[&str]) -> bool {
    left.iter()
        .all(|left_name| right.iter().all(|right_name| left_name != right_name))
}

fn requested_covariance_parameters(covariance: &CovarianceForm, basis_size: usize) -> usize {
    match covariance {
        CovarianceForm::Scalar => usize::from(basis_size > 0),
        CovarianceForm::Diagonal => basis_size,
        CovarianceForm::Full => basis_size * (basis_size + 1) / 2,
        CovarianceForm::Structured { kind } => structured_covariance_parameters(*kind, basis_size),
        CovarianceForm::ReducedRank { rank } => rank.unwrap_or(1).saturating_mul(basis_size),
        CovarianceForm::Unsupported { .. } => 0,
    }
}

fn structured_covariance_parameters(
    _kind: super::ir::StructuredCovarianceKind,
    basis_size: usize,
) -> usize {
    match basis_size {
        0 => 0,
        1 => 1,
        _ => 2,
    }
}

fn information_budget(
    group: &GroupingAudit,
    covariance: &CovarianceForm,
    basis_dimension: usize,
    requested_covariance_parameters: usize,
) -> RandomEffectInformationBudget {
    let min_levels_variance = min_levels_variance(basis_dimension);
    let min_levels_random_intercept_fit = min_levels_random_intercept_fit();
    let min_levels_random_intercept_reliability = min_levels_random_intercept_reliability();
    let min_levels_full_covariance = if matches!(covariance, CovarianceForm::Full) {
        Some(min_levels_full_covariance(requested_covariance_parameters))
    } else {
        None
    };
    let covariance_family = covariance_family_label(covariance);

    let (status, reason) = match group.n_levels {
        None => (
            InformationBudgetStatus::NotAssessable,
            Some("grouping levels are unavailable".to_string()),
        ),
        Some(_)
            if matches!(
                covariance,
                CovarianceForm::Structured { .. }
                    | CovarianceForm::ReducedRank { .. }
                    | CovarianceForm::Unsupported { .. }
            ) =>
        {
            (
                InformationBudgetStatus::NotAssessable,
                Some(
                    "covariance family is not covered by v0 information-budget thresholds"
                        .to_string(),
                ),
            )
        }
        Some(_) if basis_dimension == 0 => (
            InformationBudgetStatus::NotAssessable,
            Some("random-effect basis is empty".to_string()),
        ),
        Some(n_levels) => {
            if let Some((n_rows, n_random_effects)) =
                row_saturated_random_effect(group, basis_dimension)
            {
                (
                    InformationBudgetStatus::TooRich,
                    Some(format!(
                        "number of observations ({n_rows}) is <= random coefficients ({n_random_effects}) for grouping factor '{}' with basis dimension {basis_dimension}; random-effect variances and the residual scale are probably not separately identifiable",
                        group.name
                    )),
                )
            } else if is_scalar_random_intercept_budget(
                covariance,
                basis_dimension,
                requested_covariance_parameters,
            ) {
                if n_levels < min_levels_random_intercept_fit {
                    (
                        InformationBudgetStatus::TooRich,
                        Some(format!(
                            "{n_levels} levels are below the v0 random-intercept fit threshold {min_levels_random_intercept_fit}"
                        )),
                    )
                } else if n_levels < min_levels_random_intercept_reliability {
                    (
                        InformationBudgetStatus::WeaklySupported,
                        Some(format!(
                            "{n_levels} levels are fit-eligible for a scalar random intercept but below the v0 reliability threshold {min_levels_random_intercept_reliability}"
                        )),
                    )
                } else {
                    (InformationBudgetStatus::Sufficient, None)
                }
            } else if let Some(min_full) = min_levels_full_covariance {
                if n_levels < min_full {
                    (
                        InformationBudgetStatus::TooRich,
                        Some(format!(
                            "{n_levels} levels are below the v0 full-covariance threshold {min_full} for {requested_covariance_parameters} covariance parameters"
                        )),
                    )
                } else if n_levels < min_levels_variance {
                    (
                        InformationBudgetStatus::WeaklySupported,
                        Some(format!(
                            "{n_levels} levels are below the v0 variance-direction threshold {min_levels_variance}"
                        )),
                    )
                } else {
                    (InformationBudgetStatus::Sufficient, None)
                }
            } else if n_levels < min_levels_variance {
                (
                    InformationBudgetStatus::WeaklySupported,
                    Some(format!(
                        "{n_levels} levels are below the v0 variance-direction threshold {min_levels_variance}"
                    )),
                )
            } else {
                (InformationBudgetStatus::Sufficient, None)
            }
        }
    };
    let effective_n = effective_n_report(
        group,
        basis_dimension,
        requested_covariance_parameters,
        min_levels_variance,
        min_levels_full_covariance,
        status,
    );

    RandomEffectInformationBudget {
        n_levels: group.n_levels,
        basis_dimension,
        covariance_family,
        requested_covariance_parameters,
        min_levels_variance,
        min_levels_full_covariance,
        effective_n,
        status,
        reason,
    }
}

fn effective_n_report(
    group: &GroupingAudit,
    basis_dimension: usize,
    covariance_parameters: usize,
    min_levels_variance: usize,
    min_levels_full_covariance: Option<usize>,
    status: InformationBudgetStatus,
) -> RandomEffectEffectiveNReport {
    let levels_per_basis_direction = ratio_usize(
        group.n_levels,
        usize::from(basis_dimension > 0) * basis_dimension,
    );
    let levels_per_covariance_parameter = ratio_usize(group.n_levels, covariance_parameters);
    let rows_per_covariance_parameter = ratio_usize(group.n_observations, covariance_parameters);
    let total_rows_can_mislead = match (group.n_observations, group.n_levels) {
        (Some(rows), Some(levels)) => {
            rows > levels
                && matches!(
                    status,
                    InformationBudgetStatus::WeaklySupported | InformationBudgetStatus::TooRich
                )
        }
        _ => false,
    };

    RandomEffectEffectiveNReport {
        n_rows: group.n_observations,
        n_levels: group.n_levels,
        min_obs_per_level: group.min_obs_per_level,
        max_obs_per_level: group.max_obs_per_level,
        basis_dimension,
        covariance_parameters,
        levels_per_basis_direction,
        levels_per_covariance_parameter,
        rows_per_covariance_parameter,
        total_rows_can_mislead,
        explanation: effective_n_explanation(group, total_rows_can_mislead),
        recommendation: effective_n_recommendation(
            status,
            covariance_parameters,
            min_levels_variance,
            min_levels_full_covariance,
            row_saturated_random_effect(group, basis_dimension).is_some(),
        ),
    }
}

fn ratio_usize(numerator: Option<usize>, denominator: usize) -> Option<f64> {
    if denominator == 0 {
        None
    } else {
        numerator.map(|value| value as f64 / denominator as f64)
    }
}

fn effective_n_explanation(group: &GroupingAudit, total_rows_can_mislead: bool) -> String {
    match (group.n_observations, group.n_levels) {
        (Some(rows), Some(levels)) if total_rows_can_mislead => format!(
            "{rows} rows are clustered into {levels} grouping levels; covariance support is limited by grouping levels, not by total rows"
        ),
        (Some(rows), Some(levels)) => format!(
            "{rows} rows are clustered into {levels} grouping levels; grouping levels are the effective n for random-effect distribution support"
        ),
        _ => "grouping-level effective n is not assessable because grouping levels are unavailable"
            .to_string(),
    }
}

fn effective_n_recommendation(
    status: InformationBudgetStatus,
    covariance_parameters: usize,
    min_levels_variance: usize,
    min_levels_full_covariance: Option<usize>,
    row_saturated: bool,
) -> String {
    match status {
        InformationBudgetStatus::TooRich if row_saturated => {
            "random-effect coefficients saturate the rows for this term; options include dropping unsupported random slopes, splitting or simplifying the random-effect structure, treating the grouping factor as fixed, or collecting more observations per grouping level".to_string()
        }
        InformationBudgetStatus::Sufficient => {
            "information budget is sufficient under v0 thresholds".to_string()
        }
        InformationBudgetStatus::WeaklySupported if covariance_parameters == 1 => format!(
            "scalar random intercept is fit-eligible but low-reliability; precision is weak and boundary risk is elevated, and more than {} grouping levels would support routine inference",
            min_levels_random_intercept_reliability()
        ),
        InformationBudgetStatus::WeaklySupported => format!(
            "options include using design_compiled to simplify or withhold confirmatory inference, treating the grouping factor as fixed when it is designed or directly compared, or collecting at least {min_levels_variance} grouping levels for these variance directions"
        ),
        InformationBudgetStatus::TooRich => {
            if let Some(min_full) = min_levels_full_covariance {
                format!(
                    "full covariance asks for {covariance_parameters} parameter(s); options include using design_compiled with diagonal or reduced-rank covariance, treating the grouping factor as fixed when it is designed or directly compared, or collecting at least {min_full} grouping levels"
                )
            } else {
                "options include using design_compiled to simplify unsupported random-effect structure, treating the grouping factor as fixed when it is designed or directly compared, or collecting more grouping levels".to_string()
            }
        }
        InformationBudgetStatus::NotAssessable => {
            "fix grouping-factor/basis specification before fitting or mark inference unavailable"
                .to_string()
        }
    }
}

fn effective_covariance_form(
    covariance: &CovarianceForm,
    basis_dimension: usize,
) -> CovarianceForm {
    match covariance {
        CovarianceForm::Scalar if basis_dimension > 1 => CovarianceForm::Full,
        other => other.clone(),
    }
}

fn min_levels_variance(basis_dimension: usize) -> usize {
    5.max(2 * basis_dimension + 1)
}

fn min_levels_random_intercept_fit() -> usize {
    2
}

fn min_levels_random_intercept_reliability() -> usize {
    5
}

fn min_levels_full_covariance(n_covariance_parameters: usize) -> usize {
    10.max(5 * n_covariance_parameters)
}

fn is_scalar_random_intercept_budget(
    covariance: &CovarianceForm,
    basis_dimension: usize,
    requested_covariance_parameters: usize,
) -> bool {
    matches!(covariance, CovarianceForm::Scalar)
        && basis_dimension == 1
        && requested_covariance_parameters == 1
}

fn row_saturated_random_effect(
    group: &GroupingAudit,
    basis_dimension: usize,
) -> Option<(usize, usize)> {
    let n_rows = group.n_observations?;
    let n_levels = group.n_levels?;
    let n_random_effects = n_levels.checked_mul(basis_dimension)?;
    (basis_dimension > 0 && n_rows <= n_random_effects).then_some((n_rows, n_random_effects))
}

fn covariance_family_label(covariance: &CovarianceForm) -> String {
    match covariance {
        CovarianceForm::Scalar => "scalar".to_string(),
        CovarianceForm::Diagonal => "diagonal".to_string(),
        CovarianceForm::Full => "full".to_string(),
        CovarianceForm::Structured { kind } => format!("structured:{kind}"),
        CovarianceForm::ReducedRank { rank } => match rank {
            Some(rank) => format!("reduced_rank:{rank}"),
            None => "reduced_rank".to_string(),
        },
        CovarianceForm::Unsupported { reason } => format!("unsupported:{reason}"),
    }
}

fn information_budget_diagnostic(
    term: &RandomTermIr,
    budget: &RandomEffectInformationBudget,
) -> Diagnostic {
    let few_level_random_intercept = budget.status == InformationBudgetStatus::WeaklySupported
        && budget.basis_dimension == 1
        && budget.requested_covariance_parameters == 1
        && budget.min_levels_full_covariance.is_none();
    let severity = match budget.status {
        InformationBudgetStatus::TooRich => DiagnosticSeverity::Warning,
        InformationBudgetStatus::WeaklySupported if few_level_random_intercept => {
            DiagnosticSeverity::Warning
        }
        InformationBudgetStatus::WeaklySupported => DiagnosticSeverity::Info,
        InformationBudgetStatus::Sufficient | InformationBudgetStatus::NotAssessable => {
            DiagnosticSeverity::Info
        }
    };
    let mut diagnostic = Diagnostic::new(
        if few_level_random_intercept {
            DiagnosticCode::RandomEffectFewLevels
        } else {
            DiagnosticCode::CovarianceTooRich
        },
        severity,
        DiagnosticStage::DesignAudit,
        budget.reason.clone().unwrap_or_else(|| {
            "random-effect information budget is not sufficient for the requested covariance"
                .to_string()
        }),
    )
    .with_affected_terms(vec![term.source_syntax.text.clone()]);

    diagnostic
        .payload
        .insert("n_levels".to_string(), serde_json::json!(budget.n_levels));
    diagnostic.payload.insert(
        "basis_dimension".to_string(),
        serde_json::json!(budget.basis_dimension),
    );
    diagnostic.payload.insert(
        "requested_covariance_parameters".to_string(),
        serde_json::json!(budget.requested_covariance_parameters),
    );
    diagnostic.payload.insert(
        "n_random_effects".to_string(),
        serde_json::json!(budget
            .effective_n
            .n_levels
            .and_then(|levels| levels.checked_mul(budget.basis_dimension))),
    );
    diagnostic.payload.insert(
        "row_saturated".to_string(),
        serde_json::json!(matches!(
            (
                budget.effective_n.n_rows,
                budget
                    .effective_n
                    .n_levels
                    .and_then(|levels| levels.checked_mul(budget.basis_dimension)),
            ),
            (Some(rows), Some(n_random_effects)) if rows <= n_random_effects
        )),
    );
    diagnostic.payload.insert(
        "min_levels_variance".to_string(),
        serde_json::json!(budget.min_levels_variance),
    );
    diagnostic.payload.insert(
        "min_levels_random_intercept_fit".to_string(),
        serde_json::json!(min_levels_random_intercept_fit()),
    );
    diagnostic.payload.insert(
        "min_levels_random_intercept_reliability".to_string(),
        serde_json::json!(min_levels_random_intercept_reliability()),
    );
    diagnostic.payload.insert(
        "min_levels_full_covariance".to_string(),
        serde_json::json!(budget.min_levels_full_covariance),
    );
    diagnostic
}

fn support_note_diagnostic(
    term: &RandomTermIr,
    budget: &RandomEffectInformationBudget,
) -> Diagnostic {
    let mut diagnostic = Diagnostic::new(
        DiagnosticCode::SupportNote,
        DiagnosticSeverity::Info,
        DiagnosticStage::DesignAudit,
        "the requested covariance structure is information-hungry relative to the observed grouping levels",
    )
    .with_affected_terms(vec![term.id.clone()])
    .with_suggested_actions(vec![
        "The requested covariance structure is information-hungry relative to the observed grouping levels.".to_string(),
    ]);
    diagnostic
        .payload
        .insert("group".to_string(), serde_json::json!(term.group.label()));
    diagnostic.payload.insert(
        "covariance_family".to_string(),
        serde_json::json!(budget.covariance_family),
    );
    diagnostic.payload.insert(
        "requested_covariance_parameters".to_string(),
        serde_json::json!(budget.requested_covariance_parameters),
    );
    diagnostic
        .payload
        .insert("n_levels".to_string(), serde_json::json!(budget.n_levels));
    diagnostic.payload.insert(
        "policy_threshold".to_string(),
        serde_json::json!(support_note_policy_threshold(budget)),
    );
    diagnostic
}

fn support_note_policy_threshold(budget: &RandomEffectInformationBudget) -> usize {
    budget.min_levels_full_covariance.unwrap_or_else(|| {
        if budget.basis_dimension == 1
            && budget.requested_covariance_parameters == 1
            && budget.min_levels_full_covariance.is_none()
        {
            min_levels_random_intercept_reliability()
        } else {
            budget.min_levels_variance
        }
    })
}

fn structural_refusal_diagnostic(term: &RandomTermIr, basis: &BasisAudit) -> Diagnostic {
    let mut diagnostic = Diagnostic::new(
        DiagnosticCode::StructuralRefusal,
        DiagnosticSeverity::Info,
        DiagnosticStage::DesignAudit,
        format!(
            "`{}` does not vary within `{}`, so a `{}`-level `{}` slope cannot be estimated from this design",
            basis.name,
            term.group.label(),
            term.group.label(),
            basis.name
        ),
    )
    .with_affected_terms(vec![term.source_syntax.text.clone()])
    .with_suggested_actions(vec![format!(
        "`{}` does not vary within `{}`, so a `{}`-level `{}` slope cannot be estimated from this design.",
        basis.name,
        term.group.label(),
        term.group.label(),
        basis.name
    )]);
    diagnostic
        .payload
        .insert("group".to_string(), serde_json::json!(term.group.label()));
    diagnostic
        .payload
        .insert("slope".to_string(), serde_json::json!(basis.name));
    diagnostic.payload.insert(
        "reason".to_string(),
        serde_json::json!("slope_variable_does_not_vary_within_group"),
    );
    diagnostic
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::compile_formula_ir;
    use crate::formula::parse_formula;
    use crate::model::data::{CategoricalContrast, ContrastSource, DataFrame};

    fn repeated_subject_data() -> DataFrame {
        let mut data = DataFrame::new();
        data.add_numeric("y", vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        data.add_numeric("x", vec![0.0, 1.0, 0.0, 1.0]).unwrap();
        data.add_categorical(
            "subject",
            vec!["s1", "s1", "s2", "s2"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        )
        .unwrap();
        data.add_categorical(
            "item",
            vec!["i1", "i2", "i1", "i2"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        )
        .unwrap();
        data
    }

    fn many_subject_data(n_subjects: usize) -> DataFrame {
        many_subject_data_with_obs(n_subjects, 2)
    }

    fn many_subject_data_with_obs(n_subjects: usize, obs_per_subject: usize) -> DataFrame {
        let mut y = Vec::with_capacity(n_subjects * obs_per_subject);
        let mut x = Vec::with_capacity(n_subjects * obs_per_subject);
        let mut subject = Vec::with_capacity(n_subjects * obs_per_subject);
        for idx in 0..n_subjects {
            for obs in 0..obs_per_subject {
                y.push(idx as f64 + obs as f64);
                x.push(obs as f64);
                subject.push(format!("s{}", idx + 1));
            }
        }

        let mut data = DataFrame::new();
        data.add_numeric("y", y).unwrap();
        data.add_numeric("x", x).unwrap();
        data.add_categorical("subject", subject).unwrap();
        data
    }

    fn categorical_subject_data(n_subjects: usize) -> DataFrame {
        let levels = ["A", "B", "C"];
        let mut y = Vec::with_capacity(n_subjects * levels.len());
        let mut x = Vec::with_capacity(n_subjects * levels.len());
        let mut cond = Vec::with_capacity(n_subjects * levels.len());
        let mut subject = Vec::with_capacity(n_subjects * levels.len());
        for subject_index in 0..n_subjects {
            for (level_index, level) in levels.iter().enumerate() {
                y.push(subject_index as f64 + level_index as f64);
                x.push(level_index as f64 + 1.0);
                cond.push((*level).to_string());
                subject.push(format!("s{}", subject_index + 1));
            }
        }

        let mut data = DataFrame::new();
        data.add_numeric("y", y).unwrap();
        data.add_numeric("x", x).unwrap();
        data.add_categorical("cond", cond).unwrap();
        data.add_categorical("subject", subject).unwrap();
        data
    }

    fn categorical_subject_data_with_explicit_contrast() -> DataFrame {
        let mut data = DataFrame::new();
        data.add_numeric("y", vec![1.0, 2.0, 1.5, 2.5, 1.2, 2.2])
            .unwrap();
        data.add_categorical_with_contrast(
            "anchor",
            vec!["low", "high", "low", "high", "low", "high"]
                .into_iter()
                .map(str::to_string)
                .collect(),
            vec!["low".to_string(), "high".to_string()],
            CategoricalContrast::new(
                vec!["low".to_string(), "high".to_string()],
                DMatrix::from_row_slice(2, 1, &[0.5, -0.5]),
                vec!["hi_minus_lo".to_string()],
                false,
                ContrastSource::Custom,
            )
            .unwrap(),
        )
        .unwrap();
        data.add_categorical(
            "subject",
            vec!["s1", "s1", "s2", "s2", "s3", "s3"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        )
        .unwrap();
        data
    }

    fn repeated_condition_data() -> DataFrame {
        let mut data = DataFrame::new();
        data.add_numeric("y", vec![1.0, 2.0, 2.5, 3.5]).unwrap();
        data.add_categorical(
            "condition",
            vec!["A", "B", "A", "B"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        )
        .unwrap();
        data.add_categorical(
            "subject",
            vec!["s1", "s1", "s2", "s2"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        )
        .unwrap();
        data
    }

    fn repeated_crossed_data() -> DataFrame {
        let mut data = DataFrame::new();
        data.add_numeric("y", vec![1.0, 1.2, 2.0, 2.2, 1.5, 1.7, 2.5, 2.7])
            .unwrap();
        data.add_numeric("x", vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
            .unwrap();
        data.add_categorical(
            "subject",
            vec!["s1", "s1", "s1", "s1", "s2", "s2", "s2", "s2"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        )
        .unwrap();
        data.add_categorical(
            "item",
            vec!["i1", "i1", "i2", "i2", "i1", "i1", "i2", "i2"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        )
        .unwrap();
        data
    }

    fn assert_close(actual: Option<f64>, expected: f64) {
        let actual = actual.expect("expected ratio to be assessable");
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn optimizer_certificate_accepts_joint_radius_stop() {
        let params = vec![0.4, 0.8];
        let lower_bounds = vec![f64::NEG_INFINITY, 0.0];
        let mut optsum = OptSummary::new(params.clone());
        optsum.return_value = "JOINT_LAPLACE:RADIUS_REACHED".to_string();
        optsum.finitial = 23.4;
        optsum.fmin = 22.8;
        optsum.feval = 18;
        optsum.max_feval = 200;
        optsum.final_params = params.clone();
        optsum.final_trust_radius = Some(1.0e-5);

        let certificate = OptimizerCertificate::from_opt_summary_with_context(
            &optsum,
            &params,
            &lower_bounds,
            Some(64),
        );

        assert!(
            certificate.evidence.optimizer_stop.acceptable_stop,
            "TrustBQ radius convergence should be accepted after unwrapping the joint GLMM status"
        );
        assert_eq!(certificate.status, FitStatus::ConvergedInterior);
        assert!(!certificate.evidence.optimizer_stop.budget_exhausted);
        assert_eq!(
            certificate.evidence.optimizer_stop.return_code.as_deref(),
            Some("JOINT_LAPLACE:RADIUS_REACHED")
        );
    }

    #[test]
    fn optimizer_certificate_accepts_joint_ftol_at_budget_boundary() {
        let params = vec![0.4, 0.8];
        let lower_bounds = vec![f64::NEG_INFINITY, 0.0];
        let mut optsum = OptSummary::new(params.clone());
        optsum.return_value = "JOINT_LAPLACE:FTOL_REACHED".to_string();
        optsum.finitial = 23.4;
        optsum.fmin = 22.8;
        optsum.feval = 578;
        optsum.max_feval = 578;
        optsum.final_params = params.clone();

        let certificate = OptimizerCertificate::from_opt_summary_with_context(
            &optsum,
            &params,
            &lower_bounds,
            Some(64),
        );

        assert!(
            certificate.evidence.optimizer_stop.acceptable_stop,
            "a clean joint FTOL stop should not be downgraded just because feval equals max_feval"
        );
        assert_eq!(certificate.status, FitStatus::ConvergedInterior);
        assert!(!certificate.evidence.optimizer_stop.budget_exhausted);
        assert_eq!(
            certificate.evidence.optimizer_stop.return_code.as_deref(),
            Some("JOINT_LAPLACE:FTOL_REACHED")
        );
    }

    #[test]
    fn optimizer_certificate_rejects_nonstationary_ftol_stop() {
        let params = vec![0.23, 0.14];
        let lower_bounds = vec![0.0, 0.0];
        let mut optsum = OptSummary::new(params.clone());
        optsum.return_value = "FTOL_REACHED".to_string();
        optsum.finitial = 900.0;
        optsum.fmin = 818.44;
        optsum.feval = 43;
        optsum.final_params = params.clone();

        let mut certificate = OptimizerCertificate::from_opt_summary_with_context(
            &optsum,
            &params,
            &lower_bounds,
            Some(727),
        );
        assert_eq!(certificate.status, FitStatus::ConvergedInterior);

        certificate.apply_derivative_evidence(
            OptimizerDerivativeEvidence {
                method: EvidenceMethod::FiniteDifference,
                gradient: vec![17.7, -2.0],
                hessian: Some(DMatrix::identity(2, 2)),
            },
            1.0e-3,
            1.0e-5,
        );

        assert_eq!(certificate.status, FitStatus::NotOptimized);
        assert_eq!(certificate.free_gradient_norm, Some(17.7));
        assert!(certificate
            .checks
            .iter()
            .any(|check| matches!(check, CertificateCheck::DerivativeMismatch { kind, .. } if kind == "free_gradient_kkt_mismatch")));
        assert!(certificate
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == DiagnosticCode::OptimizerNonconvergence));
    }

    #[test]
    fn design_audit_reports_full_rank_fixed_effects() {
        let formula = parse_formula("y ~ x + (1 | subject)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let audit = audit_design(&semantic, &repeated_subject_data());

        assert_eq!(audit.fixed_effect_rank.status, RankStatus::FullRank);
        assert_eq!(audit.fixed_effect_rank.rank, Some(2));
        assert_eq!(audit.fixed_effects.n_columns, 2);
        assert_eq!(audit.fixed_effects.columns[0].name, "(Intercept)");
        assert_eq!(
            audit.fixed_effects.terms[1].status,
            FixedEffectTermStatus::Estimable
        );
    }

    #[test]
    fn design_audit_flags_rank_deficient_fixed_effects() {
        let mut data = repeated_subject_data();
        data.add_numeric("x2", vec![0.0, 2.0, 0.0, 2.0]).unwrap();
        let formula = parse_formula("y ~ x + x2 + (1 | subject)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let audit = audit_design(&semantic, &data);

        assert_eq!(audit.fixed_effect_rank.status, RankStatus::RankDeficient);
        assert_eq!(audit.fixed_effect_rank.rank, Some(2));
        assert_eq!(audit.fixed_effect_rank.expected, Some(3));
        assert!(!audit.fixed_effects.aliased_columns.is_empty());
        assert!(audit
            .diagnostics
            .iter()
            .any(|d| d.code == DiagnosticCode::FixedEffectRankDeficient));
    }

    #[test]
    fn design_audit_reports_empty_factor_cells() {
        let mut data = DataFrame::new();
        data.add_numeric("y", vec![1.0, 2.0, 3.0]).unwrap();
        data.add_categorical(
            "site",
            vec!["s1", "s1", "s2"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        )
        .unwrap();
        data.add_categorical(
            "season",
            vec!["pre", "post", "pre"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        )
        .unwrap();
        let formula = parse_formula("y ~ site * season").unwrap();
        let semantic = compile_formula_ir(&formula);
        let audit = audit_design(&semantic, &data);

        assert_eq!(audit.fixed_effects.empty_cells.len(), 1);
        assert_eq!(audit.fixed_effects.empty_cells[0].term, "site:season");
        assert_eq!(
            audit.fixed_effects.empty_cells[0].levels,
            vec!["s2".to_string(), "post".to_string()]
        );
        assert!(audit
            .diagnostics
            .iter()
            .any(|d| d.code == DiagnosticCode::FixedEffectEmptyCell));
    }

    #[test]
    fn design_audit_reports_missing_fixed_effect_column() {
        let formula = parse_formula("y ~ missing + (1 | subject)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let audit = audit_design(&semantic, &repeated_subject_data());

        assert_eq!(
            audit.fixed_effects.terms[1].status,
            FixedEffectTermStatus::NotEstimable
        );
        assert!(audit
            .diagnostics
            .iter()
            .any(|d| d.code == DiagnosticCode::FixedEffectColumnMissing));
    }

    #[test]
    fn design_audit_flags_zerocorr_factor_decorrelation() {
        let formula = parse_formula("y ~ x + (1 + cond + x || subject)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let audit = audit_design(&semantic, &categorical_subject_data(6));

        let diagnostic = audit
            .diagnostics
            .iter()
            .find(|d| {
                d.code == DiagnosticCode::CovarianceAssumption
                    && d.payload.get("reason") == Some(&serde_json::json!("double_bar_factor_term"))
            })
            .expect("factor inside || should get a decorrelation diagnostic");
        assert_eq!(diagnostic.severity, DiagnosticSeverity::Info);
        assert_eq!(diagnostic.stage, DiagnosticStage::DesignAudit);
        assert_eq!(
            diagnostic.payload.get("factor"),
            Some(&serde_json::json!("cond"))
        );
        assert_eq!(
            diagnostic.payload.get("n_levels"),
            Some(&serde_json::json!(3))
        );
        assert_eq!(
            diagnostic.payload.get("correlated_block_equivalent"),
            Some(&serde_json::json!("(0 + cond | subject)"))
        );
        // Exactly one: the numeric slope `x` in the same || block must not trigger.
        assert_eq!(
            audit
                .diagnostics
                .iter()
                .filter(|d| d.payload.get("reason")
                    == Some(&serde_json::json!("double_bar_factor_term")))
                .count(),
            1
        );
        // Attached to the owning random-term audit as well.
        assert!(audit.random_terms.iter().any(|term| term
            .diagnostics
            .iter()
            .any(
                |d| d.payload.get("reason") == Some(&serde_json::json!("double_bar_factor_term"))
            )));
    }

    #[test]
    fn design_audit_zerocorr_factor_decorrelation_is_silent_without_factor_or_zerocorr() {
        // Numeric-only || block: nothing to say.
        let formula = parse_formula("y ~ x + (1 + x || subject)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let audit = audit_design(&semantic, &categorical_subject_data(6));
        assert!(
            !audit
                .diagnostics
                .iter()
                .any(|d| d.payload.get("reason")
                    == Some(&serde_json::json!("double_bar_factor_term")))
        );

        // Correlated factor block: within-factor covariances are estimated.
        let formula = parse_formula("y ~ x + (1 + cond | subject)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let audit = audit_design(&semantic, &categorical_subject_data(6));
        assert!(
            !audit
                .diagnostics
                .iter()
                .any(|d| d.payload.get("reason")
                    == Some(&serde_json::json!("double_bar_factor_term")))
        );
    }

    #[test]
    fn design_audit_flags_fixed_random_intercept_redundancy() {
        let formula = parse_formula("y ~ subject + x + (1 | subject)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let audit = audit_design(&semantic, &many_subject_data(6));

        let diagnostic = audit
            .diagnostics
            .iter()
            .find(|d| d.code == DiagnosticCode::FixedRandomRedundant)
            .expect("fixed/random redundancy should be diagnosed");
        assert_eq!(diagnostic.severity, DiagnosticSeverity::Warning);
        assert_eq!(diagnostic.stage, DiagnosticStage::DesignAudit);
        assert_eq!(
            diagnostic.affected_terms,
            vec!["(1 | subject)".to_string(), "subject".to_string()]
        );
        assert!(audit.random_terms[0]
            .diagnostics
            .iter()
            .any(|d| d.code == DiagnosticCode::FixedRandomRedundant));
    }

    #[test]
    fn design_audit_reports_grouping_counts_and_slope_support() {
        let formula = parse_formula("y ~ x + (1 + x | subject)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let audit = audit_design(&semantic, &repeated_subject_data());

        let term = &audit.random_terms[0];
        assert_eq!(term.group.name, "subject");
        assert_eq!(term.group.n_observations, Some(4));
        assert_eq!(term.group.n_levels, Some(2));
        assert_eq!(term.group.min_obs_per_level, Some(2));
        assert_eq!(term.group.median_obs_per_level, Some(2));
        assert_eq!(term.group.max_obs_per_level, Some(2));
        assert_eq!(term.group.repeated, Some(true));
        assert_eq!(term.requested_covariance_parameters, 3);
        assert_eq!(term.basis[1].name, "x");
        assert_eq!(term.basis[1].supported, Some(true));
        assert_eq!(
            term.information_budget.status,
            InformationBudgetStatus::TooRich
        );
        assert_eq!(term.information_budget.min_levels_variance, 5);
        assert_eq!(term.information_budget.min_levels_full_covariance, Some(15));
        let effective_n = &term.information_budget.effective_n;
        assert_eq!(effective_n.n_rows, Some(4));
        assert_eq!(effective_n.n_levels, Some(2));
        assert_close(effective_n.levels_per_basis_direction, 1.0);
        assert_close(effective_n.levels_per_covariance_parameter, 2.0 / 3.0);
        assert_close(effective_n.rows_per_covariance_parameter, 4.0 / 3.0);
        assert!(effective_n.total_rows_can_mislead);
        assert!(effective_n.recommendation.contains("saturate"));
        assert!(audit
            .diagnostics
            .iter()
            .any(|d| d.code == DiagnosticCode::CovarianceTooRich));
    }

    #[test]
    fn design_audit_reports_median_grouping_count_for_unbalanced_groups() {
        let mut data = DataFrame::new();
        data.add_numeric("y", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
            .unwrap();
        data.add_numeric("x", vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
            .unwrap();
        data.add_categorical(
            "subject",
            vec!["s1", "s2", "s2", "s3", "s3", "s3", "s3"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        )
        .unwrap();

        let formula = parse_formula("y ~ x + (1 | subject)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let audit = audit_design(&semantic, &data);

        let group = &audit.random_terms[0].group;
        assert_eq!(group.min_obs_per_level, Some(1));
        assert_eq!(group.median_obs_per_level, Some(2));
        assert_eq!(group.max_obs_per_level, Some(4));
    }

    #[test]
    fn design_audit_reports_sufficient_random_effect_information_budget() {
        let formula = parse_formula("y ~ x + (1 + x | subject)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let audit = audit_design(&semantic, &many_subject_data_with_obs(18, 3));

        let budget = &audit.random_terms[0].information_budget;
        assert_eq!(budget.n_levels, Some(18));
        assert_eq!(budget.basis_dimension, 2);
        assert_eq!(budget.requested_covariance_parameters, 3);
        assert_eq!(budget.min_levels_variance, 5);
        assert_eq!(budget.min_levels_full_covariance, Some(15));
        assert_eq!(budget.status, InformationBudgetStatus::Sufficient);
        assert_eq!(budget.effective_n.n_rows, Some(54));
        assert_eq!(budget.effective_n.n_levels, Some(18));
        assert_close(budget.effective_n.levels_per_basis_direction, 9.0);
        assert_close(budget.effective_n.levels_per_covariance_parameter, 6.0);
        assert_close(budget.effective_n.rows_per_covariance_parameter, 18.0);
        assert!(!budget.effective_n.total_rows_can_mislead);
        assert!(budget.effective_n.recommendation.contains("sufficient"));
    }

    #[test]
    fn design_audit_flags_structured_covariance_as_parsed_refused() {
        let formula = parse_formula("y ~ x + cs(1 + x | subject)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let audit = audit_design(&semantic, &many_subject_data(12));

        let diagnostic = audit
            .diagnostics
            .iter()
            .find(|diagnostic| {
                diagnostic.code == DiagnosticCode::Unsupported
                    && diagnostic
                        .payload
                        .get("covariance_family")
                        .is_some_and(|value| value == "compound_symmetry")
            })
            .expect("structured covariance refusal diagnostic");
        assert_eq!(diagnostic.severity, DiagnosticSeverity::Error);
        assert_eq!(diagnostic.stage, DiagnosticStage::DesignAudit);
        assert_eq!(
            diagnostic.payload.get("support_status"),
            Some(&serde_json::json!("parsed_refused"))
        );
        assert_eq!(
            audit.covariance_kernels.kernels[0].support_status,
            CovarianceSupportStatus::ParsedRefused
        );
        assert_eq!(
            audit.covariance_kernels.kernels[0].expected_parameter_count,
            2
        );
    }

    #[test]
    fn design_audit_flags_row_saturated_random_effect_term() {
        let formula = parse_formula("y ~ x + (1 + x | subject)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let audit = audit_design(&semantic, &many_subject_data(100));

        let term = &audit.random_terms[0];
        let budget = &term.information_budget;
        assert_eq!(budget.n_levels, Some(100));
        assert_eq!(budget.basis_dimension, 2);
        assert_eq!(budget.status, InformationBudgetStatus::TooRich);
        assert!(budget
            .reason
            .as_deref()
            .unwrap()
            .contains("number of observations (200) is <= random coefficients (200)"));
        assert!(budget.effective_n.recommendation.contains("saturate"));

        let diagnostic = audit
            .diagnostics
            .iter()
            .find(|d| d.code == DiagnosticCode::CovarianceTooRich)
            .expect("row-saturated random term should be diagnosed");
        assert_eq!(
            diagnostic.payload.get("n_random_effects"),
            Some(&serde_json::json!(200))
        );
        assert_eq!(
            diagnostic.payload.get("row_saturated"),
            Some(&serde_json::json!(true))
        );
    }

    #[test]
    fn design_audit_expands_treatment_coded_categorical_random_basis() {
        let formula = parse_formula("y ~ cond + (1 + cond | subject)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let audit = audit_design(&semantic, &categorical_subject_data(40));

        let term = &audit.random_terms[0];
        assert_eq!(term.basis_size, 3);
        assert_eq!(
            term.basis
                .iter()
                .map(|basis| basis.name.as_str())
                .collect::<Vec<_>>(),
            vec!["intercept", "cond: B", "cond: C"]
        );
        assert_eq!(
            term.basis
                .iter()
                .map(|basis| basis.kind.as_str())
                .collect::<Vec<_>>(),
            vec!["intercept", "categorical_dummy", "categorical_dummy"]
        );
        assert_eq!(term.requested_covariance_parameters, 6);
        assert_eq!(term.information_budget.basis_dimension, 3);
        assert_eq!(term.information_budget.covariance_family, "full");

        let diagnostic = term
            .diagnostics
            .iter()
            .find(|d| d.code == DiagnosticCode::FormulaCanonicalized)
            .expect("basis expansion diagnostic should be attached to term");
        assert_eq!(
            diagnostic.payload.get("semantic_basis"),
            Some(&serde_json::json!(["intercept", "cond"]))
        );
        assert_eq!(
            diagnostic.payload.get("expanded_basis"),
            Some(&serde_json::json!(["intercept", "cond: B", "cond: C"]))
        );
    }

    #[test]
    fn design_audit_expands_cell_means_categorical_random_basis() {
        let formula = parse_formula("y ~ cond + (0 + cond | subject)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let audit = audit_design(&semantic, &categorical_subject_data(40));

        let term = &audit.random_terms[0];
        assert_eq!(term.basis_size, 3);
        assert_eq!(
            term.basis
                .iter()
                .map(|basis| basis.name.as_str())
                .collect::<Vec<_>>(),
            vec!["cond: A", "cond: B", "cond: C"]
        );
        assert!(term
            .basis
            .iter()
            .all(|basis| basis.kind.as_str() == "categorical_cell"));
        assert_eq!(term.requested_covariance_parameters, 6);
        assert_eq!(term.information_budget.covariance_family, "full");

        let diagnostic = term
            .diagnostics
            .iter()
            .find(|d| d.code == DiagnosticCode::FormulaCanonicalized)
            .expect("cell-means expansion diagnostic should be attached to term");
        assert_eq!(
            diagnostic.payload.get("semantic_basis"),
            Some(&serde_json::json!(["cond"]))
        );
        assert_eq!(
            diagnostic.payload.get("expanded_basis"),
            Some(&serde_json::json!(["cond: A", "cond: B", "cond: C"]))
        );
    }

    #[test]
    fn design_audit_explains_no_intercept_factor_bypasses_explicit_contrast() {
        let formula = parse_formula("y ~ anchor + (0 + anchor | subject)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let audit = audit_design(
            &semantic,
            &categorical_subject_data_with_explicit_contrast(),
        );

        let term = &audit.random_terms[0];
        assert_eq!(term.basis_size, 2);
        assert_eq!(
            term.basis
                .iter()
                .map(|basis| basis.name.as_str())
                .collect::<Vec<_>>(),
            vec!["anchor: low", "anchor: high"]
        );
        assert!(term
            .basis
            .iter()
            .all(|basis| basis.kind.as_str() == "categorical_cell"));

        let diagnostic = term
            .diagnostics
            .iter()
            .find(|d| d.code == DiagnosticCode::FormulaCanonicalized)
            .expect("cell-means expansion diagnostic should be attached to term");
        assert!(diagnostic
            .message
            .contains("uses cell-means coding by no-intercept categorical formula semantics"));
        assert!(diagnostic
            .message
            .contains("supplied contrast basis for anchor was not used"));
        assert_eq!(
            diagnostic.payload.get("basis_coding"),
            Some(&serde_json::json!("cell_means"))
        );
        assert_eq!(
            diagnostic.payload.get("contrast_available"),
            Some(&serde_json::json!(true))
        );
        assert_eq!(
            diagnostic.payload.get("contrast_used"),
            Some(&serde_json::json!(false))
        );
        assert_eq!(
            diagnostic.payload.get("contrast_variables"),
            Some(&serde_json::json!(["anchor"]))
        );
        assert_eq!(
            diagnostic.payload.get("reason"),
            Some(&serde_json::json!(
                "no_intercept_categorical_formula_semantics"
            ))
        );
    }

    #[test]
    fn design_audit_expands_cell_means_interaction_random_basis() {
        let formula = parse_formula("y ~ x * cond + (0 + x:cond | subject)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let audit = audit_design(&semantic, &categorical_subject_data(40));

        let term = &audit.random_terms[0];
        assert_eq!(term.basis_size, 3);
        assert_eq!(
            term.basis
                .iter()
                .map(|basis| basis.name.as_str())
                .collect::<Vec<_>>(),
            vec!["x:cond: A", "x:cond: B", "x:cond: C"]
        );
        assert!(term
            .basis
            .iter()
            .all(|basis| basis.kind.as_str() == "interaction"));
        assert_eq!(term.requested_covariance_parameters, 6);
        assert!(term
            .diagnostics
            .iter()
            .any(|d| d.code == DiagnosticCode::FormulaCanonicalized));
    }

    #[test]
    fn design_audit_reports_weak_scalar_information_budget() {
        let formula = parse_formula("y ~ x + (1 | subject)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let audit = audit_design(&semantic, &repeated_subject_data());

        let budget = &audit.random_terms[0].information_budget;
        assert_eq!(budget.n_levels, Some(2));
        assert_eq!(budget.basis_dimension, 1);
        assert_eq!(budget.requested_covariance_parameters, 1);
        assert_eq!(budget.min_levels_variance, 5);
        assert_eq!(budget.min_levels_full_covariance, None);
        assert_eq!(budget.status, InformationBudgetStatus::WeaklySupported);
        assert_eq!(budget.effective_n.n_rows, Some(4));
        assert_close(budget.effective_n.levels_per_covariance_parameter, 2.0);
        assert_close(budget.effective_n.rows_per_covariance_parameter, 4.0);
        assert!(budget.effective_n.total_rows_can_mislead);
        assert!(budget
            .effective_n
            .recommendation
            .contains("fit-eligible but low-reliability"));
        let diagnostic = audit
            .diagnostics
            .iter()
            .find(|d| d.code == DiagnosticCode::RandomEffectFewLevels)
            .expect("few-level scalar random intercept should be diagnosed");
        assert_eq!(diagnostic.severity, DiagnosticSeverity::Warning);
        assert_eq!(
            diagnostic.payload.get("min_levels_random_intercept_fit"),
            Some(&serde_json::json!(2))
        );
        assert_eq!(
            diagnostic
                .payload
                .get("min_levels_random_intercept_reliability"),
            Some(&serde_json::json!(5))
        );

        let support_note = audit.random_terms[0]
            .diagnostics
            .iter()
            .find(|d| d.code == DiagnosticCode::SupportNote)
            .expect("weakly supported random term should emit a support note");
        assert_eq!(support_note.severity, DiagnosticSeverity::Info);
        assert_eq!(support_note.stage, DiagnosticStage::DesignAudit);
        assert_eq!(
            support_note.payload.get("group"),
            Some(&serde_json::json!("subject"))
        );
        assert_eq!(
            support_note.payload.get("covariance_family"),
            Some(&serde_json::json!("scalar"))
        );
        assert_eq!(
            support_note.payload.get("policy_threshold"),
            Some(&serde_json::json!(5))
        );
    }

    #[test]
    fn design_audit_emits_scope_note_for_unmodeled_possible_slope() {
        let formula = parse_formula("y ~ x + (1 | subject)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let audit = audit_design(&semantic, &repeated_subject_data());

        let diagnostic = audit.random_terms[0]
            .diagnostics
            .iter()
            .find(|d| d.code == DiagnosticCode::ScopeNote)
            .expect("within-group fixed-effect variation should emit a scope note");
        assert_eq!(diagnostic.severity, DiagnosticSeverity::Info);
        assert_eq!(diagnostic.affected_terms, vec!["r0".to_string()]);
        assert_eq!(
            diagnostic.payload.get("group"),
            Some(&serde_json::json!("subject"))
        );
        assert_eq!(
            diagnostic.payload.get("fixed_effect"),
            Some(&serde_json::json!("x"))
        );
        assert_eq!(
            diagnostic.payload.get("varies_within_group"),
            Some(&serde_json::json!(true))
        );
    }

    #[test]
    fn design_audit_suppresses_scope_note_when_slope_is_in_split_block() {
        let formula = parse_formula("y ~ x + (1 | subject) + (0 + x | subject)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let audit = audit_design(&semantic, &repeated_subject_data());

        assert!(!audit
            .diagnostics
            .iter()
            .any(|d| d.code == DiagnosticCode::ScopeNote));
    }

    #[test]
    fn design_audit_reports_split_double_bar_scalar_budgets() {
        let formula = parse_formula("y ~ x + (1 + x || subject)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let audit = audit_design(&semantic, &many_subject_data(6));

        assert_eq!(audit.random_terms.len(), 2);
        for term in &audit.random_terms {
            let budget = &term.information_budget;
            assert_eq!(budget.covariance_family, "scalar");
            assert_eq!(budget.requested_covariance_parameters, 1);
            assert_eq!(budget.min_levels_full_covariance, None);
            assert_eq!(budget.status, InformationBudgetStatus::Sufficient);
        }
    }

    #[test]
    fn design_audit_reports_cell_grouping_counts() {
        let formula = parse_formula("y ~ x + (1 | subject:item)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let audit = audit_design(&semantic, &repeated_subject_data());

        let term = &audit.random_terms[0];
        assert_eq!(term.group.name, "subject:item");
        assert_eq!(term.group.n_observations, Some(4));
        assert_eq!(term.group.n_levels, Some(4));
        assert_eq!(term.group.min_obs_per_level, Some(1));
        assert_eq!(term.group.median_obs_per_level, Some(1));
        assert_eq!(term.group.max_obs_per_level, Some(1));
        assert_eq!(term.group.repeated, Some(false));
    }

    #[test]
    fn design_audit_flags_repeated_unit_without_random_intercept() {
        let formula = parse_formula("y ~ condition").unwrap();
        let semantic = compile_formula_ir(&formula);
        let audit = audit_design(&semantic, &repeated_condition_data());

        let missing = &audit.covariance_kernels.missing_dependence_paths;
        assert_eq!(missing.len(), 1);
        assert_eq!(missing[0].unit, "subject");
        assert_eq!(missing[0].path, DependencePathKind::Marginal);
        assert_eq!(missing[0].suggested_random_term, "(1 | subject)");

        let diagnostic = audit
            .diagnostics
            .iter()
            .find(|d| d.code == DiagnosticCode::RepeatedUnitUnmodeled)
            .expect("repeated subject should be diagnosed as unmodeled");
        assert_eq!(diagnostic.severity, DiagnosticSeverity::Warning);
        assert_eq!(
            diagnostic.payload.get("suggested_random_term"),
            Some(&serde_json::json!("(1 | subject)"))
        );
    }

    #[test]
    fn cell_kernel_does_not_cover_crossed_marginal_units() {
        let formula = parse_formula("y ~ x + (1 | subject:item)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let audit = audit_design(&semantic, &repeated_crossed_data());

        let missing_units = audit
            .covariance_kernels
            .missing_dependence_paths
            .iter()
            .map(|missing| missing.unit.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            missing_units,
            ["item", "subject"]
                .into_iter()
                .collect::<std::collections::BTreeSet<_>>()
        );

        let cell = audit
            .covariance_kernels
            .repeated_units
            .iter()
            .find(|unit| unit.unit == "subject:item")
            .expect("cell path should be audited separately");
        assert_eq!(cell.path, DependencePathKind::Cell);
        assert_eq!(cell.covered_by_terms, vec!["r0".to_string()]);
    }

    #[test]
    fn crossed_expansion_covers_marginal_and_cell_paths() {
        let formula = parse_formula("y ~ x + (1 | subject*item)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let audit = audit_design(&semantic, &repeated_crossed_data());

        assert!(audit.covariance_kernels.missing_dependence_paths.is_empty());
        assert!(!audit
            .diagnostics
            .iter()
            .any(|d| d.code == DiagnosticCode::RepeatedUnitUnmodeled));
    }

    #[test]
    fn design_audit_flags_slope_without_within_group_variation() {
        let mut data = DataFrame::new();
        data.add_numeric("y", vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        data.add_numeric("x", vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        data.add_categorical(
            "subject",
            vec!["s1", "s1", "s2", "s2"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        )
        .unwrap();
        let formula = parse_formula("y ~ x + (1 + x | subject)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let audit = audit_design(&semantic, &data);

        let term = &audit.random_terms[0];
        assert_eq!(term.basis[1].supported, Some(false));
        assert!(term
            .diagnostics
            .iter()
            .any(|d| d.code == DiagnosticCode::RandomSlopeUnsupported));
        let unsupported = term
            .diagnostics
            .iter()
            .find(|d| d.code == DiagnosticCode::RandomSlopeUnsupported)
            .expect("random slope unsupported diagnostic should remain");
        let structural = term
            .diagnostics
            .iter()
            .find(|d| d.code == DiagnosticCode::StructuralRefusal)
            .expect("structural refusal should accompany unsupported slope");
        assert_eq!(structural.severity, DiagnosticSeverity::Info);
        assert_eq!(structural.affected_terms, unsupported.affected_terms);
        assert_eq!(
            structural.payload.get("group"),
            Some(&serde_json::json!("subject"))
        );
        assert_eq!(
            structural.payload.get("slope"),
            Some(&serde_json::json!("x"))
        );
        assert_eq!(
            structural.payload.get("reason"),
            Some(&serde_json::json!(
                "slope_variable_does_not_vary_within_group"
            ))
        );
    }

    #[test]
    fn design_audit_round_trips_json() {
        let formula = parse_formula("y ~ x + (1 + x | subject)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let audit = audit_design(&semantic, &repeated_subject_data());

        let json = serde_json::to_string(&audit).unwrap();
        let decoded: DesignAudit = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, audit);
    }

    #[test]
    fn optimizer_certificate_round_trips_json() {
        let certificate = OptimizerCertificate::not_assessed();

        let json = serde_json::to_string(&certificate).unwrap();
        let decoded: OptimizerCertificate = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded, certificate);
    }
}

/// Optimizer certificate shape. v0 can populate this from existing optimizer
/// summaries while leaving unavailable checks explicit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimizerCertificate {
    pub status: FitStatus,
    pub optimizer_name: Option<String>,
    pub objective_value: Option<f64>,
    pub iterations: Option<usize>,
    pub evidence: ConvergenceEvidence,
    #[serde(default, skip_serializing_if = "OptimizerControlEvidence::is_default")]
    pub optimizer_control: OptimizerControlEvidence,
    pub verification: Option<ConvergenceVerification>,
    pub free_gradient_norm: Option<f64>,
    pub projected_gradient_norm: Option<f64>,
    pub hessian_eigen_min: Option<f64>,
    pub hessian_rank: Option<usize>,
    pub information_rank: Option<usize>,
    pub checks: Vec<CertificateCheck>,
    pub diagnostics: Vec<Diagnostic>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConvergenceEvidence {
    pub optimizer_stop: OptimizerStopEvidence,
    pub parameter_space: ParameterSpaceEvidence,
    pub sample_size: SampleSizeContext,
    pub gradient: GradientEvidence,
    pub hessian: HessianEvidence,
    pub certification_quality: EvidenceQuality,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimizerStopEvidence {
    pub return_code: Option<String>,
    pub acceptable_stop: bool,
    pub budget_exhausted: bool,
    pub function_evaluations: Option<usize>,
    pub max_function_evaluations: Option<usize>,
    pub max_time_seconds: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_trust_radius: Option<f64>,
    pub initial_objective: Option<f64>,
    pub final_objective: Option<f64>,
    pub objective_delta: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptimizerControlEvidence {
    pub optimizer_source: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub caller_set_fields: Vec<String>,
}

impl Default for OptimizerControlEvidence {
    fn default() -> Self {
        Self {
            optimizer_source: "auto".to_string(),
            caller_set_fields: Vec::new(),
        }
    }
}

impl OptimizerControlEvidence {
    pub fn is_default(&self) -> bool {
        self.optimizer_source == "auto" && self.caller_set_fields.is_empty()
    }

    fn from_opt_summary(optsum: &OptSummary) -> Self {
        Self {
            optimizer_source: optsum.optimizer_source_name().to_string(),
            caller_set_fields: optsum.caller_set_fields.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParameterSpaceEvidence {
    pub n_theta: usize,
    pub n_free: usize,
    pub n_boundary: usize,
    pub boundary_indices: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SampleSizeContext {
    pub n_observations: Option<usize>,
    pub n_theta: usize,
    pub observations_per_theta: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GradientEvidence {
    pub method: EvidenceMethod,
    pub raw_gradient_norm: Option<f64>,
    pub scaled_gradient_norm: Option<f64>,
    pub free_gradient_norm: Option<f64>,
    pub projected_gradient_norm: Option<f64>,
    pub kkt_boundary_gradient_max: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HessianEvidence {
    pub method: EvidenceMethod,
    pub quality: EvidenceQuality,
    pub min_eigenvalue: Option<f64>,
    pub condition_number: Option<f64>,
    pub rank: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OptimizerDerivativeEvidence {
    pub method: EvidenceMethod,
    pub gradient: Vec<f64>,
    pub hessian: Option<DMatrix<f64>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceMethod {
    Exact,
    FiniteDifference,
    OptimizerReported,
    NotAvailable { reason: String },
    NotAssessed { reason: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceQuality {
    Certified,
    Approximate { reason: String },
    Unavailable { reason: String },
    NotAssessed { reason: String },
    Failed { reason: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConvergenceVerification {
    pub status: ConvergenceVerificationStatus,
    pub objective_tolerance: f64,
    pub theta_tolerance: f64,
    pub beta_tolerance: f64,
    pub reference_objective: Option<f64>,
    pub reference_theta: Vec<f64>,
    pub reference_beta: Vec<f64>,
    pub reference_effective_ranks: Vec<usize>,
    pub runs: Vec<ConvergenceVerificationRun>,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConvergenceVerificationStatus {
    NotRun,
    RestartAgrees,
    OptimizerConsensus,
    Fragile,
    Unstable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConvergenceVerificationRun {
    pub label: String,
    pub optimizer_name: Option<String>,
    pub return_code: Option<String>,
    pub objective_value: Option<f64>,
    pub objective_delta: Option<f64>,
    pub max_abs_theta_delta: Option<f64>,
    pub max_abs_beta_delta: Option<f64>,
    pub effective_ranks: Vec<usize>,
    pub agrees: bool,
    pub diagnostics: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CertificateCheck {
    FreeGradientOk {
        tolerance: f64,
        value: f64,
    },
    BoundaryGradientOk {
        tolerance: f64,
        value: f64,
    },
    HessianPsdOnActiveSubspace {
        min_eigenvalue: f64,
    },
    RankOk {
        rank: usize,
        expected: usize,
    },
    NotAssessed {
        reason: String,
    },
    DerivativeMismatch {
        kind: String,
        observed: Option<f64>,
        tolerance: Option<f64>,
        regime: String,
        message: String,
    },
    Failed {
        code: String,
        message: String,
    },
}

impl OptimizerCertificate {
    pub fn not_assessed() -> Self {
        Self {
            status: FitStatus::NotAssessed,
            optimizer_name: None,
            objective_value: None,
            iterations: None,
            evidence: ConvergenceEvidence::not_assessed(),
            optimizer_control: OptimizerControlEvidence::default(),
            verification: None,
            free_gradient_norm: None,
            projected_gradient_norm: None,
            hessian_eigen_min: None,
            hessian_rank: None,
            information_rank: None,
            checks: vec![CertificateCheck::NotAssessed {
                reason: "optimizer certificate not run".to_string(),
            }],
            diagnostics: Vec::new(),
        }
    }

    pub fn from_opt_summary(optsum: &OptSummary, theta: &[f64], lower_bounds: &[f64]) -> Self {
        Self::from_opt_summary_with_context(optsum, theta, lower_bounds, None)
    }

    pub fn from_opt_summary_with_context(
        optsum: &OptSummary,
        theta: &[f64],
        lower_bounds: &[f64],
        n_observations: Option<usize>,
    ) -> Self {
        let optimizer_name = Some(optsum.optimizer_name().to_string());
        let objective_finite = optsum.fmin.is_finite();
        let objective_value = objective_finite.then_some(optsum.fmin);
        let iterations = usize::try_from(optsum.feval).ok();
        let boundary_indices = boundary_parameter_indices(optsum, theta, lower_bounds);
        let optimizer_stop_ok = optimizer_stop_is_acceptable(&optsum.return_value)
            && !optimizer_budget_exhausted(optsum);
        let optimizer_ok = optimizer_stop_ok && objective_finite;
        let evidence = ConvergenceEvidence::from_opt_summary(
            optsum,
            theta,
            n_observations,
            &boundary_indices,
            optimizer_ok,
        );

        if !optsum.is_fitted() {
            return Self {
                status: FitStatus::NotOptimized,
                optimizer_name,
                objective_value,
                iterations,
                evidence,
                optimizer_control: OptimizerControlEvidence::from_opt_summary(optsum),
                verification: None,
                free_gradient_norm: None,
                projected_gradient_norm: None,
                hessian_eigen_min: None,
                hessian_rank: None,
                information_rank: None,
                checks: vec![CertificateCheck::NotAssessed {
                    reason: "model has not been optimized".to_string(),
                }],
                diagnostics: vec![Diagnostic::new(
                    DiagnosticCode::OptimizerNotAssessed,
                    DiagnosticSeverity::Info,
                    DiagnosticStage::Certification,
                    "optimizer certificate is unavailable before fitting",
                )
                .with_suggested_actions(vec![
                    "fit the model before reading convergence evidence".to_string(),
                    "verify convergence after fitting if optimizer agreement matters (verify_convergence, where the host exposes it)"
                        .to_string(),
                ])],
            };
        }

        let mut checks = Vec::new();
        let mut diagnostics = Vec::new();

        if optimizer_ok {
            checks.push(CertificateCheck::NotAssessed {
                reason: "free-gradient KKT check requires derivative support".to_string(),
            });
            checks.push(CertificateCheck::NotAssessed {
                reason: "projected boundary-gradient KKT check requires derivative support"
                    .to_string(),
            });
            checks.push(CertificateCheck::NotAssessed {
                reason: "active-subspace Hessian check requires derivative support".to_string(),
            });
            if let Some(reason) = optimizer_recovery_reason(&optsum.return_value) {
                let mut diagnostic = Diagnostic::new(
                    DiagnosticCode::OptimizerRecovery,
                    DiagnosticSeverity::Info,
                    DiagnosticStage::Certification,
                    format!("optimizer recovered after covariance KKT-guided restart ({reason})"),
                )
                .with_suggested_actions(vec![
                    "treat this as recovered convergence, not first-pass clean convergence"
                        .to_string(),
                    "inspect the optimizer trace and covariance certificate if the boundary direction is scientifically central"
                        .to_string(),
                ]);
                diagnostic.payload.insert(
                    "return_code".to_string(),
                    serde_json::json!(optsum.return_value),
                );
                diagnostic
                    .payload
                    .insert("recovery_reason".to_string(), serde_json::json!(reason));
                diagnostic.payload.insert(
                    "function_evaluations".to_string(),
                    serde_json::json!(optsum.feval.max(0) as usize),
                );
                if let Some(radius) = optsum.final_trust_radius {
                    diagnostic
                        .payload
                        .insert("final_trust_radius".to_string(), serde_json::json!(radius));
                }
                diagnostics.push(diagnostic);
            }
        } else if !objective_finite {
            checks.push(CertificateCheck::Failed {
                code: "non_finite_objective".to_string(),
                message:
                    "optimizer did not certify convergence because the final objective is non-finite"
                        .to_string(),
            });
            let mut diagnostic = Diagnostic::new(
                DiagnosticCode::OptimizerNonconvergence,
                DiagnosticSeverity::Warning,
                DiagnosticStage::Certification,
                "optimizer reported a stop but the final objective is non-finite; convergence is not certified",
            )
            .with_suggested_actions(vec![
                "treat this fit as not optimized even if the optimizer return code looks acceptable"
                    .to_string(),
                "scale predictors or the response and compare against the original parameterization"
                    .to_string(),
                "try an alternate optimizer and compare the objective and theta".to_string(),
            ]);
            diagnostic.payload.insert(
                "return_code".to_string(),
                serde_json::json!(optsum.return_value),
            );
            diagnostic
                .payload
                .insert("objective_finite".to_string(), serde_json::json!(false));
            diagnostic.payload.insert(
                "objective_value".to_string(),
                serde_json::json!(format!("{}", optsum.fmin)),
            );
            diagnostic.payload.insert(
                "budget_exhausted".to_string(),
                serde_json::json!(optimizer_budget_exhausted(optsum)),
            );
            diagnostic.payload.insert(
                "function_evaluations".to_string(),
                serde_json::json!(optsum.feval.max(0) as usize),
            );
            if optsum.max_feval > 0 {
                diagnostic.payload.insert(
                    "max_function_evaluations".to_string(),
                    serde_json::json!(optsum.max_feval as usize),
                );
            }
            diagnostics.push(diagnostic);
        } else {
            checks.push(CertificateCheck::Failed {
                code: optsum.return_value.clone(),
                message: "optimizer did not report an acceptable convergence stop".to_string(),
            });
            let mut diagnostic = Diagnostic::new(
                DiagnosticCode::OptimizerNonconvergence,
                DiagnosticSeverity::Warning,
                DiagnosticStage::Certification,
                format!(
                    "optimizer stopped before an acceptable convergence criterion with return code '{}'",
                    optsum.return_value
                ),
            )
            .with_suggested_actions(vec![
                "increase the optimizer function-evaluation or time budget".to_string(),
                "try an alternate optimizer and compare the objective and theta".to_string(),
                "scale predictors or the response if finite-difference gradients look unstable"
                    .to_string(),
            ]);
            diagnostic.payload.insert(
                "return_code".to_string(),
                serde_json::json!(optsum.return_value),
            );
            diagnostic.payload.insert(
                "budget_exhausted".to_string(),
                serde_json::json!(optimizer_budget_exhausted(optsum)),
            );
            diagnostic.payload.insert(
                "function_evaluations".to_string(),
                serde_json::json!(optsum.feval.max(0) as usize),
            );
            if optsum.max_feval > 0 {
                diagnostic.payload.insert(
                    "max_function_evaluations".to_string(),
                    serde_json::json!(optsum.max_feval as usize),
                );
            }
            diagnostics.push(diagnostic);
        }

        for &index in &boundary_indices {
            let lower = lower_bounds.get(index).copied().unwrap_or(f64::NAN);
            let value = theta.get(index).copied().unwrap_or(f64::NAN);
            let parameter_label = format!("covariance parameter {}", index + 1);
            let mut diagnostic = Diagnostic::new(
                DiagnosticCode::BoundaryParameter,
                DiagnosticSeverity::Info,
                DiagnosticStage::Certification,
                format!("{parameter_label} is on its lower bound"),
            )
            .with_affected_terms(vec![parameter_label])
            .with_suggested_actions(vec![
                "treat a boundary covariance estimate as a valid fitted boundary, not by itself an optimizer failure".to_string(),
                "inspect the Effective Covariance section for unsupported random-effect directions".to_string(),
                "consider diagonal covariance, a simpler random-effect term, or design_compiled policy if the boundary direction is not scientifically central".to_string(),
            ]);
            diagnostic
                .payload
                .insert("theta_index".to_string(), serde_json::json!(index));
            diagnostic
                .payload
                .insert("value".to_string(), serde_json::json!(value));
            diagnostic
                .payload
                .insert("lower_bound".to_string(), serde_json::json!(lower));
            diagnostics.push(diagnostic);
        }

        let status = if !optimizer_ok {
            FitStatus::NotOptimized
        } else if boundary_indices.is_empty() {
            FitStatus::ConvergedInterior
        } else {
            FitStatus::ConvergedBoundary
        };

        Self {
            status,
            optimizer_name,
            objective_value,
            iterations,
            evidence,
            optimizer_control: OptimizerControlEvidence::from_opt_summary(optsum),
            verification: None,
            free_gradient_norm: None,
            projected_gradient_norm: None,
            hessian_eigen_min: None,
            hessian_rank: None,
            information_rank: None,
            checks,
            diagnostics,
        }
    }

    pub fn apply_derivative_evidence(
        &mut self,
        derivatives: OptimizerDerivativeEvidence,
        gradient_tolerance: f64,
        hessian_tolerance: f64,
    ) {
        if !self.evidence.optimizer_stop.acceptable_stop {
            return;
        }

        let n_theta = self.evidence.parameter_space.n_theta;
        if derivatives.gradient.len() != n_theta {
            self.checks.push(CertificateCheck::DerivativeMismatch {
                kind: "derivative_dimension_mismatch".to_string(),
                observed: Some(derivatives.gradient.len() as f64),
                tolerance: Some(n_theta as f64),
                regime: derivative_mismatch_regime(self),
                message: format!(
                    "gradient length {} does not match theta dimension {n_theta}",
                    derivatives.gradient.len()
                ),
            });
            self.evidence.certification_quality = EvidenceQuality::Approximate {
                reason: "derivative certificate dimensions did not match theta".to_string(),
            };
            return;
        }

        remove_derivative_not_assessed_checks(&mut self.checks);

        let boundary_mask = boundary_mask(n_theta, &self.evidence.parameter_space.boundary_indices);
        let raw_gradient_norm = max_abs_norm(&derivatives.gradient);
        let free_gradient_norm = derivatives
            .gradient
            .iter()
            .enumerate()
            .filter_map(|(index, value)| (!boundary_mask[index]).then_some(value.abs()))
            .fold(0.0, f64::max);
        let boundary_violation_max = derivatives
            .gradient
            .iter()
            .enumerate()
            .filter_map(|(index, value)| boundary_mask[index].then_some((-*value).max(0.0)))
            .fold(0.0, f64::max);
        let projected_gradient_norm = derivatives
            .gradient
            .iter()
            .enumerate()
            .map(|(index, value)| {
                if boundary_mask[index] {
                    (-*value).max(0.0)
                } else {
                    value.abs()
                }
            })
            .fold(0.0, f64::max);
        let objective_scale = self.objective_value.unwrap_or(1.0).abs().max(1.0);
        let scaled_gradient_norm = raw_gradient_norm / objective_scale;

        self.free_gradient_norm = Some(free_gradient_norm);
        self.projected_gradient_norm = Some(projected_gradient_norm);
        self.evidence.gradient = GradientEvidence {
            method: derivatives.method.clone(),
            raw_gradient_norm: Some(raw_gradient_norm),
            scaled_gradient_norm: Some(scaled_gradient_norm),
            free_gradient_norm: Some(free_gradient_norm),
            projected_gradient_norm: Some(projected_gradient_norm),
            kkt_boundary_gradient_max: Some(boundary_violation_max),
        };

        let mut failures = Vec::new();
        let mut convergence_failed = false;
        const MATERIAL_SCALED_GRADIENT_TOLERANCE: f64 = 1.0e-4;
        if free_gradient_norm <= gradient_tolerance {
            self.checks.push(CertificateCheck::FreeGradientOk {
                tolerance: gradient_tolerance,
                value: free_gradient_norm,
            });
        } else {
            convergence_failed =
                free_gradient_norm / objective_scale > MATERIAL_SCALED_GRADIENT_TOLERANCE;
            let message = format!(
                "free-gradient norm {free_gradient_norm:.6e} exceeds tolerance {gradient_tolerance:.6e}"
            );
            failures.push(message.clone());
            self.checks.push(CertificateCheck::DerivativeMismatch {
                kind: "free_gradient_kkt_mismatch".to_string(),
                observed: Some(free_gradient_norm),
                tolerance: Some(gradient_tolerance),
                regime: derivative_mismatch_regime(self),
                message,
            });
        }

        if boundary_violation_max <= gradient_tolerance {
            self.checks.push(CertificateCheck::BoundaryGradientOk {
                tolerance: gradient_tolerance,
                value: boundary_violation_max,
            });
        } else {
            convergence_failed |=
                boundary_violation_max / objective_scale > MATERIAL_SCALED_GRADIENT_TOLERANCE;
            let message = format!(
                "boundary KKT gradient violation {boundary_violation_max:.6e} exceeds tolerance {gradient_tolerance:.6e}"
            );
            failures.push(message.clone());
            self.checks.push(CertificateCheck::DerivativeMismatch {
                kind: "boundary_gradient_kkt_mismatch".to_string(),
                observed: Some(boundary_violation_max),
                tolerance: Some(gradient_tolerance),
                regime: derivative_mismatch_regime(self),
                message,
            });
        }

        match derivatives.hessian {
            Some(hessian) if hessian.nrows() == n_theta && hessian.ncols() == n_theta => {
                let active = active_hessian_summary(&hessian, &boundary_mask, hessian_tolerance);
                self.hessian_eigen_min = active.min_eigenvalue;
                self.hessian_rank = active.rank;
                self.information_rank = active.rank;
                self.evidence.hessian = HessianEvidence {
                    method: derivatives.method.clone(),
                    quality: if active.psd_ok && active.rank_ok {
                        approximate_or_certified_quality(
                            &derivatives.method,
                            "finite-difference active-subspace Hessian is positive semidefinite",
                        )
                    } else if !active.psd_ok {
                        EvidenceQuality::Approximate {
                            reason: "active-subspace Hessian has negative curvature".to_string(),
                        }
                    } else {
                        EvidenceQuality::Approximate {
                            reason: "active-subspace Hessian is rank deficient".to_string(),
                        }
                    },
                    min_eigenvalue: active.min_eigenvalue,
                    condition_number: active.condition_number,
                    rank: active.rank,
                };

                if active.psd_ok {
                    self.checks
                        .push(CertificateCheck::HessianPsdOnActiveSubspace {
                            min_eigenvalue: active.min_eigenvalue.unwrap_or(0.0),
                        });
                } else {
                    convergence_failed = true;
                    let min_eigen = active.min_eigenvalue.unwrap_or(f64::NAN);
                    let message = format!(
                        "active-subspace Hessian minimum eigenvalue {min_eigen:.6e} is below tolerance -{hessian_tolerance:.6e}"
                    );
                    failures.push(message.clone());
                    self.checks.push(CertificateCheck::DerivativeMismatch {
                        kind: "hessian_active_subspace_not_psd".to_string(),
                        observed: Some(min_eigen),
                        tolerance: Some(-hessian_tolerance),
                        regime: derivative_mismatch_regime(self),
                        message,
                    });
                }

                if active.rank_ok {
                    self.checks.push(CertificateCheck::RankOk {
                        rank: active.rank.unwrap_or(0),
                        expected: active.expected_rank,
                    });
                } else {
                    let rank = active.rank.unwrap_or(0);
                    let message = format!(
                        "active-subspace Hessian rank {rank} is below expected rank {}",
                        active.expected_rank
                    );
                    failures.push(message.clone());
                    self.checks.push(CertificateCheck::DerivativeMismatch {
                        kind: "hessian_active_subspace_rank_deficient".to_string(),
                        observed: Some(rank as f64),
                        tolerance: Some(active.expected_rank as f64),
                        regime: derivative_mismatch_regime(self),
                        message,
                    });
                }
            }
            Some(hessian) => {
                let message = format!(
                    "Hessian shape {}x{} does not match theta dimension {n_theta}",
                    hessian.nrows(),
                    hessian.ncols()
                );
                failures.push(message.clone());
                self.evidence.hessian = HessianEvidence {
                    method: derivatives.method.clone(),
                    quality: EvidenceQuality::Approximate {
                        reason: message.clone(),
                    },
                    min_eigenvalue: None,
                    condition_number: None,
                    rank: None,
                };
                self.checks.push(CertificateCheck::DerivativeMismatch {
                    kind: "hessian_dimension_mismatch".to_string(),
                    observed: Some((hessian.nrows() * hessian.ncols()) as f64),
                    tolerance: Some((n_theta * n_theta) as f64),
                    regime: derivative_mismatch_regime(self),
                    message,
                });
            }
            None => {
                let reason =
                    "active-subspace Hessian could not be computed by the derivative backend"
                        .to_string();
                self.evidence.hessian = HessianEvidence {
                    method: EvidenceMethod::NotAvailable {
                        reason: reason.clone(),
                    },
                    quality: EvidenceQuality::Unavailable {
                        reason: reason.clone(),
                    },
                    min_eigenvalue: None,
                    condition_number: None,
                    rank: None,
                };
                self.checks.push(CertificateCheck::NotAssessed { reason });
            }
        }

        self.evidence.certification_quality = if failures.is_empty() {
            approximate_or_certified_quality(
                &derivatives.method,
                "finite-difference KKT and Hessian checks passed",
            )
        } else {
            EvidenceQuality::Approximate {
                reason: failures.join("; "),
            }
        };

        // An optimizer return code is only one piece of convergence
        // evidence. If a fitted interior point materially fails the
        // free-gradient KKT check on both raw and objective-relative scales
        // (or has negative active-subspace curvature), it is not an optimized
        // solution even when the backend emitted FTOL/XTOL. The relative gate
        // prevents finite-difference noise on very large deviance scales from
        // mislabelling objective-equivalent fits. Keep the raw stop evidence
        // intact for auditability, but make the public fit status honest about
        // substantive derivative failure.
        if convergence_failed {
            self.status = FitStatus::NotOptimized;
            let mut diagnostic = Diagnostic::new(
                DiagnosticCode::OptimizerNonconvergence,
                DiagnosticSeverity::Warning,
                DiagnosticStage::Certification,
                "optimizer stop was accepted, but derivative checks rejected convergence",
            )
            .with_suggested_actions(vec![
                "treat this fit as not optimized despite the optimizer return code".to_string(),
                "restart from the fitted parameters or compare an alternate optimizer".to_string(),
            ]);
            diagnostic.payload.insert(
                "derivative_failures".to_string(),
                serde_json::json!(failures),
            );
            self.diagnostics.push(diagnostic);
        }
    }

    pub fn mark_derivative_checks_not_assessed(&mut self, reason: impl Into<String>) {
        let reason = reason.into();
        remove_derivative_not_assessed_checks(&mut self.checks);
        self.checks.push(CertificateCheck::NotAssessed {
            reason: format!("free-gradient KKT check skipped: {reason}"),
        });
        self.checks.push(CertificateCheck::NotAssessed {
            reason: format!("projected boundary-gradient KKT check skipped: {reason}"),
        });
        self.checks.push(CertificateCheck::NotAssessed {
            reason: format!("active-subspace Hessian check skipped: {reason}"),
        });
        self.evidence.gradient = GradientEvidence {
            method: EvidenceMethod::NotAssessed {
                reason: reason.clone(),
            },
            raw_gradient_norm: None,
            scaled_gradient_norm: None,
            free_gradient_norm: None,
            projected_gradient_norm: None,
            kkt_boundary_gradient_max: None,
        };
        self.evidence.hessian = HessianEvidence {
            method: EvidenceMethod::NotAssessed {
                reason: reason.clone(),
            },
            quality: EvidenceQuality::NotAssessed {
                reason: reason.clone(),
            },
            min_eigenvalue: None,
            condition_number: None,
            rank: None,
        };
        if self.evidence.optimizer_stop.acceptable_stop {
            self.evidence.certification_quality = EvidenceQuality::Approximate {
                reason: format!(
                    "optimizer stop accepted; derivative KKT/Hessian inspection skipped: {reason}"
                ),
            };
        }
    }
}

impl ConvergenceVerification {
    pub fn not_run(reason: impl Into<String>) -> Self {
        Self {
            status: ConvergenceVerificationStatus::NotRun,
            objective_tolerance: 0.0,
            theta_tolerance: 0.0,
            beta_tolerance: 0.0,
            reference_objective: None,
            reference_theta: Vec::new(),
            reference_beta: Vec::new(),
            reference_effective_ranks: Vec::new(),
            runs: Vec::new(),
            message: reason.into(),
        }
    }
}

impl ConvergenceEvidence {
    fn not_assessed() -> Self {
        Self {
            optimizer_stop: OptimizerStopEvidence {
                return_code: None,
                acceptable_stop: false,
                budget_exhausted: false,
                function_evaluations: None,
                max_function_evaluations: None,
                max_time_seconds: None,
                final_trust_radius: None,
                initial_objective: None,
                final_objective: None,
                objective_delta: None,
            },
            parameter_space: ParameterSpaceEvidence {
                n_theta: 0,
                n_free: 0,
                n_boundary: 0,
                boundary_indices: Vec::new(),
            },
            sample_size: SampleSizeContext {
                n_observations: None,
                n_theta: 0,
                observations_per_theta: None,
            },
            gradient: GradientEvidence::not_available("optimizer certificate not run".to_string()),
            hessian: HessianEvidence::not_available("optimizer certificate not run".to_string()),
            certification_quality: EvidenceQuality::NotAssessed {
                reason: "optimizer certificate not run".to_string(),
            },
        }
    }

    fn from_opt_summary(
        optsum: &OptSummary,
        theta: &[f64],
        n_observations: Option<usize>,
        boundary_indices: &[usize],
        optimizer_ok: bool,
    ) -> Self {
        let n_boundary = boundary_indices.len();
        let n_theta = theta.len();
        let n_free = n_theta.saturating_sub(n_boundary);
        let final_objective = optsum.fmin.is_finite().then_some(optsum.fmin);
        let initial_objective = optsum.finitial.is_finite().then_some(optsum.finitial);

        Self {
            optimizer_stop: OptimizerStopEvidence {
                return_code: if optsum.return_value.is_empty() {
                    None
                } else {
                    Some(optsum.return_value.clone())
                },
                acceptable_stop: optimizer_ok,
                budget_exhausted: optimizer_budget_exhausted(optsum),
                function_evaluations: usize::try_from(optsum.feval).ok(),
                max_function_evaluations: (optsum.max_feval > 0)
                    .then(|| usize::try_from(optsum.max_feval).ok())
                    .flatten(),
                max_time_seconds: (optsum.max_time > 0.0).then_some(optsum.max_time),
                final_trust_radius: optsum.final_trust_radius,
                initial_objective,
                final_objective,
                objective_delta: initial_objective
                    .zip(final_objective)
                    .map(|(initial, final_value)| initial - final_value),
            },
            parameter_space: ParameterSpaceEvidence {
                n_theta,
                n_free,
                n_boundary,
                boundary_indices: boundary_indices.to_vec(),
            },
            sample_size: SampleSizeContext {
                n_observations,
                n_theta,
                observations_per_theta: n_observations
                    .zip((n_theta > 0).then_some(n_theta))
                    .map(|(n, p)| n as f64 / p as f64),
            },
            gradient: GradientEvidence::not_available(
                "objective gradient is not exposed by the current derivative-free optimizer path"
                    .to_string(),
            ),
            hessian: HessianEvidence::not_available(
                "active-subspace Hessian is not exposed by the current optimizer path".to_string(),
            ),
            certification_quality: if !optsum.is_fitted() {
                EvidenceQuality::NotAssessed {
                    reason: "model has not been optimized".to_string(),
                }
            } else if optimizer_ok {
                EvidenceQuality::Approximate {
                    reason:
                        "optimizer stop accepted; derivative KKT/Hessian checks are not assessed"
                            .to_string(),
                }
            } else {
                EvidenceQuality::Failed {
                    reason:
                        "optimizer stop was not acceptable or optimization budget was exhausted"
                            .to_string(),
                }
            },
        }
    }
}

impl GradientEvidence {
    fn not_available(reason: String) -> Self {
        Self {
            method: EvidenceMethod::NotAvailable { reason },
            raw_gradient_norm: None,
            scaled_gradient_norm: None,
            free_gradient_norm: None,
            projected_gradient_norm: None,
            kkt_boundary_gradient_max: None,
        }
    }
}

impl HessianEvidence {
    fn not_available(reason: String) -> Self {
        Self {
            method: EvidenceMethod::NotAvailable {
                reason: reason.clone(),
            },
            quality: EvidenceQuality::Unavailable { reason },
            min_eigenvalue: None,
            condition_number: None,
            rank: None,
        }
    }
}

fn optimizer_stop_is_acceptable(return_value: &str) -> bool {
    matches!(
        optimizer_final_status_code(return_value),
        "SUCCESS" | "FTOL_REACHED" | "XTOL_REACHED" | "STOPVAL_REACHED" | "RADIUS_REACHED"
    )
}

fn optimizer_budget_exhausted(optsum: &OptSummary) -> bool {
    if optimizer_stop_is_acceptable(&optsum.return_value) {
        return false;
    }
    let return_value = optimizer_final_status_code(&optsum.return_value);
    return_value == "MAXEVAL_REACHED"
        || return_value == "MAXTIME_REACHED"
        || (optsum.max_feval > 0 && optsum.feval >= optsum.max_feval)
}

fn optimizer_recovery_reason(return_value: &str) -> Option<&str> {
    return_value
        .strip_prefix("KKT_BOUNDARY_RESTART(")?
        .split_once(')')
        .map(|(reason, _)| reason)
        .filter(|reason| !reason.is_empty())
}

fn boundary_parameter_indices(
    optsum: &OptSummary,
    theta: &[f64],
    lower_bounds: &[f64],
) -> Vec<usize> {
    let tol = optsum.xtol_zero_abs.max(1e-12) * 10.0;
    theta
        .iter()
        .zip(lower_bounds.iter())
        .enumerate()
        .filter_map(|(index, (&value, &lower))| {
            if lower.is_finite() && (value - lower).abs() <= tol {
                Some(index)
            } else {
                None
            }
        })
        .collect()
}

fn remove_derivative_not_assessed_checks(checks: &mut Vec<CertificateCheck>) {
    checks.retain(|check| match check {
        CertificateCheck::NotAssessed { reason } => {
            !(reason.contains("gradient")
                || reason.contains("Hessian")
                || reason.contains("derivative"))
        }
        _ => true,
    });
}

fn approximate_or_certified_quality(
    method: &EvidenceMethod,
    approximate_reason: &str,
) -> EvidenceQuality {
    if matches!(method, EvidenceMethod::Exact) {
        EvidenceQuality::Certified
    } else {
        EvidenceQuality::Approximate {
            reason: approximate_reason.to_string(),
        }
    }
}

fn boundary_mask(n_theta: usize, boundary_indices: &[usize]) -> Vec<bool> {
    let mut mask = vec![false; n_theta];
    for &index in boundary_indices {
        if index < n_theta {
            mask[index] = true;
        }
    }
    mask
}

fn derivative_mismatch_regime(certificate: &OptimizerCertificate) -> String {
    let space = &certificate.evidence.parameter_space;
    if space.n_boundary > 0 {
        "boundary_theta"
    } else if space.n_theta > 0 {
        "interior_theta"
    } else {
        "unknown"
    }
    .to_string()
}

fn max_abs_norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value.abs()).fold(0.0, f64::max)
}

struct ActiveHessianSummary {
    min_eigenvalue: Option<f64>,
    condition_number: Option<f64>,
    rank: Option<usize>,
    expected_rank: usize,
    psd_ok: bool,
    rank_ok: bool,
}

fn active_hessian_summary(
    hessian: &DMatrix<f64>,
    boundary_mask: &[bool],
    hessian_tolerance: f64,
) -> ActiveHessianSummary {
    let free_indices = boundary_mask
        .iter()
        .enumerate()
        .filter_map(|(index, is_boundary)| (!*is_boundary).then_some(index))
        .collect::<Vec<_>>();
    let expected_rank = free_indices.len();
    if expected_rank == 0 {
        return ActiveHessianSummary {
            min_eigenvalue: Some(0.0),
            condition_number: Some(1.0),
            rank: Some(0),
            expected_rank,
            psd_ok: true,
            rank_ok: true,
        };
    }

    let mut active = DMatrix::zeros(expected_rank, expected_rank);
    for (row_out, &row_in) in free_indices.iter().enumerate() {
        for (col_out, &col_in) in free_indices.iter().enumerate() {
            active[(row_out, col_out)] =
                0.5 * (hessian[(row_in, col_in)] + hessian[(col_in, row_in)]);
        }
    }

    let eigen = SymmetricEigen::new(active);
    let mut min_eigenvalue = f64::INFINITY;
    let mut max_eigenvalue = f64::NEG_INFINITY;
    let mut min_positive = f64::INFINITY;
    let mut rank = 0usize;
    for value in eigen.eigenvalues.iter().copied() {
        min_eigenvalue = min_eigenvalue.min(value);
        max_eigenvalue = max_eigenvalue.max(value);
        if value > hessian_tolerance {
            rank += 1;
            min_positive = min_positive.min(value);
        }
    }

    let condition_number = if rank == expected_rank && min_positive.is_finite() {
        Some(max_eigenvalue.abs().max(hessian_tolerance) / min_positive)
    } else {
        None
    };

    ActiveHessianSummary {
        min_eigenvalue: Some(min_eigenvalue),
        condition_number,
        rank: Some(rank),
        expected_rank,
        psd_ok: min_eigenvalue >= -hessian_tolerance,
        rank_ok: rank == expected_rank,
    }
}

/// Full audit container attached to fit artifacts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FitAudit {
    pub design: Option<DesignAudit>,
    pub optimizer: Option<OptimizerCertificate>,
    pub diagnostics: Vec<Diagnostic>,
    pub recommendations: Vec<String>,
}

impl FitAudit {
    pub fn empty() -> Self {
        Self {
            design: None,
            optimizer: None,
            diagnostics: Vec::new(),
            recommendations: Vec::new(),
        }
    }
}
