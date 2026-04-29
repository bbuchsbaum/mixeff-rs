use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use super::artifact::{
    CompiledModelArtifact, DerivativeAvailability, EffectiveRankStatus, InferenceAvailability,
    ModelKind, ModelStateStatus, ObjectiveApproximation, OptimizerCertificateScope,
};
use super::audit::{BasisAudit, InformationBudgetStatus, RandomTermAudit, RankStatus};
use super::diagnostics::{Diagnostic, DiagnosticCode, DiagnosticSeverity, FitStatus};
use super::ir::{CovarianceForm, RandomCoefficient, RandomCoefficientKind, RandomTermIr};
#[cfg(test)]
use super::policy::DEFAULT_CONVERGENCE_DERIVATIVE_NPARMAX;
use super::random_term_card::{
    CrossCardConstraint, DesignSupport, ImpliedConstraintKind, RandomTermBlock, RandomTermCard,
    RoleOrigin, WithinGroupVariation, RANDOM_TERM_CARD_SCHEMA, RANDOM_TERM_CARD_SCHEMA_VERSION,
};

pub const MODEL_AUDIT_REPORT_SCHEMA: &str = "mixedmodels.model_audit_report";
pub const MODEL_AUDIT_REPORT_SCHEMA_VERSION: u32 = 2;

/// Stable user-facing summary of a compiled/fitted model artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelAuditReport {
    pub schema_name: String,
    pub schema_version: u32,
    pub requested_formula: String,
    pub sections: Vec<AuditReportSection>,
    pub random_term_cards: Vec<RandomTermCard>,
    pub cross_card_constraints: Vec<CrossCardConstraint>,
    pub diagnostics: Vec<Diagnostic>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditReportSection {
    pub title: String,
    pub lines: Vec<AuditReportLine>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditReportLine {
    pub label: String,
    pub status: AuditReportStatus,
    pub detail: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditReportStatus {
    Ok,
    Info,
    Warning,
    Error,
    NotAssessed,
}

impl ModelAuditReport {
    pub fn from_artifact(artifact: &CompiledModelArtifact) -> Self {
        let mut sections = Vec::new();
        sections.push(requested_model_section(artifact));
        sections.push(model_state_section(artifact));
        sections.push(fixed_effect_section(artifact));
        sections.push(random_effect_section(artifact));
        sections.push(random_effect_information_budget_section(artifact));
        let random_term_cards = random_term_cards(artifact);
        let cross_card_constraints = cross_card_constraints(artifact);
        sections.push(random_term_cards_section(&random_term_cards));
        sections.push(cross_card_constraints_section(&cross_card_constraints));
        sections.push(dependence_path_section(artifact));
        sections.push(parameterization_trace_section(artifact));
        sections.push(effective_covariance_section(artifact));
        sections.push(policy_section(artifact));
        sections.push(optimizer_section(artifact));
        sections.push(inference_section(artifact));
        sections.push(diagnostics_section(artifact));

        Self {
            schema_name: MODEL_AUDIT_REPORT_SCHEMA.to_string(),
            schema_version: MODEL_AUDIT_REPORT_SCHEMA_VERSION,
            requested_formula: artifact.requested_formula.clone(),
            random_term_cards,
            cross_card_constraints,
            diagnostics: report_diagnostics(artifact),
            sections,
        }
    }

    pub fn to_text(&self) -> String {
        self.to_string()
    }
}

impl fmt::Display for ModelAuditReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let attention = attention_lines(&self.sections);
        let overview_status = overview_status(&attention);

        writeln!(f, "Audit Summary:")?;
        writeln!(
            f,
            "  overall [{}]: {}",
            status_label(overview_status),
            overview_detail(&attention)
        )?;
        if attention.is_empty() {
            writeln!(
                f,
                "  attention [{}]: no warnings or unchecked inference-critical items",
                status_label(AuditReportStatus::Ok)
            )?;
        } else {
            for item in &attention {
                writeln!(
                    f,
                    "  attention [{}]: {} / {}: {}",
                    status_label(item.status),
                    item.section,
                    item.label,
                    item.detail
                )?;
            }
        }
        writeln!(f)?;

        for (section_index, section) in self.sections.iter().enumerate() {
            if section_index > 0 {
                writeln!(f)?;
            }
            writeln!(f, "{}:", section.title)?;
            for line in &section.lines {
                writeln!(
                    f,
                    "  {} [{}]: {}",
                    line.label,
                    status_label(line.status),
                    line.detail
                )?;
            }
        }
        Ok(())
    }
}

struct AttentionLine {
    section: String,
    label: String,
    status: AuditReportStatus,
    detail: String,
}

fn attention_lines(sections: &[AuditReportSection]) -> Vec<AttentionLine> {
    let mut lines = Vec::new();
    for section in sections {
        for line in &section.lines {
            let high_priority = matches!(
                line.status,
                AuditReportStatus::Warning | AuditReportStatus::Error
            );
            let unchecked_inference_critical = line.status == AuditReportStatus::NotAssessed
                && matches!(
                    section.title.as_str(),
                    "Effective Covariance" | "Optimizer" | "Inference"
                );
            if high_priority || unchecked_inference_critical {
                lines.push(AttentionLine {
                    section: section.title.clone(),
                    label: line.label.clone(),
                    status: line.status,
                    detail: line.detail.clone(),
                });
            }
        }
    }
    lines.sort_by(|left, right| status_rank(right.status).cmp(&status_rank(left.status)));
    lines
}

fn overview_status(attention: &[AttentionLine]) -> AuditReportStatus {
    attention
        .iter()
        .map(|line| line.status)
        .max_by_key(|status| status_rank(*status))
        .unwrap_or(AuditReportStatus::Ok)
}

fn overview_detail(attention: &[AttentionLine]) -> String {
    if attention.is_empty() {
        return "ready: no warnings or unchecked inference-critical items".to_string();
    }

    let errors = attention
        .iter()
        .filter(|line| line.status == AuditReportStatus::Error)
        .count();
    let warnings = attention
        .iter()
        .filter(|line| line.status == AuditReportStatus::Warning)
        .count();
    let unchecked = attention
        .iter()
        .filter(|line| line.status == AuditReportStatus::NotAssessed)
        .count();

    let mut parts = Vec::new();
    if errors > 0 {
        parts.push(format!("{errors} error(s)"));
    }
    if warnings > 0 {
        parts.push(format!("{warnings} warning(s)"));
    }
    if unchecked > 0 {
        parts.push(format!("{unchecked} not checked item(s)"));
    }
    format!(
        "{}; review attention lines before treating inference as routine",
        parts.join(", ")
    )
}

fn model_state_section(artifact: &CompiledModelArtifact) -> AuditReportSection {
    let summary = artifact.model_state_summary();
    let mut lines = vec![
        AuditReportLine {
            label: "requested".to_string(),
            status: model_state_status(summary.requested.status),
            detail: model_stage_detail(&summary.requested),
        },
        AuditReportLine {
            label: "semantic".to_string(),
            status: model_state_status(summary.semantic.status),
            detail: model_stage_detail(&summary.semantic),
        },
        AuditReportLine {
            label: "supported".to_string(),
            status: model_state_status(summary.supported.status),
            detail: model_stage_detail(&summary.supported),
        },
        AuditReportLine {
            label: "fitted".to_string(),
            status: model_state_status(summary.fitted.status),
            detail: model_stage_detail(&summary.fitted),
        },
    ];

    lines.push(AuditReportLine {
        label: "changes".to_string(),
        status: model_changes_status(&summary.changes),
        detail: model_changes_detail(&summary.changes),
    });

    AuditReportSection {
        title: "Model State".to_string(),
        lines,
    }
}

fn requested_model_section(artifact: &CompiledModelArtifact) -> AuditReportSection {
    AuditReportSection {
        title: "Requested Model".to_string(),
        lines: vec![
            AuditReportLine {
                label: "formula".to_string(),
                status: AuditReportStatus::Info,
                detail: artifact.requested_formula.clone(),
            },
            AuditReportLine {
                label: "model kind".to_string(),
                status: AuditReportStatus::Info,
                detail: model_kind_label(artifact.model_boundary.model_kind).to_string(),
            },
            AuditReportLine {
                label: "distribution/link".to_string(),
                status: AuditReportStatus::Info,
                detail: format!(
                    "{}/{}",
                    artifact.model_boundary.response_distribution, artifact.model_boundary.link
                ),
            },
            AuditReportLine {
                label: "objective".to_string(),
                status: AuditReportStatus::Info,
                detail: objective_approximation_label(
                    &artifact.model_boundary.objective_approximation,
                ),
            },
            AuditReportLine {
                label: "certificate scope".to_string(),
                status: AuditReportStatus::Info,
                detail: optimizer_certificate_scope_label(
                    artifact.model_boundary.optimizer_certificate_scope,
                )
                .to_string(),
            },
            AuditReportLine {
                label: "fixed terms".to_string(),
                status: AuditReportStatus::Info,
                detail: artifact.semantic_model.fixed_terms.join(", "),
            },
            AuditReportLine {
                label: "random terms".to_string(),
                status: AuditReportStatus::Info,
                detail: artifact.semantic_model.random_terms.len().to_string(),
            },
            AuditReportLine {
                label: "theta maps".to_string(),
                status: AuditReportStatus::Info,
                detail: format!("{} map(s)", artifact.theta_maps.len()),
            },
        ],
    }
}

fn model_stage_detail(stage: &super::artifact::ModelStageState) -> String {
    let mut detail = format!(
        "status={}; formula={}; random_terms={}",
        model_state_status_label(stage.status),
        stage.formula,
        stage.random_terms.len()
    );
    if let Some(reason) = &stage.reason {
        detail.push_str("; reason=");
        detail.push_str(reason);
    }
    detail
}

fn model_changes_status(changes: &[super::artifact::ModelStateChange]) -> AuditReportStatus {
    if changes.is_empty() {
        AuditReportStatus::Ok
    } else if changes
        .iter()
        .any(|change| change.status == super::artifact::ModelChangeStatus::Applied)
    {
        AuditReportStatus::Info
    } else {
        AuditReportStatus::Warning
    }
}

fn model_changes_detail(changes: &[super::artifact::ModelStateChange]) -> String {
    if changes.is_empty() {
        return "none".to_string();
    }

    changes
        .iter()
        .map(|change| {
            format!(
                "{:?}:{:?}:{} -> {}",
                change.status, change.trigger, change.affected_term, change.reason
            )
        })
        .collect::<Vec<_>>()
        .join(" | ")
}

fn fixed_effect_section(artifact: &CompiledModelArtifact) -> AuditReportSection {
    let Some(audit) = &artifact.design_audit else {
        return AuditReportSection {
            title: "Fixed Effects".to_string(),
            lines: vec![not_assessed_line(
                "design audit",
                "fixed-effect audit not attached",
            )],
        };
    };

    let fixed = &audit.fixed_effects;
    let mut lines = vec![AuditReportLine {
        label: "rank".to_string(),
        status: rank_status(fixed.rank.status),
        detail: format!(
            "{} of {}",
            fixed
                .rank
                .rank
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            fixed
                .rank
                .expected
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unknown".to_string())
        ),
    }];

    lines.push(AuditReportLine {
        label: "aliased columns".to_string(),
        status: if fixed.aliased_columns.is_empty() {
            AuditReportStatus::Ok
        } else {
            AuditReportStatus::Warning
        },
        detail: if fixed.aliased_columns.is_empty() {
            "none".to_string()
        } else {
            fixed.aliased_columns.join(", ")
        },
    });

    lines.push(AuditReportLine {
        label: "empty cells".to_string(),
        status: if fixed.empty_cells.is_empty() {
            AuditReportStatus::Ok
        } else {
            AuditReportStatus::Warning
        },
        detail: fixed.empty_cells.len().to_string(),
    });

    AuditReportSection {
        title: "Fixed Effects".to_string(),
        lines,
    }
}

fn random_effect_section(artifact: &CompiledModelArtifact) -> AuditReportSection {
    let Some(audit) = &artifact.design_audit else {
        return AuditReportSection {
            title: "Random Effects".to_string(),
            lines: vec![not_assessed_line(
                "design audit",
                "random-effect audit not attached",
            )],
        };
    };

    let lines = audit
        .random_terms
        .iter()
        .map(|term| {
            let budget = &term.information_budget;
            AuditReportLine {
                label: term.term_id.clone(),
                status: information_budget_status(budget.status),
                detail: random_effect_budget_detail(term),
            }
        })
        .collect();

    AuditReportSection {
        title: "Random Effects".to_string(),
        lines,
    }
}

fn random_effect_budget_detail(term: &super::audit::RandomTermAudit) -> String {
    let budget = &term.information_budget;
    let mut detail = format!(
        "group={}, rows={}, levels={}, obs_per_level={}..{}, basis={}, covariance={}, params={}, budget={}",
        term.group.name,
        option_usize(term.group.n_observations),
        option_usize(budget.n_levels),
        option_usize(term.group.min_obs_per_level),
        option_usize(term.group.max_obs_per_level),
        budget.basis_dimension,
        budget.covariance_family,
        budget.requested_covariance_parameters,
        snake_status_budget(budget.status)
    );
    if let Some(reason) = &budget.reason {
        detail.push_str("; reason=");
        detail.push_str(reason);
    }
    detail
}

fn random_effect_information_budget_section(
    artifact: &CompiledModelArtifact,
) -> AuditReportSection {
    let Some(audit) = &artifact.design_audit else {
        return AuditReportSection {
            title: "Random-Effect Information Budget".to_string(),
            lines: vec![not_assessed_line(
                "design audit",
                "random-effect information budget not attached",
            )],
        };
    };

    let lines = audit
        .random_terms
        .iter()
        .map(|term| {
            let budget = &term.information_budget;
            AuditReportLine {
                label: term.term_id.clone(),
                status: information_budget_status(budget.status),
                detail: random_effect_information_budget_detail(term),
            }
        })
        .collect();

    AuditReportSection {
        title: "Random-Effect Information Budget".to_string(),
        lines,
    }
}

fn random_effect_information_budget_detail(term: &super::audit::RandomTermAudit) -> String {
    let budget = &term.information_budget;
    let effective_n = &budget.effective_n;
    let total_rows_note = if effective_n.total_rows_can_mislead {
        "total rows can be misleading for covariance support"
    } else {
        "grouping levels are the effective n for covariance support"
    };
    let overfit_risk = match budget.status {
        InformationBudgetStatus::TooRich => {
            "maximal covariance structure is too rich for the grouping-level budget"
        }
        InformationBudgetStatus::WeaklySupported => {
            "variance directions are weakly supported by the grouping-level budget"
        }
        InformationBudgetStatus::Sufficient => "v0 information budget is sufficient",
        InformationBudgetStatus::NotAssessable => "information budget could not be assessed",
    };

    format!(
        "levels={}, rows={}, obs_per_level={}..{}, basis={}, cov_params={}, levels/basis={}, levels/param={}, rows/param={}; {}; risk={}; recommendation={}; explanation={}",
        option_usize(effective_n.n_levels),
        option_usize(effective_n.n_rows),
        option_usize(effective_n.min_obs_per_level),
        option_usize(effective_n.max_obs_per_level),
        effective_n.basis_dimension,
        effective_n.covariance_parameters,
        option_f64(effective_n.levels_per_basis_direction),
        option_f64(effective_n.levels_per_covariance_parameter),
        option_f64(effective_n.rows_per_covariance_parameter),
        total_rows_note,
        overfit_risk,
        effective_n.recommendation,
        effective_n.explanation
    )
}

fn random_term_cards_section(cards: &[RandomTermCard]) -> AuditReportSection {
    let lines = if cards.is_empty() {
        vec![AuditReportLine {
            label: "cards".to_string(),
            status: AuditReportStatus::NotAssessed,
            detail: "none".to_string(),
        }]
    } else {
        cards
            .iter()
            .map(|card| AuditReportLine {
                label: card.term_id.clone(),
                status: information_budget_status(card.design_support.status),
                detail: random_term_card_detail(card),
            })
            .collect()
    };

    AuditReportSection {
        title: "Random Term Cards".to_string(),
        lines,
    }
}

fn cross_card_constraints_section(constraints: &[CrossCardConstraint]) -> AuditReportSection {
    let lines = if constraints.is_empty() {
        vec![AuditReportLine {
            label: "constraints".to_string(),
            status: AuditReportStatus::Ok,
            detail: "none".to_string(),
        }]
    } else {
        constraints
            .iter()
            .enumerate()
            .map(|(index, constraint)| AuditReportLine {
                label: format!("c{index}"),
                status: AuditReportStatus::Info,
                detail: format!(
                    "cards={}, basis={}, reason={}",
                    constraint.between_cards.join(" <-> "),
                    constraint.between_basis.join(" <-> "),
                    constraint.reason
                ),
            })
            .collect()
    };

    AuditReportSection {
        title: "Cross-Card Constraints".to_string(),
        lines,
    }
}

fn random_term_card_detail(card: &RandomTermCard) -> String {
    let blocks = card
        .blocks
        .iter()
        .map(|block| {
            format!(
                "basis=[{}], covariance={}, params={}",
                block.basis.join(", "),
                covariance_form_label(&block.covariance),
                block.theta_parameters
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    format!(
        "original={}, canonical={}, group={}, blocks={}",
        card.original_fragment,
        card.canonical_fragment,
        card.group.label(),
        blocks
    )
}

fn random_term_cards(artifact: &CompiledModelArtifact) -> Vec<RandomTermCard> {
    let audits_by_term = artifact
        .design_audit
        .as_ref()
        .map(|audit| {
            audit
                .random_terms
                .iter()
                .map(|term| (term.term_id.as_str(), term))
                .collect::<BTreeMap<_, _>>()
        })
        .unwrap_or_default();
    let within_group_threshold = artifact.compiler_policy.thresholds.min_within_group_sd;

    artifact
        .semantic_model
        .random_terms
        .iter()
        .map(|term| {
            let role_origin = artifact
                .semantic_model
                .role_origins
                .get(&term.id)
                .cloned()
                .unwrap_or_else(|| RoleOrigin::observed(term.role));
            random_term_card(
                term,
                audits_by_term.get(term.id.as_str()).copied(),
                within_group_threshold,
                role_origin,
            )
        })
        .collect()
}

fn random_term_card(
    term: &RandomTermIr,
    audit: Option<&RandomTermAudit>,
    within_group_threshold: f64,
    role_origin: RoleOrigin,
) -> RandomTermCard {
    let block = random_term_block(term, audit);
    RandomTermCard {
        schema_name: RANDOM_TERM_CARD_SCHEMA.to_string(),
        schema_version: RANDOM_TERM_CARD_SCHEMA_VERSION,
        term_id: term.id.clone(),
        original_fragment: term.source_syntax.user_text().to_string(),
        canonical_fragment: term.source_syntax.text.clone(),
        group: term.group.clone(),
        blocks: vec![block],
        implied_constraints: Vec::new(),
        design_support: design_support(term, audit, within_group_threshold),
        role_origin,
    }
}

fn random_term_block(term: &RandomTermIr, audit: Option<&RandomTermAudit>) -> RandomTermBlock {
    let basis = card_basis_names(term, audit);
    let intercept = card_has_intercept(term, audit);
    let slopes = card_slope_names(term, audit);
    let theta_parameters = audit
        .map(|audit| audit.requested_covariance_parameters)
        .unwrap_or_else(|| covariance_parameter_count(&term.covariance, basis.len()));
    RandomTermBlock {
        basis,
        intercept,
        slopes: slopes.clone(),
        covariance: term.covariance.clone(),
        theta_parameters,
        english: random_term_block_english(term, intercept, &slopes),
    }
}

fn design_support(
    term: &RandomTermIr,
    audit: Option<&RandomTermAudit>,
    within_group_threshold: f64,
) -> DesignSupport {
    let within_group_variation = audit
        .map(|audit| within_group_variation(&audit.basis, within_group_threshold))
        .unwrap_or_else(|| {
            term.basis
                .iter()
                .map(|basis| {
                    (
                        card_basis_display_name(basis),
                        WithinGroupVariation::NotAssessed,
                    )
                })
                .collect()
        });
    DesignSupport {
        group_levels: audit.and_then(|audit| audit.group.n_levels),
        min_rows_per_group: audit.and_then(|audit| audit.group.min_obs_per_level),
        median_rows_per_group: audit.and_then(|audit| audit.group.median_obs_per_level),
        within_group_variation,
        status: audit
            .map(|audit| audit.information_budget.status)
            .unwrap_or(InformationBudgetStatus::NotAssessable),
    }
}

fn within_group_variation(
    basis: &[BasisAudit],
    within_group_threshold: f64,
) -> BTreeMap<String, WithinGroupVariation> {
    basis
        .iter()
        .map(|basis| {
            let status = match (basis.min_within_group_sd, basis.max_within_group_sd) {
                (Some(min), Some(_)) if min > within_group_threshold => {
                    WithinGroupVariation::Present
                }
                (Some(_), Some(max)) if max <= within_group_threshold => {
                    WithinGroupVariation::Absent
                }
                (Some(min), Some(max)) if min.is_finite() && (min - max).abs() <= f64::EPSILON => {
                    WithinGroupVariation::Constant
                }
                (Some(_), Some(_)) => WithinGroupVariation::Present,
                _ => WithinGroupVariation::NotAssessed,
            };
            (basis.name.clone(), status)
        })
        .collect()
}

fn card_basis_names(term: &RandomTermIr, audit: Option<&RandomTermAudit>) -> Vec<String> {
    audit
        .map(|audit| audit.basis.iter().map(|basis| basis.name.clone()).collect())
        .filter(|basis: &Vec<String>| !basis.is_empty())
        .unwrap_or_else(|| {
            term.basis
                .iter()
                .map(card_basis_display_name)
                .collect::<Vec<_>>()
        })
}

fn card_has_intercept(term: &RandomTermIr, audit: Option<&RandomTermAudit>) -> bool {
    audit
        .map(|audit| audit.basis.iter().any(|basis| basis.kind == "intercept"))
        .unwrap_or_else(|| {
            term.basis
                .iter()
                .any(|basis| basis.kind == RandomCoefficientKind::Intercept)
        })
}

fn card_slope_names(term: &RandomTermIr, audit: Option<&RandomTermAudit>) -> Vec<String> {
    audit
        .map(|audit| {
            audit
                .basis
                .iter()
                .filter(|basis| basis.kind != "intercept")
                .map(|basis| basis.name.clone())
                .collect()
        })
        .unwrap_or_else(|| {
            term.basis
                .iter()
                .filter(|basis| {
                    matches!(
                        basis.kind,
                        RandomCoefficientKind::Slope | RandomCoefficientKind::Interaction
                    )
                })
                .map(card_basis_display_name)
                .collect()
        })
}

fn card_basis_display_name(basis: &RandomCoefficient) -> String {
    if basis.kind == RandomCoefficientKind::Intercept {
        "Intercept".to_string()
    } else {
        basis.name.clone()
    }
}

fn random_term_block_english(
    term: &RandomTermIr,
    has_intercept: bool,
    slopes: &[String],
) -> String {
    let group = quoted_identifier(&term.group.label());
    match (has_intercept, slopes) {
        (true, []) => format!("{group} units may differ in average outcome."),
        (false, [slope]) => format!(
            "{group} units may differ in their {} slope.",
            quoted_identifier(slope)
        ),
        (true, [slope]) if term.covariance == CovarianceForm::Full => format!(
            "{group} units differ in baseline and {} slope; the model estimates whether these are associated.",
            quoted_identifier(slope)
        ),
        (true, [slope]) => format!(
            "{group} units may differ in average outcome and their {} slope.",
            quoted_identifier(slope)
        ),
        (false, slopes) if !slopes.is_empty() => format!(
            "{group} units may differ in their slopes for {}.",
            quoted_list(slopes)
        ),
        (true, slopes) if !slopes.is_empty() => format!(
            "{group} units may differ in average outcome and slopes for {}.",
            quoted_list(slopes)
        ),
        _ => format!("{group} units may differ across the requested random-effect basis."),
    }
}

fn quoted_list(items: &[String]) -> String {
    items
        .iter()
        .map(|item| quoted_identifier(item))
        .collect::<Vec<_>>()
        .join(", ")
}

fn quoted_identifier(identifier: &str) -> String {
    format!("`{}`", identifier.replace('`', "\\`"))
}

fn covariance_parameter_count(covariance: &CovarianceForm, basis_size: usize) -> usize {
    match covariance {
        CovarianceForm::Scalar => 1,
        CovarianceForm::Diagonal => basis_size,
        CovarianceForm::Full => basis_size * (basis_size + 1) / 2,
        CovarianceForm::Structured { .. } => basis_size,
        CovarianceForm::ReducedRank { rank } => rank.unwrap_or(1) * basis_size,
        CovarianceForm::Unsupported { .. } => 0,
    }
}

fn covariance_form_label(covariance: &CovarianceForm) -> String {
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

fn cross_card_constraints(artifact: &CompiledModelArtifact) -> Vec<CrossCardConstraint> {
    let terms = &artifact.semantic_model.random_terms;
    let mut constraints = Vec::new();

    let mut by_block_group: BTreeMap<&str, Vec<&RandomTermIr>> = BTreeMap::new();
    for term in terms {
        if let Some(block_group) = &term.block_group {
            by_block_group
                .entry(block_group.as_str())
                .or_default()
                .push(term);
        }
    }
    for block_terms in by_block_group.values() {
        for left_index in 0..block_terms.len() {
            for right_index in (left_index + 1)..block_terms.len() {
                if let Some(constraint) = cross_card_constraint(
                    block_terms[left_index],
                    block_terms[right_index],
                    "double_bar_syntax",
                ) {
                    constraints.push(constraint);
                }
            }
        }
    }

    for left_index in 0..terms.len() {
        for right_index in (left_index + 1)..terms.len() {
            let left = &terms[left_index];
            let right = &terms[right_index];
            if left.group != right.group {
                continue;
            }
            if left.block_group.is_some() && left.block_group == right.block_group {
                continue;
            }
            if let Some(constraint) =
                cross_card_constraint(left, right, "separate_random_effect_blocks")
            {
                constraints.push(constraint);
            }
        }
    }

    constraints.sort_by(|left, right| {
        left.between_cards
            .cmp(&right.between_cards)
            .then_with(|| left.between_basis.cmp(&right.between_basis))
            .then_with(|| left.reason.cmp(&right.reason))
    });
    constraints
}

fn cross_card_constraint(
    left: &RandomTermIr,
    right: &RandomTermIr,
    reason_kind: &'static str,
) -> Option<CrossCardConstraint> {
    let left_basis = left.basis.first().map(card_basis_display_name)?;
    let right_basis = right.basis.first().map(card_basis_display_name)?;
    if left_basis == right_basis {
        return None;
    }
    let left_label = quoted_identifier(&left_basis);
    let right_label = quoted_identifier(&right_basis);
    let reason = match reason_kind {
        "double_bar_syntax" => format!(
            "double-bar syntax fixes the covariance between {left_label} and {right_label} to zero."
        ),
        _ => format!(
            "separate random-effect blocks fix the covariance between {left_label} and {right_label} to zero."
        ),
    };
    Some(CrossCardConstraint {
        kind: ImpliedConstraintKind::ZeroCovariance,
        between_cards: vec![left.id.clone(), right.id.clone()],
        between_basis: vec![left_basis, right_basis],
        reason,
    })
}

fn dependence_path_section(artifact: &CompiledModelArtifact) -> AuditReportSection {
    let Some(audit) = &artifact.design_audit else {
        return AuditReportSection {
            title: "Dependence Paths".to_string(),
            lines: vec![not_assessed_line(
                "covariance kernels",
                "dependence-path audit not attached",
            )],
        };
    };

    let graph = &audit.covariance_kernels;
    AuditReportSection {
        title: "Dependence Paths".to_string(),
        lines: vec![
            AuditReportLine {
                label: "kernels".to_string(),
                status: AuditReportStatus::Info,
                detail: covariance_kernel_detail(graph),
            },
            AuditReportLine {
                label: "repeated units".to_string(),
                status: if graph.missing_dependence_paths.is_empty() {
                    AuditReportStatus::Ok
                } else {
                    AuditReportStatus::Warning
                },
                detail: repeated_units_detail(graph),
            },
            AuditReportLine {
                label: "missing paths".to_string(),
                status: if graph.missing_dependence_paths.is_empty() {
                    AuditReportStatus::Ok
                } else {
                    AuditReportStatus::Warning
                },
                detail: missing_dependence_paths_detail(graph),
            },
        ],
    }
}

fn covariance_kernel_detail(graph: &super::audit::CovarianceKernelGraphAudit) -> String {
    if graph.kernels.is_empty() {
        return "none requested".to_string();
    }

    graph
        .kernels
        .iter()
        .map(|kernel| {
            format!(
                "{}={}({}, intercept={}, covariance={}, basis={})",
                kernel.term_id,
                dependence_path_kind_label(kernel.path),
                kernel.group,
                kernel.has_intercept,
                kernel.covariance_family,
                kernel.basis.join(", ")
            )
        })
        .collect::<Vec<_>>()
        .join(" | ")
}

fn repeated_units_detail(graph: &super::audit::CovarianceKernelGraphAudit) -> String {
    if graph.repeated_units.is_empty() {
        return "none detected by v0 heuristics".to_string();
    }

    graph
        .repeated_units
        .iter()
        .map(|unit| {
            format!(
                "{}={}({}, levels={}, obs_per_level={}..{}, covered_by={})",
                unit.unit,
                dependence_path_kind_label(unit.path),
                unit.parts.join(":"),
                unit.n_levels,
                unit.min_obs_per_level,
                unit.max_obs_per_level,
                if unit.covered_by_terms.is_empty() {
                    "none".to_string()
                } else {
                    unit.covered_by_terms.join(", ")
                }
            )
        })
        .collect::<Vec<_>>()
        .join(" | ")
}

fn missing_dependence_paths_detail(graph: &super::audit::CovarianceKernelGraphAudit) -> String {
    if graph.missing_dependence_paths.is_empty() {
        return "none".to_string();
    }

    graph
        .missing_dependence_paths
        .iter()
        .map(|missing| {
            format!(
                "{} -> {}; {}",
                missing.unit, missing.suggested_random_term, missing.reason
            )
        })
        .collect::<Vec<_>>()
        .join(" | ")
}

fn dependence_path_kind_label(path: super::audit::DependencePathKind) -> &'static str {
    match path {
        super::audit::DependencePathKind::Marginal => "marginal",
        super::audit::DependencePathKind::Cell => "cell",
        super::audit::DependencePathKind::Interaction => "interaction",
    }
}

fn effective_covariance_section(artifact: &CompiledModelArtifact) -> AuditReportSection {
    if artifact.effective_covariance.is_empty() {
        return AuditReportSection {
            title: "Effective Covariance".to_string(),
            lines: vec![not_assessed_line(
                "effective covariance rank",
                "not assessed",
            )],
        };
    }

    AuditReportSection {
        title: "Effective Covariance".to_string(),
        lines: artifact
            .effective_covariance
            .iter()
            .map(|summary| AuditReportLine {
                label: summary.term_id.clone(),
                status: effective_rank_status(summary.status),
                detail: effective_covariance_detail(summary),
            })
            .collect(),
    }
}

fn parameterization_trace_section(artifact: &CompiledModelArtifact) -> AuditReportSection {
    if artifact.covariance_parameter_traces.is_empty() {
        return AuditReportSection {
            title: "Parameterization Trace".to_string(),
            lines: vec![not_assessed_line(
                "theta map",
                "no covariance parameter traces recorded",
            )],
        };
    }

    let mut by_term: BTreeMap<String, Vec<&super::artifact::CovarianceParameterTrace>> =
        BTreeMap::new();
    for trace in &artifact.covariance_parameter_traces {
        by_term
            .entry(trace.term_id.clone())
            .or_default()
            .push(trace);
    }

    let lines = by_term
        .into_iter()
        .map(|(term_id, traces)| {
            let parmap_aligned = traces
                .iter()
                .filter_map(|trace| trace.parmap_entry.as_ref())
                .all(|entry| entry.matches_theta_map);
            let parmap_count = traces
                .iter()
                .filter(|trace| trace.parmap_entry.is_some())
                .count();
            let lambda_slots = traces
                .iter()
                .map(|trace| format!("({}, {})", trace.lambda.row, trace.lambda.col))
                .collect::<Vec<_>>()
                .join(", ");
            let varcorr_entries = unique_strings(
                traces
                    .iter()
                    .flat_map(|trace| trace.varcorr_entries.iter().map(|entry| entry.label.clone()))
                    .collect(),
            )
            .join(", ");
            let theta_indices = traces
                .iter()
                .filter_map(|trace| trace.theta.global_index.map(|idx| format!("theta[{idx}]")))
                .collect::<Vec<_>>()
                .join(", ");
            let first = traces[0];

            AuditReportLine {
                label: term_id,
                status: if parmap_count == traces.len() && parmap_aligned {
                    AuditReportStatus::Ok
                } else if parmap_count == 0 {
                    AuditReportStatus::Info
                } else {
                    AuditReportStatus::Warning
                },
                detail: format!(
                    "source={}; group={}; family={:?}; user_basis={}; optimizer_basis={}; theta_slots={}; lambda_slots={}; parmap_aligned={}/{}; varcorr_entries={}",
                    first.source_syntax,
                    first.group,
                    first.covariance_family,
                    first.user_basis.join(", "),
                    first.optimizer_basis.join(", "),
                    if theta_indices.is_empty() {
                        "none".to_string()
                    } else {
                        theta_indices
                    },
                    lambda_slots,
                    if parmap_aligned { parmap_count } else { 0 },
                    traces.len(),
                    if varcorr_entries.is_empty() {
                        "none".to_string()
                    } else {
                        varcorr_entries
                    }
                ),
            }
        })
        .collect();

    AuditReportSection {
        title: "Parameterization Trace".to_string(),
        lines,
    }
}

fn unique_strings(values: Vec<String>) -> Vec<String> {
    let mut unique = Vec::new();
    for value in values {
        if !unique.iter().any(|seen| seen == &value) {
            unique.push(value);
        }
    }
    unique
}

fn policy_section(artifact: &CompiledModelArtifact) -> AuditReportSection {
    if artifact.policy_recommendations.is_empty() {
        return AuditReportSection {
            title: "Policy Recommendations".to_string(),
            lines: vec![AuditReportLine {
                label: "maximal feasible".to_string(),
                status: AuditReportStatus::Ok,
                detail: "no advisory reductions or refusals".to_string(),
            }],
        };
    }

    AuditReportSection {
        title: "Policy Recommendations".to_string(),
        lines: artifact
            .policy_recommendations
            .iter()
            .map(|recommendation| AuditReportLine {
                label: recommendation.term_id.clone(),
                status: recommendation_status(&recommendation.diagnostics),
                detail: format!(
                    "{}: {}{}; inference={}",
                    policy_action_label(recommendation.action),
                    recommendation.reason,
                    recommendation
                        .recommended_covariance
                        .as_ref()
                        .map(|covariance| format!("; recommended covariance={covariance}"))
                        .unwrap_or_default(),
                    recommendation.inference_consequence
                ),
            })
            .collect(),
    }
}

fn policy_action_label(action: super::policy::PolicyAction) -> &'static str {
    match action {
        super::policy::PolicyAction::DropUnsupportedBasis => "drop_unsupported_basis",
        super::policy::PolicyAction::ReduceCovariance => "reduce_covariance",
        super::policy::PolicyAction::RefuseRandomTermDistribution => {
            "refuse_random_term_distribution"
        }
        super::policy::PolicyAction::MarkNotAssessable => "mark_not_assessable",
    }
}

fn optimizer_section(artifact: &CompiledModelArtifact) -> AuditReportSection {
    let Some(certificate) = &artifact.optimizer_certificate else {
        return AuditReportSection {
            title: "Optimizer".to_string(),
            lines: vec![not_assessed_line(
                "certificate",
                "model has not been fitted",
            )],
        };
    };

    let mut lines = vec![AuditReportLine {
        label: "status".to_string(),
        status: fit_status(certificate.status),
        detail: format!("{:?}", certificate.status),
    }];
    lines.push(convergence_interpretation_line(certificate));
    lines.push(AuditReportLine {
        label: "optimizer".to_string(),
        status: AuditReportStatus::Info,
        detail: certificate
            .optimizer_name
            .clone()
            .unwrap_or_else(|| "unknown".to_string()),
    });
    lines.push(AuditReportLine {
        label: "objective".to_string(),
        status: AuditReportStatus::Info,
        detail: certificate
            .objective_value
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "unknown".to_string()),
    });
    lines.push(AuditReportLine {
        label: "optimizer stop".to_string(),
        status: if certificate.evidence.optimizer_stop.acceptable_stop {
            AuditReportStatus::Ok
        } else {
            AuditReportStatus::Warning
        },
        detail: format!(
            "return_code={}; acceptable={}; budget_exhausted={}; fevals={}; objective_delta={}",
            certificate
                .evidence
                .optimizer_stop
                .return_code
                .as_deref()
                .unwrap_or("unknown"),
            certificate.evidence.optimizer_stop.acceptable_stop,
            certificate.evidence.optimizer_stop.budget_exhausted,
            option_usize(certificate.evidence.optimizer_stop.function_evaluations),
            option_f64(certificate.evidence.optimizer_stop.objective_delta)
        ),
    });
    lines.push(AuditReportLine {
        label: "parameter space".to_string(),
        status: if certificate.evidence.parameter_space.n_boundary == 0 {
            AuditReportStatus::Ok
        } else {
            AuditReportStatus::Info
        },
        detail: format!(
            "theta={}, free={}, boundary={}, boundary_indices={}",
            certificate.evidence.parameter_space.n_theta,
            certificate.evidence.parameter_space.n_free,
            certificate.evidence.parameter_space.n_boundary,
            if certificate
                .evidence
                .parameter_space
                .boundary_indices
                .is_empty()
            {
                "none".to_string()
            } else {
                certificate
                    .evidence
                    .parameter_space
                    .boundary_indices
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            }
        ),
    });
    lines.push(AuditReportLine {
        label: "sample-size context".to_string(),
        status: AuditReportStatus::Info,
        detail: format!(
            "n={}, theta={}, n/theta={}",
            option_usize(certificate.evidence.sample_size.n_observations),
            certificate.evidence.sample_size.n_theta,
            option_f64(certificate.evidence.sample_size.observations_per_theta)
        ),
    });
    lines.push(AuditReportLine {
        label: "gradient evidence".to_string(),
        status: evidence_method_status(&certificate.evidence.gradient.method),
        detail: format!(
            "method={}; raw={}; scaled={}; free={}; projected={}; kkt_boundary={}",
            evidence_method_label(&certificate.evidence.gradient.method),
            option_f64(certificate.evidence.gradient.raw_gradient_norm),
            option_f64(certificate.evidence.gradient.scaled_gradient_norm),
            option_f64(certificate.evidence.gradient.free_gradient_norm),
            option_f64(certificate.evidence.gradient.projected_gradient_norm),
            option_f64(certificate.evidence.gradient.kkt_boundary_gradient_max)
        ),
    });
    lines.push(AuditReportLine {
        label: "hessian evidence".to_string(),
        status: hessian_evidence_status(certificate),
        detail: format!(
            "method={}; quality={}; min_eigen={}; condition={}; rank={}",
            evidence_method_label(&certificate.evidence.hessian.method),
            evidence_quality_label(&certificate.evidence.hessian.quality),
            option_f64(certificate.evidence.hessian.min_eigenvalue),
            option_f64(certificate.evidence.hessian.condition_number),
            option_usize(certificate.evidence.hessian.rank)
        ),
    });
    lines.push(AuditReportLine {
        label: "certification quality".to_string(),
        status: certification_quality_status(certificate),
        detail: evidence_quality_detail(&certificate.evidence.certification_quality),
    });
    lines.push(convergence_next_steps_line(
        certificate,
        &artifact.diagnostics,
    ));

    let not_assessed = certificate
        .checks
        .iter()
        .filter(|check| matches!(check, super::audit::CertificateCheck::NotAssessed { .. }))
        .count();
    let failed = certificate
        .checks
        .iter()
        .filter(|check| matches!(check, super::audit::CertificateCheck::Failed { .. }))
        .count();
    lines.push(AuditReportLine {
        label: "derivative checks".to_string(),
        status: if failed > 0 && certificate.evidence.optimizer_stop.acceptable_stop {
            AuditReportStatus::Info
        } else if failed > 0 {
            AuditReportStatus::Warning
        } else if not_assessed == 0 {
            AuditReportStatus::Ok
        } else {
            AuditReportStatus::NotAssessed
        },
        detail: format!("{failed} failed; {not_assessed} not assessed"),
    });
    lines.push(convergence_verification_line(certificate));

    AuditReportSection {
        title: "Optimizer".to_string(),
        lines,
    }
}

/// Compact, action-oriented summary of model convergence quality.
///
/// `ConvergenceVerdict` is the single line a user should read first when
/// inspecting a fitted model. It partitions evidence into two sources —
/// the *optimizer* (gradient/Hessian/verification certificate) and the
/// *structural* design (pre-fit identifiability diagnostics like
/// row-saturated random effects, separation, collinear fixed effects).
/// Optimizer tinkering does not fix structural design failures, so the
/// `next_step` recommendation is gated on `source` to avoid suggesting
/// "increase budget" when the model is unidentifiable.
///
/// This is a derived projection — built on demand from
/// `CompiledModelArtifact::optimizer_certificate` and
/// `CompiledModelArtifact::diagnostics` via
/// [`ConvergenceVerdict::for_artifact`]. It is **not** persisted on the
/// artifact so the audit JSON schema is unchanged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConvergenceVerdict {
    pub level: ConvergenceLevel,
    pub source: ConvergenceSource,
    /// One short clause summarising the verdict for compact print.
    pub headline: String,
    /// Structured convergence/inspection evidence that backs the headline.
    pub evidence: Vec<ConvergenceVerdictEvidence>,
    /// Stable action code for downstream renderers.
    pub next_action: Option<ConvergenceNextAction>,
    /// One actionable clause; `None` for clean fits where no follow-up
    /// is recommended.
    pub next_step: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConvergenceVerdictEvidence {
    pub test_name: String,
    pub observed: Option<f64>,
    pub threshold: Option<f64>,
    pub regime: ConvergenceRegime,
    pub status: ConvergenceTestStatus,
    pub detail: String,
    pub doc_anchor: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConvergenceLevel {
    Certified,
    Ok,
    Caution,
    Failed,
    NotAssessed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConvergenceSource {
    Clean,
    Optimizer,
    Structural,
    Mixed,
    NotAssessed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConvergenceRegime {
    Unfitted,
    OptimizerStop,
    InteriorTheta,
    BoundaryTheta,
    LargeTheta,
    Verification,
    Structural,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConvergenceTestStatus {
    Passed,
    Failed,
    Skipped,
    NotAssessed,
    Informational,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConvergenceNextAction {
    FitModel,
    IncreaseBudgetOrAlternateOptimizer,
    VerifyConvergence,
    GateInferenceOnDerivativeEvidence,
    GateWeakIdentification,
    RescaleOrSimplifyRandomEffects,
    InspectEffectiveCovariance,
    TreatAgreementAsReassuring,
    CompareVerificationRuns,
    ReviseStructuralModel,
}

impl ConvergenceLevel {
    pub fn as_str(self) -> &'static str {
        match self {
            ConvergenceLevel::Certified => "certified",
            ConvergenceLevel::Ok => "ok",
            ConvergenceLevel::Caution => "caution",
            ConvergenceLevel::Failed => "failed",
            ConvergenceLevel::NotAssessed => "not assessed",
        }
    }
}

impl ConvergenceSource {
    pub fn as_str(self) -> &'static str {
        match self {
            ConvergenceSource::Clean => "clean",
            ConvergenceSource::Optimizer => "optimizer",
            ConvergenceSource::Structural => "structural",
            ConvergenceSource::Mixed => "mixed",
            ConvergenceSource::NotAssessed => "not_assessed",
        }
    }
}

impl ConvergenceVerdict {
    /// Build the verdict for a compiled model artifact. Reads both the
    /// optimizer certificate and the artifact-level diagnostics (the
    /// latter carries structural pre-fit signals like row-saturated
    /// random effects).
    pub fn for_artifact(artifact: &CompiledModelArtifact) -> Self {
        match artifact.optimizer_certificate.as_ref() {
            None => Self::for_unfitted(),
            Some(certificate) => Self::compose_with_nparmax(
                certificate,
                &artifact.diagnostics,
                artifact
                    .compiler_policy
                    .thresholds
                    .convergence_derivative_nparmax,
            ),
        }
    }

    /// The "model has not been fitted yet" verdict.
    pub fn for_unfitted() -> Self {
        Self {
            level: ConvergenceLevel::NotAssessed,
            source: ConvergenceSource::NotAssessed,
            headline: "model is not fitted".to_string(),
            evidence: vec![ConvergenceVerdictEvidence {
                test_name: "fit_state".to_string(),
                observed: None,
                threshold: None,
                regime: ConvergenceRegime::Unfitted,
                status: ConvergenceTestStatus::NotAssessed,
                detail: "model has not been fitted".to_string(),
                doc_anchor: "docs/compiler_verdicts.md#not-assessed".to_string(),
            }],
            next_action: Some(ConvergenceNextAction::FitModel),
            next_step: Some("call .fit() to populate the convergence certificate".to_string()),
        }
    }

    /// Compact one-line render: `"<level> — <headline>"`. The print path
    /// uses this; callers who need structured access read the fields
    /// directly.
    pub fn one_liner(&self) -> String {
        format!("{} — {}", self.level.as_str(), self.headline)
    }

    pub fn primary_doc_anchor(&self) -> Option<&str> {
        self.evidence
            .first()
            .map(|evidence| evidence.doc_anchor.as_str())
    }

    #[cfg(test)]
    fn compose(
        certificate: &super::audit::OptimizerCertificate,
        diagnostics: &[Diagnostic],
    ) -> Self {
        Self::compose_with_nparmax(
            certificate,
            diagnostics,
            DEFAULT_CONVERGENCE_DERIVATIVE_NPARMAX,
        )
    }

    fn compose_with_nparmax(
        certificate: &super::audit::OptimizerCertificate,
        diagnostics: &[Diagnostic],
        derivative_nparmax: usize,
    ) -> Self {
        let optimizer = optimizer_summary(certificate, derivative_nparmax);
        let structural = structural_findings(diagnostics);

        if structural.is_empty() {
            // Pure optimizer-side path. Resolve final level + source +
            // next_step from the optimizer summary alone. Reassurance
            // actions (e.g. "verification agrees") are not real
            // follow-ups, so they don't populate `next_step`.
            let next_action_kind = optimizer
                .actions
                .iter()
                .filter(|a| !a.is_reassurance())
                .min_by_key(|a| a.priority())
                .copied();
            let next_step = next_action_kind.map(|a| a.text().to_string());

            let (level, source) = match (optimizer.level, &next_step) {
                (ConvergenceLevel::Certified, None) => {
                    (ConvergenceLevel::Certified, ConvergenceSource::Clean)
                }
                (ConvergenceLevel::Ok, _) => (ConvergenceLevel::Ok, ConvergenceSource::Clean),
                (level, _) => (level, ConvergenceSource::Optimizer),
            };

            return Self {
                level,
                source,
                headline: optimizer.headline,
                evidence: optimizer.evidence,
                next_action: next_action_kind.map(|action| ConvergenceNextAction::from(action)),
                next_step,
            };
        }

        // Structural finding(s) present. Optimizer tinkering will not
        // help — pick the highest-priority structural finding for the
        // next_step and either subordinate the optimizer signal (pure
        // Structural) or surface it alongside (Mixed).
        let primary = structural
            .iter()
            .min_by_key(|finding| finding.priority())
            .expect("non-empty checked above");

        let optimizer_clean = matches!(
            optimizer.level,
            ConvergenceLevel::Certified | ConvergenceLevel::Ok
        );
        let (source, headline) = if optimizer_clean {
            (
                ConvergenceSource::Structural,
                format!("structural: {}", primary.headline()),
            )
        } else {
            (
                ConvergenceSource::Mixed,
                format!(
                    "structural: {} (optimizer: {})",
                    primary.headline(),
                    optimizer.headline
                ),
            )
        };

        Self {
            level: ConvergenceLevel::Failed,
            source,
            headline,
            evidence: structural_verdict_evidence(&structural),
            next_action: Some(primary.next_action()),
            next_step: Some(primary.next_step()),
        }
    }
}

/// Structured optimizer-side next-step.
///
/// Each variant carries its rendered text and a `is_optimizer_tinkering`
/// flag used to suppress optimizer-only advice when a structural design
/// failure has been diagnosed (the lme4 #120 lesson — you can't fix a
/// row-saturated random effect by tweaking the optimizer).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NextActionKind {
    /// "increase optimizer budget or try an alternate optimizer"
    BudgetOrAlternate,
    /// "run verify_convergence() to compare restart and alternate-optimizer agreement"
    SuggestVerify,
    /// "gate inference on derivative-backed or finite-difference stationarity evidence"
    GateInferenceOnDerivative,
    /// "gate weak-identification claims until Hessian evidence is available"
    GateWeakIdentification,
    /// "scale predictors, simplify the random-effects structure, or collect more grouping levels"
    PredictorScalingOrSimplifyRe,
    /// "inspect Effective Covariance and consider diagonal covariance, a simpler random-effect term, or design_compiled policy"
    InspectEffectiveCovariance,
    /// "treat optimizer-agreement evidence as reassuring, while keeping any boundary or rank caveats"
    TreatAgreementAsReassuring,
    /// "compare objectives, theta, and beta across verification runs"
    CompareVerificationRuns,
}

impl NextActionKind {
    fn text(self) -> &'static str {
        match self {
            NextActionKind::BudgetOrAlternate => {
                "increase optimizer budget or try an alternate optimizer"
            }
            NextActionKind::SuggestVerify => {
                "run verify_convergence() to compare restart and alternate-optimizer agreement"
            }
            NextActionKind::GateInferenceOnDerivative => {
                "gate inference on derivative-backed or finite-difference stationarity evidence"
            }
            NextActionKind::GateWeakIdentification => {
                "gate weak-identification claims until Hessian evidence is available"
            }
            NextActionKind::PredictorScalingOrSimplifyRe => {
                "scale predictors, simplify the random-effects structure, or collect more grouping levels"
            }
            NextActionKind::InspectEffectiveCovariance => {
                "inspect Effective Covariance and consider diagonal covariance, a simpler random-effect term, or design_compiled policy"
            }
            NextActionKind::TreatAgreementAsReassuring => {
                "treat optimizer-agreement evidence as reassuring, while keeping any boundary or rank caveats"
            }
            NextActionKind::CompareVerificationRuns => {
                "compare objectives, theta, and beta across verification runs"
            }
        }
    }

    /// True for optimizer-tinkering actions that are pointless when the
    /// underlying issue is structural identifiability.
    fn is_optimizer_tinkering(self) -> bool {
        matches!(
            self,
            NextActionKind::BudgetOrAlternate
                | NextActionKind::SuggestVerify
                | NextActionKind::GateInferenceOnDerivative
        )
    }

    /// True when the action is a reassurance about an already-clean fit
    /// rather than a follow-up the user should run. Excluded when the
    /// verdict picks its single `next_step`.
    fn is_reassurance(self) -> bool {
        matches!(self, NextActionKind::TreatAgreementAsReassuring)
    }

    /// Lower = more actionable, used by [`ConvergenceVerdict`] to pick a
    /// single recommendation. The audit line shows them all.
    fn priority(self) -> u8 {
        match self {
            NextActionKind::BudgetOrAlternate => 0,
            NextActionKind::InspectEffectiveCovariance => 1,
            NextActionKind::SuggestVerify => 2,
            NextActionKind::GateInferenceOnDerivative => 3,
            NextActionKind::PredictorScalingOrSimplifyRe => 3,
            NextActionKind::CompareVerificationRuns => 5,
            NextActionKind::GateWeakIdentification => 6,
            NextActionKind::TreatAgreementAsReassuring => 7,
        }
    }
}

impl From<NextActionKind> for ConvergenceNextAction {
    fn from(value: NextActionKind) -> Self {
        match value {
            NextActionKind::BudgetOrAlternate => {
                ConvergenceNextAction::IncreaseBudgetOrAlternateOptimizer
            }
            NextActionKind::SuggestVerify => ConvergenceNextAction::VerifyConvergence,
            NextActionKind::GateInferenceOnDerivative => {
                ConvergenceNextAction::GateInferenceOnDerivativeEvidence
            }
            NextActionKind::GateWeakIdentification => ConvergenceNextAction::GateWeakIdentification,
            NextActionKind::PredictorScalingOrSimplifyRe => {
                ConvergenceNextAction::RescaleOrSimplifyRandomEffects
            }
            NextActionKind::InspectEffectiveCovariance => {
                ConvergenceNextAction::InspectEffectiveCovariance
            }
            NextActionKind::TreatAgreementAsReassuring => {
                ConvergenceNextAction::TreatAgreementAsReassuring
            }
            NextActionKind::CompareVerificationRuns => {
                ConvergenceNextAction::CompareVerificationRuns
            }
        }
    }
}

/// Compact, level-aware summary of the optimizer dimension only. The
/// verdict overlays this with structural findings; the audit lines
/// (`convergence_interpretation`, `convergence_next_steps_line`) reuse
/// the same `actions` list to stay in sync.
struct OptimizerSummary {
    level: ConvergenceLevel,
    headline: String,
    actions: Vec<NextActionKind>,
    evidence: Vec<ConvergenceVerdictEvidence>,
}

fn optimizer_summary(
    certificate: &super::audit::OptimizerCertificate,
    derivative_nparmax: usize,
) -> OptimizerSummary {
    let mut level = ConvergenceLevel::Certified;
    let mut clauses: Vec<String> = Vec::new();
    let mut actions: Vec<NextActionKind> = Vec::new();
    let evidence = optimizer_verdict_evidence(certificate, derivative_nparmax);

    let bump = |current: &mut ConvergenceLevel, target: ConvergenceLevel| {
        if level_severity(target) > level_severity(*current) {
            *current = target;
        }
    };

    if !certificate.evidence.optimizer_stop.acceptable_stop {
        bump(&mut level, ConvergenceLevel::Failed);
        clauses.push("optimizer stop unacceptable".to_string());
        actions.push(NextActionKind::BudgetOrAlternate);
    }

    match certificate.status {
        FitStatus::ConvergedInterior => {
            clauses.push("interior optimum".to_string());
        }
        FitStatus::ConvergedBoundary => {
            bump(&mut level, ConvergenceLevel::Caution);
            clauses.push("boundary fit".to_string());
            actions.push(NextActionKind::InspectEffectiveCovariance);
        }
        FitStatus::ConvergedReducedRank => {
            bump(&mut level, ConvergenceLevel::Caution);
            clauses.push("reduced-rank fit".to_string());
            actions.push(NextActionKind::InspectEffectiveCovariance);
        }
        FitStatus::ConvergedPenalised => {
            bump(&mut level, ConvergenceLevel::Caution);
            clauses.push("penalised fit (not an MLE)".to_string());
        }
        FitStatus::NotIdentifiable => {
            bump(&mut level, ConvergenceLevel::Failed);
            clauses.push("not identifiable".to_string());
        }
        FitStatus::NotOptimized => {
            bump(&mut level, ConvergenceLevel::Failed);
            clauses.push("optimization did not run to completion".to_string());
        }
        FitStatus::NotAssessed => {
            bump(&mut level, ConvergenceLevel::NotAssessed);
            clauses.push("optimizer certificate not assessed".to_string());
        }
    }

    if matches!(
        certificate.evidence.gradient.method,
        super::audit::EvidenceMethod::NotAvailable { .. }
            | super::audit::EvidenceMethod::NotAssessed { .. }
    ) {
        bump(&mut level, ConvergenceLevel::Caution);
        let regime = convergence_regime(certificate, derivative_nparmax);
        if matches!(
            certificate.evidence.gradient.method,
            super::audit::EvidenceMethod::NotAssessed { .. }
        ) && matches!(
            regime,
            ConvergenceRegime::BoundaryTheta | ConvergenceRegime::LargeTheta
        ) {
            clauses.push("derivative inspection skipped by regime".to_string());
        } else {
            clauses.push("derivative inspection not assessed".to_string());
            actions.push(NextActionKind::GateInferenceOnDerivative);
        }
    }

    if let super::audit::EvidenceQuality::Failed { .. } =
        &certificate.evidence.certification_quality
    {
        bump(&mut level, ConvergenceLevel::Caution);
        if certificate.evidence.optimizer_stop.acceptable_stop {
            clauses.push("inspection note: derivative check did not pass".to_string());
        } else {
            clauses.push("certification failed".to_string());
        }
        actions.push(NextActionKind::PredictorScalingOrSimplifyRe);
    }

    match &certificate.evidence.hessian.quality {
        super::audit::EvidenceQuality::Failed { .. } => {
            bump(&mut level, ConvergenceLevel::Caution);
            if certificate.evidence.optimizer_stop.acceptable_stop {
                clauses.push("inspection note: weak Hessian".to_string());
            } else {
                clauses.push("weak Hessian".to_string());
            }
            actions.push(NextActionKind::PredictorScalingOrSimplifyRe);
        }
        super::audit::EvidenceQuality::Unavailable { .. }
        | super::audit::EvidenceQuality::NotAssessed { .. } => {
            bump(&mut level, ConvergenceLevel::Caution);
            clauses.push("Hessian evidence unavailable".to_string());
            actions.push(NextActionKind::GateWeakIdentification);
        }
        _ => {}
    }

    match certificate.verification.as_ref().map(|v| v.status) {
        Some(super::audit::ConvergenceVerificationStatus::RestartAgrees)
        | Some(super::audit::ConvergenceVerificationStatus::OptimizerConsensus) => {
            clauses.push("verification agrees".to_string());
            actions.push(NextActionKind::TreatAgreementAsReassuring);
        }
        Some(super::audit::ConvergenceVerificationStatus::Fragile) => {
            bump(&mut level, ConvergenceLevel::Caution);
            clauses.push("verification fragile".to_string());
            actions.push(NextActionKind::CompareVerificationRuns);
        }
        Some(super::audit::ConvergenceVerificationStatus::Unstable) => {
            bump(&mut level, ConvergenceLevel::Failed);
            clauses.push("verification unstable".to_string());
            actions.push(NextActionKind::CompareVerificationRuns);
        }
        Some(super::audit::ConvergenceVerificationStatus::NotRun) | None => {
            // No verification → invite it but don't downgrade an
            // otherwise-clean fit below Ok.
            if matches!(level, ConvergenceLevel::Certified) {
                level = ConvergenceLevel::Ok;
            }
            clauses.push("verification not run".to_string());
            actions.push(NextActionKind::SuggestVerify);
        }
    }

    if matches!(level, ConvergenceLevel::Certified)
        && !has_certified_derivative_evidence(certificate)
    {
        level = ConvergenceLevel::Ok;
        clauses.push("derivative evidence is approximate".to_string());
    }

    OptimizerSummary {
        level,
        headline: clauses.join("; "),
        actions,
        evidence,
    }
}

fn optimizer_verdict_evidence(
    certificate: &super::audit::OptimizerCertificate,
    derivative_nparmax: usize,
) -> Vec<ConvergenceVerdictEvidence> {
    let mut evidence = Vec::new();
    evidence.push(ConvergenceVerdictEvidence {
        test_name: "optimizer_stop".to_string(),
        observed: None,
        threshold: None,
        regime: ConvergenceRegime::OptimizerStop,
        status: if certificate.evidence.optimizer_stop.acceptable_stop {
            ConvergenceTestStatus::Passed
        } else {
            ConvergenceTestStatus::Failed
        },
        detail: optimizer_stop_detail(certificate),
        doc_anchor: "docs/compiler_verdicts.md#optimizer-stop".to_string(),
    });

    evidence.push(ConvergenceVerdictEvidence {
        test_name: "theta_regime".to_string(),
        observed: Some(certificate.evidence.parameter_space.n_theta as f64),
        threshold: Some(derivative_nparmax as f64),
        regime: convergence_regime(certificate, derivative_nparmax),
        status: ConvergenceTestStatus::Informational,
        detail: theta_regime_detail(certificate, derivative_nparmax),
        doc_anchor: theta_regime_doc_anchor(certificate, derivative_nparmax).to_string(),
    });

    for check in &certificate.checks {
        match check {
            super::audit::CertificateCheck::FreeGradientOk { tolerance, value } => {
                evidence.push(ConvergenceVerdictEvidence {
                    test_name: "free_gradient_kkt".to_string(),
                    observed: Some(*value),
                    threshold: Some(*tolerance),
                    regime: convergence_regime(certificate, derivative_nparmax),
                    status: ConvergenceTestStatus::Passed,
                    detail: format!("max free-gradient {value:.6e} <= tolerance {tolerance:.6e}"),
                    doc_anchor: "docs/compiler_verdicts.md#derivative-inspection".to_string(),
                });
            }
            super::audit::CertificateCheck::BoundaryGradientOk { tolerance, value } => {
                evidence.push(ConvergenceVerdictEvidence {
                    test_name: "boundary_gradient_kkt".to_string(),
                    observed: Some(*value),
                    threshold: Some(*tolerance),
                    regime: ConvergenceRegime::BoundaryTheta,
                    status: ConvergenceTestStatus::Passed,
                    detail: format!(
                        "max boundary-gradient violation {value:.6e} <= tolerance {tolerance:.6e}"
                    ),
                    doc_anchor: "docs/compiler_verdicts.md#boundary-and-singular-fits".to_string(),
                });
            }
            super::audit::CertificateCheck::HessianPsdOnActiveSubspace { min_eigenvalue } => {
                evidence.push(ConvergenceVerdictEvidence {
                    test_name: "active_subspace_hessian_psd".to_string(),
                    observed: Some(*min_eigenvalue),
                    threshold: Some(0.0),
                    regime: convergence_regime(certificate, derivative_nparmax),
                    status: ConvergenceTestStatus::Passed,
                    detail: format!(
                        "active-subspace Hessian minimum eigenvalue {min_eigenvalue:.6e}"
                    ),
                    doc_anchor: "docs/compiler_verdicts.md#derivative-inspection".to_string(),
                });
            }
            super::audit::CertificateCheck::RankOk { rank, expected } => {
                evidence.push(ConvergenceVerdictEvidence {
                    test_name: "active_subspace_hessian_rank".to_string(),
                    observed: Some(*rank as f64),
                    threshold: Some(*expected as f64),
                    regime: convergence_regime(certificate, derivative_nparmax),
                    status: ConvergenceTestStatus::Passed,
                    detail: format!("active-subspace Hessian rank {rank} of {expected}"),
                    doc_anchor: "docs/compiler_verdicts.md#derivative-inspection".to_string(),
                });
            }
            super::audit::CertificateCheck::NotAssessed { reason } => {
                evidence.push(ConvergenceVerdictEvidence {
                    test_name: not_assessed_test_name(reason).to_string(),
                    observed: skipped_observed(certificate, reason),
                    threshold: skipped_threshold(derivative_nparmax, reason),
                    regime: skipped_regime(certificate, derivative_nparmax, reason),
                    status: ConvergenceTestStatus::Skipped,
                    detail: reason.clone(),
                    doc_anchor: skipped_doc_anchor(reason).to_string(),
                });
            }
            super::audit::CertificateCheck::Failed { code, message } => {
                evidence.push(ConvergenceVerdictEvidence {
                    test_name: code.clone(),
                    observed: failed_check_observed(certificate, code),
                    threshold: None,
                    regime: convergence_regime(certificate, derivative_nparmax),
                    status: ConvergenceTestStatus::Failed,
                    detail: if certificate.evidence.optimizer_stop.acceptable_stop {
                        format!(
                            "post-hoc inspection failed but does not override optimizer convergence: {message}"
                        )
                    } else {
                        message.clone()
                    },
                    doc_anchor: "docs/compiler_verdicts.md#derivative-inspection".to_string(),
                });
            }
        }
    }

    if let Some(verification) = &certificate.verification {
        evidence.push(ConvergenceVerdictEvidence {
            test_name: "convergence_verification".to_string(),
            observed: Some(verification.runs.iter().filter(|run| run.agrees).count() as f64),
            threshold: Some(verification.runs.len() as f64),
            regime: ConvergenceRegime::Verification,
            status: match verification.status {
                super::audit::ConvergenceVerificationStatus::RestartAgrees
                | super::audit::ConvergenceVerificationStatus::OptimizerConsensus => {
                    ConvergenceTestStatus::Passed
                }
                super::audit::ConvergenceVerificationStatus::Fragile
                | super::audit::ConvergenceVerificationStatus::Unstable => {
                    ConvergenceTestStatus::Failed
                }
                super::audit::ConvergenceVerificationStatus::NotRun => {
                    ConvergenceTestStatus::NotAssessed
                }
            },
            detail: verification.message.clone(),
            doc_anchor: "docs/compiler_verdicts.md#verification".to_string(),
        });
    }

    evidence
}

fn optimizer_stop_detail(certificate: &super::audit::OptimizerCertificate) -> String {
    let code = certificate
        .evidence
        .optimizer_stop
        .return_code
        .as_deref()
        .unwrap_or("unknown");
    if certificate.evidence.optimizer_stop.acceptable_stop {
        format!("optimizer returned acceptable stop code {code}")
    } else {
        format!("optimizer stop code {code} is not acceptable")
    }
}

fn convergence_regime(
    certificate: &super::audit::OptimizerCertificate,
    derivative_nparmax: usize,
) -> ConvergenceRegime {
    let space = &certificate.evidence.parameter_space;
    if space.n_boundary > 0 {
        ConvergenceRegime::BoundaryTheta
    } else if space.n_theta > derivative_nparmax {
        ConvergenceRegime::LargeTheta
    } else if space.n_theta > 0 {
        ConvergenceRegime::InteriorTheta
    } else {
        ConvergenceRegime::Unknown
    }
}

fn theta_regime_detail(
    certificate: &super::audit::OptimizerCertificate,
    derivative_nparmax: usize,
) -> String {
    let space = &certificate.evidence.parameter_space;
    if space.n_boundary > 0 {
        format!(
            "theta has {} parameter(s), {} on boundary; derivative KKT checks are skipped for boundary fits",
            space.n_theta, space.n_boundary
        )
    } else if space.n_theta > derivative_nparmax {
        format!(
            "theta has {} parameter(s), above convergence_derivative_nparmax {}; finite-difference checks are skipped",
            space.n_theta, derivative_nparmax
        )
    } else {
        format!(
            "theta has {} interior parameter(s), within convergence_derivative_nparmax {}",
            space.n_theta, derivative_nparmax
        )
    }
}

fn theta_regime_doc_anchor(
    certificate: &super::audit::OptimizerCertificate,
    derivative_nparmax: usize,
) -> &'static str {
    match convergence_regime(certificate, derivative_nparmax) {
        ConvergenceRegime::BoundaryTheta => "docs/compiler_verdicts.md#boundary-and-singular-fits",
        ConvergenceRegime::LargeTheta => "docs/compiler_verdicts.md#large-theta-fits",
        _ => "docs/compiler_verdicts.md#derivative-inspection",
    }
}

fn not_assessed_test_name(reason: &str) -> &'static str {
    if reason.contains("free-gradient") {
        "free_gradient_kkt"
    } else if reason.contains("boundary-gradient") {
        "boundary_gradient_kkt"
    } else if reason.contains("Hessian") {
        "active_subspace_hessian"
    } else {
        "derivative_inspection"
    }
}

fn skipped_observed(certificate: &super::audit::OptimizerCertificate, reason: &str) -> Option<f64> {
    if reason.contains("theta dimension") {
        Some(certificate.evidence.parameter_space.n_theta as f64)
    } else {
        None
    }
}

fn skipped_threshold(derivative_nparmax: usize, reason: &str) -> Option<f64> {
    reason
        .contains("convergence_derivative_nparmax")
        .then_some(derivative_nparmax as f64)
}

fn skipped_regime(
    certificate: &super::audit::OptimizerCertificate,
    derivative_nparmax: usize,
    reason: &str,
) -> ConvergenceRegime {
    if reason.contains("boundary") {
        ConvergenceRegime::BoundaryTheta
    } else if reason.contains("large-theta") || reason.contains("theta dimension") {
        ConvergenceRegime::LargeTheta
    } else {
        convergence_regime(certificate, derivative_nparmax)
    }
}

fn skipped_doc_anchor(reason: &str) -> &'static str {
    if reason.contains("boundary") {
        "docs/compiler_verdicts.md#boundary-and-singular-fits"
    } else if reason.contains("large-theta") || reason.contains("theta dimension") {
        "docs/compiler_verdicts.md#large-theta-fits"
    } else {
        "docs/compiler_verdicts.md#derivative-inspection"
    }
}

fn failed_check_observed(
    certificate: &super::audit::OptimizerCertificate,
    code: &str,
) -> Option<f64> {
    match code {
        "free_gradient_kkt_failed" => certificate.evidence.gradient.free_gradient_norm,
        "boundary_gradient_kkt_failed" => certificate.evidence.gradient.kkt_boundary_gradient_max,
        "hessian_active_subspace_not_psd" => certificate.evidence.hessian.min_eigenvalue,
        "hessian_active_subspace_rank_deficient" => {
            certificate.evidence.hessian.rank.map(|rank| rank as f64)
        }
        _ => None,
    }
}

fn has_certified_derivative_evidence(certificate: &super::audit::OptimizerCertificate) -> bool {
    matches!(
        certificate.evidence.gradient.method,
        super::audit::EvidenceMethod::Exact
    ) && matches!(
        certificate.evidence.hessian.method,
        super::audit::EvidenceMethod::Exact
    ) && matches!(
        certificate.evidence.hessian.quality,
        super::audit::EvidenceQuality::Certified
    ) && matches!(
        certificate.evidence.certification_quality,
        super::audit::EvidenceQuality::Certified
    )
}

fn level_severity(level: ConvergenceLevel) -> u8 {
    match level {
        ConvergenceLevel::Certified => 0,
        ConvergenceLevel::Ok => 1,
        ConvergenceLevel::NotAssessed => 2,
        ConvergenceLevel::Caution => 3,
        ConvergenceLevel::Failed => 4,
    }
}

/// Pre-fit structural identifiability findings detectable from the
/// artifact's diagnostics.
#[derive(Debug, Clone, PartialEq, Eq)]
enum StructuralFinding {
    RowSaturatedRandomEffect { term: String },
    StructuralRefusal { term: String, reason: String },
    Separation { reason: String },
    NotIdentifiableOther { reason: String },
    FixedRankDeficient { reason: String },
    EmptyCell { reason: String },
    UnsupportedRandomSlope { term: String },
    RepeatedUnitUnmodeled { term: String },
}

impl StructuralFinding {
    fn priority(&self) -> u8 {
        match self {
            StructuralFinding::RowSaturatedRandomEffect { .. } => 0,
            StructuralFinding::Separation { .. } => 1,
            StructuralFinding::StructuralRefusal { .. } => 2,
            StructuralFinding::FixedRankDeficient { .. } => 3,
            StructuralFinding::EmptyCell { .. } => 4,
            StructuralFinding::UnsupportedRandomSlope { .. } => 5,
            StructuralFinding::NotIdentifiableOther { .. } => 6,
            StructuralFinding::RepeatedUnitUnmodeled { .. } => 7,
        }
    }

    fn headline(&self) -> String {
        match self {
            StructuralFinding::RowSaturatedRandomEffect { term } => {
                format!("row-saturated random effect {term}")
            }
            StructuralFinding::StructuralRefusal { term, reason } => {
                if reason.is_empty() {
                    format!("structural refusal on {term}")
                } else {
                    format!("structural refusal on {term} ({reason})")
                }
            }
            StructuralFinding::Separation { reason } => {
                if reason.is_empty() {
                    "separation; likelihood unbounded".to_string()
                } else {
                    format!("separation: {reason}")
                }
            }
            StructuralFinding::NotIdentifiableOther { reason } => {
                if reason.is_empty() {
                    "model not identifiable".to_string()
                } else {
                    format!("not identifiable: {reason}")
                }
            }
            StructuralFinding::FixedRankDeficient { reason } => {
                if reason.is_empty() {
                    "fixed-effect design rank-deficient".to_string()
                } else {
                    format!("fixed-effect rank-deficient: {reason}")
                }
            }
            StructuralFinding::EmptyCell { reason } => {
                if reason.is_empty() {
                    "fixed-effect design has empty cells".to_string()
                } else {
                    format!("empty fixed-effect cell: {reason}")
                }
            }
            StructuralFinding::UnsupportedRandomSlope { term } => {
                format!("requested random slope unsupported by within-group design ({term})")
            }
            StructuralFinding::RepeatedUnitUnmodeled { term } => {
                format!("repeated unit unmodeled ({term})")
            }
        }
    }

    fn next_step(&self) -> String {
        match self {
            StructuralFinding::RowSaturatedRandomEffect { term } => format!(
                "design has as many random effects as rows for term {term}; drop unsupported slopes, split RE structure, or treat as fixed — optimizer tuning will not help",
            ),
            StructuralFinding::StructuralRefusal { term, .. } => format!(
                "remove the slope on {term} or move to a different grouping; this is a design refusal and not optimizer-fixable",
            ),
            StructuralFinding::Separation { .. } => {
                "separation detected; use a Firth/penalised fit or drop the offending predictor".to_string()
            }
            StructuralFinding::NotIdentifiableOther { .. } => {
                "model is not identifiable under the requested contract; reduce the model or add identifying constraints".to_string()
            }
            StructuralFinding::FixedRankDeficient { .. } => {
                "fixed-effect design is rank-deficient; drop redundant predictors or aggregate factor levels".to_string()
            }
            StructuralFinding::EmptyCell { .. } => {
                "fixed-effect design has empty cells; aggregate levels or remove the offending interaction".to_string()
            }
            StructuralFinding::UnsupportedRandomSlope { term } => format!(
                "requested random slope on {term} is not supported by the within-group design; remove the slope",
            ),
            StructuralFinding::RepeatedUnitUnmodeled { term } => format!(
                "add an explicit grouping factor (e.g. (1 | {term})) to model repeated units",
            ),
        }
    }

    fn next_action(&self) -> ConvergenceNextAction {
        ConvergenceNextAction::ReviseStructuralModel
    }
}

fn structural_verdict_evidence(findings: &[StructuralFinding]) -> Vec<ConvergenceVerdictEvidence> {
    findings
        .iter()
        .map(|finding| ConvergenceVerdictEvidence {
            test_name: "structural_design".to_string(),
            observed: None,
            threshold: None,
            regime: ConvergenceRegime::Structural,
            status: ConvergenceTestStatus::Failed,
            detail: finding.headline(),
            doc_anchor: "docs/compiler_verdicts.md#structural-fit-status".to_string(),
        })
        .collect()
}

fn structural_findings(diagnostics: &[Diagnostic]) -> Vec<StructuralFinding> {
    let mut out: Vec<StructuralFinding> = Vec::new();
    for diag in diagnostics {
        match &diag.code {
            DiagnosticCode::CovarianceTooRich => {
                let row_saturated = diag
                    .payload
                    .get("row_saturated")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                if row_saturated {
                    let term = diag
                        .affected_terms
                        .first()
                        .cloned()
                        .unwrap_or_else(|| "<random term>".to_string());
                    out.push(StructuralFinding::RowSaturatedRandomEffect { term });
                }
            }
            DiagnosticCode::StructuralRefusal => {
                let term = diag
                    .affected_terms
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "<random term>".to_string());
                out.push(StructuralFinding::StructuralRefusal {
                    term,
                    reason: diag.message.clone(),
                });
            }
            DiagnosticCode::NotIdentifiable => {
                let reason = diag.message.to_lowercase();
                if reason.contains("separation") || reason.contains("separated") {
                    out.push(StructuralFinding::Separation {
                        reason: diag.message.clone(),
                    });
                } else {
                    out.push(StructuralFinding::NotIdentifiableOther {
                        reason: diag.message.clone(),
                    });
                }
            }
            DiagnosticCode::FixedEffectRankDeficient => {
                out.push(StructuralFinding::FixedRankDeficient {
                    reason: diag.message.clone(),
                });
            }
            DiagnosticCode::FixedEffectEmptyCell => {
                out.push(StructuralFinding::EmptyCell {
                    reason: diag.message.clone(),
                });
            }
            DiagnosticCode::RandomSlopeUnsupported => {
                let term = diag
                    .affected_terms
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "<random term>".to_string());
                out.push(StructuralFinding::UnsupportedRandomSlope { term });
            }
            DiagnosticCode::RepeatedUnitUnmodeled => {
                let term = diag
                    .affected_terms
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "<grouping>".to_string());
                out.push(StructuralFinding::RepeatedUnitUnmodeled { term });
            }
            _ => {}
        }
    }
    out
}

fn convergence_interpretation_line(
    certificate: &super::audit::OptimizerCertificate,
) -> AuditReportLine {
    let (status, detail) = convergence_interpretation(certificate);
    AuditReportLine {
        label: "convergence interpretation".to_string(),
        status,
        detail,
    }
}

fn convergence_interpretation(
    certificate: &super::audit::OptimizerCertificate,
) -> (AuditReportStatus, String) {
    let mut status = fit_status(certificate.status);
    let mut parts = Vec::new();

    if !certificate.evidence.optimizer_stop.acceptable_stop {
        status = max_status(status, AuditReportStatus::Warning);
        parts.push(
            "optimizer did not report an acceptable stop; convergence is not certified".to_string(),
        );
    } else {
        parts.push("optimizer reported an acceptable stop".to_string());
    }

    match certificate.status {
        FitStatus::ConvergedInterior => {
            parts.push("solution is interior to the theta bounds".to_string());
        }
        FitStatus::ConvergedBoundary => {
            status = max_status(status, AuditReportStatus::Info);
            parts.push(
                "solution is on a parameter boundary; this is not by itself an optimizer failure"
                    .to_string(),
            );
        }
        FitStatus::ConvergedReducedRank => {
            status = max_status(status, AuditReportStatus::Info);
            parts.push(
                "effective covariance is reduced rank; unsupported directions are weakly identified, not proof of zero population variance"
                    .to_string(),
            );
        }
        FitStatus::ConvergedPenalised => {
            status = max_status(status, AuditReportStatus::Info);
            parts.push(
                "fit is penalised; it should not be read as an ordinary maximum-likelihood estimate"
                    .to_string(),
            );
        }
        FitStatus::NotIdentifiable => {
            status = max_status(status, AuditReportStatus::Warning);
            parts.push("model is not identifiable under the current contract".to_string());
        }
        FitStatus::NotOptimized => {
            status = max_status(status, AuditReportStatus::Warning);
            parts.push("optimization did not produce a certified fitted optimum".to_string());
        }
        FitStatus::NotAssessed => {
            status = max_status(status, AuditReportStatus::NotAssessed);
            parts.push("optimizer certificate has not been assessed".to_string());
        }
    }

    match &certificate.evidence.gradient.method {
        super::audit::EvidenceMethod::NotAvailable { reason }
        | super::audit::EvidenceMethod::NotAssessed { reason } => {
            status = max_status(status, AuditReportStatus::NotAssessed);
            parts.push(format!("stationarity is not certified: {reason}"));
        }
        method => {
            parts.push(format!(
                "stationarity checked by {}",
                evidence_method_label(method)
            ));
        }
    }

    if let super::audit::EvidenceQuality::Failed { reason } =
        &certificate.evidence.certification_quality
    {
        let inspection_only = certificate.evidence.optimizer_stop.acceptable_stop;
        status = max_status(
            status,
            if inspection_only {
                AuditReportStatus::Info
            } else {
                AuditReportStatus::Warning
            },
        );
        if reason.contains("gradient") || reason.contains("KKT") {
            if inspection_only {
                parts.push(format!(
                    "post-hoc stationarity inspection did not pass but does not override optimizer convergence: {reason}"
                ));
            } else {
                parts.push(format!("stationarity is not certified: {reason}"));
            }
        } else {
            parts.push(format!("certification failed: {reason}"));
        }
    }

    match &certificate.evidence.hessian.quality {
        super::audit::EvidenceQuality::Failed { reason } => {
            let inspection_only = certificate.evidence.optimizer_stop.acceptable_stop;
            status = max_status(
                status,
                if inspection_only {
                    AuditReportStatus::Info
                } else {
                    AuditReportStatus::Warning
                },
            );
            if inspection_only {
                parts.push(format!(
                    "post-hoc Hessian inspection did not pass but does not override optimizer convergence: {reason}"
                ));
            } else {
                parts.push(format!(
                    "Hessian check failed or is flat; weak identification is possible: {reason}"
                ));
            }
        }
        super::audit::EvidenceQuality::Unavailable { reason }
        | super::audit::EvidenceQuality::NotAssessed { reason } => {
            status = max_status(status, AuditReportStatus::NotAssessed);
            parts.push(format!(
                "Hessian weak-identification check is unavailable: {reason}"
            ));
        }
        quality => {
            parts.push(format!(
                "Hessian evidence is {}",
                evidence_quality_label(quality)
            ));
        }
    }

    if let Some(verification) = &certificate.verification {
        match verification.status {
            super::audit::ConvergenceVerificationStatus::RestartAgrees
            | super::audit::ConvergenceVerificationStatus::OptimizerConsensus => {
                parts.push(
                    "bounded verification agrees with the fitted optimum; remaining warnings are more likely numerical or structural than optimizer instability"
                        .to_string(),
                );
            }
            super::audit::ConvergenceVerificationStatus::Fragile => {
                status = max_status(status, AuditReportStatus::Warning);
                parts.push(
                    "bounded verification is fragile; compare objectives and theta before routine inference"
                        .to_string(),
                );
            }
            super::audit::ConvergenceVerificationStatus::Unstable => {
                status = max_status(status, AuditReportStatus::Error);
                parts.push("bounded verification did not reproduce the fitted optimum".to_string());
            }
            super::audit::ConvergenceVerificationStatus::NotRun => {
                status = max_status(status, AuditReportStatus::NotAssessed);
                parts.push("bounded convergence verification was not run".to_string());
            }
        }
    } else {
        status = max_status(status, AuditReportStatus::NotAssessed);
        parts.push("bounded convergence verification was not run".to_string());
    }

    (status, parts.join("; "))
}

fn convergence_next_steps_line(
    certificate: &super::audit::OptimizerCertificate,
    diagnostics: &[Diagnostic],
) -> AuditReportLine {
    let mut kinds: Vec<NextActionKind> = Vec::new();

    if !certificate.evidence.optimizer_stop.acceptable_stop {
        kinds.push(NextActionKind::BudgetOrAlternate);
    }
    if certificate.verification.is_none() {
        kinds.push(NextActionKind::SuggestVerify);
    }
    if matches!(
        certificate.evidence.gradient.method,
        super::audit::EvidenceMethod::NotAvailable { .. }
            | super::audit::EvidenceMethod::NotAssessed { .. }
    ) && !derivative_inspection_skipped_by_regime(certificate)
    {
        kinds.push(NextActionKind::GateInferenceOnDerivative);
    }
    if matches!(
        certificate.evidence.hessian.quality,
        super::audit::EvidenceQuality::Unavailable { .. }
            | super::audit::EvidenceQuality::NotAssessed { .. }
    ) {
        kinds.push(NextActionKind::GateWeakIdentification);
    }
    if matches!(
        certificate.evidence.hessian.quality,
        super::audit::EvidenceQuality::Failed { .. }
    ) || matches!(
        certificate.evidence.certification_quality,
        super::audit::EvidenceQuality::Failed { .. }
    ) {
        kinds.push(NextActionKind::PredictorScalingOrSimplifyRe);
    }
    if matches!(
        certificate.status,
        FitStatus::ConvergedBoundary | FitStatus::ConvergedReducedRank
    ) {
        kinds.push(NextActionKind::InspectEffectiveCovariance);
    }
    if let Some(verification) = &certificate.verification {
        match verification.status {
            super::audit::ConvergenceVerificationStatus::RestartAgrees
            | super::audit::ConvergenceVerificationStatus::OptimizerConsensus => {
                kinds.push(NextActionKind::TreatAgreementAsReassuring);
            }
            super::audit::ConvergenceVerificationStatus::Fragile
            | super::audit::ConvergenceVerificationStatus::Unstable => {
                kinds.push(NextActionKind::CompareVerificationRuns);
            }
            super::audit::ConvergenceVerificationStatus::NotRun => {}
        }
    }

    // Structural overlay: pre-fit identifiability failures cannot be
    // fixed by optimizer tinkering. Suppress optimizer-tinkering
    // recommendations (issue lme4#120 lesson) and append the
    // highest-priority structural step.
    let structural = structural_findings(diagnostics);
    if !structural.is_empty() {
        kinds.retain(|kind| !kind.is_optimizer_tinkering());
    }

    let mut actions: Vec<String> = kinds.into_iter().map(|k| k.text().to_string()).collect();
    if let Some(primary) = structural.iter().min_by_key(|f| f.priority()) {
        actions.push(primary.next_step());
    }

    actions.sort();
    actions.dedup();

    AuditReportLine {
        label: "convergence next steps".to_string(),
        status: if actions.is_empty() {
            AuditReportStatus::Ok
        } else {
            action_status(certificate)
        },
        detail: if actions.is_empty() {
            "none beyond routine model checks".to_string()
        } else {
            actions.join(" | ")
        },
    }
}

fn derivative_inspection_skipped_by_regime(
    certificate: &super::audit::OptimizerCertificate,
) -> bool {
    match &certificate.evidence.gradient.method {
        super::audit::EvidenceMethod::NotAssessed { reason } => {
            reason.contains("boundary") || reason.contains("convergence_derivative_nparmax")
        }
        _ => false,
    }
}

fn action_status(certificate: &super::audit::OptimizerCertificate) -> AuditReportStatus {
    if matches!(
        certificate
            .verification
            .as_ref()
            .map(|verification| verification.status),
        Some(super::audit::ConvergenceVerificationStatus::Unstable)
    ) {
        AuditReportStatus::Error
    } else if !certificate.evidence.optimizer_stop.acceptable_stop
        || matches!(
            certificate
                .verification
                .as_ref()
                .map(|verification| verification.status),
            Some(super::audit::ConvergenceVerificationStatus::Fragile)
        )
    {
        AuditReportStatus::Warning
    } else if matches!(
        certificate.evidence.hessian.quality,
        super::audit::EvidenceQuality::Failed { .. }
    ) || matches!(
        certificate.evidence.certification_quality,
        super::audit::EvidenceQuality::Failed { .. }
    ) {
        AuditReportStatus::Info
    } else if certificate.verification.is_none()
        || matches!(
            certificate.evidence.gradient.method,
            super::audit::EvidenceMethod::NotAvailable { .. }
                | super::audit::EvidenceMethod::NotAssessed { .. }
        )
        || matches!(
            certificate.evidence.hessian.quality,
            super::audit::EvidenceQuality::Unavailable { .. }
                | super::audit::EvidenceQuality::NotAssessed { .. }
        )
    {
        AuditReportStatus::NotAssessed
    } else {
        AuditReportStatus::Info
    }
}

fn convergence_verification_line(
    certificate: &super::audit::OptimizerCertificate,
) -> AuditReportLine {
    let Some(verification) = &certificate.verification else {
        return AuditReportLine {
            label: "convergence verification".to_string(),
            status: AuditReportStatus::NotAssessed,
            detail: "not run; call verify_convergence() to compare bounded restarts and alternate optimizer fits"
                .to_string(),
        };
    };

    let agreeing = verification.runs.iter().filter(|run| run.agrees).count();
    AuditReportLine {
        label: "convergence verification".to_string(),
        status: convergence_verification_status(verification.status),
        detail: format!(
            "status={}; runs={}; agreeing={}; objective_tol={:.3e}; theta_tol={:.3e}; beta_tol={:.3e}; {}",
            convergence_verification_status_label(verification.status),
            verification.runs.len(),
            agreeing,
            verification.objective_tolerance,
            verification.theta_tolerance,
            verification.beta_tolerance,
            verification.message
        ),
    }
}

fn convergence_verification_status(
    status: super::audit::ConvergenceVerificationStatus,
) -> AuditReportStatus {
    match status {
        super::audit::ConvergenceVerificationStatus::NotRun => AuditReportStatus::NotAssessed,
        super::audit::ConvergenceVerificationStatus::RestartAgrees
        | super::audit::ConvergenceVerificationStatus::OptimizerConsensus => AuditReportStatus::Ok,
        super::audit::ConvergenceVerificationStatus::Fragile => AuditReportStatus::Warning,
        super::audit::ConvergenceVerificationStatus::Unstable => AuditReportStatus::Error,
    }
}

fn convergence_verification_status_label(
    status: super::audit::ConvergenceVerificationStatus,
) -> &'static str {
    match status {
        super::audit::ConvergenceVerificationStatus::NotRun => "not_run",
        super::audit::ConvergenceVerificationStatus::RestartAgrees => "restart_agrees",
        super::audit::ConvergenceVerificationStatus::OptimizerConsensus => "optimizer_consensus",
        super::audit::ConvergenceVerificationStatus::Fragile => "fragile",
        super::audit::ConvergenceVerificationStatus::Unstable => "unstable",
    }
}

fn evidence_method_label(method: &super::audit::EvidenceMethod) -> String {
    match method {
        super::audit::EvidenceMethod::Exact => "exact".to_string(),
        super::audit::EvidenceMethod::FiniteDifference => "finite_difference".to_string(),
        super::audit::EvidenceMethod::OptimizerReported => "optimizer_reported".to_string(),
        super::audit::EvidenceMethod::NotAvailable { reason } => {
            format!("not_available ({reason})")
        }
        super::audit::EvidenceMethod::NotAssessed { reason } => {
            format!("not_assessed ({reason})")
        }
    }
}

fn evidence_method_status(method: &super::audit::EvidenceMethod) -> AuditReportStatus {
    match method {
        super::audit::EvidenceMethod::Exact
        | super::audit::EvidenceMethod::FiniteDifference
        | super::audit::EvidenceMethod::OptimizerReported => AuditReportStatus::Ok,
        super::audit::EvidenceMethod::NotAvailable { .. }
        | super::audit::EvidenceMethod::NotAssessed { .. } => AuditReportStatus::NotAssessed,
    }
}

fn evidence_quality_label(quality: &super::audit::EvidenceQuality) -> String {
    match quality {
        super::audit::EvidenceQuality::Certified => "certified".to_string(),
        super::audit::EvidenceQuality::Approximate { reason } => {
            format!("approximate ({reason})")
        }
        super::audit::EvidenceQuality::Unavailable { reason } => {
            format!("unavailable ({reason})")
        }
        super::audit::EvidenceQuality::NotAssessed { reason } => {
            format!("not_assessed ({reason})")
        }
        super::audit::EvidenceQuality::Failed { reason } => format!("failed ({reason})"),
    }
}

fn evidence_quality_detail(quality: &super::audit::EvidenceQuality) -> String {
    evidence_quality_label(quality)
}

fn evidence_quality_status(quality: &super::audit::EvidenceQuality) -> AuditReportStatus {
    match quality {
        super::audit::EvidenceQuality::Certified => AuditReportStatus::Ok,
        super::audit::EvidenceQuality::Approximate { .. } => AuditReportStatus::Info,
        super::audit::EvidenceQuality::Unavailable { .. }
        | super::audit::EvidenceQuality::NotAssessed { .. } => AuditReportStatus::NotAssessed,
        super::audit::EvidenceQuality::Failed { .. } => AuditReportStatus::Warning,
    }
}

fn hessian_evidence_status(certificate: &super::audit::OptimizerCertificate) -> AuditReportStatus {
    if certificate.evidence.optimizer_stop.acceptable_stop
        && matches!(
            certificate.evidence.hessian.quality,
            super::audit::EvidenceQuality::Failed { .. }
        )
    {
        AuditReportStatus::Info
    } else {
        evidence_quality_status(&certificate.evidence.hessian.quality)
    }
}

fn certification_quality_status(
    certificate: &super::audit::OptimizerCertificate,
) -> AuditReportStatus {
    if certificate.evidence.optimizer_stop.acceptable_stop
        && matches!(
            certificate.evidence.certification_quality,
            super::audit::EvidenceQuality::Failed { .. }
        )
    {
        AuditReportStatus::Info
    } else {
        evidence_quality_status(&certificate.evidence.certification_quality)
    }
}

fn inference_section(artifact: &CompiledModelArtifact) -> AuditReportSection {
    let mut lines = vec![match &artifact.model_boundary.inference_availability {
        InferenceAvailability::Available { method } => AuditReportLine {
            label: "finite-sample inference".to_string(),
            status: AuditReportStatus::Ok,
            detail: format!("available via {method}"),
        },
        InferenceAvailability::Unsupported { reason } => AuditReportLine {
            label: "finite-sample inference".to_string(),
            status: AuditReportStatus::NotAssessed,
            detail: reason.clone(),
        },
        InferenceAvailability::NotAssessed { reason } => {
            not_assessed_line("finite-sample inference", reason)
        }
    }];

    lines.push(match &artifact.model_boundary.covariance_derivatives {
        DerivativeAvailability::Available => AuditReportLine {
            label: "covariance derivatives".to_string(),
            status: AuditReportStatus::Ok,
            detail: "available".to_string(),
        },
        DerivativeAvailability::NotAvailable { reason } => AuditReportLine {
            label: "covariance derivatives".to_string(),
            status: AuditReportStatus::NotAssessed,
            detail: reason.clone(),
        },
        DerivativeAvailability::NotAssessed { reason } => {
            not_assessed_line("covariance derivatives", reason)
        }
    });

    AuditReportSection {
        title: "Inference".to_string(),
        lines,
    }
}

fn diagnostics_section(artifact: &CompiledModelArtifact) -> AuditReportSection {
    let diagnostics = report_diagnostics(artifact);
    if diagnostics.is_empty() {
        return AuditReportSection {
            title: "Diagnostics".to_string(),
            lines: vec![AuditReportLine {
                label: "diagnostics".to_string(),
                status: AuditReportStatus::Ok,
                detail: "none".to_string(),
            }],
        };
    }

    AuditReportSection {
        title: "Diagnostics".to_string(),
        lines: diagnostics
            .iter()
            .map(|diagnostic| AuditReportLine {
                label: diagnostic_code_label(&diagnostic.code).to_string(),
                status: diagnostic_severity_status(diagnostic.severity),
                detail: diagnostic_detail(diagnostic),
            })
            .collect(),
    }
}

fn report_diagnostics(artifact: &CompiledModelArtifact) -> Vec<Diagnostic> {
    let mut diagnostics = artifact.diagnostics.clone();
    if let Some(certificate) = &artifact.optimizer_certificate {
        for diagnostic in &certificate.diagnostics {
            let duplicate = diagnostics.iter().any(|existing| {
                existing.code == diagnostic.code
                    && existing.message == diagnostic.message
                    && existing.affected_terms == diagnostic.affected_terms
            });
            if !duplicate {
                diagnostics.push(diagnostic.clone());
            }
        }
    }
    diagnostics
}

fn max_status(left: AuditReportStatus, right: AuditReportStatus) -> AuditReportStatus {
    if status_rank(right) > status_rank(left) {
        right
    } else {
        left
    }
}

fn effective_covariance_detail(summary: &super::artifact::EffectiveCovarianceSummary) -> String {
    let supported = format_directions("supported direction", &summary.directions);
    let unsupported = format_directions("unsupported direction", &summary.unsupported_directions);
    let mut parts = vec![
        format!(
            "requested rank {}, supported rank {}",
            summary.requested_rank, summary.supported_rank
        ),
        format!("basis={}", summary.requested_basis.join(", ")),
    ];

    if !supported.is_empty() {
        parts.push(supported);
    }
    if !unsupported.is_empty() {
        parts.push(unsupported);
    }
    if !summary.inference_consequence.is_empty() {
        parts.push(format!(
            "inference consequence: {}",
            summary.inference_consequence
        ));
    }
    if let Some(submodel) = &summary.interpretable_submodel {
        parts.push(format_interpretable_submodel(submodel));
    }

    parts.join("; ")
}

fn format_interpretable_submodel(submodel: &super::artifact::InterpretableSubmodel) -> String {
    format!(
        "interpretable submodel suggestion: {}; dominant loadings={}; objective gap={:.3}; within tolerance={}",
        submodel.suggested_formula,
        format_dominant_loadings(&submodel.loadings_dominant),
        submodel.objective_gap,
        submodel.within_tolerance
    )
}

fn format_directions(
    prefix: &str,
    directions: &[super::artifact::SupportedCovarianceDirection],
) -> String {
    directions
        .iter()
        .map(|direction| {
            let loadings = if direction.user_scale_summary.is_empty() {
                format_loadings(&direction.loadings)
            } else {
                direction.user_scale_summary.clone()
            };
            let mut detail = format!("{prefix} {}: {loadings}", direction.label);
            if let Some(variance_explained) = direction.variance_explained {
                detail.push_str(&format!(" ({variance_explained:.3} variance explained)"));
            }
            detail
        })
        .collect::<Vec<_>>()
        .join("; ")
}

fn format_loadings(loadings: &[super::artifact::BasisLoading]) -> String {
    loadings
        .iter()
        .map(|loading| format!("{:.3}*{}", loading.loading, loading.basis))
        .collect::<Vec<_>>()
        .join(" + ")
}

fn format_dominant_loadings(loadings: &[super::artifact::DominantLoading]) -> String {
    loadings
        .iter()
        .map(|loading| format!("{:.3}*{}", loading.loading, loading.basis))
        .collect::<Vec<_>>()
        .join(" + ")
}

fn diagnostic_detail(diagnostic: &Diagnostic) -> String {
    let mut parts = vec![diagnostic.message.clone()];
    if !diagnostic.affected_terms.is_empty() {
        parts.push(format!("affected={}", diagnostic.affected_terms.join(", ")));
    }
    if !diagnostic.suggested_actions.is_empty() {
        parts.push(format!(
            "suggested={}",
            diagnostic.suggested_actions.join(" | ")
        ));
    }
    parts.join("; ")
}

fn not_assessed_line(label: &str, detail: &str) -> AuditReportLine {
    AuditReportLine {
        label: label.to_string(),
        status: AuditReportStatus::NotAssessed,
        detail: detail.to_string(),
    }
}

fn rank_status(status: RankStatus) -> AuditReportStatus {
    match status {
        RankStatus::FullRank => AuditReportStatus::Ok,
        RankStatus::RankDeficient => AuditReportStatus::Warning,
        RankStatus::NotAssessed => AuditReportStatus::NotAssessed,
    }
}

fn effective_rank_status(status: EffectiveRankStatus) -> AuditReportStatus {
    match status {
        EffectiveRankStatus::FullRank => AuditReportStatus::Ok,
        EffectiveRankStatus::ReducedRank => AuditReportStatus::Info,
        EffectiveRankStatus::NotAssessed => AuditReportStatus::NotAssessed,
    }
}

fn information_budget_status(status: InformationBudgetStatus) -> AuditReportStatus {
    match status {
        InformationBudgetStatus::Sufficient => AuditReportStatus::Ok,
        InformationBudgetStatus::WeaklySupported => AuditReportStatus::Info,
        InformationBudgetStatus::TooRich => AuditReportStatus::Warning,
        InformationBudgetStatus::NotAssessable => AuditReportStatus::NotAssessed,
    }
}

fn recommendation_status(diagnostics: &[Diagnostic]) -> AuditReportStatus {
    diagnostics
        .iter()
        .map(|diagnostic| diagnostic_severity_status(diagnostic.severity))
        .max_by_key(|status| status_rank(*status))
        .unwrap_or(AuditReportStatus::Info)
}

fn diagnostic_severity_status(severity: DiagnosticSeverity) -> AuditReportStatus {
    match severity {
        DiagnosticSeverity::Info => AuditReportStatus::Info,
        DiagnosticSeverity::Warning => AuditReportStatus::Warning,
        DiagnosticSeverity::Error => AuditReportStatus::Error,
    }
}

fn fit_status(status: FitStatus) -> AuditReportStatus {
    match status {
        FitStatus::ConvergedInterior => AuditReportStatus::Ok,
        FitStatus::ConvergedBoundary
        | FitStatus::ConvergedReducedRank
        | FitStatus::ConvergedPenalised => AuditReportStatus::Info,
        FitStatus::NotIdentifiable | FitStatus::NotOptimized => AuditReportStatus::Warning,
        FitStatus::NotAssessed => AuditReportStatus::NotAssessed,
    }
}

fn model_state_status(status: ModelStateStatus) -> AuditReportStatus {
    match status {
        ModelStateStatus::Requested
        | ModelStateStatus::Canonical
        | ModelStateStatus::Supported
        | ModelStateStatus::Fitted => AuditReportStatus::Ok,
        ModelStateStatus::AdvisoryChanges | ModelStateStatus::Reduced => AuditReportStatus::Info,
        ModelStateStatus::Refused => AuditReportStatus::Warning,
        ModelStateStatus::NotAssessed => AuditReportStatus::NotAssessed,
    }
}

fn model_state_status_label(status: ModelStateStatus) -> &'static str {
    match status {
        ModelStateStatus::Requested => "requested",
        ModelStateStatus::Canonical => "canonical",
        ModelStateStatus::Supported => "supported",
        ModelStateStatus::AdvisoryChanges => "advisory_changes",
        ModelStateStatus::Refused => "refused",
        ModelStateStatus::Fitted => "fitted",
        ModelStateStatus::Reduced => "reduced",
        ModelStateStatus::NotAssessed => "not_assessed",
    }
}

fn status_rank(status: AuditReportStatus) -> u8 {
    match status {
        AuditReportStatus::Ok => 0,
        AuditReportStatus::Info => 1,
        AuditReportStatus::NotAssessed => 2,
        AuditReportStatus::Warning => 3,
        AuditReportStatus::Error => 4,
    }
}

fn status_label(status: AuditReportStatus) -> &'static str {
    match status {
        AuditReportStatus::Ok => "OK",
        AuditReportStatus::Info => "INFO",
        AuditReportStatus::Warning => "WARNING",
        AuditReportStatus::Error => "ERROR",
        AuditReportStatus::NotAssessed => "NOT CHECKED",
    }
}

fn option_usize(value: Option<usize>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

fn option_f64(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.2}"))
        .unwrap_or_else(|| "unknown".to_string())
}

fn snake_status_budget(status: InformationBudgetStatus) -> &'static str {
    match status {
        InformationBudgetStatus::Sufficient => "sufficient",
        InformationBudgetStatus::WeaklySupported => "weakly_supported",
        InformationBudgetStatus::TooRich => "too_rich",
        InformationBudgetStatus::NotAssessable => "not_assessable",
    }
}

fn model_kind_label(kind: ModelKind) -> &'static str {
    match kind {
        ModelKind::LinearMixedModel => "linear_mixed_model",
        ModelKind::GeneralizedLinearMixedModel => "generalized_linear_mixed_model",
    }
}

fn objective_approximation_label(approximation: &ObjectiveApproximation) -> String {
    match approximation {
        ObjectiveApproximation::ExactGaussian => "exact_gaussian".to_string(),
        ObjectiveApproximation::Pirls => "pirls".to_string(),
        ObjectiveApproximation::Laplace { inner } => format!("laplace(inner={inner})"),
        ObjectiveApproximation::AdaptiveGaussHermite { n_points } => match n_points {
            Some(n_points) => format!("adaptive_gauss_hermite(n_points={n_points})"),
            None => "adaptive_gauss_hermite".to_string(),
        },
        ObjectiveApproximation::NotAssessed => "not_assessed".to_string(),
    }
}

fn optimizer_certificate_scope_label(scope: OptimizerCertificateScope) -> &'static str {
    match scope {
        OptimizerCertificateScope::ExactObjective => "exact_objective",
        OptimizerCertificateScope::ApproximatedObjective => "approximated_objective",
        OptimizerCertificateScope::NotAssessed => "not_assessed",
    }
}

fn diagnostic_code_label(code: &DiagnosticCode) -> &'static str {
    match code {
        DiagnosticCode::FormulaCanonicalized => "formula_canonicalized",
        DiagnosticCode::FormulaCanonicalizationUnsupported => {
            "formula_canonicalization_unsupported"
        }
        DiagnosticCode::DuplicateRandomTerm => "duplicate_random_term",
        DiagnosticCode::ConflictingCovariance => "conflicting_covariance",
        DiagnosticCode::CrossingLikelyUnintended => "crossing_likely_unintended",
        DiagnosticCode::FixedEffectColumnMissing => "fixed_effect_column_missing",
        DiagnosticCode::FixedEffectRankDeficient => "fixed_effect_rank_deficient",
        DiagnosticCode::FixedEffectEmptyCell => "fixed_effect_empty_cell",
        DiagnosticCode::RandomSlopeWithoutIntercept => "random_slope_without_intercept",
        DiagnosticCode::FixedRandomRedundant => "fixed_random_redundant",
        DiagnosticCode::RepeatedUnitUnmodeled => "repeated_unit_unmodeled",
        DiagnosticCode::RandomSlopeUnsupported => "random_slope_unsupported",
        DiagnosticCode::RandomEffectFewLevels => "random_effect_few_levels",
        DiagnosticCode::CovarianceTooRich => "covariance_too_rich",
        DiagnosticCode::CovarianceReduced => "covariance_reduced",
        DiagnosticCode::BoundaryParameter => "boundary_parameter",
        DiagnosticCode::NearUnitRandomEffectCorrelation => "near_unit_random_effect_correlation",
        DiagnosticCode::NotIdentifiable => "not_identifiable",
        DiagnosticCode::OptimizerNotAssessed => "optimizer_not_assessed",
        DiagnosticCode::InferenceUnavailable => "inference_unavailable",
        DiagnosticCode::SerializationNotAssessed => "serialization_not_assessed",
        DiagnosticCode::Unsupported => "unsupported",
        DiagnosticCode::ScopeNote => "scope_note",
        DiagnosticCode::SupportNote => "support_note",
        DiagnosticCode::SyntaxExpansion => "syntax_expansion",
        DiagnosticCode::CovarianceAssumption => "covariance_assumption",
        DiagnosticCode::StructuralRefusal => "structural_refusal",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::{compile_formula_ir, CompiledModelArtifact};
    use crate::formula::parse_formula;
    use crate::model::data::DataFrame;

    fn small_grouped_data() -> DataFrame {
        let mut data = DataFrame::new();
        data.add_numeric("y", vec![1.0, 2.0, 3.0, 4.0]);
        data.add_numeric("x", vec![0.0, 1.0, 0.0, 1.0]);
        data.add_categorical(
            "subject",
            vec!["s1", "s1", "s2", "s2"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        );
        data
    }

    fn repeated_unmodeled_data() -> DataFrame {
        let mut data = DataFrame::new();
        data.add_numeric("y", vec![1.0, 2.0, 2.5, 3.5]);
        data.add_categorical(
            "condition",
            vec!["A", "B", "A", "B"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        );
        data.add_categorical(
            "subject",
            vec!["s1", "s1", "s2", "s2"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        );
        data
    }

    #[test]
    fn report_runs_on_unfitted_artifact() {
        let formula = parse_formula("y ~ x + (1 + x | subject)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let mut artifact = CompiledModelArtifact::new(formula.to_string(), semantic);
        artifact.attach_design_audit(&small_grouped_data());

        let report = ModelAuditReport::from_artifact(&artifact);
        let text = report.to_text();

        assert!(text.contains("Requested Model"));
        assert!(text.contains("Random Effects"));
        assert!(text.contains("Random-Effect Information Budget"));
        assert!(text.contains("Random Term Cards"));
        assert!(text.contains("Cross-Card Constraints"));
        assert!(text.contains("levels/param=0.67"));
        assert!(text.contains("total rows can be misleading"));
        assert!(text.contains("Policy Recommendations"));
        assert!(text.contains("Optimizer"));
        assert!(text.contains("model has not been fitted"));
    }

    #[test]
    fn report_surfaces_missing_dependence_paths() {
        let formula = parse_formula("y ~ condition").unwrap();
        let semantic = compile_formula_ir(&formula);
        let mut artifact = CompiledModelArtifact::new(formula.to_string(), semantic);
        artifact.attach_design_audit(&repeated_unmodeled_data());

        let report = ModelAuditReport::from_artifact(&artifact);
        let text = report.to_text();

        assert!(text.contains("Dependence Paths"));
        assert!(text.contains("missing paths [WARNING]"));
        assert!(text.contains("subject -> (1 | subject)"));
        assert!(text.contains("repeated_unit_unmodeled"));
    }

    #[test]
    fn report_round_trips_json() {
        let formula = parse_formula("y ~ x + (1 + x | subject)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let mut artifact = CompiledModelArtifact::new(formula.to_string(), semantic);
        artifact.attach_design_audit(&small_grouped_data());

        let report = ModelAuditReport::from_artifact(&artifact);
        let json = serde_json::to_string(&report).unwrap();
        let decoded: ModelAuditReport = serde_json::from_str(&json).unwrap();

        assert_eq!(report.schema_version, 2);
        assert_eq!(report.random_term_cards.len(), 1);
        assert_eq!(report.random_term_cards[0].term_id, "r0");
        assert_eq!(
            report.random_term_cards[0].design_support.group_levels,
            Some(2)
        );
        assert!(report.cross_card_constraints.is_empty());
        assert_eq!(decoded, report);
    }

    #[test]
    fn random_term_cards_use_semantic_role_origin_side_table() {
        let formula = parse_formula("y ~ x + (1 | subject)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let mut artifact = CompiledModelArtifact::new(formula.to_string(), semantic);
        artifact.attach_design_audit(&small_grouped_data());

        let role_origin = RoleOrigin {
            declared_by_user: true,
            observed_from_data: false,
            role: super::super::ir::GroupingRole::Item,
        };
        artifact
            .semantic_model
            .role_origins
            .insert("r0".to_string(), role_origin.clone());

        let report = ModelAuditReport::from_artifact(&artifact);
        assert_eq!(report.random_term_cards[0].role_origin, role_origin);
    }

    #[test]
    fn double_bar_and_split_blocks_have_structurally_identical_cards() {
        let double_bar_formula = parse_formula("y ~ x + (1 + x || subject)").unwrap();
        let mut double_bar = CompiledModelArtifact::new(
            double_bar_formula.to_string(),
            compile_formula_ir(&double_bar_formula),
        );
        double_bar.attach_design_audit(&small_grouped_data());

        let split_formula = parse_formula("y ~ x + (1 | subject) + (0 + x | subject)").unwrap();
        let mut split = CompiledModelArtifact::new(
            split_formula.to_string(),
            compile_formula_ir(&split_formula),
        );
        split.attach_design_audit(&small_grouped_data());

        let double_bar_report = ModelAuditReport::from_artifact(&double_bar);
        let split_report = ModelAuditReport::from_artifact(&split);

        assert_eq!(double_bar_report.random_term_cards.len(), 2);
        assert_eq!(split_report.random_term_cards.len(), 2);
        assert_eq!(
            cards_without_original_fragments(double_bar_report.random_term_cards.clone()),
            cards_without_original_fragments(split_report.random_term_cards.clone())
        );
        assert_eq!(double_bar_report.cross_card_constraints.len(), 1);
        assert_eq!(split_report.cross_card_constraints.len(), 1);
        assert_ne!(
            double_bar_report.cross_card_constraints[0].reason,
            split_report.cross_card_constraints[0].reason
        );
        assert!(double_bar_report.cross_card_constraints[0]
            .reason
            .contains("double-bar syntax"));
        assert!(split_report.cross_card_constraints[0]
            .reason
            .contains("separate random-effect blocks"));
        assert_eq!(
            constraints_without_reasons(double_bar_report.cross_card_constraints.clone()),
            constraints_without_reasons(split_report.cross_card_constraints.clone())
        );
    }

    #[test]
    fn random_term_card_wording_is_nonempty_single_sentence_and_non_moralizing() {
        let formula = parse_formula("y ~ x + (1 + x || subject)").unwrap();
        let mut artifact =
            CompiledModelArtifact::new(formula.to_string(), compile_formula_ir(&formula));
        artifact.attach_design_audit(&small_grouped_data());
        let report = ModelAuditReport::from_artifact(&artifact);

        for card in &report.random_term_cards {
            for block in &card.blocks {
                assert_clean_wording(&block.english);
            }
            for constraint in &card.implied_constraints {
                assert_clean_wording(&constraint.reason);
            }
        }
        for constraint in &report.cross_card_constraints {
            assert_clean_wording(&constraint.reason);
        }
    }

    fn cards_without_original_fragments(
        mut cards: Vec<super::super::random_term_card::RandomTermCard>,
    ) -> Vec<super::super::random_term_card::RandomTermCard> {
        for card in &mut cards {
            card.original_fragment.clear();
        }
        cards
    }

    fn constraints_without_reasons(
        mut constraints: Vec<super::super::random_term_card::CrossCardConstraint>,
    ) -> Vec<super::super::random_term_card::CrossCardConstraint> {
        for constraint in &mut constraints {
            constraint.reason.clear();
        }
        constraints
    }

    fn assert_clean_wording(text: &str) {
        assert!(!text.trim().is_empty());
        assert!(
            sentence_terminator_count(text) == 1 && text.trim_end().ends_with('.'),
            "wording must be one sentence with terminal punctuation: {text}"
        );
        let lower = text.to_ascii_lowercase();
        for forbidden in [
            "suggested starting model",
            "we recommend",
            "you should",
            "try ",
            "drop the random slope",
        ] {
            assert!(
                !lower.contains(forbidden),
                "wording contains forbidden phrase `{forbidden}`: {text}"
            );
        }
    }

    fn sentence_terminator_count(text: &str) -> usize {
        text.chars()
            .filter(|character| matches!(character, '.' | '?' | '!'))
            .count()
    }

    // ---------------------------------------------------------------
    // ConvergenceVerdict tests
    // ---------------------------------------------------------------

    use super::super::audit::{
        ConvergenceVerification, ConvergenceVerificationStatus, EvidenceMethod, EvidenceQuality,
        OptimizerCertificate,
    };
    use super::super::diagnostics::{DiagnosticCode, DiagnosticSeverity, DiagnosticStage};

    /// Build a baseline certificate at `ConvergedInterior` with acceptable
    /// stop, exact gradient evidence, and certified Hessian evidence. Tests
    /// mutate specific fields to exercise individual verdict branches.
    fn clean_certificate() -> OptimizerCertificate {
        let mut cert = OptimizerCertificate::not_assessed();
        cert.status = FitStatus::ConvergedInterior;
        cert.evidence.optimizer_stop.acceptable_stop = true;
        cert.evidence.optimizer_stop.return_code = Some("SUCCESS".to_string());
        cert.evidence.gradient.method = EvidenceMethod::Exact;
        cert.evidence.hessian.method = EvidenceMethod::Exact;
        cert.evidence.hessian.quality = EvidenceQuality::Certified;
        cert.evidence.certification_quality = EvidenceQuality::Certified;
        cert
    }

    fn certificate_with_verification(
        status: ConvergenceVerificationStatus,
    ) -> OptimizerCertificate {
        let mut cert = clean_certificate();
        cert.verification = Some(ConvergenceVerification {
            status,
            objective_tolerance: 1e-6,
            theta_tolerance: 1e-6,
            beta_tolerance: 1e-6,
            reference_objective: Some(0.0),
            reference_theta: vec![0.0],
            reference_beta: vec![0.0],
            reference_effective_ranks: Vec::new(),
            runs: Vec::new(),
            message: format!("{status:?}"),
        });
        cert
    }

    fn diag(code: DiagnosticCode, message: &str, terms: &[&str]) -> Diagnostic {
        Diagnostic::new(
            code,
            DiagnosticSeverity::Warning,
            DiagnosticStage::DesignAudit,
            message,
        )
        .with_affected_terms(terms.iter().map(|t| t.to_string()).collect())
    }

    fn row_saturated_diag(term: &str) -> Diagnostic {
        let mut d = diag(
            DiagnosticCode::CovarianceTooRich,
            "row-saturated random effect",
            &[term],
        );
        d.payload
            .insert("row_saturated".to_string(), serde_json::json!(true));
        d
    }

    #[test]
    fn verdict_clean_fit_with_verification_is_certified_clean() {
        let cert = certificate_with_verification(ConvergenceVerificationStatus::RestartAgrees);
        let v = ConvergenceVerdict::compose(&cert, &[]);
        assert_eq!(v.level, ConvergenceLevel::Certified);
        assert_eq!(v.source, ConvergenceSource::Clean);
        assert!(
            v.next_step.is_none(),
            "certified clean fits need no next step"
        );
        assert!(v.headline.contains("interior"));
        assert!(v.headline.contains("verification agrees"));
    }

    #[test]
    fn verdict_finite_difference_verified_fit_is_ok_not_certified() {
        let mut cert = certificate_with_verification(ConvergenceVerificationStatus::RestartAgrees);
        cert.evidence.gradient.method = EvidenceMethod::FiniteDifference;
        cert.evidence.hessian.method = EvidenceMethod::FiniteDifference;
        cert.evidence.certification_quality = EvidenceQuality::Approximate {
            reason: "finite-difference derivative evidence".to_string(),
        };

        let v = ConvergenceVerdict::compose(&cert, &[]);

        assert_eq!(v.level, ConvergenceLevel::Ok);
        assert_eq!(v.source, ConvergenceSource::Clean);
        assert!(v.next_step.is_none());
        assert!(v.headline.contains("derivative evidence is approximate"));
    }

    #[test]
    fn verdict_clean_fit_without_verification_is_ok_with_verify_hint() {
        let cert = clean_certificate();
        let v = ConvergenceVerdict::compose(&cert, &[]);
        assert_eq!(v.level, ConvergenceLevel::Ok);
        assert_eq!(v.source, ConvergenceSource::Clean);
        let next = v.next_step.expect("verify hint expected");
        assert!(next.contains("verify_convergence"));
    }

    #[test]
    fn verdict_boundary_fit_is_caution_optimizer() {
        let mut cert = clean_certificate();
        cert.status = FitStatus::ConvergedBoundary;
        let v = ConvergenceVerdict::compose(&cert, &[]);
        assert_eq!(v.level, ConvergenceLevel::Caution);
        assert_eq!(v.source, ConvergenceSource::Optimizer);
        let next = v.next_step.expect("boundary fits suggest a follow-up");
        assert!(next.contains("Effective Covariance") || next.contains("verify_convergence"));
    }

    #[test]
    fn verdict_weak_gradient_suggests_verify_convergence() {
        let mut cert = clean_certificate();
        cert.evidence.gradient.method = EvidenceMethod::NotAvailable {
            reason: "derivative-free optimizer".to_string(),
        };
        let v = ConvergenceVerdict::compose(&cert, &[]);
        assert_eq!(v.source, ConvergenceSource::Optimizer);
        let next = v.next_step.expect("weak gradient demands a follow-up");
        // BudgetOrAlternate doesn't fire (acceptable_stop=true), so the
        // most actionable item is SuggestVerify.
        assert!(
            next.contains("verify_convergence"),
            "expected verify_convergence hint, got: {next}"
        );
    }

    #[test]
    fn verdict_failed_hessian_is_caution_with_predictor_scaling() {
        let mut cert = clean_certificate();
        cert.evidence.hessian.quality = EvidenceQuality::Failed {
            reason: "singular Hessian".to_string(),
        };
        let v = ConvergenceVerdict::compose(&cert, &[]);
        assert_eq!(v.source, ConvergenceSource::Optimizer);
        // With acceptable_stop=true and verification absent, SuggestVerify
        // is the highest-priority single recommendation; Hessian failure
        // bumps the level to Caution.
        assert_eq!(v.level, ConvergenceLevel::Caution);
    }

    #[test]
    fn verdict_unacceptable_stop_is_failed() {
        let mut cert = clean_certificate();
        cert.evidence.optimizer_stop.acceptable_stop = false;
        cert.evidence.optimizer_stop.return_code = Some("MAXEVAL_REACHED".to_string());
        let v = ConvergenceVerdict::compose(&cert, &[]);
        assert_eq!(v.level, ConvergenceLevel::Failed);
        assert_eq!(v.source, ConvergenceSource::Optimizer);
        let next = v.next_step.expect("unacceptable stop demands an action");
        assert!(next.contains("budget") || next.contains("alternate optimizer"));
    }

    #[test]
    fn verdict_row_saturated_re_is_structural_failed() {
        let cert = clean_certificate();
        let diags = vec![row_saturated_diag("(1 + x | g)")];
        let v = ConvergenceVerdict::compose(&cert, &diags);
        assert_eq!(v.level, ConvergenceLevel::Failed);
        assert_eq!(v.source, ConvergenceSource::Structural);
        assert!(v.headline.contains("structural"));
        assert!(v.headline.contains("row-saturated"));
        let next = v
            .next_step
            .expect("structural failure must surface an action");
        assert!(
            !next.contains("increase optimizer budget"),
            "must not suggest optimizer tinkering on structural failure: {next}"
        );
        assert!(
            !next.contains("verify_convergence"),
            "must not suggest verify_convergence on structural failure: {next}"
        );
        assert!(next.contains("drop") || next.contains("split") || next.contains("treat"));
        assert!(next.contains("optimizer tuning will not help"));
    }

    #[test]
    fn verdict_separation_is_structural_directs_to_penalty() {
        let cert = clean_certificate();
        let diags = vec![diag(
            DiagnosticCode::NotIdentifiable,
            "complete separation detected on predictor x",
            &[],
        )];
        let v = ConvergenceVerdict::compose(&cert, &diags);
        assert_eq!(v.level, ConvergenceLevel::Failed);
        assert_eq!(v.source, ConvergenceSource::Structural);
        let next = v.next_step.expect("separation must surface an action");
        assert!(
            next.to_lowercase().contains("firth") || next.to_lowercase().contains("penalised"),
            "expected Firth/penalised hint, got: {next}"
        );
    }

    #[test]
    fn verdict_fixed_rank_deficient_is_structural() {
        let cert = clean_certificate();
        let diags = vec![diag(
            DiagnosticCode::FixedEffectRankDeficient,
            "X has rank 2 of 3",
            &[],
        )];
        let v = ConvergenceVerdict::compose(&cert, &diags);
        assert_eq!(v.level, ConvergenceLevel::Failed);
        assert_eq!(v.source, ConvergenceSource::Structural);
        let next = v.next_step.expect("rank-deficient design needs an action");
        assert!(next.contains("rank-deficient") || next.contains("redundant"));
    }

    #[test]
    fn verdict_mixed_picks_structural_next_step() {
        let mut cert = clean_certificate();
        cert.status = FitStatus::ConvergedBoundary; // optimizer-side caution
        let diags = vec![row_saturated_diag("(1 + x | g)")];
        let v = ConvergenceVerdict::compose(&cert, &diags);
        // Structural always promotes to Failed (issue #120 lesson).
        assert_eq!(v.level, ConvergenceLevel::Failed);
        assert_eq!(v.source, ConvergenceSource::Mixed);
        // Headline should mention both axes.
        assert!(v.headline.contains("structural"));
        assert!(v.headline.contains("optimizer"));
        // Next step is structural (not boundary advice).
        let next = v.next_step.expect("mixed must surface a structural action");
        assert!(next.contains("optimizer tuning will not help"));
    }

    #[test]
    fn verdict_for_unfitted_artifact_is_not_assessed() {
        let v = ConvergenceVerdict::for_unfitted();
        assert_eq!(v.level, ConvergenceLevel::NotAssessed);
        assert_eq!(v.source, ConvergenceSource::NotAssessed);
        assert!(v.headline.contains("not fitted"));
        assert!(v.next_step.as_deref().unwrap_or("").contains(".fit()"));
    }

    #[test]
    fn verdict_round_trips_json() {
        let cert = certificate_with_verification(ConvergenceVerificationStatus::RestartAgrees);
        let v = ConvergenceVerdict::compose(&cert, &[]);
        let json = serde_json::to_string(&v).unwrap();
        let decoded: ConvergenceVerdict = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, v);
    }

    #[test]
    fn next_steps_line_suppresses_optimizer_tinkering_on_structural_finding() {
        // A non-acceptable optimizer stop normally emits "increase optimizer
        // budget"; a structural-source finding must suppress it.
        let mut cert = clean_certificate();
        cert.evidence.optimizer_stop.acceptable_stop = false;
        cert.evidence.gradient.method = EvidenceMethod::NotAvailable {
            reason: "derivative-free path".to_string(),
        };
        let diags = vec![row_saturated_diag("(1 + x | g)")];
        let line = convergence_next_steps_line(&cert, &diags);
        assert!(
            !line.detail.contains("increase optimizer budget"),
            "structural finding should suppress 'increase optimizer budget': {}",
            line.detail
        );
        assert!(
            !line.detail.contains("run verify_convergence()"),
            "structural finding should suppress verify_convergence hint: {}",
            line.detail
        );
        assert!(
            !line.detail.contains("derivative-backed"),
            "structural finding should suppress derivative-gating hint: {}",
            line.detail
        );
        assert!(
            line.detail.contains("optimizer tuning will not help"),
            "structural action must be present: {}",
            line.detail
        );
    }

    #[test]
    fn next_steps_line_preserves_optimizer_actions_when_no_structural_finding() {
        // Reduced-rank fits trigger optimizer-side recommendations and must
        // be preserved bit-identically — they are NOT structural.
        let mut cert = clean_certificate();
        cert.status = FitStatus::ConvergedReducedRank;
        let line = convergence_next_steps_line(&cert, &[]);
        // SuggestVerify still fires (no verification on this certificate),
        // and the boundary/reduced-rank hint must be present.
        assert!(line.detail.contains("Effective Covariance"));
        assert!(line.detail.contains("verify_convergence"));
    }
}
