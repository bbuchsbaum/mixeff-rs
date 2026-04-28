use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use super::artifact::{
    CompiledModelArtifact, DerivativeAvailability, EffectiveRankStatus, InferenceAvailability,
    ModelKind, ModelStateStatus, ObjectiveApproximation, OptimizerCertificateScope,
};
use super::audit::{InformationBudgetStatus, RankStatus};
use super::diagnostics::{Diagnostic, DiagnosticCode, DiagnosticSeverity, FitStatus};

pub const MODEL_AUDIT_REPORT_SCHEMA: &str = "mixedmodels.model_audit_report";
pub const MODEL_AUDIT_REPORT_SCHEMA_VERSION: u32 = 1;

/// Stable user-facing summary of a compiled/fitted model artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelAuditReport {
    pub schema_name: String,
    pub schema_version: u32,
    pub requested_formula: String,
    pub sections: Vec<AuditReportSection>,
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
        status: evidence_quality_status(&certificate.evidence.hessian.quality),
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
        status: evidence_quality_status(&certificate.evidence.certification_quality),
        detail: evidence_quality_detail(&certificate.evidence.certification_quality),
    });
    lines.push(convergence_next_steps_line(certificate));

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
        status: if failed > 0 {
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
        status = max_status(status, AuditReportStatus::Warning);
        if reason.contains("gradient") || reason.contains("KKT") {
            parts.push(format!("stationarity is not certified: {reason}"));
        } else {
            parts.push(format!("certification failed: {reason}"));
        }
    }

    match &certificate.evidence.hessian.quality {
        super::audit::EvidenceQuality::Failed { reason } => {
            status = max_status(status, AuditReportStatus::Warning);
            parts.push(format!(
                "Hessian check failed or is flat; weak identification is possible: {reason}"
            ));
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
) -> AuditReportLine {
    let mut actions = Vec::new();

    if !certificate.evidence.optimizer_stop.acceptable_stop {
        actions.push("increase optimizer budget or try an alternate optimizer".to_string());
    }
    if certificate.verification.is_none() {
        actions.push(
            "run verify_convergence() to compare restart and alternate-optimizer agreement"
                .to_string(),
        );
    }
    if matches!(
        certificate.evidence.gradient.method,
        super::audit::EvidenceMethod::NotAvailable { .. }
            | super::audit::EvidenceMethod::NotAssessed { .. }
    ) {
        actions.push(
            "gate inference on derivative-backed or finite-difference stationarity evidence"
                .to_string(),
        );
    }
    if matches!(
        certificate.evidence.hessian.quality,
        super::audit::EvidenceQuality::Unavailable { .. }
            | super::audit::EvidenceQuality::NotAssessed { .. }
    ) {
        actions.push(
            "gate weak-identification claims until Hessian evidence is available".to_string(),
        );
    }
    if matches!(
        certificate.evidence.hessian.quality,
        super::audit::EvidenceQuality::Failed { .. }
    ) || matches!(
        certificate.evidence.certification_quality,
        super::audit::EvidenceQuality::Failed { .. }
    ) {
        actions.push(
            "scale predictors, simplify the random-effects structure, or collect more grouping levels"
                .to_string(),
        );
    }
    if matches!(
        certificate.status,
        FitStatus::ConvergedBoundary | FitStatus::ConvergedReducedRank
    ) {
        actions.push(
            "inspect Effective Covariance and consider diagonal covariance, a simpler random-effect term, or design_compiled policy"
                .to_string(),
        );
    }
    if let Some(verification) = &certificate.verification {
        match verification.status {
            super::audit::ConvergenceVerificationStatus::RestartAgrees
            | super::audit::ConvergenceVerificationStatus::OptimizerConsensus => {
                actions.push(
                    "treat optimizer-agreement evidence as reassuring, while keeping any boundary or rank caveats"
                        .to_string(),
                );
            }
            super::audit::ConvergenceVerificationStatus::Fragile
            | super::audit::ConvergenceVerificationStatus::Unstable => {
                actions.push(
                    "compare objectives, theta, and beta across verification runs".to_string(),
                );
            }
            super::audit::ConvergenceVerificationStatus::NotRun => {}
        }
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
            certificate.evidence.hessian.quality,
            super::audit::EvidenceQuality::Failed { .. }
        )
        || matches!(
            certificate.evidence.certification_quality,
            super::audit::EvidenceQuality::Failed { .. }
        )
        || matches!(
            certificate
                .verification
                .as_ref()
                .map(|verification| verification.status),
            Some(super::audit::ConvergenceVerificationStatus::Fragile)
        )
    {
        AuditReportStatus::Warning
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

        assert_eq!(decoded, report);
    }
}
