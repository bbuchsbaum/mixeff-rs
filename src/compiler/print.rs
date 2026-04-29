//! Default print and explicit drilldowns for compiled-model artifacts.
//!
//! The compiler/audit surface is rich — `ModelExplanation`,
//! `ModelAuditReport`, `model_state_summary`, `changes`,
//! `covariance_parameter_traces` — but the **default** print of a
//! fitted model must not unload all of it on the user. Per
//! [PRD § 15](../../docs/compiler_contract_v0_prd.md), the default
//! output is a single canonical summary, with the heavy reports kept
//! one explicit method call away.
//!
//! This module provides:
//!
//! - [`ModelPrint`] — the compact default summary that
//!   `LinearMixedModel`/`GeneralizedLinearMixedModel`'s `Display`
//!   impl produces. Carries fit status, the requested formula, the
//!   effective formula when it differs, the top few diagnostics,
//!   inference availability, fit mode, and a one-liner naming the
//!   available drilldowns.
//! - [`ParameterizationDrilldown`] — the explicit drilldown for the
//!   source-to-fitted parameterization trace. Wraps the artifact's
//!   `covariance_parameter_traces` (source syntax, semantic term,
//!   expanded basis, ThetaMap, `Lambda`, `parmap`, VarCorr entries)
//!   and renders them in a stable text form.
//!
//! The other drilldowns named in PRD § 15 already have first-class
//! types/displays elsewhere:
//!
//! - `explain_model()` → [`super::explain::ModelExplanation`]
//! - `audit()` → [`super::report::ModelAuditReport`]
//! - `changes()` → [`super::artifact::CompiledModelArtifact::changes`]

use std::fmt;

use std::collections::BTreeMap;

use super::artifact::{
    CompiledModelArtifact, CovarianceParameterTrace, FitMode, InferenceAvailability, ModelKind,
};
use super::diagnostics::{Diagnostic, DiagnosticSeverity, FitStatus};
use super::report::{ConvergenceLevel, ConvergenceVerdict};

/// Maximum number of top diagnostics shown in [`ModelPrint`].
///
/// Tighter than the audit report's full list — the default print is a
/// pointer, not a transcript. Diagnostics beyond this are visible via
/// `audit_report()` (and explicitly counted in the print's overflow
/// line).
pub const MODEL_PRINT_TOP_DIAGNOSTICS: usize = 3;

/// Compact default summary of a compiled (or fitted) model artifact.
///
/// Constructed via [`ModelPrint::from_artifact`]. `Display` renders
/// the canonical short form referenced by PRD § 15: one fit-status
/// line, formulas (showing only the effective form when it differs
/// from the requested one), a small number of top diagnostics, an
/// inference-availability line, and a drilldowns pointer.
#[derive(Debug, Clone)]
pub struct ModelPrint {
    /// LMM vs GLMM, lifted from the artifact's `ModelBoundary`.
    pub model_kind: ModelKind,
    /// Top-level fit status from the optimizer certificate, or `None`
    /// when the artifact has not been fitted yet.
    pub fit_status: Option<FitStatus>,
    /// Fit-mode boundary (Confirmatory / Exploratory / Predictive)
    /// derived from the artifact's `ReproducibilityRecord::fit_intent`.
    pub fit_mode: FitMode,
    /// The user-supplied formula string.
    pub requested_formula: String,
    /// The compiler-supported / effective formula string. `None` when
    /// the compiler did not have to rewrite the requested formula.
    pub effective_formula: Option<String>,
    /// First [`MODEL_PRINT_TOP_DIAGNOSTICS`] diagnostics, sorted with
    /// errors first, warnings next, info last; ties broken by source
    /// order so the same artifact prints stably across runs.
    pub top_diagnostics: Vec<Diagnostic>,
    /// Total diagnostic count on the artifact. Used to tell the reader
    /// when the top-N has been truncated.
    pub diagnostic_total: usize,
    /// Inference availability copied from the artifact's
    /// `ModelBoundary`.
    pub inference: InferenceAvailability,
    /// Compact convergence verdict — combines the optimizer certificate
    /// with structural pre-fit diagnostics. Always populated, even for
    /// unfitted artifacts (in which case it reports `not assessed`). See
    /// [`ConvergenceVerdict`] for the typed projection backing the
    /// one-line render.
    pub verdict: ConvergenceVerdict,
}

impl ModelPrint {
    /// Build a [`ModelPrint`] from a compiled artifact.
    pub fn from_artifact(artifact: &CompiledModelArtifact) -> Self {
        let fit_status = artifact.optimizer_certificate.as_ref().map(|c| c.status);
        let mut diagnostics: Vec<Diagnostic> = artifact.diagnostics.clone();
        diagnostics.sort_by_key(|d| diagnostic_priority(d.severity));
        let diagnostic_total = diagnostics.len();
        let top_diagnostics = diagnostics
            .into_iter()
            .take(MODEL_PRINT_TOP_DIAGNOSTICS)
            .collect();
        let effective_formula = artifact
            .effective_formula
            .as_ref()
            .filter(|effective| effective.as_str() != artifact.requested_formula.as_str())
            .cloned();
        Self {
            model_kind: artifact.model_boundary.model_kind,
            fit_status,
            fit_mode: artifact.reproducibility.fit_intent.mode(),
            requested_formula: artifact.requested_formula.clone(),
            effective_formula,
            top_diagnostics,
            diagnostic_total,
            inference: artifact.model_boundary.inference_availability.clone(),
            verdict: ConvergenceVerdict::for_artifact(artifact),
        }
    }
}

impl fmt::Display for ModelPrint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status_label = match self.fit_status {
            None => "not fitted".to_string(),
            Some(status) => format!("{:?}", status),
        };
        writeln!(
            f,
            "{:?} [{}]: {:?}",
            self.model_kind, status_label, self.fit_mode
        )?;
        if let Some(anchor) = self.verdict.primary_doc_anchor() {
            writeln!(
                f,
                "  convergence: {} — {} ({})",
                self.verdict.level.as_str(),
                self.verdict.headline,
                anchor
            )?;
        } else {
            writeln!(
                f,
                "  convergence: {} — {}",
                self.verdict.level.as_str(),
                self.verdict.headline
            )?;
        }
        if let Some(next_step) = &self.verdict.next_step {
            if !matches!(self.verdict.level, ConvergenceLevel::Certified) {
                writeln!(f, "    next: {}", next_step)?;
            }
        }
        writeln!(f, "  formula: {}", self.requested_formula)?;
        if let Some(effective) = &self.effective_formula {
            writeln!(f, "  effective: {}", effective)?;
        }
        if self.top_diagnostics.is_empty() {
            writeln!(f, "  diagnostics: none")?;
        } else {
            writeln!(
                f,
                "  diagnostics ({} shown of {}):",
                self.top_diagnostics.len(),
                self.diagnostic_total
            )?;
            for diagnostic in &self.top_diagnostics {
                writeln!(
                    f,
                    "    [{}] {:?}: {}",
                    severity_label(diagnostic.severity),
                    diagnostic.code,
                    diagnostic.message
                )?;
            }
        }
        writeln!(f, "  inference: {}", inference_label(&self.inference))?;
        write!(
            f,
            "  drilldowns: explain_model(), audit_report(), parameterization(), changes()"
        )
    }
}

fn diagnostic_priority(severity: DiagnosticSeverity) -> u8 {
    match severity {
        DiagnosticSeverity::Error => 0,
        DiagnosticSeverity::Warning => 1,
        DiagnosticSeverity::Info => 2,
    }
}

fn severity_label(severity: DiagnosticSeverity) -> &'static str {
    match severity {
        DiagnosticSeverity::Error => "error",
        DiagnosticSeverity::Warning => "warning",
        DiagnosticSeverity::Info => "info",
    }
}

fn inference_label(inference: &InferenceAvailability) -> String {
    match inference {
        InferenceAvailability::Available { method } => format!("available ({})", method),
        InferenceAvailability::NotAssessed { reason } => format!("not assessed ({})", reason),
        InferenceAvailability::Unsupported { reason } => format!("unsupported ({})", reason),
    }
}

/// Source-to-fitted parameterization drilldown.
///
/// Wraps the artifact's [`CovarianceParameterTrace`] vector. Each
/// trace records, per random term, the source syntax, the resolved
/// semantic random-term label, the basis column names, the
/// `theta_slots`, the `lambda_slots`, the `parmap_entries`, and the
/// `varcorr_entries`. Rendering is one term per block, indented for
/// readability; consumers who need machine-readable access should
/// reach into [`CompiledModelArtifact::covariance_parameter_traces`]
/// directly.
#[derive(Debug, Clone)]
pub struct ParameterizationDrilldown {
    pub requested_formula: String,
    pub effective_formula: Option<String>,
    pub traces: Vec<CovarianceParameterTrace>,
}

impl ParameterizationDrilldown {
    /// Build a [`ParameterizationDrilldown`] from a compiled artifact.
    pub fn from_artifact(artifact: &CompiledModelArtifact) -> Self {
        Self {
            requested_formula: artifact.requested_formula.clone(),
            effective_formula: artifact.effective_formula.clone(),
            traces: artifact.covariance_parameter_traces.clone(),
        }
    }
}

impl fmt::Display for ParameterizationDrilldown {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Parameterization:")?;
        writeln!(f, "  requested formula: {}", self.requested_formula)?;
        if let Some(effective) = &self.effective_formula {
            writeln!(f, "  effective formula: {}", effective)?;
        }
        if self.traces.is_empty() {
            writeln!(f, "  random terms: none")?;
            return Ok(());
        }

        // The artifact carries one trace row per (term, theta-slot) pair;
        // group by term_id so the drilldown reads as one block per random
        // term while preserving source order via the encounter index.
        let mut groups: BTreeMap<usize, Vec<&CovarianceParameterTrace>> = BTreeMap::new();
        let mut term_first_seen: BTreeMap<String, usize> = BTreeMap::new();
        for trace in &self.traces {
            let next_index = term_first_seen.len();
            let order = *term_first_seen
                .entry(trace.term_id.clone())
                .or_insert(next_index);
            groups.entry(order).or_default().push(trace);
        }

        writeln!(f, "  random terms ({}):", groups.len())?;
        for (order, traces) in &groups {
            let header = traces.first().expect("non-empty group");
            writeln!(f, "    [{}] {}", order, header.term_id)?;
            writeln!(f, "      group: {}", header.group)?;
            writeln!(f, "      source: {}", header.source_syntax)?;
            writeln!(f, "      user basis: {}", header.user_basis.join(", "))?;
            if header.optimizer_basis != header.user_basis {
                writeln!(
                    f,
                    "      optimizer basis: {}",
                    header.optimizer_basis.join(", ")
                )?;
            }
            writeln!(f, "      covariance family: {:?}", header.covariance_family)?;
            writeln!(f, "      slots ({}):", traces.len())?;
            for trace in traces {
                let theta_value = format_optional(trace.theta.value);
                let theta_idx = trace
                    .theta
                    .global_index
                    .map(|i| i.to_string())
                    .unwrap_or_else(|| "-".to_string());
                writeln!(
                    f,
                    "        θ[{}] {} {:?} {:?} = {}",
                    theta_idx,
                    trace.theta.name,
                    trace.theta.constraint,
                    trace.theta.status,
                    theta_value
                )?;
                let lambda_value = format_optional(trace.lambda.value);
                writeln!(
                    f,
                    "          Λ[{}, {}] ({} × {}) = {}",
                    trace.lambda.row,
                    trace.lambda.col,
                    trace.lambda.row_basis,
                    trace.lambda.col_basis,
                    lambda_value
                )?;
                if let Some(parmap) = &trace.parmap_entry {
                    let agreement = if parmap.matches_theta_map {
                        "ok"
                    } else {
                        "mismatch"
                    };
                    writeln!(
                        f,
                        "          parmap → term {} Λ[{}, {}] ({})",
                        parmap.term_index, parmap.lambda_row, parmap.lambda_col, agreement
                    )?;
                }
                for entry in &trace.varcorr_entries {
                    let value = format_optional(entry.value);
                    writeln!(
                        f,
                        "          VarCorr {:?} {} ({}) = {}",
                        entry.kind,
                        entry.label,
                        entry.basis.join(" × "),
                        value
                    )?;
                }
            }
            let _ = order;
        }
        Ok(())
    }
}

fn format_optional(value: Option<f64>) -> String {
    match value {
        Some(v) => format!("{v}"),
        None => "—".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::ir::compile_formula_ir;
    use crate::formula::parse_formula;

    fn sample_artifact() -> CompiledModelArtifact {
        let formula = parse_formula("y ~ 1 + x + (1 + x | g)").unwrap();
        let semantic = compile_formula_ir(&formula);
        CompiledModelArtifact::new("y ~ 1 + x + (1 + x | g)", semantic)
    }

    #[test]
    fn model_print_renders_compact_summary_for_unfitted_artifact() {
        let artifact = sample_artifact();
        let print = ModelPrint::from_artifact(&artifact);
        assert!(print.fit_status.is_none());
        let rendered = print.to_string();
        assert!(
            rendered.starts_with("LinearMixedModel [not fitted]:"),
            "rendered: {rendered}"
        );
        assert!(rendered.contains("formula: y ~ 1 + x + (1 + x | g)"));
        assert!(rendered.contains("drilldowns:"));
    }

    #[test]
    fn model_print_truncates_diagnostics_to_top_n() {
        let mut artifact = sample_artifact();
        for index in 0..(MODEL_PRINT_TOP_DIAGNOSTICS + 2) {
            artifact.diagnostics.push(Diagnostic::new(
                super::super::diagnostics::DiagnosticCode::Unsupported,
                if index == 0 {
                    DiagnosticSeverity::Error
                } else {
                    DiagnosticSeverity::Warning
                },
                super::super::diagnostics::DiagnosticStage::DesignAudit,
                format!("synthetic diagnostic {index}"),
            ));
        }
        let print = ModelPrint::from_artifact(&artifact);
        assert_eq!(print.top_diagnostics.len(), MODEL_PRINT_TOP_DIAGNOSTICS);
        assert!(print.diagnostic_total > MODEL_PRINT_TOP_DIAGNOSTICS);
        // Errors must sort before warnings.
        assert!(matches!(
            print.top_diagnostics[0].severity,
            DiagnosticSeverity::Error
        ));
        let rendered = print.to_string();
        assert!(rendered.contains(&format!(
            "diagnostics ({} shown of {})",
            MODEL_PRINT_TOP_DIAGNOSTICS, print.diagnostic_total
        )));
    }

    #[test]
    fn model_print_shows_effective_formula_only_when_it_differs() {
        let mut artifact = sample_artifact();
        artifact.effective_formula = Some(artifact.requested_formula.clone());
        let print = ModelPrint::from_artifact(&artifact);
        assert!(
            print.effective_formula.is_none(),
            "should suppress identical effective formula"
        );

        artifact.effective_formula = Some("y ~ 1 + x + (1 | g)".to_string());
        let print = ModelPrint::from_artifact(&artifact);
        assert_eq!(
            print.effective_formula.as_deref(),
            Some("y ~ 1 + x + (1 | g)")
        );
        let rendered = print.to_string();
        assert!(rendered.contains("effective: y ~ 1 + x + (1 | g)"));
    }

    #[test]
    fn model_print_emits_convergence_line_for_unfitted_artifact() {
        let artifact = sample_artifact();
        let print = ModelPrint::from_artifact(&artifact);
        assert!(print.fit_status.is_none());
        let rendered = print.to_string();
        assert!(
            rendered.contains("convergence: not assessed"),
            "expected 'convergence: not assessed' line, got:\n{rendered}"
        );
        // Unfitted artifact should expose a `next: ...` line pointing at
        // .fit() — not certified, so the `next:` line is emitted.
        assert!(
            rendered.contains("next: ") && rendered.contains(".fit()"),
            "expected next: line pointing at .fit(), got:\n{rendered}"
        );
    }

    #[test]
    fn model_print_structural_failure_renders_failed_with_next_step() {
        use super::super::audit::OptimizerCertificate;
        use super::super::diagnostics::{
            Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticStage,
        };

        let mut artifact = sample_artifact();
        let mut cert = OptimizerCertificate::not_assessed();
        cert.status = FitStatus::ConvergedInterior;
        cert.evidence.optimizer_stop.acceptable_stop = true;
        artifact.optimizer_certificate = Some(cert);

        let mut diag = Diagnostic::new(
            DiagnosticCode::CovarianceTooRich,
            DiagnosticSeverity::Warning,
            DiagnosticStage::DesignAudit,
            "row-saturated random effect",
        )
        .with_affected_terms(vec!["(1 + x | g)".to_string()]);
        diag.payload
            .insert("row_saturated".to_string(), serde_json::json!(true));
        artifact.diagnostics.push(diag);

        let print = ModelPrint::from_artifact(&artifact);
        let rendered = print.to_string();
        assert!(
            rendered.contains("convergence: failed"),
            "expected 'convergence: failed' line, got:\n{rendered}"
        );
        assert!(
            rendered.contains("structural"),
            "verdict headline should mention structural source, got:\n{rendered}"
        );
        assert!(
            rendered.contains("next: "),
            "structural failure must surface a 'next:' line, got:\n{rendered}"
        );
        assert!(
            rendered.contains("optimizer tuning will not help"),
            "structural next-step must call out that optimizer tuning won't help, got:\n{rendered}"
        );
        assert!(
            !rendered.contains("increase optimizer budget"),
            "structural source must not suggest optimizer tinkering, got:\n{rendered}"
        );
    }

    #[test]
    fn model_print_caution_boundary_renders_with_next_step() {
        use super::super::audit::{EvidenceMethod, EvidenceQuality, OptimizerCertificate};

        let mut artifact = sample_artifact();
        let mut cert = OptimizerCertificate::not_assessed();
        cert.status = FitStatus::ConvergedBoundary;
        cert.evidence.optimizer_stop.acceptable_stop = true;
        cert.evidence.gradient.method = EvidenceMethod::Exact;
        cert.evidence.hessian.method = EvidenceMethod::Exact;
        cert.evidence.hessian.quality = EvidenceQuality::Certified;
        cert.evidence.certification_quality = EvidenceQuality::Certified;
        artifact.optimizer_certificate = Some(cert);

        let print = ModelPrint::from_artifact(&artifact);
        let rendered = print.to_string();
        assert!(
            rendered.contains("convergence: caution"),
            "expected 'convergence: caution' line, got:\n{rendered}"
        );
        assert!(
            rendered.contains("next: "),
            "caution-level fit must surface a 'next:' line, got:\n{rendered}"
        );
    }

    #[test]
    fn model_print_certified_fit_omits_next_line() {
        use super::super::audit::{
            ConvergenceVerification, ConvergenceVerificationStatus, EvidenceMethod,
            EvidenceQuality, OptimizerCertificate,
        };

        let mut artifact = sample_artifact();
        let mut cert = OptimizerCertificate::not_assessed();
        cert.status = FitStatus::ConvergedInterior;
        cert.evidence.optimizer_stop.acceptable_stop = true;
        cert.evidence.gradient.method = EvidenceMethod::Exact;
        cert.evidence.hessian.method = EvidenceMethod::Exact;
        cert.evidence.hessian.quality = EvidenceQuality::Certified;
        cert.evidence.certification_quality = EvidenceQuality::Certified;
        cert.verification = Some(ConvergenceVerification {
            status: ConvergenceVerificationStatus::RestartAgrees,
            objective_tolerance: 1e-6,
            theta_tolerance: 1e-6,
            beta_tolerance: 1e-6,
            reference_objective: Some(0.0),
            reference_theta: Vec::new(),
            reference_beta: Vec::new(),
            reference_effective_ranks: Vec::new(),
            runs: Vec::new(),
            message: "restart agrees".to_string(),
        });
        artifact.optimizer_certificate = Some(cert);

        let print = ModelPrint::from_artifact(&artifact);
        let rendered = print.to_string();
        assert!(
            rendered.contains("convergence: certified"),
            "expected 'convergence: certified' line, got:\n{rendered}"
        );
        // Certified fits must not emit a `next:` continuation — the
        // verdict line stands alone.
        assert!(
            !rendered.contains("next: "),
            "certified fits must not emit a 'next:' line, got:\n{rendered}"
        );
    }

    #[test]
    fn parameterization_drilldown_renders_random_term_traces() {
        let artifact = sample_artifact();
        let drilldown = ParameterizationDrilldown::from_artifact(&artifact);
        assert!(
            !drilldown.traces.is_empty(),
            "expected at least one random-term trace"
        );
        let rendered = drilldown.to_string();
        assert!(rendered.starts_with("Parameterization:"));
        assert!(rendered.contains("requested formula:"));
        assert!(rendered.contains("random terms"));
        assert!(rendered.contains("slots ("));
        assert!(rendered.contains("θ["));
        assert!(rendered.contains("Λ["));
    }
}
