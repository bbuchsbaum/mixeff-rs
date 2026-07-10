use std::fmt;

use serde::{Deserialize, Serialize};

use crate::formula::Formula;

use super::diagnostics::Diagnostic;
use super::ir::{
    compile_formula_ir, CovarianceForm, InterceptPolicy, RandomCoefficientKind, RandomTermIr,
    SemanticModel,
};

/// User-facing explanation backed by semantic IR.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelExplanation {
    pub semantic_model: SemanticModel,
    pub sections: Vec<ExplanationSection>,
    pub diagnostics: Vec<Diagnostic>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExplanationSection {
    pub title: String,
    pub lines: Vec<String>,
}

pub fn explain_model(formula: &Formula) -> ModelExplanation {
    let semantic_model = compile_formula_ir(formula);
    ModelExplanation::from_semantic_model(semantic_model)
}

impl ModelExplanation {
    pub fn from_semantic_model(semantic_model: SemanticModel) -> Self {
        let mut sections = Vec::new();

        sections.push(ExplanationSection {
            title: "Fixed effects".to_string(),
            lines: semantic_model
                .fixed_terms
                .iter()
                .map(|term| match term.as_str() {
                    "1" => "The model estimates an overall intercept.".to_string(),
                    other => format!("`{other}` contributes to the population-average prediction."),
                })
                .collect(),
        });

        for term in &semantic_model.random_terms {
            let mut lines = vec![random_effect_summary(term), covariance_summary(term)];
            lines.extend(term.covariance_story.dependence.iter().cloned());
            lines.push(formula_detail(term));

            sections.push(ExplanationSection {
                title: format!("Random effect for {}", term.group.label()),
                lines,
            });
        }

        Self {
            diagnostics: semantic_model.diagnostics.clone(),
            semantic_model,
            sections,
        }
    }

    pub fn to_text(&self) -> String {
        self.to_string()
    }
}

fn random_effect_summary(term: &RandomTermIr) -> String {
    let group = term.group.label();
    let slopes = term
        .basis
        .iter()
        .filter(|basis| basis.kind != RandomCoefficientKind::Intercept)
        .map(|basis| format!("`{}`", basis.name))
        .collect::<Vec<_>>();
    match (term.intercept, slopes.as_slice()) {
        (InterceptPolicy::Included, []) => {
            format!("Each `{group}` level may have its own baseline value.")
        }
        (InterceptPolicy::Included, slopes) => format!(
            "Each `{group}` level may have its own baseline value and its own slope for {}.",
            slopes.join(", ")
        ),
        (InterceptPolicy::Omitted, []) => {
            format!("This term does not add a separate baseline value for each `{group}` level.")
        }
        (InterceptPolicy::Omitted, slopes) => format!(
            "Each `{group}` level may have its own slope for {}; this term does not add group-specific baselines.",
            slopes.join(", ")
        ),
    }
}

fn covariance_summary(term: &RandomTermIr) -> String {
    let group = term.group.label();
    match &term.covariance {
        CovarianceForm::Scalar => format!("The model estimates one between-`{group}` variance."),
        CovarianceForm::Diagonal => {
            "The group-specific coefficients vary independently in the fitted basis.".to_string()
        }
        CovarianceForm::Full => {
            "The group-specific coefficients may be correlated with one another.".to_string()
        }
        CovarianceForm::Structured { kind } => {
            format!("The group-specific coefficients use a structured {kind} covariance.")
        }
        CovarianceForm::ReducedRank { rank } => match rank {
            Some(rank) => format!("The group-specific covariance is limited to rank {rank}."),
            None => "The group-specific covariance uses a reduced-rank form.".to_string(),
        },
        CovarianceForm::Unsupported { reason } => {
            format!("This covariance form is not supported: {reason}.")
        }
    }
}

fn formula_detail(term: &RandomTermIr) -> String {
    let written = term.source_syntax.user_text();
    let canonical = &term.source_syntax.text;
    if written == canonical {
        format!("Formula detail: `{canonical}`.")
    } else {
        format!("Formula detail: written as `{written}`; expanded to `{canonical}`.")
    }
}

impl fmt::Display for ModelExplanation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (idx, section) in self.sections.iter().enumerate() {
            if idx > 0 {
                writeln!(f)?;
            }
            writeln!(f, "{}:", section.title)?;
            for line in &section.lines {
                writeln!(f, "  {line}")?;
            }
        }
        if !self.diagnostics.is_empty() {
            writeln!(f)?;
            writeln!(f, "Diagnostics:")?;
            for diagnostic in &self.diagnostics {
                writeln!(f, "  {:?}: {}", diagnostic.code, diagnostic.message)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formula::parse_formula;

    #[test]
    fn explanation_runs_without_fitting() {
        let formula = parse_formula("y ~ x + (1 + x | subject)").unwrap();
        let explanation = explain_model(&formula);
        let text = explanation.to_text();

        assert!(text.contains("Random effect for subject"));
        assert!(text.contains("own baseline value and its own slope for `x`"));
        assert!(text.contains("observations sharing subject are correlated"));
        assert!(!text.contains("Random effect r0"));
    }

    #[test]
    fn explanation_reports_slope_only() {
        let formula = parse_formula("y ~ x + (0 + x | subject)").unwrap();
        let explanation = explain_model(&formula);
        let text = explanation.to_text();

        assert!(text.contains("does not add group-specific baselines"));
        assert!(text.contains("RandomSlopeWithoutIntercept"));
    }
}
