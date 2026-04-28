use std::fmt;

use serde::{Deserialize, Serialize};

use crate::formula::Formula;

use super::diagnostics::Diagnostic;
use super::ir::{compile_formula_ir, InterceptPolicy, SemanticModel};

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
                .map(|term| format!("{term}: population-level term"))
                .collect(),
        });

        for term in &semantic_model.random_terms {
            let mut lines = Vec::new();
            lines.push(format!("group: {}", term.group.label()));
            lines.push(format!("source: {}", term.source_syntax.text));
            lines.push(match term.intercept {
                InterceptPolicy::Included => "random intercept: yes".to_string(),
                InterceptPolicy::Omitted => "random intercept: no".to_string(),
            });
            lines.push(format!(
                "varying coefficients: {}",
                term.basis
                    .iter()
                    .map(|b| b.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
            lines.push(format!("covariance model: {:?}", term.covariance));
            lines.extend(term.covariance_story.assumptions.iter().cloned());
            lines.extend(term.covariance_story.dependence.iter().cloned());

            sections.push(ExplanationSection {
                title: format!("Random effect {}", term.id),
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

        assert!(text.contains("Random effect r0"));
        assert!(text.contains("random intercept: yes"));
        assert!(text.contains("observations sharing subject are correlated"));
    }

    #[test]
    fn explanation_reports_slope_only() {
        let formula = parse_formula("y ~ x + (0 + x | subject)").unwrap();
        let explanation = explain_model(&formula);
        let text = explanation.to_text();

        assert!(text.contains("random intercept: no"));
        assert!(text.contains("RandomSlopeWithoutIntercept"));
    }
}
