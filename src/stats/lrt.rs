//! Likelihood ratio tests for nested mixed models.

use std::fmt;

use crate::model::traits::MixedModelFit;

const LOG_LIK_TOL: f64 = 1.0e-10;

/// Result of a likelihood ratio test comparing nested models.
#[derive(Debug, Clone)]
pub struct LikelihoodRatioTest {
    /// Number of observations (must be equal across models).
    pub nobs: usize,
    /// Canonical formula label for each model, when available.
    pub formulas: Vec<String>,
    /// Degrees of freedom for each model.
    pub dof: Vec<usize>,
    /// Log-likelihood for each model.
    pub loglik: Vec<f64>,
    /// Deviance (-2 * loglik) for each model.
    pub deviance: Vec<f64>,
    /// Chi-squared statistics (between successive models).
    pub chisq: Vec<f64>,
    /// Degrees of freedom for each chi-squared test.
    pub chisq_dof: Vec<usize>,
    /// P-values for each test.
    pub pvalues: Vec<f64>,
}

impl LikelihoodRatioTest {
    /// Perform a likelihood ratio test on two or more nested models.
    ///
    /// Models should be provided in order from smallest to largest.
    pub fn test(models: &[&dyn MixedModelFit]) -> Result<Self, String> {
        let formulas = models
            .iter()
            .map(|m| m.formula_label().unwrap_or_else(|| "NA".to_string()))
            .collect();
        Self::test_with_formulas(models, formulas)
    }

    /// Perform a likelihood ratio test with explicit formula labels.
    pub fn test_with_formulas(
        models: &[&dyn MixedModelFit],
        formulas: Vec<String>,
    ) -> Result<Self, String> {
        if models.len() < 2 {
            return Err("At least two models are needed".to_string());
        }
        if formulas.len() != models.len() {
            return Err("Formula labels must match the number of models".to_string());
        }

        let nobs = models[0].nobs();
        for m in models {
            if m.nobs() != nobs {
                return Err("All models must have the same number of observations".to_string());
            }
        }

        let dof: Vec<usize> = models.iter().map(|m| m.dof()).collect();
        let loglik: Vec<f64> = models.iter().map(|m| m.loglikelihood()).collect();
        let deviance: Vec<f64> = loglik.iter().map(|ll| -2.0 * ll).collect();

        let mut chisq = Vec::new();
        let mut chisq_dof = Vec::new();
        let mut pvalues = Vec::new();

        for i in 1..models.len() {
            if dof[i] <= dof[i - 1] {
                return Err("Likelihood ratio test is only valid for nested models".to_string());
            }
            if loglik[i] + LOG_LIK_TOL < loglik[i - 1] {
                return Err(
                    "Log-likelihood must not be lower in models with more degrees of freedom"
                        .to_string(),
                );
            }

            let chi = 2.0 * (loglik[i] - loglik[i - 1]).abs();
            let ddof = dof[i] - dof[i - 1];
            use statrs::distribution::{ChiSquared, ContinuousCDF};
            let dist = ChiSquared::new(ddof as f64).unwrap();
            let pval = 1.0 - dist.cdf(chi);
            chisq.push(chi);
            chisq_dof.push(ddof);
            pvalues.push(pval);
        }

        Ok(LikelihoodRatioTest {
            nobs,
            formulas,
            dof,
            loglik,
            deviance,
            chisq,
            chisq_dof,
            pvalues,
        })
    }

    /// Extract the p-value when exactly one comparison is present.
    pub fn pvalue(&self) -> Result<f64, String> {
        match self.pvalues.as_slice() {
            [pvalue] => Ok(*pvalue),
            _ => Err("Cannot extract only one p-value from a multiple test result.".to_string()),
        }
    }

    /// Render the likelihood-ratio test as a markdown table.
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str("|                                          | model-dof | -2 logLik |  χ² | χ²-dof | P(>χ²) |\n");
        out.push_str("|:---------------------------------------- | ---------:| ---------:| ---:| ------:|:------ |\n");

        out.push_str(&format!(
            "| {:<40} | {:>9} | {:>9} | {:>3} | {:>6} | {:<6} |\n",
            escape_markdown_pipes(&self.formulas[0]),
            self.dof[0],
            (2.0 * self.loglik[0]).round() as i64,
            "",
            "",
            ""
        ));

        for i in 1..self.formulas.len() {
            out.push_str(&format!(
                "| {:<40} | {:>9} | {:>9} | {:>3} | {:>6} | {:<6} |\n",
                escape_markdown_pipes(&self.formulas[i]),
                self.dof[i],
                (2.0 * self.loglik[i]).round() as i64,
                self.chisq[i - 1].round() as i64,
                self.chisq_dof[i - 1],
                format_pvalue(self.pvalues[i - 1])
            ));
        }

        out
    }
}

impl fmt::Display for LikelihoodRatioTest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Likelihood-ratio test: {} models fitted on {} observations",
            self.formulas.len(),
            self.nobs
        )?;
        writeln!(f, "Model Formulae")?;
        for (idx, formula) in self.formulas.iter().enumerate() {
            writeln!(f, "{}: {}", idx + 1, formula)?;
        }

        let rows = self.plaintext_rows();
        let widths = column_widths(&rows);
        let rule_len = widths.iter().sum::<usize>() + 2 * (widths.len() - 1);
        let rule = "─".repeat(rule_len);
        writeln!(f, "{rule}")?;

        for (row_idx, row) in rows.iter().enumerate() {
            for (col_idx, cell) in row.iter().enumerate() {
                if col_idx > 0 {
                    write!(f, "  ")?;
                }
                if col_idx == 0 {
                    write!(f, "{cell:<width$}", width = widths[col_idx])?;
                } else {
                    write!(f, "{cell:>width$}", width = widths[col_idx])?;
                }
            }
            if row_idx == 0 {
                writeln!(f)?;
                writeln!(f, "{rule}")?;
            } else if row_idx + 1 < rows.len() {
                writeln!(f)?;
            }
        }

        write!(f, "\n{rule}")
    }
}

impl LikelihoodRatioTest {
    fn plaintext_rows(&self) -> Vec<Vec<String>> {
        let mut rows = vec![vec![
            "".to_string(),
            "DoF".to_string(),
            "-2 logLik".to_string(),
            "χ²".to_string(),
            "χ²-dof".to_string(),
            "P(>χ²)".to_string(),
        ]];

        rows.push(vec![
            "[1]".to_string(),
            self.dof[0].to_string(),
            format!("{:.4}", self.deviance[0]),
            String::new(),
            String::new(),
            String::new(),
        ]);

        for i in 1..self.formulas.len() {
            rows.push(vec![
                format!("[{}]", i + 1),
                self.dof[i].to_string(),
                format!("{:.4}", self.deviance[i]),
                format!("{:.4}", self.chisq[i - 1]),
                self.chisq_dof[i - 1].to_string(),
                format_pvalue(self.pvalues[i - 1]),
            ]);
        }

        rows
    }
}

fn column_widths(rows: &[Vec<String>]) -> Vec<usize> {
    (0..rows[0].len())
        .map(|col| rows.iter().map(|row| row[col].len()).max().unwrap_or(0))
        .collect()
}

fn escape_markdown_pipes(label: &str) -> String {
    label.replace('|', "\\|")
}

fn format_pvalue(pvalue: f64) -> String {
    if !pvalue.is_finite() {
        return String::new();
    }
    if pvalue <= 0.0 {
        return "<1e-99".to_string();
    }
    if pvalue < 1.0e-4 {
        let exponent = (-pvalue.log10()).floor().max(1.0) as i32;
        return format!("<1e-{exponent:02}");
    }
    format!("{pvalue:.4}")
}

#[cfg(test)]
mod tests {
    use nalgebra::{DMatrix, DVector};

    use super::*;
    use crate::types::OptSummary;

    #[derive(Clone)]
    struct DummyFit {
        nobs: usize,
        dof: usize,
        loglik: f64,
        formula: Option<String>,
        response: DVector<f64>,
        model_matrix: DMatrix<f64>,
        optsum: OptSummary,
    }

    impl DummyFit {
        fn new(nobs: usize, dof: usize, loglik: f64, formula: Option<&str>) -> Self {
            Self {
                nobs,
                dof,
                loglik,
                formula: formula.map(str::to_string),
                response: DVector::zeros(nobs),
                model_matrix: DMatrix::zeros(nobs, 0),
                optsum: OptSummary::new(Vec::new()),
            }
        }
    }

    impl MixedModelFit for DummyFit {
        fn nobs(&self) -> usize {
            self.nobs
        }

        fn dof(&self) -> usize {
            self.dof
        }

        fn coef(&self) -> DVector<f64> {
            DVector::zeros(0)
        }

        fn fixef(&self) -> DVector<f64> {
            DVector::zeros(0)
        }

        fn coef_names(&self) -> Vec<String> {
            Vec::new()
        }

        fn vcov(&self) -> DMatrix<f64> {
            DMatrix::zeros(0, 0)
        }

        fn stderror(&self) -> DVector<f64> {
            DVector::zeros(0)
        }

        fn fitted(&self) -> DVector<f64> {
            DVector::zeros(self.nobs)
        }

        fn residuals(&self) -> DVector<f64> {
            DVector::zeros(self.nobs)
        }

        fn response(&self) -> &DVector<f64> {
            &self.response
        }

        fn model_matrix(&self) -> &DMatrix<f64> {
            &self.model_matrix
        }

        fn objective(&self) -> f64 {
            -2.0 * self.loglik
        }

        fn loglikelihood(&self) -> f64 {
            self.loglik
        }

        fn formula_label(&self) -> Option<String> {
            self.formula.clone()
        }

        fn is_fitted(&self) -> bool {
            true
        }

        fn is_singular(&self) -> bool {
            false
        }

        fn opt_summary(&self) -> &OptSummary {
            &self.optsum
        }

        fn theta(&self) -> Vec<f64> {
            Vec::new()
        }

        fn dispersion(&self, _sqr: bool) -> f64 {
            1.0
        }

        fn ranef(&self) -> Vec<DMatrix<f64>> {
            Vec::new()
        }
    }

    #[test]
    fn test_lrt_rejects_non_increasing_dof() {
        let m0 = DummyFit::new(180, 6, -876.0, Some("m0"));
        let m1 = DummyFit::new(180, 4, -875.0, Some("m1"));
        let err = LikelihoodRatioTest::test(&[&m0, &m1]).unwrap_err();

        assert_eq!(err, "Likelihood ratio test is only valid for nested models");
    }

    #[test]
    fn test_lrt_rejects_decreasing_loglikelihood() {
        let m0 = DummyFit::new(180, 4, -876.0, Some("m0"));
        let m1 = DummyFit::new(180, 6, -877.0, Some("m1"));
        let err = LikelihoodRatioTest::test(&[&m0, &m1]).unwrap_err();

        assert_eq!(
            err,
            "Log-likelihood must not be lower in models with more degrees of freedom"
        );
    }

    #[test]
    fn test_pvalue_requires_a_single_comparison() {
        let m0 = DummyFit::new(180, 4, -897.0, Some("m0"));
        let m1 = DummyFit::new(180, 5, -890.0, Some("m1"));
        let m2 = DummyFit::new(180, 6, -876.0, Some("m2"));

        let lrt_single = LikelihoodRatioTest::test(&[&m0, &m1]).unwrap();
        assert!(lrt_single.pvalue().unwrap() < 0.01);

        let lrt_multiple = LikelihoodRatioTest::test(&[&m0, &m1, &m2]).unwrap();
        assert_eq!(
            lrt_multiple.pvalue().unwrap_err(),
            "Cannot extract only one p-value from a multiple test result."
        );
    }

    #[test]
    fn test_lrt_display_includes_formula_table() {
        let m0 = DummyFit::new(180, 4, -897.0, Some("reaction ~ 1 + days + (1 | subj)"));
        let m1 = DummyFit::new(
            180,
            6,
            -876.0,
            Some("reaction ~ 1 + days + (1 + days | subj)"),
        );
        let lrt = LikelihoodRatioTest::test(&[&m0, &m1]).unwrap();
        let out = lrt.to_string();

        assert!(out.contains("Likelihood-ratio test: 2 models fitted on 180 observations"));
        assert!(out.contains("1: reaction ~ 1 + days + (1 | subj)"));
        assert!(out.contains("2: reaction ~ 1 + days + (1 + days | subj)"));
        assert!(out.contains("[2]"));
        assert!(out.contains("1752.0000"));
        assert!(out.contains("42.0000"));
        assert!(out.contains("<1e-09"));
    }

    #[test]
    fn test_lrt_markdown_matches_julia_style_table() {
        let m0 = DummyFit::new(180, 4, -897.0, Some("reaction ~ 1 + days + (1 | subj)"));
        let m1 = DummyFit::new(
            180,
            6,
            -876.0,
            Some("reaction ~ 1 + days + (1 + days | subj)"),
        );
        let lrt = LikelihoodRatioTest::test(&[&m0, &m1]).unwrap();

        assert_eq!(
            lrt.to_markdown(),
            concat!(
                "|                                          | model-dof | -2 logLik |  χ² | χ²-dof | P(>χ²) |\n",
                "|:---------------------------------------- | ---------:| ---------:| ---:| ------:|:------ |\n",
                "| reaction ~ 1 + days + (1 \\| subj)        |         4 |     -1794 |     |        |        |\n",
                "| reaction ~ 1 + days + (1 + days \\| subj) |         6 |     -1752 |  42 |      2 | <1e-09 |\n"
            )
        );
    }
}
