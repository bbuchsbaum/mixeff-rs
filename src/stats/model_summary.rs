//! Mixed-model summary tables used by MIME-style renderers.

use serde::{Deserialize, Serialize};
use statrs::distribution::{ChiSquared, ContinuousCDF};
use std::collections::BTreeMap;

use crate::compiler::GlmmFitMetadata;
use crate::compiler::{
    FixedEffectInferenceMethod, FixedEffectInferenceRowKind, FixedEffectStatisticName,
};
use crate::model::traits::MixedModelFit;
use crate::model::{GeneralizedLinearMixedModel, LinearMixedModel};
use crate::stats::{CoefTable, VarCorr};

/// Stable schema name for serialized post-fit summaries.
pub const FIT_SUMMARY_SCHEMA: &str = "mixedmodels.fit_summary";
/// Stable schema version for serialized post-fit summaries.
pub const FIT_SUMMARY_SCHEMA_VERSION: &str = "1.0.0";

/// Versioned, downstream-friendly fit-summary payload.
///
/// This is the JSON-facing summary contract for wrappers that need fitted
/// objective values, optimizer metadata, fixed-effect tables, variance
/// components, and the rendered-model summary data in one stable envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct FitSummaryPayload {
    /// Stable schema name.
    pub schema_name: String,
    /// Stable schema version.
    pub schema_version: String,
    /// Model family of the fitted object, such as `linear_mixed_model`.
    pub model_kind: String,
    /// Formula label, when available.
    pub formula: Option<String>,
    /// GLMM response family label, when applicable.
    pub family: Option<String>,
    /// GLMM link-function label, when applicable.
    pub link: Option<String>,
    /// Number of observations.
    pub nobs: usize,
    /// Model degrees of freedom.
    pub dof: usize,
    /// Fitted objective value.
    pub objective: f64,
    /// Fitted log-likelihood.
    pub loglikelihood: f64,
    /// Akaike information criterion.
    pub aic: f64,
    /// Bayesian information criterion.
    pub bic: f64,
    /// Whether the model has been fitted.
    pub is_fitted: bool,
    /// Whether the fitted random-effect covariance is singular.
    pub is_singular: bool,
    /// Covariance-parameter vector.
    pub theta: Vec<f64>,
    /// Residual scale or GLMM dispersion parameter.
    pub dispersion: f64,
    /// Optimizer selected by the fit.
    pub optimizer: String,
    /// Backend-specific optimizer code.
    pub optimizer_code: String,
    /// Optimizer backend name.
    pub optimizer_backend: String,
    /// Optimizer return status.
    pub optimizer_status: String,
    /// GLMM estimation method label, when applicable.
    pub estimation_method: Option<String>,
    /// GLMM objective-definition label, when applicable.
    pub objective_definition: Option<String>,
    /// GLMM response-constants convention, when applicable.
    pub response_constants: Option<String>,
    /// Number of adaptive Gauss-Hermite quadrature points for GLMMs.
    pub n_agq: Option<usize>,
    /// Fallback status for GLMM fitting, when a fallback path was used.
    pub fallback_status: Option<String>,
    /// Fixed or estimated response-family parameters, when applicable.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub family_parameters: BTreeMap<String, f64>,
    /// Provenance for response-family parameters, keyed like `family_parameters`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub family_parameter_sources: BTreeMap<String, String>,
    /// Number of objective evaluations.
    pub feval: i64,
    /// Fixed-effect coefficient table.
    pub coefficients: CoefTable,
    /// Random-effect variance-covariance table.
    pub varcorr: VarCorr,
    /// Renderable compact model summary.
    pub summary: ModelSummary,
}

/// One row in a model summary table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ModelSummaryRow {
    /// Row label, usually a coefficient or random-effect term name.
    pub label: String,
    /// Estimate shown in the fixed-effect columns, when applicable.
    pub estimate: Option<f64>,
    /// Standard error shown in the fixed-effect columns, when applicable.
    pub std_error: Option<f64>,
    /// Wald statistic shown in the fixed-effect columns, when available.
    pub z_stat: Option<f64>,
    /// P-value shown in the fixed-effect columns, when available.
    pub pvalue: Option<f64>,
    /// Per-group random-effect scale values aligned with `sigma_headers`.
    pub sigma_values: Vec<Option<f64>>,
}

/// A markdown/HTML/LaTeX-ready summary of a fitted mixed model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ModelSummary {
    /// Headers for random-effect scale columns.
    pub sigma_headers: Vec<String>,
    /// Summary rows in display order.
    pub rows: Vec<ModelSummaryRow>,
}

impl ModelSummary {
    /// Construct a summary from a linear mixed model.
    pub fn from_linear_model(model: &LinearMixedModel) -> Self {
        let sigma = model.sigma();
        let varcorr = VarCorr::from_reterms(&model.reterms, sigma, Some(sigma));
        let coeftable = model.coeftable();
        summary_from_parts_with_pvalues(
            &coeftable.names,
            &coeftable.estimates,
            &coeftable.std_errors,
            Some(&coeftable.p_values),
            &varcorr,
            Some("Residual"),
            Some(sigma),
        )
    }

    /// Construct a summary from a generalized linear mixed model.
    pub fn from_generalized_model(model: &GeneralizedLinearMixedModel) -> Self {
        let scale = model.random_effect_scale();
        let residual_label = if model.family.has_dispersion() {
            Some("Dispersion")
        } else {
            None
        };
        let residual_value = if model.family.has_dispersion() {
            Some(scale)
        } else {
            None
        };
        let varcorr = VarCorr::from_reterms(&model.lmm.reterms, scale, None);
        let coeftable = generalized_coeftable(model);
        summary_from_coeftable(&coeftable, &varcorr, residual_label, residual_value)
    }

    /// Render the summary as a markdown table.
    pub fn to_markdown(&self) -> String {
        let header = self.header_row();
        let rows = self.markdown_rows();
        let mut all_rows = Vec::with_capacity(rows.len() + 1);
        all_rows.push(header.clone());
        all_rows.extend(rows.iter().cloned());

        let widths = column_widths(&all_rows);
        let mut out = String::new();
        out.push_str(&format_markdown_row(
            &header,
            &widths,
            &alignments(self.sigma_headers.len()),
        ));
        out.push('\n');
        out.push_str(&format_markdown_alignment(
            &widths,
            &alignments(self.sigma_headers.len()),
        ));
        out.push('\n');
        for row in rows {
            out.push_str(&format_markdown_row(
                &row,
                &widths,
                &alignments(self.sigma_headers.len()),
            ));
            out.push('\n');
        }
        out
    }

    /// Render the summary as an HTML table.
    pub fn to_html(&self) -> String {
        let header = self.header_row();
        let rows = self.markdown_rows();
        let mut out = String::from("<table><tr>");
        for cell in &header {
            out.push_str(&format!("<th align=\"left\">{}</th>", html_escape(cell)));
        }
        out.push_str("</tr>");
        for row in rows {
            out.push_str("<tr>");
            for (idx, cell) in row.iter().enumerate() {
                let align = if idx == 0 { "left" } else { "right" };
                out.push_str(&format!("<td align=\"{align}\">{}</td>", html_escape(cell)));
            }
            out.push_str("</tr>");
        }
        out.push_str("</table>\n");
        out
    }

    /// Render the summary as a LaTeX table.
    pub fn to_latex(&self) -> String {
        let header = self.latex_header_row();
        let rows = self.latex_rows();
        let mut out = String::new();
        let cols = "l | ".to_string() + &vec!["r"; header.len() - 1].join(" | ");

        out.push_str("\\begin{tabular}\n");
        out.push_str(&format!("{{{cols}}}\n"));
        out.push_str(&header.join(" & "));
        out.push_str(" \\\\\n");
        for row in rows {
            out.push_str(&row.join(" & "));
            out.push_str(" \\\\\n");
        }
        out.push_str("\\end{tabular}\n");
        out
    }

    fn header_row(&self) -> Vec<String> {
        let mut header = vec![
            String::new(),
            "Est.".to_string(),
            "SE".to_string(),
            "z".to_string(),
            "p".to_string(),
        ];
        header.extend(self.sigma_headers.iter().cloned());
        header
    }

    fn latex_header_row(&self) -> Vec<String> {
        let mut header = vec![
            String::new(),
            "Est.".to_string(),
            "SE".to_string(),
            "z".to_string(),
            "p".to_string(),
        ];
        header.extend(
            self.sigma_headers
                .iter()
                .map(|h| sigma_header_to_latex(h))
                .collect::<Vec<_>>(),
        );
        header
    }

    fn markdown_rows(&self) -> Vec<Vec<String>> {
        self.rows
            .iter()
            .map(|row| {
                let mut cells = vec![
                    row.label.clone(),
                    format_optional(row.estimate, format_fixed4),
                    format_optional(row.std_error, format_fixed4),
                    format_optional(row.z_stat, format_fixed2),
                    format_optional(row.pvalue, format_pvalue),
                ];
                cells.extend(
                    row.sigma_values
                        .iter()
                        .map(|value| format_optional(*value, format_fixed4)),
                );
                cells
            })
            .collect()
    }

    fn latex_rows(&self) -> Vec<Vec<String>> {
        self.rows
            .iter()
            .map(|row| {
                let mut cells = vec![
                    latex_escape(&row.label),
                    format_optional(row.estimate, format_fixed4),
                    format_optional(row.std_error, format_fixed4),
                    format_optional(row.z_stat, format_fixed2),
                    format_optional(row.pvalue, format_pvalue),
                ];
                cells.extend(
                    row.sigma_values
                        .iter()
                        .map(|value| format_optional(*value, format_fixed4)),
                );
                cells
            })
            .collect()
    }
}

impl FitSummaryPayload {
    /// Build a fit-summary payload from a fitted or pre-fit linear model.
    pub fn from_linear_model(model: &LinearMixedModel) -> Self {
        Self::from_parts(
            "linear_mixed_model",
            model,
            None,
            None,
            model.coeftable(),
            model.varcorr(),
            ModelSummary::from_linear_model(model),
        )
    }

    /// Build a fit-summary payload from a fitted or pre-fit generalized model.
    pub fn from_generalized_model(model: &GeneralizedLinearMixedModel) -> Self {
        let mut payload = Self::from_parts(
            "generalized_linear_mixed_model",
            model,
            model.family_kind().map(family_label),
            model.link_kind().map(link_label),
            generalized_coeftable(model),
            model.varcorr(),
            ModelSummary::from_generalized_model(model),
        );
        if let Some(metadata) = model.compiler_artifact().glmm_fit_metadata.as_ref() {
            payload.family_parameters = metadata.family_parameters.clone();
            payload.family_parameter_sources = metadata.family_parameter_sources.clone();
        } else if let Some(theta) = model.negative_binomial_theta() {
            payload
                .family_parameters
                .insert("negative_binomial_theta".to_string(), theta);
            payload
                .family_parameters
                .insert("negative_binomial_variance_power".to_string(), 2.0);
            payload.family_parameter_sources.insert(
                "negative_binomial_theta".to_string(),
                if model.negative_binomial_theta_estimated() {
                    "estimated".to_string()
                } else {
                    "fixed".to_string()
                },
            );
            payload.family_parameter_sources.insert(
                "negative_binomial_variance_power".to_string(),
                "fixed".to_string(),
            );
        }
        payload
    }

    fn from_parts<M: MixedModelFit>(
        model_kind: &str,
        model: &M,
        family: Option<&'static str>,
        link: Option<&'static str>,
        coefficients: CoefTable,
        varcorr: VarCorr,
        summary: ModelSummary,
    ) -> Self {
        let opt = model.opt_summary();
        let glmm_metadata = family.map(|_| GlmmFitMetadata::from_opt_summary(opt));
        FitSummaryPayload {
            schema_name: FIT_SUMMARY_SCHEMA.to_string(),
            schema_version: FIT_SUMMARY_SCHEMA_VERSION.to_string(),
            model_kind: model_kind.to_string(),
            formula: model.formula_label(),
            family: family.map(str::to_string),
            link: link.map(str::to_string),
            nobs: model.nobs(),
            dof: model.dof(),
            objective: model.objective(),
            loglikelihood: model.loglikelihood(),
            aic: model.aic(),
            bic: model.bic(),
            is_fitted: model.is_fitted(),
            is_singular: model.is_singular(),
            theta: model.theta(),
            dispersion: model.dispersion(false),
            optimizer: opt.optimizer_name().to_string(),
            optimizer_code: opt.optimizer_code().to_string(),
            optimizer_backend: opt.backend_name().to_string(),
            optimizer_status: opt.return_value.clone(),
            estimation_method: glmm_metadata
                .as_ref()
                .map(|metadata| metadata.estimation_method.clone()),
            objective_definition: glmm_metadata
                .as_ref()
                .map(|metadata| metadata.objective_definition.clone()),
            response_constants: glmm_metadata
                .as_ref()
                .map(|metadata| metadata.response_constants.clone()),
            n_agq: family.map(|_| opt.n_agq),
            fallback_status: glmm_metadata
                .as_ref()
                .and_then(|metadata| metadata.fallback_status.clone()),
            family_parameters: glmm_metadata
                .as_ref()
                .map(|metadata| metadata.family_parameters.clone())
                .unwrap_or_default(),
            family_parameter_sources: glmm_metadata
                .as_ref()
                .map(|metadata| metadata.family_parameter_sources.clone())
                .unwrap_or_default(),
            feval: opt.feval,
            coefficients,
            varcorr,
            summary,
        }
    }
}

fn family_label(family: crate::model::traits::Family) -> &'static str {
    match family {
        crate::model::traits::Family::Normal => "normal",
        crate::model::traits::Family::Bernoulli => "bernoulli",
        crate::model::traits::Family::Binomial => "binomial",
        crate::model::traits::Family::Poisson => "poisson",
        crate::model::traits::Family::NegativeBinomial => "negative_binomial",
        crate::model::traits::Family::Gamma => "gamma",
        crate::model::traits::Family::InverseGaussian => "inverse_gaussian",
    }
}

fn link_label(link: crate::model::traits::LinkFunction) -> &'static str {
    match link {
        crate::model::traits::LinkFunction::Identity => "identity",
        crate::model::traits::LinkFunction::Log => "log",
        crate::model::traits::LinkFunction::Logit => "logit",
        crate::model::traits::LinkFunction::Probit => "probit",
        crate::model::traits::LinkFunction::Cloglog => "cloglog",
        crate::model::traits::LinkFunction::Inverse => "inverse",
        crate::model::traits::LinkFunction::Sqrt => "sqrt",
    }
}

fn generalized_coeftable(model: &GeneralizedLinearMixedModel) -> CoefTable {
    if let Some(table) = model
        .compiler_artifact()
        .fixed_effect_inference_table
        .as_ref()
    {
        let rows = table
            .rows
            .iter()
            .filter(|row| row.kind == FixedEffectInferenceRowKind::Coefficient)
            .collect::<Vec<_>>();
        if !rows.is_empty() {
            let names = rows.iter().map(|row| row.label.clone()).collect::<Vec<_>>();
            let estimates = rows
                .iter()
                .map(|row| row.estimate.unwrap_or(f64::NAN))
                .collect::<Vec<_>>();
            let std_errors = rows
                .iter()
                .map(|row| row.std_error.unwrap_or(f64::NAN))
                .collect::<Vec<_>>();
            let statistics = rows
                .iter()
                .map(|row| row.statistic.unwrap_or(f64::NAN))
                .collect::<Vec<_>>();
            let p_values = rows
                .iter()
                .map(|row| row.p_value.unwrap_or(f64::NAN))
                .collect::<Vec<_>>();
            let p_value_reasons = rows
                .iter()
                .map(|row| {
                    row.p_value.map(|_| None).unwrap_or_else(|| {
                        row.reason.clone().or_else(|| {
                            Some("fixed-effect inference is unavailable for this row".to_string())
                        })
                    })
                })
                .collect::<Vec<_>>();
            let df = rows
                .iter()
                .map(|row| row.denominator_df)
                .collect::<Vec<_>>();
            let statistic_name = rows
                .iter()
                .find_map(|row| row.statistic_name)
                .map(fixed_effect_statistic_name_label)
                .unwrap_or("z");
            let method = rows
                .first()
                .map(|row| fixed_effect_inference_method_label(row.method))
                .unwrap_or("not-computed");
            return CoefTable::from_df_inference(
                names,
                estimates,
                std_errors,
                statistics,
                p_values,
                p_value_reasons,
                df,
                statistic_name,
                method,
            );
        }
    }

    CoefTable::new(
        model.coef_names(),
        model.coef().as_slice().to_vec(),
        model.stderror().as_slice().to_vec(),
    )
}

fn summary_from_coeftable(
    coeftable: &CoefTable,
    varcorr: &VarCorr,
    residual_label: Option<&str>,
    residual_value: Option<f64>,
) -> ModelSummary {
    let sigma_headers = varcorr
        .components
        .iter()
        .map(|comp| format!("σ_{}", comp.group))
        .collect::<Vec<_>>();
    let sigma_maps = varcorr
        .components
        .iter()
        .map(|comp| {
            comp.names
                .iter()
                .cloned()
                .zip(comp.std_dev.iter().copied())
                .collect::<std::collections::HashMap<_, _>>()
        })
        .collect::<Vec<_>>();

    let mut rows = Vec::new();
    for index in 0..coeftable.names.len() {
        let label = &coeftable.names[index];
        rows.push(ModelSummaryRow {
            label: label.clone(),
            estimate: coeftable
                .estimates
                .get(index)
                .copied()
                .filter(|value| value.is_finite()),
            std_error: coeftable
                .std_errors
                .get(index)
                .copied()
                .filter(|value| value.is_finite()),
            z_stat: coeftable
                .z_values
                .get(index)
                .copied()
                .filter(|value| value.is_finite()),
            pvalue: coeftable
                .p_values
                .get(index)
                .copied()
                .filter(|value| value.is_finite()),
            sigma_values: sigma_maps
                .iter()
                .map(|map| map.get(label).copied())
                .collect(),
        });
    }

    let mut seen = coeftable.names.clone();
    for comp in &varcorr.components {
        for name in &comp.names {
            if seen.contains(name) {
                continue;
            }
            rows.push(ModelSummaryRow {
                label: name.clone(),
                estimate: None,
                std_error: None,
                z_stat: None,
                pvalue: None,
                sigma_values: sigma_maps
                    .iter()
                    .map(|map| map.get(name).copied())
                    .collect(),
            });
            seen.push(name.clone());
        }
    }

    if let (Some(label), Some(value)) = (residual_label, residual_value) {
        rows.push(ModelSummaryRow {
            label: label.to_string(),
            estimate: Some(value),
            std_error: None,
            z_stat: None,
            pvalue: None,
            sigma_values: vec![None; sigma_headers.len()],
        });
    }

    ModelSummary {
        sigma_headers,
        rows,
    }
}

fn summary_from_parts_with_pvalues(
    coef_names: &[String],
    coef: &[f64],
    stderror: &[f64],
    pvalues: Option<&[f64]>,
    varcorr: &VarCorr,
    residual_label: Option<&str>,
    residual_value: Option<f64>,
) -> ModelSummary {
    let sigma_headers = varcorr
        .components
        .iter()
        .map(|comp| format!("σ_{}", comp.group))
        .collect::<Vec<_>>();
    let sigma_maps = varcorr
        .components
        .iter()
        .map(|comp| {
            comp.names
                .iter()
                .cloned()
                .zip(comp.std_dev.iter().copied())
                .collect::<std::collections::HashMap<_, _>>()
        })
        .collect::<Vec<_>>();

    let chisq1 = ChiSquared::new(1.0).unwrap();
    let mut rows = Vec::new();

    for (index, ((label, est), se)) in coef_names
        .iter()
        .zip(coef.iter())
        .zip(stderror.iter())
        .enumerate()
    {
        let z = if *se > 0.0 && se.is_finite() {
            Some(*est / *se)
        } else {
            None
        };
        let pvalue = pvalues
            .and_then(|values| values.get(index).copied())
            .filter(|value| value.is_finite())
            .or_else(|| {
                pvalues
                    .is_none()
                    .then(|| z.map(|zv| 1.0 - chisq1.cdf(zv * zv)))
                    .flatten()
            });
        rows.push(ModelSummaryRow {
            label: label.clone(),
            estimate: Some(*est),
            std_error: Some(*se),
            z_stat: z,
            pvalue,
            sigma_values: sigma_maps
                .iter()
                .map(|map| map.get(label).copied())
                .collect(),
        });
    }

    let mut seen = coef_names.to_vec();
    for comp in &varcorr.components {
        for name in &comp.names {
            if seen.contains(name) {
                continue;
            }
            rows.push(ModelSummaryRow {
                label: name.clone(),
                estimate: None,
                std_error: None,
                z_stat: None,
                pvalue: None,
                sigma_values: sigma_maps
                    .iter()
                    .map(|map| map.get(name).copied())
                    .collect(),
            });
            seen.push(name.clone());
        }
    }

    if let (Some(label), Some(value)) = (residual_label, residual_value) {
        rows.push(ModelSummaryRow {
            label: label.to_string(),
            estimate: Some(value),
            std_error: None,
            z_stat: None,
            pvalue: None,
            sigma_values: vec![None; sigma_headers.len()],
        });
    }

    ModelSummary {
        sigma_headers,
        rows,
    }
}

fn fixed_effect_statistic_name_label(name: FixedEffectStatisticName) -> &'static str {
    match name {
        FixedEffectStatisticName::Z => "z",
        FixedEffectStatisticName::T => "t",
        FixedEffectStatisticName::F => "F",
        FixedEffectStatisticName::ChiSquare => "chisq",
    }
}

fn fixed_effect_inference_method_label(method: FixedEffectInferenceMethod) -> &'static str {
    match method {
        FixedEffectInferenceMethod::AsymptoticWaldZ => "wald-z",
        FixedEffectInferenceMethod::Satterthwaite => "satterthwaite",
        FixedEffectInferenceMethod::KenwardRoger => "kenward-roger",
        FixedEffectInferenceMethod::Bootstrap => "bootstrap",
        FixedEffectInferenceMethod::NotComputed => "not-computed",
    }
}

fn alignments(sigma_columns: usize) -> Vec<Alignment> {
    let mut align = vec![
        Alignment::Left,
        Alignment::Right,
        Alignment::Right,
        Alignment::Right,
        Alignment::Right,
    ];
    align.extend((0..sigma_columns).map(|_| Alignment::Right));
    align
}

#[derive(Clone, Copy)]
enum Alignment {
    Left,
    Right,
}

fn column_widths(rows: &[Vec<String>]) -> Vec<usize> {
    (0..rows[0].len())
        .map(|col| {
            rows.iter()
                .map(|row| row[col].chars().count())
                .max()
                .unwrap_or(0)
        })
        .collect()
}

fn format_markdown_row(row: &[String], widths: &[usize], align: &[Alignment]) -> String {
    let mut out = String::new();
    for ((cell, width), alignment) in row.iter().zip(widths.iter()).zip(align.iter()) {
        match alignment {
            Alignment::Left => out.push_str(&format!("| {:<width$} ", cell, width = *width)),
            Alignment::Right => out.push_str(&format!("| {:>width$} ", cell, width = *width)),
        }
    }
    out.push('|');
    out
}

fn format_markdown_alignment(widths: &[usize], align: &[Alignment]) -> String {
    let mut out = String::new();
    for (width, alignment) in widths.iter().zip(align.iter()) {
        match alignment {
            Alignment::Left => {
                out.push_str("|:");
                out.push_str(&"-".repeat(*width));
                out.push(' ');
            }
            Alignment::Right => {
                out.push_str("| ");
                out.push_str(&"-".repeat(*width));
                out.push(':');
            }
        }
    }
    out.push('|');
    out
}

fn format_optional(value: Option<f64>, f: fn(f64) -> String) -> String {
    value.map(f).unwrap_or_default()
}

fn format_fixed4(value: f64) -> String {
    format!("{value:.4}")
}

fn format_fixed2(value: f64) -> String {
    format!("{value:.2}")
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

fn sigma_header_to_latex(header: &str) -> String {
    if let Some(rest) = header.strip_prefix("σ_") {
        format!("$\\sigma_\\text{{{}}}$", latex_escape(rest))
    } else {
        latex_escape(header)
    }
}

fn html_escape(input: &str) -> String {
    input
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn latex_escape(input: &str) -> String {
    input.replace('_', "\\_")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formula::parse_formula;
    use crate::model::{DataFrame, LinkFunction};
    use crate::model::{Family, GeneralizedLinearMixedModel, LinearMixedModel};

    fn row(
        label: &str,
        estimate: Option<f64>,
        std_error: Option<f64>,
        z_stat: Option<f64>,
        pvalue: Option<f64>,
        sigma_values: &[Option<f64>],
    ) -> ModelSummaryRow {
        ModelSummaryRow {
            label: label.to_string(),
            estimate,
            std_error,
            z_stat,
            pvalue,
            sigma_values: sigma_values.to_vec(),
        }
    }

    #[test]
    fn test_markdown_lmm_matches_julia_example() {
        let summary = ModelSummary {
            sigma_headers: vec!["σ_subj".to_string()],
            rows: vec![
                row(
                    "(Intercept)",
                    Some(251.4051),
                    Some(6.6323),
                    Some(37.91),
                    Some(0.0),
                    &[Some(23.7805)],
                ),
                row(
                    "days",
                    Some(10.4673),
                    Some(1.5022),
                    Some(6.97),
                    Some(5e-12),
                    &[Some(5.7168)],
                ),
                row("Residual", Some(25.5918), None, None, None, &[None]),
            ],
        };

        assert_eq!(
            summary.to_markdown(),
            concat!(
                "|             |     Est. |     SE |     z |      p |  σ_subj |\n",
                "|:----------- | --------:| ------:| -----:| ------:| -------:|\n",
                "| (Intercept) | 251.4051 | 6.6323 | 37.91 | <1e-99 | 23.7805 |\n",
                "| days        |  10.4673 | 1.5022 |  6.97 | <1e-11 |  5.7168 |\n",
                "| Residual    |  25.5918 |        |       |        |         |\n"
            )
        );
    }

    #[test]
    fn test_markdown_re_without_fe_matches_julia_example() {
        let summary = ModelSummary {
            sigma_headers: vec!["σ_subj".to_string(), "σ_item".to_string()],
            rows: vec![
                row(
                    "(Intercept)",
                    Some(2092.3713),
                    Some(76.9426),
                    Some(27.19),
                    Some(0.0),
                    &[None, Some(349.7858)],
                ),
                row("spkr: new", None, None, None, None, &[Some(258.9242), None]),
                row("spkr: old", None, None, None, None, &[Some(377.3837), None]),
                row("load: yes", None, None, None, None, &[None, Some(142.5331)]),
                row("Residual", Some(800.3224), None, None, None, &[None, None]),
            ],
        };

        assert_eq!(
            summary.to_markdown(),
            concat!(
                "|             |      Est. |      SE |     z |      p |   σ_subj |   σ_item |\n",
                "|:----------- | ---------:| -------:| -----:| ------:| --------:| --------:|\n",
                "| (Intercept) | 2092.3713 | 76.9426 | 27.19 | <1e-99 |          | 349.7858 |\n",
                "| spkr: new   |           |         |       |        | 258.9242 |          |\n",
                "| spkr: old   |           |         |       |        | 377.3837 |          |\n",
                "| load: yes   |           |         |       |        |          | 142.5331 |\n",
                "| Residual    |  800.3224 |         |       |        |          |          |\n"
            )
        );
    }

    #[test]
    fn test_markdown_glmm_matches_julia_example() {
        let summary = ModelSummary {
            sigma_headers: vec!["σ_subj".to_string(), "σ_item".to_string()],
            rows: vec![
                row(
                    "(Intercept)",
                    Some(0.1956),
                    Some(0.4052),
                    Some(0.48),
                    Some(0.6294),
                    &[Some(1.3398), Some(0.4953)],
                ),
                row(
                    "anger",
                    Some(0.0576),
                    Some(0.0168),
                    Some(3.43),
                    Some(0.0006),
                    &[None, None],
                ),
                row(
                    "gender: M",
                    Some(0.3208),
                    Some(0.1913),
                    Some(1.68),
                    Some(0.0935),
                    &[None, None],
                ),
                row(
                    "btype: scold",
                    Some(-1.0583),
                    Some(0.2568),
                    Some(-4.12),
                    Some(5e-5),
                    &[None, None],
                ),
                row(
                    "btype: shout",
                    Some(-2.1048),
                    Some(0.2585),
                    Some(-8.14),
                    Some(1e-15),
                    &[None, None],
                ),
                row(
                    "situ: self",
                    Some(-1.0550),
                    Some(0.2103),
                    Some(-5.02),
                    Some(1e-6),
                    &[None, None],
                ),
            ],
        };

        assert_eq!(
            summary.to_markdown(),
            concat!(
                "|              |    Est. |     SE |     z |      p | σ_subj | σ_item |\n",
                "|:------------ | -------:| ------:| -----:| ------:| ------:| ------:|\n",
                "| (Intercept)  |  0.1956 | 0.4052 |  0.48 | 0.6294 | 1.3398 | 0.4953 |\n",
                "| anger        |  0.0576 | 0.0168 |  3.43 | 0.0006 |        |        |\n",
                "| gender: M    |  0.3208 | 0.1913 |  1.68 | 0.0935 |        |        |\n",
                "| btype: scold | -1.0583 | 0.2568 | -4.12 | <1e-04 |        |        |\n",
                "| btype: shout | -2.1048 | 0.2585 | -8.14 | <1e-15 |        |        |\n",
                "| situ: self   | -1.0550 | 0.2103 | -5.02 | <1e-06 |        |        |\n"
            )
        );
    }

    #[test]
    fn test_latex_uses_sigma_subscripts() {
        let summary = ModelSummary {
            sigma_headers: vec!["σ_subj".to_string(), "σ_item".to_string()],
            rows: vec![row(
                "x",
                Some(1.0),
                Some(0.1),
                Some(10.0),
                Some(1e-6),
                &[None, None],
            )],
        };
        let out = summary.to_latex();

        assert!(out.starts_with("\\begin{tabular}\n{l | r | r | r | r | r | r}\n"));
        assert!(out.contains("$\\sigma_\\text{subj}$"));
        assert!(out.contains("$\\sigma_\\text{item}$"));
    }

    #[test]
    fn test_from_linear_model_adds_random_effect_only_rows() {
        let mut data = DataFrame::new();
        data.add_numeric("y", vec![2.0, 2.5, 3.0, 3.2, 4.0, 4.4, 5.0, 5.1])
            .unwrap();
        data.add_numeric("x", vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
            .unwrap();
        data.add_categorical(
            "g",
            vec![
                "a".to_string(),
                "a".to_string(),
                "b".to_string(),
                "b".to_string(),
                "c".to_string(),
                "c".to_string(),
                "d".to_string(),
                "d".to_string(),
            ],
        )
        .unwrap();

        let formula = parse_formula("y ~ 1 + (1 + x | g)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.set_theta(&[0.5, 0.0, 0.25]).unwrap();
        model.update_l().unwrap();
        model.optsum.feval = 1;

        let summary = ModelSummary::from_linear_model(&model);
        let labels = summary
            .rows
            .iter()
            .map(|row| row.label.as_str())
            .collect::<Vec<_>>();

        assert_eq!(summary.sigma_headers, vec!["σ_g".to_string()]);
        assert!(labels.contains(&"(Intercept)"));
        assert!(labels.contains(&"x"));
        assert!(labels.contains(&"Residual"));
        assert_eq!(model.summary_markdown(), summary.to_markdown());
        assert_eq!(model.summary_html(), summary.to_html());
        assert_eq!(model.summary_latex(), summary.to_latex());
        assert_eq!(
            model.varcorr().to_markdown(),
            VarCorr::from_model(&model.reterms, model.sigma()).to_markdown()
        );
        assert_eq!(
            model.block_description().to_markdown(),
            crate::stats::BlockDescription::from_linear_model(&model).to_markdown()
        );
    }

    #[test]
    fn fit_summary_payload_for_lmm_is_versioned_and_serializable() {
        let mut data = DataFrame::new();
        data.add_numeric("y", vec![2.0, 2.5, 3.0, 3.2, 4.0, 4.4, 5.0, 5.1])
            .unwrap();
        data.add_numeric("x", vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
            .unwrap();
        data.add_categorical(
            "g",
            vec![
                "a".to_string(),
                "a".to_string(),
                "b".to_string(),
                "b".to_string(),
                "c".to_string(),
                "c".to_string(),
                "d".to_string(),
                "d".to_string(),
            ],
        )
        .unwrap();

        let formula = parse_formula("y ~ 1 + x + (1 | g)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.set_theta(&[0.5]).unwrap();
        model.update_l().unwrap();
        model.optsum.feval = 7;
        model.optsum.return_value = "FTOL_REACHED".to_string();

        let payload = FitSummaryPayload::from_linear_model(&model);
        assert_eq!(payload.schema_name, FIT_SUMMARY_SCHEMA);
        assert_eq!(payload.schema_version, FIT_SUMMARY_SCHEMA_VERSION);
        assert_eq!(payload.model_kind, "linear_mixed_model");
        assert_eq!(payload.optimizer_backend, "native");
        assert_eq!(payload.feval, 7);
        assert_eq!(payload.family, None);
        assert_eq!(payload.link, None);

        let json = serde_json::to_string(&payload).unwrap();
        let decoded: FitSummaryPayload = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.schema_name, FIT_SUMMARY_SCHEMA);
        assert_eq!(decoded.schema_version, FIT_SUMMARY_SCHEMA_VERSION);
        assert_eq!(decoded.coefficients.names, payload.coefficients.names);
        assert_eq!(
            decoded.varcorr.components.len(),
            payload.varcorr.components.len()
        );
        assert_eq!(
            decoded
                .summary
                .rows
                .iter()
                .map(|row| &row.label)
                .collect::<Vec<_>>(),
            payload
                .summary
                .rows
                .iter()
                .map(|row| &row.label)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn summary_table_components_are_serde_roundtrippable() {
        let coefs = CoefTable::new(vec!["(Intercept)".to_string()], vec![1.25], vec![0.5]);
        let varcorr = VarCorr {
            components: vec![crate::stats::VarCorrComponent {
                group: "g".to_string(),
                names: vec!["(Intercept)".to_string()],
                std_dev: vec![0.75],
                correlations: Vec::new(),
            }],
            residual_sd: Some(1.0),
            residual_source: crate::model::summary_estimates::ResidualSource::EstimatedSigma,
        };
        let summary = ModelSummary {
            sigma_headers: vec!["σ_g".to_string()],
            rows: vec![row(
                "(Intercept)",
                Some(1.25),
                Some(0.5),
                Some(2.5),
                Some(0.012),
                &[Some(0.75)],
            )],
        };

        let coefs_rt: CoefTable =
            serde_json::from_str(&serde_json::to_string(&coefs).unwrap()).unwrap();
        let varcorr_rt: VarCorr =
            serde_json::from_str(&serde_json::to_string(&varcorr).unwrap()).unwrap();
        let summary_rt: ModelSummary =
            serde_json::from_str(&serde_json::to_string(&summary).unwrap()).unwrap();

        assert_eq!(coefs_rt, coefs);
        assert_eq!(varcorr_rt, varcorr);
        assert_eq!(summary_rt, summary);
    }

    #[test]
    fn test_from_generalized_model_omits_residual_for_bernoulli() {
        let mut data = DataFrame::new();
        data.add_numeric("y", vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
            .unwrap();
        data.add_numeric("x", vec![-1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
            .unwrap();
        data.add_categorical(
            "g",
            vec![
                "a".to_string(),
                "a".to_string(),
                "b".to_string(),
                "b".to_string(),
                "c".to_string(),
                "c".to_string(),
            ],
        )
        .unwrap();

        let formula = parse_formula("y ~ 1 + x + (1 | g)").unwrap();
        let mut model = GeneralizedLinearMixedModel::new(
            formula,
            &data,
            Family::Bernoulli,
            Some(LinkFunction::Logit),
        )
        .unwrap();
        model.lmm.set_theta(&[0.75]).unwrap();
        model.lmm.update_l().unwrap();
        model.beta[0] = 0.2;
        model.beta[1] = 0.1;
        model.lmm.optsum.feval = 1;

        let summary = ModelSummary::from_generalized_model(&model);
        let labels = summary
            .rows
            .iter()
            .map(|row| row.label.as_str())
            .collect::<Vec<_>>();

        assert_eq!(summary.sigma_headers, vec!["σ_g".to_string()]);
        assert!(labels.contains(&"(Intercept)"));
        assert!(labels.contains(&"x"));
        assert!(!labels.contains(&"Residual"));
        assert!(!labels.contains(&"Dispersion"));
        assert_eq!(model.summary_markdown(), summary.to_markdown());
        assert_eq!(model.summary_html(), summary.to_html());
        assert_eq!(model.summary_latex(), summary.to_latex());
        assert_eq!(
            model.varcorr().to_markdown(),
            VarCorr::from_reterms(&model.lmm.reterms, model.dispersion(false), None).to_markdown()
        );
        assert_eq!(
            model.block_description().to_markdown(),
            crate::stats::BlockDescription::from_generalized_model(&model).to_markdown()
        );
    }

    #[test]
    fn fit_summary_payload_for_glmm_names_family_and_link() {
        let mut data = DataFrame::new();
        data.add_numeric("y", vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
            .unwrap();
        data.add_numeric("x", vec![-1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
            .unwrap();
        data.add_categorical(
            "g",
            vec![
                "a".to_string(),
                "a".to_string(),
                "b".to_string(),
                "b".to_string(),
                "c".to_string(),
                "c".to_string(),
            ],
        )
        .unwrap();

        let formula = parse_formula("y ~ 1 + x + (1 | g)").unwrap();
        let mut model = GeneralizedLinearMixedModel::new(
            formula,
            &data,
            Family::Bernoulli,
            Some(LinkFunction::Logit),
        )
        .unwrap();
        model.lmm.set_theta(&[0.75]).unwrap();
        model.lmm.update_l().unwrap();
        model.beta[0] = 0.2;
        model.beta[1] = 0.1;
        model.lmm.optsum.feval = 3;

        let payload = FitSummaryPayload::from_generalized_model(&model);
        assert_eq!(payload.model_kind, "generalized_linear_mixed_model");
        assert_eq!(payload.family.as_deref(), Some("bernoulli"));
        assert_eq!(payload.link.as_deref(), Some("logit"));
        assert_eq!(
            payload.estimation_method.as_deref(),
            Some("fast_pirls_profiled")
        );
        assert_eq!(
            payload.objective_definition.as_deref(),
            Some("profiled_glmm_deviance")
        );
        assert_eq!(payload.response_constants.as_deref(), Some("dropped"));
        assert_eq!(payload.n_agq, Some(1));
        assert_eq!(payload.fallback_status, None);
        assert_eq!(payload.schema_version, FIT_SUMMARY_SCHEMA_VERSION);
    }
}
