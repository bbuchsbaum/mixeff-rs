use serde::{Deserialize, Serialize};

use super::audit::{DesignAudit, InformationBudgetStatus, RandomTermAudit};
use super::diagnostics::{Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticStage};
use super::ir::{CovarianceForm, InterceptPolicy, RandomTermIr, SemanticModel};

pub const DEFAULT_CONVERGENCE_DERIVATIVE_NPARMAX: usize = 10;
pub const DEFAULT_CHOLESKY_ZERO_PAD_TOLERANCE: f64 = f64::EPSILON;

fn default_cholesky_zero_pad_tolerance() -> f64 {
    DEFAULT_CHOLESKY_ZERO_PAD_TOLERANCE
}

fn is_default_cholesky_zero_pad_tolerance(value: &f64) -> bool {
    value.to_bits() == DEFAULT_CHOLESKY_ZERO_PAD_TOLERANCE.to_bits()
}

/// Deterministic compiler policy used for v0 recommendations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompilerPolicy {
    pub random_strategy: RandomStrategy,
    pub thresholds: CompilerThresholds,
    pub apply_design_time_reductions: bool,
}

impl Default for CompilerPolicy {
    fn default() -> Self {
        Self {
            random_strategy: RandomStrategy::MaximalFeasible,
            thresholds: CompilerThresholds::default(),
            apply_design_time_reductions: false,
        }
    }
}

impl CompilerPolicy {
    pub fn as_specified() -> Self {
        Self {
            random_strategy: RandomStrategy::AsSpecified,
            thresholds: CompilerThresholds::default(),
            apply_design_time_reductions: false,
        }
    }

    pub fn maximal_feasible() -> Self {
        Self {
            random_strategy: RandomStrategy::MaximalFeasible,
            thresholds: CompilerThresholds::default(),
            apply_design_time_reductions: true,
        }
    }

    pub fn maximal_feasible_advisory() -> Self {
        Self::default()
    }

    pub fn design_compiled() -> Self {
        Self::maximal_feasible()
    }

    pub fn predictive() -> Self {
        Self {
            random_strategy: RandomStrategy::Predictive,
            thresholds: CompilerThresholds::default(),
            apply_design_time_reductions: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RandomStrategy {
    AsSpecified,
    MaximalFeasible,
    Regularized,
    Predictive,
}

/// Named v0 thresholds. These defaults mirror the PRD and must remain
/// deterministic for identical formula/data/policy inputs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompilerThresholds {
    pub min_levels_random_intercept_fit: usize,
    pub min_levels_random_intercept_reliability: usize,
    pub max_condition_number: f64,
    pub min_within_group_sd: f64,
    pub max_basis_pairwise_abs_corr: f64,
    pub min_observations_per_supported_level: usize,
    pub effective_rank_relative_tolerance: f64,
    pub effective_rank_absolute_tolerance: f64,
    pub convergence_derivative_nparmax: usize,
    #[serde(
        default = "default_cholesky_zero_pad_tolerance",
        skip_serializing_if = "is_default_cholesky_zero_pad_tolerance"
    )]
    pub cholesky_zero_pad_tolerance: f64,
}

impl Default for CompilerThresholds {
    fn default() -> Self {
        Self {
            min_levels_random_intercept_fit: 2,
            min_levels_random_intercept_reliability: 5,
            max_condition_number: 1e10,
            min_within_group_sd: 1e-8,
            max_basis_pairwise_abs_corr: 0.999,
            min_observations_per_supported_level: 2,
            effective_rank_relative_tolerance: 1e-6,
            effective_rank_absolute_tolerance: 1e-10,
            convergence_derivative_nparmax: DEFAULT_CONVERGENCE_DERIVATIVE_NPARMAX,
            cholesky_zero_pad_tolerance: DEFAULT_CHOLESKY_ZERO_PAD_TOLERANCE,
        }
    }
}

impl CompilerThresholds {
    pub fn min_levels_variance(&self, basis_dimension: usize) -> usize {
        5.max(2 * basis_dimension + 1)
    }

    pub fn min_levels_full_covariance(&self, covariance_parameters: usize) -> usize {
        10.max(5 * covariance_parameters)
    }

    pub fn effective_rank_tolerance(&self, max_eigenvalue: f64) -> f64 {
        self.effective_rank_absolute_tolerance
            .max(self.effective_rank_relative_tolerance * max_eigenvalue.max(0.0))
    }

    pub fn reproducibility_entries(&self) -> Vec<(String, String)> {
        let mut entries = vec![
            (
                "min_levels_random_intercept_fit".to_string(),
                self.min_levels_random_intercept_fit.to_string(),
            ),
            (
                "min_levels_random_intercept_reliability".to_string(),
                self.min_levels_random_intercept_reliability.to_string(),
            ),
            (
                "min_levels_variance".to_string(),
                "max(5, 2*d_basis + 1)".to_string(),
            ),
            (
                "min_levels_full_cov".to_string(),
                "max(10, 5*n_cov_params)".to_string(),
            ),
            (
                "max_condition_number".to_string(),
                self.max_condition_number.to_string(),
            ),
            (
                "min_within_group_sd".to_string(),
                self.min_within_group_sd.to_string(),
            ),
            (
                "max_basis_pairwise_abs_corr".to_string(),
                self.max_basis_pairwise_abs_corr.to_string(),
            ),
            (
                "min_observations_per_supported_level".to_string(),
                self.min_observations_per_supported_level.to_string(),
            ),
            (
                "effective_rank_relative_tolerance".to_string(),
                self.effective_rank_relative_tolerance.to_string(),
            ),
            (
                "effective_rank_absolute_tolerance".to_string(),
                self.effective_rank_absolute_tolerance.to_string(),
            ),
            (
                "convergence_derivative_nparmax".to_string(),
                self.convergence_derivative_nparmax.to_string(),
            ),
        ];
        if !is_default_cholesky_zero_pad_tolerance(&self.cholesky_zero_pad_tolerance) {
            entries.push((
                "cholesky_zero_pad_tolerance".to_string(),
                self.cholesky_zero_pad_tolerance.to_string(),
            ));
        }
        entries
    }
}

/// An advisory policy decision. This does not mean the fitted model changed;
/// actual model changes must be recorded separately as ReductionRecord values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyRecommendation {
    pub term_id: String,
    pub source_syntax: String,
    pub action: PolicyAction,
    pub reason: String,
    pub current_covariance: String,
    pub recommended_covariance: Option<String>,
    pub inference_consequence: String,
    pub diagnostics: Vec<Diagnostic>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyAction {
    DropUnsupportedBasis,
    ReduceCovariance,
    RefuseRandomTermDistribution,
    MarkNotAssessable,
}

pub fn recommend_policy(
    semantic_model: &SemanticModel,
    design_audit: &DesignAudit,
    policy: &CompilerPolicy,
) -> Vec<PolicyRecommendation> {
    match policy.random_strategy {
        RandomStrategy::AsSpecified => Vec::new(),
        RandomStrategy::MaximalFeasible
        | RandomStrategy::Regularized
        | RandomStrategy::Predictive => semantic_model
            .random_terms
            .iter()
            .zip(design_audit.random_terms.iter())
            .flat_map(|(term, audit)| recommendations_for_term(term, audit, policy))
            .collect(),
    }
}

fn recommendations_for_term(
    term: &RandomTermIr,
    audit: &RandomTermAudit,
    policy: &CompilerPolicy,
) -> Vec<PolicyRecommendation> {
    let mut recommendations = Vec::new();

    let unsupported_basis = audit
        .basis
        .iter()
        .filter(|basis| basis.supported == Some(false))
        .map(|basis| basis.name.clone())
        .collect::<Vec<_>>();
    if !unsupported_basis.is_empty() {
        recommendations.push(drop_unsupported_basis_recommendation(
            term,
            audit,
            unsupported_basis,
        ));
    }

    let budget = &audit.information_budget;
    match budget.status {
        InformationBudgetStatus::NotAssessable => {
            recommendations.push(not_assessable_recommendation(term, audit));
        }
        InformationBudgetStatus::WeaklySupported => {
            if !is_fit_eligible_few_level_random_intercept(audit, policy) {
                recommendations.push(refuse_random_distribution_recommendation(
                    term,
                    audit,
                    budget.reason.clone().unwrap_or_else(|| {
                        "random-effect distribution is weakly supported".to_string()
                    }),
                ));
            }
        }
        InformationBudgetStatus::TooRich => {
            if let Some(n_levels) = budget.n_levels {
                if is_row_saturated_random_effect(audit) {
                    recommendations.push(refuse_random_distribution_recommendation(
                        term,
                        audit,
                        budget.reason.clone().unwrap_or_else(|| {
                            "random-effect coefficients saturate the available rows".to_string()
                        }),
                    ));
                    return recommendations;
                }
                if is_scalar_random_intercept_budget(audit) {
                    recommendations.push(refuse_random_distribution_recommendation(
                        term,
                        audit,
                        budget.reason.clone().unwrap_or_else(|| {
                            format!(
                                "{n_levels} levels are below the v0 random-intercept fit threshold {}",
                                policy.thresholds.min_levels_random_intercept_fit
                            )
                        }),
                    ));
                    return recommendations;
                }
                let min_variance = policy
                    .thresholds
                    .min_levels_variance(budget.basis_dimension);
                if n_levels < min_variance {
                    recommendations.push(refuse_random_distribution_recommendation(
                        term,
                        audit,
                        format!(
                            "{n_levels} levels are below the v0 variance-direction threshold {min_variance}"
                        ),
                    ));
                } else if budget.covariance_family == "full" {
                    recommendations.push(reduce_full_covariance_recommendation(term, audit));
                }
            } else {
                recommendations.push(not_assessable_recommendation(term, audit));
            }
        }
        InformationBudgetStatus::Sufficient => {}
    }

    recommendations
}

fn is_fit_eligible_few_level_random_intercept(
    audit: &RandomTermAudit,
    policy: &CompilerPolicy,
) -> bool {
    is_scalar_random_intercept_budget(audit)
        && audit
            .information_budget
            .n_levels
            .map(|n_levels| {
                n_levels >= policy.thresholds.min_levels_random_intercept_fit
                    && n_levels < policy.thresholds.min_levels_random_intercept_reliability
            })
            .unwrap_or(false)
}

fn is_scalar_random_intercept_budget(audit: &RandomTermAudit) -> bool {
    audit.basis_size == 1
        && audit.information_budget.requested_covariance_parameters == 1
        && audit
            .basis
            .first()
            .map(|basis| basis.kind == "intercept")
            .unwrap_or(false)
}

fn is_row_saturated_random_effect(audit: &RandomTermAudit) -> bool {
    let effective_n = &audit.information_budget.effective_n;
    match (effective_n.n_rows, effective_n.n_levels) {
        (Some(rows), Some(levels)) => {
            let n_random_effects = levels.saturating_mul(effective_n.basis_dimension);
            effective_n.basis_dimension > 0 && rows <= n_random_effects
        }
        _ => false,
    }
}

fn drop_unsupported_basis_recommendation(
    term: &RandomTermIr,
    audit: &RandomTermAudit,
    unsupported_basis: Vec<String>,
) -> PolicyRecommendation {
    let recommended_covariance = if term.intercept == InterceptPolicy::Included
        && audit.basis_size > unsupported_basis.len()
    {
        Some("scalar_or_diagonal_on_supported_basis".to_string())
    } else {
        None
    };
    let reason = format!(
        "basis direction(s) unsupported by within-group variation: {}",
        unsupported_basis.join(", ")
    );
    let mut recommendation = recommendation(
        term,
        PolicyAction::DropUnsupportedBasis,
        reason,
        recommended_covariance,
        "fixed-effect inference would be conditional on the supported random-effect basis"
            .to_string(),
        DiagnosticCode::RandomSlopeUnsupported,
        DiagnosticSeverity::Warning,
    );
    recommendation.diagnostics[0].payload.insert(
        "unsupported_basis".to_string(),
        serde_json::json!(unsupported_basis),
    );
    recommendation
}

fn reduce_full_covariance_recommendation(
    term: &RandomTermIr,
    audit: &RandomTermAudit,
) -> PolicyRecommendation {
    let reason = audit
        .information_budget
        .reason
        .clone()
        .unwrap_or_else(|| "full covariance exceeds the v0 information budget".to_string());
    recommendation(
        term,
        PolicyAction::ReduceCovariance,
        reason,
        Some("diagonal".to_string()),
        "correlation parameters would not be interpreted as confirmatory".to_string(),
        DiagnosticCode::CovarianceTooRich,
        DiagnosticSeverity::Warning,
    )
}

fn refuse_random_distribution_recommendation(
    term: &RandomTermIr,
    audit: &RandomTermAudit,
    reason: String,
) -> PolicyRecommendation {
    let mut recommendation = recommendation(
        term,
        PolicyAction::RefuseRandomTermDistribution,
        reason,
        None,
        "confirmatory fixed-effect p-values should be withheld or recomputed after a declared design-level change"
            .to_string(),
        DiagnosticCode::CovarianceTooRich,
        DiagnosticSeverity::Warning,
    );
    recommendation.diagnostics[0].payload.insert(
        "n_levels".to_string(),
        serde_json::json!(audit.information_budget.n_levels),
    );
    recommendation
}

fn not_assessable_recommendation(
    term: &RandomTermIr,
    audit: &RandomTermAudit,
) -> PolicyRecommendation {
    recommendation(
        term,
        PolicyAction::MarkNotAssessable,
        audit
            .information_budget
            .reason
            .clone()
            .unwrap_or_else(|| "random-effect information budget is not assessable".to_string()),
        None,
        "no automatic confirmatory reduction is recommended by v0".to_string(),
        DiagnosticCode::Unsupported,
        DiagnosticSeverity::Info,
    )
}

fn recommendation(
    term: &RandomTermIr,
    action: PolicyAction,
    reason: String,
    recommended_covariance: Option<String>,
    inference_consequence: String,
    diagnostic_code: DiagnosticCode,
    severity: DiagnosticSeverity,
) -> PolicyRecommendation {
    let diagnostic = Diagnostic::new(
        diagnostic_code,
        severity,
        DiagnosticStage::DesignAudit,
        reason.clone(),
    )
    .with_affected_terms(vec![term.source_syntax.text.clone()]);

    PolicyRecommendation {
        term_id: term.id.clone(),
        source_syntax: term.source_syntax.text.clone(),
        action,
        reason,
        current_covariance: covariance_label(&term.covariance),
        recommended_covariance,
        inference_consequence,
        diagnostics: vec![diagnostic],
    }
}

fn covariance_label(covariance: &CovarianceForm) -> String {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::{audit_design, compile_formula_ir};
    use crate::formula::parse_formula;
    use crate::model::data::DataFrame;

    fn grouped_data(n_groups: usize) -> DataFrame {
        grouped_data_with_obs(n_groups, 2)
    }

    fn grouped_data_with_obs(n_groups: usize, obs_per_group: usize) -> DataFrame {
        let mut data = DataFrame::new();
        let mut y = Vec::new();
        let mut x = Vec::new();
        let mut group = Vec::new();
        for idx in 0..n_groups {
            for obs in 0..obs_per_group {
                y.push(idx as f64 + obs as f64);
                x.push(obs as f64);
                group.push(format!("g{}", idx + 1));
            }
        }
        data.add_numeric("y", y).unwrap();
        data.add_numeric("x", x).unwrap();
        data.add_categorical("group", group).unwrap();
        data
    }

    #[test]
    fn as_specified_policy_makes_no_recommendations() {
        let formula = parse_formula("y ~ x + (1 + x | group)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let audit = audit_design(&semantic, &grouped_data(2));

        let recommendations = recommend_policy(&semantic, &audit, &CompilerPolicy::as_specified());
        assert!(recommendations.is_empty());
    }

    #[test]
    fn maximal_feasible_refuses_random_distribution_with_too_few_levels() {
        let formula = parse_formula("y ~ x + (1 + x | group)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let audit = audit_design(&semantic, &grouped_data(2));

        let recommendations =
            recommend_policy(&semantic, &audit, &CompilerPolicy::maximal_feasible());
        assert!(recommendations.iter().any(|rec| {
            rec.action == PolicyAction::RefuseRandomTermDistribution && rec.term_id == "r0"
        }));
    }

    #[test]
    fn maximal_feasible_does_not_refuse_fit_eligible_few_level_random_intercept() {
        let formula = parse_formula("y ~ x + (1 | group)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let audit = audit_design(&semantic, &grouped_data(2));

        assert_eq!(
            audit.random_terms[0].information_budget.status,
            InformationBudgetStatus::WeaklySupported
        );
        let recommendations =
            recommend_policy(&semantic, &audit, &CompilerPolicy::maximal_feasible());
        assert!(
            recommendations.is_empty(),
            "fit-eligible low-reliability scalar random intercepts should warn in audit, not trigger design-time refusal"
        );
    }

    #[test]
    fn maximal_feasible_reduces_full_covariance_when_correlations_are_too_rich() {
        let formula = parse_formula("y ~ x + (1 + x | group)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let audit = audit_design(&semantic, &grouped_data_with_obs(6, 3));

        let recommendations =
            recommend_policy(&semantic, &audit, &CompilerPolicy::maximal_feasible());
        let rec = recommendations
            .iter()
            .find(|rec| rec.action == PolicyAction::ReduceCovariance)
            .expect("full covariance should be reduced");
        assert_eq!(rec.current_covariance, "full");
        assert_eq!(rec.recommended_covariance.as_deref(), Some("diagonal"));
    }

    #[test]
    fn maximal_feasible_refuses_row_saturated_random_distribution() {
        let formula = parse_formula("y ~ x + (1 + x | group)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let audit = audit_design(&semantic, &grouped_data(100));

        let recommendations =
            recommend_policy(&semantic, &audit, &CompilerPolicy::maximal_feasible());
        let rec = recommendations
            .iter()
            .find(|rec| rec.action == PolicyAction::RefuseRandomTermDistribution)
            .expect("row-saturated random term should be refused");
        assert!(rec.reason.contains("number of observations (200)"));
        assert!(rec.reason.contains("random coefficients (200)"));
    }

    #[test]
    fn maximal_feasible_drops_unsupported_slope_basis() {
        let mut data = DataFrame::new();
        data.add_numeric("y", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        data.add_numeric("x", vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
            .unwrap();
        data.add_categorical(
            "group",
            vec!["a", "a", "b", "b", "c", "c"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        )
        .unwrap();
        let formula = parse_formula("y ~ x + (1 + x | group)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let audit = audit_design(&semantic, &data);

        let recommendations =
            recommend_policy(&semantic, &audit, &CompilerPolicy::maximal_feasible());
        assert!(recommendations
            .iter()
            .any(|rec| rec.action == PolicyAction::DropUnsupportedBasis));
    }

    #[test]
    fn policy_recommendations_round_trip_json() {
        let formula = parse_formula("y ~ x + (1 + x | group)").unwrap();
        let semantic = compile_formula_ir(&formula);
        let audit = audit_design(&semantic, &grouped_data(6));
        let recommendations =
            recommend_policy(&semantic, &audit, &CompilerPolicy::maximal_feasible());

        let json = serde_json::to_string(&recommendations).unwrap();
        let decoded: Vec<PolicyRecommendation> = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, recommendations);
    }

    #[test]
    fn cholesky_zero_pad_tolerance_defaults_and_serializes_when_custom() {
        let default_thresholds = CompilerThresholds::default();
        let json = serde_json::to_string(&default_thresholds).unwrap();
        assert!(!json.contains("cholesky_zero_pad_tolerance"));

        let decoded: CompilerThresholds = serde_json::from_str(&json).unwrap();
        assert_eq!(
            decoded.cholesky_zero_pad_tolerance,
            DEFAULT_CHOLESKY_ZERO_PAD_TOLERANCE
        );

        let custom = CompilerThresholds {
            cholesky_zero_pad_tolerance: 0.0,
            ..CompilerThresholds::default()
        };
        let custom_json = serde_json::to_string(&custom).unwrap();
        assert!(custom_json.contains("cholesky_zero_pad_tolerance"));
        assert!(custom
            .reproducibility_entries()
            .iter()
            .any(|(name, value)| name == "cholesky_zero_pad_tolerance" && value == "0"));
    }
}
