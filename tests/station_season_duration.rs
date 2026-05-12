use std::collections::HashMap;

use mixeff_rs::compiler::{
    DiagnosticCode, EffectiveRankStatus, FitStatus, InformationBudgetStatus,
};
use mixeff_rs::datasets;
use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::LinearMixedModel;

fn forum_formula() -> &'static str {
    "effect ~ 1 + duration + (1 + duration | sites) + (1 + duration | season)"
}

#[test]
fn forum_model_is_flagged_as_over_requested_before_fit() {
    let (data, _) = datasets::load("station_season_duration").unwrap();
    let formula = parse_formula(forum_formula()).unwrap();
    let model = LinearMixedModel::new(formula, &data, None).unwrap();
    let audit = model.design_audit().expect("design audit should attach");

    let by_group: HashMap<_, _> = audit
        .random_terms
        .iter()
        .map(|term| (term.group.name.as_str(), term))
        .collect();

    for group in ["sites", "season"] {
        let term = by_group
            .get(group)
            .unwrap_or_else(|| panic!("missing random term for {group}"));
        assert_eq!(term.group.n_levels, Some(3));
        assert_eq!(term.basis_size, 2);
        assert_eq!(term.requested_covariance_parameters, 3);
        assert_eq!(
            term.information_budget.status,
            InformationBudgetStatus::TooRich,
            "{group} should be too rich because three levels cannot support a \
             two-dimensional correlated random-effect covariance"
        );
        assert_eq!(
            term.information_budget
                .effective_n
                .levels_per_covariance_parameter,
            Some(1.0)
        );
        assert!(term
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == DiagnosticCode::CovarianceTooRich));
    }

    let report_text = model.audit_report().to_text();
    assert!(report_text.contains("Random-Effect Information Budget"));
    assert!(report_text.contains("sites"));
    assert!(report_text.contains("season"));
    assert!(report_text.contains("levels=3"));
    assert!(report_text.contains("budget=too_rich"));
    assert!(report_text.contains("too rich"));
}

#[test]
fn forum_model_fit_is_not_reported_as_clean_interior_convergence() {
    let (data, _) = datasets::load("station_season_duration").unwrap();
    let formula = parse_formula(forum_formula()).unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

    model.fit(false).unwrap();

    let certificate = model
        .optimizer_certificate()
        .expect("optimizer certificate should attach after fit");
    assert_ne!(
        certificate.status,
        FitStatus::ConvergedInterior,
        "the user-facing answer should not be ordinary clean convergence; \
         the requested covariance structure is unsupported by the three-level \
         site and season grouping factors"
    );
    assert!(
        matches!(
            certificate.status,
            FitStatus::ConvergedBoundary
                | FitStatus::ConvergedReducedRank
                | FitStatus::NotIdentifiable
                | FitStatus::NotOptimized
        ),
        "unexpected fit status for over-requested forum model: {:?}",
        certificate.status
    );
    assert!(model
        .compiler_artifact()
        .effective_covariance
        .iter()
        .any(|summary| summary.status == EffectiveRankStatus::ReducedRank));

    let fitted_report = model.audit_report().to_text();
    assert!(fitted_report.contains("too rich"));
    assert!(fitted_report.contains("convergence interpretation"));
    assert!(
        fitted_report.contains("ConvergedReducedRank")
            || fitted_report.contains("ConvergedBoundary")
            || fitted_report.contains("NotIdentifiable")
            || fitted_report.contains("NotOptimized"),
        "fitted audit report should name the non-interior status:\n{fitted_report}"
    );
}
