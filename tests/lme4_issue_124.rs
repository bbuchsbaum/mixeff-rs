use mixeff_rs::compiler::{DiagnosticCode, DiagnosticSeverity};
use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::{DataFrame, Family, GeneralizedLinearMixedModel, MixedModelFit};

// toy: lme4 GH-124 reproducer with one `x2 = 1.0` outlier row.
fn lme4_issue_124_complete_separation_data() -> DataFrame {
    lme4_issue_124_data(&[(999, 0.0)])
}

// toy helper for `lme4_issue_124_complete_separation_data`: 1000 rows,
// 20 groups × 50 obs, deterministic Bernoulli-ish y; row count and
// structure mirror the upstream lme4 issue (GH-124).
fn lme4_issue_124_data(x2_rows: &[(usize, f64)]) -> DataFrame {
    let mut y = Vec::with_capacity(1000);
    let mut x = Vec::with_capacity(1000);
    let mut x2 = Vec::with_capacity(1000);
    let mut f = Vec::with_capacity(1000);

    for row in 0..1000 {
        let group = row / 50 + 1;
        f.push(group.to_string());

        // Deterministic analogue of the issue's runif(1000), with enough
        // irregularity to avoid coupling this test to a sorted predictor.
        let x_value = (((row * 37 + 17) % 1000) as f64 + 0.5) / 1000.0;
        x.push(x_value);

        // Balanced Bernoulli-like response with local variation in every group.
        let baseline_y = if (row * 29 + 11) % 100 < 50 { 1.0 } else { 0.0 };
        let y_value = x2_rows
            .iter()
            .find_map(|&(x2_row, forced_y)| (x2_row == row).then_some(forced_y))
            .unwrap_or(baseline_y);
        y.push(y_value);

        x2.push(if x2_rows.iter().any(|&(x2_row, _)| x2_row == row) {
            1.0
        } else {
            0.0
        });
    }

    let mut data = DataFrame::new();
    data.add_numeric("y", y).unwrap();
    data.add_numeric("x", x).unwrap();
    data.add_numeric("x2", x2).unwrap();
    data.add_categorical("f", f).unwrap();
    data
}

fn fit_issue_124_formula(formula: &str) -> GeneralizedLinearMixedModel {
    let data = lme4_issue_124_complete_separation_data();
    fit_formula_on_data(formula, &data)
}

fn fit_formula_on_data(formula: &str, data: &DataFrame) -> GeneralizedLinearMixedModel {
    let formula_text = formula.to_string();
    let formula = parse_formula(formula).unwrap();
    let mut model = GeneralizedLinearMixedModel::new(formula, data, Family::Binomial, None)
        .unwrap_or_else(|error| panic!("failed to construct `{formula_text}`: {error}"));
    model
        .fit_with_options(true, 1, false)
        .unwrap_or_else(|error| panic!("failed to fit `{formula_text}`: {error}"));
    model
}

#[test]
fn lme4_issue_124_near_separation_fits_with_finite_parameters() {
    let baseline = fit_issue_124_formula("y ~ 1 + x + (1 | f)");
    let mut separated = fit_issue_124_formula("y ~ 1 + x + x2 + (1 | f)");

    assert!(
        baseline.fixef().iter().all(|value| value.is_finite()),
        "baseline fixed effects should be finite: {:?}",
        baseline.fixef()
    );
    assert!(
        separated.fixef().iter().all(|value| value.is_finite()),
        "separation-stress fixed effects should be finite: {:?}",
        separated.fixef()
    );
    assert!(
        separated.theta().iter().all(|value| value.is_finite()),
        "theta should remain finite: {:?}",
        separated.theta()
    );

    let x2_beta = separated.fixef()[2];
    assert!(
        x2_beta < -5.0,
        "isolated zero outcome at x2=1 should drive a large negative coefficient, got {x2_beta}"
    );
    assert!(
        separated.deviance(1).is_finite(),
        "Laplace deviance should remain finite"
    );

    let diagnostic = separated
        .compiler_artifact()
        .diagnostics
        .iter()
        .find(|diagnostic| diagnostic.code == DiagnosticCode::BinomialSeparation)
        .expect("issue-shaped fit should surface a fixed-effect separation diagnostic");
    assert_eq!(diagnostic.severity, DiagnosticSeverity::Warning);
    assert_eq!(diagnostic.affected_terms, vec!["x2".to_string()]);
    assert!(diagnostic.message.contains("x2 = 1"));
    assert!(diagnostic.message.contains("all such rows have y = 0"));
    assert_eq!(
        diagnostic.payload.get("n_at_value"),
        Some(&serde_json::json!(1))
    );
    assert_eq!(
        diagnostic.payload.get("separation_kind"),
        Some(&serde_json::json!("quasi_complete_fixed_effect"))
    );
}

#[test]
fn lme4_issue_124_diagnostic_does_not_fire_for_rare_mixed_x2_level() {
    let data = lme4_issue_124_data(&[(998, 1.0), (999, 0.0)]);
    let model = fit_formula_on_data("y ~ 1 + x + x2 + (1 | f)", &data);

    assert!(
        !model
            .compiler_artifact()
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == DiagnosticCode::BinomialSeparation),
        "rare x2 support with both outcomes should not be called separation"
    );
}

#[test]
fn lme4_issue_124_diagnostic_does_not_fire_for_baseline_model() {
    let model = fit_issue_124_formula("y ~ 1 + x + (1 | f)");

    assert!(
        !model
            .compiler_artifact()
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == DiagnosticCode::BinomialSeparation),
        "baseline model has no binary fixed-effect column with one-sided outcome support"
    );
}
