#![cfg(feature = "unstable-internals")]

use mixeff_rs::datasets;
use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::{Family, GeneralizedLinearMixedModel, LinearMixedModel, MixedModelFit};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Fixture {
    tolerances: Tolerances,
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct Tolerances {
    beta_abs: f64,
    sigma_abs: f64,
    theta_abs: f64,
    objective_abs: f64,
    loglik_abs: f64,
    fitted_abs: f64,
    varcorr_sd_abs: f64,
    varcorr_corr_abs: f64,
}

#[derive(Debug, Deserialize)]
struct Case {
    id: String,
    rust_formula: String,
    reml: bool,
    covariance_family: String,
    beta: Vec<f64>,
    sigma: f64,
    theta: Vec<f64>,
    objective: f64,
    loglik: f64,
    fitted_head: Vec<f64>,
    varcorr: Vec<VarCorrRow>,
}

#[derive(Debug, Deserialize)]
struct VarCorrRow {
    group: String,
    var2: Option<String>,
    sdcor: f64,
}

fn fixture() -> Fixture {
    serde_json::from_str(include_str!(
        "fixtures/parity/lme4_covariance_families.json"
    ))
    .unwrap()
}

fn assert_close(label: &str, actual: f64, expected: f64, tol: f64) {
    let delta = (actual - expected).abs();
    assert!(
        delta <= tol,
        "{label}: actual={actual:.12}, expected={expected:.12}, delta={delta:.3e}, tol={tol:.3e}"
    );
}

fn assert_vec_close(label: &str, actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_close(&format!("{label}[{idx}]"), *actual, *expected, tol);
    }
}

#[test]
fn lme4_full_and_diagonal_covariance_fixtures_match() {
    let fixture = fixture();
    let (data, _) = datasets::load("sleepstudy").unwrap();

    for case in &fixture.cases {
        let formula = parse_formula(&case.rust_formula).unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(case.reml).unwrap();

        assert_vec_close(
            &format!("{} beta", case.id),
            MixedModelFit::coef(&model).as_slice(),
            &case.beta,
            fixture.tolerances.beta_abs,
        );
        assert_close(
            &format!("{} sigma", case.id),
            model.sigma(),
            case.sigma,
            fixture.tolerances.sigma_abs,
        );
        assert_vec_close(
            &format!("{} theta", case.id),
            &model.theta(),
            &case.theta,
            fixture.tolerances.theta_abs,
        );
        assert_close(
            &format!("{} objective", case.id),
            model.objective_value(),
            case.objective,
            fixture.tolerances.objective_abs,
        );
        assert_close(
            &format!("{} loglik", case.id),
            MixedModelFit::loglikelihood(&model),
            case.loglik,
            fixture.tolerances.loglik_abs,
        );
        let fitted = MixedModelFit::fitted(&model);
        assert_vec_close(
            &format!("{} fitted_head", case.id),
            &fitted.as_slice()[..case.fitted_head.len()],
            &case.fitted_head,
            fixture.tolerances.fitted_abs,
        );
        let varcorr = model.varcorr();
        let actual_sd = varcorr
            .components
            .iter()
            .flat_map(|component| component.std_dev.iter().copied())
            .collect::<Vec<_>>();
        let expected_sd = case
            .varcorr
            .iter()
            .filter(|row| row.group != "Residual" && row.var2.is_none())
            .map(|row| row.sdcor)
            .collect::<Vec<_>>();
        assert_vec_close(
            &format!("{} VarCorr sd", case.id),
            &actual_sd,
            &expected_sd,
            fixture.tolerances.varcorr_sd_abs,
        );
        let actual_corr = varcorr
            .components
            .iter()
            .flat_map(|component| component.correlations.iter().copied())
            .collect::<Vec<_>>();
        let mut expected_corr = case
            .varcorr
            .iter()
            .filter(|row| row.group != "Residual" && row.var2.is_some())
            .map(|row| row.sdcor)
            .collect::<Vec<_>>();
        if expected_corr.is_empty() && case.covariance_family == "diagonal" {
            expected_corr.resize(actual_corr.len(), 0.0);
        }
        assert_vec_close(
            &format!("{} VarCorr corr", case.id),
            &actual_corr,
            &expected_corr,
            fixture.tolerances.varcorr_corr_abs,
        );

        let artifact = serde_json::to_value(model.compiler_artifact()).unwrap();
        let theta_maps = artifact["theta_maps"].as_array().unwrap();
        assert_eq!(theta_maps.len(), 1, "{} theta-map count", case.id);
        assert_eq!(
            theta_maps[0]["map"]["support_status"], "supported",
            "{} support status",
            case.id
        );
        assert_eq!(
            theta_maps[0]["map"]["covariance_family"], case.covariance_family,
            "{} covariance family",
            case.id
        );
        assert_eq!(
            artifact["covariance_parameter_traces"]
                .as_array()
                .unwrap()
                .len(),
            case.theta.len(),
            "{} trace count",
            case.id
        );
    }
}

#[test]
fn structured_covariance_wrappers_are_parsed_refused_with_artifact_status() {
    let (data, _) = datasets::load("sleepstudy").unwrap();

    for formula_text in [
        "Reaction ~ 1 + Days + cs(1 + Days | Subject)",
        "Reaction ~ 1 + Days + ar1(0 + Days | Subject)",
    ] {
        let formula = parse_formula(formula_text).unwrap();
        let err = LinearMixedModel::new(formula.clone(), &data, None).unwrap_err();
        assert_eq!(err.code(), "unsupported");
        assert!(err.to_string().contains("not fitted in v1.0"));
        let glmm_err =
            GeneralizedLinearMixedModel::new(formula.clone(), &data, Family::Poisson, None)
                .unwrap_err();
        assert_eq!(glmm_err.code(), "unsupported");
        assert!(glmm_err.to_string().contains("not fitted in v1.0"));

        let semantic = mixeff_rs::compiler::compile_formula_ir(&formula);
        let mut artifact =
            mixeff_rs::compiler::CompiledModelArtifact::new(formula.to_string(), semantic);
        artifact.attach_design_audit(&data);
        let value = serde_json::to_value(&artifact).unwrap();
        assert_eq!(
            value["semantic_model"]["random_terms"][0]["covariance_support"],
            "parsed_refused"
        );
        assert_eq!(
            value["design_audit"]["covariance_kernels"]["kernels"][0]["support_status"],
            "parsed_refused"
        );
        assert_eq!(
            value["theta_maps"][0]["map"]["support_status"],
            "parsed_refused"
        );
        assert_eq!(
            value["covariance_parameter_traces"][0]["theta"]["status"],
            "not_assessed"
        );
    }
}
