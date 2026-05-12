use serde::Deserialize;

use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::{
    DataFrame, Family, GeneralizedLinearMixedModel, LinkFunction, MixedModelFit,
};

#[allow(dead_code)]
#[derive(Deserialize)]
struct GlmmLinkFixture {
    schema_version: String,
    source: String,
    cases: Vec<GlmmLinkCase>,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct GlmmLinkCase {
    id: String,
    family: String,
    link: String,
    formula: String,
    n_agq: usize,
    y: Vec<f64>,
    weights: Option<Vec<f64>>,
    x: Vec<f64>,
    group: Vec<String>,
    lme4: Lme4Reference,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct Lme4Reference {
    engine: String,
    version: String,
    status: String,
    beta: Vec<f64>,
    theta: Vec<f64>,
    objective: f64,
    deviance: f64,
    loglik: f64,
    aic: f64,
    bic: f64,
    fitted_mu_head: Vec<f64>,
    is_singular: bool,
}

fn fixture() -> GlmmLinkFixture {
    serde_json::from_str(include_str!("fixtures/parity/glmm_link_lme4.json")).unwrap()
}

fn data_frame(case: &GlmmLinkCase) -> DataFrame {
    let mut data = DataFrame::new();
    data.add_numeric("y", case.y.clone()).unwrap();
    data.add_numeric("x", case.x.clone()).unwrap();
    data.add_categorical("group", case.group.clone()).unwrap();
    data
}

fn family(case: &GlmmLinkCase) -> Family {
    match case.family.as_str() {
        "binomial" => Family::Binomial,
        "poisson" => Family::Poisson,
        other => panic!("unsupported fixture family {other}"),
    }
}

fn link(case: &GlmmLinkCase) -> LinkFunction {
    match case.link.as_str() {
        "probit" => LinkFunction::Probit,
        "cloglog" => LinkFunction::Cloglog,
        "sqrt" => LinkFunction::Sqrt,
        other => panic!("unsupported fixture link {other}"),
    }
}

fn fit_case(case: &GlmmLinkCase) -> GeneralizedLinearMixedModel {
    let data = data_frame(case);
    let formula = parse_formula(&case.formula).unwrap();
    let mut model = if let Some(weights) = &case.weights {
        GeneralizedLinearMixedModel::new_with_weights(
            formula,
            &data,
            family(case),
            Some(link(case)),
            weights.clone(),
        )
        .unwrap()
    } else {
        GeneralizedLinearMixedModel::new(formula, &data, family(case), Some(link(case))).unwrap()
    };
    model.fit_with_options(true, case.n_agq, false).unwrap();
    model
}

fn assert_close(id: &str, quantity: &str, actual: f64, want: f64, epsilon: f64, relative: f64) {
    let tolerance = epsilon.max(relative * want.abs());
    assert!(
        (actual - want).abs() <= tolerance,
        "{id} {quantity}: rust={actual}, lme4={want}, tolerance={tolerance}"
    );
}

#[test]
fn test_requested_glmm_links_match_lme4_fixtures() {
    let expected = fixture();
    assert_eq!(expected.schema_version, "1.0.0");
    assert!(expected.source.contains("lme4"));
    assert_eq!(expected.cases.len(), 3);

    for case in &expected.cases {
        assert_eq!(case.lme4.engine, "lme4::glmer");
        assert_eq!(case.lme4.status, "fit");
        assert!(case.lme4.version.contains("lme4"));

        let model = fit_case(case);
        assert_eq!(model.nobs(), case.y.len());
        assert_eq!(model.fixef().len(), case.lme4.beta.len());
        assert_eq!(model.theta().len(), case.lme4.theta.len());
        assert!(model.objective().is_finite(), "{} objective", case.id);
        assert!(model.loglikelihood().is_finite(), "{} loglik", case.id);
        assert!(model.fitted().iter().all(|value| value.is_finite()));

        for (actual, want) in model.fixef().iter().zip(case.lme4.beta.iter()) {
            assert_close(&case.id, "beta", *actual, *want, 8e-2, 8e-2);
        }
        for (actual, want) in model.theta().iter().zip(case.lme4.theta.iter()) {
            assert_close(&case.id, "theta", *actual, *want, 2.5e-1, 2.5e-1);
        }
        for (actual, want) in model
            .fitted()
            .iter()
            .take(case.lme4.fitted_mu_head.len())
            .zip(case.lme4.fitted_mu_head.iter())
        {
            assert_close(&case.id, "fitted mu", *actual, *want, 2e-1, 2e-1);
        }
    }
}
