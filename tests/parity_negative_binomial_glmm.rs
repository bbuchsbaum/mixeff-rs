use approx::assert_relative_eq;
use serde::Deserialize;

use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::data::DataFrame;
use mixeff_rs::model::generalized::{GeneralizedLinearMixedModel, GlmmFitOptions};
use mixeff_rs::model::linear::OptimizerControl;
use mixeff_rs::model::traits::{LinkFunction, MixedModelFit};
use mixeff_rs::types::Optimizer;

#[allow(dead_code)]
#[derive(Deserialize)]
struct NegativeBinomialFixture {
    schema_version: String,
    source: String,
    formula: String,
    family: String,
    link: String,
    n_agq: usize,
    nobs: usize,
    data_recipe: DataRecipe,
    data: FixtureData,
    lme4: Lme4Reference,
    tolerances: Tolerances,
    notes: Vec<String>,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct DataRecipe {
    groups: usize,
    observations_per_group: usize,
    intercept: f64,
    slope: f64,
    theta: f64,
    quantile_multiplier: usize,
    quantile_modulus: usize,
    quantile_denominator: usize,
}

#[derive(Deserialize)]
struct FixtureData {
    y: Vec<f64>,
    x: Vec<f64>,
    group: Vec<String>,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct Lme4Reference {
    engine: String,
    version: String,
    fixed_theta: FitReference,
    estimated_theta: FitReference,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct FitReference {
    theta: f64,
    beta: Vec<f64>,
    re_theta: Vec<f64>,
    loglik: f64,
    deviance: f64,
    fitted_mu_head: Vec<f64>,
    singular: bool,
}

#[derive(Deserialize)]
struct Tolerances {
    fixed_beta_abs: f64,
    fixed_loglik_abs: f64,
    estimated_beta_abs: f64,
    estimated_theta_abs: f64,
    estimated_loglik_abs: f64,
    fitted_mu_abs: f64,
    re_theta_abs: f64,
}

fn fixture() -> NegativeBinomialFixture {
    serde_json::from_str(include_str!("fixtures/parity/negative_binomial_glmm.json")).unwrap()
}

fn fixture_data() -> DataFrame {
    let expected = fixture();
    let mut data = DataFrame::new();
    data.add_numeric("y", expected.data.y).unwrap();
    data.add_numeric("x", expected.data.x).unwrap();
    data.add_categorical("group", expected.data.group).unwrap();
    data
}

fn fit_options(max_feval: usize) -> GlmmFitOptions {
    let control = OptimizerControl::auto()
        .with_optimizer(Optimizer::PatternSearch)
        .with_max_feval(max_feval);
    GlmmFitOptions::fast_laplace().with_optimizer_control(control)
}

#[test]
fn test_negative_binomial_fixed_theta_tracks_lme4_fixture() {
    let expected = fixture();
    assert_eq!(expected.schema_version, "1.0.0");
    assert!(expected.source.contains("lme4"));
    assert_eq!(expected.family, "negative_binomial");
    assert_eq!(expected.link, "log");

    let data = fixture_data();
    assert_eq!(data.nrow(), expected.nobs);
    let formula = parse_formula(&expected.formula).unwrap();
    let mut model = GeneralizedLinearMixedModel::new_negative_binomial(
        formula,
        &data,
        expected.lme4.fixed_theta.theta,
        Some(LinkFunction::Log),
    )
    .unwrap();

    model.fit_with_glmm_options(fit_options(160)).unwrap();

    assert_eq!(model.nobs(), expected.nobs);
    assert_eq!(
        model.negative_binomial_theta(),
        Some(expected.lme4.fixed_theta.theta)
    );
    assert!(!model.negative_binomial_theta_estimated());
    assert_eq!(model.dof(), model.fixef().len() + model.theta().len());
    assert_relative_eq!(
        model.loglikelihood(),
        expected.lme4.fixed_theta.loglik,
        epsilon = expected.tolerances.fixed_loglik_abs
    );
    for (actual, want) in model
        .fixef()
        .iter()
        .zip(expected.lme4.fixed_theta.beta.iter())
    {
        assert_relative_eq!(*actual, *want, epsilon = expected.tolerances.fixed_beta_abs);
    }
    for (actual, want) in model
        .theta()
        .iter()
        .zip(expected.lme4.fixed_theta.re_theta.iter())
    {
        assert_relative_eq!(*actual, *want, epsilon = expected.tolerances.re_theta_abs);
    }
    for (actual, want) in model
        .fitted()
        .iter()
        .take(expected.lme4.fixed_theta.fitted_mu_head.len())
        .zip(expected.lme4.fixed_theta.fitted_mu_head.iter())
    {
        assert_relative_eq!(*actual, *want, epsilon = expected.tolerances.fitted_mu_abs);
    }

    let metadata = model
        .compiler_artifact()
        .glmm_fit_metadata
        .as_ref()
        .expect("fixed NB fit should record metadata");
    assert_eq!(
        metadata
            .family_parameter_sources
            .get("negative_binomial_theta")
            .map(String::as_str),
        Some("fixed")
    );
}

#[test]
fn test_negative_binomial_estimated_theta_tracks_glmer_nb_fixture() {
    let expected = fixture();
    let data = fixture_data();
    let formula = parse_formula(&expected.formula).unwrap();
    let mut model = GeneralizedLinearMixedModel::new_negative_binomial_estimated(
        formula,
        &data,
        Some(expected.lme4.fixed_theta.theta),
        Some(LinkFunction::Log),
    )
    .unwrap();

    model.fit_with_glmm_options(fit_options(160)).unwrap();

    let theta = model
        .negative_binomial_theta()
        .expect("estimated NB fit should retain final theta");
    assert!(model.negative_binomial_theta_estimated());
    assert_relative_eq!(
        theta,
        expected.lme4.estimated_theta.theta,
        epsilon = expected.tolerances.estimated_theta_abs
    );
    assert_relative_eq!(
        model.loglikelihood(),
        expected.lme4.estimated_theta.loglik,
        epsilon = expected.tolerances.estimated_loglik_abs
    );
    for (actual, want) in model
        .fixef()
        .iter()
        .zip(expected.lme4.estimated_theta.beta.iter())
    {
        assert_relative_eq!(
            *actual,
            *want,
            epsilon = expected.tolerances.estimated_beta_abs
        );
    }
    for (actual, want) in model
        .fitted()
        .iter()
        .take(expected.lme4.estimated_theta.fitted_mu_head.len())
        .zip(expected.lme4.estimated_theta.fitted_mu_head.iter())
    {
        assert_relative_eq!(*actual, *want, epsilon = expected.tolerances.fitted_mu_abs);
    }

    let metadata = model
        .compiler_artifact()
        .glmm_fit_metadata
        .as_ref()
        .expect("estimated NB fit should record metadata");
    assert_eq!(
        metadata
            .family_parameter_sources
            .get("negative_binomial_theta")
            .map(String::as_str),
        Some("estimated")
    );
    assert!(metadata
        .family_parameters
        .get("negative_binomial_theta_outer_iterations")
        .is_some_and(|value| *value >= 1.0));
}

#[test]
fn test_negative_binomial_estimated_theta_joint_laplace_path_is_labelled() {
    let expected = fixture();
    let data = fixture_data();
    let formula = parse_formula(&expected.formula).unwrap();
    let mut model = GeneralizedLinearMixedModel::new_negative_binomial_estimated(
        formula,
        &data,
        Some(expected.lme4.fixed_theta.theta),
        Some(LinkFunction::Log),
    )
    .unwrap();
    let control = OptimizerControl::auto()
        .with_optimizer(Optimizer::TrustBq)
        .with_max_feval(40);

    model
        .fit_with_glmm_options(GlmmFitOptions::joint_laplace().with_optimizer_control(control))
        .unwrap();

    assert!(model.loglikelihood().is_finite());
    assert!(model
        .negative_binomial_theta()
        .is_some_and(|theta| theta > 0.0));
    let metadata = model
        .compiler_artifact()
        .glmm_fit_metadata
        .as_ref()
        .expect("joint NB fit should record metadata");
    assert!(
        matches!(
            metadata.estimation_method.as_str(),
            "joint_laplace" | "fallback_fast_pirls"
        ),
        "unexpected NB joint-path method label: {}",
        metadata.estimation_method
    );
    assert_eq!(
        metadata
            .family_parameter_sources
            .get("negative_binomial_theta")
            .map(String::as_str),
        Some("estimated")
    );
}
