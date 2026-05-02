use approx::assert_relative_eq;
use serde::Deserialize;

use mixedmodels::formula::parse_formula;
use mixedmodels::model::data::DataFrame;
use mixedmodels::model::generalized::GeneralizedLinearMixedModel;
use mixedmodels::model::traits::{Family, LinkFunction, MixedModelFit};
#[cfg(not(feature = "nlopt"))]
use mixedmodels::types::Optimizer;

#[allow(dead_code)]
#[derive(Deserialize)]
struct GammaGlmmFixture {
    schema_version: String,
    source: String,
    formula: String,
    family: String,
    link: String,
    n_agq: usize,
    nobs: usize,
    dof: usize,
    data_recipe: DataRecipe,
    rust_reference: FitReference,
    engines: Vec<EngineReference>,
    notes: Vec<String>,
}

#[derive(Deserialize)]
struct DataRecipe {
    groups: usize,
    observations_per_group: usize,
    intercept: f64,
    slope: f64,
    group_effects: Vec<f64>,
    wiggle_base: f64,
    wiggle_step: f64,
    wiggle_modulus: usize,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct FitReference {
    beta: Vec<f64>,
    theta: Vec<f64>,
    dispersion_sigma: f64,
    dispersion_phi: f64,
    objective: f64,
    loglik: f64,
    fitted_mu_head: Vec<f64>,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct EngineReference {
    engine: String,
    status: String,
    version: Option<String>,
    beta: Option<Vec<f64>>,
    theta: Option<Vec<f64>>,
    dispersion: Option<f64>,
    objective: Option<f64>,
    loglik: Option<f64>,
    verdict: String,
    note: String,
}

fn fixture() -> GammaGlmmFixture {
    serde_json::from_str(include_str!("fixtures/parity/gamma_glmm_engines.json")).unwrap()
}

// toy: matches `fixtures/parity/gamma_glmm_engines.json`; row order is
// part of the parity assertion (see `reversed_gamma_log_data` invariance test).
fn gamma_log_data() -> DataFrame {
    let expected = fixture();
    let recipe = expected.data_recipe;

    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut group = Vec::new();
    for g in 0..recipe.groups {
        for obs in 0..recipe.observations_per_group {
            let xv = obs as f64 - 2.0;
            let eta = recipe.intercept + recipe.slope * xv + recipe.group_effects[g];
            let wiggle = recipe.wiggle_base
                + recipe.wiggle_step * ((g + obs) % recipe.wiggle_modulus) as f64;
            y.push(eta.exp() * wiggle);
            x.push(xv);
            group.push(format!("g{}", g + 1));
        }
    }

    let mut data = DataFrame::new();
    data.add_numeric("y", y).unwrap();
    data.add_numeric("x", x).unwrap();
    data.add_categorical("group", group).unwrap();
    data
}

// toy: row-reversed `gamma_log_data`; tests fit-invariance to row order.
fn reversed_gamma_log_data() -> DataFrame {
    let data = gamma_log_data();
    let mut indices = (0..data.nrow()).collect::<Vec<_>>();
    indices.reverse();

    let mut reversed = DataFrame::new();
    reversed
        .add_numeric(
            "y",
            indices
                .iter()
                .map(|&idx| data.numeric("y").unwrap()[idx])
                .collect(),
        )
        .unwrap();
    reversed
        .add_numeric(
            "x",
            indices
                .iter()
                .map(|&idx| data.numeric("x").unwrap()[idx])
                .collect(),
        )
        .unwrap();
    reversed
        .add_categorical(
            "group",
            indices
                .iter()
                .map(|&idx| data.categorical("group").unwrap().values[idx].clone())
                .collect(),
        )
        .unwrap();
    reversed
}

fn fit_gamma_log(data: &DataFrame, formula: &str, n_agq: usize) -> GeneralizedLinearMixedModel {
    let formula = parse_formula(formula).unwrap();
    let mut model =
        GeneralizedLinearMixedModel::new(formula, data, Family::Gamma, Some(LinkFunction::Log))
            .unwrap();
    model.fit_with_options(true, n_agq, false).unwrap();
    model
}

#[cfg(feature = "nlopt")]
#[test]
fn test_gamma_log_glmm_matches_mixedmodels_jl_fixture() {
    let expected = fixture();
    assert_eq!(expected.schema_version, "1.0.0");
    assert!(expected.source.contains("MixedModels.jl"));
    assert_eq!(expected.formula, "y ~ 1 + x + (1 | group)");
    assert_eq!(expected.family, "gamma");
    assert_eq!(expected.link, "log");

    let data = gamma_log_data();
    assert_eq!(data.nrow(), expected.nobs);

    let model = fit_gamma_log(&data, &expected.formula, expected.n_agq);

    assert_eq!(model.nobs(), expected.nobs);
    assert_eq!(model.dof(), expected.dof);
    assert_eq!(model.theta().len(), expected.rust_reference.theta.len());
    assert_eq!(model.fixef().len(), expected.rust_reference.beta.len());

    for (actual, want) in model
        .theta()
        .iter()
        .zip(expected.rust_reference.theta.iter())
    {
        assert_relative_eq!(*actual, *want, epsilon = 1e-12, max_relative = 1e-12);
    }
    for (actual, want) in model
        .fixef()
        .iter()
        .zip(expected.rust_reference.beta.iter())
    {
        assert_relative_eq!(*actual, *want, epsilon = 1e-10, max_relative = 1e-10);
    }

    assert_relative_eq!(
        model.dispersion(false),
        expected.rust_reference.dispersion_sigma,
        epsilon = 1e-12,
        max_relative = 1e-12
    );
    assert_relative_eq!(
        model.dispersion(true),
        expected.rust_reference.dispersion_phi,
        epsilon = 1e-12,
        max_relative = 1e-12
    );
    assert_relative_eq!(
        model.objective(),
        expected.rust_reference.objective,
        epsilon = 1e-10,
        max_relative = 1e-10
    );
    assert_relative_eq!(
        model.loglikelihood(),
        expected.rust_reference.loglik,
        epsilon = 1e-10,
        max_relative = 1e-10
    );
    for (actual, want) in model
        .fitted()
        .iter()
        .take(expected.rust_reference.fitted_mu_head.len())
        .zip(expected.rust_reference.fitted_mu_head.iter())
    {
        assert_relative_eq!(*actual, *want, epsilon = 1e-10, max_relative = 1e-10);
    }

    let julia = expected
        .engines
        .iter()
        .find(|engine| engine.engine == "MixedModels.jl")
        .expect("fixture records MixedModels.jl reference");
    assert_eq!(julia.status, "fit");
    assert_eq!(julia.verdict, "parity_reference");
    for (actual, want) in model.fixef().iter().zip(julia.beta.as_ref().unwrap()) {
        assert_relative_eq!(*actual, *want, epsilon = 2e-5, max_relative = 2e-5);
    }
    for (actual, want) in model.theta().iter().zip(julia.theta.as_ref().unwrap()) {
        assert_relative_eq!(*actual, *want, epsilon = 1e-7);
    }
    assert_relative_eq!(
        model.objective(),
        julia.objective.unwrap(),
        epsilon = 1e-7,
        max_relative = 1e-7
    );

    let lme4 = expected
        .engines
        .iter()
        .find(|engine| engine.engine == "lme4::glmer")
        .expect("fixture records lme4 reference");
    assert_eq!(lme4.status, "fit");
    assert_eq!(lme4.verdict, "documented_divergence");
    assert!(lme4.version.as_deref().unwrap_or("").contains("lme4"));
    assert!(
        lme4.theta.as_ref().unwrap()[0] > 1.0,
        "glmer's Gamma dispersion profiling should remain documented as a non-oracle divergence"
    );
    assert!(lme4.beta.as_ref().unwrap()[0].is_finite());
    assert!(lme4.dispersion.unwrap().is_finite());
    assert!(lme4.loglik.unwrap().is_finite());

    let glmm_tmb = expected
        .engines
        .iter()
        .find(|engine| engine.engine == "glmmTMB")
        .expect("fixture records glmmTMB availability");
    assert_eq!(glmm_tmb.status, "unavailable");
    assert_eq!(glmm_tmb.verdict, "not_run");
    assert!(glmm_tmb.note.contains("not installed"));

    assert!(expected.notes.iter().any(|note| note.contains("glmer")));
}

#[cfg(not(feature = "nlopt"))]
#[test]
fn test_gamma_log_glmm_native_cobyla_preserves_fixture_contract() {
    let expected = fixture();
    let data = gamma_log_data();
    let model = fit_gamma_log(&data, &expected.formula, expected.n_agq);

    assert_eq!(expected.schema_version, "1.0.0");
    assert!(expected.source.contains("MixedModels.jl"));
    assert_eq!(expected.family, "gamma");
    assert_eq!(expected.link, "log");
    assert_eq!(model.nobs(), expected.nobs);
    assert_eq!(model.dof(), expected.dof);
    assert_eq!(model.lmm.optsum.optimizer, Optimizer::Cobyla);
    assert_eq!(model.lmm.optsum.backend.label(), "native");
    assert_eq!(model.theta().len(), expected.rust_reference.theta.len());
    assert_eq!(model.fixef().len(), expected.rust_reference.beta.len());
    assert!(model.objective().is_finite());
    assert!(model.loglikelihood().is_finite());
    assert!(model.dispersion(false).is_finite());
    assert!(model.dispersion(true).is_finite());
    for fitted in model
        .fitted()
        .iter()
        .take(expected.rust_reference.fitted_mu_head.len())
    {
        assert!(fitted.is_finite());
        assert!(*fitted > 0.0);
    }
}

#[test]
fn test_gamma_log_fit_is_invariant_to_row_order() {
    let expected = fixture();
    let ordered = fit_gamma_log(&gamma_log_data(), &expected.formula, expected.n_agq);
    let reversed = fit_gamma_log(
        &reversed_gamma_log_data(),
        &expected.formula,
        expected.n_agq,
    );

    assert_relative_eq!(
        ordered.objective(),
        reversed.objective(),
        epsilon = 1e-8,
        max_relative = 1e-8
    );
    assert_relative_eq!(
        ordered.dispersion(true),
        reversed.dispersion(true),
        epsilon = 1e-8,
        max_relative = 1e-8
    );
    for (actual, want) in ordered.fixef().iter().zip(reversed.fixef().iter()) {
        assert_relative_eq!(*actual, *want, epsilon = 1e-8, max_relative = 1e-8);
    }
}
