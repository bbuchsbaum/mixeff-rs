#![cfg(not(feature = "nlopt"))]
#![cfg(feature = "unstable-internals")]

use mixeff_rs::datasets;
use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::{
    DataFrame, Family, GeneralizedLinearMixedModel, LinkFunction, MixedModelFit,
};
use mixeff_rs::types::Optimizer;

fn assert_native_glmm_fit(
    model: &GeneralizedLinearMixedModel,
    n_agq: usize,
    expected_beta: usize,
    expected_theta: usize,
) {
    assert_eq!(model.fixef().len(), expected_beta);
    assert_eq!(model.theta().len(), expected_theta);
    assert!(model.fixef().iter().all(|value| value.is_finite()));
    assert!(model.theta().iter().all(|value| value.is_finite()));
    assert!(model.objective().is_finite());
    assert!(model.loglikelihood().is_finite());
    assert!(model.dispersion(false).is_finite());
    assert!(model.dispersion(true).is_finite());
    assert_eq!(model.lmm().optsum().optimizer, Optimizer::Cobyla);
    assert_eq!(model.lmm().optsum().backend.label(), "native");
    assert_eq!(model.lmm().optsum().n_agq, n_agq);
    assert!(model.lmm().optsum().feval > 0);
    assert!(model.lmm().optsum().fmin.is_finite());
    assert!(!model.lmm().optsum().fit_log.is_empty());
    let certificate = model
        .compiler_artifact()
        .optimizer_certificate
        .as_ref()
        .expect("native GLMM fit should record an optimizer certificate");
    assert_eq!(certificate.optimizer_name.as_deref(), Some("cobyla"));
    assert!(certificate.objective_value.unwrap().is_finite());
    assert_eq!(certificate.evidence.parameter_space.n_theta, expected_theta);
    assert_eq!(
        certificate.evidence.sample_size.n_observations,
        Some(model.nobs())
    );
}

fn weighted_cbpp_model(n_agq: usize) -> GeneralizedLinearMixedModel {
    let (data, _) = datasets::load("cbpp").unwrap();
    let incidence = data.numeric("incidence").unwrap();
    let size = data.numeric("size").unwrap();
    let proportion: Vec<f64> = incidence
        .iter()
        .zip(size.iter())
        .map(|(&y, &n)| y / n)
        .collect();
    let weights = size.to_vec();
    let mut data = data.clone();
    data.add_numeric("proportion", proportion).unwrap();

    let formula = parse_formula("proportion ~ 1 + period + (1 | herd)").unwrap();
    let mut model = GeneralizedLinearMixedModel::new_with_weights(
        formula,
        &data,
        Family::Binomial,
        None,
        weights,
    )
    .unwrap();
    model.fit_with_options(true, n_agq, false).unwrap();
    model
}

// toy: 4 groups × Poisson with explicit `exposure` offset vector;
// covers the offset-contract path of the native GLMM compiler.
fn poisson_offset_data() -> (DataFrame, Vec<f64>) {
    let group_effects = [-0.25, 0.1, 0.35, -0.05];
    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut exposure = Vec::new();
    let mut group = Vec::new();

    for (g, group_effect) in group_effects.iter().enumerate() {
        for obs in 0..6 {
            let xv = obs as f64 - 2.5;
            let expv = 0.8 + 0.2 * obs as f64 + 0.15 * g as f64;
            let eta = -0.15 + 0.22 * xv + group_effect;
            let mean = expv * eta.exp();
            let count = (mean + ((g + obs) % 3) as f64 * 0.35).round().max(0.0);
            y.push(count);
            x.push(xv);
            exposure.push(expv);
            group.push(format!("g{}", g + 1));
        }
    }

    let offset = exposure.iter().map(|value| value.ln()).collect::<Vec<_>>();
    let mut data = DataFrame::new();
    data.add_numeric("y", y).unwrap();
    data.add_numeric("x", x).unwrap();
    data.add_numeric("exposure", exposure).unwrap();
    data.add_categorical("group", group).unwrap();
    (data, offset)
}

// toy: 5 groups Gamma/Log GLMM; covers the dispersion-contract path of
// the native GLMM compiler.
fn gamma_log_data() -> DataFrame {
    let group_effects = [-0.18, 0.0, 0.22, 0.08, -0.12];
    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut group = Vec::new();

    for (g, group_effect) in group_effects.iter().enumerate() {
        for obs in 0..5 {
            let xv = obs as f64 - 2.0;
            let eta = 0.65 + 0.16 * xv + group_effect;
            let wiggle = 0.9 + 0.04 * ((g + 2 * obs) % 5) as f64;
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

#[test]
fn default_native_binomial_logit_covers_weights_and_agq() {
    let mut weighted = weighted_cbpp_model(5);
    assert_native_glmm_fit(&weighted, 5, 4, 1);
    assert_eq!(weighted.nobs(), 56);
    assert_eq!(weighted.dof(), 5);
    assert!(
        (weighted.deviance(1) - weighted.deviance(5)).abs() > 0.05,
        "weighted Binomial AGQ deviance should remain distinct from Laplace"
    );

    let (data, _) = datasets::load("cbpp").unwrap();
    let incidence = data.numeric("incidence").unwrap();
    let size = data.numeric("size").unwrap();
    let proportion: Vec<f64> = incidence
        .iter()
        .zip(size.iter())
        .map(|(&y, &n)| y / n)
        .collect();
    let mut data = data.clone();
    data.add_numeric("proportion", proportion).unwrap();
    let formula = parse_formula("proportion ~ 1 + period + (1 | herd)").unwrap();
    let mut unit_weight =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Binomial, None).unwrap();
    unit_weight.fit_with_options(true, 5, false).unwrap();

    assert!(
        (weighted.deviance(5) - unit_weight.deviance(5)).abs() > 1.0,
        "Binomial case weights should materially affect the AGQ deviance"
    );
}

#[test]
fn default_native_joint_laplace_is_reachable_without_nlopt() {
    let (data, _) = datasets::load("cbpp").unwrap();
    let incidence = data.numeric("incidence").unwrap();
    let size = data.numeric("size").unwrap();
    let proportion: Vec<f64> = incidence
        .iter()
        .zip(size.iter())
        .map(|(&y, &n)| y / n)
        .collect();
    let weights = size.to_vec();
    let mut data = data.clone();
    data.add_numeric("proportion", proportion).unwrap();

    let formula = parse_formula("proportion ~ 1 + period + (1 | herd)").unwrap();
    let mut model = GeneralizedLinearMixedModel::new_with_weights(
        formula,
        &data,
        Family::Binomial,
        None,
        weights,
    )
    .unwrap();

    model.fit_with_options(false, 1, false).unwrap();

    assert!(
        model.lmm().optsum().return_value.contains("JOINT_LAPLACE"),
        "native fast=false should now be a labelled joint Laplace path or fallback, got {}",
        model.lmm().optsum().return_value
    );
    assert_eq!(model.lmm().optsum().backend.label(), "native");
    assert!(model.objective().is_finite());
    assert!(model.loglikelihood().is_finite());
    let metadata = model
        .compiler_artifact()
        .glmm_fit_metadata
        .as_ref()
        .expect("native joint GLMM fit should record metadata");
    assert!(
        matches!(
            metadata.estimation_method.as_str(),
            "joint_laplace" | "fallback_fast_pirls"
        ),
        "unexpected native joint fit metadata: {metadata:?}"
    );
}

#[test]
fn default_native_poisson_log_covers_offset_contract() {
    let (data, offset) = poisson_offset_data();
    let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();
    let mut offset_model = GeneralizedLinearMixedModel::new_with_offset(
        formula.clone(),
        &data,
        Family::Poisson,
        None,
        offset.clone(),
    )
    .unwrap();
    offset_model.fit_with_options(true, 1, false).unwrap();

    assert_native_glmm_fit(&offset_model, 1, 2, 1);
    assert_eq!(offset_model.offset.len(), offset.len());
    for (actual, want) in offset_model.offset.iter().zip(offset.iter()) {
        assert!((*actual - *want).abs() < 1e-12);
    }
    assert!(offset_model.fitted().iter().all(|mu| *mu > 0.0));

    let mut no_offset =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Poisson, None).unwrap();
    no_offset.fit_with_options(true, 1, false).unwrap();
    assert!(
        (offset_model.deviance(1) - no_offset.deviance(1)).abs() > 0.01,
        "Poisson offset should change the fitted Laplace deviance"
    );
}

#[test]
fn default_native_gamma_log_covers_dispersion_contract() {
    let data = gamma_log_data();
    let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();
    let mut model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Gamma, Some(LinkFunction::Log))
            .unwrap();
    model.fit_with_options(true, 1, false).unwrap();

    assert_native_glmm_fit(&model, 1, 2, 1);
    assert_eq!(model.nobs(), 25);
    assert!(model.dispersion(false) > 0.0);
    assert!(model.dispersion(true) > 0.0);
    assert!(model.fitted().iter().all(|mu| *mu > 0.0));
}
