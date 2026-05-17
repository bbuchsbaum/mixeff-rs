#![cfg(feature = "unstable-internals")]

use approx::assert_relative_eq;

use mixeff_rs::datasets;
use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::{DataFrame, LinearMixedModel, MixedModelFit};

fn fit_lmm(data: &DataFrame, formula: &str, reml: bool) -> LinearMixedModel {
    let formula = parse_formula(formula).unwrap();
    let mut model = LinearMixedModel::new(formula, data, None).unwrap();
    model.fit(reml).unwrap();
    model
}

fn assert_vector_close(label: &str, left: &[f64], right: &[f64], epsilon: f64) {
    assert_eq!(left.len(), right.len(), "{label}: length mismatch");
    for (idx, (l, r)) in left.iter().zip(right.iter()).enumerate() {
        assert_relative_eq!(*l, *r, epsilon = epsilon, max_relative = epsilon);
        assert!(
            (l - r).abs() <= epsilon.max(epsilon * r.abs()),
            "{label}[{idx}]: {l} != {r}"
        );
    }
}

#[test]
fn row_order_invariance_preserves_sleepstudy_fit_and_levels() {
    let (data, _) = datasets::load("sleepstudy").unwrap();
    let rows = (0..data.nrow()).rev().collect::<Vec<_>>();
    let reversed = data.select_rows(&rows).unwrap();

    let original_subject = data.categorical("Subject").unwrap();
    let reversed_subject = reversed.categorical("Subject").unwrap();
    assert_eq!(
        original_subject.levels, reversed_subject.levels,
        "row permutation must preserve canonical factor-level order"
    );

    let formula = "Reaction ~ 1 + Days + (1 + Days | Subject)";
    let original = fit_lmm(&data, formula, true);
    let permuted = fit_lmm(&reversed, formula, true);

    assert_relative_eq!(
        original.objective(),
        permuted.objective(),
        epsilon = 1e-6,
        max_relative = 1e-8
    );
    assert_vector_close(
        "beta",
        original.coef().as_slice(),
        permuted.coef().as_slice(),
        1e-7,
    );
    assert_vector_close("theta", &original.theta(), &permuted.theta(), 1e-6);
    assert_relative_eq!(original.sigma(), permuted.sigma(), epsilon = 1e-7);
}

#[test]
fn nested_formula_expansion_matches_explicit_nested_terms() {
    let (data, _) = datasets::load("oxide").unwrap();
    let shorthand = fit_lmm(&data, "Thickness ~ 1 + (1 | Lot/Wafer)", true);
    let explicit = fit_lmm(&data, "Thickness ~ 1 + (1 | Lot) + (1 | Lot:Wafer)", true);

    assert_relative_eq!(
        shorthand.objective(),
        explicit.objective(),
        epsilon = 1e-6,
        max_relative = 1e-8
    );
    assert_vector_close(
        "beta",
        shorthand.coef().as_slice(),
        explicit.coef().as_slice(),
        1e-8,
    );
    assert_vector_close("theta", &shorthand.theta(), &explicit.theta(), 1e-6);
    assert_relative_eq!(shorthand.sigma(), explicit.sigma(), epsilon = 1e-7);
}

#[test]
fn double_bar_matches_explicit_independent_random_terms() {
    let (data, _) = datasets::load("sleepstudy").unwrap();
    let double_bar = fit_lmm(&data, "Reaction ~ 1 + Days + (1 + Days || Subject)", false);
    let explicit = fit_lmm(
        &data,
        "Reaction ~ 1 + Days + (1 | Subject) + (0 + Days | Subject)",
        false,
    );

    assert_relative_eq!(
        double_bar.objective(),
        explicit.objective(),
        epsilon = 1e-2,
        max_relative = 1e-6
    );
    assert_vector_close(
        "beta",
        double_bar.coef().as_slice(),
        explicit.coef().as_slice(),
        1e-6,
    );
    assert_vector_close("theta", &double_bar.theta(), &explicit.theta(), 1e-4);
}

#[test]
fn predictor_rescaling_preserves_sleepstudy_fitted_values() {
    let (mut scaled, _) = datasets::load("sleepstudy").unwrap();
    let days_scaled = scaled
        .numeric("Days")
        .unwrap()
        .iter()
        .map(|value| value * 10.0)
        .collect::<Vec<_>>();
    scaled.add_numeric("Days10", days_scaled).unwrap();
    let (data, _) = datasets::load("sleepstudy").unwrap();

    let ordinary = fit_lmm(&data, "Reaction ~ 1 + Days + (1 + Days | Subject)", false);
    let rescaled = fit_lmm(
        &scaled,
        "Reaction ~ 1 + Days10 + (1 + Days10 | Subject)",
        false,
    );

    assert_relative_eq!(
        ordinary.objective(),
        rescaled.objective(),
        epsilon = 1e-4,
        max_relative = 1e-7
    );
    assert_vector_close(
        "fitted",
        ordinary.fitted().as_slice(),
        rescaled.fitted().as_slice(),
        1e-4,
    );
    assert_relative_eq!(ordinary.coef()[0], rescaled.coef()[0], epsilon = 1e-6);
    assert_relative_eq!(
        ordinary.coef()[1],
        rescaled.coef()[1] * 10.0,
        epsilon = 1e-5,
        max_relative = 1e-7
    );
}

#[test]
fn boundary_certificate_is_stable_under_row_permutation() {
    let (data, _) = datasets::load("dyestuff2").unwrap();
    let rows = (0..data.nrow()).rev().collect::<Vec<_>>();
    let reversed = data.select_rows(&rows).unwrap();

    let original = fit_lmm(&data, "Yield ~ 1 + (1 | Batch)", true);
    let permuted = fit_lmm(&reversed, "Yield ~ 1 + (1 | Batch)", true);

    assert!(original.is_singular());
    assert!(permuted.is_singular());
    assert_relative_eq!(
        original.objective(),
        permuted.objective(),
        epsilon = 1e-6,
        max_relative = 1e-8
    );
}
