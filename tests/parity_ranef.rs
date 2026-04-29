#![cfg(feature = "nlopt")]

use approx::assert_relative_eq;
use nalgebra::DMatrix;
use serde::Deserialize;

use mixedmodels::formula::parse_formula;
use mixedmodels::model::data::DataFrame;
use mixedmodels::model::linear::LinearMixedModel;
use mixedmodels::model::traits::MixedModelFit;

#[derive(Deserialize)]
struct RanefFixture {
    schema_version: String,
    formula: String,
    nobs: usize,
    theta: Vec<f64>,
    beta: Vec<f64>,
    ranef_u: Vec<Vec<Vec<f64>>>,
    ranef_b: Vec<Vec<Vec<f64>>>,
}

fn fixture() -> RanefFixture {
    serde_json::from_str(include_str!("fixtures/parity/kb07_ranef.json")).unwrap()
}

fn kb07_style_data() -> DataFrame {
    let subj_effects = [-1.0, 0.5, 1.2, -0.4, -0.3];
    let subj_slopes = [-0.3, 0.2, 0.1, -0.2, 0.4];
    let item_effects = [-0.2, 0.4, -0.1, 0.3];
    let mut y = Vec::with_capacity(20);
    let mut x = Vec::with_capacity(20);
    let mut subj = Vec::with_capacity(20);
    let mut item = Vec::with_capacity(20);

    for s in 0..5 {
        for i in 0..4 {
            let xi = i as f64;
            let row = s * 4 + i + 1;
            y.push(
                20.0 + 2.0 * xi
                    + subj_effects[s]
                    + item_effects[i]
                    + subj_slopes[s] * xi
                    + ((row % 7) as f64 - 3.0) * 0.03,
            );
            x.push(xi);
            subj.push(format!("S{}", s + 1));
            item.push(format!("I{}", i + 1));
        }
    }

    let mut data = DataFrame::new();
    data.add_numeric("y", y).unwrap();
    data.add_numeric("x", x).unwrap();
    data.add_categorical("subj", subj).unwrap();
    data.add_categorical("item", item).unwrap();
    data
}

fn assert_matrix_close(actual: &DMatrix<f64>, expected: &[Vec<f64>], eps: f64) {
    assert_eq!(actual.nrows(), expected.len());
    assert_eq!(actual.ncols(), expected[0].len());
    for row in 0..actual.nrows() {
        for col in 0..actual.ncols() {
            assert_relative_eq!(actual[(row, col)], expected[row][col], epsilon = eps);
        }
    }
}

fn fitted_model() -> (LinearMixedModel, RanefFixture) {
    let expected = fixture();
    assert_eq!(expected.schema_version, "1.0.0");
    let formula = parse_formula(&expected.formula).unwrap();
    let data = kb07_style_data();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(true).unwrap();
    (model, expected)
}

#[test]
fn test_kb07_ranef_u_matches_julia() {
    let (model, expected) = fitted_model();
    assert_eq!(model.nobs(), expected.nobs);
    let beta = model.fixef();
    for (actual, want) in beta.iter().zip(expected.beta.iter()) {
        assert_relative_eq!(*actual, *want, epsilon = 1e-8);
    }
    for (actual, want) in model.theta().iter().zip(expected.theta.iter()) {
        assert_relative_eq!(*actual, *want, epsilon = 5e-4);
    }

    let ranef = model.ranef_u();
    assert_eq!(ranef.len(), expected.ranef_u.len());
    for (actual, want) in ranef.iter().zip(expected.ranef_u.iter()) {
        assert_matrix_close(actual, want, 2e-6);
    }
}

#[test]
fn test_kb07_ranef_b_matches_julia() {
    let (model, expected) = fitted_model();
    let ranef = model.ranef_b();
    assert_eq!(ranef.len(), expected.ranef_b.len());
    for (actual, want) in ranef.iter().zip(expected.ranef_b.iter()) {
        assert_matrix_close(actual, want, 1e-6);
    }
}

#[test]
fn test_three_re_term_ranef() {
    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut g1 = Vec::new();
    let mut g2 = Vec::new();
    let mut g3 = Vec::new();
    for i in 0..36 {
        let xi = (i % 4) as f64;
        let a = i % 3;
        let b = (i / 3) % 4;
        let c = (i / 12) % 3;
        y.push(
            5.0 + 0.7 * xi
                + [-0.4, 0.2, 0.5][a]
                + [0.3, -0.1, 0.4, -0.2][b]
                + [-0.25, 0.35, -0.1][c],
        );
        x.push(xi);
        g1.push(format!("g1_{}", a + 1));
        g2.push(format!("g2_{}", b + 1));
        g3.push(format!("g3_{}", c + 1));
    }

    let mut data = DataFrame::new();
    data.add_numeric("y", y).unwrap();
    data.add_numeric("x", x).unwrap();
    data.add_categorical("g1", g1).unwrap();
    data.add_categorical("g2", g2).unwrap();
    data.add_categorical("g3", g3).unwrap();

    let formula = parse_formula("y ~ 1 + x + (1 | g1) + (1 | g2) + (1 | g3)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(true).unwrap();

    let u = model.ranef_u();
    let b = model.ranef_b();
    assert_eq!(u.len(), 3);
    assert_eq!(b.len(), 3);
    for (u_term, b_term) in u.iter().zip(b.iter()) {
        assert_eq!(u_term.nrows(), 1);
        assert_eq!(b_term.nrows(), 1);
        assert!(u_term.iter().all(|value| value.is_finite()));
        assert!(b_term.iter().all(|value| value.is_finite()));
    }
}
