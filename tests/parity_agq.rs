#![cfg(feature = "unstable-internals")]

use approx::assert_relative_eq;
use serde::Deserialize;

use mixeff_rs::datasets;
use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::generalized::GeneralizedLinearMixedModel;
use mixeff_rs::model::traits::{Family, MixedModelFit};
use mixeff_rs::types::gh_norm;
#[cfg(not(feature = "nlopt"))]
use mixeff_rs::types::Optimizer;

#[allow(dead_code)]
#[derive(Deserialize)]
struct AgqFixture {
    schema_version: String,
    formula: String,
    n_agq: usize,
    nobs: usize,
    dof: usize,
    theta: Vec<f64>,
    beta: Vec<f64>,
    objective: f64,
    deviance_laplace: f64,
    deviance_agq: f64,
}

fn fixture() -> AgqFixture {
    serde_json::from_str(include_str!("fixtures/parity/cbpp_agq5.json")).unwrap()
}

fn cbpp_model(n_agq: usize) -> GeneralizedLinearMixedModel {
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

#[test]
fn test_gh_norm_constants_match_julia() {
    let expected: &[(&[f64], &[f64])] = &[
        (&[0.0], &[1.0]),
        (&[-1.0, 1.0], &[0.5, 0.5]),
        (
            &[-1.7320508075688772, 0.0, 1.7320508075688772],
            &[0.16666666666666666, 0.6666666666666666, 0.16666666666666666],
        ),
        (
            &[
                -2.3344142183389747,
                -0.7419637843027263,
                0.7419637843027263,
                2.3344142183389747,
            ],
            &[
                0.045875854768068484,
                0.45412414523193156,
                0.45412414523193156,
                0.045875854768068484,
            ],
        ),
        (
            &[
                -2.856_970_013_872_804,
                -1.3556261799742646,
                0.0,
                1.3556261799742646,
                2.856_970_013_872_804,
            ],
            &[
                0.011257411327720636,
                0.22207592200561266,
                0.5333333333333335,
                0.22207592200561266,
                0.011257411327720636,
            ],
        ),
        (
            &[
                -3.324_257_433_552_12,
                -1.8891758777537095,
                -0.6167065901925946,
                0.6167065901925946,
                1.8891758777537095,
                3.324_257_433_552_12,
            ],
            &[
                0.0025557844020562123,
                0.08861574604191458,
                0.40882846955602925,
                0.40882846955602925,
                0.08861574604191458,
                0.0025557844020562123,
            ],
        ),
        (
            &[
                -3.7504397177257385,
                -2.366759410734537,
                -1.1544053947399682,
                0.0,
                1.1544053947399682,
                2.366759410734537,
                3.7504397177257385,
            ],
            &[
                0.0005482688559722207,
                0.030757123967586463,
                0.24012317860501267,
                0.4571428571428572,
                0.24012317860501267,
                0.030757123967586463,
                0.0005482688559722207,
            ],
        ),
        (
            &[
                -4.14454718612589,
                -2.802485861287542,
                -1.6365190424351077,
                -0.5390798113513762,
                0.5390798113513762,
                1.6365190424351077,
                2.802485861287542,
                4.14454718612589,
            ],
            &[
                0.00011261453837536793,
                0.0096352201207882,
                0.1172399076617589,
                0.37301225767907753,
                0.37301225767907753,
                0.1172399076617589,
                0.0096352201207882,
                0.00011261453837536793,
            ],
        ),
        (
            &[
                -4.512745863399781,
                -3.2054290028564667,
                -2.07684797867783,
                -1.0232556637891335,
                0.0,
                1.0232556637891335,
                2.07684797867783,
                3.2054290028564667,
                4.512745863399781,
            ],
            &[
                0.000022345844007746417,
                0.0027891413212317744,
                0.04991640676521791,
                0.24409750289493887,
                0.40634920634920746,
                0.24409750289493887,
                0.04991640676521791,
                0.0027891413212317744,
                0.000022345844007746417,
            ],
        ),
    ];

    for (k0, (z, w)) in expected.iter().enumerate() {
        let rule = gh_norm(k0 + 1);
        assert_eq!(rule.z.len(), z.len());
        for (actual, want) in rule.z.iter().zip(z.iter()) {
            assert_relative_eq!(*actual, *want, epsilon = 1e-13);
        }
        for (actual, want) in rule.w.iter().zip(w.iter()) {
            assert_relative_eq!(*actual, *want, epsilon = 1e-13);
        }
    }
}

#[cfg(feature = "nlopt")]
#[test]
fn test_cbpp_agq5_deviance_matches_julia() {
    let expected = fixture();
    assert_eq!(expected.schema_version, "1.0.0");
    let mut model = cbpp_model(expected.n_agq);

    assert_eq!(model.nobs(), expected.nobs);
    assert_eq!(model.dof(), expected.dof);
    assert_eq!(expected.formula, "proportion ~ 1 + period + (1 | herd)");
    let model_theta = model.theta();
    assert_eq!(model_theta.len(), expected.theta.len());
    assert_eq!(model.beta.len(), expected.beta.len());
    for (actual, want) in model_theta.iter().zip(expected.theta.iter()) {
        assert_relative_eq!(*actual, *want, epsilon = 1e-8);
    }
    for (actual, want) in model.beta.iter().zip(expected.beta.iter()) {
        assert_relative_eq!(*actual, *want, epsilon = 1e-8);
    }

    assert_relative_eq!(model.objective(), expected.objective, epsilon = 1e-6);
    assert_relative_eq!(model.deviance(1), expected.deviance_laplace, epsilon = 1e-6);
    assert_relative_eq!(
        model.deviance(expected.n_agq),
        expected.deviance_agq,
        epsilon = 1e-6
    );
}

#[cfg(not(feature = "nlopt"))]
#[test]
fn test_cbpp_agq5_native_cobyla_fit_records_agq_contract() {
    let expected = fixture();
    let mut model = cbpp_model(expected.n_agq);

    assert_eq!(model.nobs(), expected.nobs);
    assert_eq!(model.dof(), expected.dof);
    assert_eq!(expected.formula, "proportion ~ 1 + period + (1 | herd)");
    assert_eq!(model.lmm().optsum.optimizer, Optimizer::Cobyla);
    assert_eq!(model.lmm().optsum.backend.label(), "native");
    assert_eq!(model.lmm().optsum.n_agq, expected.n_agq);
    assert!(model.objective().is_finite());
    assert!(model.deviance(1).is_finite());
    assert!(model.deviance(expected.n_agq).is_finite());
    assert!(
        (model.deviance(1) - model.deviance(expected.n_agq)).abs() > 0.05,
        "native AGQ path should remain distinct from the Laplace approximation"
    );
}

#[test]
fn test_agq_scales_correctly_with_strong_intercept_variance() {
    let mut model = cbpp_model(5);
    let laplace = model.deviance(1);
    let agq = model.deviance(5);
    assert!(model.theta()[0] > 0.5);
    assert!(
        (laplace - agq).abs() > 0.05,
        "AGQ should not collapse to the Laplace deviance when the intercept variance is material"
    );
}
