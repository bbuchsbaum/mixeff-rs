use approx::assert_relative_eq;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use serde::Deserialize;

use mixeff_rs::error::MixedModelError;
use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::data::DataFrame;
use mixeff_rs::model::linear::LinearMixedModel;
use mixeff_rs::model::traits::MixedModelFit;

#[derive(Deserialize)]
struct RankFixture {
    schema_version: String,
    formula: String,
    nobs: usize,
    fixed_effect_rank: usize,
    dof: usize,
    ml: MetricBlock,
    reml: RemlBlock,
}

#[derive(Deserialize)]
struct MetricBlock {
    objective: f64,
    aic: f64,
    bic: f64,
    sigma: f64,
}

#[derive(Deserialize)]
struct RemlBlock {
    objective: f64,
    sigma: f64,
    varest: f64,
}

fn fixture() -> RankFixture {
    serde_json::from_str(include_str!("fixtures/parity/rank_deficient_metrics.json")).unwrap()
}

// toy: 24 rows with `x2 = 2 * x` (deliberately collinear); paired with
// `fixtures/parity/rank_deficient_metrics.json` for the rank/dof/AIC test.
fn rank_deficient_data() -> DataFrame {
    let n = 24;
    let x: Vec<f64> = (0..n).map(|i| (i % 4) as f64).collect();
    let x2: Vec<f64> = x.iter().map(|value| 2.0 * value).collect();
    let group_effects = [-1.2, 0.8, 0.3, -0.4, 1.1, -0.6];
    let mut y = Vec::with_capacity(n);
    let mut g = Vec::with_capacity(n);
    for i in 0..n {
        let group = i / 4;
        y.push(10.0 + 1.5 * x[i] + group_effects[group] + ((i + 1) % 5) as f64 * 0.07 - 0.14);
        g.push(format!("g{}", group + 1));
    }

    let mut data = DataFrame::new();
    data.add_numeric("y", y).unwrap();
    data.add_numeric("x", x).unwrap();
    data.add_numeric("x2", x2).unwrap();
    data.add_categorical("g", g).unwrap();
    data
}

// toy: parameterized GH-809 reproducer; seeded with `StdRng(809)` so
// different (n, p) shapes yield deterministic data the test sweeps over.
fn issue_809_wide_fixed_effect_data(n: usize, p: usize) -> DataFrame {
    let mut rng = rand::rngs::StdRng::seed_from_u64(809);
    let mut y = Vec::with_capacity(n);
    let mut group = Vec::with_capacity(n);
    let group_means: Vec<f64> = (1..=15).map(|value| value as f64).collect();
    let mut x_cols: Vec<Vec<f64>> = (0..p).map(|_| Vec::with_capacity(n)).collect();

    for row in 0..n {
        let group_index = row % group_means.len();
        let group_mean = group_means[group_index];
        let mut eta = -group_mean;
        for col in 0..p {
            let draw: f64 = StandardNormal.sample(&mut rng);
            let value = group_mean + draw;
            x_cols[col].push(value);
            eta += if col % 2 == 0 {
                0.3 * value
            } else {
                -0.3 * value
            };
        }
        y.push(eta + ((row * 7 + 3) % 11) as f64 * 0.03);
        group.push(format!("g{}", group_index + 1));
    }

    let mut data = DataFrame::new();
    data.add_numeric("y", y).unwrap();
    for (col, values) in x_cols.into_iter().enumerate() {
        data.add_numeric(&format!("x{}", col + 1), values).unwrap();
    }
    data.add_categorical("group", group).unwrap();
    data
}

fn issue_809_formula(p: usize) -> String {
    format!(
        "y ~ {} + (1 | group)",
        (1..=p)
            .map(|idx| format!("x{idx}"))
            .collect::<Vec<_>>()
            .join(" + ")
    )
}

#[test]
fn test_rank_deficient_aic_bic_matches_julia() {
    let expected = fixture();
    assert_eq!(expected.schema_version, "1.0.0");
    let data = rank_deficient_data();
    let formula = parse_formula(&expected.formula).unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    assert_eq!(model.nobs(), expected.nobs);
    assert_eq!(model.feterm.rank, expected.fixed_effect_rank);
    assert_eq!(model.dof(), expected.dof);
    assert_relative_eq!(model.objective(), expected.ml.objective, epsilon = 1e-8);
    assert_relative_eq!(model.aic(), expected.ml.aic, epsilon = 1e-8);
    assert_relative_eq!(model.bic(), expected.ml.bic, epsilon = 1e-8);
    assert_relative_eq!(model.sigma(), expected.ml.sigma, epsilon = 2e-7);
}

#[test]
fn test_rank_deficient_sigma2_reml_matches_julia() {
    let expected = fixture();
    let data = rank_deficient_data();
    let formula = parse_formula(&expected.formula).unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(true).unwrap();

    assert_eq!(model.feterm.rank, expected.fixed_effect_rank);
    assert_relative_eq!(model.objective(), expected.reml.objective, epsilon = 1e-8);
    assert_relative_eq!(model.sigma(), expected.reml.sigma, epsilon = 2e-7);
    assert_relative_eq!(model.varest(), expected.reml.varest, epsilon = 4e-8);
}

#[test]
fn test_rank_deficient_dof_matches_julia() {
    let expected = fixture();
    let data = rank_deficient_data();
    let formula = parse_formula(&expected.formula).unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    assert_eq!(model.feterm.rank, expected.fixed_effect_rank);
    assert_eq!(model.dof(), expected.dof);
}

#[test]
fn test_issue_809_wide_fixed_effects_are_rank_saturated_not_fit_success() {
    let n = 24;
    let p = 24;
    let data = issue_809_wide_fixed_effect_data(n, p);
    let formula = parse_formula(&issue_809_formula(p)).unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

    assert_eq!(model.nobs(), n);
    assert_eq!(model.feterm.rank, n);
    assert_eq!(model.dof_residual(), 0);

    let err = model
        .fit(false)
        .expect_err("wide fixed-effect design should not be treated as an ordinary fit");
    assert!(matches!(
        err,
        MixedModelError::RankSaturatedFixedEffects { rank: 24, nobs: 24 }
    ));
    assert!(err
        .to_string()
        .contains("rank(X) = 24 and n = 24, leaving zero residual degrees of freedom"));
}
