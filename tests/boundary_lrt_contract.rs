use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::{DataFrame, LinearMixedModel, MixedModelFit};
use mixeff_rs::stats::{
    BoundaryLikelihoodRatioTest, BoundaryLrtStatus, LinearModelFit, ModelComparisonClass,
    BOUNDARY_LRT_SCHEMA, BOUNDARY_LRT_SCHEMA_VERSION,
};
use nalgebra::{DMatrix, DVector};

fn random_intercept_data() -> DataFrame {
    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut group = Vec::new();
    for g in 0..8 {
        let shift = (g as f64 - 3.5) * 1.25;
        for i in 0..5 {
            let x_value = i as f64 - 2.0;
            let noise = ((g + i) % 3) as f64 * 0.15;
            y.push(10.0 + 0.7 * x_value + shift + noise);
            x.push(x_value);
            group.push(format!("G{g}"));
        }
    }
    let mut data = DataFrame::new();
    data.add_numeric("y", y).unwrap();
    data.add_numeric("x", x).unwrap();
    data.add_categorical("group", group).unwrap();
    data
}

fn intercept_only_lm(data: &DataFrame) -> LinearModelFit {
    let y = data.numeric("y").unwrap();
    let response = DVector::from_column_slice(y);
    let model_matrix = DMatrix::from_element(y.len(), 1, 1.0);
    LinearModelFit::fit(response, model_matrix, Some("y ~ 1".to_string())).unwrap()
}

#[test]
fn boundary_lrt_certifies_one_added_random_intercept_parameter() {
    let data = random_intercept_data();
    let smaller = intercept_only_lm(&data);

    let formula = parse_formula("y ~ 1 + (1 | group)").unwrap();
    let mut larger = LinearMixedModel::new(formula, &data, None).unwrap();
    larger.fit(false).unwrap();

    let lrt = BoundaryLikelihoodRatioTest::variance_component(&smaller, &larger);
    assert_eq!(lrt.schema_name, BOUNDARY_LRT_SCHEMA);
    assert_eq!(lrt.schema_version, BOUNDARY_LRT_SCHEMA_VERSION);
    assert_eq!(lrt.status, BoundaryLrtStatus::Available);
    assert!(matches!(
        lrt.comparison_class,
        Some(
            ModelComparisonClass::NestedRandomEffects
                | ModelComparisonClass::SameFixedEffectsCovarianceDifference
        )
    ));
    assert_eq!(lrt.ordinary_chisq_dof, Some(1));
    assert_eq!(lrt.mixture.len(), 2);
    assert_eq!(lrt.mixture[0].point_mass_at, Some(0.0));
    assert_eq!(lrt.mixture[1].chisq_df, Some(1));
    assert!(lrt.statistic.unwrap().is_finite());
    assert!((0.0..=1.0).contains(&lrt.pvalue.unwrap()));
}

#[test]
fn boundary_lrt_refuses_fixed_effect_comparisons() {
    let data = random_intercept_data();

    let f0 = parse_formula("y ~ 1 + (1 | group)").unwrap();
    let mut m0 = LinearMixedModel::new(f0, &data, None).unwrap();
    m0.fit(false).unwrap();

    let f1 = parse_formula("y ~ 1 + x + (1 | group)").unwrap();
    let mut m1 = LinearMixedModel::new(f1, &data, None).unwrap();
    m1.fit(false).unwrap();

    let lrt = BoundaryLikelihoodRatioTest::variance_component(
        &m0 as &dyn MixedModelFit,
        &m1 as &dyn MixedModelFit,
    );
    assert_eq!(lrt.status, BoundaryLrtStatus::Unsupported);
    assert_eq!(
        lrt.reason_code.as_deref(),
        Some("boundary_lrt_requires_variance_component_comparison")
    );
    assert!(lrt.pvalue.is_none());
}
