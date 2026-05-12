use serde::Deserialize;

use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::data::DataFrame;
use mixeff_rs::model::linear::LinearMixedModel;
use mixeff_rs::model::traits::MixedModelFit;

#[derive(Deserialize)]
struct ParmapFixture {
    schema_version: String,
    source: String,
    formula: String,
    nobs: usize,
    grouping: String,
    cnames: Vec<String>,
    linear_indices_column_major: Vec<usize>,
    parmap_zero_based: Vec<ParmapEntry>,
}

#[derive(Deserialize)]
struct ParmapEntry {
    term: usize,
    row: usize,
    col: usize,
}

fn fixture() -> ParmapFixture {
    serde_json::from_str(include_str!("fixtures/parity/parmap_vsize3.json")).unwrap()
}

// toy: 4 subjects × 5 obs of vsize=3 random-effects data; paired with
// `fixtures/parity/parmap_vsize3.json` for the lower-triangular index test.
fn parmap_vsize3_data() -> DataFrame {
    let subj_effects = [-0.8, 0.35, 0.6, -0.15];
    let mut y = Vec::with_capacity(20);
    let mut x = Vec::with_capacity(20);
    let mut z = Vec::with_capacity(20);
    let mut subj = Vec::with_capacity(20);

    for subject in 0..4 {
        for obs in 0..5 {
            let xv = obs as f64 - 2.0;
            let zv = (obs % 3) as f64 - 1.0 + subject as f64 * 0.1;
            y.push(3.0 + 0.5 * xv - 0.2 * zv + subj_effects[subject]);
            x.push(xv);
            z.push(zv);
            subj.push(format!("S{}", subject + 1));
        }
    }

    let mut data = DataFrame::new();
    data.add_numeric("y", y).unwrap();
    data.add_numeric("x", x).unwrap();
    data.add_numeric("z", z).unwrap();
    data.add_categorical("subj", subj).unwrap();
    data
}

#[test]
fn test_parmap_vsize3_lower_triangular_order() {
    let expected = fixture();
    assert_eq!(expected.schema_version, "1.0.0");
    assert!(expected.source.contains("MixedModels.jl"));

    let data = parmap_vsize3_data();
    let formula = parse_formula(&expected.formula).unwrap();
    let model = LinearMixedModel::new(formula, &data, None).unwrap();

    assert_eq!(model.nobs(), expected.nobs);
    assert_eq!(model.reterms.len(), 1);
    assert_eq!(model.reterms[0].grouping_name, expected.grouping);
    assert_eq!(model.reterms[0].cnames, expected.cnames);
    assert_eq!(model.reterms[0].vsize, 3);
    assert_eq!(model.reterms[0].inds, expected.linear_indices_column_major);

    let expected_parmap = expected
        .parmap_zero_based
        .iter()
        .map(|entry| (entry.term, entry.row, entry.col))
        .collect::<Vec<_>>();
    assert_eq!(model.parmap, expected_parmap);
}
