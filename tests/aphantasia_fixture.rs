use std::fs;
use std::path::PathBuf;

use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::{DataFrame, Family, GeneralizedLinearMixedModel, MixedModelFit};
use serde_json::Value;

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/aphantasia")
}

fn fixture_path(relative: &str) -> PathBuf {
    fixture_root().join(relative)
}

fn reference_json() -> Value {
    let path = fixture_path("reference.json");
    serde_json::from_str(
        &fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {path:?}: {error}")),
    )
    .unwrap_or_else(|error| panic!("parse {path:?}: {error}"))
}

fn reference_model<'a>(reference: &'a Value, case_id: &str) -> &'a Value {
    reference
        .get("models")
        .and_then(Value::as_object)
        .and_then(|models| models.get(case_id))
        .unwrap_or_else(|| panic!("reference model `{case_id}` missing"))
}

fn csv_row_count(relative: &str) -> usize {
    let path = fixture_path(relative);
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(&path)
        .unwrap_or_else(|error| panic!("open {path:?}: {error}"));
    rdr.records()
        .map(|record| record.unwrap_or_else(|error| panic!("read {path:?}: {error}")))
        .count()
}

fn load_prepared_case(
    case_id: &str,
    numeric_columns: &[&str],
    categorical_columns: &[&str],
) -> DataFrame {
    let path = fixture_path(&format!("prepared/{case_id}.csv"));
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(&path)
        .unwrap_or_else(|error| panic!("open {path:?}: {error}"));
    let headers = rdr
        .headers()
        .unwrap_or_else(|error| panic!("headers {path:?}: {error}"))
        .clone();

    let numeric_idx = numeric_columns
        .iter()
        .map(|column| {
            headers
                .iter()
                .position(|header| header == *column)
                .unwrap_or_else(|| panic!("numeric column `{column}` missing in {path:?}"))
        })
        .collect::<Vec<_>>();
    let categorical_idx = categorical_columns
        .iter()
        .map(|column| {
            headers
                .iter()
                .position(|header| header == *column)
                .unwrap_or_else(|| panic!("categorical column `{column}` missing in {path:?}"))
        })
        .collect::<Vec<_>>();

    let mut numeric_data = vec![Vec::new(); numeric_columns.len()];
    let mut categorical_data = vec![Vec::new(); categorical_columns.len()];
    for record in rdr.records() {
        let record = record.unwrap_or_else(|error| panic!("read {path:?}: {error}"));
        for (slot, &idx) in numeric_idx.iter().enumerate() {
            let raw = record.get(idx).unwrap_or("");
            let value = raw.parse::<f64>().unwrap_or_else(|error| {
                panic!(
                    "parse numeric `{}` value `{raw}` in {path:?}: {error}",
                    numeric_columns[slot]
                )
            });
            numeric_data[slot].push(value);
        }
        for (slot, &idx) in categorical_idx.iter().enumerate() {
            categorical_data[slot].push(record.get(idx).unwrap_or("").to_string());
        }
    }

    let mut data = DataFrame::new();
    for (column, values) in numeric_columns.iter().zip(numeric_data) {
        data.add_numeric(column, values)
            .unwrap_or_else(|error| panic!("add numeric `{column}`: {error}"));
    }
    for (column, values) in categorical_columns.iter().zip(categorical_data) {
        data.add_categorical(column, values)
            .unwrap_or_else(|error| panic!("add categorical `{column}`: {error}"));
    }
    data
}

fn intact_glmm_data() -> DataFrame {
    load_prepared_case(
        "intact",
        &["correct", "soa_s"],
        &["participant", "item", "group", "mask", "block"],
    )
}

fn aphantasia_budget_grid() -> Vec<i64> {
    match std::env::var("MIXEFF_APHANTASIA_BUDGETS") {
        Ok(raw) => raw
            .split(',')
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .map(|part| {
                part.parse::<i64>().unwrap_or_else(|error| {
                    panic!("parse MIXEFF_APHANTASIA_BUDGETS value `{part}`: {error}")
                })
            })
            .collect(),
        Err(_) => vec![40, 80, 150, 300],
    }
}

#[cfg(all(not(feature = "nlopt"), feature = "unstable-internals"))]
fn fallback_joint_metric<'a>(
    model: &'a GeneralizedLinearMixedModel,
    key: &str,
) -> Option<&'a Value> {
    model
        .compiler_artifact()
        .diagnostics
        .iter()
        .find(|diagnostic| format!("{:?}", diagnostic.code) == "OptimizerRecovery")
        .and_then(|diagnostic| diagnostic.payload.get(key))
}

#[test]
fn aphantasia_fixture_snapshot_has_expected_reference_contract() {
    let cargo_toml =
        fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml"))
            .expect("read Cargo.toml");
    assert!(
        cargo_toml.contains("\"tests/fixtures/\""),
        "aphantasia fixture must remain under the crate exclude path"
    );

    let reference = reference_json();
    let intact = reference_model(&reference, "intact");
    assert_eq!(intact["model_type"], "glmm");
    assert_eq!(intact["family"], "binomial(logit)");
    assert_eq!(
        intact["formula"],
        "correct ~ group * mask * soa_s + block + (1 + mask + soa_s || participant) + (1 | item)"
    );
    assert_eq!(intact["nobs"].as_u64(), Some(5_760));
    assert!((intact["logLik"].as_f64().unwrap() + 1297.8855539491306).abs() < 1.0e-9);
    assert_eq!(csv_row_count("prepared/intact.csv"), 5_760);

    let combined = reference_model(&reference, "combined");
    assert_eq!(combined["model_type"], "glmm");
    assert_eq!(combined["family"], "binomial(logit)");
    assert_eq!(combined["nobs"].as_u64(), Some(23_040));
    assert!((combined["logLik"].as_f64().unwrap() + 11284.046284852095).abs() < 1.0e-9);
    assert_eq!(csv_row_count("prepared/combined.csv"), 23_040);
}

#[test]
fn intact_prepared_frame_builds_native_glmm_design_without_refitting() {
    let data = intact_glmm_data();
    assert_eq!(data.nrow(), 5_760);
    assert!(
        data.numeric("correct")
            .unwrap()
            .iter()
            .all(|value| *value == 0.0 || *value == 1.0),
        "`correct` must be a Bernoulli 0/1 response"
    );
    assert_eq!(
        data.categorical("group").unwrap().levels,
        vec!["aphant".to_string(), "control".to_string()]
    );
    assert_eq!(
        data.categorical("mask").unwrap().levels,
        vec!["masked".to_string(), "unmasked".to_string()]
    );

    let formula = parse_formula(
        "correct ~ group * mask * soa_s + block + (1 + mask + soa_s || participant) + (1 | item)",
    )
    .unwrap();
    let model = GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();

    assert_eq!(model.nobs(), 5_760);
    assert_eq!(model.fixef().len(), 9);
    assert_eq!(model.theta().len(), 4);
}

#[cfg(all(not(feature = "nlopt"), feature = "unstable-internals"))]
#[test]
#[ignore = "real aphantasia intact GLMM takes minutes; run intentionally for optimizer diagnostics"]
fn intact_native_joint_budget_grid_reports_real_fixture_progress() {
    let reference = reference_json();
    let intact_ref = reference_model(&reference, "intact");
    let reference_loglik = intact_ref["logLik"].as_f64().unwrap();
    let data = intact_glmm_data();
    let formula = parse_formula(
        "correct ~ group * mask * soa_s + block + (1 + mask + soa_s || participant) + (1 | item)",
    )
    .unwrap();

    let mut profiled =
        GeneralizedLinearMixedModel::new(formula.clone(), &data, Family::Bernoulli, None).unwrap();
    profiled.fit_with_options(true, 1, false).unwrap();
    let profiled_gap = (profiled.loglikelihood() - reference_loglik).abs();

    let budgets = aphantasia_budget_grid();
    assert!(
        !budgets.is_empty(),
        "MIXEFF_APHANTASIA_BUDGETS must name at least one budget"
    );
    for max_feval in budgets {
        let mut joint =
            GeneralizedLinearMixedModel::new(formula.clone(), &data, Family::Bernoulli, None)
                .unwrap();
        joint.lmm_mut().optsum_mut().max_feval = max_feval;
        joint.fit_with_options(false, 1, false).unwrap();
        let joint_gap = (joint.loglikelihood() - reference_loglik).abs();
        let joint_feval = fallback_joint_metric(&joint, "joint_feval")
            .and_then(Value::as_i64)
            .unwrap_or_else(|| joint.lmm().optsum().feval);
        let joint_fmin = fallback_joint_metric(&joint, "joint_fmin")
            .and_then(Value::as_f64)
            .unwrap_or_else(|| joint.objective());
        eprintln!(
            "aphantasia intact budget={max_feval} joint_feval={} final_feval={} joint_fmin={} loglik={} gap={} profiled_gap={} status={}",
            joint_feval,
            joint.lmm().optsum().feval,
            joint_fmin,
            joint.loglikelihood(),
            joint_gap,
            profiled_gap,
            joint.lmm().optsum().return_value
        );
        assert!(joint_feval <= max_feval);
        assert!(joint_gap.is_finite());
    }
}
