//! A fresh process must be able to query a snapshot without a live model/cache.
use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::generalized::GlmmPredictionScale;
use mixeff_rs::model::linear::NewReLevels;
use mixeff_rs::model::{
    DataFrame, Family, GeneralizedLinearMixedModel, LinearMixedModel, MixedModelFit,
};
use serde_json::{json, Value};

fn data() -> DataFrame {
    let mut data = DataFrame::new();
    let mut y = Vec::new();
    let mut counts = Vec::new();
    let mut x = Vec::new();
    let mut groups = Vec::new();
    for group in 0..8 {
        for observation in 0..10 {
            let value = observation as f64 / 5.0 - 1.0;
            let noise = ((group * 13 + observation * 7) as f64).sin();
            x.push(value);
            y.push(3.0 + 0.7 * value + group as f64 * 0.3 + noise);
            counts.push(
                ((0.8 + 0.2 * value + group as f64 * 0.05).exp() + noise)
                    .round()
                    .max(0.0),
            );
            groups.push(format!("g{group}"));
        }
    }
    data.add_numeric("y", y).unwrap();
    data.add_numeric("counts", counts).unwrap();
    data.add_numeric("x", x).unwrap();
    data.add_categorical("group", groups).unwrap();
    data
}

fn lmm_results(model: &LinearMixedModel) -> Value {
    let data = data();
    let value = json!({
        "beta": model.coef().as_slice(),
        "theta": model.theta(),
        "objective": model.objective(),
        "vcov": model.vcov(),
        "conditional_variance": model.cond_var(),
        "predictions": model.predict_new(&data, NewReLevels::Error).unwrap(),
        "uncertainty": model.predict_new_variance(&data, NewReLevels::Error).unwrap(),
        "inference": model.fixed_effect_inference_table(),
        "summary": mixeff_rs::stats::model_summary::FitSummaryPayload::from_linear_model(model),
    });
    value
}

fn glmm_results(model: &GeneralizedLinearMixedModel) -> Value {
    let data = data();
    json!({
        "beta": model.coef().as_slice(),
        "theta": model.theta(),
        "objective": model.objective(),
        "vcov": model.vcov(),
        "predictions": model.predict_new(&data, GlmmPredictionScale::Response, NewReLevels::Error).unwrap(),
        "uncertainty": model.predict_new_variance(&data, GlmmPredictionScale::Response, NewReLevels::Error).unwrap(),
        "summary": mixeff_rs::stats::model_summary::FitSummaryPayload::from_generalized_model(model),
    })
}

fn assert_json_close(actual: &Value, expected: &Value, path: &str) {
    match (actual, expected) {
        (Value::Number(a), Value::Number(b)) => {
            let (a, b) = (a.as_f64().unwrap(), b.as_f64().unwrap());
            assert!(
                (a - b).abs() <= 1e-9 * (1.0 + b.abs()),
                "{path}: {a} != {b}"
            );
        }
        (Value::Array(a), Value::Array(b)) => {
            assert_eq!(a.len(), b.len(), "{path}: length");
            for (index, (a, b)) in a.iter().zip(b).enumerate() {
                assert_json_close(a, b, &format!("{path}[{index}]"));
            }
        }
        (Value::Object(a), Value::Object(b)) => {
            assert_eq!(a.len(), b.len(), "{path}: keys");
            for (key, b) in b {
                assert_json_close(
                    a.get(key).expect("missing key"),
                    b,
                    &format!("{path}.{key}"),
                );
            }
        }
        _ => assert_eq!(actual, expected, "{path}"),
    }
}

#[test]
fn snapshots_support_post_fit_queries_in_a_fresh_process() {
    let data = data();
    let mut lmm =
        LinearMixedModel::new(parse_formula("y ~ x + (1 | group)").unwrap(), &data, None).unwrap();
    lmm.fit(true).unwrap();
    let mut glmm = GeneralizedLinearMixedModel::new(
        parse_formula("counts ~ x + (1 | group)").unwrap(),
        &data,
        Family::Poisson,
        None,
    )
    .unwrap();
    glmm.fit_with_options(false, 1, false).unwrap();
    let expected = json!({"lmm": lmm_results(&lmm), "glmm": glmm_results(&glmm)});
    let snapshots =
        json!({"lmm": lmm.snapshot_json().unwrap(), "glmm": glmm.snapshot_json().unwrap()});
    drop((lmm, glmm, data));
    let directory = std::env::temp_dir().join(format!(
        "mixeff-restore-process-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&directory).unwrap();
    std::fs::write(directory.join("snapshots.json"), snapshots.to_string()).unwrap();
    std::fs::write(directory.join("expected.json"), expected.to_string()).unwrap();
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "restored_process_child",
            "--ignored",
            "--nocapture",
        ])
        .env("MIXEFF_RESTORE_PROCESS_DIR", &directory)
        .output()
        .unwrap();
    std::fs::remove_dir_all(&directory).unwrap();
    assert!(
        output.status.success(),
        "child failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
#[ignore = "invoked by snapshots_support_post_fit_queries_in_a_fresh_process"]
fn restored_process_child() {
    let directory =
        std::path::PathBuf::from(std::env::var_os("MIXEFF_RESTORE_PROCESS_DIR").unwrap());
    let snapshots: Value =
        serde_json::from_slice(&std::fs::read(directory.join("snapshots.json")).unwrap()).unwrap();
    let expected: Value =
        serde_json::from_slice(&std::fs::read(directory.join("expected.json")).unwrap()).unwrap();
    let lmm = LinearMixedModel::restore_json(snapshots["lmm"].as_str().unwrap()).unwrap();
    let glmm =
        GeneralizedLinearMixedModel::restore_json(snapshots["glmm"].as_str().unwrap()).unwrap();
    assert_json_close(&lmm_results(&lmm), &expected["lmm"], "lmm");
    assert_json_close(&glmm_results(&glmm), &expected["glmm"], "glmm");
}
