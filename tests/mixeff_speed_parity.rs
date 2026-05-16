use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use serde_json::Value;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read_json(path: PathBuf) -> Value {
    let text = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|error| panic!("parse {}: {error}", path.display()))
}

fn results_by_case(file: &Value) -> HashMap<String, &Value> {
    file.get("results")
        .and_then(Value::as_array)
        .expect("benchmark JSON must contain a results array")
        .iter()
        .map(|row| {
            let id = row
                .get("case_id")
                .and_then(Value::as_str)
                .expect("benchmark row must contain case_id")
                .to_string();
            (id, row)
        })
        .collect()
}

fn required_number<'a>(row: &'a Value, field: &str, case_id: &str) -> f64 {
    row.get(field)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("{case_id}: field `{field}` must be numeric"))
}

#[test]
fn mixeff_speed_parity_results_cover_expected_cases_and_pass() {
    let root = repo_root().join("comparison").join("mixeff");
    let rust = read_json(root.join("rust_results.json"));
    let lme4 = read_json(root.join("lme4_results.json"));
    let report = fs::read_to_string(root.join("REPORT.md")).expect("read mixeff REPORT.md");

    assert_eq!(
        rust.get("schema_name").and_then(Value::as_str),
        Some("mixedmodels.mixeff_speed_parity")
    );
    assert_eq!(
        lme4.get("schema_name").and_then(Value::as_str),
        Some("mixedmodels.mixeff_speed_parity")
    );

    let rust_by_case = results_by_case(&rust);
    let lme4_by_case = results_by_case(&lme4);
    let expected_cases = [
        "brown_rt_full",
        "iamciera_max_model",
        "sdamr_speeddate_maximal_crossed",
        "sdamr_speeddate_uncorrelated_crossed",
    ];

    for case_id in expected_cases {
        let rust_row = rust_by_case
            .get(case_id)
            .unwrap_or_else(|| panic!("rust_results.json missing `{case_id}`"));
        let lme4_row = lme4_by_case
            .get(case_id)
            .unwrap_or_else(|| panic!("lme4_results.json missing `{case_id}`"));

        assert_eq!(
            rust_row.get("status").and_then(Value::as_str),
            Some("ok"),
            "{case_id}: Rust benchmark row must be ok"
        );
        assert_eq!(
            lme4_row.get("status").and_then(Value::as_str),
            Some("ok"),
            "{case_id}: lme4 benchmark row must be ok"
        );

        let rust_ms = required_number(rust_row, "fit_time_ms_min", case_id);
        let lme4_ms = required_number(lme4_row, "fit_time_ms_min", case_id);
        assert!(
            rust_ms.is_finite() && rust_ms > 0.0,
            "{case_id}: Rust minimum timing must be finite and positive"
        );
        assert!(
            lme4_ms.is_finite() && lme4_ms > 0.0,
            "{case_id}: lme4 minimum timing must be finite and positive"
        );
        assert!(
            lme4_ms / rust_ms >= 1.0,
            "{case_id}: recorded Rust timing {rust_ms:.1}ms is slower than lme4 {lme4_ms:.1}ms"
        );

        assert!(
            rust_row.get("fevals").and_then(Value::as_i64).unwrap_or(0) > 0,
            "{case_id}: Rust fevals must be recorded"
        );
        assert!(
            lme4_row.get("fevals").and_then(Value::as_i64).unwrap_or(0) > 0,
            "{case_id}: lme4 fevals must be recorded"
        );
    }

    assert!(
        !report.contains("rust_slower"),
        "REPORT.md should not contain rust_slower rows after regenerating parity results"
    );
}
