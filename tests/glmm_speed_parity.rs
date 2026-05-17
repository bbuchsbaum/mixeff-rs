//! Speed gate for representative GLMM comparison rows.
//!
//! The comparison harness records Rust and lme4 timings for every dataset fit.
//! This test turns the GLMM subset into an explicit performance artifact:
//! routine rows must have timings, fevals, optimizer return codes, and a
//! threshold classification; stress rows must stay gated out of routine regen.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use serde_json::Value;

#[derive(Clone, Copy)]
struct GlmmSpeedCase {
    dataset: &'static str,
    formula: &'static str,
    family: &'static str,
    link: &'static str,
    estimator: &'static str,
    status: &'static str,
    minimum_speedup: Option<f64>,
    known_slow_bead: Option<&'static str>,
}

const SPEED_CASES: &[GlmmSpeedCase] = &[
    GlmmSpeedCase {
        dataset: "cbpp",
        formula: "incidence / size ~ 1 + period + (1 | herd)",
        family: "Binomial",
        link: "Logit",
        estimator: "Laplace",
        status: "ok",
        minimum_speedup: Some(1.0),
        known_slow_bead: None,
    },
    GlmmSpeedCase {
        dataset: "grouseticks",
        formula: "TICKS ~ 1 + YEAR + cHEIGHT + (1 | BROOD) + (1 | INDEX) + (1 | LOCATION)",
        family: "Poisson",
        link: "Log",
        estimator: "Laplace",
        status: "ok",
        minimum_speedup: Some(1.0),
        known_slow_bead: Some("bd-01KRSQYRHF8VK627HZ6Z23CP93"),
    },
    GlmmSpeedCase {
        dataset: "verbagg",
        formula: "r2 ~ 1 + Anger + Gender + btype + situ + mode + (1 | id) + (1 | item)",
        family: "Binomial",
        link: "Logit",
        estimator: "Laplace",
        status: "ok",
        minimum_speedup: Some(1.0),
        known_slow_bead: None,
    },
    GlmmSpeedCase {
        dataset: "tungara_single_caller",
        formula: "did_focal_follower_overlap_preceding_call_YN ~ 1 + preceding_caller_mass * distance + (1 | chorus_ID) + (1 + preceding_caller_mass | chorus_ID:focal_toe_clip_number)",
        family: "Binomial",
        link: "Logit",
        estimator: "Laplace",
        status: "skipped_stress",
        minimum_speedup: None,
        known_slow_bead: None,
    },
];

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read_json(path: &str) -> Value {
    let path = repo_root().join(path);
    serde_json::from_str(
        &fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}")),
    )
    .unwrap_or_else(|e| panic!("parse {path:?}: {e}"))
}

fn key(dataset: &str, formula: &str, family: &str, link: &str, estimator: &str) -> String {
    format!("{dataset}\n{formula}\n{family}\n{link}\n{estimator}")
}

fn record_key(record: &Value) -> String {
    key(
        record
            .get("dataset")
            .and_then(Value::as_str)
            .unwrap_or("<missing dataset>"),
        record
            .get("formula")
            .and_then(Value::as_str)
            .unwrap_or("<missing formula>"),
        record
            .get("family")
            .and_then(Value::as_str)
            .unwrap_or("<missing family>"),
        record
            .get("link")
            .and_then(Value::as_str)
            .unwrap_or("<missing link>"),
        record
            .get("estimator")
            .and_then(Value::as_str)
            .unwrap_or("<missing estimator>"),
    )
}

fn case_key(case: GlmmSpeedCase) -> String {
    key(
        case.dataset,
        case.formula,
        case.family,
        case.link,
        case.estimator,
    )
}

fn results_by_key(file: &Value, label: &str) -> BTreeMap<String, Value> {
    let mut by_key = BTreeMap::new();
    for record in file
        .get("results")
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{label}: missing results[]"))
    {
        let key = record_key(record);
        let old = by_key.insert(key.clone(), record.clone());
        assert!(old.is_none(), "{label}: duplicate comparison key {key}");
    }
    by_key
}

fn field_f64(record: &Value, field: &str, key: &str) -> f64 {
    let value = record
        .get(field)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("{key}: missing numeric `{field}`"));
    assert!(value.is_finite(), "{key}: `{field}` is not finite: {value}");
    value
}

fn positive_i64(record: &Value, field: &str, key: &str) -> i64 {
    let value = record
        .get(field)
        .and_then(Value::as_i64)
        .unwrap_or_else(|| panic!("{key}: missing integer `{field}`"));
    assert!(value > 0, "{key}: `{field}` must be positive");
    value
}

fn nonempty_str<'a>(record: &'a Value, field: &str, key: &str) -> &'a str {
    let value = record
        .get(field)
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("{key}: missing string `{field}`"));
    assert!(!value.is_empty(), "{key}: `{field}` must not be empty");
    value
}

fn assert_ok_speed_payload(record: &Value, key: &str) {
    assert_eq!(
        record.get("status").and_then(Value::as_str),
        Some("ok"),
        "{key}: speed row must be ok"
    );
    assert!(
        field_f64(record, "fit_time_ms_min", key) > 0.0,
        "{key}: fit_time_ms_min must be positive"
    );
    positive_i64(record, "optimizer_fevals", key);
    for field in ["optimizer", "optimizer_backend", "optimizer_return_code"] {
        nonempty_str(record, field, key);
    }
}

#[test]
fn glmm_speed_rows_cover_representative_cases_and_thresholds() {
    let rust = read_json("comparison/rust_results.json");
    let lme4 = read_json("comparison/lme4_results.json");
    let rust_by_key = results_by_key(&rust, "rust_results.json");
    let lme4_by_key = results_by_key(&lme4, "lme4_results.json");

    for case in SPEED_CASES {
        let key = case_key(*case);
        let rust_row = rust_by_key
            .get(&key)
            .unwrap_or_else(|| panic!("rust_results.json missing speed row {key}"));
        let lme4_row = lme4_by_key
            .get(&key)
            .unwrap_or_else(|| panic!("lme4_results.json missing speed row {key}"));
        assert_eq!(
            rust_row.get("status").and_then(Value::as_str),
            Some(case.status),
            "rust_results.json: unexpected status for {key}"
        );
        assert_eq!(
            lme4_row.get("status").and_then(Value::as_str),
            Some(case.status),
            "lme4_results.json: unexpected status for {key}"
        );

        if case.status == "skipped_stress" {
            let reason = rust_row.get("error").and_then(Value::as_str).unwrap_or("");
            assert!(
                reason.contains("MIXEDMODELS_INCLUDE_STRESS=1"),
                "{key}: stress row must name the opt-in env var"
            );
            continue;
        }

        assert_ok_speed_payload(rust_row, &key);
        assert_ok_speed_payload(lme4_row, &key);

        let rust_ms = field_f64(rust_row, "fit_time_ms_min", &key);
        let lme4_ms = field_f64(lme4_row, "fit_time_ms_min", &key);
        let speedup = lme4_ms / rust_ms;
        assert!(
            speedup.is_finite() && speedup > 0.0,
            "{key}: speedup must be finite and positive"
        );

        if let Some(minimum) = case.minimum_speedup {
            if let Some(bead) = case.known_slow_bead {
                assert!(
                    speedup < minimum,
                    "{key}: known slow row now passes {minimum:.2}x; remove `{bead}` from the speed allowlist and enforce the threshold"
                );
            } else {
                assert!(
                    speedup >= minimum,
                    "{key}: Rust speedup {speedup:.2}x is below threshold {minimum:.2}x"
                );
            }
        }
    }
}

#[test]
fn glmm_speed_report_exposes_ratios_fevals_and_optimizer_codes() {
    let report =
        fs::read_to_string(repo_root().join("comparison").join("REPORT.md")).expect("read report");
    assert!(
        report.contains("| Dataset | Formula | Est | n | t_R (ms, min) | t_Rust (ms, min) | speedup | R fevals | Rust fevals | Rust optimizer |"),
        "comparison/REPORT.md must expose GLMM speed ratios, fevals, and optimizer labels"
    );
    assert!(
        report.contains("/bobyqa") || report.contains("/cobyla"),
        "comparison/REPORT.md must expose Rust optimizer backend and return code"
    );
    for case in SPEED_CASES {
        assert!(
            report.contains(&format!("`{}`", case.dataset)),
            "comparison/REPORT.md missing representative GLMM speed dataset `{}`",
            case.dataset
        );
    }
}
