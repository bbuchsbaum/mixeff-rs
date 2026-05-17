use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;

use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Deserialize)]
struct DifficultScoreboard {
    schema_version: String,
    required_axes: Vec<String>,
    certification_statuses: Vec<String>,
    scenario: Vec<DifficultScenario>,
}

#[derive(Debug, Deserialize)]
struct DifficultScenario {
    id: String,
    axis: String,
    description: String,
    data_source: String,
    generator: String,
    dataset: String,
    formula: String,
    family: String,
    link: String,
    estimator: String,
    comparator_engines: Vec<String>,
    release_relevance: String,
    scorecard_class: String,
    certification_status: String,
    certification_claim: String,
    required_metrics: Vec<String>,
    rust_certification: String,
    comparator_certification: String,
    #[serde(default)]
    comparator_certifiable: Option<bool>,
    #[serde(default)]
    rust_time_multiplier: Option<f64>,
    #[serde(default)]
    comparator_time_multiplier: Option<f64>,
    #[serde(default)]
    comparator_extra_ms: Option<f64>,
    #[serde(default)]
    unit_test_name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ParityScorecard {
    row: Vec<ParityScorecardRow>,
}

#[derive(Debug, Deserialize)]
struct ParityScorecardRow {
    dataset: String,
    formula: String,
    family: String,
    link: String,
    estimator: String,
    #[serde(rename = "class")]
    class_name: String,
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn key(dataset: &str, formula: &str, family: &str, link: &str, estimator: &str) -> String {
    format!("{dataset}\n{formula}\n{family}\n{link}\n{estimator}")
}

fn scenario_key(scenario: &DifficultScenario) -> String {
    key(
        &scenario.dataset,
        &scenario.formula,
        &scenario.family,
        &scenario.link,
        &scenario.estimator,
    )
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

fn difficult_scoreboard() -> DifficultScoreboard {
    let path = repo_root().join("comparison/difficult_model_scoreboard.toml");
    toml::from_str(&fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}")))
        .unwrap_or_else(|e| panic!("parse {path:?}: {e}"))
}

fn parity_scorecard_by_key() -> BTreeMap<String, String> {
    let path = repo_root().join("comparison/parity_scorecard.toml");
    let scorecard: ParityScorecard =
        toml::from_str(&fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}")))
            .unwrap_or_else(|e| panic!("parse {path:?}: {e}"));
    scorecard
        .row
        .into_iter()
        .map(|row| {
            (
                key(
                    &row.dataset,
                    &row.formula,
                    &row.family,
                    &row.link,
                    &row.estimator,
                ),
                row.class_name,
            )
        })
        .collect()
}

fn read_json(path: &str) -> Value {
    let path = repo_root().join(path);
    serde_json::from_str(
        &fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}")),
    )
    .unwrap_or_else(|e| panic!("parse {path:?}: {e}"))
}

fn results_by_key(path: &str) -> BTreeMap<String, Value> {
    let json = read_json(path);
    json.get("results")
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{path}: missing results[]"))
        .iter()
        .map(|record| (record_key(record), record.clone()))
        .collect()
}

fn field_f64(record: &Value, field: &str, key: &str) -> f64 {
    let value = record
        .get(field)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("{key}: missing numeric `{field}`"));
    assert!(value.is_finite(), "{key}: `{field}` is not finite: {value}");
    value
}

fn field_str<'a>(record: &'a Value, field: &str, key: &str) -> &'a str {
    record
        .get(field)
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("{key}: missing string `{field}`"))
}

fn numeric_array(record: &Value, field: &str, key: &str) -> Vec<f64> {
    record
        .get(field)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{key}: missing array `{field}`"))
        .iter()
        .map(|value| {
            let value = value
                .as_f64()
                .unwrap_or_else(|| panic!("{key}: `{field}` contains non-numeric {value}"));
            assert!(
                value.is_finite(),
                "{key}: `{field}` contains non-finite {value}"
            );
            value
        })
        .collect()
}

fn beta_by_name(record: &Value, key: &str) -> BTreeMap<String, f64> {
    let names = record
        .get("coef_names")
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{key}: missing coef_names"))
        .iter()
        .map(|value| {
            value
                .as_str()
                .unwrap_or_else(|| panic!("{key}: non-string coef name {value}"))
                .replace(": ", "")
        })
        .collect::<Vec<_>>();
    let beta = numeric_array(record, "beta", key);
    assert_eq!(names.len(), beta.len(), "{key}: beta/name length mismatch");
    names.into_iter().zip(beta).collect()
}

fn max_abs_delta(left: &[f64], right: &[f64], label: &str) -> f64 {
    assert_eq!(left.len(), right.len(), "{label}: vector length mismatch");
    left.iter()
        .zip(right.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max)
}

fn beta_delta(left: &Value, right: &Value, key: &str) -> f64 {
    let left = beta_by_name(left, key);
    let right = beta_by_name(right, key);
    assert_eq!(
        left.keys().collect::<Vec<_>>(),
        right.keys().collect::<Vec<_>>(),
        "{key}: coefficient-name sets differ"
    );
    left.iter()
        .map(|(name, value)| {
            (value
                - right
                    .get(name)
                    .unwrap_or_else(|| panic!("{key}: missing beta for {name}")))
            .abs()
        })
        .fold(0.0_f64, f64::max)
}

fn time_to_certified_fit_ms(record: &Value, multiplier: f64, extra_ms: f64, key: &str) -> f64 {
    let fit_ms = field_f64(record, "fit_time_ms_min", key);
    let total = fit_ms * multiplier + extra_ms;
    assert!(
        total.is_finite() && total >= 0.0,
        "{key}: invalid time_to_certified_fit_ms {total}"
    );
    total
}

fn comparison_backed<'a>(
    scenario: &DifficultScenario,
    rows: &'a BTreeMap<String, Value>,
    label: &str,
) -> &'a Value {
    let key = scenario_key(scenario);
    rows.get(&key)
        .unwrap_or_else(|| panic!("{label} missing difficult scenario row {key}"))
}

fn uses_joint_glmm_gate(scenario: &DifficultScenario) -> bool {
    matches!(
        scenario.rust_certification.as_str(),
        "certified_joint_laplace" | "certified_joint_agq"
    ) && scenario.scorecard_class == "release_blocking_parity"
}

#[test]
fn difficult_scoreboard_covers_required_axes_and_reuses_parity_scorecard() {
    let scoreboard = difficult_scoreboard();
    let parity = parity_scorecard_by_key();
    assert_eq!(scoreboard.schema_version, "1.0.0");

    let statuses = scoreboard
        .certification_statuses
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    for label in [
        "release_blocking_parity",
        "documented_divergence",
        "diagnostic_contract",
        "performance_known_slow",
        "experimental_recovery",
    ] {
        assert!(
            statuses.contains(label),
            "difficult-model release label {label} must be present"
        );
    }
    let mut seen_ids = BTreeSet::new();
    let mut seen_axes = BTreeSet::new();
    let mut saw_non_certifiable_comparator = false;

    for scenario in &scoreboard.scenario {
        assert!(
            seen_ids.insert(&scenario.id),
            "duplicate id {}",
            scenario.id
        );
        assert!(
            !scenario.description.trim().is_empty(),
            "{}: description is required",
            scenario.id
        );
        assert!(
            !scenario.data_source.trim().is_empty() && !scenario.generator.trim().is_empty(),
            "{}: data source and generator are required",
            scenario.id
        );
        assert!(
            !scenario.comparator_engines.is_empty(),
            "{}: comparator engines are required",
            scenario.id
        );
        assert!(
            !scenario.release_relevance.trim().is_empty(),
            "{}: release relevance is required",
            scenario.id
        );
        assert!(
            statuses.contains(&scenario.certification_status),
            "{}: unknown certification status {}",
            scenario.id,
            scenario.certification_status
        );
        assert!(
            !scenario.certification_claim.trim().is_empty(),
            "{}: certification claim is required",
            scenario.id
        );
        assert!(
            !scenario.rust_certification.trim().is_empty()
                && !scenario.comparator_certification.trim().is_empty(),
            "{}: certification workflow labels are required",
            scenario.id
        );
        seen_axes.insert(scenario.axis.clone());

        if scenario.dataset != "unit_test" {
            let key = scenario_key(scenario);
            let scorecard_class = parity
                .get(&key)
                .unwrap_or_else(|| panic!("scoreboard row is not in parity scorecard: {key}"));
            assert_eq!(
                scorecard_class, &scenario.scorecard_class,
                "{}: scorecard class must match parity_scorecard.toml",
                scenario.id
            );
        } else {
            assert!(
                matches!(
                    scenario.scorecard_class.as_str(),
                    "diagnostic_contract" | "experimental_recovery"
                ),
                "{}: unit-test rows use diagnostic_contract or experimental_recovery, got {}",
                scenario.id,
                scenario.scorecard_class
            );
        }

        if scenario.comparator_certifiable == Some(false) {
            saw_non_certifiable_comparator = true;
        }
    }

    let required_axes = scoreboard
        .required_axes
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    assert_eq!(
        seen_axes, required_axes,
        "difficult scoreboard must cover every required pathology axis exactly at least once"
    );
    assert!(
        saw_non_certifiable_comparator,
        "scoreboard should include at least one comparator workflow with no certified fit"
    );
}

#[test]
fn comparison_backed_scenarios_compute_certification_times_and_metrics() {
    let scoreboard = difficult_scoreboard();
    let rust = results_by_key("comparison/rust_results.json");
    let lme4 = results_by_key("comparison/lme4_results.json");

    for scenario in scoreboard
        .scenario
        .iter()
        .filter(|scenario| scenario.dataset != "unit_test")
    {
        let key = scenario_key(scenario);
        let rust_row = comparison_backed(scenario, &rust, "rust_results.json");
        if uses_joint_glmm_gate(scenario) {
            let is_joint_agq = scenario.rust_certification == "certified_joint_agq";
            assert!(
                scenario.required_metrics.iter().any(|m| m == "objective_delta")
                    && scenario.required_metrics.iter().any(|m| m == "beta_delta")
                    && scenario.required_metrics.iter().any(|m| m == "theta_delta")
                    && scenario
                        .required_metrics
                        .iter()
                        .any(|m| m == "certification_status"),
                "{}: certified joint GLMM rows must declare the metrics verified by glmm_comparison_gates",
                scenario.id
            );
            let lme4_row = comparison_backed(scenario, &lme4, "lme4_results.json");
            assert_eq!(
                field_str(lme4_row, "status", &key),
                "ok",
                "{}: certified joint GLMM rows still require an ok lme4 reference artifact",
                scenario.id
            );
            assert_eq!(
                field_str(rust_row, "objective_definition", &key),
                if is_joint_agq {
                    "joint_glmm_agq_deviance"
                } else {
                    "joint_glmm_laplace_deviance"
                },
                "{}: certified joint GLMM row must use the joint objective artifact",
                scenario.id
            );
            assert_eq!(
                field_str(rust_row, "response_constants", &key),
                "included",
                "{}: certified joint GLMM row must retain response constants",
                scenario.id
            );
        }
        assert_eq!(
            field_str(rust_row, "status", &key),
            "ok",
            "{}: Rust difficult scenario must have a recorded fit or diagnostic row",
            scenario.id
        );

        if scenario
            .required_metrics
            .iter()
            .any(|metric| metric == "optimizer_status")
        {
            assert!(
                !field_str(rust_row, "optimizer_return_code", &key)
                    .trim()
                    .is_empty(),
                "{}: Rust optimizer status is required",
                scenario.id
            );
        }
        if scenario
            .required_metrics
            .iter()
            .any(|metric| metric == "singular_status")
        {
            assert!(
                rust_row
                    .get("is_singular")
                    .and_then(Value::as_bool)
                    .is_some(),
                "{}: Rust singular status is required",
                scenario.id
            );
        }
        if scenario
            .required_metrics
            .iter()
            .any(|metric| metric == "fevals")
        {
            assert!(
                field_f64(rust_row, "optimizer_fevals", &key) >= 0.0,
                "{}: Rust feval count must be present",
                scenario.id
            );
        }
        if scenario
            .required_metrics
            .iter()
            .any(|metric| metric == "wall_time_ms" || metric == "time_to_certified_fit_ms")
        {
            let rust_time = time_to_certified_fit_ms(
                rust_row,
                scenario.rust_time_multiplier.unwrap_or(1.0),
                0.0,
                &key,
            );
            assert!(
                rust_time >= 0.0,
                "{}: Rust time_to_certified_fit_ms should be computable",
                scenario.id
            );
        }

        let lme4_row = comparison_backed(scenario, &lme4, "lme4_results.json");
        if scenario.comparator_certifiable.unwrap_or(true) {
            assert_eq!(
                field_str(lme4_row, "status", &key),
                "ok",
                "{}: certifiable comparator scenario must have an ok reference fit",
                scenario.id
            );
            let comparator_time = time_to_certified_fit_ms(
                lme4_row,
                scenario.comparator_time_multiplier.unwrap_or(1.0),
                scenario.comparator_extra_ms.unwrap_or(0.0),
                &key,
            );
            assert!(
                comparator_time >= 0.0,
                "{}: comparator time_to_certified_fit_ms should be computable",
                scenario.id
            );
        } else {
            assert_ne!(
                field_str(lme4_row, "status", &key),
                "ok",
                "{}: non-certifiable comparator should not be an ordinary ok fit",
                scenario.id
            );
        }

        if scenario
            .required_metrics
            .iter()
            .any(|metric| metric == "warnings")
        {
            assert!(
                lme4_row.get("warnings").and_then(Value::as_array).is_some(),
                "{}: comparator warnings must be recorded",
                scenario.id
            );
        }

        if scenario
            .required_metrics
            .iter()
            .any(|metric| metric == "objective_delta")
        {
            assert_eq!(
                rust_row.get("response_constants").and_then(Value::as_str),
                lme4_row.get("response_constants").and_then(Value::as_str),
                "{}: objective delta requires comparable objective conventions",
                scenario.id
            );
            let delta = (field_f64(rust_row, "objective", &key)
                - field_f64(lme4_row, "objective", &key))
            .abs();
            assert!(delta.is_finite(), "{}: objective delta", scenario.id);
        }

        if scenario
            .required_metrics
            .iter()
            .any(|metric| metric == "beta_delta")
        {
            let delta = beta_delta(rust_row, lme4_row, &key);
            assert!(delta.is_finite(), "{}: beta delta", scenario.id);
        }

        if scenario
            .required_metrics
            .iter()
            .any(|metric| metric == "theta_delta")
        {
            let delta = max_abs_delta(
                &numeric_array(rust_row, "theta", &key),
                &numeric_array(lme4_row, "theta", &key),
                &format!("{key}: theta"),
            );
            assert!(delta.is_finite(), "{}: theta delta", scenario.id);
        }
    }
}

/// Extract the source of the test function named `name` from a `#[cfg(test)]`
/// module. The block runs from `fn {name}` up to the next test-module item
/// (`\n    fn ` or `\n    #[`), which is enough to capture the fixture/formula
/// setup at the top of every unit test we point at. Returns `None` if the
/// function is not present.
fn test_source_block<'a>(src: &'a str, name: &str) -> Option<&'a str> {
    let start = src.find(&format!("fn {name}"))?;
    let rest = &src[start..];
    // Skip past the signature so the boundary scan doesn't stop on it.
    let after_sig = rest.find('{').map(|i| i + 1).unwrap_or(0);
    let end = rest[after_sig..]
        .find("\n    fn ")
        .or_else(|| rest[after_sig..].find("\n    #["))
        .map(|i| after_sig + i)
        .unwrap_or(rest.len());
    Some(&rest[..end])
}

/// A unit-test scenario is consistent with the test it points at only when
/// the test actually constructs the declared formula and family. This is the
/// guard that comparison-backed rows get for free (by key-matching the
/// results JSON) but unit-test rows previously lacked.
fn unit_scenario_matches_test(block: &str, formula: &str, family: &str) -> bool {
    if !block.contains(formula) {
        return false;
    }
    if family == "Gaussian" {
        // Gaussian/Identity unit tests drive the LMM directly.
        block.contains("LinearMixedModel")
    } else {
        block.contains(&format!("Family::{family}"))
    }
}

#[test]
fn unit_test_backed_recovery_scenarios_point_to_existing_tests() {
    let scoreboard = difficult_scoreboard();
    let linear_rs = fs::read_to_string(repo_root().join("src/model/linear.rs"))
        .expect("read src/model/linear.rs");
    let generalized_rs = fs::read_to_string(repo_root().join("src/model/generalized.rs"))
        .expect("read src/model/generalized.rs");
    let mut unit_rows = 0;

    for scenario in scoreboard
        .scenario
        .iter()
        .filter(|scenario| scenario.dataset == "unit_test")
    {
        unit_rows += 1;
        let test_name = scenario
            .unit_test_name
            .as_deref()
            .unwrap_or_else(|| panic!("{}: unit_test_name is required", scenario.id));

        let block = test_source_block(&linear_rs, test_name)
            .or_else(|| test_source_block(&generalized_rs, test_name))
            .unwrap_or_else(|| {
                panic!(
                    "{}: unit-test backed scenario points to missing test {test_name}",
                    scenario.id
                )
            });

        // The core anti-drift guard: the referenced test must actually fit
        // the formula/family the manifest claims for this scenario, not just
        // happen to share a function name.
        assert!(
            unit_scenario_matches_test(block, &scenario.formula, &scenario.family),
            "{}: unit-test {test_name} does not construct the declared scenario \
             (formula `{}`, family `{}`); the manifest and the test have drifted",
            scenario.id,
            scenario.formula,
            scenario.family
        );

        assert!(
            scenario
                .required_metrics
                .iter()
                .any(|metric| metric == "certification_status"),
            "{}: unit-test recovery rows must record certification status",
            scenario.id
        );
    }

    assert!(
        unit_rows > 0,
        "at least one unit-test-backed recovery row is required"
    );
}

/// Acceptance guard for bd-01KRVGFDKGK71KXHA0X5CAANVE: a deliberately drifted
/// unit-test scenario must be rejected by the contract above. This pins the
/// detector itself so the guard cannot silently regress to a name-only check.
#[test]
fn drifted_unit_test_scenario_is_rejected() {
    let synthetic = "    fn t_block() {\n        \
        let formula = parse_formula(\"y ~ 1 + x + (1 | group)\").unwrap();\n        \
        let model = GeneralizedLinearMixedModel::new(formula, &data, Family::Gamma, None);\n    }\n    \
        fn next_test() {}\n";
    let block = test_source_block(synthetic, "t_block").expect("block found");

    // Exact declared scenario: accepted.
    assert!(unit_scenario_matches_test(
        block,
        "y ~ 1 + x + (1 | group)",
        "Gamma"
    ));
    // Drifted formula (the bd-01KRVA2201SY7W2TSZEANCERG5 Finding 1 case): rejected.
    assert!(!unit_scenario_matches_test(
        block,
        "y ~ 1 + x + (1 | g)",
        "Gamma"
    ));
    // Drifted family: rejected.
    assert!(!unit_scenario_matches_test(
        block,
        "y ~ 1 + x + (1 | group)",
        "Poisson"
    ));
    // Missing function: no block, so the real test would panic with a clear
    // missing-test message.
    assert!(test_source_block(synthetic, "absent_test").is_none());
}
