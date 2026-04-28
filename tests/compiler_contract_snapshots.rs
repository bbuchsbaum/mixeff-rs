use std::fs;
use std::path::Path;

use mixedmodels::compiler::{
    certify, compile_formula_ir, expected_statuses, generated_x_values, CompiledModelArtifact,
    CompilerPolicy, FitStatus, GeneratorSpec, ModelAuditReport,
};
use mixedmodels::datasets;
use mixedmodels::formula::parse_formula;
use mixedmodels::model::{DataFrame, GeneralizedLinearMixedModel, LinearMixedModel};
use mixedmodels::model::{Family, LinkFunction};
use serde::Serialize;

const UPDATE_ENV: &str = "MIXEDMODELS_UPDATE_WIRE_FIXTURES";

fn compiler_contract_data() -> DataFrame {
    let mut data = DataFrame::new();
    data.add_numeric("y", vec![1.0, 2.1, 3.2, 4.1, 5.0, 6.2]);
    data.add_numeric("x", vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
    data.add_numeric("x2", vec![0.0, 2.0, 0.0, 2.0, 0.0, 2.0]);
    data.add_categorical(
        "subject",
        vec!["s1", "s1", "s2", "s2", "s3", "s3"]
            .into_iter()
            .map(str::to_string)
            .collect(),
    );
    data
}

fn compiler_contract_artifact() -> CompiledModelArtifact {
    let formula = parse_formula("y ~ x + x2 + (1 + x | subject)").unwrap();
    let semantic = compile_formula_ir(&formula);
    let mut artifact = CompiledModelArtifact::new(formula.to_string(), semantic);
    artifact.attach_design_audit(&compiler_contract_data());
    artifact
}

fn sleepstudy_artifact() -> CompiledModelArtifact {
    let (data, meta) = datasets::load("sleepstudy").unwrap();
    let formula = parse_formula(&meta.fits[0].formula).unwrap();
    let model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.compiler_artifact().clone()
}

fn sleepstudy_artifact_for_formula(formula_text: &str) -> CompiledModelArtifact {
    let (data, _meta) = datasets::load("sleepstudy").unwrap();
    let formula = parse_formula(formula_text).unwrap();
    let model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.compiler_artifact().clone()
}

fn penicillin_artifact() -> CompiledModelArtifact {
    let (data, meta) = datasets::load("penicillin").unwrap();
    let formula = parse_formula(&meta.fits[0].formula).unwrap();
    let model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.compiler_artifact().clone()
}

fn confounded_fixed_random_data() -> DataFrame {
    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut subject = Vec::new();
    for idx in 0..6 {
        y.push(idx as f64);
        y.push(idx as f64 + 0.5);
        x.push(0.0);
        x.push(1.0);
        subject.push(format!("s{}", idx + 1));
        subject.push(format!("s{}", idx + 1));
    }

    let mut data = DataFrame::new();
    data.add_numeric("y", y);
    data.add_numeric("x", x);
    data.add_categorical("subject", subject);
    data
}

fn confounded_fixed_random_artifact() -> CompiledModelArtifact {
    let formula = parse_formula("y ~ subject + x + (1 | subject)").unwrap();
    let model = LinearMixedModel::new(formula, &confounded_fixed_random_data(), None).unwrap();
    model.compiler_artifact().clone()
}

fn categorical_random_basis_data() -> DataFrame {
    let levels = ["A", "B", "C"];
    let mut y = Vec::new();
    let mut cond = Vec::new();
    let mut subject = Vec::new();

    for subject_index in 0..40 {
        for (level_index, level) in levels.iter().enumerate() {
            y.push(subject_index as f64 + level_index as f64);
            cond.push((*level).to_string());
            subject.push(format!("s{}", subject_index + 1));
        }
    }

    let mut data = DataFrame::new();
    data.add_numeric("y", y);
    data.add_categorical("cond", cond);
    data.add_categorical("subject", subject);
    data
}

fn categorical_random_basis_artifact() -> CompiledModelArtifact {
    let formula = parse_formula("y ~ cond + (1 + cond | subject)").unwrap();
    let model = LinearMixedModel::new(formula, &categorical_random_basis_data(), None).unwrap();
    model.compiler_artifact().clone()
}

fn cbpp_glmm_artifact() -> CompiledModelArtifact {
    let (data, _meta) = datasets::load("cbpp").unwrap();
    let formula = parse_formula("incidence ~ 1 + period + (1 | herd)").unwrap();
    let model = GeneralizedLinearMixedModel::new(
        formula,
        &data,
        Family::Binomial,
        Some(LinkFunction::Logit),
    )
    .unwrap();
    model.compiler_artifact().clone()
}

fn rank_mixture_data() -> DataFrame {
    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut group = Vec::new();

    for group_index in 0..18 {
        let eta = group_index as f64 - 8.5;
        let intercept_shift = 0.07 * eta;
        let slope_shift = 0.03 * eta;
        for x_value in [-1.0, -0.25, 0.25, 1.0] {
            y.push(10.0 + intercept_shift + (2.0 + slope_shift) * x_value);
            x.push(x_value);
            group.push(format!("g{}", group_index + 1));
        }
    }

    let mut data = DataFrame::new();
    data.add_numeric("y", y);
    data.add_numeric("x", x);
    data.add_categorical("group", group);
    data
}

fn rank_mixture_artifact() -> CompiledModelArtifact {
    let formula = parse_formula("y ~ x + (1 + x | group)").unwrap();
    let mut model = LinearMixedModel::new(formula, &rank_mixture_data(), None).unwrap();
    model.fit(true).unwrap();
    model.verify_convergence().unwrap();
    model.compiler_artifact().clone()
}

fn design_compiled_data() -> DataFrame {
    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut group = Vec::new();

    for group_index in 0..6 {
        y.push(group_index as f64);
        y.push(group_index as f64 + 1.0);
        x.push(0.0);
        x.push(1.0);
        group.push(format!("g{}", group_index + 1));
        group.push(format!("g{}", group_index + 1));
    }

    let mut data = DataFrame::new();
    data.add_numeric("y", y);
    data.add_numeric("x", x);
    data.add_categorical("group", group);
    data
}

fn design_compiled_artifact() -> CompiledModelArtifact {
    let formula = parse_formula("y ~ x + (1 + x | group)").unwrap();
    let model = LinearMixedModel::new_with_compiler_policy(
        formula,
        &design_compiled_data(),
        None,
        CompilerPolicy::design_compiled(),
    )
    .unwrap();
    model.compiler_artifact().clone()
}

fn pedagogical_diagnostics_data() -> DataFrame {
    let mut data = DataFrame::new();
    data.add_numeric("y", vec![1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]);
    data.add_numeric("x", vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]);
    data.add_categorical(
        "subject",
        vec!["s1", "s1", "s1", "s1", "s2", "s2", "s2", "s2"]
            .into_iter()
            .map(str::to_string)
            .collect(),
    );
    data.add_categorical(
        "school",
        vec!["A", "A", "A", "A", "B", "B", "B", "B"]
            .into_iter()
            .map(str::to_string)
            .collect(),
    );
    data.add_categorical(
        "class",
        vec!["c1", "c1", "c2", "c2", "c1", "c1", "c2", "c2"]
            .into_iter()
            .map(str::to_string)
            .collect(),
    );
    data.add_categorical(
        "batch",
        vec!["b1", "b1", "b1", "b1", "b2", "b2", "b2", "b2"]
            .into_iter()
            .map(str::to_string)
            .collect(),
    );
    data.add_categorical(
        "item",
        vec!["i1", "i2", "i1", "i2", "i3", "i4", "i3", "i4"]
            .into_iter()
            .map(str::to_string)
            .collect(),
    );
    data
}

fn pedagogical_diagnostics_artifact() -> CompiledModelArtifact {
    let formula = parse_formula(
        "y ~ x + (1 | subject) + (1 + x | school/class) + (1 + x || batch) + (1 | item) + (0 + x | item)",
    )
    .unwrap();
    let semantic = compile_formula_ir(&formula);
    let mut artifact = CompiledModelArtifact::new(formula.to_string(), semantic);
    artifact.attach_design_audit(&pedagogical_diagnostics_data());
    artifact
}

fn singular_prefit_artifact() -> CompiledModelArtifact {
    let (data, meta) = datasets::load("singular").unwrap();
    let formula = parse_formula(&meta.fits[0].formula).unwrap();
    let model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.compiler_artifact().clone()
}

fn pathology_fixture_paths() -> Vec<&'static str> {
    vec![
        "tests/fixtures/pathology_corpus/easy.toml",
        "tests/fixtures/pathology_corpus/boundary.toml",
        "tests/fixtures/pathology_corpus/reduced_rank.toml",
        "tests/fixtures/pathology_corpus/refusal.toml",
    ]
}

fn load_pathology_spec(relative_path: &str) -> GeneratorSpec {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(relative_path);
    let text = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {relative_path}: {error}"));
    toml::from_str(&text).unwrap_or_else(|error| panic!("failed to parse {relative_path}: {error}"))
}

fn pathology_data(spec: &GeneratorSpec) -> DataFrame {
    let x_values = generated_x_values(spec);
    let mut group = Vec::with_capacity(x_values.len());
    for (group_index, &group_size) in spec.group_sizes.iter().enumerate() {
        for _ in 0..group_size {
            group.push(format!("g{}", group_index + 1));
        }
    }

    let beta0 = spec.fe_truth.first().copied().unwrap_or(0.0);
    let beta1 = spec.fe_truth.get(1).copied().unwrap_or(0.0);
    let y = x_values
        .iter()
        .enumerate()
        .map(|(row, x)| {
            let signal = beta0 + beta1 * x;
            let deterministic_noise = ((row as f64 + spec.seed as f64) * 0.37).sin();
            signal + spec.residual_sd * 0.05 * deterministic_noise
        })
        .collect::<Vec<_>>();

    let mut data = DataFrame::new();
    data.add_numeric("y", y);
    data.add_numeric("x", x_values.clone());
    if spec.fe_truth.len() > 2 {
        let x2 = match spec.fixed_design {
            mixedmodels::compiler::FixedDesign::CollinearFixedEffect => x_values,
            _ => x_values.iter().map(|value| value * value).collect(),
        };
        data.add_numeric("x2", x2);
    }
    data.add_categorical("group", group);
    data
}

fn pathology_formula(spec: &GeneratorSpec) -> &'static str {
    if spec.fe_truth.len() > 2 {
        "y ~ 1 + x + x2 + (1 + x | group)"
    } else {
        "y ~ 1 + x + (1 + x | group)"
    }
}

fn status_in_expected_set(status: FitStatus, expected: &[FitStatus]) -> bool {
    expected.contains(&status)
}

fn pretty_json<T: Serialize>(value: &T) -> String {
    let mut text = serde_json::to_string_pretty(value).unwrap();
    text.push('\n');
    text
}

fn json_section_by_title<'a>(value: &'a serde_json::Value, title: &str) -> &'a serde_json::Value {
    value["sections"]
        .as_array()
        .unwrap()
        .iter()
        .find(|section| section["title"] == title)
        .unwrap_or_else(|| panic!("missing report section {title}"))
}

fn random_term_card_fixture_value(report: &ModelAuditReport) -> serde_json::Value {
    let mut object = serde_json::Map::new();
    object.insert(
        "random_term_cards".to_string(),
        serde_json::to_value(&report.random_term_cards).unwrap(),
    );
    object.insert(
        "cross_card_constraints".to_string(),
        serde_json::to_value(&report.cross_card_constraints).unwrap(),
    );
    serde_json::Value::Object(object)
}

fn json_diff_paths(left: &serde_json::Value, right: &serde_json::Value) -> Vec<String> {
    let mut paths = Vec::new();
    collect_json_diff_paths("", left, right, &mut paths);
    paths.sort();
    paths
}

fn collect_json_diff_paths(
    path: &str,
    left: &serde_json::Value,
    right: &serde_json::Value,
    paths: &mut Vec<String>,
) {
    match (left, right) {
        (serde_json::Value::Object(left), serde_json::Value::Object(right)) => {
            let keys = left
                .keys()
                .chain(right.keys())
                .collect::<std::collections::BTreeSet<_>>();
            for key in keys {
                let next_path = format!("{path}/{key}");
                match (left.get(key), right.get(key)) {
                    (Some(left), Some(right)) => {
                        collect_json_diff_paths(&next_path, left, right, paths);
                    }
                    _ => paths.push(next_path),
                }
            }
        }
        (serde_json::Value::Array(left), serde_json::Value::Array(right)) => {
            let len = left.len().max(right.len());
            for index in 0..len {
                let next_path = format!("{path}/{index}");
                match (left.get(index), right.get(index)) {
                    (Some(left), Some(right)) => {
                        collect_json_diff_paths(&next_path, left, right, paths);
                    }
                    _ => paths.push(next_path),
                }
            }
        }
        _ if left == right => {}
        _ => paths.push(path.to_string()),
    }
}

fn assert_wire_fixture(relative_path: &str, actual: &str) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(relative_path);
    if std::env::var_os(UPDATE_ENV).is_some() {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&path, actual).unwrap();
        return;
    }

    let expected = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {relative_path}: {error}"));
    assert_eq!(
        expected, actual,
        "wire fixture mismatch for {relative_path}; set {UPDATE_ENV}=1 to regenerate"
    );
}

#[test]
fn pathology_corpus_statuses_match_certificate_sets() {
    for relative_path in pathology_fixture_paths() {
        let spec = load_pathology_spec(relative_path);
        let certificate = certify(&spec);
        let expected = expected_statuses(&certificate);
        assert!(
            !expected.is_empty(),
            "{relative_path} should produce at least one acceptable status"
        );

        let data = pathology_data(&spec);
        let formula = parse_formula(pathology_formula(&spec)).unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

        let observed_status = if expected.contains(&FitStatus::NotIdentifiable) {
            FitStatus::NotIdentifiable
        } else {
            model.fit(true).unwrap();
            model.optimizer_certificate().unwrap().status
        };

        assert!(
            status_in_expected_set(observed_status, &expected),
            "{relative_path}: observed {observed_status:?} not in certificate-derived set {expected:?}; certificate={certificate:?}"
        );

        for seed in &spec.seed_sweep {
            let mut sweep_spec = spec.clone();
            sweep_spec.seed = *seed;
            let sweep_expected = expected_statuses(&certify(&sweep_spec));
            assert_eq!(
                sweep_expected, expected,
                "{relative_path}: seed {seed} changed the certificate-derived status set"
            );
        }
    }
}

#[test]
fn compiled_artifact_matches_wire_fixture() {
    let artifact = compiler_contract_artifact();
    let json = pretty_json(&artifact);
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(
        value["schema"]["schema_name"],
        "mixedmodels.compiled_model_artifact"
    );
    assert_eq!(value["schema"]["schema_version"], 1);
    assert_eq!(
        value["design_audit"]["schema_name"],
        "mixedmodels.design_audit"
    );
    assert_eq!(
        value["design_audit"]["covariance_kernels"]["kernels"][0]["path"],
        "marginal"
    );
    assert_eq!(
        value["design_audit"]["covariance_kernels"]["missing_dependence_paths"]
            .as_array()
            .unwrap()
            .len(),
        0
    );
    assert_eq!(
        value["theta_maps"][0]["map"]["schema_name"],
        "mixedmodels.theta_map"
    );
    assert_eq!(value["covariance_parameter_traces"][0]["term_id"], "r0");
    assert_eq!(
        value["covariance_parameter_traces"][0]["lambda"]["row_basis"],
        "intercept"
    );
    assert!(value["covariance_parameter_traces"][0]["parmap_entry"].is_null());
    assert_wire_fixture(
        "tests/fixtures/compiler_contract/compiled_artifact_v1.json",
        &json,
    );
}

#[test]
fn audit_report_matches_wire_fixture() {
    let artifact = compiler_contract_artifact();
    let report = ModelAuditReport::from_artifact(&artifact);
    let json = pretty_json(&report);
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(value["schema_name"], "mixedmodels.model_audit_report");
    assert_eq!(value["schema_version"], 2);
    assert_eq!(
        value["requested_formula"],
        "y ~ 1 + x + x2 + (1 + x | subject)"
    );
    assert_eq!(value["random_term_cards"].as_array().unwrap().len(), 1);
    assert_eq!(
        value["random_term_cards"][0]["schema_name"],
        "mixedmodels.random_term_card"
    );
    assert_eq!(
        value["random_term_cards"][0]["design_support"]["status"],
        "too_rich"
    );
    assert_eq!(value["cross_card_constraints"].as_array().unwrap().len(), 0);
    assert_eq!(
        json_section_by_title(&value, "Dependence Paths")["lines"][2]["detail"],
        "none"
    );
    assert_wire_fixture(
        "tests/fixtures/compiler_contract/model_audit_report_v1.json",
        &json,
    );
}

#[test]
fn sleepstudy_artifact_matches_wire_fixture() {
    let artifact = sleepstudy_artifact();
    let json = pretty_json(&artifact);
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(
        value["requested_formula"],
        "Reaction ~ 1 + Days + (1 + Days | Subject)"
    );
    assert_eq!(
        value["design_audit"]["fixed_effect_rank"]["status"],
        "full_rank"
    );
    assert_eq!(
        value["design_audit"]["random_terms"][0]["information_budget"]["status"],
        "sufficient"
    );
    assert_eq!(value["policy_recommendations"].as_array().unwrap().len(), 0);
    assert_wire_fixture(
        "tests/fixtures/compiler_contract/sleepstudy_artifact_v1.json",
        &json,
    );
}

#[test]
fn sleepstudy_audit_report_matches_wire_fixture() {
    let artifact = sleepstudy_artifact();
    let report = ModelAuditReport::from_artifact(&artifact);
    let json = pretty_json(&report);
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(value["schema_name"], "mixedmodels.model_audit_report");
    assert_eq!(
        value["requested_formula"],
        "Reaction ~ 1 + Days + (1 + Days | Subject)"
    );
    assert_eq!(
        json_section_by_title(&value, "Random Effects")["lines"][0]["detail"],
        "group=Subject, rows=180, levels=18, obs_per_level=10..10, basis=2, covariance=full, params=3, budget=sufficient"
    );
    assert!(
        json_section_by_title(&value, "Random-Effect Information Budget")["lines"][0]["detail"]
            .as_str()
            .unwrap()
            .contains("levels/param=6.00")
    );
    assert!(
        json_section_by_title(&value, "Random-Effect Information Budget")["lines"][0]["detail"]
            .as_str()
            .unwrap()
            .contains("v0 information budget is sufficient")
    );
    assert_eq!(
        json_section_by_title(&value, "Random-Effect Information Budget")["lines"][0]["status"],
        "ok"
    );
    assert_wire_fixture(
        "tests/fixtures/compiler_contract/sleepstudy_model_audit_report_v1.json",
        &json,
    );
}

#[test]
fn sleepstudy_random_term_card_matches_wire_fixture() {
    let artifact = sleepstudy_artifact();
    let report = ModelAuditReport::from_artifact(&artifact);
    assert_eq!(report.random_term_cards.len(), 1);
    assert!(report.cross_card_constraints.is_empty());

    let card = &report.random_term_cards[0];
    assert_eq!(card.schema_name, "mixedmodels.random_term_card");
    assert_eq!(card.schema_version, 1);
    assert_eq!(card.term_id, "r0");
    assert_eq!(card.group.label(), "Subject");
    assert_eq!(card.blocks.len(), 1);
    assert_eq!(card.blocks[0].basis, vec!["intercept", "Days"]);
    assert_eq!(card.blocks[0].theta_parameters, 3);
    assert_eq!(card.design_support.group_levels, Some(18));
    assert_eq!(card.design_support.min_rows_per_group, Some(10));
    assert_eq!(card.design_support.median_rows_per_group, Some(10));
    assert_eq!(
        card.blocks[0].english,
        "`Subject` units differ in baseline and `Days` slope; the model estimates whether these are associated."
    );

    let json = pretty_json(card);
    assert_wire_fixture(
        "tests/fixtures/compiler_contract/sleepstudy_random_term_card_v1.json",
        &json,
    );
}

#[test]
fn sleepstudy_double_bar_and_split_cards_match_wire_fixture() {
    let double_bar_artifact =
        sleepstudy_artifact_for_formula("Reaction ~ Days + (1 + Days || Subject)");
    let split_artifact =
        sleepstudy_artifact_for_formula("Reaction ~ Days + (1 | Subject) + (0 + Days | Subject)");

    let double_bar_report = ModelAuditReport::from_artifact(&double_bar_artifact);
    let split_report = ModelAuditReport::from_artifact(&split_artifact);
    let double_bar = random_term_card_fixture_value(&double_bar_report);
    let split = random_term_card_fixture_value(&split_report);

    assert_eq!(
        json_diff_paths(&double_bar, &split),
        vec![
            "/cross_card_constraints/0/reason",
            "/random_term_cards/0/original_fragment",
            "/random_term_cards/1/original_fragment",
        ]
    );

    let fixture = serde_json::json!({
        "double_bar": double_bar,
        "split_blocks": split,
    });
    let json = pretty_json(&fixture);
    assert_wire_fixture(
        "tests/fixtures/compiler_contract/sleepstudy_double_bar_split_random_term_cards_v1.json",
        &json,
    );
}

#[test]
fn penicillin_artifact_matches_wire_fixture() {
    let artifact = penicillin_artifact();
    let json = pretty_json(&artifact);
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(
        value["requested_formula"],
        "diameter ~ 1 + (1 | plate) + (1 | sample)"
    );
    assert_eq!(
        value["design_audit"]["fixed_effect_rank"]["status"],
        "full_rank"
    );
    assert_eq!(
        value["semantic_model"]["random_terms"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        value["design_audit"]["random_terms"][0]["information_budget"]["status"],
        "sufficient"
    );
    assert_eq!(
        value["design_audit"]["random_terms"][1]["information_budget"]["status"],
        "sufficient"
    );
    assert_eq!(value["policy_recommendations"].as_array().unwrap().len(), 0);
    assert_wire_fixture(
        "tests/fixtures/compiler_contract/penicillin_artifact_v1.json",
        &json,
    );
}

#[test]
fn penicillin_audit_report_matches_wire_fixture() {
    let artifact = penicillin_artifact();
    let report = ModelAuditReport::from_artifact(&artifact);
    let json = pretty_json(&report);
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(value["schema_name"], "mixedmodels.model_audit_report");
    assert_eq!(
        value["requested_formula"],
        "diameter ~ 1 + (1 | plate) + (1 | sample)"
    );
    assert_eq!(
        json_section_by_title(&value, "Random Effects")["lines"][0]["detail"],
        "group=plate, rows=144, levels=24, obs_per_level=6..6, basis=1, covariance=scalar, params=1, budget=sufficient"
    );
    assert_eq!(
        json_section_by_title(&value, "Random Effects")["lines"][1]["detail"],
        "group=sample, rows=144, levels=6, obs_per_level=24..24, basis=1, covariance=scalar, params=1, budget=sufficient"
    );
    assert!(
        json_section_by_title(&value, "Random-Effect Information Budget")["lines"][1]["detail"]
            .as_str()
            .unwrap()
            .contains("levels/param=6.00")
    );
    assert_eq!(
        json_section_by_title(&value, "Random-Effect Information Budget")["lines"][0]["status"],
        "ok"
    );
    assert_wire_fixture(
        "tests/fixtures/compiler_contract/penicillin_model_audit_report_v1.json",
        &json,
    );
}

#[test]
fn confounded_fixed_random_artifact_matches_wire_fixture() {
    let artifact = confounded_fixed_random_artifact();
    let json = pretty_json(&artifact);
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(
        value["requested_formula"],
        "y ~ 1 + subject + x + (1 | subject)"
    );
    assert_eq!(
        value["design_audit"]["fixed_effect_rank"]["status"],
        "full_rank"
    );
    assert_eq!(
        value["design_audit"]["random_terms"][0]["information_budget"]["status"],
        "sufficient"
    );
    assert_eq!(value["diagnostics"][0]["code"], "fixed_random_redundant");
    assert_wire_fixture(
        "tests/fixtures/compiler_contract/confounded_fixed_random_artifact_v1.json",
        &json,
    );
}

#[test]
fn confounded_fixed_random_audit_report_matches_wire_fixture() {
    let artifact = confounded_fixed_random_artifact();
    let report = ModelAuditReport::from_artifact(&artifact);
    let json = pretty_json(&report);
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(value["schema_name"], "mixedmodels.model_audit_report");
    assert_eq!(
        value["requested_formula"],
        "y ~ 1 + subject + x + (1 | subject)"
    );
    assert_eq!(
        json_section_by_title(&value, "Diagnostics")["lines"][0]["label"],
        "fixed_random_redundant"
    );
    assert_eq!(
        json_section_by_title(&value, "Diagnostics")["lines"][0]["status"],
        "warning"
    );
    assert_wire_fixture(
        "tests/fixtures/compiler_contract/confounded_fixed_random_model_audit_report_v1.json",
        &json,
    );
}

#[test]
fn categorical_random_basis_artifact_matches_wire_fixture() {
    let artifact = categorical_random_basis_artifact();
    let json = pretty_json(&artifact);
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(
        value["requested_formula"],
        "y ~ 1 + cond + (1 + cond | subject)"
    );
    assert_eq!(
        value["semantic_model"]["random_terms"][0]["basis"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        value["theta_maps"][0]["map"]["user_basis"],
        serde_json::json!(["intercept", "cond"])
    );
    assert_eq!(
        value["theta_maps"][0]["map"]["optimizer_basis"],
        serde_json::json!(["intercept", "cond: B", "cond: C"])
    );
    assert_eq!(
        value["covariance_parameter_traces"][1]["lambda"]["row_basis"],
        "cond: B"
    );
    assert_eq!(
        value["covariance_parameter_traces"][1]["parmap_entry"]["matches_theta_map"],
        true
    );
    assert_eq!(value["design_audit"]["random_terms"][0]["basis_size"], 3);
    assert_eq!(
        value["design_audit"]["random_terms"][0]["requested_covariance_parameters"],
        6
    );
    assert_eq!(
        value["design_audit"]["random_terms"][0]["diagnostics"][0]["code"],
        "formula_canonicalized"
    );
    assert_eq!(
        value["design_audit"]["random_terms"][0]["diagnostics"][0]["payload"]["expanded_basis"],
        serde_json::json!(["intercept", "cond: B", "cond: C"])
    );
    assert_wire_fixture(
        "tests/fixtures/compiler_contract/categorical_random_basis_artifact_v1.json",
        &json,
    );
}

#[test]
fn cbpp_glmm_artifact_matches_wire_fixture() {
    let artifact = cbpp_glmm_artifact();
    let json = pretty_json(&artifact);
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(
        value["requested_formula"],
        "incidence ~ 1 + period + (1 | herd)"
    );
    assert_eq!(
        value["model_boundary"]["model_kind"],
        "generalized_linear_mixed_model"
    );
    assert_eq!(value["model_boundary"]["response_distribution"], "binomial");
    assert_eq!(value["model_boundary"]["link"], "logit");
    assert_eq!(
        value["model_boundary"]["optimizer_certificate_scope"],
        "approximated_objective"
    );
    assert_eq!(
        value["model_boundary"]["inference_availability"]["unsupported"]["reason"],
        "LMM finite-sample methods such as Satterthwaite/KR are unsupported for GLMMs in compiler v0"
    );
    assert_wire_fixture(
        "tests/fixtures/compiler_contract/cbpp_glmm_artifact_v1.json",
        &json,
    );
}

#[test]
fn cbpp_glmm_audit_report_matches_wire_fixture() {
    let artifact = cbpp_glmm_artifact();
    let report = ModelAuditReport::from_artifact(&artifact);
    let json = pretty_json(&report);
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(value["schema_name"], "mixedmodels.model_audit_report");
    assert_eq!(
        json_section_by_title(&value, "Requested Model")["lines"][1]["detail"],
        "generalized_linear_mixed_model"
    );
    assert_eq!(
        json_section_by_title(&value, "Requested Model")["lines"][2]["detail"],
        "binomial/logit"
    );
    assert_eq!(
        json_section_by_title(&value, "Optimizer")["lines"][0]["detail"],
        "model has not been fitted"
    );
    assert_eq!(
        json_section_by_title(&value, "Inference")["lines"][0]["detail"],
        "LMM finite-sample methods such as Satterthwaite/KR are unsupported for GLMMs in compiler v0"
    );
    assert_wire_fixture(
        "tests/fixtures/compiler_contract/cbpp_glmm_model_audit_report_v1.json",
        &json,
    );
}

#[test]
fn rank_mixture_artifact_matches_wire_fixture() {
    let artifact = rank_mixture_artifact();
    let json = pretty_json(&artifact);
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(value["requested_formula"], "y ~ 1 + x + (1 + x | group)");
    assert_eq!(value["effective_covariance"][0]["term_id"], "r0");
    assert_eq!(value["effective_covariance"][0]["status"], "reduced_rank");
    assert_eq!(value["effective_covariance"][0]["supported_rank"], 1);
    assert_eq!(
        value["effective_covariance"][0]["directions"][0]["loadings"][0]["basis"],
        "intercept"
    );
    assert_eq!(
        value["covariance_parameter_traces"][1]["varcorr_entries"][1]["kind"],
        "correlation"
    );
    assert!(value["covariance_parameter_traces"][1]["theta"]["value"]
        .as_f64()
        .unwrap()
        .is_finite());
    assert_eq!(
        value["optimizer_certificate"]["evidence"]["optimizer_stop"]["acceptable_stop"],
        true
    );
    assert_eq!(
        value["optimizer_certificate"]["evidence"]["parameter_space"]["n_theta"],
        3
    );
    assert_eq!(
        value["optimizer_certificate"]["evidence"]["sample_size"]["n_observations"],
        72
    );
    assert_eq!(
        value["optimizer_certificate"]["evidence"]["gradient"]["method"],
        "finite_difference"
    );
    assert_eq!(
        value["optimizer_certificate"]["verification"]["status"],
        "unstable"
    );
    assert!(
        value["optimizer_certificate"]["verification"]["runs"]
            .as_array()
            .unwrap()
            .len()
            >= 2
    );
    assert_wire_fixture(
        "tests/fixtures/compiler_contract/rank_mixture_artifact_v1.json",
        &json,
    );
}

#[test]
fn rank_mixture_audit_report_matches_wire_fixture() {
    let artifact = rank_mixture_artifact();
    let report = ModelAuditReport::from_artifact(&artifact);
    let json = pretty_json(&report);
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();

    let section = json_section_by_title(&value, "Effective Covariance");
    assert_eq!(section["lines"][0]["label"], "r0");
    assert_eq!(section["lines"][0]["status"], "info");
    assert!(section["lines"][0]["detail"]
        .as_str()
        .unwrap()
        .contains("supported direction PC1: 0.919*intercept + 0.394*x"));
    assert!(section["lines"][0]["detail"]
        .as_str()
        .unwrap()
        .contains("unsupported direction PC2: -0.394*intercept + 0.919*x"));
    let optimizer = json_section_by_title(&value, "Optimizer");
    assert_eq!(optimizer["lines"][1]["label"], "convergence interpretation");
    assert!(optimizer["lines"][1]["detail"]
        .as_str()
        .unwrap()
        .contains("unsupported directions are weakly identified"));
    assert_eq!(optimizer["lines"][4]["label"], "optimizer stop");
    assert_eq!(optimizer["lines"][4]["status"], "ok");
    assert_eq!(optimizer["lines"][7]["label"], "gradient evidence");
    assert_eq!(optimizer["lines"][7]["status"], "ok");
    assert_eq!(optimizer["lines"][10]["label"], "convergence next steps");
    assert!(optimizer["lines"][10]["detail"]
        .as_str()
        .unwrap()
        .contains("inspect Effective Covariance"));
    assert_eq!(optimizer["lines"][12]["label"], "convergence verification");
    assert_eq!(optimizer["lines"][12]["status"], "error");
    assert_wire_fixture(
        "tests/fixtures/compiler_contract/rank_mixture_model_audit_report_v1.json",
        &json,
    );
}

#[test]
fn rank_mixture_model_state_matches_wire_fixture() {
    let artifact = rank_mixture_artifact();
    let state = artifact.model_state_summary();
    let json = pretty_json(&state);
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(value["schema_name"], "mixedmodels.model_state_summary");
    assert_eq!(value["requested"]["status"], "requested");
    assert_eq!(value["supported"]["status"], "supported");
    assert_eq!(value["fitted"]["status"], "reduced");
    assert_eq!(value["changes"][0]["status"], "applied");
    assert_eq!(value["changes"][0]["trigger"], "certificate_time_boundary");
    assert_eq!(
        value["fitted"]["random_terms"][0]["supported_rank"],
        serde_json::json!(1)
    );
    assert_wire_fixture(
        "tests/fixtures/compiler_contract/rank_mixture_model_state_v1.json",
        &json,
    );
}

#[test]
fn design_compiled_artifact_matches_wire_fixture() {
    let artifact = design_compiled_artifact();
    let json = pretty_json(&artifact);
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(value["effective_formula"], "y ~ 1 + x + (1 + x || group)");
    assert_eq!(
        value["reproducibility"]["fit_intent"],
        "confirmatory_design_compiled"
    );
    assert_eq!(value["theta_maps"].as_array().unwrap().len(), 2);
    assert_eq!(value["theta_maps"][0]["family"], "scalar");
    assert_eq!(value["theta_maps"][1]["family"], "scalar");
    assert_eq!(
        value["effective_semantic_model"]["random_terms"][0]["block_group"],
        "bg0"
    );
    assert_eq!(
        value["effective_semantic_model"]["random_terms"][1]["block_group"],
        "bg0"
    );
    assert_eq!(value["reductions"][0]["trigger"], "design_time");
    assert_wire_fixture(
        "tests/fixtures/compiler_contract/design_compiled_artifact_v1.json",
        &json,
    );
}

#[test]
fn design_compiled_model_state_matches_wire_fixture() {
    let artifact = design_compiled_artifact();
    let state = artifact.model_state_summary();
    let json = pretty_json(&state);
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(value["supported"]["status"], "reduced");
    assert_eq!(
        value["supported"]["formula"],
        "y ~ 1 + x + (1 + x || group)"
    );
    assert_eq!(value["changes"][0]["status"], "applied");
    assert_eq!(value["changes"][0]["trigger"], "design_time");
    assert_wire_fixture(
        "tests/fixtures/compiler_contract/design_compiled_model_state_v1.json",
        &json,
    );
}

#[test]
fn pedagogical_diagnostics_artifact_matches_wire_fixture() {
    let artifact = pedagogical_diagnostics_artifact();
    let json = pretty_json(&artifact);
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert!(json.contains("\"code\": \"scope_note\""));
    assert!(json.contains("\"code\": \"support_note\""));
    assert!(json.contains("\"code\": \"syntax_expansion\""));
    assert!(json.contains("\"code\": \"covariance_assumption\""));
    assert!(json.contains("\"code\": \"structural_refusal\""));
    assert!(json.contains("\"expansion_kind\": \"nested\""));
    assert!(json.contains("\"reason\": \"double_bar_syntax\""));
    assert!(json.contains("\"reason\": \"separate_random_effect_blocks\""));
    assert_eq!(
        value["design_audit"]["random_terms"][0]["group"]["median_obs_per_level"],
        serde_json::json!(4)
    );
    assert_wire_fixture(
        "tests/fixtures/compiler_contract/pedagogical_diagnostics_artifact_v1.json",
        &json,
    );
}

#[test]
fn singular_prefit_audit_report_matches_wire_fixture() {
    let artifact = singular_prefit_artifact();
    let report = ModelAuditReport::from_artifact(&artifact);
    let json = pretty_json(&report);
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(
        value["requested_formula"],
        "y ~ 1 + A + B + C + A:B + A:C + B:C + A:B:C + (1 + A + B + C + A:B + A:C + B:C + A:B:C | group)"
    );
    let random = &json_section_by_title(&value, "Random Effects")["lines"][0];
    assert_eq!(random["status"], "warning");
    assert!(random["detail"]
        .as_str()
        .unwrap()
        .contains("rows=150, levels=10, obs_per_level=15..15"));
    assert!(random["detail"]
        .as_str()
        .unwrap()
        .contains("basis=8, covariance=full, params=36, budget=too_rich"));
    assert!(random["detail"].as_str().unwrap().contains("threshold 180"));

    let budget = &json_section_by_title(&value, "Random-Effect Information Budget")["lines"][0];
    assert_eq!(budget["status"], "warning");
    assert!(budget["detail"]
        .as_str()
        .unwrap()
        .contains("levels/param=0.28"));
    assert!(budget["detail"]
        .as_str()
        .unwrap()
        .contains("total rows can be misleading"));
    assert!(budget["detail"]
        .as_str()
        .unwrap()
        .contains("diagonal/reduced-rank covariance"));

    let policy = &json_section_by_title(&value, "Policy Recommendations")["lines"][0];
    assert_eq!(policy["status"], "warning");
    assert!(policy["detail"]
        .as_str()
        .unwrap()
        .contains("refuse_random_term_distribution"));
    assert!(policy["detail"]
        .as_str()
        .unwrap()
        .contains("variance-direction threshold 17"));
    assert!(policy["detail"]
        .as_str()
        .unwrap()
        .contains("confirmatory fixed-effect p-values should be withheld"));

    assert_wire_fixture(
        "tests/fixtures/compiler_contract/singular_prefit_model_audit_report_v1.json",
        &json,
    );
}

fn intercept_dominant_reduced_rank_artifact() -> CompiledModelArtifact {
    // Truth: sigma^2_intercept = 4.0, sigma^2_slope = 0.04, rho = 1.0
    // -> Sigma_truth = [[4.0, 0.4], [0.4, 0.04]] (rank 1, dominant axis = intercept).
    let mut data = DataFrame::new();
    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut group = Vec::new();
    for g in 0..30_usize {
        let intercept_shift = (g as f64 - 14.5) * 0.4;
        let slope_shift = intercept_shift * 0.1;
        for k in 0..6_usize {
            let x_val = (k as f64 - 2.5) * 0.4;
            y.push(10.0 + intercept_shift + (2.0 + slope_shift) * x_val);
            x.push(x_val);
            group.push(format!("g{}", g + 1));
        }
    }
    data.add_numeric("y", y);
    data.add_numeric("x", x);
    data.add_categorical("group", group);

    let formula = parse_formula("y ~ x + (1 + x | group)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(true).unwrap();
    model.compiler_artifact().clone()
}

#[test]
fn intercept_dominant_reduced_rank_emits_interpretable_submodel() {
    let artifact = intercept_dominant_reduced_rank_artifact();
    let json = pretty_json(&artifact);
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(value["requested_formula"], "y ~ 1 + x + (1 + x | group)");
    assert!(value["effective_formula"].is_null());
    assert_eq!(value["effective_covariance"][0]["status"], "reduced_rank");
    let suggestion = &value["effective_covariance"][0]["interpretable_submodel"];
    assert!(
        !suggestion.is_null(),
        "expected interpretable_submodel to be populated, got: {}",
        value["effective_covariance"][0]
    );
    assert_eq!(suggestion["suggested_formula"], "(1 | group)");
    assert_eq!(suggestion["loadings_dominant"][0]["basis"], "intercept");
    assert!(
        suggestion["loadings_dominant"][0]["loading"]
            .as_f64()
            .unwrap()
            .abs()
            >= 0.95,
        "dominant loading must clear the 0.95 threshold"
    );
    let gap = suggestion["objective_gap"].as_f64().unwrap();
    assert!(
        gap >= 0.0 && gap.is_finite(),
        "objective_gap must be a non-negative finite number, got {gap}"
    );
    assert_eq!(suggestion["within_tolerance"], serde_json::json!(true));
    assert_eq!(
        value["diagnostics"][0]["payload"]["interpretable_submodel"]["suggested_formula"],
        "(1 | group)"
    );
    assert_wire_fixture(
        "tests/fixtures/compiler_contract/intercept_dominant_reduced_rank_artifact_v1.json",
        &json,
    );
}

#[test]
fn intercept_dominant_reduced_rank_audit_report_matches_wire_fixture() {
    let artifact = intercept_dominant_reduced_rank_artifact();
    let report = ModelAuditReport::from_artifact(&artifact);
    let json = pretty_json(&report);
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();

    let section = json_section_by_title(&value, "Effective Covariance");
    let detail = section["lines"][0]["detail"].as_str().unwrap();
    assert!(detail.contains("interpretable submodel suggestion: (1 | group)"));
    assert!(detail.contains("dominant loadings="));
    assert!(detail.contains("objective gap="));
    assert!(detail.contains("within tolerance=true"));
    assert_wire_fixture(
        "tests/fixtures/compiler_contract/intercept_dominant_reduced_rank_model_audit_report_v1.json",
        &json,
    );
}

#[test]
fn rank_mixture_does_not_emit_interpretable_submodel() {
    let artifact = rank_mixture_artifact();
    let json = pretty_json(&artifact);
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(value["effective_covariance"][0]["status"], "reduced_rank");
    assert!(
        value["effective_covariance"][0]
            .get("interpretable_submodel")
            .map(|v| v.is_null())
            .unwrap_or(true),
        "rank_mixture's mixed PC1 must NOT trigger an interpretable_submodel suggestion"
    );
}

#[test]
fn full_rank_terms_never_carry_interpretable_submodel() {
    let (data, meta) = datasets::load("sleepstudy").unwrap();
    let formula = parse_formula(&meta.fits[0].formula).unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(true).unwrap();
    let artifact = model.compiler_artifact().clone();
    let json = pretty_json(&artifact);
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();

    let entries = value["effective_covariance"]
        .as_array()
        .expect("effective_covariance must be present after fit");
    assert!(
        !entries.is_empty(),
        "sleepstudy fit should populate effective_covariance"
    );
    for entry in entries {
        if entry["status"] == "full_rank" {
            assert!(
                entry
                    .get("interpretable_submodel")
                    .map(|v| v.is_null())
                    .unwrap_or(true),
                "full-rank covariance summaries must not carry interpretable_submodel"
            );
        }
    }
}
