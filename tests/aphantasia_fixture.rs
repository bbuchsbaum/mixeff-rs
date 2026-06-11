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
    let mut count = 0usize;
    for record in rdr.records() {
        record.unwrap_or_else(|error| panic!("read {path:?}: {error}"));
        count += 1;
    }
    count
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

#[cfg(all(not(feature = "nlopt"), feature = "unstable-internals"))]
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
fn max_abs_fixef_drift(model: &GeneralizedLinearMixedModel, reference: &Value) -> f64 {
    let expected = reference["fixef"]
        .as_object()
        .expect("reference fixef must be an object");
    let ref_beta = |name: &str| {
        expected
            .get(name)
            .and_then(Value::as_f64)
            .unwrap_or_else(|| panic!("reference fixef `{name}` missing or non-numeric"))
    };
    let b0 = ref_beta("(Intercept)");
    let b_group = ref_beta("groupaphant");
    let b_mask = ref_beta("maskmasked");
    let b_soa = ref_beta("soa_s");
    let b_block = ref_beta("block2");
    let b_group_mask = ref_beta("groupaphant:maskmasked");
    let b_group_soa = ref_beta("groupaphant:soa_s");
    let b_mask_soa = ref_beta("maskmasked:soa_s");
    let b_group_mask_soa = ref_beta("groupaphant:maskmasked:soa_s");

    let expected_native = [
        b0 + b_group + b_mask + b_group_mask,
        -b_group - b_group_mask,
        -b_mask - b_group_mask,
        b_soa + b_group_soa + b_mask_soa + b_group_mask_soa,
        b_group_mask,
        -b_group_soa - b_group_mask_soa,
        -b_mask_soa - b_group_mask_soa,
        b_group_mask_soa,
        b_block,
    ];

    model
        .coef()
        .iter()
        .zip(expected_native.iter())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f64, f64::max)
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
    let fixtures_excluded =
        cargo_toml.contains("\"tests/fixtures/\"") || cargo_toml.contains("\"tests/\"");
    assert!(
        fixtures_excluded,
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
fn combined_glmm_data() -> DataFrame {
    load_prepared_case(
        "combined",
        &["correct", "soa_s"],
        &["participant", "item", "group", "mask", "block", "stimtype"],
    )
}

/// Regression for bd-01KTQJFZNF5H034B5WKWKJQRDF.
///
/// The joint Laplace route on the COMBINED model used to declare FTOL after
/// ~99 evaluations at the profiled start (the trust_bq stagnation guard
/// tripped before any descent) and discard the joint candidate for the
/// labelled fast-PIRLS fallback. With the descent-gated stagnation stop the
/// optimizer must now actually descend from the profiled start and return a
/// joint result.
///
/// Note the lme4 reference for this formula is NOT a parity target here:
/// lme4's `||` keeps the within-factor correlation of `mask` (a full 2x2
/// block, 6 theta), while the native `||` drops it (4 theta), and lme4's
/// fitted combined optimum needs that off-diagonal. The native-family joint
/// optimum sits ~1.9 logLik above lme4's; the explicit expansion
/// `(1|p) + (0+mask|p) + (0+soa_s|p) + (1|item)` reproduces lme4's family
/// and its optimum (see `probe_aphantasia_combined`).
#[cfg(all(not(feature = "nlopt"), feature = "unstable-internals"))]
#[test]
#[ignore = "real aphantasia combined GLMM takes ~15 minutes; run intentionally for optimizer diagnostics"]
fn combined_native_joint_descends_from_profiled_start_without_fallback() {
    let reference = reference_json();
    let combined_ref = reference_model(&reference, "combined");
    let reference_loglik = combined_ref["logLik"].as_f64().unwrap();
    let data = combined_glmm_data();
    let formula = parse_formula(
        "correct ~ group * mask * soa_s * stimtype + block + (1 + mask + soa_s || participant) + (1 | item)",
    )
    .unwrap();

    let mut profiled =
        GeneralizedLinearMixedModel::new(formula.clone(), &data, Family::Bernoulli, None).unwrap();
    profiled.fit_with_options(true, 1, false).unwrap();
    let profiled_loglik = profiled.loglikelihood();

    let mut joint =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();
    joint.fit_with_options(false, 1, false).unwrap();
    let joint_loglik = joint.loglikelihood();
    let return_value = &joint.lmm().optsum().return_value;

    // Pre-fix: FTOL at ~99 evals, zero descent, fallback to fast-PIRLS.
    assert!(
        return_value.starts_with("JOINT_LAPLACE:"),
        "combined joint fit fell back instead of returning a joint result: {return_value:?}"
    );
    assert!(
        joint.lmm().optsum().feval > 300,
        "joint optimizer stopped suspiciously early: feval={}",
        joint.lmm().optsum().feval
    );
    assert!(
        joint_loglik > profiled_loglik + 1.0,
        "joint Laplace should materially improve the profiled start: joint={joint_loglik:.4} profiled={profiled_loglik:.4}"
    );
    // Native-family optimum (zero-correlation mask dummy) sits ~1.9 logLik
    // above lme4's 6-theta optimum; allow headroom but catch regressions back
    // toward the profiled start (~5.0 above).
    let gap = (joint_loglik - reference_loglik).abs();
    assert!(
        gap < 2.5,
        "combined joint logLik gap to lme4 should stay below 2.5 (native-family optimum ~1.9): {gap:.4}"
    );
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
    let profiled_fixef_drift = max_abs_fixef_drift(&profiled, intact_ref);

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
        let joint_fixef_drift = max_abs_fixef_drift(&joint, intact_ref);
        eprintln!(
            "aphantasia intact budget={max_feval} joint_feval={} final_feval={} joint_fmin={} loglik={} gap={} profiled_gap={} fixef_drift={} profiled_fixef_drift={} coef_names={:?} coef={:?} status={}",
            joint_feval,
            joint.lmm().optsum().feval,
            joint_fmin,
            joint.loglikelihood(),
            joint_gap,
            profiled_gap,
            joint_fixef_drift,
            profiled_fixef_drift,
            joint.coef_names(),
            joint.coef(),
            joint.lmm().optsum().return_value
        );
        assert!(joint_feval <= max_feval);
        assert!(joint_gap.is_finite());
        assert!(joint_fixef_drift.is_finite());
        assert!(
            joint_gap <= profiled_gap,
            "joint budget path should improve the lme4 logLik gap: joint={joint_gap}, profiled={profiled_gap}"
        );
        assert!(
            joint_fixef_drift <= profiled_fixef_drift,
            "joint budget path should improve fixed-effect drift: joint={joint_fixef_drift}, profiled={profiled_fixef_drift}"
        );
        if max_feval >= 40 {
            assert!(
                joint_gap < 0.5,
                "budget >= 40 should close most of the intact logLik gap; got {joint_gap}"
            );
            assert!(
                joint_fixef_drift < 0.1,
                "budget >= 40 should reduce intact max fixed-effect drift below 0.1; got {joint_fixef_drift}"
            );
        }
    }
}
