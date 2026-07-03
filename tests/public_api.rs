use std::fs;
use std::path::Path;
use std::process::Command;

use mixeff_rs::error::MixedModelError;
use mixeff_rs::model::{
    parametricbootstrap, ActiveFaceRefit, BootstrapFailedRefitPolicy, BootstrapInterval,
    BootstrapIntervalMethod, BootstrapQuantile, BootstrapRefitOptions, BootstrapReplicate,
    BootstrapRunMetadata, BootstrapRunPayload, BootstrapSeedRecord, BootstrapTarget,
    BootstrapTargetKind, CategoricalColumn, Column, DataFrame, Family, FitToleranceOverrides,
    FixedEffectNullBootstrapTarget, FixedEffectNullCovariancePolicy, GeneralizedLinearMixedModel,
    GlmmFitOptions, GlmmPredictionScale, LinearMixedModel, LinkFunction, MixedModelBootstrap,
    MixedModelFit, NewReLevels, OptimizerChoice, OptimizerControl, PredictionVarianceMethod,
    PredictionVariancePayload, PredictionVarianceRow, PredictionVarianceStatus,
    RandomEffectTermInfo, TrustBqSampleReuse, TrustBqStartLadder, BOOTSTRAP_RUN_SCHEMA,
    BOOTSTRAP_RUN_SCHEMA_VERSION,
};
use mixeff_rs::stats::{
    assess_model_comparison_sequence, coeftable_to_markdown, profile, profile_beta, profile_betas,
    profile_confint_payload, profile_sigma, profile_theta, profile_theta_scalar,
    restore_replicates, restorereplicates, save_replicates, savereplicates, shortest_cov_int,
    BlockDescription, BoundaryLikelihoodRatioTest, BoundaryLrtMixtureComponent, BoundaryLrtStatus,
    CoefTable, CoefTablePValuePolicy, ConfintRow, FixedEffectComparison, LikelihoodRatioTest,
    LinearModelFit, MixedModelProfile, ModelComparisonAlternative, ModelComparisonAssessment,
    ModelComparisonClass, ModelSummary, ModelSummaryRow, ProfileLikelihoodCiPayload,
    ProfileLikelihoodCiRow, ProfileRow, RandomEffectComparison, VarCorr, VarCorrComponent,
    BOUNDARY_LRT_SCHEMA, BOUNDARY_LRT_SCHEMA_VERSION, FIT_SUMMARY_SCHEMA,
    FIT_SUMMARY_SCHEMA_VERSION, PROFILE_LIKELIHOOD_CI_SCHEMA, PROFILE_LIKELIHOOD_CI_SCHEMA_VERSION,
};
use mixeff_rs::types::MatrixBlock;

fn source(relative: &str) -> String {
    fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join(relative))
        .unwrap_or_else(|error| panic!("failed to read {relative}: {error}"))
}

fn toml_string(value: &Path) -> String {
    format!("{:?}", value.to_string_lossy())
}

#[test]
fn model_and_stats_barrels_do_not_use_glob_reexports() {
    for path in ["src/model/mod.rs", "src/stats/mod.rs"] {
        let text = source(path);
        assert!(
            !text.contains("pub use ") || !text.contains("::*"),
            "{path} must use explicit pub use lists"
        );
    }
}

#[test]
fn unstable_storage_helpers_are_not_top_level_reexports() {
    let model_mod = source("src/model/mod.rs");
    let stats_mod = source("src/stats/mod.rs");

    assert!(
        !model_mod.contains("MatrixBlock"),
        "MatrixBlock should remain under types::matrix_block, not model::*"
    );
    assert!(
        !stats_mod.contains("NaturalCubicSpline"),
        "NaturalCubicSpline should remain under stats::spline, not stats::*"
    );
}

#[test]
fn downstream_crate_cannot_import_internal_barrel_items() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let temp_root = std::env::temp_dir().join(format!(
        "mixeff-rs-public-api-negative-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&temp_root);
    fs::create_dir_all(temp_root.join("src")).unwrap();
    fs::write(
        temp_root.join("Cargo.toml"),
        format!(
            r#"[package]
name = "mixeff_rs_public_api_negative"
version = "0.0.0"
edition = "2021"

[dependencies]
mixeff-rs = {{ path = {} }}
"#,
            toml_string(manifest_dir)
        ),
    )
    .unwrap();
    fs::write(
        temp_root.join("src/lib.rs"),
        r#"use mixeff_rs::model::MatrixBlock;
use mixeff_rs::stats::NaturalCubicSpline;

pub fn touch_internal_api() {
    let _ = core::mem::size_of::<MatrixBlock>();
    let _ = NaturalCubicSpline::fit(&[0.0, 1.0], &[0.0, 1.0]);
}
"#,
    )
    .unwrap();

    let output = Command::new(env!("CARGO"))
        .arg("check")
        .arg("--quiet")
        .arg("--manifest-path")
        .arg(temp_root.join("Cargo.toml"))
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);

    let _ = fs::remove_dir_all(&temp_root);

    assert!(
        !output.status.success(),
        "downstream crate unexpectedly imported internal barrel items"
    );
    assert!(
        stderr.contains("MatrixBlock"),
        "expected MatrixBlock import failure, got:\n{stderr}"
    );
    assert!(
        stderr.contains("NaturalCubicSpline"),
        "expected NaturalCubicSpline import failure, got:\n{stderr}"
    );
}

#[test]
fn downstream_crate_cannot_touch_sealed_model_internals() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let temp_root = std::env::temp_dir().join(format!(
        "mixeff-rs-public-api-sealed-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&temp_root);
    fs::create_dir_all(temp_root.join("src")).unwrap();
    fs::write(
        temp_root.join("Cargo.toml"),
        format!(
            r#"[package]
name = "mixeff_rs_public_api_sealed"
version = "0.0.0"
edition = "2021"

[dependencies]
mixeff-rs = {{ path = {} }}
"#,
            toml_string(manifest_dir)
        ),
    )
    .unwrap();
    fs::write(
        temp_root.join("src/lib.rs"),
        r#"use mixeff_rs::model::{CategoricalColumn, GeneralizedLinearMixedModel, LinearMixedModel};

pub fn touch_lmm(model: &LinearMixedModel) {
    let _ = &model.feterm;
    let _ = &model.parmap;
    let _ = &model.a_blocks;
    let _ = &model.compiler_artifact;
}

pub fn touch_glmm(model: &GeneralizedLinearMixedModel) {
    let _ = &model.lmm;
    let _ = &model.theta;
    let _ = &model.b;
    let _ = &model.mu;
}

pub fn construct_non_exhaustive_column() -> CategoricalColumn {
    CategoricalColumn {
        levels: Vec::new(),
        refs: Vec::new(),
        values: Vec::new(),
        contrast: None,
    }
}
"#,
    )
    .unwrap();

    let output = Command::new(env!("CARGO"))
        .arg("check")
        .arg("--quiet")
        .arg("--manifest-path")
        .arg(temp_root.join("Cargo.toml"))
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);

    let _ = fs::remove_dir_all(&temp_root);

    assert!(
        !output.status.success(),
        "downstream crate unexpectedly touched sealed model internals"
    );
    for needle in [
        "field `feterm`",
        "field `parmap`",
        "field `a_blocks`",
        "field `compiler_artifact`",
        "field `lmm`",
        "field `theta`",
        "field `b`",
        "field `mu`",
        "non-exhaustive",
    ] {
        assert!(
            stderr.contains(needle),
            "expected `{needle}` in sealed API failure, got:\n{stderr}"
        );
    }
}

#[test]
fn intended_model_barrel_exports_compile_for_downstream_users() {
    fn assert_type<T>() {}
    fn assert_trait<T: ?Sized>() {}

    assert_type::<CategoricalColumn>();
    assert_type::<Column>();
    assert_type::<DataFrame>();
    assert_type::<GeneralizedLinearMixedModel>();
    assert_type::<LinearMixedModel>();
    assert_type::<GlmmFitOptions>();
    assert_type::<GlmmPredictionScale>();
    assert_type::<OptimizerChoice>();
    assert_type::<OptimizerControl>();
    assert_type::<ActiveFaceRefit>();
    assert_type::<TrustBqSampleReuse>();
    assert_type::<TrustBqStartLadder>();
    assert_type::<FitToleranceOverrides>();
    assert_type::<NewReLevels>();
    assert_type::<PredictionVarianceMethod>();
    assert_type::<PredictionVariancePayload>();
    assert_type::<PredictionVarianceRow>();
    assert_type::<PredictionVarianceStatus>();
    assert_type::<BootstrapReplicate>();
    assert_type::<MixedModelBootstrap>();
    assert_type::<BootstrapIntervalMethod>();
    assert_type::<BootstrapQuantile>();
    assert_type::<BootstrapInterval>();
    assert_type::<BootstrapTargetKind>();
    assert_type::<BootstrapTarget>();
    assert_type::<BootstrapFailedRefitPolicy>();
    assert_type::<BootstrapSeedRecord>();
    assert_type::<BootstrapRefitOptions>();
    assert_type::<BootstrapRunMetadata>();
    assert_type::<BootstrapRunPayload>();
    assert_type::<FixedEffectNullCovariancePolicy>();
    assert_type::<FixedEffectNullBootstrapTarget>();
    assert_type::<RandomEffectTermInfo>();
    assert_type::<Family>();
    assert_type::<LinkFunction>();
    assert_trait::<dyn MixedModelFit>();

    let _ = LinearMixedModel::fixed_effect_fitted
        as fn(&LinearMixedModel) -> mixeff_rs::nalgebra::DVector<f64>;
    let _ = GeneralizedLinearMixedModel::predict_new
        as fn(
            &GeneralizedLinearMixedModel,
            &DataFrame,
            GlmmPredictionScale,
            NewReLevels,
        ) -> mixeff_rs::error::Result<Vec<Option<f64>>>;
    let _ = GeneralizedLinearMixedModel::profile_theta
        as fn(
            &mut GeneralizedLinearMixedModel,
            usize,
            f64,
        ) -> mixeff_rs::error::Result<mixeff_rs::stats::MixedModelProfile>;
    let _ = LinearMixedModel::predict_new_variance
        as fn(
            &LinearMixedModel,
            &DataFrame,
            NewReLevels,
        ) -> mixeff_rs::error::Result<PredictionVariancePayload>;
    let _ = LinearMixedModel::predict_new_variance_with_level
        as fn(
            &LinearMixedModel,
            &DataFrame,
            NewReLevels,
            f64,
        ) -> mixeff_rs::error::Result<PredictionVariancePayload>;
    let _ = GeneralizedLinearMixedModel::predict_new_variance
        as fn(
            &GeneralizedLinearMixedModel,
            &DataFrame,
            GlmmPredictionScale,
            NewReLevels,
        ) -> mixeff_rs::error::Result<PredictionVariancePayload>;
    let _ = GeneralizedLinearMixedModel::predict_new_variance_with_level
        as fn(
            &GeneralizedLinearMixedModel,
            &DataFrame,
            GlmmPredictionScale,
            NewReLevels,
            f64,
        ) -> mixeff_rs::error::Result<PredictionVariancePayload>;
    let _ = BOOTSTRAP_RUN_SCHEMA;
    let _ = BOOTSTRAP_RUN_SCHEMA_VERSION;
    let _ = parametricbootstrap::<rand::rngs::StdRng>;
}

#[test]
fn intended_types_exports_compile_for_downstream_users() {
    fn assert_type<T>() {}

    assert_type::<MatrixBlock>();
}

#[test]
fn stable_result_and_error_types_are_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<MixedModelError>();
    assert_send_sync::<LinearMixedModel>();
    assert_send_sync::<GeneralizedLinearMixedModel>();
    assert_send_sync::<CoefTable>();
    assert_send_sync::<ModelSummary>();
    assert_send_sync::<ModelSummaryRow>();
    assert_send_sync::<VarCorr>();
    assert_send_sync::<VarCorrComponent>();
}

#[test]
fn intended_stats_barrel_exports_compile_for_downstream_users() {
    fn assert_type<T>() {}

    assert_type::<BlockDescription>();
    assert_type::<BoundaryLikelihoodRatioTest>();
    assert_type::<BoundaryLrtMixtureComponent>();
    assert_type::<BoundaryLrtStatus>();
    assert_type::<CoefTable>();
    assert_type::<CoefTablePValuePolicy>();
    assert_type::<FixedEffectComparison>();
    assert_type::<LikelihoodRatioTest>();
    assert_type::<LinearModelFit>();
    assert_type::<mixeff_rs::stats::FitSummaryPayload>();
    assert_type::<ModelComparisonAlternative>();
    assert_type::<ModelComparisonAssessment>();
    assert_type::<ModelComparisonClass>();
    assert_type::<ModelSummary>();
    assert_type::<ModelSummaryRow>();
    assert_type::<ProfileRow>();
    assert_type::<RandomEffectComparison>();
    assert_type::<MixedModelProfile>();
    assert_type::<ConfintRow>();
    assert_type::<ProfileLikelihoodCiPayload>();
    assert_type::<ProfileLikelihoodCiRow>();
    assert_type::<VarCorr>();
    assert_type::<VarCorrComponent>();

    let _ = coeftable_to_markdown as fn(&CoefTable) -> String;
    let _ = shortest_cov_int as fn(&mut [f64], f64) -> (f64, f64);
    let _ = assess_model_comparison_sequence;
    let _ = BOUNDARY_LRT_SCHEMA;
    let _ = BOUNDARY_LRT_SCHEMA_VERSION;
    let _ = FIT_SUMMARY_SCHEMA;
    let _ = FIT_SUMMARY_SCHEMA_VERSION;
    let _ = profile;
    let _ = profile_beta;
    let _ = profile_betas;
    let _ = profile_confint_payload;
    let _ = profile_sigma;
    let _ = profile_theta;
    let _ = profile_theta_scalar;
    let _ = PROFILE_LIKELIHOOD_CI_SCHEMA;
    let _ = PROFILE_LIKELIHOOD_CI_SCHEMA_VERSION;
    let _ = save_replicates::<Vec<u8>>;
    let _ = savereplicates::<Vec<u8>>;
    let _ = restore_replicates::<&[u8]>;
    let _ = restorereplicates::<&[u8]>;
}
