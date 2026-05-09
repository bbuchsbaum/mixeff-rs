use std::fs;
use std::path::Path;
use std::process::Command;

use mixedmodels::model::{
    parametricbootstrap, BatchOptimizerControl, BatchOptions, BatchThetaGrouping, BatchWarmStart, BootstrapFailedRefitPolicy, BootstrapInterval, BootstrapIntervalMethod,
    BootstrapQuantile, BootstrapRefitOptions, BootstrapReplicate, BootstrapRunMetadata,
    BootstrapRunPayload, BootstrapSeedRecord, BootstrapTarget, BootstrapTargetKind,
    CategoricalColumn, Column, ConvergenceVerificationOptions, DataFrame, Family,
    FixedEffectNullBootstrapTarget, FixedEffectNullCovariancePolicy, GeneralizedLinearMixedModel,
    KenwardRogerAdjustedVcov, KenwardRogerLbDdf, KenwardRogerSigmaG, LinearMixedModel, LinearMixedModelBatch,
    LinkFunction, MixedModelBootstrap, MixedModelFit, ModelDims, NewReLevels, RandomEffectTermInfo, ResponseBatchFit, ResponseBatchMode,
    ResponseColumnDiagnostic, ResponseDiagnosticReason, ResponseFitStatus, ResponseMatrixProfile,
    ThetaBatch, VcovVarparEstimate, BOOTSTRAP_RUN_SCHEMA, BOOTSTRAP_RUN_SCHEMA_VERSION,
};
use mixedmodels::stats::{
    assess_model_comparison_sequence, coeftable_to_markdown, profile, profile_beta, profile_betas,
    profile_sigma, profile_theta, profile_theta_scalar, restore_replicates, restorereplicates,
    save_replicates, savereplicates, shortest_cov_int, BlockDescription, CoefTable,
    CoefTablePValuePolicy, ConfintRow, FixedEffectComparison, LikelihoodRatioTest, LinearModelFit,
    MixedModelProfile, ModelComparisonAlternative, ModelComparisonAssessment, ModelComparisonClass,
    ModelSummary, ModelSummaryRow, ProfileRow, RandomEffectComparison, VarCorr, VarCorrComponent,
};
use mixedmodels::types::MatrixBlock;

fn source(relative: &str) -> String {
    fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join(relative))
        .unwrap_or_else(|error| panic!("failed to read {relative}: {error}"))
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
        "mixedmodels-public-api-negative-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&temp_root);
    fs::create_dir_all(temp_root.join("src")).unwrap();
    fs::write(
        temp_root.join("Cargo.toml"),
        format!(
            r#"[package]
name = "mixedmodels_public_api_negative"
version = "0.0.0"
edition = "2021"

[dependencies]
mixedmodels = {{ path = "{}" }}
"#,
            manifest_dir.display()
        ),
    )
    .unwrap();
    fs::write(
        temp_root.join("src/lib.rs"),
        r#"use mixedmodels::model::MatrixBlock;
use mixedmodels::stats::NaturalCubicSpline;

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
fn intended_model_barrel_exports_compile_for_downstream_users() {
    fn assert_type<T>() {}
    fn assert_trait<T: ?Sized>() {}

    assert_type::<CategoricalColumn>();
    assert_type::<Column>();
    assert_type::<DataFrame>();
    assert_type::<GeneralizedLinearMixedModel>();
    assert_type::<LinearMixedModel>();
    assert_type::<LinearMixedModelBatch>();
    assert_type::<ModelDims>();
    assert_type::<NewReLevels>();
    assert_type::<ResponseMatrixProfile>();
    assert_type::<BatchOptions>();
    assert_type::<BatchOptimizerControl>();
    assert_type::<BatchThetaGrouping>();
    assert_type::<BatchWarmStart>();
    assert_type::<ResponseBatchFit>();
    assert_type::<ResponseBatchMode>();
    assert_type::<ResponseColumnDiagnostic>();
    assert_type::<ResponseDiagnosticReason>();
    assert_type::<ResponseFitStatus>();
    assert_type::<ThetaBatch>();
    assert_type::<VcovVarparEstimate>();
    assert_type::<KenwardRogerSigmaG>();
    assert_type::<KenwardRogerAdjustedVcov>();
    assert_type::<KenwardRogerLbDdf>();
    assert_type::<ConvergenceVerificationOptions>();
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
fn intended_stats_barrel_exports_compile_for_downstream_users() {
    fn assert_type<T>() {}

    assert_type::<BlockDescription>();
    assert_type::<CoefTable>();
    assert_type::<CoefTablePValuePolicy>();
    assert_type::<FixedEffectComparison>();
    assert_type::<LikelihoodRatioTest>();
    assert_type::<LinearModelFit>();
    assert_type::<ModelComparisonAlternative>();
    assert_type::<ModelComparisonAssessment>();
    assert_type::<ModelComparisonClass>();
    assert_type::<ModelSummary>();
    assert_type::<ModelSummaryRow>();
    assert_type::<ProfileRow>();
    assert_type::<RandomEffectComparison>();
    assert_type::<MixedModelProfile>();
    assert_type::<ConfintRow>();
    assert_type::<VarCorr>();
    assert_type::<VarCorrComponent>();

    let _ = coeftable_to_markdown as fn(&CoefTable) -> String;
    let _ = shortest_cov_int as fn(&mut [f64], f64) -> (f64, f64);
    let _ = assess_model_comparison_sequence;
    let _ = profile;
    let _ = profile_beta;
    let _ = profile_betas;
    let _ = profile_sigma;
    let _ = profile_theta;
    let _ = profile_theta_scalar;
    let _ = save_replicates::<Vec<u8>>;
    let _ = savereplicates::<Vec<u8>>;
    let _ = restore_replicates::<&[u8]>;
    let _ = restorereplicates::<&[u8]>;
}
