#![cfg(not(feature = "nlopt"))]
#![cfg(feature = "unstable-internals")]
//! Regression for bd-01KTQ7T8DPAGA1GS93R71D6QKF.
//!
//! Correlated-slope crossed binomial joint Laplace fits that match glmer to
//! numerical-noise levels used to surface fit_status=not_optimized: the
//! certification gradient was probed at a finite-difference step where the
//! inner-PIRLS deviance error dominates flat directions, and a genuine small
//! residual gradient at the derivative-free stop was never polished away. The
//! certification path now escalates noisy probe components to trusted steps
//! and finishes assessed stationarity failures with the damped-Newton polish,
//! so a glmer-equivalent correlated-slope optimum must certify.

use std::fs;
use std::path::PathBuf;

use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::traits::MixedModelFit;
use mixeff_rs::model::{DataFrame, Family, GeneralizedLinearMixedModel, LinkFunction};
use serde_json::Value;

fn fixture() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/regression/glmm_correlated_slope_crossed.json");
    serde_json::from_str(
        &fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}")),
    )
    .unwrap_or_else(|e| panic!("parse {path:?}: {e}"))
}

#[test]
fn correlated_slope_crossed_joint_laplace_certifies_at_glmer_mle() {
    let data = fixture();
    let glmer_loglik = data["glmer_loglik"].as_f64().unwrap();
    let glmer_fixef: Vec<f64> = data["glmer_fixef"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    let nums = |key: &str| -> Vec<f64> {
        data[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect()
    };
    let cats = |key: &str, prefix: &str| -> Vec<String> {
        data[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| format!("{prefix}{}", v.as_i64().unwrap()))
            .collect()
    };

    let mut df = DataFrame::new();
    df.add_numeric("y", nums("y")).unwrap();
    df.add_numeric("x", nums("x")).unwrap();
    df.add_categorical("sub", cats("sub", "s")).unwrap();
    df.add_categorical("item", cats("item", "i")).unwrap();

    let formula = parse_formula("y ~ x + (1 + x | sub) + (1 + x | item)").unwrap();
    let mut joint =
        GeneralizedLinearMixedModel::new(formula, &df, Family::Binomial, Some(LinkFunction::Logit))
            .unwrap();
    joint.fit_with_options(false, 1, false).unwrap();

    let dloglik = (joint.loglikelihood() - glmer_loglik).abs();
    assert!(
        dloglik < 5.0e-3,
        "joint Laplace log-likelihood off glmer by {dloglik:.3e} (joint={:.6} glmer={glmer_loglik:.6})",
        joint.loglikelihood()
    );
    let max_dfixef = joint
        .coef()
        .iter()
        .zip(glmer_fixef.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_dfixef < 5.0e-3,
        "joint Laplace fixef off glmer by {max_dfixef:.3e} (joint={:?} glmer={glmer_fixef:?})",
        joint.coef().as_slice()
    );

    let return_value = &joint.lmm().optsum().return_value;
    assert!(
        return_value.starts_with("JOINT_LAPLACE:"),
        "expected a joint-Laplace return code, got {return_value:?}"
    );

    let certificate = joint
        .compiler_artifact()
        .optimizer_certificate
        .as_ref()
        .expect("joint fit must carry an optimizer certificate");
    assert_eq!(
        certificate.status,
        mixeff_rs::compiler::FitStatus::ConvergedInterior,
        "glmer-equivalent correlated-slope joint fit must certify, not read not_optimized (free gradient {:?})",
        certificate.free_gradient_norm
    );
}
