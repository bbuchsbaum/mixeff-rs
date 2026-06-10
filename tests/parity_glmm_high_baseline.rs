#![cfg(not(feature = "nlopt"))]
#![cfg(feature = "unstable-internals")]
//! Regression for bd-01KT40T6FGVXQQ9N50G2HM0ZZE.
//!
//! On a well-conditioned, high-baseline random-intercept binomial GLMM the
//! native joint Laplace optimizer (trust_bq) used to stall at the fast-PIRLS
//! profiled start: the trust region's stagnation guard tripped after a few
//! rejected steps while the radius was still far larger than the small move to
//! the optimum, so it reported a premature interior convergence with the
//! intercept biased toward zero by ~0.03 and the log-likelihood ~5e-3 short of
//! the glmer MLE. The fix withholds the stagnation stop until the search has
//! actually descended below the starting objective, letting the radius contract
//! until a step is accepted and the joint optimum is reached.

use std::fs;
use std::path::PathBuf;

use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::traits::MixedModelFit;
use mixeff_rs::model::{DataFrame, Family, GeneralizedLinearMixedModel, LinkFunction};
use serde_json::Value;

fn fixture_at(name: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/regression")
        .join(name);
    serde_json::from_str(
        &fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}")),
    )
    .unwrap_or_else(|e| panic!("parse {path:?}: {e}"))
}

fn fixture() -> Value {
    fixture_at("glmm_high_baseline_random_intercept.json")
}

fn build_model(data: &Value) -> GeneralizedLinearMixedModel {
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

    let formula = parse_formula("y ~ x + (1 | sub) + (1 | item)").unwrap();
    GeneralizedLinearMixedModel::new(formula, &df, Family::Binomial, Some(LinkFunction::Logit))
        .unwrap()
}

#[test]
fn high_baseline_random_intercept_joint_laplace_reaches_glmer_mle() {
    let data = fixture();
    let glmer_loglik = data["glmer_loglik"].as_f64().unwrap();
    let glmer_fixef: Vec<f64> = data["glmer_fixef"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    // The fast-PIRLS profiled fit is the joint optimizer's deterministic start;
    // on this surface it lands biased low. The joint Laplace fit must improve on
    // it and reach the glmer MLE rather than stalling at the start.
    let mut fast = build_model(&data);
    fast.fit_with_options(true, 1, false).unwrap();
    let fast_loglik = fast.loglikelihood();

    let mut joint = build_model(&data);
    joint.fit_with_options(false, 1, false).unwrap();
    let joint_loglik = joint.loglikelihood();
    let joint_fixef = joint.coef();

    // The joint optimizer descended past the profiled start (the early-stop is
    // gone): pre-fix `joint_loglik == fast_loglik` because the search was pinned
    // at the start.
    assert!(
        joint_loglik > fast_loglik + 1.0e-4,
        "joint Laplace did not improve on the fast-PIRLS start: joint={joint_loglik:.6} fast={fast_loglik:.6}"
    );

    // And it reaches the glmer MLE within the tight band the issue calls for
    // (pre-fix this was ~3e-2 on the intercept, ~5e-3 on the log-likelihood).
    let dloglik = (joint_loglik - glmer_loglik).abs();
    assert!(
        dloglik < 5.0e-2,
        "joint Laplace log-likelihood off glmer by {dloglik:.3e} (joint={joint_loglik:.6} glmer={glmer_loglik:.6})"
    );
    let max_dfixef = joint_fixef
        .iter()
        .zip(glmer_fixef.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_dfixef < 5.0e-3,
        "joint Laplace fixef off glmer by {max_dfixef:.3e} (joint={:?} glmer={glmer_fixef:?})",
        joint_fixef.as_slice()
    );

    // The joint optimizer should have returned its own joint-Laplace optimum,
    // not discarded it for the labelled fast-PIRLS fallback.
    let return_value = &joint.lmm().optsum().return_value;
    assert!(
        return_value.starts_with("JOINT_LAPLACE:"),
        "expected a joint-Laplace return code, got {return_value:?}"
    );
}

/// Regression for bd-01KTQFTH6J0ZFGR5RMV28HAX44 (and the correlated-slope
/// sibling bd-01KTQ7T8DPAGA1GS93R71D6QKF).
///
/// On the larger (n=2880, seed-101) high-baseline crossed random-intercept
/// fixture the trust_bq ftol stop used to rest ~1e-3 deviance short of the
/// stationary point inside a steep narrow valley: the certification gradient
/// genuinely read ~0.7 there, so a glmer-equivalent fit surfaced as
/// fit_status=not_optimized. The certification path now (a) re-probes failing
/// gradient components at escalated finite-difference steps so inner-PIRLS
/// deviance noise cannot fake a failure, and (b) finishes a *genuine*
/// assessed failure with the damped-Newton stationarity polish before
/// certifying. The result must reach the glmer MLE and certify as a real
/// interior convergence — this is the status assertion the smaller fixture's
/// test deliberately left out while the defect was open.
#[test]
fn high_baseline_random_intercept_large_joint_laplace_certifies_at_glmer_mle() {
    let data = fixture_at("glmm_high_baseline_random_intercept_large.json");
    let glmer_loglik = data["glmer_loglik"].as_f64().unwrap();
    let glmer_fixef: Vec<f64> = data["glmer_fixef"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let glmer_theta: Vec<f64> = data["glmer_theta"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    let mut joint = build_model(&data);
    joint.fit_with_options(false, 1, false).unwrap();

    let dloglik = (joint.loglikelihood() - glmer_loglik).abs();
    assert!(
        dloglik < 5.0e-3,
        "joint Laplace log-likelihood off glmer by {dloglik:.3e}"
    );
    let max_dfixef = joint
        .coef()
        .iter()
        .zip(glmer_fixef.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_dfixef < 5.0e-3,
        "joint Laplace fixef off glmer by {max_dfixef:.3e}"
    );
    let max_dtheta = joint
        .theta()
        .iter()
        .zip(glmer_theta.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_dtheta < 5.0e-3,
        "joint Laplace theta off glmer by {max_dtheta:.3e}"
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
        "glmer-equivalent joint fit must certify, not read not_optimized (free gradient {:?})",
        certificate.free_gradient_norm
    );
}
