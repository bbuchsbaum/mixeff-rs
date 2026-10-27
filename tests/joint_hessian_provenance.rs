#![cfg(feature = "unstable-internals")]

use mixeff_rs::compiler::{
    CertificateCheck, EvidenceMethod, EvidenceQuality, FixedEffectCovarianceMethod,
    FixedEffectCovarianceStatus, IncompleteCheckStatus,
};
use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::{DataFrame, Family, GeneralizedLinearMixedModel};

fn poisson_fixture() -> DataFrame {
    let group_effects = [-0.35, -0.1, 0.15, 0.4];
    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut group = Vec::new();
    for (group_index, group_effect) in group_effects.iter().enumerate() {
        for observation in 0..8 {
            let x_value = observation as f64 - 3.5;
            let eta = 1.0 + 0.13 * x_value + group_effect;
            // Deterministic, non-separated count variation around the mean.
            y.push((eta.exp() + ((group_index + observation) % 3) as f64 * 0.25).round());
            x.push(x_value);
            group.push(format!("g{group_index}"));
        }
    }
    let mut data = DataFrame::new();
    data.add_numeric("y", y).expect("response");
    data.add_numeric("x", x).expect("predictor");
    data.add_categorical("group", group).expect("group");
    data
}

#[test]
fn available_joint_covariance_does_not_rewrite_optimizer_hessian_scope() {
    let data = poisson_fixture();
    let formula = parse_formula("y ~ 1 + x + (1 | group)").expect("formula");
    let mut model =
        GeneralizedLinearMixedModel::new(formula, &data, Family::Poisson, None).expect("model");
    model
        .fit_with_options(false, 1, false)
        .expect("joint-Laplace fit");

    // Requesting the artifact executes the deferred inference Hessian on a
    // clone. Its successful covariance must not promote or replace the
    // fit-time optimizer certificate's deliberately unassessed full Hessian.
    let artifact = model.compiler_artifact();
    let covariance = artifact
        .fixed_effect_covariance_matrix
        .as_ref()
        .expect("joint covariance payload");
    assert_eq!(covariance.status, FixedEffectCovarianceStatus::Available);
    assert_eq!(
        covariance.method,
        FixedEffectCovarianceMethod::JointLaplaceActiveHessian
    );
    assert!(covariance.notes.iter().any(|note| note ==
        "fixed-effect covariance derived as twice the beta block of the inverse central-finite-difference Hessian of the joint-laplace -2 log L objective over beta plus interior theta parameters"));
    assert!(covariance.notes.iter().any(|note| note ==
        "this post-fit inference Hessian is distinct from the fit-time optimizer certificate and does not revise optimizer convergence status or certification quality"));

    let certificate = artifact
        .optimizer_certificate
        .as_ref()
        .expect("optimizer certificate");
    let scope_reason = "full joint Hessian intentionally not assessed in the optimizer certificate; joint-Laplace inference separately attempts an active-space Hessian over beta plus interior theta parameters and does not revise optimizer certification";
    assert!(matches!(
        &certificate.evidence.hessian.method,
        EvidenceMethod::NotAssessed { reason } if reason == scope_reason
    ));
    assert!(matches!(
        &certificate.evidence.hessian.quality,
        EvidenceQuality::NotAssessed { reason } if reason == scope_reason
    ));
    assert!(matches!(
        &certificate.evidence.certification_quality,
        EvidenceQuality::Unavailable { reason } if reason.contains(scope_reason)
    ));
    assert!(certificate.checks.iter().any(|check| matches!(
        check,
        CertificateCheck::Incomplete { evidence }
            if evidence.check_name == "active_subspace_hessian_psd"
                && evidence.status == IncompleteCheckStatus::NotAssessed
                && evidence.reason == scope_reason
    )));
}
