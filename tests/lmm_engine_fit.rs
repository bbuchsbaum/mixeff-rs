// Engine-level fit tests migrated from src/model/linear/tests.rs
// (ranked-audit M3). Public-API only; internals-bound tests stay inline.

mod common;
#[allow(unused_imports)]
use common::*;

use approx::assert_relative_eq;
#[allow(unused_imports)]
use mixeff_rs::error::*;
use mixeff_rs::formula::parse_formula;
#[allow(unused_imports)]
use mixeff_rs::model::data::{Column, DataFrame};
#[allow(unused_imports)]
use mixeff_rs::model::fixed_design::*;
#[allow(unused_imports)]
use mixeff_rs::model::linear::*;
#[allow(unused_imports)]
use mixeff_rs::model::traits::MixedModelFit;
#[allow(unused_imports)]
use mixeff_rs::stats::*;
#[allow(unused_imports)]
use mixeff_rs::types::*;

#[test]
fn test_lmm_refuses_structured_random_covariance_before_fit() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + ar1(0 + days | subj)").unwrap();
    let err = LinearMixedModel::new(formula, &data, None).unwrap_err();
    assert_eq!(err.code(), "unsupported");
    assert!(err.to_string().contains("ar1"));
    assert!(err.to_string().contains("not fitted in v1.0"));
}

#[test]
fn test_lmm_audit_report_updates_after_fit() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

    let prefit_report = model.audit_report().to_text();
    assert!(prefit_report.contains("Optimizer"));
    assert!(prefit_report.contains("model has not been fitted"));

    model.fit(false).unwrap();

    let fitted_report = model.audit_report().to_text();
    assert!(fitted_report.contains("ConvergedInterior"));
    assert!(fitted_report.contains("pattern_search"));
    assert!(fitted_report.contains("convergence interpretation"));
    assert!(fitted_report.contains("verify_convergence"));
}

#[test]
fn test_scalar_covariance_kkt_certificate_interior_converged() {
    let data = shared_julia_parity_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

    model.fit(false).unwrap();

    let certificate = model.scalar_covariance_kkt_certificate().unwrap();
    assert_eq!(certificate.blocks.len(), 1);
    let block = &certificate.blocks[0];
    assert_eq!(
        block.classification,
        CovarianceKktClassification::InteriorConverged
    );
    assert!(block.variance > certificate.variance_tolerance);
    assert!(block.score.abs() <= certificate.score_tolerance);
    assert!(certificate.residual.is_finite());
}

#[test]
fn test_scalar_covariance_kkt_certificate_valid_zero_variance() {
    let data = singular_re_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

    model.fit(false).unwrap();

    let certificate = model.scalar_covariance_kkt_certificate().unwrap();
    assert_eq!(certificate.blocks.len(), 1);
    let block = &certificate.blocks[0];
    assert_eq!(
        block.classification,
        CovarianceKktClassification::ValidZeroVariance
    );
    assert!(block.variance <= certificate.variance_tolerance);
    assert!(block.score >= -certificate.score_tolerance);
    assert!(certificate.residual.is_finite());
}

#[test]
fn test_scalar_covariance_kkt_certificate_flags_invalid_boundary_stop() {
    let data = shared_julia_parity_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

    model.fit(false).unwrap();
    model.set_theta(&[0.0]).unwrap();

    let certificate = model.scalar_covariance_kkt_certificate().unwrap();
    assert_eq!(certificate.blocks.len(), 1);
    let block = &certificate.blocks[0];
    assert_eq!(
        block.classification,
        CovarianceKktClassification::InvalidBoundaryStop
    );
    assert!(block.variance <= certificate.variance_tolerance);
    assert!(block.score < -certificate.score_tolerance);
    assert!(certificate.residual.is_finite());
}

#[test]
fn test_scalar_covariance_kkt_certificate_marks_tiny_positive_variance_weak() {
    let data = shared_julia_parity_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

    model.fit(false).unwrap();
    model.set_theta(&[1e-3]).unwrap();

    let certificate = model.scalar_covariance_kkt_certificate().unwrap();
    assert_eq!(certificate.blocks.len(), 1);
    let block = &certificate.blocks[0];
    assert_eq!(
        block.classification,
        CovarianceKktClassification::WeakIdentification
    );
    assert!(block.variance > certificate.variance_tolerance);
    assert!(block.score.abs() > certificate.score_tolerance);
    assert!(certificate.residual.is_finite());
}

#[test]
fn test_vector_re_fit_is_invariant_to_row_order() {
    let data = simulate_sleepstudy_like(10, 5, 42);
    let order: Vec<usize> = (0..data.nrow()).rev().collect();
    let permuted = permute_rows(&data, &order);
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();

    let mut model_a = LinearMixedModel::new(formula.clone(), &data, None).unwrap();
    let mut model_b = LinearMixedModel::new(formula, &permuted, None).unwrap();

    model_a.fit(true).unwrap();
    model_b.fit(true).unwrap();

    assert_relative_eq!(
        model_a.objective_value(),
        model_b.objective_value(),
        epsilon = 1e-7,
        max_relative = 1e-7
    );
    assert_relative_eq!(
        model_a.sigma(),
        model_b.sigma(),
        epsilon = 1e-3,
        max_relative = 1e-3
    );

    let beta_a = model_a.beta();
    let beta_b = model_b.beta();
    for i in 0..beta_a.len() {
        assert_relative_eq!(beta_a[i], beta_b[i], epsilon = 1e-4, max_relative = 1e-4);
    }

    let theta_a = model_a.theta();
    let theta_b = model_b.theta();
    for i in 0..theta_a.len() {
        assert_relative_eq!(theta_a[i], theta_b[i], epsilon = 5e-3, max_relative = 5e-3);
    }
}

#[test]
fn test_response_accessor_matches_stored_response() {
    let data = shared_julia_parity_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let model = LinearMixedModel::new(formula, &data, None).unwrap();

    let y = model.y();
    let response = MixedModelFit::response(&model);

    assert_eq!(response.len(), y.len());
    for idx in 0..y.len() {
        assert_relative_eq!(response[idx], y[idx], epsilon = 1e-12, max_relative = 1e-12);
    }
}

// ── Tests ported from MixedModels.jl/test/pls.jl ────────────────────────

#[test]
fn test_ml_loglikelihood_aic_bic_relationships() {
    // Verify the algebraic relationships: ll = -obj/2, aic, bic.
    // Matches Julia's convention: objective already includes n*log(2π).
    let data = shared_julia_parity_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap(); // ML

    let n = model.nobs() as f64;
    let k = model.dof() as f64;
    let obj = model.objective_value();
    let ll = MixedModelFit::loglikelihood(&model);

    // ML: loglikelihood = -objective / 2
    assert_relative_eq!(ll, -obj / 2.0, epsilon = 1e-12);

    // AIC = -2*ll + 2*k
    assert_relative_eq!(
        MixedModelFit::aic(&model),
        -2.0 * ll + 2.0 * k,
        epsilon = 1e-12
    );

    // BIC = -2*ll + k*ln(n)
    assert_relative_eq!(
        MixedModelFit::bic(&model),
        -2.0 * ll + k * n.ln(),
        epsilon = 1e-12
    );
}

#[test]
fn test_ml_nobs_and_dof_scalar_re() {
    // 6 subjects × 4 days = 24 obs; dof = p(2) + n_theta(1) + 1(sigma) = 4
    let data = shared_julia_parity_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    assert_eq!(MixedModelFit::nobs(&model), 24);
    assert_eq!(MixedModelFit::dof(&model), 4);
}

#[test]
fn test_ml_ranef_dimensions_scalar_re() {
    // (1|subj): vsize=1, 6 subjects → matrix is 1×6
    let data = shared_julia_parity_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let ranef = model.ranef_b();
    assert_eq!(ranef.len(), 1, "one grouping factor");
    assert_eq!(ranef[0].nrows(), 1, "scalar RE: vsize = 1");
    assert_eq!(ranef[0].ncols(), 6, "6 subjects");
}

#[test]
fn test_is_singular_reflects_theta_at_lower_bound() {
    // After fitting non-degenerate data: not singular.
    // Driving theta to lower bound → singular.
    let data = shared_julia_parity_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    assert!(
        !model.is_singular(),
        "non-degenerate fit should not be singular"
    );

    let fitted_theta = model.theta();
    let lb = model.lower_bounds();
    model.set_theta(&lb).unwrap(); // θ = [0.0] → at lower bound
    assert!(model.is_singular(), "theta at lower bound must be singular");

    model.set_theta(&fitted_theta).unwrap();
    assert!(
        !model.is_singular(),
        "restored theta should not be singular"
    );
}

#[test]
fn test_lmm_set_theta_propagates_remat_err() {
    let data = shared_julia_parity_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

    let err = model.set_theta(&[]).unwrap_err();

    assert!(matches!(err, MixedModelError::DimensionMismatch(_)));
}

#[test]
fn test_set_theta_does_not_panic_on_bad_input() {
    let data = shared_julia_parity_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| model.set_theta(&[])));

    assert!(result.is_ok());
    assert!(matches!(
        result.unwrap(),
        Err(MixedModelError::DimensionMismatch(_))
    ));
}

#[test]
fn test_singular_re_fit_is_singular() {
    // Synthetic data: all group means identical (SS_B = 0).
    // Mirrors pls.jl "Dyestuff2" testset spirit: when between-group variance
    // is zero, θ → 0 and the model is singular.
    let data = singular_re_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap(); // ML

    assert!(model.is_singular(), "fit with SS_B=0 must be singular");
    assert_relative_eq!(model.theta()[0], 0.0, epsilon = 1e-10);
}

#[test]
fn lmm_builder_matches_direct_construction_byte_for_byte() {
    let df = dyestuff_fixture();
    for criterion in [ModelCriterion::Ml, ModelCriterion::Reml] {
        let reml = criterion.is_reml();

        let mut direct =
            LinearMixedModel::new(parse_formula("yield ~ 1 + (1 | batch)").unwrap(), &df, None)
                .unwrap();
        direct.fit(reml).unwrap();

        let built =
            LinearMixedModelBuilder::new(parse_formula("yield ~ 1 + (1 | batch)").unwrap(), &df)
                .fit(if criterion.is_reml() {
                    FitOptions::reml()
                } else {
                    FitOptions::ml()
                })
                .unwrap();

        assert_eq!(
            built.coef(),
            direct.coef(),
            "builder coef must match direct ({criterion:?})"
        );
        assert_eq!(
            built.objective(),
            direct.objective(),
            "builder objective must match direct ({criterion:?})"
        );
    }
}

#[test]
fn lmm_fit_options_reject_bad_start_theta_before_fitting() {
    let df = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &df, None).unwrap();

    let err = model
        .fit_with_options(
            FitOptions::reml()
                .with_optimizer_control(OptimizerControl::auto().with_start_theta(vec![0.1, 0.2])),
        )
        .expect_err("wrong-length start theta should be rejected");

    assert_eq!(err.code(), "invalid_argument");
    assert!(!model.is_fitted());
}

// ── Parity tests against Julia MixedModels.jl ──────────────────────────

#[test]
fn test_dyestuff_ml_matches_julia() {
    // Mirrors pls.jl "Dyestuff" testset (ML fit).
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap(); // ML

    assert_eq!(model.nobs(), 30);
    assert_eq!(model.dof(), 3);
    assert_relative_eq!(model.theta()[0], 0.7525806540074477, epsilon = 1e-4);
    assert_relative_eq!(model.fixef()[0], 1527.5, epsilon = 1e-6);
    assert_relative_eq!(model.sigma(), 49.51010035223816, epsilon = 1e-3);
    assert_relative_eq!(model.stderror()[0], 17.694552929494222, epsilon = 1e-2);
    assert_relative_eq!(model.objective_value(), 327.32705988112673, epsilon = 1e-3);
    // Julia: loglikelihood(fm1) ≈ -163.663... = -327.327/2
    assert_relative_eq!(
        model.loglikelihood(),
        -327.32705988112673 / 2.0,
        epsilon = 1e-3
    );
}

#[test]
fn test_deviance_varpar_matches_ml_scalar_fit_and_restores_state() {
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let theta_before = model.theta();
    let objective_before = model.objective_value();
    let vcov_before = model.vcov();
    let varpar = fitted_varpar(&model);

    let deviance = model.deviance_varpar(&varpar, false).unwrap();

    assert_relative_eq!(deviance, objective_before, epsilon = 1e-8);
    assert_eq!(model.theta(), theta_before);
    assert_relative_eq!(model.objective_value(), objective_before, epsilon = 1e-10);
    assert_relative_eq!(model.vcov(), vcov_before, epsilon = 1e-10);
}

#[test]
fn test_deviance_varpar_matches_reml_vector_fit_and_restores_state() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(true).unwrap();

    let theta_before = model.theta();
    let objective_before = model.objective_value();
    let vcov_before = model.vcov();
    let varpar = fitted_varpar(&model);

    let deviance = model.deviance_varpar(&varpar, true).unwrap();

    assert_relative_eq!(deviance, objective_before, epsilon = 1e-8);
    assert_eq!(model.theta(), theta_before);
    assert_relative_eq!(model.objective_value(), objective_before, epsilon = 1e-10);
    assert_relative_eq!(model.vcov(), vcov_before, epsilon = 1e-10);
}

#[test]
fn test_deviance_varpar_rejects_invalid_inputs_without_changing_state() {
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let theta_before = model.theta();
    let objective_before = model.objective_value();
    let mut varpar = fitted_varpar(&model);
    varpar[0] = -1.0;
    assert!(model.deviance_varpar(&varpar, false).is_err());

    let mut varpar = fitted_varpar(&model);
    *varpar.last_mut().unwrap() = 0.0;
    assert!(model.deviance_varpar(&varpar, false).is_err());

    assert_eq!(model.theta(), theta_before);
    assert_relative_eq!(model.objective_value(), objective_before, epsilon = 1e-10);
}

#[test]
fn test_dyestuff_aic_bic_matches_julia() {
    // Mirrors pls.jl "Dyestuff":
    //   aic(fm1) ≈ 333.32705988112673
    //   bic(fm1) ≈ 337.5306520261132
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let obj = model.objective_value(); // -2*loglik
    let k = model.dof() as f64;
    let n = model.nobs() as f64;
    let aic = obj + 2.0 * k;
    let bic = obj + k * n.ln();

    assert_relative_eq!(aic, 333.32705988112673, epsilon = 1e-3);
    assert_relative_eq!(bic, 337.5306520261132, epsilon = 1e-3);
}

#[test]
fn test_dyestuff_re_std_dev_matches_julia() {
    // Mirrors pls.jl: first(first(fm1.σs)) ≈ 37.260343703061764
    // RE std dev = lambda * sigma = 0.7526 * 49.51 ≈ 37.26
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let vc = model.varcorr();
    assert_eq!(vc.components.len(), 1);
    let comp = &vc.components[0];
    assert_eq!(comp.group, "batch");
    assert_relative_eq!(comp.std_dev[0], 37.260343703061764, epsilon = 0.1);
}

#[test]
fn test_dyestuff_reml_matches_julia() {
    // Mirrors pls.jl "Dyestuff" REML refit.
    // Julia: objective ≈ 319.6542768422576
    //        vcov[0,0] ≈ 375.7167103872769 (variance of intercept under REML)
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(true).unwrap(); // REML

    assert_relative_eq!(model.objective_value(), 319.6542768422576, epsilon = 1e-3);
    // REML vcov of the intercept
    let v = model.vcov();
    assert_eq!(v.nrows(), 1);
    assert_relative_eq!(v[(0, 0)], 375.7167103872769, epsilon = 1.0);
}

#[test]
fn test_sleepstudy_vector_re_matches_julia() {
    // Mirrors pls.jl "sleep" testset (last model: (1 + days | subj)).
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap(); // ML

    assert_relative_eq!(model.objective_value(), 1751.9393444636682, epsilon = 0.01);
    let theta = model.theta();
    assert_eq!(theta.len(), 3);
    assert_relative_eq!(theta[0], 0.9292297167514472, epsilon = 1e-3);
    assert_relative_eq!(theta[1], 0.01816466496782548, epsilon = 1e-3);
    assert_relative_eq!(theta[2], 0.22264601131030412, epsilon = 1e-3);

    // coef() returns in original formula order: [intercept, days]
    let coef = MixedModelFit::coef(&model);
    assert_relative_eq!(coef[0], 251.40510484848454, epsilon = 0.01);
    assert_relative_eq!(coef[1], 10.467285959596126, epsilon = 0.01);

    let se = model.stderror();
    assert_relative_eq!(se[0], 6.632295312722272, epsilon = 0.1);
    assert_relative_eq!(se[1], 1.5022387911441102, epsilon = 0.05);

    assert_relative_eq!(model.loglikelihood(), -875.9696722318341, epsilon = 0.01);
}

#[test]
fn test_penicillin_crossed_re_matches_julia() {
    // Mirrors pls.jl "penicillin" testset.
    // Formula: diameter ~ 1 + (1 | plate) + (1 | sample)
    let data = penicillin_fixture();
    let formula = parse_formula("diameter ~ 1 + (1 | plate) + (1 | sample)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap(); // ML

    assert_eq!(model.nobs(), 144);

    assert_relative_eq!(model.objective_value(), 332.1883486700085, epsilon = 0.01);

    let coef = MixedModelFit::coef(&model);
    assert_relative_eq!(coef[0], 22.97222222222222, epsilon = 1e-4);

    assert_relative_eq!(model.stderror()[0], 0.7446037806555799, epsilon = 0.01);

    // θ[0] = plate RE, θ[1] = sample RE
    let theta = model.theta();
    assert_eq!(theta.len(), 2);
    assert_relative_eq!(theta[0], 1.5375939045981573, epsilon = 0.01);
    assert_relative_eq!(theta[1], 3.219792193110907, epsilon = 0.01);
}

#[test]
fn test_dyestuff2_singular_fit_matches_julia() {
    // Mirrors pls.jl "Dyestuff2" testset.
    // The within-batch variance dominates → RE collapses to 0 (singular).
    let data = dyestuff2_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap(); // ML

    // Julia: fm.θ ≈ zeros(1)
    assert!(
        model.theta()[0].abs() < 1e-6,
        "theta should be ~0 for singular fit, got {}",
        model.theta()[0]
    );
    // Julia: objective(fm) ≈ 162.87303665382575
    assert_relative_eq!(model.objective_value(), 162.87303665382575, epsilon = 1e-3);
    // Julia: coef(fm) ≈ [5.6656]
    let coef = MixedModelFit::coef(&model);
    assert_relative_eq!(coef[0], 5.6656, epsilon = 1e-3);
    // Julia: stderror(fm) ≈ [0.6669857396443264]
    assert_relative_eq!(model.stderror()[0], 0.6669857396443264, epsilon = 1e-3);
    // Julia: logdet(fm) ≈ 0.0 (RE variance = 0 → Λ diagonal = 0)
    assert_relative_eq!(model.logdet_re(), 0.0, epsilon = 1e-8);
    // Julia: issingular(fm) == true
    assert!(model.is_singular(), "Dyestuff2 fit should be singular");
}

#[test]
fn test_dyestuff_logdet_pwrss_varest() {
    // Mirrors pls.jl "Dyestuff" testset — additional metrics after ML fit.
    // Julia: logdet(fm1) ≈ 8.06014611206176
    //        varest(fm1) ≈ 2451.2500368886936  (= sigma^2)
    //        pwrss(fm1)  ≈ 73537.50110666081
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    assert_relative_eq!(model.logdet_re(), 8.06014611206176, epsilon = 1e-3);
    assert_relative_eq!(
        model.sigma() * model.sigma(),
        2451.2500368886936,
        epsilon = 1.0
    );
    assert_relative_eq!(model.pwrss(), 73537.50110666081, epsilon = 10.0);
}

#[test]
fn test_penicillin_logdet_and_varest() {
    // Mirrors pls.jl "penicillin" testset — additional metrics.
    // Julia: varest(fm) ≈ 0.30242510228527864
    //        logdet(fm) ≈ 95.74676552743833
    let data = penicillin_fixture();
    let formula = parse_formula("diameter ~ 1 + (1 | plate) + (1 | sample)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    assert_relative_eq!(
        model.sigma() * model.sigma(),
        0.30242510228527864,
        epsilon = 1e-4
    );
    assert_relative_eq!(model.logdet_re(), 95.74676552743833, epsilon = 0.1);
}

#[test]
fn test_sleepstudy_random_slope_only_matches_julia() {
    // Mirrors pls.jl: fmrs = reaction ~ 1 + days + (0 + days | subj)
    // Random slope only (no random intercept).
    // Julia: objective ≈ 1774.080315280526, θ ≈ [0.24353985601485326]
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (0 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    assert_relative_eq!(model.objective_value(), 1774.080315280526, epsilon = 0.01);
    let theta = model.theta();
    assert_eq!(theta.len(), 1, "random-slope-only has scalar theta");
    assert_relative_eq!(theta[0], 0.24353985601485326, epsilon = 1e-3);
}

#[test]
fn test_pastes_nested_re_matches_julia() {
    // Mirrors pls.jl "pastes" testset.
    // Julia formula: strength ~ 1 + (1 | batch / cask)
    // which expands to: strength ~ 1 + (1 | batch) + (1 | batch:cask)
    // We use pre-computed batch_cask interaction column.
    // Julia: objective ≈ 247.9944658624955
    //        coef ≈ [60.0533333333333]
    //        stderror ≈ [0.6421355774401101]
    //        θ ≈ [3.5269029347766856, 1.3299137410046242]
    let data = pastes_fixture();
    let formula = parse_formula("strength ~ 1 + (1 | batch) + (1 | batch_cask)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    assert_eq!(model.nobs(), 60);
    assert_relative_eq!(model.objective_value(), 247.9944658624955, epsilon = 0.01);

    let coef = MixedModelFit::coef(&model);
    assert_relative_eq!(coef[0], 60.0533333333333, epsilon = 1e-3);

    assert_relative_eq!(model.stderror()[0], 0.6421355774401101, epsilon = 0.01);

    let theta = model.theta();
    assert_eq!(theta.len(), 2);
    // Julia sorts by decreasing nranef: θ[0] = batch:cask RE (30 levels), θ[1] = batch RE (10 levels)
    #[cfg(feature = "nlopt")]
    let theta_epsilon = 0.05;
    #[cfg(not(feature = "nlopt"))]
    let theta_epsilon = 0.09;
    assert_relative_eq!(theta[0], 3.5269029347766856, epsilon = theta_epsilon);
    assert_relative_eq!(theta[1], 1.3299137410046242, epsilon = theta_epsilon);
}

#[test]
fn test_weighted_model_matches_julia() {
    // Mirrors pls.jl "wts" testset.
    // Julia: m2 = fit(@formula(a ~ 1 + b + (1 | c)), data; wts=w1)
    //   θ ≈ [0.2951818091809752]
    //   stderror ≈ [0.964016663994572, 3.6309691484830533]
    //   vcov ≈ [[0.9293, -2.5575], [-2.5575, 13.1839]]
    let (df, w1) = weighted_lmm_fixture();

    let formula = parse_formula("a ~ 1 + b + (1 | c)").unwrap();
    let mut model = LinearMixedModel::new(formula, &df, Some(&w1)).unwrap();
    model.fit(false).unwrap();

    assert_relative_eq!(model.theta()[0], 0.2951818091809752, epsilon = 1e-3);
    let se = model.stderror();
    assert_eq!(se.len(), 2);
    assert_relative_eq!(se[0], 0.964016663994572, epsilon = 0.01);
    assert_relative_eq!(se[1], 3.6309691484830533, epsilon = 0.1);
    // Julia: vcov ≈ [[0.9293 -2.5575], [-2.5575 13.1839]]
    let v = model.vcov();
    assert_relative_eq!(v[(0, 0)], 0.9293281284592235, epsilon = 0.01);
    assert_relative_eq!(v[(0, 1)], -2.5575260810649962, epsilon = 0.05);
    assert_relative_eq!(v[(1, 0)], -2.5575260810649962, epsilon = 0.05);
    assert_relative_eq!(v[(1, 1)], 13.18393695723575, epsilon = 0.1);
}

#[test]
fn test_sleepstudy_re_std_devs_match_julia() {
    // Mirrors pls.jl "sleep":
    //   first(std(fm)) ≈ [23.78066438213187, 5.7168446983832775]
    //   VarCorr RE correlation between intercept and days ≈ +0.08
    //   fm.corr (fixed-effects correlation) ≈ [1.0 -0.1376; -0.1376 1.0]
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let vc = model.varcorr();
    assert_eq!(vc.components.len(), 1);
    let comp = &vc.components[0];
    assert_eq!(comp.group, "subj");
    assert_eq!(comp.std_dev.len(), 2);
    // Julia: first(std(fm)) ≈ [23.78066438213187, 5.7168446983832775]
    assert_relative_eq!(comp.std_dev[0], 23.78066438213187, epsilon = 0.1);
    assert_relative_eq!(comp.std_dev[1], 5.7168446983832775, epsilon = 0.1);
    // VarCorr RE correlation: theta[1] / ||row_1(lambda)|| ≈ +0.08
    assert_eq!(comp.correlations.len(), 1);
    assert_relative_eq!(comp.correlations[0], 0.0813, epsilon = 0.01);

    // fm.corr in Julia is vcov(m; corr=true) — the fixed-effects correlation,
    // NOT VarCorr. Julia: stderror ≈ [6.6323, 1.5022], corr[0,1] ≈ -0.1376.
    let vcov = model.vcov();
    let se = model.stderror();
    assert_relative_eq!(se[0], 6.632295312722272, epsilon = 0.01);
    assert_relative_eq!(se[1], 1.5022387911441102, epsilon = 0.01);
    let fe_corr = vcov[(0, 1)] / (se[0] * se[1]);
    assert_relative_eq!(fe_corr, -0.13755599049585931, epsilon = 0.01);
}

#[test]
fn test_sleepstudy_vector_re_logdet_and_pwrss() {
    // Mirrors pls.jl "sleep" testset — additional metrics.
    // Julia: logdet(fm) ≈ 73.90350673367566
    //        pwrss(fm)  ≈ 117889.27379003687
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    assert_relative_eq!(model.logdet_re(), 73.90350673367566, epsilon = 0.1);
    assert_relative_eq!(model.pwrss(), 117889.27379003687, epsilon = 100.0);
}

#[cfg(feature = "nlopt")]
#[test]
fn test_penicillin_varcorr_std_devs_match_julia() {
    // Mirrors pls.jl "penicillin": std(fm) ≈ [[0.8456], [1.7707], [0.5499]]
    // std[0] = plate RE, std[1] = sample RE, residual sigma = 0.5499
    let data = penicillin_fixture();
    let formula = parse_formula("diameter ~ 1 + (1 | plate) + (1 | sample)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let sigma = model.sigma();
    // Julia: only(last(std)) ≈ 0.549931906953287 (residual sigma)
    assert_relative_eq!(sigma, 0.549931906953287, epsilon = 1e-4);

    let vc = model.varcorr();
    assert_eq!(vc.components.len(), 2);
    // plate RE
    assert_eq!(vc.components[0].group, "plate");
    assert_relative_eq!(
        vc.components[0].std_dev[0],
        0.845571948075415,
        epsilon = 1e-4
    );
    // sample RE
    assert_eq!(vc.components[1].group, "sample");
    assert_relative_eq!(
        vc.components[1].std_dev[0],
        1.770666460750787,
        epsilon = 1e-4
    );
    // residual
    assert_relative_eq!(vc.residual_sd.unwrap(), sigma, epsilon = 1e-12);
}

// Parity against MixedModels.jl reference fit (NLopt BOBYQA); the
// native no-default-features path lands slightly away in sigma^2.
#[cfg(feature = "nlopt")]
#[test]
fn test_pastes_varcorr_and_logdet_match_julia() {
    // Mirrors pls.jl "pastes":
    //   only(first(stdd)) ≈ 2.904   (batch:cask RE std dev, 30 levels — first in nranef sort)
    //   only(stdd[2])     ≈ 1.095   (batch RE std dev, 10 levels — second)
    //   only(last(stdd))  ≈ 0.823   (residual sigma)
    //   varest(fm) ≈ 0.677999727889528
    //   logdet(fm) ≈ 101.03834542101686
    let data = pastes_fixture();
    let formula = parse_formula("strength ~ 1 + (1 | batch) + (1 | batch_cask)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let sigma = model.sigma();
    assert_relative_eq!(sigma, 0.8234073887751603, epsilon = 1e-4);
    assert_relative_eq!(sigma * sigma, 0.677999727889528, epsilon = 1e-4);
    assert_relative_eq!(model.logdet_re(), 101.03834542101686, epsilon = 0.1);

    let vc = model.varcorr();
    assert_eq!(vc.components.len(), 2);
    // Julia sorts RE terms by decreasing nranef: batch:cask (30 levels) first, batch (10) second.
    // Julia: first(std) ≈ 2.904 (batch:cask, 30 levels), stdd[2] ≈ 1.095 (batch, 10 levels)
    let batch_comp = vc
        .components
        .iter()
        .find(|c| c.group == "batch")
        .expect("batch component");
    let cask_comp = vc
        .components
        .iter()
        .find(|c| c.group == "batch_cask")
        .expect("batch_cask component");
    assert_relative_eq!(cask_comp.std_dev[0], 2.90407793598792, epsilon = 1e-3);
    assert_relative_eq!(batch_comp.std_dev[0], 1.0950608007768226, epsilon = 1e-4);
    // residual
    assert_relative_eq!(vc.residual_sd.unwrap(), sigma, epsilon = 1e-12);
}

#[test]
fn test_dyestuff2_sigma_matches_julia() {
    // Mirrors pls.jl "Dyestuff2": std(fm)[2] ≈ [3.6532313513746537]
    // (residual sigma; RE collapses to 0 in singular fit)
    let data = dyestuff2_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    assert_relative_eq!(model.sigma(), 3.6532313513746537, epsilon = 1e-4);
}

#[test]
fn test_pastes_batch_cask_only_model() {
    // models(:pastes)[1] = strength ~ 1 + (1 | batch & cask) — cask-within-batch only.
    // Julia: objective ≈ 247.9944658624955 for the full nested model (last);
    //   the simpler model (batch & cask only) has fewer RE levels.
    // Here we just verify it fits and has sane values.
    let data = pastes_fixture();
    let formula = parse_formula("strength ~ 1 + (1 | batch_cask)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    assert_eq!(model.nobs(), 60);
    // Intercept ≈ mean(strength)
    let coef = MixedModelFit::coef(&model);
    assert_relative_eq!(coef[0], 60.0533333333333, epsilon = 0.1);
    // This simpler model must have lower DOF than the full nested model
    assert_eq!(model.dof(), 3); // 1 FE + 1 RE theta + 1 sigma
}

#[test]
fn test_dyestuff_cond_is_one() {
    // Mirrors pls.jl: cond(fm1) == ones(1)
    // Scalar RE has a 1×1 Lambda → condition number is always 1.
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let c = model.cond();
    assert_eq!(c.len(), 1);
    assert_relative_eq!(c[0], 1.0, epsilon = 1e-12);
}

#[test]
fn test_sleepstudy_vector_re_cond_matches_julia() {
    // Mirrors pls.jl: only(cond(fm)) ≈ 4.175266438717022
    // Vector RE Lambda is 2×2 lower-triangular; condition number > 1.
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let c = model.cond();
    assert_eq!(c.len(), 1);
    assert_relative_eq!(c[0], 4.175266438717022, epsilon = 0.01);
}

#[test]
fn test_dof_residual_matches_julia() {
    // Mirrors pls.jl: dof_residual(fm1) ≥ 0
    // For dyestuff: nobs=30, rank=1 (intercept only) → dof_residual=29
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    assert_eq!(model.dof_residual(), 29); // 30 obs - 1 FE
    assert!(model.dof_residual() > 0);
}

#[test]
fn test_sleepstudy_dof_residual() {
    // Sleepstudy: nobs=180, rank=2 (intercept + days) → dof_residual=178
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    assert_eq!(model.dof_residual(), 178); // 180 obs - 2 FE
}

#[test]
fn test_dyestuff_response_and_model_matrix() {
    // Mirrors pls.jl: modelmatrix(fm1) == ones(30,1), response == ds.yield
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let x = model.model_matrix();
    assert_eq!(x.nrows(), 30);
    assert_eq!(x.ncols(), 1);
    // Intercept-only FE → all ones
    assert!(x.iter().all(|&v| (v - 1.0).abs() < 1e-12));

    let y = model.response();
    assert_eq!(y.len(), 30);
    // First batch A: 5 values with mean ~1538
    let mean_y = y.mean();
    assert_relative_eq!(mean_y, 1527.5, epsilon = 1e-6);
}

// ── condVar parity with MixedModels.jl/test/pls.jl ─────────────────────

#[test]
fn test_dyestuff_condvar_shape() {
    // pls.jl: @test length(cv) == 1; @test size(first(cv)) == (1, 1, 6)
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let cv = model.cond_var();
    assert_eq!(cv.len(), 1, "one RE term");
    assert_eq!(cv[0].len(), 6, "6 batch levels");
    assert_eq!(cv[0][0].nrows(), 1);
    assert_eq!(cv[0][0].ncols(), 1);
}

#[test]
fn test_penicillin_condvar_matches_julia() {
    // pls.jl:
    //   @test length(cv) == 2
    //   @test size(first(cv)) == (1, 1, 24)
    //   @test size(last(cv)) == (1, 1, 6)
    //   @test first(first(cv)) ≈ 0.07331356908917808 rtol = 1.e-4
    //   @test last(last(cv))  ≈ 0.04051591717427688 rtol = 1.e-4
    let data = penicillin_fixture();
    let formula = parse_formula("diameter ~ 1 + (1 | plate) + (1 | sample)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let cv = model.cond_var();
    assert_eq!(cv.len(), 2);

    // first term = plate (24 levels, sorted first by nranef)
    assert_eq!(cv[0].len(), 24);
    assert_eq!(cv[0][0].nrows(), 1);
    assert_relative_eq!(cv[0][0][(0, 0)], 0.07331356908917808, epsilon = 1e-4);

    // last term = sample (6 levels)
    assert_eq!(cv[1].len(), 6);
    assert_relative_eq!(cv[1][5][(0, 0)], 0.04051591717427688, epsilon = 1e-4);
}

#[test]
fn test_sleepstudy_condvar_matches_julia() {
    // pls.jl:
    //   @test size(cv1) == (2, 2, 18)
    //   @test first(cv1) ≈ 140.96755256125914 rtol = 1.e-4   → cv[0][0][(0,0)]
    //   @test last(cv1)  ≈ 5.157794803497628  rtol = 1.e-4   → cv[0][17][(1,1)]
    //   @test cv1[2]     ≈ -20.604544204749537 rtol = 1.e-4  → cv[0][0][(1,0)]
    //   (Julia column-major: cv1[2] = cv1[2,1,1] = row 2, col 1, level 1 = (1,0) 0-indexed)
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let cv = model.cond_var();
    assert_eq!(cv.len(), 1);
    assert_eq!(cv[0].len(), 18);
    assert_eq!(cv[0][0].nrows(), 2);
    assert_eq!(cv[0][0].ncols(), 2);

    assert_relative_eq!(cv[0][0][(0, 0)], 140.96755256125914, epsilon = 1.0);
    assert_relative_eq!(cv[0][17][(1, 1)], 5.157794803497628, epsilon = 0.1);
    assert_relative_eq!(cv[0][0][(1, 0)], -20.604544204749537, epsilon = 0.5);
}

#[test]
fn test_ranef_u_regression_current_outputs() {
    let data = penicillin_fixture();
    let formula = parse_formula("diameter ~ 1 + (1 | plate) + (1 | sample)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let rfu = model.ranef_u();

    assert_eq!(rfu.len(), 2);
    assert_relative_eq!(rfu[0][(0, 0)], 0.5231574704291094, epsilon = 1e-3);
    assert_relative_eq!(rfu[1][(0, 5)], -0.9323155679350466, epsilon = 1e-3);
}

#[test]
fn test_dyestuff_ranef_u_sums_to_zero() {
    // pls.jl: @test abs(sum(only(rfu))) < 1.e-5
    // The u vector for a balanced model sums to zero (BLUP property).
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let rfu = model.ranef_u();
    assert_eq!(rfu.len(), 1);
    let u_sum: f64 = rfu[0].iter().sum();
    assert!(
        u_sum.abs() < 1e-4,
        "sum of u (dyestuff) should be ≈ 0, got {u_sum}"
    );
}

#[cfg(feature = "nlopt")]
#[test]
fn test_sleepstudy_ranef_u_shape_and_first_element() {
    // pls.jl:
    //   @test size(first(u3)) == (2, 18)
    //   @test first(only(u3)) ≈ 3.030047743065841 atol = 0.001
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let u3 = model.ranef_u();
    assert_eq!(u3.len(), 1, "one RE term");
    assert_eq!(u3[0].nrows(), 2, "vsize = 2 (intercept + slope)");
    assert_eq!(u3[0].ncols(), 18, "18 subjects");

    // Julia's first(only(u3)) is the (1,1) element (intercept for first subject)
    assert_relative_eq!(u3[0][(0, 0)], 3.030047743065841, epsilon = 0.001);
}

#[cfg(feature = "nlopt")]
#[test]
fn test_sleepstudy_ranef_b_first_element() {
    // pls.jl: @test first(only(b3)) ≈ 2.8156104060324334 atol = 0.001
    // b = Λ * u  (conditional mode on original scale)
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let b3 = model.ranef_b();
    assert_eq!(b3.len(), 1);
    assert_eq!(b3[0].nrows(), 2);
    assert_eq!(b3[0].ncols(), 18);
    assert_relative_eq!(b3[0][(0, 0)], 2.8156104060324334, epsilon = 0.001);
}

#[test]
fn test_penicillin_ranef_u_first_element() {
    // pls.jl: @test first(first(rfu)) ≈ 0.5231574704291094 rtol = 1.e-4
    // penicillin has 2 RE terms (plate, sample); rfu is sorted by decreasing nranef.
    // first(rfu) → the term with more levels (24 plates).
    let data = penicillin_fixture();
    let formula = parse_formula("diameter ~ 1 + (1 | plate) + (1 | sample)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let rfu = model.ranef_u();
    assert_eq!(rfu.len(), 2, "two RE terms");

    // Determine which term is plate (24 levels) — it should sort first
    let first_term = &rfu[0];
    let first_u = first_term[(0, 0)];
    assert_relative_eq!(first_u, 0.5231574704291094, epsilon = 1e-3);
}

#[test]
fn test_penicillin_ranef_b_last_element() {
    // pls.jl: @test last(last(rfb)) ≈ -3.0018241391465703 rtol = 1.e-4
    // last(rfb) is the term with fewer levels (6 samples).
    let data = penicillin_fixture();
    let formula = parse_formula("diameter ~ 1 + (1 | plate) + (1 | sample)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let rfb = model.ranef_b();
    assert_eq!(rfb.len(), 2);

    // last term (fewer levels = samples, 6 levels), last element
    let last_term = &rfb[rfb.len() - 1];
    let last_b = last_term[(0, last_term.ncols() - 1)];
    assert_relative_eq!(last_b, -3.0018241391465703, epsilon = 1e-3);
}

// ── std / logdet / varest / model_size / refit / simulate parity ─────────

#[test]
fn test_penicillin_varest_and_logdet() {
    // pls.jl:
    //   @test varest(fm) ≈ 0.30242510228527864 atol=0.0001
    //   @test logdet(fm) ≈ 95.74676552743833 atol=0.005
    let data = penicillin_fixture();
    let formula = parse_formula("diameter ~ 1 + (1 | plate) + (1 | sample)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    assert_relative_eq!(model.varest(), 0.30242510228527864, epsilon = 1e-4);
    assert_relative_eq!(model.logdet(), 95.74676552743833, epsilon = 0.05);
}

#[test]
fn test_penicillin_std_devs() {
    // pls.jl:
    //   stdd = std(fm)
    //   @test only(first(stdd)) ≈ 0.845571948075415 atol=0.0001   # plate
    //   @test only(stdd[2]) ≈ 1.770666460750787 atol=0.0001       # sample
    //   @test only(last(stdd)) ≈ 0.549931906953287 atol=0.0001    # sigma
    let data = penicillin_fixture();
    let formula = parse_formula("diameter ~ 1 + (1 | plate) + (1 | sample)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let stdd = model.std_devs();
    // reterms sorted by decreasing nranef: plate (24) first, sample (6) second
    assert_relative_eq!(stdd[0][0], 0.845571948075415, epsilon = 1e-3);
    assert_relative_eq!(stdd[1][0], 1.770666460750787, epsilon = 1e-3);
    assert_relative_eq!(stdd[2][0], 0.549931906953287, epsilon = 1e-3); // sigma
}

#[test]
fn test_penicillin_model_size() {
    // pls.jl: @test size(fm) == (144, 1, 30, 2)
    // n=144, p=1, nranef=24+6=30, nretrms=2
    let data = penicillin_fixture();
    let formula = parse_formula("diameter ~ 1 + (1 | plate) + (1 | sample)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    assert_eq!(model.model_size(), (144, 1, 30, 2));
}

#[test]
fn test_sleepstudy_model_size() {
    // pls.jl: @test size(fm) == (180, 2, 36, 1) for the vector RE model
    // n=180, p=2, nranef=18*2=36, nretrms=1
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    assert_eq!(model.model_size(), (180, 2, 36, 1));
}

#[test]
fn test_dyestuff_refit_new_response() {
    // pls.jl: refit!(fm, new_y); @test objective(fm) ≈ 327.32705988112673 atol=0.001
    // (refitting a dyestuff2-like model with the dyestuff yields)
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();
    let dev_before = model.objective_value();

    // Refit with constant-shifted response (should converge to different value)
    let new_y: Vec<f64> = model.y().iter().map(|&y| y + 100.0).collect();
    model.refit(&new_y).unwrap();

    // β (intercept) should shift by 100; deviance should be unchanged
    assert_relative_eq!(model.objective_value(), dev_before, epsilon = 1e-4);
}

#[test]
fn stateless_transform_end_to_end_fit() {
    // log(reaction) ~ days + I(days^2) + (1 | subj) fits, and the
    // transform labels surface as the response name and a coefficient
    // name byte-identical to what R prints.
    let data = sleepstudy_fixture();
    let formula = parse_formula("log(reaction) ~ days + I(days^2) + (1 | subj)").unwrap();
    assert_eq!(formula.response, "log(reaction)");
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let names = model.coef_names();
    assert!(
        names.iter().any(|n| n == "I(days^2)"),
        "coef_names should contain `I(days^2)`, got {names:?}"
    );
    assert!(names.iter().any(|n| n == "days"));
    // The objective is finite (it actually fit).
    assert!(model.objective_value().is_finite());
}

// ── Cook's distance parity tests (pls.jl line 705) ───────────────────────

// ── Cook's distance parity tests (pls.jl line 705) ───────────────────────

#[test]
fn test_cooks_distance_length() {
    // cooksdistance(model) should have length n.
    // Uses first(models(:sleepstudy)) = reaction ~ 1 + days + (1 | subj)
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let d = model.cooks_distance();
    assert_eq!(d.len(), data.nrow());
}

#[test]
fn test_cooks_distance_nonnegative() {
    // All Cook's distances should be ≥ 0.
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let d = model.cooks_distance();
    for (i, &di) in d.iter().enumerate() {
        assert!(
            di >= 0.0,
            "Cook's distance[{}] should be non-negative, got {}",
            i,
            di
        );
    }
}

#[test]
fn test_cooks_distance_parity_sleepstudy() {
    // pls.jl line 705-760: lme4 reference values for Cook's distance.
    // Model: first(models(:sleepstudy)) = reaction ~ 1 + days + (1 | subj)
    //
    // Julia uses:  D_i = (r_i/(1-h_i))^2 * h_i / (varest(m) * p)
    // where p = rank of fixed-effects matrix = 2.
    //
    // We compare the first 10 values at rtol=0.10 (10%).
    let lme4_cooks: Vec<f64> = vec![
        0.1270714,
        0.1267805,
        0.243096,
        0.0002437091,
        0.03145029,
        0.2954052,
        0.04550505,
        0.3552723,
        0.1984806,
        0.4518805,
    ];

    let data = sleepstudy_fixture();
    // first(models(:sleepstudy)) — intercept-only RE per subject
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let d = model.cooks_distance();

    for (i, &expected) in lme4_cooks.iter().enumerate() {
        let got = d[i];
        let rel_err = ((got - expected) / expected).abs();
        assert!(
            rel_err < 0.10,
            "Cook's distance[{}]: expected {:.6}, got {:.6} (rel err {:.2}%)",
            i,
            expected,
            got,
            rel_err * 100.0
        );
    }
}

#[test]
fn test_cooks_distance_sum_finite() {
    // Sum should be finite (no NaN/Inf from degenerate h_i).
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let d = model.cooks_distance();
    let s: f64 = d.iter().sum();
    assert!(s.is_finite(), "Sum of Cook's distances should be finite");
}
