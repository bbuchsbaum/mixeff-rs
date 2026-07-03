// Engine-level inference tests migrated from src/model/linear/tests.rs
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
fn test_ml_fixef_and_stderror() {
    // reaction ~ 1 + days: two fixef, both SE positive
    let data = shared_julia_parity_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let fixef = MixedModelFit::fixef(&model);
    let se = MixedModelFit::stderror(&model);

    assert_eq!(fixef.len(), 2);
    assert_eq!(se.len(), 2);
    assert!(se[0] > 0.0, "intercept SE must be positive");
    assert!(se[1] > 0.0, "slope SE must be positive");
}

#[test]
fn test_ml_wald_confint() {
    let data = shared_julia_parity_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let coef = MixedModelFit::coef(&model);
    let se = MixedModelFit::stderror(&model);
    let names = MixedModelFit::coef_names(&model);
    let z = 1.959_963_984_540_054_f64; // qnorm(0.975)

    let ci = model.wald_confint(0.95);
    assert_eq!(ci.len(), coef.len());
    for (i, row) in ci.iter().enumerate() {
        assert_eq!(row.parameter, names[i]);
        assert_relative_eq!(row.estimate, coef[i], epsilon = 1e-12);
        assert_relative_eq!(row.lower, coef[i] - z * se[i], epsilon = 1e-9);
        assert_relative_eq!(row.upper, coef[i] + z * se[i], epsilon = 1e-9);
        assert!(row.lower < row.estimate && row.estimate < row.upper);
    }

    // Higher coverage widens every interval.
    let ci99 = model.wald_confint(0.99);
    for (a, b) in ci.iter().zip(ci99.iter()) {
        assert!((b.upper - b.lower) > (a.upper - a.lower));
    }
}

#[test]
fn test_lrt_nested_scalar_re_models() {
    // LRT comparing reaction ~ 1 + (1|subj) vs reaction ~ 1 + days + (1|subj).
    // The second model adds one FE parameter: chisq_dof == 1.
    use mixeff_rs::stats::lrt::LikelihoodRatioTest;

    let data = shared_julia_parity_fixture();
    let f0 = parse_formula("reaction ~ 1 + (1 | subj)").unwrap();
    let f1 = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();

    let mut m0 = LinearMixedModel::new(f0, &data, None).unwrap();
    let mut m1 = LinearMixedModel::new(f1, &data, None).unwrap();
    m0.fit(false).unwrap();
    m1.fit(false).unwrap();

    let lrt =
        LikelihoodRatioTest::test(&[&m0 as &dyn MixedModelFit, &m1 as &dyn MixedModelFit]).unwrap();

    // χ² = 2*(ll1 - ll0)
    let expected_chisq =
        2.0 * (MixedModelFit::loglikelihood(&m1) - MixedModelFit::loglikelihood(&m0));
    assert_relative_eq!(lrt.chisq[0], expected_chisq, epsilon = 1e-10);

    // Adding `days` costs 1 dof
    assert_eq!(lrt.chisq_dof[0], 1);

    // Fuller model has better (larger) log-likelihood
    assert!(MixedModelFit::loglikelihood(&m1) > MixedModelFit::loglikelihood(&m0));

    // p-value in [0, 1]
    assert!(lrt.pvalues[0] >= 0.0 && lrt.pvalues[0] <= 1.0);
}

#[test]
fn test_vcov_varpar_rejects_boundary_hessian_without_changing_state() {
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let theta_before = model.theta();
    let objective_before = model.objective_value();
    let mut varpar = fitted_varpar(&model);
    varpar[0] = 0.0;

    let err = model.vcov_varpar(&varpar, false).unwrap_err();

    assert!(err.to_string().contains("lower bound"));
    assert_eq!(model.theta(), theta_before);
    assert_relative_eq!(model.objective_value(), objective_before, epsilon = 1e-10);
}

#[test]
fn test_lrt_sleepstudy_matches_julia() {
    // Mirrors likelihoodratiotest.jl "likelihoodratio test":
    //   fm0: reaction ~ 1 + (1 + days | subj)  [no days in FE, dof=5]
    //   fm1: reaction ~ 1 + days + (1 + days | subj) [days in FE, dof=6]
    // Julia: chisq ≈ 23.5365, dof=1, p < 1e-5
    use mixeff_rs::stats::lrt::LikelihoodRatioTest;
    let data = sleepstudy_fixture();

    let f0 = parse_formula("reaction ~ 1 + (1 + days | subj)").unwrap();
    let mut m0 = LinearMixedModel::new(f0, &data, None).unwrap();
    m0.fit(false).unwrap();

    let f1 = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut m1 = LinearMixedModel::new(f1, &data, None).unwrap();
    m1.fit(false).unwrap();

    assert!(
        m0.objective_value() > m1.objective_value(),
        "fm0 should have larger objective"
    );
    assert_eq!(m0.dof(), 5);
    assert_eq!(m1.dof(), 6);

    let lrt = LikelihoodRatioTest::test(&[&m0 as &dyn MixedModelFit, &m1]).unwrap();
    assert_eq!(lrt.chisq_dof[0], 1);
    assert_relative_eq!(lrt.chisq[0], 23.5365, epsilon = 0.05);
    assert!(lrt.pvalues[0] < 1e-5);
}

#[test]
fn test_pastes_lrt_pvalue_matches_julia() {
    // Mirrors pls.jl "pastes": lrt = likelihoodratiotest(models(:pastes)...)
    //   last(lrt.pvalues) ≈ 0.5233767965780878
    // models(:pastes)[1] = strength ~ 1 + (1 | batch & cask)  (cask-within-batch only)
    // models(:pastes)[2] = strength ~ 1 + (1 | batch / cask)  (batch + batch:cask)
    let data = pastes_fixture();

    // Simpler model: batch:cask interaction only (no batch main effect)
    let formula1 = parse_formula("strength ~ 1 + (1 | batch_cask)").unwrap();
    let mut m1 = LinearMixedModel::new(formula1, &data, None).unwrap();
    m1.fit(false).unwrap();

    // Richer model: batch main RE + batch:cask interaction RE
    let formula2 = parse_formula("strength ~ 1 + (1 | batch) + (1 | batch_cask)").unwrap();
    let mut m2 = LinearMixedModel::new(formula2, &data, None).unwrap();
    m2.fit(false).unwrap();

    use mixeff_rs::model::traits::MixedModelFit;
    use mixeff_rs::stats::lrt::LikelihoodRatioTest;
    let lrt =
        LikelihoodRatioTest::test(&[&m1 as &dyn MixedModelFit, &m2 as &dyn MixedModelFit]).unwrap();
    assert_eq!(lrt.pvalues.len(), 1);
    assert_relative_eq!(lrt.pvalues[0], 0.5233767965780878, epsilon = 0.01);
}

// ── LRT parity tests (likelihoodratiotest.jl) ────────────────────────────

#[test]
fn test_lrt_sleepstudy_deviances_and_chisq() {
    // likelihoodratiotest.jl:
    //   fm0 = reaction ~ 1 + (1 + days | subj)       → deviance ≈ 1775.4759, dof = 5
    //   fm1 = reaction ~ 1 + days + (1 + days | subj) → deviance ≈ 1751.9393, dof = 6
    //   lrt.chisq[0] ≈ 23.5365, p-value < 1e-5
    use mixeff_rs::stats::lrt::LikelihoodRatioTest;

    let data = sleepstudy_fixture();

    let f0 = parse_formula("reaction ~ 1 + (1 + days | subj)").unwrap();
    let mut fm0 = LinearMixedModel::new(f0, &data, None).unwrap();
    fm0.fit(false).unwrap();

    let f1 = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut fm1 = LinearMixedModel::new(f1, &data, None).unwrap();
    fm1.fit(false).unwrap();

    // deviance = -2 * loglikelihood
    let dev0 = -2.0 * fm0.loglikelihood();
    let dev1 = -2.0 * fm1.loglikelihood();
    assert_relative_eq!(dev0, 1775.4759, epsilon = 0.1);
    assert_relative_eq!(dev1, 1751.9393, epsilon = 0.1);

    assert_eq!(fm0.dof(), 5);
    assert_eq!(fm1.dof(), 6);

    let lrt = LikelihoodRatioTest::test(&[&fm0 as &dyn MixedModelFit, &fm1 as &dyn MixedModelFit])
        .unwrap();

    assert_relative_eq!(lrt.chisq[0], 23.5365, epsilon = 0.05);
    assert!(
        lrt.pvalues[0] < 1e-5,
        "p-value should be < 1e-5, got {}",
        lrt.pvalues[0]
    );
}

// ── coeftable parity tests (pls.jl "coeftable" testset) ──────────────────

#[test]
fn test_coeftable_dyestuff_shape() {
    // pls.jl: ct = coeftable(only(models(:dyestuff)))
    //         @test [3, 4] == [ct.teststatcol, ct.pvalcol]
    // In our 0-indexed struct: z_values is column 2, p_values is column 3.
    // We verify the table has 1 row (intercept-only FE) and reasonable values.
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let ct = model.coeftable();

    // Dyestuff has one FE: (Intercept)
    assert_eq!(ct.len(), 1);
    assert_eq!(ct.names[0], "(Intercept)");

    // Estimate ≈ 1527.5 (mean of yield)
    assert_relative_eq!(ct.estimates[0], 1527.5, epsilon = 1.0);

    // z = estimate / SE should be very large (≈ 86)
    assert!(
        ct.z_values[0] > 50.0,
        "z for intercept should be large, got {}",
        ct.z_values[0]
    );

    // p-value should be essentially zero
    assert!(
        ct.p_values[0] < 1e-10,
        "p should be ≈0, got {}",
        ct.p_values[0]
    );
}

#[test]
fn test_coeftable_sleepstudy_two_rows() {
    // sleepstudy: FE = (Intercept) + days → 2 rows in coeftable
    // pls.jl: coef ≈ [251.405, 10.467], stderror ≈ [6.632, 1.502]
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let ct = model.coeftable();
    assert_eq!(ct.len(), 2);

    // Both should have small p-values (both highly significant)
    for i in 0..2 {
        assert!(
            ct.p_values[i] < 0.01,
            "coef[{}] p-value {} should be < 0.01",
            i,
            ct.p_values[i]
        );
        // z = estimate / SE should be non-zero and finite
        assert!(ct.z_values[i].is_finite(), "z[{}] should be finite", i);
    }

    // SE should be positive
    for se in &ct.std_errors {
        assert!(*se > 0.0, "SE should be positive, got {}", se);
    }
}

#[test]
fn test_coeftable_p_values_consistent_with_stderror() {
    // coeftable p-values should be consistent with stderror:
    // z = coef / SE,  p = 2*(1-Φ(|z|))
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let ct = model.coeftable();
    let coefs = MixedModelFit::coef(&model);
    let se = model.stderror();

    for i in 0..ct.len() {
        let expected_z = coefs[i] / se[i];
        assert_relative_eq!(ct.z_values[i], expected_z, epsilon = 1e-10);
    }
}
