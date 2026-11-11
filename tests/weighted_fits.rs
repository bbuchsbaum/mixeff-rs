//! Regression tests for case-weighted mixed models (issues #15, #16).
//!
//! The data are generated here so the tests need no fixtures: 15 groups of
//! 16 rows with random intercepts and slopes, and case weights in (0.5, 2)
//! whose logs sum to a positive number (the case issue #15 mishandled).

use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::{
    DataFrame, Family, FitOptions, GeneralizedLinearMixedModel, GeneralizedLinearMixedModelBuilder,
    LinearMixedModel, MixedModelFit, OptimizerControl,
};
use mixeff_rs::nalgebra::{DMatrix, DVector};
use mixeff_rs::stats::{
    BoundaryLikelihoodRatioTest, BoundaryLrtStatus, LikelihoodRatioTest, LinearModelFit,
    ModelComparisonTable,
};
use mixeff_rs::types::Optimizer;

/// Deterministic uniform draws in (0, 1).
struct Lcg(u64);
impl Lcg {
    fn unif(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        (-2.0 * self.unif().ln()).sqrt() * (std::f64::consts::TAU * self.unif()).cos()
    }
}

struct Data {
    frame: DataFrame,
    y: Vec<f64>,
    x: Vec<f64>,
    weights: Vec<f64>,
}

fn data() -> Data {
    let mut rng = Lcg(20261109);
    let (mut y, mut x, mut g, mut weights) = (vec![], vec![], vec![], vec![]);
    for group in 0..15 {
        let (b0, b1) = (0.7 * rng.normal(), 0.3 * rng.normal());
        for i in 0..16 {
            let xi = i as f64 / 5.0 - 1.5;
            let wi = 0.5 + 1.5 * rng.unif();
            x.push(xi);
            weights.push(wi);
            y.push(1.0 + b0 + (0.5 + b1) * xi + 0.4 * rng.normal() / wi.sqrt());
            g.push(format!("g{group:02}"));
        }
    }
    let mut frame = DataFrame::new();
    frame.add_numeric("y", y.clone()).unwrap();
    frame.add_numeric("x", x.clone()).unwrap();
    frame.add_categorical("g", g).unwrap();
    Data {
        frame,
        y,
        x,
        weights,
    }
}

fn fit(
    formula: &str,
    data: &Data,
    weights: Option<&[f64]>,
    optimizer: Option<Optimizer>,
) -> LinearMixedModel {
    let formula = parse_formula(formula).unwrap();
    let mut model = LinearMixedModel::new(formula, &data.frame, weights).unwrap();
    let mut control = OptimizerControl::auto();
    if let Some(optimizer) = optimizer {
        control = control.with_optimizer(optimizer);
    }
    model
        .fit_with_options(FitOptions::ml().with_optimizer_control(control))
        .unwrap();
    model
}

/// ML log-likelihood of `y ~ x` by weighted least squares, in closed form.
fn weighted_linear_loglik(data: &Data, weights: &[f64]) -> f64 {
    let sum =
        |f: &dyn Fn(usize) -> f64| -> f64 { (0..data.y.len()).map(|i| weights[i] * f(i)).sum() };
    let (sw, sx, sy) = (sum(&|_| 1.0), sum(&|i| data.x[i]), sum(&|i| data.y[i]));
    let sxx = sum(&|i| data.x[i] * data.x[i]);
    let sxy = sum(&|i| data.x[i] * data.y[i]);
    let slope = (sw * sxy - sx * sy) / (sw * sxx - sx * sx);
    let intercept = (sy - slope * sx) / sw;
    let rss = sum(&|i| (data.y[i] - intercept - slope * data.x[i]).powi(2));
    let n = data.y.len() as f64;
    -n / 2.0 * ((2.0 * std::f64::consts::PI * rss / n).ln() + 1.0)
        + 0.5 * weights.iter().map(|w| w.ln()).sum::<f64>()
}

#[test]
fn weighted_fit_is_the_same_for_every_optimizer_and_weight_scale() {
    let data = data();
    assert!(data.weights.iter().map(|w| w.ln()).sum::<f64>() > 0.0);
    let optimizers = [
        ("auto", None),
        ("trust_bq", Some(Optimizer::TrustBq)),
        ("cobyla", Some(Optimizer::Cobyla)),
        ("pattern_search", Some(Optimizer::PatternSearch)),
    ];
    let reference = fit(
        "y ~ x + (1 + x | g)",
        &data,
        Some(&data.weights),
        Some(Optimizer::PatternSearch),
    );
    let reference_sd: Vec<f64> = reference
        .theta()
        .iter()
        .map(|t| t * reference.sigma())
        .collect();
    // The optimum is away from the start theta = [1, 0, 1].
    assert!((reference.theta()[1]).abs() > 0.1);

    for (label, optimizer) in optimizers {
        // Rescaling every weight rescales sigma^2 only: the likelihood and
        // the random-effect standard deviations are unchanged.
        for scale in [0.1, 1.0, 10.0] {
            let weights: Vec<f64> = data.weights.iter().map(|w| w * scale).collect();
            let model = fit("y ~ x + (1 + x | g)", &data, Some(&weights), optimizer);
            let context = format!("{label}, weights x{scale}");
            assert!(
                (model.loglikelihood() - reference.loglikelihood()).abs() < 1e-5,
                "{context}: loglik {} vs {}",
                model.loglikelihood(),
                reference.loglikelihood()
            );
            for (sd, expected) in model.theta().iter().zip(&reference_sd) {
                let sd = sd * model.sigma();
                assert!(
                    (sd - expected).abs() < 2e-3,
                    "{context}: random-effect sd {sd} vs {expected}"
                );
            }
            assert!(
                (model.sigma() / scale.sqrt() - reference.sigma()).abs() < 1e-3,
                "{context}: sigma"
            );
            // Stored optimizer values are on the user-facing objective scale.
            let summary = model.opt_summary();
            assert!(
                (summary.fmin - model.objective()).abs() < 1e-8 * model.objective().abs(),
                "{context}: optsum.fmin {} vs objective {}",
                summary.fmin,
                model.objective()
            );
            assert!(summary.fmin < summary.finitial, "{context}: no progress");
            assert!(
                model.snapshot_json().is_ok(),
                "{context}: fitted state must be restorable"
            );
        }
    }
}

#[test]
fn weighted_linear_model_likelihood_matches_closed_form_and_the_lmm_scale() {
    let data = data();
    let design = DMatrix::from_fn(
        data.y.len(),
        2,
        |row, col| {
            if col == 0 {
                1.0
            } else {
                data.x[row]
            }
        },
    );
    let response = DVector::from_column_slice(&data.y);

    let weighted =
        LinearModelFit::fit_weighted(response.clone(), design.clone(), &data.weights, None)
            .unwrap();
    let expected = weighted_linear_loglik(&data, &data.weights);
    assert!((weighted.loglikelihood() - expected).abs() < 1e-9);
    assert_eq!(weighted.case_weights().unwrap(), data.weights);

    // Unit weights reproduce the ordinary fit exactly.
    let ones = vec![1.0; data.y.len()];
    let unit = LinearModelFit::fit_weighted(response.clone(), design.clone(), &ones, None).unwrap();
    let ordinary = LinearModelFit::fit(response.clone(), design.clone(), None).unwrap();
    assert!((unit.loglikelihood() - ordinary.loglikelihood()).abs() < 1e-10);
    assert!((unit.coef() - ordinary.coef()).norm() < 1e-10);
    assert!((unit.vcov() - ordinary.vcov()).norm() < 1e-12);
    assert!(ordinary.case_weights().is_none());

    // A mixed model's likelihood is never below its fixed-effects reference
    // (the random-effect variance can be zero), with the same weights.
    let full = fit("y ~ x + (1 | g)", &data, Some(&data.weights), None);
    assert!(full.loglikelihood() >= weighted.loglikelihood());
    let reference = LinearModelFit::fixed_effects_reference(&full, None).unwrap();
    assert!((reference.loglikelihood() - expected).abs() < 1e-8);

    assert!(
        LinearModelFit::fit_weighted(response.clone(), design.clone(), &ones[1..], None).is_err()
    );
    let mut negative = ones.clone();
    negative[0] = -1.0;
    assert!(LinearModelFit::fit_weighted(response, design, &negative, None).is_err());
}

#[test]
fn boundary_lrt_of_a_weighted_model_uses_the_weighted_reference() {
    let data = data();
    for weights in [None, Some(data.weights.as_slice())] {
        let full = fit("y ~ x + (1 | g)", &data, weights, None);
        let reduced = LinearModelFit::fixed_effects_reference(&full, Some("y ~ x".into())).unwrap();
        let test = BoundaryLikelihoodRatioTest::variance_component(&reduced, &full);
        let unit = vec![1.0; data.y.len()];
        let expected =
            2.0 * (full.loglikelihood() - weighted_linear_loglik(&data, weights.unwrap_or(&unit)));
        assert_eq!(test.status, BoundaryLrtStatus::Available);
        assert!(
            (test.statistic.unwrap() - expected).abs() < 1e-6,
            "statistic {} vs {expected}",
            test.statistic.unwrap()
        );
    }

    // Explicit unit weights are the unweighted model: either side may omit them.
    let ones = vec![1.0; data.y.len()];
    let unit_full = fit("y ~ x + (1 | g)", &data, Some(&ones), None);
    let plain_full = fit("y ~ x + (1 | g)", &data, None, None);
    let plain_reduced =
        LinearModelFit::fixed_effects_reference(&plain_full, Some("y ~ x".into())).unwrap();
    let unit_reduced =
        LinearModelFit::fixed_effects_reference(&unit_full, Some("y ~ x".into())).unwrap();
    let mixed = [
        BoundaryLikelihoodRatioTest::variance_component(&plain_reduced, &unit_full),
        BoundaryLikelihoodRatioTest::variance_component(&unit_reduced, &plain_full),
    ];
    let plain = BoundaryLikelihoodRatioTest::variance_component(&plain_reduced, &plain_full);
    for test in mixed {
        assert_eq!(test.status, BoundaryLrtStatus::Available);
        assert!((test.statistic.unwrap() - plain.statistic.unwrap()).abs() < 1e-6);
    }

    // Weights are compared relatively: tiny weights that differ by a factor
    // of two on alternate rows are a different weighting scheme.
    let tiny: Vec<f64> = vec![1e-12; data.y.len()];
    let alternating: Vec<f64> = (0..data.y.len())
        .map(|i| if i % 2 == 0 { 1e-12 } else { 2e-12 })
        .collect();
    let tiny_full = fit("y ~ x + (1 | g)", &data, Some(&tiny), None);
    let design = tiny_full.model_matrix().clone();
    let response = tiny_full.response().clone();
    let same = LinearModelFit::fit_weighted(response.clone(), design.clone(), &tiny, None).unwrap();
    let different = LinearModelFit::fit_weighted(response, design, &alternating, None).unwrap();
    let accepted = BoundaryLikelihoodRatioTest::variance_component(&same, &tiny_full);
    assert_eq!(accepted.status, BoundaryLrtStatus::Available);
    let rejected = BoundaryLikelihoodRatioTest::variance_component(&different, &tiny_full);
    assert_ne!(rejected.status, BoundaryLrtStatus::Available);
    assert!(rejected.reason.as_deref().unwrap().contains("case weights"));

    // An unweighted reference for a weighted model is refused, not tested.
    let full = fit("y ~ x + (1 | g)", &data, Some(&data.weights), None);
    let unweighted =
        LinearModelFit::fit(full.response().clone(), full.model_matrix().clone(), None).unwrap();
    let refused = BoundaryLikelihoodRatioTest::variance_component(&unweighted, &full);
    assert_ne!(refused.status, BoundaryLrtStatus::Available);
    assert!(refused.statistic.is_none() && refused.pvalue.is_none());
    assert!(refused.reason.as_deref().unwrap().contains("case weights"));
}

#[test]
fn likelihood_ratio_test_refuses_models_fitted_with_different_case_weights() {
    let data = data();
    let full = fit("y ~ x + (1 | g)", &data, Some(&data.weights), None);

    // The weighted reference gives the closed-form statistic.
    let reduced = LinearModelFit::fixed_effects_reference(&full, Some("y ~ x".into())).unwrap();
    let lrt = LikelihoodRatioTest::test(&[&reduced as &dyn MixedModelFit, &full]).unwrap();
    let expected = 2.0 * (full.loglikelihood() - weighted_linear_loglik(&data, &data.weights));
    assert!(
        (lrt.chisq[0] - expected).abs() < 1e-6,
        "statistic {} vs {expected}",
        lrt.chisq[0]
    );

    // An unweighted reference is refused rather than tested.
    let unweighted =
        LinearModelFit::fit(full.response().clone(), full.model_matrix().clone(), None).unwrap();
    let err = LikelihoodRatioTest::test(&[&unweighted as &dyn MixedModelFit, &full]).unwrap_err();
    assert!(err.contains("case weights"), "{err}");
}

/// Binomial proportions with trial-count prior weights, generated here:
/// 12 groups of 8 rows, 4 to 19 trials per row.
fn binomial_frame() -> (DataFrame, Vec<f64>) {
    let mut rng = Lcg(20261111);
    let (mut proportion, mut x, mut g, mut trials) = (vec![], vec![], vec![], vec![]);
    for group in 0..12 {
        let b0 = 0.6 * rng.normal();
        for i in 0..8 {
            let xi = i as f64 / 3.5 - 1.0;
            let p = 1.0 / (1.0 + (-(-0.3 + 0.8 * xi + b0)).exp());
            let n = 4 + (rng.unif() * 16.0) as usize;
            let successes = (0..n).filter(|_| rng.unif() < p).count();
            proportion.push(successes as f64 / n as f64);
            x.push(xi);
            trials.push(n as f64);
            g.push(format!("g{group:02}"));
        }
    }
    let mut frame = DataFrame::new();
    frame.add_numeric("proportion", proportion).unwrap();
    frame.add_numeric("x", x).unwrap();
    frame.add_categorical("g", g).unwrap();
    (frame, trials)
}

fn binomial_glmm(
    formula: &str,
    weights: Option<&[f64]>,
    frame: &DataFrame,
) -> GeneralizedLinearMixedModel {
    let builder = GeneralizedLinearMixedModelBuilder::new(
        parse_formula(formula).unwrap(),
        frame,
        Family::Binomial,
    );
    match weights {
        Some(weights) => builder.weights(weights.to_vec()),
        None => builder,
    }
    .fit()
    .unwrap()
}

#[test]
fn glmm_comparisons_refuse_models_fitted_with_different_prior_weights() {
    // Trial-count prior weights are part of the binomial data, so models
    // fitted with different weights are not nested fits of the same data.
    let (frame, trials) = binomial_frame();
    let doubled: Vec<f64> = trials.iter().map(|n| 2.0 * n).collect();

    let small = binomial_glmm("proportion ~ 1 + (1 | g)", Some(&trials), &frame);
    let large = binomial_glmm("proportion ~ 1 + x + (1 | g)", Some(&trials), &frame);
    let reweighted = binomial_glmm("proportion ~ 1 + x + (1 | g)", Some(&doubled), &frame);
    assert_eq!(large.case_weights().as_deref(), Some(trials.as_slice()));

    // GLMMs are likelihood fits, so a fixed-effect LRT between them is valid
    // (their working-LMM summary must not make them look like REML fits).
    assert!(!small.is_reml() && !large.is_reml());
    let same = LikelihoodRatioTest::test(&[&small as &dyn MixedModelFit, &large]).unwrap();
    let expected = 2.0 * (large.loglikelihood() - small.loglikelihood());
    assert!(expected > 0.0 && (same.chisq[0] - expected).abs() < 1e-9);
    let table = ModelComparisonTable::compare(&[&small as &dyn MixedModelFit, &large]).unwrap();
    assert!(table.rows[1].lrt_available);

    let err = LikelihoodRatioTest::test(&[&small as &dyn MixedModelFit, &reweighted]).unwrap_err();
    assert!(err.contains("case weights"), "{err}");
    let table =
        ModelComparisonTable::compare(&[&small as &dyn MixedModelFit, &reweighted]).unwrap();
    assert!(!table.rows[1].lrt_available);
    assert!(table.rows[1].chisq.is_none() && table.rows[1].pvalue.is_none());

    // Unweighted GLMMs report no case weights, and match explicit unit weights.
    let ones = vec![1.0; trials.len()];
    let unweighted = binomial_glmm("proportion ~ 1 + (1 | g)", None, &frame);
    assert!(unweighted.case_weights().is_none());
    let unit = binomial_glmm("proportion ~ 1 + x + (1 | g)", Some(&ones), &frame);
    assert!(LikelihoodRatioTest::test(&[&unweighted as &dyn MixedModelFit, &unit]).is_ok());
}
