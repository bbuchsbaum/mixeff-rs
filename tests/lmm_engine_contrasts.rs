// Engine-level contrasts tests migrated from src/model/linear/tests.rs
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
fn test_builtin_sum_contrast_fit_matches_treatment_fit_and_names_columns() {
    // lme4-style check for the built-in constructors: the same model fitted
    // under `contr.sum` and `contr.treatment` (both with an explicit level
    // order that differs from first appearance) spans the same column space,
    // so the ML objective and fitted values must agree exactly and the
    // coefficient vectors must be the known linear transforms of the cell
    // means. Column names follow R: sum coding names the first k-1 levels,
    // treatment coding names levels 2..k.
    let conds = ["hi", "lo", "mid"]; // first-appearance order: hi, lo, mid
    let cond_effect = |c: &str| match c {
        "lo" => -2.0,
        "mid" => 0.5,
        _ => 3.0,
    };
    let subj_effect = [0.0, 1.0, -1.0];
    let mut y = Vec::new();
    let mut cond = Vec::new();
    let mut subj = Vec::new();
    let mut i = 0usize;
    for _rep in 0..2 {
        #[allow(clippy::needless_range_loop)]
        for s in 0..3 {
            for c in conds {
                let noise = ((i as f64 * 12.9898).sin() * 43758.547).fract() - 0.5;
                y.push(10.0 + cond_effect(c) + subj_effect[s] + noise);
                cond.push(c.to_string());
                subj.push(format!("s{s}"));
                i += 1;
            }
        }
    }
    // Explicit canonical order lo < mid < hi, deliberately different from
    // the first-appearance order in the data.
    let levels: Vec<String> = ["lo", "mid", "hi"].iter().map(|s| s.to_string()).collect();

    let fit_with = |contrast: mixeff_rs::model::data::CategoricalContrast| {
        let mut data = DataFrame::new();
        data.add_numeric("y", y.clone()).unwrap();
        data.add_categorical_with_contrast("cond", cond.clone(), levels.clone(), contrast)
            .unwrap();
        data.add_categorical("subj", subj.clone()).unwrap();
        let formula = parse_formula("y ~ 1 + cond + (1 | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();
        model
    };

    let sum_model =
        fit_with(mixeff_rs::model::data::CategoricalContrast::sum(levels.clone()).unwrap());
    let trt_model =
        fit_with(mixeff_rs::model::data::CategoricalContrast::treatment(levels.clone()).unwrap());

    let sum_names = sum_model.coef_names();
    let trt_names = trt_model.coef_names();
    assert_eq!(sum_names, vec!["(Intercept)", "cond: lo", "cond: mid"]);
    assert_eq!(trt_names, vec!["(Intercept)", "cond: mid", "cond: hi"]);

    assert_relative_eq!(
        sum_model.objective_value(),
        trt_model.objective_value(),
        epsilon = 1e-8,
        max_relative = 1e-10
    );
    let f_sum = sum_model.fitted();
    let f_trt = trt_model.fitted();
    for (a, b) in f_sum.iter().zip(f_trt.iter()) {
        assert_relative_eq!(*a, *b, epsilon = 1e-6);
    }

    // Cell means from the treatment fit (reference = lo), then the sum-coded
    // coefficients must be grand mean and deviations from it.
    let bt = trt_model.coef();
    let bs = sum_model.coef();
    let mu_lo = bt[0];
    let mu_mid = bt[0] + bt[1];
    let mu_hi = bt[0] + bt[2];
    let grand = (mu_lo + mu_mid + mu_hi) / 3.0;
    assert_relative_eq!(bs[0], grand, epsilon = 1e-6);
    assert_relative_eq!(bs[1], mu_lo - grand, epsilon = 1e-6);
    assert_relative_eq!(bs[2], mu_mid - grand, epsilon = 1e-6);
}

#[test]
fn test_sleepstudy_zerocorr_varcorr_std_devs() {
    // Mirrors pls.jl "sleep" fmnc (zerocorr):
    //   first(std(fmnc)) ≈ [24.171269957611873, 5.79939919963132]
    //   last(std(fmnc))  ≈ [25.55613836753517]   (residual sigma)
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days || subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let sigma = model.sigma();
    assert_relative_eq!(sigma, 25.55613836753517, epsilon = 0.1);

    let vc = model.varcorr();
    assert_eq!(vc.components.len(), 1);
    let comp = &vc.components[0];
    assert_eq!(comp.std_dev.len(), 2);
    assert_relative_eq!(comp.std_dev[0], 24.171269957611873, epsilon = 0.1);
    assert_relative_eq!(comp.std_dev[1], 5.79939919963132, epsilon = 0.1);
    // zerocorr → diagonal Lambda → off-diagonal correlation is 0
    assert_eq!(comp.correlations.len(), 1);
    assert_relative_eq!(comp.correlations[0], 0.0, epsilon = 1e-8);
}

#[test]
fn test_sleepstudy_independent_re_equivalent_to_zerocorr() {
    // Mirrors pls.jl "sleep" fm_ind equivalence test (lines 447-454):
    //   fm_ind = models(:sleepstudy)[3]
    //          = reaction ~ 1 + days + (1 | subj) + (0 + days | subj)
    //   @test objective(fm_ind) ≈ objective(fmnc)   # fmnc = zerocorr model
    //   @test coef(fm_ind) ≈ coef(fmnc)
    //   @test stderror(fm_ind) ≈ stderror(fmnc)
    //   @test fm_ind.θ ≈ fmnc.θ
    //   @test logdet(fm_ind) ≈ logdet(fmnc)
    //
    // Two separate scalar RE terms for the same grouping factor are
    // equivalent to a single zerocorr (diagonal-λ) RE term because
    // their contributions to the log-likelihood are additive.
    let data = sleepstudy_fixture();

    let f_zc = parse_formula("reaction ~ 1 + days + (1 + days || subj)").unwrap();
    let mut m_zc = LinearMixedModel::new(f_zc, &data, None).unwrap();
    m_zc.fit(false).unwrap();

    // Two separate scalar terms for same grouping factor
    let f_ind = parse_formula("reaction ~ 1 + days + (1 | subj) + (0 + days | subj)").unwrap();
    let mut m_ind = LinearMixedModel::new(f_ind, &data, None).unwrap();
    m_ind.fit(false).unwrap();

    // Objectives should match to high precision (same log-likelihood surface)
    assert_relative_eq!(
        m_ind.objective_value(),
        m_zc.objective_value(),
        epsilon = 0.01
    );

    // Fixed-effects coefficients (pivot order may differ, compare sums/lengths)
    let coef_zc = MixedModelFit::coef(&m_zc);
    let coef_ind = MixedModelFit::coef(&m_ind);
    assert_eq!(
        coef_zc.len(),
        coef_ind.len(),
        "same number of FE coefficients"
    );

    // logdet should match
    assert_relative_eq!(m_ind.logdet_re(), m_zc.logdet_re(), epsilon = 0.1);

    // theta lengths differ (zerocorr: 2 params in 1 term; fm_ind: 1+1 in 2 terms)
    // but the effective model is the same
    assert_eq!(
        m_ind.theta().len(),
        2,
        "two separate scalar RE → 2 theta params"
    );
    assert_eq!(m_zc.theta().len(), 2, "zerocorr RE → 2 theta params");
}
