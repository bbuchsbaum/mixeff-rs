//! Gamma / inverse-Gaussian GLMM scale conventions against lme4::glmer.
//!
//! The engine uses lme4's conventions for these families: the conditional
//! density's dispersion is the mean unit deviance (the family `aic()` that
//! glmer's Laplace criterion calls), the reported residual SD is lme4's
//! `sigma()` = sqrt((Pearson RSS + ||u||²) / n), and that same scale rescales
//! the fixed-effect covariance and the random-effect SDs. lme4's printed
//! logLik for these families includes the `aic()`'s `+2` term, so it is
//! exactly 1.0 below the engine's (true Laplace) logLik.
//! Reference values: tests/fixtures/parity/dispersion_glmm_lme4_scale.provenance.json.

use approx::assert_relative_eq;
use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::data::DataFrame;
use mixeff_rs::model::generalized::GeneralizedLinearMixedModel;
use mixeff_rs::model::traits::{Family, LinkFunction, MixedModelFit};

fn data() -> DataFrame {
    let text = include_str!("fixtures/parity/dispersion_glmm_lme4_scale.csv");
    let (mut y, mut x, mut g) = (vec![], vec![], vec![]);
    for line in text.lines().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        y.push(fields[0].parse::<f64>().unwrap());
        x.push(fields[1].parse::<f64>().unwrap());
        g.push(fields[2].trim_matches('"').to_string());
    }
    let mut data = DataFrame::new();
    data.add_numeric("y", y).unwrap();
    data.add_numeric("x", x).unwrap();
    data.add_categorical("g", g).unwrap();
    data
}

struct Lme4 {
    loglik: f64,
    fixef: [f64; 2],
    theta: f64,
    sigma: f64,
    se: [f64; 2],
}

fn check(family: Family, fast: bool, want: Lme4, tol: f64) {
    let data = data();
    let mut model = GeneralizedLinearMixedModel::new(
        parse_formula("y ~ x + (1 | g)").unwrap(),
        &data,
        family,
        Some(LinkFunction::Log),
    )
    .unwrap();
    model.fit_with_options(fast, 1, false).unwrap();
    let label = format!("{family:?} fast={fast}");
    assert_relative_eq!(
        model.loglikelihood() - 1.0,
        want.loglik,
        epsilon = 50.0 * tol
    );
    for (got, expected) in model.fixef().iter().zip(want.fixef) {
        assert!(
            (got - expected).abs() < tol,
            "{label}: fixef {got} vs {expected}"
        );
    }
    let theta = model.theta()[0];
    assert!(
        (theta - want.theta).abs() < tol,
        "{label}: theta {theta} vs {}",
        want.theta
    );
    let sigma = model.dispersion(false);
    assert!(
        (sigma - want.sigma).abs() < tol,
        "{label}: sigma {sigma} vs {}",
        want.sigma
    );
    // vcov() is the fixed-effect covariance on lme4's sigma() scale (the
    // fast path's Wald table is deliberately uncertified, so stderror()
    // reports NaN refusals there).
    let vcov = model.vcov();
    let se: Vec<f64> = (0..vcov.nrows()).map(|i| vcov[(i, i)].sqrt()).collect();
    for (got, expected) in se.iter().zip(want.se) {
        assert!(
            (got - expected).abs() < tol,
            "{label}: se {got} vs {expected}"
        );
    }
    // Residual SD, random-effect SD and vcov share one scale.
    let re_sd = model.varcorr().components[0].std_dev[0];
    assert_relative_eq!(re_sd, theta * sigma, max_relative = 1e-10);
}

#[test]
fn gamma_log_glmm_matches_lme4_nagq0_and_laplace() {
    check(
        Family::Gamma,
        true,
        Lme4 {
            loglik: -146.864034955,
            fixef: [0.5379763560, 0.1796992748],
            theta: 0.4808987,
            sigma: 0.4954208,
            se: [0.07635102, 0.07159531],
        },
        2e-4,
    );
    check(
        Family::Gamma,
        false,
        Lme4 {
            loglik: -146.850838195,
            fixef: [0.5251647064, 0.1719770404],
            theta: 0.4802132059,
            sigma: 0.4976379863,
            se: [0.10390939994, 0.06958738214],
        },
        5e-4,
    );
}

#[test]
fn inverse_gaussian_log_glmm_matches_lme4_nagq0_and_laplace() {
    check(
        Family::InverseGaussian,
        true,
        Lme4 {
            loglik: -150.918215401,
            fixef: [0.5157539429, 0.1819428100],
            theta: 0.5346403,
            sigma: 0.3910504,
            se: [0.07120974, 0.07258217],
        },
        5e-4,
    );
    // lme4's own optimum for this fit is ~1e-3 loose (its logLik is lower
    // than the engine's at the shared criterion); compare at 2e-3.
    check(
        Family::InverseGaussian,
        false,
        Lme4 {
            loglik: -150.904529833,
            fixef: [0.5264315024, 0.1724057892],
            theta: 0.536094,
            sigma: 0.3886201,
            se: [0.1073307, 0.07942297],
        },
        2e-3,
    );
}
