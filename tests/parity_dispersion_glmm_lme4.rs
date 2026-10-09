//! Estimated-dispersion GLMMs (Gamma, inverse Gaussian, Gaussian with a log
//! link) against lme4 2.1-0's `glmer`.
//!
//! lme4 2.1 profiles the dispersion φ inside PIRLS (`glmerControl(disp_method
//! = "moment", disp_dof_correction = TRUE)`): the working weights carry 1/φ,
//! θ is the absolute random-effect SD on the link scale (VarCorr is θ, not
//! θσ), `sigma() = sqrt(φ)` with φ = deviance / (n − rank([X, Z])), and the
//! logLik is the plain Laplace / AGQ value (the family `aic()`'s `+2` is no
//! longer added). The engine's default
//! [`GlmmDispersionMethod::Moment`] reproduces all of that;
//! [`GlmmDispersionMethod::Legacy`] reproduces `disp_method = "old/buggy"`.
//!
//! lme4's `vcov()` for an `nAGQ = 0` fit is `sigma()^2 * unsc()`, but under
//! the moment method `unsc()` is already φ-weighted, so that double-counts φ
//! (lme4 2.1's own `use.hessian` check warns that it disagrees with the
//! Laplace Hessian by a factor ≈ σ²). The engine reports `unsc()` itself;
//! the test pins both facts.
//!
//! Reference values: tests/fixtures/parity/dispersion_glmm_lme4_2_1.json
//! (scripts/regenerate_lme4_dispersion_glmm_fixtures.R).

use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::data::DataFrame;
use mixeff_rs::model::generalized::GeneralizedLinearMixedModel;
use mixeff_rs::model::traits::{Family, LinkFunction, MixedModelFit};
use mixeff_rs::model::GlmmDispersionMethod;
use serde::Deserialize;

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

#[derive(Deserialize)]
struct Fixture {
    schema_version: String,
    lme4_version: String,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct Case {
    family: String,
    n_agq: usize,
    disp_method: String,
    disp_dof_correction: bool,
    fixef: Vec<f64>,
    theta: f64,
    sigma: f64,
    phi: f64,
    #[serde(rename = "logLik")]
    loglik: f64,
    deviance: f64,
    #[serde(rename = "AIC")]
    aic: f64,
    varcorr_sd: f64,
    vcov: Vec<Vec<f64>>,
    vcov_rx: Vec<Vec<f64>>,
    unscaled_rx: Vec<Vec<f64>>,
}

fn fixture() -> Fixture {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "fixtures/parity/dispersion_glmm_lme4_2_1.json"
    ))
    .unwrap();
    assert_eq!(fixture.schema_version, "1.0.0");
    assert!(fixture.lme4_version.starts_with("2.1"));
    fixture
}

fn family(name: &str) -> Family {
    match name {
        "gamma" => Family::Gamma,
        "inverse_gaussian" => Family::InverseGaussian,
        "gaussian_log" => Family::Normal,
        other => panic!("unknown family {other}"),
    }
}

fn unit_deviance(family: Family, y: f64, mu: f64) -> f64 {
    match family {
        Family::Gamma => -2.0 * ((y / mu).ln() - (y - mu) / mu),
        Family::InverseGaussian => (y - mu).powi(2) / (y * mu * mu),
        Family::Normal => (y - mu).powi(2),
        _ => unreachable!(),
    }
}

/// Per-quantity absolute tolerances for one row.
struct Tolerances {
    loglik: f64,
    fixef: f64,
    theta: f64,
    sigma: f64,
    se: f64,
}

fn tolerances(case: &Case) -> Tolerances {
    if case.disp_method != "moment" {
        // lme4's old/buggy Laplace optimum is ~2e-4 loose (the engine's
        // logLik is higher there); the profiled engine path lands 3e-4 below.
        return Tolerances {
            loglik: 5e-4,
            fixef: 2e-4,
            theta: 2e-4,
            sigma: 5e-5,
            se: 2e-4,
        };
    }
    if case.n_agq == 0 {
        // The profiled θ search stops on a 1e-8 absolute objective change,
        // and its inner PIRLS on a 1e-5 one (both MixedModels.jl defaults);
        // without NLopt the derivative-free fallback stops on a θ grid of
        // ~1e-3 where the criterion is flat.
        Tolerances {
            loglik: 3e-4,
            fixef: 5e-4,
            theta: 3e-3,
            sigma: 5e-4,
            se: 5e-4,
        }
    } else {
        Tolerances {
            loglik: 1e-4,
            fixef: 2e-4,
            theta: 2e-4,
            sigma: 2e-5,
            se: 2e-4,
        }
    }
}

fn fit(case: &Case) -> GeneralizedLinearMixedModel {
    let family = family(&case.family);
    let mut model = GeneralizedLinearMixedModel::new(
        parse_formula("y ~ x + (1 | g)").unwrap(),
        &data(),
        family,
        Some(LinkFunction::Log),
    )
    .unwrap();
    assert_eq!(model.dispersion_method(), GlmmDispersionMethod::Moment);
    match case.disp_method.as_str() {
        "moment" => {}
        "old/buggy" => {
            model.set_dispersion_method(GlmmDispersionMethod::Legacy);
        }
        other => panic!("unknown disp_method {other}"),
    }
    model.set_dispersion_dof_correction(case.disp_dof_correction);
    let (fast, n_agq) = match case.n_agq {
        0 => (true, 1),
        n => (false, n),
    };
    model.fit_with_options(fast, n_agq, false).unwrap();
    model
}

fn check(case: &Case) {
    let label = format!(
        "{} nAGQ={} disp_method={} dof={}",
        case.family, case.n_agq, case.disp_method, case.disp_dof_correction
    );
    let tol = tolerances(case);
    let model = fit(case);
    let family = family(&case.family);

    let close = |what: &str, got: f64, want: f64, tol: f64| {
        assert!(
            (got - want).abs() <= tol,
            "{label}: {what} {got} vs lme4 {want} (|diff| {:.3e} > {tol:.1e})",
            (got - want).abs()
        );
    };
    close("logLik", model.loglikelihood(), case.loglik, tol.loglik);
    close("AIC", model.aic(), case.aic, 2.0 * tol.loglik);
    for (got, want) in model.fixef().iter().zip(&case.fixef) {
        close("fixef", *got, *want, tol.fixef);
    }
    close("theta", model.theta()[0], case.theta, tol.theta);
    close("sigma", model.dispersion(false), case.sigma, tol.sigma);
    if case.disp_method == "moment" {
        // resp$phi() stays 1 under old/buggy.
        close(
            "phi",
            model.dispersion(true),
            case.phi,
            2.0 * case.sigma * tol.sigma,
        );
    }
    let y = model.response();
    let deviance: f64 = model
        .fitted()
        .iter()
        .zip(y.iter())
        .map(|(&mu, &y)| unit_deviance(family, y, mu))
        .sum();
    close(
        "deviance",
        deviance,
        case.deviance,
        50.0 * tol.fixef * case.deviance,
    );
    // lme4 2.1's VarCorr is θ for every GLMM. Legacy keeps the pre-2.1
    // engine (and lme4 <= 2.0) convention θ·σ, since its θ is relative to σ.
    let varcorr_sd = if case.disp_method == "moment" {
        case.varcorr_sd
    } else {
        case.theta * case.sigma
    };
    close(
        "VarCorr sd",
        model.varcorr().components[0].std_dev[0],
        varcorr_sd,
        tol.theta * case.sigma.max(1.0),
    );

    // Fixed-effect covariance: the Laplace / AGQ Hessian for nAGQ >= 1
    // (lme4's default vcov), the PIRLS RX for nAGQ = 0.
    let vcov = model.vcov();
    let want = if case.n_agq > 1 {
        // The engine's AGQ fits report the PIRLS RX covariance (no AGQ
        // Hessian); lme4 differentiates the AGQ deviance numerically.
        &case.unscaled_rx
    } else if case.n_agq == 0 && case.disp_method == "moment" {
        // lme4 reports sigma^2 * unsc() here; see the module docs.
        let s2 = case.sigma * case.sigma;
        for (row_v, row_u) in case.vcov.iter().zip(&case.unscaled_rx) {
            for (v, u) in row_v.iter().zip(row_u) {
                assert!((v - s2 * u).abs() <= 1e-10 * v.abs().max(1e-12));
            }
        }
        &case.unscaled_rx
    } else {
        &case.vcov
    };
    for i in 0..2 {
        close("se", vcov[(i, i)].sqrt(), want[i][i].sqrt(), tol.se);
    }
}

fn run(family: &str, n_agq: usize) {
    let fixture = fixture();
    let cases: Vec<&Case> = fixture
        .cases
        .iter()
        .filter(|case| case.family == family && case.n_agq == n_agq)
        .collect();
    assert!(!cases.is_empty());
    for case in cases {
        check(case);
    }
}

#[test]
fn gamma_log_glmm_matches_lme4_2_1_nagq0() {
    run("gamma", 0);
}

#[test]
fn gamma_log_glmm_matches_lme4_2_1_laplace() {
    run("gamma", 1);
}

#[test]
fn gamma_log_glmm_matches_lme4_2_1_agq() {
    run("gamma", 5);
}

#[test]
fn inverse_gaussian_log_glmm_matches_lme4_2_1_nagq0() {
    run("inverse_gaussian", 0);
}

#[test]
fn inverse_gaussian_log_glmm_matches_lme4_2_1_laplace() {
    run("inverse_gaussian", 1);
}

#[test]
fn inverse_gaussian_log_glmm_matches_lme4_2_1_agq() {
    run("inverse_gaussian", 5);
}

#[test]
fn gaussian_log_glmm_matches_lme4_2_1_nagq0() {
    run("gaussian_log", 0);
}

#[test]
fn gaussian_log_glmm_matches_lme4_2_1_laplace() {
    run("gaussian_log", 1);
}

#[test]
fn gaussian_log_glmm_matches_lme4_2_1_agq() {
    run("gaussian_log", 5);
}
