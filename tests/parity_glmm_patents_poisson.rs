//! Regression: joint Laplace Poisson GLMM on the m-clark patents data.
//!
//! `ncit ~ opposition + biopharm + (1 | year)` failed on macOS arm64 with
//! "joint GLMM starting conditional-mode solve did not converge" while it
//! passed on x86-64. Near the conditional mode a fixed-β Newton step lowers
//! the PIRLS criterion by about score²/2, far below the rounding error of
//! that 4809-term sum; the objective-only line search then accepted or
//! rejected the step on platform rounding, and a rejection left the solve
//! stalled above its score tolerance at the fast-PIRLS start. Perturbing a
//! covariate by a few parts in 1e4 reproduced the failure on x86-64, so the
//! robustness sweep below exercises the same coin toss deterministically.
//!
//! Reference: lme4 2.1.0 `glmer(..., family = poisson)` (default nAGQ = 1);
//! see `mclark_patents_poisson.provenance.json`.

use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::traits::MixedModelFit;
use mixeff_rs::model::{DataFrame, Family, GeneralizedLinearMixedModel, LinkFunction};

const GLMER_THETA: f64 = 0.361148387865;
const GLMER_FIXEF: [f64; 3] = [0.222636556916, 0.384909147589, 0.126412412449];
const GLMER_LOGLIK: f64 = -9588.3923326;

fn build_model(opposition_scale: f64) -> GeneralizedLinearMixedModel {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/regression/mclark_patents_poisson.csv"
    );
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let (mut ncit, mut opposition, mut biopharm, mut year) = (vec![], vec![], vec![], vec![]);
    for line in text.lines().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        assert_eq!(fields.len(), 4, "malformed row {line:?}");
        ncit.push(fields[0].parse::<f64>().unwrap());
        opposition.push(fields[1].parse::<f64>().unwrap() * opposition_scale);
        biopharm.push(fields[2].parse::<f64>().unwrap());
        year.push(fields[3].trim_matches('"').to_string());
    }
    assert_eq!(ncit.len(), 4809);

    let mut df = DataFrame::new();
    df.add_numeric("ncit", ncit).unwrap();
    df.add_numeric("opposition", opposition).unwrap();
    df.add_numeric("biopharm", biopharm).unwrap();
    df.add_categorical("year", year).unwrap();
    let formula = parse_formula("ncit ~ opposition + biopharm + (1 | year)").unwrap();
    GeneralizedLinearMixedModel::new(formula, &df, Family::Poisson, Some(LinkFunction::Log))
        .unwrap()
}

#[test]
fn patents_poisson_joint_laplace_matches_glmer() {
    let mut model = build_model(1.0);
    model
        .fit_with_options(false, 1, false)
        .expect("joint Laplace Poisson GLMM fit");

    let return_value = &model.opt_summary().return_value;
    assert!(
        return_value.starts_with("JOINT_LAPLACE:"),
        "expected a joint-Laplace result, got {return_value:?}"
    );

    let theta = model.theta();
    assert!(
        (theta[0] - GLMER_THETA).abs() < 1e-4,
        "theta {theta:?} vs glmer {GLMER_THETA}"
    );
    let fixef = model.coef();
    for (index, (got, want)) in fixef.iter().zip(GLMER_FIXEF).enumerate() {
        assert!(
            (got - want).abs() < 1e-4,
            "fixef[{index}] {got} vs glmer {want}"
        );
    }
    let loglik = model.loglikelihood();
    assert!(
        (loglik - GLMER_LOGLIK).abs() < 1e-5,
        "logLik {loglik} vs glmer {GLMER_LOGLIK}"
    );
}

/// Covariate scalings at which the pre-fix engine failed the starting
/// conditional-mode solve on x86-64 (8795083 and 5db36cd), so the guard does
/// not depend on reproducing arm64 rounding.
#[test]
fn patents_poisson_joint_start_survives_rounding_perturbations() {
    for scale in [
        0.998079, 0.998504, 0.99864, 0.999592, 0.999609, 0.999983, 1.000187, 1.001326, 1.002,
    ] {
        let mut model = build_model(scale);
        model
            .fit_with_options(false, 1, false)
            .unwrap_or_else(|error| panic!("opposition scaled by {scale}: {error}"));
        let loglik = model.loglikelihood();
        assert!(
            (loglik - GLMER_LOGLIK).abs() < 1e-5,
            "opposition scaled by {scale}: logLik {loglik}"
        );
    }
}
