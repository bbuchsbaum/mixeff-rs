//! Negative-binomial θ estimation against lme4::glmer.nb.
//!
//! glmer.nb maximizes the GLMM's Laplace log-likelihood over log θ_NB
//! (`optTheta`); the engine does the same after its conditional start, so θ,
//! the log-likelihood and the fixed effects agree. Reference values:
//! tests/fixtures/parity/nb_glmer_theta.provenance.json.

use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::data::DataFrame;
use mixeff_rs::model::generalized::GeneralizedLinearMixedModel;
use mixeff_rs::model::traits::MixedModelFit;

fn data() -> DataFrame {
    let text = include_str!("fixtures/parity/nb_glmer_theta.csv");
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

#[test]
fn estimated_nb_theta_matches_glmer_nb() {
    let data = data();
    let mut model = GeneralizedLinearMixedModel::new_negative_binomial_estimated(
        parse_formula("y ~ x + (1 | g)").unwrap(),
        &data,
        None,
        None,
    )
    .unwrap();
    model.fit_with_options(false, 1, false).unwrap();

    let theta_nb = model.negative_binomial_theta().unwrap();
    assert!((theta_nb - 2.015601784).abs() < 2e-3, "theta_nb {theta_nb}");
    let loglik = model.loglikelihood();
    assert!((loglik - -424.650373688).abs() < 1e-3, "logLik {loglik}");
    for (got, want) in model.fixef().iter().zip([0.8739943573, 0.3437253199]) {
        assert!((got - want).abs() < 1e-3, "fixef {got} vs {want}");
    }
    assert!((model.theta()[0] - 0.5117918).abs() < 2e-3);

    // The profiled fast path maximizes its own Laplace log-likelihood the
    // same way and lands on the same θ_NB to within its optimizer tolerance.
    let mut fast = GeneralizedLinearMixedModel::new_negative_binomial_estimated(
        parse_formula("y ~ x + (1 | g)").unwrap(),
        &data,
        None,
        None,
    )
    .unwrap();
    fast.fit_with_options(true, 1, false).unwrap();
    assert!((fast.negative_binomial_theta().unwrap() - 2.015601784).abs() < 0.02);
}
