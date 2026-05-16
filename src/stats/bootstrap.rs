//! Parametric bootstrap helpers.
//!
//! The fitted-model side of the parametric bootstrap lives in
//! [`crate::model::linear::parametricbootstrap`]; this module re-exports the
//! result types and exposes the `shortest_cov_int` utility used to summarize
//! replicate distributions.

use std::io::{Read, Write};

use nalgebra::DVector;

use crate::error::{MixedModelError, Result};
use crate::model::generalized::GeneralizedLinearMixedModel;
use crate::model::linear::LinearMixedModel;
pub use crate::model::linear::{
    parametricbootstrap, BootstrapFailedRefitPolicy, BootstrapInterval, BootstrapIntervalMethod,
    BootstrapQuantile, BootstrapRefitOptions, BootstrapReplicate, BootstrapRunMetadata,
    BootstrapRunPayload, BootstrapSeedRecord, BootstrapTarget, BootstrapTargetKind,
    FixedEffectNullBootstrapTarget, FixedEffectNullCovariancePolicy, MixedModelBootstrap,
    BOOTSTRAP_RUN_SCHEMA, BOOTSTRAP_RUN_SCHEMA_VERSION,
};
use crate::model::traits::{Family, MixedModelFit};

/// Save bootstrap replicates as JSON.
///
/// Mirrors the Julia `savereplicates(io, pb)` surface while using a portable
/// JSON representation in Rust.
pub fn savereplicates<W: Write>(
    writer: W,
    bootstrap: &MixedModelBootstrap,
) -> std::result::Result<(), serde_json::Error> {
    bootstrap.save_replicates(writer)
}

/// Rust-style alias for [`savereplicates`].
pub fn save_replicates<W: Write>(
    writer: W,
    bootstrap: &MixedModelBootstrap,
) -> std::result::Result<(), serde_json::Error> {
    savereplicates(writer, bootstrap)
}

/// Restore bootstrap replicates from JSON and validate dimensions against `model`.
///
/// Mirrors Julia's `restorereplicates(io, model)` shape: the model is used as a
/// template guard so stale or mismatched replicate files are rejected up front.
pub fn restorereplicates<R: Read>(
    reader: R,
    model: &LinearMixedModel,
) -> Result<MixedModelBootstrap> {
    let bootstrap = MixedModelBootstrap::restore_replicates(reader).map_err(|err| {
        MixedModelError::InvalidArgument(format!("could not restore bootstrap replicates: {err}"))
    })?;
    bootstrap.validate_for_model(model)?;
    Ok(bootstrap)
}

/// Rust-style alias for [`restorereplicates`].
pub fn restore_replicates<R: Read>(
    reader: R,
    model: &LinearMixedModel,
) -> Result<MixedModelBootstrap> {
    restorereplicates(reader, model)
}

/// Run a full-model parametric bootstrap for a fitted GLMM.
///
/// The response simulation is family-specific and refits a cloned model for
/// every replicate, preserving the fitted model's offsets, case weights, AGQ
/// setting, optimizer options, and cold-start refit semantics.
pub fn parametricbootstrap_glmm<R: rand::Rng>(
    rng: &mut R,
    n_rep: usize,
    model: &GeneralizedLinearMixedModel,
) -> Result<MixedModelBootstrap> {
    if !model.is_fitted() {
        return Err(MixedModelError::InvalidArgument(
            "GLMM parametric bootstrap requires a fitted model".to_string(),
        ));
    }
    if matches!(model.family, Family::Normal | Family::InverseGaussian) {
        return Err(MixedModelError::Unsupported(format!(
            "{:?} GLMM parametric bootstrap is not implemented; no certified family-specific response simulator is available",
            model.family
        )));
    }

    let mut fits = Vec::with_capacity(n_rep);
    for _ in 0..n_rep {
        let y_sim = simulate_glmm_response(rng, model)?;
        let mut work = model.clone();
        match work.refit(&y_sim) {
            Ok(_) => {
                fits.push(BootstrapReplicate {
                    objective: work.objective(),
                    sigma: work.dispersion(false),
                    beta: work.fixef(),
                    se: work.stderror(),
                    theta: work.theta(),
                });
            }
            Err(_) => {
                fits.push(BootstrapReplicate {
                    objective: f64::NAN,
                    sigma: f64::NAN,
                    beta: model.fixef(),
                    se: DVector::from_element(model.fixef().len(), f64::NAN),
                    theta: model.theta(),
                });
            }
        }
    }

    Ok(MixedModelBootstrap { fits })
}

fn simulate_glmm_response<R: rand::Rng>(
    rng: &mut R,
    model: &GeneralizedLinearMixedModel,
) -> Result<Vec<f64>> {
    use rand_distr::{Bernoulli, Binomial, Distribution, Gamma as GammaDistribution, Poisson};

    let mu = model.fitted();
    if mu.len() != model.nobs() {
        return Err(MixedModelError::InvalidArgument(format!(
            "fitted mean length ({}) does not match number of observations ({})",
            mu.len(),
            model.nobs()
        )));
    }

    match model.family {
        Family::Bernoulli => mu
            .iter()
            .map(|&p| {
                let p = probability_for_draw(p)?;
                let draw = Bernoulli::new(p).map_err(|err| {
                    MixedModelError::InvalidArgument(format!(
                        "could not create Bernoulli bootstrap draw: {err}"
                    ))
                })?;
                Ok(if draw.sample(rng) { 1.0 } else { 0.0 })
            })
            .collect(),
        Family::Binomial => mu
            .iter()
            .enumerate()
            .map(|(obs, &p)| {
                let p = probability_for_draw(p)?;
                let trials = binomial_trials_for_observation(model, obs)?;
                let draw = Binomial::new(trials, p).map_err(|err| {
                    MixedModelError::InvalidArgument(format!(
                        "could not create Binomial bootstrap draw: {err}"
                    ))
                })?;
                Ok(draw.sample(rng) as f64 / trials as f64)
            })
            .collect(),
        Family::Poisson => mu
            .iter()
            .map(|&lambda| {
                let lambda = positive_mean_for_draw(lambda, "Poisson")?;
                let draw = Poisson::new(lambda).map_err(|err| {
                    MixedModelError::InvalidArgument(format!(
                        "could not create Poisson bootstrap draw: {err}"
                    ))
                })?;
                Ok(draw.sample(rng) as f64)
            })
            .collect(),
        Family::Gamma => {
            let phi = model.dispersion(true);
            if !phi.is_finite() || phi <= 0.0 {
                return Err(MixedModelError::InvalidArgument(format!(
                    "Gamma GLMM bootstrap requires positive finite phi = dispersion(true); got {phi}"
                )));
            }
            let shape = 1.0 / phi;
            mu.iter()
                .map(|&mean| {
                    let mean = positive_mean_for_draw(mean, "Gamma")?;
                    let scale = mean * phi;
                    let draw = GammaDistribution::new(shape, scale).map_err(|err| {
                        MixedModelError::InvalidArgument(format!(
                            "could not create Gamma bootstrap draw with shape = 1 / phi and scale = mu * phi: {err}"
                        ))
                    })?;
                    Ok(draw.sample(rng))
                })
                .collect()
        }
        Family::Normal | Family::InverseGaussian => Err(MixedModelError::Unsupported(format!(
            "{:?} GLMM parametric bootstrap is not implemented; no certified family-specific response simulator is available",
            model.family
        ))),
    }
}

fn probability_for_draw(p: f64) -> Result<f64> {
    if !p.is_finite() {
        return Err(MixedModelError::InvalidArgument(format!(
            "GLMM bootstrap probability is non-finite: {p}"
        )));
    }
    const TOL: f64 = 1e-12;
    if p < -TOL || p > 1.0 + TOL {
        return Err(MixedModelError::InvalidArgument(format!(
            "GLMM bootstrap probability must be in [0, 1]; got {p}"
        )));
    }
    Ok(p.clamp(0.0, 1.0))
}

fn positive_mean_for_draw(mean: f64, family: &str) -> Result<f64> {
    if !mean.is_finite() || mean < 0.0 {
        return Err(MixedModelError::InvalidArgument(format!(
            "{family} GLMM bootstrap mean must be finite and non-negative; got {mean}"
        )));
    }
    Ok(mean.max(f64::MIN_POSITIVE))
}

fn binomial_trials_for_observation(model: &GeneralizedLinearMixedModel, obs: usize) -> Result<u64> {
    let weight = model.wt.get(obs).copied().unwrap_or(1.0);
    if !weight.is_finite() || weight <= 0.0 {
        return Err(MixedModelError::InvalidArgument(format!(
            "Binomial GLMM bootstrap trial weight at observation {obs} must be positive and finite; got {weight}"
        )));
    }
    let rounded = weight.round();
    if (weight - rounded).abs() > 1e-8 {
        return Err(MixedModelError::Unsupported(format!(
            "Binomial GLMM parametric bootstrap requires integer trial weights; observation {obs} has weight {weight}"
        )));
    }
    let trials = rounded as u64;
    if trials == 0 {
        return Err(MixedModelError::InvalidArgument(format!(
            "Binomial GLMM bootstrap trial count at observation {obs} rounded to zero"
        )));
    }
    Ok(trials)
}

/// Shortest coverage interval containing `level` proportion of values.
///
/// Sorts `v` ascending in place, then scans every contiguous window of size
/// `ceil(n * level)` and returns the narrowest one. Mirrors the
/// `shortestcovint` summary helper used by the Julia bootstrap surface.
pub fn shortest_cov_int(v: &mut [f64], level: f64) -> (f64, f64) {
    assert!((0.0..1.0).contains(&level), "level must be in (0, 1)");
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    let ilen = ((n as f64) * level).ceil() as usize;
    if ilen >= n {
        return (v[0], v[n - 1]);
    }
    let mut min_len = f64::INFINITY;
    let mut best_i = 0;
    for i in 0..=(n - ilen) {
        let len = v[i + ilen - 1] - v[i];
        if len < min_len {
            min_len = len;
            best_i = i;
        }
    }
    (v[best_i], v[best_i + ilen - 1])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formula::parse_formula;
    use crate::model::data::DataFrame;
    use crate::model::traits::LinkFunction;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn bernoulli_glmm_fixture() -> GeneralizedLinearMixedModel {
        let mut data = DataFrame::new();
        data.add_numeric(
            "y",
            vec![0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
        )
        .unwrap();
        data.add_numeric(
            "x",
            vec![
                -1.2, -0.8, -0.4, 0.0, 0.4, 0.8, -1.0, -0.5, 0.2, 0.6, 1.0, 1.4,
            ],
        )
        .unwrap();
        data.add_categorical(
            "group",
            vec![
                "g1", "g1", "g1", "g2", "g2", "g2", "g3", "g3", "g3", "g4", "g4", "g4",
            ]
            .into_iter()
            .map(str::to_string)
            .collect(),
        )
        .unwrap();
        let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();
        let mut model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();
        model.lmm.optsum.max_feval = 80;
        model.fit_with_options(true, 1, false).unwrap();
        model
    }

    fn grouped_binomial_glmm_fixture() -> GeneralizedLinearMixedModel {
        let weights = vec![5.0, 8.0, 6.0, 10.0, 7.0, 9.0, 5.0, 8.0];
        let successes = [1.0, 4.0, 2.0, 7.0, 3.0, 6.0, 1.0, 5.0];
        let mut data = DataFrame::new();
        data.add_numeric(
            "prop",
            successes
                .iter()
                .zip(weights.iter())
                .map(|(success, weight)| success / weight)
                .collect(),
        )
        .unwrap();
        data.add_numeric("x", vec![-1.0, -0.7, -0.2, 0.1, 0.4, 0.8, 1.0, 1.3])
            .unwrap();
        data.add_categorical(
            "herd",
            vec!["h1", "h1", "h2", "h2", "h3", "h3", "h4", "h4"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        )
        .unwrap();
        let formula = parse_formula("prop ~ 1 + x + (1 | herd)").unwrap();
        let mut model = GeneralizedLinearMixedModel::new_with_weights(
            formula,
            &data,
            Family::Binomial,
            None,
            weights,
        )
        .unwrap();
        model.lmm.optsum.max_feval = 80;
        model.fit_with_options(true, 1, false).unwrap();
        model
    }

    fn gamma_glmm_fixture() -> GeneralizedLinearMixedModel {
        let mut data = DataFrame::new();
        data.add_numeric("y", vec![2.20, 2.72, 3.60, 4.37, 5.85, 2.61, 3.24, 4.10])
            .unwrap();
        data.add_numeric("x", vec![-2.0, -1.0, 0.0, 1.0, 2.0, -2.0, -1.0, 0.0])
            .unwrap();
        data.add_categorical(
            "group",
            vec!["g1", "g1", "g1", "g1", "g1", "g2", "g2", "g2"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        )
        .unwrap();
        let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();
        let mut model = GeneralizedLinearMixedModel::new(
            formula,
            &data,
            Family::Gamma,
            Some(LinkFunction::Log),
        )
        .unwrap();
        model.lmm.optsum.max_feval = 80;
        model.fit_with_options(true, 1, false).unwrap();
        model
    }

    fn poisson_offset_glmm_fixture() -> GeneralizedLinearMixedModel {
        let mut data = DataFrame::new();
        data.add_numeric("y", vec![0.0, 1.0, 2.0, 1.0, 3.0, 4.0, 1.0, 2.0])
            .unwrap();
        data.add_numeric("x", vec![-1.0, -0.5, 0.0, 0.4, 0.8, 1.2, -0.8, 0.6])
            .unwrap();
        data.add_categorical(
            "site",
            vec!["s1", "s1", "s2", "s2", "s3", "s3", "s4", "s4"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        )
        .unwrap();
        let offset = vec![1.0_f64, 1.2, 0.8, 1.5, 2.0, 1.7, 0.9, 1.1]
            .into_iter()
            .map(f64::ln)
            .collect();
        let formula = parse_formula("y ~ 1 + x + (1 | site)").unwrap();
        let mut model = GeneralizedLinearMixedModel::new_with_offset(
            formula,
            &data,
            Family::Poisson,
            None,
            offset,
        )
        .unwrap();
        model.lmm.optsum.max_feval = 80;
        model.fit_with_options(true, 1, false).unwrap();
        model
    }

    #[test]
    fn test_shortest_cov_int_narrow_window() {
        let mut v: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let (lo, hi) = shortest_cov_int(&mut v, 0.95);
        // 95-element window over uniformly-spaced integers spans at most 94.
        assert!(hi - lo <= 95.0);
        assert!(lo >= 0.0 && hi <= 99.0);
    }

    #[test]
    fn test_shortest_cov_int_full_coverage() {
        let mut v = vec![1.0, 5.0, 7.0];
        let (lo, hi) = shortest_cov_int(&mut v, 0.99);
        // ceil(3 * 0.99) == 3, so the only window is the full vector.
        assert_eq!((lo, hi), (1.0, 7.0));
    }

    #[test]
    fn test_shortest_cov_int_picks_tightest_cluster() {
        // Tight cluster at [10, 11, 12] vs. spread elsewhere: with level=0.6
        // (ceil(5 * 0.6) = 3) the tight cluster wins.
        let mut v = vec![0.0, 10.0, 11.0, 12.0, 100.0];
        let (lo, hi) = shortest_cov_int(&mut v, 0.6);
        assert_eq!((lo, hi), (10.0, 12.0));
    }

    #[test]
    fn test_glmm_parametricbootstrap_is_seed_reproducible() {
        let model = bernoulli_glmm_fixture();
        let mut rng_a = StdRng::seed_from_u64(20260429);
        let mut rng_b = StdRng::seed_from_u64(20260429);

        let a = parametricbootstrap_glmm(&mut rng_a, 3, &model).unwrap();
        let b = parametricbootstrap_glmm(&mut rng_b, 3, &model).unwrap();

        assert_eq!(a, b);
        assert_eq!(a.len(), 3);
        for fit in a.fits {
            assert_eq!(fit.beta.len(), model.fixef().len());
            assert_eq!(fit.se.len(), model.fixef().len());
            assert_eq!(fit.theta.len(), model.theta().len());
        }
    }

    #[test]
    fn test_binomial_glmm_bootstrap_respects_trial_weights() {
        let model = grouped_binomial_glmm_fixture();
        let mut rng = StdRng::seed_from_u64(20260430);

        let y = simulate_glmm_response(&mut rng, &model).unwrap();
        assert_eq!(y.len(), model.nobs());
        for (obs, value) in y.iter().enumerate() {
            let trials = model.wt[obs];
            let successes = value * trials;
            assert!(successes >= 0.0 && successes <= trials);
            assert!(
                (successes - successes.round()).abs() < 1e-8,
                "binomial bootstrap response should be successes/trials; got {successes}/{trials}"
            );
        }

        let bsamp = parametricbootstrap_glmm(&mut rng, 2, &model).unwrap();
        assert_eq!(bsamp.len(), 2);
    }

    #[test]
    fn test_poisson_glmm_bootstrap_preserves_offset_refit_path() {
        let model = poisson_offset_glmm_fixture();
        let mut rng = StdRng::seed_from_u64(20260501);
        let original_offset = model.offset.clone();

        let bsamp = parametricbootstrap_glmm(&mut rng, 2, &model).unwrap();

        assert_eq!(bsamp.len(), 2);
        assert_eq!(model.offset, original_offset);
        assert!(bsamp
            .fits
            .iter()
            .all(|fit| fit.objective.is_finite() && fit.beta.len() == model.fixef().len()));
    }

    #[test]
    fn test_gamma_glmm_parametricbootstrap_uses_positive_gamma_draws() {
        let model = gamma_glmm_fixture();
        let mut rng = StdRng::seed_from_u64(20260429);
        let y = simulate_glmm_response(&mut rng, &model).unwrap();
        assert!(y.iter().all(|value| value.is_finite() && *value > 0.0));
        assert!(
            y.iter().any(|value| (value - value.round()).abs() > 1e-6),
            "Gamma bootstrap should use continuous Gamma draws, not Gaussian residual or count draws"
        );

        let bsamp = parametricbootstrap_glmm(&mut rng, 2, &model).unwrap();
        assert_eq!(bsamp.len(), 2);
        assert!(bsamp
            .fits
            .iter()
            .all(|fit| fit.sigma.is_finite() && fit.sigma > 0.0));
    }
}
