//! Parametric bootstrap helpers.
//!
//! The fitted-model side of the parametric bootstrap lives in
//! [`crate::model::linear::parametricbootstrap`]; this module re-exports the
//! result types and exposes the `shortest_cov_int` utility used to summarize
//! replicate distributions.

use std::io::{Read, Write};

use crate::error::{MixedModelError, Result};
use crate::model::generalized::GeneralizedLinearMixedModel;
pub use crate::model::linear::{
    parametricbootstrap, try_parametricbootstrap, BootstrapFailedRefitPolicy, BootstrapInterval,
    BootstrapIntervalMethod, BootstrapQuantile, BootstrapRefitOptions, BootstrapReplicate,
    BootstrapRunMetadata, BootstrapRunPayload, BootstrapSeedRecord, BootstrapTarget,
    BootstrapTargetKind, FixedEffectNullBootstrapTarget, FixedEffectNullCovariancePolicy,
    MixedModelBootstrap, BOOTSTRAP_RUN_SCHEMA, BOOTSTRAP_RUN_SCHEMA_VERSION,
};
use crate::model::linear::{FitProgressPhase, LinearMixedModel};
use crate::model::traits::Family;

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

/// GLMM parametric bootstrap.
///
/// For each replicate: simulate a response from the fitted model under a
/// fresh draw of the random effects (see
/// [`GeneralizedLinearMixedModel::simulate_response`]), refit a clone of the
/// template GLMM, and record its objective, dispersion, β, standard errors,
/// and θ. Supported for Bernoulli, Binomial, Poisson, fixed-theta
/// NegativeBinomial, and Gamma responses. InverseGaussian and Normal-as-GLM
/// are refused until they have certified family-specific response simulators.
///
/// A replicate whose refit fails numerically is recorded with `NaN`
/// objective/σ/SE (matching the LMM [`parametricbootstrap`] convention) so
/// downstream summaries can filter on finiteness.
pub fn parametricbootstrap_glmm<R: rand::Rng>(
    rng: &mut R,
    n_rep: usize,
    model: &GeneralizedLinearMixedModel,
) -> Result<MixedModelBootstrap> {
    use crate::model::traits::MixedModelFit;

    if !model.is_fitted() {
        return Err(MixedModelError::InvalidArgument(
            "GLMM parametric bootstrap requires a fitted model".to_string(),
        ));
    }

    match model.family {
        Family::Bernoulli
        | Family::Binomial
        | Family::Poisson
        | Family::NegativeBinomial
        | Family::Gamma => {}
        Family::InverseGaussian | Family::Normal => {
            return Err(MixedModelError::Unsupported(format!(
                "{:?} GLMM parametric bootstrap is not implemented; no certified \
                 family-specific response simulator is available",
                model.family
            )));
        }
    }

    let mut fits = Vec::with_capacity(n_rep);
    let mut last_progress = 0usize;
    for replicate in 0..n_rep {
        let y_sim = model.simulate_response(rng)?;
        let mut work = model.clone();
        match work.refit(&y_sim) {
            Ok(_) => {
                let beta = MixedModelFit::coef(&work);
                fits.push(BootstrapReplicate {
                    objective: work.objective(),
                    sigma: work.dispersion(false),
                    // Descriptive replicate SEs (finite for successful
                    // refits), not the certified-Wald `stderror` surface,
                    // which refuses with NaN for uncertified fits.
                    se: work.bootstrap_replicate_standard_errors(),
                    beta,
                    theta: work.theta(),
                });
            }
            Err(error @ MixedModelError::Interrupted(_)) => return Err(error),
            Err(_) => {
                let beta = MixedModelFit::coef(&work);
                fits.push(BootstrapReplicate {
                    objective: f64::NAN,
                    sigma: f64::NAN,
                    se: nalgebra::DVector::from_element(beta.len(), f64::NAN),
                    beta,
                    theta: work.theta(),
                });
            }
        }
        if let Some(callback) = &model.lmm.progress_callback {
            callback.report_if_due(
                FitProgressPhase::Bootstrap,
                replicate + 1,
                Some(n_rep),
                &mut last_progress,
            )?;
        }
    }

    Ok(MixedModelBootstrap { fits })
}

/// Shortest coverage interval containing `level` proportion of values.
///
/// Sorts `v` in place, then scans every contiguous window of size
/// `ceil(n * level)` over the *finite* values and returns the narrowest one.
/// Faithful port of MixedModels.jl `shortestcovint` (bootstrap.jl:473-486),
/// including its skip of non-finite elements at the sorted ends.
///
/// This is a public helper documented for summarizing bootstrap replicate
/// vectors (`boot.objectives()`, `boot.sigmas()`), which **deliberately
/// contain `NaN`** for replicates whose refit failed (see the `Err` arm of
/// the resampling loop above). It must therefore never panic on non-finite
/// input: sorting uses a total order ([`f64::total_cmp`], which ranks `NaN`
/// last just as Julia's `sort` does) rather than `partial_cmp().unwrap()`,
/// and non-finite ends are trimmed before the window scan. If there are
/// fewer than `ceil(n*level)` finite values the degenerate full-range
/// `(v[0], v[n-1])` is returned (Julia's fallback), never a crash.
pub fn shortest_cov_int(v: &mut [f64], level: f64) -> (f64, f64) {
    assert!(
        level > 0.0 && level < 1.0,
        "level must be in the open interval (0, 1)"
    );
    let n = v.len();
    if n == 0 {
        return (f64::NAN, f64::NAN);
    }
    // Total order so NaN cannot panic the comparator; NaN sorts to the end
    // (matching Julia `sort`'s `isless` placement of NaN).
    v.sort_by(f64::total_cmp);
    let ilen = ((n as f64) * level).ceil() as usize;

    // First/last finite index in the sorted slice; the non-finite tail (the
    // NaN failed-refit sentinels) is skipped (Julia: findfirst/findlast).
    let (Some(start), Some(stop)) = (
        v.iter().position(|x| x.is_finite()),
        v.iter().rposition(|x| x.is_finite()),
    ) else {
        // No finite replicates at all — degenerate fallback, not a panic.
        return (v[0], v[n - 1]);
    };
    // Not enough finite values to fill the interval
    // (Julia: `stop < start + ilen - 1`). `ilen >= 1` since level > 0.
    if ilen == 0 || stop < start + ilen - 1 {
        return (v[0], v[n - 1]);
    }

    let mut min_len = f64::INFINITY;
    let mut best_i = start;
    for i in start..=(stop + 1 - ilen) {
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
        GeneralizedLinearMixedModel::new(formula, &data, Family::Gamma, Some(LinkFunction::Log))
            .unwrap()
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
    fn test_shortest_cov_int_does_not_panic_on_nan_failed_refits() {
        // Regression for audit 05·H1 / mote bd-01KRXCQ93S101ZC87ZG5BF7HRJ:
        // failed bootstrap refits deliberately record f64::NAN in the
        // objective/sigma series this public helper is documented to
        // summarize. It must trim the non-finite tail (Julia shortestcovint),
        // not panic via partial_cmp().unwrap().
        let mut v = vec![10.0, f64::NAN, 11.0, 12.0, f64::NAN, 0.0, 100.0];
        let (lo, hi) = shortest_cov_int(&mut v, 0.6);
        // 5 finite values; ceil(7*0.6)=5 -> only one finite window
        // [0,10,11,12,100] so the interval spans the finite range.
        assert_eq!((lo, hi), (0.0, 100.0));
        assert!(lo.is_finite() && hi.is_finite());

        // Tighter level still ignores the NaNs and finds the tight cluster.
        let mut v2 = vec![f64::NAN, 0.0, 10.0, 11.0, 12.0, 100.0, f64::NAN];
        let (lo2, hi2) = shortest_cov_int(&mut v2, 0.4); // ceil(7*0.4)=3
        assert_eq!((lo2, hi2), (10.0, 12.0));
    }

    #[test]
    fn test_shortest_cov_int_all_nan_is_degenerate_not_panic() {
        let mut v = vec![f64::NAN, f64::NAN, f64::NAN];
        let (lo, hi) = shortest_cov_int(&mut v, 0.95);
        assert!(lo.is_nan() && hi.is_nan());
        // Empty input is also a non-panicking degenerate case.
        let mut empty: Vec<f64> = vec![];
        let (elo, ehi) = shortest_cov_int(&mut empty, 0.95);
        assert!(elo.is_nan() && ehi.is_nan());
    }

    fn poisson_glmm_fixture() -> GeneralizedLinearMixedModel {
        let mut y = Vec::new();
        let mut x = Vec::new();
        let mut g = Vec::new();
        for grp in 0..5 {
            for obs in 0..8 {
                let xv = obs as f64 - 3.5;
                let eta = 0.8 + 0.1 * xv + [-0.2, 0.1, 0.0, 0.15, -0.05][grp];
                y.push(eta.exp().round().max(1.0));
                x.push(xv);
                g.push(format!("g{}", grp + 1));
            }
        }
        let mut data = DataFrame::new();
        data.add_numeric("y", y).unwrap();
        data.add_numeric("x", x).unwrap();
        data.add_categorical("g", g).unwrap();
        let formula = parse_formula("y ~ 1 + x + (1 | g)").unwrap();
        let mut model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Poisson, None).unwrap();
        model.fit().unwrap();
        model
    }

    fn bernoulli_glmm_fixture() -> GeneralizedLinearMixedModel {
        let mut y = Vec::new();
        let mut x = Vec::new();
        let mut g = Vec::new();
        for grp in 0..6 {
            for obs in 0..8 {
                let xv = obs as f64 - 3.5;
                // Deterministic, non-separable binary pattern.
                let bit = ((grp + obs) % 2) as f64;
                y.push(bit);
                x.push(xv);
                g.push(format!("g{}", grp + 1));
            }
        }
        let mut data = DataFrame::new();
        data.add_numeric("y", y).unwrap();
        data.add_numeric("x", x).unwrap();
        data.add_categorical("g", g).unwrap();
        let formula = parse_formula("y ~ 1 + x + (1 | g)").unwrap();
        let mut model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();
        model.fit().unwrap();
        model
    }

    #[test]
    fn test_glmm_parametricbootstrap_requires_fitted_model() {
        let model = gamma_glmm_fixture();
        let mut rng = StdRng::seed_from_u64(99);
        let err = parametricbootstrap_glmm(&mut rng, 1, &model)
            .expect_err("GLMM parametric bootstrap requires a fitted template");

        match err {
            MixedModelError::InvalidArgument(msg) => {
                assert!(msg.contains("requires a fitted model"));
            }
            other => panic!("expected InvalidArgument error, got {other:?}"),
        }
    }

    #[test]
    fn test_poisson_glmm_parametricbootstrap_runs_and_is_seed_deterministic() {
        let model = poisson_glmm_fixture();

        let mut rng1 = StdRng::seed_from_u64(20260515);
        let boot1 = parametricbootstrap_glmm(&mut rng1, 12, &model)
            .expect("Poisson GLMM parametric bootstrap must be supported");
        assert_eq!(boot1.fits.len(), 12);

        // Most replicates refit successfully with finite β.
        let finite = boot1
            .fits
            .iter()
            .filter(|r| r.objective.is_finite() && r.beta.iter().all(|b| b.is_finite()))
            .count();
        assert!(
            finite >= 10,
            "expected most Poisson PB replicates to converge, got {finite}/12"
        );
        for r in &boot1.fits {
            assert_eq!(r.beta.len(), 2, "intercept + x");
            assert_eq!(r.theta.len(), 1, "one (1|g) variance component");
        }

        // Same seed → identical replicates.
        let mut rng2 = StdRng::seed_from_u64(20260515);
        let boot2 = parametricbootstrap_glmm(&mut rng2, 12, &model).unwrap();
        for (a, b) in boot1.fits.iter().zip(boot2.fits.iter()) {
            assert_eq!(a.beta.as_slice(), b.beta.as_slice());
            assert_eq!(a.theta, b.theta);
        }
    }

    #[test]
    fn test_bernoulli_glmm_parametricbootstrap_runs() {
        let model = bernoulli_glmm_fixture();
        let mut rng = StdRng::seed_from_u64(42);
        let boot = parametricbootstrap_glmm(&mut rng, 10, &model)
            .expect("Bernoulli GLMM parametric bootstrap must be supported");
        assert_eq!(boot.fits.len(), 10);
        let finite = boot
            .fits
            .iter()
            .filter(|r| r.beta.iter().all(|b| b.is_finite()))
            .count();
        assert!(
            finite >= 8,
            "expected most Bernoulli PB replicates to have finite β, got {finite}/10"
        );
    }

    #[test]
    fn test_glmm_simulate_response_respects_family_support() {
        // Bernoulli draws are 0/1.
        let bern = bernoulli_glmm_fixture();
        let mut rng = StdRng::seed_from_u64(7);
        let ys = bern.simulate_response(&mut rng).unwrap();
        assert!(ys.iter().all(|&v| v == 0.0 || v == 1.0));

        // Poisson draws are non-negative integers.
        let pois = poisson_glmm_fixture();
        let ys = pois.simulate_response(&mut rng).unwrap();
        assert!(ys.iter().all(|&v| v >= 0.0 && v.fract() == 0.0));

        // Gamma draws are strictly positive and dispersion-aware.
        let mut gamma = gamma_glmm_fixture();
        gamma.fit().unwrap();
        let ys = gamma.simulate_response(&mut rng).unwrap();
        assert!(ys.iter().all(|&v| v.is_finite() && v > 0.0));
    }

    #[test]
    fn test_gamma_glmm_parametricbootstrap_runs_with_positive_draws() {
        let mut model = gamma_glmm_fixture();
        model.fit().unwrap();
        let mut rng = StdRng::seed_from_u64(20260429);
        let boot = parametricbootstrap_glmm(&mut rng, 8, &model)
            .expect("Gamma GLMM parametric bootstrap must use positive family draws");

        assert_eq!(boot.fits.len(), 8);
        let finite = boot
            .fits
            .iter()
            .filter(|r| r.objective.is_finite() && r.beta.iter().all(|b| b.is_finite()))
            .count();
        assert!(
            finite >= 6,
            "expected most Gamma PB replicates to converge, got {finite}/8"
        );
        for r in &boot.fits {
            assert_eq!(r.beta.len(), 2, "intercept + x");
            assert_eq!(r.theta.len(), 1, "one (1|group) variance component");
        }
    }
}
