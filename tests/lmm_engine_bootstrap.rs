// Engine-level bootstrap tests migrated from src/model/linear/tests.rs
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
use nalgebra::DVector;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

// ── Parametric bootstrap parity tests (bootstrap.jl) ─────────────────────

#[test]
fn try_parametricbootstrap_propagates_host_interrupt() {
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let bootstrap_events = Arc::new(AtomicUsize::new(0));
    let callback_events = Arc::clone(&bootstrap_events);
    let callback = FitProgressCallback::new(move |progress| {
        if progress.phase == FitProgressPhase::Bootstrap {
            callback_events.fetch_add(1, Ordering::SeqCst);
            return Err(MixedModelError::Interrupted("test interrupt".to_string()));
        }
        Ok(())
    });
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model
        .fit_with_options(FitOptions::ml().with_progress_callback(callback))
        .unwrap();

    let mut rng = StdRng::seed_from_u64(1234321);
    let error = try_parametricbootstrap(&mut rng, 5, &model).unwrap_err();

    assert_eq!(error.code(), "interrupted");
    assert_eq!(bootstrap_events.load(Ordering::SeqCst), 1);
}

#[test]
fn test_parametricbootstrap_length() {
    // bootstrap.jl line 98: length(bsamp.objective) == 100
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let mut rng = StdRng::seed_from_u64(1234321);
    let bsamp = parametricbootstrap(&mut rng, 5, &model);
    assert_eq!(bsamp.len(), 5);
    assert_eq!(bsamp.objectives().len(), 5);
    assert_eq!(bsamp.sigmas().len(), 5);
    assert_eq!(bsamp.thetas().len(), 5);
}

#[test]
fn test_parametricbootstrap_objectives_finite() {
    // Each replicate should converge to a finite objective.
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let mut rng = StdRng::seed_from_u64(42);
    let bsamp = parametricbootstrap(&mut rng, 10, &model);

    let n_finite = bsamp
        .objectives()
        .iter()
        .filter(|&&o| o.is_finite())
        .count();
    assert!(
        n_finite >= 8,
        "At least 8 out of 10 replicates should converge; got {}",
        n_finite
    );
}

#[test]
fn test_parametricbootstrap_sigma_positive() {
    // All converged σ values should be positive.
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let mut rng = StdRng::seed_from_u64(99);
    let bsamp = parametricbootstrap(&mut rng, 5, &model);

    for rep in &bsamp.fits {
        if rep.sigma.is_finite() {
            assert!(
                rep.sigma > 0.0,
                "Bootstrap σ should be positive, got {}",
                rep.sigma
            );
        }
    }
}

#[test]
fn test_parametricbootstrap_theta_length() {
    // bootstrap.jl: keys(first(bsamp.fits)) includes :θ.
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let n_theta = model.n_theta();
    let mut rng = StdRng::seed_from_u64(0);
    let bsamp = parametricbootstrap(&mut rng, 3, &model);

    for rep in &bsamp.fits {
        assert_eq!(
            rep.theta.len(),
            n_theta,
            "Bootstrap θ length mismatch: expected {}, got {}",
            n_theta,
            rep.theta.len()
        );
    }
}

#[test]
fn test_parametricbootstrap_save_restore_round_trip() {
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let mut rng = StdRng::seed_from_u64(20260428);
    let bsamp = parametricbootstrap(&mut rng, 4, &model);

    let mut bytes = Vec::new();
    mixeff_rs::stats::savereplicates(&mut bytes, &bsamp).unwrap();
    let restored = mixeff_rs::stats::restorereplicates(bytes.as_slice(), &model).unwrap();

    assert_eq!(restored.len(), bsamp.len());
    for (actual, expected) in restored.fits.iter().zip(bsamp.fits.iter()) {
        assert_relative_eq!(actual.objective, expected.objective, epsilon = 1e-12);
        assert_relative_eq!(actual.sigma, expected.sigma, epsilon = 1e-12);
        assert_eq!(actual.beta.len(), expected.beta.len());
        for (a, e) in actual.beta.iter().zip(expected.beta.iter()) {
            assert_relative_eq!(*a, *e, epsilon = 1e-12);
        }
        assert_eq!(actual.se.len(), expected.se.len());
        for (a, e) in actual.se.iter().zip(expected.se.iter()) {
            assert_relative_eq!(*a, *e, epsilon = 1e-12);
        }
        assert_eq!(actual.theta.len(), expected.theta.len());
        for (a, e) in actual.theta.iter().zip(expected.theta.iter()) {
            assert_relative_eq!(*a, *e, epsilon = 1e-12);
        }
    }
}

#[test]
fn test_parametricbootstrap_save_restore_preserves_nan_status() {
    let bsamp = MixedModelBootstrap {
        fits: vec![BootstrapReplicate {
            objective: f64::NAN,
            sigma: f64::NAN,
            beta: DVector::from_vec(vec![1.0, 2.0]),
            se: DVector::from_vec(vec![f64::NAN, f64::NAN]),
            theta: vec![0.5],
        }],
    };

    let mut bytes = Vec::new();
    bsamp.save_replicates(&mut bytes).unwrap();
    let restored = MixedModelBootstrap::restore_replicates(bytes.as_slice()).unwrap();

    assert_eq!(restored.len(), 1);
    assert!(restored.fits[0].objective.is_nan());
    assert!(restored.fits[0].sigma.is_nan());
    assert_eq!(restored.fits[0].beta, DVector::from_vec(vec![1.0, 2.0]));
    assert!(restored.fits[0].se.iter().all(|value| value.is_nan()));
    assert_eq!(restored.fits[0].theta, vec![0.5]);
}

#[test]
fn test_parametricbootstrap_run_metadata_records_accounting_and_boundary_rate() {
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let bsamp = MixedModelBootstrap {
        fits: vec![
            BootstrapReplicate {
                objective: 1.0,
                sigma: 2.0,
                beta: DVector::from_vec(vec![10.0]),
                se: DVector::from_vec(vec![1.0]),
                theta: vec![0.0],
            },
            BootstrapReplicate {
                objective: 2.0,
                sigma: 3.0,
                beta: DVector::from_vec(vec![11.0]),
                se: DVector::from_vec(vec![1.2]),
                theta: vec![0.5],
            },
            BootstrapReplicate {
                objective: f64::NAN,
                sigma: f64::NAN,
                beta: DVector::from_vec(vec![f64::NAN]),
                se: DVector::from_vec(vec![f64::NAN]),
                theta: vec![0.5],
            },
        ],
    };
    let statistics = [1.0, f64::NAN, 3.0];

    let metadata = bsamp.run_metadata_for_model(
        &model,
        BootstrapTarget::full_model_distribution("dyestuff full model"),
        5,
        BootstrapFailedRefitPolicy::Exclude,
        BootstrapSeedRecord::std_rng(20260429),
        BootstrapRefitOptions::from_model(&model),
        Some("abs_t".to_string()),
        Some(&statistics),
        Some(0.25),
    );

    assert_eq!(metadata.schema_name, BOOTSTRAP_RUN_SCHEMA);
    assert_eq!(metadata.schema_version, BOOTSTRAP_RUN_SCHEMA_VERSION);
    assert_eq!(
        metadata.target.kind,
        BootstrapTargetKind::FullModelDistribution
    );
    assert_eq!(metadata.requested_replicates, 5);
    assert_eq!(metadata.completed_replicates, 3);
    assert_eq!(metadata.successful_replicates, 2);
    assert_eq!(metadata.failed_refits, 1);
    assert_eq!(
        metadata.failed_refit_policy,
        BootstrapFailedRefitPolicy::Exclude
    );
    assert_eq!(metadata.boundary_count, 1);
    assert_eq!(metadata.boundary_rate, Some(0.5));
    assert_eq!(metadata.finite_statistic_count, Some(2));
    assert_relative_eq!(metadata.mcse.unwrap(), (0.25_f64 * 0.75 / 2.0).sqrt());
    assert!(metadata
        .notes
        .iter()
        .any(|note| note.contains("do not certify fixed-effect hypothesis-test")));
    assert!(metadata
        .notes
        .iter()
        .any(|note| note.contains("requested 5 bootstrap")));

    let payload = bsamp.into_run_payload(metadata);
    let json = serde_json::to_string(&payload).unwrap();
    let decoded: BootstrapRunPayload = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded.metadata.successful_replicates, 2);
    assert_eq!(decoded.replicates.len(), 3);
}

#[test]
fn test_parametricbootstrap_quantile_summaries() {
    let bsamp = deterministic_bootstrap_sample();
    let rows = bsamp.quantiles(0.5).unwrap();

    let objective = rows
        .iter()
        .find(|row| row.parameter == "objective")
        .unwrap();
    assert_eq!(objective.n, 5);
    assert_eq!(objective.value, 30.0);

    let beta1 = rows.iter().find(|row| row.parameter == "beta[1]").unwrap();
    assert_eq!(beta1.value, 12.0);

    let se0 = rows.iter().find(|row| row.parameter == "se[0]").unwrap();
    assert_relative_eq!(se0.value, 0.7, epsilon = 1e-12);

    let theta0 = rows.iter().find(|row| row.parameter == "theta[0]").unwrap();
    assert_relative_eq!(theta0.value, 0.3, epsilon = 1e-12);
}

#[test]
fn test_parametricbootstrap_percentile_intervals() {
    let bsamp = deterministic_bootstrap_sample();
    let rows = bsamp.percentile_intervals(0.8).unwrap();

    let objective = rows
        .iter()
        .find(|row| row.parameter == "objective")
        .unwrap();
    assert_eq!(objective.method, BootstrapIntervalMethod::Percentile);
    assert_eq!(objective.n, 5);
    assert_relative_eq!(objective.lower, 14.0, epsilon = 1e-12);
    assert_relative_eq!(objective.upper, 46.0, epsilon = 1e-12);

    let sigma = rows.iter().find(|row| row.parameter == "sigma").unwrap();
    assert_relative_eq!(sigma.lower, 1.4, epsilon = 1e-12);
    assert_relative_eq!(sigma.upper, 4.6, epsilon = 1e-12);
}

#[test]
fn test_parametricbootstrap_shortest_intervals_filter_nonfinite() {
    let bsamp = MixedModelBootstrap {
        fits: vec![
            BootstrapReplicate {
                objective: f64::NAN,
                sigma: 0.0,
                beta: DVector::from_vec(vec![0.0]),
                se: DVector::from_vec(vec![0.0]),
                theta: vec![0.0],
            },
            BootstrapReplicate {
                objective: 10.0,
                sigma: 10.0,
                beta: DVector::from_vec(vec![10.0]),
                se: DVector::from_vec(vec![10.0]),
                theta: vec![10.0],
            },
            BootstrapReplicate {
                objective: 11.0,
                sigma: 11.0,
                beta: DVector::from_vec(vec![11.0]),
                se: DVector::from_vec(vec![11.0]),
                theta: vec![11.0],
            },
            BootstrapReplicate {
                objective: 12.0,
                sigma: 12.0,
                beta: DVector::from_vec(vec![12.0]),
                se: DVector::from_vec(vec![12.0]),
                theta: vec![12.0],
            },
            BootstrapReplicate {
                objective: 100.0,
                sigma: 100.0,
                beta: DVector::from_vec(vec![100.0]),
                se: DVector::from_vec(vec![100.0]),
                theta: vec![100.0],
            },
        ],
    };

    let rows = bsamp.shortest_intervals(0.6).unwrap();
    let objective = rows
        .iter()
        .find(|row| row.parameter == "objective")
        .unwrap();
    assert_eq!(objective.method, BootstrapIntervalMethod::Shortest);
    assert_eq!(objective.n, 4);
    assert_eq!((objective.lower, objective.upper), (10.0, 12.0));

    let sigma = rows.iter().find(|row| row.parameter == "sigma").unwrap();
    assert_eq!(sigma.n, 5);
    assert_eq!((sigma.lower, sigma.upper), (10.0, 12.0));
}

#[test]
fn test_parametricbootstrap_summaries_reject_bad_inputs() {
    let bsamp = deterministic_bootstrap_sample();
    assert!(matches!(
        bsamp.quantiles(1.2),
        Err(MixedModelError::InvalidArgument(_))
    ));
    assert!(matches!(
        bsamp.percentile_intervals(1.0),
        Err(MixedModelError::InvalidArgument(_))
    ));

    let mismatched = MixedModelBootstrap {
        fits: vec![
            BootstrapReplicate {
                objective: 1.0,
                sigma: 1.0,
                beta: DVector::from_vec(vec![1.0]),
                se: DVector::from_vec(vec![1.0]),
                theta: vec![1.0],
            },
            BootstrapReplicate {
                objective: 2.0,
                sigma: 2.0,
                beta: DVector::from_vec(vec![1.0, 2.0]),
                se: DVector::from_vec(vec![1.0]),
                theta: vec![1.0],
            },
        ],
    };
    assert!(matches!(
        mismatched.quantiles(0.5),
        Err(MixedModelError::InvalidArgument(_))
    ));
}

#[test]
fn test_parametricbootstrap_sigma_near_fitted() {
    // Over many replicates the mean bootstrap σ should be close to the
    // fitted σ (within 50% — bootstrap estimates have high variance for
    // small n, but the mean should be in the right ballpark).
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let fitted_sigma = model.sigma();

    let mut rng = StdRng::seed_from_u64(1234321);
    let bsamp = parametricbootstrap(&mut rng, 30, &model);

    let finite_sigmas: Vec<f64> = bsamp
        .sigmas()
        .into_iter()
        .filter(|s| s.is_finite())
        .collect();
    assert!(
        !finite_sigmas.is_empty(),
        "Should have at least one converged replicate"
    );

    let mean_sigma = finite_sigmas.iter().sum::<f64>() / finite_sigmas.len() as f64;
    let rel_err = ((mean_sigma - fitted_sigma) / fitted_sigma).abs();
    assert!(
        rel_err < 0.50,
        "Mean bootstrap σ {:.4} should be within 50% of fitted σ {:.4}",
        mean_sigma,
        fitted_sigma
    );
}
