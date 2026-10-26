use mixeff_rs::error::MixedModelError;
use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::linear::{parametricbootstrap_with_options, BootstrapExecutionOptions};
use mixeff_rs::model::{
    DataFrame, Family, FitOptions, FitProgressCallback, GeneralizedLinearMixedModel,
    GlmmFitOptions, LinearMixedModel, LinkFunction,
};
use mixeff_rs::stats::bootstrap::parametricbootstrap_glmm_with_options;
use rand::{rngs::StdRng, SeedableRng};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

fn data(counts: bool) -> DataFrame {
    let mut data = DataFrame::new();
    let x: Vec<f64> = (0..48).map(|i| (i % 8) as f64 - 3.5).collect();
    let y: Vec<f64> = (0..48)
        .map(|i| {
            if counts {
                (1 + (i * 7 + i / 8) % 5) as f64
            } else {
                2.0 + 0.4 * x[i] + (i / 8) as f64 * 0.3 + ((i * 7) % 11) as f64 * 0.13
            }
        })
        .collect();
    data.add_numeric("x", x).unwrap();
    data.add_numeric("y", y).unwrap();
    data.add_categorical("g", (0..48).map(|i| format!("g{}", i / 8)).collect())
        .unwrap();
    data
}

fn interrupt_on_host(calls: Arc<AtomicUsize>) -> FitProgressCallback {
    let host = std::thread::current().id();
    FitProgressCallback::new(move |_| {
        assert_eq!(std::thread::current().id(), host);
        calls.fetch_add(1, Ordering::SeqCst);
        Err(MixedModelError::Interrupted("host interrupt".into()))
    })
}

#[test]
fn restored_lmm_reattaches_interrupt_without_changing_fitted_state() {
    let formula = parse_formula("y ~ 1 + x + (1 | g)").unwrap();
    let mut fitted = LinearMixedModel::new(formula, &data(false), None).unwrap();
    fitted.fit_with_options(FitOptions::ml()).unwrap();
    let snapshot = fitted.snapshot_json().unwrap();
    for threads in [1, 2] {
        let mut restored = LinearMixedModel::restore_json(&snapshot).unwrap();
        let before = restored.snapshot_json().unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        restored.set_progress_callback(Some(interrupt_on_host(calls.clone())));
        assert_eq!(before, restored.snapshot_json().unwrap());
        let result = parametricbootstrap_with_options(
            &mut StdRng::seed_from_u64(42),
            2,
            &restored,
            &BootstrapExecutionOptions { threads },
        );
        assert!(matches!(result, Err(MixedModelError::Interrupted(_))));
        assert!(calls.load(Ordering::SeqCst) > 0);
        assert_eq!(before, restored.snapshot_json().unwrap());
        restored.set_progress_callback(None);
        let result = parametricbootstrap_with_options(
            &mut StdRng::seed_from_u64(42),
            2,
            &restored,
            &BootstrapExecutionOptions { threads },
        )
        .unwrap();
        assert_eq!(result.fits.len(), 2);
        assert_eq!(before, restored.snapshot_json().unwrap());
    }
}

#[test]
fn restored_glmm_reattaches_interrupt_without_changing_fitted_state() {
    let formula = parse_formula("y ~ 1 + x + (1 | g)").unwrap();
    let mut fitted = GeneralizedLinearMixedModel::new(
        formula,
        &data(true),
        Family::Poisson,
        Some(LinkFunction::Log),
    )
    .unwrap();
    fitted
        .fit_with_glmm_options(GlmmFitOptions::fast_laplace())
        .unwrap();
    let snapshot = fitted.snapshot_json().unwrap();
    for threads in [1, 2] {
        let mut restored = GeneralizedLinearMixedModel::restore_json(&snapshot).unwrap();
        let before = restored.snapshot_json().unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        restored.set_progress_callback(Some(interrupt_on_host(calls.clone())));
        assert_eq!(before, restored.snapshot_json().unwrap());
        let result = parametricbootstrap_glmm_with_options(
            &mut StdRng::seed_from_u64(42),
            2,
            &restored,
            &BootstrapExecutionOptions { threads },
        );
        assert!(matches!(result, Err(MixedModelError::Interrupted(_))));
        assert!(calls.load(Ordering::SeqCst) > 0);
        assert_eq!(before, restored.snapshot_json().unwrap());
        restored.set_progress_callback(None);
        let result = parametricbootstrap_glmm_with_options(
            &mut StdRng::seed_from_u64(42),
            2,
            &restored,
            &BootstrapExecutionOptions { threads },
        )
        .unwrap();
        assert_eq!(result.fits.len(), 2);
        assert_eq!(before, restored.snapshot_json().unwrap());
    }
}
