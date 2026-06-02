use approx::assert_relative_eq;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::Deserialize;
use serde_json::Value;

use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::data::DataFrame;
use mixeff_rs::model::generalized::GeneralizedLinearMixedModel;
use mixeff_rs::model::traits::{Family, LinkFunction, MixedModelFit};
use mixeff_rs::stats::bootstrap::parametricbootstrap_glmm;
#[cfg(not(feature = "nlopt"))]
use mixeff_rs::types::Optimizer;

#[allow(dead_code)]
#[derive(Deserialize)]
struct GammaGlmmFixture {
    schema_version: String,
    source: String,
    formula: String,
    family: String,
    link: String,
    n_agq: usize,
    nobs: usize,
    dof: usize,
    data_recipe: DataRecipe,
    rust_reference: FitReference,
    engines: Vec<EngineReference>,
    notes: Vec<String>,
}

#[derive(Deserialize)]
struct DataRecipe {
    groups: usize,
    observations_per_group: usize,
    intercept: f64,
    slope: f64,
    group_effects: Vec<f64>,
    wiggle_base: f64,
    wiggle_step: f64,
    wiggle_modulus: usize,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct FitReference {
    beta: Vec<f64>,
    theta: Vec<f64>,
    dispersion_sigma: f64,
    dispersion_phi: f64,
    objective: f64,
    loglik: f64,
    fitted_mu_head: Vec<f64>,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct EngineReference {
    engine: String,
    status: String,
    version: Option<String>,
    beta: Option<Vec<f64>>,
    theta: Option<Vec<f64>>,
    dispersion: Option<f64>,
    objective: Option<f64>,
    loglik: Option<f64>,
    verdict: String,
    note: String,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct GammaIssue643Fixture {
    schema_version: String,
    source: String,
    issue_url: String,
    formula: String,
    family: String,
    link: String,
    n_agq: usize,
    nobs: usize,
    data_recipe: Issue643Recipe,
    data: Issue643Data,
    observed_summary: DistributionSummary,
    engines: Vec<Issue643EngineReference>,
    stress_contract: Issue643StressContract,
    notes: Vec<String>,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct Issue643Recipe {
    r_seed: u64,
    n: usize,
    y_distribution: String,
    group_distribution: String,
    groups: usize,
}

#[derive(Deserialize)]
struct Issue643Data {
    y: Vec<f64>,
    group: Vec<usize>,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct Issue643EngineReference {
    engine: String,
    status: String,
    version: String,
    beta: Value,
    theta: Value,
    theta_scale: String,
    random_effect_sd: Option<f64>,
    dispersion_phi: Option<f64>,
    objective: Option<f64>,
    loglik: Option<f64>,
    simulation_summary: Option<DistributionSummary>,
    verdict: String,
    note: String,
}

#[allow(dead_code)]
#[derive(Clone, Deserialize)]
struct DistributionSummary {
    mean: f64,
    sd: f64,
    min: f64,
    max: f64,
    q50: f64,
    q90: f64,
    q95: f64,
    q99: f64,
}

#[derive(Deserialize)]
struct Issue643StressContract {
    max_reasonable_sim_q99: f64,
    max_reasonable_sim_max: f64,
    max_reasonable_bootstrap_mean_ratio: f64,
    min_glmer_to_glmmtmb_theta_ratio: f64,
}

fn fixture() -> GammaGlmmFixture {
    serde_json::from_str(include_str!("fixtures/parity/gamma_glmm_engines.json")).unwrap()
}

fn lme4_issue_643_fixture() -> GammaIssue643Fixture {
    serde_json::from_str(include_str!("fixtures/parity/gamma_glmm_lme4_643.json")).unwrap()
}

fn issue_643_engine<'a>(
    fixture: &'a GammaIssue643Fixture,
    engine_name: &str,
) -> &'a Issue643EngineReference {
    fixture
        .engines
        .iter()
        .find(|engine| engine.engine == engine_name)
        .unwrap_or_else(|| panic!("fixture records {engine_name} reference"))
}

fn scalar_or_first(value: &Value) -> f64 {
    value
        .as_f64()
        .or_else(|| value.as_array().and_then(|values| values.first()?.as_f64()))
        .expect("numeric scalar or first array element")
}

fn summarize(values: &[f64]) -> DistributionSummary {
    assert!(!values.is_empty());
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let sd = if values.len() > 1 {
        let ss = values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>();
        (ss / (values.len() - 1) as f64).sqrt()
    } else {
        0.0
    };
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite sorted values"));
    DistributionSummary {
        mean,
        sd,
        min: sorted[0],
        max: sorted[sorted.len() - 1],
        q50: empirical_quantile(&sorted, 0.50),
        q90: empirical_quantile(&sorted, 0.90),
        q95: empirical_quantile(&sorted, 0.95),
        q99: empirical_quantile(&sorted, 0.99),
    }
}

fn empirical_quantile(sorted: &[f64], p: f64) -> f64 {
    let h = (sorted.len() - 1) as f64 * p;
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = h - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

fn lme4_issue_643_data(fixture: &GammaIssue643Fixture) -> DataFrame {
    assert_eq!(fixture.data.y.len(), fixture.nobs);
    assert_eq!(fixture.data.group.len(), fixture.nobs);

    let mut data = DataFrame::new();
    data.add_numeric("y", fixture.data.y.clone()).unwrap();
    data.add_categorical(
        "grp",
        fixture
            .data
            .group
            .iter()
            .map(|group| format!("g{group}"))
            .collect(),
    )
    .unwrap();
    data
}

// toy: matches `fixtures/parity/gamma_glmm_engines.json`; row order is
// part of the parity assertion (see `reversed_gamma_log_data` invariance test).
fn gamma_log_data() -> DataFrame {
    let expected = fixture();
    let recipe = expected.data_recipe;

    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut group = Vec::new();
    for g in 0..recipe.groups {
        for obs in 0..recipe.observations_per_group {
            let xv = obs as f64 - 2.0;
            let eta = recipe.intercept + recipe.slope * xv + recipe.group_effects[g];
            let wiggle = recipe.wiggle_base
                + recipe.wiggle_step * ((g + obs) % recipe.wiggle_modulus) as f64;
            y.push(eta.exp() * wiggle);
            x.push(xv);
            group.push(format!("g{}", g + 1));
        }
    }

    let mut data = DataFrame::new();
    data.add_numeric("y", y).unwrap();
    data.add_numeric("x", x).unwrap();
    data.add_categorical("group", group).unwrap();
    data
}

// toy: row-reversed `gamma_log_data`; tests fit-invariance to row order.
fn reversed_gamma_log_data() -> DataFrame {
    let data = gamma_log_data();
    let mut indices = (0..data.nrow()).collect::<Vec<_>>();
    indices.reverse();

    let mut reversed = DataFrame::new();
    reversed
        .add_numeric(
            "y",
            indices
                .iter()
                .map(|&idx| data.numeric("y").unwrap()[idx])
                .collect(),
        )
        .unwrap();
    reversed
        .add_numeric(
            "x",
            indices
                .iter()
                .map(|&idx| data.numeric("x").unwrap()[idx])
                .collect(),
        )
        .unwrap();
    reversed
        .add_categorical(
            "group",
            indices
                .iter()
                .map(|&idx| data.categorical("group").unwrap().values[idx].clone())
                .collect(),
        )
        .unwrap();
    reversed
}

fn fit_gamma_log(data: &DataFrame, formula: &str, n_agq: usize) -> GeneralizedLinearMixedModel {
    let formula = parse_formula(formula).unwrap();
    let mut model =
        GeneralizedLinearMixedModel::new(formula, data, Family::Gamma, Some(LinkFunction::Log))
            .unwrap();
    model.fit_with_options(true, n_agq, false).unwrap();
    model
}

#[cfg(feature = "nlopt")]
#[test]
fn test_gamma_log_glmm_matches_mixedmodels_jl_fixture() {
    let expected = fixture();
    assert_eq!(expected.schema_version, "1.0.0");
    assert!(expected.source.contains("MixedModels.jl"));
    assert_eq!(expected.formula, "y ~ 1 + x + (1 | group)");
    assert_eq!(expected.family, "gamma");
    assert_eq!(expected.link, "log");

    let data = gamma_log_data();
    assert_eq!(data.nrow(), expected.nobs);

    let model = fit_gamma_log(&data, &expected.formula, expected.n_agq);

    assert_eq!(model.nobs(), expected.nobs);
    assert_eq!(model.dof(), expected.dof);
    assert_eq!(model.theta().len(), expected.rust_reference.theta.len());
    assert_eq!(model.fixef().len(), expected.rust_reference.beta.len());

    for (actual, want) in model
        .theta()
        .iter()
        .zip(expected.rust_reference.theta.iter())
    {
        assert_relative_eq!(*actual, *want, epsilon = 1e-12, max_relative = 1e-12);
    }
    for (actual, want) in model
        .fixef()
        .iter()
        .zip(expected.rust_reference.beta.iter())
    {
        assert_relative_eq!(*actual, *want, epsilon = 1e-10, max_relative = 1e-10);
    }

    assert_relative_eq!(
        model.dispersion(false),
        expected.rust_reference.dispersion_sigma,
        epsilon = 1e-12,
        max_relative = 1e-12
    );
    assert_relative_eq!(
        model.dispersion(true),
        expected.rust_reference.dispersion_phi,
        epsilon = 1e-12,
        max_relative = 1e-12
    );
    assert_relative_eq!(
        model.objective(),
        expected.rust_reference.objective,
        epsilon = 1e-10,
        max_relative = 1e-10
    );
    assert_relative_eq!(
        model.loglikelihood(),
        expected.rust_reference.loglik,
        epsilon = 1e-10,
        max_relative = 1e-10
    );
    for (actual, want) in model
        .fitted()
        .iter()
        .take(expected.rust_reference.fitted_mu_head.len())
        .zip(expected.rust_reference.fitted_mu_head.iter())
    {
        assert_relative_eq!(*actual, *want, epsilon = 1e-10, max_relative = 1e-10);
    }

    let julia = expected
        .engines
        .iter()
        .find(|engine| engine.engine == "MixedModels.jl")
        .expect("fixture records MixedModels.jl reference");
    assert_eq!(julia.status, "fit");
    assert_eq!(julia.verdict, "parity_reference");
    for (actual, want) in model.fixef().iter().zip(julia.beta.as_ref().unwrap()) {
        assert_relative_eq!(*actual, *want, epsilon = 2e-5, max_relative = 2e-5);
    }
    for (actual, want) in model.theta().iter().zip(julia.theta.as_ref().unwrap()) {
        assert_relative_eq!(*actual, *want, epsilon = 1e-7);
    }
    assert_relative_eq!(
        model.objective(),
        julia.objective.unwrap(),
        epsilon = 1e-7,
        max_relative = 1e-7
    );

    let lme4 = expected
        .engines
        .iter()
        .find(|engine| engine.engine == "lme4::glmer")
        .expect("fixture records lme4 reference");
    assert_eq!(lme4.status, "fit");
    assert_eq!(lme4.verdict, "documented_divergence");
    assert!(lme4.version.as_deref().unwrap_or("").contains("lme4"));
    assert!(
        lme4.theta.as_ref().unwrap()[0] > 1.0,
        "glmer's Gamma dispersion profiling should remain documented as a non-oracle divergence"
    );
    assert!(lme4.beta.as_ref().unwrap()[0].is_finite());
    assert!(lme4.dispersion.unwrap().is_finite());
    assert!(lme4.loglik.unwrap().is_finite());

    let glmm_tmb = expected
        .engines
        .iter()
        .find(|engine| engine.engine == "glmmTMB")
        .expect("fixture records glmmTMB availability");
    assert_eq!(glmm_tmb.status, "unavailable");
    assert_eq!(glmm_tmb.verdict, "not_run");
    assert!(glmm_tmb.note.contains("not installed"));

    assert!(expected.notes.iter().any(|note| note.contains("glmer")));
}

#[cfg(not(feature = "nlopt"))]
#[test]
fn test_gamma_log_glmm_native_cobyla_preserves_fixture_contract() {
    let expected = fixture();
    let data = gamma_log_data();
    let model = fit_gamma_log(&data, &expected.formula, expected.n_agq);

    assert_eq!(expected.schema_version, "1.0.0");
    assert!(expected.source.contains("MixedModels.jl"));
    assert_eq!(expected.family, "gamma");
    assert_eq!(expected.link, "log");
    assert_eq!(model.nobs(), expected.nobs);
    assert_eq!(model.dof(), expected.dof);
    assert_eq!(model.opt_summary().optimizer, Optimizer::Cobyla);
    assert_eq!(model.opt_summary().backend.label(), "native");
    assert_eq!(model.theta().len(), expected.rust_reference.theta.len());
    assert_eq!(model.fixef().len(), expected.rust_reference.beta.len());
    assert!(model.objective().is_finite());
    assert!(model.loglikelihood().is_finite());
    assert!(model.dispersion(false).is_finite());
    assert!(model.dispersion(true).is_finite());
    for fitted in model
        .fitted()
        .iter()
        .take(expected.rust_reference.fitted_mu_head.len())
    {
        assert!(fitted.is_finite());
        assert!(*fitted > 0.0);
    }
}

#[test]
fn test_gamma_log_fit_is_invariant_to_row_order() {
    let expected = fixture();
    let ordered = fit_gamma_log(&gamma_log_data(), &expected.formula, expected.n_agq);
    let reversed = fit_gamma_log(
        &reversed_gamma_log_data(),
        &expected.formula,
        expected.n_agq,
    );

    assert_relative_eq!(
        ordered.objective(),
        reversed.objective(),
        epsilon = 1e-8,
        max_relative = 1e-8
    );
    assert_relative_eq!(
        ordered.dispersion(true),
        reversed.dispersion(true),
        epsilon = 1e-8,
        max_relative = 1e-8
    );
    for (actual, want) in ordered.fixef().iter().zip(reversed.fixef().iter()) {
        assert_relative_eq!(*actual, *want, epsilon = 1e-8, max_relative = 1e-8);
    }
}

#[test]
fn test_gamma_glmm_parametric_bootstrap_is_seeded_and_positive() {
    let expected = fixture();
    let model = fit_gamma_log(&gamma_log_data(), &expected.formula, expected.n_agq);

    let mut rng_a = StdRng::seed_from_u64(0x4741_4d4d_415f_2026);
    let mut rng_b = StdRng::seed_from_u64(0x4741_4d4d_415f_2026);
    let boot_a = parametricbootstrap_glmm(&mut rng_a, 4, &model).unwrap();
    let boot_b = parametricbootstrap_glmm(&mut rng_b, 4, &model).unwrap();

    assert_eq!(boot_a.fits.len(), 4);
    assert_eq!(boot_b.fits.len(), 4);
    for (idx, (a, b)) in boot_a.fits.iter().zip(boot_b.fits.iter()).enumerate() {
        assert!(
            a.objective.is_finite(),
            "Gamma bootstrap replicate {idx} objective should refit cleanly"
        );
        assert!(
            a.sigma.is_finite() && a.sigma > 0.0,
            "Gamma bootstrap replicate {idx} dispersion should be positive"
        );
        assert_eq!(a.beta.len(), expected.rust_reference.beta.len());
        assert_eq!(a.theta.len(), expected.rust_reference.theta.len());
        for value in a.beta.iter().chain(a.se.iter()) {
            assert!(
                value.is_finite(),
                "Gamma bootstrap replicate {idx} coefficient/SE is not finite"
            );
        }
        for value in &a.theta {
            assert!(
                value.is_finite(),
                "Gamma bootstrap replicate {idx} theta is not finite"
            );
        }

        assert_relative_eq!(a.objective, b.objective, epsilon = 0.0);
        assert_relative_eq!(a.sigma, b.sigma, epsilon = 0.0);
        for (actual, want) in a.beta.iter().zip(b.beta.iter()) {
            assert_relative_eq!(*actual, *want, epsilon = 0.0);
        }
        for (actual, want) in a.theta.iter().zip(b.theta.iter()) {
            assert_relative_eq!(*actual, *want, epsilon = 0.0);
        }
    }
}

#[test]
fn test_lme4_issue_643_gamma_log_fixture_records_engine_divergence() {
    let fixture = lme4_issue_643_fixture();
    assert_eq!(fixture.schema_version, "1.0.0");
    assert!(fixture.source.contains("set.seed(123)"));
    assert_eq!(fixture.issue_url, "https://github.com/lme4/lme4/issues/643");
    assert_eq!(fixture.formula, "y ~ 1 + (1 | grp)");
    assert_eq!(fixture.family, "gamma");
    assert_eq!(fixture.link, "log");
    assert_eq!(fixture.data_recipe.r_seed, 123);
    assert_eq!(fixture.data_recipe.n, 1000);
    assert_eq!(fixture.data_recipe.groups, 20);
    assert_eq!(fixture.data.y.len(), fixture.nobs);
    assert_eq!(fixture.data.group.len(), fixture.nobs);
    assert!(fixture.data.y.iter().all(|y| y.is_finite() && *y > 0.0));
    assert!(fixture
        .data
        .group
        .iter()
        .all(|group| (1..=fixture.data_recipe.groups).contains(group)));

    let observed = summarize(&fixture.data.y);
    assert_relative_eq!(
        observed.mean,
        fixture.observed_summary.mean,
        epsilon = 1e-15,
        max_relative = 1e-15
    );
    assert_relative_eq!(
        observed.max,
        fixture.observed_summary.max,
        epsilon = 1e-15,
        max_relative = 1e-15
    );
    assert_relative_eq!(
        observed.q99,
        fixture.observed_summary.q99,
        epsilon = 1e-14,
        max_relative = 1e-14
    );

    let glmm_tmb = issue_643_engine(&fixture, "glmmTMB");
    let mixedmodels = issue_643_engine(&fixture, "MixedModels.jl");
    let glmer = issue_643_engine(&fixture, "lme4::glmer");

    assert_eq!(glmm_tmb.status, "fit");
    assert_eq!(
        glmm_tmb.verdict,
        "direct_mle_reference_with_local_tmb_warning"
    );
    assert!(glmm_tmb.note.contains("TMB package-version mismatch"));
    assert!(glmm_tmb.version.contains("glmmTMB"));

    assert_eq!(mixedmodels.status, "fit");
    assert_eq!(
        mixedmodels.verdict,
        "supporting_reference_with_dispersion_warning"
    );
    assert!(mixedmodels
        .note
        .contains("dispersion-family results are not reliable"));
    assert_relative_eq!(scalar_or_first(&mixedmodels.theta), 0.0, epsilon = 0.0);

    assert_eq!(glmer.status, "fit");
    assert_eq!(glmer.verdict, "documented_divergence");
    assert!(glmer.note.contains("not as the sole oracle"));
    assert!(glmer.version.contains("lme4"));

    let glmm_tmb_theta = scalar_or_first(&glmm_tmb.theta);
    let glmer_theta = scalar_or_first(&glmer.theta);
    assert!(
        glmm_tmb_theta < 1e-3,
        "glmmTMB should put the random-intercept scale near zero; got {glmm_tmb_theta}"
    );
    assert!(
        glmer_theta / glmm_tmb_theta >= fixture.stress_contract.min_glmer_to_glmmtmb_theta_ratio,
        "lme4#643 sentinel should preserve the Gamma random-effect scale divergence: \
         glmer={glmer_theta}, glmmTMB={glmm_tmb_theta}"
    );
    assert_relative_eq!(
        scalar_or_first(&glmm_tmb.beta),
        scalar_or_first(&mixedmodels.beta),
        epsilon = 1e-6,
        max_relative = 1e-6
    );
}

#[test]
fn test_lme4_issue_643_gamma_log_simulation_and_bootstrap_are_sane() {
    let fixture = lme4_issue_643_fixture();
    let data = lme4_issue_643_data(&fixture);
    let model = fit_gamma_log(&data, &fixture.formula, fixture.n_agq);

    assert_eq!(model.nobs(), fixture.nobs);
    assert_eq!(model.fixef().len(), 1);
    assert_eq!(model.theta().len(), 1);
    assert!(model.objective().is_finite());
    assert!(model.loglikelihood().is_finite());
    assert!(model.dispersion(true).is_finite() && model.dispersion(true) > 0.0);
    assert!(model.theta()[0].is_finite() && model.theta()[0] >= 0.0);

    let glmm_tmb = issue_643_engine(&fixture, "glmmTMB");
    assert_relative_eq!(
        model.fixef()[0],
        scalar_or_first(&glmm_tmb.beta),
        epsilon = 0.25,
        max_relative = 0.25
    );

    let mut rng = StdRng::seed_from_u64(0x6436_414d_4d41_2026);
    for rep in 0..4 {
        let y_sim = model.simulate_response(&mut rng).unwrap();
        assert_eq!(y_sim.len(), fixture.nobs);
        assert!(
            y_sim.iter().all(|y| y.is_finite() && *y > 0.0),
            "lme4#643 Gamma simulation replicate {rep} must stay positive and finite"
        );
        let sim = summarize(&y_sim);
        assert!(
            sim.max <= fixture.stress_contract.max_reasonable_sim_max,
            "lme4#643 Gamma simulation replicate {rep} has an excessive tail max: {}",
            sim.max
        );
        assert!(
            sim.q99 <= fixture.stress_contract.max_reasonable_sim_q99,
            "lme4#643 Gamma simulation replicate {rep} has an excessive q99: {}",
            sim.q99
        );
        assert!(
            sim.mean / fixture.observed_summary.mean
                <= fixture.stress_contract.max_reasonable_bootstrap_mean_ratio,
            "lme4#643 Gamma simulation replicate {rep} shifted away from the observed mean: \
             observed={}, simulated={}",
            fixture.observed_summary.mean,
            sim.mean
        );
    }

    let boot = parametricbootstrap_glmm(&mut rng, 2, &model).unwrap();
    assert_eq!(boot.fits.len(), 2);
    for (idx, fit) in boot.fits.iter().enumerate() {
        assert!(
            fit.objective.is_finite(),
            "lme4#643 Gamma bootstrap replicate {idx} objective should refit cleanly"
        );
        assert!(
            fit.sigma.is_finite() && fit.sigma > 0.0,
            "lme4#643 Gamma bootstrap replicate {idx} dispersion should be positive"
        );
        assert_eq!(fit.beta.len(), 1);
        assert_eq!(fit.theta.len(), 1);
        assert!(fit.beta[0].is_finite());
        assert!(fit.theta[0].is_finite() && fit.theta[0] >= 0.0);
        assert!(
            fit.beta[0].exp() / fixture.observed_summary.mean
                <= fixture.stress_contract.max_reasonable_bootstrap_mean_ratio,
            "lme4#643 Gamma bootstrap replicate {idx} fitted mean shifted too far: beta={}",
            fit.beta[0]
        );
        for value in fit.se.iter() {
            assert!(
                value.is_finite(),
                "lme4#643 Gamma bootstrap replicate {idx} SE should be finite"
            );
        }
    }
}
