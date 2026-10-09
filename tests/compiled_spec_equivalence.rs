//! Compile-once construction and live-model reuse.
//!
//! 1. A model built from a [`CompiledModelSpec`] (compiled once, inspected,
//!    detached with `into_owned`, then built) is bit-identical to the
//!    one-step constructors: same fitted parameters, same artifact JSON, same
//!    fitted-state snapshot.
//! 2. Follow-on computations on one fitted model kept alive (a host handle
//!    cache) are bit-identical to running each on a freshly rebuilt and
//!    refitted model, in any order.
//! 3. Deferred certificate evidence: the fit-time accessors a host reads do
//!    not force it, and completing it in place reproduces the lazily
//!    inspected artifact exactly.

use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::{
    CompiledModelSpec, DataFrame, Family, FitOptions, GeneralizedLinearMixedModel,
    GeneralizedLinearMixedModelBuilder, GlmmFitOptions, LinearMixedModel, LinkFunction,
    MixedModelFit,
};

/// Deterministic crossed design: 16 subjects x 10 items, two replicates.
fn crossed_frame() -> DataFrame {
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut unif = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let subj_eff: Vec<f64> = (0..24)
        .map(|i| ((i * 37 % 11) as f64 - 5.0) * 0.15)
        .collect();
    let subj_slope: Vec<f64> = (0..24)
        .map(|i| ((i * 13 % 7) as f64 - 3.0) * 0.05)
        .collect();
    let item_eff: Vec<f64> = (0..10).map(|i| ((i * 7 % 9) as f64 - 4.0) * 0.2).collect();
    let (mut y, mut yp, mut x, mut b, mut cnt, mut prop, mut trials, mut off, mut w) = (
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    let (mut subj, mut item, mut cond) = (Vec::new(), Vec::new(), Vec::new());
    for s in 0..16 {
        for (i, item_effect) in item_eff.iter().enumerate() {
            for r in 0..2 {
                let xv = ((s + 2 * i + r) % 7) as f64 / 3.0 - 1.0;
                let c = (s + i + r) % 3;
                let eta = 0.3
                    + (0.5 + subj_slope[s]) * xv
                    + [0.0, 0.4, -0.3][c]
                    + subj_eff[s]
                    + item_effect;
                let e = unif() - 0.5;
                y.push(eta + 0.6 * e);
                yp.push((eta + 0.6 * e).exp());
                x.push(xv);
                let p = 1.0 / (1.0 + (-(eta - 0.3)).exp());
                b.push(if unif() < p { 1.0 } else { 0.0 });
                let lam = (0.4 * eta + 0.5).exp();
                cnt.push((lam + 1.5 * (unif() - 0.5)).max(0.0).round());
                let n = 2.0 + ((s + i) % 4) as f64;
                let k = (n * p + (unif() - 0.5)).round().clamp(0.0, n);
                prop.push(k / n);
                trials.push(n);
                off.push(((s % 3) as f64 + 1.0).ln());
                w.push(0.5 + ((s * 3 + i) % 5) as f64 * 0.25);
                subj.push(format!("s{s:02}"));
                item.push(format!("i{i:02}"));
                cond.push(["a", "b", "c"][c].to_string());
            }
        }
    }
    let mut df = DataFrame::new();
    df.add_numeric("y", y).unwrap();
    df.add_numeric("yp", yp).unwrap();
    df.add_numeric("x", x).unwrap();
    df.add_numeric("b", b).unwrap();
    df.add_numeric("cnt", cnt).unwrap();
    df.add_numeric("prop", prop).unwrap();
    df.add_numeric("trials", trials).unwrap();
    df.add_numeric("off", off).unwrap();
    df.add_numeric("w", w).unwrap();
    df.add_categorical("subj", subj).unwrap();
    df.add_categorical("item", item).unwrap();
    df.add_categorical("cond", cond).unwrap();
    df
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn json<T: serde::Serialize>(value: &T) -> String {
    serde_json::to_string(value).expect("serializable")
}

fn assert_same_lmm(direct: &LinearMixedModel, compiled: &LinearMixedModel, label: &str) {
    assert_eq!(
        bits(&direct.theta()),
        bits(&compiled.theta()),
        "{label}: theta"
    );
    assert_eq!(
        bits(direct.coef().as_slice()),
        bits(compiled.coef().as_slice()),
        "{label}: beta"
    );
    assert_eq!(
        direct.objective().to_bits(),
        compiled.objective().to_bits(),
        "{label}: objective"
    );
    assert_eq!(
        bits(direct.stderror().as_slice()),
        bits(compiled.stderror().as_slice()),
        "{label}: stderror"
    );
    assert_eq!(
        bits(direct.fitted().as_slice()),
        bits(compiled.fitted().as_slice()),
        "{label}: fitted"
    );
    assert_eq!(
        json(direct.compiler_artifact()),
        json(compiled.compiler_artifact()),
        "{label}: post-fit artifact"
    );
    // Snapshot refusals (the engine declines some fitted states) must
    // agree too.
    assert_eq!(
        format!("{:?}", direct.snapshot_json()),
        format!("{:?}", compiled.snapshot_json()),
        "{label}: fitted-state snapshot (recipe, basis, state)"
    );
}

const LMM_CASES: &[(&str, bool, bool)] = &[
    // (formula, reml, weighted)
    (
        "y ~ 1 + x + cond + (1 + x | subj) + (1 | item)",
        true,
        false,
    ),
    (
        "y ~ 1 + x + cond + (1 + x | subj) + (1 | item)",
        false,
        true,
    ),
    // in-formula transform: the spec carries the lowered frame
    ("log(yp) ~ 1 + x + (1 | subj) + (1 | item)", true, false),
    // `||` over a factor: expanded at compile time
    ("y ~ 1 + x + (1 + cond || subj) + (1 | item)", true, false),
];

#[test]
fn lmm_from_compiled_spec_is_bit_identical_to_one_step_construction() {
    let df = crossed_frame();
    let weights = df.numeric("w").unwrap().to_vec();
    for &(formula, reml, weighted) in LMM_CASES {
        let w = weighted.then_some(weights.as_slice());
        let options = || {
            if reml {
                FitOptions::reml()
            } else {
                FitOptions::ml()
            }
        };

        let mut direct = LinearMixedModel::new(parse_formula(formula).unwrap(), &df, w).unwrap();
        direct.fit_with_options(options()).unwrap();

        // Host route: compile once, inspect, detach from the data, then
        // build and fit without a second compile/audit.
        let spec = CompiledModelSpec::compile(parse_formula(formula).unwrap(), &df).unwrap();
        assert!(spec.artifact().design_audit.is_some());
        assert_eq!(spec.nrow(), df.nrow());
        let spec = spec.into_owned();
        let mut compiled = LinearMixedModel::from_compiled(spec, w).unwrap();
        compiled.fit_with_options(options()).unwrap();

        assert_same_lmm(&direct, &compiled, formula);
    }
}

#[test]
fn unfitted_models_from_both_routes_carry_the_same_artifact() {
    let df = crossed_frame();
    for &(formula, _, _) in LMM_CASES {
        let direct = LinearMixedModel::new(parse_formula(formula).unwrap(), &df, None).unwrap();
        let spec = CompiledModelSpec::compile(parse_formula(formula).unwrap(), &df).unwrap();
        let compiled = LinearMixedModel::from_compiled(spec, None).unwrap();
        assert_eq!(
            json(direct.compiler_artifact()),
            json(compiled.compiler_artifact()),
            "{formula}"
        );
    }
}

#[test]
fn compile_refuses_what_construction_refuses() {
    let df = crossed_frame();
    assert!(CompiledModelSpec::compile(parse_formula("y ~ 1 + x").unwrap(), &df).is_err());
}

struct GlmmCase {
    label: &'static str,
    formula: &'static str,
    family: Family,
    link: Option<LinkFunction>,
    weights: Option<&'static str>,
    offset: Option<&'static str>,
    nb_estimate: bool,
    joint: bool,
}

const GLMM_CASES: &[GlmmCase] = &[
    GlmmCase {
        label: "bernoulli fast",
        formula: "b ~ 1 + x + (1 | subj) + (1 | item)",
        family: Family::Bernoulli,
        link: None,
        weights: None,
        offset: None,
        nb_estimate: false,
        joint: false,
    },
    GlmmCase {
        label: "bernoulli joint",
        formula: "b ~ 1 + x + (1 | subj)",
        family: Family::Bernoulli,
        link: None,
        weights: None,
        offset: None,
        nb_estimate: false,
        joint: true,
    },
    GlmmCase {
        label: "binomial trials",
        formula: "prop ~ 1 + x + (1 | subj)",
        family: Family::Binomial,
        link: Some(LinkFunction::Logit),
        weights: Some("trials"),
        offset: None,
        nb_estimate: false,
        joint: false,
    },
    GlmmCase {
        label: "poisson offset",
        formula: "cnt ~ 1 + x + (1 | subj) + (1 | item)",
        family: Family::Poisson,
        link: None,
        weights: None,
        offset: Some("off"),
        nb_estimate: false,
        joint: false,
    },
    GlmmCase {
        label: "negative binomial estimated theta",
        formula: "cnt ~ 1 + x + (1 | subj)",
        family: Family::NegativeBinomial,
        link: None,
        weights: None,
        offset: None,
        nb_estimate: true,
        joint: false,
    },
];

fn configure<'a>(
    mut builder: GeneralizedLinearMixedModelBuilder<'a>,
    case: &GlmmCase,
    df: &DataFrame,
) -> GeneralizedLinearMixedModelBuilder<'a> {
    if let Some(link) = case.link {
        builder = builder.link(link);
    }
    if let Some(column) = case.weights {
        builder = builder.weights(df.numeric(column).unwrap().to_vec());
    }
    if let Some(column) = case.offset {
        builder = builder.offset(df.numeric(column).unwrap().to_vec());
    }
    if case.nb_estimate {
        builder = builder.estimate_negative_binomial_theta(None);
    }
    builder
}

fn glmm_options(case: &GlmmCase) -> GlmmFitOptions {
    if case.joint {
        GlmmFitOptions::joint_laplace()
    } else {
        GlmmFitOptions::fast_laplace()
    }
    .with_verbose(false)
}

fn fit_glmm_direct(case: &GlmmCase, df: &DataFrame) -> GeneralizedLinearMixedModel {
    configure(
        GeneralizedLinearMixedModelBuilder::new(
            parse_formula(case.formula).unwrap(),
            df,
            case.family,
        ),
        case,
        df,
    )
    .fit_with_glmm_options(glmm_options(case))
    .unwrap()
}

fn fit_glmm_compiled(case: &GlmmCase, df: &DataFrame) -> GeneralizedLinearMixedModel {
    let spec = CompiledModelSpec::compile(parse_formula(case.formula).unwrap(), df)
        .unwrap()
        .into_owned();
    configure(
        GeneralizedLinearMixedModelBuilder::from_compiled(spec, case.family),
        case,
        df,
    )
    .fit_with_glmm_options(glmm_options(case))
    .unwrap()
}

#[test]
fn glmm_from_compiled_spec_is_bit_identical_to_builder_construction() {
    let df = crossed_frame();
    for case in GLMM_CASES {
        let direct = fit_glmm_direct(case, &df);
        let compiled = fit_glmm_compiled(case, &df);
        let label = case.label;
        assert_eq!(
            bits(&direct.theta()),
            bits(&compiled.theta()),
            "{label}: theta"
        );
        assert_eq!(
            bits(direct.coef().as_slice()),
            bits(compiled.coef().as_slice()),
            "{label}: beta"
        );
        assert_eq!(
            direct.loglikelihood().to_bits(),
            compiled.loglikelihood().to_bits(),
            "{label}: loglik"
        );
        assert_eq!(
            direct.negative_binomial_theta().map(f64::to_bits),
            compiled.negative_binomial_theta().map(f64::to_bits),
            "{label}: nb theta"
        );
        assert_eq!(
            json(direct.compiler_artifact()),
            json(compiled.compiler_artifact()),
            "{label}: post-fit artifact"
        );
        assert_eq!(
            format!("{:?}", direct.snapshot_json()),
            format!("{:?}", compiled.snapshot_json()),
            "{label}: fitted-state snapshot"
        );
    }
}

#[test]
fn compiled_glmm_builder_refuses_a_second_policy() {
    let df = crossed_frame();
    let spec =
        CompiledModelSpec::compile(parse_formula("b ~ 1 + x + (1 | subj)").unwrap(), &df).unwrap();
    let err = GeneralizedLinearMixedModelBuilder::from_compiled(spec, Family::Bernoulli)
        .compiler_policy(Default::default())
        .build();
    assert!(err.is_err());
}

#[cfg(feature = "unstable-internals")]
mod unstable {
    use super::*;
    use mixeff_rs::compiler::{
        compile_formula_ir, CompiledModelArtifact, ContrastMatrix, ContrastRhs,
        FixedEffectHypothesis, FixedEffectTermTestType, FixedEffectTestMethod,
    };
    use mixeff_rs::model::linear::{parametricbootstrap_with_options, BootstrapExecutionOptions};
    use mixeff_rs::model::{
        BootstrapFailedRefitPolicy, FixedEffectBootstrapOptions, GlmmPredictionScale, NewReLevels,
    };
    use mixeff_rs::stats::profile::{profile_confint_payload_with_options, ProfileOptions};
    use mixeff_rs::stats::FitSummaryPayload;
    use nalgebra::{DMatrix, DVector};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// The pre-fit artifact the R bridge's separate compile entry point
    /// builds today. For formulas without transforms or `||` factor terms
    /// the spec's artifact is the same document.
    #[test]
    fn spec_artifact_matches_the_standalone_compile_and_audit() {
        let df = crossed_frame();
        let formula = "y ~ 1 + x + cond + (1 + x | subj) + (1 | item)";
        let parsed = parse_formula(formula).unwrap();
        let mut standalone =
            CompiledModelArtifact::new(parsed.to_string(), compile_formula_ir(&parsed));
        standalone.attach_design_audit(&df);
        let spec = CompiledModelSpec::compile(parsed, &df).unwrap();
        assert_eq!(json(&standalone), json(spec.artifact()));
    }

    fn hypotheses(p: usize) -> Vec<FixedEffectHypothesis> {
        (0..p)
            .map(|j| {
                let mut row = vec![0.0; p];
                row[j] = 1.0;
                FixedEffectHypothesis::new(
                    format!("b{j}"),
                    ContrastMatrix::new(DMatrix::from_row_slice(1, p, &row)).unwrap(),
                    ContrastRhs::new(DVector::from_row_slice(&[0.0])).unwrap(),
                )
                .unwrap()
            })
            .collect()
    }

    fn joint_hypothesis(p: usize) -> FixedEffectHypothesis {
        let mut l = DMatrix::zeros(2, p);
        l[(0, 1)] = 1.0;
        l[(1, 2)] = 1.0;
        l[(1, 3)] = -1.0;
        FixedEffectHypothesis::new(
            "joint",
            ContrastMatrix::new(l).unwrap(),
            ContrastRhs::new(DVector::zeros(2)).unwrap(),
        )
        .unwrap()
    }

    fn newdata() -> DataFrame {
        let mut nd = DataFrame::new();
        nd.add_numeric("x", vec![-1.0, 0.0, 0.5, 1.0]).unwrap();
        nd.add_categorical(
            "cond",
            ["a", "b", "c", "a"].iter().map(|s| s.to_string()).collect(),
        )
        .unwrap();
        nd.add_categorical(
            "subj",
            ["s00", "s05", "new", "s15"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        )
        .unwrap();
        nd.add_categorical(
            "item",
            ["i00", "i09", "i03", "zz"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        )
        .unwrap();
        nd
    }

    /// Every follow-on computation the R bridge runs after a fit, as one
    /// string per computation. `&mut` computations (profile) run on a
    /// clone, as a handle cache must.
    fn follow_ons(model: &LinearMixedModel) -> Vec<(String, String)> {
        let p = model.coef().len();
        let mut out = Vec::new();
        for (name, method) in [
            ("satterthwaite", FixedEffectTestMethod::Satterthwaite),
            ("kenward_roger", FixedEffectTestMethod::KenwardRoger),
            ("asymptotic", FixedEffectTestMethod::AsymptoticWaldZ),
        ] {
            out.push((
                format!("contrast/{name}"),
                json(&model.fixed_effect_contrast_inference_table(hypotheses(p), method)),
            ));
            out.push((
                format!("joint/{name}"),
                json(&model.fixed_effect_contrast_inference_row(
                    mixeff_rs::compiler::FixedEffectInferenceRowKind::Contrast,
                    joint_hypothesis(p),
                    method,
                )),
            ));
            out.push((
                format!("terms/{name}"),
                json(&model.fixed_effect_term_inference_table_for_type(
                    method,
                    FixedEffectTermTestType::TypeIII,
                )),
            ));
        }
        out.push((
            "bootstrap_null".into(),
            json(&model.fixed_effect_null_bootstrap_inference_table(
                hypotheses(p)[1..2].to_vec(),
                FixedEffectBootstrapOptions {
                    requested_replicates: 12,
                    failed_refit_policy: BootstrapFailedRefitPolicy::Exclude,
                    seed: Some(7),
                    threads: 1,
                },
            )),
        ));
        let mut rng = StdRng::seed_from_u64(11);
        let boot = parametricbootstrap_with_options(
            &mut rng,
            10,
            model,
            &BootstrapExecutionOptions { threads: 2 },
        )
        .unwrap();
        out.push(("parametric_bootstrap".into(), format!("{:?}", boot.fits)));
        let mut work = model.clone();
        out.push((
            "profile".into(),
            json(
                &profile_confint_payload_with_options(
                    &mut work,
                    0.95,
                    &ProfileOptions { threads: 1 },
                )
                .unwrap(),
            ),
        ));
        let nd = newdata();
        out.push((
            "predict_new".into(),
            format!(
                "{:?}",
                model.predict_new(&nd, NewReLevels::Population).unwrap()
            ),
        ));
        out.push((
            "predict_new_variance".into(),
            json(
                &model
                    .predict_new_variance_with_level(&nd, NewReLevels::Population, 0.9)
                    .unwrap(),
            ),
        ));
        out.push(("cond_var".into(), format!("{:?}", model.cond_var())));
        let mut sim_rng = StdRng::seed_from_u64(3);
        out.push((
            "simulate".into(),
            format!("{:?}", model.simulate(&mut sim_rng).as_slice()),
        ));
        out.push((
            "fit_summary".into(),
            json(&FitSummaryPayload::from_linear_model(model)),
        ));
        out
    }

    fn fit_lmm(df: &DataFrame) -> LinearMixedModel {
        let spec = CompiledModelSpec::compile(
            parse_formula("y ~ 1 + x + cond + (1 + x | subj) + (1 | item)").unwrap(),
            df,
        )
        .unwrap();
        let mut model = LinearMixedModel::from_compiled(spec, None).unwrap();
        model.fit_with_options(FitOptions::reml()).unwrap();
        model
    }

    #[test]
    fn a_live_fitted_lmm_answers_follow_ons_exactly_like_cold_refits() {
        let df = crossed_frame();
        let live = fit_lmm(&df);
        // Inspect the certificate first, as the bridge's fit entry point does.
        let _ = live.compiler_artifact();
        let first = follow_ons(&live);
        // Reuse the same live model again (caches warm, after a profile ran
        // on a clone and a bootstrap ran from it) ...
        let second = follow_ons(&live);
        // ... and compare both to a cold rebuild + refit per computation.
        let cold = follow_ons(&fit_lmm(&df));
        for ((name, a), ((_, b), (_, c))) in first.iter().zip(second.iter().zip(cold.iter())) {
            assert_eq!(a, b, "{name}: live reuse drifted");
            assert_eq!(a, c, "{name}: live model differs from a cold refit");
        }
        // The live model's fitted state is untouched by the follow-ons.
        assert_eq!(
            live.snapshot_json().unwrap(),
            fit_lmm(&df).snapshot_json().unwrap()
        );
    }

    #[test]
    fn a_live_fitted_glmm_answers_follow_ons_exactly_like_cold_refits() {
        let df = crossed_frame();
        let nd = newdata();
        for case in GLMM_CASES
            .iter()
            .filter(|c| c.offset.is_none() && c.weights.is_none())
        {
            let run = |model: &GeneralizedLinearMixedModel| {
                let mut rng = StdRng::seed_from_u64(5);
                let boot = mixeff_rs::stats::bootstrap::parametricbootstrap_glmm_with_options(
                    &mut rng,
                    4,
                    model,
                    &BootstrapExecutionOptions { threads: 1 },
                )
                .unwrap();
                let pv = model
                    .predict_new_variance_with_level(
                        &nd,
                        GlmmPredictionScale::Response,
                        NewReLevels::Population,
                        0.95,
                    )
                    .map(|p| json(&p))
                    .unwrap_or_else(|e| e.to_string());
                let mut verify = model.clone();
                let verification = verify.verify_convergence().map(|v| json(&v));
                (
                    format!("{:?}", boot.fits),
                    pv,
                    format!("{verification:?}"),
                    json(&FitSummaryPayload::from_generalized_model(model)),
                )
            };
            let live = fit_glmm_compiled(case, &df);
            let first = run(&live);
            let second = run(&live);
            let cold = run(&fit_glmm_compiled(case, &df));
            assert_eq!(first, second, "{}: live reuse drifted", case.label);
            assert_eq!(first, cold, "{}: live differs from cold refit", case.label);
        }
    }

    /// The artifact minus what deferred evidence may still change: the
    /// optimizer certificate, and for GLMMs the profiled-optimum
    /// certificate's provenance diagnostic.
    fn without_certificate(artifact: &CompiledModelArtifact, glmm: bool) -> serde_json::Value {
        let mut value = serde_json::to_value(artifact).unwrap();
        let object = value.as_object_mut().unwrap();
        object.remove("optimizer_certificate");
        if glmm {
            object.remove("diagnostics");
        }
        value
    }

    #[test]
    fn lmm_deferred_certificate_survives_fit_time_reads_and_completes_exactly() {
        let df = crossed_frame();
        let mut model = fit_lmm(&df);
        assert!(model.certificate_evidence_pending());
        // Everything the bridge reads at fit time, except the inspected
        // artifact itself.
        let _ = (
            model.coef(),
            model.coef_names(),
            model.stderror(),
            model.fixed_effect_fitted(),
            model.fitted(),
            model.residuals(),
            model.loglikelihood(),
            model.aic(),
            model.bic(),
            model.varcorr(),
            model.ranef(),
            model.opt_summary(),
            FitSummaryPayload::from_linear_model(&model),
        );
        let deferred = json(model.compiler_artifact_deferred());
        assert!(model.certificate_evidence_pending());

        let untouched = model.clone();
        let inspected = json(untouched.compiler_artifact());
        assert_ne!(deferred, inspected, "evidence was actually deferred");
        assert_eq!(
            without_certificate(model.compiler_artifact_deferred(), false),
            without_certificate(untouched.compiler_artifact(), false),
            "only the certificate differs while deferred"
        );
        model.complete_certificate_evidence();
        assert!(!model.certificate_evidence_pending());
        assert_eq!(json(model.compiler_artifact_deferred()), inspected);
        assert_eq!(json(model.compiler_artifact()), inspected);
    }

    #[test]
    fn glmm_deferred_certificate_survives_fit_time_reads_and_completes_exactly() {
        let df = crossed_frame();
        for case in GLMM_CASES {
            let mut model = fit_glmm_compiled(case, &df);
            let pending = model.certificate_evidence_pending();
            let _ = (
                model.coef(),
                model.stderror(),
                model.fitted(),
                model.residuals(),
                model.loglikelihood(),
                model.varcorr(),
                model.ranef(),
                FitSummaryPayload::from_generalized_model(&model),
            );
            let deferred = model.compiler_artifact_deferred().clone();
            assert_eq!(
                model.certificate_evidence_pending(),
                pending,
                "{}: fit-time reads forced the certificate",
                case.label
            );
            let untouched = model.clone();
            let inspected = json(untouched.compiler_artifact());
            assert_eq!(
                without_certificate(&deferred, true),
                without_certificate(untouched.compiler_artifact(), true),
                "{}: only the certificate differs while deferred",
                case.label
            );
            if !pending {
                assert_eq!(json(&deferred), inspected, "{}", case.label);
            }
            model.complete_certificate_evidence();
            assert!(!model.certificate_evidence_pending(), "{}", case.label);
            assert_eq!(
                json(model.compiler_artifact_deferred()),
                inspected,
                "{}",
                case.label
            );
            assert_eq!(json(model.compiler_artifact()), inspected, "{}", case.label);
        }
    }
}
