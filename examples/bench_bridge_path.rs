//! Host-bridge cost model: compile-then-fit as two calls, with and without a
//! second compile/audit, plus live-model reuse for follow-on inference.
//!
//! Mirrors what the R bridge does per `lmm()`/`glmm()` call:
//!
//! * `current`: call 1 translates the columns into a `DataFrame` and runs
//!   the compiler + design audit (artifact JSON back to the host); call 2
//!   translates the columns again and constructs the model, which compiles
//!   and audits again, fits, and serializes the inspected artifact (forcing
//!   the deferred certificate evidence).
//! * `compiled`: call 1 translates once and builds a `CompiledModelSpec`
//!   (kept alive by the host); call 2 builds the model from it.
//! * `compiled+deferred`: as `compiled`, serializing the artifact without
//!   completing the deferred certificate evidence (§5.4 design option).
//!
//! Then a follow-on — LMM: one Satterthwaite contrast table over all
//! coefficients; GLMM: prediction intervals for 200 rows — cold (translate +
//! construct + refit, as the bridge does today) vs on the live fitted model.
//!
//! Run (release, no NLopt, as the R package builds the engine):
//!     cargo run --release --no-default-features --features unstable-internals \
//!         --example bench_bridge_path [crossed|insteval|verbagg|glmm-crossed|all]

use std::time::{Duration, Instant};

use mixeff_rs::compiler::{
    compile_formula_ir, CompiledModelArtifact, ContrastMatrix, ContrastRhs, FixedEffectHypothesis,
    FixedEffectTestMethod,
};
use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::{
    CompiledModelSpec, DataFrame, Family, FitOptions, GeneralizedLinearMixedModel,
    GeneralizedLinearMixedModelBuilder, GlmmFitOptions, GlmmPredictionScale, LinearMixedModel,
    MixedModelFit, NewReLevels,
};
use mixeff_rs::stats::FitSummaryPayload;
use nalgebra::{DMatrix, DVector};

/// The host's wire form: numeric columns as f64, categorical columns as
/// string values plus their level set (what the R bridge receives).
struct Wire {
    numeric: Vec<(String, Vec<f64>)>,
    categorical: Vec<(String, Vec<String>, Vec<String>)>,
}

impl Wire {
    fn from_frame(df: &DataFrame) -> Self {
        let mut numeric = Vec::new();
        let mut categorical = Vec::new();
        for name in df.column_names() {
            if let Some(values) = df.numeric(name) {
                numeric.push((name.to_string(), values.to_vec()));
            } else if let Some(col) = df.categorical(name) {
                categorical.push((name.to_string(), col.values.clone(), col.levels.clone()));
            }
        }
        Self {
            numeric,
            categorical,
        }
    }

    /// The bridge's `build_dataframe`.
    fn translate(&self) -> DataFrame {
        self.translate_head(usize::MAX)
    }

    fn translate_head(&self, rows: usize) -> DataFrame {
        let mut df = DataFrame::new();
        for (name, values) in &self.numeric {
            let k = rows.min(values.len());
            df.add_numeric(name, values[..k].to_vec()).unwrap();
        }
        for (name, values, levels) in &self.categorical {
            let k = rows.min(values.len());
            df.add_categorical_with_levels(name, values[..k].to_vec(), levels.clone())
                .unwrap();
        }
        df
    }
}

#[derive(Clone, Copy)]
enum Kind {
    Lmm,
    Glmm(Family),
}

#[derive(Default, Clone, Copy)]
struct Phases {
    call1: Duration,
    call2: Duration,
    follow_on: Duration,
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
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

enum Fitted {
    Lmm(Box<LinearMixedModel>),
    Glmm(Box<GeneralizedLinearMixedModel>),
}

fn fit_from(kind: Kind, spec: Option<CompiledModelSpec<'_>>, df: &DataFrame, f: &str) -> Fitted {
    match kind {
        Kind::Lmm => {
            let mut m = match spec {
                Some(spec) => LinearMixedModel::from_compiled(spec, None).unwrap(),
                None => LinearMixedModel::new(parse_formula(f).unwrap(), df, None).unwrap(),
            };
            m.fit_with_options(FitOptions::reml()).unwrap();
            Fitted::Lmm(Box::new(m))
        }
        Kind::Glmm(family) => {
            let builder = match spec {
                Some(spec) => GeneralizedLinearMixedModelBuilder::from_compiled(spec, family),
                None => {
                    GeneralizedLinearMixedModelBuilder::new(parse_formula(f).unwrap(), df, family)
                }
            };
            let m = builder
                .fit_with_glmm_options(GlmmFitOptions::fast_laplace().with_verbose(false))
                .unwrap();
            Fitted::Glmm(Box::new(m))
        }
    }
}

/// What the bridge serializes after a fit.
fn payload(fitted: &Fitted, deferred: bool) -> usize {
    match fitted {
        Fitted::Lmm(m) => {
            let artifact = if deferred {
                m.compiler_artifact_deferred()
            } else {
                m.compiler_artifact()
            };
            let a = serde_json::to_string(artifact).unwrap();
            let s = serde_json::to_string(&FitSummaryPayload::from_linear_model(m)).unwrap();
            let v = (m.coef(), m.stderror(), m.fitted(), m.residuals(), m.ranef());
            a.len() + s.len() + v.2.len()
        }
        Fitted::Glmm(m) => {
            let artifact = if deferred {
                m.compiler_artifact_deferred()
            } else {
                m.compiler_artifact()
            };
            let a = serde_json::to_string(artifact).unwrap();
            let s = serde_json::to_string(&FitSummaryPayload::from_generalized_model(m)).unwrap();
            let v = (m.coef(), m.stderror(), m.fitted(), m.residuals(), m.ranef());
            a.len() + s.len() + v.2.len()
        }
    }
}

fn follow_on(fitted: &Fitted, newdata: &DataFrame) -> usize {
    match fitted {
        Fitted::Lmm(m) => {
            let t = m.fixed_effect_contrast_inference_table(
                hypotheses(m.coef().len()),
                FixedEffectTestMethod::Satterthwaite,
            );
            serde_json::to_string(&t).unwrap().len()
        }
        Fitted::Glmm(m) => {
            let p = m
                .predict_new_variance_with_level(
                    newdata,
                    GlmmPredictionScale::Response,
                    NewReLevels::Population,
                    0.95,
                )
                .unwrap();
            serde_json::to_string(&p).unwrap().len()
        }
    }
}

fn current(wire: &Wire, f: &str, kind: Kind) -> Phases {
    let t = Instant::now();
    let df = wire.translate();
    let parsed = parse_formula(f).unwrap();
    let mut artifact = CompiledModelArtifact::new(parsed.to_string(), compile_formula_ir(&parsed));
    artifact.attach_design_audit(&df);
    std::hint::black_box(serde_json::to_string(&artifact).unwrap());
    drop(df);
    let call1 = t.elapsed();

    let t = Instant::now();
    let df = wire.translate();
    let fitted = fit_from(kind, None, &df, f);
    std::hint::black_box(payload(&fitted, false));
    let call2 = t.elapsed();

    // Follow-on today: translate, construct and refit cold, then answer.
    let t = Instant::now();
    let df = wire.translate();
    let cold = fit_from(kind, None, &df, f);
    std::hint::black_box(follow_on(&cold, &wire.translate_head(200)));
    let follow = t.elapsed();
    Phases {
        call1,
        call2,
        follow_on: follow,
    }
}

fn compiled(wire: &Wire, f: &str, kind: Kind, deferred: bool) -> Phases {
    let t = Instant::now();
    let df = wire.translate();
    let spec = CompiledModelSpec::compile(parse_formula(f).unwrap(), &df).unwrap();
    std::hint::black_box(serde_json::to_string(spec.artifact()).unwrap());
    // The host keeps the spec alive between the two calls.
    let spec = spec.into_owned();
    drop(df);
    let call1 = t.elapsed();

    let t = Instant::now();
    let empty = DataFrame::new();
    let fitted = fit_from(kind, Some(spec), &empty, f);
    std::hint::black_box(payload(&fitted, deferred));
    let call2 = t.elapsed();

    // Follow-on on the live handle.
    let t = Instant::now();
    std::hint::black_box(follow_on(&fitted, &wire.translate_head(200)));
    let follow = t.elapsed();
    Phases {
        call1,
        call2,
        follow_on: follow,
    }
}

fn median(mut xs: Vec<Duration>) -> Duration {
    xs.sort();
    xs[xs.len() / 2]
}

fn run(label: &str, df: &DataFrame, f: &str, kind: Kind, reps: usize) {
    let wire = Wire::from_frame(df);
    println!("\n== {label}: n = {}, formula: {f}", df.nrow());

    // Phase breakdown of one compile+audit and one translation.
    let t = Instant::now();
    let tdf = wire.translate();
    let translate = t.elapsed();
    let t = Instant::now();
    std::hint::black_box(CompiledModelSpec::compile(parse_formula(f).unwrap(), &tdf).unwrap());
    let compile = t.elapsed();
    println!(
        "   translate columns: {:.1} ms   compile+audit: {:.1} ms",
        ms(translate),
        ms(compile)
    );

    let mut rows: Vec<(&str, Vec<Phases>)> = vec![
        ("current", Vec::new()),
        ("compiled", Vec::new()),
        ("compiled+deferred", Vec::new()),
    ];
    for _ in 0..reps {
        rows[0].1.push(current(&wire, f, kind));
        rows[1].1.push(compiled(&wire, f, kind, false));
        rows[2].1.push(compiled(&wire, f, kind, true));
    }
    println!(
        "   {:<18} {:>10} {:>10} {:>11} {:>16}",
        "path", "call1 ms", "call2 ms", "total ms", "follow-on ms"
    );
    for (name, phases) in rows {
        let c1 = median(phases.iter().map(|p| p.call1).collect());
        let c2 = median(phases.iter().map(|p| p.call2).collect());
        let tot = median(phases.iter().map(|p| p.call1 + p.call2).collect());
        let fo = median(phases.iter().map(|p| p.follow_on).collect());
        println!(
            "   {:<18} {:>10.1} {:>10.1} {:>11.1} {:>16.1}",
            name,
            ms(c1),
            ms(c2),
            ms(tot),
            ms(fo)
        );
    }
}

fn crossed_frame(n_subj: usize, n_item: usize, binary: bool) -> DataFrame {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut unif = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let su: Vec<f64> = (0..n_subj).map(|_| unif() - 0.5).collect();
    let ss: Vec<f64> = (0..n_subj).map(|_| 0.3 * (unif() - 0.5)).collect();
    let iu: Vec<f64> = (0..n_item).map(|_| unif() - 0.5).collect();
    let n = n_subj * n_item;
    let (mut y, mut x) = (Vec::with_capacity(n), Vec::with_capacity(n));
    let (mut subj, mut item, mut cond) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for s in 0..n_subj {
        for (i, item_effect) in iu.iter().enumerate() {
            let xv = unif() * 2.0 - 1.0;
            let c = (s + i) % 3;
            let eta = 0.2 + (0.5 + ss[s]) * xv + [0.0, 0.3, -0.2][c] + su[s] + item_effect;
            let yv = if binary {
                let p = 1.0 / (1.0 + (-eta).exp());
                f64::from(unif() < p)
            } else {
                eta + 0.8 * (unif() - 0.5)
            };
            y.push(yv);
            x.push(xv);
            subj.push(format!("s{s}"));
            item.push(format!("i{i}"));
            cond.push(["a", "b", "c"][c].to_string());
        }
    }
    let mut df = DataFrame::new();
    df.add_numeric("y", y).unwrap();
    df.add_numeric("x", x).unwrap();
    df.add_categorical("subj", subj).unwrap();
    df.add_categorical("item", item).unwrap();
    df.add_categorical("cond", cond).unwrap();
    df
}

fn main() {
    let which = std::env::args().nth(1).unwrap_or_else(|| "all".to_string());
    let reps: usize = std::env::var("BENCH_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let all = which == "all";
    if all || which == "crossed" {
        let df = crossed_frame(2000, 100, false);
        run(
            "crossed LMM 200k",
            &df,
            "y ~ 1 + x + cond + (1 + x | subj) + (1 | item)",
            Kind::Lmm,
            reps,
        );
    }
    if all || which == "insteval" {
        let (df, _) = mixeff_rs::datasets::load("insteval").unwrap();
        run(
            "InstEval LMM",
            &df,
            "y ~ 1 + service + dept + (1 | d) + (1 | s)",
            Kind::Lmm,
            reps,
        );
    }
    if all || which == "verbagg" {
        let (raw, _) = mixeff_rs::datasets::load("verbagg").unwrap();
        // The fixture stores the response as an N/Y factor.
        let mut df = DataFrame::new();
        for name in raw.column_names() {
            if let Some(values) = raw.numeric(name) {
                df.add_numeric(name, values.to_vec()).unwrap();
            } else if let Some(col) = raw.categorical(name) {
                if name == "r2" {
                    let y = col.values.iter().map(|v| f64::from(v == "Y")).collect();
                    df.add_numeric(name, y).unwrap();
                } else {
                    df.add_categorical_with_levels(name, col.values.clone(), col.levels.clone())
                        .unwrap();
                }
            }
        }
        run(
            "VerbAgg GLMM (bernoulli, fast)",
            &df,
            "r2 ~ 1 + Anger + Gender + btype + situ + mode + (1 | id) + (1 | item)",
            Kind::Glmm(Family::Bernoulli),
            reps,
        );
    }
    if all || which == "glmm-crossed" {
        let df = crossed_frame(1000, 50, true);
        run(
            "crossed GLMM 50k (bernoulli, fast)",
            &df,
            "y ~ 1 + x + cond + (1 | subj) + (1 | item)",
            Kind::Glmm(Family::Bernoulli),
            reps,
        );
    }
}
