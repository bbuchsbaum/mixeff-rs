//! Reproducer for GitHub issue #24, items 2, 4 and 5 (joint-Laplace GLMM).
//!
//! Simulates the issue's synthetic crossed binary design (150 subjects x 80
//! trials, 300 items) and fits
//! `y ~ lag + block + trial + (1 + immediate + postlog || sid) + (1 | item)`
//! by joint Laplace, reporting the stop label, evaluation counts, the
//! certificate's free-gradient norm and status, the objective and wall time.
//! `ns(trial, 3)` is replaced by a centred numeric trial. With `REPRO_HARD=1`
//! the design is closer to the issue's: a cubic trial basis, 0/1
//! `immediate` (lag 0) and `postlog` (second half of the session) slopes, and
//! a near-zero `postlog` slope variance.
//!
//!   cargo run --release --example glmm_issue24_repro -- [mode] [optimizer]
//!
//! `mode`: `default` (default), `maxfeval=N`, `warm` (fit, then refit with
//! the fitted θ as `start_theta`), `verify` (fit, then `verify_convergence`).
//! `optimizer`: `auto` (default) or `trustbq`. `REPRO_SEED` (default 24)
//! changes the simulated data set.

use std::time::Instant;

use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::traits::MixedModelFit;
use mixeff_rs::model::{
    DataFrame, Family, GeneralizedLinearMixedModel, GlmmFitOptions, LinkFunction, OptimizerControl,
};
use mixeff_rs::types::Optimizer;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

fn hard() -> bool {
    std::env::var("REPRO_HARD").is_ok_and(|value| value == "1")
}

fn simulate() -> DataFrame {
    let hard = hard();
    let seed = std::env::var("REPRO_SEED")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(24);
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let (n_subj, per, n_item) = (150usize, 80usize, 300usize);
    let lags = ["0", "1", "2", "4", "8", "16", "24", "32"];
    let subj_re: Vec<[f64; 3]> = (0..n_subj)
        .map(|_| {
            [
                0.8 * normal.sample(&mut rng),
                0.35 * normal.sample(&mut rng),
                if hard { 0.05 } else { 0.25 } * normal.sample(&mut rng),
            ]
        })
        .collect();
    let item_re: Vec<f64> = (0..n_item).map(|_| 0.4 * normal.sample(&mut rng)).collect();
    let n = n_subj * per;
    let (mut trial2, mut trial3) = (Vec::with_capacity(n), Vec::with_capacity(n));
    let (mut y, mut lag, mut block, mut trial, mut immediate, mut postlog, mut sid, mut item) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for (s, re) in subj_re.iter().enumerate() {
        for t in 0..per {
            let it = rng.gen_range(0..n_item);
            let lag_index = rng.gen_range(0..lags.len());
            let blk = rng.gen_range(0..2usize);
            let (imm, post): (f64, f64) = if hard {
                // Indicator covariates tied to the design, as in a lure
                // paradigm: `immediate` is lag 0, `postlog` the second half.
                (
                    if lag_index == 0 { 1.0 } else { 0.0 },
                    if t >= per / 2 { 1.0 } else { 0.0 },
                )
            } else {
                (normal.sample(&mut rng), normal.sample(&mut rng))
            };
            let tr = (t as f64 - 39.5) / 40.0;
            let eta = -0.3 + 0.08 * lag_index as f64 - 0.2 * blk as f64 + 0.3 * tr - 0.15 * tr * tr
                + 0.4 * imm
                + 0.2 * post
                + re[0]
                + re[1] * imm
                + re[2] * post
                + item_re[it];
            let p = 1.0 / (1.0 + (-eta).exp());
            y.push(if rng.gen::<f64>() < p { 1.0 } else { 0.0 });
            lag.push(lags[lag_index].to_string());
            block.push(format!("b{}", blk + 1));
            trial.push(tr);
            trial2.push(tr * tr - 1.0 / 3.0);
            trial3.push(tr * tr * tr);
            immediate.push(imm);
            postlog.push(post);
            sid.push(format!("s{s}"));
            item.push(format!("i{it}"));
        }
    }
    let mut data = DataFrame::new();
    data.add_numeric("y", y).unwrap();
    data.add_categorical("lag", lag).unwrap();
    data.add_categorical("block", block).unwrap();
    data.add_numeric("trial", trial).unwrap();
    data.add_numeric("trial2", trial2).unwrap();
    data.add_numeric("trial3", trial3).unwrap();
    data.add_numeric("immediate", immediate).unwrap();
    data.add_numeric("postlog", postlog).unwrap();
    data.add_categorical("sid", sid).unwrap();
    data.add_categorical("item", item).unwrap();
    data
}

fn new_model(data: &DataFrame) -> GeneralizedLinearMixedModel {
    let formula =
        parse_formula("y ~ lag + block + trial + (1 + immediate + postlog || sid) + (1 | item)")
            .unwrap();
    GeneralizedLinearMixedModel::new(formula, data, Family::Binomial, Some(LinkFunction::Logit))
        .unwrap()
}

fn report(tag: &str, model: &GeneralizedLinearMixedModel, seconds: f64) {
    let optsum = model.lmm().optsum();
    let cert = model.compiler_artifact().optimizer_certificate.as_ref();
    println!("== {tag}");
    println!("  wall:          {seconds:.1}s");
    println!(
        "  stop:          {}  ({} / {} joint evals; initializer {:?} evals, {:?})",
        optsum.return_value,
        optsum.feval,
        optsum.max_feval,
        optsum.initializer_feval,
        optsum.initializer_return_value
    );
    println!("  objective:     {:.8}", model.objective());
    println!("  logLik:        {:.8}", model.loglikelihood());
    println!("  theta:         {:?}", model.theta());
    if let Some(cert) = cert {
        println!("  status:        {:?}", cert.status);
        println!("  free_grad:     {:?} (tol 0.02)", cert.free_gradient_norm);
        if let Some(decrement) = &cert.stationarity_decrement {
            println!(
                "  decrement gap: {:?} ({:?})",
                decrement.eager.objective_gap, decrement.eager.verdict
            );
            let rounded: Vec<String> = decrement
                .parameter_indices
                .iter()
                .zip(&decrement.gradient)
                .map(|(index, value)| format!("{index}:{value:.3}"))
                .collect();
            println!("  gradient:      [{}]", rounded.join(", "));
        }
        for diagnostic in &cert.diagnostics {
            println!("  diag {:?}: {}", diagnostic.code, diagnostic.message);
        }
    }
}

fn options(optimizer: Option<Optimizer>, control: OptimizerControl) -> GlmmFitOptions {
    let control = match optimizer {
        Some(optimizer) => control.with_optimizer(optimizer),
        None => control,
    };
    GlmmFitOptions::joint_laplace().with_optimizer_control(control)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = args.first().cloned().unwrap_or_else(|| "default".into());
    let optimizer = match args.get(1).map(String::as_str) {
        Some("trustbq") => Some(Optimizer::TrustBq),
        _ => None,
    };
    let data = simulate();
    let mut control = OptimizerControl::default();
    if let Some(n) = mode.strip_prefix("maxfeval=") {
        control = control.with_max_feval(n.parse().unwrap());
    }
    let mut model = new_model(&data);
    let start = Instant::now();
    model
        .fit_with_glmm_options(options(optimizer, control))
        .unwrap();
    report(&mode, &model, start.elapsed().as_secs_f64());

    if mode == "warm" {
        let theta = model.theta().to_vec();
        let mut warm = new_model(&data);
        let start = Instant::now();
        warm.fit_with_glmm_options(options(
            optimizer,
            OptimizerControl::default().with_start_theta(theta),
        ))
        .unwrap();
        report(
            "warm start_theta = fitted theta",
            &warm,
            start.elapsed().as_secs_f64(),
        );
        println!(
            "  dObjective vs cold: {:.3e}",
            warm.objective() - model.objective()
        );
    }
    if mode == "verify" {
        let start = Instant::now();
        let verification = model.verify_convergence().unwrap();
        println!(
            "== verify_convergence ({:.1}s): {:?}",
            start.elapsed().as_secs_f64(),
            verification.status
        );
        for run in &verification.runs {
            println!(
                "  {:<22} {:?} obj {:?} dObj {:?} agrees {}",
                run.label, run.return_code, run.objective_value, run.objective_delta, run.agrees
            );
            for diagnostic in &run.diagnostics {
                println!("    {diagnostic}");
            }
        }
    }
}
