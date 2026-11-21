//! GH #24 item 1 reproducer: a crossed LMM with 10 independent variance
//! terms, `(1 + x1 + … + x5 || sid) + (1 + x1 + x2 + x3 || item)`, sized
//! either just under or above the blocked-gradient selected-inverse guard.
//!
//! ```text
//! cargo run --release --example probe_issue24_lmm10 -- 272 500 160   # over the old guard
//! cargo run --release --example probe_issue24_lmm10 -- 200 400 150   # under it
//! ```
//!
//! Fully synthetic (mirrors `issue_reprex_lmm10_large.R` from the issue,
//! with `ns(trial, 3)` replaced by a linear trial term).

use std::error::Error;
use std::time::Instant;

use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::data::DataFrame;
use mixeff_rs::model::linear::LinearMixedModel;
use mixeff_rs::model::traits::MixedModelFit;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

fn simulate(n_sid: usize, n_item: usize, per: usize, seed: u64) -> DataFrame {
    let mut rng = StdRng::seed_from_u64(seed);
    let z = Normal::new(0.0, 1.0).unwrap();
    let n = n_sid * per;
    let sd_s = [0.3, 0.1, 0.08, 0.05, 0.04, 0.03];
    let sd_i = [0.15, 0.06, 0.04, 0.02];
    let us: Vec<[f64; 6]> = (0..n_sid)
        .map(|_| std::array::from_fn(|k| sd_s[k] * z.sample(&mut rng)))
        .collect();
    let ui: Vec<[f64; 4]> = (0..n_item)
        .map(|_| std::array::from_fn(|k| sd_i[k] * z.sample(&mut rng)))
        .collect();
    let lags = ["0", "1", "2", "4", "8", "16", "24", "32"];
    let (mut y, mut sid, mut item, mut lag, mut block, mut trial) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    let mut xs: Vec<Vec<f64>> = (0..5).map(|_| Vec::with_capacity(n)).collect();
    for (s, u_s) in us.iter().enumerate() {
        for _ in 0..per {
            let it = rng.gen_range(0..n_item);
            let lg = rng.gen_range(0..lags.len());
            let x: [f64; 5] = std::array::from_fn(|_| z.sample(&mut rng));
            let mut mu = -0.5 + if lg == 0 { 0.1 } else { 0.0 } + u_s[0] + ui[it][0];
            for k in 0..5 {
                mu += u_s[k + 1] * x[k];
            }
            for k in 0..3 {
                mu += ui[it][k + 1] * x[k];
            }
            y.push(mu + 0.45 * z.sample(&mut rng));
            sid.push(format!("s{s}"));
            item.push(format!("i{it}"));
            lag.push(lags[lg].to_string());
            block.push(format!("b{}", rng.gen_range(1..=2)));
            trial.push(rng.gen_range(1.0..190.0) / 100.0);
            for k in 0..5 {
                xs[k].push(x[k]);
            }
        }
    }
    let mut df = DataFrame::new();
    df.add_numeric("y", y).unwrap();
    for (k, col) in xs.into_iter().enumerate() {
        df.add_numeric(&format!("x{}", k + 1), col).unwrap();
    }
    df.add_numeric("trial", trial).unwrap();
    df.add_categorical("sid", sid).unwrap();
    df.add_categorical("item", item).unwrap();
    df.add_categorical("lag", lag).unwrap();
    df.add_categorical("block", block).unwrap();
    df
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().expect("integer argument"))
        .collect();
    let (n_sid, n_item, per) = match args.as_slice() {
        [a, b, c] => (*a, *b, *c),
        _ => (272, 500, 160),
    };
    let data = simulate(n_sid, n_item, per, 11);
    let formula = parse_formula(
        "y ~ lag * block + trial + (1 + x1 + x2 + x3 + x4 + x5 || sid) + (1 + x1 + x2 + x3 || item)",
    )?;
    let mut model = LinearMixedModel::new(formula, &data, None)?;
    let start = Instant::now();
    model.fit(true)?;
    let elapsed = start.elapsed().as_secs_f64();
    let certificate = model
        .optimizer_certificate()
        .expect("fitted model should have an optimizer certificate");
    println!(
        "n={} sid={n_sid} item={n_item} theta={} elapsed={elapsed:.1}s",
        n_sid * per,
        model.theta().len()
    );
    println!(
        "objective={:.6} fevals={} stop={:?}",
        model.opt_summary().fmin,
        model.opt_summary().feval,
        model.opt_summary().return_value
    );
    println!(
        "status={:?} free_gradient_norm={:?} gradient_method={:?}",
        certificate.status, certificate.free_gradient_norm, certificate.evidence.gradient.method
    );
    for check in &certificate.checks {
        let text = format!("{check:?}");
        if text.starts_with("DerivativeMismatch") || text.starts_with("FreeGradient") {
            println!("  check: {text}");
        }
    }
    println!("diagnostics: {}", certificate.diagnostics.len());
    for d in &certificate.diagnostics {
        println!("  [{:?}/{:?}] {}", d.severity, d.code, d.message);
    }
    Ok(())
}
