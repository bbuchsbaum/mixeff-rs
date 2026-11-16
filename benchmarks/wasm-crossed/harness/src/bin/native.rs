//! Native driver: `cargo run --release --bin native -- <reps> [design...]`.
//! Emits one JSON object per timed call on stdout (same schema as the Node
//! driver), so `../summarize.py` can treat both identically.

use std::time::Instant;

use mixeff_wasm_crossed_bench as bench;

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn main() {
    let mut args = std::env::args().skip(1);
    let reps: u64 = args.next().map_or(30, |a| a.parse().unwrap());
    let designs: Vec<(usize, usize)> = {
        let picked: Vec<usize> = args.map(|a| a.parse().unwrap()).collect();
        if picked.is_empty() {
            bench::DESIGNS.to_vec()
        } else {
            picked.into_iter().map(|n| (n, n)).collect()
        }
    };
    for (n_p, n_i) in designs {
        // Untimed warm-up (page faults, allocator growth), matching the Node
        // driver where it also lets the JIT tier up.
        bench::simulate(n_p, n_i, 999_999);
        bench::fit();
        bench::satterthwaite();
        // Fresh fits: construction + REML fit, then the Satterthwaite table.
        for seed in 1..=reps {
            bench::simulate(n_p, n_i, seed);
            let t = Instant::now();
            let s = bench::fit();
            let fit_ms = ms(t);
            let t = Instant::now();
            let sat = bench::satterthwaite();
            let satt_ms = ms(t);
            let reason = bench::satterthwaite_reason();
            emit(
                "native",
                "fit",
                n_p,
                n_i,
                seed,
                fit_ms,
                Some(satt_ms),
                &s,
                Some((&sat, reason)),
            );
        }
        // Warm refits of one template (compile once, refit many).
        bench::prepare_template(n_p, n_i, 0);
        for seed in 1..=reps {
            bench::simulate(n_p, n_i, 1_000 + seed);
            let t = Instant::now();
            let s = bench::refit();
            let fit_ms = ms(t);
            emit("native", "refit", n_p, n_i, seed, fit_ms, None, &s, None);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn emit(
    platform: &str,
    mode: &str,
    n_p: usize,
    n_i: usize,
    seed: u64,
    fit_ms: f64,
    satt_ms: Option<f64>,
    s: &[f64],
    sat: Option<(&[f64], String)>,
) {
    let v = serde_json::json!({
        "platform": platform, "mode": mode, "n_participants": n_p, "n_items": n_i,
        "seed": seed, "fit_ms": fit_ms, "satterthwaite_ms": satt_ms,
        "beta_condition": s[0], "se_condition": s[1], "singular": s[2] != 0.0,
        "converged": s[3] != 0.0, "feval": s[4], "objective": s[5],
        "satterthwaite_df": sat.as_ref().map(|x| x.0[0]),
        "satterthwaite_p": sat.as_ref().map(|x| x.0[1]),
        "satterthwaite_method": sat.as_ref().map(|x| x.1.clone()),
    });
    println!("{v}");
}
