//! Merge `comparison/asymptotic/{rust,lme4}_results.json` into a single
//! markdown table at `comparison/asymptotic/REPORT.md`.

use std::fs;
use std::path::PathBuf;

use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct RustResult {
    label: String,
    n_subjects: usize,
    n_obs: usize,
    fit_time_ms_min: f64,
    fit_time_ms_median: f64,
    parse_build_ms_median: f64,
    fit_only_ms_median: f64,
    fevals: i64,
    optimizer: String,
    objective: f64,
    sigma: f64,
}

#[derive(Debug, Deserialize)]
struct RResult {
    label: String,
    n_obs: usize,
    fit_time_ms_min: f64,
    fit_time_ms_median: f64,
    fevals: Option<i64>,
    objective: f64,
    sigma: f64,
}

#[derive(Debug, Deserialize)]
struct RustFile {
    tool: String,
    results: Vec<RustResult>,
}

#[derive(Debug, Deserialize)]
struct RFile {
    tool: String,
    results: Vec<RResult>,
}

fn comparison_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("comparison")
        .join("asymptotic")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cmp = comparison_root();
    let rust: RustFile = serde_json::from_str(&fs::read_to_string(cmp.join("rust_results.json"))?)?;
    let r: RFile = serde_json::from_str(&fs::read_to_string(cmp.join("lme4_results.json"))?)?;

    let mut report = String::new();
    report.push_str("# Asymptotic Benchmark — Rust vs lme4\n\n");
    report.push_str(&format!(
        "Synthetic sleepstudy-shaped data, formula `reaction ~ 1 + days + (1 + days | subj)`.\n\n\
         Sources: **{}** vs **{}**.\n\n\
         Each side runs 3 warmup + 5 measured fits; medians reported.\n\n",
        rust.tool, r.tool
    ));

    report.push_str("| label | n | t_R median (ms) | t_Rust median (ms) | speedup (median) | t_Rust min | t_R min | speedup (min) | Rust fevals | R fevals | Δ obj | Δ σ |\n");
    report.push_str("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");

    for rs in &rust.results {
        let rr = r.results.iter().find(|x| x.label == rs.label);
        let row_r = match rr {
            Some(x) => x,
            None => {
                report.push_str(&format!(
                    "| {} | {} | — | {:.2} | — | {:.2} | — | — | {} | — | — | — |\n",
                    rs.label, rs.n_obs, rs.fit_time_ms_median, rs.fit_time_ms_min, rs.fevals
                ));
                continue;
            }
        };
        let speedup_med = row_r.fit_time_ms_median / rs.fit_time_ms_median;
        let speedup_min = row_r.fit_time_ms_min / rs.fit_time_ms_min;
        let d_obj = (rs.objective - row_r.objective).abs();
        let d_sigma = (rs.sigma - row_r.sigma).abs();
        report.push_str(&format!(
            "| {} | {} | {:.1} | {:.2} | **{:.1}×** | {:.2} | {:.1} | {:.1}× | {} | {} | {:.4} | {:.4} |\n",
            rs.label,
            rs.n_obs,
            row_r.fit_time_ms_median,
            rs.fit_time_ms_median,
            speedup_med,
            rs.fit_time_ms_min,
            row_r.fit_time_ms_min,
            speedup_min,
            rs.fevals,
            row_r.fevals.map(|x| x.to_string()).unwrap_or("—".into()),
            d_obj,
            d_sigma,
        ));
    }

    report.push_str("\n## Rust phase breakdown (median)\n\n");
    report.push_str("| label | n | parse + build | fit (optimizer) | total | optimizer |\n");
    report.push_str("|---|---:|---:|---:|---:|---|\n");
    for rs in &rust.results {
        report.push_str(&format!(
            "| {} | {} | {:.2} ms | {:.2} ms | {:.2} ms | {} |\n",
            rs.label,
            rs.n_obs,
            rs.parse_build_ms_median,
            rs.fit_only_ms_median,
            rs.fit_time_ms_median,
            rs.optimizer,
        ));
    }

    let out = comparison_root().join("REPORT.md");
    fs::write(&out, &report)?;
    println!("wrote {}", out.display());
    Ok(())
}
