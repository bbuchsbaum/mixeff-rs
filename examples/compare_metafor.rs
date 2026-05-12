//! Cross-check the Rust summary-estimate (meta-analysis) LMM front door
//! against `metafor::rma.mv` on the Berkey et al. (1995) BCG vaccine
//! dataset.
//!
//! Companion: `scripts/compare_metafor.R` writes
//! `comparison/metafor/bcg_yi_vi.csv` (the shared input fixture) and
//! `comparison/metafor/metafor_results.json` (the R-side fit). This
//! example reads the same CSV, fits via
//! `LinearMixedModel::from_summary_estimates`, and writes
//! `comparison/metafor/rust_results.json` with the same fields so the
//! two JSON files can be diffed directly.
//!
//! Run after the R script:
//!     Rscript scripts/compare_metafor.R
//!     cargo run --release --example compare_metafor

use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use serde_json::json;

use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::data::DataFrame;
use mixeff_rs::model::linear::LinearMixedModel;
use mixeff_rs::model::summary_estimates::SummaryEstimateOptions;
use mixeff_rs::model::traits::MixedModelFit;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let repo_root = find_repo_root()?;
    let csv_path = repo_root
        .join("comparison")
        .join("metafor")
        .join("bcg_yi_vi.csv");
    let out_path = repo_root
        .join("comparison")
        .join("metafor")
        .join("rust_results.json");

    if !csv_path.exists() {
        eprintln!(
            "missing fixture {} — run `Rscript scripts/compare_metafor.R` first.",
            csv_path.display()
        );
        std::process::exit(1);
    }

    let (trials, yi, vi) = read_bcg_csv(&csv_path)?;
    let n = trials.len();

    let mut df = DataFrame::new();
    df.add_numeric("yi", yi.clone())?;
    df.add_numeric("vi", vi.clone())?;
    let trial_levels: Vec<String> = trials.iter().map(|t| t.to_string()).collect();
    df.add_categorical("trial", trial_levels)?;

    let formula = parse_formula("yi ~ 1 + (1 | trial)")?;
    let mut model = LinearMixedModel::from_summary_estimates(
        formula,
        &df,
        "yi",
        "vi",
        SummaryEstimateOptions::default(),
    )?;
    model.fit(true)?;

    let beta = model.coef();
    let vcov = model.vcov();
    let se = vcov[(0, 0)].sqrt();
    let tau_sd = model.varcorr().components[0].std_dev[0];
    let tau_sq = tau_sd * tau_sd;
    let log_lik = model.loglikelihood();

    let payload = json!({
        "schema_version": 1,
        "source": "mixeff_rs::LinearMixedModel::from_summary_estimates",
        "method": "REML",
        "fixture": "Berkey 1995 BCG vaccine (n=13)",
        "formula": "yi ~ 1 + (1 | trial)",
        "n_studies": n,
        "beta": [beta[0]],
        "se":   [se],
        "vcov_beta": [[vcov[(0, 0)]]],
        "tau_sq": tau_sq,
        "tau_sd": tau_sd,
        "log_likelihood": log_lik,
    });

    fs::write(&out_path, serde_json::to_string_pretty(&payload)? + "\n")?;
    println!("wrote {}", out_path.display());
    println!(
        "beta = {}, tau^2 = {}, logLik = {}",
        beta[0], tau_sq, log_lik
    );

    let metafor_path = repo_root
        .join("comparison")
        .join("metafor")
        .join("metafor_results.json");
    if metafor_path.exists() {
        report_parity(&metafor_path, beta[0], se, tau_sq, log_lik)?;
    } else {
        println!(
            "metafor_results.json not present — skipping numerical parity check.\n\
             Install metafor and run scripts/compare_metafor.R to generate it."
        );
    }

    Ok(())
}

fn report_parity(
    metafor_path: &PathBuf,
    rust_beta: f64,
    rust_se: f64,
    rust_tau_sq: f64,
    rust_loglik: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let raw = fs::read_to_string(metafor_path)?;
    let payload: serde_json::Value = serde_json::from_str(&raw)?;

    let metafor_beta = first_scalar(&payload["beta"]).ok_or("metafor beta missing")?;
    let metafor_se = first_scalar(&payload["se"]).ok_or("metafor se missing")?;
    let metafor_tau_sq = payload["tau_sq"]
        .as_f64()
        .ok_or("metafor tau_sq missing")?;
    let metafor_loglik = payload["log_likelihood"]
        .as_f64()
        .ok_or("metafor log_likelihood missing")?;

    println!();
    println!("Numerical parity vs metafor::rma.mv:");
    let pairs = [
        ("beta",   rust_beta,   metafor_beta,   1e-6),
        ("se",     rust_se,     metafor_se,     1e-5),
        ("tau_sq", rust_tau_sq, metafor_tau_sq, 1e-4),
    ];
    let mut failed = 0u32;
    for (name, rust, refv, tol) in pairs {
        let diff = (rust - refv).abs();
        let ok = diff <= tol;
        let mark = if ok { "OK" } else { "FAIL" };
        println!(
            "  {mark:>4} {name:<10} rust={rust:.10} metafor={refv:.10} |delta|={diff:.3e} tol={tol:.0e}"
        );
        if !ok {
            failed += 1;
        }
    }

    // log-likelihood is informational only: metafor and the lme4-style
    // PLS REML log-likelihood used by this crate differ by a constant
    // normalization term (the convention split is documented at
    // https://www.metafor-project.org/doku.php/tips:rma_vs_lm_lme_lmer
    // among other places). The constant is study-set-dependent but
    // model-independent, so likelihood-ratio statistics on the same
    // dataset still match across packages.
    let ll_diff = (rust_loglik - metafor_loglik).abs();
    println!(
        "  INFO log_lik   rust={rust_loglik:.10} metafor={metafor_loglik:.10} |delta|={ll_diff:.3e} \
         (REML normalization differs across packages; not a parity failure)"
    );

    if failed > 0 {
        eprintln!("{failed} inferential field(s) outside tolerance — see above.");
        std::process::exit(2);
    }
    println!("All inferential fields within tolerance.");
    Ok(())
}

fn find_repo_root() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let mut here = std::env::current_dir()?;
    loop {
        if here.join("Cargo.toml").exists() {
            return Ok(here);
        }
        if !here.pop() {
            return Err("could not find Cargo.toml ancestor".into());
        }
    }
}

fn read_bcg_csv(
    path: &PathBuf,
) -> Result<(Vec<i64>, Vec<f64>, Vec<f64>), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    let header = lines.next().ok_or("empty CSV")??;
    let header_cols: Vec<&str> = header.split(',').map(str::trim).collect();
    let trial_idx = header_cols
        .iter()
        .position(|c| *c == "trial" || *c == "\"trial\"")
        .ok_or("missing trial column")?;
    let yi_idx = header_cols
        .iter()
        .position(|c| *c == "yi" || *c == "\"yi\"")
        .ok_or("missing yi column")?;
    let vi_idx = header_cols
        .iter()
        .position(|c| *c == "vi" || *c == "\"vi\"")
        .ok_or("missing vi column")?;

    let mut trials = Vec::new();
    let mut yi = Vec::new();
    let mut vi = Vec::new();
    for line in lines {
        let row = line?;
        if row.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = row.split(',').map(str::trim).collect();
        trials.push(strip_quotes(cols[trial_idx]).parse::<i64>()?);
        yi.push(strip_quotes(cols[yi_idx]).parse::<f64>()?);
        vi.push(strip_quotes(cols[vi_idx]).parse::<f64>()?);
    }
    Ok((trials, yi, vi))
}

fn strip_quotes(s: &str) -> &str {
    s.trim_matches('"')
}

/// Read a numeric value that R / jsonlite may have written either as a
/// scalar (after `auto_unbox = TRUE`) or as a length-1 array.
fn first_scalar(value: &serde_json::Value) -> Option<f64> {
    if let Some(x) = value.as_f64() {
        return Some(x);
    }
    value.as_array().and_then(|a| a.first()).and_then(|v| v.as_f64())
}
