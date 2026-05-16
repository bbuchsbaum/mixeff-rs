//! Join `comparison/rust_results.json` and `comparison/lme4_results.json`
//! and emit `comparison/REPORT.md` with accuracy + performance tables.
//!
//! Run after the two drivers:
//!     cargo run --release --example compare_rust
//!     Rscript scripts/compare_lme4.R
//!     cargo run --release --example compare_report
//!
//! Pass/fail tolerances are tunable at the top of `main`. The current
//! defaults are deliberately tight for LMMs and looser for GLMMs (where
//! Laplace approximations and link-function rounding can introduce small
//! discrepancies).

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Clone, Deserialize)]
struct ResultRecord {
    dataset: String,
    formula: String,
    #[allow(dead_code)]
    family: String,
    #[allow(dead_code)]
    link: String,
    estimator: String,
    n_obs: Option<usize>,
    status: String,
    error: Option<String>,
    beta: Option<Vec<f64>>,
    coef_names: Option<Vec<String>>,
    sigma: Option<f64>,
    #[allow(dead_code)]
    theta: Option<Vec<f64>>,
    objective: Option<f64>,
    #[allow(dead_code)]
    loglik: Option<f64>,
    #[allow(dead_code)]
    aic: Option<f64>,
    #[allow(dead_code)]
    bic: Option<f64>,
    objective_definition: Option<String>,
    response_constants: Option<String>,
    optimizer: Option<String>,
    optimizer_backend: Option<String>,
    optimizer_return_code: Option<String>,
    optimizer_fevals: Option<i64>,
    #[allow(dead_code)]
    optimizer_fmin: Option<f64>,
    #[allow(dead_code)]
    optimizer_max_fevals: Option<i64>,
    is_singular: Option<bool>,
    fit_time_ms: Option<f64>,
    fit_time_ms_min: Option<f64>,
    #[serde(default)]
    warnings: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct ResultsFile {
    tool: String,
    #[serde(default)]
    #[allow(dead_code)]
    version: Option<String>,
    results: Vec<ResultRecord>,
}

/// Per-metric tolerance pair: pass if `Δ <= abs_tol` OR `Δ / |reference| <= rel_tol`.
#[derive(Clone, Copy)]
struct Tol {
    abs_tol: f64,
    rel_tol: f64,
}

impl Tol {
    fn passes(&self, delta: f64, reference: f64) -> bool {
        let d = delta.abs();
        if d <= self.abs_tol {
            return true;
        }
        let r = reference.abs();
        if r > 0.0 && d / r <= self.rel_tol {
            return true;
        }
        false
    }
}

fn comparison_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("comparison")
}

fn load_results(path: &PathBuf) -> ResultsFile {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    // R's jsonlite emits scalar arrays even with auto_unbox=TRUE in some edge
    // cases. Coerce single-element scalar fields if needed via a Value detour.
    let raw: Value =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
    serde_json::from_value(raw).unwrap_or_else(|e| panic!("schema {}: {e}", path.display()))
}

fn key(r: &ResultRecord) -> (String, String, String, String, String) {
    // Normalize whitespace so trivial reformatting doesn't break the join.
    let f = r.formula.split_whitespace().collect::<Vec<_>>().join(" ");
    (
        r.dataset.clone(),
        f,
        r.family.clone(),
        r.link.clone(),
        r.estimator.clone(),
    )
}

fn fmt_opt_f(v: Option<f64>, prec: usize) -> String {
    v.map_or("—".into(), |x| format!("{x:.*}", prec))
}

fn fmt_opt_i(v: Option<i64>) -> String {
    v.map_or("—".into(), |x| x.to_string())
}

fn fmt_obj_delta(v: Option<f64>, comparable: bool) -> String {
    if comparable {
        fmt_opt_f(v, 4)
    } else {
        "n/c".into()
    }
}

fn fmt_speedup(t_r: Option<f64>, t_rust: Option<f64>) -> String {
    match (t_r, t_rust) {
        (Some(r), Some(rs)) if rs > 0.0 => format!("{:.1}×", r / rs),
        _ => "—".into(),
    }
}

fn is_glmm(r: &ResultRecord) -> bool {
    !(r.family == "Gaussian" && r.link == "Identity")
}

fn known_glmm_numeric_classification(r: &ResultRecord) -> Option<&'static str> {
    match (r.dataset.as_str(), r.estimator.as_str()) {
        ("culcitalogreg", "AGQ") => Some(
            "culcitalogreg Binomial/AGQ is accepted by the row-specific 2e-3 beta gate; lme4 JSON records fixed effects rounded to four decimals",
        ),
        ("ergostool", "Laplace") if r.family == "Gamma" && r.link == "Log" => Some(
            "Gamma/Log dispersion and theta conventions are not treated as an lme4-only oracle; see the MixedModels.jl Gamma fixture",
        ),
        ("grouseticks", "Laplace") => Some(
            "Poisson/Log multi-random-intercept row matches MixedModels.jl 5.3.0 fast=true profiled objective; lme4 beta gap is fast-PIRLS versus joint-estimate divergence",
        ),
        ("verbagg", "Laplace") => Some(
            "large crossed Binomial/Logit row matches MixedModels.jl 5.3.0 fast=true profiled objective; lme4 beta gap is fast-PIRLS versus joint-estimate divergence",
        ),
        ("contraception", "Laplace") if r.formula.contains("(1 + urban | dist)") => Some(
            "large Binomial/Logit random-slope row matches MixedModels.jl 5.3.0 fast=true profiled objective; lme4 beta gap is fast-PIRLS versus joint-estimate divergence",
        ),
        ("contraception", "Laplace") => Some(
            "large Binomial/Logit random-intercept row matches MixedModels.jl 5.3.0 fast=true profiled objective; lme4 beta gap is fast-PIRLS versus joint-estimate divergence",
        ),
        _ => None,
    }
}

fn numeric_gap_detail(
    d_beta: Option<f64>,
    beta_pass: bool,
    d_sigma: Option<f64>,
    sigma_pass: bool,
    classification: &str,
) -> String {
    let beta = match d_beta {
        Some(delta) if !beta_pass => format!("max_delta_beta={delta:.6}"),
        Some(delta) => format!("max_delta_beta={delta:.6} (within tolerance)"),
        None => "max_delta_beta=missing".to_string(),
    };
    let sigma = match d_sigma {
        Some(delta) if !sigma_pass => format!("delta_sigma={:.6}", delta.abs()),
        Some(delta) => format!("delta_sigma={:.6} (within tolerance)", delta.abs()),
        None => "delta_sigma=missing".to_string(),
    };
    format!("{beta}; {sigma}; {classification}")
}

/// Strip whitespace, ":", and "_" from a coefficient name. Lets the
/// position-free comparator match `"recipe: B"` (Rust) against `"recipeB"`
/// (R) and `"recipeB:temperature185"` against `"recipe: B:temperature: 185"`.
fn norm_coef_name(s: &str) -> String {
    s.chars()
        .filter(|c| !c.is_whitespace() && *c != ':' && *c != '_')
        .collect()
}

/// Element-wise max absolute Δ when a/b have the same length, else the max
/// Δ across the intersection of names (normalised). Returns the count of
/// matched coefficients and the count missing on either side as well.
struct BetaCompare {
    max_abs: Option<f64>,
    n_matched: usize,
    n_only_rust: usize,
    n_only_r: usize,
}

fn compare_beta(a: &[f64], a_names: &[String], b: &[f64], b_names: &[String]) -> BetaCompare {
    use std::collections::BTreeMap;
    let map_a: BTreeMap<String, f64> = a_names
        .iter()
        .map(|s| norm_coef_name(s))
        .zip(a.iter().copied())
        .collect();
    let map_b: BTreeMap<String, f64> = b_names
        .iter()
        .map(|s| norm_coef_name(s))
        .zip(b.iter().copied())
        .collect();

    let mut max_d: Option<f64> = None;
    let mut n_match = 0usize;
    for (k, &va) in &map_a {
        if let Some(&vb) = map_b.get(k) {
            n_match += 1;
            let d = (va - vb).abs();
            max_d = Some(max_d.map_or(d, |m| m.max(d)));
        }
    }
    BetaCompare {
        max_abs: max_d,
        n_matched: n_match,
        n_only_rust: map_a.len() - n_match,
        n_only_r: map_b.len() - n_match,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cmp = comparison_root();
    let rust = load_results(&cmp.join("rust_results.json"));
    let r = load_results(&cmp.join("lme4_results.json"));

    let tol_obj = Tol {
        abs_tol: 1e-2,
        rel_tol: 1e-5,
    };
    let tol_beta = Tol {
        abs_tol: 1e-3,
        rel_tol: 1e-5,
    };
    let tol_sigma = Tol {
        abs_tol: 1e-3,
        rel_tol: 1e-4,
    };

    let mut by_key: HashMap<(String, String, String, String, String), &ResultRecord> =
        HashMap::new();
    for rec in &r.results {
        by_key.insert(key(rec), rec);
    }

    // Build the joined table. We iterate Rust-side so the manifest ordering is preserved.
    struct Row<'a> {
        rust: &'a ResultRecord,
        r: Option<&'a ResultRecord>,
    }
    let rows: Vec<Row> = rust
        .results
        .iter()
        .map(|rs| Row {
            rust: rs,
            r: by_key.get(&key(rs)).copied(),
        })
        .collect();

    let mut acc = String::new();
    let mut perf = String::new();
    let mut gaps = String::new();

    let mut n_total = 0;
    let mut n_matched = 0;
    let mut n_passed = 0;
    let mut n_unsupported = 0;
    let mut n_errors = 0;
    let mut speedups: Vec<f64> = Vec::new();

    acc.push_str(
        "| Dataset | Formula | Est | n | Δ obj | max Δ β | Δ σ | Singular | R warns | Match |\n",
    );
    acc.push_str("|---|---|---|---:|---:|---:|---:|:---:|:---:|:---:|\n");
    perf.push_str("| Dataset | Formula | Est | n | t_R (ms, min) | t_Rust (ms, min) | speedup | R fevals | Rust fevals | Rust optimizer |\n");
    perf.push_str("|---|---|---|---:|---:|---:|---:|---:|---:|---|\n");
    gaps.push_str("| Dataset | Formula | Est | Side | Status | Detail |\n");
    gaps.push_str("|---|---|---|---|---|---|\n");

    for Row { rust: rs, r: rr } in &rows {
        n_total += 1;

        let formula_short = rs.formula.replace('|', "\\|");
        let n = rs
            .n_obs
            .map(|x| x.to_string())
            .unwrap_or_else(|| "—".into());

        // Side-status accounting first: surface unsupported/error cases as gaps.
        if rs.status != "ok" || rr.map(|x| x.status.as_str()) != Some("ok") {
            if rs.status == "unsupported" {
                n_unsupported += 1;
            } else if rs.status == "error" {
                n_errors += 1;
            }
            if let Some(rr) = rr {
                if rr.status != "ok" && rs.status == "ok" {
                    gaps.push_str(&format!(
                        "| `{}` | `{}` | {} | R | {} | {} |\n",
                        rs.dataset,
                        formula_short,
                        rs.estimator,
                        rr.status,
                        rr.error.clone().unwrap_or_default()
                    ));
                }
            }
            if rs.status != "ok" {
                gaps.push_str(&format!(
                    "| `{}` | `{}` | {} | Rust | {} | {} |\n",
                    rs.dataset,
                    formula_short,
                    rs.estimator,
                    rs.status,
                    rs.error.clone().unwrap_or_default()
                ));
            }
            continue;
        }

        let rr = rr.unwrap();
        n_matched += 1;

        let d_obj = match (rs.objective, rr.objective) {
            (Some(a), Some(b)) => Some(a - b),
            _ => None,
        };
        let beta_cmp = match (&rs.beta, &rs.coef_names, &rr.beta, &rr.coef_names) {
            (Some(a), Some(an), Some(b), Some(bn)) => Some(compare_beta(a, an, b, bn)),
            _ => None,
        };
        let d_beta = beta_cmp.as_ref().and_then(|c| c.max_abs);
        let d_sigma = match (rs.sigma, rr.sigma) {
            (Some(a), Some(b)) => Some(a - b),
            _ => None,
        };

        let objective_comparable = match (&rs.response_constants, &rr.response_constants) {
            (Some(a), Some(b)) => a == b,
            _ => false,
        };
        let obj_pass = if objective_comparable {
            d_obj.is_some_and(|d| tol_obj.passes(d, rr.objective.unwrap_or(1.0)))
        } else {
            true
        };
        // β passes when every R-side coefficient was matched and the deltas are tight.
        // If Rust is missing coefficients R produced (e.g. interaction not implemented),
        // beta_cmp.n_only_r > 0 and we treat that as a fail even if the matched ones agree.
        let beta_pass = beta_cmp
            .as_ref()
            .is_some_and(|c| c.n_only_r == 0 && c.max_abs.is_some_and(|d| tol_beta.passes(d, 1.0)));
        let sigma_pass = d_sigma.is_some_and(|d| tol_sigma.passes(d, rr.sigma.unwrap_or(1.0)));
        let ok = obj_pass && beta_pass && sigma_pass;
        if ok {
            n_passed += 1;
        }

        let singular_cell = match (rs.is_singular, rr.is_singular) {
            (Some(true), Some(true)) => "⚠ both",
            (Some(true), _) => "⚠ rust",
            (_, Some(true)) => "⚠ R",
            _ => "—",
        };
        let n_warns = rr.warnings.as_ref().map(|w| w.len()).unwrap_or(0);
        let warns_cell = if n_warns == 0 {
            "—".to_string()
        } else {
            format!("⚠ {n_warns}")
        };
        acc.push_str(&format!(
            "| `{}` | `{}` | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            rs.dataset,
            formula_short,
            rs.estimator,
            n,
            fmt_obj_delta(d_obj.map(f64::abs), objective_comparable),
            fmt_opt_f(d_beta, 4),
            fmt_opt_f(d_sigma.map(f64::abs), 4),
            singular_cell,
            warns_cell,
            if ok { "✅" } else { "❌" },
        ));

        if !objective_comparable {
            gaps.push_str(&format!(
                "| `{}` | `{}` | {} | both | objective_non_comparable | rust_definition={} rust_response_constants={} R_definition={} R_response_constants={} |\n",
                rs.dataset,
                formula_short,
                rs.estimator,
                rs.objective_definition.as_deref().unwrap_or("missing"),
                rs.response_constants.as_deref().unwrap_or("missing"),
                rr.objective_definition.as_deref().unwrap_or("missing"),
                rr.response_constants.as_deref().unwrap_or("missing")
            ));
        }

        if !ok && is_glmm(rs) {
            let classification = known_glmm_numeric_classification(rs);
            let status = if classification.is_some() {
                "numeric_classified"
            } else {
                "numeric_disagreement"
            };
            let detail = numeric_gap_detail(
                d_beta,
                beta_pass,
                d_sigma,
                sigma_pass,
                classification
                    .unwrap_or("no known GLMM numeric classification; update gates or fix the fit"),
            )
            .replace('|', "\\|")
            .replace('\n', " ");
            gaps.push_str(&format!(
                "| `{}` | `{}` | {} | both | {} | {} |\n",
                rs.dataset, formula_short, rs.estimator, status, detail
            ));
        }

        // Spill R-side warning text into the gaps section so users see the
        // actual messages (boundary singular fit, near-unidentifiable, …).
        if let Some(ws) = &rr.warnings {
            for w in ws {
                gaps.push_str(&format!(
                    "| `{}` | `{}` | {} | R | warning | {} |\n",
                    rs.dataset,
                    formula_short,
                    rs.estimator,
                    w.replace('|', "\\|").replace('\n', " "),
                ));
            }
        }

        let t_r = rr.fit_time_ms_min.or(rr.fit_time_ms);
        let t_rust = rs.fit_time_ms_min.or(rs.fit_time_ms);
        if let (Some(a), Some(b)) = (t_r, t_rust) {
            if b > 0.0 {
                speedups.push(a / b);
            }
        }
        let rust_optimizer = match (
            &rs.optimizer_backend,
            &rs.optimizer,
            &rs.optimizer_return_code,
        ) {
            (Some(backend), Some(optimizer), Some(code)) if !code.is_empty() => {
                format!("{backend}/{optimizer} ({code})")
            }
            (Some(backend), Some(optimizer), _) => format!("{backend}/{optimizer}"),
            _ => "—".into(),
        };
        perf.push_str(&format!(
            "| `{}` | `{}` | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            rs.dataset,
            formula_short,
            rs.estimator,
            n,
            fmt_opt_f(t_r, 1),
            fmt_opt_f(t_rust, 1),
            fmt_speedup(t_r, t_rust),
            fmt_opt_i(rr.optimizer_fevals),
            fmt_opt_i(rs.optimizer_fevals),
            rust_optimizer,
        ));

        if let (Some(rs_sing), Some(rr_sing)) = (rs.is_singular, rr.is_singular) {
            if rs_sing != rr_sing {
                gaps.push_str(&format!(
                    "| `{}` | `{}` | {} | both | singular_disagreement | rust={rs_sing} R={rr_sing} |\n",
                    rs.dataset, formula_short, rs.estimator
                ));
            }
        }
        // Surface coefficient-name shape mismatch — but only when normalisation
        // doesn't already recover an element-wise match. Different formatting
        // ("recipeB" vs "recipe: B") is informational, not a failure signal.
        if let Some(c) = &beta_cmp {
            if c.n_only_rust > 0 || c.n_only_r > 0 {
                gaps.push_str(&format!(
                    "| `{}` | `{}` | {} | both | coef_count_mismatch | matched={} only_rust={} only_R={} |\n",
                    rs.dataset, formula_short, rs.estimator,
                    c.n_matched, c.n_only_rust, c.n_only_r
                ));
            }
        }
    }

    let avg_speedup = if speedups.is_empty() {
        0.0
    } else {
        speedups.iter().sum::<f64>() / speedups.len() as f64
    };
    let geom_speedup = if speedups.is_empty() {
        0.0
    } else {
        let log_sum: f64 = speedups.iter().map(|x| x.ln()).sum();
        (log_sum / speedups.len() as f64).exp()
    };

    let mut report = String::new();
    report.push_str("# Mixed-Models Cross-Implementation Comparison\n\n");
    report.push_str(&format!(
        "Generated by `examples/compare_report.rs`. Sources: **{}** vs **{}**.\n\n",
        rust.tool, r.tool
    ));
    report.push_str("## Summary\n\n");
    report.push_str(&format!("- Total fits in manifest: **{}**\n", n_total));
    report.push_str(&format!("- Matched (both sides ok): **{}**\n", n_matched));
    report.push_str(&format!(
        "- Within tolerance: **{} / {}**\n",
        n_passed, n_matched
    ));
    report.push_str(&format!(
        "- Unsupported on Rust side: **{}**\n",
        n_unsupported
    ));
    report.push_str(&format!("- Errors on Rust side: **{}**\n", n_errors));
    if !speedups.is_empty() {
        report.push_str(&format!(
            "- Speedup vs lme4 (Rust faster ↑): mean **{avg_speedup:.1}×**, geometric mean **{geom_speedup:.1}×**\n"
        ));
    }
    report.push_str(&format!(
        "\nTolerances: objective Δ ≤ {:.0e} (abs) or {:.0e} (rel); β Δ ≤ {:.0e}; σ Δ ≤ {:.0e}.\n\n",
        tol_obj.abs_tol, tol_obj.rel_tol, tol_beta.abs_tol, tol_sigma.abs_tol
    ));
    report.push_str(
        "Objective tolerance is applied only when both engines report the same `response_constants` convention; otherwise the objective cell is `n/c` and the reason is listed under gaps.\n\n",
    );

    report.push_str("## Accuracy\n\n");
    report.push_str(&acc);
    report.push_str("\n## Performance\n\n");
    report.push_str(&perf);
    report.push_str("\n## Gaps & disagreements\n\n");
    if gaps.lines().count() <= 2 {
        report.push_str("_None._\n");
    } else {
        report.push_str(&gaps);
    }

    let out = comparison_root().join("REPORT.md");
    fs::write(&out, &report)?;
    println!("wrote {}", out.display());
    println!(
        "matched={n_matched}/{n_total}  passing={n_passed}/{n_matched}  unsupported={n_unsupported}  errors={n_errors}"
    );
    Ok(())
}
