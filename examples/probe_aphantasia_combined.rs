//! Diagnostics for bd-01KTQJFZNF5H034B5WKWKJQRDF: the joint GLMM optimizer on
//! the aphantasia COMBINED model used to declare FTOL convergence after ~99
//! evals from the profiled start and fall back to fast-PIRLS.
//!
//! Default mode fits the native (4-theta, `||`) formula profiled + joint and
//! dumps the joint descent trajectory. Findings from the bead investigation:
//! the premature stop is fixed by the descent-gated trust_bq stagnation stop;
//! the residual ~1.9 logLik gap to lme4 is a model-family difference (lme4's
//! `||` keeps the within-factor mask correlation; the native `||` drops it),
//! and on the lme4-exact expanded formula the joint optimizer reaches and
//! slightly beats glmer's optimum.
//!
//!   cargo run --release --no-default-features --features unstable-internals \
//!       --example probe_aphantasia_combined [max_feval]
//!
//! Probe modes (env vars, mutually exclusive):
//! - `MIXEFF_PROBE_JOINT6`: standard joint fit on the lme4-exact expanded
//!   formula `(1|p) + (0+mask|p) + (0+soa_s|p) + (1|item)`.
//! - `MIXEFF_PROBE_SEED`: joint fit on the expanded formula seeded at lme4's
//!   published theta (reachability test).
//! - `MIXEFF_PROBE_INTACT`: objective parity cross-checks on the intact
//!   subset (profiled vs joint criterion at lme4's theta, 6- and 4-theta fits).
//! - `MIXEFF_PROBE_PIRLS_BUDGET`: joint-criterion evaluations at lme4's theta
//!   under both mask-block Cholesky name assignments.
//! - `MIXEFF_PROBE_LME4_FAMILY`: profiled-criterion evaluations + fit on the
//!   expanded formula, with a warm-path walk toward lme4's theta.
//! - `MIXEFF_PROBE_SKIP_JOINT`: stop after the profiled fit and theta
//!   permutation table.

use std::path::PathBuf;

use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::traits::MixedModelFit;
use mixeff_rs::model::{DataFrame, Family, GeneralizedLinearMixedModel};
use serde_json::Value;

fn fixture_path(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/aphantasia")
        .join(relative)
}

fn load_combined() -> Result<DataFrame, Box<dyn std::error::Error>> {
    let numeric_columns = ["correct", "soa_s"];
    let categorical_columns = ["participant", "item", "group", "mask", "block", "stimtype"];
    let path = fixture_path("prepared/combined.csv");
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(&path)?;
    let headers = rdr.headers()?.clone();
    let numeric_idx = numeric_columns
        .iter()
        .map(|column| headers.iter().position(|header| header == *column).unwrap())
        .collect::<Vec<_>>();
    let categorical_idx = categorical_columns
        .iter()
        .map(|column| headers.iter().position(|header| header == *column).unwrap())
        .collect::<Vec<_>>();
    let mut numeric_data = vec![Vec::new(); numeric_columns.len()];
    let mut categorical_data = vec![Vec::new(); categorical_columns.len()];
    for record in rdr.records() {
        let record = record?;
        for (slot, &idx) in numeric_idx.iter().enumerate() {
            numeric_data[slot].push(record.get(idx).unwrap().parse::<f64>()?);
        }
        for (slot, &idx) in categorical_idx.iter().enumerate() {
            categorical_data[slot].push(record.get(idx).unwrap().to_string());
        }
    }
    let mut data = DataFrame::new();
    for (column, values) in numeric_columns.iter().zip(numeric_data) {
        data.add_numeric(column, values)?;
    }
    for (column, values) in categorical_columns.iter().zip(categorical_data) {
        data.add_categorical(column, values)?;
    }
    Ok(data)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let max_feval = std::env::args().nth(1).map(|raw| raw.parse::<i64>().unwrap());
    let reference: Value =
        serde_json::from_str(&std::fs::read_to_string(fixture_path("reference.json"))?)?;
    let reference_loglik = reference["models"]["combined"]["logLik"].as_f64().unwrap();

    let data = load_combined()?;
    let formula = parse_formula(
        "correct ~ group * mask * soa_s * stimtype + block + (1 + mask + soa_s || participant) + (1 | item)",
    )?;

    // lme4 reference theta components. lme4's || expansion gives mask a 2x2
    // Cholesky block [maskmasked=8.5e-5; cross=0.3125; maskunmasked=0.4583].
    let lme4_part_int = 0.4004889448009537_f64;
    let lme4_part_mask = (0.3125031504858162_f64.powi(2) + 0.4583396899603748_f64.powi(2)).sqrt();
    let lme4_part_soa = 0.0913524051423621_f64;
    let lme4_item_int = 1.151895898039061_f64;

    // Standard joint fit (profiled start, default budget) on the lme4-exact
    // 6-theta family — the configuration downstream would run if it switched
    // combined to the explicit || expansion.
    if std::env::var("MIXEFF_PROBE_JOINT6").is_ok() {
        let lme4_formula = parse_formula(
            "correct ~ group * mask * soa_s * stimtype + block + (1 | participant) + (0 + mask | participant) + (0 + soa_s | participant) + (1 | item)",
        )?;
        let mut joint6 =
            GeneralizedLinearMixedModel::new(lme4_formula, &data, Family::Bernoulli, None)?;
        if let Some(max_feval) = max_feval {
            joint6.lmm_mut().optsum_mut().max_feval = max_feval;
        }
        joint6.fit_with_options(false, 1, false)?;
        let optsum = joint6.lmm().optsum();
        println!(
            "[combined joint6 default-start] logLik={:.4} gap={:.4} feval={} status={} fmin={:.4} finitial={:.4} theta={:?}",
            joint6.loglikelihood(),
            joint6.loglikelihood() - reference_loglik,
            optsum.feval,
            optsum.return_value,
            optsum.fmin,
            optsum.finitial,
            joint6.theta()
        );
        if let Some(cert) = joint6.compiler_artifact().optimizer_certificate.as_ref() {
            println!("fit_status: {:?}", cert.status);
        }
        return Ok(());
    }

    // Seed the joint optimizer at lme4's theta (6-theta family, beta from
    // fast-PIRLS at that theta) and see whether it descends to the reference
    // deviance — reachability test for the optimizer rather than the surface.
    if std::env::var("MIXEFF_PROBE_SEED").is_ok() {
        let lme4_formula = parse_formula(
            "correct ~ group * mask * soa_s * stimtype + block + (1 | participant) + (0 + mask | participant) + (0 + soa_s | participant) + (1 | item)",
        )?;
        let lme4_theta = [
            lme4_item_int,
            0.4583396899603748,
            0.3125031504858162,
            8.509615040182939e-05,
            lme4_part_int,
            lme4_part_soa,
        ];
        let mut seeded =
            GeneralizedLinearMixedModel::new(lme4_formula, &data, Family::Bernoulli, None)?;
        if let Some(max_feval) = max_feval {
            seeded.lmm_mut().optsum_mut().max_feval = max_feval;
        }
        seeded.fit_joint_glmm_from_custom_theta(&lme4_theta, 1)?;
        let optsum = seeded.lmm().optsum();
        println!(
            "[combined seeded@lme4theta] logLik={:.4} gap={:.4} feval={} status={} fmin={:.4} finitial={:.4} theta={:?}",
            seeded.loglikelihood(),
            seeded.loglikelihood() - reference_loglik,
            optsum.feval,
            optsum.return_value,
            optsum.fmin,
            optsum.finitial,
            seeded.theta()
        );
        let mut best = f64::INFINITY;
        for (index, entry) in optsum.fit_log.iter().enumerate() {
            if entry.objective < best {
                best = entry.objective;
                println!("  eval {:>4}: objective={:.6} (new best)", index + 1, entry.objective);
            }
        }
        return Ok(());
    }

    // Cross-check on the INTACT subset: evaluate our objective at lme4's
    // intact theta in lme4's exact (6-theta) family. lme4 reports
    // logLik = -1297.8856 (deviance 2595.7711) there.
    if std::env::var("MIXEFF_PROBE_INTACT").is_ok() {
        let intact_reference_loglik = reference["models"]["intact"]["logLik"].as_f64().unwrap();
        let intact_data = {
            let numeric_columns = ["correct", "soa_s"];
            let categorical_columns = ["participant", "item", "group", "mask", "block"];
            let path = fixture_path("prepared/intact.csv");
            let mut rdr = csv::ReaderBuilder::new()
                .has_headers(true)
                .from_path(&path)?;
            let headers = rdr.headers()?.clone();
            let numeric_idx = numeric_columns
                .iter()
                .map(|column| headers.iter().position(|header| header == *column).unwrap())
                .collect::<Vec<_>>();
            let categorical_idx = categorical_columns
                .iter()
                .map(|column| headers.iter().position(|header| header == *column).unwrap())
                .collect::<Vec<_>>();
            let mut numeric_data = vec![Vec::new(); numeric_columns.len()];
            let mut categorical_data = vec![Vec::new(); categorical_columns.len()];
            for record in rdr.records() {
                let record = record?;
                for (slot, &idx) in numeric_idx.iter().enumerate() {
                    numeric_data[slot].push(record.get(idx).unwrap().parse::<f64>()?);
                }
                for (slot, &idx) in categorical_idx.iter().enumerate() {
                    categorical_data[slot].push(record.get(idx).unwrap().to_string());
                }
            }
            let mut frame = DataFrame::new();
            for (column, values) in numeric_columns.iter().zip(numeric_data) {
                frame.add_numeric(column, values)?;
            }
            for (column, values) in categorical_columns.iter().zip(categorical_data) {
                frame.add_categorical(column, values)?;
            }
            frame
        };
        let intact_formula = parse_formula(
            "correct ~ group * mask * soa_s + block + (1 | participant) + (0 + mask | participant) + (0 + soa_s | participant) + (1 | item)",
        )?;
        // Theta order: item (320), mask block (144: L11,L21,L22), int (72), soa (72).
        let assignments: [(&str, [f64; 6]); 2] = [
            (
                "unmasked-first",
                [
                    1.506988612903417,
                    0.9350989330923332,
                    0.9488292385331001,
                    0.0,
                    4.372341038540467e-08,
                    0.5156626191820546,
                ],
            ),
            (
                "masked-first",
                [
                    1.506988612903417,
                    0.0,
                    0.9488292385331001,
                    0.9350989330923332,
                    4.372341038540467e-08,
                    0.5156626191820546,
                ],
            ),
        ];
        for (label, theta) in assignments {
            let mut walker = GeneralizedLinearMixedModel::new(
                intact_formula.clone(),
                &intact_data,
                Family::Bernoulli,
                None,
            )?;
            match walker.profiled_deviance_at_theta(&theta, 1) {
                Ok(objective) => println!(
                    "[intact {label}] profiled objective at lme4 theta: {objective:.4} (reference {:.4})",
                    -2.0 * intact_reference_loglik
                ),
                Err(error) => println!("[intact {label}] profiled error {error}"),
            }
            match walker.joint_deviance_at_theta_with_profiled_beta(&theta, 1) {
                Ok(objective) => println!(
                    "[intact {label}] JOINT objective at lme4 theta: {objective:.4} (reference {:.4})",
                    -2.0 * intact_reference_loglik
                ),
                Err(error) => println!("[intact {label}] joint error {error}"),
            }
        }
        let mut fit6 = GeneralizedLinearMixedModel::new(
            intact_formula.clone(),
            &intact_data,
            Family::Bernoulli,
            None,
        )?;
        fit6.fit_with_options(true, 1, false)?;
        println!(
            "[intact 6-theta profiled fit] logLik={:.4} gap={:.4} theta={:?} status={}",
            fit6.loglikelihood(),
            fit6.loglikelihood() - intact_reference_loglik,
            fit6.theta(),
            fit6.lmm().optsum().return_value
        );
        let intact_4theta = parse_formula(
            "correct ~ group * mask * soa_s + block + (1 + mask + soa_s || participant) + (1 | item)",
        )?;
        let mut fit4 =
            GeneralizedLinearMixedModel::new(intact_4theta, &intact_data, Family::Bernoulli, None)?;
        fit4.fit_with_options(true, 1, false)?;
        println!(
            "[intact 4-theta profiled fit] logLik={:.4} gap={:.4} theta={:?} status={}",
            fit4.loglikelihood(),
            fit4.loglikelihood() - intact_reference_loglik,
            fit4.theta(),
            fit4.lmm().optsum().return_value
        );
        return Ok(());
    }

    // Fast standalone check: does raising the per-evaluation PIRLS budget at
    // lme4's theta (in lme4's exact family) collapse the objective gap?
    if std::env::var("MIXEFF_PROBE_PIRLS_BUDGET").is_ok() {
        let lme4_formula = parse_formula(
            "correct ~ group * mask * soa_s * stimtype + block + (1 | participant) + (0 + mask | participant) + (0 + soa_s | participant) + (1 | item)",
        )?;
        let lme4_theta = [
            lme4_item_int,
            0.4583396899603748,
            0.3125031504858162,
            8.509615040182939e-05,
            lme4_part_int,
            lme4_part_soa,
        ];
        let lme4_theta_alt = [
            lme4_item_int,
            8.509615040182939e-05,
            0.3125031504858162,
            0.4583396899603748,
            lme4_part_int,
            lme4_part_soa,
        ];
        for (label, theta) in [("unmasked-first", lme4_theta), ("masked-first", lme4_theta_alt)] {
            let mut walker = GeneralizedLinearMixedModel::new(
                lme4_formula.clone(),
                &data,
                Family::Bernoulli,
                None,
            )?;
            match walker.joint_deviance_at_theta_with_profiled_beta(&theta, 1) {
                Ok(objective) => println!(
                    "[combined {label}] JOINT objective at lme4 theta: {objective:.4} (reference {:.4})",
                    -2.0 * reference_loglik
                ),
                Err(error) => println!("[combined {label}] joint error {error}"),
            }
        }
        return Ok(());
    }

    let mut profiled =
        GeneralizedLinearMixedModel::new(formula.clone(), &data, Family::Bernoulli, None)?;
    profiled.fit_with_options(true, 1, false)?;
    let profiled_gap = profiled.loglikelihood() - reference_loglik;
    println!(
        "profiled: logLik={:.4} gap={:.4} feval={} status={}",
        profiled.loglikelihood(),
        profiled_gap,
        profiled.lmm().optsum().feval,
        profiled.lmm().optsum().return_value
    );
    println!("profiled theta: {:?}", profiled.theta());
    let candidates: [(&str, Vec<f64>); 7] = [
        (
            "sanity: profiled fit theta",
            profiled.theta().to_vec(),
        ),
        (
            "item,part(int,mask,soa)",
            vec![lme4_item_int, lme4_part_int, lme4_part_mask, lme4_part_soa],
        ),
        (
            "item,part(int,soa,mask)",
            vec![lme4_item_int, lme4_part_int, lme4_part_soa, lme4_part_mask],
        ),
        (
            "item,part(mask,int,soa)",
            vec![lme4_item_int, lme4_part_mask, lme4_part_int, lme4_part_soa],
        ),
        (
            "item,part(mask,soa,int)",
            vec![lme4_item_int, lme4_part_mask, lme4_part_soa, lme4_part_int],
        ),
        (
            "item,part(soa,int,mask)",
            vec![lme4_item_int, lme4_part_soa, lme4_part_int, lme4_part_mask],
        ),
        (
            "item,part(soa,mask,int)",
            vec![lme4_item_int, lme4_part_soa, lme4_part_mask, lme4_part_int],
        ),
    ];
    for (label, theta) in candidates {
        let mut at_ref =
            GeneralizedLinearMixedModel::new(formula.clone(), &data, Family::Bernoulli, None)?;
        match at_ref.profiled_deviance_at_theta(&theta, 1) {
            Ok(objective) => println!(
                "profiled deviance at lme4 theta [{label}]: {objective:.4} (reference deviance {:.4})",
                -2.0 * reference_loglik
            ),
            Err(error) => println!("profiled deviance at lme4 theta [{label}]: error {error}"),
        }
    }

    // lme4's || expansion keeps the within-factor 2x2 correlation for `mask`
    // ((0 + mask | participant) full block); native || drops it. Express
    // lme4's exact family natively and check whether its optimum is reachable.
    if std::env::var("MIXEFF_PROBE_LME4_FAMILY").is_ok() {
        let lme4_formula = parse_formula(
            "correct ~ group * mask * soa_s * stimtype + block + (1 | participant) + (0 + mask | participant) + (0 + soa_s | participant) + (1 | item)",
        )?;
        // Expected theta order (reterms sorted by decreasing n_ranef, stable):
        // item.int (640), mask block (144: L11, L21, L22), part.int (72),
        // part.soa (72). Two candidate assignments of lme4's Cholesky names to
        // the native (masked, unmasked) column order.
        let block_candidates: [(&str, Vec<f64>); 2] = [
            (
                "L11=maskmasked(8.5e-5)",
                vec![
                    lme4_item_int,
                    8.509615040182939e-05,
                    0.3125031504858162,
                    0.4583396899603748,
                    lme4_part_int,
                    lme4_part_soa,
                ],
            ),
            (
                "L11=maskunmasked(0.4583)",
                vec![
                    lme4_item_int,
                    0.4583396899603748,
                    0.3125031504858162,
                    8.509615040182939e-05,
                    lme4_part_int,
                    lme4_part_soa,
                ],
            ),
        ];
        for (label, theta) in block_candidates {
            let mut at_ref = GeneralizedLinearMixedModel::new(
                lme4_formula.clone(),
                &data,
                Family::Bernoulli,
                None,
            )?;
            println!("lme4-family model theta dim: {}", at_ref.theta().len());
            match at_ref.profiled_deviance_at_theta(&theta, 1) {
                Ok(objective) => println!(
                    "lme4-family profiled deviance at lme4 theta [{label}]: {objective:.4} (reference {:.4})",
                    -2.0 * reference_loglik
                ),
                Err(error) => {
                    println!("lme4-family profiled deviance at lme4 theta [{label}]: error {error}")
                }
            }
        }
        let mut lme4_family_fit =
            GeneralizedLinearMixedModel::new(lme4_formula.clone(), &data, Family::Bernoulli, None)?;
        lme4_family_fit.fit_with_options(true, 1, false)?;
        println!(
            "lme4-family profiled fit: logLik={:.4} gap={:.4} feval={} status={} theta={:?}",
            lme4_family_fit.loglikelihood(),
            lme4_family_fit.loglikelihood() - reference_loglik,
            lme4_family_fit.lmm().optsum().feval,
            lme4_family_fit.lmm().optsum().return_value,
            lme4_family_fit.theta()
        );

        // Warm-started path from the fitted theta to lme4's theta: PIRLS
        // reuses the previous point's modes at each step, so a large drop at
        // the endpoint vs the cold evaluation above indicts cold-start PIRLS
        // convergence rather than an objective mismatch.
        let fitted_theta = lme4_family_fit.theta().to_vec();
        let lme4_theta = [
            lme4_item_int,
            0.4583396899603748,
            0.3125031504858162,
            8.509615040182939e-05,
            lme4_part_int,
            lme4_part_soa,
        ];
        let mut walker =
            GeneralizedLinearMixedModel::new(lme4_formula, &data, Family::Bernoulli, None)?;
        for step in 0..=10 {
            let weight = f64::from(step) / 10.0;
            let theta = fitted_theta
                .iter()
                .zip(lme4_theta.iter())
                .map(|(a, b)| a + weight * (b - a))
                .collect::<Vec<_>>();
            match walker.profiled_deviance_at_theta(&theta, 1) {
                Ok(objective) => println!("warm path w={weight:.1}: {objective:.4}"),
                Err(error) => println!("warm path w={weight:.1}: error {error}"),
            }
        }
        // Re-evaluate the endpoint twice more for PIRLS settling.
        for round in 0..2 {
            if let Ok(objective) = walker.profiled_deviance_at_theta(&lme4_theta, 1) {
                println!("warm endpoint repeat {round}: {objective:.4}");
            }
        }
        // Default evaluations cap PIRLS at GLMM_PIRLS_MAX_ITER=10 from reset
        // modes; check whether a larger budget collapses the gap to lme4.
        for budget in [10usize, 25, 50, 100, 200] {
            match walker.profiled_deviance_at_theta_with_pirls_budget(&lme4_theta, 1, budget) {
                Ok(objective) => {
                    println!("endpoint with PIRLS budget {budget}: {objective:.4}")
                }
                Err(error) => println!("endpoint with PIRLS budget {budget}: error {error}"),
            }
        }
        return Ok(());
    }

    if std::env::var("MIXEFF_PROBE_SKIP_JOINT").is_ok() {
        return Ok(());
    }

    let mut joint = GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None)?;
    if let Some(max_feval) = max_feval {
        joint.lmm_mut().optsum_mut().max_feval = max_feval;
    }
    joint.fit_with_options(false, 1, false)?;
    let optsum = joint.lmm().optsum();
    println!(
        "joint:    logLik={:.4} gap={:.4} feval={} max_feval={} status={} fmin={:.4} finitial={:.4}",
        joint.loglikelihood(),
        joint.loglikelihood() - reference_loglik,
        optsum.feval,
        optsum.max_feval,
        optsum.return_value,
        optsum.fmin,
        optsum.finitial
    );
    println!("joint theta: {:?}", joint.theta());

    let log = &optsum.fit_log;
    println!("fit_log entries: {}", log.len());
    let mut best = f64::INFINITY;
    for (index, entry) in log.iter().enumerate() {
        if entry.objective < best {
            best = entry.objective;
            println!("  eval {:>4}: objective={:.6} (new best)", index + 1, entry.objective);
        }
    }

    if let Some(cert) = joint.compiler_artifact().optimizer_certificate.as_ref() {
        println!("fit_status: {:?}", cert.status);
        for diagnostic in &cert.diagnostics {
            println!(
                "diag {:?} [{:?}]: {}",
                diagnostic.code, diagnostic.severity, diagnostic.message
            );
        }
    }
    for diagnostic in &joint.compiler_artifact().diagnostics {
        let code = format!("{:?}", diagnostic.code);
        if code.contains("Optimizer") {
            println!(
                "artifact diag {code}: {} payload={}",
                diagnostic.message,
                serde_json::json!(diagnostic.payload)
            );
        }
    }
    Ok(())
}
