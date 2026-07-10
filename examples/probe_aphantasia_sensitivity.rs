//! Probe joint-Laplace convergence labels on the 18,240-row sensitivity GLMM.
//!
//! ```text
//! cargo run --release --no-default-features --features unstable-internals \
//!   --example probe_aphantasia_sensitivity -- 25 100 500
//! ```

use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::traits::MixedModelFit;
use mixeff_rs::model::{DataFrame, Family, GeneralizedLinearMixedModel};

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/aphantasia/prepared/sensitivity.csv")
}

fn load_data() -> Result<DataFrame, Box<dyn Error>> {
    let numeric_columns = ["correct", "soa_s"];
    let categorical_columns = ["participant", "item", "group", "mask", "block"];
    let mut reader = csv::Reader::from_path(fixture_path())?;
    let headers = reader.headers()?.clone();
    let numeric_indices = numeric_columns
        .iter()
        .map(|column| headers.iter().position(|header| header == *column).unwrap())
        .collect::<Vec<_>>();
    let categorical_indices = categorical_columns
        .iter()
        .map(|column| headers.iter().position(|header| header == *column).unwrap())
        .collect::<Vec<_>>();
    let mut numeric_data = vec![Vec::new(); numeric_columns.len()];
    let mut categorical_data = vec![Vec::new(); categorical_columns.len()];
    for record in reader.records() {
        let record = record?;
        for (slot, &index) in numeric_indices.iter().enumerate() {
            numeric_data[slot].push(record[index].parse::<f64>()?);
        }
        for (slot, &index) in categorical_indices.iter().enumerate() {
            categorical_data[slot].push(record[index].to_string());
        }
    }
    let mut data = DataFrame::new();
    for (column, values) in numeric_columns.into_iter().zip(numeric_data) {
        data.add_numeric(column, values)?;
    }
    for (column, values) in categorical_columns.into_iter().zip(categorical_data) {
        data.add_categorical(column, values)?;
    }
    Ok(data)
}

fn main() -> Result<(), Box<dyn Error>> {
    let budgets = std::env::args()
        .skip(1)
        .map(|raw| raw.parse::<i64>())
        .collect::<Result<Vec<_>, _>>()?;
    let budgets = if budgets.is_empty() {
        vec![-1]
    } else {
        budgets
    };
    let data = load_data()?;
    let formula = parse_formula(
        "correct ~ group * mask * soa_s + block + \
         (1 + mask + soa_s || participant) + (1 | item)",
    )?;

    for budget in budgets {
        let mut model =
            GeneralizedLinearMixedModel::new(formula.clone(), &data, Family::Bernoulli, None)?;
        if budget > 0 {
            model.lmm_mut().optsum_mut().max_feval = budget;
        }
        let start = Instant::now();
        model.fit_with_options(false, 1, false)?;
        let certificate = model
            .compiler_artifact()
            .optimizer_certificate
            .as_ref()
            .expect("fit should attach an optimizer certificate");
        println!(
            "budget={} elapsed={:.3}s objective={:.6} logLik={:.6} fevals={} max_feval={} stop={} typed_stop={:?} status={:?} free_gradient={:?} quality={:?}",
            budget,
            start.elapsed().as_secs_f64(),
            model.objective(),
            model.loglikelihood(),
            model.opt_summary().feval,
            model.opt_summary().max_feval,
            model.opt_summary().return_value,
            model.opt_summary().convergence_status(),
            certificate.status,
            certificate.free_gradient_norm,
            certificate.evidence.certification_quality,
        );
        for diagnostic in &certificate.diagnostics {
            println!(
                "  diagnostic={:?} message={} payload={}",
                diagnostic.code,
                diagnostic.message,
                serde_json::to_string(&diagnostic.payload)?
            );
        }
    }
    Ok(())
}
