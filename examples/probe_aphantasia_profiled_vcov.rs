//! Verify the profiled-GLMM covariance scale on the aphantasia primary model.
//!
//! ```text
//! cargo run --release --no-default-features --features unstable-internals \
//!   --example probe_aphantasia_profiled_vcov
//! ```

use std::error::Error;
use std::path::PathBuf;

use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::{DataFrame, Family, GeneralizedLinearMixedModel, MixedModelFit};

fn load_data() -> Result<DataFrame, Box<dyn Error>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/aphantasia/prepared/primary.csv");
    let numeric_columns = ["correct", "soa_s"];
    let categorical_columns = ["participant", "item", "group", "mask", "block"];
    let mut reader = csv::Reader::from_path(path)?;
    let headers = reader.headers()?.clone();
    let mut numeric = vec![Vec::new(); numeric_columns.len()];
    let mut categorical = vec![Vec::new(); categorical_columns.len()];
    for record in reader.records() {
        let record = record?;
        for (slot, column) in numeric_columns.iter().enumerate() {
            let index = headers.iter().position(|header| header == *column).unwrap();
            numeric[slot].push(record[index].parse::<f64>()?);
        }
        for (slot, column) in categorical_columns.iter().enumerate() {
            let index = headers.iter().position(|header| header == *column).unwrap();
            categorical[slot].push(record[index].to_string());
        }
    }
    let mut data = DataFrame::new();
    for (column, values) in numeric_columns.into_iter().zip(numeric) {
        data.add_numeric(column, values)?;
    }
    for (column, values) in categorical_columns.into_iter().zip(categorical) {
        data.add_categorical(column, values)?;
    }
    Ok(data)
}

fn main() -> Result<(), Box<dyn Error>> {
    let data = load_data()?;
    // Explicit expansion matches lme4's six-theta family for factor `||`.
    let formula = parse_formula(
        "correct ~ group * mask * soa_s + block + (1 | participant) + \
         (0 + mask | participant) + (0 + soa_s | participant) + (1 | item)",
    )?;
    let mut model = GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None)?;
    model.fit_with_options(true, 1, false)?;

    let inner_sigma = model.lmm().sigma();
    let raw = model.lmm().vcov();
    let corrected = model.vcov();
    println!("inner working-LMM sigma={inner_sigma:.9}");
    println!("expected SE multiplier={:.9}", 1.0 / inner_sigma);
    for (index, name) in model.coef_names().iter().enumerate() {
        let raw_se = raw[(index, index)].sqrt();
        let corrected_se = corrected[(index, index)].sqrt();
        println!(
            "{name}: raw_se={raw_se:.9} corrected_se={corrected_se:.9} multiplier={:.9}",
            corrected_se / raw_se
        );
    }
    Ok(())
}
