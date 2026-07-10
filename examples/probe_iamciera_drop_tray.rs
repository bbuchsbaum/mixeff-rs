//! Reproduce the iamciera `drop_tray` ML TrustBQ cold-start failure.
//!
//! By default this reads the sibling `mixeff` checkout's fixture. Override the
//! path with the first command-line argument.
//!
//! ```text
//! cargo run --release --no-default-features --example probe_iamciera_drop_tray
//! ```

use std::error::Error;
use std::path::PathBuf;

use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::data::DataFrame;
use mixeff_rs::model::linear::{FitOptions, LinearMixedModel, OptimizerControl};
use mixeff_rs::model::traits::MixedModelFit;
use mixeff_rs::types::Optimizer;

fn load_fixture(path: &PathBuf) -> Result<DataFrame, Box<dyn Error>> {
    let mut reader = csv::ReaderBuilder::new().delimiter(b'\t').from_path(path)?;
    let mut response = Vec::new();
    let mut il = Vec::new();
    let mut row = Vec::new();
    let mut col = Vec::new();

    let row_limit = std::env::var("IAMCIERA_ROWS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok());
    for (index, record) in reader.records().enumerate() {
        if row_limit.is_some_and(|limit| index >= limit) {
            break;
        }
        let record = record?;
        let abs_stom: f64 = record[1].parse()?;
        response.push(abs_stom.sqrt());
        il.push(record[3].to_string());
        row.push(record[4].to_string());
        col.push(record[6].to_string());
    }

    let mut data = DataFrame::new();
    data.add_numeric("trans_abs_stom", response)?;
    data.add_categorical("il", il)?;
    data.add_categorical("row", row)?;
    data.add_categorical("col", col)?;
    Ok(data)
}

fn fit(
    data: &DataFrame,
    reml: bool,
    start_theta: Option<Vec<f64>>,
) -> Result<LinearMixedModel, Box<dyn Error>> {
    let formula = parse_formula("trans_abs_stom ~ il + (1 | row) + (1 | col)")?;
    let mut model = LinearMixedModel::new(formula, data, None)?;
    let mut control = OptimizerControl::auto().with_optimizer(Optimizer::TrustBq);
    if let Some(theta) = start_theta {
        control = control.with_start_theta(theta);
    }
    let options = if reml {
        FitOptions::reml()
    } else {
        FitOptions::ml()
    }
    .with_optimizer_control(control);
    model.fit_with_options(options)?;
    Ok(model)
}

fn print_fit(label: &str, model: &LinearMixedModel) {
    let certificate = model
        .optimizer_certificate()
        .expect("fitted model should have an optimizer certificate");
    println!(
        "{label}: objective={:.9}, logLik={:.9}, theta={:?}, initial_step={:?}, fevals={}, stop={}, radius={:?}, status={:?}, free_gradient_norm={:?}",
        model.opt_summary().fmin,
        -0.5 * model.opt_summary().fmin,
        model.theta(),
        model.opt_summary().initial_step,
        model.opt_summary().feval,
        model.opt_summary().return_value,
        model.opt_summary().final_trust_radius,
        certificate.status,
        certificate.free_gradient_norm,
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from("../../mixeff/tests/fixtures/iamciera_modeling_example.txt")
        });
    let data = load_fixture(&path)?;

    let cold_ml = fit(&data, false, None)?;
    print_fit("cold ML", &cold_ml);

    let reml = fit(&data, true, None)?;
    print_fit("cold REML", &reml);

    let warm_ml = fit(&data, false, Some(reml.theta()))?;
    print_fit("REML-start ML", &warm_ml);
    Ok(())
}
