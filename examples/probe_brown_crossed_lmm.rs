//! Reproduce the Brown (2021) crossed-vector REML optimizer fixture.
//!
//! By default this reads the sibling `mixeff` checkout's fixture. Override the
//! path with the first command-line argument.
//!
//! ```text
//! cargo run --release --no-default-features --example probe_brown_crossed_lmm
//! ```

use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::data::DataFrame;
use mixeff_rs::model::linear::{FitOptions, LinearMixedModel};
use mixeff_rs::model::traits::MixedModelFit;
use mixeff_rs::types::Optimizer;

fn load_fixture(path: &PathBuf) -> Result<DataFrame, Box<dyn Error>> {
    let mut reader = csv::Reader::from_path(path)?;
    let mut pid = Vec::new();
    let mut rt = Vec::new();
    let mut snr = Vec::new();
    let mut modality = Vec::new();
    let mut stim = Vec::new();

    for record in reader.records() {
        let record = record?;
        pid.push(record[0].to_string());
        rt.push(record[1].parse::<f64>()?);
        snr.push(match &record[2] {
            "Easy" => 0.0,
            "Hard" => 1.0,
            value => return Err(format!("unknown SNR value {value}").into()),
        });
        modality.push(match &record[3] {
            "Audio-only" => 0.0,
            "Audiovisual" => 1.0,
            value => return Err(format!("unknown modality value {value}").into()),
        });
        stim.push(record[4].to_string());
    }

    let mut data = DataFrame::new();
    data.add_numeric("RT", rt)?;
    data.add_numeric("SNR", snr)?;
    data.add_numeric("modality", modality)?;
    data.add_categorical("stim", stim)?;
    data.add_categorical("PID", pid)?;
    Ok(data)
}

fn fit(data: &DataFrame, optimizer: Option<Optimizer>) -> Result<LinearMixedModel, Box<dyn Error>> {
    let formula = parse_formula(
        "RT ~ 1 + modality + SNR + modality:SNR + \
         (0 + modality | stim) + (1 | stim) + (1 + modality + SNR | PID)",
    )?;
    let mut model = LinearMixedModel::new(formula, data, None)?;
    let options = match optimizer {
        Some(optimizer) => FitOptions::reml().with_optimizer(optimizer),
        None => FitOptions::reml(),
    };
    model.fit_with_options(options)?;
    Ok(model)
}

fn print_fit(label: &str, model: &LinearMixedModel, elapsed: std::time::Duration) {
    let certificate = model
        .optimizer_certificate()
        .expect("fitted model should have an optimizer certificate");
    println!(
        "{label}: elapsed={:.3}s objective={:.6}, theta={:?}, sigma={:.6}, fevals={}, stop={}, radius={:?}, status={:?}, free_gradient_norm={:?}, certification={:?}",
        elapsed.as_secs_f64(),
        model.opt_summary().fmin,
        model.theta(),
        model.sigma(),
        model.opt_summary().feval,
        model.opt_summary().return_value,
        model.opt_summary().final_trust_radius,
        certificate.status,
        certificate.free_gradient_norm,
        certificate.evidence.certification_quality,
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from("../../mixeff/tests/fixtures/brown_rt_dummy_data_interaction.csv")
        });
    let data = load_fixture(&path)?;

    let start = Instant::now();
    let automatic = fit(&data, None)?;
    print_fit("automatic native", &automatic, start.elapsed());

    let start = Instant::now();
    let cobyla = fit(&data, Some(Optimizer::Cobyla))?;
    print_fit("COBYLA oracle", &cobyla, start.elapsed());
    Ok(())
}
