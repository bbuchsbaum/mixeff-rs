use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use csv::{ReaderBuilder, StringRecord};
use mixedmodels::formula::parse_formula;
use mixedmodels::model::{DataFrame, LinearMixedModel, MixedModelFit};

struct BenchCase {
    name: &'static str,
    formula: &'static str,
    load: fn() -> Result<DataFrame, Box<dyn Error>>,
}

fn fixture_path(env_name: &str, default: &str) -> PathBuf {
    env::var_os(env_name)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(default))
}

fn header_index(headers: &StringRecord, name: &str) -> Result<usize, Box<dyn Error>> {
    headers
        .iter()
        .position(|header| header == name)
        .ok_or_else(|| format!("missing `{name}` column").into())
}

fn field<'a>(
    record: &'a StringRecord,
    headers: &StringRecord,
    name: &str,
) -> Result<&'a str, Box<dyn Error>> {
    let idx = header_index(headers, name)?;
    record
        .get(idx)
        .ok_or_else(|| format!("missing `{name}` value").into())
}

fn parse_f64_value(
    record: &StringRecord,
    headers: &StringRecord,
    name: &str,
) -> Result<f64, Box<dyn Error>> {
    Ok(field(record, headers, name)?.parse::<f64>()?)
}

fn parse_optional_f64(record: &StringRecord, headers: &StringRecord, name: &str) -> Option<f64> {
    let value = field(record, headers, name).ok()?;
    if value.is_empty() || value == "NA" {
        None
    } else {
        value.parse::<f64>().ok()
    }
}

fn load_brown_rt() -> Result<DataFrame, Box<dyn Error>> {
    let path = fixture_path(
        "BROWN_RT_CSV",
        "/Users/bbuchsbaum/code/mixeff/tests/fixtures/brown_rt_dummy_data.csv",
    );
    let mut rdr = csv::Reader::from_path(&path)?;
    let headers = rdr.headers()?.clone();
    let mut pid = Vec::new();
    let mut rt = Vec::new();
    let mut modality = Vec::new();
    let mut stim = Vec::new();

    for record in rdr.records() {
        let record = record?;
        pid.push(field(&record, &headers, "PID")?.to_string());
        rt.push(parse_f64_value(&record, &headers, "RT")?);
        modality.push(match field(&record, &headers, "modality")? {
            "Audio-only" => 0.0,
            _ => 1.0,
        });
        stim.push(field(&record, &headers, "stim")?.to_string());
    }

    let mut data = DataFrame::new();
    data.add_categorical("PID", pid)?;
    data.add_numeric("RT", rt)?;
    data.add_numeric("modality", modality)?;
    data.add_categorical("stim", stim)?;
    Ok(data)
}

fn load_iamciera_stomata() -> Result<DataFrame, Box<dyn Error>> {
    let path = fixture_path(
        "IAMCIERA_STOMATA_TSV",
        "/Users/bbuchsbaum/code/mixeff/tests/fixtures/iamciera_modeling_example.txt",
    );
    let mut rdr = ReaderBuilder::new().delimiter(b'\t').from_path(&path)?;
    let headers = rdr.headers()?.clone();
    let mut trans_abs_stom = Vec::new();
    let mut il = Vec::new();
    let mut tray = Vec::new();
    let mut row = Vec::new();
    let mut col = Vec::new();

    for record in rdr.records() {
        let record = record?;
        trans_abs_stom.push(parse_f64_value(&record, &headers, "abs_stom")?.sqrt());
        il.push(field(&record, &headers, "il")?.to_string());
        tray.push(field(&record, &headers, "tray")?.to_string());
        row.push(field(&record, &headers, "row")?.to_string());
        col.push(field(&record, &headers, "col")?.to_string());
    }

    let mut data = DataFrame::new();
    data.add_numeric("trans_abs_stom", trans_abs_stom)?;
    data.add_categorical("il", il)?;
    data.add_categorical("tray", tray)?;
    data.add_categorical("row", row)?;
    data.add_categorical("col", col)?;
    Ok(data)
}

fn load_sdamr_speeddate() -> Result<DataFrame, Box<dyn Error>> {
    let path = fixture_path(
        "SDAMR_SPEEDDATE_CSV",
        "/Users/bbuchsbaum/code/mixeff/tests/fixtures/sdamr_speeddate_lmm.csv",
    );
    let mut rdr = csv::Reader::from_path(&path)?;
    let headers = rdr.headers()?.clone();
    let mut other_like = Vec::new();
    let mut other_attr_c = Vec::new();
    let mut other_intel_c = Vec::new();
    let mut attr_by_intel = Vec::new();
    let mut iid = Vec::new();
    let mut pid = Vec::new();

    for record in rdr.records() {
        let record = record?;
        let Some(like) = parse_optional_f64(&record, &headers, "other_like") else {
            continue;
        };
        let Some(attr) = parse_optional_f64(&record, &headers, "other_attr_c") else {
            continue;
        };
        let Some(intel) = parse_optional_f64(&record, &headers, "other_intel_c") else {
            continue;
        };
        other_like.push(like);
        other_attr_c.push(attr);
        other_intel_c.push(intel);
        attr_by_intel.push(attr * intel);
        iid.push(field(&record, &headers, "iid")?.to_string());
        pid.push(field(&record, &headers, "pid")?.to_string());
    }

    let mut data = DataFrame::new();
    data.add_numeric("other_like", other_like)?;
    data.add_numeric("other_attr_c", other_attr_c)?;
    data.add_numeric("other_intel_c", other_intel_c)?;
    data.add_numeric("attr_by_intel", attr_by_intel)?;
    data.add_categorical("iid", iid)?;
    data.add_categorical("pid", pid)?;
    Ok(data)
}

fn cases() -> Vec<BenchCase> {
    vec![
        BenchCase {
            name: "brown_rt_full",
            formula: "RT ~ 1 + modality + (1 + modality | PID) + (1 + modality | stim)",
            load: load_brown_rt,
        },
        BenchCase {
            name: "iamciera_max_model",
            formula: "trans_abs_stom ~ il + (1 | tray) + (1 | row) + (1 | col)",
            load: load_iamciera_stomata,
        },
        BenchCase {
            name: "sdamr_speeddate_maximal_crossed",
            formula: "other_like ~ other_attr_c + other_intel_c + attr_by_intel + \
                (1 + other_attr_c + other_intel_c + attr_by_intel | iid) + \
                (1 + other_attr_c + other_intel_c + attr_by_intel | pid)",
            load: load_sdamr_speeddate,
        },
        BenchCase {
            name: "sdamr_speeddate_uncorrelated_crossed",
            formula: "other_like ~ other_attr_c + other_intel_c + attr_by_intel + \
                (1 + other_attr_c + other_intel_c || iid) + \
                (1 + other_attr_c + other_intel_c || pid)",
            load: load_sdamr_speeddate,
        },
    ]
}

fn max_feval() -> i64 {
    env::var("MIXEFF_BENCH_MAXEVAL")
        .or_else(|_| env::var("BROWN_RT_MAXEVAL"))
        .ok()
        .and_then(|value| value.parse::<i64>().ok())
        .unwrap_or(2_000)
}

fn profile_evals() -> usize {
    env::var("MIXEFF_PROFILE_EVALS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0)
}

fn run_case(case: &BenchCase, maxeval: i64) -> Result<(), Box<dyn Error>> {
    eprintln!("case: {}", case.name);

    let start = Instant::now();
    let data = (case.load)()?;
    eprintln!("load: {:?}", start.elapsed());
    eprintln!("rows: {}", data.nrow());

    let formula = parse_formula(case.formula)?;
    let start = Instant::now();
    let mut model = LinearMixedModel::new(formula, &data, None)?;
    eprintln!("construct: {:?}", start.elapsed());
    eprintln!(
        "theta={}, reterms={}, q={}",
        model.theta().len(),
        model.reterms.len(),
        model
            .reterms
            .iter()
            .map(|term| term.n_ranef())
            .sum::<usize>()
    );
    for (idx, term) in model.reterms.iter().enumerate() {
        eprintln!(
            "term[{idx}] group={} levels={} vsize={} nranef={}",
            term.grouping_name,
            term.n_levels(),
            term.vsize,
            term.n_ranef()
        );
    }

    let theta0 = model.theta();
    let start = Instant::now();
    let objective0 = model.objective_at(theta0.as_slice())?;
    eprintln!(
        "objective_at(theta0): {:?} value={objective0}",
        start.elapsed()
    );

    let profile_n = profile_evals();
    if profile_n > 0 {
        let mut set_theta_total = 0.0;
        let mut update_l_total = 0.0;
        let mut objective_value_total = 0.0;
        let mut objective_at_total = 0.0;
        let mut last_obj = objective0;
        for _ in 0..profile_n {
            let start = Instant::now();
            model.set_theta(&theta0)?;
            set_theta_total += start.elapsed().as_secs_f64() * 1000.0;

            let start = Instant::now();
            model.update_l()?;
            update_l_total += start.elapsed().as_secs_f64() * 1000.0;

            let start = Instant::now();
            let _ = model.objective_value();
            objective_value_total += start.elapsed().as_secs_f64() * 1000.0;

            let start = Instant::now();
            last_obj = model.objective_at(theta0.as_slice())?;
            objective_at_total += start.elapsed().as_secs_f64() * 1000.0;
        }
        let denom = profile_n as f64;
        eprintln!(
            "profile_evals={profile_n} mean_ms: set_theta={:.3} update_l={:.3} objective_value={:.3} objective_at={:.3} last_obj={last_obj}",
            set_theta_total / denom,
            update_l_total / denom,
            objective_value_total / denom,
            objective_at_total / denom,
        );
    }

    model.optsum.max_feval = maxeval;
    let start = Instant::now();
    model.fit(true)?;
    eprintln!("fit: {:?}", start.elapsed());
    eprintln!(
        "feval={} return={} objective={} sigma={}",
        model.optsum.feval,
        model.optsum.return_value,
        model.objective_value(),
        model.sigma()
    );
    eprintln!("beta={:?}", MixedModelFit::coef(&model));
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let selected = env::args().nth(1);
    let maxeval = max_feval();
    let cases = cases();
    let selected_cases: Vec<&BenchCase> = match selected.as_deref() {
        None | Some("all") => cases.iter().collect(),
        Some(name) => cases.iter().filter(|case| case.name == name).collect(),
    };

    if selected_cases.is_empty() {
        let names = cases
            .iter()
            .map(|case| case.name)
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!("unknown benchmark case; choose one of: {names}").into());
    }

    for (idx, case) in selected_cases.iter().enumerate() {
        if idx > 0 {
            eprintln!();
        }
        run_case(case, maxeval)?;
    }
    Ok(())
}
