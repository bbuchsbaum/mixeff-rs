use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use csv::{ReaderBuilder, StringRecord};
use mixedmodels::formula::parse_formula;
use mixedmodels::model::{DataFrame, LinearMixedModel, MixedModelFit};
use serde::Serialize;

#[derive(Clone, Copy)]
struct BenchCase {
    id: &'static str,
    fixture: &'static str,
    formula: &'static str,
    estimator: &'static str,
    load: fn() -> Result<DataFrame, Box<dyn Error>>,
}

#[derive(Serialize)]
struct BenchmarkFile {
    schema_name: &'static str,
    schema_version: &'static str,
    engine: &'static str,
    tool: &'static str,
    build_profile: &'static str,
    warmups: usize,
    repeats: usize,
    max_feval: i64,
    results: Vec<BenchResult>,
}

#[derive(Serialize)]
struct RandomTermSummary {
    group: String,
    levels: usize,
    vsize: usize,
    nranef: usize,
}

#[derive(Serialize)]
struct BenchResult {
    case_id: String,
    fixture: String,
    formula: String,
    estimator: String,
    n_obs: Option<usize>,
    q: Option<usize>,
    n_theta: Option<usize>,
    random_terms: Vec<RandomTermSummary>,
    fit_time_ms_min: Option<f64>,
    fit_time_ms_median: Option<f64>,
    fit_time_ms_repeats: usize,
    fevals: Option<i64>,
    return_value: Option<String>,
    objective: Option<f64>,
    sigma: Option<f64>,
    beta: Option<Vec<f64>>,
    coef_names: Option<Vec<String>>,
    status: String,
    error: Option<String>,
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
            id: "brown_rt_full",
            fixture: "brown_rt_dummy_data",
            formula: "RT ~ 1 + modality + (1 + modality | PID) + (1 + modality | stim)",
            estimator: "REML",
            load: load_brown_rt,
        },
        BenchCase {
            id: "iamciera_max_model",
            fixture: "iamciera_modeling_example",
            formula: "trans_abs_stom ~ il + (1 | tray) + (1 | row) + (1 | col)",
            estimator: "REML",
            load: load_iamciera_stomata,
        },
        BenchCase {
            id: "sdamr_speeddate_maximal_crossed",
            fixture: "sdamr_speeddate_lmm",
            formula: "other_like ~ other_attr_c + other_intel_c + attr_by_intel + \
                (1 + other_attr_c + other_intel_c + attr_by_intel | iid) + \
                (1 + other_attr_c + other_intel_c + attr_by_intel | pid)",
            estimator: "REML",
            load: load_sdamr_speeddate,
        },
        BenchCase {
            id: "sdamr_speeddate_uncorrelated_crossed",
            fixture: "sdamr_speeddate_lmm",
            formula: "other_like ~ other_attr_c + other_intel_c + attr_by_intel + \
                (1 + other_attr_c + other_intel_c || iid) + \
                (1 + other_attr_c + other_intel_c || pid)",
            estimator: "REML",
            load: load_sdamr_speeddate,
        },
    ]
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_i64(name: &str, default: i64) -> i64 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<i64>().ok())
        .unwrap_or(default)
}

fn selected_cases(all_cases: &[BenchCase]) -> Result<Vec<BenchCase>, Box<dyn Error>> {
    let selected = env::args().nth(1);
    match selected.as_deref() {
        None | Some("all") => Ok(all_cases.to_vec()),
        Some(name) => {
            let cases = all_cases
                .iter()
                .copied()
                .filter(|case| case.id == name)
                .collect::<Vec<_>>();
            if cases.is_empty() {
                let names = all_cases
                    .iter()
                    .map(|case| case.id)
                    .collect::<Vec<_>>()
                    .join(", ");
                Err(format!("unknown benchmark case `{name}`; choose one of: {names}").into())
            } else {
                Ok(cases)
            }
        }
    }
}

fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        Some((sorted[mid - 1] + sorted[mid]) / 2.0)
    } else {
        Some(sorted[mid])
    }
}

fn run_case(case: BenchCase, warmups: usize, repeats: usize, max_feval: i64) -> BenchResult {
    eprintln!("case: {}", case.id);

    let Ok(data) = (case.load)() else {
        return BenchResult {
            case_id: case.id.to_string(),
            fixture: case.fixture.to_string(),
            formula: case.formula.to_string(),
            estimator: case.estimator.to_string(),
            n_obs: None,
            q: None,
            n_theta: None,
            random_terms: Vec::new(),
            fit_time_ms_min: None,
            fit_time_ms_median: None,
            fit_time_ms_repeats: 0,
            fevals: None,
            return_value: None,
            objective: None,
            sigma: None,
            beta: None,
            coef_names: None,
            status: "error".to_string(),
            error: Some("failed to load fixture".to_string()),
        };
    };
    let Ok(formula) = parse_formula(case.formula) else {
        return BenchResult {
            case_id: case.id.to_string(),
            fixture: case.fixture.to_string(),
            formula: case.formula.to_string(),
            estimator: case.estimator.to_string(),
            n_obs: Some(data.nrow()),
            q: None,
            n_theta: None,
            random_terms: Vec::new(),
            fit_time_ms_min: None,
            fit_time_ms_median: None,
            fit_time_ms_repeats: 0,
            fevals: None,
            return_value: None,
            objective: None,
            sigma: None,
            beta: None,
            coef_names: None,
            status: "error".to_string(),
            error: Some("failed to parse formula".to_string()),
        };
    };
    let Ok(template) = LinearMixedModel::new(formula.clone(), &data, None) else {
        return BenchResult {
            case_id: case.id.to_string(),
            fixture: case.fixture.to_string(),
            formula: case.formula.to_string(),
            estimator: case.estimator.to_string(),
            n_obs: Some(data.nrow()),
            q: None,
            n_theta: None,
            random_terms: Vec::new(),
            fit_time_ms_min: None,
            fit_time_ms_median: None,
            fit_time_ms_repeats: 0,
            fevals: None,
            return_value: None,
            objective: None,
            sigma: None,
            beta: None,
            coef_names: None,
            status: "error".to_string(),
            error: Some("failed to construct model".to_string()),
        };
    };

    let n_obs = data.nrow();
    let q = template
        .reterms
        .iter()
        .map(|term| term.n_ranef())
        .sum::<usize>();
    let n_theta = template.theta().len();
    let random_terms = template
        .reterms
        .iter()
        .map(|term| RandomTermSummary {
            group: term.grouping_name.clone(),
            levels: term.n_levels(),
            vsize: term.vsize,
            nranef: term.n_ranef(),
        })
        .collect::<Vec<_>>();

    for _ in 0..warmups {
        match LinearMixedModel::new(formula.clone(), &data, None) {
            Ok(mut model) => {
                model.optsum.max_feval = max_feval;
                if let Err(error) = model.fit(true) {
                    eprintln!("  warmup failed: {error}");
                }
            }
            Err(error) => eprintln!("  warmup construction failed: {error}"),
        }
    }

    let mut times = Vec::with_capacity(repeats);
    let mut last_model: Option<LinearMixedModel> = None;
    let mut last_error = None;
    for run in 0..repeats {
        let start = Instant::now();
        match LinearMixedModel::new(formula.clone(), &data, None).and_then(|mut model| {
            model.optsum.max_feval = max_feval;
            model.fit(true)?;
            Ok(model)
        }) {
            Ok(model) => {
                let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
                eprintln!(
                    "  run {}: fit={elapsed_ms:.1} ms feval={} objective={:.6}",
                    run + 1,
                    model.optsum.feval,
                    model.objective_value()
                );
                times.push(elapsed_ms);
                last_model = Some(model);
            }
            Err(error) => {
                let message = error.to_string();
                eprintln!("  run {} failed: {message}", run + 1);
                last_error = Some(message);
            }
        }
    }

    if let Some(model) = last_model {
        BenchResult {
            case_id: case.id.to_string(),
            fixture: case.fixture.to_string(),
            formula: case.formula.to_string(),
            estimator: case.estimator.to_string(),
            n_obs: Some(n_obs),
            q: Some(q),
            n_theta: Some(n_theta),
            random_terms,
            fit_time_ms_min: times.iter().copied().reduce(f64::min),
            fit_time_ms_median: median(&times),
            fit_time_ms_repeats: times.len(),
            fevals: Some(model.optsum.feval),
            return_value: Some(model.optsum.return_value.clone()),
            objective: Some(model.objective_value()),
            sigma: Some(model.sigma()),
            beta: Some(MixedModelFit::coef(&model).iter().copied().collect()),
            coef_names: Some(model.coef_names()),
            status: "ok".to_string(),
            error: None,
        }
    } else {
        BenchResult {
            case_id: case.id.to_string(),
            fixture: case.fixture.to_string(),
            formula: case.formula.to_string(),
            estimator: case.estimator.to_string(),
            n_obs: Some(n_obs),
            q: Some(q),
            n_theta: Some(n_theta),
            random_terms,
            fit_time_ms_min: None,
            fit_time_ms_median: None,
            fit_time_ms_repeats: times.len(),
            fevals: None,
            return_value: None,
            objective: None,
            sigma: None,
            beta: None,
            coef_names: None,
            status: "error".to_string(),
            error: last_error,
        }
    }
}

fn default_out_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("comparison")
        .join("mixeff")
        .join("rust_results.json")
}

fn tool_label() -> &'static str {
    if cfg!(feature = "nlopt") {
        "cargo run --release --features nlopt --example bench_mixeff_parity"
    } else {
        "cargo run --release --example bench_mixeff_parity"
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let warmups = env_usize("MIXEFF_BENCH_WARMUPS", 0);
    let repeats = env_usize("MIXEFF_BENCH_REPEATS", 1);
    let max_feval = env_i64("MIXEFF_BENCH_MAXEVAL", 10_000);
    let cases = cases();
    let selected = selected_cases(&cases)?;
    let results = selected
        .into_iter()
        .map(|case| run_case(case, warmups, repeats, max_feval))
        .collect::<Vec<_>>();

    let output = BenchmarkFile {
        schema_name: "mixedmodels.mixeff_speed_parity",
        schema_version: "1.0.0",
        engine: "mixedmodels-rust",
        tool: tool_label(),
        build_profile: if cfg!(debug_assertions) {
            "debug-assertions"
        } else {
            "release"
        },
        warmups,
        repeats,
        max_feval,
        results,
    };

    let out_path = env::var_os("MIXEFF_BENCH_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(default_out_path);
    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&out_path, serde_json::to_string_pretty(&output)?)?;
    eprintln!("wrote {}", out_path.display());
    Ok(())
}
