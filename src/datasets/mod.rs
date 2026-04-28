//! Reference datasets for testing, benchmarking, and parity work.
//!
//! Datasets live under `<repo>/datasets/<name>/` with two files each:
//! - `data.csv` — observations (factors as character labels).
//! - `meta.toml` — schema, recommended formula(s), and (where known)
//!   reference fit values from `lme4` or `MixedModels.jl`.
//!
//! The full registry is `datasets/REGISTRY.md`. Use [`load`] to pull a named
//! dataset into a [`DataFrame`](crate::model::DataFrame), with categorical
//! columns reconstructed in the canonical level order recorded in
//! `meta.toml` (so factor-coding lines up with the reference fits).

use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};

use crate::model::DataFrame;

/// Errors raised by the dataset loader. Distinct from [`crate::error::MixedModelError`]
/// because the loader is a dev/test convenience, not part of the model fit path.
#[derive(Debug, thiserror::Error)]
pub enum DatasetError {
    #[error("dataset `{0}` not found at {1}")]
    NotFound(String, PathBuf),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("csv error: {0}")]
    Csv(#[from] csv::Error),
    #[error("toml parse error: {0}")]
    Toml(#[from] toml::de::Error),
    #[error("schema mismatch in `{0}`: {1}")]
    Schema(String, String),
    #[error("expected numeric value in column `{column}` but got `{value}`")]
    BadNumeric { column: String, value: String },
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ColumnType {
    Numeric,
    Categorical,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ColumnSpec {
    pub name: String,
    #[serde(rename = "type")]
    pub kind: ColumnType,
    #[serde(default)]
    pub levels: Option<Vec<String>>,
    #[serde(default)]
    pub unit: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ExpectedFit {
    #[serde(default)]
    pub beta: Option<Vec<f64>>,
    #[serde(default)]
    pub sigma: Option<f64>,
    #[serde(default)]
    pub re_sigmas: Option<Vec<f64>>,
    #[serde(default)]
    pub re_corr: Option<f64>,
    #[serde(default)]
    pub theta: Option<Vec<f64>>,
    #[serde(default)]
    pub objective: Option<f64>,
    #[serde(default)]
    pub is_singular: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FitSpec {
    pub formula: String,
    pub family: String,
    pub link: String,
    pub estimator: String,
    #[serde(default)]
    pub weights: Option<String>,
    #[serde(default)]
    pub expected: Option<ExpectedFit>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct Tags {
    #[serde(default)]
    pub structure: Vec<String>,
    #[serde(default)]
    pub difficulty: Option<String>,
    #[serde(default)]
    pub family: Option<String>,
    #[serde(default)]
    pub notes: Option<String>,
}

/// Parsed `meta.toml` describing one dataset.
#[derive(Debug, Clone, Deserialize)]
pub struct Meta {
    pub name: String,
    pub source: String,
    #[serde(default)]
    pub license: Option<String>,
    pub n_rows: usize,
    pub description: String,
    pub columns: Vec<ColumnSpec>,
    #[serde(default)]
    pub fits: Vec<FitSpec>,
    #[serde(default)]
    pub tags: Tags,
}

/// Locate the `datasets/` directory. Resolution order:
/// 1. `MIXEDMODELS_DATASETS_DIR` env var, if set.
/// 2. `<CARGO_MANIFEST_DIR>/datasets/`.
fn datasets_root() -> PathBuf {
    if let Ok(p) = std::env::var("MIXEDMODELS_DATASETS_DIR") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("datasets")
}

/// Read just the metadata for a dataset (no CSV parse).
pub fn load_meta(name: &str) -> Result<Meta, DatasetError> {
    let dir = datasets_root().join(name);
    if !dir.is_dir() {
        return Err(DatasetError::NotFound(name.to_string(), dir));
    }
    let meta_path = dir.join("meta.toml");
    let text = fs::read_to_string(&meta_path)?;
    Ok(toml::from_str(&text)?)
}

/// Load a named dataset and return `(DataFrame, Meta)`.
///
/// The `DataFrame` columns are emitted in the order declared in `meta.toml`
/// (which is also the column order in `data.csv`). Categorical columns use
/// the canonical level order from `meta.toml`, not first-appearance order.
pub fn load(name: &str) -> Result<(DataFrame, Meta), DatasetError> {
    let meta = load_meta(name)?;
    let csv_path = datasets_root().join(name).join("data.csv");
    let df = read_csv_with_schema(&csv_path, &meta)?;
    if df.nrow() != meta.n_rows {
        return Err(DatasetError::Schema(
            name.to_string(),
            format!("meta declared {} rows, csv has {}", meta.n_rows, df.nrow()),
        ));
    }
    Ok((df, meta))
}

fn read_csv_with_schema(path: &Path, meta: &Meta) -> Result<DataFrame, DatasetError> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)?;
    let headers: Vec<String> = rdr.headers()?.iter().map(|s| s.to_string()).collect();

    // Map declared column → header index.
    let col_idx: Vec<usize> = meta
        .columns
        .iter()
        .map(|c| {
            headers.iter().position(|h| h == &c.name).ok_or_else(|| {
                DatasetError::Schema(
                    meta.name.clone(),
                    format!("column `{}` missing from CSV header", c.name),
                )
            })
        })
        .collect::<Result<_, _>>()?;

    // Per-column accumulators.
    let mut numeric: Vec<Option<Vec<f64>>> = meta
        .columns
        .iter()
        .map(|c| matches!(c.kind, ColumnType::Numeric).then(Vec::new))
        .collect();
    let mut strings: Vec<Option<Vec<String>>> = meta
        .columns
        .iter()
        .map(|c| matches!(c.kind, ColumnType::Categorical).then(Vec::new))
        .collect();

    for rec in rdr.records() {
        let rec = rec?;
        for (i, col) in meta.columns.iter().enumerate() {
            let raw = rec.get(col_idx[i]).unwrap_or("");
            match col.kind {
                ColumnType::Numeric => {
                    let v = raw.parse::<f64>().map_err(|_| DatasetError::BadNumeric {
                        column: col.name.clone(),
                        value: raw.to_string(),
                    })?;
                    numeric[i].as_mut().unwrap().push(v);
                }
                ColumnType::Categorical => {
                    strings[i].as_mut().unwrap().push(raw.to_string());
                }
            }
        }
    }

    let mut df = DataFrame::new();
    for (i, col) in meta.columns.iter().enumerate() {
        match col.kind {
            ColumnType::Numeric => {
                let v = numeric[i].take().unwrap();
                df.add_numeric(&col.name, v);
            }
            ColumnType::Categorical => {
                let v = strings[i].take().unwrap();
                if let Some(levels) = &col.levels {
                    df.add_categorical_with_levels(&col.name, v, levels.clone());
                } else {
                    df.add_categorical(&col.name, v);
                }
            }
        }
    }
    Ok(df)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_sleepstudy() {
        let (df, meta) = load("sleepstudy").expect("load sleepstudy");
        assert_eq!(meta.name, "sleepstudy");
        assert_eq!(df.nrow(), 180);
        assert_eq!(df.ncol(), 3);

        let reaction = df.numeric("Reaction").unwrap();
        assert_eq!(reaction.len(), 180);
        // First row in CSV: Reaction=249.56, Days=0, Subject=308.
        assert!((reaction[0] - 249.56).abs() < 1e-9);
        let days = df.numeric("Days").unwrap();
        assert_eq!(days[0], 0.0);
        assert_eq!(days[179], 9.0);

        let subj = df.categorical("Subject").unwrap();
        assert_eq!(subj.n_levels(), 18);
        // Canonical level order from meta.toml: 308 first, 372 last.
        assert_eq!(subj.levels[0], "308");
        assert_eq!(subj.levels[17], "372");

        // Recommended fits attached.
        assert!(meta.fits.iter().any(|f| f.estimator == "REML"));
        let reml = meta.fits.iter().find(|f| f.estimator == "REML").unwrap();
        let exp = reml.expected.as_ref().unwrap();
        let beta = exp.beta.as_ref().unwrap();
        assert_eq!(beta.len(), 2);
        assert!((beta[0] - 251.40510).abs() < 1e-3);
    }

    #[test]
    fn loads_dyestuff_singular_pair() {
        let (_, m1) = load("dyestuff").unwrap();
        let (_, m2) = load("dyestuff2").unwrap();
        assert_eq!(m1.n_rows, 30);
        assert_eq!(m2.n_rows, 30);
        // dyestuff2 advertises is_singular = true so callers know to expect θ=0.
        let ml2 = m2.fits.iter().find(|f| f.estimator == "ML").unwrap();
        assert_eq!(ml2.expected.as_ref().unwrap().is_singular, Some(true));
    }

    #[test]
    fn loads_cbpp_glmm() {
        let (df, meta) = load("cbpp").unwrap();
        assert_eq!(df.nrow(), 56);
        let fit = &meta.fits[0];
        assert_eq!(fit.family, "Binomial");
        assert_eq!(fit.link, "Logit");
        assert_eq!(fit.weights.as_deref(), Some("size"));
    }

    #[test]
    fn loads_pastes_nested() {
        let (df, meta) = load("pastes").unwrap();
        assert_eq!(df.nrow(), 60);
        let sample = df.categorical("sample").unwrap();
        assert_eq!(sample.n_levels(), 30);
        assert!(meta.tags.structure.iter().any(|s| s == "nested"));
    }

    #[test]
    fn loads_penicillin_crossed() {
        let (df, meta) = load("penicillin").unwrap();
        assert_eq!(df.nrow(), 144);
        assert_eq!(df.categorical("plate").unwrap().n_levels(), 24);
        assert_eq!(df.categorical("sample").unwrap().n_levels(), 6);
        assert!(meta.tags.structure.iter().any(|s| s == "crossed"));
    }

    #[test]
    fn loads_singular_maximal_case() {
        let (df, meta) = load("singular").unwrap();
        assert_eq!(df.nrow(), 150);
        assert_eq!(df.categorical("group").unwrap().n_levels(), 10);
        assert_eq!(df.numeric("A").unwrap()[0], 5.7);
        assert!(meta.tags.structure.iter().any(|s| s == "reduced_rank"));

        let maximal = &meta.fits[0];
        assert_eq!(maximal.formula, "y ~ 1 + A * B * C + (A * B * C | group)");
        assert_eq!(maximal.expected.as_ref().unwrap().is_singular, Some(true));
    }

    /// Sanity-check every Tier-1 + Tier-2 dataset that lives in the repo.
    /// Anything we ship must at least parse cleanly and match its declared row count.
    #[test]
    fn all_shipped_datasets_load() {
        // Keep this list in sync with datasets/REGISTRY.md.
        let names = [
            "sleepstudy",
            "dyestuff",
            "dyestuff2",
            "pastes",
            "penicillin",
            "cbpp",
            "cake",
            "verbagg",
            "grouseticks",
            "ergostool",
            "machines",
            "orthodont",
            "oats",
            "rail",
            "kb07",
            "singular",
            "tungara_single_caller",
        ];
        for name in names {
            let dir = datasets_root().join(name);
            if !dir.is_dir() {
                // Skip datasets not yet dumped (CI may run before the R script).
                eprintln!("skipping {name}: directory missing at {dir:?}");
                continue;
            }
            let (df, meta) = load(name).unwrap_or_else(|e| panic!("load {name}: {e}"));
            assert_eq!(df.nrow(), meta.n_rows, "row mismatch for {name}");
            for col in &meta.columns {
                assert!(
                    df.has_column(&col.name),
                    "{name}: missing column {}",
                    col.name
                );
            }
        }
    }
}
