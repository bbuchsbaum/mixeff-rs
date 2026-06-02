// Unstable-internals public surface: when the feature is off this module is
// `pub(crate)` and its types / re-exports have no in-crate consumer, reading
// as `unused_imports` / `dead_code`. Suppress only in that configuration;
// full linting stays when `unstable-internals` is enabled. See
// src/compiler/mod.rs for rationale.
#![cfg_attr(not(feature = "unstable-internals"), allow(unused_imports, dead_code))]
//! Reference datasets for testing, benchmarking, and parity work.
//!
//! Datasets live under `<repo>/datasets/<name>/` with two files each:
//! - `data.csv` — observations (factors as character labels).
//! - `meta.toml` — schema, recommended formula(s), and (where known)
//!   reference fit values from `lme4` or `MixedModels.jl`.
//!
//! The full registry is `datasets/REGISTRY.md`. Use [`load`] to pull a named
//! dataset into a [`crate::model::DataFrame`], with categorical
//! columns reconstructed in the canonical level order recorded in
//! `meta.toml` (so factor-coding lines up with the reference fits).

use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};

use crate::error::MixedModelError;
use crate::model::DataFrame;

/// Errors raised by the dataset loader. Distinct from [`crate::error::MixedModelError`]
/// because the loader is a dev/test convenience, not part of the model fit path.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
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
    #[error("dataframe construction error: {0}")]
    DataFrame(#[from] MixedModelError),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
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

#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
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

#[derive(Debug, Clone, Deserialize, PartialEq)]
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

/// Regeneration provenance for a dataset's pinned reference values.
///
/// Auto-managed by the dump scripts (R / Julia / synthesized). Lives in a
/// sibling `provenance.toml` so the hand-authored `meta.toml` stays stable.
/// All fields are optional so older datasets without a provenance file
/// still parse — they just deserialize to `Provenance::default()`.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Provenance {
    /// Display string, e.g. `"lme4 2.0.1"`.
    #[serde(default)]
    pub tool: Option<String>,
    #[serde(default)]
    pub tool_name: Option<String>,
    #[serde(default)]
    pub tool_version: Option<String>,
    /// Underlying language runtime (e.g. R 4.5.1, Julia 1.12.4).
    #[serde(default)]
    pub r_version: Option<String>,
    #[serde(default)]
    pub julia_version: Option<String>,
    /// ISO-8601 timestamp of last regeneration.
    #[serde(default)]
    pub date: Option<String>,
    /// `uname -srm` of the regenerating host.
    #[serde(default)]
    pub host: Option<String>,
    /// Path to the script that produced this file (relative to repo root).
    #[serde(default)]
    pub regenerator: Option<String>,
    /// Optimizer used for the reference fit (e.g. `"bobyqa"`, `"nlopt"`).
    #[serde(default)]
    pub optimizer: Option<String>,
    /// Free-form notes (e.g. seed for synthesized datasets).
    #[serde(default)]
    pub notes: Option<String>,
}

/// One pinned reference fit emitted by the auto-managed `expected.toml`.
///
/// Matches an entry in `meta.fits[]` by `(formula, estimator)`. When loaded,
/// merged into the corresponding `FitSpec.expected` if that field is `None`.
/// Hand-authored `[fits.expected]` in `meta.toml` always wins (no clobber).
#[derive(Debug, Clone, Deserialize)]
struct ExpectedEntry {
    formula: String,
    estimator: String,
    #[serde(flatten)]
    expected: ExpectedFit,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct ExpectedFile {
    #[serde(default, rename = "expected")]
    entries: Vec<ExpectedEntry>,
}

/// Parsed `meta.toml` describing one dataset, plus any auto-managed
/// sibling files (`provenance.toml`, `expected.toml`).
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
    /// Regeneration metadata loaded from sibling `provenance.toml`.
    /// `None` for datasets that have not yet been touched by the dump
    /// scripts; populated by the loader after Phase 1.
    #[serde(skip)]
    pub provenance: Option<Provenance>,
}

/// Locate the `datasets/` directory. Resolution order:
/// 1. `MIXEDMODELS_DATASETS_DIR` env var, if set.
/// 2. `<CARGO_MANIFEST_DIR>/datasets/`.
pub fn datasets_root() -> PathBuf {
    if let Ok(p) = std::env::var("MIXEDMODELS_DATASETS_DIR") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("datasets")
}

/// One concrete fit case — a dataset paired with one of its recommended
/// formulas. Yielded by [`iter_cases`]. Owned so callers don't need to
/// juggle lifetimes against the on-disk catalog scan.
#[derive(Debug, Clone)]
pub struct Case {
    /// Dataset directory name (also `meta.name`).
    pub name: String,
    pub meta: Meta,
    pub fit: FitSpec,
    /// Index of `fit` within `meta.fits` — useful when callers want to
    /// preserve canonical ordering.
    pub fit_index: usize,
}

/// Iterate every shipped dataset's [`Meta`].
///
/// Result order is dataset-name-sorted, so iteration is deterministic
/// across machines. Datasets with malformed `meta.toml` are skipped with
/// a stderr warning rather than panicking — the catalog should remain
/// usable even when one dataset is in a transient bad state.
pub fn iter() -> impl Iterator<Item = Meta> {
    let mut out: Vec<Meta> = Vec::new();
    if let Ok(entries) = fs::read_dir(datasets_root()) {
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() || !path.join("meta.toml").is_file() {
                continue;
            }
            let name = match path.file_name().and_then(|s| s.to_str()) {
                Some(s) => s.to_string(),
                None => continue,
            };
            match load_meta(&name) {
                Ok(m) => out.push(m),
                Err(e) => eprintln!("datasets::iter: skipping {name}: {e}"),
            }
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out.into_iter()
}

/// Iterate every (dataset, fit) pair flattened across the catalog.
///
/// Order: dataset-name-sorted, then by `fit_index` within each dataset.
/// This is the function `comparison/manifest.json` is derived from.
pub fn iter_cases() -> impl Iterator<Item = Case> {
    let mut out: Vec<Case> = Vec::new();
    for meta in iter() {
        let name = meta.name.clone();
        for (idx, fit) in meta.fits.iter().enumerate() {
            out.push(Case {
                name: name.clone(),
                meta: meta.clone(),
                fit: fit.clone(),
                fit_index: idx,
            });
        }
    }
    out.into_iter()
}

/// Iterate datasets matching a predicate. Convenience over
/// [`iter`] + filter for the common parameterized-test pattern.
pub fn iter_where<F>(predicate: F) -> impl Iterator<Item = Meta>
where
    F: FnMut(&Meta) -> bool,
{
    iter().filter(predicate)
}

/// Datasets carrying a given structure tag (e.g. `"crossed"`,
/// `"random_slope"`, `"nested"`).
pub fn iter_with_tag(tag: &str) -> impl Iterator<Item = Meta> {
    let tag = tag.to_string();
    iter_where(move |m| m.tags.structure.iter().any(|s| s == &tag))
}

/// Datasets at a given difficulty level (`"easy"`, `"moderate"`,
/// `"boundary"`, `"stress"`).
pub fn iter_difficulty(level: &str) -> impl Iterator<Item = Meta> {
    let level = level.to_string();
    iter_where(move |m| m.tags.difficulty.as_deref() == Some(level.as_str()))
}

/// Datasets whose top-level family tag matches (e.g. `"binomial"`,
/// `"poisson"`). Datasets without a `tags.family` declaration are
/// skipped — declare the tag on every GLMM fixture to keep this
/// selector reliable.
pub fn iter_family(family: &str) -> impl Iterator<Item = Meta> {
    let family = family.to_string();
    iter_where(move |m| m.tags.family.as_deref() == Some(family.as_str()))
}

/// Load a dataset by name and return one specific fit case.
///
/// Convenience over [`load`] + index, used when a test wants the fit's
/// formula/estimator/expected without rebuilding them from `Meta`.
pub fn case(name: &str, fit_index: usize) -> Result<(DataFrame, Meta, FitSpec), DatasetError> {
    let (df, meta) = load(name)?;
    let fit = meta
        .fits
        .get(fit_index)
        .ok_or_else(|| {
            DatasetError::Schema(
                name.to_string(),
                format!(
                    "fit_index {fit_index} out of range (have {} fits)",
                    meta.fits.len()
                ),
            )
        })?
        .clone();
    Ok((df, meta, fit))
}

/// Look up the pinned reference fit for a (dataset, formula, estimator)
/// triple. Returns `None` if the dataset has no matching fit or no
/// pinned `[fits.expected]` (inline or sibling).
pub fn expected_for(
    name: &str,
    formula: &str,
    estimator: &str,
) -> Result<Option<ExpectedFit>, DatasetError> {
    let meta = load_meta(name)?;
    Ok(meta
        .fits
        .into_iter()
        .find(|f| f.formula == formula && f.estimator == estimator)
        .and_then(|f| f.expected))
}

/// Read just the metadata for a dataset (no CSV parse).
///
/// Loads `meta.toml` and merges in any sibling `provenance.toml` and
/// `expected.toml` produced by the dump scripts. Hand-authored
/// `[fits.expected]` blocks in `meta.toml` are never overwritten.
pub fn load_meta(name: &str) -> Result<Meta, DatasetError> {
    let dir = datasets_root().join(name);
    if !dir.is_dir() {
        return Err(DatasetError::NotFound(name.to_string(), dir));
    }
    let meta_path = dir.join("meta.toml");
    let text = fs::read_to_string(&meta_path)?;
    let mut meta: Meta = toml::from_str(&text)?;

    // Sibling provenance.toml — auto-managed by the dump scripts.
    let prov_path = dir.join("provenance.toml");
    if prov_path.is_file() {
        let prov_text = fs::read_to_string(&prov_path)?;
        meta.provenance = Some(toml::from_str(&prov_text)?);
    }

    // Sibling expected.toml — auto-managed pinned reference fits.
    // Merged into meta.fits[i].expected when the inline field is None.
    let exp_path = dir.join("expected.toml");
    if exp_path.is_file() {
        let exp_text = fs::read_to_string(&exp_path)?;
        let exp_file: ExpectedFile = toml::from_str(&exp_text)?;
        for entry in exp_file.entries {
            if let Some(slot) = meta
                .fits
                .iter_mut()
                .find(|f| f.formula == entry.formula && f.estimator == entry.estimator)
            {
                if slot.expected.is_none() {
                    slot.expected = Some(entry.expected);
                }
            } else {
                return Err(DatasetError::Schema(
                    name.to_string(),
                    format!(
                        "expected.toml entry (formula=`{}`, estimator=`{}`) does not match any meta.toml [[fits]] row",
                        entry.formula, entry.estimator
                    ),
                ));
            }
        }
    }

    Ok(meta)
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
                df.add_numeric(&col.name, v)?;
            }
            ColumnType::Categorical => {
                let v = strings[i].take().unwrap();
                if let Some(levels) = &col.levels {
                    df.add_categorical_with_levels(&col.name, v, levels.clone())?;
                } else {
                    df.add_categorical(&col.name, v)?;
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

    #[test]
    fn loads_station_season_duration_diagnostic_case() {
        let (df, meta) = load("station_season_duration").unwrap();
        assert_eq!(meta.name, "station_season_duration");
        assert_eq!(df.nrow(), 54);
        assert_eq!(df.ncol(), 6);

        assert_eq!(
            df.categorical("duration").unwrap().levels,
            vec!["4d".to_string(), "7d".to_string()]
        );
        assert_eq!(
            df.categorical("season").unwrap().levels,
            vec!["mon".to_string(), "post".to_string(), "pre".to_string()]
        );
        assert_eq!(
            df.categorical("sites").unwrap().levels,
            vec!["s1".to_string(), "s2".to_string(), "s3".to_string()]
        );

        let effect = df.numeric("effect").unwrap();
        assert!((effect[0] - 7305.91).abs() < 1e-9);
        assert!((effect[53] - 6987.5).abs() < 1e-9);

        assert!(meta.fits.iter().any(|fit| fit.formula
            == "effect ~ 1 + duration + (1 + duration | sites) + (1 + duration | season)"));
        assert!(meta
            .tags
            .structure
            .iter()
            .any(|tag| tag == "weakly_supported"));
        assert_eq!(meta.tags.difficulty.as_deref(), Some("boundary"));
    }

    #[test]
    fn loads_nested_constant_response_diagnostic_case() {
        let (df, meta) = load("nested_constant_response").unwrap();
        assert_eq!(meta.name, "nested_constant_response");
        assert_eq!(df.nrow(), 24);
        assert_eq!(df.ncol(), 4);

        assert_eq!(
            df.categorical("studyarea").unwrap().levels,
            vec!["A".to_string(), "B".to_string(), "C".to_string()]
        );
        assert_eq!(
            df.categorical("teriid").unwrap().levels,
            vec![
                "t1".to_string(),
                "t2".to_string(),
                "t3".to_string(),
                "t4".to_string()
            ]
        );

        let spm = df.numeric("spm").unwrap();
        let y = df.numeric("logterrisize").unwrap();
        assert_eq!(spm[0], 4.0);
        assert_eq!(spm[1], 9.0);
        assert_eq!(y[0], y[1]);
        assert!(meta
            .fits
            .iter()
            .any(|fit| fit.formula == "logterrisize ~ 1 + spm + (1 | studyarea/teriid)"));
        assert!(meta
            .tags
            .structure
            .iter()
            .any(|tag| tag == "duplicated_response"));
    }

    #[test]
    fn iter_returns_every_shipped_dataset_in_sorted_order() {
        let names: Vec<String> = iter().map(|m| m.name).collect();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted, "iter() must yield datasets in sorted order");
        assert!(names.contains(&"sleepstudy".to_string()));
        assert!(names.contains(&"kb07".to_string()));
        assert!(names.len() >= 26);
    }

    #[test]
    fn iter_cases_flattens_all_recommended_fits() {
        let cases: Vec<Case> = iter_cases().collect();
        assert!(cases.len() >= iter().count());
        // Sleepstudy has two fits (REML + ML); kb07 has three.
        let sleep_count = cases.iter().filter(|c| c.name == "sleepstudy").count();
        assert!(sleep_count >= 2, "sleepstudy should yield ≥2 cases");
        let kb07_count = cases.iter().filter(|c| c.name == "kb07").count();
        assert_eq!(kb07_count, 3, "kb07 has three recommended formulas");
        // fit_index must align with meta.fits ordering.
        for case in cases.iter().filter(|c| c.name == "kb07") {
            assert_eq!(case.fit, case.meta.fits[case.fit_index]);
        }
    }

    #[test]
    fn iter_with_tag_and_difficulty_select_correctly() {
        let crossed: Vec<String> = iter_with_tag("crossed").map(|m| m.name).collect();
        assert!(crossed.contains(&"penicillin".to_string()));
        assert!(crossed.contains(&"kb07".to_string()));

        let stress: Vec<String> = iter_difficulty("stress").map(|m| m.name).collect();
        assert!(stress.contains(&"kb07".to_string()));

        let binomial: Vec<String> = iter_family("binomial").map(|m| m.name).collect();
        assert!(binomial.contains(&"cbpp".to_string()));
        assert!(binomial.contains(&"verbagg".to_string()));
    }

    #[test]
    fn case_returns_owned_fit_and_dataframe() {
        let (df, meta, fit) = case("sleepstudy", 0).unwrap();
        assert_eq!(meta.name, "sleepstudy");
        assert_eq!(df.nrow(), 180);
        assert!(fit.formula.contains("Reaction"));
        assert!(case("sleepstudy", 99).is_err());
    }

    #[test]
    fn expected_for_finds_pinned_value() {
        // sleepstudy carries a hand-pinned [fits.expected] block in meta.toml.
        let exp = expected_for(
            "sleepstudy",
            "Reaction ~ 1 + Days + (1 + Days | Subject)",
            "REML",
        )
        .unwrap()
        .expect("sleepstudy REML should be pinned");
        let beta = exp.beta.expect("β pinned");
        assert!((beta[0] - 251.40510).abs() < 1e-3);

        // Unknown formula → None, not Err.
        assert!(expected_for("sleepstudy", "Reaction ~ 1", "REML")
            .unwrap()
            .is_none());
    }

    #[test]
    fn loads_arabidopsis_overdispersed_poisson() {
        let (df, meta) = load("arabidopsis").unwrap();
        assert_eq!(meta.name, "arabidopsis");
        assert_eq!(df.nrow(), 625);
        assert_eq!(df.categorical("reg").unwrap().n_levels(), 3);
        assert_eq!(df.categorical("popu").unwrap().n_levels(), 9);
        assert_eq!(df.categorical("gen").unwrap().n_levels(), 24);
        assert_eq!(
            df.categorical("nutrient").unwrap().levels,
            vec!["1".to_string(), "8".to_string()]
        );
        // Check the overdispersion: var/mean ratio of total.fruits should be ≫ 1.
        let y = df.numeric("total.fruits").unwrap();
        let mean = y.iter().sum::<f64>() / y.len() as f64;
        let var = y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (y.len() - 1) as f64;
        assert!(
            var / mean > 30.0,
            "Arabidopsis fruit counts should be wildly overdispersed (var/mean = {:.2})",
            var / mean
        );
        // Poisson Laplace fit pinned via lme4.
        let fit = &meta.fits[0];
        assert_eq!(fit.family, "Poisson");
        assert!(fit.expected.is_some());
        assert!(meta.tags.structure.iter().any(|s| s == "overdispersed"));
    }

    #[test]
    fn loads_insteval_large_n_crossed() {
        let (df, meta) = load("insteval").unwrap();
        assert_eq!(meta.name, "insteval");
        assert_eq!(df.nrow(), 73421);
        assert_eq!(df.categorical("s").unwrap().n_levels(), 2972);
        assert_eq!(df.categorical("d").unwrap().n_levels(), 1128);
        assert_eq!(df.categorical("dept").unwrap().n_levels(), 14);
        assert_eq!(meta.tags.difficulty.as_deref(), Some("stress"));

        // Both fits pinned via lme4. Adding service + dept reduces the
        // objective slightly — sanity-check that the expected blocks are
        // distinct and ordered correctly.
        assert_eq!(meta.fits.len(), 2);
        let scalar = meta.fits[0].expected.as_ref().unwrap();
        let with_fe = meta.fits[1].expected.as_ref().unwrap();
        assert!(with_fe.objective.unwrap() < scalar.objective.unwrap());
        // Single intercept in the scalar fit; 15 betas with service + dept.
        assert_eq!(scalar.beta.as_ref().unwrap().len(), 1);
        assert_eq!(with_fe.beta.as_ref().unwrap().len(), 15);
    }

    #[test]
    fn loads_mrk17_exp1_optimizer_stress() {
        let (df, meta) = load("mrk17_exp1").unwrap();
        assert_eq!(meta.name, "mrk17_exp1");
        assert_eq!(df.nrow(), 16409);
        assert_eq!(df.categorical("subj").unwrap().n_levels(), 73);
        assert_eq!(df.categorical("item").unwrap().n_levels(), 240);
        assert_eq!(
            df.categorical("F").unwrap().levels,
            vec!["LF".to_string(), "HF".to_string()]
        );
        // Derived `speed = 1000 / rt` is the response in all three fits.
        assert!(df.numeric("speed").is_some());
        assert!(df.numeric("rt").is_some());

        // Three pinned fits matching kb07's structure: scalar baseline,
        // zerocorr maximal, full maximal. The zerocorr fit goes singular
        // on this design — useful for testing how the diagnostic surface
        // distinguishes "expected boundary on diagonal RE" from drift.
        assert_eq!(meta.fits.len(), 3);
        let scalar = meta.fits[0].expected.as_ref().unwrap();
        let zerocorr = meta.fits[1].expected.as_ref().unwrap();
        let maximal = meta.fits[2].expected.as_ref().unwrap();
        assert_eq!(zerocorr.is_singular, Some(true));
        // Objectives strictly improve scalar → zerocorr → maximal.
        assert!(scalar.objective.unwrap() > zerocorr.objective.unwrap());
        assert!(zerocorr.objective.unwrap() > maximal.objective.unwrap());
        assert_eq!(meta.tags.difficulty.as_deref(), Some("stress"));
    }

    #[test]
    fn loads_contraception_hierarchical_binomial() {
        let (df, meta) = load("contraception").unwrap();
        assert_eq!(meta.name, "contraception");
        assert_eq!(df.nrow(), 1934);
        assert_eq!(df.categorical("dist").unwrap().n_levels(), 60);
        assert_eq!(
            df.categorical("urban").unwrap().levels,
            vec!["Y".to_string(), "N".to_string()]
        );
        // `use` is numeric 0/1 (converted from Y/N during dump).
        let use_col = df.numeric("use").expect("use is numeric 0/1");
        assert!(use_col.iter().all(|v| *v == 0.0 || *v == 1.0));

        // Both fits pinned. Random-slope variant strictly improves
        // objective and surfaces a non-trivial intercept/urban correlation.
        assert_eq!(meta.fits.len(), 2);
        let scalar = meta.fits[0].expected.as_ref().expect("scalar fit pinned");
        let slope = meta.fits[1]
            .expected
            .as_ref()
            .expect("random-slope fit pinned");
        assert!(slope.objective.unwrap() < scalar.objective.unwrap());
        // Strong negative re_corr (intercept ↑ ⇒ urban-effect ↓ within district).
        assert!(slope.re_corr.unwrap() < -0.5);
    }

    #[test]
    fn loads_gopherdat2_offset_glmm() {
        let (df, meta) = load("gopherdat2").unwrap();
        assert_eq!(meta.name, "gopherdat2");
        assert_eq!(df.nrow(), 30);
        assert_eq!(df.categorical("Site").unwrap().n_levels(), 10);
        assert_eq!(df.categorical("year").unwrap().n_levels(), 3);
        assert!(df.numeric("Area").is_some());
        let fit = &meta.fits[0];
        assert!(fit.formula.contains("offset(log(Area))"));
        assert_eq!(fit.family, "Poisson");
        assert!(meta.tags.structure.iter().any(|s| s == "offset"));
        // lme4 pins the singular Site-level random intercept (σ → 0).
        let exp = fit.expected.as_ref().expect("Laplace fit pinned");
        assert_eq!(exp.is_singular, Some(true));
    }

    #[test]
    fn loads_culcitalogreg_block_design_glmm() {
        let (df, meta) = load("culcitalogreg").unwrap();
        assert_eq!(meta.name, "culcitalogreg");
        assert_eq!(df.nrow(), 80);
        assert_eq!(df.categorical("block").unwrap().n_levels(), 10);
        assert_eq!(
            df.categorical("ttt").unwrap().levels,
            vec![
                "none".to_string(),
                "crabs".to_string(),
                "shrimp".to_string(),
                "both".to_string(),
            ]
        );
        // Both Laplace and AGQ are pinned; their objectives differ
        // meaningfully (small-N AGQ correction).
        assert_eq!(meta.fits.len(), 2);
        let laplace_obj = meta.fits[0].expected.as_ref().unwrap().objective.unwrap();
        let agq_obj = meta.fits[1].expected.as_ref().unwrap().objective.unwrap();
        assert!(meta.fits[0].estimator == "Laplace");
        assert!(meta.fits[1].estimator == "AGQ");
        assert!(
            (laplace_obj - agq_obj).abs() > 0.1,
            "Laplace ({laplace_obj}) and AGQ ({agq_obj}) should differ on small-N binomial"
        );
    }

    #[test]
    fn loads_oxide_three_level_nested() {
        let (df, meta) = load("oxide").unwrap();
        assert_eq!(meta.name, "oxide");
        assert_eq!(df.nrow(), 72);
        assert_eq!(df.categorical("Lot").unwrap().n_levels(), 8);
        assert_eq!(df.categorical("Wafer").unwrap().n_levels(), 3);
        assert_eq!(df.categorical("Site").unwrap().n_levels(), 3);
        assert!(meta.tags.structure.iter().any(|s| s == "three_level"));
        // Sugar form `(1 | Lot/Wafer)` is pinned by Julia; the explicit
        // `(1 | Lot:Wafer)` form is recorded as a recommended formula
        // but not yet pinned (MixedModels.jl 5.x dispatch quirk).
        let sugar = meta
            .fits
            .iter()
            .find(|f| f.formula == "Thickness ~ 1 + (1 | Lot/Wafer)")
            .expect("sugar nested formula present");
        let expected = sugar.expected.as_ref().expect("sugar fit pinned");
        assert!((expected.objective.unwrap() - 454.02206930988217).abs() < 1e-6);
    }

    /// Every shipped dataset must have provenance recorded and at least
    /// one pinned reference fit (inline `[fits.expected]` or sibling
    /// `expected.toml`). This is the Phase-1 hygiene invariant; Phase 5
    /// will add stricter re-fit tolerance checks on top.
    #[test]
    fn every_dataset_has_provenance_and_pinned_fit() {
        let root = datasets_root();
        let entries = std::fs::read_dir(&root).expect("read datasets/");
        let mut checked = 0usize;
        for entry in entries {
            let entry = entry.unwrap();
            if !entry.file_type().unwrap().is_dir() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            // Skip non-dataset directories (e.g. README, future synthetic/ tier).
            if !entry.path().join("meta.toml").is_file() {
                continue;
            }
            let meta = load_meta(&name).unwrap_or_else(|e| panic!("load_meta {name}: {e}"));
            assert!(
                meta.provenance.is_some(),
                "{name}: missing provenance.toml — re-run scripts/dump_datasets.R \
                 (or scripts/dump_julia_datasets.jl / dump_synthesized_datasets.R)"
            );
            let n_pinned = meta.fits.iter().filter(|f| f.expected.is_some()).count();
            assert!(
                n_pinned > 0,
                "{name}: no pinned reference fits in meta.toml or expected.toml"
            );
            checked += 1;
        }
        assert!(
            checked >= 26,
            "expected at least 26 shipped datasets, saw {checked}"
        );
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
            "oxide",
            "singular",
            "tungara_single_caller",
            "station_season_duration",
            "nested_constant_response",
            "gopherdat2",
            "culcitalogreg",
            "rare_event_bernoulli",
            "contraception",
            "mrk17_exp1",
            "insteval",
            "arabidopsis",
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
