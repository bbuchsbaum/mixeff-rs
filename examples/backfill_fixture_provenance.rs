//! One-shot backfill of `<stem>.provenance.json` siblings for every existing
//! golden under `tests/fixtures/{compiler_contract,parity}/`.
//!
//! Phase 4 of fixture/dataset unification (mote bd-01KQMZV0GSNTZBMB4385NQFR4D).
//!
//! Run once after cherry-picking Phase 4:
//!     cargo run --example backfill_fixture_provenance
//!
//! Idempotent: existing provenance siblings are never overwritten. To
//! refresh a single provenance file, delete it and re-run, or invoke its
//! original regenerator (`MIXEDMODELS_UPDATE_FIXTURES=1 cargo test ...`).
//!
//! The provenance schema is intentionally minimal:
//!     {
//!       "schema_version": "1.0",
//!       "generated_at": "<ISO-8601 UTC>",
//!       "regenerator": "<command that should refresh this file>",
//!       "notes": "..."
//!     }
//!
//! Fields like `crate_commit`, `source_case`, and `reference_engine` are
//! populated by future regenerator scripts (R / Julia) when they emit
//! provenance alongside their JSON output. Backfilled files leave those
//! null because the original generation context has been lost.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn iso8601_utc_now() -> String {
    // Avoid pulling in chrono just for this. Compute YYYY-MM-DDTHH:MM:SSZ
    // from epoch seconds via the well-known Howard Hinnant date algorithm.
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let (year, month, day) = civil_from_days(secs.div_euclid(86400));
    let s_of_day = secs.rem_euclid(86400);
    let h = s_of_day / 3600;
    let m = (s_of_day % 3600) / 60;
    let s = s_of_day % 60;
    format!("{year:04}-{month:02}-{day:02}T{h:02}:{m:02}:{s:02}Z")
}

// Howard Hinnant — `days_from_civil` inverse, days since 1970-01-01.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 {
        (mp + 3) as u32
    } else {
        (mp - 9) as u32
    };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

fn current_commit() -> Option<String> {
    let out = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(repo_root())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Best-guess regenerator command for a golden file, based on its directory.
fn guess_regenerator(path: &Path) -> &'static str {
    let parent = path
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|s| s.to_str())
        .unwrap_or("");
    match parent {
        "compiler_contract" => {
            "MIXEDMODELS_UPDATE_FIXTURES=1 cargo test --test compiler_contract_snapshots"
        }
        "parity" => {
            "scripts/regenerate_julia_parity_fixtures.jl  (or scripts/parity_pathologies.{R,jl})"
        }
        _ => "<unknown — see tests/fixtures/README.md>",
    }
}

fn write_provenance_if_missing(
    json_path: &Path,
    crate_commit: Option<&str>,
    timestamp: &str,
) -> std::io::Result<bool> {
    let stem = json_path
        .file_stem()
        .and_then(|s| s.to_str())
        .expect("UTF-8 stem");
    let prov_path = json_path.with_file_name(format!("{stem}.provenance.json"));
    if prov_path.exists() {
        return Ok(false);
    }
    let regenerator = guess_regenerator(json_path);
    let commit_field = match crate_commit {
        Some(c) => format!("\"{c}\""),
        None => "null".to_string(),
    };
    let body = format!(
        "{{\n  \"schema_version\": \"1.0\",\n  \"generated_at\": \"{timestamp}\",\n  \"crate_commit\": {commit_field},\n  \"regenerator\": \"{regenerator}\",\n  \"source_case\": null,\n  \"reference_engine\": null,\n  \"notes\": \"Backfilled by examples/backfill_fixture_provenance — original generation context lost. Refresh by running the regenerator command above.\"\n}}\n"
    );
    fs::write(&prov_path, body)?;
    Ok(true)
}

fn walk_directory(dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    if !dir.is_dir() {
        return Ok(out);
    }
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        if !name.ends_with(".json") || name.ends_with(".provenance.json") {
            continue;
        }
        out.push(path);
    }
    out.sort();
    Ok(out)
}

fn main() -> std::io::Result<()> {
    let root = repo_root();
    let timestamp = iso8601_utc_now();
    let crate_commit = current_commit();

    let dirs = [
        root.join("tests/fixtures/compiler_contract"),
        root.join("tests/fixtures/parity"),
    ];

    let mut written = 0usize;
    let mut skipped = 0usize;
    for dir in &dirs {
        for json_path in walk_directory(dir)? {
            if write_provenance_if_missing(&json_path, crate_commit.as_deref(), &timestamp)? {
                println!("wrote provenance for {}", json_path.display());
                written += 1;
            } else {
                skipped += 1;
            }
        }
    }
    println!("\n{written} written, {skipped} already present.");
    Ok(())
}
