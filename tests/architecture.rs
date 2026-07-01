use std::fs;
use std::path::Path;

const PURE_LAYER_DIRS: &[&str] = &["src/linalg", "src/types"];
const FORBIDDEN_UPSTREAM_MODULES: &[&str] = &["model", "stats", "compiler", "pathology", "formula"];

#[test]
fn test_no_compiler_certificate_module() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    assert!(
        !root.join("src/compiler/certificate.rs").exists(),
        "compiler/certificate.rs was deleted; pathology::certificate is the single source of truth"
    );

    let compiler_mod = fs::read_to_string(root.join("src/compiler/mod.rs")).unwrap();
    assert!(
        !compiler_mod.contains("mod certificate") && !compiler_mod.contains("pub mod certificate"),
        "compiler::certificate must not be reintroduced"
    );
}

#[test]
fn test_linalg_and_types_do_not_import_upstream_layers() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut violations = Vec::new();
    let mut saw_linalg_error_import = false;
    let mut saw_types_error_import = false;
    let mut saw_types_linalg_import = false;

    for dir in PURE_LAYER_DIRS {
        for path in rust_files_under(&root.join(dir)) {
            let rel_path = path.strip_prefix(root).unwrap_or(&path);
            let source = fs::read_to_string(&path).unwrap_or_else(|err| {
                panic!("failed to read {}: {err}", rel_path.display());
            });
            let mut crate_group_depth = 0usize;

            for (line_idx, line) in source.lines().enumerate() {
                let line_no = line_idx + 1;
                let code = line_without_comment(line);
                let compact = compact_code(code);

                if compact.contains("crate::error") && rel_path.starts_with("src/linalg") {
                    saw_linalg_error_import = true;
                }
                if compact.contains("crate::error") && rel_path.starts_with("src/types") {
                    saw_types_error_import = true;
                }
                if compact.contains("crate::linalg") && rel_path.starts_with("src/types") {
                    saw_types_linalg_import = true;
                }

                for module in FORBIDDEN_UPSTREAM_MODULES {
                    if contains_crate_module(&compact, module)
                        || contains_forbidden_in_crate_group(&compact, module, crate_group_depth)
                    {
                        violations.push(format!("{}:{line_no}: {line}", rel_path.display()));
                    }
                }

                crate_group_depth = update_crate_group_depth(crate_group_depth, &compact);
            }
        }
    }

    assert!(
        violations.is_empty(),
        "linalg/types must not import model/stats/compiler/pathology/formula:\n{}",
        violations.join("\n")
    );
    assert!(
        saw_linalg_error_import,
        "architecture smoke check expected linalg to be allowed to use crate::error"
    );
    assert!(
        saw_types_error_import,
        "architecture smoke check expected types to be allowed to use crate::error"
    );
    assert!(
        saw_types_linalg_import,
        "architecture smoke check expected types to be allowed to use crate::linalg"
    );
}

/// The LMM objective kernel (`model::kernel`) is numerical-core machinery:
/// it may use error/types and the block-Cholesky entry points in
/// `model::linear`, but must not depend on the stats/compiler/report layers.
#[test]
fn test_lmm_kernel_does_not_import_upper_layers() {
    const KERNEL_FORBIDDEN_MODULES: &[&str] =
        &["stats", "compiler", "pathology", "guide", "datasets"];

    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let rel_path = "src/model/kernel.rs";
    let source = fs::read_to_string(root.join(rel_path)).unwrap_or_else(|err| {
        panic!("failed to read {rel_path}: {err}");
    });

    let mut violations = Vec::new();
    let mut crate_group_depth = 0usize;
    for (line_idx, line) in source.lines().enumerate() {
        let line_no = line_idx + 1;
        let code = line_without_comment(line);
        let compact = compact_code(code);

        for module in KERNEL_FORBIDDEN_MODULES {
            if contains_crate_module(&compact, module)
                || contains_forbidden_in_crate_group(&compact, module, crate_group_depth)
            {
                violations.push(format!("{rel_path}:{line_no}: {line}"));
            }
        }

        crate_group_depth = update_crate_group_depth(crate_group_depth, &compact);
    }

    assert!(
        violations.is_empty(),
        "model/kernel.rs must not import stats/compiler/pathology/guide/datasets:\n{}",
        violations.join("\n")
    );
}

fn rust_files_under(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();
    collect_rust_files(dir, &mut files);
    files.sort();
    files
}

fn collect_rust_files(dir: &Path, files: &mut Vec<std::path::PathBuf>) {
    for entry in fs::read_dir(dir).unwrap_or_else(|err| {
        panic!("failed to read directory {}: {err}", dir.display());
    }) {
        let path = entry.unwrap().path();
        if path.is_dir() {
            collect_rust_files(&path, files);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            files.push(path);
        }
    }
}

fn line_without_comment(line: &str) -> &str {
    line.split_once("//").map(|(code, _)| code).unwrap_or(line)
}

fn compact_code(code: &str) -> String {
    code.chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>()
}

fn contains_crate_module(compact: &str, module: &str) -> bool {
    let marker = format!("crate::{module}");
    compact
        .match_indices(&marker)
        .any(|(idx, _)| is_module_boundary(compact, idx + marker.len()))
}

fn contains_forbidden_in_crate_group(
    compact: &str,
    module: &str,
    crate_group_depth: usize,
) -> bool {
    let marker = format!("crate::{{{module}");
    compact.contains(&marker)
        || (crate_group_depth > 0
            && compact
                .match_indices(module)
                .any(|(idx, _)| is_module_token(compact, idx, module)))
}

fn update_crate_group_depth(mut depth: usize, compact: &str) -> usize {
    let mut rest = compact;
    while let Some(start) = rest.find("crate::{") {
        depth += 1;
        rest = &rest[start + "crate::{".len()..];
    }
    depth = depth.saturating_sub(compact.matches('}').count());
    depth
}

fn is_module_boundary(text: &str, idx: usize) -> bool {
    match text[idx..].chars().next() {
        Some(ch) => !matches!(ch, 'a'..='z' | 'A'..='Z' | '0'..='9' | '_'),
        None => true,
    }
}

fn is_module_token(text: &str, idx: usize, module: &str) -> bool {
    let starts_on_boundary = match text[..idx].chars().next_back() {
        Some(ch) => !matches!(ch, 'a'..='z' | 'A'..='Z' | '0'..='9' | '_'),
        None => true,
    };
    starts_on_boundary && is_module_boundary(text, idx + module.len())
}
