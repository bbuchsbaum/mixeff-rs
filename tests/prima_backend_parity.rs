use serde::Deserialize;
use std::collections::BTreeSet;

use mixedmodels::types::opt_summary::OptimizerBackend;
use mixedmodels::types::{OptSummary, Optimizer};

#[derive(Debug, Deserialize)]
struct PrimaBackendMatrix {
    schema_version: String,
    source: String,
    notes: Vec<String>,
    backends: Vec<PrimaBackendCase>,
}

#[derive(Debug, Deserialize)]
struct PrimaBackendCase {
    julia_backend: String,
    julia_optimizer: String,
    rust_optimizer: String,
    rust_backend: String,
    rust_optimizer_name: String,
    rust_optimizer_code: String,
    rust_status: String,
    rust_feature: String,
    required_library: String,
    opt_params: Vec<String>,
    numeric_parity: String,
}

fn fixture() -> PrimaBackendMatrix {
    serde_json::from_str(include_str!("fixtures/parity/prima_backend_matrix.json")).unwrap()
}

fn optimizer_by_variant(name: &str) -> Optimizer {
    match name {
        "PrimaBobyqa" => Optimizer::PrimaBobyqa,
        "PrimaCobyla" => Optimizer::PrimaCobyla,
        "PrimaLincoa" => Optimizer::PrimaLincoa,
        "PrimaNewuoa" => Optimizer::PrimaNewuoa,
        other => panic!("unexpected optimizer variant in PRIMA fixture: {other}"),
    }
}

#[test]
fn prima_backend_manifest_matches_rust_optimizer_labels() {
    let expected = fixture();
    assert_eq!(expected.schema_version, "1.0.0");
    assert!(expected.source.contains("MixedModels.jl"));
    assert!(expected.notes.iter().any(|note| note.contains("libprimac")));
    assert_eq!(expected.backends.len(), 4);

    let mut julia_optimizers = expected
        .backends
        .iter()
        .map(|case| case.julia_optimizer.as_str())
        .collect::<Vec<_>>();
    julia_optimizers.sort_unstable();
    assert_eq!(julia_optimizers, ["bobyqa", "cobyla", "lincoa", "newuoa"]);

    for case in &expected.backends {
        let optimizer = optimizer_by_variant(&case.rust_optimizer);
        let mut optsum = OptSummary::new(vec![0.1, 0.2]);
        optsum.optimizer = optimizer;

        assert_eq!(case.julia_backend, "prima");
        assert_eq!(case.rust_backend, optimizer.canonical_backend().label());
        assert_eq!(case.rust_backend, optsum.backend_name());
        assert_eq!(case.rust_optimizer_name, optsum.optimizer_name());
        assert_eq!(case.rust_optimizer_code, optsum.optimizer_code());
        assert_eq!(
            case.opt_params,
            OptimizerBackend::Prima
                .opt_params()
                .iter()
                .map(|param| param.to_string())
                .collect::<Vec<_>>()
        );
        assert_eq!(case.rust_feature, "prima");
        assert_eq!(case.required_library, "libprimac");
        assert!(case.numeric_parity.starts_with("deferred"));
    }
}

#[test]
fn prima_backend_manifest_keeps_only_bobyqa_wired_for_now() {
    let expected = fixture();
    let unique_julia_optimizers = expected
        .backends
        .iter()
        .map(|case| case.julia_optimizer.as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        unique_julia_optimizers.len(),
        expected.backends.len(),
        "PRIMA backend manifest must keep one row per Julia optimizer"
    );

    let wired = expected
        .backends
        .iter()
        .filter(|case| case.rust_status == "feature_gated_system_lib")
        .map(|case| case.rust_optimizer.as_str())
        .collect::<Vec<_>>();
    assert_eq!(wired, ["PrimaBobyqa"]);

    let reserved = expected
        .backends
        .iter()
        .filter(|case| case.rust_status == "reserved_unavailable")
        .map(|case| case.rust_optimizer.as_str())
        .collect::<Vec<_>>();
    assert_eq!(reserved, ["PrimaCobyla", "PrimaLincoa", "PrimaNewuoa"]);

    for case in expected
        .backends
        .iter()
        .filter(|case| case.rust_status == "reserved_unavailable")
    {
        assert_eq!(
            case.numeric_parity, "deferred_unwired_optimizer",
            "reserved PRIMA optimizers should be explicit unavailable states, not silent skips"
        );
    }
}

#[test]
fn prima_backend_markdown_mentions_every_manifest_row() {
    let expected = fixture();
    let markdown = include_str!("../docs/prima_backend_parity.md");

    for case in &expected.backends {
        assert!(markdown.contains(&case.julia_optimizer));
        assert!(markdown.contains(&case.rust_optimizer));
        assert!(markdown.contains(&case.rust_status));
    }
    assert!(markdown.contains("libprimac"));
    assert!(markdown.contains("not a default dependency"));
}
