use approx::assert_relative_eq;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct EngineReference {
    schema_version: String,
    fixture: String,
    stratum: String,
    engine: String,
    status: String,
    warnings: Vec<String>,
    converged: bool,
    objective: Option<f64>,
    theta: Vec<f64>,
    beta: Vec<f64>,
    sigma: Option<f64>,
    loglik: Option<f64>,
    runtime_ms: Option<f64>,
    #[serde(default)]
    singular: Option<bool>,
}

#[derive(Debug, Clone, Copy)]
struct RustOutcome {
    fixture: &'static str,
    certificate_stratum: &'static str,
    status: &'static str,
}

const RUST_OUTCOMES: &[RustOutcome] = &[
    RustOutcome {
        fixture: "easy_full_rank",
        certificate_stratum: "easy",
        status: "ConvergedInterior",
    },
    RustOutcome {
        fixture: "reduced_rank_unit_correlation",
        certificate_stratum: "reduced_rank",
        status: "ConvergedReducedRank",
    },
];

fn references() -> Vec<EngineReference> {
    [
        include_str!("fixtures/pathology_corpus/easy_full_rank/parity/lme4.json"),
        include_str!("fixtures/pathology_corpus/easy_full_rank/parity/mmjl.json"),
        include_str!("fixtures/pathology_corpus/reduced_rank_unit_correlation/parity/lme4.json"),
        include_str!("fixtures/pathology_corpus/reduced_rank_unit_correlation/parity/mmjl.json"),
    ]
    .into_iter()
    .map(|json| serde_json::from_str(json).unwrap())
    .collect()
}

fn reference<'a>(refs: &'a [EngineReference], fixture: &str, engine: &str) -> &'a EngineReference {
    refs.iter()
        .find(|reference| reference.fixture == fixture && reference.engine == engine)
        .unwrap_or_else(|| panic!("missing {engine} reference for {fixture}"))
}

fn rust_outcome(fixture: &str) -> RustOutcome {
    RUST_OUTCOMES
        .iter()
        .copied()
        .find(|outcome| outcome.fixture == fixture)
        .unwrap_or_else(|| panic!("missing Rust outcome for {fixture}"))
}

fn verdict(lme4: &EngineReference, mmjl: &EngineReference, rust: RustOutcome) -> &'static str {
    if rust.certificate_stratum == "easy"
        && lme4.status == "ok"
        && mmjl.status == "ok"
        && rust.status == "ConvergedInterior"
    {
        "parity"
    } else {
        "documented_divergence"
    }
}

fn scoreboard_markdown(refs: &[EngineReference]) -> String {
    let mut table = String::from(
        "| fixture | certificate stratum | lme4 | MixedModels.jl | rust | verdict |\n\
         | --- | --- | --- | --- | --- | --- |\n",
    );
    for fixture in ["easy_full_rank", "reduced_rank_unit_correlation"] {
        let lme4 = reference(refs, fixture, "lme4::lmer");
        let mmjl = reference(refs, fixture, "MixedModels.jl");
        let rust = rust_outcome(fixture);
        table.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} |\n",
            fixture,
            rust.certificate_stratum,
            lme4.status,
            mmjl.status,
            rust.status,
            verdict(lme4, mmjl, rust)
        ));
    }
    table
}

#[test]
fn cross_engine_reference_json_shape_is_stable() {
    for reference in references() {
        assert_eq!(reference.schema_version, "1.0.0");
        assert!(!reference.fixture.is_empty());
        assert!(!reference.stratum.is_empty());
        assert!(reference.warnings.iter().all(|warning| !warning.is_empty()));
        assert!(matches!(
            reference.status.as_str(),
            "ok" | "error" | "unavailable"
        ));
        assert!(reference.runtime_ms.unwrap_or(0.0) >= 0.0);
        assert!(
            reference
                .warnings
                .iter()
                .all(|warning| !warning.trim().is_empty()),
            "warnings should be omitted or carry non-empty messages"
        );
        if reference.status == "ok" {
            assert!(reference.converged || reference.singular == Some(true));
            assert!(reference.objective.unwrap().is_finite());
            assert_relative_eq!(
                reference.objective.unwrap(),
                -2.0 * reference.loglik.unwrap(),
                epsilon = 1e-8,
                max_relative = 1e-8
            );
            assert!(!reference.beta.is_empty());
            assert!(reference.sigma.unwrap().is_finite());
            assert!(reference.loglik.unwrap().is_finite());
            assert!(reference.theta.iter().all(|value| value.is_finite()));
        }
    }
}

#[test]
fn fully_identified_fixture_checks_loglik_before_coefficients() {
    let refs = references();
    let lme4 = reference(&refs, "easy_full_rank", "lme4::lmer");
    let mmjl = reference(&refs, "easy_full_rank", "MixedModels.jl");

    let lme4_loglik = lme4.loglik.unwrap();
    let mmjl_loglik = mmjl.loglik.unwrap();
    let n = 72.0;
    assert!(
        ((lme4_loglik - mmjl_loglik).abs() / n) < 1e-8,
        "log-likelihood parity must hold before coefficient parity is asserted"
    );

    for (actual, expected) in lme4.beta.iter().zip(mmjl.beta.iter()) {
        assert_relative_eq!(*actual, *expected, epsilon = 1e-8, max_relative = 1e-8);
    }
}

#[test]
fn cross_engine_scoreboard_is_reference_order_invariant() {
    let refs = references();
    let baseline = scoreboard_markdown(&refs);

    let mut reversed = refs;
    reversed.reverse();

    assert_eq!(
        scoreboard_markdown(&reversed),
        baseline,
        "scoreboard verdicts must depend on fixture/engine keys, not JSON load order"
    );
}

#[test]
fn cross_engine_scoreboard_documents_divergence_without_oracle_promotion() {
    let refs = references();
    let table = scoreboard_markdown(&refs);

    assert!(table.contains("easy_full_rank | easy"));
    assert!(table.contains("easy_full_rank | easy | ok | ok | ConvergedInterior | parity"));
    assert!(table.contains(
        "reduced_rank_unit_correlation | reduced_rank | ok | ok | ConvergedReducedRank | documented_divergence"
    ));

    let reduced_lme4 = reference(&refs, "reduced_rank_unit_correlation", "lme4::lmer");
    let reduced_mmjl = reference(&refs, "reduced_rank_unit_correlation", "MixedModels.jl");
    assert_eq!(reduced_lme4.status, "ok");
    assert_eq!(reduced_mmjl.status, "ok");
    assert_eq!(reduced_lme4.stratum, "reduced-rank");
    assert!(
        (reduced_lme4.objective.unwrap() - reduced_mmjl.objective.unwrap()).abs() > 1.0,
        "reduced-rank fixture records engine behavior as comparison data, not a single oracle"
    );

    assert!(table.lines().any(|line| line.contains("parity")));
    assert!(table
        .lines()
        .any(|line| line.contains("documented_divergence")));
}
