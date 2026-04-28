//! Pathology corpus integration test.
//!
//! For each stratum (easy, boundary, reduced_rank, refusal) we:
//!
//! 1. Build a `GeneratorSpec` describing the design + truth.
//! 2. Run `certify(&spec)` — pure linear algebra, no engine — to derive the
//!    identifiability certificate and the *set* of acceptable
//!    `FitStatus` values per the contract.
//! 3. Generate the dataset deterministically from `spec.seed`.
//! 4. Build + fit a `LinearMixedModel` (or capture the construction/fit
//!    error).
//! 5. Reduce the outcome to a single effective `FitStatus` via
//!    `effective_status` and assert it is a member of the expected set.
//!
//! The expected set is intentionally not a single value: truth on a
//! contract boundary can legitimately surface as more than one status
//! depending on optimizer landing point. The test asserts membership, never
//! equality, so it does not flake on optimizer noise — but a `Refusal` that
//! becomes a `Converged` (or vice versa) is always a regression.
//!
//! See `bd-01KQ8FRGFQEQT8J217YB02D7CB` for the foundation issue and
//! `tests/fixtures/pathology_corpus/README.md` for stratum rationale.

use nalgebra::dmatrix;

use mixedmodels::compiler::FitStatus;
use mixedmodels::error::MixedModelError;
use mixedmodels::formula::parse_formula;
use mixedmodels::model::{Family, GeneralizedLinearMixedModel, LinearMixedModel, LinkFunction};
use mixedmodels::pathology::{
    block_diagonal_crossings, certify, collinear_fe, detect_separation, effective_status,
    effective_status_from_artifact, empty_crossings, expected_statuses, extreme_prevalence,
    generate, inferred_axes, lint_single_axis, map_error_to_status, near_singular_re,
    pareto_sizes, scale_mismatch, set_group_sizes, singletons_with_slope, BoundaryKind,
    Certificate, ExpectedStatusSet, FeSeparationKind, GeneratorSpec, PathologyAxis,
    SeparationKind, StructuralIssue, WEAK_ID_THRESHOLD,
};

/// Build the four foundation-stratum specs.
mod fixtures {
    use super::*;

    /// Easy: 30 groups × 6 obs, 1 random intercept + 1 random slope, mild
    /// correlation, generous within-group variation. Should converge cleanly
    /// to the interior of the parameter space.
    pub fn easy() -> GeneratorSpec {
        GeneratorSpec::lmm(
            "easy",
            42,
            vec![6; 30],
            vec![1.0, 2.0],
            true,
            1,
            dmatrix![4.0, 0.5; 0.5, 1.0],
        )
    }

    /// Boundary: same structure as `easy` but with the slope variance set
    /// to zero in truth. The optimizer must drive σ²_slope to its lower
    /// bound. Acceptable: `ConvergedBoundary` or `ConvergedInterior`
    /// (engine landing slightly off zero on noise).
    pub fn boundary() -> GeneratorSpec {
        GeneratorSpec::lmm(
            "boundary_zero_slope",
            42,
            vec![6; 30],
            vec![1.0, 2.0],
            true,
            1,
            dmatrix![4.0, 0.0; 0.0, 0.0],
        )
    }

    /// Reduced-rank: 2-D random structure with truth correlation ρ = 1
    /// exactly. rank(Σ_truth) = 1, so the supported variance manifold is
    /// 1-D. Acceptable: `ConvergedReducedRank` or `ConvergedBoundary`
    /// (correlation parameter on its boundary).
    pub fn reduced_rank() -> GeneratorSpec {
        let mut spec = GeneratorSpec::lmm(
            "reduced_rank",
            42,
            vec![6; 30],
            vec![1.0, 2.0],
            true,
            1,
            dmatrix![4.0, 0.0; 0.0, 4.0],
        );
        near_singular_re(&mut spec, 1.0);
        spec
    }

    /// Refusal: 6 singleton groups with a `(1 + x | g)` structure. Slope
    /// variance is structurally unidentifiable (no within-group x
    /// variation). Acceptable: `NotIdentifiable`, `NotOptimized`, or
    /// `ConvergedBoundary` (engine pinned slope variance to zero).
    pub fn refusal() -> GeneratorSpec {
        GeneratorSpec::lmm(
            "refusal_singletons",
            42,
            vec![1; 6],
            vec![1.0, 2.0],
            true,
            1,
            dmatrix![4.0, 0.5; 0.5, 1.0],
        )
    }

    /// Imbalance via `pareto_sizes`: 30 groups with right-skewed sizes
    /// drawn deterministically from a fixed seed. Same identifiability
    /// stratum as `easy` (full rank, far from boundary) — the transform
    /// stresses the optimizer's handling of unequal cluster sizes, not
    /// identifiability.
    pub fn imbalance() -> GeneratorSpec {
        let sizes = pareto_sizes(7, 30, 1.5, 6.0);
        GeneratorSpec::lmm(
            "imbalance_pareto",
            42,
            sizes,
            vec![1.0, 2.0],
            true,
            1,
            dmatrix![4.0, 0.5; 0.5, 1.0],
        )
    }

    /// Scale mismatch: predictor x1 scaled by 1e3. Conditioning of X
    /// degrades by three orders of magnitude. Same identifiability
    /// stratum as `easy`; the transform stresses numerical conditioning.
    pub fn scale_mismatch_fixture() -> GeneratorSpec {
        let mut spec = GeneratorSpec::lmm(
            "scale_mismatch_1e3",
            42,
            vec![6; 30],
            vec![1.0, 2.0],
            true,
            1,
            dmatrix![4.0, 0.5; 0.5, 1.0],
        );
        scale_mismatch(&mut spec, vec![1e3]);
        spec
    }

    /// Perfectly collinear fixed-effect predictors: x1 and x2 have
    /// population correlation ρ = 1, so X is rank-deficient in
    /// expectation. The certificate flags this as
    /// [`StructuralIssue::CollinearFixedEffects`].
    pub fn collinear_fe_perfect() -> GeneratorSpec {
        let mut spec = GeneratorSpec::lmm(
            "collinear_fe_rho_one",
            42,
            vec![6; 30],
            vec![1.0, 2.0, 3.0],
            true,
            1,
            dmatrix![4.0, 0.5; 0.5, 1.0],
        );
        collinear_fe(&mut spec, 0, 1, 1.0);
        spec
    }

    /// Extreme prevalence: Bernoulli/Logit with intercept shift -5
    /// pushing prevalence below 1%. The transform also strips
    /// `n_re_slopes` to 0 because Bernoulli + random slopes + tiny
    /// prevalence is a separate pathology axis (see
    /// `bd-01KQ8FS7HK6TX2TMX0J0XFGYFD`).
    pub fn extreme_prevalence_low() -> GeneratorSpec {
        let mut spec = GeneratorSpec::lmm(
            "extreme_prevalence_negative_5",
            42,
            vec![20; 30],
            vec![0.0, 0.5],
            true,
            0,
            dmatrix![1.0],
        );
        extreme_prevalence(&mut spec, -5.0);
        spec
    }

    /// Singletons via the transform helper rather than constructed
    /// inline. Pathologically equivalent to `refusal` but exercises
    /// `singletons_with_slope` directly.
    pub fn singletons_via_transform() -> GeneratorSpec {
        let mut spec = GeneratorSpec::lmm(
            "singletons_via_transform",
            42,
            vec![6; 8],
            vec![1.0, 2.0],
            true,
            1,
            dmatrix![4.0, 0.5; 0.5, 1.0],
        );
        singletons_with_slope(&mut spec, 8);
        spec
    }

    /// Random-slope singletons: explicit fixture for bullet 1 of
    /// `bd-01KQ8FV99FN980Q2G7Z0KDWCZN`. 12 groups × 1 obs each with a
    /// `(1 + x | g)` slope requested. Within-group slope variance is
    /// structurally unidentifiable (no within-group x variation), so
    /// the certificate flags
    /// [`StructuralIssue::SingletonsWithSlope`] independent of seed and
    /// without any engine call.
    pub fn random_slope_singletons() -> GeneratorSpec {
        let mut spec = GeneratorSpec::lmm(
            "random_slope_singletons",
            7,
            vec![1; 12],
            vec![0.5, 1.5],
            true,
            1,
            dmatrix![2.0, 0.0; 0.0, 1.0],
        );
        singletons_with_slope(&mut spec, 12);
        spec
    }

    /// Crossed REs with structural empty crossings: explicit fixture
    /// for bullet 2 of `bd-01KQ8FV99FN980Q2G7Z0KDWCZN`. 4 disjoint
    /// `4×4` blocks of `(subj × item)` crossings. The bipartite cell
    /// graph has 4 disconnected components, so the certificate flags
    /// [`StructuralIssue::DisconnectedCrossings`]. Has a primary RE
    /// `(1 | g)` and a secondary intercept-only RE `(1 | h)`, so the
    /// engine constructs two `ReMat`s and the model sits on the
    /// crossed-RE fit path that the multi-grouping-factor blocked
    /// system targets.
    pub fn crossed_block_diagonal() -> GeneratorSpec {
        let mut spec = GeneratorSpec::lmm(
            "crossed_block_diagonal_4x4x4",
            42,
            vec![1; 1],
            vec![1.0],
            true,
            0,
            dmatrix![1.5],
        );
        spec.group_name = "g".into();
        block_diagonal_crossings(&mut spec, "h", 4, 4, 0.8);
        spec
    }

    /// Weakly-identified design: a small (3 obs/group × 4 groups = 12
    /// total) dataset with two predictors at population correlation
    /// `ρ = 0.99` and a structurally connected `(1 + x | g)` random
    /// effect. Identifiable in principle (rank-2 corr matrix, no
    /// structural issue) but the dimensionless weak-id index falls below
    /// [`WEAK_ID_THRESHOLD`], so [`expected_statuses`] widens to admit
    /// `ConvergedReducedRank` alongside the usual converged set.
    pub fn weakly_identified() -> GeneratorSpec {
        let mut spec = GeneratorSpec::lmm(
            "weakly_identified_near_collinear",
            42,
            vec![3; 4],
            vec![1.0, 2.0, 3.0],
            true,
            1,
            dmatrix![4.0, 0.5; 0.5, 1.0],
        );
        collinear_fe(&mut spec, 0, 1, 0.99);
        spec
    }

    /// Fixed-effect-only separation fixture for
    /// `bd-01KQ8FS7HK6TX2TMX0J0XFGYFD`. A Bernoulli/Logit design with
    /// an enormous slope (1e6) drives the realised response to track
    /// the sign of the predictor exactly: `|β · x|` dominates the
    /// random-intercept noise (`σ_RE = 1`) for every observation
    /// outside an `x ≈ 0` band of width `~ σ_RE / |β|` ≈ `1e-6`, which
    /// is statistically empty under `x ~ N(0, 1)` at corpus sizes. The
    /// FE design `[1, x]` therefore admits the hyperplane `x = 0` as
    /// a complete separator. Group sizes are kept large (20 obs each)
    /// so the chance of any individual group falling on one side of
    /// the separator is vanishingly small (≈ 2 · 0.5^20 per group);
    /// for the chosen seed the realised data has zero conditionally-
    /// separated groups, isolating FE separation from the conditional
    /// tier.
    pub fn fe_separation_extreme_slope() -> GeneratorSpec {
        let mut spec = GeneratorSpec::lmm(
            "fe_separation_extreme_slope",
            7,
            vec![20; 30],
            vec![0.0, 1e6],
            true,
            0,
            dmatrix![1.0],
        );
        spec.family = Family::Bernoulli;
        spec.link = LinkFunction::Logit;
        spec.residual_sd = 0.0;
        spec
    }

    /// Conditional-only separation fixture for
    /// `bd-01KQ8FS7HK6TX2TMX0J0XFGYFD`. A Bernoulli/Logit design with
    /// rare-event prevalence (intercept shift -1.5 → base p ≈ 0.18)
    /// and a mild slope (0.5) is unlikely to admit a separating
    /// hyperplane in `[1, x]`, but with only 4 observations per group
    /// the per-group probability of all-zero outcomes is roughly
    /// `0.82^4 ≈ 0.45`, so multiple groups end up conditionally
    /// separated under a fixed seed. The resulting structural issue is
    /// `SeparationKind::Conditional`, distinct from FE-only and Both.
    pub fn conditional_separation_rare_events() -> GeneratorSpec {
        let mut spec = GeneratorSpec::lmm(
            "conditional_separation_rare_events",
            42,
            vec![4; 20],
            vec![0.0, 0.5],
            true,
            0,
            dmatrix![1.0],
        );
        extreme_prevalence(&mut spec, -1.5);
        spec
    }

    /// Separation-stratum fixture for `bd-01KQ8FSHVBDS85KS0KM4867VBK`.
    /// Bernoulli/Logit design with an extreme intercept shift drives
    /// the realised response to all-zero with overwhelming probability,
    /// firing both tiers of the separation detector: FE (Konis 2007
    /// complete separation) and conditional (every group has all-zero
    /// outcomes). The certificate flags
    /// `SeparationKind::Both { fe_kind: Complete, n_groups: 30 }` and
    /// [`expected_statuses`] admits `ConvergedPenalised` alongside
    /// `NotIdentifiable` and `NotOptimized`.
    pub fn separation_extreme_prevalence() -> GeneratorSpec {
        let mut spec = GeneratorSpec::lmm(
            "separation_extreme_prevalence_negative_15",
            42,
            vec![20; 30],
            vec![0.0, 0.5],
            true,
            0,
            dmatrix![1.0],
        );
        extreme_prevalence(&mut spec, -15.0);
        spec
    }

    /// Crossed REs with random sparse cells, but all retained levels
    /// participate in a single connected component. Counter-example
    /// fixture: certificate must *not* flag a structural issue when the
    /// bipartite graph stays connected, even at low density. Useful as
    /// a regression target alongside `crossed_block_diagonal` to make
    /// sure the disconnection detector keys on graph topology and not
    /// raw density.
    pub fn crossed_sparse_connected() -> GeneratorSpec {
        let mut spec = GeneratorSpec::lmm(
            "crossed_sparse_connected",
            42,
            vec![1; 12],
            vec![1.0],
            true,
            0,
            dmatrix![1.5],
        );
        // Density 0.5 with seed 11 gives a sparse-but-connected graph
        // for the 12×12 design. The exact connectivity is verified by
        // the `crossed_sparse_connected_has_one_component` test below;
        // if the seed ever produces a disconnected sample, that test
        // will surface it before any contract assertion runs.
        empty_crossings(&mut spec, "h", 12, 0.6, 0.5, 11);
        spec
    }
}

fn try_fit(spec: &GeneratorSpec) -> Result<LinearMixedModel, MixedModelError> {
    let out = generate(spec);
    let formula = parse_formula(&out.formula).map_err(MixedModelError::from)?;
    let mut model = LinearMixedModel::new(formula, &out.data, None)?;
    model.fit(true)?;
    Ok(model)
}

fn assert_status_in_set(spec: &GeneratorSpec) -> (Certificate, ExpectedStatusSet, FitStatus) {
    let cert = certify(spec);
    let expected = expected_statuses(&cert);

    let result = try_fit(spec);
    let status = match &result {
        Ok(model) => effective_status(Ok(model)),
        Err(err) => effective_status(Err(err)),
    };

    assert!(
        expected.contains(status),
        "fixture '{}': engine returned {:?} but expected one of {:?} ({})",
        spec.label,
        status,
        expected.allowed,
        expected.rationale
    );

    (cert, expected, status)
}

#[test]
fn easy_stratum_converges_to_interior() {
    let spec = fixtures::easy();
    let (cert, expected, status) = assert_status_in_set(&spec);
    assert!(cert.boundary_directions.is_empty());
    assert!(cert.structural_issue.is_none());
    assert_eq!(expected.allowed, vec![FitStatus::ConvergedInterior]);
    assert_eq!(status, FitStatus::ConvergedInterior);
}

#[test]
fn boundary_stratum_lands_on_or_near_zero_variance() {
    let spec = fixtures::boundary();
    let (cert, expected, _) = assert_status_in_set(&spec);
    assert!(cert
        .boundary_directions
        .iter()
        .any(|b| matches!(b, mixedmodels::pathology::BoundaryKind::ZeroVariance { .. })));
    assert!(expected.contains(FitStatus::ConvergedBoundary));
}

#[test]
fn reduced_rank_stratum_collapses_to_supported_subspace() {
    let spec = fixtures::reduced_rank();
    let (cert, expected, _) = assert_status_in_set(&spec);
    assert!(cert.re_rank_truth < cert.re_rank_requested);
    assert!(expected.contains(FitStatus::ConvergedReducedRank));
}

#[test]
fn refusal_stratum_rejects_or_pins_slope_to_boundary() {
    let spec = fixtures::refusal();
    let (cert, expected, _) = assert_status_in_set(&spec);
    assert!(cert.structural_issue.is_some());
    assert!(expected.contains(FitStatus::NotIdentifiable));
}

#[test]
fn certify_is_deterministic_and_seed_independent() {
    // Identifiability certification must depend on the design + truth, not
    // the seed. Same spec with two different seeds → identical certificates.
    let mut a = fixtures::reduced_rank();
    let mut b = fixtures::reduced_rank();
    a.seed = 1;
    b.seed = 999;
    let cert_a = certify(&a);
    let cert_b = certify(&b);
    assert_eq!(cert_a, cert_b);
}

#[test]
fn imbalance_transform_produces_right_skewed_group_sizes() {
    let spec = fixtures::imbalance();
    let max = *spec.group_sizes.iter().max().unwrap() as f64;
    let min = *spec.group_sizes.iter().min().unwrap() as f64;
    assert!(
        max / min >= 3.0,
        "imbalance sizes too uniform: min={min} max={max}; pareto_sizes is supposed to be skewed"
    );
    // Identifiability is the same as `easy` — imbalance is a numerical
    // pathology, not a structural one.
    let (cert, expected, status) = assert_status_in_set(&spec);
    assert!(cert.structural_issue.is_none());
    assert!(expected.contains(status));
}

#[test]
fn scale_mismatch_transform_preserves_identifiability() {
    let spec = fixtures::scale_mismatch_fixture();
    assert_eq!(spec.fe_scales, vec![1e3]);
    let (cert, expected, status) = assert_status_in_set(&spec);
    // Scale mismatch is a conditioning pathology; identifiability is
    // unaffected and the engine should still converge cleanly.
    assert!(cert.structural_issue.is_none());
    assert!(expected.contains(status));
}

#[test]
fn collinear_fe_transform_triggers_structural_unidentifiability() {
    let spec = fixtures::collinear_fe_perfect();
    let cert = certify(&spec);
    assert!(
        matches!(
            cert.structural_issue,
            Some(StructuralIssue::CollinearFixedEffects { .. })
        ),
        "expected CollinearFixedEffects, got {:?}",
        cert.structural_issue
    );
    assert_eq!(cert.fe_rank_truth, 1);
    assert_eq!(cert.fe_rank_requested, 2);

    let expected = expected_statuses(&cert);
    assert!(expected.contains(FitStatus::NotIdentifiable));

    let result = try_fit(&spec);
    let status = match &result {
        Ok(model) => effective_status(Ok(model)),
        Err(err) => effective_status(Err(err)),
    };
    assert!(
        expected.contains(status),
        "fixture '{}': engine returned {:?} but expected one of {:?}",
        spec.label,
        status,
        expected.allowed
    );
}

#[test]
fn extreme_prevalence_transform_promotes_to_bernoulli_logit() {
    let spec = fixtures::extreme_prevalence_low();
    assert_eq!(spec.family, Family::Bernoulli);
    assert_eq!(spec.link, LinkFunction::Logit);
    assert_eq!(spec.binary_intercept_shift, -5.0);

    let out = generate(&spec);
    let y_col = out.data.numeric(&spec.response_name).unwrap();
    let prevalence: f64 = y_col.iter().sum::<f64>() / y_col.len() as f64;
    assert!(
        prevalence < 0.10,
        "expected prevalence < 10% with intercept shift -5, got {prevalence:.3}"
    );

    // GLMM fit via dispatch — extreme prevalence may or may not exhibit
    // separation in this realised sample; we assert only that the
    // returned status is *some* member of the FitStatus enum and that
    // it doesn't crash. Tighter assertions land under
    // bd-01KQ8FS7HK6TX2TMX0J0XFGYFD (separation detection).
    let formula = parse_formula(&out.formula).unwrap();
    let status =
        match GeneralizedLinearMixedModel::new(formula, &out.data, spec.family, Some(spec.link)) {
            Ok(mut model) => match model.fit() {
                Ok(_) => effective_status_from_artifact(model.compiler_artifact()),
                Err(e) => map_error_to_status(&e),
            },
            Err(e) => map_error_to_status(&e),
        };
    // Sanity: status must be one of the legitimate enum values, not a panic.
    assert!(matches!(
        status,
        FitStatus::ConvergedInterior
            | FitStatus::ConvergedBoundary
            | FitStatus::ConvergedReducedRank
            | FitStatus::NotIdentifiable
            | FitStatus::NotOptimized
            | FitStatus::NotAssessed
    ));
}

#[test]
fn singletons_via_transform_matches_inline_singleton_fixture() {
    let spec = fixtures::singletons_via_transform();
    assert!(spec.group_sizes.iter().all(|&n| n == 1));
    let cert = certify(&spec);
    assert!(matches!(
        cert.structural_issue,
        Some(StructuralIssue::SingletonsWithSlope { .. })
    ));
    let expected = expected_statuses(&cert);
    assert!(expected.contains(FitStatus::NotIdentifiable));
}

#[test]
fn transforms_compose_without_field_collisions() {
    // Apply two orthogonal transforms (scale_mismatch + near_singular_re)
    // on top of an `easy` base spec; both should take effect, certificate
    // should reflect both.
    let mut spec = fixtures::easy();
    spec.label = "compose_scale_and_near_singular".into();
    scale_mismatch(&mut spec, vec![1e2]);
    near_singular_re(&mut spec, 0.999);

    assert_eq!(spec.fe_scales, vec![1e2]);
    let off = spec.re_cov_truth[(0, 1)];
    let denom = (spec.re_cov_truth[(0, 0)] * spec.re_cov_truth[(1, 1)]).sqrt();
    let realised_rho = off / denom;
    assert!((realised_rho - 0.999).abs() < 1e-6);

    let cert = certify(&spec);
    // Near-singular but not exactly singular → still rank 2 by tol
    // (UNIT_CORRELATION_TOL = 1e-6, so ρ=0.999 is detectably below 1)
    assert!(cert.structural_issue.is_none());
}

#[test]
fn set_group_sizes_overrides_existing_sizes() {
    let mut spec = fixtures::easy();
    let original_total = spec.n_total();
    set_group_sizes(&mut spec, vec![3; 10]);
    assert_eq!(spec.n_total(), 30);
    assert_ne!(spec.n_total(), original_total);
}

// --- bd-01KQ8FV99FN980Q2G7Z0KDWCZN -------------------------------------
// Random-slope singletons + crossed REs with empty crossings. Bullet 1
// (singletons) is structurally identical to the existing `refusal` /
// `singletons_via_transform` fixtures but is asserted under its own
// explicit name so the corpus has one named fixture per pathology
// listed in the issue. Bullet 2 (crossed empty crossings) is the new
// pathology axis that required extending `GeneratorSpec` with an
// optional secondary grouping factor.

#[test]
fn random_slope_singletons_certifies_structurally_unidentifiable() {
    let spec = fixtures::random_slope_singletons();
    let cert = certify(&spec);
    assert!(matches!(
        cert.structural_issue,
        Some(StructuralIssue::SingletonsWithSlope { .. })
    ));
    let expected = expected_statuses(&cert);
    assert!(expected.contains(FitStatus::NotIdentifiable));
    // Engine probe: must land in the acceptable set without panicking.
    let (_cert, _expected, _status) = assert_status_in_set(&spec);
}

#[test]
fn random_slope_singletons_certificate_is_engine_free_and_seed_independent() {
    // Bullet 1's acceptance: certificate must be derivable without any
    // engine call and must not depend on the data seed. Two seeds, one
    // certificate.
    let mut a = fixtures::random_slope_singletons();
    let mut b = fixtures::random_slope_singletons();
    a.seed = 1;
    b.seed = 999;
    assert_eq!(certify(&a), certify(&b));
}

#[test]
fn crossed_block_diagonal_certifies_disconnected_components() {
    let spec = fixtures::crossed_block_diagonal();

    // Generator-only sanity: the spec carries a crossed secondary group.
    assert!(spec.crossed.is_some());
    let cells = spec.crossed_cells().unwrap();
    // 4 blocks × 4×4 cells = 64.
    assert_eq!(cells.len(), 64);

    // Certificate path is engine-free: pure linear algebra over the cells.
    let cert = certify(&spec);
    let summary = cert
        .crossed_summary
        .as_ref()
        .expect("crossed_summary must be populated for crossed specs");
    assert_eq!(summary.n_primary, 16);
    assert_eq!(summary.n_secondary, 16);
    assert_eq!(summary.n_components, 4);
    assert!(summary.primary_orphans.is_empty());
    assert!(summary.secondary_orphans.is_empty());

    assert!(matches!(
        cert.structural_issue,
        Some(StructuralIssue::DisconnectedCrossings { n_components: 4 })
    ));

    let expected = expected_statuses(&cert);
    assert!(expected.contains(FitStatus::NotIdentifiable));
    assert!(expected.contains(FitStatus::ConvergedInterior));
}

#[test]
fn crossed_block_diagonal_certificate_is_seed_independent() {
    let mut a = fixtures::crossed_block_diagonal();
    let mut b = fixtures::crossed_block_diagonal();
    a.seed = 1;
    b.seed = 999;
    assert_eq!(certify(&a), certify(&b));
}

#[test]
fn crossed_block_diagonal_engine_runs_with_two_grouping_factors() {
    // Acceptance criterion: the crossed-RE fixture must "verifiably touch
    // BlockedSparse code paths". The engine in this codebase uses the
    // multi-`ReMat` blocked-Cholesky path (see `promote_crossed_fill_in_blocks`
    // in src/model/linear.rs) once `model.reterms.len() >= 2`, which is
    // the regime BlockedSparse was introduced for. We therefore probe
    // the model's reterm count rather than instantiating BlockedSparse
    // directly.
    let spec = fixtures::crossed_block_diagonal();
    let out = generate(&spec);
    let formula = parse_formula(&out.formula).unwrap();
    let mut model = LinearMixedModel::new(formula, &out.data, None)
        .expect("crossed-RE LMM must construct on a structurally disconnected design");
    assert_eq!(
        model.reterms.len(),
        2,
        "crossed fixture must produce two grouping factors (g, h)"
    );
    // Fit can return Ok or an error depending on optimizer landing; the
    // contract is `effective_status` lands inside the expected set.
    let cert = certify(&spec);
    let expected = expected_statuses(&cert);
    let status = match model.fit(true) {
        Ok(_) => effective_status(Ok(&model)),
        Err(err) => effective_status(Err(&err)),
    };
    assert!(
        expected.contains(status),
        "crossed_block_diagonal: engine returned {:?} but expected one of {:?} ({})",
        status,
        expected.allowed,
        expected.rationale
    );
}

// --- bd-01KQ8FT90WXSG9VSZQH30HZY9P -------------------------------------
// Dimensionless weak-identification index. The certificate must produce
// the same `weak_id_score` after a 1e3 rescaling of any predictor — both
// uniform and per-axis — and the `weak_identification` flag must widen
// `expected_statuses` to admit `ConvergedReducedRank`.

#[test]
fn weak_id_score_invariant_under_uniform_and_per_axis_rescale() {
    // Uniform rescaling: scale_mismatch with all-equal scales → same score
    // as the un-rescaled spec.
    let base = fixtures::easy();
    let mut uniform = fixtures::easy();
    uniform.label = "easy_scaled_uniform".into();
    scale_mismatch(&mut uniform, vec![1e3]);

    let cert_base = certify(&base);
    let cert_uniform = certify(&uniform);
    assert!(
        (cert_base.weak_id_score - cert_uniform.weak_id_score).abs() < 1e-9,
        "weak_id_score moved under uniform 1e3 rescale: {} vs {}",
        cert_base.weak_id_score,
        cert_uniform.weak_id_score
    );

    // Per-axis rescaling on a multi-predictor spec.
    let mut multi = GeneratorSpec::lmm(
        "weak_id_multi_predictor",
        42,
        vec![6; 30],
        vec![1.0, 2.0, 3.0],
        true,
        1,
        dmatrix![4.0, 0.5; 0.5, 1.0],
    );
    let cert_multi = certify(&multi);
    scale_mismatch(&mut multi, vec![1.0, 1e3]);
    let cert_multi_scaled = certify(&multi);
    assert!(
        (cert_multi.weak_id_score - cert_multi_scaled.weak_id_score).abs() < 1e-9,
        "weak_id_score moved under per-axis 1e3 rescale: {} vs {}",
        cert_multi.weak_id_score,
        cert_multi_scaled.weak_id_score
    );
}

#[test]
fn weakly_identified_fixture_widens_expected_status_set() {
    let spec = fixtures::weakly_identified();
    let cert = certify(&spec);

    // Sanity: structurally identifiable (no FE-collinearity refusal).
    assert!(
        cert.structural_issue.is_none(),
        "weakly_identified should be structurally identifiable, got {:?}",
        cert.structural_issue
    );
    // The dimensionless score sits below the threshold.
    assert!(
        cert.weak_identification,
        "expected weak_identification = true, got score = {} threshold = {}",
        cert.weak_id_score, cert.weak_id_threshold
    );
    assert!(
        cert.weak_id_score < WEAK_ID_THRESHOLD,
        "weak_id_score {} should be below WEAK_ID_THRESHOLD {}",
        cert.weak_id_score,
        WEAK_ID_THRESHOLD
    );

    let exp = expected_statuses(&cert);
    assert!(
        exp.contains(FitStatus::ConvergedReducedRank),
        "weakly_identified set should admit ConvergedReducedRank, got {:?}",
        exp.allowed
    );
    assert!(exp.contains(FitStatus::ConvergedInterior));
}

#[test]
fn easy_fixture_does_not_trigger_weak_identification() {
    // Regression guard: the `easy` stratum must remain *not* weakly
    // identified, otherwise the threshold has been calibrated too tight.
    let spec = fixtures::easy();
    let cert = certify(&spec);
    assert!(
        !cert.weak_identification,
        "easy fixture should not be flagged weakly identified (score = {}, threshold = {})",
        cert.weak_id_score, cert.weak_id_threshold
    );
}

// --- bd-01KQ8FSHVBDS85KS0KM4867VBK -------------------------------------
// ConvergedPenalised admittance for separation-stratum fixtures. The
// certificate flags Bernoulli + extreme-intercept-shift specs as
// `StructuralIssue::Separation { kind: Unspecified }` (placeholder
// detection — proper LP-based detection lands under
// bd-01KQ8FS7HK6TX2TMX0J0XFGYFD). The dedicated `expected_statuses`
// branch returns `{NotIdentifiable, NotOptimized, ConvergedPenalised}`
// — narrower than the generic structural-issue set, because honestly
// reporting a separated logistic as `ConvergedInterior` would lie about
// the existence of an MLE.

#[test]
fn fe_separation_fixture_certifies_fe_only() {
    // Acceptance for bd-01KQ8FS7HK6TX2TMX0J0XFGYFD bullet 1: an FE-only
    // separation fixture must produce `SeparationKind::FixedEffect`
    // and admit refusal/penalised outcomes in expected_statuses.
    let spec = fixtures::fe_separation_extreme_slope();
    let cert = certify(&spec);
    assert!(
        matches!(
            cert.structural_issue,
            Some(StructuralIssue::Separation {
                kind: SeparationKind::FixedEffect(FeSeparationKind::Complete),
            })
        ),
        "expected SeparationKind::FixedEffect(Complete), got {:?}",
        cert.structural_issue
    );

    let exp = expected_statuses(&cert);
    assert!(exp.contains(FitStatus::NotIdentifiable));
    assert!(exp.contains(FitStatus::ConvergedPenalised));
    assert!(!exp.contains(FitStatus::ConvergedInterior));

    // Round-trip the rich report: confirm FE separation is detected
    // and conditional separation is empty.
    let report = detect_separation(&spec);
    assert_eq!(report.fe_kind, Some(FeSeparationKind::Complete));
    assert!(
        report.conditional_groups.is_empty(),
        "expected no conditional separation; got groups {:?}",
        report.conditional_groups
    );
    let beta = report
        .hyperplane_direction
        .expect("expected hyperplane direction for FE-separated fixture");
    // The slope (β[1]) must dominate the intercept (β[0]) since the
    // separator is ~ x = 0.
    assert!(
        beta[1].abs() > beta[0].abs(),
        "expected slope to dominate intercept in hyperplane β = {beta:?}"
    );
}

#[test]
fn conditional_separation_fixture_certifies_conditional_only() {
    // Acceptance for bd-01KQ8FS7HK6TX2TMX0J0XFGYFD bullet 2: a
    // conditional-only separation fixture must produce
    // `SeparationKind::Conditional` (not `Both`, not `FixedEffect`)
    // and admit refusal/penalised outcomes in expected_statuses.
    let spec = fixtures::conditional_separation_rare_events();
    let cert = certify(&spec);
    let n_groups_seen = match cert.structural_issue {
        Some(StructuralIssue::Separation {
            kind: SeparationKind::Conditional { n_groups },
        }) => n_groups,
        other => panic!("expected SeparationKind::Conditional, got {other:?}"),
    };
    assert!(
        n_groups_seen >= 1,
        "expected at least one conditionally-separated group, got {n_groups_seen}"
    );

    let exp = expected_statuses(&cert);
    assert!(exp.contains(FitStatus::NotIdentifiable));
    assert!(exp.contains(FitStatus::ConvergedPenalised));
    assert!(!exp.contains(FitStatus::ConvergedInterior));

    let report = detect_separation(&spec);
    assert!(
        report.fe_kind.is_none(),
        "expected no FE separation; got fe_kind = {:?}",
        report.fe_kind
    );
    assert!(
        !report.conditional_groups.is_empty(),
        "expected at least one conditionally-separated group"
    );
}

#[test]
fn separation_detector_runs_quickly_on_corpus_sizes() {
    // Acceptance for bd-01KQ8FS7HK6TX2TMX0J0XFGYFD: LP-based detection
    // must run "well under fitting time" at corpus sizes. Rather than
    // benchmarking against an actual fit (which is timing-flaky in CI),
    // we assert a generous wall-clock budget — separation detection on
    // the corpus's largest Bernoulli fixture should comfortably complete
    // in under 250ms even on a debug build.
    let spec = fixtures::fe_separation_extreme_slope();
    let start = std::time::Instant::now();
    let _report = detect_separation(&spec);
    let elapsed = start.elapsed();
    assert!(
        elapsed < std::time::Duration::from_millis(250),
        "detect_separation took {elapsed:?} (budget: 250ms)"
    );
}

#[test]
fn separation_extreme_prevalence_certifies_both_tiers() {
    // Extreme prevalence (shift=-15) drives realised y to all-zero with
    // overwhelming probability, which trips both tiers of the
    // separation detector: FE (complete — `signs_i · x_i' β > 0` is
    // achievable with β = (-1, 0)) and conditional (every group has
    // all-zero outcomes).
    let spec = fixtures::separation_extreme_prevalence();
    let cert = certify(&spec);
    let n_primary = spec.group_sizes.len();
    assert!(
        matches!(
            cert.structural_issue,
            Some(StructuralIssue::Separation {
                kind: SeparationKind::Both {
                    fe_kind: FeSeparationKind::Complete,
                    n_groups,
                }
            }) if n_groups == n_primary
        ),
        "expected StructuralIssue::Separation::Both {{ Complete, {n_primary} }}, got {:?}",
        cert.structural_issue
    );

    // Round-trip the rich report and confirm it agrees with the cert.
    let report = detect_separation(&spec);
    assert_eq!(report.fe_kind, Some(FeSeparationKind::Complete));
    assert_eq!(report.conditional_groups.len(), n_primary);
    assert!(report.hyperplane_direction.is_some());
}

#[test]
fn separation_expected_statuses_admits_converged_penalised() {
    // Acceptance criterion (bd-01KQ8FSHVBDS85KS0KM4867VBK): a
    // separation-stratum fixture's expected status set must include
    // `ConvergedPenalised`, alongside the refusal options.
    let spec = fixtures::separation_extreme_prevalence();
    let cert = certify(&spec);
    let exp = expected_statuses(&cert);
    assert!(
        exp.contains(FitStatus::ConvergedPenalised),
        "separation set should admit ConvergedPenalised, got {:?}",
        exp.allowed
    );
    assert!(
        exp.contains(FitStatus::NotIdentifiable),
        "separation set should still admit NotIdentifiable (refusal path), got {:?}",
        exp.allowed
    );
    assert!(
        exp.contains(FitStatus::NotOptimized),
        "separation set should still admit NotOptimized, got {:?}",
        exp.allowed
    );
    // Critical contract claim: a separated logistic cannot honestly land
    // on a standard Converged* status. Asserting these are *excluded*
    // guards against accidentally widening the separation branch back to
    // the generic structural-issue set.
    assert!(
        !exp.contains(FitStatus::ConvergedInterior),
        "separation set must not admit ConvergedInterior, got {:?}",
        exp.allowed
    );
    assert!(
        !exp.contains(FitStatus::ConvergedBoundary),
        "separation set must not admit ConvergedBoundary, got {:?}",
        exp.allowed
    );
    assert!(
        !exp.contains(FitStatus::ConvergedReducedRank),
        "separation set must not admit ConvergedReducedRank, got {:?}",
        exp.allowed
    );
}

#[test]
fn crossed_sparse_connected_does_not_trigger_disconnected_crossings() {
    let spec = fixtures::crossed_sparse_connected();
    let cert = certify(&spec);
    let summary = cert
        .crossed_summary
        .as_ref()
        .expect("crossed_summary must be populated for crossed specs");
    // Sanity: the chosen seed produced a connected graph (excluding any
    // orphan levels). If a seed change makes this fail, swap the seed
    // and document why.
    assert_eq!(
        summary.n_components, 1,
        "expected one component, got {} (orphans: primary={:?}, secondary={:?})",
        summary.n_components, summary.primary_orphans, summary.secondary_orphans
    );
    assert!(
        !matches!(
            cert.structural_issue,
            Some(StructuralIssue::DisconnectedCrossings { .. })
        ),
        "connected sparse design must not be flagged as disconnected"
    );
}

// --- bd-01KQ8FVHD7WCN88RYJX1Y81NEP -------------------------------------
// GLMM stratum hygiene: every corpus fixture must exercise at most one
// of {Separation, AdaptiveGaussHermite, Overdispersion}. Combining
// these axes obscures which subsystem (PIRLS, AGQ, link, dispersion)
// is at fault when a regression hits. See
// `tests/fixtures/pathology_corpus/README.md` for the rationale.

fn all_fixture_specs() -> Vec<GeneratorSpec> {
    vec![
        fixtures::easy(),
        fixtures::boundary(),
        fixtures::reduced_rank(),
        fixtures::refusal(),
        fixtures::imbalance(),
        fixtures::scale_mismatch_fixture(),
        fixtures::collinear_fe_perfect(),
        fixtures::extreme_prevalence_low(),
        fixtures::singletons_via_transform(),
        fixtures::random_slope_singletons(),
        fixtures::crossed_block_diagonal(),
        fixtures::weakly_identified(),
        fixtures::crossed_sparse_connected(),
        fixtures::fe_separation_extreme_slope(),
        fixtures::conditional_separation_rare_events(),
        fixtures::separation_extreme_prevalence(),
    ]
}

#[test]
fn every_corpus_fixture_satisfies_single_axis_policy() {
    let mut violations = Vec::new();
    for spec in all_fixture_specs() {
        let cert = certify(&spec);
        if let Err(message) = lint_single_axis(&spec, &cert) {
            violations.push(message);
        }
    }
    assert!(
        violations.is_empty(),
        "single-axis policy violations:\n  {}",
        violations.join("\n  ")
    );
}

#[test]
fn separation_fixtures_carry_only_separation_axis_among_glmm_axes() {
    // Sanity: the three Bernoulli separation fixtures must tag exactly
    // PathologyAxis::Separation among the GLMM-policy axes (AGQ and
    // Overdispersion must be absent). LinkNonlinearity and
    // IdentifiabilityCore tags are exempt from this assertion since
    // the lint deliberately ignores them.
    let separation_specs = [
        fixtures::fe_separation_extreme_slope(),
        fixtures::conditional_separation_rare_events(),
        fixtures::separation_extreme_prevalence(),
    ];
    for spec in &separation_specs {
        let cert = certify(spec);
        let axes = inferred_axes(spec, &cert);
        assert!(
            axes.contains(&PathologyAxis::Separation),
            "fixture '{}' should tag PathologyAxis::Separation; got {:?}",
            spec.label,
            axes
        );
        assert!(
            !axes.contains(&PathologyAxis::AdaptiveGaussHermite),
            "fixture '{}' must not combine separation with AGQ; got {:?}",
            spec.label,
            axes
        );
        assert!(
            !axes.contains(&PathologyAxis::Overdispersion),
            "fixture '{}' must not combine separation with Overdispersion; got {:?}",
            spec.label,
            axes
        );
    }
}

#[test]
fn lint_rejects_constructed_multi_axis_spec() {
    // Deliberately construct a Bernoulli fixture that combines
    // separation (extreme intercept shift) with AGQ stress
    // (singleton groups) to confirm the lint actually fires when its
    // invariant is broken. The constructed spec is local to this
    // test — it does NOT live in `fixtures::` and so the
    // `every_corpus_fixture_satisfies_single_axis_policy` test stays
    // green.
    let mut spec = GeneratorSpec::lmm(
        "synthetic_multi_axis_violation",
        99,
        vec![1; 30], // singleton groups → AGQ stress
        vec![0.0, 0.5],
        true,
        0,
        nalgebra::dmatrix![1.0],
    );
    spec.family = Family::Bernoulli;
    spec.link = LinkFunction::Logit;
    spec.residual_sd = 0.0;
    extreme_prevalence(&mut spec, -15.0); // separation

    let cert = certify(&spec);
    let axes = inferred_axes(&spec, &cert);
    assert!(
        axes.contains(&PathologyAxis::Separation),
        "synthetic spec should still tag Separation; got {axes:?}"
    );
    assert!(
        axes.contains(&PathologyAxis::AdaptiveGaussHermite),
        "synthetic spec should tag AGQ from singleton groups; got {axes:?}"
    );

    let result = lint_single_axis(&spec, &cert);
    assert!(
        result.is_err(),
        "lint must reject Separation + AGQ combination; got Ok"
    );
    let message = result.unwrap_err();
    assert!(
        message.contains("Separation") && message.contains("AdaptiveGaussHermite"),
        "lint error must name both offending axes; got {message}"
    );
}

#[test]
fn easy_fixture_tags_no_glmm_policy_axes() {
    // Counter-example: the canonical `easy` fixture is Gaussian and
    // structurally identified, so it must tag *no* GLMM-policy axes
    // (and at most IdentifiabilityCore among LMM-only tags is irrelevant
    // here — easy doesn't trip even that).
    let spec = fixtures::easy();
    let cert = certify(&spec);
    let axes = inferred_axes(&spec, &cert);
    let glmm_axes: Vec<_> = axes
        .iter()
        .copied()
        .filter(|a| {
            matches!(
                a,
                PathologyAxis::Separation
                    | PathologyAxis::AdaptiveGaussHermite
                    | PathologyAxis::Overdispersion
            )
        })
        .collect();
    assert!(
        glmm_axes.is_empty(),
        "easy fixture should tag zero GLMM-policy axes; got {glmm_axes:?}"
    );
}
