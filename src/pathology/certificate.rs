//! Identifiability certificate: pure-linear-algebra classifier of a
//! [`GeneratorSpec`]'s relation to the contract's fit-status manifold.
//!
//! The certificate must remain *engine-free*. It looks at the spec — design
//! sizes, true random-effects covariance, group structure — and produces
//! the [`Certificate`] record. [`expected_statuses`] then converts that
//! record into the *set* of [`FitStatus`] values any conformant fit engine
//! must produce.
//!
//! Why a set rather than a single value? Truth on a boundary
//! (e.g. true `σ²_slope = 0`) can legitimately surface as either
//! `ConvergedBoundary` (engine drove the parameter to zero) or
//! `ConvergedInterior` (engine landed slightly off zero on noise). Asserting
//! membership in the acceptable set keeps the harness robust to optimizer
//! noise near boundaries while still catching real regressions: a `Refusal`
//! that becomes a `Converged` status, or vice versa, is always a failure.

use nalgebra::{DMatrix, SymmetricEigen};

use super::separation::{detect_separation, FeSeparationKind};
use super::spec::GeneratorSpec;
use crate::compiler::{CompiledModelArtifact, EffectiveRankStatus, FitStatus};
use crate::error::{LinAlgError, MixedModelError};
use crate::model::{Family, LinearMixedModel};

/// Numerical zero threshold for variance components in truth.
const ZERO_VARIANCE_TOL: f64 = 1e-10;
/// Distance from |ρ|=1 below which we count a correlation as on the boundary.
const UNIT_CORRELATION_TOL: f64 = 1e-6;
/// Multiplicative threshold on the trace for counting an eigenvalue as
/// "effectively zero" when computing rank(Σ_truth).
const RANK_REL_TOL: f64 = 1e-12;
/// Default cutoff below which the dimensionless weak-identification index
/// flags a design as weakly identified. Calibrated empirically against the
/// pathology corpus (see `tests/fixtures/pathology_corpus/calibration.md`);
/// designs sitting at or below the cutoff have `expected_statuses` widened
/// to admit `ConvergedReducedRank` alongside the usual converged set.
pub const WEAK_ID_THRESHOLD: f64 = 10.0;

/// Current contract version for pathology-corpus fixture expectations.
///
/// Bump this only with a deliberate fixture migration pass that re-evaluates
/// each fixture's expected [`FitStatus`] set and records the rationale in the
/// contract version log.
pub const PATHOLOGY_CORPUS_CONTRACT_VERSION: &str = "v0.3";

/// Analytically-derived classification of a [`GeneratorSpec`]'s
/// identifiability.
///
/// Computed from linear algebra on the *spec*, never from a fit outcome.
/// Used by [`expected_statuses`] to derive the acceptable contract status
/// set for the corresponding fit.
#[derive(Debug, Clone, PartialEq)]
pub struct Certificate {
    /// Effective rank of Σ_truth, eigenvalues thresholded relative to trace.
    pub re_rank_truth: usize,
    /// Requested random-effects rank q = re_intercept + n_re_slopes.
    pub re_rank_requested: usize,
    /// Eigenvalues of Σ_truth, sorted descending.
    pub re_cov_eigvals: Vec<f64>,
    /// Eigenvalues of the (correlation-form) expected Fisher information at
    /// truth, sorted descending. Used to compute [`Self::weak_id_score`].
    /// Reducing the Fisher information to its correlation form (diagonal
    /// scaled to ones) makes the spectrum invariant to per-axis predictor
    /// rescaling, so the corresponding weak-identification index is
    /// dimensionless. See [`fisher_correlation_eigvals`] for construction.
    pub fisher_eigvals: Vec<f64>,
    /// Dimensionless weak-identification index:
    /// `n * lambda_min(I) / trace(I)`, where `I` is the expected Fisher
    /// information at truth in correlation form.
    /// Invariant to (i) uniform rescaling of the response, (ii) per-axis
    /// rescaling of any fixed-effect predictor. Lower values indicate a
    /// design where some parameter direction is weakly identified;
    /// `weak_id_score < WEAK_ID_THRESHOLD` flips [`Self::weak_identification`]
    /// on and widens [`expected_statuses`] to admit `ConvergedReducedRank`.
    pub weak_id_score: f64,
    /// Threshold used to classify the design as weakly identified.
    /// Defaults to [`WEAK_ID_THRESHOLD`]; carried on the certificate so
    /// downstream tooling can introspect the cutoff that produced
    /// [`Self::weak_identification`].
    pub weak_id_threshold: f64,
    /// `true` when `weak_id_score < weak_id_threshold`. The flag is
    /// information-only: structural issues, boundary directions, and
    /// reduced-rank truth still take precedence in [`expected_statuses`].
    /// When this flag is `true` and the design is otherwise well-posed,
    /// the acceptable status set widens to include `ConvergedReducedRank`
    /// in recognition of the fact that weakly-identified directions can
    /// legitimately collapse during fitting.
    pub weak_identification: bool,
    /// Effective rank of the population fixed-effect predictor covariance
    /// matrix `S = diag(scales) · corr · diag(scales)`. With identity
    /// correlation this equals [`Self::fe_rank_requested`]; the
    /// `collinear_fe` transform can drive it lower.
    pub fe_rank_truth: usize,
    /// Number of fixed-effect predictors (excluding intercept).
    pub fe_rank_requested: usize,
    /// Boundary directions present in the *truth*. Empty for an "easy"
    /// design.
    pub boundary_directions: Vec<BoundaryKind>,
    /// Total observations across all groups.
    pub n_total: usize,
    /// Number of free parameters (rough: p + q(q+1)/2 + 1 for residual).
    pub n_params_estimated: usize,
    /// Smallest group cardinality.
    pub min_group_size: usize,
    /// Largest group cardinality.
    pub max_group_size: usize,
    /// Set if the design is structurally unidentifiable.
    pub structural_issue: Option<StructuralIssue>,
    /// Family label for diagnostic strings (informational only).
    pub family_label: String,
    /// Spec label, propagated for diagnostic strings.
    pub label: String,
    /// Summary of the crossed-RE bipartite cell graph, when the spec is
    /// crossed. `None` for single-grouping-factor designs.
    pub crossed_summary: Option<CrossedSummary>,
}

/// Pure-spec summary of a crossed `(primary × secondary)` design's cell
/// pattern.
///
/// Computed from [`super::spec::GeneratorSpec::crossed_cells`] alone — no
/// data draw and no engine call. Used by [`expected_statuses`] to flag
/// designs whose bipartite cell graph is disconnected, where between-
/// component random-effect contrasts are unidentifiable from data.
#[derive(Debug, Clone, PartialEq)]
pub struct CrossedSummary {
    /// Number of primary grouping levels (== `spec.group_sizes.len()`).
    pub n_primary: usize,
    /// Number of secondary grouping levels (== `crossed.n_levels`).
    pub n_secondary: usize,
    /// Number of populated `(primary, secondary)` cells.
    pub n_cells: usize,
    /// Number of connected components in the bipartite cell graph
    /// considering only levels that appear in at least one cell.
    pub n_components: usize,
    /// Primary levels that appear in zero cells (no observations).
    pub primary_orphans: Vec<usize>,
    /// Secondary levels that appear in zero cells (no observations).
    pub secondary_orphans: Vec<usize>,
}

/// Kinds of contract boundary the *truth* sits on.
#[derive(Debug, Clone, PartialEq)]
pub enum BoundaryKind {
    /// Diagonal element of Σ_truth is numerically zero — variance component
    /// on its lower bound.
    ZeroVariance { re_index: usize },
    /// Off-diagonal correlation is ±1 to within tolerance — correlation on
    /// the unit-disk boundary.
    UnitCorrelation { i: usize, j: usize, sign: i8 },
}

/// Classes of structural unidentifiability that the certificate detects
/// before any fitting attempt.
#[derive(Debug, Clone, PartialEq)]
pub enum StructuralIssue {
    /// A random-slope structure was requested but a majority of groups have
    /// fewer observations than `re_dim`, so within-group slope variation is
    /// missing. The likelihood is flat along the corresponding random-slope
    /// variance direction.
    SingletonsWithSlope {
        groups_too_small: usize,
        re_dim: usize,
    },
    /// Σ_truth has rank strictly less than its requested dimension *and*
    /// the data is drawn so the random-effect realisations live on the
    /// degenerate subspace exactly. Typically combined with a unit
    /// correlation in truth.
    DegenerateRandomEffectsCovariance,
    /// Two or more fixed-effect predictors have an exactly-degenerate
    /// population correlation (`rho = ±1`), so the corresponding columns
    /// of X are collinear in expectation. The fixed-effects design is
    /// rank-deficient regardless of seed.
    CollinearFixedEffects { rank: usize, requested: usize },
    /// The crossed `(primary × secondary)` cell graph has more than one
    /// connected component. Levels in different components never co-occur,
    /// so between-component random-effect contrasts are unidentifiable
    /// from data even though within-component fits remain well-posed.
    /// A conformant engine may still converge cleanly inside each
    /// component — see [`expected_statuses`] for the acceptable status
    /// set under this issue.
    DisconnectedCrossings { n_components: usize },
    /// The likelihood is unbounded because the design exhibits
    /// (fixed-effect or conditional) separation. The MLE does not exist;
    /// the contract response is either `NotIdentifiable` (the engine
    /// refuses) or `ConvergedPenalised` (the engine applies a Firth-style
    /// penalty and returns a well-defined penalised estimate). See the
    /// Refusal-vs-ConvergedPenalised decision tree in
    /// `docs/mixed_model_compiler_inference_contract.md`.
    ///
    /// Detection runs `super::separation::detect_separation(spec)`,
    /// which generates a representative draw via `spec.seed`, runs the
    /// Konis (2007) LP for fixed-effect separation, and scans for
    /// conditionally-separated groups (all-zero/all-one outcomes). The
    /// `kind` payload encodes which tier(s) fired; the rich
    /// [`super::separation::SeparationReport`] (hyperplane direction,
    /// exact group indices) is recomputed by callers that need it.
    Separation { kind: SeparationKind },
    /// The spec itself is malformed (e.g. `re_cov_truth` is not `q×q` for
    /// `q = re_dim()`, or `n_re_slopes` exceeds the number of fixed-effect
    /// predictors). This is an invalid input, not a property of a valid
    /// design, but [`certify`] is contractually total and pure: it reports
    /// the malformation here instead of panicking or indexing out of bounds.
    MalformedSpec { detail: String },
}

/// Two-tier separation classification carried inside
/// [`StructuralIssue::Separation`].
///
/// Copy-shaped so the structural-issue enum keeps a small payload; the
/// rich [`super::separation::SeparationReport`] (hyperplane direction,
/// exact group indices) sits one call away via
/// [`super::separation::detect_separation`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeparationKind {
    /// Linear separation in the fixed-effects design alone (Konis 2007
    /// trichotomy). No grouping level is conditionally separated.
    FixedEffect(FeSeparationKind),
    /// Conditional separation within `n_groups` grouping levels (each
    /// has all-zero or all-one outcomes). The fixed-effects design
    /// admits no separating hyperplane.
    Conditional { n_groups: usize },
    /// Both tiers fired: FE separation *and* `n_groups` conditionally-
    /// separated grouping levels. The most pathological combination,
    /// usually produced by extreme-prevalence Bernoulli specs.
    Both {
        fe_kind: FeSeparationKind,
        n_groups: usize,
    },
}

/// The set of [`FitStatus`] values any conformant fit engine must produce
/// for a design matching its [`Certificate`].
#[derive(Debug, Clone)]
pub struct ExpectedStatusSet {
    pub allowed: Vec<FitStatus>,
    pub rationale: String,
}

impl ExpectedStatusSet {
    pub fn contains(&self, status: FitStatus) -> bool {
        self.allowed.contains(&status)
    }
}

/// Compute the identifiability certificate for a generator spec.
///
/// **Pure linear algebra.** This function must not call any fitting engine,
/// must not draw data, and must not depend on `seed`. Its output depends
/// solely on the design (group sizes, q, family) and the truth covariance.
pub fn certify(spec: &GeneratorSpec) -> Certificate {
    let q = spec.re_dim();
    let sigma = &spec.re_cov_truth;

    // Contract: certify is *total* and *pure linear algebra* — it must not
    // panic. A malformed spec (truth covariance not q×q, or more random
    // slopes than fixed-effect predictors) would otherwise index out of
    // bounds below (and panic again via detect_separation → generate). Detect
    // it up front and return a well-formed certificate flagged
    // `MalformedSpec`, without touching the mis-sized matrix.
    let n_predictors = spec.n_fe_predictors();
    let shape_problem = if sigma.nrows() != q || sigma.ncols() != q {
        Some(format!(
            "re_cov_truth is {}×{} but re_dim() = {q}",
            sigma.nrows(),
            sigma.ncols()
        ))
    } else if spec.n_re_slopes > n_predictors {
        Some(format!(
            "spec requests {} random slopes but only {n_predictors} \
             fixed-effect predictors exist",
            spec.n_re_slopes
        ))
    } else {
        None
    };
    if let Some(detail) = shape_problem {
        return Certificate {
            re_rank_truth: 0,
            re_rank_requested: q,
            re_cov_eigvals: Vec::new(),
            fisher_eigvals: Vec::new(),
            weak_id_score: f64::NAN,
            weak_id_threshold: WEAK_ID_THRESHOLD,
            weak_identification: false,
            fe_rank_truth: 0,
            fe_rank_requested: n_predictors,
            boundary_directions: Vec::new(),
            n_total: spec.n_total(),
            n_params_estimated: 0,
            min_group_size: spec.group_sizes.iter().copied().min().unwrap_or(0),
            max_group_size: spec.group_sizes.iter().copied().max().unwrap_or(0),
            structural_issue: Some(StructuralIssue::MalformedSpec { detail }),
            family_label: format!("{:?}", spec.family),
            label: spec.label.clone(),
            crossed_summary: None,
        };
    }

    let (eigvals, _) = sorted_eigvals(sigma);
    let re_rank_truth = effective_rank(&eigvals);

    let mut boundary = Vec::new();
    for i in 0..q {
        if sigma[(i, i)].abs() < ZERO_VARIANCE_TOL {
            boundary.push(BoundaryKind::ZeroVariance { re_index: i });
        }
    }
    for i in 0..q {
        for j in (i + 1)..q {
            let denom = (sigma[(i, i)] * sigma[(j, j)]).sqrt();
            if denom > ZERO_VARIANCE_TOL {
                let rho = sigma[(i, j)] / denom;
                if (rho.abs() - 1.0).abs() < UNIT_CORRELATION_TOL {
                    boundary.push(BoundaryKind::UnitCorrelation {
                        i,
                        j,
                        sign: if rho > 0.0 { 1 } else { -1 },
                    });
                }
            }
        }
    }

    let min_g = spec.group_sizes.iter().copied().min().unwrap_or(0);
    let max_g = spec.group_sizes.iter().copied().max().unwrap_or(0);
    let n_total = spec.n_total();

    let fe_rank_requested = spec.n_fe_predictors();
    let fe_rank_truth = compute_fe_rank(spec, fe_rank_requested);

    let crossed_summary = spec.crossed.as_ref().map(|c| {
        let cells = spec.crossed_cells().unwrap_or_default();
        summarise_crossing(spec.group_sizes.len(), c.n_levels, &cells)
    });

    let structural_issue = detect_structural_issue(
        spec,
        q,
        fe_rank_truth,
        fe_rank_requested,
        crossed_summary.as_ref(),
    );

    let p = spec.fe_truth.len();
    let n_params_estimated = p + q * (q + 1) / 2 + 1;

    let fisher_eigvals = fisher_correlation_eigvals(spec);
    let weak_id_score = compute_weak_id_score(n_total, &fisher_eigvals);
    let weak_id_threshold = WEAK_ID_THRESHOLD;
    let weak_identification = weak_id_score.is_finite() && weak_id_score < weak_id_threshold;

    Certificate {
        re_rank_truth,
        re_rank_requested: q,
        re_cov_eigvals: eigvals,
        fisher_eigvals,
        weak_id_score,
        weak_id_threshold,
        weak_identification,
        fe_rank_truth,
        fe_rank_requested,
        boundary_directions: boundary,
        n_total,
        n_params_estimated,
        min_group_size: min_g,
        max_group_size: max_g,
        structural_issue,
        family_label: format!("{:?}", spec.family),
        label: spec.label.clone(),
        crossed_summary,
    }
}

/// Spectrum of the expected Fisher information at truth, reduced to
/// correlation form so the result is dimensionless.
///
/// For an LMM with truth `(β, Σ, σ²)` and i.i.d. predictors drawn from
/// `N(0, S_X)` with `S_X = D · C · D` (D = diag(scales), C = predictor
/// correlation matrix), the population fixed-effects Fisher information
/// is proportional to `S_X` up to a `1/σ²` and a factor that depends on
/// the random-effects projector. Its eigenvalues inherit the scale of D
/// — multiplying any predictor by 10 multiplies the corresponding row
/// and column of `S_X` by 10. Reducing to correlation form
/// (`I_corr = D^{-1/2} · S_X · D^{-1/2} = C`) cancels D and yields a
/// matrix whose eigenvalues are scale-invariant. This is the matrix on
/// which we compute `lambda_min` and `trace` for the weak-identification
/// index.
///
/// The intercept column is included as an extra all-ones row of `S_X`,
/// uncorrelated with the slopes by construction. We treat it as a
/// fully-identified direction by always emitting an extra eigenvalue
/// equal to 1.0 (its diagonal in correlation form), so even an
/// intercept-only design ("no slopes") has a well-defined unit
/// trace and admits a finite score.
///
/// Returns the eigenvalues of `I_corr` sorted in **descending** order.
pub fn fisher_correlation_eigvals(spec: &GeneratorSpec) -> Vec<f64> {
    let n_pred = spec.n_fe_predictors();
    if n_pred == 0 {
        return vec![1.0];
    }
    let corr = if spec.fe_corr_matrix.nrows() == n_pred && spec.fe_corr_matrix.ncols() == n_pred {
        spec.fe_corr_matrix.clone()
    } else {
        DMatrix::identity(n_pred, n_pred)
    };
    let (mut eigs, _) = sorted_eigvals(&corr);
    // Add the intercept slot at unity (already sorted-descending: 1.0 ≤ 1.0
    // for an identity-correlation slope block, so push and re-sort).
    eigs.push(1.0);
    eigs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    eigs
}

/// Compute the dimensionless weak-identification index from the
/// correlation-form Fisher spectrum.
///
/// `score = n * lambda_min(I_corr) / trace(I_corr)`. Returns
/// [`f64::INFINITY`] when the spectrum is empty (no parameters) so
/// callers can treat "no parameters" as trivially identified.
fn compute_weak_id_score(n: usize, eigs: &[f64]) -> f64 {
    if eigs.is_empty() {
        return f64::INFINITY;
    }
    let trace: f64 = eigs.iter().sum();
    if !trace.is_finite() || trace.abs() < 1e-15 {
        return f64::INFINITY;
    }
    let lambda_min = eigs
        .iter()
        .copied()
        .fold(f64::INFINITY, |acc, v| acc.min(v))
        .max(0.0);
    (n as f64) * lambda_min / trace
}

/// Build a [`CrossedSummary`] from the materialised cell list.
///
/// Computes connected components on the bipartite graph whose edges are
/// the cells, treating primary nodes as `0..n_primary` and secondary nodes
/// as `n_primary..n_primary + n_secondary`. Orphan levels (zero cells) are
/// excluded from component counting and reported separately.
fn summarise_crossing(
    n_primary: usize,
    n_secondary: usize,
    cells: &[(usize, usize)],
) -> CrossedSummary {
    let mut primary_present = vec![false; n_primary];
    let mut secondary_present = vec![false; n_secondary];
    for &(i, j) in cells {
        if i < n_primary {
            primary_present[i] = true;
        }
        if j < n_secondary {
            secondary_present[j] = true;
        }
    }

    let total = n_primary + n_secondary;
    let mut parent: Vec<usize> = (0..total).collect();
    fn find(parent: &mut [usize], x: usize) -> usize {
        let mut root = x;
        while parent[root] != root {
            root = parent[root];
        }
        let mut cur = x;
        while parent[cur] != root {
            let next = parent[cur];
            parent[cur] = root;
            cur = next;
        }
        root
    }
    for &(i, j) in cells {
        if i >= n_primary || j >= n_secondary {
            continue;
        }
        let a = find(&mut parent, i);
        let b = find(&mut parent, n_primary + j);
        if a != b {
            parent[a] = b;
        }
    }
    let mut roots = std::collections::BTreeSet::new();
    for k in 0..n_primary {
        if primary_present[k] {
            roots.insert(find(&mut parent, k));
        }
    }
    for k in 0..n_secondary {
        if secondary_present[k] {
            roots.insert(find(&mut parent, n_primary + k));
        }
    }

    let primary_orphans = (0..n_primary).filter(|i| !primary_present[*i]).collect();
    let secondary_orphans = (0..n_secondary)
        .filter(|j| !secondary_present[*j])
        .collect();

    CrossedSummary {
        n_primary,
        n_secondary,
        n_cells: cells.len(),
        n_components: roots.len(),
        primary_orphans,
        secondary_orphans,
    }
}

fn compute_fe_rank(spec: &GeneratorSpec, requested: usize) -> usize {
    if requested == 0 {
        return 0;
    }
    if spec.fe_corr_matrix.nrows() != requested || spec.fe_corr_matrix.ncols() != requested {
        return requested;
    }
    let (eigvals, _) = sorted_eigvals(&spec.fe_corr_matrix);
    effective_rank(&eigvals)
}

fn detect_structural_issue(
    spec: &GeneratorSpec,
    q: usize,
    fe_rank_truth: usize,
    fe_rank_requested: usize,
    crossed_summary: Option<&CrossedSummary>,
) -> Option<StructuralIssue> {
    if fe_rank_truth < fe_rank_requested {
        return Some(StructuralIssue::CollinearFixedEffects {
            rank: fe_rank_truth,
            requested: fe_rank_requested,
        });
    }
    if let Some(summary) = crossed_summary {
        if summary.n_components > 1 {
            return Some(StructuralIssue::DisconnectedCrossings {
                n_components: summary.n_components,
            });
        }
    }
    // Two-tier separation detection (bd-01KQ8FS7HK6TX2TMX0J0XFGYFD):
    // a binomial spec is run through the Konis (2007) LP plus a
    // per-group all-zero/all-one scan via `detect_separation`. Either
    // tier alone is enough to flag the design — the MLE does not exist
    // and the contract response is Refusal or ConvergedPenalised. The
    // call is intentionally only made for Bernoulli specs (other
    // families have no separation pathology) so non-binomial fixtures
    // remain pure-spec, seed-independent classifications.
    if matches!(spec.family, Family::Bernoulli) {
        let report = detect_separation(spec);
        let kind = match (report.fe_kind, report.conditional_groups.is_empty()) {
            (Some(fe_kind), true) => Some(SeparationKind::FixedEffect(fe_kind)),
            (None, false) => Some(SeparationKind::Conditional {
                n_groups: report.conditional_groups.len(),
            }),
            (Some(fe_kind), false) => Some(SeparationKind::Both {
                fe_kind,
                n_groups: report.conditional_groups.len(),
            }),
            (None, true) => None,
        };
        if let Some(kind) = kind {
            return Some(StructuralIssue::Separation { kind });
        }
    }
    if spec.n_re_slopes == 0 {
        return None;
    }
    // For crossed designs the "within-group" observation count is the
    // count of cells whose primary index equals each group, not
    // `group_sizes` (which is a stub for crossed specs). Compute
    // per-primary cell counts when crossed; otherwise read group_sizes.
    let per_group: Vec<usize> = if spec.crossed.is_some() {
        let mut counts = vec![0usize; spec.group_sizes.len()];
        if let Some(cells) = spec.crossed_cells() {
            for (i, _) in cells {
                if i < counts.len() {
                    counts[i] += 1;
                }
            }
        }
        counts
    } else {
        spec.group_sizes.clone()
    };
    let too_small = per_group.iter().filter(|&&n| n < q.max(2)).count();
    let majority = per_group.len().div_ceil(2);
    if too_small >= majority && !per_group.is_empty() {
        return Some(StructuralIssue::SingletonsWithSlope {
            groups_too_small: too_small,
            re_dim: q,
        });
    }
    None
}

/// Map a certificate to its acceptable contract status set.
///
/// **The acceptable set is a current-engine-conformance set, not a pure
/// contract claim.** The strict contract for a rank-deficient truth is
/// `{ConvergedReducedRank, ConvergedBoundary}`, but today's optimizer
/// certificate path does not yet promote `ConvergedInterior` to
/// `ConvergedReducedRank` on every rank-deficient truth (the realised
/// MLE may sit at a positive variance even when truth places rank-1
/// mass on the manifold). The set therefore includes `ConvergedInterior`
/// for boundary and reduced-rank strata. As reduced-rank detection
/// improves, the set should narrow — that narrowing is the corpus-
/// versioning workflow tracked under `bd-01KQ8FV0FYKVT3CHZWXPYW1NPY`.
///
/// Precedence (top to bottom):
/// 1. Structural issue → `{NotIdentifiable, NotOptimized, ConvergedBoundary, ConvergedReducedRank}`
///    (engines may refuse, fail to optimise, pin a parameter to its
///    boundary, or collapse to a lower-rank fit — all legitimate
///    responses to a structurally under-identified design).
/// 2. Reduced-rank truth → `{ConvergedReducedRank, ConvergedBoundary, ConvergedInterior}`.
/// 3. Unit correlation → same as reduced-rank.
/// 4. Zero-variance component → `{ConvergedBoundary, ConvergedInterior}`.
/// 5. Otherwise (well-identified, well-conditioned) → `{ConvergedInterior}`.
pub fn expected_statuses(cert: &Certificate) -> ExpectedStatusSet {
    use FitStatus::*;

    if let Some(issue) = &cert.structural_issue {
        // Separation is its own branch: the MLE genuinely doesn't exist
        // (the likelihood is unbounded), so the contract response is
        // either Refusal (`NotIdentifiable`/`NotOptimized`) or a Firth-
        // style penalised fit (`ConvergedPenalised`). Standard converged
        // statuses must NOT be admitted here — a Bernoulli logistic with
        // structural separation cannot honestly land on `ConvergedInterior`.
        // See `docs/mixed_model_compiler_inference_contract.md` for the
        // Refusal-vs-ConvergedPenalised decision tree.
        if let StructuralIssue::Separation { kind } = issue {
            return ExpectedStatusSet {
                allowed: vec![NotIdentifiable, NotOptimized, ConvergedPenalised],
                rationale: format!(
                    "separation candidate ({:?}) in '{}': MLE undefined; \
                     refuse or apply a penalised fit (e.g. Firth)",
                    kind, cert.label
                ),
            };
        }
        // ConvergedInterior is included because the *expectation*-level
        // rank deficiency (e.g. ρ=1 collinearity) propagates through
        // sqrt_psd as an eigenvalue near ε rather than exactly zero, so
        // the realised X may be numerically rank-full even when truth
        // is rank-deficient. Today's engine then converges cleanly on
        // the noisy sample. Tightening this back to {NotIdentifiable,
        // NotOptimized, ConvergedBoundary, ConvergedReducedRank} is the
        // corpus-versioning workflow under bd-01KQ8FV0FYKVT3CHZWXPYW1NPY,
        // and depends on FE rank detection improving in the engine's
        // pivoted QR path.
        //
        // Disconnected crossings are a special case: per-component the
        // model is well-posed, so a conformant engine commonly returns
        // `ConvergedInterior`. The certificate still flags the issue so
        // downstream tooling (parity scoreboard, weak-id index) can
        // distinguish "fit succeeded inside each component" from "fit
        // succeeded on a fully connected design".
        let rationale = match issue {
            StructuralIssue::DisconnectedCrossings { n_components } => format!(
                "crossed design splits into {} disconnected components in '{}'",
                n_components, cert.label
            ),
            _ => format!("structural issue in '{}': {:?}", cert.label, issue),
        };
        return ExpectedStatusSet {
            allowed: vec![
                NotIdentifiable,
                NotOptimized,
                ConvergedBoundary,
                ConvergedReducedRank,
                ConvergedInterior,
            ],
            rationale,
        };
    }

    let has_unit_corr = cert
        .boundary_directions
        .iter()
        .any(|b| matches!(b, BoundaryKind::UnitCorrelation { .. }));
    let has_zero_var = cert
        .boundary_directions
        .iter()
        .any(|b| matches!(b, BoundaryKind::ZeroVariance { .. }));
    let reduced_rank = cert.re_rank_truth < cert.re_rank_requested;

    if reduced_rank {
        return ExpectedStatusSet {
            allowed: vec![ConvergedReducedRank, ConvergedBoundary, ConvergedInterior],
            rationale: format!(
                "truth Σ has rank {} < requested rank {}",
                cert.re_rank_truth, cert.re_rank_requested
            ),
        };
    }

    if has_unit_corr {
        return ExpectedStatusSet {
            allowed: vec![ConvergedReducedRank, ConvergedBoundary, ConvergedInterior],
            rationale: "unit correlation in truth (correlation parameter on boundary)".into(),
        };
    }

    if has_zero_var {
        return ExpectedStatusSet {
            allowed: vec![ConvergedBoundary, ConvergedInterior],
            rationale: "zero variance component in truth".into(),
        };
    }

    if cert.weak_identification {
        // Weakly-identified directions can plausibly collapse during
        // fitting even when the design is structurally identifiable: the
        // optimizer may land on a lower-rank fit, on the interior, or on
        // a parameter boundary depending on noise. The strict-interior
        // contract is too tight here, so we widen the set to include
        // `ConvergedReducedRank` alongside the usual converged statuses.
        return ExpectedStatusSet {
            allowed: vec![ConvergedInterior, ConvergedBoundary, ConvergedReducedRank],
            rationale: format!(
                "weak identification index {:.3e} below threshold {:.3e}",
                cert.weak_id_score, cert.weak_id_threshold
            ),
        };
    }

    ExpectedStatusSet {
        allowed: vec![ConvergedInterior],
        rationale: "design well-identified, far from contract boundaries".into(),
    }
}

/// Effective contract status for a fitted (or failed-to-fit) LMM.
///
/// Convenience wrapper around [`effective_status_from_artifact`]. Maps
/// constructor / fit errors via [`map_error_to_status`] and otherwise
/// delegates to the artifact-based helper, which is also reusable from
/// the GLMM path (`GeneralizedLinearMixedModel::compiler_artifact()`).
pub fn effective_status(fit_outcome: Result<&LinearMixedModel, &MixedModelError>) -> FitStatus {
    match fit_outcome {
        Err(err) => map_error_to_status(err),
        Ok(model) => effective_status_from_artifact(model.compiler_artifact()),
    }
}

/// Effective contract status from a [`CompiledModelArtifact`] alone, usable
/// for both LMM and GLMM (both expose `compiler_artifact()`).
///
/// Combines the optimizer certificate's status with the design audit's
/// reduced-rank summaries: `ConvergedInterior` / `ConvergedBoundary` is
/// promoted to `ConvergedReducedRank` when any random-effect term has
/// `EffectiveRankStatus::ReducedRank`.
pub fn effective_status_from_artifact(artifact: &CompiledModelArtifact) -> FitStatus {
    use FitStatus::*;
    let cert_status = match &artifact.optimizer_certificate {
        Some(c) => c.status,
        None => return NotAssessed,
    };
    if matches!(cert_status, ConvergedInterior | ConvergedBoundary) {
        let any_reduced = artifact
            .effective_covariance
            .iter()
            .any(|s| s.status == EffectiveRankStatus::ReducedRank);
        if any_reduced {
            return ConvergedReducedRank;
        }
    }
    cert_status
}

/// Map a [`MixedModelError`] to the most informative `FitStatus`.
///
/// Rank/PSD/separation-style errors → `NotIdentifiable`; everything else
/// (optimizer divergence, dimension mismatch, etc.) → `NotOptimized`.
pub fn map_error_to_status(err: &MixedModelError) -> FitStatus {
    use FitStatus::*;
    match err {
        MixedModelError::Singular(_)
        | MixedModelError::RankSaturatedFixedEffects { .. }
        | MixedModelError::PosDefException => NotIdentifiable,
        MixedModelError::LinAlg(LinAlgError::RankDeficient { .. })
        | MixedModelError::LinAlg(LinAlgError::Singular)
        | MixedModelError::LinAlg(LinAlgError::NotPositiveDefinite) => NotIdentifiable,
        MixedModelError::ConstantResponse | MixedModelError::NoRandomEffects => NotIdentifiable,
        _ => NotOptimized,
    }
}

fn sorted_eigvals(sigma: &DMatrix<f64>) -> (Vec<f64>, SymmetricEigen<f64, nalgebra::Dyn>) {
    let eig = SymmetricEigen::new(sigma.clone());
    let mut eigvals: Vec<f64> = eig.eigenvalues.iter().copied().collect();
    eigvals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    (eigvals, eig)
}

fn effective_rank(eigvals: &[f64]) -> usize {
    if eigvals.is_empty() {
        return 0;
    }
    let trace: f64 = eigvals.iter().sum();
    let abs_trace = trace.abs().max(1e-15);
    let threshold = (abs_trace * RANK_REL_TOL).max(1e-15);
    eigvals.iter().filter(|&&v| v > threshold).count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::dmatrix;

    #[test]
    fn certify_is_engine_free_for_easy_design() {
        let spec = GeneratorSpec::lmm(
            "easy",
            42,
            vec![6; 30],
            vec![1.0, 2.0],
            true,
            1,
            dmatrix![4.0, 0.5; 0.5, 1.0],
        );
        let cert = certify(&spec);
        assert_eq!(cert.re_rank_truth, 2);
        assert_eq!(cert.re_rank_requested, 2);
        assert!(cert.boundary_directions.is_empty());
        assert!(cert.structural_issue.is_none());
        let exp = expected_statuses(&cert);
        assert_eq!(exp.allowed, vec![FitStatus::ConvergedInterior]);
    }

    #[test]
    fn certify_does_not_panic_on_malformed_spec() {
        // Regression for audit 06·H2 / mote bd-01KRXCQ98S78SBNG0AHP22YB28:
        // certify is contractually total. A truth covariance whose size
        // disagrees with re_dim() must yield a MalformedSpec certificate,
        // not an out-of-bounds index panic.
        let mut spec = GeneratorSpec::lmm(
            "malformed_dim",
            42,
            vec![6; 10],
            vec![1.0, 2.0],
            true,
            1, // re_dim() == 2
            dmatrix![4.0, 0.5; 0.5, 1.0],
        );
        spec.re_cov_truth = DMatrix::identity(3, 3); // now 3×3 ≠ 2

        let cert = certify(&spec); // must not panic
        assert!(matches!(
            cert.structural_issue,
            Some(StructuralIssue::MalformedSpec { .. })
        ));
        // Downstream consumers still work and treat it as non-identifiable.
        let exp = expected_statuses(&cert);
        assert!(exp.allowed.contains(&FitStatus::NotIdentifiable));
    }

    #[test]
    fn generate_refuses_malformed_spec_instead_of_panicking() {
        // Regression for audit 06·H1: the certify → detect_separation →
        // generate path must return Err, not panic via an assert!.
        let mut spec = GeneratorSpec::lmm(
            "malformed_dim_gen",
            7,
            vec![5; 8],
            vec![1.0, 2.0],
            true,
            1,
            dmatrix![4.0, 0.5; 0.5, 1.0],
        );
        spec.re_cov_truth = DMatrix::identity(3, 3);

        let err = super::super::spec::generate(&spec).unwrap_err();
        assert!(matches!(err, MixedModelError::InvalidArgument(_)));
        // detect_separation already handles Err — it must not panic either.
        let _ = detect_separation(&spec);
    }

    #[test]
    fn certify_detects_zero_variance_boundary() {
        let spec = GeneratorSpec::lmm(
            "boundary_zero_slope",
            42,
            vec![6; 30],
            vec![1.0, 2.0],
            true,
            1,
            dmatrix![4.0, 0.0; 0.0, 0.0],
        );
        let cert = certify(&spec);
        assert!(cert
            .boundary_directions
            .iter()
            .any(|b| matches!(b, BoundaryKind::ZeroVariance { re_index: 1 })));
        let exp = expected_statuses(&cert);
        assert!(exp.contains(FitStatus::ConvergedBoundary));
    }

    #[test]
    fn certify_detects_reduced_rank_truth() {
        let mut spec = GeneratorSpec::lmm(
            "reduced_rank",
            42,
            vec![6; 30],
            vec![1.0, 2.0],
            true,
            1,
            dmatrix![4.0, 0.0; 0.0, 4.0],
        );
        // Push correlation to ~1 → rank 1
        super::super::transforms::near_singular_re(&mut spec, 1.0);
        let cert = certify(&spec);
        assert_eq!(cert.re_rank_truth, 1);
        assert_eq!(cert.re_rank_requested, 2);
        let exp = expected_statuses(&cert);
        assert!(exp.contains(FitStatus::ConvergedReducedRank));
    }

    #[test]
    fn certify_detects_collinear_fixed_effects() {
        let mut spec = GeneratorSpec::lmm(
            "collinear_fe",
            42,
            vec![6; 30],
            vec![1.0, 2.0, 3.0],
            true,
            1,
            dmatrix![4.0, 0.5; 0.5, 1.0],
        );
        // Force ρ=1 between x1 and x2
        super::super::transforms::collinear_fe(&mut spec, 0, 1, 1.0);
        let cert = certify(&spec);
        assert_eq!(cert.fe_rank_truth, 1);
        assert_eq!(cert.fe_rank_requested, 2);
        assert!(matches!(
            cert.structural_issue,
            Some(StructuralIssue::CollinearFixedEffects {
                rank: 1,
                requested: 2
            })
        ));
        let exp = expected_statuses(&cert);
        assert!(exp.contains(FitStatus::NotIdentifiable));
    }

    #[test]
    fn weak_id_score_is_invariant_under_uniform_predictor_rescale() {
        // Acceptance criterion (bd-01KQ8FT90WXSG9VSZQH30HZY9P): the
        // dimensionless weak-identification index must not move when a
        // predictor is rescaled by 1e3.
        let mut spec_a = GeneratorSpec::lmm(
            "weak_id_scale_a",
            42,
            vec![6; 30],
            vec![1.0, 2.0, 3.0],
            true,
            1,
            dmatrix![4.0, 0.5; 0.5, 1.0],
        );
        let mut spec_b = spec_a.clone();
        spec_b.label = "weak_id_scale_b".into();
        super::super::transforms::scale_mismatch(&mut spec_b, vec![1e3, 1e3]);

        let cert_a = certify(&spec_a);
        let cert_b = certify(&spec_b);
        assert!(
            (cert_a.weak_id_score - cert_b.weak_id_score).abs() < 1e-9,
            "weak_id_score changed under uniform 1e3 rescale: {} vs {}",
            cert_a.weak_id_score,
            cert_b.weak_id_score
        );

        // Per-axis rescaling (only one predictor scaled by 1e3) must
        // also be invariant — the index is reduced to correlation form.
        super::super::transforms::scale_mismatch(&mut spec_a, vec![1.0, 1e3]);
        let cert_c = certify(&spec_a);
        assert!(
            (cert_a.weak_id_score - cert_c.weak_id_score).abs() < 1e-9,
            "weak_id_score changed under per-axis rescale: {} vs {}",
            cert_a.weak_id_score,
            cert_c.weak_id_score
        );
    }

    #[test]
    fn weak_id_score_drops_with_collinear_predictors() {
        // A near-collinear predictor pair (rho close to 1) is the canonical
        // weakly-identified design even before structural FE-collinearity
        // refusal triggers. Confirm the index responds monotonically.
        let mut spec = GeneratorSpec::lmm(
            "weak_id_collinear",
            42,
            vec![6; 30],
            vec![1.0, 2.0, 3.0],
            true,
            1,
            dmatrix![4.0, 0.5; 0.5, 1.0],
        );
        let baseline = certify(&spec).weak_id_score;
        super::super::transforms::collinear_fe(&mut spec, 0, 1, 0.999);
        let near_singular = certify(&spec).weak_id_score;
        assert!(
            near_singular < baseline,
            "expected near-singular score {near_singular} < baseline {baseline}"
        );
        assert!(
            near_singular < WEAK_ID_THRESHOLD,
            "near-singular score should fall below the weak-id threshold {WEAK_ID_THRESHOLD}"
        );
    }

    #[test]
    fn weak_id_widens_status_set_on_otherwise_easy_design() {
        // A design that is structurally identified but sits below the
        // dimensionless threshold (e.g. tiny n, near-collinear predictors)
        // must widen `expected_statuses` to admit ConvergedReducedRank
        // alongside the usual converged statuses.
        let mut spec = GeneratorSpec::lmm(
            "weak_id_small_n",
            42,
            vec![1; 4],
            vec![0.5, 1.0, 1.0],
            true,
            0,
            dmatrix![1.0],
        );
        super::super::transforms::collinear_fe(&mut spec, 0, 1, 0.999);
        let cert = certify(&spec);
        assert!(cert.weak_identification);
        // No structural issue (collinear_fe at 0.999 stays under unit-tol).
        assert!(cert.structural_issue.is_none());
        let exp = expected_statuses(&cert);
        assert!(exp.contains(FitStatus::ConvergedReducedRank));
        assert!(exp.contains(FitStatus::ConvergedInterior));
    }

    #[test]
    fn certify_detects_singletons_with_slope() {
        let spec = GeneratorSpec::lmm(
            "refusal_singletons",
            42,
            vec![1; 6],
            vec![1.0, 2.0],
            true,
            1,
            dmatrix![4.0, 0.5; 0.5, 1.0],
        );
        let cert = certify(&spec);
        assert!(matches!(
            cert.structural_issue,
            Some(StructuralIssue::SingletonsWithSlope { .. })
        ));
        let exp = expected_statuses(&cert);
        assert!(exp.contains(FitStatus::NotIdentifiable));
    }
}
