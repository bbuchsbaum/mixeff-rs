//! Composable transforms over [`GeneratorSpec`].
//!
//! Each transform is `fn(&mut GeneratorSpec, ...)` so callers can stack
//! pathologies on top of a base spec:
//!
//! ```ignore
//! let mut spec = GeneratorSpec::lmm(...);
//! near_singular_re(&mut spec, 0.999);
//! scale_mismatch(&mut spec, vec![1.0, 1e6]);
//! ```
//!
//! ## Composability
//!
//! Most transforms commute when they touch disjoint fields
//! (`scale_mismatch` ∥ `extreme_prevalence`, `near_singular_re` ∥ `set_group_sizes`).
//! Transforms that touch the same field are last-writer-wins; in particular,
//! `set_group_sizes`, `singletons_with_slope`, and feeding `pareto_sizes`
//! into `set_group_sizes` are mutually exclusive in practice — the final
//! call wins.
//!
//! ## Out of scope (separate motes)
//!
//! - Two-tier separation detection coupled to [`extreme_prevalence`] is a
//!   certificate refinement tracked under
//!   `bd-01KQ8FS7HK6TX2TMX0J0XFGYFD`.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Pareto};

use crate::model::{Family, LinkFunction};

use super::spec::{CrossedSpec, GeneratorSpec};

/// Set the (0, 1) off-diagonal of `re_cov_truth` to a target Pearson
/// correlation `rho`, preserving the existing diagonal variances.
///
/// This is the canonical "near-singular random effects" pathology: as
/// `|rho|` approaches 1, the rank of Σ_truth drops by 1, and the
/// corresponding fit should be classified `ConvergedReducedRank` (or
/// `ConvergedBoundary` on the correlation parameter).
///
/// Requires `re_cov_truth` to be at least 2×2; otherwise this is a no-op.
pub fn near_singular_re(spec: &mut GeneratorSpec, rho: f64) {
    let q = spec.re_cov_truth.nrows();
    if q < 2 {
        return;
    }
    let s00 = spec.re_cov_truth[(0, 0)];
    let s11 = spec.re_cov_truth[(1, 1)];
    let off = rho * (s00 * s11).sqrt();
    spec.re_cov_truth[(0, 1)] = off;
    spec.re_cov_truth[(1, 0)] = off;
}

/// Replace the spec's group sizes wholesale.
///
/// Any prior call to [`singletons_with_slope`] or [`set_group_sizes`] is
/// overwritten — last writer wins.
pub fn set_group_sizes(spec: &mut GeneratorSpec, sizes: Vec<usize>) {
    spec.group_sizes = sizes;
}

/// Force one observation per group across `n_groups` groups.
///
/// When the spec has `n_re_slopes >= 1`, this design has *no within-group
/// variation in the random-slope predictor*, so the slope variance is
/// structurally unidentifiable — the certificate flags this via
/// [`crate::pathology::StructuralIssue::SingletonsWithSlope`].
pub fn singletons_with_slope(spec: &mut GeneratorSpec, n_groups: usize) {
    spec.group_sizes = vec![1; n_groups];
}

/// Promote the spec to `(Bernoulli, Logit)` and shift the linear predictor
/// to push prevalence toward 0 (negative shift) or 1 (positive shift).
///
/// `intercept_shift` adds to η at sample time *before* the inverse-logit.
/// A shift of ±5 typically yields prevalences below 1% or above 99% on
/// standard-normal predictors and is the canonical "rare events" pathology.
///
/// Note: separation detection is not yet wired into the certificate (see
/// `bd-01KQ8FS7HK6TX2TMX0J0XFGYFD`); a fixture using this transform will
/// currently fall through to the family-specific fit path without
/// preflight separation refusal.
pub fn extreme_prevalence(spec: &mut GeneratorSpec, intercept_shift: f64) {
    spec.family = Family::Bernoulli;
    spec.link = LinkFunction::Logit;
    spec.binary_intercept_shift = intercept_shift;
    spec.residual_sd = 0.0;
}

/// Set per-predictor scales. `scales[j]` multiplies the j-th fixed-effect
/// predictor at sample time, so wildly mismatched scales (e.g. `[1.0, 1e6]`)
/// produce poorly conditioned `X`.
///
/// If `scales.len() != n_fe_predictors` the supplied vector is used as-is
/// and missing entries default to 1.0 inside the generator. Excess entries
/// are simply ignored.
pub fn scale_mismatch(spec: &mut GeneratorSpec, scales: Vec<f64>) {
    spec.fe_scales = scales;
}

/// Set the population-level Pearson correlation between predictors `i` and
/// `j` in the fixed-effects design.
///
/// The diagonal of `fe_corr_matrix` stays at 1.0; only the (i, j) and
/// (j, i) off-diagonals are set to `rho`. With `rho = 1.0` the pair becomes
/// perfectly collinear and the certificate flags
/// [`crate::pathology::StructuralIssue::CollinearFixedEffects`].
///
/// No-op if `i == j`, if either index is out of range, or if the spec has
/// fewer than two fixed-effect predictors.
pub fn collinear_fe(spec: &mut GeneratorSpec, i: usize, j: usize, rho: f64) {
    let n = spec.n_fe_predictors();
    if i == j || i >= n || j >= n {
        return;
    }
    spec.fe_corr_matrix[(i, j)] = rho;
    spec.fe_corr_matrix[(j, i)] = rho;
}

/// Attach a crossed secondary grouping factor with random cell dropout.
///
/// Each cell of the full `n_primary × n_secondary` crossing is included
/// independently with probability `density`, using its own RNG seeded by
/// `seed` for reproducibility. The resulting cell list is stored on the
/// spec as [`CrossedSpec::cells`]; observations are emitted one-per-cell at
/// generation time.
///
/// `n_primary` is `spec.group_sizes.len()` (untouched). `group_sizes` is
/// retained verbatim — for crossed designs the per-primary observation
/// count is implicit in the cell list, not in `group_sizes`. The previous
/// `crossed` field, if any, is overwritten last-writer-wins.
///
/// `density` is clamped to `[0.0, 1.0]`. With `density = 1.0` this reduces
/// to a full Cartesian product; with very low density the bipartite cell
/// graph may become disconnected — see [`block_diagonal_crossings`] for
/// the canonical disconnected pattern that exercises
/// [`crate::pathology::StructuralIssue::DisconnectedCrossings`].
pub fn empty_crossings(
    spec: &mut GeneratorSpec,
    secondary_name: impl Into<String>,
    n_secondary: usize,
    re_var: f64,
    density: f64,
    seed: u64,
) {
    let n_primary = spec.group_sizes.len();
    let p = density.clamp(0.0, 1.0);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut cells = Vec::new();
    for i in 0..n_primary {
        for j in 0..n_secondary {
            if rng.gen::<f64>() < p {
                cells.push((i, j));
            }
        }
    }
    spec.crossed = Some(CrossedSpec::from_cells(
        secondary_name,
        n_secondary,
        re_var,
        cells,
    ));
}

/// Attach a crossed secondary grouping factor with a block-diagonal cell
/// pattern.
///
/// The bipartite cell graph splits into `n_blocks` disjoint components,
/// each a `block_size × block_size` complete sub-bipartite-graph. This is
/// the canonical "structurally empty crossings" pathology: levels in
/// different blocks never co-occur, so the design's bipartite incidence
/// graph is disconnected and the certificate flags
/// [`crate::pathology::StructuralIssue::DisconnectedCrossings`].
///
/// Both the primary group count and the secondary `n_levels` are set to
/// `n_blocks * block_size`; `spec.group_sizes` is rewritten to a stub
/// `vec![1; n_primary]` (its contents are unused for crossed designs) and
/// any pre-existing `crossed` field is overwritten last-writer-wins.
pub fn block_diagonal_crossings(
    spec: &mut GeneratorSpec,
    secondary_name: impl Into<String>,
    block_size: usize,
    n_blocks: usize,
    re_var: f64,
) {
    assert!(block_size >= 1, "block_size must be ≥ 1");
    assert!(n_blocks >= 1, "n_blocks must be ≥ 1");
    let n_levels = block_size * n_blocks;
    spec.group_sizes = vec![1; n_levels];
    let mut cells = Vec::with_capacity(block_size * block_size * n_blocks);
    for b in 0..n_blocks {
        let start = b * block_size;
        for i in 0..block_size {
            for j in 0..block_size {
                cells.push((start + i, start + j));
            }
        }
    }
    spec.crossed = Some(CrossedSpec::from_cells(
        secondary_name,
        n_levels,
        re_var,
        cells,
    ));
}

/// Generate Pareto-distributed (right-skewed) group sizes deterministically
/// from `seed`.
///
/// Returns a `Vec<usize>` of length `n_groups` with values in
/// `[1, mean_size · 50]`. Useful as input to [`set_group_sizes`]:
///
/// ```ignore
/// let sizes = pareto_sizes(123, 30, 1.5, 6.0);
/// set_group_sizes(&mut spec, sizes);
/// ```
///
/// The `alpha` parameter controls tail heaviness — smaller `alpha` →
/// heavier tail (more imbalance). `alpha = 1.5` gives a moderately skewed
/// distribution suitable for typical imbalance pathologies.
pub fn pareto_sizes(seed: u64, n_groups: usize, alpha: f64, mean_size: f64) -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(seed);
    let pareto = Pareto::new(1.0, alpha).unwrap();
    let cap = (mean_size * 50.0).max(2.0);
    (0..n_groups)
        .map(|_| {
            let raw = pareto.sample(&mut rng) * mean_size / (alpha / (alpha - 1.0).max(0.5));
            (raw.clamp(1.0, cap)).round() as usize
        })
        .collect()
}
