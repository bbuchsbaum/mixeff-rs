//! Two-tier separation detection for binomial GLMM specs.
//!
//! Logistic separation in mixed models has two distinct flavours:
//!
//! 1. **Fixed-effect separation** — the (X, y) design admits a hyperplane
//!    that perfectly (or quasi-perfectly) classifies the response. Detected
//!    via a small linear program in the spirit of Konis (2007).
//! 2. **Conditional separation** — at least one grouping level has all-
//!    success or all-failure outcomes, so the per-group random intercept
//!    has no MLE (drifts to ±∞). Detected by a per-group response scan.
//!
//! Single-tier detection misses the conditional case, which is the most
//! common GLMM blowup pattern in real clustered binary data (rare events
//! combined with many small clusters). The two-tier report drives the
//! `Refusal`-vs-`ConvergedPenalised` admittance set in
//! [`super::certificate::expected_statuses`].
//!
//! # Where the data comes from
//!
//! `certify` is documented as engine-free, seed-independent,
//! pure-linear-algebra over the *spec*. The LP-based FE detector and the
//! per-group conditional scan need realised `(X, y, groups)` and so are
//! kept *outside* `certify`. [`detect_separation`] generates a single
//! representative draw via [`super::spec::generate`] using `spec.seed`,
//! runs the LP and the scan, and returns a [`SeparationReport`]. The
//! certificate's spec-only `StructuralIssue::Separation` flag remains
//! the seed-independent surface that `expected_statuses` consumes; the
//! richer report is for diagnostics, parity scoreboards, and the test
//! harness.

use minilp::{ComparisonOp, LinearExpr, OptimizationDirection, Problem};
use nalgebra::DMatrix;

use super::spec::{generate, GeneratorSpec};
use crate::model::Family;

/// Margin tolerance for classifying the LP optimum.
///
/// Below this threshold the LP optimum is treated as exactly zero.
/// Calibrated against minilp's default termination tolerances.
const SEPARATION_MARGIN_TOL: f64 = 1e-6;

/// Symmetric box bound on β in the Konis LP.
///
/// The LP objective is invariant to scaling β; box-bounding the
/// variables makes the problem bounded and is standard for the
/// trichotomy formulation. The exact bound is irrelevant up to the
/// sign of the optimum, so we use a small constant for numerical
/// hygiene.
const BETA_BOX: f64 = 1.0;

/// Symmetric bound on the auxiliary margin variable ε.
///
/// `|Z_i β| ≤ p · max|x_ij| · BETA_BOX`, so a generous bound prevents
/// minilp from misclassifying an unbounded LP. We pick a constant
/// large enough for any pathology corpus design.
const EPSILON_BOUND: f64 = 1e6;

/// Konis (2007) trichotomy classifier for the fixed-effects design.
///
/// Variants describe the *separated* cases only — overlap (the
/// no-separation case) is reported as `None` in the parent
/// [`SeparationReport::fe_kind`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeSeparationKind {
    /// `∃ β ≠ 0` with `signs_i · x_i' β > 0` for every observation.
    /// The likelihood is unbounded; the MLE does not exist; refusal or
    /// a Firth-style penalty are the only contract-conformant responses.
    Complete,
    /// `∃ β ≠ 0` with `signs_i · x_i' β ≥ 0` for every observation and
    /// equality on at least one. The likelihood remains unbounded along
    /// the corresponding hyperplane direction; the contract response is
    /// identical to the complete case but the diagnostic surface differs
    /// (some observations sit exactly on the separator).
    QuasiComplete,
}

/// Combined two-tier separation report.
#[derive(Debug, Clone, PartialEq)]
pub struct SeparationReport {
    /// `Some(kind)` when the LP detected fixed-effect separation;
    /// `None` for overlap (no FE separation).
    pub fe_kind: Option<FeSeparationKind>,
    /// 0-based grouping-factor levels with all-zero or all-one outcomes.
    /// Empty when no grouping level is conditionally separated.
    pub conditional_groups: Vec<usize>,
    /// Direction `β` realising the LP optimum, when FE separation was
    /// detected. The hyperplane is `{x : x' β = 0}` in the fixed-effect
    /// design space (intercept first); `None` when overlap or when the
    /// LP could not be solved.
    pub hyperplane_direction: Option<Vec<f64>>,
}

impl SeparationReport {
    /// Empty report — no FE separation, no conditional separation.
    pub fn empty() -> Self {
        Self {
            fe_kind: None,
            conditional_groups: Vec::new(),
            hyperplane_direction: None,
        }
    }

    /// `true` when either tier detected separation.
    pub fn is_separated(&self) -> bool {
        self.fe_kind.is_some() || !self.conditional_groups.is_empty()
    }

    /// Number of conditionally-separated groups.
    pub fn n_conditional_groups(&self) -> usize {
        self.conditional_groups.len()
    }
}

/// Run both separation tiers on a representative draw of `spec`.
///
/// For non-Bernoulli specs separation is not applicable and the
/// returned report is empty. For Bernoulli specs the function calls
/// [`super::spec::generate`] using `spec.seed`, builds the realised
/// design `(X, y, groups)`, and runs:
///
/// 1. [`detect_fe_separation`] (LP-based Konis trichotomy) over the
///    full design including the intercept column.
/// 2. [`detect_conditional_separation`] over `(y, group_refs)`.
pub fn detect_separation(spec: &GeneratorSpec) -> SeparationReport {
    if !matches!(spec.family, Family::Bernoulli) {
        return SeparationReport::empty();
    }

    let Ok(out) = generate(spec) else {
        return SeparationReport::empty();
    };
    let n_pred = spec.n_fe_predictors();
    let y = match out.data.numeric(&spec.response_name) {
        Some(v) => v.to_vec(),
        None => return SeparationReport::empty(),
    };
    let n_obs = y.len();
    if n_obs == 0 {
        return SeparationReport::empty();
    }

    let mut x = DMatrix::zeros(n_obs, n_pred + 1);
    for i in 0..n_obs {
        x[(i, 0)] = 1.0;
    }
    for j in 0..n_pred {
        let col_name = format!("x{}", j + 1);
        if let Some(col) = out.data.numeric(&col_name) {
            for i in 0..n_obs {
                x[(i, j + 1)] = col[i];
            }
        }
    }

    let (fe_kind, hyperplane_direction) = detect_fe_separation(&x, &y);

    let conditional_groups = match out.data.categorical(&spec.group_name) {
        Some(col) => {
            let groups: Vec<usize> = col.refs.iter().map(|&r| r as usize).collect();
            detect_conditional_separation(&y, &groups)
        }
        None => Vec::new(),
    };

    SeparationReport {
        fe_kind,
        conditional_groups,
        hyperplane_direction,
    }
}

/// LP-based Konis (2007) trichotomy classifier for the fixed-effects
/// design `(X, y)`.
///
/// `y` must hold values in `{0.0, 1.0}`; any non-zero value is treated
/// as a success. Returns `(None, None)` for overlap and
/// `(Some(kind), Some(beta))` otherwise.
///
/// # Algorithm
///
/// Let `signs = 2y - 1 ∈ {-1, +1}^n` and `Z_i = signs_i · X_i`. We
/// solve two LPs:
///
/// **LP-A** (Konis "max-margin"):
/// ```text
/// max  ε
/// s.t. Z β - ε · 1 ≥ 0,  -1 ≤ β_j ≤ 1
/// ```
/// `ε* > tol` ⇒ `Complete` separation: there is a strict separating
/// hyperplane.
///
/// **LP-B** (residual non-trivial cone): only run when LP-A returns
/// `ε* ≤ tol`.
/// ```text
/// max  Σ_i Z_i β
/// s.t. Z β ≥ 0,  -1 ≤ β_j ≤ 1
/// ```
/// `objective > tol` ⇒ `QuasiComplete`: a non-trivial β satisfying
/// `Zβ ≥ 0` with at least one strict positive entry exists.
/// `objective ≤ tol` ⇒ overlap: only β = 0 satisfies the constraints
/// up to tolerance, so no separating hyperplane exists.
pub fn detect_fe_separation(
    x: &DMatrix<f64>,
    y: &[f64],
) -> (Option<FeSeparationKind>, Option<Vec<f64>>) {
    let n = x.nrows();
    let p = x.ncols();
    if n == 0 || p == 0 || y.len() != n {
        return (None, None);
    }

    let mut z = DMatrix::zeros(n, p);
    for i in 0..n {
        let sign = if y[i] > 0.5 { 1.0 } else { -1.0 };
        for j in 0..p {
            z[(i, j)] = sign * x[(i, j)];
        }
    }

    // LP-A.
    let mut prob = Problem::new(OptimizationDirection::Maximize);
    let beta_vars: Vec<_> = (0..p)
        .map(|_| prob.add_var(0.0, (-BETA_BOX, BETA_BOX)))
        .collect();
    let eps_var = prob.add_var(1.0, (-EPSILON_BOUND, EPSILON_BOUND));
    for i in 0..n {
        let mut expr = LinearExpr::empty();
        for j in 0..p {
            let zij = z[(i, j)];
            if zij != 0.0 {
                expr.add(beta_vars[j], zij);
            }
        }
        expr.add(eps_var, -1.0);
        prob.add_constraint(expr, ComparisonOp::Ge, 0.0);
    }
    let solution = match prob.solve() {
        Ok(s) => s,
        Err(_) => return (None, None),
    };
    let eps_star = solution[eps_var];
    let beta_star: Vec<f64> = beta_vars.iter().map(|&v| solution[v]).collect();

    if eps_star > SEPARATION_MARGIN_TOL {
        return (Some(FeSeparationKind::Complete), Some(beta_star));
    }

    // LP-B (only reached when ε* ≤ tol).
    let mut prob_b = Problem::new(OptimizationDirection::Maximize);
    let column_sums: Vec<f64> = (0..p).map(|j| z.column(j).iter().sum()).collect();
    let beta_b: Vec<_> = (0..p)
        .map(|j| prob_b.add_var(column_sums[j], (-BETA_BOX, BETA_BOX)))
        .collect();
    for i in 0..n {
        let mut expr = LinearExpr::empty();
        for j in 0..p {
            let zij = z[(i, j)];
            if zij != 0.0 {
                expr.add(beta_b[j], zij);
            }
        }
        prob_b.add_constraint(expr, ComparisonOp::Ge, 0.0);
    }
    let sol_b = match prob_b.solve() {
        Ok(s) => s,
        Err(_) => return (None, None),
    };
    if sol_b.objective() > SEPARATION_MARGIN_TOL {
        let beta_b_star: Vec<f64> = beta_b.iter().map(|&v| sol_b[v]).collect();
        return (Some(FeSeparationKind::QuasiComplete), Some(beta_b_star));
    }

    (None, None)
}

/// Per-group conditional separation scan.
///
/// Returns the sorted list of grouping levels whose binary outcomes
/// are all zero or all one. `groups[i]` is the 0-based level index for
/// observation `i`; `y[i]` is treated as a success when `> 0.5`.
pub fn detect_conditional_separation(y: &[f64], groups: &[usize]) -> Vec<usize> {
    if y.is_empty() || groups.len() != y.len() {
        return Vec::new();
    }
    let n_groups = groups.iter().copied().max().map(|m| m + 1).unwrap_or(0);
    if n_groups == 0 {
        return Vec::new();
    }
    let mut sums = vec![0u64; n_groups];
    let mut counts = vec![0u64; n_groups];
    for (i, &g) in groups.iter().enumerate() {
        if g >= n_groups {
            continue;
        }
        if y[i] > 0.5 {
            sums[g] += 1;
        }
        counts[g] += 1;
    }
    let mut separated: Vec<usize> = (0..n_groups)
        .filter(|&g| counts[g] > 0 && (sums[g] == 0 || sums[g] == counts[g]))
        .collect();
    separated.sort_unstable();
    separated
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_design(rows: &[(f64, f64)]) -> (DMatrix<f64>, Vec<f64>) {
        // Each row is (x, y). Design columns: [intercept, x].
        let n = rows.len();
        let mut x = DMatrix::zeros(n, 2);
        let mut y = Vec::with_capacity(n);
        for (i, &(xi, yi)) in rows.iter().enumerate() {
            x[(i, 0)] = 1.0;
            x[(i, 1)] = xi;
            y.push(yi);
        }
        (x, y)
    }

    #[test]
    fn fe_complete_separation_strict_sign_split() {
        // x < 0 → y = 0, x > 0 → y = 1: textbook complete separation.
        let (x, y) = make_design(&[
            (-2.0, 0.0),
            (-1.5, 0.0),
            (-1.0, 0.0),
            (1.0, 1.0),
            (1.5, 1.0),
            (2.0, 1.0),
        ]);
        let (kind, beta) = detect_fe_separation(&x, &y);
        assert_eq!(kind, Some(FeSeparationKind::Complete));
        let beta = beta.expect("expected hyperplane direction for separated design");
        // Slope should be positive (since y = 1 happens on positive x).
        assert!(
            beta[1] > SEPARATION_MARGIN_TOL,
            "slope should be positive: {beta:?}"
        );
    }

    #[test]
    fn fe_quasi_complete_separation_with_tie() {
        // x ≤ 0 → y = 0, x ≥ 0 → y = 1, with x = 0 in both classes.
        let (x, y) = make_design(&[
            (-2.0, 0.0),
            (-1.0, 0.0),
            (0.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
            (2.0, 1.0),
        ]);
        let (kind, _beta) = detect_fe_separation(&x, &y);
        assert_eq!(kind, Some(FeSeparationKind::QuasiComplete));
    }

    #[test]
    fn fe_overlap_no_separation() {
        // Random-ish 1-D pattern with overlapping classes.
        let (x, y) = make_design(&[
            (-1.0, 0.0),
            (-0.5, 1.0),
            (0.0, 0.0),
            (0.5, 1.0),
            (1.0, 0.0),
            (1.5, 1.0),
        ]);
        let (kind, beta) = detect_fe_separation(&x, &y);
        assert_eq!(kind, None);
        assert!(beta.is_none());
    }

    #[test]
    fn conditional_separation_picks_all_zero_and_all_one_groups() {
        let y = vec![0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0];
        let groups = vec![0, 0, 0, 1, 1, 2, 2, 2];
        // group 0: all 0 (separated). group 1: 1/2 (not separated). group 2: all 1 (separated).
        let separated = detect_conditional_separation(&y, &groups);
        assert_eq!(separated, vec![0, 2]);
    }

    #[test]
    fn conditional_separation_handles_singleton_groups() {
        let y = vec![0.0, 1.0, 0.0];
        let groups = vec![0, 1, 2];
        let separated = detect_conditional_separation(&y, &groups);
        assert_eq!(separated, vec![0, 1, 2]);
    }

    #[test]
    fn conditional_separation_empty_inputs() {
        assert!(detect_conditional_separation(&[], &[]).is_empty());
    }
}
