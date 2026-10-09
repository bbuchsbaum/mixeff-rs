//! Estimated-dispersion GLMMs (Gamma, inverse Gaussian, Gaussian with a
//! non-identity link): lme4 2.1-0's `glmerControl(disp_method = "moment")`.
//!
//! lme4 before 2.1 computed the PIRLS working weights with the dispersion
//! fixed at φ = 1, so the conditional modes, the Laplace log-determinant and
//! hence θ were biased whenever φ ≠ 1 (lme4 GH #557, #643, #936). lme4 2.1
//! profiles φ in a nested fixed-point loop around PIRLS (`profilePhi()` in
//! lme4's `src/external.cpp`):
//!
//! * the working weights are `w μ'(η)² / (φ V(μ))` and the PIRLS criterion
//!   is `dev/φ + ‖u‖²`, so `u ~ N(0, I)` and `b = Λu` is on the absolute
//!   link scale (θ is not relative to σ);
//! * each outer step sets `φ_new = dev / (Σw − q_eff)` (the moment estimator,
//!   `q_eff = rank([X, Z])` under `disp_dof_correction = TRUE`) and damps
//!   `log φ ← 0.9 log φ + 0.1 log φ_new`, starting from φ = 1, stopping when
//!   `|φ_new − φ| / φ < 1e-8` or after `maxPhiIter = 100` steps; the
//!   *damped* value is the reported `sigma()²`;
//! * the Laplace / AGQ criterion keeps the family `aic()` conditional
//!   density with the mean-deviance plug-in `dev / Σw` (no dof correction);
//!   lme4 2.1 also drops the `+2` that `aic()` adds, so its logLik is the
//!   plain Laplace value (this engine never carried that `+2`).
//!
//! The engine replicates that iteration exactly (same start, damping, stop
//! rule and cap) so that θ, β, σ and logLik agree with lme4 2.1 to optimizer
//! tolerance. [`GlmmDispersionMethod::Legacy`] keeps the pre-2.1 (unit-φ
//! working weights) behaviour, matching `disp_method = "old/buggy"`.

use super::*;

/// How the dispersion of an estimated-dispersion GLMM (Gamma, inverse
/// Gaussian, Gaussian with a non-identity link) enters the PIRLS fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum GlmmDispersionMethod {
    /// lme4 2.1's default `glmerControl(disp_method = "moment")`: φ is
    /// profiled in a damped fixed-point loop around PIRLS, the working
    /// weights are divided by φ, θ is the absolute random-effect SD on the
    /// link scale and `sigma() = sqrt(φ)`.
    #[default]
    Moment,
    /// lme4 < 2.1 (`disp_method = "old/buggy"`): working weights computed
    /// with φ = 1, `sigma() = sqrt(pwrss / n)`, and θ relative to σ.
    Legacy,
}

/// lme4 2.1 `glmerControl` dispersion controls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct DispersionControl {
    pub(crate) method: GlmmDispersionMethod,
    /// `disp_dof_correction`: divide the deviance by `Σw − rank([X, Z])`
    /// rather than `Σw` in the moment estimator.
    pub(crate) dof_correction: bool,
    /// `maxPhiIter`: cap on the nested φ iterations per objective evaluation.
    pub(crate) max_phi_iter: usize,
    /// Solve PIRLS at every damped step instead of replaying the trajectory
    /// on an interpolant (tests compare the two).
    pub(crate) exact_phi_trajectory: bool,
}

impl Default for DispersionControl {
    fn default() -> Self {
        Self {
            method: GlmmDispersionMethod::Moment,
            dof_correction: true,
            max_phi_iter: 100,
            exact_phi_trajectory: false,
        }
    }
}

/// lme4's `phiTol` for the nested dispersion loop.
const PHI_TOL: f64 = 1.0e-8;

/// Chebyshev nodes of the moment-map interpolant over the whole φ path.
const SURROGATE_PATH_NODES: usize = 13;
/// Chebyshev nodes of the interpolant near the fixed point.
const SURROGATE_TAIL_NODES: usize = 7;
/// Half-width (in log φ) of the near-fixed-point interpolant.
const SURROGATE_TAIL_RADIUS: f64 = 0.5;
/// Secant steps allowed to locate the fixed point.
const SURROGATE_SECANT_STEPS: usize = 12;

/// β-varying PIRLS tolerance for the interpolation nodes, relative to the
/// Laplace objective's magnitude.
const SURROGATE_NODE_PIRLS_RTOL: f64 = 1.0e-12;

/// Below this `maxPhiIter` the step-by-step loop is no dearer than the
/// surrogate, so it is used directly.
const SURROGATE_MIN_PHI_ITER: usize = SURROGATE_PATH_NODES + SURROGATE_TAIL_NODES + 10;

/// lme4's damped update `log φ ← 0.9 log φ + 0.1 log φ_new`.
fn damped_log_phi(log_phi: f64, log_new: f64) -> f64 {
    0.9 * log_phi + 0.1 * log_new
}

/// Largest dense Schur complement (columns of `[Z_2 … Z_K, X]` after the
/// first random-effect term is eliminated level by level) the rank of
/// `[X, Z]` is computed for. Beyond it the dof correction is skipped, as
/// lme4 does when its sparse QR fails.
const MAX_DENSE_RANK_COLUMNS: usize = 2_500;

/// Squared relative residual below which a column counts as dependent.
const RANK_TOL: f64 = 1.0e-9;

/// Barycentric interpolant through Chebyshev points of the first kind on
/// `[lo, hi]` (widened by 1% so the end points are covered).
struct Interpolant {
    lo: f64,
    hi: f64,
    nodes: Vec<f64>,
    values: Vec<f64>,
}

impl Interpolant {
    fn sample<M>(
        model: &mut M,
        eval: &mut impl FnMut(&mut M, f64) -> Result<Option<f64>>,
        lo: f64,
        hi: f64,
        n: usize,
    ) -> Result<Option<Self>> {
        let pad = 0.01 * (hi - lo) + 1.0e-6;
        let (lo, hi) = (lo - pad, hi + pad);
        let nodes: Vec<f64> = (0..n)
            .map(|k| {
                let angle = std::f64::consts::PI * (k as f64 + 0.5) / n as f64;
                0.5 * (lo + hi) + 0.5 * (hi - lo) * angle.cos()
            })
            .collect();
        // Visit the nodes from φ = 1 outwards so each PIRLS solve
        // warm-starts from a nearby one.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&i, &j| nodes[i].abs().total_cmp(&nodes[j].abs()));
        let mut values = vec![0.0; n];
        for k in order {
            let Some(value) = eval(model, nodes[k])? else {
                return Ok(None);
            };
            values[k] = value;
        }
        Ok(Some(Self {
            lo,
            hi,
            nodes,
            values,
        }))
    }

    fn contains(&self, x: f64) -> bool {
        (self.lo..=self.hi).contains(&x)
    }

    fn eval(&self, x: f64) -> f64 {
        let n = self.nodes.len();
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for k in 0..n {
            let diff = x - self.nodes[k];
            if diff == 0.0 {
                return self.values[k];
            }
            let angle = std::f64::consts::PI * (k as f64 + 0.5) / n as f64;
            let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
            let weight = sign * angle.sin() / diff;
            numerator += weight * self.values[k];
            denominator += weight;
        }
        numerator / denominator
    }
}

impl GeneralizedLinearMixedModel {
    /// The dispersion method used by this model's fits.
    pub fn dispersion_method(&self) -> GlmmDispersionMethod {
        self.dispersion_control.method
    }

    /// Choose how the dispersion of an estimated-dispersion GLMM enters the
    /// fit (lme4 2.1 `glmerControl(disp_method=)`); the default is
    /// [`GlmmDispersionMethod::Moment`]. Takes effect at the next fit and is
    /// ignored for fixed-dispersion families.
    pub fn set_dispersion_method(&mut self, method: GlmmDispersionMethod) -> &mut Self {
        self.dispersion_control.method = method;
        self
    }

    /// lme4 2.1 `glmerControl(disp_dof_correction=)`: whether the moment
    /// estimator divides the deviance by `Σw − rank([X, Z])` (default) or by
    /// `Σw`. Takes effect at the next fit.
    pub fn set_dispersion_dof_correction(&mut self, correct: bool) -> &mut Self {
        self.dispersion_control.dof_correction = correct;
        self
    }

    /// lme4 2.1 `glmerControl(maxPhiIter=)` (default 100). Takes effect at
    /// the next fit.
    pub fn set_max_phi_iter(&mut self, max_phi_iter: usize) -> Result<&mut Self> {
        if max_phi_iter == 0 {
            return Err(MixedModelError::InvalidArgument(
                "max_phi_iter must be a positive integer".to_string(),
            ));
        }
        self.dispersion_control.max_phi_iter = max_phi_iter;
        Ok(self)
    }

    /// Whether this model profiles an estimated dispersion φ inside PIRLS
    /// (an estimated-dispersion family under the moment method).
    pub(crate) fn profiles_dispersion(&self) -> bool {
        self.family.has_dispersion()
            && self.dispersion_control.method == GlmmDispersionMethod::Moment
    }

    /// The φ dividing the deviance in the PIRLS criterion and the working
    /// weights: the profiled dispersion, or 1.
    pub(crate) fn pirls_phi(&self) -> f64 {
        if self.profiles_dispersion() {
            self.pirls_phi
        } else {
            1.0
        }
    }

    /// `rank([X, Z])` subtracted from `Σw` in the moment estimator, or 0
    /// when the correction is disabled or the rank is unavailable.
    pub(crate) fn dispersion_dof_rank(&self) -> f64 {
        if !self.dispersion_control.dof_correction {
            return 0.0;
        }
        self.design_rank_cache
            .get_or_init(|| fixed_random_design_rank(&self.lmm.feterm, &self.lmm.reterms))
            .map_or(0.0, |rank| rank as f64)
    }

    fn total_prior_weight(&self) -> f64 {
        if self.wt.is_empty() {
            self.y.len() as f64
        } else {
            self.wt.iter().sum()
        }
    }

    pub(crate) fn weighted_deviance(&self) -> f64 {
        (0..self.y.len())
            .map(|i| self.case_weight(i) * self.dev_resid_component(self.y[i], self.mu[i]))
            .sum()
    }

    /// PIRLS at φ = exp(`log_phi`) (warm-started unless `reset`), returning
    /// the convergence flag and `log(dev / denominator)` (the log of the
    /// moment update), or `None` when that is not finite.
    fn pirls_at_log_phi(
        &mut self,
        log_phi: f64,
        vary_beta: bool,
        verbose: bool,
        max_iter: usize,
        reset: bool,
        denominator: f64,
    ) -> Result<(bool, Option<f64>)> {
        self.pirls_phi = log_phi.exp();
        let converged = self.pirls_iterations(vary_beta, verbose, max_iter, reset)?;
        let phi_new = self.weighted_deviance() / denominator;
        let log_new = (phi_new.is_finite() && phi_new > 0.0).then(|| phi_new.ln());
        Ok((converged, log_new))
    }

    /// lme4 2.1 `profilePhi()` verbatim: one PIRLS solve per damped step.
    fn exact_phi_trajectory(
        &mut self,
        vary_beta: bool,
        verbose: bool,
        max_iter: usize,
        reset_modes: bool,
        denominator: f64,
        max_phi_iter: usize,
    ) -> Result<bool> {
        let mut log_phi = 0.0_f64;
        let mut converged = false;
        for outer in 0..max_phi_iter {
            let (pirls_converged, log_new) = self.pirls_at_log_phi(
                log_phi,
                vary_beta,
                verbose,
                max_iter,
                reset_modes && outer == 0,
                denominator,
            )?;
            converged = pirls_converged;
            let Some(log_new) = log_new else { break };
            let phi_converged = (log_new - log_phi).exp_m1().abs() < PHI_TOL;
            log_phi = damped_log_phi(log_phi, log_new);
            if phi_converged {
                break;
            }
        }
        self.pirls_phi = log_phi.exp();
        Ok(converged)
    }

    /// The same damped trajectory as [`exact_phi_trajectory`](Self::exact_phi_trajectory), with the
    /// moment map `g: ℓ ↦ log(dev(e^ℓ) / denominator)` replaced by Chebyshev
    /// interpolants of exact (tightly converged) PIRLS evaluations, and the
    /// last step (whose conditional modes the fit keeps) solved exactly.
    ///
    /// The fixed point ℓ* = g(ℓ*) is located first by secant steps; one
    /// interpolant covers the whole path [ℓ*, 0] and a second, narrow one the
    /// neighbourhood of ℓ* where the damped iteration spends its last steps
    /// (errors made on the way in are damped by 0.9 per remaining step).
    /// The reproduced φ agrees with the step-by-step loop to ~1e-9 at about a
    /// third of the PIRLS solves. Returns `None` (after which the caller runs
    /// the exact loop) when any stage is not clean.
    fn surrogate_phi_trajectory(
        &mut self,
        vary_beta: bool,
        verbose: bool,
        max_iter: usize,
        reset_modes: bool,
        denominator: f64,
        max_phi_iter: usize,
    ) -> Result<Option<bool>> {
        let (converged0, Some(g0)) =
            self.pirls_at_log_phi(0.0, vary_beta, verbose, max_iter, reset_modes, denominator)?
        else {
            return Ok(None);
        };
        if g0.exp_m1().abs() < PHI_TOL {
            // lme4 stops after its first step; the modes are those at φ = 1.
            self.pirls_phi = damped_log_phi(0.0, g0).exp();
            return Ok(Some(converged0));
        }
        self.pirls_tolerance_override =
            Some(SURROGATE_NODE_PIRLS_RTOL * self.laplace_objective().abs().max(1.0));
        let sampled =
            self.sample_moment_map(g0, vary_beta, verbose, max_iter, denominator, max_phi_iter);
        self.pirls_tolerance_override = None;
        let Some(last_solved) = sampled? else {
            return Ok(None);
        };
        let (converged, Some(g_last)) = self.pirls_at_log_phi(
            last_solved,
            vary_beta,
            verbose,
            max_iter,
            false,
            denominator,
        )?
        else {
            return Ok(None);
        };
        self.pirls_phi = damped_log_phi(last_solved, g_last).exp();
        Ok(Some(converged))
    }

    /// Locate ℓ*, build the interpolants and replay the damped iteration on
    /// them, returning the last log φ at which lme4 would solve PIRLS.
    fn sample_moment_map(
        &mut self,
        g0: f64,
        vary_beta: bool,
        verbose: bool,
        max_iter: usize,
        denominator: f64,
        max_phi_iter: usize,
    ) -> Result<Option<f64>> {
        let mut eval = |model: &mut Self, log_phi: f64| -> Result<Option<f64>> {
            Ok(model
                .pirls_at_log_phi(log_phi, vary_beta, verbose, max_iter, false, denominator)?
                .1)
        };
        // Secant on f(ℓ) = g(ℓ) − ℓ from ℓ = 0 and the undamped step ℓ = g0.
        let (mut x0, mut f0) = (0.0_f64, g0);
        let mut x1 = g0;
        let Some(g1) = eval(self, x1)? else {
            return Ok(None);
        };
        let mut f1 = g1 - x1;
        let span = g0.abs().max(1.0);
        let mut fixed_point = None;
        for _ in 0..SURROGATE_SECANT_STEPS {
            if f1.abs() < 1.0e-9 {
                fixed_point = Some(x1);
                break;
            }
            if f1 == f0 {
                return Ok(None);
            }
            let x2 = x1 - f1 * (x1 - x0) / (f1 - f0);
            if !x2.is_finite() || (x2 - x1).abs() > 10.0 * span {
                return Ok(None);
            }
            let Some(g2) = eval(self, x2)? else {
                return Ok(None);
            };
            (x0, f0, x1, f1) = (x1, f1, x2, g2 - x2);
        }
        let Some(fixed_point) = fixed_point else {
            return Ok(None);
        };

        let path = Interpolant::sample(
            self,
            &mut eval,
            fixed_point.min(0.0),
            fixed_point.max(0.0),
            SURROGATE_PATH_NODES,
        )?;
        let Some(path) = path else {
            return Ok(None);
        };
        // The tail: the part of the path within `radius` of ℓ*, plus a small
        // margin beyond it.
        let radius = fixed_point.abs().min(SURROGATE_TAIL_RADIUS);
        let toward_start = -fixed_point.signum() * radius;
        let beyond = -toward_start * 0.05;
        let (tail_lo, tail_hi) = (
            (fixed_point + beyond).min(fixed_point + toward_start),
            (fixed_point + beyond).max(fixed_point + toward_start),
        );
        let Some(tail) =
            Interpolant::sample(self, &mut eval, tail_lo, tail_hi, SURROGATE_TAIL_NODES)?
        else {
            return Ok(None);
        };

        let mut log_phi = 0.0_f64;
        let mut last_solved = 0.0_f64;
        for outer in 0..max_phi_iter {
            let g = if outer == 0 {
                g0
            } else if tail.contains(log_phi) {
                tail.eval(log_phi)
            } else if path.contains(log_phi) {
                path.eval(log_phi)
            } else {
                return Ok(None);
            };
            last_solved = log_phi;
            let phi_converged = (g - log_phi).exp_m1().abs() < PHI_TOL;
            log_phi = damped_log_phi(log_phi, g);
            if phi_converged {
                break;
            }
        }
        Ok(Some(last_solved))
    }

    /// Run PIRLS, profiling φ around it as lme4 2.1's `profilePhi()` does
    /// for estimated-dispersion families, then refresh the working weights
    /// and factorization at the accepted mean and final φ (lme4 GH #998) so
    /// the Laplace log-determinant, AGQ node scales and RX describe the
    /// accepted modes.
    pub(super) fn pirls_profiling_dispersion(
        &mut self,
        vary_beta: bool,
        verbose: bool,
        max_iter: usize,
        reset_modes: bool,
    ) -> Result<bool> {
        let converged = if self.profiles_dispersion() {
            let denominator =
                (self.total_prior_weight() - self.dispersion_dof_rank()).max(f64::MIN_POSITIVE);
            let max_phi_iter = self.dispersion_control.max_phi_iter.max(1);
            let surrogate = if self.dispersion_control.exact_phi_trajectory
                || max_phi_iter <= SURROGATE_MIN_PHI_ITER
            {
                None
            } else {
                self.surrogate_phi_trajectory(
                    vary_beta,
                    verbose,
                    max_iter,
                    reset_modes,
                    denominator,
                    max_phi_iter,
                )?
            };
            match surrogate {
                Some(converged) => converged,
                None => self.exact_phi_trajectory(
                    vary_beta,
                    verbose,
                    max_iter,
                    reset_modes,
                    denominator,
                    max_phi_iter,
                )?,
            }
        } else {
            self.pirls_iterations(vary_beta, verbose, max_iter, reset_modes)?
        };
        if vary_beta || self.profiles_dispersion() {
            // The fixed-β solve already ends on a refreshed factorization;
            // only φ's final damping step can have moved it since.
            let n = self.y.len();
            let mut sqrtwts = vec![0.0f64; n];
            let mut working_y = vec![0.0f64; n];
            self.update_pirls_working_state(vary_beta, &mut sqrtwts, &mut working_y)?;
        }
        self.refresh_dispersion();
        Ok(converged)
    }
}

/// `rank([X, Z])` (lme4's `computeQEff()`), or `None` when it is too costly.
///
/// The first random-effect term's columns are eliminated level by level
/// (its Gram matrix is block diagonal); the remaining columns' Gram matrix
/// is projected off that span and its rank found by pivoted Cholesky on
/// unit-normalized columns.
pub(crate) fn fixed_random_design_rank(
    feterm: &crate::types::FeTerm,
    reterms: &[ReMat],
) -> Option<usize> {
    let n = feterm.x.nrows();
    let p = feterm.rank;
    let Some((first, rest)) = reterms.split_first() else {
        return Some(p);
    };
    let mut offsets = Vec::with_capacity(rest.len());
    let mut m = 0usize;
    for term in rest {
        offsets.push(m);
        m += term.vsize * term.n_levels();
    }
    let x_offset = m;
    m += p;
    if m > MAX_DENSE_RANK_COLUMNS {
        return None;
    }

    // Sparse row of the remaining columns for one observation.
    let row = |obs: usize, cols: &mut Vec<(usize, f64)>| {
        cols.clear();
        for (term, &offset) in rest.iter().zip(&offsets) {
            let base = offset + term.refs[obs] as usize * term.vsize;
            for s in 0..term.vsize {
                let value = term.z[(s, obs)];
                if value != 0.0 {
                    cols.push((base + s, value));
                }
            }
        }
        for j in 0..p {
            let value = feterm.x[(obs, j)];
            if value != 0.0 {
                cols.push((x_offset + j, value));
            }
        }
    };

    let mut gram = DMatrix::<f64>::zeros(m, m);
    let mut cols = Vec::new();
    let vs = first.vsize;
    let n_levels = first.n_levels();
    let mut level_obs: Vec<Vec<usize>> = vec![Vec::new(); n_levels];
    for obs in 0..n {
        level_obs[first.refs[obs] as usize].push(obs);
        row(obs, &mut cols);
        for &(a, va) in &cols {
            for &(b, vb) in &cols {
                gram[(a, b)] += va * vb;
            }
        }
    }
    let original_diag: Vec<f64> = (0..m).map(|j| gram[(j, j)]).collect();

    let mut rank = 0usize;
    for obs_in_level in &level_obs {
        // G_l = Σ z zᵀ and B_l = Σ z cᵀ (only the touched columns).
        let mut g = DMatrix::<f64>::zeros(vs, vs);
        let mut touched: Vec<usize> = Vec::new();
        let mut b_entries: std::collections::BTreeMap<usize, DVector<f64>> =
            std::collections::BTreeMap::new();
        for &obs in obs_in_level {
            let z: Vec<f64> = (0..vs).map(|s| first.z[(s, obs)]).collect();
            for a in 0..vs {
                for b in 0..vs {
                    g[(a, b)] += z[a] * z[b];
                }
            }
            row(obs, &mut cols);
            for &(c, value) in &cols {
                let entry = b_entries.entry(c).or_insert_with(|| DVector::zeros(vs));
                for s in 0..vs {
                    entry[s] += z[s] * value;
                }
            }
        }
        touched.extend(b_entries.keys().copied());
        let eigen = SymmetricEigen::new(g);
        let max_eigen = eigen.eigenvalues.iter().fold(0.0_f64, |a, &v| a.max(v));
        if max_eigen <= 0.0 {
            continue;
        }
        let mut whitened: Vec<DVector<f64>> = Vec::new();
        for (k, &lambda) in eigen.eigenvalues.iter().enumerate() {
            if lambda > max_eigen * 1.0e-10 {
                rank += 1;
                let vector = eigen.eigenvectors.column(k);
                // Projection row (vᵀ B_l) / sqrt(λ) over the touched columns.
                let projected = DVector::from_iterator(
                    touched.len(),
                    touched
                        .iter()
                        .map(|c| vector.dot(&b_entries[c]) / lambda.sqrt()),
                );
                whitened.push(projected);
            }
        }
        for projected in &whitened {
            for (i, &a) in touched.iter().enumerate() {
                for (j, &b) in touched.iter().enumerate() {
                    gram[(a, b)] -= projected[i] * projected[j];
                }
            }
        }
    }

    // Pivoted Cholesky on the normalized projected Gram matrix.
    let scale: Vec<f64> = original_diag
        .iter()
        .map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 })
        .collect();
    for a in 0..m {
        for b in 0..m {
            gram[(a, b)] *= scale[a] * scale[b];
        }
    }
    let mut active: Vec<usize> = (0..m).collect();
    while let Some((position, &pivot)) = active
        .iter()
        .enumerate()
        .max_by(|(_, &a), (_, &b)| gram[(a, a)].total_cmp(&gram[(b, b)]))
    {
        let pivot_value = gram[(pivot, pivot)];
        if pivot_value <= RANK_TOL {
            break;
        }
        rank += 1;
        active.swap_remove(position);
        let column: Vec<f64> = active.iter().map(|&j| gram[(j, pivot)]).collect();
        for (i, &a) in active.iter().enumerate() {
            for (j, &b) in active.iter().enumerate() {
                gram[(a, b)] -= column[i] * column[j] / pivot_value;
            }
        }
    }
    Some(rank)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formula::parse_formula;

    fn fixture_data() -> DataFrame {
        let text = include_str!("../../../tests/fixtures/parity/dispersion_glmm_lme4_scale.csv");
        let (mut y, mut x, mut g) = (vec![], vec![], vec![]);
        for line in text.lines().skip(1) {
            let fields: Vec<&str> = line.split(',').collect();
            y.push(fields[0].parse::<f64>().unwrap());
            x.push(fields[1].parse::<f64>().unwrap());
            g.push(fields[2].trim_matches('"').to_string());
        }
        let mut data = DataFrame::new();
        data.add_numeric("y", y).unwrap();
        data.add_numeric("x", x).unwrap();
        data.add_categorical("g", g).unwrap();
        data
    }

    fn model(family: Family) -> GeneralizedLinearMixedModel {
        GeneralizedLinearMixedModel::new(
            parse_formula("y ~ x + (1 | g)").unwrap(),
            &fixture_data(),
            family,
            Some(LinkFunction::Log),
        )
        .unwrap()
    }

    /// gamma_glmm_engines.json's toy design: φ ≈ 0.0027, so the damped path
    /// spans ~6 in log φ and is still 2e-4 short of the fixed point after
    /// 100 steps.
    fn small_phi_model() -> GeneralizedLinearMixedModel {
        let effects = [-0.25, 0.1, 0.3, -0.15];
        let (mut y, mut x, mut g) = (vec![], vec![], vec![]);
        for (group, effect) in effects.iter().enumerate() {
            for obs in 0..5 {
                let xv = obs as f64 - 2.0;
                let wiggle = 1.0 + 0.06 * ((group + obs) % 3) as f64;
                y.push((1.2 + 0.25 * xv + effect).exp() * wiggle);
                x.push(xv);
                g.push(format!("g{}", group + 1));
            }
        }
        let mut data = DataFrame::new();
        data.add_numeric("y", y).unwrap();
        data.add_numeric("x", x).unwrap();
        data.add_categorical("group", g).unwrap();
        GeneralizedLinearMixedModel::new(
            parse_formula("y ~ 1 + x + (1 | group)").unwrap(),
            &data,
            Family::Gamma,
            Some(LinkFunction::Log),
        )
        .unwrap()
    }

    #[test]
    fn surrogate_phi_trajectory_matches_lme4_on_a_long_path() {
        // lme4 2.1's devfun at theta = 0.2244366538916937 (nAGQ = 0):
        // phi 0.002663294, deviance criterion 4.680432.
        let mut exact = small_phi_model();
        exact.dispersion_control.exact_phi_trajectory = true;
        let mut fast = small_phi_model();
        for model in [&mut exact, &mut fast] {
            model
                .update_pirls_at_theta(&[0.2244366538916937], true)
                .unwrap();
            assert!((model.pirls_phi - 0.002663294).abs() < 1e-9);
            assert!((model.profiled_outer_objective(1) - 4.680432).abs() < 1e-5);
        }
        assert!((exact.pirls_phi - fast.pirls_phi).abs() / exact.pirls_phi < 1e-7);
    }

    fn dense_design_rank(model: &GeneralizedLinearMixedModel) -> usize {
        let n = model.y.len();
        let p = model.lmm.feterm.rank;
        let q: usize = model
            .lmm
            .reterms
            .iter()
            .map(|t| t.vsize * t.n_levels())
            .sum();
        let mut full = DMatrix::<f64>::zeros(n, p + q);
        for obs in 0..n {
            for j in 0..p {
                full[(obs, j)] = model.lmm.feterm.x[(obs, j)];
            }
            let mut offset = p;
            for term in &model.lmm.reterms {
                for s in 0..term.vsize {
                    full[(obs, offset + term.refs[obs] as usize * term.vsize + s)] =
                        term.z[(s, obs)];
                }
                offset += term.vsize * term.n_levels();
            }
        }
        let svd = full.svd(false, false);
        let max = svd.singular_values.max();
        svd.singular_values
            .iter()
            .filter(|&&v| v > max * 1e-9)
            .count()
    }

    #[test]
    fn design_rank_matches_dense_svd_for_crossed_nested_and_slope_terms() {
        let n = 96;
        let (mut y, mut x, mut g, mut h, mut k) = (vec![], vec![], vec![], vec![], vec![]);
        for i in 0..n {
            y.push(1.0 + ((i * 37) % 11) as f64 / 7.0);
            // Constant within some `g` levels, so those slope columns are
            // collinear with the intercepts.
            x.push(if i % 12 < 6 {
                0.5
            } else {
                ((i * 7) % 5) as f64 - 2.0
            });
            g.push(format!("g{}", i % 12));
            h.push(format!("h{}", i % 5));
            // `k` is nested in `g` (and so adds levels aliased with it).
            k.push(format!("k{}", i % 24));
        }
        let mut data = DataFrame::new();
        data.add_numeric("y", y).unwrap();
        data.add_numeric("x", x).unwrap();
        data.add_categorical("g", g).unwrap();
        data.add_categorical("h", h).unwrap();
        data.add_categorical("k", k).unwrap();
        for formula in [
            "y ~ x + (1 | g)",
            "y ~ x + (1 + x | g)",
            "y ~ x + (1 | g) + (1 | h)",
            "y ~ x + (1 + x | g) + (1 | h) + (1 | k)",
            "y ~ 1 + (1 | g) + (1 | k)",
        ] {
            let model = GeneralizedLinearMixedModel::new(
                parse_formula(formula).unwrap(),
                &data,
                Family::Gamma,
                Some(LinkFunction::Log),
            )
            .unwrap();
            assert_eq!(
                fixed_random_design_rank(&model.lmm.feterm, &model.lmm.reterms),
                Some(dense_design_rank(&model)),
                "{formula}"
            );
        }
    }

    #[test]
    fn design_rank_matches_lme4_q_eff() {
        // lme4 2.1 reports dims[["qEff"]] = 16 for this design (2 fixed
        // columns + 15 groups, one shared intercept direction).
        let model = model(Family::Gamma);
        assert_eq!(
            fixed_random_design_rank(&model.lmm.feterm, &model.lmm.reterms),
            Some(16)
        );
    }

    #[test]
    fn surrogate_phi_trajectory_reproduces_the_step_by_step_loop() {
        for family in [Family::Gamma, Family::InverseGaussian, Family::Normal] {
            for vary_beta in [true, false] {
                for theta in [0.05, 0.3, 1.2] {
                    let mut exact = model(family);
                    exact.dispersion_control.exact_phi_trajectory = true;
                    let mut fast = model(family);
                    exact.update_pirls_at_theta(&[theta], vary_beta).unwrap();
                    fast.update_pirls_at_theta(&[theta], vary_beta).unwrap();
                    let label = format!("{family:?} vary_beta={vary_beta} theta={theta}");
                    let rel = (exact.pirls_phi - fast.pirls_phi).abs() / exact.pirls_phi;
                    // The fixed-β (joint) solve is converged to a 1e-8 score, so
                    // the two agree to interpolation accuracy; the profiled
                    // (β-varying) PIRLS stops at a 1e-5 objective change, which
                    // bounds how closely any two φ paths can agree.
                    let (phi_tol, objective_tol) = if vary_beta {
                        (5e-5, 2e-4)
                    } else {
                        (1e-8, 1e-7)
                    };
                    assert!(
                        rel < phi_tol,
                        "{label}: phi {} vs {}",
                        exact.pirls_phi,
                        fast.pirls_phi
                    );
                    let (a, b) = (
                        exact.deviance_with_response_constants(1),
                        fast.deviance_with_response_constants(1),
                    );
                    assert!(
                        (a - b).abs() < objective_tol,
                        "{label}: objective {a} vs {b}"
                    );
                }
            }
        }
    }
}
