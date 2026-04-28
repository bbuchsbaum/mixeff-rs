//! Profile-likelihood confidence intervals for linear mixed models.
//!
//! This is a partial port of `MixedModels.jl/src/profile/`. The current
//! scope covers the residual-scale profile (σ) together with the shared
//! [`MixedModelProfile`] container and [`MixedModelProfile::confint`]. The
//! β, θ, and variance-component profiles will be added in later passes.
//!
//! The basic idea: fix one model parameter at a series of values on either
//! side of its estimate, refit the model with that parameter held constant,
//! and record the signed square root of the excess objective,
//!
//! ```text
//!     ζ(θ) = sign(θ − θ̂) · sqrt(obj(θ) − obj(θ̂))
//! ```
//!
//! `ζ` is (approximately) standard normal under the usual regularity
//! conditions, so a profile-likelihood CI at level `1 − α` is obtained by
//! interpolating the `ζ ↔ parameter` map and reading off the parameter at
//! `ζ = ±Φ⁻¹(1 − α/2)`.

use std::collections::BTreeMap;

use crate::error::{MixedModelError, Result};
use crate::model::LinearMixedModel;
use crate::stats::spline::NaturalCubicSpline;

/// One row of a profile table.
///
/// Mirrors (a subset of) the row type used by `MixedModels.jl`. Every row
/// records the ζ value and the full parameter vector at the corresponding
/// conditional fit so that multiple parameters can share a single table.
#[derive(Debug, Clone)]
pub struct ProfileRow {
    /// Name of the parameter being profiled (e.g. `"σ"`, `"β1"`, `"θ3"`).
    pub p: String,
    /// Signed square root of the excess objective, `sign·√(obj − fmin)`.
    pub zeta: f64,
    /// Residual standard deviation at this row (either the profile target
    /// or the refit value when profiling something else).
    pub sigma: f64,
    /// Fixed-effects coefficients at this row.
    pub beta: Vec<f64>,
    /// θ vector at this row.
    pub theta: Vec<f64>,
}

impl ProfileRow {
    fn estimate_row(name: &str, m: &LinearMixedModel) -> Self {
        ProfileRow {
            p: name.to_string(),
            zeta: 0.0,
            sigma: m.sigma(),
            beta: m.beta().iter().cloned().collect(),
            theta: m.theta(),
        }
    }
}

/// Profile of a fitted linear mixed model.
///
/// Holds the raw table of `(ζ, parameter)` evaluations together with natural
/// cubic-spline interpolants in both directions. `fwd[p]` maps the value of
/// parameter `p` to `ζ`; `rev[p]` inverts the map.
#[derive(Debug, Clone)]
pub struct MixedModelProfile {
    /// All rows collected across every profiled parameter.
    pub tbl: Vec<ProfileRow>,
    /// Forward splines: parameter value → ζ.
    pub fwd: BTreeMap<String, NaturalCubicSpline>,
    /// Reverse splines: ζ → parameter value.
    pub rev: BTreeMap<String, NaturalCubicSpline>,
}

/// One row of a profile-likelihood confidence interval table.
#[derive(Debug, Clone)]
pub struct ConfintRow {
    pub parameter: String,
    pub estimate: f64,
    pub lower: f64,
    pub upper: f64,
}

impl MixedModelProfile {
    /// Compute profile-likelihood confidence intervals for every profiled
    /// parameter at the requested confidence level (default 0.95).
    ///
    /// Uses the reverse spline to map `ζ = ±z_{1-α/2}` back to the
    /// parameter scale.
    pub fn confint(&self, level: f64) -> Result<Vec<ConfintRow>> {
        if !(level > 0.0 && level < 1.0) {
            return Err(MixedModelError::InvalidArgument(format!(
                "confint level must be in (0,1); got {level}"
            )));
        }
        // For a single parameter, profile-likelihood CIs use χ²(1) quantiles
        // — the cutoff is sqrt of that quantile, i.e. the normal quantile.
        let cutoff = normal_inverse_cdf(0.5 + level / 2.0);

        let mut rows = Vec::with_capacity(self.rev.len());
        for (name, spline) in &self.rev {
            let estimate = spline.eval(0.0);
            let mut lower = spline.eval(-cutoff);
            let mut upper = spline.eval(cutoff);
            if lower > upper {
                std::mem::swap(&mut lower, &mut upper);
            }
            rows.push(ConfintRow {
                parameter: name.clone(),
                estimate,
                lower,
                upper,
            });
        }
        Ok(rows)
    }

    /// Retrieve a single confidence interval by parameter name.
    pub fn confint_for(&self, parameter: &str, level: f64) -> Result<ConfintRow> {
        self.confint(level)?
            .into_iter()
            .find(|r| r.parameter == parameter)
            .ok_or_else(|| {
                MixedModelError::InvalidArgument(format!("parameter {parameter} was not profiled"))
            })
    }

    /// Rows for a specific profiled parameter.
    pub fn rows_for(&self, parameter: &str) -> Vec<&ProfileRow> {
        self.tbl.iter().filter(|r| r.p == parameter).collect()
    }
}

// ===========================================================================
// σ profile
// ===========================================================================

/// Profile the residual standard deviation σ of a fitted linear mixed model.
///
/// The model must already be fitted with σ estimated from the data
/// (i.e. `optsum.sigma` is `None`). The profile walks σ outward from its
/// estimate, refitting θ at each trial value with σ held fixed, and
/// stops once `|ζ|` exceeds `threshold` (default 4).
///
/// On return the model is restored to its original fitted state.
pub fn profile_sigma(m: &mut LinearMixedModel, threshold: f64) -> Result<MixedModelProfile> {
    if m.optsum.sigma.is_some() {
        return Err(MixedModelError::InvalidArgument(format!(
            "Can't profile σ because it is already fixed at {:?}",
            m.optsum.sigma
        )));
    }
    if m.optsum.feval <= 0 {
        return Err(MixedModelError::InvalidArgument(
            "profile_sigma: model must be fitted first".into(),
        ));
    }

    // ----- Snapshot everything we will need to restore afterward -----
    let saved_reml = m.optsum.reml;
    let saved_final = m.optsum.final_params.clone();
    let saved_initial = m.optsum.initial.clone();
    let saved_fmin = m.optsum.fmin;
    let saved_feval = m.optsum.feval;
    let saved_finitial = m.optsum.finitial;
    let saved_fit_log = m.optsum.fit_log.clone();
    let saved_return = m.optsum.return_value.clone();

    let sigma_hat = m.sigma();
    let obj_hat = saved_fmin;
    let theta_hat = saved_final.clone();

    // Collect rows, starting at the estimate.
    let mut rows: Vec<ProfileRow> = Vec::new();
    rows.push(ProfileRow::estimate_row("σ", m));

    // Run: this closure refits the model at a candidate σ, records one row,
    // and returns ζ for the walker. It mutates `m` and `rows` and propagates
    // refit failures.
    let refit_at_sigma = |m: &mut LinearMixedModel,
                          rows: &mut Vec<ProfileRow>,
                          sigma_val: f64,
                          negative_side: bool|
     -> Result<f64> {
        m.optsum.sigma = Some(sigma_val);
        m.optsum.feval = 0;
        m.optsum.fit_log.clear();
        m.optsum.fmin = f64::INFINITY;
        m.optsum.finitial = f64::INFINITY;
        m.optsum.return_value.clear();
        // Warm-start from the previous fit's θ.
        m.optsum.initial = theta_hat.clone();
        // Safeguard initial θ within lower bounds.
        let lb = m.lower_bounds();
        for (v, &l) in m.optsum.initial.iter_mut().zip(lb.iter()) {
            if l.is_finite() && *v < l {
                *v = l;
            }
        }
        m.fit(saved_reml)?;
        let obj = m.optsum.fmin;
        let diff = (obj - obj_hat).max(0.0);
        let zeta = if negative_side {
            -diff.sqrt()
        } else {
            diff.sqrt()
        };
        rows.push(ProfileRow {
            p: "σ".to_string(),
            zeta,
            sigma: sigma_val,
            beta: m.beta().iter().cloned().collect(),
            theta: m.theta(),
        });
        Ok(zeta)
    };

    // _facsz: step factor chosen so that one step moves ζ by ~0.5.
    let facsz = {
        let probe = sigma_hat * (1.0_f64 / 64.0).exp();
        m.optsum.sigma = Some(probe);
        m.optsum.feval = 0;
        m.optsum.fit_log.clear();
        m.optsum.fmin = f64::INFINITY;
        m.optsum.finitial = f64::INFINITY;
        m.optsum.return_value.clear();
        m.optsum.initial = theta_hat.clone();
        let lb = m.lower_bounds();
        for (v, &l) in m.optsum.initial.iter_mut().zip(lb.iter()) {
            if l.is_finite() && *v < l {
                *v = l;
            }
        }
        m.fit(saved_reml)?;
        let obj = m.optsum.fmin;
        let delta = (obj - obj_hat).max(0.0);
        if delta <= 0.0 {
            // Extremely flat profile — fall back to a fixed 5 % step.
            1.05
        } else {
            ((1.0_f64 / 64.0) / (2.0 * delta.sqrt())).exp()
        }
    };
    debug_assert!(facsz.is_finite() && facsz > 1.0);

    // ----- Walk the negative ζ side (σ decreasing) -----
    let mut sigma_v = sigma_hat / facsz;
    let max_points = 60;
    let mut iter = 0;
    loop {
        iter += 1;
        if iter > max_points {
            break;
        }
        let zeta = refit_at_sigma(m, &mut rows, sigma_v, true)?;
        if zeta <= -threshold {
            break;
        }
        sigma_v /= facsz;
        if sigma_v <= 0.0 {
            break;
        }
    }

    // At this point `rows` has [estimate, neg1, neg2, ...]. Sort by σ so the
    // ζ and σ columns are monotone; the spline fitter needs strictly
    // increasing x.
    rows.sort_by(|a, b| a.sigma.partial_cmp(&b.sigma).unwrap());

    // ----- Walk the positive ζ side (σ increasing) -----
    let mut sigma_v = sigma_hat * facsz;
    let mut iter = 0;
    loop {
        iter += 1;
        if iter > max_points {
            break;
        }
        let zeta = refit_at_sigma(m, &mut rows, sigma_v, false)?;
        if zeta >= threshold {
            break;
        }
        sigma_v *= facsz;
    }

    // Re-sort now that we have pushed positive-side rows at the end.
    rows.sort_by(|a, b| a.sigma.partial_cmp(&b.sigma).unwrap());

    // Deduplicate any rows sharing the same σ (can happen if facsz ≈ 1).
    rows.dedup_by(|a, b| (a.sigma - b.sigma).abs() < 1e-14);

    // ----- Restore the original model state -----
    m.optsum.sigma = None;
    m.optsum.reml = saved_reml;
    m.optsum.initial = saved_initial;
    m.optsum.final_params = saved_final.clone();
    m.optsum.fit_log = saved_fit_log;
    m.optsum.fmin = saved_fmin;
    m.optsum.feval = saved_feval;
    m.optsum.finitial = saved_finitial;
    m.optsum.return_value = saved_return;
    m.set_theta(&saved_final)?;
    m.update_l()?;

    // ----- Build splines -----
    let sigmas: Vec<f64> = rows.iter().map(|r| r.sigma).collect();
    let zetas: Vec<f64> = rows.iter().map(|r| r.zeta).collect();
    if sigmas.len() < 3 {
        return Err(MixedModelError::Optimization(format!(
            "profile_sigma: only {} evaluations produced — try a larger threshold",
            sigmas.len()
        )));
    }

    let fwd = NaturalCubicSpline::fit(&sigmas, &zetas)?;
    // Reverse spline requires ζ to be strictly increasing; because rows are
    // sorted by σ and the profile is monotone in σ this should hold, but we
    // defensively check.
    for w in zetas.windows(2) {
        if !(w[1] > w[0]) {
            return Err(MixedModelError::Optimization(
                "profile_sigma: ζ(σ) table is not strictly monotone — refusing to invert".into(),
            ));
        }
    }
    let rev = NaturalCubicSpline::fit(&zetas, &sigmas)?;

    let mut fwd_map = BTreeMap::new();
    fwd_map.insert("σ".to_string(), fwd);
    let mut rev_map = BTreeMap::new();
    rev_map.insert("σ".to_string(), rev);

    Ok(MixedModelProfile {
        tbl: rows,
        fwd: fwd_map,
        rev: rev_map,
    })
}

/// Public entry point matching the shape of `MixedModels.jl::profile`.
///
/// Currently only profiles σ; later revisions will add β, θ, and
/// variance-component profiles.
pub fn profile(m: &mut LinearMixedModel) -> Result<MixedModelProfile> {
    profile_sigma(m, 4.0)
}

// ===========================================================================
// Normal quantile helper (Beasley-Springer-Moro style approximation)
// ===========================================================================
//
// We only need this for the confidence-interval cutoff `Φ⁻¹(1 − α/2)`, where
// the typical inputs are 0.975, 0.995, 0.95, etc. A short rational
// approximation is accurate to ~1e-7 in the region we care about and avoids
// adding a statrs dependency just for this.

fn normal_inverse_cdf(p: f64) -> f64 {
    // Beasley-Springer / Moro algorithm.
    // Reference: Moro, "The Full Monte", Risk, 1995.
    const A: [f64; 4] = [
        2.50662823884,
        -18.61500062529,
        41.39119773534,
        -25.44106049637,
    ];
    const B: [f64; 4] = [
        -8.47351093090,
        23.08336743743,
        -21.06224101826,
        3.13082909833,
    ];
    const C: [f64; 9] = [
        0.3374754822726147,
        0.9761690190917186,
        0.1607979714918209,
        0.0276438810333863,
        0.0038405729373609,
        0.0003951896511919,
        0.0000321767881768,
        0.0000002888167364,
        0.0000003960315187,
    ];
    let u = p - 0.5;
    if u.abs() < 0.42 {
        let r = u * u;
        u * (((A[3] * r + A[2]) * r + A[1]) * r + A[0])
            / ((((B[3] * r + B[2]) * r + B[1]) * r + B[0]) * r + 1.0)
    } else {
        let r = if u < 0.0 { p } else { 1.0 - p };
        let r = (-r.ln()).ln();
        let x = C[0]
            + r * (C[1]
                + r * (C[2]
                    + r * (C[3] + r * (C[4] + r * (C[5] + r * (C[6] + r * (C[7] + r * C[8])))))));
        if u < 0.0 {
            -x
        } else {
            x
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formula::parse_formula;
    use crate::model::data::DataFrame;

    #[test]
    fn normal_quantile_matches_known_values() {
        // Standard two-sided cutoffs.
        assert!((normal_inverse_cdf(0.975) - 1.959963984540054).abs() < 1e-6);
        assert!((normal_inverse_cdf(0.995) - 2.5758293035489004).abs() < 1e-6);
        assert!((normal_inverse_cdf(0.5) - 0.0).abs() < 1e-10);
    }

    fn dyestuff_fixture() -> DataFrame {
        let yields: Vec<f64> = vec![
            1545.0, 1440.0, 1440.0, 1520.0, 1580.0, //
            1540.0, 1555.0, 1490.0, 1560.0, 1495.0, //
            1595.0, 1550.0, 1605.0, 1510.0, 1560.0, //
            1445.0, 1440.0, 1595.0, 1465.0, 1545.0, //
            1595.0, 1630.0, 1515.0, 1635.0, 1625.0, //
            1520.0, 1455.0, 1450.0, 1480.0, 1445.0, //
        ];
        let batches: Vec<String> = "ABCDEF"
            .chars()
            .flat_map(|c| std::iter::repeat_n(c.to_string(), 5))
            .collect();
        let mut df = DataFrame::new();
        df.add_numeric("yield", yields);
        df.add_categorical("batch", batches);
        df
    }

    #[test]
    fn profile_sigma_dyestuff_returns_consistent_table() {
        let data = dyestuff_fixture();
        let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap(); // ML fit
        let sigma_hat = model.sigma();
        let fmin = model.optsum.fmin;
        let theta_hat = model.theta();

        let pr = profile_sigma(&mut model, 4.0).expect("σ profile should succeed");

        // There should be at least the estimate row plus points on both sides.
        assert!(pr.tbl.len() >= 5, "got {} rows", pr.tbl.len());
        assert!(pr.tbl.iter().any(|r| r.zeta < -0.5));
        assert!(pr.tbl.iter().any(|r| r.zeta > 0.5));

        // Estimate row should have ζ ≈ 0 at σ ≈ σ̂.
        let at_estimate = pr
            .tbl
            .iter()
            .min_by(|a, b| a.zeta.abs().partial_cmp(&b.zeta.abs()).unwrap())
            .unwrap();
        assert!(at_estimate.zeta.abs() < 1e-6);
        assert!((at_estimate.sigma - sigma_hat).abs() / sigma_hat < 1e-3);

        // ζ is monotone in σ.
        let rows_sigma: Vec<_> = pr.rows_for("σ").into_iter().collect();
        for w in rows_sigma.windows(2) {
            assert!(
                w[1].zeta > w[0].zeta - 1e-9,
                "zeta not monotone: {} -> {}",
                w[0].zeta,
                w[1].zeta
            );
        }

        // Model should be restored: σ free, same theta, same fmin.
        assert!(model.optsum.sigma.is_none());
        assert!((model.optsum.fmin - fmin).abs() < 1e-8);
        let theta_restored = model.theta();
        assert_eq!(theta_restored.len(), theta_hat.len());
        for (a, b) in theta_restored.iter().zip(theta_hat.iter()) {
            assert!((a - b).abs() < 1e-8);
        }

        // 95 % CI should bracket the estimate and be reasonably tight.
        let ci = pr.confint(0.95).unwrap();
        assert_eq!(ci.len(), 1);
        let sigma_ci = &ci[0];
        assert_eq!(sigma_ci.parameter, "σ");
        assert!(sigma_ci.lower < sigma_hat);
        assert!(sigma_ci.upper > sigma_hat);
        // Sanity: the CI shouldn't span more than a factor of 3 either way.
        assert!(sigma_ci.lower > sigma_hat / 3.0);
        assert!(sigma_ci.upper < sigma_hat * 3.0);
    }
}
