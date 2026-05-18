//! Profile-likelihood confidence intervals for linear mixed models.
//!
//! This is a partial port of `MixedModels.jl/src/profile/`. The current
//! scope covers the residual-scale profile (σ), θ profiles, and ML fixed-effect
//! β profiles together
//! with the shared [`MixedModelProfile`] container and
//! [`MixedModelProfile::confint`].
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

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

use crate::error::{MixedModelError, Result};
use crate::model::traits::MixedModelFit;
use crate::model::LinearMixedModel;
use crate::stats::spline::NaturalCubicSpline;

/// Stable schema name for serialized profile-likelihood CI payloads.
pub const PROFILE_LIKELIHOOD_CI_SCHEMA: &str = "mixedmodels.profile_likelihood_ci";
/// Stable schema version for serialized profile-likelihood CI payloads.
pub const PROFILE_LIKELIHOOD_CI_SCHEMA_VERSION: &str = "1.0.0";

/// One row of a profile table.
///
/// Mirrors (a subset of) the row type used by `MixedModels.jl`. Every row
/// records the ζ value and the full parameter vector at the corresponding
/// conditional fit so that multiple parameters can share a single table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfintRow {
    /// Parameter name.
    pub parameter: String,
    /// Fitted estimate.
    pub estimate: f64,
    /// Lower confidence limit.
    pub lower: f64,
    /// Upper confidence limit.
    pub upper: f64,
}

/// One serializable profile-likelihood CI row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProfileLikelihoodCiRow {
    /// Parameter name.
    pub parameter: String,
    /// Fitted estimate.
    pub estimate: f64,
    /// Lower confidence limit.
    pub lower: f64,
    /// Upper confidence limit.
    pub upper: f64,
    /// Confidence level used to compute the interval.
    pub level: f64,
    /// Interval method label.
    pub method: String,
    /// Regularity note for the interval.
    pub regularity: String,
    /// Whether the lower limit was clamped at a nonnegative boundary.
    pub boundary_clamped_lower: bool,
}

/// Serializable profile-likelihood CI payload for R and other bindings.
///
/// The raw profile table is included, but spline interpolants are not part of
/// the wire contract. Consumers that need intervals should read `intervals`;
/// consumers that need diagnostics can inspect `profile_rows`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProfileLikelihoodCiPayload {
    /// Stable schema name.
    pub schema_name: String,
    /// Stable schema version.
    pub schema_version: String,
    /// Confidence level used for all intervals.
    pub level: f64,
    /// Fit criterion used by the profiled model.
    pub fit_criterion: String,
    /// Computed profile-likelihood intervals.
    pub intervals: Vec<ProfileLikelihoodCiRow>,
    /// Raw profile rows retained for diagnostics.
    pub profile_rows: Vec<ProfileRow>,
    /// Reader-facing caveats and interpretation notes.
    pub notes: Vec<String>,
}

impl ProfileLikelihoodCiPayload {
    /// Serialize this payload to compact JSON.
    pub fn to_json(&self) -> std::result::Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Serialize this payload to pretty-printed JSON.
    pub fn to_json_pretty(&self) -> std::result::Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize a profile-likelihood CI payload from JSON.
    pub fn from_json(input: &str) -> std::result::Result<Self, serde_json::Error> {
        serde_json::from_str(input)
    }
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
            let estimate = self
                .profile_estimate(name)
                .unwrap_or_else(|| spline.eval(0.0));

            // The reverse spline maps ζ → parameter; its knots span only the
            // ζ range the profile walker actually reached. Evaluating it at
            // ±cutoff when the walker stopped early (boundary or max_points)
            // would *extrapolate*, fabricating a CI bound no refit supports —
            // a fake statistic. Detect that and report the unsupported bound
            // as `NaN` (lme4's `confint` likewise returns `NA` for a
            // truncated profile) instead of a silently-extrapolated number.
            let zk = spline.knots_x();
            let tol = 1e-9;
            let (zmin, zmax) = (
                zk.first().copied().unwrap_or(f64::NEG_INFINITY),
                zk.last().copied().unwrap_or(f64::INFINITY),
            );
            let lower_supported = -cutoff >= zmin - tol;
            let upper_supported = cutoff <= zmax + tol;
            let touches_zero = self.profile_touches_nonnegative_boundary(name);

            let mut lower = if lower_supported {
                spline.eval(-cutoff)
            } else if touches_zero {
                // Not extrapolation: the grid is short on the lower side
                // precisely because the parameter cannot go below its
                // nonnegative boundary, which the profile reached. The
                // lower limit is legitimately that boundary.
                0.0
            } else {
                f64::NAN
            };
            let mut upper = if upper_supported {
                spline.eval(cutoff)
            } else {
                f64::NAN
            };
            if lower > upper {
                std::mem::swap(&mut lower, &mut upper);
            }
            if lower < 0.0 && touches_zero {
                lower = 0.0;
            }
            // NaN comparisons are false, so an undetermined (NaN) bound
            // neither trips this guard nor is falsely accepted — it is
            // surfaced verbatim as the honest "not determined" signal.
            if lower > estimate || upper < estimate {
                return Err(MixedModelError::Optimization(format!(
                    "confint for {name}: profile interval [{lower}, {upper}] does not bracket estimate {estimate}"
                )));
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

    /// Build a serializable profile-likelihood CI payload.
    pub fn confint_payload(&self, level: f64, reml: bool) -> Result<ProfileLikelihoodCiPayload> {
        let intervals = self
            .confint(level)?
            .into_iter()
            .map(|row| {
                let boundary_clamped_lower =
                    row.lower == 0.0 && self.profile_touches_nonnegative_boundary(&row.parameter);
                ProfileLikelihoodCiRow {
                    parameter: row.parameter,
                    estimate: row.estimate,
                    lower: row.lower,
                    upper: row.upper,
                    level,
                    method: "profile_likelihood".to_string(),
                    regularity: if boundary_clamped_lower {
                        "nonnegative_parameter_boundary_clamped".to_string()
                    } else {
                        "regular_profile_likelihood".to_string()
                    },
                    boundary_clamped_lower,
                }
            })
            .collect();

        let mut notes = vec![
            "profile-likelihood intervals are computed by spline inversion of signed-root deviance values".to_string(),
            "profile rows are serialized for diagnostics; spline coefficients are intentionally not part of the wire contract".to_string(),
        ];
        if reml {
            notes.push(
                "REML profile payloads omit fixed-effect beta profiles; beta profile intervals require ML fits in this contract"
                    .to_string(),
            );
        }

        Ok(ProfileLikelihoodCiPayload {
            schema_name: PROFILE_LIKELIHOOD_CI_SCHEMA.to_string(),
            schema_version: PROFILE_LIKELIHOOD_CI_SCHEMA_VERSION.to_string(),
            level,
            fit_criterion: if reml { "REML" } else { "ML" }.to_string(),
            intervals,
            profile_rows: self.tbl.clone(),
            notes,
        })
    }

    /// Rows for a specific profiled parameter.
    pub fn rows_for(&self, parameter: &str) -> Vec<&ProfileRow> {
        self.tbl.iter().filter(|r| r.p == parameter).collect()
    }

    fn profile_touches_nonnegative_boundary(&self, parameter: &str) -> bool {
        let Some(values) = self.parameter_values(parameter) else {
            return false;
        };
        values.iter().all(|value| *value >= -1e-12)
            && values.iter().any(|value| value.abs() < 1e-10)
    }

    fn profile_estimate(&self, parameter: &str) -> Option<f64> {
        self.tbl
            .iter()
            .filter(|row| row.p == parameter)
            .min_by(|left, right| left.zeta.abs().partial_cmp(&right.zeta.abs()).unwrap())
            .and_then(|row| profile_row_parameter_value(row, parameter))
    }

    fn parameter_values(&self, parameter: &str) -> Option<Vec<f64>> {
        if parameter == "σ" {
            return Some(
                self.rows_for(parameter)
                    .into_iter()
                    .map(|row| row.sigma)
                    .collect(),
            );
        }
        if let Some(index) = parameter_index(parameter, 'β') {
            return Some(
                self.rows_for(parameter)
                    .into_iter()
                    .map(|row| row.beta[index])
                    .collect(),
            );
        }
        if let Some(index) = parameter_index(parameter, 'θ') {
            return Some(
                self.rows_for(parameter)
                    .into_iter()
                    .map(|row| row.theta[index])
                    .collect(),
            );
        }
        None
    }
}

fn parameter_index(parameter: &str, prefix: char) -> Option<usize> {
    parameter
        .strip_prefix(prefix)?
        .parse::<usize>()
        .ok()?
        .checked_sub(1)
}

fn profile_row_parameter_value(row: &ProfileRow, parameter: &str) -> Option<f64> {
    if parameter == "σ" {
        return Some(row.sigma);
    }
    if let Some(index) = parameter_index(parameter, 'β') {
        return row.beta.get(index).copied();
    }
    if let Some(index) = parameter_index(parameter, 'θ') {
        return row.theta.get(index).copied();
    }
    None
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
    let sigma_min = 1e-6 * sigma_hat;
    let mut sigma_v = sigma_hat / facsz;
    let max_points = 60;
    let mut iter = 0;
    loop {
        iter += 1;
        if iter > max_points {
            break;
        }
        if sigma_v <= sigma_min {
            break;
        }
        let zeta = refit_at_sigma(m, &mut rows, sigma_v, true)?;
        if zeta <= -threshold {
            break;
        }
        let next_sigma = sigma_v / facsz;
        if next_sigma <= sigma_min {
            break;
        }
        sigma_v = next_sigma;
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

/// Profile one θ covariance parameter of a fitted linear mixed model.
///
/// The target θ coordinate is fixed at each profile point while the remaining
/// θ coordinates are conditionally optimized by a bounded coordinate search.
/// β and σ remain profiled by the model objective. This is intentionally
/// conservative but gives a real one-coordinate profile for scalar and
/// multi-θ models without introducing a second optimizer stack.
pub fn profile_theta(
    m: &mut LinearMixedModel,
    index: usize,
    threshold: f64,
) -> Result<MixedModelProfile> {
    let n_theta = m.n_theta();
    if index >= n_theta {
        return Err(MixedModelError::InvalidArgument(format!(
            "profile_theta index {index} is out of bounds for {n_theta} θ parameter(s)"
        )));
    }
    if m.optsum.feval <= 0 {
        return Err(MixedModelError::InvalidArgument(
            "profile_theta: model must be fitted first".into(),
        ));
    }

    let parameter = format!("θ{}", index + 1);
    let theta_hat_vector = m.theta();
    let theta_hat = theta_hat_vector[index];
    let obj_hat = m.optsum.fmin;
    let saved_theta = theta_hat_vector.clone();
    let saved_fmin = m.optsum.fmin;
    let lower_bounds = m.lower_bounds();

    let lower = lower_bounds
        .get(index)
        .copied()
        .filter(|value| value.is_finite())
        .unwrap_or(f64::NEG_INFINITY);
    let min_step = (theta_hat.abs() * 1e-8).max(1e-10);

    let mut rows = vec![ProfileRow::estimate_row(&parameter, m)];
    let mut evaluate = |m: &mut LinearMixedModel,
                        fixed_value: f64,
                        start: &mut Vec<f64>,
                        negative_side: bool|
     -> Result<f64> {
        let (conditional_theta, obj) =
            optimize_theta_profile_point(m, index, fixed_value, start, &lower_bounds)?;
        *start = conditional_theta;
        let diff = (obj - obj_hat).max(0.0);
        let zeta = if negative_side {
            -diff.sqrt()
        } else {
            diff.sqrt()
        };
        rows.push(ProfileRow {
            p: parameter.clone(),
            zeta,
            sigma: m.sigma(),
            beta: m.beta().iter().cloned().collect(),
            theta: m.theta(),
        });
        Ok(zeta)
    };

    let facsz = {
        let probe = if theta_hat.abs() > min_step {
            theta_hat * (1.0_f64 / 64.0).exp()
        } else {
            theta_hat + 0.05
        };
        let probe_start = theta_hat_vector.clone();
        let (_, obj) = optimize_theta_profile_point(m, index, probe, &probe_start, &lower_bounds)?;
        theta_profile_step_factor_from_probe(theta_hat, probe, obj_hat, obj)
    };

    let max_points = 60;
    let mut negative_start = theta_hat_vector.clone();
    if theta_hat > lower + min_step {
        let mut theta_v = next_theta_profile_value(theta_hat, facsz, lower, true, min_step);
        let mut iter = 0;
        loop {
            iter += 1;
            if iter > max_points {
                break;
            }
            let zeta = evaluate(m, theta_v, &mut negative_start, true)?;
            if zeta <= -threshold || theta_v <= lower + min_step {
                break;
            }
            let next = next_theta_profile_value(theta_v, facsz, lower, true, min_step);
            if (theta_v - next).abs() <= min_step {
                break;
            }
            theta_v = next;
        }
    }

    let mut positive_start = theta_hat_vector.clone();
    let mut theta_v = next_theta_profile_value(theta_hat, facsz, lower, false, min_step);
    let mut iter = 0;
    loop {
        iter += 1;
        if iter > max_points {
            break;
        }
        let zeta = evaluate(m, theta_v, &mut positive_start, false)?;
        if zeta >= threshold {
            break;
        }
        theta_v = next_theta_profile_value(theta_v, facsz, lower, false, min_step);
    }

    rows.sort_by(|a, b| a.theta[index].partial_cmp(&b.theta[index]).unwrap());
    rows.dedup_by(|a, b| (a.theta[index] - b.theta[index]).abs() < 1e-14);

    m.set_theta(&saved_theta)?;
    m.update_l()?;
    m.optsum.fmin = saved_fmin;

    let mut fwd_map = BTreeMap::new();
    let mut rev_map = BTreeMap::new();
    add_profile_splines(&parameter, &rows, &mut fwd_map, &mut rev_map, |row| {
        row.theta[index]
    })?;

    Ok(MixedModelProfile {
        tbl: rows,
        fwd: fwd_map,
        rev: rev_map,
    })
}

/// Profile the single θ covariance parameter of a fitted linear mixed model.
pub fn profile_theta_scalar(m: &mut LinearMixedModel, threshold: f64) -> Result<MixedModelProfile> {
    if m.n_theta() != 1 {
        return Err(MixedModelError::InvalidArgument(format!(
            "profile_theta_scalar requires exactly one θ parameter; model has {}",
            m.n_theta()
        )));
    }
    profile_theta(m, 0, threshold)
}

fn next_theta_profile_value(
    current: f64,
    factor: f64,
    lower: f64,
    negative_side: bool,
    min_step: f64,
) -> f64 {
    if negative_side {
        if current.abs() > min_step {
            (current / factor).max(lower)
        } else {
            (current - min_step).max(lower)
        }
    } else if current.abs() > min_step {
        current * factor
    } else {
        current + min_step.max(0.05)
    }
}

fn theta_profile_step_factor_from_probe(
    theta_hat: f64,
    probe: f64,
    obj_hat: f64,
    obj_probe: f64,
) -> f64 {
    const TARGET_ZETA_STEP: f64 = 0.5;
    const FALLBACK_FACTOR: f64 = 1.05;
    const MIN_FACTOR: f64 = 1.000_001;
    const MAX_FACTOR: f64 = 1.25;

    let zeta_step = (obj_probe - obj_hat).max(0.0).sqrt();
    if !zeta_step.is_finite() || zeta_step <= 0.0 {
        return FALLBACK_FACTOR;
    }

    // Choose a multiplicative θ step whose local linearized ζ increment is
    // about 0.5.  The older raw-θ distance heuristic could not shrink below
    // 1.01, which underpopulated sharply curved profiles.
    let probe_log_step = if theta_hat.abs() > 0.0 && probe > 0.0 {
        (probe / theta_hat).abs().ln().abs()
    } else {
        0.0
    };
    let candidate = if probe_log_step.is_finite() && probe_log_step > 0.0 {
        (probe_log_step * TARGET_ZETA_STEP / zeta_step).exp()
    } else {
        let probe_step = (probe - theta_hat).abs();
        if !probe_step.is_finite() || probe_step <= 0.0 {
            FALLBACK_FACTOR
        } else {
            (probe_step * TARGET_ZETA_STEP / zeta_step).exp()
        }
    };

    if candidate.is_finite() {
        candidate.clamp(MIN_FACTOR, MAX_FACTOR)
    } else {
        FALLBACK_FACTOR
    }
}

fn optimize_theta_profile_point(
    m: &mut LinearMixedModel,
    fixed_index: usize,
    fixed_value: f64,
    start: &[f64],
    lower_bounds: &[f64],
) -> Result<(Vec<f64>, f64)> {
    let n_theta = m.n_theta();
    let mut best_theta = start.to_vec();
    if best_theta.len() != n_theta {
        return Err(MixedModelError::DimensionMismatch(format!(
            "profile optimizer start has length {}, expected {n_theta}",
            best_theta.len()
        )));
    }
    best_theta[fixed_index] = fixed_value;
    for (idx, value) in best_theta.iter_mut().enumerate() {
        if idx == fixed_index {
            continue;
        }
        if let Some(lower) = lower_bounds
            .get(idx)
            .copied()
            .filter(|value| value.is_finite())
        {
            if *value < lower {
                *value = lower;
            }
        }
    }

    let mut best_obj = m.objective_at(&best_theta)?;
    let mut steps = best_theta
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            if idx == fixed_index {
                0.0
            } else {
                (value.abs() * 0.1).max(0.05)
            }
        })
        .collect::<Vec<_>>();

    for _ in 0..80 {
        let mut improved = false;
        for idx in 0..n_theta {
            if idx == fixed_index {
                continue;
            }
            for direction in [1.0, -1.0] {
                let mut candidate = best_theta.clone();
                candidate[idx] += direction * steps[idx];
                if let Some(lower) = lower_bounds
                    .get(idx)
                    .copied()
                    .filter(|value| value.is_finite())
                {
                    if candidate[idx] < lower {
                        candidate[idx] = lower;
                    }
                }
                if (candidate[idx] - best_theta[idx]).abs() < 1e-12 {
                    continue;
                }
                let obj = m.objective_at(&candidate)?;
                if obj + 1e-8 < best_obj {
                    best_obj = obj;
                    best_theta = candidate;
                    improved = true;
                }
            }
        }
        if !improved {
            let mut max_step = 0.0_f64;
            for (idx, step) in steps.iter_mut().enumerate() {
                if idx == fixed_index {
                    continue;
                }
                *step *= 0.5;
                max_step = max_step.max(*step);
            }
            if max_step < 1e-5 {
                break;
            }
        }
    }

    best_obj = m.objective_at(&best_theta)?;
    Ok((best_theta, best_obj))
}

// ===========================================================================
// β profile
// ===========================================================================

/// Profile one active fixed-effect coefficient of an ML-fitted linear mixed
/// model.
///
/// The target β coordinate is fixed at each profile point while the remaining
/// fixed effects, θ, and σ are profiled. The constrained objective is computed
/// from the dense marginal covariance `V = I + ZΛΛ'Z'`, so this is deliberately
/// limited to unweighted ML fits until the blocked PLS path exposes the same
/// fixed-β constraint.
pub fn profile_beta(
    m: &mut LinearMixedModel,
    index: usize,
    threshold: f64,
) -> Result<MixedModelProfile> {
    let p = m.feterm.rank;
    if index >= p {
        return Err(MixedModelError::InvalidArgument(format!(
            "profile_beta index {index} is out of bounds for {p} active fixed effect(s)"
        )));
    }
    if m.optsum.feval <= 0 {
        return Err(MixedModelError::InvalidArgument(
            "profile_beta: model must be fitted first".into(),
        ));
    }
    if m.optsum.reml {
        return Err(MixedModelError::InvalidArgument(
            "profile_beta currently requires an ML fit; refit with REML=false".into(),
        ));
    }
    if !m.sqrtwts.is_empty() {
        return Err(MixedModelError::InvalidArgument(
            "profile_beta currently does not support observation weights".into(),
        ));
    }

    let parameter = format!("β{}", index + 1);
    let beta_hat_vector = m.beta().iter().cloned().collect::<Vec<_>>();
    let beta_hat = beta_hat_vector[index];
    let theta_hat_vector = m.theta();
    let saved_theta = theta_hat_vector.clone();
    let saved_fmin = m.optsum.fmin;
    let lower_bounds = m.lower_bounds();
    let (_, _, obj_hat) = fixed_beta_profile_components(m, &theta_hat_vector, index, beta_hat)?;

    let se = m.stderror().get(index).copied().unwrap_or(f64::NAN);
    let initial_step = if se.is_finite() && se > 0.0 {
        0.35 * se
    } else {
        (beta_hat.abs() * 0.05).max(0.1)
    };

    let mut rows = vec![ProfileRow::estimate_row(&parameter, m)];
    let mut evaluate = |m: &mut LinearMixedModel,
                        fixed_value: f64,
                        start: &mut Vec<f64>,
                        negative_side: bool|
     -> Result<f64> {
        let (conditional_theta, beta, sigma, obj) =
            optimize_beta_profile_point(m, index, fixed_value, start, &lower_bounds)?;
        *start = conditional_theta.clone();
        let diff = (obj - obj_hat).max(0.0);
        let zeta = if negative_side {
            -diff.sqrt()
        } else {
            diff.sqrt()
        };
        rows.push(ProfileRow {
            p: parameter.clone(),
            zeta,
            sigma,
            beta,
            theta: conditional_theta,
        });
        Ok(zeta)
    };

    let max_points = 60;
    let step_growth = 1.35;

    let mut negative_start = theta_hat_vector.clone();
    let mut distance = initial_step;
    for _ in 0..max_points {
        let zeta = evaluate(m, beta_hat - distance, &mut negative_start, true)?;
        if zeta <= -threshold {
            break;
        }
        distance *= step_growth;
    }

    let mut positive_start = theta_hat_vector.clone();
    let mut distance = initial_step;
    for _ in 0..max_points {
        let zeta = evaluate(m, beta_hat + distance, &mut positive_start, false)?;
        if zeta >= threshold {
            break;
        }
        distance *= step_growth;
    }

    rows.sort_by(|a, b| a.beta[index].partial_cmp(&b.beta[index]).unwrap());
    rows.dedup_by(|a, b| (a.beta[index] - b.beta[index]).abs() < 1e-12);

    m.set_theta(&saved_theta)?;
    m.update_l()?;
    m.optsum.fmin = saved_fmin;

    let mut fwd_map = BTreeMap::new();
    let mut rev_map = BTreeMap::new();
    add_profile_splines(&parameter, &rows, &mut fwd_map, &mut rev_map, |row| {
        row.beta[index]
    })?;

    Ok(MixedModelProfile {
        tbl: rows,
        fwd: fwd_map,
        rev: rev_map,
    })
}

/// Profile every active fixed-effect coefficient of an ML-fitted LMM.
pub fn profile_betas(m: &mut LinearMixedModel, threshold: f64) -> Result<MixedModelProfile> {
    let p = m.feterm.rank;
    let mut tbl = Vec::new();
    let mut fwd = BTreeMap::new();
    let mut rev = BTreeMap::new();
    for index in 0..p {
        let beta = profile_beta(m, index, threshold)?;
        tbl.extend(beta.tbl);
        fwd.extend(beta.fwd);
        rev.extend(beta.rev);
    }
    Ok(MixedModelProfile { tbl, fwd, rev })
}

fn optimize_beta_profile_point(
    m: &mut LinearMixedModel,
    fixed_index: usize,
    fixed_value: f64,
    start: &[f64],
    lower_bounds: &[f64],
) -> Result<(Vec<f64>, Vec<f64>, f64, f64)> {
    let n_theta = m.n_theta();
    let mut best_theta = start.to_vec();
    if best_theta.len() != n_theta {
        return Err(MixedModelError::DimensionMismatch(format!(
            "profile_beta optimizer start has length {}, expected {n_theta}",
            best_theta.len()
        )));
    }
    for (idx, value) in best_theta.iter_mut().enumerate() {
        if let Some(lower) = lower_bounds
            .get(idx)
            .copied()
            .filter(|value| value.is_finite())
        {
            if *value < lower {
                *value = lower;
            }
        }
    }

    let (_, _, mut best_obj) =
        fixed_beta_profile_components(m, &best_theta, fixed_index, fixed_value)?;
    let mut steps = best_theta
        .iter()
        .map(|value| (value.abs() * 0.1).max(0.05))
        .collect::<Vec<_>>();

    for _ in 0..80 {
        let mut improved = false;
        for idx in 0..n_theta {
            for direction in [1.0, -1.0] {
                let mut candidate = best_theta.clone();
                candidate[idx] += direction * steps[idx];
                if let Some(lower) = lower_bounds
                    .get(idx)
                    .copied()
                    .filter(|value| value.is_finite())
                {
                    if candidate[idx] < lower {
                        candidate[idx] = lower;
                    }
                }
                if (candidate[idx] - best_theta[idx]).abs() < 1e-12 {
                    continue;
                }
                let (_, _, obj) =
                    fixed_beta_profile_components(m, &candidate, fixed_index, fixed_value)?;
                if obj + 1e-8 < best_obj {
                    best_obj = obj;
                    best_theta = candidate;
                    improved = true;
                }
            }
        }
        if !improved {
            let mut max_step = 0.0_f64;
            for step in &mut steps {
                *step *= 0.5;
                max_step = max_step.max(*step);
            }
            if max_step < 1e-5 {
                break;
            }
        }
    }

    let (best_beta, best_sigma, best_obj) =
        fixed_beta_profile_components(m, &best_theta, fixed_index, fixed_value)?;
    Ok((best_theta, best_beta, best_sigma, best_obj))
}

fn fixed_beta_profile_components(
    m: &mut LinearMixedModel,
    theta: &[f64],
    fixed_index: usize,
    fixed_value: f64,
) -> Result<(Vec<f64>, f64, f64)> {
    m.set_theta(theta)?;
    let v = marginal_relative_covariance(m);
    let chol = v.cholesky().ok_or_else(|| {
        MixedModelError::Optimization(
            "profile_beta: marginal covariance is not positive definite".into(),
        )
    })?;
    let x = m.feterm.full_rank_x().into_owned();
    let n = x.nrows();
    let p = x.ncols();
    if fixed_index >= p {
        return Err(MixedModelError::InvalidArgument(format!(
            "profile_beta index {fixed_index} is out of bounds for {p} active fixed effect(s)"
        )));
    }

    let adjusted = &m.y - x.column(fixed_index) * fixed_value;
    let free_p = p - 1;
    let mut beta = vec![0.0; p];
    beta[fixed_index] = fixed_value;
    let residual = if free_p == 0 {
        adjusted
    } else {
        let mut x_free = DMatrix::zeros(n, free_p);
        let mut free_col = 0;
        for col in 0..p {
            if col == fixed_index {
                continue;
            }
            x_free.set_column(free_col, &x.column(col));
            free_col += 1;
        }
        let vinv_x = chol.solve(&x_free);
        let adjusted_matrix = DMatrix::from_column_slice(n, 1, adjusted.as_slice());
        let vinv_y = chol.solve(&adjusted_matrix);
        let xt_vinv_x = x_free.transpose() * vinv_x;
        let xt_vinv_y = x_free.transpose() * vinv_y;
        let beta_free = xt_vinv_x.lu().solve(&xt_vinv_y).ok_or_else(|| {
            MixedModelError::Optimization(
                "profile_beta: constrained fixed-effects system is singular".into(),
            )
        })?;
        let mut free_col = 0;
        for col in 0..p {
            if col == fixed_index {
                continue;
            }
            beta[col] = beta_free[(free_col, 0)];
            free_col += 1;
        }
        adjusted - x_free * beta_free.column(0)
    };

    let residual_matrix = DMatrix::from_column_slice(n, 1, residual.as_slice());
    let vinv_residual = chol.solve(&residual_matrix);
    let pwrss = residual.dot(&vinv_residual.column(0)).max(0.0);
    let denom = n as f64;
    if pwrss <= 0.0 || !pwrss.is_finite() {
        return Err(MixedModelError::Optimization(format!(
            "profile_beta: invalid constrained pwrss {pwrss}"
        )));
    }
    let logdet_v = 2.0
        * chol
            .l()
            .diagonal()
            .iter()
            .map(|value| value.ln())
            .sum::<f64>();
    let objective = logdet_v + denom * (1.0 + (2.0 * std::f64::consts::PI * pwrss / denom).ln());
    let sigma = (pwrss / denom).sqrt();
    Ok((beta, sigma, objective))
}

fn marginal_relative_covariance(m: &LinearMixedModel) -> DMatrix<f64> {
    let n = m.dims.n;
    let mut v = DMatrix::<f64>::identity(n, n);
    for re in &m.reterms {
        let cov = &re.lambda * re.lambda.transpose();
        for i in 0..n {
            let level_i = re.refs[i];
            for j in 0..=i {
                if re.refs[j] != level_i {
                    continue;
                }
                let mut value = 0.0;
                for row in 0..re.vsize {
                    for col in 0..re.vsize {
                        value += re.z[(row, i)] * cov[(row, col)] * re.z[(col, j)];
                    }
                }
                v[(i, j)] += value;
                if i != j {
                    v[(j, i)] += value;
                }
            }
        }
    }
    v
}

/// Public entry point matching the shape of `MixedModels.jl::profile`.
///
/// Profiles σ and θ for fitted LMMs. For ML fits, active fixed-effect β
/// profiles are included as well. REML β profiles are deliberately omitted
/// until a certified REML fixed-effect profile contract exists.
pub fn profile(m: &mut LinearMixedModel) -> Result<MixedModelProfile> {
    let sigma = profile_sigma(m, 4.0)?;
    let n_theta = m.n_theta();
    let mut tbl = Vec::new();
    let mut fwd = BTreeMap::new();
    let mut rev = BTreeMap::new();
    tbl.extend(sigma.tbl);
    fwd.extend(sigma.fwd);
    rev.extend(sigma.rev);
    if !m.optsum.reml {
        let beta = profile_betas(m, 4.0)?;
        tbl.extend(beta.tbl);
        fwd.extend(beta.fwd);
        rev.extend(beta.rev);
    }
    for index in 0..n_theta {
        let theta = profile_theta(m, index, 4.0)?;
        tbl.extend(theta.tbl);
        fwd.extend(theta.fwd);
        rev.extend(theta.rev);
    }
    Ok(MixedModelProfile { tbl, fwd, rev })
}

/// Compute a serializable profile-likelihood CI payload for a fitted LMM.
pub fn profile_confint_payload(
    m: &mut LinearMixedModel,
    level: f64,
) -> Result<ProfileLikelihoodCiPayload> {
    let reml = m.optsum.reml;
    profile(m)?.confint_payload(level, reml)
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

fn add_profile_splines(
    parameter: &str,
    rows: &[ProfileRow],
    fwd_map: &mut BTreeMap<String, NaturalCubicSpline>,
    rev_map: &mut BTreeMap<String, NaturalCubicSpline>,
    value_of: impl Fn(&ProfileRow) -> f64,
) -> Result<()> {
    let values: Vec<f64> = rows.iter().map(value_of).collect();
    let zetas: Vec<f64> = rows.iter().map(|r| r.zeta).collect();
    if values.len() < 5 {
        return Err(MixedModelError::Optimization(format!(
            "profile_{parameter}: only {} evaluations produced — refusing sparse profile spline; try a larger threshold",
            values.len()
        )));
    }
    for w in zetas.windows(2) {
        if !(w[1] > w[0]) {
            return Err(MixedModelError::Optimization(format!(
                "profile_{parameter}: ζ table is not strictly monotone — refusing to invert"
            )));
        }
    }
    fwd_map.insert(
        parameter.to_string(),
        NaturalCubicSpline::fit(&values, &zetas)?,
    );
    rev_map.insert(
        parameter.to_string(),
        NaturalCubicSpline::fit(&zetas, &values)?,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use serde::Deserialize;

    use crate::datasets;
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
        df.add_numeric("yield", yields).unwrap();
        df.add_categorical("batch", batches).unwrap();
        df
    }

    fn random_slope_fixture() -> DataFrame {
        let mut y = Vec::new();
        let mut x = Vec::new();
        let mut g = Vec::new();
        for group in 0..8 {
            let intercept_offset = (group as f64 - 3.5) * 4.0;
            let slope_offset = ((group % 4) as f64 - 1.5) * 1.8;
            for day in 0..5 {
                let x_value = day as f64 - 2.0;
                let noise = ((group + day) % 3) as f64 - 1.0;
                y.push(100.0 + 8.0 * x_value + intercept_offset + slope_offset * x_value + noise);
                x.push(x_value);
                g.push(format!("G{group}"));
            }
        }
        let mut df = DataFrame::new();
        df.add_numeric("y", y).unwrap();
        df.add_numeric("x", x).unwrap();
        df.add_categorical("g", g).unwrap();
        df
    }

    #[test]
    fn confint_reports_nan_instead_of_extrapolating_past_profile_grid() {
        // Regression for audit 05·M1 / mote bd-01KRXCR3P1D3BMX7SREFCWJ1MM:
        // a CI whose cutoff lies beyond the computed ζ grid must NOT be a
        // silently-extrapolated finite number; it must be reported as NaN
        // (not determined), like lme4's NA for a truncated profile.
        let data = dyestuff_fixture();
        let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();
        let pr = profile_sigma(&mut model, 4.0).expect("σ profile");

        // A standard 95% level (cutoff ≈ 1.96, well inside the ±~4 grid)
        // must still yield finite, bracketed bounds.
        let row95 = pr.confint_for("σ", 0.95).expect("95% confint");
        assert!(
            row95.lower.is_finite() && row95.upper.is_finite(),
            "95% bounds must be finite, got [{}, {}]",
            row95.lower,
            row95.upper
        );

        // An extreme level whose cutoff Φ⁻¹(0.5 + level/2) lies strictly
        // past the computed reverse-spline ζ span, forcing what was
        // previously a silent extrapolation.
        let zmax = pr.rev["σ"].knots_x().last().copied().unwrap();
        let level = 1.0 - 1e-10;
        let cutoff = normal_inverse_cdf(0.5 + level / 2.0);
        assert!(
            cutoff > zmax,
            "test precondition: cutoff {cutoff} must exceed grid ζ_max {zmax}"
        );
        let row = pr.confint_for("σ", level).expect("extreme-level confint");
        // Dyestuff residual σ (~50) is far from 0, so its profile is
        // truncated on *both* sides at this absurd level — neither bound is
        // determined by any refit, so both must be NaN (honest) rather than
        // silently extrapolated finite numbers.
        assert!(
            row.lower.is_nan() && row.upper.is_nan(),
            "bounds past the ζ grid must be NaN (not extrapolated), got [{}, {}]",
            row.lower,
            row.upper
        );
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

    #[test]
    fn test_profile_sigma_clamp_no_walk_below_threshold() {
        let data = dyestuff_fixture();
        let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();
        let sigma_min = 1e-6 * model.sigma();

        let pr = profile_sigma(&mut model, 4.0).expect("σ profile should succeed");
        let min_profiled_sigma = pr
            .rows_for("σ")
            .into_iter()
            .map(|row| row.sigma)
            .fold(f64::INFINITY, f64::min);

        assert!(min_profiled_sigma > sigma_min);
    }

    #[test]
    fn test_profile_confint_brackets_estimate() {
        let mut rev = BTreeMap::new();
        rev.insert(
            "σ".to_string(),
            NaturalCubicSpline::fit(&[-2.0, 0.0, 2.0], &[10.0, 5.0, 6.0]).unwrap(),
        );
        let profile = MixedModelProfile {
            tbl: vec![
                ProfileRow {
                    p: "σ".to_string(),
                    zeta: -2.0,
                    sigma: 10.0,
                    beta: Vec::new(),
                    theta: Vec::new(),
                },
                ProfileRow {
                    p: "σ".to_string(),
                    zeta: 0.0,
                    sigma: 5.0,
                    beta: Vec::new(),
                    theta: Vec::new(),
                },
                ProfileRow {
                    p: "σ".to_string(),
                    zeta: 2.0,
                    sigma: 6.0,
                    beta: Vec::new(),
                    theta: Vec::new(),
                },
            ],
            fwd: BTreeMap::new(),
            rev,
        };

        let err = profile.confint(0.95).unwrap_err();
        assert!(err.to_string().contains("does not bracket estimate"));
    }

    #[test]
    fn test_profile_sigma_logspace_walk_consistent() {
        let data = dyestuff_fixture();
        let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let pr = profile_sigma(&mut model, 4.0).expect("σ profile should succeed");
        let mut negative_side_sigmas: Vec<f64> = pr
            .rows_for("σ")
            .into_iter()
            .filter(|row| row.zeta < -1e-8)
            .map(|row| row.sigma)
            .collect();
        negative_side_sigmas.sort_by(|a, b| b.partial_cmp(a).unwrap());

        if negative_side_sigmas.len() >= 3 {
            let ratios: Vec<f64> = negative_side_sigmas
                .windows(2)
                .map(|window| window[0] / window[1])
                .collect();
            for window in ratios.windows(2) {
                assert!((window[0] - window[1]).abs() <= 1e-10 * window[0].max(1.0));
            }
        }
    }

    #[test]
    fn test_profile_theta_dense_grid_on_curved_profile() {
        let theta_hat = 1.0;
        let probe = theta_hat * (1.0_f64 / 64.0).exp();
        let factor = theta_profile_step_factor_from_probe(theta_hat, probe, 100.0, 500.0);

        assert!(
            factor < 1.01,
            "high-curvature θ profiles must be allowed below the old 1.01 floor; got {factor}"
        );
        assert!(factor > 1.0);

        let zeta_probe = (500.0_f64 - 100.0).sqrt();
        let predicted_next_zeta = zeta_probe * factor.ln() / (probe / theta_hat).ln();
        assert!(
            (predicted_next_zeta - 0.5).abs() < 1e-6,
            "target ζ step should be about 0.5, got {predicted_next_zeta}"
        );
    }

    #[test]
    fn test_profile_theta_sparse_grid_emits_warning_or_errors() {
        let rows = vec![
            ProfileRow {
                p: "θ1".to_string(),
                zeta: -1.0,
                sigma: 1.0,
                beta: Vec::new(),
                theta: vec![0.9],
            },
            ProfileRow {
                p: "θ1".to_string(),
                zeta: 0.0,
                sigma: 1.0,
                beta: Vec::new(),
                theta: vec![1.0],
            },
            ProfileRow {
                p: "θ1".to_string(),
                zeta: 1.0,
                sigma: 1.0,
                beta: Vec::new(),
                theta: vec![1.1],
            },
        ];
        let mut fwd = BTreeMap::new();
        let mut rev = BTreeMap::new();

        let err =
            add_profile_splines("θ1", &rows, &mut fwd, &mut rev, |row| row.theta[0]).unwrap_err();

        assert!(err.to_string().contains("refusing sparse profile spline"));
        assert!(fwd.is_empty());
        assert!(rev.is_empty());
    }

    #[test]
    fn test_profile_theta_julia_parity_fixture_keeps_scalar_theta_supported() {
        let fixture = profile_parity_fixture();
        let case = fixture
            .cases
            .iter()
            .find(|case| case.id == "dyestuff_scalar_re_ml")
            .expect("dyestuff scalar θ parity case should be present");
        let expected = case.parameters.get("θ1").unwrap();
        let (data, _) = datasets::load(&case.dataset).unwrap();
        let formula = parse_formula(&case.rust_formula).unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(case.reml).unwrap();

        let pr = profile_theta(&mut model, 0, 4.0).expect("θ1 profile should succeed");
        let rows = pr.rows_for("θ1");
        assert!(
            rows.len() >= 8,
            "target-ζ θ profile should populate a dense grid, got {} rows",
            rows.len()
        );

        let actual = pr.confint_for("θ1", case.level).unwrap();
        assert!(
            (actual.estimate - expected.estimate).abs() <= 0.15 * expected.estimate.abs().max(1.0)
        );
        assert!((actual.lower - expected.lower).abs() <= 0.15 * expected.lower.abs().max(1.0));
        assert!((actual.upper - expected.upper).abs() <= 0.15 * expected.upper.abs().max(1.0));
    }

    #[test]
    fn profile_theta_scalar_dyestuff_returns_consistent_table() {
        let data = dyestuff_fixture();
        let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();
        let theta_hat = model.theta()[0];
        let fmin = model.optsum.fmin;

        let pr = profile_theta_scalar(&mut model, 4.0).expect("θ1 profile should succeed");

        assert!(pr.tbl.len() >= 5, "got {} rows", pr.tbl.len());
        assert!(pr.tbl.iter().any(|r| r.zeta < -0.5));
        assert!(pr.tbl.iter().any(|r| r.zeta > 0.5));
        assert!(pr.tbl.iter().all(|row| row.p == "θ1"));

        let at_estimate = pr
            .tbl
            .iter()
            .min_by(|a, b| a.zeta.abs().partial_cmp(&b.zeta.abs()).unwrap())
            .unwrap();
        assert!(at_estimate.zeta.abs() < 1e-6);
        assert!((at_estimate.theta[0] - theta_hat).abs() / theta_hat < 1e-3);

        for w in pr.rows_for("θ1").windows(2) {
            assert!(
                w[1].zeta > w[0].zeta - 1e-9,
                "zeta not monotone: {} -> {}",
                w[0].zeta,
                w[1].zeta
            );
        }

        assert!((model.theta()[0] - theta_hat).abs() < 1e-8);
        assert!((model.optsum.fmin - fmin).abs() < 1e-8);

        let ci = pr.confint(0.95).unwrap();
        assert_eq!(ci.len(), 1);
        let theta_ci = &ci[0];
        assert_eq!(theta_ci.parameter, "θ1");
        assert!(theta_ci.lower < theta_hat);
        assert!(theta_ci.upper > theta_hat);
        assert!(theta_ci.lower >= 0.0);
    }

    #[test]
    fn profile_theta_optimizes_remaining_theta_coordinates() {
        let data = random_slope_fixture();
        let formula = parse_formula("y ~ 1 + x + (1 + x || g)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();
        assert_eq!(model.n_theta(), 2);
        let theta_hat = model.theta();
        let fmin = model.optsum.fmin;

        let pr = profile_theta(&mut model, 1, 2.5).expect("θ2 profile should succeed");

        assert!(pr.tbl.len() >= 5, "got {} rows", pr.tbl.len());
        assert!(pr.tbl.iter().all(|row| row.p == "θ2"));
        assert!(pr.tbl.iter().any(|row| row.zeta < -0.5));
        assert!(pr.tbl.iter().any(|row| row.zeta > 0.5));
        assert!(pr.tbl.iter().any(|row| {
            (row.theta[1] - theta_hat[1]).abs() > 1e-4 && (row.theta[0] - theta_hat[0]).abs() > 1e-6
        }));

        for w in pr.rows_for("θ2").windows(2) {
            assert!(
                w[1].zeta > w[0].zeta - 1e-9,
                "zeta not monotone: {} -> {}",
                w[0].zeta,
                w[1].zeta
            );
        }

        let restored = model.theta();
        assert_eq!(restored.len(), theta_hat.len());
        for (actual, expected) in restored.iter().zip(theta_hat.iter()) {
            assert!((actual - expected).abs() < 1e-8);
        }
        assert!((model.optsum.fmin - fmin).abs() < 1e-8);

        let ci = pr.confint_for("θ2", 0.90).unwrap();
        assert!(ci.lower <= theta_hat[1]);
        assert!(ci.upper >= theta_hat[1]);
    }

    #[test]
    fn profile_beta_dyestuff_returns_constrained_ml_table() {
        let data = dyestuff_fixture();
        let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();
        let beta_hat = model.beta()[0];
        let theta_hat = model.theta();
        let fmin = model.optsum.fmin;
        let (_, sigma_at_hat, obj_at_hat) =
            fixed_beta_profile_components(&mut model, &theta_hat, 0, beta_hat).unwrap();
        assert!((obj_at_hat - fmin).abs() < 1e-6);
        assert!((sigma_at_hat - model.sigma()).abs() < 1e-6);

        let pr = profile_beta(&mut model, 0, 3.0).expect("β1 profile should succeed");

        assert!(pr.tbl.len() >= 5, "got {} rows", pr.tbl.len());
        assert!(pr.tbl.iter().all(|row| row.p == "β1"));
        assert!(pr.tbl.iter().any(|row| row.zeta < -0.5));
        assert!(pr.tbl.iter().any(|row| row.zeta > 0.5));

        let at_estimate = pr
            .tbl
            .iter()
            .min_by(|a, b| a.zeta.abs().partial_cmp(&b.zeta.abs()).unwrap())
            .unwrap();
        assert!(at_estimate.zeta.abs() < 1e-6);
        assert!((at_estimate.beta[0] - beta_hat).abs() < 1e-6);

        for w in pr.rows_for("β1").windows(2) {
            assert!(
                w[1].zeta > w[0].zeta - 1e-8,
                "zeta not monotone: {} -> {}",
                w[0].zeta,
                w[1].zeta
            );
        }

        let restored = model.theta();
        for (actual, expected) in restored.iter().zip(theta_hat.iter()) {
            assert!((actual - expected).abs() < 1e-8);
        }
        assert!((model.optsum.fmin - fmin).abs() < 1e-8);

        let ci = pr.confint_for("β1", 0.95).unwrap();
        assert!(ci.lower < beta_hat);
        assert!(ci.upper > beta_hat);
    }

    #[test]
    fn profile_beta_reml_is_explicitly_unavailable() {
        let data = dyestuff_fixture();
        let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(true).unwrap();

        let err = profile_beta(&mut model, 0, 3.0).unwrap_err();
        assert!(
            err.to_string().contains("requires an ML fit"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn profile_dyestuff_combines_sigma_and_scalar_theta() {
        let data = dyestuff_fixture();
        let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let pr = profile(&mut model).expect("combined profile should succeed");
        assert!(!pr.rows_for("σ").is_empty());
        assert!(!pr.rows_for("θ1").is_empty());
        assert!(!pr.rows_for("β1").is_empty());
        assert_eq!(profile_row_order(&pr), vec!["σ", "β1", "θ1"]);

        let ci = pr.confint(0.95).unwrap();
        let parameters = ci
            .iter()
            .map(|row| row.parameter.as_str())
            .collect::<Vec<_>>();
        assert!(parameters.contains(&"σ"));
        assert!(parameters.contains(&"θ1"));
        assert!(parameters.contains(&"β1"));
    }

    #[derive(Debug, Deserialize)]
    struct ProfileParityFixture {
        schema_version: String,
        source: String,
        cases: Vec<ProfileParityCase>,
    }

    #[derive(Debug, Deserialize)]
    struct ProfileParityCase {
        id: String,
        dataset: String,
        rust_formula: String,
        reml: bool,
        level: f64,
        #[serde(default)]
        rust_row_order: Vec<String>,
        #[serde(default)]
        parameters: BTreeMap<String, ProfileParityInterval>,
        #[serde(default)]
        unsupported_reason: Option<String>,
        #[serde(default)]
        slow_reason: Option<String>,
    }

    #[derive(Debug, Deserialize)]
    struct ProfileParityInterval {
        estimate: f64,
        lower: f64,
        upper: f64,
    }

    fn profile_parity_fixture() -> ProfileParityFixture {
        serde_json::from_str(include_str!(
            "../../tests/fixtures/profile_likelihood_julia_parity_v1.json"
        ))
        .expect("profile likelihood parity fixture should deserialize")
    }

    fn profile_row_order(pr: &MixedModelProfile) -> Vec<String> {
        let mut order = Vec::new();
        for row in &pr.tbl {
            if !order.iter().any(|name| name == &row.p) {
                order.push(row.p.clone());
            }
        }
        order
    }

    fn profile_parity_tolerance(parameter: &str, expected: f64) -> f64 {
        let scale = expected.abs().max(1.0);
        if parameter.starts_with('θ') {
            0.15 * scale
        } else {
            0.02 * scale
        }
    }

    #[test]
    fn profile_likelihood_julia_parity_fixture_is_versioned() {
        let fixture = profile_parity_fixture();
        assert_eq!(fixture.schema_version, "1.0.0");
        assert!(fixture.source.contains("MixedModels.jl"));
        assert!(
            fixture
                .cases
                .iter()
                .any(|case| case.id == "kb07_scalar_crossed_reml"
                    && case.unsupported_reason.is_some())
        );
        assert!(fixture
            .cases
            .iter()
            .any(|case| case.id == "sleepstudy_random_intercept_ml" && case.slow_reason.is_some()));
    }

    #[test]
    fn profile_likelihood_confint_matches_julia_fixture_for_supported_cases() {
        let fixture = profile_parity_fixture();
        let run_slow = std::env::var_os("MIXEDMODELS_RUN_SLOW_PROFILE_PARITY").is_some();
        for case in fixture
            .cases
            .iter()
            .filter(|case| case.unsupported_reason.is_none())
            .filter(|case| run_slow || case.slow_reason.is_none())
        {
            let (data, _) = datasets::load(&case.dataset).unwrap();
            let formula = parse_formula(&case.rust_formula).unwrap();
            let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
            model.fit(case.reml).unwrap();

            let pr = profile(&mut model)
                .unwrap_or_else(|error| panic!("profile failed for {}: {error}", case.id));
            assert_eq!(
                profile_row_order(&pr),
                case.rust_row_order,
                "row order mismatch for {}",
                case.id
            );

            let actual = pr
                .confint(case.level)
                .unwrap_or_else(|error| panic!("confint failed for {}: {error}", case.id))
                .into_iter()
                .map(|row| (row.parameter.clone(), row))
                .collect::<BTreeMap<_, _>>();

            for (parameter, expected) in &case.parameters {
                let actual = actual
                    .get(parameter)
                    .unwrap_or_else(|| panic!("{} missing parameter {parameter}", case.id));
                for (label, got, want) in [
                    ("estimate", actual.estimate, expected.estimate),
                    ("lower", actual.lower, expected.lower),
                    ("upper", actual.upper, expected.upper),
                ] {
                    let tolerance = profile_parity_tolerance(parameter, want);
                    assert!(
                        (got - want).abs() <= tolerance,
                        "{} {parameter} {label}: got {got}, expected {want}, tolerance {tolerance}",
                        case.id
                    );
                }
            }
        }
    }
}
