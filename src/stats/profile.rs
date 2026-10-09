//! Profile-likelihood confidence intervals for linear mixed models.
//!
//! This is a partial port of `MixedModels.jl/src/profile/`. The current
//! scope covers the residual-scale profile (σ), θ profiles, lme4-scale
//! standard-deviation/correlation profiles (`.sig01`, …; see
//! [`profile_sdcor`]), and ML fixed-effect β profiles together
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
    #[serde(with = "nan_as_null")]
    pub zeta: f64,
    /// Residual standard deviation at this row (either the profile target
    /// or the refit value when profiling something else).
    #[serde(with = "nan_as_null")]
    pub sigma: f64,
    /// Fixed-effects coefficients at this row.
    #[serde(with = "nan_as_null_vec")]
    pub beta: Vec<f64>,
    /// θ vector at this row.
    #[serde(with = "nan_as_null_vec")]
    pub theta: Vec<f64>,
    /// lme4-scale variance parameters at this row (`.sig01`, …: one per θ
    /// slot — a standard deviation for a diagonal slot, a correlation for an
    /// off-diagonal one), recorded by the SD/correlation-scale profile.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sdcor: Option<Vec<f64>>,
}

impl ProfileRow {
    fn estimate_row(name: &str, m: &LinearMixedModel) -> Self {
        ProfileRow {
            p: name.to_string(),
            zeta: 0.0,
            sigma: m.sigma(),
            beta: m.beta().iter().cloned().collect(),
            theta: m.theta(),
            sdcor: None,
        }
    }
}

/// Outcome of one per-parameter profile (or of its interval inversion).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProfileParameterStatus {
    /// Strictly monotone profile; the interval is the spline inversion.
    #[default]
    Ok,
    /// ζ stopped being strictly monotone away from the estimate (lme4:
    /// "non-monotonic profile"). Only the monotone segment containing the
    /// estimate is used: a bound inside it is reported, a bound beyond it is
    /// `NaN`. With fewer than five monotone points the parameter has no
    /// spline and both bounds are `NaN`.
    NonMonotone,
    /// The spline interval does not bracket the estimate; both bounds are
    /// reported as `NaN`.
    NotBracketing,
    /// The profile could not be computed at all (the conditional refits or
    /// the spline construction failed); both bounds are `NaN`.
    Failed,
}

impl ProfileParameterStatus {
    /// Stable snake_case code (the serialized form).
    pub fn code(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::NonMonotone => "non_monotone",
            Self::NotBracketing => "not_bracketing",
            Self::Failed => "failed",
        }
    }
}

/// Typed record of a per-parameter profile that was not regular.
///
/// A failing parameter no longer aborts [`profile`]: its record is kept here
/// and the remaining parameters are returned (lme4 warns and reports `NA`
/// for the affected interval).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProfileParameterDiagnostic {
    /// What went wrong.
    pub status: ProfileParameterStatus,
    /// Human-readable reason (the underlying error text or the ζ range used).
    pub reason: String,
    /// Fitted value of the parameter, used for the interval row of a
    /// parameter whose profile has no rows (`NaN` when unknown).
    #[serde(with = "nan_as_null")]
    pub estimate: f64,
}

/// Profile of a fitted linear mixed model.
///
/// Holds the raw table of `(ζ, parameter)` evaluations together with natural
/// cubic-spline interpolants in both directions. `fwd[p]` maps the value of
/// parameter `p` to `ζ`; `rev[p]` inverts the map.
#[derive(Debug, Clone, Default)]
pub struct MixedModelProfile {
    /// All rows collected across every profiled parameter.
    pub tbl: Vec<ProfileRow>,
    /// Forward splines: parameter value → ζ.
    pub fwd: BTreeMap<String, NaturalCubicSpline>,
    /// Reverse splines: ζ → parameter value.
    pub rev: BTreeMap<String, NaturalCubicSpline>,
    /// Parameters whose profile was not regular (non-monotone or failed),
    /// keyed by parameter name. Parameters absent here profiled cleanly.
    pub diagnostics: BTreeMap<String, ProfileParameterDiagnostic>,
}

/// Serialize non-finite `f64` as JSON `null` and read `null` back as `NaN`
/// (serde_json writes `NaN` as `null` but refuses to read it into `f64`).
mod nan_as_null {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(value: &f64, serializer: S) -> Result<S::Ok, S::Error> {
        if value.is_finite() {
            serializer.serialize_f64(*value)
        } else {
            serializer.serialize_none()
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f64, D::Error> {
        Ok(Option::<f64>::deserialize(deserializer)?.unwrap_or(f64::NAN))
    }
}

/// [`nan_as_null`] for `Vec<f64>`.
mod nan_as_null_vec {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(values: &[f64], serializer: S) -> Result<S::Ok, S::Error> {
        values
            .iter()
            .map(|v| v.is_finite().then_some(*v))
            .collect::<Vec<_>>()
            .serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<f64>, D::Error> {
        Ok(Vec::<Option<f64>>::deserialize(deserializer)?
            .into_iter()
            .map(|v| v.unwrap_or(f64::NAN))
            .collect())
    }
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
///
/// Bounds that could not be determined are `NaN` in Rust and `null` on the
/// wire; `status`/`reason` say why.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProfileLikelihoodCiRow {
    /// Parameter name.
    pub parameter: String,
    /// Fitted estimate.
    #[serde(with = "nan_as_null")]
    pub estimate: f64,
    /// Lower confidence limit.
    #[serde(with = "nan_as_null")]
    pub lower: f64,
    /// Upper confidence limit.
    #[serde(with = "nan_as_null")]
    pub upper: f64,
    /// Confidence level used to compute the interval.
    pub level: f64,
    /// Interval method label.
    pub method: String,
    /// Regularity note for the interval.
    pub regularity: String,
    /// Whether the lower limit was clamped at a nonnegative boundary.
    pub boundary_clamped_lower: bool,
    /// Per-parameter profile status (`ok` unless the profile was
    /// non-monotone, did not bracket the estimate, or failed).
    #[serde(default)]
    pub status: ProfileParameterStatus,
    /// Reason for a non-`ok` status.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
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
        let mut rows = Vec::new();
        for (row, status, _) in self.interval_rows(level)? {
            if status == ProfileParameterStatus::NotBracketing && !row.parameter.starts_with(".sig")
            {
                return Err(MixedModelError::Optimization(format!(
                    "confint for {}: profile interval does not bracket estimate {}",
                    row.parameter, row.estimate
                )));
            }
            rows.push(row);
        }
        Ok(rows)
    }

    /// Every profiled parameter's interval with its status, in parameter-name
    /// order. Parameters recorded in [`Self::diagnostics`] without a spline
    /// get a `NaN` interval; a spline interval that does not bracket the
    /// estimate becomes `NaN` with status
    /// [`ProfileParameterStatus::NotBracketing`].
    fn interval_rows(
        &self,
        level: f64,
    ) -> Result<Vec<(ConfintRow, ProfileParameterStatus, Option<String>)>> {
        if !(level > 0.0 && level < 1.0) {
            return Err(MixedModelError::InvalidArgument(format!(
                "confint level must be in (0,1); got {level}"
            )));
        }
        // For a single parameter, profile-likelihood CIs use χ²(1) quantiles
        // — the cutoff is sqrt of that quantile, i.e. the normal quantile.
        let cutoff = normal_inverse_cdf(0.5 + level / 2.0);

        let mut names: Vec<&String> = self.rev.keys().chain(self.diagnostics.keys()).collect();
        names.sort();
        names.dedup();

        let mut rows = Vec::with_capacity(names.len());
        for name in names {
            let diagnostic = self.diagnostics.get(name);
            let mut status = diagnostic.map_or(ProfileParameterStatus::Ok, |d| d.status);
            let mut reason = diagnostic.map(|d| d.reason.clone());
            let Some(spline) = self.rev.get(name) else {
                let estimate = self
                    .profile_estimate(name)
                    .or_else(|| diagnostic.map(|d| d.estimate))
                    .unwrap_or(f64::NAN);
                rows.push((
                    ConfintRow {
                        parameter: name.clone(),
                        estimate,
                        lower: f64::NAN,
                        upper: f64::NAN,
                    },
                    status,
                    reason,
                ));
                continue;
            };
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
            if reason.is_none() && (lower.is_nan() || upper.is_nan()) {
                let sides = match (lower.is_nan(), upper.is_nan()) {
                    (true, true) => "either side",
                    (true, false) => "the lower side",
                    _ => "the upper side",
                };
                reason = Some(format!(
                    "the profile did not reach |ζ| = {cutoff} on {sides} (flat or truncated profile); that bound is not determined"
                ));
            }
            if lower > estimate || upper < estimate {
                // An inconsistent spline interval reports "not determined"
                // for this parameter instead of failing the whole table.
                reason = Some(format!(
                    "profile interval [{lower}, {upper}] does not bracket estimate {estimate}"
                ));
                status = ProfileParameterStatus::NotBracketing;
                lower = f64::NAN;
                upper = f64::NAN;
            }
            rows.push((
                ConfintRow {
                    parameter: name.clone(),
                    estimate,
                    lower,
                    upper,
                },
                status,
                reason,
            ));
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
        let intervals: Vec<ProfileLikelihoodCiRow> = self
            .interval_rows(level)?
            .into_iter()
            .map(|(row, status, reason)| {
                let boundary_clamped_lower =
                    row.lower == 0.0 && self.profile_touches_nonnegative_boundary(&row.parameter);
                let regularity = match status {
                    ProfileParameterStatus::NonMonotone => "non_monotone_profile",
                    ProfileParameterStatus::NotBracketing => "profile_not_bracketing",
                    ProfileParameterStatus::Failed => "profile_failed",
                    ProfileParameterStatus::Ok if boundary_clamped_lower => {
                        "nonnegative_parameter_boundary_clamped"
                    }
                    ProfileParameterStatus::Ok => "regular_profile_likelihood",
                };
                ProfileLikelihoodCiRow {
                    parameter: row.parameter,
                    estimate: row.estimate,
                    lower: row.lower,
                    upper: row.upper,
                    level,
                    method: "profile_likelihood".to_string(),
                    regularity: regularity.to_string(),
                    boundary_clamped_lower,
                    status,
                    reason,
                }
            })
            .collect();

        let mut notes = vec![
            "profile-likelihood intervals are computed by spline inversion of signed-root deviance values".to_string(),
            "profile rows are serialized for diagnostics; spline coefficients are intentionally not part of the wire contract".to_string(),
        ];
        if self
            .rev
            .keys()
            .chain(self.diagnostics.keys())
            .any(|name| name.starts_with(".sig"))
        {
            notes.push(
                "`.sigNN` rows are lme4-scale intervals (confint(method = \"profile\")): θ slot NN in engine θ order, a random-effect standard deviation for a diagonal slot and a correlation for an off-diagonal slot; like lme4 they profile the ML deviance (REML fits are refitted by ML). `θNN` rows remain relative-Cholesky-scale intervals."
                    .to_string(),
            );
        }
        if intervals
            .iter()
            .any(|row| row.status != ProfileParameterStatus::Ok)
        {
            notes.push(
                "rows with a non-`ok` status had an irregular profile (non-monotone, not bracketing the estimate, or failed); their undetermined bounds are null and `reason` explains why, while the other parameters' intervals are unaffected (lme4 warns and reports NA)"
                    .to_string(),
            );
        }
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
        // θ is dimensionless (relative Cholesky scale) and a boundary θ
        // estimate is anything within `THETA_BOUNDARY_TOL` of zero.
        let zero_tol = if parameter.starts_with('θ') {
            THETA_BOUNDARY_TOL
        } else {
            1e-10
        };
        values.iter().all(|value| *value >= -1e-12)
            && values.iter().any(|value| value.abs() < zero_tol)
    }

    fn profile_estimate(&self, parameter: &str) -> Option<f64> {
        self.tbl
            .iter()
            .filter(|row| row.p == parameter)
            .min_by(|left, right| left.zeta.abs().partial_cmp(&right.zeta.abs()).unwrap())
            .and_then(|row| profile_row_parameter_value(row, parameter))
    }

    fn parameter_values(&self, parameter: &str) -> Option<Vec<f64>> {
        if parameter.starts_with(".sig") {
            return self
                .rows_for(parameter)
                .into_iter()
                .map(|row| profile_row_parameter_value(row, parameter))
                .collect();
        }
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

fn sdcor_parameter_index(parameter: &str) -> Option<usize> {
    parameter
        .strip_prefix(".sig")?
        .parse::<usize>()
        .ok()?
        .checked_sub(1)
}

fn profile_row_parameter_value(row: &ProfileRow, parameter: &str) -> Option<f64> {
    if parameter == "σ" || parameter == ".sigma" {
        return Some(row.sigma);
    }
    if let Some(index) = sdcor_parameter_index(parameter) {
        return row.sdcor.as_ref()?.get(index).copied();
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
            sdcor: None,
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
        diagnostics: BTreeMap::new(),
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
            sdcor: None,
        });
        Ok(zeta)
    };

    let max_points = 60;
    if !lower.is_finite() {
        // An unbounded coordinate (an off-diagonal relative-Cholesky entry)
        // can be negative and must be able to cross zero, so the
        // multiplicative walk used for nonnegative diagonal entries does not
        // apply: dividing a negative θ by the step factor *increases* it
        // (mislabelling the side and producing a decreasing ζ table), and no
        // multiplicative step ever crosses zero. Walk additively instead,
        // with a step sized from a local probe so Δζ ≈ 0.5 per point.
        let probe_h = (theta_hat.abs() * 0.05).max(0.01);
        let probe_start = theta_hat_vector.clone();
        let (_, probe_obj) = optimize_theta_profile_point(
            m,
            index,
            theta_hat + probe_h,
            &probe_start,
            &lower_bounds,
        )?;
        let probe_zeta = (probe_obj - obj_hat).max(0.0).sqrt();
        let base_step = if probe_zeta.is_finite() && probe_zeta > 0.0 {
            (probe_h * 0.5 / probe_zeta).clamp(probe_h * 0.1, probe_h * 50.0)
        } else {
            probe_h
        };
        for negative_side in [true, false] {
            let direction = if negative_side { -1.0 } else { 1.0 };
            let mut start = theta_hat_vector.clone();
            let mut step = base_step;
            let mut theta_v = theta_hat + direction * step;
            let mut last_zeta = 0.0_f64;
            for _ in 0..max_points {
                let zeta = evaluate(m, theta_v, &mut start, negative_side)?;
                if zeta.abs() >= threshold {
                    break;
                }
                if (zeta - last_zeta).abs() < 0.25 {
                    step *= 1.5;
                }
                last_zeta = zeta;
                theta_v += direction * step;
            }
        }
    } else {
        // An estimate numerically on the nonnegative boundary (a θ of 1e-8
        // is a converged boundary fit, not an interior optimum) has no
        // lower side to walk, and multiplicative steps from it never get
        // anywhere; treat it like an exact boundary estimate.
        let at_boundary = theta_hat - lower <= THETA_BOUNDARY_TOL;
        let facsz = {
            let probe = if theta_hat.abs() > min_step && !at_boundary {
                theta_hat * (1.0_f64 / 64.0).exp()
            } else {
                theta_hat + 0.05
            };
            let probe_start = theta_hat_vector.clone();
            let (_, obj) =
                optimize_theta_profile_point(m, index, probe, &probe_start, &lower_bounds)?;
            theta_profile_step_factor_from_probe(theta_hat, probe, obj_hat, obj)
        };

        let mut negative_start = theta_hat_vector.clone();
        if !at_boundary && theta_hat > lower + min_step {
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
        let mut theta_v = if at_boundary {
            theta_hat + 0.05
        } else {
            next_theta_profile_value(theta_hat, facsz, lower, false, min_step)
        };
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
    }

    rows.sort_by(|a, b| a.theta[index].partial_cmp(&b.theta[index]).unwrap());
    rows.dedup_by(|a, b| (a.theta[index] - b.theta[index]).abs() < 1e-14);

    m.set_theta(&saved_theta)?;
    m.update_l()?;
    m.optsum.fmin = saved_fmin;

    let mut fwd_map = BTreeMap::new();
    let mut rev_map = BTreeMap::new();
    let mut diagnostics = BTreeMap::new();
    add_profile_splines(
        &parameter,
        &mut rows,
        Some(theta_hat),
        &mut fwd_map,
        &mut rev_map,
        &mut diagnostics,
        |row| row.theta[index],
    )?;

    Ok(MixedModelProfile {
        tbl: rows,
        fwd: fwd_map,
        rev: rev_map,
        diagnostics,
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

/// Distance from the lower bound below which a fitted θ is treated as an
/// estimate on the boundary by [`profile_theta`]. θ is on the relative
/// (σ-scaled, dimensionless) Cholesky scale, so this is an absolute
/// tolerance.
const THETA_BOUNDARY_TOL: f64 = 1e-6;

/// Largest ζ reversal treated as optimizer noise on a flat profile plateau
/// rather than as a genuinely non-monotone profile.
const PROFILE_PLATEAU_NOISE: f64 = 1e-6;

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
            sdcor: None,
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
    let mut diagnostics = BTreeMap::new();
    add_profile_splines(
        &parameter,
        &mut rows,
        Some(beta_hat),
        &mut fwd_map,
        &mut rev_map,
        &mut diagnostics,
        |row| row.beta[index],
    )?;

    Ok(MixedModelProfile {
        tbl: rows,
        fwd: fwd_map,
        rev: rev_map,
        diagnostics,
    })
}

/// Profile every active fixed-effect coefficient of an ML-fitted LMM.
pub fn profile_betas(m: &mut LinearMixedModel, threshold: f64) -> Result<MixedModelProfile> {
    let mut out = MixedModelProfile::default();
    for index in 0..m.feterm.rank {
        out.merge(profile_beta(m, index, threshold)?);
    }
    Ok(out)
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
        for (col, beta_col) in beta.iter_mut().enumerate() {
            if col == fixed_index {
                continue;
            }
            *beta_col = beta_free[(free_col, 0)];
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

// ===========================================================================
// SD / correlation (lme4 `.sigNN`) profiles
// ===========================================================================

/// lme4-scale variance parameters for a θ vector and residual σ: one value
/// per θ slot (in θ order), a standard deviation `σ·‖Λ[r,:]‖` for a diagonal
/// slot and the correlation for an off-diagonal slot. This is the layout of
/// lme4's `.sig01`, `.sig02`, … within each random-effect term.
pub fn sdcor_from_theta(m: &LinearMixedModel, theta: &[f64], sigma: f64) -> Vec<f64> {
    let lambdas = block_lambdas(m, theta);
    m.parmap
        .iter()
        .map(|&(block, row, col)| {
            let lambda = &lambdas[block];
            let cov = |i: usize, j: usize| -> f64 {
                (0..lambda.ncols())
                    .map(|k| lambda[(i, k)] * lambda[(j, k)])
                    .sum()
            };
            if row == col {
                sigma * cov(row, row).max(0.0).sqrt()
            } else {
                let denom = (cov(row, row) * cov(col, col)).sqrt();
                if denom > 0.0 {
                    (cov(row, col) / denom).clamp(-1.0, 1.0)
                } else {
                    0.0
                }
            }
        })
        .collect()
}

fn block_lambdas(m: &LinearMixedModel, theta: &[f64]) -> Vec<DMatrix<f64>> {
    let mut lambdas: Vec<DMatrix<f64>> = m
        .reterms
        .iter()
        .map(|term| DMatrix::zeros(term.vsize, term.vsize))
        .collect();
    for (&(block, row, col), &value) in m.parmap.iter().zip(theta) {
        lambdas[block][(row, col)] = value;
    }
    lambdas
}

/// Inverse of [`sdcor_from_theta`]: the θ vector (lower Cholesky factor of
/// each relative covariance block) for lme4-scale parameters and σ, or
/// `None` when they do not define a positive semi-definite covariance.
pub fn theta_from_sdcor(m: &LinearMixedModel, sdcor: &[f64], sigma: f64) -> Option<Vec<f64>> {
    if sdcor.len() != m.parmap.len() || !(sigma > 0.0) {
        return None;
    }
    let mut sds: Vec<Vec<f64>> = m.reterms.iter().map(|t| vec![0.0; t.vsize]).collect();
    let mut cors: Vec<DMatrix<f64>> = m
        .reterms
        .iter()
        .map(|t| DMatrix::identity(t.vsize, t.vsize))
        .collect();
    for (&(block, row, col), &value) in m.parmap.iter().zip(sdcor) {
        if !value.is_finite() {
            return None;
        }
        if row == col {
            if value < 0.0 {
                return None;
            }
            sds[block][row] = value / sigma;
        } else {
            if value.abs() > 1.0 {
                return None;
            }
            cors[block][(row, col)] = value;
            cors[block][(col, row)] = value;
        }
    }
    let mut factors = Vec::with_capacity(m.reterms.len());
    for (block, term) in m.reterms.iter().enumerate() {
        let k = term.vsize;
        let cov = DMatrix::from_fn(k, k, |i, j| {
            cors[block][(i, j)] * sds[block][i] * sds[block][j]
        });
        factors.push(semidefinite_cholesky(&cov)?);
    }
    Some(
        m.parmap
            .iter()
            .map(|&(block, row, col)| factors[block][(row, col)])
            .collect(),
    )
}

/// Lower Cholesky factor of a positive semi-definite matrix (zero pivots
/// allowed), or `None` when the matrix is indefinite.
fn semidefinite_cholesky(a: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    let n = a.nrows();
    let scale = (0..n)
        .map(|i| a[(i, i)].abs())
        .fold(0.0, f64::max)
        .max(1e-300);
    let tol = 1e-10 * scale;
    let mut l = DMatrix::zeros(n, n);
    for j in 0..n {
        let d = a[(j, j)] - (0..j).map(|k| l[(j, k)] * l[(j, k)]).sum::<f64>();
        if d < -tol {
            return None;
        }
        let ljj = d.max(0.0).sqrt();
        l[(j, j)] = ljj;
        for i in (j + 1)..n {
            let r = a[(i, j)] - (0..j).map(|k| l[(i, k)] * l[(j, k)]).sum::<f64>();
            if ljj > tol.sqrt() {
                l[(i, j)] = r / ljj;
            } else if r.abs() > tol.sqrt() * scale.sqrt() {
                return None;
            }
        }
    }
    Some(l)
}

/// lme4 name for SD/correlation parameter `index` (0-based θ slot) or, when
/// `index == n_theta`, the residual scale.
fn sdcor_parameter_name(index: usize, n_theta: usize) -> String {
    if index == n_theta {
        ".sigma".to_string()
    } else {
        format!(".sig{:02}", index + 1)
    }
}

struct SdcorObjective<'a> {
    model: &'a mut LinearMixedModel,
    reml: bool,
    n_theta: usize,
}

impl SdcorObjective<'_> {
    /// Deviance at lme4-scale parameters `p = (sdcor…, σ)`; `+∞` when they
    /// are infeasible.
    fn eval(&mut self, p: &[f64]) -> f64 {
        let sigma = p[self.n_theta];
        let Some(theta) = theta_from_sdcor(self.model, &p[..self.n_theta], sigma) else {
            return f64::INFINITY;
        };
        let mut varpar = theta;
        varpar.push(sigma);
        self.model
            .deviance_varpar(&varpar, self.reml)
            .ok()
            .filter(|value| value.is_finite())
            .unwrap_or(f64::INFINITY)
    }

    fn clamp(&self, index: usize, value: f64) -> f64 {
        if index == self.n_theta {
            value.max(1e-12)
        } else if self.model.parmap[index].1 == self.model.parmap[index].2 {
            value.max(0.0)
        } else {
            value.clamp(-1.0, 1.0)
        }
    }

    /// Minimize over every coordinate except `fixed` (bound-constrained
    /// trust-region search on scaled coordinates, polished by a coordinate
    /// pattern search), starting from `start`.
    fn minimize_others(&mut self, fixed: usize, start: &[f64]) -> (Vec<f64>, f64) {
        let free: Vec<usize> = (0..start.len()).filter(|&i| i != fixed).collect();
        if free.is_empty() {
            let obj = self.eval(start);
            return (start.to_vec(), obj);
        }
        let scale: Vec<f64> = free
            .iter()
            .map(|&i| {
                if i < self.n_theta && self.model.parmap[i].1 != self.model.parmap[i].2 {
                    1.0
                } else {
                    start[i].abs().max(1e-3)
                }
            })
            .collect();
        let lower: Vec<f64> = free
            .iter()
            .zip(&scale)
            .map(|(&i, s)| {
                if i == self.n_theta {
                    1e-8
                } else if self.model.parmap[i].1 == self.model.parmap[i].2 {
                    0.0
                } else {
                    -1.0 / s
                }
            })
            .collect();
        let upper: Vec<f64> = free
            .iter()
            .zip(&scale)
            .map(|(&i, s)| {
                if i < self.n_theta && self.model.parmap[i].1 != self.model.parmap[i].2 {
                    1.0 / s
                } else {
                    1e6
                }
            })
            .collect();
        let initial: Vec<f64> = free
            .iter()
            .zip(&scale)
            .map(|(&i, s)| start[i] / s)
            .collect();
        let options = crate::optimizer::trust_bq::TrustBqOptions {
            initial_radius: 0.05,
            final_radius: 1e-8,
            max_evaluations: 2000,
            ..Default::default()
        };
        let base = start.to_vec();
        let trust = crate::optimizer::trust_bq::minimize_with_progress(
            &initial,
            &lower,
            &upper,
            options,
            |y: &[f64]| {
                let mut p = base.clone();
                for ((&i, s), v) in free.iter().zip(&scale).zip(y) {
                    p[i] = v * s;
                }
                let value = self.eval(&p);
                Ok(if value.is_finite() { value } else { 1e300 })
            },
            |_| Ok(false),
        );
        let mut polished_start = start.to_vec();
        if let Ok(result) = trust {
            for ((&i, s), v) in free.iter().zip(&scale).zip(&result.x) {
                polished_start[i] = self.clamp(i, v * s);
            }
            if self.eval(&polished_start) > self.eval(start) {
                polished_start = start.to_vec();
            }
        }
        self.pattern_search_others(fixed, &polished_start)
    }

    /// Coordinate pattern search over every coordinate except `fixed`.
    fn pattern_search_others(&mut self, fixed: usize, start: &[f64]) -> (Vec<f64>, f64) {
        let mut best = start.to_vec();
        let mut best_obj = self.eval(&best);
        let mut steps: Vec<f64> = best
            .iter()
            .enumerate()
            .map(|(i, v)| {
                if i == fixed {
                    0.0
                } else if i < self.n_theta && self.model.parmap[i].1 != self.model.parmap[i].2 {
                    0.01
                } else {
                    (v.abs() * 0.01).max(1e-3)
                }
            })
            .collect();
        for _ in 0..200 {
            let mut improved = false;
            for i in 0..best.len() {
                if i == fixed {
                    continue;
                }
                for direction in [1.0, -1.0] {
                    let mut candidate = best.clone();
                    candidate[i] = self.clamp(i, candidate[i] + direction * steps[i]);
                    if (candidate[i] - best[i]).abs() < 1e-14 {
                        continue;
                    }
                    let obj = self.eval(&candidate);
                    if obj + 1e-10 < best_obj {
                        best_obj = obj;
                        best = candidate;
                        improved = true;
                        steps[i] *= 1.5;
                        break;
                    }
                }
            }
            if !improved {
                let mut max_rel = 0.0_f64;
                for (i, step) in steps.iter_mut().enumerate() {
                    if i == fixed {
                        continue;
                    }
                    *step *= 0.5;
                    max_rel = max_rel.max(*step / best[i].abs().max(1e-3));
                }
                if max_rel < 1e-7 {
                    break;
                }
            }
        }
        (best, best_obj)
    }
}

/// Profile one lme4-scale variance parameter (`.sigNN` — an SD or a
/// correlation of a random-effect term — or `.sigma` for `index == n_theta`),
/// refitting every other variance parameter at each point (fixed effects
/// are profiled out).
///
/// Like lme4, a REML fit is first refitted by ML and the ML deviance is
/// profiled (the `"σ"`/`"θ…"` profiles of a REML fit profile the REML
/// criterion instead).
///
/// This is lme4's `profile()` parameterization (`devfun2`), so its
/// intervals are the ones `confint(fit, method = "profile")` reports for
/// `.sig01`, …, `.sigma`; θ-scale profiles remain available from
/// [`profile_theta`].
pub fn profile_sdcor(
    m: &mut LinearMixedModel,
    index: usize,
    threshold: f64,
) -> Result<MixedModelProfile> {
    let n_theta = m.n_theta();
    if index > n_theta {
        return Err(MixedModelError::InvalidArgument(format!(
            "profile_sdcor index {index} is out of bounds for {n_theta} θ parameter(s) plus σ"
        )));
    }
    if m.optsum.feval <= 0 {
        return Err(MixedModelError::InvalidArgument(
            "profile_sdcor: model must be fitted first".into(),
        ));
    }
    if m.optsum.sigma.is_some() {
        return Err(MixedModelError::InvalidArgument(
            "profile_sdcor requires σ to be estimated, not fixed".into(),
        ));
    }
    if m.optsum.reml {
        // lme4's profile() (devfun2) always profiles the ML deviance of the
        // ML refit (`refitML`), also for REML fits; do the same so the
        // intervals are the ones `confint(fit, method = "profile")` reports.
        let mut ml = m.clone();
        // Reset the fit state (as `refit` does) so `fit` accepts the clone.
        ml.optsum.feval = 0;
        ml.fit(false)?;
        return profile_sdcor(&mut ml, index, threshold);
    }

    let parameter = sdcor_parameter_name(index, n_theta);
    let saved_theta = m.theta();
    let saved_fmin = m.optsum.fmin;
    let sigma_hat = m.sigma();
    let reml = m.optsum.reml;
    let mut estimate = sdcor_from_theta(m, &saved_theta, sigma_hat);
    estimate.push(sigma_hat);
    let is_correlation = index < n_theta && m.parmap[index].1 != m.parmap[index].2;

    let result = (|| -> Result<MixedModelProfile> {
        let mut objective = SdcorObjective {
            model: &mut *m,
            reml,
            n_theta,
        };
        let obj_hat = objective.eval(&estimate);
        if !obj_hat.is_finite() {
            return Err(MixedModelError::Optimization(format!(
                "profile_{parameter}: deviance is not finite at the estimate"
            )));
        }
        let value_hat = estimate[index];
        let row_for = |objective: &SdcorObjective, p: &[f64], zeta: f64| -> ProfileRow {
            let theta = theta_from_sdcor(objective.model, &p[..n_theta], p[n_theta])
                .unwrap_or_else(|| vec![f64::NAN; n_theta]);
            ProfileRow {
                p: parameter.clone(),
                zeta,
                sigma: p[n_theta],
                beta: Vec::new(),
                theta,
                sdcor: Some(p[..n_theta].to_vec()),
            }
        };
        let mut rows = vec![row_for(&objective, &estimate, 0.0)];

        // Step size from a local quadratic probe: aim for Δζ ≈ 0.3 per step.
        let probe_h = if is_correlation {
            0.02
        } else {
            (value_hat.abs() * 0.02).max(1e-4)
        };
        let mut probe = estimate.clone();
        probe[index] = objective.clamp(index, value_hat + probe_h);
        let (_, probe_obj) = objective.minimize_others(index, &probe);
        let curvature = ((probe_obj - obj_hat).max(1e-12)) / (probe[index] - value_hat).powi(2);
        let mut base_step = (0.3 / curvature.sqrt()).max(probe_h);
        if is_correlation {
            base_step = base_step.min(0.1);
        }

        for direction in [-1.0, 1.0] {
            let mut current = estimate.clone();
            let mut step = base_step;
            let mut last_zeta = 0.0_f64;
            for _ in 0..60 {
                let target = objective.clamp(index, current[index] + direction * step);
                if (target - current[index]).abs() < 1e-12 {
                    break;
                }
                let mut start = current.clone();
                start[index] = target;
                let (point, obj) = objective.minimize_others(index, &start);
                if !obj.is_finite() {
                    break;
                }
                let zeta = direction * (obj - obj_hat).max(0.0).sqrt();
                rows.push(row_for(&objective, &point, zeta));
                current = point;
                let at_bound = objective.clamp(index, target + direction * 1e-9) == target;
                if zeta.abs() >= threshold || at_bound {
                    break;
                }
                if (zeta - last_zeta).abs() < 0.15 {
                    step *= 1.6;
                }
                last_zeta = zeta;
            }
        }

        rows.sort_by(|a, b| {
            a.zeta
                .partial_cmp(&b.zeta)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let value_of = |row: &ProfileRow| -> f64 {
            if index == n_theta {
                row.sigma
            } else {
                row.sdcor.as_ref().map_or(f64::NAN, |v| v[index])
            }
        };
        // Points pinned at a boundary (SD at 0, |ρ| at 1) can repeat ζ; keep
        // the first so the ζ table stays strictly monotone.
        rows.dedup_by(|a, b| a.zeta <= b.zeta + 1e-12 || (value_of(a) - value_of(b)).abs() < 1e-14);

        let mut fwd = BTreeMap::new();
        let mut rev = BTreeMap::new();
        let mut diagnostics = BTreeMap::new();
        add_profile_splines(
            &parameter,
            &mut rows,
            Some(value_hat),
            &mut fwd,
            &mut rev,
            &mut diagnostics,
            value_of,
        )?;
        Ok(MixedModelProfile {
            tbl: rows,
            fwd,
            rev,
            diagnostics,
        })
    })();

    m.set_theta(&saved_theta)?;
    m.update_l()?;
    m.optsum.fmin = saved_fmin;
    result
}

/// SD/correlation-scale profiles (`.sig01`, …) for every θ slot.
///
/// A parameter whose profile cannot be computed (for example a correlation
/// pinned at ±1) does not fail the whole table: it gets no spline and its
/// typed reason is recorded in [`MixedModelProfile::diagnostics`] (its
/// interval is then `NaN`). `.sigma` is not repeated because the σ profile
/// is already the `"σ"` entry.
pub fn profile_sdcors(m: &mut LinearMixedModel, threshold: f64) -> MixedModelProfile {
    let pristine = m.clone();
    let mut out = MixedModelProfile::default();
    for index in 0..m.n_theta() {
        match profile_sdcor(m, index, threshold) {
            Ok(profile) => out.merge(profile),
            Err(error) => {
                *m = pristine.clone();
                out.merge(failed_profile(m, ProfileTask::SdCor(index), &error));
            }
        }
    }
    out
}

impl MixedModelProfile {
    /// Append another (disjoint) per-parameter profile.
    fn merge(&mut self, other: MixedModelProfile) {
        self.tbl.extend(other.tbl);
        self.fwd.extend(other.fwd);
        self.rev.extend(other.rev);
        self.diagnostics.extend(other.diagnostics);
    }
}

/// One independent per-parameter profile of [`profile`].
#[derive(Debug, Clone, Copy)]
enum ProfileTask {
    Sigma,
    Beta(usize),
    Theta(usize),
    /// lme4-scale `.sigNN` profile.
    SdCor(usize),
}

impl ProfileTask {
    fn parameter(self, n_theta: usize) -> String {
        match self {
            ProfileTask::Sigma => "σ".to_string(),
            ProfileTask::Beta(index) => format!("β{}", index + 1),
            ProfileTask::Theta(index) => format!("θ{}", index + 1),
            ProfileTask::SdCor(index) => sdcor_parameter_name(index, n_theta),
        }
    }

    /// Fitted value of the task's parameter (`NaN` when the parameter is
    /// not defined on the fit's own scale, e.g. `.sigNN` of a REML fit,
    /// which lme4 profiles on the ML refit).
    fn estimate(self, m: &LinearMixedModel) -> f64 {
        match self {
            ProfileTask::Sigma => m.sigma(),
            ProfileTask::Beta(index) => m.beta().get(index).copied().unwrap_or(f64::NAN),
            ProfileTask::Theta(index) => m.theta().get(index).copied().unwrap_or(f64::NAN),
            ProfileTask::SdCor(_) if m.optsum.reml => f64::NAN,
            ProfileTask::SdCor(index) => sdcor_from_theta(m, &m.theta(), m.sigma())
                .get(index)
                .copied()
                .unwrap_or(f64::NAN),
        }
    }
}

/// The per-parameter profiles of [`profile`], in table order.
fn profile_tasks(m: &LinearMixedModel) -> Result<Vec<ProfileTask>> {
    if m.optsum.feval <= 0 {
        return Err(MixedModelError::InvalidArgument(
            "profile: model must be fitted first".into(),
        ));
    }
    if m.optsum.sigma.is_some() {
        return Err(MixedModelError::InvalidArgument(format!(
            "Can't profile σ because it is already fixed at {:?}",
            m.optsum.sigma
        )));
    }
    let mut tasks = vec![ProfileTask::Sigma];
    if !m.optsum.reml {
        tasks.extend((0..m.feterm.rank).map(ProfileTask::Beta));
    }
    tasks.extend((0..m.n_theta()).map(ProfileTask::Theta));
    tasks.extend((0..m.n_theta()).map(ProfileTask::SdCor));
    Ok(tasks)
}

/// Run one per-parameter profile on `work` (a model in the fitted state).
///
/// A host interrupt is propagated. Any other failure is confined to this
/// parameter: the returned profile has no rows or splines, only a
/// [`ProfileParameterDiagnostic`] (computed from `fitted`, the untouched
/// fitted model), and the flag is `true` so the caller can restore `work`
/// (a failed profile may return before restoring the fitted state).
fn run_profile_task(
    work: &mut LinearMixedModel,
    fitted: &LinearMixedModel,
    task: ProfileTask,
) -> Result<(MixedModelProfile, bool)> {
    let result = match task {
        ProfileTask::Sigma => profile_sigma(work, 4.0),
        ProfileTask::Beta(index) => profile_beta(work, index, 4.0),
        ProfileTask::Theta(index) => profile_theta(work, index, 4.0),
        ProfileTask::SdCor(index) => profile_sdcor(work, index, 4.0),
    };
    match result {
        Ok(profile) => Ok((profile, false)),
        Err(error @ MixedModelError::Interrupted(_)) => Err(error),
        Err(error) => Ok((failed_profile(fitted, task, &error), true)),
    }
}

/// A rows-free profile recording why `task`'s profile failed.
fn failed_profile(
    fitted: &LinearMixedModel,
    task: ProfileTask,
    error: &MixedModelError,
) -> MixedModelProfile {
    let reason = error.to_string();
    let status = if reason.contains(NON_MONOTONE_FRAGMENT) {
        ProfileParameterStatus::NonMonotone
    } else {
        ProfileParameterStatus::Failed
    };
    let mut out = MixedModelProfile::default();
    out.diagnostics.insert(
        task.parameter(fitted.n_theta()),
        ProfileParameterDiagnostic {
            status,
            reason,
            estimate: task.estimate(fitted),
        },
    );
    out
}

/// Public entry point matching the shape of `MixedModels.jl::profile`.
///
/// Profiles σ and θ for fitted LMMs, plus the lme4-scale `.sigNN`
/// parameters. For ML fits, active fixed-effect β profiles are included as
/// well. REML β profiles are deliberately omitted until a certified REML
/// fixed-effect profile contract exists.
///
/// A per-parameter profile that fails (for example a non-monotone ζ table,
/// a failed conditional refit, or β profiles of a weighted fit) does not
/// abort the profile: like lme4, which warns and reports `NA` for that
/// interval, the parameter is recorded in
/// [`MixedModelProfile::diagnostics`] with a typed
/// [`ProfileParameterStatus`] and reason, and every other parameter is
/// returned. Only a host interrupt or an unfitted / fixed-σ model is an
/// error.
pub fn profile(m: &mut LinearMixedModel) -> Result<MixedModelProfile> {
    let tasks = profile_tasks(m)?;
    let fitted = m.clone();
    let mut out = MixedModelProfile::default();
    for task in tasks {
        let (piece, failed) = run_profile_task(m, &fitted, task)?;
        if failed {
            *m = fitted.clone();
        }
        out.merge(piece);
    }
    Ok(out)
}

/// Execution controls for [`profile_with_options`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileOptions {
    /// Worker threads (default 1 = serial). With more than one thread the
    /// independent per-parameter profiles (σ, each ML β, each θ) run
    /// concurrently, each on its own copy of the fitted model; the result
    /// is bit-identical to the serial profile for every thread count.
    /// Workers never invoke the host progress/interrupt callback: the
    /// calling thread polls it while it waits (phase
    /// [`FitProgressPhase::Profile`](crate::model::linear::FitProgressPhase::Profile))
    /// and, on an interrupt, stops the workers and returns the error.
    #[serde(default = "default_profile_threads")]
    pub threads: usize,
}

impl Default for ProfileOptions {
    fn default() -> Self {
        Self { threads: 1 }
    }
}

fn default_profile_threads() -> usize {
    1
}

/// [`profile`] with execution options (worker threads).
///
/// `threads == 1` is exactly [`profile`]. With more threads, the σ profile,
/// each ML β profile, each θ profile and each lme4-scale `.sigNN` profile
/// run on scoped worker threads, each
/// on a fresh copy of the fitted model (every per-parameter profile starts
/// from, and restores, the fitted state, so the copies see exactly what the
/// serial sweep sees), and the pieces are assembled in the serial order.
/// `m` itself is left untouched.
pub fn profile_with_options(
    m: &mut LinearMixedModel,
    options: &ProfileOptions,
) -> Result<MixedModelProfile> {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    use crate::model::linear::{FitProgressCallback, FitProgressPhase};

    crate::parallel::validate_threads(options.threads)?;
    if options.threads == 1 {
        return profile(m);
    }

    let tasks = profile_tasks(m)?;

    // Workers get a pure-Rust callback that only watches the cancel flag,
    // so refits on worker threads can stop early without touching the host.
    let cancel = Arc::new(AtomicBool::new(false));
    let mut template = m.clone();
    template.progress_callback = Some({
        let cancel = Arc::clone(&cancel);
        FitProgressCallback::new(move |_| {
            if cancel.load(Ordering::Acquire) {
                Err(MixedModelError::Interrupted(
                    "profile cancelled by the host".to_string(),
                ))
            } else {
                Ok(())
            }
        })
    });

    let host_callback = m.progress_callback.clone();
    let mut polls = 0usize;
    let mut last_reported = 0usize;
    let mut poll = || -> Result<()> {
        polls += 1;
        match &host_callback {
            Some(callback) => {
                callback.report_if_due(FitProgressPhase::Profile, polls, None, &mut last_reported)
            }
            None => Ok(()),
        }
    };
    let pieces = crate::parallel::map_with_workers_polled(
        options.threads,
        tasks,
        || (),
        |_, task| {
            let mut work = template.clone();
            run_profile_task(&mut work, &template, task).map(|(piece, _)| piece)
        },
        &cancel,
        &mut poll,
    )?;

    let mut out = MixedModelProfile::default();
    for piece in pieces {
        out.merge(piece?);
    }
    Ok(out)
}

/// [`profile_confint_payload`] with execution options (worker threads).
pub fn profile_confint_payload_with_options(
    m: &mut LinearMixedModel,
    level: f64,
    options: &ProfileOptions,
) -> Result<ProfileLikelihoodCiPayload> {
    let reml = m.optsum.reml;
    profile_with_options(m, options)?.confint_payload(level, reml)
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

/// Message fragment of the non-monotone profile error (used to classify a
/// failed per-parameter profile).
const NON_MONOTONE_MESSAGE: &str = "ζ table is not strictly monotone";

/// Fragment shared by every non-monotone profile error (σ uses "ζ(σ) table").
const NON_MONOTONE_FRAGMENT: &str = "not strictly monotone";

/// Build the forward/reverse splines for one profiled parameter.
///
/// `rows` are reordered by parameter value. When `estimate_value` locates the
/// estimate row, the table is checked outward from it: ζ must strictly
/// increase with the parameter. A tail whose ζ reverses by no more than
/// [`PROFILE_PLATEAU_NOISE`] is a flat plateau where the conditional refits
/// are pure rounding noise and is dropped silently. A larger reversal is a
/// non-monotone profile (lme4: "non-monotonic profile"): the rows beyond it
/// are dropped, only the monotone segment containing the estimate is used,
/// and a [`ProfileParameterStatus::NonMonotone`] diagnostic is recorded.
/// Dropped rows are removed from `rows` so the table matches the splines.
fn add_profile_splines(
    parameter: &str,
    rows: &mut Vec<ProfileRow>,
    estimate_value: Option<f64>,
    fwd_map: &mut BTreeMap<String, NaturalCubicSpline>,
    rev_map: &mut BTreeMap<String, NaturalCubicSpline>,
    diagnostics: &mut BTreeMap<String, ProfileParameterDiagnostic>,
    value_of: impl Fn(&ProfileRow) -> f64,
) -> Result<()> {
    let mut non_monotone = None;
    if let Some(estimate) = estimate_value {
        rows.sort_by(|a, b| {
            value_of(a)
                .partial_cmp(&value_of(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if let Some(est) = rows.iter().position(|row| value_of(row) == estimate) {
            let increasing =
                |lo: &ProfileRow, hi: &ProfileRow| hi.zeta > lo.zeta && value_of(hi) > value_of(lo);
            let mut end = rows.len();
            for i in est + 1..rows.len() {
                if !increasing(&rows[i - 1], &rows[i]) {
                    end = i;
                    let reversal = rows[i - 1].zeta - rows[i].zeta;
                    if !(reversal <= PROFILE_PLATEAU_NOISE) {
                        non_monotone.get_or_insert((rows[i - 1].zeta, rows[i].zeta));
                    }
                    break;
                }
            }
            let mut start = 0;
            for i in (0..est).rev() {
                if !increasing(&rows[i], &rows[i + 1]) {
                    start = i + 1;
                    let reversal = rows[i].zeta - rows[i + 1].zeta;
                    if !(reversal <= PROFILE_PLATEAU_NOISE) {
                        non_monotone = Some((rows[i + 1].zeta, rows[i].zeta));
                    }
                    break;
                }
            }
            rows.truncate(end);
            rows.drain(..start);
        }
    }
    let values: Vec<f64> = rows.iter().map(&value_of).collect();
    let zetas: Vec<f64> = rows.iter().map(|r| r.zeta).collect();
    if let Some((before, after)) = non_monotone {
        let reason = format!(
            "profile_{parameter}: {NON_MONOTONE_MESSAGE} (ζ moved from {before} to {after} away from the estimate); only the monotone ζ range [{}, {}] around the estimate is used",
            zetas.first().copied().unwrap_or(f64::NAN),
            zetas.last().copied().unwrap_or(f64::NAN),
        );
        if values.len() < 5 {
            return Err(MixedModelError::Optimization(reason));
        }
        diagnostics.insert(
            parameter.to_string(),
            ProfileParameterDiagnostic {
                status: ProfileParameterStatus::NonMonotone,
                reason,
                estimate: estimate_value.unwrap_or(f64::NAN),
            },
        );
    }
    if values.len() < 5 {
        return Err(MixedModelError::Optimization(format!(
            "profile_{parameter}: only {} evaluations produced — refusing sparse profile spline; try a larger threshold",
            values.len()
        )));
    }
    for w in zetas.windows(2) {
        if !(w[1] > w[0]) {
            return Err(MixedModelError::Optimization(format!(
                "profile_{parameter}: {NON_MONOTONE_MESSAGE} — refusing to invert"
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
                    sdcor: None,
                },
                ProfileRow {
                    p: "σ".to_string(),
                    zeta: 0.0,
                    sigma: 5.0,
                    beta: Vec::new(),
                    theta: Vec::new(),
                    sdcor: None,
                },
                ProfileRow {
                    p: "σ".to_string(),
                    zeta: 2.0,
                    sigma: 6.0,
                    beta: Vec::new(),
                    theta: Vec::new(),
                    sdcor: None,
                },
            ],
            fwd: BTreeMap::new(),
            rev,
            diagnostics: BTreeMap::new(),
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
                sdcor: None,
            },
            ProfileRow {
                p: "θ1".to_string(),
                zeta: 0.0,
                sigma: 1.0,
                beta: Vec::new(),
                theta: vec![1.0],
                sdcor: None,
            },
            ProfileRow {
                p: "θ1".to_string(),
                zeta: 1.0,
                sigma: 1.0,
                beta: Vec::new(),
                theta: vec![1.1],
                sdcor: None,
            },
        ];
        let mut fwd = BTreeMap::new();
        let mut rev = BTreeMap::new();

        let mut rows = rows;
        let mut diagnostics = BTreeMap::new();
        let err = add_profile_splines(
            "θ1",
            &mut rows,
            Some(1.0),
            &mut fwd,
            &mut rev,
            &mut diagnostics,
            |row| row.theta[0],
        )
        .unwrap_err();

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
        // `.sig01` is the lme4-scale (SD) profile of the batch intercept.
        assert_eq!(profile_row_order(&pr), vec!["σ", "β1", "θ1", ".sig01"]);

        let ci = pr.confint(0.95).unwrap();
        let parameters = ci
            .iter()
            .map(|row| row.parameter.as_str())
            .collect::<Vec<_>>();
        assert!(parameters.contains(&"σ"));
        assert!(parameters.contains(&"θ1"));
        assert!(parameters.contains(&"β1"));
    }

    fn profile_bits(pr: &MixedModelProfile) -> Vec<(String, Vec<u64>)> {
        pr.tbl
            .iter()
            .map(|row| {
                let mut bits = vec![row.zeta.to_bits(), row.sigma.to_bits()];
                bits.extend(row.beta.iter().map(|v| v.to_bits()));
                bits.extend(row.theta.iter().map(|v| v.to_bits()));
                (row.p.clone(), bits)
            })
            .collect()
    }

    /// The per-parameter profiles run on worker threads must reproduce the
    /// serial profile bit for bit (rows, order, and splines via confint),
    /// and leave the model's fitted state untouched.
    #[test]
    fn profile_with_threads_is_bit_identical_to_serial() {
        for (data, formula) in [
            (dyestuff_fixture(), "yield ~ 1 + (1 | batch)"),
            (random_slope_fixture(), "y ~ 1 + x + (1 + x | g)"),
        ] {
            let mut model =
                LinearMixedModel::new(parse_formula(formula).unwrap(), &data, None).unwrap();
            model.fit(false).unwrap();
            let theta_before = model.theta();
            let serial = profile(&mut model).unwrap();
            for threads in [1, 2, 3] {
                let parallel =
                    profile_with_options(&mut model, &ProfileOptions { threads }).unwrap();
                assert_eq!(
                    profile_bits(&parallel),
                    profile_bits(&serial),
                    "{formula} threads={threads}"
                );
                let (a, b) = (
                    parallel.confint(0.95).unwrap(),
                    serial.confint(0.95).unwrap(),
                );
                assert_eq!(format!("{a:?}"), format!("{b:?}"));
            }
            assert_eq!(model.theta(), theta_before);
        }
        let data = dyestuff_fixture();
        let mut model = LinearMixedModel::new(
            parse_formula("yield ~ 1 + (1 | batch)").unwrap(),
            &data,
            None,
        )
        .unwrap();
        model.fit(false).unwrap();
        assert!(profile_with_options(&mut model, &ProfileOptions { threads: 0 }).is_err());
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
            // The Julia reference has no lme4-scale `.sigNN` profiles.
            let order = profile_row_order(&pr)
                .into_iter()
                .filter(|name| !name.starts_with(".sig"))
                .collect::<Vec<_>>();
            assert_eq!(
                order, case.rust_row_order,
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

    #[test]
    fn sdcor_profile_intervals_match_lme4_confint_profile() {
        // lme4: confint(lmer(Reaction ~ Days + (Days | Subject), sleepstudy))
        let (data, _) = datasets::load("sleepstudy").unwrap();
        let formula = parse_formula("Reaction ~ Days + (Days | Subject)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(true).unwrap();
        let theta_before = model.theta();

        // Round trip between θ and the lme4 scale.
        let sdcor = sdcor_from_theta(&model, &model.theta(), model.sigma());
        let back = theta_from_sdcor(&model, &sdcor, model.sigma()).unwrap();
        for (a, b) in back.iter().zip(model.theta()) {
            assert!((a - b).abs() < 1e-10);
        }

        let expected = [
            (".sig01", 14.3814181608, 37.7159953183),
            (".sig02", -0.4815007583, 0.6849862733),
            (".sig03", 3.8011640533, 8.7533807531),
        ];
        let profile = profile_sdcors(&mut model, 4.0);
        let rows = profile.confint(0.95).unwrap();
        for (name, lower, upper) in expected {
            let row = rows
                .iter()
                .find(|row| row.parameter == name)
                .unwrap_or_else(|| panic!("{name} profiled: {rows:?}"));
            let tol = 2e-3 * (upper - lower);
            assert!(
                (row.lower - lower).abs() < tol,
                "{name} lower {} vs {lower}",
                row.lower
            );
            assert!(
                (row.upper - upper).abs() < tol,
                "{name} upper {} vs {upper}",
                row.upper
            );
        }
        // The model is restored and the θ-scale profile is still available.
        assert_eq!(model.theta(), theta_before);
        let sigma = profile_sdcor(&mut model, 3, 4.0).unwrap();
        let row = &sigma.confint(0.95).unwrap()[0];
        assert_eq!(row.parameter, ".sigma");
        assert!((row.lower - 22.8982668821).abs() < 0.02, "{row:?}");
        assert!((row.upper - 28.8579965114).abs() < 0.02, "{row:?}");
    }

    #[test]
    fn sdcor_profile_scalar_term_ml_matches_lme4() {
        // lme4: confint(lmer(Reaction ~ Days + (1 | Subject), sleepstudy,
        //                    REML = FALSE))[".sig01", ]
        let (data, _) = datasets::load("sleepstudy").unwrap();
        let formula = parse_formula("Reaction ~ Days + (1 | Subject)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();
        let payload = profile_confint_payload(&mut model, 0.95).unwrap();
        let sig = payload
            .intervals
            .iter()
            .find(|row| row.parameter == ".sig01")
            .expect(".sig01 interval in the payload");
        assert!((sig.lower - 26.007120448).abs() < 0.05, "{sig:?}");
        assert!((sig.upper - 52.93598353).abs() < 0.05, "{sig:?}");
        // θ rows remain.
        assert!(payload.intervals.iter().any(|row| row.parameter == "θ1"));
    }

    /// Synthetic `y ~ 1 + x + (1 + x | g)` data (a small LCG + Box–Muller
    /// simulation; the same generator produced the CSVs the lme4 reference
    /// values below were computed from).
    fn synthetic_random_slope(seed: u64) -> DataFrame {
        let mut state = seed * 7919 + 1;
        let mut uniform = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let mut normal = || {
            let (a, b) = (uniform().max(1e-300), uniform());
            (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
        };
        let n_groups = 6 + (seed % 5) as usize;
        let per_group = 4 + (seed % 3) as usize;
        let slope_sd = [0.05, 0.2, 0.5, 1.0][(seed % 4) as usize];
        let (mut y, mut x, mut g) = (Vec::new(), Vec::new(), Vec::new());
        for group in 0..n_groups {
            let u0 = normal();
            let u1 = slope_sd * normal();
            for _ in 0..per_group {
                let xv = normal();
                x.push(xv);
                y.push(1.0 + 0.5 * xv + u0 + u1 * xv + 0.7 * normal());
                g.push(format!("g{group}"));
            }
        }
        let mut df = DataFrame::new();
        df.add_numeric("y", y).unwrap();
        df.add_numeric("x", x).unwrap();
        df.add_categorical("g", g).unwrap();
        df
    }

    fn fit_synthetic_random_slope(seed: u64) -> LinearMixedModel {
        let data = synthetic_random_slope(seed);
        let mut model = LinearMixedModel::new(
            parse_formula("y ~ 1 + x + (1 + x | g)").unwrap(),
            &data,
            None,
        )
        .unwrap();
        model.fit(false).unwrap();
        model
    }

    fn payload_row<'a>(
        payload: &'a ProfileLikelihoodCiPayload,
        parameter: &str,
    ) -> &'a ProfileLikelihoodCiRow {
        payload
            .intervals
            .iter()
            .find(|row| row.parameter == parameter)
            .unwrap_or_else(|| panic!("{parameter} missing from payload"))
    }

    /// lme4 reference: R 4.3.3, lme4 1.1.35.1,
    /// `confint(lmer(y ~ 1 + x + (1 + x | g), d, REML = FALSE),
    /// method = "profile", oldNames = TRUE)` on `synthetic_random_slope(seed)`
    /// (written to CSV with Rust's round-trip float formatting).
    ///
    /// Seed 3: lme4 profiles cleanly; the engine's `.sig02` (correlation)
    /// profile turns non-monotone far in the tail (ζ ≈ −3.95 → −2.87, past
    /// the 95% cutoff), which used to abort the whole profile payload. Now
    /// the monotone segment around the estimate still yields lme4's interval
    /// and the parameter is flagged `non_monotone`.
    ///
    /// Seed 1: a singular fit (slope SD at 0) whose correlation profile is
    /// non-monotone; lme4 warns ("non-monotonic profile for .sig02") and
    /// falls back to linear interpolation, reporting the uninformative
    /// [-1, 1]. The engine reports `NaN` bounds with a typed
    /// `non_monotone` status while every other interval matches lme4.
    #[test]
    fn non_monotone_correlation_profile_is_recorded_not_fatal() {
        let cases: [(u64, &[(&str, f64, f64)]); 2] = [
            (
                3,
                &[
                    (".sig01", 0.731777154, 2.13554526),
                    (".sig02", -0.403132904, 0.85876699),
                    (".sig03", 0.396933478, 1.92730617),
                    ("σ", 0.523282650, 1.04156835),
                    ("β1", 0.193922981, 2.03346729),
                    ("β2", -0.058305737, 1.58048353),
                ],
            ),
            (
                1,
                &[
                    (".sig01", 0.768155033, 2.31653535),
                    (".sig03", 0.0, 0.40481800),
                    ("σ", 0.292303778, 0.53460347),
                    ("β1", -0.034432107, 2.07429377),
                    ("β2", 0.428645580, 0.83691715),
                ],
            ),
        ];
        for (seed, expected) in cases {
            let mut model = fit_synthetic_random_slope(seed);
            let payload = profile_confint_payload(&mut model, 0.95)
                .unwrap_or_else(|e| panic!("seed {seed}: payload must not abort: {e}"));
            for &(parameter, lower, upper) in expected {
                let row = payload_row(&payload, parameter);
                assert!(
                    (row.lower - lower).abs() < 5e-3 && (row.upper - upper).abs() < 5e-3,
                    "seed {seed} {parameter}: got [{}, {}], lme4 [{lower}, {upper}]",
                    row.lower,
                    row.upper
                );
            }
            let corr = payload_row(&payload, ".sig02");
            assert_eq!(
                corr.status,
                ProfileParameterStatus::NonMonotone,
                "seed {seed}: {corr:?}"
            );
            assert_eq!(corr.regularity, "non_monotone_profile");
            assert!(corr
                .reason
                .as_deref()
                .is_some_and(|r| r.contains("not strictly monotone")));
            if seed == 1 {
                assert!(corr.lower.is_nan() && corr.upper.is_nan(), "{corr:?}");
            }
            // Every θ profile now succeeds (the off-diagonal θ2 has a
            // negative estimate in seed 1, which the old multiplicative walk
            // mislabelled as a non-monotone profile and aborted on).
            for theta in ["θ1", "θ2", "θ3"] {
                let row = payload_row(&payload, theta);
                assert_eq!(
                    row.status,
                    ProfileParameterStatus::Ok,
                    "seed {seed} {row:?}"
                );
            }
            assert!(payload.notes.iter().any(|n| n.contains("non-`ok` status")));

            // The NaN bounds travel as JSON null and read back as NaN.
            let json = payload.to_json().unwrap();
            let decoded = ProfileLikelihoodCiPayload::from_json(&json).unwrap();
            let back = decoded
                .intervals
                .iter()
                .find(|row| row.parameter == ".sig02")
                .unwrap();
            assert_eq!(back.status, ProfileParameterStatus::NonMonotone);
            assert_eq!(back.lower.is_nan(), corr.lower.is_nan());
            assert_eq!(back.upper.is_nan(), corr.upper.is_nan());
        }
    }

    /// Threaded profiles stay bit-identical to the serial one when a
    /// per-parameter profile is irregular, diagnostics included.
    #[test]
    fn irregular_profile_threads_match_serial() {
        let mut model = fit_synthetic_random_slope(1);
        let serial = profile(&mut model).unwrap();
        assert!(!serial.diagnostics.is_empty());
        for threads in [2, 3] {
            let parallel = profile_with_options(&mut model, &ProfileOptions { threads }).unwrap();
            assert_eq!(profile_bits(&parallel), profile_bits(&serial));
            assert_eq!(parallel.diagnostics, serial.diagnostics);
            assert_eq!(
                format!("{:?}", parallel.confint(0.95).unwrap()),
                format!("{:?}", serial.confint(0.95).unwrap())
            );
        }
    }

    /// A profile that cannot be computed at all (β profiles of a weighted
    /// fit) is recorded as `failed` with NaN bounds instead of aborting the
    /// σ/θ intervals.
    #[test]
    fn failed_beta_profiles_are_recorded_per_parameter() {
        let data = dyestuff_fixture();
        let weights = (0..30)
            .map(|i| 1.0 + (i % 3) as f64 * 0.5)
            .collect::<Vec<_>>();
        let mut model = LinearMixedModel::new(
            parse_formula("yield ~ 1 + (1 | batch)").unwrap(),
            &data,
            Some(&weights),
        )
        .unwrap();
        model.fit(false).unwrap();
        let beta_hat = model.beta()[0];
        let payload = profile_confint_payload(&mut model, 0.95).unwrap();
        let beta = payload_row(&payload, "β1");
        assert_eq!(beta.status, ProfileParameterStatus::Failed);
        assert_eq!(beta.regularity, "profile_failed");
        assert!(beta.lower.is_nan() && beta.upper.is_nan());
        assert_eq!(beta.estimate, beta_hat);
        assert!(beta
            .reason
            .as_deref()
            .is_some_and(|r| r.contains("observation weights")));
        let sigma = payload_row(&payload, "σ");
        assert_eq!(sigma.status, ProfileParameterStatus::Ok);
        assert!(sigma.lower < sigma.estimate && sigma.estimate < sigma.upper);
    }
}
