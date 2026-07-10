//! Parametric bootstrap for the linear mixed model: replicate storage,
//! interval construction, and fixed-effect null refit orchestration. Moved
//! verbatim from the former single-file `linear.rs`.

use super::*;

// ── Parametric bootstrap ──────────────────────────────────────────────────────

/// A single parametric bootstrap replicate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BootstrapReplicate {
    /// Profile-likelihood objective (deviance or REML criterion).
    #[serde(with = "json_f64")]
    pub objective: f64,
    /// Residual standard deviation σ.
    #[serde(with = "json_f64")]
    pub sigma: f64,
    /// Fixed-effects coefficients (pivot order).
    #[serde(with = "json_dvector_f64")]
    pub beta: DVector<f64>,
    /// Fixed-effects standard errors (pivot order).
    #[serde(default = "default_bootstrap_se", with = "json_dvector_f64")]
    pub se: DVector<f64>,
    /// Variance-component θ parameters.
    pub theta: Vec<f64>,
}

/// Collection of parametric bootstrap replicates.
///
/// Mirrors `MixedModelBootstrap` in Julia's MixedModels.jl.
///
/// Produced by [`parametricbootstrap`].  Each replicate stores the
/// objective, residual σ, fixed-effects β, standard errors, and covariance θ for a
/// model fitted to a simulated response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MixedModelBootstrap {
    /// One entry per bootstrap replicate.
    pub fits: Vec<BootstrapReplicate>,
}

/// Confidence-interval construction method for bootstrap summaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum BootstrapIntervalMethod {
    /// Equal-tail percentile interval.
    Percentile,
    /// Shortest contiguous interval covering the requested level.
    Shortest,
}

/// One quantile row for a bootstrap statistic.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BootstrapQuantile {
    /// Statistic name: `objective`, `sigma`, `beta[i]`, or `theta[i]`.
    pub parameter: String,
    /// Requested probability in `[0, 1]`.
    pub probability: f64,
    /// Quantile value.
    pub value: f64,
    /// Number of finite replicate values used.
    pub n: usize,
}

/// One confidence-interval row for a bootstrap statistic.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BootstrapInterval {
    /// Statistic name: `objective`, `sigma`, `beta[i]`, or `theta[i]`.
    pub parameter: String,
    /// Requested coverage level in `(0, 1)`.
    pub level: f64,
    /// Lower endpoint.
    pub lower: f64,
    /// Upper endpoint.
    pub upper: f64,
    /// Number of finite replicate values used.
    pub n: usize,
    /// Interval construction method.
    pub method: BootstrapIntervalMethod,
}

/// Stable schema name for bootstrap-run payloads.
pub const BOOTSTRAP_RUN_SCHEMA: &str = "mixedmodels.bootstrap_run";
/// Stable schema version for bootstrap-run payloads.
pub const BOOTSTRAP_RUN_SCHEMA_VERSION: &str = "1.0.0";

/// Target distribution represented by a bootstrap run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum BootstrapTargetKind {
    /// Full fitted-model data-generating distribution.
    FullModelDistribution,
    /// Fixed-effect null distribution for a contrast.
    FixedEffectNull,
    /// Cluster-resampled empirical distribution.
    ClusterResample,
}

/// Description of the bootstrap target.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BootstrapTarget {
    /// Target distribution kind.
    pub kind: BootstrapTargetKind,
    /// Human-readable target label.
    pub label: String,
    /// Contrast label for fixed-effect null targets.
    pub contrast_label: Option<String>,
}

impl BootstrapTarget {
    /// Build a full-model bootstrap target.
    pub fn full_model_distribution(label: impl Into<String>) -> Self {
        Self {
            kind: BootstrapTargetKind::FullModelDistribution,
            label: label.into(),
            contrast_label: None,
        }
    }

    /// Build a fixed-effect null bootstrap target.
    pub fn fixed_effect_null(label: impl Into<String>, contrast_label: impl Into<String>) -> Self {
        Self {
            kind: BootstrapTargetKind::FixedEffectNull,
            label: label.into(),
            contrast_label: Some(contrast_label.into()),
        }
    }

    /// Build a cluster-resampling bootstrap target.
    pub fn cluster_resample(label: impl Into<String>) -> Self {
        Self {
            kind: BootstrapTargetKind::ClusterResample,
            label: label.into(),
            contrast_label: None,
        }
    }
}

/// Policy for bootstrap refits that fail numerically.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum BootstrapFailedRefitPolicy {
    /// Exclude failed refits from bootstrap summaries.
    Exclude,
    /// Count failed refits as extreme replicates for p-value accounting.
    CountExtreme,
    /// Stop the bootstrap when the first refit fails.
    Abort,
}

/// Options for fixed-effect bootstrap inference.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FixedEffectBootstrapOptions {
    /// Number of bootstrap replicates requested.
    pub requested_replicates: usize,
    /// How to handle failed bootstrap refits.
    pub failed_refit_policy: BootstrapFailedRefitPolicy,
    /// Optional deterministic seed for `StdRng`.
    pub seed: Option<u64>,
}

impl Default for FixedEffectBootstrapOptions {
    fn default() -> Self {
        Self {
            requested_replicates: 999,
            failed_refit_policy: BootstrapFailedRefitPolicy::Exclude,
            seed: None,
        }
    }
}

/// Reproducibility record for a bootstrap run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BootstrapSeedRecord {
    /// Random-number generator name.
    pub rng: String,
    /// Recorded seed when available.
    pub seed: Option<u64>,
    /// Human-readable reproducibility note.
    pub reproducibility_note: String,
}

impl BootstrapSeedRecord {
    /// Build a record for an unrecorded seed.
    pub fn unspecified() -> Self {
        Self {
            rng: "unknown".to_string(),
            seed: None,
            reproducibility_note:
                "seed was not recorded; bootstrap run is not exactly reproducible".to_string(),
        }
    }

    /// Build a record for a `StdRng` seed.
    pub fn std_rng(seed: u64) -> Self {
        Self {
            rng: "StdRng".to_string(),
            seed: Some(seed),
            reproducibility_note: "bootstrap seed recorded by Rust caller".to_string(),
        }
    }
}

/// Refit settings used inside bootstrap replicates.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BootstrapRefitOptions {
    /// Whether refits use REML.
    pub reml: bool,
    /// Optimizer backend label.
    pub backend: String,
    /// Optimizer label.
    pub optimizer: String,
}

impl BootstrapRefitOptions {
    /// Capture refit options from a fitted model.
    pub fn from_model(model: &LinearMixedModel) -> Self {
        Self {
            reml: model.optsum.reml,
            backend: model.optsum.backend_name().to_string(),
            optimizer: model.optsum.optimizer_name().to_string(),
        }
    }
}

/// Metadata attached to a bootstrap-run payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BootstrapRunMetadata {
    /// Stable schema name.
    pub schema_name: String,
    /// Stable schema version.
    pub schema_version: String,
    /// Bootstrap target.
    pub target: BootstrapTarget,
    /// Number of replicates requested by the caller.
    pub requested_replicates: usize,
    /// Number of replicates actually collected.
    pub completed_replicates: usize,
    /// Number of replicates with finite successful refits.
    pub successful_replicates: usize,
    /// Number of failed refits.
    pub failed_refits: usize,
    /// Failed-refit accounting policy.
    pub failed_refit_policy: BootstrapFailedRefitPolicy,
    /// Successful refits ending on a covariance boundary.
    pub boundary_count: usize,
    /// Boundary refit fraction among successful replicates.
    pub boundary_rate: Option<f64>,
    /// Randomness/reproducibility record.
    pub seed_record: BootstrapSeedRecord,
    /// Refit settings used for bootstrap replicates.
    pub refit_options: BootstrapRefitOptions,
    /// Statistic label when scalar replicate statistics are attached.
    pub statistic_label: Option<String>,
    /// Count of finite attached replicate statistics.
    pub finite_statistic_count: Option<usize>,
    /// Monte Carlo standard error for a bootstrap p-value, when available.
    pub mcse: Option<f64>,
    /// Reader-facing caveats and diagnostics.
    pub notes: Vec<String>,
}

/// Serializable bootstrap run with metadata and optional scalar summaries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BootstrapRunPayload {
    /// Run metadata and accounting.
    pub metadata: BootstrapRunMetadata,
    /// Per-replicate fitted quantities.
    pub replicates: MixedModelBootstrap,
    /// Optional scalar statistic per replicate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replicate_statistics: Option<Vec<f64>>,
    /// Optional bootstrap intervals for scalar statistics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intervals: Option<Vec<BootstrapInterval>>,
}

/// Covariance policy for fixed-effect null bootstrap simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum FixedEffectNullCovariancePolicy {
    /// Reuse the fitted random-effect covariance and residual scale.
    ReuseFittedCovariance,
}

/// Null data-generating state for fixed-effect bootstrap simulation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FixedEffectNullBootstrapTarget {
    /// Bootstrap target descriptor.
    pub target: BootstrapTarget,
    /// Covariance handling policy.
    pub covariance_policy: FixedEffectNullCovariancePolicy,
    /// Coefficient names aligned with `beta_fitted` and `beta_null`.
    pub coefficient_names: Vec<String>,
    /// Fitted fixed-effect coefficients.
    #[serde(with = "json_dvector_f64")]
    pub beta_fitted: DVector<f64>,
    /// Null-constrained fixed-effect coefficients.
    #[serde(with = "json_dvector_f64")]
    pub beta_null: DVector<f64>,
    /// Fitted covariance-parameter vector reused under the null.
    pub theta: Vec<f64>,
    /// Fitted residual scale reused under the null.
    #[serde(with = "json_f64")]
    pub sigma: f64,
    /// Whether the source model was REML-fitted.
    pub reml: bool,
    /// Reader-facing caveats and diagnostics.
    pub notes: Vec<String>,
}

impl MixedModelBootstrap {
    /// Number of replicates.
    pub fn len(&self) -> usize {
        self.fits.len()
    }

    /// `true` if no replicates were collected.
    pub fn is_empty(&self) -> bool {
        self.fits.is_empty()
    }

    /// Objectives across all replicates.
    pub fn objectives(&self) -> Vec<f64> {
        self.fits.iter().map(|f| f.objective).collect()
    }

    /// Residual σ values across all replicates.
    pub fn sigmas(&self) -> Vec<f64> {
        self.fits.iter().map(|f| f.sigma).collect()
    }

    /// Fixed-effects β vectors across all replicates, shape `n_replicates × p`.
    pub fn betas(&self) -> Vec<DVector<f64>> {
        self.fits.iter().map(|f| f.beta.clone()).collect()
    }

    /// Fixed-effects standard-error vectors across all replicates.
    pub fn standard_errors(&self) -> Vec<DVector<f64>> {
        self.fits.iter().map(|f| f.se.clone()).collect()
    }

    /// Julia-style alias for [`MixedModelBootstrap::standard_errors`].
    pub fn ses(&self) -> Vec<DVector<f64>> {
        self.standard_errors()
    }

    /// θ parameter vectors across all replicates.
    pub fn thetas(&self) -> Vec<Vec<f64>> {
        self.fits.iter().map(|f| f.theta.clone()).collect()
    }

    /// Quantiles for all scalar bootstrap statistics.
    ///
    /// Non-finite replicate values are ignored parameter-by-parameter. The
    /// quantile rule is linear interpolation between adjacent order statistics
    /// (R's type-7 convention).
    pub fn quantiles(&self, probability: f64) -> Result<Vec<BootstrapQuantile>> {
        validate_probability(probability)?;

        self.parameter_series()?
            .into_iter()
            .map(|(parameter, mut values)| {
                values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                Ok(BootstrapQuantile {
                    parameter,
                    probability,
                    value: quantile_sorted(&values, probability),
                    n: values.len(),
                })
            })
            .collect()
    }

    /// Equal-tail percentile confidence intervals for all scalar bootstrap statistics.
    pub fn percentile_intervals(&self, level: f64) -> Result<Vec<BootstrapInterval>> {
        validate_level(level)?;
        let alpha = (1.0 - level) / 2.0;

        self.parameter_series()?
            .into_iter()
            .map(|(parameter, mut values)| {
                values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                Ok(BootstrapInterval {
                    parameter,
                    level,
                    lower: quantile_sorted(&values, alpha),
                    upper: quantile_sorted(&values, 1.0 - alpha),
                    n: values.len(),
                    method: BootstrapIntervalMethod::Percentile,
                })
            })
            .collect()
    }

    /// Alias for equal-tail percentile intervals.
    pub fn confidence_intervals(&self, level: f64) -> Result<Vec<BootstrapInterval>> {
        self.percentile_intervals(level)
    }

    /// Shortest contiguous confidence intervals for all scalar bootstrap statistics.
    ///
    /// This mirrors the `shortestcovint` summary helper used by MixedModels.jl.
    pub fn shortest_intervals(&self, level: f64) -> Result<Vec<BootstrapInterval>> {
        validate_level(level)?;

        self.parameter_series()?
            .into_iter()
            .map(|(parameter, mut values)| {
                let (lower, upper) = shortest_interval(&mut values, level);
                Ok(BootstrapInterval {
                    parameter,
                    level,
                    lower,
                    upper,
                    n: values.len(),
                    method: BootstrapIntervalMethod::Shortest,
                })
            })
            .collect()
    }

    /// Save bootstrap replicates as JSON.
    ///
    /// The JSON form is intentionally just the replicate collection, so it can
    /// be restored independently and then validated against a model template.
    pub fn save_replicates<W: std::io::Write>(
        &self,
        writer: W,
    ) -> std::result::Result<(), serde_json::Error> {
        serde_json::to_writer(writer, self)
    }

    /// Restore bootstrap replicates from JSON.
    pub fn restore_replicates<R: std::io::Read>(
        reader: R,
    ) -> std::result::Result<Self, serde_json::Error> {
        serde_json::from_reader(reader)
    }

    /// Validate restored replicate dimensions against a model template.
    pub fn validate_for_model(&self, model: &LinearMixedModel) -> Result<()> {
        let expected_beta = model.feterm.rank;
        let expected_theta = model.n_theta();

        for (idx, fit) in self.fits.iter().enumerate() {
            if fit.beta.len() != expected_beta {
                return Err(MixedModelError::InvalidArgument(format!(
                    "bootstrap replicate {idx} beta length ({}) does not match model fixed-effect rank ({expected_beta})",
                    fit.beta.len()
                )));
            }
            if fit.theta.len() != expected_theta {
                return Err(MixedModelError::InvalidArgument(format!(
                    "bootstrap replicate {idx} theta length ({}) does not match model theta length ({expected_theta})",
                    fit.theta.len()
                )));
            }
            if !fit.se.is_empty() && fit.se.len() != expected_beta {
                return Err(MixedModelError::InvalidArgument(format!(
                    "bootstrap replicate {idx} se length ({}) does not match model fixed-effect rank ({expected_beta})",
                    fit.se.len()
                )));
            }
        }

        Ok(())
    }

    /// Build metadata for these replicates against a fitted model template.
    pub fn run_metadata_for_model(
        &self,
        model: &LinearMixedModel,
        target: BootstrapTarget,
        requested_replicates: usize,
        failed_refit_policy: BootstrapFailedRefitPolicy,
        seed_record: BootstrapSeedRecord,
        refit_options: BootstrapRefitOptions,
        statistic_label: Option<String>,
        statistic_values: Option<&[f64]>,
        p_value: Option<f64>,
    ) -> BootstrapRunMetadata {
        let lower_bounds = model.lower_bounds();
        let successful_replicates = self.fits.iter().filter(|fit| fit.is_successful()).count();
        let boundary_count = self
            .fits
            .iter()
            .filter(|fit| fit.is_successful() && fit.is_boundary_refit(&lower_bounds, 1e-8))
            .count();
        let finite_statistic_count =
            statistic_values.map(|values| values.iter().filter(|value| value.is_finite()).count());
        let boundary_rate = (successful_replicates > 0)
            .then_some(boundary_count as f64 / successful_replicates as f64);
        let mcse = p_value.and_then(|p| {
            (p.is_finite() && (0.0..=1.0).contains(&p) && successful_replicates > 0)
                .then_some((p * (1.0 - p) / successful_replicates as f64).sqrt())
        });

        let mut notes = Vec::new();
        if target.kind == BootstrapTargetKind::FullModelDistribution {
            notes.push(
                "full-model bootstrap distributions do not certify fixed-effect hypothesis-test p-values"
                    .to_string(),
            );
        }
        if requested_replicates != self.len() {
            notes.push(format!(
                "requested {requested_replicates} bootstrap replicate(s), collected {}",
                self.len()
            ));
        }
        if successful_replicates < self.len() {
            notes.push(format!(
                "{} bootstrap refit(s) did not produce finite estimates",
                self.len() - successful_replicates
            ));
        }
        if boundary_count > 0 {
            notes.push(format!(
                "{boundary_count} successful bootstrap refit(s) ended on a covariance boundary"
            ));
        }

        BootstrapRunMetadata {
            schema_name: BOOTSTRAP_RUN_SCHEMA.to_string(),
            schema_version: BOOTSTRAP_RUN_SCHEMA_VERSION.to_string(),
            target,
            requested_replicates,
            completed_replicates: self.len(),
            successful_replicates,
            failed_refits: self.len().saturating_sub(successful_replicates),
            failed_refit_policy,
            boundary_count,
            boundary_rate,
            seed_record,
            refit_options,
            statistic_label,
            finite_statistic_count,
            mcse,
            notes,
        }
    }

    /// Attach metadata and return a bootstrap payload.
    pub fn into_run_payload(self, metadata: BootstrapRunMetadata) -> BootstrapRunPayload {
        BootstrapRunPayload {
            metadata,
            replicates: self,
            replicate_statistics: None,
            intervals: None,
        }
    }

    /// Attach metadata plus scalar replicate statistics and return a payload.
    pub fn into_run_payload_with_statistics(
        self,
        metadata: BootstrapRunMetadata,
        replicate_statistics: Vec<f64>,
    ) -> BootstrapRunPayload {
        BootstrapRunPayload {
            metadata,
            replicates: self,
            replicate_statistics: Some(replicate_statistics),
            intervals: None,
        }
    }

    /// Attach metadata, scalar statistics, and intervals and return a payload.
    pub fn into_run_payload_with_statistics_and_intervals(
        self,
        metadata: BootstrapRunMetadata,
        replicate_statistics: Vec<f64>,
        intervals: Vec<BootstrapInterval>,
    ) -> BootstrapRunPayload {
        BootstrapRunPayload {
            metadata,
            replicates: self,
            replicate_statistics: Some(replicate_statistics),
            intervals: Some(intervals),
        }
    }

    fn parameter_series(&self) -> Result<Vec<(String, Vec<f64>)>> {
        if self.fits.is_empty() {
            return Err(MixedModelError::InvalidArgument(
                "cannot summarize an empty bootstrap sample".to_string(),
            ));
        }

        let beta_len = self.fits[0].beta.len();
        let se_len = self.fits[0].se.len();
        let theta_len = self.fits[0].theta.len();
        for (idx, fit) in self.fits.iter().enumerate() {
            if fit.beta.len() != beta_len {
                return Err(MixedModelError::InvalidArgument(format!(
                    "bootstrap replicate {idx} beta length ({}) does not match first replicate ({beta_len})",
                    fit.beta.len()
                )));
            }
            if fit.se.len() != se_len {
                return Err(MixedModelError::InvalidArgument(format!(
                    "bootstrap replicate {idx} se length ({}) does not match first replicate ({se_len})",
                    fit.se.len()
                )));
            }
            if fit.theta.len() != theta_len {
                return Err(MixedModelError::InvalidArgument(format!(
                    "bootstrap replicate {idx} theta length ({}) does not match first replicate ({theta_len})",
                    fit.theta.len()
                )));
            }
        }

        let mut series = Vec::with_capacity(2 + beta_len + se_len + theta_len);
        series.push((
            "objective".to_string(),
            self.fits
                .iter()
                .map(|fit| fit.objective)
                .filter(|value| value.is_finite())
                .collect(),
        ));
        series.push((
            "sigma".to_string(),
            self.fits
                .iter()
                .map(|fit| fit.sigma)
                .filter(|value| value.is_finite())
                .collect(),
        ));

        for idx in 0..beta_len {
            series.push((
                format!("beta[{idx}]"),
                self.fits
                    .iter()
                    .map(|fit| fit.beta[idx])
                    .filter(|value| value.is_finite())
                    .collect(),
            ));
        }
        for idx in 0..se_len {
            series.push((
                format!("se[{idx}]"),
                self.fits
                    .iter()
                    .map(|fit| fit.se[idx])
                    .filter(|value| value.is_finite())
                    .collect(),
            ));
        }
        for idx in 0..theta_len {
            series.push((
                format!("theta[{idx}]"),
                self.fits
                    .iter()
                    .map(|fit| fit.theta[idx])
                    .filter(|value| value.is_finite())
                    .collect(),
            ));
        }

        series.retain(|(_, values): &(String, Vec<f64>)| !values.is_empty());
        if series.is_empty() {
            return Err(MixedModelError::InvalidArgument(
                "bootstrap sample has no finite scalar statistics to summarize".to_string(),
            ));
        }

        Ok(series)
    }
}

impl BootstrapReplicate {
    pub(super) fn is_successful(&self) -> bool {
        self.objective.is_finite()
            && self.sigma.is_finite()
            && self.beta.iter().all(|value| value.is_finite())
            && self.se.iter().all(|value| value.is_finite())
            && self.theta.iter().all(|value| value.is_finite())
    }

    fn is_boundary_refit(&self, lower_bounds: &[f64], tolerance: f64) -> bool {
        self.theta.iter().enumerate().any(|(idx, theta)| {
            lower_bounds
                .get(idx)
                .copied()
                .filter(|lower| lower.is_finite())
                .is_some_and(|lower| *theta <= lower + tolerance)
        })
    }
}

fn default_bootstrap_se() -> DVector<f64> {
    DVector::zeros(0)
}

fn validate_probability(probability: f64) -> Result<()> {
    if probability.is_finite() && (0.0..=1.0).contains(&probability) {
        Ok(())
    } else {
        Err(MixedModelError::InvalidArgument(format!(
            "quantile probability must be in [0,1]; got {probability}"
        )))
    }
}

pub(super) fn validate_level(level: f64) -> Result<()> {
    if level.is_finite() && (0.0..1.0).contains(&level) {
        Ok(())
    } else {
        Err(MixedModelError::InvalidArgument(format!(
            "confidence level must be in (0,1); got {level}"
        )))
    }
}

pub(super) fn quantile_sorted(values: &[f64], probability: f64) -> f64 {
    debug_assert!(!values.is_empty());
    if values.len() == 1 {
        return values[0];
    }
    let h = probability * (values.len() - 1) as f64;
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;
    if lo == hi {
        values[lo]
    } else {
        values[lo] + (h - lo as f64) * (values[hi] - values[lo])
    }
}

fn shortest_interval(values: &mut [f64], level: f64) -> (f64, f64) {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = values.len();
    let ilen = ((n as f64) * level).ceil() as usize;
    if ilen >= n {
        return (values[0], values[n - 1]);
    }
    let mut min_len = f64::INFINITY;
    let mut best_i = 0;
    for i in 0..=(n - ilen) {
        let len = values[i + ilen - 1] - values[i];
        if len < min_len {
            min_len = len;
            best_i = i;
        }
    }
    (values[best_i], values[best_i + ilen - 1])
}

mod json_f64 {
    use serde::de::Error;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(value: &f64, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if value.is_finite() {
            serializer.serialize_f64(*value)
        } else if value.is_nan() {
            serializer.serialize_str("NaN")
        } else if value.is_sign_positive() {
            serializer.serialize_str("Infinity")
        } else {
            serializer.serialize_str("-Infinity")
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> std::result::Result<f64, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum JsonF64 {
            Number(f64),
            Special(String),
        }

        match JsonF64::deserialize(deserializer)? {
            JsonF64::Number(value) => Ok(value),
            JsonF64::Special(value) => match value.as_str() {
                "NaN" => Ok(f64::NAN),
                "Infinity" => Ok(f64::INFINITY),
                "-Infinity" => Ok(f64::NEG_INFINITY),
                _ => Err(D::Error::custom(format!(
                    "invalid non-finite float marker `{value}`"
                ))),
            },
        }
    }
}

mod json_dvector_f64 {
    use nalgebra::DVector;
    use serde::de::Error;
    use serde::ser::SerializeSeq;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(value: &DVector<f64>, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(value.len()))?;
        for entry in value.iter() {
            seq.serialize_element(&JsonF64(*entry))?;
        }
        seq.end()
    }

    pub fn deserialize<'de, D>(deserializer: D) -> std::result::Result<DVector<f64>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let values = Vec::<JsonF64>::deserialize(deserializer)?;
        Ok(DVector::from_vec(
            values.into_iter().map(|value| value.0).collect(),
        ))
    }

    struct JsonF64(f64);

    impl Serialize for JsonF64 {
        fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            if self.0.is_finite() {
                serializer.serialize_f64(self.0)
            } else if self.0.is_nan() {
                serializer.serialize_str("NaN")
            } else if self.0.is_sign_positive() {
                serializer.serialize_str("Infinity")
            } else {
                serializer.serialize_str("-Infinity")
            }
        }
    }

    impl<'de> Deserialize<'de> for JsonF64 {
        fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            #[derive(Deserialize)]
            #[serde(untagged)]
            enum JsonF64Value {
                Number(f64),
                Special(String),
                Null(Option<()>),
            }

            match JsonF64Value::deserialize(deserializer)? {
                JsonF64Value::Number(value) => Ok(JsonF64(value)),
                JsonF64Value::Special(value) => match value.as_str() {
                    "NaN" => Ok(JsonF64(f64::NAN)),
                    "Infinity" => Ok(JsonF64(f64::INFINITY)),
                    "-Infinity" => Ok(JsonF64(f64::NEG_INFINITY)),
                    _ => Err(D::Error::custom(format!(
                        "invalid non-finite float marker `{value}`"
                    ))),
                },
                JsonF64Value::Null(None) => Ok(JsonF64(f64::NAN)),
                JsonF64Value::Null(Some(())) => Err(D::Error::custom(
                    "invalid unit value in floating-point vector",
                )),
            }
        }
    }
}

/// Run a parametric bootstrap for a fitted `LinearMixedModel`.
///
/// For each of `n_rep` replicates:
/// 1. Simulate a new response from the fitted model.
/// 2. Refit the model to the simulated response.
/// 3. Record `(objective, σ, β, se, θ)`.
///
/// Returns a [`MixedModelBootstrap`] holding all replicates.
///
/// Mirrors `parametricbootstrap(rng, n, m)` in Julia's MixedModels.jl.
///
/// # Arguments
/// * `rng`   – random-number generator (e.g. `rand::rngs::StdRng`)
/// * `n_rep` – number of bootstrap replicates
/// * `model` – a *fitted* `LinearMixedModel` (used as the template)
pub fn parametricbootstrap<R: rand::Rng>(
    rng: &mut R,
    n_rep: usize,
    model: &LinearMixedModel,
) -> MixedModelBootstrap {
    // Preserve the original infallible API. Hosts that install an interrupt
    // callback should call `try_parametricbootstrap`, whose `Result` can carry
    // callback errors across the FFI boundary.
    let mut template = model.clone();
    template.progress_callback = None;
    run_parametricbootstrap(rng, n_rep, &template, false)
        .expect("an LMM bootstrap without a host callback is infallible")
}

/// Run a parametric bootstrap with host progress and interrupt propagation.
///
/// Unlike [`parametricbootstrap`], callback errors are returned immediately
/// instead of being classified as numerical replicate failures.
pub fn try_parametricbootstrap<R: rand::Rng>(
    rng: &mut R,
    n_rep: usize,
    model: &LinearMixedModel,
) -> Result<MixedModelBootstrap> {
    run_parametricbootstrap(rng, n_rep, model, true)
}

fn run_parametricbootstrap<R: rand::Rng>(
    rng: &mut R,
    n_rep: usize,
    model: &LinearMixedModel,
    report_progress: bool,
) -> Result<MixedModelBootstrap> {
    let mut fits = Vec::with_capacity(n_rep);
    let mut last_progress = 0usize;

    for replicate in 0..n_rep {
        // Simulate from the template (always use the original fitted model).
        let y_sim = model.simulate(rng);

        // Fresh clone of the template for this replicate. Replicates record
        // only (objective, sigma, beta, se, theta), so the optimizer
        // certificate's finite-difference derivative diagnostics are skipped;
        // the KKT-guided boundary restart still runs because it can change
        // the fitted estimates.
        let mut work = model.clone();
        work.suppress_derivative_diagnostics = true;

        match work.refit(y_sim.as_slice()) {
            Ok(()) => {
                fits.push(BootstrapReplicate {
                    objective: work.objective(),
                    sigma: work.sigma(),
                    beta: work.beta(),
                    se: work.stderror(),
                    theta: work.theta(),
                });
            }
            Err(error @ MixedModelError::Interrupted(_)) => return Err(error),
            Err(_) => {
                // On numerical failure, record the current (possibly partial) state.
                // Julia silently records the last accepted iterate in such cases.
                let beta = work.beta();
                fits.push(BootstrapReplicate {
                    objective: f64::NAN,
                    sigma: f64::NAN,
                    se: DVector::from_element(beta.len(), f64::NAN),
                    beta,
                    theta: work.theta(),
                });
            }
        }
        if report_progress {
            if let Some(callback) = &model.progress_callback {
                callback.report_if_due(
                    FitProgressPhase::Bootstrap,
                    replicate + 1,
                    Some(n_rep),
                    &mut last_progress,
                )?;
            }
        }
    }

    Ok(MixedModelBootstrap { fits })
}
