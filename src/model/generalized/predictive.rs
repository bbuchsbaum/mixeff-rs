//! GLMM predictive-distribution quadrature and quantile helpers.
//!
//! Moved verbatim from the former single-file `generalized.rs` during the
//! module split (bd-01KWG1BKEWB91RXAXC0350SFMK). No logic changes.

use super::*;

pub(crate) fn clean_glmm_prediction_variance_component(value: f64) -> Option<f64> {
    if !value.is_finite() || value < -1.0e-10 {
        return None;
    }
    Some(value.max(0.0))
}

/// Plug-in predictive summary for one future observation on the response
/// scale: law-of-total-variance moment plus predictive-distribution quantile
/// bounds.
pub(crate) struct GlmmFutureObservation {
    pub(crate) variance: f64,
    pub(crate) lower: f64,
    pub(crate) upper: f64,
}

/// Gauss-Hermite node count for predictive (future-observation) mixtures.
pub(crate) const GLMM_PREDICTIVE_QUADRATURE_POINTS: usize = 21;

/// Floor applied to conditional means before constructing count-family
/// predictive components, since the statrs distributions require a strictly
/// positive rate/mean.
pub(crate) const GLMM_PREDICTIVE_MEAN_FLOOR: f64 = 1.0e-12;

/// Smallest `t` (as f64) with `cdf(t) >= p` for a discrete mixture supported
/// on the non-negative integers. Doubles an upper bracket from `mean_hint`,
/// then binary-searches. `None` if the bracket never reaches `p`.
pub(crate) fn discrete_mixture_quantile(
    cdf: &dyn Fn(u64) -> f64,
    p: f64,
    mean_hint: f64,
) -> Option<f64> {
    let mut hi: u64 = if mean_hint.is_finite() && mean_hint > 1.0 {
        mean_hint.ceil() as u64
    } else {
        1
    };
    let mut expansions = 0;
    while cdf(hi) < p {
        if expansions >= 96 {
            return None;
        }
        hi = hi.saturating_mul(2).saturating_add(1);
        expansions += 1;
    }
    let mut lo: u64 = 0;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if cdf(mid) >= p {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    Some(lo as f64)
}

/// Quantile of a continuous mixture CDF by bracket expansion and bisection.
/// `domain_floor` clamps the lower bracket for positive-support families.
pub(crate) fn continuous_mixture_quantile(
    cdf: &dyn Fn(f64) -> f64,
    p: f64,
    domain_floor: Option<f64>,
    center: f64,
    spread: f64,
) -> Option<f64> {
    if !center.is_finite() || !spread.is_finite() {
        return None;
    }
    let step = spread.max(center.abs() * 1.0e-6).max(1.0e-12);
    let mut lo = center - 10.0 * step;
    let mut hi = center + 10.0 * step;
    if let Some(floor) = domain_floor {
        lo = lo.max(floor);
        hi = hi.max(floor + step);
    }
    let mut expansions = 0;
    while cdf(hi) < p {
        if expansions >= 256 || !hi.is_finite() {
            return None;
        }
        hi += (hi - lo).max(step);
        expansions += 1;
    }
    expansions = 0;
    while cdf(lo) > p {
        if expansions >= 256 || !lo.is_finite() {
            return None;
        }
        lo -= (hi - lo).max(step);
        if let Some(floor) = domain_floor {
            if lo <= floor {
                lo = floor;
                break;
            }
        }
        expansions += 1;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if !(mid > lo && mid < hi) {
            break;
        }
        if cdf(mid) >= p {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    Some(hi)
}

/// `ln Phi(x)` with an asymptotic tail for arguments where the direct CDF
/// underflows (needed by the inverse-Gaussian CDF's exponentially weighted
/// second term).
pub(crate) fn standard_normal_ln_cdf(x: f64) -> f64 {
    if x < -37.0 {
        // Mills-ratio asymptotic: Phi(x) ~ phi(x) / |x| for x -> -inf.
        -0.5 * x * x - (-x).ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
    } else {
        Normal::new(0.0, 1.0)
            .unwrap()
            .cdf(x)
            .max(f64::MIN_POSITIVE)
            .ln()
    }
}

/// CDF of the inverse-Gaussian distribution with mean `mu` and shape
/// `lambda`, evaluated in log space so the `exp(2 lambda / mu)` factor cannot
/// overflow against the matching normal tail.
pub(crate) fn inverse_gaussian_cdf(t: f64, mu: f64, lambda: f64) -> f64 {
    if t <= 0.0 {
        return 0.0;
    }
    let standard_normal = Normal::new(0.0, 1.0).unwrap();
    let sqrt_term = (lambda / t).sqrt();
    let first = standard_normal.cdf(sqrt_term * (t / mu - 1.0));
    let second = (2.0 * lambda / mu + standard_normal_ln_cdf(-sqrt_term * (t / mu + 1.0))).exp();
    (first + second).clamp(0.0, 1.0)
}

impl GeneralizedLinearMixedModel {
    /// Predictions for new data with configurable scale and unseen-level
    /// handling.
    ///
    /// The fixed-effect design is rebuilt with the training-time categorical
    /// encoding and the random-effects contribution uses the fitted GLMM
    /// conditional modes. On [`GlmmPredictionScale::Response`], the fitted
    /// inverse link is applied after the link-scale predictor is assembled.
    pub fn predict_new(
        &self,
        newdata: &DataFrame,
        scale: GlmmPredictionScale,
        new_re_levels: NewReLevels,
    ) -> Result<Vec<Option<f64>>> {
        self.predict_new_with_offset(newdata, None, scale, new_re_levels)
    }

    /// Predictions for new data with an explicit new-data offset vector.
    ///
    /// Use this variant for offset GLMMs. The offset is added on link scale
    /// before response-scale transformation. When `offset` is `None`, new rows
    /// are predicted with zero offset.
    pub fn predict_new_with_offset(
        &self,
        newdata: &DataFrame,
        offset: Option<&[f64]>,
        scale: GlmmPredictionScale,
        new_re_levels: NewReLevels,
    ) -> Result<Vec<Option<f64>>> {
        if self.lmm.optsum.feval <= 0 {
            return Err(MixedModelError::NotFitted);
        }
        if let Some(offset) = offset {
            validate_offset(offset, newdata.nrow())?;
        }

        let mut eta =
            self.lmm
                .linear_predict_new_with_state(newdata, &self.beta, &self.b, new_re_levels)?;
        if let Some(offset) = offset {
            for (prediction, offset_i) in eta.iter_mut().zip(offset.iter()) {
                if let Some(value) = prediction.as_mut() {
                    *value += *offset_i;
                }
            }
        }

        match scale {
            GlmmPredictionScale::Link => Ok(eta),
            GlmmPredictionScale::Response => Ok(eta
                .into_iter()
                .map(|prediction| prediction.map(|value| self.link.linkinv(value)))
                .collect()),
        }
    }

    /// Prediction-variance payload for GLMM new-data predictions.
    ///
    /// Rows are marked [`PredictionVarianceStatus::Available`] when the fit
    /// carries certified optimum evidence: joint-Laplace fits with an
    /// available fixed-effect inference artifact, or profiled fast-PIRLS fits
    /// whose post-fit profiled-optimum certificate passed its stationarity
    /// and curvature gates. Uncertified fits return the same working-Hessian
    /// delta-method numbers and mark rows
    /// [`PredictionVarianceStatus::Degraded`] with the certificate failure in
    /// the row reason. Response-scale rows additionally carry plug-in
    /// future-observation `prediction_variance` and predictive-quantile
    /// `prediction_lower`/`prediction_upper` columns for families that
    /// support them. New-level cases remain unavailable with row-level
    /// reasons.
    pub fn predict_new_variance(
        &self,
        newdata: &DataFrame,
        scale: GlmmPredictionScale,
        new_re_levels: NewReLevels,
    ) -> Result<PredictionVariancePayload> {
        self.predict_new_variance_with_level(newdata, scale, new_re_levels, 0.95)
    }

    /// Prediction-variance payload for GLMM new-data predictions at an
    /// explicit confidence level.
    pub fn predict_new_variance_with_level(
        &self,
        newdata: &DataFrame,
        scale: GlmmPredictionScale,
        new_re_levels: NewReLevels,
        level: f64,
    ) -> Result<PredictionVariancePayload> {
        let z = prediction_interval_cutoff(level)?;
        let link_predictions =
            self.predict_new(newdata, GlmmPredictionScale::Link, new_re_levels)?;
        let predictions = match scale {
            GlmmPredictionScale::Link => link_predictions.clone(),
            GlmmPredictionScale::Response => {
                self.predict_new(newdata, GlmmPredictionScale::Response, new_re_levels)?
            }
        };
        let mut payload =
            self.lmm
                .predict_new_variance_with_level(newdata, new_re_levels, level)?;
        let inner_lmm_scale = self.lmm.sigma();
        let glmm_covariance_scale = self
            .glmm_conditional_prediction_covariance_scale()
            .ok_or_else(|| {
                MixedModelError::InvalidArgument(
                    "GLMM prediction covariance scale is non-finite".to_string(),
                )
            })?;
        if !inner_lmm_scale.is_finite() || inner_lmm_scale <= 0.0 {
            return Err(MixedModelError::InvalidArgument(format!(
                "inner LMM prediction covariance scale must be positive and finite; got {inner_lmm_scale}"
            )));
        }
        // The delegated LMM payload is scaled by the inner LMM's sigma()
        // convention. GLMM predict(se.fit) parity follows lme4's GLMM scale:
        // 1 for unscaled families and ML sqrt(pwrss / N) for scaled families.
        let glmm_scale_multiplier = (glmm_covariance_scale / inner_lmm_scale).powi(2);
        let joint_laplace_conditional_variance =
            self.certified_joint_laplace_fixed_covariance().is_some();
        let pirls_certified_conditional_variance = !joint_laplace_conditional_variance
            && matches!(self.pirls_profiled_optimum_certificate, Some(Ok(_)));
        let certified_conditional_variance =
            joint_laplace_conditional_variance || pirls_certified_conditional_variance;
        let pirls_certificate_failure = match &self.pirls_profiled_optimum_certificate {
            Some(Err(reason)) if !joint_laplace_conditional_variance => Some(reason.clone()),
            _ => None,
        };

        let certified_geometry_label = if joint_laplace_conditional_variance {
            "final joint-laplace PIRLS/Laplace conditional-mode covariance"
        } else {
            "final fast-PIRLS profiled conditional-mode covariance at the certified profiled optimum"
        };
        let available_note = match scale {
            GlmmPredictionScale::Link => {
                format!(
                    "GLMM link-scale fitted-mean prediction variance uses the {certified_geometry_label} over fixed and random effects, rescaled to the GLMM covariance scale; theta uncertainty is not included"
                )
            }
            GlmmPredictionScale::Response => {
                format!(
                    "GLMM response-scale fitted-mean prediction variance uses delta-method link propagation from the {certified_geometry_label} over fixed and random effects, rescaled to the GLMM covariance scale; theta uncertainty is not included"
                )
            }
        };
        let fit_is_joint = self
            .lmm
            .compiler_artifact
            .glmm_fit_metadata
            .as_ref()
            .is_some_and(|metadata| metadata.estimation_method.starts_with("joint"));
        let uncertified_clause = match &pirls_certificate_failure {
            Some(failure) => format!(
                "the fast-PIRLS profiled optimum certificate was not issued ({failure}); refit with GlmmFitOptions::joint_laplace() for certified conditional prediction variance"
            ),
            None if fit_is_joint => {
                "the joint GLMM Hessian certificate did not pass quality gates, so conditional prediction variance is not certified for this fit"
                    .to_string()
            }
            None => "no certified optimum evidence is available for this fit; refit with GlmmFitOptions::joint_laplace() for certified conditional prediction variance"
                .to_string(),
        };
        let degraded_reason = match scale {
            GlmmPredictionScale::Link => {
                format!(
                    "GLMM link-scale prediction variance uses PIRLS/Laplace working-Hessian geometry; {uncertified_clause}"
                )
            }
            GlmmPredictionScale::Response => {
                format!(
                    "GLMM response-scale prediction variance uses delta-method link propagation from PIRLS/Laplace working-Hessian geometry; {uncertified_clause}"
                )
            }
        };
        let future_observation_support: std::result::Result<(), String> = match scale {
            GlmmPredictionScale::Link => Err(
                "future-observation prediction intervals are response-scale objects; request GlmmPredictionScale::Response for prediction_variance and prediction bounds"
                    .to_string(),
            ),
            GlmmPredictionScale::Response => self.glmm_future_observation_family_support(),
        };
        let mut future_observation_row_failures = std::collections::BTreeSet::new();

        for row in &mut payload.rows {
            row.prediction = predictions[row.row];
            row.fixed_variance = row.fixed_variance.and_then(|value| {
                clean_glmm_prediction_variance_component(value * glmm_scale_multiplier)
            });
            row.random_variance = row.random_variance.and_then(|value| {
                clean_glmm_prediction_variance_component(value * glmm_scale_multiplier)
            });
            row.fixed_random_covariance = row
                .fixed_random_covariance
                .map(|value| value * glmm_scale_multiplier)
                .filter(|value| value.is_finite());
            row.combined_variance = row.combined_variance.and_then(|value| {
                clean_glmm_prediction_variance_component(value * glmm_scale_multiplier)
            });
            row.se_fit = row.combined_variance.map(f64::sqrt);

            let link_scale_se = row.combined_variance.map(f64::sqrt);
            let derivative = match (scale, link_predictions[row.row]) {
                (GlmmPredictionScale::Link, Some(_)) => Some(1.0),
                (GlmmPredictionScale::Response, Some(eta)) => {
                    let value = self.link.mu_eta(eta);
                    (value.is_finite()).then_some(value)
                }
                (_, None) => None,
            };
            let variance_multiplier = derivative.map(|value| value * value);

            if let Some(multiplier) = variance_multiplier {
                row.fixed_variance = row
                    .fixed_variance
                    .map(|value| (value * multiplier).max(0.0));
                row.random_variance = row
                    .random_variance
                    .map(|value| (value * multiplier).max(0.0));
                row.fixed_random_covariance =
                    row.fixed_random_covariance.map(|value| value * multiplier);
                row.combined_variance = row
                    .combined_variance
                    .map(|value| (value * multiplier).max(0.0));
                row.se_fit = row.combined_variance.map(f64::sqrt);
            } else {
                row.random_variance = None;
                row.fixed_random_covariance = None;
                row.combined_variance = None;
                row.se_fit = None;
            }

            if row.status == PredictionVarianceStatus::Available {
                // Response-scale symmetric bounds can escape the family's
                // valid range near the boundary; compute the interval on the
                // link scale and map both ends through the inverse link
                // (ordered, since some links are decreasing).
                let bounds = match (scale, row.prediction, row.se_fit) {
                    (GlmmPredictionScale::Link, Some(prediction), Some(se_fit)) => {
                        Some((prediction - z * se_fit, prediction + z * se_fit))
                    }
                    (GlmmPredictionScale::Response, Some(_), Some(_)) => {
                        match (link_predictions[row.row], link_scale_se) {
                            (Some(eta), Some(link_se)) => {
                                let one = self.link.linkinv(eta - z * link_se);
                                let other = self.link.linkinv(eta + z * link_se);
                                (one.is_finite() && other.is_finite())
                                    .then(|| (one.min(other), one.max(other)))
                            }
                            _ => None,
                        }
                    }
                    _ => None,
                };
                if let Some((lower, upper)) = bounds {
                    row.confidence_lower = Some(lower);
                    row.confidence_upper = Some(upper);
                } else {
                    row.confidence_lower = None;
                    row.confidence_upper = None;
                }
                row.prediction_variance = None;
                row.prediction_lower = None;
                row.prediction_upper = None;
                if future_observation_support.is_ok() {
                    if let (Some(eta), Some(link_se)) = (link_predictions[row.row], link_scale_se) {
                        match self.glmm_future_observation(eta, link_se, level) {
                            Ok(future) => {
                                row.prediction_variance =
                                    clean_glmm_prediction_variance_component(future.variance);
                                row.prediction_lower = Some(future.lower);
                                row.prediction_upper = Some(future.upper);
                            }
                            Err(reason) => {
                                future_observation_row_failures.insert(reason);
                            }
                        }
                    }
                }
                if certified_conditional_variance {
                    row.reason = None;
                } else {
                    row.status = PredictionVarianceStatus::Degraded;
                    row.reason = Some(degraded_reason.clone());
                }
            } else {
                row.prediction_variance = None;
                row.confidence_lower = None;
                row.confidence_upper = None;
                row.prediction_lower = None;
                row.prediction_upper = None;
                let existing_reason = row.reason.clone().unwrap_or_else(|| {
                    "GLMM prediction variance is unavailable for this row".to_string()
                });
                row.reason = Some(format!("{existing_reason}; {degraded_reason}"));
            }
        }

        payload.method = if joint_laplace_conditional_variance {
            PredictionVarianceMethod::GlmmJointLaplaceConditionalDelta
        } else if pirls_certified_conditional_variance {
            PredictionVarianceMethod::GlmmPirlsProfiledCertifiedConditionalDelta
        } else {
            PredictionVarianceMethod::GlmmPirlsLaplaceWorkingDelta
        };
        payload.notes = if certified_conditional_variance {
            vec![
                available_note,
                format!(
                    "fixed, random, and fixed/random covariance components are transformed together from the {certified_geometry_label} geometry"
                ),
            ]
        } else {
            vec![
                degraded_reason,
                "fixed/random components are transformed from the inner PIRLS working LMM variance geometry"
                    .to_string(),
            ]
        };
        match &future_observation_support {
            Ok(()) => {
                payload.notes.push(format!(
                    "future-observation prediction_variance and prediction bounds are plug-in predictive summaries: the family conditional distribution (dispersion/size parameters treated as known at their estimates, future case weight 1) is mixed over link-scale fitted-mean uncertainty with {GLMM_PREDICTIVE_QUADRATURE_POINTS}-point Gauss-Hermite quadrature; bounds are predictive-distribution quantiles and prediction_variance is the law-of-total-variance moment; theta uncertainty is not included"
                ));
            }
            Err(reason) => {
                payload.notes.push(format!(
                    "future-observation prediction intervals are not reported: {reason}"
                ));
            }
        }
        for failure in future_observation_row_failures {
            payload.notes.push(format!(
                "future-observation prediction columns are unavailable for some rows: {failure}"
            ));
        }
        if matches!(scale, GlmmPredictionScale::Response) {
            payload.notes.push(
                "response-scale confidence bounds are link-scale Wald bounds mapped through the inverse link so they respect the family's valid range; se_fit remains delta-method on the response scale"
                    .to_string(),
            );
        }
        Ok(payload)
    }

    pub(super) fn glmm_conditional_prediction_covariance_scale(&self) -> Option<f64> {
        if !self.family.has_dispersion() {
            return Some(1.0);
        }
        let pwrss = self.lmm.pwrss();
        if !pwrss.is_finite() || pwrss < 0.0 {
            return None;
        }
        let denom = self.y.len().max(1) as f64;
        Some((pwrss / denom).max(f64::MIN_POSITIVE).sqrt())
    }

    /// Whether this model's family supports closed-form plug-in
    /// future-observation summaries for new rows.
    fn glmm_future_observation_family_support(&self) -> std::result::Result<(), String> {
        match self.family {
            Family::Binomial => Err(
                "future-observation prediction intervals for a grouped binomial response require the future observation's trial count, which newdata does not carry; model unit-trial rows with Family::Bernoulli for future-observation intervals"
                    .to_string(),
            ),
            _ => Ok(()),
        }
    }

    /// Plug-in predictive summary for one future observation.
    ///
    /// The family conditional distribution (dispersion / NB size treated as
    /// known at their estimates, future case weight 1) is mixed over the
    /// link-scale Gaussian fitted-mean uncertainty with normalized
    /// Gauss-Hermite quadrature. `variance` is the law-of-total-variance
    /// moment; `lower`/`upper` are predictive-distribution quantiles, so for
    /// discrete families they are integers and the interval is conservative
    /// (coverage at least `level`).
    fn glmm_future_observation(
        &self,
        eta: f64,
        link_se: f64,
        level: f64,
    ) -> std::result::Result<GlmmFutureObservation, String> {
        if !eta.is_finite() || !link_se.is_finite() || link_se < 0.0 {
            return Err(
                "link-scale prediction or its standard error is not finite and non-negative"
                    .to_string(),
            );
        }
        let lower_p = (1.0 - level) / 2.0;
        let upper_p = 1.0 - lower_p;
        let quadrature = gh_norm(GLMM_PREDICTIVE_QUADRATURE_POINTS);
        let mut nodes = Vec::with_capacity(quadrature.len());
        for (&z_node, &weight) in quadrature.z.iter().zip(quadrature.w.iter()) {
            let mu = self.link.linkinv(eta + link_se * z_node);
            if !mu.is_finite() {
                return Err(
                    "predictive quadrature produced a non-finite conditional mean".to_string(),
                );
            }
            nodes.push((mu, weight));
        }

        let mean: f64 = nodes.iter().map(|(mu, w)| w * mu).sum();
        let second_moment: f64 = nodes.iter().map(|(mu, w)| w * mu * mu).sum();
        let mean_variance = (second_moment - mean * mean).max(0.0);
        let dispersion = if self.family.has_dispersion() {
            self.dispersion(true)
        } else {
            1.0
        };
        if !dispersion.is_finite() || dispersion <= 0.0 {
            return Err(format!(
                "family dispersion estimate {dispersion} is not usable for predictive variance"
            ));
        }
        let family_variance: f64 = nodes
            .iter()
            .map(|(mu, w)| w * dispersion * self.variance(*mu))
            .sum();
        if !family_variance.is_finite() || family_variance < 0.0 {
            return Err(
                "family conditional variance is not finite over the predictive quadrature"
                    .to_string(),
            );
        }
        let variance = mean_variance + family_variance;
        let spread = variance.sqrt();

        let (lower, upper) = match self.family {
            Family::Bernoulli => {
                let prob_zero: f64 = nodes.iter().map(|(mu, w)| w * (1.0 - mu)).sum();
                let quantile = |p: f64| if prob_zero >= p { 0.0 } else { 1.0 };
                (quantile(lower_p), quantile(upper_p))
            }
            Family::Poisson => {
                let components = nodes
                    .iter()
                    .map(|(mu, w)| {
                        PoissonDist::new(mu.max(GLMM_PREDICTIVE_MEAN_FLOOR))
                            .map(|distribution| (distribution, *w))
                            .map_err(|err| format!("poisson predictive component: {err}"))
                    })
                    .collect::<std::result::Result<Vec<_>, String>>()?;
                let cdf = |t: u64| {
                    components
                        .iter()
                        .map(|(distribution, w)| w * distribution.cdf(t))
                        .sum::<f64>()
                };
                let quantile = |p: f64| {
                    discrete_mixture_quantile(&cdf, p, mean).ok_or_else(|| {
                        "poisson predictive quantile search did not converge".to_string()
                    })
                };
                (quantile(lower_p)?, quantile(upper_p)?)
            }
            Family::NegativeBinomial => {
                let size = self
                    .negative_binomial_theta
                    .filter(|theta| theta.is_finite() && *theta > 0.0)
                    .ok_or_else(|| {
                        "negative-binomial predictive quantiles require a positive finite size parameter"
                            .to_string()
                    })?;
                let components = nodes
                    .iter()
                    .map(|(mu, w)| {
                        let mu = mu.max(GLMM_PREDICTIVE_MEAN_FLOOR);
                        NegativeBinomialDist::new(size, size / (size + mu))
                            .map(|distribution| (distribution, *w))
                            .map_err(|err| format!("negative-binomial predictive component: {err}"))
                    })
                    .collect::<std::result::Result<Vec<_>, String>>()?;
                let cdf = |t: u64| {
                    components
                        .iter()
                        .map(|(distribution, w)| w * distribution.cdf(t))
                        .sum::<f64>()
                };
                let quantile = |p: f64| {
                    discrete_mixture_quantile(&cdf, p, mean).ok_or_else(|| {
                        "negative-binomial predictive quantile search did not converge".to_string()
                    })
                };
                (quantile(lower_p)?, quantile(upper_p)?)
            }
            Family::Gamma => {
                let shape = 1.0 / dispersion;
                let components = nodes
                    .iter()
                    .map(|(mu, w)| {
                        if !(*mu > 0.0) {
                            return Err(
                                "predictive quadrature produced conditional means outside the gamma family domain"
                                    .to_string(),
                            );
                        }
                        GammaDist::new(shape, shape / mu)
                            .map(|distribution| (distribution, *w))
                            .map_err(|err| format!("gamma predictive component: {err}"))
                    })
                    .collect::<std::result::Result<Vec<_>, String>>()?;
                let cdf = |t: f64| {
                    components
                        .iter()
                        .map(|(distribution, w)| w * distribution.cdf(t))
                        .sum::<f64>()
                };
                let quantile = |p: f64| {
                    continuous_mixture_quantile(&cdf, p, Some(0.0), mean, spread).ok_or_else(|| {
                        "gamma predictive quantile search did not converge".to_string()
                    })
                };
                (quantile(lower_p)?, quantile(upper_p)?)
            }
            Family::InverseGaussian => {
                let lambda = 1.0 / dispersion;
                for (mu, _) in &nodes {
                    if !(*mu > 0.0) {
                        return Err(
                            "predictive quadrature produced conditional means outside the inverse-Gaussian family domain"
                                .to_string(),
                        );
                    }
                }
                let cdf = |t: f64| {
                    nodes
                        .iter()
                        .map(|(mu, w)| w * inverse_gaussian_cdf(t, *mu, lambda))
                        .sum::<f64>()
                };
                let quantile = |p: f64| {
                    continuous_mixture_quantile(&cdf, p, Some(0.0), mean, spread).ok_or_else(|| {
                        "inverse-Gaussian predictive quantile search did not converge".to_string()
                    })
                };
                (quantile(lower_p)?, quantile(upper_p)?)
            }
            Family::Normal => {
                let sigma = dispersion.sqrt();
                let components = nodes
                    .iter()
                    .map(|(mu, w)| {
                        Normal::new(*mu, sigma)
                            .map(|distribution| (distribution, *w))
                            .map_err(|err| format!("gaussian predictive component: {err}"))
                    })
                    .collect::<std::result::Result<Vec<_>, String>>()?;
                let cdf = |t: f64| {
                    components
                        .iter()
                        .map(|(distribution, w)| w * distribution.cdf(t))
                        .sum::<f64>()
                };
                let quantile = |p: f64| {
                    continuous_mixture_quantile(&cdf, p, None, mean, spread).ok_or_else(|| {
                        "gaussian predictive quantile search did not converge".to_string()
                    })
                };
                (quantile(lower_p)?, quantile(upper_p)?)
            }
            other => {
                return Err(format!(
                    "future-observation predictive quantiles are not implemented for {other:?}"
                ));
            }
        };
        if !(lower.is_finite() && upper.is_finite() && lower <= upper) {
            return Err("predictive quantiles are not finite and ordered".to_string());
        }

        Ok(GlmmFutureObservation {
            variance,
            lower,
            upper,
        })
    }

    /// Simulate a new response vector under a fresh draw of the random
    /// effects (the parametric-bootstrap data step).
    ///
    /// Draws `b_i = Λ_i u_i` with `u_i ~ N(0, I)`, forms the linear
    /// predictor `η = offset + Xβ̂ + Σ Z_i b_i`, maps it to `μ = g⁻¹(η)`,
    /// and samples the response from the conditional family with mean `μ`:
    /// Bernoulli → `{0, 1}`, Poisson → counts, Binomial → success
    /// proportion over the per-observation trial size (prior weights,
    /// default `1`), and Gamma → positive draws with `shape = 1 / phi`
    /// and `scale = mu * phi`, where `phi = dispersion(true)`.
    /// InverseGaussian and Normal-as-GLM are refused because they do not
    /// yet have certified family-specific response simulators.
    pub fn simulate_response<R: rand::Rng>(&self, rng: &mut R) -> Result<Vec<f64>> {
        use rand_distr::{Binomial, Distribution, Gamma as GammaDistribution, Normal, Poisson};

        match self.family {
            Family::Bernoulli
            | Family::Binomial
            | Family::Poisson
            | Family::NegativeBinomial
            | Family::Gamma => {}
            Family::InverseGaussian | Family::Normal => {
                return Err(MixedModelError::Unsupported(format!(
                    "{:?} GLMM parametric bootstrap is not implemented; no certified \
                     family-specific response simulator is available",
                    self.family
                )));
            }
        }
        let gamma_phi = if matches!(self.family, Family::Gamma) {
            let phi = self.dispersion(true);
            if !phi.is_finite() || phi <= 0.0 {
                return Err(MixedModelError::InvalidArgument(format!(
                    "Gamma GLMM bootstrap requires positive finite phi = dispersion(true); got {phi}"
                )));
            }
            Some(phi)
        } else {
            None
        };
        let negative_binomial_theta = if matches!(self.family, Family::NegativeBinomial) {
            Some(self.require_negative_binomial_theta()?)
        } else {
            None
        };

        let n = self.eta.len();
        let x = self.lmm.feterm.full_rank_x();
        let mut eta = &self.offset + x * &self.beta;

        let normal01 = Normal::new(0.0, 1.0).unwrap();
        for rt in &self.lmm.reterms {
            let n_levels = rt.n_levels();
            let u = DMatrix::from_fn(rt.vsize, n_levels, |_, _| normal01.sample(rng));
            let b = &rt.lambda * &u;
            let bvec = DVector::from_column_slice(b.as_slice());
            for (obs, &ref_idx) in rt.refs.iter().enumerate() {
                let r = ref_idx as usize;
                for s in 0..rt.vsize {
                    eta[obs] += rt.z[(s, obs)] * bvec[r * rt.vsize + s];
                }
            }
        }

        let mut y = vec![0.0f64; n];
        for (i, yi) in y.iter_mut().enumerate() {
            let mu = self.link.linkinv(eta[i]);
            if !mu.is_finite() {
                return Err(MixedModelError::InvalidArgument(format!(
                    "simulated conditional mean is non-finite at observation {i}"
                )));
            }
            match self.family {
                Family::Bernoulli => {
                    let p = mu.clamp(0.0, 1.0);
                    *yi = f64::from(rng.gen::<f64>() < p);
                }
                Family::Binomial => {
                    let p = mu.clamp(0.0, 1.0);
                    let trials = if self.wt.is_empty() { 1.0 } else { self.wt[i] };
                    let n_trials = trials.round().max(0.0) as u64;
                    if n_trials == 0 {
                        *yi = 0.0;
                    } else {
                        let count = Binomial::new(n_trials, p)
                            .map_err(|e| {
                                MixedModelError::InvalidArgument(format!(
                                    "binomial draw failed at observation {i}: {e}"
                                ))
                            })?
                            .sample(rng) as f64;
                        *yi = count / trials;
                    }
                }
                Family::Poisson => {
                    let lambda = mu.max(f64::MIN_POSITIVE);
                    *yi = Poisson::new(lambda)
                        .map_err(|e| {
                            MixedModelError::InvalidArgument(format!(
                                "poisson draw failed at observation {i}: {e}"
                            ))
                        })?
                        .sample(rng);
                }
                Family::NegativeBinomial => {
                    let theta = negative_binomial_theta.expect("NB theta computed above");
                    let mean = mu.max(f64::MIN_POSITIVE);
                    let lambda = GammaDistribution::new(theta, mean / theta)
                        .map_err(|e| {
                            MixedModelError::InvalidArgument(format!(
                                "negative-binomial gamma-mixture draw failed at observation {i}: {e}"
                            ))
                        })?
                        .sample(rng);
                    *yi = Poisson::new(lambda.max(f64::MIN_POSITIVE))
                        .map_err(|e| {
                            MixedModelError::InvalidArgument(format!(
                                "negative-binomial poisson draw failed at observation {i}: {e}"
                            ))
                        })?
                        .sample(rng);
                }
                Family::Gamma => {
                    let phi = gamma_phi.expect("Gamma phi computed above");
                    let mean = if mu > 0.0 {
                        mu
                    } else if mu == 0.0 {
                        f64::MIN_POSITIVE
                    } else {
                        return Err(MixedModelError::InvalidArgument(format!(
                            "Gamma draw requires positive conditional mean at observation {i}; got {mu}"
                        )));
                    };
                    let shape = 1.0 / phi;
                    let scale = mean * phi;
                    *yi = GammaDistribution::new(shape, scale)
                        .map_err(|e| {
                            MixedModelError::InvalidArgument(format!(
                                "Gamma draw failed at observation {i}: {e}"
                            ))
                        })?
                        .sample(rng);
                }
                Family::InverseGaussian | Family::Normal => {
                    unreachable!("dispersion families refused above")
                }
            }
        }
        Ok(y)
    }
}
