//! PIRLS working-response and exponential-family numerics helpers.
//!
//! Moved verbatim from the former single-file `generalized.rs` during the
//! module split (bd-01KWG1BKEWB91RXAXC0350SFMK). No logic changes.

use super::*;

#[derive(Debug, Clone, Copy)]
pub(crate) struct BinaryColumnSplit {
    pub(crate) low: f64,
    pub(crate) high: f64,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct OutcomeCounts {
    n: usize,
    successes: usize,
    failures: usize,
}

pub(crate) fn is_binary_response(value: f64) -> bool {
    (value - 0.0).abs() < 1e-12 || (value - 1.0).abs() < 1e-12
}

pub(crate) fn is_nonnegative_integer_response(value: f64) -> bool {
    value >= 0.0 && (value - value.round()).abs() < 1e-12
}

pub(crate) fn is_intercept_column(name: &str) -> bool {
    matches!(name, "1" | "(Intercept)" | "Intercept" | "intercept")
}

pub(crate) fn random_effect_term_label(reterm: &ReMat) -> String {
    let columns = reterm
        .cnames
        .iter()
        .map(|name| {
            if is_intercept_column(name) {
                "1"
            } else {
                name.as_str()
            }
        })
        .collect::<Vec<_>>()
        .join(" + ");
    format!("({columns} | {})", reterm.grouping_name)
}

pub(crate) fn binary_column_split(values: impl Iterator<Item = f64>) -> Option<BinaryColumnSplit> {
    let mut unique = Vec::new();
    for value in values {
        if !value.is_finite() {
            return None;
        }
        if unique
            .iter()
            .all(|seen: &f64| (value - *seen).abs() > 1e-12)
        {
            unique.push(value);
            if unique.len() > 2 {
                return None;
            }
        }
    }
    if unique.len() != 2 {
        return None;
    }
    unique.sort_by(|a, b| a.total_cmp(b));
    Some(BinaryColumnSplit {
        low: unique[0],
        high: unique[1],
    })
}

pub(crate) fn outcome_counts_for_value(
    values: impl Iterator<Item = f64>,
    y: impl Iterator<Item = f64>,
    target: f64,
) -> OutcomeCounts {
    let mut counts = OutcomeCounts {
        n: 0,
        successes: 0,
        failures: 0,
    };
    for (value, response) in values.zip(y) {
        if (value - target).abs() <= 1e-12 {
            counts.n += 1;
            if response > 0.5 {
                counts.successes += 1;
            } else {
                counts.failures += 1;
            }
        }
    }
    counts
}

pub(crate) fn separation_diagnostic_for_side(
    column_name: &str,
    value: f64,
    side: OutcomeCounts,
    complement: OutcomeCounts,
) -> Option<Diagnostic> {
    if side.n == 0
        || complement.n == 0
        || !side_is_pure(side)
        || !complement_has_opposite(side, complement)
    {
        return None;
    }

    let outcome = if side.successes == side.n { 1 } else { 0 };
    let kind = if side_is_pure(complement) {
        "complete_fixed_effect"
    } else {
        "quasi_complete_fixed_effect"
    };
    let rows = if side.n == 1 { "row" } else { "rows" };
    let value_label = format_column_value(value);
    let mut diagnostic = Diagnostic::new(
        DiagnosticCode::BinomialSeparation,
        DiagnosticSeverity::Warning,
        DiagnosticStage::Certification,
        format!(
            "Possible separation in binomial model: `{column_name} = {value_label}` occurs in {} {rows}, and all such rows have y = {outcome}. The coefficient for `{column_name}` may be unbounded; standard errors, Wald tests, and p-values for this term are unreliable.",
            side.n
        ),
    )
    .with_affected_terms(vec![column_name.to_string()])
    .with_suggested_actions(vec![
        "inspect the corresponding rows or levels for sparse outcome support".to_string(),
        "consider removing or combining rare predictors, or use penalized/Bayesian logistic mixed modeling".to_string(),
        "report inference for this term as unreliable if the model is retained".to_string(),
    ]);
    diagnostic
        .payload
        .insert("term".to_string(), serde_json::json!(column_name));
    diagnostic
        .payload
        .insert("value".to_string(), serde_json::json!(value));
    diagnostic
        .payload
        .insert("n_at_value".to_string(), serde_json::json!(side.n));
    diagnostic.payload.insert(
        "successes_at_value".to_string(),
        serde_json::json!(side.successes),
    );
    diagnostic.payload.insert(
        "failures_at_value".to_string(),
        serde_json::json!(side.failures),
    );
    diagnostic.payload.insert(
        "complement_successes".to_string(),
        serde_json::json!(complement.successes),
    );
    diagnostic.payload.insert(
        "complement_failures".to_string(),
        serde_json::json!(complement.failures),
    );
    diagnostic
        .payload
        .insert("separation_kind".to_string(), serde_json::json!(kind));
    Some(diagnostic)
}

pub(crate) fn side_is_pure(counts: OutcomeCounts) -> bool {
    counts.n > 0 && (counts.successes == counts.n || counts.failures == counts.n)
}

pub(crate) fn complement_has_opposite(side: OutcomeCounts, complement: OutcomeCounts) -> bool {
    if side.successes == side.n {
        complement.failures > 0
    } else {
        complement.successes > 0
    }
}

pub(crate) fn format_column_value(value: f64) -> String {
    if (value.round() - value).abs() < 1e-12 {
        format!("{value:.0}")
    } else {
        format!("{value:.6}")
    }
}

#[cfg(test)]
pub(crate) fn pirls_working_observation(
    family: Family,
    link: LinkFunction,
    y: f64,
    eta: f64,
    mu: f64,
    case_weight: f64,
) -> (f64, f64) {
    pirls_working_observation_with_family_parameters(family, link, None, y, eta, mu, case_weight)
}

pub(crate) fn pirls_working_observation_with_family_parameters(
    family: Family,
    link: LinkFunction,
    negative_binomial_theta: Option<f64>,
    y: f64,
    eta: f64,
    mu: f64,
    case_weight: f64,
) -> (f64, f64) {
    let (working_mu, eta_for_derivative) = bounded_pirls_mean_and_eta(family, link, mu, eta);
    let dmu_deta = link.mu_eta(eta_for_derivative);
    let var_mu = glmm_variance(family, working_mu, negative_binomial_theta);
    let weight = if dmu_deta.is_finite() && var_mu.is_finite() && var_mu > 0.0 {
        case_weight * dmu_deta * dmu_deta / var_mu
    } else {
        0.0
    };
    let resid = if !dmu_deta.is_finite() || dmu_deta.abs() < 1e-15 {
        0.0
    } else {
        (y - working_mu) / dmu_deta
    };
    (weight.max(0.0).sqrt(), eta + resid)
}

#[cfg(test)]
pub(crate) fn pirls_working_observation_with_offset(
    family: Family,
    link: LinkFunction,
    y: f64,
    eta: f64,
    mu: f64,
    case_weight: f64,
    offset: f64,
) -> (f64, f64) {
    let (sqrt_weight, working_response) =
        pirls_working_observation(family, link, y, eta, mu, case_weight);
    (sqrt_weight, working_response - offset)
}

pub(crate) fn pirls_working_observation_with_offset_and_family_parameters(
    family: Family,
    link: LinkFunction,
    negative_binomial_theta: Option<f64>,
    y: f64,
    eta: f64,
    mu: f64,
    case_weight: f64,
    offset: f64,
) -> (f64, f64) {
    let (sqrt_weight, working_response) = pirls_working_observation_with_family_parameters(
        family,
        link,
        negative_binomial_theta,
        y,
        eta,
        mu,
        case_weight,
    );
    (sqrt_weight, working_response - offset)
}

pub(crate) fn bounded_pirls_mean_and_eta(
    family: Family,
    link: LinkFunction,
    mu: f64,
    eta: f64,
) -> (f64, f64) {
    const BOUNDED_MEAN_EPS: f64 = 1e-15;
    const LOG_LINK_ETA_BOUND: f64 = 30.0;
    if matches!(family, Family::Bernoulli | Family::Binomial) {
        let bounded_mu = mu.clamp(BOUNDED_MEAN_EPS, 1.0 - BOUNDED_MEAN_EPS);
        (bounded_mu, link.link(bounded_mu))
    } else if matches!(family, Family::Poisson | Family::NegativeBinomial) {
        match link {
            LinkFunction::Log => {
                let bounded_eta = eta.clamp(-LOG_LINK_ETA_BOUND, LOG_LINK_ETA_BOUND);
                (bounded_eta.exp(), bounded_eta)
            }
            LinkFunction::Sqrt => {
                let bounded_mu = mu.max(BOUNDED_MEAN_EPS);
                let min_eta = bounded_mu.sqrt();
                let bounded_eta = if eta.abs() < min_eta {
                    if eta.is_sign_negative() {
                        -min_eta
                    } else {
                        min_eta
                    }
                } else {
                    eta
                };
                (bounded_eta * bounded_eta, bounded_eta)
            }
            _ => (mu, eta),
        }
    } else {
        (mu, eta)
    }
}

pub(crate) fn glmm_variance(family: Family, mu: f64, negative_binomial_theta: Option<f64>) -> f64 {
    match family {
        Family::NegativeBinomial => {
            let theta = negative_binomial_theta
                .unwrap_or(1.0)
                .max(f64::MIN_POSITIVE);
            mu + mu * mu / theta
        }
        _ => family.variance(mu),
    }
}

pub(crate) const NEGATIVE_BINOMIAL_THETA_MIN: f64 = 1.0e-8;

pub(crate) const NEGATIVE_BINOMIAL_THETA_MAX: f64 = 1.0e8;

pub(crate) const NEGATIVE_BINOMIAL_THETA_MAX_ITERS: usize = 8;

pub(crate) const NEGATIVE_BINOMIAL_THETA_TOL: f64 = 1.0e-5;

pub(crate) const NEGATIVE_BINOMIAL_THETA_FINAL_REFIT_TOL: f64 = 1.0e-8;

pub(crate) fn clamp_negative_binomial_theta(theta: f64) -> f64 {
    theta.clamp(NEGATIVE_BINOMIAL_THETA_MIN, NEGATIVE_BINOMIAL_THETA_MAX)
}

pub(crate) fn negative_binomial_deviance_component(y: f64, mu: f64, theta: f64) -> f64 {
    let mu = mu.max(f64::MIN_POSITIVE);
    let theta = theta.max(f64::MIN_POSITIVE);
    let first = if y == 0.0 { 0.0 } else { y * (y / mu).ln() };
    let second = (y + theta) * ((y + theta) / (mu + theta)).ln();
    2.0 * (first - second)
}

pub(crate) fn negative_binomial_loglik_observation(y: f64, mu: f64, theta: f64) -> f64 {
    let mu = mu.max(f64::MIN_POSITIVE);
    let theta = theta.max(f64::MIN_POSITIVE);
    ln_gamma(y + theta) - ln_gamma(theta) - ln_gamma(y + 1.0)
        + theta * (theta / (theta + mu)).ln()
        + if y == 0.0 {
            0.0
        } else {
            y * (mu / (theta + mu)).ln()
        }
}

pub(crate) fn negative_binomial_theta_moment_start(y: &[f64], weights: Option<&[f64]>) -> f64 {
    let (sum_w, mean_num) =
        y.iter()
            .enumerate()
            .fold((0.0, 0.0), |(sum_w, mean_num), (idx, &value)| {
                let weight = weights
                    .and_then(|weights| weights.get(idx).copied())
                    .unwrap_or(1.0);
                (sum_w + weight, mean_num + weight * value)
            });
    if sum_w <= 0.0 {
        return 1.0;
    }
    let mean = mean_num / sum_w;
    let variance = y.iter().enumerate().fold(0.0, |acc, (idx, &value)| {
        let weight = weights
            .and_then(|weights| weights.get(idx).copied())
            .unwrap_or(1.0);
        acc + weight * (value - mean).powi(2)
    }) / sum_w.max(1.0);

    if variance > mean && mean > 0.0 {
        clamp_negative_binomial_theta(mean * mean / (variance - mean))
    } else {
        NEGATIVE_BINOMIAL_THETA_MAX.sqrt()
    }
}

pub(crate) fn estimate_negative_binomial_theta_conditional(
    y: &[f64],
    mu: &[f64],
    weights: Option<&[f64]>,
) -> f64 {
    if y.len() != mu.len() || y.is_empty() {
        return 1.0;
    }
    let log_min = NEGATIVE_BINOMIAL_THETA_MIN.ln();
    let log_max = NEGATIVE_BINOMIAL_THETA_MAX.ln();
    let weighted_loglik = |log_theta: f64| -> f64 {
        let theta = log_theta.exp();
        let mut total = 0.0;
        for (idx, (&y_i, &mu_i)) in y.iter().zip(mu.iter()).enumerate() {
            let weight = weights
                .and_then(|weights| weights.get(idx).copied())
                .unwrap_or(1.0);
            if !weight.is_finite() || weight <= 0.0 {
                continue;
            }
            let contribution = negative_binomial_loglik_observation(y_i, mu_i, theta);
            if !contribution.is_finite() {
                return f64::NEG_INFINITY;
            }
            total += weight * contribution;
        }
        total
    };

    let inv_phi = (5.0_f64.sqrt() - 1.0) / 2.0;
    let mut a = log_min;
    let mut b = log_max;
    let mut c = b - inv_phi * (b - a);
    let mut d = a + inv_phi * (b - a);
    let mut fc = weighted_loglik(c);
    let mut fd = weighted_loglik(d);

    for _ in 0..96 {
        if (b - a).abs() <= 1.0e-8 {
            break;
        }
        if fc < fd {
            a = c;
            c = d;
            fc = fd;
            d = a + inv_phi * (b - a);
            fd = weighted_loglik(d);
        } else {
            b = d;
            d = c;
            fd = fc;
            c = b - inv_phi * (b - a);
            fc = weighted_loglik(c);
        }
    }

    let mut candidates = vec![a, b, c, d];
    let moment = negative_binomial_theta_moment_start(y, weights);
    candidates.push(moment.ln());
    candidates
        .into_iter()
        .filter(|value| value.is_finite())
        .max_by(|left, right| weighted_loglik(*left).total_cmp(&weighted_loglik(*right)))
        .map(|log_theta| clamp_negative_binomial_theta(log_theta.exp()))
        .unwrap_or(moment)
}

pub(crate) fn relative_theta_change(old: f64, new: f64) -> f64 {
    if !old.is_finite() || !new.is_finite() {
        return f64::INFINITY;
    }
    (new - old).abs() / old.abs().max(1.0)
}

pub(crate) fn pirls_converged(obj: f64, accepted_obj: f64, tol: f64) -> bool {
    (obj - accepted_obj).abs() < tol
}

pub(crate) fn validate_case_weights(weights: &[f64], n_obs: usize) -> Result<()> {
    if weights.len() != n_obs {
        return Err(MixedModelError::InvalidArgument(format!(
            "case weights length ({}) does not match number of observations ({n_obs})",
            weights.len()
        )));
    }
    for (i, &w) in weights.iter().enumerate() {
        if !w.is_finite() || w <= 0.0 {
            return Err(MixedModelError::InvalidArgument(format!(
                "case weight at index {i} must be finite and positive (got {w})"
            )));
        }
    }
    Ok(())
}

pub(crate) fn validate_offset(offset: &[f64], n_obs: usize) -> Result<()> {
    if offset.len() != n_obs {
        return Err(MixedModelError::InvalidArgument(format!(
            "offset length ({}) does not match number of observations ({n_obs})",
            offset.len()
        )));
    }
    for (idx, &value) in offset.iter().enumerate() {
        if !value.is_finite() {
            return Err(MixedModelError::InvalidArgument(format!(
                "offset at index {idx} must be finite (got {value})"
            )));
        }
    }
    Ok(())
}

pub(crate) fn validate_supported_glmm_family_link(
    family: Family,
    link: LinkFunction,
) -> Result<()> {
    let supported = match family {
        Family::Bernoulli | Family::Binomial => {
            matches!(
                link,
                LinkFunction::Logit | LinkFunction::Probit | LinkFunction::Cloglog
            )
        }
        Family::Poisson => matches!(link, LinkFunction::Log | LinkFunction::Sqrt),
        Family::NegativeBinomial => matches!(link, LinkFunction::Log),
        // Dispersion-family GLMMs predate this explicit binary/Poisson support
        // matrix; keep their existing sensible links while preserving the
        // Normal+Identity LMM redirect above.
        Family::Gamma | Family::InverseGaussian => {
            matches!(link, LinkFunction::Log | LinkFunction::Inverse)
        }
        Family::Normal => matches!(
            link,
            LinkFunction::Log | LinkFunction::Inverse | LinkFunction::Sqrt
        ),
    };
    if supported {
        Ok(())
    } else {
        Err(MixedModelError::UnsupportedFamilyLink {
            family: family_label(family).to_string(),
            link: link_label(link).to_string(),
        })
    }
}

pub(crate) fn validate_negative_binomial_theta_request(
    family: Family,
    theta: Option<f64>,
    estimate_theta: bool,
) -> Result<()> {
    match (family, theta, estimate_theta) {
        (Family::NegativeBinomial, Some(theta), _) if theta.is_finite() && theta > 0.0 => Ok(()),
        (Family::NegativeBinomial, Some(theta), true) => Err(MixedModelError::InvalidArgument(
            format!("negative-binomial theta start must be positive and finite (got {theta})"),
        )),
        (Family::NegativeBinomial, Some(theta), false) => Err(MixedModelError::InvalidArgument(
            format!("negative-binomial fixed theta must be positive and finite (got {theta})"),
        )),
        (Family::NegativeBinomial, None, true) => Ok(()),
        (Family::NegativeBinomial, None, false) => Err(MixedModelError::InvalidArgument(
            "negative-binomial GLMM requires a positive fixed theta, or explicit theta \
             estimation via GeneralizedLinearMixedModel::new_negative_binomial_estimated(...) \
             / GeneralizedLinearMixedModelBuilder::estimate_negative_binomial_theta(...); use \
             GeneralizedLinearMixedModel::new_negative_binomial(...) or \
             GeneralizedLinearMixedModelBuilder::negative_binomial_theta(...) for fixed theta"
                .to_string(),
        )),
        (_, Some(_), _) | (_, None, true) => Err(MixedModelError::InvalidArgument(
            "negative-binomial theta options can only be supplied with Family::NegativeBinomial"
                .to_string(),
        )),
        (_, None, false) => Ok(()),
    }
}

pub(crate) fn initialize_negative_binomial_theta(
    family: Family,
    theta: Option<f64>,
    estimate_theta: bool,
    response: Option<&[f64]>,
) -> Result<Option<f64>> {
    if family != Family::NegativeBinomial {
        return Ok(None);
    }
    if let Some(theta) = theta {
        return Ok(Some(clamp_negative_binomial_theta(theta)));
    }
    if estimate_theta {
        let y = response.ok_or_else(|| {
            MixedModelError::InvalidArgument(
                "negative-binomial theta estimation requires a numeric response".to_string(),
            )
        })?;
        return Ok(Some(negative_binomial_theta_moment_start(y, None)));
    }
    Err(MixedModelError::InvalidArgument(
        "negative-binomial GLMM requires a positive fixed theta".to_string(),
    ))
}

pub(crate) fn validate_glmm_response_domain(
    family: Family,
    link: LinkFunction,
    y: &[f64],
) -> Result<()> {
    for (idx, &value) in y.iter().enumerate() {
        if !value.is_finite() {
            return Err(MixedModelError::InvalidArgument(format!(
                "response at index {idx} must be finite for GLMM construction (got {value})"
            )));
        }
        if family == Family::Bernoulli && !is_binary_response(value) {
            return Err(MixedModelError::InvalidArgument(format!(
                "bernoulli GLMM response must be exactly 0 or 1; index {idx} has {value}"
            )));
        }
        if family == Family::Binomial
            && !(0.0..=1.0).contains(&value)
            && !is_nonnegative_integer_response(value)
        {
            return Err(MixedModelError::InvalidArgument(format!(
                "binomial GLMM response must be a proportion in [0, 1] or a non-negative integer count; index {idx} has {value}"
            )));
        }
        if family == Family::Poisson && value < 0.0 {
            return Err(MixedModelError::InvalidArgument(format!(
                "poisson GLMM response must be non-negative; index {idx} has {value}"
            )));
        }
        if family == Family::NegativeBinomial && !is_nonnegative_integer_response(value) {
            return Err(MixedModelError::InvalidArgument(format!(
                "negative-binomial GLMM response must be a non-negative integer count; index {idx} has {value}"
            )));
        }
        if matches!(family, Family::Gamma | Family::InverseGaussian) && value <= 0.0 {
            return Err(MixedModelError::InvalidArgument(format!(
                "{} GLMM response must be strictly positive; index {idx} has {value}",
                family_label(family)
            )));
        }
        if family == Family::Normal && link == LinkFunction::Sqrt && value < 0.0 {
            return Err(MixedModelError::InvalidArgument(format!(
                "gaussian GLMM with sqrt link requires non-negative responses; index {idx} has {value}"
            )));
        }
    }
    Ok(())
}

pub(crate) fn initial_response_mean(
    family: Family,
    y: &DVector<f64>,
    weights: &[f64],
) -> Option<f64> {
    if y.is_empty() {
        return None;
    }
    let mut weighted_sum = 0.0;
    let mut weight_sum = 0.0;
    for (idx, value) in y.iter().enumerate() {
        let weight = weights.get(idx).copied().unwrap_or(1.0);
        weighted_sum += weight * value;
        weight_sum += weight;
    }
    if weight_sum <= 0.0 {
        return None;
    }
    let mean = weighted_sum / weight_sum;
    Some(match family {
        Family::Bernoulli | Family::Binomial => mean.clamp(1e-6, 1.0 - 1e-6),
        Family::Poisson | Family::NegativeBinomial | Family::Gamma | Family::InverseGaussian => {
            mean.max(1e-6)
        }
        Family::Normal => mean.max(0.0),
    })
}

pub(crate) fn initial_mean_for_link(family: Family, mean: f64) -> f64 {
    match family {
        Family::Bernoulli | Family::Binomial => mean.clamp(1e-6, 1.0 - 1e-6),
        Family::Poisson | Family::NegativeBinomial | Family::Gamma | Family::InverseGaussian => {
            mean.max(1e-6)
        }
        Family::Normal => mean.max(0.0),
    }
}

pub(crate) fn family_label(family: Family) -> &'static str {
    match family {
        Family::Normal => "gaussian",
        Family::Bernoulli => "bernoulli",
        Family::Binomial => "binomial",
        Family::Poisson => "poisson",
        Family::NegativeBinomial => "negative_binomial",
        Family::Gamma => "gamma",
        Family::InverseGaussian => "inverse_gaussian",
    }
}

pub(crate) fn link_label(link: LinkFunction) -> &'static str {
    match link {
        LinkFunction::Identity => "identity",
        LinkFunction::Log => "log",
        LinkFunction::Logit => "logit",
        LinkFunction::Probit => "probit",
        LinkFunction::Cloglog => "cloglog",
        LinkFunction::Inverse => "inverse",
        LinkFunction::Sqrt => "sqrt",
    }
}

impl GeneralizedLinearMixedModel {
    /// PIRLS: Penalized Iteratively Reweighted Least Squares.
    ///
    /// Updates β and u until convergence. The working response and weights
    /// are derived from the current μ = g⁻¹(Xβ + Zb).
    ///
    /// * `vary_beta` – if false, β is held fixed and only u is updated
    ///
    /// Returns `Ok(true)` if PIRLS reached its convergence tolerance within
    /// the iteration budget, `Ok(false)` if it exhausted the budget without
    /// converging (the conditional modes are the best seen but unverified).
    /// The non-converged case is deliberately *not* an `Err`: callers decide
    /// how to surface it (the final fit records a diagnostic; interior
    /// optimizer probes tolerate it). `Err` is reserved for hard linear-
    /// algebra/state failures.
    pub fn pirls(&mut self, vary_beta: bool, verbose: bool) -> Result<bool> {
        self.pirls_with_options(vary_beta, verbose, GLMM_PIRLS_MAX_ITER, true)
    }

    fn pirls_with_options(
        &mut self,
        vary_beta: bool,
        verbose: bool,
        max_iter: usize,
        reset_modes: bool,
    ) -> Result<bool> {
        // Mirrors MixedModels.jl/src/generalizedlinearmixedmodel.jl pirls!
        // (lines 614-669): step-halving toward the previous accepted iterate
        // whenever a fresh IRLS step would worsen the Laplace objective. Keeps
        // the outer optimizer's view of obj(θ) consistent across probes —
        // without this, BOBYQA on multi-RE GLMM surfaces (e.g. grouseticks
        // Poisson) sees noisy values and reports `RoundoffLimited`.
        let tol = 1.0e-5;
        let max_halvings = 10;

        let n = self.y.len();

        // Reset the conditional modes when callers need deterministic probe
        // values instead of path-dependent warm starts.
        if reset_modes {
            for u in self.u.iter_mut() {
                u.fill(0.0);
            }
        }
        for (i, rt) in self.lmm.reterms.iter().enumerate() {
            self.b[i] = &rt.lambda * &self.u[i];
        }
        self.update_eta();

        // Save the initial accepted state for halving. The 1.0001 slack is
        // only an acceptance bound for the first step-halving loop; convergence
        // is compared with the uninflated accepted objective.
        let mut u_prev: Vec<DMatrix<f64>> = self.u.clone();
        let mut beta_prev = self.beta.clone();
        let mut obj0 = self.laplace_objective();
        let mut halving_bound = obj0 * 1.0001;

        let mut sqrtwts = vec![0.0f64; n];
        let mut working_y = vec![0.0f64; n];

        // Whether PIRLS reached its convergence tolerance within `max_iter`.
        // Returned to the caller so a non-converged conditional-mode solve is
        // *observable* rather than silently accepted (audit 03·H1). We do not
        // hard-error inside the loop: the outer optimizer legitimately probes
        // near the variance-component boundary where an interior step may
        // exhaust halving, and turning that into an error perturbs the
        // soft-barrier search away from valid boundary fits.
        let mut converged = false;
        let progress_callback = self.lmm.progress_callback.clone();
        let total_iterations = max_iter.max(1);
        let mut last_progress = 0usize;

        for iter in 0..total_iterations {
            if let Some(callback) = &progress_callback {
                callback.report_if_due(
                    FitProgressPhase::Pirls,
                    iter + 1,
                    Some(total_iterations),
                    &mut last_progress,
                )?;
            }
            // --- Compute IRLS weights and working response ---
            for obs in 0..n {
                let mu_obs = self.mu[obs];
                let eta_obs = self.eta[obs];
                let y_obs = self.y[obs];

                let case_w = if self.wt.is_empty() {
                    1.0
                } else {
                    self.wt[obs]
                };
                (sqrtwts[obs], working_y[obs]) =
                    pirls_working_observation_with_offset_and_family_parameters(
                        self.family,
                        self.link,
                        self.negative_binomial_theta,
                        y_obs,
                        eta_obs,
                        mu_obs,
                        case_w,
                        self.offset[obs],
                    );
            }

            // --- Update the LMM with new IRLS weights ---
            self.lmm.update_irls_weights(&sqrtwts, &working_y)?;
            self.lmm.update_l()?;

            // --- Propose new β / u from the LMM solution ---
            let new_u = if vary_beta {
                self.beta = self.lmm.beta();
                self.lmm.ranef_u()
            } else {
                self.ranef_u_given_beta(&self.beta)
            };
            for (i, rt) in self.lmm.reterms.iter().enumerate() {
                self.u[i].copy_from(&new_u[i]);
                self.b[i] = &rt.lambda * &self.u[i];
            }
            self.update_eta();
            let mut obj = self.laplace_objective();

            // --- Step-halving: average toward the previous accepted state
            //     until obj is no worse, up to `max_halvings` averagings. ---
            // A non-finite obj must count as "worse": `NaN > bound` is false,
            // so without the explicit check a NaN/Inf iterate would skip
            // halving and be silently accepted (audit 03·H2 defense-in-depth;
            // the family μ-floors above are the primary fix).
            let mut nhalf = 0;
            while (!obj.is_finite() || obj > halving_bound) && nhalf < max_halvings {
                nhalf += 1;
                for i in 0..self.u.len() {
                    self.u[i] = 0.5 * (&self.u[i] + &u_prev[i]);
                }
                if vary_beta {
                    self.beta = 0.5 * (&self.beta + &beta_prev);
                }
                for (i, rt) in self.lmm.reterms.iter().enumerate() {
                    self.b[i] = &rt.lambda * &self.u[i];
                }
                self.update_eta();
                obj = self.laplace_objective();
            }

            if verbose {
                eprintln!("  PIRLS iter {iter}: obj = {obj:.6} (nhalf = {nhalf})");
            }

            if pirls_converged(obj, obj0, tol) {
                converged = true;
                break;
            }

            // Accept iterate as the new previous state.
            for i in 0..self.u.len() {
                u_prev[i].copy_from(&self.u[i]);
            }
            beta_prev = self.beta.clone();
            obj0 = obj;
            halving_bound = obj;
        }

        self.refresh_dispersion();

        Ok(converged)
    }

    /// Conditional modes of the random effects with β held fixed.
    ///
    /// `LinearMixedModel::ranef_u()` intentionally profiles β before forming
    /// residuals. The joint GLMM objective needs the lme4-style
    /// `nAGQ > 0` surface where the candidate β is part of the outer parameter
    /// vector, so the inner PIRLS step must solve only for `u` conditional on
    /// that β.
    fn ranef_u_given_beta(&self, beta: &DVector<f64>) -> Vec<DMatrix<f64>> {
        let k = self.lmm.reterms.len();
        let p = self.lmm.feterm.rank;
        let n = self.lmm.dims.n;
        let wtxy = &self.lmm.xy_mat.wtxy;

        let mut wr = vec![0.0f64; n];
        for obs in 0..n {
            let mut val = wtxy[(obs, p)];
            for q in 0..p {
                val -= wtxy[(obs, q)] * beta[q];
            }
            wr[obs] = val;
        }

        let mut c_vecs = Vec::with_capacity(k);
        for re in &self.lmm.reterms {
            let vs = re.vsize;
            let nranef = re.n_ranef();
            let n_levels = re.n_levels();

            let mut c = vec![0.0; nranef];
            for (obs, &wr_obs) in wr.iter().enumerate() {
                let r = re.refs[obs] as usize;
                for s in 0..vs {
                    c[r * vs + s] += re.wtz[(s, obs)] * wr_obs;
                }
            }

            let lambda = &re.lambda;
            let mut c_scaled = vec![0.0; nranef];
            for lev in 0..n_levels {
                for i in 0..vs {
                    let mut val = 0.0;
                    for row in i..vs {
                        val += lambda[(row, i)] * c[lev * vs + row];
                    }
                    c_scaled[lev * vs + i] = val;
                }
            }
            c_vecs.push(DVector::from_vec(c_scaled));
        }

        let mut v_vecs: Vec<DVector<f64>> = Vec::with_capacity(k);
        for j in 0..k {
            let nranef_j = self.lmm.reterms[j].n_ranef();
            let mut rhs = c_vecs[j].clone();

            for (m, v_m) in v_vecs.iter().enumerate().take(j) {
                let l_jm = self.lmm.l_blocks[glmm_block_index(j, m)].as_dense();
                for row in 0..nranef_j {
                    let mut dot = 0.0;
                    for col in 0..v_m.len() {
                        dot += l_jm[(row, col)] * v_m[col];
                    }
                    rhs[row] -= dot;
                }
            }

            let mut v_j = rhs.as_slice().to_vec();
            solve_dense_lower_against_rhs(
                &self.lmm.l_blocks[glmm_block_index(j, j)].as_dense(),
                &mut v_j,
            );
            v_vecs.push(DVector::from_vec(v_j));
        }

        let mut u_vecs: Vec<DVector<f64>> = vec![DVector::zeros(0); k];
        for j in (0..k).rev() {
            let nranef_j = self.lmm.reterms[j].n_ranef();
            let mut rhs = v_vecs[j].clone();

            for m in (j + 1)..k {
                let l_mj = self.lmm.l_blocks[glmm_block_index(m, j)].as_dense();
                let u_m = &u_vecs[m];
                for row in 0..nranef_j {
                    let mut dot = 0.0;
                    for col in 0..u_m.len() {
                        dot += l_mj[(col, row)] * u_m[col];
                    }
                    rhs[row] -= dot;
                }
            }

            let mut u_j = rhs.as_slice().to_vec();
            solve_dense_upper_from_lower_transpose_against_rhs(
                &self.lmm.l_blocks[glmm_block_index(j, j)].as_dense(),
                &mut u_j,
            );
            u_vecs[j] = DVector::from_vec(u_j);
        }

        self.lmm
            .reterms
            .iter()
            .zip(u_vecs)
            .map(|(rt, u)| DMatrix::from_column_slice(rt.vsize, rt.n_levels(), u.as_slice()))
            .collect()
    }

    /// Laplace approximation objective: deviance residuals + u penalty + log|L|.
    pub fn laplace_objective(&self) -> f64 {
        // For binomial-with-trials data the response is a per-trial proportion
        // and `wt[i]` is the trial count; weighting the per-observation
        // deviance contribution by `wt[i]` recovers the binomial deviance.
        let dev: f64 = (0..self.y.len())
            .map(|i| self.case_weight(i) * self.dev_resid_component(self.y[i], self.mu[i]))
            .sum();
        let u_penalty: f64 = self
            .u
            .iter()
            .map(|u| u.iter().map(|x| x * x).sum::<f64>())
            .sum();
        dev + u_penalty + self.lmm_logdet()
    }

    fn u_penalty(&self) -> f64 {
        self.u
            .iter()
            .map(|u| u.iter().map(|x| x * x).sum::<f64>())
            .sum()
    }

    fn minus_two_loglik_observation(&self, index: usize) -> f64 {
        let y = self.y[index];
        let mu = self.mu[index].max(f64::MIN_POSITIVE);
        match self.family {
            Family::Bernoulli | Family::Binomial => {
                let trials = self.case_weight(index).max(0.0);
                let successes = (trials * y).clamp(0.0, trials);
                let failures = trials - successes;
                let p = mu.clamp(1.0e-15, 1.0 - 1.0e-15);
                let log_choose = if trials == 0.0 {
                    0.0
                } else {
                    ln_gamma(trials + 1.0) - ln_gamma(successes + 1.0) - ln_gamma(failures + 1.0)
                };
                let success_term = if successes == 0.0 {
                    0.0
                } else {
                    successes * p.ln()
                };
                let failure_term = if failures == 0.0 {
                    0.0
                } else {
                    failures * (1.0 - p).ln()
                };
                -2.0 * (log_choose + success_term + failure_term)
            }
            Family::Poisson => {
                let count_term = if y == 0.0 { 0.0 } else { y * mu.ln() };
                -2.0 * (count_term - mu - ln_gamma(y + 1.0))
            }
            Family::NegativeBinomial => {
                let theta = self
                    .negative_binomial_theta
                    .expect("negative-binomial GLMM stores fixed theta");
                let loglik = ln_gamma(y + theta) - ln_gamma(theta) - ln_gamma(y + 1.0)
                    + theta * (theta / (theta + mu)).ln()
                    + if y == 0.0 {
                        0.0
                    } else {
                        y * (mu / (theta + mu)).ln()
                    };
                -2.0 * loglik
            }
            Family::Gamma => {
                let phi = self.dispersion(true).max(f64::MIN_POSITIVE);
                let shape = 1.0 / phi;
                let scale = mu * phi;
                -2.0 * ((shape - 1.0) * y.ln() - y / scale - shape * scale.ln() - ln_gamma(shape))
            }
            Family::Normal => {
                let variance = self.dispersion(true).max(f64::MIN_POSITIVE);
                let residual = y - mu;
                (2.0 * std::f64::consts::PI * variance).ln() + residual * residual / variance
            }
            Family::InverseGaussian => {
                let phi = self.dispersion(true).max(f64::MIN_POSITIVE);
                (2.0 * std::f64::consts::PI * phi * y.powi(3)).ln()
                    + (y - mu).powi(2) / (phi * y * mu * mu)
            }
        }
    }

    /// Additive difference between the current dropped-constant Laplace
    /// objective and the same conditional objective with response constants
    /// retained.
    ///
    /// For Poisson, negative-binomial, and binomial-family GLMMs this is an
    /// observation-only constant once family parameters are fixed.
    /// Dispersion families also depend on the current scale
    /// convention, so callers should treat those values as explicit metadata
    /// rather than as a cross-engine parity claim.
    pub fn response_constants_offset(&self) -> f64 {
        let dropped: f64 = (0..self.y.len())
            .map(|i| self.case_weight(i) * self.dev_resid_component(self.y[i], self.mu[i]))
            .sum();
        let included: f64 = (0..self.y.len())
            .map(|i| self.minus_two_loglik_observation(i))
            .sum();
        included - dropped
    }

    /// Laplace objective with response normalising constants retained.
    ///
    /// This is the objective convention needed for meaningful comparison to
    /// `lme4`'s `-2 logLik` scale. It deliberately lives alongside
    /// [`laplace_objective`](Self::laplace_objective) so current fast-PIRLS
    /// fitting and comparison artifacts keep their existing dropped-constant
    /// semantics while certified joint GLMM parity is promoted row by row.
    pub fn laplace_objective_with_response_constants(&self) -> f64 {
        (0..self.y.len())
            .map(|i| self.minus_two_loglik_observation(i))
            .sum::<f64>()
            + self.u_penalty()
            + self.lmm_logdet()
    }

    /// Deviance of the GLMM.
    ///
    /// For `n_agq <= 1`, returns the Laplace approximation
    /// (`laplace_objective`).
    ///
    /// For `n_agq > 1`, returns the deviance evaluated by `n_agq`-point
    /// adaptive Gauss-Hermite quadrature. AGQ is only defined for models
    /// with a single scalar random-effects term; on multi-term or
    /// vector-valued RE models, calling with `n_agq > 1` is a programmer
    /// error (use [`validate_agq`](Self::validate_agq) up front, or call
    /// via [`fit_with_options`](Self::fit_with_options) which preflights).
    ///
    /// Mutates internal `u`, `eta`, `mu` during the AGQ sweep but restores
    /// observable state before returning.
    pub fn deviance(&mut self, n_agq: usize) -> f64 {
        if n_agq <= 1 {
            return self.laplace_objective();
        }
        // Hard runtime check (not debug_assert!): in release a violated
        // invariant here would otherwise feed a multi-/vector-valued RE model
        // into the single-scalar AGQ math below, silently producing wrong
        // numbers (or an opaque index panic) rather than a clear refusal.
        assert!(
            self.is_single_scalar_re(),
            "AGQ with n_agq > 1 requires exactly one scalar random-effects term; \
             callers must invoke validate_agq() before reaching this path"
        );

        let n_levels = self.u[0].ncols();
        let n_obs = self.y.len();

        // Snapshot u₀ (a flat vector of length n_levels since vsize == 1).
        let u0_flat: Vec<f64> = self.u[0].as_slice().to_vec();
        debug_assert_eq!(u0_flat.len(), n_levels);

        // Per-group sd from the diagonal of the (1,1) Cholesky block:
        // sd[g] = 1 / |L₁₁_diag[g]|.
        let l11_diag = self.l11_diag();
        debug_assert_eq!(l11_diag.len(), n_levels);
        let sd: Vec<f64> = l11_diag.iter().map(|d| 1.0 / d.abs()).collect();

        // Group index per observation. Clone to release the borrow on
        // `self.lmm` so we can call `update_eta(&mut self)` inside the loop.
        let refs: Vec<u32> = self.lmm.reterms[0].refs.clone();

        // devc0[g] = u₀[g]² + Σ_{i in group g} devresid_i  (at the conditional modes)
        let mut devc0 = vec![0.0_f64; n_levels];
        for (g, &uv) in u0_flat.iter().enumerate() {
            devc0[g] = uv * uv;
        }
        for i in 0..n_obs {
            devc0[refs[i] as usize] +=
                self.case_weight(i) * self.dev_resid_component(self.y[i], self.mu[i]);
        }

        // Sweep over GH nodes.
        let rule = crate::types::gh_norm(n_agq);
        let mut mult = vec![0.0_f64; n_levels];
        let mut devc = vec![0.0_f64; n_levels];

        // From here on `u[0]`/`eta`/`mu` are perturbed at each node. The guard
        // restores them when this scope ends — including if the sweep panics.
        let mut work = AgqRestoreGuard {
            glmm: self,
            u0_flat: u0_flat.clone(),
        };

        for (&z, &w) in rule.z.iter().zip(rule.w.iter()) {
            if w == 0.0 {
                continue;
            }
            if z == 0.0 {
                // devc == devc0, exp(0) * w simplifies to w
                for g in 0..n_levels {
                    mult[g] += w;
                }
                continue;
            }
            // u[g] = u₀[g] + z * sd[g]
            for g in 0..n_levels {
                work.u[0][(0, g)] = u0_flat[g] + z * sd[g];
            }
            work.update_eta();
            // devc[g] = u[g]² + Σ devresid_i (per group)
            for g in 0..n_levels {
                let uv = work.u[0][(0, g)];
                devc[g] = uv * uv;
            }
            for i in 0..n_obs {
                devc[refs[i] as usize] +=
                    work.case_weight(i) * work.dev_resid_component(work.y[i], work.mu[i]);
            }
            // mult[g] += exp((z² + devc0[g] - devc[g]) / 2) * w
            let z2 = z * z;
            for g in 0..n_levels {
                mult[g] += ((z2 + devc0[g] - devc[g]) * 0.5).exp() * w;
            }
        }

        // `work` drops here, restoring u and η/μ (also on a panic above).
        drop(work);

        let sum_devc0: f64 = devc0.iter().sum();
        let log_mult: f64 = mult.iter().map(|m| m.ln()).sum();
        let log_sd: f64 = sd.iter().map(|s| s.ln()).sum();
        sum_devc0 - 2.0 * (log_mult + log_sd)
    }

    /// Deviance with response normalising constants retained.
    ///
    /// For `n_agq <= 1`, this is the Laplace objective on the `-2 logLik`
    /// scale. For AGQ, the quadrature objective is shifted by the same
    /// response-constant offset used by the Laplace path.
    pub fn deviance_with_response_constants(&mut self, n_agq: usize) -> f64 {
        if n_agq <= 1 {
            return self.laplace_objective_with_response_constants();
        }
        let offset = self.response_constants_offset();
        self.deviance(n_agq) + offset
    }

    fn case_weight(&self, obs: usize) -> f64 {
        if self.wt.is_empty() {
            1.0
        } else {
            self.wt[obs]
        }
    }

    /// True iff the model has exactly one random-effects term and that
    /// term has `vsize == 1` (a scalar random effect).
    pub fn is_single_scalar_re(&self) -> bool {
        self.lmm.reterms.len() == 1 && self.lmm.reterms[0].vsize == 1
    }

    /// Diagonal of the (1,1) block of the lower-Cholesky factor `L`.
    ///
    /// For a single scalar RE term this is a per-level vector of length
    /// `n_levels`. Used by [`deviance`](Self::deviance) to derive AGQ
    /// node spacings.
    fn l11_diag(&self) -> Vec<f64> {
        matrix_block_diag(&self.lmm.l_blocks[0])
    }

    /// Log-determinant from the LMM's Cholesky factor.
    pub(super) fn lmm_logdet(&self) -> f64 {
        // Delegate to the internal LMM's block structure
        let k = self.lmm.dims.nretrms;
        let mut logdet = 0.0;
        for j in 0..k {
            let idx = j * (j + 1) / 2 + j; // block_index(j, j)
            logdet += match &self.lmm.l_blocks[idx] {
                MatrixBlock::Dense(m) => {
                    let n = m.nrows().min(m.ncols());
                    (0..n).map(|i| m[(i, i)].abs().ln()).sum::<f64>()
                }
                MatrixBlock::Diagonal(v) => v.iter().map(|x| x.abs().ln()).sum::<f64>(),
                MatrixBlock::BlockDiagonal(blocks) => blocks
                    .iter()
                    .map(|blk| {
                        let n = blk.nrows();
                        (0..n).map(|i| blk[(i, i)].abs().ln()).sum::<f64>()
                    })
                    .sum::<f64>(),
                MatrixBlock::Sparse(m) => {
                    let dense = MatrixBlock::Sparse(m.clone()).as_dense();
                    let n = dense.nrows().min(dense.ncols());
                    (0..n).map(|i| dense[(i, i)].abs().ln()).sum::<f64>()
                }
            };
        }
        2.0 * logdet
    }

    /// Returns whether the inner PIRLS converged (see [`Self::pirls`]).
    pub(super) fn update_pirls_at_theta(&mut self, theta: &[f64], vary_beta: bool) -> Result<bool> {
        self.update_pirls_at_theta_with_options(theta, vary_beta, GLMM_PIRLS_MAX_ITER, true)
    }

    pub(super) fn update_pirls_at_theta_with_options(
        &mut self,
        theta: &[f64],
        vary_beta: bool,
        max_iter: usize,
        reset_modes: bool,
    ) -> Result<bool> {
        if theta.len() != self.theta.len() {
            return Err(MixedModelError::DimensionMismatch(format!(
                "theta vector length {} does not match fitted GLMM theta length {}",
                theta.len(),
                self.theta.len()
            )));
        }
        if !theta.iter().all(|value| value.is_finite()) {
            return Err(MixedModelError::InvalidArgument(
                "theta values must be finite".to_string(),
            ));
        }
        self.lmm.set_theta(theta)?;
        self.lmm.update_l()?;
        self.theta = theta.to_vec();
        let converged = self.pirls_with_options(vary_beta, false, max_iter, reset_modes)?;
        Ok(converged)
    }

    pub(super) fn penalized_pirls_deviance_at_theta(&mut self, theta: &[f64], n_agq: usize) -> f64 {
        if self.pending_progress_error.is_some() {
            return f64::INFINITY;
        }
        match self.update_pirls_at_theta(theta, true) {
            Ok(_) => {
                let deviance = self.deviance(n_agq);
                if deviance.is_finite() {
                    deviance
                } else {
                    f64::INFINITY
                }
            }
            Err(MixedModelError::Interrupted(message)) => {
                self.pending_progress_error = Some(message);
                f64::INFINITY
            }
            Err(_) => f64::INFINITY,
        }
    }

    pub(super) fn refresh_dispersion(&mut self) {
        self.dispersion = self.estimated_dispersion_scale();
    }

    fn estimated_dispersion_scale(&self) -> f64 {
        if let Some(theta) = self.negative_binomial_theta {
            return theta;
        }
        if !self.family.has_dispersion() {
            return 1.0;
        }

        let pearson = self.pearson_dispersion_numerator();
        let denom = self.y.len().saturating_sub(self.lmm.feterm.rank).max(1) as f64;
        let variance = (pearson / denom).max(f64::MIN_POSITIVE);
        variance.sqrt()
    }

    pub(super) fn estimate_negative_binomial_theta_given_fit(&self) -> Result<f64> {
        if self.family != Family::NegativeBinomial {
            return Err(MixedModelError::InvalidArgument(
                "negative-binomial theta estimation is only valid for Family::NegativeBinomial"
                    .to_string(),
            ));
        }
        let weights = (!self.wt.is_empty()).then_some(self.wt.as_slice());
        Ok(estimate_negative_binomial_theta_conditional(
            self.y.as_slice(),
            self.mu.as_slice(),
            weights,
        ))
    }

    pub(super) fn pearson_dispersion_numerator(&self) -> f64 {
        let mut total = 0.0;
        for obs in 0..self.y.len() {
            let mu = self.mu[obs];
            let variance = self.variance(mu);
            if !variance.is_finite() || variance <= 0.0 {
                continue;
            }
            let residual = self.y[obs] - mu;
            total += self.case_weight(obs) * residual * residual / variance;
        }
        total
    }
}
