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
