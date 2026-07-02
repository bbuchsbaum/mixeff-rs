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
