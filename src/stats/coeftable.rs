//! Coefficient table for fixed-effects estimates.
//!
//! Mirrors `coeftable(model)` in MixedModels.jl / StatsModels.jl.
//! Each row corresponds to one fixed-effect term and contains
//! the estimate, standard error, z-statistic, and two-sided Wald p-value.

use std::fmt;

use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, Normal};

use crate::stats::profile::ConfintRow;

/// Policy for fixed-effect coefficient p-values in [`CoefTable`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum CoefTablePValuePolicy {
    /// Compute two-sided asymptotic Wald-z p-values.
    AsymptoticWaldZ,
    /// Mark p-values as unavailable with a shared reason.
    Unavailable {
        /// Explanation stored in each unavailable p-value row.
        reason: String,
    },
}

/// A coefficient table for the fixed-effects of a mixed model.
///
/// Mirrors `StatsModels.CoefTable` from Julia.  Columns are:
/// 1. Estimate (β)
/// 2. Std. Error
/// 3. z (Wald z-statistic = β / SE)
/// 4. Pr(>|z|) (two-sided p-value from standard normal)
///
/// The p-values use the z-distribution (large-sample approximation),
/// matching the default behaviour of `coeftable` in MixedModels.jl.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CoefTable {
    /// Column names (one per fixed-effects term).
    pub names: Vec<String>,
    /// Point estimates (β in pivot-free original column order).
    pub estimates: Vec<f64>,
    /// Standard errors of the estimates.
    pub std_errors: Vec<f64>,
    /// Test statistics (`estimate / std_error`). The reference distribution
    /// is named by [`statistic_name`](Self::statistic_name).
    pub z_values: Vec<f64>,
    /// Two-sided p-values. Computed from the standard normal for the
    /// default Wald-z method, or from a t-distribution with the row's
    /// [`df`](Self::df) for Satterthwaite/Kenward-Roger tables.
    pub p_values: Vec<f64>,
    /// Reason a p-value is missing. `None` means the p-value was computed.
    pub p_value_reasons: Vec<Option<String>>,
    /// Name of the reference distribution for the statistic column,
    /// e.g. `"z"` (asymptotic Wald) or `"t"` (Satterthwaite/Kenward-Roger).
    pub statistic_name: String,
    /// Inference method label, e.g. `"wald-z"`, `"satterthwaite"`,
    /// `"kenward-roger"`. Lets downstream clients see that a table is *not*
    /// asymptotic Wald-z even when the statistic column looks the same.
    pub method: String,
    /// Per-row denominator degrees of freedom. `None` for asymptotic
    /// methods (Wald-z); `Some(df)` for Satterthwaite/Kenward-Roger.
    pub df: Vec<Option<f64>>,
}

impl CoefTable {
    /// Construct a `CoefTable` directly from components.
    pub fn new(names: Vec<String>, estimates: Vec<f64>, std_errors: Vec<f64>) -> Self {
        Self::new_with_p_value_policy(
            names,
            estimates,
            std_errors,
            CoefTablePValuePolicy::AsymptoticWaldZ,
        )
    }

    /// Construct a table whose p-values are explicitly unavailable.
    pub fn new_without_p_values(
        names: Vec<String>,
        estimates: Vec<f64>,
        std_errors: Vec<f64>,
        reason: impl Into<String>,
    ) -> Self {
        Self::new_with_p_value_policy(
            names,
            estimates,
            std_errors,
            CoefTablePValuePolicy::Unavailable {
                reason: reason.into(),
            },
        )
    }

    /// Construct a table with an explicit p-value policy.
    pub fn new_with_p_value_policy(
        names: Vec<String>,
        estimates: Vec<f64>,
        std_errors: Vec<f64>,
        policy: CoefTablePValuePolicy,
    ) -> Self {
        let n = names.len();
        debug_assert_eq!(estimates.len(), n);
        debug_assert_eq!(std_errors.len(), n);

        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut z_values = Vec::with_capacity(n);
        let mut p_values = Vec::with_capacity(n);
        let mut p_value_reasons = Vec::with_capacity(n);

        for i in 0..n {
            let se = std_errors[i];
            let z = if se > 0.0 {
                estimates[i] / se
            } else {
                f64::NAN
            };
            z_values.push(z);
            match &policy {
                CoefTablePValuePolicy::AsymptoticWaldZ if z.is_finite() => {
                    p_values.push(2.0 * (1.0 - normal.cdf(z.abs())));
                    p_value_reasons.push(None);
                }
                CoefTablePValuePolicy::AsymptoticWaldZ => {
                    p_values.push(f64::NAN);
                    p_value_reasons.push(Some(
                        "standard error is unavailable, so the Wald z p-value is unavailable"
                            .to_string(),
                    ));
                }
                CoefTablePValuePolicy::Unavailable { reason } => {
                    p_values.push(f64::NAN);
                    p_value_reasons.push(Some(reason.clone()));
                }
            }
        }

        let df = vec![None; n];
        CoefTable {
            names,
            estimates,
            std_errors,
            z_values,
            p_values,
            p_value_reasons,
            statistic_name: "z".to_string(),
            method: "wald-z".to_string(),
            df,
        }
    }

    /// Construct a table from precomputed degrees-of-freedom-based inference
    /// (Satterthwaite / Kenward-Roger). Statistics and p-values are taken as
    /// given (not recomputed from the normal distribution) because their
    /// reference distribution is a t with the supplied per-row `df`.
    #[allow(clippy::too_many_arguments)]
    pub fn from_df_inference(
        names: Vec<String>,
        estimates: Vec<f64>,
        std_errors: Vec<f64>,
        statistics: Vec<f64>,
        p_values: Vec<f64>,
        p_value_reasons: Vec<Option<String>>,
        df: Vec<Option<f64>>,
        statistic_name: impl Into<String>,
        method: impl Into<String>,
    ) -> Self {
        let n = names.len();
        debug_assert_eq!(estimates.len(), n);
        debug_assert_eq!(std_errors.len(), n);
        debug_assert_eq!(statistics.len(), n);
        debug_assert_eq!(p_values.len(), n);
        debug_assert_eq!(p_value_reasons.len(), n);
        debug_assert_eq!(df.len(), n);
        CoefTable {
            names,
            estimates,
            std_errors,
            z_values: statistics,
            p_values,
            p_value_reasons,
            statistic_name: statistic_name.into(),
            method: method.into(),
            df,
        }
    }

    /// Number of rows (fixed-effects terms).
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// `true` if there are no fixed-effects terms.
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// Two-sided Wald confidence intervals for each coefficient at the given
    /// coverage `level` (e.g. `0.95`): `estimate ± z(1-α/2) · SE`.
    ///
    /// Rows align with the table's existing row order. A row's `lower`/`upper`
    /// are `NaN` when its standard error is non-positive/non-finite or `level`
    /// is not in `(0, 1)`. This is the large-sample interval; for small
    /// samples prefer the profile-likelihood CIs in
    /// [`crate::stats::profile`](mod@crate::stats::profile).
    pub fn wald_confint(&self, level: f64) -> Vec<ConfintRow> {
        let z = if level > 0.0 && level < 1.0 {
            Normal::new(0.0, 1.0)
                .unwrap()
                .inverse_cdf(1.0 - (1.0 - level) / 2.0)
        } else {
            f64::NAN
        };

        (0..self.len())
            .map(|i| {
                let estimate = self.estimates[i];
                let se = self.std_errors[i];
                let (lower, upper) = if z.is_finite() && se.is_finite() && se > 0.0 {
                    (estimate - z * se, estimate + z * se)
                } else {
                    (f64::NAN, f64::NAN)
                };
                ConfintRow {
                    parameter: self.names[i].clone(),
                    estimate,
                    lower,
                    upper,
                }
            })
            .collect()
    }
}

fn format_df(df: Option<f64>) -> String {
    match df {
        Some(v) if v.is_finite() => format!("{v:.2}"),
        _ => String::new(),
    }
}

impl fmt::Display for CoefTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let stat = &self.statistic_name;
        let stat_col = format!("{stat} value");
        let p_col = format!("Pr(>|{stat}|)");
        writeln!(f, "Method: {}", self.method)?;
        writeln!(
            f,
            "{:<20} {:>12} {:>12} {:>10} {:>10} {:>12}",
            "Name", "Estimate", "Std.Error", "df", stat_col, p_col
        )?;
        writeln!(
            f,
            "{:<20} {:>12} {:>12} {:>10} {:>10} {:>12}",
            "----", "--------", "---------", "--", "-------", "--------"
        )?;
        for i in 0..self.len() {
            let p_str = format_p_value(self.p_values[i], self.p_value_reasons[i].as_deref());
            writeln!(
                f,
                "{:<20} {:>12.4} {:>12.4} {:>10} {:>10.4} {:>12}",
                self.names[i],
                self.estimates[i],
                self.std_errors[i],
                format_df(self.df.get(i).copied().flatten()),
                self.z_values[i],
                p_str
            )?;
        }
        Ok(())
    }
}

/// Render the coefficient table as a Markdown table.
pub fn coeftable_to_markdown(ct: &CoefTable) -> String {
    let stat = &ct.statistic_name;
    let mut out = String::new();
    out.push_str(&format!("*Method: {}*\n\n", ct.method));
    out.push_str(&format!(
        "| Name | Estimate | Std.Error | df | {stat} | Pr(>|{stat}|) |\n"
    ));
    out.push_str("|:-----|----------:|----------:|---:|--:|----------:|\n");
    for i in 0..ct.len() {
        let p_str = format_p_value(ct.p_values[i], ct.p_value_reasons[i].as_deref());
        let df_str = format_df(ct.df.get(i).copied().flatten());
        out.push_str(&format!(
            "| {} | {:.4} | {:.4} | {} | {:.4} | {} |\n",
            ct.names[i], ct.estimates[i], ct.std_errors[i], df_str, ct.z_values[i], p_str
        ));
    }
    out
}

fn format_p_value(value: f64, missing_reason: Option<&str>) -> String {
    if let Some(reason) = missing_reason {
        return format!("NA ({reason})");
    }
    if !value.is_finite() {
        return "NA".to_string();
    }
    if value < 0.001 {
        "<0.001".to_string()
    } else {
        format!("{value:.4}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Basic sanity: z = estimate / se, p is two-sided.
    #[test]
    fn test_coeftable_z_and_p() {
        let ct = CoefTable::new(
            vec!["(Intercept)".to_string(), "x".to_string()],
            vec![10.0, 2.0],
            vec![2.0, 0.5],
        );
        assert_relative_eq!(ct.z_values[0], 5.0, epsilon = 1e-12);
        assert_relative_eq!(ct.z_values[1], 4.0, epsilon = 1e-12);
        assert_eq!(ct.p_value_reasons, vec![None, None]);
        // p for z=5: ≈ 5.73e-7
        assert!(ct.p_values[0] < 1e-5);
        // p for z=4: ≈ 6.3e-5
        assert!(ct.p_values[1] < 1e-3);
    }

    /// Symmetric about zero: z and -z give the same p-value.
    #[test]
    fn test_coeftable_p_symmetric() {
        let ct = CoefTable::new(
            vec!["pos".to_string(), "neg".to_string()],
            vec![3.0, -3.0],
            vec![1.0, 1.0],
        );
        assert_relative_eq!(ct.p_values[0], ct.p_values[1], epsilon = 1e-12);
    }

    /// Zero SE → NaN z and p.
    #[test]
    fn test_coeftable_zero_se_gives_nan() {
        let ct = CoefTable::new(vec!["b".to_string()], vec![1.0], vec![0.0]);
        assert!(ct.z_values[0].is_nan());
        assert!(ct.p_values[0].is_nan());
        assert!(ct.p_value_reasons[0]
            .as_deref()
            .unwrap()
            .contains("standard error is unavailable"));
    }

    #[test]
    fn test_coeftable_can_carry_missing_p_value_reasons() {
        let ct = CoefTable::new_without_p_values(
            vec!["x".to_string()],
            vec![1.0],
            vec![0.5],
            "exploratory fit intent does not permit ordinary p-values",
        );

        assert_eq!(ct.z_values[0], 2.0);
        assert!(ct.p_values[0].is_nan());
        assert_eq!(
            ct.p_value_reasons[0].as_deref(),
            Some("exploratory fit intent does not permit ordinary p-values")
        );
        assert!(coeftable_to_markdown(&ct).contains("NA (exploratory fit intent"));
    }

    /// Wald CI: estimate ± z(1-α/2)·SE, symmetric about the estimate.
    #[test]
    fn test_coeftable_wald_confint() {
        let ct = CoefTable::new(
            vec!["(Intercept)".to_string(), "x".to_string()],
            vec![10.0, 2.0],
            vec![2.0, 0.5],
        );
        let ci = ct.wald_confint(0.95);
        let z = 1.959_963_984_540_054_f64; // qnorm(0.975)
        assert_eq!(ci.len(), 2);
        assert_eq!(ci[0].parameter, "(Intercept)");
        assert_relative_eq!(ci[0].estimate, 10.0, epsilon = 1e-12);
        assert_relative_eq!(ci[0].lower, 10.0 - z * 2.0, epsilon = 1e-9);
        assert_relative_eq!(ci[0].upper, 10.0 + z * 2.0, epsilon = 1e-9);
        assert_relative_eq!(ci[1].lower, 2.0 - z * 0.5, epsilon = 1e-9);
        assert_relative_eq!(ci[1].upper, 2.0 + z * 0.5, epsilon = 1e-9);
        // Midpoint is the estimate.
        assert_relative_eq!((ci[1].lower + ci[1].upper) / 2.0, 2.0, epsilon = 1e-12);
    }

    /// Non-positive SE and invalid level both yield NaN bounds.
    #[test]
    fn test_coeftable_wald_confint_degenerate() {
        let ct = CoefTable::new(vec!["b".to_string()], vec![1.0], vec![0.0]);
        let ci = ct.wald_confint(0.95);
        assert!(ci[0].lower.is_nan() && ci[0].upper.is_nan());

        let ct2 = CoefTable::new(vec!["b".to_string()], vec![1.0], vec![0.5]);
        for bad in [0.0, 1.0, -0.1, 1.5] {
            let ci2 = ct2.wald_confint(bad);
            assert!(
                ci2[0].lower.is_nan() && ci2[0].upper.is_nan(),
                "level {bad} must give NaN bounds"
            );
        }
        // Wider level → wider interval.
        let w90 = {
            let r = ct2.wald_confint(0.90);
            r[0].upper - r[0].lower
        };
        let w99 = {
            let r = ct2.wald_confint(0.99);
            r[0].upper - r[0].lower
        };
        assert!(w99 > w90);
    }

    /// Display smoke test.
    #[test]
    fn test_coeftable_display() {
        let ct = CoefTable::new(vec!["(Intercept)".to_string()], vec![1526.0], vec![17.7]);
        let s = format!("{}", ct);
        assert!(s.contains("(Intercept)"));
        assert!(s.contains("1526"));
    }

    /// pls.jl "coeftable" testset: teststatcol=3, pvalcol=4.
    /// In our struct, z_values is index 2 (0-based), p_values is index 3.
    /// We verify the p-value for the dyestuff intercept is very small.
    #[test]
    fn test_coeftable_dyestuff_intercept_significant() {
        // Dyestuff: coef ≈ 1527.5, stderror ≈ 17.69
        // z ≈ 1527.5/17.69 ≈ 86.3 → p ≈ 0
        let ct = CoefTable::new(vec!["(Intercept)".to_string()], vec![1527.5], vec![17.69]);
        let z = ct.z_values[0];
        assert!(
            z > 80.0,
            "z should be very large for dyestuff intercept, got {}",
            z
        );
        let p = ct.p_values[0];
        assert!(p < 1e-10, "p-value should be near 0, got {}", p);
    }
}
