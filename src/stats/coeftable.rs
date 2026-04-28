//! Coefficient table for fixed-effects estimates.
//!
//! Mirrors `coeftable(model)` in MixedModels.jl / StatsModels.jl.
//! Each row corresponds to one fixed-effect term and contains
//! the estimate, standard error, z-statistic, and two-sided Wald p-value.

use std::fmt;

use statrs::distribution::{ContinuousCDF, Normal};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoefTablePValuePolicy {
    AsymptoticWaldZ,
    Unavailable { reason: String },
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
#[derive(Debug, Clone)]
pub struct CoefTable {
    /// Column names (one per fixed-effects term).
    pub names: Vec<String>,
    /// Point estimates (β in pivot-free original column order).
    pub estimates: Vec<f64>,
    /// Standard errors of the estimates.
    pub std_errors: Vec<f64>,
    /// z-statistics: estimate / std_error.
    pub z_values: Vec<f64>,
    /// Two-sided Wald p-values from the standard normal distribution.
    pub p_values: Vec<f64>,
    /// Reason a p-value is missing. `None` means the p-value was computed.
    pub p_value_reasons: Vec<Option<String>>,
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

        CoefTable {
            names,
            estimates,
            std_errors,
            z_values,
            p_values,
            p_value_reasons,
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
}

impl fmt::Display for CoefTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{:<20} {:>12} {:>12} {:>10} {:>12}",
            "Name", "Estimate", "Std.Error", "z value", "Pr(>|z|)"
        )?;
        writeln!(
            f,
            "{:<20} {:>12} {:>12} {:>10} {:>12}",
            "----", "--------", "---------", "-------", "--------"
        )?;
        for i in 0..self.len() {
            let p_str = format_p_value(self.p_values[i], self.p_value_reasons[i].as_deref());
            writeln!(
                f,
                "{:<20} {:>12.4} {:>12.4} {:>10.4} {:>12}",
                self.names[i], self.estimates[i], self.std_errors[i], self.z_values[i], p_str
            )?;
        }
        Ok(())
    }
}

/// Render the coefficient table as a Markdown table.
pub fn coeftable_to_markdown(ct: &CoefTable) -> String {
    let mut out = String::new();
    out.push_str("| Name | Estimate | Std.Error | z | Pr(>|z|) |\n");
    out.push_str("|:-----|----------:|----------:|--:|----------:|\n");
    for i in 0..ct.len() {
        let p_str = format_p_value(ct.p_values[i], ct.p_value_reasons[i].as_deref());
        out.push_str(&format!(
            "| {} | {:.4} | {:.4} | {:.4} | {} |\n",
            ct.names[i], ct.estimates[i], ct.std_errors[i], ct.z_values[i], p_str
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
