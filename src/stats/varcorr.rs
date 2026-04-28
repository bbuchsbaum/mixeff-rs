//! Variance-covariance components display (VarCorr).

use std::fmt;

use crate::types::ReMat;

/// Variance and correlation of random effects.
#[derive(Debug, Clone)]
pub struct VarCorr {
    /// One entry per random-effects term.
    pub components: Vec<VarCorrComponent>,
    /// Residual standard deviation.
    pub residual_sd: Option<f64>,
}

/// Variance components for a single random-effects grouping factor.
#[derive(Debug, Clone)]
pub struct VarCorrComponent {
    /// Name of the grouping factor.
    pub group: String,
    /// Column names.
    pub names: Vec<String>,
    /// Standard deviations.
    pub std_dev: Vec<f64>,
    /// Correlation matrix (lower triangle, row-major).
    /// Empty for scalar random effects.
    pub correlations: Vec<f64>,
}

impl VarCorr {
    /// Extract VarCorr from random-effects terms with explicit scaling.
    pub fn from_reterms(reterms: &[ReMat], sd_scale: f64, residual_sd: Option<f64>) -> Self {
        let mut components = Vec::new();

        for rt in reterms {
            let s = rt.vsize;
            let lambda = &rt.lambda;

            // Standard deviations: scale * ||row_i(Λ)||
            let mut std_dev = Vec::with_capacity(s);
            for i in 0..s {
                let mut row_norm_sq = 0.0;
                for j in 0..=i {
                    row_norm_sq += lambda[(i, j)] * lambda[(i, j)];
                }
                std_dev.push(sd_scale * row_norm_sq.sqrt());
            }

            let mut correlations = Vec::new();
            if s > 1 {
                let mut normalized = vec![vec![0.0; s]; s];
                for i in 0..s {
                    let row_norm = std_dev[i] / sd_scale;
                    if row_norm > 0.0 {
                        for j in 0..=i {
                            normalized[i][j] = lambda[(i, j)] / row_norm;
                        }
                    }
                }
                for i in 1..s {
                    for j in 0..i {
                        let dot: f64 = (0..=j).map(|k| normalized[i][k] * normalized[j][k]).sum();
                        correlations.push(dot);
                    }
                }
            }

            components.push(VarCorrComponent {
                group: rt.grouping_name.clone(),
                names: rt.cnames.clone(),
                std_dev,
                correlations,
            });
        }

        VarCorr {
            components,
            residual_sd,
        }
    }

    /// Extract VarCorr from a fitted mixed model.
    pub fn from_model(reterms: &[ReMat], sigma: f64) -> Self {
        Self::from_reterms(reterms, sigma, Some(sigma))
    }

    /// Render a markdown summary table.
    pub fn to_markdown(&self) -> String {
        let has_corr = self
            .components
            .iter()
            .any(|comp| !comp.correlations.is_empty());
        let mut out = String::new();

        if has_corr {
            out.push_str("|          | Column      |  Variance |  Std.Dev | Corr. |\n");
            out.push_str("|:-------- |:----------- | ---------:| --------:| -----:|\n");
        } else {
            out.push_str("|          | Column      |  Variance |  Std.Dev |\n");
            out.push_str("|:-------- |:----------- | ---------:| --------:|\n");
        }

        for comp in &self.components {
            for (i, name) in comp.names.iter().enumerate() {
                let var = comp.std_dev[i] * comp.std_dev[i];
                let group = if i == 0 { comp.group.as_str() } else { "" };
                if has_corr {
                    let corr_text = if i == 0 || comp.correlations.is_empty() {
                        String::new()
                    } else {
                        let offset = i * (i - 1) / 2;
                        (0..i)
                            .map(|j| format!("{:+.2}", comp.correlations[offset + j]))
                            .collect::<Vec<_>>()
                            .join(" ")
                    };
                    out.push_str(&format!(
                        "| {:<8} | {:<11} | {:>9.6} | {:>8.6} | {} |\n",
                        group, name, var, comp.std_dev[i], corr_text
                    ));
                } else {
                    out.push_str(&format!(
                        "| {:<8} | {:<11} | {:>9.6} | {:>8.6} |\n",
                        group, name, var, comp.std_dev[i]
                    ));
                }
            }
        }

        if let Some(sigma) = self.residual_sd {
            if has_corr {
                out.push_str(&format!(
                    "| {:<8} | {:<11} | {:>9.6} | {:>8.6} |  |\n",
                    "Residual",
                    "",
                    sigma * sigma,
                    sigma
                ));
            } else {
                out.push_str(&format!(
                    "| {:<8} | {:<11} | {:>9.6} | {:>8.6} |\n",
                    "Residual",
                    "",
                    sigma * sigma,
                    sigma
                ));
            }
        }

        out
    }
}

impl fmt::Display for VarCorr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Variance components:")?;
        writeln!(
            f,
            "{:<12} {:<16} {:>12} {:>10}  {}",
            "Groups", "Name", "Variance", "Std.Dev.", "Corr."
        )?;

        for comp in &self.components {
            for (i, name) in comp.names.iter().enumerate() {
                let var = comp.std_dev[i] * comp.std_dev[i];
                if i == 0 {
                    write!(
                        f,
                        "{:<12} {:<16} {:>12.4} {:>10.4}",
                        comp.group, name, var, comp.std_dev[i]
                    )?;
                } else {
                    write!(
                        f,
                        "{:<12} {:<16} {:>12.4} {:>10.4}",
                        "", name, var, comp.std_dev[i]
                    )?;
                }
                // Print correlations for this row
                if i > 0 && !comp.correlations.is_empty() {
                    let offset = i * (i - 1) / 2;
                    for j in 0..i {
                        write!(f, " {:>+6.2}", comp.correlations[offset + j])?;
                    }
                }
                writeln!(f)?;
            }
        }

        if let Some(sigma) = self.residual_sd {
            writeln!(
                f,
                "{:<12} {:<16} {:>12.4} {:>10.4}",
                "Residual",
                "",
                sigma * sigma,
                sigma
            )?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::DMatrix;

    fn scalar_reterm() -> ReMat {
        let mut re = ReMat::new(
            "subj".to_string(),
            vec![0, 1, 2],
            vec!["s1".to_string(), "s2".to_string(), "s3".to_string()],
            vec!["(Intercept)".to_string()],
            DMatrix::from_row_slice(1, 3, &[1.0, 1.0, 1.0]),
        );
        re.set_theta(&[0.75]);
        re
    }

    fn vector_reterm() -> ReMat {
        let mut re = ReMat::new(
            "item".to_string(),
            vec![0, 1, 0, 1],
            vec!["i1".to_string(), "i2".to_string()],
            vec!["(Intercept)".to_string(), "days".to_string()],
            DMatrix::from_row_slice(2, 4, &[1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
        );
        re.set_theta(&[2.0, 0.5, 1.0]);
        re
    }

    #[test]
    fn test_to_markdown_with_corr_and_residual() {
        let vc = VarCorr::from_reterms(&[scalar_reterm(), vector_reterm()], 3.0, Some(3.0));
        let out = vc.to_markdown();

        assert!(out.contains("|          | Column      |  Variance |  Std.Dev | Corr. |"));
        assert!(out.contains("| subj     | (Intercept) |  5.062500 | 2.250000 |"));
        assert!(out.contains("| item     | (Intercept) | 36.000000 | 6.000000 |"));
        assert!(out.contains("|          | days        | 11.250000 | 3.354102 | +0.45 |"));
        assert!(out.contains("| Residual |             |  9.000000 | 3.000000 |  |"));
    }

    #[test]
    fn test_to_markdown_without_corr_or_residual() {
        let vc = VarCorr::from_reterms(&[scalar_reterm()], 2.0, None);
        let out = vc.to_markdown();

        assert!(out.contains("|          | Column      |  Variance |  Std.Dev |"));
        assert!(!out.contains("Corr."));
        assert!(out.contains("| subj     | (Intercept) |  2.250000 | 1.500000 |"));
        assert!(!out.contains("Residual"));
    }

    // ── Tests ported from MixedModels.jl/test/pls.jl (VarCorr section) ─────

    #[test]
    fn test_scalar_re_std_dev_is_sigma_times_theta() {
        // scalar RE: λ = [[0.75]], σ = 2.0 → std_dev = 2.0 * 0.75 = 1.5
        let vc = VarCorr::from_reterms(&[scalar_reterm()], 2.0, Some(2.0));
        let comp = &vc.components[0];

        assert_eq!(comp.std_dev.len(), 1);
        assert_relative_eq!(comp.std_dev[0], 1.5, epsilon = 1e-12);
        assert!(
            comp.correlations.is_empty(),
            "scalar RE has no correlations"
        );
        assert_relative_eq!(vc.residual_sd.unwrap(), 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_vector_re_std_dev_and_correlation() {
        // vector RE: λ = [[2.0, 0], [0.5, 1.0]], σ = 1.0
        // std_dev[0] = 1.0 * √(2²)           = 2.0
        // std_dev[1] = 1.0 * √(0.5² + 1²)    = √1.25
        // corr[0]    = dot(norm_row0, norm_row1) = 0.5 / √1.25
        let vc = VarCorr::from_reterms(&[vector_reterm()], 1.0, Some(1.0));
        let comp = &vc.components[0];

        assert_eq!(comp.std_dev.len(), 2);
        assert_relative_eq!(comp.std_dev[0], 2.0, epsilon = 1e-12);
        assert_relative_eq!(comp.std_dev[1], f64::sqrt(1.25), epsilon = 1e-12);

        assert_eq!(comp.correlations.len(), 1);
        assert_relative_eq!(comp.correlations[0], 0.5 / f64::sqrt(1.25), epsilon = 1e-12);
    }

    #[test]
    fn test_varcorr_group_and_column_names() {
        let vc = VarCorr::from_reterms(&[scalar_reterm(), vector_reterm()], 1.0, Some(1.0));
        assert_eq!(vc.components[0].group, "subj");
        assert_eq!(vc.components[0].names, vec!["(Intercept)"]);
        assert_eq!(vc.components[1].group, "item");
        assert_eq!(vc.components[1].names, vec!["(Intercept)", "days"]);
    }
}
