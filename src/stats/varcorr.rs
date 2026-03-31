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
    /// Extract VarCorr from a fitted mixed model.
    pub fn from_model(reterms: &[ReMat], sigma: f64) -> Self {
        let mut components = Vec::new();

        for rt in reterms {
            let s = rt.vsize;
            let lambda = &rt.lambda;

            // Standard deviations: σ * ||row_i(Λ)||
            let mut std_dev = Vec::with_capacity(s);
            for i in 0..s {
                let mut row_norm_sq = 0.0;
                for j in 0..=i {
                    row_norm_sq += lambda[(i, j)] * lambda[(i, j)];
                }
                std_dev.push(sigma * row_norm_sq.sqrt());
            }

            // Correlations (for vector-valued RE)
            let mut correlations = Vec::new();
            if s > 1 {
                // Normalize rows
                let mut normalized = vec![vec![0.0; s]; s];
                for i in 0..s {
                    let row_norm = std_dev[i] / sigma;
                    if row_norm > 0.0 {
                        for j in 0..=i {
                            normalized[i][j] = lambda[(i, j)] / row_norm;
                        }
                    }
                }
                // Compute correlations: ρ(i,j) = dot(row_i, row_j) for i > j
                for i in 1..s {
                    for j in 0..i {
                        let dot: f64 = (0..=j)
                            .map(|k| normalized[i][k] * normalized[j][k])
                            .sum();
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
            residual_sd: Some(sigma),
        }
    }
}

impl fmt::Display for VarCorr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Variance components:")?;
        writeln!(f, "{:<12} {:<16} {:>12} {:>10}  {}",
                 "Groups", "Name", "Variance", "Std.Dev.", "Corr.")?;

        for comp in &self.components {
            for (i, name) in comp.names.iter().enumerate() {
                let var = comp.std_dev[i] * comp.std_dev[i];
                if i == 0 {
                    write!(f, "{:<12} {:<16} {:>12.4} {:>10.4}",
                           comp.group, name, var, comp.std_dev[i])?;
                } else {
                    write!(f, "{:<12} {:<16} {:>12.4} {:>10.4}",
                           "", name, var, comp.std_dev[i])?;
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
            writeln!(f, "{:<12} {:<16} {:>12.4} {:>10.4}",
                     "Residual", "", sigma * sigma, sigma)?;
        }

        Ok(())
    }
}
