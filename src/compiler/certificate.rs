use serde::{Deserialize, Serialize};

use nalgebra::{DMatrix, SymmetricEigen};

use crate::compiler::diagnostics::FitStatus;
use crate::linalg::pivot::stats_rank_with_tol;

const RANK_TOLERANCE: f64 = 1e-8;
const EIGEN_TOLERANCE: f64 = 1e-8;

/// Generator-level specification for contract pathology fixtures.
///
/// This is intentionally independent of the fitting engine. It describes the
/// design and truth used to generate a fixture so the expected fit-status set
/// can be derived from linear algebra rather than optimizer behavior.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeneratorSpec {
    pub name: Option<String>,
    pub stratum: Option<String>,
    pub group_sizes: Vec<usize>,
    pub fe_truth: Vec<f64>,
    pub re_cov_truth: Vec<Vec<f64>>,
    pub family: String,
    pub link: String,
    pub intercept_for_prevalence: Option<f64>,
    pub seed: u64,
    #[serde(default = "default_residual_sd")]
    pub residual_sd: f64,
    #[serde(default)]
    pub fixed_design: FixedDesign,
    #[serde(default)]
    pub seed_sweep: Vec<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FixedDesign {
    BalancedX,
    CollinearFixedEffect,
    ConstantX,
}

impl Default for FixedDesign {
    fn default() -> Self {
        Self::BalancedX
    }
}

/// Linear-algebra certificate for a generated pathology fixture.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Certificate {
    pub fe_rank: usize,
    pub fe_columns: usize,
    pub re_rank: usize,
    pub re_dimension: usize,
    pub boundary_directions: Vec<usize>,
    pub separation: bool,
    pub fisher_eigvals: Vec<f64>,
    pub sample_size_per_param: f64,
}

/// Certify a fixture specification without constructing or fitting a model.
pub fn certify(spec: &GeneratorSpec) -> Certificate {
    let x = fixed_design_matrix(spec);
    let (fe_rank, _) = stats_rank_with_tol(&x, RANK_TOLERANCE);
    let fe_columns = x.ncols();
    let fisher = x.transpose() * x;
    let fisher_eigvals = sorted_eigenvalues(&fisher);

    let re_cov = covariance_matrix(&spec.re_cov_truth);
    let re_eigvals = sorted_eigenvalues(&re_cov);
    let max_re_eig = re_eigvals
        .iter()
        .copied()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let re_tol = EIGEN_TOLERANCE.max(EIGEN_TOLERANCE * max_re_eig);
    let boundary_directions = re_eigvals
        .iter()
        .enumerate()
        .filter_map(|(index, value)| (*value <= re_tol).then_some(index))
        .collect::<Vec<_>>();
    let re_rank = re_eigvals.len().saturating_sub(boundary_directions.len());

    let covariance_params = covariance_parameter_count(re_cov.ncols()).max(1);
    let sample_size_per_param = spec.group_sizes.len() as f64 / covariance_params as f64;

    Certificate {
        fe_rank,
        fe_columns,
        re_rank,
        re_dimension: re_cov.ncols(),
        boundary_directions,
        separation: detects_design_separation(spec),
        fisher_eigvals,
        sample_size_per_param,
    }
}

/// Deterministic status set implied by the certificate.
///
/// Near-boundary truth legitimately admits multiple optimizer outcomes, so this
/// returns acceptable statuses rather than a single expected status.
pub fn expected_statuses(certificate: &Certificate) -> Vec<FitStatus> {
    if certificate.separation
        || certificate.fe_rank < certificate.fe_columns
        || certificate.sample_size_per_param < 1.0
    {
        return vec![FitStatus::NotIdentifiable, FitStatus::NotOptimized];
    }

    if certificate.re_rank < certificate.re_dimension {
        return vec![
            FitStatus::ConvergedReducedRank,
            FitStatus::ConvergedBoundary,
            FitStatus::ConvergedInterior,
        ];
    }

    if !certificate.boundary_directions.is_empty() {
        return vec![FitStatus::ConvergedBoundary, FitStatus::ConvergedInterior];
    }

    vec![FitStatus::ConvergedInterior]
}

/// One composable fixture transform used by the first pathology corpus slice.
pub fn near_singular_re(rho: f64) -> Vec<Vec<f64>> {
    let rho = rho.clamp(-1.0, 1.0);
    vec![vec![1.0, rho], vec![rho, 1.0]]
}

pub fn generated_x_values(spec: &GeneratorSpec) -> Vec<f64> {
    spec.group_sizes
        .iter()
        .flat_map(|&group_size| {
            (0..group_size).map(move |index| match spec.fixed_design {
                FixedDesign::ConstantX => 0.0,
                FixedDesign::BalancedX | FixedDesign::CollinearFixedEffect => {
                    if group_size <= 1 {
                        0.0
                    } else {
                        -1.0 + 2.0 * index as f64 / (group_size - 1) as f64
                    }
                }
            })
        })
        .collect()
}

fn fixed_design_matrix(spec: &GeneratorSpec) -> DMatrix<f64> {
    let x_values = generated_x_values(spec);
    let n_rows = x_values.len();
    let n_cols = match spec.fixed_design {
        FixedDesign::CollinearFixedEffect => spec.fe_truth.len().max(3),
        _ => spec.fe_truth.len().max(1),
    };
    let mut x = DMatrix::zeros(n_rows, n_cols);
    for row in 0..n_rows {
        x[(row, 0)] = 1.0;
        if n_cols > 1 {
            x[(row, 1)] = x_values[row];
        }
        if n_cols > 2 {
            x[(row, 2)] = match spec.fixed_design {
                FixedDesign::CollinearFixedEffect => x_values[row],
                _ => x_values[row] * x_values[row],
            };
        }
        for col in 3..n_cols {
            x[(row, col)] = x_values[row].powi(col as i32);
        }
    }
    x
}

fn covariance_matrix(values: &[Vec<f64>]) -> DMatrix<f64> {
    let n = values.len();
    let mut matrix = DMatrix::zeros(n, n);
    for (row, values_row) in values.iter().enumerate() {
        for (col, value) in values_row.iter().enumerate().take(n) {
            matrix[(row, col)] = *value;
        }
    }
    matrix
}

fn sorted_eigenvalues(matrix: &DMatrix<f64>) -> Vec<f64> {
    if matrix.nrows() == 0 || matrix.ncols() == 0 {
        return Vec::new();
    }
    let eig = SymmetricEigen::new(matrix.clone());
    let mut values = eig.eigenvalues.iter().copied().collect::<Vec<_>>();
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    values
}

fn covariance_parameter_count(dimension: usize) -> usize {
    dimension * (dimension + 1) / 2
}

fn detects_design_separation(spec: &GeneratorSpec) -> bool {
    spec.family.eq_ignore_ascii_case("binomial")
        && matches!(spec.fixed_design, FixedDesign::ConstantX)
}

fn default_residual_sd() -> f64 {
    0.5
}

#[cfg(test)]
mod tests {
    use super::*;

    fn easy_spec() -> GeneratorSpec {
        GeneratorSpec {
            name: Some("easy".to_string()),
            stratum: Some("easy".to_string()),
            group_sizes: vec![6; 12],
            fe_truth: vec![2.0, 1.0],
            re_cov_truth: vec![vec![0.5, 0.1], vec![0.1, 0.25]],
            family: "gaussian".to_string(),
            link: "identity".to_string(),
            intercept_for_prevalence: None,
            seed: 11,
            residual_sd: 0.5,
            fixed_design: FixedDesign::BalancedX,
            seed_sweep: Vec::new(),
        }
    }

    #[test]
    fn certificate_identifies_full_rank_easy_spec() {
        let certificate = certify(&easy_spec());

        assert_eq!(certificate.fe_rank, certificate.fe_columns);
        assert_eq!(certificate.re_rank, 2);
        assert!(certificate.boundary_directions.is_empty());
        assert_eq!(
            expected_statuses(&certificate),
            vec![FitStatus::ConvergedInterior]
        );
    }

    #[test]
    fn near_singular_transform_has_boundary_direction_at_unit_correlation() {
        let mut spec = easy_spec();
        spec.re_cov_truth = near_singular_re(1.0);

        let certificate = certify(&spec);

        assert_eq!(certificate.re_rank, 1);
        assert!(expected_statuses(&certificate).contains(&FitStatus::ConvergedReducedRank));
    }

    #[test]
    fn certificate_flags_collinear_fixed_design_as_not_identifiable() {
        let mut spec = easy_spec();
        spec.fixed_design = FixedDesign::CollinearFixedEffect;

        let certificate = certify(&spec);

        assert!(certificate.fe_rank < certificate.fe_columns);
        assert!(expected_statuses(&certificate).contains(&FitStatus::NotIdentifiable));
    }
}
