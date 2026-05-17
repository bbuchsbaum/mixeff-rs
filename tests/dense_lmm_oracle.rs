#![cfg(feature = "unstable-internals")]

use approx::assert_relative_eq;
use nalgebra::{DMatrix, DVector};

use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::{DataFrame, LinearMixedModel, MixedModelFit};

#[derive(Clone, Copy)]
enum DenseRandomTerm<'a> {
    Scalar {
        group: &'a [usize],
        theta_index: usize,
    },
    Slope2 {
        group: &'a [usize],
        x: &'a [f64],
        theta_index: usize,
    },
}

fn fit(data: &DataFrame, formula: &str, reml: bool) -> LinearMixedModel {
    let formula = parse_formula(formula).unwrap();
    let mut model = LinearMixedModel::new(formula, data, None).unwrap();
    model.fit(reml).unwrap();
    model
}

fn group_count(group: &[usize]) -> usize {
    group.iter().copied().max().unwrap_or(0) + 1
}

fn dense_random_design(
    n: usize,
    terms: &[DenseRandomTerm<'_>],
    theta: &[f64],
    sigma: f64,
) -> DMatrix<f64> {
    let cols = terms
        .iter()
        .map(|term| match term {
            DenseRandomTerm::Scalar { group, .. } => group_count(group),
            DenseRandomTerm::Slope2 { group, .. } => 2 * group_count(group),
        })
        .sum();
    let mut z_lambda = DMatrix::zeros(n, cols);
    let mut offset = 0;
    for term in terms {
        match *term {
            DenseRandomTerm::Scalar { group, theta_index } => {
                let scale = theta[theta_index];
                for (row, &level) in group.iter().enumerate() {
                    z_lambda[(row, offset + level)] = scale;
                }
                offset += group_count(group);
            }
            DenseRandomTerm::Slope2 {
                group,
                x,
                theta_index,
            } => {
                let l11 = theta[theta_index];
                let l21 = theta[theta_index + 1];
                let l22 = theta[theta_index + 2];
                let n_groups = group_count(group);
                for (row, (&level, &xv)) in group.iter().zip(x.iter()).enumerate() {
                    let col = offset + 2 * level;
                    z_lambda[(row, col)] = l11 + xv * l21;
                    z_lambda[(row, col + 1)] = xv * l22;
                }
                offset += 2 * n_groups;
            }
        }
    }
    z_lambda * sigma
}

fn chol_logdet(matrix: &DMatrix<f64>) -> f64 {
    let chol = matrix.clone().cholesky().expect("positive definite matrix");
    2.0 * chol
        .l()
        .diagonal()
        .iter()
        .map(|value| value.ln())
        .sum::<f64>()
}

fn solve_spd(matrix: &DMatrix<f64>, rhs: &DMatrix<f64>) -> DMatrix<f64> {
    matrix
        .clone()
        .cholesky()
        .expect("positive definite matrix")
        .solve(rhs)
}

fn dense_profile_objective(
    y: &DVector<f64>,
    x: &DMatrix<f64>,
    terms: &[DenseRandomTerm<'_>],
    theta: &[f64],
    sigma: f64,
    reml: bool,
) -> Result<f64, &'static str> {
    let n = y.len();
    let z_lambda = dense_random_design(n, terms, theta, sigma);
    let v = &z_lambda * z_lambda.transpose() + DMatrix::<f64>::identity(n, n) * sigma.powi(2);
    let vinv_y = solve_spd(&v, &DMatrix::from_column_slice(n, 1, y.as_slice()));
    let vinv_x = solve_spd(&v, x);
    let xt_vinv_x = x.transpose() * &vinv_x;
    let Some(xt_chol) = xt_vinv_x.clone().cholesky() else {
        return Err("rank_deficient_fixed_effects");
    };
    let beta = xt_chol.solve(&(x.transpose() * vinv_y));
    let residual = y - x * beta.column(0);
    let q = (residual.transpose()
        * solve_spd(&v, &DMatrix::from_column_slice(n, 1, residual.as_slice())))[(0, 0)];
    let logdet_v = chol_logdet(&v);
    if reml {
        let p = x.ncols();
        let m = n - p;
        Ok(logdet_v
            + chol_logdet(&xt_vinv_x)
            + (m as f64) * (1.0 + (2.0 * std::f64::consts::PI * q / m as f64).ln()))
    } else {
        Ok(logdet_v + (n as f64) * (1.0 + (2.0 * std::f64::consts::PI * q / n as f64).ln()))
    }
}

fn scalar_data() -> (DataFrame, Vec<usize>, DMatrix<f64>, DVector<f64>) {
    let group = vec![0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3];
    let x = vec![
        -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0,
    ];
    let y = group
        .iter()
        .zip(x.iter())
        .map(|(&g, &xv)| 2.0 + 0.7 * xv + [-0.5, 0.2, 0.45, -0.15][g] + 0.08 * xv * xv)
        .collect::<Vec<_>>();
    let mut data = DataFrame::new();
    data.add_numeric("y", y.clone()).unwrap();
    data.add_numeric("x", x.clone()).unwrap();
    data.add_categorical("g", group.iter().map(|g| format!("g{g}")).collect())
        .unwrap();
    let xmat = DMatrix::from_fn(y.len(), 2, |row, col| if col == 0 { 1.0 } else { x[row] });
    (data, group, xmat, DVector::from_vec(y))
}

#[test]
fn dense_oracle_matches_scalar_random_intercept_ml() {
    let (data, group, xmat, y) = scalar_data();
    let model = fit(&data, "y ~ 1 + x + (1 | g)", false);
    let objective = dense_profile_objective(
        &y,
        &xmat,
        &[DenseRandomTerm::Scalar {
            group: &group,
            theta_index: 0,
        }],
        &model.theta(),
        model.sigma(),
        false,
    )
    .unwrap();
    assert_relative_eq!(
        model.objective(),
        objective,
        epsilon = 1e-6,
        max_relative = 1e-8
    );
}

#[test]
fn dense_oracle_matches_random_intercept_slope_ml() {
    let (data, group, xmat, y) = scalar_data();
    let x = data.numeric("x").unwrap().to_vec();
    let model = fit(&data, "y ~ 1 + x + (1 + x | g)", false);
    let objective = dense_profile_objective(
        &y,
        &xmat,
        &[DenseRandomTerm::Slope2 {
            group: &group,
            x: &x,
            theta_index: 0,
        }],
        &model.theta(),
        model.sigma(),
        false,
    )
    .unwrap();
    assert_relative_eq!(
        model.objective(),
        objective,
        epsilon = 1e-6,
        max_relative = 1e-8
    );
}

#[test]
fn dense_oracle_matches_crossed_random_intercepts_ml() {
    let site = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
    let item = vec![0, 1, 2, 0, 1, 2, 0, 1, 2];
    let x = vec![-1.0, 0.0, 1.0, -0.5, 0.5, 1.5, -1.5, -0.25, 0.75];
    let y = site
        .iter()
        .zip(item.iter())
        .zip(x.iter())
        .map(|((&s, &i), &xv)| 1.5 + 0.4 * xv + [-0.3, 0.15, 0.25][s] + [0.2, -0.1, 0.05][i])
        .collect::<Vec<_>>();
    let mut data = DataFrame::new();
    data.add_numeric("y", y.clone()).unwrap();
    data.add_numeric("x", x.clone()).unwrap();
    data.add_categorical("site", site.iter().map(|g| format!("s{g}")).collect())
        .unwrap();
    data.add_categorical("item", item.iter().map(|g| format!("i{g}")).collect())
        .unwrap();
    let xmat = DMatrix::from_fn(y.len(), 2, |row, col| if col == 0 { 1.0 } else { x[row] });
    let model = fit(&data, "y ~ 1 + x + (1 | site) + (1 | item)", false);
    let objective_site_item = dense_profile_objective(
        &DVector::from_vec(y),
        &xmat,
        &[
            DenseRandomTerm::Scalar {
                group: &site,
                theta_index: 0,
            },
            DenseRandomTerm::Scalar {
                group: &item,
                theta_index: 1,
            },
        ],
        &model.theta(),
        model.sigma(),
        false,
    )
    .unwrap();
    let objective_item_site = dense_profile_objective(
        model.response(),
        &xmat,
        &[
            DenseRandomTerm::Scalar {
                group: &item,
                theta_index: 0,
            },
            DenseRandomTerm::Scalar {
                group: &site,
                theta_index: 1,
            },
        ],
        &model.theta(),
        model.sigma(),
        false,
    )
    .unwrap();
    let delta_site_item = (model.objective() - objective_site_item).abs();
    let delta_item_site = (model.objective() - objective_item_site).abs();
    // The no-default native optimizer can stop farther from the tiny crossed
    // optimum than the nlopt-backed default path. Keep this crossed case as a
    // smoke oracle, while scalar and slope cases above enforce exact agreement.
    let tolerance = if cfg!(feature = "nlopt") { 2e-2 } else { 2.0 };
    assert!(
        delta_site_item.min(delta_item_site) <= tolerance,
        "crossed dense oracle mismatch: formula-order delta {delta_site_item}, optimizer-order delta {delta_item_site}"
    );
}

#[test]
fn dense_oracle_matches_near_zero_variance_boundary_ml() {
    let group = vec![0, 0, 1, 1, 2, 2, 3, 3];
    let x = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let noise = [0.01, -0.02, 0.015, -0.005, -0.012, 0.018, -0.009, 0.006];
    let y = x
        .iter()
        .zip(noise.iter())
        .map(|(&xv, &eps)| 1.0 + 0.5 * xv + eps)
        .collect::<Vec<_>>();
    let mut data = DataFrame::new();
    data.add_numeric("y", y.clone()).unwrap();
    data.add_numeric("x", x.clone()).unwrap();
    data.add_categorical("g", group.iter().map(|g| format!("g{g}")).collect())
        .unwrap();
    let xmat = DMatrix::from_fn(y.len(), 2, |row, col| if col == 0 { 1.0 } else { x[row] });
    let model = fit(&data, "y ~ 1 + x + (1 | g)", false);
    let objective = dense_profile_objective(
        &DVector::from_vec(y),
        &xmat,
        &[DenseRandomTerm::Scalar {
            group: &group,
            theta_index: 0,
        }],
        &model.theta(),
        model.sigma(),
        false,
    )
    .unwrap();
    assert!(model.is_singular(), "fixture should land on the boundary");
    assert_relative_eq!(
        model.objective(),
        objective,
        epsilon = 1e-5,
        max_relative = 1e-8
    );
}

#[test]
fn dense_oracle_rejects_rank_deficient_fixed_effect_design() {
    let (mut data, group, _, y) = scalar_data();
    let x = data.numeric("x").unwrap().to_vec();
    data.add_numeric("x_dup", x.iter().map(|value| 2.0 * value).collect())
        .unwrap();
    let x_rank_def = DMatrix::from_fn(y.len(), 3, |row, col| match col {
        0 => 1.0,
        1 => x[row],
        _ => 2.0 * x[row],
    });
    let model = fit(&data, "y ~ 1 + x + x_dup + (1 | g)", false);
    let covariance = model.fixed_effect_covariance_matrix();
    assert!(
        covariance.matrix.is_none(),
        "production model should expose rank-deficient fixed-effect covariance as unavailable"
    );
    assert_eq!(
        covariance.reason.as_deref(),
        Some("rank_deficient_fixed_effects")
    );
    let result = dense_profile_objective(
        &y,
        &x_rank_def,
        &[DenseRandomTerm::Scalar {
            group: &group,
            theta_index: 0,
        }],
        &model.theta(),
        model.sigma(),
        false,
    );
    assert_eq!(result.unwrap_err(), "rank_deficient_fixed_effects");
}
