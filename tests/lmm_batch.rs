use approx::assert_relative_eq;
use nalgebra::DMatrix;

use mixedmodels::formula::parse_formula;
use mixedmodels::model::{
    BatchOptimizerControl, BatchOptions, BatchThetaGrouping, BatchWarmStart, DataFrame,
    LinearMixedModel, LinearMixedModelBatch, MixedModelFit, ResponseBatchMode, ResponseFitStatus,
    ThetaBatch,
};

fn random_intercept_data() -> DataFrame {
    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut group = Vec::new();
    for g in 0..10 {
        let shift = (g as f64 - 4.5) * 0.8;
        for obs in 0..5 {
            let xv = obs as f64 - 2.0;
            let eps = ((g * 17 + obs * 11) % 13) as f64 * 0.03 - 0.18;
            y.push(2.0 + 1.4 * xv + shift + eps);
            x.push(xv);
            group.push(format!("g{g:02}"));
        }
    }

    let mut data = DataFrame::new();
    data.add_numeric("y", y).unwrap();
    data.add_numeric("x", x).unwrap();
    data.add_categorical("group", group).unwrap();
    data
}

fn fitted_model() -> LinearMixedModel {
    let data = random_intercept_data();
    let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(true).unwrap();
    model
}

fn response_matrix(model: &LinearMixedModel) -> DMatrix<f64> {
    let y = model.response();
    let mut responses = DMatrix::zeros(y.len(), 3);
    for row in 0..y.len() {
        responses[(row, 0)] = y[row];
        responses[(row, 1)] = 0.75 * y[row] + 3.0;
        responses[(row, 2)] = responses[(row, 1)];
    }
    responses
}

#[test]
fn profile_at_theta_matches_existing_matrix_profiler() {
    let model = fitted_model();
    let responses = response_matrix(&model);
    let theta = model.theta();
    let existing = model.profile_response_matrix(&responses, true).unwrap();
    let batch = LinearMixedModelBatch::from_model(&model).unwrap();

    let fit = batch
        .fit_responses(
            &responses,
            ResponseBatchMode::ProfileAtTheta { theta, reml: true },
        )
        .unwrap();

    assert!(fit
        .status
        .iter()
        .all(|status| *status == ResponseFitStatus::Success));
    for col in 0..responses.ncols() {
        assert_relative_eq!(fit.sigma[col], existing.sigma[col], epsilon = 1e-9);
        assert_relative_eq!(fit.pwrss[col], existing.pwrss[col], epsilon = 1e-7);
        assert_relative_eq!(fit.objective[col], existing.objectives[col], epsilon = 1e-8);
        for row in 0..fit.beta.nrows() {
            assert_relative_eq!(
                fit.beta[(row, col)],
                existing.beta[(row, col)],
                epsilon = 1e-9
            );
        }
    }
}

#[test]
fn profile_at_theta_is_invariant_to_chunk_size() {
    let model = fitted_model();
    let responses = response_matrix(&model);
    let theta = model.theta();
    let chunked = LinearMixedModelBatch::from_model_with_options(
        &model,
        BatchOptions {
            chunk_columns: 1,
            max_failures: None,
        },
    )
    .unwrap();
    let unchunked = LinearMixedModelBatch::from_model_with_options(
        &model,
        BatchOptions {
            chunk_columns: responses.ncols(),
            max_failures: None,
        },
    )
    .unwrap();

    let fit_chunked = chunked
        .fit_responses(
            &responses,
            ResponseBatchMode::ProfileAtTheta {
                theta: theta.clone(),
                reml: true,
            },
        )
        .unwrap();
    let fit_unchunked = unchunked
        .fit_responses(
            &responses,
            ResponseBatchMode::ProfileAtTheta { theta, reml: true },
        )
        .unwrap();

    for col in 0..responses.ncols() {
        assert_relative_eq!(
            fit_chunked.sigma[col],
            fit_unchunked.sigma[col],
            epsilon = 1e-10
        );
        assert_relative_eq!(
            fit_chunked.objective[col],
            fit_unchunked.objective[col],
            epsilon = 1e-10
        );
        for row in 0..fit_chunked.beta.nrows() {
            assert_relative_eq!(
                fit_chunked.beta[(row, col)],
                fit_unchunked.beta[(row, col)],
                epsilon = 1e-10
            );
        }
    }
}

#[test]
fn invalid_and_constant_columns_are_column_local() {
    let model = fitted_model();
    let y = model.response();
    let mut responses = DMatrix::zeros(y.len(), 3);
    for row in 0..y.len() {
        responses[(row, 0)] = y[row];
        responses[(row, 1)] = 4.2;
        responses[(row, 2)] = y[row];
    }
    responses[(3, 2)] = f64::NAN;

    let batch = LinearMixedModelBatch::from_model(&model).unwrap();
    let fit = batch
        .fit_responses(
            &responses,
            ResponseBatchMode::ProfileAtTheta {
                theta: model.theta(),
                reml: true,
            },
        )
        .unwrap();

    assert_eq!(fit.status[0], ResponseFitStatus::Success);
    assert_eq!(fit.status[1], ResponseFitStatus::ConstantResponse);
    assert_eq!(fit.status[2], ResponseFitStatus::InvalidResponse);
    assert_eq!(fit.success_count(), 1);
    assert!(fit.sigma[0].is_finite());
    assert!(fit.sigma[1].is_nan());
    assert!(fit.sigma[2].is_nan());
}

#[test]
fn optimize_shared_theta_profiles_at_returned_theta() {
    let model = fitted_model();
    let responses = response_matrix(&model);
    let batch = LinearMixedModelBatch::from_model(&model).unwrap();
    let control = BatchOptimizerControl {
        max_evaluations: 80,
        theta_tolerance: 1e-5,
        objective_tolerance: 1e-8,
        initial_step: Some(vec![0.25]),
        options: BatchOptions {
            chunk_columns: 2,
            max_failures: None,
        },
    };

    let fit = batch
        .fit_responses(
            &responses,
            ResponseBatchMode::OptimizeSharedTheta {
                reml: true,
                control,
            },
        )
        .unwrap();
    let theta = match &fit.theta {
        ThetaBatch::Shared(theta) => theta.clone(),
        other => panic!("expected shared theta, got {other:?}"),
    };
    let profiled = batch
        .fit_responses(
            &responses,
            ResponseBatchMode::ProfileAtTheta { theta, reml: true },
        )
        .unwrap();

    for col in 0..responses.ncols() {
        assert_eq!(fit.status[col], ResponseFitStatus::Success);
        assert_relative_eq!(fit.objective[col], profiled.objective[col], epsilon = 1e-8);
        assert_relative_eq!(fit.sigma[col], profiled.sigma[col], epsilon = 1e-8);
    }
}

#[test]
fn optimize_per_column_agrees_with_scalar_refit_loop() {
    let model = fitted_model();
    let responses = response_matrix(&model);
    let batch = LinearMixedModelBatch::from_model(&model).unwrap();
    let control = BatchOptimizerControl {
        max_evaluations: 160,
        theta_tolerance: 1e-5,
        objective_tolerance: 1e-8,
        initial_step: Some(vec![0.25]),
        options: BatchOptions {
            chunk_columns: 1,
            max_failures: None,
        },
    };

    let fit = batch
        .fit_responses(
            &responses,
            ResponseBatchMode::OptimizePerColumn {
                reml: true,
                warm_start: BatchWarmStart::TemplateTheta,
                control,
            },
        )
        .unwrap();
    let theta = match &fit.theta {
        ThetaBatch::PerColumn(theta) => theta,
        other => panic!("expected per-column theta, got {other:?}"),
    };

    for col in 0..responses.ncols() {
        let mut scalar = model.clone();
        scalar.refit(responses.column(col).as_slice()).unwrap();
        assert_eq!(fit.status[col], ResponseFitStatus::Success);
        assert_relative_eq!(fit.objective[col], scalar.objective(), epsilon = 2e-3);
        assert_relative_eq!(fit.sigma[col], scalar.sigma(), epsilon = 2e-3);
        assert_relative_eq!(theta[(0, col)], scalar.theta()[0], epsilon = 2e-3);
    }
}

#[test]
fn optimize_grouped_returns_one_theta_per_requested_group() {
    let model = fitted_model();
    let responses = response_matrix(&model);
    let batch = LinearMixedModelBatch::from_model(&model).unwrap();
    let control = BatchOptimizerControl {
        max_evaluations: 80,
        theta_tolerance: 1e-5,
        objective_tolerance: 1e-8,
        initial_step: Some(vec![0.25]),
        options: BatchOptions {
            chunk_columns: 2,
            max_failures: None,
        },
    };

    let fit = batch
        .fit_responses(
            &responses,
            ResponseBatchMode::OptimizeGrouped {
                reml: true,
                grouping: BatchThetaGrouping::ColumnGroups(vec![0, 1, 1]),
                control,
            },
        )
        .unwrap();

    assert!(fit
        .status
        .iter()
        .all(|status| *status == ResponseFitStatus::Success));
    let (theta, groups) = match &fit.theta {
        ThetaBatch::Grouped {
            theta,
            group_for_column,
        } => (theta, group_for_column),
        other => panic!("expected grouped theta, got {other:?}"),
    };
    assert_eq!(groups, &vec![0, 1, 1]);
    assert_eq!(theta.ncols(), 2);
    assert_relative_eq!(fit.objective[1], fit.objective[2], epsilon = 1e-10);
    assert_relative_eq!(fit.sigma[1], fit.sigma[2], epsilon = 1e-10);
    for row in 0..fit.beta.nrows() {
        assert_relative_eq!(fit.beta[(row, 1)], fit.beta[(row, 2)], epsilon = 1e-10);
    }
}
