use approx::assert_relative_eq;
use nalgebra::DMatrix;

use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::{
    batch::{
        BatchOptimizerControl, BatchOptions, BatchThetaGrouping, BatchWarmStart,
        LinearMixedModelBatch, ResponseBatchMode, ResponseFitStatus, ThetaBatch,
    },
    DataFrame, LinearMixedModel, MixedModelFit,
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
            ..BatchOptions::default()
        },
    )
    .unwrap();
    let unchunked = LinearMixedModelBatch::from_model_with_options(
        &model,
        BatchOptions {
            chunk_columns: responses.ncols(),
            ..BatchOptions::default()
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
            ..BatchOptions::default()
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
            ..BatchOptions::default()
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
            ..BatchOptions::default()
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

#[cfg(not(feature = "rayon"))]
#[test]
fn requesting_rayon_parallelism_without_feature_is_a_typed_error() {
    use mixeff_rs::model::batch::BatchParallelism;

    let model = fitted_model();
    let result = LinearMixedModelBatch::from_model_with_options(
        &model,
        BatchOptions {
            parallelism: BatchParallelism::Rayon,
            ..BatchOptions::default()
        },
    );
    assert!(
        result.is_err(),
        "Rayon without the feature must not fall back to serial"
    );
}

#[cfg(feature = "rayon")]
mod parallel_determinism {
    use super::*;
    use mixeff_rs::model::batch::BatchParallelism;

    fn wide_response_matrix(model: &LinearMixedModel, q: usize) -> DMatrix<f64> {
        let y = model.response();
        DMatrix::from_fn(y.len(), q, |row, col| {
            let scale = 0.5 + 0.05 * col as f64;
            let shift = (col as f64 - q as f64 / 2.0) * 0.3;
            scale * y[row] + shift + ((row * 7 + col * 13) % 11) as f64 * 0.01
        })
    }

    /// Bitwise equality for batch fits: exact `to_bits` agreement on every
    /// floating-point entry (so NaN placeholders compare equal) plus exact
    /// status/diagnostic agreement.
    fn assert_fits_bitwise_identical(
        left: &mixeff_rs::model::batch::ResponseBatchFit,
        right: &mixeff_rs::model::batch::ResponseBatchFit,
    ) {
        let bits = |values: &[f64]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(left.beta.as_slice()), bits(right.beta.as_slice()));
        assert_eq!(bits(left.sigma.as_slice()), bits(right.sigma.as_slice()));
        assert_eq!(bits(left.pwrss.as_slice()), bits(right.pwrss.as_slice()));
        assert_eq!(
            bits(left.objective.as_slice()),
            bits(right.objective.as_slice())
        );
        match (&left.theta, &right.theta) {
            (ThetaBatch::Shared(a), ThetaBatch::Shared(b)) => assert_eq!(bits(a), bits(b)),
            (ThetaBatch::PerColumn(a), ThetaBatch::PerColumn(b)) => {
                assert_eq!(bits(a.as_slice()), bits(b.as_slice()))
            }
            (
                ThetaBatch::Grouped {
                    theta: a,
                    group_for_column: ga,
                },
                ThetaBatch::Grouped {
                    theta: b,
                    group_for_column: gb,
                },
            ) => {
                assert_eq!(bits(a.as_slice()), bits(b.as_slice()));
                assert_eq!(ga, gb);
            }
            (
                ThetaBatch::Adaptive {
                    theta: a,
                    group_for_column: ga,
                    refined_columns: ra,
                },
                ThetaBatch::Adaptive {
                    theta: b,
                    group_for_column: gb,
                    refined_columns: rb,
                },
            ) => {
                assert_eq!(bits(a.as_slice()), bits(b.as_slice()));
                assert_eq!(ga, gb);
                assert_eq!(ra, rb);
            }
            (a, b) => panic!("theta variants differ: {a:?} vs {b:?}"),
        }
        assert_eq!(left.status, right.status);
        assert_eq!(left.diagnostics, right.diagnostics);
    }

    #[test]
    fn parallel_profile_at_theta_is_identical_to_serial() {
        let model = fitted_model();
        let responses = wide_response_matrix(&model, 24);
        let theta = model.theta();

        let serial = LinearMixedModelBatch::from_model_with_options(
            &model,
            BatchOptions {
                chunk_columns: 3,
                ..BatchOptions::default()
            },
        )
        .unwrap();
        let parallel = LinearMixedModelBatch::from_model_with_options(
            &model,
            BatchOptions {
                chunk_columns: 3,
                parallelism: BatchParallelism::Rayon,
                ..BatchOptions::default()
            },
        )
        .unwrap();

        let mode = |theta: Vec<f64>| ResponseBatchMode::ProfileAtTheta { theta, reml: true };
        let fit_serial = serial
            .fit_responses(&responses, mode(theta.clone()))
            .unwrap();
        let fit_parallel = parallel.fit_responses(&responses, mode(theta)).unwrap();

        assert_fits_bitwise_identical(&fit_serial, &fit_parallel);
    }

    #[test]
    fn parallel_per_column_is_identical_to_serial() {
        let model = fitted_model();
        let responses = wide_response_matrix(&model, 17);

        let batch = LinearMixedModelBatch::from_model(&model).unwrap();
        let control = |parallelism| BatchOptimizerControl {
            max_evaluations: 120,
            theta_tolerance: 1e-5,
            objective_tolerance: 1e-8,
            initial_step: Some(vec![0.25]),
            options: BatchOptions {
                chunk_columns: 4,
                parallelism,
                ..BatchOptions::default()
            },
        };

        let mode = |parallelism| ResponseBatchMode::OptimizePerColumn {
            reml: true,
            warm_start: BatchWarmStart::TemplateTheta,
            control: control(parallelism),
        };

        let fit_serial = batch
            .fit_responses(&responses, mode(BatchParallelism::Serial))
            .unwrap();
        let fit_parallel = batch
            .fit_responses(&responses, mode(BatchParallelism::Rayon))
            .unwrap();

        assert_fits_bitwise_identical(&fit_serial, &fit_parallel);
    }

    #[test]
    fn parallel_wide_profile_crosses_chunk_fanout_thresholds() {
        // The chunk-parallel branch in LmmWorkspace::profile_columns only
        // engages with >= 4 chunks and >= 16384 profiled cells (columns x n).
        // With n = 50 rows, q = 340 and chunk_columns = 64 crosses both, so
        // this test genuinely executes the rayon fan-out + in-order scatter.
        let model = fitted_model();
        let responses = wide_response_matrix(&model, 340);
        let theta = model.theta();

        let build = |parallelism| {
            LinearMixedModelBatch::from_model_with_options(
                &model,
                BatchOptions {
                    chunk_columns: 64,
                    parallelism,
                    ..BatchOptions::default()
                },
            )
            .unwrap()
        };
        let mode = |theta: Vec<f64>| ResponseBatchMode::ProfileAtTheta { theta, reml: true };

        let fit_serial = build(BatchParallelism::Serial)
            .fit_responses(&responses, mode(theta.clone()))
            .unwrap();
        let fit_parallel = build(BatchParallelism::Rayon)
            .fit_responses(&responses, mode(theta))
            .unwrap();

        assert_eq!(fit_serial.success_count(), 340);
        assert_fits_bitwise_identical(&fit_serial, &fit_parallel);
    }

    #[test]
    fn parallel_adaptive_is_identical_to_serial() {
        use mixeff_rs::model::batch::AdaptiveGroupingControl;

        let model = fitted_model();
        let responses = wide_response_matrix(&model, 12);
        let batch = LinearMixedModelBatch::from_model(&model).unwrap();

        let mode = |parallelism| ResponseBatchMode::OptimizeAdaptive {
            reml: true,
            control: BatchOptimizerControl {
                max_evaluations: 150,
                theta_tolerance: 1e-5,
                objective_tolerance: 1e-8,
                initial_step: Some(vec![0.25]),
                options: BatchOptions {
                    chunk_columns: 3,
                    parallelism,
                    ..BatchOptions::default()
                },
            },
            adaptive: AdaptiveGroupingControl::default(),
        };

        let fit_serial = batch
            .fit_responses(&responses, mode(BatchParallelism::Serial))
            .unwrap();
        let fit_parallel = batch
            .fit_responses(&responses, mode(BatchParallelism::Rayon))
            .unwrap();

        assert_fits_bitwise_identical(&fit_serial, &fit_parallel);
    }

    #[test]
    fn parallel_column_failures_keep_deterministic_diagnostics() {
        let model = fitted_model();
        let mut responses = wide_response_matrix(&model, 9);
        for row in 0..responses.nrows() {
            responses[(row, 2)] = 1.5; // constant
            responses[(row, 6)] = f64::NAN; // invalid
        }

        let batch = LinearMixedModelBatch::from_model(&model).unwrap();
        let mode = |parallelism| ResponseBatchMode::OptimizePerColumn {
            reml: true,
            warm_start: BatchWarmStart::TemplateTheta,
            control: BatchOptimizerControl {
                max_evaluations: 120,
                theta_tolerance: 1e-5,
                objective_tolerance: 1e-8,
                initial_step: Some(vec![0.25]),
                options: BatchOptions {
                    parallelism,
                    ..BatchOptions::default()
                },
            },
        };

        let fit_serial = batch
            .fit_responses(&responses, mode(BatchParallelism::Serial))
            .unwrap();
        let fit_parallel = batch
            .fit_responses(&responses, mode(BatchParallelism::Rayon))
            .unwrap();

        assert_eq!(fit_serial.status[2], ResponseFitStatus::ConstantResponse);
        assert_eq!(fit_serial.status[6], ResponseFitStatus::InvalidResponse);
        assert_fits_bitwise_identical(&fit_serial, &fit_parallel);
    }
}

mod streaming {
    use super::*;
    use mixeff_rs::error::{MixedModelError, Result};
    use mixeff_rs::model::batch::{
        ResponseBatchSink, ResponseColumnRow, ResponseDiagnosticReason, SinkFlow,
    };

    #[derive(Debug, Clone, PartialEq)]
    struct OwnedRow {
        column: usize,
        status: ResponseFitStatus,
        beta: Option<Vec<f64>>,
        sigma: f64,
        objective: f64,
        theta: Option<Vec<f64>>,
        diagnostic: Option<ResponseDiagnosticReason>,
    }

    #[derive(Default)]
    struct RecordingSink {
        rows: Vec<OwnedRow>,
        stop_after: Option<usize>,
        error_after: Option<usize>,
    }

    impl ResponseBatchSink for RecordingSink {
        fn on_column(&mut self, row: ResponseColumnRow<'_>) -> Result<SinkFlow> {
            self.rows.push(OwnedRow {
                column: row.column,
                status: row.status,
                beta: row.beta.map(|b| b.to_vec()),
                sigma: row.sigma,
                objective: row.objective,
                theta: row.theta.map(|t| t.to_vec()),
                diagnostic: row.diagnostic.map(|d| d.reason),
            });
            if self.error_after.is_some_and(|n| self.rows.len() >= n) {
                return Err(MixedModelError::InvalidArgument(
                    "sink exploded".to_string(),
                ));
            }
            if self.stop_after.is_some_and(|n| self.rows.len() >= n) {
                return Ok(SinkFlow::Stop);
            }
            Ok(SinkFlow::Continue)
        }
    }

    fn per_column_mode() -> ResponseBatchMode {
        ResponseBatchMode::OptimizePerColumn {
            reml: true,
            warm_start: BatchWarmStart::TemplateTheta,
            control: BatchOptimizerControl {
                max_evaluations: 120,
                theta_tolerance: 1e-5,
                objective_tolerance: 1e-8,
                initial_step: Some(vec![0.25]),
                options: BatchOptions::default(),
            },
        }
    }

    #[test]
    fn streaming_per_column_matches_materialized_rows() {
        let model = fitted_model();
        let responses = response_matrix(&model);
        let batch = LinearMixedModelBatch::from_model(&model).unwrap();

        let materialized = batch.fit_responses(&responses, per_column_mode()).unwrap();
        let mut sink = RecordingSink::default();
        let streamed = batch
            .fit_responses_streaming(&responses, per_column_mode(), &mut sink)
            .unwrap();

        // One row per column, ascending, values identical to materialized.
        let columns: Vec<usize> = sink.rows.iter().map(|row| row.column).collect();
        assert_eq!(columns, vec![0, 1, 2]);
        for row in &sink.rows {
            let col = row.column;
            assert_eq!(row.status, materialized.status[col]);
            assert_eq!(
                row.sigma.to_bits(),
                materialized.sigma[col].to_bits(),
                "sigma mismatch in column {col}"
            );
            assert_eq!(
                row.objective.to_bits(),
                materialized.objective[col].to_bits()
            );
            let beta = row.beta.as_ref().expect("fitted column emits beta");
            for (r, value) in beta.iter().enumerate() {
                assert_eq!(value.to_bits(), materialized.beta[(r, col)].to_bits());
            }
            let theta = row.theta.as_ref().expect("fitted column emits theta");
            let expected = match &materialized.theta {
                ThetaBatch::PerColumn(matrix) => matrix.column(col).as_slice().to_vec(),
                other => panic!("expected per-column theta, got {other:?}"),
            };
            assert_eq!(theta, &expected);
        }
        // The streaming call also returns the same materialized result.
        assert_eq!(streamed.status, materialized.status);
    }

    #[test]
    fn streaming_emits_classification_failures_then_fitted_columns() {
        let model = fitted_model();
        let y = model.response();
        let mut responses = DMatrix::zeros(y.len(), 3);
        for row in 0..y.len() {
            responses[(row, 0)] = y[row];
            responses[(row, 1)] = 4.2; // constant
            responses[(row, 2)] = 0.5 * y[row] + 1.0;
        }

        let batch = LinearMixedModelBatch::from_model(&model).unwrap();
        let mut sink = RecordingSink::default();
        let fit = batch
            .fit_responses_streaming(
                &responses,
                ResponseBatchMode::ProfileAtTheta {
                    theta: model.theta(),
                    reml: true,
                },
                &mut sink,
            )
            .unwrap();

        let columns: Vec<usize> = sink.rows.iter().map(|row| row.column).collect();
        assert_eq!(
            columns,
            vec![1, 0, 2],
            "failures first, then fitted columns"
        );
        assert_eq!(sink.rows[0].status, ResponseFitStatus::ConstantResponse);
        assert_eq!(
            sink.rows[0].diagnostic,
            Some(ResponseDiagnosticReason::ConstantResponse)
        );
        assert!(sink.rows[0].beta.is_none());
        assert_eq!(fit.status[1], ResponseFitStatus::ConstantResponse);
        assert_eq!(fit.status[0], ResponseFitStatus::Success);
    }

    #[test]
    fn sink_stop_marks_unemitted_columns_unsupported() {
        let model = fitted_model();
        let responses = response_matrix(&model);
        let batch = LinearMixedModelBatch::from_model(&model).unwrap();

        let mut sink = RecordingSink {
            stop_after: Some(2),
            ..RecordingSink::default()
        };
        let fit = batch
            .fit_responses_streaming(&responses, per_column_mode(), &mut sink)
            .unwrap();

        assert_eq!(sink.rows.len(), 2);
        assert_eq!(fit.status[0], ResponseFitStatus::Success);
        assert_eq!(fit.status[1], ResponseFitStatus::Success);
        assert_eq!(fit.status[2], ResponseFitStatus::Unsupported);
        assert!(fit
            .diagnostics
            .iter()
            .any(|d| d.column == 2 && d.reason == ResponseDiagnosticReason::SinkStopped));
        // Unemitted columns are never populated.
        assert!(fit.sigma[2].is_nan());
        match &fit.theta {
            ThetaBatch::PerColumn(theta) => assert!(theta[(0, 2)].is_nan()),
            other => panic!("expected per-column theta, got {other:?}"),
        }
    }

    #[test]
    fn sink_error_propagates() {
        let model = fitted_model();
        let responses = response_matrix(&model);
        let batch = LinearMixedModelBatch::from_model(&model).unwrap();

        let mut sink = RecordingSink {
            error_after: Some(1),
            ..RecordingSink::default()
        };
        let result = batch.fit_responses_streaming(&responses, per_column_mode(), &mut sink);
        assert!(matches!(result, Err(MixedModelError::InvalidArgument(_))));
    }
}

mod adaptive {
    use super::*;
    use mixeff_rs::model::batch::{AdaptiveGroupingControl, ResponseDiagnosticReason};

    fn control() -> BatchOptimizerControl {
        BatchOptimizerControl {
            max_evaluations: 200,
            theta_tolerance: 1e-5,
            objective_tolerance: 1e-8,
            initial_step: Some(vec![0.25]),
            options: BatchOptions::default(),
        }
    }

    fn adaptive_mode(adaptive: AdaptiveGroupingControl) -> ResponseBatchMode {
        ResponseBatchMode::OptimizeAdaptive {
            reml: true,
            control: control(),
            adaptive,
        }
    }

    /// Two theta clusters: columns 0-2 are affine transforms of `y` (same
    /// variance ratio, so identical theta); columns 3-4 add deterministic
    /// row noise (inflated residual variance, so much smaller theta).
    fn two_cluster_responses(model: &LinearMixedModel) -> DMatrix<f64> {
        let y = model.response();
        let noise = |row: usize| ((row * 37 % 17) as f64 - 8.0) * 0.15;
        DMatrix::from_fn(y.len(), 5, |row, col| match col {
            0 => y[row],
            1 => 1.1 * y[row] + 0.5,
            2 => 0.9 * y[row] - 0.2,
            3 => y[row] + noise(row),
            _ => 1.05 * (y[row] + noise(row)) + 1.0,
        })
    }

    #[test]
    fn adaptive_discovers_theta_clusters() {
        let model = fitted_model();
        let responses = two_cluster_responses(&model);
        let batch = LinearMixedModelBatch::from_model(&model).unwrap();

        let fit = batch
            .fit_responses(
                &responses,
                adaptive_mode(AdaptiveGroupingControl::default()),
            )
            .unwrap();

        assert!(fit.status.iter().all(|status| matches!(
            status,
            ResponseFitStatus::Success | ResponseFitStatus::Boundary
        )));
        let ThetaBatch::Adaptive {
            group_for_column, ..
        } = &fit.theta
        else {
            panic!("expected adaptive theta, got {:?}", fit.theta);
        };
        assert_eq!(group_for_column[0], group_for_column[1]);
        assert_eq!(group_for_column[0], group_for_column[2]);
        assert_eq!(group_for_column[3], group_for_column[4]);
        assert_ne!(group_for_column[0], group_for_column[3]);
    }

    #[test]
    fn adaptive_matches_per_column_within_refinement_tolerance() {
        let model = fitted_model();
        let responses = two_cluster_responses(&model);
        let batch = LinearMixedModelBatch::from_model(&model).unwrap();

        let refinement_objective_tolerance = 0.1;
        let fit = batch
            .fit_responses(
                &responses,
                adaptive_mode(AdaptiveGroupingControl {
                    refinement_objective_tolerance,
                    ..AdaptiveGroupingControl::default()
                }),
            )
            .unwrap();
        let per_column = batch
            .fit_responses(
                &responses,
                ResponseBatchMode::OptimizePerColumn {
                    reml: true,
                    warm_start: BatchWarmStart::TemplateTheta,
                    control: control(),
                },
            )
            .unwrap();

        for col in 0..responses.ncols() {
            let excess = fit.objective[col] - per_column.objective[col];
            assert!(
                // The contract guarantees closeness to the *probe* objective;
                // comparing to the full per-column optimum needs extra slack
                // for whatever the 60-eval probe left on the table.
                excess <= refinement_objective_tolerance + 1e-2,
                "column {col}: adaptive objective {} vs per-column {}",
                fit.objective[col],
                per_column.objective[col]
            );
        }
    }

    #[test]
    fn adaptive_is_deterministic() {
        let model = fitted_model();
        let responses = two_cluster_responses(&model);
        let batch = LinearMixedModelBatch::from_model(&model).unwrap();

        let first = batch
            .fit_responses(
                &responses,
                adaptive_mode(AdaptiveGroupingControl::default()),
            )
            .unwrap();
        let second = batch
            .fit_responses(
                &responses,
                adaptive_mode(AdaptiveGroupingControl::default()),
            )
            .unwrap();

        assert_eq!(first, second);
    }

    #[test]
    fn adaptive_refines_columns_a_bad_group_would_hurt() {
        let model = fitted_model();
        let responses = two_cluster_responses(&model);
        let batch = LinearMixedModelBatch::from_model(&model).unwrap();

        // Force every column into one group, with a strict refinement
        // tolerance: the minority cluster must be refined individually.
        let refinement_objective_tolerance = 1e-6;
        let fit = batch
            .fit_responses(
                &responses,
                adaptive_mode(AdaptiveGroupingControl {
                    theta_similarity_tolerance: 1e9,
                    refinement_objective_tolerance,
                    ..AdaptiveGroupingControl::default()
                }),
            )
            .unwrap();

        let ThetaBatch::Adaptive {
            group_for_column,
            refined_columns,
            ..
        } = &fit.theta
        else {
            panic!("expected adaptive theta, got {:?}", fit.theta);
        };
        assert!(group_for_column.iter().all(|&group| group == 0));
        assert!(
            !refined_columns.is_empty(),
            "a single forced group must trigger refinement"
        );
        assert!(fit
            .diagnostics
            .iter()
            .any(|d| d.reason == ResponseDiagnosticReason::AdaptiveRefinement));
        assert!(fit.status.iter().all(|status| matches!(
            status,
            ResponseFitStatus::Success | ResponseFitStatus::Boundary
        )));

        // Refined or not, every column stays near its per-column optimum.
        let per_column = batch
            .fit_responses(
                &responses,
                ResponseBatchMode::OptimizePerColumn {
                    reml: true,
                    warm_start: BatchWarmStart::TemplateTheta,
                    control: control(),
                },
            )
            .unwrap();
        for col in 0..responses.ncols() {
            assert!(
                fit.objective[col] - per_column.objective[col] <= 1e-2,
                "column {col}: adaptive {} vs per-column {}",
                fit.objective[col],
                per_column.objective[col]
            );
        }
    }

    #[test]
    fn adaptive_failures_stay_column_local() {
        let model = fitted_model();
        let mut responses = two_cluster_responses(&model);
        for row in 0..responses.nrows() {
            responses[(row, 1)] = 7.7; // constant
        }

        let batch = LinearMixedModelBatch::from_model(&model).unwrap();
        let fit = batch
            .fit_responses(
                &responses,
                adaptive_mode(AdaptiveGroupingControl::default()),
            )
            .unwrap();

        assert_eq!(fit.status[1], ResponseFitStatus::ConstantResponse);
        let ThetaBatch::Adaptive {
            group_for_column, ..
        } = &fit.theta
        else {
            panic!("expected adaptive theta, got {:?}", fit.theta);
        };
        assert_eq!(group_for_column[1], usize::MAX);
        for col in [0usize, 2, 3, 4] {
            assert!(matches!(
                fit.status[col],
                ResponseFitStatus::Success | ResponseFitStatus::Boundary
            ));
            assert!(fit.sigma[col].is_finite());
        }
    }
}
