//! Response-matrix batch fitting for linear mixed models.
//!
//! The batch API treats the columns of an `n x q` response matrix as
//! independent LMM responses that share one fixed/random-effect structure.
//! It does not estimate cross-response covariance.

use nalgebra::{DMatrix, DVector};

use crate::error::{MixedModelError, Result};
use crate::model::kernel::{select_response_columns, LmmObjectiveKernel, LmmWorkspace};
use crate::model::linear::{LinearMixedModel, ResponseMatrixProfile};

/// Cached invariant structure for fitting independent LMM response columns.
///
/// Construction extracts a lean profiling kernel from the template model
/// (dims, fixed design, random-effect structure, θ bounds and mapping,
/// structural blocks); the template model itself is not retained.
#[derive(Debug, Clone)]
pub struct LinearMixedModelBatch {
    kernel: LmmObjectiveKernel,
    options: BatchOptions,
}

/// Batch execution controls that are independent of optimizer tolerances.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchOptions {
    /// Number of response columns profiled at a time.
    pub chunk_columns: usize,
    /// Optional guardrail for callers that want to stop after many failures.
    pub max_failures: Option<usize>,
    /// Execution strategy for independent chunk/column work. The default is
    /// serial; parallel execution is deterministic (identical results and
    /// diagnostic ordering) but must be requested explicitly.
    pub parallelism: BatchParallelism,
}

impl Default for BatchOptions {
    fn default() -> Self {
        Self {
            chunk_columns: 64,
            max_failures: None,
            parallelism: BatchParallelism::Serial,
        }
    }
}

/// Execution strategy for independent batch work units.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum BatchParallelism {
    /// Process chunks and columns sequentially (the default).
    #[default]
    Serial,
    /// Process independent chunks/columns on a rayon thread pool. Requires
    /// the `rayon` cargo feature; requesting it without the feature is a
    /// typed error rather than a silent serial fallback. Results and
    /// diagnostic ordering are identical to serial execution.
    Rayon,
}

impl BatchParallelism {
    fn is_parallel(self) -> bool {
        matches!(self, BatchParallelism::Rayon)
    }

    fn validate(self) -> Result<()> {
        if self.is_parallel() && !cfg!(feature = "rayon") {
            return Err(MixedModelError::InvalidArgument(
                "BatchParallelism::Rayon requires building with the `rayon` cargo feature"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

/// Optimizer controls used by shared, grouped, and per-column theta modes.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchOptimizerControl {
    pub max_evaluations: i64,
    pub objective_tolerance: f64,
    pub theta_tolerance: f64,
    pub initial_step: Option<Vec<f64>>,
    pub options: BatchOptions,
}

impl Default for BatchOptimizerControl {
    fn default() -> Self {
        Self {
            max_evaluations: 1_000,
            objective_tolerance: 1e-8,
            theta_tolerance: 1e-5,
            initial_step: None,
            options: BatchOptions::default(),
        }
    }
}

/// Warm-start policy for per-column theta optimization.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum BatchWarmStart {
    /// Start each column at the template model's current theta.
    TemplateTheta,
    /// First optimize one shared theta and use it for every column.
    SharedTheta,
    /// Start every column at the same caller-provided theta.
    Fixed(Vec<f64>),
    /// Start column `j` at column `j` of an `ntheta x q` matrix.
    Provided(DMatrix<f64>),
}

/// Grouping definition for grouped shared-theta optimization.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum BatchThetaGrouping {
    /// `groups[j]` is the group id for response column `j`.
    ColumnGroups(Vec<usize>),
}

/// Batch fitting modes for independent response columns.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ResponseBatchMode {
    /// Profile every response column at a caller-supplied theta.
    ProfileAtTheta { theta: Vec<f64>, reml: bool },
    /// Optimize one theta against the aggregate objective across columns.
    OptimizeSharedTheta {
        reml: bool,
        control: BatchOptimizerControl,
    },
    /// Optimize theta independently for each response column.
    OptimizePerColumn {
        reml: bool,
        warm_start: BatchWarmStart,
        control: BatchOptimizerControl,
    },
    /// Optimize one theta per caller-supplied column group.
    OptimizeGrouped {
        reml: bool,
        grouping: BatchThetaGrouping,
        control: BatchOptimizerControl,
    },
    /// Discover theta groups adaptively: probe every column cheaply, group
    /// columns with similar probe thetas, optimize one theta per group, and
    /// individually refine only the columns the group theta fails (see
    /// [`AdaptiveGroupingControl`]). Every regroup/refine decision is
    /// deterministic and recorded in the result.
    OptimizeAdaptive {
        reml: bool,
        control: BatchOptimizerControl,
        adaptive: AdaptiveGroupingControl,
    },
}

/// Controls for adaptive theta grouping and refinement.
///
/// The adaptive mode guarantees that every column's final objective is
/// within [`refinement_objective_tolerance`] of its cheap per-column probe
/// objective: columns whose group theta is worse than that are refined
/// individually at full budget (warm-started from their probe theta, so
/// refinement can only improve on the probe).
///
/// [`refinement_objective_tolerance`]: AdaptiveGroupingControl::refinement_objective_tolerance
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveGroupingControl {
    /// Optimizer evaluation budget for the cheap per-column probe pass.
    pub probe_max_evaluations: i64,
    /// Cluster radius: a column joins the first group whose representative
    /// probe theta agrees with its own within this infinity-norm distance.
    pub theta_similarity_tolerance: f64,
    /// A column is refined individually when its objective at the group
    /// theta exceeds its probe objective by more than this.
    pub refinement_objective_tolerance: f64,
}

impl Default for AdaptiveGroupingControl {
    fn default() -> Self {
        Self {
            probe_max_evaluations: 60,
            theta_similarity_tolerance: 0.05,
            refinement_objective_tolerance: 0.1,
        }
    }
}

impl AdaptiveGroupingControl {
    fn validate(&self) -> Result<()> {
        if self.probe_max_evaluations <= 0 {
            return Err(MixedModelError::InvalidArgument(
                "adaptive probe_max_evaluations must be positive".to_string(),
            ));
        }
        if !self.theta_similarity_tolerance.is_finite() || self.theta_similarity_tolerance < 0.0 {
            return Err(MixedModelError::InvalidArgument(
                "adaptive theta_similarity_tolerance must be finite and non-negative".to_string(),
            ));
        }
        if !self.refinement_objective_tolerance.is_finite()
            || self.refinement_objective_tolerance < 0.0
        {
            return Err(MixedModelError::InvalidArgument(
                "adaptive refinement_objective_tolerance must be finite and non-negative"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

/// Theta values returned by a batch fit.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ThetaBatch {
    Shared(Vec<f64>),
    PerColumn(DMatrix<f64>),
    Grouped {
        theta: DMatrix<f64>,
        group_for_column: Vec<usize>,
    },
    /// Result of adaptive grouping: the final `ntheta x q` theta per column,
    /// the discovered group of every column (`usize::MAX` for columns that
    /// were never successfully probed — validation or probe failures), and
    /// the ascending list of columns that were individually refined away
    /// from their group.
    Adaptive {
        theta: DMatrix<f64>,
        group_for_column: Vec<usize>,
        refined_columns: Vec<usize>,
    },
}

/// Per-column fit status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ResponseFitStatus {
    Success,
    Boundary,
    InvalidResponse,
    ConstantResponse,
    OptimizerFailed,
    Unsupported,
}

/// Structured diagnostic reason for a response-column outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ResponseDiagnosticReason {
    NonFiniteResponse,
    ConstantResponse,
    BoundaryTheta,
    OptimizerFailed,
    UnsupportedMode,
    /// A streaming sink requested stop before this column was fitted.
    SinkStopped,
    /// Adaptive grouping refined this column individually because the group
    /// theta missed its probe objective by more than the refinement
    /// tolerance.
    AdaptiveRefinement,
}

/// Flow control returned by a streaming sink after each emitted column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SinkFlow {
    /// Keep fitting and emitting columns.
    Continue,
    /// Stop: no further columns are fitted or emitted; unprocessed columns
    /// are marked [`ResponseFitStatus::Unsupported`] with a
    /// [`ResponseDiagnosticReason::SinkStopped`] diagnostic.
    Stop,
}

/// One finalized response column, borrowed from the batch engine.
#[derive(Debug, Clone, Copy)]
pub struct ResponseColumnRow<'a> {
    /// Index of the response column in the caller's response matrix.
    pub column: usize,
    /// Final status for this column.
    pub status: ResponseFitStatus,
    /// Fixed-effects estimates (length `p`); `None` for unfitted columns.
    pub beta: Option<&'a [f64]>,
    /// Profiled residual scale; NaN for unfitted columns.
    pub sigma: f64,
    /// Penalized weighted residual sum of squares; NaN for unfitted columns.
    pub pwrss: f64,
    /// Profiled objective; NaN for unfitted columns.
    pub objective: f64,
    /// θ at which the column was profiled; `None` for unfitted columns.
    pub theta: Option<&'a [f64]>,
    /// Column-local diagnostic, when one was recorded.
    pub diagnostic: Option<&'a ResponseColumnDiagnostic>,
}

/// Streaming consumer of per-column batch results.
///
/// Every column is emitted exactly once, in the engine's deterministic
/// processing order: validation failures during classification, then fitted
/// columns (ascending columns for profile/per-column modes, group order for
/// grouped mode, probe failures then group-creation order for adaptive
/// mode). Serial per-column execution emits each column as soon as it is
/// optimized; parallel execution computes all columns first and then emits
/// in the same order. Returning [`SinkFlow::Stop`] or an error stops the
/// batch; the materialized result then marks every unemitted column as
/// unsupported, so a column is populated in the result if and only if the
/// sink saw it.
pub trait ResponseBatchSink {
    /// Consume one finalized column.
    fn on_column(&mut self, row: ResponseColumnRow<'_>) -> Result<SinkFlow>;
}

/// Internal adapter that tracks stop state for an optional sink.
struct SinkDriver<'s> {
    sink: Option<&'s mut dyn ResponseBatchSink>,
    stopped: bool,
}

impl SinkDriver<'_> {
    fn new(sink: Option<&mut dyn ResponseBatchSink>) -> SinkDriver<'_> {
        SinkDriver {
            sink,
            stopped: false,
        }
    }

    fn emit(&mut self, row: ResponseColumnRow<'_>) -> Result<()> {
        debug_assert!(!self.stopped, "emit called after sink stop");
        if let Some(sink) = self.sink.as_deref_mut() {
            if sink.on_column(row)? == SinkFlow::Stop {
                self.stopped = true;
            }
        }
        Ok(())
    }
}

/// Column-local diagnostic emitted by the batch engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResponseColumnDiagnostic {
    pub column: usize,
    pub reason: ResponseDiagnosticReason,
    pub message: String,
}

/// Column-local outcome of one per-column θ optimization.
enum ColumnFit {
    OptimizerFailed,
    Fitted {
        theta: Vec<f64>,
        profile: ResponseMatrixProfile,
        boundary: bool,
    },
}

/// Stable column-major batch result.
#[derive(Debug, Clone, PartialEq)]
pub struct ResponseBatchFit {
    pub beta: DMatrix<f64>,
    pub sigma: DVector<f64>,
    pub pwrss: DVector<f64>,
    pub objective: DVector<f64>,
    pub theta: ThetaBatch,
    pub status: Vec<ResponseFitStatus>,
    pub diagnostics: Vec<ResponseColumnDiagnostic>,
}

impl ResponseBatchFit {
    pub fn success_count(&self) -> usize {
        self.status
            .iter()
            .filter(|status| {
                matches!(
                    status,
                    ResponseFitStatus::Success | ResponseFitStatus::Boundary
                )
            })
            .count()
    }
}

impl LinearMixedModelBatch {
    /// Cache the invariant structure from a template LMM.
    pub fn from_model(model: &LinearMixedModel) -> Result<Self> {
        Self::from_model_with_options(model, BatchOptions::default())
    }

    /// Cache the invariant structure from a template LMM with explicit options.
    pub fn from_model_with_options(
        model: &LinearMixedModel,
        options: BatchOptions,
    ) -> Result<Self> {
        if options.chunk_columns == 0 {
            return Err(MixedModelError::InvalidArgument(
                "batch chunk_columns must be positive".to_string(),
            ));
        }
        options.parallelism.validate()?;
        Ok(Self {
            kernel: LmmObjectiveKernel::from_model(model)?,
            options,
        })
    }

    pub fn options(&self) -> &BatchOptions {
        &self.options
    }

    pub fn fit_responses(
        &self,
        responses: &DMatrix<f64>,
        mode: ResponseBatchMode,
    ) -> Result<ResponseBatchFit> {
        self.fit_responses_impl(responses, mode, None)
    }

    /// Fit response columns while streaming each finalized column to `sink`.
    ///
    /// The materialized result is still returned; it contains exactly the
    /// columns the sink saw (see [`ResponseBatchSink`] for the emission
    /// order and stop semantics). Sink errors abort the batch and propagate.
    pub fn fit_responses_streaming(
        &self,
        responses: &DMatrix<f64>,
        mode: ResponseBatchMode,
        sink: &mut dyn ResponseBatchSink,
    ) -> Result<ResponseBatchFit> {
        self.fit_responses_impl(responses, mode, Some(sink))
    }

    fn fit_responses_impl(
        &self,
        responses: &DMatrix<f64>,
        mode: ResponseBatchMode,
        sink: Option<&mut dyn ResponseBatchSink>,
    ) -> Result<ResponseBatchFit> {
        self.validate_response_rows(responses)?;
        let mut driver = SinkDriver::new(sink);
        match mode {
            ResponseBatchMode::ProfileAtTheta { theta, reml } => {
                self.fit_profile_at_theta(responses, &theta, reml, &self.options, &mut driver)
            }
            ResponseBatchMode::OptimizeSharedTheta { reml, control } => {
                self.fit_optimize_shared_theta(responses, reml, &control, &mut driver)
            }
            ResponseBatchMode::OptimizePerColumn {
                reml,
                warm_start,
                control,
            } => self.fit_optimize_per_column(responses, reml, warm_start, &control, &mut driver),
            ResponseBatchMode::OptimizeGrouped {
                reml,
                grouping,
                control,
            } => self.fit_optimize_grouped(responses, reml, grouping, &control, &mut driver),
            ResponseBatchMode::OptimizeAdaptive {
                reml,
                control,
                adaptive,
            } => self.fit_optimize_adaptive(responses, reml, &control, &adaptive, &mut driver),
        }
    }

    fn validate_response_rows(&self, responses: &DMatrix<f64>) -> Result<()> {
        if responses.nrows() != self.kernel.n() {
            return Err(MixedModelError::DimensionMismatch(format!(
                "response matrix has {} rows, expected {}",
                responses.nrows(),
                self.kernel.n()
            )));
        }
        Ok(())
    }

    fn fit_profile_at_theta(
        &self,
        responses: &DMatrix<f64>,
        theta: &[f64],
        reml: bool,
        options: &BatchOptions,
        driver: &mut SinkDriver<'_>,
    ) -> Result<ResponseBatchFit> {
        self.kernel.validate_theta(theta)?;
        let mut result = self.empty_result(responses.ncols(), ThetaBatch::Shared(theta.to_vec()));
        let valid_columns = self.classify_responses(responses, &mut result, options, driver)?;
        if valid_columns.is_empty() {
            return Ok(result);
        }
        if driver.stopped {
            self.mark_sink_stopped(&valid_columns, &mut result);
            return Ok(result);
        }

        let boundary = self.kernel.theta_on_boundary(theta);
        let mut workspace = self.kernel.workspace();
        let profile = workspace.profile_columns_at_theta(
            theta,
            responses,
            reml,
            &valid_columns,
            options.chunk_columns,
            options.parallelism.is_parallel(),
        )?;
        self.scatter_profile(
            &profile,
            &valid_columns,
            &mut result,
            boundary,
            theta,
            driver,
        )?;
        Ok(result)
    }

    fn fit_optimize_shared_theta(
        &self,
        responses: &DMatrix<f64>,
        reml: bool,
        control: &BatchOptimizerControl,
        driver: &mut SinkDriver<'_>,
    ) -> Result<ResponseBatchFit> {
        self.validate_control(control)?;
        let mut result = self.empty_result(
            responses.ncols(),
            ThetaBatch::Shared(self.kernel.template_theta().to_vec()),
        );
        let valid_columns =
            self.classify_responses(responses, &mut result, &control.options, driver)?;
        if valid_columns.is_empty() {
            return Ok(result);
        }
        if driver.stopped {
            self.mark_sink_stopped(&valid_columns, &mut result);
            return Ok(result);
        }

        let parallel = control.options.parallelism.is_parallel();
        let mut workspace = self.kernel.workspace();
        let initial = self.kernel.projected_theta(self.kernel.template_theta())?;
        let outcome =
            self.optimize_theta(&mut workspace, initial, control, |workspace, theta| {
                Ok(workspace
                    .profile_columns_at_theta(
                        theta,
                        responses,
                        reml,
                        &valid_columns,
                        control.options.chunk_columns,
                        parallel,
                    )
                    .map(|profile| profile.total_objective)
                    .unwrap_or(f64::INFINITY))
            })?;
        let mut theta = outcome.best_theta;
        LinearMixedModel::rectify_theta_columns(
            &mut theta,
            self.kernel.parmap(),
            self.kernel.reterm_count(),
        );
        let boundary = self.kernel.theta_on_boundary(&theta);
        let profile = workspace.profile_columns_at_theta(
            &theta,
            responses,
            reml,
            &valid_columns,
            control.options.chunk_columns,
            parallel,
        )?;
        self.scatter_profile(
            &profile,
            &valid_columns,
            &mut result,
            boundary,
            &theta,
            driver,
        )?;
        result.theta = ThetaBatch::Shared(theta);
        Ok(result)
    }

    fn fit_optimize_per_column(
        &self,
        responses: &DMatrix<f64>,
        reml: bool,
        warm_start: BatchWarmStart,
        control: &BatchOptimizerControl,
        driver: &mut SinkDriver<'_>,
    ) -> Result<ResponseBatchFit> {
        self.validate_control(control)?;
        let q = responses.ncols();
        let ntheta = self.kernel.template_theta().len();
        let mut theta_out = DMatrix::from_element(ntheta, q, f64::NAN);
        let mut result = self.empty_result(q, ThetaBatch::PerColumn(theta_out.clone()));
        let valid_columns =
            self.classify_responses(responses, &mut result, &control.options, driver)?;
        if valid_columns.is_empty() {
            return Ok(result);
        }
        if driver.stopped {
            self.mark_sink_stopped(&valid_columns, &mut result);
            return Ok(result);
        }

        let shared_start = match warm_start {
            BatchWarmStart::SharedTheta => {
                // The pre-pass is internal: its columns are not emitted.
                let shared = self.fit_optimize_shared_theta(
                    responses,
                    reml,
                    control,
                    &mut SinkDriver::new(None),
                )?;
                match shared.theta {
                    ThetaBatch::Shared(theta) => Some(theta),
                    _ => None,
                }
            }
            _ => None,
        };

        if control.options.parallelism.is_parallel() && valid_columns.len() > 1 {
            let column_fits = self.optimize_columns(
                &valid_columns,
                responses,
                reml,
                &warm_start,
                shared_start.as_deref(),
                q,
                control,
            )?;
            for (index, (column, fit)) in column_fits.into_iter().enumerate() {
                if driver.stopped {
                    self.mark_sink_stopped(&valid_columns[index..], &mut result);
                    break;
                }
                self.finalize_column_fit(column, fit, ntheta, &mut theta_out, &mut result, driver)?;
            }
        } else {
            // The serial path interleaves compute and emission so a sink
            // stop also saves the remaining columns' optimization work.
            let mut workspace = self.kernel.workspace();
            for (index, &column) in valid_columns.iter().enumerate() {
                if driver.stopped {
                    self.mark_sink_stopped(&valid_columns[index..], &mut result);
                    break;
                }
                let fit = self.optimize_single_column(
                    &mut workspace,
                    column,
                    responses,
                    reml,
                    &warm_start,
                    shared_start.as_deref(),
                    q,
                    control,
                )?;
                self.finalize_column_fit(column, fit, ntheta, &mut theta_out, &mut result, driver)?;
            }
        }

        result.theta = ThetaBatch::PerColumn(theta_out);
        Ok(result)
    }

    /// Optimize θ independently for each listed column, serially or on the
    /// rayon pool depending on the control's parallelism; results are always
    /// returned in the order of `columns`.
    #[allow(clippy::too_many_arguments)]
    fn optimize_columns(
        &self,
        columns: &[usize],
        responses: &DMatrix<f64>,
        reml: bool,
        warm_start: &BatchWarmStart,
        shared_start: Option<&[f64]>,
        q: usize,
        control: &BatchOptimizerControl,
    ) -> Result<Vec<(usize, ColumnFit)>> {
        // A single column cannot amortize the thread-pool dispatch; keep it
        // on the serial path (results are identical either way).
        if control.options.parallelism.is_parallel() && columns.len() > 1 {
            #[cfg(feature = "rayon")]
            {
                use rayon::prelude::*;
                columns
                    .par_iter()
                    .map_init(
                        || self.kernel.workspace(),
                        |workspace, &column| {
                            self.optimize_single_column(
                                workspace,
                                column,
                                responses,
                                reml,
                                warm_start,
                                shared_start,
                                q,
                                control,
                            )
                            .map(|fit| (column, fit))
                        },
                    )
                    .collect::<Result<Vec<_>>>()
            }
            #[cfg(not(feature = "rayon"))]
            {
                unreachable!("validate_control rejects Rayon without the `rayon` feature")
            }
        } else {
            let mut workspace = self.kernel.workspace();
            let mut fits = Vec::with_capacity(columns.len());
            for &column in columns {
                let fit = self.optimize_single_column(
                    &mut workspace,
                    column,
                    responses,
                    reml,
                    warm_start,
                    shared_start,
                    q,
                    control,
                )?;
                fits.push((column, fit));
            }
            Ok(fits)
        }
    }

    /// Record one per-column outcome in the result and emit it to the sink.
    fn finalize_column_fit(
        &self,
        column: usize,
        fit: ColumnFit,
        ntheta: usize,
        theta_out: &mut DMatrix<f64>,
        result: &mut ResponseBatchFit,
        driver: &mut SinkDriver<'_>,
    ) -> Result<()> {
        match fit {
            ColumnFit::OptimizerFailed => {
                result.status[column] = ResponseFitStatus::OptimizerFailed;
                result.diagnostics.push(ResponseColumnDiagnostic {
                    column,
                    reason: ResponseDiagnosticReason::OptimizerFailed,
                    message: "per-column theta optimization failed".to_string(),
                });
                self.emit_failure_row(result, column, driver)
            }
            ColumnFit::Fitted {
                theta,
                profile,
                boundary,
            } => {
                for row in 0..ntheta {
                    theta_out[(row, column)] = theta[row];
                }
                self.scatter_profile(&profile, &[column], result, boundary, &theta, driver)
            }
        }
    }

    /// Optimize θ for one response column and profile it at the optimum.
    ///
    /// Shared by the serial and parallel per-column paths so both produce
    /// identical results; column-local optimizer failures are encoded in
    /// [`ColumnFit`] while validation errors propagate as `Err`.
    #[allow(clippy::too_many_arguments)]
    fn optimize_single_column(
        &self,
        workspace: &mut LmmWorkspace<'_>,
        column: usize,
        responses: &DMatrix<f64>,
        reml: bool,
        warm_start: &BatchWarmStart,
        shared_start: Option<&[f64]>,
        q: usize,
        control: &BatchOptimizerControl,
    ) -> Result<ColumnFit> {
        let initial = self.warm_start_for_column(warm_start, shared_start, column, q)?;
        let single = select_response_columns(responses, &[column]);
        let outcome = self.optimize_theta(workspace, initial, control, |workspace, theta| {
            Ok(workspace
                .profile_columns_at_theta(
                    theta,
                    &single,
                    reml,
                    &[0],
                    control.options.chunk_columns,
                    false,
                )
                .map(|profile| profile.total_objective)
                .unwrap_or(f64::INFINITY))
        });

        let Ok(outcome) = outcome else {
            return Ok(ColumnFit::OptimizerFailed);
        };

        let mut theta = outcome.best_theta;
        LinearMixedModel::rectify_theta_columns(
            &mut theta,
            self.kernel.parmap(),
            self.kernel.reterm_count(),
        );
        let boundary = self.kernel.theta_on_boundary(&theta);
        let profile = workspace.profile_columns_at_theta(
            &theta,
            &single,
            reml,
            &[0],
            control.options.chunk_columns,
            false,
        )?;
        Ok(ColumnFit::Fitted {
            theta,
            profile,
            boundary,
        })
    }

    fn fit_optimize_grouped(
        &self,
        responses: &DMatrix<f64>,
        reml: bool,
        grouping: BatchThetaGrouping,
        control: &BatchOptimizerControl,
        driver: &mut SinkDriver<'_>,
    ) -> Result<ResponseBatchFit> {
        self.validate_control(control)?;
        let q = responses.ncols();
        let BatchThetaGrouping::ColumnGroups(group_for_column) = grouping;
        if group_for_column.len() != q {
            return Err(MixedModelError::DimensionMismatch(format!(
                "theta grouping has {} entries, expected one for each of {q} response columns",
                group_for_column.len()
            )));
        }

        let ntheta = self.kernel.template_theta().len();
        let group_count = group_for_column
            .iter()
            .copied()
            .max()
            .map_or(0, |max| max + 1);
        let mut group_theta = DMatrix::from_element(ntheta, group_count, f64::NAN);
        let mut result = self.empty_result(
            q,
            ThetaBatch::Grouped {
                theta: group_theta.clone(),
                group_for_column: group_for_column.clone(),
            },
        );
        let valid_columns =
            self.classify_responses(responses, &mut result, &control.options, driver)?;
        if valid_columns.is_empty() {
            return Ok(result);
        }

        let parallel = control.options.parallelism.is_parallel();
        let mut workspace = self.kernel.workspace();
        for group in 0..group_count {
            if driver.stopped {
                // Columns in groups that have not run yet are exactly the
                // valid columns whose group id has not been processed.
                let remaining: Vec<usize> = valid_columns
                    .iter()
                    .copied()
                    .filter(|&column| group_for_column[column] >= group)
                    .collect();
                self.mark_sink_stopped(&remaining, &mut result);
                break;
            }
            let group_columns: Vec<usize> = valid_columns
                .iter()
                .copied()
                .filter(|&column| group_for_column[column] == group)
                .collect();
            if group_columns.is_empty() {
                continue;
            }
            let initial = self.kernel.projected_theta(self.kernel.template_theta())?;
            let outcome =
                self.optimize_theta(&mut workspace, initial, control, |workspace, theta| {
                    Ok(workspace
                        .profile_columns_at_theta(
                            theta,
                            responses,
                            reml,
                            &group_columns,
                            control.options.chunk_columns,
                            parallel,
                        )
                        .map(|profile| profile.total_objective)
                        .unwrap_or(f64::INFINITY))
                })?;
            let mut theta = outcome.best_theta;
            LinearMixedModel::rectify_theta_columns(
                &mut theta,
                self.kernel.parmap(),
                self.kernel.reterm_count(),
            );
            for row in 0..ntheta {
                group_theta[(row, group)] = theta[row];
            }
            let boundary = self.kernel.theta_on_boundary(&theta);
            let profile = workspace.profile_columns_at_theta(
                &theta,
                responses,
                reml,
                &group_columns,
                control.options.chunk_columns,
                parallel,
            )?;
            self.scatter_profile(
                &profile,
                &group_columns,
                &mut result,
                boundary,
                &theta,
                driver,
            )?;
        }

        result.theta = ThetaBatch::Grouped {
            theta: group_theta,
            group_for_column,
        };
        Ok(result)
    }

    /// Adaptive grouping: probe → cluster → group-optimize → refine.
    ///
    /// 1. Optimize one shared θ over all valid columns (aggregate).
    /// 2. Probe every column with a cheap per-column optimization
    ///    warm-started at the shared θ.
    /// 3. Cluster columns deterministically: ascending first-fit on the
    ///    probe θs within the similarity tolerance; the first member's probe
    ///    θ is the group representative.
    /// 4. Optimize one θ per multi-column group (warm-started at the
    ///    representative); singleton groups go straight to a full-budget
    ///    per-column optimization.
    /// 5. Refine any column whose objective at its group θ exceeds its probe
    ///    objective by more than the refinement tolerance, recording an
    ///    [`ResponseDiagnosticReason::AdaptiveRefinement`] diagnostic.
    ///
    /// Every column therefore ends within the refinement tolerance of its
    /// probe objective, and refinement (warm-started at the probe θ) can
    /// only improve on the probe.
    fn fit_optimize_adaptive(
        &self,
        responses: &DMatrix<f64>,
        reml: bool,
        control: &BatchOptimizerControl,
        adaptive: &AdaptiveGroupingControl,
        driver: &mut SinkDriver<'_>,
    ) -> Result<ResponseBatchFit> {
        self.validate_control(control)?;
        adaptive.validate()?;
        let q = responses.ncols();
        let ntheta = self.kernel.template_theta().len();
        let mut theta_out = DMatrix::from_element(ntheta, q, f64::NAN);
        let mut group_for_column = vec![usize::MAX; q];
        let mut refined_columns: Vec<usize> = Vec::new();
        let mut result = self.empty_result(
            q,
            ThetaBatch::Adaptive {
                theta: theta_out.clone(),
                group_for_column: group_for_column.clone(),
                refined_columns: refined_columns.clone(),
            },
        );
        let valid_columns =
            self.classify_responses(responses, &mut result, &control.options, driver)?;
        let finish = |theta_out: DMatrix<f64>,
                      group_for_column: Vec<usize>,
                      refined_columns: Vec<usize>,
                      mut result: ResponseBatchFit| {
            result.theta = ThetaBatch::Adaptive {
                theta: theta_out,
                group_for_column,
                refined_columns,
            };
            Ok(result)
        };
        if valid_columns.is_empty() {
            return finish(theta_out, group_for_column, refined_columns, result);
        }
        if driver.stopped {
            self.mark_sink_stopped(&valid_columns, &mut result);
            return finish(theta_out, group_for_column, refined_columns, result);
        }

        let parallel = control.options.parallelism.is_parallel();
        let mut workspace = self.kernel.workspace();

        // Phase 1: shared θ over all valid columns.
        let initial = self.kernel.projected_theta(self.kernel.template_theta())?;
        let outcome =
            self.optimize_theta(&mut workspace, initial, control, |workspace, theta| {
                Ok(workspace
                    .profile_columns_at_theta(
                        theta,
                        responses,
                        reml,
                        &valid_columns,
                        control.options.chunk_columns,
                        parallel,
                    )
                    .map(|profile| profile.total_objective)
                    .unwrap_or(f64::INFINITY))
            })?;
        let mut shared_theta = outcome.best_theta;
        LinearMixedModel::rectify_theta_columns(
            &mut shared_theta,
            self.kernel.parmap(),
            self.kernel.reterm_count(),
        );
        let shared_theta = self.kernel.projected_theta(&shared_theta)?;

        // Phase 2: cheap per-column probe warm-started at the shared θ.
        let probe_control = BatchOptimizerControl {
            max_evaluations: adaptive.probe_max_evaluations,
            ..control.clone()
        };
        let probe_fits = self.optimize_columns(
            &valid_columns,
            responses,
            reml,
            &BatchWarmStart::Fixed(shared_theta),
            None,
            q,
            &probe_control,
        )?;
        let mut probe: Vec<Option<(Vec<f64>, f64)>> = vec![None; q];
        for (column, fit) in probe_fits {
            match fit {
                ColumnFit::OptimizerFailed => {
                    // Uphold the sink contract: a probe failure after a sink
                    // stop is an unemitted column, so it reads SinkStopped
                    // rather than carrying an unemitted failure status.
                    if driver.stopped {
                        self.mark_sink_stopped(&[column], &mut result);
                        continue;
                    }
                    result.status[column] = ResponseFitStatus::OptimizerFailed;
                    result.diagnostics.push(ResponseColumnDiagnostic {
                        column,
                        reason: ResponseDiagnosticReason::OptimizerFailed,
                        message: "adaptive probe optimization failed".to_string(),
                    });
                    self.emit_failure_row(&result, column, driver)?;
                }
                ColumnFit::Fitted { theta, profile, .. } => {
                    probe[column] = Some((theta, profile.objectives[0]));
                }
            }
        }

        // Phase 3: deterministic ascending first-fit clustering on probe θs.
        let mut groups: Vec<Vec<usize>> = Vec::new();
        let mut representatives: Vec<Vec<f64>> = Vec::new();
        for &column in &valid_columns {
            let Some((probe_theta, _)) = probe[column].as_ref() else {
                continue;
            };
            let assigned = representatives.iter().position(|representative| {
                representative
                    .iter()
                    .zip(probe_theta.iter())
                    .all(|(a, b)| (a - b).abs() <= adaptive.theta_similarity_tolerance)
            });
            match assigned {
                Some(group) => groups[group].push(column),
                None => {
                    groups.push(vec![column]);
                    representatives.push(probe_theta.clone());
                }
            }
        }
        for (group, members) in groups.iter().enumerate() {
            for &column in members {
                group_for_column[column] = group;
            }
        }

        // Phases 4-5: per-group optimization with column-local refinement.
        // Deterministic processing order: groups in creation order, members
        // ascending within each group.
        let plan: Vec<usize> = groups.iter().flatten().copied().collect();
        let mut plan_position = 0usize;
        'groups: for (group, members) in groups.iter().enumerate() {
            if driver.stopped {
                self.mark_sink_stopped(&plan[plan_position..], &mut result);
                break;
            }

            if members.len() == 1 {
                // Singleton group: full-budget per-column optimization,
                // warm-started at the probe θ.
                let column = members[0];
                let (probe_theta, _) = probe[column].clone().expect("probed singleton");
                let fit = self
                    .optimize_columns(
                        &[column],
                        responses,
                        reml,
                        &BatchWarmStart::Fixed(probe_theta),
                        None,
                        q,
                        control,
                    )?
                    .pop()
                    .map(|(_, fit)| fit)
                    .expect("one column requested");
                self.finalize_column_fit(column, fit, ntheta, &mut theta_out, &mut result, driver)?;
                plan_position += 1;
                continue;
            }

            let initial = self.kernel.projected_theta(&representatives[group])?;
            let outcome =
                self.optimize_theta(&mut workspace, initial, control, |workspace, theta| {
                    Ok(workspace
                        .profile_columns_at_theta(
                            theta,
                            responses,
                            reml,
                            members,
                            control.options.chunk_columns,
                            parallel,
                        )
                        .map(|profile| profile.total_objective)
                        .unwrap_or(f64::INFINITY))
                })?;
            let mut group_theta = outcome.best_theta;
            LinearMixedModel::rectify_theta_columns(
                &mut group_theta,
                self.kernel.parmap(),
                self.kernel.reterm_count(),
            );
            let group_boundary = self.kernel.theta_on_boundary(&group_theta);
            let group_profile = workspace.profile_columns_at_theta(
                &group_theta,
                responses,
                reml,
                members,
                control.options.chunk_columns,
                parallel,
            )?;

            for (local, &column) in members.iter().enumerate() {
                if driver.stopped {
                    self.mark_sink_stopped(&plan[plan_position..], &mut result);
                    break 'groups;
                }
                let (probe_theta, probe_objective) =
                    probe[column].clone().expect("probed group member");
                let excess = group_profile.objectives[local] - probe_objective;
                if excess > adaptive.refinement_objective_tolerance {
                    result.diagnostics.push(ResponseColumnDiagnostic {
                        column,
                        reason: ResponseDiagnosticReason::AdaptiveRefinement,
                        message: format!(
                            "group {group} theta objective exceeded probe objective by \
                             {excess:.6e}; column refined individually"
                        ),
                    });
                    let fit = self
                        .optimize_columns(
                            &[column],
                            responses,
                            reml,
                            &BatchWarmStart::Fixed(probe_theta),
                            None,
                            q,
                            control,
                        )?
                        .pop()
                        .map(|(_, fit)| fit)
                        .expect("one column requested");
                    if matches!(fit, ColumnFit::Fitted { .. }) {
                        refined_columns.push(column);
                    }
                    self.finalize_column_fit(
                        column,
                        fit,
                        ntheta,
                        &mut theta_out,
                        &mut result,
                        driver,
                    )?;
                } else {
                    for row in 0..ntheta {
                        theta_out[(row, column)] = group_theta[row];
                    }
                    let single = single_column_profile(&group_profile, local);
                    self.scatter_profile(
                        &single,
                        &[column],
                        &mut result,
                        group_boundary,
                        &group_theta,
                        driver,
                    )?;
                }
                plan_position += 1;
            }
        }

        finish(theta_out, group_for_column, refined_columns, result)
    }

    fn validate_control(&self, control: &BatchOptimizerControl) -> Result<()> {
        if control.max_evaluations <= 0 {
            return Err(MixedModelError::InvalidArgument(
                "batch optimizer max_evaluations must be positive".to_string(),
            ));
        }
        if control.theta_tolerance <= 0.0 || !control.theta_tolerance.is_finite() {
            return Err(MixedModelError::InvalidArgument(
                "batch optimizer theta_tolerance must be finite and positive".to_string(),
            ));
        }
        if control.objective_tolerance < 0.0 || !control.objective_tolerance.is_finite() {
            return Err(MixedModelError::InvalidArgument(
                "batch optimizer objective_tolerance must be finite and non-negative".to_string(),
            ));
        }
        if control.options.chunk_columns == 0 {
            return Err(MixedModelError::InvalidArgument(
                "batch chunk_columns must be positive".to_string(),
            ));
        }
        control.options.parallelism.validate()?;
        if let Some(step) = &control.initial_step {
            if step.len() != self.kernel.template_theta().len() {
                return Err(MixedModelError::DimensionMismatch(format!(
                    "batch optimizer initial_step has length {}, expected {}",
                    step.len(),
                    self.kernel.template_theta().len()
                )));
            }
        }
        Ok(())
    }

    fn warm_start_for_column(
        &self,
        warm_start: &BatchWarmStart,
        shared_start: Option<&[f64]>,
        column: usize,
        q: usize,
    ) -> Result<Vec<f64>> {
        match warm_start {
            BatchWarmStart::TemplateTheta => {
                self.kernel.projected_theta(self.kernel.template_theta())
            }
            BatchWarmStart::SharedTheta => self
                .kernel
                .projected_theta(shared_start.unwrap_or(self.kernel.template_theta())),
            BatchWarmStart::Fixed(theta) => self.kernel.projected_theta(theta),
            BatchWarmStart::Provided(theta) => {
                if theta.nrows() != self.kernel.template_theta().len() || theta.ncols() != q {
                    return Err(MixedModelError::DimensionMismatch(format!(
                        "provided warm-start theta matrix has shape {} x {}, expected {} x {q}",
                        theta.nrows(),
                        theta.ncols(),
                        self.kernel.template_theta().len()
                    )));
                }
                self.kernel.projected_theta(theta.column(column).as_slice())
            }
        }
    }

    fn optimize_theta<F>(
        &self,
        workspace: &mut LmmWorkspace<'_>,
        initial: Vec<f64>,
        control: &BatchOptimizerControl,
        mut objective: F,
    ) -> Result<crate::model::linear::PatternSearchOutcome>
    where
        F: FnMut(&mut LmmWorkspace<'_>, &[f64]) -> Result<f64>,
    {
        self.kernel.validate_theta(&initial)?;
        let finitial = objective(workspace, &initial)?;
        let step = control
            .initial_step
            .clone()
            .unwrap_or_else(|| vec![0.5; initial.len()]);
        let step_tol = vec![control.theta_tolerance; initial.len()];
        LinearMixedModel::run_multivariate_pattern_search(
            initial,
            finitial,
            self.kernel.lower_bounds(),
            step,
            &step_tol,
            control.max_evaluations,
            control.objective_tolerance,
            |theta| objective(workspace, theta),
        )
    }

    fn classify_responses(
        &self,
        responses: &DMatrix<f64>,
        result: &mut ResponseBatchFit,
        options: &BatchOptions,
        driver: &mut SinkDriver<'_>,
    ) -> Result<Vec<usize>> {
        let mut valid = Vec::new();
        let mut failures = 0usize;
        for col in 0..responses.ncols() {
            if driver.stopped {
                let remaining: Vec<usize> = (col..responses.ncols()).collect();
                self.mark_sink_stopped(&remaining, result);
                break;
            }
            let mut min = f64::INFINITY;
            let mut max = f64::NEG_INFINITY;
            let mut finite = true;
            for row in 0..responses.nrows() {
                let value = responses[(row, col)];
                if !value.is_finite() {
                    finite = false;
                    break;
                }
                min = min.min(value);
                max = max.max(value);
            }
            if !finite {
                failures += 1;
                result.status[col] = ResponseFitStatus::InvalidResponse;
                result.diagnostics.push(ResponseColumnDiagnostic {
                    column: col,
                    reason: ResponseDiagnosticReason::NonFiniteResponse,
                    message: "response column contains a non-finite value".to_string(),
                });
                self.emit_failure_row(result, col, driver)?;
            } else if (max - min) < f64::EPSILON {
                failures += 1;
                result.status[col] = ResponseFitStatus::ConstantResponse;
                result.diagnostics.push(ResponseColumnDiagnostic {
                    column: col,
                    reason: ResponseDiagnosticReason::ConstantResponse,
                    message: "response column is constant".to_string(),
                });
                self.emit_failure_row(result, col, driver)?;
            } else {
                valid.push(col);
            }
            if options
                .max_failures
                .is_some_and(|max_failures| failures >= max_failures)
            {
                for trailing in (col + 1)..responses.ncols() {
                    // If the sink stopped, unemitted trailing columns read
                    // SinkStopped, keeping the emitted/populated invariant.
                    if driver.stopped {
                        self.mark_sink_stopped(&[trailing], result);
                        continue;
                    }
                    result.status[trailing] = ResponseFitStatus::Unsupported;
                    result.diagnostics.push(ResponseColumnDiagnostic {
                        column: trailing,
                        reason: ResponseDiagnosticReason::UnsupportedMode,
                        message: "batch max_failures guard stopped before this column".to_string(),
                    });
                    self.emit_failure_row(result, trailing, driver)?;
                }
                break;
            }
        }
        Ok(valid)
    }

    fn scatter_profile(
        &self,
        profile: &ResponseMatrixProfile,
        columns: &[usize],
        result: &mut ResponseBatchFit,
        boundary: bool,
        theta: &[f64],
        driver: &mut SinkDriver<'_>,
    ) -> Result<()> {
        for (local, &column) in columns.iter().enumerate() {
            if driver.stopped {
                self.mark_sink_stopped(&columns[local..], result);
                return Ok(());
            }
            for row in 0..profile.beta.nrows() {
                result.beta[(row, column)] = profile.beta[(row, local)];
            }
            result.sigma[column] = profile.sigma[local];
            result.pwrss[column] = profile.pwrss[local];
            result.objective[column] = profile.objectives[local];
            result.status[column] = if boundary {
                result.diagnostics.push(ResponseColumnDiagnostic {
                    column,
                    reason: ResponseDiagnosticReason::BoundaryTheta,
                    message: "theta is on a covariance lower bound".to_string(),
                });
                ResponseFitStatus::Boundary
            } else {
                ResponseFitStatus::Success
            };
            let beta_column = profile.beta.column(local);
            driver.emit(ResponseColumnRow {
                column,
                status: result.status[column],
                beta: Some(beta_column.as_slice()),
                sigma: profile.sigma[local],
                pwrss: profile.pwrss[local],
                objective: profile.objectives[local],
                theta: Some(theta),
                diagnostic: if boundary {
                    result.diagnostics.last()
                } else {
                    None
                },
            })?;
        }
        Ok(())
    }

    /// Emit a value-free row for a column whose status and (optional)
    /// diagnostic were just recorded in `result`.
    fn emit_failure_row(
        &self,
        result: &ResponseBatchFit,
        column: usize,
        driver: &mut SinkDriver<'_>,
    ) -> Result<()> {
        driver.emit(ResponseColumnRow {
            column,
            status: result.status[column],
            beta: None,
            sigma: f64::NAN,
            pwrss: f64::NAN,
            objective: f64::NAN,
            theta: None,
            diagnostic: result
                .diagnostics
                .last()
                .filter(|diagnostic| diagnostic.column == column),
        })
    }

    /// Mark columns skipped after a sink stop; they are never emitted.
    fn mark_sink_stopped(&self, columns: &[usize], result: &mut ResponseBatchFit) {
        for &column in columns {
            result.status[column] = ResponseFitStatus::Unsupported;
            result.diagnostics.push(ResponseColumnDiagnostic {
                column,
                reason: ResponseDiagnosticReason::SinkStopped,
                message: "sink requested stop before this column".to_string(),
            });
        }
    }

    fn empty_result(&self, q: usize, theta: ThetaBatch) -> ResponseBatchFit {
        ResponseBatchFit {
            beta: DMatrix::from_element(self.kernel.p(), q, f64::NAN),
            sigma: DVector::from_element(q, f64::NAN),
            pwrss: DVector::from_element(q, f64::NAN),
            objective: DVector::from_element(q, f64::NAN),
            theta,
            status: vec![ResponseFitStatus::OptimizerFailed; q],
            diagnostics: Vec::new(),
        }
    }
}

/// Extract one column of a multi-column profile as a standalone profile.
fn single_column_profile(profile: &ResponseMatrixProfile, local: usize) -> ResponseMatrixProfile {
    ResponseMatrixProfile {
        beta: DMatrix::from_column_slice(
            profile.beta.nrows(),
            1,
            profile.beta.column(local).as_slice(),
        ),
        sigma: DVector::from_element(1, profile.sigma[local]),
        pwrss: DVector::from_element(1, profile.pwrss[local]),
        objectives: DVector::from_element(1, profile.objectives[local]),
        total_objective: profile.objectives[local],
        logdet_re: profile.logdet_re,
        logdet_xx: profile.logdet_xx,
    }
}
