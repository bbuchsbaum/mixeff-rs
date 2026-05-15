//! Response-matrix batch fitting for linear mixed models.
//!
//! The batch API treats the columns of an `n x q` response matrix as
//! independent LMM responses that share one fixed/random-effect structure.
//! It does not estimate cross-response covariance.

use nalgebra::{DMatrix, DVector};

use crate::error::{MixedModelError, Result};
use crate::model::linear::{
    create_structural_al, profile_response_matrix_with_l_blocks, update_l_from_parts,
    LinearMixedModel, ResponseMatrixProfile,
};
use crate::types::{MatrixBlock, ReMat};

/// Cached invariant structure for fitting independent LMM response columns.
#[derive(Debug, Clone)]
pub struct LinearMixedModelBatch {
    template: LinearMixedModel,
    reterms: Vec<ReMat>,
    x: DMatrix<f64>,
    structural_a: Vec<MatrixBlock>,
    structural_l: Vec<MatrixBlock>,
    template_theta: Vec<f64>,
    lower_bounds: Vec<f64>,
    parmap: Vec<(usize, usize, usize)>,
    cholesky_zero_pad_tolerance: f64,
    options: BatchOptions,
}

/// Batch execution controls that are independent of optimizer tolerances.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchOptions {
    /// Number of response columns profiled at a time.
    pub chunk_columns: usize,
    /// Optional guardrail for callers that want to stop after many failures.
    pub max_failures: Option<usize>,
}

impl Default for BatchOptions {
    fn default() -> Self {
        Self {
            chunk_columns: 64,
            max_failures: None,
        }
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
}

/// Column-local diagnostic emitted by the batch engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResponseColumnDiagnostic {
    pub column: usize,
    pub reason: ResponseDiagnosticReason,
    pub message: String,
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
        let x = model.feterm.full_rank_x().into_owned();
        let (structural_a, structural_l) = create_structural_al(&model.reterms, &x)?;
        Ok(Self {
            template: model.clone(),
            reterms: model.reterms.clone(),
            x,
            structural_a,
            structural_l,
            template_theta: model.theta(),
            lower_bounds: model.lower_bounds(),
            parmap: model.parmap.clone(),
            cholesky_zero_pad_tolerance: model
                .compiler_policy()
                .thresholds
                .cholesky_zero_pad_tolerance,
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
        self.validate_response_rows(responses)?;
        match mode {
            ResponseBatchMode::ProfileAtTheta { theta, reml } => {
                self.fit_profile_at_theta(responses, &theta, reml, &self.options)
            }
            ResponseBatchMode::OptimizeSharedTheta { reml, control } => {
                self.fit_optimize_shared_theta(responses, reml, &control)
            }
            ResponseBatchMode::OptimizePerColumn {
                reml,
                warm_start,
                control,
            } => self.fit_optimize_per_column(responses, reml, warm_start, &control),
            ResponseBatchMode::OptimizeGrouped {
                reml,
                grouping,
                control,
            } => self.fit_optimize_grouped(responses, reml, grouping, &control),
        }
    }

    fn validate_response_rows(&self, responses: &DMatrix<f64>) -> Result<()> {
        if responses.nrows() != self.template.dims.n {
            return Err(MixedModelError::DimensionMismatch(format!(
                "response matrix has {} rows, expected {}",
                responses.nrows(),
                self.template.dims.n
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
    ) -> Result<ResponseBatchFit> {
        self.validate_theta(theta)?;
        let mut result = self.empty_result(responses.ncols(), ThetaBatch::Shared(theta.to_vec()));
        let valid_columns = self.classify_responses(responses, &mut result, options);
        if valid_columns.is_empty() {
            return Ok(result);
        }

        let boundary = self.theta_on_boundary(theta);
        let profile =
            self.profile_columns_at_theta(theta, responses, reml, &valid_columns, options)?;
        self.scatter_profile(&profile, &valid_columns, &mut result, boundary);
        Ok(result)
    }

    fn fit_optimize_shared_theta(
        &self,
        responses: &DMatrix<f64>,
        reml: bool,
        control: &BatchOptimizerControl,
    ) -> Result<ResponseBatchFit> {
        self.validate_control(control)?;
        let mut result = self.empty_result(
            responses.ncols(),
            ThetaBatch::Shared(self.template_theta.clone()),
        );
        let valid_columns = self.classify_responses(responses, &mut result, &control.options);
        if valid_columns.is_empty() {
            return Ok(result);
        }

        let initial = self.projected_theta(&self.template_theta)?;
        let outcome = self.optimize_theta(initial, control, |theta| {
            Ok(self
                .profile_columns_at_theta(theta, responses, reml, &valid_columns, &control.options)
                .map(|profile| profile.total_objective)
                .unwrap_or(f64::INFINITY))
        })?;
        let mut theta = outcome.best_theta;
        LinearMixedModel::rectify_theta_columns(&mut theta, &self.parmap, self.reterms.len());
        let boundary = self.theta_on_boundary(&theta);
        let profile = self.profile_columns_at_theta(
            &theta,
            responses,
            reml,
            &valid_columns,
            &control.options,
        )?;
        result.theta = ThetaBatch::Shared(theta);
        self.scatter_profile(&profile, &valid_columns, &mut result, boundary);
        Ok(result)
    }

    fn fit_optimize_per_column(
        &self,
        responses: &DMatrix<f64>,
        reml: bool,
        warm_start: BatchWarmStart,
        control: &BatchOptimizerControl,
    ) -> Result<ResponseBatchFit> {
        self.validate_control(control)?;
        let q = responses.ncols();
        let ntheta = self.template_theta.len();
        let mut theta_out = DMatrix::from_element(ntheta, q, f64::NAN);
        let mut result = self.empty_result(q, ThetaBatch::PerColumn(theta_out.clone()));
        let valid_columns = self.classify_responses(responses, &mut result, &control.options);
        if valid_columns.is_empty() {
            return Ok(result);
        }

        let shared_start = match warm_start {
            BatchWarmStart::SharedTheta => {
                let shared = self.fit_optimize_shared_theta(responses, reml, control)?;
                match shared.theta {
                    ThetaBatch::Shared(theta) => Some(theta),
                    _ => None,
                }
            }
            _ => None,
        };

        for &column in &valid_columns {
            let initial =
                self.warm_start_for_column(&warm_start, shared_start.as_deref(), column, q)?;
            let single = select_response_columns(responses, &[column]);
            let outcome = self.optimize_theta(initial, control, |theta| {
                Ok(self
                    .profile_columns_at_theta(theta, &single, reml, &[0], &control.options)
                    .map(|profile| profile.total_objective)
                    .unwrap_or(f64::INFINITY))
            });

            let Ok(outcome) = outcome else {
                result.status[column] = ResponseFitStatus::OptimizerFailed;
                result.diagnostics.push(ResponseColumnDiagnostic {
                    column,
                    reason: ResponseDiagnosticReason::OptimizerFailed,
                    message: "per-column theta optimization failed".to_string(),
                });
                continue;
            };

            let mut theta = outcome.best_theta;
            LinearMixedModel::rectify_theta_columns(&mut theta, &self.parmap, self.reterms.len());
            for row in 0..ntheta {
                theta_out[(row, column)] = theta[row];
            }
            let boundary = self.theta_on_boundary(&theta);
            let profile =
                self.profile_columns_at_theta(&theta, &single, reml, &[0], &control.options)?;
            self.scatter_profile(&profile, &[column], &mut result, boundary);
        }

        result.theta = ThetaBatch::PerColumn(theta_out);
        Ok(result)
    }

    fn fit_optimize_grouped(
        &self,
        responses: &DMatrix<f64>,
        reml: bool,
        grouping: BatchThetaGrouping,
        control: &BatchOptimizerControl,
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

        let ntheta = self.template_theta.len();
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
        let valid_columns = self.classify_responses(responses, &mut result, &control.options);
        if valid_columns.is_empty() {
            return Ok(result);
        }

        for group in 0..group_count {
            let group_columns: Vec<usize> = valid_columns
                .iter()
                .copied()
                .filter(|&column| group_for_column[column] == group)
                .collect();
            if group_columns.is_empty() {
                continue;
            }
            let initial = self.projected_theta(&self.template_theta)?;
            let outcome = self.optimize_theta(initial, control, |theta| {
                Ok(self
                    .profile_columns_at_theta(
                        theta,
                        responses,
                        reml,
                        &group_columns,
                        &control.options,
                    )
                    .map(|profile| profile.total_objective)
                    .unwrap_or(f64::INFINITY))
            })?;
            let mut theta = outcome.best_theta;
            LinearMixedModel::rectify_theta_columns(&mut theta, &self.parmap, self.reterms.len());
            for row in 0..ntheta {
                group_theta[(row, group)] = theta[row];
            }
            let boundary = self.theta_on_boundary(&theta);
            let profile = self.profile_columns_at_theta(
                &theta,
                responses,
                reml,
                &group_columns,
                &control.options,
            )?;
            self.scatter_profile(&profile, &group_columns, &mut result, boundary);
        }

        result.theta = ThetaBatch::Grouped {
            theta: group_theta,
            group_for_column,
        };
        Ok(result)
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
        if let Some(step) = &control.initial_step {
            if step.len() != self.template_theta.len() {
                return Err(MixedModelError::DimensionMismatch(format!(
                    "batch optimizer initial_step has length {}, expected {}",
                    step.len(),
                    self.template_theta.len()
                )));
            }
        }
        Ok(())
    }

    fn validate_theta(&self, theta: &[f64]) -> Result<()> {
        if theta.len() != self.template_theta.len() {
            return Err(MixedModelError::DimensionMismatch(format!(
                "theta vector has length {}, expected {}",
                theta.len(),
                self.template_theta.len()
            )));
        }
        if theta.iter().any(|value| !value.is_finite()) {
            return Err(MixedModelError::InvalidArgument(
                "theta vector must contain only finite values".to_string(),
            ));
        }
        if let Some((index, (&value, &lower))) = theta
            .iter()
            .zip(self.lower_bounds.iter())
            .enumerate()
            .find(|(_, (&value, &lower))| lower.is_finite() && value < lower)
        {
            return Err(MixedModelError::InvalidArgument(format!(
                "theta[{index}] = {value} is below lower bound {lower}"
            )));
        }
        Ok(())
    }

    fn projected_theta(&self, theta: &[f64]) -> Result<Vec<f64>> {
        if theta.len() != self.template_theta.len() {
            return Err(MixedModelError::DimensionMismatch(format!(
                "theta vector has length {}, expected {}",
                theta.len(),
                self.template_theta.len()
            )));
        }
        let mut projected = theta.to_vec();
        for (value, &lower) in projected.iter_mut().zip(self.lower_bounds.iter()) {
            if lower.is_finite() && *value < lower {
                *value = lower;
            }
        }
        Ok(projected)
    }

    fn warm_start_for_column(
        &self,
        warm_start: &BatchWarmStart,
        shared_start: Option<&[f64]>,
        column: usize,
        q: usize,
    ) -> Result<Vec<f64>> {
        match warm_start {
            BatchWarmStart::TemplateTheta => self.projected_theta(&self.template_theta),
            BatchWarmStart::SharedTheta => {
                self.projected_theta(shared_start.unwrap_or(&self.template_theta))
            }
            BatchWarmStart::Fixed(theta) => self.projected_theta(theta),
            BatchWarmStart::Provided(theta) => {
                if theta.nrows() != self.template_theta.len() || theta.ncols() != q {
                    return Err(MixedModelError::DimensionMismatch(format!(
                        "provided warm-start theta matrix has shape {} x {}, expected {} x {q}",
                        theta.nrows(),
                        theta.ncols(),
                        self.template_theta.len()
                    )));
                }
                self.projected_theta(theta.column(column).as_slice())
            }
        }
    }

    fn optimize_theta<F>(
        &self,
        initial: Vec<f64>,
        control: &BatchOptimizerControl,
        mut objective: F,
    ) -> Result<crate::model::linear::PatternSearchOutcome>
    where
        F: FnMut(&[f64]) -> Result<f64>,
    {
        self.validate_theta(&initial)?;
        let finitial = objective(&initial)?;
        let step = control
            .initial_step
            .clone()
            .unwrap_or_else(|| vec![0.5; initial.len()]);
        let step_tol = vec![control.theta_tolerance; initial.len()];
        LinearMixedModel::run_multivariate_pattern_search(
            initial,
            finitial,
            &self.lower_bounds,
            step,
            &step_tol,
            control.max_evaluations,
            control.objective_tolerance,
            objective,
        )
    }

    fn profile_columns_at_theta(
        &self,
        theta: &[f64],
        responses: &DMatrix<f64>,
        reml: bool,
        columns: &[usize],
        options: &BatchOptions,
    ) -> Result<ResponseMatrixProfile> {
        self.validate_theta(theta)?;
        let mut reterms = self.reterms.clone();
        set_reterms_theta(&mut reterms, theta)?;
        let mut l_blocks = self.structural_l.clone();
        update_l_from_parts(
            &self.structural_a,
            &mut l_blocks,
            &reterms,
            self.cholesky_zero_pad_tolerance,
        )?;

        let p = self.template.dims.p;
        let mut beta = DMatrix::from_element(p, columns.len(), f64::NAN);
        let mut sigma = DVector::from_element(columns.len(), f64::NAN);
        let mut pwrss = DVector::from_element(columns.len(), f64::NAN);
        let mut objectives = DVector::from_element(columns.len(), f64::NAN);
        let mut total_objective = 0.0;
        let mut logdet_re = f64::NAN;
        let mut logdet_xx = f64::NAN;

        let mut dest_offset = 0;
        for (chunk_start, chunk_columns) in columns.chunks(options.chunk_columns).enumerate() {
            let chunk = select_response_columns(responses, chunk_columns);
            let profile = profile_response_matrix_with_l_blocks(
                &reterms,
                &self.x,
                &chunk,
                &l_blocks,
                reml,
                self.template.dims.n,
                self.template.dims.p,
            )?;
            if chunk_start == 0 {
                logdet_re = profile.logdet_re;
                logdet_xx = profile.logdet_xx;
            }
            for source_col in 0..chunk_columns.len() {
                let local = dest_offset + source_col;
                for row in 0..p {
                    beta[(row, local)] = profile.beta[(row, source_col)];
                }
                sigma[local] = profile.sigma[source_col];
                pwrss[local] = profile.pwrss[source_col];
                objectives[local] = profile.objectives[source_col];
                total_objective += profile.objectives[source_col];
            }
            dest_offset += chunk_columns.len();
        }

        Ok(ResponseMatrixProfile {
            beta,
            sigma,
            pwrss,
            objectives,
            total_objective,
            logdet_re,
            logdet_xx,
        })
    }

    fn classify_responses(
        &self,
        responses: &DMatrix<f64>,
        result: &mut ResponseBatchFit,
        options: &BatchOptions,
    ) -> Vec<usize> {
        let mut valid = Vec::new();
        let mut failures = 0usize;
        for col in 0..responses.ncols() {
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
            } else if (max - min) < f64::EPSILON {
                failures += 1;
                result.status[col] = ResponseFitStatus::ConstantResponse;
                result.diagnostics.push(ResponseColumnDiagnostic {
                    column: col,
                    reason: ResponseDiagnosticReason::ConstantResponse,
                    message: "response column is constant".to_string(),
                });
            } else {
                valid.push(col);
            }
            if options
                .max_failures
                .is_some_and(|max_failures| failures >= max_failures)
            {
                for trailing in (col + 1)..responses.ncols() {
                    result.status[trailing] = ResponseFitStatus::Unsupported;
                    result.diagnostics.push(ResponseColumnDiagnostic {
                        column: trailing,
                        reason: ResponseDiagnosticReason::UnsupportedMode,
                        message: "batch max_failures guard stopped before this column".to_string(),
                    });
                }
                break;
            }
        }
        valid
    }

    fn scatter_profile(
        &self,
        profile: &ResponseMatrixProfile,
        columns: &[usize],
        result: &mut ResponseBatchFit,
        boundary: bool,
    ) {
        for (local, &column) in columns.iter().enumerate() {
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
        }
    }

    fn empty_result(&self, q: usize, theta: ThetaBatch) -> ResponseBatchFit {
        ResponseBatchFit {
            beta: DMatrix::from_element(self.template.dims.p, q, f64::NAN),
            sigma: DVector::from_element(q, f64::NAN),
            pwrss: DVector::from_element(q, f64::NAN),
            objective: DVector::from_element(q, f64::NAN),
            theta,
            status: vec![ResponseFitStatus::OptimizerFailed; q],
            diagnostics: Vec::new(),
        }
    }

    fn theta_on_boundary(&self, theta: &[f64]) -> bool {
        theta
            .iter()
            .zip(self.lower_bounds.iter())
            .any(|(&value, &lower)| {
                lower.is_finite()
                    && (value - lower).abs() <= self.template.optsum.xtol_zero_abs.max(1e-12) * 10.0
            })
    }
}

fn set_reterms_theta(reterms: &mut [ReMat], theta: &[f64]) -> Result<()> {
    let mut offset = 0;
    for reterm in reterms {
        let ntheta = reterm.n_theta();
        if offset + ntheta > theta.len() {
            return Err(MixedModelError::DimensionMismatch(format!(
                "theta vector ended before random-effect term with {ntheta} parameter(s)"
            )));
        }
        reterm.set_theta(&theta[offset..offset + ntheta])?;
        offset += ntheta;
    }
    if offset != theta.len() {
        return Err(MixedModelError::DimensionMismatch(format!(
            "theta vector has {} entries, but random-effect structure uses {offset}",
            theta.len()
        )));
    }
    Ok(())
}

fn select_response_columns(responses: &DMatrix<f64>, columns: &[usize]) -> DMatrix<f64> {
    DMatrix::from_fn(responses.nrows(), columns.len(), |row, col| {
        responses[(row, columns[col])]
    })
}
