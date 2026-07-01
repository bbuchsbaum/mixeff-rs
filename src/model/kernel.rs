//! Internal θ → profiled-objective kernel for LMM response profiling.
//!
//! [`LmmObjectiveKernel`] owns the invariant `[Z X]` structure extracted from
//! a fitted [`LinearMixedModel`] (structural `A`/`L` block templates, fixed
//! design, θ parameter map and bounds), while [`LmmWorkspace`] owns the
//! mutable per-worker buffers (a `ReMat` copy carrying λ and the blocked
//! Cholesky factor) that a θ evaluation mutates. Splitting the two lets
//! optimizer callbacks and batch fitting reuse one workspace across many
//! objective evaluations instead of re-cloning the block structure each time,
//! and lets parallel workers share one kernel with independent workspaces.
//!
//! This module deliberately stays on the numerical-core side of the crate:
//! it may use `error`, `types`, and the block-Cholesky entry points in
//! `model::linear`, but must not depend on `stats`, `compiler`, `guide`, or
//! `pathology` (see `tests/architecture.rs`).

use nalgebra::DMatrix;

use crate::error::{MixedModelError, Result};
use crate::model::linear::{
    create_structural_al, profile_response_matrix_with_l_blocks, update_l_from_parts,
    LinearMixedModel, ResponseMatrixProfile,
};
use crate::types::{MatrixBlock, ReMat};

/// Invariant θ → profiled-objective structure shared by all workspaces.
#[derive(Debug, Clone)]
pub(crate) struct LmmObjectiveKernel {
    reterms: Vec<ReMat>,
    x: DMatrix<f64>,
    structural_a: Vec<MatrixBlock>,
    structural_l: Vec<MatrixBlock>,
    template_theta: Vec<f64>,
    lower_bounds: Vec<f64>,
    parmap: Vec<(usize, usize, usize)>,
    n: usize,
    p: usize,
    cholesky_zero_pad_tolerance: f64,
    xtol_zero_abs: f64,
}

impl LmmObjectiveKernel {
    /// Extract the invariant profiling structure from a template model.
    pub(crate) fn from_model(model: &LinearMixedModel) -> Result<Self> {
        let x = model.feterm.full_rank_x().into_owned();
        let (structural_a, structural_l) = create_structural_al(&model.reterms, &x)?;
        Ok(Self {
            reterms: model.reterms.clone(),
            x,
            structural_a,
            structural_l,
            template_theta: model.theta(),
            lower_bounds: model.lower_bounds(),
            parmap: model.parmap.clone(),
            n: model.dims.n,
            p: model.dims.p,
            cholesky_zero_pad_tolerance: model
                .compiler_policy()
                .thresholds
                .cholesky_zero_pad_tolerance,
            xtol_zero_abs: model.optsum.xtol_zero_abs,
        })
    }

    pub(crate) fn n(&self) -> usize {
        self.n
    }

    pub(crate) fn p(&self) -> usize {
        self.p
    }

    pub(crate) fn reterm_count(&self) -> usize {
        self.reterms.len()
    }

    pub(crate) fn template_theta(&self) -> &[f64] {
        &self.template_theta
    }

    pub(crate) fn lower_bounds(&self) -> &[f64] {
        &self.lower_bounds
    }

    pub(crate) fn parmap(&self) -> &[(usize, usize, usize)] {
        &self.parmap
    }

    /// Allocate a mutable workspace bound to this kernel's structure.
    pub(crate) fn workspace(&self) -> LmmWorkspace<'_> {
        LmmWorkspace {
            kernel: self,
            reterms: self.reterms.clone(),
            l_blocks: self.structural_l.clone(),
        }
    }

    /// Reject θ vectors with the wrong length, non-finite entries, or
    /// entries below their lower bound.
    pub(crate) fn validate_theta(&self, theta: &[f64]) -> Result<()> {
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

    /// Clamp θ onto the feasible region defined by the lower bounds.
    pub(crate) fn projected_theta(&self, theta: &[f64]) -> Result<Vec<f64>> {
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

    /// Whether θ sits (numerically) on a covariance lower bound.
    pub(crate) fn theta_on_boundary(&self, theta: &[f64]) -> bool {
        theta
            .iter()
            .zip(self.lower_bounds.iter())
            .any(|(&value, &lower)| {
                lower.is_finite() && (value - lower).abs() <= self.xtol_zero_abs.max(1e-12) * 10.0
            })
    }
}

/// Mutable per-worker buffers for θ evaluation against one kernel.
#[derive(Debug, Clone)]
pub(crate) struct LmmWorkspace<'k> {
    kernel: &'k LmmObjectiveKernel,
    reterms: Vec<ReMat>,
    l_blocks: Vec<MatrixBlock>,
}

impl LmmWorkspace<'_> {
    /// Distribute a θ vector onto the per-term λ factors.
    pub(crate) fn set_theta(&mut self, theta: &[f64]) -> Result<()> {
        set_reterms_theta(&mut self.reterms, theta)
    }

    /// Recompute the blocked Cholesky factor from the structural `A` blocks
    /// at the workspace's current λ values.
    pub(crate) fn update_l(&mut self) -> Result<()> {
        update_l_from_parts(
            &self.kernel.structural_a,
            &mut self.l_blocks,
            &self.reterms,
            self.kernel.cholesky_zero_pad_tolerance,
        )
    }

    /// Validate θ, install it, and refactorize in one step.
    pub(crate) fn factorize_at(&mut self, theta: &[f64]) -> Result<()> {
        self.kernel.validate_theta(theta)?;
        self.set_theta(theta)?;
        self.update_l()
    }

    /// Profile every column of `responses` at the current factorization.
    pub(crate) fn profile(
        &self,
        responses: &DMatrix<f64>,
        reml: bool,
    ) -> Result<ResponseMatrixProfile> {
        profile_response_matrix_with_l_blocks(
            &self.reterms,
            &self.kernel.x,
            responses,
            &self.l_blocks,
            reml,
            self.kernel.n,
            self.kernel.p,
        )
    }

    /// Factorize at θ and profile the selected response columns in chunks.
    ///
    /// Results are packed in the order of `columns`; the log-determinant
    /// terms are shared across columns and taken from the first chunk.
    pub(crate) fn profile_columns_at_theta(
        &mut self,
        theta: &[f64],
        responses: &DMatrix<f64>,
        reml: bool,
        columns: &[usize],
        chunk_columns: usize,
        parallel: bool,
    ) -> Result<ResponseMatrixProfile> {
        self.factorize_at(theta)?;
        self.profile_columns(responses, reml, columns, chunk_columns, parallel)
    }

    /// Profile the selected response columns in chunks at the current
    /// factorization.
    ///
    /// With `parallel` set, independent chunks are profiled on the rayon
    /// thread pool; the scatter below still runs serially in chunk order, so
    /// the packed results (including the floating-point `total_objective`
    /// accumulation) are identical to serial execution.
    pub(crate) fn profile_columns(
        &self,
        responses: &DMatrix<f64>,
        reml: bool,
        columns: &[usize],
        chunk_columns: usize,
        parallel: bool,
    ) -> Result<ResponseMatrixProfile> {
        debug_assert!(chunk_columns > 0, "chunk_columns must be positive");
        // Identity selection in one chunk (the per-column optimizer's shape
        // on every objective evaluation): the packed result is exactly the
        // chunk profile, so skip the column gather and repack entirely.
        if columns.len() <= chunk_columns
            && columns.len() == responses.ncols()
            && columns.iter().enumerate().all(|(i, &c)| i == c)
        {
            return self.profile(responses, reml);
        }
        // Fan-out per objective evaluation only pays for itself once there
        // are enough independent chunks carrying enough total work; below
        // these thresholds the rayon dispatch overhead measurably regresses
        // small problems. This is purely an execution heuristic — results
        // are identical either way.
        const MIN_PARALLEL_CHUNKS: usize = 4;
        const MIN_PARALLEL_WORK: usize = 16_384; // profiled columns × observations
        let chunks: Vec<&[usize]> = columns.chunks(chunk_columns).collect();
        let profiles: Vec<ResponseMatrixProfile> = if parallel
            && chunks.len() >= MIN_PARALLEL_CHUNKS
            && columns.len().saturating_mul(self.kernel.n) >= MIN_PARALLEL_WORK
        {
            #[cfg(feature = "rayon")]
            {
                use rayon::prelude::*;
                chunks
                    .par_iter()
                    .map(|chunk| self.profile(&select_response_columns(responses, chunk), reml))
                    .collect::<Result<Vec<_>>>()?
            }
            #[cfg(not(feature = "rayon"))]
            {
                return Err(MixedModelError::InvalidArgument(
                    "parallel batch profiling requires the `rayon` cargo feature".to_string(),
                ));
            }
        } else {
            chunks
                .iter()
                .map(|chunk| self.profile(&select_response_columns(responses, chunk), reml))
                .collect::<Result<Vec<_>>>()?
        };

        let p = self.kernel.p;
        let mut beta = DMatrix::from_element(p, columns.len(), f64::NAN);
        let mut sigma = nalgebra::DVector::from_element(columns.len(), f64::NAN);
        let mut pwrss = nalgebra::DVector::from_element(columns.len(), f64::NAN);
        let mut objectives = nalgebra::DVector::from_element(columns.len(), f64::NAN);
        let mut total_objective = 0.0;
        let mut logdet_re = f64::NAN;
        let mut logdet_xx = f64::NAN;

        let mut dest_offset = 0;
        for (chunk_start, (chunk_columns, profile)) in
            chunks.iter().zip(profiles.iter()).enumerate()
        {
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
}

/// Distribute a flat θ vector across the per-term λ factors.
pub(crate) fn set_reterms_theta(reterms: &mut [ReMat], theta: &[f64]) -> Result<()> {
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

/// Gather the listed response columns into a dense contiguous matrix.
pub(crate) fn select_response_columns(responses: &DMatrix<f64>, columns: &[usize]) -> DMatrix<f64> {
    DMatrix::from_fn(responses.nrows(), columns.len(), |row, col| {
        responses[(row, columns[col])]
    })
}
