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
    create_structural_al, profile_response_matrix_with_scratch, update_l_from_parts,
    LinearMixedModel, ModelDims, ProfileScratch, ProfiledGradientInputs, ResponseMatrixProfile,
};
use crate::types::matrix_block::block_index;
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
            scratch: ProfileScratch::default(),
            gradient_fe: Vec::new(),
            gradient_trailing_l: MatrixBlock::Dense(DMatrix::zeros(0, 0)),
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
    /// Reusable profile buffers (Phase 6 T6.1): a θ evaluation over a
    /// chunk allocates only its output vectors.
    scratch: ProfileScratch,
    /// `[X|y]ᵀ Z_j` blocks for the per-column gradient oracle (Phase 6
    /// T6.2): rows `0..p` copied once from the structural `Xᵀ Z_j`, row `p`
    /// rewritten per evaluation from the profile's `Z_jᵀ y`.
    gradient_fe: Vec<MatrixBlock>,
    /// `[[L_xx, 0], [βᵀ L_xx, √pwrss]]` for the per-column gradient oracle.
    gradient_trailing_l: MatrixBlock,
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

    /// Profile every column of `responses` at the current factorization
    /// on caller-owned scratch.
    pub(crate) fn profile_with_scratch(
        &self,
        responses: &DMatrix<f64>,
        reml: bool,
        scratch: &mut ProfileScratch,
    ) -> Result<ResponseMatrixProfile> {
        profile_response_matrix_with_scratch(
            &self.reterms,
            &self.kernel.x,
            responses,
            &self.l_blocks,
            reml,
            self.kernel.n,
            self.kernel.p,
            scratch,
        )
    }

    /// Profile every column of `responses` at the current factorization
    /// on the workspace's own scratch.
    pub(crate) fn profile(
        &mut self,
        responses: &DMatrix<f64>,
        reml: bool,
    ) -> Result<ResponseMatrixProfile> {
        let mut scratch = std::mem::take(&mut self.scratch);
        let profile = self.profile_with_scratch(responses, reml, &mut scratch);
        self.scratch = scratch;
        profile
    }

    /// Objective and analytic gradient for one response column at `theta`
    /// (Phase 6 T6.2): factorize, profile the column on the workspace
    /// scratch, then evaluate the Phase 5 gradient on the structural blocks
    /// augmented with the column's `Z_jᵀ y`, `Xᵀ y` and the profiled
    /// `(β, pwrss)` folded into a `(p + 1) × (p + 1)` trailing factor.
    pub(crate) fn objective_and_gradient_for_column(
        &mut self,
        theta: &[f64],
        y: &DMatrix<f64>,
        reml: bool,
    ) -> Result<(f64, Vec<f64>)> {
        if y.ncols() != 1 {
            return Err(MixedModelError::DimensionMismatch(format!(
                "column gradient oracle expects one response column, got {}",
                y.ncols()
            )));
        }
        self.factorize_at(theta)?;
        let profile = self.profile(y, reml)?;
        let k = self.reterms.len();
        let p = self.kernel.p;
        let base = k * (k + 1) / 2;
        if self.gradient_fe.len() != k {
            self.gradient_fe = (0..k)
                .map(|j| {
                    let xz = self.kernel.structural_a[base + j].as_dense();
                    let mut block = DMatrix::zeros(p + 1, xz.ncols());
                    block.rows_mut(0, p).copy_from(&xz);
                    MatrixBlock::Dense(block)
                })
                .collect();
        }
        let zty = self.scratch.response_re_cross_products();
        for (j, block) in self.gradient_fe.iter_mut().enumerate() {
            if let MatrixBlock::Dense(block) = block {
                let column = zty[j].column(0);
                for c in 0..block.ncols() {
                    block[(p, c)] = column[c];
                }
            }
        }
        let pwrss = profile.pwrss[0];
        let beta = profile.beta.column(0);
        let mut trailing = DMatrix::zeros(p + 1, p + 1);
        crate::types::matrix_block::with_dense_block(&self.l_blocks[block_index(k, k)], |l_xx| {
            trailing.view_mut((0, 0), (p, p)).copy_from(l_xx);
            for c in 0..p {
                let mut value = 0.0;
                for r in 0..p {
                    value += beta[r] * l_xx[(r, c)];
                }
                trailing[(p, c)] = value;
            }
        });
        trailing[(p, p)] = pwrss.max(0.0).sqrt();
        self.gradient_trailing_l = MatrixBlock::Dense(trailing);
        let gradient = ProfiledGradientInputs {
            a_blocks: &self.kernel.structural_a,
            l_blocks: &self.l_blocks,
            reterms: &self.reterms,
            dims: ModelDims {
                n: self.kernel.n,
                p,
                nretrms: k,
            },
            reml,
            sigma: None,
            fe_blocks: Some(&self.gradient_fe),
            trailing_l: Some(&self.gradient_trailing_l),
        }
        .profiled_gradient()?;
        Ok((profile.total_objective, gradient))
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
        &mut self,
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
                let this: &LmmWorkspace<'_> = &*self;
                chunks
                    .par_iter()
                    .map_init(ProfileScratch::default, |scratch, chunk| {
                        this.profile_with_scratch(
                            &select_response_columns(responses, chunk),
                            reml,
                            scratch,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?
            }
            #[cfg(not(feature = "rayon"))]
            {
                return Err(MixedModelError::InvalidArgument(
                    "parallel batch profiling requires the `rayon` cargo feature".to_string(),
                ));
            }
        } else {
            let mut scratch = std::mem::take(&mut self.scratch);
            let profiles = chunks
                .iter()
                .map(|chunk| {
                    self.profile_with_scratch(
                        &select_response_columns(responses, chunk),
                        reml,
                        &mut scratch,
                    )
                })
                .collect::<Result<Vec<_>>>();
            self.scratch = scratch;
            profiles?
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

#[cfg(test)]
mod gradient_oracle_tests {
    use super::*;
    use crate::formula::parse_formula;
    use crate::model::data::DataFrame;
    use crate::model::traits::MixedModelFit;

    /// The batch column oracle (structural blocks augmented with the
    /// column's `Zᵀy`, `Xᵀy`, `β`, `pwrss`) must reproduce the model's own
    /// objective and analytic gradient for the same response.
    #[test]
    fn column_oracle_matches_the_model_gradient() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        use rand_distr::{Distribution, Normal};
        let mut rng = StdRng::seed_from_u64(5);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let (mut reaction, mut days, mut subj) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..24 {
            let u0 = normal.sample(&mut rng);
            let u1 = normal.sample(&mut rng);
            for d in 0..8 {
                let x = d as f64;
                reaction.push(
                    250.0 + 10.0 * x + 20.0 * u0 + 4.0 * u1 * x + 20.0 * normal.sample(&mut rng),
                );
                days.push(x);
                subj.push(format!("S{i:02}"));
            }
        }
        let mut df = DataFrame::new();
        df.add_numeric("reaction", reaction).unwrap();
        df.add_numeric("days", days).unwrap();
        df.add_categorical("subj", subj).unwrap();
        for (formula, reml) in [
            ("reaction ~ 1 + days + (1 + days | subj)", true),
            ("reaction ~ 1 + days + (1 + days | subj)", false),
            ("reaction ~ 1 + days + (1 | subj)", true),
        ] {
            let mut model =
                LinearMixedModel::new(parse_formula(formula).unwrap(), &df, None).unwrap();
            model.fit(reml).unwrap();
            let kernel = LmmObjectiveKernel::from_model(&model);
            let kernel = kernel.unwrap();
            let mut workspace = kernel.workspace();
            let y =
                DMatrix::from_column_slice(model.response().len(), 1, model.response().as_slice());
            for theta in [
                model.theta(),
                model.theta().iter().map(|t| t + 0.2).collect::<Vec<_>>(),
            ] {
                let (objective, gradient) = workspace
                    .objective_and_gradient_for_column(&theta, &y, reml)
                    .unwrap();
                let (model_objective, model_gradient) =
                    model.objective_and_gradient_at(&theta).unwrap();
                assert!(
                    (objective - model_objective).abs() <= 1e-8 * model_objective.abs().max(1.0),
                    "{formula} reml={reml}: objective {objective} vs {model_objective}"
                );
                for (k, (a, b)) in gradient.iter().zip(&model_gradient).enumerate() {
                    assert!(
                        (a - b).abs() <= 1e-8 * b.abs().max(1.0),
                        "{formula} reml={reml}: gradient[{k}] {a} vs {b}"
                    );
                }
            }
        }
    }
}
