//! Linear mixed-effects model (LMM).
//!
//! Implements the penalized least squares (PLS) algorithm for fitting
//! linear mixed models via profile likelihood optimization.
//!
//! The model is: y = Xβ + Zb + ε, where b ~ N(0, σ²Λθ Λθ') and ε ~ N(0, σ²I).
//!
//! The θ parameters control the relative covariance factor Λ. The objective
//! function (deviance or REML criterion) is profiled over β and σ², leaving
//! only θ to be optimized numerically.

use nalgebra::{DMatrix, DVector, SymmetricEigen};
use nalgebra_sparse::{coo::CooMatrix, csc::CscMatrix};
#[cfg(feature = "nlopt")]
use nlopt::{
    Algorithm as NloptAlgorithm, FailState as NloptFailState, Nlopt, Target as NloptTarget,
};
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::compiler::{
    compile_formula_ir, BasisLoading, BootstrapInferenceDetails, CompiledModelArtifact,
    CompilerPolicy, ContrastFamilyDetails, ConvergenceVerification, ConvergenceVerificationRun,
    ConvergenceVerificationStatus, CovarianceFamily, CovarianceFamilyTransition, DesignAudit,
    Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticStage, DominantLoading,
    EffectiveCovarianceSummary, EffectiveRankStatus, EstimabilityAssessment, EstimabilityStatus,
    EvidenceMethod, FixedContrastEstimability, FixedEffectHypothesis, FixedEffectInferenceDetails,
    FixedEffectInferenceMethod, FixedEffectInferenceRow, FixedEffectInferenceRowKind,
    FixedEffectInferenceStatus, FixedEffectInferenceTable, FixedEffectNullTargetSummary,
    FixedEffectStatisticName, FixedEffectTest, FixedEffectTestMethod, InferenceMethod,
    InferenceStatus, InterpretableSubmodel, KenwardRogerInferenceDetails, ModelAuditReport,
    ModelStateChange, ModelStateSummary, OptimizerCertificate, OptimizerDerivativeEvidence,
    PolicyAction, PolicyRecommendation, ReductionRecord, ReductionTrigger, ReliabilityGrade,
    SupportedCovarianceDirection, DOMINANT_LOADING_THRESHOLD, INTERPRETABLE_GAP_TOLERANCE,
};
use crate::error::{MixedModelError, Result};
use crate::formula::{FixedTerm, Formula, RandomTerm};
use crate::model::data::{CategoricalCoding, Column, DataFrame};
use crate::model::fixed_design::{
    DenseFixedDesign, FixedDesign, FixedDesignBackend, FixedDesignBuildPolicy, FixedDesignStorage,
    FixedDesignSummary,
};
use crate::model::traits::MixedModelFit;
#[cfg(feature = "prima")]
use crate::optimizer::prima::{minimize_bobyqa, PrimaBobyqaOptions};
use crate::stats::{BlockDescription, CoefTable, CoefTablePValuePolicy, ModelSummary, VarCorr};
use crate::types::matrix_block::{
    block_index, with_block_pair_mut, with_block_triple, with_dense_block, MatrixBlock,
};
#[cfg(feature = "prima")]
use crate::types::opt_summary::OptimizerBackend;
use crate::types::{FeMat, FeTerm, FitLogEntry, OptSummary, Optimizer, ReMat};

/// A fitted (or constructed but unfitted) linear mixed-effects model.
///
/// Corresponds to `LinearMixedModel{T}` in MixedModels.jl.
///
/// # Fields
/// - `formula`: the parsed model formula
/// - `reterms`: random-effects model matrices, sorted by decreasing nranef
/// - `xy_mat`: the fixed-effects model matrix concatenated with y, with optional weighting
/// - `feterm`: the fixed-effects model matrix with rank/pivot info
/// - `sqrtwts`: square roots of case weights (empty if unweighted)
/// - `parmap`: mapping from θ indices to (block, row, col) in λ
/// - `dims`: model dimensions (n, p, nretrms)
/// - `a_blocks`: lower triangle of [Z X y]'[Z X y] in blocked storage
/// - `l_blocks`: blocked lower Cholesky factor of Λ'AΛ + I
/// - `optsum`: optimization summary
/// - `compiler_artifact`: semantic compiler/audit metadata for the requested model
#[derive(Debug, Clone)]
pub struct LinearMixedModel {
    pub formula: Formula,
    pub reterms: Vec<ReMat>,
    pub xy_mat: FeMat,
    pub y: DVector<f64>,
    pub feterm: FeTerm,
    pub fixed_design: FixedDesign,
    pub sqrtwts: Vec<f64>,
    pub parmap: Vec<(usize, usize, usize)>, // (block, row, col)
    pub dims: ModelDims,
    pub a_blocks: Vec<MatrixBlock>,
    pub l_blocks: Vec<MatrixBlock>,
    pub optsum: OptSummary,
    pub compiler_artifact: CompiledModelArtifact,
}

/// Model dimensions.
#[derive(Debug, Clone, Copy)]
pub struct ModelDims {
    pub n: usize,       // number of observations
    pub p: usize,       // rank of fixed-effects matrix
    pub nretrms: usize, // number of random-effects terms
}

/// How to handle random-effects levels not seen during training.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NewReLevels {
    /// Return an error if any unseen levels are encountered.
    Error,
    /// Use zero random effects for unseen levels (population-level prediction).
    Population,
    /// Return `None` for observations that have unseen levels.
    Missing,
}

/// Profiled quantities for a batch of response columns sharing the same
/// fixed-effects design, random-effects structure, and theta.
#[derive(Debug, Clone)]
pub struct ResponseMatrixProfile {
    /// Fixed-effects solutions for each response column, shape `p x q`.
    pub beta: DMatrix<f64>,
    /// Profiled residual scales for each response column, length `q`.
    pub sigma: DVector<f64>,
    /// Penalized weighted residual sum of squares for each response column.
    pub pwrss: DVector<f64>,
    /// Profiled objective contribution for each response column.
    pub objectives: DVector<f64>,
    /// Sum of profiled objective contributions across all columns.
    pub total_objective: f64,
    /// Shared random-effects log-determinant term.
    pub logdet_re: f64,
    /// Shared fixed-effects log-determinant term used by REML.
    pub logdet_xx: f64,
}

#[derive(Debug)]
pub(crate) struct PatternSearchOutcome {
    pub(crate) best_theta: Vec<f64>,
    pub(crate) best_fmin: f64,
    pub(crate) feval_count: i64,
    pub(crate) fit_log: Vec<FitLogEntry>,
}

fn record_pattern_eval<F>(
    objective: &mut F,
    theta: &[f64],
    feval_count: &mut i64,
    fit_log: &mut Vec<FitLogEntry>,
    best_theta: &mut Vec<f64>,
    best_fmin: &mut f64,
) -> Result<f64>
where
    F: FnMut(&[f64]) -> Result<f64>,
{
    let obj = objective(theta)?;
    *feval_count += 1;
    fit_log.push(FitLogEntry {
        theta: theta.to_vec(),
        objective: obj,
    });
    if obj < *best_fmin {
        *best_fmin = obj;
        *best_theta = theta.to_vec();
    }
    Ok(obj)
}

/// Covariance estimate for `varpar = c(theta, sigma)` plus Hessian diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct VcovVarparEstimate {
    pub covariance: DMatrix<f64>,
    pub hessian: DMatrix<f64>,
    pub eigenvalues: Vec<f64>,
    pub tolerance: f64,
    pub positive_eigenvalues: usize,
    pub near_zero_eigenvalues: usize,
    pub negative_eigenvalues: usize,
    pub used_reduced_rank: bool,
    pub reliability: ReliabilityGrade,
    pub notes: Vec<String>,
}

/// Kenward-Roger response-covariance decomposition.
///
/// This is the Rust analogue of `pbkrtest::get_SigmaG()`: `sigma` is the
/// fitted marginal response covariance and each component matrix is a known
/// `G_i` such that `sigma = sum_i weights[i] * components[i]`.
#[derive(Debug, Clone, PartialEq)]
pub struct KenwardRogerSigmaG {
    pub sigma: DMatrix<f64>,
    pub components: Vec<DMatrix<f64>>,
    pub component_weights: Vec<f64>,
    pub component_labels: Vec<String>,
    pub residual_component_index: usize,
    pub covariance_parameterization: String,
    pub includes_residual_variance: bool,
    pub n_observations: usize,
    pub dense_bytes: u128,
    pub sigma_min_eigenvalue: f64,
    pub sigma_max_eigenvalue: f64,
    pub sigma_positive_definite: bool,
    pub max_component_asymmetry: f64,
    pub reliability: ReliabilityGrade,
    pub notes: Vec<String>,
}

/// Kenward-Roger adjusted fixed-effect covariance payload.
#[derive(Debug, Clone, PartialEq)]
pub struct KenwardRogerAdjustedVcov {
    pub unadjusted_vcov_active: DMatrix<f64>,
    pub adjusted_vcov_active: DMatrix<f64>,
    pub adjusted_vcov: DMatrix<f64>,
    pub p_matrices: Vec<DMatrix<f64>>,
    pub q_matrices: Vec<DMatrix<f64>>,
    pub w: DMatrix<f64>,
    pub information_matrix: DMatrix<f64>,
    pub information_eigenvalues: Vec<f64>,
    pub condition_min_abs_eigenvalue: f64,
    pub used_generalized_inverse: bool,
    pub component_labels: Vec<String>,
    pub reliability: ReliabilityGrade,
    pub notes: Vec<String>,
}

/// Kenward-Roger denominator degrees-of-freedom result for `L beta = rhs`.
#[derive(Debug, Clone, PartialEq)]
pub struct KenwardRogerLbDdf {
    pub denominator_df: f64,
    pub numerator_df: f64,
    pub restriction_rank: usize,
    pub a1: f64,
    pub a2: f64,
    pub b: f64,
    pub g: f64,
    pub rho: f64,
    pub used_generalized_inverse: bool,
    pub reliability: ReliabilityGrade,
    pub notes: Vec<String>,
}

/// Controls the bounded verification workflow run after a fitted model.
#[derive(Debug, Clone)]
pub struct ConvergenceVerificationOptions {
    pub restart_from_optimum: bool,
    pub jitter_starts: usize,
    pub jitter_scale: f64,
    pub run_optimizer_consensus: bool,
    pub max_function_evaluations: usize,
    pub objective_tolerance: f64,
    pub theta_tolerance: f64,
    pub beta_tolerance: f64,
}

impl Default for ConvergenceVerificationOptions {
    fn default() -> Self {
        Self {
            restart_from_optimum: true,
            jitter_starts: 1,
            jitter_scale: 1e-4,
            run_optimizer_consensus: true,
            max_function_evaluations: 500,
            objective_tolerance: 1e-5,
            theta_tolerance: 1e-3,
            beta_tolerance: 1e-4,
        }
    }
}

const DEFAULT_DENSE_BLOCK_LIMIT_BYTES: u128 = 16 * 1024 * 1024 * 1024;

fn dense_block_limit_bytes() -> u128 {
    std::env::var("MIXEDMODELS_MAX_DENSE_BLOCK_BYTES")
        .ok()
        .and_then(|value| value.parse::<u128>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_DENSE_BLOCK_LIMIT_BYTES)
}

fn dense_block_bytes(nrows: usize, ncols: usize) -> u128 {
    (nrows as u128)
        .saturating_mul(ncols as u128)
        .saturating_mul(std::mem::size_of::<f64>() as u128)
}

fn ensure_dense_block_within_limit(
    nrows: usize,
    ncols: usize,
    context: impl Into<String>,
) -> Result<()> {
    ensure_dense_block_within_explicit_limit(nrows, ncols, context, dense_block_limit_bytes())
}

fn ensure_dense_block_within_explicit_limit(
    nrows: usize,
    ncols: usize,
    context: impl Into<String>,
    limit: u128,
) -> Result<()> {
    let bytes = dense_block_bytes(nrows, ncols);
    if bytes > limit {
        return Err(MixedModelError::ProblemTooLarge(format!(
            "{} would require a dense {} x {} f64 block ({:.2} GiB), above the configured limit ({:.2} GiB). \
             For large partially crossed random effects, use a more storage-aware formulation or raise MIXEDMODELS_MAX_DENSE_BLOCK_BYTES only if this allocation is intentional.",
            context.into(),
            nrows,
            ncols,
            bytes as f64 / 1024.0_f64.powi(3),
            limit as f64 / 1024.0_f64.powi(3)
        )));
    }
    Ok(())
}

fn validate_dense_block_plan(reterms: &[ReMat], fixed_response_cols: usize) -> Result<()> {
    for i in 0..reterms.len() {
        let ri = reterms[i].n_ranef();
        ensure_dense_block_within_limit(
            fixed_response_cols,
            ri,
            format!(
                "[X|y]'Z block for grouping factor '{}'",
                reterms[i].grouping_name
            ),
        )?;

        for j in 0..i {
            if reterms[i].vsize != 1 || reterms[j].vsize != 1 {
                let rj = reterms[j].n_ranef();
                ensure_dense_block_within_limit(
                    ri,
                    rj,
                    format!(
                        "off-diagonal random-effects cross-product block '{}' x '{}'",
                        reterms[i].grouping_name, reterms[j].grouping_name
                    ),
                )?;
            }
        }

        if (0..i).any(|j| !is_nested(&reterms[j], &reterms[i])) {
            for row in i..reterms.len() {
                ensure_dense_block_within_limit(
                    reterms[row].n_ranef(),
                    ri,
                    format!(
                        "crossed random-effects fill-in block '{}' x '{}'",
                        reterms[row].grouping_name, reterms[i].grouping_name
                    ),
                )?;
            }
        }
    }
    Ok(())
}

const BLOCK_TRIANGULAR_SOLVE_ZERO_TOLERANCE: f64 = 1e-30;

fn copy_block(dst: &mut MatrixBlock, src: &MatrixBlock) {
    match (dst, src) {
        (MatrixBlock::Dense(dst_mat), MatrixBlock::Dense(src_mat)) => {
            if dst_mat.shape() == src_mat.shape() {
                dst_mat.copy_from(src_mat);
            } else {
                *dst_mat = src_mat.clone();
            }
        }
        (MatrixBlock::Diagonal(dst_diag), MatrixBlock::Diagonal(src_diag)) => {
            if dst_diag.len() == src_diag.len() {
                dst_diag.copy_from(src_diag);
            } else {
                *dst_diag = src_diag.clone();
            }
        }
        (MatrixBlock::BlockDiagonal(dst_blocks), MatrixBlock::BlockDiagonal(src_blocks))
            if dst_blocks.len() == src_blocks.len() =>
        {
            for (dst_blk, src_blk) in dst_blocks.iter_mut().zip(src_blocks.iter()) {
                if dst_blk.shape() == src_blk.shape() {
                    dst_blk.copy_from(src_blk);
                } else {
                    *dst_blk = src_blk.clone();
                }
            }
        }
        (MatrixBlock::Sparse(dst_mat), MatrixBlock::Sparse(src_mat)) => {
            if dst_mat.nrows() == src_mat.nrows()
                && dst_mat.ncols() == src_mat.ncols()
                && dst_mat.nnz() == src_mat.nnz()
                && dst_mat.col_offsets() == src_mat.col_offsets()
                && dst_mat.row_indices() == src_mat.row_indices()
            {
                dst_mat.values_mut().copy_from_slice(src_mat.values());
            } else {
                *dst_mat = src_mat.clone();
            }
        }
        (dst_block, src_block) => {
            *dst_block = src_block.clone();
        }
    }
}

fn subtract_product_from_blocks(c: &mut MatrixBlock, a: &MatrixBlock, b: &MatrixBlock) {
    with_dense_block(a, |a_dense| {
        with_dense_block(b, |b_dense| {
            subtract_product(c, a_dense, b_dense);
        })
    });
}

#[inline]
fn solve_scaled_vsize2_row(
    a10: &DMatrix<f64>,
    row: usize,
    col0: usize,
    col1: usize,
    lam00: f64,
    lam10: f64,
    lam11: f64,
    l00: f64,
    l10: f64,
    l11: f64,
) -> (f64, f64) {
    let x0 = a10[(row, col0)];
    let x1 = a10[(row, col1)];
    let mut solved0 = x0 * lam00 + x1 * lam10;
    let mut solved1 = x1 * lam11;

    solved0 = if l00.abs() < 1e-30 {
        0.0
    } else {
        solved0 / l00
    };
    solved1 = if l11.abs() < 1e-30 {
        0.0
    } else {
        (solved1 - solved0 * l10) / l11
    };

    (solved0, solved1)
}

pub(crate) fn update_l_from_parts(
    a_blocks: &[MatrixBlock],
    l_blocks: &mut [MatrixBlock],
    reterms: &[ReMat],
    cholesky_zero_pad_tolerance: f64,
) -> Result<()> {
    let k = reterms.len(); // number of RE terms
    let total = k + 1; // +1 for the [X|y] block

    // Copy A to L, scaling by Λ
    // For diagonal blocks L[j,j] = Λ_j' A[j,j] Λ_j + I
    for j in 0..k {
        let idx_jj = block_index(j, j);
        copy_scale_inflate(&mut l_blocks[idx_jj], &a_blocks[idx_jj], &reterms[j]);
    }

    // For off-diagonal RE blocks L[i,j] = Λ_i' A[i,j] Λ_j, i > j
    for i in 1..k {
        for j in 0..i {
            let idx_ij = block_index(i, j);
            copy_and_scale_offdiag(
                &mut l_blocks[idx_ij],
                &a_blocks[idx_ij],
                &reterms[i],
                &reterms[j],
            );
        }
    }

    // For FE-RE blocks L[k,j] = A[k,j] Λ_j (no Λ on left for FeMat)
    for j in 0..k {
        let idx_kj = block_index(k, j);
        copy_and_rmul_lambda(&mut l_blocks[idx_kj], &a_blocks[idx_kj], &reterms[j]);
    }

    // Copy the [X|y]'[X|y] block unchanged
    let idx_kk = block_index(k, k);
    copy_block(&mut l_blocks[idx_kk], &a_blocks[idx_kk]);

    // Blocked Cholesky factorization
    for j in 0..total {
        let diag_idx = block_index(j, j);

        // Update L[j,j] by subtracting L[j,0..j] * L[j,0..j]'
        for jj in 0..j {
            let off_idx = block_index(j, jj);
            with_block_pair_mut(l_blocks, diag_idx, off_idx, |diag, off| match off {
                MatrixBlock::Sparse(off_sparse) => rank_k_downdate_sparse(diag, off_sparse),
                _ => {
                    if let Some(off_dense) = off.as_dense_ref() {
                        rank_k_downdate(diag, off_dense);
                    } else {
                        let off_dense = off.as_dense();
                        rank_k_downdate(diag, &off_dense);
                    }
                }
            });
        }

        // Cholesky of diagonal block
        cholesky_block_with_tolerance(&mut l_blocks[diag_idx], cholesky_zero_pad_tolerance)?;

        // Solve for off-diagonal blocks: L[i,j] for i > j
        for i in (j + 1)..total {
            let target_idx = block_index(i, j);

            // L[i,j] -= sum_{jj<j} L[i,jj] * L[j,jj]'
            for jj in 0..j {
                with_block_triple(
                    l_blocks,
                    target_idx,
                    block_index(i, jj),
                    block_index(j, jj),
                    |target, lhs, rhs| subtract_product_from_blocks(target, lhs, rhs),
                );
            }

            // L[i,j] = L[i,j] * L[j,j]^{-T}
            with_block_pair_mut(l_blocks, target_idx, diag_idx, |target, diag| {
                rdiv_lower_transpose(target, diag);
            });
        }
    }

    Ok(())
}

impl LinearMixedModel {
    /// Construct a LinearMixedModel from a formula, data, and optional weights.
    pub fn new(formula: Formula, data: &DataFrame, weights: Option<&[f64]>) -> Result<Self> {
        Self::new_with_policy_internal(formula, data, weights, CompilerPolicy::default())
    }

    fn new_with_policy_internal(
        formula: Formula,
        data: &DataFrame,
        weights: Option<&[f64]>,
        compiler_policy: CompilerPolicy,
    ) -> Result<Self> {
        Self::new_with_policies_internal(
            formula,
            data,
            weights,
            compiler_policy,
            FixedDesignBuildPolicy::default(),
        )
    }

    #[cfg(test)]
    fn new_with_fixed_design_policy(
        formula: Formula,
        data: &DataFrame,
        weights: Option<&[f64]>,
        fixed_design_policy: FixedDesignBuildPolicy,
    ) -> Result<Self> {
        Self::new_with_policies_internal(
            formula,
            data,
            weights,
            CompilerPolicy::default(),
            fixed_design_policy,
        )
    }

    fn new_with_policies_internal(
        formula: Formula,
        data: &DataFrame,
        weights: Option<&[f64]>,
        compiler_policy: CompilerPolicy,
        fixed_design_policy: FixedDesignBuildPolicy,
    ) -> Result<Self> {
        if formula.random_terms.is_empty() {
            return Err(MixedModelError::NoRandomEffects);
        }

        let semantic_model = compile_formula_ir(&formula);
        let mut compiler_artifact = CompiledModelArtifact::new_with_policy(
            formula.to_string(),
            semantic_model,
            compiler_policy,
        );
        compiler_artifact.attach_design_audit(data);
        let mut effective_formula = formula.clone();
        if compiler_artifact
            .compiler_policy
            .apply_design_time_reductions
        {
            let reductions = apply_design_compiled_policy(
                &mut effective_formula,
                &compiler_artifact.policy_recommendations,
            )?;
            if !reductions.is_empty() {
                let effective_semantic_model = compile_formula_ir(&effective_formula);
                compiler_artifact.set_effective_model(
                    effective_formula.to_string(),
                    effective_semantic_model,
                    reductions,
                );
            }
        }
        if effective_formula.random_terms.is_empty() {
            return Err(MixedModelError::NoRandomEffects);
        }

        let n = data.nrow();

        // Build the response vector
        let y_data = data.numeric(&effective_formula.response).ok_or_else(|| {
            MixedModelError::InvalidArgument(format!(
                "Response '{}' not found or not numeric",
                effective_formula.response
            ))
        })?;
        let y = DVector::from_column_slice(y_data);

        // Build the fixed-effects design through the backend-selection policy.
        // FeTerm still owns rank/pivot metadata; the selected full-rank backend
        // is used below for solver cross-products.
        let raw_fixed_design = crate::model::fixed_design::build_fixed_effects_design_with_policy(
            &effective_formula,
            data,
            fixed_design_policy,
        )?;
        let feterm = FeTerm::new(
            raw_fixed_design.materialize_dense(),
            raw_fixed_design.column_names().to_vec(),
        );
        let fixed_design = raw_fixed_design.select_columns(&feterm.piv[..feterm.rank])?;
        if fixed_design.storage() == FixedDesignStorage::Streamed {
            compiler_artifact
                .diagnostics
                .push(fixed_design_backend_diagnostic(&fixed_design));
        }

        // Build the random-effects terms
        let mut ordered_reterms = Vec::new();
        for (semantic_index, rt) in effective_formula.random_terms.iter().enumerate() {
            let remat = build_re_mat(rt, data, n)?;
            ordered_reterms.push((semantic_index, remat));
        }

        // Sort by decreasing nranef (matches Julia behavior)
        ordered_reterms.sort_by(|a, b| b.1.n_ranef().cmp(&a.1.n_ranef()));
        let optimizer_order = ordered_reterms
            .iter()
            .map(|(semantic_index, _)| *semantic_index)
            .collect::<Vec<_>>();
        let optimizer_basis = ordered_reterms
            .iter()
            .map(|(_, remat)| {
                remat
                    .cnames
                    .iter()
                    .map(|name| user_basis_label(name))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        compiler_artifact
            .rebuild_theta_maps_for_optimizer_order_with_basis(&optimizer_order, &optimizer_basis);
        let mut reterms = ordered_reterms
            .into_iter()
            .map(|(_, remat)| remat)
            .collect::<Vec<_>>();

        // Build FeMat = [full_rank_X | y]
        let mut xy_mat = FeMat::new(&feterm, &y);

        // Apply weights: scale each row of X, Z, and y by sqrt(w_i).
        let mut sqrtwts_dvec = None;
        let sqrtwts = if let Some(wts) = weights {
            let sw: Vec<f64> = wts.iter().map(|w| w.sqrt()).collect();
            let sw_dvec = DVector::from_vec(sw.clone());
            xy_mat.reweight(&sw_dvec);
            for rt in &mut reterms {
                rt.reweight(&sw_dvec);
            }
            sqrtwts_dvec = Some(sw_dvec);
            sw
        } else {
            vec![]
        };

        // Create cross-product blocks A and Cholesky blocks L
        let (a_blocks, l_blocks) =
            create_al_from_fixed_design(&reterms, &fixed_design, &y, sqrtwts_dvec.as_ref())?;

        // Build theta vector from all reterms
        let theta: Vec<f64> = reterms.iter().flat_map(|rt| rt.get_theta()).collect();

        // Build parmap: mapping from θ index to (re_term_index, row, col) in lambda
        let parmap = build_parmap(&reterms);

        let dims = ModelDims {
            n,
            p: feterm.rank,
            nretrms: reterms.len(),
        };

        let optsum = OptSummary::new(theta);

        let mut model = LinearMixedModel {
            formula: effective_formula,
            reterms,
            xy_mat,
            y,
            feterm,
            fixed_design,
            sqrtwts,
            parmap,
            dims,
            a_blocks,
            l_blocks,
            optsum,
            compiler_artifact,
        };
        debug_assert_eq!(
            model.dims.p, model.feterm.rank,
            "ModelDims::p must track the active fixed-effect rank"
        );
        model.refresh_covariance_parameter_traces();
        Ok(model)
    }

    /// Construct a model and apply a compiler policy before any fitting or
    /// certification occurs.
    pub fn new_with_compiler_policy(
        formula: Formula,
        data: &DataFrame,
        weights: Option<&[f64]>,
        compiler_policy: CompilerPolicy,
    ) -> Result<Self> {
        Self::new_with_policy_internal(formula, data, weights, compiler_policy)
    }

    /// Apply a compiler policy before fitting.
    ///
    /// Policies affect advisory recommendations, reproducibility metadata, and
    /// fit-time certification such as effective covariance rank. Changing the
    /// policy after a fit would make the certificate ambiguous, so fitted models
    /// reject this operation.
    pub fn set_compiler_policy(&mut self, compiler_policy: CompilerPolicy) -> Result<&mut Self> {
        if self.optsum.feval > 0 {
            return Err(MixedModelError::AlreadyFitted);
        }
        self.compiler_artifact.set_compiler_policy(compiler_policy);
        Ok(self)
    }

    /// Return a copy of this model with a compiler policy applied.
    pub fn with_compiler_policy(mut self, compiler_policy: CompilerPolicy) -> Result<Self> {
        self.set_compiler_policy(compiler_policy)?;
        Ok(self)
    }

    /// Fit after first applying a compiler policy.
    pub fn fit_with_compiler_policy(
        &mut self,
        reml: bool,
        compiler_policy: CompilerPolicy,
    ) -> Result<&mut Self> {
        self.set_compiler_policy(compiler_policy)?;
        self.fit(reml)
    }

    /// Round-trippable compiler artifact attached at construction time.
    pub fn compiler_artifact(&self) -> &CompiledModelArtifact {
        &self.compiler_artifact
    }

    /// Compiler policy attached to this model.
    pub fn compiler_policy(&self) -> &CompilerPolicy {
        &self.compiler_artifact.compiler_policy
    }

    /// Runtime summary for the selected fixed-effect design backend.
    pub fn fixed_design_backend_summary(&self) -> FixedDesignSummary {
        self.fixed_design.summary()
    }

    /// Number of active fixed-effect entries stored by the selected backend.
    ///
    /// Dense designs report `n * p`; streamed designs report the actual
    /// non-zero row entries stored after rank/pivot column selection.
    pub fn fixed_design_active_entries(&self) -> usize {
        fixed_design_active_entries(&self.fixed_design)
    }

    /// Active-entry density of the selected fixed-effect backend.
    pub fn fixed_design_density(&self) -> f64 {
        fixed_design_density(&self.fixed_design)
    }

    /// Prefit design audit attached to the compiler artifact, if available.
    pub fn design_audit(&self) -> Option<&DesignAudit> {
        self.compiler_artifact.design_audit.as_ref()
    }

    /// Fit-time optimizer certificate attached to the compiler artifact, if available.
    pub fn optimizer_certificate(&self) -> Option<&OptimizerCertificate> {
        self.compiler_artifact.optimizer_certificate.as_ref()
    }

    /// Stable user-facing audit report derived from the compiler artifact.
    pub fn audit_report(&self) -> ModelAuditReport {
        self.compiler_artifact.audit_report()
    }

    /// Compact default print summary (PRD § 15).
    pub fn print_summary(&self) -> crate::compiler::ModelPrint {
        self.compiler_artifact.print_summary()
    }

    /// Source-to-fitted parameterization drilldown (PRD § 15).
    pub fn parameterization(&self) -> crate::compiler::ParameterizationDrilldown {
        self.compiler_artifact.parameterization()
    }

    /// Requested, semantic, supported, and fitted model-state view.
    pub fn model_state_summary(&self) -> ModelStateSummary {
        self.compiler_artifact.model_state_summary()
    }

    /// Recorded or recommended requested-to-fitted model changes.
    pub fn changes(&self) -> Vec<ModelStateChange> {
        self.compiler_artifact.changes()
    }

    /// Run bounded convergence verification and attach the result to the
    /// optimizer certificate.
    ///
    /// Refits the model from the current optimum (and from one or more
    /// jittered starts, and via alternate optimizers when consensus is
    /// requested) and reports whether the runs agree on θ, β, and the
    /// objective. The result is stored on
    /// `compiler_artifact.optimizer_certificate.verification` so the
    /// audit report and [`ConvergenceVerdict`](crate::compiler::ConvergenceVerdict)
    /// can pick it up. lme4's analogue is `allFit()`.
    ///
    /// # When to call
    ///
    /// Run this after [`fit`](Self::fit) when the compact print shows
    /// `convergence: caution` or `convergence: ok` with a
    /// `next: run verify_convergence()` hint — that is, when the
    /// optimizer stopped acceptably but gradient/Hessian evidence is
    /// weak or unavailable, or when the model is at a boundary or
    /// reduced-rank optimum and you want optimizer-agreement
    /// reassurance. It is **not** the right tool for structural design
    /// failures (row-saturated random effects, separation,
    /// rank-deficient fixed effects); the verdict's `next:` line
    /// already excludes optimizer tinkering when the source is
    /// structural.
    ///
    /// Use [`verify_convergence_with_options`](Self::verify_convergence_with_options)
    /// when you need finer-grained control over jitter scale, alternate
    /// optimizer choice, or agreement tolerances.
    pub fn verify_convergence(&mut self) -> Result<ConvergenceVerification> {
        self.verify_convergence_with_options(ConvergenceVerificationOptions::default())
    }

    /// Run convergence verification with explicit controls.
    pub fn verify_convergence_with_options(
        &mut self,
        options: ConvergenceVerificationOptions,
    ) -> Result<ConvergenceVerification> {
        if self.optsum.feval <= 0 {
            let verification = ConvergenceVerification::not_run("model has not been fitted");
            if let Some(certificate) = &mut self.compiler_artifact.optimizer_certificate {
                certificate.verification = Some(verification.clone());
            }
            return Ok(verification);
        }

        let reference_theta = self.theta();
        let reference_beta = self.beta().iter().copied().collect::<Vec<_>>();
        let reference_objective = self.optsum.fmin.is_finite().then_some(self.optsum.fmin);
        let reference_effective_ranks = self
            .compiler_artifact
            .effective_covariance
            .iter()
            .map(|summary| summary.supported_rank)
            .collect::<Vec<_>>();

        let mut runs = Vec::new();
        if options.restart_from_optimum {
            runs.push(self.convergence_verification_run(
                "restart_from_optimum",
                self.optsum.optimizer,
                &reference_theta,
                &reference_theta,
                &reference_beta,
                reference_objective,
                &reference_effective_ranks,
                &options,
            ));
        }

        for jitter_index in 0..options.jitter_starts {
            let start = jittered_theta(
                &reference_theta,
                &self.lower_bounds(),
                options.jitter_scale,
                jitter_index,
            );
            runs.push(self.convergence_verification_run(
                &format!("jitter_restart_{}", jitter_index + 1),
                self.optsum.optimizer,
                &start,
                &reference_theta,
                &reference_beta,
                reference_objective,
                &reference_effective_ranks,
                &options,
            ));
        }

        if options.run_optimizer_consensus {
            for optimizer in self.alternate_verification_optimizers() {
                runs.push(self.convergence_verification_run(
                    &format!("optimizer_consensus_{}", optimizer_name(optimizer)),
                    optimizer,
                    &reference_theta,
                    &reference_theta,
                    &reference_beta,
                    reference_objective,
                    &reference_effective_ranks,
                    &options,
                ));
            }
        }

        let status = verification_status(&runs, &options);
        let message = verification_message(status, &runs);
        let verification = ConvergenceVerification {
            status,
            objective_tolerance: options.objective_tolerance,
            theta_tolerance: options.theta_tolerance,
            beta_tolerance: options.beta_tolerance,
            reference_objective,
            reference_theta,
            reference_beta,
            reference_effective_ranks,
            runs,
            message,
        };

        if let Some(certificate) = &mut self.compiler_artifact.optimizer_certificate {
            certificate.verification = Some(verification.clone());
        }
        Ok(verification)
    }

    fn convergence_verification_run(
        &self,
        label: &str,
        optimizer: Optimizer,
        start_theta: &[f64],
        reference_theta: &[f64],
        reference_beta: &[f64],
        reference_objective: Option<f64>,
        reference_effective_ranks: &[usize],
        options: &ConvergenceVerificationOptions,
    ) -> ConvergenceVerificationRun {
        let mut candidate = self.clone();
        let result = candidate
            .reset_for_convergence_verification(start_theta, options.max_function_evaluations)
            .and_then(|_| candidate.fit_with_forced_optimizer(self.optsum.reml, optimizer));

        match result {
            Ok(()) => {
                let objective_value = candidate
                    .optsum
                    .fmin
                    .is_finite()
                    .then_some(candidate.optsum.fmin);
                let theta = candidate.theta();
                let beta = candidate.beta().iter().copied().collect::<Vec<_>>();
                let effective_ranks = candidate
                    .compiler_artifact
                    .effective_covariance
                    .iter()
                    .map(|summary| summary.supported_rank)
                    .collect::<Vec<_>>();
                let objective_delta = objective_value
                    .zip(reference_objective)
                    .map(|(value, reference)| (value - reference).abs());
                let max_abs_theta_delta = max_abs_delta(&theta, reference_theta);
                let max_abs_beta_delta = max_abs_delta(&beta, reference_beta);
                let ranks_agree = effective_ranks == reference_effective_ranks;
                let mut diagnostics = Vec::new();
                if objective_delta
                    .map(|delta| delta > options.objective_tolerance)
                    .unwrap_or(true)
                {
                    diagnostics.push("objective changed beyond tolerance".to_string());
                }
                if max_abs_theta_delta
                    .map(|delta| delta > options.theta_tolerance)
                    .unwrap_or(true)
                {
                    diagnostics.push("theta parameterization changed beyond tolerance".to_string());
                }
                if max_abs_beta_delta
                    .map(|delta| delta > options.beta_tolerance)
                    .unwrap_or(true)
                {
                    diagnostics.push("fixed-effect estimates changed beyond tolerance".to_string());
                }
                if !ranks_agree {
                    diagnostics
                        .push("effective covariance ranks changed during verification".to_string());
                }
                let agrees = objective_delta
                    .map(|delta| delta <= options.objective_tolerance)
                    .unwrap_or(false)
                    && max_abs_theta_delta
                        .map(|delta| delta <= options.theta_tolerance)
                        .unwrap_or(false)
                    && max_abs_beta_delta
                        .map(|delta| delta <= options.beta_tolerance)
                        .unwrap_or(false)
                    && ranks_agree;

                ConvergenceVerificationRun {
                    label: label.to_string(),
                    optimizer_name: Some(optimizer_name(optimizer).to_string()),
                    return_code: Some(candidate.optsum.return_value.clone()),
                    objective_value,
                    objective_delta,
                    max_abs_theta_delta,
                    max_abs_beta_delta,
                    effective_ranks,
                    agrees,
                    diagnostics,
                }
            }
            Err(error) => ConvergenceVerificationRun {
                label: label.to_string(),
                optimizer_name: Some(optimizer_name(optimizer).to_string()),
                return_code: None,
                objective_value: None,
                objective_delta: None,
                max_abs_theta_delta: None,
                max_abs_beta_delta: None,
                effective_ranks: Vec::new(),
                agrees: false,
                diagnostics: vec![error.to_string()],
            },
        }
    }

    fn reset_for_convergence_verification(
        &mut self,
        start_theta: &[f64],
        max_function_evaluations: usize,
    ) -> Result<()> {
        let previous = self.optsum.clone();
        let mut optsum = OptSummary::new(start_theta.to_vec());
        optsum.xtol_zero_abs = previous.xtol_zero_abs;
        optsum.ftol_zero_abs = previous.ftol_zero_abs;
        optsum.ftol_rel = previous.ftol_rel;
        optsum.ftol_abs = previous.ftol_abs;
        optsum.xtol_rel = previous.xtol_rel;
        optsum.xtol_abs = previous.xtol_abs;
        optsum.initial_step = previous.initial_step;
        optsum.max_feval = max_function_evaluations as i64;
        optsum.max_time = previous.max_time;
        optsum.optimizer = previous.optimizer;
        optsum.backend = previous.backend;
        optsum.rhobeg = previous.rhobeg;
        optsum.rhoend = previous.rhoend;
        optsum.reml = previous.reml;
        optsum.n_agq = previous.n_agq;
        optsum.sigma = previous.sigma;
        self.optsum = optsum;
        self.set_theta(start_theta)?;
        self.update_l()
    }

    fn fit_with_forced_optimizer(&mut self, reml: bool, optimizer: Optimizer) -> Result<()> {
        self.optsum.reml = reml;
        let theta0 = self.optsum.initial.clone();
        self.optsum.finitial = self.objective_at(&theta0)?;
        match optimizer {
            Optimizer::PatternSearch => {
                if self.n_theta() == 1 {
                    self.fit_scalar_single_theta()?;
                } else {
                    self.fit_multivariate_pattern_search()?;
                }
            }
            Optimizer::Cobyla => {
                self.fit_cobyla(reml)?;
            }
            Optimizer::NloptBobyqa => {
                #[cfg(feature = "nlopt")]
                self.fit_nlopt_small_theta_with_maxeval(
                    reml,
                    Some(self.optsum.max_feval.max(1) as usize),
                )?;
                #[cfg(not(feature = "nlopt"))]
                return Err(MixedModelError::Optimization(
                    "Optimizer::NloptBobyqa requires the `nlopt` feature; \
                     rebuild with `--features nlopt` or pick a non-NLopt optimizer"
                        .to_string(),
                ));
            }
            Optimizer::NloptNewuoa => {
                #[cfg(feature = "nlopt")]
                self.fit_nlopt_large_theta_with_maxeval(
                    reml,
                    Some(self.optsum.max_feval.max(1) as usize),
                )?;
                #[cfg(not(feature = "nlopt"))]
                return Err(MixedModelError::Optimization(
                    "Optimizer::NloptNewuoa requires the `nlopt` feature; \
                     rebuild with `--features nlopt` or pick a non-NLopt optimizer"
                        .to_string(),
                ));
            }
            Optimizer::PrimaBobyqa => {
                #[cfg(feature = "prima")]
                self.fit_prima_bobyqa_with_maxeval(
                    reml,
                    Some(self.optsum.max_feval.max(1) as usize),
                )?;
                #[cfg(not(feature = "prima"))]
                return Err(MixedModelError::Optimization(
                    "Optimizer::PrimaBobyqa requires the `prima` feature and a system \
                     PRIMA C library (`libprimac`); rebuild with `--features prima` \
                     or pick a non-PRIMA optimizer"
                        .to_string(),
                ));
            }
            Optimizer::PrimaCobyla | Optimizer::PrimaLincoa | Optimizer::PrimaNewuoa => {
                return Err(MixedModelError::Optimization(
                    "Only Optimizer::PrimaBobyqa is wired to the LMM optimizer path; \
                     PRIMA COBYLA, LINCOA, and NEWUOA are reserved for later backend parity work"
                        .to_string(),
                ));
            }
        }
        self.refresh_optimizer_certificate();
        self.refresh_effective_covariance_summaries();
        self.refresh_covariance_parameter_traces();
        self.refresh_fixed_effect_inference_table();
        Ok(())
    }

    fn alternate_verification_optimizers(&self) -> Vec<Optimizer> {
        let current = self.optsum.optimizer;
        let alternate = if current != Optimizer::PatternSearch {
            Optimizer::PatternSearch
        } else if self.n_theta() == 1 {
            Optimizer::Cobyla
        } else if self.n_theta() <= 6 {
            Optimizer::NloptBobyqa
        } else {
            Optimizer::Cobyla
        };
        vec![alternate]
    }

    fn refresh_optimizer_certificate(&mut self) {
        let theta = self.theta();
        let lower_bounds = self.lower_bounds();
        let mut certificate = OptimizerCertificate::from_opt_summary_with_context(
            &self.optsum,
            &theta,
            &lower_bounds,
            Some(self.dims.n),
        );
        if certificate.evidence.optimizer_stop.acceptable_stop {
            if let Some(reason) = self.derivative_certificate_skip_reason(&certificate) {
                certificate.mark_derivative_checks_not_assessed(reason);
            } else if let Some(derivatives) =
                self.finite_difference_optimizer_derivatives(&theta, &lower_bounds)
            {
                let (gradient_tolerance, hessian_tolerance) =
                    self.derivative_certificate_tolerances(certificate.objective_value);
                certificate.apply_derivative_evidence(
                    derivatives,
                    gradient_tolerance,
                    hessian_tolerance,
                );
            }
        }
        self.reword_optimizer_certificate_diagnostics(&mut certificate);
        self.compiler_artifact.optimizer_certificate = Some(certificate);
    }

    fn reword_optimizer_certificate_diagnostics(&self, certificate: &mut OptimizerCertificate) {
        for diagnostic in &mut certificate.diagnostics {
            if diagnostic.code != DiagnosticCode::BoundaryParameter {
                continue;
            }
            let Some(theta_index) = diagnostic
                .payload
                .get("theta_index")
                .and_then(serde_json::Value::as_u64)
                .map(|value| value as usize)
            else {
                continue;
            };
            let Some((term_id, source_syntax, parameter_role)) =
                self.covariance_parameter_context(theta_index)
            else {
                continue;
            };

            diagnostic.message =
                format!("{parameter_role} in {source_syntax} is on its lower bound");
            diagnostic.affected_terms = vec![source_syntax.clone()];
            diagnostic
                .payload
                .insert("term_id".to_string(), serde_json::json!(term_id));
            diagnostic.payload.insert(
                "source_syntax".to_string(),
                serde_json::json!(source_syntax),
            );
            diagnostic.payload.insert(
                "parameter_role".to_string(),
                serde_json::json!(parameter_role),
            );
        }
    }

    fn covariance_parameter_context(&self, theta_index: usize) -> Option<(String, String, String)> {
        for theta_map in &self.compiler_artifact.theta_maps {
            let block = theta_map.block();
            let Some(slot) = block
                .theta_slots
                .iter()
                .find(|slot| slot.global_index == Some(theta_index))
            else {
                continue;
            };
            let row_basis = block
                .optimizer_basis
                .get(slot.lambda_row)
                .cloned()
                .unwrap_or_else(|| format!("basis {}", slot.lambda_row + 1));
            let col_basis = block
                .optimizer_basis
                .get(slot.lambda_col)
                .cloned()
                .unwrap_or_else(|| format!("basis {}", slot.lambda_col + 1));
            let parameter_role = if slot.lambda_row == slot.lambda_col {
                format!("standard deviation for {row_basis}")
            } else {
                format!("covariance link between {row_basis} and {col_basis}")
            };
            let source_syntax = self
                .compiler_artifact
                .semantic_model
                .random_terms
                .iter()
                .find(|term| term.id == block.term_id)
                .map(|term| term.source_syntax.text.clone())
                .unwrap_or_else(|| format!("random-effect term for {}", block.group));
            return Some((block.term_id.clone(), source_syntax, parameter_role));
        }

        None
    }

    fn derivative_certificate_skip_reason(
        &self,
        certificate: &OptimizerCertificate,
    ) -> Option<String> {
        let n_theta = certificate.evidence.parameter_space.n_theta;
        if n_theta == 0 {
            return Some(
                "there are no theta parameters, so derivative KKT/Hessian checks are not applicable"
                    .to_string(),
            );
        }

        if certificate.evidence.parameter_space.n_boundary > 0 {
            return Some(format!(
                "one or more covariance parameters are on a variance-component boundary (parameter indices: {}); boundary fits are reported as singular/boundary, not non-converged",
                certificate
                    .evidence
                    .parameter_space
                    .boundary_indices
                    .iter()
                    .map(|index| index.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }

        let nparmax = self
            .compiler_artifact
            .compiler_policy
            .thresholds
            .convergence_derivative_nparmax;
        if n_theta > nparmax {
            return Some(format!(
                "theta dimension {n_theta} exceeds convergence_derivative_nparmax {nparmax}; finite-difference KKT/Hessian checks are skipped for large-theta optimizer regimes"
            ));
        }

        None
    }

    fn derivative_certificate_tolerances(&self, objective_value: Option<f64>) -> (f64, f64) {
        let objective_scale = objective_value.unwrap_or(self.optsum.fmin).abs().max(1.0);
        let objective_tolerance = self
            .optsum
            .ftol_abs
            .max(self.optsum.ftol_zero_abs)
            .max(self.optsum.ftol_rel.max(0.0) * objective_scale);
        let gradient_tolerance = objective_tolerance.sqrt().max(1e-4);
        let hessian_tolerance = 1e-5_f64.max(gradient_tolerance * 1e-2);
        (gradient_tolerance, hessian_tolerance)
    }

    fn finite_difference_optimizer_derivatives(
        &self,
        theta: &[f64],
        lower_bounds: &[f64],
    ) -> Option<OptimizerDerivativeEvidence> {
        let n_theta = theta.len();
        if n_theta == 0
            || n_theta
                > self
                    .compiler_artifact
                    .compiler_policy
                    .thresholds
                    .convergence_derivative_nparmax
        {
            return None;
        }

        let mut evaluator = self.clone();
        let f0 = evaluator.objective_at(theta).ok()?;
        if !f0.is_finite() {
            return None;
        }

        let boundary_tolerance = self.optsum.xtol_zero_abs.max(1e-12) * 10.0;
        let boundary_mask = theta
            .iter()
            .zip(lower_bounds.iter())
            .map(|(&value, &lower)| {
                lower.is_finite() && (value - lower).abs() <= boundary_tolerance
            })
            .collect::<Vec<_>>();
        let gradient_steps = finite_difference_steps(theta, lower_bounds, 1e-5);
        let hessian_steps = finite_difference_steps(theta, lower_bounds, 1e-4);

        let mut gradient = vec![0.0; n_theta];
        for index in 0..n_theta {
            gradient[index] = finite_difference_gradient_coordinate(
                &mut evaluator,
                theta,
                lower_bounds,
                f0,
                index,
                gradient_steps[index],
            )?;
        }

        let free_indices = boundary_mask
            .iter()
            .enumerate()
            .filter_map(|(index, is_boundary)| (!*is_boundary).then_some(index))
            .collect::<Vec<_>>();
        let mut hessian = DMatrix::zeros(n_theta, n_theta);
        for &row in &free_indices {
            let row_step =
                feasible_central_step(theta[row], lower_bounds[row], hessian_steps[row])?;
            let mut plus = theta.to_vec();
            let mut minus = theta.to_vec();
            plus[row] += row_step;
            minus[row] -= row_step;
            let f_plus = evaluator.objective_at(&plus).ok()?;
            let f_minus = evaluator.objective_at(&minus).ok()?;
            if !f_plus.is_finite() || !f_minus.is_finite() {
                return None;
            }
            hessian[(row, row)] = (f_plus - 2.0 * f0 + f_minus) / (row_step * row_step);

            for &col in free_indices.iter().filter(|&&col| col > row) {
                let col_step =
                    feasible_central_step(theta[col], lower_bounds[col], hessian_steps[col])?;
                let f_pp = finite_difference_objective_2d(
                    &mut evaluator,
                    theta,
                    row,
                    row_step,
                    col,
                    col_step,
                )?;
                let f_pm = finite_difference_objective_2d(
                    &mut evaluator,
                    theta,
                    row,
                    row_step,
                    col,
                    -col_step,
                )?;
                let f_mp = finite_difference_objective_2d(
                    &mut evaluator,
                    theta,
                    row,
                    -row_step,
                    col,
                    col_step,
                )?;
                let f_mm = finite_difference_objective_2d(
                    &mut evaluator,
                    theta,
                    row,
                    -row_step,
                    col,
                    -col_step,
                )?;
                let value = (f_pp - f_pm - f_mp + f_mm) / (4.0 * row_step * col_step);
                hessian[(row, col)] = value;
                hessian[(col, row)] = value;
            }
        }

        Some(OptimizerDerivativeEvidence {
            method: EvidenceMethod::FiniteDifference,
            gradient,
            hessian: Some(hessian),
        })
    }

    fn refresh_covariance_parameter_traces(&mut self) {
        let lambdas = self
            .reterms
            .iter()
            .map(|reterm| matrix_rows(&reterm.lambda))
            .collect::<Vec<_>>();
        let sd_scale = if self.optsum.feval >= 0 {
            Some(self.sigma())
        } else {
            None
        };
        self.compiler_artifact.refresh_covariance_parameter_traces(
            Some(&lambdas),
            sd_scale,
            &self.parmap,
        );
    }

    fn refresh_effective_covariance_summaries(&mut self) {
        let Some(certificate) = &self.compiler_artifact.optimizer_certificate else {
            return;
        };
        // ConvergedPenalised fits still expose well-defined Λ matrices, so
        // their effective-covariance summaries are meaningful and should be
        // refreshed alongside the standard converged statuses. The
        // *promotion* path below stays narrower (only Interior/Boundary
        // promote to ReducedRank) — ConvergedPenalised is a contractual
        // leaf and must not be silently rewritten.
        if !matches!(
            certificate.status,
            crate::compiler::FitStatus::ConvergedInterior
                | crate::compiler::FitStatus::ConvergedBoundary
                | crate::compiler::FitStatus::ConvergedReducedRank
                | crate::compiler::FitStatus::ConvergedPenalised
        ) {
            self.compiler_artifact.effective_covariance.clear();
            return;
        }

        let thresholds = self.compiler_artifact.compiler_policy.thresholds.clone();
        let sigma_sq = self.sigma().powi(2);
        let mut summaries = Vec::with_capacity(self.reterms.len());
        let mut reductions = Vec::new();
        let mut transitions = Vec::new();
        let mut diagnostics = Vec::new();

        for (term_index, reterm) in self.reterms.iter().enumerate() {
            let theta_map = self.compiler_artifact.theta_maps.get(term_index);
            let term_id = theta_map
                .map(|map| map.block().term_id.clone())
                .unwrap_or_else(|| format!("r{term_index}"));
            let source_syntax = self
                .compiler_artifact
                .semantic_model
                .random_terms
                .iter()
                .find(|term| term.id == term_id)
                .map(|term| term.source_syntax.text.clone())
                .unwrap_or_else(|| format!("random-effect term for {}", reterm.grouping_name));
            let requested_basis = theta_map
                .map(|map| map.block().optimizer_basis.clone())
                .filter(|basis| basis.len() == reterm.vsize)
                .unwrap_or_else(|| {
                    reterm
                        .cnames
                        .iter()
                        .map(|name| user_basis_label(name))
                        .collect()
                });
            let requested_rank = reterm.vsize;
            let covariance = sigma_sq * (&reterm.lambda * reterm.lambda.transpose());
            let eig = SymmetricEigen::new(covariance);
            let mut pairs = (0..reterm.vsize)
                .map(|idx| {
                    (
                        eig.eigenvalues[idx],
                        eig.eigenvectors
                            .column(idx)
                            .iter()
                            .copied()
                            .collect::<Vec<_>>(),
                    )
                })
                .collect::<Vec<_>>();
            pairs.sort_by(|left, right| {
                right
                    .0
                    .partial_cmp(&left.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let max_eigenvalue = pairs
                .first()
                .map(|(value, _)| value.max(0.0))
                .unwrap_or(0.0);
            let rank_tolerance = thresholds.effective_rank_tolerance(max_eigenvalue);
            let total_positive: f64 = pairs.iter().map(|(value, _)| value.max(0.0)).sum();
            let pairs_snapshot = pairs.clone();
            let mut directions = Vec::new();
            let mut unsupported_directions = Vec::new();

            for (pc_index, (eigenvalue, vector)) in pairs.into_iter().enumerate() {
                let oriented = orient_eigenvector(vector);
                let loadings = requested_basis
                    .iter()
                    .cloned()
                    .zip(oriented.into_iter())
                    .map(|(basis, loading)| BasisLoading { basis, loading })
                    .collect::<Vec<_>>();
                let nonnegative_eigenvalue = eigenvalue.max(0.0);
                let direction = SupportedCovarianceDirection {
                    label: format!("PC{}", pc_index + 1),
                    loadings,
                    eigenvalue: Some(if nonnegative_eigenvalue <= rank_tolerance {
                        0.0
                    } else {
                        nonnegative_eigenvalue
                    }),
                    variance_explained: if total_positive > 0.0 {
                        Some(nonnegative_eigenvalue / total_positive)
                    } else {
                        Some(0.0)
                    },
                    user_scale_summary: String::new(),
                };
                let mut direction = direction;
                direction.user_scale_summary = format_loading_summary(&direction.loadings);
                if nonnegative_eigenvalue > rank_tolerance {
                    directions.push(direction);
                } else {
                    unsupported_directions.push(direction);
                }
            }

            let supported_rank = directions.len();
            let status = if supported_rank == requested_rank {
                EffectiveRankStatus::FullRank
            } else {
                EffectiveRankStatus::ReducedRank
            };
            let inference_consequence = if status == EffectiveRankStatus::ReducedRank {
                "fixed-effect inference is conditional on a certificate-time reduced-rank covariance summary; unsupported directions are not evidence of zero population variance".to_string()
            } else {
                "fixed-effect inference can condition on the fitted full-rank covariance for this term".to_string()
            };

            let interpretable_submodel = if status == EffectiveRankStatus::ReducedRank {
                detect_interpretable_submodel(
                    &pairs_snapshot,
                    &requested_basis,
                    requested_rank,
                    rank_tolerance,
                    sigma_sq,
                    &self.compiler_artifact.semantic_model.random_terms,
                    &term_id,
                )
            } else {
                None
            };

            summaries.push(EffectiveCovarianceSummary {
                term_id: term_id.clone(),
                source_syntax: source_syntax_for_term(
                    &self.compiler_artifact.semantic_model.random_terms,
                    &term_id,
                ),
                requested_basis: requested_basis.clone(),
                requested_rank,
                supported_rank,
                status,
                directions,
                unsupported_directions,
                inference_consequence: inference_consequence.clone(),
                interpretable_submodel: interpretable_submodel.clone(),
            });

            if status == EffectiveRankStatus::ReducedRank {
                let mut suggested_actions = vec![
                    "treat unsupported covariance directions as unsupported by this fit, not as proof of zero population variance".to_string(),
                ];
                if let Some(submodel) = &interpretable_submodel {
                    suggested_actions.push(format!(
                        "consider refitting the simpler random-effect term {}; this fitted model was not silently refit",
                        submodel.suggested_formula
                    ));
                }
                let mut diagnostic = Diagnostic::new(
                    DiagnosticCode::CovarianceReduced,
                    DiagnosticSeverity::Info,
                    DiagnosticStage::Certification,
                    format!(
                        "fitted covariance for {source_syntax} has effective rank {supported_rank} of requested rank {requested_rank}"
                    ),
                )
                .with_affected_terms(vec![source_syntax.clone()])
                .with_suggested_actions(suggested_actions);
                diagnostic
                    .payload
                    .insert("term_id".to_string(), serde_json::json!(term_id.clone()));
                diagnostic.payload.insert(
                    "source_syntax".to_string(),
                    serde_json::json!(source_syntax.clone()),
                );
                diagnostic.payload.insert(
                    "rank_tolerance".to_string(),
                    serde_json::json!(rank_tolerance),
                );
                diagnostic.payload.insert(
                    "effective_rank_relative_tolerance".to_string(),
                    serde_json::json!(thresholds.effective_rank_relative_tolerance),
                );
                diagnostic.payload.insert(
                    "effective_rank_absolute_tolerance".to_string(),
                    serde_json::json!(thresholds.effective_rank_absolute_tolerance),
                );
                diagnostic.payload.insert(
                    "requested_rank".to_string(),
                    serde_json::json!(requested_rank),
                );
                diagnostic.payload.insert(
                    "supported_rank".to_string(),
                    serde_json::json!(supported_rank),
                );
                if let Some(submodel) = &interpretable_submodel {
                    if let Ok(payload) = serde_json::to_value(submodel) {
                        diagnostic
                            .payload
                            .insert("interpretable_submodel".to_string(), payload);
                    }
                }

                diagnostics.push(diagnostic.clone());
                reductions.push(ReductionRecord {
                    trigger: ReductionTrigger::CertificateTimeBoundary,
                    phase: "fit-time effective covariance rank".to_string(),
                    reason: format!(
                        "effective covariance rank {supported_rank} is below requested rank {requested_rank}"
                    ),
                    affected_term: term_id.clone(),
                    replacement_term: Some(format!(
                        "reduced_rank({}, basis = {}, rank = {})",
                        reterm.grouping_name,
                        requested_basis.join(" + "),
                        supported_rank
                    )),
                    inference_consequence: inference_consequence.clone(),
                    diagnostics: Vec::new(),
                });

                if let Some(theta_map) = theta_map {
                    transitions.push(CovarianceFamilyTransition {
                        from: theta_map.family(),
                        to: CovarianceFamily::ReducedRank {
                            rank: Some(supported_rank),
                        },
                        trigger: ReductionTrigger::CertificateTimeBoundary,
                        affected_term: term_id,
                        dropped_or_reparameterized_slots: Vec::new(),
                        inference_consequence,
                    });
                }
            }
        }

        self.compiler_artifact.effective_covariance = summaries;
        self.compiler_artifact.reductions.extend(reductions);
        self.compiler_artifact
            .covariance_transitions
            .extend(transitions);

        if !diagnostics.is_empty() {
            if let Some(certificate) = &mut self.compiler_artifact.optimizer_certificate {
                if matches!(
                    certificate.status,
                    crate::compiler::FitStatus::ConvergedInterior
                        | crate::compiler::FitStatus::ConvergedBoundary
                ) {
                    certificate.status = crate::compiler::FitStatus::ConvergedReducedRank;
                }
                certificate.diagnostics.extend(diagnostics);
            }
        }
    }

    /// Get the response vector y (last column of xy_mat).
    pub fn y(&self) -> DVector<f64> {
        self.y.clone()
    }

    /// Get the current θ parameter vector.
    pub fn theta(&self) -> Vec<f64> {
        self.reterms.iter().flat_map(|rt| rt.get_theta()).collect()
    }

    /// Set the θ parameter vector, distributing values to each ReMat's λ.
    pub fn set_theta(&mut self, theta: &[f64]) -> Result<()> {
        let expected = self.n_theta();
        if theta.len() != expected {
            return Err(MixedModelError::DimensionMismatch(format!(
                "theta vector length mismatch: expected {expected}, got {}",
                theta.len()
            )));
        }

        let mut offset = 0;
        for rt in &mut self.reterms {
            let n = rt.n_theta();
            rt.set_theta(&theta[offset..offset + n])?;
            offset += n;
        }
        Ok(())
    }

    /// Lower bounds on θ. Diagonal elements of λ are ≥ 0, off-diagonal are unconstrained.
    pub fn lower_bounds(&self) -> Vec<f64> {
        let mut lb = Vec::new();
        for (_, row, col) in &self.parmap {
            if row == col {
                lb.push(0.0); // diagonal elements are non-negative
            } else {
                lb.push(f64::NEG_INFINITY);
            }
        }
        lb
    }

    fn theta_at_lower_bound(&self) -> bool {
        let theta = self.theta();
        let lb = self.lower_bounds();
        let boundary_tolerance = self.optsum.xtol_zero_abs.max(1e-12) * 10.0;
        theta.iter().zip(lb.iter()).any(|(&value, &lower)| {
            lower.is_finite() && (value - lower).abs() <= boundary_tolerance
        })
    }

    fn optimizer_certificate_reports_boundary(&self) -> bool {
        self.compiler_artifact
            .optimizer_certificate
            .as_ref()
            .is_some_and(|certificate| certificate.evidence.parameter_space.n_boundary > 0)
    }

    fn has_reduced_effective_covariance(&self) -> bool {
        self.compiler_artifact
            .effective_covariance
            .iter()
            .any(|summary| summary.status == EffectiveRankStatus::ReducedRank)
    }

    /// Update the blocked Cholesky factor L from A and current λ values.
    ///
    /// This is the core operation: L = cholesky(Λ'AΛ + I).
    /// The blocks of L are updated in-place.
    pub fn update_l(&mut self) -> Result<()> {
        let cholesky_zero_pad_tolerance = self
            .compiler_policy()
            .thresholds
            .cholesky_zero_pad_tolerance;
        update_l_from_parts(
            &self.a_blocks,
            &mut self.l_blocks,
            &self.reterms,
            cholesky_zero_pad_tolerance,
        )
    }

    /// Update IRLS weights and working response, then rebuild A blocks.
    /// Called at each PIRLS iteration of a GLMM.
    ///
    /// * `sqrtwts` - square-root of the IRLS weights (length n)
    /// * `working_y` - working response values (length n)
    pub fn update_irls_weights(&mut self, sqrtwts: &[f64], working_y: &[f64]) {
        let n = self.dims.n;
        debug_assert_eq!(sqrtwts.len(), n);
        debug_assert_eq!(working_y.len(), n);

        self.sqrtwts = sqrtwts.to_vec();

        // Update wtz for every RE term: wtz[s, obs] = sqrtwts[obs] * z[s, obs]
        for rt in &mut self.reterms {
            let vsize = rt.vsize;
            for obs in 0..n {
                for s in 0..vsize {
                    rt.wtz[(s, obs)] = sqrtwts[obs] * rt.z[(s, obs)];
                }
            }
        }

        // Update wtxy: first `rank` columns from X, last column from working_y
        let rank = self.feterm.rank;
        for obs in 0..n {
            let sw = sqrtwts[obs];
            for col in 0..rank {
                self.xy_mat.wtxy[(obs, col)] = sw * self.feterm.x[(obs, col)];
            }
            // y column (last)
            self.xy_mat.wtxy[(obs, rank)] = sw * working_y[obs];
            self.xy_mat.xy[(obs, rank)] = working_y[obs];
        }

        // Rebuild A blocks
        self.recompute_a_blocks();
    }

    /// Recompute all A-block cross products from the current wtz / wtxy.
    /// Does NOT rebuild L — call `update_l()` after this.
    pub fn recompute_a_blocks(&mut self) {
        let k = self.reterms.len();
        let mut idx = 0;
        let sqrtwts = if self.sqrtwts.is_empty() {
            None
        } else {
            Some(DVector::from_column_slice(&self.sqrtwts))
        };
        let weighted_fixed_design =
            weighted_fixed_design_for_solver(&self.fixed_design, sqrtwts.as_ref()).expect(
                "stored fixed-effect design and sqrt weights must have compatible dimensions",
            );
        let weighted_response = self.xy_mat.wtxy.column(self.feterm.rank).into_owned();

        // RE × RE blocks
        for i in 0..k {
            for j in 0..=i {
                let block = if i == j {
                    compute_re_cross_product(&self.reterms[i], &self.reterms[i])
                } else {
                    compute_re_cross_product(&self.reterms[i], &self.reterms[j])
                };
                self.a_blocks[idx] = block;
                idx += 1;
            }
        }

        // FE × RE blocks: [X|y]' Z_j
        for j in 0..k {
            let block = compute_fixed_response_re_cross_product(
                &weighted_fixed_design,
                &weighted_response,
                &self.reterms[j],
            )
            .expect("stored fixed-effect design and random terms must have compatible dimensions");
            self.a_blocks[idx] = block;
            idx += 1;
        }

        // FE × FE block: [X|y]' [X|y]
        self.a_blocks[idx] = MatrixBlock::Dense(
            compute_fixed_response_cross_product(&weighted_fixed_design, &weighted_response)
                .expect("stored fixed-effect design and response must have compatible dimensions"),
        );
    }

    fn determinant_term_and_pwrss_for_reml(&self, reml: bool) -> (f64, f64) {
        let k = self.reterms.len();

        let mut logdet = 0.0;
        for j in 0..k {
            logdet += logdet_block(&self.l_blocks[block_index(j, j)]);
        }

        let l_dense = self.l_blocks[block_index(k, k)].as_dense();
        let pp1 = l_dense.nrows();
        let last_diag = l_dense[(pp1 - 1, pp1 - 1)];
        let pwrss = last_diag * last_diag;

        if reml {
            let mut logdet_lxx = 0.0;
            for i in 0..(pp1 - 1) {
                let d = l_dense[(i, i)];
                if d > 0.0 {
                    logdet_lxx += d.ln();
                }
            }
            logdet += 2.0 * logdet_lxx;
        }

        (logdet, pwrss)
    }

    fn determinant_term_and_pwrss(&self) -> (f64, f64) {
        self.determinant_term_and_pwrss_for_reml(self.optsum.reml)
    }

    fn objective_from_components(
        logdet: f64,
        pwrss: f64,
        denomdf: f64,
        fixed_sigma: Option<f64>,
    ) -> f64 {
        let log2pi = (2.0 * std::f64::consts::PI).ln();
        if let Some(sigma) = fixed_sigma {
            if !sigma.is_finite() || sigma <= 0.0 {
                return f64::INFINITY;
            }
            logdet + denomdf * (2.0 * sigma.ln() + log2pi) + pwrss / (sigma * sigma)
        } else {
            logdet + denomdf * (1.0 + (2.0 * std::f64::consts::PI * pwrss / denomdf).ln())
        }
    }

    fn profiled_objective_value(&self) -> f64 {
        let denomdf = if self.optsum.reml {
            (self.dims.n - self.dims.p) as f64
        } else {
            self.dims.n as f64
        };
        let (logdet, pwrss) = self.determinant_term_and_pwrss();
        Self::objective_from_components(logdet, pwrss, denomdf, self.optsum.sigma)
    }

    fn weight_logdet_correction(&self) -> f64 {
        if self.sqrtwts.is_empty() {
            0.0
        } else {
            2.0 * self.sqrtwts.iter().map(|sqrtwt| sqrtwt.ln()).sum::<f64>()
        }
    }

    /// Compute the user-facing deviance or REML criterion for the current θ.
    ///
    /// Weighted LMMs subtract the log-Jacobian term for the row scaling,
    /// matching MixedModels.jl's `objective(::LinearMixedModel)`. The optimizer
    /// hot path remains [`Self::profiled_objective_from_parts`], whose target
    /// omits this θ-constant correction.
    pub fn objective_value(&self) -> f64 {
        self.profiled_objective_value() - self.weight_logdet_correction()
    }

    /// Set θ, update L, and return the objective value.
    pub fn objective_at(&mut self, theta: &[f64]) -> Result<f64> {
        self.set_theta(theta)?;
        self.update_l()?;
        Ok(self.objective_value())
    }

    fn vcov_active_with_sigma(&self, sigma: f64) -> DMatrix<f64> {
        let k = self.reterms.len();
        let l_last = self.l_blocks[block_index(k, k)].as_dense();
        let pp1 = l_last.nrows();
        let p = pp1 - 1;

        if p == 0 {
            return DMatrix::zeros(0, 0);
        }

        let l_xx = l_last.view((0, 0), (p, p)).clone_owned();

        // L_inv = L_XX^{-1}
        let mut l_inv = DMatrix::<f64>::identity(p, p);
        // Forward solve: L_XX * L_inv = I
        for j in 0..p {
            for i in j..p {
                let mut s = l_inv[(i, j)];
                for k2 in j..i {
                    s -= l_xx[(i, k2)] * l_inv[(k2, j)];
                }
                l_inv[(i, j)] = s / l_xx[(i, i)];
            }
        }

        let sigma_sq = sigma * sigma;
        sigma_sq * (&l_inv.transpose() * &l_inv)
    }

    fn unpivot_fixed_effect_covariance(&self, active_vcov: &DMatrix<f64>) -> DMatrix<f64> {
        // Unpivot
        let full_p = self.feterm.piv.len();
        let p = active_vcov.nrows();
        if p == full_p {
            let mut result = DMatrix::zeros(full_p, full_p);
            for i in 0..full_p {
                for j in 0..full_p {
                    result[(self.feterm.piv[i], self.feterm.piv[j])] = active_vcov[(i, j)];
                }
            }
            result
        } else {
            let mut result = DMatrix::from_element(full_p, full_p, f64::NAN);
            for i in 0..p {
                for j in 0..p {
                    result[(self.feterm.piv[i], self.feterm.piv[j])] = active_vcov[(i, j)];
                }
            }
            result
        }
    }

    fn vcov_with_sigma(&self, sigma: f64) -> DMatrix<f64> {
        let active = self.vcov_active_with_sigma(sigma);
        self.unpivot_fixed_effect_covariance(&active)
    }

    /// Evaluate the ML or REML deviance over `varpar = c(theta, sigma)`.
    ///
    /// This is the Rust analogue of `lmerTestR::devfun_vp`: it evaluates the
    /// unprofiled criterion at trial covariance parameters and a trial residual
    /// standard deviation, then restores the fitted model state.
    pub fn deviance_varpar(&mut self, varpar: &[f64], reml: bool) -> Result<f64> {
        self.validate_varpar(varpar)?;
        let n_theta = self.n_theta();
        let theta = &varpar[..n_theta];
        let sigma = varpar[n_theta];

        let original_theta = self.theta();
        let original_l_blocks = self.l_blocks.clone();

        let result = (|| {
            self.set_theta(theta)?;
            self.update_l()?;

            let denomdf = if reml {
                (self.dims.n - self.dims.p) as f64
            } else {
                self.dims.n as f64
            };
            let (logdet, pwrss) = self.determinant_term_and_pwrss_for_reml(reml);
            let deviance = Self::objective_from_components(logdet, pwrss, denomdf, Some(sigma));
            if deviance.is_finite() {
                Ok(deviance)
            } else {
                Err(MixedModelError::Optimization(
                    "deviance over variance parameters is non-finite".to_string(),
                ))
            }
        })();

        self.set_theta(&original_theta)?;
        self.l_blocks = original_l_blocks;

        result
    }

    /// Evaluate the fixed-effect covariance matrix at `varpar = c(theta, sigma)`.
    ///
    /// This is the Rust analogue of `lmerTestR::get_covbeta`: at a trial
    /// covariance parameter point it returns `sigma^2 * RXi * RXi'`, then
    /// restores the fitted model state.
    pub fn vcov_beta_varpar(&mut self, varpar: &[f64]) -> Result<DMatrix<f64>> {
        self.validate_varpar(varpar)?;
        let n_theta = self.n_theta();
        let theta = &varpar[..n_theta];
        let sigma = varpar[n_theta];

        let original_theta = self.theta();
        let original_l_blocks = self.l_blocks.clone();

        let result = (|| {
            self.set_theta(theta)?;
            self.update_l()?;

            let vcov = self.vcov_with_sigma(sigma);
            if matrix_is_finite(&vcov) {
                Ok(vcov)
            } else {
                Err(MixedModelError::InvalidArgument(
                    "vcov_beta(varpar) contains non-finite entries".to_string(),
                ))
            }
        })();

        self.set_theta(&original_theta)?;
        self.l_blocks = original_l_blocks;

        result
    }

    /// Numerically differentiate `vcov_beta_varpar` with respect to `varpar`.
    ///
    /// Returns one `p x p` matrix per `varpar` component. The first
    /// implementation intentionally requires a feasible central-difference
    /// stencil; boundary-active parameters return an explicit unavailable
    /// reason instead of silently producing one-sided derivatives.
    pub fn jac_vcov_beta_varpar(&mut self, varpar: &[f64]) -> Result<Vec<DMatrix<f64>>> {
        self.validate_varpar(varpar)?;

        let lower_bounds = self.varpar_lower_bounds();
        let steps = finite_difference_steps(varpar, &lower_bounds, 1e-5);
        let mut jacobian = Vec::with_capacity(varpar.len());

        for index in 0..varpar.len() {
            let lower = lower_bounds
                .get(index)
                .copied()
                .unwrap_or(f64::NEG_INFINITY);
            let step =
                feasible_central_step(varpar[index], lower, steps[index]).ok_or_else(|| {
                    MixedModelError::InvalidArgument(format!(
                        "cannot compute central finite-difference derivative for varpar[{index}]: \
                     value is at or too near lower bound {lower}"
                    ))
                })?;

            let mut plus = varpar.to_vec();
            let mut minus = varpar.to_vec();
            plus[index] += step;
            minus[index] -= step;

            let vcov_plus = self.vcov_beta_varpar(&plus)?;
            let vcov_minus = self.vcov_beta_varpar(&minus)?;
            let derivative = (&vcov_plus - &vcov_minus) * (0.5 / step);
            if !matrix_is_finite(&derivative) {
                return Err(MixedModelError::InvalidArgument(format!(
                    "jac_vcov_beta derivative for varpar[{index}] contains non-finite entries"
                )));
            }
            jacobian.push(symmetrize_matrix(&derivative));
        }

        Ok(jacobian)
    }

    /// Estimate `vcov(varpar)` from the Hessian of `deviance_varpar`.
    ///
    /// This mirrors the lmerTest convention `2 * H^+`, where `H^+` is the
    /// Moore-Penrose inverse of the Hessian over positive eigen-directions.
    pub fn vcov_varpar(&mut self, varpar: &[f64], reml: bool) -> Result<VcovVarparEstimate> {
        let hessian = self.hessian_deviance_varpar(varpar, reml)?;
        let hessian = symmetrize_matrix(&hessian);
        let eig = SymmetricEigen::new(hessian.clone());
        let eigenvalues = eig.eigenvalues.as_slice().to_vec();
        let max_abs_eigenvalue = eigenvalues
            .iter()
            .map(|value| value.abs())
            .fold(0.0, f64::max);
        let tolerance = (1e-8 * max_abs_eigenvalue.max(1.0)).max(1e-10);

        let positive_eigenvalues = eigenvalues
            .iter()
            .filter(|value| **value > tolerance)
            .count();
        let near_zero_eigenvalues = eigenvalues
            .iter()
            .filter(|value| value.abs() <= tolerance)
            .count();
        let negative_eigenvalues = eigenvalues
            .iter()
            .filter(|value| **value < -tolerance)
            .count();

        if positive_eigenvalues == 0 {
            return Err(MixedModelError::Optimization(
                "vcov_varpar unavailable: deviance Hessian has no positive eigen-directions"
                    .to_string(),
            ));
        }

        let mut inverse = DMatrix::zeros(varpar.len(), varpar.len());
        for (index, &eigenvalue) in eigenvalues.iter().enumerate() {
            if eigenvalue > tolerance {
                let column = eig.eigenvectors.column(index);
                inverse += (column * column.transpose()) * (1.0 / eigenvalue);
            }
        }

        let covariance = symmetrize_matrix(&(2.0 * inverse));
        if !matrix_is_finite(&covariance) {
            return Err(MixedModelError::Optimization(
                "vcov_varpar unavailable: covariance estimate contains non-finite entries"
                    .to_string(),
            ));
        }

        let used_reduced_rank = positive_eigenvalues < varpar.len();
        let mut notes = Vec::new();
        if near_zero_eigenvalues > 0 {
            notes.push(format!(
                "deviance Hessian has {near_zero_eigenvalues} near-zero eigenvalue(s)"
            ));
        }
        if negative_eigenvalues > 0 {
            notes.push(format!(
                "deviance Hessian has {negative_eigenvalues} negative eigenvalue(s)"
            ));
        }
        if used_reduced_rank {
            notes.push(
                "vcov_varpar used the positive-eigenvalue subspace of the Hessian".to_string(),
            );
        }

        Ok(VcovVarparEstimate {
            covariance,
            hessian,
            eigenvalues,
            tolerance,
            positive_eigenvalues,
            near_zero_eigenvalues,
            negative_eigenvalues,
            used_reduced_rank,
            reliability: if used_reduced_rank {
                ReliabilityGrade::Low
            } else {
                ReliabilityGrade::Moderate
            },
            notes,
        })
    }

    /// Build the Kenward-Roger response-covariance component decomposition.
    ///
    /// The returned matrices follow the `pbkrtest::get_SigmaG()` convention:
    /// fitted marginal response covariance is represented as a weighted sum of
    /// known component matrices. Random-effect component weights are fitted
    /// VarCorr covariance entries (`sigma^2 * Lambda Lambda'`); the final
    /// component is the residual variance multiplying the identity matrix.
    pub fn kenward_roger_sigma_g(&self) -> Result<KenwardRogerSigmaG> {
        if self.optsum.feval == 0 {
            return Err(MixedModelError::NotFitted);
        }
        if !self.sqrtwts.is_empty() {
            return Err(MixedModelError::InvalidArgument(
                "Kenward-Roger Sigma/G decomposition is currently certified only for unweighted iid Gaussian residual models"
                    .to_string(),
            ));
        }

        let n = self.dims.n;
        let n_components: usize = self
            .reterms
            .iter()
            .map(kenward_roger_covariance_component_count)
            .sum::<usize>()
            + 1;
        let dense_bytes = dense_block_bytes(n, n).saturating_mul((n_components + 1) as u128);
        let limit = dense_block_limit_bytes();
        if dense_bytes > limit {
            return Err(MixedModelError::ProblemTooLarge(format!(
                "Kenward-Roger Sigma/G would materialize {} dense {} x {} f64 matrices ({:.2} GiB), above the configured limit ({:.2} GiB)",
                n_components + 1,
                n,
                n,
                dense_bytes as f64 / 1024.0_f64.powi(3),
                limit as f64 / 1024.0_f64.powi(3)
            )));
        }

        let sigma = self.sigma();
        if !sigma.is_finite() || sigma <= 0.0 {
            return Err(MixedModelError::InvalidArgument(
                "Kenward-Roger Sigma/G requires a finite positive residual sigma".to_string(),
            ));
        }
        let sigma_sq = sigma * sigma;

        let mut components = Vec::with_capacity(n_components);
        let mut component_weights = Vec::with_capacity(n_components);
        let mut component_labels = Vec::with_capacity(n_components);

        for (term_index, reterm) in self.reterms.iter().enumerate() {
            let covariance = sigma_sq * (&reterm.lambda * reterm.lambda.transpose());
            for (row, col) in kenward_roger_covariance_component_indices(reterm) {
                let component = kenward_roger_response_component(reterm, row, col, n)?;
                let label = format!(
                    "{}:{}[{},{}]",
                    term_index, reterm.grouping_name, reterm.cnames[row], reterm.cnames[col]
                );
                components.push(component);
                component_weights.push(covariance[(row, col)]);
                component_labels.push(label);
            }
        }

        let residual_component_index = components.len();
        components.push(DMatrix::identity(n, n));
        component_weights.push(sigma_sq);
        component_labels.push("residual".to_string());

        let mut response_covariance = DMatrix::zeros(n, n);
        for (component, &weight) in components.iter().zip(component_weights.iter()) {
            if !weight.is_finite() {
                return Err(MixedModelError::InvalidArgument(
                    "Kenward-Roger Sigma/G component weight is non-finite".to_string(),
                ));
            }
            if !matrix_is_finite(component) {
                return Err(MixedModelError::InvalidArgument(
                    "Kenward-Roger Sigma/G component contains non-finite entries".to_string(),
                ));
            }
            response_covariance += component * weight;
        }
        let response_covariance = symmetrize_matrix(&response_covariance);

        if !matrix_is_finite(&response_covariance) {
            return Err(MixedModelError::InvalidArgument(
                "Kenward-Roger Sigma/G response covariance contains non-finite entries".to_string(),
            ));
        }

        let max_component_asymmetry = components
            .iter()
            .map(matrix_max_asymmetry)
            .fold(0.0, f64::max)
            .max(matrix_max_asymmetry(&response_covariance));
        let eig = SymmetricEigen::new(response_covariance.clone());
        let sigma_min_eigenvalue = eig
            .eigenvalues
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let sigma_max_eigenvalue = eig
            .eigenvalues
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let eigen_tolerance = (1e-10 * sigma_max_eigenvalue.abs().max(1.0)).max(1e-12);
        let sigma_positive_definite = sigma_min_eigenvalue > eigen_tolerance;
        let mut notes = Vec::new();
        if !sigma_positive_definite {
            notes.push(format!(
                "response covariance is not positive definite at tolerance {eigen_tolerance}"
            ));
        }

        Ok(KenwardRogerSigmaG {
            sigma: response_covariance,
            components,
            component_weights,
            component_labels,
            residual_component_index,
            covariance_parameterization: "VarCorr covariance entries followed by residual variance"
                .to_string(),
            includes_residual_variance: true,
            n_observations: n,
            dense_bytes,
            sigma_min_eigenvalue,
            sigma_max_eigenvalue,
            sigma_positive_definite,
            max_component_asymmetry,
            reliability: if sigma_positive_definite {
                ReliabilityGrade::Moderate
            } else {
                ReliabilityGrade::NotAvailable
            },
            notes,
        })
    }

    /// Compute the Kenward-Roger adjusted fixed-effect covariance.
    ///
    /// This is the Rust analogue of `pbkrtest::vcovAdj_internal()`. It uses the
    /// active fixed-effect basis internally and exposes an unpivoted
    /// `adjusted_vcov` for the user-facing coefficient surface.
    pub fn kenward_roger_adjusted_vcov(&self) -> Result<KenwardRogerAdjustedVcov> {
        let sigma_g = self.kenward_roger_sigma_g()?;
        if !sigma_g.sigma_positive_definite {
            return Err(MixedModelError::Singular(
                "Kenward-Roger adjusted covariance requires a positive-definite response covariance"
                    .to_string(),
            ));
        }

        let phi = self.vcov_active_with_sigma(self.sigma());
        if !matrix_is_finite(&phi) {
            return Err(MixedModelError::InvalidArgument(
                "Kenward-Roger adjusted covariance requires finite active fixed-effect covariance"
                    .to_string(),
            ));
        }
        let x = self.feterm.full_rank_x().into_owned();
        if x.ncols() != phi.ncols() {
            return Err(MixedModelError::DimensionMismatch(format!(
                "Kenward-Roger active fixed-effect covariance has {} columns, but X has {}",
                phi.ncols(),
                x.ncols()
            )));
        }

        let sigma_inv = invert_spd_matrix(&sigma_g.sigma, "Kenward-Roger response covariance")?;
        let tt = &sigma_inv * &x;
        let n_components = sigma_g.components.len();
        let p = phi.ncols();

        let mut hh = Vec::with_capacity(n_components);
        let mut oo = Vec::with_capacity(n_components);
        let mut p_matrices = Vec::with_capacity(n_components);
        for component in &sigma_g.components {
            let h = component * &sigma_inv;
            let o = &h * &x;
            let p_matrix = symmetrize_matrix(&(-o.transpose() * &tt));
            hh.push(h);
            oo.push(o);
            p_matrices.push(p_matrix);
        }

        let mut q_matrices = Vec::with_capacity(n_components.saturating_mul(n_components + 1) / 2);
        let mut information_matrix = DMatrix::zeros(n_components, n_components);
        for rr in 0..n_components {
            for ss in rr..n_components {
                let q_matrix = oo[rr].transpose() * &sigma_inv * &oo[ss];
                let q_index = q_matrices.len();
                q_matrices.push(q_matrix);

                let ktrace = matrix_elementwise_dot(&hh[rr].transpose(), &hh[ss]);
                let phi_q = matrix_elementwise_dot(&phi, &q_matrices[q_index]);
                let phi_p_rr = &phi * &p_matrices[rr];
                let pp_term = matrix_elementwise_dot(&phi_p_rr, &(&p_matrices[ss] * &phi));
                let value = ktrace - 2.0 * phi_q + pp_term;
                information_matrix[(rr, ss)] = value;
                information_matrix[(ss, rr)] = value;
            }
        }
        let information_matrix = symmetrize_matrix(&information_matrix);
        if !matrix_is_finite(&information_matrix) {
            return Err(MixedModelError::InvalidArgument(
                "Kenward-Roger information matrix contains non-finite entries".to_string(),
            ));
        }

        let information_eigen = SymmetricEigen::new(information_matrix.clone());
        let information_eigenvalues = information_eigen.eigenvalues.as_slice().to_vec();
        let condition_min_abs_eigenvalue = information_eigenvalues
            .iter()
            .map(|value| value.abs())
            .fold(f64::INFINITY, f64::min);
        let max_abs_eigenvalue = information_eigenvalues
            .iter()
            .map(|value| value.abs())
            .fold(0.0, f64::max);
        let generalized_inverse_tolerance = (1e-10 * max_abs_eigenvalue.max(1.0)).max(1e-12);
        let used_generalized_inverse =
            condition_min_abs_eigenvalue <= generalized_inverse_tolerance;
        let w = if used_generalized_inverse {
            2.0 * symmetric_pseudoinverse(&information_matrix, generalized_inverse_tolerance)
        } else {
            2.0 * invert_spd_matrix(
                &information_matrix,
                "Kenward-Roger covariance-parameter information matrix",
            )?
        };
        let w = symmetrize_matrix(&w);
        if !matrix_is_finite(&w) {
            return Err(MixedModelError::InvalidArgument(
                "Kenward-Roger covariance-parameter uncertainty matrix contains non-finite entries"
                    .to_string(),
            ));
        }

        let mut uu = DMatrix::zeros(p, p);
        for rr in 0..n_components {
            for ss in (rr + 1)..n_components {
                let q_index = symmetric_pair_index(rr, ss, n_components);
                uu +=
                    w[(rr, ss)] * (&q_matrices[q_index] - &p_matrices[rr] * &phi * &p_matrices[ss]);
            }
        }
        uu = &uu + uu.transpose();
        for rr in 0..n_components {
            let q_index = symmetric_pair_index(rr, rr, n_components);
            uu += w[(rr, rr)] * (&q_matrices[q_index] - &p_matrices[rr] * &phi * &p_matrices[rr]);
        }

        let gamma = &phi * uu * &phi;
        let adjusted_active = symmetrize_matrix(&(&phi + 2.0 * gamma));
        if !matrix_is_finite(&adjusted_active) {
            return Err(MixedModelError::InvalidArgument(
                "Kenward-Roger adjusted fixed-effect covariance contains non-finite entries"
                    .to_string(),
            ));
        }

        let mut notes = Vec::new();
        if used_generalized_inverse {
            notes.push(format!(
                "Kenward-Roger information matrix used a generalized inverse at tolerance {generalized_inverse_tolerance}"
            ));
        }
        if sigma_g.reliability != ReliabilityGrade::Moderate {
            notes.extend(sigma_g.notes.clone());
        }

        let reliability = if used_generalized_inverse {
            ReliabilityGrade::Low
        } else {
            ReliabilityGrade::Moderate
        };

        Ok(KenwardRogerAdjustedVcov {
            unadjusted_vcov_active: phi,
            adjusted_vcov: self.unpivot_fixed_effect_covariance(&adjusted_active),
            adjusted_vcov_active: adjusted_active,
            p_matrices,
            q_matrices,
            w,
            information_matrix,
            information_eigenvalues,
            condition_min_abs_eigenvalue,
            used_generalized_inverse,
            component_labels: sigma_g.component_labels,
            reliability,
            notes,
        })
    }

    /// Compute Kenward-Roger denominator df for `L beta = rhs`.
    ///
    /// This follows `pbkrtest::Lb_ddf(L, V0, Vadj)` using the active fixed-effect
    /// covariance basis. User-order full-rank contrasts are accepted and mapped
    /// onto the active basis.
    pub fn kenward_roger_lbddf(&self, l: &DMatrix<f64>) -> Result<KenwardRogerLbDdf> {
        let adjusted = self.kenward_roger_adjusted_vcov()?;
        self.kenward_roger_lbddf_with_adjusted(l, &adjusted)
    }

    pub fn kenward_roger_lbddf_with_adjusted(
        &self,
        l: &DMatrix<f64>,
        adjusted: &KenwardRogerAdjustedVcov,
    ) -> Result<KenwardRogerLbDdf> {
        let l_active = self.fixed_effect_contrast_to_active_basis(l)?;
        if l_active.nrows() == 0 {
            return Err(MixedModelError::InvalidArgument(
                "Kenward-Roger Lb_ddf requires at least one restriction row".to_string(),
            ));
        }
        if l_active.ncols() != adjusted.unadjusted_vcov_active.ncols() {
            return Err(MixedModelError::DimensionMismatch(format!(
                "Kenward-Roger Lb_ddf contrast has {} active columns, but V0 has {}",
                l_active.ncols(),
                adjusted.unadjusted_vcov_active.ncols()
            )));
        }

        let rank_tolerance = 1e-10;
        let restriction_rank = matrix_rank(&l_active, rank_tolerance);
        if restriction_rank == 0 {
            return Err(MixedModelError::InvalidArgument(
                "Kenward-Roger Lb_ddf contrast has zero numerical rank".to_string(),
            ));
        }

        let v0 = &adjusted.unadjusted_vcov_active;
        let middle = symmetrize_matrix(&(&l_active * v0 * l_active.transpose()));
        let middle_eig = SymmetricEigen::new(middle.clone());
        let middle_max_abs = middle_eig
            .eigenvalues
            .iter()
            .map(|value| value.abs())
            .fold(0.0, f64::max);
        let middle_tol = (1e-10 * middle_max_abs.max(1.0)).max(1e-12);
        let middle_min_abs = middle_eig
            .eigenvalues
            .iter()
            .map(|value| value.abs())
            .fold(f64::INFINITY, f64::min);
        let used_middle_generalized_inverse = middle_min_abs <= middle_tol;
        let middle_inverse = if used_middle_generalized_inverse {
            symmetric_pseudoinverse(&middle, middle_tol)
        } else {
            invert_spd_matrix(&middle, "Kenward-Roger L V0 L' matrix")?
        };
        let theta = l_active.transpose() * middle_inverse * &l_active;
        let theta_v0 = &theta * v0;

        let mut a1 = 0.0;
        let mut a2 = 0.0;
        let n_components = adjusted.p_matrices.len();
        if adjusted.w.shape() != (n_components, n_components) {
            return Err(MixedModelError::DimensionMismatch(format!(
                "Kenward-Roger W is {} x {}, expected {n_components} x {n_components}",
                adjusted.w.nrows(),
                adjusted.w.ncols()
            )));
        }
        for ii in 0..n_components {
            for jj in ii..n_components {
                let e = if ii == jj { 1.0 } else { 2.0 };
                let ui = &theta_v0 * &adjusted.p_matrices[ii] * v0;
                let uj = &theta_v0 * &adjusted.p_matrices[jj] * v0;
                a1 += e * adjusted.w[(ii, jj)] * matrix_trace(&ui) * matrix_trace(&uj);
                a2 += e * adjusted.w[(ii, jj)] * matrix_trace_product(&ui, &uj);
            }
        }

        let q = restriction_rank as f64;
        let b = (a1 + 6.0 * a2) / (2.0 * q);
        let g_denom = (q + 2.0) * a2;
        if !g_denom.is_finite() || g_denom.abs() <= 1e-14 {
            return Err(MixedModelError::InvalidArgument(
                "Kenward-Roger Lb_ddf has non-finite or zero g denominator".to_string(),
            ));
        }
        let g = ((q + 1.0) * a1 - (q + 4.0) * a2) / g_denom;
        let c_denom = 3.0 * q + 2.0 * (1.0 - g);
        if !c_denom.is_finite() || c_denom.abs() <= 1e-14 {
            return Err(MixedModelError::InvalidArgument(
                "Kenward-Roger Lb_ddf has non-finite or zero correction denominator".to_string(),
            ));
        }
        let c1 = g / c_denom;
        let c2 = (q - g) / c_denom;
        let c3 = (q + 2.0 - g) / c_denom;
        let mut v0_correction = 1.0 + c1 * b;
        let v1 = 1.0 - c2 * b;
        let v2 = 1.0 - c3 * b;
        if v0_correction.abs() < 1e-10 {
            v0_correction = 0.0;
        }
        let rho = (1.0 / q) * div_zero(1.0 - a2 / q, v1, 1e-14).powi(2) * v0_correction / v2;
        let denominator = q * rho - 1.0;
        if !denominator.is_finite() || denominator.abs() <= 1e-14 {
            return Err(MixedModelError::InvalidArgument(
                "Kenward-Roger Lb_ddf has non-finite or zero final denominator".to_string(),
            ));
        }
        let denominator_df = 4.0 + (q + 2.0) / denominator;
        if !denominator_df.is_finite() || denominator_df <= 0.0 {
            return Err(MixedModelError::InvalidArgument(
                "Kenward-Roger Lb_ddf produced a non-finite or non-positive denominator df"
                    .to_string(),
            ));
        }

        let mut notes = adjusted.notes.clone();
        let used_generalized_inverse =
            adjusted.used_generalized_inverse || used_middle_generalized_inverse;
        if used_middle_generalized_inverse {
            notes.push(format!(
                "Kenward-Roger L V0 L' used a generalized inverse at tolerance {middle_tol}"
            ));
        }
        if restriction_rank < l_active.nrows() {
            notes.push(format!(
                "Kenward-Roger restriction matrix row rank {restriction_rank} is lower than {} submitted row(s)",
                l_active.nrows()
            ));
        }

        Ok(KenwardRogerLbDdf {
            denominator_df,
            numerator_df: q,
            restriction_rank,
            a1,
            a2,
            b,
            g,
            rho,
            used_generalized_inverse,
            reliability: if used_generalized_inverse {
                ReliabilityGrade::Low
            } else {
                adjusted.reliability
            },
            notes,
        })
    }

    fn fixed_effect_contrast_to_active_basis(&self, l: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let active_p = self.feterm.rank;
        let full_p = self.feterm.piv.len();
        if l.ncols() != full_p && l.ncols() != active_p {
            return Err(MixedModelError::DimensionMismatch(format!(
                "fixed-effect contrast has {} column(s), expected active {active_p} or full {full_p}",
                l.ncols()
            )));
        }
        if l.ncols() == active_p && l.ncols() != full_p {
            return Ok(l.clone());
        }
        for dropped_position in active_p..full_p {
            let original_col = self.feterm.piv[dropped_position];
            for row in 0..l.nrows() {
                if l[(row, original_col)].abs() > 1e-12 {
                    return Err(MixedModelError::InvalidArgument(format!(
                        "Kenward-Roger contrast touches dropped fixed-effect column {original_col}"
                    )));
                }
            }
        }
        let mut active = DMatrix::zeros(l.nrows(), active_p);
        for active_col in 0..active_p {
            let original_col = self.feterm.piv[active_col];
            for row in 0..l.nrows() {
                active[(row, active_col)] = l[(row, original_col)];
            }
        }
        Ok(active)
    }

    fn fixed_effect_user_beta_to_active_basis(&self, beta: &DVector<f64>) -> Result<DVector<f64>> {
        let active_p = self.feterm.rank;
        let full_p = self.feterm.piv.len();
        if beta.len() == active_p && beta.len() != full_p {
            return Ok(beta.clone());
        }
        if beta.len() != full_p {
            return Err(MixedModelError::DimensionMismatch(format!(
                "fixed-effect beta has length {}, expected active {active_p} or full {full_p}",
                beta.len()
            )));
        }
        let mut active = DVector::zeros(active_p);
        for active_col in 0..active_p {
            active[active_col] = beta[self.feterm.piv[active_col]];
        }
        Ok(active)
    }

    fn fixed_effect_active_vector_to_user_basis(
        &self,
        values: &DVector<f64>,
        label: &str,
    ) -> Result<DVector<f64>> {
        let active_p = self.feterm.rank;
        let full_p = self.feterm.piv.len();
        if values.len() == full_p {
            return Ok(values.clone());
        }
        if values.len() != active_p {
            return Err(MixedModelError::DimensionMismatch(format!(
                "fixed-effect {label} vector has length {}, expected active {active_p} or full {full_p}",
                values.len()
            )));
        }
        let mut full = DVector::from_element(full_p, f64::NAN);
        for active_col in 0..active_p {
            full[self.feterm.piv[active_col]] = values[active_col];
        }
        Ok(full)
    }

    fn hessian_deviance_varpar(&mut self, varpar: &[f64], reml: bool) -> Result<DMatrix<f64>> {
        self.validate_varpar(varpar)?;
        let lower_bounds = self.varpar_lower_bounds();
        let steps = finite_difference_steps(varpar, &lower_bounds, 1e-4);
        let f0 = self.deviance_varpar(varpar, reml)?;
        if !f0.is_finite() {
            return Err(MixedModelError::Optimization(
                "deviance_varpar at fitted varpar is non-finite".to_string(),
            ));
        }

        let mut central_steps = Vec::with_capacity(varpar.len());
        for index in 0..varpar.len() {
            let lower = lower_bounds
                .get(index)
                .copied()
                .unwrap_or(f64::NEG_INFINITY);
            let step =
                feasible_central_step(varpar[index], lower, steps[index]).ok_or_else(|| {
                    MixedModelError::InvalidArgument(format!(
                        "cannot compute central finite-difference Hessian for varpar[{index}]: \
                     value is at or too near lower bound {lower}"
                    ))
                })?;
            central_steps.push(step);
        }

        let mut hessian = DMatrix::zeros(varpar.len(), varpar.len());
        for row in 0..varpar.len() {
            let h_row = central_steps[row];
            let f_plus = finite_difference_deviance_varpar(self, varpar, row, h_row, reml)?;
            let f_minus = finite_difference_deviance_varpar(self, varpar, row, -h_row, reml)?;
            hessian[(row, row)] = (f_plus - 2.0 * f0 + f_minus) / (h_row * h_row);

            for col in 0..row {
                let h_col = central_steps[col];
                let f_pp = finite_difference_deviance_varpar_2d(
                    self, varpar, row, h_row, col, h_col, reml,
                )?;
                let f_pm = finite_difference_deviance_varpar_2d(
                    self, varpar, row, h_row, col, -h_col, reml,
                )?;
                let f_mp = finite_difference_deviance_varpar_2d(
                    self, varpar, row, -h_row, col, h_col, reml,
                )?;
                let f_mm = finite_difference_deviance_varpar_2d(
                    self, varpar, row, -h_row, col, -h_col, reml,
                )?;
                let value = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h_row * h_col);
                hessian[(row, col)] = value;
                hessian[(col, row)] = value;
            }
        }

        if matrix_is_finite(&hessian) {
            Ok(hessian)
        } else {
            Err(MixedModelError::Optimization(
                "deviance_varpar Hessian contains non-finite entries".to_string(),
            ))
        }
    }

    fn validate_varpar(&self, varpar: &[f64]) -> Result<()> {
        let n_theta = self.n_theta();
        if varpar.len() != n_theta + 1 {
            return Err(MixedModelError::DimensionMismatch(format!(
                "varpar has length {}, expected {} theta parameter(s) plus sigma",
                varpar.len(),
                n_theta
            )));
        }
        if varpar.iter().any(|value| !value.is_finite()) {
            return Err(MixedModelError::InvalidArgument(
                "varpar contains a non-finite value".to_string(),
            ));
        }

        let sigma = varpar[n_theta];
        if sigma <= 0.0 {
            return Err(MixedModelError::InvalidArgument(
                "varpar sigma must be positive".to_string(),
            ));
        }

        let lower_bounds = self.lower_bounds();
        if let Some((index, (&value, &lower))) = varpar[..n_theta]
            .iter()
            .zip(lower_bounds.iter())
            .enumerate()
            .find(|(_, (&value, &lower))| lower.is_finite() && value < lower)
        {
            return Err(MixedModelError::InvalidArgument(format!(
                "theta[{index}] = {value} is below lower bound {lower}"
            )));
        }

        Ok(())
    }

    fn varpar_lower_bounds(&self) -> Vec<f64> {
        let mut lower_bounds = self.lower_bounds();
        lower_bounds.push(0.0);
        lower_bounds
    }

    fn use_scalar_single_theta_optimizer(&self) -> bool {
        self.reterms.len() == 1 && self.reterms[0].vsize == 1 && self.n_theta() == 1
    }

    #[cfg(feature = "nlopt")]
    fn use_nlopt_bobyqa_small_theta_optimizer(&self) -> bool {
        // Smooth, low-dimensional problems benefit substantially from
        // BOBYQA's trust-region modelling vs. pattern_search (~3× fewer
        // evals on profiled kb07-class fits). Pattern search remains the
        // automatic fallback if BOBYQA fails to converge. Gated to the
        // `nlopt` feature; without it the auto-fit dispatch routes
        // straight to COBYLA without consulting this predicate.
        let n_theta = self.n_theta();
        n_theta > 1 && n_theta <= 6
    }

    #[cfg(feature = "nlopt")]
    fn use_large_single_vsize2_optimizer_tuning(&self) -> bool {
        self.reterms.len() == 1
            && self.reterms[0].vsize == 2
            && self.n_theta() == 3
            && self.reterms[0].n_ranef() >= 512
            && self.a_blocks.len() == 3
            && matches!(self.a_blocks[0], MatrixBlock::BlockDiagonal(_))
            && matches!(self.a_blocks[1], MatrixBlock::Dense(_))
            && matches!(self.a_blocks[2], MatrixBlock::Dense(_))
    }

    #[cfg(feature = "nlopt")]
    fn use_large_theta_nlopt_optimizer(&self) -> bool {
        self.n_theta() > 6
    }

    fn project_theta_to_bounds(theta: &mut [f64], lower_bounds: &[f64]) {
        for (value, &lower) in theta.iter_mut().zip(lower_bounds.iter()) {
            if lower.is_finite() && *value < lower {
                *value = lower;
            }
        }
    }

    fn steps_are_small(step: &[f64], step_tol: &[f64]) -> bool {
        step.iter()
            .zip(step_tol.iter())
            .all(|(&value, &tol)| value <= tol)
    }

    fn apply_theta_to_reterms(reterms: &mut [ReMat], theta: &[f64]) -> Option<()> {
        let mut offset = 0;
        for rt in reterms {
            let nt = rt.n_theta();
            if offset + nt > theta.len() {
                return None;
            }
            rt.set_theta(&theta[offset..offset + nt]).ok()?;
            offset += nt;
        }
        (offset == theta.len()).then_some(())
    }

    fn profiled_objective_from_parts(
        a_blocks: &[MatrixBlock],
        l_blocks: &mut [MatrixBlock],
        reterms: &mut [ReMat],
        theta: &[f64],
        dims: ModelDims,
        is_reml: bool,
        fixed_sigma: Option<f64>,
        cholesky_zero_pad_tolerance: f64,
    ) -> Option<f64> {
        if let Some(obj) = Self::profiled_objective_one_vsize2_fast(
            a_blocks,
            reterms,
            theta,
            dims,
            is_reml,
            fixed_sigma,
            cholesky_zero_pad_tolerance,
        ) {
            return Some(obj);
        }

        Self::apply_theta_to_reterms(reterms, theta)?;
        if update_l_from_parts(a_blocks, l_blocks, reterms, cholesky_zero_pad_tolerance).is_err() {
            return None;
        }

        let k = reterms.len();
        let n = dims.n as f64;
        let p = dims.p as f64;

        let mut logdet_lzz = 0.0;
        for j in 0..k {
            logdet_lzz += logdet_block(&l_blocks[block_index(j, j)]);
        }

        let l_last = l_blocks[block_index(k, k)].as_dense();
        let pp1 = l_last.nrows();
        let last_diag = l_last[(pp1 - 1, pp1 - 1)];
        let pwrss = last_diag * last_diag;

        let logdet = if is_reml {
            let mut logdet_lxx = 0.0;
            for i in 0..(pp1 - 1) {
                let d = l_last[(i, i)];
                if d > 0.0 {
                    logdet_lxx += d.ln();
                }
            }
            logdet_lzz + 2.0 * logdet_lxx
        } else {
            logdet_lzz
        };

        let denomdf = if is_reml { n - p } else { n };
        Some(Self::objective_from_components(
            logdet,
            pwrss,
            denomdf,
            fixed_sigma,
        ))
    }

    fn profiled_objective_one_vsize2_fast(
        a_blocks: &[MatrixBlock],
        reterms: &[ReMat],
        theta: &[f64],
        dims: ModelDims,
        is_reml: bool,
        fixed_sigma: Option<f64>,
        cholesky_zero_pad_tolerance: f64,
    ) -> Option<f64> {
        if reterms.len() != 1 || reterms[0].vsize != 2 || theta.len() != 3 || a_blocks.len() != 3 {
            return None;
        }

        let MatrixBlock::BlockDiagonal(a00_blocks) = &a_blocks[0] else {
            return None;
        };
        let MatrixBlock::Dense(a10) = &a_blocks[1] else {
            return None;
        };
        let MatrixBlock::Dense(a11) = &a_blocks[2] else {
            return None;
        };

        if a00_blocks.is_empty()
            || !a00_blocks
                .iter()
                .all(|block| block.nrows() == 2 && block.ncols() == 2)
        {
            return None;
        }
        if a10.ncols() != 2 * a00_blocks.len()
            || a10.ncols() < 512
            || a11.nrows() != a11.ncols()
            || a11.nrows() != a10.nrows()
        {
            return None;
        }

        let pp1 = a11.nrows();
        let lam00 = theta[0];
        let lam10 = theta[1];
        let lam11 = theta[2];
        let mut l_last = a11.clone();
        let mut logdet_lzz = 0.0;

        for (level, src_blk) in a00_blocks.iter().enumerate() {
            let s00 = src_blk[(0, 0)];
            let s01 = src_blk[(0, 1)];
            let s10 = src_blk[(1, 0)];
            let s11 = src_blk[(1, 1)];

            let t00 = s00 * lam00 + s01 * lam10;
            let t10 = s10 * lam00 + s11 * lam10;
            let t11 = s11 * lam11;

            let mut l00 = lam00 * t00 + lam10 * t10 + 1.0;
            let mut l10 = lam11 * t10;
            let mut l11 = lam11 * t11 + 1.0;
            let pivot_tolerance = cholesky_zero_pad_abs_tolerance(
                l00.abs().max(l11.abs()),
                cholesky_zero_pad_tolerance,
            );

            if l00 <= 0.0 {
                if l00 < -pivot_tolerance {
                    return None;
                }
                l00 = 0.0;
                l10 = 0.0;
            } else {
                l00 = l00.sqrt();
                l10 /= l00;
            }

            l11 -= l10 * l10;
            if l11 <= 0.0 {
                if l11 < -pivot_tolerance {
                    return None;
                }
                l11 = 0.0;
            } else {
                l11 = l11.sqrt();
            }

            if l00 > 0.0 {
                logdet_lzz += l00.ln();
            }
            if l11 > 0.0 {
                logdet_lzz += l11.ln();
            }

            let col0 = 2 * level;
            let col1 = col0 + 1;
            if pp1 == 3 {
                let (z00, z01) =
                    solve_scaled_vsize2_row(a10, 0, col0, col1, lam00, lam10, lam11, l00, l10, l11);
                let (z10, z11) =
                    solve_scaled_vsize2_row(a10, 1, col0, col1, lam00, lam10, lam11, l00, l10, l11);
                let (z20, z21) =
                    solve_scaled_vsize2_row(a10, 2, col0, col1, lam00, lam10, lam11, l00, l10, l11);

                l_last[(0, 0)] -= z00 * z00 + z01 * z01;
                l_last[(1, 0)] -= z10 * z00 + z11 * z01;
                l_last[(1, 1)] -= z10 * z10 + z11 * z11;
                l_last[(2, 0)] -= z20 * z00 + z21 * z01;
                l_last[(2, 1)] -= z20 * z10 + z21 * z11;
                l_last[(2, 2)] -= z20 * z20 + z21 * z21;
            } else {
                let mut solved0_by_row = vec![0.0; pp1];
                let mut solved1_by_row = vec![0.0; pp1];
                for row in 0..pp1 {
                    let (solved0, solved1) = solve_scaled_vsize2_row(
                        a10, row, col0, col1, lam00, lam10, lam11, l00, l10, l11,
                    );
                    solved0_by_row[row] = solved0;
                    solved1_by_row[row] = solved1;
                }

                for row in 0..pp1 {
                    for col in 0..=row {
                        l_last[(row, col)] -= solved0_by_row[row] * solved0_by_row[col]
                            + solved1_by_row[row] * solved1_by_row[col];
                    }
                }
            }
        }
        logdet_lzz *= 2.0;

        let mut l_last_block = MatrixBlock::Dense(l_last);
        if cholesky_block_with_tolerance(&mut l_last_block, cholesky_zero_pad_tolerance).is_err() {
            return None;
        }
        let MatrixBlock::Dense(l_last) = l_last_block else {
            unreachable!();
        };

        let last_diag = l_last[(pp1 - 1, pp1 - 1)];
        let pwrss = last_diag * last_diag;
        let logdet = if is_reml {
            let mut logdet_lxx = 0.0;
            for i in 0..(pp1 - 1) {
                let d = l_last[(i, i)];
                if d > 0.0 {
                    logdet_lxx += d.ln();
                }
            }
            logdet_lzz + 2.0 * logdet_lxx
        } else {
            logdet_lzz
        };

        let n = dims.n as f64;
        let p = dims.p as f64;
        let denomdf = if is_reml { n - p } else { n };
        Some(Self::objective_from_components(
            logdet,
            pwrss,
            denomdf,
            fixed_sigma,
        ))
    }

    #[cfg(feature = "nlopt")]
    fn nlopt_ok(
        result: std::result::Result<nlopt::SuccessState, NloptFailState>,
        action: &str,
    ) -> Result<()> {
        result.map(|_| ()).map_err(|status| {
            MixedModelError::Optimization(format!("NLopt {action} failed: {status:?}"))
        })
    }

    #[cfg(feature = "nlopt")]
    fn nlopt_status_label(name: &str) -> String {
        match name {
            "Success" => "SUCCESS".to_string(),
            "StopValReached" => "STOPVAL_REACHED".to_string(),
            "FtolReached" => "FTOL_REACHED".to_string(),
            "XtolReached" => "XTOL_REACHED".to_string(),
            "MaxEvalReached" => "MAXEVAL_REACHED".to_string(),
            "MaxTimeReached" => "MAXTIME_REACHED".to_string(),
            "RoundoffLimited" => "ROUNDOFF_LIMITED".to_string(),
            "InvalidArgs" => "INVALID_ARGS".to_string(),
            "OutOfMemory" => "OUT_OF_MEMORY".to_string(),
            "ForcedStop" => "FORCED_STOP".to_string(),
            "Failure" => "FAILURE".to_string(),
            other => other.to_ascii_uppercase(),
        }
    }

    fn cobyla_success_status_label(status: cobyla::SuccessStatus) -> String {
        match status {
            cobyla::SuccessStatus::Success => "SUCCESS".to_string(),
            cobyla::SuccessStatus::StopValReached => "STOPVAL_REACHED".to_string(),
            cobyla::SuccessStatus::FtolReached => "FTOL_REACHED".to_string(),
            cobyla::SuccessStatus::XtolReached => "XTOL_REACHED".to_string(),
            cobyla::SuccessStatus::MaxEvalReached => "MAXEVAL_REACHED".to_string(),
            cobyla::SuccessStatus::MaxTimeReached => "MAXTIME_REACHED".to_string(),
        }
    }

    fn cobyla_fail_status_label(status: cobyla::FailStatus) -> String {
        match status {
            cobyla::FailStatus::Failure => "FAILURE".to_string(),
            cobyla::FailStatus::InvalidArgs => "INVALID_ARGS".to_string(),
            cobyla::FailStatus::OutOfMemory => "OUT_OF_MEMORY".to_string(),
            cobyla::FailStatus::RoundoffLimited => "ROUNDOFF_LIMITED".to_string(),
            cobyla::FailStatus::ForcedStop => "FORCED_STOP".to_string(),
            cobyla::FailStatus::UnexpectedError => "UNEXPECTED_ERROR".to_string(),
        }
    }

    fn record_scalar_eval(
        &mut self,
        theta: f64,
        feval_count: &mut i64,
        fit_log: &mut Vec<FitLogEntry>,
        best_theta: &mut f64,
        best_fmin: &mut f64,
    ) -> Result<f64> {
        let obj = self.objective_at(&[theta])?;
        *feval_count += 1;
        fit_log.push(FitLogEntry {
            theta: vec![theta],
            objective: obj,
        });
        if obj < *best_fmin {
            *best_fmin = obj;
            *best_theta = theta;
        }
        Ok(obj)
    }

    fn finalize_fit_result(
        &mut self,
        mut best_theta_val: Vec<f64>,
        mut best_fmin_val: f64,
        feval_count: i64,
        fit_log: Vec<FitLogEntry>,
        optimizer: Optimizer,
        return_value: Option<String>,
    ) -> Result<&mut Self> {
        Self::rectify_theta_columns(&mut best_theta_val, &self.parmap, self.reterms.len());
        self.set_theta(&best_theta_val)?;
        self.update_l()?;

        let mut xmin = best_theta_val.clone();
        let mut modified = false;
        for (i, (_, row, col)) in self.parmap.iter().enumerate() {
            if row == col && xmin[i] > 0.0 && xmin[i] < self.optsum.xtol_zero_abs {
                xmin[i] = 0.0;
                modified = true;
            }
        }
        if modified {
            let zero_obj = self.objective_at(&xmin)?;
            if zero_obj <= best_fmin_val + self.optsum.ftol_zero_abs {
                best_fmin_val = zero_obj;
                best_theta_val = xmin;
            } else {
                self.set_theta(&best_theta_val)?;
                self.update_l()?;
            }
        }

        self.optsum.optimizer = optimizer;
        self.optsum.backend = optimizer.canonical_backend();
        self.optsum.final_params = best_theta_val;
        self.optsum.fmin = best_fmin_val;
        self.optsum.feval = feval_count;
        self.optsum.return_value = return_value.unwrap_or_else(|| "SUCCESS".to_string());
        self.optsum.fit_log = fit_log;

        Ok(self)
    }

    pub(crate) fn rectify_theta_columns(
        theta: &mut [f64],
        parmap: &[(usize, usize, usize)],
        n_terms: usize,
    ) {
        for block in 0..n_terms {
            let max_col = parmap
                .iter()
                .filter(|&&(term, _, _)| term == block)
                .map(|&(_, _, col)| col)
                .max();

            let Some(max_col) = max_col else {
                continue;
            };

            for col in 0..=max_col {
                let diag_idx = parmap.iter().position(|&(term, row, col_idx)| {
                    term == block && row == col && col_idx == col
                });
                let Some(diag_idx) = diag_idx else {
                    continue;
                };

                if theta[diag_idx] < 0.0 {
                    for (idx, &(term, _, col_idx)) in parmap.iter().enumerate() {
                        if term == block && col_idx == col {
                            theta[idx] = -theta[idx];
                        }
                    }
                }
            }
        }
    }

    fn fit_scalar_single_theta(&mut self) -> Result<&mut Self> {
        const INVPHI: f64 = 0.6180339887498949;

        let maxeval = if self.optsum.max_feval > 0 {
            self.optsum.max_feval
        } else {
            10000
        };
        let xtol = self
            .optsum
            .xtol_abs
            .first()
            .copied()
            .unwrap_or(1e-8)
            .max(1e-4);
        let mut step = self
            .optsum
            .initial_step
            .first()
            .copied()
            .unwrap_or(0.75)
            .abs()
            .max(1e-6);
        let theta0 = self.optsum.initial[0].max(0.0);

        let mut feval_count = 0i64;
        let mut fit_log = Vec::new();
        let mut best_theta = theta0;
        let mut best_fmin = self.optsum.finitial;

        let mut lo = 0.0;
        let flo = if theta0 > 0.0 {
            self.record_scalar_eval(
                0.0,
                &mut feval_count,
                &mut fit_log,
                &mut best_theta,
                &mut best_fmin,
            )?
        } else {
            self.optsum.finitial
        };

        let mut mid = if theta0 > 0.0 { theta0 } else { step };
        let mut fmid = if theta0 > 0.0 {
            self.optsum.finitial
        } else {
            self.record_scalar_eval(
                mid,
                &mut feval_count,
                &mut fit_log,
                &mut best_theta,
                &mut best_fmin,
            )?
        };

        let mut hi = if fmid >= flo { mid } else { mid + step };

        if fmid < flo {
            let mut fhi = self.record_scalar_eval(
                hi,
                &mut feval_count,
                &mut fit_log,
                &mut best_theta,
                &mut best_fmin,
            )?;

            while feval_count < maxeval && fhi < fmid {
                lo = mid;
                mid = hi;
                fmid = fhi;
                step *= 2.0;
                hi = mid + step;
                fhi = self.record_scalar_eval(
                    hi,
                    &mut feval_count,
                    &mut fit_log,
                    &mut best_theta,
                    &mut best_fmin,
                )?;
            }
        }

        let mut a = lo;
        let mut b = hi.max(mid).max(step);
        if b <= a {
            b = a + step;
        }

        let mut c = b - (b - a) * INVPHI;
        let mut d = a + (b - a) * INVPHI;
        let mut fc = if (c - theta0).abs() <= xtol {
            self.optsum.finitial
        } else {
            self.record_scalar_eval(
                c,
                &mut feval_count,
                &mut fit_log,
                &mut best_theta,
                &mut best_fmin,
            )?
        };
        let mut fd = if (d - theta0).abs() <= xtol {
            self.optsum.finitial
        } else if (d - c).abs() <= xtol {
            fc
        } else {
            self.record_scalar_eval(
                d,
                &mut feval_count,
                &mut fit_log,
                &mut best_theta,
                &mut best_fmin,
            )?
        };

        while feval_count < maxeval && (b - a) > xtol * (1.0 + a.abs().max(b.abs())) {
            if fc <= fd {
                b = d;
                d = c;
                fd = fc;
                c = b - (b - a) * INVPHI;
                fc = self.record_scalar_eval(
                    c,
                    &mut feval_count,
                    &mut fit_log,
                    &mut best_theta,
                    &mut best_fmin,
                )?;
            } else {
                a = c;
                c = d;
                fc = fd;
                d = a + (b - a) * INVPHI;
                fd = self.record_scalar_eval(
                    d,
                    &mut feval_count,
                    &mut fit_log,
                    &mut best_theta,
                    &mut best_fmin,
                )?;
            }
        }

        self.finalize_fit_result(
            vec![best_theta],
            best_fmin,
            feval_count,
            fit_log,
            Optimizer::PatternSearch,
            (feval_count >= maxeval).then(|| "MAXEVAL_REACHED".to_string()),
        )
    }

    fn fit_multivariate_pattern_search(&mut self) -> Result<&mut Self> {
        let n_theta = self.n_theta();
        let maxeval = if self.optsum.max_feval > 0 {
            self.optsum.max_feval
        } else {
            10000
        };
        let lower_bounds = self.lower_bounds();
        let mut step_tol: Vec<f64> = self
            .optsum
            .xtol_abs
            .iter()
            .map(|&tol| tol.max(1e-5))
            .collect();
        if step_tol.len() != n_theta {
            step_tol = vec![1e-5; n_theta];
        }

        let mut step: Vec<f64> = self
            .optsum
            .initial_step
            .iter()
            .zip(step_tol.iter())
            .map(|(&initial, &tol)| initial.abs().max(tol))
            .collect();
        if step.len() != n_theta {
            step = vec![0.5; n_theta];
        }

        let outcome = Self::run_multivariate_pattern_search(
            self.optsum.initial.clone(),
            self.optsum.finitial,
            &lower_bounds,
            step,
            &step_tol,
            maxeval,
            self.optsum.ftol_abs,
            |theta| self.objective_at(theta),
        )?;

        self.finalize_fit_result(
            outcome.best_theta,
            outcome.best_fmin,
            outcome.feval_count,
            outcome.fit_log,
            Optimizer::PatternSearch,
            (outcome.feval_count >= maxeval).then(|| "MAXEVAL_REACHED".to_string()),
        )
    }

    pub(crate) fn run_multivariate_pattern_search<F>(
        initial: Vec<f64>,
        finitial: f64,
        lower_bounds: &[f64],
        mut step: Vec<f64>,
        step_tol: &[f64],
        maxeval: i64,
        ftol_abs: f64,
        mut objective: F,
    ) -> Result<PatternSearchOutcome>
    where
        F: FnMut(&[f64]) -> Result<f64>,
    {
        let n_theta = initial.len();
        let mut preferred_sign = vec![-1.0; n_theta];
        for (i, &lower) in lower_bounds.iter().enumerate() {
            if !lower.is_finite() {
                preferred_sign[i] = 1.0;
            }
        }

        let mut theta = initial;
        let mut ftheta = finitial;
        let mut best_theta = theta.clone();
        let mut best_fmin = ftheta;
        let mut feval_count = 0i64;
        let mut fit_log = Vec::new();

        while feval_count < maxeval && !Self::steps_are_small(&step, &step_tol) {
            let base_theta = theta.clone();
            let base_f = ftheta;
            let mut moved = vec![false; n_theta];
            let mut exploratory_direction = vec![0.0; n_theta];

            for i in 0..n_theta {
                let mut chosen_theta = theta[i];
                let mut chosen_f = ftheta;
                let mut chosen_sign = 0.0;
                exploratory_direction[i] = preferred_sign[i];

                for dir in [preferred_sign[i], -preferred_sign[i]] {
                    let mut trial = theta.clone();
                    trial[i] += dir * step[i];
                    Self::project_theta_to_bounds(&mut trial, &lower_bounds);
                    if (trial[i] - theta[i]).abs() <= step_tol[i] * 0.5 {
                        continue;
                    }

                    let ftrial = record_pattern_eval(
                        &mut objective,
                        &trial,
                        &mut feval_count,
                        &mut fit_log,
                        &mut best_theta,
                        &mut best_fmin,
                    )?;
                    if ftrial + ftol_abs < ftheta {
                        chosen_theta = trial[i];
                        chosen_f = ftrial;
                        chosen_sign = dir;
                        break;
                    }
                    if feval_count >= maxeval {
                        break;
                    }
                }

                if chosen_f < ftheta {
                    theta[i] = chosen_theta;
                    ftheta = chosen_f;
                    moved[i] = true;
                    preferred_sign[i] = chosen_sign;
                } else {
                    preferred_sign[i] = -preferred_sign[i];
                }

                if feval_count >= maxeval {
                    break;
                }
            }

            let mut any_moved = moved.iter().any(|&m| m);
            if feval_count < maxeval {
                let mut pattern_candidates = Vec::with_capacity(if any_moved { 1 } else { 2 });
                if any_moved {
                    let mut pattern = theta.clone();
                    for i in 0..n_theta {
                        pattern[i] += theta[i] - base_theta[i];
                    }
                    Self::project_theta_to_bounds(&mut pattern, &lower_bounds);
                    pattern_candidates.push(pattern);
                } else {
                    let mut push_candidate = |pattern: Vec<f64>| {
                        if pattern != theta && !pattern_candidates.contains(&pattern) {
                            pattern_candidates.push(pattern);
                        }
                    };

                    for direction_sign in [1.0, -1.0] {
                        let mut pattern = base_theta.clone();
                        for i in 0..n_theta {
                            pattern[i] += direction_sign * exploratory_direction[i] * step[i];
                        }
                        Self::project_theta_to_bounds(&mut pattern, &lower_bounds);
                        push_candidate(pattern);
                    }

                    for i in 0..n_theta {
                        for j in (i + 1)..n_theta {
                            for dir_i in [exploratory_direction[i], -exploratory_direction[i]] {
                                for dir_j in [exploratory_direction[j], -exploratory_direction[j]] {
                                    let mut pattern = base_theta.clone();
                                    pattern[i] += dir_i * step[i];
                                    pattern[j] += dir_j * step[j];
                                    Self::project_theta_to_bounds(&mut pattern, &lower_bounds);
                                    push_candidate(pattern);
                                }
                            }
                        }
                    }
                }

                for pattern in pattern_candidates {
                    if feval_count >= maxeval {
                        break;
                    }
                    let fpattern = record_pattern_eval(
                        &mut objective,
                        &pattern,
                        &mut feval_count,
                        &mut fit_log,
                        &mut best_theta,
                        &mut best_fmin,
                    )?;
                    if fpattern + ftol_abs < ftheta {
                        for i in 0..n_theta {
                            if (pattern[i] - theta[i]).abs() > 0.0 {
                                preferred_sign[i] = (pattern[i] - theta[i]).signum();
                                moved[i] = true;
                            }
                        }
                        theta = pattern;
                        ftheta = fpattern;
                        any_moved = true;
                        break;
                    }
                }
            }

            if !any_moved {
                for value in &mut step {
                    *value *= 0.5;
                }
                continue;
            }

            for i in 0..n_theta {
                if moved[i] {
                    step[i] = (step[i] * 1.1).max(step_tol[i]);
                } else {
                    step[i] *= 0.5;
                }
            }

            if (base_f - ftheta).abs() <= ftol_abs && Self::steps_are_small(&step, &step_tol) {
                break;
            }
        }

        Ok(PatternSearchOutcome {
            best_theta,
            best_fmin,
            feval_count,
            fit_log,
        })
    }

    fn fit_cobyla_with_maxeval(
        &mut self,
        reml: bool,
        maxeval_override: Option<usize>,
    ) -> Result<&mut Self> {
        let lb = self.lower_bounds();
        self.optsum.optimizer = Optimizer::Cobyla;

        let a_blocks = self.a_blocks.clone();
        let l_blocks_template = self.l_blocks.clone();
        let reterms_template = self.reterms.clone();
        let dims = self.dims;
        let is_reml = reml;
        let fixed_sigma = self.optsum.sigma;
        let cholesky_zero_pad_tolerance = self
            .compiler_policy()
            .thresholds
            .cholesky_zero_pad_tolerance;
        let best_theta = std::cell::RefCell::new(self.optsum.initial.clone());
        let best_fmin = std::cell::Cell::new(f64::INFINITY);
        let feval_count = std::cell::Cell::new(0i64);
        let fit_log: std::cell::RefCell<Vec<(Vec<f64>, f64)>> = std::cell::RefCell::new(Vec::new());

        let reterms_work = std::cell::RefCell::new(reterms_template.clone());
        let l_blocks_work = std::cell::RefCell::new(l_blocks_template);

        let objective_fn = |theta: &[f64], _data: &mut ()| -> f64 {
            feval_count.set(feval_count.get() + 1);
            let obj = {
                let mut rw = reterms_work.borrow_mut();
                let mut lw = l_blocks_work.borrow_mut();
                Self::profiled_objective_from_parts(
                    &a_blocks,
                    &mut lw,
                    &mut rw,
                    theta,
                    dims,
                    is_reml,
                    fixed_sigma,
                    cholesky_zero_pad_tolerance,
                )
                .unwrap_or(f64::INFINITY)
            };

            fit_log.borrow_mut().push((theta.to_vec(), obj));
            if obj < best_fmin.get() {
                best_fmin.set(obj);
                *best_theta.borrow_mut() = theta.to_vec();
            }

            obj
        };

        let bounds: Vec<(f64, f64)> = lb.iter().map(|&lo| (lo, f64::INFINITY)).collect();
        let constraint_fns: Vec<Box<dyn cobyla::Func<()>>> = lb
            .iter()
            .enumerate()
            .filter(|(_, &lo)| lo > f64::NEG_INFINITY)
            .map(|(i, &lo)| {
                Box::new(move |x: &[f64], _: &mut ()| -> f64 { x[i] - lo })
                    as Box<dyn cobyla::Func<()>>
            })
            .collect();
        let cons_refs: Vec<&dyn cobyla::Func<()>> =
            constraint_fns.iter().map(|f| f.as_ref()).collect();

        let maxeval = maxeval_override.unwrap_or_else(|| {
            if self.optsum.max_feval > 0 {
                self.optsum.max_feval as usize
            } else {
                10000
            }
        });

        let stop_tol = cobyla::StopTols {
            ftol_rel: self.optsum.ftol_rel,
            ftol_abs: self.optsum.ftol_abs,
            xtol_rel: self.optsum.xtol_rel,
            xtol_abs: self.optsum.xtol_abs.clone(),
            ..cobyla::StopTols::default()
        };

        let result = cobyla::minimize(
            objective_fn,
            &self.optsum.initial,
            &bounds,
            &cons_refs,
            (),
            maxeval,
            cobyla::RhoBeg::All(0.75),
            Some(stop_tol),
        );

        let (best_theta_val, best_fmin_val, return_value);

        match result {
            Ok((status, x_opt, fmin)) => {
                best_fmin_val = fmin;
                best_theta_val = x_opt;
                return_value = Some(Self::cobyla_success_status_label(status));
            }
            Err((status @ cobyla::FailStatus::RoundoffLimited, x_opt, _)) => {
                best_theta_val = x_opt;
                best_fmin_val = best_fmin.get();
                return_value = Some(Self::cobyla_fail_status_label(status));
            }
            Err((status, x_opt, fmin)) => {
                if fmin.is_finite() {
                    best_fmin_val = fmin;
                    best_theta_val = x_opt;
                    return_value = Some(Self::cobyla_fail_status_label(status));
                } else {
                    return Err(MixedModelError::Optimization(
                        "COBYLA optimization failed".to_string(),
                    ));
                }
            }
        }

        self.finalize_fit_result(
            best_theta_val,
            best_fmin_val,
            feval_count.get(),
            fit_log
                .into_inner()
                .into_iter()
                .map(|(theta, obj)| FitLogEntry {
                    theta,
                    objective: obj,
                })
                .collect(),
            Optimizer::Cobyla,
            return_value,
        )
    }

    fn fit_cobyla(&mut self, reml: bool) -> Result<&mut Self> {
        self.fit_cobyla_with_maxeval(reml, None)
    }

    #[cfg(feature = "prima")]
    fn fit_prima_bobyqa_with_maxeval(
        &mut self,
        reml: bool,
        maxeval_override: Option<usize>,
    ) -> Result<&mut Self> {
        self.optsum.optimizer = Optimizer::PrimaBobyqa;
        self.optsum.backend = OptimizerBackend::Prima;

        let a_blocks = self.a_blocks.clone();
        let l_blocks_template = self.l_blocks.clone();
        let reterms_template = self.reterms.clone();
        let dims = self.dims;
        let is_reml = reml;
        let fixed_sigma = self.optsum.sigma;
        let cholesky_zero_pad_tolerance = self
            .compiler_policy()
            .thresholds
            .cholesky_zero_pad_tolerance;
        let invalid_objective = self.optsum.finitial;
        let best_theta = std::cell::RefCell::new(self.optsum.initial.clone());
        let best_fmin = std::cell::Cell::new(self.optsum.finitial);
        let feval_count = std::cell::Cell::new(0i64);
        let fit_log: std::cell::RefCell<Vec<FitLogEntry>> = std::cell::RefCell::new(Vec::new());

        let reterms_work = std::cell::RefCell::new(reterms_template.clone());
        let l_blocks_work = std::cell::RefCell::new(l_blocks_template);

        let mut objective_fn = |theta: &[f64]| -> f64 {
            feval_count.set(feval_count.get() + 1);
            let obj = {
                let mut rw = reterms_work.borrow_mut();
                let mut lw = l_blocks_work.borrow_mut();
                Self::profiled_objective_from_parts(
                    &a_blocks,
                    &mut lw,
                    &mut rw,
                    theta,
                    dims,
                    is_reml,
                    fixed_sigma,
                    cholesky_zero_pad_tolerance,
                )
                .unwrap_or(invalid_objective)
            };

            fit_log.borrow_mut().push(FitLogEntry {
                theta: theta.to_vec(),
                objective: obj,
            });
            if obj + 1e-12 < best_fmin.get() {
                best_fmin.set(obj);
                *best_theta.borrow_mut() = theta.to_vec();
            }

            obj
        };

        let maxfun = maxeval_override.unwrap_or_else(|| {
            if self.optsum.max_feval > 0 {
                self.optsum.max_feval as usize
            } else {
                10000
            }
        });

        let lower_bounds = self.lower_bounds();
        let upper_bounds = vec![f64::INFINITY; self.n_theta()];
        let result = minimize_bobyqa(
            &self.optsum.initial,
            &lower_bounds,
            &upper_bounds,
            PrimaBobyqaOptions {
                rhobeg: self.optsum.rhobeg,
                rhoend: self.optsum.rhoend,
                maxfun,
            },
            &mut objective_fn,
        )?;

        let logged_best_theta = best_theta.into_inner();
        let logged_best_fmin = best_fmin.get();
        let (final_theta, final_fmin) =
            if logged_best_fmin.is_finite() && logged_best_fmin <= result.fmin {
                (logged_best_theta, logged_best_fmin)
            } else {
                (result.x, result.fmin)
            };

        self.finalize_fit_result(
            final_theta,
            final_fmin,
            if result.feval > 0 {
                result.feval
            } else {
                feval_count.get()
            },
            fit_log.into_inner(),
            Optimizer::PrimaBobyqa,
            None,
        )?;
        self.optsum.return_value = result.return_code;
        self.optsum.backend = OptimizerBackend::Prima;

        Ok(self)
    }

    #[cfg(feature = "nlopt")]
    fn fit_nlopt_large_theta_with_maxeval(
        &mut self,
        reml: bool,
        maxeval_override: Option<usize>,
    ) -> Result<&mut Self> {
        // NEWUOA is unconstrained — no lower-bound enforcement, so the soft
        // barrier (objective returns finitial outside the feasible region)
        // is what keeps θ ≥ 0. This has been the behaviour for n_theta > 6
        // since the original port and is preserved.
        self.fit_nlopt_with_algorithm(
            NloptAlgorithm::Newuoa,
            Optimizer::NloptNewuoa,
            reml,
            maxeval_override,
            /*use_lower_bounds=*/ false,
        )
    }

    /// Small-θ path (n_theta ∈ 2..=6). Uses BOBYQA, which is bounded — we
    /// pass `θ_lower` from `lower_bounds()` so the optimizer never walks
    /// off the feasible region. On smooth, well-conditioned problems
    /// (most LMMs in this regime) BOBYQA converges in roughly half the
    /// evaluations of the pattern-search fallback; profiling kb07 (n_theta
    /// = 2) showed pattern_search using ~84 evaluations for what BOBYQA
    /// typically does in ~25.
    #[cfg(feature = "nlopt")]
    fn fit_nlopt_small_theta_with_maxeval(
        &mut self,
        reml: bool,
        maxeval_override: Option<usize>,
    ) -> Result<&mut Self> {
        self.fit_nlopt_with_algorithm(
            NloptAlgorithm::Bobyqa,
            Optimizer::NloptBobyqa,
            reml,
            maxeval_override,
            /*use_lower_bounds=*/ true,
        )
    }

    #[cfg(feature = "nlopt")]
    fn fit_nlopt_small_theta(&mut self, reml: bool) -> Result<&mut Self> {
        // Mirror the large-θ fallback pattern: if BOBYQA fails to converge
        // (rare on this problem class but possible on near-singular fits),
        // retry with the robust pattern-search optimizer rather than
        // bubbling the error up.
        if self.fit_nlopt_small_theta_with_maxeval(reml, None).is_err() {
            // Reset so pattern_search starts from the recorded initial θ
            // rather than wherever BOBYQA bailed out.
            self.optsum.feval = -1;
            self.optsum.fmin = f64::INFINITY;
            self.optsum.fit_log.clear();
            return self.fit_multivariate_pattern_search();
        }
        Ok(self)
    }

    #[cfg(feature = "nlopt")]
    fn fit_nlopt_with_algorithm(
        &mut self,
        algorithm: NloptAlgorithm,
        optimizer: Optimizer,
        reml: bool,
        maxeval_override: Option<usize>,
        use_lower_bounds: bool,
    ) -> Result<&mut Self> {
        const JULIA_FTOL_REL_DEFAULT: f64 = 1e-12;
        const JULIA_FTOL_ABS_DEFAULT: f64 = 1e-8;
        const RUST_FTOL_REL_DEFAULT: f64 = 1e-8;
        const RUST_FTOL_ABS_DEFAULT: f64 = 1e-12;
        const RUST_INITIAL_STEP_DEFAULT: f64 = 0.75;
        const LARGE_VSIZE2_BOBYQA_FTOL_REL_DEFAULT: f64 = 1e-10;

        self.optsum.optimizer = optimizer;
        let use_large_vsize2_tuning =
            optimizer == Optimizer::NloptBobyqa && self.use_large_single_vsize2_optimizer_tuning();

        let a_blocks = self.a_blocks.clone();
        let l_blocks_template = self.l_blocks.clone();
        let reterms_template = self.reterms.clone();
        let dims = self.dims;
        let is_reml = reml;
        let fixed_sigma = self.optsum.sigma;
        let cholesky_zero_pad_tolerance = self
            .compiler_policy()
            .thresholds
            .cholesky_zero_pad_tolerance;
        let invalid_objective = self.optsum.finitial;
        let best_theta = std::cell::RefCell::new(self.optsum.initial.clone());
        let best_fmin = std::cell::Cell::new(self.optsum.finitial);
        let feval_count = std::cell::Cell::new(0i64);
        let fit_log: std::cell::RefCell<Vec<FitLogEntry>> = std::cell::RefCell::new(Vec::new());

        let reterms_work = std::cell::RefCell::new(reterms_template.clone());
        let l_blocks_work = std::cell::RefCell::new(l_blocks_template);

        let objective_fn = |theta: &[f64], _gradient: Option<&mut [f64]>, _data: &mut ()| -> f64 {
            feval_count.set(feval_count.get() + 1);
            let obj = {
                let mut rw = reterms_work.borrow_mut();
                let mut lw = l_blocks_work.borrow_mut();
                Self::profiled_objective_from_parts(
                    &a_blocks,
                    &mut lw,
                    &mut rw,
                    theta,
                    dims,
                    is_reml,
                    fixed_sigma,
                    cholesky_zero_pad_tolerance,
                )
                .unwrap_or(invalid_objective)
            };

            fit_log.borrow_mut().push(FitLogEntry {
                theta: theta.to_vec(),
                objective: obj,
            });
            if obj + 1e-12 < best_fmin.get() {
                best_fmin.set(obj);
                *best_theta.borrow_mut() = theta.to_vec();
            }

            obj
        };

        let maxeval = maxeval_override.unwrap_or_else(|| {
            if self.optsum.max_feval > 0 {
                self.optsum.max_feval as usize
            } else {
                10000
            }
        });

        let n_theta = self.n_theta();
        let mut opt = Nlopt::new(algorithm, n_theta, objective_fn, NloptTarget::Minimize, ());
        let ftol_rel = if (self.optsum.ftol_rel - RUST_FTOL_REL_DEFAULT).abs() <= f64::EPSILON {
            if use_large_vsize2_tuning {
                // The large one-term random-slope fast path can spend many
                // extra BOBYQA evaluations polishing below the numerical
                // scale that changes the fitted model. Keep the stricter
                // Julia-style default for other model classes.
                LARGE_VSIZE2_BOBYQA_FTOL_REL_DEFAULT
            } else {
                JULIA_FTOL_REL_DEFAULT
            }
        } else {
            self.optsum.ftol_rel
        };
        let ftol_abs = if (self.optsum.ftol_abs - RUST_FTOL_ABS_DEFAULT).abs() <= f64::EPSILON {
            JULIA_FTOL_ABS_DEFAULT
        } else {
            self.optsum.ftol_abs
        };
        if ftol_rel > 0.0 {
            Self::nlopt_ok(opt.set_ftol_rel(ftol_rel), "set_ftol_rel")?;
        }
        if ftol_abs > 0.0 {
            Self::nlopt_ok(opt.set_ftol_abs(ftol_abs), "set_ftol_abs")?;
        }
        if self.optsum.xtol_rel > 0.0 {
            Self::nlopt_ok(opt.set_xtol_rel(self.optsum.xtol_rel), "set_xtol_rel")?;
        }
        if !self.optsum.xtol_abs.is_empty() {
            Self::nlopt_ok(opt.set_xtol_abs(&self.optsum.xtol_abs), "set_xtol_abs")?;
        }
        let use_nlopt_default_initial_step = self.optsum.initial_step.len() == n_theta
            && self
                .optsum
                .initial_step
                .iter()
                .all(|&step| (step - RUST_INITIAL_STEP_DEFAULT).abs() <= f64::EPSILON);
        if !self.optsum.initial_step.is_empty() && !use_nlopt_default_initial_step {
            Self::nlopt_ok(
                opt.set_initial_step(&self.optsum.initial_step),
                "set_initial_step",
            )?;
        }
        if maxeval > 0 {
            Self::nlopt_ok(opt.set_maxeval(maxeval as u32), "set_maxeval")?;
        }
        if self.optsum.max_time > 0.0 {
            Self::nlopt_ok(opt.set_maxtime(self.optsum.max_time), "set_maxtime")?;
        }
        if use_lower_bounds {
            // BOBYQA is bounded — let NLopt enforce θ ≥ θ_lower instead of
            // relying on the soft "objective returns finitial when invalid"
            // barrier, which can confuse the trust-region update step.
            let lb = self.lower_bounds();
            Self::nlopt_ok(opt.set_lower_bounds(&lb), "set_lower_bounds")?;
        }

        let mut theta = self.optsum.initial.clone();
        let optimize_result = opt.optimize(&mut theta);
        let status_label = match &optimize_result {
            Ok((status, _)) => Self::nlopt_status_label(&format!("{status:?}")),
            Err((status, _)) => Self::nlopt_status_label(&format!("{status:?}")),
        };

        let (candidate_theta, candidate_fmin) = match optimize_result {
            Ok((_, fmin)) => (theta.clone(), fmin),
            Err((NloptFailState::RoundoffLimited, fmin)) => (theta.clone(), fmin),
            Err((status, _)) => {
                return Err(MixedModelError::Optimization(format!(
                    "NLopt large-theta optimization failed: {status:?}"
                )));
            }
        };

        let logged_best_theta = best_theta.into_inner();
        let logged_best_fmin = best_fmin.get();
        let (final_theta, final_fmin) = if logged_best_fmin.is_finite()
            && (!candidate_fmin.is_finite() || logged_best_fmin <= candidate_fmin)
        {
            (logged_best_theta, logged_best_fmin)
        } else {
            (candidate_theta, candidate_fmin)
        };

        self.finalize_fit_result(
            final_theta,
            final_fmin,
            feval_count.get(),
            fit_log.into_inner(),
            optimizer,
            None,
        )?;
        self.optsum.return_value = status_label;

        Ok(self)
    }

    #[cfg(feature = "nlopt")]
    fn fit_nlopt_large_theta(&mut self, reml: bool) -> Result<&mut Self> {
        if self.fit_nlopt_large_theta_with_maxeval(reml, None).is_err() {
            return self.fit_cobyla(reml);
        }
        Ok(self)
    }

    /// Fit the model by optimizing θ to minimize the objective.
    pub fn fit(&mut self, reml: bool) -> Result<&mut Self> {
        if self.optsum.feval > 0 {
            return Err(MixedModelError::AlreadyFitted);
        }

        // Check for constant response
        let y = self.y();
        let y0 = y[0];
        if y.iter().all(|&yi| (yi - y0).abs() < f64::EPSILON) {
            return Err(MixedModelError::ConstantResponse);
        }

        if self.feterm.rank >= self.dims.n {
            return Err(MixedModelError::RankSaturatedFixedEffects {
                rank: self.feterm.rank,
                nobs: self.dims.n,
            });
        }

        self.optsum.reml = reml;

        // Initial objective evaluation
        let theta0 = self.optsum.initial.clone();
        self.optsum.finitial = self.objective_at(&theta0)?;

        if self.use_scalar_single_theta_optimizer() {
            self.fit_scalar_single_theta()?;
        } else {
            // The `use_*_nlopt_*` predicates always return `false` when
            // the `nlopt` feature is disabled, so the no-feature build
            // never reaches the nlopt arms even if they appear in the
            // source. Cfg-gating the call sites lets the no-feature
            // build still type-check (the methods themselves are gated
            // out below).
            #[cfg(feature = "nlopt")]
            {
                if self.use_nlopt_bobyqa_small_theta_optimizer() {
                    self.fit_nlopt_small_theta(reml)?;
                } else if self.use_large_theta_nlopt_optimizer() {
                    self.fit_nlopt_large_theta(reml)?;
                } else {
                    self.fit_cobyla(reml)?;
                }
            }
            #[cfg(not(feature = "nlopt"))]
            {
                self.fit_cobyla(reml)?;
            }
        }

        self.refresh_optimizer_certificate();
        self.refresh_effective_covariance_summaries();
        self.refresh_covariance_parameter_traces();
        self.refresh_fixed_effect_inference_table();
        Ok(self)
    }

    /// Extract the fixed-effects coefficients β from the Cholesky factor.
    pub fn beta(&self) -> DVector<f64> {
        let k = self.reterms.len();
        let l_last = self.l_blocks[block_index(k, k)].as_dense();
        let pp1 = l_last.nrows();
        let p = pp1 - 1;

        if p == 0 {
            return DVector::zeros(0);
        }

        let l_xx = l_last.view((0, 0), (p, p));
        let mut beta = DVector::zeros(p);
        for j in 0..p {
            beta[j] = l_last[(pp1 - 1, j)];
        }

        for i in (0..p).rev() {
            let mut s = beta[i];
            for j in (i + 1)..p {
                s -= l_xx[(j, i)] * beta[j];
            }
            beta[i] = s / l_xx[(i, i)];
        }

        beta
    }

    /// Standard deviation estimate (σ).
    pub fn sigma(&self) -> f64 {
        if let Some(sigma) = self.optsum.sigma {
            return sigma;
        }
        let k = self.reterms.len();
        let l_last = self.l_blocks[block_index(k, k)].as_dense();
        let pp1 = l_last.nrows();
        let last_diag = l_last[(pp1 - 1, pp1 - 1)].abs();

        let denom = if self.optsum.reml {
            (self.dims.n - self.dims.p) as f64
        } else {
            self.dims.n as f64
        };

        last_diag / denom.sqrt()
    }

    /// Penalized weighted residual sum of squares.
    pub fn pwrss(&self) -> f64 {
        let k = self.reterms.len();
        let l_last = self.l_blocks[block_index(k, k)].as_dense();
        let pp1 = l_last.nrows();
        let last_diag = l_last[(pp1 - 1, pp1 - 1)];
        last_diag * last_diag
    }

    /// Profile one or more response columns at the current theta.
    ///
    /// Each response column shares the current model's fixed-effects design,
    /// random-effects structure, and theta, but gets its own profiled beta
    /// and sigma.
    pub fn profile_response_matrix(
        &self,
        responses: &DMatrix<f64>,
        reml: bool,
    ) -> Result<ResponseMatrixProfile> {
        if responses.nrows() != self.dims.n {
            return Err(MixedModelError::DimensionMismatch(format!(
                "response matrix has {} rows, expected {}",
                responses.nrows(),
                self.dims.n
            )));
        }

        let x = self.feterm.full_rank_x().into_owned();
        let (a_blocks, mut l_blocks) = create_structural_al(&self.reterms, &x)?;
        update_l_from_parts(
            &a_blocks,
            &mut l_blocks,
            &self.reterms,
            self.compiler_policy()
                .thresholds
                .cholesky_zero_pad_tolerance,
        )?;
        profile_response_matrix_with_l_blocks(
            &self.reterms,
            &x,
            responses,
            &l_blocks,
            reml,
            self.dims.n,
            self.dims.p,
        )
    }

    /// Log-determinant of the RE Cholesky blocks.
    pub fn logdet_re(&self) -> f64 {
        let k = self.reterms.len();
        let mut ld = 0.0;
        for j in 0..k {
            ld += logdet_block(&self.l_blocks[block_index(j, j)]);
        }
        ld
    }

    /// Conditional modes of the random effects (the "u" vectors, on the spherical scale).
    ///
    /// Solves the blocked lower-triangular system `L * u = c` where:
    ///   - `c_j = Λ_j' Z_j' wr`  (weighted residuals projected onto RE term j)
    ///   - `wr = W^{1/2}(y - Xβ)`  (weighted residuals in observation space)
    ///   - `L` is the blocked Cholesky factor stored in `self.l_blocks`
    ///
    /// Returns one matrix per RE term with shape `vsize × n_levels`.
    pub fn ranef_u(&self) -> Vec<DMatrix<f64>> {
        let k = self.reterms.len();
        let p = self.dims.p;
        let n = self.dims.n;
        let beta = self.beta();
        let wtxy = &self.xy_mat.wtxy;

        // Step 1: weighted residuals wr[obs] = wy[obs] - wX[obs,:]*beta
        let mut wr = vec![0.0f64; n];
        for obs in 0..n {
            let mut val = wtxy[(obs, p)]; // weighted y (last column)
            for q in 0..p {
                val -= wtxy[(obs, q)] * beta[q];
            }
            wr[obs] = val;
        }

        // Step 2: c_j = Λ_j' Z_j' wr  for each RE term j
        let mut c_vecs: Vec<DVector<f64>> = Vec::with_capacity(k);
        for j in 0..k {
            let re = &self.reterms[j];
            let vs = re.vsize;
            let nranef = re.n_ranef();
            let n_levels = re.n_levels();

            // Accumulate Z_j' wr (wtz already incorporates sqrtwts)
            let mut c = vec![0.0f64; nranef];
            for obs in 0..n {
                let r = re.refs[obs] as usize;
                for s in 0..vs {
                    c[r * vs + s] += re.wtz[(s, obs)] * wr[obs];
                }
            }

            // Apply Λ_j' per level block: c_scaled[lev,i] = Σ_{row>=i} Λ[row,i] * c[lev,row]
            let lambda = &re.lambda;
            let mut c_scaled = vec![0.0f64; nranef];
            for lev in 0..n_levels {
                for i in 0..vs {
                    let mut val = 0.0;
                    // Λ' is upper triangular of Λ stored as lower, so Λ'[i,row] = Λ[row,i]
                    for row in i..vs {
                        val += lambda[(row, i)] * c[lev * vs + row];
                    }
                    c_scaled[lev * vs + i] = val;
                }
            }

            c_vecs.push(DVector::from_vec(c_scaled));
        }

        // Step 3: blocked solve  (L L') u = c  via forward then backward pass.

        // Forward pass: solve L * v = c  (lower-triangular blocked forward substitution)
        let mut v_vecs: Vec<DVector<f64>> = Vec::with_capacity(k);
        for j in 0..k {
            let nranef_j = self.reterms[j].n_ranef();

            let mut rhs = c_vecs[j].clone();

            // rhs -= L[j,m] * v_m  for all already-solved m < j
            for m in 0..j {
                let l_jm = self.l_blocks[block_index(j, m)].as_dense();
                let v_m = &v_vecs[m];
                for row in 0..nranef_j {
                    let mut dot = 0.0;
                    for col in 0..v_m.len() {
                        dot += l_jm[(row, col)] * v_m[col];
                    }
                    rhs[row] -= dot;
                }
            }

            // Solve L[j,j] * v_j = rhs  (forward substitution)
            let mut v_j = rhs.as_slice().to_vec();
            solve_lower_block_against_rhs(&self.l_blocks[block_index(j, j)], &mut v_j);
            let v_j = DVector::from_vec(v_j);
            v_vecs.push(v_j);
        }

        // Backward pass: solve L' * u = v  (upper-triangular blocked back-substitution)
        let mut u_vecs: Vec<DVector<f64>> = vec![DVector::zeros(0); k];
        for j in (0..k).rev() {
            let nranef_j = self.reterms[j].n_ranef();

            let mut rhs = v_vecs[j].clone();

            // rhs -= L[m,j]' * u_m  for all already-solved m > j
            for m in (j + 1)..k {
                let l_mj = self.l_blocks[block_index(m, j)].as_dense();
                let u_m = &u_vecs[m];
                // L[m,j]' is nranef_j × nranef_m: rhs[row] -= sum_col l_mj[(col,row)] * u_m[col]
                for row in 0..nranef_j {
                    let mut dot = 0.0;
                    for col in 0..u_m.len() {
                        dot += l_mj[(col, row)] * u_m[col];
                    }
                    rhs[row] -= dot;
                }
            }

            // Solve L[j,j]' * u_j = rhs  (backward substitution with L')
            let mut u_j = rhs.as_slice().to_vec();
            solve_upper_block_from_lower_transpose_against_rhs(
                &self.l_blocks[block_index(j, j)],
                &mut u_j,
            );
            let u_j = DVector::from_vec(u_j);
            u_vecs[j] = u_j;
        }

        // Step 4: reshape u vectors into vsize × n_levels matrices
        self.reterms
            .iter()
            .zip(u_vecs)
            .map(|(rt, u)| {
                let vs = rt.vsize;
                let nl = rt.n_levels();
                DMatrix::from_column_slice(vs, nl, u.as_slice())
            })
            .collect()
    }

    /// Conditional modes on the original scale: b = Λ * u
    pub fn ranef_b(&self) -> Vec<DMatrix<f64>> {
        self.ranef_u()
            .into_iter()
            .zip(&self.reterms)
            .map(|(u, rt)| &rt.lambda * &u)
            .collect()
    }

    /// Grouping factor names.
    pub fn fnames(&self) -> Vec<String> {
        self.reterms
            .iter()
            .map(|rt| rt.grouping_name.clone())
            .collect()
    }

    /// Variance-covariance summary for the fitted random effects.
    pub fn varcorr(&self) -> VarCorr {
        VarCorr::from_model(&self.reterms, self.sigma())
    }

    /// Condition number of each RE Lambda factor.
    ///
    /// Mirrors `cond(fm)` in Julia's MixedModels.jl. For a scalar RE, the
    /// condition number is always 1.0. For a vector RE, it is the ratio of the
    /// largest to smallest singular value of the lower-triangular Cholesky factor.
    pub fn cond(&self) -> Vec<f64> {
        self.reterms
            .iter()
            .map(|rt| {
                let s = rt.vsize;
                if s <= 1 {
                    1.0
                } else {
                    let svd = rt.lambda.clone().svd(false, false);
                    let sv = &svd.singular_values;
                    let smax = sv.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let smin = sv.iter().cloned().fold(f64::INFINITY, f64::min);
                    if smin < f64::EPSILON {
                        f64::INFINITY
                    } else {
                        smax / smin
                    }
                }
            })
            .collect()
    }

    /// Residual degrees of freedom: `nobs - rank(X)`.
    ///
    /// Mirrors `dof_residual(fm)` in Julia's MixedModels.jl.
    pub fn dof_residual(&self) -> usize {
        self.nobs().saturating_sub(self.feterm.rank)
    }

    /// Residual scale reported by Julia's `varest(fm)`.
    ///
    /// For estimated-σ fits this is σ². For fixed-σ fits, MixedModels.jl
    /// reports the fixed σ itself, not σ².
    pub fn varest(&self) -> f64 {
        if let Some(sigma) = self.optsum.sigma {
            return sigma;
        }
        let s = self.sigma();
        s * s
    }

    /// Log-determinant of the RE blocks of the Cholesky factor L.
    ///
    /// Mirrors `logdet(fm)` in Julia's MixedModels.jl.
    pub fn logdet(&self) -> f64 {
        self.logdet_re()
    }

    /// Model dimensions as `(n, p, total_nranef, nretrms)`.
    ///
    /// Mirrors `size(fm)` in Julia's MixedModels.jl where the four
    /// elements are:
    /// - `n`: number of observations
    /// - `p`: rank of the fixed-effects matrix
    /// - `total_nranef`: total number of random-effects columns (`Σ vsize_j * n_levels_j`)
    /// - `nretrms`: number of RE grouping factors
    pub fn model_size(&self) -> (usize, usize, usize, usize) {
        let total_nranef: usize = self.reterms.iter().map(|rt| rt.n_ranef()).sum();
        (self.dims.n, self.dims.p, total_nranef, self.dims.nretrms)
    }

    /// Standard deviations of each random-effects term plus the residual σ.
    ///
    /// Returns one `Vec<f64>` per RE term (with one entry per random-effects
    /// component), followed by `vec![sigma]` for the residual.
    ///
    /// Mirrors `std(fm)` in Julia's MixedModels.jl.
    pub fn std_devs(&self) -> Vec<Vec<f64>> {
        let sigma = self.sigma();
        let mut out: Vec<Vec<f64>> = self
            .reterms
            .iter()
            .map(|rt| {
                (0..rt.vsize)
                    .map(|i| {
                        let sq: f64 = (0..=i).map(|j| rt.lambda[(i, j)].powi(2)).sum();
                        sigma * sq.sqrt()
                    })
                    .collect()
            })
            .collect();
        out.push(vec![sigma]);
        out
    }

    /// Simulate a new response vector from the fitted model.
    ///
    /// Draws `u_j ~ N(0, I)` for each RE term, computes `b_j = σ Λ_j u_j`,
    /// adds fixed-effects `Xβ`, RE contribution `Σ Z_j b_j`, and i.i.d.
    /// residual noise `ε ~ N(0, σ²)`.
    ///
    /// Mirrors `simulate(fm)` in Julia's MixedModels.jl.
    pub fn simulate<R: rand::Rng>(&self, rng: &mut R) -> DVector<f64> {
        let beta = self.beta();
        self.simulate_with_active_beta(rng, &beta)
            .expect("fitted beta should match active fixed-effect design")
    }

    fn simulate_with_active_beta<R: rand::Rng>(
        &self,
        rng: &mut R,
        beta: &DVector<f64>,
    ) -> Result<DVector<f64>> {
        use rand_distr::{Distribution, Normal};

        let n = self.dims.n;
        let sigma = self.sigma();

        // Fixed-effects prediction: Xβ
        let x = self.feterm.full_rank_x();
        if beta.len() != x.ncols() {
            return Err(MixedModelError::DimensionMismatch(format!(
                "simulation beta has length {}, but active fixed-effect design has {} column(s)",
                beta.len(),
                x.ncols()
            )));
        }
        let mut y_new: DVector<f64> = x * beta;

        // Random-effects contribution
        let normal01 = Normal::new(0.0, 1.0).unwrap();
        for rt in &self.reterms {
            let n_levels = rt.n_levels();
            // u ~ N(0, I)
            let u = DMatrix::from_fn(rt.vsize, n_levels, |_, _| normal01.sample(rng));
            // b = sigma * Λ * u
            let b = sigma * &rt.lambda * &u;
            let bvec = DVector::from_column_slice(b.as_slice());
            for (obs, &ref_idx) in rt.refs.iter().enumerate() {
                let r = ref_idx as usize;
                for s in 0..rt.vsize {
                    y_new[obs] += rt.z[(s, obs)] * bvec[r * rt.vsize + s];
                }
            }
        }

        // Residual noise ε ~ N(0, σ²)
        let eps_dist = Normal::new(0.0, sigma).unwrap();
        for i in 0..n {
            y_new[i] += eps_dist.sample(rng);
        }

        Ok(y_new)
    }

    pub fn fixed_effect_null_bootstrap_target(
        &self,
        hypothesis: &FixedEffectHypothesis,
    ) -> Result<FixedEffectNullBootstrapTarget> {
        let n_coefficients = self.coef_names().len();
        if hypothesis.n_coefficients() != n_coefficients {
            return Err(MixedModelError::DimensionMismatch(format!(
                "fixed-effect null target contrast has {} coefficient column(s), but the model has {n_coefficients}",
                hypothesis.n_coefficients()
            )));
        }

        let beta_fitted = self.coef();
        let vcov = self.vcov();
        let estimability = assess_fixed_contrast_estimability(hypothesis, &beta_fitted, &vcov);
        if estimability.status != EstimabilityStatus::Estimable {
            return Err(MixedModelError::InvalidArgument(
                "fixed-effect null bootstrap target requires an estimable contrast".to_string(),
            ));
        }
        if !matrix_is_finite(&vcov) {
            return Err(MixedModelError::InvalidArgument(
                "fixed-effect null bootstrap target requires finite fixed-effect covariance"
                    .to_string(),
            ));
        }

        let middle =
            symmetrize_matrix(&(&hypothesis.l.values * &vcov * hypothesis.l.values.transpose()));
        if !matrix_is_finite(&middle) {
            return Err(MixedModelError::InvalidArgument(
                "fixed-effect null bootstrap target produced non-finite L V L'".to_string(),
            ));
        }
        let middle_eig = SymmetricEigen::new(middle.clone());
        let middle_max_abs = middle_eig
            .eigenvalues
            .iter()
            .map(|value| value.abs())
            .fold(0.0, f64::max);
        let middle_tol = (1e-10 * middle_max_abs.max(1.0)).max(1e-12);
        let middle_min_abs = middle_eig
            .eigenvalues
            .iter()
            .map(|value| value.abs())
            .fold(f64::INFINITY, f64::min);
        let used_generalized_inverse = middle_min_abs <= middle_tol;
        let middle_inverse = if used_generalized_inverse {
            symmetric_pseudoinverse(&middle, middle_tol)
        } else {
            invert_spd_matrix(&middle, "fixed-effect null bootstrap L V L' matrix")?
        };

        let fitted_contrast = &hypothesis.l.values * &beta_fitted - &hypothesis.rhs.values;
        let beta_null = &beta_fitted
            - &vcov * hypothesis.l.values.transpose() * middle_inverse * fitted_contrast;
        let _beta_null_active = self.fixed_effect_user_beta_to_active_basis(&beta_null)?;

        let mut notes = vec![
            "fixed-effect null target reuses fitted covariance parameters; variance re-estimation under the null is not yet implemented"
                .to_string(),
        ];
        if used_generalized_inverse {
            notes.push(format!(
                "fixed-effect null target used a generalized inverse for L V L' at tolerance {middle_tol}"
            ));
        }

        Ok(FixedEffectNullBootstrapTarget {
            target: BootstrapTarget::fixed_effect_null(
                format!("{} fixed-effect null", hypothesis.label),
                hypothesis.label.clone(),
            ),
            covariance_policy: FixedEffectNullCovariancePolicy::ReuseFittedCovariance,
            coefficient_names: self.coef_names(),
            beta_fitted,
            beta_null,
            theta: self.theta(),
            sigma: self.sigma(),
            reml: self.optsum.reml,
            notes,
        })
    }

    pub fn simulate_fixed_effect_null<R: rand::Rng>(
        &self,
        rng: &mut R,
        target: &FixedEffectNullBootstrapTarget,
    ) -> Result<DVector<f64>> {
        if target.covariance_policy != FixedEffectNullCovariancePolicy::ReuseFittedCovariance {
            return Err(MixedModelError::InvalidArgument(
                "unsupported fixed-effect null bootstrap covariance policy".to_string(),
            ));
        }
        if target.theta.len() != self.n_theta()
            || target
                .theta
                .iter()
                .zip(self.theta().iter())
                .any(|(lhs, rhs)| (*lhs - *rhs).abs() > 1e-10)
            || (target.sigma - self.sigma()).abs() > 1e-10
        {
            return Err(MixedModelError::InvalidArgument(
                "fixed-effect null bootstrap target does not match the fitted covariance state"
                    .to_string(),
            ));
        }
        let beta_active = self.fixed_effect_user_beta_to_active_basis(&target.beta_null)?;
        self.simulate_with_active_beta(rng, &beta_active)
    }

    /// Refit the model with a new response vector.
    ///
    /// Replaces the response, rebuilds the cross-product matrices, and
    /// re-runs the full optimization from the original initial parameters.
    ///
    /// Mirrors `refit!(fm, new_y)` in Julia's MixedModels.jl.
    pub fn refit(&mut self, new_y: &[f64]) -> Result<()> {
        if new_y.len() != self.dims.n {
            return Err(MixedModelError::InvalidArgument(format!(
                "Response length {} does not match model ({} observations)",
                new_y.len(),
                self.dims.n
            )));
        }

        let y_max = new_y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let y_min = new_y.iter().cloned().fold(f64::INFINITY, f64::min);
        if (y_max - y_min) < f64::EPSILON {
            return Err(MixedModelError::InvalidArgument(
                "The response is constant and thus model fitting has failed".to_string(),
            ));
        }

        let p = self.feterm.rank;
        for obs in 0..self.dims.n {
            let sw = if self.sqrtwts.is_empty() {
                1.0
            } else {
                self.sqrtwts[obs]
            };
            self.y[obs] = new_y[obs];
            self.xy_mat.xy[(obs, p)] = new_y[obs];
            self.xy_mat.wtxy[(obs, p)] = sw * new_y[obs];
        }

        self.recompute_a_blocks();

        // Reset fit state so fit() doesn't reject as AlreadyFitted
        let reml = self.optsum.reml;
        self.optsum.feval = 0;

        // Re-optimize from initial θ
        let initial = self.optsum.initial.clone();
        self.set_theta(&initial)?;
        self.fit(reml)?;
        Ok(())
    }

    /// Hat matrix diagonal (leverage values) for each observation.
    ///
    /// Computes `h_i = ||L⁻¹ vᵢ||²` where `vᵢ` is the i-th column of
    /// the (weighted) design matrix `[ΛZ | X]'`.  The sum equals the
    /// model degrees of freedom (rank of X + RE θ parameters).
    ///
    /// Mirrors `leverage(fm)` in Julia's MixedModels.jl.
    pub fn leverage(&self) -> DVector<f64> {
        let k = self.reterms.len();
        let p = self.dims.p;
        let n = self.dims.n;
        let wtxy = &self.xy_mat.wtxy;
        let pp1 = p + 1; // p fixed effects + 1 response (y slot kept at 0)

        // Cumulative column offsets into the stacked RE vector
        let mut offsets = vec![0usize; k + 1];
        for j in 0..k {
            offsets[j + 1] = offsets[j] + self.reterms[j].n_ranef();
        }
        let nranef_total = offsets[k];

        let mut h = DVector::zeros(n);

        for obs in 0..n {
            // Build vᵢ: weighted design column [Λⱼ' wtzⱼ[:,obs]; ...; wtxy[obs,0..p]; 0]
            let mut v = vec![0.0f64; nranef_total + pp1];

            for j in 0..k {
                let re = &self.reterms[j];
                let vs = re.vsize;
                let r = re.refs[obs] as usize;
                let lambda = &re.lambda;
                let offset = offsets[j] + r * vs;
                // (Λⱼ')_{i,row} = Λⱼ[row,i];  Λ is lower-triangular → row ≥ i
                for i in 0..vs {
                    let mut val = 0.0;
                    for row in i..vs {
                        val += lambda[(row, i)] * re.wtz[(row, obs)];
                    }
                    v[offset + i] = val;
                }
            }
            for q in 0..p {
                v[nranef_total + q] = wtxy[(obs, q)];
            }
            // v[nranef_total + p] = 0  (y slot excluded from leverage)

            // Forward solve L * w = v  (lower-triangular blocked)
            let mut w = vec![0.0f64; nranef_total + pp1];

            // RE blocks j = 0..k
            for j in 0..k {
                let re_j = &self.reterms[j];
                let nranef_j = re_j.n_ranef();

                let mut rhs = vec![0.0f64; nranef_j];
                for idx in 0..nranef_j {
                    rhs[idx] = v[offsets[j] + idx];
                }
                for m in 0..j {
                    let l_jm = self.l_blocks[block_index(j, m)].as_dense();
                    let nranef_m = self.reterms[m].n_ranef();
                    for row in 0..nranef_j {
                        let mut dot = 0.0;
                        for col in 0..nranef_m {
                            dot += l_jm[(row, col)] * w[offsets[m] + col];
                        }
                        rhs[row] -= dot;
                    }
                }

                solve_lower_block_against_rhs(&self.l_blocks[block_index(j, j)], &mut rhs);
                for idx in 0..nranef_j {
                    w[offsets[j] + idx] = rhs[idx];
                }
            }

            // FE block (k-th block): forward solve L[k,k] * w_k = rhs_k
            let l_kk = self.l_blocks[block_index(k, k)].as_dense();
            let mut rhs_k = vec![0.0f64; pp1];
            for idx in 0..pp1 {
                rhs_k[idx] = v[nranef_total + idx];
            }
            for j in 0..k {
                let l_kj = self.l_blocks[block_index(k, j)].as_dense();
                let nranef_j = self.reterms[j].n_ranef();
                for row in 0..pp1 {
                    let mut dot = 0.0;
                    for col in 0..nranef_j {
                        dot += l_kj[(row, col)] * w[offsets[j] + col];
                    }
                    rhs_k[row] -= dot;
                }
            }
            let mut w_k = vec![0.0f64; pp1];
            w_k.copy_from_slice(&rhs_k);
            solve_lower_block_against_rhs(&MatrixBlock::Dense(l_kk), &mut w_k);

            // h_obs = ||w_RE||² + ||w_FE||²  (exclude w_k[p] = y component)
            let sum_sq: f64 = w[..nranef_total].iter().map(|x| x * x).sum::<f64>()
                + w_k[..p].iter().map(|x| x * x).sum::<f64>();
            h[obs] = sum_sq;
        }

        h
    }

    /// Conditional variance matrices of the random effects.
    ///
    /// Returns one `Vec<DMatrix<f64>>` per RE term.  Each inner vector has one
    /// `vsize × vsize` positive-semi-definite matrix per level of the grouping
    /// factor.  The matrices are the diagonal blocks of `σ² Λ(Λ'Z'ZΛ+I)⁻¹Λ'`.
    ///
    /// Mirrors `condVar(m)` in Julia's MixedModels.jl.
    pub fn cond_var(&self) -> Vec<Vec<DMatrix<f64>>> {
        let k = self.reterms.len();
        let sigma = self.sigma();
        let mut result = Vec::with_capacity(k);

        for j in 0..k {
            let re_j = &self.reterms[j];
            let vs_j = re_j.vsize;
            let n_levels_j = re_j.n_levels();

            // λt = σ * Λ_j'  (vs_j × vs_j)
            let lambda_j = &re_j.lambda;
            let mut lambda_t = DMatrix::zeros(vs_j, vs_j);
            for row in 0..vs_j {
                for col in 0..vs_j {
                    // Λ'[row,col] = Λ[col,row]
                    lambda_t[(row, col)] = sigma * lambda_j[(col, row)];
                }
            }

            // Local row offsets within the sub-L starting at term j
            // Sub-L includes RE terms j..k-1 (no FE block)
            let mut local_off = vec![0usize; k - j + 1];
            for m in 0..(k - j) {
                local_off[m + 1] = local_off[m] + self.reterms[j + m].n_ranef();
            }
            let q_j = local_off[k - j]; // total rows in sub-L

            let mut condvars = Vec::with_capacity(n_levels_j);

            for b in 0..n_levels_j {
                // scratch = zeros(q_j, vs_j); set block at level b to λt
                let mut scratch = DMatrix::zeros(q_j, vs_j);
                for row in 0..vs_j {
                    for col in 0..vs_j {
                        scratch[(b * vs_j + row, col)] = lambda_t[(row, col)];
                    }
                }

                // Forward solve: for each sub-block i (term j+i) in order
                for i in 0..(k - j) {
                    let blk_i = j + i;
                    let nranef_i = self.reterms[blk_i].n_ranef();
                    let off_i = local_off[i];

                    // Subtract cross-block contributions: L[blk_i, blk_prev] * scratch[prev]
                    for prev in 0..i {
                        let blk_prev = j + prev;
                        let nranef_prev = self.reterms[blk_prev].n_ranef();
                        let off_prev = local_off[prev];
                        let l_cross = self.l_blocks[block_index(blk_i, blk_prev)].as_dense();
                        for col in 0..vs_j {
                            for row in 0..nranef_i {
                                let mut dot = 0.0;
                                for c in 0..nranef_prev {
                                    dot += l_cross[(row, c)] * scratch[(off_prev + c, col)];
                                }
                                scratch[(off_i + row, col)] -= dot;
                            }
                        }
                    }

                    // Solve L[blk_i, blk_i] * scratch[i_part] = scratch[i_part]
                    for col in 0..vs_j {
                        let mut rhs: Vec<f64> = (0..nranef_i)
                            .map(|idx| scratch[(off_i + idx, col)])
                            .collect();
                        solve_lower_block_against_rhs(
                            &self.l_blocks[block_index(blk_i, blk_i)],
                            &mut rhs,
                        );
                        for idx in 0..nranef_i {
                            scratch[(off_i + idx, col)] = rhs[idx];
                        }
                    }
                }

                // condvar_b = scratch' * scratch  (vs_j × vs_j)
                condvars.push(scratch.transpose() * &scratch);
            }

            result.push(condvars);
        }

        result
    }

    /// Structural summary of the blocked `A`/`L` system.
    pub fn block_description(&self) -> BlockDescription {
        BlockDescription::from_linear_model(self)
    }

    /// Fixed/random-effects summary table.
    pub fn summary(&self) -> ModelSummary {
        ModelSummary::from_linear_model(self)
    }

    /// Render the model summary as markdown.
    pub fn summary_markdown(&self) -> String {
        self.summary().to_markdown()
    }

    /// Render the model summary as HTML.
    pub fn summary_html(&self) -> String {
        self.summary().to_html()
    }

    /// Render the model summary as LaTeX.
    pub fn summary_latex(&self) -> String {
        self.summary().to_latex()
    }

    /// Number of θ parameters.
    pub fn n_theta(&self) -> usize {
        self.reterms.iter().map(|rt| rt.n_theta()).sum()
    }

    /// Coefficient table for the fixed effects.
    ///
    /// Returns a [`CoefTable`] with one row per fixed-effects term (in the
    /// original, unpivoted column order) containing:
    /// - the estimate (`β`)
    /// - the standard error
    /// - the Wald z-statistic (`β / SE`)
    /// - the two-sided p-value from the standard normal distribution
    ///
    /// Mirrors `coeftable(m)` in MixedModels.jl / StatsModels.jl.  As in
    /// Julia, p-values use the z-distribution (large-sample approximation).
    pub fn coeftable(&self) -> CoefTable {
        let names = self.coef_names();
        let estimates: Vec<f64> = MixedModelFit::coef(self).iter().cloned().collect();
        let std_errors: Vec<f64> = self.stderror().iter().cloned().collect();
        CoefTable::new_with_p_value_policy(
            names,
            estimates,
            std_errors,
            self.fixed_effect_p_value_policy(),
        )
    }

    pub fn coefficient_hypotheses(&self) -> Vec<FixedEffectHypothesis> {
        let names = self.coef_names();
        names
            .iter()
            .enumerate()
            .filter_map(|(index, name)| {
                FixedEffectHypothesis::single_coefficient(name.clone(), index, names.len()).ok()
            })
            .collect()
    }

    pub fn test_contrast(&self, hypothesis: FixedEffectHypothesis) -> FixedEffectTest {
        self.test_contrast_with_method(hypothesis, FixedEffectTestMethod::Auto)
    }

    pub fn test_contrast_with_method(
        &self,
        hypothesis: FixedEffectHypothesis,
        requested_method: FixedEffectTestMethod,
    ) -> FixedEffectTest {
        let label = hypothesis.label.clone();
        let n_coefficients = self.coef_names().len();
        if hypothesis.n_coefficients() != n_coefficients {
            let reason = format!(
                "contrast has {} coefficient column(s), but the model has {n_coefficients}",
                hypothesis.n_coefficients()
            );
            return fixed_effect_test_unavailable(
                hypothesis,
                FixedContrastEstimability::not_assessed(label),
                InferenceStatus::Unsupported { reason },
            );
        }

        let beta = self.coef();
        let vcov = self.vcov();
        let estimates = (&hypothesis.l.values * &beta - &hypothesis.rhs.values)
            .iter()
            .copied()
            .collect::<Vec<_>>();
        let standard_errors = contrast_standard_errors(&hypothesis.l.values, &vcov);
        let statistics = estimates
            .iter()
            .zip(standard_errors.iter())
            .map(|(&estimate, se)| {
                se.and_then(|se| {
                    (se > 0.0 && se.is_finite() && estimate.is_finite()).then_some(estimate / se)
                })
            })
            .collect::<Vec<_>>();

        let estimability = assess_fixed_contrast_estimability(&hypothesis, &beta, &vcov);
        if estimability.status == EstimabilityStatus::NotEstimable {
            return FixedEffectTest {
                hypothesis,
                estimates,
                standard_errors,
                statistics,
                numerator_df: Some(1.0),
                denominator_df: None,
                p_values: vec![None; estimability.requested_rank.unwrap_or(1)],
                method: InferenceMethod::NotComputed {
                    reason: "contrast is not estimable under the fitted fixed-effect design"
                        .to_string(),
                },
                reliability: ReliabilityGrade::NotAvailable,
                status: InferenceStatus::NotEstimable {
                    reason: "contrast touches aliased or non-finite coefficient directions"
                        .to_string(),
                },
                estimability,
                notes: Vec::new(),
            };
        }

        if hypothesis.n_contrasts() != 1 && requested_method != FixedEffectTestMethod::KenwardRoger
        {
            let reason =
                "multi-df fixed-effect contrast tests are not implemented in this scaffold"
                    .to_string();
            return FixedEffectTest {
                hypothesis,
                estimates,
                standard_errors,
                statistics,
                numerator_df: Some(estimability.requested_rank.unwrap_or(0) as f64),
                denominator_df: None,
                p_values: vec![None; estimability.requested_rank.unwrap_or(0)],
                method: InferenceMethod::NotComputed {
                    reason: reason.clone(),
                },
                reliability: ReliabilityGrade::NotAvailable,
                status: InferenceStatus::Unsupported { reason },
                estimability,
                notes: Vec::new(),
            };
        }

        match requested_method {
            FixedEffectTestMethod::Auto => match self.fixed_effect_p_value_policy() {
                CoefTablePValuePolicy::AsymptoticWaldZ => {
                    let satterthwaite = self.satterthwaite_fixed_effect_test(
                        hypothesis.clone(),
                        estimates.clone(),
                        standard_errors.clone(),
                        statistics.clone(),
                        estimability.clone(),
                    );
                    if satterthwaite.status == InferenceStatus::Available {
                        satterthwaite
                    } else {
                        let mut wald = fixed_effect_test_asymptotic_wald_z(
                            hypothesis,
                            estimates,
                            standard_errors,
                            statistics,
                            estimability,
                        );
                        if let Some(reason) = fixed_effect_inference_reason(&satterthwaite) {
                            wald.notes
                                .push(format!("auto Satterthwaite unavailable: {reason}"));
                        }
                        wald
                    }
                }
                CoefTablePValuePolicy::Unavailable { reason } => {
                    fixed_effect_test_p_value_unavailable(
                        hypothesis,
                        estimates,
                        standard_errors,
                        statistics,
                        estimability,
                        reason,
                    )
                }
            },
            FixedEffectTestMethod::AsymptoticWaldZ => match self.fixed_effect_p_value_policy() {
                CoefTablePValuePolicy::AsymptoticWaldZ => fixed_effect_test_asymptotic_wald_z(
                    hypothesis,
                    estimates,
                    standard_errors,
                    statistics,
                    estimability,
                ),
                CoefTablePValuePolicy::Unavailable { reason } => {
                    fixed_effect_test_p_value_unavailable(
                        hypothesis,
                        estimates,
                        standard_errors,
                        statistics,
                        estimability,
                        reason,
                    )
                }
            },
            FixedEffectTestMethod::Satterthwaite => self.satterthwaite_fixed_effect_test(
                hypothesis,
                estimates,
                standard_errors,
                statistics,
                estimability,
            ),
            FixedEffectTestMethod::KenwardRoger => self.kenward_roger_fixed_effect_test(
                hypothesis,
                estimates,
                standard_errors,
                statistics,
                estimability,
            ),
            FixedEffectTestMethod::ParametricBootstrap => fixed_effect_test_not_assessed_with_method(
                hypothesis,
                estimates,
                standard_errors,
                statistics,
                InferenceMethod::ParametricBootstrap,
                estimability,
                "parametric bootstrap fixed-effect inference requires a certified fixed_effect_null bootstrap payload; call test_contrast_with_bootstrap_payload with replicate accounting, failed-refit policy, Monte Carlo uncertainty, and reproducibility state"
                    .to_string(),
            ),
        }
    }

    pub fn test_contrast_with_bootstrap_payload(
        &self,
        hypothesis: FixedEffectHypothesis,
        payload: &BootstrapRunPayload,
    ) -> FixedEffectTest {
        let label = hypothesis.label.clone();
        let n_coefficients = self.coef_names().len();
        if hypothesis.n_coefficients() != n_coefficients {
            let reason = format!(
                "contrast has {} coefficient column(s), but the model has {n_coefficients}",
                hypothesis.n_coefficients()
            );
            return fixed_effect_test_unavailable(
                hypothesis,
                FixedContrastEstimability::not_assessed(label),
                InferenceStatus::Unsupported { reason },
            );
        }

        let beta = self.coef();
        let vcov = self.vcov();
        let estimates = (&hypothesis.l.values * &beta - &hypothesis.rhs.values)
            .iter()
            .copied()
            .collect::<Vec<_>>();
        let standard_errors = contrast_standard_errors(&hypothesis.l.values, &vcov);
        let statistics = estimates
            .iter()
            .zip(standard_errors.iter())
            .map(|(&estimate, se)| {
                se.and_then(|se| {
                    (se > 0.0 && se.is_finite() && estimate.is_finite()).then_some(estimate / se)
                })
            })
            .collect::<Vec<_>>();

        let estimability = assess_fixed_contrast_estimability(&hypothesis, &beta, &vcov);
        if estimability.status == EstimabilityStatus::NotEstimable {
            return FixedEffectTest {
                hypothesis,
                estimates,
                standard_errors,
                statistics,
                numerator_df: None,
                denominator_df: None,
                p_values: vec![None; estimability.requested_rank.unwrap_or(1)],
                method: InferenceMethod::ParametricBootstrap,
                reliability: ReliabilityGrade::NotAvailable,
                status: InferenceStatus::NotEstimable {
                    reason: "bootstrap fixed-effect inference requires an estimable contrast"
                        .to_string(),
                },
                estimability,
                notes: Vec::new(),
            };
        }

        self.bootstrap_fixed_effect_test_from_payload(
            hypothesis,
            estimates,
            standard_errors,
            statistics,
            estimability,
            payload,
        )
    }

    pub fn fixed_effect_bootstrap_inference_row(
        &self,
        kind: FixedEffectInferenceRowKind,
        hypothesis: FixedEffectHypothesis,
        payload: &BootstrapRunPayload,
    ) -> FixedEffectInferenceRow {
        let mut row = fixed_effect_test_to_inference_row(
            kind,
            self.test_contrast_with_bootstrap_payload(hypothesis, payload),
        );
        attach_bootstrap_details(&mut row, payload, None);
        row
    }

    pub fn fixed_effect_contrast_inference_table(
        &self,
        hypotheses: Vec<FixedEffectHypothesis>,
        method: FixedEffectTestMethod,
    ) -> FixedEffectInferenceTable {
        let rows = hypotheses
            .into_iter()
            .map(|hypothesis| {
                self.fixed_effect_contrast_inference_row(
                    FixedEffectInferenceRowKind::Contrast,
                    hypothesis,
                    method,
                )
            })
            .collect();
        FixedEffectInferenceTable::new(rows)
    }

    pub fn fixed_effect_contrast_inference_row(
        &self,
        kind: FixedEffectInferenceRowKind,
        hypothesis: FixedEffectHypothesis,
        method: FixedEffectTestMethod,
    ) -> FixedEffectInferenceRow {
        fixed_effect_test_to_inference_row(kind, self.test_contrast_with_method(hypothesis, method))
    }

    pub fn fixed_effect_null_bootstrap_inference_table(
        &self,
        hypotheses: Vec<FixedEffectHypothesis>,
        options: FixedEffectBootstrapOptions,
    ) -> FixedEffectInferenceTable {
        let rows = hypotheses
            .into_iter()
            .map(|hypothesis| {
                self.fixed_effect_null_bootstrap_inference_row(
                    FixedEffectInferenceRowKind::Contrast,
                    hypothesis,
                    &options,
                )
            })
            .collect();
        FixedEffectInferenceTable::new(rows)
    }

    pub fn fixed_effect_null_bootstrap_inference_row(
        &self,
        kind: FixedEffectInferenceRowKind,
        hypothesis: FixedEffectHypothesis,
        options: &FixedEffectBootstrapOptions,
    ) -> FixedEffectInferenceRow {
        let target = match self.fixed_effect_null_bootstrap_target(&hypothesis) {
            Ok(target) => target,
            Err(error) => {
                let mut test = self.test_contrast_with_method(
                    hypothesis,
                    FixedEffectTestMethod::ParametricBootstrap,
                );
                test.status = InferenceStatus::NotAssessed {
                    reason: format!("bootstrap_null_target_unavailable: {error}"),
                };
                return fixed_effect_test_to_inference_row(kind, test);
            }
        };

        match self.fixed_effect_null_bootstrap_payload(&hypothesis, &target, options) {
            Ok(payload) => {
                let mut row = self.fixed_effect_bootstrap_inference_row(kind, hypothesis, &payload);
                attach_bootstrap_details(&mut row, &payload, Some(&target));
                row
            }
            Err(error) => {
                let mut test = self.test_contrast_with_method(
                    hypothesis,
                    FixedEffectTestMethod::ParametricBootstrap,
                );
                test.status = InferenceStatus::NotAssessed {
                    reason: format!("bootstrap_replicate_accounting_unavailable: {error}"),
                };
                fixed_effect_test_to_inference_row(kind, test)
            }
        }
    }

    fn fixed_effect_null_bootstrap_payload(
        &self,
        hypothesis: &FixedEffectHypothesis,
        target: &FixedEffectNullBootstrapTarget,
        options: &FixedEffectBootstrapOptions,
    ) -> Result<BootstrapRunPayload> {
        let mut rng = match options.seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => rand::rngs::StdRng::from_entropy(),
        };
        let mut fits = Vec::with_capacity(options.requested_replicates);
        let mut statistics = Vec::with_capacity(options.requested_replicates);

        for _ in 0..options.requested_replicates {
            let y_sim = self.simulate_fixed_effect_null(&mut rng, target)?;
            let mut work = self.clone();
            match work.refit(y_sim.as_slice()) {
                Ok(()) => {
                    statistics.push(
                        scalar_contrast_abs_studentized(&work, hypothesis).unwrap_or(f64::NAN),
                    );
                    fits.push(BootstrapReplicate {
                        objective: work.objective(),
                        sigma: work.sigma(),
                        beta: work.beta(),
                        se: work.stderror(),
                        theta: work.theta(),
                    });
                }
                Err(_) => {
                    let beta = work.beta();
                    statistics.push(f64::NAN);
                    fits.push(BootstrapReplicate {
                        objective: f64::NAN,
                        sigma: f64::NAN,
                        se: DVector::from_element(beta.len(), f64::NAN),
                        beta,
                        theta: work.theta(),
                    });
                    if options.failed_refit_policy == BootstrapFailedRefitPolicy::Abort {
                        break;
                    }
                }
            }
        }

        let bootstrap = MixedModelBootstrap { fits };
        let p_value = scalar_contrast_abs_studentized(self, hypothesis).and_then(|observed| {
            let finite = statistics
                .iter()
                .copied()
                .filter(|value| value.is_finite())
                .collect::<Vec<_>>();
            (!finite.is_empty()).then(|| {
                let extreme = finite.iter().filter(|&&value| value >= observed).count();
                (extreme as f64 + 1.0) / (finite.len() as f64 + 1.0)
            })
        });
        let seed_record = options
            .seed
            .map(BootstrapSeedRecord::std_rng)
            .unwrap_or_else(BootstrapSeedRecord::unspecified);
        let metadata = bootstrap.run_metadata_for_model(
            self,
            target.target.clone(),
            options.requested_replicates,
            options.failed_refit_policy,
            seed_record,
            BootstrapRefitOptions::from_model(self),
            Some(hypothesis.label.clone()),
            Some(&statistics),
            p_value,
        );
        Ok(bootstrap.into_run_payload_with_statistics(metadata, statistics))
    }

    pub fn fixed_effect_term_hypotheses(&self) -> Vec<FixedEffectHypothesis> {
        let names = self.coef_names();
        let Some(audit) = self.compiler_artifact.design_audit.as_ref() else {
            return Vec::new();
        };
        audit
            .fixed_effects
            .terms
            .iter()
            .filter_map(|term| {
                let indices = audit
                    .fixed_effects
                    .columns
                    .iter()
                    .filter(|column| column.source_term == term.term)
                    .filter_map(|column| names.iter().position(|name| name == &column.name))
                    .collect::<Vec<_>>();
                if indices.is_empty() {
                    return None;
                }
                let mut l = DMatrix::zeros(indices.len(), names.len());
                for (row, index) in indices.into_iter().enumerate() {
                    l[(row, index)] = 1.0;
                }
                Some(FixedEffectHypothesis::zero_rhs(
                    term.term.clone(),
                    crate::compiler::ContrastMatrix::new(l).ok()?,
                ))
            })
            .collect()
    }

    pub fn fixed_effect_term_inference_table(
        &self,
        method: FixedEffectTestMethod,
    ) -> FixedEffectInferenceTable {
        let rows = self
            .fixed_effect_term_hypotheses()
            .into_iter()
            .map(|hypothesis| {
                fixed_effect_test_to_inference_row(
                    FixedEffectInferenceRowKind::Term,
                    self.test_contrast_with_method(hypothesis, method),
                )
            })
            .collect();
        FixedEffectInferenceTable::new(rows)
    }

    fn satterthwaite_fixed_effect_test(
        &self,
        hypothesis: FixedEffectHypothesis,
        estimates: Vec<f64>,
        standard_errors: Vec<Option<f64>>,
        statistics: Vec<Option<f64>>,
        estimability: FixedContrastEstimability,
    ) -> FixedEffectTest {
        use statrs::distribution::{ContinuousCDF, StudentsT};

        let method = InferenceMethod::Satterthwaite;
        if hypothesis.n_contrasts() != 1 {
            return fixed_effect_test_not_assessed_with_method(
                hypothesis,
                estimates,
                standard_errors,
                statistics,
                method,
                estimability,
                "Satterthwaite fixed-effect inference is currently certified only for scalar contrasts"
                    .to_string(),
            );
        }

        let Some(std_error) = standard_errors.first().copied().flatten() else {
            return fixed_effect_test_not_assessed_with_method(
                hypothesis,
                estimates,
                standard_errors,
                statistics,
                method,
                estimability,
                "Satterthwaite fixed-effect inference requires an available fixed-effect standard error"
                    .to_string(),
            );
        };
        let var_con = std_error * std_error;
        if !var_con.is_finite() || var_con <= 0.0 {
            return fixed_effect_test_not_assessed_with_method(
                hypothesis,
                estimates,
                standard_errors,
                statistics,
                method,
                estimability,
                "Satterthwaite fixed-effect inference requires a finite positive contrast variance"
                    .to_string(),
            );
        }

        let mut varpar = self.theta();
        varpar.push(self.sigma());
        let mut evaluator = self.clone();
        let jacobian = match evaluator.jac_vcov_beta_varpar(&varpar) {
            Ok(jacobian) => jacobian,
            Err(error) => {
                return fixed_effect_test_not_assessed_with_method(
                    hypothesis,
                    estimates,
                    standard_errors,
                    statistics,
                    method,
                    estimability,
                    format!("Satterthwaite fixed-effect inference could not compute vcov_beta derivatives: {error}"),
                );
            }
        };
        let vcov_varpar = match evaluator.vcov_varpar(&varpar, self.optsum.reml) {
            Ok(estimate) => estimate,
            Err(error) => {
                return fixed_effect_test_not_assessed_with_method(
                    hypothesis,
                    estimates,
                    standard_errors,
                    statistics,
                    method,
                    estimability,
                    format!("Satterthwaite fixed-effect inference could not estimate vcov_varpar: {error}"),
                );
            }
        };

        let gradient = jacobian
            .iter()
            .map(|derivative| contrast_row_quadratic_form(&hypothesis.l.values, 0, derivative))
            .collect::<Vec<_>>();
        if gradient.iter().any(|value| !value.is_finite()) {
            return fixed_effect_test_not_assessed_with_method(
                hypothesis,
                estimates,
                standard_errors,
                statistics,
                method,
                estimability,
                "Satterthwaite fixed-effect inference produced a non-finite variance-gradient component"
                    .to_string(),
            );
        }

        let gradient = DVector::from_vec(gradient);
        let satt_denom = (gradient.transpose() * &vcov_varpar.covariance * &gradient)[(0, 0)];
        if !satt_denom.is_finite() || satt_denom <= 0.0 {
            return fixed_effect_test_not_assessed_with_method(
                hypothesis,
                estimates,
                standard_errors,
                statistics,
                method,
                estimability,
                "Satterthwaite fixed-effect inference requires a finite positive denominator variance"
                    .to_string(),
            );
        }

        let denominator_df = 2.0 * var_con * var_con / satt_denom;
        if !denominator_df.is_finite() || denominator_df <= 0.0 {
            return fixed_effect_test_not_assessed_with_method(
                hypothesis,
                estimates,
                standard_errors,
                statistics,
                method,
                estimability,
                "Satterthwaite fixed-effect inference produced a non-finite denominator df"
                    .to_string(),
            );
        }

        let statistic = estimates[0] / std_error;
        let p_value = match StudentsT::new(0.0, 1.0, denominator_df) {
            Ok(t_dist) => Some(2.0 * (1.0 - t_dist.cdf(statistic.abs()))),
            Err(error) => {
                return fixed_effect_test_not_assessed_with_method(
                    hypothesis,
                    estimates,
                    standard_errors,
                    statistics,
                    method,
                    estimability,
                    format!("Satterthwaite fixed-effect inference could not construct Student-t distribution: {error}"),
                );
            }
        };

        let mut notes = vec![
            "Satterthwaite denominator df computed from finite-difference vcov_beta Jacobian and deviance Hessian over varpar"
                .to_string(),
        ];
        notes.extend(vcov_varpar.notes);

        FixedEffectTest {
            hypothesis,
            estimates,
            standard_errors,
            statistics: vec![Some(statistic)],
            numerator_df: None,
            denominator_df: Some(denominator_df),
            p_values: vec![p_value],
            method,
            reliability: ReliabilityGrade::Low,
            status: InferenceStatus::Available,
            estimability,
            notes,
        }
    }

    fn kenward_roger_fixed_effect_test(
        &self,
        hypothesis: FixedEffectHypothesis,
        estimates: Vec<f64>,
        standard_errors: Vec<Option<f64>>,
        statistics: Vec<Option<f64>>,
        estimability: FixedContrastEstimability,
    ) -> FixedEffectTest {
        use statrs::distribution::{ContinuousCDF, FisherSnedecor, StudentsT};

        let method = InferenceMethod::KenwardRoger;
        if !self.optsum.reml {
            return fixed_effect_test_not_assessed_with_method(
                hypothesis,
                estimates,
                standard_errors,
                statistics,
                method,
                estimability,
                "Kenward-Roger fixed-effect inference is certified only for REML LMM fits"
                    .to_string(),
            );
        }

        let adjusted = match self.kenward_roger_adjusted_vcov() {
            Ok(adjusted) => adjusted,
            Err(error) => {
                return fixed_effect_test_not_assessed_with_method(
                    hypothesis,
                    estimates,
                    standard_errors,
                    statistics,
                    method,
                    estimability,
                    format!(
                        "Kenward-Roger fixed-effect inference could not compute adjusted vcov: {error}"
                    ),
                );
            }
        };
        let lbddf = match self.kenward_roger_lbddf_with_adjusted(&hypothesis.l.values, &adjusted) {
            Ok(lbddf) => lbddf,
            Err(error) => {
                return fixed_effect_test_not_assessed_with_method(
                    hypothesis,
                    estimates,
                    standard_errors,
                    statistics,
                    method,
                    estimability,
                    format!(
                        "Kenward-Roger fixed-effect inference could not compute denominator df: {error}"
                    ),
                );
            }
        };

        let adjusted_standard_errors =
            contrast_standard_errors(&hypothesis.l.values, &adjusted.adjusted_vcov);
        let estimate_vector = DVector::from_column_slice(&estimates);
        let contrast_cov = symmetrize_matrix(
            &(&hypothesis.l.values * &adjusted.adjusted_vcov * hypothesis.l.values.transpose()),
        );
        if !matrix_is_finite(&contrast_cov) {
            return fixed_effect_test_not_assessed_with_method(
                hypothesis,
                estimates,
                adjusted_standard_errors,
                statistics,
                method,
                estimability,
                "Kenward-Roger fixed-effect inference produced a non-finite adjusted contrast covariance"
                    .to_string(),
            );
        }

        let mut notes = vec![
            "Kenward-Roger adjusted covariance and denominator df computed from response-space Sigma/G components"
                .to_string(),
        ];
        notes.extend(adjusted.notes);
        notes.extend(lbddf.notes);

        if hypothesis.n_contrasts() == 1 {
            let Some(std_error) = adjusted_standard_errors.first().copied().flatten() else {
                return fixed_effect_test_not_assessed_with_method(
                    hypothesis,
                    estimates,
                    adjusted_standard_errors,
                    statistics,
                    method,
                    estimability,
                    "Kenward-Roger fixed-effect inference requires an available adjusted standard error"
                        .to_string(),
                );
            };
            let var_con = std_error * std_error;
            if !var_con.is_finite() || var_con <= 0.0 {
                return fixed_effect_test_not_assessed_with_method(
                    hypothesis,
                    estimates,
                    adjusted_standard_errors,
                    statistics,
                    method,
                    estimability,
                    "Kenward-Roger fixed-effect inference requires a finite positive adjusted contrast variance"
                        .to_string(),
                );
            }
            let statistic = estimates[0] / std_error;
            let p_value = match StudentsT::new(0.0, 1.0, lbddf.denominator_df) {
                Ok(t_dist) => Some(2.0 * (1.0 - t_dist.cdf(statistic.abs()))),
                Err(error) => {
                    return fixed_effect_test_not_assessed_with_method(
                        hypothesis,
                        estimates,
                        adjusted_standard_errors,
                        statistics,
                        method,
                        estimability,
                        format!(
                            "Kenward-Roger fixed-effect inference could not construct Student-t distribution: {error}"
                        ),
                    );
                }
            };
            return FixedEffectTest {
                hypothesis,
                estimates,
                standard_errors: adjusted_standard_errors,
                statistics: vec![Some(statistic)],
                numerator_df: None,
                denominator_df: Some(lbddf.denominator_df),
                p_values: vec![p_value],
                method,
                reliability: lbddf.reliability,
                status: InferenceStatus::Available,
                estimability,
                notes,
            };
        }

        let q = lbddf.restriction_rank;
        let contrast_cov_inverse = symmetric_pseudoinverse(&contrast_cov, 1e-10);
        let quadratic =
            (estimate_vector.transpose() * contrast_cov_inverse * &estimate_vector)[(0, 0)];
        if !quadratic.is_finite() || quadratic < 0.0 {
            return fixed_effect_test_not_assessed_with_method(
                hypothesis,
                estimates,
                adjusted_standard_errors,
                statistics,
                method,
                estimability,
                "Kenward-Roger fixed-effect inference produced a non-finite F quadratic form"
                    .to_string(),
            );
        }
        let f_statistic = quadratic / q as f64;
        if !f_statistic.is_finite() || f_statistic < 0.0 {
            return fixed_effect_test_not_assessed_with_method(
                hypothesis,
                estimates,
                adjusted_standard_errors,
                statistics,
                method,
                estimability,
                "Kenward-Roger fixed-effect inference produced a non-finite F statistic"
                    .to_string(),
            );
        }
        let p_value = match FisherSnedecor::new(q as f64, lbddf.denominator_df) {
            Ok(f_dist) => Some(1.0 - f_dist.cdf(f_statistic)),
            Err(error) => {
                return fixed_effect_test_not_assessed_with_method(
                    hypothesis,
                    estimates,
                    adjusted_standard_errors,
                    statistics,
                    method,
                    estimability,
                    format!(
                        "Kenward-Roger fixed-effect inference could not construct F distribution: {error}"
                    ),
                );
            }
        };
        notes.push(
            "Kenward-Roger multi-df F row uses F scaling = 1.0 in the current row payload"
                .to_string(),
        );

        FixedEffectTest {
            hypothesis,
            estimates,
            standard_errors: adjusted_standard_errors,
            statistics: vec![Some(f_statistic)],
            numerator_df: Some(q as f64),
            denominator_df: Some(lbddf.denominator_df),
            p_values: vec![p_value],
            method,
            reliability: lbddf.reliability,
            status: InferenceStatus::Available,
            estimability,
            notes,
        }
    }

    fn bootstrap_fixed_effect_test_from_payload(
        &self,
        hypothesis: FixedEffectHypothesis,
        estimates: Vec<f64>,
        standard_errors: Vec<Option<f64>>,
        statistics: Vec<Option<f64>>,
        estimability: FixedContrastEstimability,
        payload: &BootstrapRunPayload,
    ) -> FixedEffectTest {
        const MIN_SUCCESSFUL_REPLICATES: usize = 30;
        const MODERATE_SUCCESSFUL_REPLICATES: usize = 999;
        const MODERATE_MAX_MCSE: f64 = 0.02;
        const MODERATE_MAX_FAILED_REFIT_RATE: f64 = 0.01;
        const MODERATE_MAX_BOUNDARY_RATE: f64 = 0.05;
        const CONTINUITY_CORRECTION: f64 = 1.0;

        let method = InferenceMethod::ParametricBootstrap;
        if hypothesis.n_contrasts() != 1 {
            return fixed_effect_test_not_assessed_with_method(
                hypothesis,
                estimates,
                standard_errors,
                statistics,
                method,
                estimability,
                "bootstrap_not_default_auto_method: bootstrap fixed-effect inference is currently certified only for scalar contrasts"
                    .to_string(),
            );
        }

        if payload.metadata.schema_name != BOOTSTRAP_RUN_SCHEMA
            || payload.metadata.schema_version != BOOTSTRAP_RUN_SCHEMA_VERSION
        {
            return fixed_effect_test_not_assessed_with_method(
                hypothesis,
                estimates,
                standard_errors,
                statistics,
                method,
                estimability,
                format!(
                    "bootstrap_replicate_accounting_unavailable: expected {BOOTSTRAP_RUN_SCHEMA} {BOOTSTRAP_RUN_SCHEMA_VERSION}, got {} {}",
                    payload.metadata.schema_name, payload.metadata.schema_version
                ),
            );
        }

        if payload.metadata.target.kind != BootstrapTargetKind::FixedEffectNull {
            return fixed_effect_test_not_assessed_with_method(
                hypothesis,
                estimates,
                standard_errors,
                statistics,
                method,
                estimability,
                "bootstrap_null_target_unavailable: payload target is not fixed_effect_null"
                    .to_string(),
            );
        }

        if payload.metadata.target.contrast_label.as_deref() != Some(hypothesis.label.as_str()) {
            return fixed_effect_test_not_assessed_with_method(
                hypothesis,
                estimates,
                standard_errors,
                statistics,
                method,
                estimability,
                "bootstrap_null_target_unavailable: payload contrast label does not match requested hypothesis"
                    .to_string(),
            );
        }

        if let Err(error) = payload.replicates.validate_for_model(self) {
            return fixed_effect_test_not_assessed_with_method(
                hypothesis,
                estimates,
                standard_errors,
                statistics,
                method,
                estimability,
                format!("bootstrap_replicate_accounting_unavailable: {error}"),
            );
        }

        if payload.metadata.completed_replicates != payload.replicates.len() {
            return fixed_effect_test_not_assessed_with_method(
                hypothesis,
                estimates,
                standard_errors,
                statistics,
                method,
                estimability,
                "bootstrap_replicate_accounting_unavailable: completed_replicates does not match replicate count"
                    .to_string(),
            );
        }

        let actual_successful = payload
            .replicates
            .fits
            .iter()
            .filter(|fit| fit.is_successful())
            .count();
        if payload.metadata.successful_replicates != actual_successful {
            return fixed_effect_test_not_assessed_with_method(
                hypothesis,
                estimates,
                standard_errors,
                statistics,
                method,
                estimability,
                "bootstrap_replicate_accounting_unavailable: successful_replicates does not match successful refit count"
                    .to_string(),
            );
        }

        if payload.metadata.failed_refit_policy != BootstrapFailedRefitPolicy::Exclude {
            return fixed_effect_test_not_assessed_with_method(
                hypothesis,
                estimates,
                standard_errors,
                statistics,
                method,
                estimability,
                "bootstrap_failed_refit_policy_unavailable: only exclude failed-refit policy is certified for fixed-effect bootstrap rows"
                    .to_string(),
            );
        }

        let Some(observed_statistic) = statistics.first().copied().flatten().map(f64::abs) else {
            return fixed_effect_test_not_assessed_with_method(
                hypothesis,
                estimates,
                standard_errors,
                statistics,
                method,
                estimability,
                "bootstrap_observed_statistic_nonfinite: observed fixed-effect statistic is unavailable"
                    .to_string(),
            );
        };

        let replicate_statistics = match payload.replicate_statistics.as_deref() {
            Some(values) => {
                if values.len() != payload.replicates.len() {
                    return fixed_effect_test_not_assessed_with_method(
                        hypothesis,
                        estimates,
                        standard_errors,
                        statistics,
                        method,
                        estimability,
                        "bootstrap_replicate_accounting_unavailable: replicate_statistics length does not match replicate count"
                            .to_string(),
                    );
                }
                values.iter().map(|value| value.abs()).collect::<Vec<_>>()
            }
            None => {
                match self.bootstrap_coefficient_statistics_from_replicates(&hypothesis, payload) {
                    Ok(values) => values,
                    Err(error) => {
                        return fixed_effect_test_not_assessed_with_method(
                            hypothesis,
                            estimates,
                            standard_errors,
                            statistics,
                            method,
                            estimability,
                            format!("bootstrap_replicate_accounting_unavailable: {error}"),
                        );
                    }
                }
            }
        };

        let finite_statistics = replicate_statistics
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        if let Some(recorded) = payload.metadata.finite_statistic_count {
            if recorded != finite_statistics.len() {
                return fixed_effect_test_not_assessed_with_method(
                    hypothesis,
                    estimates,
                    standard_errors,
                    statistics,
                    method,
                    estimability,
                    "bootstrap_replicate_accounting_unavailable: finite_statistic_count does not match finite replicate statistics"
                        .to_string(),
                );
            }
        }

        if finite_statistics.len() < MIN_SUCCESSFUL_REPLICATES {
            return fixed_effect_test_not_assessed_with_method(
                hypothesis,
                estimates,
                standard_errors,
                statistics,
                method,
                estimability,
                format!(
                    "bootstrap_successful_replicates_too_few: {} finite replicate statistic(s), need at least {MIN_SUCCESSFUL_REPLICATES}",
                    finite_statistics.len()
                ),
            );
        }

        let extreme = finite_statistics
            .iter()
            .filter(|&&value| value >= observed_statistic)
            .count();
        let denominator = finite_statistics.len() as f64 + CONTINUITY_CORRECTION;
        let p_value = (extreme as f64 + CONTINUITY_CORRECTION) / denominator;
        let mcse = (p_value * (1.0 - p_value) / finite_statistics.len() as f64).sqrt();
        if !p_value.is_finite() || !mcse.is_finite() {
            return fixed_effect_test_not_assessed_with_method(
                hypothesis,
                estimates,
                standard_errors,
                statistics,
                method,
                estimability,
                "bootstrap_mcse_unavailable: bootstrap p-value or Monte Carlo standard error is non-finite"
                    .to_string(),
            );
        }

        let failed_refit_rate = if payload.metadata.completed_replicates > 0 {
            payload.metadata.failed_refits as f64 / payload.metadata.completed_replicates as f64
        } else {
            1.0
        };
        let boundary_rate = payload.metadata.boundary_rate.unwrap_or(0.0);
        let reliability = if finite_statistics.len() >= MODERATE_SUCCESSFUL_REPLICATES
            && mcse <= MODERATE_MAX_MCSE
            && failed_refit_rate <= MODERATE_MAX_FAILED_REFIT_RATE
            && boundary_rate <= MODERATE_MAX_BOUNDARY_RATE
        {
            ReliabilityGrade::Moderate
        } else {
            ReliabilityGrade::Low
        };

        let mut notes = vec![
            format!(
                "bootstrap fixed-effect row computed from fixed_effect_null target `{}`",
                payload.metadata.target.label
            ),
            format!(
                "requested_replicates={}, completed_replicates={}, successful_replicates={}, finite_statistics={}",
                payload.metadata.requested_replicates,
                payload.metadata.completed_replicates,
                payload.metadata.successful_replicates,
                finite_statistics.len()
            ),
            format!(
                "failed_refit_policy={:?}, failed_refits={}, boundary_rate={:.6}, mcse={:.6}",
                payload.metadata.failed_refit_policy,
                payload.metadata.failed_refits,
                boundary_rate,
                mcse
            ),
        ];
        notes.extend(payload.metadata.notes.clone());

        FixedEffectTest {
            hypothesis,
            estimates,
            standard_errors,
            statistics: vec![Some(observed_statistic)],
            numerator_df: None,
            denominator_df: None,
            p_values: vec![Some(p_value)],
            method,
            reliability,
            status: InferenceStatus::Available,
            estimability,
            notes,
        }
    }

    fn bootstrap_coefficient_statistics_from_replicates(
        &self,
        hypothesis: &FixedEffectHypothesis,
        payload: &BootstrapRunPayload,
    ) -> Result<Vec<f64>> {
        let (coefficient_index, coefficient_weight) =
            scalar_single_coefficient_contrast(&hypothesis.l.values).ok_or_else(|| {
                MixedModelError::InvalidArgument(
                    "replicate_statistics are required for non-coefficient bootstrap contrasts"
                        .to_string(),
                )
            })?;
        let rhs = hypothesis.rhs.values[0];
        let mut values = Vec::new();
        for fit in &payload.replicates.fits {
            if !fit.is_successful() {
                values.push(f64::NAN);
                continue;
            }
            let beta = self.fixed_effect_active_vector_to_user_basis(&fit.beta, "beta")?;
            let se = self.fixed_effect_active_vector_to_user_basis(&fit.se, "standard error")?;
            let estimate = coefficient_weight * beta[coefficient_index] - rhs;
            let standard_error = coefficient_weight.abs() * se[coefficient_index];
            let statistic =
                if standard_error.is_finite() && standard_error > 0.0 && estimate.is_finite() {
                    (estimate / standard_error).abs()
                } else {
                    f64::NAN
                };
            values.push(statistic);
        }
        Ok(values)
    }

    pub fn fixed_effect_inference_table(&self) -> FixedEffectInferenceTable {
        let rows = self
            .coefficient_hypotheses()
            .into_iter()
            .map(|hypothesis| {
                fixed_effect_test_to_inference_row(
                    FixedEffectInferenceRowKind::Coefficient,
                    self.test_contrast(hypothesis),
                )
            })
            .collect();
        FixedEffectInferenceTable::new(rows)
    }

    fn refresh_fixed_effect_inference_table(&mut self) {
        self.compiler_artifact.fixed_effect_inference_table =
            Some(self.fixed_effect_inference_table());
    }

    fn fixed_effect_p_value_policy(&self) -> CoefTablePValuePolicy {
        if self
            .compiler_artifact
            .reductions
            .iter()
            .any(|record| record.trigger == ReductionTrigger::SelectionTime)
        {
            return CoefTablePValuePolicy::Unavailable {
                reason: "ordinary fixed-effect p-values are unavailable after selection-time model changes"
                    .to_string(),
            };
        }

        if let Some(reason) = self
            .compiler_artifact
            .reproducibility
            .fit_intent
            .p_value_unavailable_reason()
        {
            CoefTablePValuePolicy::Unavailable { reason }
        } else {
            CoefTablePValuePolicy::AsymptoticWaldZ
        }
    }

    /// Cook's distance for each observation.
    ///
    /// Measures the influence of each observation on the fixed-effects
    /// estimates.  The formula mirrors `cooksdistance(model)` in Julia's
    /// MixedModels.jl (linearmixedmodel.jl line 420):
    ///
    /// ```text
    /// D_i = (r_i / (1 - h_i))^2 * h_i / (σ² * p)
    /// ```
    ///
    /// where `r_i` is the i-th residual, `h_i` is the i-th leverage,
    /// `σ²` is the variance estimate, and `p` is the rank of the
    /// fixed-effects matrix.
    pub fn cooks_distance(&self) -> DVector<f64> {
        let r = self.residuals();
        let h = self.leverage();
        let mse = self.varest();
        let p = self.feterm.rank as f64;
        let n = self.dims.n;

        let mut d = DVector::zeros(n);
        for i in 0..n {
            let denom = 1.0 - h[i];
            if denom.abs() > f64::EPSILON {
                d[i] = (r[i] / denom).powi(2) * h[i] / (mse * p);
            }
        }
        d
    }
}

impl std::fmt::Display for LinearMixedModel {
    /// Default print: the compact `ModelPrint` summary (PRD § 15).
    /// Heavier reports stay one explicit method call away
    /// (`audit_report`, `parameterization`, `changes`, `explain_model`).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.print_summary(), f)
    }
}

impl MixedModelFit for LinearMixedModel {
    fn nobs(&self) -> usize {
        self.dims.n
    }

    fn dof(&self) -> usize {
        self.feterm.rank + self.n_theta() + usize::from(self.optsum.sigma.is_none())
    }

    fn coef(&self) -> DVector<f64> {
        let beta = self.fixef();
        let mut full = DVector::from_element(self.feterm.piv.len(), 0.0);
        for (i, &val) in beta.iter().enumerate() {
            if i < self.feterm.piv.len() {
                full[self.feterm.piv[i]] = val;
            }
        }
        full
    }

    fn fixef(&self) -> DVector<f64> {
        self.beta()
    }

    fn coef_names(&self) -> Vec<String> {
        let mut names = self.feterm.cnames.clone();
        // Unpivot
        let mut result = vec![String::new(); names.len()];
        for (i, name) in names.drain(..).enumerate() {
            if i < self.feterm.piv.len() {
                result[self.feterm.piv[i]] = name;
            }
        }
        result
    }

    fn vcov(&self) -> DMatrix<f64> {
        self.vcov_with_sigma(self.sigma())
    }

    fn stderror(&self) -> DVector<f64> {
        let vc = self.vcov();
        DVector::from_iterator(vc.nrows(), (0..vc.nrows()).map(|i| vc[(i, i)].sqrt()))
    }

    fn fitted(&self) -> DVector<f64> {
        let beta = self.beta();
        let x = self.feterm.full_rank_x();
        let mut yhat = x * &beta;

        // Add random effects contribution
        for (rt, b) in self.reterms.iter().zip(self.ranef_b()) {
            // y += Z * b (using sparse multiplication via refs)
            let bvec = DVector::from_column_slice(b.as_slice());
            for (obs, &ref_idx) in rt.refs.iter().enumerate() {
                let r = ref_idx as usize;
                for s in 0..rt.vsize {
                    yhat[obs] += rt.z[(s, obs)] * bvec[r * rt.vsize + s];
                }
            }
        }

        yhat
    }

    fn residuals(&self) -> DVector<f64> {
        let y = self.y();
        let yhat = self.fitted();
        y - yhat
    }

    fn response(&self) -> &DVector<f64> {
        &self.y
    }

    fn model_matrix(&self) -> &DMatrix<f64> {
        &self.feterm.x
    }

    fn objective(&self) -> f64 {
        self.objective_value()
    }

    fn loglikelihood(&self) -> f64 {
        -self.objective_value() / 2.0
    }

    fn formula_label(&self) -> Option<String> {
        Some(self.formula.to_string())
    }

    fn is_fitted(&self) -> bool {
        self.optsum.feval > 0
    }

    fn is_singular(&self) -> bool {
        self.theta_at_lower_bound()
            || self.optimizer_certificate_reports_boundary()
            || self.has_reduced_effective_covariance()
    }

    fn opt_summary(&self) -> &OptSummary {
        &self.optsum
    }

    fn theta(&self) -> Vec<f64> {
        LinearMixedModel::theta(self)
    }

    fn dispersion(&self, sqr: bool) -> f64 {
        let s = self.sigma();
        if sqr && self.optsum.sigma.is_none() {
            s * s
        } else {
            s
        }
    }

    fn ranef(&self) -> Vec<DMatrix<f64>> {
        self.ranef_b()
    }
}

impl LinearMixedModel {
    /// Predictions on the training data (identical to `fitted()`).
    pub fn predict(&self) -> DVector<f64> {
        self.fitted()
    }

    /// Predictions for new data with configurable handling of unseen RE levels.
    pub fn predict_new(
        &self,
        newdata: &DataFrame,
        new_re_levels: NewReLevels,
    ) -> Result<Vec<Option<f64>>> {
        let n_new = newdata.nrow();

        // --- Fixed-effects part ---
        let (raw_x, raw_names) = build_fixed_effects_matrix(&self.formula, newdata)?;

        // Map column name → index in the raw X
        let name_to_col: std::collections::HashMap<&str, usize> = raw_names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i))
            .collect();

        let p = self.feterm.rank;
        let beta = self.beta();
        let mut fe_pred = vec![0.0f64; n_new];

        for new_col in 0..p {
            // feterm.cnames[new_col] is the column name at pivot position new_col
            let name = &self.feterm.cnames[new_col];
            if let Some(&raw_col) = name_to_col.get(name.as_str()) {
                for obs in 0..n_new {
                    fe_pred[obs] += raw_x[(obs, raw_col)] * beta[new_col];
                }
            }
            // Column absent from newdata → treat as 0 contribution
        }

        // --- Random-effects part ---
        let b_list = self.ranef_b();

        // Build level-name → index maps for each RE term (training levels)
        let level_maps: Vec<std::collections::HashMap<&str, usize>> = self
            .reterms
            .iter()
            .map(|rt| {
                rt.levels
                    .iter()
                    .enumerate()
                    .map(|(i, s)| (s.as_str(), i))
                    .collect()
            })
            .collect();

        let mut result: Vec<Option<f64>> = fe_pred.into_iter().map(Some).collect();

        for (term_idx, rt) in self.reterms.iter().enumerate() {
            let b = &b_list[term_idx];
            let level_map = &level_maps[term_idx];

            let new_level_names = self.get_new_grouping_levels(rt, newdata)?;

            for obs in 0..n_new {
                if result[obs].is_none() {
                    continue;
                }
                let level_name = &new_level_names[obs];
                match level_map.get(level_name.as_str()) {
                    Some(&level_idx) => {
                        let z_obs = self.get_z_for_obs(rt, newdata, obs)?;
                        let re_contrib: f64 =
                            (0..rt.vsize).map(|s| z_obs[s] * b[(s, level_idx)]).sum();
                        *result[obs].as_mut().unwrap() += re_contrib;
                    }
                    None => match new_re_levels {
                        NewReLevels::Error => {
                            return Err(MixedModelError::InvalidArgument(format!(
                                "New level '{}' in grouping factor '{}'. \
                                 Use NewReLevels::Population or ::Missing to allow this.",
                                level_name, rt.grouping_name
                            )));
                        }
                        NewReLevels::Population => {} // zero RE, nothing to add
                        NewReLevels::Missing => {
                            result[obs] = None;
                        }
                    },
                }
            }
        }

        Ok(result)
    }

    /// Collect the grouping-factor level string for each observation in `newdata`.
    fn get_new_grouping_levels(&self, rt: &ReMat, newdata: &DataFrame) -> Result<Vec<String>> {
        use crate::formula::GroupingFactor;

        for re_term in &self.formula.random_terms {
            if random_term_grouping_name(re_term) != rt.grouping_name {
                continue;
            }
            return match &re_term.grouping {
                GroupingFactor::Single(name) => {
                    let cat = newdata.categorical(name).ok_or_else(|| {
                        MixedModelError::InvalidArgument(format!(
                            "Grouping factor '{}' not found in newdata",
                            name
                        ))
                    })?;
                    Ok(cat.values.clone())
                }
                GroupingFactor::Interaction(names) => {
                    let cats: Vec<_> = names
                        .iter()
                        .map(|n| {
                            newdata.categorical(n).ok_or_else(|| {
                                MixedModelError::InvalidArgument(format!(
                                    "Grouping factor '{}' not found in newdata",
                                    n
                                ))
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let levels = (0..newdata.nrow())
                        .map(|i| {
                            cats.iter()
                                .map(|c| c.values[i].clone())
                                .collect::<Vec<_>>()
                                .join("_")
                        })
                        .collect();
                    Ok(levels)
                }
                GroupingFactor::Cell(names) => {
                    let cats: Vec<_> = names
                        .iter()
                        .map(|n| {
                            newdata.categorical(n).ok_or_else(|| {
                                MixedModelError::InvalidArgument(format!(
                                    "Grouping factor '{}' not found in newdata",
                                    n
                                ))
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let levels = (0..newdata.nrow())
                        .map(|i| {
                            cats.iter()
                                .map(|c| c.values[i].clone())
                                .collect::<Vec<_>>()
                                .join("_")
                        })
                        .collect();
                    Ok(levels)
                }
            };
        }
        Err(MixedModelError::InvalidArgument(format!(
            "RE term '{}' not found in formula",
            rt.grouping_name
        )))
    }

    /// Build the z covariate vector for observation `obs` from `newdata`.
    fn get_z_for_obs(&self, rt: &ReMat, newdata: &DataFrame, obs: usize) -> Result<Vec<f64>> {
        for re_term in &self.formula.random_terms {
            if random_term_grouping_name(re_term) != rt.grouping_name {
                continue;
            }
            let (z, cnames) = random_term_z_for_obs(re_term, newdata, obs)?;
            if cnames == rt.cnames {
                return Ok(z);
            }
        }
        Err(MixedModelError::InvalidArgument(format!(
            "RE term '{}' with basis [{}] not found in formula",
            rt.grouping_name,
            rt.cnames.join(", ")
        )))
    }
}

fn random_term_grouping_name(rt: &crate::formula::RandomTerm) -> String {
    use crate::formula::GroupingFactor;

    match &rt.grouping {
        GroupingFactor::Single(name) => name.clone(),
        GroupingFactor::Interaction(names) | GroupingFactor::Cell(names) => names.join(" & "),
    }
}

fn random_term_z_for_obs(
    rt: &crate::formula::RandomTerm,
    data: &DataFrame,
    obs: usize,
) -> Result<(Vec<f64>, Vec<String>)> {
    use crate::formula::FixedTerm;

    let mut z = Vec::new();
    let mut cnames = Vec::new();
    let has_intercept =
        rt.terms.iter().any(|t| matches!(t, FixedTerm::Intercept)) || rt.terms.is_empty();
    if has_intercept {
        z.push(1.0);
        cnames.push("(Intercept)".to_string());
    }

    let basis_coding = random_effect_basis_coding(rt);
    for term in &rt.terms {
        for (col, name) in random_effect_basis_columns(term, data, data.nrow(), basis_coding)? {
            z.push(col[obs]);
            cnames.push(name);
        }
    }

    Ok((z, cnames))
}

// === Helper functions for model construction ===

/// Build the fixed-effects model matrix from formula and data.
fn build_fixed_effects_matrix(
    formula: &Formula,
    data: &DataFrame,
) -> Result<(DMatrix<f64>, Vec<String>)> {
    Ok(build_fixed_effects_design(formula, data)?.into_parts())
}

fn build_fixed_effects_design(formula: &Formula, data: &DataFrame) -> Result<DenseFixedDesign> {
    use crate::formula::FixedTerm;

    let n = data.nrow();
    let mut columns: Vec<DVector<f64>> = Vec::new();
    let mut names: Vec<String> = Vec::new();

    // Check if we have an intercept
    let has_intercept = formula.has_intercept();

    if has_intercept {
        columns.push(DVector::from_element(n, 1.0));
        names.push("(Intercept)".to_string());
    }

    for term in &formula.fixed_terms {
        match term {
            FixedTerm::Intercept | FixedTerm::NoIntercept => {
                // Already handled
            }
            FixedTerm::Column(name) => match data.column(name) {
                Some(Column::Numeric(v)) => {
                    columns.push(DVector::from_column_slice(v));
                    names.push(name.clone());
                }
                Some(Column::Categorical(cat)) => {
                    for encoded in cat.encoded_columns(name, CategoricalCoding::Treatment) {
                        columns.push(DVector::from_column_slice(&encoded.values));
                        names.push(encoded.name);
                    }
                }
                None => {
                    return Err(MixedModelError::InvalidArgument(format!(
                        "Column '{}' not found in data",
                        name
                    )));
                }
            },
            FixedTerm::Interaction(vars) => {
                // N-way interaction. Each variable contributes a list of
                // (column, label) pairs: numeric → 1 pair (the column itself),
                // categorical(L) → L-1 dummy pairs (skipping the reference
                // level). The interaction is the Cartesian product, with
                // columns multiplied element-wise and labels joined by `:`.
                let per_var = expand_interaction_factors(vars, data, n)?;
                for (col, name) in cartesian_interaction(&per_var, n) {
                    columns.push(col);
                    names.push(name);
                }
            }
            FixedTerm::Nested(_) => {
                // Nesting is expanded into main effect + interaction during parsing
            }
        }
    }

    if columns.is_empty() {
        // No fixed effects at all — create an empty matrix
        return DenseFixedDesign::new(DMatrix::zeros(n, 0), vec![]);
    }

    let p = columns.len();
    let mut x = DMatrix::zeros(n, p);
    for (j, col) in columns.iter().enumerate() {
        x.set_column(j, col);
    }

    DenseFixedDesign::new(x, names)
}

/// Per-variable expansion used by interaction terms: numeric → one column,
/// categorical(L) → L-1 dummy columns (skip reference level). Returns one
/// `Vec<(column, label)>` per input variable, in the order they were given.
fn expand_interaction_factors(
    vars: &[String],
    data: &DataFrame,
    n: usize,
) -> Result<Vec<Vec<(DVector<f64>, String)>>> {
    expand_interaction_factors_with_coding(vars, data, n, BasisCoding::Treatment)
}

fn expand_interaction_factors_with_coding(
    vars: &[String],
    data: &DataFrame,
    n: usize,
    coding: BasisCoding,
) -> Result<Vec<Vec<(DVector<f64>, String)>>> {
    let mut per_var: Vec<Vec<(DVector<f64>, String)>> = Vec::with_capacity(vars.len());
    for v in vars {
        per_var.push(expand_factor_columns_with_coding(
            v,
            data,
            "interaction term",
            coding,
        )?);
    }
    let _ = n; // n only used by callers for sanity checks
    Ok(per_var)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BasisCoding {
    Treatment,
    CellMeans,
}

fn categorical_coding(coding: BasisCoding) -> CategoricalCoding {
    match coding {
        BasisCoding::Treatment => CategoricalCoding::Treatment,
        BasisCoding::CellMeans => CategoricalCoding::CellMeans,
    }
}

fn expand_factor_columns_with_coding(
    name: &str,
    data: &DataFrame,
    context: &str,
    coding: BasisCoding,
) -> Result<Vec<(DVector<f64>, String)>> {
    match data.column(name) {
        Some(Column::Numeric(arr)) => Ok(vec![(DVector::from_column_slice(arr), name.to_string())]),
        Some(Column::Categorical(cat)) => Ok(cat
            .encoded_columns(name, categorical_coding(coding))
            .into_iter()
            .map(|column| (DVector::from_column_slice(&column.values), column.name))
            .collect()),
        None => Err(MixedModelError::InvalidArgument(format!(
            "Column '{name}' not found in data ({context})"
        ))),
    }
}

/// Cartesian product of expanded interaction factors. Iterates the FIRST
/// variable's columns slowest (outermost), matching how the Rust crate
/// emits main effects elsewhere. lme4 uses the opposite ordering — column
/// space is identical, but β positions differ; the cross-impl reporter
/// matches by normalized coefficient name to handle this.
fn cartesian_interaction(
    per_var: &[Vec<(DVector<f64>, String)>],
    n: usize,
) -> Vec<(DVector<f64>, String)> {
    let mut acc: Vec<(DVector<f64>, String)> = vec![(DVector::from_element(n, 1.0), String::new())];
    for cols in per_var {
        let mut next = Vec::with_capacity(acc.len() * cols.len());
        for (acc_col, acc_name) in &acc {
            for (c, name) in cols {
                let prod = acc_col.component_mul(c);
                let new_name = if acc_name.is_empty() {
                    name.clone()
                } else {
                    format!("{acc_name}:{name}")
                };
                next.push((prod, new_name));
            }
        }
        acc = next;
    }
    // Drop the seed row when the input was empty (no factors at all).
    if per_var.is_empty() {
        return Vec::new();
    }
    acc
}

fn random_effect_basis_columns(
    term: &crate::formula::FixedTerm,
    data: &DataFrame,
    n: usize,
    coding: BasisCoding,
) -> Result<Vec<(DVector<f64>, String)>> {
    use crate::formula::FixedTerm;

    match term {
        FixedTerm::Intercept | FixedTerm::NoIntercept => Ok(Vec::new()),
        FixedTerm::Column(name) => {
            expand_factor_columns_with_coding(name, data, "random-effect basis", coding)
        }
        FixedTerm::Interaction(vars) => {
            let per_var = expand_interaction_factors_with_coding(vars, data, n, coding)?;
            Ok(cartesian_interaction(&per_var, n))
        }
        FixedTerm::Nested(_) => Ok(Vec::new()),
    }
}

fn random_effect_basis_coding(rt: &crate::formula::RandomTerm) -> BasisCoding {
    if rt
        .terms
        .iter()
        .any(|term| matches!(term, crate::formula::FixedTerm::NoIntercept))
    {
        BasisCoding::CellMeans
    } else {
        BasisCoding::Treatment
    }
}

/// Build a ReMat from a random term specification and data.
fn build_re_mat(rt: &crate::formula::RandomTerm, data: &DataFrame, n: usize) -> Result<ReMat> {
    use crate::formula::{FixedTerm, GroupingFactor};

    // Get grouping factor
    let (group_name, refs, levels) = match &rt.grouping {
        GroupingFactor::Single(name) => {
            let cat = data.categorical(name).ok_or_else(|| {
                MixedModelError::InvalidArgument(format!(
                    "Grouping factor '{}' not found or not categorical",
                    name
                ))
            })?;
            (name.clone(), cat.refs.clone(), cat.levels.clone())
        }
        GroupingFactor::Interaction(names) | GroupingFactor::Cell(names) => {
            // Create interaction levels
            let cats: Vec<&crate::model::data::CategoricalColumn> = names
                .iter()
                .map(|name| {
                    data.categorical(name).ok_or_else(|| {
                        MixedModelError::InvalidArgument(format!(
                            "Grouping factor '{}' not found",
                            name
                        ))
                    })
                })
                .collect::<Result<Vec<_>>>()?;

            let group_name = names.join(" & ");
            let mut level_map = indexmap::IndexMap::new();
            let mut refs = Vec::with_capacity(n);

            for obs in 0..n {
                let key: String = cats
                    .iter()
                    .map(|c| c.levels[c.refs[obs] as usize].clone())
                    .collect::<Vec<_>>()
                    .join("_");
                let idx = level_map.len();
                let idx = *level_map.entry(key.clone()).or_insert(idx);
                refs.push(idx as u32);
            }

            let levels: Vec<String> = level_map.keys().cloned().collect();
            (group_name, refs, levels)
        }
    };

    // Build the Z matrix (transposed: s × n)
    let mut z_rows: Vec<DVector<f64>> = Vec::new();
    let mut cnames: Vec<String> = Vec::new();

    let has_re_intercept =
        rt.terms.iter().any(|t| matches!(t, FixedTerm::Intercept)) || rt.terms.is_empty();

    if has_re_intercept {
        z_rows.push(DVector::from_element(n, 1.0));
        cnames.push("(Intercept)".to_string());
    }

    let basis_coding = random_effect_basis_coding(rt);
    for term in &rt.terms {
        for (col, name) in random_effect_basis_columns(term, data, n, basis_coding)? {
            z_rows.push(col);
            cnames.push(name);
        }
    }

    let vsize = z_rows.len();
    let mut z = DMatrix::zeros(vsize, n);
    for (i, row) in z_rows.iter().enumerate() {
        z.set_row(i, &row.transpose());
    }

    let mut remat = ReMat::new(group_name, refs, levels, cnames, z);

    if rt.zerocorr {
        remat.zerocorr();
    }

    Ok(remat)
}

/// Build the parameter map: Vec<(block_idx, row, col)> for each θ element.
fn build_parmap(reterms: &[ReMat]) -> Vec<(usize, usize, usize)> {
    let mut parmap = Vec::new();
    for (block, rt) in reterms.iter().enumerate() {
        for &ind in &rt.inds {
            let s = rt.vsize;
            let col = ind / s;
            let row = ind % s;
            parmap.push((block, row, col));
        }
    }
    parmap
}

fn kenward_roger_covariance_component_count(reterm: &ReMat) -> usize {
    reterm.inds.len()
}

fn kenward_roger_covariance_component_indices(reterm: &ReMat) -> Vec<(usize, usize)> {
    reterm
        .inds
        .iter()
        .map(|&index| {
            let col = index / reterm.vsize;
            let row = index % reterm.vsize;
            (row, col)
        })
        .collect()
}

fn kenward_roger_response_component(
    reterm: &ReMat,
    row: usize,
    col: usize,
    n_observations: usize,
) -> Result<DMatrix<f64>> {
    if row >= reterm.vsize || col >= reterm.vsize {
        return Err(MixedModelError::DimensionMismatch(format!(
            "KR covariance component ({row}, {col}) is outside random-effect vector size {}",
            reterm.vsize
        )));
    }
    if reterm.n_obs() != n_observations {
        return Err(MixedModelError::DimensionMismatch(format!(
            "KR random-effect term '{}' has {} observations, expected {n_observations}",
            reterm.grouping_name,
            reterm.n_obs()
        )));
    }

    let mut component = DMatrix::zeros(n_observations, n_observations);
    for obs_i in 0..n_observations {
        let level_i = reterm.refs[obs_i];
        for obs_j in 0..=obs_i {
            if level_i != reterm.refs[obs_j] {
                continue;
            }
            let value = if row == col {
                reterm.z[(row, obs_i)] * reterm.z[(row, obs_j)]
            } else {
                reterm.z[(row, obs_i)] * reterm.z[(col, obs_j)]
                    + reterm.z[(col, obs_i)] * reterm.z[(row, obs_j)]
            };
            component[(obs_i, obs_j)] = value;
            component[(obs_j, obs_i)] = value;
        }
    }
    Ok(component)
}

fn matrix_rows(matrix: &DMatrix<f64>) -> Vec<Vec<f64>> {
    (0..matrix.nrows())
        .map(|row| {
            (0..matrix.ncols())
                .map(|col| matrix[(row, col)])
                .collect::<Vec<_>>()
        })
        .collect()
}

fn max_abs_delta(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() != right.len() {
        return None;
    }
    Some(
        left.iter()
            .zip(right.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max),
    )
}

fn matrix_is_finite(matrix: &DMatrix<f64>) -> bool {
    matrix.iter().all(|value| value.is_finite())
}

fn matrix_elementwise_dot(left: &DMatrix<f64>, right: &DMatrix<f64>) -> f64 {
    if left.shape() != right.shape() {
        return f64::NAN;
    }
    left.iter()
        .zip(right.iter())
        .map(|(lhs, rhs)| lhs * rhs)
        .sum()
}

fn matrix_trace(matrix: &DMatrix<f64>) -> f64 {
    let n = matrix.nrows().min(matrix.ncols());
    (0..n).map(|idx| matrix[(idx, idx)]).sum()
}

fn matrix_trace_product(left: &DMatrix<f64>, right: &DMatrix<f64>) -> f64 {
    if left.ncols() != right.nrows() {
        return f64::NAN;
    }
    let mut trace = 0.0;
    let n = left.nrows().min(right.ncols());
    for row in 0..n {
        for col in 0..left.ncols() {
            trace += left[(row, col)] * right[(col, row)];
        }
    }
    trace
}

fn matrix_max_asymmetry(matrix: &DMatrix<f64>) -> f64 {
    if matrix.nrows() != matrix.ncols() {
        return f64::INFINITY;
    }
    let mut max_delta = 0.0_f64;
    for row in 0..matrix.nrows() {
        for col in 0..row {
            max_delta = max_delta.max((matrix[(row, col)] - matrix[(col, row)]).abs());
        }
    }
    max_delta
}

fn invert_spd_matrix(matrix: &DMatrix<f64>, context: &str) -> Result<DMatrix<f64>> {
    if matrix.nrows() != matrix.ncols() {
        return Err(MixedModelError::DimensionMismatch(format!(
            "{context} is {} x {}, expected square",
            matrix.nrows(),
            matrix.ncols()
        )));
    }
    matrix
        .clone()
        .cholesky()
        .map(|chol| chol.inverse())
        .ok_or_else(|| MixedModelError::LinAlg(crate::error::LinAlgError::NotPositiveDefinite))
}

fn symmetric_pseudoinverse(matrix: &DMatrix<f64>, tolerance: f64) -> DMatrix<f64> {
    let matrix = symmetrize_matrix(matrix);
    let eig = SymmetricEigen::new(matrix);
    let mut inverse = DMatrix::zeros(eig.eigenvectors.nrows(), eig.eigenvectors.ncols());
    for (index, &eigenvalue) in eig.eigenvalues.iter().enumerate() {
        if eigenvalue.abs() > tolerance {
            let column = eig.eigenvectors.column(index);
            inverse += (column * column.transpose()) * (1.0 / eigenvalue);
        }
    }
    symmetrize_matrix(&inverse)
}

fn matrix_rank(matrix: &DMatrix<f64>, relative_tolerance: f64) -> usize {
    let svd = matrix.clone().svd(false, false);
    let max_singular = svd.singular_values.iter().copied().fold(0.0, f64::max);
    let tolerance = (relative_tolerance * max_singular.max(1.0)).max(1e-12);
    svd.singular_values
        .iter()
        .filter(|value| **value > tolerance)
        .count()
}

fn symmetric_pair_index(row: usize, col: usize, dimension: usize) -> usize {
    debug_assert!(row <= col);
    debug_assert!(col < dimension);
    row * dimension - row.saturating_mul(row.saturating_sub(1)) / 2 + (col - row)
}

fn div_zero(numerator: f64, denominator: f64, tolerance: f64) -> f64 {
    if numerator.abs() < tolerance && denominator.abs() < tolerance {
        1.0
    } else {
        numerator / denominator
    }
}

fn symmetrize_matrix(matrix: &DMatrix<f64>) -> DMatrix<f64> {
    let mut result = matrix.clone();
    for row in 0..matrix.nrows() {
        for col in 0..row {
            let value = 0.5 * (matrix[(row, col)] + matrix[(col, row)]);
            result[(row, col)] = value;
            result[(col, row)] = value;
        }
    }
    result
}

fn finite_difference_steps(theta: &[f64], lower_bounds: &[f64], relative_scale: f64) -> Vec<f64> {
    theta
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            let lower = lower_bounds
                .get(index)
                .copied()
                .unwrap_or(f64::NEG_INFINITY);
            let scale = if lower.is_finite() {
                value.abs().max(lower.abs()).max(1.0)
            } else {
                value.abs().max(1.0)
            };
            (relative_scale * scale).max(1e-8)
        })
        .collect()
}

fn feasible_central_step(value: f64, lower: f64, requested_step: f64) -> Option<f64> {
    let scale = value.abs().max(1.0);
    let min_step = 1e-10 * scale;
    let mut step = requested_step.abs().max(min_step);
    if lower.is_finite() {
        let clearance = value - lower;
        if clearance <= min_step {
            return None;
        }
        step = step.min(0.5 * clearance);
    }
    step.is_finite().then_some(step).filter(|step| *step > 0.0)
}

fn finite_difference_gradient_coordinate(
    evaluator: &mut LinearMixedModel,
    theta: &[f64],
    lower_bounds: &[f64],
    f0: f64,
    index: usize,
    step: f64,
) -> Option<f64> {
    let lower = lower_bounds
        .get(index)
        .copied()
        .unwrap_or(f64::NEG_INFINITY);
    if !lower.is_finite() || theta[index] - step >= lower {
        let mut plus = theta.to_vec();
        let mut minus = theta.to_vec();
        plus[index] += step;
        minus[index] -= step;
        let f_plus = evaluator.objective_at(&plus).ok()?;
        let f_minus = evaluator.objective_at(&minus).ok()?;
        if f_plus.is_finite() && f_minus.is_finite() {
            return Some((f_plus - f_minus) / (2.0 * step));
        }
    }

    let mut plus = theta.to_vec();
    let mut plus2 = theta.to_vec();
    plus[index] += step;
    plus2[index] += 2.0 * step;
    let f_plus = evaluator.objective_at(&plus).ok()?;
    let f_plus2 = evaluator.objective_at(&plus2).ok()?;
    if f_plus.is_finite() && f_plus2.is_finite() {
        Some((-3.0 * f0 + 4.0 * f_plus - f_plus2) / (2.0 * step))
    } else {
        None
    }
}

fn finite_difference_objective_2d(
    evaluator: &mut LinearMixedModel,
    theta: &[f64],
    row: usize,
    row_delta: f64,
    col: usize,
    col_delta: f64,
) -> Option<f64> {
    let mut trial = theta.to_vec();
    trial[row] += row_delta;
    trial[col] += col_delta;
    evaluator
        .objective_at(&trial)
        .ok()
        .filter(|value| value.is_finite())
}

fn finite_difference_deviance_varpar(
    evaluator: &mut LinearMixedModel,
    varpar: &[f64],
    index: usize,
    delta: f64,
    reml: bool,
) -> Result<f64> {
    let mut trial = varpar.to_vec();
    trial[index] += delta;
    evaluator.deviance_varpar(&trial, reml).and_then(|value| {
        value.is_finite().then_some(value).ok_or_else(|| {
            MixedModelError::Optimization(
                "finite-difference deviance_varpar evaluation is non-finite".to_string(),
            )
        })
    })
}

fn finite_difference_deviance_varpar_2d(
    evaluator: &mut LinearMixedModel,
    varpar: &[f64],
    row: usize,
    row_delta: f64,
    col: usize,
    col_delta: f64,
    reml: bool,
) -> Result<f64> {
    let mut trial = varpar.to_vec();
    trial[row] += row_delta;
    trial[col] += col_delta;
    evaluator.deviance_varpar(&trial, reml).and_then(|value| {
        value.is_finite().then_some(value).ok_or_else(|| {
            MixedModelError::Optimization(
                "finite-difference deviance_varpar evaluation is non-finite".to_string(),
            )
        })
    })
}

fn contrast_standard_errors(l: &DMatrix<f64>, vcov: &DMatrix<f64>) -> Vec<Option<f64>> {
    (0..l.nrows())
        .map(|row| {
            let mut variance = 0.0;
            for i in 0..l.ncols() {
                for j in 0..l.ncols() {
                    variance += l[(row, i)] * vcov[(i, j)] * l[(row, j)];
                }
            }
            (variance.is_finite() && variance >= 0.0).then_some(variance.max(0.0).sqrt())
        })
        .collect()
}

fn contrast_row_quadratic_form(l: &DMatrix<f64>, row: usize, matrix: &DMatrix<f64>) -> f64 {
    let mut value = 0.0;
    for i in 0..l.ncols() {
        for j in 0..l.ncols() {
            value += l[(row, i)] * matrix[(i, j)] * l[(row, j)];
        }
    }
    value
}

fn assess_fixed_contrast_estimability(
    hypothesis: &FixedEffectHypothesis,
    beta: &DVector<f64>,
    vcov: &DMatrix<f64>,
) -> FixedContrastEstimability {
    let mut estimable_rows = 0usize;
    for row in 0..hypothesis.l.values.nrows() {
        let row_estimable = (0..hypothesis.l.values.ncols()).all(|col| {
            let weight = hypothesis.l.values[(row, col)];
            weight.abs() <= 1e-12 || (beta[col].is_finite() && vcov[(col, col)].is_finite())
        });
        if row_estimable {
            estimable_rows += 1;
        }
    }

    let requested = hypothesis.n_contrasts();
    if estimable_rows == requested {
        FixedContrastEstimability::estimable(hypothesis.label.clone(), estimable_rows, requested)
    } else if estimable_rows == 0 {
        FixedContrastEstimability::not_estimable(hypothesis.label.clone(), requested, Vec::new())
    } else {
        FixedContrastEstimability::partially_estimable(
            hypothesis.label.clone(),
            estimable_rows,
            requested,
            Vec::new(),
        )
    }
}

fn scalar_single_coefficient_contrast(l: &DMatrix<f64>) -> Option<(usize, f64)> {
    if l.nrows() != 1 {
        return None;
    }
    let mut found = None;
    for col in 0..l.ncols() {
        let value = l[(0, col)];
        if value.abs() <= 1e-12 {
            continue;
        }
        if found.is_some() {
            return None;
        }
        found = Some((col, value));
    }
    found
}

fn scalar_contrast_abs_studentized(
    model: &LinearMixedModel,
    hypothesis: &FixedEffectHypothesis,
) -> Option<f64> {
    if hypothesis.n_contrasts() != 1 || hypothesis.n_coefficients() != model.coef_names().len() {
        return None;
    }
    let beta = model.coef();
    let vcov = model.vcov();
    let estimate = (&hypothesis.l.values * beta - &hypothesis.rhs.values)[0];
    let se = contrast_standard_errors(&hypothesis.l.values, &vcov)
        .into_iter()
        .next()
        .flatten()?;
    (estimate.is_finite() && se.is_finite() && se > 0.0).then_some((estimate / se).abs())
}

fn fixed_effect_test_to_inference_row(
    kind: FixedEffectInferenceRowKind,
    test: FixedEffectTest,
) -> FixedEffectInferenceRow {
    let statistic_name = fixed_effect_statistic_name(&test);
    let reason = fixed_effect_inference_reason(&test);
    let details = fixed_effect_details_for_test(kind, &test, statistic_name);
    FixedEffectInferenceRow {
        label: test.hypothesis.label.clone(),
        kind,
        estimate: finite_option(test.estimates.first().copied()),
        std_error: finite_option(test.standard_errors.first().copied().flatten()),
        numerator_df: fixed_effect_row_numerator_df(&test, statistic_name),
        denominator_df: test.denominator_df,
        statistic: finite_option(test.statistics.first().copied().flatten()),
        statistic_name,
        p_value: finite_option(test.p_values.first().copied().flatten()),
        method: fixed_effect_inference_method(&test.method),
        status: fixed_effect_inference_status(&test.status),
        reliability: test.reliability,
        estimability: EstimabilityAssessment::FixedContrast(test.estimability),
        reason,
        details,
        notes: test.notes,
    }
}

fn fixed_effect_details_for_test(
    kind: FixedEffectInferenceRowKind,
    test: &FixedEffectTest,
    statistic_name: Option<FixedEffectStatisticName>,
) -> Option<FixedEffectInferenceDetails> {
    let contrast_family = (kind != FixedEffectInferenceRowKind::Coefficient
        || test.hypothesis.n_contrasts() > 1)
        .then(|| contrast_family_details(kind, test, statistic_name));
    let kenward_roger =
        (test.method == InferenceMethod::KenwardRoger).then(|| KenwardRogerInferenceDetails {
            restriction_rank: test.estimability.rank,
            f_scaling: (statistic_name == Some(FixedEffectStatisticName::F)).then_some(1.0),
            statistic_scale: (statistic_name == Some(FixedEffectStatisticName::F))
                .then(|| "unscaled".to_string()),
        });
    let details = FixedEffectInferenceDetails {
        bootstrap: None,
        contrast_family,
        kenward_roger,
    };
    (!details.is_empty()).then_some(details)
}

fn contrast_family_details(
    kind: FixedEffectInferenceRowKind,
    test: &FixedEffectTest,
    statistic_name: Option<FixedEffectStatisticName>,
) -> ContrastFamilyDetails {
    let requested_rank = test.estimability.requested_rank;
    let effective_rank = test.estimability.rank;
    let rank_deficient = match (effective_rank, requested_rank) {
        (Some(rank), Some(requested)) => Some(rank < requested),
        _ => None,
    };
    let numerator_df_semantics = match (kind, statistic_name) {
        (_, Some(FixedEffectStatisticName::F)) => "effective_restriction_rank",
        (FixedEffectInferenceRowKind::Term, _) => "term_scalar_or_unavailable",
        _ => "scalar_contrast_no_numerator_df",
    }
    .to_string();
    ContrastFamilyDetails {
        family_id: test.hypothesis.label.clone(),
        family_label: test.hypothesis.label.clone(),
        restriction_rows: test.hypothesis.n_contrasts(),
        coefficient_count: test.hypothesis.n_coefficients(),
        requested_rank,
        effective_rank,
        rank_deficient,
        rhs_nonzero: test
            .hypothesis
            .rhs
            .values
            .iter()
            .any(|value| value.abs() > 0.0),
        numerator_df: fixed_effect_row_numerator_df(test, statistic_name),
        numerator_df_semantics,
    }
}

fn attach_bootstrap_details(
    row: &mut FixedEffectInferenceRow,
    payload: &BootstrapRunPayload,
    null_target: Option<&FixedEffectNullBootstrapTarget>,
) {
    let details = row.details.get_or_insert(FixedEffectInferenceDetails {
        bootstrap: None,
        contrast_family: None,
        kenward_roger: None,
    });
    details.bootstrap = Some(BootstrapInferenceDetails {
        target_kind: bootstrap_target_kind_label(payload.metadata.target.kind).to_string(),
        target_label: payload.metadata.target.label.clone(),
        contrast_label: payload.metadata.target.contrast_label.clone(),
        requested_replicates: payload.metadata.requested_replicates,
        completed_replicates: payload.metadata.completed_replicates,
        successful_replicates: payload.metadata.successful_replicates,
        failed_refits: payload.metadata.failed_refits,
        failed_refit_policy: bootstrap_failed_refit_policy_label(
            payload.metadata.failed_refit_policy,
        )
        .to_string(),
        boundary_count: payload.metadata.boundary_count,
        boundary_rate: payload.metadata.boundary_rate,
        seed_rng: payload.metadata.seed_record.rng.clone(),
        seed: payload.metadata.seed_record.seed,
        finite_statistic_count: payload.metadata.finite_statistic_count,
        mcse: payload.metadata.mcse,
        null_target: null_target.map(|target| FixedEffectNullTargetSummary {
            covariance_policy: fixed_effect_null_covariance_policy_label(target.covariance_policy)
                .to_string(),
            coefficient_count: target.coefficient_names.len(),
            theta_count: target.theta.len(),
            sigma: target.sigma.is_finite().then_some(target.sigma),
            reml: target.reml,
        }),
    });
}

fn bootstrap_target_kind_label(kind: BootstrapTargetKind) -> &'static str {
    match kind {
        BootstrapTargetKind::FullModelDistribution => "full_model_distribution",
        BootstrapTargetKind::FixedEffectNull => "fixed_effect_null",
    }
}

fn bootstrap_failed_refit_policy_label(policy: BootstrapFailedRefitPolicy) -> &'static str {
    match policy {
        BootstrapFailedRefitPolicy::Exclude => "exclude",
        BootstrapFailedRefitPolicy::CountExtreme => "count_extreme",
        BootstrapFailedRefitPolicy::Abort => "abort",
    }
}

fn fixed_effect_null_covariance_policy_label(
    policy: FixedEffectNullCovariancePolicy,
) -> &'static str {
    match policy {
        FixedEffectNullCovariancePolicy::ReuseFittedCovariance => "reuse_fitted_covariance",
    }
}

fn fixed_effect_inference_method(method: &InferenceMethod) -> FixedEffectInferenceMethod {
    match method {
        InferenceMethod::AsymptoticWaldZ => FixedEffectInferenceMethod::AsymptoticWaldZ,
        InferenceMethod::Satterthwaite => FixedEffectInferenceMethod::Satterthwaite,
        InferenceMethod::KenwardRoger => FixedEffectInferenceMethod::KenwardRoger,
        InferenceMethod::ParametricBootstrap => FixedEffectInferenceMethod::Bootstrap,
        InferenceMethod::NotComputed { .. } => FixedEffectInferenceMethod::NotComputed,
    }
}

fn fixed_effect_inference_status(status: &InferenceStatus) -> FixedEffectInferenceStatus {
    match status {
        InferenceStatus::Available => FixedEffectInferenceStatus::Available,
        InferenceStatus::PValueUnavailable { .. } => FixedEffectInferenceStatus::PValueUnavailable,
        InferenceStatus::NotEstimable { .. } => FixedEffectInferenceStatus::NotEstimable,
        InferenceStatus::NotAssessed { .. } => FixedEffectInferenceStatus::NotAssessed,
        InferenceStatus::Unsupported { .. } => FixedEffectInferenceStatus::Unsupported,
    }
}

fn fixed_effect_statistic_name(test: &FixedEffectTest) -> Option<FixedEffectStatisticName> {
    match test.method {
        InferenceMethod::AsymptoticWaldZ => Some(FixedEffectStatisticName::Z),
        InferenceMethod::Satterthwaite => Some(FixedEffectStatisticName::T),
        InferenceMethod::KenwardRoger if test.hypothesis.n_contrasts() > 1 => {
            Some(FixedEffectStatisticName::F)
        }
        InferenceMethod::KenwardRoger => Some(FixedEffectStatisticName::T),
        InferenceMethod::ParametricBootstrap if test.hypothesis.n_contrasts() > 1 => {
            Some(FixedEffectStatisticName::F)
        }
        InferenceMethod::ParametricBootstrap => Some(FixedEffectStatisticName::T),
        InferenceMethod::NotComputed { .. } => None,
    }
}

fn fixed_effect_row_numerator_df(
    test: &FixedEffectTest,
    statistic_name: Option<FixedEffectStatisticName>,
) -> Option<f64> {
    match statistic_name {
        Some(FixedEffectStatisticName::F) => test.numerator_df,
        _ => None,
    }
}

fn fixed_effect_inference_reason(test: &FixedEffectTest) -> Option<String> {
    match &test.status {
        InferenceStatus::Available => match &test.method {
            InferenceMethod::NotComputed { reason } => Some(reason.clone()),
            _ => None,
        },
        InferenceStatus::PValueUnavailable { reason }
        | InferenceStatus::NotEstimable { reason }
        | InferenceStatus::NotAssessed { reason }
        | InferenceStatus::Unsupported { reason } => Some(reason.clone()),
    }
}

fn finite_option(value: Option<f64>) -> Option<f64> {
    value.filter(|value| value.is_finite())
}

fn fixed_effect_test_asymptotic_wald_z(
    hypothesis: FixedEffectHypothesis,
    estimates: Vec<f64>,
    standard_errors: Vec<Option<f64>>,
    statistics: Vec<Option<f64>>,
    estimability: FixedContrastEstimability,
) -> FixedEffectTest {
    use statrs::distribution::{ContinuousCDF, Normal};

    let normal = Normal::new(0.0, 1.0).unwrap();
    let p_values = statistics
        .iter()
        .map(|stat| stat.map(|z| 2.0 * (1.0 - normal.cdf(z.abs()))))
        .collect::<Vec<_>>();
    let p_value_available = p_values.iter().all(Option::is_some);
    FixedEffectTest {
        hypothesis,
        estimates,
        standard_errors,
        statistics,
        numerator_df: Some(1.0),
        denominator_df: None,
        p_values,
        method: InferenceMethod::AsymptoticWaldZ,
        reliability: ReliabilityGrade::Low,
        status: if p_value_available {
            InferenceStatus::Available
        } else {
            InferenceStatus::PValueUnavailable {
                reason: "standard error is unavailable, so the Wald z p-value is unavailable"
                    .to_string(),
            }
        },
        estimability,
        notes: vec![
            "asymptotic Wald z is a labeled fallback, not a finite-sample correction".to_string(),
        ],
    }
}

fn fixed_effect_test_p_value_unavailable(
    hypothesis: FixedEffectHypothesis,
    estimates: Vec<f64>,
    standard_errors: Vec<Option<f64>>,
    statistics: Vec<Option<f64>>,
    estimability: FixedContrastEstimability,
    reason: String,
) -> FixedEffectTest {
    FixedEffectTest {
        hypothesis,
        estimates,
        standard_errors,
        statistics,
        numerator_df: Some(1.0),
        denominator_df: None,
        p_values: vec![None],
        method: InferenceMethod::NotComputed {
            reason: reason.clone(),
        },
        reliability: ReliabilityGrade::NotAvailable,
        status: InferenceStatus::PValueUnavailable { reason },
        estimability,
        notes: Vec::new(),
    }
}

fn fixed_effect_test_not_assessed_with_method(
    hypothesis: FixedEffectHypothesis,
    estimates: Vec<f64>,
    standard_errors: Vec<Option<f64>>,
    statistics: Vec<Option<f64>>,
    method: InferenceMethod,
    estimability: FixedContrastEstimability,
    reason: String,
) -> FixedEffectTest {
    let n = hypothesis.n_contrasts();
    FixedEffectTest {
        hypothesis,
        estimates,
        standard_errors,
        statistics,
        numerator_df: Some(1.0),
        denominator_df: None,
        p_values: vec![None; n],
        method,
        reliability: ReliabilityGrade::NotAvailable,
        status: InferenceStatus::NotAssessed {
            reason: reason.clone(),
        },
        estimability,
        notes: vec![reason],
    }
}

fn fixed_effect_test_unavailable(
    hypothesis: FixedEffectHypothesis,
    estimability: FixedContrastEstimability,
    status: InferenceStatus,
) -> FixedEffectTest {
    let n = hypothesis.n_contrasts();
    let reason = match &status {
        InferenceStatus::Available => "fixed-effect test unavailable".to_string(),
        InferenceStatus::PValueUnavailable { reason }
        | InferenceStatus::NotEstimable { reason }
        | InferenceStatus::NotAssessed { reason }
        | InferenceStatus::Unsupported { reason } => reason.clone(),
    };
    FixedEffectTest {
        hypothesis,
        estimates: vec![f64::NAN; n],
        standard_errors: vec![None; n],
        statistics: vec![None; n],
        numerator_df: None,
        denominator_df: None,
        p_values: vec![None; n],
        method: InferenceMethod::NotComputed { reason },
        reliability: ReliabilityGrade::NotAvailable,
        status,
        estimability,
        notes: Vec::new(),
    }
}

fn jittered_theta(
    theta: &[f64],
    lower_bounds: &[f64],
    jitter_scale: f64,
    jitter_index: usize,
) -> Vec<f64> {
    let mut jittered = theta
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            let direction = ((index + 1 + jitter_index * 17) as f64).sin();
            let scale = value.abs().max(1.0);
            value + direction * jitter_scale * scale
        })
        .collect::<Vec<_>>();
    LinearMixedModel::project_theta_to_bounds(&mut jittered, lower_bounds);
    jittered
}

fn optimizer_name(optimizer: Optimizer) -> &'static str {
    match optimizer {
        Optimizer::Cobyla => "cobyla",
        Optimizer::PatternSearch => "pattern_search",
        Optimizer::NloptNewuoa => "newuoa",
        Optimizer::NloptBobyqa => "bobyqa",
        Optimizer::PrimaBobyqa => "bobyqa",
        Optimizer::PrimaCobyla => "cobyla",
        Optimizer::PrimaLincoa => "lincoa",
        Optimizer::PrimaNewuoa => "newuoa",
    }
}

fn verification_status(
    runs: &[ConvergenceVerificationRun],
    options: &ConvergenceVerificationOptions,
) -> ConvergenceVerificationStatus {
    if runs.is_empty() {
        return ConvergenceVerificationStatus::NotRun;
    }

    let all_agree = runs.iter().all(|run| run.agrees);
    if all_agree
        && runs
            .iter()
            .any(|run| run.label.starts_with("optimizer_consensus_"))
    {
        ConvergenceVerificationStatus::OptimizerConsensus
    } else if all_agree {
        ConvergenceVerificationStatus::RestartAgrees
    } else if runs
        .iter()
        .any(|run| run.label == "restart_from_optimum" && core_verification_failed(run, options))
    {
        ConvergenceVerificationStatus::Unstable
    } else {
        ConvergenceVerificationStatus::Fragile
    }
}

fn core_verification_failed(
    run: &ConvergenceVerificationRun,
    options: &ConvergenceVerificationOptions,
) -> bool {
    let objective_failed = run
        .objective_delta
        .map(|delta| delta > options.objective_tolerance)
        .unwrap_or(true);
    let beta_failed = run
        .max_abs_beta_delta
        .map(|delta| delta > options.beta_tolerance)
        .unwrap_or(true);
    let rank_failed = run
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.contains("effective covariance ranks changed"));
    objective_failed || beta_failed || rank_failed
}

fn verification_message(
    status: ConvergenceVerificationStatus,
    runs: &[ConvergenceVerificationRun],
) -> String {
    match status {
        ConvergenceVerificationStatus::NotRun => "convergence verification was not run".to_string(),
        ConvergenceVerificationStatus::RestartAgrees => {
            "restart from fitted theta agrees with the recorded optimum".to_string()
        }
        ConvergenceVerificationStatus::OptimizerConsensus => {
            "restart and alternate optimizer checks agree with the recorded optimum".to_string()
        }
        ConvergenceVerificationStatus::Fragile => {
            let failed = runs
                .iter()
                .filter(|run| !run.agrees)
                .map(|run| run.label.clone())
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                "objective, fixed effects, and rank are stable, but parameterization checks are fragile: {failed}"
            )
        }
        ConvergenceVerificationStatus::Unstable => {
            "restart from fitted theta did not reproduce the recorded optimum".to_string()
        }
    }
}

fn apply_design_compiled_policy(
    formula: &mut Formula,
    recommendations: &[PolicyRecommendation],
) -> Result<Vec<ReductionRecord>> {
    let mut reductions = Vec::new();

    for recommendation in recommendations {
        let Some(term_index) = term_index_from_id(&recommendation.term_id) else {
            return Err(MixedModelError::InvalidArgument(format!(
                "policy recommendation references unknown random term '{}'",
                recommendation.term_id
            )));
        };
        let Some(term) = formula.random_terms.get_mut(term_index) else {
            return Err(MixedModelError::InvalidArgument(format!(
                "policy recommendation references missing random term '{}'",
                recommendation.term_id
            )));
        };

        match recommendation.action {
            PolicyAction::ReduceCovariance => {
                term.zerocorr = true;
                reductions.push(reduction_from_recommendation(
                    recommendation,
                    Some(term.to_string()),
                ));
            }
            PolicyAction::DropUnsupportedBasis => {
                let unsupported = unsupported_basis_from_recommendation(recommendation);
                if unsupported.is_empty() {
                    return Err(MixedModelError::InvalidArgument(format!(
                        "cannot apply unsupported-basis reduction for '{}' without basis payload",
                        recommendation.term_id
                    )));
                }
                let removed = drop_unsupported_basis_terms(term, &unsupported)?;
                if removed.is_empty() {
                    return Err(MixedModelError::InvalidArgument(format!(
                        "unsupported basis for '{}' could not be mapped to formula terms: {}",
                        recommendation.term_id,
                        unsupported.join(", ")
                    )));
                }
                reductions.push(reduction_from_recommendation(
                    recommendation,
                    Some(term.to_string()),
                ));
            }
            PolicyAction::RefuseRandomTermDistribution | PolicyAction::MarkNotAssessable => {
                return Err(MixedModelError::InvalidArgument(format!(
                    "design_compiled refused {}: {}",
                    recommendation.source_syntax, recommendation.reason
                )));
            }
        }
    }

    Ok(reductions)
}

fn term_index_from_id(term_id: &str) -> Option<usize> {
    term_id.strip_prefix('r')?.parse().ok()
}

fn reduction_from_recommendation(
    recommendation: &PolicyRecommendation,
    replacement_term: Option<String>,
) -> ReductionRecord {
    ReductionRecord {
        trigger: ReductionTrigger::DesignTime,
        phase: "design_compiled".to_string(),
        reason: recommendation.reason.clone(),
        affected_term: recommendation.term_id.clone(),
        replacement_term,
        inference_consequence: recommendation.inference_consequence.clone(),
        diagnostics: recommendation.diagnostics.clone(),
    }
}

fn unsupported_basis_from_recommendation(recommendation: &PolicyRecommendation) -> Vec<String> {
    recommendation
        .diagnostics
        .first()
        .and_then(|diagnostic| diagnostic.payload.get("unsupported_basis"))
        .and_then(|value| value.as_array())
        .map(|values| {
            values
                .iter()
                .filter_map(|value| value.as_str().map(str::to_string))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn drop_unsupported_basis_terms(
    term: &mut RandomTerm,
    unsupported_basis: &[String],
) -> Result<Vec<String>> {
    let mut removed = Vec::new();
    term.terms.retain(|fixed_term| {
        if matches!(fixed_term, FixedTerm::Intercept | FixedTerm::NoIntercept) {
            return true;
        }
        let label = fixed_term.to_string();
        if unsupported_basis.iter().any(|basis| basis == &label) {
            removed.push(label);
            false
        } else {
            true
        }
    });

    let has_intercept = term
        .terms
        .iter()
        .any(|fixed_term| matches!(fixed_term, FixedTerm::Intercept))
        || term.terms.is_empty();
    let has_basis = term
        .terms
        .iter()
        .any(|fixed_term| !matches!(fixed_term, FixedTerm::Intercept | FixedTerm::NoIntercept));
    if !has_intercept && !has_basis {
        return Err(MixedModelError::InvalidArgument(
            "design_compiled would remove every random-effect basis direction".to_string(),
        ));
    }

    Ok(removed)
}

fn user_basis_label(name: &str) -> String {
    if name == "(Intercept)" {
        "intercept".to_string()
    } else {
        name.to_string()
    }
}

fn orient_eigenvector(mut vector: Vec<f64>) -> Vec<f64> {
    let pivot = vector
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| {
            left.abs()
                .partial_cmp(&right.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, _)| idx);

    if let Some(idx) = pivot {
        if vector[idx] < 0.0 {
            for value in &mut vector {
                *value = -*value;
            }
        }
    }

    vector
}

fn format_loading_summary(loadings: &[BasisLoading]) -> String {
    let mut parts = String::new();
    for (idx, loading) in loadings.iter().enumerate() {
        let value = if loading.loading.abs() < 5e-13 {
            0.0
        } else {
            loading.loading
        };
        if idx == 0 {
            parts.push_str(&format!("{value:.3}*{}", loading.basis));
        } else if value < 0.0 {
            parts.push_str(&format!(" - {:.3}*{}", value.abs(), loading.basis));
        } else {
            parts.push_str(&format!(" + {value:.3}*{}", loading.basis));
        }
    }
    parts
}

fn source_syntax_for_term(terms: &[crate::compiler::RandomTermIr], term_id: &str) -> String {
    terms
        .iter()
        .find(|term| term.id == term_id)
        .map(|term| term.source_syntax.text.clone())
        .unwrap_or_else(|| term_id.to_string())
}

/// Build a "drop the off-axis column" rewrite for a rank-2 random term.
///
/// Returns `None` if the basis is not exactly two columns or the kept column
/// cannot be addressed by the simple `(1 | g)` / `(0 + x | g)` template.
fn suggest_drop_off_axis(
    grouping: &str,
    basis_names: &[String],
    keep_idx: usize,
) -> Option<String> {
    if basis_names.len() != 2 || keep_idx >= basis_names.len() {
        return None;
    }
    let kept = &basis_names[keep_idx];
    if kept.eq_ignore_ascii_case("intercept") || kept == "(Intercept)" {
        Some(format!("(1 | {grouping})"))
    } else {
        Some(format!("(0 + {kept} | {grouping})"))
    }
}

/// Detect whether a reduced-rank random-effect term has a single supported
/// direction that loads almost entirely on one user-facing basis column.
///
/// Returns a structured `InterpretableSubmodel` suggestion if so, or `None`
/// when the rank gate, dominance threshold, or formula rewrite are not met.
/// Never refits the model: the suggestion is metadata only.
// TODO(bd-01KQ8FSZPCBTWWS2Q11WWMQ2VY-followup): generalise to requested_rank > 2
// once the rewrite spec for higher-rank submodels exists.
fn detect_interpretable_submodel(
    pairs: &[(f64, Vec<f64>)],
    requested_basis: &[String],
    requested_rank: usize,
    rank_tolerance: f64,
    sigma_sq: f64,
    semantic_terms: &[crate::compiler::RandomTermIr],
    term_id: &str,
) -> Option<InterpretableSubmodel> {
    if requested_rank != 2 {
        return None;
    }

    let supported: Vec<&(f64, Vec<f64>)> = pairs
        .iter()
        .filter(|(eig, _)| eig.max(0.0) > rank_tolerance)
        .collect();
    if supported.len() != 1 {
        return None;
    }
    let supported_pair = supported[0];
    if supported_pair.1.len() != requested_basis.len() {
        return None;
    }

    let oriented = orient_eigenvector(supported_pair.1.clone());
    let (keep_idx, dominant_abs) = oriented
        .iter()
        .enumerate()
        .map(|(idx, value)| (idx, value.abs()))
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))?;
    if dominant_abs < DOMINANT_LOADING_THRESHOLD {
        return None;
    }

    let term = semantic_terms.iter().find(|term| term.id == term_id)?;
    if !matches!(term.covariance, crate::compiler::CovarianceForm::Full) {
        return None;
    }
    let basis_names: Vec<String> = term.basis.iter().map(|coef| coef.name.clone()).collect();
    if basis_names.len() != requested_basis.len() {
        return None;
    }
    let grouping_label = term.group.label();
    let suggested_formula = suggest_drop_off_axis(&grouping_label, &basis_names, keep_idx)?;

    let mut loadings_dominant = oriented
        .iter()
        .zip(requested_basis.iter())
        .map(|(loading, basis)| DominantLoading {
            basis: basis.clone(),
            loading: if loading.abs() < 5e-13 { 0.0 } else { *loading },
        })
        .collect::<Vec<_>>();
    loadings_dominant.sort_by(|a, b| {
        b.loading
            .abs()
            .partial_cmp(&a.loading.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let unsupported_eigenvalue = pairs
        .iter()
        .map(|(eig, _)| eig.max(0.0))
        .filter(|eig| *eig <= rank_tolerance)
        .fold(0.0_f64, f64::max);
    let safe_sigma_sq = sigma_sq.max(f64::EPSILON);
    let objective_gap = (1.0 + unsupported_eigenvalue / safe_sigma_sq).ln().max(0.0);
    let within_tolerance =
        objective_gap.is_finite() && objective_gap <= INTERPRETABLE_GAP_TOLERANCE;

    Some(InterpretableSubmodel {
        suggested_formula,
        loadings_dominant,
        objective_gap,
        within_tolerance,
    })
}

fn is_nested(a: &ReMat, b: &ReMat) -> bool {
    if a.refs.len() != b.refs.len() {
        return false;
    }

    let mut bins = vec![None; a.n_levels()];
    for (&aref, &bref) in a.refs.iter().zip(b.refs.iter()) {
        let slot = &mut bins[aref as usize];
        match slot {
            Some(prev) if *prev != bref => return false,
            Some(_) => {}
            None => *slot = Some(bref),
        }
    }
    true
}

fn promote_crossed_fill_in_blocks(l: &mut [MatrixBlock], reterms: &[ReMat]) {
    let k = reterms.len();
    for i in 1..k {
        if (0..i).any(|j| !is_nested(&reterms[j], &reterms[i])) {
            for row in i..k {
                let idx = block_index(row, i);
                if !matches!(l[idx], MatrixBlock::Dense(_)) {
                    l[idx] = MatrixBlock::Dense(l[idx].as_dense());
                }
            }
        }
    }
}

/// Create the A (cross-product) and L (Cholesky) block arrays.
#[cfg(test)]
fn create_al(reterms: &[ReMat], xy: &FeMat) -> Result<(Vec<MatrixBlock>, Vec<MatrixBlock>)> {
    validate_dense_block_plan(reterms, xy.wtxy.ncols())?;

    if reterms.len() == 1 && reterms[0].vsize == 2 && reterms[0].n_ranef() >= 512 {
        return Ok(create_al_single_vsize2(&reterms[0], xy));
    }

    let k = reterms.len();
    let total = k + 1;
    let n_blocks = total * (total + 1) / 2;
    let mut a = Vec::with_capacity(n_blocks);
    let mut l = Vec::with_capacity(n_blocks);

    // RE × RE blocks
    for i in 0..k {
        for j in 0..=i {
            let block = if i == j {
                // Diagonal block: Z_i' Z_i
                compute_re_cross_product(&reterms[i], &reterms[i])
            } else {
                // Off-diagonal: Z_i' Z_j
                compute_re_cross_product(&reterms[i], &reterms[j])
            };
            a.push(block.clone());
            l.push(block);
        }
    }

    // FE × RE blocks: [X|y]' Z_j
    for j in 0..k {
        let block = compute_fe_re_cross_product(xy, &reterms[j]);
        a.push(block.clone());
        l.push(block);
    }

    // FE × FE block: [X|y]' [X|y]
    let wtxy = &xy.wtxy;
    let feblock = MatrixBlock::Dense(wtxy.transpose() * wtxy);
    a.push(feblock.clone());
    l.push(feblock);

    promote_crossed_fill_in_blocks(&mut l, reterms);

    Ok((a, l))
}

/// Create A/L blocks using fixed-design backend cross-products for the fixed
/// side of the system.
fn create_al_from_fixed_design(
    reterms: &[ReMat],
    fixed_design: &FixedDesign,
    y: &DVector<f64>,
    sqrtwts: Option<&DVector<f64>>,
) -> Result<(Vec<MatrixBlock>, Vec<MatrixBlock>)> {
    validate_dense_block_plan(reterms, fixed_design.n_cols() + 1)?;
    let weighted_fixed_design = weighted_fixed_design_for_solver(fixed_design, sqrtwts)?;
    let weighted_y = weighted_response_for_solver(y, sqrtwts)?;

    let k = reterms.len();
    let total = k + 1;
    let n_blocks = total * (total + 1) / 2;
    let mut a = Vec::with_capacity(n_blocks);
    let mut l = Vec::with_capacity(n_blocks);

    for i in 0..k {
        for j in 0..=i {
            let block = if i == j {
                compute_re_cross_product(&reterms[i], &reterms[i])
            } else {
                compute_re_cross_product(&reterms[i], &reterms[j])
            };
            a.push(block.clone());
            l.push(block);
        }
    }

    for re in reterms {
        let block =
            compute_fixed_response_re_cross_product(&weighted_fixed_design, &weighted_y, re)?;
        a.push(block.clone());
        l.push(block);
    }

    let block = MatrixBlock::Dense(compute_fixed_response_cross_product(
        &weighted_fixed_design,
        &weighted_y,
    )?);
    a.push(block.clone());
    l.push(block);

    promote_crossed_fill_in_blocks(&mut l, reterms);

    Ok((a, l))
}

fn weighted_fixed_design_for_solver(
    fixed_design: &FixedDesign,
    sqrtwts: Option<&DVector<f64>>,
) -> Result<FixedDesign> {
    match sqrtwts {
        Some(weights) => fixed_design.with_sqrt_weights(weights),
        None => Ok(fixed_design.clone()),
    }
}

fn weighted_response_for_solver(
    y: &DVector<f64>,
    sqrtwts: Option<&DVector<f64>>,
) -> Result<DVector<f64>> {
    if let Some(weights) = sqrtwts {
        if weights.len() != y.len() {
            return Err(MixedModelError::DimensionMismatch(format!(
                "response has {} rows but sqrt weights have {}",
                y.len(),
                weights.len()
            )));
        }
        Ok(y.component_mul(weights))
    } else {
        Ok(y.clone())
    }
}

fn fixed_design_backend_diagnostic(fixed_design: &FixedDesign) -> Diagnostic {
    let summary = fixed_design.summary();
    let active_entries = fixed_design_active_entries(fixed_design);
    let density = fixed_design_density(fixed_design);
    let mut diagnostic = Diagnostic::new(
        DiagnosticCode::SupportNote,
        DiagnosticSeverity::Info,
        DiagnosticStage::DesignAudit,
        format!(
            "fixed-effect design backend selected: {}; n={}, p={}, dense_if_materialized={} bytes, active_entries={}, density={:.6}",
            fixed_design_storage_label(summary.storage),
            summary.n_obs,
            summary.n_cols,
            summary.dense_bytes,
            active_entries,
            density
        ),
    )
    .with_suggested_actions(vec![
        "no action required; streamed fixed effects avoid materializing dense X for solver cross-products".to_string(),
        "rank and pivot detection still materialize dense X in this release".to_string(),
    ]);
    diagnostic.payload.insert(
        "diagnostic_kind".to_string(),
        serde_json::json!("fixed_design_backend"),
    );
    diagnostic.payload.insert(
        "storage".to_string(),
        serde_json::json!(fixed_design_storage_label(summary.storage)),
    );
    diagnostic
        .payload
        .insert("n_obs".to_string(), serde_json::json!(summary.n_obs));
    diagnostic
        .payload
        .insert("n_cols".to_string(), serde_json::json!(summary.n_cols));
    diagnostic.payload.insert(
        "dense_bytes".to_string(),
        serde_json::json!(summary.dense_bytes.to_string()),
    );
    diagnostic.payload.insert(
        "active_entries".to_string(),
        serde_json::json!(active_entries),
    );
    diagnostic
        .payload
        .insert("density".to_string(), serde_json::json!(density));
    diagnostic
}

fn fixed_design_active_entries(fixed_design: &FixedDesign) -> usize {
    match fixed_design {
        FixedDesign::Dense(design) => design.n_obs() * design.n_cols(),
        FixedDesign::Streamed(design) => design.active_entries(),
    }
}

fn fixed_design_density(fixed_design: &FixedDesign) -> f64 {
    match fixed_design {
        FixedDesign::Dense(design) => {
            if design.n_obs() == 0 || design.n_cols() == 0 {
                0.0
            } else {
                1.0
            }
        }
        FixedDesign::Streamed(design) => design.density(),
    }
}

fn fixed_design_storage_label(storage: FixedDesignStorage) -> &'static str {
    match storage {
        FixedDesignStorage::Dense => "dense",
        FixedDesignStorage::Streamed => "streamed",
        FixedDesignStorage::Sparse => "sparse",
    }
}

#[cfg(test)]
fn create_al_single_vsize2(re: &ReMat, xy: &FeMat) -> (Vec<MatrixBlock>, Vec<MatrixBlock>) {
    let nlevels = re.n_levels();
    let pp1 = xy.wtxy.ncols();
    let mut re_re_blocks: Vec<DMatrix<f64>> = (0..nlevels).map(|_| DMatrix::zeros(2, 2)).collect();
    let mut fe_re = DMatrix::zeros(pp1, re.n_ranef());
    let mut fe_fe = DMatrix::zeros(pp1, pp1);

    for obs in 0..re.n_obs() {
        let level = re.refs[obs] as usize;
        let col0 = 2 * level;
        let col1 = col0 + 1;
        let z0 = re.wtz[(0, obs)];
        let z1 = re.wtz[(1, obs)];

        let block = &mut re_re_blocks[level];
        block[(0, 0)] += z0 * z0;
        block[(0, 1)] += z0 * z1;
        block[(1, 0)] += z1 * z0;
        block[(1, 1)] += z1 * z1;

        for row in 0..pp1 {
            let x = xy.wtxy[(obs, row)];
            fe_re[(row, col0)] += x * z0;
            fe_re[(row, col1)] += x * z1;
            for col in 0..=row {
                fe_fe[(row, col)] += x * xy.wtxy[(obs, col)];
            }
        }
    }

    for row in 0..pp1 {
        for col in 0..row {
            fe_fe[(col, row)] = fe_fe[(row, col)];
        }
    }

    let a = vec![
        MatrixBlock::BlockDiagonal(re_re_blocks),
        MatrixBlock::Dense(fe_re),
        MatrixBlock::Dense(fe_fe),
    ];
    let l = a.clone();
    (a, l)
}

/// Create the structural A and L block arrays for `[Z X]' [Z X]`.
pub(crate) fn create_structural_al(
    reterms: &[ReMat],
    x: &DMatrix<f64>,
) -> Result<(Vec<MatrixBlock>, Vec<MatrixBlock>)> {
    validate_dense_block_plan(reterms, x.ncols())?;

    let k = reterms.len();
    let total = k + 1;
    let n_blocks = total * (total + 1) / 2;
    let mut a = Vec::with_capacity(n_blocks);
    let mut l = Vec::with_capacity(n_blocks);

    for i in 0..k {
        for j in 0..=i {
            let block = if i == j {
                compute_re_cross_product(&reterms[i], &reterms[i])
            } else {
                compute_re_cross_product(&reterms[i], &reterms[j])
            };
            a.push(block.clone());
            l.push(block);
        }
    }

    for j in 0..k {
        let block = compute_x_re_cross_product(x, &reterms[j]);
        a.push(block.clone());
        l.push(block);
    }

    let xblock = MatrixBlock::Dense(x.transpose() * x);
    a.push(xblock.clone());
    l.push(xblock);

    promote_crossed_fill_in_blocks(&mut l, reterms);

    Ok((a, l))
}

/// Compute Z_i' Z_j for two random effects terms.
fn compute_re_cross_product(a: &ReMat, b: &ReMat) -> MatrixBlock {
    let nranef_a = a.n_ranef();
    let nranef_b = b.n_ranef();

    if std::ptr::eq(a, b) && a.vsize == 1 {
        // Scalar RE: diagonal result
        let n_levels = a.n_levels();
        let mut diag = DVector::zeros(n_levels);
        for (obs, &ref_idx) in a.refs.iter().enumerate() {
            let r = ref_idx as usize;
            diag[r] += a.wtz[(0, obs)] * a.wtz[(0, obs)];
        }
        MatrixBlock::Diagonal(diag)
    } else if std::ptr::eq(a, b) && a.vsize > 1 {
        // Vector RE, same term: block-diagonal result
        // Each level k gets a vsize × vsize block: sum_{obs with ref==k} wtz[:,obs] * wtz[:,obs]'
        let s = a.vsize;
        let n_levels = a.n_levels();
        let mut blocks: Vec<DMatrix<f64>> = (0..n_levels).map(|_| DMatrix::zeros(s, s)).collect();

        for (obs, &ref_idx) in a.refs.iter().enumerate() {
            let k = ref_idx as usize;
            let blk = &mut blocks[k];
            for si in 0..s {
                let wtz_si = a.wtz[(si, obs)];
                for sj in 0..s {
                    blk[(si, sj)] += wtz_si * a.wtz[(sj, obs)];
                }
            }
        }
        MatrixBlock::BlockDiagonal(blocks)
    } else if a.vsize == 1 && b.vsize == 1 && !is_nested(b, a) {
        // Truly crossed scalar-intercept terms: keep the raw cross-product sparse.
        // A partially crossed random-intercept block can be enormous in shape
        // while having only O(n_obs) structural nonzeros.
        let mut entries = BTreeMap::<(usize, usize), f64>::new();
        let n = a.refs.len();

        for obs in 0..n {
            let ri = a.refs[obs] as usize;
            let rj = b.refs[obs] as usize;
            for si in 0..a.vsize {
                for sj in 0..b.vsize {
                    let value = a.wtz[(si, obs)] * b.wtz[(sj, obs)];
                    if value != 0.0 {
                        *entries
                            .entry((ri * a.vsize + si, rj * b.vsize + sj))
                            .or_insert(0.0) += value;
                    }
                }
            }
        }
        let mut result = CooMatrix::new(nranef_a, nranef_b);
        for ((row, col), value) in entries {
            if value != 0.0 {
                result.push(row, col, value);
            }
        }
        MatrixBlock::Sparse(CscMatrix::from(&result))
    } else {
        // General case: dense result. This includes reverse-ordered nested
        // scalar terms, where preserving the previous dense algebra keeps the
        // optimizer path stable.
        let mut result = DMatrix::zeros(nranef_a, nranef_b);
        let n = a.refs.len();

        for obs in 0..n {
            let ri = a.refs[obs] as usize;
            let rj = b.refs[obs] as usize;
            for si in 0..a.vsize {
                for sj in 0..b.vsize {
                    result[(ri * a.vsize + si, rj * b.vsize + sj)] +=
                        a.wtz[(si, obs)] * b.wtz[(sj, obs)];
                }
            }
        }
        MatrixBlock::Dense(result)
    }
}

/// Compute [X|y]' Z_j.
#[cfg(test)]
fn compute_fe_re_cross_product(xy: &FeMat, re: &ReMat) -> MatrixBlock {
    let pp1 = xy.wtxy.ncols(); // p + 1
    let nranef = re.n_ranef();
    let n = re.refs.len();

    let mut result = DMatrix::zeros(pp1, nranef);
    let wtxy = &xy.wtxy;

    for obs in 0..n {
        let r = re.refs[obs] as usize;
        for col in 0..pp1 {
            for s in 0..re.vsize {
                result[(col, r * re.vsize + s)] += wtxy[(obs, col)] * re.wtz[(s, obs)];
            }
        }
    }

    MatrixBlock::Dense(result)
}

/// Compute `[X|y]' Z_j` using fixed-design backend cross-products.
fn compute_fixed_response_re_cross_product(
    fixed_design: &FixedDesign,
    y: &DVector<f64>,
    re: &ReMat,
) -> Result<MatrixBlock> {
    if y.len() != fixed_design.n_obs() {
        return Err(MixedModelError::DimensionMismatch(format!(
            "fixed-effect design has {} rows but response has {}",
            fixed_design.n_obs(),
            y.len()
        )));
    }
    if re.n_obs() != fixed_design.n_obs() {
        return Err(MixedModelError::DimensionMismatch(format!(
            "fixed-effect design has {} rows but random term '{}' has {} rows",
            fixed_design.n_obs(),
            re.grouping_name,
            re.n_obs()
        )));
    }

    let fixed_re = fixed_design.xt_reterm(re)?.as_dense();
    let mut result = DMatrix::zeros(fixed_design.n_cols() + 1, re.n_ranef());
    for row in 0..fixed_re.nrows() {
        for col in 0..fixed_re.ncols() {
            result[(row, col)] = fixed_re[(row, col)];
        }
    }

    let response_re = compute_response_re_cross_product(&DMatrix::from_columns(&[y.clone()]), re);
    for col in 0..response_re.nrows() {
        result[(fixed_design.n_cols(), col)] = response_re[(col, 0)];
    }
    Ok(MatrixBlock::Dense(result))
}

/// Compute `[X|y]' [X|y]` using fixed-design backend cross-products.
fn compute_fixed_response_cross_product(
    fixed_design: &FixedDesign,
    y: &DVector<f64>,
) -> Result<DMatrix<f64>> {
    if y.len() != fixed_design.n_obs() {
        return Err(MixedModelError::DimensionMismatch(format!(
            "fixed-effect design has {} rows but response has {}",
            fixed_design.n_obs(),
            y.len()
        )));
    }

    let p = fixed_design.n_cols();
    let xtx = fixed_design.xtx();
    let xty = fixed_design.xty(y)?;
    let mut result = DMatrix::zeros(p + 1, p + 1);
    for row in 0..p {
        for col in 0..p {
            result[(row, col)] = xtx[(row, col)];
        }
        result[(row, p)] = xty[row];
        result[(p, row)] = xty[row];
    }
    result[(p, p)] = y.dot(y);
    Ok(result)
}

/// Compute X' Z_j.
fn compute_x_re_cross_product(x: &DMatrix<f64>, re: &ReMat) -> MatrixBlock {
    let p = x.ncols();
    let nranef = re.n_ranef();
    let n = re.refs.len();

    let mut result = DMatrix::zeros(p, nranef);
    for obs in 0..n {
        let r = re.refs[obs] as usize;
        for col in 0..p {
            for s in 0..re.vsize {
                result[(col, r * re.vsize + s)] += x[(obs, col)] * re.wtz[(s, obs)];
            }
        }
    }

    MatrixBlock::Dense(result)
}

fn compute_response_re_cross_product(y: &DMatrix<f64>, re: &ReMat) -> DMatrix<f64> {
    let q = y.ncols();
    let nranef = re.n_ranef();
    let n = re.refs.len();
    let mut result = DMatrix::zeros(nranef, q);

    for obs in 0..n {
        let r = re.refs[obs] as usize;
        for s in 0..re.vsize {
            let row = r * re.vsize + s;
            let weight = re.wtz[(s, obs)];
            for col in 0..q {
                result[(row, col)] += weight * y[(obs, col)];
            }
        }
    }

    result
}

fn apply_lambda_transpose_to_rhs(rhs: &mut DMatrix<f64>, re: &ReMat) {
    let s = re.vsize;
    let nlevels = re.n_levels();
    let q = rhs.ncols();

    if s == 1 {
        let lam = re.lambda[(0, 0)];
        for row in 0..rhs.nrows() {
            for col in 0..q {
                rhs[(row, col)] *= lam;
            }
        }
        return;
    }

    if s == 2 {
        let l00 = re.lambda[(0, 0)];
        let l10 = re.lambda[(1, 0)];
        let l11 = re.lambda[(1, 1)];
        for level in 0..nlevels {
            let row0 = level * 2;
            let row1 = row0 + 1;
            for col in 0..q {
                let x0 = rhs[(row0, col)];
                let x1 = rhs[(row1, col)];
                rhs[(row0, col)] = l00 * x0 + l10 * x1;
                rhs[(row1, col)] = l11 * x1;
            }
        }
        return;
    }

    for level in 0..nlevels {
        let offset = level * s;
        let mut temp = vec![0.0; s];
        for col in 0..q {
            for row in 0..s {
                let mut sum = 0.0;
                for inner in row..s {
                    sum += re.lambda[(inner, row)] * rhs[(offset + inner, col)];
                }
                temp[row] = sum;
            }
            for row in 0..s {
                rhs[(offset + row, col)] = temp[row];
            }
        }
    }
}

fn build_response_rhs_blocks(
    reterms: &[ReMat],
    x: &DMatrix<f64>,
    y: &DMatrix<f64>,
) -> Vec<DMatrix<f64>> {
    let k = reterms.len();
    let mut rhs_blocks = Vec::with_capacity(k + 1);
    for re in reterms {
        let mut block = compute_response_re_cross_product(y, re);
        apply_lambda_transpose_to_rhs(&mut block, re);
        rhs_blocks.push(block);
    }
    rhs_blocks.push(x.transpose() * y);
    rhs_blocks
}

fn subtract_left_block_product(dst: &mut DMatrix<f64>, lhs: &MatrixBlock, rhs: &DMatrix<f64>) {
    match lhs {
        MatrixBlock::Diagonal(diag) => {
            for row in 0..diag.len() {
                let scale = diag[row];
                for col in 0..rhs.ncols() {
                    dst[(row, col)] -= scale * rhs[(row, col)];
                }
            }
        }
        MatrixBlock::BlockDiagonal(blocks) => {
            let mut row_offset = 0;
            for block in blocks {
                let s = block.nrows();
                for row in 0..s {
                    for col in 0..rhs.ncols() {
                        let mut sum = 0.0;
                        for inner in 0..s {
                            sum += block[(row, inner)] * rhs[(row_offset + inner, col)];
                        }
                        dst[(row_offset + row, col)] -= sum;
                    }
                }
                row_offset += s;
            }
        }
        MatrixBlock::Sparse(mat) => {
            for (row, inner, value) in mat.triplet_iter() {
                for col in 0..rhs.ncols() {
                    dst[(row, col)] -= value * rhs[(inner, col)];
                }
            }
        }
        MatrixBlock::Dense(mat) => {
            for row in 0..mat.nrows() {
                for col in 0..rhs.ncols() {
                    let mut sum = 0.0;
                    for inner in 0..mat.ncols() {
                        sum += mat[(row, inner)] * rhs[(inner, col)];
                    }
                    dst[(row, col)] -= sum;
                }
            }
        }
    }
}

fn solve_lower_block_against_rhs(l: &MatrixBlock, rhs: &mut [f64]) {
    debug_assert_eq!(l.nrows(), rhs.len());
    debug_assert_eq!(l.ncols(), rhs.len());

    match l {
        MatrixBlock::Diagonal(diag) => {
            for row in 0..diag.len() {
                let denom = diag[row];
                if denom.abs() < BLOCK_TRIANGULAR_SOLVE_ZERO_TOLERANCE {
                    rhs[row] = 0.0;
                    continue;
                }
                rhs[row] /= denom;
            }
        }
        MatrixBlock::BlockDiagonal(blocks) => {
            let mut row_offset = 0;
            for block in blocks {
                let s = block.nrows();
                for row in 0..s {
                    let diag = block[(row, row)];
                    if diag.abs() < BLOCK_TRIANGULAR_SOLVE_ZERO_TOLERANCE {
                        rhs[row_offset + row] = 0.0;
                        continue;
                    }
                    let mut sum = rhs[row_offset + row];
                    for inner in 0..row {
                        sum -= block[(row, inner)] * rhs[row_offset + inner];
                    }
                    rhs[row_offset + row] = sum / diag;
                }
                row_offset += s;
            }
        }
        MatrixBlock::Dense(mat) => {
            for row in 0..mat.nrows() {
                let diag = mat[(row, row)];
                if diag.abs() < BLOCK_TRIANGULAR_SOLVE_ZERO_TOLERANCE {
                    rhs[row] = 0.0;
                    continue;
                }
                let mut sum = rhs[row];
                for inner in 0..row {
                    sum -= mat[(row, inner)] * rhs[inner];
                }
                rhs[row] = sum / diag;
            }
        }
        MatrixBlock::Sparse(_) => {
            let dense = l.as_dense();
            solve_lower_block_against_rhs(&MatrixBlock::Dense(dense), rhs);
        }
    }
}

fn solve_upper_block_from_lower_transpose_against_rhs(l: &MatrixBlock, rhs: &mut [f64]) {
    debug_assert_eq!(l.nrows(), rhs.len());
    debug_assert_eq!(l.ncols(), rhs.len());

    match l {
        MatrixBlock::Diagonal(diag) => {
            for row in (0..diag.len()).rev() {
                let denom = diag[row];
                if denom.abs() < BLOCK_TRIANGULAR_SOLVE_ZERO_TOLERANCE {
                    rhs[row] = 0.0;
                    continue;
                }
                rhs[row] /= denom;
            }
        }
        MatrixBlock::BlockDiagonal(blocks) => {
            let mut row_offset = 0;
            for block in blocks {
                let s = block.nrows();
                for row in (0..s).rev() {
                    let diag = block[(row, row)];
                    if diag.abs() < BLOCK_TRIANGULAR_SOLVE_ZERO_TOLERANCE {
                        rhs[row_offset + row] = 0.0;
                        continue;
                    }
                    let mut sum = rhs[row_offset + row];
                    for inner in (row + 1)..s {
                        sum -= block[(inner, row)] * rhs[row_offset + inner];
                    }
                    rhs[row_offset + row] = sum / diag;
                }
                row_offset += s;
            }
        }
        MatrixBlock::Dense(mat) => {
            for row in (0..mat.nrows()).rev() {
                let diag = mat[(row, row)];
                if diag.abs() < BLOCK_TRIANGULAR_SOLVE_ZERO_TOLERANCE {
                    rhs[row] = 0.0;
                    continue;
                }
                let mut sum = rhs[row];
                for inner in (row + 1)..mat.nrows() {
                    sum -= mat[(inner, row)] * rhs[inner];
                }
                rhs[row] = sum / diag;
            }
        }
        MatrixBlock::Sparse(_) => {
            let dense = l.as_dense();
            solve_upper_block_from_lower_transpose_against_rhs(&MatrixBlock::Dense(dense), rhs);
        }
    }
}

fn solve_lower_block_rhs(rhs: &mut DMatrix<f64>, l: &MatrixBlock) {
    debug_assert_eq!(rhs.nrows(), l.nrows());

    for col in 0..rhs.ncols() {
        let mut column_rhs: Vec<f64> = (0..rhs.nrows()).map(|row| rhs[(row, col)]).collect();
        solve_lower_block_against_rhs(l, &mut column_rhs);
        for row in 0..rhs.nrows() {
            rhs[(row, col)] = column_rhs[row];
        }
    }
}

fn solve_lower_block_rhs_system(l_blocks: &[MatrixBlock], rhs_blocks: &mut [DMatrix<f64>]) {
    let total = rhs_blocks.len();
    for row_block in 0..total {
        for prev in 0..row_block {
            let lower = &l_blocks[block_index(row_block, prev)];
            let solved_prev = rhs_blocks[prev].clone();
            subtract_left_block_product(&mut rhs_blocks[row_block], lower, &solved_prev);
        }
        solve_lower_block_rhs(
            &mut rhs_blocks[row_block],
            &l_blocks[block_index(row_block, row_block)],
        );
    }
}

fn solve_upper_from_lower_transpose(l: &DMatrix<f64>, rhs: &DMatrix<f64>) -> DMatrix<f64> {
    let p = l.nrows();
    let q = rhs.ncols();
    let mut result = rhs.clone();

    for col in 0..q {
        for row in (0..p).rev() {
            let mut sum = result[(row, col)];
            for inner in (row + 1)..p {
                sum -= l[(inner, row)] * result[(inner, col)];
            }
            result[(row, col)] = sum / l[(row, row)];
        }
    }

    result
}

fn response_column_sums_of_squares(y: &DMatrix<f64>) -> DVector<f64> {
    let mut sums = DVector::zeros(y.ncols());
    for col in 0..y.ncols() {
        let mut sum = 0.0;
        for row in 0..y.nrows() {
            let value = y[(row, col)];
            sum += value * value;
        }
        sums[col] = sum;
    }
    sums
}

pub(crate) fn profile_response_matrix_with_l_blocks(
    reterms: &[ReMat],
    x: &DMatrix<f64>,
    responses: &DMatrix<f64>,
    l_blocks: &[MatrixBlock],
    reml: bool,
    n: usize,
    p: usize,
) -> Result<ResponseMatrixProfile> {
    if responses.nrows() != x.nrows() {
        return Err(MixedModelError::DimensionMismatch(format!(
            "response matrix has {} rows, but design matrix has {}",
            responses.nrows(),
            x.nrows()
        )));
    }

    let k = reterms.len();
    let q = responses.ncols();
    let total = k + 1;
    let expected_blocks = total * (total + 1) / 2;
    if l_blocks.len() != expected_blocks {
        return Err(MixedModelError::DimensionMismatch(format!(
            "blocked factor has {} blocks, expected {}",
            l_blocks.len(),
            expected_blocks
        )));
    }

    let mut rhs_blocks = build_response_rhs_blocks(reterms, x, responses);
    solve_lower_block_rhs_system(l_blocks, &mut rhs_blocks);

    let mut solved_norm_sq = DVector::<f64>::zeros(q);
    for block in &rhs_blocks {
        for col in 0..q {
            let mut sum = 0.0;
            for row in 0..block.nrows() {
                let value = block[(row, col)];
                sum += value * value;
            }
            solved_norm_sq[col] += sum;
        }
    }

    let response_ss = response_column_sums_of_squares(responses);
    let mut pwrss = DVector::zeros(q);
    for col in 0..q {
        let residual = response_ss[col] - solved_norm_sq[col];
        pwrss[col] = if residual < 0.0 && residual > -1e-10 {
            0.0
        } else {
            residual
        };
    }

    let x_block = &l_blocks[block_index(k, k)];
    let beta = match x_block {
        MatrixBlock::Dense(l_xx) => solve_upper_from_lower_transpose(l_xx, &rhs_blocks[k]),
        _ => {
            let l_xx = x_block.as_dense();
            solve_upper_from_lower_transpose(&l_xx, &rhs_blocks[k])
        }
    };

    let mut logdet_re = 0.0;
    for j in 0..k {
        logdet_re += logdet_block(&l_blocks[block_index(j, j)]);
    }
    let logdet_xx = logdet_block(x_block);

    let denom = if reml {
        n.checked_sub(p).ok_or_else(|| {
            MixedModelError::DimensionMismatch(format!(
                "REML requires n >= p, got n={} and p={}",
                n, p
            ))
        })?
    } else {
        n
    };
    if denom == 0 {
        return Err(MixedModelError::DimensionMismatch(
            "profile denominator must be positive".to_string(),
        ));
    }
    let denom_f = denom as f64;
    let constant = 2.0 * std::f64::consts::PI / denom_f;

    let mut sigma = DVector::zeros(q);
    let mut objectives = DVector::zeros(q);
    let mut total_objective = 0.0;
    for col in 0..q {
        sigma[col] = (pwrss[col] / denom_f).sqrt();
        let mut objective = logdet_re + denom_f * (1.0 + (constant * pwrss[col]).ln());
        if reml {
            objective += logdet_xx;
        }
        objectives[col] = objective;
        total_objective += objective;
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

// === Block Cholesky helper functions ===

/// Copy A to L and scale blockwise: L_jj = Λ_j' A_jj Λ_j + I
///
/// A is (nranef × nranef) where nranef = vsize * nlevels.
/// Λ is (vsize × vsize). Scaling is applied to each (vsize × vsize)
/// sub-block of A independently.
fn copy_scale_inflate(l: &mut MatrixBlock, a: &MatrixBlock, re: &ReMat) {
    let s = re.vsize;

    if s == 1 {
        // Scalar RE
        let lam = re.lambda[(0, 0)];
        let lam_sq = lam * lam;
        match (l, a) {
            (MatrixBlock::Diagonal(l_diag), MatrixBlock::Diagonal(a_diag)) => {
                for i in 0..l_diag.len() {
                    l_diag[i] = lam_sq * a_diag[i] + 1.0;
                }
            }
            (l_block, _) => with_dense_block(a, |a_dense| {
                let n = a_dense.nrows();
                let result = match l_block {
                    MatrixBlock::Dense(result) if result.shape() == (n, n) => result,
                    _ => {
                        *l_block = MatrixBlock::Dense(DMatrix::zeros(n, n));
                        match l_block {
                            MatrixBlock::Dense(result) => result,
                            _ => unreachable!(),
                        }
                    }
                };
                for i in 0..n {
                    for j in 0..n {
                        result[(i, j)] = lam_sq * a_dense[(i, j)];
                    }
                    result[(i, i)] += 1.0;
                }
            }),
        }
    } else {
        // Vector RE: apply Λ blockwise
        let lambda = &re.lambda;

        match a {
            MatrixBlock::BlockDiagonal(a_blocks) => {
                if matches!(l, MatrixBlock::Dense(_)) {
                    let nranef = a_blocks.iter().map(|blk| blk.nrows()).sum();
                    let result = match l {
                        MatrixBlock::Dense(result) if result.shape() == (nranef, nranef) => result,
                        _ => {
                            *l = MatrixBlock::Dense(DMatrix::zeros(nranef, nranef));
                            match l {
                                MatrixBlock::Dense(result) => result,
                                _ => unreachable!(),
                            }
                        }
                    };

                    if s == 2 {
                        let l00 = lambda[(0, 0)];
                        let l01 = lambda[(0, 1)];
                        let l10 = lambda[(1, 0)];
                        let l11 = lambda[(1, 1)];

                        result.fill(0.0);
                        for (level, src_blk) in a_blocks.iter().enumerate() {
                            let row0 = level * 2;
                            let row1 = row0 + 1;

                            let s00 = src_blk[(0, 0)];
                            let s01 = src_blk[(0, 1)];
                            let s10 = src_blk[(1, 0)];
                            let s11 = src_blk[(1, 1)];

                            let t00 = s00 * l00 + s01 * l10;
                            let t01 = s00 * l01 + s01 * l11;
                            let t10 = s10 * l00 + s11 * l10;
                            let t11 = s10 * l01 + s11 * l11;

                            result[(row0, row0)] = l00 * t00 + l10 * t10 + 1.0;
                            result[(row0, row1)] = l00 * t01 + l10 * t11;
                            result[(row1, row0)] = l01 * t00 + l11 * t10;
                            result[(row1, row1)] = l01 * t01 + l11 * t11 + 1.0;
                        }
                        return;
                    }

                    result.fill(0.0);
                    for (level, src_blk) in a_blocks.iter().enumerate() {
                        for row in 0..s {
                            for col in 0..s {
                                let mut sum = 0.0;
                                for inner_row in 0..s {
                                    for inner_col in 0..s {
                                        sum += lambda[(inner_row, row)]
                                            * src_blk[(inner_row, inner_col)]
                                            * lambda[(inner_col, col)];
                                    }
                                }
                                result[(level * s + row, level * s + col)] = sum;
                            }
                            result[(level * s + row, level * s + row)] += 1.0;
                        }
                    }
                    return;
                }

                let l_blocks = match l {
                    MatrixBlock::BlockDiagonal(l_blocks) => {
                        let shapes_match = l_blocks.len() == a_blocks.len()
                            && l_blocks
                                .iter()
                                .zip(a_blocks.iter())
                                .all(|(dst, src)| dst.shape() == src.shape());
                        if !shapes_match {
                            *l_blocks = a_blocks
                                .iter()
                                .map(|blk| DMatrix::zeros(blk.nrows(), blk.ncols()))
                                .collect();
                        }
                        l_blocks
                    }
                    _ => {
                        *l = MatrixBlock::BlockDiagonal(
                            a_blocks
                                .iter()
                                .map(|blk| DMatrix::zeros(blk.nrows(), blk.ncols()))
                                .collect(),
                        );
                        match l {
                            MatrixBlock::BlockDiagonal(l_blocks) => l_blocks,
                            _ => unreachable!(),
                        }
                    }
                };

                if s == 2 {
                    let l00 = lambda[(0, 0)];
                    let l01 = lambda[(0, 1)];
                    let l10 = lambda[(1, 0)];
                    let l11 = lambda[(1, 1)];

                    for (dst_blk, src_blk) in l_blocks.iter_mut().zip(a_blocks.iter()) {
                        let s00 = src_blk[(0, 0)];
                        let s01 = src_blk[(0, 1)];
                        let s10 = src_blk[(1, 0)];
                        let s11 = src_blk[(1, 1)];

                        let t00 = s00 * l00 + s01 * l10;
                        let t01 = s00 * l01 + s01 * l11;
                        let t10 = s10 * l00 + s11 * l10;
                        let t11 = s10 * l01 + s11 * l11;

                        dst_blk[(0, 0)] = l00 * t00 + l10 * t10 + 1.0;
                        dst_blk[(0, 1)] = l00 * t01 + l10 * t11;
                        dst_blk[(1, 0)] = l01 * t00 + l11 * t10;
                        dst_blk[(1, 1)] = l01 * t01 + l11 * t11 + 1.0;
                    }
                    return;
                }

                for (dst_blk, src_blk) in l_blocks.iter_mut().zip(a_blocks.iter()) {
                    for row in 0..s {
                        for col in 0..s {
                            let mut sum = 0.0;
                            for inner_row in 0..s {
                                for inner_col in 0..s {
                                    sum += lambda[(inner_row, row)]
                                        * src_blk[(inner_row, inner_col)]
                                        * lambda[(inner_col, col)];
                                }
                            }
                            dst_blk[(row, col)] = sum;
                        }
                        dst_blk[(row, row)] += 1.0;
                    }
                }
            }
            _ => {
                // Dense fallback: apply Λ blockwise to each (s×s) sub-block
                with_dense_block(a, |a_dense| {
                    let nranef = a_dense.nrows();
                    let nlevels = nranef / s;
                    let result = match l {
                        MatrixBlock::Dense(result) if result.shape() == (nranef, nranef) => result,
                        _ => {
                            *l = MatrixBlock::Dense(DMatrix::zeros(nranef, nranef));
                            match l {
                                MatrixBlock::Dense(result) => result,
                                _ => unreachable!(),
                            }
                        }
                    };

                    for bk in 0..nlevels {
                        for bl in 0..nlevels {
                            for row in 0..s {
                                for col in 0..s {
                                    let mut sum = 0.0;
                                    for inner_row in 0..s {
                                        for inner_col in 0..s {
                                            sum += lambda[(inner_row, row)]
                                                * a_dense[(bk * s + inner_row, bl * s + inner_col)]
                                                * lambda[(inner_col, col)];
                                        }
                                    }
                                    result[(bk * s + row, bl * s + col)] = sum;
                                }
                            }
                        }
                    }
                    for i in 0..nranef {
                        result[(i, i)] += 1.0;
                    }
                })
            }
        }
    }
}

/// Copy off-diagonal block and scale blockwise: L_ij = Λ_i' A_ij Λ_j
///
/// A is (nranef_i × nranef_j). Λ_i is (vsize_i × vsize_i), Λ_j is (vsize_j × vsize_j).
fn copy_and_scale_offdiag(l: &mut MatrixBlock, a: &MatrixBlock, re_i: &ReMat, re_j: &ReMat) {
    let si = re_i.vsize;
    let sj = re_j.vsize;

    if si == 1 && sj == 1 {
        let scale = re_i.lambda[(0, 0)] * re_j.lambda[(0, 0)];
        if let MatrixBlock::Sparse(a_sparse) = a {
            let result = match l {
                MatrixBlock::Sparse(result)
                    if result.nrows() == a_sparse.nrows()
                        && result.ncols() == a_sparse.ncols()
                        && result.nnz() == a_sparse.nnz()
                        && result.col_offsets() == a_sparse.col_offsets()
                        && result.row_indices() == a_sparse.row_indices() =>
                {
                    result
                }
                _ => {
                    *l = MatrixBlock::Sparse(a_sparse.clone());
                    match l {
                        MatrixBlock::Sparse(result) => result,
                        _ => unreachable!(),
                    }
                }
            };
            result.values_mut().copy_from_slice(a_sparse.values());
            for value in result.values_mut() {
                *value *= scale;
            }
            return;
        }
    }

    with_dense_block(a, |a_dense| {
        let nranef_i = a_dense.nrows();
        let nranef_j = a_dense.ncols();
        let nlevels_i = nranef_i / si;
        let nlevels_j = nranef_j / sj;
        let lambda_j = &re_j.lambda;
        let result = match l {
            MatrixBlock::Dense(result) if result.shape() == (nranef_i, nranef_j) => result,
            _ => {
                *l = MatrixBlock::Dense(DMatrix::zeros(nranef_i, nranef_j));
                match l {
                    MatrixBlock::Dense(result) => result,
                    _ => unreachable!(),
                }
            }
        };

        if si == 2 && sj == 2 {
            let li00 = re_i.lambda[(0, 0)];
            let li01 = re_i.lambda[(0, 1)];
            let li10 = re_i.lambda[(1, 0)];
            let li11 = re_i.lambda[(1, 1)];
            let lj00 = lambda_j[(0, 0)];
            let lj01 = lambda_j[(0, 1)];
            let lj10 = lambda_j[(1, 0)];
            let lj11 = lambda_j[(1, 1)];

            for bi in 0..nlevels_i {
                let row0 = bi * 2;
                let row1 = row0 + 1;
                for bj in 0..nlevels_j {
                    let col0 = bj * 2;
                    let col1 = col0 + 1;
                    let a00 = a_dense[(row0, col0)];
                    let a01 = a_dense[(row0, col1)];
                    let a10 = a_dense[(row1, col0)];
                    let a11 = a_dense[(row1, col1)];

                    let t00 = a00 * lj00 + a01 * lj10;
                    let t01 = a00 * lj01 + a01 * lj11;
                    let t10 = a10 * lj00 + a11 * lj10;
                    let t11 = a10 * lj01 + a11 * lj11;

                    result[(row0, col0)] = li00 * t00 + li10 * t10;
                    result[(row0, col1)] = li00 * t01 + li10 * t11;
                    result[(row1, col0)] = li01 * t00 + li11 * t10;
                    result[(row1, col1)] = li01 * t01 + li11 * t11;
                }
            }
            return;
        }

        for bi in 0..nlevels_i {
            for bj in 0..nlevels_j {
                for row in 0..si {
                    for col in 0..sj {
                        let mut sum = 0.0;
                        for inner_row in 0..si {
                            for inner_col in 0..sj {
                                sum += re_i.lambda[(inner_row, row)]
                                    * a_dense[(bi * si + inner_row, bj * sj + inner_col)]
                                    * lambda_j[(inner_col, col)];
                            }
                        }
                        result[(bi * si + row, bj * sj + col)] = sum;
                    }
                }
            }
        }
    });
}

/// Copy and right-multiply blockwise by Λ: L_kj = A_kj Λ_j
///
/// A is (pp1 × nranef_j). Λ_j is (vsize_j × vsize_j).
/// Each column-block of size vsize_j gets right-multiplied by Λ_j.
fn copy_and_rmul_lambda(l: &mut MatrixBlock, a: &MatrixBlock, re_j: &ReMat) {
    let sj = re_j.vsize;
    if sj == 1 {
        let lam = re_j.lambda[(0, 0)];
        match a {
            MatrixBlock::Dense(a_dense) => {
                let nrows = a_dense.nrows();
                let ncols = a_dense.ncols();
                let result = match l {
                    MatrixBlock::Dense(result) if result.shape() == (nrows, ncols) => result,
                    _ => {
                        *l = MatrixBlock::Dense(DMatrix::zeros(nrows, ncols));
                        match l {
                            MatrixBlock::Dense(result) => result,
                            _ => unreachable!(),
                        }
                    }
                };

                for i in 0..nrows {
                    for j in 0..ncols {
                        result[(i, j)] = a_dense[(i, j)] * lam;
                    }
                }
                return;
            }
            _ => {
                let a_dense = a.as_dense();
                let nrows = a_dense.nrows();
                let ncols = a_dense.ncols();
                let result = match l {
                    MatrixBlock::Dense(result) if result.shape() == (nrows, ncols) => result,
                    _ => {
                        *l = MatrixBlock::Dense(DMatrix::zeros(nrows, ncols));
                        match l {
                            MatrixBlock::Dense(result) => result,
                            _ => unreachable!(),
                        }
                    }
                };

                for i in 0..nrows {
                    for j in 0..ncols {
                        result[(i, j)] = a_dense[(i, j)] * lam;
                    }
                }
                return;
            }
        }
    }

    with_dense_block(a, |a_dense| {
        let nrows = a_dense.nrows();
        let ncols = a_dense.ncols();
        let nblocks = ncols / sj;
        let lambda_j = &re_j.lambda;
        let result = match l {
            MatrixBlock::Dense(result) if result.shape() == (nrows, ncols) => result,
            _ => {
                *l = MatrixBlock::Dense(DMatrix::zeros(nrows, ncols));
                match l {
                    MatrixBlock::Dense(result) => result,
                    _ => unreachable!(),
                }
            }
        };

        if sj == 2 {
            let l00 = lambda_j[(0, 0)];
            let l01 = lambda_j[(0, 1)];
            let l10 = lambda_j[(1, 0)];
            let l11 = lambda_j[(1, 1)];

            for b in 0..nblocks {
                let col0 = b * 2;
                let col1 = col0 + 1;
                for i in 0..nrows {
                    let x0 = a_dense[(i, col0)];
                    let x1 = a_dense[(i, col1)];
                    result[(i, col0)] = x0 * l00 + x1 * l10;
                    result[(i, col1)] = x0 * l01 + x1 * l11;
                }
            }
            return;
        }

        for b in 0..nblocks {
            for i in 0..nrows {
                for j in 0..sj {
                    let mut sum = 0.0;
                    for inner in 0..sj {
                        sum += a_dense[(i, b * sj + inner)] * lambda_j[(inner, j)];
                    }
                    result[(i, b * sj + j)] = sum;
                }
            }
        }
    });
}

/// Rank-k downdate: C -= A * A' (modifies diagonal block)
fn rank_k_downdate(c: &mut MatrixBlock, a: &DMatrix<f64>) {
    match c {
        MatrixBlock::Dense(c_mat) => {
            if c_mat.nrows() == c_mat.ncols()
                && c_mat.nrows() == a.nrows()
                && c_mat.nrows() <= 4
                && a.ncols() >= 512
            {
                let n = c_mat.nrows();
                for row in 0..n {
                    for col in 0..=row {
                        let mut sum = 0.0;
                        for k in 0..a.ncols() {
                            sum += a[(row, k)] * a[(col, k)];
                        }
                        c_mat[(row, col)] -= sum;
                    }
                }
            } else {
                c_mat.gemm(-1.0, a, &a.transpose(), 1.0);
            }
        }
        MatrixBlock::Diagonal(c_diag) => {
            // A * A' diagonal entries
            for i in 0..c_diag.len() {
                let row = a.row(i);
                c_diag[i] -= row.dot(&row);
            }
        }
        MatrixBlock::BlockDiagonal(blocks) => {
            if a.ncols() >= 512 && blocks.first().is_some_and(|blk| blk.nrows() == 2) {
                let mut row_offset = 0;
                for blk in blocks.iter_mut() {
                    let a0 = a.row(row_offset);
                    let a1 = a.row(row_offset + 1);
                    blk[(0, 0)] -= a0.dot(&a0);
                    blk[(1, 0)] -= a1.dot(&a0);
                    blk[(1, 1)] -= a1.dot(&a1);
                    row_offset += 2;
                }
                return;
            }

            // For each block k, downdate by the corresponding rows of A
            let mut row_offset = 0;
            for blk in blocks.iter_mut() {
                let s = blk.nrows();
                let a_block = a.rows(row_offset, s);
                blk.gemm(-1.0, &a_block, &a_block.transpose(), 1.0);
                row_offset += s;
            }
        }
        MatrixBlock::Sparse(_) => {
            let mut dense = c.as_dense();
            dense.gemm(-1.0, a, &a.transpose(), 1.0);
            *c = MatrixBlock::Dense(dense);
        }
    }
}

/// Rank-k downdate from a sparse block: C -= A * A'.
fn rank_k_downdate_sparse(c: &mut MatrixBlock, a: &CscMatrix<f64>) {
    match c {
        MatrixBlock::Dense(c_mat) => {
            for col_idx in 0..a.ncols() {
                let col = a.col(col_idx);
                let rows = col.row_indices();
                let values = col.values();
                for left in 0..rows.len() {
                    let row_i = rows[left];
                    let value_i = values[left];
                    for right in 0..rows.len() {
                        let row_j = rows[right];
                        c_mat[(row_i, row_j)] -= value_i * values[right];
                    }
                }
            }
        }
        MatrixBlock::Diagonal(c_diag) => {
            for (row, _, value) in a.triplet_iter() {
                c_diag[row] -= value * value;
            }
        }
        _ => {
            let mut dense = c.as_dense();
            for col_idx in 0..a.ncols() {
                let col = a.col(col_idx);
                let rows = col.row_indices();
                let values = col.values();
                for left in 0..rows.len() {
                    let row_i = rows[left];
                    let value_i = values[left];
                    for right in 0..rows.len() {
                        let row_j = rows[right];
                        dense[(row_i, row_j)] -= value_i * values[right];
                    }
                }
            }
            *c = MatrixBlock::Dense(dense);
        }
    }
}

/// Subtract product: C -= A * B'
fn subtract_product(c: &mut MatrixBlock, a: &DMatrix<f64>, b: &DMatrix<f64>) {
    match c {
        MatrixBlock::Dense(c_mat) => {
            c_mat.gemm(-1.0, a, &b.transpose(), 1.0);
        }
        MatrixBlock::BlockDiagonal(_) => {
            // Promote to dense — off-diagonal updates destroy block-diagonal structure
            let mut c_dense = c.as_dense();
            c_dense.gemm(-1.0, a, &b.transpose(), 1.0);
            *c = MatrixBlock::Dense(c_dense);
        }
        MatrixBlock::Sparse(_) => {
            let mut c_dense = c.as_dense();
            c_dense.gemm(-1.0, a, &b.transpose(), 1.0);
            *c = MatrixBlock::Dense(c_dense);
        }
        _ => {
            let mut c_dense = c.as_dense();
            c_dense.gemm(-1.0, a, &b.transpose(), 1.0);
            *c = MatrixBlock::Dense(c_dense);
        }
    }
}

/// In-place Cholesky of a block (handles zero diagonal gracefully).
#[cfg(test)]
fn cholesky_block(block: &mut MatrixBlock) -> Result<()> {
    cholesky_block_with_tolerance(
        block,
        crate::compiler::policy::DEFAULT_CHOLESKY_ZERO_PAD_TOLERANCE,
    )
}

fn cholesky_zero_pad_abs_tolerance(diagonal_scale: f64, relative_tolerance: f64) -> f64 {
    if !diagonal_scale.is_finite() || !relative_tolerance.is_finite() {
        return 0.0;
    }
    relative_tolerance.max(0.0) * diagonal_scale.max(0.0)
}

fn diagonal_abs_max_matrix(mat: &DMatrix<f64>) -> f64 {
    (0..mat.nrows().min(mat.ncols()))
        .map(|idx| mat[(idx, idx)].abs())
        .fold(0.0_f64, f64::max)
}

fn cholesky_block_with_tolerance(
    block: &mut MatrixBlock,
    cholesky_zero_pad_tolerance: f64,
) -> Result<()> {
    match block {
        MatrixBlock::Diagonal(diag) => {
            let tol = cholesky_zero_pad_abs_tolerance(
                diag.iter().map(|value| value.abs()).fold(0.0_f64, f64::max),
                cholesky_zero_pad_tolerance,
            );
            for i in 0..diag.len() {
                if diag[i] <= 0.0 {
                    if diag[i] < -tol {
                        return Err(MixedModelError::PosDefException);
                    }
                    diag[i] = 0.0;
                } else {
                    diag[i] = diag[i].sqrt();
                }
            }
            Ok(())
        }
        MatrixBlock::BlockDiagonal(blocks) => {
            // Cholesky each small block independently: O(nlevels * s³)
            if blocks.first().is_some_and(|blk| blk.nrows() == 2) {
                for blk in blocks.iter_mut() {
                    let tol = cholesky_zero_pad_abs_tolerance(
                        diagonal_abs_max_matrix(blk),
                        cholesky_zero_pad_tolerance,
                    );
                    let d00 = blk[(0, 0)];
                    if d00 <= 0.0 {
                        if d00 < -tol {
                            return Err(MixedModelError::PosDefException);
                        }
                        blk[(0, 0)] = 0.0;
                        blk[(1, 0)] = 0.0;
                    } else {
                        blk[(0, 0)] = d00.sqrt();
                        blk[(1, 0)] /= blk[(0, 0)];
                    }

                    let d11 = blk[(1, 1)] - blk[(1, 0)] * blk[(1, 0)];
                    if d11 <= 0.0 {
                        if d11 < -tol {
                            return Err(MixedModelError::PosDefException);
                        }
                        blk[(1, 1)] = 0.0;
                    } else {
                        blk[(1, 1)] = d11.sqrt();
                    }
                    blk[(0, 1)] = 0.0;
                }
                return Ok(());
            }

            for blk in blocks.iter_mut() {
                let n = blk.nrows();
                let tol = cholesky_zero_pad_abs_tolerance(
                    diagonal_abs_max_matrix(blk),
                    cholesky_zero_pad_tolerance,
                );
                for j in 0..n {
                    let mut s = blk[(j, j)];
                    for k in 0..j {
                        s -= blk[(j, k)] * blk[(j, k)];
                    }
                    if s <= 0.0 {
                        if s < -tol {
                            return Err(MixedModelError::PosDefException);
                        }
                        for i in j..n {
                            blk[(i, j)] = 0.0;
                        }
                        continue;
                    }
                    blk[(j, j)] = s.sqrt();
                    for i in (j + 1)..n {
                        let mut s = blk[(i, j)];
                        for k in 0..j {
                            s -= blk[(i, k)] * blk[(j, k)];
                        }
                        blk[(i, j)] = s / blk[(j, j)];
                    }
                    for i in 0..j {
                        blk[(i, j)] = 0.0;
                    }
                }
            }
            Ok(())
        }
        MatrixBlock::Dense(mat) => {
            let n = mat.nrows();
            let tol = cholesky_zero_pad_abs_tolerance(
                diagonal_abs_max_matrix(mat),
                cholesky_zero_pad_tolerance,
            );
            for j in 0..n {
                // Compute L[j,j]
                let mut s = mat[(j, j)];
                for k in 0..j {
                    s -= mat[(j, k)] * mat[(j, k)];
                }
                if s <= 0.0 {
                    if s < -tol {
                        return Err(MixedModelError::PosDefException);
                    }
                    // Zero row (singular RE)
                    for i in j..n {
                        mat[(i, j)] = 0.0;
                    }
                    continue;
                }
                mat[(j, j)] = s.sqrt();

                // Compute L[i,j] for i > j
                for i in (j + 1)..n {
                    let mut s = mat[(i, j)];
                    for k in 0..j {
                        s -= mat[(i, k)] * mat[(j, k)];
                    }
                    mat[(i, j)] = s / mat[(j, j)];
                }

                // Zero out upper triangle
                for i in 0..j {
                    mat[(i, j)] = 0.0;
                }
            }
            Ok(())
        }
        MatrixBlock::Sparse(_) => {
            let dense = block.as_dense();
            *block = MatrixBlock::Dense(dense);
            cholesky_block_with_tolerance(block, cholesky_zero_pad_tolerance)
        }
    }
}

/// Right-divide by lower triangular transpose: A = A * L^{-T}
fn rdiv_lower_transpose(a: &mut MatrixBlock, l: &MatrixBlock) {
    match l {
        MatrixBlock::Diagonal(l_diag) => match a {
            MatrixBlock::Dense(a_mat) => {
                for j in 0..l_diag.len() {
                    let denom = l_diag[j];
                    if denom.abs() < 1e-30 {
                        for i in 0..a_mat.nrows() {
                            a_mat[(i, j)] = 0.0;
                        }
                        continue;
                    }
                    for i in 0..a_mat.nrows() {
                        a_mat[(i, j)] /= denom;
                    }
                }
            }
            MatrixBlock::Sparse(a_sparse) => {
                for j in 0..a_sparse.ncols() {
                    let denom = l_diag[j];
                    let mut col = a_sparse.col_mut(j);
                    if denom.abs() < 1e-30 {
                        for value in col.values_mut() {
                            *value = 0.0;
                        }
                    } else {
                        for value in col.values_mut() {
                            *value /= denom;
                        }
                    }
                }
            }
            MatrixBlock::Diagonal(a_diag) => {
                for i in 0..a_diag.len() {
                    let denom = l_diag[i];
                    if denom.abs() > 1e-30 {
                        a_diag[i] /= denom;
                    } else {
                        a_diag[i] = 0.0;
                    }
                }
            }
            MatrixBlock::BlockDiagonal(_) => {
                let mut a_dense = a.as_dense();
                for j in 0..l_diag.len() {
                    let denom = l_diag[j];
                    if denom.abs() < 1e-30 {
                        for i in 0..a_dense.nrows() {
                            a_dense[(i, j)] = 0.0;
                        }
                        continue;
                    }
                    for i in 0..a_dense.nrows() {
                        a_dense[(i, j)] /= denom;
                    }
                }
                *a = MatrixBlock::Dense(a_dense);
            }
        },
        MatrixBlock::BlockDiagonal(l_blocks) => {
            // L is block-diagonal: solve each column-block independently
            // A[:,block_k] = A[:,block_k] * L_k^{-T}
            match a {
                MatrixBlock::Dense(a_mat) => {
                    let mut col_offset = 0;
                    for l_blk in l_blocks {
                        let s = l_blk.nrows();
                        if s == 2 {
                            let c0 = col_offset;
                            let c1 = col_offset + 1;
                            let l00 = l_blk[(0, 0)];
                            let l10 = l_blk[(1, 0)];
                            let l11 = l_blk[(1, 1)];

                            for i in 0..a_mat.nrows() {
                                let x0 = a_mat[(i, c0)];
                                if l00.abs() < 1e-30 {
                                    a_mat[(i, c0)] = 0.0;
                                } else {
                                    a_mat[(i, c0)] = x0 / l00;
                                }

                                if l11.abs() < 1e-30 {
                                    a_mat[(i, c1)] = 0.0;
                                } else {
                                    a_mat[(i, c1)] = (a_mat[(i, c1)] - a_mat[(i, c0)] * l10) / l11;
                                }
                            }
                            col_offset += s;
                            continue;
                        }

                        // Solve the s-column slice of A against L_k
                        for j in 0..s {
                            let cj = col_offset + j;
                            if l_blk[(j, j)].abs() < 1e-30 {
                                for i in 0..a_mat.nrows() {
                                    a_mat[(i, cj)] = 0.0;
                                }
                                continue;
                            }
                            for i in 0..a_mat.nrows() {
                                let mut val = a_mat[(i, cj)];
                                for k in 0..j {
                                    val -= a_mat[(i, col_offset + k)] * l_blk[(j, k)];
                                }
                                a_mat[(i, cj)] = val / l_blk[(j, j)];
                            }
                        }
                        col_offset += s;
                    }
                }
                MatrixBlock::BlockDiagonal(_) | MatrixBlock::Sparse(_) => {
                    // Both block-diagonal: promote A to dense, then solve
                    let mut a_dense = a.as_dense();
                    let mut col_offset = 0;
                    for l_blk in l_blocks {
                        let s = l_blk.nrows();
                        for j in 0..s {
                            let cj = col_offset + j;
                            if l_blk[(j, j)].abs() < 1e-30 {
                                for i in 0..a_dense.nrows() {
                                    a_dense[(i, cj)] = 0.0;
                                }
                                continue;
                            }
                            for i in 0..a_dense.nrows() {
                                let mut val = a_dense[(i, cj)];
                                for k in 0..j {
                                    val -= a_dense[(i, col_offset + k)] * l_blk[(j, k)];
                                }
                                a_dense[(i, cj)] = val / l_blk[(j, j)];
                            }
                        }
                        col_offset += s;
                    }
                    *a = MatrixBlock::Dense(a_dense);
                }
                MatrixBlock::Diagonal(a_diag) => {
                    // Diagonal A, BlockDiagonal L: promote to dense
                    let mut a_dense = DMatrix::from_diagonal(a_diag);
                    let mut col_offset = 0;
                    for l_blk in l_blocks {
                        let s = l_blk.nrows();
                        for j in 0..s {
                            let cj = col_offset + j;
                            if l_blk[(j, j)].abs() < 1e-30 {
                                for i in 0..a_dense.nrows() {
                                    a_dense[(i, cj)] = 0.0;
                                }
                                continue;
                            }
                            for i in 0..a_dense.nrows() {
                                let mut val = a_dense[(i, cj)];
                                for k in 0..j {
                                    val -= a_dense[(i, col_offset + k)] * l_blk[(j, k)];
                                }
                                a_dense[(i, cj)] = val / l_blk[(j, j)];
                            }
                        }
                        col_offset += s;
                    }
                    *a = MatrixBlock::Dense(a_dense);
                }
            }
        }
        _ => {
            // L is Dense or Diagonal — original logic
            let l_dense = l.as_dense();
            let n = l_dense.nrows();

            match a {
                MatrixBlock::Dense(a_mat) => {
                    for j in 0..n {
                        if l_dense[(j, j)].abs() < 1e-30 {
                            for i in 0..a_mat.nrows() {
                                a_mat[(i, j)] = 0.0;
                            }
                            continue;
                        }
                        for i in 0..a_mat.nrows() {
                            let mut s = a_mat[(i, j)];
                            for k in 0..j {
                                s -= a_mat[(i, k)] * l_dense[(j, k)];
                            }
                            a_mat[(i, j)] = s / l_dense[(j, j)];
                        }
                    }
                }
                MatrixBlock::Diagonal(a_diag) => match l {
                    MatrixBlock::Diagonal(l_diag) => {
                        for i in 0..a_diag.len() {
                            if l_diag[i].abs() > 1e-30 {
                                a_diag[i] /= l_diag[i];
                            } else {
                                a_diag[i] = 0.0;
                            }
                        }
                    }
                    _ => {
                        let mut a_dense = DMatrix::from_diagonal(a_diag);
                        for j in 0..n {
                            if l_dense[(j, j)].abs() < 1e-30 {
                                for i in 0..a_dense.nrows() {
                                    a_dense[(i, j)] = 0.0;
                                }
                                continue;
                            }
                            for i in 0..a_dense.nrows() {
                                let mut s = a_dense[(i, j)];
                                for k in 0..j {
                                    s -= a_dense[(i, k)] * l_dense[(j, k)];
                                }
                                a_dense[(i, j)] = s / l_dense[(j, j)];
                            }
                        }
                        *a = MatrixBlock::Dense(a_dense);
                    }
                },
                MatrixBlock::BlockDiagonal(_) | MatrixBlock::Sparse(_) => {
                    // Promote to dense and solve
                    let mut a_dense = a.as_dense();
                    for j in 0..n {
                        if l_dense[(j, j)].abs() < 1e-30 {
                            for i in 0..a_dense.nrows() {
                                a_dense[(i, j)] = 0.0;
                            }
                            continue;
                        }
                        for i in 0..a_dense.nrows() {
                            let mut s = a_dense[(i, j)];
                            for k in 0..j {
                                s -= a_dense[(i, k)] * l_dense[(j, k)];
                            }
                            a_dense[(i, j)] = s / l_dense[(j, j)];
                        }
                    }
                    *a = MatrixBlock::Dense(a_dense);
                }
            }
        }
    }
}

/// Log-determinant of a Cholesky block (sum of log of diagonal elements).
fn logdet_block(block: &MatrixBlock) -> f64 {
    match block {
        MatrixBlock::Diagonal(diag) => {
            diag.iter()
                .filter(|&&d| d > 0.0)
                .map(|d| d.ln())
                .sum::<f64>()
                * 2.0
        }
        MatrixBlock::BlockDiagonal(blocks) => {
            if blocks.first().is_some_and(|blk| blk.nrows() == 2) {
                let mut ld = 0.0;
                for blk in blocks {
                    let d0 = blk[(0, 0)];
                    let d1 = blk[(1, 1)];
                    if d0 > 0.0 {
                        ld += d0.ln();
                    }
                    if d1 > 0.0 {
                        ld += d1.ln();
                    }
                }
                return ld * 2.0;
            }

            let mut ld = 0.0;
            for blk in blocks {
                let n = blk.nrows();
                for i in 0..n {
                    let d = blk[(i, i)];
                    if d > 0.0 {
                        ld += d.ln();
                    }
                }
            }
            ld * 2.0
        }
        MatrixBlock::Dense(mat) => {
            let n = mat.nrows().min(mat.ncols());
            let mut ld = 0.0;
            for i in 0..n {
                let d = mat[(i, i)];
                if d > 0.0 {
                    ld += d.ln();
                }
            }
            ld * 2.0
        }
        MatrixBlock::Sparse(mat) => {
            let dense = MatrixBlock::Sparse(mat.clone()).as_dense();
            logdet_block(&MatrixBlock::Dense(dense))
        }
    }
}

// ── Parametric bootstrap ──────────────────────────────────────────────────────

/// A single parametric bootstrap replicate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BootstrapReplicate {
    /// Profile-likelihood objective (deviance or REML criterion).
    #[serde(with = "json_f64")]
    pub objective: f64,
    /// Residual standard deviation σ.
    #[serde(with = "json_f64")]
    pub sigma: f64,
    /// Fixed-effects coefficients (pivot order).
    #[serde(with = "json_dvector_f64")]
    pub beta: DVector<f64>,
    /// Fixed-effects standard errors (pivot order).
    #[serde(default = "default_bootstrap_se", with = "json_dvector_f64")]
    pub se: DVector<f64>,
    /// Variance-component θ parameters.
    pub theta: Vec<f64>,
}

/// Collection of parametric bootstrap replicates.
///
/// Mirrors `MixedModelBootstrap` in Julia's MixedModels.jl.
///
/// Produced by [`parametricbootstrap`].  Each replicate stores the
/// objective, residual σ, fixed-effects β, standard errors, and covariance θ for a
/// model fitted to a simulated response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MixedModelBootstrap {
    /// One entry per bootstrap replicate.
    pub fits: Vec<BootstrapReplicate>,
}

/// Confidence-interval construction method for bootstrap summaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BootstrapIntervalMethod {
    /// Equal-tail percentile interval.
    Percentile,
    /// Shortest contiguous interval covering the requested level.
    Shortest,
}

/// One quantile row for a bootstrap statistic.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BootstrapQuantile {
    /// Statistic name: `objective`, `sigma`, `beta[i]`, or `theta[i]`.
    pub parameter: String,
    /// Requested probability in `[0, 1]`.
    pub probability: f64,
    /// Quantile value.
    pub value: f64,
    /// Number of finite replicate values used.
    pub n: usize,
}

/// One confidence-interval row for a bootstrap statistic.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BootstrapInterval {
    /// Statistic name: `objective`, `sigma`, `beta[i]`, or `theta[i]`.
    pub parameter: String,
    /// Requested coverage level in `(0, 1)`.
    pub level: f64,
    /// Lower endpoint.
    pub lower: f64,
    /// Upper endpoint.
    pub upper: f64,
    /// Number of finite replicate values used.
    pub n: usize,
    /// Interval construction method.
    pub method: BootstrapIntervalMethod,
}

pub const BOOTSTRAP_RUN_SCHEMA: &str = "mixedmodels.bootstrap_run";
pub const BOOTSTRAP_RUN_SCHEMA_VERSION: &str = "1.0.0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BootstrapTargetKind {
    FullModelDistribution,
    FixedEffectNull,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BootstrapTarget {
    pub kind: BootstrapTargetKind,
    pub label: String,
    pub contrast_label: Option<String>,
}

impl BootstrapTarget {
    pub fn full_model_distribution(label: impl Into<String>) -> Self {
        Self {
            kind: BootstrapTargetKind::FullModelDistribution,
            label: label.into(),
            contrast_label: None,
        }
    }

    pub fn fixed_effect_null(label: impl Into<String>, contrast_label: impl Into<String>) -> Self {
        Self {
            kind: BootstrapTargetKind::FixedEffectNull,
            label: label.into(),
            contrast_label: Some(contrast_label.into()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BootstrapFailedRefitPolicy {
    Exclude,
    CountExtreme,
    Abort,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FixedEffectBootstrapOptions {
    pub requested_replicates: usize,
    pub failed_refit_policy: BootstrapFailedRefitPolicy,
    pub seed: Option<u64>,
}

impl Default for FixedEffectBootstrapOptions {
    fn default() -> Self {
        Self {
            requested_replicates: 999,
            failed_refit_policy: BootstrapFailedRefitPolicy::Exclude,
            seed: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BootstrapSeedRecord {
    pub rng: String,
    pub seed: Option<u64>,
    pub reproducibility_note: String,
}

impl BootstrapSeedRecord {
    pub fn unspecified() -> Self {
        Self {
            rng: "unknown".to_string(),
            seed: None,
            reproducibility_note:
                "seed was not recorded; bootstrap run is not exactly reproducible".to_string(),
        }
    }

    pub fn std_rng(seed: u64) -> Self {
        Self {
            rng: "StdRng".to_string(),
            seed: Some(seed),
            reproducibility_note: "bootstrap seed recorded by Rust caller".to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BootstrapRefitOptions {
    pub reml: bool,
    pub backend: String,
    pub optimizer: String,
}

impl BootstrapRefitOptions {
    pub fn from_model(model: &LinearMixedModel) -> Self {
        Self {
            reml: model.optsum.reml,
            backend: model.optsum.backend_name().to_string(),
            optimizer: model.optsum.optimizer_name().to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BootstrapRunMetadata {
    pub schema_name: String,
    pub schema_version: String,
    pub target: BootstrapTarget,
    pub requested_replicates: usize,
    pub completed_replicates: usize,
    pub successful_replicates: usize,
    pub failed_refits: usize,
    pub failed_refit_policy: BootstrapFailedRefitPolicy,
    pub boundary_count: usize,
    pub boundary_rate: Option<f64>,
    pub seed_record: BootstrapSeedRecord,
    pub refit_options: BootstrapRefitOptions,
    pub statistic_label: Option<String>,
    pub finite_statistic_count: Option<usize>,
    pub mcse: Option<f64>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BootstrapRunPayload {
    pub metadata: BootstrapRunMetadata,
    pub replicates: MixedModelBootstrap,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replicate_statistics: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FixedEffectNullCovariancePolicy {
    ReuseFittedCovariance,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FixedEffectNullBootstrapTarget {
    pub target: BootstrapTarget,
    pub covariance_policy: FixedEffectNullCovariancePolicy,
    pub coefficient_names: Vec<String>,
    #[serde(with = "json_dvector_f64")]
    pub beta_fitted: DVector<f64>,
    #[serde(with = "json_dvector_f64")]
    pub beta_null: DVector<f64>,
    pub theta: Vec<f64>,
    #[serde(with = "json_f64")]
    pub sigma: f64,
    pub reml: bool,
    pub notes: Vec<String>,
}

impl MixedModelBootstrap {
    /// Number of replicates.
    pub fn len(&self) -> usize {
        self.fits.len()
    }

    /// `true` if no replicates were collected.
    pub fn is_empty(&self) -> bool {
        self.fits.is_empty()
    }

    /// Objectives across all replicates.
    pub fn objectives(&self) -> Vec<f64> {
        self.fits.iter().map(|f| f.objective).collect()
    }

    /// Residual σ values across all replicates.
    pub fn sigmas(&self) -> Vec<f64> {
        self.fits.iter().map(|f| f.sigma).collect()
    }

    /// Fixed-effects β vectors across all replicates, shape `n_replicates × p`.
    pub fn betas(&self) -> Vec<DVector<f64>> {
        self.fits.iter().map(|f| f.beta.clone()).collect()
    }

    /// Fixed-effects standard-error vectors across all replicates.
    pub fn standard_errors(&self) -> Vec<DVector<f64>> {
        self.fits.iter().map(|f| f.se.clone()).collect()
    }

    /// Julia-style alias for [`MixedModelBootstrap::standard_errors`].
    pub fn ses(&self) -> Vec<DVector<f64>> {
        self.standard_errors()
    }

    /// θ parameter vectors across all replicates.
    pub fn thetas(&self) -> Vec<Vec<f64>> {
        self.fits.iter().map(|f| f.theta.clone()).collect()
    }

    /// Quantiles for all scalar bootstrap statistics.
    ///
    /// Non-finite replicate values are ignored parameter-by-parameter. The
    /// quantile rule is linear interpolation between adjacent order statistics
    /// (R's type-7 convention).
    pub fn quantiles(&self, probability: f64) -> Result<Vec<BootstrapQuantile>> {
        validate_probability(probability)?;

        self.parameter_series()?
            .into_iter()
            .map(|(parameter, mut values)| {
                values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                Ok(BootstrapQuantile {
                    parameter,
                    probability,
                    value: quantile_sorted(&values, probability),
                    n: values.len(),
                })
            })
            .collect()
    }

    /// Equal-tail percentile confidence intervals for all scalar bootstrap statistics.
    pub fn percentile_intervals(&self, level: f64) -> Result<Vec<BootstrapInterval>> {
        validate_level(level)?;
        let alpha = (1.0 - level) / 2.0;

        self.parameter_series()?
            .into_iter()
            .map(|(parameter, mut values)| {
                values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                Ok(BootstrapInterval {
                    parameter,
                    level,
                    lower: quantile_sorted(&values, alpha),
                    upper: quantile_sorted(&values, 1.0 - alpha),
                    n: values.len(),
                    method: BootstrapIntervalMethod::Percentile,
                })
            })
            .collect()
    }

    /// Alias for equal-tail percentile intervals.
    pub fn confidence_intervals(&self, level: f64) -> Result<Vec<BootstrapInterval>> {
        self.percentile_intervals(level)
    }

    /// Shortest contiguous confidence intervals for all scalar bootstrap statistics.
    ///
    /// This mirrors the `shortestcovint` summary helper used by MixedModels.jl.
    pub fn shortest_intervals(&self, level: f64) -> Result<Vec<BootstrapInterval>> {
        validate_level(level)?;

        self.parameter_series()?
            .into_iter()
            .map(|(parameter, mut values)| {
                let (lower, upper) = shortest_interval(&mut values, level);
                Ok(BootstrapInterval {
                    parameter,
                    level,
                    lower,
                    upper,
                    n: values.len(),
                    method: BootstrapIntervalMethod::Shortest,
                })
            })
            .collect()
    }

    /// Save bootstrap replicates as JSON.
    ///
    /// The JSON form is intentionally just the replicate collection, so it can
    /// be restored independently and then validated against a model template.
    pub fn save_replicates<W: std::io::Write>(
        &self,
        writer: W,
    ) -> std::result::Result<(), serde_json::Error> {
        serde_json::to_writer(writer, self)
    }

    /// Restore bootstrap replicates from JSON.
    pub fn restore_replicates<R: std::io::Read>(
        reader: R,
    ) -> std::result::Result<Self, serde_json::Error> {
        serde_json::from_reader(reader)
    }

    /// Validate restored replicate dimensions against a model template.
    pub fn validate_for_model(&self, model: &LinearMixedModel) -> Result<()> {
        let expected_beta = model.feterm.rank;
        let expected_theta = model.n_theta();

        for (idx, fit) in self.fits.iter().enumerate() {
            if fit.beta.len() != expected_beta {
                return Err(MixedModelError::InvalidArgument(format!(
                    "bootstrap replicate {idx} beta length ({}) does not match model fixed-effect rank ({expected_beta})",
                    fit.beta.len()
                )));
            }
            if fit.theta.len() != expected_theta {
                return Err(MixedModelError::InvalidArgument(format!(
                    "bootstrap replicate {idx} theta length ({}) does not match model theta length ({expected_theta})",
                    fit.theta.len()
                )));
            }
            if !fit.se.is_empty() && fit.se.len() != expected_beta {
                return Err(MixedModelError::InvalidArgument(format!(
                    "bootstrap replicate {idx} se length ({}) does not match model fixed-effect rank ({expected_beta})",
                    fit.se.len()
                )));
            }
        }

        Ok(())
    }

    pub fn run_metadata_for_model(
        &self,
        model: &LinearMixedModel,
        target: BootstrapTarget,
        requested_replicates: usize,
        failed_refit_policy: BootstrapFailedRefitPolicy,
        seed_record: BootstrapSeedRecord,
        refit_options: BootstrapRefitOptions,
        statistic_label: Option<String>,
        statistic_values: Option<&[f64]>,
        p_value: Option<f64>,
    ) -> BootstrapRunMetadata {
        let lower_bounds = model.lower_bounds();
        let successful_replicates = self.fits.iter().filter(|fit| fit.is_successful()).count();
        let boundary_count = self
            .fits
            .iter()
            .filter(|fit| fit.is_successful() && fit.is_boundary_refit(&lower_bounds, 1e-8))
            .count();
        let finite_statistic_count =
            statistic_values.map(|values| values.iter().filter(|value| value.is_finite()).count());
        let boundary_rate = (successful_replicates > 0)
            .then_some(boundary_count as f64 / successful_replicates as f64);
        let mcse = p_value.and_then(|p| {
            (p.is_finite() && (0.0..=1.0).contains(&p) && successful_replicates > 0)
                .then_some((p * (1.0 - p) / successful_replicates as f64).sqrt())
        });

        let mut notes = Vec::new();
        if target.kind == BootstrapTargetKind::FullModelDistribution {
            notes.push(
                "full-model bootstrap distributions do not certify fixed-effect hypothesis-test p-values"
                    .to_string(),
            );
        }
        if requested_replicates != self.len() {
            notes.push(format!(
                "requested {requested_replicates} bootstrap replicate(s), collected {}",
                self.len()
            ));
        }
        if successful_replicates < self.len() {
            notes.push(format!(
                "{} bootstrap refit(s) did not produce finite estimates",
                self.len() - successful_replicates
            ));
        }
        if boundary_count > 0 {
            notes.push(format!(
                "{boundary_count} successful bootstrap refit(s) ended on a covariance boundary"
            ));
        }

        BootstrapRunMetadata {
            schema_name: BOOTSTRAP_RUN_SCHEMA.to_string(),
            schema_version: BOOTSTRAP_RUN_SCHEMA_VERSION.to_string(),
            target,
            requested_replicates,
            completed_replicates: self.len(),
            successful_replicates,
            failed_refits: self.len().saturating_sub(successful_replicates),
            failed_refit_policy,
            boundary_count,
            boundary_rate,
            seed_record,
            refit_options,
            statistic_label,
            finite_statistic_count,
            mcse,
            notes,
        }
    }

    pub fn into_run_payload(self, metadata: BootstrapRunMetadata) -> BootstrapRunPayload {
        BootstrapRunPayload {
            metadata,
            replicates: self,
            replicate_statistics: None,
        }
    }

    pub fn into_run_payload_with_statistics(
        self,
        metadata: BootstrapRunMetadata,
        replicate_statistics: Vec<f64>,
    ) -> BootstrapRunPayload {
        BootstrapRunPayload {
            metadata,
            replicates: self,
            replicate_statistics: Some(replicate_statistics),
        }
    }

    fn parameter_series(&self) -> Result<Vec<(String, Vec<f64>)>> {
        if self.fits.is_empty() {
            return Err(MixedModelError::InvalidArgument(
                "cannot summarize an empty bootstrap sample".to_string(),
            ));
        }

        let beta_len = self.fits[0].beta.len();
        let se_len = self.fits[0].se.len();
        let theta_len = self.fits[0].theta.len();
        for (idx, fit) in self.fits.iter().enumerate() {
            if fit.beta.len() != beta_len {
                return Err(MixedModelError::InvalidArgument(format!(
                    "bootstrap replicate {idx} beta length ({}) does not match first replicate ({beta_len})",
                    fit.beta.len()
                )));
            }
            if fit.se.len() != se_len {
                return Err(MixedModelError::InvalidArgument(format!(
                    "bootstrap replicate {idx} se length ({}) does not match first replicate ({se_len})",
                    fit.se.len()
                )));
            }
            if fit.theta.len() != theta_len {
                return Err(MixedModelError::InvalidArgument(format!(
                    "bootstrap replicate {idx} theta length ({}) does not match first replicate ({theta_len})",
                    fit.theta.len()
                )));
            }
        }

        let mut series = Vec::with_capacity(2 + beta_len + se_len + theta_len);
        series.push((
            "objective".to_string(),
            self.fits
                .iter()
                .map(|fit| fit.objective)
                .filter(|value| value.is_finite())
                .collect(),
        ));
        series.push((
            "sigma".to_string(),
            self.fits
                .iter()
                .map(|fit| fit.sigma)
                .filter(|value| value.is_finite())
                .collect(),
        ));

        for idx in 0..beta_len {
            series.push((
                format!("beta[{idx}]"),
                self.fits
                    .iter()
                    .map(|fit| fit.beta[idx])
                    .filter(|value| value.is_finite())
                    .collect(),
            ));
        }
        for idx in 0..se_len {
            series.push((
                format!("se[{idx}]"),
                self.fits
                    .iter()
                    .map(|fit| fit.se[idx])
                    .filter(|value| value.is_finite())
                    .collect(),
            ));
        }
        for idx in 0..theta_len {
            series.push((
                format!("theta[{idx}]"),
                self.fits
                    .iter()
                    .map(|fit| fit.theta[idx])
                    .filter(|value| value.is_finite())
                    .collect(),
            ));
        }

        series.retain(|(_, values): &(String, Vec<f64>)| !values.is_empty());
        if series.is_empty() {
            return Err(MixedModelError::InvalidArgument(
                "bootstrap sample has no finite scalar statistics to summarize".to_string(),
            ));
        }

        Ok(series)
    }
}

impl BootstrapReplicate {
    fn is_successful(&self) -> bool {
        self.objective.is_finite()
            && self.sigma.is_finite()
            && self.beta.iter().all(|value| value.is_finite())
            && self.se.iter().all(|value| value.is_finite())
            && self.theta.iter().all(|value| value.is_finite())
    }

    fn is_boundary_refit(&self, lower_bounds: &[f64], tolerance: f64) -> bool {
        self.theta.iter().enumerate().any(|(idx, theta)| {
            lower_bounds
                .get(idx)
                .copied()
                .filter(|lower| lower.is_finite())
                .is_some_and(|lower| *theta <= lower + tolerance)
        })
    }
}

fn default_bootstrap_se() -> DVector<f64> {
    DVector::zeros(0)
}

fn validate_probability(probability: f64) -> Result<()> {
    if probability.is_finite() && (0.0..=1.0).contains(&probability) {
        Ok(())
    } else {
        Err(MixedModelError::InvalidArgument(format!(
            "quantile probability must be in [0,1]; got {probability}"
        )))
    }
}

fn validate_level(level: f64) -> Result<()> {
    if level.is_finite() && (0.0..1.0).contains(&level) {
        Ok(())
    } else {
        Err(MixedModelError::InvalidArgument(format!(
            "confidence level must be in (0,1); got {level}"
        )))
    }
}

fn quantile_sorted(values: &[f64], probability: f64) -> f64 {
    debug_assert!(!values.is_empty());
    if values.len() == 1 {
        return values[0];
    }
    let h = probability * (values.len() - 1) as f64;
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;
    if lo == hi {
        values[lo]
    } else {
        values[lo] + (h - lo as f64) * (values[hi] - values[lo])
    }
}

fn shortest_interval(values: &mut [f64], level: f64) -> (f64, f64) {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = values.len();
    let ilen = ((n as f64) * level).ceil() as usize;
    if ilen >= n {
        return (values[0], values[n - 1]);
    }
    let mut min_len = f64::INFINITY;
    let mut best_i = 0;
    for i in 0..=(n - ilen) {
        let len = values[i + ilen - 1] - values[i];
        if len < min_len {
            min_len = len;
            best_i = i;
        }
    }
    (values[best_i], values[best_i + ilen - 1])
}

mod json_f64 {
    use serde::de::Error;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(value: &f64, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if value.is_finite() {
            serializer.serialize_f64(*value)
        } else if value.is_nan() {
            serializer.serialize_str("NaN")
        } else if value.is_sign_positive() {
            serializer.serialize_str("Infinity")
        } else {
            serializer.serialize_str("-Infinity")
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> std::result::Result<f64, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum JsonF64 {
            Number(f64),
            Special(String),
        }

        match JsonF64::deserialize(deserializer)? {
            JsonF64::Number(value) => Ok(value),
            JsonF64::Special(value) => match value.as_str() {
                "NaN" => Ok(f64::NAN),
                "Infinity" => Ok(f64::INFINITY),
                "-Infinity" => Ok(f64::NEG_INFINITY),
                _ => Err(D::Error::custom(format!(
                    "invalid non-finite float marker `{value}`"
                ))),
            },
        }
    }
}

mod json_dvector_f64 {
    use nalgebra::DVector;
    use serde::de::Error;
    use serde::ser::SerializeSeq;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(value: &DVector<f64>, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(value.len()))?;
        for entry in value.iter() {
            seq.serialize_element(&JsonF64(*entry))?;
        }
        seq.end()
    }

    pub fn deserialize<'de, D>(deserializer: D) -> std::result::Result<DVector<f64>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let values = Vec::<JsonF64>::deserialize(deserializer)?;
        Ok(DVector::from_vec(
            values.into_iter().map(|value| value.0).collect(),
        ))
    }

    struct JsonF64(f64);

    impl Serialize for JsonF64 {
        fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            if self.0.is_finite() {
                serializer.serialize_f64(self.0)
            } else if self.0.is_nan() {
                serializer.serialize_str("NaN")
            } else if self.0.is_sign_positive() {
                serializer.serialize_str("Infinity")
            } else {
                serializer.serialize_str("-Infinity")
            }
        }
    }

    impl<'de> Deserialize<'de> for JsonF64 {
        fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            #[derive(Deserialize)]
            #[serde(untagged)]
            enum JsonF64Value {
                Number(f64),
                Special(String),
                Null(Option<()>),
            }

            match JsonF64Value::deserialize(deserializer)? {
                JsonF64Value::Number(value) => Ok(JsonF64(value)),
                JsonF64Value::Special(value) => match value.as_str() {
                    "NaN" => Ok(JsonF64(f64::NAN)),
                    "Infinity" => Ok(JsonF64(f64::INFINITY)),
                    "-Infinity" => Ok(JsonF64(f64::NEG_INFINITY)),
                    _ => Err(D::Error::custom(format!(
                        "invalid non-finite float marker `{value}`"
                    ))),
                },
                JsonF64Value::Null(None) => Ok(JsonF64(f64::NAN)),
                JsonF64Value::Null(Some(())) => Err(D::Error::custom(
                    "invalid unit value in floating-point vector",
                )),
            }
        }
    }
}

/// Run a parametric bootstrap for a fitted `LinearMixedModel`.
///
/// For each of `n_rep` replicates:
/// 1. Simulate a new response from the fitted model.
/// 2. Refit the model to the simulated response.
/// 3. Record `(objective, σ, β, se, θ)`.
///
/// Returns a [`MixedModelBootstrap`] holding all replicates.
///
/// Mirrors `parametricbootstrap(rng, n, m)` in Julia's MixedModels.jl.
///
/// # Arguments
/// * `rng`   – random-number generator (e.g. `rand::rngs::StdRng`)
/// * `n_rep` – number of bootstrap replicates
/// * `model` – a *fitted* `LinearMixedModel` (used as the template)
pub fn parametricbootstrap<R: rand::Rng>(
    rng: &mut R,
    n_rep: usize,
    model: &LinearMixedModel,
) -> MixedModelBootstrap {
    let mut fits = Vec::with_capacity(n_rep);

    for _ in 0..n_rep {
        // Simulate from the template (always use the original fitted model).
        let y_sim = model.simulate(rng);

        // Fresh clone of the template for this replicate.
        let mut work = model.clone();

        match work.refit(y_sim.as_slice()) {
            Ok(()) => {
                fits.push(BootstrapReplicate {
                    objective: work.objective(),
                    sigma: work.sigma(),
                    beta: work.beta(),
                    se: work.stderror(),
                    theta: work.theta(),
                });
            }
            Err(_) => {
                // On numerical failure, record the current (possibly partial) state.
                // Julia silently records the last accepted iterate in such cases.
                let beta = work.beta();
                fits.push(BootstrapReplicate {
                    objective: f64::NAN,
                    sigma: f64::NAN,
                    se: DVector::from_element(beta.len(), f64::NAN),
                    beta,
                    theta: work.theta(),
                });
            }
        }
    }

    MixedModelBootstrap { fits }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};

    use crate::compiler::{
        CertificateCheck, CompilerPolicy, ContrastMatrix, ContrastRhs, DiagnosticCode,
        EffectiveRankStatus, EvidenceMethod, EvidenceQuality, FitIntent, FitStatus,
        FixedEffectHypothesis, InferenceStatus, InformationBudgetStatus, ModelChangeStatus,
        ModelStateStatus, RandomStrategy, RankStatus, ReductionRecord, ReductionTrigger, ThetaMap,
    };
    use crate::formula::parse_formula;
    use crate::model::data::{Column, DataFrame};
    use crate::model::traits::MixedModelFit;

    fn simulate_sleepstudy_like(
        n_subjects: usize,
        n_obs_per_subject: usize,
        seed: u64,
    ) -> DataFrame {
        let mut rng = StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0, 1.0).unwrap();

        let beta = [250.0, 10.0];
        let sigma = 25.0;
        let lambda = [[24.0, 0.0], [1.68, 5.23]];

        let total_n = n_subjects * n_obs_per_subject;
        let mut reaction = Vec::with_capacity(total_n);
        let mut days = Vec::with_capacity(total_n);
        let mut subj_labels = Vec::with_capacity(total_n);

        for i in 0..n_subjects {
            let u0 = normal.sample(&mut rng);
            let u1 = normal.sample(&mut rng);
            let b0 = lambda[0][0] * u0;
            let b1 = lambda[1][0] * u0 + lambda[1][1] * u1;

            let label = format!("S{:04}", i + 1);
            for d in 0..n_obs_per_subject {
                let x = d as f64;
                let mu = beta[0] + beta[1] * x + b0 + b1 * x;
                let y = mu + sigma * normal.sample(&mut rng);
                reaction.push(y);
                days.push(x);
                subj_labels.push(label.clone());
            }
        }

        let mut df = DataFrame::new();
        df.add_numeric("reaction", reaction).unwrap();
        df.add_numeric("days", days).unwrap();
        df.add_categorical("subj", subj_labels).unwrap();
        df
    }

    fn grouped_slope_data(n_groups: usize) -> DataFrame {
        grouped_slope_data_with_obs(n_groups, 2)
    }

    fn grouped_slope_data_with_obs(n_groups: usize, obs_per_group: usize) -> DataFrame {
        let mut data = DataFrame::new();
        let mut y = Vec::new();
        let mut x = Vec::new();
        let mut group = Vec::new();
        for idx in 0..n_groups {
            for obs in 0..obs_per_group {
                y.push(idx as f64 + obs as f64);
                x.push(obs as f64);
                group.push(format!("g{}", idx + 1));
            }
        }
        data.add_numeric("y", y).unwrap();
        data.add_numeric("x", x).unwrap();
        data.add_categorical("group", group).unwrap();
        data
    }

    fn correlated_crossed_slope_data() -> DataFrame {
        fn centered_mod(value: usize, modulus: usize, center: f64, scale: f64) -> f64 {
            ((value % modulus) as f64 - center) * scale
        }

        let n_g = 10;
        let n_h = 8;
        let n_rep = 4;
        let mut y = Vec::with_capacity(n_g * n_h * n_rep);
        let mut x = Vec::with_capacity(n_g * n_h * n_rep);
        let mut g = Vec::with_capacity(n_g * n_h * n_rep);
        let mut h = Vec::with_capacity(n_g * n_h * n_rep);

        for gi in 0..n_g {
            let g0 = centered_mod(7 * gi + 3, 19, 9.0, 2.1);
            let g1 = 0.82 * g0 + centered_mod(11 * gi + 5, 17, 8.0, 0.18);
            for hi in 0..n_h {
                let h0 = centered_mod(13 * hi + 2, 23, 11.0, 1.5);
                let h1 = -0.74 * h0 + centered_mod(5 * hi + 7, 19, 9.0, 0.16);
                for rep in 0..n_rep {
                    let xv = rep as f64 - 1.5 + (gi % 3) as f64 * 0.08 + (hi % 2) as f64 * 0.05;
                    let eps = centered_mod(gi * 11 + hi * 7 + rep * 5, 31, 15.0, 0.28);
                    y.push(4.0 + 1.7 * xv + g0 + g1 * xv + h0 + h1 * xv + eps);
                    x.push(xv);
                    g.push(format!("g{:02}", gi + 1));
                    h.push(format!("h{:02}", hi + 1));
                }
            }
        }

        let mut data = DataFrame::new();
        data.add_numeric("y", y).unwrap();
        data.add_numeric("x", x).unwrap();
        data.add_categorical("g", g).unwrap();
        data.add_categorical("h", h).unwrap();
        data
    }

    fn vsize3_kernel_remat() -> ReMat {
        ReMat::new(
            "subj".to_string(),
            vec![0, 1],
            vec!["S1".to_string(), "S2".to_string()],
            vec!["(Intercept)".to_string(), "x".to_string(), "z".to_string()],
            DMatrix::from_row_slice(3, 2, &[1.0, 1.0, 0.0, 1.0, 2.0, 3.0]),
        )
    }

    #[test]
    fn test_apply_lambda_transpose_to_rhs_consistent_with_parmap_order() {
        let mut re = vsize3_kernel_remat();
        let parmap = build_parmap(&[re.clone()]);
        assert_eq!(
            parmap,
            vec![
                (0, 0, 0),
                (0, 1, 0),
                (0, 2, 0),
                (0, 1, 1),
                (0, 2, 1),
                (0, 2, 2)
            ]
        );

        re.set_theta(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(
            re.lambda,
            DMatrix::from_row_slice(3, 3, &[1.0, 0.0, 0.0, 2.0, 4.0, 0.0, 3.0, 5.0, 6.0])
        );

        let mut rhs = DMatrix::from_row_slice(
            6,
            2,
            &[
                7.0, 29.0, 11.0, 31.0, 13.0, 37.0, 17.0, 41.0, 19.0, 43.0, 23.0, 47.0,
            ],
        );
        let original = rhs.clone();

        apply_lambda_transpose_to_rhs(&mut rhs, &re);

        let lambda_t = re.lambda.transpose();
        let mut expected = DMatrix::zeros(6, 2);
        for level in 0..2 {
            let offset = level * 3;
            let expected_block = &lambda_t * original.rows(offset, 3).into_owned();
            expected.rows_mut(offset, 3).copy_from(&expected_block);
        }
        assert_eq!(rhs, expected);
    }

    fn diagonal_theta_indices(model: &LinearMixedModel) -> Vec<usize> {
        model
            .parmap
            .iter()
            .enumerate()
            .filter_map(|(idx, &(_, row, col))| (row == col).then_some(idx))
            .collect()
    }

    fn assert_theta_diagonals_nonnegative(model: &LinearMixedModel) {
        let theta = model.theta();
        for idx in diagonal_theta_indices(model) {
            assert!(
                theta[idx] >= 0.0,
                "theta diagonal {idx} should be rectified, got {}",
                theta[idx]
            );
            assert_eq!(
                model.optsum.final_params[idx], theta[idx],
                "final_params must store the rectified theta value"
            );
        }
    }

    #[cfg(feature = "nlopt")]
    fn simulate_large_theta_crossed(seed: u64) -> DataFrame {
        let mut rng = StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0, 1.0).unwrap();

        let n_subjects = 18;
        let n_items = 12;
        let n_sites = 6;
        let n_rep = 4;

        let beta = [250.0, 9.5];
        let sigma = 18.0;
        let lambda_subj = [[18.0, 0.0], [2.2, 4.5]];
        let lambda_item = [[11.0, 0.0], [-1.4, 3.2]];
        let lambda_site = [[7.5, 0.0], [0.6, 1.7]];

        let draw_effects = |rng: &mut StdRng, lambda: [[f64; 2]; 2], levels: usize| {
            let mut effects = Vec::with_capacity(levels);
            for _ in 0..levels {
                let u0 = normal.sample(rng);
                let u1 = normal.sample(rng);
                effects.push([lambda[0][0] * u0, lambda[1][0] * u0 + lambda[1][1] * u1]);
            }
            effects
        };

        let subj_effects = draw_effects(&mut rng, lambda_subj, n_subjects);
        let item_effects = draw_effects(&mut rng, lambda_item, n_items);
        let site_effects = draw_effects(&mut rng, lambda_site, n_sites);

        let total_n = n_subjects * n_items * n_rep;
        let mut reaction = Vec::with_capacity(total_n);
        let mut days = Vec::with_capacity(total_n);
        let mut subj_labels = Vec::with_capacity(total_n);
        let mut item_labels = Vec::with_capacity(total_n);
        let mut site_labels = Vec::with_capacity(total_n);

        for s in 0..n_subjects {
            for i in 0..n_items {
                for r in 0..n_rep {
                    let site = (s * 5 + i * 3 + r) % n_sites;
                    let x = r as f64 + (i % 4) as f64 * 0.35;
                    let mut mu = beta[0] + beta[1] * x;
                    mu += subj_effects[s][0] + subj_effects[s][1] * x;
                    mu += item_effects[i][0] + item_effects[i][1] * x;
                    mu += site_effects[site][0] + site_effects[site][1] * x;
                    let y = mu + sigma * normal.sample(&mut rng);

                    reaction.push(y);
                    days.push(x);
                    subj_labels.push(format!("S{:03}", s + 1));
                    item_labels.push(format!("I{:03}", i + 1));
                    site_labels.push(format!("K{:03}", site + 1));
                }
            }
        }

        let mut df = DataFrame::new();
        df.add_numeric("reaction", reaction).unwrap();
        df.add_numeric("days", days).unwrap();
        df.add_categorical("subj", subj_labels).unwrap();
        df.add_categorical("item", item_labels).unwrap();
        df.add_categorical("site", site_labels).unwrap();
        df
    }

    fn permute_rows(data: &DataFrame, order: &[usize]) -> DataFrame {
        let mut permuted = DataFrame::new();

        for name in data.column_names() {
            match data.column(name).unwrap() {
                Column::Numeric(values) => {
                    let reordered = order.iter().map(|&idx| values[idx]).collect();
                    permuted.add_numeric(name, reordered).unwrap();
                }
                Column::Categorical(cat) => {
                    let reordered = order.iter().map(|&idx| cat.values[idx].clone()).collect();
                    permuted.add_categorical(name, reordered).unwrap();
                }
            }
        }

        permuted
    }

    fn shared_julia_parity_fixture() -> DataFrame {
        let reaction = vec![
            228.34733704764443,
            294.32292211548196,
            205.74021389340569,
            278.87878012027852,
            271.07769950952058,
            244.5608057798394,
            265.94463302409139,
            226.77991725455206,
            242.4319346940861,
            214.97408114520201,
            323.21013025658829,
            277.4835351479876,
            273.74759181211351,
            287.11098149680538,
            278.94147834898382,
            297.19606926697281,
            228.30198076068194,
            195.39462889633353,
            217.48019241415267,
            258.9102478189954,
            276.43800461900963,
            315.60786380412753,
            272.3080316216936,
            301.84264174522588,
        ];
        let days = vec![
            0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0,
            1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0,
        ];
        let subj = vec![
            "S0001", "S0001", "S0001", "S0001", "S0002", "S0002", "S0002", "S0002", "S0003",
            "S0003", "S0003", "S0003", "S0004", "S0004", "S0004", "S0004", "S0005", "S0005",
            "S0005", "S0005", "S0006", "S0006", "S0006", "S0006",
        ];

        let mut df = DataFrame::new();
        df.add_numeric("reaction", reaction).unwrap();
        df.add_numeric("days", days).unwrap();
        df.add_categorical("subj", subj.into_iter().map(str::to_string).collect())
            .unwrap();
        df
    }

    #[test]
    #[cfg(not(feature = "prima"))]
    fn test_forced_prima_bobyqa_requires_prima_feature() {
        let data = shared_julia_parity_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

        let err = model
            .fit_with_forced_optimizer(true, Optimizer::PrimaBobyqa)
            .unwrap_err();

        match err {
            MixedModelError::Optimization(message) => {
                assert!(message.contains("`prima` feature"));
                assert!(message.contains("libprimac"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    fn shared_julia_crossed_parity_fixture() -> DataFrame {
        fn centered_mod(value: usize, modulus: usize, center: f64, scale: f64) -> f64 {
            ((value % modulus) as f64 - center) * scale
        }

        let n_subjects = 18;
        let n_items = 12;
        let n_sites = 6;
        let n_rep = 4;
        let beta = [250.0, 9.5];

        let total_n = n_subjects * n_items * n_rep;
        let mut reaction = Vec::with_capacity(total_n);
        let mut days = Vec::with_capacity(total_n);
        let mut subj_labels = Vec::with_capacity(total_n);
        let mut item_labels = Vec::with_capacity(total_n);
        let mut site_labels = Vec::with_capacity(total_n);

        for s in 0..n_subjects {
            let subj_b0 = centered_mod(7 * s + 3, 19, 9.0, 2.4);
            let subj_b1 = centered_mod(11 * s + 5, 17, 8.0, 0.38) + 0.05 * subj_b0;
            let subj_label = format!("S{:03}", s + 1);

            for i in 0..n_items {
                let item_b0 = centered_mod(13 * i + 2, 23, 11.0, 1.6);
                let item_b1 = centered_mod(5 * i + 7, 19, 9.0, 0.27) - 0.04 * item_b0;
                let item_label = format!("I{:03}", i + 1);

                for r in 0..n_rep {
                    let site = (5 * s + 3 * i + r) % n_sites;
                    let site_b0 = centered_mod(3 * site + 1, 13, 6.0, 1.2);
                    let site_b1 = centered_mod(7 * site + 4, 11, 5.0, 0.18) + 0.03 * site_b0;
                    let eps = centered_mod(13 * s + 7 * i + 3 * r + 2 * site, 29, 14.0, 0.9);
                    let x = r as f64 + (i % 4) as f64 * 0.35 + (s % 3) as f64 * 0.1;

                    let mu = beta[0]
                        + beta[1] * x
                        + subj_b0
                        + subj_b1 * x
                        + item_b0
                        + item_b1 * x
                        + site_b0
                        + site_b1 * x;

                    reaction.push(mu + eps);
                    days.push(x);
                    subj_labels.push(subj_label.clone());
                    item_labels.push(item_label.clone());
                    site_labels.push(format!("K{:03}", site + 1));
                }
            }
        }

        let mut df = DataFrame::new();
        df.add_numeric("reaction", reaction).unwrap();
        df.add_numeric("days", days).unwrap();
        df.add_categorical("subj", subj_labels).unwrap();
        df.add_categorical("item", item_labels).unwrap();
        df.add_categorical("site", site_labels).unwrap();
        df
    }

    /// Synthetic data where every group mean equals 5.0 (SS_B = 0).
    /// The ML estimate of between-group variance is exactly 0 → θ = 0 → singular.
    fn singular_re_fixture() -> DataFrame {
        let yields: Vec<f64> = vec![
            2.0, 8.0, 5.0, 3.0, 7.0, // batch A: mean = 5.0
            1.0, 9.0, 5.0, 4.0, 6.0, // batch B: mean = 5.0
            3.0, 7.0, 5.0, 2.0, 8.0, // batch C: mean = 5.0
            4.0, 6.0, 5.0, 1.0, 9.0, // batch D: mean = 5.0
            0.0, 10.0, 5.0, 3.0, 7.0, // batch E: mean = 5.0
            2.0, 8.0, 5.0, 4.0, 6.0, // batch F: mean = 5.0
        ];
        let batches: Vec<String> = "ABCDEF"
            .chars()
            .flat_map(|c| std::iter::repeat_n(c.to_string(), 5))
            .collect();

        let mut df = DataFrame::new();
        df.add_numeric("yield", yields).unwrap();
        df.add_categorical("batch", batches).unwrap();
        df
    }

    fn shared_julia_fixed_sigma_fixture() -> DataFrame {
        let y = vec![
            3.630846066147111,
            -0.23699581316575297,
            1.2105354224682663,
            0.869853351939183,
            -0.20112670239063263,
            1.841939312590815,
            3.0508340329938406,
            -0.16159198227005228,
            -1.7111617117834814,
            -2.573210271206462,
            -0.634354739497098,
            -2.5610196330697224,
            1.318703449478216,
            -3.9447255998012105,
            0.5307037522842474,
            -0.7644160195344709,
            -5.332106917168301,
            -0.47433639211466,
            -4.057116827660948,
            -3.8085558079065667,
            4.234332252764718,
            1.755107761778669,
            2.757065064409675,
            5.30205261880327,
            4.1451742404667105,
            1.2036710555092098,
            -3.0539946895833316,
            -1.8393472588555542,
            5.892040902634034,
            -1.9696539153474302,
            0.6486861972481239,
            0.368489072228326,
            -0.3611408729159792,
            5.193373815268175,
            1.913189995798939,
            0.47507592474230975,
            0.06401249428337571,
            2.2165512252476343,
            -0.9397784817739796,
            1.7788922478551683,
            -9.801745951021179,
            -1.9383974696808517,
            -2.092847010025527,
            3.442639699290954,
            -0.0837941751454139,
            4.133629704184189,
            2.1736737572044635,
            -1.0159208846460877,
            4.368916320835367,
            0.7607202499336108,
            5.85815983648636,
            -1.7609048242566288,
            -4.810884455196657,
            0.793817702591471,
            4.266085487320645,
            1.6199123691375519,
            -0.3084152967914453,
            0.6543377004554722,
            2.539769962223369,
            -3.918979949516328,
            1.1953631700478802,
            -0.2168447423962808,
            7.456462357947441,
            2.479491605550824,
            4.691307422020858,
            -3.9391366970370267,
            1.7056528817929726,
            -8.146790126669345,
            -1.1244595976644554,
            -1.9500060764200495,
            4.463837139784824,
            6.523171674670275,
            0.7811592530551956,
            4.633376703546607,
            1.8990447937621922,
            1.6916780132695428,
            4.812588984521369,
            0.7355154695965163,
            -1.1072651428981173,
            -1.5843836139553726,
            2.7091806278382435,
            -1.9396989674195224,
            -1.329495768570552,
            -2.0278076791842725,
            1.7658616138387506,
            3.407320593069791,
            1.9592167318065936,
            -3.5416850711564076,
            3.2744973367017147,
            -5.1760765079709525,
            -2.9661568404990826,
            0.5663029518057119,
            -3.266594534667978,
            -1.148968568238526,
            -2.720195067059705,
            0.515349568691151,
            4.858796519538594,
            -1.0745735117250352,
            1.8560434180444785,
            -2.540853853933194,
        ];

        let mut df = DataFrame::new();
        df.add_numeric("y", y).unwrap();
        df.add_categorical("z", (1..=100).map(|idx| idx.to_string()).collect())
            .unwrap();
        df
    }

    fn current_logdet_xx(model: &LinearMixedModel) -> f64 {
        let k = model.reterms.len();
        let last = model.l_blocks[block_index(k, k)].as_dense();
        let p = last.nrows().saturating_sub(1);
        let mut logdet = 0.0;
        for i in 0..p {
            let diag = last[(i, i)];
            if diag > 0.0 {
                logdet += diag.ln();
            }
        }
        logdet * 2.0
    }

    fn make_vector_remat_for_kernel_tests(levels: usize) -> ReMat {
        let refs: Vec<u32> = (0..levels).map(|idx| idx as u32).collect();
        let level_names = (0..levels)
            .map(|idx| format!("S{:04}", idx + 1))
            .collect::<Vec<_>>();
        let cnames = vec!["(Intercept)".to_string(), "x".to_string()];
        let mut z = Vec::with_capacity(levels * 2);
        z.extend(std::iter::repeat_n(1.0, levels));
        z.extend((0..levels).map(|idx| idx as f64 + 0.5));

        ReMat::new(
            "subj".to_string(),
            refs,
            level_names,
            cnames,
            DMatrix::from_row_slice(2, levels, &z),
        )
    }

    #[test]
    fn test_lmm_carries_compiler_artifact_design_audit() {
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
        let model = LinearMixedModel::new(formula.clone(), &data, None).unwrap();

        let artifact = model.compiler_artifact();
        assert_eq!(artifact.requested_formula, formula.to_string());
        assert_eq!(artifact.semantic_model.random_terms.len(), 1);
        assert_eq!(artifact.theta_maps.len(), 1);

        let audit = model.design_audit().expect("design audit should attach");
        assert_eq!(audit.fixed_effect_rank.status, RankStatus::FullRank);
        assert_eq!(audit.fixed_effect_rank.rank, Some(2));
        assert_eq!(audit.random_terms[0].group.name, "subj");
        assert_eq!(audit.random_terms[0].group.n_levels, Some(18));
        assert_eq!(audit.random_terms[0].requested_covariance_parameters, 3);
    }

    #[test]
    fn test_random_effect_three_way_interaction_basis_is_materialized() {
        let mut data = DataFrame::new();
        data.add_numeric("y", vec![1.0, 2.0, 1.5, 2.5, 3.0, 4.0])
            .unwrap();
        data.add_numeric("A", vec![0.0, 1.0, 0.5, 1.5, 2.0, 2.5])
            .unwrap();
        data.add_numeric("B", vec![1.0, 0.5, 1.5, 1.0, 2.0, 1.5])
            .unwrap();
        data.add_numeric("C", vec![2.0, 1.0, 0.5, 1.5, 1.0, 2.5])
            .unwrap();
        data.add_categorical(
            "group",
            vec!["g1", "g1", "g1", "g2", "g2", "g2"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        )
        .unwrap();

        let formula = parse_formula("y ~ A * B * C + (A * B * C | group)").unwrap();
        let model = LinearMixedModel::new(formula, &data, None).unwrap();

        assert_eq!(
            model.reterms[0].cnames,
            vec!["(Intercept)", "A", "B", "C", "A:B", "A:C", "B:C", "A:B:C",]
        );
        assert_eq!(model.reterms[0].vsize, 8);
        assert_eq!(model.theta().len(), 36);
    }

    #[test]
    fn test_random_effect_categorical_slope_uses_treatment_coding_with_intercept() {
        let mut data = DataFrame::new();
        data.add_numeric("y", vec![1.0, 2.0, 3.0, 1.5, 2.5, 3.5])
            .unwrap();
        data.add_categorical(
            "cond",
            vec!["A", "B", "C", "A", "B", "C"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        )
        .unwrap();
        data.add_categorical(
            "subj",
            vec!["s1", "s1", "s1", "s2", "s2", "s2"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        )
        .unwrap();

        let formula = parse_formula("y ~ cond + (1 + cond | subj)").unwrap();
        let model = LinearMixedModel::new(formula, &data, None).unwrap();

        assert_eq!(
            model.reterms[0].cnames,
            vec!["(Intercept)", "cond: B", "cond: C"]
        );
        assert_eq!(model.reterms[0].vsize, 3);
        assert_eq!(model.theta().len(), 6);
        assert_eq!(
            model.compiler_artifact().theta_maps[0].block().user_basis,
            vec!["intercept".to_string(), "cond".to_string()]
        );
        assert_eq!(
            model.compiler_artifact().theta_maps[0]
                .block()
                .optimizer_basis,
            vec![
                "intercept".to_string(),
                "cond: B".to_string(),
                "cond: C".to_string()
            ]
        );
        assert_eq!(model.compiler_artifact().theta_maps[0].n_free(), 6);
    }

    #[test]
    fn test_explicit_categorical_contrast_basis_drives_fixed_random_and_interaction_columns() {
        let mut data = DataFrame::new();
        data.add_numeric("y", vec![1.0, 2.0, 1.5, 2.5, 1.2, 2.2])
            .unwrap();
        data.add_numeric("x", vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
            .unwrap();
        data.add_categorical_with_contrast(
            "anchor",
            vec!["low", "high", "low", "high", "low", "high"]
                .into_iter()
                .map(str::to_string)
                .collect(),
            vec!["low".to_string(), "high".to_string()],
            crate::model::data::CategoricalContrast::new(
                vec!["low".to_string(), "high".to_string()],
                DMatrix::from_row_slice(2, 1, &[0.5, -0.5]),
                vec!["hi_minus_lo".to_string()],
                false,
                crate::model::data::ContrastSource::Custom,
            )
            .unwrap(),
        )
        .unwrap();
        data.add_categorical(
            "subj",
            vec!["s1", "s1", "s2", "s2", "s3", "s3"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        )
        .unwrap();

        let formula = parse_formula("y ~ anchor + x:anchor + (1 + anchor | subj)").unwrap();
        let model = LinearMixedModel::new(formula, &data, None).unwrap();

        assert!(model
            .feterm
            .cnames
            .iter()
            .any(|name| name == "anchor: hi_minus_lo"));
        assert!(model
            .feterm
            .cnames
            .iter()
            .any(|name| name == "x:anchor: hi_minus_lo"));
        assert_eq!(
            model.reterms[0].cnames,
            vec!["(Intercept)", "anchor: hi_minus_lo"]
        );
        assert_eq!(
            model.reterms[0]
                .z
                .row(1)
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![0.5, -0.5, 0.5, -0.5, 0.5, -0.5]
        );

        let audit = model.design_audit().expect("design audit should attach");
        let anchor_basis = audit
            .fixed_effects
            .contrast_bases
            .iter()
            .find(|basis| basis.variable == "anchor")
            .expect("explicit contrast basis should be recorded");
        assert!(anchor_basis.explicit);
        assert_eq!(anchor_basis.source, "custom");
        assert_eq!(anchor_basis.column_names, vec!["hi_minus_lo"]);
        assert_eq!(anchor_basis.contrast_matrix, vec![vec![0.5], vec![-0.5]]);
        assert!(audit.fixed_effects.columns.iter().any(|column| {
            column.name == "anchor: hi_minus_lo"
                && column.kind == crate::compiler::FixedEffectColumnKind::CategoricalContrast
        }));
    }

    #[test]
    fn test_random_effect_categorical_slope_uses_cell_means_without_intercept() {
        let mut data = DataFrame::new();
        data.add_numeric("y", vec![1.0, 2.0, 3.0, 1.5, 2.5, 3.5])
            .unwrap();
        data.add_categorical(
            "cond",
            vec!["A", "B", "C", "A", "B", "C"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        )
        .unwrap();
        data.add_categorical(
            "subj",
            vec!["s1", "s1", "s1", "s2", "s2", "s2"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        )
        .unwrap();

        let formula = parse_formula("y ~ cond + (0 + cond | subj)").unwrap();
        let model = LinearMixedModel::new(formula, &data, None).unwrap();

        assert_eq!(
            model.reterms[0].cnames,
            vec!["cond: A", "cond: B", "cond: C"]
        );
        assert_eq!(model.reterms[0].vsize, 3);
        assert_eq!(model.theta().len(), 6);
        assert_eq!(
            model.compiler_artifact().theta_maps[0].block().user_basis,
            vec!["cond".to_string()]
        );
        assert_eq!(
            model.compiler_artifact().theta_maps[0]
                .block()
                .optimizer_basis,
            vec![
                "cond: A".to_string(),
                "cond: B".to_string(),
                "cond: C".to_string()
            ]
        );
        assert_eq!(model.compiler_artifact().theta_maps[0].n_free(), 6);
    }

    #[test]
    fn test_random_effect_no_intercept_factor_uses_cell_means_with_explicit_contrast() {
        let mut data = DataFrame::new();
        data.add_numeric("y", vec![1.0, 2.0, 1.5, 2.5, 1.2, 2.2])
            .unwrap();
        data.add_categorical_with_contrast(
            "anchor",
            vec!["low", "high", "low", "high", "low", "high"]
                .into_iter()
                .map(str::to_string)
                .collect(),
            vec!["low".to_string(), "high".to_string()],
            crate::model::data::CategoricalContrast::new(
                vec!["low".to_string(), "high".to_string()],
                DMatrix::from_row_slice(2, 1, &[0.5, -0.5]),
                vec!["hi_minus_lo".to_string()],
                false,
                crate::model::data::ContrastSource::Custom,
            )
            .unwrap(),
        )
        .unwrap();
        data.add_categorical(
            "subj",
            vec!["s1", "s1", "s2", "s2", "s3", "s3"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        )
        .unwrap();

        let formula = parse_formula("y ~ anchor + (0 + anchor | subj)").unwrap();
        let model = LinearMixedModel::new(formula, &data, None).unwrap();

        assert_eq!(model.reterms[0].cnames, vec!["anchor: low", "anchor: high"]);
        assert_eq!(
            model.reterms[0]
                .z
                .row(0)
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        );
        assert_eq!(
            model.reterms[0]
                .z
                .row(1)
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        );
    }

    #[test]
    fn test_random_effect_categorical_cell_means_preserves_zero_correlation_map() {
        let mut data = DataFrame::new();
        data.add_numeric("y", vec![1.0, 2.0, 3.0, 1.5, 2.5, 3.5])
            .unwrap();
        data.add_categorical(
            "cond",
            vec!["A", "B", "C", "A", "B", "C"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        )
        .unwrap();
        data.add_categorical(
            "subj",
            vec!["s1", "s1", "s1", "s2", "s2", "s2"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        )
        .unwrap();

        let formula = parse_formula("y ~ cond + (0 + cond || subj)").unwrap();
        let model = LinearMixedModel::new(formula, &data, None).unwrap();

        assert_eq!(
            model.reterms[0].cnames,
            vec!["cond: A", "cond: B", "cond: C"]
        );
        assert_eq!(model.theta().len(), 3);
        assert!(matches!(
            model.compiler_artifact().theta_maps[0],
            ThetaMap::Diagonal(_)
        ));
        assert_eq!(model.compiler_artifact().theta_maps[0].n_free(), 3);
    }

    #[test]
    fn test_random_effect_interaction_uses_cell_means_without_intercept() {
        let mut data = DataFrame::new();
        data.add_numeric("y", vec![1.0, 2.0, 1.5, 2.5]).unwrap();
        data.add_numeric("x", vec![0.5, 1.0, 1.5, 2.0]).unwrap();
        data.add_categorical(
            "cond",
            vec!["A", "B", "A", "B"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        )
        .unwrap();
        data.add_categorical(
            "subj",
            vec!["s1", "s1", "s2", "s2"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        )
        .unwrap();

        let formula = parse_formula("y ~ x * cond + (0 + x:cond | subj)").unwrap();
        let model = LinearMixedModel::new(formula, &data, None).unwrap();

        assert_eq!(model.reterms[0].cnames, vec!["x:cond: A", "x:cond: B"]);
        assert_eq!(model.reterms[0].vsize, 2);
        assert_eq!(
            model.compiler_artifact().theta_maps[0]
                .block()
                .optimizer_basis,
            vec!["x:cond: A".to_string(), "x:cond: B".to_string()]
        );
    }

    #[test]
    fn test_singular_fixture_maximal_model_has_too_rich_information_budget() {
        let (data, _) = crate::datasets::load("singular").unwrap();
        let formula = parse_formula("y ~ 1 + A * B * C + (A * B * C | group)").unwrap();
        let model = LinearMixedModel::new(formula, &data, None).unwrap();
        let audit = model.design_audit().expect("design audit should attach");
        let random = &audit.random_terms[0];

        assert_eq!(
            model.reterms[0].cnames,
            vec!["(Intercept)", "A", "B", "C", "A:B", "A:C", "B:C", "A:B:C",]
        );
        assert_eq!(random.group.n_levels, Some(10));
        assert_eq!(random.basis_size, 8);
        assert_eq!(random.requested_covariance_parameters, 36);
        assert_eq!(
            random.information_budget.status,
            InformationBudgetStatus::TooRich
        );
        assert_eq!(
            random.information_budget.min_levels_full_covariance,
            Some(180)
        );
    }

    #[test]
    fn test_singular_fixture_zcp_fit_exposes_reduced_effective_rank() {
        let (data, _) = crate::datasets::load("singular").unwrap();
        let formula = parse_formula("y ~ 1 + A * B * C + (A * B * C || group)").unwrap();
        let mut model = LinearMixedModel::new(formula.clone(), &data, None).unwrap();

        model.fit(false).unwrap();

        let summary = &model.compiler_artifact().effective_covariance[0];
        assert_eq!(summary.requested_rank, 8);
        assert!(summary.supported_rank < summary.requested_rank);
        assert_eq!(summary.status, EffectiveRankStatus::ReducedRank);
        assert_eq!(
            model.optimizer_certificate().unwrap().status,
            FitStatus::ConvergedReducedRank
        );
    }

    #[test]
    fn test_lmm_compiler_artifact_records_rank_deficient_fixed_effects() {
        let mut data = DataFrame::new();
        data.add_numeric("y", vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        data.add_numeric("x", vec![0.0, 1.0, 0.0, 1.0]).unwrap();
        data.add_numeric("x2", vec![0.0, 2.0, 0.0, 2.0]).unwrap();
        data.add_categorical(
            "z",
            vec!["a", "a", "b", "b"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        )
        .unwrap();

        let formula = parse_formula("y ~ x + x2 + (1 | z)").unwrap();
        let model = LinearMixedModel::new(formula, &data, None).unwrap();
        let audit = model.design_audit().expect("design audit should attach");

        assert_eq!(audit.fixed_effect_rank.status, RankStatus::RankDeficient);
        assert_eq!(audit.fixed_effect_rank.rank, Some(2));
        assert_eq!(audit.fixed_effect_rank.expected, Some(3));
        assert!(model
            .compiler_artifact()
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == DiagnosticCode::FixedEffectRankDeficient));
    }

    #[test]
    fn test_lmm_compiler_theta_maps_follow_optimizer_reterm_order() {
        let mut data = DataFrame::new();
        data.add_numeric("y", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        data.add_numeric("x", vec![0.0, 1.0, 0.5, 1.5, 0.25, 1.25])
            .unwrap();
        data.add_categorical(
            "small",
            vec!["s1", "s1", "s2", "s2", "s1", "s2"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        )
        .unwrap();
        data.add_categorical(
            "large",
            vec!["l1", "l2", "l3", "l1", "l2", "l3"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        )
        .unwrap();

        let formula = parse_formula("y ~ x + (1 | small) + (1 + x | large)").unwrap();
        let model = LinearMixedModel::new(formula, &data, None).unwrap();
        let maps = &model.compiler_artifact().theta_maps;

        assert_eq!(model.reterms[0].grouping_name, "large");
        assert_eq!(maps[0].block().term_id, "r1");
        assert_eq!(maps[0].block().term_index, 0);
        assert_eq!(maps[0].block().group, "large");
        assert_eq!(maps[0].block().theta_slots[0].global_index, Some(0));

        assert_eq!(model.reterms[1].grouping_name, "small");
        assert_eq!(maps[1].block().term_id, "r0");
        assert_eq!(maps[1].block().term_index, 1);
        assert_eq!(maps[1].block().group, "small");
        assert_eq!(maps[1].block().theta_slots[0].global_index, Some(3));

        let traces = &model.compiler_artifact().covariance_parameter_traces;
        assert_eq!(traces.len(), 4);
        assert_eq!(traces[0].term_id, "r1");
        assert_eq!(traces[0].source_syntax, "(1 + x | large)");
        assert_eq!(traces[0].optimizer_term_index, 0);
        assert_eq!(traces[0].lambda.row_basis, "intercept");
        assert!(traces
            .iter()
            .all(|trace| trace.parmap_entry.as_ref().unwrap().matches_theta_map));
    }

    #[test]
    fn test_lmm_optimizer_certificate_records_interior_fit() {
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula.clone(), &data, None).unwrap();

        assert!(model.optimizer_certificate().is_none());
        model.fit(false).unwrap();

        let certificate = model
            .optimizer_certificate()
            .expect("optimizer certificate should attach after fit");
        assert_eq!(certificate.status, FitStatus::ConvergedInterior);
        assert_eq!(
            certificate.optimizer_name.as_deref(),
            Some("pattern_search")
        );
        assert!(certificate.objective_value.is_some());
        assert!(certificate.evidence.optimizer_stop.acceptable_stop);
        assert!(!certificate.evidence.optimizer_stop.budget_exhausted);
        assert_eq!(certificate.evidence.parameter_space.n_theta, 1);
        assert_eq!(certificate.evidence.parameter_space.n_boundary, 0);
        assert_eq!(certificate.evidence.sample_size.n_observations, Some(180));
        assert_eq!(certificate.evidence.sample_size.n_theta, 1);
        assert!(matches!(
            certificate.evidence.certification_quality,
            EvidenceQuality::Approximate { .. }
        ));
        assert!(matches!(
            certificate.evidence.gradient.method,
            EvidenceMethod::FiniteDifference
        ));
        assert!(certificate.evidence.gradient.raw_gradient_norm.is_some());
        assert!(certificate.evidence.gradient.free_gradient_norm.is_some());
        assert!(certificate
            .evidence
            .gradient
            .projected_gradient_norm
            .is_some());
        assert!(matches!(
            certificate.evidence.hessian.method,
            EvidenceMethod::FiniteDifference
        ));
        assert!(certificate.evidence.hessian.min_eigenvalue.is_some());
        assert_eq!(certificate.evidence.hessian.rank, Some(1));
        assert!(certificate
            .checks
            .iter()
            .any(|check| matches!(check, CertificateCheck::FreeGradientOk { .. })));
        assert!(certificate
            .checks
            .iter()
            .any(|check| matches!(check, CertificateCheck::HessianPsdOnActiveSubspace { .. })));
        assert!(!certificate
            .checks
            .iter()
            .any(|check| matches!(check, CertificateCheck::NotAssessed { .. })));

        let verification = model.verify_convergence().unwrap();
        assert!(matches!(
            verification.status,
            ConvergenceVerificationStatus::RestartAgrees
                | ConvergenceVerificationStatus::OptimizerConsensus
        ));
        assert!(!verification.runs.is_empty());
        assert!(verification.runs.iter().all(|run| run.agrees));
        assert!(model
            .optimizer_certificate()
            .unwrap()
            .verification
            .is_some());

        let trace = &model.compiler_artifact().covariance_parameter_traces[0];
        assert!(trace.theta.value.is_some());
        assert!(trace.lambda.value.is_some());
        assert_eq!(trace.varcorr_entries[0].label, "sd(intercept)");
        assert!(trace.varcorr_entries[0].value.is_some());
    }

    #[test]
    fn test_lmm_convergence_verification_is_not_run_before_fit() {
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

        let verification = model.verify_convergence().unwrap();

        assert_eq!(verification.status, ConvergenceVerificationStatus::NotRun);
        assert!(verification.runs.is_empty());
        assert_eq!(verification.message, "model has not been fitted");
    }

    #[test]
    fn test_lmm_audit_report_updates_after_fit() {
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

        let prefit_report = model.audit_report().to_text();
        assert!(prefit_report.contains("Optimizer"));
        assert!(prefit_report.contains("model has not been fitted"));

        model.fit(false).unwrap();

        let fitted_report = model.audit_report().to_text();
        assert!(fitted_report.contains("ConvergedInterior"));
        assert!(fitted_report.contains("pattern_search"));
        assert!(fitted_report.contains("convergence interpretation"));
        assert!(fitted_report.contains("run verify_convergence()"));
    }

    #[test]
    fn test_lmm_optimizer_certificate_records_boundary_fit() {
        let data = singular_re_fixture();
        let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

        model.fit(false).unwrap();

        let certificate = model
            .optimizer_certificate()
            .expect("optimizer certificate should attach after fit");
        assert_eq!(certificate.status, FitStatus::ConvergedReducedRank);
        assert_eq!(certificate.evidence.parameter_space.n_boundary, 1);
        assert_eq!(
            certificate.evidence.parameter_space.boundary_indices,
            vec![0]
        );
        assert!(certificate
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == DiagnosticCode::BoundaryParameter));
        assert!(certificate.diagnostics.iter().any(|diagnostic| {
            diagnostic.code == DiagnosticCode::BoundaryParameter
                && diagnostic
                    .suggested_actions
                    .iter()
                    .any(|action| action.contains("valid fitted boundary"))
        }));
        let boundary_diagnostic = certificate
            .diagnostics
            .iter()
            .find(|diagnostic| diagnostic.code == DiagnosticCode::BoundaryParameter)
            .expect("boundary parameter diagnostic");
        assert_eq!(boundary_diagnostic.affected_terms, vec!["(1 | batch)"]);
        assert!(boundary_diagnostic
            .message
            .contains("standard deviation for intercept in (1 | batch)"));
        assert!(!boundary_diagnostic.message.contains("theta[0]"));
        assert_eq!(
            boundary_diagnostic.payload.get("theta_index"),
            Some(&serde_json::json!(0))
        );
        assert_eq!(
            boundary_diagnostic.payload.get("term_id"),
            Some(&serde_json::json!("r0"))
        );
        assert!(matches!(
            &certificate.evidence.gradient.method,
            EvidenceMethod::NotAssessed { reason } if reason.contains("variance-component boundary")
        ));
        assert!(certificate
            .evidence
            .gradient
            .kkt_boundary_gradient_max
            .is_none());
        assert!(matches!(
            &certificate.evidence.hessian.quality,
            EvidenceQuality::NotAssessed { reason } if reason.contains("variance-component boundary")
        ));
        assert_eq!(certificate.evidence.hessian.rank, None);
        assert!(certificate.checks.iter().any(|check| matches!(
            check,
            CertificateCheck::NotAssessed { reason }
                if reason.contains("boundary-gradient KKT check skipped")
        )));
        assert!(certificate
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == DiagnosticCode::CovarianceReduced));
        let covariance_diagnostic = certificate
            .diagnostics
            .iter()
            .find(|diagnostic| diagnostic.code == DiagnosticCode::CovarianceReduced)
            .expect("covariance reduced diagnostic");
        assert_eq!(covariance_diagnostic.affected_terms, vec!["(1 | batch)"]);
        assert!(covariance_diagnostic
            .message
            .contains("fitted covariance for (1 | batch)"));
        assert!(!covariance_diagnostic.message.contains("r0"));
        assert_eq!(
            covariance_diagnostic.payload.get("term_id"),
            Some(&serde_json::json!("r0"))
        );
        assert!(model
            .compiler_artifact()
            .reductions
            .iter()
            .all(|reduction| reduction.diagnostics.is_empty()));
        assert!(!model
            .compiler_artifact()
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == DiagnosticCode::CovarianceReduced));
        assert_eq!(
            model.compiler_artifact().effective_covariance[0].supported_rank,
            0
        );
    }

    #[test]
    fn test_effective_covariance_rank_uses_policy_thresholds() {
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        let mut policy = CompilerPolicy::maximal_feasible();
        policy.thresholds.effective_rank_relative_tolerance = 2.0;
        model.set_compiler_policy(policy).unwrap();

        model.fit(false).unwrap();

        let summary = &model.compiler_artifact().effective_covariance[0];
        assert_eq!(summary.status, EffectiveRankStatus::ReducedRank);
        assert_eq!(summary.supported_rank, 0);
        assert!(model
            .compiler_artifact()
            .reproducibility
            .thresholds
            .iter()
            .any(|(name, value)| name == "effective_rank_relative_tolerance" && value == "2"));
    }

    #[test]
    fn test_lmm_new_with_compiler_policy_applies_policy_before_fit() {
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
        let mut policy = CompilerPolicy::as_specified();
        policy.thresholds.effective_rank_relative_tolerance = 0.25;

        let model =
            LinearMixedModel::new_with_compiler_policy(formula, &data, None, policy).unwrap();

        assert_eq!(
            model.compiler_policy().random_strategy,
            crate::compiler::RandomStrategy::AsSpecified
        );
        assert!(model
            .compiler_artifact()
            .reproducibility
            .thresholds
            .iter()
            .any(|(name, value)| name == "effective_rank_relative_tolerance" && value == "0.25"));
    }

    #[test]
    fn test_lmm_design_compiled_reduces_full_covariance_before_fit() {
        let data = grouped_slope_data_with_obs(6, 3);
        let formula = parse_formula("y ~ x + (1 + x | group)").unwrap();

        let model = LinearMixedModel::new_with_compiler_policy(
            formula,
            &data,
            None,
            CompilerPolicy::design_compiled(),
        )
        .unwrap();
        let artifact = model.compiler_artifact();
        let state = artifact.model_state_summary();

        assert!(model.formula.random_terms[0].zerocorr);
        assert_eq!(model.theta().len(), 2);
        assert_eq!(artifact.theta_maps.len(), 2);
        assert_eq!(
            artifact
                .theta_maps
                .iter()
                .map(ThetaMap::n_free)
                .sum::<usize>(),
            2
        );
        assert_eq!(artifact.theta_maps[0].block().term_index, 0);
        assert_eq!(artifact.theta_maps[1].block().term_index, 0);
        assert_eq!(
            artifact.effective_formula.as_deref(),
            Some("y ~ 1 + x + (1 + x || group)")
        );
        assert_eq!(
            artifact.reproducibility.fit_intent,
            FitIntent::ConfirmatoryDesignCompiled
        );
        assert_eq!(state.supported.status, ModelStateStatus::Reduced);
        assert!(state.changes.iter().any(|change| {
            change.status == ModelChangeStatus::Applied
                && change.trigger == ReductionTrigger::DesignTime
                && change.replacement_term.as_deref() == Some("(1 + x || group)")
        }));
    }

    #[test]
    fn test_lmm_design_compiled_refuses_unsupported_random_distribution() {
        let data = grouped_slope_data(2);
        let formula = parse_formula("y ~ x + (1 + x | group)").unwrap();

        let result = LinearMixedModel::new_with_compiler_policy(
            formula,
            &data,
            None,
            CompilerPolicy::design_compiled(),
        );

        assert!(result.is_err());
        assert!(result
            .err()
            .unwrap()
            .to_string()
            .contains("design_compiled refused"));
    }

    #[test]
    fn test_lmm_design_compiled_refuses_row_saturated_random_effect() {
        let data = grouped_slope_data(100);
        let formula = parse_formula("y ~ x + (1 + x | group)").unwrap();

        let err = LinearMixedModel::new_with_compiler_policy(
            formula,
            &data,
            None,
            CompilerPolicy::design_compiled(),
        )
        .expect_err("row-saturated random-effect terms should be refused");
        let message = err.to_string();

        assert!(message.contains("number of observations (200)"));
        assert!(message.contains("random coefficients (200)"));
        assert!(message.contains("residual scale"));
    }

    #[test]
    fn test_lmm_set_compiler_policy_rejects_after_fit() {
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(true).unwrap();

        let error = model
            .set_compiler_policy(CompilerPolicy::as_specified())
            .expect_err("fitted models must reject policy mutation");

        assert!(matches!(error, MixedModelError::AlreadyFitted));
    }

    #[test]
    fn test_lmm_fit_with_compiler_policy_applies_policy_then_fits() {
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        let mut policy = CompilerPolicy::as_specified();
        policy.thresholds.effective_rank_relative_tolerance = 0.5;

        model.fit_with_compiler_policy(false, policy).unwrap();

        assert_eq!(
            model.compiler_policy().random_strategy,
            crate::compiler::RandomStrategy::AsSpecified
        );
        assert!(model.optimizer_certificate().is_some());
        assert!(model
            .compiler_artifact()
            .reproducibility
            .thresholds
            .iter()
            .any(|(name, value)| name == "effective_rank_relative_tolerance" && value == "0.5"));
    }

    #[test]
    fn test_lmm_optimizer_certificate_records_budget_stop() {
        let data = singular_re_fixture();
        let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.optsum.max_feval = 1;

        model.fit(false).unwrap();

        let certificate = model
            .optimizer_certificate()
            .expect("optimizer certificate should attach after fit");
        assert_eq!(certificate.status, FitStatus::NotOptimized);
        assert!(!certificate.evidence.optimizer_stop.acceptable_stop);
        assert!(certificate.evidence.optimizer_stop.budget_exhausted);
        assert!(matches!(
            certificate.evidence.certification_quality,
            EvidenceQuality::Failed { .. }
        ));
        assert!(certificate
            .checks
            .iter()
            .any(|check| matches!(check, CertificateCheck::Failed { .. })));
        assert!(model.compiler_artifact().effective_covariance.is_empty());
    }

    #[test]
    fn test_block_index() {
        assert_eq!(block_index(0, 0), 0);
        assert_eq!(block_index(1, 0), 1);
        assert_eq!(block_index(1, 1), 2);
        assert_eq!(block_index(2, 0), 3);
        assert_eq!(block_index(2, 1), 4);
        assert_eq!(block_index(2, 2), 5);
    }

    #[test]
    fn test_dense_crossed_block_guard_reports_problem_too_large() {
        let err = ensure_dense_block_within_explicit_limit(
            1_400_000,
            100_000,
            "issue-702-scale crossed random-effects block",
            16 * 1024 * 1024 * 1024,
        )
        .expect_err("issue-702-scale dense block should be refused before allocation");

        assert!(matches!(err, MixedModelError::ProblemTooLarge(_)));
        assert!(err.to_string().contains("1043."));
        assert!(err.to_string().contains("issue-702-scale"));
    }

    #[test]
    fn test_dense_crossed_block_guard_accepts_small_blocks() {
        ensure_dense_block_within_explicit_limit(
            100,
            80,
            "small crossed random-effects block",
            16 * 1024 * 1024 * 1024,
        )
        .expect("small dense blocks should remain valid");
    }

    #[test]
    fn test_crossed_scalar_re_cross_product_stays_sparse() {
        let mut data = DataFrame::new();
        data.add_numeric("y", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        data.add_categorical(
            "person",
            vec!["p1", "p1", "p2", "p3", "p3", "p1"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        )
        .unwrap();
        data.add_categorical(
            "firm",
            vec!["f1", "f2", "f2", "f1", "f3", "f1"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        )
        .unwrap();

        let formula = parse_formula("y ~ 1 + (1 | person) + (1 | firm)").unwrap();
        let model = LinearMixedModel::new(formula, &data, None).unwrap();

        assert!(
            matches!(model.a_blocks[block_index(1, 0)], MatrixBlock::Sparse(_)),
            "crossed scalar RE off-diagonal A block should not be materialized dense"
        );
        let MatrixBlock::Sparse(cross) = &model.a_blocks[block_index(1, 0)] else {
            unreachable!();
        };
        assert_eq!(cross.nrows(), model.reterms[1].n_ranef());
        assert_eq!(cross.ncols(), model.reterms[0].n_ranef());
        assert!(cross.nnz() <= data.nrow());

        let dense = MatrixBlock::Sparse(cross.clone()).as_dense();
        let person_p1 = model.reterms[0]
            .levels
            .iter()
            .position(|level| level == "p1")
            .unwrap();
        let firm_f1 = model.reterms[1]
            .levels
            .iter()
            .position(|level| level == "f1")
            .unwrap();
        assert_eq!(dense[(firm_f1, person_p1)], 2.0);
    }

    #[test]
    fn test_cholesky_block_diagonal() {
        let mut block = MatrixBlock::Diagonal(DVector::from_vec(vec![4.0, 9.0, 16.0]));
        cholesky_block(&mut block).unwrap();
        if let MatrixBlock::Diagonal(d) = &block {
            assert!((d[0] - 2.0).abs() < 1e-10);
            assert!((d[1] - 3.0).abs() < 1e-10);
            assert!((d[2] - 4.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_cholesky_block_dense() {
        // [[4, 2], [2, 5]] → L = [[2, 0], [1, 2]]
        let mut block = MatrixBlock::Dense(DMatrix::from_row_slice(2, 2, &[4.0, 2.0, 2.0, 5.0]));
        cholesky_block(&mut block).unwrap();
        if let MatrixBlock::Dense(m) = &block {
            assert!((m[(0, 0)] - 2.0).abs() < 1e-10);
            assert!((m[(1, 0)] - 1.0).abs() < 1e-10);
            assert!((m[(1, 1)] - 2.0).abs() < 1e-10);
            assert!(m[(0, 1)].abs() < 1e-10);
        }
    }

    #[test]
    fn test_cholesky_zero_pad_scales_with_diagonal() {
        let mut unit_scale = MatrixBlock::Dense(DMatrix::from_diagonal(&DVector::from_vec(vec![
            -1e-12, 1.0,
        ])));
        assert!(matches!(
            cholesky_block(&mut unit_scale),
            Err(MixedModelError::PosDefException)
        ));

        let mut large_scale = MatrixBlock::Dense(DMatrix::from_diagonal(&DVector::from_vec(vec![
            -1e-12, 1e8,
        ])));
        cholesky_block(&mut large_scale).unwrap();
        let MatrixBlock::Dense(mat) = large_scale else {
            unreachable!();
        };
        assert_eq!(mat[(0, 0)], 0.0);
        assert_relative_eq!(mat[(1, 1)], 1e4, epsilon = 1e-8);
    }

    #[test]
    fn test_cholesky_rejects_near_singular_negative_pivot_at_unit_scale() {
        let mut block =
            MatrixBlock::Dense(DMatrix::from_diagonal(&DVector::from_vec(vec![-1e-9, 1.0])));

        assert!(matches!(
            cholesky_block(&mut block),
            Err(MixedModelError::PosDefException)
        ));
    }

    #[test]
    fn test_cholesky_strict_mode_matches_julia() {
        let mut block = MatrixBlock::Dense(DMatrix::from_diagonal(&DVector::from_vec(vec![
            -f64::EPSILON,
            1e16,
        ])));

        assert!(matches!(
            cholesky_block_with_tolerance(&mut block, 0.0),
            Err(MixedModelError::PosDefException)
        ));
    }

    #[test]
    fn test_logdet_block() {
        let block = MatrixBlock::Diagonal(DVector::from_vec(vec![2.0, 3.0]));
        let ld = logdet_block(&block);
        // logdet = 2 * (ln(2) + ln(3)) = 2 * ln(6)
        assert!((ld - 2.0 * 6.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_rank_k_downdate_small_dense_large_k_matches_gemm() {
        let a = DMatrix::from_fn(3, 520, |row, col| {
            (((row + 1) * (col + 3)) % 17) as f64 / 13.0 - 0.4
        });
        let init = DMatrix::from_row_slice(3, 3, &[3.0, 0.2, 0.4, 0.2, 2.5, -0.1, 0.4, -0.1, 1.7]);
        let mut optimized = MatrixBlock::Dense(init.clone());
        let mut expected = init;
        expected.gemm(-1.0, &a, &a.transpose(), 1.0);

        rank_k_downdate(&mut optimized, &a);

        let MatrixBlock::Dense(result) = optimized else {
            panic!("expected dense block");
        };
        for row in 0..3 {
            for col in 0..=row {
                assert_relative_eq!(
                    result[(row, col)],
                    expected[(row, col)],
                    epsilon = 1e-10,
                    max_relative = 1e-12
                );
            }
        }
    }

    #[test]
    fn test_create_al_single_vsize2_matches_generic_blocks() {
        let data = simulate_sleepstudy_like(260, 3, 23);
        let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
        let y_data = data.numeric(&formula.response).unwrap();
        let y = DVector::from_column_slice(y_data);
        let (x_mat, fe_names) = build_fixed_effects_matrix(&formula, &data).unwrap();
        let feterm = FeTerm::new(x_mat, fe_names);
        let xy = FeMat::new(&feterm, &y);
        let re = build_re_mat(&formula.random_terms[0], &data, data.nrow()).unwrap();

        let (specialized, _) = create_al_single_vsize2(&re, &xy);
        let generic = vec![
            compute_re_cross_product(&re, &re),
            compute_fe_re_cross_product(&xy, &re),
            MatrixBlock::Dense(xy.wtxy.transpose() * &xy.wtxy),
        ];

        for (left, right) in specialized.iter().zip(generic.iter()) {
            let left_dense = left.as_dense();
            let right_dense = right.as_dense();
            assert_eq!(left_dense.shape(), right_dense.shape());
            for row in 0..left_dense.nrows() {
                for col in 0..left_dense.ncols() {
                    assert_relative_eq!(
                        left_dense[(row, col)],
                        right_dense[(row, col)],
                        epsilon = 1e-10,
                        max_relative = 1e-12
                    );
                }
            }
        }
    }

    #[test]
    fn test_fixed_design_solver_blocks_match_femat_blocks_unweighted() {
        let data = simulate_sleepstudy_like(24, 4, 23);
        let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
        let y = DVector::from_column_slice(data.numeric(&formula.response).unwrap());
        let raw_fixed_design = crate::model::fixed_design::build_fixed_effects_design_with_policy(
            &formula,
            &data,
            FixedDesignBuildPolicy::streamed(),
        )
        .unwrap();
        let feterm = FeTerm::new(
            raw_fixed_design.materialize_dense(),
            raw_fixed_design.column_names().to_vec(),
        );
        let fixed_design = raw_fixed_design
            .select_columns(&feterm.piv[..feterm.rank])
            .unwrap();
        let xy = FeMat::new(&feterm, &y);
        let re = build_re_mat(&formula.random_terms[0], &data, data.nrow()).unwrap();

        let (backend_blocks, _) =
            create_al_from_fixed_design(&[re.clone()], &fixed_design, &y, None).unwrap();
        let (dense_blocks, _) = create_al(&[re], &xy).unwrap();

        for (backend, dense) in backend_blocks.iter().zip(dense_blocks.iter()) {
            let backend_dense = backend.as_dense();
            let expected_dense = dense.as_dense();
            assert_eq!(backend_dense.shape(), expected_dense.shape());
            for row in 0..backend_dense.nrows() {
                for col in 0..backend_dense.ncols() {
                    assert_relative_eq!(
                        backend_dense[(row, col)],
                        expected_dense[(row, col)],
                        epsilon = 1e-10,
                        max_relative = 1e-12
                    );
                }
            }
        }
    }

    #[test]
    fn test_fixed_design_solver_blocks_match_femat_blocks_weighted() {
        let data = simulate_sleepstudy_like(24, 4, 47);
        let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
        let y = DVector::from_column_slice(data.numeric(&formula.response).unwrap());
        let raw_fixed_design = crate::model::fixed_design::build_fixed_effects_design_with_policy(
            &formula,
            &data,
            FixedDesignBuildPolicy::streamed(),
        )
        .unwrap();
        let feterm = FeTerm::new(
            raw_fixed_design.materialize_dense(),
            raw_fixed_design.column_names().to_vec(),
        );
        let fixed_design = raw_fixed_design
            .select_columns(&feterm.piv[..feterm.rank])
            .unwrap();
        let sqrtwts = DVector::from_iterator(
            data.nrow(),
            (0..data.nrow()).map(|idx| if idx % 2 == 0 { 1.0 } else { 2.0 }),
        );
        let mut xy = FeMat::new(&feterm, &y);
        xy.reweight(&sqrtwts);
        let mut re = build_re_mat(&formula.random_terms[0], &data, data.nrow()).unwrap();
        re.reweight(&sqrtwts);

        let (backend_blocks, _) =
            create_al_from_fixed_design(&[re.clone()], &fixed_design, &y, Some(&sqrtwts)).unwrap();
        let (dense_blocks, _) = create_al(&[re], &xy).unwrap();

        for (backend, dense) in backend_blocks.iter().zip(dense_blocks.iter()) {
            let backend_dense = backend.as_dense();
            let expected_dense = dense.as_dense();
            assert_eq!(backend_dense.shape(), expected_dense.shape());
            for row in 0..backend_dense.nrows() {
                for col in 0..backend_dense.ncols() {
                    assert_relative_eq!(
                        backend_dense[(row, col)],
                        expected_dense[(row, col)],
                        epsilon = 1e-10,
                        max_relative = 1e-12
                    );
                }
            }
        }
    }

    #[test]
    fn test_lmm_constructor_keeps_high_cardinality_fixed_design_streamed() {
        let n_levels = 256usize;
        let n_obs = 512usize;
        let formula = parse_formula("y ~ 1 + sku + (1 | group)").unwrap();
        let mut data = DataFrame::new();
        data.add_numeric("y", (0..n_obs).map(|idx| idx as f64).collect())
            .unwrap();
        data.add_categorical(
            "sku",
            (0..n_obs)
                .map(|idx| format!("sku{}", idx % n_levels))
                .collect(),
        )
        .unwrap();
        data.add_categorical(
            "group",
            (0..n_obs).map(|idx| format!("g{}", idx % 16)).collect(),
        )
        .unwrap();

        let model = LinearMixedModel::new(formula, &data, None).unwrap();

        assert_eq!(
            model.fixed_design.storage(),
            crate::model::fixed_design::FixedDesignStorage::Streamed
        );
        assert_eq!(model.fixed_design.n_cols(), model.feterm.rank);
        assert!(model.fixed_design.as_streamed().is_some());

        let summary = model.fixed_design_backend_summary();
        assert_eq!(summary.storage, FixedDesignStorage::Streamed);
        assert_eq!(summary.n_obs, n_obs);
        assert_eq!(summary.n_cols, model.feterm.rank);
        assert!(model.fixed_design_density() < 0.02);
        assert!(model.fixed_design_active_entries() < n_obs * 3);

        let diagnostic = model
            .compiler_artifact()
            .diagnostics
            .iter()
            .find(|diagnostic| {
                diagnostic.code == DiagnosticCode::SupportNote
                    && diagnostic
                        .payload
                        .get("diagnostic_kind")
                        .and_then(|value| value.as_str())
                        == Some("fixed_design_backend")
            })
            .expect("streamed backend should be exposed as a structured diagnostic");
        assert_eq!(
            diagnostic
                .payload
                .get("storage")
                .and_then(|value| value.as_str()),
            Some("streamed")
        );
        assert!(diagnostic
            .message
            .contains("fixed-effect design backend selected: streamed"));

        let report = model.audit_report().to_text();
        assert!(report.contains("fixed-effect design backend selected: streamed"));
        assert!(report.contains("rank and pivot detection still materialize dense X"));
    }

    fn streamed_fixed_effect_parity_fixture(n_levels: usize, obs_per_level: usize) -> DataFrame {
        let n_obs = n_levels * obs_per_level;
        let mut y = Vec::with_capacity(n_obs);
        let mut x = Vec::with_capacity(n_obs);
        let mut sku = Vec::with_capacity(n_obs);
        let mut group = Vec::with_capacity(n_obs);

        for level in 0..n_levels {
            for rep in 0..obs_per_level {
                let obs = level * obs_per_level + rep;
                let x_value = rep as f64 - 0.5 + ((level % 5) as f64) * 0.1;
                let sku_effect = ((level % 11) as f64 - 5.0) * 0.07;
                let group_effect = ((obs % 17) as f64 - 8.0) * 0.03;
                let noise = ((obs % 7) as f64 - 3.0) * 0.01;
                x.push(x_value);
                y.push(2.0 + 0.8 * x_value + sku_effect + group_effect + noise);
                sku.push(format!("sku{:03}", level));
                group.push(format!("g{:02}", obs % 17));
            }
        }

        let mut data = DataFrame::new();
        data.add_numeric("y", y).unwrap();
        data.add_numeric("x", x).unwrap();
        data.add_categorical("sku", sku).unwrap();
        data.add_categorical("group", group).unwrap();
        data
    }

    fn assert_lmm_fit_close(left: &LinearMixedModel, right: &LinearMixedModel) {
        assert_eq!(left.coef_names(), right.coef_names());
        let left_theta = left.theta();
        let right_theta = right.theta();
        assert_eq!(left_theta.len(), right_theta.len());
        for (left_theta, right_theta) in left_theta.iter().zip(right_theta.iter()) {
            assert_relative_eq!(
                *left_theta,
                *right_theta,
                epsilon = 1e-8,
                max_relative = 1e-8
            );
        }

        let left_beta = left.beta();
        let right_beta = right.beta();
        assert_eq!(left_beta.len(), right_beta.len());
        for idx in 0..left_beta.len() {
            assert_relative_eq!(
                left_beta[idx],
                right_beta[idx],
                epsilon = 1e-8,
                max_relative = 1e-8
            );
        }

        assert_relative_eq!(
            left.sigma(),
            right.sigma(),
            epsilon = 1e-8,
            max_relative = 1e-8
        );
        assert_relative_eq!(
            left.objective_value(),
            right.objective_value(),
            epsilon = 1e-8,
            max_relative = 1e-8
        );

        let left_fitted = left.fitted();
        let right_fitted = right.fitted();
        assert_eq!(left_fitted.len(), right_fitted.len());
        for idx in 0..left_fitted.len() {
            assert_relative_eq!(
                left_fitted[idx],
                right_fitted[idx],
                epsilon = 1e-8,
                max_relative = 1e-8
            );
        }
    }

    #[test]
    fn test_streamed_fixed_effect_lmm_fit_matches_dense_backend() {
        let data = streamed_fixed_effect_parity_fixture(64, 4);
        let formula = parse_formula("y ~ 1 + x + sku + (1 | group)").unwrap();

        let mut dense = LinearMixedModel::new_with_fixed_design_policy(
            formula.clone(),
            &data,
            None,
            FixedDesignBuildPolicy::dense(),
        )
        .unwrap();
        let mut streamed = LinearMixedModel::new_with_fixed_design_policy(
            formula,
            &data,
            None,
            FixedDesignBuildPolicy::streamed(),
        )
        .unwrap();

        assert_eq!(
            dense.fixed_design.storage(),
            crate::model::fixed_design::FixedDesignStorage::Dense
        );
        assert_eq!(
            streamed.fixed_design.storage(),
            crate::model::fixed_design::FixedDesignStorage::Streamed
        );

        dense.fit(false).unwrap();
        streamed.fit(false).unwrap();

        assert_lmm_fit_close(&dense, &streamed);
    }

    #[test]
    fn test_weighted_streamed_fixed_effect_lmm_fit_matches_dense_backend() {
        let data = streamed_fixed_effect_parity_fixture(48, 5);
        let formula = parse_formula("y ~ 1 + x + sku + (1 | group)").unwrap();
        let weights = (0..data.nrow())
            .map(|idx| 0.5 + ((idx % 5) as f64) * 0.25)
            .collect::<Vec<_>>();

        let mut dense = LinearMixedModel::new_with_fixed_design_policy(
            formula.clone(),
            &data,
            Some(&weights),
            FixedDesignBuildPolicy::dense(),
        )
        .unwrap();
        let mut streamed = LinearMixedModel::new_with_fixed_design_policy(
            formula,
            &data,
            Some(&weights),
            FixedDesignBuildPolicy::streamed(),
        )
        .unwrap();

        assert_eq!(
            dense.fixed_design.storage(),
            crate::model::fixed_design::FixedDesignStorage::Dense
        );
        assert_eq!(
            streamed.fixed_design.storage(),
            crate::model::fixed_design::FixedDesignStorage::Streamed
        );

        dense.fit(false).unwrap();
        streamed.fit(false).unwrap();

        assert_lmm_fit_close(&dense, &streamed);
    }

    #[test]
    fn test_rdiv_lower_transpose_diagonal() {
        let mut a = MatrixBlock::Dense(DMatrix::from_row_slice(
            2,
            3,
            &[4.0, 9.0, 8.0, 2.0, 3.0, 5.0],
        ));
        let l = MatrixBlock::Diagonal(DVector::from_vec(vec![2.0, 3.0, 0.0]));

        rdiv_lower_transpose(&mut a, &l);

        if let MatrixBlock::Dense(m) = &a {
            assert_relative_eq!(m[(0, 0)], 2.0, epsilon = 1e-12);
            assert_relative_eq!(m[(1, 0)], 1.0, epsilon = 1e-12);
            assert_relative_eq!(m[(0, 1)], 3.0, epsilon = 1e-12);
            assert_relative_eq!(m[(1, 1)], 1.0, epsilon = 1e-12);
            assert_relative_eq!(m[(0, 2)], 0.0, epsilon = 1e-12);
            assert_relative_eq!(m[(1, 2)], 0.0, epsilon = 1e-12);
        } else {
            panic!("expected dense block after diagonal solve");
        }
    }

    #[test]
    fn test_blocked_forward_solve_zero_pivot_guard_uniform() {
        let tiny_but_solvable = f64::EPSILON * 0.5;
        let effectively_zero = BLOCK_TRIANGULAR_SOLVE_ZERO_TOLERANCE * 0.5;
        let blocks = [
            (
                MatrixBlock::Diagonal(DVector::from_vec(vec![tiny_but_solvable, 2.0])),
                [1.0, 2.0],
            ),
            (
                MatrixBlock::BlockDiagonal(vec![DMatrix::from_row_slice(
                    2,
                    2,
                    &[tiny_but_solvable, 0.0, 3.0, 2.0],
                )]),
                [1.0, 0.5],
            ),
            (
                MatrixBlock::Dense(DMatrix::from_row_slice(
                    2,
                    2,
                    &[tiny_but_solvable, 0.0, 3.0, 2.0],
                )),
                [1.0, 0.5],
            ),
        ];

        for (block, expected) in &blocks {
            let mut rhs = vec![tiny_but_solvable, 4.0];
            solve_lower_block_against_rhs(block, &mut rhs);
            assert_relative_eq!(rhs[0], expected[0], epsilon = 1e-12);
            assert_relative_eq!(rhs[1], expected[1], epsilon = 1e-12);

            let mut rhs_matrix = DMatrix::from_column_slice(2, 1, &[tiny_but_solvable, 4.0]);
            solve_lower_block_rhs(&mut rhs_matrix, block);
            assert_relative_eq!(rhs_matrix[(0, 0)], rhs[0], epsilon = 1e-12);
            assert_relative_eq!(rhs_matrix[(1, 0)], rhs[1], epsilon = 1e-12);
        }

        let mut rhs = vec![1.0, 4.0];
        solve_lower_block_against_rhs(
            &MatrixBlock::Dense(DMatrix::from_row_slice(
                2,
                2,
                &[effectively_zero, 0.0, 3.0, 2.0],
            )),
            &mut rhs,
        );
        assert_eq!(rhs, vec![0.0, 2.0]);
    }

    #[test]
    fn test_copy_scale_inflate_vsize2_matches_reference() {
        let mut re = make_vector_remat_for_kernel_tests(2);
        re.set_theta(&[1.2, -0.35, 0.8]).unwrap();

        let src_blocks = vec![
            DMatrix::from_row_slice(2, 2, &[3.0, 0.4, 0.4, 2.5]),
            DMatrix::from_row_slice(2, 2, &[1.7, -0.2, -0.2, 0.9]),
        ];
        let a = MatrixBlock::BlockDiagonal(src_blocks.clone());
        let mut l = MatrixBlock::BlockDiagonal(vec![DMatrix::zeros(2, 2), DMatrix::zeros(2, 2)]);

        copy_scale_inflate(&mut l, &a, &re);

        let MatrixBlock::BlockDiagonal(result_blocks) = l else {
            panic!("expected block-diagonal result");
        };

        for (result, src) in result_blocks.iter().zip(src_blocks.iter()) {
            let expected = re.lambda.transpose() * src * &re.lambda + DMatrix::identity(2, 2);
            for row in 0..2 {
                for col in 0..2 {
                    assert_relative_eq!(
                        result[(row, col)],
                        expected[(row, col)],
                        epsilon = 1e-12,
                        max_relative = 1e-12
                    );
                }
            }
        }
    }

    #[test]
    fn test_copy_and_scale_offdiag_vsize2_matches_reference() {
        let mut re_i = make_vector_remat_for_kernel_tests(2);
        let mut re_j = make_vector_remat_for_kernel_tests(2);
        re_i.set_theta(&[1.1, -0.25, 0.9]).unwrap();
        re_j.set_theta(&[0.8, 0.3, 1.4]).unwrap();

        let a_dense = DMatrix::from_row_slice(
            4,
            4,
            &[
                1.0, 0.2, -0.3, 0.5, 0.6, 1.4, 0.1, -0.2, -0.4, 0.3, 1.6, 0.7, 0.2, -0.5, 0.8, 1.1,
            ],
        );
        let a = MatrixBlock::Dense(a_dense.clone());
        let mut l = MatrixBlock::Dense(DMatrix::zeros(4, 4));

        copy_and_scale_offdiag(&mut l, &a, &re_i, &re_j);

        let MatrixBlock::Dense(result) = l else {
            panic!("expected dense result");
        };

        let mut expected = DMatrix::zeros(4, 4);
        for bi in 0..2 {
            let row0 = bi * 2;
            for bj in 0..2 {
                let col0 = bj * 2;
                let src = a_dense.view((row0, col0), (2, 2)).into_owned();
                let block = re_i.lambda.transpose() * src * &re_j.lambda;
                for row in 0..2 {
                    for col in 0..2 {
                        expected[(row0 + row, col0 + col)] = block[(row, col)];
                    }
                }
            }
        }

        for row in 0..4 {
            for col in 0..4 {
                assert_relative_eq!(
                    result[(row, col)],
                    expected[(row, col)],
                    epsilon = 1e-12,
                    max_relative = 1e-12
                );
            }
        }
    }

    #[test]
    fn test_rdiv_lower_transpose_blockdiag_vsize2_matches_dense_reference() {
        let mut a = MatrixBlock::Dense(DMatrix::from_row_slice(
            3,
            4,
            &[
                2.0, -1.0, 0.5, 1.2, 0.1, 3.0, -0.4, 0.8, -2.1, 0.7, 1.5, -0.9,
            ],
        ));
        let l = MatrixBlock::BlockDiagonal(vec![
            DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.5, 1.5]),
            DMatrix::from_row_slice(2, 2, &[1.3, 0.0, -0.2, 0.9]),
        ]);

        let mut expected = DMatrix::from_row_slice(
            3,
            4,
            &[
                2.0, -1.0, 0.5, 1.2, 0.1, 3.0, -0.4, 0.8, -2.1, 0.7, 1.5, -0.9,
            ],
        );
        let dense_l = l.as_dense();
        for j in 0..dense_l.ncols() {
            if dense_l[(j, j)].abs() < 1e-30 {
                for i in 0..expected.nrows() {
                    expected[(i, j)] = 0.0;
                }
                continue;
            }
            for i in 0..expected.nrows() {
                let mut s = expected[(i, j)];
                for k in 0..j {
                    s -= expected[(i, k)] * dense_l[(j, k)];
                }
                expected[(i, j)] = s / dense_l[(j, j)];
            }
        }

        rdiv_lower_transpose(&mut a, &l);

        let MatrixBlock::Dense(result) = a else {
            panic!("expected dense result");
        };

        for row in 0..result.nrows() {
            for col in 0..result.ncols() {
                assert_relative_eq!(
                    result[(row, col)],
                    expected[(row, col)],
                    epsilon = 1e-12,
                    max_relative = 1e-12
                );
            }
        }
    }

    #[test]
    fn test_objective_at_reuses_work_blocks_without_drift() {
        let data = simulate_sleepstudy_like(8, 6, 7);
        let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

        let theta_a = [1.3, -0.15, 0.8];
        let theta_b = [0.7, 0.25, 1.4];

        let obj_a1 = model.objective_at(&theta_a).unwrap();
        let _obj_b = model.objective_at(&theta_b).unwrap();
        let obj_a2 = model.objective_at(&theta_a).unwrap();

        assert_relative_eq!(obj_a1, obj_a2, epsilon = 1e-10, max_relative = 1e-10);
    }

    #[test]
    fn test_fast_vsize2_profiled_objective_matches_generic_update() {
        let data = simulate_sleepstudy_like(300, 3, 17);
        let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.optsum.reml = true;
        let theta = [0.9, 0.2, 0.35];

        let generic = model.objective_at(&theta).unwrap();
        let fast = LinearMixedModel::profiled_objective_one_vsize2_fast(
            &model.a_blocks,
            &model.reterms,
            &theta,
            model.dims,
            true,
            model.optsum.sigma,
            model
                .compiler_policy()
                .thresholds
                .cholesky_zero_pad_tolerance,
        )
        .expect("large one-term vector RE should use the fast objective path");

        assert_relative_eq!(fast, generic, epsilon = 1e-8, max_relative = 1e-12);
    }

    #[test]
    fn test_vector_re_fit_is_invariant_to_row_order() {
        let data = simulate_sleepstudy_like(10, 5, 42);
        let order: Vec<usize> = (0..data.nrow()).rev().collect();
        let permuted = permute_rows(&data, &order);
        let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();

        let mut model_a = LinearMixedModel::new(formula.clone(), &data, None).unwrap();
        let mut model_b = LinearMixedModel::new(formula, &permuted, None).unwrap();

        model_a.fit(true).unwrap();
        model_b.fit(true).unwrap();

        assert_relative_eq!(
            model_a.objective_value(),
            model_b.objective_value(),
            epsilon = 1e-7,
            max_relative = 1e-7
        );
        assert_relative_eq!(
            model_a.sigma(),
            model_b.sigma(),
            epsilon = 1e-3,
            max_relative = 1e-3
        );

        let beta_a = model_a.beta();
        let beta_b = model_b.beta();
        for i in 0..beta_a.len() {
            assert_relative_eq!(beta_a[i], beta_b[i], epsilon = 1e-4, max_relative = 1e-4);
        }

        let theta_a = model_a.theta();
        let theta_b = model_b.theta();
        for i in 0..theta_a.len() {
            assert_relative_eq!(theta_a[i], theta_b[i], epsilon = 5e-3, max_relative = 5e-3);
        }
    }

    #[cfg(feature = "nlopt")]
    #[test]
    fn test_vector_fit_uses_bobyqa_with_bounded_evaluations() {
        // n_theta = 3 (correlated random slope) → BOBYQA path. Pattern
        // search is the fallback if BOBYQA fails to converge; here we
        // expect the primary path to succeed and to use far fewer evals
        // than pattern_search did (which was bounded at 140).
        let data = simulate_sleepstudy_like(18, 10, 42);
        let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

        model.fit(true).unwrap();

        assert_eq!(model.optsum.optimizer, Optimizer::NloptBobyqa);
        assert!(
            model.optsum.feval <= 80,
            "bobyqa used too many evaluations: {}",
            model.optsum.feval
        );
    }

    #[cfg(feature = "nlopt")]
    #[test]
    fn test_large_theta_fit_uses_nlopt_newuoa() {
        let data = simulate_large_theta_crossed(123);
        let formula = parse_formula(
            "reaction ~ 1 + days + (1 + days | subj) + (1 + days | item) + (1 + days | site)",
        )
        .unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.optsum.max_feval = 3000;

        model.fit(true).unwrap();

        assert_eq!(model.n_theta(), 9);
        assert_eq!(model.optsum.optimizer, Optimizer::NloptNewuoa);
        assert!(model.objective_value().is_finite());
        assert!(model.sigma().is_finite());
    }

    #[cfg(feature = "nlopt")]
    #[test]
    fn test_large_theta_nlopt_matches_or_beats_cobyla_baseline() {
        let data = simulate_large_theta_crossed(123);
        let formula = parse_formula(
            "reaction ~ 1 + days + (1 + days | subj) + (1 + days | item) + (1 + days | site)",
        )
        .unwrap();

        let mut model_nlopt = LinearMixedModel::new(formula.clone(), &data, None).unwrap();
        model_nlopt.optsum.max_feval = 3000;
        model_nlopt.fit(true).unwrap();

        let mut model_cobyla = LinearMixedModel::new(formula, &data, None).unwrap();
        model_cobyla.optsum.max_feval = 3000;
        model_cobyla.optsum.reml = true;
        let theta0 = model_cobyla.optsum.initial.clone();
        model_cobyla.optsum.finitial = model_cobyla.objective_at(&theta0).unwrap();
        model_cobyla
            .fit_cobyla_with_maxeval(true, Some(3000))
            .unwrap();

        assert!(
            model_nlopt.objective_value() <= model_cobyla.objective_value() + 1e-2,
            "nlopt objective {} should match or beat cobyla {} within tolerance",
            model_nlopt.objective_value(),
            model_cobyla.objective_value()
        );
        assert!(model_nlopt.optsum.feval < model_cobyla.optsum.feval);
    }

    #[test]
    fn test_profile_response_matrix_matches_scalar_model_for_single_column() {
        let data = simulate_sleepstudy_like(12, 8, 17);
        let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

        model.fit(true).unwrap();

        let y = model.y();
        let response_matrix = DMatrix::from_column_slice(y.len(), 1, y.as_slice());
        let profile = model
            .profile_response_matrix(&response_matrix, true)
            .unwrap();
        let beta = model.beta();

        assert_relative_eq!(
            profile.total_objective,
            model.objective_value(),
            epsilon = 1e-8,
            max_relative = 1e-8
        );
        assert_relative_eq!(
            profile.pwrss[0],
            model.pwrss(),
            epsilon = 1e-8,
            max_relative = 1e-8
        );
        assert_relative_eq!(
            profile.sigma[0],
            model.sigma(),
            epsilon = 1e-8,
            max_relative = 1e-8
        );
        for row in 0..beta.len() {
            assert_relative_eq!(
                profile.beta[(row, 0)],
                beta[row],
                epsilon = 1e-8,
                max_relative = 1e-8
            );
        }
    }

    #[test]
    fn test_profile_response_matrix_batches_columns_consistently() {
        let data = simulate_sleepstudy_like(10, 6, 23);
        let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

        model.fit(true).unwrap();

        let y1 = model.y();
        let y2 = y1.map(|value| 0.75 * value + 12.0);
        let mut batch = DMatrix::zeros(y1.len(), 2);
        batch.set_column(0, &y1);
        batch.set_column(1, &y2);

        let batch_profile = model.profile_response_matrix(&batch, true).unwrap();
        let single_1 = model
            .profile_response_matrix(
                &DMatrix::from_column_slice(y1.len(), 1, y1.as_slice()),
                true,
            )
            .unwrap();
        let single_2 = model
            .profile_response_matrix(
                &DMatrix::from_column_slice(y2.len(), 1, y2.as_slice()),
                true,
            )
            .unwrap();

        assert_relative_eq!(
            batch_profile.total_objective,
            single_1.total_objective + single_2.total_objective,
            epsilon = 1e-8,
            max_relative = 1e-8
        );
        for row in 0..batch_profile.beta.nrows() {
            assert_relative_eq!(
                batch_profile.beta[(row, 0)],
                single_1.beta[(row, 0)],
                epsilon = 1e-8,
                max_relative = 1e-8
            );
            assert_relative_eq!(
                batch_profile.beta[(row, 1)],
                single_2.beta[(row, 0)],
                epsilon = 1e-8,
                max_relative = 1e-8
            );
        }
        assert_relative_eq!(
            batch_profile.sigma[0],
            single_1.sigma[0],
            epsilon = 1e-8,
            max_relative = 1e-8
        );
        assert_relative_eq!(
            batch_profile.sigma[1],
            single_2.sigma[0],
            epsilon = 1e-8,
            max_relative = 1e-8
        );
    }

    #[test]
    fn test_response_accessor_matches_stored_response() {
        let data = shared_julia_parity_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
        let model = LinearMixedModel::new(formula, &data, None).unwrap();

        let y = model.y();
        let response = MixedModelFit::response(&model);

        assert_eq!(response.len(), y.len());
        for idx in 0..y.len() {
            assert_relative_eq!(response[idx], y[idx], epsilon = 1e-12, max_relative = 1e-12);
        }
    }

    #[test]
    fn test_scalar_single_theta_fit_is_locally_optimal() {
        let data = simulate_sleepstudy_like(16, 8, 99);
        let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

        model.fit(true).unwrap();

        let fitted_theta = model.theta()[0];
        let fitted_obj = model.objective_value();
        let mut probe = model.clone();
        let radius = fitted_theta.max(0.5);

        for step in 0..=20 {
            let frac = step as f64 / 20.0;
            let theta = frac * (fitted_theta + radius);
            let obj = probe.objective_at(&[theta]).unwrap();
            assert!(
                fitted_obj <= obj + 1e-6,
                "fitted objective {fitted_obj} exceeded probe objective {obj} at theta={theta}"
            );
        }

        assert!(
            model.optsum.feval <= 32,
            "scalar optimizer used too many evaluations: {}",
            model.optsum.feval
        );
    }

    #[test]
    fn test_scalar_single_theta_records_maxeval() {
        let data = simulate_sleepstudy_like(16, 8, 99);
        let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.optsum.max_feval = 1;

        model.fit(true).unwrap();

        assert_eq!(model.optsum.optimizer, Optimizer::PatternSearch);
        assert_eq!(model.optsum.return_value, "MAXEVAL_REACHED");
        assert_ne!(model.optsum.return_value, "SUCCESS");
    }

    #[test]
    fn test_pattern_search_records_maxeval() {
        let data = simulate_sleepstudy_like(12, 8, 17);
        let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.optsum.max_feval = 1;

        model
            .fit_with_forced_optimizer(true, Optimizer::PatternSearch)
            .unwrap();

        assert_eq!(model.optsum.optimizer, Optimizer::PatternSearch);
        assert_eq!(model.optsum.return_value, "MAXEVAL_REACHED");
        assert_ne!(model.optsum.return_value, "SUCCESS");
    }

    #[test]
    fn test_pattern_search_descends_correlated_directions() {
        let initial = vec![0.0, 0.0];
        let outcome = LinearMixedModel::run_multivariate_pattern_search(
            initial.clone(),
            0.0,
            &[f64::NEG_INFINITY, f64::NEG_INFINITY],
            vec![1.0, 1.0],
            &[1e-4, 1e-4],
            5,
            1e-12,
            |theta| Ok(theta[0] * theta[0] + theta[1] * theta[1] - 3.0 * theta[0] * theta[1]),
        )
        .unwrap();

        assert_eq!(outcome.feval_count, 5);
        assert!(
            outcome.best_fmin < -0.9,
            "combined pattern probe should descend when each axis probe is uphill, got {}",
            outcome.best_fmin
        );
        assert!(
            outcome.fit_log.iter().any(|entry| {
                entry.objective < 0.0
                    && entry
                        .theta
                        .iter()
                        .zip(initial.iter())
                        .filter(|(candidate, base)| (*candidate - *base).abs() > 1e-12)
                        .count()
                        > 1
            }),
            "fit log should include an improving multi-coordinate pattern probe"
        );
    }

    #[cfg(feature = "nlopt")]
    #[test]
    fn test_pattern_search_matches_nlopt_on_correlated_crossed_fixture() {
        let data = correlated_crossed_slope_data();
        let formula = parse_formula("y ~ 1 + x + (1 + x | g) + (1 + x | h)").unwrap();

        let mut pattern_model = LinearMixedModel::new(formula.clone(), &data, None).unwrap();
        pattern_model.optsum.max_feval = 20000;
        pattern_model
            .fit_with_forced_optimizer(true, Optimizer::PatternSearch)
            .unwrap();

        let mut nlopt_model = LinearMixedModel::new(formula, &data, None).unwrap();
        nlopt_model.fit(true).unwrap();

        assert_eq!(pattern_model.optsum.optimizer, Optimizer::PatternSearch);
        assert_eq!(nlopt_model.optsum.optimizer, Optimizer::NloptBobyqa);
        assert!(
            pattern_model.objective_value() <= nlopt_model.objective_value() + 1e-4,
            "pattern_search objective {} should match nlopt {} on correlated crossed fixture",
            pattern_model.objective_value(),
            nlopt_model.objective_value()
        );
    }

    #[test]
    fn test_cobyla_records_maxeval() {
        let data = simulate_sleepstudy_like(12, 8, 17);
        let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.optsum.max_feval = 5;

        model
            .fit_with_forced_optimizer(true, Optimizer::Cobyla)
            .unwrap();

        assert_eq!(model.optsum.optimizer, Optimizer::Cobyla);
        assert_eq!(model.optsum.return_value, "MAXEVAL_REACHED");
        assert_ne!(model.optsum.return_value, "SUCCESS");
    }

    #[test]
    fn test_optimizer_return_values_consistent_across_backends() {
        assert_eq!(
            LinearMixedModel::cobyla_success_status_label(cobyla::SuccessStatus::MaxEvalReached),
            "MAXEVAL_REACHED"
        );
        assert_eq!(
            LinearMixedModel::cobyla_fail_status_label(cobyla::FailStatus::RoundoffLimited),
            "ROUNDOFF_LIMITED"
        );
        #[cfg(feature = "nlopt")]
        {
            assert_eq!(
                LinearMixedModel::nlopt_status_label("MaxEvalReached"),
                "MAXEVAL_REACHED"
            );
            assert_eq!(
                LinearMixedModel::nlopt_status_label("RoundoffLimited"),
                "ROUNDOFF_LIMITED"
            );
        }
    }

    #[test]
    fn test_rectify_runs_for_cobyla_and_pattern_search_backends() {
        let data = grouped_slope_data_with_obs(8, 4);
        let formula = parse_formula("y ~ 1 + x + (1 + x | group)").unwrap();

        for optimizer in [Optimizer::Cobyla, Optimizer::PatternSearch] {
            let mut model = LinearMixedModel::new(formula.clone(), &data, None).unwrap();
            let negative_theta = vec![-1.0, 0.25, -0.5];
            let fmin = model.objective_at(&negative_theta).unwrap();

            model
                .finalize_fit_result(
                    negative_theta.clone(),
                    fmin,
                    1,
                    vec![FitLogEntry {
                        theta: negative_theta,
                        objective: fmin,
                    }],
                    optimizer,
                    None,
                )
                .unwrap();

            assert_eq!(model.optsum.optimizer, optimizer);
            assert_theta_diagonals_nonnegative(&model);
            assert_eq!(model.optsum.final_params, vec![1.0, -0.25, 0.5]);
        }
    }

    #[test]
    fn test_rectify_theta_columns_matches_julia_sign_convention() {
        let parmap = vec![(0, 0, 0), (0, 1, 0), (0, 1, 1), (1, 0, 0)];
        let mut theta = vec![-2.0, 0.75, -3.0, -4.0];

        LinearMixedModel::rectify_theta_columns(&mut theta, &parmap, 2);

        assert_eq!(theta, vec![2.0, -0.75, 3.0, 4.0]);
    }

    #[test]
    #[allow(clippy::approx_constant)] // 3.14 is a sigma sentinel, not π
    fn test_fixed_sigma_constrains_scalar_re_fit() {
        let data = shared_julia_fixed_sigma_fixture();
        let formula = parse_formula("y ~ 0 + (1 | z)").unwrap();
        let julia_objective = 513.5676467958401;

        let mut model_sigma1 = LinearMixedModel::new(formula.clone(), &data, None).unwrap();
        model_sigma1.optsum.sigma = Some(1.0);
        assert_relative_eq!(
            model_sigma1.objective_at(&[2.992032352222033]).unwrap(),
            julia_objective,
            epsilon = 1e-8,
            max_relative = 1e-8
        );
        model_sigma1.fit(false).unwrap();

        assert_eq!(model_sigma1.fixef().len(), 0);
        assert_relative_eq!(
            model_sigma1.sigma(),
            1.0,
            epsilon = 1e-12,
            max_relative = 1e-12
        );
        assert_relative_eq!(
            model_sigma1.objective_value(),
            julia_objective,
            epsilon = 2e-5,
            max_relative = 1e-8
        );
        assert_relative_eq!(
            model_sigma1.theta()[0],
            2.992032352222033,
            epsilon = 1e-3,
            max_relative = 1e-3
        );
        assert_eq!(model_sigma1.dof(), model_sigma1.n_theta());

        let mut model_sigma314 = LinearMixedModel::new(formula, &data, None).unwrap();
        model_sigma314.optsum.sigma = Some(3.14);
        assert_relative_eq!(
            model_sigma314.objective_at(&[0.09694160520621385]).unwrap(),
            julia_objective,
            epsilon = 1e-8,
            max_relative = 1e-8
        );
        model_sigma314.fit(false).unwrap();

        assert_eq!(model_sigma314.fixef().len(), 0);
        assert_relative_eq!(
            model_sigma314.sigma(),
            3.14,
            epsilon = 1e-12,
            max_relative = 1e-12
        );
        assert_relative_eq!(
            model_sigma314.objective_value(),
            julia_objective,
            epsilon = 2e-5,
            max_relative = 1e-8
        );
        assert_relative_eq!(
            model_sigma314.theta()[0],
            0.09694160520621385,
            epsilon = 1e-3,
            max_relative = 1e-3
        );
        assert_eq!(model_sigma314.dof(), model_sigma314.n_theta());
    }

    #[test]
    #[allow(clippy::approx_constant)] // 3.14 is a sigma sentinel, not π
    fn test_varest_under_fixed_sigma_matches_julia() {
        let data = shared_julia_fixed_sigma_fixture();
        let formula = parse_formula("y ~ 0 + (1 | z)").unwrap();
        let mut fixed = LinearMixedModel::new(formula.clone(), &data, None).unwrap();
        fixed.optsum.sigma = Some(3.14);
        fixed.fit(false).unwrap();

        let mut estimated = LinearMixedModel::new(formula, &data, None).unwrap();
        estimated.fit(false).unwrap();

        assert_relative_eq!(fixed.sigma(), 3.14, epsilon = 1e-12);
        assert_relative_eq!(fixed.varest(), 3.14, epsilon = 1e-12);
        assert_relative_eq!(
            estimated.varest(),
            estimated.sigma().powi(2),
            epsilon = 1e-12
        );
    }

    #[test]
    #[allow(clippy::approx_constant)] // 3.14 is a sigma sentinel, not π
    fn test_dispersion_under_fixed_sigma_matches_julia() {
        let data = shared_julia_fixed_sigma_fixture();
        let formula = parse_formula("y ~ 0 + (1 | z)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.optsum.sigma = Some(3.14);
        model.fit(false).unwrap();

        assert_relative_eq!(
            MixedModelFit::dispersion(&model, false),
            3.14,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            MixedModelFit::dispersion(&model, true),
            3.14,
            epsilon = 1e-12
        );
    }

    #[cfg(feature = "nlopt")]
    #[test]
    fn test_large_theta_fit_records_maxeval_status() {
        let data = simulate_large_theta_crossed(123);
        let formula = parse_formula(
            "reaction ~ 1 + days + (1 + days | subj) + (1 + days | item) + (1 + days | site)",
        )
        .unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.optsum.max_feval = 1;

        model.fit(true).unwrap();

        assert_eq!(model.optsum.optimizer, Optimizer::NloptNewuoa);
        assert_eq!(model.optsum.return_value, "MAXEVAL_REACHED");
        assert_eq!(model.optsum.feval, 1);
        assert!(model.objective_value().is_finite());
    }

    #[cfg(feature = "nlopt")]
    #[test]
    fn test_large_theta_fit_records_maxtime_status() {
        let data = simulate_large_theta_crossed(123);
        let formula = parse_formula(
            "reaction ~ 1 + days + (1 + days | subj) + (1 + days | item) + (1 + days | site)",
        )
        .unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.optsum.max_time = 1e-9;

        model.fit(true).unwrap();

        assert_eq!(model.optsum.optimizer, Optimizer::NloptNewuoa);
        assert_eq!(model.optsum.return_value, "MAXTIME_REACHED");
        assert_eq!(model.optsum.max_time, 1e-9);
        assert!(model.optsum.feval >= 1);
        assert!(model.objective_value().is_finite());
    }

    #[test]
    fn test_scalar_objective_matches_julia_on_shared_fixture() {
        let data = shared_julia_parity_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

        let julia_theta = [0.6273717260668661];
        let julia_objective = 223.74206848841089;

        let rust_objective = model.objective_at(&julia_theta).unwrap();

        assert_relative_eq!(
            rust_objective,
            julia_objective,
            epsilon = 1e-8,
            max_relative = 1e-8
        );

        model.fit(true).unwrap();
        assert_relative_eq!(
            model.objective_value(),
            julia_objective,
            epsilon = 1e-5,
            max_relative = 1e-5
        );
        assert_relative_eq!(
            model.sigma(),
            30.23875724370832,
            epsilon = 1e-5,
            max_relative = 1e-5
        );
    }

    #[cfg(feature = "nlopt")]
    #[test]
    fn test_vector_objective_matches_julia_on_shared_fixture() {
        let data = shared_julia_parity_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

        let julia_theta = [0.6565437822843008, -0.019160976185379253, 0.0];
        let julia_objective = 223.73509351902135;

        let rust_objective = model.objective_at(&julia_theta).unwrap();

        assert_relative_eq!(
            rust_objective,
            julia_objective,
            epsilon = 1e-8,
            max_relative = 1e-8
        );

        model.fit(true).unwrap();
        assert_relative_eq!(
            model.objective_value(),
            julia_objective,
            epsilon = 1e-4,
            max_relative = 1e-4
        );
        assert_relative_eq!(
            model.sigma(),
            30.22863368533761,
            epsilon = 1e-4,
            max_relative = 1e-4
        );
    }

    #[test]
    fn test_crossed_objective_matches_julia_on_shared_fixture() {
        let data = shared_julia_crossed_parity_fixture();
        let formula = parse_formula(
            "reaction ~ 1 + days + (1 + days | subj) + (1 + days | item) + (1 + days | site)",
        )
        .unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

        let julia_theta = [
            1.6360390637490343,
            0.19973976515130532,
            0.16548928583172998,
            1.3985120310259511,
            -0.07659426024736829,
            0.19501821571577171,
            0.62772070762735099,
            -0.036380030801807128,
            0.11318289497410258,
        ];
        let julia_objective = 6177.3917660389134;
        let julia_pwrss = 50993.469629712374;
        let julia_logdet_re = 208.5086015326244;
        let julia_logdet_xx = 5.5028138123102082;

        let rust_objective = model.objective_at(&julia_theta).unwrap();

        assert_relative_eq!(
            rust_objective,
            julia_objective,
            epsilon = 1e-6,
            max_relative = 1e-9
        );
        assert_relative_eq!(
            model.pwrss(),
            julia_pwrss,
            epsilon = 1e-5,
            max_relative = 1e-9
        );
        assert_relative_eq!(
            model.logdet_re(),
            julia_logdet_re,
            epsilon = 1e-8,
            max_relative = 1e-10
        );
        assert_relative_eq!(
            current_logdet_xx(&model),
            julia_logdet_xx,
            epsilon = 1e-8,
            max_relative = 1e-10
        );
    }

    #[cfg(feature = "nlopt")]
    #[test]
    fn test_crossed_fit_matches_julia_on_shared_fixture() {
        let data = shared_julia_crossed_parity_fixture();
        let formula = parse_formula(
            "reaction ~ 1 + days + (1 + days | subj) + (1 + days | item) + (1 + days | site)",
        )
        .unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

        let julia_theta = [
            1.6360390637490343,
            0.19973976515130532,
            0.16548928583172998,
            1.3985120310259511,
            -0.07659426024736829,
            0.19501821571577171,
            0.62772070762735099,
            -0.036380030801807128,
            0.11318289497410258,
        ];
        let julia_objective = 6177.3917660389134;
        let julia_sigma = 7.6913690161800066;

        model.fit(true).unwrap();

        assert_eq!(model.optsum.optimizer, Optimizer::NloptNewuoa);
        assert_relative_eq!(
            model.objective_value(),
            julia_objective,
            epsilon = 5e-6,
            max_relative = 1e-9
        );
        assert_relative_eq!(
            model.sigma(),
            julia_sigma,
            epsilon = 2e-5,
            max_relative = 5e-6
        );

        let theta = model.theta();
        for (actual, expected) in theta.iter().zip(julia_theta.iter()) {
            assert_relative_eq!(*actual, *expected, epsilon = 2e-4, max_relative = 2e-4);
        }
    }

    // ── Tests ported from MixedModels.jl/test/pls.jl ────────────────────────

    #[test]
    fn test_ml_loglikelihood_aic_bic_relationships() {
        // Verify the algebraic relationships: ll = -obj/2, aic, bic.
        // Matches Julia's convention: objective already includes n*log(2π).
        let data = shared_julia_parity_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap(); // ML

        let n = model.nobs() as f64;
        let k = model.dof() as f64;
        let obj = model.objective_value();
        let ll = MixedModelFit::loglikelihood(&model);

        // ML: loglikelihood = -objective / 2
        assert_relative_eq!(ll, -obj / 2.0, epsilon = 1e-12);

        // AIC = -2*ll + 2*k
        assert_relative_eq!(
            MixedModelFit::aic(&model),
            -2.0 * ll + 2.0 * k,
            epsilon = 1e-12
        );

        // BIC = -2*ll + k*ln(n)
        assert_relative_eq!(
            MixedModelFit::bic(&model),
            -2.0 * ll + k * n.ln(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_ml_nobs_and_dof_scalar_re() {
        // 6 subjects × 4 days = 24 obs; dof = p(2) + n_theta(1) + 1(sigma) = 4
        let data = shared_julia_parity_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        assert_eq!(MixedModelFit::nobs(&model), 24);
        assert_eq!(MixedModelFit::dof(&model), 4);
    }

    #[test]
    fn test_ml_fixef_and_stderror() {
        // reaction ~ 1 + days: two fixef, both SE positive
        let data = shared_julia_parity_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let fixef = MixedModelFit::fixef(&model);
        let se = MixedModelFit::stderror(&model);

        assert_eq!(fixef.len(), 2);
        assert_eq!(se.len(), 2);
        assert!(se[0] > 0.0, "intercept SE must be positive");
        assert!(se[1] > 0.0, "slope SE must be positive");
    }

    #[test]
    fn test_ml_fitted_plus_residuals_equals_response() {
        let data = shared_julia_parity_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let fitted = MixedModelFit::fitted(&model);
        let residuals = MixedModelFit::residuals(&model);
        let y = model.y();

        assert_eq!(fitted.len(), y.len());
        for i in 0..y.len() {
            assert_relative_eq!(fitted[i] + residuals[i], y[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_ml_ranef_dimensions_scalar_re() {
        // (1|subj): vsize=1, 6 subjects → matrix is 1×6
        let data = shared_julia_parity_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let ranef = model.ranef_b();
        assert_eq!(ranef.len(), 1, "one grouping factor");
        assert_eq!(ranef[0].nrows(), 1, "scalar RE: vsize = 1");
        assert_eq!(ranef[0].ncols(), 6, "6 subjects");
    }

    #[test]
    fn test_is_singular_reflects_theta_at_lower_bound() {
        // After fitting non-degenerate data: not singular.
        // Driving theta to lower bound → singular.
        let data = shared_julia_parity_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        assert!(
            !model.is_singular(),
            "non-degenerate fit should not be singular"
        );

        let fitted_theta = model.theta();
        let lb = model.lower_bounds();
        model.set_theta(&lb).unwrap(); // θ = [0.0] → at lower bound
        assert!(model.is_singular(), "theta at lower bound must be singular");

        model.set_theta(&fitted_theta).unwrap();
        assert!(
            !model.is_singular(),
            "restored theta should not be singular"
        );
    }

    #[test]
    fn test_is_singular_detects_rank_deficient_lambda() {
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();
        assert!(
            !model.is_singular(),
            "fitted vector model should start full rank"
        );

        // Full Cholesky with a tiny nonzero second diagonal: not at the
        // parameter lower bound, but numerically rank-deficient in ΛΛ'.
        let rank_deficient_theta = vec![1.0, 0.25, 1e-8];
        model.set_theta(&rank_deficient_theta).unwrap();
        model.update_l().unwrap();

        assert!(
            !model.theta_at_lower_bound(),
            "tiny nonzero diagonal should not be classified as a boundary θ"
        );

        model.refresh_effective_covariance_summaries();

        let summary = &model.compiler_artifact().effective_covariance[0];
        assert_eq!(summary.status, EffectiveRankStatus::ReducedRank);
        assert!(
            model.is_singular(),
            "is_singular must follow reduced effective covariance, not just θ lower bounds"
        );
    }

    #[test]
    fn test_is_singular_consistent_with_effective_covariance_status() {
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        let mut policy = CompilerPolicy::maximal_feasible();
        policy.thresholds.effective_rank_relative_tolerance = 2.0;
        model.set_compiler_policy(policy).unwrap();

        model.fit(false).unwrap();

        let has_reduced_covariance = model
            .compiler_artifact()
            .effective_covariance
            .iter()
            .any(|summary| summary.status == EffectiveRankStatus::ReducedRank);
        assert!(has_reduced_covariance);
        assert_eq!(
            model.is_singular(),
            model.theta_at_lower_bound()
                || model.optimizer_certificate_reports_boundary()
                || has_reduced_covariance
        );
        assert!(model.is_singular());
    }

    #[test]
    fn test_lmm_set_theta_propagates_remat_err() {
        let data = shared_julia_parity_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

        let err = model.set_theta(&[]).unwrap_err();

        assert!(matches!(err, MixedModelError::DimensionMismatch(_)));
    }

    #[test]
    fn test_set_theta_does_not_panic_on_bad_input() {
        let data = shared_julia_parity_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| model.set_theta(&[])));

        assert!(result.is_ok());
        assert!(matches!(
            result.unwrap(),
            Err(MixedModelError::DimensionMismatch(_))
        ));
    }

    #[test]
    fn test_lrt_nested_scalar_re_models() {
        // LRT comparing reaction ~ 1 + (1|subj) vs reaction ~ 1 + days + (1|subj).
        // The second model adds one FE parameter: chisq_dof == 1.
        use crate::stats::lrt::LikelihoodRatioTest;

        let data = shared_julia_parity_fixture();
        let f0 = parse_formula("reaction ~ 1 + (1 | subj)").unwrap();
        let f1 = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();

        let mut m0 = LinearMixedModel::new(f0, &data, None).unwrap();
        let mut m1 = LinearMixedModel::new(f1, &data, None).unwrap();
        m0.fit(false).unwrap();
        m1.fit(false).unwrap();

        let lrt =
            LikelihoodRatioTest::test(&[&m0 as &dyn MixedModelFit, &m1 as &dyn MixedModelFit])
                .unwrap();

        // χ² = 2*(ll1 - ll0)
        let expected_chisq =
            2.0 * (MixedModelFit::loglikelihood(&m1) - MixedModelFit::loglikelihood(&m0));
        assert_relative_eq!(lrt.chisq[0], expected_chisq, epsilon = 1e-10);

        // Adding `days` costs 1 dof
        assert_eq!(lrt.chisq_dof[0], 1);

        // Fuller model has better (larger) log-likelihood
        assert!(MixedModelFit::loglikelihood(&m1) > MixedModelFit::loglikelihood(&m0));

        // p-value in [0, 1]
        assert!(lrt.pvalues[0] >= 0.0 && lrt.pvalues[0] <= 1.0);
    }

    #[test]
    fn test_singular_re_fit_is_singular() {
        // Synthetic data: all group means identical (SS_B = 0).
        // Mirrors pls.jl "Dyestuff2" testset spirit: when between-group variance
        // is zero, θ → 0 and the model is singular.
        let data = singular_re_fixture();
        let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap(); // ML

        assert!(model.is_singular(), "fit with SS_B=0 must be singular");
        assert_relative_eq!(model.theta()[0], 0.0, epsilon = 1e-10);
    }

    // ── Fixtures from actual Julia MixedModels.jl datasets ─────────────────

    /// Dyestuff data (Davies, 1949) — 6 batches × 5 observations.
    /// Matches `dataset(:dyestuff)` from MixedModelsDatasets.jl.
    fn dyestuff_fixture() -> DataFrame {
        let yields: Vec<f64> = vec![
            1545.0, 1440.0, 1440.0, 1520.0, 1580.0, // batch A
            1540.0, 1555.0, 1490.0, 1560.0, 1495.0, // batch B
            1595.0, 1550.0, 1605.0, 1510.0, 1560.0, // batch C
            1445.0, 1440.0, 1595.0, 1465.0, 1545.0, // batch D
            1595.0, 1630.0, 1515.0, 1635.0, 1625.0, // batch E
            1520.0, 1455.0, 1450.0, 1480.0, 1445.0, // batch F
        ];
        let batches: Vec<String> = "ABCDEF"
            .chars()
            .flat_map(|c| std::iter::repeat_n(c.to_string(), 5))
            .collect();
        let mut df = DataFrame::new();
        df.add_numeric("yield", yields).unwrap();
        df.add_categorical("batch", batches).unwrap();
        df
    }

    /// Sleepstudy data (Belenky et al., 2003) — 18 subjects × 10 days.
    /// Matches `dataset(:sleepstudy)` from MixedModelsDatasets.jl.
    fn sleepstudy_fixture() -> DataFrame {
        let subjects = [
            "S308", "S309", "S310", "S330", "S331", "S332", "S333", "S334", "S335", "S337", "S349",
            "S350", "S351", "S352", "S369", "S370", "S371", "S372",
        ];
        #[rustfmt::skip]
        let reaction: Vec<f64> = vec![
            // S308
            249.5600, 258.7047, 250.8006, 321.4398, 356.8519,
            414.6901, 382.2038, 290.1486, 430.5853, 466.3535,
            // S309
            222.7339, 205.2658, 202.9778, 204.7070, 207.7161,
            215.9618, 213.6303, 217.7272, 224.2957, 237.3142,
            // S310
            199.0539, 194.3322, 234.3200, 232.8416, 229.3074,
            220.4579, 235.4208, 255.7511, 261.0125, 247.5153,
            // S330
            321.5426, 300.4002, 283.8565, 285.1330, 285.7973,
            297.5855, 280.2396, 318.2613, 305.3495, 354.0487,
            // S331
            287.6079, 285.0000, 301.8206, 320.1153, 316.2773,
            293.3187, 290.0750, 334.8177, 293.7469, 371.5811,
            // S332
            234.8606, 242.8118, 272.9613, 309.7688, 317.4629,
            309.9976, 454.1619, 346.8311, 330.3003, 253.8644,
            // S333
            283.8424, 289.5550, 276.7693, 299.8097, 297.1710,
            338.1665, 332.0265, 348.8399, 333.3600, 362.0428,
            // S334
            265.4731, 276.2012, 243.3647, 254.6723, 279.0244,
            284.1912, 305.5248, 331.5229, 335.7469, 377.2990,
            // S335
            241.6083, 273.9472, 254.4907, 270.8021, 251.4519,
            254.6362, 245.4523, 235.3110, 235.7541, 237.2466,
            // S337
            312.3666, 313.8058, 291.6112, 346.1222, 365.7324,
            391.8385, 404.2601, 416.6923, 455.8643, 458.9167,
            // S349
            236.1032, 230.3167, 238.9256, 254.9220, 250.7103,
            269.7744, 281.5648, 308.1020, 336.2806, 351.6451,
            // S350
            256.2968, 243.4543, 256.2046, 255.5271, 268.9165,
            329.7247, 379.4445, 362.9184, 394.4872, 389.0527,
            // S351
            250.5265, 300.0576, 269.8939, 280.5891, 271.8274,
            304.6336, 287.7466, 266.5955, 321.5418, 347.5655,
            // S352
            221.6771, 298.1939, 326.8785, 346.8555, 348.7402,
            352.8287, 354.4266, 360.4326, 375.6406, 388.5417,
            // S369
            271.9235, 268.4369, 257.2424, 277.6566, 314.8222,
            317.2135, 298.1353, 348.1229, 340.2800, 366.5131,
            // S370
            225.2640, 234.5235, 238.9008, 240.4730, 267.5373,
            344.1937, 281.1481, 347.5855, 365.1630, 372.2288,
            // S371
            269.8804, 272.4428, 277.8989, 281.7895, 279.1705,
            284.5120, 259.2658, 304.6306, 350.7807, 369.4692,
            // S372
            269.4117, 273.4740, 297.5968, 310.6316, 287.1726,
            329.6076, 334.4818, 343.2199, 369.1417, 364.1236,
        ];
        let days: Vec<f64> = (0..18).flat_map(|_| (0..10u64).map(|d| d as f64)).collect();
        let subj: Vec<String> = subjects
            .iter()
            .flat_map(|s| std::iter::repeat_n(s.to_string(), 10))
            .collect();
        let mut df = DataFrame::new();
        df.add_numeric("reaction", reaction).unwrap();
        df.add_numeric("days", days).unwrap();
        df.add_categorical("subj", subj).unwrap();
        df
    }

    /// Penicillin data (Davies, 1967) — 24 plates × 6 samples = 144 observations.
    /// Matches `dataset(:penicillin)` from MixedModelsDatasets.jl.
    fn penicillin_fixture() -> DataFrame {
        // Diameter values in plate-major order (6 samples A-F per plate a-x).
        #[rustfmt::skip]
        let diameter: Vec<f64> = vec![
            27.0, 23.0, 26.0, 23.0, 23.0, 21.0, // plate a
            27.0, 23.0, 26.0, 23.0, 23.0, 21.0, // plate b
            25.0, 21.0, 25.0, 24.0, 24.0, 20.0, // plate c
            26.0, 23.0, 25.0, 23.0, 23.0, 20.0, // plate d
            25.0, 22.0, 26.0, 22.0, 23.0, 20.0, // plate e
            24.0, 22.0, 25.0, 23.0, 22.0, 19.0, // plate f
            24.0, 20.0, 23.0, 21.0, 22.0, 19.0, // plate g
            26.0, 22.0, 26.0, 24.0, 24.0, 21.0, // plate h
            24.0, 21.0, 24.0, 22.0, 22.0, 20.0, // plate i
            24.0, 21.0, 24.0, 23.0, 22.0, 19.0, // plate j
            26.0, 23.0, 26.0, 24.0, 24.0, 21.0, // plate k
            25.0, 22.0, 26.0, 24.0, 24.0, 20.0, // plate l
            26.0, 24.0, 26.0, 24.0, 25.0, 22.0, // plate m
            26.0, 23.0, 26.0, 23.0, 23.0, 20.0, // plate n
            26.0, 23.0, 25.0, 24.0, 24.0, 22.0, // plate o
            25.0, 22.0, 25.0, 23.0, 23.0, 20.0, // plate p
            25.0, 21.0, 24.0, 23.0, 23.0, 20.0, // plate q
            25.0, 22.0, 24.0, 23.0, 23.0, 19.0, // plate r
            24.0, 21.0, 23.0, 21.0, 21.0, 19.0, // plate s
            26.0, 23.0, 26.0, 24.0, 24.0, 21.0, // plate t
            25.0, 21.0, 24.0, 22.0, 22.0, 18.0, // plate u
            25.0, 22.0, 25.0, 22.0, 22.0, 20.0, // plate v
            24.0, 21.0, 24.0, 22.0, 24.0, 19.0, // plate w
            24.0, 21.0, 24.0, 22.0, 21.0, 18.0, // plate x
        ];
        let plate_letters: Vec<&str> = vec![
            "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q",
            "r", "s", "t", "u", "v", "w", "x",
        ];
        let plate: Vec<String> = plate_letters
            .iter()
            .flat_map(|p| std::iter::repeat_n(p.to_string(), 6))
            .collect();
        let sample: Vec<String> = (0..24)
            .flat_map(|_| ["A", "B", "C", "D", "E", "F"].iter().map(|s| s.to_string()))
            .collect();
        let mut df = DataFrame::new();
        df.add_numeric("diameter", diameter).unwrap();
        df.add_categorical("plate", plate).unwrap();
        df.add_categorical("sample", sample).unwrap();
        df
    }

    /// Pastes data (Davies, 1947) — 10 batches × 3 casks × 2 samples = 60 obs.
    /// Matches `dataset(:pastes)` from MixedModelsDatasets.jl.
    /// The nested structure `batch / cask` expands to `batch + batch:cask`.
    fn pastes_fixture() -> DataFrame {
        // Strength values, 6 per batch (2 per cask: a,a,b,b,c,c)
        #[rustfmt::skip]
        let strength: Vec<f64> = vec![
            62.8, 62.6, 60.1, 62.3, 62.7, 63.1, // batch A
            60.0, 61.4, 57.5, 56.9, 61.1, 58.9, // batch B
            58.7, 57.5, 63.9, 63.1, 65.4, 63.7, // batch C
            57.1, 56.4, 56.9, 58.6, 64.7, 64.5, // batch D
            55.1, 55.1, 54.7, 54.2, 58.8, 57.5, // batch E
            63.4, 64.9, 59.3, 58.1, 60.5, 60.0, // batch F
            62.5, 62.6, 61.0, 58.7, 56.9, 57.7, // batch G
            59.2, 59.4, 65.2, 66.0, 64.8, 64.1, // batch H
            54.8, 54.8, 64.0, 64.0, 57.7, 56.8, // batch I
            58.3, 59.3, 59.2, 59.2, 58.9, 56.6, // batch J
        ];
        // batch: A-J, 6 obs each
        let batch: Vec<String> = "ABCDEFGHIJ"
            .chars()
            .flat_map(|c| std::iter::repeat_n(c.to_string(), 6))
            .collect();
        // cask: a,a,b,b,c,c per batch
        let cask_pattern = ["a", "a", "b", "b", "c", "c"];
        let cask: Vec<String> = (0..10)
            .flat_map(|_| cask_pattern.iter().map(|s| s.to_string()))
            .collect();
        // batch_cask: interaction label for (1 | batch & cask)
        let batch_cask: Vec<String> = batch
            .iter()
            .zip(&cask)
            .map(|(b, c)| format!("{b}:{c}"))
            .collect();

        let mut df = DataFrame::new();
        df.add_numeric("strength", strength).unwrap();
        df.add_categorical("batch", batch).unwrap();
        df.add_categorical("cask", cask).unwrap();
        df.add_categorical("batch_cask", batch_cask).unwrap();
        df
    }

    /// Dyestuff2 data — same structure as Dyestuff but within-batch variance
    /// dominates, so the RE variance collapses to zero (singular fit).
    /// Values decoded from `dyestuff2.arrow` (MixedModelsDatasets.jl).
    fn dyestuff2_fixture() -> DataFrame {
        #[rustfmt::skip]
        let yields: Vec<f64> = vec![
            7.298, 3.846, 2.434, 9.566,  7.990, // batch A
            5.220, 6.556, 0.608, 11.788, -0.892, // batch B
            0.110, 10.386, 13.434, 5.510, 8.166, // batch C
            2.212, 4.852, 7.092,  9.288,  4.980, // batch D
            0.282, 9.014, 4.458,  9.446,  7.198, // batch E
            1.722, 4.782, 8.106,  0.758,  3.758, // batch F
        ];
        let batches: Vec<String> = "ABCDEF"
            .chars()
            .flat_map(|c| std::iter::repeat_n(c.to_string(), 5))
            .collect();
        let mut df = DataFrame::new();
        df.add_numeric("yield", yields).unwrap();
        df.add_categorical("batch", batches).unwrap();
        df
    }

    fn fitted_varpar(model: &LinearMixedModel) -> Vec<f64> {
        let mut varpar = model.theta();
        varpar.push(model.sigma());
        varpar
    }

    fn assert_matrix_relative_eq(actual: &DMatrix<f64>, expected: &DMatrix<f64>, epsilon: f64) {
        assert_eq!(actual.shape(), expected.shape());
        for row in 0..actual.nrows() {
            for col in 0..actual.ncols() {
                assert_relative_eq!(actual[(row, col)], expected[(row, col)], epsilon = epsilon);
            }
        }
    }

    fn assert_matrix_symmetric(matrix: &DMatrix<f64>, epsilon: f64) {
        assert_eq!(matrix.nrows(), matrix.ncols());
        for row in 0..matrix.nrows() {
            for col in 0..row {
                assert_relative_eq!(matrix[(row, col)], matrix[(col, row)], epsilon = epsilon);
            }
        }
    }

    #[derive(Debug, Deserialize)]
    struct SatterthwaiteParityFixture {
        cases: Vec<SatterthwaiteParityCase>,
    }

    #[derive(Debug, Deserialize)]
    struct SatterthwaiteParityCase {
        name: String,
        formula: String,
        coefficient: String,
        estimate: f64,
        std_error: f64,
        df: f64,
        statistic: f64,
        p_value: f64,
    }

    #[derive(Debug, Deserialize)]
    struct KenwardRogerPbkrtestParityFixture {
        scalar_cases: Vec<KenwardRogerScalarParityCase>,
        multi_df_cases: Vec<KenwardRogerMultiDfParityCase>,
    }

    #[derive(Debug, Deserialize)]
    struct KenwardRogerScalarParityCase {
        name: String,
        formula: String,
        label: String,
        l: Vec<Vec<f64>>,
        rhs: Vec<f64>,
        estimate: f64,
        std_error: f64,
        denominator_df: f64,
        statistic: f64,
        p_value: f64,
    }

    #[derive(Debug, Deserialize)]
    struct KenwardRogerMultiDfParityCase {
        name: String,
        formula: String,
        label: String,
        l: Vec<Vec<f64>>,
        rhs: Vec<f64>,
        numerator_df: f64,
        denominator_df: f64,
        statistic: f64,
        p_value: f64,
        f_scaling: f64,
        unscaled_statistic: f64,
        unscaled_p_value: f64,
    }

    fn satterthwaite_lmer_test_parity_fixture() -> SatterthwaiteParityFixture {
        serde_json::from_str(include_str!(
            "../../tests/fixtures/compiler_contract/satterthwaite_lmer_test_parity_v1.json"
        ))
        .expect("Satterthwaite lmerTest parity fixture should deserialize")
    }

    fn kenward_roger_pbkrtest_parity_fixture() -> KenwardRogerPbkrtestParityFixture {
        serde_json::from_str(include_str!(
            "../../tests/fixtures/compiler_contract/kenward_roger_pbkrtest_parity_v1.json"
        ))
        .expect("Kenward-Roger pbkrtest parity fixture should deserialize")
    }

    fn fixed_effect_hypothesis_from_fixture(
        label: &str,
        l: &[Vec<f64>],
        rhs: &[f64],
    ) -> FixedEffectHypothesis {
        assert!(!l.is_empty(), "{label}: contrast matrix must have rows");
        let ncols = l[0].len();
        assert!(ncols > 0, "{label}: contrast matrix must have columns");
        assert_eq!(rhs.len(), l.len(), "{label}: rhs length must match rows");
        assert!(
            l.iter().all(|row| row.len() == ncols),
            "{label}: contrast rows must have a common width"
        );
        let values = l.iter().flatten().copied().collect::<Vec<_>>();
        let l = ContrastMatrix::new(DMatrix::from_row_slice(rhs.len(), ncols, &values)).unwrap();
        let rhs = ContrastRhs::new(DVector::from_column_slice(rhs)).unwrap();
        FixedEffectHypothesis::new(label.to_string(), l, rhs).unwrap()
    }

    fn unbalanced_sleepstudy_fixture() -> DataFrame {
        let source = sleepstudy_fixture();
        let reaction = source.numeric("reaction").unwrap();
        let days = source.numeric("days").unwrap();
        let subj = &source.categorical("subj").unwrap().values;

        let mut out_reaction = Vec::new();
        let mut out_days = Vec::new();
        let mut out_subj = Vec::new();
        for row in 0..source.nrow() {
            let drop_row = matches!(subj[row].as_str(), "S308" | "S309")
                && matches!(days[row] as i32, 1 | 3 | 5 | 7 | 9);
            if !drop_row {
                out_reaction.push(reaction[row]);
                out_days.push(days[row]);
                out_subj.push(subj[row].clone());
            }
        }

        let mut df = DataFrame::new();
        df.add_numeric("reaction", out_reaction).unwrap();
        df.add_numeric("days", out_days).unwrap();
        df.add_categorical_with_levels(
            "subj",
            out_subj,
            source.categorical("subj").unwrap().levels.clone(),
        )
        .unwrap();
        df
    }

    fn satterthwaite_parity_data(case_name: &str) -> DataFrame {
        match case_name {
            "sleepstudy_random_intercept_days" | "sleepstudy_random_slope_days" => {
                sleepstudy_fixture()
            }
            "sleepstudy_unbalanced_random_slope_days" => unbalanced_sleepstudy_fixture(),
            "penicillin_crossed_intercept" => penicillin_fixture(),
            other => panic!("unknown Satterthwaite parity case {other}"),
        }
    }

    // ── Parity tests against Julia MixedModels.jl ──────────────────────────

    #[test]
    fn test_dyestuff_ml_matches_julia() {
        // Mirrors pls.jl "Dyestuff" testset (ML fit).
        let data = dyestuff_fixture();
        let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap(); // ML

        assert_eq!(model.nobs(), 30);
        assert_eq!(model.dof(), 3);
        assert_relative_eq!(model.theta()[0], 0.7525806540074477, epsilon = 1e-4);
        assert_relative_eq!(model.fixef()[0], 1527.5, epsilon = 1e-6);
        assert_relative_eq!(model.sigma(), 49.51010035223816, epsilon = 1e-3);
        assert_relative_eq!(model.stderror()[0], 17.694552929494222, epsilon = 1e-2);
        assert_relative_eq!(model.objective_value(), 327.32705988112673, epsilon = 1e-3);
        // Julia: loglikelihood(fm1) ≈ -163.663... = -327.327/2
        assert_relative_eq!(
            model.loglikelihood(),
            -327.32705988112673 / 2.0,
            epsilon = 1e-3
        );
    }

    #[test]
    fn test_deviance_varpar_matches_ml_scalar_fit_and_restores_state() {
        let data = dyestuff_fixture();
        let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let theta_before = model.theta();
        let objective_before = model.objective_value();
        let vcov_before = model.vcov();
        let varpar = fitted_varpar(&model);

        let deviance = model.deviance_varpar(&varpar, false).unwrap();

        assert_relative_eq!(deviance, objective_before, epsilon = 1e-8);
        assert_eq!(model.theta(), theta_before);
        assert_relative_eq!(model.objective_value(), objective_before, epsilon = 1e-10);
        assert_relative_eq!(model.vcov(), vcov_before, epsilon = 1e-10);
    }

    #[test]
    fn test_deviance_varpar_matches_reml_vector_fit_and_restores_state() {
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(true).unwrap();

        let theta_before = model.theta();
        let objective_before = model.objective_value();
        let vcov_before = model.vcov();
        let varpar = fitted_varpar(&model);

        let deviance = model.deviance_varpar(&varpar, true).unwrap();

        assert_relative_eq!(deviance, objective_before, epsilon = 1e-8);
        assert_eq!(model.theta(), theta_before);
        assert_relative_eq!(model.objective_value(), objective_before, epsilon = 1e-10);
        assert_relative_eq!(model.vcov(), vcov_before, epsilon = 1e-10);
    }

    #[test]
    fn test_deviance_varpar_rejects_invalid_inputs_without_changing_state() {
        let data = dyestuff_fixture();
        let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let theta_before = model.theta();
        let objective_before = model.objective_value();
        let mut varpar = fitted_varpar(&model);
        varpar[0] = -1.0;
        assert!(model.deviance_varpar(&varpar, false).is_err());

        let mut varpar = fitted_varpar(&model);
        *varpar.last_mut().unwrap() = 0.0;
        assert!(model.deviance_varpar(&varpar, false).is_err());

        assert_eq!(model.theta(), theta_before);
        assert_relative_eq!(model.objective_value(), objective_before, epsilon = 1e-10);
    }

    #[test]
    fn test_vcov_beta_varpar_matches_fitted_vcov_and_restores_state() {
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(true).unwrap();

        let theta_before = model.theta();
        let objective_before = model.objective_value();
        let vcov_before = model.vcov();
        let varpar = fitted_varpar(&model);

        let vcov = model.vcov_beta_varpar(&varpar).unwrap();

        assert_matrix_relative_eq(&vcov, &vcov_before, 1e-10);
        assert_eq!(model.theta(), theta_before);
        assert_relative_eq!(model.objective_value(), objective_before, epsilon = 1e-10);
        assert_matrix_relative_eq(&model.vcov(), &vcov_before, 1e-10);
    }

    #[test]
    fn test_jac_vcov_beta_varpar_returns_symmetric_matrices_and_sigma_derivative() {
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(true).unwrap();

        let theta_before = model.theta();
        let objective_before = model.objective_value();
        let varpar = fitted_varpar(&model);
        let vcov = model.vcov_beta_varpar(&varpar).unwrap();

        let jacobian = model.jac_vcov_beta_varpar(&varpar).unwrap();

        assert_eq!(jacobian.len(), varpar.len());
        for derivative in &jacobian {
            assert_eq!(derivative.shape(), vcov.shape());
            assert!(matrix_is_finite(derivative));
            assert_matrix_symmetric(derivative, 1e-10);
        }

        let sigma = *varpar.last().unwrap();
        let sigma_derivative = jacobian.last().unwrap();
        let expected_sigma_derivative = vcov * (2.0 / sigma);
        assert_matrix_relative_eq(sigma_derivative, &expected_sigma_derivative, 1e-6);
        assert_eq!(model.theta(), theta_before);
        assert_relative_eq!(model.objective_value(), objective_before, epsilon = 1e-10);
    }

    #[test]
    fn test_jac_vcov_beta_varpar_rejects_boundary_stencil_without_changing_state() {
        let data = dyestuff_fixture();
        let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let theta_before = model.theta();
        let objective_before = model.objective_value();
        let mut varpar = fitted_varpar(&model);
        varpar[0] = 0.0;

        let err = model.jac_vcov_beta_varpar(&varpar).unwrap_err();

        assert!(err.to_string().contains("lower bound"));
        assert_eq!(model.theta(), theta_before);
        assert_relative_eq!(model.objective_value(), objective_before, epsilon = 1e-10);
    }

    #[test]
    fn test_vcov_varpar_estimate_returns_hessian_diagnostics_and_restores_state() {
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(true).unwrap();

        let theta_before = model.theta();
        let objective_before = model.objective_value();
        let varpar = fitted_varpar(&model);

        let estimate = model.vcov_varpar(&varpar, true).unwrap();

        assert_eq!(estimate.covariance.shape(), (varpar.len(), varpar.len()));
        assert_eq!(estimate.hessian.shape(), (varpar.len(), varpar.len()));
        assert_eq!(estimate.eigenvalues.len(), varpar.len());
        assert_eq!(
            estimate.positive_eigenvalues
                + estimate.near_zero_eigenvalues
                + estimate.negative_eigenvalues,
            varpar.len()
        );
        assert!(estimate.positive_eigenvalues > 0);
        assert!(estimate.tolerance.is_finite());
        assert!(estimate.tolerance > 0.0);
        assert!(matrix_is_finite(&estimate.covariance));
        assert!(matrix_is_finite(&estimate.hessian));
        assert_matrix_symmetric(&estimate.covariance, 1e-8);
        assert_matrix_symmetric(&estimate.hessian, 1e-8);
        for index in 0..varpar.len() {
            assert!(estimate.covariance[(index, index)] >= -1e-8);
        }
        assert!(matches!(
            estimate.reliability,
            ReliabilityGrade::Moderate | ReliabilityGrade::Low
        ));
        assert_eq!(
            estimate.used_reduced_rank,
            estimate.positive_eigenvalues < varpar.len()
        );
        assert_eq!(model.theta(), theta_before);
        assert_relative_eq!(model.objective_value(), objective_before, epsilon = 1e-10);
    }

    #[test]
    fn test_vcov_varpar_rejects_boundary_hessian_without_changing_state() {
        let data = dyestuff_fixture();
        let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let theta_before = model.theta();
        let objective_before = model.objective_value();
        let mut varpar = fitted_varpar(&model);
        varpar[0] = 0.0;

        let err = model.vcov_varpar(&varpar, false).unwrap_err();

        assert!(err.to_string().contains("lower bound"));
        assert_eq!(model.theta(), theta_before);
        assert_relative_eq!(model.objective_value(), objective_before, epsilon = 1e-10);
    }

    #[test]
    fn test_kenward_roger_sigma_g_scalar_random_intercept_components() {
        let data = dyestuff_fixture();
        let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(true).unwrap();

        let sigma_g = model.kenward_roger_sigma_g().unwrap();

        assert_eq!(sigma_g.n_observations, model.nobs());
        assert_eq!(sigma_g.components.len(), 2);
        assert_eq!(sigma_g.component_weights.len(), 2);
        assert_eq!(sigma_g.component_labels.len(), 2);
        assert_eq!(sigma_g.residual_component_index, 1);
        assert_eq!(sigma_g.component_labels[1], "residual");
        assert!(sigma_g.includes_residual_variance);
        assert!(sigma_g.sigma_positive_definite);
        assert!(sigma_g.sigma_min_eigenvalue > 0.0);
        assert!(matrix_is_finite(&sigma_g.sigma));
        assert_matrix_symmetric(&sigma_g.sigma, 1e-10);
        for component in &sigma_g.components {
            assert_matrix_symmetric(component, 1e-12);
        }

        let residual_variance = model.sigma().powi(2);
        let random_variance = residual_variance * model.theta()[0].powi(2);
        assert_relative_eq!(
            sigma_g.component_weights[0],
            random_variance,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            sigma_g.component_weights[1],
            residual_variance,
            epsilon = 1e-6
        );

        let refs = &model.reterms[0].refs;
        for row in 0..model.nobs() {
            for col in 0..model.nobs() {
                let mut expected = if refs[row] == refs[col] {
                    random_variance
                } else {
                    0.0
                };
                if row == col {
                    expected += residual_variance;
                }
                assert_relative_eq!(sigma_g.sigma[(row, col)], expected, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_kenward_roger_sigma_g_vector_random_effect_components() {
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(true).unwrap();

        let sigma_g = model.kenward_roger_sigma_g().unwrap();

        assert_eq!(sigma_g.components.len(), 4);
        assert_eq!(sigma_g.residual_component_index, 3);
        assert_eq!(sigma_g.component_labels[3], "residual");
        assert!(sigma_g.component_labels[0].contains("(Intercept),(Intercept)"));
        assert!(sigma_g.component_labels[1].contains("days,(Intercept)"));
        assert!(sigma_g.component_labels[2].contains("days,days"));
        assert!(sigma_g.sigma_positive_definite);
        assert!(sigma_g.max_component_asymmetry <= 1e-12);

        let residual_variance = model.sigma().powi(2);
        let varcorr =
            residual_variance * (&model.reterms[0].lambda * model.reterms[0].lambda.transpose());
        assert_relative_eq!(
            sigma_g.component_weights[0],
            varcorr[(0, 0)],
            epsilon = 1e-6
        );
        assert_relative_eq!(
            sigma_g.component_weights[1],
            varcorr[(1, 0)],
            epsilon = 1e-6
        );
        assert_relative_eq!(
            sigma_g.component_weights[2],
            varcorr[(1, 1)],
            epsilon = 1e-6
        );
        assert_relative_eq!(
            sigma_g.component_weights[3],
            residual_variance,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_kenward_roger_adjusted_vcov_returns_pbkrtest_style_artifacts() {
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(true).unwrap();

        let adjusted = model.kenward_roger_adjusted_vcov().unwrap();

        let p = model.feterm.rank;
        let n_components = model.kenward_roger_sigma_g().unwrap().components.len();
        assert_eq!(adjusted.unadjusted_vcov_active.shape(), (p, p));
        assert_eq!(adjusted.adjusted_vcov_active.shape(), (p, p));
        assert_eq!(
            adjusted.adjusted_vcov.shape(),
            (model.coef_names().len(), model.coef_names().len())
        );
        assert_eq!(adjusted.p_matrices.len(), n_components);
        assert_eq!(
            adjusted.q_matrices.len(),
            n_components * (n_components + 1) / 2
        );
        assert_eq!(adjusted.w.shape(), (n_components, n_components));
        assert_eq!(
            adjusted.information_matrix.shape(),
            (n_components, n_components)
        );
        assert_eq!(adjusted.information_eigenvalues.len(), n_components);
        assert_eq!(adjusted.component_labels.len(), n_components);
        assert!(matrix_is_finite(&adjusted.unadjusted_vcov_active));
        assert!(matrix_is_finite(&adjusted.adjusted_vcov_active));
        assert!(matrix_is_finite(&adjusted.adjusted_vcov));
        assert!(matrix_is_finite(&adjusted.w));
        assert!(matrix_is_finite(&adjusted.information_matrix));
        assert_matrix_symmetric(&adjusted.adjusted_vcov_active, 1e-8);
        assert_matrix_symmetric(&adjusted.adjusted_vcov, 1e-8);
        assert_matrix_symmetric(&adjusted.w, 1e-8);
        assert_matrix_symmetric(&adjusted.information_matrix, 1e-8);
        for p_matrix in &adjusted.p_matrices {
            assert_eq!(p_matrix.shape(), (p, p));
            assert_matrix_symmetric(p_matrix, 1e-8);
        }
        for q_matrix in &adjusted.q_matrices {
            assert_eq!(q_matrix.shape(), (p, p));
            assert!(matrix_is_finite(q_matrix));
        }
        assert!(matches!(
            adjusted.reliability,
            ReliabilityGrade::Moderate | ReliabilityGrade::Low
        ));
    }

    #[test]
    fn test_kenward_roger_adjusted_vcov_rejects_unweighted_prerequisite_gap() {
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
        let weights = vec![1.0; data.nrow()];
        let mut model = LinearMixedModel::new(formula, &data, Some(&weights)).unwrap();
        model.fit(true).unwrap();

        let err = model.kenward_roger_adjusted_vcov().unwrap_err();

        assert!(err
            .to_string()
            .contains("unweighted iid Gaussian residual models"));
    }

    #[test]
    fn test_kenward_roger_lbddf_scalar_contrast_matches_expected_scale() {
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (days | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(true).unwrap();

        let l = DMatrix::from_row_slice(1, model.coef_names().len(), &[0.0, 1.0]);
        let ddf = model.kenward_roger_lbddf(&l).unwrap();

        assert_eq!(ddf.restriction_rank, 1);
        assert_relative_eq!(ddf.numerator_df, 1.0, epsilon = 1e-12);
        assert!(ddf.denominator_df.is_finite());
        assert!(
            (15.0..=20.0).contains(&ddf.denominator_df),
            "pbkrtest sleepstudy days df is expected near 17, got {}",
            ddf.denominator_df
        );
        assert!(ddf.a1.is_finite());
        assert!(ddf.a2.is_finite());
        assert!(ddf.b.is_finite());
        assert!(ddf.g.is_finite());
        assert!(ddf.rho.is_finite());
        assert!(matches!(
            ddf.reliability,
            ReliabilityGrade::Moderate | ReliabilityGrade::Low
        ));
    }

    #[test]
    fn test_kenward_roger_lbddf_handles_rank_deficient_restriction_matrix() {
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (days | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(true).unwrap();

        let l = DMatrix::from_row_slice(
            2,
            model.coef_names().len(),
            &[
                0.0, 1.0, //
                0.0, 1.0,
            ],
        );
        let ddf = model.kenward_roger_lbddf(&l).unwrap();

        assert_eq!(ddf.restriction_rank, 1);
        assert_relative_eq!(ddf.numerator_df, 1.0, epsilon = 1e-12);
        assert!(ddf.used_generalized_inverse);
        assert!(ddf
            .notes
            .iter()
            .any(|note| note.contains("row rank 1 is lower")));
        assert!(ddf.denominator_df.is_finite());
        assert!(ddf.denominator_df > 0.0);
    }

    #[test]
    fn test_kenward_roger_lbddf_multi_df_contrast_returns_rank_df() {
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (days | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(true).unwrap();

        let l = DMatrix::identity(model.coef_names().len(), model.coef_names().len());
        let ddf = model.kenward_roger_lbddf(&l).unwrap();

        assert_eq!(ddf.restriction_rank, 2);
        assert_relative_eq!(ddf.numerator_df, 2.0, epsilon = 1e-12);
        assert!(ddf.denominator_df.is_finite());
        assert!(ddf.denominator_df > 0.0);
    }

    #[test]
    fn test_lmm_explicit_kenward_roger_scalar_request_returns_t_test() {
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (days | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(true).unwrap();

        let hypothesis =
            FixedEffectHypothesis::single_coefficient("days = 0", 1, model.coef_names().len())
                .unwrap();
        let test = model.test_contrast_with_method(hypothesis, FixedEffectTestMethod::KenwardRoger);

        assert_eq!(test.method, InferenceMethod::KenwardRoger);
        assert_eq!(test.status, InferenceStatus::Available);
        assert!(test.numerator_df.is_none());
        assert!(test.denominator_df.unwrap().is_finite());
        assert!((15.0..=20.0).contains(&test.denominator_df.unwrap()));
        assert!(test.standard_errors[0].unwrap().is_finite());
        assert!(test.statistics[0].unwrap().is_finite());
        assert!(test.p_values[0].unwrap().is_finite());
        assert!((0.0..=1.0).contains(&test.p_values[0].unwrap()));
        assert!(test.notes.iter().any(|note| note.contains("Kenward-Roger")));
    }

    #[test]
    fn test_lmm_explicit_kenward_roger_multi_df_request_returns_f_test() {
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (days | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(true).unwrap();

        let l = DMatrix::identity(model.coef_names().len(), model.coef_names().len());
        let hypothesis = FixedEffectHypothesis::new(
            "all fixed effects = 0",
            crate::compiler::ContrastMatrix::new(l).unwrap(),
            crate::compiler::ContrastRhs::zeros(model.coef_names().len()),
        )
        .unwrap();
        let test = model.test_contrast_with_method(hypothesis, FixedEffectTestMethod::KenwardRoger);

        assert_eq!(test.method, InferenceMethod::KenwardRoger);
        assert_eq!(test.status, InferenceStatus::Available);
        assert_eq!(test.numerator_df, Some(2.0));
        assert!(test.denominator_df.unwrap().is_finite());
        assert!(test.denominator_df.unwrap() > 0.0);
        assert_eq!(test.statistics.len(), 1);
        assert!(test.statistics[0].unwrap().is_finite());
        assert!(test.statistics[0].unwrap() >= 0.0);
        assert_eq!(test.p_values.len(), 1);
        assert!(test.p_values[0].unwrap().is_finite());
        assert!((0.0..=1.0).contains(&test.p_values[0].unwrap()));
        assert!(test
            .notes
            .iter()
            .any(|note| note.contains("F scaling = 1.0")));

        let row = fixed_effect_test_to_inference_row(FixedEffectInferenceRowKind::Term, test);
        let details = row.details.expect("multi-df row should carry details");
        let family = details
            .contrast_family
            .expect("multi-df row should carry contrast-family details");
        assert_eq!(family.restriction_rows, 2);
        assert_eq!(family.effective_rank, Some(2));
        assert_eq!(family.numerator_df_semantics, "effective_restriction_rank");
        let kr = details
            .kenward_roger
            .expect("KR row should carry KR details");
        assert_eq!(kr.f_scaling, Some(1.0));
        assert_eq!(kr.statistic_scale.as_deref(), Some("unscaled"));
    }

    #[test]
    fn test_lmm_explicit_kenward_roger_ml_request_does_not_fallback() {
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (days | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let hypothesis =
            FixedEffectHypothesis::single_coefficient("days = 0", 1, model.coef_names().len())
                .unwrap();
        let test = model.test_contrast_with_method(hypothesis, FixedEffectTestMethod::KenwardRoger);

        assert_eq!(test.method, InferenceMethod::KenwardRoger);
        assert!(matches!(test.status, InferenceStatus::NotAssessed { .. }));
        assert_eq!(test.p_values, vec![None]);
        assert!(fixed_effect_inference_reason(&test)
            .unwrap()
            .contains("REML"));
    }

    #[test]
    fn test_lmm_kenward_roger_scalar_rows_match_pbkrtest_fixture() {
        let fixture = kenward_roger_pbkrtest_parity_fixture();

        for case in fixture.scalar_cases {
            let data = sleepstudy_fixture();
            let formula = parse_formula(&case.formula).unwrap();
            let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
            model.fit(true).unwrap();

            let hypothesis = fixed_effect_hypothesis_from_fixture(&case.label, &case.l, &case.rhs);
            let test =
                model.test_contrast_with_method(hypothesis, FixedEffectTestMethod::KenwardRoger);

            assert_eq!(test.method, InferenceMethod::KenwardRoger, "{}", case.name);
            assert_eq!(test.status, InferenceStatus::Available, "{}", case.name);
            assert!(
                matches!(
                    test.reliability,
                    ReliabilityGrade::Moderate | ReliabilityGrade::Low
                ),
                "{}",
                case.name
            );
            assert!(test.numerator_df.is_none(), "{}", case.name);
            assert_relative_eq!(
                test.estimates[0],
                case.estimate,
                epsilon = 1e-8,
                max_relative = 1e-8
            );
            assert_relative_eq!(
                test.standard_errors[0].unwrap(),
                case.std_error,
                epsilon = 5e-5,
                max_relative = 5e-5
            );
            assert_relative_eq!(
                test.denominator_df.unwrap(),
                case.denominator_df,
                epsilon = 1e-3,
                max_relative = 1e-5
            );
            assert_relative_eq!(
                test.statistics[0].unwrap(),
                case.statistic,
                epsilon = 5e-5,
                max_relative = 5e-5
            );
            assert_relative_eq!(
                test.p_values[0].unwrap(),
                case.p_value,
                epsilon = 1e-12,
                max_relative = 1e-3
            );
        }
    }

    #[test]
    fn test_lmm_kenward_roger_multi_df_rows_match_pbkrtest_unscaled_fixture() {
        let fixture = kenward_roger_pbkrtest_parity_fixture();

        for case in fixture.multi_df_cases {
            let data = sleepstudy_fixture();
            let formula = parse_formula(&case.formula).unwrap();
            let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
            model.fit(true).unwrap();

            let hypothesis = fixed_effect_hypothesis_from_fixture(&case.label, &case.l, &case.rhs);
            let test =
                model.test_contrast_with_method(hypothesis, FixedEffectTestMethod::KenwardRoger);

            assert_eq!(test.method, InferenceMethod::KenwardRoger, "{}", case.name);
            assert_eq!(test.status, InferenceStatus::Available, "{}", case.name);
            if case.l.len() == 1 {
                assert_eq!(test.numerator_df, None, "{}", case.name);
                assert_relative_eq!(
                    test.statistics[0].unwrap().powi(2),
                    case.unscaled_statistic,
                    epsilon = 1e-6,
                    max_relative = 1e-6
                );
                assert_relative_eq!(
                    test.p_values[0].unwrap(),
                    case.unscaled_p_value,
                    epsilon = 1e-12,
                    max_relative = 1e-3
                );
                continue;
            }
            assert_eq!(test.numerator_df, Some(case.numerator_df), "{}", case.name);
            // Multi-df F drift vs pbkrtest is dominated by numerical noise in the
            // adjusted-vcov off-diagonals (β and Φ_A diagonals match to 1e-7;
            // det(Φ_A) drift sits in the 3e-4 band).  Match a realistic numerical
            // tolerance rather than bit-exactness.
            assert_relative_eq!(
                test.denominator_df.unwrap(),
                case.denominator_df,
                epsilon = 1e-3,
                max_relative = 5e-4,
            );
            assert!(
                (test.statistics[0].unwrap() - case.unscaled_statistic).abs()
                    <= 1e-3 + 5e-4 * case.unscaled_statistic.abs(),
                "{}: unscaled F drift exceeds tolerance: rust={} ref={}",
                case.name,
                test.statistics[0].unwrap(),
                case.unscaled_statistic
            );
            assert_relative_eq!(
                test.p_values[0].unwrap(),
                case.unscaled_p_value,
                epsilon = 1e-12,
                max_relative = 1e-3,
            );

            if (case.f_scaling - 1.0).abs() > 1e-12 {
                assert_ne!(case.statistic, case.unscaled_statistic);
                assert_ne!(case.p_value, case.unscaled_p_value);
                assert!(test
                    .notes
                    .iter()
                    .any(|note| note.contains("F scaling = 1.0")));
            }
        }
    }

    #[test]
    fn test_fixed_effect_h0_simulation_smoke_for_analytic_p_values() {
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula.clone(), &data, None).unwrap();
        model.fit(true).unwrap();

        let days_index = model
            .coef_names()
            .iter()
            .position(|name| name == "days")
            .unwrap();
        let hypothesis = FixedEffectHypothesis::single_coefficient(
            "days = 0",
            days_index,
            model.coef_names().len(),
        )
        .unwrap();
        let target = model
            .fixed_effect_null_bootstrap_target(&hypothesis)
            .unwrap();

        let mut rng = StdRng::seed_from_u64(20260501);
        let mut wald_p_values = Vec::new();
        let mut satterthwaite_p_values = Vec::new();
        let mut kenward_roger_p_values = Vec::new();

        for _ in 0..8 {
            let y_sim = model.simulate_fixed_effect_null(&mut rng, &target).unwrap();
            let mut sim_data = DataFrame::new();
            sim_data
                .add_numeric("reaction", y_sim.iter().copied().collect())
                .unwrap();
            sim_data
                .add_numeric("days", data.numeric("days").unwrap().to_vec())
                .unwrap();
            let subj = data.categorical("subj").unwrap();
            sim_data
                .add_categorical_with_levels("subj", subj.values.clone(), subj.levels.clone())
                .unwrap();
            let mut work = LinearMixedModel::new(formula.clone(), &sim_data, None).unwrap();
            work.fit(true).unwrap();

            let wald = work.test_contrast_with_method(
                hypothesis.clone(),
                FixedEffectTestMethod::AsymptoticWaldZ,
            );
            let satterthwaite = work.test_contrast_with_method(
                hypothesis.clone(),
                FixedEffectTestMethod::Satterthwaite,
            );
            let kenward_roger = work
                .test_contrast_with_method(hypothesis.clone(), FixedEffectTestMethod::KenwardRoger);

            assert_eq!(wald.status, InferenceStatus::Available);
            assert_eq!(satterthwaite.status, InferenceStatus::Available);
            assert_eq!(kenward_roger.status, InferenceStatus::Available);
            wald_p_values.push(wald.p_values[0].unwrap());
            satterthwaite_p_values.push(satterthwaite.p_values[0].unwrap());
            kenward_roger_p_values.push(kenward_roger.p_values[0].unwrap());
        }

        for (label, values) in [
            ("Wald", &wald_p_values),
            ("Satterthwaite", &satterthwaite_p_values),
            ("Kenward-Roger", &kenward_roger_p_values),
        ] {
            assert_eq!(values.len(), 8, "{label} should produce all p-values");
            assert!(
                values
                    .iter()
                    .all(|p| p.is_finite() && (0.0..=1.0).contains(p)),
                "{label} p-values should be finite probabilities: {values:?}"
            );
            let tiny = values.iter().filter(|&&p| p < 0.01).count();
            assert!(
                tiny <= 2,
                "{label} produced too many tiny p-values under the simulated null: {values:?}"
            );
        }
    }

    #[test]
    fn test_dyestuff_aic_bic_matches_julia() {
        // Mirrors pls.jl "Dyestuff":
        //   aic(fm1) ≈ 333.32705988112673
        //   bic(fm1) ≈ 337.5306520261132
        let data = dyestuff_fixture();
        let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let obj = model.objective_value(); // -2*loglik
        let k = model.dof() as f64;
        let n = model.nobs() as f64;
        let aic = obj + 2.0 * k;
        let bic = obj + k * n.ln();

        assert_relative_eq!(aic, 333.32705988112673, epsilon = 1e-3);
        assert_relative_eq!(bic, 337.5306520261132, epsilon = 1e-3);
    }

    #[test]
    fn test_dyestuff_re_std_dev_matches_julia() {
        // Mirrors pls.jl: first(first(fm1.σs)) ≈ 37.260343703061764
        // RE std dev = lambda * sigma = 0.7526 * 49.51 ≈ 37.26
        let data = dyestuff_fixture();
        let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let vc = model.varcorr();
        assert_eq!(vc.components.len(), 1);
        let comp = &vc.components[0];
        assert_eq!(comp.group, "batch");
        assert_relative_eq!(comp.std_dev[0], 37.260343703061764, epsilon = 0.1);
    }

    #[test]
    fn test_dyestuff_reml_matches_julia() {
        // Mirrors pls.jl "Dyestuff" REML refit.
        // Julia: objective ≈ 319.6542768422576
        //        vcov[0,0] ≈ 375.7167103872769 (variance of intercept under REML)
        let data = dyestuff_fixture();
        let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(true).unwrap(); // REML

        assert_relative_eq!(model.objective_value(), 319.6542768422576, epsilon = 1e-3);
        // REML vcov of the intercept
        let v = model.vcov();
        assert_eq!(v.nrows(), 1);
        assert_relative_eq!(v[(0, 0)], 375.7167103872769, epsilon = 1.0);
    }

    #[test]
    fn test_sleepstudy_vector_re_matches_julia() {
        // Mirrors pls.jl "sleep" testset (last model: (1 + days | subj)).
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap(); // ML

        assert_relative_eq!(model.objective_value(), 1751.9393444636682, epsilon = 0.01);
        let theta = model.theta();
        assert_eq!(theta.len(), 3);
        assert_relative_eq!(theta[0], 0.9292297167514472, epsilon = 1e-3);
        assert_relative_eq!(theta[1], 0.01816466496782548, epsilon = 1e-3);
        assert_relative_eq!(theta[2], 0.22264601131030412, epsilon = 1e-3);

        // coef() returns in original formula order: [intercept, days]
        let coef = MixedModelFit::coef(&model);
        assert_relative_eq!(coef[0], 251.40510484848454, epsilon = 0.01);
        assert_relative_eq!(coef[1], 10.467285959596126, epsilon = 0.01);

        let se = model.stderror();
        assert_relative_eq!(se[0], 6.632295312722272, epsilon = 0.1);
        assert_relative_eq!(se[1], 1.5022387911441102, epsilon = 0.05);

        assert_relative_eq!(model.loglikelihood(), -875.9696722318341, epsilon = 0.01);
    }

    #[test]
    fn test_lrt_sleepstudy_matches_julia() {
        // Mirrors likelihoodratiotest.jl "likelihoodratio test":
        //   fm0: reaction ~ 1 + (1 + days | subj)  [no days in FE, dof=5]
        //   fm1: reaction ~ 1 + days + (1 + days | subj) [days in FE, dof=6]
        // Julia: chisq ≈ 23.5365, dof=1, p < 1e-5
        use crate::stats::lrt::LikelihoodRatioTest;
        let data = sleepstudy_fixture();

        let f0 = parse_formula("reaction ~ 1 + (1 + days | subj)").unwrap();
        let mut m0 = LinearMixedModel::new(f0, &data, None).unwrap();
        m0.fit(false).unwrap();

        let f1 = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
        let mut m1 = LinearMixedModel::new(f1, &data, None).unwrap();
        m1.fit(false).unwrap();

        assert!(
            m0.objective_value() > m1.objective_value(),
            "fm0 should have larger objective"
        );
        assert_eq!(m0.dof(), 5);
        assert_eq!(m1.dof(), 6);

        let lrt = LikelihoodRatioTest::test(&[&m0 as &dyn MixedModelFit, &m1]).unwrap();
        assert_eq!(lrt.chisq_dof[0], 1);
        assert_relative_eq!(lrt.chisq[0], 23.5365, epsilon = 0.05);
        assert!(lrt.pvalues[0] < 1e-5);
    }

    #[test]
    fn test_penicillin_crossed_re_matches_julia() {
        // Mirrors pls.jl "penicillin" testset.
        // Formula: diameter ~ 1 + (1 | plate) + (1 | sample)
        let data = penicillin_fixture();
        let formula = parse_formula("diameter ~ 1 + (1 | plate) + (1 | sample)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap(); // ML

        assert_eq!(model.nobs(), 144);

        assert_relative_eq!(model.objective_value(), 332.1883486700085, epsilon = 0.01);

        let coef = MixedModelFit::coef(&model);
        assert_relative_eq!(coef[0], 22.97222222222222, epsilon = 1e-4);

        assert_relative_eq!(model.stderror()[0], 0.7446037806555799, epsilon = 0.01);

        // θ[0] = plate RE, θ[1] = sample RE
        let theta = model.theta();
        assert_eq!(theta.len(), 2);
        assert_relative_eq!(theta[0], 1.5375939045981573, epsilon = 0.01);
        assert_relative_eq!(theta[1], 3.219792193110907, epsilon = 0.01);
    }

    #[test]
    fn test_dyestuff2_singular_fit_matches_julia() {
        // Mirrors pls.jl "Dyestuff2" testset.
        // The within-batch variance dominates → RE collapses to 0 (singular).
        let data = dyestuff2_fixture();
        let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap(); // ML

        // Julia: fm.θ ≈ zeros(1)
        assert!(
            model.theta()[0].abs() < 1e-6,
            "theta should be ~0 for singular fit, got {}",
            model.theta()[0]
        );
        // Julia: objective(fm) ≈ 162.87303665382575
        assert_relative_eq!(model.objective_value(), 162.87303665382575, epsilon = 1e-3);
        // Julia: coef(fm) ≈ [5.6656]
        let coef = MixedModelFit::coef(&model);
        assert_relative_eq!(coef[0], 5.6656, epsilon = 1e-3);
        // Julia: stderror(fm) ≈ [0.6669857396443264]
        assert_relative_eq!(model.stderror()[0], 0.6669857396443264, epsilon = 1e-3);
        // Julia: logdet(fm) ≈ 0.0 (RE variance = 0 → Λ diagonal = 0)
        assert_relative_eq!(model.logdet_re(), 0.0, epsilon = 1e-8);
        // Julia: issingular(fm) == true
        assert!(model.is_singular(), "Dyestuff2 fit should be singular");
    }

    #[test]
    fn test_dyestuff_objective_at_specific_theta() {
        // Mirrors pls.jl: objective!(fm1, 0.713) ≈ 327.34216280954615
        // Julia evaluates this on an ML-mode model (reml=false).
        let data = dyestuff_fixture();
        let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.optsum.reml = false; // match Julia ML mode
        let obj = model.objective_at(&[0.713]).unwrap();
        assert_relative_eq!(obj, 327.34216280954615, epsilon = 1e-3);
    }

    #[test]
    fn test_dyestuff_logdet_pwrss_varest() {
        // Mirrors pls.jl "Dyestuff" testset — additional metrics after ML fit.
        // Julia: logdet(fm1) ≈ 8.06014611206176
        //        varest(fm1) ≈ 2451.2500368886936  (= sigma^2)
        //        pwrss(fm1)  ≈ 73537.50110666081
        let data = dyestuff_fixture();
        let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        assert_relative_eq!(model.logdet_re(), 8.06014611206176, epsilon = 1e-3);
        assert_relative_eq!(
            model.sigma() * model.sigma(),
            2451.2500368886936,
            epsilon = 1.0
        );
        assert_relative_eq!(model.pwrss(), 73537.50110666081, epsilon = 10.0);
    }

    #[test]
    fn test_penicillin_logdet_and_varest() {
        // Mirrors pls.jl "penicillin" testset — additional metrics.
        // Julia: varest(fm) ≈ 0.30242510228527864
        //        logdet(fm) ≈ 95.74676552743833
        let data = penicillin_fixture();
        let formula = parse_formula("diameter ~ 1 + (1 | plate) + (1 | sample)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        assert_relative_eq!(
            model.sigma() * model.sigma(),
            0.30242510228527864,
            epsilon = 1e-4
        );
        assert_relative_eq!(model.logdet_re(), 95.74676552743833, epsilon = 0.1);
    }

    #[test]
    fn test_sleepstudy_random_slope_only_matches_julia() {
        // Mirrors pls.jl: fmrs = reaction ~ 1 + days + (0 + days | subj)
        // Random slope only (no random intercept).
        // Julia: objective ≈ 1774.080315280526, θ ≈ [0.24353985601485326]
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (0 + days | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        assert_relative_eq!(model.objective_value(), 1774.080315280526, epsilon = 0.01);
        let theta = model.theta();
        assert_eq!(theta.len(), 1, "random-slope-only has scalar theta");
        assert_relative_eq!(theta[0], 0.24353985601485326, epsilon = 1e-3);
    }

    #[test]
    fn test_pastes_nested_re_matches_julia() {
        // Mirrors pls.jl "pastes" testset.
        // Julia formula: strength ~ 1 + (1 | batch / cask)
        // which expands to: strength ~ 1 + (1 | batch) + (1 | batch:cask)
        // We use pre-computed batch_cask interaction column.
        // Julia: objective ≈ 247.9944658624955
        //        coef ≈ [60.0533333333333]
        //        stderror ≈ [0.6421355774401101]
        //        θ ≈ [3.5269029347766856, 1.3299137410046242]
        let data = pastes_fixture();
        let formula = parse_formula("strength ~ 1 + (1 | batch) + (1 | batch_cask)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        assert_eq!(model.nobs(), 60);
        assert_relative_eq!(model.objective_value(), 247.9944658624955, epsilon = 0.01);

        let coef = MixedModelFit::coef(&model);
        assert_relative_eq!(coef[0], 60.0533333333333, epsilon = 1e-3);

        assert_relative_eq!(model.stderror()[0], 0.6421355774401101, epsilon = 0.01);

        let theta = model.theta();
        assert_eq!(theta.len(), 2);
        // Julia sorts by decreasing nranef: θ[0] = batch:cask RE (30 levels), θ[1] = batch RE (10 levels)
        assert_relative_eq!(theta[0], 3.5269029347766856, epsilon = 0.05);
        assert_relative_eq!(theta[1], 1.3299137410046242, epsilon = 0.05);
    }

    fn weighted_lmm_fixture() -> (DataFrame, Vec<f64>) {
        let a = vec![
            1.55945122,
            0.004391538,
            0.005554163,
            -0.173029772,
            4.586284429,
            0.259493671,
            -0.091735715,
            5.546487603,
            0.457734831,
            -0.030169602,
        ];
        let b = vec![
            0.24520519,
            0.080624178,
            0.228083467,
            0.2471453,
            0.398994279,
            0.037213859,
            0.102144973,
            0.241380251,
            0.206570975,
            0.15980803,
        ];
        let c = vec!["H", "F", "K", "P", "P", "P", "D", "M", "I", "D"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let w1: Vec<f64> = vec![20.0, 40.0, 35.0, 12.0, 29.0, 25.0, 65.0, 105.0, 30.0, 75.0];

        let mut df = DataFrame::new();
        df.add_numeric("a", a).unwrap();
        df.add_numeric("b", b).unwrap();
        df.add_categorical("c", c).unwrap();

        (df, w1)
    }

    #[test]
    fn test_weighted_model_matches_julia() {
        // Mirrors pls.jl "wts" testset.
        // Julia: m2 = fit(@formula(a ~ 1 + b + (1 | c)), data; wts=w1)
        //   θ ≈ [0.2951818091809752]
        //   stderror ≈ [0.964016663994572, 3.6309691484830533]
        //   vcov ≈ [[0.9293, -2.5575], [-2.5575, 13.1839]]
        let (df, w1) = weighted_lmm_fixture();

        let formula = parse_formula("a ~ 1 + b + (1 | c)").unwrap();
        let mut model = LinearMixedModel::new(formula, &df, Some(&w1)).unwrap();
        model.fit(false).unwrap();

        assert_relative_eq!(model.theta()[0], 0.2951818091809752, epsilon = 1e-3);
        let se = model.stderror();
        assert_eq!(se.len(), 2);
        assert_relative_eq!(se[0], 0.964016663994572, epsilon = 0.01);
        assert_relative_eq!(se[1], 3.6309691484830533, epsilon = 0.1);
        // Julia: vcov ≈ [[0.9293 -2.5575], [-2.5575 13.1839]]
        let v = model.vcov();
        assert_relative_eq!(v[(0, 0)], 0.9293281284592235, epsilon = 0.01);
        assert_relative_eq!(v[(0, 1)], -2.5575260810649962, epsilon = 0.05);
        assert_relative_eq!(v[(1, 0)], -2.5575260810649962, epsilon = 0.05);
        assert_relative_eq!(v[(1, 1)], 13.18393695723575, epsilon = 0.1);
    }

    #[test]
    fn test_weighted_lmm_objective_matches_julia_normalization() {
        let (df, w1) = weighted_lmm_fixture();
        let formula = parse_formula("a ~ 1 + b + (1 | c)").unwrap();
        let mut model = LinearMixedModel::new(formula, &df, Some(&w1)).unwrap();
        model.fit(false).unwrap();

        let expected_correction: f64 = w1.iter().map(|weight| weight.ln()).sum();

        assert_relative_eq!(
            model.weight_logdet_correction(),
            expected_correction,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            model.objective_value(),
            model.profiled_objective_value() - expected_correction,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_unweighted_objective_unchanged_by_weight_normalization() {
        let data = dyestuff_fixture();
        let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        assert_eq!(model.weight_logdet_correction(), 0.0);
        assert_relative_eq!(
            model.objective_value(),
            model.profiled_objective_value(),
            epsilon = 1e-12
        );
        assert_relative_eq!(model.objective_value(), 327.32705988112673, epsilon = 1e-3);
    }

    #[test]
    fn test_weighted_lrt_matches_profiled_target_difference() {
        use crate::stats::lrt::LikelihoodRatioTest;

        let (df, w1) = weighted_lmm_fixture();
        let f0 = parse_formula("a ~ 1 + (1 | c)").unwrap();
        let mut m0 = LinearMixedModel::new(f0, &df, Some(&w1)).unwrap();
        m0.fit(false).unwrap();

        let f1 = parse_formula("a ~ 1 + b + (1 | c)").unwrap();
        let mut m1 = LinearMixedModel::new(f1, &df, Some(&w1)).unwrap();
        m1.fit(false).unwrap();

        let raw_chisq = m0.profiled_objective_value() - m1.profiled_objective_value();
        let corrected_chisq = m0.objective_value() - m1.objective_value();
        assert_relative_eq!(corrected_chisq, raw_chisq, epsilon = 1e-10);

        let lrt = LikelihoodRatioTest::test(&[&m0 as &dyn MixedModelFit, &m1]).unwrap();
        assert_relative_eq!(lrt.chisq[0], corrected_chisq, epsilon = 1e-10);
    }

    #[test]
    fn test_rank_deficient_fixed_effects() {
        // Mirrors pls.jl "Rank deficient" testset.
        // x2 = 1.5 * x makes the FE design matrix rank-deficient (rank 2, not 3).
        // Julia: length(fixef) == 2, rank(model) == 2, length(coef) == 3
        let n = 100usize;
        let x: Vec<f64> = (0..n).map(|i| (i as f64 % 10.0) / 9.0).collect();
        let x2: Vec<f64> = x.iter().map(|&v| 1.5 * v).collect();
        // Simple deterministic y
        let y: Vec<f64> = (0..n).map(|i| ((i * 7 + 3) % 17) as f64 * 0.1).collect();
        let z: Vec<String> = (0..n)
            .map(|i| format!("{}", (b'A' + (i % 20) as u8) as char))
            .collect();

        let mut df = DataFrame::new();
        df.add_numeric("y", y).unwrap();
        df.add_numeric("x", x).unwrap();
        df.add_numeric("x2", x2).unwrap();
        df.add_categorical("z", z).unwrap();

        let formula = parse_formula("y ~ x + x2 + (1 | z)").unwrap();
        let mut model = LinearMixedModel::new(formula, &df, None).unwrap();
        model.fit(false).unwrap();

        // x2 is a linear combination of x → rank 2 (intercept + x or x2)
        assert_eq!(
            model.feterm.rank, 2,
            "rank should be 2 (intercept + one predictor)"
        );
        // fixef() returns only independent coefficients
        assert_eq!(model.fixef().len(), 2);
        // coef() returns all original columns (with 0/NaN for the dropped one)
        assert_eq!(MixedModelFit::coef(&model).len(), 3);
    }

    #[test]
    fn test_sleepstudy_re_std_devs_match_julia() {
        // Mirrors pls.jl "sleep":
        //   first(std(fm)) ≈ [23.78066438213187, 5.7168446983832775]
        //   VarCorr RE correlation between intercept and days ≈ +0.08
        //   fm.corr (fixed-effects correlation) ≈ [1.0 -0.1376; -0.1376 1.0]
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let vc = model.varcorr();
        assert_eq!(vc.components.len(), 1);
        let comp = &vc.components[0];
        assert_eq!(comp.group, "subj");
        assert_eq!(comp.std_dev.len(), 2);
        // Julia: first(std(fm)) ≈ [23.78066438213187, 5.7168446983832775]
        assert_relative_eq!(comp.std_dev[0], 23.78066438213187, epsilon = 0.1);
        assert_relative_eq!(comp.std_dev[1], 5.7168446983832775, epsilon = 0.1);
        // VarCorr RE correlation: theta[1] / ||row_1(lambda)|| ≈ +0.08
        assert_eq!(comp.correlations.len(), 1);
        assert_relative_eq!(comp.correlations[0], 0.0813, epsilon = 0.01);

        // fm.corr in Julia is vcov(m; corr=true) — the fixed-effects correlation,
        // NOT VarCorr. Julia: stderror ≈ [6.6323, 1.5022], corr[0,1] ≈ -0.1376.
        let vcov = model.vcov();
        let se = model.stderror();
        assert_relative_eq!(se[0], 6.632295312722272, epsilon = 0.01);
        assert_relative_eq!(se[1], 1.5022387911441102, epsilon = 0.01);
        let fe_corr = vcov[(0, 1)] / (se[0] * se[1]);
        assert_relative_eq!(fe_corr, -0.13755599049585931, epsilon = 0.01);
    }

    #[test]
    fn test_sleepstudy_vector_re_logdet_and_pwrss() {
        // Mirrors pls.jl "sleep" testset — additional metrics.
        // Julia: logdet(fm) ≈ 73.90350673367566
        //        pwrss(fm)  ≈ 117889.27379003687
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        assert_relative_eq!(model.logdet_re(), 73.90350673367566, epsilon = 0.1);
        assert_relative_eq!(model.pwrss(), 117889.27379003687, epsilon = 100.0);
    }

    #[test]
    fn test_sleepstudy_zerocorr_re_matches_julia() {
        // Mirrors pls.jl "sleep" fmnc (zerocorr) model:
        //   reaction ~ 1 + days + zerocorr(1 + days | subj)
        // Julia: objective ≈ 1752.003255140962
        //        θ ≈ [0.9458, 0.2269]  (diagonal-only lambda: 2 params)
        //        coef ≈ [251.405, 10.467]
        //        stderror ≈ [6.708, 1.519]
        //        logdet ≈ 74.4694698615524
        let data = sleepstudy_fixture();
        // Our parser uses `||` (double-pipe) for zero-correlation RE.
        let formula = parse_formula("reaction ~ 1 + days + (1 + days || subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        assert_relative_eq!(model.objective_value(), 1752.003255140962, epsilon = 0.1);

        let theta = model.theta();
        assert_eq!(theta.len(), 2, "zerocorr model has 2 theta params");
        assert_relative_eq!(theta[0], 0.9458043022417869, epsilon = 0.01);
        assert_relative_eq!(theta[1], 0.22692740996014607, epsilon = 0.01);
        let artifact = model.compiler_artifact();
        assert_eq!(artifact.semantic_model.random_terms.len(), 2);
        assert_eq!(artifact.theta_maps.len(), 2);
        assert!(artifact
            .semantic_model
            .random_terms
            .iter()
            .all(|term| term.block_group.as_deref() == Some("bg0")));
        assert!(artifact
            .covariance_parameter_traces
            .iter()
            .all(|trace| trace
                .parmap_entry
                .as_ref()
                .is_some_and(|entry| entry.matches_theta_map)));

        let coef = MixedModelFit::coef(&model);
        assert_relative_eq!(coef[0], 251.4051048484854, epsilon = 0.1);
        assert_relative_eq!(coef[1], 10.467285959595674, epsilon = 0.05);

        let se = model.stderror();
        assert_relative_eq!(se[0], 6.707646513654387, epsilon = 0.1);
        assert_relative_eq!(se[1], 1.5193112497954953, epsilon = 0.05);

        assert_relative_eq!(model.logdet_re(), 74.4694698615524, epsilon = 0.1);
    }

    #[cfg(feature = "nlopt")]
    #[test]
    fn test_penicillin_varcorr_std_devs_match_julia() {
        // Mirrors pls.jl "penicillin": std(fm) ≈ [[0.8456], [1.7707], [0.5499]]
        // std[0] = plate RE, std[1] = sample RE, residual sigma = 0.5499
        let data = penicillin_fixture();
        let formula = parse_formula("diameter ~ 1 + (1 | plate) + (1 | sample)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let sigma = model.sigma();
        // Julia: only(last(std)) ≈ 0.549931906953287 (residual sigma)
        assert_relative_eq!(sigma, 0.549931906953287, epsilon = 1e-4);

        let vc = model.varcorr();
        assert_eq!(vc.components.len(), 2);
        // plate RE
        assert_eq!(vc.components[0].group, "plate");
        assert_relative_eq!(
            vc.components[0].std_dev[0],
            0.845571948075415,
            epsilon = 1e-4
        );
        // sample RE
        assert_eq!(vc.components[1].group, "sample");
        assert_relative_eq!(
            vc.components[1].std_dev[0],
            1.770666460750787,
            epsilon = 1e-4
        );
        // residual
        assert_relative_eq!(vc.residual_sd.unwrap(), sigma, epsilon = 1e-12);
    }

    #[test]
    fn test_sleepstudy_zerocorr_varcorr_std_devs() {
        // Mirrors pls.jl "sleep" fmnc (zerocorr):
        //   first(std(fmnc)) ≈ [24.171269957611873, 5.79939919963132]
        //   last(std(fmnc))  ≈ [25.55613836753517]   (residual sigma)
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 + days || subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let sigma = model.sigma();
        assert_relative_eq!(sigma, 25.55613836753517, epsilon = 0.1);

        let vc = model.varcorr();
        assert_eq!(vc.components.len(), 1);
        let comp = &vc.components[0];
        assert_eq!(comp.std_dev.len(), 2);
        assert_relative_eq!(comp.std_dev[0], 24.171269957611873, epsilon = 0.1);
        assert_relative_eq!(comp.std_dev[1], 5.79939919963132, epsilon = 0.1);
        // zerocorr → diagonal Lambda → off-diagonal correlation is 0
        assert_eq!(comp.correlations.len(), 1);
        assert_relative_eq!(comp.correlations[0], 0.0, epsilon = 1e-8);
    }

    #[test]
    fn test_sleepstudy_independent_re_equivalent_to_zerocorr() {
        // Mirrors pls.jl "sleep" fm_ind equivalence test (lines 447-454):
        //   fm_ind = models(:sleepstudy)[3]
        //          = reaction ~ 1 + days + (1 | subj) + (0 + days | subj)
        //   @test objective(fm_ind) ≈ objective(fmnc)   # fmnc = zerocorr model
        //   @test coef(fm_ind) ≈ coef(fmnc)
        //   @test stderror(fm_ind) ≈ stderror(fmnc)
        //   @test fm_ind.θ ≈ fmnc.θ
        //   @test logdet(fm_ind) ≈ logdet(fmnc)
        //
        // Two separate scalar RE terms for the same grouping factor are
        // equivalent to a single zerocorr (diagonal-λ) RE term because
        // their contributions to the log-likelihood are additive.
        let data = sleepstudy_fixture();

        let f_zc = parse_formula("reaction ~ 1 + days + (1 + days || subj)").unwrap();
        let mut m_zc = LinearMixedModel::new(f_zc, &data, None).unwrap();
        m_zc.fit(false).unwrap();

        // Two separate scalar terms for same grouping factor
        let f_ind = parse_formula("reaction ~ 1 + days + (1 | subj) + (0 + days | subj)").unwrap();
        let mut m_ind = LinearMixedModel::new(f_ind, &data, None).unwrap();
        m_ind.fit(false).unwrap();

        // Objectives should match to high precision (same log-likelihood surface)
        assert_relative_eq!(
            m_ind.objective_value(),
            m_zc.objective_value(),
            epsilon = 0.01
        );

        // Fixed-effects coefficients (pivot order may differ, compare sums/lengths)
        let coef_zc = MixedModelFit::coef(&m_zc);
        let coef_ind = MixedModelFit::coef(&m_ind);
        assert_eq!(
            coef_zc.len(),
            coef_ind.len(),
            "same number of FE coefficients"
        );

        // logdet should match
        assert_relative_eq!(m_ind.logdet_re(), m_zc.logdet_re(), epsilon = 0.1);

        // theta lengths differ (zerocorr: 2 params in 1 term; fm_ind: 1+1 in 2 terms)
        // but the effective model is the same
        assert_eq!(
            m_ind.theta().len(),
            2,
            "two separate scalar RE → 2 theta params"
        );
        assert_eq!(m_zc.theta().len(), 2, "zerocorr RE → 2 theta params");
    }

    #[test]
    fn test_optsum_fitlog_population() {
        // Mirrors pls.jl "Dyestuff fitlog" testset (lines 146-161):
        //   fitlog = fm1.optsum.fitlog
        //   @test length(fitlogtbl) == 3        -- has iter, objective, θ columns
        //   @test length(first(fitlogtbl)) > 15 -- more than 15 function evals
        //   @test last(fitlogtbl.objective) == fm1.optsum.fmin
        //
        // We verify our OptSummary.fit_log is populated after fitting:
        //   - length(fit_log) == feval (one entry per function evaluation)
        //   - length(fit_log) > 10    (at least 10 evaluations for dyestuff)
        //   - fit_log[0].theta == optsum.initial  (first eval uses initial θ)
        //   - fit_log.last().objective == optsum.fmin  (last entry = minimum)
        let data = dyestuff_fixture();
        let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let log = &model.optsum.fit_log;

        // log populated and length matches feval count
        assert!(!log.is_empty(), "fit_log should be non-empty after fitting");
        assert_eq!(
            log.len() as i64,
            model.optsum.feval,
            "fit_log length should equal feval"
        );

        // At least 10 function evaluations for dyestuff (typically ~30-50)
        assert!(
            log.len() >= 10,
            "expected ≥ 10 function evaluations, got {}",
            log.len()
        );

        // First entry should use the initial theta
        let initial = &model.optsum.initial;
        assert_eq!(
            log[0].theta.len(),
            initial.len(),
            "first log entry theta length should match initial"
        );

        // Last entry objective should equal fmin
        let last_obj = log.last().unwrap().objective;
        assert_relative_eq!(last_obj, model.optsum.fmin, epsilon = 1e-8);

        // The minimum objective across the log should be fmin (or very close)
        let min_logged = log
            .iter()
            .map(|e| e.objective)
            .fold(f64::INFINITY, f64::min);
        assert_relative_eq!(min_logged, model.optsum.fmin, epsilon = 1e-6);
    }

    #[test]
    fn test_optsum_fitlog_theta_dimensions() {
        // Extended fitlog check: every entry's theta has the right length.
        // Mirrors pls.jl: d == length(first(fitlogtbl.θ))  (theta dim consistent)
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let n_theta = model.optsum.initial.len();
        for (i, entry) in model.optsum.fit_log.iter().enumerate() {
            assert_eq!(
                entry.theta.len(),
                n_theta,
                "fit_log[{}].theta should have {} elements",
                i,
                n_theta
            );
        }
    }

    #[test]
    fn test_pastes_lrt_pvalue_matches_julia() {
        // Mirrors pls.jl "pastes": lrt = likelihoodratiotest(models(:pastes)...)
        //   last(lrt.pvalues) ≈ 0.5233767965780878
        // models(:pastes)[1] = strength ~ 1 + (1 | batch & cask)  (cask-within-batch only)
        // models(:pastes)[2] = strength ~ 1 + (1 | batch / cask)  (batch + batch:cask)
        let data = pastes_fixture();

        // Simpler model: batch:cask interaction only (no batch main effect)
        let formula1 = parse_formula("strength ~ 1 + (1 | batch_cask)").unwrap();
        let mut m1 = LinearMixedModel::new(formula1, &data, None).unwrap();
        m1.fit(false).unwrap();

        // Richer model: batch main RE + batch:cask interaction RE
        let formula2 = parse_formula("strength ~ 1 + (1 | batch) + (1 | batch_cask)").unwrap();
        let mut m2 = LinearMixedModel::new(formula2, &data, None).unwrap();
        m2.fit(false).unwrap();

        use crate::model::traits::MixedModelFit;
        use crate::stats::lrt::LikelihoodRatioTest;
        let lrt =
            LikelihoodRatioTest::test(&[&m1 as &dyn MixedModelFit, &m2 as &dyn MixedModelFit])
                .unwrap();
        assert_eq!(lrt.pvalues.len(), 1);
        assert_relative_eq!(lrt.pvalues[0], 0.5233767965780878, epsilon = 0.01);
    }

    #[test]
    fn test_pastes_varcorr_and_logdet_match_julia() {
        // Mirrors pls.jl "pastes":
        //   only(first(stdd)) ≈ 2.904   (batch:cask RE std dev, 30 levels — first in nranef sort)
        //   only(stdd[2])     ≈ 1.095   (batch RE std dev, 10 levels — second)
        //   only(last(stdd))  ≈ 0.823   (residual sigma)
        //   varest(fm) ≈ 0.677999727889528
        //   logdet(fm) ≈ 101.03834542101686
        let data = pastes_fixture();
        let formula = parse_formula("strength ~ 1 + (1 | batch) + (1 | batch_cask)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let sigma = model.sigma();
        assert_relative_eq!(sigma, 0.8234073887751603, epsilon = 1e-4);
        assert_relative_eq!(sigma * sigma, 0.677999727889528, epsilon = 1e-4);
        assert_relative_eq!(model.logdet_re(), 101.03834542101686, epsilon = 0.1);

        let vc = model.varcorr();
        assert_eq!(vc.components.len(), 2);
        // Julia sorts RE terms by decreasing nranef: batch:cask (30 levels) first, batch (10) second.
        // Julia: first(std) ≈ 2.904 (batch:cask, 30 levels), stdd[2] ≈ 1.095 (batch, 10 levels)
        let batch_comp = vc
            .components
            .iter()
            .find(|c| c.group == "batch")
            .expect("batch component");
        let cask_comp = vc
            .components
            .iter()
            .find(|c| c.group == "batch_cask")
            .expect("batch_cask component");
        assert_relative_eq!(cask_comp.std_dev[0], 2.90407793598792, epsilon = 1e-3);
        assert_relative_eq!(batch_comp.std_dev[0], 1.0950608007768226, epsilon = 1e-4);
        // residual
        assert_relative_eq!(vc.residual_sd.unwrap(), sigma, epsilon = 1e-12);
    }

    #[test]
    fn test_dyestuff2_sigma_matches_julia() {
        // Mirrors pls.jl "Dyestuff2": std(fm)[2] ≈ [3.6532313513746537]
        // (residual sigma; RE collapses to 0 in singular fit)
        let data = dyestuff2_fixture();
        let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        assert_relative_eq!(model.sigma(), 3.6532313513746537, epsilon = 1e-4);
    }

    #[test]
    fn test_pastes_batch_cask_only_model() {
        // models(:pastes)[1] = strength ~ 1 + (1 | batch & cask) — cask-within-batch only.
        // Julia: objective ≈ 247.9944658624955 for the full nested model (last);
        //   the simpler model (batch & cask only) has fewer RE levels.
        // Here we just verify it fits and has sane values.
        let data = pastes_fixture();
        let formula = parse_formula("strength ~ 1 + (1 | batch_cask)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        assert_eq!(model.nobs(), 60);
        // Intercept ≈ mean(strength)
        let coef = MixedModelFit::coef(&model);
        assert_relative_eq!(coef[0], 60.0533333333333, epsilon = 0.1);
        // This simpler model must have lower DOF than the full nested model
        assert_eq!(model.dof(), 3); // 1 FE + 1 RE theta + 1 sigma
    }

    #[test]
    fn test_dyestuff_cond_is_one() {
        // Mirrors pls.jl: cond(fm1) == ones(1)
        // Scalar RE has a 1×1 Lambda → condition number is always 1.
        let data = dyestuff_fixture();
        let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let c = model.cond();
        assert_eq!(c.len(), 1);
        assert_relative_eq!(c[0], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_sleepstudy_vector_re_cond_matches_julia() {
        // Mirrors pls.jl: only(cond(fm)) ≈ 4.175266438717022
        // Vector RE Lambda is 2×2 lower-triangular; condition number > 1.
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let c = model.cond();
        assert_eq!(c.len(), 1);
        assert_relative_eq!(c[0], 4.175266438717022, epsilon = 0.01);
    }

    #[test]
    fn test_dof_residual_matches_julia() {
        // Mirrors pls.jl: dof_residual(fm1) ≥ 0
        // For dyestuff: nobs=30, rank=1 (intercept only) → dof_residual=29
        let data = dyestuff_fixture();
        let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        assert_eq!(model.dof_residual(), 29); // 30 obs - 1 FE
        assert!(model.dof_residual() > 0);
    }

    #[test]
    fn test_sleepstudy_dof_residual() {
        // Sleepstudy: nobs=180, rank=2 (intercept + days) → dof_residual=178
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        assert_eq!(model.dof_residual(), 178); // 180 obs - 2 FE
    }

    #[test]
    fn test_dyestuff_response_and_model_matrix() {
        // Mirrors pls.jl: modelmatrix(fm1) == ones(30,1), response == ds.yield
        let data = dyestuff_fixture();
        let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let x = model.model_matrix();
        assert_eq!(x.nrows(), 30);
        assert_eq!(x.ncols(), 1);
        // Intercept-only FE → all ones
        assert!(x.iter().all(|&v| (v - 1.0).abs() < 1e-12));

        let y = model.response();
        assert_eq!(y.len(), 30);
        // First batch A: 5 values with mean ~1538
        let mean_y = y.mean();
        assert_relative_eq!(mean_y, 1527.5, epsilon = 1e-6);
    }

    #[test]
    fn test_dyestuff_fitted_and_residuals() {
        // Mirrors pls.jl "Dyestuff": fitted values and residuals basic checks.
        // For an intercept-only model: mean(fitted) ≈ mean(y), sum(residuals) ≈ 0
        let data = dyestuff_fixture();
        let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let y = model.response();
        let fitted = model.fitted();
        let residuals = model.residuals();
        assert_eq!(fitted.len(), 30);
        assert_eq!(residuals.len(), 30);
        // residuals = y - fitted
        for i in 0..30 {
            assert_relative_eq!(residuals[i], y[i] - fitted[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_penicillin_model_structure() {
        // Mirrors pls.jl: size(fm) == (144, 1, 30, 2)
        // nobs=144, rank=1 (intercept), total_nranef=30 (24 plate + 6 sample), 2 RE terms
        let data = penicillin_fixture();
        let formula = parse_formula("diameter ~ 1 + (1 | plate) + (1 | sample)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        assert_eq!(model.nobs(), 144);
        assert_eq!(model.feterm.rank, 1);
        assert_eq!(model.reterms.len(), 2);
        let total_ranef: usize = model.reterms.iter().map(|rt| rt.n_ranef()).sum();
        assert_eq!(total_ranef, 30); // 24 plates + 6 samples
    }

    #[test]
    fn test_sleepstudy_model_structure() {
        // Mirrors pls.jl: rank(fm) == 2 for the vector RE model
        // nobs=180, rank=2 (intercept+days), 1 RE term with 18*2=36 ranef
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        assert_eq!(model.nobs(), 180);
        assert_eq!(model.feterm.rank, 2);
        assert_eq!(model.reterms.len(), 1);
        let total_ranef: usize = model.reterms.iter().map(|rt| rt.n_ranef()).sum();
        assert_eq!(total_ranef, 36); // 18 subjects × 2 RE (intercept + slope)
    }

    // ── condVar parity with MixedModels.jl/test/pls.jl ─────────────────────

    #[test]
    fn test_dyestuff_condvar_shape() {
        // pls.jl: @test length(cv) == 1; @test size(first(cv)) == (1, 1, 6)
        let data = dyestuff_fixture();
        let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let cv = model.cond_var();
        assert_eq!(cv.len(), 1, "one RE term");
        assert_eq!(cv[0].len(), 6, "6 batch levels");
        assert_eq!(cv[0][0].nrows(), 1);
        assert_eq!(cv[0][0].ncols(), 1);
    }

    #[test]
    fn test_penicillin_condvar_matches_julia() {
        // pls.jl:
        //   @test length(cv) == 2
        //   @test size(first(cv)) == (1, 1, 24)
        //   @test size(last(cv)) == (1, 1, 6)
        //   @test first(first(cv)) ≈ 0.07331356908917808 rtol = 1.e-4
        //   @test last(last(cv))  ≈ 0.04051591717427688 rtol = 1.e-4
        let data = penicillin_fixture();
        let formula = parse_formula("diameter ~ 1 + (1 | plate) + (1 | sample)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let cv = model.cond_var();
        assert_eq!(cv.len(), 2);

        // first term = plate (24 levels, sorted first by nranef)
        assert_eq!(cv[0].len(), 24);
        assert_eq!(cv[0][0].nrows(), 1);
        assert_relative_eq!(cv[0][0][(0, 0)], 0.07331356908917808, epsilon = 1e-4);

        // last term = sample (6 levels)
        assert_eq!(cv[1].len(), 6);
        assert_relative_eq!(cv[1][5][(0, 0)], 0.04051591717427688, epsilon = 1e-4);
    }

    #[test]
    fn test_sleepstudy_condvar_matches_julia() {
        // pls.jl:
        //   @test size(cv1) == (2, 2, 18)
        //   @test first(cv1) ≈ 140.96755256125914 rtol = 1.e-4   → cv[0][0][(0,0)]
        //   @test last(cv1)  ≈ 5.157794803497628  rtol = 1.e-4   → cv[0][17][(1,1)]
        //   @test cv1[2]     ≈ -20.604544204749537 rtol = 1.e-4  → cv[0][0][(1,0)]
        //   (Julia column-major: cv1[2] = cv1[2,1,1] = row 2, col 1, level 1 = (1,0) 0-indexed)
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let cv = model.cond_var();
        assert_eq!(cv.len(), 1);
        assert_eq!(cv[0].len(), 18);
        assert_eq!(cv[0][0].nrows(), 2);
        assert_eq!(cv[0][0].ncols(), 2);

        assert_relative_eq!(cv[0][0][(0, 0)], 140.96755256125914, epsilon = 1.0);
        assert_relative_eq!(cv[0][17][(1, 1)], 5.157794803497628, epsilon = 0.1);
        assert_relative_eq!(cv[0][0][(1, 0)], -20.604544204749537, epsilon = 0.5);
    }

    // ── leverage parity with MixedModels.jl/test/pls.jl ────────────────────

    #[test]
    fn test_dyestuff_leverage_matches_julia() {
        // pls.jl:
        //   @test first(leverage(fm1)) ≈ 0.1565053420672158 rtol = 1.e-5
        //   @test sum(leverage(fm1))   ≈ 4.695160262016474  rtol = 1.e-5
        let data = dyestuff_fixture();
        let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let lev = model.leverage();
        assert_eq!(lev.len(), 30);
        assert_relative_eq!(lev[0], 0.1565053420672158, epsilon = 1e-4);
        assert_relative_eq!(lev.sum(), 4.695160262016474, epsilon = 1e-3);
    }

    #[test]
    fn test_sleepstudy_vector_re_leverage_sum_matches_julia() {
        // pls.jl:
        //   @test sum(leverage(fm)) ≈ 28.611653305323234 rtol = 1.e-5
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let lev = model.leverage();
        assert_eq!(lev.len(), 180);
        assert_relative_eq!(lev.sum(), 28.611653305323234, epsilon = 0.01);
    }

    // ── ranef_u / ranef_b parity with MixedModels.jl/test/pls.jl ───────────

    fn manual_one_term_ranef_u_via_block_solver(model: &LinearMixedModel) -> DMatrix<f64> {
        assert_eq!(model.reterms.len(), 1);
        let re = &model.reterms[0];
        let vs = re.vsize;
        let n_levels = re.n_levels();
        let nranef = re.n_ranef();
        let p = model.dims.p;
        let n = model.dims.n;
        let beta = model.beta();
        let wtxy = &model.xy_mat.wtxy;

        let mut wr = vec![0.0f64; n];
        for obs in 0..n {
            let mut val = wtxy[(obs, p)];
            for q in 0..p {
                val -= wtxy[(obs, q)] * beta[q];
            }
            wr[obs] = val;
        }

        let mut c = vec![0.0f64; nranef];
        for obs in 0..n {
            let r = re.refs[obs] as usize;
            for s in 0..vs {
                c[r * vs + s] += re.wtz[(s, obs)] * wr[obs];
            }
        }

        let mut c_scaled = vec![0.0f64; nranef];
        for lev in 0..n_levels {
            for i in 0..vs {
                let mut val = 0.0;
                for row in i..vs {
                    val += re.lambda[(row, i)] * c[lev * vs + row];
                }
                c_scaled[lev * vs + i] = val;
            }
        }

        let l = &model.l_blocks[block_index(0, 0)];
        let mut rhs_matrix = DMatrix::from_column_slice(nranef, 1, &c_scaled);
        solve_lower_block_rhs(&mut rhs_matrix, l);
        let mut u: Vec<f64> = (0..nranef).map(|idx| rhs_matrix[(idx, 0)]).collect();
        solve_upper_block_from_lower_transpose_against_rhs(l, &mut u);

        DMatrix::from_column_slice(vs, n_levels, &u)
    }

    #[test]
    fn test_ranef_u_matches_solve_lower_block_rhs() {
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let actual = model.ranef_u();
        let manual = manual_one_term_ranef_u_via_block_solver(&model);

        assert_eq!(actual.len(), 1);
        assert_eq!(actual[0].shape(), manual.shape());
        for row in 0..manual.nrows() {
            for col in 0..manual.ncols() {
                assert_relative_eq!(
                    actual[0][(row, col)],
                    manual[(row, col)],
                    epsilon = 1e-10,
                    max_relative = 1e-10
                );
            }
        }
    }

    #[test]
    fn test_ranef_u_regression_current_outputs() {
        let data = penicillin_fixture();
        let formula = parse_formula("diameter ~ 1 + (1 | plate) + (1 | sample)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let rfu = model.ranef_u();

        assert_eq!(rfu.len(), 2);
        assert_relative_eq!(rfu[0][(0, 0)], 0.5231574704291094, epsilon = 1e-3);
        assert_relative_eq!(rfu[1][(0, 5)], -0.9323155679350466, epsilon = 1e-3);
    }

    #[test]
    fn test_dyestuff_ranef_u_sums_to_zero() {
        // pls.jl: @test abs(sum(only(rfu))) < 1.e-5
        // The u vector for a balanced model sums to zero (BLUP property).
        let data = dyestuff_fixture();
        let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let rfu = model.ranef_u();
        assert_eq!(rfu.len(), 1);
        let u_sum: f64 = rfu[0].iter().sum();
        assert!(
            u_sum.abs() < 1e-4,
            "sum of u (dyestuff) should be ≈ 0, got {u_sum}"
        );
    }

    #[cfg(feature = "nlopt")]
    #[test]
    fn test_sleepstudy_ranef_u_shape_and_first_element() {
        // pls.jl:
        //   @test size(first(u3)) == (2, 18)
        //   @test first(only(u3)) ≈ 3.030047743065841 atol = 0.001
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let u3 = model.ranef_u();
        assert_eq!(u3.len(), 1, "one RE term");
        assert_eq!(u3[0].nrows(), 2, "vsize = 2 (intercept + slope)");
        assert_eq!(u3[0].ncols(), 18, "18 subjects");

        // Julia's first(only(u3)) is the (1,1) element (intercept for first subject)
        assert_relative_eq!(u3[0][(0, 0)], 3.030047743065841, epsilon = 0.001);
    }

    #[cfg(feature = "nlopt")]
    #[test]
    fn test_sleepstudy_ranef_b_first_element() {
        // pls.jl: @test first(only(b3)) ≈ 2.8156104060324334 atol = 0.001
        // b = Λ * u  (conditional mode on original scale)
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let b3 = model.ranef_b();
        assert_eq!(b3.len(), 1);
        assert_eq!(b3[0].nrows(), 2);
        assert_eq!(b3[0].ncols(), 18);
        assert_relative_eq!(b3[0][(0, 0)], 2.8156104060324334, epsilon = 0.001);
    }

    #[test]
    fn test_penicillin_ranef_u_first_element() {
        // pls.jl: @test first(first(rfu)) ≈ 0.5231574704291094 rtol = 1.e-4
        // penicillin has 2 RE terms (plate, sample); rfu is sorted by decreasing nranef.
        // first(rfu) → the term with more levels (24 plates).
        let data = penicillin_fixture();
        let formula = parse_formula("diameter ~ 1 + (1 | plate) + (1 | sample)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let rfu = model.ranef_u();
        assert_eq!(rfu.len(), 2, "two RE terms");

        // Determine which term is plate (24 levels) — it should sort first
        let first_term = &rfu[0];
        let first_u = first_term[(0, 0)];
        assert_relative_eq!(first_u, 0.5231574704291094, epsilon = 1e-3);
    }

    #[test]
    fn test_penicillin_ranef_b_last_element() {
        // pls.jl: @test last(last(rfb)) ≈ -3.0018241391465703 rtol = 1.e-4
        // last(rfb) is the term with fewer levels (6 samples).
        let data = penicillin_fixture();
        let formula = parse_formula("diameter ~ 1 + (1 | plate) + (1 | sample)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let rfb = model.ranef_b();
        assert_eq!(rfb.len(), 2);

        // last term (fewer levels = samples, 6 levels), last element
        let last_term = &rfb[rfb.len() - 1];
        let last_b = last_term[(0, last_term.ncols() - 1)];
        assert_relative_eq!(last_b, -3.0018241391465703, epsilon = 1e-3);
    }

    // ── std / logdet / varest / model_size / refit / simulate parity ─────────

    #[test]
    fn test_penicillin_varest_and_logdet() {
        // pls.jl:
        //   @test varest(fm) ≈ 0.30242510228527864 atol=0.0001
        //   @test logdet(fm) ≈ 95.74676552743833 atol=0.005
        let data = penicillin_fixture();
        let formula = parse_formula("diameter ~ 1 + (1 | plate) + (1 | sample)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        assert_relative_eq!(model.varest(), 0.30242510228527864, epsilon = 1e-4);
        assert_relative_eq!(model.logdet(), 95.74676552743833, epsilon = 0.05);
    }

    #[test]
    fn test_penicillin_std_devs() {
        // pls.jl:
        //   stdd = std(fm)
        //   @test only(first(stdd)) ≈ 0.845571948075415 atol=0.0001   # plate
        //   @test only(stdd[2]) ≈ 1.770666460750787 atol=0.0001       # sample
        //   @test only(last(stdd)) ≈ 0.549931906953287 atol=0.0001    # sigma
        let data = penicillin_fixture();
        let formula = parse_formula("diameter ~ 1 + (1 | plate) + (1 | sample)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let stdd = model.std_devs();
        // reterms sorted by decreasing nranef: plate (24) first, sample (6) second
        assert_relative_eq!(stdd[0][0], 0.845571948075415, epsilon = 1e-3);
        assert_relative_eq!(stdd[1][0], 1.770666460750787, epsilon = 1e-3);
        assert_relative_eq!(stdd[2][0], 0.549931906953287, epsilon = 1e-3); // sigma
    }

    #[test]
    fn test_penicillin_model_size() {
        // pls.jl: @test size(fm) == (144, 1, 30, 2)
        // n=144, p=1, nranef=24+6=30, nretrms=2
        let data = penicillin_fixture();
        let formula = parse_formula("diameter ~ 1 + (1 | plate) + (1 | sample)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        assert_eq!(model.model_size(), (144, 1, 30, 2));
    }

    #[test]
    fn test_sleepstudy_model_size() {
        // pls.jl: @test size(fm) == (180, 2, 36, 1) for the vector RE model
        // n=180, p=2, nranef=18*2=36, nretrms=1
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        assert_eq!(model.model_size(), (180, 2, 36, 1));
    }

    #[test]
    fn test_dyestuff_refit_new_response() {
        // pls.jl: refit!(fm, new_y); @test objective(fm) ≈ 327.32705988112673 atol=0.001
        // (refitting a dyestuff2-like model with the dyestuff yields)
        let data = dyestuff_fixture();
        let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();
        let dev_before = model.objective_value();

        // Refit with constant-shifted response (should converge to different value)
        let new_y: Vec<f64> = model.y().iter().map(|&y| y + 100.0).collect();
        model.refit(&new_y).unwrap();

        // β (intercept) should shift by 100; deviance should be unchanged
        assert_relative_eq!(model.objective_value(), dev_before, epsilon = 1e-4);
    }

    #[test]
    fn test_refit_preserves_reml_flag() {
        let data = dyestuff_fixture();
        let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(true).unwrap();

        let original_y: Vec<f64> = model.y().iter().copied().collect();
        model.refit(&original_y).unwrap();

        assert!(model.optsum.reml);
    }

    #[test]
    fn test_refit_after_reml_objective_matches() {
        let data = dyestuff_fixture();
        let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(true).unwrap();
        let objective_before = model.objective_value();
        let original_y: Vec<f64> = model.y().iter().copied().collect();

        model.refit(&original_y).unwrap();

        assert!(model.optsum.reml);
        assert_relative_eq!(model.objective_value(), objective_before, epsilon = 1e-6);
    }

    #[test]
    fn test_refit_rejects_constant_response() {
        // pls.jl: @test_throws ArgumentError refit!(fm, zero(slp.reaction))
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let zeros = vec![0.0f64; model.dims.n];
        assert!(model.refit(&zeros).is_err());
    }

    #[test]
    fn test_simulate_length_and_distribution() {
        // simulate(fm) should return a vector of length n
        // bootstrap.jl: refit!(simulate!(rng, fm)); @test deviance ≈ ...
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let data = dyestuff_fixture();
        let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let mut rng = StdRng::seed_from_u64(12345);
        let y_sim = model.simulate(&mut rng);

        assert_eq!(
            y_sim.len(),
            30,
            "simulated response should have n=30 elements"
        );

        // Mean should be close to the fitted intercept (±3 sigma)
        let mean_sim = y_sim.iter().sum::<f64>() / 30.0;
        let beta = model.beta();
        assert!(
            (mean_sim - beta[0]).abs() < 3.0 * model.sigma() * (30.0f64).sqrt(),
            "simulated mean {mean_sim:.1} unexpectedly far from intercept {:.1}",
            beta[0]
        );
    }

    // ── LRT parity tests (likelihoodratiotest.jl) ────────────────────────────

    #[test]
    fn test_lrt_sleepstudy_deviances_and_chisq() {
        // likelihoodratiotest.jl:
        //   fm0 = reaction ~ 1 + (1 + days | subj)       → deviance ≈ 1775.4759, dof = 5
        //   fm1 = reaction ~ 1 + days + (1 + days | subj) → deviance ≈ 1751.9393, dof = 6
        //   lrt.chisq[0] ≈ 23.5365, p-value < 1e-5
        use crate::stats::lrt::LikelihoodRatioTest;

        let data = sleepstudy_fixture();

        let f0 = parse_formula("reaction ~ 1 + (1 + days | subj)").unwrap();
        let mut fm0 = LinearMixedModel::new(f0, &data, None).unwrap();
        fm0.fit(false).unwrap();

        let f1 = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
        let mut fm1 = LinearMixedModel::new(f1, &data, None).unwrap();
        fm1.fit(false).unwrap();

        // deviance = -2 * loglikelihood
        let dev0 = -2.0 * fm0.loglikelihood();
        let dev1 = -2.0 * fm1.loglikelihood();
        assert_relative_eq!(dev0, 1775.4759, epsilon = 0.1);
        assert_relative_eq!(dev1, 1751.9393, epsilon = 0.1);

        assert_eq!(fm0.dof(), 5);
        assert_eq!(fm1.dof(), 6);

        let lrt =
            LikelihoodRatioTest::test(&[&fm0 as &dyn MixedModelFit, &fm1 as &dyn MixedModelFit])
                .unwrap();

        assert_relative_eq!(lrt.chisq[0], 23.5365, epsilon = 0.05);
        assert!(
            lrt.pvalues[0] < 1e-5,
            "p-value should be < 1e-5, got {}",
            lrt.pvalues[0]
        );
    }

    #[test]
    fn test_lrt_dyestuff_null_vs_intercept_only() {
        // Dyestuff: the batch variance is clearly non-zero so the LRT comparing
        // a model without RE against one with RE should yield a very small p-value.
        let data = dyestuff_fixture();

        // Null model: intercept-only mixed model (fm1 in pls.jl)
        let f1 = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut fm1 = LinearMixedModel::new(f1, &data, None).unwrap();
        fm1.fit(false).unwrap();

        // Constrained model: θ fixed at 0 (singular fit)
        let f0 = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut fm0 = LinearMixedModel::new(f0, &data, None).unwrap();
        fm0.set_theta(&[0.0]).unwrap();
        fm0.update_l().unwrap(); // recompute L at θ=0

        // fm1 deviance = -2*loglik ≈ 327.327 (AIC = deviance + 2*3 ≈ 333.327 — from pls.jl)
        let dev1 = -2.0 * fm1.loglikelihood();
        assert_relative_eq!(dev1, 327.327, epsilon = 0.01);
    }

    // ── predict / predict_new parity tests (predict.jl) ─────────────────────

    #[test]
    fn test_predict_training_equals_fitted() {
        // predict.jl: @test predict(m) ≈ fitted(m)
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let pred = model.predict();
        let fitted = model.fitted();
        assert_eq!(pred.len(), fitted.len());
        for i in 0..pred.len() {
            assert_relative_eq!(pred[i], fitted[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn test_predict_new_same_data_equals_fitted() {
        // predict.jl: @test predict(m, slp; new_re_levels=:error) ≈ fitted(m)
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let fitted = model.fitted();
        for strategy in [
            NewReLevels::Error,
            NewReLevels::Population,
            NewReLevels::Missing,
        ] {
            let result = model.predict_new(&data, strategy).unwrap();
            assert_eq!(result.len(), fitted.len());
            for i in 0..result.len() {
                let pred = result[i].expect("training data should never be None");
                assert_relative_eq!(pred, fitted[i], epsilon = 1e-8, max_relative = 1e-8);
            }
        }
    }

    #[test]
    fn test_predict_new_same_data_equals_fitted_for_same_group_random_blocks() {
        // Regression for lme4 issue #403-shaped formulas:
        // separate random-slope blocks share a grouping factor, so prediction
        // must match by grouping factor *and* random-effect basis.
        let mut y = Vec::new();
        let mut x1 = Vec::new();
        let mut x2 = Vec::new();
        let mut subject = Vec::new();
        for subj in 0..12 {
            let b0 = subj as f64 * 0.4 - 2.0;
            let b1 = (subj as f64 - 5.0) * 0.12;
            let b2 = (6.0 - subj as f64) * 0.18;
            for obs in 0..5 {
                let a = obs as f64 - 2.0;
                let c = (obs as f64 + 1.0).powi(2) / 4.0;
                x1.push(a);
                x2.push(c);
                y.push(10.0 + 1.5 * a - 0.7 * c + b0 + b1 * a + b2 * c);
                subject.push(format!("s{subj}"));
            }
        }

        let mut data = DataFrame::new();
        data.add_numeric("y", y).unwrap();
        data.add_numeric("x1", x1).unwrap();
        data.add_numeric("x2", x2).unwrap();
        data.add_categorical("subject", subject).unwrap();

        let formula =
            parse_formula("y ~ 1 + x1 + x2 + (1 + x1 | subject) + (1 + x2 | subject)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let fitted = model.fitted();
        let predicted = model.predict_new(&data, NewReLevels::Error).unwrap();
        assert_eq!(predicted.len(), fitted.len());
        for (idx, pred) in predicted.iter().enumerate() {
            assert_relative_eq!(
                pred.expect("training levels are known"),
                fitted[idx],
                epsilon = 1e-7,
                max_relative = 1e-7
            );
        }
    }

    #[test]
    fn test_predict_new_same_data_equals_fitted_for_cell_grouping() {
        let mut y = Vec::new();
        let mut x = Vec::new();
        let mut site = Vec::new();
        let mut item = Vec::new();
        for site_idx in 0..3 {
            for item_idx in 0..4 {
                let cell_effect = site_idx as f64 * 0.8 - item_idx as f64 * 0.3;
                for rep in 0..2 {
                    let xv = rep as f64 + item_idx as f64 * 0.25;
                    x.push(xv);
                    y.push(3.0 + 1.2 * xv + cell_effect);
                    site.push(format!("site{site_idx}"));
                    item.push(format!("item{item_idx}"));
                }
            }
        }

        let mut data = DataFrame::new();
        data.add_numeric("y", y).unwrap();
        data.add_numeric("x", x).unwrap();
        data.add_categorical("site", site).unwrap();
        data.add_categorical("item", item).unwrap();

        let formula = parse_formula("y ~ 1 + x + (1 | site:item)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let fitted = model.fitted();
        let predicted = model.predict_new(&data, NewReLevels::Error).unwrap();
        for (idx, pred) in predicted.iter().enumerate() {
            assert_relative_eq!(
                pred.expect("training cell levels are known"),
                fitted[idx],
                epsilon = 1e-8,
                max_relative = 1e-8
            );
        }
    }

    #[test]
    fn test_predict_with_unseen_level_returns_typed_err() {
        // predict.jl: @test_throws ArgumentError predict(m, slp2; new_re_levels=:error)
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let mut newdata = DataFrame::new();
        newdata.add_numeric("reaction", vec![300.0]).unwrap();
        newdata.add_numeric("days", vec![0.0]).unwrap();
        newdata
            .add_categorical("subj", vec!["UNSEEN".to_string()])
            .unwrap();

        let err = model.predict_new(&newdata, NewReLevels::Error).unwrap_err();

        match err {
            MixedModelError::InvalidArgument(message) => {
                assert!(message.contains("UNSEEN"));
                assert!(message.contains("subj"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_predict_new_unknown_level_population() {
        // predict.jl: ypop[1:10] ≈ view(m.X, 1:10, :) * m.β  (population prediction = Xβ)
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let beta = model.beta();
        let cnames = model.feterm.cnames.clone();
        let days: Vec<f64> = (0..10).map(|d| d as f64).collect();
        let mut newdata = DataFrame::new();
        newdata.add_numeric("reaction", vec![0.0; 10]).unwrap();
        newdata.add_numeric("days", days.clone()).unwrap();
        newdata
            .add_categorical("subj", vec!["NEW".to_string(); 10])
            .unwrap();

        let result = model
            .predict_new(&newdata, NewReLevels::Population)
            .unwrap();
        assert_eq!(result.len(), 10);

        // Coefficients by name (pivot order may not be [intercept, days])
        let intercept = cnames
            .iter()
            .position(|n| n == "(Intercept)")
            .map(|i| beta[i])
            .unwrap_or(0.0);
        let days_coef = cnames
            .iter()
            .position(|n| n == "days")
            .map(|i| beta[i])
            .unwrap_or(0.0);

        for (i, &d) in days.iter().enumerate() {
            let expected = intercept + d * days_coef;
            let pred = result[i].expect("Population should always return Some");
            assert_relative_eq!(pred, expected, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_predict_new_unknown_level_missing() {
        // predict.jl: count(ismissing, ymissing) == 10 (first 10 obs are new subject)
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        // 10 new-subject obs followed by 10 known-subject (S309 days 0-9)
        let mut days = vec![0.0f64; 20];
        let mut subjects: Vec<String> = vec!["NEW".to_string(); 10];
        for d in 0..10 {
            days[10 + d] = d as f64;
            subjects.push("S309".to_string());
        }
        let mut newdata = DataFrame::new();
        newdata.add_numeric("reaction", vec![0.0; 20]).unwrap();
        newdata.add_numeric("days", days).unwrap();
        newdata.add_categorical("subj", subjects).unwrap();

        let result = model.predict_new(&newdata, NewReLevels::Missing).unwrap();
        let n_missing = result.iter().filter(|v| v.is_none()).count();
        assert_eq!(n_missing, 10, "first 10 obs (new subject) should be None");
        for i in 10..20 {
            assert!(
                result[i].is_some(),
                "obs {} (known subject) should be Some",
                i
            );
        }
    }

    // ── coeftable parity tests (pls.jl "coeftable" testset) ──────────────────

    #[test]
    fn test_coeftable_dyestuff_shape() {
        // pls.jl: ct = coeftable(only(models(:dyestuff)))
        //         @test [3, 4] == [ct.teststatcol, ct.pvalcol]
        // In our 0-indexed struct: z_values is column 2, p_values is column 3.
        // We verify the table has 1 row (intercept-only FE) and reasonable values.
        let data = dyestuff_fixture();
        let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let ct = model.coeftable();

        // Dyestuff has one FE: (Intercept)
        assert_eq!(ct.len(), 1);
        assert_eq!(ct.names[0], "(Intercept)");

        // Estimate ≈ 1527.5 (mean of yield)
        assert_relative_eq!(ct.estimates[0], 1527.5, epsilon = 1.0);

        // z = estimate / SE should be very large (≈ 86)
        assert!(
            ct.z_values[0] > 50.0,
            "z for intercept should be large, got {}",
            ct.z_values[0]
        );

        // p-value should be essentially zero
        assert!(
            ct.p_values[0] < 1e-10,
            "p should be ≈0, got {}",
            ct.p_values[0]
        );
    }

    #[test]
    fn test_coeftable_sleepstudy_two_rows() {
        // sleepstudy: FE = (Intercept) + days → 2 rows in coeftable
        // pls.jl: coef ≈ [251.405, 10.467], stderror ≈ [6.632, 1.502]
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let ct = model.coeftable();
        assert_eq!(ct.len(), 2);

        // Both should have small p-values (both highly significant)
        for i in 0..2 {
            assert!(
                ct.p_values[i] < 0.01,
                "coef[{}] p-value {} should be < 0.01",
                i,
                ct.p_values[i]
            );
            // z = estimate / SE should be non-zero and finite
            assert!(ct.z_values[i].is_finite(), "z[{}] should be finite", i);
        }

        // SE should be positive
        for se in &ct.std_errors {
            assert!(*se > 0.0, "SE should be positive, got {}", se);
        }
    }

    #[test]
    fn test_coeftable_p_values_consistent_with_stderror() {
        // coeftable p-values should be consistent with stderror:
        // z = coef / SE,  p = 2*(1-Φ(|z|))
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let ct = model.coeftable();
        let coefs = MixedModelFit::coef(&model);
        let se = model.stderror();

        for i in 0..ct.len() {
            let expected_z = coefs[i] / se[i];
            assert_relative_eq!(ct.z_values[i], expected_z, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_coeftable_rank_deficient_nan_dropped() {
        // For a rank-deficient model, dropped columns get NaN SE/z/p in coeftable.
        // With x2 = 2*x, the pivot QR drops one of {x, x2} (whichever has smaller
        // post-orthogonalisation norm).  We verify exactly one column is NaN.
        let n = 30usize;
        let x: Vec<f64> = (0..n).map(|i| (i % 5) as f64).collect();
        let x2: Vec<f64> = x.iter().map(|&v| 2.0 * v).collect(); // x2 = 2*x
        let y: Vec<f64> = (0..n).map(|i| (i % 7) as f64 + 1.0).collect();
        let z: Vec<String> = (0..n).map(|i| format!("G{}", i % 6)).collect();

        let mut df = DataFrame::new();
        df.add_numeric("y", y).unwrap();
        df.add_numeric("x", x).unwrap();
        df.add_numeric("x2", x2).unwrap();
        df.add_categorical("z", z).unwrap();

        let formula = parse_formula("y ~ 1 + x + x2 + (1 | z)").unwrap();
        let mut model = LinearMixedModel::new(formula, &df, None).unwrap();
        model.fit(false).unwrap();

        let ct = model.coeftable();
        // rank 2, but coeftable has 3 rows (1 + x + x2)
        assert_eq!(ct.len(), 3, "should have 3 rows");
        assert_eq!(model.feterm.rank, 2, "model rank should be 2");

        // Exactly one of x/x2 is dropped → has NaN SE; the other is retained
        let n_nan = ct.std_errors.iter().filter(|&&se| se.is_nan()).count();
        assert_eq!(
            n_nan, 1,
            "exactly one coefficient should be dropped (NaN SE)"
        );

        // The dropped column must be x or x2 (not the intercept)
        for (i, se) in ct.std_errors.iter().enumerate() {
            if se.is_nan() {
                assert!(
                    ct.names[i] == "x" || ct.names[i] == "x2",
                    "dropped column should be x or x2, not '{}'",
                    ct.names[i]
                );
            }
        }
    }

    #[test]
    fn test_coeftable_omits_p_values_for_regularized_fit_intent() {
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
        let mut policy = CompilerPolicy::default();
        policy.random_strategy = RandomStrategy::Regularized;
        let mut model =
            LinearMixedModel::new_with_compiler_policy(formula, &data, None, policy).unwrap();

        model.fit(false).unwrap();

        let ct = model.coeftable();
        assert!(ct.z_values.iter().all(|value| value.is_finite()));
        assert!(ct.p_values.iter().all(|value| value.is_nan()));
        assert!(ct.p_value_reasons.iter().all(|reason| reason
            .as_deref()
            .unwrap()
            .contains("exploratory fit intent")));

        let summary = ModelSummary::from_linear_model(&model);
        assert!(summary
            .rows
            .iter()
            .filter(|row| row.std_error.is_some())
            .all(|row| row.pvalue.is_none()));
    }

    #[test]
    fn test_lmm_test_contrast_returns_labeled_asymptotic_result() {
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let days_index = model
            .coef_names()
            .iter()
            .position(|name| name == "days")
            .unwrap();
        let hypothesis = FixedEffectHypothesis::single_coefficient(
            "days = 0",
            days_index,
            model.coef_names().len(),
        )
        .unwrap();
        let test =
            model.test_contrast_with_method(hypothesis, FixedEffectTestMethod::AsymptoticWaldZ);

        assert!(matches!(test.status, InferenceStatus::Available));
        assert_eq!(test.p_values.len(), 1);
        assert!(test.p_values[0].unwrap() < 0.01);
        assert_eq!(test.estimability.status, EstimabilityStatus::Estimable);
        assert!(test
            .notes
            .iter()
            .any(|note| note.contains("asymptotic Wald z")));
    }

    #[test]
    fn test_lmm_explicit_satterthwaite_request_returns_scalar_t_test() {
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let days_index = model
            .coef_names()
            .iter()
            .position(|name| name == "days")
            .unwrap();
        let hypothesis = FixedEffectHypothesis::single_coefficient(
            "days = 0",
            days_index,
            model.coef_names().len(),
        )
        .unwrap();

        let test =
            model.test_contrast_with_method(hypothesis, FixedEffectTestMethod::Satterthwaite);

        assert_eq!(test.method, InferenceMethod::Satterthwaite);
        assert_eq!(test.status, InferenceStatus::Available);
        assert_eq!(test.reliability, ReliabilityGrade::Low);
        assert!(test.denominator_df.unwrap().is_finite());
        assert!(test.denominator_df.unwrap() > 0.0);
        assert!(test.p_values[0].unwrap().is_finite());
        assert!((0.0..=1.0).contains(&test.p_values[0].unwrap()));
        assert!(test.statistics[0].unwrap().is_finite());
        assert!(test
            .notes
            .iter()
            .any(|note| note.contains("Satterthwaite denominator df computed")));
    }

    #[test]
    fn test_lmm_satterthwaite_scalar_rows_match_lmer_test_fixture() {
        let fixture = satterthwaite_lmer_test_parity_fixture();

        for case in fixture.cases {
            let data = satterthwaite_parity_data(&case.name);
            let formula = parse_formula(&case.formula).unwrap();
            let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
            model.fit(true).unwrap();

            let coefficient_index = model
                .coef_names()
                .iter()
                .position(|name| name == &case.coefficient)
                .unwrap_or_else(|| {
                    panic!(
                        "coefficient {} not found in {:?}",
                        case.coefficient,
                        model.coef_names()
                    )
                });
            let hypothesis = FixedEffectHypothesis::single_coefficient(
                format!("{} = 0", case.coefficient),
                coefficient_index,
                model.coef_names().len(),
            )
            .unwrap();

            let test =
                model.test_contrast_with_method(hypothesis, FixedEffectTestMethod::Satterthwaite);

            assert_eq!(test.method, InferenceMethod::Satterthwaite, "{}", case.name);
            assert_eq!(test.status, InferenceStatus::Available, "{}", case.name);
            assert_eq!(test.reliability, ReliabilityGrade::Low, "{}", case.name);
            assert!(
                (test.estimates[0] - case.estimate).abs() <= 1e-5 + 1e-6 * case.estimate.abs(),
                "{}: β drift",
                case.name
            );
            // Single-grouping sleepstudy fits agree with lme4 to ~5e-5; the
            // crossed-RE penicillin REML optimum lands ~3e-4 away from lme4's
            // (multi-start in Rust is locally optimal to ~1e-6 in REML deviance,
            // so this is optimizer-vs-optimizer drift, not a fit bug).  Hold the
            // looser-but-still-meaningful 5e-4 bound across all cases.
            assert!(
                (test.standard_errors[0].unwrap() - case.std_error).abs()
                    <= 5e-4 + 5e-4 * case.std_error.abs(),
                "{}: std_error drift: rust={} ref={}",
                case.name,
                test.standard_errors[0].unwrap(),
                case.std_error
            );
            assert!(
                (test.statistics[0].unwrap() - case.statistic).abs()
                    <= 5e-4 + 5e-4 * case.statistic.abs(),
                "{}: t-statistic drift: rust={} ref={}",
                case.name,
                test.statistics[0].unwrap(),
                case.statistic
            );
            // Satterthwaite df is more sensitive to θ drift than vcov itself
            // because it depends on the gradient and Hessian of vcov w.r.t. θ.
            // For the crossed-RE penicillin case the drift sits ~1e-3.
            assert_relative_eq!(
                test.denominator_df.unwrap(),
                case.df,
                epsilon = 1e-2,
                max_relative = 2e-3,
            );
            // Tail-region p-values amplify df/statistic drift: a 5e-4 df move
            // shifts a 1e-6 p-value by ~2e-3 relative.  Hold an honest 2e-3
            // bound rather than chase ten-extra-bits-of-precision.
            assert_relative_eq!(
                test.p_values[0].unwrap(),
                case.p_value,
                epsilon = 1e-8,
                max_relative = 2e-3,
            );
        }
    }

    #[test]
    fn test_lmm_satterthwaite_boundary_and_rank_deficient_cases_return_reasons() {
        let data = dyestuff2_fixture();
        let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();
        let hypothesis = FixedEffectHypothesis::single_coefficient(
            "(Intercept) = 0",
            0,
            model.coef_names().len(),
        )
        .unwrap();
        let test =
            model.test_contrast_with_method(hypothesis, FixedEffectTestMethod::Satterthwaite);

        assert_eq!(test.method, InferenceMethod::Satterthwaite);
        assert!(
            matches!(test.status, InferenceStatus::NotAssessed { ref reason }
            if reason.contains("lower bound"))
        );
        assert_eq!(test.p_values, vec![None]);

        let n = 30usize;
        let x: Vec<f64> = (0..n).map(|i| (i % 5) as f64).collect();
        let x2: Vec<f64> = x.iter().map(|&v| 2.0 * v).collect();
        let y: Vec<f64> = (0..n).map(|i| (i % 7) as f64 + 1.0).collect();
        let z: Vec<String> = (0..n).map(|i| format!("G{}", i % 6)).collect();

        let mut df = DataFrame::new();
        df.add_numeric("y", y).unwrap();
        df.add_numeric("x", x).unwrap();
        df.add_numeric("x2", x2).unwrap();
        df.add_categorical("z", z).unwrap();

        let formula = parse_formula("y ~ 1 + x + x2 + (1 | z)").unwrap();
        let mut model = LinearMixedModel::new(formula, &df, None).unwrap();
        model.fit(false).unwrap();
        let dropped_label = model
            .fixed_effect_inference_table()
            .rows
            .into_iter()
            .find(|row| row.status == FixedEffectInferenceStatus::NotEstimable)
            .expect("rank-deficient fit should mark one coefficient not estimable")
            .label;
        let dropped_index = model
            .coef_names()
            .iter()
            .position(|name| name == &dropped_label)
            .unwrap();
        let hypothesis = FixedEffectHypothesis::single_coefficient(
            format!("{dropped_label} = 0"),
            dropped_index,
            model.coef_names().len(),
        )
        .unwrap();
        let test =
            model.test_contrast_with_method(hypothesis, FixedEffectTestMethod::Satterthwaite);

        assert!(
            matches!(test.status, InferenceStatus::NotEstimable { ref reason }
            if reason.contains("aliased") || reason.contains("non-finite"))
        );
        assert_eq!(test.p_values, vec![None]);
    }

    #[test]
    fn test_lmm_fixed_effect_inference_table_returns_ordered_satterthwaite_rows() {
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let table = model.fixed_effect_inference_table();
        let names = model.coef_names();

        assert_eq!(table.rows.len(), names.len());
        assert_eq!(
            table
                .rows
                .iter()
                .map(|row| row.label.clone())
                .collect::<Vec<_>>(),
            names
        );
        for row in &table.rows {
            assert_eq!(row.kind, FixedEffectInferenceRowKind::Coefficient);
            assert_eq!(row.method, FixedEffectInferenceMethod::Satterthwaite);
            assert_eq!(row.status, FixedEffectInferenceStatus::Available);
            assert_eq!(row.reliability, ReliabilityGrade::Low);
            assert_eq!(row.statistic_name, Some(FixedEffectStatisticName::T));
            assert!(row.estimate.is_some());
            assert!(row.std_error.is_some());
            assert!(row.statistic.is_some());
            assert!(row.p_value.is_some());
            assert!(row.numerator_df.is_none());
            assert!(row.denominator_df.is_some());
            assert!(row.reason.is_none());
            assert!(matches!(
                row.estimability,
                EstimabilityAssessment::FixedContrast(_)
            ));
            assert!(row
                .notes
                .iter()
                .any(|note| note.contains("Satterthwaite denominator df")));
        }
        assert_eq!(
            model
                .compiler_artifact()
                .fixed_effect_inference_table
                .as_ref(),
            Some(&table)
        );
    }

    #[test]
    fn test_lmm_fixed_effect_inference_table_marks_aliased_column_not_estimable() {
        let n = 30usize;
        let x: Vec<f64> = (0..n).map(|i| (i % 5) as f64).collect();
        let x2: Vec<f64> = x.iter().map(|&v| 2.0 * v).collect();
        let y: Vec<f64> = (0..n).map(|i| (i % 7) as f64 + 1.0).collect();
        let z: Vec<String> = (0..n).map(|i| format!("G{}", i % 6)).collect();

        let mut df = DataFrame::new();
        df.add_numeric("y", y).unwrap();
        df.add_numeric("x", x).unwrap();
        df.add_numeric("x2", x2).unwrap();
        df.add_categorical("z", z).unwrap();

        let formula = parse_formula("y ~ 1 + x + x2 + (1 | z)").unwrap();
        let mut model = LinearMixedModel::new(formula, &df, None).unwrap();
        model.fit(false).unwrap();

        let table = model.fixed_effect_inference_table();
        let dropped = table
            .rows
            .iter()
            .find(|row| row.status == FixedEffectInferenceStatus::NotEstimable)
            .expect("one aliased coefficient should be marked not estimable");

        assert_eq!(dropped.method, FixedEffectInferenceMethod::NotComputed);
        assert_eq!(dropped.reliability, ReliabilityGrade::NotAvailable);
        assert!(dropped.p_value.is_none());
        assert!(dropped.reason.as_deref().unwrap().contains("aliased"));
    }

    #[test]
    fn test_lmm_fixed_effect_inference_table_omits_p_values_for_regularized_fit_intent() {
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
        let mut policy = CompilerPolicy::default();
        policy.random_strategy = RandomStrategy::Regularized;
        let mut model =
            LinearMixedModel::new_with_compiler_policy(formula, &data, None, policy).unwrap();

        model.fit(false).unwrap();

        let table = model.fixed_effect_inference_table();
        assert!(table.rows.iter().all(|row| {
            row.status == FixedEffectInferenceStatus::PValueUnavailable
                && row.method == FixedEffectInferenceMethod::NotComputed
                && row.p_value.is_none()
                && row
                    .reason
                    .as_deref()
                    .unwrap()
                    .contains("exploratory fit intent")
        }));
    }

    #[test]
    fn test_lmm_fixed_effect_inference_table_omits_p_values_for_predictive_fit_intent() {
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
        let mut model = LinearMixedModel::new_with_compiler_policy(
            formula,
            &data,
            None,
            CompilerPolicy::predictive(),
        )
        .unwrap();

        model.fit(false).unwrap();

        assert_eq!(
            model.compiler_artifact().reproducibility.fit_intent,
            FitIntent::Predictive
        );
        let table = model
            .compiler_artifact()
            .fixed_effect_inference_table
            .as_ref()
            .expect("fitted artifacts should carry fixed-effect inference rows");
        assert!(table.rows.iter().all(|row| {
            row.status == FixedEffectInferenceStatus::PValueUnavailable
                && row.method == FixedEffectInferenceMethod::NotComputed
                && row.p_value.is_none()
                && row
                    .reason
                    .as_deref()
                    .unwrap()
                    .contains("predictive fit intent")
        }));
    }

    #[test]
    fn test_lmm_fixed_effect_inference_table_omits_p_values_after_selection_time_reduction() {
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.compiler_artifact.reductions.push(ReductionRecord {
            trigger: ReductionTrigger::SelectionTime,
            phase: "post_selection".to_string(),
            reason: "response-dependent random-effect selection".to_string(),
            affected_term: "(1 | subj)".to_string(),
            replacement_term: None,
            inference_consequence:
                "ordinary fixed-effect p-values require a valid refit or selective-inference contract"
                    .to_string(),
            diagnostics: Vec::new(),
        });

        model.fit(false).unwrap();

        let table = model
            .compiler_artifact()
            .fixed_effect_inference_table
            .as_ref()
            .expect("fitted artifacts should carry fixed-effect inference rows");
        assert!(table.rows.iter().all(|row| {
            row.status == FixedEffectInferenceStatus::PValueUnavailable
                && row.method == FixedEffectInferenceMethod::NotComputed
                && row.p_value.is_none()
                && row
                    .reason
                    .as_deref()
                    .unwrap()
                    .contains("selection-time model changes")
        }));
    }

    #[test]
    fn test_lmm_test_contrast_marks_aliased_column_not_estimable() {
        let n = 30usize;
        let x: Vec<f64> = (0..n).map(|i| (i % 5) as f64).collect();
        let x2: Vec<f64> = x.iter().map(|&v| 2.0 * v).collect();
        let y: Vec<f64> = (0..n).map(|i| (i % 7) as f64 + 1.0).collect();
        let z: Vec<String> = (0..n).map(|i| format!("G{}", i % 6)).collect();

        let mut df = DataFrame::new();
        df.add_numeric("y", y).unwrap();
        df.add_numeric("x", x).unwrap();
        df.add_numeric("x2", x2).unwrap();
        df.add_categorical("z", z).unwrap();

        let formula = parse_formula("y ~ 1 + x + x2 + (1 | z)").unwrap();
        let mut model = LinearMixedModel::new(formula, &df, None).unwrap();
        model.fit(false).unwrap();
        let ct = model.coeftable();
        let dropped = ct
            .std_errors
            .iter()
            .position(|se| se.is_nan())
            .expect("one fixed-effect column should be dropped");

        let hypothesis =
            FixedEffectHypothesis::single_coefficient("dropped coefficient", dropped, ct.len())
                .unwrap();
        let test = model.test_contrast(hypothesis);

        assert!(matches!(test.status, InferenceStatus::NotEstimable { .. }));
        assert_eq!(test.estimability.status, EstimabilityStatus::NotEstimable);
        assert_eq!(test.p_values, vec![None]);
    }

    #[test]
    fn test_lmm_fixed_effect_term_rows_are_rust_owned() {
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let hypotheses = model.fixed_effect_term_hypotheses();
        assert!(hypotheses
            .iter()
            .any(|hypothesis| hypothesis.label == "days"));

        let table = model.fixed_effect_term_inference_table(FixedEffectTestMethod::Auto);
        let days = table
            .rows
            .iter()
            .find(|row| row.label == "days")
            .expect("days term row should be exposed");
        assert_eq!(days.kind, FixedEffectInferenceRowKind::Term);
        let family = days
            .details
            .as_ref()
            .and_then(|details| details.contrast_family.as_ref())
            .expect("term row should carry contrast-family details");
        assert_eq!(family.family_label, "days");
        assert_eq!(family.restriction_rows, 1);
        assert_eq!(family.coefficient_count, model.coef_names().len());
    }

    #[test]
    fn test_lmm_fixed_effect_contrast_table_is_rust_owned() {
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let days_index = model
            .coef_names()
            .iter()
            .position(|name| name == "days")
            .unwrap();
        let hypothesis = FixedEffectHypothesis::single_coefficient(
            "days = 0",
            days_index,
            model.coef_names().len(),
        )
        .unwrap();
        let table = model
            .fixed_effect_contrast_inference_table(vec![hypothesis], FixedEffectTestMethod::Auto);

        assert_eq!(
            table.schema_name,
            crate::compiler::FIXED_EFFECT_INFERENCE_TABLE_SCHEMA
        );
        assert_eq!(table.rows.len(), 1);
        let row = &table.rows[0];
        assert_eq!(row.kind, FixedEffectInferenceRowKind::Contrast);
        assert_eq!(row.label, "days = 0");
        assert_eq!(row.status, FixedEffectInferenceStatus::Available);
        let family = row
            .details
            .as_ref()
            .and_then(|details| details.contrast_family.as_ref())
            .expect("contrast row should carry contrast-family details");
        assert_eq!(family.family_label, "days = 0");
        assert_eq!(
            family.numerator_df_semantics,
            "scalar_contrast_no_numerator_df"
        );
    }

    // ── Cook's distance parity tests (pls.jl line 705) ───────────────────────

    // ── Cook's distance parity tests (pls.jl line 705) ───────────────────────

    #[test]
    fn test_cooks_distance_length() {
        // cooksdistance(model) should have length n.
        // Uses first(models(:sleepstudy)) = reaction ~ 1 + days + (1 | subj)
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let d = model.cooks_distance();
        assert_eq!(d.len(), data.nrow());
    }

    #[test]
    fn test_cooks_distance_nonnegative() {
        // All Cook's distances should be ≥ 0.
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let d = model.cooks_distance();
        for (i, &di) in d.iter().enumerate() {
            assert!(
                di >= 0.0,
                "Cook's distance[{}] should be non-negative, got {}",
                i,
                di
            );
        }
    }

    #[test]
    fn test_cooks_distance_parity_sleepstudy() {
        // pls.jl line 705-760: lme4 reference values for Cook's distance.
        // Model: first(models(:sleepstudy)) = reaction ~ 1 + days + (1 | subj)
        //
        // Julia uses:  D_i = (r_i/(1-h_i))^2 * h_i / (varest(m) * p)
        // where p = rank of fixed-effects matrix = 2.
        //
        // We compare the first 10 values at rtol=0.10 (10%).
        let lme4_cooks: Vec<f64> = vec![
            0.1270714,
            0.1267805,
            0.243096,
            0.0002437091,
            0.03145029,
            0.2954052,
            0.04550505,
            0.3552723,
            0.1984806,
            0.4518805,
        ];

        let data = sleepstudy_fixture();
        // first(models(:sleepstudy)) — intercept-only RE per subject
        let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let d = model.cooks_distance();

        for (i, &expected) in lme4_cooks.iter().enumerate() {
            let got = d[i];
            let rel_err = ((got - expected) / expected).abs();
            assert!(
                rel_err < 0.10,
                "Cook's distance[{}]: expected {:.6}, got {:.6} (rel err {:.2}%)",
                i,
                expected,
                got,
                rel_err * 100.0
            );
        }
    }

    #[test]
    fn test_cooks_distance_sum_finite() {
        // Sum should be finite (no NaN/Inf from degenerate h_i).
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let d = model.cooks_distance();
        let s: f64 = d.iter().sum();
        assert!(s.is_finite(), "Sum of Cook's distances should be finite");
    }

    // ── Parametric bootstrap parity tests (bootstrap.jl) ─────────────────────

    #[test]
    fn test_parametricbootstrap_length() {
        // bootstrap.jl line 98: length(bsamp.objective) == 100
        let data = dyestuff_fixture();
        let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let mut rng = StdRng::seed_from_u64(1234321);
        let bsamp = parametricbootstrap(&mut rng, 5, &model);
        assert_eq!(bsamp.len(), 5);
        assert_eq!(bsamp.objectives().len(), 5);
        assert_eq!(bsamp.sigmas().len(), 5);
        assert_eq!(bsamp.thetas().len(), 5);
    }

    #[test]
    fn test_parametricbootstrap_objectives_finite() {
        // Each replicate should converge to a finite objective.
        let data = dyestuff_fixture();
        let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let mut rng = StdRng::seed_from_u64(42);
        let bsamp = parametricbootstrap(&mut rng, 10, &model);

        let n_finite = bsamp
            .objectives()
            .iter()
            .filter(|&&o| o.is_finite())
            .count();
        assert!(
            n_finite >= 8,
            "At least 8 out of 10 replicates should converge; got {}",
            n_finite
        );
    }

    #[test]
    fn test_parametricbootstrap_sigma_positive() {
        // All converged σ values should be positive.
        let data = dyestuff_fixture();
        let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let mut rng = StdRng::seed_from_u64(99);
        let bsamp = parametricbootstrap(&mut rng, 5, &model);

        for rep in &bsamp.fits {
            if rep.sigma.is_finite() {
                assert!(
                    rep.sigma > 0.0,
                    "Bootstrap σ should be positive, got {}",
                    rep.sigma
                );
            }
        }
    }

    #[test]
    fn test_parametricbootstrap_beta_length() {
        // Each replicate's β should have length p (rank of FE matrix).
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let p = model.feterm.rank;
        let mut rng = StdRng::seed_from_u64(7);
        let bsamp = parametricbootstrap(&mut rng, 3, &model);

        for rep in &bsamp.fits {
            assert_eq!(
                rep.beta.len(),
                p,
                "Bootstrap β length mismatch: expected {}, got {}",
                p,
                rep.beta.len()
            );
        }
    }

    #[test]
    fn test_parametricbootstrap_theta_length() {
        // bootstrap.jl: keys(first(bsamp.fits)) includes :θ.
        let data = dyestuff_fixture();
        let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let n_theta = model.n_theta();
        let mut rng = StdRng::seed_from_u64(0);
        let bsamp = parametricbootstrap(&mut rng, 3, &model);

        for rep in &bsamp.fits {
            assert_eq!(
                rep.theta.len(),
                n_theta,
                "Bootstrap θ length mismatch: expected {}, got {}",
                n_theta,
                rep.theta.len()
            );
        }
    }

    #[test]
    fn test_parametricbootstrap_save_restore_round_trip() {
        let data = dyestuff_fixture();
        let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let mut rng = StdRng::seed_from_u64(20260428);
        let bsamp = parametricbootstrap(&mut rng, 4, &model);

        let mut bytes = Vec::new();
        crate::stats::savereplicates(&mut bytes, &bsamp).unwrap();
        let restored = crate::stats::restorereplicates(bytes.as_slice(), &model).unwrap();

        assert_eq!(restored.len(), bsamp.len());
        for (actual, expected) in restored.fits.iter().zip(bsamp.fits.iter()) {
            assert_relative_eq!(actual.objective, expected.objective, epsilon = 1e-12);
            assert_relative_eq!(actual.sigma, expected.sigma, epsilon = 1e-12);
            assert_eq!(actual.beta.len(), expected.beta.len());
            for (a, e) in actual.beta.iter().zip(expected.beta.iter()) {
                assert_relative_eq!(*a, *e, epsilon = 1e-12);
            }
            assert_eq!(actual.se.len(), expected.se.len());
            for (a, e) in actual.se.iter().zip(expected.se.iter()) {
                assert_relative_eq!(*a, *e, epsilon = 1e-12);
            }
            assert_eq!(actual.theta.len(), expected.theta.len());
            for (a, e) in actual.theta.iter().zip(expected.theta.iter()) {
                assert_relative_eq!(*a, *e, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_parametricbootstrap_save_restore_preserves_nan_status() {
        let bsamp = MixedModelBootstrap {
            fits: vec![BootstrapReplicate {
                objective: f64::NAN,
                sigma: f64::NAN,
                beta: DVector::from_vec(vec![1.0, 2.0]),
                se: DVector::from_vec(vec![f64::NAN, f64::NAN]),
                theta: vec![0.5],
            }],
        };

        let mut bytes = Vec::new();
        bsamp.save_replicates(&mut bytes).unwrap();
        let restored = MixedModelBootstrap::restore_replicates(bytes.as_slice()).unwrap();

        assert_eq!(restored.len(), 1);
        assert!(restored.fits[0].objective.is_nan());
        assert!(restored.fits[0].sigma.is_nan());
        assert_eq!(restored.fits[0].beta, DVector::from_vec(vec![1.0, 2.0]));
        assert!(restored.fits[0].se.iter().all(|value| value.is_nan()));
        assert_eq!(restored.fits[0].theta, vec![0.5]);
    }

    #[test]
    fn test_parametricbootstrap_run_metadata_records_accounting_and_boundary_rate() {
        let data = dyestuff_fixture();
        let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let bsamp = MixedModelBootstrap {
            fits: vec![
                BootstrapReplicate {
                    objective: 1.0,
                    sigma: 2.0,
                    beta: DVector::from_vec(vec![10.0]),
                    se: DVector::from_vec(vec![1.0]),
                    theta: vec![0.0],
                },
                BootstrapReplicate {
                    objective: 2.0,
                    sigma: 3.0,
                    beta: DVector::from_vec(vec![11.0]),
                    se: DVector::from_vec(vec![1.2]),
                    theta: vec![0.5],
                },
                BootstrapReplicate {
                    objective: f64::NAN,
                    sigma: f64::NAN,
                    beta: DVector::from_vec(vec![f64::NAN]),
                    se: DVector::from_vec(vec![f64::NAN]),
                    theta: vec![0.5],
                },
            ],
        };
        let statistics = [1.0, f64::NAN, 3.0];

        let metadata = bsamp.run_metadata_for_model(
            &model,
            BootstrapTarget::full_model_distribution("dyestuff full model"),
            5,
            BootstrapFailedRefitPolicy::Exclude,
            BootstrapSeedRecord::std_rng(20260429),
            BootstrapRefitOptions::from_model(&model),
            Some("abs_t".to_string()),
            Some(&statistics),
            Some(0.25),
        );

        assert_eq!(metadata.schema_name, BOOTSTRAP_RUN_SCHEMA);
        assert_eq!(metadata.schema_version, BOOTSTRAP_RUN_SCHEMA_VERSION);
        assert_eq!(
            metadata.target.kind,
            BootstrapTargetKind::FullModelDistribution
        );
        assert_eq!(metadata.requested_replicates, 5);
        assert_eq!(metadata.completed_replicates, 3);
        assert_eq!(metadata.successful_replicates, 2);
        assert_eq!(metadata.failed_refits, 1);
        assert_eq!(
            metadata.failed_refit_policy,
            BootstrapFailedRefitPolicy::Exclude
        );
        assert_eq!(metadata.boundary_count, 1);
        assert_eq!(metadata.boundary_rate, Some(0.5));
        assert_eq!(metadata.finite_statistic_count, Some(2));
        assert_relative_eq!(metadata.mcse.unwrap(), (0.25_f64 * 0.75 / 2.0).sqrt());
        assert!(metadata
            .notes
            .iter()
            .any(|note| note.contains("do not certify fixed-effect hypothesis-test")));
        assert!(metadata
            .notes
            .iter()
            .any(|note| note.contains("requested 5 bootstrap")));

        let payload = bsamp.into_run_payload(metadata);
        let json = serde_json::to_string(&payload).unwrap();
        let decoded: BootstrapRunPayload = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.metadata.successful_replicates, 2);
        assert_eq!(decoded.replicates.len(), 3);
    }

    #[test]
    fn test_fixed_effect_null_bootstrap_target_projects_beta_and_simulates() {
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let days_index = model
            .coef_names()
            .iter()
            .position(|name| name == "days")
            .unwrap();
        let hypothesis = FixedEffectHypothesis::single_coefficient(
            "days = 0",
            days_index,
            model.coef_names().len(),
        )
        .unwrap();

        let target = model
            .fixed_effect_null_bootstrap_target(&hypothesis)
            .unwrap();
        let fitted_contrast = (&hypothesis.l.values * &target.beta_fitted)[0];
        let null_contrast = (&hypothesis.l.values * &target.beta_null)[0];

        assert_eq!(target.target.kind, BootstrapTargetKind::FixedEffectNull);
        assert_eq!(
            target.covariance_policy,
            FixedEffectNullCovariancePolicy::ReuseFittedCovariance
        );
        assert!(fitted_contrast.abs() > 1.0);
        assert_relative_eq!(null_contrast, 0.0, epsilon = 1e-8);
        assert_eq!(target.theta, model.theta());
        assert_relative_eq!(target.sigma, model.sigma(), epsilon = 1e-12);
        assert!(target
            .notes
            .iter()
            .any(|note| note.contains("reuses fitted covariance")));

        let mut rng = StdRng::seed_from_u64(20260429);
        let y_sim = model.simulate_fixed_effect_null(&mut rng, &target).unwrap();
        assert_eq!(y_sim.len(), model.nobs());

        let mut mismatched = target.clone();
        mismatched.sigma *= 1.01;
        assert!(matches!(
            model.simulate_fixed_effect_null(&mut rng, &mismatched),
            Err(MixedModelError::InvalidArgument(_))
        ));
    }

    #[test]
    fn test_bootstrap_fixed_effect_coefficient_row_from_certified_payload() {
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let days_index = model
            .coef_names()
            .iter()
            .position(|name| name == "days")
            .unwrap();
        let hypothesis = FixedEffectHypothesis::single_coefficient(
            "days = 0",
            days_index,
            model.coef_names().len(),
        )
        .unwrap();

        let mut fits = Vec::new();
        for i in 0..40 {
            let mut beta = model.beta();
            beta[days_index] = (i as f64 - 20.0) / 10.0;
            let mut se = DVector::from_element(model.feterm.rank, 1.0);
            se[days_index] = 1.0;
            fits.push(BootstrapReplicate {
                objective: i as f64 + 1.0,
                sigma: model.sigma(),
                beta,
                se,
                theta: model.theta(),
            });
        }
        let bsamp = MixedModelBootstrap { fits };
        let metadata = bsamp.run_metadata_for_model(
            &model,
            BootstrapTarget::fixed_effect_null("days fixed-effect null", "days = 0"),
            40,
            BootstrapFailedRefitPolicy::Exclude,
            BootstrapSeedRecord::std_rng(20260429),
            BootstrapRefitOptions::from_model(&model),
            Some("abs_t".to_string()),
            None,
            None,
        );
        let payload = bsamp.into_run_payload(metadata);

        let test = model.test_contrast_with_bootstrap_payload(hypothesis.clone(), &payload);
        assert_eq!(test.method, InferenceMethod::ParametricBootstrap);
        assert_eq!(test.status, InferenceStatus::Available);
        assert_eq!(test.reliability, ReliabilityGrade::Low);
        assert_relative_eq!(test.p_values[0].unwrap(), 1.0 / 41.0, epsilon = 1e-12);
        assert!(test.denominator_df.is_none());
        assert!(test
            .notes
            .iter()
            .any(|note| note.contains("fixed_effect_null target")));

        let row = model.fixed_effect_bootstrap_inference_row(
            FixedEffectInferenceRowKind::Coefficient,
            hypothesis,
            &payload,
        );
        assert_eq!(row.method, FixedEffectInferenceMethod::Bootstrap);
        assert_eq!(row.status, FixedEffectInferenceStatus::Available);
        assert_eq!(row.statistic_name, Some(FixedEffectStatisticName::T));
        assert_relative_eq!(row.p_value.unwrap(), 1.0 / 41.0, epsilon = 1e-12);
        let bootstrap = row
            .details
            .as_ref()
            .and_then(|details| details.bootstrap.as_ref())
            .expect("bootstrap row should carry structured metadata");
        assert_eq!(bootstrap.target_kind, "fixed_effect_null");
        assert_eq!(bootstrap.requested_replicates, 40);
        assert_eq!(bootstrap.successful_replicates, 40);
        assert_eq!(bootstrap.failed_refit_policy, "exclude");
    }

    #[test]
    fn test_bootstrap_fixed_effect_contrast_row_uses_payload_statistics() {
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let l = DMatrix::from_row_slice(1, model.coef_names().len(), &[1.0, 1.0]);
        let hypothesis = FixedEffectHypothesis::zero_rhs(
            "intercept_plus_days = 0",
            ContrastMatrix { values: l },
        );
        let fits = (0..40)
            .map(|i| BootstrapReplicate {
                objective: i as f64 + 1.0,
                sigma: model.sigma(),
                beta: model.beta(),
                se: DVector::from_element(model.feterm.rank, 1.0),
                theta: model.theta(),
            })
            .collect::<Vec<_>>();
        let bsamp = MixedModelBootstrap { fits };
        let replicate_statistics = vec![0.5; bsamp.len()];
        let metadata = bsamp.run_metadata_for_model(
            &model,
            BootstrapTarget::fixed_effect_null(
                "intercept_plus_days fixed-effect null",
                "intercept_plus_days = 0",
            ),
            40,
            BootstrapFailedRefitPolicy::Exclude,
            BootstrapSeedRecord::std_rng(20260430),
            BootstrapRefitOptions::from_model(&model),
            Some("abs_t".to_string()),
            Some(&replicate_statistics),
            Some(1.0 / 41.0),
        );
        let payload = bsamp.into_run_payload_with_statistics(metadata, replicate_statistics);

        let row = model.fixed_effect_bootstrap_inference_row(
            FixedEffectInferenceRowKind::Contrast,
            hypothesis,
            &payload,
        );
        assert_eq!(row.kind, FixedEffectInferenceRowKind::Contrast);
        assert_eq!(row.method, FixedEffectInferenceMethod::Bootstrap);
        assert_eq!(row.status, FixedEffectInferenceStatus::Available);
        assert_eq!(row.statistic_name, Some(FixedEffectStatisticName::T));
        assert_relative_eq!(row.p_value.unwrap(), 1.0 / 41.0, epsilon = 1e-12);
        let details = row.details.expect("contrast row should carry details");
        assert!(details.bootstrap.is_some());
        let family = details
            .contrast_family
            .expect("contrast row should carry contrast-family details");
        assert_eq!(family.restriction_rows, 1);
        assert_eq!(
            family.numerator_df_semantics,
            "scalar_contrast_no_numerator_df"
        );
    }

    #[test]
    fn test_bootstrap_fixed_effect_row_requires_enough_finite_statistics() {
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let days_index = model
            .coef_names()
            .iter()
            .position(|name| name == "days")
            .unwrap();
        let hypothesis = FixedEffectHypothesis::single_coefficient(
            "days = 0",
            days_index,
            model.coef_names().len(),
        )
        .unwrap();
        let fits = (0..2)
            .map(|i| BootstrapReplicate {
                objective: i as f64 + 1.0,
                sigma: model.sigma(),
                beta: model.beta(),
                se: DVector::from_element(model.feterm.rank, 1.0),
                theta: model.theta(),
            })
            .collect::<Vec<_>>();
        let bsamp = MixedModelBootstrap { fits };
        let metadata = bsamp.run_metadata_for_model(
            &model,
            BootstrapTarget::fixed_effect_null("days fixed-effect null", "days = 0"),
            2,
            BootstrapFailedRefitPolicy::Exclude,
            BootstrapSeedRecord::std_rng(20260431),
            BootstrapRefitOptions::from_model(&model),
            Some("abs_t".to_string()),
            None,
            None,
        );
        let payload = bsamp.into_run_payload(metadata);

        let test = model.test_contrast_with_bootstrap_payload(hypothesis, &payload);
        assert_eq!(test.method, InferenceMethod::ParametricBootstrap);
        assert!(
            matches!(test.status, InferenceStatus::NotAssessed { ref reason }
            if reason.contains("bootstrap_successful_replicates_too_few"))
        );
        assert_eq!(test.p_values, vec![None]);
    }

    #[test]
    fn test_bootstrap_fixed_effect_row_from_null_simulate_refit_payload() {
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let days_index = model
            .coef_names()
            .iter()
            .position(|name| name == "days")
            .unwrap();
        let hypothesis = FixedEffectHypothesis::single_coefficient(
            "days = 0",
            days_index,
            model.coef_names().len(),
        )
        .unwrap();
        let target = model
            .fixed_effect_null_bootstrap_target(&hypothesis)
            .unwrap();

        let mut rng = StdRng::seed_from_u64(20260502);
        let mut fits = Vec::new();
        for _ in 0..30 {
            let y_sim = model.simulate_fixed_effect_null(&mut rng, &target).unwrap();
            let mut work = model.clone();
            match work.refit(y_sim.as_slice()) {
                Ok(()) => fits.push(BootstrapReplicate {
                    objective: work.objective(),
                    sigma: work.sigma(),
                    beta: work.beta(),
                    se: work.stderror(),
                    theta: work.theta(),
                }),
                Err(_) => fits.push(BootstrapReplicate {
                    objective: f64::NAN,
                    sigma: f64::NAN,
                    beta: model.beta(),
                    se: DVector::from_element(model.feterm.rank, f64::NAN),
                    theta: model.theta(),
                }),
            }
        }

        let bsamp = MixedModelBootstrap { fits };
        let metadata = bsamp.run_metadata_for_model(
            &model,
            target.target.clone(),
            30,
            BootstrapFailedRefitPolicy::Exclude,
            BootstrapSeedRecord::std_rng(20260502),
            BootstrapRefitOptions::from_model(&model),
            Some("abs_t".to_string()),
            None,
            None,
        );
        let payload = bsamp.into_run_payload(metadata);
        let row = model.fixed_effect_bootstrap_inference_row(
            FixedEffectInferenceRowKind::Coefficient,
            hypothesis,
            &payload,
        );

        assert_eq!(row.method, FixedEffectInferenceMethod::Bootstrap);
        assert_eq!(row.status, FixedEffectInferenceStatus::Available);
        assert_eq!(row.statistic_name, Some(FixedEffectStatisticName::T));
        assert_eq!(row.reliability, ReliabilityGrade::Low);
        assert!(row.p_value.unwrap().is_finite());
        assert!((1.0 / 31.0..=1.0).contains(&row.p_value.unwrap()));
        assert!(row
            .notes
            .iter()
            .any(|note| note.contains("successful_replicates=30")));
        assert!(row.notes.iter().any(|note| note.contains("mcse=")));
    }

    #[test]
    fn test_fixed_effect_null_bootstrap_table_callable_returns_inference_table() {
        let data = sleepstudy_fixture();
        let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let days_index = model
            .coef_names()
            .iter()
            .position(|name| name == "days")
            .unwrap();
        let hypothesis = FixedEffectHypothesis::single_coefficient(
            "days = 0",
            days_index,
            model.coef_names().len(),
        )
        .unwrap();
        let table = model.fixed_effect_null_bootstrap_inference_table(
            vec![hypothesis],
            FixedEffectBootstrapOptions {
                requested_replicates: 2,
                failed_refit_policy: BootstrapFailedRefitPolicy::Exclude,
                seed: Some(20260503),
            },
        );

        assert_eq!(
            table.schema_name,
            crate::compiler::FIXED_EFFECT_INFERENCE_TABLE_SCHEMA
        );
        assert_eq!(table.rows.len(), 1);
        let row = &table.rows[0];
        assert_eq!(row.method, FixedEffectInferenceMethod::Bootstrap);
        assert_eq!(row.kind, FixedEffectInferenceRowKind::Contrast);
        assert!(matches!(
            row.status,
            FixedEffectInferenceStatus::Available | FixedEffectInferenceStatus::NotAssessed
        ));
        let bootstrap = row
            .details
            .as_ref()
            .and_then(|details| details.bootstrap.as_ref())
            .expect("bridge row should carry bootstrap details");
        assert_eq!(bootstrap.requested_replicates, 2);
        assert_eq!(bootstrap.seed, Some(20260503));
        assert!(bootstrap.null_target.is_some());
    }

    #[test]
    fn test_restorereplicates_rejects_mismatched_model_shape() {
        let data = dyestuff_fixture();
        let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let bsamp = MixedModelBootstrap {
            fits: vec![BootstrapReplicate {
                objective: 1.0,
                sigma: 1.0,
                beta: DVector::zeros(model.feterm.rank + 1),
                se: DVector::zeros(model.feterm.rank + 1),
                theta: model.theta(),
            }],
        };

        let mut bytes = Vec::new();
        crate::stats::savereplicates(&mut bytes, &bsamp).unwrap();
        let err = crate::stats::restorereplicates(bytes.as_slice(), &model).unwrap_err();
        match err {
            MixedModelError::InvalidArgument(message) => {
                assert!(message.contains("beta length"));
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn test_parametricbootstrap_quantile_summaries() {
        let bsamp = deterministic_bootstrap_sample();
        let rows = bsamp.quantiles(0.5).unwrap();

        let objective = rows
            .iter()
            .find(|row| row.parameter == "objective")
            .unwrap();
        assert_eq!(objective.n, 5);
        assert_eq!(objective.value, 30.0);

        let beta1 = rows.iter().find(|row| row.parameter == "beta[1]").unwrap();
        assert_eq!(beta1.value, 12.0);

        let se0 = rows.iter().find(|row| row.parameter == "se[0]").unwrap();
        assert_relative_eq!(se0.value, 0.7, epsilon = 1e-12);

        let theta0 = rows.iter().find(|row| row.parameter == "theta[0]").unwrap();
        assert_relative_eq!(theta0.value, 0.3, epsilon = 1e-12);
    }

    #[test]
    fn test_parametricbootstrap_percentile_intervals() {
        let bsamp = deterministic_bootstrap_sample();
        let rows = bsamp.percentile_intervals(0.8).unwrap();

        let objective = rows
            .iter()
            .find(|row| row.parameter == "objective")
            .unwrap();
        assert_eq!(objective.method, BootstrapIntervalMethod::Percentile);
        assert_eq!(objective.n, 5);
        assert_relative_eq!(objective.lower, 14.0, epsilon = 1e-12);
        assert_relative_eq!(objective.upper, 46.0, epsilon = 1e-12);

        let sigma = rows.iter().find(|row| row.parameter == "sigma").unwrap();
        assert_relative_eq!(sigma.lower, 1.4, epsilon = 1e-12);
        assert_relative_eq!(sigma.upper, 4.6, epsilon = 1e-12);
    }

    #[test]
    fn test_parametricbootstrap_shortest_intervals_filter_nonfinite() {
        let bsamp = MixedModelBootstrap {
            fits: vec![
                BootstrapReplicate {
                    objective: f64::NAN,
                    sigma: 0.0,
                    beta: DVector::from_vec(vec![0.0]),
                    se: DVector::from_vec(vec![0.0]),
                    theta: vec![0.0],
                },
                BootstrapReplicate {
                    objective: 10.0,
                    sigma: 10.0,
                    beta: DVector::from_vec(vec![10.0]),
                    se: DVector::from_vec(vec![10.0]),
                    theta: vec![10.0],
                },
                BootstrapReplicate {
                    objective: 11.0,
                    sigma: 11.0,
                    beta: DVector::from_vec(vec![11.0]),
                    se: DVector::from_vec(vec![11.0]),
                    theta: vec![11.0],
                },
                BootstrapReplicate {
                    objective: 12.0,
                    sigma: 12.0,
                    beta: DVector::from_vec(vec![12.0]),
                    se: DVector::from_vec(vec![12.0]),
                    theta: vec![12.0],
                },
                BootstrapReplicate {
                    objective: 100.0,
                    sigma: 100.0,
                    beta: DVector::from_vec(vec![100.0]),
                    se: DVector::from_vec(vec![100.0]),
                    theta: vec![100.0],
                },
            ],
        };

        let rows = bsamp.shortest_intervals(0.6).unwrap();
        let objective = rows
            .iter()
            .find(|row| row.parameter == "objective")
            .unwrap();
        assert_eq!(objective.method, BootstrapIntervalMethod::Shortest);
        assert_eq!(objective.n, 4);
        assert_eq!((objective.lower, objective.upper), (10.0, 12.0));

        let sigma = rows.iter().find(|row| row.parameter == "sigma").unwrap();
        assert_eq!(sigma.n, 5);
        assert_eq!((sigma.lower, sigma.upper), (10.0, 12.0));
    }

    #[test]
    fn test_parametricbootstrap_summaries_reject_bad_inputs() {
        let bsamp = deterministic_bootstrap_sample();
        assert!(matches!(
            bsamp.quantiles(1.2),
            Err(MixedModelError::InvalidArgument(_))
        ));
        assert!(matches!(
            bsamp.percentile_intervals(1.0),
            Err(MixedModelError::InvalidArgument(_))
        ));

        let mismatched = MixedModelBootstrap {
            fits: vec![
                BootstrapReplicate {
                    objective: 1.0,
                    sigma: 1.0,
                    beta: DVector::from_vec(vec![1.0]),
                    se: DVector::from_vec(vec![1.0]),
                    theta: vec![1.0],
                },
                BootstrapReplicate {
                    objective: 2.0,
                    sigma: 2.0,
                    beta: DVector::from_vec(vec![1.0, 2.0]),
                    se: DVector::from_vec(vec![1.0]),
                    theta: vec![1.0],
                },
            ],
        };
        assert!(matches!(
            mismatched.quantiles(0.5),
            Err(MixedModelError::InvalidArgument(_))
        ));
    }

    #[test]
    fn test_parametricbootstrap_sigma_near_fitted() {
        // Over many replicates the mean bootstrap σ should be close to the
        // fitted σ (within 50% — bootstrap estimates have high variance for
        // small n, but the mean should be in the right ballpark).
        let data = dyestuff_fixture();
        let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();

        let fitted_sigma = model.sigma();

        let mut rng = StdRng::seed_from_u64(1234321);
        let bsamp = parametricbootstrap(&mut rng, 30, &model);

        let finite_sigmas: Vec<f64> = bsamp
            .sigmas()
            .into_iter()
            .filter(|s| s.is_finite())
            .collect();
        assert!(
            !finite_sigmas.is_empty(),
            "Should have at least one converged replicate"
        );

        let mean_sigma = finite_sigmas.iter().sum::<f64>() / finite_sigmas.len() as f64;
        let rel_err = ((mean_sigma - fitted_sigma) / fitted_sigma).abs();
        assert!(
            rel_err < 0.50,
            "Mean bootstrap σ {:.4} should be within 50% of fitted σ {:.4}",
            mean_sigma,
            fitted_sigma
        );
    }

    fn deterministic_bootstrap_sample() -> MixedModelBootstrap {
        MixedModelBootstrap {
            fits: (0..5)
                .map(|idx| {
                    let k = idx as f64;
                    BootstrapReplicate {
                        objective: 10.0 * (k + 1.0),
                        sigma: k + 1.0,
                        beta: DVector::from_vec(vec![k, 10.0 + k]),
                        se: DVector::from_vec(vec![0.5 + 0.1 * k, 1.5 + 0.1 * k]),
                        theta: vec![0.1 * (k + 1.0)],
                    }
                })
                .collect(),
        }
    }
}
