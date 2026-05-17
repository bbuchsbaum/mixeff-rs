# Linalg Primitive Audit

Status: post-1.0 decision record for `bd-01KRNPZAT86D69A9G7BJVFNQEA`.

This note records the current state of the internal `src/linalg` primitives
that were ported from MixedModels.jl but are not all used by the active LMM fit
path. It is intentionally not a rewrite plan. The 1.0 decision is to keep the
tested ports in place under the documented `#[allow(dead_code)]` gate and
revisit wiring or deletion after release.

## Current Caller State

The original mote said the ported primitives had zero non-test callers. That is
no longer exactly true.

| Module | Public items | Non-test callers today | Current interpretation |
| --- | --- | --- | --- |
| `linalg::pivot` | `pivoted_qr`, `pivoted_qr_with_tol`, `stats_rank`, `stats_rank_with_tol` | Yes: `compiler::audit` uses `pivoted_qr_with_tol` and `stats_rank_with_tol`; `stats::lrt` and `types::fe_term` use the `stats_rank` facade. Only the no-tol `pivoted_qr` wrapper has no non-test caller. | Keep. This is active compiler/statistics infrastructure, so the module-level `#[allow(dead_code)]` was removed; the allowance is now scoped to the single uncalled `pivoted_qr` wrapper in `pivot.rs`. The MGS-vs-Householder parity question remains. |
| `linalg::chol_unblocked` | `chol_unblocked`, `chol_unblocked_diag`, `chol_unblocked_blocks` | No non-test caller found. | Retain as a tested reference port until the post-1.0 numerical-engine decision. Do not wire into `model::linear` without benchmark and parity evidence. |
| `linalg::logdet` | `logdet_triangular`, `logdet_diag`, `logdet_block_diagonal`, `logdet_from_chol`, `logdet_block_diagonal_from_chol` | No non-test caller found. | Retain as simple tested reference functions. The active fit path has specialized `MatrixBlock` logdet code. |
| `linalg::rank_update` | `rank_update_dense`, `rank_update_diag`, `rank_update_sparse_dense`, `rank_update_sparse` | No non-test caller found. | Retain as tested MixedModels.jl-style ports. Wiring these into the fit path would be a numerical refactor, not cleanup. |
| `linalg::block_ops` | `copy_scale_inflate`, `copy_scale_offdiag`, `copy_rmul_lambda`, `lmul_lambda_transpose`, `rmul_lambda` | No non-test caller found. | Retain as tested reference block operations. The active blocked PLS path has bespoke code in `model::linear`. |

## Decision Boundary

For 1.0, keep the current arrangement:

- `linalg` stays internal, not public API.
- `pivot` remains active because rank and audit code now depend on it; its
  module-level dead_code allowance was removed and scoped down to the single
  uncalled `pivoted_qr` wrapper so the gate reflects reality.
- unused-but-tested ports remain available for post-1.0 numerical work.
- the documented `#[allow(dead_code)]` stays on the four internal modules
  (`block_ops`, `chol_unblocked`, `logdet`, `rank_update`) whose only current
  callers are unit tests.

The follow-up decision should be made only after comparing three options:

1. keep the reference ports as internal test-backed numerical scaffolding;
2. wire specific functions into `model::linear` with parity and performance
   evidence;
3. delete truly unused ports when no roadmap item needs them.

## Risks If We Wire Them In

Wiring these primitives into the fit path is not a mechanical cleanup. It could
change:

- Cholesky zero-padding and singularity behavior;
- block ordering and determinant accounting;
- sparse/dense promotion behavior;
- performance of the profiled objective and TrustBQ benchmark rows.

Any wire-in attempt should carry at least:

- unit tests for the primitive;
- a model-level parity test on sleepstudy or another known fixture;
- a difficult-model/pathology row if singularity behavior changes;
- an objective/factorization benchmark before and after the change.

## Risks If We Delete Them

Deletion would reduce local code surface, but it would also remove tested
reference implementations that are useful for:

- comparing the bespoke `model::linear` kernels against simpler algebra;
- future Householder/pivoted-QR work;
- future active-face or MMTrust-PSD prototypes;
- diagnosing rank-update or log-determinant regressions.

Given the current release posture, deletion is lower-value than keeping the
tested ports internal and documented.

