# Audit 01 — Numerical Core (linalg/ + types/)

Auditor 1/7, release-candidate hardening pass. READ-ONLY.
Scope: `src/linalg/*.rs`, `src/types/*.rs`. Cross-checked against `MixedModels.jl/`.

## Live-path map (established before judging severity)

- **LIVE / critical:** `linalg/pivot.rs` (`stats_rank`, `stats_rank_with_tol`,
  `pivoted_qr_with_tol`) consumed by `compiler::audit`, `stats::lrt`,
  `types::fe_term::FeTerm`. `types::re_mat::ReMat` (`set_theta`/`get_theta`/
  `reweight`/`lambda`/`refs`/`inds`/`lower_triangular_indices`/
  `linear_to_subscript`) consumed throughout `model/linear.rs` and
  `model/generalized.rs`. `types::gauss_hermite::gh_norm` consumed by GLMM AGQ.
  `types::opt_summary::OptSummary` consumed by the fit drivers.
  `types::matrix_block` (`block_index`, `with_block_triple`, etc.) consumed by
  `model/linear.rs`/`stats`.
- **NOT in active fit path (tested ports, no non-test caller):**
  `linalg/chol_unblocked.rs`, `linalg/rank_update.rs`, `linalg/logdet.rs`,
  `linalg/block_ops.rs` (linear.rs carries its own `copy_scale_inflate` /
  blocked Cholesky), and `ReMat::sparse()` + `adj_a` + `nzs_as_mat`,
  `types::blocked_sparse::BlockedSparse`, `types::uniform_block_diagonal`,
  `types::ragged_array`. This is documented in `linalg/mod.rs` and
  `docs/linalg_primitive_audit.md`. Severity for defects in these is capped
  because they cannot affect a released fit, but they are still reported
  because they are public API and parity reference.

No `unsafe` blocks exist anywhere in scope. Commendable.

## Severity summary

- CRITICAL: 0
- HIGH: 1
- MEDIUM: 4
- LOW: 6
- Commendations: 6

No CRITICAL/HIGH at HIGH confidence ⇒ this slice does not by itself make the RC
unwise, but the HIGH item (a Rust-only rank shortcut with no Julia counterpart)
should be resolved or explicitly contract-documented before release.

---

## HIGH

### H1 — `quick_full_rank_identity_pivot` is a Rust-only rank criterion with no Julia counterpart (parity / correctness risk on the live path)
File: `src/linalg/pivot.rs:211`, `:257-295`
Confidence: MEDIUM

`stats_rank_with_tol` short-circuits through `quick_full_rank_identity_pivot`
for `n == 1` and `n == 2` *before* the pivoted-QR path. Julia's `statsrank`
(`MixedModels.jl/src/linalg/pivot.jl`) has **no such shortcut**: it always runs
`pivoted_qr` and derives rank from `abs.(diag(R))`.

The shortcut's full-rank test for `n == 2` is
`sigma_min > (ranktol * sigma_max * 100.0).max(f64::EPSILON.sqrt())`
(`pivot.rs:293-294`). Two divergences from the QR criterion
(`|R[i,i]| > ranktol * |R[0,0]|`, `compute_rank_from_r:159`):

1. The threshold is inflated **100×** (`ranktol * sigma_max * 100.0`).
2. An absolute floor `sqrt(eps) ≈ 1.49e-8` is applied regardless of column
   scale (`.max(f64::EPSILON.sqrt())`).

Consequence: a 2-column design that is *marginally* rank-deficient under the
`ranktol=1e-8` QR criterion (the value `compiler::audit` and `stats::lrt` pass)
can be reported full-rank `(2, [0,1])` by the shortcut, while the Julia
reference (and the crate's own QR path) report rank 1. Because `FeTerm::new`,
`compiler::audit` (`audit.rs:1294`, design-rank/identifiability gate) and
`stats::lrt` (`lrt.rs:54,1036,1055`, df computation for likelihood-ratio tests)
all consume `stats_rank`, a missed rank deficiency silently changes residual
df, the X'X factorization path, and LRT degrees of freedom — exactly the
"no hidden model surgery / explicit identifiability" stance in
`docs/mixed_model_compiler_inference_contract.md`.

Reproduction reasoning: construct a 2-column `n×2` X whose smaller singular
value σ_min satisfies `1e-8·σ_max < σ_min < 1.49e-8` (between the true QR
tolerance and the `sqrt(eps)` floor), e.g. two nearly-collinear scaled
columns. The shortcut returns full rank; `pivoted_qr_with_tol(...,1e-8)` on the
same matrix returns rank 1. Disagreement is a real, constructible input, not
just floating-point noise.

Suggested fix: either (a) delete the shortcut and always go through
`pivoted_qr_with_tol` (closest to Julia, simplest to certify), or (b) make the
shortcut's accept condition *provably no looser* than `compute_rank_from_r`
(drop the `*100.0` and the `sqrt(eps)` floor; use `ranktol * sigma_max`) AND
add a test that fuzzes random 1- and 2-column matrices asserting
`stats_rank == rank-from-pivoted_qr` for both. Also add a near-boundary parity
test against a Julia-dumped expectation.

---

## MEDIUM

### M1 — `stats_rank` rank count uses strict `>` where Julia's `searchsortedlast` is inclusive `≥` at the tolerance boundary
File: `src/linalg/pivot.rs:229` (`filter(|&&d| d > cmp).count()`),
also `:224`, and `compute_rank_from_r:173` (`> threshold`)
Confidence: LOW

Julia: `rank = searchsortedlast(dvec, cmp; rev=true)` over the non-increasing
`dvec = abs.(diag(R))`. `searchsortedlast(...; rev=true)` returns the count of
entries `≥ cmp` (boundary inclusive). Rust counts entries strictly `> cmp`.
A column whose pivoted R-diagonal magnitude is *exactly* `ranktol·|R[0,0]|`
is kept full-rank by Julia and dropped by Rust. This is an exact-equality
edge at the tolerance boundary — unlikely with real data but a deterministic
parity divergence if a Julia parity fixture lands on it, and it compounds H1.

Fix: use `>=` to match `searchsortedlast` inclusivity in both `:229` and the
`:224` `last(dvec) > cmp` early-out and `compute_rank_from_r:173`, or document
the intentional strictness in the parity contract with a Julia-cross-checked
fixture proving no current fixture is affected.

### M2 — `chol_unblocked` singular-tolerance differs from Julia's exact `< 0` test (numerical-parity, capped: not in fit path)
File: `src/linalg/chol_unblocked.rs:48-73`
Confidence: MEDIUM

Julia `cholUnblocked!(::StridedMatrix)` (`linalg/cholUnblocked.jl`) for n≥3
calls LAPACK `potrf!` and throws `PosDefException` on the *first non-positive*
pivot (`d < 0`), with **no zero-row fallback** — the n==1/n==2 hand-rolled
branches likewise throw on `A[i] < 0`. The Rust port instead introduces a
relative tolerance `tol = n·eps·max_diag` and *zeros the row* when
`-tol ≤ d ≤ tol` (`:63-73`), only erroring when `d < -tol`. This is a
deliberate, documented behavior change ("singular covariance" handling) and is
defensible, but it is **not** what the cited Julia reference does, so any
parity dump touching a singular block will diverge (Julia errors, Rust
continues with a rank-deficient L). Capped to MEDIUM because `chol_unblocked`
has no non-test caller (linear.rs uses its own factorization). Flag: the
zero-row path means a later `logdet_from_chol` over this L will compute
`ln(0) = -inf` (see L3) — the contract for "singular → zeroed row" must define
what the determinant of such a factor means.
Fix: document the intentional divergence in `docs/linalg_primitive_audit.md`
and either keep these ports out of any parity fixture or add a Julia-side
shim; if they are ever wired into the fit path, the tolerance and zero-row
semantics need a dedicated parity contract.

### M3 — `triplet_to_csc` via `CooMatrix` sums duplicates but ordering/zeros differ from Julia `sparse(II,J,V)` (capped: `ReMat::sparse` not in fit path)
File: `src/types/re_mat.rs:254-279`, `:386-394`
Confidence: LOW

`ReMat::sparse()` filters `if val != 0.0` before pushing triplets (`:272`),
so structural zeros produced by `Λ'·z` are dropped from the pattern. Julia's
`adjA`/`sparse(A::ReMat)` builds the pattern from `vec(z)` *without* dropping
zeros, so the sparsity pattern (and any downstream code that iterates
structural nonzeros) can differ for designs with exact-zero covariate cells.
Not in the active fit path (no caller of `ReMat::sparse`), hence capped.
Fix: either keep explicit zeros to match Julia's pattern, or document that
`ReMat::sparse` returns a value-pruned pattern and is not parity-faithful.

### M4 — `gh_norm` eigen-derived nodes are not antisymmetrized to exact parity for general k vs Julia’s `SVector` path
File: `src/types/gauss_hermite.rs:50-96`
Confidence: LOW

The construction mirrors Julia (symmetrize `z = (v - reverse(v))/2`,
`w = normalize((w+reverse(w))/2, 1)`), and k=1,2,3 are hard-coded identically.
Residual risk: nalgebra `symmetric_eigen` eigenvector sign/order conventions
differ from Julia/LAPACK `eigen(SymTridiagonal(...))`. Signs cancel under
`v*v`, and the explicit ascending sort + symmetrization remove ordering and
sign sensitivity, so this is likely fine — but there is no parity test against
Julia for k ∈ {5,7,9,11} (only internal symmetry / sum-to-one checks). Since
`gh_norm` is live for GLMM AGQ (`generalized.rs:1118`), determinism of the
AGQ objective depends on it.
Fix: add a parity fixture comparing `gh_norm(k).z/.w` to Julia `GHnorm(k)` for
k up to the max AGQ points the crate supports.

---

## LOW

### L1 — `gauss_hermite.rs:76` `partial_cmp(...).unwrap()` panics on NaN eigenvalue
File: `src/types/gauss_hermite.rs:76`
Confidence: HIGH (mechanism), LOW (reachability)
`indices.sort_by(|a,b| eigenvalues[a].partial_cmp(&eigenvalues[b]).unwrap())`
panics if any eigenvalue is NaN. For a finite symmetric tridiagonal with
`sqrt(1..k-1)` off-diagonals this cannot occur, so practically unreachable,
but it is an `unwrap` on `partial_cmp` (the canonical Rust footgun). Fix:
`.unwrap_or(std::cmp::Ordering::Equal)` or `total_cmp`.

### L2 — `gh_norm` panics (`assert!(k > 0)`) — reachable only from internal callers
File: `src/types/gauss_hermite.rs:51`
Confidence: HIGH
`GaussHermiteNormalized::new(0)` / `gh_norm(0)` panics. `n_agq` is a `usize`
in `OptSummary` defaulting to 1; if any path forwards a user `nAGQ=0` it
panics rather than erroring. Confirm callers clamp `n_agq >= 1`; otherwise
return `Result`. (Out-of-scope to fully trace; flag for the GLMM auditor.)

### L3 — `logdet_triangular` returns `-inf`/`NaN` silently for zero/negative diagonal
File: `src/linalg/logdet.rs:21-35`, `:70-72`
Confidence: HIGH (mechanism), LOW (impact — not in fit path)
`s += l[(i,i)].ln()` yields `-inf` for a zero diagonal (the exact output of the
M2 singular-row path) and `NaN` for a negative diagonal, with no guard and no
error. `logdet_from_chol`/`logdet_block_diagonal` propagate it into an
objective. Not in the active fit path, but if these ports are ever wired in,
a singular block silently poisons the deviance. Fix: when used for an
objective, detect non-finite contributions and return
`Result`/`LinAlgError` rather than `-inf`.

### L4 — `rank_update_sparse` lower-triangle branch can write the upper triangle for unsorted CSC rows
File: `src/linalg/rank_update.rs:210-222`
Confidence: MEDIUM (mechanism), LOW (impact — not in fit path)
The inner loop assumes column row-indices are ascending so that `kk >= jj`
implies `row_k >= row_j`; it then *defensively* swaps to `(row_j, row_k)` when
not. Julia's `rankUpdate!(::SparseMatrixCSC)` writes `Cd[rv[kk], rvj]`
unconditionally relying on sorted `rowval`. The Rust swap means: if CSC rows
are unsorted, Rust scatters some contributions into the *upper* triangle of C
while the rest of the codebase reads only the lower triangle ⇒ silently lost
mass, not a panic. nalgebra-sparse CSC is normally sorted, so low impact and
not in fit path. Fix: assert sorted row indices (debug_assert at minimum) and
drop the swap, matching Julia exactly; or always normalize to lower triangle
`C[max,min]`.

### L5 — `UniformBlockDiagonal::Index` returns `&'static ZERO`; mutation through other APIs cannot reach off-block, but the asymmetry is a footgun
File: `src/types/uniform_block_diagonal.rs:159-181`
Confidence: HIGH (mechanism), LOW (impact — not in fit path)
`Index<(usize,usize)>` returns `&ZERO` for off-block positions and panics
nowhere even for out-of-bounds indices (unlike `get`, which asserts bounds at
`:91-98`). So `ubd[(huge, huge)]` returns `0.0` silently while `ubd.get(huge,
huge)` panics — inconsistent bounds contract between the two accessors. Not in
fit path. Fix: bounds-check in `index` to match `get`, or document the
divergence.

### L6 — Integer size math `vsize * n_levels`, `m*m*k`, `i*(i+1)/2+j` unguarded against overflow
File: `src/types/re_mat.rs:362` (`vsize*n_levels`), `:155`;
`src/types/uniform_block_diagonal.rs:58,80`;
`src/types/matrix_block.rs:103-106` (`block_index`)
Confidence: LOW
These are `usize` products with no `checked_mul`. On 64-bit, realistic model
sizes cannot overflow, so this is LOW, but `block_index(i,j)=i*(i+1)/2+j` is
on the live path and `i*(i+1)` overflows for `i ≳ 4.3e9` (irrelevant for
real block counts) — and more practically, `block_index` only `debug_assert!`s
`i >= j` (`matrix_block.rs:104`): in release a caller passing `i < j` gets a
silently wrong packed index, not a panic. Fix: keep `debug_assert` but
consider a `#[track_caller]` `assert!` for `block_index` since it indexes the
core blocked-Cholesky storage; wrong index ⇒ silent wrong factor.

---

## Commendations

1. No `unsafe` anywhere in scope.
2. `chol_unblocked` correctly rejects non-finite diagonals (`:59-61`) and has
   thorough singular/zero/NaN/Inf tests.
3. `pivot.rs` Householder + LAPACK partial-norm downdate with the
   reorthogonalization safeguard (`:127-145`) is a faithful, well-documented
   `xLAQPS` port; the intercept-preservation re-run mirrors Julia
   `pivot.jl:27-34` correctly, and the `dead_code` scoping in `linalg/mod.rs`
   is honest about what is and isn't live.
4. `matrix_block::with_block_triple` does a real bounds check returning
   `DimensionMismatch` (`:143-147`) instead of relying on indexing panic —
   good defensive design on the live blocked path.
5. `ReMat::set_theta` validates length and returns `Result` without partially
   mutating on error (test `:505` proves the invariant); `lower_triangular_indices`
   / `linear_to_subscript` are column-major and match Julia `getθ!` ordering.
6. Error types are propagated as `LinAlgError`/`MixedModelError` (not
   stringified) consistent with the crate contract; tolerances in `OptSummary`
   match the documented Julia defaults.

## Verdict for this slice

REQUEST CHANGES (non-blocking on its own): resolve or contract-document **H1**
(Rust-only `quick_full_rank_identity_pivot` divergence on the live rank path)
and decide M1's boundary semantics before tagging the RC. Everything else is
LOW or capped MEDIUM in non-fit-path ports and can be tracked post-1.0, but
H1 touches `compiler::audit`/`lrt`/`FeTerm` and the inference contract.
