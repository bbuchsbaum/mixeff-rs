# Performance lead plan: levers 1-7 over MixedModels.jl and lme4

Status: APPROVED 2026-09-05 (Claude session). Copied from the plan-mode file as
`docs/plans/2026-09-05-performance-lead-plan.md` in the repo.

## Context

The native TrustBQ model-reuse patch (mote bd-01M1S3CXKWMC52HYW3Z3S16HN7,
uncommitted) closed most of the evaluation-count gap. The measurements taken
afterwards show the remaining wall time is not in the optimizer:

| Where the time goes (release NLopt build) | Evidence |
|---|---|
| Model construction: 90-100 ns/row (sleepstudy shapes), 340 ns/row (kb07); 85-95% of scalar fits, 40-59% of vector fits at n=10k | phase split run this session; trace: DataFrame deep clone in `Formula::materialize` (`src/formula/terms.rs:189-192`), eager design audit with a second dense X + second pivoted QR + three extra row passes (`src/compiler/audit.rs:935,1437,445-470,516`), redundant n×p copies (`src/types/fe_term.rs:55-90`, `src/types/fe_mat.rs:41-63`, `src/types/re_mat.rs:156-157`) |
| Post-fit finite-difference certificate: ≈2·d² hidden objective evaluations (d=9 → ~160 evals ≈ 25 ms of crossed_large's 96 ms) | `refresh_optimizer_certificate` → `finite_difference_optimizer_derivatives` (`src/model/linear/optimizer.rs:345-370, 507-620`) |
| Refits restart cold from `optsum.initial` and clone the template per replicate; ~40% of a bootstrap replicate is non-evaluation overhead | `LinearMixedModel::refit` (`src/model/linear/mod.rs:3414-3454`), `run_parametricbootstrap` (`src/model/linear/bootstrap.rs:920-936`), `bootstrap_refit_bench` output |
| Crossed per-evaluation path allocates 8×/49-162 KB per eval; scalar/vector kernels are allocation-free | `benchmarks/perf_baseline.json` pins |
| Evaluation counts: NLopt 39-69 (vector), 268-412 (crossed); Julia ~320-411 on crossed; both derivative-free | harness + Julia references |

Standing vs Julia today (construction + fit on both sides): vector 3.0-7.1×,
scalar 1.1-1.2× at n ≥ 5000, crossed 1.2-2.9×. vs lme4: 16-26× (asymptotic
report), GLMM 1.1-20×. No Julia GLMM comparison exists.

Goal: turn each lever into a measured, gated change; keep every fitted
objective inside the existing reference gates; no hidden model surgery
(`docs/mixed_model_compiler_inference_contract.md`); no blanket speed claims
without same-host paired evidence (`docs/difficult_model_release_contract.md`).

## Working rules for every task

- One mote issue per task (`mote new ... --tag perf`), `mote preflight`/`begin`
  on the exact files, `mote note` decisions, `mote done` with the numbers.
- Paired A/B protocol (promote the session scripts to
  `scripts/bench_paired.sh` + `scripts/bench_compare.py`): two binaries from
  the same commit, 5 rounds alternating order, medians; objective gate
  `|F_cand − F_base| ≤ 1e-6·(1+|F_ref|)`; wall wins counted per scenario.
- Gates on every merge: `cargo test` (default and `--no-default-features`),
  `cargo clippy --all-targets` both feature sets, `perf_gate` (objective,
  feval, alloc pins), cross-engine parity fixtures unchanged.
- Numbers go in the commit message and the mote note; nothing merges on a
  single unpaired run.
- Phases 1-4 touch disjoint files and can run as parallel mote-reserved
  slices; Phase 5 depends on nothing but should start after Phase 2 lands
  (it replaces Phase 2's finite differences). Phases 6-7 are independent.

## Phase 0: Measurement foundation (1 day)

T0.1 Phase-split columns in the harness. Extend
`examples/optimizer_bench_harness.rs` `FitRecord`/CSV with `build_ms`,
`fit_ms` (optimizer only), `postfit_ms` (time after the optimizer returns:
requires a `FitPhaseTimings` struct filled in `fit_with_options` around
`finalize_fit_result` and the `refresh_*` calls at
`src/model/linear/optimizer.rs:3484-3490`, exposed via `optsum` or a new
accessor). Keep `median_ms` as the all-in number so the Julia comparison
stays apples-to-apples (`scripts/bench_julia.jl:113-117` times build+fit).

T0.2 Regenerate Julia references on this host. Julia + MixedModels 5.3.0,
DataFrames, JSON3 are installed. Run `scripts/bench_julia.jl`, move the
hard-coded `Reference` constants out of the harness into
`benchmarks/julia_reference.json` with provenance (host, Julia and
MixedModels versions, date); harness loads it at start. Add GLMM rows to the
Julia script (see Phase 7).

T0.3 Baseline snapshot. Archive the harness CSV for the current `main`
(release and native) under `comparison/optimizer/<date>/` and re-pin
`benchmarks/perf_baseline.json` after the TrustBQ patch is committed.

Exit: harness prints phase columns; Julia references carry provenance;
baseline archived.

## Phase 1: Construction (3-5 days, lever 1)

Target: scalar_10000 construction 0.93 ms → ≤ 0.30 ms; kb07 build
0.61 ms → ≤ 0.25 ms; scalar rows ≥ 3× vs Julia; no objective change.

T1.1 Remove the DataFrame deep clone. `Formula::materialize`
(`src/formula/terms.rs:189`) returns `Cow<'_, DataFrame>` (borrow when
`derived.is_empty()`); `new_with_policies_internal`
(`src/model/linear/mod.rs:1108`) consumes the `Cow`. This alone removes n
`String` clones per categorical column (`CategoricalColumn.values`,
`src/model/data.rs:33-41`). Do not remove the public `values` field now
(SemVer); file a 2.0 note to deprecate it.

T1.2 Single-pass, allocation-light design audit. In `src/compiler/audit.rs`:
`grouping_audit`, `response_constant_within_group_diagnostic`,
`scope_note_diagnostics`, `single_grouping_counts` currently clone `refs`
and build `Vec<Vec<f64>>` per level. Replace with one pass over `refs`
accumulating per-level count / sum / sum-of-squares / min / max into
`Vec<f64>` of length q, borrowing `&[u32]`. Interaction groupings
(`composite_grouping_counts`, `empty_cells_for_interaction`) key on
`(u32,u32)`→`u64` instead of joined `String`s. Audit outputs must be
byte-identical (existing audit tests, compiler report fixtures).

T1.3 One QR, one X. `audit_fixed_effects` builds a second dense X and runs
`pivoted_qr_with_tol` on a clone (`src/linalg/pivot.rs:61`); `FeTerm::new`
runs `stats_rank` again (`src/types/fe_term.rs:55-90`). Compute the pivoted
QR once (in `FeTerm::new`) and hand its rank/pivot/column-norm result to the
audit (reorder: build the fixed design before `attach_design_audit`, or pass
a `FixedRankAssessment` into it). Remove the audit's private dense X.

T1.4 Lazy weighted copies and scratch. `FeMat.wtxy` (`fe_mat.rs:61`),
`ReMat.wtz` and `scratch` (`re_mat.rs:156-157`) become `Option`/lazily built
only when weights are set (`sqrtwts` non-empty) or the first weighted
operation runs; `select_columns` copy (`fixed_design.rs:582-590`) replaced
by a column-index view where the consumer only reads. This aligns with
`docs/lazy_fixed_design_materialization.md`.

T1.5 Cross-product kernels (added after profiling, see below). The
`[X|y]'Z`, `[X|y]'[X|y]`, and `Z'Z` builders in
`src/model/linear/blocks.rs` (`compute_re_cross_product`,
`compute_fixed_response_re_cross_product`,
`compute_fixed_response_cross_product`, and the `FixedDesign::xtx/xty/
xt_reterm` backends) index `DMatrix` element-wise with bounds checks in
row loops. Rewrite them over column slices (`as_slice()` / column
iterators with the level reference driving the accumulator index), and
build `A` once and clone into `L` only for blocks that the factorization
overwrites.

T1.6 Audit remainder: `scope_note_diagnostics` and `audit_random_term`
(re-widened refs, response-constant two passes, basis audits) after T1.2;
`build_re_mat` clones of `refs`/`levels` and the dense `z` + CSC adjoint +
`wtz` + scratch quadruple.

T1.7 Gate. `profile_kb07` build phase, the new harness `build_ms`, and
`streamed_rank_bench`; parity fixtures; `perf_gate`. Report ns/row before
and after for scalar_10000, vector_10000, kb07, crossed_large.

Construction profile after T1.1 + T1.2 (env-gated probes, release build,
median of 6 constructions, µs; scalar_10000 total 599, vector_10000 735,
crossed_large 1855):

| stage | scalar_10000 | vector_10000 | crossed_large |
|---|---:|---:|---:|
| compile IR + artifact | 4 | 5 | 21 |
| audit: fixed effects (2nd X + QR) | 55 | 63 | 94 |
| audit: random terms | 84 | 164 | 356 |
| audit: scope notes | 70 | 2 | 6 |
| fixed design build | 7 | 12 | 9 |
| FeTerm QR + pivot + select copies | 71 | 94 | 102 |
| ReMat builds | 73 | 124 | 364 |
| FeMat `[X|y]` + weighted clone | 14 | 36 | 50 |
| A blocks: weighted copies | 14 | 11 | 16 |
| A blocks: `Z'Z` | 38 | 63 | 288 |
| A blocks: `[X|y]'Z` | 92 | 99 | 261 |
| A blocks: `[X|y]'[X|y]` | 66 | 78 | 81 |
| snapshot + rest | 15 | 13 | 11 |

No single stage dominates; the cross-product kernels (T1.5, ~33%) and the
audit (T1.2 remainder + T1.3, ~35%) are the two largest groups.

Phase 1 status (paired, both profiles, objectives/theta/evals bit-identical
throughout):

| task | construction effect | notes |
|---|---|---|
| T1.1 borrow the frame | 1.10-1.44× | largest single win |
| T1.2 single-pass audit | within noise on single-factor shapes | exact; wins on interaction groupings |
| T1.5 slice kernels, Cow copies | 1.03-1.17× | nalgebra order pinned by tests |
| T1.3 one QR, identity pivot in place | 1.03-1.27× | audit pivot reused when matrices are bit-equal |

Phase 1 closed (T1.7 gate, release profile, medians of 5 paired rounds
vs the Phase 0 archive; objectives/theta/evals bit-identical throughout;
same-host Julia references):

| scenario | construction P0 → now | total P0 → now | vs Julia P0 → now |
|---|---:|---:|---:|
| scalar_5000 | 0.476 → 0.247 ms (1.93×) | 0.568 → 0.329 ms (1.72×) | 1.17× → 2.01× |
| scalar_10000 | 0.795 → 0.497 ms (1.60×) | 0.948 → 0.652 ms (1.46×) | 1.22× → 1.77× |
| scalar_deep_200x50 | 0.712 → 0.503 ms (1.41×) | 0.750 → 0.543 ms (1.38×) | 1.26× → 1.74× |
| vector_1000 | 0.098 → 0.063 ms (1.56×) | 0.573 → 0.421 ms (1.36×) | 3.44× → 4.68× |
| vector_10000 | 1.000 → 0.610 ms (1.64×) | 2.488 → 2.099 ms (1.19×) | 6.70× → 7.94× |
| vector_deep_200x50 | 0.847 → 0.540 ms (1.57×) | 1.661 → 1.171 ms (1.42×) | 2.58× → 3.65× |
| crossed_large | 2.145 → 1.425 ms (1.50×) | ~unchanged (construction is 2%) | 1.94× |
| kb07 build (profile_kb07) | 0.61 → 0.41 ms | | |

The 3× construction target was not reached: with the public-API
constraint above and the audit kept eager, the remaining ~0.5 ms at
n = 10 000 is spread over many O(n) passes (audit random-term and
scope-note statistics, ReMat build, the audit's own X and QR, the
remaining X copies) each already close to a single tight pass. Further
gains need the structural changes deferred to 2.0 (shared X ownership,
lazy weighted copies, dropping `adj_a`) or a lazy audit.

Constraint found for T1.4: `ReMat::{wtz, scratch, adj_a}` and
`FeMat::wtxy` are public fields of public types (`mixeff_rs::types`), so
making the weighted copies lazy is a 2.0 API change, not a 1.x
optimization. T1.4 is therefore folded into T1.6 as "build the same
fields more cheaply" (build `z` column-major directly instead of per-row
transposes, integer keys for interaction groupings, no repeated
`levels`/`refs` clones), and the lazy-copy design is deferred to
`docs/lazy_fixed_design_materialization.md`. The unused `adj_a` CSC
adjoint is built on every construction; dropping it is also a 2.0 item.

## Phase 2: Post-fit derivative certificate (1-2 days, lever 2)

Target: crossed_large wall −20% or better with unchanged certificate
semantics.

T2.1 Measure. Instrument `refresh_optimizer_certificate` once (feature- or
env-gated, not shipped) to record FD evaluation count and time per harness
row; store in the mote issue.

T2.2 Defer derivative evidence. `OptimizerCertificate` already has a
not-assessed state (`mark_derivative_checks_not_assessed`). Add a
`Deferred` state and compute the FD gradient/Hessian on first demand from
the consumers: `optimizer_certificate()` (`mod.rs:1436`), `fit_status`
(`src/compiler/report.rs`), `inference.rs:1037`, `pathology/certificate.rs:758`,
`verify_convergence` (`optimizer.rs:70-160`), and the GLMM metadata paths
that mutate the certificate (`src/model/generalized/metadata.rs:363-392`).
Implementation: keep `refresh_optimizer_certificate` eager for the cheap
stop-evidence part; move the FD block behind `ensure_derivative_evidence(&mut
self)` called by those accessors; `compiler_policy` gains
`eager_derivative_certificate: bool` (default false; CI parity lanes set
true) so behaviour is auditable. Bootstrap's `suppress_derivative_diagnostics`
stays.

T2.3 Gate. Certificate unit tests, report JSON round-trip, `fit_status`
fixtures unchanged when evidence is requested; paired harness shows the
crossed rows' `postfit_ms` drop; no change to `feval` (FD evals were never
counted).

Phase 2 shipped (no policy flag was needed). The fit tail records the
derivative checks as deferred; `inspection_artifact()` completes a cloned
artifact on the first `&self` inspection (every public accessor that
exposes the artifact or certificate, plus the Satterthwaite reliability
grade) and `ensure_derivative_evidence()` completes in place before
mutating paths (`verify_convergence`). A deferred-marker guard means a
certificate installed wholesale by the GLMM drivers is never completed
with LMM derivatives. Unit tests pin eager == inspected, refit reset, and
verification completion.

Paired vs 909e42f (5 alternating rounds, objectives/theta/evals
bit-identical, perf_gate OK):

| scenario | post-fit ms before → after | total speedup (release / native) |
|---|---:|---:|
| vector_1000 | 0.102 → 0.006 | 1.31× / 0.87× (native noise; evals equal) |
| vector_10000 | 0.495 → 0.009 | 1.29× / 1.11× |
| crossed_small | 2.50 → 0.024 | 1.60× / 1.52× |
| crossed_medium | 9.18 → 0.032 | 1.46× / 1.37× |
| crossed_large | 31.2 → 0.036 | 1.43× / 1.52× |

Same-host standing vs Julia (release): crossed_large 187.5 / 73.7 ms =
2.5× (was 1.9×), crossed_small 2.0× (was 1.3×), vector_10000 10.1×.

## Phase 3: Warm-started refits (2-3 days, lever 3)

Target: bootstrap replicate cost −50% on vector_10000 (2.0 ms → ≤ 1.0 ms)
with replicate θ/β within existing bootstrap fixture tolerances.

T3.1 Refit start policy. Add `RefitStart { Initial, Fitted }` to
`LinearMixedModel::refit` (new `refit_with_start`; `refit` keeps `Initial`
for Julia `refit!` parity). `Fitted` seeds `OptimizerControl.start_theta`
(`mod.rs:803`) from the template optimum and contracts the first step: NLopt
`initial_step`/`rhobeg` scaled to 1/8 of the family default, TrustBQ
`initial_radius = policy.initial_radius / 8` reusing the ladder warm-start
branch (`optimizer.rs` full-stage options). Bootstrap and profile refits use
`Fitted`.

T3.2 Replicate workspace. In `run_parametricbootstrap`: clone the template
once, refit the same `work` model each replicate (refit already resets
state); fuse `simulate` + y write + `recompute_a_blocks` into one pass
producing `Z'y`, `X'y`, `y'y` (only the y-dependent blocks of A change).
Same treatment for the cluster-resample path
(`src/model/linear/inference.rs:611`) is out of scope (it rebuilds the
design by necessity; benefits from Phase 1).

T3.3 Gate. `bootstrap_refit_bench` columns `clone_us`, `recompute_a_us`,
`refit_us`, `fit_feval`; bootstrap statistical fixtures; a paired run of
`parametricbootstrap` (1000 reps, sleepstudy) against Julia
`parametricbootstrap` on this host added to `bench_julia.jl`.

Phase 3 shipped: `RefitStart::{Initial, Fitted, From(theta)}` with
`refit_with_start` (`refit` keeps `Initial`); warm starts contract the
first optimizer step to `WARM_REFIT_INITIAL_STEP` (0.75/8); the
parametric bootstrap refits one reused working copy from the template
optimum (fresh clone only after a failed replicate); the bootstrap LRT and
fixed-effect null bootstrap warm-start from their templates; refits
refresh only the response-dependent A entries (`y'Z_j` rows, `X'y`,
`y'y`) through the same kernels as the full rebuild (bit-identical), with
the full rebuild kept as the fallback for non-dense fixed blocks.

`bootstrap_refit_bench` (200 replicates, release/NLopt), per replicate:

| scenario | before | after | evals before → after |
|---|---:|---:|---:|
| vector_1000 | 297 µs | 185 µs (1.60×) | 54 → 45 |
| scalar_1000 | 65 µs | 37 µs (1.73×) | 23 → 20 |
| vector_10000 | 1693 µs | 1164 µs (1.45×) | 51 → 46 |

The −50% target on vector_10000 was not reached: the non-evaluation
overhead is gone (refit ≈ evals × per-evaluation cost), but BOBYQA still
needs ~45 evaluations from a warm start. Fewer evaluations per replicate
now need Phase 5 (gradient-based steps) or a warm-start-aware native
TrustBQ route.

## Phase 4: Crossed per-evaluation workspace (2-3 days, lever 5)

Target: `allocs_per_eval` for crossed rows 8 → ≤ 1, `eval_median_us` −10 to
−20% on crossed rows, objective bit-identical (allocation-only change).

Confirmed allocation sites (all per evaluation):
- `determinant_term_and_pwrss_for_reml` (`src/model/linear/mod.rs:1763`) and
  its optimizer twin (`optimizer.rs:1693`) clone the trailing `(p+1)×(p+1)`
  block via `as_dense()` just to read its diagonal. This is the entire
  1 alloc / 72 bytes on scalar and vector rows.
- `rdiv_lower_transpose` (`blocks.rs:2349`) clones a whole Dense diagonal
  L block per off-diagonal solve; with three crossed terms that is three
  `nranef_j²` clones per evaluation, read-only. Dominant crossed waste.
- `as_dense()` promotions and reassignments in `rank_k_downdate`
  (`blocks.rs:1894, 1975`), `subtract_product` (`:2002-2012`),
  `rdiv_lower_transpose` other arms (`:2224, 2295, 2402`),
  `with_dense_block` temporaries in `subtract_product_from_blocks`
  (`:154-160`), `update_l_from_parts` off-block (`:265`),
  `copy_and_rmul_lambda` scalar arm (`:1673`).
- The `DMatrix::zeros` calls inside the θ loop (`blocks.rs:1307-1705`) are
  first-evaluation only and already reuse buffers afterwards.

T4.1 Read diagonals without cloning: replace both `as_dense()` reads with a
`MatrixBlock::diagonal_iter()` / `dense_entry(i, j)` accessor.

T4.2 Make `rdiv_lower_transpose` read the Dense block through a borrow
(`with_block_pair_mut` at `src/types/matrix_block.rs:118` already gives
disjoint mutable/immutable access) instead of cloning.

T4.3 Promote-once policy: blocks that get promoted to Dense during the first
`update_l` stay Dense (`promote_crossed_fill_in_blocks`, `blocks.rs:317`,
already does this for fill-in); extend it so `rank_k_downdate` /
`subtract_product` targets never re-promote per evaluation, and route the
remaining `with_dense_block` temporaries through a per-model
`UpdateWorkspace` (scratch `DMatrix`s sized on first use, owned by
`LinearMixedModel` and `LmmWorkspace`). `gemm_sub_abt` /
`gemm_sub_rows_aat` (`blocks.rs:1779-1855`, both the nalgebra and faer
variants) are already allocation-free; give them the workspace only if T4.3
needs a scratch operand.

T4.4 Gate. `perf_gate` alloc pins re-pinned to 0 for every row;
`objective_eval_bench` paired; objectives bit-identical (assert equality in
the paired script). faer stays opt-in (rounding-level drift).

Phase 4 shipped (exact): the trailing-block diagonal reads in the
objective and the certificate twin borrow the dense block instead of
cloning it; the dense-L arm of `rdiv_lower_transpose` borrows L; and the
three copy kernels (`copy_block`, `copy_and_scale_offdiag`,
`copy_and_rmul_lambda`) fill an L block the factorization previously
promoted to dense in place (`fill_dense_from_block`) instead of reverting
it to the sparse/diagonal variant of A, which the next factorization only
promoted again. `perf_gate` allocation pins after the change: scalar and
vector rows 1 → 0 allocations per evaluation; crossed rows 8 → 4 (49 KB →
39 KB, 162 KB → 123 KB). An allocation-size trace shows the four residual
crossed allocations are matrixmultiply's panel-packing buffers inside
nalgebra's gemm (sizes equal the L-block products), not crate code; at
~100 ns each they are left alone. Objectives and evaluation counts are
bit-identical (perf_gate objective/feval pins). Paired runs (5 alternating
rounds, host load average ≈ 20 from other sessions) show no measurable
wall-time change: LMM rows median 1.02×, GLMM rows 0.98×, all within
noise. Conclusion: the per-evaluation path was already arithmetic-bound;
the remaining per-evaluation levers are algorithmic (Phase 5) or
kernel-level (fused weighted A rebuild, cheaper PIRLS row kernels), not
allocation hygiene. The perf baseline's allocation pins were re-pinned by
hand to the new counts; its timing baselines were deliberately left as
recorded on a quiet host.

## Phase 5: Analytic gradient and a gradient-based optimizer (3-4 weeks, staged, lever 4)

Facts from the objective path. Blocks are the packed lower triangle of
`[Z_1..Z_k, (X|y)]` cross products (`src/types/matrix_block.rs:12-22`,
`block_index` at `:113`; variants Dense / Sparse / Diagonal /
BlockDiagonal). `update_l_from_parts` (`src/model/linear/blocks.rs:213`)
forms `L` block by block: `copy_scale_inflate` (`L_jj = Λⱼᵀ A_jj Λⱼ + I`,
`:1289`), `copy_and_scale_offdiag` (`:1506`), `copy_and_rmul_lambda`
(`:1620`), `cholesky_block_with_tolerance` (`:2041`),
`rdiv_lower_transpose` (`:2181`), `rank_k_downdate` (`:1856`),
`subtract_product` (`:1995`). The objective is
`logdet + denomdf·(1 + ln(2π·pwrss/denomdf))` with
`logdet = Σ logdet_block(L_jj)` over RE blocks (+ `2Σ ln L_xx[i,i]` for REML)
and `pwrss = L[pp1-1, pp1-1]²` (`mod.rs:1757-1795`). Λ enters only through
`ReMat::set_theta` writing `theta` into `lambda` at `inds`
(`src/types/re_mat.rs:54, 206`), so `∂Λ/∂θ_k` is a single unit entry in one
term's `vsize×vsize` slot. Nothing analytic exists today: every derivative
is finite difference (`optimizer.rs:507`, `mod.rs:2113`, `mod.rs:2783`),
and the ported primitives in `src/linalg/{rank_update,chol_unblocked,
block_ops,logdet}.rs` are unused by the model path (mote
bd-01KRNPZAT86D69A9G7BJVFNQEA).

Approach: forward-mode differentiation of the blocked Cholesky, not the
trace formula. For each direction `k`, `dA_k = EₖᵀAΛ + ΛᵀAEₖ` touches only
the blocks in that term's row/column, and the Cholesky derivative
recursion `dL = L·Φ(L⁻¹ dM L⁻ᵀ)` (Φ = lower triangle with halved diagonal)
maps one-to-one onto the existing block operations, so the tangent pass is
the same sequence of `copy_scale_inflate` / `rdiv_lower_transpose` /
`rank_k_downdate` calls with tangent operands. Then
`∂logdet/∂θ_k = 2Σ dL_ii/L_ii` and `∂pwrss/∂θ_k = 2·L_yy·dL_yy`. All `d`
directions share the same triangular solves, so they run as one multi-RHS
tangent pass; cost target ≤ 2 evaluation-equivalents for the whole
gradient at d ≤ 10.

S5.1 Math note + dense reference (2-3 days). Write
`docs/profiled_deviance_gradient.md` with the formulas above (ML, REML,
profiled and fixed σ, weights via `weight_logdet_correction`). Implement a
dense reference (`as_dense()` of the whole system, nalgebra Cholesky, dense
tangent) behind `unstable-internals`, and validate against
`finite_difference_optimizer_derivatives` to 1e-6 relative on sleepstudy,
kb07, crossed_small, and a θ=0 boundary case. Go/no-go: agreement on all
four.

S5.2 Blocked tangent pass (1-2 weeks). `tangent_update_l_from_parts(a,
l, reterms, directions) -> [dL blocks]` in `blocks.rs` reusing the block
ops with tangent variants; `profiled_objective_gradient_from_parts` in
`optimizer.rs` next to `profiled_objective_from_parts` (`:1645`). Fast-path
twins for `profiled_objective_one_vsize1_fast` / `_vsize2_fast`
(`optimizer.rs:1805, 1848`) since scalar and single-vector models dominate
usage. Go/no-go: gradient cost ≤ 2 evaluation-equivalents (measured with
`objective_eval_bench` style timing) and agreement with S5.1 to 1e-10.

S5.1 shipped: `docs/profiled_deviance_gradient.md` derives the gradient
from the envelope theorem (pwrss term `−2 r̂ᵀ Z_w E_k û`) and trace
formulas (`2 tr(M⁻¹ Λᵀ A_ZZ E_k)` for logdet M; `C⁻¹` with `Q = M⁻¹R`,
`V = A_ZZ Λ Q` for the REML logdet C) rather than the tangent Cholesky
pass sketched above: the trace form needs one selected inverse shared by
every direction, so its cost is d-independent, whereas the tangent pass
scales with d. The dense reference (`dense_reference_gradient`) matches
the certificate's central differences to 1e-5 (one-sided 1e-3 at θ = 0)
on vector, scalar and weighted sleepstudy shapes and on crossed vector
terms, ML and REML.

S5.2 shipped: the production gradient (`profiled_gradient_at_current_theta`,
`objective_and_gradient_at` in `src/model/linear/gradient.rs`) reads only
the A and L blocks: `e_j = Z_jᵀ(y − Xβ)` from the `[X|y]ᵀZ_j` block, `û`
and the REML `Q` from one blocked solve with p + 1 right-hand sides,
`g_j = e_j − Σ_i S_ji Λ_i û_i` from the A blocks, and the REML term folded
to `2((V − T) C⁻¹ Qᵀ)[a, b]`. Single-term models run allocation-free
per-level kernels (scalar, unrolled s = 2, generic s ≥ 3); several terms
use the block Takahashi recursion with only the level-diagonal blocks of
the leading term's `Z_00` formed, gemm-backed solves, and one `q_j³`
product per dense diagonal block. Agreement with the dense reference:
1e-9 relative on single-term (s = 1, 2, 3), weighted, crossed and nested
fits, including θ = 0 slots. Cost (release, `gradient::cost` probes, host
load average ≈ 40): scalar_10000 ML 0.89 / REML 1.24, vector_10000 ML
0.93 / REML 1.47, crossed 60 × 40 × 12 (d = 7) REML 1.93
evaluation-equivalents of a full `update_l` evaluation, so the ≤ 2
go/no-go holds. Caveat carried into S5.3: the optimizer's single-term
fast objective path is 2–2.5× cheaper than `update_l`, so an
objective-plus-gradient pair costs 2.7–4.9× one fast evaluation on
single-term rows and ≈ 2.9× on the crossed row. The gradient optimizer
must therefore cut evaluations by more than that factor to win wall
time, which the crossed rows (250–400 evaluations today) can deliver and
the d ≤ 3 single-term rows (30–70) may not. S5.3 gates the oracle by
family (multi-term or d ≥ 4 first) and measures single-term rows
separately; a fused fast-path objective-plus-gradient kernel is the
fallback if single-term rows need it. The `mod gradient` `dead_code`
allowance stays until S5.3 wires the oracle into the optimizer.

S5.3 Gradient-aware TrustBQ (1 week). Add a `GradientOracle` option to
`src/optimizer/trust_bq.rs`: model gradient from the oracle (no axis
stencil), Hessian from symmetric-rank-one/BFGS secant updates seeded by one
FD-of-gradient pass (2d evals), same trust-region acceptance, bounds via the
existing projection. Family policy selects it when the oracle is available.
Go/no-go: vector rows ≤ 25 evaluations (NLopt 39-69), crossed_large ≤ 200
(408), all objective gates pass, boundary/singular cases (θ=0 rows, weak
identification fixtures) converge with acceptable stops.

S5.4 Replace FD in the certificate (2-3 days). `finite_difference_optimizer_derivatives`
uses the analytic gradient and an FD-of-gradient Hessian (2d evals instead
of 2d²); Satterthwaite/KR varpar Jacobian (`mod.rs:2117-2149`) and Hessian
(`mod.rs:2786-2830`) switch to the same oracle. Then Phase 2's deferral can
stay or be reverted to eager, whichever the numbers justify.

## Phase 6: Response-matrix batch (2 weeks, lever 6)

Facts: the shared-θ machinery in `docs/multivariate_shared_theta.md`
(steps 1-4) already exists: `LmmObjectiveKernel` (`src/model/kernel.rs:28-42`)
caches the θ-invariant `[Z X]` structure (`create_structural_al`,
`blocks.rs:600`), `LmmWorkspace` (`kernel.rs:157`) holds per-worker `L`,
and `profile_response_matrix_with_l_blocks` (`blocks.rs:1170`) does the
multi-column forward solve sharing `logdet` across columns. Modes
`ProfileAtTheta / OptimizeSharedTheta / OptimizePerColumn /
OptimizeGrouped / OptimizeAdaptive` and `BatchWarmStart::{TemplateTheta,
SharedTheta, Fixed, Provided}` are public (`src/model/batch.rs:102-166`).
Rayon fan-out exists at chunk level (`kernel.rs:245-259`, thresholds 4
chunks / 16 384 work) and column level (`batch.rs:664-707`). Known gaps:
`build_response_rhs_blocks` (`blocks.rs:928`) allocates one `DMatrix` per
RE term plus `x.tr_mul(y)` on every profile call; no neighbour-chained warm
start; `docs/response_matrix_batch_lmm.md:44` ("all modes are serial") is
stale and `OptimizeAdaptive` is undocumented; GLMM batch is deferred by
design (`:63-66`).

T6.1 Per-profile workspace: hoist the RHS blocks and the `XᵀY` buffer into
`LmmWorkspace` (sized for the chunk width once), so a θ evaluation over a
chunk allocates nothing. Gate with the same alloc counter as `perf_gate`
on `bench_response_matrix_batch`.

T6.2 Chained warm starts for `OptimizePerColumn` / `OptimizeAdaptive`: new
`BatchWarmStart::Chained { order: Option<Vec<usize>> }` seeding column
`j` from the previous fitted column (or a caller-supplied neighbour order
for spatial data), with the contracted first step from Phase 3. Measure
evaluations per column before/after on the batch bench.

T6.3 Parallel defaults: paired run of chunk-level vs column-level fan-out
and of the two thresholds; keep bit-identical results; decide whether
`BatchParallelism::Rayon` becomes the default when the feature is compiled.

T6.4 Docs: bring `docs/response_matrix_batch_lmm.md` in line (parallelism,
`OptimizeAdaptive`, `Chained`), and mark `multivariate_shared_theta.md`
steps 1-4 as shipped. Multivariate covariance (step 6) and GLMM batch stay
out of scope.

T6.5 Gate: `bench_response_matrix_batch` vs the lme4 loop
(`scripts/bench_response_matrix_lme4.R`) and vs one Julia fit × columns from
the regenerated references; batch fixtures unchanged.

## Phase 7: GLMM (1-2 weeks, lever 7)

Facts: `scripts/bench_julia.jl` has no GLMM scenario; the only Julia GLMM
calls are in the parity scripts (`scripts/parity_dump_julia.jl:153`,
`regenerate_julia_parity_fixtures.jl`). lme4 timings come from
`scripts/compare_lme4.R` (`glmerControl(calc.derivs=FALSE, tolPwrss=1e-9)`,
nAGQ 7 for AGQ rows) driven by `comparison/manifest.json`, recorded in
`comparison/{rust,lme4}_results.json` and gated by
`tests/glmm_speed_parity.rs` (`minimum_speedup: 1.0` on cbpp, grouseticks,
verbagg, culcitalogreg). Recorded lme4/Rust ms: cbpp 33/0.75, grouseticks
333/248, verbagg 5537/395, arabidopsis 111/68, contraception 186/25.

The GLMM post-fit tail is heavier than the LMM one and runs unconditionally
(`src/model/generalized/optimizer.rs:350-372, 478-491, 626-648` →
`metadata.rs:304 record_glmm_fit_metadata`): `finalize_theta_after_optimizer`
re-runs a full PIRLS plus a deviance; `certify_pirls_profiled_optimum`
(`mod.rs:1298`) computes a finite-difference certification gradient with
step escalation (`mod.rs:1133`) and a finite-difference θ-Hessian over
interior indices (`mod.rs:1183`), each probe re-running PIRLS. On
grouseticks (3 RE terms, d=3, 1.3× vs lme4) this tail is the prime suspect.

T7.1 Evidence first. Add GLMM blocks to `scripts/bench_julia.jl` using
`MixedModels.dataset(:cbpp / :verbagg / :grouseticks / :contra)` (same data
as `datasets/*`, which were dumped from Julia by
`scripts/dump_julia_datasets.jl`) with `fit(MixedModel, f, df, Binomial() |
Poisson() | Bernoulli(); fast=true)` and `fast=false` variants; write
results into `benchmarks/julia_reference.json`. Add a `ScenarioKind::Glmm`
branch to the harness that loads from `datasets/` via
`src/datasets/mod.rs` and fits `GeneralizedLinearMixedModel`, printing the
same columns. Regenerate the lme4 rows on this host through
`compare_rust` + `compare_lme4.R`.

T7.1 result (same host; Julia MixedModels 5.3.0; harness GLMM rows via
`--features unstable-internals`; objectives of the profiled rows match
Julia `fast=true` to 1e-6):

| row | Rust ms (evals) | Julia fast ms (evals) | Julia full ms (evals) | Rust vs Julia fast |
|---|---:|---:|---:|---:|
| cbpp (Binomial, 1 RE) | 0.83 (19) | 1.69 (22) | 3.08 (63) | 2.0× |
| grouseticks (Poisson, 3 RE) | 417 (48) | 850 (69) | 37 865 (3253) | 2.0× |
| verbagg (Bernoulli, 2 crossed RE, n=7584) | 565 (41) | 77 (33) | 2204 (1028) | **0.14×** |
| contra intercept (Bernoulli, 1 RE, n=1934) | 35.7 (33) | 10.0 (20) | 97 (254) | **0.28×** |
| contra slope (Bernoulli, 1 vector RE) | 73.7 (52) | 41.1 (55) | 258 (382) | **0.56×** |
| arabidopsis (Poisson, 3 RE) | 81.8 (122) | not in MixedModels | | (lme4 111 ms) |
| cbpp joint Laplace (`fast: false`) | 5.68 (98) | | 3.08 (63) | 0.54×; the joint objective is on a different scale (184.05 vs 100.10), so only timing is comparable |

Reading: the evaluation counts are similar to Julia's, so the gap is
per-evaluation PIRLS cost (verbagg ≈ 14 ms per θ evaluation vs Julia ≈ 2.3
ms; contra intercept ≈ 1.1 ms vs 0.5 ms), not the optimizer. That makes
T7.4 (PIRLS per-iteration cost) the GLMM lever, with T7.2 (the
certification tail) second.

T7.2 Measure and defer the GLMM certification tail. Instrument PIRLS
re-runs inside `record_glmm_fit_metadata`; then apply Phase 2's `Deferred`
state to `certify_pirls_profiled_optimum` and the joint-path
`joint_laplace_certification_gradient` (`joint.rs:686`), keeping the
`apply_pirls_profiled_optimum_evidence` / `mark_derivative_checks_not_assessed`
state machine (`metadata.rs:363-392`) and the separation / near-unit
correlation diagnostics eager (they are cheap). Also drop the redundant
post-optimizer PIRLS in `finalize_theta_after_optimizer` when the last
optimizer evaluation was at the accepted θ (cache the PIRLS state keyed on
θ bits, as TrustBQ's sample cache already does).

T7.3 GLMM refit warm start. `GeneralizedLinearMixedModel::refit`
(`generalized/optimizer.rs:20`) restarts from `optsum.initial`; add the same
`RefitStart::Fitted` policy as Phase 3, seeding θ and β (PIRLS start) from
the template. Bootstrap for GLMMs uses it.

T7.4 Construction and PIRLS workspace. Phase 1 applies unchanged (the GLMM
wraps a `LinearMixedModel`). Profile `pirls.rs` per-iteration allocations
with the same alloc-count instrumentation as `perf_gate` and hoist them.

T7.4 first slice shipped (pulled ahead of T7.2 because the probe put the
A-block rebuild, not the certification tail, first): the weighted
`recompute_a_blocks` builds `[X|y]'Z` and `[X|y]'[X|y]` from the existing
weighted `wtxy` (no weighted `FixedDesign` copy per iteration), and the
scalar×scalar `Z'Z` cross blocks keep a cached structural CSC pattern and
refresh values in place instead of rebuilding through a keyed map. Both
are bit-identical. Paired vs 08f7733 (GLMM rows, 5 alternating rounds,
35/35 wins): cbpp 1.16×, grouseticks 1.33×, verbagg 1.67× (560 → 336 ms),
contra 1.20×, arabidopsis 1.25×. Per-iteration split after this slice:
verbagg 825 µs = A rebuild 46% / weights 15% / η+objective 13% / Cholesky
14% / solve 9%; contra 149 µs = A rebuild 50%; grouseticks 629 µs =
Cholesky 78% (Phase 4). Remaining GLMM levers, in order: Phase 4's
Cholesky workspace (grouseticks, verbagg), a fused single-pass weighted
A rebuild (verbagg, contra), cheaper weight/η/objective row kernels, and
PIRLS iteration count (6–8 per θ vs Julia's ~5).

T7.5 Gate. `tests/glmm_speed_parity.rs` rows re-pinned upward on the new
same-host numbers (not lowered); GLMM parity fixtures unchanged (the lme4
references need `tolPwrss=1e-9`, see memory note); new Julia GLMM rows
reported alongside lme4 in `comparison/REPORT.md`.

## Verification (end-to-end)

1. `cargo test` and `cargo test --no-default-features`; `cargo clippy
   --all-targets` for both.
2. `cargo run --release --features unstable-internals --example perf_gate`
   against the re-pinned baseline.
3. Paired harness (release and native profiles) vs the archived Phase 0
   baseline; every row must pass the objective gate; report evals, build_ms,
   fit_ms, postfit_ms, and speedup vs the regenerated Julia references.
4. `bootstrap_refit_bench` and `profile_kb07` before/after per phase.
5. Cross-engine parity fixtures (`tests/parity_*`) unchanged.

## Sequencing summary (decisions taken: analytic gradient is the first strategic phase; Phases 1-4 run sequentially, one task at a time)

| Order | Phase | Effort | Expected effect |
|---|---|---|---|
| 0 | Measurement foundation | 1 day | trustworthy same-host references, phase columns |
| 1 | Construction | 3-5 days | scalar rows 1.1× → ~3-4× vs Julia; +30-50% vector; all large-n fits |
| 2 | Deferred FD certificate | 1-2 days | crossed rows −20-25% wall |
| 2b | GLMM T7.1-T7.2 (same pattern as Phase 2) | 2-3 days | first Julia GLMM comparison; grouseticks/arabidopsis tail removed |
| 3 | Warm refits | 2-3 days | bootstrap/profile 2× per replicate |
| 4 | Crossed workspace | 2-3 days | zero allocs per eval; crossed per-eval −10-20% |
| 5 | Analytic gradient + optimizer | 3-4 weeks staged | evals ÷2-3, exact certificates, faster KR/Satterthwaite |
| 7 | GLMM T7.3-T7.5 | 3-5 days | GLMM warm refits, PIRLS workspace, gates re-pinned |
| 6 | Batch workspace + chaining + parallel defaults | 2 weeks | no competitor on the neuroimaging workload |

Each task: mote issue → reserve files → implement → paired benchmark and
full gates → commit with the numbers → `mote done`. The first execution
step copies this plan to `docs/plans/2026-09-05-performance-lead-plan.md`
and commits the pending TrustBQ model-reuse patch so Phase 0 baselines
include it.
