# May 9 Stash Salvage Audit

This note records the disposition of the preserved May 9 pre-release stash.
The stash has been saved as branch `salvage/stash-2026-05-09-original`
without `vendor/lmerTestR/`. That branch is a reference archive, not a
merge target: it was based on `e1e1dec` and overlaps heavily with later
release, TrustBQ, GLMM, and inference work on `main`.

## Reconstituted Now

- Gamma GLMM parametric bootstrap. Current `main` still refused this path,
  while the stash contained a working family-specific draw rule. The restored
  implementation uses the current `GeneralizedLinearMixedModel::simulate_response`
  random-effects simulation path and adds Gamma draws with
  `shape = 1 / phi`, `scale = mu * phi`, where `phi = dispersion(true)`.
- GLMM comparison artifacts and gates. The stash's comparison work was ported
  in a current-main shape: `examples/compare_rust.rs` now emits supported GLMM
  fits, objective/response-constant conventions, optimizer metadata, and fevals;
  `examples/compare_report.rs` classifies GLMM objective non-comparability and
  known fast-PIRLS numeric gaps; `tests/glmm_comparison_gates.rs` and
  `tests/glmm_speed_parity.rs` keep the generated artifacts executable.
- MixedModels.jl fast-oracle fixture for the large current fast-PIRLS rows.
  `tests/fixtures/parity/glmm_fast_oracles.json` explains the current
  contraception, grouseticks, and verbagg divergences from lme4. It is a
  drift guard for the current implementation mode, not a claim that fast-PIRLS
  is the final GLMM target.
- Cluster-resample full-model contrast payloads. The useful part of the
  stash-era bootstrap work was restored as a current-main estimator
  distribution target: cluster draws resample committed `DataFrame` rows by
  grouping factor, relabel duplicated sampled clusters, refit the full model,
  and return replicate statistics plus percentile intervals. Model-comparison
  bootstrap LRT was not duplicated because current `main` already exposes
  `stats::parametric_bootstrap_lrt`.

## Superseded On Main

- Satterthwaite and Kenward-Roger scaffolding. Current `main` already has
  release-ready contract docs, inference-table support, parity fixtures, and
  public result contracts that supersede the older stash shape.
- Stash-era non-fast GLMM comparison path. Current `main` explicitly rejects
  `fit_with_options(fast = false)` for GLMMs, so the old harness expectation
  that small binomial rows use a joint beta/theta path is stale. The current
  comparison artifacts classify cbpp and culcitalogreg as fast-PIRLS rows with
  lme4 beta gaps instead of direct lme4 parity gates.

## Candidate Later Slices

- Objective/kernel experiments in `src/model/linear.rs`, especially the
  dense/sparse cross-product and `faer`/`dyn-stack` trials. These should be
  evaluated against the isolated per-evaluation benchmark before porting.
- GLMM grouseticks speed gap. The regenerated speed artifact shows Rust
  `grouseticks` at about 0.90x lme4 on this run, so the speed gate keeps it
  as an explicit known-slow row tracked by `bd-01KRSQYRHF8VK627HZ6Z23CP93`.

## GLMM `fast=false` Joint Optimizer Inspection

The stash's `fast=false` GLMM path is not a small compatibility patch. It adds
a `GlmmFitMode` switch, changes the outer optimizer vector from θ to `[β; θ]`,
introduces a fixed beta box bound, threads the mode through NLopt, COBYLA, and
PatternSearch, and adds joint-parameter finalization/certification paths. It
also broadens several GLMM internals from crate-private to public and changes
initial beta handling.

That is potentially useful research material for a future joint GLMM optimizer,
but it conflicts with the current 1.0 contract in `docs/glmm_support_contract.md`:
`fast = true` is the supported mode and `fast = false` must return an explicit
unsupported error. The stash has only a small smoke test for the joint path and
does not establish parity for the current GLMM comparison rows. Preserve it as
reference material only; any revival should be a new feature epic with explicit
numeric fixtures for cbpp/culcita-style rows, no public-internal visibility
expansion, and no change to the stable unsupported `fast=false` contract until
the joint path is fully certified.

## Faer / Objective-Kernel Inspection

The stash's `faer` work is not ready to reconstitute as a direct patch.
It adds `faer` and `dyn-stack` dependencies, changes profile settings, and
rewrites substantial parts of `src/model/linear.rs` around sparse/dense
cross-products and Cholesky factorization. The potentially valuable ideas are
specific kernels such as sparse-sparse transpose subtraction, sparse-dense
transpose subtraction, dense-sparse transpose subtraction, and a `faer`
Cholesky trial.

Those ideas are also deeply entangled with stale solver code from the
`e1e1dec` line and with old public/internal boundaries that current `main`
has since replaced. Porting them wholesale would risk undoing TrustBQ,
release-boundary, and GLMM changes. The right next step is a measured kernel
experiment against the current isolated per-evaluation benchmark and
`examples/profile_pls_kernel.rs`, not a merge from the salvage branch. Accept
only a narrow kernel change that shows a repeatable per-evaluation win and
keeps the current public API, optimizer contracts, and comparison artifacts
unchanged.

Specific candidate routing:

- Sparse-sparse / sparse-dense / dense-sparse transpose subtraction belongs
  behind a current `subtract_product_from_blocks` microbenchmark first. Use
  `examples/objective_eval_bench.rs` crossed rows to confirm an end-to-end
  per-evaluation win before adding any special-case kernel.
- The stash's broader crossed-block sparsity policy should be tested on
  crossed scalar/vector rows because it can change memory shape and fill-in.
  Do not port it as part of a generic cleanup.
- The `faer` dense Cholesky trial should be evaluated only for large dense
  diagonal blocks, with `examples/profile_pls_kernel.rs` isolating
  Cholesky/downdate stages and `objective_eval_bench` confirming the public
  objective path. A new dependency is acceptable only after a repeatable win
  survives both checks.
- Stash profile/Cargo changes are rejected. The current unstable-internals
  profiling examples stay feature-gated and the release profile remains under
  current package policy.

## Do Not Restore

- `vendor/lmerTestR/`. It was only a reference copy and should remain outside
  the Rust repository.
