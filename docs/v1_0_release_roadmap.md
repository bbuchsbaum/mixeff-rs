# mixeff-rs v1.0 Release Roadmap

**Status:** living planning document, last revised 2026-05-15. Internal — not
part of the published crate (the `docs/` directory is excluded from the
package) and not a stability contract. Phase A (publishability) has landed:
crates.io metadata, `README.md`/`LICENSE`, trimmed public API,
`#[non_exhaustive]` enums, builder/prelude, and runnable quickstart are in
place. Later phases track the remaining hardening work.
**Method:** synthesis of five parallel sub-agent audits covering (1) core numerics, (2) formula/data, (3) stats/inference, (4) tests/benchmarks/CI, (5) public API/docs/ergonomics.

---

## Executive summary

The crate's numerical core is structurally faithful to MixedModels.jl and the inference layer is unusually disciplined for a pre-1.0 Rust crate (typed refusals, stable reason codes, versioned JSON schemas). The Phase A publishability blockers below have since been closed; several silent-acceptance and stability hazards in later phases must still close before tagging 1.0. (Original audit framing — "not publishable to crates.io today" — retained below for historical context.)

The risk surface concentrates in five themes:

| Theme | Severity | Where |
|---|---|---|
| Public API too wide / no SemVer guardrails | **high** | `lib.rs`, every `pub` enum |
| Crates.io publishing metadata missing | **high** | `Cargo.toml`, repo root |
| Formula parser silent-accepts malformed input | **high** | `src/formula/parser.rs` |
| Numerical foot-guns (unsafe ptrs, hard-coded clamps, MGS pivot) | **high** | `src/linalg/pivot.rs`, `src/model/{linear,generalized}.rs` |
| CI is minimum-viable; parity gate manual | **medium** | `.github/workflows/ci.yml`, `scripts/check_julia_parity_fixtures.sh` |

The inference layer needs only connective tissue (Wald CIs, GLMM PB for binomial/Poisson, harness completion). The compiler/IR work that landed on `main` is sound but currently `pub mod`-exposed, which is premature.

---

## Current state by area

### 1. Core numerics — structurally faithful, three concentrated risks

**Solid:** `chol_unblocked`, `rank_update_*`, `logdet_*`, `update_l_from_parts`, profiled objective math, Gauss-Hermite cache. All small, well-tested, faithful ports.

**Fragile:**
- **Pivoted QR uses Modified Gram-Schmidt** without reorthogonalization (`src/linalg/pivot.rs:36-112`); Julia uses LAPACK Householder. On near-rank-deficient designs the kept columns differ — parity tests at `:339` already acknowledge this gap.
- **Hard-coded `1e-30` zero-clamps** in `solve_scaled_vsize2_row` and `rdiv_lower_transpose` (`src/model/linear.rs:426-435, 9957-10046`) diverge from the policy-controlled `cholesky_zero_pad_abs_tolerance` everywhere else. Silent behavior change at sub-microvariance σ scales.
- **Raw-pointer `unsafe` aliasing** in the GLMM optimizer closure (`src/model/generalized.rs:1018-1031, 1094-1114`). Sound today; foot-gun forever.
- **User-reachable `.expect(...)`** in `recompute_a_blocks` (`linear.rs:1848, 1873, 1881`) — reachable from PIRLS via `update_irls_weights`.
- **PIRLS step-halving acceptance bound** drifts from Julia's once-only `1.0001` slack (`generalized.rs:478, 554`).
- **`subtract_product`** silently promotes block-diagonal/sparse targets to dense without diagnostic (`linear.rs:9765`) — can blow up large fits to many GiB.
- **Constant-response detection** is `f64::EPSILON`-strict (`linear.rs:4199`) — too tight for FP-noisy inputs.

**Julia parity gaps:** MGS vs LAPACK QR (kept-column drift), Rust's relative-tolerance vs Julia's strict-positive Cholesky padding, PIRLS acceptance-bound semantics, optimizer fallback ladder has no Julia analogue.

### 2. Formula parser & data layer — coverage mostly there, silent-accept bugs

**Implemented:** `*`, `:`, `/`, `(re | g)`, `(re || g)`, `(re | g1 & g2)`, intercept rules — verified in `parser.rs`.

**Missing vs lme4:**
- No function-call atoms (`I(x^2)`, `poly(x,2)`, `scale(x)`, `log(x)`, `offset(z)`) — tokenizer rejects `^` and `,` outright.
- No backtick identifiers.
- No formula-level contrast specification.
- `FixedTerm::Nested` variant is dead code; `fixed_design.rs:853` silently drops it.

**Silent-acceptance bugs (real):**
- Trailing operators: `y ~ x + (1|g) +` parses cleanly.
- `y ~ -x1` parses as `1 + x1` (drop instead of error).
- `y ~ x1 - x2` parses as `1 + x1 + x2` (subtraction silently treated as addition — **lme4 uses `-` for term removal**).
- `y ~ (1|g) (1|h)` without `+` parses cleanly.
- `2*x1` becomes column `"2"`, error deferred to design-build time.

**Data layer:**
- `ContrastSource::{Sum, Helmert, Polynomial}` are decorative labels; `CategoricalContrast` is a struct (`src/model/data.rs:42`) and only treatment coding is actually built — there are no constructors for the non-treatment bases.
- Reference level is first-appearance order (lme4 uses alphabetical) — silent parity hazard.
- NaN/Inf not rejected in `add_numeric` (`data.rs:266`); propagates into Cholesky.
- Train/predict factor consistency: predict-time rebuild does not carry training `levels` forward (`linear.rs:6772-6905`).

**Multivariate readiness:** `FeMat` bakes `[X | y]` at width `rank+1`, hard-coded everywhere downstream. `docs/multivariate_shared_theta.md` cannot be delivered without API-breaking surgery on `FeMat` and `Formula.response`.

### 3. Stats / inference — most disciplined area; connective tissue needed

**Production:** `CoefTable` (with `PValuePolicy::Unavailable{reason}` channel), `VarCorr` (residual-source aware), `ModelSummary`, `LikelihoodRatioTest` + `ModelComparisonTable` (typed taxonomy, stable reason codes, MLrefit policy), `BoundaryLikelihoodRatioTest` (50:50 mixture, matches contract), `profile_likelihood` (σ, scalar/vector θ, ML β; REML β explicitly refused), `parametricbootstrap` (LMM only, schema `mixedmodels.bootstrap_run` 1.0.0).

**Contract conformance:** boundary-LRT and profile-likelihood JSON contracts are aligned with `tests/boundary_lrt_contract.rs` and `tests/profile_likelihood_json.rs` proving the schemas. `mixed_model_compiler_inference_contract.md` row-level discipline lives on `FixedEffectInferenceTable`.

**Gaps:**
- `CoefTable` / `ModelSummary` never surface Satterthwaite/KR rows — clients must detour through `FixedEffectInferenceTable`. Downstream-misleading.
- No `wald_confint(model, level)` — trivial; expected by every R client.
- GLMM bootstrap is a refusal stub for *all* families.
- No PB-LRT route, despite boundary-LRT refusal text pointing users to it.
- `inference_simulation_harness.md` unfulfilled — no `reduced_rank`/`boundary` strata, no MC-SE, no seeded determinism.
- Three different status vocabularies (`BoundaryLrtStatus`, `FixedEffectInferenceStatus`, `CoefTablePValuePolicy`) for the same idea.

### 4. Tests, benchmarks, CI — substantive tests, weak infrastructure

**Tests:** default features now include NLopt, so the default gate covers the
BOBYQA/NEWUOA release optimizer path; `--no-default-features` remains the
native TrustBQ/COBYLA fallback gate. Public-API negative test (`tests/public_api.rs`)
spawns a downstream `cargo check` to assert internal types aren't leaked —
proper contract test. Boundary-LRT and profile-JSON contract tests are
substantive.

**Gaps:**
- GLMM unit coverage is thin (45 tests for a major subsystem); covered mostly via integration parity fixtures.
- `tests/compiler_contract_snapshots.rs` (25 tests) runs under default features
  because NLopt is now enabled by default.

**CI (`.github/workflows/ci.yml`, just added):** includes default/NLopt,
`--no-default-features`, unstable-internals, clippy/fmt/doc, and scheduled
parity coverage. **Missing:** release-mode tests, `cargo audit`/`cargo deny`,
and coverage reporting.

**Downstream optimizer profiles:** the Rust crate's default build is the fast
NLopt-backed release path. Downstream packages that need a dependency-light
native build can use `mixeff-rs` with `default-features = false`, selecting
TrustBQ for multi-theta LMMs while keeping the same public model surface. See
`docs/optimizer_profiles.md`.

**Parity:** `scripts/check_julia_parity_fixtures.sh` exists with a tolerant JSON differ (`compare_json_tolerant.py`), but is **not wired into CI**. Checked-in fixtures can rot silently when MixedModels.jl evolves.

**Benchmarks:** no `criterion`, no `benches/` directory, no regression-detection plumbing, no archived baselines.

### 5. Public API, docs, ergonomics — not crates.io-publishable today

**Crates.io blockers:**
- No `README.md` at repo root.
- No `LICENSE` file at repo root (only `MIT` declared in metadata).
- `Cargo.toml` missing `repository`, `homepage`, `documentation`, `readme`, `keywords`, `categories`, `rust-version`, `authors`, `exclude` (the `MixedModels.jl/` checkout would explode publish size).

**API surface too wide:**
- 9 top-level `pub mod`s; `types`, `linalg`, `compiler`, `pathology` should be hidden or feature-gated.
- `pub use ::*` glob re-exports in `src/linalg/mod.rs:18-22`.
- **Zero `#[non_exhaustive]`** on any public enum — `MixedModelError` (15 variants), `Family`, `LinkFunction`, `Optimizer`, `BootstrapIntervalMethod`, `ModelComparisonClass`, etc. Every one is a SemVer-major to extend.
- `LinearMixedModel::fit(reml: bool)` opaque boolean; GLMM has 5 constructors — needs builder.
- `nalgebra::DMatrix`/`DVector` in signatures locks users to a specific minor version.

**Docs:** 5 doctests exist (incl. `src/lib.rs:9`, `src/formula/parser.rs:784`), so it is not true that there are none — but the crate-level quickstart in `lib.rs` is `no_run` and still placeholder-like, which is the real gap. `docs/` mixes 24 stability contracts, internal design notes, and audit reports with no index. No `examples/quickstart.rs` — all 19 examples are benchmarks, parity dumps, or internal probes.

---

## v1.0 blockers (consolidated, prioritized)

### Phase A — publishability (1 week)

1. **Crates.io metadata:** add `README.md`, `LICENSE` file, fill `Cargo.toml` with `repository`, `keywords`, `categories`, `rust-version`, `exclude` (especially `MixedModels.jl/`).
2. **Trim public API:** demote `pub mod types`, `pub mod linalg`, `pub mod compiler`, `pub mod pathology` to either `pub(crate)`, or behind an `unstable-internals` feature flag. Replace `pub use ::*` globs with explicit lists.
3. **Add `#[non_exhaustive]`** to every public enum (`MixedModelError`, `Family`, `LinkFunction`, `Optimizer`, `BootstrapIntervalMethod`, `ContrastSource`, `ModelComparisonClass`, `ModelComparisonReasonCode`, `RandomTermExpansion`, `NewReLevels`, `CategoricalCoding`, `Column`).
4. **Crate-level runnable example** in `lib.rs`. Add `examples/quickstart.rs` (LMM) and `examples/glmm_quickstart.rs` (GLMM).

### Phase B — correctness gates (2 weeks)

5. **Replace MGS pivoted QR with Householder** (`src/linalg/pivot.rs`). Until this lands, parity claims for rank-deficient designs are false. Reorthogonalization is the minimum stop-gap.
6. **Remove `unsafe` raw pointers** from GLMM optimizer closures (`src/model/generalized.rs:1018-1031, 1094-1114`); follow the `Cell`/`RefCell`-callback pattern the LMM NLopt path already uses.
7. **Unify the `1e-30` zero-clamp constants** with the policy-controlled tolerance (`src/model/linear.rs:426-435, 9957-10046`).
8. **Replace user-reachable `.expect(...)`** in `recompute_a_blocks` and `with_block_triple` with `Result` returns.
9. **Formula parser strictness:** reject trailing `+`/`-`, reject adjacent RE blocks without `+`, honor `-` as term removal at top level, error on numeric-literal terms (`2*x1`). Add a parser-level grammar property test.
10. **`FixedTerm::Nested` cleanup:** either implement it in `fixed_design.rs:853` or remove the variant and its `Display` impl.
11. **NaN/Inf rejection** in `DataFrame::add_numeric` (`data.rs:266`).
12. **Train/predict factor consistency:** carry training `levels` snapshot through the fit so predict-time design rebuild can't silently reorder dummy columns (`linear.rs:6772-6905`).

### Phase C — inference surface (1-2 weeks)

13. **`wald_confint(model, level)`** on `CoefTable` (`lower`/`upper` columns) and on `MixedModelFit`.
14. **`CoefTable` exposes Satterthwaite/KR rows** when available — either re-export `FixedEffectInferenceTable` from `stats::*` as the authoritative coeftable, or extend `CoefTable` with `df`, `statistic_name`, `method` fields.
15. **GLMM parametric bootstrap** for Bernoulli/Binomial/Poisson/Gamma.
16. **PB-LRT route** for one added variance component, or downgrade the boundary-LRT refusal text that points users to it.
17. **Inference simulation harness completion:** `reduced_rank` and `boundary` scenarios, replicate counts, MC-SE, fixed-seed determinism, harness output declares its own schema.

### Phase D — CI & parity gates (1 week)

18. **Clippy cleanup (prerequisite for #19's gate):** `cargo clippy --all-targets -- -D warnings` currently fails hard — ~72 lib errors plus ~100 lib-test errors. This is real code work, not CI wiring. Land a clippy cleanup (or a documented, narrow `#[allow]` allowlist policy) *before* promising `-D warnings` in CI.
19. **CI matrix expansion:** keep macOS + Windows default legs, clippy with `-D warnings`, `cargo fmt --check`, `cargo doc --no-deps -D warnings`, `--no-default-features` leg, `--features prima` leg, MSRV-pinned leg.
20. **Wire `check_julia_parity_fixtures.sh` into a scheduled CI job** (weekly is sufficient).
21. **No-default-feature alternative** for `tests/compiler_contract_snapshots.rs` if native-optimizer artifact coverage becomes necessary.
22. **`cargo deny` / `cargo audit`** in CI.

---

## v1.0 nice-to-haves (not blocking)

- `mixeff_rs::prelude` module: `DataFrame`, `LinearMixedModel`, `GeneralizedLinearMixedModel`, `MixedModelFit`, `Family`, `LinkFunction`, `parse_formula`, `Result`, `MixedModelError`.
- `LinearMixedModelBuilder` + `FitOptions { criterion, tolerance, optimizer_override, ... }` to retire `fit(bool)` and the 5-way GLMM constructor zoo.
- `pub use nalgebra;` so downstream code doesn't pin our minor version.
- `docs/README.md` index splitting **stability contracts** (KR, Satterthwaite, profile-likelihood, boundary-LRT, bootstrap, model comparison) from **internal design** (compiler PRD, multivariate plan, parity audits).
- `criterion` benches in `benches/` for the PLS kernel and θ→objective loop; archive baseline JSON, compare with `compare_json_tolerant.py` thresholds.
- Built-in contrast constructors: `CategoricalContrast::sum/helmert/polynomial(levels)`.
- Backtick identifiers and `I(...)` minimal subset.
- Unified status enum vocabulary across `BoundaryLrtStatus` / `FixedEffectInferenceStatus` / `CoefTablePValuePolicy`.
- RAII guard in `deviance(n_agq>1)` so `u` is restored on panic.
- `CHANGELOG.md` + SemVer policy doc naming which (if any) modules remain unstable.
- Re-orthogonalization pass in MGS as a transition before #5 lands properly.
- `criterion`-based PIRLS regression test on the grouseticks Poisson fixture.
- `Formula::variables()` helper for ingest tooling.

---

## Cross-cutting themes

**"The numerics are fine; the framing is what's pre-1.0."** Four of the five audits concluded the underlying mathematics is structurally correct and well-tested. The 1.0 blockers are mostly in the layers wrapping that math: the public API contract, the formula parser's strictness, the inference surface's connective tissue, CI infrastructure, and the publishing checklist.

**Compiler/IR is in flux and shouldn't be 1.0 surface.** `src/compiler/` has ~40+ public types tied to a v0 PRD that is still being shaped. Keeping `pub mod compiler` in the 1.0 contract means every IR refinement is a major bump. Feature-flag it.

**Multivariate Y is post-1.0.** `docs/multivariate_shared_theta.md` requires API-breaking surgery on `FeMat`, `Formula.response`, and every cross-product builder. Either land this *before* freezing 1.0 (~3-4 weeks of focused work) or accept that 1.0 ships single-response and a 2.0 will deliver multivariate. **Recommendation: ship single-response 1.0.**

**The R-client seam is close to ready.** Versioned JSON schemas, stable reason codes, and typed refusals are in place for boundary-LRT, profile-likelihood, and bootstrap. The remaining work is one round-trip test per schema plus a unified status vocabulary — small.

---

## Suggested release sequence

| Milestone | Scope | Estimated effort |
|---|---|---|
| **0.2.0** | Phase A (publishability) — first crates.io release; explicitly tagged "API may change" | 1 week |
| **0.3.0** | Phase B (correctness gates) — silent-acceptance bugs and numerical foot-guns closed | 2 weeks |
| **0.4.0** | Phase C (inference surface) — Wald CIs, GLMM PB, harness complete | 1-2 weeks |
| **0.5.0** | Phase D (CI / parity gates) — full matrix, scheduled parity, audit/deny | 1 week |
| **1.0.0-rc.1** | Phase E (polish) — nice-to-haves, docs index, builder, prelude, CHANGELOG | 1-2 weeks |
| **1.0.0** | tag after one rc cycle without API-breaking feedback | 1 week soak |

**Total estimated effort to 1.0: 7-9 weeks of focused work.**

---

## Out of scope for 1.0 (explicit)

- Multivariate response (`cbind(y1, y2) ~ ...`) — post-1.0; 2.0 candidate.
- InverseGaussian / Normal-as-GLM parametric bootstrap.
- Kenward-Roger for crossed/nested designs beyond the current scalar-test scope.
- `I()` / formula-level transformations beyond a minimal subset.
- `polars`/`arrow` integration as a first-class boundary (users continue to convert to `DataFrame` at the seam).
- Profile likelihood for GLMM.

These should each be filed as mote issues with `--tag post-v1` so the boundary is explicit.
