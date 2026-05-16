# Versioning Policy

This document is the authoritative versioning contract for the `mixeff-rs`
crate. It defines what counts as a breaking change for each surface the crate
exposes — including the numerical output, the formula DSL, and the serialized
JSON contracts consumed by downstream R and Python wrappers.

It supersedes the narrower stability notes in
[`docs/semver_policy.md`](docs/semver_policy.md), which remains as the
module-by-module stable/unstable inventory. Where the two disagree, this
document wins; `docs/semver_policy.md` should be read as the appendix that
enumerates the exact stable/unstable module list.

Companion documents:
- [`CHANGELOG.md`](CHANGELOG.md) — release-by-release record.
- [`docs/semver_policy.md`](docs/semver_policy.md) — stable vs. unstable module inventory.
- [`docs/julia_parity_fixture_drift.md`](docs/julia_parity_fixture_drift.md) — the Julia parity gate.

---

## 1. Scope: the five versioned surfaces

`mixeff-rs` is a numerical library co-developed against a Julia reference
(`MixedModels.jl`) and consumed by downstream R/Python wrapper packages. It
therefore has **five distinct surfaces**, each with its own breaking-change
definition. A release is MAJOR if *any* surface has a breaking change.

| # | Surface | What it is |
|---|---------|------------|
| A | Rust API | Public types, functions, traits, modules, MSRV |
| B | Numerical output | Fitted objective, θ, β, σ, vcov, ranef values |
| C | Formula DSL | The lme4/R formula mini-language accepted by `parse_formula` |
| D | Serialized JSON contracts | `mixedmodels.*` schemas consumed over the wire by R/Python |
| E | Julia-parity contract | The numerical agreement promised against `MixedModels.jl` |

The stable Rust module list is fixed by `docs/semver_policy.md` and asserted by
`tests/public_api.rs`. The `unstable-internals` feature (`compiler`,
`pathology`, `datasets`) is **outside all guarantees below** and may change in
any release.

---

## 2. SemVer interpretation

After `1.0.0` the crate follows [SemVer 2.0.0](https://semver.org/). Pre-1.0
semantics are in §5.

### 2.A — Rust API

**MAJOR (breaking):**
- Removing or renaming any item in the stable surface.
- Changing a public signature in a non-additive way (new required parameter,
  changed return type, changed trait bound).
- Adding a variant to a public enum that is **not** `#[non_exhaustive]`. (All
  public enums are `#[non_exhaustive]`; this is a backstop, not an expected
  path.)
- Adding a non-defaulted public field to a publicly constructible struct.
- A **major** version bump of a dependency that appears in the public API
  (notably `nalgebra` — `DMatrix`/`DVector` appear in signatures, and
  `mixeff_rs::nalgebra` is re-exported). The re-export *path* is stable; the
  version behind it is not.
- Removing a Cargo feature, or changing a default feature in a way that removes
  default-built API. Adding the *default* `nlopt` path or the optional `prima`
  path is additive; making `nlopt` non-default would be MAJOR.

**MINOR (additive):**
- New functions, methods, modules, traits.
- New variants on `#[non_exhaustive]` enums.
- New optional Cargo features that do not alter default behavior.
- New defaulted struct fields where the struct is `#[non_exhaustive]` or
  builder-constructed.
- MSRV increases (see §4 — versioned as MINOR, never silent).

**PATCH:**
- Bug fixes that do not change any stable signature.
- Numerical-accuracy fixes within the tolerance band of §2.B.
- Documentation, internal refactors, performance work with no observable API
  or numerical-output change.

### 2.B — Numerical output

This is the hard question for a numerical library and is given its own policy
in §3. Summary of the rule:

- A numerical-output change that keeps every reported quantity within the
  **documented parity tolerance** (§3) is **PATCH** (if a correctness fix) or
  **MINOR** (if it rides along with an additive feature).
- A numerical-output change that moves any reported quantity **beyond
  tolerance** is **MAJOR**, *unless* it is gated behind an opt-in (a new
  optimizer profile, a new Cargo feature, or an explicit `FitOptions` setting)
  whose default preserves prior output.
- A change correcting a **provably wrong** number (a bug that produced an
  answer outside tolerance of the mathematically correct value, or an
  upstream `MixedModels.jl` correction we are tracking) is **PATCH** even if
  large in magnitude. Wrong answers carry no compatibility guarantee. Such a
  fix must be called out explicitly in `CHANGELOG.md` under a
  **`### Fixed (numerical)`** heading with the affected quantities and a
  before/after example.

### 2.C — Formula DSL

The formula DSL is a user-facing language and is versioned like a parser.

**MAJOR (breaking):**
- Changing the *meaning* of an expression that previously parsed and fit. The
  canonical hazard: today `y ~ x1 - x2` is silently treated as `1 + x1 + x2`;
  correcting `-` to lme4 term-removal semantics **changes the fitted model**
  for existing valid-looking input and is therefore MAJOR (or gated behind a
  dialect/strict flag whose default is the old behavior until the next MAJOR).
- Removing support for a previously accepted construct.
- Changing the categorical reference-level rule (first-appearance order today;
  any move toward lme4 alphabetical ordering changes β and is MAJOR — it is
  also a §2.B numerical break).

**MINOR (additive):**
- Accepting new syntax that was previously rejected (`I(x^2)`, `poly(x, 2)`,
  backtick identifiers, function-call atoms, additional contrast bases).
- New, opt-in stricter parsing modes.

**PATCH:**
- Rejecting input that was *silently mis-parsed* and would have produced a
  wrong model (trailing `+`/`-`, adjacent RE blocks without `+`, numeric-
  literal terms like `2*x1`). Turning a silent-acceptance footgun into a
  typed `FormulaError` is a **bug fix**, not a breaking change, because the
  prior behavior had no defensible contract. These must still be
  CHANGELOG-noted under **`### Fixed (parser strictness)`**.

  Rationale: SemVer protects *correct* observable behavior. A formula that
  silently fit the wrong model was never a behavior a user could rely on.

### 2.D — Serialized JSON contracts

Each `mixedmodels.*` schema carries its own `schema_name` + `schema_version`
(e.g. `mixedmodels.boundary_lrt` `1.0.0`,
`mixedmodels.profile_likelihood_ci` `1.0.0`,
`mixedmodels.parametric_bootstrap_lrt` `1.0.0`,
`mixedmodels.bootstrap_run` `1.0.0`,
`mixedmodels.fit_summary` `1.0.0`,
`mixedmodels.fixed_effect_inference_table` `1.0.0`). **These schema versions
are versioned independently of the crate version** and follow SemVer in their
own right:

- **Schema PATCH** — clarification, doc-only, no wire change.
- **Schema MINOR** — adding an optional field. Consumers using
  forward-compatible parsing (ignore-unknown-fields) are unaffected. A crate
  MINOR may carry a schema MINOR.
- **Schema MAJOR** — removing/renaming a field, changing a field's type or
  units, changing the meaning of an existing field, or changing a stable
  reason-code string. A schema MAJOR is **always a crate MAJOR**, because
  downstream wrappers deserialize these directly.

Rules:
- The crate must never emit a schema-MAJOR change without bumping the crate to
  MAJOR and bumping the affected `*_SCHEMA_VERSION` constant.
- Stable reason-code strings (e.g.
  `boundary_lrt_requires_variance_component_comparison`,
  `boundary_lrt_mixture_weights_not_certified`) are part of the wire contract:
  renaming one is a schema-MAJOR change. Adding a new reason code is
  schema-MINOR **only if** consumers are documented to treat unknown codes as
  an opaque refusal (they are).
- Stable error-code strings returned by `MixedModelError::code()` and
  `LinAlgError::code()` are likewise part of the downstream binding contract.
  Renaming or reusing one for a different condition is a MAJOR change; adding a
  new code for a new `#[non_exhaustive]` variant is MINOR when consumers are
  documented to handle unknown codes opaquely.
- Schemas behind `unstable-internals` (the `compiler`/IR schemas:
  `mixedmodels.compiled_model_artifact`, `mixedmodels.semantic_model`,
  `mixedmodels.theta_map`, `mixedmodels.random_term_card`, etc.) are **not**
  covered and may change without a crate MAJOR. Downstream code must not
  deserialize them under a stability expectation.

### 2.E — Julia-parity contract

The promise "agrees with `MixedModels.jl`" is itself versioned via the
checked-in fixtures under `tests/fixtures/parity/` and the drift gate
`scripts/check_julia_parity_fixtures.sh` (default tolerance: `abs=1e-7`,
`rel=1e-8`, per `scripts/compare_json_tolerant.py`).

- The parity contract is pinned to a **specific `MixedModels.jl` version**,
  recorded in the fixture provenance. That pinned version is part of the
  release notes for any parity-sensitive release.
- Tracking an **upstream `MixedModels.jl` bug fix** that moves our output to
  match a corrected reference is a crate **PATCH** (it makes a wrong number
  right), even though fixtures are regenerated.
- Tracking an upstream **behavioral/algorithmic change** in `MixedModels.jl`
  that moves results beyond tolerance for *correct* prior inputs is a crate
  **MAJOR** (or gated), like any §3 algorithmic change.
- **Documented divergences** (e.g. the `reduced_rank_unit_correlation` fixture,
  where the Rust pathology certificate intentionally classifies differently
  from lme4/MixedModels.jl) are part of the contract, not regressions. The
  Rust pathology certificate — not the external engines — is the contract
  oracle. Removing or reclassifying a documented divergence is a MAJOR change.

---

## 3. Are bit-level numerical changes breaking? — the policy

**Short answer: no, not within a documented tolerance. Yes, beyond it, unless
opt-in.**

### 3.1 The tolerance band

`mixeff-rs` guarantees its checked-in parity fixture quantities to a
**parity tolerance** of:

> **absolute `1e-7` or relative `1e-8`** (whichever is satisfied), applied
> element-wise to: the objective (deviance / -2 log-likelihood), θ, β, σ,
> the fixed-effect covariance matrix, and ranef BLUPs, on the parity fixture
> corpus and on any input of comparable conditioning when evaluated through the
> canonical fixture/parity pipeline.

This is exactly the tolerance the Julia drift gate already enforces
(`scripts/compare_json_tolerant.py`, `--abs-tol 1e-7 --rel-tol 1e-8`), so the
public guarantee and the CI gate are the **same number** by construction. No
new machinery is introduced — the guarantee is the test that already runs.

Bit-for-bit reproducibility is **explicitly not promised.** It is infeasible
for an iterative optimizer (BOBYQA/NEWUOA/COBYLA/TrustBQ) whose path depends on
LLVM codegen, FMA contraction, BLAS kernels, and platform math libraries.
Promising bit-stability would make every compiler upgrade a MAJOR bump and is
not defensible for a numerical library.

Fit-level optimizer smoke tests are a related but separate release gate. On
flat, large-θ objectives, two optimizer paths can reach indistinguishable
objective values while reporting slightly different θ/σ coordinates. Those
tests must still prove the same optimizer family, an accepted stop, and an
objective inside the documented parity band; they may use a looser coordinate
tolerance when the stricter fixed-fixture gate above is unchanged. Any such
case should be commented in the test so it is not mistaken for a fixture-drift
acceptance.

### 3.2 Classification of numerical changes

| Change kind | Effect | SemVer |
|---|---|---|
| Optimizer converges to the same optimum within tolerance (codegen/FMA/BLAS drift, refactor) | within band | PATCH / not a release-gated change |
| Tightened/loosened convergence tolerance that keeps results within the parity band | within band | PATCH |
| Bug fix correcting a number that was outside tolerance of the true value | corrects wrong output | PATCH (CHANGELOG `### Fixed (numerical)`) |
| Tracking an upstream `MixedModels.jl` correction | corrects wrong output | PATCH |
| New optimizer/algorithm selected by default, moving correct results beyond band | beyond band, default | **MAJOR** |
| New optimizer/algorithm available only via `FitOptions`/feature, default unchanged | beyond band, opt-in | MINOR |
| Changing the default optimizer profile such that converged results shift beyond band | beyond band, default | **MAJOR** |
| Reference-level / contrast-coding default change (also a §2.C break) | beyond band, default | **MAJOR** |

### 3.3 The opt-in escape hatch

Algorithmic improvements that move *correct* results beyond tolerance are not
forbidden between MAJORs — they must be **opt-in with an output-preserving
default**:

- a new value on the `Optimizer` enum or a new optimizer profile, selected via
  `FitOptions`, **not** by changing the default;
- or a new Cargo feature whose absence preserves prior output.

The default fit path of a non-MAJOR release must keep producing output within
the parity band of the previous release on the fixture corpus. The Julia drift
gate is the enforcing test; it must pass (or be explicitly re-accepted with
provenance) for every release, and a re-accept that moves the default beyond
band is by definition a MAJOR release.

### 3.4 Why this policy

- **Downstream R/Python users run statistical analyses.** They need stable
  *conclusions* (CIs, LRT decisions, point estimates), not stable bits. A
  `1e-7` band is far tighter than any reported precision and far below
  inferential significance, so conclusions are protected while implementation
  freedom is retained.
- **The guarantee equals an existing gate.** Reusing the Julia drift
  tolerance means the policy is testable today and cannot drift away from CI.
- **Wrong answers have no warranty.** Treating correctness fixes as PATCH
  (with loud CHANGELOG entries) is standard for numerical libraries (LAPACK,
  SciPy) and avoids the perverse outcome of shipping a known-wrong number to
  preserve "compatibility."
- **Algorithmic progress stays possible** via the opt-in hatch without
  silently changing users' results.

---

## 4. MSRV policy

- The Minimum Supported Rust Version is declared in `Cargo.toml`
  (`rust-version`) and is currently **1.85**. This is the first stable
  toolchain with Rust 2024 edition support, which is required by current
  dependency resolution.
- An MSRV increase is a **MINOR** version bump, never a PATCH, and is **never
  silent**: it must appear in `CHANGELOG.md` under a `### Changed` entry naming
  the new MSRV and the feature that forced it.
- MSRV is **not** treated as a MAJOR bump. Rationale: MSRV bumps are routine
  for a library tracking a moving toolchain; treating each as MAJOR would
  exhaust the MAJOR channel and devalue it. Downstream consumers that must pin
  an old toolchain can pin a compatible `mixeff-rs` MINOR.
- CI runs an MSRV-pinned leg so the declared MSRV is enforced, not aspirational.
- Bumping a dependency in a way that *transitively* raises the effective MSRV
  is treated the same as raising MSRV directly (MINOR + CHANGELOG note).

---

## 5. Pre-1.0 vs post-1.0

### Pre-1.0 (current — `0.x`)

No `1.0.0` has been tagged. Per Cargo's `0.x` semantics, **any release,
including a MINOR bump, may contain breaking changes** on any of the five
surfaces. Consumers needing stability before 1.0 must pin an exact version
(`=0.x.y`). The release sequence to 1.0 is in
[`docs/v1_0_release_roadmap.md`](docs/v1_0_release_roadmap.md).

During `0.x`, schema versions may still advance independently; downstream
wrappers should already key off `schema_name`+`schema_version`, not the crate
version, so this habit is in place before 1.0.

### What `1.0.0` promises downstream R/Python consumers

Tagging `1.0.0` is a promise to the wrapper packages specifically:

1. **Stable JSON wire contracts.** The `1.0.0` `mixedmodels.*` schemas
   (boundary-LRT, profile-likelihood CI, parametric-bootstrap LRT,
   bootstrap-run, fixed-effect inference table, fit-summary) will not change in
   a wire-incompatible way without a crate MAJOR. Wrappers may deserialize
   them directly and rely on stable reason codes and the typed-refusal
   channels (`PValuePolicy::Unavailable{reason}`, the `boundary_lrt_*` reason
   strings).
2. **Stable numerical conclusions.** Reported quantities stay within the §3
   parity band across MINOR/PATCH releases on the fixture corpus; statistical
   conclusions do not change under a non-MAJOR upgrade.
3. **Stable formula meaning.** A formula that fits a given model under `1.x`
   fits the same model (within the parity band) under any later `1.y`.
4. **A stable, deliberately narrow Rust API** (the modules enumerated in
   `docs/semver_policy.md`, asserted by `tests/public_api.rs`). The
   `unstable-internals` surface (compiler/IR, pathology, datasets) is
   explicitly excluded so IR evolution is not a MAJOR break.
5. **An explicit refusal contract.** Unsupported analyses fail loudly with a
   typed error and stable reason code rather than returning a fabricated
   number (per
   [`docs/mixed_model_compiler_inference_contract.md`](docs/mixed_model_compiler_inference_contract.md)).
   Refusals are part of the contract; converting a refusal into a (correct)
   answer is MINOR, converting an answer into a refusal is MAJOR.

Out of scope for `1.0` (their absence is not a defect; they are 2.0
candidates): multivariate response, Gamma GLMM bootstrap, Kenward-Roger beyond
the scalar-test scope, full `I()`/formula transformations, first-class
polars/arrow ingestion, GLMM profile likelihood.

---

## 6. Deprecation process and support window

1. **Announce.** A deprecated item is marked `#[deprecated(since = "x.y.z",
   note = "use … instead")]` and listed in `CHANGELOG.md` under
   `### Deprecated` with the replacement and the planned removal MAJOR.
2. **Grace period.** A deprecated stable-surface item remains functional for
   **at least one full MINOR cycle and never less than 6 months**, whichever
   is longer, before it may be removed.
3. **Removal.** Removal happens only in a **MAJOR** release. The removal is
   listed in `CHANGELOG.md` under `### Removed` with a migration note.
4. **Schema deprecation.** A deprecated JSON field is retained and still
   populated for the same grace period; its deprecation is documented in the
   relevant `docs/*_contract.md` and the schema version is bumped (MINOR for
   "field now optional/deprecated", MAJOR at removal).
5. **Security/correctness exception.** A change required to fix a soundness or
   correctness defect (e.g. a number outside tolerance of the true value, an
   unsound `unsafe` block) may bypass the grace period. It is shipped as the
   minimum bump that honors §2–§3 (PATCH for a pure correctness fix; MAJOR if
   it unavoidably changes a correct, in-tolerance result) with a prominent
   CHANGELOG entry.
6. **Support window.** Only the latest MAJOR line receives feature work. The
   immediately preceding MAJOR line receives **security and correctness
   backports for 12 months** after the next MAJOR is tagged. Older lines are
   unsupported.

---

## 7. Relationship to downstream R / Python packages

The Rust crate version and the downstream wrapper versions are **decoupled**.
Wrappers have their own release cadence (CRAN / PyPI conventions) and their
own user-facing API. They are bound to the crate **only through the versioned
JSON schemas and the pinned crate version they vendor/link**, never through
the crate's Rust SemVer directly.

### Compatibility matrix

Each wrapper repository maintains a `COMPATIBILITY.md` table; the canonical
source is the schema versions, not the crate version:

| `mixeff-rs` crate | Schema set | R `mixeff` pkg | Python `mixeff` pkg | Pinned `MixedModels.jl` |
|---|---|---|---|---|
| `1.0.x` | boundary_lrt 1.0.0 · profile_likelihood_ci 1.0.0 · parametric_bootstrap_lrt 1.0.0 · bootstrap_run 1.0.0 · fixed_effect_inference_table 1.0.0 · fit_summary 1.0.0 | `≥ 1.0` | `≥ 1.0` | `4.x` (recorded in fixture provenance) |
| _(future row per crate MINOR/MAJOR; only schema-version columns are load-bearing)_ | | | | |

Rules:
- **Wrappers key off `schema_name`+`schema_version`, not the crate version.**
  On a mismatch the wrapper must fail with a typed schema error (e.g.
  `mm_schema_error`) and must never silently reinterpret payloads or recompute
  intervals/p-values it received as a contract (per the R-binding rule in the
  contract docs and `docs/r_layer_proposal.md`).
- A crate MINOR/PATCH that does not move any schema version requires **no**
  wrapper release.
- A crate MAJOR, or any schema-MAJOR, requires a coordinated wrapper release;
  the matrix row is added before the crate MAJOR is tagged.
- The **`MixedModels.jl` pinned version** is part of the matrix because the
  parity contract (§2.E) is meaningful only relative to a specific reference
  version. Regenerating fixtures against a new `MixedModels.jl` updates this
  column and is release-noted.

---

## 8. Release checklist (versioning-relevant subset)

Before tagging any release (see `RELEASE_CHECKLIST.md` for the full runbook):

1. Decide the bump by taking the **maximum** required across surfaces A–E.
2. Run the Julia parity drift gate
   (`scripts/check_julia_parity_fixtures.sh`); if results moved, classify per
   §3 and confirm the chosen bump (a beyond-band default move ⇒ MAJOR).
3. Confirm `tests/public_api.rs` still passes (stable surface unchanged unless
   intentionally MAJOR).
4. Bump any `*_SCHEMA_VERSION` constant that changed and update the matching
   `docs/*_contract.md`.
5. Update `CHANGELOG.md`, including any `### Fixed (numerical)` /
   `### Fixed (parser strictness)` / `### Deprecated` / `### Removed`
   sections.
6. Update the compatibility matrix and the pinned `MixedModels.jl` version if
   fixtures were regenerated.
7. Update MSRV note if the toolchain floor moved (MINOR + CHANGELOG).
