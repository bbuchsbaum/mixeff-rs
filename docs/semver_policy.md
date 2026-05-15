# SemVer Policy

This document defines what `mixeff-rs` considers a breaking change, which parts
of the public surface are covered by stability guarantees once **1.0.0** ships,
and which parts are explicitly **not** covered.

It is a companion to [`../CHANGELOG.md`](../CHANGELOG.md) and the phased plan in
[`v1_0_release_roadmap.md`](v1_0_release_roadmap.md).

## Pre-1.0 (current state)

No `1.0.0` has been tagged. While the version is `0.x`, **any release may
contain breaking changes**, including minor-version bumps, per Cargo's `0.x`
semantics. Pin an exact version if you need stability before 1.0.

## Versioning after 1.0.0

Once `1.0.0` is tagged, the crate follows [SemVer 2.0.0](https://semver.org/):

- **MAJOR** — a breaking change to the stable surface (see below).
- **MINOR** — backward-compatible additions (new functions, new enum variants
  on `#[non_exhaustive]` enums, new methods, new modules).
- **PATCH** — backward-compatible bug fixes, including numerical-accuracy fixes.

### What counts as a breaking change

- Removing or renaming any item in the stable surface.
- Changing a public function/method signature in a non-additive way.
- Adding a variant to a public enum that is **not** `#[non_exhaustive]`.
- Adding a required field to a public struct with public fields, or to a
  struct constructible by a public constructor, in a non-additive way.
- Raising the MSRV (treated as a minor bump with a CHANGELOG note, not a
  major bump, but never silently).

### What is *not* a breaking change

- Adding variants to a `#[non_exhaustive]` enum. All public enums
  (`MixedModelError`, `Family`, `LinkFunction`, `Optimizer`,
  `BootstrapIntervalMethod`, `ContrastSource`, `ModelComparisonClass`,
  `ModelComparisonReasonCode`, `RandomTermExpansion`, `NewReLevels`,
  `CategoricalCoding`, `Column`, …) are `#[non_exhaustive]` for this reason —
  match on them with a wildcard arm.
- Numerical-output changes that move a fitted value within documented
  tolerance to track an upstream MixedModels.jl correction. The objective,
  θ ordering, and factor-block layout are parity-stable; a *bug fix* that
  corrects a wrong number is a PATCH, not a breaking change. Cross-language
  parity is verified by the scheduled Julia parity CI gate.
- Changes to anything in the **unstable surface** (next section).

## Stable surface (covered by guarantees after 1.0.0)

- `mixeff_rs::prelude`
- `mixeff_rs::formula` — `parse_formula`, the `Formula` AST, `FormulaError`.
- `mixeff_rs::model` — `DataFrame`, `LinearMixedModel`,
  `GeneralizedLinearMixedModel`, `MixedModelFit`, `Family`, `LinkFunction`,
  and the documented fit / inference / bootstrap entry points.
- `mixeff_rs::stats` — post-fit summaries (`varcorr`, `coeftable`,
  `model_summary`, `lrt`, `bootstrap`, `profile`) and their documented JSON
  contracts.
- `mixeff_rs::error` — `MixedModelError`, `Result`.
- `mixeff_rs::types` — the typed model-matrix containers intentionally exposed
  for advanced callers (e.g. `MatrixBlock`; this is asserted by
  `tests/public_api.rs`).
- The re-exported `mixeff_rs::nalgebra` **path** is stable; the nalgebra
  *version* behind it follows nalgebra's own SemVer and a major nalgebra bump
  is a major bump here.

## Unstable surface (NOT covered by guarantees)

The following are gated behind the opt-in **`unstable-internals`** Cargo
feature and are **not** part of the 1.0 stability contract. On a default
build they are `pub(crate)` (not reachable downstream); enabling
`unstable-internals` exposes them as `pub`. Anything reachable only via that
feature may change in any release without a major bump:

- `mixeff_rs::compiler` — the model-IR / compiler surface (~40+ public types)
  is tied to a v0 PRD that is still being shaped. Every IR refinement would
  otherwise force a major bump; it is explicitly excluded.
- `mixeff_rs::pathology` — diagnostic/identifiability internals.
- `mixeff_rs::datasets` — bundled reference fixtures; contents and provenance
  metadata may change as parity fixtures are regenerated.

`mixeff_rs::linalg` is already `pub(crate)` and is not public surface at all.

## Out of scope for 1.0

The following are deliberately deferred to post-1.0 (2.0 candidates) and their
absence is not a defect:

- Multivariate response (`cbind(y1, y2) ~ ...`).
- Gamma GLMM parametric bootstrap.
- Kenward-Roger beyond the current scalar-test scope.
- Full `I()` / formula-level transformations beyond a minimal subset.
- First-class `polars` / `arrow` ingestion (convert to `DataFrame` at the seam).
- Profile likelihood for GLMM.

## MSRV

The minimum supported Rust version is **1.80** (declared in `Cargo.toml`).
An MSRV increase is called out in the CHANGELOG and treated as at least a
minor bump; it is never done silently in a patch.
