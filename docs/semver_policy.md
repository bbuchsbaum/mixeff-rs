# Module-Inventory Appendix to VERSIONING.md

This document is the module-inventory appendix to
[`VERSIONING.md`](../VERSIONING.md) (the authoritative versioning contract).
See `VERSIONING.md` for breaking-change rules across the Rust API, numerical
output, formula DSL, JSON schemas, and Julia parity.

Where the two documents conflict, `VERSIONING.md` wins. This document is
restricted to the enumeration below: which modules are in the stable surface,
which are behind `unstable-internals`, and which `#[non_exhaustive]` enums
exist.

---

## Stable surface (covered by SemVer guarantees after 1.0.0)

The following modules and their documented public items are part of the stable
1.0 API, asserted by `tests/public_api.rs`:

- **`mixeff_rs::prelude`** — glob-import bundle: `DataFrame`,
  `LinearMixedModel`, `GeneralizedLinearMixedModel`, `MixedModelFit`,
  `Family`, `LinkFunction`, `parse_formula`, `Result`, `MixedModelError`.

- **`mixeff_rs::formula`** — `parse_formula`, the `Formula` AST types
  (`Formula`, `FixedTerm`, `RandomTerm`, `GroupingFactor`), `FormulaError`.

- **`mixeff_rs::model`** — `DataFrame`, `LinearMixedModel`,
  `LinearMixedModelBuilder`, `GeneralizedLinearMixedModel`,
  `GeneralizedLinearMixedModelBuilder`, `FitOptions`, `ModelCriterion`,
  `MixedModelFit`, `Family`, `LinkFunction`, and all documented fit /
  inference / bootstrap entry points.

- **`mixeff_rs::stats`** — post-fit summaries (`varcorr`, `coeftable`,
  `model_summary`, `lrt`, `bootstrap`, `profile`) and their documented JSON
  contracts (schema names and versions declared in that module, including
  `mixedmodels.fit_summary` `1.0.0`).

- **`mixeff_rs::error`** — `MixedModelError`, `Result`
  (`std::result::Result<T, MixedModelError>`), and stable
  `MixedModelError::code()` / `LinAlgError::code()` strings for downstream
  bindings.

- **`mixeff_rs::types`** — typed model-matrix containers intentionally
  exposed for advanced callers (e.g. `MatrixBlock`, `OptSummary`,
  `FitLogEntry`, `Optimizer`, `GaussHermiteNormalized`).

- **`mixeff_rs::nalgebra`** (re-export via `pub use nalgebra`) — the path is
  stable; the nalgebra *version* behind it follows nalgebra's own SemVer and
  a major nalgebra bump is a major crate bump.

---

## Unstable surface (NOT covered by SemVer guarantees)

The following modules are gated behind the opt-in **`unstable-internals`**
Cargo feature. On a default build they are `pub(crate)`; enabling
`unstable-internals` makes them `pub`. They may change in any release without
a major version bump:

- **`mixeff_rs::compiler`** — model-IR / compiler surface (~40+ public
  types). The compiler/IR is still being shaped and every IR refinement would
  otherwise force a major bump; it is explicitly excluded from the 1.0
  contract. Stable JSON schemas emitted by the compiler (e.g.
  `mixedmodels.compiled_model_artifact`, `mixedmodels.semantic_model`,
  `mixedmodels.theta_map`, `mixedmodels.random_term_card`) are similarly
  excluded — consumers must not rely on them under a stability expectation.

- **`mixeff_rs::pathology`** — diagnostic and identifiability-classification
  internals. Pathology certificates themselves (the typed refusal channel) are
  stable; the internals that compute them are not.

- **`mixeff_rs::datasets`** — bundled reference fixtures; contents and
  provenance metadata may change as parity fixtures are regenerated.

Not reachable downstream at all (always `pub(crate)`):

- **`mixeff_rs::linalg`** — numerical primitives (blocked Cholesky, pivoted
  QR, rank updates). Internal to the fit path; demoted from `pub` for v1.0.

- **`mixeff_rs::optimizer`** (private `mod optimizer`) — internal optimizer
  dispatch; not re-exported.

---

## `#[non_exhaustive]` enum inventory

All public enums are `#[non_exhaustive]` so that adding variants is a
**MINOR** change, not MAJOR. Match on them with a wildcard arm. The current
set:

| Enum | Module |
|------|--------|
| `MixedModelError` | `mixeff_rs::error` |
| `Family` | `mixeff_rs::model` |
| `LinkFunction` | `mixeff_rs::model` |
| `Optimizer` | `mixeff_rs::types` |
| `BootstrapIntervalMethod` | `mixeff_rs::stats` |
| `ContrastSource` | `mixeff_rs::model` |
| `ModelComparisonClass` | `mixeff_rs::stats` |
| `ModelComparisonReasonCode` | `mixeff_rs::stats` |
| `RandomTermExpansion` | `mixeff_rs::formula` |
| `NewReLevels` | `mixeff_rs::model` |
| `CategoricalCoding` | `mixeff_rs::model` |
| `Column` | `mixeff_rs::types` |

Adding a variant to any of these enums is a **MINOR** release. Downstream
code must match with `_ => { /* … */ }` or similar.

---

## `#[non_exhaustive]` public struct inventory

Public structs that expose data for reading but should not be constructed by
downstream code through struct literals are also `#[non_exhaustive]`. Use
constructors, builders, accessors, or serde payload constructors instead.

| Struct | Module |
|--------|--------|
| `LinearMixedModel` | `mixeff_rs::model` |
| `GeneralizedLinearMixedModel` | `mixeff_rs::model` |
| `OptSummary` | `mixeff_rs::types` |
| `CategoricalColumn` | `mixeff_rs::model` |
| `CategoricalContrast` | `mixeff_rs::model` |
| `CoefTable` | `mixeff_rs::stats` |
| `ModelSummary` | `mixeff_rs::stats` |
| `ModelSummaryRow` | `mixeff_rs::stats` |
| `VarCorr` | `mixeff_rs::stats` |
| `VarCorrComponent` | `mixeff_rs::stats` |
| `FitSummaryPayload` | `mixeff_rs::stats` |

Adding a field to any of these structs is a **MINOR** release when the field is
defaultable for serde or unavailable to downstream struct literals. Removing or
renaming an existing public field remains a **MAJOR** change unless the type is
behind `unstable-internals`.

---

## Out of scope for 1.0

The following are deliberately deferred (2.0 candidates); their absence is
not a defect:

- Multivariate response (`cbind(y1, y2) ~ ...`).
- Gamma GLMM parametric bootstrap.
- Kenward-Roger beyond the current scalar-test scope.
- Full `I()` / formula-level transformations beyond a minimal subset.
- First-class `polars` / `arrow` ingestion.
- Profile likelihood for GLMM.
