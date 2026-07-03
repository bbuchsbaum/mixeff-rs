# Changelog

All notable changes to `mixeff-rs` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
once 1.0.0 ships. See [VERSIONING.md](VERSIONING.md) for the authoritative
versioning contract (breaking-change rules across the Rust API, numerical output,
formula DSL, JSON schemas, and Julia parity) and
[docs/semver_policy.md](docs/semver_policy.md) for the module-by-module stable
vs. `unstable-internals` surface inventory.

## [Unreleased]

No changes yet.

## [1.0.0-rc.1] - 2026-07-03

Work toward the 1.0.0 release. The numerics (PLS/PIRLS, blocked Cholesky,
profiled (RE)ML) are structurally stable; the remaining churn is in the public
API framing, the inference surface, and release infrastructure.

### Added

- `VERSIONING.md` — authoritative versioning policy covering the five versioned
  surfaces (Rust API, numerical output, formula DSL, JSON schemas, Julia parity),
  SemVer interpretation, numerical-tolerance band, MSRV policy, deprecation
  process, and downstream R/Python compatibility matrix.
- `RELEASE_CHECKLIST.md` — step-by-step release runbook (pre-release gates,
  version/changelog bump, package verification, tag, publish, soak, post-release).
- `LinearMixedModelBuilder` + `GeneralizedLinearMixedModelBuilder` +
  `FitOptions` / `ModelCriterion` — fluent construction that collapses the
  `fit(reml: bool)` boolean and the GLMM `new_with_*` constructor set into one
  chained surface. Additive; the existing constructors still work.
- `mixeff_rs::prelude` — glob-import module bundling `DataFrame`,
  `LinearMixedModel`, `GeneralizedLinearMixedModel`, `MixedModelFit`, `Family`,
  `LinkFunction`, `parse_formula`, `Result`, and `MixedModelError`.
- `pub use nalgebra;` — downstream code can name the exact `nalgebra` this
  crate builds against (it appears in public signatures) without pinning its
  own dependency to our minor version.
- `wald_confint` on `CoefTable` and `MixedModelFit`; `CoefTable` surfaces
  Satterthwaite / Kenward-Roger degrees-of-freedom rows.
- GLMM parametric bootstrap for Bernoulli / Binomial / Poisson families;
  parametric-bootstrap LRT route for one added variance component.
- Inference simulation harness (`examples/inference_route_simulation.rs`) with
  a stable JSON output schema.
- `mixedmodels.fit_summary` `1.0.0` — versioned JSON envelope for downstream
  wrappers that need objective values, optimizer metadata, coefficient tables,
  variance components, and summary-table rows in one stable payload.
- Stable `MixedModelError::code()` and `LinAlgError::code()` machine strings
  for downstream bindings that must branch without parsing display text.
- Default-feature compiler-contract coverage
  (`tests/compiler_contract_structure.rs`) so wire-serialization regressions
  are caught on every CI run, not only the NLopt leg.
- Expanded CI matrix (Linux/macOS/Windows, `--no-default-features`,
  `--features nlopt`, `--features prima`, MSRV-pinned leg), scheduled Julia
  parity gate, and a `cargo deny` / `cargo audit` supply-chain job.

### Changed

- `docs/semver_policy.md` reduced to the module-inventory appendix of
  `VERSIONING.md`; the broader policy prose that duplicated `VERSIONING.md` has
  been removed and replaced with a header note cross-linking to
  `VERSIONING.md` as the authoritative source.
- `#[non_exhaustive]` added to public enums and model/result structs where
  downstream construction should stay builder/accessor based; adding fields or
  variants is no longer a SemVer-major change.
- Solver internals on `LinearMixedModel` and `GeneralizedLinearMixedModel`
  sealed behind accessors / trait methods so 1.0 does not freeze the PLS,
  Cholesky, compiler-artifact, or PIRLS working-state layout.
- Numerical primitives (`linalg`) demoted from `pub` to `pub(crate)` — they are
  internal to the fit path, not part of the stable API.
- `compiler`, `datasets`, and `pathology` are no longer part of the default
  public API. They are `pub` only under the new opt-in `unstable-internals`
  Cargo feature (and `pub(crate)` otherwise), so the in-flux compiler/IR is not
  frozen into the 1.0 SemVer contract. Internal crate code is unaffected;
  downstream code that needs them must enable `unstable-internals`. CI runs an
  `unstable-internals` leg on every push so that surface stays tested.
- MSRV declared honestly as Rust 1.85, matching the current dependency graph's
  Rust 2024 edition requirement.
- Documented, deliberately narrow crate-level Clippy `#[allow]` policy for
  lints that would obscure the reference algorithms or change numeric
  semantics (see `src/lib.rs`).

### Notes

- This is a 1.0 release candidate. The stable API and wire-contract surface are
  intended to soak before the final 1.0.0 tag; any breaking RC feedback will
  require a new `-rc.N` release and restart the soak clock.
- Multivariate response (`cbind(y1, y2) ~ ...`), Gamma GLMM bootstrap,
  Kenward-Roger beyond the current scalar-test scope, full `I()` /
  formula-level transformations, first-class `polars`/`arrow` ingestion, and
  GLMM profile likelihood are explicitly **out of scope for 1.0** and tracked
  as post-1.0 work.

[Unreleased]: https://github.com/bbuchsbaum/mixeff-rs/compare/v1.0.0-rc.1...HEAD
[1.0.0-rc.1]: https://github.com/bbuchsbaum/mixeff-rs/releases/tag/v1.0.0-rc.1
