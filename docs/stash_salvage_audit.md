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

## Superseded On Main

- Satterthwaite and Kenward-Roger scaffolding. Current `main` already has
  release-ready contract docs, inference-table support, parity fixtures, and
  public result contracts that supersede the older stash shape.

## Candidate Later Slices

- Bootstrap LRT and fixed-effect bootstrap payload extensions.
- GLMM comparison and speed-parity harnesses under `comparison/`, `examples/`,
  and `tests/fixtures/parity/`.
- Objective/kernel experiments in `src/model/linear.rs`, especially the
  dense/sparse cross-product and `faer`/`dyn-stack` trials. These should be
  evaluated against the isolated per-evaluation benchmark before porting.

## Do Not Restore

- `vendor/lmerTestR/`. It was only a reference copy and should remain outside
  the Rust repository.
