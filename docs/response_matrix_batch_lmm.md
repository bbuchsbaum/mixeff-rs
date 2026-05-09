# Response-Matrix Batch LMM API

The response-matrix batch API fits or profiles independent linear mixed-model
responses that share one compiled model structure.  A response matrix has shape
`n_observations x n_responses`; each column is treated as a separate Gaussian
LMM response with the same fixed-effect design, random-effect structure,
grouping levels, theta parameterization, bounds, and optimizer controls.

This is not a multivariate mixed model.  The API does not estimate
cross-response covariance and intentionally uses generic vocabulary such as
response columns, not domain-specific terms.

## Entry Point

Use a template `LinearMixedModel` to construct the cache:

```rust
let batch = LinearMixedModelBatch::from_model(&model)?;
let fit = batch.fit_responses(
    &responses,
    ResponseBatchMode::ProfileAtTheta {
        theta: model.theta(),
        reml: true,
    },
)?;
```

`LinearMixedModelBatch::from_model` materializes and caches the invariant
full-rank fixed design plus structural blocked cross-product pattern for
`[Z X]'[Z X]`.  Response-dependent quantities are computed from the supplied
matrix columns.

## Modes

- `ProfileAtTheta` profiles beta, sigma, PWRSS, and objective for every valid
  response column at one caller-supplied theta.  The structural factorization is
  built once per theta and reused across chunks.
- `OptimizeSharedTheta` optimizes one theta against the aggregate objective,
  then returns per-column profiled quantities at that shared theta.
- `OptimizePerColumn` optimizes theta independently per valid response column
  using the cached structure and the requested warm-start policy.
- `OptimizeGrouped` optimizes one shared theta per caller-supplied column group.

All modes are serial and deterministic.  `BatchOptions::chunk_columns` bounds
the number of response columns profiled at once without changing output shape or
results.

## Results And Failures

`ResponseBatchFit` is column-major:

- `beta`: `p x q`
- `sigma`, `pwrss`, `objective`: length `q`
- `theta`: shared, grouped, or `ntheta x q`
- `status`: one `ResponseFitStatus` per response column
- `diagnostics`: structured column-local reason rows

Non-finite and constant response columns are reported in `status` and
`diagnostics`; they do not abort other columns.  Dimension mismatches and invalid
shared theta values remain API errors because the shared model contract is not
valid.

## Boundaries

The LMM batch contract is the stable first target.  GLMM response-matrix fitting
is deferred because PIRLS weights and working responses are column-specific.

## Amortization Benchmark

Use the paired Rust and R harnesses to compare response-matrix profiling against
an `lme4::lmer()` loop over generated response columns:

```text
MIXEDMODELS_RESPONSE_BATCH_QS=1,4,16,64 cargo run --example bench_response_matrix_batch
MIXEDMODELS_RESPONSE_BATCH_QS=1,4,16,64 Rscript scripts/bench_response_matrix_lme4.R
```

The benchmark uses affine variants of the original response in `dyestuff`,
`sleepstudy`, and `penicillin`.  Those columns preserve the same fixed/random
design and relative variance structure, so the comparison isolates how much
model-structure work is amortized when fitting many independent responses.
