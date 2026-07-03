# Lazy Fixed-Design Materialization

Status: design for `bd-01KWJ1Z8FF1Y8VHA8350JT9GSQ`.

This design covers the next fixed-design refactor after the current streamed
cross-product backend. The goal is to make high-cardinality fixed-effect
designs stay sparse/streamed for fitting while preserving the existing dense
path for ordinary small models.

## Current Shape

`FixedDesign::Streamed` already avoids materializing the active fixed-effect
design for several core products:

- `xtx()`
- `xty(y)`
- `xt_reterm(re)`
- `row_dot_beta(row, beta)`

The scalar LMM constructor still keeps three persistent dense copies:

- `FeTerm.x`: the pivoted fixed-effect design
- `FeMat.xy`: `[X | y]`
- `FeMat.wtxy`: weighted `[X | y]`

For streamed designs this undercuts the memory contract. At `n = 1_000_000`
and moderate `p`, those copies are tens to hundreds of MB even though the
solver blocks can be built from streamed cross-products.

## Target Boundary

Split the scalar fit path into two concepts:

1. **Fixed design basis**

   A rank-aware view of `X` with column names, pivot/rank metadata, storage
   summary, and row-dot support. It may be dense or streamed.

2. **Response RHS**

   Response-specific products: `X'y`, `Z'y`, weighted `y`, and `y'y`. These are
   computed lazily from the fixed design, random terms, weights, and the current
   response vector.

The solver should build `[Z X]' [Z X]` once from the fixed design and random
terms, then attach the response RHS separately. The scalar response case is
then the one-column form of the same split needed by
`docs/multivariate_shared_theta.md`.

## Proposed Types

Introduce internal types before changing public APIs:

```rust
pub(crate) enum FixedDesignBasis {
    Dense(FeTerm),
    Streamed {
        design: FixedDesign,
        rank: usize,
        piv: Vec<usize>,
        cnames: Vec<String>,
    },
}

pub(crate) struct ResponseRhs {
    weighted_y: DVector<f64>,
    xty: DVector<f64>,
    yty: f64,
    zty: Vec<DMatrix<f64>>,
}
```

`FixedDesignBasis` owns rank/pivot metadata without requiring every backend to
own `n x p` dense storage. Dense basis can keep using `FeTerm` directly.
Streamed basis keeps the selected `FixedDesign` plus rank metadata and only
materializes dense `X` when an API truly requires it.

`ResponseRhs` is cheap relative to `X`: it is `O(p + q_re)` rather than
`O(n * p)`.

## Migration Phases

### Phase 1: Rename the contract without changing behavior

- Introduce `FixedDesignBasis` as a wrapper around the existing `FeTerm`.
- Keep `LinearMixedModel.feterm` and `xy_mat` available.
- Add tests proving `FixedDesignBasis::Dense` produces the same `xtx`, `xty`,
  row dot, rank, and column names as the current path.

This is a low-risk bridge for later patches.

### Phase 2: Stop using `FeMat` for solver blocks

- Build FE/response solver blocks from `FixedDesign` plus `ResponseRhs`.
- Keep `FeMat` only as a compatibility cache for methods that still read
  `xy_mat`.
- Move block construction toward:
  - `xx = fixed_design.xtx()`
  - `xy = fixed_design.xty(y)`
  - `yy = y'y`
  - `xz = fixed_design.xt_reterm(re)`
  - `yz = z'y`

Acceptance: current dense tests stay byte-identical; streamed tests match dense
fits without requiring `FeMat.xy` or `FeMat.wtxy` in block construction.

### Phase 3: Make dense materialization lazy

- Replace persistent `FeTerm.x` for streamed designs with `FixedDesignBasis`.
- Add an explicit method such as `materialize_fixed_design()` for APIs that need
  a dense matrix for inference or compatibility.
- Keep ordinary small designs dense by default; do not change the default user
  experience for common models.

Acceptance: high-cardinality streamed constructor no longer stores an `n x p`
matrix in `FeTerm.x`, and no `FeMat.xy`/`FeMat.wtxy` dense copies are created
unless a dense compatibility method is called.

### Phase 4: Share with multivariate RHS

- Generalize `ResponseRhs` to `ResponseBatchRhs`.
- Store `X'Y`, per-term `Z'Y`, and `diag(Y'Y)`.
- Reuse the shared `[Z X]` factorization and solve multiple RHS columns at
  fixed theta.

This is the bridge to `docs/multivariate_shared_theta.md`; multivariate
response support should not reintroduce `[X | y]` as the primary structure.

## API Compatibility

Public methods that currently expose dense quantities should stay available:

- `fixed_design_matrix()` may materialize on demand.
- fixed-effect inference may request a dense basis explicitly until its own
  streamed contrast path exists.
- prediction should continue using `FixedDesign::row_dot_beta`.

The key rule is that compatibility methods may materialize dense data, but the
fit constructor and optimizer path should not keep dense `n x p` copies for a
streamed design.

## Test Gates

- Dense scalar fits remain unchanged against current tests.
- Streamed-vs-dense fit parity for high-cardinality fixed effects.
- Weighted streamed-vs-dense parity.
- `fixed_design_backend_summary()` still reports streamed storage.
- A memory-shape regression test for a high-cardinality streamed design:
  persistent storage must scale with active entries and `p`, not `n * p`.
- Multivariate fixed-theta differential tests later compare batched RHS results
  against fitting scalar responses one column at a time.

## Non-Goals

- Do not remove dense support.
- Do not make formula parsing multivariate in this refactor.
- Do not rewrite fixed-effect inference in the first slice.
- Do not hand-roll a sparse linear algebra backend; keep this at the
  cross-product/RHS ownership boundary.
