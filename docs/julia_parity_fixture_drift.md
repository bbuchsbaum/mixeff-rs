# Julia Parity Fixture Drift Gate

The checked-in parity fixtures under `tests/fixtures/parity/` and the
MixedModels.jl pathology references are generated reference data, not ordinary
Rust snapshots. Run the drift gate when changing fixture-generation logic,
upgrading Julia/MixedModels.jl, or preparing a parity-sensitive release:

```sh
scripts/check_julia_parity_fixtures.sh
```

The gate regenerates Julia-backed fixtures into a temporary directory and
compares them against the checked-in JSON with tight numeric tolerances. Runtime
fields in pathology references are ignored; schema, source/version strings,
model status, dimensions, coefficients, objectives, and random-effect payloads
must match.

If the drift is intentional, inspect the reported differences and then accept
the regenerated fixtures explicitly:

```sh
scripts/check_julia_parity_fixtures.sh --accept
```

This script is intentionally separate from the default Cargo test suite because
it requires a working Julia environment with MixedModels.jl, DataFrames, and
their transitive dependencies available.
