# Test fixtures

JSON golden files used by integration tests in `tests/*.rs`. Every checked-in
golden must have a sibling `<stem>.provenance.json` recording how it was
produced. The hygiene test
`fixture_hygiene::every_golden_has_provenance_sibling` enforces this.

## Layout

```
tests/fixtures/
├── compiler_contract/        # compiler artifacts, audit reports, inference tables
│   ├── <name>.json           # the golden being asserted against
│   └── <name>.provenance.json # auto-managed regeneration metadata
├── parity/                   # numerical parity vs MixedModels.jl / lme4
│   ├── <name>.json
│   └── <name>.provenance.json
└── pathology_corpus/         # generator specs (TOML, not JSON — different contract)
```

The `pathology_corpus/` files use a different format (TOML strata + Rust-resident
generator specs) and have their own contract version (`v0.3` at time of writing).
They are *not* covered by the provenance hygiene test.

## Provenance schema

```json
{
  "schema_version": "1.0",
  "generated_at": "2026-05-02T19:30:00Z",
  "crate_commit": "<git sha at regeneration time>",
  "regenerator": "<command that refreshes this golden>",
  "source_case": {                       // optional; null for non-parity goldens
    "dataset": "sleepstudy",
    "formula": "Reaction ~ 1 + Days + (1 + Days | Subject)",
    "estimator": "REML"
  },
  "reference_engine": "lme4 1.1-35.5",   // optional; null when no external engine
  "notes": "free-form"                   // optional
}
```

`schema_version` is required. Other fields may be `null` when the original
generation context is not recoverable (for example, the Phase 4 backfill).

## Update env var

To regenerate goldens, set `MIXEDMODELS_UPDATE_FIXTURES=1` and run the test
that owns the golden. The legacy `MIXEDMODELS_UPDATE_WIRE_FIXTURES` is still
recognized by tests that haven't migrated yet — both names are accepted.

```bash
MIXEDMODELS_UPDATE_FIXTURES=1 cargo test --test compiler_contract_snapshots
```

When new goldens land, the regenerator should also write the sibling
`<stem>.provenance.json` with all fields populated. The Phase 4 backfill
(`examples/backfill_fixture_provenance`) only fills `schema_version`,
`generated_at`, `crate_commit`, and `regenerator`; future regenerator
upgrades will fill `source_case` and `reference_engine`.

## Adding a new golden

1. Author the test that asserts against it.
2. Run with the update env var set so the test writes the JSON.
3. Manually write or update `<stem>.provenance.json`. Minimum:
   `schema_version`, `generated_at`, `regenerator`. Add `source_case` and
   `reference_engine` when the golden encodes a fit against an external engine.
4. Verify `cargo test --test fixture_hygiene` passes.

## Inline test data — when not to use a fixture

Integration tests under `tests/*.rs` may build a `DataFrame` inline as long
as the data is **toy**: small (≤30 rows), bespoke, deterministic, and
intended to exercise a specific compiler/inference path rather than to
benchmark numerics. Those tests should mark the builder function with a
short justifying comment, e.g.:

```rust
// toy: 5 subjects × 4 items, parameterized to match the parmap structure
// asserted by `fixtures/parity/parmap_vsize3.json`. Promoting to a
// vendored fixture would only obscure the structural recipe.
fn parmap_vsize3_data() -> DataFrame { … }
```

Promote inline data to `datasets/<name>/` only when:

- the same shape is built in two test files with minor variations (deduplication win), **or**
- it represents a real research design that would be useful to the wider catalog (parameterized tests, benches), **or**
- a downstream R/parity workflow needs to consume the same data via a CSV.

For everything else, inline is the right choice — the data construction
*is* the documentation. Keeping it in the test file (rather than a CSV in
`datasets/`) preserves the recipe (`subject_effects = [-1.0, 0.5, ...]`)
in plain sight where the assertions can reference it.
