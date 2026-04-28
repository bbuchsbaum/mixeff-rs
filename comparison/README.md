# Cross-implementation comparison harness

Three-stage workflow producing `comparison/REPORT.md` — accuracy + performance
tables for every `[[fits]]` block in `datasets/*/meta.toml`, side-by-side
against R `lme4`.

```bash
cargo run --release --example compare_rust   # writes manifest.json + rust_results.json
Rscript scripts/compare_lme4.R               # reads manifest.json, writes lme4_results.json
cargo run --release --example compare_report # joins both, writes REPORT.md
```

Each stage is independent — re-run only the side that changed. The R driver
honours `[[columns]].levels` so factor encoding matches the canonical order
recorded in `meta.toml`.

## Files

| File                    | Producer                          | Consumed by                          |
| ----------------------- | --------------------------------- | ------------------------------------ |
| `manifest.json`         | `examples/compare_rust.rs`        | `scripts/compare_lme4.R`             |
| `rust_results.json`     | `examples/compare_rust.rs`        | `examples/compare_report.rs`         |
| `lme4_results.json`     | `scripts/compare_lme4.R`          | `examples/compare_report.rs`         |
| `REPORT.md`             | `examples/compare_report.rs`      | humans                               |

## Schema

Each entry in `*_results.json::results[]`:

```json
{
  "dataset": "sleepstudy",
  "formula": "Reaction ~ 1 + Days + (1 + Days | Subject)",
  "family": "Gaussian", "link": "Identity", "estimator": "REML",
  "n_obs": 180,
  "status": "ok | unsupported | error | not_implemented | skipped",
  "error": null,
  "beta": [251.405, 10.467],
  "coef_names": ["(Intercept)", "Days"],
  "sigma": 25.5918,
  "theta": [0.929, 0.018, 0.222],
  "objective": 1743.628,
  "loglik": -871.814, "aic": 1755.628, "bic": 1774.787,
  "is_singular": false,
  "fit_time_ms": 0.13,        // cold (first repeat)
  "fit_time_ms_min": 0.10,    // best of 3 repeats
  "fit_time_ms_repeats": 3
}
```

`status` semantics:

- `ok` — fit succeeded; all numeric fields populated.
- `unsupported` — a known feature gap (e.g. categorical predictor in a
  random-slope; reported with the underlying error). Rust-side only.
- `error` — fit raised an unexpected error.
- `not_implemented` — fit family/link not yet wired into the driver
  (currently: GLMMs on the Rust side).
- `skipped` — the manifest was skipped before fitting (rare).

## Tolerances

`compare_report.rs` declares them at the top of `main`:

| Metric    | Absolute | Relative |
| --------- | -------- | -------- |
| objective | 1e-2     | 1e-5     |
| β (max Δ) | 1e-3     | 1e-5     |
| σ         | 1e-3     | 1e-4     |

A pass means `Δ ≤ abs_tol` **OR** `Δ / |reference| ≤ rel_tol`. Tighten or
loosen these if you're comparing against a different optimizer baseline.

## Known limitations (current Rust side)

- **`*` operator does not expand interactions**: `a*b` produces `a + b`
  instead of `a + b + a:b`. Surfaced by `cake`, `oats`, `orthodont` (all
  show ❌ in the accuracy table).
- **Categorical predictors in random slopes** are rejected with
  `not found or not numeric`. Surfaced by `kb07` maximal/zerocorr fits and
  `machines :: (Machine | Worker)`.
- **GLMMs** (cbpp, verbagg, grouseticks, tungara_single_caller) are emitted to the manifest but
  marked `not_implemented` until the driver wires up the
  `GeneralizedLinearMixedModel` path.
- **Coefficient-name formatting**: Rust uses `"Type: T2"` (space-colon),
  R uses `"TypeT2"`. Cosmetic; doesn't affect numerical match.

## Adding a new dataset

1. Drop a `data.csv` + `meta.toml` under `datasets/<name>/` (see
   `datasets/REGISTRY.md`).
2. Re-run the three stages. The harness picks up the new dataset
   automatically.

If the dataset's recommended fit is currently in `unsupported` /
`not_implemented` territory, the report will show it as a gap row rather
than failing the run — fix the underlying capability first.
