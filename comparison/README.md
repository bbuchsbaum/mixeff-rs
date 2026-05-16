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

Supported GLMM families are part of the routine harness. The Rust driver emits
Laplace and supported AGQ rows for Bernoulli, Binomial, Poisson, and Gamma
fits; stress-tier GLMM rows are skipped unless `MIXEDMODELS_INCLUDE_STRESS=1`
is set.

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
  "status": "ok | unsupported | error | not_implemented | skipped | skipped_stress",
  "error": null,
  "beta": [251.405, 10.467],
  "coef_names": ["(Intercept)", "Days"],
  "sigma": 25.5918,
  "theta": [0.929, 0.018, 0.222],
  "objective": 1743.628,
  "loglik": -871.814, "aic": 1755.628, "bic": 1774.787,
  "objective_definition": "restricted_deviance",
  "response_constants": "not_applicable",
  "optimizer": "cobyla",
  "optimizer_backend": "native",
  "optimizer_return_code": "FTOL_REACHED",
  "optimizer_fevals": 37,
  "optimizer_fmin": 1743.628,
  "optimizer_max_fevals": 10000,
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
  (should be rare; the GLMM driver is wired for supported families).
- `skipped` — the manifest was skipped before fitting (rare).
- `skipped_stress` — stress-tier fixture omitted from routine comparison
  regeneration; set `MIXEDMODELS_INCLUDE_STRESS=1` to include it.

`objective_definition` and `response_constants` make GLMM likelihood/deviance
semantics explicit. Rust GLMM rows currently report
`response_constants = "dropped"` for the profiled GLMM objective; lme4 GLMM
rows report `response_constants = "included"` for `-2 * logLik`. The report
marks objective deltas as non-comparable (`n/c`) when those conventions differ
instead of failing the row on a constant-term convention.

Timing fields are executable contract data, not only report decoration.
`fit_time_ms` is the first cold repeat and `fit_time_ms_min` is the best of the
recorded repeats. GLMM speed gates require representative routine rows to have
positive timings, optimizer labels, return codes, function-evaluation counts,
and Rust minimum fit time at least as fast as the corresponding `lme4` minimum
unless the row is explicitly fenced.

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

- **GLMM objective/logLik constants**: Binomial and Poisson engines do not all
  report response-family constants on the same scale yet. The report now
  fences those objective deltas as `n/c`; β/σ diagnostics remain visible.
- **Fixed-only formulas** are rejected by the mixed-model driver. The
  comparison table keeps those rows visible as `error` rather than silently
  treating them as linear models.
- **Stress fixtures** are skipped by default on both sides so routine
  regeneration finishes quickly. Set `MIXEDMODELS_INCLUDE_STRESS=1` to include
  them.
- **Coefficient-name formatting**: Rust uses `"Type: T2"` (space-colon),
  R uses `"TypeT2"`. Cosmetic; doesn't affect numerical match.

## Release checks

Run this loop before publishing GLMM-facing changes:

```bash
cargo run --release --example compare_rust
Rscript scripts/compare_lme4.R
cargo run --release --example compare_report
cargo test --test glmm_comparison_gates
cargo test --test glmm_speed_parity
cargo test --test glmm_artifact_contract
```

For the large crossed Poisson speed row, collect a focused local profile with:

```bash
MIXEDMODELS_PROFILE_REPEATS=100 cargo run --release --example profile_grouseticks_glmm
```

`comparison/REPORT.md` should have no unclassified GLMM numeric disagreements,
and every routine row covered by `tests/glmm_speed_parity.rs` should meet its
speed threshold.

## Adding a new dataset

1. Drop a `data.csv` + `meta.toml` under `datasets/<name>/` (see
   `datasets/REGISTRY.md`).
2. Re-run the three stages. The harness picks up the new dataset
   automatically.

If the dataset's recommended fit is currently in `unsupported` /
`not_implemented` territory, the report will show it as a gap row rather
than failing the run — fix the underlying capability first.
