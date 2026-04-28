# Reference Datasets

Small-to-medium mixed-models datasets used as fixtures for tests, parity
checks against `lme4` / `MixedModels.jl`, and benchmarking. Each dataset
ships as a directory under `datasets/<name>/` containing:

- `data.csv` — observations, with factor columns written as their character
  labels (no row names, UTF-8, `\n` line endings).
- `meta.toml` — schema, recommended formula(s), and (where known) reference
  fit values.
- `_levels.txt` *(optional)* — factor level order as recorded by the dump
  script. Useful for cross-checking `meta.toml` after re-dumping; not
  consumed by the loader.

Load from Rust:

```rust
use mixedmodels::datasets::load;
let (df, meta) = load("sleepstudy")?;
```

Re-dump from source (R + lme4/nlme, optionally Julia + MixedModels.jl):

```bash
Rscript scripts/dump_datasets.R              # tier 1
Rscript scripts/dump_datasets.R --tier2      # tier 1 + tier 2
julia --project=MixedModels.jl scripts/dump_julia_datasets.jl   # tier 3 (kb07)
```

Run the lme4 vs Rust comparison harness:

```bash
cargo run --release --example compare_rust   # writes comparison/manifest.json + rust_results.json
Rscript scripts/compare_lme4.R               # writes comparison/lme4_results.json
cargo run --release --example compare_report # writes comparison/REPORT.md
```

See `comparison/README.md` for schema, tolerances, and known gaps.

Override the load path with the `MIXEDMODELS_DATASETS_DIR` env var.

---

## Tier 1 — classics, must work

These are the floor of any mixed-models implementation. If any of them
regress, fix that before anything else.

| Name         | n    | Source            | Structure                           | Difficulty | Notes                         |
| ------------ | ---- | ----------------- | ----------------------------------- | ---------- | ----------------------------- |
| `sleepstudy` |  180 | `lme4::sleepstudy`| random intercept + slope, correlated| easy       | The canonical sanity check    |
| `dyestuff`   |   30 | `lme4::Dyestuff`  | scalar RE                           | easy       | Simplest variance components  |
| `dyestuff2`  |   30 | `lme4::Dyestuff2` | scalar RE                           | **boundary** | σ_b → 0 (singular fit)      |
| `pastes`     |   60 | `lme4::Pastes`    | nested (cask within batch)          | easy       | `sample` is the interaction   |
| `penicillin` |  144 | `lme4::Penicillin`| crossed (plate × sample)            | easy       | Both REs are scalar           |
| `cbpp`       |   56 | `lme4::cbpp`      | scalar RE, binomial GLMM            | moderate   | Smallest GLMM fixture         |

## Tier 2 — variety

Adds split-plot, longitudinal, large-crossed-RE GLMM, and overdispersed
counts. Reference fit values are not yet pinned in `meta.toml` — the loader
just verifies row counts and column types.

| Name         |  n   | Source            | Structure                           | Family     | Difficulty |
| ------------ | ---- | ----------------- | ----------------------------------- | ---------- | ---------- |
| `cake`       |  270 | `lme4::cake`      | split-plot (recipe × temperature)   | gaussian   | easy       |
| `verbagg`    | 7584 | `lme4::VerbAgg`   | crossed (id × item), 316 subjects   | binomial   | moderate   |
| `grouseticks`|  403 | `lme4::grouseticks`| nested + observation-level RE      | poisson    | moderate   |
| `ergostool`  |   36 | `nlme::ergoStool` | scalar RE on subject                | gaussian   | easy       |
| `machines`   |   54 | `nlme::Machines`  | crossed factors + interaction RE    | gaussian   | moderate   |
| `orthodont`  |  108 | `nlme::Orthodont` | longitudinal, age × Sex             | gaussian   | easy       |
| `oats`       |   72 | `nlme::Oats`      | split-plot (Block / Variety)        | gaussian   | easy       |
| `rail`       |   18 | `nlme::Rail`      | scalar RE (1 factor)                | gaussian   | easy       |

## Tier 3 — stress / boundary

Datasets that probe optimizer robustness, large crossed-RE scaling, and
near-singular covariance. Sourced from `MixedModels.jl` rather than `lme4`.

| Name      |  n   | Source                           | Structure                         | Difficulty | Notes |
| --------- | ---- | -------------------------------- | --------------------------------- | ---------- | ----- |
| `kb07`    | 1789 | `MixedModels.jl :kb07`           | crossed (subj × item), 6 fixed effects | **stress** | Kliegl & Bates (2007); maximal RE model often singular |
| `singular`|  150 | Cross Validated / GitHub mirror  | 8-D random coefficient covariance | **boundary** | Maximal model is singular without obvious VarCorr symptoms; rePCA/effective-rank story |
| `tungara_single_caller`| 2955 | Dryad doi:10.5061/dryad.3n5tb2rrz | binomial GLMM with cell-level random slope | **stress** | Public fallback found while investigating lme4 GH720; exact GH720 data is unavailable |

### Future additions (not yet vendored)

- **InstEval** (`MixedModels.jl :insteval`, ~73k rows) — large crossed-RE
  scaling fixture. Too big to commit; fetch lazily and store outside the
  repo, or subset.
- **Contraception** (`MixedModels.jl :contra`) — Bangladesh contraceptive-use
  data, hierarchical binomial GLMM.
- **Oxide** (`MixedModels.jl :oxide`) — semiconductor variance components,
  highly nested (lot/wafer/site).
- **mrk17_exp1** (`MixedModels.jl :mrk17_exp1`) — Masson, Rabe & Kliegl (2017)
  lexical-decision data; another optimizer stress fixture.

Add these by extending `scripts/dump_julia_datasets.jl` and writing a
matching `meta.toml`.

---

## `meta.toml` schema

```toml
name        = "<short id>"          # must match directory name
source      = "<package>::<obj>"    # e.g. "lme4::sleepstudy"
license     = "<spdx or note>"      # optional
n_rows      = <integer>             # cross-checked by the loader
description = """multi-line prose"""

[[columns]]
name   = "..."
type   = "numeric" | "categorical"
levels = ["...", ...]               # required only if you want canonical order
unit   = "..."                      # optional, free-form

[[fits]]
formula   = "y ~ ..."               # R/lme4-style; one formula per recommended fit
family    = "Gaussian" | "Binomial" | "Poisson" | ...
link      = "Identity" | "Logit"   | "Log"     | ...
estimator = "REML" | "ML" | "Laplace" | "AGQ"
weights   = "<column>"              # optional (binomial trials, etc.)

[fits.expected]                     # optional reference values for parity
beta        = [...]
sigma       = <f64>                 # residual SD
re_sigmas   = [...]                 # one per random-effects σ
re_corr     = <f64>                 # for the standard 2-D random-slope case
theta       = [...]                 # raw θ vector, when pinned
objective   = <f64>                 # -2 logLik (deviance) or -2 REML logLik
is_singular = true                  # set when any RE σ hits zero

[tags]
structure  = ["random_intercept", "random_slope", "nested", "crossed",
              "split_plot", "interaction_re", "observation_level_re",
              "scalar_re", "longitudinal", "balanced", "glmm", ...]
difficulty = "easy" | "moderate" | "boundary" | "stress"
family     = "binomial" | "poisson" | ...   # echoed at top level for filtering
notes      = "..."
```
