# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A Rust port of Julia's [MixedModels.jl](https://github.com/JuliaStats/MixedModels.jl) for fitting linear and generalized linear mixed-effects models. The `MixedModels.jl/` directory contains the upstream Julia source as the **reference implementation** — when porting algorithms or tracking down numerical discrepancies, read the corresponding Julia source there first. `src/main.rs` is a stub; this is a library crate.

## Common commands

```bash
cargo build                      # debug build
cargo build --release            # release build (use this for benchmarks)
cargo test                       # run all unit tests (tests are inline #[cfg(test)] modules in src/)
cargo test <name>                # run a single test by substring match
cargo test -- --nocapture        # show println! output during tests
cargo clippy --all-targets       # lint
cargo run --release --example bench_rust    # Rust-side benchmark suite
cargo run --release --example parity_dump   # dump fits as JSON for cross-checking with Julia
```

There is no integration-test directory; tests live next to the code in `#[cfg(test)] mod tests` blocks.

## Cross-language parity & benchmarking

This crate is co-developed with the Julia reference. Two paired workflows exist:

- **Benchmarks**: `examples/bench_rust.rs` ↔ `scripts/bench_julia.jl`. Both simulate the same sleepstudy-like data, fit identical formulas (e.g. `reaction ~ 1 + days + (1 + days | subj)`) over scaling scenarios, and emit CSV.
- **Parity dumps**: `examples/parity_dump.rs` ↔ `scripts/parity_dump_julia.jl`. Both serialize a fitted model's θ, β, σ, objective, and Cholesky blocks to JSON for diffing.

When you change anything that could move numerical output (objective, θ ordering, factor block layout, optimizer choice), regenerate both dumps and compare. The Julia parity script needs `MixedModels`, `DataFrames`, `JSON3` available.

## Architecture

The crate is a layered tower; lower layers know nothing about upper ones.

```
formula  →  model  →  stats         (high-level API)
              ↑
            types  →  linalg        (numerical core)
              ↑
            error
```

### `linalg/` — numerical primitives
Building blocks for the blocked Cholesky update used by the PLS step:
- `chol_unblocked` (rank-revealing scalar Cholesky), `pivot` (pivoted QR for fixed-effects rank), `rank_update` (downdate/update of L blocks), `block_ops` (typed gemm/trsm over `MatrixBlock`s), `logdet` (det from L).

### `types/` — typed model-matrix containers
- `FeTerm`, `FeMat` — fixed-effects design with rank/pivot info; stores `[X | y]` so the joint blocked system can be built in one pass.
- `ReMat` — random-effects design for one grouping factor (the per-term Λ_θ, Z, and refs).
- `UniformBlockDiagonal`, `BlockedSparse`, `RaggedArray` — specialized storage for the regular structure of grouped random effects.
- `OptSummary`, `FitLogEntry`, `Optimizer` — optimization state, tolerances, fit log.
- `GaussHermiteNormalized`, `gh_norm` — quadrature nodes for adaptive Gauss-Hermite (GLMM AGQ).

### `formula/` — formula AST
- `parser.rs` is a recursive-descent parser for R/lme4 syntax. Supported: `*`, `:`, `/`, `(re | g)`, `(re || g)` zero-correlation, `(re | g1 & g2)` interactions, explicit `0 +`/`-1`/`1 +` intercept handling.
- `terms.rs` defines the AST (`Formula`, `FixedTerm`, `RandomTerm`, `GroupingFactor`).

### `model/` — fit drivers
- `data.rs` — minimal column-oriented `DataFrame` with `Numeric` and `Categorical` columns. Categorical levels are encoded by first-appearance order. Real callers convert from polars/arrow into this.
- `linear.rs` — `LinearMixedModel`: PLS / profiled (RE)ML. Stores blocked `A = [Z X y]'[Z X y]` and its updated lower Cholesky `L` per θ. Multiple optimizer paths chosen automatically:
  - `fit_scalar_single_theta` for one-θ scalar problems
  - `fit_multivariate_pattern_search` for moderate-θ problems
  - `fit_nlopt_large_theta` for large-θ (NLopt), with `fit_cobyla` as the fallback
  Entry point is `LinearMixedModel::fit(reml: bool)`. The optimizer must be chosen by `fit`, not by callers — when adding a new code path, gate it inside `fit` with a `use_*_optimizer()` predicate to match the existing structure.
- `generalized.rs` — `GeneralizedLinearMixedModel`: PIRLS for conditional modes, optional adaptive Gauss-Hermite quadrature. Wraps an internal `LinearMixedModel` for the local Laplace approximation. `Family::Normal + LinkFunction::Identity` is rejected — it must use `LinearMixedModel`.
- `traits.rs` — `MixedModelFit` (the cross-cutting interface for fitted models: `coef`, `vcov`, `fitted`, `aic`/`bic`, `theta`, `ranef`, …), plus `Family` and `LinkFunction` enums.

### `stats/` — post-fit summaries
`varcorr`, `coeftable`, `model_summary`, `block_description`, `lrt`, `bootstrap`, `profile`, `spline` (used by profile CIs).

### `error.rs`
`MixedModelError` is the top-level error; `LinAlgError` is its numerical-layer companion (`Result<T> = std::result::Result<T, MixedModelError>`). `From<LinAlgError>` and `From<FormulaError>` already exist — propagate with `?`, don't stringify.

## Design notes worth reading before non-trivial work

`docs/` contains forward-looking design contracts that constrain how new features should land. Read them before adding inference/diagnostic surfaces or extending to multivariate Y:

- `mixed_model_compiler_inference_contract.md` — the project's stance on inference: no fake p-values, no hidden model surgery, explicit identifiability/refusal paths. Diagnostics live in the Rust crate; the R layer is meant to be a client.
- `multivariate_shared_theta.md` — planned split between the shared `[Z X]` factorization and per-response right-hand sides. Today's `FeMat` bakes `[X | y]` together; multivariate work will need to decouple these.

## Coordination & issue tracking — mote

> **Issue tracking in this repository is `mote` — not `beads`/`bd`.**
> This explicitly overrides any parent or global instruction (for example a
> higher-level `~/.claude/CLAUDE.md` or `/Users/bbuchsbaum/code/CLAUDE.md` that
> mentions `bd`/beads). Do **not** run `bd` in this repo. Note that mote issue
> IDs are written with a `bd-` prefix (e.g. `bd-01KR...`); that prefix belongs
> to mote and does **not** mean the beads CLI.

This repository uses **mote** for local issue tracking and lightweight coordination
between agents. The `.mote/` op log is the source of truth for current work,
claims, reservations, and project memory. See `AGENTS.md` for the full protocol —
this section and `AGENTS.md` are kept deliberately in sync; if you change one,
mirror the change in the other.

**Before editing files** — check health, find or create an issue, and reserve paths:

```bash
mote doctor                                    # health check
mote actor show                                # confirm stable actor name
mote board                                     # current state
mote ready                                     # actionable, unblocked work
```

If `.mote/` is missing, run `mote init` then `mote actor set <stable-name>`
(prefer stable names like `codex-impl`, `codex-tests`, `codex-docs`, or the
human's name — do not invent a fresh actor per turn).

Work from an existing issue when one matches; otherwise:

```bash
mote new "Short task title" -p 1 --tag <area>
```

**Reserve paths before touching them.** Reservations are advisory but the
coordination contract in this repo:

```bash
mote preflight --issue <mote-id> --paths <path> [<path> ...]
mote begin <mote-id> --paths <path> [<path> ...] --note "starting work"
```

If preflight/begin reports a conflict, inspect the owner with
`mote who-has <path>` and coordinate or pick a non-overlapping slice — do not
edit conflicting paths. Keep reservations narrow (exact files for focused work,
directories only when truly needed). If scope grows, run `mote preflight` again
and reserve the added paths before editing.

**During work** — record material decisions, blockers, and progress:

```bash
mote note <mote-id> --kind progress "what changed"
mote note <mote-id> --kind decision "decision and rationale"
mote note <mote-id> --kind blocker "what is blocked"
```

**Finishing** — pick the verb that matches the outcome:

```bash
mote done <mote-id> --note "finished"                              # completed
mote note <mote-id> --kind progress "state and next step"          # pausing:
mote release <mote-id>                                             #   then release
mote handoff <mote-id> --to <actor> --note "..." --release         # handoff
```

**Repository policy:**
- Do not hand-edit `.mote/ops/*.json`; publish changes through the `mote` CLI.
- Keep `.mote/` out of git unless the project explicitly decides to version it.
- When reporting status, cite the mote issue id for active or completed work.

When ending a session: leave issues in a clean state (done/released/handed off),
run `cargo test` + `cargo clippy`.
