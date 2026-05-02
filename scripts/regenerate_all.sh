#!/usr/bin/env bash
# Re-run every dataset / fixture regenerator in dependency order, in one
# go. Used when bumping lme4 or MixedModels.jl, or when refreshing all
# pinned numbers after a numerical change.
#
# Phase 5 of fixture/dataset unification (mote bd-01KQMZX24V1S8T12HTAWWV95QY).
#
# Usage:
#     bash scripts/regenerate_all.sh                # full regen (CSV + pin)
#     bash scripts/regenerate_all.sh --pin-only     # skip CSV dump, just refit
#     bash scripts/regenerate_all.sh --dry-run      # log what would run
#
# The script tolerates missing R or Julia by skipping those steps with a
# loud warning. Failing partway leaves the working tree in a partially
# regenerated state — `git diff` after the run shows what changed.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." &>/dev/null && pwd)"
cd "$REPO_ROOT"

PIN_ONLY=0
DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --pin-only) PIN_ONLY=1 ;;
        --dry-run)  DRY_RUN=1 ;;
        *) echo "unknown flag: $arg" >&2; exit 2 ;;
    esac
done

run() {
    echo
    echo "==> $*"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        "$@"
    fi
}

skip() {
    echo
    echo "[skip] $1"
}

have() { command -v "$1" &>/dev/null; }

# ---- 1. Tier 1+2 vendored datasets (R / lme4 / nlme) ------------------

if have Rscript; then
    if [[ "$PIN_ONLY" -eq 1 ]]; then
        run Rscript scripts/dump_datasets.R --tier2 --pin-only
    else
        run Rscript scripts/dump_datasets.R --tier2
    fi
else
    skip "Rscript not found — Tier-1+2 dataset regeneration skipped"
fi

# ---- 2. Tier 3 (kb07) and any future MixedModels.jl-only datasets -----

if have julia; then
    if [[ "$PIN_ONLY" -eq 1 ]]; then
        run julia --project=MixedModels.jl scripts/dump_julia_datasets.jl --pin-only
    else
        run julia --project=MixedModels.jl scripts/dump_julia_datasets.jl
    fi
else
    skip "julia not found — kb07 (and future MixedModels.jl datasets) skipped"
fi

# ---- 3. Synthesized / vendored-without-package datasets ---------------
#       (tungara, singular, station_season_duration, nested_constant_response)

if have Rscript; then
    run Rscript scripts/dump_synthesized_datasets.R
else
    skip "Rscript not found — synthesized dataset pinning skipped"
fi

# ---- 4. comparison/manifest.json (derived from datasets/REGISTRY) -----

if have cargo; then
    run cargo run --release --example compare_rust
else
    skip "cargo not found — comparison/manifest.json regeneration skipped"
fi

# ---- 5. Pathology + parity JSON fixtures (R + Julia) ------------------
#       Re-run only the existing scripts. Each writes its own JSON; once
#       Phase 4-followup extends them to also write provenance siblings,
#       this step also handles tests/fixtures/parity/*.provenance.json.

if have Rscript && [[ -f scripts/parity_pathologies.R ]]; then
    run Rscript scripts/parity_pathologies.R
fi
if have julia && [[ -f scripts/parity_pathologies.jl ]]; then
    run julia --project=MixedModels.jl scripts/parity_pathologies.jl
fi
if have julia && [[ -f scripts/regenerate_julia_parity_fixtures.jl ]]; then
    run julia --project=MixedModels.jl scripts/regenerate_julia_parity_fixtures.jl
fi

# ---- 6. Backfill any provenance siblings that are missing -------------

if have cargo; then
    run cargo run --example backfill_fixture_provenance
fi

# ---- 7. Final report --------------------------------------------------

echo
echo "==> regenerate_all.sh complete"
echo
if [[ "$DRY_RUN" -eq 0 ]]; then
    echo "Review changes with: git status && git diff --stat"
    echo "Then run hygiene tests: cargo test --test fixture_hygiene"
fi
