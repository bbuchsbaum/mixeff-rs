#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/check_julia_parity_fixtures.sh [--accept] [--keep-temp]

Regenerate Julia-backed parity fixtures into a temporary directory and compare
them against checked-in JSON. This is an explicit parity gate for local release
checks or CI jobs where Julia/MixedModels.jl is available.

Options:
  --accept     Copy regenerated fixtures over checked-in fixtures after a clean generation.
  --keep-temp  Print and keep the temporary output directory for inspection.
USAGE
}

accept=0
keep_temp=0
for arg in "$@"; do
  case "$arg" in
    --accept)
      accept=1
      ;;
    --keep-temp)
      keep_temp=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      usage >&2
      exit 2
      ;;
  esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v julia >/dev/null 2>&1; then
  echo "julia is required for the parity fixture drift gate" >&2
  exit 127
fi

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/mixedmodels-julia-parity.XXXXXX")"
if [[ "$keep_temp" -eq 0 ]]; then
  trap 'rm -rf "$tmp_dir"' EXIT
else
  echo "keeping regenerated fixtures in $tmp_dir"
fi

echo "regenerating Julia parity fixtures into $tmp_dir"
julia scripts/regenerate_julia_parity_fixtures.jl --out-dir="$tmp_dir"

mkdir -p "$tmp_dir/tests/fixtures/pathology_corpus/easy_full_rank/parity"
mkdir -p "$tmp_dir/tests/fixtures/pathology_corpus/reduced_rank_unit_correlation/parity"
julia scripts/parity_pathologies.jl \
  --fixture=tests/fixtures/pathology_corpus/easy.toml \
  --out="$tmp_dir/tests/fixtures/pathology_corpus/easy_full_rank/parity/mmjl.json"
julia scripts/parity_pathologies.jl \
  --fixture=tests/fixtures/pathology_corpus/reduced_rank.toml \
  --out="$tmp_dir/tests/fixtures/pathology_corpus/reduced_rank_unit_correlation/parity/mmjl.json"

fixtures=(
  tests/fixtures/parity/cbpp_agq5.json
  tests/fixtures/parity/kb07_ranef.json
  tests/fixtures/parity/parmap_vsize3.json
  tests/fixtures/parity/rank_deficient_metrics.json
  tests/fixtures/parity/gamma_glmm_engines.json
  tests/fixtures/parity/glmm_fast_oracles.json
  tests/fixtures/pathology_corpus/easy_full_rank/parity/mmjl.json
  tests/fixtures/pathology_corpus/reduced_rank_unit_correlation/parity/mmjl.json
)

if [[ "$accept" -eq 1 ]]; then
  for fixture in "${fixtures[@]}"; do
    install -m 0644 "$tmp_dir/$fixture" "$fixture"
  done
  echo "accepted regenerated Julia parity fixtures"
  exit 0
fi

for fixture in "${fixtures[@]}"; do
  echo "checking $fixture"
  if [[ "$fixture" == tests/fixtures/pathology_corpus/* ]]; then
    python scripts/compare_json_tolerant.py --abs-tol=1e-7 --rel-tol=1e-8 --ignore=/runtime_ms "$fixture" "$tmp_dir/$fixture"
  elif [[ "$fixture" == tests/fixtures/parity/glmm_fast_oracles.json ]]; then
    # generated_at is the regeneration date; optimizer feval counts are
    # informational and can shift across platforms/BLAS without any
    # numerical drift in the oracle values themselves.
    python scripts/compare_json_tolerant.py --abs-tol=1e-7 --rel-tol=1e-8 \
      --ignore=/generated_at \
      --ignore=/rows/0/optimizer_fevals \
      --ignore=/rows/1/optimizer_fevals \
      --ignore=/rows/2/optimizer_fevals \
      --ignore=/rows/3/optimizer_fevals \
      "$fixture" "$tmp_dir/$fixture"
  else
    python scripts/compare_json_tolerant.py --abs-tol=1e-7 --rel-tol=1e-8 "$fixture" "$tmp_dir/$fixture"
  fi
done

echo "Julia parity fixtures match checked-in references"
