#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/bench_report.sh [--out-dir=DIR] [--with-optimizer-harness] [--with-parity] [--skip-gate]

Run the repeatable benchmark subset for a branch and emit a summary report
with command provenance. Opt-in and side-effect free: results go to the
output directory (default benchmarks/local_report/, gitignored) and no
comparison artifacts or checked-in baselines are mutated.

What runs by default:
  1. perf_gate (compare mode)         — deterministic regression gate against
                                        benchmarks/perf_baseline.json
  2. objective_eval_bench             — per-evaluation timing CSV
  3. bench_response_matrix_batch      — batch amortization CSV

Options:
  --out-dir=DIR              Output directory (default benchmarks/local_report)
  --with-optimizer-harness   Also run optimizer_bench_harness (slower)
  --with-parity              Also run the external parity gates. Requires
                             Julia + MixedModels.jl and R + lme4; each gate
                             reports SKIPPED (127) when its engine is missing.
                             Equivalent manual commands:
                               bash scripts/check_julia_parity_fixtures.sh
                               bash scripts/check_release_lme4_parity.sh
  --skip-gate                Skip the perf_gate step (e.g. before a baseline
                             has been generated on this machine)
USAGE
}

out_dir="benchmarks/local_report"
with_optimizer=0
with_parity=0
skip_gate=0
for arg in "$@"; do
  case "$arg" in
    --out-dir=*) out_dir="${arg#--out-dir=}" ;;
    --with-optimizer-harness) with_optimizer=1 ;;
    --with-parity) with_parity=1 ;;
    --skip-gate) skip_gate=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
mkdir -p "$out_dir"

report="$out_dir/REPORT.md"
status_overall="ok"

step() {
  echo
  echo "== $*"
}

{
  echo "# Benchmark report"
  echo
  echo "| provenance | value |"
  echo "| --- | --- |"
  echo "| date | $(date -u '+%Y-%m-%dT%H:%M:%SZ') |"
  echo "| commit | $(git rev-parse HEAD 2>/dev/null || echo unknown)$(git diff --quiet 2>/dev/null || echo ' (dirty)') |"
  echo "| branch | $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown) |"
  echo "| rustc | $(rustc --version) |"
  echo "| host | $(uname -sm) |"
  echo
} > "$report"

if [[ "$skip_gate" -eq 0 ]]; then
  step "perf_gate (deterministic regression gate)"
  gate_log="$out_dir/perf_gate.txt"
  gate_status="PASS"
  if ! cargo run --release --features unstable-internals --example perf_gate >"$gate_log" 2>&1; then
    gate_status="FAIL"
    status_overall="failed"
  fi
  tail -n 20 "$gate_log"
  {
    echo "## perf_gate: $gate_status"
    echo
    echo '```'
    grep -E "^(FAIL|TIME|perf gate)" "$gate_log" || tail -n 3 "$gate_log"
    echo '```'
    echo
  } >> "$report"
fi

step "objective_eval_bench"
cargo run --release --features unstable-internals --example objective_eval_bench \
  > "$out_dir/objective_eval_bench.csv"
{
  echo "## objective_eval_bench (median us/eval)"
  echo
  echo '```'
  cut -d, -f2,20,24 "$out_dir/objective_eval_bench.csv" | column -s, -t
  echo '```'
  echo
} >> "$report"

step "bench_response_matrix_batch"
cargo run --release --features unstable-internals --example bench_response_matrix_batch \
  > "$out_dir/response_matrix_batch.csv"
{
  echo "## bench_response_matrix_batch (per-response ms)"
  echo
  echo '```'
  cut -d, -f2,4,5,7 "$out_dir/response_matrix_batch.csv" | column -s, -t
  echo '```'
  echo
} >> "$report"

if [[ "$with_optimizer" -eq 1 ]]; then
  step "optimizer_bench_harness"
  cargo run --release --example optimizer_bench_harness \
    > "$out_dir/optimizer_bench_harness.csv"
  {
    echo "## optimizer_bench_harness"
    echo
    echo '```'
    head -n 1 "$out_dir/optimizer_bench_harness.csv"
    tail -n +2 "$out_dir/optimizer_bench_harness.csv" | head -n 20
    echo '```'
    echo
  } >> "$report"
fi

if [[ "$with_parity" -eq 1 ]]; then
  step "external parity gates"
  for gate in check_julia_parity_fixtures check_release_lme4_parity; do
    gate_log="$out_dir/$gate.txt"
    gate_status="PASS"
    gate_rc=0
    bash "scripts/$gate.sh" >"$gate_log" 2>&1 || gate_rc=$?
    if [[ "$gate_rc" -eq 127 ]]; then
      gate_status="SKIPPED (engine missing)"
    elif [[ "$gate_rc" -ne 0 ]]; then
      gate_status="FAIL"
      status_overall="failed"
    fi
    echo "$gate: $gate_status"
    {
      echo "## $gate: $gate_status"
      echo
      echo '```'
      tail -n 5 "$gate_log"
      echo '```'
      echo
    } >> "$report"
  done
fi

echo "overall: $status_overall" >> "$report"

step "report written to $report"
if [[ "$status_overall" != "ok" ]]; then
  exit 1
fi
