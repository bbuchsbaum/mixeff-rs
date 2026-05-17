#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

tmpdir="$(mktemp -d)"
cleanup() {
  for path in \
    comparison/manifest.json \
    comparison/rust_results.json \
    comparison/lme4_results.json \
    comparison/REPORT.md
  do
    if [[ -f "$tmpdir/$path" ]]; then
      cp "$tmpdir/$path" "$path"
    fi
  done
  rm -rf "$tmpdir"
}
trap cleanup EXIT

for path in \
  comparison/manifest.json \
  comparison/rust_results.json \
  comparison/lme4_results.json \
  comparison/REPORT.md
do
  mkdir -p "$tmpdir/$(dirname "$path")"
  cp "$path" "$tmpdir/$path"
done

cargo run --release --features unstable-internals --example compare_rust
Rscript scripts/compare_lme4.R
cargo run --release --features unstable-internals --example compare_report

cargo test --features unstable-internals --test parity_scorecard
