#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Missing external engines are a skipped diagnostic, not a false failure:
# say exactly what is absent and exit with the conventional 127.
if ! command -v Rscript >/dev/null 2>&1; then
  echo "SKIPPED: Rscript not found; the lme4 parity refresh needs R with lme4 + jsonlite installed" >&2
  exit 127
fi
if ! Rscript -e 'quit(status = as.integer(!requireNamespace("lme4", quietly = TRUE)))' >/dev/null 2>&1; then
  echo "SKIPPED: R is present but the lme4 package is not installed" >&2
  exit 127
fi

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
