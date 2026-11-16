#!/usr/bin/env bash
# Reproduce the crossed-LMM benchmark (native release + wasm32 under Node).
# Requires: rustup target wasm32-unknown-unknown, wasm-bindgen CLI 0.2.126
# (matching the harness's pinned wasm-bindgen), node.
# Usage: bash benchmarks/wasm-crossed/run.sh [reps]   (default 30)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
reps="${1:-30}"
target="${CARGO_TARGET_DIR:-$here/harness/target}"
export CARGO_TARGET_DIR="$target"
manifest="$here/harness/Cargo.toml"
mkdir -p "$here/results"

cargo build --release --manifest-path "$manifest" --bin native
"$target/release/native" "$reps" > "$here/results/native.jsonl"

cargo build --release --manifest-path "$manifest" --lib --target wasm32-unknown-unknown
wasm-bindgen --target nodejs --out-dir "$target/pkg" \
  "$target/wasm32-unknown-unknown/release/mixeff_wasm_crossed_bench.wasm"
node "$here/run_node.mjs" "$target/pkg" "$reps" > "$here/results/wasm32-node.jsonl"

python3 "$here/summarize.py" "$here/results/native.jsonl" "$here/results/wasm32-node.jsonl" \
  > "$here/results/summary.md"
cat "$here/results/summary.md"
