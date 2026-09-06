#!/usr/bin/env zsh
# Paired A/B benchmark of optimizer_bench_harness: a base commit vs the
# working tree, built into separate binaries, run in alternating order over
# several rounds so machine drift affects both sides equally.
#
# Usage:
#   scripts/bench_paired.sh <base-commit> <tag> [rounds=5] [profile=nlopt|native|glmm] [ENV=VALUE ...]
#
# Output CSVs land in target/bench_paired/<tag>/{base,cand}_r<N>.csv; summarize
# them with scripts/bench_compare.py <tag>.
#
# Profiles: nlopt (default features), native (--no-default-features), glmm
# (--features unstable-internals, dataset-registry GLMM rows; combine with
# MIXEFF_BENCH_SCENARIO=glmm to run only those rows).
set -euo pipefail

if (( $# < 2 )); then
  sed -n '2,15p' "$0"
  exit 2
fi

BASE_COMMIT=$1
TAG=$2
ROUNDS=${3:-5}
PROFILE=${4:-nlopt}
shift $(( $# < 4 ? $# : 4 ))

REPO=$(git rev-parse --show-toplevel)
OUT=$REPO/target/bench_paired/$TAG
BIN=$OUT/bin
WT=$REPO/target/bench_paired/base_wt_$TAG
mkdir -p "$BIN"

case $PROFILE in
  nlopt)  FLAGS=() ;;
  native) FLAGS=(--no-default-features) ;;
  glmm)   FLAGS=(--features unstable-internals) ;;
  *) echo "unknown profile: $PROFILE" >&2; exit 2 ;;
esac

echo "building candidate (working tree, $PROFILE)"
(cd "$REPO" && cargo build --release "${FLAGS[@]}" --example optimizer_bench_harness >/dev/null)
cp "$REPO/target/release/examples/optimizer_bench_harness" "$BIN/harness_cand"

echo "building base ($BASE_COMMIT, $PROFILE)"
git -C "$REPO" worktree add --detach "$WT" "$BASE_COMMIT" >/dev/null
trap 'git -C "$REPO" worktree remove --force "$WT" >/dev/null 2>&1 || true' EXIT
(cd "$WT" && CARGO_TARGET_DIR="$WT/target" cargo build --release "${FLAGS[@]}" --example optimizer_bench_harness >/dev/null)
cp "$WT/target/release/examples/optimizer_bench_harness" "$BIN/harness_base"

for r in $(seq 1 "$ROUNDS"); do
  if (( r % 2 == 1 )); then order=(base cand); else order=(cand base); fi
  for which in "${order[@]}"; do
    env "$@" "$BIN/harness_$which" > "$OUT/${which}_r$r.csv" 2>/dev/null
  done
  echo "round $r/$ROUNDS done"
done
echo "results in $OUT; summarize with: python3 scripts/bench_compare.py $TAG"
