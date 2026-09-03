#!/usr/bin/env python3
"""Analyze paired baseline/candidate TrustBQ benchmark CSVs."""
from __future__ import annotations

import csv
import glob
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "ab-results")
AFFECTED = {
    "vector_180",
    "vector_1000",
    "vector_5000",
    "vector_10000",
    "vector_deep_200x50",
    "crossed_small",
    "crossed_medium",
    "crossed_large",
}


def read_runs(pattern: str):
    by_scenario = defaultdict(list)
    for path in sorted(glob.glob(str(ROOT / pattern))):
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["scenario"] in AFFECTED:
                    row["source_file"] = Path(path).name
                    by_scenario[row["scenario"]].append(row)
    return by_scenario


def median_num(rows, key):
    return statistics.median(float(row[key]) for row in rows)


baseline = read_runs("baseline-*.csv")
candidate = read_runs("candidate-*.csv")
if set(baseline) != AFFECTED or set(candidate) != AFFECTED:
    raise SystemExit(
        f"missing scenarios: baseline={sorted(AFFECTED-set(baseline))}, "
        f"candidate={sorted(AFFECTED-set(candidate))}"
    )

summary = []
for scenario in sorted(AFFECTED):
    b = baseline[scenario]
    c = candidate[scenario]
    if len(b) != len(c):
        raise SystemExit(f"unpaired run counts for {scenario}: {len(b)} vs {len(c)}")
    for row in b + c:
        if row["objective_pass"].lower() != "true":
            raise SystemExit(f"objective reference failure: {scenario}: {row}")
    b_eval = median_num(b, "feval_median")
    c_eval = median_num(c, "feval_median")
    b_ms = median_num(b, "median_ms")
    c_ms = median_num(c, "median_ms")
    # Compare candidate and baseline objectives using the repository's own
    # reference tolerance, not equality at floating-point noise level.
    b_obj = median_num(b, "objective")
    c_obj = median_num(c, "objective")
    tol = max(median_num(b, "objective_tolerance"), median_num(c, "objective_tolerance"))
    if abs(c_obj - b_obj) > tol:
        raise SystemExit(
            f"candidate objective drift exceeds tolerance for {scenario}: "
            f"baseline={b_obj}, candidate={c_obj}, tolerance={tol}"
        )
    if c_eval > b_eval:
        raise SystemExit(
            f"candidate increased evaluations for {scenario}: {b_eval} -> {c_eval}"
        )
    summary.append(
        {
            "scenario": scenario,
            "baseline_fevals": b_eval,
            "candidate_fevals": c_eval,
            "feval_reduction_pct": 100.0 * (b_eval - c_eval) / b_eval,
            "baseline_median_ms": b_ms,
            "candidate_median_ms": c_ms,
            "wall_speedup": b_ms / c_ms,
            "baseline_objective": b_obj,
            "candidate_objective": c_obj,
            "objective_delta": c_obj - b_obj,
            "objective_tolerance": tol,
            "paired_runs": len(b),
        }
    )

total_b = sum(row["baseline_fevals"] for row in summary)
total_c = sum(row["candidate_fevals"] for row in summary)
total_reduction = 100.0 * (total_b - total_c) / total_b
if total_reduction < 20.0:
    raise SystemExit(f"aggregate evaluation reduction too small: {total_reduction:.2f}%")

out = {
    "affected_scenarios": len(summary),
    "paired_runs_per_scenario": min(row["paired_runs"] for row in summary),
    "baseline_total_fevals": total_b,
    "candidate_total_fevals": total_c,
    "aggregate_feval_reduction_pct": total_reduction,
    "median_wall_speedup": statistics.median(row["wall_speedup"] for row in summary),
    "all_objectives_equivalent": True,
    "rows": summary,
}
(ROOT / "summary.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

with (ROOT / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)

print(json.dumps(out, indent=2))
