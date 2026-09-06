#!/usr/bin/env python3
"""Summarize a scripts/bench_paired.sh run.

Usage: scripts/bench_compare.py <tag> [--dir target/bench_paired]

Per scenario: median evaluations, all-in median ms, construction and
post-fit medians (when the harness emitted them), the objective difference
against the paired gate |F_cand - F_base| <= 1e-6 (1 + |F_ref|), the
largest theta difference, and how many rounds the candidate won on wall
time. "pass" requires every round of both sides to pass the harness's own
reference-objective gate and the paired objective gate.
"""
import csv
import glob
import json
import os
import statistics
import sys


def load(directory, which):
    rows = {}
    for path in sorted(glob.glob(os.path.join(directory, f"{which}_r*.csv"))):
        rnd = int(path.rsplit("_r", 1)[1].split(".")[0])
        with open(path) as handle:
            for row in csv.DictReader(handle):
                rows.setdefault(row["scenario"], {})[rnd] = row
    return rows


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    tag = sys.argv[1]
    base_dir = "target/bench_paired"
    if "--dir" in sys.argv:
        base_dir = sys.argv[sys.argv.index("--dir") + 1]
    directory = os.path.join(base_dir, tag)
    base = load(directory, "base")
    cand = load(directory, "cand")
    if not base or not cand:
        sys.exit(f"no paired CSVs under {directory}")

    sample = next(iter(next(iter(base.values())).values()))
    has_phase = "build_ms_median" in sample
    header = (
        f"{'scenario':26s} {'d':>2s} {'B_ev':>6s} {'C_ev':>6s} {'red%':>6s} "
        f"{'B_ms':>9s} {'C_ms':>9s} {'speed':>6s} "
    )
    if has_phase:
        header += f"{'B_build':>8s} {'C_build':>8s} {'bld_x':>6s} {'B_post':>7s} {'C_post':>7s} "
    header += (
        f"{'objD(C-B)':>11s} {'tol':>9s} {'maxdTheta':>9s} {'pass':>5s} {'wins':>5s}"
    )
    print(header)

    tot_b = tot_c = 0.0
    speeds = []
    all_pass = True
    wins_total = pairs_total = 0
    for scenario, rb in base.items():
        if scenario not in cand:
            continue
        rc = cand[scenario]
        rounds = sorted(set(rb) & set(rc))
        if not rounds:
            continue
        med = lambda rows, key: statistics.median(float(rows[r][key]) for r in rounds)
        ev_b, ev_c = med(rb, "feval_median"), med(rc, "feval_median")
        ms_b, ms_c = med(rb, "median_ms"), med(rc, "median_ms")
        ob = statistics.mean(float(rb[r]["objective"]) for r in rounds)
        oc = statistics.mean(float(rc[r]["objective"]) for r in rounds)
        tol = float(rb[rounds[0]]["objective_tolerance"])
        ok = all(
            rb[r]["objective_pass"] == "true" and rc[r]["objective_pass"] == "true"
            for r in rounds
        )
        equiv = abs(oc - ob) <= tol
        wins = sum(
            1 for r in rounds if float(rc[r]["median_ms"]) < float(rb[r]["median_ms"])
        )
        wins_total += wins
        pairs_total += len(rounds)
        tb = json.loads(rb[rounds[0]]["theta"])
        tc = json.loads(rc[rounds[0]]["theta"])
        dtheta = max((abs(a - b) for a, b in zip(tb, tc)), default=0.0)
        tot_b += ev_b
        tot_c += ev_c
        speeds.append(ms_b / ms_c if ms_c > 0 else float("nan"))
        passed = ok and equiv
        all_pass &= passed
        line = (
            f"{scenario:26s} {rb[rounds[0]]['d_theta']:>2s} {ev_b:6.0f} {ev_c:6.0f} "
            f"{100 * (ev_b - ev_c) / ev_b if ev_b else 0:6.2f} {ms_b:9.3f} {ms_c:9.3f} "
            f"{ms_b / ms_c if ms_c else float('nan'):6.3f} "
        )
        if has_phase:
            bb, cb = med(rb, "build_ms_median"), med(rc, "build_ms_median")
            bp, cp = med(rb, "postfit_ms_median"), med(rc, "postfit_ms_median")
            line += f"{bb:8.3f} {cb:8.3f} {bb / cb if cb else float('nan'):6.2f} {bp:7.3f} {cp:7.3f} "
        line += f"{oc - ob:11.2e} {tol:9.2e} {dtheta:9.2e} {str(passed):>5s} {wins}/{len(rounds)}"
        print(line)

    print(
        f"\nTOTAL evals base={tot_b:.0f} cand={tot_c:.0f} "
        f"reduction={100 * (tot_b - tot_c) / tot_b if tot_b else 0:.2f}%  "
        f"median speedup={statistics.median(speeds):.3f}x  "
        f"wall wins={wins_total}/{pairs_total}  all gates pass={all_pass}"
    )


if __name__ == "__main__":
    main()
