"""Summarize benchmark JSON lines into a Markdown table.

Usage: python3 summarize.py results/native.jsonl results/wasm32-node.jsonl

Certification estimate: wall time for 1,000 replicate analyses of one design
= 1,000 x mean per-replicate time (mean, not median, because totals add up
means), divided by the worker count under ideal scaling. "fit" rows are a
fresh construction + REML fit per replicate; "refit" rows are the
compile-once / warm-refit path; "+satt" adds the Satterthwaite table cost.
"""

import json
import statistics
import sys
from collections import defaultdict


def pct(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


rows = defaultdict(list)
for path in sys.argv[1:]:
    with open(path) as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                key = (r["platform"], r["mode"], r["n_participants"], r["n_items"])
                rows[key].append(r)

HEADER = (
    "| platform | mode | design | n | median ms | p90 ms | mean ms | "
    "Satt median ms | Satt df available | median Satt df | singular | converged | "
    "1k reps, 1 worker | 1k reps, 8 workers |"
)
print(HEADER)
print("|" + "---|" * (HEADER.count("|") - 1))
for key in sorted(rows, key=lambda k: (k[0], k[1], k[2])):
    platform, mode, n_p, n_i = key
    rs = rows[key]
    t = [r["fit_ms"] for r in rs]
    satt = [r["satterthwaite_ms"] for r in rs if r["satterthwaite_ms"] is not None]
    # NaN serializes as null, so a missing df means Satterthwaite was refused.
    df = [r["satterthwaite_df"] for r in rs if r.get("satterthwaite_df") is not None]
    sing = sum(r["singular"] for r in rs) / len(rs)
    conv = sum(r["converged"] for r in rs) / len(rs)
    mean = statistics.fmean(t)
    # Per-replicate cost: the fit plus, for fresh fits, the Satterthwaite table.
    per_rep_ms = mean + (statistics.fmean(satt) if satt else 0.0)
    total_s = 1000 * per_rep_ms / 1000.0
    label = mode + ("+satt" if satt else "")
    satt_med = f"{statistics.median(satt):.2f}" if satt else "-"
    satt_avail = f"{len(df) / len(rs):.0%}" if satt else "-"
    df_med = f"{statistics.median(df):.1f}" if df else "-"
    print(
        f"| {platform} | {label} | {n_p}x{n_i} | {len(rs)} | "
        f"{statistics.median(t):.1f} | {pct(t, 0.9):.1f} | {mean:.1f} | "
        f"{satt_med} | {satt_avail} | {df_med} | {sing:.0%} | {conv:.0%} | "
        f"{total_s:.1f} s | {total_s / 8:.1f} s |"
    )
