#!/usr/bin/env python3
"""Behavioural drift check for the reduced-rank MixedModels.jl pathology fixture.

`tests/fixtures/pathology_corpus/reduced_rank_unit_correlation/parity/mmjl.json`
is fitted to exactly rank-1 data with zero residual noise. Its REML objective is
unbounded below along theta[0] = theta[1] -> inf, sigma -> 0, so MixedModels.jl
stops at an arbitrary, platform-dependent point (theta ~ 3.5e3 on macOS,
7e3-8.2e3 on Linux; objective -917 to -1017). There is no reproducible optimum
to compare digits against (VERSIONING.md section 3.1, named exception).

Instead this checker compares the identity fields exactly and verifies that both
the checked-in and the regenerated fixture exhibit the pathology signature:

* theta[0], theta[1] > 1e3 (escape along the unbounded direction),
* theta[0] and theta[1] agree to 1e-4 relative (unit correlation),
* sigma < 1e-3 (residual scale collapse),
* objective < -500 and |objective - lme4 objective| > 1 (consistent with
  tests/cross_engine_scoreboard.rs, which records the engines' disagreement),
* objective == -2 * loglik (1e-8 relative), all numbers finite.

theta/beta/sigma/objective/loglik digits are deliberately not compared.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

EXACT_FIELDS = (
    "schema_version",
    "fixture",
    "stratum",
    "engine",
    "version",
    "source",
    "status",
    "warnings",
    "converged",
    "parity_check",
    "parity_note",
)

THETA_ESCAPE_MIN = 1e3
THETA_UNIT_CORR_REL = 1e-4
SIGMA_COLLAPSE_MAX = 1e-3
OBJECTIVE_MAX = -500.0
LME4_OBJECTIVE_MIN_GAP = 1.0


def signature_errors(label: str, doc: dict, lme4_objective: float) -> list[str]:
    errors: list[str] = []

    def err(message: str) -> None:
        errors.append(f"{label}: {message}")

    if doc.get("status") != "ok" or doc.get("converged") is not True:
        err(
            f"expected status ok / converged true, got {doc.get('status')!r} / {doc.get('converged')!r}"
        )
        return errors
    theta = doc.get("theta") or []
    numbers = [
        *theta,
        *(doc.get("beta") or []),
        doc.get("sigma"),
        doc.get("objective"),
        doc.get("loglik"),
    ]
    if len(theta) != 3 or len(doc.get("beta") or []) != 2:
        err(
            f"expected 3 theta and 2 beta values, got {len(theta)} and {len(doc.get('beta') or [])}"
        )
        return errors
    if not all(
        isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
        for v in numbers
    ):
        err("non-finite or missing numeric field")
        return errors

    t0, t1 = theta[0], theta[1]
    if not (t0 > THETA_ESCAPE_MIN and t1 > THETA_ESCAPE_MIN):
        err(
            f"theta[0:2]={t0!r},{t1!r} not beyond {THETA_ESCAPE_MIN:g} (no escape along the unbounded direction)"
        )
    elif abs(t0 - t1) > THETA_UNIT_CORR_REL * max(abs(t0), abs(t1)):
        err(
            f"theta[0]={t0!r} and theta[1]={t1!r} differ by more than {THETA_UNIT_CORR_REL:g} relative"
        )
    if not doc["sigma"] < SIGMA_COLLAPSE_MAX:
        err(
            f"sigma={doc['sigma']!r} not below {SIGMA_COLLAPSE_MAX:g} (no residual collapse)"
        )
    objective = doc["objective"]
    if not objective < OBJECTIVE_MAX:
        err(f"objective={objective!r} not below {OBJECTIVE_MAX:g}")
    if not abs(objective - lme4_objective) > LME4_OBJECTIVE_MIN_GAP:
        err(
            f"objective={objective!r} within {LME4_OBJECTIVE_MIN_GAP:g} of lme4 objective {lme4_objective!r}"
        )
    if not math.isclose(objective, -2.0 * doc["loglik"], rel_tol=1e-8, abs_tol=1e-8):
        err(f"objective={objective!r} != -2*loglik={-2.0 * doc['loglik']!r}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("expected", type=Path, help="checked-in mmjl.json")
    parser.add_argument("actual", type=Path, help="regenerated mmjl.json")
    parser.add_argument(
        "--lme4",
        type=Path,
        required=True,
        help="checked-in lme4.json for the same fixture",
    )
    args = parser.parse_args()

    expected = json.loads(args.expected.read_text(encoding="utf-8"))
    actual = json.loads(args.actual.read_text(encoding="utf-8"))
    lme4 = json.loads(args.lme4.read_text(encoding="utf-8"))
    lme4_objective = lme4.get("objective")
    if not isinstance(lme4_objective, (int, float)) or not math.isfinite(
        lme4_objective
    ):
        print(
            f"pathology signature: lme4 objective missing in {args.lme4}",
            file=sys.stderr,
        )
        return 1

    errors: list[str] = []
    for field in EXACT_FIELDS:
        if field not in expected:
            errors.append(f"checked-in fixture lacks identity field {field!r}")
        elif expected.get(field) != actual.get(field):
            errors.append(
                f"/{field}: expected={expected.get(field)!r} actual={actual.get(field)!r}"
            )
    errors += signature_errors("checked-in", expected, lme4_objective)
    errors += signature_errors("regenerated", actual, lme4_objective)

    if errors:
        print(
            f"pathology signature drift: {args.expected} != {args.actual}",
            file=sys.stderr,
        )
        for line in errors:
            print(f"  - {line}", file=sys.stderr)
        return 1
    print(
        f"pathology signature holds: theta[0]={actual['theta'][0]:.6g}, sigma={actual['sigma']:.3g}, "
        f"objective={actual['objective']:.6g} (lme4 {lme4_objective:.6g}); digits not compared by design"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
