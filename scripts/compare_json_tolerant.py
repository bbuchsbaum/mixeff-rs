#!/usr/bin/env python3
"""Compare JSON files with tight numeric tolerances.

This is intentionally small and dependency-free so parity drift checks can run
on machines that only have Python from the system toolchain.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expected", type=Path)
    parser.add_argument("actual", type=Path)
    parser.add_argument("--abs-tol", type=float, default=1e-8)
    parser.add_argument("--rel-tol", type=float, default=1e-8)
    parser.add_argument(
        "--ignore",
        action="append",
        default=[],
        help="JSON pointer path to ignore, for volatile fields such as /runtime_ms",
    )
    parser.add_argument("--max-diffs", type=int, default=40)
    return parser.parse_args()


def pointer(path: tuple[str, ...]) -> str:
    if not path:
        return "/"
    return "/" + "/".join(part.replace("~", "~0").replace("/", "~1") for part in path)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def compare(
    expected: Any,
    actual: Any,
    path: tuple[str, ...],
    ignored: set[str],
    abs_tol: float,
    rel_tol: float,
    diffs: list[str],
) -> None:
    where = pointer(path)
    if where in ignored:
        return

    if is_number(expected) and is_number(actual):
        if not (math.isfinite(float(expected)) and math.isfinite(float(actual))):
            if expected != actual:
                diffs.append(f"{where}: non-finite mismatch expected={expected!r} actual={actual!r}")
            return
        if not math.isclose(float(expected), float(actual), abs_tol=abs_tol, rel_tol=rel_tol):
            delta = float(actual) - float(expected)
            diffs.append(
                f"{where}: numeric drift expected={expected!r} actual={actual!r} delta={delta:.6g}"
            )
        return

    if type(expected) is not type(actual):
        diffs.append(
            f"{where}: type mismatch expected={type(expected).__name__} actual={type(actual).__name__}"
        )
        return

    if isinstance(expected, dict):
        expected_keys = set(expected)
        actual_keys = set(actual)
        for key in sorted(expected_keys - actual_keys):
            key_path = pointer(path + (key,))
            if key_path not in ignored:
                diffs.append(f"{key_path}: missing from actual")
        for key in sorted(actual_keys - expected_keys):
            key_path = pointer(path + (key,))
            if key_path not in ignored:
                diffs.append(f"{key_path}: unexpected in actual")
        for key in sorted(expected_keys & actual_keys):
            compare(expected[key], actual[key], path + (key,), ignored, abs_tol, rel_tol, diffs)
        return

    if isinstance(expected, list):
        if len(expected) != len(actual):
            diffs.append(f"{where}: length mismatch expected={len(expected)} actual={len(actual)}")
            return
        for index, (left, right) in enumerate(zip(expected, actual)):
            compare(left, right, path + (str(index),), ignored, abs_tol, rel_tol, diffs)
        return

    if expected != actual:
        diffs.append(f"{where}: value mismatch expected={expected!r} actual={actual!r}")


def main() -> int:
    args = parse_args()
    diffs: list[str] = []
    compare(
        load_json(args.expected),
        load_json(args.actual),
        (),
        set(args.ignore),
        args.abs_tol,
        args.rel_tol,
        diffs,
    )
    if diffs:
        print(f"JSON drift: {args.expected} != {args.actual}", file=sys.stderr)
        for diff in diffs[: args.max_diffs]:
            print(f"  - {diff}", file=sys.stderr)
        if len(diffs) > args.max_diffs:
            print(f"  ... {len(diffs) - args.max_diffs} more differences", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
