#!/usr/bin/env python3
"""Fail when a hosted setup introduces new ``pip check`` conflicts."""

from __future__ import annotations

import argparse
from pathlib import Path


def conflict_lines(path: Path) -> set[str]:
    return {
        line.strip() for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def new_conflicts(before: Path, after: Path) -> list[str]:
    return sorted(conflict_lines(after) - conflict_lines(before))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args()
    before = conflict_lines(args.before)
    after = conflict_lines(args.after)
    introduced = sorted(after - before)
    if introduced:
        print("Hosted setup introduced pip dependency conflicts:")
        for conflict in introduced:
            print(f"  {conflict}")
        return 1
    print(
        "No new pip dependency conflicts "
        f"(provider baseline={len(before)}, after={len(after)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
