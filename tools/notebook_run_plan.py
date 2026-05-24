#!/usr/bin/env python3
"""Print a per-framework plan of what `make run-*` will actually execute.

For each requested framework, reads `_notebooks/<fw>/MANIFEST.mk` to know
the full notebook set split by resource pool (CPU / GPU / multi-GPU),
then checks which `.executed` stamps are missing. Prints a compact
table at the start of a run so a human reading the log sees the shape
of the work — how many notebooks are pending in each pool, on which
framework — before the parallel `[fw] foo: start` lines start mixing
together.

Usage:
    python3 tools/notebook_run_plan.py                  # all frameworks
    python3 tools/notebook_run_plan.py -f mxnet -f jax  # subset
    python3 tools/notebook_run_plan.py --list           # also list pending paths

The script is read-only and exits 0 even if a manifest is missing
(prints `(manifest missing)` for that framework) so the surrounding
recipe is never derailed by plan-printing.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

FRAMEWORKS = ("pytorch", "tensorflow", "jax", "mxnet")
ROOT = Path(__file__).resolve().parent.parent

POOL_VARS = {
    "cpu": "EXECUTED_CPU_{fw}",
    "gpu": "EXECUTED_GPU_{fw}",
    "multi": "EXECUTED_MULTI_GPU_{fw}",
}


def _parse_manifest(path: Path) -> dict[str, list[Path]] | None:
    """Return {pool: [stamp_path, ...]} or None if manifest is missing."""
    if not path.exists():
        return None
    text = path.read_text()
    pools: dict[str, list[Path]] = {}
    for pool, var_tmpl in POOL_VARS.items():
        # Variable name has framework appended; manifest declares it as
        #   EXECUTED_CPU_pytorch := a.executed b.executed ...
        m = re.search(rf'^{var_tmpl.format(fw="[a-z]+")}\s*:?=\s*(.*)$', text, re.MULTILINE)
        if not m:
            pools[pool] = []
            continue
        pools[pool] = [ROOT / p for p in m.group(1).split() if p]
    return pools


def _pending(pools: dict[str, list[Path]]) -> dict[str, list[Path]]:
    return {pool: [p for p in stamps if not p.exists()]
            for pool, stamps in pools.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-f", "--framework", action="append", choices=FRAMEWORKS,
                        help="Limit to one framework (repeatable).")
    parser.add_argument("--list", action="store_true",
                        help="Also list pending notebook paths under each row.")
    args = parser.parse_args()

    fws = args.framework or list(FRAMEWORKS)
    rows = []
    total_pending = 0
    for fw in fws:
        manifest = ROOT / "_notebooks" / fw / "MANIFEST.mk"
        pools = _parse_manifest(manifest)
        if pools is None:
            rows.append((fw, None, None, None))
            continue
        pending = _pending(pools)
        total = sum(len(v) for v in pools.values())
        pend_total = sum(len(v) for v in pending.values())
        total_pending += pend_total
        rows.append((fw, pools, pending, (pend_total, total)))

    print()
    print("=== Notebook run plan ===")
    print(f"  {'framework':<11} {'pending':>11}    {'CPU':>4} {'GPU':>4} {'multi-GPU':>9}")
    for fw, pools, pending, totals in rows:
        if pools is None:
            print(f"  {fw:<11} {'(manifest missing)':>11}")
            continue
        pend_total, total = totals
        marker = "(all done)" if pend_total == 0 else ""
        print(f"  {fw:<11} {pend_total:>4d} / {total:<4d}    "
              f"{len(pending['cpu']):>4d} {len(pending['gpu']):>4d} {len(pending['multi']):>9d}  "
              f"{marker}")
        if args.list and pend_total:
            for pool in ("cpu", "gpu", "multi"):
                for stamp in pending[pool]:
                    rel = stamp.relative_to(ROOT / "_notebooks" / fw).with_suffix("")
                    print(f"      [{pool:>5}] {rel}")
    if total_pending == 0:
        print("  (nothing to do — all stamps present)")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
