#!/usr/bin/env python3
"""Print a post-run summary of which notebooks executed and which failed.

For each framework, walks `_notebooks/<fw>/chapter_*/`, pairs each `.ipynb`
with its `.executed` stamp, and reports:

    OK / FAILED / TOTAL   per framework
    list of failing notebook paths (with the last error message we can find)

Intended to be called at the tail of `run-notebooks-<fw>` / `run-all-notebooks`
so a partial run leaves a clear "which ones broke?" report even when `-k`
let Make grind past first-failure.

The report is best-effort: any I/O error per notebook is swallowed so the
summary never masks the real build outcome.

Usage:
    python3 tools/notebook_run_summary.py                 # all frameworks
    python3 tools/notebook_run_summary.py -f mxnet        # one framework
    python3 tools/notebook_run_summary.py --no-errors     # skip last-error
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FRAMEWORKS = ("pytorch", "tensorflow", "jax", "mxnet")
ROOT = Path(__file__).resolve().parent.parent


def last_error(ipynb: Path) -> str | None:
    """Return a one-line summary of the last error cell in *ipynb*, or None."""
    try:
        nb = json.loads(ipynb.read_text())
    except Exception:
        return None
    for cell in reversed(nb.get("cells", [])):
        for output in reversed(cell.get("outputs", []) or []):
            if output.get("output_type") == "error":
                ename = output.get("ename", "Error")
                evalue = (output.get("evalue", "") or "").splitlines()
                first = evalue[0] if evalue else ""
                msg = f"{ename}: {first}".strip().rstrip(":")
                return msg[:200]
    return None


def scan(framework: str, include_errors: bool) -> dict:
    base = ROOT / "_notebooks" / framework
    notebooks = sorted(base.rglob("chapter_*/*.ipynb")) if base.exists() else []
    ok: list[Path] = []
    failed: list[tuple[Path, str | None]] = []
    for nb in notebooks:
        stamp = nb.with_suffix(".executed")
        if stamp.exists():
            ok.append(nb)
        else:
            err = last_error(nb) if include_errors else None
            failed.append((nb, err))
    return {"total": len(notebooks), "ok": ok, "failed": failed}


def format_report(framework: str, result: dict) -> str:
    total = result["total"]
    n_ok = len(result["ok"])
    n_fail = len(result["failed"])
    if total == 0:
        return f"  {framework:10s}  no notebooks found under _notebooks/{framework}/"
    line = f"  {framework:10s}  OK {n_ok:>3d} / {total:>3d}   FAILED {n_fail:>3d}"
    if n_fail == 0:
        return line
    fail_lines = [line]
    for nb, err in result["failed"]:
        rel = nb.relative_to(ROOT / "_notebooks" / framework).with_suffix("")
        fail_lines.append(f"      FAIL  {rel}")
        if err:
            fail_lines.append(f"            └─ {err}")
    return "\n".join(fail_lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-f", "--framework", action="append", choices=FRAMEWORKS,
                        help="Limit to one framework (repeatable).")
    parser.add_argument("--no-errors", dest="errors", action="store_false",
                        help="Skip the last-error line for failing notebooks.")
    args = parser.parse_args()

    fws = args.framework or list(FRAMEWORKS)
    print()
    print("=== Notebook run summary ===")
    for fw in fws:
        result = scan(fw, include_errors=args.errors)
        print(format_report(fw, result))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
