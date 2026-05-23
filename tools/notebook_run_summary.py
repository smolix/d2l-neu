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


def _parse_nb(ipynb: Path):
    try:
        return json.loads(ipynb.read_text())
    except Exception:
        return None


def last_error_from(nb: dict | None) -> str | None:
    """One-line summary of the last error cell in *nb*, or None."""
    if nb is None:
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


def has_executed_cells(nb: dict | None) -> bool | None:
    """True if *nb* contains code cells with a populated execution_count
    (i.e. nbconvert --execute actually ran). False if all code cells have
    execution_count=None (regenerated but never executed). None if the
    notebook has no code cells or couldn't be parsed."""
    if nb is None:
        return None
    has_code = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        has_code = True
        if cell.get("execution_count") is not None:
            return True
    return False if has_code else None


def scan(framework: str, include_errors: bool) -> dict:
    base = ROOT / "_notebooks" / framework
    notebooks = sorted(base.rglob("chapter_*/*.ipynb")) if base.exists() else []
    ok: list[Path] = []
    stale: list[tuple[Path, str | None]] = []
    failed: list[tuple[Path, str | None]] = []
    for nb_path in notebooks:
        stamp = nb_path.with_suffix(".executed")
        nb = _parse_nb(nb_path)
        err = last_error_from(nb) if include_errors else None
        executed = has_executed_cells(nb)
        if not stamp.exists():
            failed.append((nb_path, err))
            continue
        # A stamp without an actually-executed .ipynb is a leftover. nbconvert
        # --execute populates execution_count on code cells; gen_notebooks
        # writes them as None. So a stamp paired with all-None code cells
        # means the last execution attempt failed (no `touch $@` ran) but the
        # stamp from an earlier run is still on disk.
        if executed is False:
            stale.append((nb_path, err))
            continue
        ok.append(nb_path)
    return {"total": len(notebooks), "ok": ok, "stale": stale, "failed": failed}


def format_report(framework: str, result: dict) -> str:
    total = result["total"]
    n_ok = len(result["ok"])
    n_stale = len(result["stale"])
    n_fail = len(result["failed"])
    if total == 0:
        return f"  {framework:10s}  no notebooks found under _notebooks/{framework}/"
    line = f"  {framework:10s}  OK {n_ok:>3d} / {total:>3d}   STALE {n_stale:>2d}   FAILED {n_fail:>3d}"
    out = [line]
    for nb, err in result["failed"]:
        rel = nb.relative_to(ROOT / "_notebooks" / framework).with_suffix("")
        out.append(f"      FAIL   {rel}")
        if err:
            out.append(f"             └─ {err}")
    for nb, err in result["stale"]:
        rel = nb.relative_to(ROOT / "_notebooks" / framework).with_suffix("")
        out.append(f"      STALE  {rel}  (.ipynb has no executed cells — last run failed or notebook regenerated since)")
        if err:
            out.append(f"             └─ {err}")
    return "\n".join(out)


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
