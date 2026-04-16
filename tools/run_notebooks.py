#!/usr/bin/env python3
"""Execute Jupyter notebooks for a given framework.

Runs each notebook in _notebooks/<framework>/ using nbconvert's execute
preprocessor.  Produces executed notebooks in-place and writes a summary
report to stdout.

Usage:
    python tools/run_notebooks.py pytorch          # run all pytorch notebooks
    python tools/run_notebooks.py jax --timeout 600
    python tools/run_notebooks.py pytorch --glob "chapter_linear*/**"
    python tools/run_notebooks.py pytorch --list   # dry-run: list notebooks
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "_notebooks"


def find_notebooks(framework, glob_pattern=None):
    """Return sorted list of .ipynb paths for the given framework."""
    fw_dir = NOTEBOOKS_DIR / framework
    if not fw_dir.is_dir():
        print(f"Error: {fw_dir} does not exist. Run ./build.sh notebooks first.",
              file=sys.stderr)
        sys.exit(1)

    if glob_pattern:
        nbs = sorted(fw_dir.glob(glob_pattern))
        nbs = [nb for nb in nbs if nb.suffix == ".ipynb"]
    else:
        nbs = sorted(fw_dir.rglob("*.ipynb"))
    return nbs


def execute_notebook(nb_path, timeout=600, kernel="python3"):
    """Execute a single notebook in-place via jupyter nbconvert.

    Returns (success: bool, elapsed: float, error_msg: str | None).
    """
    t0 = time.time()
    result = subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            f"--ExecutePreprocessor.timeout={timeout}",
            f"--ExecutePreprocessor.kernel_name={kernel}",
            str(nb_path),
        ],
        capture_output=True,
        text=True,
        timeout=timeout + 60,  # extra margin for nbconvert overhead
    )
    elapsed = time.time() - t0

    if result.returncode == 0:
        return True, elapsed, None
    else:
        # Extract useful error from stderr
        err = result.stderr.strip()
        # Try to find the actual exception
        lines = err.splitlines()
        # Look for CellExecutionError or similar
        short_err = None
        for i, line in enumerate(lines):
            if "CellExecutionError" in line or "error" in line.lower():
                short_err = "\n".join(lines[max(0, i - 2):i + 5])
                break
        if not short_err:
            short_err = "\n".join(lines[-10:])  # last 10 lines
        return False, elapsed, short_err


def main():
    parser = argparse.ArgumentParser(description="Execute d2l notebooks")
    parser.add_argument("framework", choices=["pytorch", "tensorflow", "jax", "mxnet"])
    parser.add_argument("--timeout", type=int, default=600,
                        help="Per-cell timeout in seconds (default: 600)")
    parser.add_argument("--glob", type=str, default=None,
                        help="Glob pattern to select notebooks (e.g. 'chapter_linear*/**')")
    parser.add_argument("--list", action="store_true",
                        help="List notebooks without executing")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Continue running after a notebook fails")
    args = parser.parse_args()

    nbs = find_notebooks(args.framework, args.glob)
    if not nbs:
        print("No notebooks found.")
        return

    print(f"Found {len(nbs)} notebooks for {args.framework}")

    if args.list:
        for nb in nbs:
            print(f"  {nb.relative_to(NOTEBOOKS_DIR)}")
        return

    passed, failed, errors = 0, 0, []
    for i, nb in enumerate(nbs, 1):
        rel = nb.relative_to(NOTEBOOKS_DIR)
        print(f"\n[{i}/{len(nbs)}] {rel} ", end="", flush=True)

        ok, elapsed, err = execute_notebook(nb, timeout=args.timeout)
        if ok:
            passed += 1
            print(f"OK ({elapsed:.1f}s)")
        else:
            failed += 1
            errors.append((rel, err))
            print(f"FAILED ({elapsed:.1f}s)")
            print(f"  Error: {err[:200] if err else 'unknown'}")
            if not args.continue_on_error:
                print("\nStopping on first failure. Use --continue-on-error to keep going.")
                break

    # Summary
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, "
          f"{len(nbs) - passed - failed} skipped")
    if errors:
        print(f"\nFailed notebooks:")
        for rel, err in errors:
            print(f"  - {rel}")
    print(f"{'='*60}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
