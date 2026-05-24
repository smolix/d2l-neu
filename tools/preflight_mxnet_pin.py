#!/usr/bin/env python3
"""Keep pyproject.toml's mxnet wheel pin in lockstep with `../mxnet/dist/`.

Why this exists: pyproject pins the custom MXNet build via a `file://`
path. uv records that path in `uv.lock` and, on `uv sync --extra X`
(any X), walks the full lockfile to compute cross-extra resolution.
For path-direct sources that means reading the wheel's METADATA from
disk. If the pinned wheel is missing (e.g. a newer local build
replaced it), the sync aborts — for every extra, not just mxnet.
Conflict declarations don't save us: they only affect install-time,
not lock validation.

This script is a no-op when the pinned wheel exists. When it's
missing, it looks in `../mxnet/dist/` for the newest compatible wheel
and, if found, runs `tools/update_mxnet_wheel.py --source local`
followed by `uv lock` so the lock matches reality. If no wheel is
found at all, it fails loudly with the actionable next step.

Designed to be a fast preflight on `uv sync` — it short-circuits
within a millisecond when the pin is already valid.

Usage:
    python3 tools/preflight_mxnet_pin.py            # auto-repair + relock
    python3 tools/preflight_mxnet_pin.py --check    # report only (exit 1 on mismatch)
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
DIST_DIR = ROOT.parent / "mxnet" / "dist"
UPDATE_SCRIPT = ROOT / "tools" / "update_mxnet_wheel.py"

PIN_RE = re.compile(
    r'"mxnet @ (?P<scheme>https|file)://(?P<rest>[^"]+?)'
    r'(?P<marker>\s*;\s*python_version\s*==\s*\'3\.\d+\')?"',
    re.MULTILINE)


def current_pin_path() -> Path | None:
    """Return the on-disk path the current mxnet pin points at, or None
    if the pin uses an https:// URL (no local file involved)."""
    m = PIN_RE.search(PYPROJECT.read_text())
    if not m:
        sys.exit("preflight: no mxnet pin found in pyproject.toml")
    if m.group("scheme") != "file":
        return None
    # `file://` URL: the rest is a percent-encoded absolute path.
    return Path(urllib.parse.unquote(m.group("rest")))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="Report only; exit 1 if pin doesn't match disk.")
    args = parser.parse_args()

    pinned = current_pin_path()
    if pinned is None:
        # https:// pin — uv will fetch from GitHub; nothing to validate here.
        return 0
    if pinned.exists():
        # Already in lockstep; fast path.
        return 0

    print(f"preflight: pinned mxnet wheel is missing on disk:")
    print(f"  pin:  {pinned}")

    if not DIST_DIR.is_dir():
        print(f"  ERROR: local dist dir not found either: {DIST_DIR}")
        print(f"  fix:  build the wheel, or repin via "
              f"`python3 tools/update_mxnet_wheel.py --source github`")
        return 1

    candidates = sorted(
        p for p in DIST_DIR.iterdir()
        if p.name.startswith("mxnet-")
        and p.name.endswith("-cp312-cp312-linux_x86_64.whl"))
    if not candidates:
        print(f"  ERROR: no cp312 wheel in {DIST_DIR}")
        print(f"  fix:  build the wheel, or repin via "
              f"`python3 tools/update_mxnet_wheel.py --source github`")
        return 1

    newest = candidates[-1].name
    print(f"  found: {newest} in {DIST_DIR}")
    if args.check:
        print("  (--check: not modifying pyproject)")
        return 1

    print("preflight: auto-bumping pin and relocking…")
    rc = subprocess.call(
        [sys.executable, str(UPDATE_SCRIPT), "--source", "local"],
        cwd=ROOT)
    if rc != 0:
        print(f"  ERROR: update_mxnet_wheel.py exited {rc}")
        return rc
    rc = subprocess.call(["uv", "lock"], cwd=ROOT)
    if rc != 0:
        print(f"  ERROR: `uv lock` exited {rc}")
        return rc
    print("preflight: mxnet pin repaired; sync can proceed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
