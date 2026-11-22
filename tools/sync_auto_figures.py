#!/usr/bin/env python3
"""Keep the two copies of a figure, img/<id>.svg and img/auto/<id>.svg, identical.

Slides resolve ``@fig:<id>`` from ``img/auto/`` (tools/gen_slides.py), the
book from ``img/``.  A figure that lives in both places must be the same
file, or a slide shows an older picture than the book:

* matplotlib generators (tools/gen_mdl_*_figures.py, ...) write ``img/<id>.svg``;
  only some of them also copied it to ``img/auto/``, so ``make figures`` left
  57 slide copies stale (2026-11-22).  For these the book copy is the source.
* the JS diagram pipeline (tools/build_module_diagrams.mjs,
  tools/figstyle/remake_ch02.py --js) writes ``img/auto/<id>.svg`` and
  refreshes the book copy itself; a mismatch there is reported, never guessed.

Usage:
    python3 tools/sync_auto_figures.py           # copy img/ -> img/auto/ (matplotlib figures)
    python3 tools/sync_auto_figures.py --check   # exit 1 if any pair differs

``make figures`` runs the sync and then the check; ``tools/lint_source.py
--corpus`` runs the check.
"""

from __future__ import annotations

import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def differing_pairs(repo: str = REPO) -> list[str]:
    """Ids whose img/auto/<id>.svg and img/<id>.svg both exist and differ."""
    auto = os.path.join(repo, "img", "auto")
    out = []
    if not os.path.isdir(auto):
        return out
    for name in sorted(os.listdir(auto)):
        if not name.endswith(".svg"):
            continue
        book = os.path.join(repo, "img", name)
        if os.path.exists(book) and _read(book) != _read(os.path.join(auto, name)):
            out.append(name[:-4])
    return out


def _is_matplotlib(path: str) -> bool:
    return b"matplotlib" in _read(path)[:4096]


def main(argv: list[str]) -> int:
    check = "--check" in argv
    diff = differing_pairs()
    if check:
        for fid in diff:
            print(f"img/auto/{fid}.svg differs from img/{fid}.svg "
                  "(run: python3 tools/sync_auto_figures.py)")
        return 1 if diff else 0
    rc = 0
    for fid in diff:
        book = os.path.join(REPO, "img", f"{fid}.svg")
        if _is_matplotlib(book):
            shutil.copyfile(book, os.path.join(REPO, "img", "auto", f"{fid}.svg"))
            print(f"synced img/auto/{fid}.svg <- img/{fid}.svg")
        else:
            print(f"img/auto/{fid}.svg differs from img/{fid}.svg and is not a "
                  "matplotlib figure: regenerate it with "
                  "tools/build_module_diagrams.mjs", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
