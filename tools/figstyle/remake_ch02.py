#!/usr/bin/env python3
"""Chapter 2 (preliminaries) figures — chapter-by-chapter review loop
(docs/figure-style-guide.md §9).  UNDER REVIEW — not yet approved.

Three sources, one pass:
  * 17 JS-pipeline figures (diagrams/*.mjs): re-rendered with the
    token-unified engine (node diagrams/render.mjs), then tight-cropped
    (tools/figstyle/tightcrop.py) and synced img/auto/ -> img/.
  * polygon-circle.svg: legacy Cairo original remade here (matplotlib).
  * mdl-prelim-cosine.svg: owned by tools/gen_mdl_prelim_figures.py,
    which now uses figstyle.mpl — run that script, not this one, for it.

Run:  python3 tools/figstyle/remake_ch02.py
Requires node (for the JS pipeline) and rsvg-convert (for tightcrop).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from figstyle import tokens as T
from figstyle.tightcrop import crop

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMG = os.path.join(REPO, "img")
AUTO = os.path.join(IMG, "auto")

def js_figures():
    """Re-render the ENTIRE JS registry (every id belongs to a chapter-2
    module), tight-crop, and sync book-referenced copies img/auto -> img/."""
    out = subprocess.run(["node", os.path.join(REPO, "diagrams", "render.mjs"),
                          "--list"], check=True, capture_output=True,
                         text=True, cwd=REPO)
    ids = out.stdout.split()
    subprocess.run(["node", os.path.join(REPO, "diagrams", "render.mjs"),
                    "--out", AUTO], check=True, cwd=REPO)
    for fid in sorted(ids):
        src = os.path.join(AUTO, f"{fid}.svg")
        print(crop(src))
        dst = os.path.join(IMG, f"{fid}.svg")
        if os.path.exists(dst):          # book-referenced copy stays in sync
            shutil.copyfile(src, dst)


# --------------------------------------------------------------------------- #
# polygon-circle — POINT: inscribed polygons approach the circle as the       #
# number of sides grows (Archimedes' route to pi).                            #
# --------------------------------------------------------------------------- #

def fig_polygon_circle():
    import matplotlib
    matplotlib.use("svg")
    import matplotlib.pyplot as plt
    import numpy as np

    from figstyle.mpl import save, use_style

    use_style(hashsalt="figstyle-ch02")
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.8))
    for ax, n in zip(axes, (4, 8, 32)):
        t = np.linspace(0, 2 * np.pi, 400)
        ax.plot(np.cos(t), np.sin(t), color=T.FAINT, lw=T.SW_HAIR,
                ls=(0, (4, 3)))
        p = np.linspace(0, 2 * np.pi, n + 1) + np.pi / 2
        ax.fill(np.cos(p), np.sin(p), facecolor=T.BLUE.tint,
                edgecolor=T.BLUE.base, lw=T.SW_BOX, zorder=2)
        ax.text(0, -1.32, f"$n = {n}$", ha="center", va="top",
                fontsize=15, color=T.INK)
        ax.set_aspect("equal")
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.55, 1.25)
        ax.axis("off")
    save(fig, "polygon-circle", out_dir=IMG)


def main():
    js_figures()
    fig_polygon_circle()
    print("chapter-2 figures refreshed")


if __name__ == "__main__":
    main()
