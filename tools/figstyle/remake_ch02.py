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
# Layout mirrors the ORIGINAL figure (Alex, ch2 review): four panels          #
# n = 4, 5, 6, 8, solid circle, filled polygon, dashed radial spokes.         #
# --------------------------------------------------------------------------- #

def fig_polygon_circle():
    import matplotlib
    matplotlib.use("svg")
    import matplotlib.pyplot as plt
    import numpy as np

    from figstyle.mpl import save, use_style

    use_style(hashsalt="figstyle-ch02")
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 1.95))
    fig.subplots_adjust(wspace=0.18)
    for ax, (n, phase) in zip(axes, [(4, np.pi / 4), (5, np.pi / 2),
                                     (6, 0.0), (8, np.pi / 2)]):
        t = np.linspace(0, 2 * np.pi, 400)
        ax.plot(np.cos(t), np.sin(t), color=T.MUTED, lw=1.5)
        p = np.linspace(0, 2 * np.pi, n + 1) + phase
        ax.fill(np.cos(p), np.sin(p), facecolor=T.BLUE.tint,
                edgecolor=T.BLUE.base, lw=T.SW_BOX, zorder=2)
        for k in range(n):     # dashed spokes, center to each vertex
            ax.plot([0, np.cos(p[k])], [0, np.sin(p[k])], color=T.MUTED,
                    lw=1.2, ls=(0, (4, 3)), zorder=3)
        ax.plot(0, 0, "o", color=T.INK, ms=3, zorder=4)
        ax.set_aspect("equal")
        ax.set_xlim(-1.12, 1.12)
        ax.set_ylim(-1.12, 1.12)
        ax.axis("off")
    save(fig, "polygon-circle", out_dir=IMG)


# --------------------------------------------------------------------------- #
# linear-algebra-dot / -norms — remade in matplotlib so the equations are     #
# real (Computer Modern) math, per Alex's ch2 review.  These OVERRIDE the     #
# JS-pipeline versions in both img/ and img/auto/.                            #
# --------------------------------------------------------------------------- #

def _save_both(fig, name):
    from figstyle.mpl import save
    save(fig, name, out_dir=IMG)
    import shutil
    shutil.copyfile(os.path.join(IMG, f"{name}.svg"),
                    os.path.join(AUTO, f"{name}.svg"))


def fig_la_dot():
    import matplotlib
    matplotlib.use("svg")
    import matplotlib.pyplot as plt
    import numpy as np

    from figstyle.mpl import arrow, axis_cross, clean_axes, right_angle, use_style

    use_style(hashsalt="figstyle-ch02")
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    a = np.array([3.4, 1.0])
    b = np.array([1.5, 2.7])
    an = a / np.linalg.norm(a)
    proj = float(b @ an) * an
    axis_cross(ax, (-0.35, 4.1), (-0.3, 3.2))
    ax.plot([b[0], proj[0]], [b[1], proj[1]], ls=(0, (4, 3)), color=T.MUTED,
            lw=1.6)
    right_angle(ax, proj, -an, b - proj, size=0.2)
    arrow(ax, (0, 0), a, color=T.BLUE.base, lw=2.6)
    arrow(ax, (0, 0), b, color=T.ORANGE.base, lw=2.6)
    from matplotlib.patches import Arc
    a1 = np.degrees(np.arctan2(a[1], a[0]))
    a2 = np.degrees(np.arctan2(b[1], b[0]))
    ax.add_patch(Arc((0, 0), 1.3, 1.3, angle=0, theta1=a1, theta2=a2,
                     color=T.MUTED, lw=1.6))
    mid = np.radians((a1 + a2) / 2)
    ax.text(0.92 * np.cos(mid), 0.92 * np.sin(mid), r"$\theta$",
            color=T.MUTED, fontsize=16, ha="center", va="center")
    ax.text(a[0] + 0.12, a[1], r"$\mathbf{a}$", color=T.BLUE.dark,
            fontsize=17, ha="left", va="center")
    ax.text(b[0], b[1] + 0.22, r"$\mathbf{b}$", color=T.ORANGE.dark,
            fontsize=17, ha="center", va="bottom")
    ax.text(1.9, -0.75,
            r"$\mathbf{a} \cdot \mathbf{b} \,=\, \sum_i a_i b_i"
            r" \,=\, \|\mathbf{a}\|\,\|\mathbf{b}\|\,\cos\theta$",
            fontsize=15.5, color=T.INK, ha="center", va="center")
    clean_axes(ax, lim=((-0.5, 4.3), (-1.15, 3.3)), hide=True)
    _save_both(fig, "linear-algebra-dot")


def fig_la_norms():
    import matplotlib
    matplotlib.use("svg")
    import matplotlib.pyplot as plt

    from figstyle.mpl import arrow, axis_cross, clean_axes, use_style

    use_style(hashsalt="figstyle-ch02")
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    axis_cross(ax, (-0.4, 3.9), (-0.35, 4.6))
    # the two legs (Manhattan route), then the vector itself (Euclidean)
    ax.plot([0, 3], [0, 0], color=T.ORANGE.base, lw=2.6,
            solid_capstyle="round", zorder=2)
    ax.plot([3, 3], [0, 4], color=T.ORANGE.base, lw=2.6,
            solid_capstyle="round", zorder=2)
    arrow(ax, (0, 0), (3, 4), color=T.BLUE.base, lw=2.8)
    ax.text(1.5, -0.32, r"$3$", color=T.ORANGE.dark, fontsize=16,
            ha="center", va="top")
    ax.text(3.22, 2.0, r"$4$", color=T.ORANGE.dark, fontsize=16,
            ha="left", va="center")
    ax.text(1.28, 2.16, r"$5$", color=T.BLUE.dark, fontsize=16,
            ha="right", va="bottom")
    ax.text(2.55, 4.15, r"$\mathbf{v} = (3, 4)$", color=T.INK, fontsize=15.5,
            ha="left", va="bottom")
    ax.text(1.75, -1.4,
            r"$\|\mathbf{v}\|_2 = \sqrt{3^2 + 4^2} = 5$  (Euclidean)",
            fontsize=15, color=T.BLUE.dark, ha="center", va="center")
    ax.text(1.75, -2.12,
            r"$\|\mathbf{v}\|_1 = 3 + 4 = 7$  (Manhattan)",
            fontsize=15, color=T.ORANGE.dark, ha="center", va="center")
    clean_axes(ax, lim=((-0.6, 4.6), (-2.5, 4.75)), hide=True)
    _save_both(fig, "linear-algebra-norms")


def main():
    js_figures()
    fig_polygon_circle()
    fig_la_dot()       # overrides the JS render (real math in the equation)
    fig_la_norms()
    print("chapter-2 figures refreshed")


if __name__ == "__main__":
    main()
