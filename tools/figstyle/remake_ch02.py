#!/usr/bin/env python3
"""Chapter 2 (preliminaries) figures — chapter-by-chapter review loop
(docs/figure-style-guide.md §9).  UNDER REVIEW — not yet approved.

Sources, one pass here plus two companion scripts:
  * every JS-pipeline figure in diagrams/registry.mjs (diagrams/*.mjs):
    re-rendered with the token-unified engine (node diagrams/render.mjs),
    text outlined to paths in the bundled fonts
    (tools/figstyle/outline_svg.py), tight-cropped to the outlined glyphs
    (tools/figstyle/tightcrop.py), and synced img/auto/ -> img/ for the
    ids the book also uses (an img/<id>.svg that already exists).
  * polygon-circle.svg: legacy Cairo original remade here (matplotlib).
  * linear-algebra-dot.svg, linear-algebra-norms.svg: matplotlib, here,
    written to both img/ and img/auto/ (the JS registry no longer defines
    these ids).
  * calculus-gradient-field.svg, calculus-partial-slices.svg: matplotlib,
    owned by tools/figstyle/remake_ch02_calculus.py — run that script.
  * mdl-prelim-cosine/-linear-maps/-eigen.svg: owned by
    tools/gen_mdl_prelim_figures.py — run that script.

Run:  .venv-pytorch/bin/python tools/figstyle/remake_ch02.py
      .venv-pytorch/bin/python tools/figstyle/remake_ch02.py --js ID [ID ...]
      (the second form runs only the JS pipeline, for the listed ids)
Requires node (for the JS pipeline), rsvg-convert (for tightcrop), and
fontTools + PIL (both in .venv-pytorch).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from figstyle import tokens as T
from figstyle.outline_svg import outline
from figstyle.tightcrop import crop

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMG = os.path.join(REPO, "img")
AUTO = os.path.join(IMG, "auto")

def js_figures(ids=None):
    """Re-render the ENTIRE JS registry (every id belongs to a chapter-2
    module) or just `ids`, outline the text, tight-crop, and sync
    book-referenced copies img/auto -> img/.

    Outline BEFORE cropping: the crop then measures the glyphs that ship,
    not whatever fallback fonts this machine's rsvg substitutes for the
    live <text>, so the viewBox is exact and the same on every machine."""
    render = os.path.join(REPO, "diagrams", "render.mjs")
    if not ids:
        out = subprocess.run(["node", render, "--list"], check=True,
                             capture_output=True, text=True, cwd=REPO)
        ids = out.stdout.split()
    subprocess.run(["node", render, "--out", AUTO, *ids], check=True, cwd=REPO)
    for fid in sorted(ids):
        src = os.path.join(AUTO, f"{fid}.svg")
        print(outline(src))
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
# linear-algebra-dot / -norms — matplotlib, so the math is real (Computer     #
# Modern), per Alex's ch2 review.  Written to both img/ (book) and img/auto/  #
# (slides, via @fig:).                                                        #
# --------------------------------------------------------------------------- #

def _save_both(fig, name):
    from figstyle.mpl import save
    save(fig, name, out_dir=IMG)
    import shutil
    shutil.copyfile(os.path.join(IMG, f"{name}.svg"),
                    os.path.join(AUTO, f"{name}.svg"))


def fig_la_dot():
    """POINT: x^T y = ||x|| times the signed length ||y|| cos(theta) of the
    projection of y onto the line through x (the shadow, bracketed).  The
    formula itself lives in the caption, not in the figure."""
    import matplotlib
    matplotlib.use("svg")
    import matplotlib.pyplot as plt
    import numpy as np

    from figstyle.mpl import arrow, clean_axes, right_angle, use_style

    use_style(hashsalt="figstyle-ch02")
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    x = np.array([3.4, 1.0])
    y = np.array([1.5, 2.7])
    xn = x / np.linalg.norm(x)
    proj = float(y @ xn) * xn
    # no coordinate axes: the dot product does not depend on them
    ax.plot([y[0], proj[0]], [y[1], proj[1]], ls=(0, (4, 3)), color=T.MUTED,
            lw=1.6)
    right_angle(ax, proj, -xn, y - proj, size=0.2)
    arrow(ax, (0, 0), x, color=T.BLUE.base, lw=2.6)
    arrow(ax, (0, 0), y, color=T.ORANGE.base, lw=2.6)
    # the shadow of y on x: a bracket just below x, from 0 to proj
    nrm = np.array([xn[1], -xn[0]])          # unit normal, pointing down
    off, tick = 0.3, 0.1
    p0, p1 = off * nrm, proj + off * nrm
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=T.GREEN.base, lw=2.2)
    for p in (p0, p1):
        ax.plot([p[0] - tick * nrm[0], p[0] + tick * nrm[0]],
                [p[1] - tick * nrm[1], p[1] + tick * nrm[1]],
                color=T.GREEN.base, lw=2.2)
    lab = (p0 + p1) / 2 + 0.34 * nrm
    ax.text(lab[0], lab[1], r"$\|\mathbf{y}\|\cos\theta$",
            color=T.GREEN.dark, fontsize=16, ha="center", va="center",
            rotation=np.degrees(np.arctan2(xn[1], xn[0])))
    from matplotlib.patches import Arc
    a1 = np.degrees(np.arctan2(x[1], x[0]))
    a2 = np.degrees(np.arctan2(y[1], y[0]))
    ax.add_patch(Arc((0, 0), 1.3, 1.3, angle=0, theta1=a1, theta2=a2,
                     color=T.INK, lw=1.6))
    mid = np.radians((a1 + a2) / 2)
    ax.text(0.95 * np.cos(mid), 0.95 * np.sin(mid), r"$\theta$",
            color=T.INK, fontsize=17, ha="center", va="center")
    ax.text(x[0] + 0.12, x[1], r"$\mathbf{x}$", color=T.BLUE.dark,
            fontsize=18, ha="left", va="center")
    ax.text(y[0], y[1] + 0.2, r"$\mathbf{y}$", color=T.ORANGE.dark,
            fontsize=18, ha="center", va="bottom")
    clean_axes(ax, lim=((-0.45, 4.25), (-0.95, 3.25)), hide=True)
    _save_both(fig, "linear-algebra-dot")


def fig_la_norms():
    """POINT: for x = (3, -4) (the vector of the code cells), the l2 norm is
    the straight path (5) and the l1 norm the path along the grid of unit
    blocks (3 + 4 = 7)."""
    import matplotlib
    matplotlib.use("svg")
    import matplotlib.pyplot as plt

    from figstyle.mpl import arrow, axis_cross, clean_axes, use_style

    use_style(hashsalt="figstyle-ch02")
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    # the 3 x 4 grid of unit blocks the l1 path walks along
    for gx in range(4):
        ax.plot([gx, gx], [0, -4], color=T.HAIRLINE, lw=1.2, zorder=1)
    for gy in range(5):
        ax.plot([0, 3], [-gy, -gy], color=T.HAIRLINE, lw=1.2, zorder=1)
    axis_cross(ax, (-0.4, 4.1), (-4.55, 0.9), color=T.INK)
    # the two legs (the walk along the grid), then the vector itself
    ax.plot([0, 3], [0, 0], color=T.ORANGE.base, lw=3.2,
            solid_capstyle="round", zorder=3)
    ax.plot([3, 3], [0, -4], color=T.ORANGE.base, lw=3.2,
            solid_capstyle="round", zorder=3)
    arrow(ax, (0, 0), (3, -4), color=T.BLUE.base, lw=2.8)
    ax.text(1.5, 0.22, r"$3$", color=T.ORANGE.dark, fontsize=17,
            ha="center", va="bottom")
    ax.text(3.22, -2.0, r"$4$", color=T.ORANGE.dark, fontsize=17,
            ha="left", va="center")
    ax.text(3.22, -4.0, r"$\mathbf{x} = (3, {-4})$", color=T.INK, fontsize=16,
            ha="left", va="center")
    ax.text(1.1, -3.4, r"$\|\mathbf{x}\|_2 = 5$", color=T.BLUE.dark,
            fontsize=17, ha="center", va="center", zorder=2,
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
    ax.text(3.22, 0.62, r"$\|\mathbf{x}\|_1 = 3 + 4 = 7$",
            color=T.ORANGE.dark, fontsize=17, ha="left", va="center")
    clean_axes(ax, lim=((-0.5, 6.2), (-4.65, 1.0)), hide=True)
    _save_both(fig, "linear-algebra-norms")


def main(argv):
    if argv[:1] == ["--js"]:
        js_figures(argv[1:])
        return
    js_figures()
    fig_polygon_circle()
    fig_la_dot()
    fig_la_norms()
    print("chapter-2 figures refreshed")


if __name__ == "__main__":
    main(sys.argv[1:])
