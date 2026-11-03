#!/usr/bin/env python3
"""Generate the new illustrative figure of "Estimating Mutual Information"
(``chapter_mdl-information-theory/mdl-mi-estimation.md``) in the chapter's
shared house style.

The shared style lives in ``gen_mdl_figures.py``; importing it applies the
``plt.rcParams`` (fixed ``svg.hashsalt`` and ``metadata={'Date': None}`` in
``save()`` make re-runs byte-for-byte identical) and exposes the palette.  The
section's other figure (``mdl-it-infonce-pos-neg``) stays in
``gen_mdl_infotheory_figures.py``.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_infotheory_mi_estimation_figures.py

Writes ``img/mdl-it-ksg-box.svg`` (and the copy in ``img/auto/`` that the
slides read).
"""

from __future__ import annotations

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, INK = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.INK

from matplotlib.patches import Rectangle  # noqa: E402


def _copy_to_auto(name):
    shutil.copyfile(os.path.join(fl.IMG_DIR, f"{name}.svg"),
                    os.path.join(fl.IMG_DIR, "auto", f"{name}.svg"))


def fig_ksg_box():
    """The KSG count at one sample point i of a 40-point Gaussian sample
    (rho = 0.8): the Chebyshev box of half-width r_i reaching the m-th joint
    neighbour (m = 3), the vertical strip |x - x_i| < r_i and the horizontal
    strip |y - y_i| < r_i.  All counts in the labels are computed from the
    drawn sample, so they are right by construction."""
    rng = np.random.default_rng(313)         # the figure's data; fixed so the
    n, rho, m = 40, 0.8, 3                  # SVG is byte-idempotent
    x = rng.standard_normal(n)
    y = rho * x + np.sqrt(1 - rho**2) * rng.standard_normal(n)
    # Drop any sample that would hide under an earlier dot, so that every
    # counted sample is visible and the labels match what the reader counts.
    keep = [j for j in range(n)
            if all(np.hypot(x[j] - x[k], y[j] - y[k]) > 0.05 for k in range(j))]
    x, y, n = x[keep], y[keep], len(keep)
    i = int(np.argmin((x - 0.2) ** 2 + (y - 0.3) ** 2))
    dx, dy = np.abs(x - x[i]), np.abs(y - y[i])
    d = np.maximum(dx, dy)
    d[i] = np.inf
    order = np.argsort(d)
    r = d[order[m - 1]]                      # half-width: m-th neighbour
    inside = order[:m - 1]                   # strictly inside the box
    boundary = order[m - 1]                  # on the box's edge
    in_x = (dx < r) & (np.arange(n) != i)
    in_y = (dy < r) & (np.arange(n) != i)
    n_x, n_y = int(in_x.sum()), int(in_y.sum())
    only_x = in_x & ~in_y
    only_y = in_y & ~in_x
    lo, hi = -2.7, 2.7

    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    # Strips first (full height / full width), then the box on top.
    ax.add_patch(Rectangle((x[i] - r, lo), 2 * r, hi - lo, color=BLUE,
                           alpha=0.12, lw=0, zorder=0))
    ax.add_patch(Rectangle((lo, y[i] - r), hi - lo, 2 * r, color=GREEN,
                           alpha=0.12, lw=0, zorder=0))
    ax.add_patch(Rectangle((x[i] - r, y[i] - r), 2 * r, 2 * r,
                           facecolor=ORANGE, alpha=0.15, lw=0, zorder=1))
    ax.add_patch(Rectangle((x[i] - r, y[i] - r), 2 * r, 2 * r,
                           fill=False, edgecolor=ORANGE, lw=1.8, zorder=2))

    rest = ~(only_x | only_y) & (np.arange(n) != i)
    rest[inside] = False
    rest[boundary] = False
    ax.plot(x[rest], y[rest], "o", color=GRAY, ms=6, zorder=3)
    ax.plot(x[only_x], y[only_x], "o", color=BLUE, ms=6.5, zorder=3)
    ax.plot(x[only_y], y[only_y], "o", color=GREEN, ms=6.5, zorder=3)
    ax.plot(x[inside], y[inside], "o", color=ORANGE, ms=7, zorder=4)
    ax.plot(x[boundary], y[boundary], "o", mfc="white", mec=ORANGE, mew=1.8,
            ms=8, zorder=4)
    ax.plot(x[i], y[i], "*", color=BLUE, ms=15, zorder=5, mec="white",
            mew=0.6)

    box = dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.9)
    # The star's name sits in the empty upper-left corner of the box.
    ax.text(x[i] - 0.06, y[i] + 0.08, r"$i$", fontsize=15, ha="right",
            va="bottom", color=INK, zorder=6)
    # r_i: a double-headed arrow from the star to the box's right edge,
    # labelled underneath (an inside neighbour sits above it).
    ax.annotate("", xy=(x[i] + r, y[i]), xytext=(x[i], y[i]),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.1,
                                shrinkA=6, shrinkB=0), zorder=6)
    ax.text(x[i] + r / 2 + 0.04, y[i] - 0.07, r"$r_i$", fontsize=15,
            ha="center", va="top", color=INK, zorder=6)
    # The neighbour count, just outside the box's top-left corner (left of
    # the vertical strip, where there are no samples).
    ax.text(x[i] - r - 0.06, y[i] + r + 0.06, rf"$m - 1 = {m - 1}$ inside",
            fontsize=13, ha="right", va="bottom", color=INK, zorder=6)
    ax.text(x[i] + r / 2 - 0.2, hi - 0.1, rf"$n_x^{{(i)}} = {n_x}$",
            fontsize=14, ha="center", va="top", color=INK, zorder=6,
            bbox=box)
    ax.text(hi - 0.08, y[i], rf"$n_y^{{(i)}} = {n_y}$", fontsize=14,
            ha="right", va="center", color=INK, zorder=6, bbox=box)
    # Coordinate axes along the frame (an axis through the origin would run
    # through the strips and their labels): black, no ticks.
    fl.clean_axes(ax, lim=((lo, hi), (lo, hi)))
    for side in ("left", "bottom"):
        ax.spines[side].set_color("black")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(r"$x$", fontsize=15, color=INK)
    ax.set_ylabel(r"$y$", fontsize=15, color=INK, rotation=0, labelpad=10)
    fl.save(fig, "mdl-it-ksg-box")
    _copy_to_auto("mdl-it-ksg-box")
    return n_x, n_y, r


FIGURES = [fig_ksg_box]


def main():
    fl.WRITTEN.clear()
    for fn in FIGURES:
        out = fn()
        print(f"{fn.__name__}: {out}")
    for path in fl.WRITTEN:
        print(f"wrote {path} ({os.path.getsize(path):,d} bytes)")


if __name__ == "__main__":
    main()
