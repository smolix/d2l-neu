#!/usr/bin/env python3
"""Generate the illustrative figure for the "Linear Regression ->
Generalization" section (``chapter_linear-regression/generalization.md``) in
the one shared house style.

One figure: the classical picture of training and test error against model
complexity.  Training error falls steadily; test error first falls (less
underfitting) and then rises (more overfitting), so it is smallest at an
intermediate complexity and never drops below the noise level.  The curves are
built from the decomposition the section teaches rather than placed by hand:
test error = squared bias + variance + noise, with bias falling and variance
rising in the complexity, and training error falling below the noise level as
the model starts to fit the noise.  The section's own experiment (polynomials
of growing degree) measures the same shape; this figure is its general form.

The shared style lives in ``figstyle.mpl``; ``use_style`` applies the
``plt.rcParams`` (fixed ``svg.hashsalt``; ``save`` drops the SVG date) so
re-runs are byte-for-byte identical.

Run with the repo's pytorch venv (matplotlib + numpy are available):

    .venv-pytorch/bin/python tools/gen_mdl_generalization_figures.py

The figure is written to ``img/mdl-gen-capacity-error.svg``.
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle.mpl as fl  # the unified style (docs/figure-style-guide.md)
from figstyle import tokens as T

fl.use_style(hashsalt="mdl-gen")

FS = 12.5          # in-figure label size (pt), legible at column width


def fig_capacity_error():
    """POINT: as complexity grows, training error keeps falling while test
    error falls, bottoms out, and rises again; the gap between them widens.
    Left of the minimum the model underfits, right of it the model overfits,
    and no model gets below the noise level."""
    c = np.linspace(0.35, 6.0, 500)            # model complexity (schematic)
    c_max = 6.6                                 # where training error would hit 0
    bias2 = 0.9 / c ** 1.6                      # falls with complexity
    noise = 0.15                                # irreducible error
    var = 0.022 * c ** 2.2                      # grows with complexity
    test = bias2 + noise + var
    train = bias2 + noise * (1 - c / c_max)     # fits more of the noise

    i_best = int(np.argmin(test))
    c_best, e_best = c[i_best], test[i_best]
    ymax = 1.75

    fig, ax = plt.subplots(figsize=(6.0, 3.7))

    # noise level: a faint reference line, labeled at the left where the
    # curves are high above it
    ax.plot([c[0], c[-1]], [noise, noise], color=T.FAINT, lw=T.SW_HAIR,
            ls=(0, (1.5, 2.5)), zorder=1)
    ax.text(c[0] + 0.05, noise + 0.03, "noise level", color=T.INK,
            fontsize=FS - 1.5, ha="left", va="bottom")

    # the section's computed plots draw training error as a solid blue line
    # and test error as a dashed magenta one; keep the same code here
    ax.plot(c, train, color=T.BLUE.base, lw=T.SW_LINE, zorder=3)
    ax.plot(c, test, color=T.PURPLE.base, lw=T.SW_HEAVY - 0.5,
            ls=(0, (5, 2.5)), zorder=4)

    # the best complexity: dot on the minimum, dashed drop line to the axis
    ax.plot([c_best, c_best], [0, e_best], color=T.FAINT, lw=T.SW_HAIR,
            ls=(0, (4, 3)), zorder=2)
    ax.plot([c_best], [e_best], "o", color=T.PURPLE.base, ms=7, zorder=5)
    ax.text(c_best, -0.035, "best complexity", color=T.INK, fontsize=FS - 1.5,
            ha="center", va="top")

    # curve names, placed where each curve is clear of the other
    xt = 4.3          # above the rising curve, ending where it is lowest
    ax.text(xt, test[np.argmin(np.abs(c - xt))] + 0.05, "test error",
            color=T.INK, fontsize=FS, ha="right", va="bottom")
    xg = 3.9          # above the falling curve, starting where it is highest
    ax.text(xg, train[np.argmin(np.abs(c - xg))] + 0.05, "training error",
            color=T.INK, fontsize=FS, ha="left", va="bottom")

    # generalization gap: a double arrow between the curves on the right
    xa = 5.55
    ia = int(np.argmin(np.abs(c - xa)))
    ax.annotate("", xy=(xa, test[ia] - 0.015), xytext=(xa, train[ia] + 0.015),
                arrowprops=dict(arrowstyle="<|-|>", color=T.INK, lw=1.4,
                                mutation_scale=11, shrinkA=0, shrinkB=0))
    ax.text(xa - 0.12, (test[ia] + train[ia]) / 2, "generalization\ngap",
            color=T.INK, fontsize=FS - 1, ha="right", va="center")

    # the two regimes, either side of the best complexity
    ax.text(c_best - 0.25, ymax * 0.93, "underfitting", color=T.INK,
            fontsize=FS, ha="right", va="top")
    ax.text(c_best + 0.25, ymax * 0.93, "overfitting", color=T.INK,
            fontsize=FS, ha="left", va="top")

    ax.set_xlim(0.0, c[-1] + 0.15)
    ax.set_ylim(0.0, ymax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("model complexity", fontsize=FS, labelpad=20)
    ax.set_ylabel("error", fontsize=FS)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(T.INK)
        ax.spines[side].set_linewidth(T.SW_HAIR)
    fl.save(fig, "mdl-gen-capacity-error")


FIGURES = [fig_capacity_error]


def main():
    start = len(fl.WRITTEN)
    for fn in FIGURES:
        fn()
    written = fl.WRITTEN[start:]
    print(f"Wrote {len(written)} figure(s):")
    for p in written:
        print(f"  {os.path.basename(p):36s} {os.path.getsize(p):>8,d} bytes")


if __name__ == "__main__":
    main()
