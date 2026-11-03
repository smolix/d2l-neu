#!/usr/bin/env python3
"""Generate the new illustrative figures of "Numerical Stability and
Conditioning" (``chapter_mdl-optimization/mdl-numerical-stability-conditioning.md``)
in the chapter's shared house style.

The shared style lives in ``gen_mdl_figures.py``; importing it applies the
``plt.rcParams`` (fixed ``svg.hashsalt`` and ``metadata={'Date': None}`` in
``save()`` make re-runs byte-for-byte identical) and exposes the palette.  The
section's two older figures (``mdl-opt-fp-number-line`` and
``mdl-opt-conditioning-ellipse``) stay in ``gen_mdl_optimization_figures.py``.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_optimization_numerical_figures.py

Writes ``img/mdl-opt-two-lines.svg`` and ``img/mdl-opt-softplus-max.svg``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, INK = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.INK

LBL = 13          # the chapter's in-figure label size
BOX = dict(boxstyle="round,pad=0.15", fc="white", ec="none")


def fig_two_lines():
    """A linear system is sensitive when its two lines are nearly parallel.

    (a) x1 + x2 = 2 and x1 - x2 = 0 meet at (1, 1); shifting the second
        right-hand side by 0.1 moves the crossing to (1.05, 0.95)
        (amplification 1, kappa = 1).
    (b) x1 + x2 = 2 and x1 + 1.1 x2 = 2.1 also meet at (1, 1); the same shift
        of b2 by 0.1 moves the crossing to (0, 2) (amplification 29,
        kappa ~ 42).  All lines and crossings are exact.
    Both panels share one window, so they render at the same size and scale.
    """
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.6, 4.6))
    lim = ((-0.45, 2.65), (-0.45, 2.65))
    xs = np.linspace(-0.45, 2.65, 2)

    def common(ax, kappa_label):
        fl.axis_cross(ax, (-0.4, 2.6), (-0.4, 2.6), color="black")
        ax.text(2.55, -0.08, r"$x_1$", color="black", fontsize=LBL,
                ha="right", va="top")
        ax.text(0.07, 2.58, r"$x_2$", color="black", fontsize=LBL,
                ha="left", va="top")
        ax.plot(xs, 2 - xs, color=BLUE, lw=2.0)                 # x1 + x2 = 2
        ax.text(2.55, 2.55, kappa_label, color="black", fontsize=LBL + 1,
                ha="right", va="top")
        fl.clean_axes(ax, lim=lim, hide=True)

    # ---------------- (a) perpendicular lines ---------------- #
    common(axa, r"$\kappa=1$")
    xo = np.linspace(-0.45, 2.25, 2)          # stop below the kappa label
    axa.plot(xo, xo, color=ORANGE, lw=2.0)                      # x1 - x2 = 0
    axa.plot(xo + 0.1, xo, color=ORANGE, lw=1.8, ls=(0, (5, 3)))  # = 0.1
    axa.text(0.55, 2.0, r"$x_1+x_2=2$", color=BLUE, fontsize=LBL,
             ha="left", va="top", bbox=BOX)
    axa.text(2.0, 1.55, r"$x_1-x_2=0$", color=ORANGE, fontsize=LBL,
             ha="left", va="top", bbox=BOX)
    axa.text(1.55, 0.98, r"$b_2+0.1$", color=ORANGE, fontsize=LBL,
             ha="left", va="top", bbox=BOX)
    axa.plot(1, 1, "o", color="black", ms=6, zorder=5)
    axa.plot(1.05, 0.95, "o", color=ORANGE, ms=6, zorder=5)
    axa.text(0.86, 0.99, r"$(1,1)$", color="black", fontsize=LBL,
             ha="right", va="center", bbox=BOX)
    axa.text(1.05, 0.42, r"$(1.05,0.95)$", color=ORANGE, fontsize=LBL,
             ha="center", va="top", bbox=BOX)

    # ---------------- (b) nearly parallel lines ---------------- #
    common(axb, r"$\kappa\approx42$")
    axb.plot(xs, (2.1 - xs) / 1.1, color=ORANGE, lw=2.0)        # = 2.1
    axb.plot(xs, (2.2 - xs) / 1.1, color=ORANGE, lw=1.8,
             ls=(0, (5, 3)))                                     # = 2.2
    axb.text(1.40, 0.36, r"$x_1+x_2=2$", color=BLUE, fontsize=LBL,
             ha="right", va="top", bbox=BOX)
    axb.text(1.30, 1.06, r"$x_1+1.1x_2=2.1$", color=ORANGE, fontsize=LBL,
             ha="left", va="bottom", bbox=BOX)
    axb.text(2.0, 0.56, r"$b_2+0.1$", color=ORANGE, fontsize=LBL,
             ha="left", va="bottom", bbox=BOX)
    fl.arrow(axb, (1, 1), (0.03, 1.97), color="black", lw=1.4, mut=12)
    axb.plot(1, 1, "o", color="black", ms=6, zorder=5)
    axb.plot(0, 2, "o", color=ORANGE, ms=6, zorder=5)
    axb.text(0.12, 2.12, r"$(0,2)$", color=ORANGE, fontsize=LBL,
             ha="left", va="bottom", bbox=BOX)
    axb.text(0.66, 1.64, r"$\|\Delta\mathbf{x}\|/\|\mathbf{x}\|=1$",
             color="black", fontsize=LBL, ha="left", va="bottom", bbox=BOX)
    axb.text(0.88, 0.90, r"$(1,1)$", color="black", fontsize=LBL,
             ha="right", va="top", bbox=BOX)

    fl.save(fig, "mdl-opt-two-lines")


def fig_softplus_max():
    """Log-sum-exp is a soft maximum: lse(0, t) = log(1 + e^t) against
    max(0, t) on t in [-3, 3]; the gap is largest at t = 0, where it is
    log 2 (the sandwich bound with n = 2), and vanishes as |t| grows."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    t = np.linspace(-3, 3, 301)
    # only the t-axis: the curve's height above it is the quantity of
    # interest, and a vertical axis at t = 0 would cover the log 2 bracket
    ax.annotate("", xy=(3.3, 0), xytext=(-3.2, 0),
                arrowprops=dict(arrowstyle="->", color="black", lw=0.9))
    ax.plot([0, 0], [-0.08, 0.0], color="black", lw=0.9)
    ax.text(0, -0.14, r"$0$", color="black", fontsize=LBL, ha="center",
            va="top")
    ax.plot(t, np.maximum(0, t), color=GRAY, lw=2.0, ls=(0, (5, 3)))
    ax.plot(t, np.logaddexp(0, t), color=BLUE, lw=2.4)
    # the gap log 2 at t = 0: a short bracket from the kink to the curve
    ax.annotate("", xy=(0, np.log(2)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="|-|,widthA=0.3,widthB=0.3",
                                color=ORANGE, lw=1.6, shrinkA=0, shrinkB=0))
    ax.text(-0.10, 0.27, r"$\log 2$", color=ORANGE, fontsize=LBL,
            ha="right", va="center")
    ax.text(-3.0, 0.75, r"$\mathrm{lse}(0,t)=\log(1+e^t)$", color=BLUE,
            fontsize=LBL, ha="left", va="bottom", bbox=BOX)
    ax.text(1.75, 1.45, r"$\max(0,t)$", color="black", fontsize=LBL,
            ha="left", va="top", bbox=BOX)
    ax.text(3.25, -0.08, r"$t$", color="black", fontsize=LBL, ha="right",
            va="top")
    fl.clean_axes(ax, lim=((-3.25, 3.4), (-0.35, 3.15)), hide=True,
                  equal=False)
    fl.save(fig, "mdl-opt-softplus-max")


FIGURES = [fig_two_lines, fig_softplus_max]


def main():
    fl.WRITTEN.clear()
    for fn in FIGURES:
        fn()
    for p in fl.WRITTEN:
        assert os.path.getsize(p) > 0, f"empty: {p}"
        print(f"  {os.path.basename(p):34s} {os.path.getsize(p):>8,d} bytes")


if __name__ == "__main__":
    main()
