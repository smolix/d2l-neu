#!/usr/bin/env python3
"""Generate the new illustrative figure of "Random Variables"
(``chapter_mdl-probability-statistics/mdl-random-variables.md``) in the
chapter's shared house style.

The shared style lives in ``gen_mdl_figures.py``; importing it applies the
``plt.rcParams`` (fixed ``svg.hashsalt`` and ``metadata={'Date': None}`` in
``save()`` make re-runs byte-for-byte identical) and exposes the palette.  The
section's older figures (density, c.d.f., inverse transform, Chebyshev,
marginal, conditional slice, covariance, correlation) stay in
``gen_mdl_probability_figures.py``.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_probability_random_variables_figures.py

Writes ``img/mdl-prob-change-of-variables.svg`` (and the copy in
``img/auto/`` that the slides read).
"""

from __future__ import annotations

import math
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GRAY, INK = fl.BLUE, fl.ORANGE, fl.GRAY, fl.INK


def _normal_pdf(x):
    return np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)


def _Phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def fig_change_of_variables():
    """Change of variables for Y = e^X with X ~ N(0,1).  The main panel is the
    graph of g(x) = e^x; the bottom strip is p_X, the left strip is p_Y drawn
    sideways.  The interval [0.9, 1.3] under p_X maps to [e^0.9, e^1.3] =
    [2.46, 3.67] under p_Y; both shaded regions carry the same probability
    0.087, so where g stretches the interval, p_Y is lower by g'(x)."""
    x0, x1 = 0.9, 1.3
    y0, y1 = math.exp(x0), math.exp(x1)
    mass = _Phi(x1) - _Phi(x0)
    xs = np.linspace(-2.2, 1.8, 400)
    ys = np.linspace(1e-3, 6.0, 600)
    p_x = _normal_pdf(xs)
    p_y = _normal_pdf(np.log(ys)) / ys

    fig = plt.figure(figsize=(7.4, 5.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 3.0],
                          height_ratios=[3.0, 1.0], wspace=0.16, hspace=0.08)
    ax = fig.add_subplot(gs[0, 1])
    axb = fig.add_subplot(gs[1, 1], sharex=ax)
    axl = fig.add_subplot(gs[0, 0], sharey=ax)

    # main panel: the map g(x) = e^x and the guides of the interval
    ax.plot(xs, np.exp(xs), color=BLUE, lw=2.4, zorder=3)
    for xv, yv in ((x0, y0), (x1, y1)):
        ax.plot([xv, xv], [0, yv], "--", color=GRAY, lw=1.0, zorder=1)
        ax.plot([-2.2, xv], [yv, yv], "--", color=GRAY, lw=1.0, zorder=1)
    ax.text(1.25, 4.35, r"$y=g(x)=e^{x}$", color=BLUE, fontsize=15,
            ha="right", va="center")
    # the equality of the two shaded areas, stated once in the empty corner
    ax.text(-2.05, 5.85, "both shaded areas:\n" + rf"$P(0.9<X<1.3)={mass:.3f}$",
            ha="left", va="top", fontsize=13, color=INK, linespacing=1.5)
    ax.set_xlim(-2.2, 1.8)
    ax.set_ylim(0, 6.0)
    ax.set_yticks([0, 2, 4, 6])
    # the y name sits above the axis, clear of the sideways density
    ax.text(-2.2, 6.2, r"$y$", fontsize=14, ha="center", va="bottom",
            clip_on=False)
    ax.tick_params(labelbottom=False, bottom=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # bottom strip: p_X with the interval [x0, x1] shaded
    axb.plot(xs, p_x, color=ORANGE, lw=2.2)
    m = (xs >= x0) & (xs <= x1)
    axb.fill_between(xs[m], 0, p_x[m], color=ORANGE, alpha=0.35, lw=0)
    axb.annotate("", xy=(x1, 0.30), xytext=(x0, 0.30),
                 arrowprops=dict(arrowstyle="<->", color=INK, lw=1.2,
                                 shrinkA=0, shrinkB=0))
    axb.text((x0 + x1) / 2, 0.33, r"$\Delta x$", ha="center", va="bottom",
             fontsize=14, color=INK)
    axb.text(-1.25, 0.30, r"$p_X$", color=ORANGE, fontsize=15, ha="right",
             va="center")
    axb.set_ylim(0, 0.46)
    axb.set_yticks([])
    axb.set_xticks([-2, -1, 0, 1])
    axb.set_xlabel(r"$x$", fontsize=14)
    for s in ("top", "right", "left"):
        axb.spines[s].set_visible(False)

    # left strip: p_Y drawn sideways (density grows to the left)
    axl.plot(p_y, ys, color=ORANGE, lw=2.2)
    k = (ys >= y0) & (ys <= y1)
    axl.fill_betweenx(ys[k], 0, p_y[k], color=ORANGE, alpha=0.35, lw=0)
    axl.annotate("", xy=(0.27, y1), xytext=(0.27, y0),
                 arrowprops=dict(arrowstyle="<->", color=INK, lw=1.2,
                                 shrinkA=0, shrinkB=0))
    axl.text(0.30, (y0 + y1) / 2, r"$\Delta y$", ha="right", va="center",
             fontsize=14, color=INK)
    axl.text(0.45, 1.25, r"$p_Y$", color=ORANGE, fontsize=15, ha="center",
             va="center")
    axl.set_xlim(0.75, 0)                # inverted: density grows leftwards
    axl.set_xticks([])
    axl.tick_params(labelleft=False, left=False)
    for s in ("top", "left", "bottom"):
        axl.spines[s].set_visible(False)
    for a in (ax, axb, axl):
        for s in a.spines.values():
            s.set_color("black")
    fl.save(fig, "mdl-prob-change-of-variables")
    shutil.copyfile(
        os.path.join(fl.IMG_DIR, "mdl-prob-change-of-variables.svg"),
        os.path.join(fl.IMG_DIR, "auto", "mdl-prob-change-of-variables.svg"))


FIGURES = [fig_change_of_variables]


def main():
    fl.WRITTEN.clear()
    for fn in FIGURES:
        fn()
    for path in fl.WRITTEN:
        print(f"wrote {path} ({os.path.getsize(path):,d} bytes)")


if __name__ == "__main__":
    main()
