#!/usr/bin/env python3
"""Generate the new illustrative figure of "Stochastic Differential
Equations" (``chapter_mdl-dynamics/mdl-sdes.md``) in the chapter's shared
house style.

The shared style lives in ``gen_mdl_figures.py``; importing it applies the
``plt.rcParams`` (fixed ``svg.hashsalt`` and ``metadata={'Date': None}`` in
``save()`` make re-runs byte-for-byte identical) and exposes the palette.  The
section's older figures (``mdl-dyn-sde-paths``, ``mdl-dyn-brownian-paths``,
``mdl-dyn-qv-convergence``, ``mdl-dyn-ou-mean-reversion``,
``mdl-dyn-strong-weak``) stay in ``gen_mdl_dynamics_figures.py``.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_dynamics_sdes_figures.py

Writes ``img/mdl-dyn-ito-jensen.svg``.  Nothing is random.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GRAY, INK = fl.BLUE, fl.ORANGE, fl.GRAY, fl.INK
WHITE_BOX = dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5)


def fig_ito_jensen():
    """The Ito correction as Jensen's gap.  For phi(x) = x^2, x = 0.7 and
    dt = 0.25, one step of +-sqrt(dt) lands at 0.2 or 1.2, where phi is 0.04
    and 1.44; their average 0.74 lies above phi(0.7) = 0.49 by
    (1/2) phi'' dt = 0.25."""
    x, dt = 0.7, 0.25
    lo, hi = x - np.sqrt(dt), x + np.sqrt(dt)
    phi = lambda u: u ** 2
    avg = 0.5 * (phi(lo) + phi(hi))

    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    xs = np.linspace(-0.3, 1.37, 300)
    fl.axis_cross(ax, (-0.3, 1.6), (-0.15, 1.9), color=INK)
    ax.plot(xs, phi(xs), color=BLUE, lw=2.2, zorder=2)
    ax.text(1.27, 1.30, r"$\phi(x)=x^2$", color=INK, fontsize=14,
            ha="left", va="top", bbox=WHITE_BOX, zorder=5)
    ax.plot([lo, hi], [phi(lo), phi(hi)], color=GRAY, lw=1.4, ls="--",
            zorder=1)
    ax.plot([lo, lo], [0, phi(lo)], color=GRAY, lw=0.8, ls=":", zorder=1)
    ax.plot([x, x], [0, phi(x)], color=GRAY, lw=0.8, ls=":", zorder=1)
    ax.plot([lo, hi], [phi(lo), phi(hi)], "o", color=BLUE, ms=6, zorder=4)
    ax.plot([x], [phi(x)], "o", color=BLUE, ms=6, zorder=4)
    ax.plot([x], [avg], "o", color=ORANGE, ms=6, zorder=4)
    ax.annotate("", xy=(x, avg), xytext=(x, phi(x)),
                arrowprops=dict(arrowstyle="<|-|>", color=ORANGE, lw=1.6,
                                shrinkA=3, shrinkB=3, mutation_scale=12),
                zorder=3)
    ax.text(x + 0.17, 0.5 * (avg + phi(x)),
            r"$\frac{1}{2}\phi''\,\Delta t=0.25$", color=INK, fontsize=14,
            ha="left", va="center", bbox=WHITE_BOX, zorder=5)
    # the two marked heights: phi(x) below the curve, the average above the
    # chord, both clear of every line
    ax.text(x + 0.06, phi(x) - 0.06, r"$\phi(x)=0.49$", color=INK, fontsize=13,
            ha="left", va="top", bbox=WHITE_BOX, zorder=5)
    ax.text(x - 0.05, avg + 0.02, "average $0.74$", color=INK,
            fontsize=13, ha="right", va="bottom", bbox=WHITE_BOX, zorder=5)
    ax.set_xticks([lo, x, hi])
    ax.set_xticklabels([r"$x-\sqrt{\Delta t}$", r"$x$", r"$x+\sqrt{\Delta t}$"],
                       fontsize=14)
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0, pad=6, labelcolor=INK)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlim(-0.3, 1.6)
    ax.set_ylim(-0.15, 1.9)
    ax.set_aspect("auto")
    fl.save(fig, "mdl-dyn-ito-jensen")


FIGURES = [fig_ito_jensen]


def main():
    start = len(fl.WRITTEN)
    for fn in FIGURES:
        fn()
    for p in fl.WRITTEN[start:]:
        assert os.path.getsize(p) > 0, f"empty: {p}"
        print(f"  wrote {os.path.basename(p)}")


if __name__ == "__main__":
    main()
