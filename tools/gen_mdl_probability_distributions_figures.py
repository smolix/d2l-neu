#!/usr/bin/env python3
"""Generate the new illustrative figure of "Distributions"
(``chapter_mdl-probability-statistics/mdl-distributions.md``) in the chapter's
shared house style.

The shared style lives in ``gen_mdl_figures.py``; importing it applies the
``plt.rcParams`` (fixed ``svg.hashsalt`` and ``metadata={'Date': None}`` in
``save()`` make re-runs byte-for-byte identical) and exposes the palette.  The
section's other four figures (family tree, p.m.f. and density galleries,
Gaussian contours) stay in ``gen_mdl_probability_figures.py``.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_probability_distributions_figures.py

Writes ``img/mdl-prob-poisson-timeline.svg`` (and the copy in ``img/auto/``
that the slides read).
"""

from __future__ import annotations

import math
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
INK = fl.INK
BLUE, ORANGE = fl.BLUE, fl.ORANGE


def _copy_to_auto(name):
    """The slides inline ``@fig:<id>`` from ``img/auto/<id>.svg``."""
    shutil.copyfile(os.path.join(fl.IMG_DIR, f"{name}.svg"),
                    os.path.join(fl.IMG_DIR, "auto", f"{name}.svg"))


def fig_poisson_timeline():
    """One stream of rate-4 events seen two ways.  (a) Event times of a Poisson
    process on [0, 3] minutes (fixed generator seed; the three unit windows
    hold 4, 2 and 5 events), with the first three gaps T1, T2, T3 marked.
    (b) The Poisson(4) p.m.f. of the count per minute, P(N=0) = e^{-4}.
    (c) The survival function P(T > t) = e^{-4t} of the wait, on a log axis:
    P(T > 1) = e^{-4}, the same number as P(N = 0)."""
    rate = 4.0
    rng = np.random.default_rng(208)   # chosen so the windows hold 4, 2, 5
    gaps = rng.exponential(1 / rate, 40)
    times = np.cumsum(gaps)
    times = times[times < 3.0]
    counts = [int(((times >= i) & (times < i + 1)).sum()) for i in range(3)]

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.0),
                             gridspec_kw=dict(width_ratios=[2.6, 1, 1],
                                              wspace=0.32))
    # (a) the timeline -----------------------------------------------------
    ax = axes[0]
    for i in range(3):
        ax.fill_between([i, i + 1], 0, 0.88, lw=0,
                        color=fl.T.BLUE.tint if i % 2 == 0 else fl.T.PANEL)
        ax.text(i + 0.5, 0.98, f"$N={counts[i]}$", ha="center", va="bottom",
                fontsize=15, color=INK)
    for t in times:
        ax.plot([t, t], [0, 0.62], color=BLUE, lw=2.4,
                solid_capstyle="butt")
        ax.plot(t, 0.62, "o", color=BLUE, ms=5)
    # the first three gaps, below the axis and its tick labels
    edges = np.concatenate([[0.0], times[:3]])
    for j in range(3):
        a, b = edges[j], edges[j + 1]
        ax.annotate("", xy=(b, -0.48), xytext=(a, -0.48),
                    arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.6,
                                    shrinkA=0, shrinkB=0, mutation_scale=11))
        ax.text((a + b) / 2, -0.58, f"$T_{j + 1}$", ha="center", va="top",
                fontsize=14, color=fl.T.ORANGE.dark)
    ax.set_xlim(-0.03, 3.03); ax.set_ylim(-0.90, 1.30)
    ax.spines["bottom"].set_position(("data", 0))
    ax.spines["bottom"].set_color("black")
    ax.set_xticks([0, 1, 2, 3])
    ax.tick_params(axis="x", colors="black", labelsize=13)
    ax.set_yticks([])
    for s in ("left", "top", "right"):
        ax.spines[s].set_visible(False)
    ax.text(3.03, -0.30, "time (min)", fontsize=14, color="black",
            ha="right", va="top")

    # (b) the count per minute ---------------------------------------------
    ax = axes[1]
    ks = np.arange(11)
    pmf = np.array([math.exp(-rate) * rate ** k / math.factorial(k)
                    for k in ks])
    ax.vlines(ks, 0, pmf, color=BLUE, lw=2.2)
    ax.plot(ks, pmf, "o", color=BLUE, ms=4, clip_on=False)
    ax.vlines([0], 0, pmf[:1], color=ORANGE, lw=2.6)
    ax.plot([0], pmf[:1], "o", color=ORANGE, ms=5, clip_on=False)
    ax.text(4.5, 0.235, r"$e^{-4}\approx0.018$", fontsize=14,
            color=fl.T.ORANGE.dark, ha="left", va="center")
    ax.set_xlim(-0.6, 10.6); ax.set_ylim(0, 0.26)
    ax.set_xticks([0, 4, 8]); ax.set_yticks([0, 0.1, 0.2])
    ax.tick_params(colors="black", labelsize=13)
    ax.set_xlabel("requests per minute", fontsize=14, color="black")
    ax.set_ylabel("$P(N=k)$", fontsize=14, color="black")

    # (c) the wait --------------------------------------------------------
    ax = axes[2]
    t = np.linspace(0, 1.5, 300)
    ax.semilogy(t, np.exp(-rate * t), color=BLUE, lw=2.2)
    ax.plot([1, 1], [1e-3, math.exp(-rate)], color=fl.T.FAINT, lw=1.0,
            ls=(0, (3, 3)))
    ax.plot([0, 1], [math.exp(-rate)] * 2, color=fl.T.FAINT, lw=1.0,
            ls=(0, (3, 3)))
    ax.plot([1], [math.exp(-rate)], "o", color=ORANGE, ms=6, zorder=4)
    ax.text(1.04, 0.045, r"$e^{-4}\approx0.018$", fontsize=14,
            color=fl.T.ORANGE.dark, ha="left", va="bottom")
    ax.set_xlim(0, 1.5); ax.set_ylim(1e-3, 1.2)
    ax.set_xticks([0, 0.5, 1, 1.5])
    ax.set_xticklabels(["0", "0.5", "1", "1.5"])
    ax.tick_params(colors="black", labelsize=13)
    ax.set_xlabel("wait $t$ (min)", fontsize=14, color="black")
    ax.set_ylabel(r"$P(T>t)$", fontsize=14, color="black")
    for a in axes[1:]:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fl.save(fig, "mdl-prob-poisson-timeline")
    _copy_to_auto("mdl-prob-poisson-timeline")


FIGURES = [fig_poisson_timeline]


def main():
    before = len(fl.WRITTEN)
    for fn in FIGURES:
        fn()
    for p in fl.WRITTEN[before:]:
        assert os.path.getsize(p) > 0, f"empty: {p}"
        print("wrote", os.path.relpath(p))


if __name__ == "__main__":
    main()
