#!/usr/bin/env python3
"""Generate the illustrative figures of "Uniform Convergence and
Generalization"
(``chapter_mdl-probability-statistics/mdl-generalization-bounds.md``)
in the chapter's shared house style.

The shared style lives in ``gen_mdl_figures.py``; importing it applies the
``plt.rcParams`` (fixed ``svg.hashsalt`` and ``metadata={'Date': None}`` in
``save()`` make re-runs byte-for-byte identical) and exposes the palette.
Both figures are exact computations (binomial probabilities, an enumeration
of all 256 sign patterns), so no random numbers are drawn.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_probability_generalization_bounds_figures.py

Writes ``img/mdl-prob-gen-best-of-many.svg`` and
``img/mdl-prob-gen-rademacher.svg`` (and the copies in ``img/auto/`` that
the slides read).
"""

from __future__ import annotations

import itertools
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

from scipy.stats import binom  # noqa: E402

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, INK = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.INK
LIGHT = fl.LIGHT


def _black_axes(ax):
    """Left/bottom spines in black, no top/right spines."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("black")
    ax.tick_params(colors="black", labelsize=13)


def _save_both(fig, name):
    fl.save(fig, name)
    shutil.copyfile(os.path.join(fl.IMG_DIR, f"{name}.svg"),
                    os.path.join(fl.IMG_DIR, "auto", f"{name}.svg"))


def fig_best_of_many():
    """Expected histogram of the head fractions of 1,000 fair coins flipped
    100 times each (bar height 1000 * Binom(100, 1/2).pmf(k) at k/100), on a
    log scale so that the tail is visible.  Marks: the single-coin Hoeffding
    95% band [0.364, 0.636] (shaded), the union-bound band edges 0.270 and
    0.730 for 1,000 coins (dashed), and the typical position 0.66 of the best
    coin (median of the maximum of 1,000 binomials)."""
    k_coins, m, delta = 1000, 100, 0.05
    ks = np.arange(20, 81)
    counts = k_coins * binom.pmf(ks, m, 0.5)
    one = np.sqrt(np.log(2 / delta) / (2 * m))
    many = np.sqrt(np.log(2 * k_coins / delta) / (2 * m))

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ymin, ymax = 0.01, 900
    ax.axvspan(0.5 - one, 0.5 + one, color=LIGHT, alpha=0.35, lw=0,
               zorder=0)
    ax.bar(ks / m, counts, width=0.0085, color=fl.T.BLUE.tint,
           edgecolor=BLUE, linewidth=0.8, zorder=2)
    for x in (0.5 - many, 0.5 + many):
        ax.plot([x, x], [ymin, 330], "--", color=ORANGE, lw=1.8, zorder=3)
    for (x0, x1), y, text, col in (
            ((0.5 - one, 0.5 + one), 115, "one coin, 95%", INK),
            ((0.5 - many, 0.5 + many), 330, "all 1,000 coins, 95%", ORANGE)):
        ax.annotate("", xy=(x0, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="<->", color=col, lw=1.2,
                                    shrinkA=0, shrinkB=0))
        ax.text(0.5, y * 1.25, text, color=col, fontsize=14, ha="center",
                va="bottom")
    # the best coin: typical value 0.66
    best = 0.66
    y_best = k_coins * binom.pmf(66, m, 0.5)
    ax.plot([best], [y_best], "o", color=GREEN, ms=9, zorder=5)
    ax.annotate("best coin,\ntypically 0.66", xy=(best, y_best * 1.35),
                xytext=(best, 14), color=INK, fontsize=14, ha="center",
                va="bottom",
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.9,
                                shrinkA=2, shrinkB=2))
    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(0.24, 0.76)
    ax.set_yticks([0.01, 0.1, 1, 10, 100])
    ax.set_yticklabels(["0.01", "0.1", "1", "10", "100"])
    ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7])
    ax.set_xlabel("fraction of heads in 100 flips", fontsize=15)
    ax.set_ylabel("expected number of coins", fontsize=14)
    ax.minorticks_off()
    _black_axes(ax)
    _save_both(fig, "mdl-prob-gen-best-of-many")


def fig_rademacher():
    """(a) Eight unit vectors x_i at angles 10 + 45 i degrees with one sign
    pattern (+,+,-,+,-,-,+,-): the signed sum s = sum_i eps_i x_i =
    (1.83, 1.16) and the maximizer w* = s / ||s|| of the linear correlation;
    the best correlation is ||s|| / 8 = 0.27.  (b) The exact distribution of
    ||s|| / 8 over all 256 sign patterns (10 distinct values): mean 0.316
    (the empirical Rademacher complexity) against the Cauchy--Schwarz--Jensen
    bound 1/sqrt(8) = 0.354."""
    n = 8
    ang = np.deg2rad(10 + 45 * np.arange(n))
    X = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    eps = np.array([1, 1, -1, 1, -1, -1, 1, -1], float)
    s = eps @ X
    w_star = s / np.linalg.norm(s)

    xlim, ylim = (-1.45, 2.25), (-1.45, 1.65)
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(11.6, 4.5),
        gridspec_kw=dict(width_ratios=[1.3, 1.0], wspace=0.22))

    # (a) one sign draw
    tt = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(tt), np.sin(tt), color=LIGHT, lw=1.0, zorder=0)
    fl.axis_cross(ax, (xlim[0], xlim[1] - 0.05), (ylim[0], ylim[1] - 0.05),
                  color="black")
    for xi, e in zip(X, eps):
        ax.plot([0, xi[0]], [0, xi[1]], color=LIGHT, lw=1.0, zorder=1)
        if e > 0:
            ax.plot(*xi, "o", ms=9, color=BLUE, zorder=4)
        else:
            ax.plot(*xi, "o", ms=9, mfc="white", mec=BLUE, mew=1.8,
                    zorder=4)
        ax.text(*(1.2 * xi), "+" if e > 0 else "−", color=INK,
                fontsize=16, ha="center", va="center")
    fl.arrow(ax, (0, 0), s, color=ORANGE, lw=3.0, mut=18)
    fl.arrow(ax, (0, 0), w_star, color=GREEN, lw=3.0, mut=16)
    perp = np.array([-w_star[1], w_star[0]])
    ax.text(*(s + 0.08 * w_star + 0.12 * perp),
            r"$\mathbf{s}=\Sigma_i\,\varepsilon_i\mathbf{x}_i$",
            color=ORANGE, fontsize=15, ha="center", va="bottom")
    ax.text(*(0.5 * w_star + 0.3 * perp), r"$\mathbf{w}^*$", color=GREEN,
            fontsize=15, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"))
    ax.text(xlim[1] - 0.02, ylim[0] + 0.05,
            rf"$\frac{{1}}{{n}}\|\mathbf{{s}}\|={np.linalg.norm(s) / n:.2f}$",
            color=INK, fontsize=15, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=LIGHT))
    fl.clean_axes(ax, lim=(xlim, ylim), hide=True)

    # (b) all 256 sign patterns
    vals = np.array([np.linalg.norm(np.array(e) @ X) / n
                     for e in itertools.product([-1, 1], repeat=n)])
    uniq, cnt = np.unique(vals.round(9), return_counts=True)
    prob = cnt / cnt.sum()
    mean, bound = vals.mean(), 1 / np.sqrt(n)
    bx.vlines(uniq, 0, prob, color=BLUE, lw=3.0, zorder=3)
    bx.plot(uniq, prob, "o", color=BLUE, ms=6, zorder=4)
    top = 0.36
    bx.plot([mean, mean], [0, top * 0.86], "--", color=GREEN, lw=2.0,
            zorder=2)
    bx.plot([bound, bound], [0, top * 0.86], ":", color=ORANGE, lw=2.4,
            zorder=2)
    bx.text(mean - 0.01, top * 0.88,
            rf"$\widehat{{\mathfrak{{R}}}}_S={mean:.3f}$", color=GREEN,
            fontsize=14, ha="right", va="bottom")
    bx.text(bound + 0.01, top * 0.88, rf"$Br/\sqrt{{n}}={bound:.3f}$",
            color=ORANGE, fontsize=14, ha="left", va="bottom")
    bx.set_xlim(-0.02, 0.7)
    bx.set_ylim(0, top)
    bx.set_xticks([0, 0.2, 0.4, 0.6])
    bx.set_yticks([0, 0.1, 0.2, 0.3])
    bx.set_xlabel(r"best correlation $\frac{1}{n}\|\mathbf{s}\|$",
                  fontsize=15)
    bx.set_ylabel("probability", fontsize=14)
    _black_axes(bx)

    for a, tag in ((ax, "(a)"), (bx, "(b)")):
        a.text(0.0, 1.0, tag, transform=a.transAxes, fontsize=14,
               color=INK, ha="left", va="bottom")
    _save_both(fig, "mdl-prob-gen-rademacher")


FIGURES = [fig_best_of_many, fig_rademacher]


def main():
    fl.WRITTEN.clear()
    for fn in FIGURES:
        fn()
    for path in fl.WRITTEN:
        print(f"wrote {path} ({os.path.getsize(path):,d} bytes)")


if __name__ == "__main__":
    main()
