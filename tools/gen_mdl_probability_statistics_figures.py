#!/usr/bin/env python3
"""Generate the new illustrative figure of "Statistics"
(``chapter_mdl-probability-statistics/mdl-statistics.md``) in the chapter's
shared house style.

The shared style lives in ``gen_mdl_figures.py``; importing it applies the
``plt.rcParams`` (fixed ``svg.hashsalt`` and ``metadata={'Date': None}`` in
``save()`` make re-runs byte-for-byte identical) and exposes the palette.  The
section's older figures (sampling distribution, U-curve, significance, power,
coverage, bootstrap, type I/II matrix) stay in
``gen_mdl_probability_figures.py``.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_probability_statistics_figures.py

Writes ``img/mdl-prob-stat-mse-triangle.svg``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

from matplotlib.patches import Arc  # noqa: E402

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, INK = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.INK


def fig_mse_triangle():
    """The bias-variance decomposition as Pythagoras.  Each estimator of
    theta = 1 (n = 30 draws of N(1, 16)) is the point (|bias|, se); its
    distance from the origin is sqrt(MSE).  The shrunk mean 0.65 * mean sits at
    (0.35, 0.475): a right triangle with hypotenuse 0.59.  The sample mean sits
    at (0, 0.730) and the median at (0, 0.894) (its finite-sample standard
    error at n = 30).  A dashed quarter circle of radius 0.730 is the set of
    estimators as good as the sample mean; the shrunk mean lies inside it."""
    v = 16.0 / 30.0
    se_mean = np.sqrt(v)                 # 0.730
    bias_s, se_s = 0.35, 0.65 * se_mean  # 0.35, 0.475
    rmse_s = np.hypot(bias_s, se_s)      # 0.590
    se_med = np.sqrt(0.80)               # 0.894, measured at n = 30

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    fl.axis_cross(ax, (0, 1.05), (0, 1.02), color=INK)

    # equal-MSE contour through the sample mean
    ax.add_patch(Arc((0, 0), 2 * se_mean, 2 * se_mean, theta1=0, theta2=90,
                     color=GRAY, lw=1.0, ls="--"))
    # the sample mean and the median: points on the vertical axis
    ax.plot([0], [se_mean], "o", color=BLUE, ms=7, zorder=6)
    ax.text(-0.04, se_mean, r"$\bar x$: se $=0.73$", color=INK, fontsize=13,
            ha="right", va="center")
    ax.plot([0], [se_med], "o", color=GRAY, ms=7, zorder=6)
    ax.text(-0.04, se_med, r"median: se $=0.89$", color=INK, fontsize=13,
            ha="right", va="center")

    # the shrunk mean: two legs and the hypotenuse
    ax.plot([0, bias_s], [0, 0], color=ORANGE, lw=3.0, solid_capstyle="butt",
            zorder=4)
    ax.plot([bias_s, bias_s], [0, se_s], color=BLUE, lw=2.2, zorder=4)
    ax.plot([0, bias_s], [0, se_s], color=GREEN, lw=3.0, zorder=5)
    ax.plot([bias_s], [se_s], "o", color=GREEN, ms=7, zorder=6)
    fl.right_angle(ax, (bias_s, 0), (-1, 0), (0, 1), size=0.035, color=GRAY)
    ax.text(bias_s / 2, -0.035, r"$|\mathrm{bias}|=0.35$", color=ORANGE,
            fontsize=13, ha="center", va="top")
    # low on the leg, where the dashed arc is still far to the right
    ax.text(bias_s + 0.03, 0.3 * se_s, r"se $=0.47$", color=BLUE, fontsize=13,
            ha="left", va="center")
    # hypotenuse label: along the segment, pushed 16 pt up-left of it
    ang = np.degrees(np.arctan2(se_s, bias_s))
    nrm = np.array([-se_s, bias_s]) / rmse_s
    ax.annotate(r"$\sqrt{\mathrm{MSE}}=0.59$", xy=(0.6 * bias_s, 0.6 * se_s),
                xytext=tuple(16 * nrm), textcoords="offset points",
                color=GREEN, fontsize=13, ha="center", va="center",
                rotation=ang, rotation_mode="anchor")
    ax.text(bias_s, se_s + 0.035, r"$0.65\,\bar x$", color=INK,
            fontsize=13, ha="center", va="bottom")

    ax.text(1.05, -0.035, r"$|\mathrm{bias}|$", color=INK, fontsize=13,
            ha="right", va="top")
    ax.text(0.0, 1.06, "standard error", color=INK, fontsize=13,
            ha="center", va="bottom")
    ax.text(-0.06, -0.035, r"$\theta$", color=INK, fontsize=14, ha="right",
            va="top")
    fl.clean_axes(ax, lim=((-0.62, 1.08), (-0.12, 1.12)), hide=True)
    fl.save(fig, "mdl-prob-stat-mse-triangle")


FIGURES = [fig_mse_triangle]


def main():
    fl.WRITTEN.clear()
    for fn in FIGURES:
        fn()
    for path in fl.WRITTEN:
        print(f"wrote {path} ({os.path.getsize(path):,d} bytes)")


if __name__ == "__main__":
    main()
