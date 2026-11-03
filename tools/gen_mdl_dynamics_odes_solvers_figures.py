#!/usr/bin/env python3
"""Generate the new illustrative figure of "Ordinary Differential Equations and
Numerical Solvers" (``chapter_mdl-dynamics/mdl-odes-solvers.md``) in the
chapter's shared house style.

The shared style lives in ``gen_mdl_figures.py``; importing it applies the
``plt.rcParams`` (fixed ``svg.hashsalt`` and ``metadata={'Date': None}`` in
``save()`` make re-runs byte-for-byte identical) and exposes the palette.  The
section's older figures (velocity field, uniqueness fan, phase portraits,
stability regions) stay in ``gen_mdl_dynamics_figures.py``.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_dynamics_odes_solvers_figures.py

Writes ``img/mdl-dyn-solver-geometry.svg``.  Every point is computed from the
solver it shows, applied to dx/dt = x from x = 1 with one step h = 1.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, INK = fl.BLUE, fl.ORANGE, fl.INK
WHITE_BOX = dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5)
save = fl.save


def _spines(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("black")


def _tick(ax, t, x, slope, half=0.08, color=ORANGE):
    """A short segment of the given slope centred at (t, x): a measured slope."""
    ax.plot([t - half, t + half], [x - half * slope, x + half * slope],
            color=color, lw=3.0, solid_capstyle="round", zorder=4)


def fig_solver_geometry():
    """One step of size h = 1 on dx/dt = x from x = 1, for (a) forward Euler,
    (b) the midpoint method and (c) classical RK4.  The exact curve e^t is
    drawn in every panel; probe points carry short ticks of the slope the
    method measures there; the blue segment is the step actually taken."""
    f = lambda x: x
    x0, h = 1.0, 1.0
    # Euler
    x_eu = x0 + h * f(x0)                                   # 2
    # midpoint
    x_half = x0 + 0.5 * h * f(x0)                           # 1.5
    x_mid = x0 + h * f(x_half)                              # 2.5
    # RK4 probes
    k1 = f(x0)
    k2 = f(x0 + 0.5 * h * k1)
    k3 = f(x0 + 0.5 * h * k2)
    k4 = f(x0 + h * k3)
    x_rk = x0 + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)         # 2.7083
    exact = np.e

    t = np.linspace(0.0, 1.0, 200)
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.0), sharey=True,
                             gridspec_kw=dict(wspace=0.12))
    for i, ax in enumerate(axes):
        ax.plot(t, np.exp(t), color=INK, lw=1.3, zorder=2)
        ax.plot([1.0], [exact], "o", mfc="white", mec=INK, mew=1.6, ms=8,
                zorder=6)
        ax.plot([0.0], [x0], "o", color=INK, ms=6, zorder=6)
        if i < 2:
            ax.text(0.97, exact + 0.13, "exact: $2.718$", fontsize=13,
                    color=INK, ha="right", va="bottom", zorder=7)
        ax.text(0.80, 1.40, r"$x=e^t$", fontsize=13, color=INK, ha="left",
                va="center", zorder=7)
        ax.set_xlim(-0.05, 1.12)
        ax.set_ylim(0.85, 3.05)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_xlabel(r"$t$", fontsize=15)
        _spines(ax)
    axes[0].set_yticks([1, 1.5, 2, 2.5])
    axes[0].set_ylabel(r"$x$", fontsize=15)

    # (a) Euler ---------------------------------------------------------------
    ax = axes[0]
    ax.plot([0.0, 1.0], [x0, x_eu], color=BLUE, lw=2.4, zorder=3)
    ax.plot([1.0], [x_eu], "o", color=BLUE, ms=8, zorder=6)
    ax.text(1.1, x_eu - 0.16, "Euler: $2$", fontsize=13, color=INK,
            ha="right", va="top", bbox=WHITE_BOX, zorder=7)

    # (b) midpoint ------------------------------------------------------------
    ax = axes[1]
    ax.plot([0.0, 0.5], [x0, x_half], color=ORANGE, lw=1.6, ls="--", zorder=3)
    ax.plot([0.5], [x_half], "o", color=ORANGE, ms=7, zorder=5)
    _tick(ax, 0.5, x_half, f(x_half))
    ax.text(0.5, x_half - 0.22, "slope $1.5$", fontsize=13, color=INK,
            ha="center", va="top", zorder=7)
    ax.plot([0.0, 1.0], [x0, x_mid], color=BLUE, lw=2.4, zorder=3)
    ax.plot([1.0], [x_mid], "o", color=BLUE, ms=8, zorder=6)
    ax.text(1.1, x_mid - 0.58, "midpoint: $2.5$", fontsize=13, color=INK,
            ha="right", va="center", bbox=WHITE_BOX, zorder=7)

    # (c) RK4 -----------------------------------------------------------------
    ax = axes[2]
    probes = [(0.0, x0, k1), (0.5, x0 + 0.5 * h * k1, k2),
              (0.5, x0 + 0.5 * h * k2, k3), (1.0, x0 + h * k3, k4)]
    # the two half-step probes; the full-step probe to (1, 2.75) would lie on
    # top of the blue step and is shown by its point and tick only
    for tp, xp, _ in probes[1:3]:
        ax.plot([0.0, tp], [x0, xp], color=ORANGE, lw=1.4, ls="--", zorder=3)
    ax.plot([0.0, 1.0], [x0, x_rk], color=BLUE, lw=2.4, zorder=3)
    for tp, xp, k in probes:
        ax.plot([tp], [xp], "o", color=ORANGE, ms=7, zorder=5)
        _tick(ax, tp, xp, k)
    for (tp, xp, k), x_txt, y_txt, ha in (
            (probes[0], 0.13, 0.98, "left"), (probes[1], 0.60, 1.36, "left"),
            (probes[2], 0.40, 1.82, "right"), (probes[3], 0.88, 2.86, "right")):
        ax.text(x_txt, y_txt, f"${k:g}$", fontsize=13, color=INK, ha=ha,
                va="center", bbox=WHITE_BOX, zorder=7)
    ax.plot([1.0], [x_rk], "o", color=BLUE, ms=8, zorder=6)
    ax.annotate("RK4: $2.708$\nexact: $2.718$", xy=(1.0, x_rk - 0.07),
                xytext=(1.1, 1.95), fontsize=13, color=INK, ha="right",
                va="top", zorder=7, bbox=WHITE_BOX,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.8,
                                shrinkA=2, shrinkB=2))

    for ax, lab in zip(axes, "abc"):
        ax.text(0.0, 1.02, f"({lab})", transform=ax.transAxes, ha="left",
                va="bottom", fontsize=15, color=INK)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.91, bottom=0.15)
    save(fig, "mdl-dyn-solver-geometry")


FIGURES = [fig_solver_geometry]


def main():
    start = len(fl.WRITTEN)
    for fn in FIGURES:
        fn()
    for p in fl.WRITTEN[start:]:
        assert os.path.getsize(p) > 0, f"empty: {p}"
        print(f"  {os.path.basename(p):34s} {os.path.getsize(p):>9,d} bytes")


if __name__ == "__main__":
    main()
