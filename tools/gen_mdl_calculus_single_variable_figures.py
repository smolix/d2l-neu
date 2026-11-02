#!/usr/bin/env python3
"""Generate the figures added to "Single Variable Calculus"
(``chapter_mdl-calculus/mdl-single-variable-calculus.md``) in the shared house
style of ``gen_mdl_figures.py``.

* ``mdl-cal-second-derivative-1d`` -- the second-derivative test as a 3x3
  grid (rows: bowl, dome, the silent case x^3 / x^4 / -x^4; columns f'', f', f).
* ``mdl-cal-newton-step`` -- one Newton step on f(x) = x^4/4 - x from x_t = 2.
* ``mdl-cal-softplus-relu`` -- softplus_beta approaching ReLU, and its slope
  sigma(beta x) filling the subdifferential [0, 1] at the corner.

All curves are computed from their formulas.  Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_calculus_single_variable_figures.py

The script is byte-idempotent (the shared ``save()`` fixes the SVG hash salt
and drops the date).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY
INK = fl.INK
FS = 13        # in-figure labels and legends
FS_TITLE = 14  # functional panel and column titles


def _bare(ax):
    """No ticks and no frame: the black cross through the origin is the axis."""
    ax.set_xticks([]); ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)


def fig_second_derivative_test():
    """Rows: f = x^2/2 (minimum), f = -x^2/2 (maximum), and the three functions
    x^3, x^4, -x^4 with f'(0) = f''(0) = 0 (inflection, minimum, maximum).
    Columns: f'', f', f.  Every curve is the true derivative of the next."""
    x = np.linspace(-1.1, 1.1, 400)
    one = np.ones_like(x)
    rows = [
        ("$f''(0)>0$:\nminimum",
         [(one, x, 0.5 * x ** 2, BLUE, None)]),
        ("$f''(0)<0$:\nmaximum",
         [(-one, -x, -0.5 * x ** 2, BLUE, None)]),
        ("$f''(0)=0$:\nundecided",
         [(6 * x, 3 * x ** 2, x ** 3, BLUE, "$x^3$"),
          (12 * x ** 2, 4 * x ** 3, x ** 4, ORANGE, "$x^4$"),
          (-12 * x ** 2, -4 * x ** 3, -x ** 4, GREEN, "$-x^4$")]),
    ]
    titles = ["$f''$", "$f'$", "$f$"]

    fig, axes = plt.subplots(3, 3, figsize=(9.6, 8.4))
    for r, (row_label, curves) in enumerate(rows):
        for c in range(3):
            ax = axes[r, c]
            ax.axhline(0, color="black", lw=1.0, zorder=1)
            ax.axvline(0, color="black", lw=1.0, zorder=1)
            ys = []
            for curve in curves:
                y, col, lab = curve[c], curve[3], curve[4]
                ax.plot(x, y, color=col, lw=2.4, zorder=3,
                        label=lab if c == 2 else None)
                ys.append(y)
            ax.plot([0], [0], "o", color="black", ms=5, zorder=4)
            lo = min(min(y.min() for y in ys), 0.0)
            hi = max(max(y.max() for y in ys), 0.0)
            span = max(hi - lo, 1.0)
            ax.set_ylim(lo - 0.15 * span, hi + 0.15 * span)
            ax.set_xlim(-1.15, 1.15)
            _bare(ax)
            if r == 0:
                ax.set_title(titles[c], fontsize=FS_TITLE, color="black")
        axes[r, 0].text(-0.12, 0.5, row_label, transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=FS, color="black")
    # every corner of the row-3 f panel holds a curve: legend to its right
    axes[2, 2].legend(loc="center left", bbox_to_anchor=(1.0, 0.5),
                      fontsize=FS, frameon=False, handlelength=1.6)
    fig.subplots_adjust(wspace=0.12, hspace=0.18)
    fl.save(fig, "mdl-cal-second-derivative-1d")


def fig_newton_step():
    """One Newton step on f(x) = x^4/4 - x from x_t = 2: the model q matches
    value, slope and curvature at x_t; its minimizer is x_{t+1} = 2 - 7/12."""
    f = lambda t: t ** 4 / 4 - t
    xt = 2.0
    ft, d1, d2 = f(xt), xt ** 3 - 1, 3 * xt ** 2          # 2, 7, 12
    q = lambda t: ft + d1 * (t - xt) + 0.5 * d2 * (t - xt) ** 2
    xn = xt - d1 / d2                                      # 1.41667
    xstar = 1.0
    xs = np.linspace(0.5, 2.4, 400)
    ybot = -1.3

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(xs, f(xs), color=BLUE, lw=2.4, zorder=3)
    ax.plot(xs, q(xs), "--", color=ORANGE, lw=2.0, zorder=2)
    for xv, yv in ((xstar, f(xstar)), (xn, q(xn)), (xt, ft)):
        ax.plot([xv, xv], [ybot, yv], ":", color=GRAY, lw=1.0, zorder=1)
    ax.plot([xt], [ft], "o", color="black", ms=6, zorder=5)
    ax.plot([xn], [q(xn)], "o", color=ORANGE, ms=6, zorder=5)
    ax.plot([xstar], [f(xstar)], "o", mfc="white", mec="black", mew=1.6,
            ms=7, zorder=5)
    ax.text(2.3, 3.0, "$f$", color="black", fontsize=FS,
            ha="left", va="center")
    ax.text(0.78, 3.6, "model $q$", color=ORANGE, fontsize=FS,
            ha="left", va="center")
    ax.set_xlim(0.5, 2.4); ax.set_ylim(ybot, 4.6)
    ax.set_xticks([xstar, xn, xt])
    ax.set_xticklabels([r"$x^\star$", "$x_{t+1}$", "$x_t$"])
    ax.tick_params(labelcolor="black", labelsize=FS)
    ax.set_yticks([])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black"); ax.spines["bottom"].set_color("black")
    fl.save(fig, "mdl-cal-newton-step")


def fig_softplus_relu():
    """Left: softplus_beta(x) = log(1 + e^{beta x}) / beta for beta = 1, 4, 16
    against ReLU.  Right: the slopes sigma(beta x) steepen into ReLU's step;
    at the corner they fill [0, 1]."""
    x = np.linspace(-1.5, 1.5, 600)
    betas = [(1, GRAY), (4, BLUE), (16, ORANGE)]
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(9.6, 3.6))

    for beta, col in betas:
        axl.plot(x, np.logaddexp(0, beta * x) / beta, color=col, lw=2.0,
                 label=rf"$\beta={beta}$")
    axl.plot(x, np.maximum(0, x), color="black", lw=2.4, label="ReLU")
    axl.set_title(r"$\mathrm{softplus}_\beta$", fontsize=FS_TITLE)
    axl.legend(loc="upper left", fontsize=FS, frameon=False)
    axl.set_xlim(-1.5, 1.5); axl.set_ylim(-0.05, 1.75)

    for beta, col in betas:
        axr.plot(x, 1 / (1 + np.exp(-beta * x)), color=col, lw=2.0)
    axr.plot([-1.5, 0], [0, 0], color="black", lw=2.4)
    axr.plot([0, 1.5], [1, 1], color="black", lw=2.4)
    axr.plot([0, 0], [0, 1], "--", color="black", lw=1.4)
    axr.text(0.12, 0.3, r"$\partial\,\mathrm{ReLU}(0)=[0,1]$", fontsize=FS,
             color="black", ha="left", va="center",
             bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    axr.set_title(r"slope $\sigma(\beta x)$", fontsize=FS_TITLE)
    axr.set_xlim(-1.5, 1.5); axr.set_ylim(-0.08, 1.12)
    axr.set_yticks([0, 0.5, 1]); axr.set_yticklabels(["0", r"$\frac{1}{2}$", "1"])

    for ax in (axl, axr):
        ax.set_xticks([-1, 0, 1])
        ax.tick_params(labelcolor="black", labelsize=FS)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("black"); ax.spines["bottom"].set_color("black")
    axl.set_yticks([0, 1])
    fig.subplots_adjust(wspace=0.25)
    fl.save(fig, "mdl-cal-softplus-relu")


FIGURES = [
    fig_second_derivative_test,
    fig_newton_step,
    fig_softplus_relu,
]


def main():
    start = len(fl.WRITTEN)
    for fn in FIGURES:
        fn()
    written = fl.WRITTEN[start:]
    print(f"\nWrote {len(written)} figures to {fl.IMG_DIR}:")
    for p in written:
        size = os.path.getsize(p)
        assert size > 0, f"empty: {p}"
        with open(p, "r", encoding="utf-8") as fh:
            assert "<svg" in fh.read(400), f"not valid SVG: {p}"
        print(f"  {os.path.basename(p):36s} {size:>8,d} bytes")


if __name__ == "__main__":
    main()
