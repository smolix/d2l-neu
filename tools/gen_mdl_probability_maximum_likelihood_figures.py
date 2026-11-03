#!/usr/bin/env python3
"""Generate the new illustrative figure of "Maximum Likelihood"
(``chapter_mdl-probability-statistics/mdl-maximum-likelihood.md``) in the
chapter's shared house style.

The shared style lives in ``gen_mdl_figures.py``; importing it applies the
``plt.rcParams`` (fixed ``svg.hashsalt`` and ``metadata={'Date': None}`` in
``save()`` make re-runs byte-for-byte identical) and exposes the palette.  The
section's older figures (``mdl-prob-mle-kl``, ``mdl-prob-mle-fisher``,
``mdl-prob-map-prior``, ``mdl-prob-elbo``) stay in
``gen_mdl_probability_figures.py``.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_probability_maximum_likelihood_figures.py

Writes ``img/mdl-prob-mle-gauss-laplace.svg`` (and the copy in ``img/auto/``
that the slides read).
"""

from __future__ import annotations

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GRAY, INK = fl.BLUE, fl.ORANGE, fl.GRAY, fl.INK


def _black_axes(ax):
    """Black left/bottom coordinate axes, no top/right frame."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("black")
    ax.tick_params(colors="black")


def fig_gauss_laplace():
    """Fitting one constant c to the data {1, 2, 3, 4, 100}.
    (a) The per-example NLLs of a Gaussian (sigma = 1) and a Laplace (s = 1)
    noise model as functions of the residual: r^2/2 grows without bound in
    slope, |r| has slope +-1 everywhere.  (b) The summed Gaussian NLL
    (1/2) sum (y_i - c)^2 is a parabola with its minimum at the mean, 22.
    (c) The summed Laplace NLL sum |y_i - c| is piecewise linear with kinks at
    the data and its minimum at the median, 3 (value 101)."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    c = np.linspace(-3, 108, 2221)
    sq = 0.5 * ((y[None, :] - c[:, None]) ** 2).sum(1)
    ab = np.abs(y[None, :] - c[:, None]).sum(1)
    c_mean, c_med = y.mean(), np.median(y)

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7))

    # (a) per-example losses against the residual
    ax = axes[0]
    r = np.linspace(-4, 4, 801)
    ax.plot(r, r ** 2 / 2, color=BLUE, lw=2.2, label=r"Gaussian: $r^2/2$")
    ax.plot(r, np.abs(r), color=ORANGE, lw=2.2, label=r"Laplace: $|r|$")
    ax.set_xlim(-4, 4)
    ax.set_ylim(0, 8)
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_yticks([0, 2, 4, 6, 8])
    ax.set_xlabel(r"residual $r = y - \hat y$", fontsize=13)
    ax.set_ylabel("per-example NLL", fontsize=13)
    ax.legend(loc="upper center", fontsize=12, frameon=False)
    ax.set_title("(a) one residual", fontsize=13)

    # (b), (c) summed NLL of a constant fit c, with a rug of the data
    # (b) starts its y-axis at 3,000 so the shallow dip to the minimum
    # 3,805 is visible; (c) starts at 0.  Each minimum's label sits in empty
    # space: inside the bowl above the dot in (b), below the rising arm in (c).
    for ax, f, col, cstar, lo, ticks, title, lab, (tx, ty, ha) in [
        (axes[1], sq, BLUE, c_mean, 3000.0, [0, 22, 50, 100],
         "(b) squared loss, summed", r"mean $= 22$", (22, 7600, "center")),
        (axes[2], ab, ORANGE, c_med, 0.0, [3, 50, 100],
         "(c) absolute loss, summed", r"median $= 3$", (12, 45, "left")),
    ]:
        fstar = float(np.interp(cstar, c, f))
        ax.plot(c, f, color=col, lw=2.2)
        hi = f.max() * 1.04
        ax.set_ylim(lo, hi)
        ax.set_xlim(c[0], c[-1])
        # vertical drop from the minimum to the axis
        ax.plot([cstar, cstar], [lo, fstar], color=GRAY, lw=1.0, ls=":")
        ax.plot([cstar], [fstar], "o", color=col, ms=7, zorder=6)
        # rug: the five data points as short ticks on the c-axis
        for yi in y:
            ax.plot([yi, yi], [lo, lo + 0.035 * (hi - lo)], color="black",
                    lw=1.4, zorder=5)
        ax.text(tx, ty, lab, color=INK, fontsize=13, ha=ha, va="center")
        ax.set_xticks(ticks)
        ax.set_xlabel(r"fitted constant $c$", fontsize=13)
        ax.set_ylabel("summed NLL", fontsize=13)
        ax.set_title(title, fontsize=13)

    for ax in axes:
        _black_axes(ax)
    fig.tight_layout()
    fl.save(fig, "mdl-prob-mle-gauss-laplace")
    shutil.copyfile(os.path.join(fl.IMG_DIR, "mdl-prob-mle-gauss-laplace.svg"),
                    os.path.join(fl.IMG_DIR, "auto",
                                 "mdl-prob-mle-gauss-laplace.svg"))


FIGURES = [fig_gauss_laplace]


def main():
    fl.WRITTEN.clear()
    for fn in FIGURES:
        fn()
    for path in fl.WRITTEN:
        print(f"wrote {path} ({os.path.getsize(path):,d} bytes)")


if __name__ == "__main__":
    main()
