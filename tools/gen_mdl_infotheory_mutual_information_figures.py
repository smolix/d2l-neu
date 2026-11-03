#!/usr/bin/env python3
"""Generate the new illustrative figures of "Mutual Information"
(``chapter_mdl-information-theory/mdl-mutual-information.md``) in the
chapter's shared house style.

The shared style lives in ``gen_mdl_figures.py``; importing it applies the
``plt.rcParams`` (fixed ``svg.hashsalt`` and ``metadata={'Date': None}`` in
``save()`` make re-runs byte-for-byte identical) and exposes the palette.  The
section's older figures (``mdl-it-mi-overlap``, ``mdl-it-capacity-rd``,
``mdl-it-ib-tradeoff``) stay in ``gen_mdl_infotheory_figures.py``.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_infotheory_mutual_information_figures.py

Writes ``img/mdl-it-mi-corr-vs-mi.svg`` and ``img/mdl-it-bsc-chain.svg``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

from matplotlib.patches import FancyBboxPatch  # noqa: E402

np, plt = fl.np, fl.plt
BLUE, ORANGE, GRAY, INK = fl.BLUE, fl.ORANGE, fl.GRAY, fl.INK
save = fl.save


def fig_mi_corr_vs_mi():
    """Three scatters of 1,500 points: (a) the Gaussian pair at rho = 0.9
    (I = 0.83 nats); (b) X ~ N(0,1), Y = X^2 + 0.3 eps (rho = 0, I = 1.14 nats
    by quadrature); (c) an independent pair (rho = 0, I = 0).  Correlation
    cannot tell (b) from (c); mutual information ranks (b) above (a).  The
    generator may seed (figures are fixed artwork; notebooks are not)."""
    rng = np.random.default_rng(0)
    n = 1500
    x = rng.standard_normal((3, n))
    e = rng.standard_normal((3, n))
    panels = [
        (x[0], 0.9 * x[0] + np.sqrt(1 - 0.81) * e[0], (-3.5, 3.5),
         r"$\rho=0.90$, $I=0.83$ nats"),
        (x[1], x[1] ** 2 + 0.3 * e[1], (-1.0, 9.0),
         r"$\rho=0$, $I=1.14$ nats"),
        (x[2], e[2], (-3.5, 3.5), r"$\rho=0$, $I=0$"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6),
                             gridspec_kw=dict(wspace=0.32))
    for ax, (u, v, ylim, title) in zip(axes, panels):
        ax.scatter(u, v, s=6, color=BLUE, alpha=0.35, lw=0, rasterized=False)
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(*ylim)
        ax.set_xticks([-3, 0, 3])
        ax.set_xlabel(r"$x$", fontsize=14)
        ax.set_ylabel(r"$y$", fontsize=14)
        ax.set_title(title, fontsize=14)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("black")
    save(fig, "mdl-it-mi-corr-vs-mi")


def _box(ax, xy, text, color):
    """A rounded node of the chain diagram, centred at ``xy``."""
    w, h = 1.0, 0.72
    ax.add_patch(FancyBboxPatch((xy[0] - w / 2, xy[1] - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="white", ec=color, lw=2.0, zorder=3))
    ax.text(xy[0], xy[1], text, ha="center", va="center", fontsize=18,
            color=INK, zorder=4)


def fig_bsc_chain():
    """The running Markov chain X -> Y -> Z: a fair bit, flipped with
    probability 0.1, then flipped again with probability 0.2 (one flip of
    0.26 overall).  Beneath, the three numbers the section computes:
    I(X;Y) = 0.368, I(X;Z) = 0.120, I(X;Z|Y) = 0."""
    fig, ax = plt.subplots(figsize=(7.6, 2.5))
    xs = (0.0, 3.0, 6.0)
    for x0, name in zip(xs, ("$X$", "$Y$", "$Z$")):
        _box(ax, (x0, 0.0), name, BLUE)
    for a, b, lab in ((xs[0], xs[1], r"flip w.p. $0.1$"),
                      (xs[1], xs[2], r"flip w.p. $0.2$")):
        fl.arrow(ax, (a + 0.56, 0.0), (b - 0.56, 0.0), color=INK, lw=1.8,
                 mut=16)
        ax.text((a + b) / 2, 0.16, lab, ha="center", va="bottom",
                fontsize=14, color=INK)
    # The composite channel X -> Z, drawn as an arc above the boxes.
    ax.annotate("", xy=(xs[2], 0.40), xytext=(xs[0], 0.40),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.6,
                                mutation_scale=15,
                                connectionstyle="arc3,rad=-0.22"))
    ax.text(3.0, 1.12, r"one flip w.p. $0.1\cdot0.8+0.9\cdot0.2=0.26$",
            ha="center", va="bottom", fontsize=14, color=ORANGE)
    ax.text(3.0, -0.66,
            r"$I(X;Y)=0.368$     $I(X;Z)=0.120$     $I(X;Z\mid Y)=0$  (nats)",
            ha="center", va="top", fontsize=14, color=INK)
    ax.set_xlim(-0.7, 6.7)
    ax.set_ylim(-1.1, 1.55)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "mdl-it-bsc-chain")


FIGURES = [fig_mi_corr_vs_mi, fig_bsc_chain]


def main():
    fl.WRITTEN.clear()
    for fn in FIGURES:
        fn()
    for path in fl.WRITTEN:
        print(f"wrote {path} ({os.path.getsize(path):,d} bytes)")


if __name__ == "__main__":
    main()
