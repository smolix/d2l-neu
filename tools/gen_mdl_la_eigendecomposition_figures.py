#!/usr/bin/env python3
"""Generate the new illustrative figures of the Eigendecompositions section
(``chapter_mdl-linear-algebra/mdl-eigendecomposition.md``) in the shared
house style of ``gen_mdl_figures.py``.

Importing ``gen_mdl_figures`` applies the shared ``rcParams`` (fixed
``svg.hashsalt``; ``save()`` drops the SVG date), so re-runs are byte-for-byte
identical.  The other figures of the section live in ``gen_mdl_figures.py``.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_la_eigendecomposition_figures.py

Writes ``img/mdl-la-markov-iterates.svg`` and ``img/mdl-la-transient-cancel.svg``
(the latter also to ``img/auto/`` for the slides).
"""

from __future__ import annotations

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
T = fl.T
BLUE, ORANGE, GREEN, GRAY, INK = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.INK
arrow, clean_axes, axis_cross = fl.arrow, fl.clean_axes, fl.axis_cross
LBL = 15


def fig_markov_iterates():
    """The two-page surfer: p_k = P^k (1, 0) slides along x + y = 1 toward the
    stationary pi = (1/3, 2/3), the point where the eigenvector (1, 2) of
    lambda = 1 meets the line; the error lies along (1, -1), lambda = 1/4."""
    P = np.array([[0.5, 0.25], [0.5, 0.75]])
    ps = [np.array([1.0, 0.0])]
    for _ in range(4):
        ps.append(P @ ps[-1])
    pi = np.array([1 / 3, 2 / 3])

    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    lim = ((-0.12, 1.15), (-0.12, 1.15))
    axis_cross(ax, lim[0], lim[1], color="black")
    ax.plot([-0.05, 1.1], [1.05, -0.1], color=GRAY, lw=1.3, zorder=1)
    ax.text(0.79, 0.30, r"$x+y=1$", color="black", fontsize=14,
            ha="left", va="bottom")
    # eigenvector line of lambda = 1 through the origin, ending at pi
    ax.plot([0, pi[0]], [0, pi[1]], "--", color=GREEN, lw=1.8, zorder=2)
    ax.text(0.10, 0.36, r"$(1,2)$, $\lambda=1$", color=T.GREEN.dark,
            fontsize=14, ha="center", va="center",
            rotation=np.degrees(np.arctan(2.0)))
    # the error direction (1, -1), lambda = 1/4: drawn just below the line
    d = np.array([1.0, -1.0]) / np.sqrt(2)
    tail = pi - 0.05
    arrow(ax, tail, tail + 0.30 * d, color=GREEN, lw=2.2)
    ax.text(0.44, 0.20, "$(1,{-1})$, $\\lambda=1/4$\n" r"$\times\frac{1}{4}$ per step",
            color=T.GREEN.dark, fontsize=14, ha="center", va="center")
    cols = fl.BLUE_CMAP(np.linspace(0.35, 1.0, len(ps)))
    for k, (p, c) in enumerate(zip(ps, cols)):
        ax.plot(p[0], p[1], "o", color=c, ms=8, zorder=5)
    for k, off, ha, va in [(0, (-0.06, -0.04), "center", "top"),
                           (1, (0.04, 0.03), "left", "bottom"),
                           (2, (0.04, 0.03), "left", "bottom")]:
        ax.text(ps[k][0] + off[0], ps[k][1] + off[1], rf"$\mathbf{{p}}_{k}$",
                color=T.BLUE.dark, fontsize=LBL, ha=ha, va=va)
    ax.plot(pi[0], pi[1], "o", color=ORANGE, ms=10, zorder=6)
    ax.text(0.36, 0.75, r"$\boldsymbol{\pi}=(1/3,\,2/3)$",
            color=T.ORANGE.dark, fontsize=LBL, ha="left", va="bottom")
    clean_axes(ax, lim=lim, hide=True)
    fl.save(fig, "mdl-la-markov-iterates")


def fig_transient_cancel():
    """Why M = [[0.9, 4], [0, 0.8]] grows before it decays: e2 = 40 v1 + v2 is
    the difference of two nearly cancelling eigen-components (k = 0); after six
    steps they have decayed by 0.9^6 and 0.8^6 and no longer cancel (k = 6)."""
    v1 = np.array([1.0, 0.0])
    v2 = np.array([-40.0, 1.0])
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.0), sharey=True)
    for ax, k in zip(axes, (0, 6)):
        a = 40 * 0.9 ** k * v1
        b = 0.8 ** k * v2
        s = a + b
        ax.axhline(0, color="black", lw=0.8, zorder=1)
        ax.axvline(0, color="black", lw=0.8, zorder=1)
        arrow(ax, (0, 0), a, color=BLUE, lw=2.4)
        arrow(ax, (0, 0), b, color=ORANGE, lw=2.4)
        arrow(ax, (0, 0), s, color="black", lw=3.0, mut=16)
        if k == 0:
            ax.text(a[0], 0.13, r"$40\,\mathbf{v}_1=(40,0)$", color=T.BLUE.dark,
                    fontsize=14, ha="right", va="bottom")
            ax.text(b[0], b[1] + 0.12, r"$\mathbf{v}_2=(-40,1)$",
                    color=T.ORANGE.dark, fontsize=14, ha="left", va="bottom")
            ax.text(1.5, 1.08, r"$\mathbf{e}_2=40\,\mathbf{v}_1+\mathbf{v}_2$",
                    color="black", fontsize=14, ha="left", va="center")
            ax.set_title(r"$k=0$: length 1", fontsize=15, color="black")
        else:
            ax.text(a[0] + 1.0, -0.10, r"$40\cdot0.9^6\,\mathbf{v}_1$",
                    color=T.BLUE.dark, fontsize=14, ha="left", va="top")
            ax.text(b[0], b[1] + 0.12, r"$0.8^6\,\mathbf{v}_2$",
                    color=T.ORANGE.dark, fontsize=14, ha="left", va="bottom")
            ax.text(s[0] + 1.0, s[1] + 0.22,
                    r"$\mathbf{M}^6\mathbf{e}_2=(10.8,\,0.26)$",
                    color="black", fontsize=14, ha="left", va="bottom")
            ax.set_title(r"$k=6$: length 10.8", fontsize=15, color="black")
        ax.set_xlim(-46, 46)
        ax.set_ylim(-0.45, 1.45)
        ax.set_aspect("auto")
        ax.axis("off")
    fl.save(fig, "mdl-la-transient-cancel")
    # the slides read @fig figures from img/auto
    shutil.copyfile(os.path.join(fl.IMG_DIR, "mdl-la-transient-cancel.svg"),
                    os.path.join(fl.IMG_DIR, "auto", "mdl-la-transient-cancel.svg"))


FIGURES = [fig_markov_iterates, fig_transient_cancel]


def main():
    for fn in FIGURES:
        fn()
    for p in fl.WRITTEN:
        print(f"  {os.path.basename(p):32s} {os.path.getsize(p):>8,d} bytes")


if __name__ == "__main__":
    main()
