#!/usr/bin/env python3
"""Generate the illustrative figure for chapter_attention/attention-at-scale.md
(A5) in the shared house style defined in ``gen_mdl_figures.py``.

Figure:
  mdl-attention-online-softmax.svg -- the tiling picture behind online softmax
      and FlashAttention: the n x n score matrix is processed one key block at
      a time; only the current n x c stripe plus a small running state is ever
      in memory.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_attention_a5.py

Output goes to ``img/mdl-attention-<id>.svg``.  Byte-idempotent: no
timestamps, no unseeded randomness.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT
Rectangle = fl.Rectangle

WHITE_BOX = dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2.5)


def fig_online_softmax() -> None:
    n_blocks, side = 6, 6.0            # score matrix drawn as a 6x6-block square
    bw = side / n_blocks               # block (stripe) width
    active = 3                         # index of the stripe being processed

    fig, ax = plt.subplots(figsize=(9.2, 4.6))

    # --- the n x n score matrix, one vertical stripe per key block ---------
    for b in range(n_blocks):
        x0 = b * bw
        if b < active:                 # already folded into the running state
            face, edge, ls, alpha = BLUE, BLUE, "-", 0.22
        elif b == active:              # the one stripe in memory right now
            face, edge, ls, alpha = ORANGE, ORANGE, "-", 0.55
        else:                          # not yet computed
            face, edge, ls, alpha = "none", GRAY, "--", 1.0
        ax.add_patch(Rectangle((x0, 0), bw, side, facecolor=face,
                               edgecolor=edge, linestyle=ls, alpha=alpha,
                               linewidth=1.4))
    ax.add_patch(Rectangle((0, 0), side, side, facecolor="none",
                           edgecolor="black", linewidth=1.6))

    # direction labels
    ax.annotate("", xy=(2.1, 6.4), xytext=(0.0, 6.4),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.4))
    ax.text(1.05, 6.58, "keys ($n$ columns)", ha="center", va="bottom",
            fontsize=13, color="black")
    ax.annotate("", xy=(-0.5, 3.6), xytext=(-0.5, 6.0),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.4))
    ax.text(-0.85, 4.8, "queries ($n$ rows)", ha="center", va="center",
            fontsize=13, color="black", rotation=90)

    ax.text(3.0, -0.75,
            r"scores $\mathbf{Q}\mathbf{K}^\top\!/\sqrt{d}\,$:"
            " one block at a time",
            ha="center", va="center", fontsize=13.5, color="black")

    # per-region annotations
    ax.text(1.5, 3.0, "folded into\nthe running\nstate", ha="center",
            va="center", fontsize=13, color="black", bbox=WHITE_BOX)
    ax.text(3.5, 3.0, "current $n \\times c$ block", ha="center",
            va="center", fontsize=13, color="black", rotation=90)
    ax.text(5.15, 3.0, "not yet\ncomputed", ha="center", va="center",
            fontsize=13, color=GRAY, bbox=WHITE_BOX)

    # --- the running state -------------------------------------------------
    sx = 7.7                            # left edge of the state group
    ax.add_patch(Rectangle((sx, 0), 0.28, side, facecolor=GREEN,
                           edgecolor=GREEN, alpha=0.45))
    ax.add_patch(Rectangle((sx + 0.62, 0), 0.28, side, facecolor=GREEN,
                           edgecolor=GREEN, alpha=0.45))
    ax.add_patch(Rectangle((sx + 1.24, 0), 0.95, side, facecolor=BLUE,
                           edgecolor=BLUE, alpha=0.30))
    ax.text(sx + 0.14, -0.32, r"$\mathbf{m}$", ha="center", va="top",
            fontsize=14, color="black")
    ax.text(sx + 0.76, -0.32, r"$\mathbf{s}$", ha="center", va="top",
            fontsize=14, color="black")
    ax.text(sx + 1.72, -0.32, r"$\mathbf{O}$", ha="center", va="top",
            fontsize=14, color="black")
    ax.text(sx + 1.1, -0.95, "running state: $O(nd)$", ha="center",
            va="top", fontsize=13, color="black")

    # arrow: top of the current stripe -> state group, arcing over the matrix
    ax.annotate("", xy=(sx - 0.2, 5.75), xytext=(3.55, 6.12),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.6,
                                connectionstyle="arc3,rad=-0.22"))
    ax.text(4.35, 7.05, "rescale by $e^{\\,m - m'}$, accumulate",
            ha="center", va="bottom", fontsize=13, color="black")

    ax.set_xlim(-1.1, 10.0)
    ax.set_ylim(-1.55, 7.55)
    ax.set_aspect("equal")
    ax.axis("off")
    fl.save(fig, "mdl-attention-online-softmax")


if __name__ == "__main__":
    fig_online_softmax()
    print("\n".join(fl.WRITTEN))
