#!/usr/bin/env python3
"""Generate the head-split shape figure for
chapter_attention/multihead-attention.md (A8) in the shared house style
defined in ``gen_mdl_figures.py``.

Figures:
  mdl-attention-head-split.svg -- heads as column slices of one projection:
      the (n, d) query projection is cut into H slices of width d/H, the
      slices become an explicit head axis, each head attends on its own, and
      the outputs are put side by side again for the output projection.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_attention_a8.py

All figures are written to ``img/mdl-attention-<id>.svg``.  Byte-idempotent:
no timestamps, no random draws.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

from matplotlib.patches import Rectangle

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY
HEADS = [BLUE, ORANGE, GREEN]       # H = 3 heads, one colour each
CELL = 0.3                          # side of one matrix cell
N, DH = 4, 2                        # rows (tokens), head width
D = DH * len(HEADS)                 # model width


def _grid(ax, x, y_top, rows, cols, color, alpha=0.22, lw=1.8, z=2):
    """A rows x cols matrix with its top-left corner at (x, y_top)."""
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(Rectangle(
                (x + c * CELL, y_top - (r + 1) * CELL), CELL, CELL,
                facecolor=color, alpha=alpha, edgecolor="white", lw=0.8,
                zorder=z))
    ax.add_patch(Rectangle((x, y_top - rows * CELL), cols * CELL,
                           rows * CELL, facecolor="none", edgecolor=color,
                           lw=lw, zorder=z + 1))


def _striped(ax, x, y_top):
    """An n x d matrix whose column slices are coloured by head."""
    for h, color in enumerate(HEADS):
        _grid(ax, x + h * DH * CELL, y_top, N, DH, color, lw=1.4)
    ax.add_patch(Rectangle((x, y_top - N * CELL), D * CELL, N * CELL,
                           facecolor="none", edgecolor="black", lw=1.8,
                           zorder=5))


def _stack(ax, x, y_top, labels):
    """H separate n x d_h matrices, one per head, fanned diagonally."""
    step_x, step_y = 0.36, 0.30
    for h in reversed(range(len(HEADS))):
        xh, yh = x + h * step_x, y_top - h * step_y
        ax.add_patch(Rectangle((xh, yh - N * CELL), DH * CELL, N * CELL,
                               facecolor="white", edgecolor="none",
                               zorder=2 + 3 * (len(HEADS) - h)))
        _grid(ax, xh, yh, N, DH, HEADS[h], alpha=0.3,
              z=2 + 3 * (len(HEADS) - h))
    ax.text(x + DH * CELL / 2, y_top + 0.12, labels[0], ha="center",
            va="bottom", fontsize=14, color=fl.T.BLUE.dark)
    last = len(HEADS) - 1
    ax.text(x + last * step_x + DH * CELL + 0.1,
            y_top - last * step_y - N * CELL / 2, labels[1], ha="left",
            va="center", fontsize=14, color=fl.T.GREEN.dark)


def _step(ax, x0, x1, y, text):
    fl.arrow(ax, (x0, y), (x1, y), color=GRAY, lw=1.8, mut=13)
    ax.text((x0 + x1) / 2, y + 0.12, text, ha="center", va="bottom",
            fontsize=13, color="black")


def fig_head_split() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 2.9))
    ax.set_aspect("equal")
    ax.axis("off")

    top = 1.6
    mid = top - N * CELL / 2
    below = top - N * CELL - 2 * 0.30 - 0.4   # row of shape captions

    # 1. the query projection, one matrix of width d
    x1 = 0.0
    _striped(ax, x1, top)
    ax.text(x1 + D * CELL / 2, top + 0.12, "projected queries",
            ha="center", va="bottom", fontsize=14, color="black")
    ax.text(x1 + D * CELL / 2, below, "(batch, $n$, $d$)", ha="center",
            va="center", fontsize=13, color="black")

    # 2. split_heads: H matrices of width d_h
    x2 = x1 + D * CELL + 1.55
    _step(ax, x1 + D * CELL + 0.15, x2 - 0.15, mid, "split")
    _stack(ax, x2, top, (r"head 1", r"head $H$"))
    ax.text(x2 + 0.36 + DH * CELL / 2, below, "(batch, $H$, $n$, $d_h$)",
            ha="center", va="center", fontsize=13, color="black")

    # 3. attention in every head at once: outputs Z_1 ... Z_H
    x3 = x2 + 2 * 0.36 + DH * CELL + 2.3
    _step(ax, x2 + 2 * 0.36 + DH * CELL + 0.75, x3 - 0.15, mid,
          "attend")
    _stack(ax, x3, top, (r"$\mathbf{Z}_1$", r"$\mathbf{Z}_H$"))
    ax.text(x3 + 0.36 + DH * CELL / 2, below, "(batch, $H$, $n$, $d_h$)",
            ha="center", va="center", fontsize=13, color="black")

    # 4. merge_heads: side by side again, then W_o
    x4 = x3 + 2 * 0.36 + DH * CELL + 2.0
    _step(ax, x3 + 2 * 0.36 + DH * CELL + 0.75, x4 - 0.15, mid, "merge")
    _striped(ax, x4, top)
    ax.text(x4 + D * CELL / 2, top + 0.12,
            r"$[\mathbf{Z}_1, \ldots, \mathbf{Z}_H]$", ha="center",
            va="bottom", fontsize=15, color="black")
    ax.text(x4 + D * CELL / 2, below, "(batch, $n$, $d$)", ha="center",
            va="center", fontsize=13, color="black")
    x5 = x4 + D * CELL
    _step(ax, x5 + 0.15, x5 + 1.25, mid, r"$\mathbf{W}_o$")

    ax.set_xlim(-0.1, x5 + 1.3)
    ax.set_ylim(below - 0.3, top + 0.6)
    fl.save(fig, "mdl-attention-head-split")


ALL_FIGURES = [fig_head_split]


def main():
    start = len(fl.WRITTEN)
    for f in ALL_FIGURES:
        f()
    for path in fl.WRITTEN[start:]:
        print(f"wrote {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
