#!/usr/bin/env python3
"""Generate the illustrative figures for chapter_transformers/transformer-block.md
(B1) and chapter_transformers/gpt.md (B2) in the shared house style defined in
``gen_mdl_figures.py``.

Figures:
  mdl-transformers-block-anatomy.svg -- the transformer block as two sublayers
      reading from and writing to a residual stream, in the two normalization
      arrangements: post-LN (2017) and pre-LN (modern).

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_transformers_b1b2.py

All figures are written to ``img/mdl-transformers-<id>.svg``.  Byte-idempotent:
no timestamps, no unseeded randomness.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT

from matplotlib.patches import Circle, FancyBboxPatch


# --------------------------------------------------------------------------- #
# B1: block anatomy -- residual stream + two sublayers, post-LN vs pre-LN     #
# --------------------------------------------------------------------------- #

SX = 2.1          # x of the residual stream
BX0, BX1 = 4.6, 9.2   # sublayer box left/right edges
NW = 1.7          # norm box width


def _box(ax, x0, x1, y0, y1, text, fc, ec):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.10",
        fc=fc, ec=ec, lw=1.6, zorder=3))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center",
            fontsize=13, color="black", zorder=4)


def _add_circle(ax, y):
    ax.add_patch(Circle((SX, y), 0.30, fc="white", ec="black", lw=1.5,
                        zorder=5))
    ax.text(SX, y - 0.02, "+", ha="center", va="center", fontsize=15,
            color="black", zorder=6)


def _stream(ax, y0, y1):
    fl.arrow(ax, (SX, y0), (SX, y1), color=BLUE, lw=3.0, mut=18)


def _elbow(ax, pts, color, lw=1.8, arrow_end=True):
    """Polyline through pts; arrowhead on the final segment."""
    xs, ys = zip(*pts)
    ax.plot(xs[:-1] + (xs[-2],), ys[:-1] + (ys[-2],), color=color, lw=lw,
            solid_capstyle="round", zorder=2)
    if arrow_end:
        fl.arrow(ax, pts[-2], pts[-1], color=color, lw=lw, mut=13)
    else:
        ax.plot(xs[-2:], ys[-2:], color=color, lw=lw, zorder=2)


def _panel_postln(ax):
    # stream segments (bottom -> top), interrupted by norm boxes on the stream
    _stream(ax, 0.0, 3.55)
    _box(ax, SX - NW / 2, SX + NW / 2, 3.55, 4.45, "norm", "white", GRAY)
    _stream(ax, 4.45, 8.05)
    _box(ax, SX - NW / 2, SX + NW / 2, 8.05, 8.95, "norm", "white", GRAY)
    _stream(ax, 8.95, 10.6)

    # attention sublayer: read at y=0.9, write into + at y=2.9
    _box(ax, BX0, BX1, 1.3, 2.5, "multi-head\nattention", "#ffe8d0", ORANGE)
    _elbow(ax, [(SX, 0.9), (6.9, 0.9), (6.9, 1.3)], ORANGE)
    _elbow(ax, [(6.9, 2.5), (6.9, 2.9), (SX + 0.30, 2.9)], ORANGE)
    _add_circle(ax, 2.9)

    # FFN sublayer: read at y=5.4, write into + at y=7.4
    _box(ax, BX0, BX1, 5.8, 7.0, "feed-forward\nnetwork", "#ddf0dd", GREEN)
    _elbow(ax, [(SX, 5.4), (6.9, 5.4), (6.9, 5.8)], GREEN)
    _elbow(ax, [(6.9, 7.0), (6.9, 7.4), (SX + 0.30, 7.4)], GREEN)
    _add_circle(ax, 7.4)

    ax.set_title("post-LN (2017)", fontsize=14, color="black")


def _panel_preln(ax):
    # one uninterrupted stream: sublayers only add to it
    _stream(ax, 0.0, 10.6)

    # attention sublayer with norm on the branch
    _box(ax, BX0, BX1 - 2.7, 0.5, 1.3, "norm", "white", GRAY)
    _box(ax, BX0, BX1, 1.9, 3.1, "multi-head\nattention", "#ffe8d0", ORANGE)
    _elbow(ax, [(SX, 0.9), (BX0, 0.9)], ORANGE)
    _elbow(ax, [(BX1 - 2.7, 0.9), (6.9, 0.9), (6.9, 1.9)], ORANGE)
    _elbow(ax, [(6.9, 3.1), (6.9, 3.5), (SX + 0.30, 3.5)], ORANGE)
    _add_circle(ax, 3.5)

    # FFN sublayer with norm on the branch
    _box(ax, BX0, BX1 - 2.7, 5.0, 5.8, "norm", "white", GRAY)
    _box(ax, BX0, BX1, 6.4, 7.6, "feed-forward\nnetwork", "#ddf0dd", GREEN)
    _elbow(ax, [(SX, 5.4), (BX0, 5.4)], GREEN)
    _elbow(ax, [(BX1 - 2.7, 5.4), (6.9, 5.4), (6.9, 6.4)], GREEN)
    _elbow(ax, [(6.9, 7.6), (6.9, 8.0), (SX + 0.30, 8.0)], GREEN)
    _add_circle(ax, 8.0)

    # label the untouched stream
    ax.text(SX - 0.45, 9.5, "residual\nstream", ha="right", va="center",
            fontsize=13, color=BLUE)

    ax.set_title("pre-LN (modern)", fontsize=14, color="black")


def fig_block_anatomy() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.6))
    for ax, panel in zip(axes, (_panel_postln, _panel_preln)):
        panel(ax)
        ax.text(SX + 0.42, 0.12, r"$\mathbf{x}$", ha="left", va="bottom",
                fontsize=14, color="black")
        ax.set_xlim(-0.6, 9.7)
        ax.set_ylim(-0.2, 11.0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.subplots_adjust(wspace=0.06)
    fl.save(fig, "mdl-transformers-block-anatomy")


if __name__ == "__main__":
    fig_block_anatomy()
    for path in fl.WRITTEN:
        print(f"wrote {os.path.relpath(path)}")
