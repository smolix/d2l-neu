#!/usr/bin/env python3
"""Generate the illustrative figures for chapter_transformers/encoders-decoders.md
(B4) in the shared house style defined in ``gen_mdl_figures.py``.

Figures:
  mdl-transformers-three-wirings.svg -- the three attention patterns one block
      supports: encoder-only (full bidirectional square), decoder-only (causal
      triangle), encoder-decoder (bidirectional source square + cross-attention
      rectangle + causal target triangle), drawn in matrix convention (query
      position increases downward) to match the notebook's heatmaps.
  mdl-transformers-latent-bottleneck.svg -- the Perceiver idea: M learned
      latents cross-attend into an input of length N (cost O(MN)), then
      process among themselves (cost O(M^2)).

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_transformers_b4.py

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

from matplotlib.patches import FancyBboxPatch, Patch, Rectangle


# --------------------------------------------------------------------------- #
# three wirings -- attention-pattern grids                                    #
# --------------------------------------------------------------------------- #

CELL = 1.0     # grid pitch
PAD = 0.10     # gap around each cell


def _cell(ax, row, col, color, filled=True):
    """One grid cell in matrix convention: row 0 at the top."""
    x, y = col * CELL + PAD, -(row + 1) * CELL + PAD
    w = CELL - 2 * PAD
    if filled:
        ax.add_patch(FancyBboxPatch((x, y), w, w,
                                    boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=color, ec="black", lw=0.7, zorder=3))
    else:
        ax.add_patch(FancyBboxPatch((x, y), w, w,
                                    boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc="none", ec=LIGHT, lw=0.7, zorder=2))


def _grid(ax, n, allowed, title):
    """Draw an n x n attention grid; allowed(row, col) -> color or None."""
    for r in range(n):
        for c in range(n):
            color = allowed(r, c)
            _cell(ax, r, c, color, filled=color is not None)
    ax.set_title(title, fontsize=14, color="black", pad=10)
    # axis arrows: keys along the top (rightward), queries down the left side
    fl.arrow(ax, (0, 0.45), (n * CELL, 0.45), color="black", lw=1.2, mut=11)
    fl.arrow(ax, (-0.45, 0), (-0.45, -n * CELL), color="black", lw=1.2, mut=11)
    ax.text(n * CELL / 2, 0.75, "key position", ha="center", va="bottom",
            fontsize=13, color="black")
    ax.text(-0.78, -n * CELL / 2, "query position", ha="right", va="center",
            fontsize=13, color="black", rotation=90)
    fl.clean_axes(ax, lim=((-2.4, n * CELL + 0.4), (-n * CELL - 1.7, 1.9)),
                  hide=True)


def fig_three_wirings():
    n, S = 8, 4          # 8 x 8 grids; encoder-decoder splits 4 source + 4 target
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.6))

    _grid(axes[0], n, lambda r, c: BLUE, "encoder-only")
    _grid(axes[1], n, lambda r, c: GREEN if c <= r else None, "decoder-only")

    def encdec(r, c):
        if r < S:                      # source queries: encoder self-attention
            return BLUE if c < S else None
        if c < S:                      # target queries reading the source
            return ORANGE
        return GREEN if (c - S) <= (r - S) else None  # causal among targets

    _grid(axes[2], n, encdec, "encoder–decoder")
    # split lines + source/target group labels on the third panel
    ax = axes[2]
    ax.plot([S * CELL, S * CELL], [0, -n * CELL], color="black", lw=1.0,
            ls=(0, (4, 3)), zorder=4)
    ax.plot([0, n * CELL], [-S * CELL, -S * CELL], color="black", lw=1.0,
            ls=(0, (4, 3)), zorder=4)
    for x0, name in ((0, "source"), (S, "target")):
        ax.text((x0 + S / 2) * CELL, -n * CELL - 0.55, name, ha="center",
                va="top", fontsize=13, color="black", style="italic")
    for y0, name in ((0, "source"), (S, "target")):
        ax.text(n * CELL + 0.35, -(y0 + S / 2) * CELL, name, ha="left",
                va="center", fontsize=13, color="black", style="italic",
                rotation=270)

    fig.legend(handles=[
        Patch(fc=BLUE, ec="black", lw=0.7, label="bidirectional self-attention"),
        Patch(fc=GREEN, ec="black", lw=0.7, label="causal self-attention"),
        Patch(fc=ORANGE, ec="black", lw=0.7, label="cross-attention"),
    ], loc="lower center", ncol=3, fontsize=13, frameon=False,
        bbox_to_anchor=(0.5, -0.02), handlelength=1.2, handleheight=1.2)
    fig.subplots_adjust(wspace=0.16, bottom=0.02, top=0.98)
    fl.save(fig, "mdl-transformers-three-wirings")


# --------------------------------------------------------------------------- #
# latent bottleneck -- the Perceiver idea                                     #
# --------------------------------------------------------------------------- #

def fig_latent_bottleneck():
    N, M = 14, 4
    IN_Y, LAT_Y, OUT_Y = 0.0, 3.0, 6.0       # rows: input, latents, outputs
    SQ = 0.72                                 # token square size

    fig, ax = plt.subplots(figsize=(9.2, 4.9))

    def token_row(y, count, x0, color, pitch=1.0):
        xs = [x0 + i * pitch for i in range(count)]
        for x in xs:
            ax.add_patch(FancyBboxPatch(
                (x - SQ / 2, y - SQ / 2), SQ, SQ,
                boxstyle="round,pad=0.02,rounding_size=0.10",
                fc=color, ec="black", lw=0.9, zorder=3))
        return xs

    in_x = token_row(IN_Y, N, 0.0, "#c6dbef")            # light blue inputs
    lat_x = token_row(LAT_Y, M, 4.75, ORANGE, pitch=1.15)
    out_x = token_row(OUT_Y, M, 4.75, "#a1d99b", pitch=1.15)  # light green

    # cross-attention fan: every latent reads every input (thin); one latent's
    # fan highlighted; arrows follow the information flow, input -> latent
    for j, lx in enumerate(lat_x):
        for i, ix in enumerate(in_x):
            hot = (j == 1)
            ax.annotate("", xy=(lx, LAT_Y - SQ / 2 - 0.03),
                        xytext=(ix, IN_Y + SQ / 2 + 0.03),
                        arrowprops=dict(arrowstyle="->", color=ORANGE,
                                        lw=1.3 if hot else 0.8,
                                        alpha=0.85 if hot else 0.16,
                                        shrinkA=0, shrinkB=0,
                                        mutation_scale=9))

    # latent self-attention: all-pairs arcs between output latents' inputs
    for j, lx in enumerate(lat_x):
        for k, ox in enumerate(out_x):
            ax.annotate("", xy=(ox, OUT_Y - SQ / 2 - 0.03),
                        xytext=(lx, LAT_Y + SQ / 2 + 0.03),
                        arrowprops=dict(arrowstyle="->", color=GREEN,
                                        lw=0.9, alpha=0.55,
                                        shrinkA=0, shrinkB=0,
                                        mutation_scale=9))

    # stage labels on the right, clear of the arrows
    ax.text(11.0, (IN_Y + LAT_Y) / 2 + 0.15,
            "cross-attention\n$O(MN)$", ha="left", va="center", fontsize=14,
            color="black")
    ax.text(11.0, (LAT_Y + OUT_Y) / 2,
            "self-attention + FFN\n$O(M^2)$", ha="left", va="center",
            fontsize=14, color="black")

    # row labels on the left
    ax.text(-1.0, IN_Y, "input\n(length $N$)", ha="right", va="center",
            fontsize=14, color="black")
    ax.text(3.5, LAT_Y, "$M$ learned latents\n($M \\ll N$)", ha="right",
            va="center", fontsize=14, color="black")
    ax.text(3.5, OUT_Y, "latent summary", ha="right", va="center",
            fontsize=14, color="black")

    fl.clean_axes(ax, lim=((-4.1, 14.6), (-0.9, 6.9)), hide=True,
                  equal=False)
    fl.save(fig, "mdl-transformers-latent-bottleneck")


if __name__ == "__main__":
    fl.WRITTEN.clear()
    fig_three_wirings()
    fig_latent_bottleneck()
    for path in fl.WRITTEN:
        print("wrote", os.path.relpath(path, os.path.dirname(fl.IMG_DIR)))
