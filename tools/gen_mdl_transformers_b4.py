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
SPLIT_GAP = 0.16   # extra gap at the source/target block boundary


def _pos(i, split):
    """Offset of grid line i (row or column), widened past the split."""
    return i * CELL + (SPLIT_GAP if split is not None and i >= split else 0.0)


def _cell(ax, row, col, color, filled=True, split=None):
    """One grid cell in matrix convention: row 0 at the top."""
    x, y = _pos(col, split) + PAD, -(_pos(row, split) + CELL) + PAD
    w = CELL - 2 * PAD
    if filled:
        ax.add_patch(FancyBboxPatch((x, y), w, w,
                                    boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=color, ec="black", lw=0.7, zorder=3))
    else:
        ax.add_patch(FancyBboxPatch((x, y), w, w,
                                    boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc="none", ec=LIGHT, lw=0.7, zorder=2))


def _grid(ax, n, allowed, title, split=None):
    """Draw an n x n attention grid; allowed(row, col) -> color or None.
    `split` widens the gap before row/column `split` (block boundary)."""
    for r in range(n):
        for c in range(n):
            color = allowed(r, c)
            _cell(ax, r, c, color, filled=color is not None, split=split)
    span = _pos(n, split)
    # axis arrows: keys along the top (rightward), queries down the left side
    fl.arrow(ax, (0, 0.45), (span, 0.45), color="black", lw=1.2, mut=11)
    fl.arrow(ax, (-0.45, 0), (-0.45, -span), color="black", lw=1.2, mut=11)
    ax.text(span / 2, 0.75, "key position", ha="center", va="bottom",
            fontsize=13, color="black")
    ax.text(-0.78, -span / 2, "query position", ha="right", va="center",
            fontsize=13, color="black", rotation=90)
    x0, x1 = -2.4, n * CELL + 0.4 + SPLIT_GAP   # same window in all panels
    fl.clean_axes(ax, lim=((x0, x1), (-n * CELL - SPLIT_GAP - 1.7, 1.9)),
                  hide=True)
    # title centred on the grid (not on the axes, which include the
    # "query position" label on the left)
    ax.set_title(title, fontsize=14, color="black", pad=10,
                 x=(span / 2 - x0) / (x1 - x0))


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

    _grid(axes[2], n, encdec, "encoder–decoder", split=S)
    # split lines (centred in the widened block gap) + source/target group
    # labels on the third panel
    ax = axes[2]
    span, mid = _pos(n, S), S * CELL + SPLIT_GAP / 2
    ax.plot([mid, mid], [0, -span], color="black", lw=1.0,
            ls=(0, (4, 3)), zorder=4)
    ax.plot([0, span], [-mid, -mid], color="black", lw=1.0,
            ls=(0, (4, 3)), zorder=4)
    for c0, name in ((0, "source"), (S, "target")):
        xc = (_pos(c0, S) + _pos(c0 + S, S) - SPLIT_GAP * (c0 == 0)) / 2
        ax.text(xc, -span - 0.55, name, ha="center",
                va="top", fontsize=13, color="black", style="italic")
    for r0, name in ((0, "source"), (S, "target")):
        yc = (_pos(r0, S) + _pos(r0 + S, S) - SPLIT_GAP * (r0 == 0)) / 2
        ax.text(span + 0.35, -yc, name, ha="left",
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

    in_x = token_row(IN_Y, N, 0.0, fl.T.BLUE.tint)            # light blue inputs
    lat_x = token_row(LAT_Y, M, 4.75, ORANGE, pitch=1.15)
    out_x = token_row(OUT_Y, M, 4.75, fl.T.GREEN.tint, pitch=1.15)  # light green

    # cross-attention fan: every latent reads every input (thin); one latent's
    # fan highlighted.  Plain lines (flow is bottom -> top): fourteen
    # arrowheads converging on one box only make a blot.
    for j, lx in enumerate(lat_x):
        for i, ix in enumerate(in_x):
            hot = (j == 1)
            ax.plot([ix, lx], [IN_Y + SQ / 2 + 0.03, LAT_Y - SQ / 2 - 0.03],
                    color=ORANGE, lw=1.4 if hot else 0.8,
                    alpha=0.85 if hot else 0.16, zorder=1 if hot else 0,
                    solid_capstyle="butt")

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
            "cross-attention\n$O(MNd)$", ha="left", va="center", fontsize=14,
            color="black")
    ax.text(11.0, (LAT_Y + OUT_Y) / 2,
            "$L$ blocks of\nself-attention + FFN\n$O(LM^2d)$", ha="left",
            va="center",
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
