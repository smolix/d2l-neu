#!/usr/bin/env python3
"""Generate the illustrative figures for chapter_transformers/encoders-decoders.md
(B4) in the shared house style defined in ``gen_mdl_figures.py``.

Figures:
  mdl-transformers-three-wirings.svg -- the three attention patterns one block
      supports: encoder-only (full bidirectional square), decoder-only (causal
      triangle), encoder-decoder (bidirectional source square + cross-attention
      rectangle + causal target triangle), drawn in matrix convention (query
      position increases downward) to match the notebook's heatmaps.  All
      cells are attention (orange role); cross-attention cells are lighter.
  mdl-transformers-latent-bottleneck.svg -- the Perceiver idea: M learned
      latents cross-attend into an input of length N (cost O(MN)), then
      process among themselves (cost O(M^2)).  Stream tokens blue, attention
      edges orange; one latent's read drawn in full, the rest as a band.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_transformers_b4.py

All figures are written to ``img/mdl-transformers-<id>.svg`` with unitless
width/height (CSS px), as the figure style guide asks.  Byte-idempotent: no
timestamps, no unseeded randomness.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT

import re

from matplotlib.patches import FancyBboxPatch, Polygon

ATTN = fl.T.ORANGE      # attention role
STREAM = fl.T.BLUE      # stream role: tokens and latents


# --------------------------------------------------------------------------- #
# three wirings -- attention-pattern grids                                    #
# --------------------------------------------------------------------------- #

CELL = 1.0         # grid pitch
PAD = 0.09         # gap around each cell
SPLIT_GAP = 0.22   # extra gap at the source/target block boundary
CROSS_ALPHA = 0.38  # cross-attention cells: the attention colour, lighter


def _pos(i, split):
    """Offset of grid line i (row or column), widened past the split."""
    return i * CELL + (SPLIT_GAP if split is not None and i >= split else 0.0)


def _cell(ax, row, col, kind, split=None):
    """One grid cell in matrix convention (row 0 at the top).  kind is
    'self', 'cross', or None (masked)."""
    x, y = _pos(col, split) + PAD, -(_pos(row, split) + CELL) + PAD
    w = CELL - 2 * PAD
    style = "round,pad=0.02,rounding_size=0.12"
    if kind is None:
        ax.add_patch(FancyBboxPatch((x, y), w, w, boxstyle=style, fc="none",
                                    ec=LIGHT, lw=0.8, zorder=2))
    else:
        alpha = 1.0 if kind == "self" else CROSS_ALPHA
        ax.add_patch(FancyBboxPatch((x, y), w, w, boxstyle=style,
                                    fc=ATTN.base, alpha=alpha, ec="none",
                                    zorder=3))
        ax.add_patch(FancyBboxPatch((x, y), w, w, boxstyle=style, fc="none",
                                    ec="black", lw=0.8, zorder=4))


WIN = None   # shared window, set in fig_three_wirings


def _grid(ax, n, kind, title, split=None):
    """Draw an n x n attention grid; kind(row, col) -> 'self'/'cross'/None.
    `split` widens the gap before row/column `split` (block boundary)."""
    for r in range(n):
        for c in range(n):
            _cell(ax, r, c, kind(r, c), split=split)
    span = _pos(n, split)
    # axis arrows: keys along the top (rightward), queries down the left
    fl.arrow(ax, (0, 0.42), (span, 0.42), color="black", lw=1.3, mut=12)
    fl.arrow(ax, (-0.42, 0), (-0.42, -span), color="black", lw=1.3, mut=12)
    ax.text(span / 2, 0.68, "key", ha="center", va="bottom", fontsize=14,
            color="black")
    ax.text(-0.68, -span / 2, "query", ha="right", va="center", fontsize=14,
            color="black", rotation=90)
    (x0, x1), _ = WIN
    fl.clean_axes(ax, lim=WIN, hide=True)
    ax.set_title(title, fontsize=15, color="black", pad=6,
                 x=(span / 2 - x0) / (x1 - x0))


def fig_three_wirings():
    global WIN
    n, S = 6, 3          # 6 x 6 grids; encoder-decoder: 3 source + 3 target
    span = _pos(n, S)
    WIN = ((-1.55, span + 1.15), (-span - 1.25, 1.55))  # same in all panels
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.7))

    _grid(axes[0], n, lambda r, c: "self", "encoder-only")
    _grid(axes[1], n, lambda r, c: "self" if c <= r else None,
          "decoder-only")

    def encdec(r, c):
        if r < S:                      # source queries: encoder self-attention
            return "self" if c < S else None
        if c < S:                      # target queries read the source
            return "cross"
        return "self" if c <= r else None  # causal among targets

    _grid(axes[2], n, encdec, "encoder–decoder", split=S)
    # source/target group labels on the third panel: columns below the grid,
    # rows to its right (centred on each 3-cell block)
    ax = axes[2]
    for i0, name in ((0, "source"), (S, "target")):
        lo, hi = _pos(i0, S), _pos(i0 + S, S) - (SPLIT_GAP if i0 == 0 else 0)
        mid = (lo + hi) / 2
        ax.text(mid, -span - 0.25, name, ha="center", va="top", fontsize=13,
                color="black", style="italic")
        ax.text(span + 0.25, -mid, name, ha="left", va="center", fontsize=13,
                color="black", style="italic", rotation=270)
    fig.subplots_adjust(wspace=0.04, left=0.0, right=1.0, bottom=0.0,
                        top=0.9)
    fl.save(fig, "mdl-transformers-three-wirings")


# --------------------------------------------------------------------------- #
# latent bottleneck -- the Perceiver idea                                     #
# --------------------------------------------------------------------------- #

def fig_latent_bottleneck():
    N, M = 12, 4
    IN_Y, LAT_Y, OUT_Y = 0.0, 2.6, 5.0       # rows: input, latents, summary
    SQ = 0.62                                 # token square size
    PITCH_IN, PITCH_LAT = 0.82, 0.95
    fig, ax = plt.subplots(figsize=(7.2, 3.5))

    def token_row(y, count, x0, fc, pitch):
        xs = [x0 + i * pitch for i in range(count)]
        for x in xs:
            ax.add_patch(FancyBboxPatch(
                (x - SQ / 2, y - SQ / 2), SQ, SQ,
                boxstyle="round,pad=0.02,rounding_size=0.10",
                fc=fc, ec="black", lw=0.9, zorder=3))
        return xs

    in_x = token_row(IN_Y, N, 0.0, STREAM.tint, PITCH_IN)
    mid = in_x[-1] / 2
    lat_x0 = mid - (M - 1) * PITCH_LAT / 2
    lat_x = token_row(LAT_Y, M, lat_x0, STREAM.base, PITCH_LAT)
    out_x = token_row(OUT_Y, M, lat_x0, STREAM.base, PITCH_LAT)
    top_in, bot_lat = IN_Y + SQ / 2 + 0.03, LAT_Y - SQ / 2 - 0.03

    # every latent reads every input: a light band behind the fan ...
    ax.add_patch(Polygon([(in_x[0], top_in), (in_x[-1], top_in),
                          (lat_x[-1], bot_lat), (lat_x[0], bot_lat)],
                         closed=True, fc=ATTN.tint, ec="none", zorder=0))
    # ... and one latent's read drawn in full
    hot = lat_x[1]
    for ix in in_x:
        ax.plot([ix, hot], [top_in, bot_lat], color=ATTN.base, lw=1.5,
                zorder=1, solid_capstyle="butt")

    # latent self-attention: all M^2 pairs, one block shown
    top_lat, bot_out = LAT_Y + SQ / 2 + 0.03, OUT_Y - SQ / 2 - 0.03
    for lx in lat_x:
        for ox in out_x:
            ax.plot([lx, ox], [top_lat, bot_out], color=ATTN.base, lw=1.1,
                    zorder=1, solid_capstyle="butt")

    # cost brackets on the right, each next to the stage it names
    def bracket(x, y0, y1, text):
        ax.plot([x - 0.12, x, x, x - 0.12], [y0, y0, y1, y1], color="black",
                lw=1.2)
        ax.text(x + 0.2, (y0 + y1) / 2, text, ha="left", va="center",
                fontsize=14, color="black")

    bracket(in_x[-1] + 0.65, IN_Y, LAT_Y,
            "cross-attention\n$O(MNd)$")
    bracket(lat_x[-1] + 0.75, LAT_Y, OUT_Y,
            "$L$ blocks of\nself-attention + FFN\n$O(LM^2d)$")

    # row labels on the left
    ax.text(-0.6, IN_Y, "input\n(length $N$)", ha="right", va="center",
            fontsize=14, color="black")
    ax.text(lat_x[0] - 0.65, LAT_Y, "$M$ learned\nlatents", ha="right",
            va="center", fontsize=14, color="black")
    ax.text(lat_x[0] - 0.65, OUT_Y, "latent\nsummary", ha="right",
            va="center", fontsize=14, color="black")

    fl.clean_axes(ax, lim=((-2.6, in_x[-1] + 3.0), (-0.55, OUT_Y + 0.55)),
                  hide=True)
    fl.save(fig, "mdl-transformers-latent-bottleneck")


def _unitless(path):
    """Rewrite matplotlib's pt-sized root width/height as unitless CSS px
    (1 pt = 4/3 px), keeping the viewBox (figure guide: unitless sizes)."""
    with open(path) as fh:
        svg = fh.read()
    head, rest = svg.split("<svg", 1)
    tag, body = rest.split(">", 1)
    tag = re.sub(r'(width|height)="([0-9.]+)pt"',
                 lambda m: f'{m.group(1)}="{float(m.group(2)) * 4 / 3:.2f}"',
                 tag)
    with open(path, "w") as fh:
        fh.write(head + "<svg" + tag + ">" + body)


if __name__ == "__main__":
    fl.WRITTEN.clear()
    fig_three_wirings()
    fig_latent_bottleneck()
    for path in fl.WRITTEN:
        _unitless(path)
        print("wrote", os.path.relpath(path, os.path.dirname(fl.IMG_DIR)))
