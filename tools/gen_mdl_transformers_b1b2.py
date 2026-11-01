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
    # zorder above the wires: arrow ends may tuck under the (opaque) box
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.10",
        fc=fc, ec=ec, lw=1.6, zorder=3.5))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center",
            fontsize=13, color="black", zorder=4)


def _add_circle(ax, y):
    ax.add_patch(Circle((SX, y), 0.30, fc="white", ec="black", lw=1.5,
                        zorder=5))
    ax.text(SX, y - 0.02, "+", ha="center", va="center", fontsize=15,
            color="black", zorder=6)


def _stream(ax, y0, y1, head=True, x=None):
    """A stream segment; head=False when the segment runs into a box or an
    (+) with no room for an arrowhead -- the end tucks under the opaque
    box instead."""
    x = SX if x is None else x
    if head:
        fl.arrow(ax, (x, y0), (x, y1), color=BLUE, lw=3.0, mut=18)
    else:
        ax.plot([x, x], [y0, y1], color=BLUE, lw=3.0, zorder=2,
                solid_capstyle="round")


def _elbow(ax, pts, color, lw=1.8, arrow_end=True):
    """Polyline through pts (mitred corners, butt ends, so nothing overshoots
    a turn); arrowhead on the final segment."""
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="butt",
            solid_joinstyle="miter", zorder=2)
    if arrow_end:
        fl.arrow(ax, pts[-2], pts[-1], color=color, lw=lw, mut=13)


# Shared vertical rhythm for both block-anatomy panels: the two sublayers sit
# at the same heights, so only the norm boxes differ between post-LN and
# pre-LN.  Box extents are the FancyBboxPatch interiors (pad 0.10 outside).
PAD = 0.10
Y_READ = (0.9, 5.6)                 # where each branch reads the stream
Y_BOX = ((2.0, 3.1), (6.7, 7.8))    # attention / FFN box interiors
Y_ADD = (3.75, 8.45)                # where each branch writes (the +)
Y_TOP = 10.6
BC = 6.9                            # branch column (box centre)
SUBLAYERS = (("multi-head\nattention", "ORANGE"),
             ("feed-forward\nnetwork", "GREEN"))


def _sublayer(ax, k, prenorm):
    text, cname = SUBLAYERS[k]
    col, tint = getattr(fl.T, cname).base, getattr(fl.T, cname).tint
    y_r, (b0, b1), y_a = Y_READ[k], Y_BOX[k], Y_ADD[k]
    _box(ax, BX0, BX1, b0, b1, text, tint, col)
    if prenorm:
        # norm on the branch, at the read height
        n1 = BX1 - 2.7
        _box(ax, BX0, n1, y_r - 0.35, y_r + 0.35, "norm", "white", GRAY)
        _elbow(ax, [(SX, y_r), (BX0 - PAD, y_r)], col)
        _elbow(ax, [(n1 + PAD, y_r), (BC, y_r), (BC, b0 - PAD)], col)
    else:
        _elbow(ax, [(SX, y_r), (BC, y_r), (BC, b0 - PAD)], col)
    _elbow(ax, [(BC, b1 + PAD), (BC, y_a), (SX + 0.30, y_a)], col)
    _add_circle(ax, y_a)


def _panel_postln(ax):
    # stream segments (bottom -> top), interrupted by a norm box on the
    # stream just above each (+)
    y = 0.0
    for y_a in Y_ADD:
        n0, n1 = y_a + 0.68, y_a + 1.28
        _stream(ax, y, n0, head=False)
        _box(ax, SX - NW / 2, SX + NW / 2, n0, n1, "norm", "white", GRAY)
        y = n1
    _stream(ax, y, Y_TOP)
    for k in range(2):
        _sublayer(ax, k, prenorm=False)
    ax.set_title("post-LN (2017)", fontsize=14, color="black")


def _panel_preln(ax):
    # one uninterrupted stream: sublayers only add to it
    _stream(ax, 0.0, Y_TOP)
    for k in range(2):
        _sublayer(ax, k, prenorm=True)
    # label the untouched stream
    ax.text(SX - 0.45, 9.8, "residual\nstream", ha="right", va="center",
            fontsize=13, color=BLUE)
    ax.set_title("pre-LN (modern)", fontsize=14, color="black")


def fig_block_anatomy() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.6))
    for ax, panel in zip(axes, (_panel_postln, _panel_preln)):
        panel(ax)
        # input label left of the stream foot, clear of the branch lines
        ax.text(SX - 0.40, 0.25, r"$\mathbf{x}$", ha="right", va="center",
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


# --------------------------------------------------------------------------- #
# B1: normalization-placement taxonomy -- post-LN, pre-LN, post-sublayer      #
# (OLMo 2), pre+post (Gemma 3), one sublayer each on the same stream          #
# --------------------------------------------------------------------------- #

MSX = 1.5           # x of the residual stream in the mini panels
MBX0, MBX1 = 3.3, 7.7   # branch box left/right edges
MBC = (MBX0 + MBX1) / 2  # branch column center


def _mini_panel(ax, order, stream_norm, title):
    """One sublayer on a residual stream; `order` lists the branch boxes
    bottom-to-top ('norm'/'attn'); `stream_norm` puts a norm on the stream
    itself after the addition (post-LN)."""
    heights = {"norm": 0.85, "attn": 1.15}
    # one fixed height for all four panels: the tallest branch (Gemma 3:
    # norm, attention, norm) still leaves a clear stretch of stream above
    # its (+), so the panels are directly comparable
    # gaps leave every arrow a visible shaft: interiors are 1.25 apart, and
    # the box pads (0.10 each side) still leave ~1.05 of line + head
    gap, y_read, y_top = 1.25, 0.6, 9.3
    mnw = 2.3                          # stream-norm box width (fits "norm")

    # branch boxes, bottom to top
    y, spans = y_read + 1.15, []
    for kind in order:
        h = heights[kind]
        spans.append((kind, y, y + h))
        y += h + gap
    y_add = y - gap + 0.5

    # stream (interrupted by a norm box only for post-LN)
    if stream_norm:
        ax.add_patch(Circle((MSX, y_add), 0.30, fc="white", ec="black",
                            lw=1.5, zorder=5))
        n0, n1 = y_add + 0.72, y_add + 1.57
        _stream(ax, 0.0, n0 + 0.10, head=False, x=MSX)
        _box(ax, MSX - mnw / 2, MSX + mnw / 2, n0, n1, "norm", "white", GRAY)
        _stream(ax, n1 - 0.10, y_top, x=MSX)
    else:
        ax.add_patch(Circle((MSX, y_add), 0.30, fc="white", ec="black",
                            lw=1.5, zorder=5))
        fl.arrow(ax, (MSX, 0.0), (MSX, y_top), color=BLUE, lw=3.0, mut=18)
    ax.text(MSX, y_add - 0.02, "+", ha="center", va="center", fontsize=15,
            color="black", zorder=6)

    # the branch: read, boxes, write into the +
    _elbow(ax, [(MSX, y_read), (MBC, y_read), (MBC, spans[0][1] - 0.10)],
           ORANGE)
    for kind, y0, y1 in spans:
        if kind == "attn":
            _box(ax, MBX0, MBX1, y0, y1, "attention", fl.T.ORANGE.tint, ORANGE)
        else:
            _box(ax, MBC - mnw / 2, MBC + mnw / 2, y0, y1, "norm", "white",
                 GRAY)
    for (_, _, top), (_, nxt, _) in zip(spans, spans[1:]):
        _elbow(ax, [(MBC, top + 0.10), (MBC, nxt - 0.10)], ORANGE)
    _elbow(ax, [(MBC, spans[-1][2] + 0.10), (MBC, y_add), (MSX + 0.30, y_add)],
           ORANGE)

    # input label left of the stream foot, clear of the branch line
    ax.text(MSX - 0.40, 0.2, r"$\mathbf{x}$", ha="right", va="center",
            fontsize=14, color="black")
    ax.set_title(title, fontsize=13, color="black")
    ax.set_xlim(-0.5, 8.3)
    ax.set_ylim(-0.2, 9.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def fig_norm_taxonomy() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(10.6, 3.6))
    panels = [
        (["attn"], True, "post-LN\n(2017)"),
        (["norm", "attn"], False, "pre-LN\n(default)"),
        (["attn", "norm"], False, "post-sublayer\n(OLMo 2)"),
        (["norm", "attn", "norm"], False, "pre + post\n(Gemma 3)"),
    ]
    for ax, (order, stream_norm, title) in zip(axes, panels):
        _mini_panel(ax, order, stream_norm, title)
    fig.subplots_adjust(wspace=0.05)
    fl.save(fig, "mdl-transformers-norm-taxonomy")


if __name__ == "__main__":
    fig_block_anatomy()
    fig_norm_taxonomy()
    for path in fl.WRITTEN:
        print(f"wrote {os.path.relpath(path)}")
