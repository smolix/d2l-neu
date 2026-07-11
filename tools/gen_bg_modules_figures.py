#!/usr/bin/env python3
"""Generate the illustrative figures for the "Builders' Guide" module-system
sections (``chapter_builders-guide-v2/model-construction.md`` and
``chapter_builders-guide-v2/parameters-state-memory.md``) in the one shared
house style defined in ``gen_mdl_figures.py``.

The notebooks / prose reference the generated files with no drawing code
(like the slide SVGs).

Run with the repo's pytorch venv (matplotlib + numpy are available):

    .venv-pytorch/bin/python tools/gen_bg_modules_figures.py

All figures are written to ``img/bg-<id>.svg``.  The script is idempotent:
re-running overwrites byte-for-byte (fixed ``svg.hashsalt`` + no timestamp
metadata, both set once in ``gen_mdl_figures.py``).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


# --------------------------------------------------------------------------- #
# Small chapter-local helpers shared by the four figures.                     #
# --------------------------------------------------------------------------- #

def box(ax, cx, cy, w, h, color, head, sub=None, fontsize=13, subsize=11,
        subcolor="black", zorder=None):
    """A rounded box (fill + crisp edge, the standard two-layer trick used
    across the mdl figure generators) with a bold coloured header and an
    optional black subtitle line.  Returns anchor points for connectors.
    ``zorder`` (when given) sets the stacking of both patch layers and the
    text, so a box can be drawn on top of connectors routed behind it."""
    x0, y0 = cx - w / 2, cy - h / 2
    pk = {} if zorder is None else {"zorder": zorder}
    tk = {} if zorder is None else {"zorder": zorder + 1}
    if zorder is not None:
        # opaque white base so connectors routed BEHIND the box are fully
        # hidden (not showing through the translucent colour fill).
        ax.add_patch(FancyBboxPatch(
            (x0, y0), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
            linewidth=0, edgecolor="none", facecolor="white", alpha=1.0, **pk))
    for fc, a in [(color, 0.14), ("none", 1.0)]:
        ax.add_patch(FancyBboxPatch(
            (x0, y0), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
            linewidth=1.6, edgecolor=color, facecolor=fc, alpha=a, **pk))
    if sub:
        ax.text(cx, cy + 0.15 * h, head, ha="center", va="center",
                fontsize=fontsize, color=color, fontweight="bold", **tk)
        ax.text(cx, cy - 0.34 * h, sub, ha="center", va="center",
                fontsize=subsize, color=subcolor, **tk)
    else:
        ax.text(cx, cy, head, ha="center", va="center", fontsize=fontsize,
                color=color, fontweight="bold", **tk)
    return dict(cx=cx, cy=cy, top=(cx, cy + h / 2), bottom=(cx, cy - h / 2),
                left=(cx - w / 2, cy), right=(cx + w / 2, cy))


def elbow(ax, parent_pt, child_pts, y_trunk, color=GRAY, lw=1.3):
    """Tree connector: a vertical drop from ``parent_pt`` to a horizontal
    trunk at ``y_trunk``, then a vertical drop from the trunk into each of
    ``child_pts``.  Plain lines (no arrowheads): this is a "contains"
    hierarchy, not a data-flow diagram."""
    px, py = parent_pt
    ax.plot([px, px], [py, y_trunk], color=color, lw=lw, zorder=1,
            solid_capstyle="round")
    xs = [c[0] for c in child_pts]
    ax.plot([min(xs), max(xs)], [y_trunk, y_trunk], color=color, lw=lw,
            zorder=1, solid_capstyle="round")
    for cx, cy in child_pts:
        ax.plot([cx, cx], [y_trunk, cy], color=color, lw=lw, zorder=1,
                solid_capstyle="round")


# =========================================================================== #
# 1. model-construction.md: the module tree                                  #
# =========================================================================== #

def fig_module_tree():
    """Replaces the old hand-drawn ``blocks.svg``.  Bottom row: small layer
    boxes; grouped into block boxes; composing into one model box at top.
    Plain "contains" connectors (no arrowheads) since this is a hierarchy, not
    a data-flow graph.  Colour encodes tree level (layer/block/model), echoing
    "layers compose into blocks, blocks compose into models; every model is a
    tree.\""""
    fig, ax = plt.subplots(figsize=(8.0, 4.6))

    model = box(ax, 6.0, 6.05, 3.0, 0.85, ORANGE, "model", fontsize=14)

    block_xs = [2.2, 6.0, 9.8]
    blocks = [box(ax, bx, 3.85, 2.6, 0.8, GREEN, "block", fontsize=12.5)
              for bx in block_xs]

    layer_offsets = [-0.98, 0.0, 0.98]
    layer_rows = []
    for bx in block_xs:
        row = [box(ax, bx + off, 1.5, 0.72, 0.52, BLUE, "layer", fontsize=10)
               for off in layer_offsets]
        layer_rows.append(row)

    elbow(ax, model["bottom"], [b["top"] for b in blocks], y_trunk=5.30)
    for blk, row in zip(blocks, layer_rows):
        elbow(ax, blk["bottom"], [l["top"] for l in row], y_trunk=2.55)

    ax.set_xlim(0.55, 11.45)
    ax.set_ylim(0.95, 6.75)
    ax.set_aspect("equal")
    ax.axis("off")
    fl.save(fig, "bg-module-tree")


# =========================================================================== #
# 2. parameters-state-memory.md, "Tied Parameters": one shared matrix         #
# =========================================================================== #

def fig_weight_tying():
    """One shared weight matrix W (a tall V x d rectangle) referenced from two
    call sites: the embedding (token id -> row of W) and the output head
    (logits = h W^T).  Minimal: two use-site boxes plus the matrix -- both
    arrows point INTO the matrix, since the picture is about aliasing (two
    pointers, one tensor), not data flow."""
    fig, ax = plt.subplots(figsize=(7.4, 4.7))

    # the shared matrix: tall and narrow (|V| rows >> d columns), with a few
    # faint guide lines suggesting rows.  Shortened by one "cell" top and
    # bottom (4 cells instead of 6) so the figure is more compact.
    mx, my, mw = 5.0, 2.55, 1.5
    cell = 3.5 / 6.0
    mh = 4 * cell
    m0x, m0y = mx - mw / 2, my - mh / 2
    m_top = m0y + mh
    ax.add_patch(Rectangle((m0x, m0y), mw, mh, facecolor=GRAY, alpha=0.16,
                           edgecolor="black", lw=1.8, zorder=2))
    for k in range(1, 4):
        yy = m0y + k * mh / 4
        ax.plot([m0x, m0x + mw], [yy, yy], color=GRAY, lw=0.8, alpha=0.6,
                zorder=2)
    ax.text(mx, my + 0.30, r"$\mathbf{W}$", ha="center", va="center",
            fontsize=20, color="black", zorder=4)
    ax.text(mx, my - 0.34, r"$|V|\times d$", ha="center", va="center",
            fontsize=13, color="black", zorder=4)

    # both call sites reference the same tensor: draw the arrows FIRST, at a
    # z-order above the grey matrix but below the two coloured boxes, and start
    # each tail from inside its box so the box (drawn on top) hides the origin
    # and the arrow emerges cleanly from the box's lower edge.
    yb = 4.55
    for cx, tgt_dx, color in [(1.75, -0.55, GREEN), (8.25, 0.55, ORANGE)]:
        ax.add_patch(FancyArrowPatch(
            (cx, yb), (mx + tgt_dx, m_top - 0.30), arrowstyle="-|>",
            mutation_scale=16, color=color, lw=2.2, zorder=3,
            shrinkA=0, shrinkB=0))

    emb = box(ax, 1.75, yb, 3.1, 1.05, GREEN, "embedding",
              "token id $\\to$ row of $\\mathbf{W}$", fontsize=13.5,
              subsize=11, zorder=5)
    head = box(ax, 8.25, yb, 3.1, 1.05, ORANGE, "output head",
               r"logits $=h\,\mathbf{W}^\top$", fontsize=13.5, subsize=11,
               zorder=5)

    ax.text(mx, m0y - 0.42, "one tensor, two call sites: gradients sum",
            ha="center", va="center", fontsize=12, color="black")

    ax.set_xlim(0.1, 9.9)
    ax.set_ylim(0.55, 5.35)
    ax.set_aspect("equal")
    ax.axis("off")
    fl.save(fig, "bg-weight-tying")


# =========================================================================== #
# 3. parameters-state-memory.md, "Counting Parameters, Counting Bytes"        #
# =========================================================================== #

def fig_memory_ledger():
    """Two aligned horizontal stacked bars, same width-per-byte, so areas
    compare honestly: fp32 Adam training (weights 4 + grads 4 + m 4 + v 4 =
    16 B) above mixed-precision / ZeRO-style training (fp16 weight 2 + fp16
    grad 2 + fp32 master weight 4 + m 4 + v 4 = 16 B).  The right half (the
    two Adam moments) is byte-identical and pixel-aligned between the two
    bars; only the left half (weights/gradients) differs."""
    fig, ax = plt.subplots(figsize=(8.4, 3.6))

    bar_h = 0.85
    y_top, y_bot = 3.05, 1.05

    def seg(y, x0, w, color, alpha, label, leader=False, label_dx=0.0):
        ax.add_patch(Rectangle((x0, y), w, bar_h, facecolor=color, alpha=alpha,
                               edgecolor="white", lw=1.4, zorder=2))
        cx = x0 + w / 2
        if leader:
            # elbow leader: a short stub straight up from the segment centre,
            # then a jog to ``label_dx`` away from it so two adjacent narrow
            # (2-byte) segments' labels don't collide with each other.
            lx = cx + label_dx
            yk = y + bar_h + 0.20
            ax.plot([cx, cx], [y + bar_h, yk], color="black", lw=1.0, zorder=3)
            ax.plot([cx, lx], [yk, yk], color="black", lw=1.0, zorder=3)
            ax.plot([lx, lx], [yk, y + bar_h + 0.42], color="black", lw=1.0,
                    zorder=3)
            ax.text(lx, y + bar_h + 0.52, label, ha="center", va="bottom",
                    fontsize=11, color="black")
        else:
            ax.text(cx, y + bar_h / 2, label, ha="center", va="center",
                    fontsize=11.5, color="black")

    # row 1: fp32 Adam, 4 equal 4-byte segments
    seg(y_top, 0, 4, BLUE, 0.55, "weights\n4 B")
    seg(y_top, 4, 4, ORANGE, 0.55, "grads\n4 B")
    seg(y_top, 8, 4, GREEN, 0.55, "$m$\n4 B")
    seg(y_top, 12, 4, GRAY, 0.55, "$v$\n4 B")
    ax.text(-0.35, y_top + bar_h / 2, "fp32 Adam", ha="right", va="center",
            fontsize=12.5, color="black")
    ax.text(16.35, y_top + bar_h / 2, "16 B", ha="left", va="center",
            fontsize=12.5, color="black", fontweight="bold")

    # row 2: mixed precision / ZeRO -- same total width (16 units), narrow
    # 2-byte segments get a leader line so their labels don't get truncated
    seg(y_bot, 0, 2, BLUE, 0.30, "fp16 wt\n2 B", leader=True, label_dx=-0.7)
    seg(y_bot, 2, 2, ORANGE, 0.30, "fp16 grad\n2 B", leader=True, label_dx=0.9)
    seg(y_bot, 4, 4, BLUE, 0.55, "fp32 master\n4 B")
    seg(y_bot, 8, 4, GREEN, 0.55, "$m$\n4 B")
    seg(y_bot, 12, 4, GRAY, 0.55, "$v$\n4 B")
    ax.text(-0.35, y_bot + bar_h / 2, "mixed precision\n(ZeRO)", ha="right",
            va="center", fontsize=12.5, color="black")
    ax.text(16.35, y_bot + bar_h / 2, "16 B", ha="left", va="center",
            fontsize=12.5, color="black", fontweight="bold")

    # shared byte axis below both bars (black, per house style)
    y_axis = y_bot - 0.30
    ax.plot([0, 16], [y_axis, y_axis], color="black", lw=1.1)
    for k in range(0, 17, 4):
        ax.plot([k, k], [y_axis - 0.06, y_axis + 0.06], color="black", lw=1.1)
        ax.text(k, y_axis - 0.20, f"{k} B", ha="center", va="top",
                fontsize=10.5, color="black")

    ax.set_xlim(-3.3, 18.3)
    ax.set_ylim(y_axis - 0.55, y_top + bar_h + 0.45)
    # NOT equal aspect: with an x-span of ~22 units and a y-span of ~5, a
    # square-pixel constraint would squeeze the whole drawing into a thin
    # strip of the figure height, blowing the (fixed-point-size) labels up
    # relative to the row spacing and running them into each other. This is a
    # bar chart, not a geometric mapping, so free aspect is the right choice.
    ax.set_aspect("auto")
    ax.axis("off")
    fl.save(fig, "bg-memory-ledger")


# NB: bg-residual-block was restyled to the gallery look and now lives in
# tools/gen_bg_arch_figures.py (kept separate so the two families don't share
# global rcParams).
FIGURES = [fig_module_tree, fig_weight_tying, fig_memory_ledger]


def main():
    # Verify only the figures THIS script writes (the shared module's WRITTEN
    # list also tracks every other chapter's figures, which we don't run
    # here).
    start = len(fl.WRITTEN)
    for fn in FIGURES:
        fn()
    written = fl.WRITTEN[start:]
    print(f"\nWrote {len(written)} figures to {fl.IMG_DIR}:")
    for p in written:
        size = os.path.getsize(p)
        assert os.path.exists(p), f"missing: {p}"
        assert size > 0, f"empty: {p}"
        with open(p, "r", encoding="utf-8") as fh:
            assert "<svg" in fh.read(400), f"not valid SVG: {p}"
        print(f"  {os.path.basename(p):32s} {size:>8,d} bytes")
    print(f"\nAll {len(written)} SVGs verified present, non-empty, valid.")


if __name__ == "__main__":
    main()
