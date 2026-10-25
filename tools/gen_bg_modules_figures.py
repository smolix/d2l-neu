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

FIGURES = [fig_weight_tying]


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
