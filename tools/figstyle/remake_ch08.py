#!/usr/bin/env python3
"""Chapter 8 (modern convnets) legacy-figure remakes — chapter-by-chapter
review loop (docs/figure-style-guide.md §9).  UNDER REVIEW.

Writes -> img/:
  functionclasses.svg   the reviewed pilot, promoted to its home chapter
  residual-block.svg    regular vs residual block (original composition,
                        token colors)

The chapter's thirteen arch-* figures are owned by
gen_arch_convmodern_figures.py on the tokenized arch_diagrams library.
filters.png / regnet-fig.png are reproduced paper figures — untouched.

Run:  python3 tools/figstyle/remake_ch08.py
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from figstyle import tokens as T
from figstyle.svg import Figure, Span, sub, var

IMG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "img")


# --------------------------------------------------------------------------- #
# functionclasses — the reviewed pilot (non-monotonic non-nested distances),  #
# promoted verbatim from the pilot batch.                                     #
# --------------------------------------------------------------------------- #

def fig_functionclasses():
    """The ORIGINAL figure's organic blob shapes, verbatim (extracted from
    the legacy SVG into _functionclasses_paths.py) — only the fills,
    strokes, and labels are re-tokenized.  Alex, ch8 review: replicate the
    original for both panels."""
    from figstyle._functionclasses_paths import BLOBS, STARS

    f = Figure()
    S = 1.5                                      # original canvas x 1.5
    FILL = {100: T.PAPER, 80: "#F0F2F5", 60: "#E2E6EA",
            40: "#D2D8DD", 20: "#C0C7CE", 0: T.BLUE.tint}

    parts = []
    for lvl, d in BLOBS:
        stroke = T.BLUE.base if lvl == 0 else T.MUTED
        sw = 1.35 if lvl == 0 else 1.2           # pre-scale widths
        parts.append(f'<path d="{d}" fill="{FILL[lvl]}" stroke="{stroke}"'
                     f' stroke-width="{sw}"/>')
    for d in STARS:
        parts.append(f'<path d="{d}" fill="{T.GOLD.tint}"'
                     f' stroke="{T.GOLD.base}" stroke-width="1.2"/>')
    f.raw(f'<g transform="scale({S})">' + "".join(parts) + "</g>",
          0, 0, 465 * S, 191 * S)

    def flabel(x, y, n, color=T.INK):
        f.text(x * S, y * S, [Span("F", "i"), sub(str(n))],
               size=T.FS_LABEL, color=color)

    # label positions read from the original's glyph placements
    flabel(52.5, 82, 6)
    flabel(81.6, 82, 5)
    flabel(116.9, 82, 4)
    flabel(152.2, 71.8, 3)
    flabel(182.6, 81.8, 2)
    flabel(216.2, 103.5, 1, T.BLUE.dark)
    flabel(289.6, 95.2, 6)
    flabel(307.8, 95.2, 5)
    flabel(324.3, 95.2, 4)
    flabel(377.2, 104.5, 1, T.BLUE.dark)
    flabel(377.2, 128.5, 2)
    flabel(377.2, 141.2, 3)
    for x in (214.0, 373.4):
        f.text(x * S, 12 * S, [var("f"), Span("*", "r", script=1)],
               size=T.FS_LABEL, anchor="start")
    f.text(125 * S, 182 * S, "non-nested function classes",
           size=T.FS_LABEL, color=T.INK)
    f.text(371 * S, 182 * S, "nested function classes",
           size=T.FS_LABEL, color=T.INK)

    f.save("functionclasses", out_dir=IMG,
           desc="Non-nested versus nested function classes around f*.")


# --------------------------------------------------------------------------- #
# residual-block — POINT: a residual block learns g(x) and adds x back, so    #
# the block only has to learn a CORRECTION.  Original two-panel composition   #
# (regular block left, residual block right), token colors only.             #
# --------------------------------------------------------------------------- #

def fig_residual_block():
    f = Figure()
    BW = 200.0

    def panel(ox, residual):
        # inside the dashed box, bottom -> top: weight, activation, weight
        ys = [258.0, 196.0, 134.0]
        labels = ["weight layer", "activation function", "weight layer"]
        accents = [T.BLUE, T.BLUE, None]     # original shades the lower two
        f.rect(ox - BW / 2 - 20, 108, BW + 40, 190, fill="none",
               stroke=T.INK, sw=T.SW_BOX, dash=T.DASH_SOFT)
        for y, lab, a in zip(ys, labels, accents):
            f.block(ox, y, lab, accent=a, min_w=BW)
        # one CONTIGUOUS arrow from x through the dashed border into the
        # first block (the border is permeable — it is not a node)
        f.arrow(ox, 352, ox, 281, stroke=T.INK)
        for ya, yb in ((237, 217), (175, 155)):
            f.arrow(ox, ya, ox, yb, stroke=T.INK)
        # exit arrow: to the (+) on the right, ONTO the activation box on
        # the left (it must touch it)
        f.arrow(ox, 113, ox, 72 if residual else 11, stroke=T.INK)
        f.text(ox - 12, 88, [var("g" if residual else "f"), Span("(", "r"),
                             var("x"), Span(")", "r")],
               size=T.FS_LABEL, anchor="end")
        if residual:
            # (+) centered between the dashed box and the activation box
            f.pill_op(ox, 58, "+")
            # the skip: x routed around the dashed box into the (+)
            f.ortho_arrow([(ox, 330), (ox + BW / 2 + 52, 330),
                           (ox + BW / 2 + 52, 58), (ox + 15, 58)],
                          stroke=T.INK)
            f.text(ox + BW / 2 + 66, 74, [var("x")], size=T.FS_LABEL,
                   anchor="start")
            f.text(ox + 20, 30, [var("f"), Span("(", "r"), var("x"),
                                 Span(") = ", "r"), var("g"), Span("(", "r"),
                                 var("x"), Span(") + ", "r"), var("x")],
                   size=T.FS_SMALL, anchor="start")
            f.arrow(ox, 44, ox, 11, stroke=T.INK)
        # the top activation, outside the block
        f.block(ox, -12, "activation function", min_w=BW)
        f.arrow(ox, -34, ox, -54, stroke=T.INK)
        # input
        f.text(ox, 372, [var("x")], size=T.FS_LABEL)

    panel(0.0, residual=False)
    panel(400.0, residual=True)
    f.save("residual-block", out_dir=IMG,
           desc="A regular block versus a residual block: learn a correction.")


def main():
    fig_functionclasses()
    fig_residual_block()
    print("wrote chapter-8 remakes to", IMG)


if __name__ == "__main__":
    main()
