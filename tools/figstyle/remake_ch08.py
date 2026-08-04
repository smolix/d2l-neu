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

def star(f, cx, cy, r_out=18.0, r_in=7.2, color=T.GOLD):
    pts = []
    for i in range(10):
        r = r_out if i % 2 == 0 else r_in
        a = -math.pi / 2 + i * math.pi / 5
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    d = "M" + " L".join(f"{x:.2f} {y:.2f}" for x, y in pts) + " Z"
    f.path(d, cx - r_out, cy - r_out, cx + r_out, cy + r_out,
           fill=color.tint, stroke=color.base, sw=T.SW_BOX)


def fig_functionclasses():
    """POINT: with non-nested classes, growing capacity does NOT
    monotonically approach f*; nested classes can only improve."""
    f = Figure()
    GRAYS = [T.PAPER, "#F0F2F5", "#E2E6EA", "#D2D8DD", "#C0C7CE"]

    def flabel(x, y, n, color=T.INK):
        f.text(x, y, [Span("F", "i"), sub(str(n))], size=T.FS_LABEL,
               color=color)

    star(f, 350, -20)
    f.text(374, -20, [var("f"), Span("*", "r", script=1)], size=T.FS_LABEL,
           anchor="start")
    rects = [(0, 10, 240, 180), (30, 70, 210, 170), (120, -10, 190, 150),
             (95, 95, 180, 140), (215, 20, 145, 115)]
    for i, (x, y, w, h) in enumerate(rects):
        f.rect(x, y, w, h, fill=GRAYS[i], stroke=T.MUTED, sw=T.SW_HAIR, r=26)
    f.rect(35, 120, 95, 70, fill=T.BLUE.tint, stroke=T.BLUE.base,
           sw=T.SW_BOX, r=20)
    flabel(24, 34, 6)
    flabel(52, 222, 5)
    flabel(288, 12, 4)
    flabel(248, 214, 3)
    flabel(336, 44, 2)
    flabel(82, 155, 1, T.BLUE.dark)
    f.text(178, 282, "non-nested function classes", size=T.FS_LABEL,
           color=T.INK)

    X0 = 520.0
    nested = [(X0, 0, 300, 230), (X0 + 30, 12, 246, 206),
              (X0 + 60, 24, 192, 182), (X0 + 90, 36, 138, 158),
              (X0 + 112, 48, 94, 116)]
    for i, (x, y, w, h) in enumerate(nested):
        f.rect(x, y, w, h, fill=GRAYS[i], stroke=T.MUTED, sw=T.SW_HAIR, r=22)
    f.rect(X0 + 124, 60, 70, 72, fill=T.BLUE.tint, stroke=T.BLUE.base,
           sw=T.SW_BOX, r=14)
    for i in range(3):
        flabel(X0 + 15 + 30 * i, 115, 6 - i)
    flabel(X0 + 159, 179, 3)
    flabel(X0 + 159, 148, 2)
    flabel(X0 + 159, 96, 1, T.BLUE.dark)
    star(f, X0 + 296, -34)
    f.text(X0 + 320, -34, [var("f"), Span("*", "r", script=1)],
           size=T.FS_LABEL, anchor="start")
    f.text(X0 + 150, 282, "nested function classes", size=T.FS_LABEL,
           color=T.INK)

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
        for ya, yb in ((300, 279), (237, 217), (175, 155)):
            f.arrow(ox, ya, ox, yb, stroke=T.INK)
        # exit arrow, labelled f(x) (left) / g(x) (right)
        f.arrow(ox, 113, ox, 62 if residual else 40, stroke=T.INK)
        f.text(ox - 12, 88, [var("g" if residual else "f"), Span("(", "r"),
                             var("x"), Span(")", "r")],
               size=T.FS_LABEL, anchor="end")
        if residual:
            f.pill_op(ox, 48, "+")
            # the skip: x routed around the dashed box into the (+)
            f.ortho_arrow([(ox, 330), (ox + BW / 2 + 52, 330),
                           (ox + BW / 2 + 52, 48), (ox + 15, 48)],
                          stroke=T.INK)
            f.text(ox + BW / 2 + 66, 64, [var("x")], size=T.FS_LABEL,
                   anchor="start")
            f.text(ox + 16, 22, [var("f"), Span("(", "r"), var("x"),
                                 Span(") = ", "r"), var("g"), Span("(", "r"),
                                 var("x"), Span(") + ", "r"), var("x")],
                   size=T.FS_SMALL, anchor="start")
            f.arrow(ox, 36, ox, 16, stroke=T.INK)
        # the top activation, outside the block
        f.block(ox, -12, "activation function", min_w=BW)
        f.arrow(ox, -34, ox, -54, stroke=T.INK)
        # input
        f.arrow(ox, 352, ox, 302, stroke=T.INK)
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
