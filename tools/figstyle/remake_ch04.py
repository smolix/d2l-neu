#!/usr/bin/env python3
"""Chapter 4 (linear classification) legacy-figure remake — chapter-by-
chapter review loop (docs/figure-style-guide.md §9).  UNDER REVIEW.

Writes -> img/:
  softmaxreg.svg    softmax regression as a fully connected single layer

The chapter's five mdl-clf-* figures are owned by
gen_mdl_classification_figures.py / gen_mdl_vcdim_figures.py, which
inherit the token layer through gen_mdl_figures.py.  The chapter's PNGs
(cat-dog photos, popvssoda map) are out of scope.

Run:  python3 tools/figstyle/remake_ch04.py
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from figstyle import tokens as T
from figstyle.svg import Figure, sub, var

IMG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "img")


# --------------------------------------------------------------------------- #
# softmaxreg — POINT: softmax regression is one fully connected layer:        #
# every input feeds every output.  Same family as mlp.svg / singleneuron.svg. #
# --------------------------------------------------------------------------- #

def fig_softmaxreg():
    f = Figure()
    R = 21.0
    Y_IN, Y_OUT = 130.0, 0.0
    XIN = [155.0, 265.0, 375.0, 485.0]
    XOUT = [210.0, 320.0, 430.0]

    # edges first (all-to-all), quiet gray with heads
    for xa in XIN:
        for xb in XOUT:
            ang = math.atan2(Y_OUT - Y_IN, xb - xa)
            f.arrow(xa + R * math.cos(ang), Y_IN + R * math.sin(ang),
                    xb - (R + 2) * math.cos(ang),
                    Y_OUT - (R + 2) * math.sin(ang),
                    stroke=T.FAINT, sw=T.SW_HAIR)

    for i, x in enumerate(XIN, 1):
        f.circle(x, Y_IN, R, fill=T.PAPER, stroke=T.INK, sw=T.SW_BOX)
        f.text(x, Y_IN, [var("x"), sub(str(i))], size=T.FS_LABEL)
    for i, x in enumerate(XOUT, 1):
        f.circle(x, Y_OUT, R, fill=T.GREEN.tint, stroke=T.GREEN.base,
                 sw=T.SW_BOX)
        f.text(x, Y_OUT, [var("o"), sub(str(i))], size=T.FS_LABEL)

    lx = XIN[0] - 180
    f.text(lx, Y_OUT, "OUTPUT LAYER", size=T.FS_TINY, color=T.MUTED,
           anchor="start", tracking=T.LETTERSPACE_CAPS)
    f.text(lx, Y_IN, "INPUT LAYER", size=T.FS_TINY, color=T.MUTED,
           anchor="start", tracking=T.LETTERSPACE_CAPS)

    f.save("softmaxreg", out_dir=IMG,
           desc="Softmax regression: one fully connected layer.")


def main():
    fig_softmaxreg()
    print("wrote chapter-4 legacy remake to", IMG)


if __name__ == "__main__":
    main()
