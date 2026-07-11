#!/usr/bin/env python3
"""Gallery-style (Raschka-derived) figures for the Builders' Guide (chapter 6).

These are the ch6 figures that have been *restyled* from the original house
style into the network-architecture gallery style
(``tools/arch_diagrams.py`` / docs/convnet-rewrite/figure-style.md): vertical
spine, white pills, an accent-tinted block container, rectilinear skips into a
⊕.  As more ch6 figures migrate to the gallery style they move here from
``gen_bg_<topic>_figures.py`` (which stays house style).  Kept in a separate
file so the two families never share global rcParams (each is byte-idempotent
under its own ``svg.hashsalt``).

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_bg_arch_figures.py
"""

from __future__ import annotations

import os

from arch_diagrams import (
    Diagram, save, WRITTEN,
    ACCENT, ACCENT_TINT, INK, GRAY_TEXT,
    PILL_H, PILL_STEP, PLUS_R, LABEL_BAND,
)


# =========================================================================== #
# model-construction.md, "Forward Is Just Python": residual wiring x + body(x) #
# =========================================================================== #

def fig_residual_block():
    """The wiring of ``x + body(x)`` as a vertical residual block, laid out
    like ``arch-resnet-block``: the main line runs up through Linear -> ReLU ->
    Linear (the body, in the accent-tinted block panel); an identity skip
    leaves below the first pill, routes up the side, and rejoins the spine at a
    ⊕; the sum leaves the top as ``x + body(x)``."""
    x = 195
    pw = 128
    x_out = x + 92                       # skip routes up this column
    y0 = 116                             # first (bottom) pill center
    ops = ["Linear", "ReLU", "Linear"]

    ys = [y0 + i * PILL_STEP for i in range(len(ops))]
    y_plus = ys[-1] + PILL_STEP          # ⊕ one step above the last op

    panel = (x - pw / 2 - 22, y0 - PILL_H / 2 - 23,
             x_out + 22, y_plus + PLUS_R + LABEL_BAND + 12)
    W, H = panel[2] + 180, panel[3] + 46   # generous; save() crops to content

    d = Diagram(W, H)
    d.container(*panel, fill=ACCENT_TINT)
    # two lines so the label clears the black output arrow on the spine
    d.stage_label(panel[0] + 14, panel[3] - 9, "Residual\nBlock")

    # spine: input anchor -> body ops -> ⊕ -> output
    y_anchor = panel[1] - 34
    d.arrow(x, y_anchor + 13, ys[0] - PILL_H / 2)
    for i, (op, y) in enumerate(zip(ops, ys)):
        d.pill(x, y, op, w=pw)
        if i < len(ops) - 1:
            d.arrow(x, y + PILL_H / 2, ys[i + 1] - PILL_H / 2)
    d.arrow(x, ys[-1] + PILL_H / 2, y_plus - PLUS_R)

    # identity skip: leave below the first pill, up the side, into the ⊕.
    # Label sits fully to the right, outside the blue panel.
    y_branch = y0 - PILL_H / 2 - 11
    d.skip(x, y_branch, y_plus, x_out)
    d.shape_note(panel[2] + 12, 0.5 * (y_branch + y_plus), "identity", ha="left")

    # sum leaves the top
    y_top_arrow = panel[3] + 28
    d.arrow(x, y_plus + PLUS_R, y_top_arrow)
    d.anchor(x, y_top_arrow + 14, "x + body(x)")
    d.anchor(x, y_anchor, "x")

    # explaining text in the right margin (no leader: the ⊕ is self-evident)
    lh = 12.5 * 1.25
    d.callout(panel[2] + 12, y_plus + lh, [
        [("body learns a", INK, 1)],
        [("correction ", INK, 1), ("body(x)", ACCENT, 1)],
        [("added back to ", INK, 1), ("x", ACCENT, 1)],
    ], ha="left")

    save(d.fig, "bg-residual-block")


FIGURES = [fig_residual_block]


def main():
    start = len(WRITTEN)
    for fn in FIGURES:
        fn()
    written = WRITTEN[start:]
    print(f"\nWrote {len(written)} figures:")
    for p in written:
        size = os.path.getsize(p)
        assert size > 0 and os.path.exists(p)
        with open(p, "r", encoding="utf-8") as fh:
            assert "<svg" in fh.read(400)
        print(f"  {os.path.basename(p):30s} {size:>8,d} bytes")
    print(f"\nAll {len(written)} SVGs verified present, non-empty, valid.")


if __name__ == "__main__":
    main()
