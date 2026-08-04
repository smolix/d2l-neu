#!/usr/bin/env python3
"""Chapter 3 (linear regression) legacy-figure remakes — chapter-by-chapter
review loop (docs/figure-style-guide.md §9).  UNDER REVIEW.

Writes -> img/:
  fit-linreg.svg      data, fitted line, and residuals
  singleneuron.svg    linear regression as a single-neuron network

The chapter's eight mdl-* figures are owned by their gen_mdl_* generators,
which now inherit the token layer through gen_mdl_figures.py (guide §9
step 2).  img/neuron.svg (hand-drawn biological artwork) is out of scope —
never redraw, like photographs.

Run:  python3 tools/figstyle/remake_ch03.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from figstyle import tokens as T
from figstyle.svg import Figure, sub, var

IMG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "img")


# --------------------------------------------------------------------------- #
# fit-linreg — POINT: the fitted line minimizes the residuals, the vertical   #
# gaps between each observation y_i and its prediction y_hat_i.               #
# --------------------------------------------------------------------------- #

def fig_fit_linreg():
    import matplotlib
    matplotlib.use("svg")
    import matplotlib.pyplot as plt
    import numpy as np

    from figstyle.mpl import axis_cross, clean_axes, use_style, save

    use_style(hashsalt="figstyle-ch03")
    fig, ax = plt.subplots(figsize=(4.6, 3.5))

    # the ORIGINAL composition, verbatim (Alex: "that one is perfect") —
    # only the palette changes: ink line, token-blue stems and open circles
    xs = np.array([0.85, 1.5, 2.35, 3.35, 4.35])
    resid = np.array([-0.75, 0.55, 0.95, -0.85, 0.65])
    line = lambda x: 0.52 * x + 0.9
    ys = line(xs) + resid

    axis_cross(ax, (-0.25, 5.3), (-0.3, 4.4))
    ax.plot([0.15, 5.05], [line(0.15), line(5.05)], color=T.INK,
            lw=2.4, zorder=3)
    for x, y in zip(xs, ys):
        ax.plot([x, x], [line(x), y], color=T.BLUE.base, lw=2.4, zorder=2)
    ax.plot(xs, ys, "o", mfc="white", mec=T.BLUE.base, mew=2.2, ms=7.5,
            zorder=4)

    # labels exactly as in the original: y-hat above the line at the 4th
    # stem's intercept, y at that stem's observation circle
    xi = xs[3]
    ax.text(xi - 0.08, line(xi) + 0.22, r"$\hat{y}^{(i)}$",
            color=T.INK, fontsize=16, ha="right", va="bottom")
    ax.text(xi + 0.16, ys[3] - 0.10, r"$y^{(i)}$", color=T.INK, fontsize=16,
            ha="left", va="top")

    ax.text(5.15, -0.42, r"$x$", color=T.INK, fontsize=17, ha="center")
    ax.text(-0.42, 4.25, r"$y$", color=T.INK, fontsize=17, va="center")
    clean_axes(ax, lim=((-0.55, 5.45), (-0.55, 4.5)), hide=True)
    save(fig, "fit-linreg", out_dir=IMG)


# --------------------------------------------------------------------------- #
# singleneuron — POINT: linear regression is a one-neuron network: d inputs,  #
# one output.  Matches the approved img/mlp.svg style exactly.                #
# --------------------------------------------------------------------------- #

def fig_singleneuron():
    import math

    f = Figure()
    R = 21.0
    Y_IN, Y_OUT = 130.0, 0.0
    XIN = [150.0, 260.0, 370.0, 480.0]     # x1, x2, (dots), xd
    XOUT = 315.0

    # edges first, quiet gray with heads (feed-forward semantics)
    for i, x in enumerate(XIN):
        if i == 2:
            continue
        ang = math.atan2(Y_OUT - Y_IN, XOUT - x)
        f.arrow(x + R * math.cos(ang), Y_IN + R * math.sin(ang),
                XOUT - (R + 2) * math.cos(ang),
                Y_OUT - (R + 2) * math.sin(ang),
                stroke=T.FAINT, sw=T.SW_HAIR)

    for i, x in enumerate(XIN):
        if i == 2:                                  # ellipsis column
            for k in (-14, 0, 14):
                f.circle(x + k, Y_IN, 2.6, fill=T.INK, stroke="none", sw=0)
            continue
        f.circle(x, Y_IN, R, fill=T.PAPER, stroke=T.INK, sw=T.SW_BOX)
        f.text(x, Y_IN, [var("x"), sub("d" if i == 3 else str(i + 1))],
               size=T.FS_LABEL)
    f.circle(XOUT, Y_OUT, R, fill=T.GREEN.tint, stroke=T.GREEN.base,
             sw=T.SW_BOX)
    f.text(XOUT, Y_OUT, [var("o"), sub("1")], size=T.FS_LABEL)

    lx = XIN[0] - 175
    f.text(lx, Y_OUT, "OUTPUT LAYER", size=T.FS_TINY, color=T.MUTED,
           anchor="start", tracking=T.LETTERSPACE_CAPS)
    f.text(lx, Y_IN, "INPUT LAYER", size=T.FS_TINY, color=T.MUTED,
           anchor="start", tracking=T.LETTERSPACE_CAPS)

    f.save("singleneuron", out_dir=IMG,
           desc="Linear regression as a single-neuron network.")


def main():
    fig_fit_linreg()
    fig_singleneuron()
    print("wrote chapter-3 legacy remakes to", IMG)


if __name__ == "__main__":
    main()
