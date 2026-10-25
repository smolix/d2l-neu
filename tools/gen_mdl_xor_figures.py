#!/usr/bin/env python3
"""Generate the illustrative figures for the "Multilayer Perceptrons"
chapter (``chapter_multilayer-perceptrons``) in the one shared house style
defined in ``gen_mdl_figures.py``.

The prose references the generated files with no drawing code (like the slide
SVGs).  Figures that show a computed result use real numerical computation, so
the pictures are exact, not sketched: the XOR feature map below is the actual
forward pass of the hand-built two-ReLU network, not a hand-placed cartoon.

Run with the repo's pytorch venv (matplotlib + numpy are available):

    .venv-pytorch/bin/python tools/gen_mdl_xor_figures.py

All figures are written to ``img/mdl-mlp-<id>.svg``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT


def fig_xor():
    """XOR is not linearly separable, but a hidden layer of two ReLU units
    makes it so.

    Both panels show the hand-built network of mlp.md,
        h = ReLU(x W1 + b1),  W1 = [[1, 1], [1, 1]],  b1 = (0, -1),
        o = h1 - 2 h2,
    whose output equals the XOR label on the four corners of the unit
    square (asserted below).

    Left, input space: the corners labeled by XOR.  The network predicts
    class 1 where o > 1/2.  Since o depends on x only through s = x1 + x2
    (it is the tent ReLU(s) - 2 ReLU(s - 1)), this region is the band
    1/2 < x1 + x2 < 3/2 between two parallel lines: a boundary that no single
    line can draw.

    Right, hidden space: the same corners after the hidden layer.  The two
    class-1 corners land on the same point (1, 0), and the single line
    h1 - 2 h2 = 1/2 separates the classes.  Both panels share one scale
    (equal aspect, width ratios = x spans), and both use the same colors:
    blue = class 0, orange = class 1, green = the decision boundary.
    """
    T = fl.T
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1, 1, 0])                     # XOR labels
    W1 = np.array([[1.0, 1.0], [1.0, 1.0]])
    b1 = np.array([0.0, -1.0])
    w2 = np.array([1.0, -2.0])
    H = np.maximum(X @ W1 + b1, 0.0)
    o = H @ w2
    assert np.array_equal(o, y), o                 # the network IS XOR
    print("  hidden H =", H.tolist(), " output o =", o.tolist())

    fill = {0: T.BLUE.base, 1: T.ORANGE.base}
    tint = {0: T.BLUE.tint, 1: T.ORANGE.tint}
    line_kw = dict(color=T.GREEN.base, lw=T.SW_LINE, ls=(0, (4, 3)), zorder=3)
    lx, ly = (-0.5, 1.5), (-0.5, 1.5)              # input-space window
    hx, hy = (-0.5, 2.5), (-0.5, 1.5)              # hidden-space window
    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(7.0, 2.9),
        gridspec_kw=dict(width_ratios=[lx[1] - lx[0], hx[1] - hx[0]],
                         wspace=0.28))

    def corner(ax, p, lab, text=None, dx=0.0, dy=0.0, **kw):
        ax.plot(*p, "o", color=fill[lab], ms=16, zorder=5)
        ax.text(*p, str(lab), color="white", fontsize=13, fontweight="bold",
                ha="center", va="center", zorder=6)
        if text:
            ax.text(p[0] + dx, p[1] + dy, text, color=T.INK, fontsize=15,
                    zorder=6, **kw)

    # ---- left: input space, decision band of the network ----------------
    xs = np.array(lx)
    axa.fill_between(xs, ly[0], ly[1], color=tint[0], lw=0, zorder=0)
    axa.fill_between(xs, 0.5 - xs, 1.5 - xs, color=tint[1], lw=0, zorder=1)
    axa.plot(xs, 0.5 - xs, **line_kw)
    axa.plot(xs, 1.5 - xs, **line_kw)
    for p, lab in zip(X, y):
        corner(axa, p, lab)
    axa.set_xlim(*lx); axa.set_ylim(*ly)
    axa.set_xticks([0, 1]); axa.set_yticks([0, 1])
    axa.set_xlabel("$x_1$"); axa.set_ylabel("$x_2$", rotation=0,
                                            labelpad=12)

    # ---- right: hidden space, one separating line -----------------------
    hs = np.array(hx)
    line = (hs - 0.5) / 2.0                        # h1 - 2 h2 = 1/2
    axb.fill_between(hs, line, hy[1], color=tint[0], lw=0, zorder=0)
    axb.fill_between(hs, hy[0], line, color=tint[1], lw=0, zorder=0)
    axb.plot(hs, line, **line_kw)
    corner(axb, H[0], 0, "(0, 0)", dx=0.0, dy=0.24, ha="center",
           va="bottom")
    corner(axb, H[1], 1, "(0, 1), (1, 0)", dx=0.0, dy=-0.24, ha="center",
           va="top")
    corner(axb, H[3], 0, "(1, 1)", dx=0.0, dy=0.24, ha="center",
           va="bottom")
    axb.set_xlim(*hx); axb.set_ylim(*hy)
    axb.set_xticks([0, 1, 2]); axb.set_yticks([0, 1])
    axb.set_xlabel("$h_1$"); axb.set_ylabel("$h_2$", rotation=0,
                                            labelpad=12)

    for ax in (axa, axb):
        ax.set_aspect("equal")
        ax.spines["left"].set_color(T.INK)
        ax.spines["bottom"].set_color(T.INK)
        ax.tick_params(colors=T.INK, labelsize=14)
    fl.save(fig, "mdl-mlp-xor")


FIGURES = [fig_xor]


def main():
    # Verify only the figures THIS script writes (the shared module's WRITTEN
    # list also tracks the Linear Algebra figures, which we don't run here).
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
