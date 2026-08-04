#!/usr/bin/env python3
"""Chapter 7 convolution-mechanics figures, in the approved mechanics family
(docs/convnet-rewrite/figure-style.md §scope: white grids, shaded cells, ink
strokes) drawn with arch_diagrams.MechDiagram — replacing the legacy Cairo
SVGs that had no generator.  Every numeric value is COMPUTED (real
cross-correlations), never typed in.

Writes img/{correlation,conv-pad,conv-stride,conv-multi-in,conv-1x1,
conv-reuse,pooling}.svg.  Byte-idempotent.

Run:  .venv-pytorch/bin/python tools/gen_arch_convmech_figures.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from arch_diagrams import (FAMILY_BLUE, GRID_FS, INK, SANS, MechDiagram,
                           save)
from figstyle import tokens as T

CELL = 26.0


def corr2d(X, K, stride=1):
    """Plain cross-correlation (the figures show its real outputs)."""
    h, w = K.shape
    H = (X.shape[0] - h) // stride + 1
    W = (X.shape[1] - w) // stride + 1
    Y = np.zeros((H, W), dtype=int)
    for i in range(H):
        for j in range(W):
            Y[i, j] = int((X[i * stride:i * stride + h,
                            j * stride:j * stride + w] * K).sum())
    return Y


def vals(A):
    return {(r, c): str(int(A[r, c])) for r in range(A.shape[0])
            for c in range(A.shape[1])}


def window(r0, c0, h=2, w=2):
    return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


X33 = np.arange(9).reshape(3, 3)
K22 = np.arange(4).reshape(2, 2)


# --------------------------------------------------------------------------- #

def fig_correlation():
    d = MechDiagram(560, 130)
    Y = corr2d(X33, K22)
    TY = 20 + 3 * CELL + 10
    d.grid(20, 20, 3, 3, CELL, shaded=window(0, 0), values=vals(X33),
           title="Input", title_y=TY)
    d.op(140, 59, "*")
    d.grid(180, 33, 2, 2, CELL, shaded=window(0, 0), values=vals(K22),
           title="Kernel", title_y=TY)
    d.op(270, 59, "=")
    d.grid(310, 33, 2, 2, CELL, shaded={(0, 0)}, values=vals(Y),
           title="Output", title_y=TY)
    save(d.fig, "correlation")


def fig_conv_pad():
    d = MechDiagram(640, 190)
    Xp = np.pad(X33, 1)
    Y = corr2d(Xp, K22)
    TY = 20 + 5 * CELL + 10
    d.grid(20, 20, 5, 5, CELL, dashed=True, shaded=window(0, 0),
           values=vals(Xp), title="Input (padded)", title_y=TY)
    d.grid(20 + CELL, 20 + CELL, 3, 3, CELL, frame_only=True)
    d.op(190, 85, "*")
    d.grid(230, 59, 2, 2, CELL, shaded=window(0, 0), values=vals(K22),
           title="Kernel", title_y=TY)
    d.op(320, 85, "=")
    d.grid(360, 33, 4, 4, CELL, shaded={(0, 0)}, values=vals(Y),
           title="Output", title_y=TY)
    save(d.fig, "conv-pad")


def fig_conv_stride():
    d = MechDiagram(600, 190)
    Xp = np.pad(X33, 1)
    K = K22
    # stride 3 vertically, 2 horizontally (the section's worked example)
    Y = np.array([[int((Xp[i:i + 2, j:j + 2] * K).sum())
                   for j in (0, 2)] for i in (0, 3)])
    shaded = window(0, 2) | window(3, 0)      # 2nd column step + 2nd row step
    TY = 20 + 5 * CELL + 10
    d.grid(20, 20, 5, 5, CELL, dashed=True, shaded=shaded, values=vals(Xp),
           title="Input (padded)", title_y=TY)
    d.grid(20 + CELL, 20 + CELL, 3, 3, CELL, frame_only=True)
    d.op(190, 85, "*")
    d.grid(230, 59, 2, 2, CELL, shaded=window(0, 0), values=vals(K),
           title="Kernel", title_y=TY)
    d.op(320, 85, "=")
    d.grid(360, 59, 2, 2, CELL, shaded={(0, 1), (1, 0)}, values=vals(Y),
           title="Output", title_y=TY)
    save(d.fig, "conv-stride")


def fig_conv_multi_in():
    d = MechDiagram(760, 200)
    X0, X1 = X33, X33 + 1
    K0, K1 = np.arange(4).reshape(2, 2), np.arange(1, 5).reshape(2, 2)
    Y = corr2d(X0, K0) + corr2d(X1, K1)   # = [[56, 72], [104, 120]]
    TY = 30 + 3 * CELL + 7 + 10
    d.grid_stack(20, 30, 2, 3, 3, CELL, shaded_front=window(0, 0),
                 title="Input", title_y=TY)
    d.op(140, 80, "*")
    d.grid_stack(170, 55, 2, 2, 2, CELL, shade_backs=True,
                 shaded_front=window(0, 0), title="Kernel", title_y=TY)
    d.op(255, 80, "=")
    # per-channel products, stacked vertically, then summed
    d.grid(290, 108, 3, 3, 22, shaded=window(0, 0), values=vals(X1))
    d.op(376, 141, "*")
    d.grid(392, 119, 2, 2, 22, shaded=window(0, 0), values=vals(K1))
    d.grid(290, 8, 3, 3, 22, shaded=window(0, 0), values=vals(X0))
    d.op(376, 41, "*")
    d.grid(392, 19, 2, 2, 22, shaded=window(0, 0), values=vals(K0))
    d.op(363, 96, "+")
    d.op(490, 80, "=")
    d.grid(520, 58, 2, 2, CELL, shaded={(0, 0)}, values=vals(Y),
           title="Output", title_y=TY)
    save(d.fig, "conv-multi-in")


def fig_conv_1x1():
    d = MechDiagram(620, 170)
    marks = {(0, 0), (2, 2)}
    d.grid_stack(20, 20, 3, 3, 3, CELL, shaded_front=marks, shade_backs=False,
                 title="Input")
    d.op(160, 70, "*")
    d.grid_stack(195, 55, 3, 1, 1, CELL, shade_backs=True,
                 shaded_front={(0, 0)}, title="Kernel")
    d.op(280, 70, "=")
    d.grid_stack(315, 30, 2, 3, 3, CELL, shaded_front=marks,
                 title="Output")
    save(d.fig, "conv-1x1")


def fig_conv_reuse():
    """How often each input pixel is used by a k x k kernel: the per-axis
    count ramps 1, 2, ..., k one cell at a time from the border, so a cell's
    use count is the product of its row and column counts (1 in corners,
    k^2 in the interior) — the original's banded layout, computed."""
    from matplotlib.patches import Rectangle

    d = MechDiagram(640, 180)
    S = 150.0                       # panel side
    CELLS = 6                       # underlying cell count per side
    for p, k in enumerate((1, 2, 3)):
        x0 = 20 + p * (S + 60)
        # per-axis bands: 1-cell steps 1..k-1, plateau k, mirrored
        edges = list(range(k)) + [CELLS - m for m in range(k - 1, -1, -1)]
        edges = sorted(set(e for e in edges if 0 <= e <= CELLS))
        counts = [min(a + 1, k, CELLS - a, CELLS - k + 1)
                  for a in edges[:-1]]
        for i, (ya, yb) in enumerate(zip(edges, edges[1:])):
            for j, (xa, xb) in enumerate(zip(edges, edges[1:])):
                n = counts[i] * counts[j]
                alpha = 0.08 + 0.52 * (n / (k * k))
                d.ax.add_patch(Rectangle(
                    (x0 + xa / CELLS * S, 20 + ya / CELLS * S),
                    (xb - xa) / CELLS * S, (yb - ya) / CELLS * S,
                    facecolor=T.BLUE.base, alpha=alpha, edgecolor=INK,
                    linewidth=1.2, zorder=4))
                d.ax.text(x0 + (xa + xb) / 2 / CELLS * S,
                          20 + (ya + yb) / 2 / CELLS * S, str(n),
                          fontsize=GRID_FS, color=INK, family=SANS,
                          ha="center", va="center", zorder=6)
        d.ax.text(x0 + S / 2, 20 + S + 12, f"{k}×{k} kernel",
                  fontsize=GRID_FS, color=INK, family=SANS, ha="center",
                  va="bottom", zorder=6)
    save(d.fig, "conv-reuse")


def fig_pooling():
    d = MechDiagram(560, 140)
    X = X33
    Y = np.array([[int(X[i:i + 2, j:j + 2].max()) for j in (0, 1)]
                  for i in (0, 1)])
    TY = 20 + 3 * CELL + 10
    d.grid(20, 20, 3, 3, CELL, shaded=window(0, 0), values=vals(X),
           title="Input", title_y=TY)
    # the op box, mechanics-family style
    from matplotlib.patches import FancyBboxPatch
    d.ax.add_patch(FancyBboxPatch((150, 45), 130, 44,
                                  boxstyle="round,pad=0.0,rounding_size=6",
                                  facecolor="white", edgecolor=INK,
                                  linewidth=1.8, zorder=5))
    d.ax.text(215, 67, "2×2 max-pooling", fontsize=GRID_FS, color=INK,
              family=SANS, ha="center", va="center", zorder=6)
    d.op(310, 67, "=")
    d.grid(350, 41, 2, 2, CELL, shaded={(0, 0)}, values=vals(Y),
           title="Output", title_y=TY)
    save(d.fig, "pooling")


def main():
    for fn in (fig_correlation, fig_conv_pad, fig_conv_stride,
               fig_conv_multi_in, fig_conv_1x1, fig_conv_reuse, fig_pooling):
        fn()
    print("wrote chapter-7 mechanics figures")


if __name__ == "__main__":
    main()
