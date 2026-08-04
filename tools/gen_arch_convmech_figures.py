#!/usr/bin/env python3
"""Chapter 7 convolution-mechanics figures, in the approved mechanics family
(docs/convnet-rewrite/figure-style.md §scope: white grids, shaded cells, ink
strokes) drawn with arch_diagrams.MechDiagram — replacing the legacy Cairo
SVGs that had no generator.  Every numeric value is COMPUTED (real
cross-correlations), never typed in.  Panel spacing follows the ORIGINALS'
tight proportions (ops get ~GAP*2 of air, not a gulf — Alex, ch7 review).

Writes img/{correlation,conv-pad,conv-stride,conv-multi-in,conv-1x1,
conv-reuse,pooling}.svg.  Byte-idempotent.

Run:  .venv-pytorch/bin/python tools/gen_arch_convmech_figures.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from arch_diagrams import (FAMILY_BLUE, FAMILY_BLUE_DARK, GRID_FS, INK, SANS,
                           MechDiagram, save)
from figstyle import tokens as T

CELL = 26.0
OP = 19.0        # air on each side of a * / = / + operator


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
    d = MechDiagram(330, 130)
    Y = corr2d(X33, K22)
    TY = 20 + 3 * CELL + 10
    x = 20
    w, _ = d.grid(x, 20, 3, 3, CELL, shaded=window(0, 0), values=vals(X33),
                  title="Input", title_y=TY)
    x += w + OP
    d.op(x, 59, "*")
    x += OP
    w, _ = d.grid(x, 33, 2, 2, CELL, shaded=window(0, 0), values=vals(K22),
                  title="Kernel", title_y=TY)
    x += w + OP
    d.op(x, 59, "=")
    x += OP
    d.grid(x, 33, 2, 2, CELL, shaded={(0, 0)}, values=vals(Y),
           title="Output", title_y=TY)
    save(d.fig, "correlation")


def fig_conv_pad():
    d = MechDiagram(440, 190)
    Xp = np.pad(X33, 1)
    Y = corr2d(Xp, K22)
    TY = 20 + 5 * CELL + 10
    x = 20
    w, _ = d.grid(x, 20, 5, 5, CELL, dashed=True, shaded=window(0, 0),
                  values=vals(Xp), title="Input (padded)", title_y=TY)
    d.grid(x + CELL, 20 + CELL, 3, 3, CELL, frame_only=True)
    x += w + OP
    d.op(x, 85, "*")
    x += OP
    w, _ = d.grid(x, 59, 2, 2, CELL, shaded=window(0, 0), values=vals(K22),
                  title="Kernel", title_y=TY)
    x += w + OP
    d.op(x, 85, "=")
    x += OP
    d.grid(x, 33, 4, 4, CELL, shaded={(0, 0)}, values=vals(Y),
           title="Output", title_y=TY)
    save(d.fig, "conv-pad")


def fig_conv_stride():
    d = MechDiagram(400, 190)
    Xp = np.pad(X33, 1)
    K = K22
    # stride 3 vertically, 2 horizontally (the section's worked example)
    Y = np.array([[int((Xp[i:i + 2, j:j + 2] * K).sum())
                   for j in (0, 2)] for i in (0, 3)])
    shaded = window(0, 2) | window(3, 0)      # 2nd column step + 2nd row step
    TY = 20 + 5 * CELL + 10
    x = 20
    w, _ = d.grid(x, 20, 5, 5, CELL, dashed=True, shaded=shaded,
                  values=vals(Xp), title="Input (padded)", title_y=TY)
    d.grid(x + CELL, 20 + CELL, 3, 3, CELL, frame_only=True)
    x += w + OP
    d.op(x, 85, "*")
    x += OP
    w, _ = d.grid(x, 59, 2, 2, CELL, shaded=window(0, 0), values=vals(K),
                  title="Kernel", title_y=TY)
    x += w + OP
    d.op(x, 85, "=")
    x += OP
    d.grid(x, 59, 2, 2, CELL, shaded={(0, 1), (1, 0)}, values=vals(Y),
           title="Output", title_y=TY)
    save(d.fig, "conv-stride")


def fig_conv_multi_in():
    d = MechDiagram(560, 200)
    X0, X1 = X33, X33 + 1
    K0, K1 = np.arange(4).reshape(2, 2), np.arange(1, 5).reshape(2, 2)
    Y = corr2d(X0, K0) + corr2d(X1, K1)   # = [[56, 72], [104, 120]]

    # left: the stacked view — values and window shading on BOTH sheets
    TY = 30 + 3 * CELL + 8 + 10
    OFF = 8.0
    x = 20
    d.grid(x + OFF, 30 + OFF, 3, 3, CELL, shaded=window(0, 0),
           values=vals(X1), zorder=2)
    d.grid(x, 30, 3, 3, CELL, shaded=window(0, 0), values=vals(X0))
    d.ax.text(x + (3 * CELL + OFF) / 2, TY, "Input", fontsize=GRID_FS,
              color=INK, family=SANS, ha="center", va="bottom", zorder=6)
    x += 3 * CELL + OFF + OP
    d.op(x, 80, "*")
    x += OP
    d.grid(x + OFF, 55 + OFF, 2, 2, CELL, shaded=window(0, 0),
           values=vals(K1), zorder=2)
    d.grid(x, 55, 2, 2, CELL, shaded=window(0, 0), values=vals(K0))
    d.ax.text(x + (2 * CELL + OFF) / 2, TY, "Kernel", fontsize=GRID_FS,
              color=INK, family=SANS, ha="center", va="bottom", zorder=6)
    x += 2 * CELL + OFF + OP
    d.op(x, 80, "=")
    x += OP

    # right: the two per-channel products, '+' between the KERNELS
    c = 21.0
    d.grid(x, 104, 3, 3, c, shaded=window(0, 0), values=vals(X1))
    d.op(x + 3 * c + 13, 135, "*", fs=14)
    kx = x + 3 * c + 26
    d.grid(kx, 114, 2, 2, c, shaded=window(0, 0), values=vals(K1))
    d.grid(x, 8, 3, 3, c, shaded=window(0, 0), values=vals(X0))
    d.op(x + 3 * c + 13, 39, "*", fs=14)
    d.grid(kx, 18, 2, 2, c, shaded=window(0, 0), values=vals(K0))
    d.op(kx + c, 88, "+")                     # between the two kernels
    x = kx + 2 * c + OP
    d.op(x, 80, "=")
    x += OP
    d.grid(x, 58, 2, 2, CELL, shaded={(0, 0)}, values=vals(Y),
           title="Output", title_y=TY)
    save(d.fig, "conv-multi-in")


def fig_conv_1x1():
    """1x1 convolution mixes channels: an output cell reads the SAME spatial
    position across every input channel.  Two traced positions — (0,0) in
    the light shade, (2,2) in the dark — pierce the full input depth; two
    kernel groups (one per output channel) map them to the two output
    sheets: (0,0) lands on the front sheet, (2,2) on the back one."""
    d = MechDiagram(430, 190)
    OFF = 8.0
    marks_l, marks_d = {(0, 0)}, {(2, 2)}

    # input: 3 channel sheets, both traced positions shaded on EVERY sheet
    x, y = 20, 30
    for i in (2, 1, 0):
        d.grid(x + i * OFF, y + i * OFF, 3, 3, CELL, shaded=marks_l,
               shaded_dark=marks_d, zorder=4 - i)
    tw = 3 * CELL + 2 * OFF
    TY = y + tw + 10
    d.ax.text(x + tw / 2, TY, "Input", fontsize=GRID_FS, color=INK,
              family=SANS, ha="center", va="bottom", zorder=6)
    x += tw + OP
    d.op(x, 85, "*")
    x += OP

    # kernel: TWO groups of three 1x1 weights (one group per output channel)
    for i in (2, 1, 0):        # output-channel 1 group (light)
        d.grid(x + i * OFF, 78 + i * OFF, 1, 1, CELL, shaded={(0, 0)},
               zorder=4 - i)
    for i in (2, 1, 0):        # output-channel 2 group (dark)
        d.grid(x + i * OFF, 26 + i * OFF, 1, 1, CELL,
               shaded_dark={(0, 0)}, zorder=4 - i)
    kw = CELL + 2 * OFF
    d.ax.text(x + kw / 2, TY, "Kernel", fontsize=GRID_FS, color=INK,
              family=SANS, ha="center", va="bottom", zorder=6)
    x += kw + OP
    d.op(x, 85, "=")
    x += OP

    # output: 2 channel sheets — (0,0) light on the FRONT sheet, (2,2) dark
    # on the BACK sheet
    d.grid(x + OFF, y + OFF, 3, 3, CELL, shaded_dark=marks_d, zorder=3)
    d.grid(x, y, 3, 3, CELL, shaded=marks_l, zorder=4)
    ow = 3 * CELL + OFF
    d.ax.text(x + ow / 2, TY, "Output", fontsize=GRID_FS, color=INK,
              family=SANS, ha="center", va="bottom", zorder=6)
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
    d = MechDiagram(400, 140)
    X = X33
    Y = np.array([[int(X[i:i + 2, j:j + 2].max()) for j in (0, 1)]
                  for i in (0, 1)])
    TY = 20 + 3 * CELL + 10
    x = 20
    w, _ = d.grid(x, 20, 3, 3, CELL, shaded=window(0, 0), values=vals(X),
                  title="Input", title_y=TY)
    x += w + OP
    # the op box, mechanics-family style
    from matplotlib.patches import FancyBboxPatch
    bw = 128
    d.ax.add_patch(FancyBboxPatch((x, 45), bw, 44,
                                  boxstyle="round,pad=0.0,rounding_size=6",
                                  facecolor="white", edgecolor=INK,
                                  linewidth=1.8, zorder=5))
    d.ax.text(x + bw / 2, 67, "2×2 max-pooling", fontsize=GRID_FS, color=INK,
              family=SANS, ha="center", va="center", zorder=6)
    x += bw + OP
    d.op(x, 67, "=")
    x += OP
    d.grid(x, 41, 2, 2, CELL, shaded={(0, 0)}, values=vals(Y),
           title="Output", title_y=TY)
    save(d.fig, "pooling")


def main():
    for fn in (fig_correlation, fig_conv_pad, fig_conv_stride,
               fig_conv_multi_in, fig_conv_1x1, fig_conv_reuse, fig_pooling):
        fn()
    print("wrote chapter-7 mechanics figures")


if __name__ == "__main__":
    main()
