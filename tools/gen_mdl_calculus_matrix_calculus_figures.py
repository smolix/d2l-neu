#!/usr/bin/env python3
"""Generate the new illustrative figures of the "Matrix Calculus" section
(``chapter_mdl-calculus/mdl-matrix-calculus.md``) in the shared house style.

Two figures:

* ``mdl-cal-relu-net-jacobian``: the 2-2-1 ReLU network of the section at
  x = (1, 1), with its inactive unit greyed, beside its Jacobian written as the
  product of three matrices.
* ``mdl-cal-jacobian-chain-order``: the two orders of multiplying
  u^T J3 J2 J1 (h = n = 1000), drawn with blocks of their true relative shape,
  with the cost of every step.

The chapter's other figures (including ``mdl-cal-jacobian-ellipse``) live in
``gen_mdl_calculus_figures.py``.  The shared style lives in
``gen_mdl_figures.py``; importing it applies the rcParams, and its ``save()``
keeps the output byte-identical across runs.

    .venv-pytorch/bin/python tools/gen_mdl_calculus_matrix_calculus_figures.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

from figstyle import tokens as T
from matplotlib.patches import FancyBboxPatch, Rectangle

np, plt = fl.np, fl.plt
INK = fl.INK
FS = 13          # one size for every in-figure label (the chapter's size)
FS_CELL = 14     # numbers inside matrix cells and node boxes


# --------------------------------------------------------------------------- #
# The 2-2-1 ReLU network and its Jacobian as a product of three matrices      #
# --------------------------------------------------------------------------- #

def _matrix(ax, x0, ytop, entries, shade=(), cell=0.5):
    """Draw a matrix of numbers with its top-left corner at (x0, ytop).
    ``shade`` lists the (row, col) cells to grey out.  Returns the right x."""
    rows, cols = len(entries), len(entries[0])
    for i in range(rows):
        for j in range(cols):
            grey = (i, j) in shade
            ax.add_patch(Rectangle((x0 + j * cell, ytop - (i + 1) * cell),
                                   cell, cell, lw=0.9, zorder=2,
                                   facecolor=T.HAIRLINE if grey else T.PAPER,
                                   edgecolor=INK))
            ax.text(x0 + (j + 0.5) * cell, ytop - (i + 0.5) * cell,
                    f"${entries[i][j]}$", ha="center", va="center",
                    fontsize=FS_CELL, zorder=3,
                    color=T.MUTED if grey else INK)
    return x0 + cols * cell


def fig_relu_net_jacobian():
    """The 2-2-1 ReLU network at x = (1, 1).  The second hidden unit has
    pre-activation 3*1 - 1*1 - 4 = -2, so it is off (grey, dashed); its mask
    entry is 0 and the second row of W1 never reaches the output.  Right: the
    Jacobian J_g = w2^T diag(m) W1 = (2, 4) with the cells the mask removes
    shaded."""
    fig, ax = plt.subplots(figsize=(10.6, 3.0))
    bw, bh = 1.3, 0.5                         # node box width and height
    pos = {"x1": (0.0, 1.7), "x2": (0.0, 0.0),
           "h1": (3.1, 1.7), "h2": (3.1, 0.0), "g": (5.9, 0.85)}
    text = {"x1": r"$x_1=1$", "x2": r"$x_2=1$", "h1": r"$a_1=3$",
            "h2": r"$a_2=-2$", "g": r"$g=6$"}

    def box(k, dead=False):
        cx, cy = pos[k]
        ax.add_patch(FancyBboxPatch(
            (cx - bw / 2, cy - bh / 2), bw, bh,
            boxstyle="round,pad=0.02,rounding_size=0.12", zorder=3, lw=1.2,
            facecolor=T.PANEL if dead else T.BLUE.tint,
            edgecolor=T.FAINT if dead else INK,
            linestyle=(0, (4, 2.5)) if dead else "-"))
        ax.text(cx, cy, text[k], ha="center", va="center", fontsize=FS_CELL,
                color=T.MUTED if dead else INK, zorder=4)

    def link(a, b, w, frac, side, dead=False, off=0.3):
        (ax0, ay0), (bx0, by0) = pos[a], pos[b]
        p0 = np.array([ax0 + bw / 2, ay0])
        p1 = np.array([bx0 - bw / 2, by0])
        fl.arrow(ax, p0, p1, color=T.FAINT if dead else fl.GRAY, lw=1.3, mut=11)
        q = p0 + frac * (p1 - p0)              # where the weight label sits
        d = (p1 - p0) / np.linalg.norm(p1 - p0)
        nrm = np.array([-d[1], d[0]]) * side   # offset off the line
        ax.text(*(q + off * nrm), f"${w}$", ha="center", va="center",
                fontsize=FS, color=T.MUTED if dead else INK, zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.4))

    link("x1", "h1", "1", 0.5, +1)
    # crossing edges: label each near its target, on the outer side
    link("x2", "h1", "2", 0.8, -1, off=0.22)
    link("x1", "h2", "3", 0.8, +1, off=0.22)
    link("x2", "h2", "-1", 0.5, -1)
    link("h1", "g", "2", 0.5, +1)
    link("h2", "g", "-1", 0.5, -1, dead=True)
    for k in pos:
        box(k, dead=(k == "h2"))
    cx, cy = pos["h2"]
    ax.text(cx, cy - bh / 2 - 0.12, r"bias $-4$: off", ha="center", va="top",
            fontsize=FS, color=INK)
    cx, cy = pos["h1"]
    ax.text(cx, cy + bh / 2 + 0.12, r"bias $0$: on", ha="center",
            va="bottom", fontsize=FS, color=INK)

    # --- the product w2^T diag(m) W1 = (2, 4), drawn as number grids ---
    cell, ytop = 0.5, 1.35
    x = 7.6
    lab_y = ytop - 2 * cell - 0.18
    xr = _matrix(ax, x, ytop - 0.5 * cell, [["2", "-1"]], shade={(0, 1)})
    ax.text(0.5 * (x + xr), lab_y, r"$\mathbf{w}_2^\top$", ha="center",
            va="top", fontsize=FS)
    x = xr + 0.25
    xr = _matrix(ax, x, ytop, [["1", "0"], ["0", "0"]], shade={(1, 1)})
    ax.text(0.5 * (x + xr), lab_y, r"$\mathrm{diag}(\mathbf{m})$",
            ha="center", va="top", fontsize=FS)
    x = xr + 0.25
    xr = _matrix(ax, x, ytop, [["1", "2"], ["3", "-1"]],
                 shade={(1, 0), (1, 1)})
    ax.text(0.5 * (x + xr), lab_y, r"$\mathbf{W}_1$", ha="center", va="top",
            fontsize=FS)
    ax.text(xr + 0.3, ytop - cell, "$=$", ha="center", va="center",
            fontsize=FS_CELL + 2)
    x = xr + 0.6
    xr = _matrix(ax, x, ytop - 0.5 * cell, [["2", "4"]])
    ax.text(0.5 * (x + xr), lab_y, r"$\mathbf{J}_g$", ha="center", va="top",
            fontsize=FS)

    ax.set_xlim(-0.75, xr + 0.1)
    ax.set_ylim(-0.75, 2.35)
    ax.set_aspect("equal")
    ax.axis("off")
    fl.save(fig, "mdl-cal-relu-net-jacobian")


# --------------------------------------------------------------------------- #
# Two orders of multiplying u^T J3 J2 J1                                      #
# --------------------------------------------------------------------------- #

S = 0.95          # side of an h x h block
BAR = S / 9       # height of a 1 x h row (drawn thicker than true 1/1000)


def _block(ax, x, yc, kind, fill, edge=INK):
    """Draw a 'bar' (1 x h row) or a 'square' (h x h) with its left edge at x,
    centred vertically on yc.  Returns the right x."""
    h = BAR if kind == "bar" else S
    ax.add_patch(Rectangle((x, yc - h / 2), S, h, facecolor=fill,
                           edgecolor=edge, lw=1.1, zorder=3))
    return x + S


def _step(ax, x, yc, left, right, result, names, cost, out_fill, out_edge):
    """One multiplication: left . right = result, with names under the two
    operands and the cost under the group.  Returns the right x."""
    xa = x
    x = _block(ax, x, yc, left[0], left[1], left[2])
    ax.text(x + 0.13, yc, r"$\cdot$", ha="center", va="center",
            fontsize=FS_CELL + 4)
    x += 0.26
    xb = x
    x = _block(ax, x, yc, right[0], right[1], right[2])
    ax.text(x + 0.18, yc, "$=$", ha="center", va="center", fontsize=FS_CELL)
    x += 0.36
    xc = x
    x = _block(ax, x, yc, result, out_fill, out_edge)
    y_name = yc - S / 2 - 0.08
    for xx, nm in zip((xa, xb), names):
        ax.text(xx + S / 2, y_name, nm, ha="center", va="top", fontsize=FS)
    ax.text(0.5 * (xa + x), y_name - 0.42, cost, ha="center", va="top",
            fontsize=FS, color=INK)
    return x


def fig_jacobian_chain_order():
    """u^T J3 J2 J1 with h = n = 1000, multiplied from the left (top row:
    every intermediate is a 1 x h row, three matrix-vector products of 10^6
    each) and from the right (bottom row: two matrix-matrix products of 10^9
    each, then one matrix-vector product)."""
    fig, ax = plt.subplots(figsize=(10.2, 4.6))
    fac = T.PANEL                                   # the given factors
    org = (T.ORANGE.tint, T.ORANGE.base)            # row intermediates
    blu = (T.BLUE.tint, T.BLUE.base)                # square intermediates
    gap = 0.55
    ya, yb = 2.75, 0.0                              # row centres

    ax.text(-0.1, ya + S / 2 + 0.25,
            r"from the left: $3\times10^6$ multiplications",
            ha="left", va="bottom", fontsize=FS, color=INK)
    x = 0.0
    x = _step(ax, x, ya, ("bar", fac, INK), ("square", fac, INK), "bar",
              (r"$\mathbf{u}^\top$", r"$\mathbf{J}_3$"), r"$h^2=10^6$", *org)
    x = _step(ax, x + gap, ya, ("bar", *org), ("square", fac, INK), "bar",
              ("", r"$\mathbf{J}_2$"), r"$h^2=10^6$", *org)
    x = _step(ax, x + gap, ya, ("bar", *org), ("square", fac, INK), "bar",
              ("", r"$\mathbf{J}_1$"), r"$hn=10^6$", *org)

    ax.text(-0.1, yb + S / 2 + 0.25,
            r"from the right: $2.001\times10^9$ multiplications",
            ha="left", va="bottom", fontsize=FS, color=INK)
    x = 0.0
    x = _step(ax, x, yb, ("square", fac, INK), ("square", fac, INK), "square",
              (r"$\mathbf{J}_2$", r"$\mathbf{J}_1$"), r"$h^2n=10^9$", *blu)
    x = _step(ax, x + gap, yb, ("square", fac, INK), ("square", *blu),
              "square", (r"$\mathbf{J}_3$", ""), r"$h^2n=10^9$", *blu)
    x = _step(ax, x + gap, yb, ("bar", fac, INK), ("square", *blu), "bar",
              (r"$\mathbf{u}^\top$", ""), r"$hn=10^6$", *org)

    ax.text(x, yb - S / 2 - 0.95, r"$h=n=1000$", ha="right", va="top",
            fontsize=FS, color=INK)
    ax.set_xlim(-0.15, x + 0.05)
    ax.set_ylim(yb - S / 2 - 1.3, ya + S / 2 + 0.7)
    ax.set_aspect("equal")
    ax.axis("off")
    fl.save(fig, "mdl-cal-jacobian-chain-order")


FIGURES = [fig_relu_net_jacobian, fig_jacobian_chain_order]


def main():
    start = len(fl.WRITTEN)
    for fn in FIGURES:
        fn()
    for p in fl.WRITTEN[start:]:
        assert os.path.getsize(p) > 0, f"empty: {p}"
        print(f"  {os.path.basename(p):36s} {os.path.getsize(p):>8,d} bytes")


if __name__ == "__main__":
    main()
