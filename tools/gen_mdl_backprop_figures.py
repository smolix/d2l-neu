#!/usr/bin/env python3
"""Generate the illustrative figure(s) for the backpropagation section
(``chapter_multilayer-perceptrons/backprop.md``) in the one shared house style
defined in ``gen_mdl_figures.py``.

The section's worked example pushes concrete numbers through a tiny 2-2-1 ReLU
network.  This figure draws that example as a computational graph, with the
forward pass in black and the backward pass overlaid in blue, so the "full"
backprop treatment carries a figure at least as informative as the autograd
preview in :numref:`sec_autograd`.  The numbers are *computed* here (not typed
by hand) so the picture is guaranteed to agree with the prose.

Run with the repo's pytorch venv (matplotlib + numpy are available):

    .venv-pytorch/bin/python tools/gen_mdl_backprop_figures.py

All figures are written to ``img/mdl-mlp-<id>.svg``.  The script is idempotent:
re-running overwrites byte-for-byte.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT

from matplotlib.patches import FancyBboxPatch

# Black for the forward pass (matches autograd-comp-graph.svg's forward arrows);
# the shared BLUE carries the backward pass.
FWD = "#15181C"


def _num(x):
    x = float(x) + 0.0          # normalize -0.0 -> 0.0
    if abs(x) < 1e-9:
        x = 0.0
    return f"{x:.0f}" if abs(x - round(x)) < 1e-9 else f"{x:g}"


def _fmt_vec(v):
    """Compact bracketed row, integers shown without a trailing .0."""
    return "[" + ",\\ ".join(_num(x) for x in np.ravel(v)) + "]"


def _matrix2x2(ax, cx, cy, M, color, fontsize=12):
    """Draw a 2x2 matrix of numbers flanked by drawn square brackets, centred at
    (cx, cy).  mathtext has no matrix environment, so we lay it out by hand; this
    keeps the figure self-contained and crisp."""
    M = np.asarray(M, float)
    col_dx, row_dy = 0.34, 0.30
    cols = [cx - col_dx, cx + col_dx]
    rows = [cy + row_dy, cy - row_dy]
    for i in range(2):
        for j in range(2):
            ax.text(cols[j], rows[i], _num(M[i, j]), ha="center", va="center",
                    fontsize=fontsize, color=color)
    bx, by = col_dx + 0.30, row_dy + 0.18      # bracket half-extent
    tick = 0.12
    for sx, x_in in [(-1, +1), (+1, -1)]:       # left then right bracket
        bxx = cx + sx * bx
        ax.plot([bxx, bxx], [cy - by, cy + by], color=color, lw=1.3)
        ax.plot([bxx, bxx + x_in * tick], [cy + by, cy + by], color=color, lw=1.3)
        ax.plot([bxx, bxx + x_in * tick], [cy - by, cy - by], color=color, lw=1.3)


def fig_backprop_graph():
    r"""The section's worked example as a computational graph.

    Network: x -> z = W1 x -> h = relu(z) -> o = W2 h -> L = 1/2 (o-y)^2,
    with lambda = 0 (no regularization, to keep the picture about the chain
    rule).  Forward values flow left-to-right in black; backpropagation flows
    right-to-left in blue, each return arrow labelled with the gradient that
    arrives there, and the two parameter gradients drop off below their nodes.
    Every number is recomputed here so the figure cannot drift from the text."""
    # --- recompute the worked example exactly (single source of truth) ---
    x = np.array([1.0, 2.0])
    W1 = np.array([[1.0, -1.0], [0.0, 1.0]])
    W2 = np.array([[2.0, -1.0]])
    y = 0.0

    z = W1 @ x                       # [-1, 2]
    h = np.maximum(0.0, z)           # [0, 2]   (first unit dead)
    o = float((W2 @ h)[0])           # -2
    L = 0.5 * (o - y) ** 2           # 2

    dLdo = o - y                     # -2
    dLdW2 = dLdo * h                 # [0, -4]
    dLdh = (W2.T * dLdo).ravel()     # [-4, 2]
    relu_grad = (z > 0).astype(float)
    dLdz = dLdh * relu_grad          # [0, 2]
    dLdW1 = np.outer(dLdz, x)        # [[0,0],[2,4]]

    # sanity: the values the prose asserts
    assert L == 2.0
    assert np.allclose(dLdW2, [0.0, -4.0])
    assert np.allclose(dLdz, [0.0, 2.0])
    assert np.allclose(dLdW1, [[0.0, 0.0], [2.0, 4.0]])

    fig, ax = plt.subplots(figsize=(11.2, 4.2))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.axis("off")

    yc = 3.4                        # node-row centre
    bw, bh = 2.05, 1.05             # node box size
    centres = [1.55, 5.0, 8.0, 11.0, 14.3]   # x of x, z, h, o, L

    def node(cx, name, value, color):
        x0, y0 = cx - bw / 2, yc - bh / 2
        for fc, a in [(color, 0.12), ("none", 1.0)]:
            ax.add_patch(FancyBboxPatch(
                (x0, y0), bw, bh,
                boxstyle="round,pad=0.02,rounding_size=0.12",
                linewidth=1.8, edgecolor=color, facecolor=fc, alpha=a))
        ax.text(cx, yc + 0.20, name, ha="center", va="center",
                fontsize=13, color=color)
        ax.text(cx, yc - 0.27, value, ha="center", va="center",
                fontsize=12, color=GRAY, usetex=False)
        return dict(l=x0, r=x0 + bw, cx=cx)

    nx = node(centres[0], r"$\mathbf{x}$", f"${_fmt_vec(x)}$", GREEN)
    nz = node(centres[1], r"$\mathbf{z}$", f"${_fmt_vec(z)}$", BLUE)
    nh = node(centres[2], r"$\mathbf{h}$", f"${_fmt_vec(h)}$", BLUE)
    no = node(centres[3], r"$o$", f"${o:.0f}$", BLUE)
    nL = node(centres[4], r"$L$", f"${L:.0f}$", ORANGE)
    nodes = [nx, nz, nh, no, nL]

    # --- forward arrows (black, above the row), labelled with the operation ---
    fwd_ops = [r"$\times\,\mathbf{W}^{(1)}$", r"$\phi=\mathrm{ReLU}$",
               r"$\times\,\mathbf{W}^{(2)}$", r"$\frac{1}{2}(o-y)^2$"]
    yf = yc + bh / 2 + 0.05
    for (a, b), op in zip(zip(nodes[:-1], nodes[1:]), fwd_ops):
        x0, x1 = a["r"] + 0.12, b["l"] - 0.12
        fl.arrow(ax, (x0, yf), (x1, yf), color=FWD, lw=2.0, mut=13)
        ax.text((x0 + x1) / 2, yf + 0.34, op, ha="center", va="bottom",
                fontsize=11.5, color=FWD)

    # --- backward arrows (blue, below the row), labelled with the gradient ---
    back_lbls = [
        r"$\frac{\partial L}{\partial \mathbf{z}}=" + _fmt_vec(dLdz) + "$",
        r"$\frac{\partial L}{\partial \mathbf{h}}=" + _fmt_vec(dLdh) + "$",
        r"$\frac{\partial L}{\partial o}=" + f"{dLdo:.0f}" + "$",
        r"seed $1$",
    ]
    yb = yc - bh / 2 - 0.05
    for (a, b), lbl in zip(zip(nodes[:-1], nodes[1:]), back_lbls):
        x0, x1 = b["l"] - 0.12, a["r"] + 0.12
        fl.arrow(ax, (x0, yb), (x1, yb), color=BLUE, lw=2.0, mut=13)
        ax.text((x0 + x1) / 2, yb - 0.30, lbl, ha="center", va="top",
                fontsize=11.5, color=BLUE)

    # --- parameter gradients (the payoff) drop off below their layers, blue ---
    y_drop_tip = yc - bh / 2 - 1.30
    y_lbl = y_drop_tip - 0.30
    for n in (nz, no):
        ax.annotate("", xy=(n["cx"], y_drop_tip), xytext=(n["cx"], yb - 0.05),
                    arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.6,
                                    shrinkA=0, shrinkB=0, mutation_scale=12))

    # dL/dW1 is a 2x2 matrix: label on the left, hand-laid matrix on the right
    ax.text(nz["cx"] - 0.30, y_lbl - 0.30,
            r"$\frac{\partial L}{\partial \mathbf{W}^{(1)}}=$",
            ha="right", va="center", fontsize=13, color=BLUE)
    _matrix2x2(ax, nz["cx"] + 0.62, y_lbl - 0.30, dLdW1, BLUE, fontsize=12)

    # dL/dW2 is a row vector
    ax.text(no["cx"], y_lbl,
            r"$\frac{\partial L}{\partial \mathbf{W}^{(2)}}=" + _fmt_vec(dLdW2) + "$",
            ha="center", va="top", fontsize=13, color=BLUE)

    # --- corner direction labels, echoing autograd-comp-graph.svg ---
    ax.text(centres[0] - bw / 2, yc + 1.55, r"forward $\rightarrow$",
            ha="left", va="center", fontsize=11, color=FWD, fontweight="bold")
    ax.text(centres[-1] + bw / 2, yc - 1.55, r"$\leftarrow$ backward",
            ha="right", va="center", fontsize=11, color=BLUE, fontweight="bold")

    fl.save(fig, "mdl-mlp-backprop-graph")


def fig_mdl_mlp_forward_graph():
    r"""Computational graph of the *forward* pass, in the same house style as
    :func:`fig_backprop_graph` so the two read as a matched pair.

    This replaces the legacy hand-drawn ``forward.svg`` (black/white
    glyph-outline squares-and-circles).  It draws the full forward computation
    the section describes, including the weight-decay branch:

        x --(x W1)--> z --(phi)--> h --(x W2)--> o --(loss)--> L
        W1, W2 --(reg)--> s,   then   J = L + s.

    Everything flows left-to-right and bottom-to-top, in black (``FWD``): this is
    the forward pass *only*, with no backward (blue) sweep.  Node shapes,
    palette roles (GREEN input, BLUE intermediates, ORANGE loss/objective, GRAY
    given parameters), fonts, box size, and the ``fl.arrow`` helper match the
    backprop graph exactly.  No numbers are shown here: the figure is the
    schematic the prose calls for, and its numeric twin is the backprop graph."""
    fig, ax = plt.subplots(figsize=(11.2, 5.6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8.0)
    ax.set_aspect("equal")
    ax.axis("off")

    bw, bh = 2.05, 1.05             # node box size -- identical to backprop graph
    y_chain = 1.45                  # row A (bottom): the x -> ... -> L chain
    y_param = 4.20                  # row B (middle): the weight matrices
    y_obj = 6.55                    # row C (top): regularizer s, objective J
    centres = [1.55, 5.0, 8.0, 11.0, 14.3]   # x of x, z, h, o, L (as backprop)

    def node(cx, cy, name, color):
        """A rounded node box with a single symbol, in the shared house style:
        the same FancyBboxPatch + twin-fill look as the backprop graph's
        ``node``, minus the numeric value caption (this figure is schematic)."""
        x0, y0 = cx - bw / 2, cy - bh / 2
        for fc, a in [(color, 0.12), ("none", 1.0)]:
            ax.add_patch(FancyBboxPatch(
                (x0, y0), bw, bh,
                boxstyle="round,pad=0.02,rounding_size=0.12",
                linewidth=1.8, edgecolor=color, facecolor=fc, alpha=a))
        ax.text(cx, cy, name, ha="center", va="center",
                fontsize=13, color=color)
        return dict(l=x0, r=x0 + bw, cx=cx, t=y0 + bh, b=y0, cy=cy)

    # --- row A: main chain, input -> intermediates -> loss ---
    nx = node(centres[0], y_chain, r"$\mathbf{x}$", GREEN)
    nz = node(centres[1], y_chain, r"$\mathbf{z}$", BLUE)
    nh = node(centres[2], y_chain, r"$\mathbf{h}$", BLUE)
    no = node(centres[3], y_chain, r"$\mathbf{o}$", BLUE)
    nL = node(centres[4], y_chain, r"$L$", ORANGE)
    chain = [nx, nz, nh, no, nL]

    # forward arrows along the chain (black, above the row), labelled with the op.
    # The two affine transitions also receive a weight dropping in from above, so
    # their labels are nudged left of the arrow midpoint to clear that drop-arrow.
    fwd_ops = [(r"$\times\,\mathbf{W}^{(1)}$", -0.55), (r"$\phi$", 0.0),
               (r"$\times\,\mathbf{W}^{(2)}$", -0.55), (r"$l(\mathbf{o},y)$", 0.0)]
    yf = y_chain + bh / 2 + 0.05
    for (a, b), (op, dx) in zip(zip(chain[:-1], chain[1:]), fwd_ops):
        x0, x1 = a["r"] + 0.12, b["l"] - 0.12
        fl.arrow(ax, (x0, yf), (x1, yf), color=FWD, lw=2.0, mut=13)
        ax.text((x0 + x1) / 2 + dx, yf + 0.30, op, ha="center", va="bottom",
                fontsize=11.5, color=FWD)

    # --- row B: the weight matrices, sitting above their affine transitions ---
    # (GRAY = "given" parameters, distinct from the BLUE computed intermediates.)
    cW1 = (nx["r"] + nz["l"]) / 2        # above the x -> z affine arrow
    cW2 = (nh["r"] + no["l"]) / 2        # above the h -> o affine arrow
    nW1 = node(cW1, y_param, r"$\mathbf{W}^{(1)}$", GRAY)
    nW2 = node(cW2, y_param, r"$\mathbf{W}^{(2)}$", GRAY)

    # Each weight drops straight down into its affine forward-arrow.
    for nW in (nW1, nW2):
        fl.arrow(ax, (nW["cx"], nW["b"] - 0.02), (nW["cx"], yf + 0.02),
                 color=FWD, lw=1.8, mut=12)

    # --- row C: regularizer s (from the weights) and objective J = L + s ---
    cs = (cW1 + cW2) / 2                 # s centred over the two weight nodes
    ns = node(cs, y_obj, r"$s$", BLUE)
    nJ = node(centres[4], y_obj, r"$J$", ORANGE)   # J directly above L

    # Both weights feed up into the regularizer s (the weight-decay branch).
    for nW in (nW1, nW2):
        fl.arrow(ax, (nW["cx"], nW["t"] + 0.02), (ns["cx"], ns["b"] - 0.02),
                 color=FWD, lw=1.8, mut=12)
    ax.text(cs, ns["t"] + 0.18, r"$\frac{\lambda}{2}\|\cdot\|_\mathrm{F}^2$",
            ha="center", va="bottom", fontsize=11.5, color=FWD)

    # The objective combines s (rightward) and the loss L (rising):  J = L + s.
    fl.arrow(ax, (ns["r"] + 0.12, y_obj), (nJ["l"] - 0.12, y_obj),
             color=FWD, lw=2.0, mut=13)
    fl.arrow(ax, (nL["cx"], nL["t"] + 0.02), (nJ["cx"], nJ["b"] - 0.02),
             color=FWD, lw=2.0, mut=13)
    ax.text((ns["r"] + nJ["l"]) / 2, y_obj + 0.30, r"$J=L+s$",
            ha="center", va="bottom", fontsize=11.5, color=FWD)

    # --- corner direction label, echoing the backprop graph (forward only) ---
    ax.text(centres[0] - bw / 2, y_obj + 0.95, r"forward $\rightarrow$",
            ha="left", va="center", fontsize=11, color=FWD, fontweight="bold")

    fl.save(fig, "mdl-mlp-forward-graph")


FIGURES = [fig_backprop_graph, fig_mdl_mlp_forward_graph]


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
