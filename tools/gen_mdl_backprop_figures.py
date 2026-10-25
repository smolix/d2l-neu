#!/usr/bin/env python3
"""Generate the computational-graph figures of the backpropagation section
(``chapter_multilayer-perceptrons/backprop.md``).

Three figures, drawn with the deterministic SVG composer (figstyle.svg) in the
visual grammar of the computational graphs of the automatic differentiation
section (``img/autograd-comp-graph.svg``): rounded nodes, parameters on a gray
fill, the forward pass in blue (the ``stream`` role), the backward pass in red
(the ``grad`` role), values in black and gradients in red inside the nodes.

* ``mdl-mlp-backprop-small``  -- e = (a + b) c with its forward values and
  backward gradients (the smallest worked example).
* ``mdl-mlp-forward-graph``   -- the forward graph of the one-hidden-layer MLP
  with weight decay, with the shape of every node for the MLP of the
  implementation section (n examples, 784 inputs, 256 hidden units, 10 classes).
* ``mdl-mlp-backprop-graph``  -- the section's 2-2-1 worked example: the same
  graph with every value and every gradient.

Every number is computed here, so the pictures cannot drift from the prose.
Coordinates are px (1 SVG unit = 1 CSS px), y downward.  Run with the repo's
pytorch venv (it has fontTools):

    .venv-pytorch/bin/python tools/gen_mdl_backprop_figures.py

The output is byte-idempotent.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from figstyle import tokens as T  # noqa: E402
from figstyle.svg import Figure, Span, var  # noqa: E402
from figstyle.textmetrics import measure  # noqa: E402

FWD = T.BLUE.base        # forward pass: arrows and their labels
BWD = T.RED.base         # backward pass: arrows
GRAD = T.RED.dark        # gradient values and backward labels (text grade)
EDGE = T.MUTED           # node outlines
SW = 2.0                 # arrow stroke (pairs of arrows run 16 px apart)
FS_VAL = 15.0            # values and gradients, monospace
OFF = 8.0                # half-distance between a forward/backward arrow pair

WRITTEN: list[str] = []


# --------------------------------------------------------------------------- #
# Text helpers                                                                #
# --------------------------------------------------------------------------- #

def bold(text):
    """A vector or matrix symbol (bold upright, as in the prose)."""
    return Span(text, "b")


def sup(text):
    return Span(text, "r", script=+1)


def W(layer):
    return [bold("W"), sup(f"({layer})")]


def num(x):
    """Integer-valued float -> string with a true minus sign; -0 -> 0."""
    x = float(np.ravel(x)[0]) + 0.0
    return f"{x:.0f}".replace("-", "−") if x != 0 else "0"


def span_width(spans, size):
    total = 0.0
    for sp in spans:
        fam, sty = {"r": ("sans", "regular"), "b": ("sans", "bold"),
                    "i": ("sans", "italic"), "m": ("mono", "regular")}[sp.kind]
        total += measure(sp.text, size * (0.72 if sp.script else 1.0),
                         fam, sty).width
    return total


def label(f, x, y, parts, *, size=T.FS_SMALL, color=T.INK, anchor="middle"):
    """Typeset spans, where the string "ODOT" draws a Hadamard-product circle
    (the bundled fonts have no U+2299)."""
    if all(isinstance(p, Span) for p in parts):
        parts = [parts]
    r = 0.30 * size
    widths = [2 * r + 0.25 * size if p == "ODOT" else span_width(p, size)
              for p in parts]
    total = sum(widths)
    x0 = {"middle": x - total / 2, "end": x - total}.get(anchor, x)
    for p, w in zip(parts, widths):
        if p == "ODOT":
            cx, cy = x0 + w / 2, y
            f.circle(cx, cy, r, fill=T.PAPER, stroke=color, sw=1.2)
            f.circle(cx, cy, 1.3, fill=color, stroke=None, sw=0)
        else:
            f.text(x0, y, p, size=size, color=color, anchor="start")
        x0 += w


# --------------------------------------------------------------------------- #
# Graph helpers                                                               #
# --------------------------------------------------------------------------- #

class Node:
    """A rounded node: a title row, then rows of (text, color) or matrices."""

    def __init__(self, f, cx, cy, title, rows=(), *, w=84.0, h=None,
                 param=False, title_size=T.FS_LABEL):
        self.cx, self.cy, self.w = cx, cy, w
        heights = [r.get("h", 20.0) for r in rows]
        self.h = h or (34.0 + sum(heights) + (8.0 if rows else 0.0))
        x0, y0 = cx - w / 2, cy - self.h / 2
        f.rect(x0, y0, w, self.h, fill=T.PANEL if param else T.PAPER,
               stroke=EDGE, sw=T.SW_BOX, r=T.R_BLOCK)
        f.text(cx, y0 + 18.0, title, size=title_size)
        if rows:
            f.line(x0 + 10, y0 + 34.0, x0 + w - 10, y0 + 34.0,
                   stroke=T.HAIRLINE, sw=1.5)
        y = y0 + 38.0
        for r, hh in zip(rows, heights):
            if "matrix" in r:
                matrix(f, cx, y + hh / 2, r["matrix"], r["color"])
            elif "spans" in r:
                f.text(cx, y + hh / 2, r["spans"], size=T.FS_SMALL,
                       color=r["color"])
            else:
                f.text(cx, y + hh / 2, [Span(r["text"], "m")], size=FS_VAL,
                       color=r["color"])
            y += hh
        self.l, self.r = x0, x0 + w
        self.t, self.b = y0, y0 + self.h


def matrix(f, cx, cy, M, color):
    """A small matrix of integers in monospace, between drawn brackets."""
    M = np.atleast_2d(M)
    rows = [" ".join(f"{num(x):>2}" for x in row) for row in M]
    lh = 17.0
    width = max(measure(s, FS_VAL, "mono").width for s in rows)
    for i, s in enumerate(rows):
        f.text(cx, cy + (i - (len(rows) - 1) / 2) * lh, [Span(s, "m")],
               size=FS_VAL, color=color)
    hx, hy = width / 2 + 5, len(rows) * lh / 2 + 1
    for sgn in (-1, 1):
        x = cx + sgn * hx
        f.polyline([(x - sgn * 4, cy - hy), (x, cy - hy), (x, cy + hy),
                    (x - sgn * 4, cy + hy)], stroke=color, sw=1.3)


def legend(f, x, y, text, color, anchor="start", var_last=False):
    """A bold legend line; ``var_last`` sets its last word as a variable."""
    if var_last:
        head, tail = text.rsplit(" ", 1)
        spans = [Span(head + " ", "b"), Span(tail, "i")]
    else:
        spans = [Span(text, "b")]
    f.text(x, y, spans, size=T.FS_SMALL, color=color, anchor=anchor)


def h_pair(f, a, b, y, *, fwd=None, bwd=None, back=True):
    """Forward arrow a -> b above y, backward arrow b -> a below y, with
    optional labels (lists of spans) above and below the pair."""
    x0, x1 = a.r + 3, b.l - 3
    f.arrow(x0, y - OFF, x1, y - OFF, stroke=FWD, sw=SW)
    if back:
        f.arrow(x1, y + OFF, x0, y + OFF, stroke=BWD, sw=SW)
    xm = (x0 + x1) / 2
    if fwd:
        label(f, xm, y - OFF - 14, fwd, color=FWD)
    if bwd:
        label(f, xm, y + OFF + 15, bwd, color=GRAD)


def v_pair(f, top, bottom, x, *, bwd=None, back=True):
    """Forward arrow down from a parameter node into its layer; backward arrow
    up beside it, labelled on the right."""
    y0, y1 = top.b + 3, bottom.t - 3
    f.arrow(x - OFF, y0, x - OFF, y1, stroke=FWD, sw=SW)
    if back:
        f.arrow(x + OFF, y1, x + OFF, y0, stroke=BWD, sw=SW)
    if bwd:
        label(f, x + OFF + 8, (y0 + y1) / 2, bwd, color=GRAD, anchor="start")


def save(f, name, desc):
    path = f.save(name, desc=desc)
    WRITTEN.append(path)


# --------------------------------------------------------------------------- #
# e = (a + b) c                                                               #
# --------------------------------------------------------------------------- #

def fig_backprop_small():
    """POINT: the backward pass visits the nodes in reverse order; an addition
    passes its incoming gradient unchanged, a multiplication scales it by the
    other input, whose forward value must therefore be kept."""
    a, b, c = 2.0, 1.0, -3.0
    u = a + b
    e = u * c
    de = 1.0
    du, dc = de * c, de * u          # multiply: scale by the other input
    da, db = du * 1.0, du * 1.0      # add: pass through
    assert (u, e, da, db, dc) == (3.0, -9.0, -3.0, -3.0, 3.0)

    f = Figure()
    w = 108.0

    def node(cx, cy, title, v, g):
        return Node(f, cx, cy, title,
                    [dict(text=num(v), color=T.INK),
                     dict(text=num(g), color=GRAD)], w=w)

    na = node(60, 60, [var("a")], a, da)
    nb = node(60, 220, [var("b")], b, db)
    nu = node(290, 140, [var("u"), Span(" = "), var("a"), Span(" + "),
                         var("b")], u, du)
    nc = node(290, 300, [var("c")], c, dc)
    ne = node(520, 220, [var("e"), Span(" = "), var("u"), Span(" · "),
                         var("c")], e, de)

    def edge(src, dst, side, bwd_label):
        """Arrow pair on the line between two node centers.  ``side`` = +1
        puts the backward arrow below the forward one (in screen terms),
        -1 above; its label goes further out on the same side."""
        dx, dy = dst.cx - src.cx, dst.cy - src.cy
        L = math.hypot(dx, dy)
        ux, uy = dx / L, dy / L
        nx, ny = -uy, ux                      # unit normal, pointing down
        t0 = (w / 2) / abs(dx)                # leave src through its side
        t1 = 1 - (w / 2) / abs(dx)
        p0 = (src.cx + dx * t0 + 3 * ux, src.cy + dy * t0 + 3 * uy)
        p1 = (src.cx + dx * t1 - 3 * ux, src.cy + dy * t1 - 3 * uy)
        fo, bo = -side * OFF, side * OFF
        f.arrow(p0[0] + nx * fo, p0[1] + ny * fo,
                p1[0] + nx * fo, p1[1] + ny * fo, stroke=FWD, sw=SW)
        f.arrow(p1[0] + nx * bo, p1[1] + ny * bo,
                p0[0] + nx * bo, p0[1] + ny * bo, stroke=BWD, sw=SW)
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        lo = side * (OFF + 17)
        label(f, mx + nx * lo, my + ny * lo, bwd_label, color=GRAD)

    times = Span("×")
    edge(na, nu, +1, [times, Span("1")])
    edge(nb, nu, -1, [times, Span("1")])
    edge(nu, ne, +1, [times, var("c")])
    edge(nc, ne, -1, [times, var("u")])

    legend(f, 6, 300, "forward pass: values", FWD)
    legend(f, 6, 324, "backward pass: gradients of e", GRAD, var_last=True)
    save(f, "mdl-mlp-backprop-small",
         "Computational graph of e = (a + b) c at a = 2, b = 1, c = -3 with "
         "forward values and backward gradients.")


# --------------------------------------------------------------------------- #
# The MLP: forward graph and the worked example                               #
# --------------------------------------------------------------------------- #

XS = [50.0, 204.0, 358.0, 512.0, 666.0]   # x of X, Z, H, O, L (both figures)


def fig_forward_graph():
    """POINT: the forward pass of a one-hidden-layer MLP with weight decay is
    a graph whose nodes are the layer's intermediate values; each weight
    reaches the objective along two paths (the loss and the penalty), and the
    shapes say what the values cost to keep."""
    f = Figure()
    yc = 330.0

    def chain(cx, title, shape):
        return Node(f, cx, yc, title, [dict(text=shape, color=T.MUTED)])

    nX = chain(XS[0], [bold("X")], "n×784")
    nZ = chain(XS[1], [bold("Z")], "n×256")
    nH = chain(XS[2], [bold("H")], "n×256")
    nO = chain(XS[3], [bold("O")], "n×10")
    nL = chain(XS[4], [var("L")], "scalar")

    yp = 205.0
    nW1 = Node(f, XS[1], yp, W(1), [dict(text="784×256", color=T.MUTED)],
               param=True)
    nW2 = Node(f, XS[3], yp, W(2), [dict(text="256×10", color=T.MUTED)],
               param=True)
    ys = 80.0
    ns = Node(f, XS[2], ys, [var("s")],
              [dict(spans=[Span("penalty")], color=T.MUTED)])
    nJ = Node(f, XS[4], ys, [var("J")],
              [dict(spans=[var("L"), Span(" + "), var("s")], color=T.MUTED)])

    for a, b, lab in [(nX, nZ, None), (nZ, nH, [var("ϕ")]),
                      (nH, nO, None), (nO, nL, None)]:
        x0, x1 = a.r + 3, b.l - 3
        f.arrow(x0, yc, x1, yc, stroke=FWD, sw=SW)
        if lab:
            label(f, (x0 + x1) / 2, yc - 16, lab, color=FWD)
    for nW, n in [(nW1, nZ), (nW2, nO)]:
        f.arrow(n.cx, nW.b + 3, n.cx, n.t - 3, stroke=FWD, sw=SW)
    # both weights also feed the weight-decay penalty s (a fork)
    for nW, sx in [(nW1, ns.l + 20), (nW2, ns.r - 20)]:
        f.arrow(nW.cx, nW.t - 3, sx, ns.b + 3, stroke=FWD, sw=SW)
    f.arrow(ns.r + 3, ys, nJ.l - 3, ys, stroke=FWD, sw=SW)
    f.arrow(nL.cx, nL.t - 3, nL.cx, nJ.b + 3, stroke=FWD, sw=SW)

    legend(f, XS[0] - 42, ys, "forward pass", FWD)
    save(f, "mdl-mlp-forward-graph",
         "Forward computational graph of a one-hidden-layer MLP with weight "
         "decay, with the shape of every node.")


def fig_backprop_graph():
    """POINT: the worked example, with every forward value and every
    gradient; the backward pass reaches W(2) before W(1), and the inactive
    first hidden unit zeros the first column of the gradient of W(1)."""
    X = np.array([[1.0, 2.0]])
    W1 = np.array([[1.0, 0.0], [-1.0, 1.0]])
    W2 = np.array([[2.0], [-1.0]])
    y = 0.0
    Z = X @ W1
    H = np.maximum(Z, 0.0)
    O = H @ W2
    L = 0.5 * float(((O - y) ** 2).sum())
    G_O = O - y
    dW2 = H.T @ G_O
    G_H = G_O @ W2.T
    G_Z = G_H * (Z > 0)
    dW1 = X.T @ G_Z
    assert L == 2.0 and np.allclose(dW1, [[0, 2], [0, 4]])
    assert np.allclose(dW2, [[0], [-4]]) and np.allclose(G_Z, [[0, 2]])

    f = Figure()
    yc = 330.0

    def chain(cx, title, value, grad=None):
        """A value node; 1 x 2 rows are drawn as bracketed matrices, as in
        the prose (parentheses would denote column vectors)."""
        def row(v, color):
            if v is None:
                return dict(text="", color=color, h=24.0)
            if isinstance(v, np.ndarray):
                return dict(matrix=v, color=color, h=24.0)
            return dict(text=v, color=color, h=24.0)
        return Node(f, cx, yc, title, [row(value, T.INK), row(grad, GRAD)])

    nX = chain(XS[0], [bold("X")], X)
    nZ = chain(XS[1], [bold("Z")], Z, G_Z)
    nH = chain(XS[2], [bold("H")], H, G_H)
    nO = chain(XS[3], [var("o")], num(O), num(G_O))
    nL = chain(XS[4], [var("L")], num(L), "1")

    yp = 160.0
    mrow = dict(h=40.0)
    nW1 = Node(f, XS[1], yp, W(1),
               [dict(matrix=W1, color=T.INK, **mrow),
                dict(matrix=dW1, color=GRAD, **mrow)], param=True)
    nW2 = Node(f, XS[3], yp, W(2),
               [dict(matrix=W2, color=T.INK, **mrow),
                dict(matrix=dW2, color=GRAD, **mrow)], param=True)

    T_ = sup("T")
    h_pair(f, nX, nZ, yc, back=False)
    h_pair(f, nZ, nH, yc, fwd=[Span("ReLU")],
           bwd=["ODOT", [Span(" "), var("ϕ"), Span("′("), bold("Z"),
                         Span(")")]])
    h_pair(f, nH, nO, yc, bwd=[[Span("· "), bold("W"), sup("(2)T")]])
    h_pair(f, nO, nL, yc, fwd=[[Span("½("), var("o"), Span(" − "),
                                var("y"), Span(")"), sup("2")]],
           bwd=[[var("o"), Span(" − "), var("y")]])
    v_pair(f, nW1, nZ, XS[1], bwd=[[bold("X"), T_, Span(" ·")]])
    v_pair(f, nW2, nO, XS[3], bwd=[[bold("H"), T_, Span(" ·")]])

    legend(f, XS[2], nW1.t + 10, "forward pass: values", FWD, anchor="middle")
    legend(f, XS[2], nW1.t + 34, "backward pass: gradients of L", GRAD,
           anchor="middle", var_last=True)
    save(f, "mdl-mlp-backprop-graph",
         "The worked example as a computational graph with forward values and "
         "backward gradients.")


FIGURES = [fig_backprop_small, fig_forward_graph, fig_backprop_graph]


def main():
    for fn in FIGURES:
        fn()
    print(f"Wrote {len(WRITTEN)} figures:")
    for p in WRITTEN:
        size = os.path.getsize(p)
        assert size > 0, f"empty: {p}"
        print(f"  {os.path.basename(p):32s} {size:>8,d} bytes")


if __name__ == "__main__":
    main()
