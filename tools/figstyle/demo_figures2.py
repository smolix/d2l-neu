#!/usr/bin/env python3
"""Second batch of pilot remakes (10 pairs) — see demo_figures.py.

Composer remakes (Gen A legacy originals):
  qkv.svg, attention.svg, forward.svg, book-org.svg, seq2seq-state.svg,
  functionclasses.svg
matplotlib-theme remakes (Gen D tab10 originals):
  mdl-la-angle.svg, mdl-la-projection.svg, mdl-la-span.svg,
  mdl-la-eig-ellipse.svg

Run:  python3 tools/figstyle/demo_figures2.py
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from figstyle import tokens as T
from figstyle.svg import Figure, Span, sub, var
from figstyle.textmetrics import measure

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "img", "figstyle-demo")


def caps(f, x, y, text, anchor="middle"):
    f.text(x, y, text.upper(), size=T.FS_TINY, color=T.MUTED, anchor=anchor,
           tracking=T.LETTERSPACE_CAPS)


def star(f, cx, cy, r_out=18.0, r_in=7.2, color=T.GOLD):
    pts = []
    for i in range(10):
        r = r_out if i % 2 == 0 else r_in
        a = -math.pi / 2 + i * math.pi / 5
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    d = "M" + " L".join(f"{x:.2f} {y:.2f}" for x, y in pts) + " Z"
    f.path(d, cx - r_out, cy - r_out, cx + r_out, cy + r_out,
           fill=color.tint, stroke=color.base, sw=T.SW_BOX)


def multiblock(f, cx, cy, lines, *, role=None, w=170.0, lh=22.0):
    """Block with several centered text lines (line 1 may be a Span list)."""
    a = T.ROLE[role] if role else None
    h = lh * len(lines) + 2 * T.PAD_BLOCK_Y
    f.rect(cx - w / 2, cy - h / 2, w, h,
           fill=a.tint if a else T.PAPER, stroke=a.base if a else T.INK,
           sw=T.SW_BOX, r=T.R_BLOCK)
    y0 = cy - lh * (len(lines) - 1) / 2
    for i, ln in enumerate(lines):
        f.text(cx, y0 + i * lh, ln, size=T.FS_SMALL, color=T.INK)
    return (cx - w / 2, cy - h / 2, w, h)


# =========================================================================== #
# 1. qkv.svg — attention pooling by similarity                                #
# =========================================================================== #

def fig_qkv():
    f = Figure()
    KX, AX, VX, OX = 40.0, 240.0, 440.0, 620.0
    ROWS = [48.0, 106.0, 164.0, 222.0, 280.0]
    BW, BH = 70.0, 40.0

    caps(f, KX, 0, "keys")
    caps(f, AX, 0, "attention weights")
    caps(f, VX, 0, "values")
    caps(f, OX, 0, "output")

    # weights panel (the attention mechanism itself -> orange)
    PW = 184.0
    f.rect(AX - PW / 2, ROWS[0] - 28, PW, ROWS[-1] - ROWS[0] + 56,
           fill=T.ORANGE.tint, stroke=T.ORANGE.base, sw=T.SW_BOX, r=14)

    names = ["1", "2", None, None, "m"]
    for y, nm in zip(ROWS, names):
        # key block + arrow into the panel
        if nm:
            f.block(KX, y, [var("k"), sub(nm)], role="stream", min_w=BW)
            f.text(AX, y, [Span("α(", "r"), var("q"), Span(", ", "r"),
                           var("k"), sub(nm), Span(")", "r")],
                   size=T.FS_SMALL)
            f.block(VX, y, [var("v"), sub(nm)], role="stream", min_w=BW)
        else:
            f.circle(KX, y, 2.2, fill=T.INK, stroke="none", sw=0)
            f.circle(AX, y, 2.2, fill=T.INK, stroke="none", sw=0)
            f.circle(VX, y, 2.2, fill=T.INK, stroke="none", sw=0)
        f.arrow(KX + BW / 2 + 2, y, AX - PW / 2 - 3, y, stroke=T.INK)
        # weighting relationship: dashed (soft), not a hard data edge
        f.arrow(AX + PW / 2 + 2, y, VX - BW / 2 - 3, y, stroke=T.INK,
                dash=T.DASH_SOFT)
        # weighted values converge on the output (fan across its edge so
        # the heads stay distinct)
        ty = ROWS[2] + (ROWS.index(y) - 2) * 10.0
        f.arrow(VX + BW / 2 + 2, y, OX - 52, ty, stroke=T.INK)

    f.block(OX, ROWS[2], "output", role="ffn", min_w=80)

    # the query drives the weights: block below the panel, arrow up into it
    QY = ROWS[-1] + 88
    f.block(AX, QY, [var("q")], min_w=58)
    f.arrow(AX, QY - 24, AX, ROWS[-1] + 34, stroke=T.INK)
    caps(f, KX, QY, "query")

    f.save("qkv", out_dir=OUT,
           desc="Attention pooling: query-key similarities weight the values.")


# =========================================================================== #
# 2. attention.svg — attention reads from memory                              #
# =========================================================================== #

def fig_attention_memory():
    f = Figure()
    XS = [40.0, 132.0, 224.0]     # memory columns
    QX = 356.0                    # query/output column
    BW, BH = 58.0, 64.0

    # memory panel behind values + keys
    f.panel(-24, 6, 312, 282, label="memory")

    def tallblock(cx, cy, accent=None):
        a = accent
        f.rect(cx - BW / 2, cy - BH / 2, BW, BH,
               fill=a.tint if a else T.PAPER,
               stroke=a.base if a else T.INK, sw=T.SW_BOX, r=T.R_BLOCK)

    # attention bar spans out of the panel toward the query column
    BAR_Y = 160.0
    for x in XS:                                     # values (top) -> down
        tallblock(x, 72, T.BLUE)
        f.arrow(x, 104, x, BAR_Y - 22, stroke=T.INK)
    f.rect(-4, BAR_Y - 21, QX + 33, 42, fill=T.PAPER, stroke=T.INK,
           sw=T.SW_BOX, r=T.R_BLOCK)
    f.text((QX + 29) / 2, BAR_Y, "attention", size=T.FS_LABEL)
    for x in XS:                                     # keys (bottom) -> up
        tallblock(x, 248, T.BLUE)
        f.arrow(x, 216, x, BAR_Y + 22, stroke=T.INK)

    tallblock(QX, 248)                               # query (external)
    f.arrow(QX, 216, QX, BAR_Y + 22, stroke=T.INK)
    tallblock(QX, 72, T.GREEN)                       # output
    f.arrow(QX, BAR_Y - 22, QX, 104, stroke=T.INK)

    caps(f, -36, 72, "values", anchor="end")
    caps(f, -36, 248, "keys", anchor="end")
    caps(f, QX + 40, 72, "output", anchor="start")
    caps(f, QX + 40, 248, "query", anchor="start")

    f.save("attention", out_dir=OUT,
           desc="Attention over a memory of key-value pairs, driven by a query.")


# =========================================================================== #
# 3. forward.svg — forward computational graph of a one-hidden-layer net      #
# =========================================================================== #

def fig_forward():
    f = Figure()
    YB, YM, YT = 220.0, 118.0, 30.0   # bottom / middle / top rows
    R = 20.0                          # op-circle radius
    STEP = 78.0

    def opc(x, y, label, sz=T.FS_LABEL):
        f.circle(x, y, R, fill=T.BLUE.tint, stroke=T.BLUE.base, sw=T.SW_BOX)
        f.text(x, y, label, size=sz)

    def sq(x, y, label, role=None, w=46.0):
        return f.block(x, y, label, role=role, min_w=w, h=40.0)

    def edge(x1, y1, x2, y2):
        f.arrow(x1, y1, x2, y2, stroke=T.INK)

    # bottom (data) row: x -> (x) -> z -> phi -> h -> (x) -> o -> l
    X = [i * STEP for i in range(8)]
    sq(X[0], YB, [var("x")])
    opc(X[1], YB, "×")
    sq(X[2], YB, [var("z")])
    opc(X[3], YB, [var("φ")])
    sq(X[4], YB, [var("h")])
    opc(X[5], YB, "×")
    sq(X[6], YB, [var("o")])
    opc(X[7], YB, [var("l")])
    for i in range(7):
        left = X[i] + (23 if i % 2 == 0 else R + 2)
        right = X[i + 1] - (26 if (i + 1) % 2 == 0 else R + 3)
        edge(left, YB, right, YB)

    # parameters (purple) feed the products and the regularizer
    sq(X[1], YM, [Span("W", "b"), Span("(1)", "r", script=1)], role="embed",
       w=62)
    sq(X[5], YM, [Span("W", "b"), Span("(2)", "r", script=1)], role="embed",
       w=62)
    edge(X[1], YM + 21, X[1], YB - R - 3)
    edge(X[5], YM + 21, X[5], YB - R - 3)
    opc(X[3], YM, [Span("ℓ", "i"), sub("2")], sz=T.FS_SMALL)
    edge(X[1] + 32, YM, X[3] - R - 3, YM)
    edge(X[5] - 32, YM, X[3] + R + 3, YM)

    # objective row: l2 -> s -> + -> J ;  l -> L -> + ;  y -> l
    sq(X[3], YT, [var("s")])
    edge(X[3], YM - R - 3, X[3], YT + 21)
    PL = 448.0
    opc(PL, YT, "+")
    edge(X[3] + 24, YT, PL - R - 3, YT)
    sq(PL + 78, YT, [Span("J", "i")])
    edge(PL + R + 3, YT, PL + 78 - 26, YT)
    sq(504, YM, [Span("L", "i")])
    edge(X[7] - 13, YB - 16, 516, YM + 21)    # l -> L
    edge(491, YM - 16, PL + 12, YT + R - 3)   # L -> +
    sq(600, YM, [var("y")])
    edge(586, YM + 17, X[7] + 13, YB - 16)    # y -> l

    f.save("forward", out_dir=OUT,
           desc="Forward computational graph: data path, parameters, and loss.")


# =========================================================================== #
# 4. book-org.svg — chapter dependency graph                                  #
# =========================================================================== #

def fig_book_org():
    f = Figure()
    L, M, R = -300.0, 0.0, 300.0
    ROW = [0.0, 66.0, 138.0, 216.0, 288.0, 372.0, 452.0, 548.0]
    W2, W1 = 228.0, 205.0

    # foundations (white), methods (blue), applications (green)
    N = {}

    def node(key, x, y, lines, role=None, w=W1):
        N[key] = (x, y) + tuple(multiblock(f, x, y, lines, role=role, w=w))

    node("1", M, ROW[0], ["1. introduction"])
    node("2", M, ROW[1], ["2. preliminaries"])
    node("34", M, ROW[2], ["3–4. linear neural", "networks"])
    node("5", M, ROW[3], ["5. multilayer", "perceptrons"])
    node("6", M, ROW[4], ["6. builders’ guide"], role="stream")
    node("12", R, ROW[4], ["12. optimization", "algorithms"], role="stream")
    node("7", L, ROW[5], ["7. convolutional", "neural networks"], role="stream")
    node("9", M, ROW[5], ["9. recurrent", "neural networks"], role="stream")
    node("13", R, ROW[5], ["13. computational", "performance"], role="stream")
    node("8", L, ROW[6], ["8. modern convolutional", "networks"], role="stream", w=W2)
    node("10", M, ROW[6], ["10. modern recurrent", "networks"], role="stream", w=W2)
    node("11", R, ROW[7], ["11. attention and", "transformers"], role="stream", w=W2)
    node("14", L, ROW[7], ["14. computer vision"], role="ffn")
    node("nlp", M, ROW[7], ["15–16. natural language", "processing"], role="ffn", w=W2)

    def a2b(a, b):
        """Arrow from node a's bottom edge toward node b's top edge.

        Both endpoints lean toward the other node (clamped inside the
        edges) so diagonals take the shortest corridor between rows.
        """
        xa, ya, bx_a, by_a, w_a, h_a = N[a]
        xb, yb, bx_b, by_b, w_b, h_b = N[b]
        x1 = max(bx_a + 24, min(bx_a + w_a - 24, xb))
        y1 = by_a + h_a
        tx = max(bx_b + 18, min(bx_b + w_b - 18, x1))
        f.arrow(x1, y1 + 1, tx, by_b - 4, stroke=T.INK, sw=T.SW_HAIR * 1.2)

    for a, b in [("1", "2"), ("2", "34"), ("34", "5"), ("5", "6"),
                 ("5", "12"), ("6", "7"), ("6", "9"), ("6", "13"),
                 ("7", "8"), ("8", "14"), ("9", "10"),
                 ("10", "nlp"), ("12", "13"), ("13", "nlp")]:
        a2b(a, b)

    # The three long diagonals get hand-routed corridors (a straight line
    # from their source would cut through a middle-row box):
    # 13 -> 14: down the left side through the gap between rows 6 and 7.
    f.ortho_arrow([(228, N["13"][3] + N["13"][5] + 1), (228, 502),
                   (-228, 502), (-228, N["14"][3] - 4)],
                  stroke=T.INK, sw=T.SW_HAIR * 1.2)
    # 7 -> 15-16: from 7's bottom-right corner, clearing 8 and 10.
    f.arrow(-198, N["7"][3] + N["7"][5] + 1, -90, N["nlp"][3] - 4,
            stroke=T.INK, sw=T.SW_HAIR * 1.2)
    # 6 -> 11: from 6's bottom-right corner, clearing 9 and 13.
    f.arrow(100, N["6"][3] + N["6"][5] + 1, 246, N["11"][3] - 4,
            stroke=T.INK, sw=T.SW_HAIR * 1.2)

    # 11 -> 15-16 is horizontal (same row)
    x11, y11, bx11, by11, w11, h11 = N["11"]
    xn, yn, bxn, byn, wn, hn = N["nlp"]
    f.arrow(bx11 - 2, y11, bxn + wn + 4, yn, stroke=T.INK, sw=T.SW_HAIR * 1.2)

    f.save("book-org", out_dir=OUT,
           desc="Chapter dependency graph: foundations, methods, applications.")


# =========================================================================== #
# 5. seq2seq-state.svg — encoder-decoder through a state                      #
# =========================================================================== #

def fig_seq2seq_state():
    f = Figure()
    EX, SX, DX = 90.0, 268.0, 446.0
    RY, EY, IY, FY = 150.0, 214.0, 272.0, 84.0

    # column labels sit tight above each column's topmost element — no
    # dead band between label and diagram
    caps(f, EX, RY - 44, "encoder")
    caps(f, DX, FY - 42, "decoder")

    f.text(EX, IY, "sources", size=T.FS_LABEL, color=T.MUTED)
    f.arrow(EX, IY - 16, EX, EY + 24, stroke=T.INK)
    f.block(EX, EY, "embedding", min_w=140)
    f.arrow(EX, EY - 23, EX, RY + 24, stroke=T.INK)
    f.block(EX, RY, "recurrent", role="stream", min_w=140)
    f.text(EX - 88, RY, [Span("n", "i"), Span(" ×", "r")], size=T.FS_SMALL,
           color=T.MUTED, anchor="end")

    f.line(EX + 70, RY, SX - 55, RY, stroke=T.INK, sw=T.SW_LINE)
    f.block(SX, RY, "state", role="state", min_w=110)
    f.arrow(SX + 55, RY, DX - 74, RY, stroke=T.INK)

    f.text(DX, IY, "targets", size=T.FS_LABEL, color=T.MUTED)
    f.arrow(DX, IY - 16, DX, EY + 24, stroke=T.INK)
    f.block(DX, EY, "embedding", min_w=140)
    f.arrow(DX, EY - 23, DX, RY + 24, stroke=T.INK)
    f.block(DX, RY, "recurrent", role="stream", min_w=140)
    f.text(DX + 88, RY, [Span("× ", "r"), Span("n", "i")], size=T.FS_SMALL,
           color=T.MUTED, anchor="start")
    f.arrow(DX, RY - 23, DX, FY + 24, stroke=T.INK)
    f.block(DX, FY, "fully connected", min_w=140)

    f.save("seq2seq-state", out_dir=OUT,
           desc="Encoder-decoder: the encoder's state initializes the decoder.")


# =========================================================================== #
# 6. functionclasses.svg — non-nested vs nested function classes              #
# =========================================================================== #

def fig_functionclasses():
    """The POINT of this figure (do not lose it): with non-nested classes,
    growing capacity does NOT monotonically approach f* — F4 and F2 come
    close while F5, F3 drift away and F1 is farthest.  With nested classes,
    every enlargement contains the previous class, so the best approximation
    can only improve."""
    f = Figure()
    GRAYS = [T.PAPER, "#F0F2F5", "#E2E6EA", "#D2D8DD", "#C0C7CE"]

    def flabel(x, y, n, color=T.INK):
        f.text(x, y, [Span("F", "i"), sub(str(n))], size=T.FS_LABEL,
               color=color)

    # (a) non-nested: classes overlap and WANDER — their distance to f*
    # zigzags (F6 ~ mid, F5 farther, F4 near, F3 farther, F2 nearest gray,
    # F1 farthest of all).
    star(f, 350, -20)
    f.text(374, -20, [var("f"), Span("*", "r", script=1)], size=T.FS_LABEL,
           anchor="start")
    rects = [(0, 10, 240, 180), (30, 70, 210, 170), (120, -10, 190, 150),
             (95, 95, 180, 140), (215, 20, 145, 115)]
    for i, (x, y, w, h) in enumerate(rects):     # shapes first ...
        f.rect(x, y, w, h, fill=GRAYS[i], stroke=T.MUTED, sw=T.SW_HAIR, r=26)
    f.rect(35, 120, 95, 70, fill=T.BLUE.tint, stroke=T.BLUE.base,
           sw=T.SW_BOX, r=20)
    # ... then labels, each in a spot no later shape covers
    flabel(24, 34, 6)
    flabel(52, 222, 5)
    flabel(288, 12, 4)
    flabel(248, 214, 3)
    flabel(336, 44, 2)
    flabel(82, 155, 1, T.BLUE.dark)
    f.text(178, 282, "non-nested function classes", size=T.FS_LABEL,
           color=T.INK)

    # (b) nested: concentric classes — capacity grows without losing F1.
    # Left bands hold the F6/F5/F4 labels; bottom bands hold F3/F2.
    X0 = 520.0
    nested = [(X0, 0, 300, 230), (X0 + 30, 12, 246, 206),
              (X0 + 60, 24, 192, 182), (X0 + 90, 36, 138, 158),
              (X0 + 112, 48, 94, 116)]
    for i, (x, y, w, h) in enumerate(nested):   # rings first ...
        f.rect(x, y, w, h, fill=GRAYS[i], stroke=T.MUTED, sw=T.SW_HAIR, r=22)
    f.rect(X0 + 124, 60, 70, 72, fill=T.BLUE.tint, stroke=T.BLUE.base,
           sw=T.SW_BOX, r=14)
    for i in range(3):                          # ... then labels on top
        flabel(X0 + 15 + 30 * i, 115, 6 - i)    # F6 F5 F4 in the left bands
    flabel(X0 + 159, 179, 3)                    # F3, F2 in the bottom bands
    flabel(X0 + 159, 148, 2)
    flabel(X0 + 159, 96, 1, T.BLUE.dark)
    star(f, X0 + 296, -34)  # top-right of the nested panel
    f.text(X0 + 320, -34, [var("f"), Span("*", "r", script=1)],
           size=T.FS_LABEL, anchor="start")
    f.text(X0 + 150, 282, "nested function classes", size=T.FS_LABEL,
           color=T.INK)

    f.save("functionclasses", out_dir=OUT,
           desc="Non-nested versus nested function classes around f*.")


# =========================================================================== #
# 7-10. matplotlib-theme remakes of mdl-la-* figures                          #
# =========================================================================== #

def mpl_figs():
    import matplotlib
    matplotlib.use("svg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Arc, Polygon

    from figstyle.mpl import (arrow, axis_cross, clean_axes, right_angle,
                              save, use_style)

    use_style(hashsalt="figstyle-demo")

    # ---- mdl-la-angle ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    v = np.array([3.0, 0.0])
    w = np.array([1.2, 2.6])
    LBL = 19
    axis_cross(ax, (-0.4, 3.8), (-0.5, 2.9))
    arrow(ax, (0, 0), v, color=T.BLUE.base, lw=3.0)
    arrow(ax, (0, 0), w, color=T.ORANGE.base, lw=3.0)
    a1 = np.degrees(np.arctan2(v[1], v[0]))
    a2 = np.degrees(np.arctan2(w[1], w[0]))
    ax.add_patch(Arc((0, 0), 1.7, 1.7, angle=0, theta1=a1, theta2=a2,
                     color=T.MUTED, lw=T.SW_LINE))
    mid = np.radians((a1 + a2) / 2)
    ax.text(1.2 * np.cos(mid), 1.2 * np.sin(mid), r"$\theta$",
            color=T.MUTED, fontsize=20, ha="center", va="center")
    ax.text(v[0], -0.34, r"$\mathbf{v}$", color=T.BLUE.dark, fontsize=LBL,
            ha="center", va="center")
    ax.text(w[0] - 0.05, w[1] + 0.28, r"$\mathbf{w}$", color=T.ORANGE.dark,
            fontsize=LBL, ha="center", va="center")
    clean_axes(ax, lim=((-0.5, 3.9), (-0.6, 3.0)), hide=True)
    save(fig, "mdl-la-angle", out_dir=OUT)

    # ---- mdl-la-projection -------------------------------------------------
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(8.4, 3.7))
    LBL, FML = 18, 16
    LIM = ((-0.4, 4.1), (-0.5, 2.7))
    XR, YR = (-0.3, 4.0), (-0.4, 2.6)

    w1 = np.array([3.2, 0.6])
    v1 = np.array([1.6, 2.2])
    wn = w1 / np.linalg.norm(w1)
    proj = float(v1 @ wn) * wn
    axis_cross(axa, XR, YR)
    arrow(axa, (0, 0), w1, color=T.ORANGE.base, lw=2.8)
    arrow(axa, (0, 0), v1, color=T.BLUE.base, lw=3.0)
    axa.plot([0, proj[0]], [0, proj[1]], color=T.GREEN.base, lw=6,
             solid_capstyle="round", zorder=2)
    axa.plot([proj[0], v1[0]], [proj[1], v1[1]], ls=(0, (4, 3)),
             color=T.MUTED, lw=1.6)
    right_angle(axa, proj, -wn, (v1 - proj), size=0.22, color=T.MUTED)
    axa.text(w1[0] + 0.05, w1[1] - 0.24, r"$\mathbf{w}$", color=T.ORANGE.dark,
             ha="left", fontsize=LBL)
    axa.text(v1[0] - 0.12, v1[1] + 0.24, r"$\mathbf{v}$", color=T.BLUE.dark,
             ha="center", fontsize=LBL)
    axa.text(proj[0] * 0.55 + 0.05, proj[1] * 0.55 + 0.32,
             r"$\|\mathbf{v}\|\cos\theta$", color=T.GREEN.dark, ha="center",
             fontsize=FML)
    axa.text((proj[0] + v1[0]) / 2 + 0.26, (proj[1] + v1[1]) / 2,
             r"$\mathbf{r}$", color=T.MUTED, ha="left", fontsize=LBL)
    clean_axes(axa, lim=LIM, hide=True)

    w2 = np.array([3.4, 1.3])
    v2 = 0.62 * w2
    axis_cross(axb, XR, YR)
    arrow(axb, (0, 0), w2, color=T.ORANGE.base, lw=2.8)
    axb.plot([0, 0.86 * v2[0]], [0, 0.86 * v2[1]], color=T.GREEN.base, lw=6,
             solid_capstyle="round", zorder=2)
    arrow(axb, (0, 0), v2, color=T.BLUE.base, lw=3.0)
    axb.text(w2[0] + 0.05, w2[1] + 0.02, r"$\mathbf{w}$",
             color=T.ORANGE.dark, ha="left", fontsize=LBL)
    axb.text(v2[0] * 0.5 - 0.12, v2[1] * 0.5 + 0.34, r"$\mathbf{v}$",
             color=T.BLUE.dark, ha="center", fontsize=LBL)
    axb.text(1.6, 1.5,
             r"$|\mathbf{v}\cdot\mathbf{w}| = \|\mathbf{v}\|\,\|\mathbf{w}\|$",
             ha="center", va="center", fontsize=FML, color=T.INK)
    clean_axes(axb, lim=LIM, hide=True)
    save(fig, "mdl-la-projection", out_dir=OUT)

    # ---- mdl-la-span -------------------------------------------------------
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(8.8, 3.9))
    LBL, FML = 18, 16
    LIM = ((-3.4, 6.2), (-2.6, 4.6))
    XR, YR = (-3.2, 6.0), (-2.4, 4.4)
    ang = np.radians(-10.0)
    Rm = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])

    v = Rm @ np.array([2.0, 1.0])
    axis_cross(axa, XR, YR)
    axa.plot([-1.6 * v[0], 2.9 * v[0]], [-1.6 * v[1], 2.9 * v[1]],
             ls=(0, (4, 3)), color=T.MUTED, lw=1.6)
    arrow(axa, (0, 0), v, color=T.BLUE.base, lw=3.0)
    axa.text(v[0] + 0.10, v[1] + 0.42, r"$\mathbf{v}$", color=T.BLUE.dark,
             ha="left", fontsize=LBL)
    d = v / np.linalg.norm(v)
    perp = np.array([-d[1], d[0]])
    lpos = 2.3 * v + 0.5 * perp
    axa.text(lpos[0], lpos[1], r"$\mathrm{span}(\mathbf{v})$", color=T.MUTED,
             fontsize=FML, ha="center", va="center",
             rotation=np.degrees(np.arctan2(d[1], d[0])),
             rotation_mode="anchor")
    clean_axes(axa, lim=LIM, hide=True)

    u = Rm @ np.array([2.0, 1.0])
    w = Rm @ np.array([0.5, 1.5])
    coeffs = range(-1, 4)
    for a_ in coeffs:
        p0, p1 = a_ * u + coeffs[0] * w, a_ * u + coeffs[-1] * w
        axb.plot([p0[0], p1[0]], [p0[1], p1[1]], color=T.FAINT,
                 lw=T.SW_HAIR * 0.8)
    for b_ in coeffs:
        p0, p1 = coeffs[0] * u + b_ * w, coeffs[-1] * u + b_ * w
        axb.plot([p0[0], p1[0]], [p0[1], p1[1]], color=T.FAINT,
                 lw=T.SW_HAIR * 0.8)
    x = 2 * u + w
    axis_cross(axb, XR, YR)
    axb.plot([2 * u[0], x[0]], [2 * u[1], x[1]], ls=(0, (4, 3)),
             color=T.MUTED, lw=1.6)
    axb.plot([w[0], x[0]], [w[1], x[1]], ls=(0, (4, 3)), color=T.MUTED,
             lw=1.6)
    arrow(axb, (0, 0), u, color=T.BLUE.base, lw=3.0)
    arrow(axb, (0, 0), w, color=T.ORANGE.base, lw=3.0)
    arrow(axb, (0, 0), x, color=T.GREEN.base, lw=3.2)
    axb.text(u[0] + 0.05, u[1] - 0.44, r"$\mathbf{u}$", color=T.BLUE.dark,
             ha="center", fontsize=LBL)
    axb.text(w[0] - 0.30, w[1] + 0.30, r"$\mathbf{w}$", color=T.ORANGE.dark,
             ha="right", fontsize=LBL)
    axb.text(x[0] - 0.9, x[1] + 1.15, r"$\mathbf{x}=2\mathbf{u}+\mathbf{w}$",
             color=T.GREEN.dark, ha="center", va="center", fontsize=FML,
             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none",
                       alpha=0.92))
    clean_axes(axb, lim=LIM, hide=True)
    save(fig, "mdl-la-span", out_dir=OUT)

    # ---- mdl-la-eig-ellipse ------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2))
    t = np.linspace(0, 2 * np.pi, 400)
    circle = np.vstack([np.cos(t), np.sin(t)])
    LBL = 17
    specs = [np.array([[2.0, 0.0], [0.0, -1.0]]),
             np.array([[2.0, 1.0], [1.0, 2.0]])]
    for ax, A in zip(axes, specs):
        ellipse = A @ circle
        wv, V = np.linalg.eigh(A)
        ax.plot(circle[0], circle[1], ls=(0, (4, 3)), color=T.FAINT, lw=1.2)
        ax.plot(ellipse[0], ellipse[1], color=T.BLUE.base, lw=3.0)
        order = np.argsort(-np.abs(wv))
        labels = [r"$\lambda_1$", r"$\lambda_2$"]
        for rank, idx in enumerate(order):
            vec = V[:, idx]
            lam = wv[idx]
            tip = lam * vec
            acc = T.GREEN if lam >= 0 else T.ORANGE
            arrow(ax, (0, 0), tip, color=acc.base, lw=3.2)
            dd = tip / np.linalg.norm(tip)
            pp = np.array([-dd[1], dd[0]])
            sign = "" if lam >= 0 else " (flip)"
            if abs(dd[1]) < 0.35:
                lpos = tip + np.array([0.18 * np.sign(dd[0]), 1.02])
                ha = "center"
            elif abs(dd[0]) < 0.35:
                lpos = np.array([-0.15, tip[1] + 0.55 * np.sign(tip[1])])
                ha = "right"
            else:
                lpos = tip + 0.90 * dd + 0.14 * pp
                ha = "center"
            ax.text(lpos[0], lpos[1], rf"{labels[rank]}$={lam:.2g}${sign}",
                    color=acc.dark, fontsize=LBL, ha=ha, va="center")
        m = 3.0
        clean_axes(ax, lim=((-m, m), (-m, m)), hide=True)
        axis_cross(ax, (-m, m), (-m, m))
    save(fig, "mdl-la-eig-ellipse", out_dir=OUT)


def main():
    os.makedirs(OUT, exist_ok=True)
    fig_qkv()
    fig_attention_memory()
    fig_forward()
    fig_book_org()
    fig_seq2seq_state()
    fig_functionclasses()
    mpl_figs()
    print("wrote batch-2 figures to", OUT)


if __name__ == "__main__":
    main()
