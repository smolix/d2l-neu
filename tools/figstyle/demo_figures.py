#!/usr/bin/env python3
"""Pilot figures for the unified style — regenerated from three legacy
generations (see docs/figure-style-guide.md §Pilots).

Writes to img/figstyle-demo/ (does NOT touch the live corpus):
  mlp.svg                            <- remake of img/mlp.svg           (Gen A)
  seq2seq-attention.svg              <- remake of img/seq2seq-attention.svg (Gen A)
  mdl-transformers-block-anatomy.svg <- remake of the tab10 original    (Gen D)
  mdl-la-vector-add.svg              <- remake via the matplotlib theme (Gen D)

Run:  python3 tools/figstyle/demo_figures.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from figstyle import tokens as T
from figstyle.svg import Figure, Span, sub, var
from figstyle.textmetrics import measure

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "img", "figstyle-demo")


# =========================================================================== #
# 1. MLP network graph (remake of img/mlp.svg)                                #
# =========================================================================== #

def fig_mlp():
    import math

    f = Figure()
    R = 21.0                      # node radius
    ROWY = {"out": 0.0, "hid": 140.0, "in": 280.0}
    XC = 300.0                    # column center

    def xs(n, gap=100.0):
        return [XC + (i - (n - 1) / 2) * gap for i in range(n)]

    nodes = {
        "in": (xs(4), "x", None),          # white, ink outline
        "hid": (xs(5), "h", T.BLUE),
        "out": (xs(3), "o", T.GREEN),
    }

    # Edges first (underneath): directed, with arrowheads (they carry the
    # feed-forward semantics), but in a quiet gray so the nodes stay figure.
    for a, b in (("in", "hid"), ("hid", "out")):
        for xa in nodes[a][0]:
            for xb in nodes[b][0]:
                ya, yb = ROWY[a], ROWY[b]
                ang = math.atan2(yb - ya, xb - xa)
                f.arrow(xa + R * math.cos(ang), ya + R * math.sin(ang),
                        xb - (R + 2) * math.cos(ang),
                        yb - (R + 2) * math.sin(ang),
                        stroke=T.FAINT, sw=T.SW_HAIR)

    for row, (positions, letter, accent) in nodes.items():
        for i, x in enumerate(positions, 1):
            if accent:
                f.circle(x, ROWY[row], R, fill=accent.tint,
                         stroke=accent.base, sw=T.SW_BOX)
            else:
                f.circle(x, ROWY[row], R, fill=T.PAPER, stroke=T.INK,
                         sw=T.SW_BOX)
            f.text(x, ROWY[row], [var(letter), sub(str(i))], size=T.FS_LABEL)

    # Layer labels: quiet caps, left-aligned like panel labels.
    lx = min(nodes["in"][0]) - 175
    for row, lab in (("out", "output layer"), ("hid", "hidden layer"),
                     ("in", "input layer")):
        f.text(lx, ROWY[row], lab.upper(), size=T.FS_TINY, color=T.MUTED,
               anchor="start", tracking=T.LETTERSPACE_CAPS)

    f.save("mlp", out_dir=OUT,
           desc="A multilayer perceptron with one hidden layer.")


# =========================================================================== #
# 2. Transformer block anatomy: post-LN vs pre-LN (remake, Gen D original)    #
# =========================================================================== #

def _tblock(f: Figure, ox: float, *, pre_ln: bool, title: str):
    """One panel. ox = x of the residual spine. y grows DOWN; data flows UP."""
    BX = ox + 180          # branch-block center x
    BW = 208.0             # branch-block width (shared -> aligned edges)

    def branch(y_take, y_block, y_join, label, role):
        """Spine -> horizontal takeoff -> riser -> block -> rejoin at (+).

        In pre-LN panels, the norm block sits ON the horizontal takeoff
        edge (the original figure's trick): the stream visibly passes
        through norm before entering the sublayer.
        """
        a = T.ROLE[role]
        # takeoff: spine -> right along y_take, then rise into the block
        f.ortho_arrow([(ox, y_take), (BX, y_take), (BX, y_block + 24)],
                      stroke=a.base, sw=T.SW_LINE)
        if pre_ln:
            f.block((ox + BX) / 2, y_take, "norm", min_w=86)
        f.block(BX, y_block, label, role=role, min_w=BW)
        # rejoin: block top -> up -> left into the (+) on the spine
        f.ortho_arrow([(BX, y_block - 24), (BX, y_join), (ox + 13, y_join)],
                      stroke=a.base, sw=T.SW_LINE)
        f.pill_op(ox, y_join, "+")

    Y0, Y1 = 390.0, 14.0   # spine bottom / top (y-down; flow upward)
    f.text(ox, Y0 + 32, [var("x")], size=T.FS_LABEL)
    # the residual stream: THE emphasized path -> heavy blue
    f.line(ox, Y0, ox, Y1 + 14, stroke=T.BLUE.base, sw=T.SW_HEAVY)
    f.arrow(ox, Y1 + 16, ox, Y1, stroke=T.BLUE.base, sw=T.SW_HEAVY)

    if pre_ln:
        branch(350, 288, 226, "multi-head attention", "attention")
        branch(182, 120, 58, "feed-forward network", "ffn")
    else:
        branch(350, 310, 266, "multi-head attention", "attention")
        f.block(ox, 224, "norm", min_w=86)
        branch(182, 142, 98, "feed-forward network", "ffn")
        f.block(ox, 56, "norm", min_w=86)

    f.text((ox + BX + BW / 2) / 2, -26, title, size=T.FS_TITLE, color=T.INK)


def fig_block_anatomy():
    f = Figure()
    _tblock(f, 0, pre_ln=False, title="post-LN (2017)")
    _tblock(f, 420, pre_ln=True, title="pre-LN (modern)")
    # annotate the residual stream once, in the free zone below the right
    # panel's attention takeoff (leaders point AT things, from open space)
    f.leader(520, 392, 434, 372)
    f.text(526, 392, "residual stream", size=T.FS_SMALL, color=T.MUTED,
           anchor="start")
    f.save("mdl-transformers-block-anatomy", out_dir=OUT,
           desc="Post-LN versus pre-LN transformer block wiring.")


# =========================================================================== #
# 3. seq2seq with attention (remake of img/seq2seq-attention.svg, Gen A)      #
# =========================================================================== #

def _twoline(f: Figure, cx, cy, line1, line2, *, accent=None, min_w=120.0):
    """A block with a name line and a symbol line (width = min_w exactly,
    so callers can do edge arithmetic)."""
    w, h = min_w, 54.0
    a = accent
    f.rect(cx - w / 2, cy - h / 2, w, h,
           fill=a.tint if a else T.PAPER,
           stroke=a.base if a else T.INK, sw=T.SW_BOX, r=T.R_BLOCK)
    f.text(cx, cy - 11, line1, size=T.FS_SMALL, color=T.INK)
    f.text(cx, cy + 13, line2, size=T.FS_SMALL, color=T.INK)
    return w, h


def fig_seq2seq_attention():
    """Topology matches img/seq2seq-attention.svg exactly:
    Et -> encoder -> {hidden state Ht, outputs E't};  Ht -> decoder (init);
    E't -> key/value -> attention;  H't (decoder state) -> query -> attention;
    attention -> context -> decoder;  Dt -> decoder -> {H't, outputs D't}.
    """
    f = Figure()
    ENC, ATT, CTX, DEC = 95.0, 355.0, 510.0, 690.0
    HW = 112.0  # two-line box width

    # --- decoder column (top-right; decoder sits high, like the original) --
    f.block(DEC, 110, "decoder", role="stream", min_w=220)
    f.arrow(DEC - 60, 87, DEC - 60, 61, stroke=T.INK)
    f.arrow(DEC + 60, 87, DEC + 60, 61, stroke=T.INK)
    _twoline(f, DEC - 60, 30, "hidden state",
             [var("H′"), sub("t")], accent=T.TEAL, min_w=HW)
    _twoline(f, DEC + 60, 30, "outputs",
             [var("D′"), sub("t")], accent=None, min_w=HW)

    # --- encoder column (bottom-left) --------------------------------------
    f.block(ENC, 326, [Span("E", "i"), sub("t")], min_w=52)
    f.arrow(ENC, 303, ENC, 274, stroke=T.INK)
    f.block(ENC, 252, "encoder", role="stream", min_w=220)
    f.arrow(ENC - 60, 229, ENC - 60, 201, stroke=T.INK)
    f.arrow(ENC + 60, 229, ENC + 60, 201, stroke=T.INK)
    _twoline(f, ENC - 60, 170, "hidden state",
             [var("H"), sub("t")], accent=T.TEAL, min_w=HW)
    _twoline(f, ENC + 60, 170, "outputs",
             [var("E′"), sub("t")], accent=None, min_w=HW)

    # encoder hidden state initializes the decoder: up, then right into it
    f.ortho_arrow([(ENC - 60, 141), (ENC - 60, 110), (DEC - 114, 110)],
                  stroke=T.INK)

    # --- attention row ------------------------------------------------------
    f.block(ATT, 170, "attention", role="attention", min_w=130)
    kx0, kx1 = ENC + 60 + HW / 2, ATT - 65
    f.arrow(kx0, 159, kx1, 159, stroke=T.INK)
    f.arrow(kx0, 181, kx1, 181, stroke=T.INK)
    f.text((kx0 + kx1) / 2, 144, "key", size=T.FS_TINY, color=T.MUTED)
    f.text((kx0 + kx1) / 2, 197, "value", size=T.FS_TINY, color=T.MUTED)

    # decoder hidden state issues the query: left, then down into attention
    f.ortho_arrow([(DEC - 60 - HW / 2, 30), (ATT, 30), (ATT, 146)],
                  stroke=T.INK)
    f.text((DEC - 60 - HW / 2 + ATT) / 2, 16, "query", size=T.FS_TINY,
           color=T.MUTED)

    # attention -> context -> up into the decoder
    f.arrow(ATT + 65, 170, CTX - 52, 170, stroke=T.INK)
    f.block(CTX, 170, "context", min_w=100)
    f.ortho_arrow([(CTX + 52, 170), (DEC, 170), (DEC, 133)], stroke=T.INK)

    # decoder token input, from below (long riser right of the context box)
    f.block(DEC + 60, 326, [Span("D", "i"), sub("t")], min_w=52)
    f.arrow(DEC + 60, 303, DEC + 60, 133, stroke=T.INK)

    f.save("seq2seq-attention", out_dir=OUT,
           desc="Sequence-to-sequence model with attention.")


# =========================================================================== #
# 4. Vector addition (matplotlib theme; remake of img/mdl-la-vector-add.svg)  #
# =========================================================================== #

def fig_vector_add():
    import matplotlib
    matplotlib.use("svg")
    import matplotlib.pyplot as plt
    import numpy as np

    from figstyle.mpl import arrow, axis_cross, clean_axes, save, use_style

    use_style(hashsalt="figstyle-demo")
    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    u = np.array([3.0, 1.0])
    v = np.array([1.0, 2.0])
    LBL = 18
    axis_cross(ax, (-0.4, 4.6), (-0.35, 3.15))
    # tip-to-tail construction ghosts (structure = neutral, dashed)
    ax.plot([0, v[0]], [0, v[1]], ls=(0, (4, 3)), color=T.FAINT, lw=T.SW_HAIR)
    ax.plot([v[0], u[0] + v[0]], [v[1], u[1] + v[1]], ls=(0, (4, 3)),
            color=T.FAINT, lw=T.SW_HAIR)
    arrow(ax, (0, 0), u, color=T.BLUE.base, lw=2.6)
    arrow(ax, u, u + v, color=T.ORANGE.base, lw=2.6)
    arrow(ax, (0, 0), u + v, color=T.GREEN.base, lw=3.2)
    ax.text(1.5, 0.24, r"$\mathbf{u}$", color=T.BLUE.dark, fontsize=LBL)
    ax.text(4.08, 1.85, r"$\mathbf{v}$", color=T.ORANGE.dark, fontsize=LBL,
            ha="left")
    ax.text(1.55, 1.85, r"$\mathbf{u}+\mathbf{v}$", color=T.GREEN.dark,
            fontsize=LBL)
    clean_axes(ax, lim=((-0.5, 4.7), (-0.45, 3.25)), hide=True)
    save(fig, "mdl-la-vector-add", out_dir=OUT)


# =========================================================================== #
# 5. Token specimen sheet (visual reference card for the style guide)         #
# =========================================================================== #

def fig_specimen():
    f = Figure()

    # -- accent trios ------------------------------------------------------ #
    f.text(0, -6, "ACCENT TRIOS", size=T.FS_TINY, color=T.MUTED,
           anchor="start", tracking=T.LETTERSPACE_CAPS)
    order = ["blue", "orange", "green", "purple", "red", "teal", "gold",
             "slate"]
    roles = {"blue": "stream", "orange": "attention", "green": "ffn",
             "purple": "embed", "red": "grad", "teal": "state",
             "gold": "highlight", "slate": "norm"}
    for i, name in enumerate(order):
        a = T.ACCENTS[name]
        x = i * 102
        f.rect(x, 12, 86, 40, fill=a.tint, stroke=a.base, sw=T.SW_BOX,
               r=T.R_BLOCK)
        f.text(x + 43, 32, name, size=T.FS_SMALL, color=T.INK)
        f.text(x + 43, 66, roles[name], size=T.FS_TINY, color=a.dark)

    # -- type scale ---------------------------------------------------------#
    y = 112
    f.text(0, y, "TYPE SCALE", size=T.FS_TINY, color=T.MUTED,
           anchor="start", tracking=T.LETTERSPACE_CAPS)
    specs = [(T.FS_TITLE, "r", T.INK, "Title 19 — panel titles"),
             (T.FS_LABEL, "r", T.INK, "Label 17 — block and axis labels"),
             (T.FS_SMALL, "r", T.MUTED, "Small 14.5 — secondary annotations"),
             (T.FS_TINY, "r", T.MUTED, "TINY 13 CAPS — GROUP LABELS")]
    for j, (sz, kind, col, txt) in enumerate(specs):
        f.text(0, y + 26 + j * 30, [Span(txt, kind)], size=sz, color=col,
               anchor="start",
               tracking=T.LETTERSPACE_CAPS if sz == T.FS_TINY else 0.0)

    # -- stroke ladder + line semantics ------------------------------------ #
    x0 = 430
    f.text(x0, y, "STROKES & LINES", size=T.FS_TINY, color=T.MUTED,
           anchor="start", tracking=T.LETTERSPACE_CAPS)
    rows = [("hair 1.5", dict(sw=T.SW_HAIR)),
            ("box 2.0", dict(sw=T.SW_BOX)),
            ("line 2.5", dict(sw=T.SW_LINE)),
            ("heavy 4.0", dict(sw=T.SW_HEAVY, stroke=T.BLUE.base)),
            ("dashed = optional", dict(sw=T.SW_LINE, dash=T.DASH_SOFT))]
    for j, (lab, kw) in enumerate(rows):
        yy = y + 26 + j * 24
        f.line(x0, yy, x0 + 140, yy, **{"stroke": T.INK, **kw})
        f.text(x0 + 154, yy, lab, size=T.FS_SMALL, color=T.MUTED,
               anchor="start")

    # -- arrow grammar ------------------------------------------------------#
    yy = y + 26 + 5 * 24 + 6
    f.arrow(x0, yy, x0 + 140, yy, stroke=T.INK)
    f.text(x0 + 154, yy, "data flow", size=T.FS_SMALL, color=T.MUTED,
           anchor="start")
    yy += 26
    f.leader(x0, yy, x0 + 140, yy)
    f.text(x0 + 154, yy, "annotation leader", size=T.FS_SMALL, color=T.MUTED,
           anchor="start")

    f.save("specimen", out_dir=OUT,
           desc="figstyle design-token specimen sheet.")


def main():
    os.makedirs(OUT, exist_ok=True)
    fig_mlp()
    fig_block_anatomy()
    fig_seq2seq_attention()
    fig_vector_add()
    fig_specimen()
    print("wrote 5 figures to", OUT)


if __name__ == "__main__":
    main()
