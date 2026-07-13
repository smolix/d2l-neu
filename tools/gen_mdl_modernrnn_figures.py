#!/usr/bin/env python3
"""Generate the illustrative figures for the "Modern Recurrent Neural
Networks" chapter (``chapter_recurrent-modern``) in the one shared house style
defined in ``gen_mdl_figures.py``.

Thirteen figures across the chapter's four sections:

  * 10.1 "Gated Recurrent Networks" (sec_lstm) -- four restyled carryovers,
    written under new house-style names so the old hand-drawn SVGs
    (``img/lstm-0..3.svg``, ``img/gru-1..3.svg``, ``img/deep-rnn.svg``,
    ``img/birnn.svg``) are left untouched: the LSTM memory cell, the GRU
    cell, a two-layer deep RNN unrolled over three steps, and a
    bidirectional RNN.
  * 10.2 "Encoder-Decoder Models for Sequence Transduction" (sec_encoder-
    decoder) -- the encoder-decoder interface, and the unrolled RNN
    encoder-decoder with teacher forcing (kept because the attention
    chapter references it).
  * 10.3 "Linear Recurrence and State Space Models" (sec_ssm) -- the
    parallel prefix (doubling) scan, the three equivalent views of a linear
    state space model (continuous-time ODE, unrolled recurrence,
    convolution), a computed HiPPO-LegS online compression/reconstruction,
    and the S4D residual block.
  * 10.4 "Selective State Space Models" (sec_mamba) -- the selective
    copying task, the Mamba block, and the chapter's "three answers"
    summary strip (gate it / linearize it / select it).

The notebooks / prose reference the generated files with no drawing code (like
the slide SVGs).  Figures that show a *computed* result (e.g. the HiPPO
reconstruction, the SSM step response) use real numerical computation so the
pictures are exact, not sketches; purely schematic figures (cell wiring,
unrolled-graph diagrams, block diagrams, ...) use ``set_aspect("equal")`` and
the shared drawing helpers (``arrow``/``vlabel``/``clean_axes``/``axis_cross``/
``right_angle``).

Run with the repo's pytorch venv (matplotlib + numpy are available):

    .venv-pytorch/bin/python tools/gen_mdl_modernrnn_figures.py

All figures are written to ``img/mdl-modernrnn-<id>.svg``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT
PURPLE = "#9467bd"

from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


# =========================================================================== #
# 10.1 "Gated Recurrent Networks" (sec_lstm): the LSTM cell, the GRU cell, a  #
# two-layer deep RNN, and a bidirectional RNN.                               #
#                                                                              #
# Four restyled carryovers, written under new house-style names so the old   #
# hand-drawn SVGs (``img/lstm-0..3.svg``, ``img/gru-1..3.svg``,              #
# ``img/deep-rnn.svg``, ``img/birnn.svg``) are left untouched.               #
#                                                                              #
# Colour code (consistent across the four figures and with the ch. 9         #
# diagrams): inputs X blue, hidden/cell state boxes green/orange, computation#
# heads and elementwise ops grey, outputs grey-boxed.                        #
# =========================================================================== #

def _box(ax, cx, cy, w, h, text, color, fontsize=14, text_color="black",
         weight="bold", lw=1.8, ls="-"):
    """A rounded state box (faint fill + solid coloured edge), centred text."""
    x, y = cx - w / 2, cy - h / 2
    for fc, a in [(color, 0.12), ("none", 1.0)]:
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=lw, edgecolor=color, facecolor=fc, alpha=a,
            linestyle=ls))
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color=text_color)


def _head(ax, cx, cy, text, w=0.95, h=0.72, fontsize=13):
    """A computation head (sigma / tanh layer): grey rounded box."""
    _box(ax, cx, cy, w, h, text, GRAY, fontsize=fontsize, weight="normal")


def _op(ax, cx, cy, text, r=0.26, fontsize=13):
    """An elementwise operator node: small circle with a symbol."""
    ax.add_patch(Circle((cx, cy), r, facecolor="white", edgecolor=GRAY,
                        linewidth=1.6, zorder=4))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            color="black", zorder=5)


def _dot(ax, cx, cy, r=0.06):
    """A branch point on a wire."""
    ax.add_patch(Circle((cx, cy), r, facecolor=GRAY, edgecolor="none",
                        zorder=4))


def fig_lstm_cell():
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    ax.set_aspect("equal")
    ax.axis("off")

    y_cell, y_head, y_bus = 5.05, 2.75, 1.15
    xF, xI, xC, xO = 3.0, 4.5, 5.7, 7.6   # head x-positions
    x_prod = (xI + xC) / 2                # I * C~ product column
    x_out = 8.75                          # output-gate column
    bw, bh = 1.25, 0.85                   # state-box size
    hw, hh = 0.82, 0.72                   # head-box size
    r = 0.26                              # op-circle radius

    # --- cell-state conveyor (top) ------------------------------------------
    _box(ax, 1.05, y_cell, bw, bh, r"$\mathbf{C}_{t-1}$", ORANGE)
    _op(ax, xF, y_cell, r"$\odot$")
    _op(ax, x_prod, y_cell, r"$+$")
    _box(ax, 9.9, y_cell, bw, bh, r"$\mathbf{C}_{t}$", ORANGE)
    fl.arrow(ax, (1.05 + bw / 2, y_cell), (xF - r, y_cell), color=GRAY)
    fl.arrow(ax, (xF + r, y_cell), (x_prod - r, y_cell), color=GRAY)
    fl.arrow(ax, (x_prod + r, y_cell), (9.9 - bw / 2, y_cell), color=GRAY)

    # --- input bus (bottom): H_{t-1} and X_t feed all four heads -------------
    _box(ax, 1.05, 1.85, bw, bh, r"$\mathbf{H}_{t-1}$", GREEN)
    _box(ax, 1.05, 0.55, bw, bh, r"$\mathbf{X}_{t}$", BLUE)
    xj = 2.15                              # bus junction
    fl.arrow(ax, (1.05 + bw / 2, 1.85), (xj - 0.03, y_bus + 0.04), color=GRAY,
             lw=1.8, mut=12)
    fl.arrow(ax, (1.05 + bw / 2, 0.55), (xj - 0.03, y_bus - 0.04), color=GRAY,
             lw=1.8, mut=12)
    ax.plot([xj, xO], [y_bus, y_bus], color=GRAY, lw=1.8, zorder=1)
    _dot(ax, xj, y_bus)
    for x in (xF, xI, xC, xO):
        if x != xO:
            _dot(ax, x, y_bus)
        fl.arrow(ax, (x, y_bus), (x, y_head - hh / 2), color=GRAY, lw=1.8,
                 mut=12)

    # --- the four heads -------------------------------------------------------
    _head(ax, xF, y_head, r"$\sigma$", w=hw, h=hh)
    _head(ax, xI, y_head, r"$\sigma$", w=hw, h=hh)
    _head(ax, xC, y_head, r"$\tanh$", w=hw + 0.28, h=hh, fontsize=12)
    _head(ax, xO, y_head, r"$\sigma$", w=hw, h=hh)

    # forget gate: straight up into the conveyor product
    fl.arrow(ax, (xF, y_head + hh / 2), (xF, y_cell - r), color=GRAY)
    ax.text(xF - 0.18, (y_head + y_cell) / 2 + 0.12, r"$\mathbf{F}_t$",
            ha="right", va="center", fontsize=14, color="black")

    # input gate and input node: join at an elementwise product, then up into +
    y_ic = 3.95
    _op(ax, x_prod, y_ic, r"$\odot$")
    fl.arrow(ax, (xI, y_head + hh / 2), (x_prod - r * 0.72, y_ic - r * 0.72),
             color=GRAY, lw=1.8, mut=12)
    fl.arrow(ax, (xC, y_head + hh / 2), (x_prod + r * 0.72, y_ic - r * 0.72),
             color=GRAY, lw=1.8, mut=12)
    fl.arrow(ax, (x_prod, y_ic + r), (x_prod, y_cell - r), color=GRAY)
    ax.text(xI - 0.42, y_head + hh / 2 + 0.34, r"$\mathbf{I}_t$", ha="center",
            va="center", fontsize=14, color="black")
    ax.text(xC + 0.52, y_head + hh / 2 + 0.34, r"$\tilde{\mathbf{C}}_t$",
            ha="center", va="center", fontsize=14, color="black")

    # output branch: conveyor -> tanh -> (x) with output gate -> H_t
    y_tanh = 3.95
    _dot(ax, x_out, y_cell)
    _head(ax, x_out, y_tanh, r"$\tanh$", w=hw + 0.28, h=0.62, fontsize=12)
    ax.plot([x_out, x_out], [y_cell, y_tanh + 0.31], color=GRAY, lw=1.8,
            zorder=1)
    _op(ax, x_out, y_head, r"$\odot$")
    fl.arrow(ax, (x_out, y_tanh - 0.31), (x_out, y_head + r), color=GRAY,
             lw=1.8, mut=12)
    fl.arrow(ax, (xO + hw / 2, y_head), (x_out - r, y_head), color=GRAY,
             lw=1.8, mut=12)
    ax.text((xO + hw / 2 + x_out - r) / 2, y_head - 0.46, r"$\mathbf{O}_t$",
            ha="center", va="center", fontsize=14, color="black")
    fl.arrow(ax, (x_out + r, y_head), (9.9 - bw / 2, y_head), color=GRAY)
    _box(ax, 9.9, y_head, bw, bh, r"$\mathbf{H}_{t}$", GREEN)

    ax.set_xlim(0.2, 10.75)
    ax.set_ylim(0.0, 5.7)
    fl.save(fig, "mdl-modernrnn-lstm-cell")


def fig_gru_cell():
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    ax.set_aspect("equal")
    ax.axis("off")

    y_conv, y_head, y_bus = 4.85, 2.45, 0.95
    xR, xZ, xH = 2.75, 4.15, 6.05          # head x-positions
    x_blend = 7.15                          # (1 - Z) * candidate column
    bw, bh = 1.25, 0.85
    hw, hh = 0.82, 0.72
    r = 0.26

    # --- state conveyor (top): H_{t-1} -> (x Z) -> (+) -> H_t -----------------
    _box(ax, 1.0, y_conv, bw, bh, r"$\mathbf{H}_{t-1}$", GREEN)
    _op(ax, xZ, y_conv, r"$\odot$")
    _op(ax, x_blend, y_conv, r"$+$")
    _box(ax, 9.55, y_conv, bw, bh, r"$\mathbf{H}_{t}$", GREEN)
    fl.arrow(ax, (1.0 + bw / 2, y_conv), (xZ - r, y_conv), color=GRAY)
    fl.arrow(ax, (xZ + r, y_conv), (x_blend - r, y_conv), color=GRAY)
    fl.arrow(ax, (x_blend + r, y_conv), (9.55 - bw / 2, y_conv), color=GRAY)

    # --- input bus: X_t plus a tap of H_{t-1} feed the gate heads -------------
    _box(ax, 1.0, 0.95, bw, bh, r"$\mathbf{X}_{t}$", BLUE)
    xj = 1.0 + bw / 2
    ax.plot([xj, xH], [y_bus, y_bus], color=GRAY, lw=1.8, zorder=1)
    # H_{t-1} tap down to the bus (gates read the previous state too)
    x_tap = 1.95
    _dot(ax, x_tap, y_conv)
    ax.plot([x_tap, x_tap], [y_conv, y_bus], color=GRAY, lw=1.8, zorder=1)
    _dot(ax, x_tap, y_bus)
    for x in (xR, xZ, xH):
        if x != xH:
            _dot(ax, x, y_bus)
        fl.arrow(ax, (x, y_bus), (x, y_head - hh / 2), color=GRAY, lw=1.8,
                 mut=12)

    # --- heads ----------------------------------------------------------------
    _head(ax, xR, y_head, r"$\sigma$", w=hw, h=hh)
    _head(ax, xZ, y_head, r"$\sigma$", w=hw, h=hh)
    _head(ax, xH, y_head, r"$\tanh$", w=hw + 0.28, h=hh, fontsize=12)

    # reset gate: filters the previous state before it enters the candidate
    y_rH = 3.6
    _op(ax, xR, y_rH, r"$\odot$")
    fl.arrow(ax, (xR, y_head + hh / 2), (xR, y_rH - r), color=GRAY, lw=1.8,
             mut=12)
    ax.text(xR - 0.18, (y_head + y_rH) / 2 - 0.02, r"$\mathbf{R}_t$",
            ha="right", va="center", fontsize=14, color="black")
    # H_{t-1} into the reset product from the conveyor above
    fl.arrow(ax, (x_tap, y_conv - 0.0), (xR - r * 0.72, y_rH + r * 0.72),
             color=GRAY, lw=1.8, mut=12)
    # gated state into the candidate head (one clean perpendicular wire
    # crossing with the update-gate riser; unavoidable in a planar layout)
    fl.arrow(ax, (xR + r * 0.72, y_rH - r * 0.72),
             (xH - (hw + 0.28) / 2, y_head + hh / 2 - 0.06), color=GRAY,
             lw=1.8, mut=12)

    # update gate up into the conveyor product, with a branch carrying the
    # complement (1 - Z) into the blend product
    fl.arrow(ax, (xZ, y_head + hh / 2), (xZ, y_conv - r), color=GRAY)
    ax.text(xZ - 0.18, 3.9, r"$\mathbf{Z}_t$", ha="right", va="center",
            fontsize=14, color="black")
    y_c = 3.6
    _op(ax, x_blend, y_c, r"$\odot$")
    _dot(ax, xZ, 4.2)
    fl.arrow(ax, (xZ, 4.2), (x_blend - r, y_c + 0.04), color=GRAY,
             lw=1.8, mut=12)
    ax.text(5.4, 4.14, r"$1-\mathbf{Z}_t$", ha="center", va="center",
            fontsize=13, color="black")

    # candidate into the blend product, then up into the convex combination
    fl.arrow(ax, (xH + 0.4, y_head + hh / 2),
             (x_blend - r * 0.6, y_c - r * 0.8), color=GRAY, lw=1.8, mut=12)
    fl.arrow(ax, (x_blend, y_c + r), (x_blend, y_conv - r), color=GRAY,
             lw=1.8, mut=12)
    ax.text(7.35, 2.95, r"$\tilde{\mathbf{H}}_t$", ha="left", va="center",
            fontsize=14, color="black")

    ax.set_xlim(0.15, 10.35)
    ax.set_ylim(0.3, 5.5)
    fl.save(fig, "mdl-modernrnn-gru-cell")


def fig_deep_rnn():
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    ax.set_aspect("equal")
    ax.axis("off")

    xs = [2.6, 5.4, 8.2]                     # three time steps
    y_in, y_h1, y_h2, y_out = 0.9, 2.9, 4.9, 6.9
    bw, bh = 1.35, 0.9
    subs = ["t-1", "t", "t+1"]
    half = bh / 2

    ax.text(0.3, y_out, "outputs", ha="left", va="center", fontsize=13,
            color="black")
    ax.text(0.3, y_h2, "layer 2", ha="left", va="center", fontsize=13,
            color="black")
    ax.text(0.3, y_h1, "layer 1", ha="left", va="center", fontsize=13,
            color="black")
    ax.text(0.3, y_in, "inputs", ha="left", va="center", fontsize=13,
            color="black")

    for cx, s in zip(xs, subs):
        _box(ax, cx, y_in, bw, bh, rf"$\mathbf{{X}}_{{{s}}}$", BLUE)
        _box(ax, cx, y_h1, bw, bh, rf"$\mathbf{{H}}_{{{s}}}^{{(1)}}$", GREEN)
        _box(ax, cx, y_h2, bw, bh, rf"$\mathbf{{H}}_{{{s}}}^{{(2)}}$", GREEN)
        _box(ax, cx, y_out, bw, bh, rf"$\mathbf{{O}}_{{{s}}}$", ORANGE)
        for y0, y1 in [(y_in, y_h1), (y_h1, y_h2), (y_h2, y_out)]:
            fl.arrow(ax, (cx, y0 + half), (cx, y1 - half), color=GRAY, lw=2.0,
                     mut=15)

    # recurrent links within each hidden layer
    for y in (y_h1, y_h2):
        for cx0, cx1 in zip(xs[:-1], xs[1:]):
            fl.arrow(ax, (cx0 + bw / 2, y), (cx1 - bw / 2, y), color=GRAY,
                     lw=2.0, mut=15)
        fl.arrow(ax, (xs[0] - 1.0, y), (xs[0] - bw / 2, y), color=GRAY,
                 lw=1.6, ls="--", mut=13)
        fl.arrow(ax, (xs[-1] + bw / 2, y), (xs[-1] + 1.35, y), color=GRAY,
                 lw=1.6, ls="--", mut=13)

    ax.set_xlim(0.1, 9.8)
    ax.set_ylim(0.2, 7.6)
    fl.save(fig, "mdl-modernrnn-deep-rnn")


def fig_bi_rnn():
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    ax.set_aspect("equal")
    ax.axis("off")

    xs = [2.6, 5.4, 8.2]                     # three time steps
    dx = 1.05                                # backward row horizontal offset
    y_in, y_f, y_b, y_out = 0.9, 2.9, 4.7, 6.7
    bw, bh = 1.3, 0.9
    subs = ["t-1", "t", "t+1"]
    half = bh / 2

    ax.text(0.05, y_out, "outputs", ha="left", va="center", fontsize=13,
            color="black")
    ax.text(0.05, y_b, "backward", ha="left", va="center", fontsize=13,
            color="black")
    ax.text(0.05, y_f, "forward", ha="left", va="center", fontsize=13,
            color="black")
    ax.text(0.05, y_in, "inputs", ha="left", va="center", fontsize=13,
            color="black")

    for cx, s in zip(xs, subs):
        _box(ax, cx, y_in, bw, bh, rf"$\mathbf{{X}}_{{{s}}}$", BLUE)
        _box(ax, cx, y_f, bw, bh,
             rf"$\overrightarrow{{\mathbf{{H}}}}_{{{s}}}$", GREEN)
        _box(ax, cx + dx, y_b, bw, bh,
             rf"$\overleftarrow{{\mathbf{{H}}}}_{{{s}}}$", ORANGE)
        _box(ax, cx + dx / 2, y_out, bw, bh, rf"$\mathbf{{O}}_{{{s}}}$", GRAY)
        # input feeds both directions
        fl.arrow(ax, (cx, y_in + half), (cx, y_f - half), color=GRAY, lw=2.0,
                 mut=15)
        fl.arrow(ax, (cx + 0.55, y_in + half), (cx + dx, y_b - half),
                 color=GRAY, lw=1.8, mut=13)
        # both directions feed the concatenated output
        fl.arrow(ax, (cx, y_f + half), (cx + dx / 2 - 0.12, y_out - half),
                 color=GRAY, lw=1.8, mut=13)
        fl.arrow(ax, (cx + dx, y_b + half), (cx + dx / 2 + 0.12, y_out - half),
                 color=GRAY, lw=1.8, mut=13)

    # forward chain runs left-to-right, backward chain right-to-left
    for cx0, cx1 in zip(xs[:-1], xs[1:]):
        fl.arrow(ax, (cx0 + bw / 2, y_f), (cx1 - bw / 2, y_f), color=GRAY,
                 lw=2.0, mut=15)
        fl.arrow(ax, (cx1 + dx - bw / 2, y_b), (cx0 + dx + bw / 2, y_b),
                 color=GRAY, lw=2.0, mut=15)
    fl.arrow(ax, (xs[0] - 1.3, y_f), (xs[0] - bw / 2, y_f), color=GRAY,
             lw=1.6, ls="--", mut=13)
    fl.arrow(ax, (xs[-1] + bw / 2, y_f), (xs[-1] + 1.3, y_f), color=GRAY,
             lw=1.6, ls="--", mut=13)
    fl.arrow(ax, (xs[-1] + dx + 1.3, y_b), (xs[-1] + dx + bw / 2, y_b),
             color=GRAY, lw=1.6, ls="--", mut=13)
    fl.arrow(ax, (xs[0] + dx - bw / 2, y_b), (xs[0] + dx - 1.3, y_b),
             color=GRAY, lw=1.6, ls="--", mut=13)

    ax.set_xlim(-0.1, 10.75)
    ax.set_ylim(0.2, 7.4)
    fl.save(fig, "mdl-modernrnn-bi-rnn")


# =========================================================================== #
# 10.2 "Encoder-Decoder Models for Sequence Transduction" (sec_encoder-      #
# decoder): the encoder-decoder interface, and the unrolled RNN encoder-     #
# decoder with teacher forcing (kept because the attention chapter           #
# references it).                                                            #
#                                                                              #
# ``_ed_box``/``_ed_arrow`` are this section's own box/arrow helpers -- they #
# take separate face/edge colours and an explicit zorder respectively, both  #
# unlike the ``_box``/``fl.arrow`` used by 10.1/10.3/10.4, so they keep      #
# distinct, section-prefixed names rather than colliding on ``_box``.        #
# =========================================================================== #

_ED_TXT = 14          # in-figure label size
_ED_SMALL = 12        # token labels


def _ed_box(ax, cx, cy, w, h, label, face, edge, fontsize=_ED_TXT,
            text_color="black"):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.6, edgecolor=edge, facecolor=face, zorder=2))
    ax.text(cx, cy, label, ha="center", va="center", color=text_color,
            fontsize=fontsize, zorder=3)


def _ed_arrow(ax, tail, tip, color="black", lw=1.8, mut=13):
    ax.annotate("", xy=tip, xytext=tail,
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                shrinkA=0, shrinkB=0, mutation_scale=mut),
                zorder=1)


def fig_encoder_decoder():
    """The encoder-decoder interface: source -> encoder -> state -> decoder ->
    target.  A single horizontal pipeline, one arrow per hand-off."""
    fig, ax = plt.subplots(figsize=(8.2, 2.5))
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 3.0)
    ax.axis("off")
    ax.set_aspect("equal")

    y = 1.5
    # source text
    ax.text(0.75, y, '"They are\nwatching ."', ha="center", va="center",
            fontsize=_ED_SMALL, color="black")
    _ed_box(ax, 3.1, y, 1.9, 1.15, "Encoder", "#eaf2fb", BLUE)
    _ed_box(ax, 7.1, y, 1.9, 1.15, "Decoder", "#fdefe1", ORANGE)
    ax.text(9.75, y, '"Ils\nregardent ."', ha="center", va="center",
            fontsize=_ED_SMALL, color="black")

    _ed_arrow(ax, (1.55, y), (2.15, y))
    _ed_arrow(ax, (4.05, y), (6.15, y))
    _ed_arrow(ax, (8.05, y), (9.05, y))

    # the fixed-shape state on the middle hand-off
    ax.text(5.1, y + 0.5, "state", ha="center", va="bottom",
            fontsize=_ED_SMALL, color=GREEN, fontstyle="italic")
    ax.plot(5.1, y, "o", color=GREEN, ms=10, zorder=4)

    fl.save(fig, "mdl-modernrnn-encoder-decoder")


def fig_seq2seq():
    """Unrolled RNN encoder-decoder with teacher forcing.  The encoder reads the
    source tokens and passes its final state as context to every decoder step;
    the decoder is fed the shifted target (`<bos>` ...) and predicts the target
    shifted left (... `<eos>`)."""
    fig, ax = plt.subplots(figsize=(10.4, 3.9))
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    ax.set_aspect("equal")

    cw, ch = 1.05, 0.8          # cell size
    yc = 2.4                    # recurrent-cell row
    dx = 1.45

    enc_tokens = ['"They"', '"are"', '"watching"', '"."']
    enc_x = [0.9 + i * dx for i in range(4)]
    dec_in = ['<bos>', '"Ils"', '"regardent"', '"."']
    dec_out = ['"Ils"', '"regardent"', '"."', '<eos>']
    dec_x = [enc_x[-1] + dx + 0.6 + i * dx for i in range(4)]
    ax.set_xlim(0, dec_x[-1] + 0.9)

    # dashed separator between encoder and decoder
    xsep = (enc_x[-1] + dec_x[0]) / 2
    ax.plot([xsep, xsep], [0.5, 4.8], ls=(0, (4, 4)), color=LIGHT, lw=1.2)
    ax.text((enc_x[1] + enc_x[2]) / 2, 4.75, "Encoder", ha="center",
            fontsize=_ED_TXT, color=BLUE)
    ax.text((dec_x[1] + dec_x[2]) / 2, 4.75, "Decoder", ha="center",
            fontsize=_ED_TXT, color=ORANGE)

    # encoder cells + inputs, left-to-right recurrence
    for i, x in enumerate(enc_x):
        _ed_box(ax, x, yc, cw, ch, "", "#eaf2fb", BLUE)
        _ed_arrow(ax, (x, 1.55), (x, yc - ch / 2))          # token -> cell
        ax.text(x, 1.35, enc_tokens[i], ha="center", va="top",
                fontsize=_ED_SMALL, color="black")
        if i > 0:
            _ed_arrow(ax, (enc_x[i - 1] + cw / 2, yc), (x - cw / 2, yc))

    # context state c = encoder final state
    xc = xsep
    ax.plot(xc, yc, "o", color=GREEN, ms=11, zorder=5)
    ax.text(xc, yc + 0.42, r"$\mathbf{c}$", ha="center", va="bottom",
            fontsize=_ED_TXT + 1, color=GREEN)
    _ed_arrow(ax, (enc_x[-1] + cw / 2, yc), (xc - 0.16, yc), color=GREEN)

    # decoder cells: teacher-forced input below, prediction above, context in
    for i, x in enumerate(dec_x):
        _ed_box(ax, x, yc, cw, ch, "", "#fdefe1", ORANGE)
        # teacher-forcing input from below
        _ed_arrow(ax, (x, 1.55), (x, yc - ch / 2))
        ax.text(x, 1.35, dec_in[i], ha="center", va="top",
                fontsize=_ED_SMALL, color="black")
        # prediction above
        _ed_arrow(ax, (x, yc + ch / 2), (x, 3.55))
        ax.text(x, 3.75, dec_out[i], ha="center", va="bottom",
                fontsize=_ED_SMALL, color="black")
        # recurrent link
        if i == 0:
            _ed_arrow(ax, (xc + 0.16, yc), (x - cw / 2, yc), color=GREEN)
        else:
            _ed_arrow(ax, (dec_x[i - 1] + cw / 2, yc), (x - cw / 2, yc))
        # context fed at every decoder step (green, from the state)
        ax.annotate("", xy=(x, yc + ch / 2 - 0.02),
                    xytext=(xc, yc + 0.30),
                    arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.1,
                                    ls=(0, (3, 3)), shrinkA=2, shrinkB=2,
                                    connectionstyle="arc3,rad=-0.12"),
                    zorder=1)

    fl.save(fig, "mdl-modernrnn-seq2seq")


# =========================================================================== #
# 10.3 "Linear Recurrence and State Space Models" (sec_ssm): the parallel    #
# prefix (doubling) scan, the three equivalent views of a linear SSM         #
# (continuous-time ODE, unrolled recurrence, convolution with the            #
# materialized kernel), a computed HiPPO-LegS online compression /           #
# reconstruction, and the S4D residual block.                                #
# =========================================================================== #

def _hippo_legs(N):
    A = np.zeros((N, N))
    for n in range(N):
        for k in range(N):
            if n > k:
                A[n, k] = -np.sqrt((2 * n + 1) * (2 * k + 1))
            elif n == k:
                A[n, k] = -(n + 1)
    B = np.sqrt(2 * np.arange(N) + 1.0)
    return A, B


def fig_scan_tree():
    T = 8
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.set_aspect("equal")
    ax.axis("off")

    dx, dy = 1.5, 1.55                     # column / row spacing
    xs = [1.9 + t * dx for t in range(T)]
    rows = [0.7, 0.7 + dy, 0.7 + 2 * dy, 0.7 + 3 * dy]
    bw, bh = 1.05, 0.66
    half = bh / 2

    def span_label(lo, hi):
        return rf"$({lo}{{:}}{hi})$" if lo != hi else rf"$({lo})$"

    # row 0: inputs; row k: after combining with stride 2^(k-1)
    strides = [1, 2, 4]
    row_names = ["inputs", "stride 1", "stride 2", "stride 4"]
    for name, y in zip(row_names, rows):
        ax.text(0.05, y, name, ha="left", va="center", fontsize=13,
                color="black")

    for r, y in enumerate(rows):
        lo_of = lambda t: max(1, t + 1 - (2 ** r - 1))
        for t in range(T):
            lo = lo_of(t)
            done = lo == 1
            color = BLUE if r == 0 else (GREEN if done else ORANGE)
            _box(ax, xs[t], y, bw, bh, span_label(lo, t + 1), color,
                 fontsize=12, weight="normal")
        if r == 0:
            continue
        s = strides[r - 1]
        for t in range(T):
            # pass-through (t < s): vertical dashed copy
            if t < s:
                fl.arrow(ax, (xs[t], rows[r - 1] + half),
                         (xs[t], y - half), color=GRAY, lw=1.2, ls="--",
                         mut=10)
            else:
                fl.arrow(ax, (xs[t], rows[r - 1] + half),
                         (xs[t], y - half), color=GRAY, lw=1.6, mut=11)
                fl.arrow(ax, (xs[t - s] + 0.22, rows[r - 1] + half),
                         (xs[t] - 0.28, y - half), color=BLUE, lw=1.4,
                         alpha=0.75, mut=11)

    ax.set_xlim(-0.1, 13.2)
    ax.set_ylim(0.1, 6.0)
    fl.save(fig, "mdl-modernrnn-scan-tree")


def fig_ssm_views():
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5),
                             gridspec_kw={"width_ratios": [1.0, 1.15, 1.0]})
    ax_ode, ax_rec, ax_conv = axes

    # ---- (a) continuous time: input pulse and first-order state response --
    ax = ax_ode
    t = np.linspace(0, 10, 400)
    u = ((t > 1.2) & (t < 4.2)).astype(float)
    # x' = a x + b u with a = -0.9, b = 0.9 (exact first-order response)
    a_c, b_c = -0.9, 0.9
    x = np.zeros_like(t)
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        x[i] = np.exp(a_c * dt) * x[i - 1] + (np.exp(a_c * dt) - 1) / a_c \
            * b_c * u[i]
    ax.plot(t, u, color=BLUE, lw=1.8)
    ax.plot(t, x, color=GREEN, lw=2.6)
    ax.text(2.7, 1.12, r"input $u(t)$", fontsize=12.5, color=BLUE,
            ha="center", va="bottom")
    ax.text(5.05, 0.82, r"state $x(t)$", fontsize=12.5, color=GREEN,
            ha="left", va="bottom")
    ax.text(7.55, 0.52, r"$\dot{x} = Ax + Bu$", fontsize=14, color="black",
            ha="center")
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.14, 1.34)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("black")

    # ---- (b) recurrence: unrolled chain -----------------------------------
    ax = ax_rec
    ax.set_aspect("equal")
    ax.axis("off")
    xs = [1.1, 3.0, 4.9]
    y_u, y_x, y_y = 0.75, 2.45, 4.15
    bw, bh = 1.1, 0.75
    half = bh / 2
    for cx, s in zip(xs, ["t-1", "t", "t+1"]):
        _box(ax, cx, y_u, bw, bh, rf"$u_{{{s}}}$", BLUE, fontsize=13)
        _box(ax, cx, y_x, bw, bh, rf"$x_{{{s}}}$", GREEN, fontsize=13)
        _box(ax, cx, y_y, bw, bh, rf"$y_{{{s}}}$", ORANGE, fontsize=13)
        fl.arrow(ax, (cx, y_u + half), (cx, y_x - half), color=GRAY, lw=1.8,
                 mut=12)
        fl.arrow(ax, (cx, y_x + half), (cx, y_y - half), color=GRAY, lw=1.8,
                 mut=12)
    for cx0, cx1 in zip(xs[:-1], xs[1:]):
        fl.arrow(ax, (cx0 + bw / 2, y_x), (cx1 - bw / 2, y_x), color=GRAY,
                 lw=1.8, mut=12)
    fl.arrow(ax, (xs[0] - 0.95, y_x), (xs[0] - bw / 2, y_x), color=GRAY,
             lw=1.4, ls="--", mut=10)
    fl.arrow(ax, (xs[-1] + bw / 2, y_x), (xs[-1] + 0.95, y_x), color=GRAY,
             lw=1.4, ls="--", mut=10)
    # one label per weight type, offset from the arrows
    ax.text((xs[0] + xs[1]) / 2, y_x + 0.32, r"$\bar{\mathbf{A}}$",
            fontsize=14, color="black", ha="center")
    ax.text(xs[1] + 0.34, (y_u + y_x) / 2, r"$\bar{\mathbf{B}}$",
            fontsize=14, color="black", ha="left", va="center")
    ax.text(xs[1] + 0.34, (y_x + y_y) / 2, r"$\mathbf{C}$",
            fontsize=14, color="black", ha="left", va="center")
    ax.set_xlim(-0.15, 6.3)
    ax.set_ylim(0.1, 4.9)

    # ---- (c) convolution: flipped kernel over the inputs, one output ------
    ax = ax_conv
    ax.set_aspect("equal")
    ax.axis("off")
    Tc = 7
    xs = [0.65 + 0.82 * i for i in range(Tc)]
    y_u, y_ker, y_out = 0.75, 2.05, 4.35
    bw, bh = 0.62, 0.62
    for i, cx in enumerate(xs):
        _box(ax, cx, y_u, bw, bh, rf"$u_{{{i + 1}}}$", BLUE, fontsize=11,
             weight="normal", lw=1.4)
    # kernel weights K_0.. over the last inputs (aligned under the output)
    kvals = 0.95 * 0.55 ** np.arange(Tc)
    for i in range(Tc):
        k = Tc - 1 - i               # kernel index applied to input i+1
        ax.plot([xs[i], xs[i]], [y_ker, y_ker + 1.15 * kvals[k]],
                color=ORANGE, lw=2.4, solid_capstyle="round")
        ax.plot([xs[i]], [y_ker + 1.15 * kvals[k]], "o", color=ORANGE,
                markersize=4)
        fl.arrow(ax, (xs[i], y_u + bh / 2), (xs[i], y_ker - 0.08),
                 color=GRAY, lw=1.2, mut=9)
    ax.plot([xs[0] - 0.3, xs[-1] + 0.3], [y_ker, y_ker], color="black",
            lw=1.0)
    ax.text(xs[2], y_ker + 0.62, r"$\bar{K}_5$", fontsize=11.5,
            color="black", ha="right", va="center")
    ax.text(xs[-1] + 0.28, y_ker + 1.15 * kvals[0], r"$\bar{K}_0$",
            fontsize=11.5, color="black", ha="left", va="center")
    cx_out = xs[-1]
    _box(ax, cx_out, y_out, 0.85, 0.66, rf"$y_{{{Tc}}}$", GREEN, fontsize=12)
    fl.arrow(ax, (cx_out, y_ker + 1.15 * kvals[0] + 0.16),
             (cx_out, y_out - 0.4), color=GRAY, lw=1.8, mut=12)
    ax.text(0.35, 4.55, r"$\mathbf{y} = \bar{\mathbf{K}} * \mathbf{u}$",
            fontsize=12.5, color="black", ha="left", va="center")
    ax.text(0.35, 3.72,
            r"$\bar{\mathbf{K}} = (\mathbf{C}\bar{\mathbf{B}},"
            r"\mathbf{C}\bar{\mathbf{A}}\bar{\mathbf{B}},"
            r"\mathbf{C}\bar{\mathbf{A}}^2\bar{\mathbf{B}},\ldots)$",
            fontsize=12, color="black", ha="left", va="center")
    ax.set_xlim(-0.1, 6.6)
    ax.set_ylim(0.1, 4.9)

    fig.subplots_adjust(wspace=0.16)
    fl.save(fig, "mdl-modernrnn-ssm-views")


def fig_hippo_reconstruction():
    from numpy.polynomial import legendre

    t = np.linspace(1e-3, 1.0, 2000)
    f = np.sin(8 * t) + 0.5 * np.sin(23 * t) + 0.3 * (t > 0.6)

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.2), sharey=True)
    Ns = [4, 16, 64]
    colors = [ORANGE, BLUE, GREEN]
    dt = t[1] - t[0]
    for ax, N, color in zip(axes, Ns, colors):
        A, B = _hippo_legs(N)
        eye = np.eye(N)
        x = np.zeros(N)
        for i, ti in enumerate(t):       # bilinear step of x' = (Ax + Bf)/t
            At = A * (dt / ti)
            x = np.linalg.solve(eye - At / 2,
                                (eye + At / 2) @ x + (dt / ti) * B * f[i])
        rec = np.zeros_like(t)
        for n in range(N):               # f(s) ~ sum_n x_n sqrt(2n+1) P_n(2s-1)
            cn = np.zeros(n + 1)
            cn[n] = 1
            rec += x[n] * np.sqrt(2 * n + 1) * legendre.legval(2 * t - 1, cn)
        err = (np.sqrt(np.mean((rec - f) ** 2))
               / np.sqrt(np.mean(f ** 2)))
        ax.plot(t, f, color="black", lw=1.3, label="signal $f$")
        ax.plot(t, rec, color=color, lw=2.4, alpha=0.9,
                label="reconstruction")
        ax.text(0.03, 1.92, rf"$N = {N}$, error ${100 * err:.0f}\,\%$",
                fontsize=13, color="black", ha="left")
        ax.set_xlim(0, 1)
        ax.set_ylim(-1.75, 2.45)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_xlabel("$s$", fontsize=13, color="black")
        ax.tick_params(labelsize=11, colors="black")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("black")
    axes[0].set_ylabel("$f(s)$", fontsize=13, color="black")
    axes[0].legend(fontsize=11, loc="lower left", frameon=False)
    fig.subplots_adjust(wspace=0.12)
    fl.save(fig, "mdl-modernrnn-hippo-reconstruction")


def fig_ssm_block():
    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    ax.set_aspect("equal")
    ax.axis("off")

    cx = 4.1                              # main stack x
    bw, bh = 3.0, 0.72
    y_in = 0.55
    y_ln1, y_ssm, y_add1 = 1.75, 2.95, 4.05
    y_ln2, y_mlp, y_add2 = 5.15, 6.35, 7.45
    y_out = 8.55
    half = bh / 2

    _box(ax, cx, y_in, 4.1, bh, r"input sequence $(T, d)$", BLUE,
         fontsize=13, weight="normal")
    _box(ax, cx, y_ln1, bw, bh, "LayerNorm", GRAY, fontsize=13,
         weight="normal")
    _box(ax, cx, y_ssm, 3.5, bh, "diagonal SSM (S4D)", GREEN, fontsize=13,
         weight="normal")
    _box(ax, cx, y_ln2, bw, bh, "LayerNorm", GRAY, fontsize=13,
         weight="normal")
    _box(ax, cx, y_mlp, bw, bh, r"gated MLP", ORANGE, fontsize=13,
         weight="normal")
    _box(ax, cx, y_out, 4.1, bh, r"output sequence $(T, d)$", BLUE,
         fontsize=13, weight="normal")
    _op(ax, cx, y_add1, "+")
    _op(ax, cx, y_add2, "+")

    # main flow
    for y0, y1, gap0, gap1 in [
            (y_in, y_ln1, half, half), (y_ln1, y_ssm, half, half),
            (y_ssm, y_add1, half, 0.26), (y_add1, y_ln2, 0.26, half),
            (y_ln2, y_mlp, half, half), (y_mlp, y_add2, half, 0.26),
            (y_add2, y_out, 0.26, half)]:
        fl.arrow(ax, (cx, y0 + gap0), (cx, y1 - gap1), color=GRAY, lw=1.9,
                 mut=13)

    # residual skips (left side)
    xl = cx - 2.6
    for y0, y1 in [(( y_in + y_ln1) / 2, y_add1), ((y_add1 + y_ln2) / 2,
                                                   y_add2)]:
        ax.plot([cx, xl], [y0, y0], color=GRAY, lw=1.7)
        ax.plot([xl, xl], [y0, y1], color=GRAY, lw=1.7)
        fl.arrow(ax, (xl, y1), (cx - 0.30, y1), color=GRAY, lw=1.7, mut=12)

    # annotations on the right
    xr = cx + 1.85
    ax.text(xr, y_ssm + 0.02,
            "$x_t = \\bar{a} \\odot x_{t-1} + \\bar{b}\\, u_t$\n"
            "$y_t = \\langle c, x_t\\rangle + d\\, u_t$",
            fontsize=12.5, color="black", ha="left", va="center")
    ax.text(xr, y_mlp + 0.02,
            r"$\mathbf{W}_o\,(\mathbf{W}_v \mathbf{y} \odot "
            r"\sigma(\mathbf{W}_g \mathbf{y}))$",
            fontsize=12.5, color="black", ha="left", va="center")

    # x L bracket
    xb = cx + 4.9
    ax.plot([xb, xb + 0.18], [y_ln1 - half, y_ln1 - half], color="black",
            lw=1.4)
    ax.plot([xb + 0.18, xb + 0.18], [y_ln1 - half, y_add2 + 0.26],
            color="black", lw=1.4)
    ax.plot([xb, xb + 0.18], [y_add2 + 0.26, y_add2 + 0.26], color="black",
            lw=1.4)
    ax.text(xb + 0.44, (y_ln1 + y_add2) / 2, r"$\times\, L$", fontsize=15,
            color="black", ha="left", va="center")

    ax.set_xlim(0.6, 10.1)
    ax.set_ylim(0.0, 9.2)
    fl.save(fig, "mdl-modernrnn-ssm-block")


# =========================================================================== #
# 10.4 "Selective State Space Models" (sec_mamba): the selective copying     #
# task, the Mamba block, and the chapter's "three answers" summary strip     #
# (gate it -> linearize it -> select it).                                    #
# =========================================================================== #

def fig_selective_copy():
    fig, ax = plt.subplots(figsize=(10.2, 3.4))
    ax.set_aspect("equal")
    ax.axis("off")

    n_ctx, n_q = 14, 4
    cw, ch = 0.78, 0.72                      # cell pitch / height
    y_in, y_out = 2.85, 0.72
    x0 = 1.05
    gap = 0.30                               # gap before the query slots

    # marked symbols: (context index, symbol, color)
    marked = [(1, "7", BLUE), (5, "3", ORANGE), (8, "8", GREEN),
              (12, "5", PURPLE)]
    sym_at = {i: (s, c) for i, s, c in marked}

    xs_ctx = [x0 + i * cw for i in range(n_ctx)]
    xs_q = [x0 + n_ctx * cw + gap + i * cw for i in range(n_q)]

    for i, cx in enumerate(xs_ctx):
        if i in sym_at:
            s, c = sym_at[i]
            _box(ax, cx, y_in, 0.66, ch, s, c, fontsize=14)
        else:
            _box(ax, cx, y_in, 0.66, ch, r"$\cdot$", LIGHT, fontsize=13,
                 weight="normal", text_color=GRAY, lw=1.2)
    for cx in xs_q:
        _box(ax, cx, y_in, 0.66, ch, "?", GRAY, fontsize=14, weight="normal",
             ls=(0, (4, 2)), lw=1.4)

    # output row: the symbols, in order, under the query slots
    for (i, s, c), cx in zip(marked, xs_q):
        _box(ax, cx, y_out, 0.66, ch, s, c, fontsize=14)

    # Manhattan wires: drop from each symbol to its own lane, run right,
    # then drop into the output cell. Lane order (first symbol lowest)
    # makes the wires crossing-free.
    y_top_in = y_in - ch / 2 - 0.02
    y_top_out = y_out + ch / 2 + 0.06
    for k, ((i, s, c), cxq) in enumerate(zip(marked, xs_q)):
        cxs = xs_ctx[i]
        lane = 1.42 + 0.20 * k
        ax.plot([cxs, cxs], [y_top_in, lane], color=c, lw=1.6, alpha=0.9)
        ax.plot([cxs, cxq], [lane, lane], color=c, lw=1.6, alpha=0.9)
        fl.arrow(ax, (cxq, lane), (cxq, y_top_out), color=c, lw=1.6, mut=11)

    # labels
    ax.text(x0 - 0.62, y_in, "input", ha="right", va="center", fontsize=14,
            color="black")
    ax.text((xs_q[0] + xs_q[-1]) / 2, y_in + ch / 2 + 0.30, "query slots",
            ha="center", va="bottom", fontsize=13, color="black")
    ax.text((xs_ctx[0] + xs_ctx[-1]) / 2, y_in + ch / 2 + 0.30,
            "symbols at random positions among filler", ha="center",
            va="bottom", fontsize=13, color="black")
    ax.text((xs_q[0] + xs_q[-1]) / 2, y_out - ch / 2 - 0.30,
            "output, in order", ha="center", va="top", fontsize=13,
            color="black")
    ax.text(x0 - 0.30, 0.30,
            "the marked positions change from example to example:\n"
            "content, not position, decides what to remember",
            ha="left", va="center", fontsize=12, color="black")

    ax.set_xlim(-0.85, 15.6)
    ax.set_ylim(-0.55, 3.95)
    fl.save(fig, "mdl-modernrnn-selective-copy")


def fig_mamba_block():
    fig, ax = plt.subplots(figsize=(9.4, 7.0))
    ax.set_aspect("equal")
    ax.axis("off")

    cx, gx = 3.7, 6.5                        # main branch x, gate branch x
    mid = (cx + gx) / 2
    bw, bh = 3.0, 0.70
    half = bh / 2
    y_in = 0.50
    y_ln = 1.62
    y_proj = 2.74
    y_conv = 3.98
    y_silu = 4.98
    y_ssm = 5.98
    y_mul = 7.05
    y_out = 8.05
    y_add = 9.05
    y_top = 9.95

    _box(ax, mid, y_in, 4.0, bh, r"input sequence $(T, d)$", BLUE,
         fontsize=13, weight="normal")
    _box(ax, mid, y_ln, 2.6, bh, "LayerNorm", GRAY, fontsize=13,
         weight="normal")
    _box(ax, mid, y_proj, 4.6, bh, r"input projection $d \to 2 \times 2d$",
         BLUE, fontsize=13, weight="normal")
    _box(ax, cx, y_conv, 3.35, bh, "causal conv, width 4", ORANGE,
         fontsize=12, weight="normal")
    _op(ax, cx, y_silu, "SiLU", r=0.40, fontsize=11)
    _box(ax, cx, y_ssm, bw, bh, "selective SSM", GREEN, fontsize=13,
         weight="normal")
    _op(ax, gx, y_silu, "SiLU", r=0.40, fontsize=11)
    _op(ax, mid, y_mul, r"$\odot$", r=0.27)
    _box(ax, mid, y_out, 4.6, bh, r"output projection $2d \to d$", BLUE,
         fontsize=13, weight="normal")
    _op(ax, mid, y_add, "+", r=0.27)
    _box(ax, mid, y_top, 4.0, bh, r"output sequence $(T, d)$", BLUE,
         fontsize=13, weight="normal")

    # main spine up to the projection
    fl.arrow(ax, (mid, y_in + half), (mid, y_ln - half), color=GRAY, lw=1.9,
             mut=13)
    fl.arrow(ax, (mid, y_ln + half), (mid, y_proj - half), color=GRAY, lw=1.9,
             mut=13)

    # fork: projection -> conv (main), projection -> gate SiLU
    fl.arrow(ax, (cx, y_proj + half), (cx, y_conv - half), color=GRAY, lw=1.9,
             mut=13)
    fl.arrow(ax, (gx, y_proj + half), (gx, y_silu - 0.40 - 0.06), color=GRAY,
             lw=1.9, mut=13)

    # main branch: conv -> SiLU -> SSM -> multiply
    fl.arrow(ax, (cx, y_conv + half), (cx, y_silu - 0.40 - 0.06), color=GRAY,
             lw=1.9, mut=13)
    fl.arrow(ax, (cx, y_silu + 0.40 + 0.06), (cx, y_ssm - half), color=GRAY,
             lw=1.9, mut=13)
    ax.plot([cx, cx], [y_ssm + half, y_mul], color=GRAY, lw=1.9)
    fl.arrow(ax, (cx, y_mul), (mid - 0.31, y_mul), color=GRAY, lw=1.9, mut=13)

    # gate branch: SiLU -> multiply
    ax.plot([gx, gx], [y_silu + 0.40 + 0.06, y_mul], color=GRAY, lw=1.9)
    fl.arrow(ax, (gx, y_mul), (mid + 0.31, y_mul), color=GRAY, lw=1.9, mut=13)
    ax.text(gx + 0.22, (y_silu + y_mul) / 2 + 0.28, "gate", fontsize=13,
            color="black", ha="left", va="center")

    # multiply -> out projection -> add -> output
    fl.arrow(ax, (mid, y_mul + 0.27), (mid, y_out - half), color=GRAY, lw=1.9,
             mut=13)
    fl.arrow(ax, (mid, y_out + half), (mid, y_add - 0.27), color=GRAY, lw=1.9,
             mut=13)
    fl.arrow(ax, (mid, y_add + 0.27), (mid, y_top - half), color=GRAY, lw=1.9,
             mut=13)

    # residual skip, far left
    xl = 0.62
    y_res0 = (y_in + y_ln) / 2
    ax.plot([mid, xl], [y_res0, y_res0], color=GRAY, lw=1.7)
    ax.plot([xl, xl], [y_res0, y_add], color=GRAY, lw=1.7)
    fl.arrow(ax, (xl, y_add), (mid - 0.31, y_add), color=GRAY, lw=1.7, mut=12)

    # Delta_t, B_t, C_t are functions of the SSM input
    ax.text(1.62, y_ssm + 0.86,
            r"$\Delta_t, \mathbf{B}_t, \mathbf{C}_t$" + "\nfrom the input",
            fontsize=12.5, color="black", ha="center", va="center")
    fl.arrow(ax, (1.80, y_ssm + 0.44), (cx - bw / 2 - 0.06, y_ssm + 0.12),
             color=GREEN, lw=1.5, mut=11)

    # x L bracket on the right
    xb = 8.35
    ax.plot([xb, xb + 0.18], [y_ln - half, y_ln - half], color="black",
            lw=1.4)
    ax.plot([xb + 0.18, xb + 0.18], [y_ln - half, y_add + 0.27],
            color="black", lw=1.4)
    ax.plot([xb, xb + 0.18], [y_add + 0.27, y_add + 0.27], color="black",
            lw=1.4)
    ax.text(xb + 0.44, (y_ln + y_add) / 2, r"$\times\, L$", fontsize=15,
            color="black", ha="left", va="center")

    ax.set_xlim(0.0, 9.6)
    ax.set_ylim(0.0, 10.55)
    fl.save(fig, "mdl-modernrnn-mamba-block")


def fig_three_answers():
    fig, ax = plt.subplots(figsize=(11.6, 3.6))
    ax.set_aspect("equal")
    ax.axis("off")

    pw, ph = 3.55, 3.0                       # panel width/height
    y_c = 1.62
    xs = [1.98, 5.93, 9.88]
    colors = [BLUE, GREEN, ORANGE]
    titles = ["1. Gate it", "2. Linearize it", "3. Select it"]
    models = ["LSTM, GRU", "minGRU, S4D", "Mamba"]
    eqs = [
        r"$\mathbf{C}_t = \mathbf{F}_t \odot \mathbf{C}_{t-1}"
        r" + \mathbf{I}_t \odot \tilde{\mathbf{C}}_t$",
        r"$\mathbf{h}_t = \bar{\mathbf{a}} \odot \mathbf{h}_{t-1}"
        r" + \bar{\mathbf{b}}\, u_t$",
        r"$\mathbf{h}_t = \bar{\mathbf{a}}_t \odot \mathbf{h}_{t-1}"
        r" + \bar{\mathbf{b}}_t$",
    ]
    notes = ["gates read the data;\nnonlinear state path",
             "fixed dynamics;\naffine state path",
             r"$\bar{\mathbf{a}}_t, \bar{\mathbf{b}}_t$ chosen by $u_t$;"
             + "\nstill an affine map"]
    props = [[("content-aware", True), ("parallel training", False)],
             [("content-aware", False), ("parallel training", True)],
             [("content-aware", True), ("parallel training", True)]]

    for cx, color, title, model, eq, note, prop in zip(
            xs, colors, titles, models, eqs, notes, props):
        x, y = cx - pw / 2, y_c - ph / 2
        for fc, a in [(color, 0.07), ("none", 1.0)]:
            ax.add_patch(FancyBboxPatch(
                (x, y), pw, ph, boxstyle="round,pad=0.03,rounding_size=0.16",
                linewidth=2.0, edgecolor=color, facecolor=fc, alpha=a))
        ax.text(cx, y_c + 1.13, title, ha="center", va="center",
                fontsize=15, fontweight="bold", color="black")
        ax.text(cx, y_c + 0.72, model, ha="center", va="center",
                fontsize=12.5, color=GRAY)
        ax.text(cx, y_c + 0.18, eq, ha="center", va="center", fontsize=12.5,
                color="black")
        ax.text(cx, y_c - 0.42, note, ha="center", va="center",
                fontsize=11.5, color="black")
        for k, (name, ok) in enumerate(prop):
            mark = "✓" if ok else "✗"
            mcol = GREEN if ok else ORANGE
            yk = y_c - 0.98 - 0.34 * k
            ax.text(cx - 1.38, yk, mark, ha="center", va="center",
                    fontsize=13, color=mcol, fontweight="bold")
            ax.text(cx - 1.16, yk, name, ha="left", va="center",
                    fontsize=11.5, color="black")

    for x0, x1 in [(xs[0] + pw / 2, xs[1] - pw / 2),
                   (xs[1] + pw / 2, xs[2] - pw / 2)]:
        fl.arrow(ax, (x0 + 0.06, y_c), (x1 - 0.06, y_c), color=GRAY, lw=2.0,
                 mut=14)

    ax.set_xlim(0.0, 11.9)
    ax.set_ylim(-0.15, 3.4)
    fl.save(fig, "mdl-modernrnn-three-answers")


# =========================================================================== #
# Driver                                                                      #
# =========================================================================== #

FIGURES = [
    fig_lstm_cell,
    fig_gru_cell,
    fig_deep_rnn,
    fig_bi_rnn,
    fig_encoder_decoder,
    fig_seq2seq,
    fig_scan_tree,
    fig_ssm_views,
    fig_hippo_reconstruction,
    fig_ssm_block,
    fig_selective_copy,
    fig_mamba_block,
    fig_three_answers,
]


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
