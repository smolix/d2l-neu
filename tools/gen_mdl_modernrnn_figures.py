#!/usr/bin/env python3
"""Generate the illustrative figures for the "State Space Models" chapter
(``chapter_recurrent-modern``) in the one shared house style defined in
``gen_mdl_figures.py``.

Fifteen figures, by section (five added in the 2026-11 review pass are
listed at their definitions):

  * "Gated Recurrent Networks" (sec_lstm) -- the LSTM memory cell and the GRU
    cell (house-style redraws of the old ``img/lstm-*.svg``/``gru-*.svg``).
  * "Linear Recurrences and Parallel Scans" (sec_linear-recurrence) -- the
    parallel prefix (doubling) scan.
  * "State Space Models" (sec_ssm) -- zero-order-hold discretization (rate
    lambda -> decay exp(Delta lambda), and the half-life per step size), and
    the three equivalent views of a linear SSM.
  * "Selective State Space Models" (sec_mamba) -- the selective copying task
    and the Mamba layer.
  * "Matrix-State Recurrences" -- the semiseparable mask lives in
    ``gen_mdl_ssm_family_figures.py``.
  * "The Delta Rule" -- the geometry of the transition I - beta k k^T.
  * "Hybrid Architectures" -- the decode-state cache plot lives in
    ``gen_mdl_ssm_hybrids_figures.py``.

(The deep-RNN, bidirectional-RNN, HiPPO-reconstruction, S4D-block and
"three answers" figures were dropped with the 2026-10 chapter rewrite, which
no longer references them; see git history.)

The encoder-decoder interface and the unrolled RNN encoder-decoder (teacher
forcing) are also written here; ``seq2seq.md`` (``sec_seq2seq``) in this
chapter references them.

The notebooks / prose reference the generated files with no drawing code (like
the slide SVGs).  Figures that show a *computed* result (e.g. the
discretization map, the SSM step response) use real numerical computation so the
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
TEAL = fl.T.TEAL.base      # hidden/cell-state role (guide §4.3)
PURPLE = fl.T.PURPLE.base  # parameter / second-state role
# coloured *text* uses the dark tone of its accent (guide §2.3, §7.3)
TEAL_TXT, ORANGE_TXT = fl.T.TEAL.dark, fl.T.ORANGE.dark

from matplotlib.patches import Circle, FancyBboxPatch


# =========================================================================== #
# "Gated Recurrent Networks" (sec_lstm): the LSTM cell and the GRU cell.    #
#                                                                              #
# Restyled carryovers, written under new house-style names so the old        #
# hand-drawn SVGs (``img/lstm-0..3.svg``, ``img/gru-1..3.svg``) are left     #
# untouched.                                                                 #
#                                                                              #
# Colour code (consistent across the figures and with the ch. 9              #
# diagrams): inputs x blue, hidden states h teal (the state role), the LSTM  #
# cell state c orange, outputs o orange, computation heads and elementwise   #
# ops grey.                                                                  #
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
    xF, xI, xC, xO = 3.0, 4.5, 5.7, 7.45  # head x-positions
    x_prod = (xI + xC) / 2                # i * c~ product column
    x_out = 8.55                          # output-gate column
    bw, bh = 1.25, 0.85                   # state-box size
    hw, hh = 0.82, 0.72                   # head-box size
    r = 0.26                              # op-circle radius

    # --- cell-state conveyor (top) ------------------------------------------
    _box(ax, 1.05, y_cell, bw, bh, r"$\mathbf{c}_{t-1}$", ORANGE)
    _op(ax, xF, y_cell, r"$\odot$")
    _op(ax, x_prod, y_cell, r"$+$")
    _box(ax, 9.9, y_cell, bw, bh, r"$\mathbf{c}_{t}$", ORANGE)
    fl.arrow(ax, (1.05 + bw / 2, y_cell), (xF - r, y_cell), color=GRAY)
    fl.arrow(ax, (xF + r, y_cell), (x_prod - r, y_cell), color=GRAY)
    fl.arrow(ax, (x_prod + r, y_cell), (9.9 - bw / 2, y_cell), color=GRAY)

    # --- input bus (bottom): h_{t-1} and x_t feed all four heads -------------
    _box(ax, 1.05, 1.85, bw, bh, r"$\mathbf{h}_{t-1}$", TEAL)
    _box(ax, 1.05, 0.55, bw, bh, r"$\mathbf{x}_{t}$", BLUE)
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
    ax.text(xF - 0.18, (y_head + y_cell) / 2 + 0.12, r"$\mathbf{f}_t$",
            ha="right", va="center", fontsize=14, color="black")

    # input gate and input node: join at an elementwise product, then up into +
    y_ic = 3.95
    _op(ax, x_prod, y_ic, r"$\odot$")
    fl.arrow(ax, (xI, y_head + hh / 2), (x_prod - r * 0.72, y_ic - r * 0.72),
             color=GRAY, lw=1.8, mut=12)
    fl.arrow(ax, (xC, y_head + hh / 2), (x_prod + r * 0.72, y_ic - r * 0.72),
             color=GRAY, lw=1.8, mut=12)
    fl.arrow(ax, (x_prod, y_ic + r), (x_prod, y_cell - r), color=GRAY)
    ax.text(xI - 0.42, y_head + hh / 2 + 0.34, r"$\mathbf{i}_t$", ha="center",
            va="center", fontsize=14, color="black")
    ax.text(xC + 0.52, y_head + hh / 2 + 0.34, r"$\tilde{\mathbf{c}}_t$",
            ha="center", va="center", fontsize=14, color="black")

    # output branch: conveyor -> tanh -> (x) with output gate -> h_t
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
    ax.text((xO + hw / 2 + x_out - r) / 2, y_head - 0.46, r"$\mathbf{o}_t$",
            ha="center", va="center", fontsize=14, color="black")
    fl.arrow(ax, (x_out + r, y_head), (9.9 - bw / 2, y_head), color=GRAY)
    _box(ax, 9.9, y_head, bw, bh, r"$\mathbf{h}_{t}$", TEAL)

    ax.set_xlim(0.2, 10.75)
    ax.set_ylim(0.0, 5.7)
    fl.save(fig, "mdl-modernrnn-lstm-cell")


def fig_gru_cell():
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    ax.set_aspect("equal")
    ax.axis("off")

    y_conv, y_head, y_bus = 4.85, 2.45, 0.95
    xR, xZ, xH = 2.75, 4.15, 6.05          # head x-positions
    x_blend = xH                            # (1 - z) * candidate column sits
    bw, bh = 1.25, 0.85                     # directly above the tanh head
    hw, hh = 0.82, 0.72
    r = 0.26

    # --- state conveyor (top): h_{t-1} -> (x z) -> (+) -> h_t -----------------
    _box(ax, 1.0, y_conv, bw, bh, r"$\mathbf{h}_{t-1}$", TEAL)
    _op(ax, xZ, y_conv, r"$\odot$")
    _op(ax, x_blend, y_conv, r"$+$")
    _box(ax, 8.55, y_conv, bw, bh, r"$\mathbf{h}_{t}$", TEAL)
    fl.arrow(ax, (1.0 + bw / 2, y_conv), (xZ - r, y_conv), color=GRAY)
    fl.arrow(ax, (xZ + r, y_conv), (x_blend - r, y_conv), color=GRAY)
    fl.arrow(ax, (x_blend + r, y_conv), (8.55 - bw / 2, y_conv), color=GRAY)

    # --- input wiring (as in the GRU equations): the x bus feeds all three
    # heads; h_{t-1} joins only the two GATE risers on its own bus above it
    # (the candidate tanh sees the previous state only through r_t (x) h).
    _box(ax, 1.0, 0.95, bw, bh, r"$\mathbf{x}_{t}$", BLUE)
    xj = 1.0 + bw / 2
    ax.plot([xj, xH], [y_bus, y_bus], color=GRAY, lw=1.8, zorder=1)
    for x in (xR, xZ, xH):
        if x != xH:
            _dot(ax, x, y_bus)
        fl.arrow(ax, (x, y_bus), (x, y_head - hh / 2), color=GRAY, lw=1.8,
                 mut=12)
    # h_{t-1} tap: down from the conveyor, right along its own bus, joining
    # the r and z risers at the dots
    x_tap, y_h = 1.95, 1.55
    _dot(ax, x_tap, y_conv)
    ax.plot([x_tap, x_tap], [y_conv, y_h], color=GRAY, lw=1.8, zorder=1)
    ax.plot([x_tap, xZ], [y_h, y_h], color=GRAY, lw=1.8, zorder=1)
    _dot(ax, xR, y_h)
    _dot(ax, xZ, y_h)

    # --- heads ----------------------------------------------------------------
    _head(ax, xR, y_head, r"$\sigma$", w=hw, h=hh)
    _head(ax, xZ, y_head, r"$\sigma$", w=hw, h=hh)
    _head(ax, xH, y_head, r"$\tanh$", w=hw + 0.28, h=hh, fontsize=12)

    # reset gate: filters the previous state before it enters the candidate
    y_rH = 3.6
    _op(ax, xR, y_rH, r"$\odot$")
    fl.arrow(ax, (xR, y_head + hh / 2), (xR, y_rH - r), color=GRAY, lw=1.8,
             mut=12)
    ax.text(xR - 0.18, (y_head + y_rH) / 2 - 0.02, r"$\mathbf{r}_t$",
            ha="right", va="center", fontsize=14, color="black")
    # h_{t-1} into the reset product from the conveyor above
    fl.arrow(ax, (x_tap, y_conv - 0.0), (xR - r * 0.72, y_rH + r * 0.72),
             color=GRAY, lw=1.8, mut=12)
    # gated state into the candidate head (one clean perpendicular wire
    # crossing with the update-gate riser; unavoidable in a planar layout)
    fl.arrow(ax, (xR + r * 0.72, y_rH - r * 0.72),
             (xH - (hw + 0.28) / 2, y_head + hh / 2 - 0.06), color=GRAY,
             lw=1.8, mut=12)

    # update gate up into the conveyor product, with a branch carrying the
    # complement (1 - z) horizontally into the blend product
    fl.arrow(ax, (xZ, y_head + hh / 2), (xZ, y_conv - r), color=GRAY)
    ax.text(xZ - 0.18, 3.9, r"$\mathbf{z}_t$", ha="right", va="center",
            fontsize=14, color="black")
    y_c = 3.6
    _op(ax, x_blend, y_c, r"$\odot$")
    _dot(ax, xZ, y_c)
    fl.arrow(ax, (xZ, y_c), (x_blend - r, y_c), color=GRAY, lw=1.8, mut=12)
    ax.text((xZ + x_blend - r) / 2, y_c + 0.23, r"$1-\mathbf{z}_t$",
            ha="center", va="bottom", fontsize=13, color="black")

    # candidate straight up into the blend product, then on up into the
    # convex combination -- one straight column above the tanh head
    fl.arrow(ax, (xH, y_head + hh / 2), (xH, y_c - r), color=GRAY, lw=1.8,
             mut=12)
    fl.arrow(ax, (x_blend, y_c + r), (x_blend, y_conv - r), color=GRAY,
             lw=1.8, mut=12)
    ax.text(xH + 0.2, (y_head + hh / 2 + y_c - r) / 2,
            r"$\tilde{\mathbf{h}}_t$", ha="left", va="center",
            fontsize=14, color="black")

    ax.set_xlim(0.15, 9.35)
    ax.set_ylim(0.3, 5.5)
    fl.save(fig, "mdl-modernrnn-gru-cell")


# =========================================================================== #
# Encoder-decoder figures (referenced from ``seq2seq.md``, ``sec_seq2seq``,  #
# in this chapter): the encoder-decoder interface, and the unrolled RNN      #
# encoder-decoder on string reversal, with teacher forcing.                  #
#                                                                              #
# ``_ed_box``/``_ed_arrow`` are this section's own box/arrow helpers -- they #
# take separate face/edge colours and an explicit zorder respectively, both  #
# unlike the ``_box``/``fl.arrow`` used elsewhere in this file, so they keep #
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
    fig, ax = plt.subplots(figsize=(8.2, 1.3))
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0.8, 2.45)
    ax.axis("off")
    ax.set_aspect("equal")

    y = 1.5
    # source text
    ax.text(0.75, y, '"They are\nwatching ."', ha="center", va="center",
            fontsize=_ED_SMALL, color="black")
    _ed_box(ax, 3.1, y, 1.9, 1.15, "Encoder", fl.T.BLUE.tint, BLUE)
    _ed_box(ax, 7.1, y, 1.9, 1.15, "Decoder", fl.T.ORANGE.tint, ORANGE)
    ax.text(9.75, y, '"Ils\nregardent ."', ha="center", va="center",
            fontsize=_ED_SMALL, color="black")

    _ed_arrow(ax, (1.55, y), (2.15, y))
    _ed_arrow(ax, (4.05, y), (6.15, y))
    _ed_arrow(ax, (8.05, y), (9.05, y))

    # the fixed-shape state on the middle hand-off
    ax.text(5.1, y + 0.5, "state", ha="center", va="bottom",
            fontsize=_ED_SMALL, color=TEAL_TXT, fontstyle="italic")
    ax.plot(5.1, y, "o", color=TEAL, ms=10, zorder=4)

    fl.save(fig, "mdl-modernrnn-encoder-decoder")


def fig_seq2seq():
    """Unrolled RNN encoder-decoder on string reversal, with teacher forcing.
    The encoder reads the source letters and passes its final state as context
    to every decoder step; the decoder is fed the reversed string shifted
    right behind a start token and predicts the reversed string."""
    fig, ax = plt.subplots(figsize=(10.4, 3.35))
    ax.set_ylim(0.75, 5.2)
    ax.axis("off")
    ax.set_aspect("equal")

    cw, ch = 1.05, 0.8          # cell size
    yc = 2.4                    # recurrent-cell row
    dx = 1.45

    enc_tokens = ["b", "a", "d", "e"]   # no "c": c names the context
    enc_x = [0.9 + i * dx for i in range(4)]
    dec_in = ["<start>", "e", "d", "a"]
    dec_out = ["e", "d", "a", "b"]
    dec_x = [enc_x[-1] + dx + 0.6 + i * dx for i in range(4)]
    ax.set_xlim(0, dec_x[-1] + 0.9)

    # dashed separator between encoder and decoder
    xsep = (enc_x[-1] + dec_x[0]) / 2
    ax.plot([xsep, xsep], [0.85, 4.8], ls=(0, (4, 4)), color=LIGHT, lw=1.2)
    ax.text((enc_x[1] + enc_x[2]) / 2, 4.75, "Encoder", ha="center",
            fontsize=_ED_TXT, color=BLUE)
    ax.text((dec_x[1] + dec_x[2]) / 2, 4.75, "Decoder", ha="center",
            fontsize=_ED_TXT, color=ORANGE_TXT)

    # encoder cells + inputs, left-to-right recurrence
    for i, x in enumerate(enc_x):
        _ed_box(ax, x, yc, cw, ch, "", fl.T.BLUE.tint, BLUE)
        _ed_arrow(ax, (x, 1.55), (x, yc - ch / 2))          # token -> cell
        ax.text(x, 1.35, enc_tokens[i], ha="center", va="top",
                fontsize=_ED_SMALL, color="black")
        if i > 0:
            _ed_arrow(ax, (enc_x[i - 1] + cw / 2, yc), (x - cw / 2, yc))

    # context state c = encoder final state
    xc = xsep
    ax.plot(xc, yc, "o", color=TEAL, ms=11, zorder=5)
    _ed_arrow(ax, (enc_x[-1] + cw / 2, yc), (xc - 0.16, yc), color=TEAL)

    # the context reaches every decoder step along one teal bus above the
    # cells: up from c, right along the bus, and down into each cell's top
    # (left of its prediction arrow)
    y_bus = yc + ch / 2 + 0.38
    x_drop = [x - 0.3 for x in dec_x]
    ax.plot([xc, xc, x_drop[-1]], [yc + 0.12, y_bus, y_bus], color=TEAL,
            lw=1.5, zorder=3, solid_joinstyle="miter")
    ax.text(xc - 0.12, y_bus, r"$\mathbf{c}$", ha="right", va="center",
            fontsize=_ED_TXT + 1, color=TEAL_TXT)
    for xd in x_drop:
        _ed_arrow(ax, (xd, y_bus), (xd, yc + ch / 2), color=TEAL, lw=1.5,
                  mut=11)

    # decoder cells: teacher-forced input below, prediction above
    for i, x in enumerate(dec_x):
        _ed_box(ax, x, yc, cw, ch, "", fl.T.ORANGE.tint, ORANGE)
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
            _ed_arrow(ax, (xc + 0.16, yc), (x - cw / 2, yc), color=TEAL)
        else:
            _ed_arrow(ax, (dec_x[i - 1] + cw / 2, yc), (x - cw / 2, yc))

    fl.save(fig, "mdl-modernrnn-seq2seq")


# =========================================================================== #
# The parallel prefix (doubling) scan ("Linear Recurrences and Parallel       #
# Scans", sec_linear-recurrence); zero-order-hold discretization and the      #
# three equivalent views of a linear SSM (continuous-time ODE, unrolled       #
# recurrence, convolution with the kernel) ("State Space Models", sec_ssm).   #
# =========================================================================== #

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
            color = BLUE if r == 0 else (TEAL if done else ORANGE)
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


def fig_ssm_discretization():
    """Zero-order hold for a real rate: the decay per step
    a-bar = exp(-Delta |lambda|) against Delta |lambda| on a log axis, with
    the half-life ln 2 / (Delta |lambda|) at four marked points and the band
    of Delta |lambda| that the S4D initialization covers (Delta in
    [0.001, 0.1], |lambda| in [1, 16]).  One panel: the complex-rate planes of
    the earlier version were dropped (the text mentions complex rates in one
    aside)."""
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    FS = 14
    lo, hi = 1e-3 * 1, 1e-1 * 16              # S4D at initialization
    ax.axvspan(lo, hi, color=GREEN, alpha=0.13, lw=0)
    ax.text(np.sqrt(lo * hi), 1.13, "S4D at initialization", fontsize=13,
            ha="center", va="bottom", color="black")
    x = np.logspace(np.log10(3e-4), np.log10(4), 400)
    ax.plot(x, np.exp(-x), color=ORANGE, lw=2.4)
    for m in [1e-3, 1e-2, 1e-1, 1.0]:
        ax.plot([m], [np.exp(-m)], "o", color=ORANGE, ms=8, mec="white",
                mew=1.0, zorder=5)
        hl = np.log(2) / m
        txt = f"{hl:.0f}" if hl >= 10 else (f"{hl:.1f}" if hl >= 1
                                             else f"{hl:.2f}")
        if m < 0.5:                       # flat part: label under the dot
            ax.text(m, np.exp(-m) - 0.08, txt, fontsize=13, ha="center",
                    va="top")
        else:                             # steep part: label left of the dot
            ax.text(m * 0.72, np.exp(-m) - 0.04, txt, fontsize=13,
                    ha="right", va="center")
    ax.text(1.3e-3, 0.36, "half-life in steps,\n"
            r"$\ln 2\,/\,(\Delta|\lambda|)$", fontsize=13, ha="left",
            va="center")
    ax.set_xscale("log")
    ax.set_xlim(3e-4, 4)
    ax.set_ylim(0, 1.12)
    ax.set_xticks([1e-3, 1e-2, 1e-1, 1])
    ax.set_xticklabels([r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$1$"])
    ax.minorticks_off()
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0", "0.5", "1"])
    ax.tick_params(labelsize=13, colors="black")
    # matplotlib 3.11.2 gives the 10^{-k} mathtext labels a deeper box, which
    # pushed the tick labels and the axis label down; pull both back up.
    ax.tick_params(axis="x", pad=1.5)
    ax.set_xlabel(r"$\Delta|\lambda|$", fontsize=FS, color="black",
                  labelpad=-2)
    ax.set_ylabel(r"$\bar{a} = e^{-\Delta|\lambda|}$", fontsize=FS,
                  color="black")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("black")

    fl.save(fig, "mdl-modernrnn-discretization")


def fig_ssm_views():
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5),
                             gridspec_kw={"width_ratios": [1.0, 1.15, 1.0]})
    ax_ode, ax_rec, ax_conv = axes

    # ---- (a) continuous time: input pulse and first-order state response --
    ax = ax_ode
    t = np.linspace(0, 10, 400)
    u = ((t > 1.2) & (t < 4.2)).astype(float)
    # h' = a h + b x with a = -0.9, b = 0.9 (exact first-order response)
    a_c, b_c = -0.9, 0.9
    x = np.zeros_like(t)
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        x[i] = np.exp(a_c * dt) * x[i - 1] + (np.exp(a_c * dt) - 1) / a_c \
            * b_c * u[i]
    ax.plot(t, u, color=BLUE, lw=1.8)
    ax.plot(t, x, color=TEAL, lw=2.6)
    ax.text(2.7, 1.12, r"input $x(t)$", fontsize=12.5, color=BLUE,
            ha="center", va="bottom")
    ax.text(5.05, 0.82, r"state $h(t)$", fontsize=12.5, color=TEAL_TXT,
            ha="left", va="bottom")
    ax.text(7.7, 0.52, r"$h' = Ah + Bx$", fontsize=14, color="black",
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
        _box(ax, cx, y_u, bw, bh, rf"$x_{{{s}}}$", BLUE, fontsize=13)
        _box(ax, cx, y_x, bw, bh, rf"$h_{{{s}}}$", TEAL, fontsize=13)
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
        _box(ax, cx, y_u, bw, bh, rf"$x_{{{i + 1}}}$", BLUE, fontsize=12,
             weight="normal", lw=1.4)
    # kernel weights K_0.. over the last inputs (aligned under the output)
    kvals = 0.95 * 0.55 ** np.arange(Tc)
    for i in range(Tc):
        k = Tc - 1 - i               # kernel index applied to input i+1
        ax.plot([xs[i], xs[i]], [y_ker, y_ker + 1.15 * kvals[k]],
                color=PURPLE, lw=2.4, solid_capstyle="round")
        ax.plot([xs[i]], [y_ker + 1.15 * kvals[k]], "o", color=PURPLE,
                markersize=4)
        fl.arrow(ax, (xs[i], y_u + bh / 2), (xs[i], y_ker - 0.08),
                 color=GRAY, lw=1.2, mut=9)
    ax.plot([xs[0] - 0.3, xs[-1] + 0.3], [y_ker, y_ker], color="black",
            lw=1.0)
    # K_4 sits on input x_3 (index Tc - 1 - 2): centred over its short stem
    ax.text(xs[2], y_ker + 0.5, r"$K_4$", fontsize=12,
            color="black", ha="center", va="center")
    ax.text(xs[-1] + 0.28, y_ker + 1.15 * kvals[0], r"$K_0$",
            fontsize=12, color="black", ha="left", va="center")
    cx_out = xs[-1]
    _box(ax, cx_out, y_out, 0.85, 0.66, rf"$y_{{{Tc}}}$", ORANGE, fontsize=12)
    fl.arrow(ax, (cx_out, y_ker + 1.15 * kvals[0] + 0.16),
             (cx_out, y_out - 0.4), color=GRAY, lw=1.8, mut=12)
    ax.text(0.05, 4.55, r"$\mathbf{y} = \mathbf{K} * \mathbf{x}$",
            fontsize=12.5, color="black", ha="left", va="center")
    ax.text(0.05, 3.72,
            r"$\mathbf{K} = (\mathbf{C}\bar{\mathbf{B}},"
            r"\mathbf{C}\bar{\mathbf{A}}\bar{\mathbf{B}},"
            r"\mathbf{C}\bar{\mathbf{A}}^2\bar{\mathbf{B}},\ldots)$",
            fontsize=12, color="black", ha="left", va="center")
    ax.set_xlim(-0.1, 6.6)
    ax.set_ylim(0.1, 4.9)

    fig.subplots_adjust(wspace=0.16)
    fl.save(fig, "mdl-modernrnn-ssm-views")


# =========================================================================== #
# "Selective State Space Models" (sec_mamba): the selective copying          #
# task and the Mamba block.                                                  #
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
    y_top_out = y_out + ch / 2
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
    """The Mamba layer as the class computes it: projections, short causal
    convolution, SiLU, selective SSM, SiLU gate, output projection.  The
    LayerNorm, residual connection and MLP around it belong to d2l.SeqLM."""
    fig, ax = plt.subplots(figsize=(8.2, 5.9))
    ax.set_aspect("equal")
    ax.axis("off")

    cx, gx = 3.7, 6.5                        # main branch x, gate branch x
    mid = (cx + gx) / 2
    bw, bh = 3.0, 0.70
    half = bh / 2
    y_in = 0.50
    y_proj = 1.62
    y_conv = 2.86
    y_silu = 3.86
    y_ssm = 4.86
    y_mul = 5.93
    y_out = 6.93
    y_top = 7.93

    _box(ax, mid, y_in, 4.0, bh, r"input sequence $(T, d)$", BLUE,
         fontsize=13, weight="normal")
    _box(ax, mid, y_proj, 4.6, bh, r"input projection $d \to 2 \times 2d$",
         BLUE, fontsize=13, weight="normal")
    _box(ax, cx, y_conv, 3.35, bh, "causal conv, width 4", ORANGE,
         fontsize=12, weight="normal")
    _op(ax, cx, y_silu, "SiLU", r=0.40, fontsize=11)
    _box(ax, cx, y_ssm, bw, bh, "selective SSM", TEAL, fontsize=13,
         weight="normal")
    _op(ax, gx, y_silu, "SiLU", r=0.40, fontsize=11)
    _op(ax, mid, y_mul, r"$\odot$", r=0.27)
    _box(ax, mid, y_out, 4.6, bh, r"output projection $2d \to d$", BLUE,
         fontsize=13, weight="normal")
    _box(ax, mid, y_top, 4.0, bh, r"output sequence $(T, d)$", BLUE,
         fontsize=13, weight="normal")

    # input -> projection
    fl.arrow(ax, (mid, y_in + half), (mid, y_proj - half), color=GRAY,
             lw=1.9, mut=13)

    # fork: projection -> conv (main), projection -> gate SiLU
    fl.arrow(ax, (cx, y_proj + half), (cx, y_conv - half), color=GRAY, lw=1.9,
             mut=13)
    fl.arrow(ax, (gx, y_proj + half), (gx, y_silu - 0.40), color=GRAY,
             lw=1.9, mut=13)

    # main branch: conv -> SiLU -> SSM -> multiply
    fl.arrow(ax, (cx, y_conv + half), (cx, y_silu - 0.40), color=GRAY,
             lw=1.9, mut=13)
    fl.arrow(ax, (cx, y_silu + 0.40), (cx, y_ssm - half), color=GRAY,
             lw=1.9, mut=13)
    ax.plot([cx, cx], [y_ssm + half, y_mul], color=GRAY, lw=1.9)
    fl.arrow(ax, (cx, y_mul), (mid - 0.27, y_mul), color=GRAY, lw=1.9, mut=13)

    # gate branch: SiLU -> multiply
    ax.plot([gx, gx], [y_silu + 0.40, y_mul], color=GRAY, lw=1.9)
    fl.arrow(ax, (gx, y_mul), (mid + 0.27, y_mul), color=GRAY, lw=1.9, mut=13)
    ax.text(gx + 0.22, (y_silu + y_mul) / 2 + 0.28, "gate", fontsize=13,
            color="black", ha="left", va="center")

    # multiply -> out projection -> output
    fl.arrow(ax, (mid, y_mul + 0.27), (mid, y_out - half), color=GRAY, lw=1.9,
             mut=13)
    fl.arrow(ax, (mid, y_out + half), (mid, y_top - half), color=GRAY, lw=1.9,
             mut=13)

    # Delta_t, B_t, C_t are functions of the SSM input
    ax.text(1.12, y_ssm + 0.86,
            r"$\Delta_t, \mathbf{B}_t, \mathbf{C}_t$" + "\nfrom the input",
            fontsize=12.5, color="black", ha="center", va="center")
    fl.arrow(ax, (1.30, y_ssm + 0.44), (cx - bw / 2, y_ssm + 0.12),
             color=TEAL, lw=1.5, mut=11)

    ax.set_xlim(0.0, 7.75)
    ax.set_ylim(0.0, 8.45)
    fl.save(fig, "mdl-modernrnn-mamba-block")


# =========================================================================== #
# "The Delta Rule" (deltanet): the geometry of the transition I - beta k k^T. #
# =========================================================================== #

def fig_delta_geometry():
    """The transition I - beta k k^T acting on s = (1.2, 0.8) in (k, k-perp)
    coordinates, for beta = 0, 0.5, 1, 2.  The images are computed by
    applying the matrix; they all lie on the dashed line through s parallel
    to k, because only the k-component (scaled by 1 - beta) changes."""
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    k = np.array([1.0, 0.0])
    s = np.array([1.2, 0.8])
    FS = 15
    xr, yr = (-1.65, 1.85), (-0.32, 1.3)
    fl.axis_cross(ax, xr, yr, color="black")

    betas = [0.0, 0.5, 1.0, 2.0]
    colors = [BLUE, PURPLE, GREEN, ORANGE]
    imgs = [(np.eye(2) - b * np.outer(k, k)) @ s for b in betas]
    # dashed guide through s parallel to k, spanning all the images
    xs_img = [p[0] for p in imgs]
    ax.plot([min(xs_img), max(xs_img)], [s[1], s[1]], color=GRAY, lw=1.5,
            ls="--", zorder=1)
    for p, c in zip(imgs, colors):
        fl.arrow(ax, (0, 0), p, color=c, lw=2.4, mut=15)
    fl.arrow(ax, (0, 0), k, color="black", lw=3.2, mut=17)

    # labels: s to the right of its tip; each beta above its tip, except
    # beta = 1 (tip on the k-perp axis), which goes up-left of the axis
    ax.text(s[0] + 0.08, s[1], r"$\mathbf{s}$", fontsize=16, ha="left",
            va="center")
    for p, b in zip(imgs, betas):
        txt = rf"$\beta = {b:g}$"
        if b == 1.0:
            ax.text(p[0] - 0.07, p[1] + 0.09, txt, fontsize=FS, ha="right",
                    va="bottom")
        else:
            ax.text(p[0], p[1] + 0.09, txt, fontsize=FS, ha="center",
                    va="bottom")
    ax.text(k[0] / 2, -0.07, r"$\mathbf{k}$", fontsize=16, ha="center",
            va="top")
    ax.text(0.07, yr[1], r"$\mathbf{k}^{\perp}$", fontsize=16, ha="left",
            va="top")

    fl.clean_axes(ax, lim=(xr, yr), hide=True)
    fl.save(fig, "mdl-modernrnn-delta-geometry")


# =========================================================================== #
# Five figures added in the 2026-11 review pass, one per section:             #
#   * "Gated Recurrent Networks" -- the shared SeqLM backbone;               #
#   * "State Space Models" -- the 2x2 design of the long-range recall test;  #
#   * "Selective State Space Models" -- the selective SSM's data flow;       #
#   * "Matrix-State Recurrences" -- the gated linear attention layer;        #
#   * "The Delta Rule" -- the three regimes of Gated DeltaNet's two gates.   #
# =========================================================================== #

def _wire(ax, pts, color=GRAY, lw=1.9):
    """A polyline wire through ``pts`` (no arrowhead)."""
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=lw, solid_joinstyle="miter",
            solid_capstyle="butt", zorder=1)


def _go(ax, tail, tip, color=GRAY, lw=1.9, mut=13):
    fl.arrow(ax, tail, tip, color=color, lw=lw, mut=mut)


def fig_seqlm_backbone():
    """d2l.SeqLM as the code builds it: embedding, then per block a pre-norm
    mixer sub-layer and an MLP sub-layer (with its own LayerNorm) on a
    residual stream, then a final LayerNorm and the linear head."""
    fig, ax = plt.subplots(figsize=(8.2, 7.9))
    ax.set_aspect("equal")
    ax.axis("off")

    xs, xb = 2.3, 6.0                 # residual stream x, branch x
    bh, half = 0.66, 0.33
    r = 0.27                          # (+) radius
    y_tok, y_emb = 0.45, 1.75
    y_ln1, y_mix, y_ln2, y_mlp = 3.25, 4.4, 5.6, 6.75
    y_top = 7.5                       # end of the block
    y_norm, y_head, y_out = 8.65, 9.95, 11.25

    _box(ax, xs, y_tok, 3.4, bh, r"bytes $x_1, \ldots, x_T$", BLUE,
         fontsize=13, weight="normal")
    _box(ax, xs, y_emb, 3.4, bh, r"embedding, $256 \to d$", BLUE,
         fontsize=13, weight="normal")
    _go(ax, (xs, y_tok + half), (xs, y_emb - half))

    # the repeated block: dashed frame around both sub-layers
    x0, x1 = 0.75, 8.4
    y0, y1 = 2.65, y_top
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.02,rounding_size=0.2",
        linewidth=1.3, edgecolor=GRAY, facecolor="none", ls=(0, (5, 3))))
    ax.text(x0 + 0.18, (y0 + y1) / 2, r"$L$ blocks", rotation=90, ha="left",
            va="center", fontsize=13, color="black")

    # residual stream: embedding -> (+) -> (+) -> final norm
    _wire(ax, [(xs, y_emb + half), (xs, y_mix - r)])
    _wire(ax, [(xs, y_mix + r), (xs, y_mlp - r)])
    _go(ax, (xs, y_mlp + r), (xs, y_norm - half))
    _op(ax, xs, y_mix, r"$+$", r=r)
    _op(ax, xs, y_mlp, r"$+$", r=r)

    # each sub-layer: branch -> LayerNorm -> mixer or MLP -> back into (+)
    for y_ln, y_sub, label, color, w in [
            (y_ln1, y_mix, "mixer", TEAL, 2.1),
            (y_ln2, y_mlp, r"MLP: $d \to 4d$, GELU, $4d \to d$", PURPLE,
             4.4)]:
        _dot(ax, xs, y_ln)
        _go(ax, (xs, y_ln), (xb - 1.05, y_ln))
        _box(ax, xb, y_ln, 2.1, bh, "LayerNorm", GRAY, fontsize=13,
             weight="normal")
        _go(ax, (xb, y_ln + half), (xb, y_sub - half))
        _box(ax, xb, y_sub, w, bh, label, color, fontsize=13,
             weight="normal")
        _go(ax, (xb - w / 2, y_sub), (xs + r, y_sub))

    # the mixers this chapter trains on text, outside the frame
    ax.text(x1 + 0.25, y_mix, "mixer:\nRNN, LSTM, GRU,\nminGRU, Mamba,\n"
            "Gated DeltaNet,\nor attention", ha="left", va="center",
            fontsize=12.5, color="black", linespacing=1.3)

    _box(ax, xs, y_norm, 3.4, bh, "LayerNorm", GRAY, fontsize=13,
         weight="normal")
    _box(ax, xs, y_head, 3.4, bh, r"linear head, $d \to 256$", BLUE,
         fontsize=13, weight="normal")
    _box(ax, xs, y_out, 3.4, bh, "next-byte logits", BLUE, fontsize=13,
         weight="normal")
    _go(ax, (xs, y_norm + half), (xs, y_head - half))
    _go(ax, (xs, y_head + half), (xs, y_out - half))

    ax.set_xlim(0.45, 10.9)
    ax.set_ylim(0.0, 11.7)
    fl.save(fig, "mdl-modernrnn-seqlm-backbone")


def fig_recall_grid():
    """The long-range recall test as a 2x2 design: layer (minGRU, S4D) by
    initial timescales (short, long).  Each cell gives the slowest initial
    half-life and the fraction of a stored value that the slowest state
    keeps over the 1,000-step delay, computed from the initializations."""
    T = 1000
    ln2 = np.log(2)
    # slowest per-step decay of each configuration (rate -1 for S4D)
    cells = {
        ("minGRU", "short"): ("gates start near 1/2", 0.5),
        ("minGRU", "long"): (r"init_timescales, $\Delta \in [0.001, 0.1]$",
                             np.exp(-1e-3)),
        ("S4D", "short"): (r"$\Delta \in [0.1, 1]$", np.exp(-0.1)),
        ("S4D", "long"): (r"default $\Delta \in [0.001, 0.1]$",
                          np.exp(-1e-3)),
    }

    def sci(v):  # m x 10^k for tiny values, two decimals otherwise
        if v < 1e-3:
            e = int(np.floor(np.log10(v)))
            return rf"${v / 10.0 ** e:.0f} \times 10^{{{e}}}$"
        return f"{v:.2f}"

    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    ax.set_aspect("equal")
    ax.axis("off")
    cw, chh = 4.1, 1.75               # cell size
    xc = {"short": 3.0, "long": 3.0 + cw + 0.3}
    yr = {"minGRU": 2.55, "S4D": 0.55}
    col_color = {"short": ORANGE, "long": TEAL}
    for (row, col), (init, a) in cells.items():
        x, y = xc[col], yr[row]
        _box(ax, x, y, cw, chh, "", col_color[col], lw=1.6)
        half_life = ln2 / -np.log(a)
        hl = "1 step" if half_life < 1.5 else f"{half_life:.0f} steps"
        ax.text(x, y + 0.52, init, ha="center", va="center", fontsize=12.5,
                color="black")
        ax.text(x, y + 0.0, f"slowest half-life: {hl}", ha="center",
                va="center", fontsize=12.5, color="black")
        ax.text(x, y - 0.52, f"kept after {T:,} steps: " + sci(a ** T),
                ha="center", va="center", fontsize=12.5, color="black")
    for row, y in yr.items():
        ax.text(xc["short"] - cw / 2 - 0.25, y, row, ha="right", va="center",
                fontsize=14, color="black")
    ax.text(xc["short"], yr["minGRU"] + chh / 2 + 0.3,
            "short initial timescales", ha="center", va="bottom",
            fontsize=14, color=ORANGE_TXT)
    ax.text(xc["long"], yr["minGRU"] + chh / 2 + 0.3,
            "long initial timescales", ha="center", va="bottom",
            fontsize=14, color=TEAL_TXT)
    ax.set_xlim(-0.3, xc["long"] + cw / 2 + 0.1)
    ax.set_ylim(yr["S4D"] - chh / 2 - 0.1, yr["minGRU"] + chh / 2 + 0.85)
    fl.save(fig, "mdl-modernrnn-recall-grid")


def fig_gla_layer():
    """GatedLinearAttention.forward: q, k, v from one projection, a short
    causal convolution and SiLU, unit-length q and k; a decay per head from
    its own projection; the chunked matrix memory; RMS normalization per
    head; a SiLU gate; the output projection."""
    fig, ax = plt.subplots(figsize=(8.6, 8.4))
    ax.set_aspect("equal")
    ax.axis("off")

    xm, xa, xg = 2.75, 6.4, 8.7       # memory branch, decay, gate
    bh, half = 0.66, 0.33
    mh = 1.05                         # memory box height
    y_in, y_bus = 0.45, 1.35
    y_qkv, y_conv, y_silu, y_norm = 2.2, 3.45, 4.6, 5.75
    y_mem = 7.1
    y_rms, y_mul, y_out, y_top = 8.45, 9.5, 10.6, 11.85
    y_dec = 4.0                       # decay box
    xmid = (xm + xg) / 2

    _box(ax, xmid, y_in, 4.0, bh, r"input sequence $(T, d)$", BLUE,
         fontsize=13, weight="normal")
    _wire(ax, [(xmid, y_in + half), (xmid, y_bus)])
    _wire(ax, [(xm, y_bus), (xg, y_bus)])
    _dot(ax, xmid, y_bus)

    # memory branch: projection, conv, SiLU, heads with unit-length q and k
    _go(ax, (xm, y_bus), (xm, y_qkv - half))
    _box(ax, xm, y_qkv, 3.2, bh, r"$\mathbf{W}_{qkv}$: $d \to 3d$", BLUE,
         fontsize=13, weight="normal")
    _go(ax, (xm, y_qkv + half), (xm, y_conv - half))
    _box(ax, xm, y_conv, 3.2, bh, "causal conv, width 4", ORANGE,
         fontsize=12.5, weight="normal")
    _go(ax, (xm, y_conv + half), (xm, y_silu - 0.4))
    _op(ax, xm, y_silu, "SiLU", r=0.40, fontsize=11)
    _go(ax, (xm, y_silu + 0.4), (xm, y_norm - half))
    _box(ax, xm, y_norm, 3.6, bh,
         r"heads; unit-length $\mathbf{q}_t, \mathbf{k}_t$", GRAY,
         fontsize=12.5, weight="normal")
    _go(ax, (xm, y_norm + half), (xm, y_mem - mh / 2))

    # decay branch: one scalar per head, from its own projection
    _dot(ax, xa, y_bus)
    _box(ax, xa, y_dec, 2.9, 1.15,
         "decay per head\n"
         r"$a_{t,h} = \sigma(\mathbf{w}_h^\top \mathbf{x}_t + b_h)$",
         GREEN, fontsize=12.5, weight="normal")
    _go(ax, (xa, y_bus), (xa, y_dec - 1.15 / 2))
    _wire(ax, [(xa, y_dec + 1.15 / 2), (xa, y_mem)])
    _go(ax, (xa, y_mem), (xm + 2.3, y_mem))

    # the matrix memory of every head
    _box(ax, xm, y_mem, 4.6, mh,
         r"$\mathbf{S}_t = a_t \mathbf{S}_{t-1} + \mathbf{v}_t"
         r" \mathbf{k}_t^\top$" "\n"
         r"$\mathbf{y}_t = \mathbf{S}_t \mathbf{q}_t$, per head, in chunks",
         TEAL, fontsize=12.5, weight="normal")
    _go(ax, (xm, y_mem + mh / 2), (xm, y_rms - half))
    _box(ax, xm, y_rms, 3.2, bh, "RMS norm per head", GRAY, fontsize=12.5,
         weight="normal")
    _wire(ax, [(xm, y_rms + half), (xm, y_mul)])
    _go(ax, (xm, y_mul), (xmid - 0.27, y_mul))

    # gate branch
    _go(ax, (xg, y_bus), (xg, y_qkv - half))
    _box(ax, xg, y_qkv, 1.7, bh, r"$\mathbf{W}_g$: $d \to d$", BLUE,
         fontsize=13, weight="normal")
    _go(ax, (xg, y_qkv + half), (xg, y_silu - 0.4))
    _op(ax, xg, y_silu, "SiLU", r=0.40, fontsize=11)
    _wire(ax, [(xg, y_silu + 0.4), (xg, y_mul)])
    _go(ax, (xg, y_mul), (xmid + 0.27, y_mul))
    ax.text(xg + 0.2, (y_silu + y_mul) / 2, "gate", fontsize=13,
            color="black", ha="left", va="center")

    _op(ax, xmid, y_mul, r"$\odot$", r=0.27)
    _go(ax, (xmid, y_mul + 0.27), (xmid, y_out - half))
    _box(ax, xmid, y_out, 4.0, bh, r"output projection $d \to d$", BLUE,
         fontsize=13, weight="normal")
    _go(ax, (xmid, y_out + half), (xmid, y_top - half))
    _box(ax, xmid, y_top, 4.0, bh, r"output sequence $(T, d)$", BLUE,
         fontsize=13, weight="normal")

    ax.set_xlim(0.35, 9.85)
    ax.set_ylim(0.0, 12.3)
    fl.save(fig, "mdl-modernrnn-gla-layer")


def fig_gated_delta_modes():
    """One step of eq_dn-gated on a 2 x 2 memory, in three regimes of the
    two gates.  Arrows are the reads S k (along the key k = e_1, blue) and
    S k_perp (along an orthogonal key, purple), before (dashed) and after
    (solid) the step; the new value v is the orange dot.  The images are
    computed by applying the matrices: S_t k = a (1 - beta) S k + beta v and
    S_t k_perp = a S k_perp."""
    S = np.array([[1.05, 0.25], [0.35, 1.0]])      # columns: S k, S k_perp
    k = np.array([1.0, 0.0])
    v = np.array([-0.9, 0.6])
    modes = [("flush", 0.05, 1.0), ("overwrite", 0.98, 1.0),
             ("accumulate", 0.98, 0.2)]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.3))
    lim = ((-1.2, 1.35), (-0.25, 1.25))
    for ax, (name, a, beta) in zip(axes, modes):
        S_new = a * S @ (np.eye(2) - beta * np.outer(k, k)) \
            + beta * np.outer(v, k)
        fl.axis_cross(ax, lim[0], lim[1], color="black")
        for col, c in [(0, BLUE), (1, PURPLE)]:
            fl.arrow(ax, (0, 0), S[:, col], color=c, lw=1.6, ls="--",
                     alpha=0.55, mut=12)
            if np.linalg.norm(S_new[:, col]) > 0.2:
                fl.arrow(ax, (0, 0), S_new[:, col], color=c, lw=2.6, mut=15)
            else:  # Too short for an arrow: mark the point
                ax.plot(*S_new[:, col], "o", color=c, ms=6, zorder=5)
        ax.plot(*v, "o", color=ORANGE, ms=9, zorder=6)
        ax.text(v[0] - 0.1, v[1], r"$\mathbf{v}_t$", ha="right",
                va="center", fontsize=15, color="black")
        ax.set_title(f"{name}\n" rf"$a_t = {a:g},\ \beta_t = {beta:g}$",
                     fontsize=14, linespacing=1.4)
        fl.clean_axes(ax, lim=lim, hide=True)
    axes[1].text(0.5, -0.04,
                 r"reads along the key $\mathbf{k}_t$ (blue) and along an "
                 "orthogonal key (purple),\nbefore the step (dashed) and "
                 "after it (solid)", ha="center", va="top", fontsize=12.5,
                 color="black", transform=axes[1].transAxes)
    fl.save(fig, "mdl-modernrnn-gated-delta-modes")


def fig_selective_ssm():
    """The selective SSM of eq_selective_ssm for one step, drawn by shape:
    the H-dimensional input x_t yields the step sizes Delta_t (H), the write
    vector B_t (N) and the read vector C_t (N); the state is H x N, one row
    of N states per channel; the output y_t has H entries."""
    H, N = 6, 4
    s = 0.5                           # cell size
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    ax.set_aspect("equal")
    ax.axis("off")

    def strip(x0, y0, n, vertical, color, fill=0.18):
        """n cells from (x0, y0) downward (vertical) or rightward."""
        for i in range(n):
            x = x0 if vertical else x0 + i * s
            y = y0 - i * s if vertical else y0
            ax.add_patch(plt.Rectangle((x, y), s, s, facecolor=color,
                                       alpha=fill, edgecolor="none"))
            ax.add_patch(plt.Rectangle((x, y), s, s, facecolor="none",
                                       edgecolor=color, lw=1.3))

    top = 3.4                         # lower edge of the top row
    bottom = top - (H - 1) * s        # lower edge of the bottom row
    y_mid = (bottom + top + s) / 2
    gap = 0.9                         # between the state and B_t, C_t
    x_in, x_d, x_g = 0.4, 4.6, 5.2    # x_t, Delta_t, state
    x_y = x_g + N * s + 1.5           # y_t
    y_b = top + s + gap               # B_t row
    y_c = bottom - gap - s            # C_t row
    xc = x_g + N * s / 2              # centre of the state

    strip(x_in, top, H, True, BLUE)
    strip(x_d, top, H, True, GREEN)
    for i in range(H):
        strip(x_g, top - i * s, N, False, TEAL, fill=0.10)
    strip(x_g, y_b, N, False, ORANGE)
    strip(x_g, y_c, N, False, PURPLE)
    strip(x_y, top, H, True, BLUE)
    for x, lbl in [(x_in, r"$\mathbf{x}_t$"), (x_d, r"$\boldsymbol{\Delta}_t$"),
                   (x_y, r"$\mathbf{y}_t$")]:
        ax.text(x + s / 2, top + s + 0.12, lbl, ha="center", va="bottom",
                fontsize=15, color="black")
    ax.text(x_g + N * s + 0.15, y_b + s / 2, r"$\mathbf{B}_t$", ha="left",
            va="center", fontsize=15, color="black")
    ax.text(x_g + N * s + 0.15, y_c + s / 2, r"$\mathbf{C}_t$", ha="left",
            va="center", fontsize=15, color="black")
    ax.text(x_g + N * s + 0.75, y_mid - 0.15, "state\n" r"$H \times N$",
            ha="center", va="top", fontsize=12.5, color="black")

    # the three projections of x_t
    x0 = x_in + s + 0.1
    _go(ax, (x0, top + s / 2), (x_g - 0.12, y_b + s / 2), color=ORANGE,
        lw=1.5, mut=11)
    _go(ax, (x0, y_mid), (x_d - 0.12, y_mid), color=GREEN, lw=1.5, mut=11)
    _go(ax, (x0, bottom + s / 2), (x_g - 0.12, y_c + s / 2), color=PURPLE,
        lw=1.5, mut=11)
    ax.text(2.6, 4.55, r"$\mathbf{W}_B$", ha="center", va="bottom",
            fontsize=14, color="black")
    ax.text(2.6, 1.6, r"$\mathbf{W}_C$", ha="center", va="top",
            fontsize=14, color="black")
    ax.text((x0 + x_d) / 2, y_mid + 0.1,
            r"$\mathrm{softplus}(\mathbf{W}_\Delta \mathbf{x}_t"
            r" + \mathbf{b}_\Delta)$", ha="center", va="bottom",
            fontsize=13, color="black")

    # write into the state, read out of it, output
    _go(ax, (xc, y_b - 0.04), (xc, top + s + 0.06), color=ORANGE, lw=1.5,
        mut=11)
    ax.text(xc + 0.12, top + s + gap / 2, "write", ha="left",
            va="center", fontsize=12.5, color="black")
    _go(ax, (xc, y_c + s + 0.04), (xc, bottom - 0.06), color=PURPLE, lw=1.5,
        mut=11)
    ax.text(xc + 0.12, bottom - gap / 2, "read", ha="left", va="center",
            fontsize=12.5, color="black")
    _go(ax, (x_g + N * s + 0.12, y_mid), (x_y - 0.12, y_mid), color=GRAY,
        lw=1.5, mut=11)

    # what one step computes, per channel c (one row of the state)
    xe = x_y + s + 0.35
    ax.text(xe, y_mid + 0.5,
            r"$\mathbf{h}_t^{(c)} = e^{\Delta_{t,c}\boldsymbol{\lambda}}"
            r" \odot \mathbf{h}_{t-1}^{(c)} + \Delta_{t,c}\, x_{t,c}\,"
            r"\mathbf{B}_t$", ha="left", va="center", fontsize=14,
            color="black")
    ax.text(xe, y_mid - 0.5,
            r"$y_{t,c} = \mathbf{C}_t^\top \mathbf{h}_t^{(c)}"
            r" + D_c\, x_{t,c}$", ha="left", va="center", fontsize=14,
            color="black")
    ax.set_xlim(0.2, 15.3)
    ax.set_ylim(y_c - 0.2, y_b + s + 0.2)
    fl.save(fig, "mdl-modernrnn-selective-ssm")

# =========================================================================== #
# Driver                                                                      #
# =========================================================================== #

FIGURES = [
    fig_lstm_cell,
    fig_gru_cell,
    fig_encoder_decoder,
    fig_seq2seq,
    fig_scan_tree,
    fig_ssm_discretization,
    fig_ssm_views,
    fig_selective_copy,
    fig_mamba_block,
    fig_delta_geometry,
    fig_seqlm_backbone,
    fig_recall_grid,
    fig_gla_layer,
    fig_gated_delta_modes,
    fig_selective_ssm,
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
