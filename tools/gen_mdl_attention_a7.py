#!/usr/bin/env python3
"""House-style replacements for the three legacy hand-drawn SVGs in
``chapter_attention`` (``img/qkv.svg``, ``img/multi-head-attention.svg``,
``img/cnn-rnn-self-attention.svg``), which were stylistically inconsistent with
the seven house-style ``mdl-attention-*`` figures.

Figures:
  mdl-attention-soft-lookup.svg -- a *worked* soft lookup (A1,
      queries-keys-values.md): one query scores three keys, a softmax turns the
      scores into weights that visibly sum to one, and the output is the
      matching convex combination of the three values -- with actual numbers,
      and a geometric panel showing the mixture inside the triangle of values.
  mdl-attention-multi-head.svg -- a clean multi-head-attention block diagram
      (A3, multihead-attention.md): shared Q/K/V feed h parallel heads, each
      with its own projections and attention, whose outputs are concatenated
      and passed through one output projection.
  mdl-attention-cnn-rnn-self-attention.svg -- the CNN / RNN / self-attention
      connectivity comparison (A5, attention-at-scale.md): local window vs.
      sequential chain vs. all-to-all, as a historical schematic.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_attention_a7.py

All figures are written to ``img/mdl-attention-<id>.svg``.  Deterministic (no
random draws, no timestamps), hence byte-idempotent via the shared ``save()``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT

from matplotlib.patches import Circle, FancyBboxPatch, Rectangle


# --------------------------------------------------------------------------- #
# Shared little schematic helpers (rounded box, op circle, plain node).       #
# --------------------------------------------------------------------------- #

def _box(ax, cx, cy, w, h, text, color, fontsize=13, text_color="black",
         weight="normal", lw=1.7, ls="-", fc_alpha=0.12):
    """A rounded box: faint colour fill + solid coloured edge, centred text."""
    x, y = cx - w / 2, cy - h / 2
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=0, edgecolor="none", facecolor=color, alpha=fc_alpha,
        zorder=2))
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=lw, edgecolor=color, facecolor="none", linestyle=ls,
        zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            color=text_color, fontweight=weight, zorder=4)


def _dot(ax, cx, cy, r=0.06, color=GRAY):
    ax.add_patch(Circle((cx, cy), r, facecolor=color, edgecolor="none",
                        zorder=5))


# =========================================================================== #
# 1. mdl-attention-soft-lookup  (redesign of qkv.svg)                         #
# =========================================================================== #

def fig_soft_lookup() -> None:
    # scores -> softmax weights (real arithmetic, rounded only for display)
    scores = np.array([2.0, 1.0, 0.1])
    w = np.exp(scores) / np.exp(scores).sum()          # 0.659, 0.242, 0.099
    wd = np.round(w, 2)                                 # 0.66, 0.24, 0.10 (sum 1)
    cols = [BLUE, ORANGE, GREEN]

    # 2-D values; the output is their weighted average (a convex combination)
    V = np.array([[1.0, 2.6], [0.5, 0.6], [3.3, 1.1]])
    o = w @ V

    fig, (axp, axg) = plt.subplots(
        1, 2, figsize=(12.2, 4.3),
        gridspec_kw={"width_ratios": [1.95, 1.0]})

    # ---------------------------------------------------------------- (a) --- #
    # pipeline: q -> keys -> softmax weights (bars) -> values -> output
    axp.set_aspect("equal")
    axp.axis("off")
    ys = [3.35, 2.05, 0.75]
    xq, xk = 0.85, 2.65
    xbar0, ulen = 4.15, 1.8                              # unit-bar origin/length
    xval, xo = 7.7, 9.95

    # query feeds every key
    _box(axp, xq, ys[1], 1.0, 0.82, r"$\mathbf{q}$", GRAY, fontsize=15)
    for k, (yi, ci, s, ww, wdi) in enumerate(zip(ys, cols, scores, w, wd), 1):
        # key
        _box(axp, xk, yi, 0.98, 0.74, rf"$\mathbf{{k}}_{{{k}}}$", ci,
             fontsize=14)
        fl.arrow(axp, (xq + 0.5, ys[1]), (xk - 0.52, yi), color=GRAY, lw=1.6,
                 mut=12)
        # score s_i above the key->weights arrow
        axp.text((xk + xbar0) / 2 + 0.05, yi + 0.32, rf"$s={s:.1f}$",
                 ha="center", va="bottom", fontsize=12, color="black")
        fl.arrow(axp, (xk + 0.52, yi), (xbar0 - 0.06, yi), color=GRAY, lw=1.6,
                 mut=12)
        # unit reference box + weight bar as a fraction of it; alpha ABOVE it
        axp.add_patch(Rectangle((xbar0, yi - 0.17), ulen, 0.34, facecolor="none",
                                edgecolor=GRAY, lw=1.0, ls="--", zorder=2))
        axp.add_patch(Rectangle((xbar0, yi - 0.17), ulen * ww, 0.34,
                                facecolor=ci, edgecolor="none", alpha=0.85,
                                zorder=3))
        axp.text(xbar0 + ulen / 2, yi + 0.28,
                 rf"$\alpha_{{{k}}}={wdi:.2f}$", ha="center", va="bottom",
                 fontsize=12.5, color="black")
        # unweighted feed into the value
        fl.arrow(axp, (xbar0 + ulen + 0.08, yi), (xval - 0.52, yi), color=GRAY,
                 lw=1.5, mut=12)
        # value
        _box(axp, xval, yi, 0.98, 0.74, rf"$\mathbf{{v}}_{{{k}}}$", ci,
             fontsize=14)
        # value contributes to the output, weighted (arrow thickness == alpha)
        fl.arrow(axp, (xval + 0.5, yi), (xo - 0.82, ys[1]), color=ci,
                 lw=1.2 + 5.0 * ww, mut=13)

    # column headers / the softmax gate label
    axp.text(xbar0 + ulen / 2, 4.05, r"softmax: $\alpha_i \propto e^{\,s_i}$",
             ha="center", va="bottom", fontsize=13, color="black")
    # the weights sum to one -- shown by the shared unit box AND spelled out
    axp.text(xbar0 + ulen / 2, -0.05,
             r"weights sum to one:  "
             rf"$\alpha_1+\alpha_2+\alpha_3={wd[0]:.2f}+{wd[1]:.2f}"
             rf"+{wd[2]:.2f}=1$",
             ha="center", va="center", fontsize=12.5, color="black")

    # output box
    _box(axp, xo, ys[1], 1.55, 1.06,
         r"$\mathbf{o}$" "\n" r"$=\sum_i \alpha_i \mathbf{v}_i$", "black",
         fontsize=13, fc_alpha=0.06)

    axp.set_xlim(0.1, 11.0)
    axp.set_ylim(-0.4, 4.4)

    # ---------------------------------------------------------------- (b) --- #
    # the output is a convex combination -> it lies inside the value triangle
    axg.set_aspect("equal")
    tri = np.vstack([V, V[0]])
    axg.plot(tri[:, 0], tri[:, 1], color=LIGHT, lw=1.4, zorder=1)
    lab_off = [(-0.05, 0.28), (-0.26, -0.18), (0.26, 0.04)]
    for vi, ci, off, k in zip(V, cols, lab_off, range(3)):
        axg.plot([o[0], vi[0]], [o[1], vi[1]], ls="--", color=ci, lw=1.3,
                 alpha=0.8, zorder=2)
        axg.plot(*vi, "o", color=ci, ms=11, zorder=4)
        axg.text(vi[0] + off[0], vi[1] + off[1],
                 rf"$\mathbf{{v}}_{{{k + 1}}}$", color=ci, fontsize=14,
                 ha="center", va="center")
        # weight label near each spoke, nudged toward the value
        m = o + 0.58 * (vi - o)
        axg.text(m[0], m[1], rf"${wd[k]:.2f}$", color=ci, fontsize=11.5,
                 ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none",
                           alpha=0.9))
    axg.plot(*o, "*", color="black", ms=15, zorder=5)
    axg.text(o[0] + 0.16, o[1] - 0.26, r"$\mathbf{o}$", color="black",
             fontsize=14, ha="left", va="top")
    axg.text(2.55, 2.72, "a convex combination:\n"
             r"$\mathbf{o}$ lies inside" "\nthe values",
             ha="center", va="center", fontsize=11.5, color="black")
    axg.set_xlim(-0.2, 4.0)
    axg.set_ylim(0.15, 3.15)
    axg.axis("off")

    fig.subplots_adjust(wspace=0.05)
    fl.save(fig, "mdl-attention-soft-lookup")


# =========================================================================== #
# 2. mdl-attention-multi-head  (redesign of multi-head-attention.svg)         #
# =========================================================================== #

def fig_multi_head() -> None:
    fig, ax = plt.subplots(figsize=(9.2, 6.6))
    ax.set_aspect("equal")
    ax.axis("off")

    y_in = 0.7                      # Q, K, V inputs
    y_bus = 1.62                    # fan-out bus
    y_proj = 2.75                   # per-head projections
    y_attn = 4.05                   # per-head attention
    y_cat = 5.35                    # concat
    y_op = 6.55                     # output projection
    y_top = 7.75                    # output

    hx = [2.55, 6.65]               # two head columns
    x_ell = (hx[0] + hx[1]) / 2     # ellipsis between the heads

    # --- Q, K, V inputs, centred under the ellipsis, on three buses --------- #
    qkv = [("Q", BLUE), ("K", BLUE), ("V", BLUE)]
    x_qkv = [x_ell - 0.95, x_ell, x_ell + 0.95]
    bus_y = [y_bus + 0.18, y_bus, y_bus - 0.18]
    for (lab, c), xin, by in zip(qkv, x_qkv, bus_y):
        _box(ax, xin, y_in, 0.8, 0.6, rf"$\mathbf{{{lab}}}$", c, fontsize=13)
        fl.arrow(ax, (xin, y_in + 0.3), (xin, by), color=GRAY, lw=1.5, mut=11)
        # bus spanning both heads at this stream's height
        ax.plot([hx[0] - 0.62, hx[1] + 0.62], [by, by], color=GRAY, lw=1.4,
                zorder=1)

    # --- two heads: project -> attention ------------------------------------ #
    for h, cx in enumerate(hx, start=1):
        # taps from the three buses up into the projection box
        for xin, by in zip([cx - 0.62, cx, cx + 0.62], bus_y):
            _dot(ax, xin, by, r=0.05)
            fl.arrow(ax, (xin, by), (xin, y_proj - 0.3), color=GRAY, lw=1.4,
                     mut=10)
        _box(ax, cx, y_proj, 2.5, 0.66,
             rf"project  $\mathbf{{W}}_{{{h}}}^{{Q}},"
             rf"\mathbf{{W}}_{{{h}}}^{{K}},\mathbf{{W}}_{{{h}}}^{{V}}$",
             BLUE, fontsize=11.5)
        fl.arrow(ax, (cx, y_proj + 0.33), (cx, y_attn - 0.33), color=GRAY,
                 lw=1.7, mut=12)
        _box(ax, cx, y_attn, 2.5, 0.78,
             "scaled dot-product\nattention", GREEN, fontsize=11.5)
        ax.text(cx, y_attn - 0.66, rf"head {h}", ha="center", va="top",
                fontsize=12, color="black")
        # attention -> concat
        fl.arrow(ax, (cx, y_attn + 0.39), (cx, y_cat - 0.33), color=GRAY,
                 lw=1.7, mut=12)

    # ellipsis + "h heads" note between the columns (in the clear gap below
    # the concat box, centred on the ellipsis column)
    ax.text(x_ell, y_attn, r"$\cdots$", ha="center", va="center", fontsize=20,
            color="black")
    ax.text(x_ell, 0.5 * (y_attn + y_cat) + 0.05, r"$h$ heads in parallel",
            ha="center", va="center", fontsize=12.5, color="black")

    # --- concat -> output projection -> output ------------------------------ #
    _box(ax, x_ell, y_cat, 6.0, 0.64, "concatenate head outputs", ORANGE,
         fontsize=13)
    fl.arrow(ax, (x_ell, y_cat + 0.32), (x_ell, y_op - 0.33), color=GRAY,
             lw=1.9, mut=13)
    _box(ax, x_ell, y_op, 3.4, 0.66, r"output projection  $\mathbf{W}^{O}$",
         BLUE, fontsize=12.5)
    fl.arrow(ax, (x_ell, y_op + 0.33), (x_ell, y_top - 0.3), color=GRAY,
             lw=1.9, mut=13)
    _box(ax, x_ell, y_top, 2.4, 0.6, "output", GRAY, fontsize=13)

    ax.set_xlim(0.6, 8.7)
    ax.set_ylim(0.2, 8.25)
    fl.save(fig, "mdl-attention-multi-head")


# =========================================================================== #
# 3. mdl-attention-cnn-rnn-self-attention                                     #
#    (redesign of cnn-rnn-self-attention.svg)                                 #
# =========================================================================== #

def _seq_nodes(ax, n=5):
    """Draw the shared row of n inputs (white) and n outputs (light blue)."""
    xin = np.arange(n)
    y_in, y_out = 0.0, 2.0
    for x in xin:
        ax.add_patch(Circle((x, y_out), 0.24, facecolor="#bcd6ef",
                            edgecolor="black", lw=1.1, zorder=4))
        ax.add_patch(Circle((x, y_in), 0.24, facecolor="white",
                            edgecolor="black", lw=1.1, zorder=4))
        ax.text(x, y_in, rf"$x_{{{x + 1}}}$", ha="center", va="center",
                fontsize=11, zorder=5)
    return xin, y_in, y_out


def _edge(ax, x0, y0, x1, y1, color=LIGHT, lw=1.0, highlight=False):
    if highlight:
        fl.arrow(ax, (x0, y0 + 0.24), (x1, y1 - 0.24), color=color, lw=lw,
                 mut=11)
    else:
        ax.plot([x0, x1], [y0 + 0.24, y1 - 0.24], color=color, lw=lw,
                zorder=1)


def fig_cnn_rnn_self_attention() -> None:
    n = 5
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.4))
    focus = 2                      # the output whose connections we highlight

    # ---- CNN: each output sees a local window (width 3) -------------------- #
    ax = axes[0]
    xin, y_in, y_out = _seq_nodes(ax, n)
    for j in xin:
        lo, hi = max(0, j - 1), min(n - 1, j + 1)
        for i in range(lo, hi + 1):
            hl = (j == focus)
            _edge(ax, i, y_in, j, y_out, color=BLUE if hl else LIGHT,
                  lw=1.8 if hl else 1.0, highlight=hl)
    ax.set_title("CNN: local window", fontsize=13.5, color="black")

    # ---- RNN: a left-to-right recurrent chain ------------------------------ #
    ax = axes[1]
    xin, y_in, y_out = _seq_nodes(ax, n)
    for j in xin:                                   # input -> its own step
        _edge(ax, j, y_in, j, y_out, color=GRAY, lw=1.3)
    for j in xin[:-1]:                              # recurrent edge along top
        fl.arrow(ax, (j + 0.24, y_out), (j + 1 - 0.24, y_out), color=BLUE,
                 lw=1.8, mut=11)
    ax.set_title("RNN: sequential chain", fontsize=13.5, color="black")

    # ---- self-attention: all-to-all ---------------------------------------- #
    ax = axes[2]
    xin, y_in, y_out = _seq_nodes(ax, n)
    for j in xin:
        for i in xin:
            hl = (j == focus)
            _edge(ax, i, y_in, j, y_out, color=BLUE if hl else LIGHT,
                  lw=1.6 if hl else 0.8, highlight=hl)
    ax.set_title("self-attention: all-to-all", fontsize=13.5, color="black")

    for ax in axes:
        ax.set_xlim(-0.7, n - 0.3)
        ax.set_ylim(-0.7, 2.7)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.subplots_adjust(wspace=0.08)
    fl.save(fig, "mdl-attention-cnn-rnn-self-attention")


ALL_FIGURES = [fig_soft_lookup, fig_multi_head, fig_cnn_rnn_self_attention]


def main():
    start = len(fl.WRITTEN)
    for f in ALL_FIGURES:
        f()
    for path in fl.WRITTEN[start:]:
        print(f"wrote {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
