#!/usr/bin/env python3
"""House-style replacements for the three legacy hand-drawn SVGs in
``chapter_attention`` (``img/qkv.svg``, ``img/multi-head-attention.svg``,
``img/cnn-rnn-self-attention.svg``), which were stylistically inconsistent with
the seven house-style ``mdl-attention-*`` figures.

Figures:
  mdl-attention-soft-lookup.svg -- a *worked* soft lookup (A1,
      queries-keys-values.md): the stated query q = (1, 1) scores three stated
      keys by -||q - k||^2, a softmax turns the scores into weights that
      visibly sum to one, and the output is the matching convex combination of
      three stated values; a geometric panel shows the mixture inside the
      triangle of values. All numbers are computed from q, K and V.
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
    """One soft lookup with stated numbers (queries-keys-values.md, fig_qkv).

    Query q = (1, 1); keys k1 = (1, 1), k2 = (2, 1), k3 = (0, 0); score
    s_i = -||q - k_i||^2 = 0, -1, -2 (the rule of the scalar example); softmax
    weights 0.67, 0.24, 0.09; values v1 = (1, 3), v2 = (0, 0), v3 = (4, 1);
    output o = 0.67 v1 + 0.24 v2 + 0.09 v3 = (1.0, 2.1). Every number in the
    figure is computed here from q, K and V, so the figure, the text and the
    caption show one computation.
    """
    q = np.array([1.0, 1.0])
    K = np.array([[1.0, 1.0], [2.0, 1.0], [0.0, 0.0]])
    V = np.array([[1.0, 3.0], [0.0, 0.0], [4.0, 1.0]])
    scores = -((q - K) ** 2).sum(1) + 0.0               # 0, -1, -2 (no -0)
    w = np.exp(scores) / np.exp(scores).sum()          # 0.665, 0.245, 0.090
    wd = np.round(w, 2)                                 # 0.67, 0.24, 0.09
    assert abs(wd.sum() - 1) < 1e-9                     # displayed sum is 1
    o = w @ V                                           # (1.03, 2.09)
    cols = [BLUE, ORANGE, GREEN]

    def vec(x):
        return "(" + ", ".join(f"{c:g}" for c in x) + ")"

    fig, (axp, axg) = plt.subplots(
        1, 2, figsize=(12.4, 4.6),
        gridspec_kw={"width_ratios": [2.15, 1.0]})

    # ---------------------------------------------------------------- (a) --- #
    # pipeline: q -> keys -> scores -> softmax weights (bars) -> values -> o
    axp.set_aspect("equal")
    axp.axis("off")
    ys = [3.45, 2.05, 0.65]
    bw, bh = 1.25, 0.98                                  # key/value box size
    xq, xk = 0.85, 2.75
    xbar0, ulen = 4.75, 1.6                              # unit-bar origin/length
    xval, xo = 7.65, 10.05
    o_h = 1.05
    land = [ys[1] + 0.25 * o_h, ys[1], ys[1] - 0.25 * o_h]

    _box(axp, xq, ys[1], 1.25, bh, r"$\mathbf{q}$" + "\n" + vec(q), GRAY,
         fontsize=14)
    for k, (yi, ci) in enumerate(zip(ys, cols), 1):
        s, ww, wdi = scores[k - 1], w[k - 1], wd[k - 1]
        _box(axp, xk, yi, bw, bh,
             rf"$\mathbf{{k}}_{{{k}}}$" + "\n" + vec(K[k - 1]), ci,
             fontsize=14)
        fl.arrow(axp, (xq + 0.64, ys[1]), (xk - bw / 2 - 0.04, yi),
                 color=GRAY, lw=1.6, mut=12)
        # score s_i above the key->weights arrow
        axp.text((xk + bw / 2 + xbar0) / 2, yi + 0.07, rf"$s_{{{k}}}={s:g}$",
                 ha="center", va="bottom", fontsize=13, color="black")
        fl.arrow(axp, (xk + bw / 2 + 0.04, yi), (xbar0 - 0.06, yi),
                 color=GRAY, lw=1.6, mut=12)
        # unit reference box + weight bar as a fraction of it; alpha above it
        axp.add_patch(Rectangle((xbar0, yi - 0.17), ulen, 0.34,
                                facecolor="none", edgecolor=GRAY, lw=1.0,
                                ls="--", zorder=2))
        axp.add_patch(Rectangle((xbar0, yi - 0.17), ulen * ww, 0.34,
                                facecolor=ci, edgecolor="none", alpha=0.85,
                                zorder=3))
        axp.text(xbar0 + ulen / 2, yi + 0.27,
                 rf"$\alpha_{{{k}}}={wdi:.2f}$", ha="center", va="bottom",
                 fontsize=13, color="black")
        fl.arrow(axp, (xbar0 + ulen + 0.08, yi), (xval - bw / 2 - 0.04, yi),
                 color=GRAY, lw=1.5, mut=12)
        _box(axp, xval, yi, bw, bh,
             rf"$\mathbf{{v}}_{{{k}}}$" + "\n" + vec(V[k - 1]), ci,
             fontsize=14)
        # value contributes to the output, weighted (arrow width ~ alpha)
        fl.arrow(axp, (xval + bw / 2 + 0.04, yi), (xo - 1.0, land[k - 1]),
                 color=ci, lw=1.2 + 5.0 * ww, mut=13)

    # column headers: the score rule and the softmax
    axp.text(xk + 0.55, 4.35,
             r"score $s_i=-\|\mathbf{q}-\mathbf{k}_i\|^2$",
             ha="center", va="bottom", fontsize=13, color="black")
    axp.text(xbar0 + ulen / 2 + 0.75, 4.35,
             r"softmax: $\alpha_i \propto e^{\,s_i}$",
             ha="center", va="bottom", fontsize=13, color="black")
    axp.text(xbar0 + ulen / 2, -0.35,
             r"$\alpha_1+\alpha_2+\alpha_3="
             rf"{wd[0]:.2f}+{wd[1]:.2f}+{wd[2]:.2f}=1$",
             ha="center", va="center", fontsize=13, color="black")

    _box(axp, xo, ys[1], 1.95, o_h,
         r"$\mathbf{o}=\Sigma_i\,\alpha_i \mathbf{v}_i$" + "\n"
         + rf"$\approx({o[0]:.1f},\ {o[1]:.1f})$", "black",
         fontsize=13.5, fc_alpha=0.06)

    axp.set_xlim(0.1, 11.1)
    axp.set_ylim(-0.6, 4.8)

    # ---------------------------------------------------------------- (b) --- #
    # the output is a convex combination -> it lies inside the value triangle
    axg.set_aspect("equal")
    tri = np.vstack([V, V[0]])
    axg.plot(tri[:, 0], tri[:, 1], color=LIGHT, lw=1.4, zorder=1)
    lab_pos = [((-0.2, 0.0), "right", "center"),
               ((-0.2, 0.0), "right", "center"),
               ((0.0, -0.25), "center", "top")]
    for vi, ci, (off, ha, va), k in zip(V, cols, lab_pos, range(3)):
        axg.plot([o[0], vi[0]], [o[1], vi[1]], ls="--", color=ci, lw=1.3,
                 alpha=0.8, zorder=2)
        axg.plot(*vi, "o", color=ci, ms=11, zorder=4)
        axg.text(vi[0] + off[0], vi[1] + off[1],
                 rf"$\mathbf{{v}}_{{{k + 1}}}\ ({wd[k]:.2f})$", color=ci,
                 fontsize=14, ha=ha, va=va)
    axg.plot(*o, "*", color="black", ms=15, zorder=5)
    axg.text(o[0] + 0.18, o[1] + 0.05, r"$\mathbf{o}$", color="black",
             fontsize=15, ha="left", va="bottom")
    axg.text(2.0, 3.45, "a convex combination:\n"
             r"$\mathbf{o}$ lies inside the triangle of values",
             ha="center", va="bottom", fontsize=13, color="black")
    axg.set_xlim(-1.3, 4.5)
    axg.set_ylim(-0.75, 4.3)
    axg.axis("off")

    fig.subplots_adjust(wspace=0.05)
    fl.save(fig, "mdl-attention-soft-lookup")


# =========================================================================== #
# 2. mdl-attention-multi-head  (redesign of multi-head-attention.svg)         #
# =========================================================================== #

def fig_multi_head() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.1))
    ax.set_aspect("equal")
    ax.axis("off")

    y_in = 1.55                     # Q, K, V labels under each head
    y_proj = 2.75                   # per-head projections
    y_attn = 4.3                    # per-head attention
    y_cat = 5.6                     # concat
    y_op = 6.8                      # output projection
    y_top = 8.0                     # output

    hx = [2.25, 6.95]               # two head columns
    x_ell = (hx[0] + hx[1]) / 2     # ellipsis between the heads
    bw = 3.75                       # width of the per-head boxes

    # --- every head reads the same Q, K, V: three short labelled arrows ---- #
    # straight into its projection box (no shared buses, no wire crossings)
    for cx in hx:
        for lab, xin in zip("QKV", [cx - 0.95, cx, cx + 0.95]):
            ax.text(xin, y_in, rf"$\mathbf{{{lab}}}$", ha="center",
                    va="center", fontsize=15, color="black")
            fl.arrow(ax, (xin, y_in + 0.24), (xin, y_proj - 0.36),
                     color=GRAY, lw=1.4, mut=10)

    # --- two heads: project -> attention ------------------------------------ #
    # the second column stands for head H (the ellipsis hides heads 2..H-1)
    for h, cx in zip(("1", "H"), hx):
        _box(ax, cx, y_proj, bw, 0.72,
             rf"$\mathbf{{W}}_{{{h}}}^{{(q)}},\ "
             rf"\mathbf{{W}}_{{{h}}}^{{(k)}},\ \mathbf{{W}}_{{{h}}}^{{(v)}}$",
             BLUE, fontsize=14)
        fl.arrow(ax, (cx, y_proj + 0.36), (cx, y_attn - 0.42), color=GRAY,
                 lw=1.7, mut=12)
        _box(ax, cx, y_attn, bw, 0.84,
             "scaled dot-product\nattention", ORANGE, fontsize=13.5)
        ax.text(cx - 0.18, (y_proj + 0.36 + y_attn - 0.42) / 2,
                rf"head ${h}$", ha="right", va="center", fontsize=13.5,
                color="black")
        # attention -> concat
        fl.arrow(ax, (cx, y_attn + 0.42), (cx, y_cat - 0.34), color=GRAY,
                 lw=1.7, mut=12)

    # ellipsis between the columns stands for heads 2..H-1
    ax.text(x_ell, y_attn, r"$\cdots$", ha="center", va="center",
            fontsize=22, color="black")

    # --- concat -> output projection -> output ------------------------------ #
    _box(ax, x_ell, y_cat, hx[1] - hx[0] + bw, 0.68,
         "concatenate head outputs", GRAY, fontsize=14)
    fl.arrow(ax, (x_ell, y_cat + 0.34), (x_ell, y_op - 0.36), color=GRAY,
             lw=1.9, mut=13)
    _box(ax, x_ell, y_op, 3.9, 0.72, r"output projection  $\mathbf{W}_o$",
         BLUE, fontsize=14)
    fl.arrow(ax, (x_ell, y_op + 0.36), (x_ell, y_top - 0.32), color=GRAY,
             lw=1.9, mut=13)
    _box(ax, x_ell, y_top, 2.4, 0.64, "output", GRAY, fontsize=14)

    ax.set_xlim(0.2, 9.0)
    ax.set_ylim(1.25, 8.45)
    fl.save(fig, "mdl-attention-multi-head")


# =========================================================================== #
# 3. mdl-attention-cnn-rnn-self-attention                                     #
#    (redesign of cnn-rnn-self-attention.svg)                                 #
# =========================================================================== #

R_NODE = 0.28                  # node radius


def _seq_nodes(ax, n=5):
    """Draw the shared row of n inputs (white) and n outputs (light blue)."""
    xin = np.arange(n)
    y_in, y_out = 0.0, 2.0
    for x in xin:
        ax.add_patch(Circle((x, y_out), R_NODE, facecolor=fl.T.BLUE.tint,
                            edgecolor="black", lw=1.1, zorder=4))
        ax.add_patch(Circle((x, y_in), R_NODE, facecolor="white",
                            edgecolor="black", lw=1.1, zorder=4))
        ax.text(x, y_in, rf"$x_{{{x + 1}}}$", ha="center", va="center",
                fontsize=13.5, zorder=5)
    return xin, y_in, y_out


def _edge(ax, x0, y0, x1, y1, color=LIGHT, lw=1.0, highlight=False):
    """Edge between two node centres, clipped radially to both circles.
    No arrowheads: flow is always input (bottom) -> output (top), and up to
    five heads converging on one small node only collide; the blue colour
    marks the highlighted edges instead."""
    d = np.array([x1 - x0, y1 - y0], float)
    d /= np.linalg.norm(d)
    a = (x0 + R_NODE * d[0], y0 + R_NODE * d[1])
    b = (x1 - R_NODE * d[0], y1 - R_NODE * d[1])
    ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw,
            zorder=2 if highlight else 1, solid_capstyle="butt")


def fig_cnn_rnn_self_attention() -> None:
    n = 5
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.9))
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
    # highlighted: everything that reaches output 3 -- inputs 1..3, each
    # entering its own step, carried forward one step at a time
    ax = axes[1]
    xin, y_in, y_out = _seq_nodes(ax, n)
    for j in xin:                                   # input -> its own step
        hl = (j <= focus)
        _edge(ax, j, y_in, j, y_out, color=BLUE if hl else LIGHT,
              lw=1.8 if hl else 1.0, highlight=hl)
    for j in xin[:-1]:                              # recurrent edge along top
        hl = (j < focus)
        fl.arrow(ax, (j + R_NODE, y_out), (j + 1 - R_NODE, y_out),
                 color=BLUE if hl else LIGHT, lw=1.8 if hl else 1.2, mut=11)
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

    # the two quantities of the cost table, one line under each panel
    notes = [r"path $\mathcal{O}(n/k)$, parallel",
             r"path $\mathcal{O}(n)$, sequential",
             r"path $\mathcal{O}(1)$, parallel"]
    for ax, note in zip(axes, notes):
        ax.text((n - 1) / 2, -0.62, note, ha="center", va="top",
                fontsize=13.5, color="black")
        ax.set_xlim(-0.7, n - 0.3)
        ax.set_ylim(-1.15, 2.6)
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
