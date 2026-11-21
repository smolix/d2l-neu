#!/usr/bin/env python3
"""Generate the illustrative figures for the "Attention" chapter sections A1
(queries-keys-values.md) and A2 (attention-scoring.md) in the shared house
style defined in ``gen_mdl_figures.py``.

Figures (both A2):

  * ``mdl-attention-shapes`` -- the shapes of softmax(Q K^T / sqrt(d)) V,
    with a causal mask on the weights.
  * ``mdl-attention-mask-algebra`` -- a padding mask AND a causal mask under
    broadcasting.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_attention_a1a2.py

All figures are written to ``img/mdl-attention-<id>.svg``.  Deterministic
(no random draws), hence byte-idempotent via the shared ``save()``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers
from matplotlib.patches import Rectangle

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT


# --------------------------------------------------------------------------- #
# A2: shapes of softmax(Q K^T / sqrt(d)) V, and the mask algebra.            #
# --------------------------------------------------------------------------- #

CELL = 0.42  # side of one matrix entry, in data units


def _matrix(ax, x, y, rows, cols, color, fill=None, label=None, shape=None,
            fontsize=16):
    """Draw a rows x cols matrix with its top-left corner at (x, y).

    ``fill`` is an optional (rows, cols) array of opacities in [0, 1]; NaN
    leaves an entry blank. The label sits in the middle, the shape below.
    """
    w, h = cols * CELL, rows * CELL
    if fill is not None:
        for i in range(rows):
            for j in range(cols):
                if not np.isnan(fill[i, j]):
                    ax.add_patch(Rectangle(
                        (x + j * CELL, y - (i + 1) * CELL), CELL, CELL,
                        facecolor=color, alpha=fill[i, j], edgecolor="none",
                        zorder=1))
    for i in range(1, rows):
        ax.plot([x, x + w], [y - i * CELL] * 2, color=LIGHT, lw=0.8,
                zorder=2)
    for j in range(1, cols):
        ax.plot([x + j * CELL] * 2, [y, y - h], color=LIGHT, lw=0.8,
                zorder=2)
    ax.add_patch(Rectangle((x, y - h), w, h, facecolor="none",
                           edgecolor=color, lw=1.8, zorder=3))
    if label:
        ax.text(x + w / 2, y - h / 2, label, ha="center", va="center",
                fontsize=fontsize, color="black", zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                          alpha=0.9))
    if shape:
        ax.text(x + w / 2, y - h - 0.15, shape, ha="center", va="top",
                fontsize=14, color="black")


def fig_attention_shapes():
    """softmax(Q K^T / sqrt(d)) V with the shape of every matrix.

    Row 1: Q (n x d) times K^T (d x m) gives the scores (n x m).
    Row 2, under the scores: the masked, row-normalized weights A (n x m)
    times V (m x d_v) give the output O (n x d_v). n = m = 6, d = d_v = 3
    are drawn; a causal mask blanks the upper triangle of A.
    """
    n, m, d, dv = 6, 6, 3, 3
    gap = 0.75
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    ax.set_aspect("equal")
    ax.axis("off")

    # deterministic "scores": a smooth pattern, no random draws
    ii, jj = np.meshgrid(np.arange(n), np.arange(m), indexing="ij")
    scores = np.cos(0.9 * ii - 0.7 * jj) + 0.35 * np.sin(1.3 * jj + ii)
    s_fill = 0.12 + 0.55 * (scores - scores.min()) / np.ptp(scores)
    causal = jj <= ii
    w = np.where(causal, np.exp(1.6 * scores), 0.0)
    w = w / w.sum(1, keepdims=True)
    a_fill = np.where(causal, 0.15 + 0.75 * w / w.max(), np.nan)

    top = 0.0
    mid = top - n * CELL / 2
    # row 1: Q x K^T = scores
    xq = 0.0
    _matrix(ax, xq, top, n, d, BLUE, fill=np.full((n, d), 0.18),
            label=r"$\mathbf{Q}$", shape=r"$n \times d$")
    ax.text(xq + d * CELL + gap / 2, mid, r"$\times$", ha="center",
            va="center", fontsize=18, color="black")
    xk = xq + d * CELL + gap
    _matrix(ax, xk, mid + d * CELL / 2, d, m, ORANGE,
            fill=np.full((d, m), 0.18), label=r"$\mathbf{K}^\top$",
            shape=r"$d \times m$")
    ax.text(xk + m * CELL + gap / 2, mid, r"$=$", ha="center", va="center",
            fontsize=18, color="black")
    xs = xk + m * CELL + gap
    _matrix(ax, xs, top, n, m, GRAY, fill=s_fill,
            label=r"$\mathbf{Q}\mathbf{K}^\top\!/\sqrt{d}$",
            shape=r"scores, $n \times m$", fontsize=15)

    # arrow scores -> A: mask, then softmax over each row
    y_a = top - n * CELL - 1.75
    xmid = xs + m * CELL / 2
    fl.arrow(ax, (xmid, top - n * CELL - 0.62), (xmid, y_a + 0.1),
             color=GRAY, lw=1.8, mut=14)
    ax.text(xmid - 0.25, (top - n * CELL - 0.62 + y_a) / 2,
            "mask, then softmax\nof each row", ha="right", va="center",
            fontsize=14, color="black")

    # row 2: A x V = O
    mid_a = y_a - n * CELL / 2
    _matrix(ax, xs, y_a, n, m, BLUE, fill=a_fill, label=r"$\mathbf{A}$",
            shape=r"weights, $n \times m$")
    ax.text(xs + m * CELL + gap / 2, mid_a, r"$\times$", ha="center",
            va="center", fontsize=18, color="black")
    xv = xs + m * CELL + gap
    _matrix(ax, xv, y_a, m, dv, GREEN, fill=np.full((m, dv), 0.18),
            label=r"$\mathbf{V}$", shape=r"$m \times d_v$")
    ax.text(xv + dv * CELL + gap / 2, mid_a, r"$=$", ha="center",
            va="center", fontsize=18, color="black")
    xo = xv + dv * CELL + gap
    _matrix(ax, xo, y_a, n, dv, "black", fill=np.full((n, dv), 0.08),
            label=r"$\mathbf{O}$", shape=r"$n \times d_v$")

    ax.set_xlim(-0.1, xo + dv * CELL + 0.1)
    ax.set_ylim(y_a - n * CELL - 0.7, top + 0.1)
    fl.save(fig, "mdl-attention-shapes")


def fig_mask_algebra():
    """padding (batch, 1, keys) AND causal (queries, keys) = composite.

    One sequence of length 6 with valid length 3. True entries are filled.
    The padding mask stores one row of keys; broadcasting repeats it for
    every query (the repeats are drawn faint).
    """
    n, valid = 6, 3
    j = np.arange(n)
    pad_row = j < valid
    causal = j[None, :] <= j[:, None]
    both = pad_row[None, :] & causal

    fig, ax = plt.subplots(figsize=(9.4, 3.9))
    ax.set_aspect("equal")
    ax.axis("off")
    gap = 1.05
    side = n * CELL

    def grid(x, mask, color, faint_rows=()):
        fill = np.where(mask, 0.75, np.nan)
        for r in faint_rows:
            fill[r] = np.where(mask[r], 0.25, np.nan)
        _matrix(ax, x, 0.0, n, n, color, fill=fill)

    def caption(x, above, below):
        ax.text(x + side / 2, 0.2, above, ha="center", va="bottom",
                fontsize=14, color="black")
        ax.text(x + side / 2, -side - 0.15, below, ha="center", va="top",
                fontsize=14, color="black")

    x0 = 0.0
    grid(x0, np.tile(pad_row, (n, 1)), ORANGE, faint_rows=range(1, n))
    caption(x0, "padding, valid length 3", "(batch, 1, keys),\nrow repeated")
    x1 = x0 + side + gap
    ax.text(x1 - gap / 2, -side / 2, r"$\wedge$", ha="center", va="center",
            fontsize=22, color="black")
    grid(x1, causal, BLUE)
    caption(x1, "causal", "(queries, keys)")
    x2 = x1 + side + gap
    ax.text(x2 - gap / 2, -side / 2, r"$=$", ha="center", va="center",
            fontsize=22, color="black")
    grid(x2, both, GREEN)
    caption(x2, "padding AND causal", "(batch, queries, keys)")

    ax.text(x0 - 0.12, -side / 2, "queries", rotation=90, ha="right",
            va="center", fontsize=14, color="black")
    ax.set_xlim(-0.6, x2 + side + 0.1)
    ax.set_ylim(-side - 1.05, 0.7)
    fl.save(fig, "mdl-attention-mask-algebra")


ALL_FIGURES = [fig_attention_shapes, fig_mask_algebra]


def main():
    for f in ALL_FIGURES:
        f()
    for path in fl.WRITTEN:
        print(f"wrote {os.path.relpath(path, os.path.dirname(fl.IMG_DIR))}")


if __name__ == "__main__":
    main()
