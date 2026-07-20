#!/usr/bin/env python3
"""Generate the illustrative figures for the matrix-state / SSM-family
sections of ``chapter_recurrent-modern`` (12.4 "The Matrix State") in the one
shared house style defined in ``gen_mdl_figures.py``.

Two figures:

  * the decay ladder -- the transition structure of the matrix-state
    recurrence gaining structure left to right (identity -> fixed scalar ->
    input-dependent scalar -> input-dependent diagonal), with the additive
    write shared by every rung and the error-correcting write of the next
    section shown as the dashed open rung;
  * the semiseparable block decomposition -- a *computed* decay mask
    ``L`` (random decays, cumulative products) partitioned into chunks:
    diagonal blocks are the within-chunk masked-attention matmuls,
    off-diagonal blocks are rank-limited and flow through the carried state.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_ssm_family_figures.py

All figures are written to ``img/mdl-modernrnn-<id>.svg``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT

from matplotlib.patches import FancyBboxPatch, Rectangle


def _box(ax, cx, cy, w, h, text, color, fontsize=15, ls="-", lw=1.8):
    """A rounded box (faint fill + coloured edge) with centred math text."""
    x, y = cx - w / 2, cy - h / 2
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.14",
        linewidth=lw, edgecolor=color, facecolor=color, alpha=0.10,
        linestyle=ls))
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.14",
        linewidth=lw, edgecolor=color, facecolor="none", linestyle=ls))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            color="black")


def fig_decay_ladder():
    """The transition D_t gaining structure; the write staying additive."""
    fig, ax = plt.subplots(figsize=(9.8, 3.0))
    bw, bh, by = 2.6, 1.5, 3.55                     # box width/height/center-y
    xs = [1.7, 5.0, 8.3, 11.6]                      # solid rung centers
    x5 = 15.2                                       # dashed open rung
    rungs = [
        (r"$\mathbf{I}$", "linear attention"),
        (r"$\gamma\,\mathbf{I}$", "RetNet"),
        (r"$a_t\,\mathbf{I}$", "Mamba-2"),
        (r"$\mathrm{diag}(\boldsymbol{\alpha}_t)$",
         "GLA · RWKV-6\n(selective SSM,\nper channel)"),
    ]
    steps = ["+ decay", "+ input\ndependence", "+ per-coordinate\ngates"]
    for (formula, name), cx in zip(rungs, xs):
        _box(ax, cx, by, bw, bh, formula, BLUE)
        ax.text(cx, by - bh / 2 - 0.35, name, ha="center", va="top",
                fontsize=11, color="black")
    for cx_a, cx_b, step in zip(xs[:-1], xs[1:], steps):
        fl.arrow(ax, (cx_a + bw / 2 + 0.12, by), (cx_b - bw / 2 - 0.12, by),
                 color=GRAY, lw=1.6)
        ax.text((cx_a + cx_b) / 2, by + bh / 2 + 0.32, step, ha="center",
                va="bottom", fontsize=10.5, color="black")
    # The open rung: the next section's error-correcting transition.
    _box(ax, x5, by, 3.4, bh,
         r"$\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^{\top}$",
         ORANGE, ls="--")
    fl.arrow(ax, (xs[-1] + bw / 2 + 0.12, by), (x5 - 1.7 - 0.12, by),
             color=GRAY, lw=1.6, ls="--")
    ax.text((xs[-1] + x5) / 2 - 0.05, by + bh / 2 + 0.32, "+ a write\nthat edits",
            ha="center", va="bottom", fontsize=10.5, color="black")
    ax.text(x5, by - bh / 2 - 0.35, "DeltaNet\n(next section)", ha="center",
            va="top", fontsize=11, color="black")
    # Shared additive write, bracketed under the four solid rungs.
    yb = 0.62
    ax.plot([xs[0] - bw / 2, xs[-1] + bw / 2], [yb, yb], color=GRAY, lw=1.2)
    for x in (xs[0] - bw / 2, xs[-1] + bw / 2):
        ax.plot([x, x], [yb, yb + 0.18], color=GRAY, lw=1.2)
    ax.text((xs[0] + xs[-1]) / 2, yb - 0.32,
            r"same additive write:  $\mathbf{S}_t = \mathbf{D}_t\,"
            r"\mathbf{S}_{t-1} + \mathbf{k}_t \mathbf{v}_t^{\top}$",
            ha="center", va="top", fontsize=12.5, color="black")
    ax.set_xlim(-0.1, 17.2)
    ax.set_ylim(-0.5, 5.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fl.save(fig, "mdl-modernrnn-decay-ladder")


def fig_semiseparable():
    """A computed decay mask L, chunk-partitioned: attention on the diagonal,
    rank-limited state passing below it."""
    rng = np.random.default_rng(3)
    T, C = 64, 16
    a = rng.uniform(0.86, 0.98, T)
    cum = np.cumsum(np.log(a))
    logL = cum[:, None] - cum[None, :]
    L = np.exp(np.where(np.tril(np.ones((T, T), bool)), logL, -np.inf))

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    shade = np.ma.masked_where(np.triu(np.ones((T, T), bool), 1), L ** 0.3)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("white")                           # Upper triangle: blank
    ax.imshow(shade, cmap=cmap, vmin=0, vmax=1.15,
              extent=(0, T, T, 0), interpolation="nearest")
    # Diagonal blocks: the within-chunk attention matmuls.
    for c in range(T // C):
        ax.add_patch(Rectangle((c * C, c * C), C, C, fill=False,
                               edgecolor=BLUE, lw=2.4))
    # Off-diagonal blocks: rank <= d_k, carried by the state.
    for c in range(1, T // C):
        for b in range(c):
            ax.add_patch(Rectangle((b * C, c * C), C, C, fill=False,
                                   edgecolor=ORANGE, lw=1.6))
    # State hand-off arrows across the chunk boundaries.
    for c in range(T // C - 1):
        corner = c * C + C
        fl.arrow(ax, (corner - 5.5, corner - 5.5), (corner + 5.5, corner + 5.5),
                 color=ORANGE, lw=2.0, mut=12)
    ax.set_xlim(0, T)
    ax.set_ylim(T, 0)
    ax.set_xlabel("key position $s$", fontsize=12, color="black")
    ax.set_ylabel("query position $t$", fontsize=12, color="black")
    ax.set_xticks(range(0, T + 1, C))
    ax.set_yticks(range(0, T + 1, C))
    ax.tick_params(colors="black", labelsize=10)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
    # Annotations, outside the matrix.
    ax.annotate("diagonal block:\n$C \\times C$ masked attention\n(a matmul)",
                xy=(38, 44), xytext=(70, 22), fontsize=12, color="black",
                va="center", ha="left",
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.6))
    ax.annotate("off-diagonal blocks:\nrank $\\leq d_k$ — the\n"
                "$d_k \\times d_v$ state, passed\nchunk to chunk",
                xy=(28, 56), xytext=(70, 52), fontsize=12, color="black",
                va="center", ha="left",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.6))
    fl.save(fig, "mdl-modernrnn-semiseparable")


# =========================================================================== #
# Driver                                                                      #
# =========================================================================== #

FIGURES = [
    fig_decay_ladder,
    fig_semiseparable,
]


def main():
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
