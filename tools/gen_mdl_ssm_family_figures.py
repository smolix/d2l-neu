#!/usr/bin/env python3
"""Generate the illustrative figures for the matrix-state / SSM-family
sections of ``chapter_recurrent-modern`` (12.4 "The Matrix State") in the one
shared house style defined in ``gen_mdl_figures.py``.

One figure (the "decay ladder" strip was dropped with the 2026-10 chapter
rewrite, which no longer references it; see git history):

  * the semiseparable block decomposition -- a *computed* decay mask
    ``L`` (random decays, cumulative products) partitioned into chunks:
    diagonal blocks are the within-chunk masked-attention matmuls,
    off-diagonal blocks are rank-limited and flow through the carried state.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_ssm_family_figures.py

The figure is written to ``img/mdl-modernrnn-<id>.svg``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT

from matplotlib.patches import Rectangle


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
    # State hand-off arrows across the chunk boundaries, shifted up-right
    # into the blank upper triangle so they cross the blue block edges in
    # white space instead of running through the grid junctions on the
    # stepped diagonal.
    off = 2.0
    for c in range(T // C - 1):
        corner = c * C + C
        fl.arrow(ax, (corner - 5.5 + off, corner - 5.5 - off),
                 (corner + 5.5 + off, corner + 5.5 - off),
                 color=ORANGE, lw=2.0, mut=12)
    ax.set_xlim(0, T)
    ax.set_ylim(T, 0)
    ax.set_xlabel("key position $s$", fontsize=13, color="black")
    ax.set_ylabel("query position $t$", fontsize=13, color="black")
    ax.set_xticks(range(0, T + 1, C))
    ax.set_yticks(range(0, T + 1, C))
    ax.tick_params(colors="black", labelsize=12)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
    # Annotations, outside the matrix, in the prose's notation (chunks of m
    # steps, a d_v x d_k state).  The diagonal-block leader runs through the
    # blank upper triangle and stops on the second block's right edge; the
    # off-diagonal leader ends in the nearest off-diagonal block.
    ax.annotate("diagonal block:\n$m \\times m$ masked attention\n(a matmul)",
                xy=(2 * C, 22), xytext=(70, 22), fontsize=13, color="black",
                va="center", ha="left",
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.6,
                                shrinkA=4, shrinkB=0))
    ax.annotate("off-diagonal blocks:\nrank $\\leq d_k$, via the\n"
                "$d_v \\times d_k$ state passed\nchunk to chunk",
                xy=(40, 57), xytext=(70, 57), fontsize=13, color="black",
                va="center", ha="left",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.6,
                                shrinkA=4, shrinkB=0))
    fl.save(fig, "mdl-modernrnn-semiseparable")


# =========================================================================== #
# Driver                                                                      #
# =========================================================================== #

FIGURES = [
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
