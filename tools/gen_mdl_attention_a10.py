#!/usr/bin/env python3
"""Generate the two sliding-window figures of windowed-linear-attention.md
(section A10) in the shared house style of ``gen_mdl_figures.py``.

Figures:

  * ``mdl-attention-window-tiles`` -- the causal mask and the sliding-window
    band (w = b) laid over a T x T grid of b x b kernel tiles, T = 16, with
    the visited tiles shaded: 136 of 256 (53.1%) against 31 of 256 (12.1%),
    the two counts the section's ``tiles_visited`` cell prints.
  * ``mdl-attention-window-receptive-field`` -- four stacked windowed layers
    (w = 4) relay a token Delta = 12 positions back along a single path of
    four hops, weight (1/4)^4 = 1/256 under uniform weights, against one hop
    of weight 1/13 in a single global layer.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_attention_a10.py

All figures are written to ``img/mdl-attention-<id>.svg``. Deterministic (no
random draws, no timestamps), hence byte-idempotent via the shared ``save()``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers
from matplotlib.patches import Circle, Rectangle

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT


# =========================================================================== #
# 1. mdl-attention-window-tiles                                               #
# =========================================================================== #

def fig_window_tiles() -> None:
    """Causal mask and window band (w = b) over a 16 x 16 grid of tiles.

    Each tile holds b x b scores; b = 8 here so the band stays visible, which
    leaves the tile counts (they depend only on T and w / b) unchanged.
    """
    T, b = 16, 8
    n = T * b
    i = np.arange(n)
    causal = i[None, :] <= i[:, None]
    window = causal & (i[:, None] - i[None, :] < b)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.0))
    for ax, mask, color, title in (
            (axes[0], causal, BLUE, "causal"),
            (axes[1], window, GREEN, r"window, $w = b$")):
        tiles = mask.reshape(T, b, T, b).any(axis=(1, 3))
        # visited tiles: faint fill; allowed scores inside them: solid fill
        for r in range(T):
            for c in range(T):
                if tiles[r, c]:
                    ax.add_patch(Rectangle((c * b, n - (r + 1) * b), b, b,
                                           facecolor=color, alpha=0.16,
                                           edgecolor="none", zorder=1))
        img = np.where(mask, 1.0, np.nan)
        ax.imshow(img, cmap=plt.matplotlib.colors.ListedColormap([color]),
                  extent=(0, n, 0, n), interpolation="nearest", alpha=0.85,
                  zorder=2)
        for k in range(1, T):                       # tile boundaries
            ax.plot([k * b, k * b], [0, n], color=LIGHT, lw=0.8, zorder=3)
            ax.plot([0, n], [k * b, k * b], color=LIGHT, lw=0.8, zorder=3)
        ax.add_patch(Rectangle((0, 0), n, n, facecolor="none",
                               edgecolor="black", lw=1.4, zorder=4))
        visited = int(tiles.sum())
        ax.set_title(f"{title}: {visited} of {T * T} tiles "
                     f"({visited / T ** 2:.1%})",
                     fontsize=14, color="black")
        ax.text(n / 2, -3, "keys", ha="center", va="top", fontsize=14,
                color="black")
        ax.set_xlim(-1, n + 1)
        ax.set_ylim(-11, n + 1)
        ax.set_aspect("equal")
        ax.axis("off")
    axes[0].text(-3, n / 2, "queries", rotation=90, ha="right", va="center",
                 fontsize=14, color="black")
    fig.subplots_adjust(wspace=0.12)
    fl.save(fig, "mdl-attention-window-tiles")


# =========================================================================== #
# 2. mdl-attention-window-receptive-field                                     #
# =========================================================================== #

R_NODE = 0.28


def _node(ax, x, y, active, fc="white"):
    ax.add_patch(Circle((x, y), R_NODE, facecolor=fc if active else "white",
                        edgecolor="black" if active else LIGHT, lw=1.1,
                        zorder=4))


def _edge(ax, x0, y0, x1, y1, highlight=False):
    """Edge between two node centres, clipped to both circles."""
    d = np.array([x1 - x0, y1 - y0], float)
    d /= np.linalg.norm(d)
    a = (x0 + R_NODE * d[0], y0 + R_NODE * d[1])
    c = (x1 - R_NODE * d[0], y1 - R_NODE * d[1])
    ax.plot([a[0], c[0]], [a[1], c[1]], color=BLUE if highlight else LIGHT,
            lw=2.2 if highlight else 0.9, zorder=3 if highlight else 1,
            solid_capstyle="butt")


def fig_window_receptive_field() -> None:
    """Four windowed layers (w = 4) against one global layer, Delta = 12."""
    w, L, delta = 4, 4, 12
    n = delta + 1                      # positions 0..12; the query is at 12
    dy = 1.25                          # vertical distance between layers
    # the only relay from position 0 runs 0, 3, 6, 9, 12: layer l holds it at
    # position l(w - 1), and each hop spans the whole window
    fig, ax = plt.subplots(figsize=(12.4, 4.6))

    # ---- left: the receptive cone of the top output through 4 windows ------ #
    # cone[l] = positions of layer l that the top output depends on
    cone = {L: {delta}}
    for layer in range(L, 0, -1):
        cone[layer - 1] = {j for p in cone[layer]
                           for j in range(max(0, p - w + 1), p + 1)}
    for layer in range(1, L + 1):
        for p in sorted(cone[layer]):
            for j in range(max(0, p - w + 1), p + 1):
                hl = p == layer * (w - 1) and j == p - (w - 1)
                _edge(ax, j, (layer - 1) * dy, p, layer * dy, highlight=hl)
    for layer in range(L + 1):
        for p in range(n):
            on_path = p == layer * (w - 1)
            _node(ax, p, layer * dy, p in cone[layer],
                  fc=fl.T.BLUE.tint if on_path else "white")
    for layer in range(L + 1):
        ax.text(-0.75, layer * dy, "input" if layer == 0 else f"layer {layer}",
                ha="right", va="center", fontsize=13.5, color="black")
    ax.text(delta / 2, L * dy + 0.75,
            rf"four windowed layers, $w = {w}$", ha="center", va="bottom",
            fontsize=14, color="black")
    ax.text(delta / 2, -0.75,
            r"one path from $\Delta = 12$ back, weight $(1/4)^4 = 1/256$",
            ha="center", va="top", fontsize=14, color="black")

    # ---- right: one global layer reaches every position in one hop -------- #
    x0 = n + 2.6
    for p in range(n):
        _edge(ax, x0 + p, 0, x0 + delta, L * dy, highlight=(p == 0))
    for p in range(n):
        _node(ax, x0 + p, 0, True, fc=fl.T.BLUE.tint if p == 0 else "white")
        _node(ax, x0 + p, L * dy, p == delta,
              fc=fl.T.BLUE.tint if p == delta else "white")
    ax.text(x0 + delta / 2, L * dy + 0.75, "one global layer", ha="center",
            va="bottom", fontsize=14, color="black")
    ax.text(x0 + delta / 2, -0.75, r"one hop, weight $1/13$", ha="center",
            va="top", fontsize=14, color="black")

    ax.set_xlim(-2.6, x0 + delta + 0.5)
    ax.set_ylim(-1.45, L * dy + 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    fl.save(fig, "mdl-attention-window-receptive-field")


ALL_FIGURES = [fig_window_tiles, fig_window_receptive_field]


def main():
    start = len(fl.WRITTEN)
    for f in ALL_FIGURES:
        f()
    for path in fl.WRITTEN[start:]:
        print(f"wrote {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
