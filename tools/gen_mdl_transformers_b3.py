#!/usr/bin/env python3
"""Generate the illustrative figures for chapter_transformers/kv-cache.md (B3)
in the shared house style defined in ``gen_mdl_figures.py``.

Figures:
  mdl-transformers-kv-cache.svg -- what one decoding step computes: naive
      generation recomputes every key, value, and score row of the prefix,
      while a KV cache stores past keys/values and computes one new row.
  mdl-transformers-gqa.svg -- multi-head / grouped-query / multi-query
      attention as a sharing pattern between query heads and key-value heads.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_transformers_b3.py

All figures are written to ``img/mdl-transformers-<id>.svg``.  Byte-idempotent:
no timestamps, no unseeded randomness.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT

from matplotlib.patches import Rectangle


# --------------------------------------------------------------------------- #
# Figure 1: what one decoding step computes, without and with the cache.      #
# --------------------------------------------------------------------------- #

T = 9  # tokens so far (the current step attends to all of them)


def _cell(ax, col, row, fc, alpha=1.0, ec=GRAY, lw=0.7, y0=0.0):
    """One unit cell at grid position (col, row); row 0 is the top row."""
    ax.add_patch(Rectangle((col, y0 + (T - 1 - row)), 1, 1, fc=fc,
                           alpha=alpha, ec=ec, lw=lw))


def _kv_strip(ax, y, colors):
    for j, c in enumerate(colors):
        ax.add_patch(Rectangle((j, y), 1, 0.8, fc=c[0], alpha=c[1],
                               ec=GRAY, lw=0.7))


def _score_panel(ax, cached):
    ys = T + 0.7  # bottom of the keys/values strip
    # keys/values strip above the score matrix
    if cached:
        _kv_strip(ax, ys, [(BLUE, 0.30)] * (T - 1) + [(ORANGE, 0.85)])
        ax.text((T - 1) / 2, ys + 1.25,
                r"$\mathbf{k}_{1..t-1}, \mathbf{v}_{1..t-1}$ from the cache",
                ha="center", va="center", fontsize=13, color="black")
        ax.text(T + 0.7, ys + 0.4, r"new $\mathbf{k}_t, \mathbf{v}_t$",
                ha="left", va="center", fontsize=13, color="black")
        fl.arrow(ax, (T + 0.6, ys + 0.4), (T + 0.05, ys + 0.4), color=GRAY,
                 lw=1.3, mut=11)
    else:
        _kv_strip(ax, ys, [(ORANGE, 0.85)] * T)
        ax.text(T / 2, ys + 1.25,
                r"all $\mathbf{k}_{1..t}, \mathbf{v}_{1..t}$ recomputed",
                ha="center", va="center", fontsize=13, color="black")
    # score matrix: lower triangle
    for i in range(T):
        for j in range(i + 1):
            if cached:
                if i == T - 1:
                    _cell(ax, j, i, ORANGE, 0.85)
                else:
                    ax.add_patch(Rectangle((j, T - 1 - i), 1, 1, fill=False,
                                           ec=LIGHT, lw=0.7, ls=(0, (2, 2))))
            else:
                _cell(ax, j, i, ORANGE, 0.85 if i == T - 1 else 0.30)
    if cached:
        ax.text(2.9, 5.6, "not computed", ha="center", va="center",
                fontsize=13, color=GRAY, rotation=0,
                bbox=dict(fc="white", ec="none", pad=2))
    # arrow from the strip down toward the current score row
    fl.arrow(ax, (T - 0.15, ys - 0.15), (T - 0.15, 1.35), color=GRAY, lw=1.4,
             mut=12)
    label = ("one new score row per step" if cached
             else "every score row redone per step")
    ax.text(T / 2, -0.85, label, ha="center", va="center", fontsize=14,
            color="black")
    ax.text(-0.55, 0.5, r"row $t$", ha="right", va="center", fontsize=13,
            color="black")
    fl.arrow(ax, (-0.45, 0.5), (-0.02, 0.5), color=GRAY, lw=1.4, mut=11)


def fig_kv_cache():
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.6))
    for ax, cached in zip(axes, (False, True)):
        _score_panel(ax, cached)
        ax.set_xlim(-2.1, T + 3.4)
        ax.set_ylim(-1.6, T + 3.0)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.subplots_adjust(wspace=0.04)
    fl.save(fig, "mdl-transformers-kv-cache")


# --------------------------------------------------------------------------- #
# Figure 2: multi-head / grouped-query / multi-query sharing patterns.        #
# --------------------------------------------------------------------------- #

H = 8  # query heads


def _heads_panel(ax, n_kv, label):
    w, gap, y_q, y_kv = 0.72, 1.0, 2.4, 0.0
    for i in range(H):
        x = i * gap
        ax.add_patch(Rectangle((x, y_q), w, w, fc=BLUE, alpha=0.55,
                               ec="black", lw=0.9))
    group = H // n_kv
    for g in range(n_kv):
        # one kv head centered under its group of query heads
        xs = [i * gap for i in range(g * group, (g + 1) * group)]
        xc = (xs[0] + xs[-1]) / 2
        ax.add_patch(Rectangle((xc, y_kv), w, w, fc=ORANGE, alpha=0.75,
                               ec="black", lw=0.9))
        for x in xs:
            ax.plot([x + w / 2, xc + w / 2], [y_q - 0.06, y_kv + w + 0.06],
                    color=GRAY, lw=1.3, zorder=1)
    ax.text((H - 1) * gap / 2 + w / 2, y_q + w + 0.55, label, ha="center",
            va="center", fontsize=14, color="black")
    ax.set_xlim(-2.15, (H - 1) * gap + w + 0.35)
    ax.set_ylim(-1.35, y_q + w + 1.1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text((H - 1) * gap / 2 + w / 2, -0.85,
            rf"cache $\propto H_{{kv}} = {n_kv}$", ha="center", va="center",
            fontsize=13, color="black")


def fig_gqa():
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 2.9))
    for ax, (n_kv, label) in zip(
            axes, ((8, "multi-head"), (2, "grouped-query"),
                   (1, "multi-query"))):
        _heads_panel(ax, n_kv, label)
    # row labels on the left panel only (all panels share xlim => same scale)
    axes[0].text(-0.35, 2.4 + 0.36, "query\nheads", ha="right", va="center",
                 fontsize=12, color="black")
    axes[0].text(-0.35, 0.36, "KV\nheads", ha="right", va="center",
                 fontsize=12, color="black")
    fig.subplots_adjust(wspace=0.08)
    fl.save(fig, "mdl-transformers-gqa")


if __name__ == "__main__":
    fig_kv_cache()
    fig_gqa()
    for p in fl.WRITTEN:
        print("wrote", p)
