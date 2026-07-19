#!/usr/bin/env python3
"""Generate the illustrative figure for chapter_transformers/moe.md (B6)
in the shared house style defined in ``gen_mdl_figures.py``.

Figure:
  mdl-transformers-moe.svg -- one token's pass through a mixture-of-experts
      layer: a linear router scores all experts, the top-k (here one)
      expert computes, its output is weighted by the router probability;
      unselected experts hold parameters but do no work for this token.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_transformers_b6.py

Output is written to ``img/mdl-transformers-moe.svg``.  Byte-idempotent:
no timestamps, no unseeded randomness.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GRAY, fl.LIGHT

from matplotlib.patches import FancyBboxPatch

E = 8               # experts
SEL = 5             # selected expert (0-based)
PROBS = [0.06, 0.04, 0.09, 0.13, 0.08, 0.31, 0.18, 0.11]  # router output


def _box(ax, cx, cy, w, h, fc, ec="black", lw=0.9, ls="-", alpha=1.0):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        fc=fc, ec=ec, lw=lw, linestyle=ls, alpha=alpha))


def fig_moe():
    fig, ax = plt.subplots(figsize=(9.4, 4.9))
    xs = [0.9 + i * 1.17 for i in range(E)]      # expert centers
    y_tok, y_rt, y_exp, y_out = 0.45, 1.85, 4.05, 5.6
    x_mid = (xs[0] + xs[-1]) / 2

    # token and router
    _box(ax, x_mid, y_tok, 1.0, 0.62, "white")
    ax.text(x_mid, y_tok, r"$\mathbf{x}$", ha="center", va="center",
            fontsize=15, color="black")
    ax.text(x_mid + 0.85, y_tok, "token", ha="left", va="center",
            fontsize=13, color="black")
    _box(ax, x_mid, y_rt, 1.7, 0.62, BLUE, alpha=0.25)
    ax.text(x_mid, y_rt, "router", ha="center", va="center", fontsize=14,
            color="black")
    ax.text(x_mid + 1.15, y_rt,
            r"$\mathbf{p} = \mathrm{softmax}(\mathbf{W}_r \mathbf{x})$",
            ha="left", va="center", fontsize=14, color="black")
    fl.arrow(ax, (x_mid, y_tok + 0.36), (x_mid, y_rt - 0.36), color="black",
             lw=1.4, mut=11)

    # experts and the routing fan
    for i, (x, p) in enumerate(zip(xs, PROBS)):
        sel = i == SEL
        _box(ax, x, y_exp, 0.92, 0.92,
             ORANGE if sel else LIGHT,
             ec="black" if sel else GRAY,
             lw=1.2 if sel else 0.9,
             ls="-" if sel else (0, (3, 2)),
             alpha=0.75 if sel else 0.45)
        ax.text(x, y_exp, rf"$\mathrm{{FFN}}_{{{i + 1}}}$", ha="center",
                va="center", fontsize=13, color="black")
        if sel:
            fl.arrow(ax, (x_mid, y_rt + 0.36), (x, y_exp - 0.52),
                     color=ORANGE, lw=2.2, mut=13)
            # in the clear wedge right of the arrow, below the fan lines
            ax.text(x_mid + 1.62, y_rt + 0.48,
                    rf"$p_{{{i + 1}}} = {p:.2f}$", ha="left", va="center",
                    fontsize=14, color="black",
                    bbox=dict(fc="white", ec="none", pad=1.5))
        else:
            ax.plot([x_mid, x], [y_rt + 0.36, y_exp - 0.52], color=GRAY,
                    lw=1.0, alpha=0.20 + 1.6 * p, zorder=1)

    # output of the selected expert, scaled by its probability
    fl.arrow(ax, (xs[SEL], y_exp + 0.52), (xs[SEL], y_out - 0.18),
             color=ORANGE, lw=2.2, mut=13)
    ax.text(xs[SEL], y_out + 0.12,
            rf"$\mathbf{{y}} = p_{{{SEL + 1}}}\,"
            rf"\mathrm{{FFN}}_{{{SEL + 1}}}(\mathbf{{x}})$",
            ha="center", va="bottom", fontsize=15, color="black")

    # annotation: unselected experts store parameters but do no work
    ax.text(xs[1] + 0.1, y_exp + 1.0, "parameters held,\nnot computed",
            ha="center", va="bottom", fontsize=13, color=GRAY)
    fl.arrow(ax, (xs[1] + 0.1, y_exp + 0.98), (xs[1], y_exp + 0.56),
             color=GRAY, lw=1.1, mut=10)

    ax.set_xlim(-0.1, xs[-1] + 1.0)
    ax.set_ylim(-0.1, 6.55)
    ax.set_aspect("equal")
    ax.axis("off")
    fl.save(fig, "mdl-transformers-moe")


if __name__ == "__main__":
    fig_moe()
    for p in fl.WRITTEN:
        print("wrote", p)
