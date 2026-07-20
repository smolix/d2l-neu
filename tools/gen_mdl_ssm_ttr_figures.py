#!/usr/bin/env python3
"""Generate the illustrative figure for the test-time-regression section of
``chapter_recurrent-modern`` (12.6 "Learning at Test Time") in the one shared
house style defined in ``gen_mdl_figures.py``.

One figure:

  * the inner/outer learning loops -- two panels contrasting what updates
    when.  Left (pretraining): the outer parameters (projections, gates,
    initial state) parameterize every inner state update, and the outer
    gradient flows from the pretraining loss back THROUGH the chain of
    inner updates.  Right (inference): the same inner chain runs online --
    the state adapts token by token -- while the outer parameters are
    frozen and no labels exist.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_ssm_ttr_figures.py

The figure is written to ``img/mdl-modernrnn-inner-outer.svg``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT

from matplotlib.patches import FancyBboxPatch


def _box(ax, cx, cy, w, h, text, color, fontsize=13, ls="-", lw=1.8):
    """A rounded box (faint fill + coloured edge) with centred black text."""
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


def _panel(ax, pretraining):
    """One panel of the two-loop diagram; identical geometry in both."""
    ax.set_aspect("equal")
    ax.axis("off")

    # ── the inner chain: state boxes S_0 → S_1 → S_2 → ⋯ → S_T ──────────
    y_state, sw, sh = 3.05, 1.06, 0.78
    xs = [1.05, 3.05, 5.05, 7.85]           # S_0, S_1, S_2, S_T
    labels = [r"$\mathbf{S}_0$", r"$\mathbf{S}_1$", r"$\mathbf{S}_2$",
              r"$\mathbf{S}_T$"]
    for x, lab in zip(xs, labels):
        _box(ax, x, y_state, sw, sh, lab, ORANGE, fontsize=14)
    for xa, xb in zip(xs[:-1], xs[1:-1]):   # arrows between drawn states
        fl.arrow(ax, (xa + sw / 2 + 0.04, y_state),
                 (xb - sw / 2 - 0.04, y_state), color=GRAY, lw=1.9, mut=13)
    ax.text((xs[2] + xs[3]) / 2, y_state, r"$\cdots$", fontsize=16,
            color=GRAY, ha="center", va="center")
    fl.arrow(ax, ((xs[2] + xs[3]) / 2 + 0.42, y_state),
             (xs[3] - sw / 2 - 0.04, y_state), color=GRAY, lw=1.9, mut=13)
    ax.plot([xs[2] + sw / 2 + 0.04, (xs[2] + xs[3]) / 2 - 0.42],
            [y_state, y_state], color=GRAY, lw=1.9)

    # the update rule, named once in the clear band above the chain
    ax.text(4.45, 4.22,
            r"inner update  $\mathbf{S}_t"
            r" = f_{\theta}(\mathbf{S}_{t-1}, \mathbf{k}_t, \mathbf{v}_t)$",
            fontsize=12.5, color="black", ha="center", va="center")

    # tokens feeding each update from below
    for x, sub in zip(xs[1:], ["1", "2", "T"]):
        fl.arrow(ax, (x, 1.18), (x, y_state - sh / 2 - 0.06), color=GRAY,
                 lw=1.6, mut=11)
        ax.text(x, 0.78, rf"$(\mathbf{{k}}_{sub}, \mathbf{{v}}_{sub})$",
                fontsize=13, color="black", ha="center", va="center")

    # ── the outer parameters θ ───────────────────────────────────────────
    th_cx, th_cy, th_w, th_h = 2.25, 5.92, 4.0, 1.25
    if pretraining:
        _box(ax, th_cx, th_cy, th_w, th_h,
             "outer parameters $\\theta$:\n"
             r"$\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v$,"
             r" gates, $\mathbf{S}_0$",
             BLUE, fontsize=12.5)
    else:
        _box(ax, th_cx, th_cy, th_w, th_h,
             "outer parameters $\\theta$\nfrozen", GRAY, fontsize=12.5)
    # θ parameterizes every inner update
    fl.arrow(ax, (th_cx, th_cy - th_h / 2 - 0.05),
             (th_cx, y_state + sh / 2 + 0.12),
             color=BLUE if pretraining else GRAY, lw=1.8, mut=12)

    # ── read-out and, in pretraining, the loss + outer gradient ─────────
    fl.arrow(ax, (7.85, y_state + sh / 2 + 0.06), (7.85, 5.92 - 0.56),
             color=GRAY, lw=1.6, mut=11)
    if pretraining:
        _box(ax, 7.85, 5.92, 2.95, 1.0,
             "pretraining loss\n$\\mathcal{L}(\\theta)$", GREEN,
             fontsize=12.5)
        ax.text(8.15, 4.55, r"$\mathbf{o}_t$", fontsize=13, color="black",
                ha="left", va="center")
        # the outer gradient, back through the inner updates
        fl.arrow(ax, (7.85 - 2.95 / 2 - 0.06, 5.92),
                 (th_cx + th_w / 2 + 0.06, 5.92),
                 color=GREEN, lw=2.2, ls="--", mut=15)
        ax.text(5.32, 4.97,
                r"$\nabla_{\theta} \mathcal{L}$: through"
                "\nevery inner update", fontsize=12.5, color="black",
                ha="center", va="center")
    else:
        _box(ax, 7.85, 5.92, 2.6, 1.0, "answers\n$\\mathbf{o}_t$",
             GREEN, fontsize=13)
        ax.text(5.2, 4.97, "no labels,\nno update to $\\theta$",
                fontsize=13, color="black", ha="center", va="center")

    ax.set_xlim(0.15, 9.45)
    ax.set_ylim(0.35, 6.95)


def fig_inner_outer():
    """Outer loop (pretraining) vs inner loop (inference): what updates when."""
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.05))
    _panel(axes[0], pretraining=True)
    _panel(axes[1], pretraining=False)
    axes[0].set_title("outer loop: pretraining", fontsize=14, pad=8)
    axes[1].set_title("inner loop alone: inference", fontsize=14, pad=8)
    fig.subplots_adjust(wspace=0.06)
    fl.save(fig, "mdl-modernrnn-inner-outer")


if __name__ == "__main__":
    fl.WRITTEN.clear()
    fig_inner_outer()
    for p in fl.WRITTEN:
        print("wrote", os.path.relpath(p))
