#!/usr/bin/env python3
"""Generate the illustrative figures for
chapter_attention/what-attention-computes.md (A6) in the shared house style
defined in ``gen_mdl_figures.py``.

Figures:
  mdl-attention-residual-stream.svg   -- tokens as vectors in a shared
      residual stream; an attention head reads from one stream (QK: where)
      and writes into another (OV: what).
  mdl-attention-induction-circuit.svg -- the two-hop induction circuit:
      a previous-token head in layer 1 composes with a match-and-copy head
      in layer 2.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_attention_a6.py

All figures are written to ``img/mdl-attention-<id>.svg``.  Byte-idempotent:
no timestamps, no unseeded randomness.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT
FancyArrowPatch = fl.FancyArrowPatch
Rectangle = fl.Rectangle


def _box(ax, cx, cy, w, h, text, fc="white", ec="black", ls="-", fontsize=14):
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, facecolor=fc,
                           edgecolor=ec, linestyle=ls, lw=1.2, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            color="black", zorder=4)


def _curve(ax, tail, tip, color, rad, lw=2.2, ls="-", zorder=2):
    ax.add_patch(FancyArrowPatch(
        tail, tip, arrowstyle="-|>", mutation_scale=16, lw=lw, ls=ls,
        color=color, connectionstyle=f"arc3,rad={rad}", zorder=zorder,
        shrinkA=2, shrinkB=2))


# --------------------------------------------------------------------------- #
# Figure 1: the residual stream                                               #
# --------------------------------------------------------------------------- #

def fig_residual_stream() -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    xs = [1.0, 2.9, 4.8]          # stream centers
    bw = 0.6                       # band width
    y0, y1 = 1.0, 4.1              # band extent

    # The streams: light vertical bands with the identity path inside.
    for cx in xs:
        ax.add_patch(Rectangle((cx - bw / 2, y0), bw, y1 - y0,
                               facecolor="#e8eef5", edgecolor="none",
                               zorder=0))
        fl.arrow(ax, (cx, y0 + 0.06), (cx, y1 - 0.06), color=BLUE, lw=1.6,
                 mut=12)

    ax.text(xs[0] - 0.62, 0.5 * (y0 + y1), "residual streams",
            rotation=90, ha="center", va="center", fontsize=13,
            color="black")

    # Tokens in, logits out.
    for i, cx in enumerate(xs):
        _box(ax, cx, 0.45, 0.72, 0.5, rf"$x_{i + 1}$")
        fl.arrow(ax, (cx, 0.72), (cx, y0 - 0.02), color="black", lw=1.2,
                 mut=11)
        fl.arrow(ax, (cx, y1 + 0.02), (cx, y1 + 0.28), color="black", lw=1.2,
                 mut=11)
        _box(ax, cx, 4.65, 1.06, 0.5, rf"$\mathrm{{logits}}_{i + 1}$",
             fontsize=12)
    ax.text(xs[0] - 0.62, 0.45, r"embed $\mathbf{E}$", ha="right",
            va="center", fontsize=13, color="black")
    ax.text(xs[0] - 0.62, 4.65, r"unembed $\mathbf{E}$", ha="right",
            va="center", fontsize=13, color="black")

    # One attention head: the query stream (3) scores every earlier stream
    # (QK, dashed) and receives a value payload from stream 1 (OV, solid).
    qy = 2.15                      # height at which the head reads
    wy = 3.05                      # height at which it writes back
    ax.plot([xs[2], xs[0]], [qy, qy], color=GRAY, ls="--", lw=1.6, zorder=1)
    ax.plot([xs[2], xs[1]], [qy - 0.28, qy - 0.28], color=GRAY, ls="--",
            lw=1.6, zorder=1)
    ax.text(0.5 * (xs[0] + xs[1]) - 0.28, qy - 0.75,
            r"QK circuit: where? $\;\alpha_{3j}$",
            ha="center", fontsize=14, color="black",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))

    _curve(ax, (xs[0] + bw / 2, qy), (xs[2] - bw / 2 - 0.02, wy), ORANGE,
           rad=-0.33)
    ax.text(0.5 * (xs[0] + xs[2]), 3.62,
            r"OV circuit: what? $\;\mathbf{W}_{\mathrm{OV}}\mathbf{h}_1$",
            ha="center", fontsize=14, color="black",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
    # The write is an addition into the stream.
    ax.plot([xs[2]], [wy], marker="o", ms=13, mfc="white", mec=ORANGE,
            mew=1.6, zorder=3)
    ax.text(xs[2], wy, "+", ha="center", va="center", fontsize=13,
            color=ORANGE, zorder=4)

    fl.clean_axes(ax, lim=((-0.6, 5.7), (0.1, 5.0)), hide=True, equal=False)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    fl.save(fig, "mdl-attention-residual-stream")


# --------------------------------------------------------------------------- #
# Figure 2: the induction circuit                                             #
# --------------------------------------------------------------------------- #

def fig_induction_circuit() -> None:
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    xs = [0.0, 1.1, 2.2, 3.3, 4.4]
    labels = ["A", "B", r"$\cdots$", "A", "?"]
    for cx, lab in zip(xs, labels):
        if lab == r"$\cdots$":
            ax.text(cx, 0.0, lab, ha="center", va="center", fontsize=16,
                    color="black")
        else:
            _box(ax, cx, 0.0, 0.7, 0.62, lab,
                 ls="--" if lab == "?" else "-",
                 ec=GRAY if lab == "?" else "black", fontsize=15)

    # Layer 1: previous-token head writes "follows A" into B's stream.
    _curve(ax, (xs[0], 0.34), (xs[1], 0.34), GREEN, rad=-0.75)
    ax.text(0.5 * (xs[0] + xs[1]), 1.14,
            "layer 1: previous-token head\nwrites “follows A”",
            ha="center", va="bottom", fontsize=13.5, color="black")

    # Layer 2: induction head at the second A matches that key, moves B.
    _curve(ax, (xs[1], -0.34), (xs[3], -0.34), ORANGE, rad=0.42)
    ax.text(0.5 * (xs[1] + xs[3]), -1.28,
            "layer 2: induction head\nmatches “follows A”, moves B",
            ha="center", va="top", fontsize=13.5, color="black")

    # The copy: predicted next token is B.
    _curve(ax, (xs[3], 0.34), (xs[4], 0.34), BLUE, rad=-0.75)
    ax.text(0.5 * (xs[3] + xs[4]), 1.14, "copy:\npredict B",
            ha="center", va="bottom", fontsize=13.5, color="black")

    fl.clean_axes(ax, lim=((-0.7, 5.2), (-2.35, 2.2)), hide=True,
                  equal=False)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    fl.save(fig, "mdl-attention-induction-circuit")


if __name__ == "__main__":
    fig_residual_stream()
    fig_induction_circuit()
    for path in fl.WRITTEN:
        print(f"wrote {os.path.relpath(path)}")
