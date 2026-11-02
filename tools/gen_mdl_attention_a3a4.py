#!/usr/bin/env python3
"""Generate the illustrative figures for chapter_attention/multihead-attention.md
(A3) and chapter_attention/positional-information.md (A4) in the shared house
style defined in ``gen_mdl_figures.py``.

Figures:
  mdl-attention-one-head-averages.svg  -- one softmax distribution per query
      vs. two heads with two distributions (the copy-both construction).
  mdl-attention-rope-rotation.svg      -- RoPE as position-dependent rotation;
      shifting both positions preserves the angle between query and key.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_attention_a3a4.py

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


# --------------------------------------------------------------------------- #
# A3: one head averages, two heads copy                                       #
# --------------------------------------------------------------------------- #

def fig_one_head_averages() -> None:
    # The copy-both task has two source positions; the query weights them.
    positions = np.array([1, 2])
    # One head: one softmax pair (any split leaves relative error 1/2).
    w_single = np.array([0.7, 0.3])
    # Two heads: two *different* soft pairs, as in the section's demo.
    w_head1 = np.array([0.88, 0.12])
    w_head2 = np.array([0.12, 0.88])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))

    ax = axes[0]
    ax.bar(positions, w_single, width=0.5, color=BLUE)
    ax.set_title("one head: one weighting", fontsize=14, color="black")
    for p, w, lab in zip(positions, w_single, (r"$\alpha_1$", r"$\alpha_2$")):
        ax.text(p, w + 0.04, lab, ha="center", va="bottom", fontsize=15,
                color="black")
    ax.text(1.5, -0.24,
            r"$\mathbf{W}_o\,(\alpha_1 \mathbf{v}_1 + \alpha_2 \mathbf{v}_2)$:"
            "\none mixture",
            ha="center", va="top", fontsize=14, color="black")

    ax = axes[1]
    ax.bar(positions - 0.14, w_head1, width=0.26, color=ORANGE)
    ax.bar(positions + 0.14, w_head2, width=0.26, color=GREEN)
    ax.set_title("two heads: two weightings", fontsize=14, color="black")
    # label each head directly above its taller bar (no legend box)
    ax.text(1 - 0.14, w_head1[0] + 0.04, "head 1", ha="center",
            va="bottom", fontsize=13, color=fl.T.ORANGE.dark)
    ax.text(2 + 0.14, w_head2[1] + 0.04, "head 2", ha="center",
            va="bottom", fontsize=13, color=fl.T.GREEN.dark)
    ax.text(1.5, -0.24,
            r"$\mathbf{W}_o\,[\mathbf{z}_1;\, \mathbf{z}_2]$:"
            "\ntwo mixtures, both values recovered",
            ha="center", va="top", fontsize=14, color="black")

    for ax in axes:
        ax.set_xticks(positions)
        ax.set_xticklabels([r"$\mathbf{v}_1$", r"$\mathbf{v}_2$"],
                           fontsize=15, color="black")
        ax.set_ylim(0, 1.08)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_xlim(0.45, 2.55)
        ax.tick_params(axis="y", labelsize=12, colors="black")
        ax.spines["left"].set_color("black")
        ax.spines["bottom"].set_color("black")

    axes[0].set_ylabel("attention weight", fontsize=14, color="black")
    fig.subplots_adjust(wspace=0.3, bottom=0.3)
    fl.save(fig, "mdl-attention-one-head-averages")


# --------------------------------------------------------------------------- #
# A4: RoPE rotation -- the score depends only on the offset                   #
# --------------------------------------------------------------------------- #

def fig_rope_rotation() -> None:
    theta = np.deg2rad(25.0)          # per-position rotation angle
    phi_q, phi_k = np.deg2rad(12.0), np.deg2rad(40.0)  # base orientations
    r_q, r_k = 0.94, 0.80             # arrow lengths (distinct, for clarity)

    def panel(ax, i, j, title):
        # Unit circle as a light reference shape.
        t = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(t), np.sin(t), color=LIGHT, lw=1.2, zorder=0)

        ang_q = phi_q + i * theta
        ang_k = phi_k + j * theta
        tip_q = (r_q * np.cos(ang_q), r_q * np.sin(ang_q))
        tip_k = (r_k * np.cos(ang_k), r_k * np.sin(ang_k))
        fl.arrow(ax, (0, 0), tip_q, color=BLUE, lw=2.2)
        fl.arrow(ax, (0, 0), tip_k, color=ORANGE, lw=2.2)

        # Labels just outside the unit circle along each arrow's direction,
        # anchored on the side facing the circle so the text extends outward
        # and never straddles the circle.
        for (ang, txt, col) in [
                (ang_q, rf"$\mathbf{{R}}_{{{i}}}\,\mathbf{{q}}$", BLUE),
                (ang_k, rf"$\mathbf{{R}}_{{{j}}}\,\mathbf{{k}}$", ORANGE)]:
            c, s_ = np.cos(ang), np.sin(ang)
            ha = "left" if c > 0.35 else ("right" if c < -0.35 else "center")
            va = "bottom" if s_ > 0.35 else ("top" if s_ < -0.35 else "center")
            ax.text(1.07 * c, 1.07 * s_, txt, ha=ha, va=va, fontsize=15,
                    color=col)

        # Arc marking the angle between the two vectors, labelled inside the
        # wedge with its value: (j - i) theta + phi = 2 theta + phi in BOTH
        # panels, which is the point of the figure.
        arc_r = 0.30
        arc = np.linspace(ang_q, ang_k, 60)
        ax.plot(arc_r * np.cos(arc), arc_r * np.sin(arc), color=GREEN, lw=2.2)
        mid = 0.5 * (ang_q + ang_k)
        # The label is wider than tall: the more sideways the wedge opens,
        # the further inward it sits to stay clear of the circle.
        r_lab = 0.72 - 0.12 * abs(np.cos(mid))
        ax.text(r_lab * np.cos(mid), r_lab * np.sin(mid),
                rf"${j - i}\theta\!+\!\varphi$",
                ha="center", va="center", fontsize=15,
                color=fl.T.GREEN.dark)

        ax.set_title(title, fontsize=14, color="black")
        ax.set_xlim(-1.62, 1.62)
        ax.set_ylim(-1.12, 1.52)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.9))
    panel(axes[0], i=1, j=3, title="positions (1, 3)")
    panel(axes[1], i=4, j=6, title="shifted by 3: positions (4, 6)")
    fig.subplots_adjust(wspace=0.12)
    fl.save(fig, "mdl-attention-rope-rotation")


if __name__ == "__main__":
    fig_one_head_averages()
    fig_rope_rotation()
    for path in fl.WRITTEN:
        print(f"wrote {os.path.relpath(path)}")
