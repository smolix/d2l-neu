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
    positions = np.arange(1, 7)
    # One head: must split its weight between the two marked positions.
    w_single = np.array([0.02, 0.44, 0.03, 0.03, 0.44, 0.04])
    # Two heads: each locks onto one of the marked positions.
    w_head1 = np.array([0.02, 0.88, 0.03, 0.03, 0.02, 0.02])
    w_head2 = np.array([0.02, 0.02, 0.03, 0.03, 0.88, 0.02])

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0))

    ax = axes[0]
    ax.bar(positions, w_single, width=0.62, color=BLUE)
    ax.set_title("one head: one distribution", fontsize=14, color="black")
    ax.text(2, w_single[1] + 0.05, r"$\alpha_1$", ha="center", fontsize=14,
            color="black")
    ax.text(5, w_single[4] + 0.05, r"$\alpha_2$", ha="center", fontsize=14,
            color="black")
    ax.text(3.5, -0.42,
            r"output $= W_o\,(\alpha_1 \mathbf{v}_1 + \alpha_2 \mathbf{v}_2)$"
            "\n— a single mixture",
            ha="center", va="top", fontsize=13, color="black")

    ax = axes[1]
    ax.bar(positions - 0.17, w_head1, width=0.32, color=ORANGE,
           label="head 1")
    ax.bar(positions + 0.17, w_head2, width=0.32, color=GREEN,
           label="head 2")
    ax.set_title("two heads: two distributions", fontsize=14, color="black")
    ax.legend(loc="upper center", fontsize=12, ncol=2,
              handlelength=1.2, columnspacing=1.0)
    ax.text(3.5, -0.42,
            r"output $= W_o\,[\mathbf{h}_1;\, \mathbf{h}_2]$"
            "\n— both values recoverable",
            ha="center", va="top", fontsize=13, color="black")

    for ax in axes:
        ax.set_xticks(positions)
        ax.set_xticklabels([r"$\mathbf{v}_1$" if p == 2 else
                            (r"$\mathbf{v}_2$" if p == 5 else str(p))
                           for p in positions], fontsize=13, color="black")
        ax.set_ylim(0, 1.06)
        ax.set_xlim(0.35, 6.65)
        ax.set_ylabel("attention weight", fontsize=13, color="black")
        ax.tick_params(axis="y", labelsize=11, colors="black")
        ax.spines["left"].set_color("black")
        ax.spines["bottom"].set_color("black")

    fig.subplots_adjust(wspace=0.3, bottom=0.28)
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

        # Labels radially outward past the arrow tips.
        for (ang, r, txt, col) in [
                (ang_q, r_q, rf"$R_{{{i}}}\,\mathbf{{q}}$", BLUE),
                (ang_k, r_k, rf"$R_{{{j}}}\,\mathbf{{k}}$", ORANGE)]:
            ax.text((r + 0.30) * np.cos(ang), (r + 0.30) * np.sin(ang),
                    txt, ha="center", va="center", fontsize=15, color=col)

        # Arc marking the angle between the two vectors; label outside the
        # circle at the arc's mid-angle, clear of both arrows.
        arc_r = 0.46
        arc = np.linspace(ang_q, ang_k, 60)
        ax.plot(arc_r * np.cos(arc), arc_r * np.sin(arc), color=GREEN, lw=2.2)
        mid = 0.5 * (ang_q + ang_k)
        ax.text(1.16 * np.cos(mid), 1.16 * np.sin(mid),
                r"$(j\!-\!i)\,\theta + \varphi$",
                ha="center", va="center", fontsize=14, color=GREEN,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))

        ax.set_title(title, fontsize=14, color="black")
        ax.set_xlim(-1.45, 1.45)
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
