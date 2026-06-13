#!/usr/bin/env python3
"""Generate the illustrative figures for the "Linear Neural Networks for
Classification" chapter (``chapter_linear-classification``) in the one shared
house style defined in ``gen_mdl_figures.py``.

The notebooks / prose reference the generated files with no drawing code (like
the slide SVGs). Figures that show a computed result use real numerical
computation so the pictures are exact, not sketched.

Run with the repo's pytorch venv (matplotlib + numpy are available):

    .venv-pytorch/bin/python tools/gen_mdl_classification_figures.py

All figures are written to ``img/mdl-clf-<id>.svg``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT

from matplotlib.patches import FancyBboxPatch


def fig_loss_accuracy():
    """The two roles of a classifier's scores. One forward pass produces a
    logit vector o; from there the picture forks. The TOP (training) branch
    softmaxes o into probabilities and reads off the differentiable
    cross-entropy loss that gradient descent minimizes. The BOTTOM (evaluation)
    branch takes the argmax to a single hard decision, compares it with the
    label, and counts it for accuracy --- a discrete number with no useful
    gradient. Both branches read the SAME o, so the numbers in the picture are
    a real, exact softmax / cross-entropy of one concrete logit vector."""
    # One concrete example: 3-class logits, true label y = 1 (the middle class).
    o = np.array([1.0, 2.2, 0.3])           # the model's score vector (logits)
    y = 1                                     # ground-truth class index
    p = np.exp(o - o.max())                   # numerically stable softmax
    p = p / p.sum()
    yhat = int(np.argmax(p))                  # hard decision
    loss = float(-np.log(p[y]))               # cross-entropy of the true class
    correct = int(yhat == y)
    # Strings shown in the figure are formatted from the SAME computed numbers,
    # so the picture can never drift from the real softmax / cross-entropy.
    o_str = "(" + ",\\ ".join(f"{v:.1f}" for v in o) + ")"
    p_str = "(" + ",\\ ".join(f"{v:.2f}" for v in p) + ")"
    loss_str = f"{loss:.2f}"
    print(f"  logits o = {o.tolist()}")
    print(f"  softmax p = {np.round(p, 3).tolist()}  ->  argmax y_hat = {yhat}")
    print(f"  true y = {y}   cross-entropy = {loss:.3f}   correct = {correct}")

    fig, ax = plt.subplots(figsize=(11.0, 4.6))
    ax.set_xlim(0, 15.4)
    ax.set_ylim(0, 6.0)
    ax.set_aspect("equal")
    ax.axis("off")

    def box(cx, cy, w, h, title, sub, color, fc_alpha=0.10, title_fs=12.5,
            sub_fs=9.5, mono=False):
        """Rounded box centred at (cx, cy) with a bold title and a smaller
        second line (rendered in math/mono). Returns useful anchor points."""
        x, y = cx - w / 2, cy - h / 2
        for fc, a in [(color, fc_alpha), ("none", 1.0)]:
            ax.add_patch(FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
                linewidth=1.8, edgecolor=color, facecolor=fc, alpha=a))
        if sub:
            ax.text(cx, cy + 0.30, title, ha="center", va="center",
                    fontsize=title_fs, fontweight="bold", color=color)
            ax.text(cx, cy - 0.34, sub, ha="center", va="center",
                    fontsize=sub_fs, color="black",
                    family="monospace" if mono else None)
        else:
            ax.text(cx, cy, title, ha="center", va="center",
                    fontsize=title_fs, fontweight="bold", color=color)
        return dict(l=x, r=x + w, t=y + h, b=y, cx=cx, cy=cy)

    y_mid = 3.0
    y_top = 4.55
    y_bot = 1.45

    # --- shared trunk: input x -> model -> logits o ---
    xin = box(1.35, y_mid, 1.7, 1.0, r"input $\mathbf{x}$", "", GRAY,
              fc_alpha=0.08, title_fs=12)
    model = box(4.0, y_mid, 2.2, 1.2, "model", r"$f_{\mathbf{w}}$", BLUE,
                title_fs=12.5, sub_fs=14)
    # logits box shows the actual score vector
    logits = box(6.95, y_mid, 2.7, 1.3, r"logits $\mathbf{o}$",
                 rf"${o_str}$", BLUE, title_fs=12, sub_fs=12)

    fl.arrow(ax, (xin["r"], y_mid), (model["l"], y_mid), color=GRAY, lw=1.8)
    fl.arrow(ax, (model["r"], y_mid), (logits["l"], y_mid), color=GRAY, lw=1.8)

    # --- the fork: create the branch boxes FIRST, so the fork arrows can
    #     terminate exactly at their left edges (no overshoot through the box) ---
    fork_x = logits["r"]
    sm = box(10.5, y_top, 2.05, 0.95, "softmax", "", ORANGE, fc_alpha=0.10,
             title_fs=11.5)
    am = box(10.5, y_bot, 2.05, 0.95, "argmax", "", GREEN, fc_alpha=0.10,
             title_fs=11.5)
    # top (training) branch and bottom (evaluation) branch
    fl.arrow(ax, (fork_x, y_mid + 0.18), (sm["l"], y_top), color=ORANGE, lw=1.9)
    fl.arrow(ax, (fork_x, y_mid - 0.18), (am["l"], y_bot), color=GREEN, lw=1.9)

    # ---------- TOP branch: softmax -> p_hat -> cross-entropy loss ----------
    phat = box(13.6, y_top, 2.9, 1.15, r"probs $\hat{\mathbf{y}}$",
               rf"${p_str}$", ORANGE, title_fs=11.5, sub_fs=11)
    fl.arrow(ax, (sm["r"], y_top), (phat["l"], y_top), color=ORANGE, lw=1.8)

    # ---------- BOTTOM branch: argmax -> y_hat -> compare with y ----------
    cmp_sub = "correct" if correct else "wrong"
    cmp = box(13.6, y_bot, 2.9, 1.15, rf"$\hat{{y}}={yhat}$  vs  $y={y}$",
              cmp_sub, GREEN, title_fs=11, sub_fs=10.5)
    fl.arrow(ax, (am["r"], y_bot), (cmp["l"], y_bot), color=GREEN, lw=1.8)

    # ---------- branch end-labels: loss (top) and accuracy (bottom) ----------
    ax.text(13.6, y_top + 1.02,
            rf"cross-entropy loss $\ell = {loss_str}$", ha="center", va="center",
            fontsize=10.5, color=ORANGE, fontweight="bold")
    ax.text(13.6, y_bot - 1.02,
            "accuracy (count correct)", ha="center", va="center",
            fontsize=10.5, color=GREEN, fontweight="bold")

    # ---------- the one teaching line per branch ----------
    ax.text(11.55, y_top - 1.18,
            "differentiable\nused for gradient descent",
            ha="center", va="center", fontsize=9, color=ORANGE, style="italic")
    ax.text(11.55, y_bot + 1.18,
            "discrete (zero gradient)\nused for benchmarks",
            ha="center", va="center", fontsize=9, color=GREEN, style="italic")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fl.save(fig, "mdl-clf-loss-accuracy")


FIGURES = [fig_loss_accuracy]


def main():
    # Verify only the figures THIS script writes (the shared module's WRITTEN
    # list also tracks the Linear Algebra figures, which we don't run here).
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
