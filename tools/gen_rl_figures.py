#!/usr/bin/env python3
"""Generate the illustrative figures for the Reinforcement Learning chapter
(chapter_reinforcement-learning) in the shared mdl house style.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_rl_figures.py

All figures are written to ``img/mdl-rl-<id>.svg``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared rcParams/hashsalt

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT


def fig_policy_vs_parameter():
    """Achiam's two-action example: two parameter updates of the same size,
    shown as the action distributions before and after each update.  The
    numbers are computed, not sketched: pi(a=1) = sigmoid(theta)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    x, width = np.arange(2), 0.34
    for ax, start in zip(axes, [0.0, 6.0]):
        for i, (th, color) in enumerate([(start, BLUE), (start + 2, ORANGE)]):
            p1 = 1.0 / (1.0 + np.exp(-th))
            ax.bar(x + (i - 0.5) * width, [p1, 1.0 - p1], width,
                   color=color, label=rf"$\theta = {th:g}$")
        ax.set_xticks(x)
        ax.set_xticklabels([r"$a = 1$", r"$a = 2$"], fontsize=13, color="black")
        ax.set_title(rf"$\theta: {start:g} \to {start + 2:g}$",
                     fontsize=14, color="black")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="y", labelsize=11)
        ax.legend(fontsize=12, loc="upper right")
    axes[0].set_ylabel(r"$\pi_\theta(a)$", fontsize=14, color="black")
    fig.subplots_adjust(wspace=0.12)
    fl.save(fig, "rl-policy-vs-parameter")


FIGURES = [
    fig_policy_vs_parameter,
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
            head = fh.read(400)
        assert "<svg" in head, f"not valid SVG: {p}"
        print(f"  {os.path.basename(p):34s} {size:>8,d} bytes")

    print(f"\nAll {len(written)} SVGs verified present, non-empty, valid.")


if __name__ == "__main__":
    main()
