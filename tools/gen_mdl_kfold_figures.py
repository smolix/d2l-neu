#!/usr/bin/env python3
"""Generate the illustrative K-fold cross-validation figure for the "Multilayer
Perceptrons -> Predicting House Prices on Kaggle" section
(``chapter_multilayer-perceptrons/kaggle-house-price.md``) in the one shared
house style defined in ``gen_mdl_figures.py``.

The notebook / prose references the generated file with no drawing code (like
the slide SVGs). The picture is schematic but its geometry is computed exactly
from ``K`` so the folds line up perfectly.

Run with the repo's pytorch venv (matplotlib + numpy are available):

    .venv-pytorch/bin/python tools/gen_mdl_kfold_figures.py

All figures are written to ``img/mdl-mlp-<id>.svg``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT

from matplotlib.patches import Rectangle


def fig_kfold():
    """K-fold cross-validation, K=5. The dataset is the horizontal axis,
    partitioned into K equal folds. Each row is one training round: fold i is
    held out for validation (orange) while the remaining K-1 folds train the
    model (blue). The final estimate averages the K validation scores -- the
    brace on the right. Geometry is computed from K, so the cells tile exactly.
    """
    K = 5
    fig, ax = plt.subplots(figsize=(7.6, 3.9))

    cell_w, cell_h = 1.0, 0.62      # one fold cell
    gap_y = 0.34                    # vertical gap between rounds
    row_h = cell_h + gap_y
    x0 = 0.0                        # left edge of the strips
    # rounds drawn top-to-bottom: round 1 on top
    y_of = lambda i: (K - 1 - i) * row_h

    for i in range(K):              # i = held-out (validation) fold, round i+1
        y = y_of(i)
        for j in range(K):          # j indexes the folds along the strip
            x = x0 + j * cell_w
            is_val = (j == i)
            fc = ORANGE if is_val else BLUE
            ax.add_patch(Rectangle((x, y), cell_w, cell_h, facecolor=fc,
                                    alpha=0.85 if is_val else 0.22,
                                    edgecolor="white", lw=1.6, zorder=2))
        # round label to the left of each strip
        ax.text(x0 - 0.28, y + cell_h / 2, rf"round ${i+1}$", ha="right",
                va="center", fontsize=10.5, color=GRAY)

    strip_w = K * cell_w
    y_top = y_of(0) + cell_h
    y_bot = y_of(K - 1)

    # fold indices along the bottom
    for j in range(K):
        ax.text(x0 + (j + 0.5) * cell_w, y_bot - 0.30, rf"${j+1}$",
                ha="center", va="top", fontsize=10, color=GRAY)
    ax.text(x0 + strip_w / 2, y_bot - 0.74, r"data split into $K=5$ folds",
            ha="center", va="top", fontsize=10.5, color=GRAY)

    # small in-figure legend (a validation chip and a training chip), placed
    # above the strips so it never overlaps a row
    ly = y_top + 0.36
    lx = x0
    ax.add_patch(Rectangle((lx, ly), 0.42, cell_h * 0.6, facecolor=ORANGE,
                           alpha=0.85, edgecolor="white", lw=1.2))
    ax.text(lx + 0.56, ly + cell_h * 0.3, "validation fold", ha="left",
            va="center", fontsize=10, color="black")
    lx2 = lx + 3.5
    ax.add_patch(Rectangle((lx2, ly), 0.42, cell_h * 0.6, facecolor=BLUE,
                           alpha=0.22, edgecolor="white", lw=1.2))
    ax.text(lx2 + 0.56, ly + cell_h * 0.3, "training folds", ha="left",
            va="center", fontsize=10, color="black")

    # brace + label on the right: the K validation scores are averaged
    bx = x0 + strip_w + 0.45
    fl.arrow(ax, (bx, y_bot + cell_h / 2), (bx, y_top - cell_h / 2),
             color=GRAY, lw=1.2)
    fl.arrow(ax, (bx, y_top - cell_h / 2), (bx, y_bot + cell_h / 2),
             color=GRAY, lw=1.2)
    ax.text(bx + 0.22, (y_top + y_bot) / 2 - cell_h / 2,
            "average the\n$K$ validation\nscores", ha="left", va="center",
            fontsize=10, color=GRAY)

    ax.set_xlim(x0 - 1.7, x0 + strip_w + 2.7)
    ax.set_ylim(y_bot - 1.05, ly + cell_h * 0.6 + 0.2)
    ax.set_aspect("equal")
    ax.axis("off")
    fl.save(fig, "mdl-mlp-kfold")


FIGURES = [fig_kfold]


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
