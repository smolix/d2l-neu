#!/usr/bin/env python3
"""Generate the illustrative figures for the "Multilayer Perceptrons" chapter
(``chapter_multilayer-perceptrons``) in the one shared house style defined in
``gen_mdl_figures.py``.

The notebooks / prose reference the generated files with no drawing code (like
the slide SVGs).  Schematic figures use ``set_aspect("equal")`` and real
numbers where the picture must be exact (e.g. the parameter count).

Run with the repo's pytorch venv (matplotlib + numpy are available):

    .venv-pytorch/bin/python tools/gen_mdl_mlp_figures.py

All figures are written to ``img/mdl-mlp-<id>.svg``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT

from matplotlib.patches import FancyBboxPatch


def fig_mlp_arch():
    """Data-flow diagram for *this section's* concrete MLP: the batched tensor
    X (n x 784) is flattened and mapped by an affine layer to a 256-dim hidden
    representation, passed through a ReLU, then mapped by a second affine layer
    to 10 logits.  Tensor states are boxes; the named operations live on the
    arrows between them.  The exact parameter count is computed, not guessed."""
    n_in, n_hid, n_out = 784, 256, 10
    p1 = n_in * n_hid + n_hid           # W1 (784 x 256) + b1 (256)
    p2 = n_hid * n_out + n_out          # W2 (256 x 10)  + b2 (10)
    total = p1 + p2
    print(f"  layer 1: {p1:,} params   layer 2: {p2:,} params   "
          f"total {total:,}")
    assert total == 203_530, total      # keep the slide deck's number honest

    fig, ax = plt.subplots(figsize=(4.4, 5.2))
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 7.7)
    ax.set_aspect("equal")
    ax.axis("off")

    cx = 3.0
    box_w, box_h = 4.2, 1.25

    def state(cy, color, head, sub):
        """A rounded tensor-state box with a bold heading and a shape caption."""
        x, y = cx - box_w / 2, cy - box_h / 2
        for fc, a in [(color, 0.12), ("none", 1.0)]:
            ax.add_patch(FancyBboxPatch(
                (x, y), box_w, box_h,
                boxstyle="round,pad=0.04,rounding_size=0.18",
                linewidth=1.8, edgecolor=color, facecolor=fc, alpha=a))
        ax.text(cx, cy + 0.24, head, ha="center", va="center",
                fontsize=13, fontweight="bold", color=color)
        ax.text(cx, cy - 0.34, sub, ha="center", va="center",
                fontsize=9.5, color="black")

    # Three tensor states, top to bottom. Centers spaced 2.75 apart: tight
    # enough to read as one compact stack, with room for the arrow labels.
    y_x, y_h, y_o = 6.6, 3.85, 1.1
    state(y_x, BLUE,   r"$\mathbf{X}$",  r"input  $n \times 784$")
    state(y_h, GREEN,  r"$\mathbf{H}$",  r"hidden  $n \times 256$")
    state(y_o, ORANGE, r"$\mathbf{O}$",  r"logits  $n \times 10$")

    half = box_h / 2

    def op_arrow(y_top, y_bot, labels):
        """Vertical arrow between two states, with stacked operation labels to
        the right of the shaft."""
        tail = (cx, y_top - half)
        tip = (cx, y_bot + half)
        fl.arrow(ax, tail, tip, color=GRAY, lw=2.0, mut=16)
        ymid = (tail[1] + tip[1]) / 2
        gap = 0.42
        y0 = ymid + gap * (len(labels) - 1) / 2
        for i, (txt, col, weight) in enumerate(labels):
            ax.text(cx + 0.45, y0 - i * gap, txt, ha="left", va="center",
                    fontsize=10.5, color=col, fontweight=weight)

    # X -> H : affine (784->256) then the nonlinearity.
    op_arrow(y_x, y_h, [
        (r"affine  $784 \to 256$", "black", "normal"),
        (r"ReLU", ORANGE, "bold"),
    ])
    # H -> O : second affine (256->10).
    op_arrow(y_h, y_o, [
        (r"affine  $256 \to 10$", "black", "normal"),
    ])

    ax.text(cx, 0.2,
            f"{total:,} parameters "
            r"($\mathbf{W}^{(1)}, \mathbf{b}^{(1)}, "
            r"\mathbf{W}^{(2)}, \mathbf{b}^{(2)}$)",
            ha="center", va="center", fontsize=9.5, color=GRAY)
    fl.save(fig, "mdl-mlp-arch")


FIGURES = [fig_mlp_arch]


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
