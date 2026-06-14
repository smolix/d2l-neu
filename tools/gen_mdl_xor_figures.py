#!/usr/bin/env python3
"""Generate the illustrative figures for the "Multilayer Perceptrons"
chapter (``chapter_multilayer-perceptrons``) in the one shared house style
defined in ``gen_mdl_figures.py``.

The prose references the generated files with no drawing code (like the slide
SVGs).  Figures that show a computed result use real numerical computation, so
the pictures are exact, not sketched: the XOR feature map below is the actual
forward pass of the hand-built two-ReLU network, not a hand-placed cartoon.

Run with the repo's pytorch venv (matplotlib + numpy are available):

    .venv-pytorch/bin/python tools/gen_mdl_xor_figures.py

All figures are written to ``img/mdl-mlp-<id>.svg``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT


def fig_xor():
    """XOR is not linearly separable, but one ReLU hidden layer fixes it.

    Left: the four inputs of the unit square, coloured by XOR label
    (class 0 = same bits, class 1 = differing bits).  No straight line
    separates the two orange corners from the two blue ones; we draw one
    candidate line and shade its (wrong) decision, to make the failure visible.

    Right: the SAME four points after the hidden layer
        h = ReLU(X W1 + b1),  W1 = [[1,1],[1,1]],  b1 = (0, -1),
    a textbook construction.  The map sends the two class-1 corners (0,1) and
    (1,0) onto the *same* hidden point (1, 0), so the cloud becomes linearly
    separable: a single line h1 - 2 h2 = 1/2 now splits the classes, i.e. the
    output neuron O = h1 - 2 h2 reproduces XOR.  Everything is computed.
    """
    # XOR truth table.  label = x1 XOR x2.
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1, 1, 0])  # XOR

    # Hand-built hidden layer (Goodfellow et al., Deep Learning, sec 6.1).
    W1 = np.array([[1.0, 1.0], [1.0, 1.0]])
    b1 = np.array([0.0, -1.0])
    H = np.maximum(X @ W1 + b1, 0.0)          # ReLU hidden features
    # Output neuron O = H @ w2 + b2 must realise XOR.
    w2 = np.array([1.0, -2.0])
    b2 = 0.0
    O = H @ w2 + b2
    assert np.allclose((O > 0.5).astype(int), y), (H, O)   # the net IS XOR

    col = {0: BLUE, 1: ORANGE}
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(8.8, 4.2))

    # ---- (a) input space: not linearly separable ----------------------------
    axa.set_title("(a) input space: no line works")
    # two candidate separators, each of which gets one corner wrong: the point
    # is that the orange diagonal and the blue diagonal cannot be split at once
    xs = np.array([-0.4, 1.4])
    axa.plot(xs, 1.5 - xs, "--", color=GRAY, lw=1.5)
    axa.plot(xs, 0.5 - xs + 1.0, "--", color=GRAY, lw=1.5, alpha=0.0)  # spacing
    axa.plot([-0.4, 1.4], [0.5, 0.5], "--", color=GRAY, lw=1.5)
    axa.text(1.95, 0.95, "two candidate\nlines, each\nmisclassifies\na corner",
             color=GRAY, fontsize=8.0, ha="center", va="center")
    for (x1, x2), lab in zip(X, y):
        axa.plot(x1, x2, "o", color=col[lab], ms=14, zorder=5)
        axa.text(x1, x2, "0" if lab == 0 else "1", color="white",
                 fontsize=9.5, fontweight="bold", ha="center", va="center",
                 zorder=6)
    axa.set_xlabel("$x_1$"); axa.set_ylabel("$x_2$")
    axa.set_xticks([0, 1]); axa.set_yticks([0, 1])
    axa.set_xlim(-0.5, 2.55); axa.set_ylim(-0.6, 1.6)
    axa.set_aspect("equal")
    axa.spines["top"].set_visible(False)
    axa.spines["right"].set_visible(False)

    # ---- (b) hidden space: separable, with the two class-1 corners merged ----
    axb.set_title(r"(b) hidden space $h=\mathrm{ReLU}(\mathbf{x}\mathbf{W}^{(1)}+\mathbf{b}^{(1)})$")
    # the separating line  h1 - 2 h2 = 1/2  (the output neuron's decision)
    h1s = np.array([-0.4, 2.4])
    h2_line = (h1s - 0.5) / 2.0
    axb.fill_between(h1s, h2_line, 1.6, color=BLUE, alpha=0.07, lw=0)
    axb.fill_between(h1s, -0.6, h2_line, color=ORANGE, alpha=0.07, lw=0)
    axb.plot(h1s, h2_line, "--", color=GREEN, lw=1.6)
    axb.text(1.74, 0.18, r"$h_1-2h_2=\frac{1}{2}$", color=GREEN, fontsize=9,
             ha="center", va="center", rotation=np.degrees(np.arctan2(0.5, 1.0)),
             rotation_mode="anchor")
    # plot hidden points; jitter only the LABELS of the two coincident corners
    seen: dict[tuple[float, float], int] = {}
    for (h1, h2), lab, (x1, x2) in zip(H, y, X):
        key = (round(h1, 6), round(h2, 6))
        seen[key] = seen.get(key, 0) + 1
        axb.plot(h1, h2, "o", color=col[lab], ms=13, zorder=5)
        # name each hidden point by the input that produced it
        dy = 0.20 if seen[key] == 1 else -0.30   # stagger the two merged labels
        axb.text(h1 + 0.16, h2 + dy, f"$({x1:.0f},{x2:.0f})$", color=col[lab],
                 fontsize=8.5, ha="left", va="center", zorder=6)
    axb.annotate("two class-1 corners\nland on the same point",
                 xy=(1.0, 0.0), xytext=(0.30, 0.95),
                 fontsize=8.5, color=GRAY, ha="center", va="center",
                 arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.0))
    axb.set_xlabel("$h_1$"); axb.set_ylabel("$h_2$")
    axb.set_xticks([0, 1, 2]); axb.set_yticks([0, 1])
    axb.set_xlim(-0.5, 2.55); axb.set_ylim(-0.6, 1.6)
    axb.set_aspect("equal")
    axb.spines["top"].set_visible(False)
    axb.spines["right"].set_visible(False)

    print("  XOR hidden features H =\n", H)
    print("  output O =", O, " -> class", (O > 0.5).astype(int), " (target", y, ")")
    fig.subplots_adjust(wspace=0.32)
    fl.save(fig, "mdl-mlp-xor")


FIGURES = [fig_xor]


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
