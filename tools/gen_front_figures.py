#!/usr/bin/env python3
"""Illustrative figures for the front matter (preface).

Generates img/front-*.svg in the shared house style of tools/gen_mdl_figures.py
(byte-idempotent: fixed svg.hashsalt and no date via ``fl.save``).

    .venv-pytorch/bin/python tools/gen_front_figures.py

fig_book_org draws the parts of the book as panels and the chapters as boxes;
arrows mark the chapter dependencies stated in the chapter introductions.
Chapter numbers follow CHAPTER_NUMBERING in tools/d2l_preprocess.py; update
them here when the book is renumbered.
"""
import os
import sys

from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # noqa: E402  (applies the shared rcParams)

np, plt, T = fl.np, fl.plt, fl.T
INK = fl.INK

# Layout in inches (y grows downward). Five columns of chapter boxes.
PITCH_X, BOX_W = 1.80, 1.56
PITCH_Y, BOX_H = 1.08, 0.82
PAD = 0.14            # panel padding around its boxes
HEAD = 0.36           # room for the panel title above the boxes
PANEL_GAP = 0.30      # vertical gap between stacked panels
FS_BOX, FS_HEAD = 12.5, 12.0

ACCENT = {"basics": T.BLUE, "advanced": T.ORANGE,
          "appendix": T.PURPLE, "tools": T.SLATE}


def _x(col):
    """Left edge of a box in column ``col`` (0-based)."""
    return col * PITCH_X


def _box(ax, x, y, w, h, text, accent):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.07",
        facecolor=accent.tint, edgecolor=accent.base, linewidth=T.SW_BOX,
        zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=FS_BOX, color=INK, linespacing=1.05, zorder=4)
    return dict(l=x, r=x + w, t=y, b=y + h, cx=x + w / 2, cy=y + h / 2)


def _panel(ax, x0, y0, x1, y1, title):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.12",
        facecolor=T.PANEL, edgecolor="none", zorder=1))
    ax.text(x0 + PAD, y0 + 0.10, title, ha="left", va="top",
            fontsize=FS_HEAD, color=INK, fontweight="bold", zorder=2)


def _arrow(ax, pts):
    """Polyline arrow through ``pts``; the head sits on the last point."""
    xs, ys = zip(*pts)
    if len(pts) > 2:
        ax.plot(xs[:-1], ys[:-1], color=INK, lw=1.6, zorder=2,
                solid_capstyle="butt", solid_joinstyle="miter")
    ax.annotate("", xy=pts[-1], xytext=pts[-2], zorder=2,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6,
                                shrinkA=0, shrinkB=0, mutation_scale=13))


def fig_book_org():
    # The figure is sized to the layout in inches, so data units are inches
    # and the fonts keep their point sizes.
    basics_bot = HEAD + PITCH_Y + BOX_H + PAD
    adv_bot = basics_bot + PANEL_GAP + HEAD + PITCH_Y + BOX_H + PAD
    app_bot = adv_bot + PANEL_GAP + HEAD + BOX_H + PAD
    x0, x1 = -PAD - 0.02, _x(4) + BOX_W + PAD + 0.02
    y0, y1 = -0.02, app_bot + 0.02
    fig = plt.figure(figsize=(x1 - x0, y1 - y0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_aspect("equal")
    ax.axis("off")

    B = {}

    # ---------------- Basics: rows 0-1, columns 0-3 ----------------------- #
    y_b = HEAD
    basics_top = 0.0
    rows_b = [y_b, y_b + PITCH_Y]
    for key, col, row, text in [
            (1, 0, 0, "1 Preliminaries"), (2, 1, 0, "2 Linear\nRegression"),
            (3, 2, 0, "3 Linear\nClassification"),
            (4, 3, 0, "4 Multilayer\nPerceptron"),
            (8, 0, 1, "8 Sequence\nModels"), (5, 1, 1, "5 Computation"),
            (6, 2, 1, "6 Convolutional\nNeural Networks"),
            (7, 3, 1, "7 Modern\nConvnets")]:
        B[key] = _box(ax, _x(col), rows_b[row], BOX_W, BOX_H, text,
                      ACCENT["basics"])
    basics_right = _x(3) + BOX_W + PAD
    _panel(ax, -PAD, basics_top, basics_right, basics_bot, "Basics")

    # ---------------- Tools: beside Basics, right edge of column 4 -------- #
    t_x0, t_x1 = basics_right + PANEL_GAP, _x(4) + BOX_W + PAD
    t_y = (basics_top + HEAD + basics_bot - PAD) / 2 - BOX_H / 2
    B[30] = _box(ax, t_x0 + PAD, t_y, t_x1 - t_x0 - 2 * PAD, BOX_H,
                 "30 Tools for\nDeep Learning", ACCENT["tools"])
    _panel(ax, t_x0, basics_top, t_x1, basics_bot, "Tools")

    # ---------------- Advanced: rows 0-1, columns 0-4 --------------------- #
    adv_top = basics_bot + PANEL_GAP
    rows_a = [adv_top + HEAD, adv_top + HEAD + PITCH_Y]
    for key, col, row, text in [
            (10, 0, 0, "10 Attention"), (11, 1, 0, "11 Transformers"),
            (9, 2, 0, "9 Optimization\nAlgorithms"),
            (14, 3, 0, "14 Reinforcement\nLearning"),
            (16, 4, 0, "16 Generative\nAdversarial\nNetworks"),
            (12, 0, 1, "12 State Space\nModels"),
            (13, 1, 1, "13 Computational\nPerformance"),
            (15, 3, 1, "15 Deep\nReinforcement\nLearning"),
            (17, 4, 1, "17 Diffusion\nModels")]:
        B[key] = _box(ax, _x(col), rows_a[row], BOX_W, BOX_H, text,
                      ACCENT["advanced"])
    _panel(ax, -PAD, adv_top, _x(4) + BOX_W + PAD, adv_bot, "Advanced")

    # ---------------- Mathematics appendix: six boxes --------------------- #
    app_top = adv_bot + PANEL_GAP
    width = _x(4) + BOX_W
    n, gap = 6, 0.14
    w6 = (width - (n - 1) * gap) / n
    for i, text in enumerate([
            "24 Linear\nAlgebra", "25 Calculus", "26 Optimization",
            "27 Probability\nand Statistics", "28 Information\nTheory",
            "29 Dynamics"]):
        B[24 + i] = _box(ax, i * (w6 + gap), app_top + HEAD, w6, BOX_H, text,
                         ACCENT["appendix"])
    _panel(ax, -PAD, app_top, width + PAD, app_bot,
           "Mathematics for Deep Learning")

    # ---------------- Dependencies ---------------------------------------- #
    def right(a, b):
        _arrow(ax, [(B[a]["r"], B[a]["cy"]), (B[b]["l"], B[b]["cy"])])

    def left(a, b):
        _arrow(ax, [(B[a]["l"], B[a]["cy"]), (B[b]["r"], B[b]["cy"])])

    def down(a, b, x=None):
        x = B[a]["cx"] if x is None else x
        _arrow(ax, [(x, B[a]["b"]), (x, B[b]["t"])])

    right(1, 2), right(2, 3), right(3, 4)
    # line wrap: Multilayer Perceptron -> Computation
    mid = (B[4]["b"] + B[5]["t"]) / 2
    _arrow(ax, [(B[4]["cx"], B[4]["b"]), (B[4]["cx"], mid),
                (B[5]["cx"], mid), (B[5]["cx"], B[5]["t"])])
    left(5, 8), right(5, 6), right(6, 7)
    # sequence models -> attention, right of the "Advanced" panel title
    down(8, 10, x=B[8]["l"] + 0.78 * BOX_W)
    right(10, 11)
    down(10, 12)                      # attention -> state space models
    down(11, 13)                      # transformers -> performance
    # optimization -> performance: down into the empty cell, then left
    _arrow(ax, [(B[9]["cx"], B[9]["b"]), (B[9]["cx"], B[13]["cy"]),
                (B[13]["r"], B[13]["cy"])])
    down(14, 15)
    down(16, 17)

    fl.save(fig, "front-book-org")


FIGURES = [fig_book_org]


def main():
    start = len(fl.WRITTEN)
    for fn in FIGURES:
        fn()
    written = fl.WRITTEN[start:]
    print(f"Wrote {len(written)} figures to {fl.IMG_DIR}:")
    for p in written:
        assert os.path.getsize(p) > 0, f"empty: {p}"
        with open(p, encoding="utf-8") as fh:
            assert "<svg" in fh.read(2000), f"not an SVG: {p}"
        print("  ", os.path.basename(p))


if __name__ == "__main__":
    main()
