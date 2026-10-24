#!/usr/bin/env python3
"""Matplotlib figures for chapter_preliminaries/calculus.md.

calculus-gradient-field.svg -- POINT: the gradient (red, the book's gradient
  colour) is perpendicular to the level sets and points uphill; gradient
  descent starts along -grad L, which does not point at the minimum, and
  zigzags toward it.  Drawn for the
  section's running example, the line model with loss
  L(w) = ||X w - y||^2, X = [[1, 0], [1, 1], [1, 2]], y = [1, 1, 0],
  so the numbers in the text (grad L(1, 0) = (2, 4), first step to
  (0.8, -0.4), minimum (7/6, -1/2)) can be read off the picture.

calculus-partial-slices.svg -- POINT: a partial derivative is the slope of
  a 1-D slice of the surface.  The same loss in 3-D near (1, 0): the slice
  w2 = 0 (the constant model's loss) and the slice w1 = 1, with dashed
  tangents of slopes 2 and 4.  Default 3-D box and panes, black axis lines.

Both figures are used in the book (img/) and on the slides (img/auto/, via
@fig:), so they are written to both.  The JS registry (diagrams/calculus.mjs)
no longer defines these ids, so remake_ch02.py's JS pass does not overwrite
them.

Run:  .venv-pytorch/bin/python tools/figstyle/remake_ch02_calculus.py
"""

from __future__ import annotations

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from figstyle import tokens as T

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMG = os.path.join(REPO, "img")
AUTO = os.path.join(IMG, "auto")

X = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]
Y = [1.0, 1.0, 0.0]


def fig_gradient_field():
    import matplotlib
    matplotlib.use("svg")
    import matplotlib.pyplot as plt
    import numpy as np

    from figstyle.mpl import arrow, save, use_style

    use_style(hashsalt="figstyle-ch02-calculus")
    Xm, y = np.array(X), np.array(Y)
    H = Xm.T @ Xm                                # L(w) = (w-w*)' H (w-w*) + L*
    w_star = np.linalg.solve(H, Xm.T @ y)       # (7/6, -1/2)

    def loss(w):
        return float(np.sum((Xm @ w - y) ** 2))

    def grad(w):
        return 2 * Xm.T @ (Xm @ w - y)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    xlim, ylim = (0.22, 2.14), (-1.3, 0.36)
    g1, g2 = np.meshgrid(np.linspace(*xlim, 400), np.linspace(*ylim, 400))
    D1, D2 = g1 - w_star[0], g2 - w_star[1]
    Z = (H[0, 0] * D1 ** 2 + 2 * H[0, 1] * D1 * D2 + H[1, 1] * D2 ** 2
         + loss(w_star))
    # No level between 0.32 and 0.7, nor between 2.0 and 4.3: the labels
    # -grad L and grad L sit in those bands, and "minimum" inside L < 0.7,
    # so no text lies on a contour line.
    levels = [0.7, 1.0, 1.5, 2.0]
    ax.contour(g1, g2, Z, levels=levels, colors=T.FAINT,
               linewidths=T.SW_HAIR, zorder=1)

    # Gradients (true direction and relative length) at points of the
    # level set L = 1, which passes through the starting point (1, 0).  The
    # angles spread the arrows around the ellipse and keep every arrowhead
    # inside the axes.
    evals, evecs = np.linalg.eigh(H)
    c = 1.0 - loss(w_star)
    scale = 0.055
    for ang in (15, 120, 200, 270, 315):
        t = np.radians(ang)
        d = (evecs[:, 0] * np.cos(t) / np.sqrt(evals[0])
             + evecs[:, 1] * np.sin(t) / np.sqrt(evals[1])) * np.sqrt(c)
        p = w_star + d
        arrow(ax, p, p + scale * grad(p), color=T.RED.base, lw=T.SW_LINE)

    # Gradient descent from (1, 0) with learning rate 0.1.
    w0 = np.array([1.0, 0.0])
    path = [w0]
    for _ in range(25):
        path.append(path[-1] - 0.1 * grad(path[-1]))
    path = np.array(path)
    ax.plot(path[1:, 0], path[1:, 1], color=T.ORANGE.base, lw=T.SW_LINE,
            zorder=3)
    ax.plot(path[1:12, 0], path[1:12, 1], "o", color=T.ORANGE.base, ms=4,
            zorder=4)
    arrow(ax, path[0], path[1], color=T.ORANGE.base, lw=T.SW_LINE)
    arrow(ax, w0, w0 + scale * grad(w0), color=T.RED.base, lw=T.SW_LINE)
    ax.plot(*w0, "o", color=T.INK, ms=5.5, zorder=5)
    ax.plot(*w_star, "o", color=T.GREEN.base, ms=7, zorder=5)

    ax.text(1.16, 0.22, r"$\nabla L$", fontsize=16, color=T.RED.dark,
            ha="left", va="center")
    ax.text(0.86, -0.15, r"$-\nabla L$", fontsize=16, color=T.ORANGE.dark,
            ha="right", va="center", zorder=6)
    ax.text(w_star[0], w_star[1] - 0.045, "minimum", fontsize=15,
            color=T.GREEN.dark, ha="center", va="top", zorder=6)

    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"$w_1$", color="black", fontsize=17)
    ax.set_ylabel(r"$w_2$", color="black", fontsize=17, rotation=0,
                  labelpad=12)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("black")
    ax.tick_params(colors="black", labelsize=13)
    save(fig, "calculus-gradient-field", out_dir=IMG)
    shutil.copyfile(os.path.join(IMG, "calculus-gradient-field.svg"),
                    os.path.join(AUTO, "calculus-gradient-field.svg"))


def fig_partial_slices():
    import matplotlib
    matplotlib.use("svg")
    import matplotlib.pyplot as plt
    import numpy as np

    from figstyle.mpl import save, use_style

    use_style(hashsalt="figstyle-ch02-calculus")
    Xm, y = np.array(X), np.array(Y)

    def loss(w1, w2):
        return ((w1 - y[0]) ** 2 + (w1 + w2 - y[1]) ** 2
                + (w1 + 2 * w2 - y[2]) ** 2)

    fig = plt.figure(figsize=(5.4, 4.4))
    ax = fig.add_subplot(projection="3d")
    ax.view_init(elev=26, azim=-58)
    g1, g2 = np.meshgrid(np.linspace(0.6, 1.4, 17), np.linspace(-0.4, 0.4, 17))
    ax.plot_wireframe(g1, g2, loss(g1, g2), color=T.FAINT, lw=0.8,
                      rstride=2, cstride=2)
    t = np.linspace(0.6, 1.4, 81)
    s2 = np.linspace(-0.4, 0.4, 81)
    ax.plot(t, 0 * t, loss(t, 0), color=T.ORANGE.base, lw=T.SW_LINE)
    ax.plot(1 + 0 * s2, s2, loss(1, s2), color=T.GREEN.base, lw=T.SW_LINE)
    d = np.linspace(-1, 1, 2)          # dashed tangents at (1, 0): slopes 2, 4
    ax.plot(1 + 0.4 * d, 0 * d, 1 + 2 * 0.4 * d, color=T.ORANGE.dark,
            lw=T.SW_LINE, ls=(0, (4, 2)))
    ax.plot(1 + 0 * d, 0.25 * d, 1 + 4 * 0.25 * d, color=T.GREEN.dark,
            lw=T.SW_LINE, ls=(0, (4, 2)))
    ax.scatter([1], [0], [1], color=T.INK, s=30, depthshade=False)
    ax.text2D(0.03, 0.90, r"slope $\partial L/\partial w_2 = 4$",
              transform=ax.transAxes, color=T.GREEN.dark, fontsize=15)
    ax.text2D(0.03, 0.82, r"slope $\partial L/\partial w_1 = 2$",
              transform=ax.transAxes, color=T.ORANGE.dark, fontsize=15)
    ax.text2D(0.86, 0.80, r"$L$", transform=ax.transAxes, color="black",
              fontsize=16, ha="center")
    ax.set_xlabel(r"$w_1$", fontsize=16, labelpad=4)
    ax.set_ylabel(r"$w_2$", fontsize=16, labelpad=4)
    ax.set_xticks([0.6, 1.0, 1.4])
    ax.set_yticks([-0.4, 0.0, 0.4])
    ax.set_zticks([0, 2, 4])
    ax.tick_params(labelsize=11, colors="black", pad=1)
    ax.set_box_aspect(None, zoom=0.88)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_color("black")
        axis.label.set_color("black")
    save(fig, "calculus-partial-slices", out_dir=IMG)
    shutil.copyfile(os.path.join(IMG, "calculus-partial-slices.svg"),
                    os.path.join(AUTO, "calculus-partial-slices.svg"))


def main():
    fig_partial_slices()
    fig_gradient_field()
    print("calculus figures refreshed")


if __name__ == "__main__":
    main()
