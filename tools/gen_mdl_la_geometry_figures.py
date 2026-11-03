#!/usr/bin/env python3
"""Generate the new illustrative figure of "Linear Algebra -> Vectors, Angles,
and Projections" (``chapter_mdl-linear-algebra/mdl-geometry-linear-algebraic-ops.md``)
in the shared house style of ``gen_mdl_figures.py``.

The notebook references the generated file with no drawing code.  The picture
is computed from the worked example of the text: the three-point line fit
with measurements y = (6, 0, 0) at times t = 0, 1, 2.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_la_geometry_figures.py

The figure is written to ``img/mdl-la-least-squares.svg``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, INK = fl.BLUE, fl.ORANGE, fl.GREEN, fl.INK

from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def fig_least_squares(elev=30, azim=115, name="mdl-la-least-squares"):
    """Least squares as projection onto the plane col(A) spanned by
    1 = (1,1,1) and t = (0,1,2): the target y = (6,0,0), its closest point
    (5,2,-1) on the plane, and the residual (1,-2,1) perpendicular to it."""
    one = np.array([1.0, 1.0, 1.0])
    t = np.array([0.0, 1.0, 2.0])
    y = np.array([6.0, 0.0, 0.0])
    A = np.stack([one, t], axis=1)
    coef = np.linalg.solve(A.T @ A, A.T @ y)     # (5, -3)
    fit = A @ coef                               # (5, 2, -1)
    r = y - fit                                  # (1, -2, 1)

    fig = plt.figure(figsize=(6.0, 4.6))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.computed_zorder = False

    # the plane col(A) as a rectangle in the orthonormal in-plane basis
    # e1 = fit/|fit|, e2 = t/|t| (t is orthogonal to the fit)
    e1, e2 = fit / np.linalg.norm(fit), t / np.linalg.norm(t)
    corners = [a * e1 + b * e2 for a, b in
               [(-0.9, -0.9), (6.6, -0.9), (6.6, 2.9), (-0.9, 2.9)]]
    patch = Poly3DCollection([corners], facecolor=BLUE, alpha=0.14,
                             edgecolor=BLUE, linewidth=0.8)
    ax.add_collection3d(patch)

    def arrow3(tail, tip, color, lw=2.2):
        d = np.asarray(tip) - np.asarray(tail)
        ax.quiver(*tail, *d, color=color, lw=lw, arrow_length_ratio=0.12)

    o = np.zeros(3)
    arrow3(o, one, GREEN)
    arrow3(o, t, GREEN)
    arrow3(o, fit, ORANGE)
    arrow3(o, y, INK, lw=1.4)
    # residual from the fitted point up to y, with a right-angle mark
    ax.plot(*np.stack([fit, y], axis=1), "--", color=INK, lw=1.6)
    u = -e1
    n = r / np.linalg.norm(r)
    s = 0.5
    sq = np.stack([fit + s * u, fit + s * (u + n), fit + s * n], axis=1)
    ax.plot(*sq, color=INK, lw=1.0)
    ax.scatter(*fit, color=ORANGE, s=22, depthshade=False)
    ax.scatter(*y, color=INK, s=22, depthshade=False)

    box = dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5)
    ax.text(*(y + 0.55 * n + 0.3 * e1), r"$\mathbf{y}=(6,0,0)$", color=INK,
            fontsize=15)
    ax.text(*(fit - 1.6 * e2 + 0.6 * e1), r"$\mathbf{A}\hat{\mathbf{x}}=(5,2,-1)$",
            color=ORANGE, fontsize=15, bbox=box)
    ax.text(*(0.5 * (fit + y) + 0.35 * e1), r"$\mathbf{r}$", color=INK,
            fontsize=16)
    ax.text(*(1.3 * one + 0.25 * e2), r"$\mathbf{1}$", color=GREEN, fontsize=16,
            bbox=box)
    ax.text(*(1.15 * t), r"$\mathbf{t}$", color=GREEN, fontsize=16,
            bbox=box)
    ax.text(*(3.2 * e1 + 2.4 * e2), r"$\mathrm{col}(\mathbf{A})$",
            color=BLUE, fontsize=15)

    pts = np.array(corners + [y])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_color(INK)
    ax.view_init(elev=elev, azim=azim)
    try:
        ax.set_box_aspect(tuple(hi - lo), zoom=1.0)
    except Exception:
        pass
    fl.save(fig, name)


FIGURES = [fig_least_squares]


def main():
    for fn in FIGURES:
        fn()
    for p in fl.WRITTEN:
        print(f"  {os.path.basename(p):28s} {os.path.getsize(p):>8,d} bytes")


if __name__ == "__main__":
    main()
