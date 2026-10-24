#!/usr/bin/env python3
"""Generate the illustrative figures for the "Preliminaries" chapter
(``chapter_preliminaries``) that are not promoted slide SVGs, in the one
shared house style.

Three figures for the linear-algebra section: the cosine-similarity panels
(two unit vectors at angle theta with their dot product cos(theta), for
theta in {0, 60, 90, 180} degrees), a 2x2 matrix acting on the unit square
(rotation, scaling, shear), and the eigenvectors of a symmetric 2x2 matrix
with the directions of its powers (power iteration).

The shared style lives in ``gen_mdl_figures.py``; importing it applies the
``plt.rcParams`` (fixed ``svg.hashsalt`` + ``metadata={'Date': None}`` in
``save()`` make re-runs byte-for-byte identical) and exposes the palette and
the drawing helpers (``arrow``/``right_angle``/``clean_axes``/``axis_cross``/
``vlabel``).  The angles and cosines are computed, not sketched.

Run with the repo's pytorch venv (matplotlib + numpy are available):

    .venv-pytorch/bin/python tools/gen_mdl_prelim_figures.py

All figures are written to ``img/mdl-prelim-<id>.svg``.
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle.mpl as fl  # the unified style (docs/figure-style-guide.md)

BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT
arrow, vlabel, right_angle = fl.arrow, fl.vlabel, fl.right_angle
clean_axes, axis_cross, save = fl.clean_axes, fl.axis_cross, fl.save

fl.use_style(hashsalt="mdl-la")

from matplotlib.patches import Arc


# --------------------------------------------------------------------------- #
# LINEAR ALGEBRA (chapter_preliminaries/linear-algebra.md)                   #
# --------------------------------------------------------------------------- #

def fig_cosine():
    """Four panels: unit vectors u (fixed along the x-axis) and v at angle
    theta in {0, 60, 90, 180} degrees, with the dot product u.v = cos(theta)
    annotated above each panel, from 1 (aligned) to -1 (opposed)."""
    # ORIGINAL layout (Alex, ch2 review: "the original layout was
    # excellent") — full-width panels, full equations above each; only the
    # colors and line weights come from the unified tokens.
    thetas_deg = [0, 60, 90, 180]
    fig, axes = plt.subplots(1, len(thetas_deg), figsize=(9.6, 2.9))

    u = np.array([1.0, 0.0])
    for ax, deg in zip(axes, thetas_deg):
        th = np.deg2rad(deg)
        v = np.array([np.cos(th), np.sin(th)])
        cos = float(u @ v)  # computed, not typed in

        # faint upper unit-circle arc for scale (both vectors have length 1)
        circ = np.linspace(np.deg2rad(-12), np.deg2rad(192), 200)
        ax.plot(np.cos(circ), np.sin(circ), color=LIGHT, lw=1.0, zorder=1)

        # the two unit vectors; nudge v slightly upward when theta = 0 so
        # the coinciding arrows stay individually visible
        arrow(ax, (0, 0), u, color=BLUE, lw=2.6)
        if deg == 0:
            eps = 0.055
            arrow(ax, (0, eps), v + (0, eps), color=ORANGE, lw=2.6)
        else:
            arrow(ax, (0, 0), v, color=ORANGE, lw=2.6)
        vlabel(ax, (1.14, -0.16 if deg == 0 else 0.0), r"$\mathbf{u}$",
               color=fl.T.BLUE.dark, ha="left", fontsize=15)
        voff = v * 1.16
        if deg == 0:
            voff = np.array([1.14, 0.20])
        elif deg == 180:                 # left of the tip, off the circle
            voff = np.array([-1.13, 0.0])
        vlabel(ax, voff, r"$\mathbf{v}$", color=fl.T.ORANGE.dark,
               ha="left" if v[0] >= 0 else "right", fontsize=15)

        # the angle arc + right-angle marker at 90 degrees
        if deg == 90:
            right_angle(ax, (0, 0), u, v, size=0.16)
        elif deg > 0:
            ax.add_patch(Arc((0, 0), 0.66, 0.66, angle=0, theta1=0,
                             theta2=deg, color=GRAY, lw=1.2))
            # on the true bisector, pushed out well past the arc so the
            # label's clearance to BOTH bounding rays (u and v) stays large
            # even for the narrower 60-degree wedge
            mid = np.deg2rad(deg / 2.0)
            ax.text(0.68 * np.cos(mid), 0.68 * np.sin(mid),
                    rf"${deg}\!^\circ$", color=GRAY,
                    ha="center", va="center", fontsize=13)

        # annotation: the dot product IS the cosine
        val = f"{cos:.2f}".rstrip("0").rstrip(".")
        if val == "-0":
            val = "0"
        ax.set_title(
            rf"$\mathbf{{u}}^\top \mathbf{{v}} = \cos {deg}\!^\circ = {val}$",
            fontsize=13.5, pad=8)

        clean_axes(ax, lim=((-1.5, 1.45), (-0.32, 1.3)), hide=True)

    fig.subplots_adjust(wspace=0.25)
    save(fig, "mdl-prelim-cosine")


def fig_linear_maps():
    """POINT: a 2x2 matrix sends the unit square to the parallelogram spanned
    by its columns A e1 and A e2.  Three panels at one shared scale:
    rotation by 90 degrees, scaling by 2 and 1/2, a shear.
    The scaling keeps the directions of e1 and e2 and the shear that of e1,
    which the eigenvector subsection points back to."""
    from matplotlib.patches import Polygon

    # (matrix, label position + alignment for A e1, for A e2), placed by
    # hand after rendering: outside each tip, clear of axes and edges
    maps = [(np.array([[0.0, -1.0], [1.0, 0.0]]),     # rotation by 90 deg
             ((0.12, 1.3), "left"), ((-1.05, -0.3), "center")),
            (np.array([[2.0, 0.0], [0.0, 0.5]]),      # scaling
             ((2.0, -0.3), "center"), ((-0.14, 0.66), "right")),
            (np.array([[1.0, 1.0], [0.0, 1.0]]),      # shear
             ((1.0, -0.3), "center"), ((1.0, 1.3), "center"))]
    lim = ((-1.45, 2.45), (-0.55, 1.55))
    span = [lim[0][1] - lim[0][0]] * 3
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.6),
                             gridspec_kw=dict(width_ratios=span, wspace=0.12))
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    for ax, (A, lab1, lab2) in zip(axes, maps):
        axis_cross(ax, lim[0], lim[1], color="black")
        ax.add_patch(Polygon(square, closed=True, fill=False, ls=(0, (4, 3)),
                             ec=GRAY, lw=1.5, zorder=2))
        img = square @ A.T
        ax.add_patch(Polygon(img, closed=True, fc=fl.T.GREEN.tint,
                             ec=fl.T.GREEN.base, lw=1.8, zorder=3))
        arrow(ax, (0, 0), A[:, 0], color=BLUE, lw=2.8)
        arrow(ax, (0, 0), A[:, 1], color=ORANGE, lw=2.8)
        for (pos, ha), text, col in (
                (lab1, r"$\mathbf{A}\mathbf{e}_1$", fl.T.BLUE.dark),
                (lab2, r"$\mathbf{A}\mathbf{e}_2$", fl.T.ORANGE.dark)):
            ax.text(pos[0], pos[1], text, color=col, fontsize=16, ha=ha,
                    va="center", zorder=5)
        clean_axes(ax, lim=lim, hide=True)
    save(fig, "mdl-prelim-linear-maps")


def fig_eigen():
    """POINT (left): S = [[2, 1], [1, 2]] maps the unit circle to an ellipse
    whose axes are its eigenvectors, stretched by 3 and 1; a vector off the
    axes (e1 -> S e1 = (2, 1)) is turned.  (right): the directions of
    S^k e1, k = 0..4, turn toward the dominant eigenvector (1, 1).
    Both panels are square windows so they render at the same height.  No
    coordinate axes: e1 lies along the x-axis and would vanish into it."""
    S = np.array([[2.0, 1.0], [1.0, 2.0]])
    v3 = np.array([1.0, 1.0]) / np.sqrt(2)          # eigenvalue 3
    v1 = np.array([-1.0, 1.0]) / np.sqrt(2)         # eigenvalue 1
    assert np.allclose(S @ v3, 3 * v3) and np.allclose(S @ v1, v1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 4.0),
                                   gridspec_kw=dict(wspace=0.08))
    t = np.linspace(0, 2 * np.pi, 400)
    circle = np.stack([np.cos(t), np.sin(t)])
    ell = S @ circle
    # left: unit circle, its image, the eigen-axes, one turned vector
    L1 = ((-3.3, 3.3), (-3.3, 3.3))
    ax1.plot(circle[0], circle[1], color=GRAY, lw=1.5, ls=(0, (4, 3)))
    ax1.plot(ell[0], ell[1], color=BLUE, lw=2.6)
    arrow(ax1, (0, 0), 3 * v3, color=GREEN, lw=2.8)
    arrow(ax1, (0, 0), 1 * v1, color=GREEN, lw=2.8)
    ax1.text(*(3 * v3 + 0.15 * np.array([-1, 1]) + 0.25 * v3),
             r"$\lambda = 3$", color=fl.T.GREEN.dark, fontsize=16,
             ha="right", va="bottom")
    ax1.text(-1.25, 1.12, r"$\lambda = 1$", color=fl.T.GREEN.dark,
             fontsize=16, ha="center", va="center")
    e1 = np.array([1.0, 0.0])
    Se1 = S @ e1
    arrow(ax1, (0, 0), e1, color=fl.T.INK, lw=2.2)
    ax1.text(0.42, -0.18, r"$\mathbf{e}_1$", color=fl.T.INK, fontsize=16,
             ha="center", va="top")
    arrow(ax1, (0, 0), Se1, color=ORANGE, lw=2.6)
    ax1.text(Se1[0] + 0.12, Se1[1] - 0.12, r"$\mathbf{S}\mathbf{e}_1$",
             color=fl.T.ORANGE.dark, fontsize=16, ha="left", va="top")
    clean_axes(ax1, lim=L1, hide=True)
    # right: normalized iterates S^k e1 / ||S^k e1|| turn toward (1, 1)
    L2 = ((-0.3, 1.35), (-0.3, 1.35))
    ax2.plot([0, 1.25 * v3[0]], [0, 1.25 * v3[1]], color=GREEN, lw=1.6,
             ls=(0, (4, 3)))
    x = e1.copy()
    for k, alpha in enumerate((0.35, 0.5, 0.65, 0.82, 1.0)):
        u = x / np.linalg.norm(x)
        arrow(ax2, (0, 0), u, color=BLUE, lw=2.6, alpha=alpha)
        if k <= 2:                     # k = 0 is horizontal: label below
            pos, va = ((u[0], u[1] - 0.09), "top") if k == 0 else \
                ((u[0] + 0.06, u[1] - 0.1), "center")
            ax2.text(pos[0], pos[1], rf"$k = {k}$", color=fl.T.BLUE.dark,
                     fontsize=15, ha="center" if k == 0 else "left", va=va)
        x = S @ x
    ax2.text(1.25 * v3[0] - 0.05, 1.25 * v3[1] + 0.02, r"$(1, 1)$",
             color=fl.T.GREEN.dark, fontsize=16, ha="right", va="bottom")
    clean_axes(ax2, lim=L2, hide=True)
    save(fig, "mdl-prelim-eigen")


FIGURES = [
    fig_cosine,
    fig_linear_maps,
    fig_eigen,
]


def main():
    # Only verify the figures this script writes (the shared module tracks all
    # figures via the same WRITTEN list).
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
        print(f"  {os.path.basename(p):30s} {size:>8,d} bytes")

    print(f"\nAll {len(written)} SVGs verified present, non-empty, valid.")


if __name__ == "__main__":
    main()
