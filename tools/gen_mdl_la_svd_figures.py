#!/usr/bin/env python3
"""Generate the new illustrative figures of "Singular Value Decomposition and
Low-Rank Approximation" (``chapter_mdl-linear-algebra/mdl-svd-low-rank.md``) in
the chapter's shared house style.

The shared style lives in ``gen_mdl_figures.py``; importing it applies the
``plt.rcParams`` (fixed ``svg.hashsalt`` and ``metadata={'Date': None}`` in
``save()`` make re-runs byte-for-byte identical) and exposes the palette and
the drawing helpers.  The section's older figures (``mdl-la-svd-action``,
``mdl-la-svd-subspaces``, ``mdl-la-eckart-young``, ``mdl-la-lora``,
``mdl-la-condition``, ``mdl-la-pca``) stay in that generator.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_la_svd_figures.py

Writes ``img/mdl-la-svd-dimension-count.svg`` and ``img/mdl-la-min-norm.svg``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, INK = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.INK
arrow, clean_axes, axis_cross, right_angle = (
    fl.arrow, fl.clean_axes, fl.axis_cross, fl.right_angle)
save = fl.save


def _plane_basis(normal):
    """Two orthonormal vectors spanning the plane with the given normal."""
    nrm = np.asarray(normal, float) / np.linalg.norm(normal)
    a = np.cross(nrm, [0.0, 0.0, 1.0] if abs(nrm[2]) < 0.9 else [1.0, 0.0, 0.0])
    a /= np.linalg.norm(a)
    return a, np.cross(nrm, a)


def fig_svd_dimension_count():
    """Eckart--Young's dimension count for n = 3, k = 1: the plane ker B and
    the plane span{v1, v2} meet in a line; on it B x = 0 while |A x| >= s2."""
    n_ker = np.array([0.3, 0.2, 1.0])      # ker B: a nearly horizontal plane
    n_top = np.array([1.0, -0.4, 0.3])     # span{v1, v2}: a nearly vertical one
    d = np.cross(n_ker, n_top)
    d /= np.linalg.norm(d)
    if d[1] < 0:
        d = -d

    fig = plt.figure(figsize=(6.8, 4.6))
    ax = fig.add_subplot(projection="3d")
    ax.computed_zorder = False
    s = np.linspace(-1.0, 1.0, 2)
    S, T = np.meshgrid(s, s)
    for normal, col, alpha in [(n_ker, BLUE, 0.30), (n_top, ORANGE, 0.30)]:
        a, b = _plane_basis(normal)
        # a square patch of the plane, scaled to stay inside the box
        P = 1.0 * (S[..., None] * a + T[..., None] * b)
        ax.plot_surface(P[..., 0], P[..., 1], P[..., 2], color=col,
                        alpha=alpha, linewidth=0, antialiased=True)
    # the intersection line, through the whole box, and a unit vector on it
    L = 1.25
    ax.plot([-L * d[0], L * d[0]], [-L * d[1], L * d[1]], [-L * d[2], L * d[2]],
            color=INK, lw=1.6, ls="--")
    ax.quiver(0, 0, 0, d[0], d[1], d[2], color=INK, lw=2.4,
              arrow_length_ratio=0.18)
    ax.scatter([0], [0], [0], color=INK, s=14, depthshade=False)

    # labels in screen coordinates, in empty corners clear of the box edges
    ax.text2D(0.005, 0.36, r"$\ker\mathbf{B}$" + "\n" + r"(dim $n-k=2$)",
              transform=ax.transAxes, color=BLUE, fontsize=14,
              ha="left", va="center",
              bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none",
                        alpha=0.9))
    ax.text2D(0.005, 0.97,
              r"$\mathrm{span}\{\mathbf{v}_1,\mathbf{v}_2\}$" + "\n"
              + r"(dim $k+1=2$)",
              transform=ax.transAxes, color=ORANGE, fontsize=14,
              ha="left", va="top")
    # the unit vector's label sits on a white patch above its tip, inside
    # the frame (clear of the box's vertical edge on the right)
    ax.text2D(0.76, 0.60,
              r"$\mathbf{x}$: $\mathbf{B}\mathbf{x}=\mathbf{0}$," + "\n"
              + r"$\|\mathbf{A}\mathbf{x}\|\geq\sigma_2$",
              transform=ax.transAxes, color=INK, fontsize=14,
              ha="right", va="bottom",
              bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none",
                        alpha=0.9))

    lim = 1.2
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_color(INK)          # blacken the axis lines only
    ax.view_init(elev=20, azim=-62)
    try:
        ax.set_box_aspect((1, 1, 1), zoom=1.05)
    except Exception:
        pass
    save(fig, "mdl-la-svd-dimension-count")


def fig_min_norm():
    """One equation, two unknowns: the solution line x1 + x2 = 2, the
    minimum-norm solution (1, 1) as the foot of the perpendicular from the
    origin, and one longer solution for comparison."""
    lo, hi = -0.6, 2.8
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    axis_cross(ax, (lo, hi), (lo, hi), color=INK)
    ax.text(hi - 0.02, -0.08, r"$x_1$", color=INK, fontsize=14,
            ha="right", va="top")
    ax.text(-0.08, hi - 0.02, r"$x_2$", color=INK, fontsize=14,
            ha="right", va="top")

    # the solution line, clipped to the window
    t = np.array([lo, 2 - lo])
    ax.plot(t, 2 - t, color=BLUE, lw=2.2)
    # its label: rotated along the line, offset outward (away from the origin)
    p = np.array([0.30, 1.70]) + 0.22 * np.array([1, 1]) / np.sqrt(2)
    ax.text(p[0], p[1], r"$x_1+x_2=2$", color=BLUE,
            fontsize=14, ha="center", va="bottom", rotation=-45,
            rotation_mode="anchor")

    # another solution, farther from the origin
    other = np.array([2.3, -0.3])
    ax.plot([0, other[0]], [0, other[1]], ls="--", color=GRAY, lw=1.5)
    ax.plot(*other, "o", color=GRAY, ms=7, zorder=5)
    ax.text(1.2, -0.30,
            rf"$\|\mathbf{{x}}\|={np.linalg.norm(other):.2f}$",
            color=GRAY, fontsize=13, ha="center", va="top")

    # the minimum-norm solution: the foot of the perpendicular
    foot = np.array([1.0, 1.0])
    ax.plot([0, foot[0]], [0, foot[1]], ls="--", color=ORANGE, lw=1.8)
    right_angle(ax, foot, (-1, -1), (1, -1), size=0.14, color=INK)
    ax.plot(*foot, "o", color=ORANGE, ms=8, zorder=6)
    ax.text(1.22, 1.02, r"$\mathbf{A}^{+}\mathbf{b}=(1,1)$" + "\n"
            + r"$\|\mathbf{x}\|=\sqrt{2}$", color=ORANGE, fontsize=13,
            ha="left", va="bottom")
    ax.plot(0, 0, "o", color=INK, ms=4.5, zorder=7)

    clean_axes(ax, lim=((lo, hi), (lo, hi)), hide=True)
    save(fig, "mdl-la-min-norm")


FIGURES = [fig_svd_dimension_count, fig_min_norm]


def main():
    for fn in FIGURES:
        fn()
    for p in fl.WRITTEN:
        print(f"  {os.path.basename(p):34s} {os.path.getsize(p):>8,d} bytes")


if __name__ == "__main__":
    main()
