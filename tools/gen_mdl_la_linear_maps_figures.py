#!/usr/bin/env python3
"""Generate the new illustrative figures of "Matrices as Linear Maps"
(``chapter_mdl-linear-algebra/mdl-linear-maps.md``) in the chapter's shared
house style.

The shared style lives in ``gen_mdl_figures.py``; importing it applies the
``plt.rcParams`` (fixed ``svg.hashsalt`` and ``metadata={'Date': None}`` in
``save()`` make re-runs byte-for-byte identical) and exposes the palette and
the drawing helpers.  The section's three older figures (``mdl-la-linear-map``,
``mdl-la-determinant``, ``mdl-la-null-collapse``) stay in that generator.
Every image below is computed by applying the matrix to the unit square.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_la_linear_maps_figures.py

Writes ``img/mdl-la-map-gallery.svg``, ``img/mdl-la-composition.svg`` and
``img/mdl-la-parallelogram-area.svg``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, LIGHT, INK = (fl.BLUE, fl.ORANGE, fl.GREEN,
                                         fl.GRAY, fl.LIGHT, fl.INK)
arrow, vlabel, clean_axes, axis_cross, right_angle = (
    fl.arrow, fl.vlabel, fl.clean_axes, fl.axis_cross, fl.right_angle)
save = fl.save

from matplotlib.patches import Polygon

def mat2(a, b, c, d):
    """Mathtext for the 2x2 matrix [[a, b], [c, d]] (mathtext has no
    matrix environment, so each column is a rule-less ``genfrac``)."""
    col = r"\genfrac{}{}{0}{0}{%s}{%s}"
    return r"$\left[" + col % (a, c) + r"\ " + col % (b, d) + r"\right]$"


UNIT = np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]])  # corners
LBL = 15


def _square_image(ax, M, lim, *, fill=GREEN, collapse=False):
    """Dashed unit square, its (shaded) image under M, and the two columns."""
    axis_cross(ax, lim[0], lim[1], color=INK)
    ax.add_patch(Polygon(UNIT.T, closed=True, fill=False, edgecolor=INK,
                         lw=1.0, ls="--", zorder=2))
    img = M @ UNIT
    if collapse:
        # image is a segment: grey halo under the column arrows
        pts = img.T[np.argsort(img[0] + 1e-3 * img[1])]
        ax.plot([pts[0, 0], pts[-1, 0]], [pts[0, 1], pts[-1, 1]], color=GRAY,
                lw=8, alpha=0.40, solid_capstyle="round", zorder=1)
    else:
        ax.add_patch(Polygon(img.T, closed=True, facecolor=fill, alpha=0.25,
                             edgecolor=fill, lw=1.6, zorder=1))
    c1, c2 = M[:, 0], M[:, 1]
    if np.allclose(c1, c2):
        # equal columns: a wide blue arrow under a narrow orange one
        arrow(ax, (0, 0), c1, color=BLUE, lw=5.0, mut=20)
        arrow(ax, (0, 0), c2, color=ORANGE, lw=2.0)
    else:
        cols = [(c1, BLUE), (c2, ORANGE)]
        cols.sort(key=lambda t: -np.linalg.norm(t[0]))  # shorter on top
        for c, cc in cols:
            if np.linalg.norm(c) > 0:
                arrow(ax, (0, 0), c, color=cc, lw=2.4)
    ax.plot(0, 0, "o", color=INK, ms=4.5, zorder=7)
    clean_axes(ax, lim=lim, hide=True)


def fig_map_gallery():
    """Six things a 2x2 matrix can do to the unit square, one panel each,
    common window [-1.3, 2.3]^2.  Per-panel titles carry the matrix and a
    two-word name (functional: the caption does not list the matrices)."""
    lim = ((-1.2, 2.25), (-1.2, 2.25))
    panels = [
        (np.array([[0.0, -1.0], [1.0, 0.0]]), "rotation by 90°",
         mat2(0, -1, 1, 0),
         False),
        (np.array([[1.0, 0.0], [0.0, -1.0]]), "reflection",
         mat2(1, 0, 0, -1),
         False),
        (np.array([[2.0, 0.0], [0.0, 0.5]]), "scaling",
         mat2(2, 0, 0, r"\frac{1}{2}"),
         False),
        (np.array([[1.0, 1.0], [0.0, 1.0]]), "shear",
         mat2(1, 1, 0, 1),
         False),
        (np.array([[1.0, 0.0], [0.0, 0.0]]), "projection",
         mat2(1, 0, 0, 0),
         True),
        (np.array([[1.0, 1.0], [1.0, 1.0]]), "rank-one collapse",
         mat2(1, 1, 1, 1),
         True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 7.0))
    fig.subplots_adjust(wspace=0.08, hspace=0.30)
    for ax, (M, name, mat, collapse), tag in zip(axes.flat, panels, "abcdef"):
        _square_image(ax, M, lim, collapse=collapse)
        ax.set_title(f"({tag}) {name}  " + mat, fontsize=LBL, pad=4)
    save(fig, "mdl-la-map-gallery")


def fig_composition():
    """The order of composition matters.  (a) shear S, then rotation R90:
    R90 S sends the unit square to (0,0), (0,1), (-1,2), (-1,1); (b) rotation,
    then shear: S R90 sends it to (0,0), (1,1), (0,1), (-1,0).  The image
    after the first map is a dotted grey outline; the final image is shaded.
    Panel titles are functional (they name the order)."""
    R = np.array([[0.0, -1.0], [1.0, 0.0]])
    S = np.array([[1.0, 1.0], [0.0, 1.0]])
    lim = ((-1.5, 2.4), (-0.7, 2.4))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2))
    fig.subplots_adjust(wspace=0.10)
    cases = [
        (axes[0], S, R @ S,
         r"(a) shear, then rotate: $\mathbf{R}_{90}\mathbf{S}$",
         [((0.0, 1.0), (0.36, 0.16)), ((-1.0, 1.0), (-0.12, 0.30))]),
        (axes[1], R, S @ R,
         r"(b) rotate, then shear: $\mathbf{S}\mathbf{R}_{90}$",
         [((1.0, 1.0), (0.42, 0.10)), ((-1.0, 0.0), (0.02, -0.32))]),
    ]
    for ax, first, M, title, labs in cases:
        mid = first @ UNIT
        ax.add_patch(Polygon(mid.T, closed=True, fill=False, edgecolor=GRAY,
                             lw=1.4, ls=(0, (1.5, 2.0)), zorder=2))
        _square_image(ax, M, lim)
        for (p, off), c in zip(labs, (BLUE, ORANGE)):
            ax.text(p[0] + off[0], p[1] + off[1],
                    f"$({p[0]:g},{p[1]:g})$".replace("-", "{-}"),
                    color=c, fontsize=LBL - 1, ha="center", va="center",
                    zorder=8)
        ax.set_title(title, fontsize=LBL, pad=4)
    save(fig, "mdl-la-composition")


def fig_parallelogram_area():
    """The image of the unit square under A = [[1, 2], [-1, 3]], the
    parallelogram (0,0), (1,-1), (3,2), (2,3), has area 5.  (a) bounding box
    [0,3] x [-1,3] (area 12) minus four right triangles (1/2, 3, 1/2, 3);
    (b) base times signed height: |a1| = sqrt 2 along a1 = (1,-1), unit normal
    n = (1,1)/sqrt 2, height a2 . n = 5/sqrt 2, foot (-1/2, 1/2).  Titles are
    the functional computations."""
    a1, a2 = np.array([1.0, -1.0]), np.array([2.0, 3.0])
    P = np.array([[0, 0], a1, a1 + a2, a2])
    lim = ((-1.25, 3.55), (-1.75, 3.55))
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(8.4, 4.9))
    fig.subplots_adjust(wspace=0.08)

    # --- (a) counting: box minus triangles ---
    axis_cross(axa, lim[0], lim[1], color=INK)
    box = np.array([[0, -1], [3, -1], [3, 3], [0, 3]])
    axa.add_patch(Polygon(box, closed=True, fill=False, edgecolor=INK,
                          lw=1.2, ls="--", zorder=2))
    tris = [
        ([[0, 0], [0, -1], [1, -1]], "1/2", (0.30, -0.66)),
        ([[1, -1], [3, -1], [3, 2]], "3", (2.35, -0.30)),
        ([[3, 2], [3, 3], [2, 3]], "1/2", (2.70, 2.66)),
        ([[2, 3], [0, 3], [0, 0]], "3", (0.65, 2.30)),
    ]
    for tri, lab, pos in tris:
        axa.add_patch(Polygon(tri, closed=True, facecolor=fl.T.ORANGE.tint,
                              edgecolor="none", zorder=0))
        axa.text(*pos, f"${lab}$", color=fl.T.ORANGE.dark, fontsize=LBL,
                 ha="center", va="center", zorder=5)
    axa.add_patch(Polygon(P, closed=True, facecolor=GREEN, alpha=0.25,
                          edgecolor=GREEN, lw=1.8, zorder=1))
    axa.text(1.5, 1.0, "$5$", color=fl.T.GREEN.dark, fontsize=LBL + 2,
             ha="center", va="center", zorder=5)
    axa.plot(0, 0, "o", color=INK, ms=4.5, zorder=7)
    axa.set_title(r"(a) $12-(\frac{1}{2}+3+\frac{1}{2}+3)=5$",
                  fontsize=LBL, pad=4)
    clean_axes(axa, lim=lim, hide=True)

    # --- (b) base times height ---
    axis_cross(axb, lim[0], lim[1], color=INK)
    axb.add_patch(Polygon(P, closed=True, facecolor=GREEN, alpha=0.18,
                          edgecolor="none", zorder=0))
    axb.plot([-1.05, 1.6], [1.05, -1.6], ls=(0, (1.5, 2.0)), color=GRAY,
             lw=1.4, zorder=1)            # the line through a1
    n = np.array([1.0, 1.0]) / np.sqrt(2.0)
    foot = np.array([-0.5, 0.5])
    axb.plot([foot[0], a2[0]], [foot[1], a2[1]], ls=(0, (1.5, 2.0)),
             color=INK, lw=1.4, zorder=2)  # the height
    right_angle(axb, foot, a1 * -1.0, a2 - foot, size=0.22, color=INK, lw=1.0)
    arrow(axb, (0, 0), n, color=GRAY, lw=1.8, mut=12)
    arrow(axb, (0, 0), a1, color=BLUE, lw=2.6)
    arrow(axb, (0, 0), a2, color=ORANGE, lw=2.6)
    axb.plot(0, 0, "o", color=INK, ms=4.5, zorder=7)
    vlabel(axb, (0.95, -1.35), r"$\mathbf{a}_1$", color=BLUE, fontsize=LBL,
           ha="left")
    vlabel(axb, (2.15, 3.22), r"$\mathbf{a}_2$", color=ORANGE, fontsize=LBL,
           ha="left")
    vlabel(axb, (0.80, 0.72), r"$\mathbf{n}$", color=GRAY, fontsize=LBL)
    # along the height, on its outer (upper-left) side
    vlabel(axb, (1.00, 2.58), r"$h=5/\sqrt{2}$", color=INK, fontsize=LBL,
           rotation=45)
    # along the base, on its outer (lower-left) side
    vlabel(axb, (0.27, -0.73), r"$\sqrt{2}$", color=BLUE, fontsize=LBL - 1,
           rotation=-45)
    axb.set_title(r"(b) $\|\mathbf{a}_1\|\,h=\sqrt{2}\cdot 5/\sqrt{2}=5$",
                  fontsize=LBL, pad=4)
    clean_axes(axb, lim=lim, hide=True)
    save(fig, "mdl-la-parallelogram-area")


FIGURES = [fig_map_gallery, fig_composition, fig_parallelogram_area]


def main():
    for fn in FIGURES:
        fn()
    for p in fl.WRITTEN:
        print(f"  {os.path.basename(p):34s} {os.path.getsize(p):>8,d} bytes")


if __name__ == "__main__":
    main()
