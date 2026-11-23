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

Writes ``img/mdl-la-map-gallery.svg``, ``img/mdl-la-composition.svg``,
``img/mdl-la-parallelogram-area.svg``, ``img/mdl-la-rank-nullity.svg`` and
``img/mdl-la-triangular-det.svg``.
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

from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

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
         r"(a) shear, then rotate: $\mathbf{R}_{90}\mathbf{T}$",
         [((0.0, 1.0), (0.36, 0.16)), ((-1.0, 1.0), (-0.12, 0.30))]),
        (axes[1], R, S @ R,
         r"(b) rotate, then shear: $\mathbf{T}\mathbf{R}_{90}$",
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


def fig_rank_nullity():
    """Rank--nullity as basis bookkeeping: a basis of R^n split into k kernel
    vectors w_1..w_k, which A sends to 0, and n-k completing vectors
    u_1..u_{n-k}, whose images A u_i form a basis of col A inside R^m.
    Schematic in the style of ``mdl-la-svd-subspaces`` (two boxes)."""
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    bw, bh = 2.6, 4.0
    dom_x, cod_x = 0.0, 6.0
    split = 1.7                       # kernel (bottom) / complement (top)
    T = fl.T

    # domain box: complement on top (blue), kernel below (grey)
    ax.add_patch(Rectangle((dom_x, split), bw, bh - split,
                           facecolor=T.BLUE.tint, edgecolor="none"))
    ax.add_patch(Rectangle((dom_x, 0), bw, split,
                           facecolor=T.SLATE.tint, edgecolor="none"))
    ax.add_patch(Rectangle((dom_x, 0), bw, bh, fill=False, edgecolor=INK,
                           lw=1.4))
    ax.plot([dom_x, dom_x + bw], [split, split], color=INK, lw=0.9, ls="--")
    ax.text(dom_x + bw / 2, 3.25,
            r"$\mathbf{u}_1,\ldots,\mathbf{u}_{n-k}$", ha="center",
            va="center", fontsize=15, color=T.BLUE.dark)
    ax.text(dom_x + bw / 2, 2.45, "extend to a basis\n" + r"of $\mathbb{R}^n$",
            ha="center", va="center", fontsize=13, color=T.BLUE.dark)
    ax.text(dom_x + bw / 2, 1.15, r"$\mathbf{w}_1,\ldots,\mathbf{w}_k$",
            ha="center", va="center", fontsize=15, color=INK)
    ax.text(dom_x + bw / 2, 0.5, r"basis of $\ker\mathbf{A}$",
            ha="center", va="center", fontsize=13, color=INK)

    # codomain box with the column space as an inner region
    cs_y0 = 1.0
    ax.add_patch(Rectangle((cod_x, 0), bw, bh, fill=False, edgecolor=INK,
                           lw=1.4))
    ax.add_patch(Rectangle((cod_x + 0.2, cs_y0), bw - 0.4, bh - cs_y0 - 0.2,
                           facecolor=T.GREEN.tint, edgecolor=GREEN, lw=1.2))
    ax.text(cod_x + bw / 2, 3.25,
            r"$\mathbf{A}\mathbf{u}_1,\ldots,\mathbf{A}\mathbf{u}_{n-k}$",
            ha="center", va="center", fontsize=15, color=T.GREEN.dark)
    ax.text(cod_x + bw / 2, 2.45,
            r"basis of $\mathrm{col}\,\mathbf{A}$" + "\n" + r"$\dim n-k$",
            ha="center", va="center", fontsize=13, color=T.GREEN.dark)
    p_zero = np.array([cod_x + bw / 2, 1.35])
    ax.plot(*p_zero, "o", color=INK, ms=6, zorder=5)
    ax.text(p_zero[0] + 0.18, p_zero[1], r"$\mathbf{0}$", ha="left",
            va="center", fontsize=15)

    ax.text(dom_x + bw / 2, bh + 0.3, r"$\mathbb{R}^n$ (input)",
            ha="center", fontsize=14)
    ax.text(cod_x + bw / 2, bh + 0.3, r"$\mathbb{R}^m$ (output)",
            ha="center", fontsize=14)

    # u_i -> A u_i (one-to-one), w_j -> 0
    ax.add_patch(FancyArrowPatch((dom_x + bw, 3.25), (cod_x + 0.2, 3.25),
                                 arrowstyle="->", mutation_scale=16,
                                 color=BLUE, lw=2.0))
    ax.text((dom_x + bw + cod_x) / 2 + 0.1, 3.42,
            r"$\mathbf{u}_i\mapsto\mathbf{A}\mathbf{u}_i$",
            ha="center", va="bottom", fontsize=14, color=T.BLUE.dark)
    ax.add_patch(FancyArrowPatch((dom_x + bw, 1.15), p_zero,
                                 arrowstyle="->", mutation_scale=14,
                                 color=GRAY, lw=1.6, linestyle=(0, (4, 3)),
                                 shrinkA=0, shrinkB=6))
    ax.text((dom_x + bw + cod_x) / 2 + 0.1, 1.0,
            r"$\mathbf{w}_j\mapsto\mathbf{0}$", ha="center", va="top",
            fontsize=14, color=INK)

    ax.set_xlim(-0.3, cod_x + bw + 0.3)
    ax.set_ylim(-0.2, bh + 0.75)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "mdl-la-rank-nullity")


def fig_triangular_det():
    """Which entries a nonzero term of the permutation formula can use.
    (a) Upper triangular 5x5: zeros below the diagonal force the identity
    permutation, so only the diagonal (shaded) enters det U.  (b) Block upper
    triangular with square blocks D1 (3x3) and D2 (2x2): the zero block forces
    every surviving permutation to stay inside the diagonal blocks, so the
    off-diagonal block never enters.  Both grids are 5x5 at one scale."""
    n = 5
    T = fl.T
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(8.4, 4.6))
    fig.subplots_adjust(wspace=0.12)

    def cell(ax, i, j, face, text="", tcol=INK, fs=14):
        ax.add_patch(Rectangle((j, n - 1 - i), 1, 1, facecolor=face,
                               edgecolor=LIGHT, lw=1.0))
        if text:
            ax.text(j + 0.5, n - 0.5 - i, text, ha="center", va="center",
                    fontsize=fs, color=tcol)

    # (a) upper triangular
    for i in range(n):
        for j in range(n):
            if i == j:
                cell(axa, i, j, T.ORANGE.tint, rf"$u_{{{i+1}{i+1}}}$",
                     T.ORANGE.dark)
            elif i < j:
                cell(axa, i, j, T.SLATE.tint, r"$*$", GRAY)
            else:
                cell(axa, i, j, "white", r"$0$", GRAY)
    axa.add_patch(Rectangle((0, 0), n, n, fill=False, edgecolor=INK, lw=1.4))
    axa.set_title(r"(a) $\det\mathbf{U}=u_{11}u_{22}\cdots u_{55}$",
                  fontsize=15, pad=8)

    # (b) block upper triangular, D1 3x3 and D2 2x2
    b = 3
    for i in range(n):
        for j in range(n):
            if (i < b) == (j < b):
                cell(axb, i, j, T.ORANGE.tint)
            elif i < b:
                cell(axb, i, j, T.SLATE.tint)
            else:
                cell(axb, i, j, "white")
    for (x0, y0, w, h) in [(0, n - b, b, b), (b, 0, n - b, n - b)]:
        axb.add_patch(Rectangle((x0, y0), w, h, fill=False,
                                edgecolor=ORANGE, lw=2.0, zorder=3))
    axb.text(b / 2, n - b / 2, r"$\mathbf{D}_1$", ha="center", va="center",
             fontsize=20, color=T.ORANGE.dark)
    axb.text(b + (n - b) / 2, (n - b) / 2, r"$\mathbf{D}_2$", ha="center",
             va="center", fontsize=20, color=T.ORANGE.dark)
    axb.text(b + (n - b) / 2, n - b / 2, r"$*$", ha="center", va="center",
             fontsize=20, color=GRAY)
    axb.text(b / 2, (n - b) / 2, r"$\mathbf{0}$", ha="center", va="center",
             fontsize=20, color=GRAY)
    axb.add_patch(Rectangle((0, 0), n, n, fill=False, edgecolor=INK, lw=1.4,
                            zorder=4))
    axb.set_title(r"(b) $\det=\det\mathbf{D}_1\,\det\mathbf{D}_2$",
                  fontsize=15, pad=8)

    for ax in (axa, axb):
        clean_axes(ax, lim=((-0.05, n + 0.05), (-0.05, n + 0.05)), hide=True)
    save(fig, "mdl-la-triangular-det")


FIGURES = [fig_map_gallery, fig_composition, fig_parallelogram_area,
           fig_rank_nullity, fig_triangular_det]


def main():
    for fn in FIGURES:
        fn()
    for p in fl.WRITTEN:
        print(f"  {os.path.basename(p):34s} {os.path.getsize(p):>8,d} bytes")


if __name__ == "__main__":
    main()
