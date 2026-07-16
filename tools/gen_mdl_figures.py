#!/usr/bin/env python3
"""Generate every illustrative figure for the "Mathematics for Deep Learning ->
Linear Algebra" chapter in one consistent, clean style as static SVGs.

These replace the previous mix of hand-drawn SVGs and inline-matplotlib figures
scattered across the notebooks.  The notebooks reference the generated files with
no drawing code (like the slide SVGs).

Run with the repo's pytorch venv (matplotlib + numpy + scipy are available):

    .venv-pytorch/bin/python tools/gen_mdl_figures.py

All figures are written to ``img/mdl-la-<id>.svg``.  Figures that teach by being
seen (vector addition, projections, eigendecompositions, SVD action, ...) use
real numerical computation (matrix application, ``eig``/``svd``/sampling) so the
pictures are exact, not sketches.  The script is idempotent: re-running
overwrites cleanly.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("svg")  # non-interactive, SVG backend -- set once, up front

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, FancyArrowPatch, Polygon, Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

# --------------------------------------------------------------------------- #
# ONE shared style, applied once.                                             #
# --------------------------------------------------------------------------- #

# A small, consistent palette reused by name everywhere.
BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"
GRAY = "#7f7f7f"
LIGHT = "#cfcfcf"  # for faint grids / construction lines

plt.rcParams.update(
    {
        "figure.dpi": 100,
        "savefig.dpi": 100,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.linewidth": 0.8,
        "axes.grid": False,           # light grid off by default
        "axes.spines.top": False,     # minimal spines
        "axes.spines.right": False,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "mathtext.fontset": "cm",
        "svg.fonttype": "path",       # embed glyphs as paths (portable rendering)
        # Fixed hash salt => deterministic clip-path / gradient ids, so re-runs
        # are byte-for-byte identical (clean git diffs).
        "svg.hashsalt": "mdl-la",
    }
)

IMG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "img")
WRITTEN: list[str] = []


def save(fig, name: str) -> None:
    """Write ``img/<name>.svg`` with tight bounding box and close the figure."""
    os.makedirs(IMG_DIR, exist_ok=True)
    path = os.path.join(IMG_DIR, f"{name}.svg")
    # metadata={'Date': None} drops the creation-timestamp comment so output is
    # reproducible across runs.
    fig.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    WRITTEN.append(path)


# --------------------------------------------------------------------------- #
# Small drawing helpers (shared style for arrows, right angles, schematics).  #
# --------------------------------------------------------------------------- #

def arrow(ax, tail, tip, color=BLUE, lw=2.0, ls="-", alpha=1.0, mut=14):
    """Draw a vector arrow from ``tail`` to ``tip``."""
    ax.annotate(
        "",
        xy=tip,
        xytext=tail,
        arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                        linestyle=ls, alpha=alpha,
                        shrinkA=0, shrinkB=0, mutation_scale=mut),
    )


def vlabel(ax, pos, text, color="black", **kw):
    ax.text(pos[0], pos[1], text, color=color,
            ha=kw.pop("ha", "center"), va=kw.pop("va", "center"), **kw)


def right_angle(ax, corner, d1, d2, size=0.18, color=GRAY, lw=1.0):
    """Draw a small square marking a right angle at ``corner`` between unit-ish
    directions ``d1`` and ``d2``."""
    corner = np.asarray(corner, float)
    d1 = np.asarray(d1, float)
    d2 = np.asarray(d2, float)
    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)
    p0 = corner
    p1 = corner + size * d1
    p2 = corner + size * (d1 + d2)
    p3 = corner + size * d2
    ax.plot([p0[0], p1[0], p2[0], p3[0], p0[0]],
            [p0[1], p1[1], p2[1], p3[1], p0[1]],
            color=color, lw=lw)


def clean_axes(ax, lim=None, hide=False, equal=True):
    """Apply the shared geometric-figure look: equal aspect, light/hidden axes."""
    if equal:
        ax.set_aspect("equal")
    if lim is not None:
        (x0, x1), (y0, y1) = lim
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
    if hide:
        ax.axis("off")
    else:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def axis_cross(ax, xr, yr, color=GRAY, lw=0.9):
    """Draw faint x/y axes through the origin for a schematic plane."""
    ax.annotate("", xy=(xr[1], 0), xytext=(xr[0], 0),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw))
    ax.annotate("", xy=(0, yr[1]), xytext=(0, yr[0]),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw))


# =========================================================================== #
# GEOMETRY (Section 1.1)                                                      #
# =========================================================================== #

def fig_vectors():
    """(a) (3,2) as a point with dashed coordinate lines; (b) the same vector
    as a translation-invariant arrow drawn in three places.

    Both panels share one coordinate box, so they render at matching size; the
    caption ("... (left) ... (right)") already names each panel, so we draw no
    redundant per-panel titles.
    """
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(8.0, 3.6))

    LBL = 12                          # one size + font (mathtext cm) for every label
    LIM = ((-1.0, 4.5), (-1.3, 3.3))  # identical box for both panels -> equal size
    XR, YR = (-0.9, 4.3), (-1.2, 3.2)

    # (a) point with dashed coordinate drops; black axes for readability
    axis_cross(axa, XR, YR, color="black")
    p = np.array([3.0, 2.0])
    axa.plot(*p, "o", color=BLUE, ms=8, zorder=5)
    axa.plot([p[0], p[0]], [0, p[1]], "--", color=LIGHT, lw=1.2)
    axa.plot([0, p[0]], [p[1], p[1]], "--", color=LIGHT, lw=1.2)
    axa.text(p[0], -0.28, r"$3$", color=BLUE, ha="center", va="top", fontsize=LBL)
    axa.text(-0.22, p[1], r"$2$", color=BLUE, ha="right", va="center", fontsize=LBL)
    vlabel(axa, (p[0] + 0.15, p[1] + 0.22), r"$(3,2)$", color=BLUE, ha="left",
           fontsize=LBL)
    clean_axes(axa, lim=LIM, hide=True)

    # (b) same vector as a translation-invariant arrow in three places -- one
    # color, since every arrow *is* the same vector.  Bases chosen so all three
    # copies fit the shared box and stay clear of one another.
    axis_cross(axb, XR, YR, color="black")
    v = np.array([3.0, 2.0])
    # the third base is nudged right off the line through the other two, so the
    # arrows do not all line up
    bases = [np.array([0.0, 0.0]), np.array([1.0, -1.1]), np.array([-0.3, 1.1])]
    for b in bases:
        arrow(axb, b, b + v, color=BLUE, lw=2.2)
    # label just past one arrowhead, offset off the shaft so it never touches
    # the vector
    vlabel(axb, (3.18, 2.28), r"$(3,2)$", color=BLUE, ha="left", va="bottom",
           fontsize=LBL)
    clean_axes(axb, lim=LIM, hide=True)

    save(fig, "mdl-la-vectors")


def fig_vector_add():
    """Tip-to-tail addition u, then v from u's tip, resultant u+v from origin."""
    fig, ax = plt.subplots(figsize=(5.4, 3.9))   # landscape ~4:3
    u = np.array([3.0, 1.0])
    v = np.array([1.0, 2.0])
    LBL = 16                                  # larger labels, readable at figure size
    LIM = ((-0.5, 4.7), (-0.45, 3.25))        # trimmed box: axes don't overshoot content
    axis_cross(ax, (-0.4, 4.6), (-0.35, 3.15), color="black")
    arrow(ax, (0, 0), u, color=BLUE, lw=2.2)
    arrow(ax, u, u + v, color=ORANGE, lw=2.2)
    arrow(ax, (0, 0), u + v, color=GREEN, lw=2.4)
    vlabel(ax, (1.5, 0.22), r"$\mathbf{u}$", color=BLUE, fontsize=LBL)
    # v runs (3,1)->(4,3); label offset to its right so it clears the shaft
    vlabel(ax, (4.05, 1.85), r"$\mathbf{v}$", color=ORANGE, ha="left", fontsize=LBL)
    vlabel(ax, (1.55, 1.82), r"$\mathbf{u}+\mathbf{v}$", color=GREEN, fontsize=LBL)
    clean_axes(ax, lim=LIM, hide=True)
    save(fig, "mdl-la-vector-add")


def fig_span():
    """(a) the span of one nonzero vector is the line through the origin in
    its direction; (b) two independent vectors u, w span the whole plane: a
    faint lattice of integer combinations a*u + b*w suggests the coverage, and
    a dashed parallelogram construction resolves x = 2u + w.

    Both panels share one coordinate box so they render at the same size; the
    caption names them, so we draw no per-panel titles.
    """
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(8.8, 3.9))

    LBL = 16                             # vector letters
    FML = 14                             # longer labels (span(v), the equation)
    LIM = ((-3.4, 6.2), (-2.6, 4.6))     # identical box for both panels -> equal size
    XR, YR = (-3.2, 6.0), (-2.4, 4.4)

    # rotate the u/v direction (and the mesh) clockwise so the lattice sits
    # more comfortably in the landscape box; the SAME rotation is applied to v
    # in (a) and to u, w in (b) so the shared u/v vector stays consistent.
    ang = np.radians(-10.0)
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])

    # (a) span of a single vector: the line {t v}
    v = R @ np.array([2.0, 1.0])
    t0, t1 = -1.6, 2.9
    axis_cross(axa, XR, YR, color="black")
    axa.plot([t0 * v[0], t1 * v[0]], [t0 * v[1], t1 * v[1]], "--",
             color=GRAY, lw=1.4)
    arrow(axa, (0, 0), v, color=BLUE, lw=2.4)
    vlabel(axa, (v[0] + 0.10, v[1] + 0.42), r"$\mathbf{v}$", color=BLUE,
           ha="left", fontsize=LBL)
    # label along the line, rotated to its direction, nudged perpendicular;
    # placed well past v so the two do not collide
    d = v / np.linalg.norm(v)
    perp = np.array([-d[1], d[0]])
    lpos = 2.3 * v + 0.5 * perp
    axa.text(lpos[0], lpos[1], r"$\mathrm{span}(\mathbf{v})$", color=GRAY,
             fontsize=FML, ha="center", va="center",
             rotation=np.degrees(np.arctan2(d[1], d[0])),
             rotation_mode="anchor")
    clean_axes(axa, lim=LIM, hide=True)

    # (b) two independent vectors span the plane
    u = R @ np.array([2.0, 1.0])
    w = R @ np.array([0.5, 1.5])
    coeffs = range(-1, 4)  # integer combinations a*u + b*w, a, b in -1..3
    for a in coeffs:  # lines of constant a (along w)
        p0, p1 = a * u + coeffs[0] * w, a * u + coeffs[-1] * w
        axb.plot([p0[0], p1[0]], [p0[1], p1[1]], color=LIGHT, lw=0.9)
    for b in coeffs:  # lines of constant b (along u)
        p0, p1 = coeffs[0] * u + b * w, coeffs[-1] * u + b * w
        axb.plot([p0[0], p1[0]], [p0[1], p1[1]], color=LIGHT, lw=0.9)
    x = 2 * u + w  # (4.5, 3.5)
    axis_cross(axb, XR, YR, color="black")
    # dashed parallelogram construction: 2u -> x and w -> x
    axb.plot([2 * u[0], x[0]], [2 * u[1], x[1]], "--", color=GRAY, lw=1.4)
    axb.plot([w[0], x[0]], [w[1], x[1]], "--", color=GRAY, lw=1.4)
    arrow(axb, (0, 0), u, color=BLUE, lw=2.4)
    arrow(axb, (0, 0), w, color=ORANGE, lw=2.4)
    arrow(axb, (0, 0), x, color=GREEN, lw=2.6)
    vlabel(axb, (u[0] + 0.05, u[1] - 0.40), r"$\mathbf{u}$", color=BLUE,
           ha="center", fontsize=LBL)
    vlabel(axb, (w[0] - 0.30, w[1] + 0.30), r"$\mathbf{w}$", color=ORANGE,
           ha="right", fontsize=LBL)
    # equation on a white patch, in the open area above x, so the lattice
    # behind it never crosses the text
    axb.text(x[0] - 0.9, x[1] + 1.15, r"$\mathbf{x}=2\mathbf{u}+\mathbf{w}$",
             color=GREEN, ha="center", va="center", fontsize=FML,
             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.92))
    clean_axes(axb, lim=LIM, hide=True)

    save(fig, "mdl-la-span")


def fig_angle():
    """Two vectors from origin with angle theta between them.  v lies on the
    x-axis, matching the WLOG choice v = (r, 0) used in the text."""
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    v = np.array([3.0, 0.0])       # v = (r, 0): aligned with the x-axis
    w = np.array([1.2, 2.6])
    LBL = 17                       # larger vector/angle symbols
    LIM = ((-0.5, 3.9), (-0.6, 3.0))
    axis_cross(ax, (-0.4, 3.8), (-0.5, 2.9), color="black")
    arrow(ax, (0, 0), v, color=BLUE, lw=2.4)
    arrow(ax, (0, 0), w, color=ORANGE, lw=2.4)
    a1 = np.degrees(np.arctan2(v[1], v[0]))
    a2 = np.degrees(np.arctan2(w[1], w[0]))
    ax.add_patch(Arc((0, 0), 1.7, 1.7, angle=0, theta1=a1, theta2=a2,
                     color=GRAY, lw=1.6))
    mid = np.radians((a1 + a2) / 2)
    vlabel(ax, (1.18 * np.cos(mid), 1.18 * np.sin(mid)), r"$\theta$", color=GRAY,
           fontsize=18)
    vlabel(ax, (v[0], -0.34), r"$\mathbf{v}$", color=BLUE, fontsize=LBL)
    vlabel(ax, (w[0] - 0.05, w[1] + 0.26), r"$\mathbf{w}$", color=ORANGE, fontsize=LBL)
    clean_axes(ax, lim=LIM, hide=True)
    save(fig, "mdl-la-angle")


def fig_projection():
    """(a) generic projection of v onto w with right-angle residual;
    (b) the Cauchy-Schwarz equality case (v collinear with w).

    Both panels share one coordinate box, so they render at the same size; the
    caption ("Left ... Right ...") names them, so we draw no per-panel titles.
    """
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(8.4, 3.7))

    LBL = 16                            # vector letters
    FML = 14                            # longer formula labels
    LIM = ((-0.4, 4.1), (-0.5, 2.7))    # identical box for both panels -> equal size
    XR, YR = (-0.3, 4.0), (-0.4, 2.6)

    # (a) generic
    w = np.array([3.2, 0.6])
    v = np.array([1.6, 2.2])
    wn = w / np.linalg.norm(w)
    proj_len = float(v @ wn)
    proj = proj_len * wn
    axis_cross(axa, XR, YR, color="black")
    arrow(axa, (0, 0), w, color=ORANGE, lw=2.2)
    arrow(axa, (0, 0), v, color=BLUE, lw=2.4)
    # projection as a thick segment along w
    axa.plot([0, proj[0]], [0, proj[1]], color=GREEN, lw=5, solid_capstyle="round",
             zorder=2)
    # residual r = v - proj (dashed) meeting w at a right angle
    axa.plot([proj[0], v[0]], [proj[1], v[1]], "--", color=GRAY, lw=1.6)
    right_angle(axa, proj, -wn, (v - proj), size=0.22, color=GRAY)
    vlabel(axa, (w[0] + 0.05, w[1] - 0.22), r"$\mathbf{w}$", color=ORANGE, ha="left",
           fontsize=LBL)
    vlabel(axa, (v[0] - 0.12, v[1] + 0.24), r"$\mathbf{v}$", color=BLUE, fontsize=LBL)
    vlabel(axa, (proj[0] * 0.55 + 0.05, proj[1] * 0.55 + 0.30),
           r"$\|\mathbf{v}\|\cos\theta$", color=GREEN, ha="center", fontsize=FML)
    vlabel(axa, ((proj[0] + v[0]) / 2 + 0.24, (proj[1] + v[1]) / 2),
           r"$\mathbf{r}$", color=GRAY, ha="left", fontsize=LBL)
    clean_axes(axa, lim=LIM, hide=True)

    # (b) equality: v collinear with w, residual vanishes
    w2 = np.array([3.4, 1.3])
    v2 = 0.62 * w2  # collinear
    axis_cross(axb, XR, YR, color="black")
    arrow(axb, (0, 0), w2, color=ORANGE, lw=2.2)
    axb.plot([0, v2[0]], [0, v2[1]], color=GREEN, lw=5, solid_capstyle="round",
             zorder=2)
    arrow(axb, (0, 0), v2, color=BLUE, lw=2.4)
    vlabel(axb, (w2[0] + 0.05, w2[1] + 0.02), r"$\mathbf{w}$", color=ORANGE,
           ha="left", fontsize=LBL)
    vlabel(axb, (v2[0] * 0.5 - 0.12, v2[1] * 0.5 + 0.34), r"$\mathbf{v}$",
           color=BLUE, fontsize=LBL)
    # equality relation, sat in the open upper-left area (lower than before)
    axb.text(1.6, 1.5,
             r"$|\mathbf{v}\cdot\mathbf{w}| = \|\mathbf{v}\|\,\|\mathbf{w}\|$",
             ha="center", va="center", fontsize=FML)
    clean_axes(axb, lim=LIM, hide=True)

    save(fig, "mdl-la-projection")


def fig_hyperplane():
    """The line w.x = b in 2-D: normal w, one level set, one shaded half-space,
    and the signed distance b/||w||.  Beside it a 3-D coordinate frame (three
    axes meeting at the origin) with the plane analog."""
    fig = plt.figure(figsize=(9.0, 3.7))
    axa = fig.add_subplot(1, 2, 1)
    axb = fig.add_subplot(1, 2, 2, projection="3d")

    LBL = 15                                   # vector letters
    FML = 14                                   # formula labels

    # --- 2-D panel ---
    w = np.array([1.0, 1.6])
    wn = w / np.linalg.norm(w)
    nrm = np.linalg.norm(w)
    b = 2.0
    L = 2.6
    perp = np.array([-wn[1], wn[0]])           # along the level set
    lim = ((-2.0, 3.4), (-1.2, 2.9))           # trimmed landscape box

    cp = (b / nrm) * wn                         # foot of perpendicular from origin
    seg = np.array([cp - L * perp, cp + L * perp])

    # shade the single half-space w.x > b (on the +w side of the line)
    far = wn * (2 * L)
    poly = np.array([seg[0], seg[1], seg[1] + far, seg[0] + far])
    axa.add_patch(Polygon(poly, closed=True, color=BLUE, alpha=0.10, lw=0))

    # the single level set w.x = b
    axa.plot(seg[:, 0], seg[:, 1], "--", color=BLUE, lw=1.8)
    text_dir = perp if perp[0] >= 0 else -perp  # keep the label upright
    line_rot = np.degrees(np.arctan2(text_dir[1], text_dir[0]))
    lab_pos = cp + 0.62 * L * text_dir + 0.34 * wn
    axa.text(lab_pos[0], lab_pos[1], r"$\mathbf{w}\!\cdot\!\mathbf{x}=b$",
             color=BLUE, fontsize=FML, ha="center", va="center",
             rotation=line_rot, rotation_mode="anchor")

    axis_cross(axa, (-1.9, 3.3), (-1.1, 2.8), color="black")
    arrow(axa, (0, 0), w, color=ORANGE, lw=2.4)
    vlabel(axa, (w[0] + 0.20, w[1] + 0.12), r"$\mathbf{w}$", color=ORANGE,
           ha="left", fontsize=LBL)

    # signed distance b/||w|| from origin to the line
    axa.plot([0, cp[0]], [0, cp[1]], color=GREEN, lw=3, solid_capstyle="round",
             zorder=4)
    dl = 0.5 * cp + np.array([-0.98, 0.30])
    axa.text(dl[0], dl[1], r"$b/\|\mathbf{w}\|$", color=GREEN, fontsize=FML,
             ha="center", va="center")
    axa.text(1.95, 2.25, r"$\mathbf{w}\cdot\mathbf{x}>b$", color=BLUE,
             fontsize=FML, ha="center")
    clean_axes(axa, lim=lim, hide=True)

    # --- 3-D panel: plane w.x = b inside the default 3-D box frame ---
    # (keep matplotlib's box + shaded panes; only blacken the axis lines)
    # a nearly-horizontal plane reads clearly from a slightly raised view
    w3 = np.array([0.45, 0.5, 1.0])
    b3 = 1.3
    gx, gy = np.meshgrid(np.linspace(0.0, 1.9, 10), np.linspace(0.0, 1.9, 10))
    gz = (b3 - w3[0] * gx - w3[1] * gy) / w3[2]
    axb.plot_surface(gx, gy, gz, color=BLUE, alpha=0.40, linewidth=0,
                     antialiased=True)
    # normal from the origin to the plane (mirrors the 2-D panel)
    fw = (b3 / (w3 @ w3)) * w3
    axb.quiver(0, 0, 0, fw[0], fw[1], fw[2], color=ORANGE, lw=2.4,
               arrow_length_ratio=0.14)
    # label the normal in the open wedge below the plane, clear of the surface
    lwp = 0.5 * fw + np.array([0.34, 0.04, -0.10])
    axb.text(lwp[0], lwp[1], lwp[2], r"$\mathbf{w}$", color=ORANGE, fontsize=LBL)
    axb.set_xlim(0, 1.9); axb.set_ylim(0, 1.9); axb.set_zlim(-0.15, 1.9)
    axb.set_xticks([]); axb.set_yticks([]); axb.set_zticks([])
    for axis in (axb.xaxis, axb.yaxis, axb.zaxis):
        axis.line.set_color("black")          # blacken the axis lines only
    axb.view_init(elev=26, azim=-54)
    try:
        axb.set_box_aspect((1, 1, 1), zoom=1.1)
    except Exception:
        pass

    save(fig, "mdl-la-hyperplane")


def fig_linear_map():
    """A unit-square grid and its image under the chapter's running matrix
    A = [[1, 2], [-1, 3]].  Both panels use the SAME scale and the SAME window
    size (so they render at the same height, and the unit square and its
    area-5 image are directly comparable); the shaded unit square maps to the
    shaded parallelogram (0,0), (1,-1), (3,2), (2,3)."""
    A = np.array([[1.0, 2.0], [-1.0, 3.0]])
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(8.6, 4.4))

    LBL = 16
    n = 3
    ts = np.linspace(0, n, 200)
    grid = list(range(n + 1))
    GRIDC = "black"

    def draw_grid(ax, M, lw=0.8):
        for k in grid:  # images of the lines x=k
            pts = M @ np.vstack([np.full_like(ts, k), ts])
            ax.plot(pts[0], pts[1], color=GRIDC, lw=lw)
        for k in grid:  # images of the lines y=k
            pts = M @ np.vstack([ts, np.full_like(ts, k)])
            ax.plot(pts[0], pts[1], color=GRIDC, lw=lw)

    unit = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

    # (a) frames the unit grid; (b) is zoomed out (NOT to the same scale) to a
    # square window that contains the whole image grid -- its corners map to
    # (0,0),(3,-3),(6,9),(9,6), so the content spans x in [0,9], y in [-3,9].
    # Both windows are square, so the panels still render at the same height.
    lima = ((-1.0, 3.6), (-0.9, 3.7))
    limb = ((-1.8, 10.8), (-3.3, 9.3))

    # (a) original grid
    draw_grid(axa, np.eye(2))
    axa.add_patch(Polygon(unit, closed=True, facecolor=GREEN, alpha=0.30, lw=0))
    arrow(axa, (0, 0), (1, 0), color=BLUE, lw=2.6)
    arrow(axa, (0, 0), (0, 1), color=ORANGE, lw=2.6)
    vlabel(axa, (1.05, -0.42), r"$\mathbf{e}_1$", color=BLUE, fontsize=LBL)
    vlabel(axa, (-0.48, 1.05), r"$\mathbf{e}_2$", color=ORANGE, fontsize=LBL)
    clean_axes(axa, lim=lima, hide=True)

    # (b) image under A -- same scale as (a)
    draw_grid(axb, A)
    axb.add_patch(Polygon((A @ unit.T).T, closed=True, facecolor=GREEN,
                          alpha=0.30, lw=0))
    a1 = A @ np.array([1, 0])  # (1, -1)
    a2 = A @ np.array([0, 1])  # (2, 3)
    arrow(axb, (0, 0), a1, color=BLUE, lw=2.6)
    arrow(axb, (0, 0), a2, color=ORANGE, lw=2.6)
    # a1 is the image of e1 -- put its label OUTSIDE the grid, below the
    # bottom edge (the image of the x-axis, the line y = -x here)
    vlabel(axb, (a1[0] + 0.7, a1[1] - 1.15), r"$\mathbf{A}\mathbf{e}_1$",
           color=BLUE, ha="center", fontsize=LBL)
    vlabel(axb, (a2[0] - 0.35, a2[1] + 0.35), r"$\mathbf{A}\mathbf{e}_2$",
           color=ORANGE, ha="right", fontsize=LBL)
    clean_axes(axb, lim=limb, hide=True)

    save(fig, "mdl-la-linear-map")


def fig_determinant():
    """Three panels: (a) unit square -> parallelogram, area = |det A|;
    (b) negative determinant (orientation flip); (c) degenerate (det 0).
    The caption names the three cases, so the panels carry no titles."""
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6))
    fig.subplots_adjust(wspace=0.05)
    unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]).T  # columns: corners

    def show_para(ax, A, degenerate=False):
        det = np.linalg.det(A)
        img = A @ unit
        # original unit square (dashed reference)
        ax.add_patch(Polygon(unit.T, closed=True, fill=False, edgecolor="black",
                             lw=1.0, ls="--"))
        col = GREEN if det > 0 else ORANGE
        if degenerate:
            # collapses to a segment
            ax.plot(img[0], img[1], color=ORANGE, lw=4, solid_capstyle="round")
        else:
            ax.add_patch(Polygon(img.T, closed=True, facecolor=col, alpha=0.28,
                                 edgecolor=col, lw=2))
        c1 = A @ np.array([1, 0])
        c2 = A @ np.array([0, 1])
        arrow(ax, (0, 0), c1, color=BLUE, lw=2.2)
        arrow(ax, (0, 0), c2, color=ORANGE, lw=2.2)
        if not degenerate:
            # a curved arrow column-1 -> column-2 shows the induced orientation:
            # counter-clockwise when det > 0, clockwise (the flip) when det < 0
            rad = 0.4 if det > 0 else -0.4
            ax.add_patch(FancyArrowPatch(0.46 * c1, 0.46 * c2,
                         connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                         mutation_scale=13, color=col, lw=1.6, zorder=6))
        area = abs(det)
        if area < 1e-9:          # exact-zero determinant (degenerate case)
            area = 0.0
        ax.text(0.5, -0.02,
                rf"area $=|\det\mathbf{{A}}|={area:.2g}$",
                transform=ax.transAxes, ha="center", va="top", fontsize=14,
                color="black")
        clean_axes(ax, lim=((-0.42, 2.25), (-0.35, 2.15)), hide=True)

    show_para(axes[0], np.array([[1.6, 0.6], [0.3, 1.4]]))
    show_para(axes[1], np.array([[0.4, 1.5], [1.4, 0.5]]))
    show_para(axes[2], np.array([[0.9, 0.6], [1.2, 0.8]]), degenerate=True)
    save(fig, "mdl-la-determinant")


def fig_null_collapse():
    """(a) the input plane with a light grid, the null-space direction of
    B = [[2,-1],[4,-2]] dashed, and three marked points (one on the null line);
    (b) the image: the whole grid lands on the column-space line y = 2x, the
    marked points land at their exact images, and the null-line point lands at
    the origin.  All images computed by applying B."""
    B = np.array([[2.0, -1.0], [4.0, -2.0]])
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(8.8, 4.4))

    # sample inputs: x1, x2 generic; x3 on the null line span{(1, 2)}
    pts = [np.array([1.0, 0.5]), np.array([-0.5, 1.0]), np.array([0.5, 1.0])]
    marks = ["o", "s", "^"]
    cols = [BLUE, ORANGE, GREEN]
    labs_a = [r"$\mathbf{x}_1$", r"$\mathbf{x}_2$", r"$\mathbf{x}_3$"]
    labs_b = [r"$\mathbf{B}\mathbf{x}_1$", r"$\mathbf{B}\mathbf{x}_2$",
              r"$\mathbf{B}\mathbf{x}_3=\mathbf{0}$"]

    LBL = 15
    ANN = 14
    rot = np.degrees(np.arctan2(2.0, 1.0))  # slope of the null / column line

    # --- (a) input plane ---
    n = 2
    for k in range(-n, n + 1):
        axa.plot([k, k], [-n, n], color=LIGHT, lw=0.9)
        axa.plot([-n, n], [k, k], color=LIGHT, lw=0.9)
    axis_cross(axa, (-2.3, 2.3), (-2.3, 2.3), color="black")
    # null-space line span{(1, 2)} (dashed): inputs sent to the origin
    axa.plot([-1.15, 1.15], [-2.3, 2.3], "--", color="black", lw=1.6)
    axa.text(-1.28, -1.62, r"null space", color="black",
             fontsize=ANN, ha="center", va="center", rotation=rot,
             rotation_mode="anchor")
    offs_a = [(0.30, -0.24), (-0.36, 0.22), (0.42, 0.04)]
    for p, m, c, lab, off in zip(pts, marks, cols, labs_a, offs_a):
        axa.plot(*p, m, color=c, ms=9, zorder=5)
        axa.text(p[0] + off[0], p[1] + off[1], lab, color=c, fontsize=LBL,
                 ha="center", va="center")
    clean_axes(axa, lim=((-2.3, 2.3), (-2.3, 2.3)), hide=True)

    # --- (b) image plane: everything lands on the column space y = 2x ---
    axis_cross(axb, (-4.8, 4.8), (-4.8, 4.8), color="black")
    # column space = span of the columns = the line y = 2x
    axb.plot([-2.3, 2.3], [-4.6, 4.6], color=GRAY, lw=5, alpha=0.5,
             solid_capstyle="round", zorder=1)
    # keep the label off the thick line and well below the Bx3=0 marker:
    # sit it lower in the empty lower-right wedge
    axb.text(-0.35, -3.05, r"column space", color="black",
             fontsize=ANN, ha="center", va="center", rotation=rot,
             rotation_mode="anchor")
    # the two columns of B, both along the line
    b1 = B @ np.array([1.0, 0.0])   # (2, 4)
    b2 = B @ np.array([0.0, 1.0])   # (-1, -2)
    arrow(axb, (0, 0), b1, color=BLUE, lw=2.4)
    arrow(axb, (0, 0), b2, color=ORANGE, lw=2.4)
    vlabel(axb, (b1[0] + 0.75, b1[1] + 0.05), r"$\mathbf{b}_1$", color=BLUE,
           fontsize=LBL)
    # b2 and Bx2 labels share the same up-left offset from their markers
    vlabel(axb, (b2[0] - 0.85, b2[1] + 0.45), r"$\mathbf{b}_2$", color=ORANGE,
           fontsize=LBL)
    offs_b = [(1.15, -0.25), (-0.85, 0.45), (1.7, -0.55)]
    for p, m, c, lab, off in zip(pts, marks, cols, labs_b, offs_b):
        q = B @ p
        axb.plot(*q, m, color=c, ms=9, zorder=5)
        axb.text(q[0] + off[0], q[1] + off[1], lab, color=c, fontsize=LBL,
                 ha="center", va="center")
    clean_axes(axb, lim=((-4.8, 4.8), (-4.8, 4.8)), hide=True)

    save(fig, "mdl-la-null-collapse")


def fig_cosine_highd():
    """Histograms of cos(theta) between random unit-vector pairs for several
    dimensions; sharpens to a spike at 0.  Sample g ~ N(0, I), normalize."""
    rng = np.random.default_rng(0)
    dims = [2, 10, 1000]
    cols = [BLUE, ORANGE, GREEN]
    n_pairs = 20000

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    bins = np.linspace(-1, 1, 80)
    for d, c in zip(dims, cols):
        a = rng.standard_normal((n_pairs, d))
        b = rng.standard_normal((n_pairs, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        b /= np.linalg.norm(b, axis=1, keepdims=True)
        cos = np.sum(a * b, axis=1)
        ax.hist(cos, bins=bins, density=True, histtype="step", color=c, lw=2.0,
                label=rf"$d={d}$")
    ax.axvline(0, color=GRAY, lw=1.0, ls="--")
    # one font (CM math) and size for every text label, so density, cos(theta),
    # and the std annotation all match
    LAB = 13
    ax.set_xlabel(r"$\cos\theta$", fontsize=LAB)
    ax.set_ylabel(r"$\mathrm{density}$", fontsize=LAB)
    # annotation moved right into the open upper-right, clear of the central
    # spike and the dashed axis
    ax.annotate(r"$\mathrm{std}\approx 1/\sqrt{d}$", xy=(0.0, 0.0),
                xytext=(0.64, 0.74), textcoords="axes fraction",
                fontsize=LAB, color="black")
    ax.legend(loc="upper left", fontsize=11)
    ax.set_aspect("auto")
    clean_axes(ax, equal=False)
    save(fig, "mdl-la-cosine-highd")


# =========================================================================== #
# EIGENDECOMPOSITION (Section 1.2)                                            #
# =========================================================================== #

def fig_eig_ellipse():
    """Unit circle -> ellipse under (a) diag(2, -1) and (b) [[2,1],[1,2]].
    Eigenvectors drawn as the ellipse axes, scaled by eigenvalues, labelled."""
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2))
    t = np.linspace(0, 2 * np.pi, 400)
    circle = np.vstack([np.cos(t), np.sin(t)])
    LBL = 15

    # the caption names the two matrices, so the panels carry no titles
    specs = [np.array([[2.0, 0.0], [0.0, -1.0]]),
             np.array([[2.0, 1.0], [1.0, 2.0]])]
    for ax, A in zip(axes, specs):
        ellipse = A @ circle
        w, V = np.linalg.eigh(A)  # symmetric: real eigenpairs, V orthonormal cols
        ax.plot(circle[0], circle[1], "--", color="black", lw=1.0, alpha=0.5)
        ax.plot(ellipse[0], ellipse[1], color=BLUE, lw=2.4)
        order = np.argsort(-np.abs(w))
        labels = [r"$\lambda_1$", r"$\lambda_2$"]
        for rank, idx in enumerate(order):
            vec = V[:, idx]
            lam = w[idx]
            tip = lam * vec  # eigenvector scaled by eigenvalue (signed)
            arrow(ax, (0, 0), tip, color=GREEN if lam >= 0 else ORANGE, lw=2.6)
            lab = labels[rank]
            sign = "" if lam >= 0 else " (flip)"
            # offset the label beyond the tip plus a perpendicular nudge so it
            # never sits under the arrowhead for near-axis-aligned eigenvectors
            d = tip / np.linalg.norm(tip)
            perp = np.array([-d[1], d[0]])
            # the eigenvectors ARE the ellipse axes, so the outward normal at a
            # tip is radial.  For a near-horizontal/vertical axis a radial label
            # would sit under the arrowhead, so drop it above/beside where the
            # ellipse has already narrowed; diagonals get pushed straight out.
            if abs(d[1]) < 0.35:            # ~horizontal axis -> label above tip
                lpos = tip + np.array([0.18 * np.sign(d[0]), 1.02])
                ha = "center"
            elif abs(d[0]) < 0.35:          # ~vertical axis -> beyond the tip,
                # left of the y-axis so the long "(flip)" text neither clips nor
                # crosses the axis line
                lpos = np.array([-0.15, tip[1] + 0.55 * np.sign(tip[1])])
                ha = "right"
            else:                           # diagonal -> straight out along axis
                lpos = tip + 0.90 * d + 0.14 * perp
                ha = "center"
            ax.text(lpos[0], lpos[1], rf"{lab}$={lam:.2g}${sign}",
                    color=GREEN if lam >= 0 else ORANGE, fontsize=LBL,
                    ha=ha, va="center")
        m = 3.0
        clean_axes(ax, lim=((-m, m), (-m, m)), hide=True)
        axis_cross(ax, (-m, m), (-m, m), color="black")
    save(fig, "mdl-la-eig-ellipse")


def fig_defective_shear():
    """The defective shear [[1,1],[0,1]] acting on a grid + unit square.  Every
    horizontal layer slides right proportionally to its height; the x-axis is
    the single surviving eigendirection (geometric multiplicity 1)."""
    A = np.array([[1.0, 1.0], [0.0, 1.0]])
    # (b) is wider so it can show the ENTIRE sheared 3x3 grid (top row slides to
    # x=6); both panels keep the same scale + height (equal aspect, same y-span,
    # width_ratios matched to the x-spans below)
    fig = plt.figure(figsize=(11.6, 4.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.6, 7.4], wspace=0.10)
    axa = fig.add_subplot(gs[0])
    axb = fig.add_subplot(gs[1])

    LBL = 15
    n = 3
    ts = np.linspace(0, n, 200)
    grid = list(range(n + 1))
    GRIDC = "black"

    def draw_grid(ax, M):
        for k in grid:  # images of the lines x=k
            pts = M @ np.vstack([np.full_like(ts, k), ts])
            ax.plot(pts[0], pts[1], color=GRIDC, lw=0.8)
        for k in grid:  # images of the lines y=k
            pts = M @ np.vstack([ts, np.full_like(ts, k)])
            ax.plot(pts[0], pts[1], color=GRIDC, lw=0.8)

    unit = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    lima = ((-0.8, 3.8), (-0.9, 3.7))    # span 4.6 x 4.6
    limb = ((-0.7, 6.7), (-0.9, 3.7))    # span 7.4 x 4.6 -> same scale + height

    # (a) original grid
    draw_grid(axa, np.eye(2))
    axa.add_patch(Polygon(unit, closed=True, facecolor=GREEN, alpha=0.22, lw=0))
    # the eigendirection: the x-axis, drawn as a green ray
    axa.plot([-0.5, 3.7], [0, 0], color=GREEN, lw=3.2, zorder=2,
             solid_capstyle="round", alpha=0.85)
    arrow(axa, (0, 0), (1, 0), color=BLUE, lw=2.4)
    arrow(axa, (0, 0), (0, 1), color=ORANGE, lw=2.4)
    vlabel(axa, (1.05, -0.42), r"$\mathbf{e}_1$", color=BLUE, fontsize=LBL)
    vlabel(axa, (-0.42, 1.05), r"$\mathbf{e}_2$", color=ORANGE, fontsize=LBL)
    clean_axes(axa, lim=lima, hide=True)

    # (b) image under the shear -- same scale as (a), whole grid visible
    draw_grid(axb, A)
    axb.add_patch(Polygon((A @ unit.T).T, closed=True, facecolor=GREEN,
                          alpha=0.22, lw=0))
    axb.plot([-0.5, 6.6], [0, 0], color=GREEN, lw=3.2, zorder=2,
             solid_capstyle="round", alpha=0.85)
    a1 = A @ np.array([1.0, 0.0])   # (1, 0): unchanged
    a2 = A @ np.array([0.0, 1.0])   # (1, 1): picked up a horizontal component
    arrow(axb, (0, 0), a1, color=BLUE, lw=2.4)
    arrow(axb, (0, 0), a2, color=ORANGE, lw=2.6)
    vlabel(axb, (1.10, -0.42), r"$\mathbf{A}\mathbf{e}_1=\mathbf{e}_1$",
           color=BLUE, ha="left", fontsize=LBL)
    # Ae2 tip is (1,1), a mesh vertex; label up-left of it, outside the mesh
    # (the mesh lies to the right of its left edge, the line y = x)
    vlabel(axb, (0.5, 1.55), r"$\mathbf{A}\mathbf{e}_2$", color=ORANGE,
           ha="center", fontsize=LBL)
    axb.text(4.6, 0.30, r"eigenspace $\lambda=1$", color=GREEN,
             fontsize=13, ha="center", va="center")
    clean_axes(axb, lim=limb, hide=True)

    save(fig, "mdl-la-defective-shear")


def fig_psd():
    """Three 3-D surface plots of z = x^T A x: bowl (PD), trough (PSD), saddle."""
    fig = plt.figure(figsize=(10.5, 3.8))
    g = np.linspace(-1.5, 1.5, 60)
    X, Y = np.meshgrid(g, g)

    # the caption names the three cases (PD / PSD / indefinite), no titles
    specs = [
        np.array([[2.0, 0.0], [0.0, 1.0]]),
        np.array([[1.0, 0.0], [0.0, 0.0]]),
        np.array([[1.0, 0.0], [0.0, -1.0]]),
    ]
    for i, A in enumerate(specs, 1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        Z = A[0, 0] * X**2 + (A[0, 1] + A[1, 0]) * X * Y + A[1, 1] * Y**2
        ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True,
                        alpha=0.92, rstride=2, cstride=2)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_xlabel("$x_1$", labelpad=-10, fontsize=13)
        ax.set_ylabel("$x_2$", labelpad=-10, fontsize=13)
        ax.view_init(elev=24, azim=-52)
        try:
            ax.set_box_aspect((1, 1, 0.8))
        except Exception:
            pass
    save(fig, "mdl-la-psd")


def fig_gershgorin():
    """Gershgorin discs for a diagonally-dominant symmetric matrix, with the
    true eigenvalues overplotted, each inside its disc."""
    A = np.array([
        [1.0, 0.1, 0.1, 0.1],
        [0.1, 3.0, 0.2, 0.3],
        [0.1, 0.2, 5.0, 0.5],
        [0.1, 0.3, 0.5, 9.0],
    ])
    n = A.shape[0]
    centers = np.diag(A)
    radii = np.sum(np.abs(A), axis=1) - np.abs(centers)
    eig = np.linalg.eigvals(A)  # true eigenvalues (real, symmetric)

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    cols = [BLUE, ORANGE, GREEN, GRAY]
    for c, r, col in zip(centers, radii, cols):
        ax.add_patch(plt.Circle((c, 0), r, facecolor=col, alpha=0.18,
                                edgecolor=col, lw=1.6))
        ax.plot(c, 0, "x", color=col, ms=10, mew=2.2, zorder=5)
    # small black dots so the coloured centre x-marks stay visible underneath
    ax.plot(eig.real, np.zeros_like(eig.real), "o", color="black", ms=4.5,
            zorder=6, label="true eigenvalues")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("real axis", fontsize=13)
    ax.set_ylabel("imag axis", fontsize=13)
    ax.set_aspect("equal")
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlim(centers.min() - radii.max() - 0.5, centers.max() + radii.max() + 0.5)
    ax.legend(loc="upper left", fontsize=12)
    clean_axes(ax, equal=True)
    save(fig, "mdl-la-gershgorin")


def fig_power_iter():
    """(a) fan of arrows: a unit vector under repeated A-then-renormalize
    swinging onto the dominant eigenvector; (b) the ratio ||A^{k+1}v||/||A^k v||
    converging to |lambda_1|."""
    A = np.array([[3.0, 1.0], [1.0, 2.0]])  # symmetric; eigvals (5±√5)/2 ≈ 3.618, 1.382
    w, V = np.linalg.eigh(A)
    order = np.argsort(-np.abs(w))
    lam1 = w[order[0]]
    lam2 = w[order[1]]
    dom = V[:, order[0]]
    if dom[0] < 0:
        dom = -dom

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.0, 4.0))

    # (a) iterate directions
    v = np.array([0.0, 1.0])  # start far from dominant direction
    v = v / np.linalg.norm(v)
    n_it = 6
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, n_it + 1))
    # all the iterates live in the first quadrant -- show only that
    axis_cross(axa, (-0.12, 1.32), (-0.12, 1.32), color="black")
    # dominant eigenvector reference ray (first quadrant only)
    axa.plot([0, 1.28 * dom[0]], [0, 1.28 * dom[1]],
             color="black", lw=1.0, ls="--")
    cur = v.copy()
    for k in range(n_it + 1):
        arrow(axa, (0, 0), cur, color=cmap[k], lw=2.2)
        nxt = A @ cur
        cur = nxt / np.linalg.norm(nxt)
    arrow(axa, (0, 0), dom, color=ORANGE, lw=2.6)
    # offset w1 perpendicular to the dominant direction so it clears the ray
    dperp = np.array([-dom[1], dom[0]])
    wpos = dom * 1.10 + 0.17 * dperp
    axa.text(wpos[0], wpos[1], r"$\mathbf{w}_1$", color=ORANGE, fontsize=16,
             ha="left", va="bottom")
    axa.text(0.08, 1.16, r"$\mathbf{x}_0$", color="black", fontsize=15, ha="left")
    clean_axes(axa, lim=((-0.15, 1.34), (-0.15, 1.34)), hide=True)

    # (b) ratio convergence
    vk = np.array([0.3, 1.0])
    ks, ratios = [], []
    cur = vk.copy()
    norms = [np.linalg.norm(cur)]
    for k in range(14):
        cur = A @ cur
        norms.append(np.linalg.norm(cur))
    norms = np.array(norms)
    ratios = norms[1:] / norms[:-1]
    ks = np.arange(1, len(ratios) + 1)
    axb.plot(ks, ratios, "o-", color=BLUE, ms=5)
    axb.axhline(abs(lam1), color="black", ls="--", lw=1.4)
    axb.text(ks[-1], abs(lam1) + 0.05, rf"$|\lambda_1|\approx{abs(lam1):.3g}$",
             ha="right", va="bottom", color="black", fontsize=16)
    axb.set_xlabel("iteration $k$", fontsize=15)
    axb.set_ylabel(r"$\|A^{k+1}v\|/\|A^k v\|$", fontsize=15)
    axb.tick_params(labelsize=12)
    # the *norm ratio* converges at the squared rate (lambda_2/lambda_1)^2;
    # sit the note low-centre, clear of the (plateaued) curve
    axb.annotate(rf"gap $\sim(\lambda_2/\lambda_1)^{{2k}}="
                 rf"({(lam2 / lam1) ** 2:.2g})^k$",
                 xy=(0.36, 0.26), xycoords="axes fraction", fontsize=15,
                 color="black")
    axb.set_aspect("auto")
    clean_axes(axb, equal=False)
    save(fig, "mdl-la-power-iter")


# =========================================================================== #
# SVD (Section 1.3)                                                           #
# =========================================================================== #

def fig_svd_action():
    """Four panels left->right showing the rotate-scale-rotate action of a
    NON-symmetric 2x2 A on the unit circle.  Exact SVD via numpy."""
    A = np.array([[1.5, 0.9], [0.2, 1.3]])  # non-symmetric => U != V
    U, s, Vt = np.linalg.svd(A)
    V = Vt.T

    t = np.linspace(0, 2 * np.pi, 400)
    circle = np.vstack([np.cos(t), np.sin(t)])

    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.2))
    # equal aspect but a SHORT window in y (content only reaches ~1.3 in y): a
    # wide-but-short box keeps the vertical axis from towering over the shapes
    mx, my = 2.4, 2.05
    lim = ((-mx, mx), (-my, my))

    def tip_label(ax, tip, text, color):
        """Place a label well beyond an arrow tip (radially outside the shape),
        nudged perpendicular so it never sits under the arrowhead."""
        tip = np.asarray(tip, float)
        d = tip / np.linalg.norm(tip)
        perp = np.array([-d[1], d[0]])
        p = tip + 0.44 * d + 0.30 * perp
        ax.text(p[0], p[1], text, color=color, fontsize=14,
                ha="center", va="center")

    # (1) input unit circle with right singular vectors v1, v2
    ax = axes[0]
    ax.set_title("(1) input", fontsize=14)
    ax.plot(circle[0], circle[1], color=GRAY, lw=1.6)
    arrow(ax, (0, 0), V[:, 0], color=BLUE, lw=2.4)
    arrow(ax, (0, 0), V[:, 1], color=ORANGE, lw=2.4)
    tip_label(ax, V[:, 0], r"$\mathbf{v}_1$", BLUE)
    tip_label(ax, V[:, 1], r"$\mathbf{v}_2$", ORANGE)

    # (2) after V^T: circle unchanged, v's rotated onto the axes
    ax = axes[1]
    ax.set_title(r"(2) after $\mathbf{V}^\top$", fontsize=14)
    c2 = Vt @ circle
    ax.plot(c2[0], c2[1], color=GRAY, lw=1.6)
    e1 = Vt @ V[:, 0]
    e2 = Vt @ V[:, 1]
    arrow(ax, (0, 0), e1, color=BLUE, lw=2.4)
    arrow(ax, (0, 0), e2, color=ORANGE, lw=2.4)

    # (3) after Sigma: axis-aligned ellipse, semi-axes sigma1, sigma2
    ax = axes[2]
    ax.set_title(r"(3) after $\mathbf{\Sigma}$", fontsize=14)
    Sig = np.diag(s)
    c3 = Sig @ Vt @ circle
    ax.plot(c3[0], c3[1], color=BLUE, lw=2.2)
    arrow(ax, (0, 0), [s[0], 0], color=BLUE, lw=2.4)
    arrow(ax, (0, 0), [0, s[1]], color=ORANGE, lw=2.4)
    # sit the sigma labels OUTSIDE the ellipse outline (below / above it)
    ax.text(s[0] * 0.5, -1.25, rf"$\sigma_1={s[0]:.2g}$", color=BLUE,
            fontsize=13, ha="center", va="top")
    ax.text(0.28, 1.2, rf"$\sigma_2={s[1]:.2g}$", color=ORANGE,
            fontsize=13, ha="left", va="bottom")

    # (4) after U: ellipse rotated; draw sigma1 u1, sigma2 u2 as its axes
    ax = axes[3]
    ax.set_title(r"(4) after $\mathbf{U}=\mathbf{A}$", fontsize=14)
    c4 = U @ Sig @ Vt @ circle
    ax.plot(c4[0], c4[1], color=BLUE, lw=2.2)
    a1 = s[0] * U[:, 0]
    a2 = s[1] * U[:, 1]
    arrow(ax, (0, 0), a1, color=BLUE, lw=2.4)
    arrow(ax, (0, 0), a2, color=ORANGE, lw=2.4)
    tip_label(ax, a1, r"$\sigma_1\mathbf{u}_1$", BLUE)
    tip_label(ax, a2, r"$\sigma_2\mathbf{u}_2$", ORANGE)

    for ax in axes:
        clean_axes(ax, lim=lim, hide=True)
        axis_cross(ax, (-mx, mx), (-my, my), color="black")
    save(fig, "mdl-la-svd-action")


def fig_svd_subspaces():
    """Strang's four fundamental subspaces (schematic, labelled)."""
    fig, ax = plt.subplots(figsize=(8.6, 4.6))

    # domain box (left) and codomain box (right)
    dom_x, cod_x = 0.0, 6.0
    bw, bh = 2.4, 4.0
    y0 = 0.0

    def box(x, split_frac, top_label, bot_label, top_col, bot_col):
        # full box outline
        ax.add_patch(Rectangle((x, y0), bw, bh, fill=False, edgecolor="black",
                               lw=1.4))
        sy = y0 + bh * (1 - split_frac)
        # top region
        ax.add_patch(Rectangle((x, sy), bw, bh - (sy - y0), facecolor=top_col,
                               alpha=0.16, lw=0))
        # bottom region
        ax.add_patch(Rectangle((x, y0), bw, sy - y0, facecolor=bot_col,
                               alpha=0.16, lw=0))
        ax.plot([x, x + bw], [sy, sy], color="black", lw=0.9, ls="--")
        ax.text(x + bw / 2, y0 + bh - (bh - (sy - y0)) / 2, top_label,
                ha="center", va="center", fontsize=13, color=top_col)
        ax.text(x + bw / 2, y0 + (sy - y0) / 2, bot_label,
                ha="center", va="center", fontsize=13, color="black")
        return sy

    sy_dom = box(dom_x, 0.62, r"row space" + "\n" + r"$\dim r$",
                 r"null space $\mathcal{N}(\mathbf{A})$", BLUE, GRAY)
    sy_cod = box(cod_x, 0.62, r"column space" + "\n" + r"$\dim r$",
                 r"left null space" + "\n" + r"$\mathcal{N}(\mathbf{A}^\top)$", GREEN, GRAY)

    ax.text(dom_x + bw / 2, y0 + bh + 0.3, r"$\mathbb{R}^n$ (domain)", ha="center",
            fontsize=14)
    ax.text(cod_x + bw / 2, y0 + bh + 0.3, r"$\mathbb{R}^m$ (codomain)", ha="center",
            fontsize=14)

    # arrow v_i |-> sigma_i u_i (row space -> column space)
    p_row = np.array([dom_x + bw, y0 + bh * 0.78])
    p_col = np.array([cod_x, y0 + bh * 0.78])
    ax.add_patch(FancyArrowPatch(p_row, p_col, arrowstyle="->",
                                 mutation_scale=16, color=BLUE, lw=2.0))
    ax.text((p_row[0] + p_col[0]) / 2, p_row[1] + 0.18,
            r"$\mathbf{v}_i \mapsto \sigma_i\mathbf{u}_i$", ha="center",
            va="bottom", fontsize=13, color=BLUE)

    # null space collapses to 0 (the zero point sits in the gap, just left of
    # the codomain box so neither the dot nor its label touch the box edge)
    p_null = np.array([dom_x + bw, y0 + bh * 0.22])
    p_zero = np.array([cod_x - 0.45, y0 + bh * 0.20])
    ax.add_patch(FancyArrowPatch(p_null, p_zero, arrowstyle="->",
                                 mutation_scale=14, color=GRAY, lw=1.6,
                                 linestyle=(0, (4, 3))))
    ax.plot(*p_zero, "o", color="black", ms=6)
    ax.text(p_zero[0], p_zero[1] + 0.32, r"$\mathbf{0}$", ha="center",
            va="bottom", fontsize=13)
    ax.text((p_null[0] + p_zero[0]) / 2, p_null[1] - 0.26, r"$\mapsto \mathbf{0}$",
            ha="center", va="top", fontsize=12, color="black")

    ax.set_xlim(-0.6, cod_x + bw + 0.6)
    ax.set_ylim(-0.8, bh + 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "mdl-la-svd-subspaces")


def _structured_image(n=128):
    """Deterministic structured grayscale image (no network) with a *slowly
    decaying* singular spectrum, so low-rank approximation is visibly lossy at
    moderate rank.  Tries ``scipy.datasets.ascent`` first (a real photo, the
    classic SVD demo); falls back to a fixed-seed synthetic image."""
    try:
        from scipy.datasets import ascent  # real 512x512 photo (needs cache)
        img = ascent().astype(float)
        return img / img.max()
    except Exception:
        pass
    # Fixed-seed synthetic image.  Smooth shapes give the leading singular
    # values; a deterministic mid-frequency texture + a radial chirp supply a
    # genuine slowly-decaying tail (so rank-20 still carries visible error).
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:n, 0:n] / n
    img = np.zeros((n, n))
    img += np.exp(-((xx - 0.3) ** 2 + (yy - 0.35) ** 2) / 0.05)
    img += 0.8 * np.exp(-((xx - 0.7) ** 2 + (yy - 0.6) ** 2) / 0.03)
    img[20:50, 70:110] += 0.6
    img[80:100, 20:60] += 0.4
    # radial chirp: spatial frequency grows with radius -> many singular values
    r = np.hypot(xx - 0.5, yy - 0.5)
    img += 0.4 * np.cos(30 * r ** 2)
    # smoothed random texture (low-pass a fixed-seed field) -> long gentle tail
    noise = rng.standard_normal((n, n))
    k = np.exp(-0.5 * (np.arange(n) - n / 2) ** 2 / 4.0 ** 2)
    k /= k.sum()
    sm = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, noise)
    sm = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, sm)
    img += 0.5 * sm / np.abs(sm).max()
    img -= img.min()
    img /= img.max()
    return img


def fig_eckart_young():
    """(a) singular-value spectrum (log y) with top-k region shaded;
    (b) reconstructions at ranks 1, 5, 20, full with relative Frobenius error."""
    img = _structured_image()
    U, s, Vt = np.linalg.svd(img, full_matrices=False)
    full_norm = np.linalg.norm(s)

    fig = plt.figure(figsize=(12.0, 3.2))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.25, 1, 1, 1, 1])
    axs = fig.add_subplot(gs[0, 0])
    axs.semilogy(np.arange(1, len(s) + 1), s, color=BLUE, lw=1.8)
    axs.axvspan(1, 20, color=ORANGE, alpha=0.15)
    axs.text(20, s[0], "top-$k$", color=ORANGE, fontsize=12, ha="left", va="top")
    axs.set_xlabel("index $i$", fontsize=12)
    axs.set_ylabel(r"$\sigma_i$ (log)", fontsize=12)
    axs.set_title("spectrum", fontsize=13)
    axs.set_aspect("auto")
    axs.spines["top"].set_visible(False)
    axs.spines["right"].set_visible(False)

    ranks = [1, 5, 20, len(s)]
    for j, k in enumerate(ranks):
        ax = fig.add_subplot(gs[0, 1 + j])
        approx = (U[:, :k] * s[:k]) @ Vt[:k, :]
        # relative Frobenius error = ||A - A_k||_F / ||A||_F
        rel = np.linalg.norm(s[k:]) / full_norm if k < len(s) else 0.0
        ax.imshow(approx, cmap="gray", vmin=0, vmax=1)
        ttl = f"full ({k})" if k == len(s) else f"rank {k}"
        ax.set_title(f"{ttl}\nerr {rel:.3f}", fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
    save(fig, "mdl-la-eckart-young")


def fig_lora():
    """LoRA schematic: frozen weight W beside a trainable low-rank bypass
    B(m x r) @ A(r x n), outputs summed.  Dimensions/parameter counts use the
    chapter's own 4096x4096, r=8 (0.39%) example."""
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    xW, xB = 2.6, 7.4           # left (W) and right (A/B) column centres
    ysplit, ymerge = 1.55, 6.0  # horizontal wiring rails

    # orthogonal wiring first (drawn under everything): x -> both columns -> +
    def wire(pts):
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=GRAY, lw=1.6,
                solid_capstyle="round", zorder=1)
    wire([(5, 0.75), (5, ysplit), (xW, ysplit), (xW, 2.4)])   # x -> W
    wire([(5, 0.75), (5, ysplit), (xB, ysplit), (xB, 2.4)])   # x -> A
    wire([(xW, 4.8), (xW, ymerge), (5, ymerge), (5, 6.16)])   # W -> +
    wire([(xB, 5.0), (xB, ymerge), (5, ymerge), (5, 6.16)])   # B -> +

    # input / output nodes
    ax.plot(5, 0.75, "o", color="black", ms=5, zorder=4)
    ax.text(5.3, 0.62, r"$\mathbf{x}$", fontsize=15, ha="left", va="center")
    arrow(ax, (5, 6.66), (5, 7.5), color=GRAY, lw=1.6, mut=12)
    ax.plot(5, 7.5, "o", color="black", ms=5, zorder=4)
    ax.text(5.3, 7.5, r"$\mathbf{h}=\mathbf{W}\mathbf{x}+\mathbf{B}\mathbf{A}\mathbf{x}$",
            fontsize=13, ha="left", va="center")

    # frozen weight W (left column) -- wide enough for the "pretrained" line
    ax.add_patch(Rectangle((0.9, 2.4), 3.4, 2.4, facecolor=BLUE, alpha=0.15,
                           edgecolor=BLUE, lw=1.8))
    ax.text(xW, 3.95, r"$\mathbf{W}$", fontsize=16, ha="center", va="center",
            color=BLUE)
    ax.text(xW, 3.2, "pretrained, frozen\n" + r"$4096\times4096$",
            fontsize=10.5, ha="center", va="center", color=BLUE)
    ax.text(xW, 1.32, r"$mn=16.8$M params", fontsize=10.5, ha="center",
            va="top", color="black")   # below the split rail, clear of wires

    # low-rank bypass (right column): A compresses to r, B expands back
    ax.add_patch(Polygon([(6.3, 2.4), (8.5, 2.4), (7.75, 3.5), (7.05, 3.5)],
                         closed=True, facecolor=ORANGE, alpha=0.18,
                         edgecolor=ORANGE, lw=1.8))
    ax.text(xB, 2.85, r"$\mathbf{A}$ ($r\times n$)", fontsize=11.5,
            ha="center", va="center", color=ORANGE)
    ax.add_patch(Polygon([(7.05, 3.9), (7.75, 3.9), (8.5, 5.0), (6.3, 5.0)],
                         closed=True, facecolor=ORANGE, alpha=0.18,
                         edgecolor=ORANGE, lw=1.8))
    ax.text(xB, 4.55, r"$\mathbf{B}$ ($m\times r$)", fontsize=11.5,
            ha="center", va="center", color=ORANGE)
    arrow(ax, (xB, 3.5), (xB, 3.9), color=ORANGE, lw=1.8, mut=11)
    ax.text(xB + 0.32, 3.7, r"$r=8$", fontsize=11, ha="left", va="center",
            color=ORANGE)
    # trainable-count note in the free right margin, beside the trapezoids
    ax.text(8.8, 4.35, "trainable:\n" + r"$r(m{+}n)\approx65.5$K"
            + "\n" + r"($0.39\%$)", fontsize=10.5, ha="left", va="center",
            color="black")

    # summation node
    ax.add_patch(plt.Circle((5, 6.4), 0.26, facecolor="white",
                            edgecolor="black", lw=1.4, zorder=5))
    ax.text(5, 6.4, "+", fontsize=15, ha="center", va="center", zorder=6)

    ax.set_xlim(0.6, 11.4)
    ax.set_ylim(0.4, 8.0)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "mdl-la-lora")


def fig_condition():
    """Two contour plots of f(x)=1/2 x^T A x: well-conditioned (near-circular,
    short GD path) and ill-conditioned (elongated valley, zig-zag GD path).
    Real GD steps draw the paths."""
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.0, 4.2))

    g = np.linspace(-1.6, 1.6, 200)
    X, Y = np.meshgrid(g, g)

    def run(ax, A, eta, x0, steps):
        Z = 0.5 * (A[0, 0] * X**2 + (A[0, 1] + A[1, 0]) * X * Y + A[1, 1] * Y**2)
        ax.contour(X, Y, Z, levels=14, colors=[GRAY], linewidths=0.8)
        path = [np.array(x0, float)]
        x = np.array(x0, float)
        for _ in range(steps):
            grad = A @ x  # grad of 1/2 x^T A x  (A symmetric)
            x = x - eta * grad
            path.append(x.copy())
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], "-o", color=ORANGE, ms=3.5, lw=1.6)
        ax.plot(0, 0, "*", color=GREEN, ms=12)
        clean_axes(ax, lim=((-1.6, 1.6), (-1.6, 1.6)), hide=True)

    # well-conditioned: near-isotropic; step size safe -> nearly straight path
    run(axa, np.array([[1.0, 0.0], [0.0, 1.3]]),
        eta=0.55, x0=(-1.3, 1.1), steps=10)
    # ill-conditioned: elongated valley; eta near stability bound -> zig-zag
    run(axb, np.array([[1.0, 0.0], [0.0, 18.0]]),
        eta=0.102, x0=(-1.3, 1.0), steps=24)
    save(fig, "mdl-la-condition")


def fig_pca():
    """2-D point cloud with the two principal directions (right singular vectors
    of the centered data) drawn from the mean, scaled by the singular values."""
    rng = np.random.default_rng(3)
    n = 350
    # anisotropic Gaussian, rotated
    cov_axes = np.array([[1.0, 0.0], [0.0, 0.28]])
    theta = np.radians(32)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    L = R @ np.sqrt(cov_axes)
    data = (L @ rng.standard_normal((2, n))).T + np.array([1.0, 0.6])

    mean = data.mean(axis=0)
    Xc = data - mean
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    pcs = Vt  # rows are principal directions
    scales = s / np.sqrt(n)  # ~ std along each PC

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.scatter(data[:, 0], data[:, 1], s=10, color=BLUE, alpha=0.45,
               edgecolors="none")
    ax.plot(*mean, "o", color="black", ms=5)
    for i, (col, lab) in enumerate(zip([ORANGE, GREEN], ["PC1", "PC2"])):
        d = pcs[i]
        if d[np.argmax(np.abs(d))] < 0:
            d = -d
        tip = mean + 2.2 * scales[i] * d
        arrow(ax, mean, tip, color=col, lw=2.8)
        ax.text(tip[0] + 0.10 * d[0], tip[1] + 0.10 * d[1] + 0.06, lab,
                color=col, fontsize=14, fontweight="bold",
                ha="center", va="center")
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$", fontsize=13)
    ax.set_ylabel("$x_2$", fontsize=13)
    clean_axes(ax, equal=True)
    save(fig, "mdl-la-pca")


def fig_transient_growth():
    """Transient amplification made visible.  Left: ||A^k||_2 for the
    non-normal A = [[0.9, 4], [0, 0.8]] versus the diagonal (normal) matrix
    with identical eigenvalues — same spectral radius, same asymptotic decay
    rate, yet A first amplifies by an order of magnitude.  Right: the
    epsilon-pseudospectra of both matrices — level sets of
    sigma_min(zI - A) — showing that tiny perturbations move A's eigenvalues
    far outside the unit disk while the normal matrix's stay put.  Same
    matrices as the accompanying notebook cell."""
    A = np.array([[0.9, 4.0], [0.0, 0.8]])
    D = np.diag([0.9, 0.8])
    ks = np.arange(0, 61)
    norm_A = np.array([np.linalg.norm(np.linalg.matrix_power(A, k), 2)
                       for k in ks])
    norm_D = np.array([np.linalg.norm(np.linalg.matrix_power(D, k), 2)
                       for k in ks])
    k_peak = int(norm_A.argmax())

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.4, 3.9))
    axa.semilogy(ks, norm_A, color=BLUE, lw=2.4, zorder=4)
    axa.semilogy(ks, norm_D, color=ORANGE, lw=2.2, ls="--", zorder=4)
    axa.axhline(1.0, color=GRAY, lw=0.9, ls=":", zorder=2)
    axa.plot([k_peak], [norm_A[k_peak]], "o", color=BLUE, ms=6, zorder=6)
    axa.annotate(f"peak at $k={k_peak}$:\ntransient growth $\\times$"
                 f"{norm_A[k_peak]:.0f}",
                 xy=(k_peak, norm_A[k_peak]),
                 xytext=(k_peak + 9, norm_A[k_peak] * 1.6),
                 fontsize=10.5, ha="left", va="center", color="black",
                 arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.2))
    axa.text(51, 0.30, "non-normal $\\mathbf{A}$", color=BLUE, fontsize=11,
             ha="center", zorder=7,
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.9,
                       pad=1.5))
    axa.text(15, 0.022, "normal, same\neigenvalues", color=ORANGE,
             fontsize=10.5, ha="center", va="center")
    axa.set_xlabel(r"$k$")
    axa.set_ylabel(r"$\|\mathbf{A}^k\|_2$")
    axa.set_ylim(1e-3, 40)
    axa.set_title("same spectrum, different transients", fontsize=12)

    # ----- right: pseudospectra via sigma_min(zI - M) on a grid -----
    gx = np.linspace(-1.6, 2.6, 320)
    gy = np.linspace(-1.7, 1.7, 260)
    GX, GY = np.meshgrid(gx, gy)
    Z = GX + 1j * GY

    def smin_grid(M):
        out = np.empty_like(GX)
        for i in range(GX.shape[0]):
            for j in range(GX.shape[1]):
                out[i, j] = np.linalg.svd(
                    Z[i, j] * np.eye(2) - M, compute_uv=False)[-1]
        return out

    levels = [0.05, 0.1, 0.2, 0.4]
    theta = np.linspace(0, 2 * np.pi, 200)
    axb.plot(np.cos(theta), np.sin(theta), color=GRAY, lw=1.0, ls=":",
             zorder=2)
    axb.contour(GX, GY, smin_grid(A), levels=levels, colors=[BLUE],
                linewidths=1.3, zorder=4)
    axb.contour(GX, GY, smin_grid(D), levels=levels, colors=[ORANGE],
                linewidths=1.1, linestyles="--", zorder=3)
    axb.plot([0.9, 0.8], [0.0, 0.0], "o", color="black", ms=5, zorder=6)
    axb.text(1.72, 1.28, "$\\varepsilon$-pseudospectra\nof $\\mathbf{A}$",
             color=BLUE, fontsize=10.5, ha="center", va="center", zorder=7,
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.9,
                       pad=1.5))
    axb.text(-0.55, -1.15, "of the normal matrix:\ntight disks",
             color=ORANGE, fontsize=10.5, ha="center", va="center", zorder=7,
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.9,
                       pad=1.5))
    axb.text(-1.18, 0.82, "unit circle", color=GRAY, fontsize=10,
             ha="center", zorder=7,
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.9,
                       pad=1.0))
    axb.set_xlabel(r"$\mathrm{Re}\,z$")
    axb.set_ylabel(r"$\mathrm{Im}\,z$")
    axb.set_aspect("equal")
    axb.set_title("tiny perturbations, runaway eigenvalues", fontsize=12)

    for ax in (axa, axb):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    save(fig, "mdl-la-transient-growth")


# =========================================================================== #
# Driver                                                                      #
# =========================================================================== #

FIGURES = [
    # geometry
    fig_vectors,
    fig_vector_add,
    fig_span,
    fig_angle,
    fig_projection,
    fig_hyperplane,
    fig_linear_map,
    fig_determinant,
    fig_null_collapse,
    fig_cosine_highd,
    # eigendecomposition
    fig_eig_ellipse,
    fig_defective_shear,
    fig_psd,
    fig_gershgorin,
    fig_power_iter,
    # svd
    fig_svd_action,
    fig_svd_subspaces,
    fig_eckart_young,
    fig_lora,
    fig_condition,
    fig_pca,
    fig_transient_growth,
]


def main():
    for fn in FIGURES:
        fn()

    print(f"\nWrote {len(WRITTEN)} figures to {IMG_DIR}:")
    for p in WRITTEN:
        size = os.path.getsize(p)
        assert os.path.exists(p), f"missing: {p}"
        assert size > 0, f"empty: {p}"
        with open(p, "r", encoding="utf-8") as fh:
            head = fh.read(400)
        assert "<svg" in head, f"not valid SVG: {p}"
        print(f"  {os.path.basename(p):28s} {size:>8,d} bytes")

    print(f"\nAll {len(WRITTEN)} SVGs verified present, non-empty, valid.")


if __name__ == "__main__":
    main()
