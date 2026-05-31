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
    as a translation-invariant arrow drawn in three places."""
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(8.0, 3.8))

    # (a) point with dashed coordinate drops
    axa.set_title("(a) a point")
    axis_cross(axa, (-0.6, 3.8), (-0.6, 2.8))
    p = np.array([3.0, 2.0])
    axa.plot(*p, "o", color=BLUE, ms=8, zorder=5)
    axa.plot([p[0], p[0]], [0, p[1]], "--", color=LIGHT, lw=1.2)
    axa.plot([0, p[0]], [p[1], p[1]], "--", color=LIGHT, lw=1.2)
    axa.text(p[0], -0.22, "3", color=BLUE, ha="center", va="top")
    axa.text(-0.18, p[1], "2", color=BLUE, ha="right", va="center")
    vlabel(axa, (p[0] + 0.12, p[1] + 0.18), r"$(3,2)$", color=BLUE, ha="left")
    clean_axes(axa, lim=((-0.6, 3.8), (-0.6, 2.8)), hide=True)

    # (b) same vector, translation-invariant arrows
    axb.set_title("(b) a direction")
    axis_cross(axb, (-0.6, 5.6), (-1.6, 4.0))
    v = np.array([3.0, 2.0])
    bases = [np.array([0.0, 0.0]), np.array([1.2, -1.3]), np.array([2.3, 1.6])]
    cols = [BLUE, ORANGE, GREEN]
    for b, c in zip(bases, cols):
        arrow(axb, b, b + v, color=c, lw=2.2)
    vlabel(axb, (1.4, 1.25), r"$(3,2)$", color=BLUE, ha="center")
    clean_axes(axb, lim=((-0.6, 5.6), (-1.6, 4.0)), hide=True)

    save(fig, "mdl-la-vectors")


def fig_vector_add():
    """Tip-to-tail addition u, then v from u's tip, resultant u+v from origin."""
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    u = np.array([3.0, 1.0])
    v = np.array([1.0, 2.0])
    axis_cross(ax, (-0.6, 4.8), (-0.6, 3.6))
    arrow(ax, (0, 0), u, color=BLUE, lw=2.2)
    arrow(ax, u, u + v, color=ORANGE, lw=2.2)
    arrow(ax, (0, 0), u + v, color=GREEN, lw=2.4)
    vlabel(ax, (1.5, 0.30), r"$\mathbf{u}$", color=BLUE)
    vlabel(ax, (3.6, 2.0), r"$\mathbf{v}$", color=ORANGE)
    vlabel(ax, (1.7, 1.9), r"$\mathbf{u}+\mathbf{v}$", color=GREEN)
    clean_axes(ax, lim=((-0.6, 4.8), (-0.6, 3.6)), hide=True)
    save(fig, "mdl-la-vector-add")


def fig_angle():
    """Two vectors from origin with angle theta drawn as an arc."""
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    u = np.array([3.0, 0.5])
    v = np.array([1.2, 2.6])
    axis_cross(ax, (-0.5, 3.8), (-0.5, 3.2))
    arrow(ax, (0, 0), u, color=BLUE, lw=2.4)
    arrow(ax, (0, 0), v, color=ORANGE, lw=2.4)
    a1 = np.degrees(np.arctan2(u[1], u[0]))
    a2 = np.degrees(np.arctan2(v[1], v[0]))
    ax.add_patch(Arc((0, 0), 1.6, 1.6, angle=0, theta1=a1, theta2=a2,
                     color=GRAY, lw=1.6))
    mid = np.radians((a1 + a2) / 2)
    vlabel(ax, (1.05 * np.cos(mid), 1.05 * np.sin(mid)), r"$\theta$", color=GRAY)
    vlabel(ax, (u[0] + 0.12, u[1]), r"$\mathbf{v}$", color=BLUE, ha="left")
    vlabel(ax, (v[0] + 0.05, v[1] + 0.18), r"$\mathbf{w}$", color=ORANGE)
    clean_axes(ax, lim=((-0.5, 3.8), (-0.5, 3.2)), hide=True)
    save(fig, "mdl-la-angle")


def fig_projection():
    """(a) generic projection of v onto w with right-angle residual;
    (b) the Cauchy-Schwarz equality case (v collinear with w)."""
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(8.4, 3.9))

    # (a) generic
    axa.set_title("(a) projection")
    w = np.array([3.2, 0.6])
    v = np.array([1.6, 2.2])
    wn = w / np.linalg.norm(w)
    proj_len = float(v @ wn)
    proj = proj_len * wn
    axis_cross(axa, (-0.4, 4.0), (-0.4, 2.8))
    arrow(axa, (0, 0), w, color=ORANGE, lw=2.2)
    arrow(axa, (0, 0), v, color=BLUE, lw=2.4)
    # projection as a thick segment along w
    axa.plot([0, proj[0]], [0, proj[1]], color=GREEN, lw=5, solid_capstyle="round",
             zorder=2)
    # residual r = v - proj (dashed) meeting w at a right angle
    axa.plot([proj[0], v[0]], [proj[1], v[1]], "--", color=GRAY, lw=1.6)
    right_angle(axa, proj, -wn, (v - proj), size=0.22, color=GRAY)
    vlabel(axa, (w[0] + 0.05, w[1] - 0.18), r"$\mathbf{w}$", color=ORANGE, ha="left")
    vlabel(axa, (v[0] - 0.1, v[1] + 0.2), r"$\mathbf{v}$", color=BLUE)
    vlabel(axa, (proj[0] * 0.55 - 0.1, proj[1] * 0.55 + 0.32),
           r"$\|\mathbf{v}\|\cos\theta$", color=GREEN, ha="center", fontsize=10)
    vlabel(axa, ((proj[0] + v[0]) / 2 + 0.22, (proj[1] + v[1]) / 2),
           r"$\mathbf{r}$", color=GRAY, ha="left")
    clean_axes(axa, lim=((-0.4, 4.0), (-0.4, 2.8)), hide=True)

    # (b) equality: v collinear with w, residual vanishes
    axb.set_title("(b) equality")
    w2 = np.array([3.4, 1.3])
    v2 = 0.62 * w2  # collinear
    axis_cross(axb, (-0.4, 4.0), (-0.4, 2.4))
    arrow(axb, (0, 0), w2, color=ORANGE, lw=2.2)
    axb.plot([0, v2[0]], [0, v2[1]], color=GREEN, lw=5, solid_capstyle="round",
             zorder=2)
    arrow(axb, (0, 0), v2, color=BLUE, lw=2.4)
    vlabel(axb, (w2[0] + 0.05, w2[1]), r"$\mathbf{w}$", color=ORANGE, ha="left")
    vlabel(axb, (v2[0] * 0.5 - 0.1, v2[1] * 0.5 + 0.32), r"$\mathbf{v}$",
           color=BLUE)
    axb.text(2.0, 2.05,
             r"$|\mathbf{v}\cdot\mathbf{w}| = \|\mathbf{v}\|\,\|\mathbf{w}\|$",
             ha="center", va="center", fontsize=10.5)
    clean_axes(axb, lim=((-0.4, 4.0), (-0.4, 2.4)), hide=True)

    save(fig, "mdl-la-projection")


def fig_hyperplane():
    """The line w.x = b in 2-D: normal w, two parallel level sets for two
    offsets b, shaded half-space, signed distance b/||w|| marked.  Plus a small
    3-D panel showing the plane analog."""
    fig = plt.figure(figsize=(8.6, 4.0))
    axa = fig.add_subplot(1, 2, 1)
    axb = fig.add_subplot(1, 2, 2, projection="3d")

    # --- 2-D panel ---
    axa.set_title("(a) a line in 2-D")
    w = np.array([1.0, 1.6])
    wn = w / np.linalg.norm(w)
    nrm = np.linalg.norm(w)
    b1, b2 = 1.4, 3.4

    L = 3.6
    perp = np.array([-wn[1], wn[0]])  # along the level set
    lim = ((-L, L), (-L, L))

    def level_line(b):
        c = (b / nrm) * wn  # closest point on line to origin
        return np.array([c - L * perp, c + L * perp])

    # shade half-space w.x > b1
    seg = level_line(b1)
    far = wn * (2 * L)  # push out along +w
    poly = np.array([seg[0], seg[1], seg[1] + far, seg[0] + far])
    axa.add_patch(Polygon(poly, closed=True, color=BLUE, alpha=0.10, lw=0))

    # keep the text upright: use the line direction whose x-component is >= 0
    text_dir = perp if perp[0] >= 0 else -perp
    line_rot = np.degrees(np.arctan2(text_dir[1], text_dir[0]))
    for b, c in ((b1, BLUE), (b2, GRAY)):
        seg = level_line(b)
        axa.plot(seg[:, 0], seg[:, 1], "--", color=c, lw=1.8)
        # label near the line's lower-right end, nudged to the +w side so it
        # clears the dashed line itself and the distance marker
        anchor = (b / nrm) * wn + 0.62 * L * text_dir
        lab_pos = anchor + 0.40 * wn
        axa.text(lab_pos[0], lab_pos[1],
                 rf"$\mathbf{{w}}\!\cdot\!\mathbf{{x}}={b:g}$", color=c,
                 fontsize=9, ha="center", va="center", rotation=line_rot,
                 rotation_mode="anchor")

    axis_cross(axa, (-L, L), (-L, L))
    arrow(axa, (0, 0), w, color=ORANGE, lw=2.4)
    vlabel(axa, (w[0] + 0.18, w[1] + 0.12), r"$\mathbf{w}$", color=ORANGE, ha="left")

    # signed distance b1/||w|| as a labeled segment from origin to the b1 line
    cp = (b1 / nrm) * wn
    axa.plot([0, cp[0]], [0, cp[1]], color=GREEN, lw=3, solid_capstyle="round",
             zorder=4)
    # label to the left of the segment so it does not sit on any line
    dl = (0.5 * cp) + np.array([-0.95, 0.15])
    axa.text(dl[0], dl[1], r"$b/\|\mathbf{w}\|$", color=GREEN, fontsize=9.5,
             ha="center", va="center")
    axa.text(0.55 * L, 0.80 * L, r"$\mathbf{w}\cdot\mathbf{x}>b$", color=BLUE,
             fontsize=9.5, ha="center")
    clean_axes(axa, lim=lim, hide=True)

    # --- 3-D panel: plane w.x = b ---
    axb.set_title("(b) a plane in 3-D")
    w3 = np.array([0.6, 0.5, 1.0])
    b3 = 0.0
    gx, gy = np.meshgrid(np.linspace(-1, 1, 12), np.linspace(-1, 1, 12))
    gz = (b3 - w3[0] * gx - w3[1] * gy) / w3[2]
    axb.plot_surface(gx, gy, gz, color=BLUE, alpha=0.35, linewidth=0,
                     antialiased=True)
    # normal vector
    axb.quiver(0, 0, 0, w3[0], w3[1], w3[2], color=ORANGE, lw=2.2,
               arrow_length_ratio=0.18)
    axb.text(w3[0], w3[1], w3[2] + 0.15, r"$\mathbf{w}$", color=ORANGE)
    axb.set_xticks([]); axb.set_yticks([]); axb.set_zticks([])
    axb.set_xlabel(""); axb.set_ylabel(""); axb.set_zlabel("")
    axb.view_init(elev=18, azim=-58)
    try:
        axb.set_box_aspect((1, 1, 0.9))
    except Exception:
        pass

    save(fig, "mdl-la-hyperplane")


def fig_linear_map():
    """A unit-square grid (light) and its image under a non-trivial 2x2 matrix
    (skew + scale), so the reader sees a matrix bending space."""
    A = np.array([[1.4, 0.9], [0.3, 1.2]])
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(8.4, 4.2))

    n = 5
    ts = np.linspace(0, n, 200)
    grid = list(range(n + 1))

    def draw_grid(ax, M, color, lw):
        for k in grid:  # vertical lines x=k
            pts = M @ np.vstack([np.full_like(ts, k), ts])
            ax.plot(pts[0], pts[1], color=color, lw=lw)
        for k in grid:  # horizontal lines y=k
            pts = M @ np.vstack([ts, np.full_like(ts, k)])
            ax.plot(pts[0], pts[1], color=color, lw=lw)

    I = np.eye(2)
    axa.set_title("(a) original grid")
    draw_grid(axa, I, LIGHT, 1.0)
    # highlight unit basis vectors
    arrow(axa, (0, 0), (1, 0), color=BLUE, lw=2.2)
    arrow(axa, (0, 0), (0, 1), color=ORANGE, lw=2.2)
    vlabel(axa, (1.0, -0.3), r"$\mathbf{e}_1$", color=BLUE)
    vlabel(axa, (-0.35, 1.0), r"$\mathbf{e}_2$", color=ORANGE)
    clean_axes(axa, lim=((-0.6, 5.6), (-0.6, 5.6)), hide=True)

    axb.set_title(r"(b) image under $\mathbf{A}$")
    draw_grid(axb, A, LIGHT, 1.0)
    a1 = A @ np.array([1, 0])
    a2 = A @ np.array([0, 1])
    arrow(axb, (0, 0), a1, color=BLUE, lw=2.4)
    arrow(axb, (0, 0), a2, color=ORANGE, lw=2.4)
    vlabel(axb, (a1[0] + 0.1, a1[1] - 0.35), r"$\mathbf{A}\mathbf{e}_1$", color=BLUE, ha="left")
    vlabel(axb, (a2[0] - 0.55, a2[1] + 0.2), r"$\mathbf{A}\mathbf{e}_2$", color=ORANGE, ha="right")
    L = (n + 0.6) * 1.6
    clean_axes(axb, lim=((-0.8, L), (-0.8, L * 0.78)), hide=True)

    save(fig, "mdl-la-linear-map")


def fig_determinant():
    """Three panels: (a) unit square -> parallelogram, area = |det A|;
    (b) negative determinant (orientation flip); (c) degenerate (det 0)."""
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.6))
    unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]).T  # columns: corners

    def show_para(ax, A, title, flip=False, degenerate=False):
        det = np.linalg.det(A)
        img = A @ unit
        # original unit square (faint)
        ax.add_patch(Polygon(unit.T, closed=True, fill=False, edgecolor=LIGHT,
                             lw=1.2, ls="--"))
        col = GREEN if det > 0 else ORANGE
        if degenerate:
            # collapses to a segment
            ax.plot(img[0], img[1], color=ORANGE, lw=4, solid_capstyle="round")
        else:
            ax.add_patch(Polygon(img.T, closed=True, facecolor=col, alpha=0.25,
                                 edgecolor=col, lw=2))
        c1 = A @ np.array([1, 0])
        c2 = A @ np.array([0, 1])
        arrow(ax, (0, 0), c1, color=BLUE, lw=2.0)
        arrow(ax, (0, 0), c2, color=ORANGE, lw=2.0)
        if flip:
            ax.add_patch(Arc((0.0, 0.0), 1.0, 1.0, angle=0, theta1=200, theta2=340,
                             color=ORANGE, lw=1.6))
            arrow(ax, (0.32, -0.42), (0.5, -0.3), color=ORANGE, lw=1.4, mut=10)
        ax.set_title(title, fontsize=11)
        area = abs(det)
        ax.text(0.5, -0.02,
                rf"area $=|\det\mathbf{{A}}|={area:.2g}$",
                transform=ax.transAxes,
                ha="center", va="top", fontsize=9.5)
        clean_axes(ax, lim=((-1.4, 2.4), (-1.4, 2.4)), hide=True)

    show_para(axes[0], np.array([[1.6, 0.6], [0.3, 1.4]]), "(a) area $>0$")
    show_para(axes[1], np.array([[0.4, 1.5], [1.4, 0.5]]),
              "(b) orientation flip", flip=True)
    show_para(axes[2], np.array([[1.4, 0.7], [2.0, 1.0]]),
              "(c) degenerate", degenerate=True)
    save(fig, "mdl-la-determinant")


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
    ax.set_xlabel(r"$\cos\theta$")
    ax.set_ylabel("density")
    ax.annotate(r"std $\approx 1/\sqrt{d}$", xy=(0.0, 0.0), xytext=(0.42, 0.78),
                textcoords="axes fraction", fontsize=10, color=GRAY)
    ax.legend(loc="upper left")
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

    specs = [
        (np.array([[2.0, 0.0], [0.0, -1.0]]), "(a) $\\mathrm{diag}(2,-1)$"),
        (np.array([[2.0, 1.0], [1.0, 2.0]]), "(b) $[[2,1],[1,2]]$"),
    ]
    for ax, (A, title) in zip(axes, specs):
        ellipse = A @ circle
        w, V = np.linalg.eigh(A)  # symmetric: real eigenpairs, V orthonormal cols
        ax.plot(circle[0], circle[1], "--", color=LIGHT, lw=1.4)
        ax.plot(ellipse[0], ellipse[1], color=BLUE, lw=2.2)
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
            lpos = tip + 0.30 * d + 0.30 * perp
            ax.text(lpos[0], lpos[1], rf"{lab}$={lam:.2g}${sign}",
                    color=GREEN if lam >= 0 else ORANGE, fontsize=9.5,
                    ha="center", va="center")
        ax.set_title(title)
        m = 2.6
        clean_axes(ax, lim=((-m, m), (-m, m)), hide=True)
        axis_cross(ax, (-m, m), (-m, m), color=GRAY, lw=0.8)
    save(fig, "mdl-la-eig-ellipse")


def fig_psd():
    """Three 3-D surface plots of z = x^T A x: bowl (PD), trough (PSD), saddle."""
    fig = plt.figure(figsize=(10.5, 3.8))
    g = np.linspace(-1.5, 1.5, 60)
    X, Y = np.meshgrid(g, g)

    specs = [
        (np.array([[2.0, 0.0], [0.0, 1.0]]), "(a) positive definite", BLUE),
        (np.array([[1.0, 0.0], [0.0, 0.0]]), "(b) positive semidefinite", GREEN),
        (np.array([[1.0, 0.0], [0.0, -1.0]]), "(c) indefinite", ORANGE),
    ]
    for i, (A, title, col) in enumerate(specs, 1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        Z = A[0, 0] * X**2 + (A[0, 1] + A[1, 0]) * X * Y + A[1, 1] * Y**2
        ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True,
                        alpha=0.92, rstride=2, cstride=2)
        ax.set_title(title, fontsize=11, pad=-2)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_xlabel("$x_1$", labelpad=-12, fontsize=9)
        ax.set_ylabel("$x_2$", labelpad=-12, fontsize=9)
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
        ax.plot(c, 0, "x", color=col, ms=7, mew=1.6)
    ax.plot(eig.real, np.zeros_like(eig.real), "o", color="black", ms=7,
            zorder=6, label="true eigenvalues")
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.set_xlabel("real axis")
    ax.set_ylabel("imag axis")
    ax.set_aspect("equal")
    ax.set_ylim(-1.3, 1.3)
    ax.set_xlim(centers.min() - radii.max() - 0.5, centers.max() + radii.max() + 0.5)
    ax.legend(loc="upper left")
    clean_axes(ax, equal=True)
    save(fig, "mdl-la-gershgorin")


def fig_power_iter():
    """(a) fan of arrows: a unit vector under repeated A-then-renormalize
    swinging onto the dominant eigenvector; (b) the ratio ||A^{k+1}v||/||A^k v||
    converging to |lambda_1|."""
    A = np.array([[2.0, 1.0], [1.0, 2.0]])  # symmetric; eigvals 3 (dominant), 1
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
    axa.set_title("(a) direction converges")
    axis_cross(axa, (-1.3, 1.3), (-1.3, 1.3), color=GRAY, lw=0.8)
    # dominant eigenvector reference ray (both directions)
    axa.plot([-1.2 * dom[0], 1.2 * dom[0]], [-1.2 * dom[1], 1.2 * dom[1]],
             color=GRAY, lw=1.2, ls="--")
    cur = v.copy()
    for k in range(n_it + 1):
        arrow(axa, (0, 0), cur, color=cmap[k], lw=2.2)
        nxt = A @ cur
        cur = nxt / np.linalg.norm(nxt)
    arrow(axa, (0, 0), dom, color=ORANGE, lw=2.6)
    axa.text(dom[0] * 1.05 + 0.05, dom[1] * 1.05, r"$\mathbf{v}_1$", color=ORANGE,
             fontsize=10, ha="left")
    axa.text(0.06, 1.14, r"$\mathbf{x}_0$", color=GRAY, fontsize=9.5, ha="left")
    clean_axes(axa, lim=((-1.3, 1.3), (-1.3, 1.3)), hide=True)

    # (b) ratio convergence
    axb.set_title("(b) ratio $\\to|\\lambda_1|$")
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
    axb.axhline(abs(lam1), color=GRAY, ls="--", lw=1.4)
    axb.text(ks[-1], abs(lam1) + 0.06, rf"$|\lambda_1|={abs(lam1):.0f}$",
             ha="right", va="bottom", color=GRAY, fontsize=10)
    axb.set_xlabel("iteration $k$")
    axb.set_ylabel(r"$\|A^{k+1}v\|/\|A^k v\|$")
    axb.annotate(rf"gap $\sim|\lambda_2/\lambda_1|^k="
                 rf"({abs(lam2/lam1):.2g})^k$",
                 xy=(0.45, 0.35), xycoords="axes fraction", fontsize=9.5,
                 color=GRAY)
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

    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.3))
    m = max(2.0, s[0] * 1.28)  # margin so axis labels never clip
    lim = ((-m, m), (-m, m))

    def tip_label(ax, tip, text, color):
        """Place a label just beyond an arrow tip, nudged perpendicular so it
        never sits under the arrowhead."""
        tip = np.asarray(tip, float)
        d = tip / np.linalg.norm(tip)
        perp = np.array([-d[1], d[0]])
        p = tip + 0.22 * d + 0.22 * perp
        ax.text(p[0], p[1], text, color=color, fontsize=9.5,
                ha="center", va="center")

    # (1) input unit circle with right singular vectors v1, v2
    ax = axes[0]
    ax.set_title("(1) input")
    ax.plot(circle[0], circle[1], color=GRAY, lw=1.6)
    arrow(ax, (0, 0), V[:, 0], color=BLUE, lw=2.4)
    arrow(ax, (0, 0), V[:, 1], color=ORANGE, lw=2.4)
    tip_label(ax, V[:, 0], r"$\mathbf{v}_1$", BLUE)
    tip_label(ax, V[:, 1], r"$\mathbf{v}_2$", ORANGE)

    # (2) after V^T: circle unchanged, v's rotated onto the axes
    ax = axes[1]
    ax.set_title(r"(2) after $\mathbf{V}^\top$")
    c2 = Vt @ circle
    ax.plot(c2[0], c2[1], color=GRAY, lw=1.6)
    e1 = Vt @ V[:, 0]
    e2 = Vt @ V[:, 1]
    arrow(ax, (0, 0), e1, color=BLUE, lw=2.4)
    arrow(ax, (0, 0), e2, color=ORANGE, lw=2.4)

    # (3) after Sigma: axis-aligned ellipse, semi-axes sigma1, sigma2
    ax = axes[2]
    ax.set_title(r"(3) after $\mathbf{\Sigma}$")
    Sig = np.diag(s)
    c3 = Sig @ Vt @ circle
    ax.plot(c3[0], c3[1], color=BLUE, lw=2.2)
    arrow(ax, (0, 0), [s[0], 0], color=BLUE, lw=2.4)
    arrow(ax, (0, 0), [0, s[1]], color=ORANGE, lw=2.4)
    ax.text(s[0] * 0.5, 0.18, rf"$\sigma_1={s[0]:.2g}$", color=BLUE, fontsize=9, ha="center")
    ax.text(0.2, s[1] * 0.5, rf"$\sigma_2={s[1]:.2g}$", color=ORANGE, fontsize=9, ha="left")

    # (4) after U: ellipse rotated; draw sigma1 u1, sigma2 u2 as its axes
    ax = axes[3]
    ax.set_title(r"(4) after $\mathbf{U}$ $=\mathbf{A}$")
    c4 = U @ Sig @ Vt @ circle
    ax.plot(c4[0], c4[1], color=BLUE, lw=2.2)
    a1 = s[0] * U[:, 0]
    a2 = s[1] * U[:, 1]
    arrow(ax, (0, 0), a1, color=BLUE, lw=2.4)
    arrow(ax, (0, 0), a2, color=ORANGE, lw=2.4)
    tip_label(ax, a1, r"$\sigma_1\mathbf{u}_1$", BLUE)
    tip_label(ax, a2, r"$\sigma_2\mathbf{u}_2$", ORANGE)

    for ax in axes:
        axis_cross(ax, (-m, m), (-m, m), color=GRAY, lw=0.7)
        clean_axes(ax, lim=lim, hide=True)
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
                ha="center", va="center", fontsize=9.5, color=top_col)
        ax.text(x + bw / 2, y0 + (sy - y0) / 2, bot_label,
                ha="center", va="center", fontsize=9.5, color=bot_col)
        return sy

    sy_dom = box(dom_x, 0.62, r"row space" + "\n" + r"$\dim r$",
                 r"null space $\mathcal{N}(\mathbf{A})$", BLUE, GRAY)
    sy_cod = box(cod_x, 0.62, r"column space" + "\n" + r"$\dim r$",
                 r"left null space" + "\n" + r"$\mathcal{N}(\mathbf{A}^\top)$", GREEN, GRAY)

    ax.text(dom_x + bw / 2, y0 + bh + 0.3, r"$\mathbb{R}^n$ (domain)", ha="center",
            fontsize=10)
    ax.text(cod_x + bw / 2, y0 + bh + 0.3, r"$\mathbb{R}^m$ (codomain)", ha="center",
            fontsize=10)

    # arrow v_i |-> sigma_i u_i (row space -> column space)
    p_row = np.array([dom_x + bw, y0 + bh * 0.78])
    p_col = np.array([cod_x, y0 + bh * 0.78])
    ax.add_patch(FancyArrowPatch(p_row, p_col, arrowstyle="->",
                                 mutation_scale=16, color=BLUE, lw=2.0))
    ax.text((p_row[0] + p_col[0]) / 2, p_row[1] + 0.18,
            r"$\mathbf{v}_i \mapsto \sigma_i\mathbf{u}_i$", ha="center",
            va="bottom", fontsize=10, color=BLUE)

    # null space collapses to 0 (the zero point sits in the gap, just left of
    # the codomain box so neither the dot nor its label touch the box edge)
    p_null = np.array([dom_x + bw, y0 + bh * 0.22])
    p_zero = np.array([cod_x - 0.45, y0 + bh * 0.20])
    ax.add_patch(FancyArrowPatch(p_null, p_zero, arrowstyle="->",
                                 mutation_scale=14, color=GRAY, lw=1.6,
                                 linestyle=(0, (4, 3))))
    ax.plot(*p_zero, "o", color="black", ms=6)
    ax.text(p_zero[0], p_zero[1] + 0.32, r"$\mathbf{0}$", ha="center",
            va="bottom", fontsize=10)
    ax.text((p_null[0] + p_zero[0]) / 2, p_null[1] - 0.26, r"$\mapsto \mathbf{0}$",
            ha="center", va="top", fontsize=9.5, color=GRAY)

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
    axs.text(20, s[0], "top-$k$", color=ORANGE, fontsize=9, ha="left", va="top")
    axs.set_xlabel("index $i$")
    axs.set_ylabel(r"$\sigma_i$ (log)")
    axs.set_title("(a) spectrum")
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
        ax.set_title(f"{ttl}\nerr {rel:.3f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    save(fig, "mdl-la-eckart-young")


def fig_condition():
    """Two contour plots of f(x)=1/2 x^T A x: well-conditioned (near-circular,
    short GD path) and ill-conditioned (elongated valley, zig-zag GD path).
    Real GD steps draw the paths."""
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.0, 4.2))

    g = np.linspace(-1.6, 1.6, 200)
    X, Y = np.meshgrid(g, g)

    def run(ax, A, title, eta, x0, steps):
        Z = 0.5 * (A[0, 0] * X**2 + (A[0, 1] + A[1, 0]) * X * Y + A[1, 1] * Y**2)
        ax.contour(X, Y, Z, levels=14, colors=[LIGHT], linewidths=0.9)
        path = [np.array(x0, float)]
        x = np.array(x0, float)
        for _ in range(steps):
            grad = A @ x  # grad of 1/2 x^T A x  (A symmetric)
            x = x - eta * grad
            path.append(x.copy())
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], "-o", color=ORANGE, ms=3.5, lw=1.6)
        ax.plot(0, 0, "*", color=GREEN, ms=12)
        ax.set_title(title)
        clean_axes(ax, lim=((-1.6, 1.6), (-1.6, 1.6)), hide=True)

    # well-conditioned: near-isotropic; step size safe -> nearly straight path
    run(axa, np.array([[1.0, 0.0], [0.0, 1.3]]), "(a) well-conditioned",
        eta=0.55, x0=(-1.3, 1.1), steps=10)
    # ill-conditioned: elongated valley; eta near stability bound -> zig-zag
    run(axb, np.array([[1.0, 0.0], [0.0, 18.0]]), "(b) ill-conditioned",
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
        ax.text(tip[0] + 0.08 * d[0], tip[1] + 0.08 * d[1] + 0.05, lab,
                color=col, fontsize=11, fontweight="bold",
                ha="center", va="center")
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    clean_axes(ax, equal=True)
    save(fig, "mdl-la-pca")


# =========================================================================== #
# Driver                                                                      #
# =========================================================================== #

FIGURES = [
    # geometry
    fig_vectors,
    fig_vector_add,
    fig_angle,
    fig_projection,
    fig_hyperplane,
    fig_linear_map,
    fig_determinant,
    fig_cosine_highd,
    # eigendecomposition
    fig_eig_ellipse,
    fig_psd,
    fig_gershgorin,
    fig_power_iter,
    # svd
    fig_svd_action,
    fig_svd_subspaces,
    fig_eckart_young,
    fig_condition,
    fig_pca,
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
