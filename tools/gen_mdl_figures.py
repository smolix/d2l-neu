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
# ONE shared style, applied once — now the unified token layer                #
# (tools/figstyle/tokens.py; see docs/figure-style-guide.md §9 step 2).       #
# The legacy names below keep every consumer script working unchanged.       #
# --------------------------------------------------------------------------- #

import figstyle.mpl as _figstyle
from figstyle import tokens as T

BLUE = _figstyle.BLUE
ORANGE = _figstyle.ORANGE
GREEN = _figstyle.GREEN
GRAY = _figstyle.GRAY
LIGHT = _figstyle.LIGHT   # faint grids / construction lines

# token-blue sequential ramp for surfaces / iterate fans (replaces viridis)
from matplotlib.colors import LinearSegmentedColormap as _LSC
BLUE_CMAP = _LSC.from_list("figblue", [T.BLUE.tint, T.BLUE.base, T.BLUE.dark])
INK = _figstyle.INK

_figstyle.use_style(hashsalt="mdl-la")

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
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        linestyle=ls, alpha=alpha,
                        shrinkA=0, shrinkB=0, mutation_scale=mut),
    )


def vlabel(ax, pos, text, color=INK, **kw):
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


def axis_cross(ax, xr, yr, color=INK, lw=0.9):
    """Draw thin black x/y axes through the origin for a schematic plane."""
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

    LBL = 15                          # one size + font (mathtext cm) for every label
    LIM = ((-1.0, 4.5), (-1.3, 3.3))  # identical box for both panels -> equal size
    XR, YR = (-0.9, 4.3), (-1.2, 3.2)

    # (a) point with dashed coordinate drops; black axes for readability
    axis_cross(axa, XR, YR)
    p = np.array([3.0, 2.0])
    axa.plot(*p, "o", color=BLUE, ms=8, zorder=5)
    axa.plot([p[0], p[0]], [0, p[1]], "--", color=GRAY, lw=1.2)
    axa.plot([0, p[0]], [p[1], p[1]], "--", color=GRAY, lw=1.2)
    axa.text(p[0], -0.28, r"$3$", color=BLUE, ha="center", va="top", fontsize=LBL)
    axa.text(-0.22, p[1], r"$2$", color=BLUE, ha="right", va="center", fontsize=LBL)
    vlabel(axa, (p[0] + 0.15, p[1] + 0.22), r"$(3,2)$", color=BLUE, ha="left",
           fontsize=LBL)
    clean_axes(axa, lim=LIM, hide=True)

    # (b) same vector as a translation-invariant arrow in three places -- one
    # color, since every arrow *is* the same vector.  Bases chosen so all three
    # copies fit the shared box and stay clear of one another.
    axis_cross(axb, XR, YR)
    v = np.array([3.0, 2.0])
    # the third base is nudged right off the line through the other two, so the
    # arrows do not all line up
    bases = [np.array([0.0, 0.0]), np.array([1.0, -1.1]), np.array([-0.3, 1.1])]
    for b in bases:
        arrow(axb, b, b + v, color=BLUE, lw=2.2)
    # label just right of the tip of the copy drawn from the origin, in the
    # gap between the other two copies' arrowheads
    vlabel(axb, (3.14, 1.95), r"$(3,2)$", color=BLUE, ha="left", va="center",
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
    axis_cross(ax, (-0.4, 4.6), (-0.35, 3.15))
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
    axis_cross(axa, XR, YR)
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
    axa.text(lpos[0], lpos[1], r"$\mathrm{span}(\mathbf{v})$", color=INK,
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
    axis_cross(axb, XR, YR)
    # dashed parallelogram construction: u -> 2u (closing the bottom side),
    # 2u -> x and w -> x
    axb.plot([u[0], 2 * u[0]], [u[1], 2 * u[1]], "--", color=GRAY, lw=1.4)
    axb.plot([2 * u[0], x[0]], [2 * u[1], x[1]], "--", color=GRAY, lw=1.4)
    axb.plot([w[0], x[0]], [w[1], x[1]], "--", color=GRAY, lw=1.4)
    arrow(axb, (0, 0), u, color=BLUE, lw=2.4)
    arrow(axb, (0, 0), w, color=ORANGE, lw=2.4)
    arrow(axb, (0, 0), x, color=GREEN, lw=2.6)
    # below-right of the tip: between the u-line and the x-axis, off both
    vlabel(axb, (u[0] + 0.22, u[1] - 0.26), r"$\mathbf{u}$", color=BLUE,
           ha="left", fontsize=LBL)
    # above-left of the tip, between the y-axis and the line through w
    vlabel(axb, (0.40, w[1] + 0.36), r"$\mathbf{w}$", color=ORANGE,
           ha="center", fontsize=LBL)
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
    axis_cross(ax, (-0.4, 3.8), (-0.5, 2.9))
    arrow(ax, (0, 0), v, color=BLUE, lw=2.4)
    arrow(ax, (0, 0), w, color=ORANGE, lw=2.4)
    a1 = np.degrees(np.arctan2(v[1], v[0]))
    a2 = np.degrees(np.arctan2(w[1], w[0]))
    ax.add_patch(Arc((0, 0), 1.7, 1.7, angle=0, theta1=a1, theta2=a2,
                     color=GRAY, lw=1.6))
    mid = np.radians((a1 + a2) / 2)
    vlabel(ax, (1.18 * np.cos(mid), 1.18 * np.sin(mid)), r"$\theta$", color=INK,
           fontsize=18)
    vlabel(ax, (v[0], -0.34), r"$\mathbf{v}$", color=BLUE, fontsize=LBL)
    vlabel(ax, (w[0] - 0.05, w[1] + 0.26), r"$\mathbf{w}$", color=ORANGE, fontsize=LBL)
    clean_axes(ax, lim=LIM, hide=True)
    save(fig, "mdl-la-angle")


def fig_projection():
    """(a) the worked projection of v=(1,3) onto w=(2,1): shadow (2,1),
    residual (-1,2) at a right angle, theta = 45 deg;
    (b) the Cauchy-Schwarz equality case, v = 1.5 w collinear with w.

    Both panels share one coordinate box, so they render at the same size; the
    caption ("(a) ... (b) ...") names them, so we draw no per-panel titles.
    """
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(8.4, 3.9))

    LBL = 15                            # vector labels
    FML = 14                            # longer formula labels
    LIM = ((-0.45, 3.9), (-0.35, 3.35))  # identical box for both panels
    XR, YR = (-0.35, 3.7), (-0.3, 3.25)

    # (a) worked example
    w = np.array([2.0, 1.0])
    v = np.array([1.0, 3.0])
    wn = w / np.linalg.norm(w)
    proj = (v @ w) / (w @ w) * w        # = (2, 1)
    axis_cross(axa, XR, YR)
    # the shadow coincides with w: a thick green band under the orange arrow
    axa.plot([0, proj[0]], [0, proj[1]], color=GREEN, lw=7,
             solid_capstyle="butt", zorder=2)
    arrow(axa, (0, 0), w, color=ORANGE, lw=2.2)
    arrow(axa, (0, 0), v, color=BLUE, lw=2.4)
    # residual r = v - proj (dashed) meeting w at a right angle
    axa.plot([proj[0], v[0]], [proj[1], v[1]], "--", color=INK, lw=1.5)
    right_angle(axa, proj, -wn, (v - proj), size=0.27, color=INK)
    vlabel(axa, (w[0] + 0.12, w[1] + 0.02), r"$\mathbf{w}=(2,1)$",
           color=ORANGE, ha="left", fontsize=LBL)
    vlabel(axa, (w[0] + 0.12, w[1] - 0.40),
           r"$=\mathrm{proj}_{\mathbf{w}}\mathbf{v}$", color=GREEN, ha="left",
           fontsize=FML)
    vlabel(axa, (v[0] + 0.10, v[1] + 0.10), r"$\mathbf{v}=(1,3)$",
           color=BLUE, ha="left", va="bottom", fontsize=LBL)
    # residual label: outward from the residual's midpoint, along +w
    rm = 0.5 * (proj + v) + 0.22 * wn
    vlabel(axa, rm, r"$\mathbf{r}=(-1,2)$", color=INK, ha="left",
           fontsize=LBL)
    # the angle theta between w and v, at the origin
    aw = np.degrees(np.arctan2(w[1], w[0]))
    av = np.degrees(np.arctan2(v[1], v[0]))
    axa.add_patch(Arc((0, 0), 1.1, 1.1, angle=0, theta1=aw, theta2=av,
                      color=INK, lw=1.2, zorder=5))
    am = np.radians(0.5 * (aw + av))
    vlabel(axa, (0.78 * np.cos(am), 0.78 * np.sin(am)), r"$\theta$",
           color=INK, fontsize=FML)
    clean_axes(axa, lim=LIM, hide=True)

    # (b) equality: v = 1.5 w collinear with w, residual vanishes
    v2 = 1.5 * w
    nrm = np.array([-wn[1], wn[0]])     # unit normal to the common line
    axis_cross(axb, XR, YR)
    arrow(axb, (0, 0), v2, color=BLUE, lw=2.4)
    arrow(axb, (0, 0), w, color=ORANGE, lw=2.4)
    vlabel(axb, 0.5 * w - 0.30 * nrm, r"$\mathbf{w}$", color=ORANGE,
           fontsize=LBL)
    vlabel(axb, 0.5 * (w + v2) + 0.55 * nrm - 0.25 * wn, r"$\mathbf{v}=1.5\,\mathbf{w}$",
           color=BLUE, fontsize=LBL)
    axb.text(1.55, 2.75,
             r"$|\mathbf{v}\cdot\mathbf{w}| = \|\mathbf{v}\|\,\|\mathbf{w}\|$",
             ha="center", va="center", fontsize=FML)
    clean_axes(axb, lim=LIM, hide=True)

    save(fig, "mdl-la-projection")


def fig_hyperplane():
    """Left: the line w.x + b = 0 for w=(2,1), b=-1 (2x+y=1), its normal w,
    the shaded half-plane w.x+b>0, the distance |b|/||w|| from the origin,
    and the point (1,1) at signed distance 2/sqrt(5) with its foot (0.2,0.6).
    Right: the plane analog inside matplotlib's 3-D box frame."""
    fig = plt.figure(figsize=(9.0, 3.8))
    axa = fig.add_subplot(1, 2, 1)
    axb = fig.add_subplot(1, 2, 2, projection="3d")

    LBL = 15                                   # vector letters
    FML = 14                                   # formula labels

    # --- 2-D panel ---
    w = np.array([2.0, 1.0])
    b = -1.0
    nrm = np.linalg.norm(w)
    wn = w / nrm
    perp = np.array([-wn[1], wn[0]])           # along the line
    lim = ((-1.15, 2.75), (-0.75, 2.0))

    foot = (-b / nrm) * wn                      # (0.4, 0.2)
    L = 6.0
    seg = np.array([foot - L * perp, foot + L * perp])
    # shade the half-space w.x + b > 0 (clipped by the window)
    far = wn * L
    poly = np.array([seg[0], seg[1], seg[1] + far, seg[0] + far])
    axa.add_patch(Polygon(poly, closed=True, color=BLUE, alpha=0.10, lw=0))
    axa.plot(seg[:, 0], seg[:, 1], "--", color=BLUE, lw=1.8)
    # line label, upright, on the unshaded side of the upper-left stretch
    text_dir = perp if perp[0] >= 0 else -perp
    rot = np.degrees(np.arctan2(text_dir[1], text_dir[0]))
    lp = foot - 1.45 * text_dir - 0.56 * wn
    axa.text(lp[0], lp[1], r"$\mathbf{w}\cdot\mathbf{x}+b=0$", color=BLUE,
             fontsize=FML, ha="center", va="center", rotation=rot,
             rotation_mode="anchor")

    axis_cross(axa, (-1.1, 2.7), (-0.7, 1.95))
    # distance from the origin to the line, along w (under the w arrow)
    axa.plot([0, foot[0]], [0, foot[1]], color=GREEN, lw=6,
             solid_capstyle="butt", zorder=3)
    arrow(axa, (0, 0), w, color=ORANGE, lw=2.2)
    vlabel(axa, (w[0] + 0.08, w[1] - 0.16), r"$\mathbf{w}$", color=ORANGE,
           ha="left", fontsize=LBL)
    axa.text(-0.08, 0.07, r"$|b|/\|\mathbf{w}\|$", color=GREEN,
             fontsize=FML, ha="right", va="bottom")

    # the point (1, 1) and its perpendicular to the line
    x = np.array([1.0, 1.0])
    score = w @ x + b
    xc = x - score / (w @ w) * w                # (0.2, 0.6)
    axa.plot([xc[0], x[0]], [xc[1], x[1]], ":", color=INK, lw=1.6, zorder=4)
    right_angle(axa, xc, perp, wn, size=0.11, color=INK)
    axa.plot([x[0]], [x[1]], "o", color=INK, ms=5, zorder=6)
    vlabel(axa, (x[0] + 0.08, x[1] + 0.12), r"$(1,1)$", color=INK,
           ha="left", fontsize=FML)
    mid = 0.5 * (x + xc) + 0.32 * perp
    vlabel(axa, mid, r"$2/\sqrt{5}$", color=INK, fontsize=FML)
    axa.text(1.95, 1.55, r"$\mathbf{w}\cdot\mathbf{x}+b>0$", color=BLUE,
             fontsize=FML, ha="center", va="center")
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
    # the normal w stands on the plane, on its visible (upper) side: it
    # starts at the foot of the perpendicular from the origin and is drawn
    # after the surface (no depth sort), so its head is never behind it
    axb.computed_zorder = False
    fw = (b3 / (w3 @ w3)) * w3
    wdir = 0.75 * w3 / np.linalg.norm(w3)
    axb.quiver(fw[0], fw[1], fw[2], wdir[0], wdir[1], wdir[2], color=ORANGE,
               lw=2.4, arrow_length_ratio=0.22)
    axb.scatter([fw[0]], [fw[1]], [fw[2]], color=ORANGE, s=18,
                depthshade=False)
    lwp = fw + wdir + np.array([0.10, 0.0, 0.06])
    axb.text(lwp[0], lwp[1], lwp[2], r"$\mathbf{w}$", color=ORANGE,
             fontsize=LBL)
    axb.set_xlim(0, 1.9); axb.set_ylim(0, 1.9); axb.set_zlim(-0.15, 1.9)
    axb.set_xticks([]); axb.set_yticks([]); axb.set_zticks([])
    for axis in (axb.xaxis, axb.yaxis, axb.zaxis):
        axis.line.set_color(INK)          # blacken the axis lines only
    axb.view_init(elev=26, azim=-54)
    try:
        axb.set_box_aspect((1, 1, 1), zoom=1.1)
    except Exception:
        pass

    save(fig, "mdl-la-hyperplane")


def fig_linear_map():
    """The chapter's running matrix A = [[1, 2], [-1, 3]] acting on the plane.
    (a) the input grid x in {0,1,2}, y in {-1,...,2}, the shaded unit square,
    e1, e2 and the running input v = (2, -1), a grid point; (b) the image grid
    under A, the shaded parallelogram (0,0), (1,-1), (3,2), (2,3), the columns
    A e1 = (1,-1), A e2 = (2,3) and A v = (0,-5).  Panel (b) is drawn at one
    third of (a)'s scale (window spans 3.2 x 4.2 against 9.6 x 12.6), which
    the caption states; both panels share the same aspect, hence the same
    height.  All images are computed by applying A."""
    A = np.array([[1.0, 2.0], [-1.0, 3.0]])
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(8.4, 5.4))
    fig.subplots_adjust(wspace=0.08)

    LBL = 16
    xs, ys = (0, 2), (-1, 2)
    ts_x = np.linspace(*xs, 50)
    ts_y = np.linspace(*ys, 50)

    def draw_grid(ax, M):
        for k in range(xs[0], xs[1] + 1):        # images of the lines x = k
            pts = M @ np.vstack([np.full_like(ts_y, k), ts_y])
            ax.plot(pts[0], pts[1], color=T.FAINT, lw=1.0, zorder=1)
        for k in range(ys[0], ys[1] + 1):        # images of the lines y = k
            pts = M @ np.vstack([ts_x, np.full_like(ts_x, k)])
            ax.plot(pts[0], pts[1], color=T.FAINT, lw=1.0, zorder=1)

    unit = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    v = np.array([2.0, -1.0])
    lima = ((-0.6, 2.6), (-1.6, 2.6))
    limb = ((-2.6, 7.0), (-5.9, 6.7))

    # (a) the input plane
    axis_cross(axa, (lima[0][0], lima[0][1]), (lima[1][0], lima[1][1]),
               color=INK)
    draw_grid(axa, np.eye(2))
    axa.add_patch(Polygon(unit, closed=True, facecolor=GREEN, alpha=0.28,
                          lw=0, zorder=0))
    arrow(axa, (0, 0), (1, 0), color=BLUE, lw=2.6)
    arrow(axa, (0, 0), (0, 1), color=ORANGE, lw=2.6)
    arrow(axa, (0, 0), v, color=INK, lw=2.2)
    axa.plot(0, 0, "o", color=INK, ms=5, zorder=6)
    vlabel(axa, (0.55, 0.24), r"$\mathbf{e}_1$", color=BLUE, fontsize=LBL)
    vlabel(axa, (-0.30, 0.72), r"$\mathbf{e}_2$", color=ORANGE, fontsize=LBL)
    vlabel(axa, (v[0] + 0.20, v[1] - 0.24), r"$\mathbf{v}$", color=INK,
           fontsize=LBL)
    clean_axes(axa, lim=lima, hide=True)

    # (b) the image under A, at one third of (a)'s scale
    axis_cross(axb, (limb[0][0], limb[0][1]), (limb[1][0], limb[1][1]),
               color=INK)
    draw_grid(axb, A)
    axb.add_patch(Polygon((A @ unit.T).T, closed=True, facecolor=GREEN,
                          alpha=0.28, lw=0, zorder=0))
    a1, a2, Av = A @ np.array([1.0, 0.0]), A @ np.array([0.0, 1.0]), A @ v
    arrow(axb, (0, 0), a1, color=BLUE, lw=2.6)
    arrow(axb, (0, 0), a2, color=ORANGE, lw=2.6)
    arrow(axb, (0, 0), Av, color=INK, lw=2.2)
    axb.plot(0, 0, "o", color=INK, ms=5, zorder=6)
    # inside the image of the cell [1, 2] x [-1, 0], clear of grid lines
    vlabel(axb, (1.0, -2.25), r"$\mathbf{A}\mathbf{e}_1$", color=BLUE,
           fontsize=LBL)
    vlabel(axb, (a2[0] - 0.55, a2[1] + 0.95), r"$\mathbf{A}\mathbf{e}_2$",
           color=ORANGE, fontsize=LBL, ha="right",
           bbox=dict(fc="white", ec="none", pad=0.5))
    vlabel(axb, (Av[0] + 0.45, Av[1] + 0.1), r"$\mathbf{A}\mathbf{v}$",
           color=INK, fontsize=LBL, ha="left")
    clean_axes(axb, lim=limb, hide=True)

    save(fig, "mdl-la-linear-map")


def fig_determinant():
    """The determinant as signed area for the section's three matrices:
    (a) A = [[1, 2], [-1, 3]], det 5, counter-clockwise; (b) A with its
    columns swapped, [[2, 1], [3, -1]], det -5: the same parallelogram with
    the orientation reversed; (c) B = [[2, 1], [4, 2]], det 0: the unit
    square collapses to the segment from 0 to (3, 6).  Panel (c) is drawn at
    half the scale of (a) and (b) (window spans 8.8 x 10.4 against
    4.4 x 5.2), stated in the caption.  The panel texts are functional (the
    values are not in the caption)."""
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 4.3))
    fig.subplots_adjust(wspace=0.06)
    unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float).T
    lim_ab = ((-0.7, 3.7), (-1.5, 3.7))
    lim_c = ((-1.4, 7.4), (-3.0, 7.4))

    def show(ax, M, tag, lim, degenerate=False):
        det = float(np.round(np.linalg.det(M), 9))
        img = M @ unit
        axis_cross(ax, lim[0], lim[1], color=INK)
        ax.add_patch(Polygon(unit.T, closed=True, fill=False, edgecolor=INK,
                             lw=1.0, ls="--", zorder=2))
        col = GREEN if det > 0 else ORANGE
        c1, c2 = M[:, 0], M[:, 1]
        if degenerate:
            ax.plot([0, 3], [0, 6], color=GRAY, lw=8, alpha=0.40,
                    solid_capstyle="round", zorder=1)
        else:
            ax.add_patch(Polygon(img.T, closed=True, facecolor=col,
                                 alpha=0.25, edgecolor=col, lw=2, zorder=1))
            rad = 0.45 if det > 0 else -0.45
            ax.add_patch(FancyArrowPatch(0.42 * c1, 0.42 * c2,
                         connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                         mutation_scale=14, color=col, lw=1.8, zorder=6))
        # longer column first, so the shorter one is drawn on top
        order = [(c1, BLUE), (c2, ORANGE)]
        order.sort(key=lambda t: -np.linalg.norm(t[0]))
        for c, cc in order:
            arrow(ax, (0, 0), c, color=cc, lw=2.4)
        ax.plot(0, 0, "o", color=INK, ms=4.5, zorder=7)
        ax.text(0.98, 0.03, tag, transform=ax.transAxes, ha="right",
                va="bottom",
                fontsize=15, color=INK)
        clean_axes(ax, lim=lim, hide=True)

    show(axes[0], np.array([[1.0, 2.0], [-1.0, 3.0]]),
         r"(a) $\det\mathbf{A}=5$", lim_ab)
    show(axes[1], np.array([[2.0, 1.0], [3.0, -1.0]]),
         r"(b) $\det=-5$", lim_ab)
    show(axes[2], np.array([[2.0, 1.0], [4.0, 2.0]]),
         r"(c) $\det\mathbf{B}=0$", lim_c, degenerate=True)
    save(fig, "mdl-la-determinant")


def fig_null_collapse():
    """B = [[2, 1], [4, 2]] collapsing the plane.  (a) the input plane: light
    grid, black axes, the null-space line span{(1, -2)} dashed, the row
    direction (2, 1) as a short grey arrow with a right-angle mark against the
    null line, and the marked inputs x1 = (1, 1), x2 = (0, 3) (same image) and
    x3 = (1, -2) (on the null line).  (b) the image plane: the column-space
    line y = 2x as a grey halo, the columns b1 = (2, 4) and b2 = (1, 2), the
    shared image (3, 6) and B x3 = 0 at the origin.  Both windows span 5 x 8,
    so the two panels share one scale.  All images computed by applying B."""
    B = np.array([[2.0, 1.0], [4.0, 2.0]])
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(8.0, 6.4))
    fig.subplots_adjust(wspace=0.10)
    LBL, ANN = 15, 14
    lima = ((-2.5, 2.5), (-3.0, 5.0))
    limb = ((-1.0, 4.0), (-1.0, 7.0))
    x1, x2, x3 = np.array([1.0, 1.0]), np.array([0.0, 3.0]), np.array([1.0, -2.0])

    # --- (a) input plane ---
    for k in range(-2, 3):
        axa.plot([k, k], [-3, 4.6], color=LIGHT, lw=0.9, zorder=0)
    for k in range(-3, 5):
        axa.plot([-2.3, 2.3], [k, k], color=LIGHT, lw=0.9, zorder=0)
    axis_cross(axa, lima[0], lima[1], color=INK)
    t = np.array([-1.45, 1.45])
    axa.plot(t, -2 * t, "--", color=INK, lw=1.6, zorder=2)
    axa.text(-2.40, 3.45, "null space", color=INK, fontsize=ANN, ha="left",
             va="center", bbox=dict(fc="white", ec="none", pad=1.0))
    # a row of B, (2, 1), is perpendicular to the null line
    r = np.array([2.0, 1.0]) / np.sqrt(5.0) * 1.25
    arrow(axa, (0, 0), r, color=GRAY, lw=1.8, mut=12)
    right_angle(axa, (0, 0), r, np.array([-1.0, 2.0]), size=0.28,
                color=GRAY, lw=1.2)
    axa.text(r[0] + 0.12, r[1] - 0.05, "row", color=GRAY, fontsize=ANN,
             ha="left", va="center", bbox=dict(fc="white", ec="none", pad=0.5))
    for p, m, c, lab, off in [
            (x1, "o", BLUE, r"$\mathbf{x}_1$", (0.38, 0.18)),
            (x2, "s", ORANGE, r"$\mathbf{x}_2$", (0.42, 0.0)),
            (x3, "^", GREEN, r"$\mathbf{x}_3$", (0.42, 0.0))]:
        axa.plot(*p, m, color=c, ms=9, zorder=5)
        axa.text(p[0] + off[0], p[1] + off[1], lab, color=c, fontsize=LBL,
                 ha="center", va="center",
                 bbox=dict(fc="white", ec="none", pad=0.5), zorder=6)
    axa.plot(0, 0, "o", color=INK, ms=4.5, zorder=7)
    clean_axes(axa, lim=lima, hide=True)

    # --- (b) image plane: everything lands on the column space y = 2x ---
    axis_cross(axb, limb[0], limb[1], color=INK)
    axb.plot([-0.45, 3.35], [-0.9, 6.7], color=GRAY, lw=7, alpha=0.40,
             solid_capstyle="round", zorder=1)
    rot = np.degrees(np.arctan2(2.0, 1.0))
    nrm = np.array([2.0, -1.0]) / np.sqrt(5.0)
    cx, cy = np.array([2.35, 4.7]) + 0.42 * nrm
    axb.text(cx, cy, "column space", color=INK, fontsize=ANN, ha="center",
             va="center", rotation=rot, rotation_mode="anchor")
    b1, b2 = B[:, 0], B[:, 1]
    arrow(axb, (0, 0), b1, color=BLUE, lw=2.4)
    arrow(axb, (0, 0), b2, color=ORANGE, lw=2.6)
    vlabel(axb, (b1[0] - 0.42, b1[1] + 0.05), r"$\mathbf{b}_1$", color=BLUE,
           fontsize=LBL)
    vlabel(axb, (b2[0] + 0.42, b2[1] - 0.25), r"$\mathbf{b}_2$",
           color=ORANGE, fontsize=LBL)
    q = B @ x1
    axb.plot(*q, "o", color=BLUE, ms=11, zorder=5)
    axb.plot(*q, "s", color=ORANGE, ms=5.5, zorder=6)
    axb.text(q[0] - 0.25, q[1] + 0.05,
             r"$\mathbf{B}\mathbf{x}_1=\mathbf{B}\mathbf{x}_2$", color=INK,
             fontsize=LBL, ha="right", va="center",
             bbox=dict(fc="white", ec="none", pad=0.5))
    axb.plot(0, 0, "^", color=GREEN, ms=9, zorder=5)
    axb.text(0.30, -0.55, r"$\mathbf{B}\mathbf{x}_3=\mathbf{0}$",
             color=GREEN, fontsize=LBL, ha="left", va="center",
             bbox=dict(fc="white", ec="none", pad=0.5))
    clean_axes(axb, lim=limb, hide=True)

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
    peaks = []
    for d, c in zip(dims, cols):
        a = rng.standard_normal((n_pairs, d))
        b = rng.standard_normal((n_pairs, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        b /= np.linalg.norm(b, axis=1, keepdims=True)
        cos = np.sum(a * b, axis=1)
        h, _, _ = ax.hist(cos, bins=bins, density=True, histtype="step",
                          color=c, lw=2.0, label=rf"$d={d}$")
        peaks.append(h.max())
    # cap the y-range so the d=2 arcsine and the d=10 bell are not squeezed
    # into the bottom tenth by the d=1000 spike; the clipped spike's true
    # height is stated at the top edge
    YMAX = 3.6
    ax.set_ylim(0, YMAX)
    ax.text(0.11, YMAX * 0.95,
            rf"$\uparrow\ \mathrm{{peak}}\approx{peaks[-1]:.0f}$",
            fontsize=15, color=GREEN, ha="left", va="center")
    ax.axvline(0, color=GRAY, lw=1.0, ls="--")
    # one font (CM math) and size for every text label, so density, cos(theta),
    # and the std annotation all match
    LAB = 15
    ax.set_xlabel(r"$\cos\theta$", fontsize=LAB)
    ax.set_ylabel(r"$\mathrm{density}$", fontsize=LAB)
    # annotation moved right into the open upper-right, clear of the central
    # spike and the dashed axis
    ax.annotate(r"$\mathrm{std}\approx 1/\sqrt{d}$", xy=(0.0, 0.0),
                xytext=(0.62, 0.62), textcoords="axes fraction",
                fontsize=LAB, color=INK)
    # just right of the d=2 edge spike at cos = -1
    ax.legend(loc="upper left", bbox_to_anchor=(0.06, 1.0), fontsize=13)
    ax.set_aspect("auto")
    clean_axes(ax, equal=False)
    save(fig, "mdl-la-cosine-highd")


# =========================================================================== #
# EIGENDECOMPOSITION (Section 1.2)                                            #
# =========================================================================== #

def fig_eig_ellipse():
    """Unit circle -> ellipse under (left) the symmetric S = [[2,1],[1,2]] and
    (right) the upper-triangular T = [[1,1],[0,2]].  Green: unit eigenvectors;
    red: their images lambda v.  For S the eigenvectors are the ellipse axes;
    for T they are not (the ellipse axes are drawn thin and grey)."""
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2))
    t = np.linspace(0, 2 * np.pi, 400)
    circle = np.vstack([np.cos(t), np.sin(t)])
    LBL = 15
    m = 2.75

    def label(ax, pos, text):
        ax.text(pos[0], pos[1], text, color=T.RED.dark, fontsize=LBL,
                ha="center", va="center")

    # ---- left: symmetric S, eigenvectors = ellipse axes
    ax = axes[0]
    S = np.array([[2.0, 1.0], [1.0, 2.0]])
    ell = S @ circle
    ax.plot(circle[0], circle[1], "--", color=INK, lw=1.0, alpha=0.5)
    ax.plot(ell[0], ell[1], color=BLUE, lw=2.4)
    for lam, vec, lpos in [(3.0, np.array([1, 1]) / np.sqrt(2), (2.2, 2.55)),
                           (1.0, np.array([1, -1]) / np.sqrt(2), (1.15, -1.05))]:
        arrow(ax, (0, 0), lam * vec, color=T.RED.base, lw=2.6)
        arrow(ax, (0, 0), vec, color=GREEN, lw=2.4)
        label(ax, lpos, rf"$\lambda={lam:.0f}$")
    clean_axes(ax, lim=((-m, m), (-m, m)), hide=True)
    axis_cross(ax, (-m, m), (-m, m), color="black")

    # ---- right: upper-triangular T, eigenvectors are not the ellipse axes
    ax = axes[1]
    Tm = np.array([[1.0, 1.0], [0.0, 2.0]])
    ell = Tm @ circle
    U, s, _ = np.linalg.svd(Tm)            # ellipse axes: s_i * U[:, i]
    ax.plot(circle[0], circle[1], "--", color=INK, lw=1.0, alpha=0.5)
    for i in range(2):
        a = s[i] * U[:, i]
        ax.plot([-a[0], a[0]], [-a[1], a[1]], color=GRAY, lw=1.2, zorder=1)
    ax.plot(ell[0], ell[1], color=BLUE, lw=2.4)
    for lam, vec, lpos in [(1.0, np.array([1.0, 0.0]), (1.62, -0.45)),
                           (2.0, np.array([1, 1]) / np.sqrt(2), (2.05, 1.12))]:
        arrow(ax, (0, 0), lam * vec, color=T.RED.base, lw=2.6)
        arrow(ax, (0, 0), vec, color=GREEN, lw=2.4)
        label(ax, lpos, rf"$\lambda={lam:.0f}$")
    clean_axes(ax, lim=((-m, m), (-m, m)), hide=True)
    axis_cross(ax, (-m, m), (-m, m), color="black")
    save(fig, "mdl-la-eig-ellipse")
    # the slides read @fig figures from img/auto
    import shutil
    shutil.copyfile(os.path.join(IMG_DIR, "mdl-la-eig-ellipse.svg"),
                    os.path.join(IMG_DIR, "auto", "mdl-la-eig-ellipse.svg"))


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
    GRIDC = GRAY

    def draw_grid(ax, M):
        for k in grid:  # images of the lines x=k
            pts = M @ np.vstack([np.full_like(ts, k), ts])
            ax.plot(pts[0], pts[1], color=GRIDC, lw=0.8)
        for k in grid:  # images of the lines y=k
            pts = M @ np.vstack([ts, np.full_like(ts, k)])
            ax.plot(pts[0], pts[1], color=GRIDC, lw=0.8)

    unit = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    lima = ((-0.8, 3.8), (-0.85, 3.15))  # span 4.6 x 4.0
    limb = ((-0.7, 6.7), (-0.85, 3.15))  # span 7.4 x 4.0 -> same scale + height

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
    vlabel(axb, (0.80, 1.22), r"$\mathbf{A}\mathbf{e}_2$", color=ORANGE,
           ha="right", fontsize=LBL)
    axb.text(4.6, 0.30, r"eigenspace $\lambda=1$", color=GREEN,
             fontsize=13, ha="center", va="center")
    clean_axes(axb, lim=limb, hide=True)

    save(fig, "mdl-la-defective-shear")


def fig_psd():
    """Three 3-D surface plots of z = x^T A x: bowl (PD), trough (PSD), saddle.
    Per-panel titles give the eigenvalues (absent from the caption)."""
    fig = plt.figure(figsize=(10.5, 4.1))
    g = np.linspace(-1.5, 1.5, 60)
    X, Y = np.meshgrid(g, g)

    specs = [
        (np.array([[2.0, 0.0], [0.0, 1.0]]), r"$\lambda = 2,\ 1$"),
        (np.array([[1.0, 0.0], [0.0, 0.0]]), r"$\lambda = 1,\ 0$"),
        (np.array([[1.0, 0.0], [0.0, -1.0]]), r"$\lambda = 1,\ {-1}$"),
    ]
    for i, (A, title) in enumerate(specs, 1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        Z = A[0, 0] * X**2 + (A[0, 1] + A[1, 0]) * X * Y + A[1, 1] * Y**2
        # Alex 2026-08-04: the original viridis reads better on 3-D relief
        ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True,
                        alpha=0.92, rstride=2, cstride=2)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_xlabel("$x_1$", labelpad=-10, fontsize=17)
        ax.set_ylabel("$x_2$", labelpad=-10, fontsize=17)
        ax.set_title(title, fontsize=16, pad=0, color="black")
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.line.set_color("black")
        ax.view_init(elev=24, azim=-52)
        try:
            ax.set_box_aspect((1, 1, 0.8))
        except Exception:
            pass
    save(fig, "mdl-la-psd")
    # the slides read @fig figures from img/auto
    import shutil
    shutil.copyfile(os.path.join(IMG_DIR, "mdl-la-psd.svg"),
                    os.path.join(IMG_DIR, "auto", "mdl-la-psd.svg"))


def fig_gershgorin():
    """Gershgorin discs of the non-symmetric 3x3 [[4,1,0],[-1,4,.5],[.5,0,1]]
    with its eigenvalues: a complex pair inside the two (nested) discs centred
    at 4 and a real eigenvalue inside the disjoint disc centred at 1."""
    A = np.array([[4.0, 1.0, 0.0],
                  [-1.0, 4.0, 0.5],
                  [0.5, 0.0, 1.0]])
    centers = np.diag(A)
    radii = np.sum(np.abs(A), axis=1) - np.abs(centers)
    eig = np.linalg.eigvals(A)

    from matplotlib.colors import to_rgba
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    cols = [BLUE, ORANGE, GREEN]
    ax.axhline(0, color="black", lw=0.9, zorder=1)
    for c, r, col in zip(centers, radii, cols):
        ax.add_patch(plt.Circle((c, 0), r, facecolor=to_rgba(col, 0.13),
                                edgecolor=col, lw=1.8, zorder=2))
    ax.plot(centers, np.zeros(3), "o", color="black", ms=4, zorder=5)
    ax.plot(eig.real, eig.imag, "x", color="black", ms=11, mew=2.4, zorder=6)
    # radius labels just outside each rim, in the disc's colour
    ax.text(4.0, 0.45, r"$r_1=1$", color=T.BLUE.dark, fontsize=14,
            ha="center", va="center", zorder=7)
    ax.text(5.12, 1.12, r"$r_2=1.5$", color=T.ORANGE.dark, fontsize=14,
            ha="left", va="bottom", zorder=7)
    ax.text(1.0, 0.58, r"$r_3=0.5$", color=T.GREEN.dark, fontsize=14,
            ha="center", va="bottom", zorder=7)
    ax.set_xlabel(r"$\mathrm{Re}\,z$", fontsize=15)
    ax.set_ylabel(r"$\mathrm{Im}\,z$", fontsize=15)
    ax.tick_params(labelsize=13)
    ax.set_aspect("equal")
    ax.set_xlim(0.3, 5.9)
    ax.set_ylim(-1.7, 1.7)
    clean_axes(ax, equal=True)
    save(fig, "mdl-la-gershgorin")
    # the slides read @fig figures from img/auto
    import shutil
    shutil.copyfile(os.path.join(IMG_DIR, "mdl-la-gershgorin.svg"),
                    os.path.join(IMG_DIR, "auto", "mdl-la-gershgorin.svg"))


def fig_power_iter():
    """Power iteration on S = [[2,1],[1,2]] from e1, log scale: the direction
    error sin(angle to v1) falls by 1/3 per step, the Rayleigh-quotient error
    by 1/9 (the squared rate, since S is symmetric)."""
    S = np.array([[2.0, 1.0], [1.0, 2.0]])
    v1 = np.array([1.0, 1.0]) / np.sqrt(2)
    ks = np.arange(0, 13)
    sin_err, rq_err = [], []
    v = np.array([1.0, 0.0])
    for _ in ks:
        u = v / np.linalg.norm(v)
        sin_err.append(np.sqrt(max(1.0 - (u @ v1) ** 2, 0.0)))
        rq_err.append(3.0 - u @ S @ u)
        v = S @ v
    sin_err, rq_err = np.array(sin_err), np.array(rq_err)

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    kk = np.linspace(0, 12, 50)
    # asymptotes: tan(angle) = 3^-k and lambda_1 - R = 2 / (9^k + 1) exactly
    ax.semilogy(kk, (1 / 3) ** kk, "--", color=GRAY, lw=1.2)
    ax.semilogy(kk, 2 * (1 / 9) ** kk, "--", color=GRAY, lw=1.2)
    ax.semilogy(ks, sin_err, "o", color=BLUE, ms=6, zorder=4)
    ax.semilogy(ks, rq_err, "s", color=ORANGE, ms=5.5, zorder=4)
    ax.text(12.3, sin_err[-1], r"$\sin\angle(\mathbf{v}_k,\mathbf{v}_1)$" "\n"
            r"$\times 1/3$ per step", color=T.BLUE.dark, fontsize=14,
            ha="left", va="center")
    ax.text(12.3, rq_err[-1], r"$\lambda_1 - R(\mathbf{v}_k)$" "\n"
            r"$\times 1/9$ per step", color=T.ORANGE.dark, fontsize=14,
            ha="left", va="center")
    ax.set_xticks([0, 3, 6, 9, 12])
    ax.set_xlim(-0.5, 12.5)
    ax.set_xlabel("iteration $k$", fontsize=15)
    ax.set_ylabel("error", fontsize=15)
    ax.tick_params(labelsize=13)
    clean_axes(ax, equal=False)
    save(fig, "mdl-la-power-iter")
    # the slides read @fig figures from img/auto
    import shutil
    shutil.copyfile(os.path.join(IMG_DIR, "mdl-la-power-iter.svg"),
                    os.path.join(IMG_DIR, "auto", "mdl-la-power-iter.svg"))


# =========================================================================== #
# SVD (Section 1.3)                                                           #
# =========================================================================== #

def fig_svd_action():
    """Four panels left->right showing the rotate-scale-rotate action of the
    shear A = [[1, 1], [0, 1]] on the unit circle: V = R(58.3 deg),
    Sigma = diag(phi, 1/phi), U = R(31.7 deg).  Exact SVD via numpy."""
    A = np.array([[1.0, 1.0], [0.0, 1.0]])
    U, s, Vt = np.linalg.svd(A)
    # LAPACK may return any joint sign choice: fix u_1 in the first quadrant
    # and det U = det V = +1, so the picture shows the rotations of the text
    if U[0, 0] < 0:
        U[:, 0] *= -1
        Vt[0] *= -1
    if np.linalg.det(U) < 0:
        U[:, 1] *= -1
        Vt[1] *= -1
    V = Vt.T

    t = np.linspace(0, 2 * np.pi, 400)
    circle = np.vstack([np.cos(t), np.sin(t)])

    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.2))
    # equal aspect, one window for all four panels (same scale); the shapes
    # reach about 1.6 in x and 1.4 in y once the labels are included
    mx, my = 2.0, 1.75
    lim = ((-mx, mx), (-my, my))

    def tip_label(ax, tip, text, color, out=0.36, side=0.0):
        """Place a label beyond an arrow tip, radially outside the shape,
        optionally nudged sideways so it never sits under the arrowhead."""
        tip = np.asarray(tip, float)
        d = tip / np.linalg.norm(tip)
        perp = np.array([-d[1], d[0]])
        p = tip + out * d + side * perp
        ax.text(p[0], p[1], text, color=color, fontsize=14,
                ha="center", va="center")

    # (1) input unit circle with right singular vectors v1, v2
    ax = axes[0]
    ax.set_title("(1) input", fontsize=14)
    ax.plot(circle[0], circle[1], color=GRAY, lw=1.6)
    arrow(ax, (0, 0), V[:, 0], color=BLUE, lw=2.4)
    arrow(ax, (0, 0), V[:, 1], color=ORANGE, lw=2.4)
    tip_label(ax, V[:, 0], r"$\mathbf{v}_1$", BLUE, side=-0.22)
    tip_label(ax, V[:, 1], r"$\mathbf{v}_2$", ORANGE, side=0.22)

    # (2) after V^T: circle unchanged, v's rotated onto the axes
    ax = axes[1]
    ax.set_title(r"(2) after $\mathbf{V}^{\!\!\top}$", fontsize=14)
    c2 = Vt @ circle
    ax.plot(c2[0], c2[1], color=GRAY, lw=1.6)
    arrow(ax, (0, 0), Vt @ V[:, 0], color=BLUE, lw=2.4)
    arrow(ax, (0, 0), Vt @ V[:, 1], color=ORANGE, lw=2.4)

    # (3) after Sigma: axis-aligned ellipse, semi-axes sigma1, sigma2
    ax = axes[2]
    ax.set_title(r"(3) after $\mathbf{\Sigma}$", fontsize=14)
    Sig = np.diag(s)
    c3 = Sig @ Vt @ circle
    ax.plot(c3[0], c3[1], color=BLUE, lw=2.2)
    arrow(ax, (0, 0), [s[0], 0], color=BLUE, lw=2.4)
    arrow(ax, (0, 0), [0, s[1]], color=ORANGE, lw=2.4)
    # labels OUTSIDE the ellipse outline (below it / above it)
    ax.text(s[0] * 0.5, -s[1] - 0.18, rf"$\sigma_1={s[0]:.2f}$", color=BLUE,
            fontsize=13, ha="center", va="top")
    ax.text(0.12, s[1] + 0.14, rf"$\sigma_2={s[1]:.2f}$", color=ORANGE,
            fontsize=13, ha="left", va="bottom")

    # (4) after U: ellipse rotated; draw sigma1 u1, sigma2 u2 as its axes
    ax = axes[3]
    ax.set_title(r"(4) after $\mathbf{U}$: $\mathbf{A}\mathbf{x}$", fontsize=14)
    c4 = U @ Sig @ Vt @ circle
    ax.plot(c4[0], c4[1], color=BLUE, lw=2.2)
    a1 = s[0] * U[:, 0]
    a2 = s[1] * U[:, 1]
    arrow(ax, (0, 0), a1, color=BLUE, lw=2.4)
    arrow(ax, (0, 0), a2, color=ORANGE, lw=2.4)
    tip_label(ax, a1, r"$\sigma_1\mathbf{u}_1$", BLUE, out=0.45, side=0.25)
    # a2 is the minor semi-axis, so the ellipse is tangent to the
    # perpendicular there: push the label purely radially, past the outline
    tip_label(ax, a2, r"$\sigma_2\mathbf{u}_2$", ORANGE, out=0.62)

    for ax in axes:
        clean_axes(ax, lim=lim, hide=True)
        axis_cross(ax, (-mx, mx), (-my, my))
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
        ax.add_patch(Rectangle((x, y0), bw, bh, fill=False, edgecolor=INK,
                               lw=1.4))
        sy = y0 + bh * (1 - split_frac)
        # top region
        ax.add_patch(Rectangle((x, sy), bw, bh - (sy - y0), facecolor=top_col,
                               alpha=0.16, lw=0))
        # bottom region
        ax.add_patch(Rectangle((x, y0), bw, sy - y0, facecolor=bot_col,
                               alpha=0.16, lw=0))
        ax.plot([x, x + bw], [sy, sy], color=INK, lw=0.9, ls="--")
        ax.text(x + bw / 2, y0 + bh - (bh - (sy - y0)) / 2, top_label,
                ha="center", va="center", fontsize=13, color=top_col)
        ax.text(x + bw / 2, y0 + (sy - y0) / 2, bot_label,
                ha="center", va="center", fontsize=13, color=INK)
        return sy

    sy_dom = box(dom_x, 0.62, r"row space" + "\n" + r"$\dim r$",
                 r"null space" + "\n" + r"$\mathcal{N}(\mathbf{A})$" + "\n"
                 + r"$\dim n-r$", BLUE, GRAY)
    sy_cod = box(cod_x, 0.62, r"column space" + "\n" + r"$\dim r$",
                 r"left null space" + "\n"
                 + r"$\mathcal{N}(\mathbf{A}^{\!\!\top})$" + "\n"
                 + r"$\dim m-r$", GREEN, GRAY)

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

    # the pseudoinverse runs the bijection backwards (dashed, below it)
    q_col = np.array([cod_x, y0 + bh * 0.56])
    q_row = np.array([dom_x + bw, y0 + bh * 0.56])
    ax.add_patch(FancyArrowPatch(q_col, q_row, arrowstyle="->",
                                 mutation_scale=16, color=GREEN, lw=1.8,
                                 linestyle=(0, (5, 3))))
    ax.text((q_col[0] + q_row[0]) / 2, q_col[1] - 0.14,
            r"$\mathbf{A}^{+}\!:\ \sigma_i\mathbf{u}_i \mapsto \mathbf{v}_i$",
            ha="center", va="top", fontsize=13, color=GREEN)

    # the null space collapses to 0, which lies in R^m where the column space
    # and the left null space meet: on the codomain's left edge, at the divider
    p_null = np.array([dom_x + bw, y0 + bh * 0.20])
    p_zero = np.array([cod_x, sy_cod])
    ax.add_patch(FancyArrowPatch(p_null, p_zero, arrowstyle="->",
                                 mutation_scale=14, color=GRAY, lw=1.6,
                                 linestyle=(0, (4, 3)), shrinkA=0, shrinkB=5))
    ax.plot(*p_zero, "o", color=INK, ms=6, zorder=5)
    ax.text(p_zero[0] - 0.22, p_zero[1] + 0.12, r"$\mathbf{0}$", ha="right",
            va="bottom", fontsize=15)

    ax.set_xlim(-0.6, cod_x + bw + 0.6)
    ax.set_ylim(-0.2, bh + 0.75)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "mdl-la-svd-subspaces")


def fig_eckart_young():
    """Eckart--Young on the committed 400x500 photograph img/cat1.jpg in
    grayscale: (a) its singular values on log-log axes, the twenty kept at
    rank 20 shaded; (b)-(e) the truncations at ranks 1, 5, 20, 50, titled
    with the relative Frobenius error and the storage fraction k(m+n+1)/mn
    (both computed here).  No network, no scipy: the same file and the same
    grayscale weights as the notebook cell."""
    import matplotlib.image as mpimg
    rgb = mpimg.imread(os.path.join(IMG_DIR, "cat1.jpg")).astype(float)
    img = rgb @ np.array([0.299, 0.587, 0.114])           # 400 x 500, 0..255
    U, s, Vt = np.linalg.svd(img, full_matrices=False)
    m, n = img.shape
    full_norm = np.linalg.norm(s)

    fig = plt.figure(figsize=(12.0, 3.2))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.25, 1, 1, 1, 1], wspace=0.12)
    axs = fig.add_subplot(gs[0, 0])
    idx = np.arange(1, len(s) + 1)
    axs.loglog(idx, s, color=BLUE, lw=1.8)
    axs.axvspan(1, 20, color=ORANGE, alpha=0.15)
    axs.text(24, s[0] * 0.9, "$k=20$", color=ORANGE, fontsize=14,
             ha="left", va="top")
    axs.set_xlim(1, len(s))
    axs.set_xlabel("index $i$", fontsize=14)
    axs.set_ylabel(r"$\sigma_i$", fontsize=14)
    axs.tick_params(labelsize=12)
    # same height as the image panels, whose height/width is m/n = 0.8
    axs.set_box_aspect(0.8 / 1.25)
    axs.spines["top"].set_visible(False)
    axs.spines["right"].set_visible(False)

    for j, k in enumerate([1, 5, 20, 50]):
        ax = fig.add_subplot(gs[0, 1 + j])
        approx = (U[:, :k] * s[:k]) @ Vt[:k, :]
        rel = np.linalg.norm(s[k:]) / full_norm   # ||A - A_k||_F / ||A||_F
        frac = k * (m + n + 1) / (m * n)
        ax.imshow(approx, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"rank {k}\nerr {rel:.2g} · {frac:.1%}", fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
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
    ax.plot(5, 0.75, "o", color=INK, ms=5, zorder=4)
    ax.text(5.3, 0.62, r"$\mathbf{x}$", fontsize=15, ha="left", va="center")
    arrow(ax, (5, 6.66), (5, 7.5), color=GRAY, lw=1.6, mut=12)
    ax.plot(5, 7.5, "o", color=INK, ms=5, zorder=4)
    ax.text(5.3, 7.5, r"$\mathbf{h}=\mathbf{W}\mathbf{x}+\mathbf{B}\mathbf{A}\mathbf{x}$",
            fontsize=14, ha="left", va="center")

    # frozen weight W (left column) -- wide enough for the "pretrained" line
    ax.add_patch(Rectangle((0.9, 2.4), 3.4, 2.4, facecolor=BLUE, alpha=0.15,
                           edgecolor=BLUE, lw=1.8))
    ax.text(xW, 3.95, r"$\mathbf{W}$", fontsize=16, ha="center", va="center",
            color=BLUE)
    ax.text(xW, 3.15, "pretrained, frozen\n" + r"$4096\times4096 = 16.8$M",
            fontsize=12, ha="center", va="center", color=BLUE)

    # low-rank bypass (right column): A compresses to r, B expands back
    ax.add_patch(Polygon([(6.3, 2.4), (8.5, 2.4), (7.75, 3.5), (7.05, 3.5)],
                         closed=True, facecolor=ORANGE, alpha=0.18,
                         edgecolor=ORANGE, lw=1.8))
    ax.text(xB, 2.85, r"$\mathbf{A}$ ($k\times n$)", fontsize=12.5,
            ha="center", va="center", color=ORANGE)
    ax.add_patch(Polygon([(7.05, 3.9), (7.75, 3.9), (8.5, 5.0), (6.3, 5.0)],
                         closed=True, facecolor=ORANGE, alpha=0.18,
                         edgecolor=ORANGE, lw=1.8))
    ax.text(xB, 4.55, r"$\mathbf{B}$ ($m\times k$)", fontsize=12.5,
            ha="center", va="center", color=ORANGE)
    arrow(ax, (xB, 3.5), (xB, 3.9), color=ORANGE, lw=1.8, mut=11)
    ax.text(xB + 0.32, 3.7, r"$k=8$", fontsize=12.5, ha="left", va="center",
            color=ORANGE)
    # trainable-count note in the free right margin, beside the trapezoids
    ax.text(8.8, 4.35, "trainable:\n" + r"$k(m{+}n)\approx65.5$K"
            + "\n" + r"($0.39\%$)", fontsize=12, ha="left", va="center",
            color=INK)

    # summation node
    ax.add_patch(plt.Circle((5, 6.4), 0.26, facecolor="white",
                            edgecolor=INK, lw=1.4, zorder=5))
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
        ax.contour(X, Y, Z, levels=10, colors=[LIGHT], linewidths=1.0)
        path = [np.array(x0, float)]
        x = np.array(x0, float)
        for _ in range(steps):
            grad = A @ x  # grad of 1/2 x^T A x  (A symmetric)
            x = x - eta * grad
            path.append(x.copy())
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], "-o", color=ORANGE, ms=3.5, lw=1.6)
        ax.plot(0, 0, "*", color=GREEN, ms=12)
        kappa = np.linalg.cond(A)
        ax.set_title(rf"$\kappa = {kappa:.2g}$", fontsize=16, pad=6)
        clean_axes(ax, lim=((-1.6, 1.6), (-1.6, 1.6)), hide=True)

    # well-conditioned: near-isotropic; step size safe -> nearly straight path
    run(axa, np.array([[1.0, 0.0], [0.0, 1.3]]),
        eta=0.55, x0=(-1.3, 1.1), steps=10)
    # ill-conditioned: elongated valley; eta near stability bound -> zig-zag
    run(axb, np.array([[1.0, 0.0], [0.0, 18.0]]),
        eta=0.102, x0=(-1.3, 1.0), steps=24)
    save(fig, "mdl-la-condition")


def fig_pca():
    """The cloud of the PCA cell: 300 points with standard deviations 3 and
    0.8 along axes rotated by pi/5, centered at (1, 0.6), and its principal
    axes (right singular vectors of the centered data) drawn from the mean,
    each two standard deviations 2 sigma_i / sqrt(n) long."""
    rng = np.random.default_rng(3)        # figures keep a fixed draw
    n = 300
    theta = np.pi / 5
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    data = (rng.standard_normal((n, 2)) * np.array([3.0, 0.8])) @ R.T \
        + np.array([1.0, 0.6])

    mean = data.mean(axis=0)
    Xc = data - mean
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    stds = s / np.sqrt(n)                  # standard deviation along each PC

    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    ax.scatter(data[:, 0], data[:, 1], s=10, color=BLUE, alpha=0.45,
               edgecolors="none")
    ax.plot(*mean, "o", color=INK, ms=5, zorder=6)
    for i, (col, lab) in enumerate(zip([ORANGE, GREEN], ["PC1", "PC2"])):
        d = Vt[i]
        if d[np.argmax(np.abs(d))] < 0:
            d = -d
        tip = mean + 2 * stds[i] * d
        arrow(ax, mean, tip, color=col, lw=2.8)
        lp = tip + 0.75 * d
        ax.text(lp[0], lp[1], lab, color=col, fontsize=14, fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                          alpha=0.9))
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$", fontsize=14)
    ax.set_ylabel("$x_2$", fontsize=14)
    clean_axes(ax, equal=True)
    save(fig, "mdl-la-pca")


def fig_transient_growth():
    """Transient growth made visible.  Left: ||M^k||_2 for the non-normal
    M = [[0.9, 4], [0, 0.8]] against Lambda = diag(0.9, 0.8), which has the
    same eigenvalues: same asymptotic slope, yet M first grows ten-fold.
    Right: the epsilon-pseudospectra (level sets of sigma_min(zI - M)) for
    eps = 0.05, 0.1, 0.2, 0.4; around Lambda they are discs of radius eps.
    Same matrices as the accompanying notebook cells."""
    M = np.array([[0.9, 4.0], [0.0, 0.8]])
    Lam = np.diag([0.9, 0.8])
    ks = np.arange(0, 61)
    norm_M = np.array([np.linalg.norm(np.linalg.matrix_power(M, k), 2)
                       for k in ks])
    norm_L = np.array([np.linalg.norm(np.linalg.matrix_power(Lam, k), 2)
                       for k in ks])
    k_peak = int(norm_M.argmax())

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.4, 3.9))
    axa.semilogy(ks, norm_M, color=BLUE, lw=2.4, zorder=4)
    axa.semilogy(ks, norm_L, color=ORANGE, lw=2.2, ls="--", zorder=4)
    axa.axhline(1.0, color=GRAY, lw=0.9, ls=":", zorder=2)
    axa.plot([k_peak], [norm_M[k_peak]], "o", color=BLUE, ms=6, zorder=6)
    axa.annotate(f"peak {norm_M[k_peak]:.1f} at $k={k_peak}$",
                 xy=(k_peak, norm_M[k_peak]),
                 xytext=(k_peak + 10, norm_M[k_peak] * 1.5),
                 fontsize=13, ha="left", va="center", color="black",
                 arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.2))
    axa.text(53, 0.64, r"$\|\mathbf{M}^k\|_2$", color=T.BLUE.dark,
             fontsize=14, ha="center", va="center", zorder=7)
    axa.text(13, 0.03, r"$\|\boldsymbol{\Lambda}^k\|_2$", color=T.ORANGE.dark,
             fontsize=14, ha="center", va="center")
    axa.set_xlabel(r"$k$", fontsize=15)
    axa.set_ylabel("spectral norm", fontsize=15)
    axa.set_ylim(1e-3, 40)

    # ----- right: pseudospectra via sigma_min(zI - X) on a grid -----
    gx = np.linspace(-1.6, 2.6, 320)
    gy = np.linspace(-1.7, 1.7, 260)
    GX, GY = np.meshgrid(gx, gy)
    Z = GX + 1j * GY

    def smin_grid(X):
        out = np.empty_like(GX)
        for i in range(GX.shape[0]):
            for j in range(GX.shape[1]):
                out[i, j] = np.linalg.svd(
                    Z[i, j] * np.eye(2) - X, compute_uv=False)[-1]
        return out

    levels = [0.05, 0.1, 0.2, 0.4]
    theta = np.linspace(0, 2 * np.pi, 200)
    axb.plot(np.cos(theta), np.sin(theta), color=GRAY, lw=1.0, ls=":",
             zorder=2)
    cs = axb.contour(GX, GY, smin_grid(M), levels=levels, colors=[BLUE],
                     linewidths=1.3, zorder=4)
    axb.contour(GX, GY, smin_grid(Lam), levels=levels, colors=[ORANGE],
                linewidths=1.1, linestyles="--", zorder=3)
    axb.text(2.6, -1.8, r"$\varepsilon = 0.05,\ 0.1,\ 0.2,\ 0.4$"
             " (inner to outer)", color="black", fontsize=12, ha="right",
             va="center", zorder=7)
    axb.plot([0.9, 0.8], [0.0, 0.0], "o", color="black", ms=5, zorder=6)
    axb.text(0.85, 1.57, r"$\varepsilon$-pseudospectra of $\mathbf{M}$",
             color=T.BLUE.dark, fontsize=13, ha="center", va="center",
             zorder=7, bbox=dict(facecolor="white", edgecolor="none",
                                 alpha=0.9, pad=1.5))
    axb.annotate(r"of $\boldsymbol{\Lambda}$: discs", xy=(0.62, -0.30),
                 xytext=(-0.8, -1.3), color=T.ORANGE.dark, fontsize=13,
                 ha="center", va="center", zorder=7,
                 bbox=dict(facecolor="white", edgecolor="none", alpha=0.9,
                           pad=1.5),
                 arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2,
                                 shrinkA=2, shrinkB=1))
    axb.text(-1.52, 1.3, "unit circle", color="black", fontsize=12,
             ha="left", va="center", zorder=7,
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.9,
                       pad=1.0))
    axb.set_xlabel(r"$\mathrm{Re}\,z$", fontsize=15)
    axb.set_ylabel(r"$\mathrm{Im}\,z$", fontsize=15)
    axb.set_aspect("equal")
    axb.set_ylim(-2.0, 1.7)

    for ax in (axa, axb):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("black")
        ax.spines["bottom"].set_color("black")
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
