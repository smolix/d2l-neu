#!/usr/bin/env python3
"""Generate every illustrative figure for the "Mathematics for Deep Learning ->
Calculus" chapter (``chapter_mdl-calculus``) in the one shared house style.

These replace the legacy appendix art (hand-drawn ``sub-area.svg``,
``rect-trans.svg``, ``sum-order.svg``, ``chain-net{1,2}.svg``,
``{pos,neg,zero}SecDer.svg``) and add the two planned diagrams
(``mdl-cal-cov-jacobian`` and ``mdl-cal-zoom-sequence``).  The notebooks /
prose reference the generated files with no drawing code (like the slide SVGs).

The shared style lives in ``gen_mdl_figures.py``; importing it applies the
``plt.rcParams`` (fixed ``svg.hashsalt`` + ``metadata={'Date': None}`` in
``save()`` make re-runs byte-for-byte identical) and exposes the palette and the
drawing helpers (``arrow``/``right_angle``/``clean_axes``/``axis_cross``/
``vlabel``).  Figures that show a *computed* result (the area-to-left curves, the
second-derivative triples, the change-of-variables stretch) use real numerical
computation so the pictures are exact, not sketches.

Run with the repo's pytorch venv (matplotlib + numpy are available):

    .venv-pytorch/bin/python tools/gen_mdl_calculus_figures.py

All figures are written to ``img/mdl-cal-<id>.svg``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

INK = fl.INK
FS = 13        # one size for every in-figure label, title and annotation
FS_NODE = 14   # symbols inside graph nodes


# --------------------------------------------------------------------------- #
# Small chapter-local helper for the schematic computational graphs.          #
# --------------------------------------------------------------------------- #

def node(ax, center, label, r=0.30, color=BLUE):
    """Draw a labelled circular node (light-blue fill, dark edge)."""
    cx, cy = center
    ax.add_patch(plt.Circle((cx, cy), r, facecolor=LIGHT, edgecolor="black",
                            lw=1.2, zorder=3))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=FS_NODE, zorder=4)


def ink_ticks(ax):
    """Named ticks ($x_0$, $a$, $b$, ...) are labels, so draw them in INK."""
    ax.tick_params(labelcolor=INK, labelsize=FS)


def edge(ax, c0, c1, r=0.30):
    """Arrow from the rim of node ``c0`` to the rim of node ``c1``."""
    c0 = np.asarray(c0, float)
    c1 = np.asarray(c1, float)
    d = c1 - c0
    d = d / np.linalg.norm(d)
    fl.arrow(ax, c0 + r * d, c1 - r * d, color=GRAY, lw=1.3, mut=11)


# =========================================================================== #
# Single-variable calculus                                                    #
# =========================================================================== #

def fig_zoom_sequence():
    """Three panels of the same smooth curve at shrinking x-ranges: as we zoom
    in around the base point the graph flattens to its tangent line."""
    f = lambda x: np.sin(x ** x)  # the curve used in the prose (sin(x**x))
    x0 = 2.0
    # finite-difference slope of the tangent at x0
    h = 1e-5
    slope = (f(x0 + h) - f(x0 - h)) / (2 * h)
    y0 = f(x0)

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.9))
    half_widths = [0.6, 0.06, 0.006]
    titles = [r"$\pm 0.6$", r"$\pm 0.06$", r"$\pm 0.006$"]
    for ax, hw, title in zip(axes, half_widths, titles):
        xs = np.linspace(x0 - hw, x0 + hw, 400)
        ax.plot(xs, f(xs), color=BLUE, lw=2.2, zorder=3)
        # the fixed tangent line at x0
        ax.plot(xs, y0 + slope * (xs - x0), "--", color=ORANGE, lw=1.6, zorder=2)
        ax.plot([x0], [y0], "o", color="black", ms=4, zorder=4)
        ax.set_title(f"zoom {title}", fontsize=14)
        ax.set_xticks([x0]); ink_ticks(ax)
        ax.set_xticklabels(["$x_0$"], fontsize=14)
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fl.save(fig, "mdl-cal-zoom-sequence")


def fig_secant_to_tangent():
    """Secants through (x, f(x)) and (x+eps, f(x+eps)) rotating into the tangent
    as eps -> 0; the limiting slope is f'(x).  Real cubic + real slopes."""
    f = lambda t: 0.15 * t ** 3 - 0.2 * t ** 2 + 0.3 * t + 0.6
    df = lambda t: 0.45 * t ** 2 - 0.4 * t + 0.3
    x0, xs = 1.0, np.linspace(-0.3, 2.7, 400)
    y0 = f(x0)

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(xs, f(xs), color=BLUE, lw=2.4, zorder=3)
    for k, eps in enumerate([1.1, 0.7, 0.35]):         # secants, eps shrinking
        slope = (f(x0 + eps) - y0) / eps
        ax.plot(xs, y0 + slope * (xs - x0), "-", color=GRAY,
                lw=1.0, alpha=0.35 + 0.15 * k, zorder=2)
        ax.plot([x0 + eps], [f(x0 + eps)], "o", color=GRAY, ms=4, zorder=4)
    ax.plot(xs, y0 + df(x0) * (xs - x0), "--", color=ORANGE, lw=2.0, zorder=5)
    ax.plot([x0], [y0], "o", color="black", ms=5, zorder=6)
    ax.text(2.15, 1.02, r"slope $f'(x)$", color=ORANGE,
            ha="left", va="top", fontsize=FS)
    ax.text(x0 + 0.08, y0 - 0.07, r"$(x,\,f(x))$", ha="left", va="top",
            fontsize=FS)
    ax.set_xlim(-0.3, 2.75); ax.set_ylim(0.1, 1.9)
    # the largest-eps secant point is x + eps; name it on the axis
    ax.set_xticks([x0, x0 + 1.1])
    ax.set_xticklabels(["$x$", r"$x+\epsilon$"]); ax.set_yticks([])
    ink_ticks(ax)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fl.save(fig, "mdl-cal-secant-to-tangent")


def fig_gd_step():
    """One gradient-descent step on a 1-D bowl.  The step $-\\eta f'(x)$ (green,
    horizontal) and the tangent (orange dashed) form a right triangle whose
    vertical leg is the first-order drop $\\eta f'(x)^2$; the curve itself lands
    at $x-\\eta f'(x)$ (green dot), a little above the tangent's prediction."""
    f, df = lambda t: 0.5 * t ** 2, lambda t: t
    x0, eta = 1.7, 0.6
    x1, xs = x0 - eta * df(x0), np.linspace(-2.4, 2.4, 400)
    y0 = f(x0)
    yt = y0 + df(x0) * (x1 - x0)            # tangent at x1: y0 - eta f'(x)^2
    ybot = -0.62

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(xs, f(xs), color=BLUE, lw=2.4, zorder=3)
    ax.plot(xs, y0 + df(x0) * (xs - x0), "--", color=ORANGE, lw=1.6, zorder=2)
    # drop lines to the named ticks on the x-axis
    ax.plot([x0, x0], [ybot, y0], ":", color=GRAY, lw=1.0, zorder=1)
    ax.plot([x1, x1], [ybot, yt], ":", color=GRAY, lw=1.0, zorder=1)
    # the step: horizontal, from x to x - eta f'(x)
    fl.arrow(ax, (x0, y0), (x1, y0), color=GREEN, lw=2.0, mut=13)
    ax.text((x0 + x1) / 2, y0 + 0.08, r"$-\eta f'(x)$", ha="center",
            va="bottom", color=GREEN, fontsize=FS)
    # first-order drop: the vertical leg of the tangent triangle
    fl.arrow(ax, (x1, y0), (x1, yt), color=ORANGE, lw=1.8, mut=12)
    ax.text(x1 - 0.12, y0 - 0.55, "first-order drop\n" + r"$\eta\,f'(x)^2$",
            ha="right", va="center", color=ORANGE, fontsize=FS,
            multialignment="right")
    ax.plot([x0], [y0], "o", color="black", ms=5, zorder=5)
    ax.plot([x1], [f(x1)], "o", color=GREEN, ms=5, zorder=5)
    ax.set_xlim(-2.4, 2.4); ax.set_ylim(ybot, 2.25)
    ax.set_xticks([x1, x0]); ax.set_xticklabels([r"$x-\eta f'(x)$", r"$x$"])
    ink_ticks(ax); ax.set_yticks([])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fl.save(fig, "mdl-cal-gd-step")


def fig_descent_lemma():
    """The descent lemma as a picture.  For $f(x)=\\sin(2x)$ the slope is
    $L$-Lipschitz with $L=4=\\max|f''|$, so the quadratic
    $q(s)=f(x)+f'(x)(s-x)+\\frac{L}{2}(s-x)^2$ is a *ceiling* over the whole
    graph ($q-f$ is convex with minimum $0$ at the base point).  Stepping to the
    parabola's minimizer --- the gradient step with $\\eta=1/L$ --- is guaranteed
    a drop of $f'(x)^2/2L$ even on the ceiling; the function, trapped below,
    drops at least as much (here visibly more: the bound is a worst-case floor)."""
    f = lambda t: np.sin(2 * t)
    df = lambda t: 2 * np.cos(2 * t)
    x0, L = 0.3, 4.0
    y0, g = f(x0), df(x0)                      # 0.564642, 1.650671
    q = lambda t: y0 + g * (t - x0) + 0.5 * L * (t - x0) ** 2
    xp = x0 - g / L                            # parabola minimizer -0.112668
    qp, fp = q(xp), f(xp)                      # 0.224053, -0.223434
    xs = np.linspace(-0.9, 0.9, 400)

    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    ax.plot(xs, f(xs), color=BLUE, lw=2.4, zorder=3)
    ax.plot(xs, q(xs), "--", color=ORANGE, lw=2.0, zorder=4)
    # the minimizer of the ceiling = the eta = 1/L gradient step
    ax.plot([xp, xp], [-1.18, qp], ":", color=GRAY, lw=1.1, zorder=2)
    xa = xp - 0.06                             # the actual-drop arrow's column
    ax.plot([xa, x0], [y0, y0], ":", color=GRAY, lw=1.1, zorder=2)
    ax.plot([x0], [y0], "o", color="black", ms=5, zorder=6)
    ax.text(0.38, 0.45, r"$(x,\,f(x))$", ha="left", va="top", fontsize=FS)
    ax.plot([xp], [qp], "o", color=ORANGE, ms=5, zorder=6)
    ax.plot([xp], [fp], "o", color=BLUE, ms=5, zorder=6)
    # guaranteed drop: from the starting height down to the ceiling's minimum;
    # label lifted high and left so it never crowds the rising orange parabola
    fl.arrow(ax, (xp, y0), (xp, qp), color=ORANGE, lw=1.8, mut=12)
    ax.annotate(r"guaranteed drop $f'(x)^2/2L\approx 0.341$",
                xy=(xp - 0.02, (y0 + qp) / 2), xytext=(-0.12, 1.42),
                ha="center", fontsize=FS, color=ORANGE,
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.0,
                                connectionstyle="arc3,rad=0.25"))
    # the function, trapped below the ceiling, drops even further: the actual
    # drop runs from the same starting height f(x) all the way down to f at the
    # step, drawn just left of the guaranteed drop so the two compare directly
    fl.arrow(ax, (xa, y0), (xa, fp), color=BLUE, lw=1.8, mut=12)
    ax.plot([xa, xp], [fp, fp], ":", color=GRAY, lw=1.1, zorder=2)
    ax.annotate("actual drop", xy=(xa - 0.02, (y0 + fp) / 2 - 0.12),
                xytext=(-0.40, -0.12), ha="right", va="center", fontsize=FS,
                color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.0,
                                connectionstyle="arc3,rad=-0.2"))
    ax.set_xlim(-0.95, 0.95); ax.set_ylim(-1.18, 1.72)
    ax.set_xticks([xp]); ax.set_xticklabels([r"$x-f'(x)/L$  ($\eta=1/L$)"])
    ink_ticks(ax); ax.set_yticks([])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fl.save(fig, "mdl-cal-descent-lemma")


def fig_relu_corner():
    """Two convex kinks and their subdifferentials. Left: $|x|$, whose corner at 0
    admits a fan of supporting lines with slopes sweeping $[-1,1]$. Right: ReLU,
    $\\max(0,x)$, whose corner admits slopes sweeping $[0,1]$. Off the corner each
    has a single tangent and the subdifferential collapses to the derivative."""
    xs = np.linspace(-2.0, 2.0, 400)
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.0, 3.9))

    # left: |x|, subgradient fan sweeping [-1, 1]
    axa.axhline(0, color=INK, lw=1.0, zorder=3)
    for s in np.linspace(-1.0, 1.0, 9):
        axa.plot(xs, s * xs, "-", color=ORANGE, lw=1.0, alpha=0.5, zorder=2)
    axa.plot(xs, np.abs(xs), color=BLUE, lw=2.6, zorder=4)
    axa.plot([0], [0], "o", color="black", ms=6, zorder=5)
    axa.text(0.0, 1.72, r"$|x|,\ \ \partial|x|(0)=[-1,1]$", color=BLUE,
              fontsize=FS, ha="center")
    axa.set_xlim(-2.0, 2.0); axa.set_ylim(-0.55, 2.0)
    # the zero line (drawn over the fan) is the only axis: no L-frame
    axa.set_xticks([]); axa.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        axa.spines[side].set_visible(False)

    # right: ReLU = max(0, x), subgradient fan sweeping [0, 1]
    axb.axhline(0, color=INK, lw=1.0, zorder=3)
    for s in np.linspace(0.0, 1.0, 6):
        axb.plot(xs, s * xs, "-", color=ORANGE, lw=1.0, alpha=0.5, zorder=2)
    axb.plot(xs, np.maximum(0.0, xs), color=BLUE, lw=2.6, zorder=4)
    axb.plot([0], [0], "o", color="black", ms=6, zorder=5)
    axb.text(0.0, 1.72, r"$\mathrm{ReLU}(x),\ \ \partial\,\mathrm{ReLU}(0)=[0,1]$", color=BLUE,
              fontsize=FS, ha="center")
    axb.set_xlim(-2.0, 2.0); axb.set_ylim(-0.55, 2.0)
    axb.set_xticks([]); axb.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        axb.spines[side].set_visible(False)
    fl.save(fig, "mdl-cal-relu-corner")


def fig_mvt():
    """The mean value theorem: the secant chord from $(a,f(a))$ to $(b,f(b))$ is
    parallel to the tangent at some interior point $\\xi$ where $f'(\\xi)$ equals
    the secant slope. Real $f$, real $\\xi$ located numerically."""
    f = lambda t: np.sin(t) + 0.25 * t
    df = lambda t: np.cos(t) + 0.25
    a, b = 0.6, 4.2
    xs = np.linspace(a - 0.3, b + 0.5, 400)
    slope = (f(b) - f(a)) / (b - a)
    grid = np.linspace(a, b, 4000)
    xi = float(grid[np.argmin(np.abs(df(grid) - slope))])   # f'(xi) = secant slope

    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    ax.plot(xs, f(xs), color=BLUE, lw=2.4, zorder=3)
    ax.plot([a, b], [f(a), f(b)], "-", color=GRAY, lw=1.6, zorder=2)
    # the parallel tangent, trimmed so it does not shoot off to the frame edge
    xt = np.linspace(xi - 1.3, xi + 1.3, 200)
    ax.plot(xt, f(xi) + slope * (xt - xi), "--", color=ORANGE, lw=2.0, zorder=4)
    # a, xi, b are positions on the x-axis: named ticks with dotted drop lines
    ybot = -0.18
    for x in (a, xi, b):
        ax.plot([x, x], [ybot, f(x)], ":", color=GRAY, lw=1.0, zorder=1)
    for x in (a, b):
        ax.plot([x], [f(x)], "o", color="black", ms=5, zorder=5)
    ax.plot([xi], [f(xi)], "o", color=ORANGE, ms=6, zorder=6)
    xm = 0.5 * (a + b) + 0.35                       # label just under the chord
    ax.text(xm, f(a) + slope * (xm - a) - 0.07, "secant", color=INK,
            fontsize=FS, ha="center", va="top")
    ax.set_xlim(a - 0.35, b + 0.65); ax.set_ylim(ybot, 1.62)
    ax.set_xticks([a, xi, b]); ax.set_xticklabels(["$a$", r"$\xi$", "$b$"])
    ink_ticks(ax); ax.get_xticklabels()[1].set_color(ORANGE)
    ax.set_yticks([])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fl.save(fig, "mdl-cal-mvt")


def fig_smooth_not_analytic():
    """$f(x)=e^{-1/x^2}$ (with $f(0)=0$) is smooth, yet every derivative vanishes
    at 0, so its Taylor series there is identically zero and agrees with $f$ only
    at the single point $x=0$: smoothness does not imply analyticity."""
    xs = np.linspace(-1.5, 1.5, 600)
    with np.errstate(divide="ignore"):
        f = np.where(xs == 0, 0.0, np.exp(-1.0 / xs ** 2))

    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    ax.axhline(0, color=INK, lw=1.0, zorder=1)
    ax.plot(xs, f, color=BLUE, lw=2.6, zorder=3, label=r"$f(x)=e^{-1/x^2}$")
    ax.plot(xs, np.zeros_like(xs), "--", color=ORANGE, lw=2.0, zorder=2,
            label=r"Taylor series at $0\ \equiv\ 0$")
    ax.plot([0], [0], "o", color="black", ms=5, zorder=4)
    ax.text(0, -0.014, "$0$", ha="center", va="top", color=INK, fontsize=FS)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-0.06, 0.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper center", fontsize=FS, frameon=False)
    # the zero line is the x-axis; an L-frame below it would be a second one
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    fl.save(fig, "mdl-cal-smooth-not-analytic")


# =========================================================================== #
# Integral calculus                                                           #
# =========================================================================== #

def fig_riemann():
    """The definite integral as a limit of Riemann sums: three panels of the same
    curve f(x)=x/(1+x^2) on [0,2] with left-endpoint rectangles of shrinking
    width epsilon.  The rectangle sum (printed in each title) marches toward the
    true area (1/2)log 5 as epsilon -> 0."""
    f = lambda t: t / (1 + t ** 2)
    a, b = 0.0, 2.0
    truth = 0.5 * np.log(5.0)
    xs = np.linspace(a, b, 500)

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1))
    for ax, eps in zip(axes, [0.5, 0.2, 0.05]):
        left = np.arange(a, b, eps)                 # left endpoints
        approx = float(np.sum(eps * f(left)))       # the real Riemann sum
        ax.bar(left, f(left), width=eps, align="edge", facecolor=BLUE,
               alpha=0.30, edgecolor=BLUE, lw=0.7, zorder=2)
        ax.plot(xs, f(xs), color="black", lw=2.0, zorder=3)
        ax.set_title(rf"$\epsilon={eps:g}$:  sum $={approx:.3f}$", fontsize=FS)
        ax.set_ylim(0, 0.6)
        ax.set_xlim(a, b)
        ax.set_yticks([])
        ax.set_xticks([a, b]); ax.set_xticklabels(["$a$", "$b$"]); ink_ticks(ax)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.text(0.5, -0.02, rf"true area $=\frac{{1}}{{2}}\log 5 = {truth:.3f}$",
             ha="center", va="top", fontsize=FS, color=INK)
    fl.save(fig, "mdl-cal-riemann")


def fig_sub_area():
    """Area under a bell-shaped curve between a and b equals the area-to-the-left
    of b minus the area-to-the-left of a:  F(b) - F(a).  Three panels with a
    minus and an equals sign, all shading the *same* real curve."""
    g = lambda t: np.exp(-0.5 * ((t - 0.0) / 1.1) ** 2)  # a clean bell curve
    xs = np.linspace(-2.6, 2.6, 500)
    a, b = -0.4, 1.3

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.7))

    def panel(ax, lo, hi, title):
        ax.plot(xs, g(xs), color=BLUE, lw=2.0, zorder=3)
        mask = (xs >= lo) & (xs <= hi)
        ax.fill_between(xs[mask], 0, g(xs[mask]), color=BLUE, alpha=0.22, lw=0)
        ax.set_title(title, fontsize=FS)
        ax.set_ylim(0, 1.18)
        ax.set_xlim(xs[0], xs[-1])
        ax.set_yticks([])
        # a is the base point the accumulated areas start from: the left edge
        ax.set_xticks([xs[0], a, b])
        ax.set_xticklabels(["$a$", "$x_1$", "$x_2$"]); ink_ticks(ax)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    lo = xs[0]
    panel(axes[0], a, b, r"$\int_{x_1}^{x_2} f$")
    panel(axes[1], lo, b, r"$\int_a^{x_2} f$")
    panel(axes[2], lo, a, r"$\int_a^{x_1} f$")
    # connecting symbols, drawn in figure coordinates between the panels
    fig.text(0.365, 0.5, r"$=$", ha="center", va="center", fontsize=18)
    fig.text(0.640, 0.5, r"$-$", ha="center", va="center", fontsize=18)
    fl.save(fig, "mdl-cal-sub-area")


def fig_rect_trans():
    """Change of variables y=u(x) as one curve seen through two rulers.  The SAME
    function ``f`` (and the same height ``f(u(x))``) is drawn over the x-axis
    (left) and over the y=u-axis (right); both slivers stand on a solid baseline.
    The base interval of width epsilon at x is the *image* of width epsilon*u'(x)
    after u, so the right sliver is wider by exactly the local stretch du/dx --
    which is the factor :eqref:`eq_mdl-change_var` inserts so the two areas agree.
    Numbers are real: u(x)=x^2/2 here, so at x0 the stretch is u'(x0)=x0."""
    # one curve f, low and broad so the slivers stand well clear of it
    f = lambda t: 0.30 + 0.42 * np.exp(-0.5 * ((t - 2.0) / 1.05) ** 2)
    u = lambda t: 0.5 * t ** 2          # the substitution; u'(x)=x is the stretch
    x0, eps = 1.5, 0.40
    du = x0                              # local stretch u'(x0) = x0 = 1.5
    h = f(u(x0))                         # the shared height f(u(x0)) = f(y0)
    y0 = u(x0)                           # 1.125 -- where the image sliver sits
    weps = eps * du                      # stretched width epsilon * u'(x)

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    base = -0.045                        # y of the solid baseline both stand on

    def baseline(ax, x1):
        """A solid x-axis the rectangle visibly sits on, with an arrowhead."""
        ax.annotate("", xy=(x1, base), xytext=(-0.05, base),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.3,
                                    shrinkA=0, shrinkB=0, mutation_scale=13),
                    zorder=4)

    def height_label(ax, x_left, text, color):
        """Vertical measure of the sliver height, labelled to its left so the
        text never lands on the curve."""
        ax.annotate("", xy=(x_left, base), xytext=(x_left, h),
                    arrowprops=dict(arrowstyle="<->", color=color, lw=1.1),
                    zorder=5)
        ax.text(x_left - 0.10, (base + h) / 2, text, ha="right", va="center",
                color=INK, fontsize=FS)

    # ---- (a) the x-world: narrow sliver of width epsilon on the x-axis ----
    xs = np.linspace(0.0, 3.0, 400)
    axa.plot(xs, f(u(xs)), color=BLUE, lw=2.4, zorder=3)
    axa.add_patch(Rectangle((x0, base), eps, h - base, facecolor=BLUE, alpha=0.26,
                            edgecolor=BLUE, lw=1.5, zorder=2))
    baseline(axa, 3.05)
    height_label(axa, x0, r"$f(u(x))$", BLUE)
    axa.annotate("", xy=(x0 + eps, base - 0.08), xytext=(x0, base - 0.08),
                 arrowprops=dict(arrowstyle="<->", color=INK, lw=1.1,
                                 shrinkA=0, shrinkB=0))
    axa.text(x0 + eps / 2, base - 0.12, r"$\epsilon$", ha="center", va="top",
             color=INK, fontsize=FS)
    axa.set_xlim(-0.55, 3.05); axa.set_ylim(-0.30, 0.80)
    axa.set_xticks([x0]); axa.set_xticklabels(["$x$"], fontsize=FS)
    axa.set_yticks([]); axa.axis("off")

    # ---- (b) the y=u-world: the IMAGE sliver, wider by u'(x), same height ----
    ys = np.linspace(0.0, 3.0, 400)
    axb.plot(ys, f(ys), color=BLUE, lw=2.4, zorder=3)
    axb.add_patch(Rectangle((y0, base), weps, h - base, facecolor=ORANGE,
                            alpha=0.30, edgecolor=ORANGE, lw=1.5, zorder=2))
    baseline(axb, 3.05)
    height_label(axb, y0, r"$f(y)$", ORANGE)
    axb.annotate("", xy=(y0 + weps, base - 0.08), xytext=(y0, base - 0.08),
                 arrowprops=dict(arrowstyle="<->", color=INK, lw=1.1,
                                 shrinkA=0, shrinkB=0))
    axb.text(y0 + weps / 2, base - 0.12, r"$\epsilon\,u'(x)$", ha="center",
             va="top", color=INK, fontsize=FS)
    axb.set_xlim(-0.55, 3.05); axb.set_ylim(-0.30, 0.80)
    axb.set_xticks([y0]); axb.set_xticklabels(["$u(x)$"], fontsize=FS)
    axb.set_yticks([]); axb.axis("off")

    # the map y = u(x) carrying the base interval from (a) to (b)
    # (set above both curves' inner ends so the label never sits on a curve)
    fig.text(0.5, 0.75, r"$y=u(x)$", ha="center", va="center", color=INK,
             fontsize=FS)
    fig.patches.append(
        FancyArrowPatch((0.45, 0.67), (0.55, 0.67),
                        transform=fig.transFigure, arrowstyle="-|>",
                        mutation_scale=16, color=GRAY, lw=1.5))
    fl.save(fig, "mdl-cal-rect-trans")


def fig_bell_surface():
    """The bell surface $z=e^{-x^2-y^2}$ over the box $[-2,2]^2$: the double
    integral is the volume between the surface and the base plane, approximated
    by tiling the whole base with $\\epsilon\\times\\epsilon$ squares
    ($\\epsilon=0.5$) and standing a box of height $f$ (at the square's centre)
    on each.  One column of boxes at a fixed $x$ is highlighted: it is one
    inner sum $\\sum_j \\epsilon f(x_i, y_j)$ of Fubini's theorem."""
    f = lambda X, Y: np.exp(-X ** 2 - Y ** 2)
    fig = plt.figure(figsize=(6.2, 5.0))
    ax = fig.add_subplot(projection="3d")

    eps = 0.5
    edges = np.arange(-2.0, 2.0 - 1e-9, eps)
    X0, Y0 = np.meshgrid(edges, edges, indexing="ij")
    X0, Y0 = X0.ravel(), Y0.ravel()
    H = f(X0 + eps / 2, Y0 + eps / 2)
    col_x = 0.0                                  # the highlighted column
    from matplotlib.colors import to_rgb
    tint = lambda c, w: tuple(w * np.array(to_rgb(c)) + (1 - w))
    elev, azim = 26, -52
    # paint the boxes back to front: depth along the viewing direction
    view = np.array([np.cos(np.deg2rad(elev)) * np.cos(np.deg2rad(azim)),
                     np.cos(np.deg2rad(elev)) * np.sin(np.deg2rad(azim))])
    depth = (X0 + eps / 2) * view[0] + (Y0 + eps / 2) * view[1]
    ax.computed_zorder = False
    for rank, k in enumerate(np.argsort(depth)):
        hi = np.isclose(X0[k], col_x)
        ax.bar3d(X0[k], Y0[k], 0.0, eps, eps, H[k],
                 color=tint(ORANGE, 0.85) if hi else tint(BLUE, 0.45),
                 shade=True, edgecolor="white", linewidth=0.5, zorder=rank)

    ax.text(-1.0, -0.4, 1.12, r"$z=e^{-x^2-y^2}$", fontsize=15, ha="center",
            color=INK)
    ax.set_xlabel("$x$", fontsize=15, labelpad=-6)
    ax.set_ylabel("$y$", fontsize=15, labelpad=-6)
    ax.set_zlabel("$z$", fontsize=15, labelpad=-6)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_color("black")
    ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.set_zlim(0, 1.05)
    ax.view_init(elev=elev, azim=azim)
    fl.save(fig, "mdl-cal-bell-surface")


def fig_sum_order():
    """The discretization behind Fubini, on one grid with corner (a, c), index
    i along x and j along y.  (1) Sum the cells of each column (fixed i, over
    j): every column collapses to one total, drawn as the strip of cells below
    the grid.  (2) Add those column totals left to right.  Reordering a finite
    sum changes nothing."""
    nx, ny = 6, 4
    fig, ax = plt.subplots(figsize=(6.0, 4.9))

    for i in range(nx):
        for j in range(ny):
            ax.add_patch(Rectangle((i, j), 1, 1, facecolor=LIGHT, alpha=0.30,
                                   edgecolor=GRAY, lw=0.8, zorder=1))

    # coordinate axes along the grid's bottom and left edges
    fl.arrow(ax, (-0.35, -0.35), (nx + 0.45, -0.35), color="black", lw=1.0,
             mut=11)
    fl.arrow(ax, (-0.35, -0.35), (-0.35, ny + 0.40), color="black", lw=1.0,
             mut=11)
    ax.text(nx + 0.55, -0.35, "$x$, index $i$", ha="left", va="center",
            color=INK, fontsize=FS)
    ax.text(-0.35, ny + 0.50, "$y$, index $j$", ha="center", va="bottom",
            color=INK, fontsize=FS)
    ax.text(-0.45, 0.0, "$(a,c)$", ha="right", va="center", color=INK,
            fontsize=FS)
    ax.plot([0], [0], "o", color="black", ms=3.5, zorder=4)

    # (1) sum WITHIN one column (fixed i, over j)
    col = 2
    for j in range(ny):
        ax.add_patch(Rectangle((col, j), 1, 1, facecolor=BLUE, alpha=0.32,
                               edgecolor=BLUE, lw=1.3, zorder=2))
    strip_y = -1.75
    fl.arrow(ax, (col + 0.5, ny - 0.18), (col + 0.5, strip_y + 1.0), color=BLUE,
             lw=2.2, mut=15)
    ax.text(col + 0.5, ny + 0.12, "(1) sum each column", ha="center",
            va="bottom", color=INK, fontsize=FS)

    # the column totals, one cell per column, aligned below the grid
    for i in range(nx):
        fc = ORANGE if i != col else BLUE
        ax.add_patch(Rectangle((i, strip_y), 1, 1, facecolor=fc, alpha=0.30,
                               edgecolor=ORANGE, lw=1.1, zorder=2))

    # (2) add the column totals together
    fl.arrow(ax, (0.18, strip_y + 0.5), (nx - 0.18, strip_y + 0.5),
             color=ORANGE, lw=2.2, mut=15)
    ax.text(nx / 2, strip_y - 0.15, "(2) add the column totals", ha="center",
            va="top", color=INK, fontsize=FS)

    # epsilon side labels on the top-right corner cell
    ax.annotate("", xy=(nx, ny + 0.12), xytext=(nx - 1, ny + 0.12),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9,
                                shrinkA=0, shrinkB=0))
    ax.text(nx - 0.5, ny + 0.20, r"$\epsilon$", ha="center", va="bottom",
            color=INK, fontsize=FS)
    ax.annotate("", xy=(nx + 0.18, ny), xytext=(nx + 0.18, ny - 1),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9,
                                shrinkA=0, shrinkB=0))
    ax.text(nx + 0.30, ny - 0.5, r"$\epsilon$", ha="left", va="center",
            color=INK, fontsize=FS)

    ax.set_xlim(-1.6, nx + 2.4)
    ax.set_ylim(strip_y - 0.85, ny + 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    fl.save(fig, "mdl-cal-sum-order")


def fig_cov_jacobian():
    """Change of variables, 1-D beside 2-D.  (a) The scalar substitution scales a
    segment of length epsilon by u'(x).  (b) A linear phi sends the unit square
    (area 1) to a parallelogram whose area is the local volume factor
    |det Dphi|; the two shapes sit side by side so each carries its own area
    label.  The det is computed from the real matrix."""
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.6, 3.2),
                                   gridspec_kw={"width_ratios": [1.9, 4.2]})

    # --- (a) the 1-D analog: a segment of length eps stretched to eps*u' ---
    eps, du = 0.85, 1.85
    yb, yo = 1.45, 0.45                  # heights of the two segments
    # a faint left rule the two segments both start from (the common origin)
    axa.plot([0, 0], [yo - 0.22, yb + 0.22], color=LIGHT, lw=1.2, zorder=1)
    axa.plot([0, eps], [yb, yb], color=BLUE, lw=7, solid_capstyle="butt",
             zorder=3)
    axa.plot([0, eps * du], [yo, yo], color=ORANGE, lw=7, solid_capstyle="butt",
             zorder=3)
    # end ticks so the lengths read cleanly
    for x, y, c in [(eps, yb, BLUE), (eps * du, yo, ORANGE)]:
        axa.plot([x, x], [y - 0.10, y + 0.10], color=c, lw=1.5, zorder=4)
    # the map u, straight down from the short segment to the stretched one
    fl.arrow(axa, (eps / 2, yb - 0.17), (eps / 2, yo + 0.17), color=GRAY,
             lw=1.4, mut=13)
    axa.text(eps / 2 + 0.08, (yb + yo) / 2, r"$u$", ha="left", va="center",
             color=INK, fontsize=15)
    axa.text(eps / 2, yb + 0.17, r"$\epsilon$", ha="center", va="bottom",
             color=INK, fontsize=FS)
    axa.text(eps * du / 2, yo - 0.17, r"$\epsilon\,u'(x)$", ha="center",
             va="top", color=INK, fontsize=FS)
    axa.set_xlim(-0.15, 1.75); axa.set_ylim(-0.05, 1.85)
    axa.set_aspect("equal")
    axa.axis("off")

    # --- (b) the 2-D version: unit square -> parallelogram under phi ---
    A = np.array([[1.5, 0.6], [0.4, 1.3]])   # linear phi = matrix multiply
    det = np.linalg.det(A)                    # 1.71
    unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]).T
    shift = np.array([[1.75], [0.0]])         # the image drawn beside the square
    img = A @ unit + shift
    axb.add_patch(Polygon(unit.T, closed=True, facecolor=BLUE, alpha=0.20,
                          edgecolor=BLUE, lw=1.6, zorder=2))
    axb.add_patch(Polygon(img.T, closed=True, facecolor=ORANGE, alpha=0.24,
                          edgecolor=ORANGE, lw=2, zorder=2))
    # the basis edges of the square and their images A e1, A e2
    o = shift[:, 0]
    fl.arrow(axb, (0, 0), (1, 0), color=BLUE, lw=1.8)
    fl.arrow(axb, (0, 0), (0, 1), color=BLUE, lw=1.8)
    fl.arrow(axb, o, o + A @ np.array([1, 0]), color=ORANGE, lw=1.8)
    fl.arrow(axb, o, o + A @ np.array([0, 1]), color=ORANGE, lw=1.8)
    # the map phi between the two shapes
    fl.arrow(axb, (1.12, 0.62), (1.88, 0.62), color=GRAY, lw=1.5, mut=14)
    axb.text(1.50, 0.70, r"$\boldsymbol{\phi}$", ha="center", va="bottom",
             color=INK, fontsize=15)
    axb.text(0.5, 0.5, "area $1$", ha="center", va="center", color=INK,
             fontsize=FS)
    cx, cy = img.mean(axis=1)
    axb.text(cx, cy, "area\n" + rf"$|\det\mathbf{{J}}_{{\boldsymbol{{\phi}}}}|={abs(det):.2g}$",
             ha="center", va="center", color=INK, fontsize=FS,
             multialignment="center")
    axb.set_xlim(-0.15, 4.05); axb.set_ylim(-0.15, 1.85)
    axb.set_aspect("equal")
    axb.axis("off")

    fl.save(fig, "mdl-cal-cov-jacobian")


# =========================================================================== #
# Multivariable calculus -- chain-rule computational graphs                   #
# =========================================================================== #

def fig_chain_net1():
    """Computational graph for the layered composition
    {w,x,y,z} -> {a,b} -> {u,v} -> f: each layer fully connected to the next.
    Nodes are values; edges show functional dependence (left depends-on right)."""
    fig, ax = plt.subplots(figsize=(5.6, 4.4))

    L1 = {"w": (0, 3), "x": (0, 2), "y": (0, 1), "z": (0, 0)}
    L2 = {"a": (1.6, 2.2), "b": (1.6, 0.8)}
    L3 = {"u": (3.2, 2.2), "v": (3.2, 0.8)}
    L4 = {"f": (4.8, 1.5)}

    # edges first (under the nodes): each layer fully connects to the next
    for c0 in L1.values():
        for c1 in L2.values():
            edge(ax, c0, c1)
    for c0 in L2.values():
        for c1 in L3.values():
            edge(ax, c0, c1)
    for c0 in L3.values():
        for c1 in L4.values():
            edge(ax, c0, c1)

    for layer in (L1, L2, L3, L4):
        for lab, c in layer.items():
            node(ax, c, f"${lab}$")

    ax.set_xlim(-0.5, 5.3); ax.set_ylim(-0.6, 3.6)
    ax.set_aspect("equal")
    ax.axis("off")
    fl.save(fig, "mdl-cal-chain-net1")


def fig_chain_net2():
    """A more subtle chain-rule graph with three paths from y to f.  Built to
    match the dependencies used in the prose:
        f <- r, s, p ;  r <- p ;  p <- y ;  s <- q ;  q <- y
    so that df/dy = df/dr dr/dp dp/dy + df/dp dp/dy + df/ds ds/dq dq/dy."""
    fig, ax = plt.subplots(figsize=(5.6, 2.8))

    Y = {"y": (0, 1.0)}
    MID = {"p": (1.5, 1.7), "q": (1.5, 0.3)}
    INNER = {"r": (3.0, 1.7), "s": (3.0, 0.3)}
    F = {"f": (4.5, 1.0)}

    edges = [
        ("y", "p"), ("y", "q"),       # y feeds p and q
        ("p", "r"),                    # p feeds r
        ("q", "s"),                    # q feeds s
    ]
    pos = {**Y, **MID, **INNER, **F}
    for s, t in edges:
        edge(ax, pos[s], pos[t])
    # f depends directly on r, s, and (the skip edge) p  -> three paths from y
    edge(ax, pos["r"], pos["f"])
    edge(ax, pos["s"], pos["f"])
    r0 = 0.30
    tail = np.array(pos["p"]) + r0 * np.array([np.cos(-0.55), np.sin(-0.55)])
    head = np.array(pos["f"]) + r0 * np.array([-1.0, 0.0])
    ax.annotate("", xy=head, xytext=tail,          # the skip edge p -> f
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.3,
                                shrinkA=0, shrinkB=0, mutation_scale=11,
                                connectionstyle="arc3,rad=0.12"))

    for layer in (Y, MID, INNER, F):
        for lab, c in layer.items():
            node(ax, c, f"${lab}$")

    ax.set_xlim(-0.5, 5.0); ax.set_ylim(-0.3, 2.3)
    ax.set_aspect("equal")
    ax.axis("off")
    fl.save(fig, "mdl-cal-chain-net2")


def fig_gradient_field():
    """Gradient and level sets: contours of a scalar field with real gradient
    arrows crossing them at right angles, pointing uphill, and longer where the
    contours bunch (the field changes fastest).  Illustrates the two propositions
    of the gradient-geometry section."""
    f = lambda X, Y: 0.5 * X ** 2 + 0.18 * Y ** 2        # an elongated bowl
    grad = lambda px, py: np.array([px, 0.36 * py])      # exact gradient
    xs = np.linspace(-2.6, 2.6, 240)
    X, Y = np.meshgrid(xs, xs)

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.contour(X, Y, f(X, Y), levels=np.linspace(0.2, 4.0, 9),
               colors=GRAY, linewidths=1.0, zorder=2)
    for px, py in [(-1.6, 1.15), (1.6, -1.4), (0.7, 1.9),
                   (-1.1, -1.0), (1.9, 0.6), (-0.5, -2.0)]:
        g = grad(px, py)
        fl.arrow(ax, (px, py), (px + 0.32 * g[0], py + 0.32 * g[1]),
                 color=ORANGE, lw=2.0, mut=12)
    ax.plot([0], [0], "o", color=BLUE, ms=7, zorder=4)     # the minimum
    ax.text(0.12, -0.32, "min", color=BLUE, fontsize=FS)
    ax.text(-1.95, 1.42, r"$\nabla \ell$", color=ORANGE, fontsize=15, ha="center",
            va="bottom", bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    ax.set_aspect("equal")
    ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.6, 2.6)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fl.save(fig, "mdl-cal-gradient-field")


def fig_taylor_quadratic():
    """The second-order Taylor model hugs the surface near the base point. The
    surface $z=x\\,e^{-x^2-y^2}$ (blue) and its quadratic Taylor approximation at
    $(-1,0)$ (orange), which matches value, gradient, and curvature there. The
    gradient $(-e^{-1},0)$ and Hessian $2e^{-1}\\mathbf{I}$ are exact."""
    f = lambda X, Y: X * np.exp(-X ** 2 - Y ** 2)
    x0, y0 = -1.0, 0.0
    e1 = np.exp(-1.0)
    # exact at (-1,0): f=-e^{-1}, grad=(-e^{-1},0), Hessian = 2 e^{-1} I
    q = lambda X, Y: (-e1) + (-e1) * (X - x0) + e1 * ((X - x0) ** 2 + (Y - y0) ** 2)

    fig = plt.figure(figsize=(6.0, 4.8))
    ax = fig.add_subplot(projection="3d")
    ax.computed_zorder = False
    gx, gy = np.linspace(-2.2, 0.7, 60), np.linspace(-1.4, 1.4, 60)
    X, Y = np.meshgrid(gx, gy)
    ax.plot_surface(X, Y, f(X, Y), color=BLUE, alpha=0.45, linewidth=0,
                    antialiased=True, zorder=2)
    # the quadratic as an orange wireframe over the blue surface: near the base
    # point its wires visibly lie *on* the surface, farther out they lift off
    # (two translucent fills would blend into one brown patch there)
    px, py = np.linspace(-1.8, -0.2, 9), np.linspace(-0.8, 0.8, 9)
    PX, PY = np.meshgrid(px, py)
    ax.plot_wireframe(PX, PY, q(PX, PY), color=ORANGE, lw=1.0, zorder=4)
    # base point on top of both surfaces; a leader drops to its (x, y)
    # footprint on the floor of the box, where the label sits
    zmin = -0.8
    ax.scatter([x0], [y0], [f(x0, y0)], color="black", s=42, zorder=6,
               depthshade=False)
    ax.plot([x0, x0], [y0, y0], [f(x0, y0), zmin], ":", color=INK,
            lw=1.2, zorder=6)
    ax.scatter([x0], [y0], [zmin], color=INK, s=14, zorder=6,
               depthshade=False)
    ax.text(x0, y0, zmin - 0.04, r"$(-1,0)$", fontsize=14, ha="center",
            va="top", zorder=6)
    ax.set_zlim(zmin, 0.5)
    ax.set_xlabel("$x$", fontsize=15, labelpad=-6)
    ax.set_ylabel("$y$", fontsize=15, labelpad=-6)
    ax.set_zlabel("$z$", fontsize=15, labelpad=-6)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=22, azim=-58)
    fl.save(fig, "mdl-cal-taylor-quadratic")


def fig_tangent_plane():
    """Tangent plane and gradient geometry. The surface $z=f(x,y)$ (blue) with its
    tangent plane at a point (orange); the augmented normal $[\nabla f,-1]$ of the
    plane drawn from the point; on the base plane, the level curves of $f$ (gray,
    the one through the point in INK) and $\nabla f$ (orange) crossing it at a
    right angle -- dropping the height coordinate of the normal leaves $\nabla f$."""
    f = lambda X, Y: 0.35 * X ** 2 + 0.6 * Y ** 2
    fx, fy = lambda X, Y: 0.70 * X, lambda X, Y: 1.20 * Y
    x0, y0 = 1.35, -0.5      # gradient leans toward +x so its arrow reads long
    z0, gx0, gy0 = f(x0, y0), fx(x0, y0), fy(x0, y0)
    plane = lambda X, Y: z0 + gx0 * (X - x0) + gy0 * (Y - y0)

    fig = plt.figure(figsize=(6.2, 5.0))
    ax = fig.add_subplot(projection="3d")
    ax.computed_zorder = False
    zbase = -0.2

    # clean level-set ellipses on the base plane (0.35x^2+0.6y^2=c => ellipse),
    # drawn as full parametric curves so they never break into dashes; the
    # level curve through the base point (c = z0) is drawn in INK: it is the
    # one the gradient arrow is normal to
    th = np.linspace(0, 2 * np.pi, 200)
    for c in [0.2, z0, 1.1]:  # all inside the drawn x, y range
        ax.plot(np.sqrt(c / 0.35) * np.cos(th), np.sqrt(c / 0.6) * np.sin(th),
                zbase, color=INK if c == z0 else GRAY,
                lw=1.5 if c == z0 else 0.9, zorder=1)

    gx, gy = np.linspace(-1.8, 1.8, 60), np.linspace(-1.8, 1.8, 60)
    X, Y = np.meshgrid(gx, gy)
    ax.plot_surface(X, Y, f(X, Y), color=BLUE, alpha=0.30, linewidth=0, zorder=2)
    # a tangent patch of half-width 1.0 around the point, opaque enough to read
    # as a plane
    px, py = np.linspace(x0 - 1.0, x0 + 1.0, 20), np.linspace(y0 - 1.0, y0 + 1.0, 20)
    PX, PY = np.meshgrid(px, py)
    ax.plot_surface(PX, PY, plane(PX, PY), color=ORANGE, alpha=0.65,
                    linewidth=0, zorder=3)
    ax.scatter([x0], [y0], [z0], color="black", s=34, zorder=7,
               depthshade=False)
    # the upward normal -[grad f, -1] = [-grad f, 1] of the plane, from the
    # point (the downward one projects onto the base-plane gradient arrow and
    # would hide it); the box aspect below is set to the data ranges so the
    # normal reads as perpendicular to the plane
    n = np.array([-gx0, -gy0, 1.0]); n = n / np.linalg.norm(n) * 1.1
    ax.quiver(x0, y0, z0, n[0], n[1], n[2], color=INK, lw=2.2,
              arrow_length_ratio=0.18, zorder=6)
    ax.text(x0 + n[0] + 0.05, y0 + n[1], z0 + n[2] + 0.08,
            r"$[-\nabla f,\,1]$", color=INK, fontsize=14, ha="left",
            va="bottom", zorder=7)
    # drop from the point to its shadow on the base plane, where the arrow starts
    ax.plot([x0, x0], [y0, y0], [z0, zbase], ":", color=INK, lw=1.2, zorder=6)

    # the gradient on the base plane, normal to the level set, pointing uphill;
    # rooted under the base point and long enough to read as a vector
    g = np.array([gx0, gy0]); g = g / np.linalg.norm(g) * 1.0
    ax.quiver(x0, y0, zbase + 0.01, g[0], g[1], 0.0, color=ORANGE, lw=2.6,
              arrow_length_ratio=0.18, zorder=5)
    # label placed beyond the arrowhead, nudged off the shaft so they don't clash
    ax.text(x0 + g[0] - 0.30, y0 + g[1] - 0.34, zbase, r"$\nabla f$",
            color=ORANGE, fontsize=14, ha="left", va="top", zorder=6)
    ax.set_xlim(-1.8, 1.8); ax.set_ylim(-1.8, 1.8); ax.set_zlim(zbase, 3.1)
    ax.set_box_aspect((3.6, 3.6, 3.3))
    ax.set_xlabel("$x$", fontsize=15, labelpad=-6)
    ax.set_ylabel("$y$", fontsize=15, labelpad=-6)
    ax.set_zlabel("$z$", fontsize=15, labelpad=-6)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=30, azim=-60)
    fl.save(fig, "mdl-cal-tangent-plane")


# =========================================================================== #
# Matrix calculus & automatic differentiation                                 #
# =========================================================================== #

def fig_jacobian_ellipse():
    """Up close, a differentiable map is its Jacobian.  The running map
    $\\mathbf f(x,y)=(x+\\sin y,\\ y+x^2/2)$ carries two circles of radius
    $0.35$ and $0.175$ around $\\mathbf x_0=(0.5,0.5)$ (left) into the output
    plane (right), where the true images (blue) sit on the ellipses
    $\\mathbf f(\\mathbf x_0)+\\mathbf J(\\mathbf x_0)\\boldsymbol\\delta$
    (orange, dashed).  Halving the radius cuts the largest gap between image
    and ellipse from 0.061 to 0.015: the remainder is $O(r^2)$."""
    F = lambda x, y: (x + np.sin(y), y + 0.5 * x ** 2)
    x0, y0 = 0.5, 0.5
    J = np.array([[1.0, np.cos(y0)], [x0, 1.0]])    # [[1, 0.8776], [0.5, 1]]
    fx0 = np.array(F(x0, y0))                       # ~ (0.979, 0.625)
    radii, widths = (0.35, 0.175), (2.2, 1.4)
    hw = 0.45                                       # half-width of the grid
    th = np.linspace(0.0, 2 * np.pi, 400)
    lines = np.linspace(-hw, hw, 7)
    ts = np.linspace(-hw, hw, 60)

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.0, 4.2))

    # --- (a) input plane: light grid + two circles of perturbations ---
    for c in lines:
        axa.plot(x0 + ts, np.full_like(ts, y0 + c), color=LIGHT, lw=0.8,
                 zorder=1)
        axa.plot(np.full_like(ts, x0 + c), y0 + ts, color=LIGHT, lw=0.8,
                 zorder=1)
    for r, lw in zip(radii, widths):
        axa.plot(x0 + r * np.cos(th), y0 + r * np.sin(th), color=BLUE, lw=lw,
                 zorder=3)
    axa.plot([x0], [y0], "o", color="black", ms=5, zorder=4)
    axa.text(x0 + 0.035, y0 - 0.035, r"$\mathbf{x}_0$", fontsize=FS,
             ha="left", va="top", zorder=4)
    axa.set_xlim(x0 - hw - 0.04, x0 + hw + 0.04)
    axa.set_ylim(y0 - hw - 0.04, y0 + hw + 0.04)
    axa.set_aspect("equal")
    axa.axis("off")

    # --- (b) output plane: warped grid, true images vs. Jacobian ellipses ---
    gxs, gys = [], []
    for c in lines:
        for gx, gy in (F(x0 + ts, y0 + c), F(np.full_like(ts, x0 + c), y0 + ts)):
            axb.plot(gx, gy, color=LIGHT, lw=0.8, zorder=1)
            gxs.append(gx); gys.append(gy)
    for r, lw in zip(radii, widths):
        circ = np.stack([x0 + r * np.cos(th), y0 + r * np.sin(th)])
        img = np.stack(F(circ[0], circ[1]))          # the true image f(circle)
        ell = fx0[:, None] + J @ (circ - [[x0], [y0]])  # the Jacobian's model
        axb.plot(img[0], img[1], color=BLUE, lw=lw, zorder=3,
                 label=r"$\mathbf{f}(\mathrm{circle})$" if r == radii[0] else None)
        axb.plot(ell[0], ell[1], "--", color=ORANGE, lw=lw - 0.2, zorder=4,
                 label=(r"$\mathbf{f}(\mathbf{x}_0)+\mathbf{J}\,\boldsymbol{\delta}$"
                        if r == radii[0] else None))
    axb.plot([fx0[0]], [fx0[1]], "o", color="black", ms=5, zorder=5)
    axb.annotate(r"$\mathbf{f}(\mathbf{x}_0)$", xy=(fx0[0], fx0[1]),
                 xytext=(fx0[0] + 0.42, fx0[1] - 0.42), fontsize=FS, color=INK,
                 ha="left", va="center", zorder=6,
                 arrowprops=dict(arrowstyle="-", color=INK, lw=0.8,
                                 linestyle=":", shrinkA=2, shrinkB=4))
    gx, gy = np.concatenate(gxs), np.concatenate(gys)
    cx, cy = 0.5 * (gx.min() + gx.max()), 0.5 * (gy.min() + gy.max())
    half = 0.5 * max(gx.max() - gx.min(), gy.max() - gy.min()) + 0.03
    axb.set_xlim(cx - half, cx + half)
    axb.set_ylim(cy - half, cy + half)
    leg = axb.legend(loc="upper left", fontsize=FS, frameon=False,
                     handlelength=1.8, borderaxespad=0.0)
    for t in leg.get_texts():
        t.set_color(INK)
    axb.set_aspect("equal")
    axb.axis("off")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.08)
    fl.save(fig, "mdl-cal-jacobian-ellipse")


def fig_fwd_vs_rev():
    """Forward- vs reverse-mode AD on a chain x -> a -> b -> ell.  Both evaluate
    left-to-right (gray); forward mode carries a tangent (JVP) in the same
    direction, one input direction per sweep; reverse mode records the chain,
    then carries an adjoint (VJP) right-to-left, one output direction per
    sweep, which for a scalar loss is the whole gradient."""
    chain, xpos = ["x", "a", "b", r"\ell"], [0.0, 1.5, 3.0, 4.5]
    fig, (axt, axb) = plt.subplots(2, 1, figsize=(7.4, 3.7))

    def draw(ax, direction, dual_color, dual_label):
        centers = [(xp, 0.0) for xp in xpos]
        for c0, c1 in zip(centers, centers[1:]):       # evaluation edges
            edge(ax, c0, c1)
        for lab, c in zip(chain, centers):
            node(ax, c, f"${lab}$", r=0.26)
        # both panels' arcs bulge the same way (downward toward the chain);
        # the reverse panel travels right-to-left, so its rad sign is flipped.
        rad = -0.3 if direction == "fwd" else 0.3
        for c0, c1 in zip(centers, centers[1:]):       # derivative propagation
            a, b = (c0, c1) if direction == "fwd" else (c1, c0)
            ax.annotate("", xy=(b[0], b[1] + 0.40), xytext=(a[0], a[1] + 0.40),
                        arrowprops=dict(arrowstyle="->", color=dual_color,
                                        lw=1.7, connectionstyle=f"arc3,rad={rad}"))
        ax.text(2.25, 0.84, dual_label, ha="center", color=INK, fontsize=FS)
        ax.set_xlim(-0.9, 5.4); ax.set_ylim(-0.55, 1.08)
        ax.set_aspect("equal"); ax.axis("off")

    draw(axt, "fwd", BLUE,
         r"forward (JVP): tangents $\dot x\to\dot a\to\dot b\to\dot\ell$ "
         "travel with the values")
    draw(axb, "rev", ORANGE,
         r"reverse (VJP): adjoints $\bar\ell=1\to\bar b\to\bar a\to\bar x$ "
         "travel back afterwards")
    fig.subplots_adjust(hspace=0.05)
    fl.save(fig, "mdl-cal-fwd-vs-rev")


def fig_tape_dag():
    """The tape of $y=r\\cdot r$ with $r=uv+u$ at $u=2$, $v=-3$.  Each box
    shows its forward value; the orange label beside it is its adjoint
    $\\bar v=\\partial y/\\partial v$ from the backward sweep.  The graph is a
    diamond, not a chain: $u$ fans out to $t=uv$ and to $r=t+u$, and $r$ feeds
    both arguments of $y=r\\cdot r$ (the doubled edge), so the adjoints of $r$
    and $u$ arrive as sums of two contributions."""
    fig, ax = plt.subplots(figsize=(9.4, 3.5))
    hh = 0.3                                    # box half-height
    pos = {"u": (0.0, 2.0), "v": (0.0, 0.4), "t": (2.9, 1.2),
           "r": (5.6, 2.0), "y": (8.6, 2.0)}
    hw = {"u": 0.62, "v": 0.7, "t": 1.05, "r": 1.2, "y": 1.2}

    def box(k, label, color=LIGHT, ec="black"):
        cx, cy = pos[k]; w = hw[k]
        ax.add_patch(FancyBboxPatch((cx - w, cy - hh), 2 * w, 2 * hh,
                     boxstyle="round,pad=0.02,rounding_size=0.12",
                     facecolor=color, edgecolor=ec, lw=1.2, zorder=3))
        ax.text(cx, cy, label, ha="center", va="center", fontsize=FS, zorder=4)

    def port(k, c):                             # a named point on a box's border
        cx, cy = pos[k]; w = hw[k]
        dx = {"E": w, "W": -w, "NE": w, "SE": w, "NW": -w, "SW": -w}[c]
        dy = {"E": 0, "W": 0, "NE": hh, "SE": -hh, "NW": hh, "SW": -hh}[c]
        return (cx + dx, cy + dy)

    def link(p, q, rad=0.0):                    # arrowhead lands on the target port
        ax.annotate("", xy=q, xytext=p, zorder=2,
                    arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.4,
                                    shrinkA=0, shrinkB=0, mutation_scale=12,
                                    connectionstyle=f"arc3,rad={rad}"))

    def adjoint(k, text, side):                 # orange adjoint above/below a box
        cx, cy = pos[k]
        dy = hh + 0.24 if side == "above" else -(hh + 0.24)
        ax.text(cx, cy + dy, text, ha="center", va="center", fontsize=FS,
                color=ORANGE, zorder=4)

    # Each edge leaves a corner of its source and enters a corner of its
    # target, so no arrow cuts through a box.
    link(port("u", "SE"), port("t", "NW"))      # u -> t
    link(port("v", "NE"), port("t", "SW"))      # v -> t
    link(port("t", "NE"), port("r", "SW"))      # t -> r
    link(port("u", "E"),  port("r", "W"))       # u -> r, above the t box
    # r feeds BOTH arguments of y = r*r: two parallel edges
    for dy in (+0.5 * hh, -0.5 * hh):
        (px, py), (qx, qy) = port("r", "E"), port("y", "W")
        link((px, py + dy), (qx, qy + dy))

    box("u", "$u=2$"); box("v", "$v=-3$")
    box("t", r"$t=uv=-6$", color=fl.T.BLUE.tint, ec=BLUE)
    box("r", r"$r=t+u=-4$", color=fl.T.BLUE.tint, ec=BLUE)
    box("y", r"$y=r\cdot r=16$", color=fl.T.ORANGE.tint, ec=ORANGE)
    adjoint("y", r"$\bar y=1$", "above")
    adjoint("r", r"$\bar r=\bar y\,r+\bar y\,r=-8$", "above")
    adjoint("u", r"$\bar u=\bar r+\bar t\,v=16$", "above")
    adjoint("t", r"$\bar t=\bar r=-8$", "below")
    adjoint("v", r"$\bar v=\bar t\,u=-16$", "below")
    ax.set_xlim(-0.9, 9.9); ax.set_ylim(-0.35, 2.75)
    ax.set_aspect("equal"); ax.axis("off")
    fl.save(fig, "mdl-cal-tape-dag")


def fig_best_parabola():
    """The best local quadratic beats the tangent line.  For $f=\\sin$ at the
    base point $x_0$, the tangent matches value and slope; the osculating
    parabola $g(x)=\\sin x_0+\\cos x_0\\,(x-x_0)-\\frac12\\sin x_0\\,(x-x_0)^2$
    also matches curvature and visibly hugs the curve over a wider window.
    All three curves are the genuine Taylor truncations of $\\sin$."""
    x0 = 0.9
    s0, c0 = np.sin(x0), np.cos(x0)
    f = np.sin
    tangent = lambda t: s0 + c0 * (t - x0)
    parab = lambda t: s0 + c0 * (t - x0) - 0.5 * s0 * (t - x0) ** 2
    xs = np.linspace(x0 - 2.1, x0 + 2.3, 500)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.axhline(0, color=INK, lw=1.0, zorder=1)
    ax.plot(xs, f(xs), color=BLUE, lw=2.6, zorder=4, label=r"$f(x)=\sin x$")
    ax.plot(xs, tangent(xs), "--", color=GRAY, lw=1.8, zorder=2,
            label="tangent (matches value, slope)")
    ax.plot(xs, parab(xs), "--", color=ORANGE, lw=2.0, zorder=3,
            label="best parabola (matches curvature too)")
    ax.plot([x0], [s0], "o", color="black", ms=5, zorder=6)
    ax.text(x0 + 0.06, s0 - 0.13, r"$x_0$", fontsize=FS, ha="left", va="top")
    ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(-1.65, 1.9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="lower right", fontsize=FS, frameon=False)
    # the zero line is the x-axis; an L-frame below it would be a second one
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    fl.save(fig, "mdl-cal-best-parabola")


def fig_lagrange_tangency():
    """Lagrange multipliers as tangency, on the worked example of the section:
    minimize $f(x,y)=x^2+y^2$ (gray circles) subject to $g(x,y)=xy=1$ (blue
    hyperbola).  At the constrained optimum $(1,1)$ the circle $x^2+y^2=2$
    (orange) touches the constraint and the two gradients align,
    $\nabla f=(2,2)=2\nabla g$ with $\nabla g=(1,1)$.  At a non-optimal feasible
    point the gradients disagree, so $\nabla f$ has a component along the
    constraint.  All arrows are the true gradients, normalized to display
    length."""
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    th = np.linspace(0, 2 * np.pi, 300)
    # contours of f: circles; the sqrt(2)-circle is the one through the optimum
    for rad, lw in [(0.8, 1.0), (np.sqrt(2.0), 1.6), (2.1, 1.0), (2.6, 1.0)]:
        col = ORANGE if abs(rad - np.sqrt(2.0)) < 1e-9 else GRAY
        ax.plot(rad * np.cos(th), rad * np.sin(th), color=col,
                lw=lw, zorder=2, alpha=0.9 if col is ORANGE else 0.8)
    ax.text(0.08, 1.48, r"$x^2+y^2=2$", color=ORANGE, fontsize=FS,
            ha="center", va="bottom")
    # the constraint xy = 1 (first-quadrant branch)
    cx = np.linspace(0.34, 3.1, 300)
    ax.plot(cx, 1.0 / cx, color=BLUE, lw=2.4, zorder=3)
    ax.text(0.62, 2.55, r"$xy=1$", color=BLUE, fontsize=FS, ha="left")

    def grads(px, py):
        gf = np.array([2 * px, 2 * py])          # true gradient of f
        gg = np.array([py, px])                  # true gradient of g = xy
        # the nabla-g arrow is a little shorter so its head stays inside the
        # outermost contour instead of sitting on it
        for g, col, lab, off, scale in [
                (gf, ORANGE, r"$\nabla f$", (0.10, 0.16), 0.62),
                (gg, GREEN, r"$\nabla g$", (-0.17, -0.02), 0.46)]:
            u = g / np.linalg.norm(g) * scale
            fl.arrow(ax, (px, py), (px + u[0], py + u[1]), color=col,
                     lw=2.0, mut=12)
            ax.text(px + u[0] + off[0], py + u[1] + off[1], lab, color=col,
                    fontsize=FS, ha="center", va="center")
        ax.plot([px], [py], "o", color="black", ms=5, zorder=6)

    # optimum (1,1): gradients parallel -- one arrow drawn shorter so both are
    # visible along the same ray, each labelled with its true value
    px, py = 1.0, 1.0
    gfu = np.array([1.0, 1.0]) / np.sqrt(2.0)
    fl.arrow(ax, (px, py), (px + 0.95 * gfu[0], py + 0.95 * gfu[1]),
             color=ORANGE, lw=2.2, mut=13)
    fl.arrow(ax, (px, py), (px + 0.50 * gfu[0], py + 0.50 * gfu[1]),
             color=GREEN, lw=2.2, mut=11)
    ax.plot([px], [py], "o", color="black", ms=5, zorder=6)
    ax.text(1.74, 1.70, r"$\nabla f=(2,2)$", fontsize=FS, ha="left",
            va="bottom", color=ORANGE,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    ax.text(1.27, 1.40, r"$\nabla g$", fontsize=FS, ha="right",
            va="bottom", color=GREEN)
    ax.text(0.93, 0.93, r"$(1,1)$", fontsize=FS, ha="right", va="top",
            color=INK, bbox=dict(facecolor="white", edgecolor="none", pad=1.0))

    # a non-optimal feasible point: gradients visibly misaligned
    grads(2.2, 1.0 / 2.2)
    ax.text(2.2, 0.28, "not optimal", fontsize=FS, ha="center", va="top",
            color=INK, bbox=dict(facecolor="white", edgecolor="none", pad=1.5))

    ax.set_aspect("equal")
    ax.set_xlim(-0.4, 3.4); ax.set_ylim(-0.4, 3.0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fl.save(fig, "mdl-cal-lagrange-tangency")


def fig_jacobian_shapes():
    """The shape of the Jacobian dictates the cheap mode.  Three glyphs: a wide
    $1\\times n$ row (a scalar loss; one VJP pass), a tall $m\\times 1$ column
    (one input; one JVP pass), and a full $m\\times n$ matrix, which costs
    $\\min(m,n)$ passes with the cheaper mode.  The pass counts under each glyph are the
    entire forward-vs-reverse cost model."""
    fig, ax = plt.subplots(figsize=(9.6, 3.4))
    cell = 0.34

    def grid(x0, y0, rows, cols, color, alpha=0.55):
        for i in range(rows):
            for j in range(cols):
                ax.add_patch(Rectangle((x0 + j * cell, y0 - (i + 1) * cell),
                                       cell, cell, facecolor=color,
                                       edgecolor="black", lw=0.7, zorder=3,
                                       alpha=alpha))

    n, m = 8, 5
    top = 2.15
    # wide row: 1 x n  (reverse mode, one VJP)
    note_y = top - m * cell - 0.25        # one baseline for all three notes
    grid(0.0, top - 0.7, 1, n, ORANGE)
    ax.text(0.5 * n * cell, top + 0.06, r"$1\times n$ (scalar loss)",
            ha="center", fontsize=FS)
    ax.text(0.5 * n * cell, note_y,
            "one VJP\n= 1 backward pass", ha="center", va="top", fontsize=FS,
            color=ORANGE)
    # tall column: m x 1  (forward mode, one JVP)
    x1 = 4.2
    grid(x1, top, m, 1, BLUE)
    ax.text(x1 + 0.5 * cell, top + 0.06, r"$m\times 1$ (one input)",
            ha="center", fontsize=FS)
    ax.text(x1 + 0.5 * cell, note_y,
            "one JVP\n= 1 forward pass", ha="center", va="top", fontsize=FS,
            color=BLUE)
    # full matrix: m x n
    x2 = 6.4
    grid(x2, top, m, n, LIGHT, alpha=1.0)   # neutral: either mode applies
    ax.text(x2 + 0.5 * n * cell, top + 0.06, r"$m\times n$ (full Jacobian)",
            ha="center", fontsize=FS)
    ax.text(x2 + 0.5 * n * cell, note_y,
            r"$\min(m,n)$ passes" + "\nwith the cheaper mode",
            ha="center", va="top", fontsize=FS, color=INK)

    ax.set_xlim(-0.4, 9.6); ax.set_ylim(-0.65, 2.45)
    ax.set_aspect("equal"); ax.axis("off")
    fl.save(fig, "mdl-cal-jacobian-shapes")


def fig_checkpointing():
    """Gradient checkpointing as a timeline.  Top track: plain reverse mode
    stores every one of the $T$ recorded values until the backward sweep
    consumes them ($T$ memory).  Bottom track: checkpointing keeps every
    $s$-th value with $s=\\sqrt T$ and, when the backward sweep reaches a
    segment, recomputes that segment forward from its checkpoint
    ($T/s+s=2\\sqrt T$ memory, about one extra forward pass in time)."""
    T, K = 16, 4                                  # tape length and spacing
    cell, gap = 0.5, 0.06
    fig, ax = plt.subplots(figsize=(9.8, 3.0))

    def track(y, kept, label):
        for i in range(T):
            filled = kept(i)
            ax.add_patch(Rectangle((i * (cell + gap), y), cell, cell,
                                   facecolor=BLUE if filled else "white",
                                   alpha=0.55 if filled else 1.0,
                                   edgecolor="black", lw=0.8, zorder=3))
        ax.text(-0.25, y + cell / 2, label, ha="right", va="center",
                fontsize=FS)

    width = T * (cell + gap) - gap
    y_top, y_bot = 2.05, 0.65
    track(y_top, lambda i: True, "store all $T$\nvalues")
    track(y_bot, lambda i: i % K == 0, "store $\\sqrt{T}$\ncheckpoints")

    # forward pass arrow above the top track
    fl.arrow(ax, (0.0, y_top + cell + 0.30), (width, y_top + cell + 0.30),
             color=GRAY, lw=1.6, mut=12)
    ax.text(width / 2, y_top + cell + 0.40, "forward pass (steps $1$ to $T$)",
            ha="center", va="bottom", fontsize=FS, color=INK)

    # the backward sweep has reached the last segment of the bottom track: it
    # runs right-to-left just above that track, after the segment is
    # recomputed forward from its checkpoint (orange, below the track)
    seg0 = 12 * (cell + gap)
    fl.arrow(ax, (width, y_bot + cell + 0.25), (seg0, y_bot + cell + 0.25),
             color=GRAY, lw=1.6, mut=12)
    ax.text(seg0 - 0.15, y_bot + cell + 0.25, "backward sweep",
            ha="right", va="center", fontsize=FS, color=INK)
    fl.arrow(ax, (seg0 + cell / 2, y_bot - 0.27),
             (width - cell / 2 + 0.15, y_bot - 0.27), color=ORANGE, lw=2.0,
             mut=12)
    ax.text(seg0 - 0.15, y_bot - 0.27,
            "recompute the segment from its checkpoint",
            ha="right", va="center", fontsize=FS, color=INK)

    # memory annotations on the right
    ax.text(width + 0.35, y_top + cell / 2, r"$T$ memory", fontsize=FS,
            va="center", color=INK)
    ax.text(width + 0.35, y_bot + cell / 2,
            r"$2\sqrt{T}$ memory," + "\none extra forward pass", fontsize=FS,
            va="center", color=INK)

    ax.set_xlim(-2.1, width + 3.3); ax.set_ylim(0.05, 3.15)
    ax.set_aspect("equal"); ax.axis("off")
    fl.save(fig, "mdl-cal-checkpointing")


def fig_signed_area():
    """Signed area: one curve crossing the axis, with the lobes where $f>0$
    counted positive (blue) and the lobe where $f<0$ counted negative (orange),
    plus the direction arrow that reversing the limits negates the total.  The
    curve and the shaded lobes are computed, not sketched."""
    f = lambda t: np.sin(1.4 * t) * np.exp(-0.12 * t)
    a, b = 0.25, 6.4
    xs = np.linspace(-0.2, 7.0, 700)
    span = np.linspace(a, b, 700)

    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    ax.plot([xs[0], xs[-1]], [0, 0], color=INK, lw=0.9, zorder=1)
    ax.plot(xs, f(xs), color=BLUE, lw=2.4, zorder=4)
    pos, neg = f(span) >= 0, f(span) < 0
    ax.fill_between(span, 0, f(span), where=pos, color=BLUE, alpha=0.25, lw=0)
    ax.fill_between(span, 0, f(span), where=neg, color=ORANGE, alpha=0.30, lw=0)
    ax.text(1.15, 0.34, r"$+$", fontsize=17, ha="center", color=INK)
    ax.text(3.45, -0.30, r"$-$", fontsize=17, ha="center", color=INK)
    ax.text(5.65, 0.22, r"$+$", fontsize=17, ha="center", color=INK)
    for x, lab in [(a, "$a$"), (b, "$b$")]:
        ax.plot([x, x], [0, f(x)], ":", color=GRAY, lw=1.0, zorder=2)
        ax.plot([x], [0], "o", color="black", ms=4, zorder=5)
        ax.text(x, -0.09, lab, ha="center", va="top", fontsize=FS)
    # direction arrows under the axis (below the curve's lowest dip):
    # integrating a->b, and the reversed b->a, which negates the total
    fl.arrow(ax, (a, -1.02), (b, -1.02), color=INK, lw=1.6, mut=12)
    ax.text(-0.05, -1.02, r"$\int_a^b f$", ha="right", va="center",
            fontsize=FS, color=INK)
    fl.arrow(ax, (b, -1.52), (a, -1.52), color=ORANGE, lw=1.6, mut=12)
    ax.text(6.55, -1.52, r"$\int_b^a f = -\int_a^b f$", ha="left",
            va="center", fontsize=FS, color=INK)
    ax.set_xlim(-0.9, 9.2); ax.set_ylim(-1.8, 1.0)
    ax.axis("off")
    fl.save(fig, "mdl-cal-signed-area")


# =========================================================================== #
# Driver                                                                      #
# =========================================================================== #

FIGURES = [
    # single-variable calculus
    fig_zoom_sequence,
    fig_secant_to_tangent,
    fig_gd_step,
    fig_descent_lemma,
    fig_relu_corner,
    fig_mvt,
    fig_smooth_not_analytic,
    fig_best_parabola,
    # integral calculus
    fig_riemann,
    fig_sub_area,
    fig_signed_area,
    fig_rect_trans,
    fig_bell_surface,
    fig_sum_order,
    fig_cov_jacobian,
    # multivariable calculus (chain-rule graphs + gradient geometry)
    fig_chain_net1,
    fig_chain_net2,
    fig_gradient_field,
    fig_taylor_quadratic,
    fig_tangent_plane,
    fig_lagrange_tangency,
    # matrix calculus & autodiff
    fig_jacobian_ellipse,
    fig_fwd_vs_rev,
    fig_tape_dag,
    fig_jacobian_shapes,
    fig_checkpointing,
]


def main():
    # Only verify the figures this script writes (the shared module also tracks
    # the Linear Algebra figures via the same WRITTEN list, which we don't run).
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
