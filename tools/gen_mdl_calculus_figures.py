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


# --------------------------------------------------------------------------- #
# Small chapter-local helper for the schematic computational graphs.          #
# --------------------------------------------------------------------------- #

def node(ax, center, label, r=0.30, color=BLUE):
    """Draw a labelled circular node (light-blue fill, dark edge)."""
    cx, cy = center
    ax.add_patch(plt.Circle((cx, cy), r, facecolor=LIGHT, edgecolor="black",
                            lw=1.2, zorder=3))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=11, zorder=4)


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

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4))
    half_widths = [0.6, 0.06, 0.006]
    titles = [r"$\pm 0.6$", r"$\pm 0.06$", r"$\pm 0.006$"]
    for ax, hw, title in zip(axes, half_widths, titles):
        xs = np.linspace(x0 - hw, x0 + hw, 400)
        ax.plot(xs, f(xs), color=BLUE, lw=2.2, zorder=3)
        # the fixed tangent line at x0
        ax.plot(xs, y0 + slope * (xs - x0), "--", color=ORANGE, lw=1.6, zorder=2)
        ax.plot([x0], [y0], "o", color="black", ms=4, zorder=4)
        ax.set_title(f"zoom {title}", fontsize=11)
        ax.set_xticks([x0]); ax.set_xticklabels(["$x_0$"])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fl.save(fig, "mdl-cal-zoom-sequence")


def _second_der_panels(name, sign):
    """Shared layout for the three second-derivative figures: f'' (constant),
    f' (line through 0), f (parabola), side by side.  ``sign`` selects +/-/0
    curvature.  All curves are the genuine derivatives of one another."""
    # f(x) = (sign/2) x^2  =>  f'(x) = sign*x,  f''(x) = sign (constant)
    x = np.linspace(-2.0, 2.0, 400)
    if sign == 0:
        f2 = np.zeros_like(x)
        f1 = np.ones_like(x) * 1.0          # constant slope
        f0 = 1.0 * x                         # straight line
        c2 = GRAY
    else:
        f2 = np.ones_like(x) * sign
        f1 = sign * x
        f0 = 0.5 * sign * x ** 2
        c2 = GREEN if sign > 0 else ORANGE

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0))
    panels = [
        (f2, r"$f^{(2)}(x)$", c2),
        (f1, r"$f^{(1)}(x)$", BLUE),
        (f0, r"$f(x)$", BLUE),
    ]
    for ax, (y, title, col) in zip(axes, panels):
        ax.axhline(0, color="black", lw=0.8)
        ax.axvline(0, color="black", lw=0.8)
        ax.plot(x, y, color=col, lw=2.4)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-2.1, 2.1)
        pad = 0.3
        ax.set_ylim(min(y.min(), -0.2) - pad, max(y.max(), 0.2) + pad)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fl.save(fig, name)


def fig_pos_second():
    """f'' > 0 constant -> f' increasing through 0 -> f has a minimum."""
    _second_der_panels("mdl-cal-pos-second", +1)


def fig_neg_second():
    """f'' < 0 constant -> f' decreasing through 0 -> f has a maximum."""
    _second_der_panels("mdl-cal-neg-second", -1)


def fig_zero_second():
    """f'' = 0 -> f' constant -> f is a straight line."""
    _second_der_panels("mdl-cal-zero-second", 0)


def fig_secant_to_tangent():
    """Secants through (x, f(x)) and (x+eps, f(x+eps)) rotating into the tangent
    as eps -> 0; the limiting slope is f'(x).  Real cubic + real slopes."""
    f = lambda t: 0.15 * t ** 3 - 0.2 * t ** 2 + 0.3 * t + 0.6
    df = lambda t: 0.45 * t ** 2 - 0.4 * t + 0.3
    x0, xs = 1.0, np.linspace(-0.3, 2.7, 400)
    y0 = f(x0)

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(xs, f(xs), color=BLUE, lw=2.4, zorder=3)
    for k, eps in enumerate([1.4, 0.8, 0.4]):          # secants, eps shrinking
        slope = (f(x0 + eps) - y0) / eps
        ax.plot(xs, y0 + slope * (xs - x0), "-", color=GRAY,
                lw=1.0, alpha=0.35 + 0.15 * k, zorder=2)
        ax.plot([x0 + eps], [f(x0 + eps)], "o", color=GRAY, ms=4, zorder=4)
    ax.plot(xs, y0 + df(x0) * (xs - x0), "--", color=ORANGE, lw=2.0, zorder=5)
    ax.plot([x0], [y0], "o", color="black", ms=5, zorder=6)
    ax.text(2.15, 1.02, r"slope $f'(x)$", color=ORANGE,
            ha="left", va="top", fontsize=12)
    ax.text(x0 + 0.30, y0 - 0.16, r"$(x,\,f(x))$", ha="left", va="top",
            fontsize=12)
    ax.set_xlim(-0.3, 2.75); ax.set_ylim(0.1, 1.9)
    ax.set_xticks([x0]); ax.set_xticklabels(["$x$"]); ax.set_yticks([])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fl.save(fig, "mdl-cal-secant-to-tangent")


def fig_gd_step():
    """One gradient-descent step on a 1-D bowl: from x move -eta f'(x) along the
    axis and land lower on the curve; the predicted drop is ~ eta [f'(x)]^2."""
    f, df = lambda t: 0.5 * t ** 2, lambda t: t
    x0, eta = 1.7, 0.6
    x1, xs = x0 - eta * df(x0), np.linspace(-2.4, 2.4, 400)

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(xs, f(xs), color=BLUE, lw=2.4, zorder=3)
    ax.plot(xs, f(x0) + df(x0) * (xs - x0), "--", color=ORANGE, lw=1.6, zorder=2)
    ax.plot([x0, x0], [-0.18, f(x0)], ":", color=GRAY, lw=1.0)
    ax.plot([x1, x1], [-0.18, f(x1)], ":", color=GRAY, lw=1.0)
    fl.arrow(ax, (x0, -0.18), (x1, -0.18), color=GREEN, lw=2.0, mut=13)
    ax.text((x0 + x1) / 2, -0.42, r"$-\eta f'(x)$", ha="center", va="top",
            color=GREEN, fontsize=11)
    ax.plot([x0], [f(x0)], "o", color="black", ms=5, zorder=5)
    ax.plot([x1], [f(x1)], "o", color=GREEN, ms=5, zorder=5)
    ax.text(x0 + 0.22, f(x0) + 0.04, r"$x$", ha="left", va="bottom")
    ax.text(x1 - 0.08, f(x1) + 0.06, r"$x-\eta f'(x)$", ha="right", va="bottom",
            color=GREEN, fontsize=11)
    ax.annotate(r"drop $\approx \eta\,[f'(x)]^2$", xy=(x1, f(x1)),
                xytext=(1.15, 1.95), fontsize=11, color=GRAY, ha="center",
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.0,
                                connectionstyle="arc3,rad=-0.3"))
    ax.set_xlim(-2.4, 2.4); ax.set_ylim(-0.65, 2.25)
    ax.set_xticks([]); ax.set_yticks([])
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
    ax.plot([xp, xp], [-0.95, qp], ":", color=GRAY, lw=1.1, zorder=2)
    ax.plot([xp, x0], [y0, y0], ":", color=GRAY, lw=1.1, zorder=2)
    ax.plot([x0], [y0], "o", color="black", ms=5, zorder=6)
    ax.text(0.38, 0.45, r"$(x,\,f(x))$", ha="left", va="top", fontsize=10.5)
    ax.plot([xp], [qp], "o", color=ORANGE, ms=5, zorder=6)
    ax.plot([xp], [fp], "o", color=BLUE, ms=5, zorder=6)
    ax.text(xp, -1.02, r"$x-f'(x)/L$  ($\eta=1/L$)", ha="center", va="top",
            fontsize=11, color=GRAY)
    # guaranteed drop: from the starting height down to the ceiling's minimum;
    # label lifted high and left so it never crowds the rising orange parabola
    fl.arrow(ax, (xp, y0), (xp, qp), color=ORANGE, lw=1.8, mut=12)
    ax.annotate(r"guaranteed drop $f'(x)^2/2L\approx 0.341$",
                xy=(xp - 0.02, (y0 + qp) / 2), xytext=(-0.12, 1.42),
                ha="center", fontsize=11, color=ORANGE,
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.0,
                                connectionstyle="arc3,rad=0.25"))
    # the function, trapped below the ceiling, drops even further
    fl.arrow(ax, (xp, qp), (xp, fp), color=BLUE, lw=1.8, mut=12)
    ax.annotate("actual drop", xy=(xp - 0.02, (qp + fp) / 2),
                xytext=(-0.34, 0.02), ha="right", va="center", fontsize=11,
                color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.0,
                                connectionstyle="arc3,rad=-0.2"))
    ax.set_xlim(-0.95, 0.95); ax.set_ylim(-1.18, 1.72)
    ax.set_xticks([]); ax.set_yticks([])
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
    axa.axhline(0, color="black", lw=0.8, zorder=1)
    for s in np.linspace(-1.0, 1.0, 9):
        axa.plot(xs, s * xs, "-", color=ORANGE, lw=1.0, alpha=0.5, zorder=2)
    axa.plot(xs, np.abs(xs), color=BLUE, lw=2.6, zorder=4)
    axa.plot([0], [0], "o", color="black", ms=6, zorder=5)
    axa.text(0.0, 1.72, r"$|x|,\ \ \partial|x|(0)=[-1,1]$", color=BLUE,
              fontsize=12, ha="center")
    axa.set_xlim(-2.0, 2.0); axa.set_ylim(-0.55, 2.0)
    axa.set_xticks([0]); axa.set_yticks([])
    axa.spines["top"].set_visible(False); axa.spines["right"].set_visible(False)

    # right: ReLU = max(0, x), subgradient fan sweeping [0, 1]
    axb.axhline(0, color="black", lw=0.8, zorder=1)
    for s in np.linspace(0.0, 1.0, 6):
        axb.plot(xs, s * xs, "-", color=ORANGE, lw=1.0, alpha=0.5, zorder=2)
    axb.plot(xs, np.maximum(0.0, xs), color=BLUE, lw=2.6, zorder=4)
    axb.plot([0], [0], "o", color="black", ms=6, zorder=5)
    axb.text(0.0, 1.72, r"$\mathrm{ReLU}(x),\ \ \partial(0)=[0,1]$", color=BLUE,
              fontsize=12, ha="center")
    axb.set_xlim(-2.0, 2.0); axb.set_ylim(-0.55, 2.0)
    axb.set_xticks([0]); axb.set_yticks([])
    axb.spines["top"].set_visible(False); axb.spines["right"].set_visible(False)
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
    xt = np.linspace(a - 0.05, b + 0.15, 200)
    ax.plot(xt, f(xi) + slope * (xt - xi), "--", color=ORANGE, lw=2.0, zorder=4)
    # endpoint dots, each labelled snug beside its own point (not adrift below)
    for x, lab, dx in [(a, "$a$", -0.30), (b, "$b$", 0.18)]:
        ax.plot([x], [f(x)], "o", color="black", ms=5, zorder=5)
        ax.text(x + dx, f(x) - 0.06, lab, ha="center", va="top", fontsize=12)
    ax.plot([xi], [f(xi)], "o", color=ORANGE, ms=6, zorder=6)
    ax.text(xi, f(xi) + 0.14, r"$\xi$", ha="center", va="bottom", color=ORANGE,
            fontsize=13)
    ax.text((a + b) / 2 + 0.25, f((a + b) / 2) - 0.62, "secant", color=GRAY,
            fontsize=12, ha="center")
    ax.set_xlim(a - 0.35, b + 0.65); ax.set_ylim(-0.18, 1.62)
    ax.set_xticks([]); ax.set_yticks([])
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
    ax.axhline(0, color="black", lw=0.8, zorder=1)
    ax.plot(xs, f, color=BLUE, lw=2.6, zorder=3, label=r"$f(x)=e^{-1/x^2}$")
    ax.plot(xs, np.zeros_like(xs), "--", color=ORANGE, lw=2.0, zorder=2,
            label=r"Taylor series at $0\ \equiv\ 0$")
    ax.plot([0], [0], "o", color="black", ms=5, zorder=4)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-0.06, 0.5)
    ax.set_xticks([0]); ax.set_yticks([])
    ax.legend(loc="upper center", fontsize=11, frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
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
        ax.set_title(rf"$\epsilon={eps:g}$:  sum $={approx:.3f}$", fontsize=10.5)
        ax.set_ylim(0, 0.6)
        ax.set_xlim(a, b)
        ax.set_yticks([])
        ax.set_xticks([a, b]); ax.set_xticklabels(["$a$", "$b$"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.text(0.5, -0.02, rf"true area $=\frac{{1}}{{2}}\log 5 = {truth:.3f}$",
             ha="center", va="top", fontsize=10.5, color=GRAY)
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
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 1.18)
        ax.set_xlim(xs[0], xs[-1])
        ax.set_yticks([])
        ax.set_xticks([a, b]); ax.set_xticklabels(["$a$", "$b$"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    lo = xs[0]
    panel(axes[0], a, b, r"area on $[a,b]$")
    panel(axes[1], lo, b, r"$F(b)$")
    panel(axes[2], lo, a, r"$F(a)$")
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
    x0, eps = 1.05, 0.40
    du = x0                              # local stretch u'(x0) = x0 = 1.05
    h = f(u(x0))                         # the shared height f(u(x0)) = f(y0)
    y0 = u(x0)                           # 0.551 -- where the image sliver sits
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
                color=color, fontsize=12)

    # ---- (a) the x-world: narrow sliver of width epsilon on the x-axis ----
    xs = np.linspace(0.0, 3.0, 400)
    axa.plot(xs, f(u(xs)), color=BLUE, lw=2.4, zorder=3)
    axa.add_patch(Rectangle((x0, base), eps, h - base, facecolor=BLUE, alpha=0.26,
                            edgecolor=BLUE, lw=1.5, zorder=2))
    baseline(axa, 3.05)
    height_label(axa, x0, r"$f(u(x))$", BLUE)
    axa.annotate("", xy=(x0 + eps, base - 0.10), xytext=(x0, base - 0.10),
                 arrowprops=dict(arrowstyle="<->", color=GRAY, lw=1.1))
    axa.text(x0 + eps / 2, base - 0.17, r"$\epsilon$", ha="center", va="top",
             color=GRAY, fontsize=12)
    axa.set_xlim(-0.55, 3.05); axa.set_ylim(-0.40, 0.92)
    axa.set_xticks([x0]); axa.set_xticklabels(["$x$"], fontsize=12)
    axa.set_yticks([]); axa.axis("off")

    # ---- (b) the y=u-world: the IMAGE sliver, wider by u'(x), same height ----
    ys = np.linspace(0.0, 3.0, 400)
    axb.plot(ys, f(ys), color=BLUE, lw=2.4, zorder=3)
    axb.add_patch(Rectangle((y0, base), weps, h - base, facecolor=ORANGE,
                            alpha=0.30, edgecolor=ORANGE, lw=1.5, zorder=2))
    baseline(axb, 3.05)
    height_label(axb, y0, r"$f(y)$", ORANGE)
    axb.annotate("", xy=(y0 + weps, base - 0.10), xytext=(y0, base - 0.10),
                 arrowprops=dict(arrowstyle="<->", color=GRAY, lw=1.1))
    axb.text(y0 + weps / 2, base - 0.17, r"$\epsilon\,u'(x)$", ha="center",
             va="top", color=GRAY, fontsize=12)
    axb.set_xlim(-0.55, 3.05); axb.set_ylim(-0.40, 0.92)
    axb.set_xticks([y0]); axb.set_xticklabels(["$u(x)$"], fontsize=12)
    axb.set_yticks([]); axb.axis("off")

    # the map y = u(x) carrying the base interval from (a) to (b)
    fig.text(0.5, 0.585, r"$y=u(x)$", ha="center", va="center", color=GRAY,
             fontsize=12)
    fig.patches.append(
        FancyArrowPatch((0.45, 0.50), (0.55, 0.50),
                        transform=fig.transFigure, arrowstyle="-|>",
                        mutation_scale=16, color=GRAY, lw=1.5))
    fl.save(fig, "mdl-cal-rect-trans")


def fig_bell_surface():
    """The bell surface $z=e^{-x^2-y^2}$ over the box $[-2,2]^2$: the double
    integral is the volume between the surface and the base plane, approximated
    by tiling the base with $\\epsilon\\times\\epsilon$ squares and standing a
    box of height $f$ on each (a few shown near $(0.5,-0.4)$)."""
    f = lambda X, Y: np.exp(-X ** 2 - Y ** 2)
    fig = plt.figure(figsize=(6.2, 5.0))
    ax = fig.add_subplot(projection="3d")
    ax.computed_zorder = False           # we control draw order explicitly

    # faint epsilon-by-epsilon tiling of the base plane (z=0), so the Riemann
    # boxes visibly *stand on* a grid covering the domain
    eps = 0.25
    ticks = np.arange(-2.0, 2.0 + 1e-9, eps)
    for t in ticks:
        ax.plot([t, t], [-2.0, 2.0], [0, 0], color=LIGHT, lw=0.5, zorder=0)
        ax.plot([-2.0, 2.0], [t, t], [0, 0], color=LIGHT, lw=0.5, zorder=0)

    # the translucent surface first, light enough that the front boxes read
    g = np.linspace(-2.0, 2.0, 90)
    X, Y = np.meshgrid(g, g)
    ax.plot_surface(X, Y, f(X, Y), color=BLUE, alpha=0.30,
                    linewidth=0, antialiased=True, zorder=2)

    # then a single clean row of boxes marching out along the near (-y) edge,
    # each rising to the surface height at its cell centre; drawn on top so they
    # read as crisp solid columns, tops sitting on the bell.
    for cx in [0.0, 0.25, 0.5, 0.75]:
        cy = -0.5
        h = float(f(cx + eps / 2, cy + eps / 2))
        ax.bar3d(cx, cy, 0.0, eps, eps, h, color=ORANGE, alpha=0.95,
                 shade=True, edgecolor="white", linewidth=0.5, zorder=5)

    ax.text(-0.2, 0.0, 1.18, r"$z=e^{-x^2-y^2}$", fontsize=12, ha="center")
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$"); ax.set_zlabel("$z$")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_zlim(0, 1.05)
    ax.view_init(elev=26, azim=-52)
    fl.save(fig, "mdl-cal-bell-surface")


def fig_sum_order():
    """The discretization behind Fubini, on one grid.  (1) Sum the cells *within*
    each column (vertical sweep); this collapses every column to a single total,
    drawn as the orange strip of cells along the grid's own bottom edge -- same
    width, aligned to the columns, part of the same hatch.  (2) Add those column
    totals left to right (horizontal sweep along the orange strip).  Reordering a
    finite sum changes nothing."""
    nx, ny = 6, 4
    fig, ax = plt.subplots(figsize=(5.8, 4.4))

    # the grid of epsilon-by-epsilon cells (the region being summed)
    for i in range(nx):
        for j in range(ny):
            ax.add_patch(Rectangle((i, j), 1, 1, facecolor=LIGHT, alpha=0.30,
                                   edgecolor=GRAY, lw=0.8, zorder=1))

    # (1) sum WITHIN one column: highlight it blue, sweep up through it
    col = 2
    for j in range(ny):
        ax.add_patch(Rectangle((col, j), 1, 1, facecolor=BLUE, alpha=0.32,
                               edgecolor=BLUE, lw=1.3, zorder=2))
    fl.arrow(ax, (col + 0.5, 0.18), (col + 0.5, ny - 0.18), color=BLUE, lw=2.2,
             mut=15)
    ax.text(col + 0.5, ny + 0.28, "(1) sum each column", ha="center",
            va="bottom", color=BLUE, fontsize=12)

    # the column totals: an orange strip of cells along the grid's bottom edge,
    # SAME width and aligned to the columns (one cell = one column's total), so
    # it is part of the same grid rather than floating above it.
    strip_y = -1.25
    for i in range(nx):
        fc = ORANGE if i != col else BLUE   # the highlighted column's total
        ax.add_patch(Rectangle((i, strip_y), 1, 1, facecolor=fc, alpha=0.30,
                               edgecolor=ORANGE, lw=1.1, zorder=2))
    # a thin connector showing the blue column collapses onto its orange total
    ax.annotate("", xy=(col + 0.5, strip_y + 1.0), xytext=(col + 0.5, -0.05),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.1,
                                shrinkA=1, shrinkB=1, mutation_scale=11),
                zorder=3)

    # (2) add the column totals together: horizontal sweep along the orange strip
    fl.arrow(ax, (0.18, strip_y + 0.5), (nx - 0.18, strip_y + 0.5),
             color=ORANGE, lw=2.2, mut=15)
    ax.text(nx / 2, strip_y - 0.30, "(2) add the column totals", ha="center",
            va="top", color=ORANGE, fontsize=12)

    # epsilon side labels on the top-left corner cell (clear of every sweep)
    ax.annotate("", xy=(1, ny + 0.12), xytext=(0, ny + 0.12),
                arrowprops=dict(arrowstyle="<->", color=GRAY, lw=0.9))
    ax.text(0.5, ny + 0.20, r"$\epsilon$", ha="center", va="bottom", color=GRAY,
            fontsize=11)
    ax.annotate("", xy=(-0.18, ny), xytext=(-0.18, ny - 1),
                arrowprops=dict(arrowstyle="<->", color=GRAY, lw=0.9))
    ax.text(-0.30, ny - 0.5, r"$\epsilon$", ha="right", va="center", color=GRAY,
            fontsize=11)

    ax.set_xlim(-0.9, nx + 0.5)
    ax.set_ylim(strip_y - 0.95, ny + 0.95)
    ax.set_aspect("equal")
    ax.axis("off")
    fl.save(fig, "mdl-cal-sum-order")


def fig_cov_jacobian():
    """Change of variables, 1-D beside 2-D.  (a) The scalar substitution scales a
    segment of length epsilon by u'(x).  (b) A linear phi sends the unit square
    (area 1) to a parallelogram whose area is the local volume factor
    |det Dphi|.  The det is computed from the real matrix."""
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.6, 4.2),
                                   gridspec_kw={"width_ratios": [1.0, 1.25]})

    # --- (a) the 1-D analog: a segment of length eps stretched to eps*u' ---
    eps, du = 0.85, 1.85
    yb, yo = 1.55, 0.45                  # heights of the two segments
    # a faint left rule the two segments both start from (the common origin)
    axa.plot([0, 0], [yo - 0.22, yb + 0.22], color=LIGHT, lw=1.2, zorder=1)
    axa.plot([0, eps], [yb, yb], color=BLUE, lw=7, solid_capstyle="butt",
             zorder=3)
    axa.plot([0, eps * du], [yo, yo], color=ORANGE, lw=7, solid_capstyle="butt",
             zorder=3)
    # end ticks so the lengths read cleanly
    for x, y, c in [(eps, yb, BLUE), (eps * du, yo, ORANGE)]:
        axa.plot([x, x], [y - 0.10, y + 0.10], color=c, lw=1.5, zorder=4)
    fl.arrow(axa, (eps / 2, yb - 0.16), (eps * du / 2, yo + 0.16), color=GRAY,
             lw=1.4, mut=13)
    axa.text(eps / 2, yb + 0.20, r"$\epsilon$", ha="center", va="bottom",
             color=BLUE, fontsize=12)
    axa.text(eps * du / 2, yo - 0.20, r"$\epsilon\,u'(x)$", ha="center",
             va="top", color=ORANGE, fontsize=12)
    axa.set_xlim(-0.15, 1.65); axa.set_ylim(0.10, 1.90)
    axa.set_aspect("equal")
    axa.axis("off")

    # --- (b) the 2-D version: unit square -> parallelogram under phi ---
    A = np.array([[1.5, 0.6], [0.4, 1.3]])   # linear phi = matrix multiply
    det = np.linalg.det(A)                    # 1.71
    unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]).T
    img = A @ unit
    # original unit square (faint) with its area shaded
    axb.add_patch(Polygon(unit.T, closed=True, facecolor=BLUE, alpha=0.20,
                          edgecolor=BLUE, lw=1.6, zorder=2))
    # image parallelogram
    axb.add_patch(Polygon(img.T, closed=True, facecolor=ORANGE, alpha=0.24,
                          edgecolor=ORANGE, lw=2, zorder=2))
    # the basis edges of the square and their images
    fl.arrow(axb, (0, 0), (1, 0), color=BLUE, lw=1.8)
    fl.arrow(axb, (0, 0), (0, 1), color=BLUE, lw=1.8)
    fl.arrow(axb, (0, 0), A @ np.array([1, 0]), color=ORANGE, lw=1.8)
    fl.arrow(axb, (0, 0), A @ np.array([0, 1]), color=ORANGE, lw=1.8)
    # the mapping arrow phi, arcing high over BOTH shapes; the label sits above
    # the arc's apex, well clear of the curve.
    axb.annotate("", xy=(1.9, 1.92), xytext=(0.5, 1.28),
                 arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.4,
                                 mutation_scale=14,
                                 connectionstyle="arc3,rad=-0.32"), zorder=4)
    axb.text(1.18, 2.16, r"$\phi$", ha="center", va="center", color=GRAY,
             fontsize=14)
    axb.text(0.32, 0.30, "area $1$", ha="center", va="center", color=BLUE,
             fontsize=11)
    axb.text(1.46, 0.92, rf"area $|\det D\phi|={abs(det):.2g}$",
             ha="center", va="center", color=ORANGE, fontsize=11)
    axb.set_xlim(-0.15, 2.35); axb.set_ylim(-0.15, 2.35)
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
        f <- a, b, u ;  a <- u ;  u <- y ;  b <- v ;  v <- y
    so that df/dy = df/da da/du du/dy + df/du du/dy + df/db db/dv dv/dy."""
    fig, ax = plt.subplots(figsize=(5.6, 2.8))

    Y = {"y": (0, 1.0)}
    MID = {"u": (1.5, 1.7), "v": (1.5, 0.3)}
    INNER = {"a": (3.0, 1.7), "b": (3.0, 0.3)}
    F = {"f": (4.5, 1.0)}

    edges = [
        ("y", "u"), ("y", "v"),       # y feeds u and v
        ("u", "a"),                    # u feeds a
        ("v", "b"),                    # v feeds b
    ]
    pos = {**Y, **MID, **INNER, **F}
    for s, t in edges:
        edge(ax, pos[s], pos[t])
    # f depends directly on a, b, and (the shortcut) u  -> three paths from y
    edge(ax, pos["a"], pos["f"])
    edge(ax, pos["b"], pos["f"])
    edge(ax, pos["u"], pos["f"])       # the extra direct u -> f path

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
    for px, py in [(-1.9, 1.1), (1.6, -1.4), (0.7, 1.9),
                   (-1.1, -1.0), (2.1, 0.5), (-0.5, -2.0)]:
        g = grad(px, py)
        fl.arrow(ax, (px, py), (px + 0.32 * g[0], py + 0.32 * g[1]),
                 color=ORANGE, lw=2.0, mut=12)
    ax.plot([0], [0], "o", color=BLUE, ms=7, zorder=4)     # the minimum
    ax.text(0.12, -0.32, "min", color=BLUE, fontsize=11)
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
    px, py = np.linspace(-1.8, -0.2, 30), np.linspace(-0.8, 0.8, 30)
    PX, PY = np.meshgrid(px, py)
    ax.plot_surface(PX, PY, q(PX, PY), color=ORANGE, alpha=0.55, linewidth=0,
                    antialiased=True, zorder=3)
    # base point on top of both surfaces, with a leader to a label set clear in
    # the empty space below-front (so it is never swallowed by the orange bowl)
    ax.scatter([x0], [y0], [f(x0, y0)], color="black", s=42, zorder=6,
               depthshade=False)
    ax.plot([x0, x0], [y0, y0], [f(x0, y0), f(x0, y0) - 0.42], color="black",
            lw=0.9, zorder=6)
    ax.text(x0, y0, f(x0, y0) - 0.52, r"$(-1,0)$", fontsize=14, ha="center",
            va="top", zorder=6)
    ax.set_xlabel("$x$", fontsize=15, labelpad=-6)
    ax.set_ylabel("$y$", fontsize=15, labelpad=-6)
    ax.set_zlabel("$z$", fontsize=15, labelpad=-6)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=22, azim=-58)
    fl.save(fig, "mdl-cal-taylor-quadratic")


def fig_tangent_plane():
    """Tangent plane and gradient geometry. The surface $z=f(x,y)$ (blue) with its
    tangent plane at a point (orange); on the base plane, the level curves of $f$
    (gray) and the gradient (orange arrow) crossing them at a right angle -- the
    graph-space companion to ``gradient is normal to the level set''."""
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
    # drawn as full parametric curves so they never break into dashes
    th = np.linspace(0, 2 * np.pi, 200)
    for c in [0.25, 0.7, 1.3, 2.1]:
        ax.plot(np.sqrt(c / 0.35) * np.cos(th), np.sqrt(c / 0.6) * np.sin(th),
                zbase, color=GRAY, lw=0.9, zorder=1)

    gx, gy = np.linspace(-1.8, 1.8, 60), np.linspace(-1.8, 1.8, 60)
    X, Y = np.meshgrid(gx, gy)
    ax.plot_surface(X, Y, f(X, Y), color=BLUE, alpha=0.35, linewidth=0, zorder=2)
    px, py = np.linspace(x0 - 0.7, x0 + 0.7, 20), np.linspace(y0 - 0.7, y0 + 0.7, 20)
    PX, PY = np.meshgrid(px, py)
    ax.plot_surface(PX, PY, plane(PX, PY), color=ORANGE, alpha=0.5, linewidth=0,
                    zorder=3)
    ax.scatter([x0], [y0], [z0], color="black", s=34, zorder=6,
               depthshade=False)

    # the gradient on the base plane, normal to the level set, pointing uphill;
    # rooted under the base point and long enough to read as a vector
    g = np.array([gx0, gy0]); g = g / np.linalg.norm(g) * 1.25
    ax.quiver(x0, y0, zbase + 0.01, g[0], g[1], 0.0, color=ORANGE, lw=2.6,
              arrow_length_ratio=0.16, zorder=5)
    # label placed beyond the arrowhead, nudged off the shaft so they don't clash
    ax.text(x0 + g[0] - 0.30, y0 + g[1] - 0.34, zbase, r"$\nabla f$",
            color=ORANGE, fontsize=13, ha="left", va="top", zorder=6)
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$"); ax.set_zlabel("$z$")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=30, azim=-60)
    fl.save(fig, "mdl-cal-tangent-plane")


# =========================================================================== #
# Matrix calculus & automatic differentiation                                 #
# =========================================================================== #

def fig_jacobian_ellipse():
    """Up close, a differentiable map is a linear map.  The nonlinear
    $\\mathbf f(x,y)=(x+\\sin y,\\ y+x^2/2)$ carries a small circle and grid
    around $\\mathbf x_0=(0.5,0.5)$ (left) into the output plane (right), where
    the true image (blue) is nearly indistinguishable from the ellipse
    $\\mathbf f(\\mathbf x_0)+\\mathbf J(\\mathbf x_0)\\boldsymbol\\delta$
    predicted by the Jacobian (orange dashed); the small mismatch is the
    $o(\\|\\boldsymbol\\delta\\|)$ remainder."""
    F = lambda x, y: (x + np.sin(y), y + 0.5 * x ** 2)
    x0, y0 = 0.5, 0.5
    J = np.array([[1.0, np.cos(y0)], [x0, 1.0]])    # [[1, 0.8776], [0.5, 1]]
    fx0 = np.array(F(x0, y0))                       # ~ (0.979, 0.625)
    r, hw = 0.35, 0.45
    th = np.linspace(0.0, 2 * np.pi, 200)
    circ = np.stack([x0 + r * np.cos(th), y0 + r * np.sin(th)])
    lines = np.linspace(-hw, hw, 7)
    ts = np.linspace(-hw, hw, 60)

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.0, 3.8))

    # --- (a) input plane: light grid + circle of perturbations around x0 ---
    for c in lines:
        axa.plot(x0 + ts, np.full_like(ts, y0 + c), color=LIGHT, lw=0.8,
                 zorder=1)
        axa.plot(np.full_like(ts, x0 + c), y0 + ts, color=LIGHT, lw=0.8,
                 zorder=1)
    axa.plot(circ[0], circ[1], color=BLUE, lw=2.2, zorder=3)
    axa.plot([x0], [y0], "o", color="black", ms=5, zorder=4)
    axa.text(x0 + 0.05, y0 + 0.05, r"$\mathbf{x}_0$", fontsize=11, zorder=4)
    axa.set_xlim(x0 - hw - 0.1, x0 + hw + 0.1)
    axa.set_ylim(y0 - hw - 0.1, y0 + hw + 0.1)
    axa.set_aspect("equal")
    axa.axis("off")

    # --- (b) output plane: warped grid, true image vs. Jacobian ellipse ---
    for c in lines:
        gx, gy = F(x0 + ts, y0 + c)
        axb.plot(gx, gy, color=LIGHT, lw=0.8, zorder=1)
        gx, gy = F(np.full_like(ts, x0 + c), y0 + ts)
        axb.plot(gx, gy, color=LIGHT, lw=0.8, zorder=1)
    img = np.stack(F(circ[0], circ[1]))             # the true image f(circle)
    ell = fx0[:, None] + J @ (circ - [[x0], [y0]])  # the Jacobian's prediction
    axb.plot(img[0], img[1], color=BLUE, lw=2.2, zorder=3)
    axb.plot(ell[0], ell[1], "--", color=ORANGE, lw=2.0, zorder=4)
    axb.plot([fx0[0]], [fx0[1]], "o", color="black", ms=5, zorder=5)
    axb.text(1.52, 1.02, r"$\mathbf{f}$", color=BLUE, fontsize=12, zorder=5)
    axb.text(0.98, 0.02, r"$\mathbf{f}(\mathbf{x}_0)+\mathbf{J}\,\boldsymbol{\delta}$",
             color=ORANGE, fontsize=11, ha="center", zorder=5)
    axb.set_aspect("equal")
    axb.axis("off")

    fl.save(fig, "mdl-cal-jacobian-ellipse")


def fig_fwd_vs_rev():
    """Forward- vs reverse-mode AD on a chain x -> a -> b -> L.  Both evaluate
    left-to-right (gray); forward propagates a tangent (JVP) the same direction,
    reverse records a tape then propagates an adjoint (VJP) right-to-left to get
    the whole scalar-loss gradient in one pass."""
    chain, xpos = ["x", "a", "b", "L"], [0.0, 1.5, 3.0, 4.5]
    fig, (axt, axb) = plt.subplots(2, 1, figsize=(6.6, 3.5))

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
        ax.text(2.25, 0.82, dual_label, ha="center", color=dual_color, fontsize=11)
        ax.set_xlim(-0.5, 5.0); ax.set_ylim(-0.55, 1.02)
        ax.set_aspect("equal"); ax.axis("off")

    draw(axt, "fwd", BLUE,
         r"forward: tangent $\dot a = J\dot x$ (JVP), one input at a time")
    draw(axb, "rev", ORANGE,
         r"reverse: adjoint $\bar a = J^{\top}\bar L$ (VJP), one output at a time")
    fig.subplots_adjust(hspace=0.05)
    fl.save(fig, "mdl-cal-fwd-vs-rev")


def fig_tape_dag():
    """The computational graph (tape) of $y=r\\cdot r$ with $r=uv+u$.  It is a
    diamond, not a chain: $u$ fans out to the product $t=uv$ and the sum
    $r=t+u$, and $r$ feeds *both* arguments of $y=r\\cdot r$ (the doubled
    edge).  Because a value can feed several consumers, the backward pass
    accumulates each adjoint over outgoing edges with ``+=``; here $\\bar r$
    receives the contribution $\\bar y\\,r$ twice."""
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    hh = 0.3                                    # box half-height
    # t sits at the vertical midpoint between u and v; r and y sit level with u,
    # so the two "downstream" boxes read as a clean top row over the t/v pair.
    pos = {"u": (0.0, 2.0), "v": (0.0, 0.4), "t": (2.3, 1.2),
           "r": (4.4, 2.0), "y": (6.5, 2.0)}
    hw = {"u": 0.42, "v": 0.42, "t": 0.72, "r": 0.8, "y": 0.8}

    def box(k, label, color=LIGHT):
        cx, cy = pos[k]; w = hw[k]
        ax.add_patch(FancyBboxPatch((cx - w, cy - hh), 2 * w, 2 * hh,
                     boxstyle="round,pad=0.02,rounding_size=0.12",
                     facecolor=color, edgecolor="black", lw=1.2, zorder=3))
        ax.text(cx, cy, label, ha="center", va="center", fontsize=10.5, zorder=4)

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

    # Each edge emerges from the source's NE/SE/E corner and enters the target's
    # NW/SW corner, so no arrow cuts through a box.
    link(port("u", "SE"), port("t", "NW"))      # u -> t
    link(port("v", "NE"), port("t", "SW"))      # v -> t
    link(port("t", "NE"), port("r", "SW"))      # t -> r
    link(port("u", "E"),  port("r", "W"))       # u -> r  (level with u, over the t box)
    # r feeds BOTH arguments of y = r*r: two clean parallel edges (NE->NW, SE->SW)
    link(port("r", "NE"), port("y", "NW"))
    link(port("r", "SE"), port("y", "SW"))

    box("u", "$u$"); box("v", "$v$")
    box("t", r"$t=uv$", color=BLUE)
    box("r", r"$r=t+u$", color=BLUE)
    box("y", r"$y=r\cdot r$", color=ORANGE)
    ax.text(0.0, 2.66, r"$u$ fans out (diamond)", color=GRAY, fontsize=11,
            ha="center")
    ax.text(5.45, 1.12, r"$\bar r=\bar y\,r+\bar y\,r$  (+= twice)",
            color=ORANGE, fontsize=11, ha="center")
    ax.set_xlim(-1.0, 7.6); ax.set_ylim(-0.1, 3.05)
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
    ax.axhline(0, color="black", lw=0.8, zorder=1)
    ax.plot(xs, f(xs), color=BLUE, lw=2.6, zorder=4, label=r"$f(x)=\sin x$")
    ax.plot(xs, tangent(xs), "--", color=GRAY, lw=1.8, zorder=2,
            label="tangent (matches value, slope)")
    ax.plot(xs, parab(xs), "--", color=ORANGE, lw=2.0, zorder=3,
            label="best parabola (matches curvature too)")
    ax.plot([x0], [s0], "o", color="black", ms=5, zorder=6)
    ax.text(x0 + 0.06, s0 - 0.13, r"$x_0$", fontsize=12, ha="left", va="top")
    ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(-1.65, 1.9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="lower right", fontsize=11, frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fl.save(fig, "mdl-cal-best-parabola")


def fig_lagrange_tangency():
    """Lagrange multipliers as tangency.  Contours of $f(x,y)=x^2+y^2$ (gray
    circles) against the constraint $g(x,y)=xy=1$ (blue hyperbola).  At the
    constrained optimum $(1,1)$ the level set of $f$ *kisses* the constraint and
    the two gradients align ($\\nabla f=(2,2)$, $\\nabla g=(1,1)$: parallel).  At
    a non-optimal feasible point the gradients disagree, so $\\nabla f$ has a
    component along the constraint and sliding along it still improves $f$.
    All arrows are the true gradients, normalized to display length."""
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    th = np.linspace(0, 2 * np.pi, 300)
    # contours of f: circles; the sqrt(2)-circle is the one through the optimum
    for rad, lw in [(0.8, 1.0), (np.sqrt(2.0), 1.6), (1.9, 1.0), (2.5, 1.0)]:
        col = ORANGE if abs(rad - np.sqrt(2.0)) < 1e-9 else GRAY
        ax.plot(rad * np.cos(th), rad * np.sin(th), color=col,
                lw=lw, zorder=2, alpha=0.9 if col is ORANGE else 0.8)
    # the constraint xy = 1 (first-quadrant branch)
    cx = np.linspace(0.34, 3.1, 300)
    ax.plot(cx, 1.0 / cx, color=BLUE, lw=2.4, zorder=3)
    ax.text(0.62, 2.55, r"$g(x,y)=c$", color=BLUE, fontsize=11, ha="left")

    def grads(px, py, scale=0.62):
        gf = np.array([2 * px, 2 * py])          # true gradient of f
        gg = np.array([py, px])                  # true gradient of g = xy
        for g, col, lab, off in [(gf, ORANGE, r"$\nabla f$", (0.10, 0.16)),
                                 (gg, GREEN, r"$\nabla g$", (0.16, -0.14))]:
            u = g / np.linalg.norm(g) * scale
            fl.arrow(ax, (px, py), (px + u[0], py + u[1]), color=col,
                     lw=2.0, mut=12)
            ax.text(px + u[0] + off[0], py + u[1] + off[1], lab, color=col,
                    fontsize=10.5, ha="center", va="center")
        ax.plot([px], [py], "o", color="black", ms=5, zorder=6)

    # optimum (1,1): gradients parallel -- one arrow drawn slightly shorter so
    # both are visible along the same ray
    px, py = 1.0, 1.0
    gfu = np.array([1.0, 1.0]) / np.sqrt(2.0)
    fl.arrow(ax, (px, py), (px + 0.95 * gfu[0], py + 0.95 * gfu[1]),
             color=ORANGE, lw=2.2, mut=13)
    fl.arrow(ax, (px, py), (px + 0.55 * gfu[0], py + 0.55 * gfu[1]),
             color=GREEN, lw=2.2, mut=11)
    ax.plot([px], [py], "o", color="black", ms=5, zorder=6)
    ax.text(1.86, 1.62, r"$\nabla f\parallel\nabla g$", fontsize=11,
            ha="left", color=ORANGE)
    ax.text(0.62, 0.50, r"optimum", fontsize=11, ha="right", va="top")

    # a non-optimal feasible point: gradients visibly misaligned
    grads(2.2, 1.0 / 2.2)
    ax.text(2.2, 0.10, "not optimal:\ngradients disagree", fontsize=11,
            ha="center", va="top", color=GRAY)

    ax.set_aspect("equal")
    ax.set_xlim(-0.4, 3.4); ax.set_ylim(-0.4, 3.0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fl.save(fig, "mdl-cal-lagrange-tangency")


def fig_jacobian_shapes():
    """The shape of the Jacobian dictates the cheap mode.  Three glyphs: a wide
    $1\\times n$ row (a scalar loss; one VJP pass), a tall $m\\times 1$ column
    (one input; one JVP pass), and a full $m\\times n$ matrix, which costs
    $\\min(m,n)$ passes either way.  The pass counts under each glyph are the
    entire forward-vs-reverse cost model."""
    fig, ax = plt.subplots(figsize=(9.6, 3.4))
    cell = 0.34

    def grid(x0, y0, rows, cols, color):
        for i in range(rows):
            for j in range(cols):
                ax.add_patch(Rectangle((x0 + j * cell, y0 - (i + 1) * cell),
                                       cell, cell, facecolor=color,
                                       edgecolor="black", lw=0.7, zorder=3,
                                       alpha=0.55))

    n, m = 8, 5
    top = 2.15
    # wide row: 1 x n  (reverse mode, one VJP)
    grid(0.0, top - 0.7, 1, n, ORANGE)
    ax.text(0.5 * n * cell, top + 0.06, r"$1\times n$ (scalar loss)",
            ha="center", fontsize=11)
    ax.text(0.5 * n * cell, top - 1.35,
            "one VJP\n= 1 backward pass", ha="center", va="top", fontsize=11,
            color=ORANGE)
    # tall column: m x 1  (forward mode, one JVP)
    x1 = 4.2
    grid(x1, top, m, 1, GREEN)
    ax.text(x1 + 0.5 * cell, top + 0.06, r"$m\times 1$", ha="center",
            fontsize=11)
    ax.text(x1 + 0.5 * cell, top - m * cell - 0.25,
            "one JVP\n= 1 forward pass", ha="center", va="top", fontsize=11,
            color=GREEN)
    # full matrix: m x n
    x2 = 6.4
    grid(x2, top, m, n, BLUE)
    ax.text(x2 + 0.5 * n * cell, top + 0.06, r"$m\times n$ (full Jacobian)",
            ha="center", fontsize=11)
    ax.text(x2 + 0.5 * n * cell, top - m * cell - 0.25,
            r"$\min(m,n)$ passes" + "\n(rows by VJP or columns by JVP)",
            ha="center", va="top", fontsize=11, color=BLUE)

    ax.set_xlim(-0.4, 9.6); ax.set_ylim(-0.65, 2.45)
    ax.set_aspect("equal"); ax.axis("off")
    fl.save(fig, "mdl-cal-jacobian-shapes")


def fig_checkpointing():
    """Gradient checkpointing as a timeline.  Top track: plain reverse mode
    stores every one of the $L$ forward activations until the backward sweep
    consumes them ($O(L)$ memory).  Bottom track: checkpointing keeps only every
    $\\sqrt L$-th activation and, when the backward sweep reaches a segment,
    recomputes that segment forward from its checkpoint ($O(\\sqrt L)$ memory,
    about one extra forward pass in time)."""
    L, K = 16, 4                                  # depth and checkpoint spacing
    cell, gap = 0.5, 0.06
    fig, ax = plt.subplots(figsize=(9.8, 3.6))

    def track(y, kept, label):
        for i in range(L):
            filled = kept(i)
            ax.add_patch(Rectangle((i * (cell + gap), y), cell, cell,
                                   facecolor=BLUE if filled else "white",
                                   alpha=0.55 if filled else 1.0,
                                   edgecolor="black", lw=0.8, zorder=3))
        ax.text(-0.25, y + cell / 2, label, ha="right", va="center",
                fontsize=11)

    width = L * (cell + gap) - gap
    y_top, y_bot = 2.15, 0.75
    track(y_top, lambda i: True,
          "store all $L$\nactivations")
    track(y_bot, lambda i: i % K == 0,
          "store $\\sqrt{L}$\ncheckpoints")

    # forward / backward sweep arrows above the top track
    fl.arrow(ax, (0.0, y_top + cell + 0.32), (width, y_top + cell + 0.32),
             color=GRAY, lw=1.6, mut=12)
    ax.text(width / 2, y_top + cell + 0.42, "forward pass (layers $1$ to $L$)",
            ha="center", va="bottom", fontsize=11, color=GRAY)

    # the recomputed segment on the bottom track: backward has reached layer 16,
    # so the segment after the last checkpoint is recomputed forward
    seg0 = 12 * (cell + gap)
    fl.arrow(ax, (seg0 + cell / 2, y_bot - 0.28),
             (width - cell / 2 + 0.15, y_bot - 0.28), color=ORANGE, lw=2.0,
             mut=12)
    ax.text((seg0 + width) / 2, y_bot - 0.40,
            "recompute segment forward\nfrom its checkpoint, then sweep it",
            ha="center", va="top", fontsize=11, color=ORANGE)
    fl.arrow(ax, (width, y_bot + cell + 0.30), (seg0, y_bot + cell + 0.30),
             color=GRAY, lw=1.6, ls="-", mut=12)
    ax.text((seg0 + width) / 2, y_bot + cell + 0.40, "backward sweep",
            ha="center", va="bottom", fontsize=11, color=GRAY)

    # memory annotations on the right
    ax.text(width + 0.35, y_top + cell / 2, r"$O(L)$ memory", fontsize=11,
            va="center", color=BLUE)
    ax.text(width + 0.35, y_bot + cell / 2,
            r"$O(\sqrt{L})$ memory," + "\n~1 extra forward pass", fontsize=11,
            va="center", color=BLUE)

    ax.set_xlim(-2.1, width + 3.1); ax.set_ylim(-0.75, 3.45)
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
    ax.axhline(0, color="black", lw=0.9, zorder=1)
    ax.plot(xs, f(xs), color=BLUE, lw=2.4, zorder=4)
    pos, neg = f(span) >= 0, f(span) < 0
    ax.fill_between(span, 0, f(span), where=pos, color=BLUE, alpha=0.25, lw=0)
    ax.fill_between(span, 0, f(span), where=neg, color=ORANGE, alpha=0.30, lw=0)
    ax.text(1.15, 0.34, r"$+$", fontsize=17, ha="center", color=BLUE)
    ax.text(3.45, -0.30, r"$-$", fontsize=17, ha="center", color=ORANGE)
    ax.text(5.65, 0.22, r"$+$", fontsize=17, ha="center", color=BLUE)
    for x, lab in [(a, "$a$"), (b, "$b$")]:
        ax.plot([x, x], [0, f(x)], ":", color=GRAY, lw=1.0, zorder=2)
        ax.plot([x], [0], "o", color="black", ms=4, zorder=5)
        ax.text(x, -0.09, lab, ha="center", va="top", fontsize=12)
    # direction arrows under the axis (below the curve's lowest dip):
    # integrating a->b, and the reversed b->a, which negates the total
    fl.arrow(ax, (a, -1.02), (b, -1.02), color=GRAY, lw=1.6, mut=12)
    ax.text(-0.15, -1.02, r"$\int_a^b f$", ha="right", va="center",
            fontsize=11, color=GRAY)
    fl.arrow(ax, (b, -1.52), (a, -1.52), color=ORANGE, lw=1.6, mut=12)
    ax.text(6.55, -1.52, r"$\int_b^a f = -\int_a^b f$", ha="left",
            va="center", fontsize=11, color=ORANGE)
    ax.set_xlim(-1.1, 9.4); ax.set_ylim(-1.95, 1.12)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fl.save(fig, "mdl-cal-signed-area")


# =========================================================================== #
# Driver                                                                      #
# =========================================================================== #

FIGURES = [
    # single-variable calculus
    fig_zoom_sequence,
    fig_pos_second,
    fig_neg_second,
    fig_zero_second,
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
