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

from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle


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
        ax.axhline(0, color=GRAY, lw=0.8)
        ax.axvline(0, color=GRAY, lw=0.8)
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
    ax.text(2.05, y0 + df(x0) * 1.05 + 0.16, r"slope $f'(x)$", color=ORANGE,
            ha="left", va="bottom", fontsize=10)
    ax.text(x0, y0 - 0.16, r"$(x,\,f(x))$", ha="center", va="top", fontsize=9)
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
    ax.plot([x0, x0], [0, f(x0)], ":", color=GRAY, lw=1.0)
    ax.plot([x1, x1], [0, f(x1)], ":", color=GRAY, lw=1.0)
    fl.arrow(ax, (x0, -0.18), (x1, -0.18), color=GREEN, lw=2.0, mut=13)
    ax.text((x0 + x1) / 2, -0.42, r"$-\eta f'(x)$", ha="center", va="top",
            color=GREEN, fontsize=10)
    ax.plot([x0], [f(x0)], "o", color="black", ms=5, zorder=5)
    ax.plot([x1], [f(x1)], "o", color=GREEN, ms=5, zorder=5)
    ax.text(x0 + 0.08, f(x0) + 0.04, r"$x$", ha="left", va="bottom")
    ax.text(x1 - 0.08, f(x1) + 0.06, r"$x-\eta f'(x)$", ha="right", va="bottom",
            color=GREEN, fontsize=9)
    ax.annotate(r"drop $\approx \eta\,[f'(x)]^2$", xy=(x1, f(x1)),
                xytext=(-2.15, 1.8), fontsize=9.5, color=GRAY,
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.0,
                                connectionstyle="arc3,rad=0.25"))
    ax.set_xlim(-2.4, 2.4); ax.set_ylim(-0.65, 2.25)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fl.save(fig, "mdl-cal-gd-step")


def fig_relu_corner():
    """Two convex kinks and their subdifferentials. Left: $|x|$, whose corner at 0
    admits a fan of supporting lines with slopes sweeping $[-1,1]$. Right: ReLU,
    $\\max(0,x)$, whose corner admits slopes sweeping $[0,1]$. Off the corner each
    has a single tangent and the subdifferential collapses to the derivative."""
    xs = np.linspace(-2.0, 2.0, 400)
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.0, 3.9))

    # left: |x|, subgradient fan sweeping [-1, 1]
    axa.axhline(0, color=GRAY, lw=0.8, zorder=1)
    for s in np.linspace(-1.0, 1.0, 9):
        axa.plot(xs, s * xs, "-", color=ORANGE, lw=1.0, alpha=0.5, zorder=2)
    axa.plot(xs, np.abs(xs), color=BLUE, lw=2.6, zorder=4)
    axa.plot([0], [0], "o", color="black", ms=6, zorder=5)
    axa.text(1.45, 1.62, r"$|x|$", color=BLUE, fontsize=12, ha="center")
    axa.set_title(r"$\partial|x|(0)=[-1,1]$", fontsize=11)
    axa.set_xlim(-2.0, 2.0); axa.set_ylim(-0.55, 2.0)
    axa.set_xticks([0]); axa.set_yticks([])
    axa.spines["top"].set_visible(False); axa.spines["right"].set_visible(False)

    # right: ReLU = max(0, x), subgradient fan sweeping [0, 1]
    axb.axhline(0, color=GRAY, lw=0.8, zorder=1)
    for s in np.linspace(0.0, 1.0, 6):
        axb.plot(xs, s * xs, "-", color=ORANGE, lw=1.0, alpha=0.5, zorder=2)
    axb.plot(xs, np.maximum(0.0, xs), color=BLUE, lw=2.6, zorder=4)
    axb.plot([0], [0], "o", color="black", ms=6, zorder=5)
    axb.text(1.35, 1.62, r"$\mathrm{ReLU}(x)$", color=BLUE, fontsize=12, ha="center")
    axb.set_title(r"$\partial\,\mathrm{ReLU}(0)=[0,1]$", fontsize=11)
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
    ax.plot(xs, f(xi) + slope * (xs - xi), "--", color=ORANGE, lw=2.0, zorder=4)
    for x, lab in [(a, "$a$"), (b, "$b$")]:
        ax.plot([x], [f(x)], "o", color="black", ms=5, zorder=5)
        ax.text(x, f(x) - 0.28, lab, ha="center", va="top", fontsize=10)
    ax.plot([xi], [f(xi)], "o", color=ORANGE, ms=6, zorder=6)
    ax.text(xi, f(xi) + 0.16, r"$\xi$", ha="center", va="bottom", color=ORANGE,
            fontsize=12)
    ax.text((a + b) / 2 + 0.3, f((a + b) / 2) - 0.7, "secant", color=GRAY,
            fontsize=9, ha="center")
    ax.set_xlim(a - 0.3, b + 0.6)
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
    ax.axhline(0, color=GRAY, lw=0.8, zorder=1)
    ax.plot(xs, f, color=BLUE, lw=2.6, zorder=3, label=r"$f(x)=e^{-1/x^2}$")
    ax.plot(xs, np.zeros_like(xs), "--", color=ORANGE, lw=2.0, zorder=2,
            label=r"Taylor series at $0\ \equiv\ 0$")
    ax.plot([0], [0], "o", color="black", ms=5, zorder=4)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-0.06, 0.5)
    ax.set_xticks([0]); ax.set_yticks([])
    ax.legend(loc="upper center", fontsize=9, frameon=False)
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
        ax.set_ylim(-0.02, 1.18)
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
    """A single thin rectangle under change of variables u(x): the sliver of
    width epsilon at x maps to a sliver of width epsilon*u'(x); to make the two
    areas agree we rescale the height by the derivative du/dx."""
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(8.6, 3.4))

    # left: f(u(x)) on the x-axis, a thin rectangle of width eps at x
    fu = lambda t: 0.55 + 0.6 * np.exp(-0.5 * ((t - 1.5) / 0.9) ** 2)
    xs = np.linspace(0.0, 3.0, 400)
    x0, eps = 1.2, 0.42
    h = fu(x0)
    axa.plot(xs, fu(xs), color=BLUE, lw=2.0, zorder=3)
    axa.add_patch(Rectangle((x0, 0), eps, h, facecolor=BLUE, alpha=0.25,
                            edgecolor=BLUE, lw=1.4, zorder=2))
    axa.annotate("", xy=(x0 + eps, -0.12), xytext=(x0, -0.12),
                 arrowprops=dict(arrowstyle="<->", color=GRAY, lw=1.0))
    axa.text(x0 + eps / 2, -0.22, r"$\epsilon$", ha="center", va="top",
             color=GRAY)
    axa.text(x0 + eps / 2, h + 0.07, r"$f(u(x))$", ha="center", va="bottom",
             color=BLUE, fontsize=10)
    axa.set_title(r"(a) width $\epsilon$, area $\approx \epsilon\,f(u(x))$",
                  fontsize=10.5)
    axa.set_xlim(0, 3); axa.set_ylim(-0.35, 1.4)
    axa.set_xticks([x0]); axa.set_xticklabels(["$x$"])
    axa.set_yticks([])
    axa.spines["top"].set_visible(False); axa.spines["right"].set_visible(False)

    # right: f(y) on the y=u-axis, the stretched rectangle of width eps*u'
    fy = lambda t: 0.55 + 0.6 * np.exp(-0.5 * ((t - 1.9) / 1.1) ** 2)
    ys = np.linspace(0.0, 3.6, 400)
    u0, du = 1.4, 1.55          # u(x) and the local stretch factor du/dx > 1
    weps = eps * du
    hh = fy(u0)
    axb.plot(ys, fy(ys), color=BLUE, lw=2.0, zorder=3)
    axb.add_patch(Rectangle((u0, 0), weps, hh, facecolor=ORANGE, alpha=0.25,
                            edgecolor=ORANGE, lw=1.4, zorder=2))
    axb.annotate("", xy=(u0 + weps, -0.12), xytext=(u0, -0.12),
                 arrowprops=dict(arrowstyle="<->", color=GRAY, lw=1.0))
    axb.text(u0 + weps / 2, -0.22, r"$\epsilon\,\frac{du}{dx}$", ha="center",
             va="top", color=GRAY, fontsize=10)
    axb.text(u0 + weps / 2, hh + 0.07, r"$f(y)$", ha="center", va="bottom",
             color=ORANGE, fontsize=10)
    axb.set_title(r"(b) width $\epsilon\,\frac{du}{dx}$, same area", fontsize=10.5)
    axb.set_xlim(0, 3.6); axb.set_ylim(-0.35, 1.4)
    axb.set_xticks([u0]); axb.set_xticklabels(["$u(x)$"])
    axb.set_yticks([])
    axb.spines["top"].set_visible(False); axb.spines["right"].set_visible(False)

    fl.save(fig, "mdl-cal-rect-trans")


def fig_sum_order():
    """A grid of epsilon-by-epsilon squares summed columns-first (1) then adding
    the column totals together (2): the discretization behind Fubini."""
    nx, ny = 6, 4
    fig, ax = plt.subplots(figsize=(5.4, 3.8))

    # the grid of unit squares
    for i in range(nx):
        for j in range(ny):
            ax.add_patch(Rectangle((i, j), 1, 1, facecolor=LIGHT, alpha=0.30,
                                   edgecolor=GRAY, lw=0.8))

    # (1) sum within one column (vertical sweep), highlighted
    col = 2
    for j in range(ny):
        ax.add_patch(Rectangle((col, j), 1, 1, facecolor=BLUE, alpha=0.30,
                               edgecolor=BLUE, lw=1.2))
    fl.arrow(ax, (col + 0.5, -0.3), (col + 0.5, ny + 0.05), color=BLUE, lw=2.0)
    ax.text(col + 0.5, ny + 0.35, "(1)", ha="center", va="bottom", color=BLUE,
            fontsize=12)

    # (2) sum the column totals together (horizontal sweep), highlighted band
    ax.add_patch(Rectangle((0, ny + 0.6), nx, 0.7, facecolor=ORANGE, alpha=0.22,
                           edgecolor=ORANGE, lw=1.2))
    fl.arrow(ax, (-0.3, ny + 0.95), (nx + 0.3, ny + 0.95), color=ORANGE, lw=2.0)
    ax.text(nx + 0.55, ny + 0.95, "(2)", ha="left", va="center", color=ORANGE,
            fontsize=12)

    # epsilon side labels on one corner square
    ax.annotate("", xy=(1, -0.35), xytext=(0, -0.35),
                arrowprops=dict(arrowstyle="<->", color=GRAY, lw=0.9))
    ax.text(0.5, -0.5, r"$\epsilon$", ha="center", va="top", color=GRAY)
    ax.annotate("", xy=(-0.35, 1), xytext=(-0.35, 0),
                arrowprops=dict(arrowstyle="<->", color=GRAY, lw=0.9))
    ax.text(-0.5, 0.5, r"$\epsilon$", ha="right", va="center", color=GRAY)

    ax.set_xlim(-1.0, nx + 1.2)
    ax.set_ylim(-0.9, ny + 1.7)
    ax.set_aspect("equal")
    ax.axis("off")
    fl.save(fig, "mdl-cal-sum-order")


def fig_cov_jacobian():
    """Change of variables in 2-D: the unit square mapped by a linear phi to a
    parallelogram, the area ratio being |det Dphi|, beside the 1-D
    rectangle-stretch idea (a segment scaled by u'(x))."""
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.0, 3.8))

    # --- (a) the 1-D analog: a segment of length eps stretched to eps*u' ---
    eps, du = 0.8, 1.9
    axa.set_title(r"(a) 1-D: length $\times\,\frac{du}{dx}$", fontsize=10.5)
    axa.plot([0, eps], [1.4, 1.4], color=BLUE, lw=6, solid_capstyle="butt",
             zorder=3)
    axa.plot([0, eps * du], [0.4, 0.4], color=ORANGE, lw=6, solid_capstyle="butt",
             zorder=3)
    fl.arrow(axa, (eps / 2, 1.28), (eps * du / 2, 0.52), color=GRAY, lw=1.3,
             mut=11)
    axa.text(eps / 2, 1.6, r"$\epsilon$", ha="center", va="bottom", color=BLUE)
    axa.text(eps * du / 2, 0.12, r"$\epsilon\,\frac{du}{dx}$", ha="center",
             va="top", color=ORANGE, fontsize=10)
    axa.set_xlim(-0.3, 2.0); axa.set_ylim(-0.1, 2.1)
    axa.axis("off")

    # --- (b) the 2-D version: unit square -> parallelogram under phi ---
    A = np.array([[1.5, 0.6], [0.4, 1.3]])   # linear phi = matrix multiply
    det = np.linalg.det(A)
    unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]).T
    img = A @ unit
    axb.set_title(r"(b) 2-D: area $\times\,|\det D\phi|$", fontsize=10.5)
    # original unit square (faint) with its area shaded
    axb.add_patch(Polygon(unit.T, closed=True, facecolor=BLUE, alpha=0.18,
                          edgecolor=BLUE, lw=1.6))
    # image parallelogram
    axb.add_patch(Polygon(img.T, closed=True, facecolor=ORANGE, alpha=0.22,
                          edgecolor=ORANGE, lw=2))
    # the basis edges of the square and their images
    fl.arrow(axb, (0, 0), (1, 0), color=BLUE, lw=1.8)
    fl.arrow(axb, (0, 0), (0, 1), color=BLUE, lw=1.8)
    fl.arrow(axb, (0, 0), A @ np.array([1, 0]), color=ORANGE, lw=1.8)
    fl.arrow(axb, (0, 0), A @ np.array([0, 1]), color=ORANGE, lw=1.8)
    # the mapping arrow phi, arcing over the top from the square to the image
    axb.annotate("", xy=(1.45, 1.95), xytext=(0.45, 1.05),
                 arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.3,
                                 connectionstyle="arc3,rad=-0.35"))
    axb.text(0.95, 1.78, r"$\phi$", ha="center", va="center", color=GRAY,
             fontsize=13)
    axb.text(0.42, 0.34, "area $1$", ha="center", va="center", color=BLUE,
             fontsize=9)
    axb.text(1.55, 1.05, rf"area $|\det D\phi|={abs(det):.2g}$",
             ha="center", va="center", color=ORANGE, fontsize=9)
    axb.set_xlim(-0.4, 2.4); axb.set_ylim(-0.4, 2.4)
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
    ax.text(0.12, -0.32, "min", color=BLUE, fontsize=9)
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
    gx, gy = np.linspace(-2.2, 0.7, 60), np.linspace(-1.4, 1.4, 60)
    X, Y = np.meshgrid(gx, gy)
    ax.plot_surface(X, Y, f(X, Y), color=BLUE, alpha=0.45, linewidth=0,
                    antialiased=True)
    px, py = np.linspace(-1.8, -0.2, 30), np.linspace(-0.8, 0.8, 30)
    PX, PY = np.meshgrid(px, py)
    ax.plot_surface(PX, PY, q(PX, PY), color=ORANGE, alpha=0.6, linewidth=0,
                    antialiased=True)
    ax.scatter([x0], [y0], [f(x0, y0)], color="black", s=26)
    ax.text(x0, y0, f(x0, y0) - 0.14, r"$(-1,0)$", fontsize=9)
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$"); ax.set_zlabel("$z$")
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
    x0, y0 = 1.1, -0.8
    z0, gx0, gy0 = f(x0, y0), fx(x0, y0), fy(x0, y0)
    plane = lambda X, Y: z0 + gx0 * (X - x0) + gy0 * (Y - y0)

    fig = plt.figure(figsize=(6.2, 5.0))
    ax = fig.add_subplot(projection="3d")
    gx, gy = np.linspace(-1.8, 1.8, 60), np.linspace(-1.8, 1.8, 60)
    X, Y = np.meshgrid(gx, gy)
    ax.plot_surface(X, Y, f(X, Y), color=BLUE, alpha=0.35, linewidth=0)
    px, py = np.linspace(x0 - 0.7, x0 + 0.7, 20), np.linspace(y0 - 0.7, y0 + 0.7, 20)
    PX, PY = np.meshgrid(px, py)
    ax.plot_surface(PX, PY, plane(PX, PY), color=ORANGE, alpha=0.5, linewidth=0)
    ax.scatter([x0], [y0], [z0], color="black", s=26)
    zbase = -0.2
    ax.contour(X, Y, f(X, Y), levels=np.linspace(0.3, 3.0, 6), colors=GRAY,
               linewidths=0.9, offset=zbase)
    g = np.array([gx0, gy0]); g = g / np.linalg.norm(g) * 0.9
    ax.quiver(x0, y0, zbase, g[0], g[1], 0.0, color=ORANGE, lw=2.0)
    ax.text(x0 + g[0], y0 + g[1] - 0.15, zbase, r"$\nabla f$", color=ORANGE,
            fontsize=11)
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$"); ax.set_zlabel("$z$")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=24, azim=-52)
    fl.save(fig, "mdl-cal-tangent-plane")


# =========================================================================== #
# Matrix calculus & automatic differentiation                                 #
# =========================================================================== #

def fig_fwd_vs_rev():
    """Forward- vs reverse-mode AD on a chain x -> a -> b -> L.  Both evaluate
    left-to-right (gray); forward propagates a tangent (JVP) the same direction,
    reverse records a tape then propagates an adjoint (VJP) right-to-left to get
    the whole scalar-loss gradient in one pass."""
    chain, xpos = ["x", "a", "b", "L"], [0.0, 1.5, 3.0, 4.5]
    fig, (axt, axb) = plt.subplots(2, 1, figsize=(6.6, 4.6))

    def draw(ax, direction, dual_color, dual_label, title):
        centers = [(xp, 0.0) for xp in xpos]
        for c0, c1 in zip(centers, centers[1:]):       # evaluation edges
            edge(ax, c0, c1)
        for lab, c in zip(chain, centers):
            node(ax, c, f"${lab}$", r=0.26)
        for c0, c1 in zip(centers, centers[1:]):       # derivative propagation
            a, b = (c0, c1) if direction == "fwd" else (c1, c0)
            ax.annotate("", xy=(b[0], b[1] + 0.6), xytext=(a[0], a[1] + 0.6),
                        arrowprops=dict(arrowstyle="->", color=dual_color,
                                        lw=1.7, connectionstyle="arc3,rad=-0.3"))
        ax.text(2.25, 1.5, dual_label, ha="center", color=dual_color, fontsize=9)
        ax.set_title(title, fontsize=10.5, loc="left")
        ax.set_xlim(-0.5, 5.0); ax.set_ylim(-0.7, 1.9)
        ax.set_aspect("equal"); ax.axis("off")

    draw(axt, "fwd", BLUE,
         r"forward: tangent $\dot a = J\dot x$ (JVP), one input at a time",
         "Forward mode")
    draw(axb, "rev", ORANGE,
         r"reverse: adjoint $\bar a = J^{\top}\bar L$ (VJP), one output at a time",
         "Reverse mode (the tape = backprop)")
    fl.save(fig, "mdl-cal-fwd-vs-rev")


def fig_tape_dag():
    """The computational graph (Wengert list) of $(uv+u)^2$ is a *diamond*: the
    input $u$ fans out to both the product $uv$ and the later $+u$, and the two
    paths reconverge at $q$, so reverse mode must *accumulate* $u$'s adjoint from
    its two children rather than overwrite it."""
    fig, ax = plt.subplots(figsize=(6.6, 3.0))

    def box(c, label, color=LIGHT, w=0.9):
        cx, cy = c
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - 0.3), w, 0.6,
                     boxstyle="round,pad=0.02,rounding_size=0.12",
                     facecolor=color, edgecolor="black", lw=1.2, zorder=3))
        ax.text(cx, cy, label, ha="center", va="center", fontsize=10.5, zorder=4)

    pos = {"u": (0.0, 1.7), "v": (0.0, 0.3), "p": (2.1, 1.0),
           "q": (4.1, 1.0), "z": (6.1, 1.0)}
    half = {"u": 0.45, "v": 0.45, "p": 0.65, "q": 0.7, "z": 0.65}

    def link(a, b):
        c0, c1 = np.array(pos[a], float), np.array(pos[b], float)
        d = (c1 - c0) / np.linalg.norm(c1 - c0)
        fl.arrow(ax, c0 + d * (half[a] + 0.05), c1 - d * (half[b] + 0.05),
                 color=GRAY, lw=1.4, mut=12)

    for a, b in [("u", "p"), ("v", "p"), ("p", "q"), ("u", "q"), ("q", "z")]:
        link(a, b)
    box(pos["u"], "$u$"); box(pos["v"], "$v$")
    box(pos["p"], r"$p=uv$", color=BLUE, w=1.3)
    box(pos["q"], r"$q=p+u$", color=BLUE, w=1.4)
    box(pos["z"], r"$z=q^2$", color=ORANGE, w=1.3)
    ax.text(1.0, 2.05, r"$u$ fans out (diamond)", color=GRAY, fontsize=8.5,
            ha="center")
    ax.set_xlim(-0.7, 7.0); ax.set_ylim(-0.3, 2.4)
    ax.set_aspect("equal"); ax.axis("off")
    fl.save(fig, "mdl-cal-tape-dag")


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
    fig_relu_corner,
    fig_mvt,
    fig_smooth_not_analytic,
    # integral calculus
    fig_riemann,
    fig_sub_area,
    fig_rect_trans,
    fig_sum_order,
    fig_cov_jacobian,
    # multivariable calculus (chain-rule graphs + gradient geometry)
    fig_chain_net1,
    fig_chain_net2,
    fig_gradient_field,
    fig_taylor_quadratic,
    fig_tangent_plane,
    # matrix calculus & autodiff
    fig_fwd_vs_rev,
    fig_tape_dag,
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
