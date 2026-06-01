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


# =========================================================================== #
# Integral calculus                                                           #
# =========================================================================== #

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


# =========================================================================== #
# Driver                                                                      #
# =========================================================================== #

FIGURES = [
    # single-variable calculus
    fig_zoom_sequence,
    fig_pos_second,
    fig_neg_second,
    fig_zero_second,
    # integral calculus
    fig_sub_area,
    fig_rect_trans,
    fig_sum_order,
    fig_cov_jacobian,
    # multivariable calculus (chain-rule graphs)
    fig_chain_net1,
    fig_chain_net2,
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
