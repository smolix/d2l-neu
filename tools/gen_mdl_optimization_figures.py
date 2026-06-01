#!/usr/bin/env python3
"""Generate the illustrative figures for the "Mathematics for Deep Learning ->
Optimization" chapter (``chapter_mdl-optimization``) in the book's one house
style, as static SVGs.

This is a *separate* generator from ``tools/gen_mdl_figures.py`` (Linear Algebra)
so the two chapters' figure sets never collide; it imports the shared style,
palette, and helpers from that module so the look is identical.  Run with the
repo's pytorch venv (matplotlib + numpy available):

    .venv-pytorch/bin/python tools/gen_mdl_optimization_figures.py

All figures are written to ``img/mdl-opt-<id>.svg``.  Like the Linear Algebra
figures, every picture that must be exact (gradient-descent iterates on a real
quadratic, real chords/tangents of a real convex function, real critical points
of a landscape) uses honest numerical computation rather than a hand sketch.
The script is idempotent: re-running overwrites byte-for-byte identically
(inherited ``svg.hashsalt`` + ``metadata={'Date': None}`` from the shared
module's ``save()``).
"""

from __future__ import annotations

import os
import sys

# Import the shared style/palette/helpers from the Linear Algebra generator so
# this chapter matches it exactly (same rcParams, same arrow()/clean_axes()/
# axis_cross(), same deterministic save()).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # noqa: E402

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT


# =========================================================================== #
# 3.1 Gradient-Based Optimization                                             #
# =========================================================================== #

def fig_gd_bowl_vs_valley():
    """The headline contrast of the whole chapter (sec_mdl-gradient-based-
    optimization, 3.1.2-3.1.3): real gradient descent on f(x)=1/2 x^T A x.

    (a) Well-conditioned, isotropic bowl (kappa=1): near-circular contours, the
        path runs essentially straight to the minimum.
    (b) Ill-conditioned valley (kappa>>1): elongated contours; with a step size
        near the stability ceiling the iterates zig-zag across the steep axis
        while crawling along the slow lambda_min axis -- the slowest mode sets
        the rate.
    """
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.0, 4.2))

    g = np.linspace(-1.7, 1.7, 220)
    X, Y = np.meshgrid(g, g)

    def run(ax, A, title, eta, x0, steps):
        Z = 0.5 * (A[0, 0] * X**2 + (A[0, 1] + A[1, 0]) * X * Y + A[1, 1] * Y**2)
        ax.contour(X, Y, Z, levels=14, colors=[LIGHT], linewidths=0.9)
        x = np.array(x0, float)
        path = [x.copy()]
        for _ in range(steps):
            x = x - eta * (A @ x)  # grad of 1/2 x^T A x  (A symmetric)
            path.append(x.copy())
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], "-o", color=ORANGE, ms=3.5, lw=1.6,
                zorder=4)
        ax.plot(0, 0, "*", color=GREEN, ms=13, zorder=5)
        ax.set_title(title)
        clean_lim = ((-1.7, 1.7), (-1.7, 1.7))
        fl.clean_axes(ax, lim=clean_lim, hide=True)
        return path

    # (a) well-conditioned: lambda = (1, 1.3), kappa ~ 1; safe step -> ~straight.
    run(axa, np.array([[1.0, 0.0], [0.0, 1.3]]),
        r"(a) well-conditioned  ($\kappa\approx1$)",
        eta=0.55, x0=(-1.35, 1.15), steps=10)

    # (b) ill-conditioned: lambda = (1, 20), kappa = 20; eta near 2/lambda_max
    #     -> the steep mode oscillates, the flat mode barely moves.
    A = np.array([[1.0, 0.0], [0.0, 20.0]])
    run(axb, A, r"(b) ill-conditioned  ($\kappa\gg1$)",
        eta=0.097, x0=(-1.4, 1.0), steps=26)
    # annotate the slow (lambda_min) axis along which progress crawls.
    axb.annotate("", xy=(1.45, 0.0), xytext=(-1.45, 0.0),
                 arrowprops=dict(arrowstyle="<->", color=GRAY, lw=1.1,
                                 linestyle=(0, (4, 3))))
    axb.text(1.05, 0.20, r"slow axis ($\lambda_{\min}$)", color=GRAY,
             fontsize=9, ha="center", va="bottom")

    save = fl.save
    save(fig, "mdl-opt-gd-bowl-vs-valley")


# =========================================================================== #
# 3.2 Convex Sets and Convex Functions                                        #
# =========================================================================== #

def fig_chord_above_graph():
    """Two equivalent lenses on convexity of a real convex function f(x)=1/4 x^2
    (sec_mdl-convexity, 3.2.2):

    (a) Chord lens: a chord between two points on the graph lies *above* the
        graph -- f(theta x + (1-theta) y) <= theta f(x) + (1-theta) f(y).
    (b) First-order lens: the tangent at a point lies *below* the graph -- the
        tangent is a global under-estimator, f(y) >= f(x) + f'(x)(y-x).
    """
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.0, 4.0))

    f = lambda t: 0.25 * t**2
    fp = lambda t: 0.5 * t            # derivative
    xs = np.linspace(-3.2, 3.2, 400)

    # ----- (a) chord above the graph -----
    axa.set_title("(a) chord lens")
    axa.plot(xs, f(xs), color=BLUE, lw=2.2, zorder=3)
    a, b = -2.4, 2.8
    pa, pb = np.array([a, f(a)]), np.array([b, f(b)])
    axa.plot([pa[0], pb[0]], [pa[1], pb[1]], color=ORANGE, lw=2.2, zorder=3)
    axa.plot([pa[0], pb[0]], [pa[1], pb[1]], "o", color=ORANGE, ms=5, zorder=4)
    # the convex-combination point on the chord vs. on the graph
    theta = 0.6
    xc = theta * a + (1 - theta) * b
    chord_y = theta * f(a) + (1 - theta) * f(b)
    graph_y = f(xc)
    axa.plot([xc, xc], [graph_y, chord_y], "--", color=GRAY, lw=1.4, zorder=2)
    axa.plot(xc, graph_y, "o", color=BLUE, ms=5, zorder=5)
    axa.plot(xc, chord_y, "o", color=ORANGE, ms=5, zorder=5)
    fl.vlabel(axa, (a - 0.05, f(a) + 0.30), r"$\mathbf{x}$", color=BLUE,
              ha="right", fontsize=11)
    fl.vlabel(axa, (b + 0.10, f(b) + 0.15), r"$\mathbf{y}$", color=BLUE,
              ha="left", fontsize=11)
    axa.text(xc - 0.15, chord_y + 0.55,
             r"$\theta f(\mathbf{x})+(1-\theta)f(\mathbf{y})$",
             color=ORANGE, fontsize=9.5, ha="center", va="bottom")
    axa.text(xc - 0.15, graph_y - 0.62, r"$f(\theta\mathbf{x}+(1-\theta)\mathbf{y})$",
             color=BLUE, fontsize=9.5, ha="center", va="top")
    fl.axis_cross(axa, (-3.6, 3.6), (-0.6, 3.0))
    fl.clean_axes(axa, lim=((-3.6, 3.6), (-0.9, 3.0)), hide=True)

    # ----- (b) tangent below the graph -----
    axb.set_title("(b) first-order lens")
    axb.plot(xs, f(xs), color=BLUE, lw=2.2, zorder=3)
    x0 = -1.4
    tang = f(x0) + fp(x0) * (xs - x0)
    axb.plot(xs, tang, color=GREEN, lw=2.0, zorder=2)
    axb.plot(x0, f(x0), "o", color=BLUE, ms=6, zorder=5)
    # mark the gap f(y) - [tangent at y] >= 0 at some y
    y0 = 1.9
    axb.plot([y0, y0], [f(x0) + fp(x0) * (y0 - x0), f(y0)], "--",
             color=GRAY, lw=1.4, zorder=2)
    axb.plot(y0, f(y0), "o", color=BLUE, ms=5, zorder=5)
    fl.vlabel(axb, (x0 - 0.05, f(x0) + 0.32), r"$\mathbf{x}$", color=BLUE,
              ha="right", fontsize=11)
    fl.vlabel(axb, (y0 + 0.12, f(y0) + 0.10), r"$\mathbf{y}$", color=BLUE,
              ha="left", fontsize=11)
    xlab = 2.15
    axb.text(xlab, f(x0) + fp(x0) * (xlab - x0) - 0.18,
             r"$f(\mathbf{x})+\nabla f(\mathbf{x})^\top(\mathbf{y}-\mathbf{x})$",
             color=GREEN, fontsize=8.5, ha="center", va="top")
    fl.axis_cross(axb, (-3.6, 3.6), (-0.6, 3.0))
    fl.clean_axes(axb, lim=((-3.6, 3.6), (-1.4, 3.0)), hide=True)

    fl.save(fig, "mdl-opt-chord-above-graph")


def fig_convex_vs_nonconvex_set():
    """Convex vs. non-convex sets (sec_mdl-convexity, 3.2.1):

    (a) a convex blob -- the segment between any two of its points stays inside;
    (b) a non-convex crescent -- a segment between two interior points exits;
    (c) the probability simplex (a hyperplane cut of the nonnegative orthant)
        and a half-space, the two convex sets DL uses most.
    """
    fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(11.0, 3.7))

    # ----- (a) convex blob: a regular-ish convex polygon -----
    axa.set_title("(a) convex")
    ang = np.linspace(0, 2 * np.pi, 9)[:-1]
    r = 1.0 + 0.12 * np.cos(3 * ang)        # gently lumpy but still convex
    blob = np.column_stack([r * np.cos(ang), r * np.sin(ang)])
    # convex hull keeps it provably convex
    from scipy.spatial import ConvexHull
    hull = ConvexHull(blob)
    poly = blob[hull.vertices]
    axa.add_patch(plt.Polygon(poly, closed=True, facecolor=BLUE, alpha=0.16,
                              edgecolor=BLUE, lw=2.0))
    p, q = np.array([-0.7, 0.45]), np.array([0.75, -0.45])
    axa.plot([p[0], q[0]], [p[1], q[1]], color=ORANGE, lw=2.0, zorder=3)
    axa.plot([p[0], q[0]], [p[1], q[1]], "o", color=ORANGE, ms=5, zorder=4)
    fl.clean_axes(axa, lim=((-1.45, 1.45), (-1.35, 1.45)), hide=True)

    # ----- (b) non-convex crescent: chord exits the set -----
    axb.set_title("(b) non-convex")
    t = np.linspace(0, 2 * np.pi, 200)
    outer = np.column_stack([np.cos(t), np.sin(t)])
    # crescent = big disc minus a shifted disc (boolean via polygon trick)
    th = np.linspace(-np.pi / 2.0, np.pi / 2.0, 100)
    right = np.column_stack([np.cos(th), np.sin(th)])           # right rim
    bite = np.column_stack([0.7 + 0.95 * np.cos(th[::-1]),
                            0.95 * np.sin(th[::-1])])           # inner bite
    cres = np.vstack([right, bite])
    axb.add_patch(plt.Polygon(cres, closed=True, facecolor=ORANGE, alpha=0.16,
                              edgecolor=ORANGE, lw=2.0))
    # a chord between two points of the crescent that passes outside it
    p2, q2 = np.array([0.15, 0.82]), np.array([0.15, -0.82])
    axb.plot([p2[0], q2[0]], [p2[1], q2[1]], color=BLUE, lw=2.0, zorder=3)
    axb.plot([p2[0], q2[0]], [p2[1], q2[1]], "o", color=BLUE, ms=5, zorder=4)
    axb.plot(0.15, 0.0, "x", color=BLUE, ms=9, mew=2.2, zorder=5)  # midpoint outside
    axb.text(0.36, 0.0, "outside", color=BLUE, fontsize=8.5, ha="left",
             va="center")
    fl.clean_axes(axb, lim=((-1.25, 1.45), (-1.25, 1.25)), hide=True)

    # ----- (c) simplex + half-space -----
    axc.set_title("(c) simplex, half-space")
    simplex = np.array([[0.0, 0.0], [1.7, 0.0], [0.0, 1.7]])
    axc.add_patch(plt.Polygon(simplex, closed=True, facecolor=GREEN, alpha=0.18,
                              edgecolor=GREEN, lw=2.0))
    axc.text(0.52, 0.5, r"$\{\mathbf{p}\succeq0,\ \mathbf{1}^\top\mathbf{p}=1\}$",
             color=GREEN, fontsize=8.5, ha="center", va="center")
    # half-space a^T x <= b, shaded
    aa = np.array([1.0, 0.7])
    aa_n = aa / np.linalg.norm(aa)
    bb = -0.55
    L = 2.4
    perp = np.array([-aa_n[1], aa_n[0]])
    c0 = bb * aa_n
    seg = np.array([c0 - L * perp, c0 + L * perp])
    far = -aa_n * (2 * L)  # into the half-space a^T x <= b
    poly = np.array([seg[0], seg[1], seg[1] + far, seg[0] + far])
    axc.add_patch(plt.Polygon(poly, closed=True, facecolor=BLUE, alpha=0.10,
                              lw=0))
    axc.plot(seg[:, 0], seg[:, 1], "--", color=BLUE, lw=1.8)
    axc.text(-0.95, -0.95, r"$\mathbf{a}^\top\mathbf{x}\leq b$", color=BLUE,
             fontsize=9, ha="center", va="center")
    fl.axis_cross(axc, (-1.4, 2.2), (-1.4, 2.2))
    fl.clean_axes(axc, lim=((-1.4, 2.2), (-1.4, 2.2)), hide=True)

    fl.save(fig, "mdl-opt-convex-vs-nonconvex-set")


def fig_local_equals_global():
    """Why convexity matters (sec_mdl-convexity, 3.2.4): a convex objective has
    one global minimum and no traps; a non-convex one has several local minima
    and a saddle, so gradient descent's limit depends on where it starts.

    (a) Convex f(x)=1/4 x^2: GD from two starts converges to the same minimum.
    (b) Non-convex multi-well landscape: real GD from two starts lands in two
        different local minima; a strict saddle separates the basins.
    """
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.0, 4.0))

    # ----- (a) convex bowl: every start -> the single global min -----
    axa.set_title("(a) convex: local $=$ global")
    f = lambda t: 0.25 * t**2
    fp = lambda t: 0.5 * t
    xs = np.linspace(-3.4, 3.4, 400)
    axa.plot(xs, f(xs), color=BLUE, lw=2.2, zorder=2)

    def gd1d(x0, fp, eta, steps):
        x = float(x0)
        xs_ = [x]
        for _ in range(steps):
            x = x - eta * fp(x)
            xs_.append(x)
        return np.array(xs_)

    for x0, col in ((-3.0, ORANGE), (3.1, GREEN)):
        p = gd1d(x0, fp, eta=0.7, steps=11)
        axa.plot(p, f(p), "-o", color=col, ms=3.2, lw=1.3, zorder=3)
    axa.plot(0, 0, "*", color="black", ms=14, zorder=5)
    fl.axis_cross(axa, (-3.8, 3.8), (-0.5, 3.0))
    fl.clean_axes(axa, lim=((-3.8, 3.8), (-0.7, 3.0)), hide=True)

    # ----- (b) non-convex: two basins + a saddle (a 1-D bumpy landscape) -----
    axb.set_title("(b) non-convex: start matters")
    # a smooth double-well-plus landscape with two distinct minima and a hump
    g = lambda t: 0.12 * t**4 - 0.9 * t**2 + 0.35 * t + 1.3
    gp = lambda t: 0.48 * t**3 - 1.8 * t + 0.35
    ts = np.linspace(-3.2, 3.2, 500)
    axb.plot(ts, g(ts), color=BLUE, lw=2.2, zorder=2)

    # locate the critical points numerically (roots of g') to label them
    roots = np.sort(np.roots([0.48, 0.0, -1.8, 0.35]).real)
    # classify by second derivative g''(t) = 1.44 t^2 - 1.8
    gpp = lambda t: 1.44 * t**2 - 1.8
    minima = [r for r in roots if gpp(r) > 0]
    maxima = [r for r in roots if gpp(r) < 0]  # the separating hump (1-D "saddle")

    for x0, col in ((-2.7, ORANGE), (2.7, GREEN)):
        p = gd1d(x0, gp, eta=0.18, steps=40)
        axb.plot(p, g(p), "-o", color=col, ms=2.6, lw=1.1, zorder=3, alpha=0.9)
    for m in minima:
        axb.plot(m, g(m), "*", color="black", ms=12, zorder=5)
    for m in maxima:
        axb.plot(m, g(m), "o", color=GRAY, ms=7, zorder=5)
        axb.text(m, g(m) + 0.22, "saddle", color=GRAY, fontsize=8.5,
                 ha="center", va="bottom")
    axb.text(minima[0], g(minima[0]) - 0.30, "local min", color="black",
             fontsize=8.5, ha="center", va="top")
    axb.text(minima[-1], g(minima[-1]) - 0.30, "global min", color="black",
             fontsize=8.5, ha="center", va="top")
    fl.clean_axes(axb, lim=((-3.4, 3.4), (-1.1, 3.0)), hide=True)
    axb.axis("off")

    fl.save(fig, "mdl-opt-local-equals-global")


# =========================================================================== #
# Driver                                                                      #
# =========================================================================== #

FIGURES = [
    # 3.1 gradient-based optimization
    fig_gd_bowl_vs_valley,
    # 3.2 convexity
    fig_chord_above_graph,
    fig_convex_vs_nonconvex_set,
    fig_local_equals_global,
]


def main():
    # save() appends to fl.WRITTEN; start from a clean slate so the report below
    # lists only this chapter's figures.
    fl.WRITTEN.clear()
    for fn in FIGURES:
        fn()

    print(f"\nWrote {len(fl.WRITTEN)} figures to {fl.IMG_DIR}:")
    for p in fl.WRITTEN:
        size = os.path.getsize(p)
        assert os.path.exists(p), f"missing: {p}"
        assert size > 0, f"empty: {p}"
        with open(p, "r", encoding="utf-8") as fh:
            head = fh.read(400)
        assert "<svg" in head, f"not valid SVG: {p}"
        print(f"  {os.path.basename(p):34s} {size:>8,d} bytes")

    print(f"\nAll {len(fl.WRITTEN)} SVGs verified present, non-empty, valid.")


if __name__ == "__main__":
    main()
