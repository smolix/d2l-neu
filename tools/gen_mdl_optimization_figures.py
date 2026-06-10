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


def fig_momentum_damping():
    """Heavy-ball momentum as a damped oscillator (sec_mdl-gradient-based-
    optimization, Momentum and Acceleration): three real heavy-ball runs on the
    ill-conditioned valley f(x, y) = 1/2 (x^2 + 10 y^2), all with the *same*
    step size, differing only in the damping knob beta:

    - beta = 0.05: over-damped -- zig-zags briefly, then crawls along the slow
      axis like plain gradient descent;
    - beta = beta* = ((sqrt(10)-1)/(sqrt(10)+1))^2 ~ 0.27: critically tuned --
      cuts through the valley fastest (the sqrt(kappa) tuning);
    - beta = 0.85: under-damped -- overshoots the minimum and loops around it.

    Note on the shared step size: the figure spec suggested eta* =
    (2/(1+sqrt(10)))^2 ~ 0.231 for all three runs, but heavy ball with
    beta = 0.05 *diverges* at that step (stability needs
    |1 + beta - eta*lambda_max| < 1 + beta, i.e. 2.31 < 2.10 fails).  We use
    the GD-optimal eta = 2/(lambda_max + lambda_min) = 2/11 instead, which is
    stable for all three betas and makes the over-damped run genuinely
    GD-like, as the caption describes.
    """
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.3), sharey=True)

    lam = np.array([1.0, 10.0])               # Hessian eigenvalues; kappa = 10
    gx = np.linspace(-2.45, 2.45, 320)
    gy = np.linspace(-1.35, 1.35, 200)
    X, Y = np.meshgrid(gx, gy)
    Z = 0.5 * (lam[0] * X**2 + lam[1] * Y**2)

    eta = 2.0 / (lam[0] + lam[1])             # GD-optimal step, ~0.182
    beta_star = ((np.sqrt(10) - 1) / (np.sqrt(10) + 1)) ** 2  # ~0.270

    def heavy_ball(beta, steps, x0=(-2.0, 1.0)):
        x_prev = np.array(x0, float)
        x = x_prev.copy()
        path = [x.copy()]
        for _ in range(steps):
            x_next = x - eta * (lam * x) + beta * (x - x_prev)
            x_prev, x = x, x_next
            path.append(x.copy())
        return np.array(path)

    runs = [
        (0.05, 34, BLUE, r"over-damped  $\beta=0.05$"),
        (beta_star, 22, GREEN, r"critical  $\beta^\star\approx0.27$"),
        (0.85, 60, ORANGE, r"under-damped  $\beta=0.85$"),
    ]
    # one regime per panel: the three paths overlap hopelessly on shared axes
    for ax, (beta, steps, col, lab) in zip(axes, runs):
        ax.contour(X, Y, Z, levels=[0.04, 0.16, 0.42, 0.9, 1.7, 2.9, 4.6, 7.0],
                   colors=[LIGHT], linewidths=0.9)
        p = heavy_ball(beta, steps)
        ax.plot(p[:, 0], p[:, 1], "-o", color=col, ms=2.6, lw=1.1, zorder=4)
        ax.plot(*p[0], "o", color=col, ms=5.5, zorder=5)
        ax.plot(0, 0, "*", color="black", ms=12, zorder=6)
        ax.set_title(lab, color=col, fontsize=10.5)
        fl.clean_axes(ax, lim=((-2.45, 2.45), (-1.35, 1.45)), hide=True)

    fig.subplots_adjust(wspace=0.06)
    fl.save(fig, "mdl-opt-momentum-damping")


def fig_sgd_noise_ball():
    """GD versus fixed-step SGD on the strongly convex bowl f = 1/2 ||x||^2
    (sec_mdl-gradient-based-optimization, Stochastic Gradients): the full-batch
    path converges to the minimizer; SGD with the same step descends like GD at
    first, then rattles inside a noise ball of squared radius ~ eta sigma^2 /
    (2 lambda).  A second dashed circle at radius/sqrt(2) shows the ball after
    halving eta.  Both dashed circles are drawn at 3x the analytic RMS radius
    (a shared visual scale, per the spec, so the ball legibly contains the
    stationary scatter); their *ratio* -- the point of the picture -- is exact.
    """
    fig, ax = plt.subplots(figsize=(6.6, 4.6))

    lam, eta, sigma = 1.0, 0.15, 0.6
    g = np.linspace(-2.6, 2.6, 240)
    X, Y = np.meshgrid(g, g)
    Z = 0.5 * (X**2 + Y**2)
    ax.contour(X, Y, Z, levels=[0.18, 0.5, 1.0, 1.7, 2.6, 3.6],
               colors=[LIGHT], linewidths=0.9)

    x0 = np.array([-2.0, 1.6])

    # full-batch GD: smooth geometric decay straight into the origin
    steps = 80
    gd = [x0.copy()]
    x = x0.copy()
    for _ in range(steps):
        x = x - eta * lam * x
        gd.append(x.copy())
    gd = np.array(gd)
    ax.plot(gd[:, 0], gd[:, 1], "-o", color=BLUE, ms=2.6, lw=1.5, zorder=4)

    # SGD: same step, gradient + N(0, sigma^2 I) noise (seeded -> reproducible)
    rng = np.random.default_rng(7)
    sgd = [x0.copy()]
    x = x0.copy()
    for _ in range(steps):
        x = x - eta * (lam * x + sigma * rng.standard_normal(2))
        sgd.append(x.copy())
    sgd = np.array(sgd)
    cut = 30                                  # descent transient as a line ...
    ax.plot(sgd[:cut + 1, 0], sgd[:cut + 1, 1], "-", color=ORANGE, lw=1.3,
            zorder=4)
    ax.plot(sgd[cut:, 0], sgd[cut:, 1], "o", color=ORANGE, ms=3.2, lw=0,
            alpha=0.8, zorder=5)              # ... last 50 iterates as scatter

    ax.plot(0, 0, "*", color=GREEN, ms=12, zorder=3)

    # noise-ball circles: r = sqrt(eta sigma^2 / (2 lambda)), and r/sqrt(2)
    # for step eta/2; both at the shared 3x visual scale.
    scale = 3.0
    r = scale * np.sqrt(eta * sigma**2 / (2 * lam))
    for radius, ls in ((r, (0, (5, 3))), (r / np.sqrt(2), (0, (3, 3)))):
        ax.add_patch(plt.Circle((0, 0), radius, fill=False, edgecolor=GRAY,
                                lw=1.3, linestyle=ls, zorder=6))
    ax.annotate(r"noise ball $\propto\sqrt{\eta}$",
                xy=(r * np.cos(0.6), -r * np.sin(0.6)), xytext=(1.05, -1.25),
                color=GRAY, fontsize=9.5, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.9))
    ax.annotate(r"step $\eta/2$",
                xy=(-(r / np.sqrt(2)) * np.cos(0.8),
                    -(r / np.sqrt(2)) * np.sin(0.8)), xytext=(-1.85, -1.25),
                color=GRAY, fontsize=9.5, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.9))

    ax.text(-1.78, 1.42, "gradient descent", color=BLUE, fontsize=9.5,
            ha="center", va="bottom")
    ax.text(-0.45, 0.95, r"SGD, fixed $\eta$", color=ORANGE, fontsize=9.5,
            ha="left", va="bottom")

    fl.clean_axes(ax, lim=((-2.45, 2.3), (-1.7, 1.95)), hide=True)
    fl.save(fig, "mdl-opt-sgd-noise-ball")


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
# 3.3 Constrained Optimization and Duality                                    #
# =========================================================================== #

def fig_lagrange_tangency():
    """The Lagrange-multiplier picture (sec_mdl-constrained-optimization-
    duality, Equality Constraints): minimizing f along the constraint curve
    g(x) = 0.  Real geometry: the constraint is the unit circle, f(x) =
    1/2 ||x - c||^2 with c = 2 x_hat outside it, so the constrained minimizer
    is x* = c/||c|| and there grad f = -(||c||-1) x* is exactly anti-parallel
    to grad g = 2 x* (grad f = -nu grad g with nu > 0).  At a non-optimal
    feasible point the level set of f *crosses* the constraint: grad f has a
    component along the curve, so sliding against it is a feasible descent.
    """
    fig, ax = plt.subplots(figsize=(6.0, 5.2))

    c = 2.0 * np.array([0.8, 0.6])            # ||c|| = 2 -> x* = (0.8, 0.6)
    xstar = c / np.linalg.norm(c)

    # constraint circle g(x) = ||x||^2 - 1 = 0
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(t), np.sin(t), color=BLUE, lw=2.2, zorder=3)

    # level sets of f: circles around c; r = 1 is *tangent* to the constraint
    # at x* (distance of centers 2 = 1 + 1), r = 2.43 passes through x0.
    x0 = np.array([np.cos(np.deg2rad(140.0)), np.sin(np.deg2rad(140.0))])
    radii = [1.0, 1.45, 1.9, float(np.linalg.norm(x0 - c))]
    for r, a in zip(radii, (0.95, 0.65, 0.45, 0.65)):
        ax.add_patch(plt.Circle(c, r, fill=False, edgecolor=ORANGE, lw=1.3,
                                alpha=a, zorder=2))
    ax.plot(*c, "+", color=ORANGE, ms=8, mew=1.6, zorder=3)

    # ---- the optimum: tangency, gradients parallel (grad f = -nu grad g) ----
    ax.plot(*xstar, "o", color="black", ms=6.5, zorder=6)
    fl.arrow(ax, xstar, xstar + 0.60 * xstar, color=BLUE, lw=2.0)      # grad g
    fl.arrow(ax, xstar, xstar - 0.50 * xstar, color=ORANGE, lw=2.0)    # grad f
    fl.vlabel(ax, xstar + 0.74 * xstar + np.array([0.16, -0.10]),
              r"$\nabla g$", color=BLUE, fontsize=10)
    fl.vlabel(ax, xstar - 0.50 * xstar + np.array([-0.02, -0.20]),
              r"$\nabla f$", color=ORANGE, fontsize=10)
    fl.vlabel(ax, xstar + np.array([0.17, -0.14]), r"$\mathbf{x}^\star$",
              color="black", fontsize=11)

    # ---- a non-optimal feasible point: grad f crosses the constraint ----
    ax.plot(*x0, "o", color="black", ms=5.5, zorder=6)
    df = (x0 - c) / np.linalg.norm(x0 - c)    # unit grad f at x0
    fl.arrow(ax, x0, x0 + 0.55 * df, color=ORANGE, lw=2.0)
    fl.vlabel(ax, x0 + 0.72 * df + np.array([0.0, 0.12]), r"$\nabla f$",
              color=ORANGE, fontsize=10)
    # feasible descent: minus the tangential component of grad f (first-order
    # move along the constraint), drawn dashed
    tang = np.array([-x0[1], x0[0]])          # unit tangent at x0
    desc = -np.dot(df, tang) * tang
    desc = 0.5 * desc / np.linalg.norm(desc)
    fl.arrow(ax, x0, x0 + desc, color=GREEN, lw=1.8, ls=(0, (4, 3)))
    fl.vlabel(ax, x0 + desc + np.array([0.04, 0.20]), "feasible\ndescent",
              color=GREEN, fontsize=9)

    ax.text(-0.55, -1.05, r"constraint $g(\mathbf{x})=0$", color=BLUE,
            fontsize=10, ha="center", va="top")
    ax.text(1.62, 2.12, r"level sets of $f$", color=ORANGE, fontsize=10,
            ha="left", va="center")

    fl.clean_axes(ax, lim=((-1.95, 2.75), (-1.45, 2.45)), hide=True)
    fl.save(fig, "mdl-opt-lagrange-tangency")


def fig_kkt_active_set():
    """Geometry of the KKT conditions (sec_mdl-constrained-optimization-
    duality, Inequalities and KKT) on the feasible region {g1 <= 0, g2 <= 0}
    with g1 = y - 1, g2 = x - 1.5 (outward normals n1 = (0,1), n2 = (1,0)).
    Real geometry, f(x) = 1/2 ||x - c||^2 in both panels:

    (a) c = (0.4, 1.9) projects onto the interior of edge g1 = 0: one active
        constraint, -grad f = lambda_1 grad g1 with lambda_1 > 0, while g2 is
        slack and lambda_2 = 0;
    (b) c = (2.3, 1.8) projects onto the corner: both constraints active, and
        -grad f lies inside the cone spanned by the two outward normals.
    """
    from matplotlib.patches import Wedge

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.2, 4.1))
    xlim, ylim = (-0.7, 2.75), (-0.55, 2.35)
    e1y, e2x = 1.0, 1.5                       # edges y = 1 and x = 1.5

    def region(ax):
        ax.add_patch(plt.Polygon(
            [[xlim[0], ylim[0]], [e2x, ylim[0]], [e2x, e1y],
             [xlim[0], e1y]], closed=True, facecolor=BLUE, alpha=0.12, lw=0))
        ax.plot([xlim[0], e2x], [e1y, e1y], color=BLUE, lw=2.0)   # g1 = 0
        ax.plot([e2x, e2x], [ylim[0], e1y], color=BLUE, lw=2.0)   # g2 = 0
        ax.text(0.05, 0.18, "feasible", color=BLUE, fontsize=9.5,
                ha="center", va="center")
        ax.text(-0.62, 1.10, r"$g_1=0$", color=BLUE, fontsize=9.5, ha="left",
                va="bottom")
        ax.text(1.60, -0.38, r"$g_2=0$", color=BLUE, fontsize=9.5, ha="left",
                va="center")

    def levels(ax, c, radii):
        for r, a in zip(radii, (0.9, 0.65, 0.4)):
            ax.add_patch(plt.Circle(c, r, fill=False, edgecolor=ORANGE,
                                    lw=1.2, alpha=a, zorder=2))
        ax.plot(*c, "+", color=ORANGE, ms=8, mew=1.6, zorder=3)

    # ---- (a) one constraint active ----
    axa.set_title("(a) one active constraint")
    region(axa)
    ca = np.array([0.4, 1.9])
    xs = np.array([ca[0], e1y])               # projection of ca on the region
    levels(axa, ca, [float(np.linalg.norm(ca - xs)), 1.35, 1.8])
    axa.plot(*xs, "o", color="black", ms=6.5, zorder=6)
    fl.arrow(axa, xs, xs + np.array([0.0, -0.62]), color=ORANGE, lw=2.0)
    axa.text(xs[0] + 0.10, xs[1] - 0.50,
             r"$-\nabla f=\lambda_1\nabla g_1,\ \lambda_1>0$",
             color=ORANGE, fontsize=9, ha="left", va="center")
    fl.vlabel(axa, xs + np.array([-0.22, 0.14]), r"$\mathbf{x}^\star$",
              color="black", fontsize=11)
    # outward normal of the active edge, shown a bit along the edge
    fl.arrow(axa, (-0.30, e1y), (-0.30, e1y + 0.45), color=BLUE, lw=1.8)
    fl.vlabel(axa, (-0.30, e1y + 0.62), r"$\nabla g_1$", color=BLUE,
              fontsize=9.5)
    axa.text(e2x + 0.13, 0.42, r"inactive: $\lambda_2=0$", color=GRAY,
             fontsize=9, ha="left", va="center", rotation=90)

    # ---- (b) corner: two constraints active ----
    axb.set_title("(b) corner: two active constraints")
    region(axb)
    cb = np.array([2.3, 1.8])
    corner = np.array([e2x, e1y])             # projection of cb on the region
    levels(axb, cb, [float(np.linalg.norm(cb - corner)), 1.55, 1.95])
    axb.add_patch(Wedge(corner, 0.62, 0.0, 90.0, facecolor=BLUE, alpha=0.18,
                        lw=0, zorder=3))
    axb.plot(*corner, "o", color="black", ms=6.5, zorder=6)
    fl.arrow(axb, corner, corner + np.array([0.0, 0.55]), color=BLUE, lw=1.8)
    fl.arrow(axb, corner, corner + np.array([0.55, 0.0]), color=BLUE, lw=1.8)
    df = (cb - corner) / np.linalg.norm(cb - corner)   # -grad f direction
    fl.arrow(axb, corner, corner + 0.62 * df, color=ORANGE, lw=2.2)
    fl.vlabel(axb, corner + np.array([-0.05, 0.70]), r"$\nabla g_1$",
              color=BLUE, fontsize=9.5, ha="right")
    fl.vlabel(axb, corner + np.array([0.72, -0.16]), r"$\nabla g_2$",
              color=BLUE, fontsize=9.5)
    axb.text(*(corner + 0.62 * df + np.array([0.07, 0.16])), r"$-\nabla f$",
             color=ORANGE, fontsize=9.5, ha="left", va="center")
    axb.text(corner[0] + 0.30, corner[1] - 0.40, "cone of\nnormals",
             color=GRAY, fontsize=8.5, ha="center", va="center")
    axb.annotate("", xy=(corner[0] + 0.30, corner[1] + 0.22),
                 xytext=(corner[0] + 0.30, corner[1] - 0.22),
                 arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8))
    fl.vlabel(axb, corner + np.array([-0.22, -0.16]), r"$\mathbf{x}^\star$",
              color="black", fontsize=11)

    for ax in (axa, axb):
        fl.clean_axes(ax, lim=(xlim, ylim), hide=True)
    fl.save(fig, "mdl-opt-kkt-active-set")


def fig_primal_dual_gap():
    """The supporting-line picture of duality (sec_mdl-constrained-
    optimization-duality, Duality): map every x to (u, t) = (constraint value,
    objective value); evaluating the dual at lambda lowers a line of slope
    -lambda until it supports the image from below, and its height at u = 0 is
    g(lambda) <= p*.  Both panels are exact computations:

    (a) convex (min x^2 s.t. 1 - x <= 0): the image is the parabola
        t = (1-u)^2; the best supporting line (lambda* = 2) touches it at
        (0, p*) -- strong duality, d* = p* = 1;
    (b) the chapter's non-convex example (min -x^2 s.t. x - 1/2 <= 0 on
        [0, 1]): the image t = -(u + 1/2)^2 is dented from below, every
        supporting line passes under the dent at u = 0, and the best one
        (lambda* = 1) certifies only d* = -1/2 < p* = -1/4: a duality gap.
    """
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.2, 4.1))

    # ---- (a) convex: strong duality ----
    axa.set_title("(a) convex $+$ Slater: strong duality")
    u = np.linspace(-0.85, 1.45, 300)
    axa.plot(u, (1 - u) ** 2, color=BLUE, lw=2.2, zorder=3)
    uf = u[u <= 0]
    axa.plot(uf, (1 - uf) ** 2, color=ORANGE, lw=3.2, zorder=4)
    p_star_a = 1.0                            # min over u <= 0 of (1-u)^2
    axa.axhline(p_star_a, color=GRAY, lw=1.2, ls=(0, (5, 3)), zorder=2)
    # best supporting line, slope -lambda* = -2: tangent at (0, p*)
    us = np.linspace(-0.85, 1.0, 2)
    axa.plot(us, 1 - 2 * us, color=GREEN, lw=1.8, zorder=3)
    # a suboptimal supporting line (lambda = 1): height at u=0 is g(1) = 3/4
    axa.plot(us, 0.75 - us, color=GRAY, lw=1.1, ls=(0, (3, 3)), zorder=2)
    axa.plot(0, 0.75, "o", color=GRAY, ms=4.5, zorder=5)
    axa.text(0.09, 0.66, r"$g(\lambda)$", color=GRAY, fontsize=9, ha="left",
             va="top")
    axa.plot(0, p_star_a, "o", color="black", ms=6, zorder=6)
    axa.text(0.09, 1.13, r"$d^\star=p^\star$", color="black", fontsize=9.5,
             ha="left", va="bottom")
    axa.text(-0.85, 1.06, r"$p^\star$", color=GRAY, fontsize=10, ha="left",
             va="bottom")
    axa.text(-0.62, 2.95, r"feasible: $u\leq0$", color=ORANGE, fontsize=9,
             ha="center", va="bottom")
    axa.text(0.78, -0.78, r"slope $-\lambda^\star$", color=GREEN, fontsize=9,
             ha="center", va="center", rotation=-39)
    fl.axis_cross(axa, (-1.05, 1.6), (-1.3, 3.6))
    axa.text(1.58, -0.18, "constraint value $u$", color=GRAY, fontsize=9,
             ha="right", va="top")
    axa.text(-0.06, 3.55, "objective value", color=GRAY, fontsize=9,
             ha="right", va="top")
    fl.clean_axes(axa, lim=((-1.05, 1.6), (-1.45, 3.7)), hide=True,
                  equal=False)

    # ---- (b) non-convex: a duality gap ----
    axb.set_title("(b) non-convex: duality gap")
    u = np.linspace(-0.5, 0.5, 300)           # image of x in [0, 1]
    axb.plot(u, -(u + 0.5) ** 2, color=BLUE, lw=2.2, zorder=3)
    uf = u[u <= 0]
    axb.plot(uf, -(uf + 0.5) ** 2, color=ORANGE, lw=3.2, zorder=4)
    p_star_b = -0.25                          # min over u <= 0: the dent
    axb.axhline(p_star_b, color=GRAY, lw=1.2, ls=(0, (5, 3)), zorder=2)
    # best supporting line, slope -lambda* = -1: touches both endpoints,
    # passes *under* the dent with height d* = -1/2 at u = 0
    us = np.linspace(-0.66, 0.56, 2)
    axb.plot(us, -0.5 - us, color=GREEN, lw=1.8, zorder=3)
    # a suboptimal supporting line (lambda = 2): g(2) = -1
    axb.plot(np.linspace(-0.62, 0.18, 2), -1 - 2 * np.linspace(-0.62, 0.18, 2),
             color=GRAY, lw=1.1, ls=(0, (3, 3)), zorder=2)
    axb.plot(0, p_star_b, "o", color="black", ms=6, zorder=6)
    axb.plot(0, -0.5, "o", color=GREEN, ms=6, zorder=6)
    axb.text(-0.045, -0.205, r"$p^\star$", color="black", fontsize=10,
             ha="right", va="bottom")
    axb.text(-0.045, -0.545, r"$d^\star$", color=GREEN, fontsize=10,
             ha="right", va="top")
    axb.annotate("", xy=(0.045, -0.255), xytext=(0.045, -0.495),
                 arrowprops=dict(arrowstyle="<->", color=GRAY, lw=1.1))
    axb.text(0.085, -0.375, "duality gap", color=GRAY, fontsize=9.5,
             ha="left", va="center")
    axb.text(-0.36, 0.10, r"feasible: $u\leq0$", color=ORANGE, fontsize=9,
             ha="center", va="bottom")
    axb.text(0.33, -0.93, r"slope $-\lambda^\star$", color=GREEN, fontsize=9,
             ha="center", va="center", rotation=-37)
    fl.axis_cross(axb, (-0.78, 0.72), (-1.32, 0.42))
    axb.text(0.70, -0.05, "constraint value $u$", color=GRAY, fontsize=9,
             ha="right", va="top")
    axb.text(-0.025, 0.40, "objective value", color=GRAY, fontsize=9,
             ha="right", va="top")
    fl.clean_axes(axb, lim=((-0.78, 0.72), (-1.38, 0.52)), hide=True,
                  equal=False)

    fl.save(fig, "mdl-opt-primal-dual-gap")


# =========================================================================== #
# 3.4 Numerical Stability and Conditioning                                    #
# =========================================================================== #

def fig_fp_number_line():
    """Floating-point numbers on the real line (sec_mdl-numerical-stability-
    conditioning, Floating-Point Arithmetic).  Schematic with a toy 2-bit
    mantissa: within each binade the representable values are evenly spaced
    and the gap doubles at every power of two; an arrow marks eps_mach (the
    gap from 1 to its right neighbor); below the smallest normal sits the
    shaded subnormal/underflow region; after an axis break, dashed verticals
    mark the overflow cliffs of fp16 (65504) and fp32 (~3.4e38).
    """
    fig, ax = plt.subplots(figsize=(8.4, 2.5))

    p = 2                                     # toy mantissa bits: 4/binade
    # binades [2^e, 2^{e+1}) for e = -2..1, i.e. [0.25, 4); subnormals fill
    # [0, 0.25) with the same gap as the smallest binade.
    ticks = []
    for e in range(-2, 2):
        lo = 2.0 ** e
        ticks += list(lo + (lo / 2**p) * np.arange(2**p))
    ticks.append(4.0)
    sub = (0.25 / 2**p) * np.arange(1, 2**p)  # subnormals

    ax.annotate("", xy=(7.05, 0), xytext=(-0.22, 0),
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.1))
    for v in ticks:
        tall = v in (0.25, 0.5, 1.0, 2.0, 4.0)
        ax.plot([v, v], [-0.13 if tall else -0.08, 0.13 if tall else 0.08],
                color=BLUE, lw=1.5 if tall else 1.1)
    for v in sub:
        ax.plot([v, v], [-0.08, 0.08], color=GRAY, lw=1.0, alpha=0.8)
    ax.plot([0, 0], [-0.13, 0.13], color=GRAY, lw=1.5)
    for v, lab in ((0, "0"), (0.25, "0.25"), (1, "1"), (2, "2"), (4, "4")):
        ax.text(v, -0.24, lab, color="black", fontsize=9, ha="center",
                va="top")

    # subnormal / underflow region below the smallest normal
    ax.add_patch(plt.Polygon([[0, -0.16], [0.25, -0.16], [0.25, 0.16],
                              [0, 0.16]], closed=True, facecolor=ORANGE,
                             alpha=0.15, lw=0))
    ax.annotate("subnormals $\\to$ underflow to $0$", xy=(0.12, 0.17),
                xytext=(0.02, 0.62), color=GRAY, fontsize=9, ha="left",
                va="center",
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.9))
    ax.text(0.25, 0.45, "smallest normal", color=GRAY, fontsize=8,
            ha="left", va="center", rotation=0)
    ax.annotate("", xy=(0.25, 0.16), xytext=(0.33, 0.38),
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8))

    # eps_mach: the gap from 1 to its right neighbor 1 + 2^-p
    eps = 2.0 ** -p
    ax.annotate("", xy=(1 + eps, 0.30), xytext=(1, 0.30),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.1))
    ax.text(1 + eps / 2, 0.40, r"$\varepsilon_{\mathrm{mach}}$",
            color="black", fontsize=10, ha="center", va="bottom")

    # gap doubling: matched double-arrows under one gap of each binade
    for lo in (1.0, 2.0):
        ax.annotate("", xy=(lo + lo / 2**p, -0.36), xytext=(lo, -0.36),
                    arrowprops=dict(arrowstyle="<->", color=GRAY, lw=1.0))
    ax.text(2.05, -0.52, "gap doubles", color=GRAY, fontsize=9, ha="center",
            va="top")

    # axis break, then the overflow cliffs (schematic positions)
    for dx in (4.62, 4.74):
        ax.plot([dx - 0.06, dx + 0.06], [-0.10, 0.10], color=GRAY, lw=1.2)
    x16, x32 = 5.45, 6.5
    ax.add_patch(plt.Polygon([[x16, -0.16], [7.0, -0.16], [7.0, 0.16],
                              [x16, 0.16]], closed=True, facecolor=ORANGE,
                             alpha=0.08, lw=0))
    ax.add_patch(plt.Polygon([[x32, -0.16], [7.0, -0.16], [7.0, 0.16],
                              [x32, 0.16]], closed=True, facecolor=ORANGE,
                             alpha=0.16, lw=0))
    for x, lab in ((x16, "fp16 max:\n$65504$"),
                   (x32, "fp32 max:\n$\\approx3.4\\times10^{38}$")):
        ax.plot([x, x], [-0.30, 0.30], color=ORANGE, lw=1.5, ls=(0, (5, 3)))
        ax.text(x, 0.38, lab, color=ORANGE, fontsize=8.5, ha="center",
                va="bottom")
    ax.text(6.22, -0.30, r"overflow $\to\infty$", color=ORANGE, fontsize=9,
            ha="center", va="top")

    fl.clean_axes(ax, lim=((-0.3, 7.15), (-0.85, 0.95)), hide=True,
                  equal=False)
    fl.save(fig, "mdl-opt-fp-number-line")


def fig_conditioning_ellipse():
    """Ridge rounds the valley into a bowl (sec_mdl-numerical-stability-
    conditioning, Conditioning): level sets of 1/2 w^T (Sigma^2 + lambda I) w
    with Sigma^2 = diag(1, 1/16), before (lambda = 0, kappa = 16) and after
    (lambda = 0.5, kappa ~ 2.7) adding the ridge term.  Real gradient descent
    at the per-panel optimal step eta = 2/(mu + L) draws both paths from the
    same start: zig-zag down the valley vs. a near-straight, much shorter run.
    Green arrows mark the principal axes, labelled with the curvatures
    sigma_1^2 + lambda and sigma_n^2 + lambda that set kappa's ratio.
    """
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.4, 5.0))
    s2 = np.array([1.0, 1.0 / 16.0])          # sigma_1^2, sigma_n^2
    xlim, ylim = (-2.2, 2.2), (-3.25, 3.25)
    gx = np.linspace(*xlim, 240)
    gy = np.linspace(*ylim, 300)
    X, Y = np.meshgrid(gx, gy)
    x0 = np.array([-0.72, 2.8])

    def panel(ax, ridge, title, steps, axis_labels):
        h = s2 + ridge                        # Hessian eigenvalues
        Z = 0.5 * (h[0] * X**2 + h[1] * Y**2)
        # levels chosen so the y-semi-axes are evenly spaced on screen
        semis = np.array([0.55, 1.05, 1.55, 2.1, 2.65, 3.2])
        ax.contour(X, Y, Z, levels=list(0.5 * h[1] * semis**2),
                   colors=[LIGHT], linewidths=0.9)
        # principal axes of the ellipse with y-semi-axis 2.1, labelled with
        # the curvatures sigma^2 + lambda along each axis
        c = 0.5 * h[1] * 2.1**2
        ex, ey = np.sqrt(2 * c / h[0]), 2.1
        fl.arrow(ax, (0, 0), (ex, 0), color=GREEN, lw=1.8)
        fl.arrow(ax, (0, 0), (0, ey), color=GREEN, lw=1.8)
        ax.text(ex * 0.58, -0.30, axis_labels[0], color=GREEN, fontsize=9,
                ha="center", va="top")
        ax.text(0.10, ey * 0.78, axis_labels[1], color=GREEN, fontsize=9,
                ha="left", va="center")
        # real GD at the optimal step for this Hessian
        eta = 2.0 / (h[0] + h[1])
        x = x0.copy()
        path = [x.copy()]
        for _ in range(steps):
            x = x - eta * (h * x)
            path.append(x.copy())
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], "-o", color=ORANGE, ms=3.2, lw=1.5,
                zorder=4)
        ax.plot(0, 0, "*", color=GREEN, ms=12, zorder=5)
        ax.set_title(title)
        fl.clean_axes(ax, lim=(xlim, ylim), hide=True)

    kappa0 = s2[0] / s2[1]
    kappa1 = (s2[0] + 0.5) / (s2[1] + 0.5)
    panel(axa, 0.0, rf"(a) $\lambda=0$:  $\kappa={kappa0:.0f}$", 28,
          (r"$\sigma_1^2$", r"$\sigma_n^2$"))
    panel(axb, 0.5, rf"(b) $\lambda=0.5$:  $\kappa\approx{kappa1:.1f}$", 12,
          (r"$\sigma_1^2+\lambda$", r"$\sigma_n^2+\lambda$"))

    fl.save(fig, "mdl-opt-conditioning-ellipse")


# =========================================================================== #
# Driver                                                                      #
# =========================================================================== #

FIGURES = [
    # 3.1 gradient-based optimization
    fig_gd_bowl_vs_valley,
    fig_momentum_damping,
    fig_sgd_noise_ball,
    # 3.2 convexity
    fig_chord_above_graph,
    fig_convex_vs_nonconvex_set,
    fig_local_equals_global,
    # 3.3 constrained optimization and duality
    fig_lagrange_tangency,
    fig_kkt_active_set,
    fig_primal_dual_gap,
    # 3.4 numerical stability and conditioning
    fig_fp_number_line,
    fig_conditioning_ellipse,
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
