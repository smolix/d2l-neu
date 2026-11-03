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

# One label size for the whole chapter (house style: ~13-16 pt in-figure
# text).  The figures were drawn at 9-12 pt, which shrank to ~8 px when the
# 9-11 in canvases were scaled to the text column.
LBL = 13


# =========================================================================== #
# 3.1 Gradient-Based Optimization                                             #
# =========================================================================== #

# Level sets of the kappa = 20 valley 1/2 (x^2 + 20 y^2), shared by the two
# figures that run gradient descent on it.  Evenly spaced semi-axes: five
# closed ellipses (x-semi-axes 0.3 ... 1.5) plus the strips of the steep walls
# at |y| = 0.5 ... 1.5.  The auto-spaced 14 levels drew only wall strips at
# panel scale -- a stack of near-parallel lines with no visible minimum.
VALLEY_LEVELS = ([0.5 * a**2 for a in (0.3, 0.6, 0.9, 1.2, 1.5)]
                 + [10.0 * b**2 for b in (0.55, 0.8, 1.05, 1.3, 1.55)])


def fig_gd_bowl_vs_valley():
    """The headline contrast of the whole chapter (sec_mdl-gradient-based-
    optimization, 3.1.2-3.1.3): real gradient descent on f(x)=1/2 x^T A x.

    (a) Well-conditioned bowl, lambda = (1, 1.3), eta = 0.55 (factors 0.45
        and 0.285): the path runs nearly straight to the minimum.
    (b) Ill-conditioned valley, lambda = (1, 20), eta = 0.097 ~ 2/L (factors
        0.903 and -0.94): the iterates zig-zag across the steep axis while
        crawling along the slow lambda_min axis.  fig_per_coordinate replays
        this exact run, so its geometry must not change.
    The panel titles carry kappa and eta (functional, absent from the caption).
    """
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.0, 4.2))

    g = np.linspace(-1.7, 1.7, 220)
    X, Y = np.meshgrid(g, g)

    def run(ax, A, title, eta, x0, steps, levels=14):
        Z = 0.5 * (A[0, 0] * X**2 + (A[0, 1] + A[1, 0]) * X * Y + A[1, 1] * Y**2)
        ax.contour(X, Y, Z, levels=levels, colors=[LIGHT], linewidths=0.9)
        x = np.array(x0, float)
        path = [x.copy()]
        for _ in range(steps):
            x = x - eta * (A @ x)  # grad of 1/2 x^T A x  (A symmetric)
            path.append(x.copy())
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], "-o", color=ORANGE, ms=3.5, lw=1.6,
                zorder=4)
        ax.plot(0, 0, "*", color=GREEN, ms=13, zorder=5)
        ax.set_title(title, color="black", fontsize=LBL)
        clean_lim = ((-1.7, 1.7), (-1.7, 1.7))
        fl.clean_axes(ax, lim=clean_lim, hide=True)
        return path

    # (a) well-conditioned: lambda = (1, 1.3), kappa ~ 1; safe step -> ~straight.
    run(axa, np.array([[1.0, 0.0], [0.0, 1.3]]),
        r"(a) $\kappa=1.3$, $\eta=0.55$",
        eta=0.55, x0=(-1.35, 1.15), steps=10)

    # (b) ill-conditioned: lambda = (1, 20), kappa = 20; eta near 2/lambda_max
    #     -> the steep mode oscillates, the flat mode barely moves.
    A = np.array([[1.0, 0.0], [0.0, 20.0]])
    run(axb, A, r"(b) $\kappa=20$, $\eta=0.097\approx2/L$",
        eta=0.097, x0=(-1.4, 1.0), steps=26, levels=VALLEY_LEVELS)
    # annotate the slow (lambda_min) axis along which progress crawls.
    # both heads sit outside the zig-zag (its leftmost iterate is x = -1.4);
    # the shaft runs under the path (zorder < 4), so the iterates stay on top
    axb.annotate("", xy=(1.63, 0.0), xytext=(-1.63, 0.0),
                 arrowprops=dict(arrowstyle="<->", color="black", lw=1.1,
                                 linestyle=(0, (4, 3)), shrinkA=0, shrinkB=0),
                 zorder=2)
    axb.text(1.0, 0.07, r"slow axis ($\lambda_{\min}$)", color="black",
             fontsize=LBL, ha="center", va="bottom",
             bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))

    save = fl.save
    save(fig, "mdl-opt-gd-bowl-vs-valley")


def fig_eta_tent():
    """The per-mode contraction factors |1 - eta*lambda| as functions of
    lambda (sec_mdl-gradient-based-optimization, The Quadratic Model): the
    "tent" whose vertex sits at lambda = 1/eta.  Two step sizes on the
    spectrum {1, 10} of the section's running example:

    - eta = 0.1: vertex at the stiff mode (its factor is exactly 0) but the
      slow mode contracts only by 0.9 -- the greedy choice;
    - eta* = 2/11: the two endpoint factors are *equal* at
      (kappa-1)/(kappa+1) = 9/11, which is the optimal-step proof drawn.

    The convergence factor rho(eta) is the higher of the two endpoint dots;
    dashed line at height 1 marks the stability boundary.
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    lmin, lmax = 1.0, 10.0
    lam = np.linspace(0.0, 11.6, 400)
    etas = [(0.1, BLUE, r"$\eta=0.1$"),
            (2.0 / (lmin + lmax), GREEN, r"$\eta^\star=2/(\lambda_{\min}+\lambda_{\max})$")]

    ax.plot([0.0, 11.6], [1.0, 1.0], color=GRAY, lw=1.2, ls=(0, (5, 3)))
    # right-anchored just short of the lambda_max gridline (at 10) -- anchoring
    # at the old x=10.75 straddled that gridline right through the "eta lambda".
    ax.text(9.85, 1.07, "instability: $|1-\\eta\\lambda|=1$", color="black",
            fontsize=LBL, ha="right", va="bottom")

    for eta, col, lab in etas:
        ax.plot(lam, np.abs(1 - eta * lam), color=col, lw=2.0, zorder=3)
        for l0 in (lmin, lmax):                # the two extreme modes
            ax.plot(l0, abs(1 - eta * l0), "o", color=col, ms=6.5, zorder=5)

    # eigenvalue gridlines
    for l0, lab in ((lmin, r"$\lambda_{\min}$"), (lmax, r"$\lambda_{\max}$")):
        # stop at the instability line, clear of the labels above it
        ax.plot([l0, l0], [0, 1.0], color=GRAY, lw=0.9, ls=(0, (2, 3)),
                zorder=1)
        ax.text(l0, -0.06, lab, color="black", fontsize=LBL, ha="center",
                va="top")

    # the balanced level of the optimal step
    rho = (lmax - lmin) / (lmax + lmin)
    ax.plot([lmin, lmax], [rho, rho], color=GREEN, lw=1.0, ls=(0, (2, 3)),
            zorder=2)
    # Direct labels at the right ends of the two tents name the step sizes
    # (the figure never said which tent was which); the eta* formula itself
    # lives in the caption.
    ax.text(11.75, abs(1 - 0.1 * 11.6), r"$\eta=0.1$", color=BLUE,
            fontsize=LBL, ha="left", va="center")
    ax.text(11.75, abs(1 - 2.0 / 11.0 * 11.6), r"$\eta^{\star}$", color=GREEN,
            fontsize=LBL, ha="left", va="center")
    # The balanced level, labelled just under its dotted line in the band
    # between the blue line (left) and the green right arm.
    ax.text(6.6, rho - 0.035, r"both endpoints at $\rho=9/11$",
            color=GREEN, fontsize=LBL, ha="center", va="top")

    # the greedy eta = 0.1: stiff mode dead, slow mode rules.  Placed in the
    # clear pocket just right of lambda_max (the blue curve stays low there),
    # so the label clears both the lambda_max gridline and the curve.
    ax.annotate("stiff mode: factor 0", xy=(10.05, 0.04), xytext=(10.2, 0.40),
                color=BLUE, fontsize=LBL, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.9))
    # right of the lambda_min dot, between the blue line and the dashed
    # instability line above it
    ax.text(1.55, 0.92, r"slow mode barely moves: $0.9$", color=BLUE,
            fontsize=LBL, ha="left", va="center")
    ax.text(11.5, -0.06, r"$\lambda$", color="black", fontsize=LBL,
            ha="right", va="top")
    ax.text(0.18, 1.27, r"$|1-\eta\lambda|$", color="black", fontsize=LBL,
            ha="left", va="top")

    # black coordinate axes (lambda to the right, the factor upward)
    fl.axis_cross(ax, (0.0, 11.6), (0.0, 1.27), color="black")
    fl.clean_axes(ax, lim=((-0.35, 12.9), (-0.22, 1.30)), hide=True,
                  equal=False)
    fl.save(fig, "mdl-opt-eta-tent")


def fig_momentum_damping():
    """Heavy-ball momentum as a damped oscillator (sec_mdl-gradient-based-
    optimization, Momentum and Acceleration): three real heavy-ball runs on the
    ill-conditioned valley f(x, y) = 1/2 (x^2 + 10 y^2), all with the *same*
    step size, differing only in the damping knob beta:

    - beta = 0.05: too little momentum, rate 0.81 -- zig-zags like gradient
      descent (the stiff mode's factor 1 + beta - eta*lambda_max is a negative
      real root);
    - beta = (1 - sqrt(2/11))^2 ~ 0.33: critically damped *for this step*,
      rate 0.57 (the slow mode's roots are a double root);
    - beta = 0.85: too much momentum, rate 0.92 -- loops around the minimum.

    All three use the gradient-descent step eta = 2/11.  The sqrt(kappa)
    tuning beta* = 0.27 belongs with the larger step eta* = 0.231, at which
    beta = 0.05 diverges (eta* lambda_max = 2.31 > 2 + 2 beta).
    """
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.3), sharey=True)

    lam = np.array([1.0, 10.0])               # Hessian eigenvalues; kappa = 10
    gx = np.linspace(-2.45, 2.45, 320)
    gy = np.linspace(-1.35, 1.35, 200)
    X, Y = np.meshgrid(gx, gy)
    Z = 0.5 * (lam[0] * X**2 + lam[1] * Y**2)

    eta = 2.0 / (lam[0] + lam[1])             # GD-optimal step, ~0.182
    beta_crit = (1 - np.sqrt(eta * lam[0])) ** 2   # ~0.329, critical here

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
        (0.05, 34, BLUE, "too little momentum\n"
         r"$\beta=0.05$, rate $0.81$"),
        (beta_crit, 18, GREEN, r"critically damped for $\eta=2/11$" "\n"
         r"$\beta\approx0.33$, rate $0.57$"),
        (0.85, 60, ORANGE, "too much momentum\n"
         r"$\beta=0.85$, rate $0.92$"),
    ]
    # one regime per panel: the three paths overlap hopelessly on shared axes
    for ax, (beta, steps, col, lab) in zip(axes, runs):
        ax.contour(X, Y, Z, levels=[0.04, 0.16, 0.42, 0.9, 1.7, 2.9, 4.6, 7.0],
                   colors=[LIGHT], linewidths=0.9)
        p = heavy_ball(beta, steps)
        ax.plot(p[:, 0], p[:, 1], "-o", color=col, ms=2.6, lw=1.1, zorder=4)
        ax.plot(*p[0], "o", color=col, ms=5.5, zorder=5)
        ax.plot(0, 0, "*", color="black", ms=12, zorder=6)
        ax.set_title(lab, color=col, fontsize=LBL)
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
    # the two leader labels sit over the outer level-set rings
    _LABEL_BOX = dict(boxstyle="round,pad=0.12", fc="white", ec="none")

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

    # on top of the SGD cloud and the GD path end, with a white rim so the
    # minimizer stays visible inside the scatter
    ax.plot(0, 0, "*", color=GREEN, ms=17, mec="white", mew=1.2, zorder=8)

    # noise-ball circles: r = sqrt(eta sigma^2 / (2 lambda)), and r/sqrt(2)
    # for step eta/2; both at the shared 3x visual scale.
    scale = 3.0
    r = scale * np.sqrt(eta * sigma**2 / (2 * lam))
    for radius, ls in ((r, (0, (5, 3))), (r / np.sqrt(2), (0, (3, 3)))):
        ax.add_patch(plt.Circle((0, 0), radius, fill=False, edgecolor=GRAY,
                                lw=1.3, linestyle=ls, zorder=6))
    ax.annotate(r"noise ball $\propto\sqrt{\eta}$",
                xy=(r * np.cos(0.6), -r * np.sin(0.6)), xytext=(1.05, -1.25),
                color="black", fontsize=LBL, ha="left", va="center",
                bbox=_LABEL_BOX,
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.9))
    ax.annotate(r"ball at step $\eta/2$",
                xy=(-(r / np.sqrt(2)) * np.cos(0.8),
                    -(r / np.sqrt(2)) * np.sin(0.8)), xytext=(-1.68, -1.25),
                color="black", fontsize=LBL, ha="center", va="center",
                bbox=_LABEL_BOX,
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.9))

    # heading set just above the GD start point so the descending line stays
    # below it and never strikes through the text.
    ax.text(-2.02, 1.70, "gradient descent", color=BLUE, fontsize=LBL,
            ha="left", va="bottom")
    ax.text(-0.45, 0.95, r"SGD, fixed $\eta$", color=ORANGE, fontsize=LBL,
            ha="left", va="bottom")

    fl.clean_axes(ax, lim=((-2.45, 2.3), (-1.7, 1.95)), hide=True)
    fl.save(fig, "mdl-opt-sgd-noise-ball")
    # the slides read @fig figures from img/auto
    import shutil
    shutil.copyfile(os.path.join(fl.IMG_DIR, "mdl-opt-sgd-noise-ball.svg"),
                    os.path.join(fl.IMG_DIR, "auto", "mdl-opt-sgd-noise-ball.svg"))


# =========================================================================== #
# Stochastic and Adaptive Methods (sec_mdl-adaptive-stochastic-methods)       #
# =========================================================================== #

def fig_per_coordinate():
    """Per-coordinate scaling and its limit (sec_mdl-adaptive-stochastic-methods,
    The Running Example, Aligned and Rotated): the valley
    f(x, y) = 1/2 (x^2 + 20 y^2) from the same start, under

    (a) gradient descent with one global step size near the stability ceiling
        (eta = 0.097, 26 steps -- the exact run of fig_gd_bowl_vs_valley's
        right panel): the stiff axis oscillates, the slow axis crawls;
    (b) Adam's per-coordinate normalization (beta_1 = 0 to isolate the
        rescaling from momentum, bias-corrected v, alpha = 0.12, 40 steps):
        both coordinates move at equal speed and the path curves into the
        minimum (f = 4e-7 after 40 steps);
    (c) the same update, step and step count on the valley rotated by 45
        degrees, with the start rotated too: the coordinate scales no longer
        match the curvature directions, and after dropping to the valley floor
        the path creeps along it (f = 0.19 after 40 steps).

    All three trajectories are honest runs of the stated updates.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    axa, axb, axc = axes

    lam = np.array([1.0, 20.0])
    th = np.pi / 4
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    H_rot = R @ np.diag(lam) @ R.T
    lim = 1.8
    g = np.linspace(-lim, lim, 240)
    X, Y = np.meshgrid(g, g)
    x0 = np.array([-1.4, 1.0])

    Z = 0.5 * (lam[0] * X**2 + lam[1] * Y**2)
    Z_rot = 0.5 * (H_rot[0, 0] * X**2 + 2 * H_rot[0, 1] * X * Y
                   + H_rot[1, 1] * Y**2)
    for ax, ZZ in ((axa, Z), (axb, Z), (axc, Z_rot)):
        ax.contour(X, Y, ZZ, levels=VALLEY_LEVELS, colors=[LIGHT],
                   linewidths=0.9)
        ax.plot(0, 0, "*", color=GREEN, ms=13, zorder=5)

    # (a) gradient descent, one global eta near 2/lambda_max.
    x, path = x0.copy(), [x0.copy()]
    for _ in range(26):
        x = x - 0.097 * lam * x
        path.append(x.copy())
    path = np.array(path)
    axa.plot(path[:, 0], path[:, 1], "-o", color=ORANGE, ms=3.5, lw=1.6,
             zorder=4)

    # (b, c) Adam's per-coordinate rescaling (beta_1 = 0, bias-corrected v).
    def scaled_path(H, start):
        x, v, path = start.copy(), np.zeros(2), [start.copy()]
        for k in range(1, 41):
            gr = H @ x
            v = 0.999 * v + 0.001 * gr * gr
            x = x - 0.12 * gr / (np.sqrt(v / (1 - 0.999**k)) + 1e-8)
            path.append(x.copy())
        return np.array(path)

    path = scaled_path(np.diag(lam), x0)
    axb.plot(path[:, 0], path[:, 1], "-o", color=BLUE, ms=3.5, lw=1.6,
             zorder=4)
    axb.annotate("equal speed on\nboth coordinates",
                 xy=(-0.85, 0.47), xytext=(-0.05, 1.2), color="black",
                 fontsize=LBL, ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"),
                 arrowprops=dict(arrowstyle="-", color="black", lw=0.9))

    path = scaled_path(H_rot, R @ x0)
    axc.plot(path[:, 0], path[:, 1], "-o", color=BLUE, ms=3.5, lw=1.6,
             zorder=4)
    # label in the empty upper-left corner, pointing at the slow stretch
    axc.annotate("creeps along\nthe floor",
                 xy=tuple(path[30]), xytext=(-0.95, 0.85), color="black",
                 fontsize=LBL, ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"),
                 arrowprops=dict(arrowstyle="-", color="black", lw=0.9))

    for ax in axes:
        fl.clean_axes(ax, lim=((-lim, lim), (-lim, lim)), hide=True)
    fig.subplots_adjust(wspace=0.08)
    fl.save(fig, "mdl-opt-per-coordinate")


def fig_schedule_zoo():
    """The schedule zoo (sec_mdl-stochastic-gradient-methods, Schedule Shapes
    under the Exact Recursion): eta_k over a fixed budget K for the four shapes
    the section compares -- constant, c/k decay, cosine, and
    warmup-stable-decay (WSD) -- all beginning with the same linear warmup over
    the first 5% of the budget.  The constant step keeps its noise floor
    proportional to eta; c/k converges for every c > 0 but reaches the O(1/k)
    rate only for c > 1/(2 mu); WSD lowers its floor only in the final decay.
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    K, eta0 = 1000.0, 1.0
    w = 0.05 * K                               # shared linear warmup
    t = np.linspace(0.0, K, 1001)

    def with_warmup(body):
        return np.where(t < w, eta0 * t / w, body)

    constant = with_warmup(np.full_like(t, eta0))
    inv_k = with_warmup(eta0 / (1 + np.maximum(t - w, 0) / 60.0))
    cosine = with_warmup(0.5 * eta0 * (1 + np.cos(
        np.pi * np.maximum(t - w, 0) / (K - w))))
    wsd = with_warmup(np.where(t < 0.8 * K, eta0,
                               eta0 * (K - t) / (0.2 * K)))

    for y, col, lab, ls, z in [
            (constant, GRAY, "constant", (0, (5, 3)), 5),
            (inv_k, GREEN, r"$c/k$", "-", 3),
            (cosine, BLUE, "cosine", "-", 3),
            (wsd, ORANGE, "WSD", "-", 3)]:
        # the dashed constant goes on top: under the WSD plateau it was hidden
        ax.plot(t, y, color=col, lw=2.0, ls=ls, zorder=z)

    ax.text(430, 1.045, "constant: noise floor $\\propto\\eta$ forever",
            color="black", fontsize=LBL, ha="center", va="bottom")
    # shifted left from x=345 -- at that anchor the "m" of "optimum," sat
    # right on the descending blue cosine curve.  Plain coloured labels next
    # to their curves: the leaders were ~10 px stubs that read as stray ticks.
    ax.text(300, 0.47, "$c/k$: converges\nfor every $c$; $O(1/k)$\nonly if $c$ is large",
            color=GREEN, fontsize=LBL, ha="center", va="center")
    # moved right/low, where the cosine curve has already dropped well below
    # the label -- the earlier placement had the curve cut through "of".
    ax.text(760, 0.40, "cosine: long tail\nof small steps", color=BLUE,
            fontsize=LBL, ha="center", va="center")
    # anchored over the flat plateau (well above the decay ramp) so the
    # descending WSD line never cuts through the two-line label.
    ax.annotate("WSD: hold the plateau,\nthen decay to shrink\nthe noise ball",
                xy=(880, wsd[880]), xytext=(630, 0.80), color=ORANGE,
                fontsize=LBL, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.9))
    # moved clear of the c/k annotation block to its right.
    ax.annotate("warmup", xy=(0.55 * w, 0.55), xytext=(120, 0.15),
                color="black",
                fontsize=LBL, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color="black", lw=0.8))

    ax.plot([0, K], [0, 0], color="black", lw=1.0)
    ax.text(K, -0.07, r"$k$", fontsize=LBL, ha="right", va="top")
    ax.text(8, 1.10, r"$\eta_k$", fontsize=LBL, ha="left", va="bottom")
    fl.clean_axes(ax, lim=((-25.0, 1030.0), (-0.16, 1.24)), hide=True,
                  equal=False)
    fl.save(fig, "mdl-opt-schedule-zoo")
    # the slides read @fig figures from img/auto
    import shutil
    shutil.copyfile(os.path.join(fl.IMG_DIR, "mdl-opt-schedule-zoo.svg"),
                    os.path.join(fl.IMG_DIR, "auto", "mdl-opt-schedule-zoo.svg"))


def fig_bias_correction():
    """Adam's startup transient and its exact cancellation
    (sec_mdl-adaptive-stochastic-methods, Adam): analytic curves, no sampling.

    (a) Under a stationary gradient scale, E[v_t] = (1 - beta_2^t) E[g^2]: the
        zero-initialized average carries only the plotted fraction of the true
        scale (63% at t = 1000 for beta_2 = 0.999); dividing by 1 - beta_2^t
        restores 1 exactly at every t.
    (b) The mis-scaling of the *raw* ratio m_t / sqrt(v_t): the factor
        (1 - beta_1^t) / sqrt(1 - beta_2^t) is 3.16 at t = 1, peaks near 6.6
        around t ~ 12 (the numerator saturates in ~10 steps, the denominator
        needs ~1000), and only decays to 1 over ~1/(1-beta_2) steps --
        uncorrected Adam takes its largest steps early, on its worst
        information.  Log-t axis so both the peak and the slow decay show.
    """
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.0, 3.6))

    t = np.arange(1, 5001)
    b1, b2 = 0.9, 0.999

    axa.axhline(1.0, color=GREEN, lw=2.0, zorder=3)
    axa.plot(t, 1 - b2**t, color=ORANGE, lw=2.0, zorder=3)
    axa.plot([1000, 1000], [0, 1 - b2**1000], color=GRAY, lw=0.9,
             ls=(0, (2, 3)))
    axa.annotate(r"$\mathbb{E}[v_t]/\overline{g^2} = 1-\beta_2^t$"
                 "\n(63% at $t=1000$)",
                 xy=(1000, 1 - b2**1000), xytext=(2500, 0.42), color=ORANGE,
                 fontsize=LBL, ha="center", va="center",
                 arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.9))
    axa.text(2500, 1.05, r"corrected: $\hat{v}_t = v_t/(1-\beta_2^t)$",
             color=GREEN, fontsize=LBL, ha="center", va="bottom")
    axa.set_xlabel(r"$t$")
    axa.set_ylabel("fraction of the true scale", fontsize=LBL)
    fl.clean_axes(axa, lim=((-100.0, 5100.0), (-0.06, 1.3)), hide=False,
                  equal=False)

    ratio = (1 - b1**t) / np.sqrt(1 - b2**t)
    tpk = int(t[np.argmax(ratio)])
    axb.axhline(1.0, color=GREEN, lw=1.2, ls=(0, (5, 3)), zorder=2)
    axb.plot(t, ratio, color=ORANGE, lw=2.0, zorder=3)
    axb.set_xscale("log")
    axb.plot(1, ratio[0], "o", color=ORANGE, ms=6, zorder=4)
    axb.plot(tpk, ratio.max(), "o", color=ORANGE, ms=6, zorder=4)
    axb.annotate(r"$3.16{\times}$ at $t=1$",
                 xy=(1, ratio[0]), xytext=(2.3, 1.75), color=ORANGE,
                 fontsize=LBL, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.9))
    axb.annotate(rf"peak ${ratio.max():.1f}{{\times}}$ at $t={tpk}$",
                 xy=(tpk, ratio.max()), xytext=(90, 6.4), color=ORANGE,
                 fontsize=LBL, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.9))
    # under the dashed line, where the orange tail cannot reach it
    axb.text(1.15, 0.85, "properly scaled: 1", color=GREEN, fontsize=LBL,
             ha="left", va="top")
    # The plotted factor names the y-axis.  It used to be a 19 pt "(b) ..."
    # title over the right panel only, which left the left panel short and
    # repeated the caption's "Right: ..." text.
    axb.set_ylabel(r"$(1-\beta_1^t)\,/\sqrt{1-\beta_2^t}$", fontsize=LBL + 1)
    axb.set_xlabel(r"$t$")
    # left edge below t = 1 so the t = 1 marker clears the y-axis line
    axb.set_xlim(0.7, 5500.0)
    axb.set_ylim(0.0, 7.2)
    axb.set_aspect("auto")

    fig.subplots_adjust(wspace=0.30)
    fl.save(fig, "mdl-opt-bias-correction")


# =========================================================================== #
# 3.2 Convex Sets and Convex Functions                                        #
# =========================================================================== #

def fig_chord_above_graph():
    """Two equivalent conditions for convexity of a real convex function
    f(x)=1/4 x^2 (sec_mdl-convexity, The First-Order Condition):

    (a) Chord condition: a chord between two points on the graph lies *above* the
        graph -- f(theta x + (1-theta) y) <= theta f(x) + (1-theta) f(y).
    (b) First-order condition: the tangent at a point lies *below* the graph -- the
        tangent is a global under-estimator, f(y) >= f(x) + f'(x)(y-x).
    """
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.0, 4.0))

    f = lambda t: 0.25 * t**2
    fp = lambda t: 0.5 * t            # derivative
    xs = np.linspace(-3.2, 3.2, 400)

    # ----- (a) chord above the graph -----
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
    fl.vlabel(axa, (a - 0.05, f(a) + 0.55), r"$\mathbf{x}$", color=BLUE,
              ha="right", fontsize=LBL + 1)
    fl.vlabel(axa, (b - 0.05, f(b) + 0.45), r"$\mathbf{y}$", color=BLUE,
              ha="right", fontsize=LBL + 1)
    # The two sides of the inequality read top to bottom down one vertical:
    # chord value (label) -> orange dot -> dashed gap -> blue dot -> graph
    # value (label).  A short leader joins the top label to its dot.
    axa.annotate(r"$\theta f(\mathbf{x})+(1-\theta)f(\mathbf{y})$",
                 xy=(xc, chord_y + 0.08), xytext=(xc, 2.62), color=ORANGE,
                 fontsize=LBL, ha="center", va="center",
                 arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.9,
                                 shrinkA=4, shrinkB=0))
    # the graph value sits beside its dot: below-left, under the x-axis and
    # clear of the y-axis (a white box keeps it off any line)
    axa.text(xc - 0.12, -0.28, r"$f(\theta\mathbf{x}+(1-\theta)\mathbf{y})$",
             color=BLUE, fontsize=LBL, ha="right", va="top",
             bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    # the vertical axis stops below the top label it would otherwise cut
    fl.axis_cross(axa, (-3.6, 3.6), (-0.6, 2.25), color="black")
    # both panels share one window so their axes sit at the same height
    fl.clean_axes(axa, lim=((-3.6, 3.6), (-1.6, 3.0)), hide=True)

    # ----- (b) tangent below the graph -----
    axb.plot(xs, f(xs), color=BLUE, lw=2.2, zorder=3)
    x0 = -1.0
    # tangent clipped to the window (it used to run off the bottom edge,
    # cutting the dashed gap segment short of the line it measures to)
    xt = np.linspace(-3.2, 2.6, 200)
    axb.plot(xt, f(x0) + fp(x0) * (xt - x0), color=GREEN, lw=2.0, zorder=2)
    axb.plot(x0, f(x0), "o", color=BLUE, ms=6, zorder=5)
    # mark the gap f(y) - [tangent at y] >= 0 at some y
    y0 = 1.9
    axb.plot([y0, y0], [f(x0) + fp(x0) * (y0 - x0), f(y0)], "--",
             color=GRAY, lw=1.4, zorder=2)
    axb.plot(y0, f(y0), "o", color=BLUE, ms=5, zorder=5)
    fl.vlabel(axb, (x0 - 0.30, f(x0) + 0.58), r"$\mathbf{x}$", color=BLUE,
              ha="right", fontsize=LBL + 1)
    # below-right of its dot, clear of the steep right arm
    fl.vlabel(axb, (y0 + 0.22, f(y0) - 0.32), r"$\mathbf{y}$", color=BLUE,
              ha="left", fontsize=LBL + 1)
    # the tangent (global under-estimator) labelled in the open band below
    # the x-axis, left of where the line itself passes through that band
    axb.text(0.15, -1.12,
             r"$f(\mathbf{x})+\nabla f(\mathbf{x})^\top(\mathbf{y}-\mathbf{x})$",
             color=GREEN, fontsize=LBL, ha="right", va="center")
    fl.axis_cross(axb, (-3.6, 3.6), (-0.6, 2.25), color="black")
    fl.clean_axes(axb, lim=((-3.6, 3.6), (-1.6, 3.0)), hide=True)

    fl.save(fig, "mdl-opt-chord-above-graph")


def fig_convex_vs_nonconvex_set():
    """Convex vs. non-convex sets (sec_mdl-convexity, 3.2.1):

    (a) a convex blob -- the segment between any two of its points stays inside;
    (b) a non-convex crescent (unit disc minus an offset disc) -- the segment
        between two points of its horns leaves the set in the middle;
    (c) the probability simplex in R^2 -- the segment p1 + p2 = 1, p >= 0 cut
        from the nonnegative orthant -- and a half-space a^T x <= b with its
        normal a, the two convex sets DL uses most.

    Fills are light tints with full-strength outlines (an alpha on the patch
    used to fade the outline away with the fill).
    """
    fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(11.0, 3.7))

    def region(ax, xy, color, alpha=0.16):
        ax.add_patch(plt.Polygon(xy, closed=True, facecolor=color,
                                 alpha=alpha, lw=0))
        ax.add_patch(plt.Polygon(xy, closed=True, fill=False,
                                 edgecolor=color, lw=2.0))

    # ----- (a) convex blob: a regular-ish convex polygon -----
    ang = np.linspace(0, 2 * np.pi, 9)[:-1]
    r = 1.0 + 0.12 * np.cos(3 * ang)        # gently lumpy but still convex
    blob = np.column_stack([r * np.cos(ang), r * np.sin(ang)])
    # convex hull keeps it provably convex
    from scipy.spatial import ConvexHull
    hull = ConvexHull(blob)
    region(axa, blob[hull.vertices], BLUE)
    p, q = np.array([-0.7, 0.45]), np.array([0.75, -0.45])
    axa.plot([p[0], q[0]], [p[1], q[1]], color=ORANGE, lw=2.0, zorder=3)
    axa.plot([p[0], q[0]], [p[1], q[1]], "o", color=ORANGE, ms=6, zorder=4)
    fl.clean_axes(axa, lim=((-1.3, 1.3), (-1.25, 1.25)), hide=True)

    # ----- (b) non-convex crescent: chord exits the set -----
    # crescent = unit disc minus the disc of radius rb at (cb, 0); the two
    # circles meet at x = xi, y = +-yi
    cb, rb = 0.55, 0.85
    xi = (1.0 + cb**2 - rb**2) / (2 * cb)
    yi = np.sqrt(1.0 - xi**2)
    a0 = np.arctan2(yi, xi)
    outer = np.linspace(a0, 2 * np.pi - a0, 160)            # through (-1, 0)
    b0 = np.arctan2(yi, xi - cb)
    inner = np.linspace(2 * np.pi - b0, b0, 160)            # through (cb-rb, 0)
    cres = np.vstack([np.column_stack([np.cos(outer), np.sin(outer)]),
                      np.column_stack([cb + rb * np.cos(inner),
                                       rb * np.sin(inner)])])
    region(axb, cres, ORANGE)
    # a chord between two points of the horns: both ends inside the
    # crescent, the middle inside the bite (outside the set)
    p2, q2 = np.array([0.05, 0.9]), np.array([0.05, -0.9])
    axb.plot([p2[0], q2[0]], [p2[1], q2[1]], color=BLUE, lw=2.0, zorder=3)
    axb.plot([p2[0], q2[0]], [p2[1], q2[1]], "o", color=BLUE, ms=6, zorder=4)
    axb.plot(0.05, 0.0, "x", color=BLUE, ms=10, mew=2.4, zorder=5)
    axb.text(0.2, 0.0, "outside", color=BLUE, fontsize=LBL, ha="left",
             va="center")
    fl.clean_axes(axb, lim=((-1.15, 1.15), (-1.25, 1.25)), hide=True)

    # ----- (c) simplex + half-space -----
    # half-space a^T x <= b, shaded, with its outward normal a
    aa = np.array([1.0, 0.7])
    aa_n = aa / np.linalg.norm(aa)
    bb = -0.55
    L = 2.4
    perp = np.array([-aa_n[1], aa_n[0]])
    c0 = bb * aa_n
    seg = np.array([c0 - L * perp, c0 + L * perp])
    far = -aa_n * (2 * L)  # into the half-space a^T x <= b
    poly = np.array([seg[0], seg[1], seg[1] + far, seg[0] + far])
    axc.add_patch(plt.Polygon(poly, closed=True, facecolor=BLUE, alpha=0.12,
                              lw=0))
    axc.plot(seg[:, 0], seg[:, 1], color=BLUE, lw=2.0)
    # normal drawn high on the boundary, in the empty upper-left quadrant
    nb = c0 + 1.3 * perp
    fl.arrow(axc, nb, nb + 0.55 * aa_n, color=BLUE, lw=1.8)
    axc.text(*(nb + 0.55 * aa_n + np.array([0.08, 0.0])), r"$\mathbf{a}$",
             color=BLUE, fontsize=LBL + 1, ha="left", va="center")
    axc.text(-0.95, -1.0, r"$\mathbf{a}^\top\mathbf{x}\leq b$", color=BLUE,
             fontsize=LBL, ha="center", va="center")
    fl.axis_cross(axc, (-1.4, 1.85), (-1.4, 1.85), color="black")
    # the probability simplex {p >= 0, 1^T p = 1}: in R^2 the segment from
    # e1 to e2 (it used to be drawn as the filled triangle 1^T p <= 1)
    axc.plot([1.0, 0.0], [0.0, 1.0], color=GREEN, lw=3.2, zorder=4,
             solid_capstyle="round")
    axc.plot([1.0, 0.0], [0.0, 1.0], "o", color=GREEN, ms=6, zorder=5)
    axc.text(0.62, 0.62, r"$\{\mathbf{p}\succeq0,\ \mathbf{1}^\top\mathbf{p}=1\}$",
             color=GREEN, fontsize=LBL, ha="left", va="bottom")
    fl.clean_axes(axc, lim=((-1.4, 2.2), (-1.4, 1.85)), hide=True)

    fl.save(fig, "mdl-opt-convex-vs-nonconvex-set")


def fig_subgradient_fan():
    """The subgradient fan (sec_mdl-convexity, The Subgradient): at the kink
    of f(x) = |x| the gradient does not exist, but every slope g in [-1, 1]
    tucks the supporting line g*x under the graph -- the subdifferential is a
    whole interval, drawn as a fan of supporting lines through the origin.
    The zero-slope member is highlighted: 0 in partial f(0) is the optimality
    certificate that makes the corner a provable minimum without a gradient.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    xs = np.linspace(-2.1, 2.1, 300)
    ax.plot(xs, np.abs(xs), color=BLUE, lw=2.4, zorder=4)

    span = np.array([-1.75, 1.75])
    for g in (-1.0, -2.0 / 3, -1.0 / 3, 1.0 / 3, 2.0 / 3, 1.0):
        ax.plot(span, g * span, color=GREEN, lw=1.2, alpha=0.55, zorder=2)
    # the zero-slope supporting line: the optimality certificate.  It lies on
    # f = 0, so no horizontal axis is drawn -- the axis used to cover it.
    ax.plot(span, 0.0 * span, color=ORANGE, lw=2.6, zorder=3)

    ax.plot(0, 0, "o", color="black", ms=6.5, zorder=6)
    # Both notes sit right of the fan's footprint (x <= 1.75), each beside
    # the lines it describes, at one size.
    ax.text(2.1, 0.0, r"$0\in\partial f(0)$: the corner" "\nis a minimum",
            color=ORANGE, fontsize=LBL, ha="left", va="center")
    ax.text(2.1, -1.15, r"every slope $g\in[-1,1]$" "\nsupports the graph:"
            "\n" r"$\partial f(0)=[-1,1]$",
            color=GREEN, fontsize=LBL, ha="left", va="center")
    ax.text(-0.95, 2.05, r"$f(x)=|x|$", color=BLUE, fontsize=LBL + 1,
            ha="center", va="center")

    ax.annotate("", xy=(0, 2.2), xytext=(0, -1.9),
                arrowprops=dict(arrowstyle="->", color="black", lw=0.9))
    # the matching horizontal axis, drawn *behind* the orange zero-slope line
    # and poking out past it on both sides, so the y-axis is not left alone
    ax.annotate("", xy=(1.96, 0.0), xytext=(-2.2, 0.0),
                arrowprops=dict(arrowstyle="->", color="black", lw=0.9,
                                shrinkA=0, shrinkB=0), zorder=1)
    fl.clean_axes(ax, lim=((-2.25, 4.05), (-1.95, 2.25)), hide=True,
                  equal=False)
    fl.save(fig, "mdl-opt-subgradient-fan")


def fig_local_equals_global():
    """Why convexity matters (sec_mdl-convexity, 3.2.4): a convex objective has
    one global minimum and no traps; a non-convex one has several local minima
    separated by a hump, so gradient descent's limit depends on where it starts.

    (a) Convex f(x)=1/4 x^2: GD from two starts converges to the same minimum.
    (b) Non-convex double well: real GD from two starts lands in two different
        minima (the deeper one global); a local maximum -- the 1-D analogue of
        a saddle -- separates the basins.
    """
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.0, 4.0))

    # ----- (a) convex bowl: every start -> the single global min -----
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
        axa.plot(p[0], f(p[0]), "o", color=col, ms=7, zorder=4)   # start
    axa.plot(0, 0, "*", color="black", ms=14, zorder=5)
    axa.text(0, -0.30, "global min", color="black", fontsize=LBL,
             ha="center", va="top")
    # no axis cross: panel (b) has none, and the two panels share one window
    # (same xlim/ylim) so they render at the same size and scale.
    fl.clean_axes(axa, lim=((-3.4, 3.4), (-1.75, 3.0)), hide=True)

    # ----- (b) non-convex: two basins separated by a hump -----
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
    maxima = [r for r in roots if gpp(r) < 0]  # the separating hump

    # eta = 0.04 (was 0.18, which jumped into each well in one step, so the
    # "paths" were single segments)
    for x0, col in ((-2.7, ORANGE), (2.7, GREEN)):
        p = gd1d(x0, gp, eta=0.04, steps=40)
        axb.plot(p, g(p), "-o", color=col, ms=3.2, lw=1.3, zorder=3)
        axb.plot(p[0], g(p[0]), "o", color=col, ms=7, zorder=4)   # start
    for m in minima:
        axb.plot(m, g(m), "*", color="black", ms=14, zorder=5)
    for m in maxima:
        # in 1-D the critical point between the basins is a local maximum
        axb.plot(m, g(m), "o", color=GRAY, ms=7, zorder=5)
        axb.text(m, g(m) + 0.22, "local max", color="black", fontsize=LBL,
                 ha="center", va="bottom")
    # label by value: the deeper well is the global minimum (the labels used
    # to be assigned left/right and named the deeper well "local")
    lo_m = min(minima, key=g)
    for m in minima:
        axb.text(m, g(m) - 0.30, "global min" if m == lo_m else "local min",
                 color="black", fontsize=LBL, ha="center", va="top")
    fl.clean_axes(axb, lim=((-3.4, 3.4), (-1.75, 3.0)), hide=True)
    axb.axis("off")

    fl.save(fig, "mdl-opt-local-equals-global")


# =========================================================================== #
# 3.3 Constrained Optimization and Duality                                    #
# =========================================================================== #

def fig_lagrange_tangency():
    """The Lagrange-multiplier picture (sec_mdl-constrained-optimization-
    duality, Equality Constraints) on the section's running example: minimize
    f(x) = 1/2 ||x - y||^2 with y = (3, 4) on the unit circle
    h(x) = 1/2 (||x||^2 - 1) = 0.  Real geometry: the level sets of f are
    circles around y (off the frame); radius 4 is tangent to the constraint at
    x* = (0.6, 0.8), where grad f = x* - y = -4 x* = -4 grad h (nu* = 4);
    radius 6 is tangent at the antipode (-0.6, -0.8), the second Lagrange
    point (nu = -6, the maximum of f on the circle); radius 5.215 passes
    through the non-optimal point x' at 150 degrees, where the level set
    crosses the circle and sliding along it against grad f is a feasible
    descent.
    """
    fig, ax = plt.subplots(figsize=(5.6, 5.7))

    y = np.array([3.0, 4.0])
    xstar = y / np.linalg.norm(y)             # (0.6, 0.8)

    # constraint circle h(x) = 0
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(t), np.sin(t), color=BLUE, lw=2.2, zorder=3)

    # level sets of f: circles around y = (3, 4), clipped by the frame
    xp = np.array([np.cos(np.deg2rad(150.0)), np.sin(np.deg2rad(150.0))])
    for r, a in ((4.0, 0.95), (float(np.linalg.norm(xp - y)), 0.7),
                 (6.0, 0.7)):
        ax.add_patch(plt.Circle(y, r, fill=False, edgecolor=ORANGE, lw=1.3,
                                alpha=a, zorder=2))

    # ---- the optimum: tangency, grad f = -4 grad h ----
    ax.plot(*xstar, "o", color="black", ms=6.5, zorder=6)
    fl.arrow(ax, xstar, xstar + 0.60 * xstar, color=BLUE, lw=2.0)      # grad h
    fl.arrow(ax, xstar, xstar - 0.50 * xstar, color=ORANGE, lw=2.0)    # grad f
    fl.vlabel(ax, xstar + 0.60 * xstar + np.array([0.30, 0.02]),
              r"$\nabla h$", color=BLUE, fontsize=LBL)
    fl.vlabel(ax, xstar - 0.50 * xstar + np.array([0.08, -0.22]),
              r"$\nabla f$", color=ORANGE, fontsize=LBL)
    fl.vlabel(ax, xstar + np.array([0.10, -0.40]), r"$\mathbf{x}^\star$",
              color="black", fontsize=LBL + 1)

    # ---- a non-optimal feasible point: the level set crosses the circle ----
    ax.plot(*xp, "o", color="black", ms=5.5, zorder=6)
    df = (xp - y) / np.linalg.norm(xp - y)    # unit grad f at x'
    fl.arrow(ax, xp, xp + 0.55 * df, color=ORANGE, lw=2.0)
    fl.vlabel(ax, xp + 0.55 * df + np.array([-0.22, -0.02]), r"$\nabla f$",
              color=ORANGE, fontsize=LBL)
    tang = np.array([-xp[1], xp[0]])          # unit tangent at x'
    desc = -np.dot(df, tang) * tang
    desc = 0.62 * desc / np.linalg.norm(desc)
    fl.arrow(ax, xp, xp + desc, color=GREEN, lw=2.0)
    fl.vlabel(ax, np.array([-0.88, 1.42]), "feasible\ndescent",
              color=GREEN, fontsize=LBL,
              bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"))

    # ---- the second Lagrange point: the farthest point of the circle ----
    anti = -xstar
    ax.plot(*anti, "o", mfc="white", mec="black", mew=1.5, ms=6.5, zorder=6)
    ax.text(anti[0] + 0.12, anti[1] - 0.20, r"$\nu=-6$: a maximum",
            color="black", fontsize=LBL, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"))

    ax.text(1.06, -0.45, r"constraint $h(\mathbf{x})=0$", color=BLUE,
            fontsize=LBL, ha="left", va="center")
    ax.text(2.35, 2.42, "level sets of $f$,\n"
            r"centre $(3,4)$ off the frame", color=ORANGE, fontsize=LBL,
            ha="right", va="top")

    fl.clean_axes(ax, lim=((-1.7, 2.4), (-1.75, 2.5)), hide=True)
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
        # dropped from y=0.18 to y=0.0 -- in panel (a) the outermost (r=1.8)
        # level-set circle swept down through the old spot.
        # lower-left corner: at 13 pt the centred spot collided with panel
        # (a)'s KKT formula
        ax.text(-0.62, -0.30, "feasible", color=BLUE, fontsize=LBL,
                ha="left", va="center")
        # moved from just above the g1=0 line (y=1.10) to just below it
        # (inside the feasible region): every on-the-line position in panel
        # (a) is swept by the active edge's own grad-g1 arrow, which has to
        # start on this line -- placing the label on the *other* side of the
        # line avoids that arrow with no need to also fight the level-set
        # circles for horizontal room.
        ax.text(-0.62, 0.68, r"$g_1=0$", color=BLUE, fontsize=LBL, ha="left",
                va="top")
        ax.text(1.60, -0.38, r"$g_2=0$", color=BLUE, fontsize=LBL, ha="left",
                va="center")

    def levels(ax, c, radii):
        for r, a in zip(radii, (0.9, 0.65, 0.4)):
            ax.add_patch(plt.Circle(c, r, fill=False, edgecolor=ORANGE,
                                    lw=1.2, alpha=a, zorder=2))
        ax.plot(*c, "+", color=ORANGE, ms=8, mew=1.6, zorder=3)

    # ---- (a) one constraint active ----
    region(axa)
    ca = np.array([0.4, 1.9])
    xs = np.array([ca[0], e1y])               # projection of ca on the region
    levels(axa, ca, [float(np.linalg.norm(ca - xs)), 1.35, 1.8])
    axa.plot(*xs, "o", color="black", ms=6.5, zorder=6)
    # shortened from 0.62 to 0.35 -- at x=0.4 (dead center of the level-set
    # circles) any arrow past ~0.4 long runs into the r=1.35 arc, whose
    # minimum height sits at exactly y=1.9-1.35=0.55.
    # -grad f at x*: points from x* toward the centre c, i.e. along the
    # active edge's outward normal (it used to be drawn as +grad f, pointing
    # into the region, beside a label about -grad f)
    fl.arrow(axa, xs, xs + np.array([0.0, 0.45]), color=ORANGE, lw=2.2)
    axa.text(xs[0] + 0.08, xs[1] + 0.33, r"$-\nabla f$", color=ORANGE,
             fontsize=LBL, ha="left", va="center")
    # moved well below all three level-set circles: each circle's minimum
    # height is y = 1.9 - r, so anything below y=1.9-1.8=0.1 (with margin)
    # clears all of them regardless of how wide the formula runs. Also
    # wrapped onto two lines and dropped further (y=-0.20 -> -0.27) -- as
    # one long line it ran past x=1.5 into the g2=0 boundary line, and its
    # top edge grazed "feasible".
    axa.text(0.12, -0.24,
             r"$-\nabla f=\lambda_1\nabla g_1,$" "\n" r"$\lambda_1>0$",
             color=ORANGE, fontsize=LBL, ha="left", va="center")
    fl.vlabel(axa, xs + np.array([-0.22, 0.14]), r"$\mathbf{x}^\star$",
              color="black", fontsize=LBL)
    # outward normal of the active edge, shown a bit along the edge -- moved
    # from x=-0.30 to x=-0.45 (any x in (-0.379, 0.4) has the r=0.9 circle's
    # left branch crossing it somewhere in y in [1.0, 1.45]; -0.45 clears
    # all three circles outright).
    fl.arrow(axa, (-0.45, e1y), (-0.45, e1y + 0.45), color=BLUE, lw=1.8)
    # label grows right from the arrow tip instead of being centered on it,
    # so it doesn't reach back left into the r=0.9 circle.
    axa.text(-0.40, e1y + 0.62, r"$\nabla g_1$", color=BLUE, fontsize=LBL,
              ha="left", va="center")
    # pushed further right of e2x -- at fontsize 11 the label's vertical run
    # otherwise cuts across the r=1.35 level-set arc around its "lambda_2".
    axa.text(e2x + 0.6, 0.42, r"inactive: $\lambda_2=0$", color="black",
             fontsize=LBL, ha="left", va="center", rotation=90)

    # ---- (b) corner: two constraints active ----
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
    # above the arrow tip, inside the innermost level set (upper-left of the
    # tip it sat on that circle and its box broke the arc)
    fl.vlabel(axb, corner + np.array([0.0, 0.60]), r"$\nabla g_1$",
              color=BLUE, fontsize=LBL, va="bottom")
    fl.vlabel(axb, corner + np.array([0.72, -0.16]), r"$\nabla g_2$",
              color=BLUE, fontsize=LBL)
    # right of the tip, outside the shaded wedge (at (0.12, -0.15) it sat on
    # the wedge's rim) and inside the innermost level set; above the tip the
    # orange "+" (unconstrained minimizer) sat right on the label
    axb.text(*(corner + 0.62 * df + np.array([0.25, -0.12])), r"$-\nabla f$",
             color=ORANGE, fontsize=LBL, ha="left", va="center",
             bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"))
    # No in-figure "normal cone" label: the two-line text needs more room
    # than the wedge or any spot beside it offers between the three
    # gradient arrows, their labels, the orange "+" and the level-set arcs
    # (above the wedge it read as a label for nothing).  The caption names
    # the shaded wedge instead.
    fl.vlabel(axb, corner + np.array([-0.22, -0.16]), r"$\mathbf{x}^\star$",
              color="black", fontsize=LBL)

    for ax in (axa, axb):
        fl.clean_axes(ax, lim=(xlim, ylim), hide=True)
    fl.save(fig, "mdl-opt-kkt-active-set")


def fig_primal_dual_gap():
    """The supporting-line picture of duality (sec_mdl-lagrangian-duality,
    Strong Duality).  Every x maps to (u, t) = (g(x), f(x)); the shaded set
    A = {(u, t) : u >= g(x), t >= f(x) for some x} holds these pairs and
    everything above and to the right of them.  Evaluating the dual at lambda
    lowers a line of slope -lambda until it supports A from below; its height
    at u = 0 is q(lambda) <= p*.  Only the best supporting line is drawn.

    (a) the section's running example, clipping x0 = (3, 4) to the unit ball:
        f = |x - x0|^2 / 2, g = (|x|^2 - 1) / 2.  At |x| = rho the smallest f
        is (5 - rho)^2 / 2 and u = (rho^2 - 1) / 2, so the lower boundary of A
        is t(u) = (5 - sqrt(1 + 2u))^2 / 2 for u >= -1/2 (vertical tangent at
        u = -1/2, slope -4 at u = 0).  A is convex and 0 is strictly feasible,
        so the line of slope -lambda* = -4 touches at (0, p*) = (0, 8).
    (b) min -x^2 on D = [0, 1] s.t. g(x) = x - 1/2 <= 0: the image curve
        {(u, -(u + 1/2)^2) : u in [-1/2, 1/2]} bends the wrong way, A is not
        convex, and the best line (slope -1) meets u = 0 at d* = -1/2, below
        p* = -1/4.
    """
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.4, 4.5))

    def axis_labels(ax, xr, yr):
        """Black origin cross with the two value-axis names at the far ends."""
        fl.axis_cross(ax, xr, yr, color="black")
        ax.text(xr[1], -0.035 * (yr[1] - yr[0]), "constraint value  $u$",
                color="black", fontsize=LBL, ha="right", va="top")
        ax.text(0.03 * (xr[1] - xr[0]), yr[1], "objective value  $t$",
                color="black", fontsize=LBL, ha="left", va="top")

    # ============ (a) the clipping problem: strong duality ============ #
    xr_a, yr_a = (-0.75, 2.2), (-1.3, 13.6)
    u = np.linspace(-0.5, 2.1, 400)
    boundary_a = 0.5 * (5 - np.sqrt(1 + 2 * u)) ** 2
    # A is everything above the boundary (and, for u < -1/2, nothing).
    axa.fill_between(u, boundary_a, yr_a[1], color=BLUE, alpha=0.08, lw=0,
                     zorder=1)
    axa.plot(u, boundary_a, color=BLUE, lw=2.0, zorder=3)
    axa.plot([-0.5, -0.5], [12.5, yr_a[1]], color=BLUE, lw=2.0, zorder=3)
    uf = u[u <= 0]
    axa.plot(uf, 0.5 * (5 - np.sqrt(1 + 2 * uf)) ** 2, color=ORANGE, lw=3.4,
             zorder=4)
    p_star_a = 8.0
    us = np.array([-0.35, 2.0])
    axa.plot(us, 8 - 4 * us, color=GREEN, lw=2.0, zorder=3)
    axa.plot(0, p_star_a, "o", color="black", ms=7, zorder=6)
    axa.text(0.1, p_star_a + 0.75, r"$d^\star=p^\star=8$", color="black",
             fontsize=LBL, ha="left", va="bottom")
    axa.text(-0.55, 11.0, "feasible\nslice\n" r"$u\leq0$", color=ORANGE,
             fontsize=LBL, ha="right", va="center")
    axa.text(1.35, 10.5, r"$\mathcal{A}$", color=BLUE, fontsize=LBL + 4,
             ha="center", va="center")
    axa.text(1.0, 2.0, r"slope $-\lambda^\star=-4$", color=GREEN,
             fontsize=LBL, ha="right", va="center")
    axis_labels(axa, xr_a, yr_a)
    fl.clean_axes(axa, lim=(xr_a, yr_a), hide=True, equal=False)

    # =============== (b) a concave objective: a duality gap =============== #
    xr_b, yr_b = (-0.92, 0.92), (-1.35, 0.72)
    u = np.linspace(-0.5, 0.5, 300)           # image of x in [0, 1]
    boundary_b = -(u + 0.5) ** 2
    # A: up and to the right of the image curve; right of u = 1/2 its floor
    # is the curve's right end, t = -1.
    ue = np.concatenate([u, [0.5, xr_b[1]]])
    floor = np.concatenate([boundary_b, [-1.0, -1.0]])
    axb.fill_between(ue, floor, yr_b[1], color=BLUE, alpha=0.09, lw=0,
                     zorder=1)
    axb.plot([0.5, xr_b[1] - 0.02], [-1.0, -1.0], color=BLUE, lw=1.2,
             ls=(0, (4, 3)), zorder=2)
    axb.plot(u, boundary_b, color=BLUE, lw=2.0, zorder=3)
    uf = u[u <= 0]
    axb.plot(uf, -(uf + 0.5) ** 2, color=ORANGE, lw=3.4, zorder=4)
    p_star_b, d_star_b = -0.25, -0.5
    line = lambda uu: -0.5 - uu
    us = np.array([-0.72, 0.78])
    axb.plot(us, line(us), color=GREEN, lw=2.0, zorder=3)
    axb.plot(0, p_star_b, "o", color="black", ms=7, zorder=6)
    axb.plot(0, d_star_b, "o", color=GREEN, ms=7, zorder=6)
    axb.annotate("", xy=(0.0, p_star_b - 0.012), xytext=(0.0, d_star_b + 0.012),
                 arrowprops=dict(arrowstyle="<->", color=GRAY, lw=1.4))
    axb.text(-0.22, 0.5 * (p_star_b + d_star_b), "duality gap", color="black",
             fontsize=LBL, ha="right", va="center")
    axb.text(-0.045, p_star_b, r"$p^\star$", color="black", fontsize=LBL,
             ha="right", va="center")
    axb.text(-0.045, d_star_b, r"$d^\star$", color=GREEN, fontsize=LBL,
             ha="right", va="center")
    axb.text(-0.85, 0.55, r"feasible slice $u\leq0$", color=ORANGE,
             fontsize=LBL, ha="left", va="center")
    axb.text(0.42, 0.30, r"$\mathcal{A}$ (not convex)", color=BLUE,
             fontsize=LBL, ha="center", va="center")
    axb.text(0.30, -0.42, r"$(g(x),f(x))$," "\n" r"$x\in[0,1]$", color=BLUE,
             fontsize=LBL, ha="left", va="center")
    axb.text(-0.88, -0.95, r"slope $-\lambda^\star=-1$", color=GREEN,
             fontsize=LBL, ha="left", va="center")
    axis_labels(axb, xr_b, yr_b)
    fl.clean_axes(axb, lim=(xr_b, yr_b), hide=True, equal=False)

    fig.subplots_adjust(wspace=0.16)
    fl.save(fig, "mdl-opt-primal-dual-gap")
    import shutil                             # slides read img/auto/
    shutil.copyfile(os.path.join(fl.IMG_DIR, "mdl-opt-primal-dual-gap.svg"),
                    os.path.join(fl.IMG_DIR, "auto", "mdl-opt-primal-dual-gap.svg"))


def fig_water_filling():
    """The water-filling picture (sec_mdl-lagrangian-duality, Worked
    Examples): each channel is a basin whose floor sits at its noise level
    N_i; pouring in the power budget P fills the quiet channels to a common
    water level theta = 1/nu, and channels whose floor is above the level stay
    dry.

    Uses the same noise floors and budget as the #constrained-water-filling
    cell (N = [0.1, 0.4, 0.8, 1.6, 2.5], P = 3) and finds the level by the same
    bisection, so the figure's theta ~ 1.4333 matches the cell's printout.
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    N = np.array([0.1, 0.4, 0.8, 1.6, 2.5])
    P = 3.0
    lo, hi = N.min(), N.min() + P
    for _ in range(60):                       # the cell's bisection
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if np.maximum(0, mid - N).sum() < P else (lo, mid)
    theta = 0.5 * (lo + hi)                   # ~ 1.4333

    width, gap = 0.84, 0.16
    for i, n in enumerate(N):
        x = i + gap / 2
        ax.add_patch(plt.Rectangle((x, 0), width, n, facecolor=GRAY,
                                   alpha=0.45, lw=0))
        ax.plot([x, x + width], [n, n], color=GRAY, lw=1.6)
        if n < theta:                         # water fills up to the level
            ax.add_patch(plt.Rectangle((x, n), width, theta - n,
                                       facecolor=BLUE, alpha=0.28, lw=0))
        else:                                 # floor above the level
            ax.text(i + 0.5, n + 0.13, "dry", color="black", fontsize=LBL,
                    ha="center", va="bottom")
        ax.text(i + 0.5, -0.13, rf"$N_{{{i + 1}}}={n}$", color="black",
                fontsize=LBL, ha="center", va="top")

    ax.plot([0.0, len(N)], [theta, theta], color=ORANGE, lw=1.7,
            ls=(0, (5, 3)), zorder=4)
    ax.text(len(N) + 0.08, theta,
            r"water level $\theta=1/\nu\approx1.43$", color=ORANGE,
            fontsize=LBL, ha="left", va="center")

    xi = 1 + 0.5                              # P_2* = theta - N_2
    ax.annotate("", xy=(xi, theta - 0.015), xytext=(xi, N[1] + 0.015),
                arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.4))
    ax.text(xi + 0.08, 0.5 * (theta + N[1]), r"$P_2^\star$",
            color=BLUE, fontsize=LBL + 1, ha="left", va="center")

    ax.plot([0.0, len(N)], [0.0, 0.0], color="black", lw=1.2)
    ax.text(2.5, -0.42, "channels, floors at the noise levels $N_i$",
            color="black", fontsize=LBL, ha="center", va="top")

    fl.clean_axes(ax, lim=((-0.15, 7.3), (-0.75, 2.95)), hide=True,
                  equal=False)
    fl.save(fig, "mdl-opt-water-filling")
    import shutil                             # slides read img/auto/
    shutil.copyfile(os.path.join(fl.IMG_DIR, "mdl-opt-water-filling.svg"),
                    os.path.join(fl.IMG_DIR, "auto", "mdl-opt-water-filling.svg"))


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
                arrowprops=dict(arrowstyle="->", color="black", lw=1.1))
    for v in ticks:
        tall = v in (0.25, 0.5, 1.0, 2.0, 4.0)
        ax.plot([v, v], [-0.13 if tall else -0.08, 0.13 if tall else 0.08],
                color=BLUE, lw=1.5 if tall else 1.1)
    for v in sub:
        ax.plot([v, v], [-0.08, 0.08], color=GRAY, lw=1.0, alpha=0.8)
    ax.plot([0, 0], [-0.13, 0.13], color="black", lw=1.5)
    # every tick label centred on its tick.  "0" and "0.25" sit only 0.25
    # apart and collided when centred; nudged apart, "0.25" stood beside the
    # 0.5 tick and read as its label.  So 0.25 moves into the "smallest
    # normal" annotation and the tall 0.5 tick gets its own label.
    for v, lab in ((0, "0"), (0.5, "0.5"), (1, "1"), (2, "2"), (4, "4")):
        ax.text(v, -0.24, lab, color="black", fontsize=LBL, ha="center",
                va="top")

    # subnormal / underflow region below the smallest normal
    ax.add_patch(plt.Polygon([[0, -0.16], [0.25, -0.16], [0.25, 0.16],
                              [0, 0.16]], closed=True, facecolor=ORANGE,
                             alpha=0.15, lw=0))
    # Two stacked gray captions on the left, well separated in height: the top
    # one names the shaded subnormal band, the lower one marks the 0.25 boundary
    # as the smallest normal.  Both are left-anchored at x ~ 0 so neither runs
    # into the eps_mach marker over x = 1.
    # relpos=(0, 0) pins the leader to the text's own bottom-left corner
    # (near x=0.02); without it, matplotlib anchors from the *center* of
    # this wide text box, which sent the leader diagonally across the
    # "smallest normal" label below.
    ax.annotate("subnormals $\\to$ underflow to $0$", xy=(0.12, 0.17),
                xytext=(0.02, 0.74), color="black", fontsize=LBL, ha="left",
                va="center",
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.9,
                                 relpos=(0, 0)))
    # xytext's x matches xy's x (0.25) -- with ha="left" and relpos=(0, 0)
    # (leader pinned to the text's own bottom-left corner), that makes the
    # leader run straight up to the smallest-normal tick instead of leaning
    # in from the left as it did anchored at x=0.07.
    ax.annotate("smallest normal $0.25$", xy=(0.25, 0.155), xytext=(0.25, 0.48),
                color="black", fontsize=LBL, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8,
                                 relpos=(0, 0)))

    # eps_mach: the gap from 1 to its right neighbor 1 + 2^-p.  The label sits
    # up and to the right of the little gap, joined by a short leader, so it
    # clears the "smallest normal" caption to its left entirely.
    eps = 2.0 ** -p
    # |-| brackets: on a 0.25-wide gap, "<->" heads overlapped into an X
    gap = dict(arrowstyle="|-|,widthA=0.25,widthB=0.25", lw=1.1,
               shrinkA=0, shrinkB=0)
    ax.annotate("", xy=(1 + eps, 0.30), xytext=(1, 0.30),
                arrowprops=dict(gap, color="black"))
    # label right beside its bracket, no leader
    ax.text(1 + eps + 0.07, 0.30, r"$\varepsilon_{\mathrm{mach}}$",
            color="black", fontsize=LBL + 1, ha="left", va="center")

    # gap doubling: matched brackets under one gap of each binade, low
    # enough (-0.56) to clear the 13 pt tick numbers anchored at -0.24.
    for lo in (1.0, 2.0):
        ax.annotate("", xy=(lo + lo / 2**p, -0.56), xytext=(lo, -0.56),
                    arrowprops=dict(gap, color=GRAY))
    ax.text(2.05, -0.68, "gap doubles", color="black", fontsize=LBL, ha="center",
            va="top")

    # axis break, then the overflow cliffs (schematic positions)
    for dx in (4.62, 4.74):
        ax.plot([dx - 0.06, dx + 0.06], [-0.10, 0.10], color="black", lw=1.2)
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
        ax.text(x, 0.38, lab, color=ORANGE, fontsize=LBL, ha="center",
                va="bottom")
    ax.text(6.22, -0.30, r"overflow $\to\infty$", color=ORANGE, fontsize=LBL,
            ha="center", va="top")

    fl.clean_axes(ax, lim=((-0.3, 7.15), (-0.95, 0.95)), hide=True,
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
    sigma_1^2 + lambda and sigma_d^2 + lambda that set kappa's ratio.
    """
    # Rotated 90 degrees from the original layout: the slow/shallow axis
    # (small eigenvalue, long semi-axis) now runs horizontal instead of
    # vertical, so the panel is wide-and-short rather than narrow-and-tall.
    # figsize and the xlim/ylim extents below are the old ylim/xlim swapped.
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.4, 3.5))
    s2 = np.array([1.0, 1.0 / 16.0])          # sigma_1^2, sigma_d^2
    xlim, ylim = (-3.25, 3.25), (-2.2, 2.2)
    gx = np.linspace(*xlim, 300)
    gy = np.linspace(*ylim, 240)
    X, Y = np.meshgrid(gx, gy)
    # start point with its two components swapped to match the rotated axes,
    # in the left half-plane: from the right, the zig-zag covered the
    # right-pointing sigma_d^2 arrow and buried the short sigma_1^2 arrow
    x0 = np.array([-2.8, -0.72])

    def panel(ax, ridge, title, steps, axis_labels):
        h = s2 + ridge                        # h[0] = sigma_1^2 (steep),
                                               # h[1] = sigma_d^2 (shallow)
        # x gets the shallow eigenvalue (long semi-axis, horizontal), y gets
        # the steep one (short semi-axis, vertical) -- this is the 90-degree
        # rotation relative to the original h[0]-on-x/h[1]-on-y layout.
        hx, hy = h[1], h[0]
        Z = 0.5 * (hx * X**2 + hy * Y**2)
        # levels chosen so the x-semi-axes are evenly spaced on screen
        semis = np.array([0.55, 1.05, 1.55, 2.1, 2.65, 3.2])
        levels = list(0.5 * hx * semis**2)
        # In the elongated panel (a) the closed ellipses use only the middle
        # third of the panel's height; add the valley walls (levels whose
        # vertical semi-axes continue the spacing up to the panel edge) so
        # the contours fill the panel as they do in (b).
        b_top = semis[-1] * np.sqrt(hx / hy)
        levels += [0.5 * hy * b**2 for b in np.arange(b_top + 0.4, 2.15, 0.4)]
        ax.contour(X, Y, Z, levels=levels, colors=[LIGHT], linewidths=0.9)
        # principal axes of the ellipse with x-semi-axis 2.1, labelled with
        # the curvatures sigma^2 + lambda along each axis
        c = 0.5 * hx * 2.1**2
        ex, ey = 2.1, np.sqrt(2 * c / hy)
        fl.arrow(ax, (0, 0), (ex, 0), color=GREEN, lw=1.8)
        fl.arrow(ax, (0, 0), (0, ey), color=GREEN, lw=1.8)
        # The long horizontal axis (shallow eigenvalue): label just under
        # the arrow tip (the path now runs in the left half-plane), the same
        # spot in both panels.  White boxes keep both labels legible over
        # the faint level sets.
        box = dict(boxstyle="round,pad=0.12", fc="white", ec="none")
        ax.text(ex - 0.05, -0.14, axis_labels[1], color=GREEN,
                fontsize=LBL + 1, ha="center", va="top", bbox=box)
        # The short vertical axis (steep eigenvalue): beside its arrowhead.
        ax.text(0.12, ey + 0.05, axis_labels[0], color=GREEN,
                fontsize=LBL + 1, ha="left", va="bottom", bbox=box)
        # real GD at the optimal step for this Hessian
        eta = 2.0 / (h[0] + h[1])
        h_rot = np.array([hx, hy])
        x = x0.copy()
        path = [x.copy()]
        for _ in range(steps):
            x = x - eta * (h_rot * x)
            path.append(x.copy())
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], "-o", color=ORANGE, ms=3.2, lw=1.5,
                zorder=4)
        ax.plot(0, 0, "*", color=GREEN, ms=12, zorder=5)
        ax.set_title(title, fontsize=LBL + 2)
        fl.clean_axes(ax, lim=(xlim, ylim), hide=True)

    kappa0 = s2[0] / s2[1]
    kappa1 = (s2[0] + 0.5) / (s2[1] + 0.5)
    panel(axa, 0.0, rf"(a) $\lambda=0$:  $\kappa={kappa0:.0f}$", 28,
          (r"$\sigma_1^2$", r"$\sigma_d^2$"))
    panel(axb, 0.5, rf"(b) $\lambda=0.5$:  $\kappa\approx{kappa1:.1f}$", 12,
          (r"$\sigma_1^2+\lambda$", r"$\sigma_d^2+\lambda$"))

    fl.save(fig, "mdl-opt-conditioning-ellipse")


# =========================================================================== #
# Driver                                                                      #
# =========================================================================== #

FIGURES = [
    # 3.1 gradient-based optimization
    fig_gd_bowl_vs_valley,
    fig_eta_tent,
    fig_momentum_damping,
    fig_sgd_noise_ball,
    # stochastic and adaptive methods
    fig_per_coordinate,
    fig_schedule_zoo,
    fig_bias_correction,
    # 3.2 convexity
    fig_chord_above_graph,
    fig_convex_vs_nonconvex_set,
    fig_subgradient_fan,
    fig_local_equals_global,
    # 3.3 constrained optimization and duality
    fig_lagrange_tangency,
    fig_kkt_active_set,
    fig_primal_dual_gap,
    fig_water_filling,
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
