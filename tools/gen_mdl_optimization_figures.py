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
    axb.text(1.05, 0.09, r"slow axis ($\lambda_{\min}$)", color=GRAY,
             fontsize=11, ha="center", va="bottom")

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

    ax.axhline(1.0, color=GRAY, lw=1.2, ls=(0, (5, 3)))
    # right-anchored just short of the lambda_max gridline (at 10) -- anchoring
    # at the old x=10.75 straddled that gridline right through the "eta lambda".
    ax.text(9.85, 1.07, "instability: $|1-\\eta\\lambda|=1$", color=GRAY,
            fontsize=11, ha="right", va="bottom")

    for eta, col, lab in etas:
        ax.plot(lam, np.abs(1 - eta * lam), color=col, lw=2.0, zorder=3)
        for l0 in (lmin, lmax):                # the two extreme modes
            ax.plot(l0, abs(1 - eta * l0), "o", color=col, ms=6.5, zorder=5)

    # eigenvalue gridlines
    for l0, lab in ((lmin, r"$\lambda_{\min}$"), (lmax, r"$\lambda_{\max}$")):
        ax.plot([l0, l0], [0, 1.12], color=GRAY, lw=0.9, ls=(0, (2, 3)),
                zorder=1)
        ax.text(l0, -0.06, lab, color="black", fontsize=11, ha="center",
                va="top")

    # the balanced level of the optimal step
    rho = (lmax - lmin) / (lmax + lmin)
    ax.plot([lmin, lmax], [rho, rho], color=GREEN, lw=1.0, ls=(0, (2, 3)),
            zorder=2)
    # The two green formulas (the eta* definition and the resulting rho) live
    # together in the one genuinely empty pocket of this figure: the dome
    # bounded below by the higher of the blue/green curves and above by the
    # rho dotted line, centered around lambda ~ 6.7 -- well clear of both
    # curves and of the two vertical eigenvalue gridlines.
    ax.text(6.7, 0.59,
            r"$\eta^\star=\frac{2}{\lambda_{\min}+\lambda_{\max}}$"
            "\n" r"$\rho(\eta^\star)=\frac{\kappa-1}{\kappa+1}=\frac{9}{11}$",
            color=GREEN, fontsize=11, ha="center", va="center")
    # "both endpoints equal" no longer shares that pocket (three stacked lines
    # there pushed the top edge flush against the rho dotted line): it moves
    # to the large empty triangle below the green descending arm, with a
    # short leader that stays under the green curve the whole way and meets
    # it exactly at the left dot -- so the leader only touches the curve at
    # its own target, never crosses it.
    ax.annotate("both endpoints equal", xy=(lmin, rho), xytext=(2.3, 0.15),
                color=GREEN, fontsize=11, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.8))

    # the greedy eta = 0.1: stiff mode dead, slow mode rules.  Placed in the
    # clear pocket just right of lambda_max (the blue curve stays low there),
    # so the label clears both the lambda_max gridline and the curve.
    ax.annotate("stiff mode killed", xy=(10.05, 0.05), xytext=(10.2, 0.34),
                color=BLUE, fontsize=11, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.9))
    # shifted right of the lambda_min dot (to x=1.55, where the blue line has
    # already dropped to ~0.845) and centered vertically at 0.92, splitting
    # the difference evenly between the blue line/dot below and the eta=1
    # instability dashed line above -- the old anchor (x=1.15) sat directly
    # over the dot, leaving no headroom that cleared both lines at once.
    ax.text(1.55, 0.92, r"slow mode barely moves: $0.9$", color=BLUE,
            fontsize=11, ha="left", va="center")
    ax.text(11.5, -0.06, r"$\lambda$", color="black", fontsize=12,
            ha="right", va="top")
    ax.text(-0.25, 1.12, r"$|1-\eta\lambda|$", color="black", fontsize=12,
            ha="left", va="bottom")

    # baseline
    ax.plot([0, 11.6], [0, 0], color="black", lw=1.0)
    fl.clean_axes(ax, lim=((-0.35, 11.75), (-0.28, 1.30)), hide=True,
                  equal=False)
    fl.save(fig, "mdl-opt-eta-tent")


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
        ax.set_title(lab, color=col, fontsize=12)
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
                color=GRAY, fontsize=11, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.9))
    ax.annotate(r"step $\eta/2$",
                xy=(-(r / np.sqrt(2)) * np.cos(0.8),
                    -(r / np.sqrt(2)) * np.sin(0.8)), xytext=(-1.85, -1.25),
                color=GRAY, fontsize=11, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.9))

    # heading set just above the GD start point so the descending line stays
    # below it and never strikes through the text.
    ax.text(-2.02, 1.70, "gradient descent", color=BLUE, fontsize=11,
            ha="left", va="bottom")
    ax.text(-0.45, 0.95, r"SGD, fixed $\eta$", color=ORANGE, fontsize=11,
            ha="left", va="bottom")

    fl.clean_axes(ax, lim=((-2.45, 2.3), (-1.7, 1.95)), hide=True)
    fl.save(fig, "mdl-opt-sgd-noise-ball")


# =========================================================================== #
# Stochastic and Adaptive Methods (sec_mdl-adaptive-stochastic-methods)       #
# =========================================================================== #

def fig_per_coordinate():
    """The adaptive-methods thesis drawn (sec_mdl-adaptive-stochastic-methods,
    Per-Coordinate Step Sizes): the same elongated quadratic
    f(x, y) = 1/2 (x^2 + 20 y^2) from the same start, under

    (a) gradient descent with one global step size near the stability ceiling
        (eta = 0.097, 26 steps -- the exact run of fig_gd_bowl_vs_valley's
        right panel): the stiff axis oscillates, the slow axis crawls;
    (b) Adam's per-coordinate normalization (beta_1 = 0 to isolate the
        rescaling from momentum, alpha = 0.12, 40 steps): both coordinates
        move at comparable speed and the path curves smoothly into the
        minimum -- the valley is effectively rounded.

    Both trajectories are honest runs of the stated updates.
    """
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.0, 4.2))

    lam = np.array([1.0, 20.0])
    g = np.linspace(-1.7, 1.7, 220)
    X, Y = np.meshgrid(g, g)
    Z = 0.5 * (lam[0] * X**2 + lam[1] * Y**2)
    x0 = np.array([-1.4, 1.0])

    for ax in (axa, axb):
        ax.contour(X, Y, Z, levels=14, colors=[LIGHT], linewidths=0.9)
        ax.plot(0, 0, "*", color=GREEN, ms=13, zorder=5)

    # (a) gradient descent, one global eta near 2/lambda_max.
    x, path = x0.copy(), [x0.copy()]
    for _ in range(26):
        x = x - 0.097 * lam * x
        path.append(x.copy())
    path = np.array(path)
    axa.plot(path[:, 0], path[:, 1], "-o", color=ORANGE, ms=3.5, lw=1.6,
             zorder=4)

    # (b) Adam's per-coordinate rescaling (beta_1 = 0, bias-corrected v).
    x, m, v = x0.copy(), np.zeros(2), np.zeros(2)
    path = [x0.copy()]
    for k in range(1, 41):
        gr = lam * x
        v = 0.999 * v + 0.001 * gr * gr
        x = x - 0.12 * gr / (np.sqrt(v / (1 - 0.999**k)) + 1e-8)
        path.append(x.copy())
    path = np.array(path)
    axb.plot(path[:, 0], path[:, 1], "-o", color=BLUE, ms=3.5, lw=1.6,
             zorder=4)
    # annotate the equal-speed opening segment.
    axb.annotate("equal speed on\nboth coordinates",
                 xy=(-0.85, 0.47), xytext=(-0.05, 1.05), color=BLUE,
                 fontsize=11, ha="center", va="center",
                 arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.9))

    for ax in (axa, axb):
        fl.clean_axes(ax, lim=((-1.7, 1.7), (-1.7, 1.7)), hide=True)
    fl.save(fig, "mdl-opt-per-coordinate")


def fig_schedule_zoo():
    """The schedule zoo (sec_mdl-adaptive-stochastic-methods, Schedules and
    Warmup): eta_t over a fixed budget K for the four shapes the section
    discusses -- constant, c/k decay, cosine, and warmup-stable-decay (WSD) --
    all beginning with the same linear warmup over the first 5% of the budget.
    A dashed gray line marks the noise-floor reading: fixed-step SGD parks on
    a floor proportional to eta, so the constant schedule keeps paying it while
    the decaying schedules pay it down; WSD does all of that paying in its
    final decay phase.
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

    for y, col, lab, ls in [
            (constant, GRAY, "constant", (0, (5, 3))),
            (inv_k, GREEN, r"$c/k$", "-"),
            (cosine, BLUE, "cosine", "-"),
            (wsd, ORANGE, "WSD", "-")]:
        ax.plot(t, y, color=col, lw=2.0, ls=ls, zorder=3)

    ax.text(430, 1.045, "constant: noise floor $\\propto\\eta$ forever",
            color=GRAY, fontsize=11, ha="center", va="bottom")
    # shifted left from x=345 -- at that anchor the "m" of "optimum," sat
    # right on the descending blue cosine curve.
    ax.annotate("$c/k$: reaches the optimum,\nif $c$ is large enough",
                xy=(210, float(inv_k[210])), xytext=(300, 0.47), color=GREEN,
                fontsize=11, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.9))
    # moved right/low, where the cosine curve has already dropped well below
    # the label -- the earlier placement had the curve cut through "of".
    ax.annotate("cosine: long tail\nof small steps",
                xy=(700, cosine[700]), xytext=(760, 0.40), color=BLUE,
                fontsize=11, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.9))
    # anchored over the flat plateau (well above the decay ramp) so the
    # descending WSD line never cuts through the two-line label.
    ax.annotate("WSD: hold the plateau,\nthen pay the ball down",
                xy=(880, wsd[880]), xytext=(500, 0.88), color=ORANGE,
                fontsize=11, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.9))
    # moved clear of the c/k annotation block to its right.
    ax.annotate("warmup", xy=(w, 0.55), xytext=(95, 0.15), color="black",
                fontsize=11, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color="black", lw=0.8))

    ax.plot([0, K], [0, 0], color="black", lw=1.0)
    ax.text(K, -0.07, r"$t$", fontsize=12, ha="right", va="top")
    ax.text(8, 1.10, r"$\eta_t$", fontsize=12, ha="left", va="bottom")
    fl.clean_axes(ax, lim=((-25.0, 1030.0), (-0.16, 1.24)), hide=True,
                  equal=False)
    fl.save(fig, "mdl-opt-schedule-zoo")


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
                 fontsize=11, ha="center", va="center",
                 arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.9))
    axa.text(2500, 1.05, r"corrected: $\hat{v}_t = v_t/(1-\beta_2^t)$",
             color=GREEN, fontsize=11, ha="center", va="bottom")
    axa.set_xlabel(r"$t$")
    fl.clean_axes(axa, lim=((-100.0, 5100.0), (-0.06, 1.3)), hide=False,
                  equal=False)

    ratio = (1 - b1**t) / np.sqrt(1 - b2**t)
    tpk = int(t[np.argmax(ratio)])
    axb.axhline(1.0, color=GREEN, lw=1.2, ls=(0, (5, 3)), zorder=2)
    axb.plot(t, ratio, color=ORANGE, lw=2.0, zorder=3)
    axb.set_xscale("log")
    axb.plot(1, ratio[0], "o", color=ORANGE, ms=6, zorder=4)
    axb.plot(tpk, ratio.max(), "o", color=ORANGE, ms=6, zorder=4)
    axb.annotate(r"$3.16\times$ at $t=1$",
                 xy=(1, ratio[0]), xytext=(2.3, 1.75), color=ORANGE,
                 fontsize=11, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.9))
    axb.annotate(rf"peak ${ratio.max():.1f}\times$ at $t={tpk}$",
                 xy=(tpk, ratio.max()), xytext=(90, 6.4), color=ORANGE,
                 fontsize=11, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.9))
    axb.text(700, 1.15, "properly scaled: 1", color=GREEN, fontsize=11,
             ha="center", va="bottom")
    axb.set_title(r"(b) raw-step inflation $(1-\beta_1^t)/\sqrt{1-\beta_2^t}$")
    axb.set_xlabel(r"$t$")
    axb.set_xlim(0.85, 5500.0)
    axb.set_ylim(0.0, 7.2)
    axb.set_aspect("auto")

    fig.subplots_adjust(wspace=0.22)
    fl.save(fig, "mdl-opt-bias-correction")


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
    # both dot labels pushed further out from the curve (dy 0.30->0.55 and
    # 0.15->0.45) -- the old, tighter offsets left them sitting right on the
    # steep parabola arms.  "y" also flips to right-anchored (text growing
    # back toward the vertex, where the arm is shallower) instead of
    # left-anchored into the steepening right arm.
    fl.vlabel(axa, (a - 0.05, f(a) + 0.55), r"$\mathbf{x}$", color=BLUE,
              ha="right", fontsize=11)
    fl.vlabel(axa, (b - 0.05, f(b) + 0.45), r"$\mathbf{y}$", color=BLUE,
              ha="right", fontsize=11)
    # shifted well left of the y-axis and raised (was centered near x=-0.47,
    # dy=0.55, straddling the vertical axis through "(1-theta)"; x=-1.4 at
    # the same height still grazed the axis and the raised-but-not-shifted
    # variants clipped the rising left arm of the parabola) -- x=-1.6,
    # dy=0.85 clears both the axis on the right and the curve on the left.
    axa.text(-1.6, chord_y + 0.85,
             r"$\theta f(\mathbf{x})+(1-\theta)f(\mathbf{y})$",
             color=ORANGE, fontsize=11, ha="center", va="bottom")
    axa.text(xc - 0.15, graph_y - 0.62, r"$f(\theta\mathbf{x}+(1-\theta)\mathbf{y})$",
             color=BLUE, fontsize=11, ha="center", va="top")
    fl.axis_cross(axa, (-3.6, 3.6), (-0.6, 3.0), color="black")
    fl.clean_axes(axa, lim=((-3.6, 3.6), (-0.9, 3.0)), hide=True)

    # ----- (b) tangent below the graph -----
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
    # label the tangent (global under-estimator) in the open band below the
    # x-axis; right-anchored so it clears the dashed gap line at y0 and never
    # crosses the curve or the faint axes.  Anchor moved left from x=1.3 to
    # x=0.5 -- at 1.3 the tangent line itself (which continues down-right
    # through this band) cut through the trailing "(y-x)".
    axb.text(0.5, -1.12,
             r"$f(\mathbf{x})+\nabla f(\mathbf{x})^\top(\mathbf{y}-\mathbf{x})$",
             color=GREEN, fontsize=11, ha="right", va="center")
    fl.axis_cross(axb, (-3.6, 3.6), (-0.6, 3.0), color="black")
    fl.clean_axes(axb, lim=((-3.6, 3.6), (-1.6, 3.0)), hide=True)

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
    axb.text(0.36, 0.0, "outside", color=BLUE, fontsize=11, ha="left",
             va="center")
    # tightened to the crescent's actual footprint (x in [0, 1.65]) instead of
    # the old symmetric box, which left a large dead margin on the left.
    fl.clean_axes(axb, lim=((-0.2, 1.85), (-1.15, 1.15)), hide=True)

    # ----- (c) simplex + half-space -----
    simplex = np.array([[0.0, 0.0], [1.7, 0.0], [0.0, 1.7]])
    axc.add_patch(plt.Polygon(simplex, closed=True, facecolor=GREEN, alpha=0.18,
                              edgecolor=GREEN, lw=2.0))
    # the label doesn't fit inside the (fairly small) triangle -- moved below
    # it, with a short leader pointing back up into the interior.
    axc.annotate(r"$\{\mathbf{p}\succeq0,\ \mathbf{1}^\top\mathbf{p}=1\}$",
                 xy=(0.85, 0.22), xytext=(0.85, -0.45), color=GREEN,
                 fontsize=11, ha="center", va="center",
                 arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.9))
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
             fontsize=11, ha="center", va="center")
    fl.axis_cross(axc, (-1.4, 2.2), (-1.4, 2.2), color="black")
    fl.clean_axes(axc, lim=((-1.4, 2.2), (-1.4, 2.2)), hide=True)

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
    # the zero-slope supporting line: the optimality certificate
    ax.plot(span, 0.0 * span, color=ORANGE, lw=2.0, zorder=3)

    ax.plot(0, 0, "o", color="black", ms=6.5, zorder=6)
    # Centered on x=0 this two-line block straddled the y-axis (it runs right
    # through the vertical arrow) no matter how it was nudged -- any text
    # wide enough to read comfortably spans well past the axis, and every
    # nearby off-axis spot still sits inside the fan's own footprint
    # (x in [-1.75, 1.75]), so it clips one of the outer lines instead.
    # Moved entirely outside that footprint, to the lower right (fontsize
    # trimmed slightly, wrapped into 3 short lines, so it clears both the
    # fan's right edge and the plot's right border) with a short leader back
    # to the outermost (g=-1) line.
    ax.annotate(r"$\partial|x|(0)=[-1,\,1]$:"
                "\nevery slope $g\\in[-1,1]$"
                "\nsupports the graph",
                xy=(1.35, -1.35), xytext=(2.65, -1.1), color=GREEN,
                fontsize=9, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.9))
    ax.text(1.92, 0.10, r"$0\in\partial f(0)$:" "\nthe corner is a minimum,"
            "\nno gradient required",
            color=ORANGE, fontsize=11, ha="left", va="bottom")
    ax.text(-1.3, 2.05, r"$f(x)=|x|$", color=BLUE, fontsize=12, ha="center",
            va="center")

    fl.axis_cross(ax, (-2.25, 2.6), (-1.9, 2.2), color="black")
    fl.clean_axes(ax, lim=((-2.25, 3.55), (-1.95, 2.25)), hide=True,
                  equal=False)
    fl.save(fig, "mdl-opt-subgradient-fan")


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
    fl.axis_cross(axa, (-3.4, 3.4), (-0.9, 3.0), color="black")
    # matches panel (b)'s box exactly (same xlim/ylim) so the two panels
    # share the same window size and scale.
    fl.clean_axes(axa, lim=((-3.4, 3.4), (-1.1, 3.0)), hide=True)

    # ----- (b) non-convex: two basins + a saddle (a 1-D bumpy landscape) -----
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
        axb.text(m, g(m) + 0.22, "saddle", color=GRAY, fontsize=11,
                 ha="center", va="bottom")
    axb.text(minima[0], g(minima[0]) - 0.30, "local min", color="black",
             fontsize=11, ha="center", va="top")
    axb.text(minima[-1], g(minima[-1]) - 0.30, "global min", color="black",
             fontsize=11, ha="center", va="top")
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
              r"$\nabla g$", color=BLUE, fontsize=11)
    fl.vlabel(ax, xstar - 0.50 * xstar + np.array([-0.02, -0.20]),
              r"$\nabla f$", color=ORANGE, fontsize=11)
    # x* label moved right and up from the tangency point -- the old
    # lower-right offset (0.30, -0.24) put it right on top of the tangent
    # r=1 level-set circle (both circles meet exactly at x*, so anything
    # close to it grazes one curve or the other).  (0.30, 0.40) clears the
    # r=1 circle, the constraint circle, and the grad-g arrow all at once.
    fl.vlabel(ax, xstar + np.array([0.30, 0.40]), r"$\mathbf{x}^\star$",
              color="black", fontsize=11)

    # ---- a non-optimal feasible point: grad f crosses the constraint ----
    ax.plot(*x0, "o", color="black", ms=5.5, zorder=6)
    df = (x0 - c) / np.linalg.norm(x0 - c)    # unit grad f at x0
    fl.arrow(ax, x0, x0 + 0.55 * df, color=ORANGE, lw=2.0)
    fl.vlabel(ax, x0 + 0.72 * df + np.array([0.0, 0.12]), r"$\nabla f$",
              color=ORANGE, fontsize=11)
    # feasible descent: minus the tangential component of grad f (first-order
    # move along the constraint), drawn dashed
    tang = np.array([-x0[1], x0[0]])          # unit tangent at x0
    desc = -np.dot(df, tang) * tang
    desc = 0.5 * desc / np.linalg.norm(desc)
    fl.arrow(ax, x0, x0 + desc, color=GREEN, lw=1.8, ls=(0, (4, 3)))
    fl.vlabel(ax, x0 + desc + np.array([0.04, 0.20]), "feasible\ndescent",
              color=GREEN, fontsize=11)

    ax.text(-0.55, -1.05, r"constraint $g(\mathbf{x})=0$", color=BLUE,
            fontsize=11, ha="center", va="top")
    # anchored in the clear upper-left corner (outside every circle) with a
    # short leader to one ring -- the previous unanchored placement sat right
    # on top of the tangent (r=1) circle's crest.
    ax.annotate(r"level sets of $f$", xy=(2.625, 2.225), xytext=(-1.85, 2.3),
                color=ORANGE, fontsize=11, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8))

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
        # dropped from y=0.18 to y=0.0 -- in panel (a) the outermost (r=1.8)
        # level-set circle swept down through the old spot.
        ax.text(0.05, 0.0, "feasible", color=BLUE, fontsize=11,
                ha="center", va="center")
        # moved from just above the g1=0 line (y=1.10) to just below it
        # (inside the feasible region): every on-the-line position in panel
        # (a) is swept by the active edge's own grad-g1 arrow, which has to
        # start on this line -- placing the label on the *other* side of the
        # line avoids that arrow with no need to also fight the level-set
        # circles for horizontal room.
        ax.text(-0.62, 0.68, r"$g_1=0$", color=BLUE, fontsize=11, ha="left",
                va="top")
        ax.text(1.60, -0.38, r"$g_2=0$", color=BLUE, fontsize=11, ha="left",
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
    fl.arrow(axa, xs, xs + np.array([0.0, -0.35]), color=ORANGE, lw=2.0)
    # moved well below all three level-set circles: each circle's minimum
    # height is y = 1.9 - r, so anything below y=1.9-1.8=0.1 (with margin)
    # clears all of them regardless of how wide the formula runs. Also
    # wrapped onto two lines and dropped further (y=-0.20 -> -0.27) -- as
    # one long line it ran past x=1.5 into the g2=0 boundary line, and its
    # top edge grazed "feasible".
    axa.text(0.15, -0.27,
             r"$-\nabla f=\lambda_1\nabla g_1,$" "\n" r"$\lambda_1>0$",
             color=ORANGE, fontsize=11, ha="left", va="center")
    fl.vlabel(axa, xs + np.array([-0.22, 0.14]), r"$\mathbf{x}^\star$",
              color="black", fontsize=11)
    # outward normal of the active edge, shown a bit along the edge -- moved
    # from x=-0.30 to x=-0.45 (any x in (-0.379, 0.4) has the r=0.9 circle's
    # left branch crossing it somewhere in y in [1.0, 1.45]; -0.45 clears
    # all three circles outright).
    fl.arrow(axa, (-0.45, e1y), (-0.45, e1y + 0.45), color=BLUE, lw=1.8)
    # label grows right from the arrow tip instead of being centered on it,
    # so it doesn't reach back left into the r=0.9 circle.
    axa.text(-0.40, e1y + 0.62, r"$\nabla g_1$", color=BLUE, fontsize=11,
              ha="left", va="center")
    # pushed further right of e2x -- at fontsize 11 the label's vertical run
    # otherwise cuts across the r=1.35 level-set arc around its "lambda_2".
    axa.text(e2x + 0.6, 0.42, r"inactive: $\lambda_2=0$", color=GRAY,
             fontsize=11, ha="left", va="center", rotation=90)

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
    fl.vlabel(axb, corner + np.array([-0.05, 0.70]), r"$\nabla g_1$",
              color=BLUE, fontsize=11, ha="right")
    fl.vlabel(axb, corner + np.array([0.72, -0.16]), r"$\nabla g_2$",
              color=BLUE, fontsize=11)
    axb.text(*(corner + 0.62 * df + np.array([0.07, 0.16])), r"$-\nabla f$",
             color=ORANGE, fontsize=11, ha="left", va="center")
    # The old spot (corner + (0.30, -0.40), with a short vertical leader
    # up to the wedge) sat inside both the r=1.13 and r=1.55 level-set
    # circles.  Those two circles are centered up at cb and their combined
    # footprint covers essentially the whole area immediately around the
    # corner and wedge, so no nearby spot clears them; the label moves well
    # down and left into the open feasible region instead, with no leader
    # (a straight one from here would have to cross the outermost, r=1.95,
    # circle to reach back to the wedge -- the wedge's own shape next to it
    # already carries the association).
    axb.text(1.1, -0.05, "cone of\nnormals",
             color=GRAY, fontsize=11, ha="center", va="center")
    fl.vlabel(axb, corner + np.array([-0.22, -0.16]), r"$\mathbf{x}^\star$",
              color="black", fontsize=11)

    for ax in (axa, axb):
        fl.clean_axes(ax, lim=(xlim, ylim), hide=True)
    fl.save(fig, "mdl-opt-kkt-active-set")


def fig_primal_dual_gap():
    """The supporting-line picture of duality (sec_mdl-constrained-
    optimization-duality, Duality): map every x to (u, t) = (constraint value,
    objective value); the *image set* G of all such pairs is shaded.  Evaluating
    the dual at lambda lowers a line of slope -lambda until it supports G from
    below; its height at the axis u = 0 is the dual value g(lambda) <= p*.  Both
    panels are exact computations, with only the single *best* supporting line
    drawn so the one idea reads cleanly:

    (a) convex (min x^2 s.t. 1 - x <= 0): the image boundary is the parabola
        t = (1-u)^2 and G is convex; the best supporting line (lambda* = 2) is
        tangent at (0, p*) -- strong duality, d* = p* = 1, the line meets the
        axis exactly at p*;
    (b) the chapter's non-convex example (min -x^2 s.t. x - 1/2 <= 0 on
        [0, 1]): the image boundary t = -(u + 1/2)^2 is dented, so G is not
        convex; the best supporting line (lambda* = 1) passes *under* the dent
        and meets the axis at d* = -1/2 < p* = -1/4 -- a visible duality gap.

    Design: G is shaded (everything on/above the boundary curve is achievable-
    or-dominated), the feasible slice u <= 0 of the boundary is drawn thick in
    orange, and exactly one green supporting line is shown per panel with its
    axis intercept marked.  Earlier versions also drew a *suboptimal* dashed
    line; it crossed the curve and the optimal line and made the panel unreadable
    (the author's "lots of lines intersecting"), so it is gone -- "lower the line
    until it supports" is carried by an arrow on the single optimal line.
    """
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.4, 4.5))

    def axis_labels(ax, xr, yr, u_label_dx=0.0):
        """Faint origin cross with the two value-axis names tucked at the far
        ends, clear of the curve and every annotation.  ``u_label_dx`` nudges
        the "constraint value u" label further right (panel (a): the green
        supporting line's segment is still visible down at that label's
        height, and a plain right-anchor at xr[1] lets the line cut through
        the text's left portion)."""
        fl.axis_cross(ax, xr, yr, color="black")
        ax.text(xr[1] + u_label_dx, -0.04 * (yr[1] - yr[0]),
                "constraint value  $u$",
                color="black", fontsize=11, ha="right", va="top")
        # objective-axis name set just right of the vertical axis at the top,
        # so it never collides with the curve in the upper-left.
        ax.text(0.04 * (xr[1] - xr[0]), yr[1], "objective value  $t$",
                color="black", fontsize=11, ha="left", va="top")

    # =================== (a) convex: strong duality =================== #
    xr_a, yr_a = (-1.25, 1.85), (-1.45, 3.8)
    u = np.linspace(-0.95, 1.7, 340)
    boundary_a = (1 - u) ** 2
    # the image set G = {(u, t) : t >= (1-u)^2}: shade everything above the
    # boundary so "support from below" is literally what the green line does.
    axa.fill_between(u, boundary_a, yr_a[1], color=BLUE, alpha=0.08, lw=0,
                     zorder=1)
    axa.plot(u, boundary_a, color=BLUE, lw=2.0, zorder=3)
    uf = u[u <= 0]
    axa.plot(uf, (1 - uf) ** 2, color=ORANGE, lw=3.4, zorder=4)
    p_star_a = 1.0                            # min over u <= 0 of (1-u)^2

    # best supporting line, slope -lambda* = -2: tangent to G at (0, p*).  The
    # tangency at (0, p*) plus the shaded G already say "lowered until it
    # supports G"; an extra arrow only crowded the panel, so we let the geometry
    # and the caption carry that, keeping the picture uncluttered.
    line = lambda uu: 1 - 2 * uu
    us = np.array([-0.55, 1.32])
    axa.plot(us, line(us), color=GREEN, lw=2.0, zorder=3)
    # the dual value is the line's height at u = 0; here it equals p*.
    axa.plot(0, p_star_a, "o", color="black", ms=7, zorder=6)

    # labels, each in clear space.
    axa.text(0.16, p_star_a + 0.30, r"$d^\star=p^\star$", color="black",
             fontsize=11, ha="left", va="center")
    axa.text(-0.93, 1.62, r"feasible" "\n" r"slice $u\leq0$", color=ORANGE,
             fontsize=11, ha="center", va="center")
    # moved to sit entirely right of the y-axis, above the d*=p* label --
    # centered on the axis (x=-0.25) it straddled the vertical arrow, and at
    # y=2.0 (even fully right-anchored) the curve's left branch reaches out
    # to u=-0.41 there, closer to the axis than this text's left edge.
    axa.text(0.08, 1.6, r"$G=\{(g(x),f_0(x))\}$", color=BLUE, fontsize=11,
             ha="left", va="center")
    # slope tag on the green line, well down-right where it sits clearly alone
    # below both the curve and the x-axis label.
    axa.text(1.02, line(1.02) + 0.07, r"slope $-\lambda^\star$", color=GREEN,
             fontsize=11, ha="left", va="bottom", rotation=-31,
             rotation_mode="anchor")
    axis_labels(axa, xr_a, yr_a, u_label_dx=0.45)
    fl.clean_axes(axa, lim=(xr_a, yr_a), hide=True, equal=False)

    # =============== (b) non-convex: a duality gap =============== #
    xr_b, yr_b = (-0.92, 0.92), (-1.35, 0.72)
    u = np.linspace(-0.5, 0.5, 300)           # image of x in [0, 1]
    boundary_b = -(u + 0.5) ** 2
    axb.fill_between(u, boundary_b, yr_b[1], color=BLUE, alpha=0.09, lw=0,
                     zorder=1)
    axb.plot(u, boundary_b, color=BLUE, lw=2.0, zorder=3)
    uf = u[u <= 0]
    axb.plot(uf, -(uf + 0.5) ** 2, color=ORANGE, lw=3.4, zorder=4)
    p_star_b = -0.25                          # min over u <= 0: the dent
    d_star_b = -0.5                           # best dual value at u = 0

    # best supporting line, slope -lambda* = -1: touches both endpoints of the
    # boundary and passes *under* the dent, meeting the axis at d* < p*.
    line = lambda uu: -0.5 - uu
    us = np.array([-0.72, 0.78])
    axb.plot(us, line(us), color=GREEN, lw=2.0, zorder=3)
    axb.plot(0, p_star_b, "o", color="black", ms=7, zorder=6)
    axb.plot(0, d_star_b, "o", color=GREEN, ms=7, zorder=6)

    # the gap: a clean vertical double-arrow between p* and d*, running
    # exactly through u=0 -- same x as the p*/d* dots themselves.  (It used
    # to be offset to x=0.045, a hair right of the dots: legible on its own,
    # but next to the dots it visibly failed to line up with the two points
    # it's supposed to be measuring.)
    axb.annotate("", xy=(0.0, p_star_b - 0.012), xytext=(0.0, d_star_b + 0.012),
                 arrowprops=dict(arrowstyle="<->", color=GRAY, lw=1.4))
    # moved further left (was x=-0.06) -- at the midpoint height the green
    # supporting line sits at u in [-0.17,-0.08], which the old anchor's
    # span reached right into.
    axb.text(-0.22, 0.5 * (p_star_b + d_star_b), "duality gap", color=GRAY,
             fontsize=11, ha="right", va="center")
    axb.text(-0.045, p_star_b, r"$p^\star$", color="black", fontsize=11,
             ha="right", va="center")
    axb.text(-0.045, d_star_b, r"$d^\star$", color=GREEN, fontsize=11,
             ha="right", va="center")

    # raised above the green supporting line -- at the old height (0.16) the
    # line's leftmost value in this x-range (0.4 at u=-0.9) cut through it.
    axb.text(-0.85, 0.55, r"feasible slice $u\leq0$", color=ORANGE,
             fontsize=11, ha="left", va="center")
    # pushed further down and right along the curve, clear of both the
    # "duality gap" label and the "slope" annotation it used to crowd.
    axb.text(0.80, -1.18, r"dented $G$", color=BLUE, fontsize=11,
             ha="left", va="center")
    # placed *below* the green line (va='top') at u=0.15: above the line is
    # where the dented-G curve runs almost parallel and close alongside it,
    # so any label above the line touches blue; below it is clear.
    axb.text(0.15, line(0.15) - 0.05, r"slope $-\lambda^\star$", color=GREEN,
             fontsize=11, ha="left", va="top", rotation=-37,
             rotation_mode="anchor")
    axis_labels(axb, xr_b, yr_b)
    fl.clean_axes(axb, lim=(xr_b, yr_b), hide=True, equal=False)

    fig.subplots_adjust(wspace=0.16)
    fl.save(fig, "mdl-opt-primal-dual-gap")


def fig_water_filling():
    """The water-filling picture (sec_mdl-constrained-optimization-duality,
    Worked Duals): each channel is a basin whose floor sits at its noise level
    n_i; pouring in the power budget P fills the quiet channels to a common
    water level w, and channels whose floor is above the waterline stay dry.

    Honest numbers: uses the *same* noise floors and budget as the
    #constrained-water-filling cell (n = [0.1, 0.4, 0.8, 1.6, 2.5], P = 3)
    and finds the level by the same bisection, so the figure's w ~ 1.4333 and
    allocations match the cell's printout digit for digit.
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    noise = np.array([0.1, 0.4, 0.8, 1.6, 2.5])
    P = 3.0
    lo, hi = noise.min(), noise.min() + P
    for _ in range(60):                       # the cell's bisection, verbatim
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if np.maximum(0, mid - noise).sum() < P else (lo, mid)
    w = 0.5 * (lo + hi)                       # ~ 1.4333

    width, gap = 0.84, 0.16
    for i, n in enumerate(noise):
        x = i + gap / 2
        # basin floor: solid gray block up to the noise level n_i
        ax.add_patch(plt.Rectangle((x, 0), width, n, facecolor=GRAY,
                                   alpha=0.45, lw=0))
        ax.plot([x, x + width], [n, n], color=GRAY, lw=1.6)
        if n < w:                             # water fills up to the level
            ax.add_patch(plt.Rectangle((x, n), width, w - n, facecolor=BLUE,
                                       alpha=0.28, lw=0))
        else:                                 # floor above the waterline
            ax.text(i + 0.5, n + 0.13, "dry", color=GRAY, fontsize=11,
                    ha="center", va="bottom")
        ax.text(i + 0.5, -0.13, rf"$n_{{{i + 1}}}={n}$", color="black",
                fontsize=11, ha="center", va="top")

    # the common water level, dashed across the whole tank
    ax.plot([0.0, len(noise)], [w, w], color=ORANGE, lw=1.7, ls=(0, (5, 3)),
            zorder=4)
    ax.text(len(noise) + 0.08, w,
            r"water level $w=1/\mu\approx1.43$", color=ORANGE, fontsize=11,
            ha="left", va="center")

    # one allocation called out: p_2* = w - n_2 on the second basin
    xi = 1 + 0.5
    ax.annotate("", xy=(xi, w - 0.015), xytext=(xi, noise[1] + 0.015),
                arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.4))
    ax.text(xi + 0.14, 0.5 * (w + noise[1]), r"$p_2^\star=w-n_2$",
            color=BLUE, fontsize=11, ha="left", va="center")

    # baseline
    ax.plot([0.0, len(noise)], [0.0, 0.0], color="black", lw=1.2)
    ax.text(2.5, -0.42, "channels, floors at the noise levels $n_i$",
            color="black", fontsize=11, ha="center", va="top")

    fl.clean_axes(ax, lim=((-0.15, 7.3), (-0.75, 2.95)), hide=True,
                  equal=False)
    fl.save(fig, "mdl-opt-water-filling")


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
    # "0" and "0.25" sit only 0.25 apart -- at fontsize 11 center-aligned
    # labels collide ("00.25"), so nudge them apart instead.
    for v, lab, ha, dx in ((0, "0", "right", -0.02), (0.25, "0.25", "left", 0.02),
                           (1, "1", "center", 0.0), (2, "2", "center", 0.0),
                           (4, "4", "center", 0.0)):
        ax.text(v + dx, -0.24, lab, color="black", fontsize=11, ha=ha,
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
                xytext=(0.02, 0.74), color=GRAY, fontsize=11, ha="left",
                va="center",
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.9,
                                 relpos=(0, 0)))
    # xytext's x matches xy's x (0.25) -- with ha="left" and relpos=(0, 0)
    # (leader pinned to the text's own bottom-left corner), that makes the
    # leader run straight up to the smallest-normal tick instead of leaning
    # in from the left as it did anchored at x=0.07.
    ax.annotate("smallest normal", xy=(0.25, 0.155), xytext=(0.25, 0.48),
                color=GRAY, fontsize=11, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8,
                                 relpos=(0, 0)))

    # eps_mach: the gap from 1 to its right neighbor 1 + 2^-p.  The label sits
    # up and to the right of the little gap, joined by a short leader, so it
    # clears the "smallest normal" caption to its left entirely.
    eps = 2.0 ** -p
    ax.annotate("", xy=(1 + eps, 0.30), xytext=(1, 0.30),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.1))
    # pushed further right (was 1.55) -- at fontsize 11 "smallest normal"
    # (now starting at 0.16) reaches almost that far and the two touched.
    ax.annotate(r"$\varepsilon_{\mathrm{mach}}$", xy=(1 + eps / 2, 0.32),
                xytext=(1.95, 0.60), color="black", fontsize=11,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color="black", lw=0.7))

    # gap doubling: matched double-arrows under one gap of each binade.
    # Dropped from -0.36 to -0.46 -- at fontsize 11 the tick-number labels
    # (anchored at -0.24) now reach low enough to touch arrows at the old
    # height.
    for lo in (1.0, 2.0):
        ax.annotate("", xy=(lo + lo / 2**p, -0.46), xytext=(lo, -0.46),
                    arrowprops=dict(arrowstyle="<->", color=GRAY, lw=1.0))
    ax.text(2.05, -0.62, "gap doubles", color=GRAY, fontsize=11, ha="center",
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
        ax.text(x, 0.38, lab, color=ORANGE, fontsize=11, ha="center",
                va="bottom")
    ax.text(6.22, -0.30, r"overflow $\to\infty$", color=ORANGE, fontsize=11,
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
    # Rotated 90 degrees from the original layout: the slow/shallow axis
    # (small eigenvalue, long semi-axis) now runs horizontal instead of
    # vertical, so the panel is wide-and-short rather than narrow-and-tall.
    # figsize and the xlim/ylim extents below are the old ylim/xlim swapped.
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.4, 3.5))
    s2 = np.array([1.0, 1.0 / 16.0])          # sigma_1^2, sigma_n^2
    xlim, ylim = (-3.25, 3.25), (-2.2, 2.2)
    gx = np.linspace(*xlim, 300)
    gy = np.linspace(*ylim, 240)
    X, Y = np.meshgrid(gx, gy)
    # start point with its two components swapped to match the rotated axes
    x0 = np.array([2.8, -0.72])

    def panel(ax, ridge, title, steps, axis_labels, short_label_pos):
        h = s2 + ridge                        # h[0] = sigma_1^2 (steep),
                                               # h[1] = sigma_n^2 (shallow)
        # x gets the shallow eigenvalue (long semi-axis, horizontal), y gets
        # the steep one (short semi-axis, vertical) -- this is the 90-degree
        # rotation relative to the original h[0]-on-x/h[1]-on-y layout.
        hx, hy = h[1], h[0]
        Z = 0.5 * (hx * X**2 + hy * Y**2)
        # levels chosen so the x-semi-axes are evenly spaced on screen
        semis = np.array([0.55, 1.05, 1.55, 2.1, 2.65, 3.2])
        ax.contour(X, Y, Z, levels=list(0.5 * hx * semis**2),
                   colors=[LIGHT], linewidths=0.9)
        # principal axes of the ellipse with x-semi-axis 2.1, labelled with
        # the curvatures sigma^2 + lambda along each axis
        c = 0.5 * hx * 2.1**2
        ex, ey = 2.1, np.sqrt(2 * c / hy)
        fl.arrow(ax, (0, 0), (ex, 0), color=GREEN, lw=1.8)
        fl.arrow(ax, (0, 0), (0, ey), color=GREEN, lw=1.8)
        # The long horizontal axis (shallow eigenvalue): the GD path's own
        # oscillation (in y, along the *steep* axis after rotation) has an
        # envelope that shrinks in lockstep with x as the path converges, so
        # its amplitude near the arrow is not small -- a label tucked right
        # under the arrow at a fixed small offset (the old ex*0.58, -0.22)
        # sat inside that envelope for panel (a)'s long 28-step path.  Placed
        # well outside both the level-set ellipses and the path's envelope
        # instead, scaled by this panel's own ex/ey so it clears in both
        # panels despite their very different eccentricities.
        ax.text(ex * 0.85, -1.6 * ey, axis_labels[1], color=GREEN,
                fontsize=11, ha="center", va="top")
        # The short vertical axis (steep eigenvalue): a diagonal leader from
        # near the arrowhead out to open space crosses several level-set
        # ellipses on the way (they fill almost the whole panel), so this is
        # plain text with no leader, tucked into the one gap between two
        # ellipse rings just above the arrowhead -- found per panel since it
        # depends on this panel's own ey.
        ax.text(*short_label_pos, axis_labels[0], color=GREEN, fontsize=11,
                ha="left", va="center")
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
        ax.set_title(title)
        fl.clean_axes(ax, lim=(xlim, ylim), hide=True)

    kappa0 = s2[0] / s2[1]
    kappa1 = (s2[0] + 0.5) / (s2[1] + 0.5)
    panel(axa, 0.0, rf"(a) $\lambda=0$:  $\kappa={kappa0:.0f}$", 28,
          (r"$\sigma_1^2$", r"$\sigma_n^2$"), short_label_pos=(0.1, 0.92))
    panel(axb, 0.5, rf"(b) $\lambda=0.5$:  $\kappa\approx{kappa1:.1f}$", 12,
          (r"$\sigma_1^2+\lambda$", r"$\sigma_n^2+\lambda$"),
          short_label_pos=(0.9, 2.05))

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
