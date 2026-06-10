#!/usr/bin/env python3
"""Generate the illustrative figures for the "Dynamics: Differential Equations
and Generative Flows" chapter (chapter_mdl-dynamics) in the project's one house
style.

These are conceptual/schematic pictures -- a velocity field with integral
curves, an SDE path cloud, a probability-flux balance, the diffusion
forward/reverse process, the score as a vector field -- pre-generated as static
SVGs and referenced from the chapter .md with no drawing code (exactly like the
slide SVGs and the Linear-Algebra figures in ``gen_mdl_figures.py``).

The shared house style (palette, ``plt.rcParams``, ``save()`` with the fixed
hashsalt + dropped Date metadata, and the ``arrow``/``vlabel``/``clean_axes``/
``axis_cross`` helpers) is imported from ``gen_mdl_figures`` so this file never
duplicates -- or diverges from -- the one style.  Run with the repo's pytorch
venv:

    .venv-pytorch/bin/python tools/gen_mdl_dynamics_figures.py

All figures are written to ``img/mdl-dyn-<id>.svg``.  Every picture is computed
from real dynamics -- integrate the actual ODE, sample the actual SDE with a
seeded RNG, evaluate real Gaussian densities and analytic scores -- so the
figures teach by being exact, not sketched.  The script is idempotent:
re-running with the fixed seed and hashsalt overwrites byte-for-byte.
"""

from __future__ import annotations

import os
import sys

# Reuse the ONE shared style + helpers from the Linear-Algebra generator. This
# pins palette, rcParams (incl. the svg.hashsalt that makes runs reproducible),
# and the save()/arrow()/clean_axes()/axis_cross()/vlabel() helpers.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT


# =========================================================================== #
# Helpers local to this chapter                                               #
# =========================================================================== #

def _rk4(f, x0, t0, t1, n):
    """Integrate dx/dt = f(t, x) with classical RK4; return the (n+1, d) path."""
    h = (t1 - t0) / n
    x = np.asarray(x0, float).copy()
    out = [x.copy()]
    t = t0
    for _ in range(n):
        k1 = f(t, x)
        k2 = f(t + h / 2, x + h / 2 * k1)
        k3 = f(t + h / 2, x + h / 2 * k2)
        k4 = f(t + h, x + h * k3)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        out.append(x.copy())
    return np.array(out)


def _gauss(x, mu, var):
    """1-D Gaussian density evaluated on array ``x``."""
    return np.exp(-0.5 * (x - mu) ** 2 / var) / np.sqrt(2 * np.pi * var)


# =========================================================================== #
# §6.1.1  An ODE as a velocity field  ->  mdl-dyn-ode-field                    #
# =========================================================================== #

def fig_ode_field():
    """An ODE read as a velocity field: a 2-D quiver of f(x) with two integral
    curves threaded through it.  Field is the linear spiral-sink
    f(x) = A x, A = [[-0.5, -1], [1, -0.5]] (eigenvalues -0.5 +/- i): real
    trajectories spiral inward, computed by RK4 on the actual field."""
    A = np.array([[-0.5, -1.0], [1.0, -0.5]])

    def f(t, x):
        return A @ x

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    m = 2.2

    # velocity field on a grid, arrows scaled to a readable, uniform length
    g = np.linspace(-m, m, 15)
    X, Y = np.meshgrid(g, g)
    U = A[0, 0] * X + A[0, 1] * Y
    V = A[1, 0] * X + A[1, 1] * Y
    ax.quiver(X, Y, U, V, color=LIGHT, angles="xy", pivot="mid",
              width=0.004, headwidth=4, headlength=5, zorder=1)

    # two integral curves following the field from different starts
    starts = [np.array([2.0, 0.2]), np.array([-0.4, -2.0])]
    cols = [BLUE, ORANGE]
    for x0, c in zip(starts, cols):
        path = _rk4(f, x0, 0.0, 8.0, 1600)
        ax.plot(path[:, 0], path[:, 1], color=c, lw=2.2, zorder=3)
        ax.plot(*x0, "o", color=c, ms=7, zorder=4)
        # a couple of tangent arrowheads along the curve to show direction
        for frac in (0.10, 0.34):
            i = int(frac * len(path))
            fl.arrow(ax, path[i], path[i + 12], color=c, lw=0.0, mut=13)

    ax.plot(0, 0, "o", color=GREEN, ms=8, zorder=5)
    ax.text(0.12, 0.16, "fixed point", color=GREEN, fontsize=9.5, ha="left")
    ax.text(starts[0][0], starts[0][1] + 0.18, r"$\mathbf{x}_0$", color=BLUE,
            fontsize=10, ha="center", va="bottom")
    ax.text(starts[1][0] - 0.12, starts[1][1], r"$\mathbf{x}_0'$", color=ORANGE,
            fontsize=10, ha="right", va="center")
    ax.text(-m + 0.1, m - 0.1, r"$\dot{\mathbf{x}}=\mathbf{f}(\mathbf{x})$",
            color=GRAY, fontsize=11, ha="left", va="top")

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    fl.clean_axes(ax, lim=((-m, m), (-m, m)), equal=True)
    fl.save(fig, "mdl-dyn-ode-field")


# =========================================================================== #
# §6.2.5  SDE path cloud  ->  mdl-dyn-sde-paths                                #
# =========================================================================== #

def fig_sde_paths():
    """Many Euler-Maruyama sample paths of the OU SDE dX = -theta X dt + sigma dW
    from one fixed start: a spreading fan of jittery trajectories, the mean curve
    X0 e^{-theta t} highlighted, and the analytic +/- 2 std time-marginal envelope
    overlaid (the OU variance (sigma^2/2theta)(1-e^{-2theta t})."""
    rng = np.random.default_rng(0)
    theta, sigma, X0 = 1.0, 0.9, 2.0
    T, n = 4.0, 800
    dt = T / n
    t = np.linspace(0, T, n + 1)

    n_paths = 40
    X = np.full((n_paths, n + 1), X0, float)
    for k in range(n):
        xi = rng.standard_normal(n_paths)
        X[:, k + 1] = (X[:, k] - theta * X[:, k] * dt
                       + sigma * np.sqrt(dt) * xi)

    mean = X0 * np.exp(-theta * t)
    var = sigma ** 2 / (2 * theta) * (1 - np.exp(-2 * theta * t))
    std = np.sqrt(var)

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for k in range(n_paths):
        ax.plot(t, X[k], color=BLUE, lw=0.6, alpha=0.28, zorder=1)
    # +/- 2 std analytic marginal envelope
    ax.fill_between(t, mean - 2 * std, mean + 2 * std, color=ORANGE, alpha=0.16,
                    zorder=2, label=r"$\pm 2\,\mathrm{std}$ marginal")
    ax.plot(t, mean + 2 * std, color=ORANGE, lw=1.4, ls="--", zorder=3)
    ax.plot(t, mean - 2 * std, color=ORANGE, lw=1.4, ls="--", zorder=3)
    ax.plot(t, mean, color=GREEN, lw=2.6, zorder=4,
            label=r"mean $X_0 e^{-\theta t}$")
    ax.plot(0, X0, "o", color="black", ms=6, zorder=5)
    ax.text(0.04, X0 + 0.06, r"$X_0$", fontsize=10, ha="left", va="bottom")

    ax.set_xlabel("time $t$")
    ax.set_ylabel("$X_t$")
    ax.set_xlim(0, T)
    ax.legend(loc="upper right")
    ax.set_aspect("auto")
    fl.clean_axes(ax, equal=False)
    fl.save(fig, "mdl-dyn-sde-paths")


# =========================================================================== #
# §6.3.3  Continuity / Fokker-Planck flux balance  ->  mdl-dyn-fokker-planck-flux
# =========================================================================== #

def fig_fokker_planck_flux():
    """A 1-D density bump p(x) with the probability current j = f p arrows
    crossing the boundary of a control region [a, b], illustrating
    'rate of change of mass inside = net inward flux' (the continuity equation
    that underlies Fokker-Planck).  Uses a real Gaussian p and a constant drift
    f, so the current j(x) = f p(x) is exact."""
    mu, var, fdrift = 0.15, 0.55, 0.8
    a, b = -0.7, 1.0
    x = np.linspace(-3.0, 3.0, 600)
    p = _gauss(x, mu, var)
    j = fdrift * p  # probability current j = f * p (constant drift)

    fig, ax = plt.subplots(figsize=(6.8, 4.0))

    # the density bump
    ax.plot(x, p, color=BLUE, lw=2.4, zorder=4)
    ax.fill_between(x, 0, p, color=BLUE, alpha=0.10, zorder=1)
    # highlight the control region mass
    sel = (x >= a) & (x <= b)
    ax.fill_between(x[sel], 0, p[sel], color=ORANGE, alpha=0.22, zorder=2)

    pa, pb = _gauss(a, mu, var), _gauss(b, mu, var)
    ax.axvline(a, color=GRAY, lw=1.2, ls="--", zorder=3)
    ax.axvline(b, color=GRAY, lw=1.2, ls="--", zorder=3)

    # flux arrows: current f*p crossing each boundary (drift > 0 => rightward).
    # length proportional to the current magnitude at that boundary.
    scale = 0.55
    yarr = 0.40
    fl.arrow(ax, (a, yarr), (a + scale * (fdrift * pa), yarr),
             color=GREEN, lw=2.6, mut=15)
    fl.arrow(ax, (b, yarr), (b + scale * (fdrift * pb), yarr),
             color=GREEN, lw=2.6, mut=15)
    # labels placed to the OUTSIDE of each boundary so they clear the arrows
    ax.text(a - 0.12, yarr, r"$j(a)=f\,p(a)$", color=GREEN, fontsize=9.5,
            ha="right", va="center")
    ax.text(b + 0.40, yarr, r"$j(b)=f\,p(b)$", color=GREEN, fontsize=9.5,
            ha="left", va="center")

    # mark a, b as explicit ticks (no free-floating text colliding with the axis)
    ax.set_xticks([-3, -2, a, b, 2, 3])
    ax.set_xticklabels(["$-3$", "$-2$", "$a$", "$b$", "$2$", "$3$"])
    ax.text((a + b) / 2, 0.10,
            r"mass in $[a,b]$",
            color=ORANGE, fontsize=10, ha="center", va="center")
    ax.text(mu, 0.65,
            r"$\partial_t\!\!\int_a^b p\,dx = j(a)-j(b)$",
            color="black", fontsize=11, ha="center", va="center")

    ax.set_xlabel("$x$")
    ax.set_ylabel("$p(x)$")
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-0.02, 0.72)
    ax.set_aspect("auto")
    fl.clean_axes(ax, equal=False)
    fl.save(fig, "mdl-dyn-fokker-planck-flux")


# =========================================================================== #
# §6.3.6 / §6.4.3  Forward (data->noise) and reverse (noise->data) process     #
#                  ->  mdl-dyn-forward-reverse                                  #
# =========================================================================== #

def fig_forward_reverse():
    """Two rows of small 1-D density panels.  Top: the FORWARD noising process
    turns a structured (bimodal) data density into a single Gaussian across a
    few time slices, left->right.  Bottom: the REVERSE process recovers the
    structured density, right->left.  Each panel is the exact VP-SDE (OU)
    marginal of the bimodal mixture: a closed-form mixture of Gaussians, so the
    shapes are computed, not sketched."""
    # bimodal data mixture p_0 = 1/2 N(-1.6, .12) + 1/2 N(1.6, .12)
    mus0 = np.array([-1.6, 1.6])
    var0 = 0.12
    w = np.array([0.5, 0.5])

    # VP-SDE / OU forward: X_t = a_t X_0 + sqrt(1 - a_t^2) Z, a_t = e^{-t/2}.
    # Marginal of the mixture is a mixture: N(a_t mu_i, a_t^2 var0 + (1-a_t^2)).
    ts = np.array([0.0, 0.5, 1.2, 2.5, 6.0])
    x = np.linspace(-3.2, 3.2, 500)

    def marginal(tt):
        at = np.exp(-tt / 2)
        var_t = at ** 2 * var0 + (1 - at ** 2)
        dens = np.zeros_like(x)
        for wi, mi in zip(w, mus0):
            dens += wi * _gauss(x, at * mi, var_t)
        return dens

    n = len(ts)
    fig, axes = plt.subplots(2, n, figsize=(2.05 * n, 4.0), sharex=True,
                             sharey=True)

    # peak height of the data density, for a shared y-scale and arrow placement
    ymax = marginal(0.0).max() * 1.12

    def panel(ax, tt, color, fill):
        d = marginal(tt)
        ax.plot(x, d, color=color, lw=2.0)
        ax.fill_between(x, 0, d, color=color, alpha=fill)
        ax.set_xlim(-3.2, 3.2)
        ax.set_ylim(0, ymax)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(LIGHT)

    # top row: forward, data (left) -> noise (right)
    for j, tt in enumerate(ts):
        c = BLUE if j == 0 else (GRAY if j < n - 1 else ORANGE)
        panel(axes[0, j], tt, c, 0.18 if 0 < j < n - 1 else 0.30)
    # bottom row: reverse, noise (left) -> data (right): same densities reversed
    for j, tt in enumerate(ts[::-1]):
        jj = n - 1 - j
        c = ORANGE if j == 0 else (GRAY if j < n - 1 else BLUE)
        panel(axes[1, j], tt, c, 0.18 if 0 < j < n - 1 else 0.30)

    axes[0, 0].set_title(r"data $p_0$", fontsize=10, color=BLUE)
    axes[0, n - 1].set_title(r"noise $p_T$", fontsize=10, color=ORANGE)
    axes[1, 0].set_title(r"noise $p_T$", fontsize=10, color=ORANGE)
    axes[1, n - 1].set_title(r"data $p_0$", fontsize=10, color=BLUE)

    # process-direction labels at the row margins
    axes[0, 0].text(-0.34, 0.5, "forward\n(noising)", transform=axes[0, 0].transAxes,
                    rotation=90, va="center", ha="center", fontsize=10, color=GRAY)
    axes[1, 0].text(-0.34, 0.5, "reverse\n(score)", transform=axes[1, 0].transAxes,
                    rotation=90, va="center", ha="center", fontsize=10, color=GREEN)

    fig.subplots_adjust(wspace=0.12, hspace=0.45, left=0.10, right=0.985,
                        top=0.90, bottom=0.06)
    # big direction arrows spanning each row
    fig.text(0.55, 0.955, r"$\longrightarrow$ add noise $\longrightarrow$",
             ha="center", fontsize=11, color=GRAY)
    fig.text(0.55, 0.025, r"$\longleftarrow$ denoise with $\nabla\log p_t$ $\longleftarrow$",
             ha="center", fontsize=11, color=GREEN)

    fl.save(fig, "mdl-dyn-forward-reverse")


# =========================================================================== #
# §6.3.5 / §6.4.1  The score as a vector field  ->  mdl-dyn-score-field        #
# =========================================================================== #

def fig_score_field():
    """The score s(x) = grad log p(x) of a 2-D two-mode Gaussian mixture, drawn
    as a quiver over the density contours.  Arrows point UP the density (toward
    the nearest mode) and vanish at the modes; computed in closed form from the
    mixture, not sketched.  s = sum_i r_i(x) * (mu_i - x)/var_i with responsibilities
    r_i, exactly grad log p for a Gaussian mixture with isotropic covariances."""
    mus = np.array([[-1.3, -0.6], [1.3, 0.6]])
    var = 0.7
    w = np.array([0.5, 0.5])

    def comp_density(P, mu):
        d2 = np.sum((P - mu) ** 2, axis=-1)
        return np.exp(-0.5 * d2 / var) / (2 * np.pi * var)

    def density(P):
        return sum(wi * comp_density(P, mu) for wi, mu in zip(w, mus))

    def score(P):
        # grad log p = (sum_i w_i N_i * (mu_i - x)/var) / (sum_i w_i N_i)
        comps = np.stack([wi * comp_density(P, mu) for wi, mu in zip(w, mus)],
                         axis=0)
        tot = comps.sum(axis=0)
        num = np.zeros(P.shape)
        for i, mu in enumerate(mus):
            num += comps[i][..., None] * (mu - P) / var
        return num / tot[..., None]

    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    m = 3.2

    # filled density contours
    gg = np.linspace(-m, m, 240)
    GX, GY = np.meshgrid(gg, gg)
    grid = np.stack([GX, GY], axis=-1)
    Z = density(grid)
    ax.contourf(GX, GY, Z, levels=10, cmap="Blues", alpha=0.85, zorder=1)
    ax.contour(GX, GY, Z, levels=6, colors=[LIGHT], linewidths=0.7, zorder=2)

    # score quiver on a coarser grid
    q = np.linspace(-m + 0.3, m - 0.3, 16)
    QX, QY = np.meshgrid(q, q)
    Pq = np.stack([QX, QY], axis=-1)
    S = score(Pq)
    # clip very long arrows in the low-density tails for readability
    mag = np.sqrt(S[..., 0] ** 2 + S[..., 1] ** 2)
    cap = 4.0
    fac = np.where(mag > cap, cap / mag, 1.0)
    U, Vc = S[..., 0] * fac, S[..., 1] * fac
    ax.quiver(QX, QY, U, Vc, color=GRAY, angles="xy", pivot="mid",
              width=0.004, headwidth=4, headlength=5, alpha=0.9, zorder=3)

    # mark the two modes (score = 0 there)
    for mu in mus:
        ax.plot(mu[0], mu[1], "o", color=ORANGE, ms=8, zorder=5)
    # one annotation, in open low-density space to the lower-right of a mode
    ax.annotate(r"$\nabla\log p=\mathbf{0}$", xy=tuple(mus[1]),
                xytext=(2.3, -1.7), color=ORANGE, fontsize=9.5,
                ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2,
                                shrinkA=2, shrinkB=6), zorder=6)
    ax.text(-m + 0.15, m - 0.15, r"$\mathbf{s}(\mathbf{x})=\nabla\log p(\mathbf{x})$",
            color="black", fontsize=11, ha="left", va="top")

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    fl.clean_axes(ax, lim=((-m, m), (-m, m)), equal=True)
    fl.save(fig, "mdl-dyn-score-field")


# =========================================================================== #
# §27.1  Uniqueness counterexample  ->  mdl-dyn-uniqueness-fan                 #
# =========================================================================== #

def fig_uniqueness_fan():
    """The fan of solutions of dx/dt = sqrt(|x|) through x0 = 0: the rest
    solution x = 0 plus one parabolic blade x_c(t) = (t - c)^2 / 4 per
    departure time c.  Every blade is the exact closed-form solution, so the
    failure of uniqueness (the field is not Lipschitz at 0) is drawn, not
    sketched."""
    tmax = 3.2
    t = np.linspace(0.0, tmax, 400)

    fig, ax = plt.subplots(figsize=(6.2, 3.9))

    # the rest solution x(t) = 0
    ax.plot([0, tmax], [0, 0], color=BLUE, lw=2.6, zorder=4)
    ax.text(2.45, -0.12, r"$x(t)\equiv 0$", color=BLUE, fontsize=10,
            ha="left", va="top")

    # one blade per departure time c (exact solution of the IVP)
    for c in (0.0, 0.5, 1.0, 1.5, 2.0):
        x = np.where(t > c, (t - c) ** 2 / 4.0, 0.0)
        ax.plot(t, x, color=ORANGE, lw=1.8, zorder=3)
        ax.plot(c, 0, "o", color=ORANGE, ms=4, zorder=5)
        ax.text(tmax + 0.06, (tmax - c) ** 2 / 4.0, f"$c={c:g}$",
                color=ORANGE, fontsize=8.5, ha="left", va="center")

    ax.text(1.78, 1.55, r"$x_c(t)=(t-c)^2/4$", color=ORANGE,
            fontsize=10.5, ha="right", va="center")
    ax.annotate("Lipschitz fails at $x=0$:\nno unique way to leave",
                xy=(0.03, 0.01), xytext=(0.55, 0.85), fontsize=9.5,
                color=GRAY, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.1,
                                shrinkA=22, shrinkB=3))

    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")
    ax.set_xlim(-0.06, 3.75)
    ax.set_ylim(-0.30, 2.75)
    ax.set_aspect("auto")
    fl.clean_axes(ax, equal=False)
    fl.save(fig, "mdl-dyn-uniqueness-fan")


# =========================================================================== #
# §27.1  Stability dictionary  ->  mdl-dyn-phase-portraits                     #
# =========================================================================== #

def fig_phase_portraits():
    """The 2-D stability dictionary as a 2x3 portrait gallery: stable node,
    unstable node, saddle / stable spiral, unstable spiral, center.  Each panel
    is a real linear field A x (direction grid) with RK4-integrated
    trajectories threaded through it and its eigenvalue signature in the
    title -- the table of the text, drawn."""
    cases = [
        ("stable node", np.array([[-1.5, 0.5], [0.5, -1.5]]),
         r"$\lambda_1,\lambda_2<0$",
         [(2.0, 1.2), (-2.0, 0.5), (0.5, -2.0)], 3.0),
        ("unstable node", np.array([[1.5, 0.5], [0.5, 1.5]]),
         r"$\lambda_1,\lambda_2>0$",
         [(0.10, 0.06), (-0.10, 0.03), (0.03, -0.10)], 2.2),
        ("saddle", np.array([[0.5, 1.2], [1.2, 0.5]]),
         r"$\lambda_1>0>\lambda_2$",
         [(-2.0, 1.8), (2.0, -1.8), (-1.0, -2.0), (1.0, 2.0)], 1.8),
        ("stable spiral", np.array([[-0.5, -1.0], [1.0, -0.5]]),
         r"$\lambda=a\pm ib,\ a<0$",
         [(2.0, 0.2), (-1.5, -1.5)], 7.0),
        ("unstable spiral", np.array([[0.25, -1.0], [1.0, 0.25]]),
         r"$\lambda=a\pm ib,\ a>0$",
         [(0.15, 0.0), (-0.15, 0.0)], 10.5),
        ("center", np.array([[0.0, -1.0], [1.0, 0.0]]),
         r"$\lambda=\pm ib$",
         [(0.6, 0.0), (1.2, 0.0), (1.8, 0.0)], 2 * np.pi),
    ]
    m = 2.0
    g = np.linspace(-m, m, 11)
    X, Y = np.meshgrid(g, g)
    cols = [BLUE, ORANGE, GREEN, BLUE]

    fig, axes = plt.subplots(2, 3, figsize=(8.6, 6.0))
    for ax, (name, A, sig, starts, T) in zip(axes.ravel(), cases):
        # direction grid, normalized to uniform length for readability
        U = A[0, 0] * X + A[0, 1] * Y
        V = A[1, 0] * X + A[1, 1] * Y
        N = np.hypot(U, V)
        N[N == 0] = 1.0
        ax.quiver(X, Y, U / N, V / N, color=LIGHT, angles="xy", pivot="mid",
                  width=0.006, headwidth=4, headlength=5, scale=16, zorder=1)

        def f(t, x, A=A):
            return A @ x

        for x0, c in zip(starts, cols):
            path = _rk4(f, np.array(x0, float), 0.0, T, 600)
            ax.plot(path[:, 0], path[:, 1], color=c, lw=1.9, zorder=3)
            # a direction arrowhead mid-path, only where it is inside the box
            for frac in (0.30, 0.62):
                i = int(frac * len(path))
                p, q = path[i], path[i + 6]
                if np.all(np.abs(q) < 0.94 * m):
                    fl.arrow(ax, p, q, color=c, lw=0.0, mut=11)

        ax.plot(0, 0, "o", color="black", ms=4, zorder=5)
        ax.set_title(f"{name}\n{sig}", fontsize=9.5)
        ax.set_xticks([])
        ax.set_yticks([])
        fl.clean_axes(ax, lim=((-m, m), (-m, m)), equal=True)

    fig.subplots_adjust(wspace=0.18, hspace=0.32, left=0.04, right=0.985,
                        top=0.91, bottom=0.03)
    fl.save(fig, "mdl-dyn-phase-portraits")


# =========================================================================== #
# §27.1  ResNet block = Euler step  ->  mdl-dyn-resnet-as-euler                #
# =========================================================================== #

def fig_resnet_as_euler():
    """Left: a stack of residual blocks x_{l+1} = x_l + f_theta(x_l) --
    identity skip plus learned branch.  Right: the same update read as
    numerical integration: three forward-Euler steps of size h = 1 (computed
    with the literal update) overlaid on the exact flow (RK4) of the same
    rotation field."""
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(8.4, 4.8), gridspec_kw=dict(width_ratios=[1.0, 1.35]))

    # -- left: the residual stack ------------------------------------------ #
    sx, bx = 1.1, 2.9                  # skip spine / branch column
    ys = [0.4, 3.0, 5.6, 8.2]          # state levels x_0 .. x_3
    axL.set_xlim(-0.6, 4.4)
    axL.set_ylim(-0.3, 8.9)
    axL.set_aspect("equal")
    axL.axis("off")
    for l in range(3):
        y0, y1 = ys[l], ys[l + 1]
        ybr = y0 + 0.45                # branch leaves the spine here
        ypl = y1 - 0.75                # the addition node
        by = (ybr + ypl) / 2.0         # branch box center height
        # identity skip straight up into the + node
        axL.plot([sx, sx], [y0, ypl - 0.45], color=GRAY, lw=1.6, zorder=2)
        fl.arrow(axL, (sx, ypl - 0.45), (sx, ypl - 0.26), color=GRAY,
                 lw=1.6, mut=11)
        # the + node and the step up to the next state
        axL.add_patch(plt.Circle((sx, ypl), 0.26, fill=False, color="black",
                                 lw=1.2, zorder=3))
        axL.text(sx, ypl, "+", fontsize=12, ha="center", va="center")
        fl.arrow(axL, (sx, ypl + 0.26), (sx, y1 - 0.08), color=GRAY, lw=1.6,
                 mut=11)
        # residual branch: out, through f_theta, back into the + node
        axL.plot([sx, bx], [ybr, ybr], color=ORANGE, lw=1.6, zorder=2)
        axL.plot([bx, bx], [ybr, by - 0.38], color=ORANGE, lw=1.6, zorder=2)
        axL.add_patch(fl.Rectangle((bx - 0.75, by - 0.38), 1.5, 0.76,
                                   facecolor="white", edgecolor=ORANGE,
                                   lw=1.5, zorder=3))
        axL.text(bx, by, r"$\mathbf{f}_\theta$", color=ORANGE, fontsize=11,
                 ha="center", va="center", zorder=4)
        axL.plot([bx, bx], [by + 0.38, ypl], color=ORANGE, lw=1.6, zorder=2)
        fl.arrow(axL, (bx, ypl), (sx + 0.26, ypl), color=ORANGE, lw=1.6,
                 mut=11)
        # state dot + label
        axL.plot(sx, y0, "o", color=BLUE, ms=6, zorder=5)
        axL.text(sx - 0.34, y0, rf"$\mathbf{{x}}_{l}$", color=BLUE,
                 fontsize=11, ha="right", va="center")
    axL.plot(sx, ys[3], "o", color=BLUE, ms=6, zorder=5)
    axL.text(sx - 0.34, ys[3], r"$\mathbf{x}_3$", color=BLUE, fontsize=11,
             ha="right", va="center")
    axL.text(sx - 0.16, (ys[0] + ys[1]) / 2.0, "identity skip", color=GRAY,
             fontsize=8.5, ha="right", va="center", rotation=90)

    # -- right: Euler with h = 1 vs the exact flow -------------------------- #
    A = np.array([[0.0, -0.5], [0.5, 0.0]])    # pure rotation field

    def f(t, x):
        return A @ x

    mR = 3.4
    g = np.linspace(-mR + 0.2, mR - 0.2, 12)
    X, Y = np.meshgrid(g, g)
    axR.quiver(X, Y, A[0, 0] * X + A[0, 1] * Y, A[1, 0] * X + A[1, 1] * Y,
               color=LIGHT, angles="xy", pivot="mid", width=0.004,
               headwidth=4, headlength=5, zorder=1)

    x0 = np.array([2.0, -1.05])
    smooth = _rk4(f, x0, 0.0, 3.0, 600)
    axR.plot(smooth[:, 0], smooth[:, 1], color=BLUE, lw=2.2, zorder=3)
    i = 320
    fl.arrow(axR, smooth[i], smooth[i + 6], color=BLUE, lw=0.0, mut=13)

    E = [x0]                                   # three Euler steps, h = 1
    for _ in range(3):
        E.append(E[-1] + A @ E[-1])
    E = np.array(E)
    axR.plot(E[:, 0], E[:, 1], color=ORANGE, lw=1.8, marker="o", ms=5,
             zorder=4)
    axR.plot(*x0, "o", color="black", ms=6, zorder=5)
    for k in range(len(E)):
        p = E[k]
        off = p / np.hypot(*p) * 0.30
        axR.text(p[0] + off[0], p[1] + off[1], rf"$\mathbf{{x}}_{k}$",
                 color="black" if k == 0 else ORANGE, fontsize=10,
                 ha="center", va="center")
    axR.text(0.75, -1.7, "exact flow", color=BLUE, fontsize=10,
             ha="center", va="center")
    axR.text(-0.9, 3.0, "Euler, $h=1$", color=ORANGE, fontsize=10,
             ha="center", va="center")

    axR.set_xticks([])
    axR.set_yticks([])
    fl.clean_axes(axR, lim=((-mR, mR), (-mR, mR)), equal=True)

    fig.subplots_adjust(left=0.02, right=0.99, top=0.97, bottom=0.13,
                        wspace=0.10)
    fig.text(0.5, 0.015,
             r"$\mathbf{x}_{l+1}=\mathbf{x}_l+\mathbf{f}_\theta(\mathbf{x}_l)"
             r"\quad\Leftrightarrow\quad"
             r"\mathbf{x}_{n+1}=\mathbf{x}_n+h\,\mathbf{f}(\mathbf{x}_n)"
             r"\ \ \mathrm{with}\ h=1$",
             ha="center", fontsize=11)
    fl.save(fig, "mdl-dyn-resnet-as-euler")


# =========================================================================== #
# §27.2  Random walk -> Brownian motion  ->  mdl-dyn-brownian-paths            #
# =========================================================================== #

def fig_brownian_paths():
    """One coin-flip walk with steps +-sqrt(dt) at dt = 0.001 (right panel),
    viewed on meshes coarsened x10 and x100 (middle, left) -- a consistent
    coarsening of the same randomness.  Each view is itself a random walk with
    step variance dt; refining the mesh fills in detail without changing the
    large-scale path, and the spread tracks the sqrt(t) envelopes."""
    rng = np.random.default_rng(11)
    n = 1000
    coins = rng.integers(0, 2, n) * 2 - 1
    Wf = np.concatenate([[0.0], np.cumsum(coins * np.sqrt(1.0 / n))])
    tf = np.linspace(0.0, 1.0, n + 1)

    fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.1), sharey=True)
    for ax, stride in zip(axes, (100, 10, 1)):
        dt = stride / n
        # +-sqrt(t) and +-2 sqrt(t) envelopes
        for s, ls in ((1.0, ":"), (2.0, "--")):
            ax.plot(tf, s * np.sqrt(tf), color=GRAY, lw=1.0, ls=ls, zorder=2)
            ax.plot(tf, -s * np.sqrt(tf), color=GRAY, lw=1.0, ls=ls, zorder=2)
        if stride > 1:                          # the fine limit path, faint
            ax.plot(tf, Wf, color=LIGHT, lw=0.9, zorder=1)
        ax.plot(tf[::stride], Wf[::stride], color=BLUE,
                lw=1.6 if stride > 1 else 1.1,
                marker="o" if stride == 100 else None, ms=3.5, zorder=3)
        ax.set_title(rf"$\Delta t = {dt:g}$  ({n // stride} steps)",
                     fontsize=10)
        ax.set_xlabel("$t$")
        ax.set_xlim(0, 1)
        ax.set_aspect("auto")
        fl.clean_axes(ax, equal=False)
    axes[0].set_ylabel("$W_t$")
    axes[0].set_ylim(-2.35, 2.35)
    axes[2].text(0.83, 2 * np.sqrt(0.83) + 0.10, r"$\pm 2\sqrt{t}$",
                 color=GRAY, fontsize=8.5, ha="center", va="bottom")
    axes[2].text(0.83, np.sqrt(0.83) + 0.10, r"$\pm\sqrt{t}$",
                 color=GRAY, fontsize=8.5, ha="center", va="bottom")
    fig.subplots_adjust(wspace=0.10, left=0.07, right=0.99, top=0.88,
                        bottom=0.17)
    fl.save(fig, "mdl-dyn-brownian-paths")


# =========================================================================== #
# §27.2  Quadratic variation  ->  mdl-dyn-qv-convergence                       #
# =========================================================================== #

def fig_qv_convergence():
    """Running partial sums of squared increments along ONE fixed Brownian
    path, on meshes of n = 16 / 256 / 4096 intervals: they converge to the
    line y = t (quadratic variation).  The same sums for the smooth path
    sin(2 pi t) (gray, same meshes) collapse to zero -- the gap that is
    stochastic calculus."""
    rng = np.random.default_rng(7)
    n_fine = 4096
    dW = np.sqrt(1.0 / n_fine) * rng.standard_normal(n_fine)
    W = np.concatenate([[0.0], np.cumsum(dW)])

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot([0, 1], [0, 1], color="black", lw=1.3, ls="--", zorder=2,
            label="$y=t$")
    for n, a in ((16, 0.32), (256, 0.62), (4096, 1.0)):
        stride = n_fine // n
        tn = np.linspace(0.0, 1.0, n + 1)
        qv = np.concatenate([[0.0], np.cumsum(np.diff(W[::stride]) ** 2)])
        ax.plot(tn, qv, color=BLUE, alpha=a, lw=1.7, zorder=3,
                label=rf"$\sum(\Delta W_i)^2$, $n={n}$")
        sm = np.sin(2 * np.pi * tn)
        qs = np.concatenate([[0.0], np.cumsum(np.diff(sm) ** 2)])
        ax.plot(tn, qs, color=GRAY, alpha=a, lw=1.5, zorder=3,
                label=r"$\sum(\Delta x_i)^2$, $x=\sin 2\pi t$"
                if n == 16 else None)

    ax.text(1.01, 1.0, r"$\rightarrow t$", color=BLUE, fontsize=10,
            ha="left", va="center")
    ax.text(1.01, 0.0, r"$\rightarrow 0$", color=GRAY, fontsize=10,
            ha="left", va="center")
    ax.set_xlabel("$t$")
    ax.set_ylabel("running sum of squared increments")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper left")
    ax.set_aspect("auto")
    fl.clean_axes(ax, equal=False)
    fl.save(fig, "mdl-dyn-qv-convergence")


# =========================================================================== #
# §27.2  OU mean reversion  ->  mdl-dyn-ou-mean-reversion                      #
# =========================================================================== #

def fig_ou_mean_reversion():
    """Mean reversion in one picture: the OU drift -theta x as a direction
    field in the (t, x) plane, Euler-Maruyama sample paths from three
    different starts relaxing to 0, the saturating +-2 sigma_t spread band,
    and the stationary Gaussian N(0, sigma^2/2 theta) drawn sideways on the
    right edge."""
    rng = np.random.default_rng(5)
    theta, sigma, T, n = 1.0, 0.9, 4.0, 800
    dt = T / n
    t = np.linspace(0, T, n + 1)

    fig, (ax, axd) = plt.subplots(
        1, 2, figsize=(7.4, 3.9), sharey=True,
        gridspec_kw=dict(width_ratios=[4.0, 1.0], wspace=0.05))

    # drift direction field (1, -theta x), normalized
    tg, xg = np.meshgrid(np.arange(0.25, T, 0.5), np.arange(-2.6, 2.7, 0.65))
    U, V = np.ones_like(xg), -theta * xg
    N = np.hypot(U, V)
    ax.quiver(tg, xg, U / N, V / N, color=LIGHT, angles="xy", pivot="mid",
              width=0.005, headwidth=4, headlength=5, scale=26, zorder=1)

    # saturating +-2 sigma_t spread band (about the relaxed mean 0)
    std = np.sqrt(sigma ** 2 / (2 * theta) * (1 - np.exp(-2 * theta * t)))
    ax.fill_between(t, -2 * std, 2 * std, color=ORANGE, alpha=0.15, zorder=2)
    ax.plot(t, 2 * std, color=ORANGE, lw=1.3, ls="--", zorder=2)
    ax.plot(t, -2 * std, color=ORANGE, lw=1.3, ls="--", zorder=2)
    ax.text(2.6, 1.42, r"$\pm 2\sigma_t$ (saturates)", color=ORANGE,
            fontsize=9, ha="left", va="bottom")

    # Euler-Maruyama sample paths from three different starts
    for x0 in (2.5, -2.0, 0.8):
        X = np.empty(n + 1)
        X[0] = x0
        xi = rng.standard_normal(n)
        for k in range(n):
            X[k + 1] = X[k] - theta * X[k] * dt + sigma * np.sqrt(dt) * xi[k]
        ax.plot(t, X, color=BLUE, lw=1.2, alpha=0.9, zorder=3)
        ax.plot(0, x0, "o", color=BLUE, ms=5, zorder=4)

    ax.set_xlabel("time $t$")
    ax.set_ylabel("$x$")
    ax.set_xlim(0, T)
    ax.set_ylim(-3.0, 3.0)
    ax.set_aspect("auto")
    fl.clean_axes(ax, equal=False)

    # the stationary density, sideways, on the right edge
    var_inf = sigma ** 2 / (2 * theta)
    xs = np.linspace(-3.0, 3.0, 300)
    dens = _gauss(xs, 0.0, var_inf)
    axd.fill_betweenx(xs, 0, dens, color=GREEN, alpha=0.25, zorder=1)
    axd.plot(dens, xs, color=GREEN, lw=2.0, zorder=2)
    axd.set_xlim(0, dens.max() * 1.25)
    axd.set_xticks([])
    for sp in ("top", "right", "bottom"):
        axd.spines[sp].set_visible(False)
    axd.spines["left"].set_color(LIGHT)
    axd.set_title("stationary", fontsize=9.5, color=GREEN)
    axd.text(dens.max() * 0.62, -2.45,
             r"$\mathcal{N}\!\left(0,\frac{\sigma^2}{2\theta}\right)$",
             color=GREEN, fontsize=10, ha="center", va="center")
    fl.save(fig, "mdl-dyn-ou-mean-reversion")


# =========================================================================== #
# §27.4  Forward noising / reverse denoising of a cloud                        #
#        ->  mdl-dyn-noising-denoising                                         #
# =========================================================================== #

def _two_moons(n, rng, noise=0.06):
    """A seeded two-moons point cloud, centered and scaled to unit RMS."""
    k = n // 2
    a1 = np.pi * rng.random(k)
    a2 = np.pi * rng.random(n - k)
    X = np.concatenate([
        np.stack([np.cos(a1), np.sin(a1)], axis=1),
        np.stack([1.0 - np.cos(a2), 0.5 - np.sin(a2)], axis=1)])
    X = X + noise * rng.standard_normal(X.shape)
    X = X - X.mean(axis=0)
    return X / X.std()


def fig_noising_denoising():
    """A two-moons cloud under the VP (OU) forward process at t = 0 / 0.7 / T,
    columns aligned by t.  Top row: forward noising, read left to right.
    Bottom row: the reverse process traverses the same marginals right to
    left, with short arrows showing the EXACT score of the noised empirical
    mixture (computed, not sketched) pointing back toward the data."""
    rng = np.random.default_rng(0)
    n = 360
    X0 = _two_moons(n, rng)
    eps_f = rng.standard_normal((n, 2))      # forward-row noise draw
    eps_r = rng.standard_normal((n, 2))      # independent reverse-row draw
    ts = (0.0, 0.7, 3.0)                     # VP clock, a_t = e^{-t}
    cols = (BLUE, GRAY, ORANGE)

    def noised(eps, tt):
        a = np.exp(-tt)
        return a * X0 + np.sqrt(1.0 - a * a) * eps

    def score(P, tt):
        # exact score of the VP marginal of the empirical data distribution:
        # a mixture of n Gaussians N(a x_i, (1 - a^2) I)
        a = np.exp(-tt)
        var = 1.0 - a * a
        D = P[:, None, :] - a * X0[None, :, :]          # (m, n, 2)
        logw = -np.sum(D ** 2, axis=2) / (2 * var)
        w = np.exp(logw - logw.max(axis=1, keepdims=True))
        w = w / w.sum(axis=1, keepdims=True)
        return -(w[..., None] * D).sum(axis=1) / var

    m = 3.1
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 5.3), sharex=True,
                             sharey=True)
    for j, tt in enumerate(ts):
        for i, eps in enumerate((eps_f, eps_r)):
            ax = axes[i, j]
            P = X0 if tt == 0.0 else noised(eps, tt)
            ax.scatter(P[:, 0], P[:, 1], s=6, color=cols[j], alpha=0.6,
                       linewidths=0, zorder=2)
            ax.set_xlim(-m, m)
            ax.set_ylim(-m, m)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
            ax.spines["bottom"].set_color(LIGHT)
            if i == 1 and tt > 0.0:          # score arrows on the noisy panels
                Q = P[::9]
                S = score(Q, tt)
                D = S / np.hypot(S[:, 0], S[:, 1])[:, None] * 0.5
                ax.quiver(Q[:, 0], Q[:, 1], D[:, 0], D[:, 1], color=GREEN,
                          angles="xy", scale_units="xy", scale=1.0,
                          width=0.006, zorder=3)

    axes[0, 0].set_title("$t=0$ (data)", fontsize=10, color=BLUE)
    axes[0, 1].set_title("$t=0.7$", fontsize=10, color=GRAY)
    axes[0, 2].set_title("$t=T$ (noise)", fontsize=10, color=ORANGE)
    axes[0, 0].text(-0.22, 0.5, "forward\n(noising)",
                    transform=axes[0, 0].transAxes, rotation=90,
                    va="center", ha="center", fontsize=10, color=GRAY)
    axes[1, 0].text(-0.22, 0.5, "reverse\n(score)",
                    transform=axes[1, 0].transAxes, rotation=90,
                    va="center", ha="center", fontsize=10, color=GREEN)

    fig.subplots_adjust(wspace=0.08, hspace=0.10, left=0.10, right=0.985,
                        top=0.89, bottom=0.075)
    fig.text(0.545, 0.965, r"$\longrightarrow$ add noise $\longrightarrow$",
             ha="center", fontsize=11, color=GRAY)
    fig.text(0.545, 0.022,
             r"$\longleftarrow$ denoise with $\nabla\log p_t$ $\longleftarrow$",
             ha="center", fontsize=11, color=GREEN)
    fl.save(fig, "mdl-dyn-noising-denoising")


# =========================================================================== #
# §27.4  Straight conditional vs curved marginal paths  ->  mdl-dyn-fm-paths   #
# =========================================================================== #

def fig_fm_paths():
    """Straight conditional flow-matching segments source -> target (two of
    them visibly crossing) against the curved trajectories of the MARGINAL
    velocity field they induce.  The marginal field is the exact posterior
    mean for Gaussian-smoothed segments, and the blue trajectories integrate
    it with RK4 -- they bend at the crossings and never intersect."""
    pairs = ((-1.2, 1.5), (0.0, -1.5), (1.2, 0.2))   # (x_0, x_1) couplings
    sig = 0.15

    def u(x, tt):
        # marginal velocity: posterior-mean of the conditional velocities
        # under p_t(x | z) = N((1-t) x0 + t x1, sig^2)
        logw = np.stack([-(x - ((1 - tt) * x0 + tt * x1)) ** 2
                         / (2 * sig ** 2) for x0, x1 in pairs])
        w = np.exp(logw - logw.max(axis=0))
        w = w / w.sum(axis=0)
        return sum(wi * (x1 - x0) for wi, (x0, x1) in zip(w, pairs))

    starts = np.array([-1.35, -1.05, -0.15, 0.15, 1.05, 1.35])
    paths = _rk4(lambda tt, x: u(x, tt), starts, 0.0, 1.0, 400)
    tgrid = np.linspace(0.0, 1.0, 401)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    # straight conditional segments
    for x0, x1 in pairs:
        ax.plot([0, 1], [x0, x1], color=ORANGE, lw=1.8, ls="--", zorder=2)
        ax.plot(0, x0, "o", color=GRAY, ms=6, zorder=4)
        ax.plot(1, x1, "s", color="black", ms=6, zorder=4)

    # mark the interior crossings of the segments
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            (a0, a1), (b0, b1) = pairs[i], pairs[j]
            den = (a1 - a0) - (b1 - b0)
            if den == 0:
                continue
            tc = (b0 - a0) / den
            if 0.02 < tc < 0.98:
                xc = a0 + (a1 - a0) * tc
                ax.plot(tc, xc, "x", color="black", ms=9, mew=2.0, zorder=5)

    # curved marginal-flow trajectories
    for k in range(len(starts)):
        ax.plot(tgrid, paths[:, k], color=BLUE, lw=1.6, zorder=3)

    ax.text(0.03, 1.85, "conditional paths: straight, may cross",
            color=ORANGE, fontsize=9.5, ha="left", va="center")
    ax.text(0.97, -1.85, "marginal flow: curved, never crosses",
            color=BLUE, fontsize=9.5, ha="right", va="center")
    ax.annotate("crossing: the posterior\nmean averages velocities",
                xy=(1.2 / 4.2, -1.2 + 2.7 * 1.2 / 4.2), fontsize=8.5,
                xytext=(0.40, -1.05), color=GRAY, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.0,
                                shrinkA=2, shrinkB=4))

    ax.set_xlabel(r"$t$ (noise $\rightarrow$ data)")
    ax.set_ylabel("$x$")
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-2.05, 2.05)
    ax.set_aspect("auto")
    fl.clean_axes(ax, equal=False)
    fl.save(fig, "mdl-dyn-fm-paths")


# =========================================================================== #
# §27.4  The two time conventions  ->  mdl-dyn-time-conventions                #
# =========================================================================== #

def fig_time_conventions():
    """The two clocks side by side: the diffusion convention (t = 0 data,
    t = T noise; sampling integrates backwards) above the flow-matching
    convention (t = 0 noise, t = 1 data; sampling integrates forwards), with
    density glyphs at the axis ends and the t -> 1 - t bridge between them."""
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0.4, 6.1)
    ax.axis("off")

    xs = np.linspace(-2.6, 2.6, 160)
    bimodal = _gauss(xs, -1.3, 0.16) + _gauss(xs, 1.3, 0.16)
    unimodal = _gauss(xs, 0.0, 1.0)

    def glyph(cx, base, dens, color):
        gx = cx + xs * (1.6 / 5.2)
        gy = base + dens / dens.max() * 0.85
        ax.plot(gx, gy, color=color, lw=1.8, zorder=3)
        ax.fill_between(gx, base, gy, color=color, alpha=0.22, zorder=2)

    def clock(base, lab_l, lab_r):
        ax.plot([2.0, 8.0], [base, base], color="black", lw=1.2, zorder=1)
        for xx, lab in ((2.0, lab_l), (8.0, lab_r)):
            ax.plot([xx, xx], [base - 0.08, base + 0.08], color="black",
                    lw=1.2)
            ax.text(xx, base - 0.22, lab, fontsize=10, ha="center", va="top")

    # -- diffusion: data at t = 0, noise at t = T; sample backwards --------- #
    bD = 4.35
    clock(bD, "$0$", "$T$")
    glyph(2.0, bD, bimodal, BLUE)
    glyph(8.0, bD, unimodal, ORANGE)
    ax.text(1.05, bD + 0.45, "data", color=BLUE, fontsize=9.5, ha="right",
            va="center")
    ax.text(8.95, bD + 0.45, "noise", color=ORANGE, fontsize=9.5, ha="left",
            va="center")
    fl.arrow(ax, (3.3, bD + 1.05), (6.7, bD + 1.05), color=GRAY, lw=1.6,
             mut=12)
    ax.text(5.0, bD + 1.22, "training: noise the data", color=GRAY,
            fontsize=9, ha="center", va="bottom")
    fl.arrow(ax, (6.7, bD - 0.55), (3.3, bD - 0.55), color=GREEN, lw=1.6,
             mut=12)
    ax.text(5.0, bD - 0.74, "sampling: reverse-time SDE / probability-flow ODE",
            color=GREEN, fontsize=9, ha="center", va="top")
    ax.text(0.15, bD + 1.45, "diffusion", fontsize=11, ha="left",
            va="center")

    # -- flow matching: noise at t = 0, data at t = 1; sample forwards ------ #
    bF = 1.45
    clock(bF, "$0$", "$1$")
    glyph(2.0, bF, unimodal, ORANGE)
    glyph(8.0, bF, bimodal, BLUE)
    ax.text(1.05, bF + 0.45, "noise", color=ORANGE, fontsize=9.5, ha="right",
            va="center")
    ax.text(8.95, bF + 0.45, "data", color=BLUE, fontsize=9.5, ha="left",
            va="center")
    fl.arrow(ax, (3.3, bF + 1.05), (6.7, bF + 1.05), color=GREEN, lw=1.6,
             mut=12)
    ax.text(5.0, bF + 1.22,
            r"sampling: integrate $\dot{\mathbf{x}}"
            r"=\mathbf{v}_\theta(\mathbf{x},t)$ forwards",
            color=GREEN, fontsize=9, ha="center", va="bottom")
    ax.text(0.15, bF + 1.45, "flow matching", fontsize=11, ha="left",
            va="center")

    ax.text(5.0, 3.05, r"to compare formulas: $t\ \mapsto\ 1-t$",
            color=GRAY, fontsize=9.5, ha="center", va="center")
    fl.save(fig, "mdl-dyn-time-conventions")


# =========================================================================== #
# Driver                                                                      #
# =========================================================================== #

FIGURES = [
    fig_ode_field,
    fig_sde_paths,
    fig_fokker_planck_flux,
    fig_forward_reverse,
    fig_score_field,
    fig_uniqueness_fan,
    fig_phase_portraits,
    fig_resnet_as_euler,
    fig_brownian_paths,
    fig_qv_convergence,
    fig_ou_mean_reversion,
    fig_noising_denoising,
    fig_fm_paths,
    fig_time_conventions,
]


def main():
    # Only verify the figures THIS script writes (gen_mdl_figures.WRITTEN may
    # already hold entries if it was imported with side effects -- it is not,
    # but be defensive and snapshot the length).
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
        print(f"  {os.path.basename(p):34s} {size:>9,d} bytes")

    print(f"\nAll {len(written)} SVGs verified present, non-empty, valid.")


if __name__ == "__main__":
    main()
