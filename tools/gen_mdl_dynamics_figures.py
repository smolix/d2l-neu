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
# Driver                                                                      #
# =========================================================================== #

FIGURES = [
    fig_ode_field,
    fig_sde_paths,
    fig_fokker_planck_flux,
    fig_forward_reverse,
    fig_score_field,
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
