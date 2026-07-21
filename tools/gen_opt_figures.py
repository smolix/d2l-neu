#!/usr/bin/env python3
"""Generate the illustrative figures for the "Optimization Algorithms"
chapter (ch. 9) in the shared house style.

Reuses the palette, rcParams (deterministic SVG hash salt), and drawing
helpers from ``tools/gen_mdl_figures.py`` so re-runs are byte-for-byte
identical.  Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_opt_figures.py

All figures are written to ``img/mdl-opt-<id>.svg``.  The two dynamical
figures (critical damping, river valley) are computed from the actual
recursions, not sketched, so the curves are honest.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # noqa: E402  (shared style + helpers)

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT


# =========================================================================== #
# momentum.md: convergence rate vs beta, critical damping                     #
# =========================================================================== #

def _hb_rate(eta_lam: float, beta: float) -> float:
    """Spectral radius of the heavy-ball iteration matrix on a scalar mode."""
    a = np.array([[1.0 + beta - eta_lam, -beta], [1.0, 0.0]])
    return float(np.max(np.abs(np.linalg.eigvals(a))))


def fig_critical_damping():
    betas = np.linspace(0.0, 0.97, 400)
    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    for eta_lam, color, ls, tag in [
        (0.2, BLUE, "-", r"$\eta\lambda = 0.2$"),
        (0.05, GREEN, "--", r"$\eta\lambda = 0.05$"),
    ]:
        rates = [_hb_rate(eta_lam, b) for b in betas]
        ax.plot(betas, rates, color=color, ls=ls, lw=2.2)
        bstar = (1.0 - np.sqrt(eta_lam)) ** 2
        ax.plot([bstar], [_hb_rate(eta_lam, bstar)], "o", color=color, ms=7,
                zorder=5)
        ax.axvline(bstar, color=color, lw=0.8, ls=":", alpha=0.6)
        ax.text(bstar, 1.015, r"$\beta^{*}$", color=color, ha="center",
                va="bottom", fontsize=14)
        # curve tag sits above the flat left arm of its own curve
        ax.text(0.02, _hb_rate(eta_lam, 0.0) + 0.022, tag, color=color,
                fontsize=13, ha="left", va="bottom")
    # regime labels for the better-conditioned (blue) curve
    ax.text(0.145, 0.585, "over-damped:\nmonotone, slow", color="black",
            fontsize=12.5, ha="center", va="center")
    ax.text(0.72, 0.60, "under-damped:\noscillates, rate $\\sqrt{\\beta}$",
            color="black", fontsize=12.5, ha="center", va="center")
    ax.annotate("", xy=(0.86, 0.945), xytext=(0.86, 0.80),
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.2))
    ax.text(0.875, 0.87, "slower", color="black", fontsize=11.5,
            ha="left", va="center")
    ax.set_xlim(0.0, 0.97)
    ax.set_ylim(0.5, 1.06)
    ax.set_xlabel(r"momentum $\beta$", fontsize=13, color="black")
    ax.set_ylabel("convergence rate per step", fontsize=13, color="black")
    ax.tick_params(labelsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fl.save(fig, "mdl-opt-critical-damping")


# =========================================================================== #
# lr-scheduler.md: the river-valley picture of warmup-stable-decay            #
# =========================================================================== #

def _river(x):
    return 0.75 * np.sin(0.9 * x) + 0.15 * x - 1.0


def _valley_loss(x, y):
    return 4.0 * (y - _river(x)) ** 2 - 0.55 * x


def _valley_grad(x, y):
    d = y - _river(x)
    dyc = 0.75 * 0.9 * np.cos(0.9 * x) + 0.15
    return np.array([-0.55 - 8.0 * d * dyc, 8.0 * d])


def _run_valley(steps, eta_fn, noise, start=(0.2, 0.2)):
    """SGD on the valley: shared, seeded noise models minibatch gradients.
    At constant eta the noise keeps re-exciting the transverse mode (the
    bouncing); decaying eta quenches it — the point of the figure."""
    p = np.array(start, dtype=float)
    traj = [p.copy()]
    for k in range(steps):
        p = p - eta_fn(k) * (_valley_grad(*p) + noise[k])
        traj.append(p.copy())
    return np.array(traj)


def fig_river_valley():
    warm, total, decay = 10, 145, 30
    top = 0.16
    floor = 0.15 * top  # decay to a small floor: settles yet still advances

    def eta_const(k):  # warmup then constant
        return top * (k + 1) / warm if k < warm else top

    # one shared, seeded noise sequence => deterministic output and the two
    # runs coincide exactly until the schedules diverge
    noise = fl.np.random.default_rng(7).normal(0.0, 2.0, size=(total + decay + 10, 2))
    const = _run_valley(total, eta_const, noise)

    # branch the decay where the shared run first crosses mid-frame — the
    # x-coordinate is a drifting random walk, so a step-count split could
    # land anywhere
    split = int(np.argmax(const[:, 0] > 6.0))

    def eta_wsd(k):
        if k < split:
            return eta_const(k)
        t = min((k - split) / decay, 1.0)
        return floor + (top - floor) * 0.5 * (1 + np.cos(np.pi * t))

    wsd = _run_valley(split + decay + 30, eta_wsd, noise)

    xs = np.linspace(-0.4, 9.6, 320)
    ys = np.linspace(-2.2, 2.6, 260)
    X, Y = np.meshgrid(xs, ys)
    Z = _valley_loss(X, Y)

    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    ax.contourf(X, Y, Z, levels=26, cmap="Blues_r", alpha=0.55)
    ax.plot(xs, _river(xs), color="black", lw=1.0, ls="--", alpha=0.7)

    ax.plot(const[:, 0], const[:, 1], color=ORANGE, lw=1.3, marker="o",
            ms=2.2, mew=0, alpha=0.9, zorder=4)
    ax.plot(wsd[split:, 0], wsd[split:, 1], color=GREEN, lw=2.6, marker="o",
            ms=3.0, mew=0, zorder=6)

    bbox = dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85)
    ax.text(0.72, 0.82, "warmup", color="black", fontsize=13, ha="left",
            va="center", bbox=bbox)
    ax.text(3.9, 2.28, "stable: bounces off the walls,\nstill moves downstream",
            color="black", fontsize=13, ha="center", va="center", bbox=bbox)
    ax.text(6.6, -1.75, "decay: settles\nto the floor", color="black",
            fontsize=13, ha="center", va="center", bbox=bbox)
    ax.annotate("", xy=(wsd[-1, 0], wsd[-1, 1] - 0.12), xytext=(6.6, -1.35),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.4))
    ax.annotate("", xy=(2.1, _river(2.1) - 0.06), xytext=(0.9, -1.9),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.0))
    ax.text(0.9, -2.02, "the river (valley floor)", color="black",
            fontsize=12, ha="left", va="top", bbox=bbox)

    ax.set_xlim(-0.4, 9.6)
    ax.set_ylim(-2.35, 2.6)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fl.save(fig, "mdl-opt-river-valley")


# =========================================================================== #
# muon.md: steepest descent under three norms                                 #
# =========================================================================== #

def fig_norm_balls():
    g = np.array([0.55, 0.85])
    ghat = g / np.linalg.norm(g)

    fig, axes = plt.subplots(1, 3, figsize=(9.9, 3.5))

    # --- (a) l2 ball: step opposite the gradient -------------------------- #
    ax = axes[0]
    th = np.linspace(0, 2 * np.pi, 200)
    ax.fill(np.cos(th), np.sin(th), color=BLUE, alpha=0.10, lw=0)
    ax.plot(np.cos(th), np.sin(th), color=BLUE, lw=1.4)
    fl.arrow(ax, (0, 0), tuple(g), color=GRAY, lw=2.0)
    ax.text(g[0] + 0.10, g[1] + 0.08, r"$\mathbf{g}$", color="black",
            fontsize=15, ha="left", va="bottom")
    fl.arrow(ax, (0, 0), tuple(-ghat), color=ORANGE, lw=2.6)
    ax.text(-ghat[0] - 0.10, -ghat[1] - 0.10, r"$-\mathbf{g}/\|\mathbf{g}\|_2$",
            color="black", fontsize=13.5, ha="right", va="top")
    ax.set_title(r"$\ell_2$: follow the gradient", fontsize=13.5, color="black")

    # --- (b) l_inf ball: step to the sign corner -------------------------- #
    ax = axes[1]
    sq = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]], float)
    ax.fill(sq[:, 0], sq[:, 1], color=BLUE, alpha=0.10, lw=0)
    ax.plot(sq[:, 0], sq[:, 1], color=BLUE, lw=1.4)
    fl.arrow(ax, (0, 0), tuple(g), color=GRAY, lw=2.0)
    ax.text(g[0] + 0.10, g[1] + 0.08, r"$\mathbf{g}$", color="black",
            fontsize=15, ha="left", va="bottom")
    fl.arrow(ax, (0, 0), (-1, -1), color=ORANGE, lw=2.6)
    ax.text(-1.02, -1.14, r"$-\mathrm{sign}(\mathbf{g})$", color="black",
            fontsize=13.5, ha="left", va="top")
    ax.set_title(r"$\ell_\infty$: keep only the signs", fontsize=13.5,
                 color="black")

    # --- (c) spectral ball in singular-value coordinates ------------------ #
    ax = axes[2]
    ax.fill([0, 1, 1, 0], [0, 0, 1, 1], color=BLUE, alpha=0.10, lw=0)
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color=BLUE, lw=1.4)
    sg = (0.92, 0.22)
    ax.plot(*sg, "o", color=GRAY, ms=8)
    ax.text(sg[0] - 0.04, sg[1] - 0.05, r"$\sigma(\mathbf{G})$", color="black",
            fontsize=13.5, ha="right", va="top")
    ax.plot([sg[0], 1.0], [sg[1], 1.0], color=GRAY, lw=1.0, ls=":")
    ax.plot(1.0, 1.0, "o", color=ORANGE, ms=10, zorder=5)
    ax.text(0.97, 1.06, r"$\mathbf{U}\mathbf{V}^{\!\top}$", color="black",
            fontsize=14.5, ha="right", va="bottom")
    ax.set_xlabel(r"$\sigma_1$ of the update", fontsize=12.5, color="black")
    ax.set_ylabel(r"$\sigma_2$ of the update", fontsize=12.5, color="black")
    ax.set_title(r"spectral: every $\sigma \to 1$", fontsize=13.5,
                 color="black")
    ax.set_xlim(-0.12, 1.30)
    ax.set_ylim(-0.12, 1.30)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.tick_params(labelsize=11)
    ax.set_aspect("equal")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    for ax in axes[:2]:
        ax.set_xlim(-1.55, 1.55)
        ax.set_ylim(-1.55, 1.55)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        fl.axis_cross(ax, (-1.45, 1.45), (-1.45, 1.45), color="black")

    fig.subplots_adjust(wspace=0.32)
    fl.save(fig, "mdl-opt-norm-balls")


# =========================================================================== #
# optimization-intro.md (Landscapes): risk vs empirical risk, the zero-        #
# gradient traps (local minima, a stationary inflection, a saddle), and a      #
# flat/saturated region.  Every curve is the real analytic function evaluated  #
# on a grid, not a sketch.                                                     #
# =========================================================================== #

def _fn_axes(ax, xlabel, ylabel):
    """Shared look for a single-function plot: black bottom/left axes + labels,
    top/right spines dropped."""
    ax.set_xlabel(xlabel, fontsize=13, color="black")
    ax.set_ylabel(ylabel, fontsize=13, color="black")
    ax.tick_params(labelsize=11, colors="black")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("black")


_ANN = dict(arrowstyle="->", color="black", lw=1.2)
_WBB = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9)


def fig_risk_gap():
    """The optimizer's target (empirical risk g) wobbles around what learning
    wants (the risk f), and their minima sit in different places.
    f(x)=x cos(pi x); g adds a fast ripple.  Minima located on the grid."""
    x = np.linspace(0.5, 1.5, 600)
    f = x * np.cos(np.pi * x)
    g = f + 0.2 * np.cos(5 * np.pi * x)
    i_f, i_g = int(np.argmin(f)), int(np.argmin(g))

    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    ax.plot(x, f, color=BLUE, lw=2.6, label=r"risk $f$")
    ax.plot(x, g, color=ORANGE, lw=1.8, label=r"empirical risk $g$")
    ax.plot(x[i_f], f[i_f], "o", color=BLUE, ms=7, zorder=5)
    ax.plot(x[i_g], g[i_g], "o", color=ORANGE, ms=7, zorder=5)

    ax.annotate("min of risk", xy=(x[i_f], f[i_f]), xytext=(1.17, -0.70),
                fontsize=12.5, color="black", ha="center", arrowprops=_ANN)
    ax.annotate("min of\nempirical risk", xy=(x[i_g], g[i_g]),
                xytext=(0.72, -0.98), fontsize=12.5, color="black",
                ha="center", arrowprops=_ANN)
    ax.set_xlim(0.5, 1.5)
    ax.set_ylim(-1.38, 0.16)
    _fn_axes(ax, r"$x$", "risk")
    ax.legend(loc="upper center", ncol=2, fontsize=12, frameon=False,
              handlelength=1.4, columnspacing=1.4)
    fl.save(fig, "mdl-opt-risk-gap")


def fig_local_minima():
    """f(x)=x cos(pi x) on [-1, 2] has a local minimum that is not global; both
    minima are found by a neighbour test on the grid, so the marks are exact."""
    x = np.linspace(-1.0, 2.0, 700)
    y = x * np.cos(np.pi * x)
    loc = 1 + np.where((y[1:-1] < y[:-2]) & (y[1:-1] < y[2:]))[0]
    gi = int(loc[np.argmin(y[loc])])          # global minimum
    li = int(loc[np.argmax(y[loc])])          # the shallower (local) minimum

    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    ax.plot(x, y, color=BLUE, lw=2.6)
    ax.plot(x[li], y[li], "o", color=ORANGE, ms=7, zorder=5)
    ax.plot(x[gi], y[gi], "o", color=GREEN, ms=7, zorder=5)
    ax.annotate("local minimum", xy=(x[li], y[li]), xytext=(-0.95, -1.05),
                fontsize=12.5, color="black", ha="left", arrowprops=_ANN)
    ax.annotate("global minimum", xy=(x[gi], y[gi]), xytext=(0.35, 0.9),
                fontsize=12.5, color="black", ha="left", arrowprops=_ANN)
    ax.set_xlim(-1.0, 2.0)
    _fn_axes(ax, r"$x$", r"$f(x)$")
    fl.save(fig, "mdl-opt-local-minima")


def fig_inflection():
    """f(x)=x^3: at x=0 both f' and f'' vanish, yet it is no extremum -- a
    stationary inflection (the 1-D cousin of a saddle).  The dashed horizontal
    tangent marks the vanishing slope."""
    x = np.linspace(-2.0, 2.0, 500)
    y = x ** 3
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    ax.plot(x, y, color=BLUE, lw=2.6)
    ax.plot([-0.8, 0.8], [0, 0], "--", color=GRAY, lw=1.5)   # tangent at 0
    ax.plot(0, 0, "o", color=ORANGE, ms=8, zorder=5)
    ax.annotate("stationary inflection\n" r"$f'(0)=f''(0)=0$", xy=(0, 0),
                xytext=(-1.95, 3.4), fontsize=12.5, color="black", ha="left",
                arrowprops=_ANN)
    ax.set_xlim(-2.0, 2.0)
    _fn_axes(ax, r"$x$", r"$f(x)=x^3$")
    fl.save(fig, "mdl-opt-inflection")


def fig_tanh_flat():
    """f(x)=tanh(x): starting from x=4 the surface is nearly flat
    (f'(4)=1-tanh^2(4)~=0.0013), so gradient descent barely moves.  The dashed
    tangent at x=4 shows how flat -- no critical point involved."""
    x = np.linspace(-2.0, 5.0, 500)
    y = np.tanh(x)
    x0 = 4.0
    y0 = np.tanh(x0)
    slope = 1 - y0 ** 2                        # 0.00134...
    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    ax.plot(x, y, color=BLUE, lw=2.6)
    xt = np.array([x0 - 1.3, x0 + 1.0])
    ax.plot(xt, y0 + slope * (xt - x0), "--", color=GRAY, lw=1.5)
    ax.plot(x0, y0, "o", color=ORANGE, ms=8, zorder=5)
    ax.annotate("vanishing gradient\n" rf"$f'(4)\approx{slope:.4f}$",
                xy=(x0, y0), xytext=(0.7, 0.1), fontsize=12.5, color="black",
                ha="left", arrowprops=_ANN)
    ax.set_xlim(-2.0, 5.0)
    _fn_axes(ax, r"$x$", r"$f(x)=\tanh x$")
    fl.save(fig, "mdl-opt-tanh-flat")


def fig_saddle():
    """Redesign of the old 3-D saddle wireframe: a contour of z = x^2 - y^2
    beside the two 1-D slices through the saddle.  Along x the surface is a
    bowl (a minimum); along y an inverted bowl (a maximum) -- so the origin is
    a critical point that is neither.  Blue = the x-slice / positive z, orange
    = the y-slice / negative z, tying the two panels together."""
    import matplotlib.colors as mcolors

    fig, (axc, axs) = plt.subplots(
        1, 2, figsize=(9.6, 4.0),
        gridspec_kw={"width_ratios": [1.08, 1.0]})

    # ---- (a) level sets of z = x^2 - y^2 --------------------------------- #
    g = np.linspace(-1.0, 1.0, 201)
    X, Y = np.meshgrid(g, g)
    Z = X ** 2 - Y ** 2
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "saddle", [ORANGE, "white", BLUE])
    # Smooth raster background (one compact embedded PNG) instead of a
    # many-thousand-polygon contourf — img/ is inline, not LFS, so keep it small.
    axc.imshow(Z, extent=[-1.0, 1.0, -1.0, 1.0], origin="lower", cmap=cmap,
               vmin=-1.0, vmax=1.0, aspect="equal", interpolation="bilinear",
               zorder=0)
    axc.contour(X, Y, Z, levels=9, colors=[GRAY], linewidths=0.6, alpha=0.55)
    # the two slice directions, coloured to match panel (b)
    axc.plot([-1.0, 1.0], [0, 0], color=BLUE, lw=2.6)       # y=0: z=x^2
    axc.plot([0, 0], [-1.0, 1.0], color=ORANGE, lw=2.6)     # x=0: z=-y^2
    axc.plot(0, 0, "o", color="black", ms=7, zorder=6)
    axc.text(0.60, 0.20, r"min along $x$", color=BLUE, fontsize=13,
             ha="center", va="center", bbox=_WBB)
    axc.text(0.24, 0.66, r"max along $y$", color=ORANGE, fontsize=13,
             ha="center", va="center", bbox=_WBB)
    axc.annotate("saddle", xy=(0, 0), xytext=(-0.9, -0.78), fontsize=12.5,
                 color="black", ha="left", arrowprops=_ANN)
    axc.set_xlim(-1.0, 1.0)
    axc.set_ylim(-1.0, 1.0)
    axc.set_aspect("equal")
    axc.set_xlabel(r"$x$", fontsize=13, color="black")
    axc.set_ylabel(r"$y$", fontsize=13, color="black")
    axc.tick_params(labelsize=11, colors="black")
    for s in ("top", "right"):
        axc.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        axc.spines[s].set_color("black")

    # ---- (b) the two slices through the saddle --------------------------- #
    s = np.linspace(-1.0, 1.0, 240)
    axs.plot(s, s ** 2, color=BLUE, lw=2.6)
    axs.plot(s, -s ** 2, color=ORANGE, lw=2.6)
    axs.axhline(0.0, color=GRAY, lw=0.8, ls=":")
    axs.plot(0, 0, "o", color="black", ms=7, zorder=6)
    axs.text(0.72, 0.60, r"$z=x^2$", color=BLUE, fontsize=13, ha="center")
    axs.text(0.72, -0.62, r"$z=-y^2$", color=ORANGE, fontsize=13, ha="center")
    axs.text(0.03, 0.14, "minimum", color=BLUE, fontsize=11.5, ha="left")
    axs.text(0.03, -0.16, "maximum", color=ORANGE, fontsize=11.5, ha="left",
             va="top")
    axs.set_xlim(-1.0, 1.0)
    axs.set_ylim(-1.1, 1.1)
    _fn_axes(axs, "distance from the saddle", r"$z$")

    fig.subplots_adjust(wspace=0.28)
    fl.save(fig, "mdl-opt-saddle")


if __name__ == "__main__":
    fig_critical_damping()
    fig_river_valley()
    fig_norm_balls()
    fig_risk_gap()
    fig_local_minima()
    fig_inflection()
    fig_tanh_flat()
    fig_saddle()
    for p in fl.WRITTEN:
        print("wrote", os.path.relpath(p))
