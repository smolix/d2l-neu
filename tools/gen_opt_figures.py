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
from matplotlib.colors import to_rgba  # noqa: E402
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
        # the guide stops short of its own beta_c tag (an axvline ran through it)
        ax.plot([bstar, bstar], [0.5, 1.0], color=color, lw=0.8, ls=":",
                alpha=0.6)
        ax.text(bstar, 1.015, r"$\beta_{\mathrm{c}}$", color=color, ha="center",
                va="bottom", fontsize=14)
        # curve tag sits above the flat left arm of its own curve
        ax.text(0.02, _hb_rate(eta_lam, 0.0) + 0.022, tag, color=color,
                fontsize=13, ha="left", va="bottom")
    # regime labels for the better-conditioned (blue) curve
    ax.text(0.145, 0.585, "overdamped:\nmonotone, slow", color="black",
            fontsize=12.5, ha="center", va="center")
    ax.text(0.785, 0.60, "underdamped:\noscillates, factor $\\sqrt{\\beta}$",
            color="black", fontsize=12.5, ha="center", va="center")
    # the head stops just under the sqrt(beta) curve instead of piercing it
    ax.annotate("", xy=(0.86, np.sqrt(0.86) - 0.03), xytext=(0.86, 0.77),
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.2))
    ax.text(0.875, 0.835, "slower", color="black", fontsize=11.5,
            ha="left", va="center")
    ax.set_xlim(0.0, 0.97)
    ax.set_ylim(0.5, 1.06)
    ax.set_xlabel(r"momentum $\beta$", fontsize=13, color="black")
    ax.set_ylabel("contraction factor per step", fontsize=13, color="black")
    # black ticks/spines, like the chapter's other function plots (_fn_axes)
    ax.tick_params(labelsize=12.5, colors="black")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("black")
    fl.save(fig, "mdl-opt-critical-damping")


def fig_nesterov_lookahead():
    """One step of heavy-ball and of Nesterov momentum on the valley."""
    h = np.array([0.2, 4.0])  # curvatures of the valley 0.1 x1^2 + 2 x2^2
    eta = 0.25
    x = np.array([-3.0, 0.6])                # current iterate x_{t-1}
    jump = np.array([0.8, -1.0])             # -eta * beta * v_{t-1}
    look = x + jump                          # look-ahead point
    hb = look - eta * h * x                  # gradient taken at x_{t-1}
    nes = look - eta * h * look              # gradient taken at the look-ahead

    xlim, ylim = (-4.1, 0.9), (-1.35, 1.15)
    xs = np.linspace(*xlim, 300)
    ys = np.linspace(*ylim, 240)
    X, Y = np.meshgrid(xs, ys)
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    ax.contour(X, Y, 0.1 * X ** 2 + 2 * Y ** 2, levels=[0.4, 0.9, 1.6, 2.5],
               colors=GRAY, linewidths=0.8, alpha=0.5)
    ax.plot([0], [0], "+", color="black", ms=10, mew=1.5)
    fl.arrow(ax, x, look, color=GRAY, lw=2.0)
    fl.arrow(ax, look, hb, color=BLUE, lw=2.2)
    fl.arrow(ax, look, nes, color=ORANGE, lw=2.2)
    ax.plot(*x, "o", color="black", ms=7, zorder=5)
    ax.plot(*look, "o", mfc="white", mec="black", ms=7, mew=1.4, zorder=5)
    ax.plot(*hb, "o", color=BLUE, ms=7, zorder=5)
    ax.plot(*nes, "o", color=ORANGE, ms=7, zorder=5)

    box = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9)
    ax.text(x[0] - 0.1, x[1] + 0.06, r"$\mathbf{x}_{t-1}$", fontsize=15,
            color="black", ha="right", va="bottom", bbox=box)
    ax.text(-2.86, 0.12, "momentum step\n" + r"$-\eta\beta\,\mathbf{v}_{t-1}$",
            fontsize=13, color="black", ha="right", va="center", bbox=box)
    ax.text(look[0] - 0.14, look[1] - 0.04, "look-ahead\npoint", fontsize=13,
            color="black", ha="right", va="top", bbox=box)
    ax.text(hb[0] + 0.16, hb[1] - 0.02,
            "heavy ball: gradient at\n" + r"$\mathbf{x}_{t-1}$, overshoots",
            fontsize=13, color=BLUE, ha="left", va="center", bbox=box)
    ax.text(nes[0] + 0.16, nes[1] + 0.16,
            "Nesterov: gradient at the\nlook-ahead point, lands\nnear the floor",
            fontsize=13, color=ORANGE, ha="left", va="bottom", bbox=box)
    ax.text(0.02, -0.1, "minimum", fontsize=12, color="black", ha="center",
            va="top", bbox=box)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fl.save(fig, "mdl-opt-nesterov-lookahead")


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

    # coarse grid: the fill is smooth, and img/ SVGs are inline (keep small)
    xs = np.linspace(-0.4, 9.6, 200)
    ys = np.linspace(-2.35, 2.6, 120)
    X, Y = np.meshgrid(xs, ys)
    Z = _valley_loss(X, Y)

    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    # light valley: white floor, walls tinting to blue (the old dark Blues_r
    # wash buried both the walls and the paths)
    from matplotlib.colors import LinearSegmentedColormap
    T = fl.T
    cmap = LinearSegmentedColormap.from_list(
        "valley", ["white", T.BLUE.tint, T.BLUE.base])
    lo = float(Z.min())
    ax.contourf(X, Y, Z, levels=np.linspace(lo, lo + 16.0, 17), cmap=cmap,
                alpha=0.55, extend="max")
    ax.plot(xs, _river(xs), color="black", lw=1.2, ls="--", alpha=0.8,
            zorder=3)

    # the constant-rate walk keeps drifting far past the frame; end it where
    # it first reaches the right edge (interpolated) rather than letting it
    # run out of the picture
    xr = 9.45
    k = int(np.argmax(const[:, 0] > xr))
    a, b = const[k - 1], const[k]
    const = np.vstack([const[:k], a + (xr - a[0]) / (b[0] - a[0]) * (b - a)])
    ax.plot(const[:, 0], const[:, 1], color=ORANGE, lw=1.3, marker="o",
            ms=2.4, mew=0, alpha=0.9, zorder=4)
    ax.plot(wsd[split:, 0], wsd[split:, 1], color=GREEN, lw=2.6, marker="o",
            ms=3.2, mew=0, zorder=6)
    ax.plot(*wsd[-1], "o", color=GREEN, ms=7, zorder=7)

    bbox = dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85)
    ax.text(0.2, 0.75, "warmup", color="black", fontsize=13, ha="left",
            va="bottom", bbox=bbox)
    ax.text(3.6, 1.95, "constant rate: bounces off the walls,\n"
            "still moves downstream", color=ORANGE, fontsize=13,
            ha="center", va="center", bbox=bbox)
    ax.text(7.6, -1.55, "decay: settles\nonto the floor", color=GREEN,
            fontsize=13, ha="center", va="center", bbox=bbox)
    # tip touches the underside of the settled green cluster (the old tip
    # stopped on an orange segment); the leader is drawn over the orange path
    ax.annotate("", xy=(8.27, 0.90), xytext=(8.15, -1.12),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.4),
                zorder=8)
    ax.annotate("", xy=(1.5, _river(1.5) - 0.08), xytext=(1.2, -1.62),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.0))
    ax.text(0.2, -1.65, "the river (valley floor)", color="black",
            fontsize=13, ha="left", va="top", bbox=bbox)

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
    # right of the tip, inside the square: above the tip it sat on the top edge
    ax.text(g[0] + 0.10, g[1] - 0.12, r"$\mathbf{g}$", color="black",
            fontsize=15, ha="left", va="center")
    fl.arrow(ax, (0, 0), (-1, -1), color=ORANGE, lw=2.6)
    # centred under the corner so it ends left of the vertical axis
    ax.text(-0.95, -1.12, r"$-\mathrm{sign}(\mathbf{g})$", color="black",
            fontsize=13.5, ha="center", va="top")
    ax.set_title(r"$\ell_\infty$: keep only the signs", fontsize=13.5,
                 color="black")

    # --- (c) spectral ball in singular-value coordinates ------------------ #
    ax = axes[2]
    ax.fill([0, 1, 1, 0], [0, 0, 1, 1], color=BLUE, alpha=0.10, lw=0)
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color=BLUE, lw=1.4)
    sg = (0.92, 0.22)
    ax.plot(*sg, "o", color=GRAY, ms=8)
    # left of the dot at its height (below it, the ")" grazed the box edge)
    ax.text(sg[0] - 0.07, sg[1], r"$\sigma(\mathbf{G})$", color="black",
            fontsize=13.5, ha="right", va="center")
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
    ax.tick_params(labelsize=13, colors="black")
    ax.set_aspect("equal")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("black")

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
# muon.md: Newton--Schulz as a scalar map on each singular value              #
# =========================================================================== #

NS_CUBIC = [(1.5, -0.5, 0.0)]
NS_JORDAN = [(3.4445, -4.7750, 2.0315)]
# Polar Express (Amsel et al., Implementation 1, safety factor folded in).
NS_POLAR = [(8.2052, -22.9019, 16.4607), (4.0664, -2.8612, 0.5184),
            (3.9096, -2.8234, 0.5250), (3.2856, -2.4153, 0.4853),
            (2.2779, -1.6198, 0.3985)]


def _ns_map(x, coeffs, steps):
    for t in range(steps):
        a, b, c = coeffs[min(t, len(coeffs) - 1)]
        x = a * x + b * x**3 + c * x**5
    return x


def fig_ns_maps():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.7))

    # --- (a) one step on [0, 1.2] ----------------------------------------- #
    x = np.linspace(0, 1.2, 400)
    ax1.axhline(1.0, color=LIGHT, lw=1.0, ls="--")
    ax1.plot(x, _ns_map(x, NS_CUBIC, 1), color=GREEN, lw=2.2)
    ax1.plot(x, _ns_map(x, NS_JORDAN, 1), color=BLUE, lw=2.2)
    ax1.text(1.05, 0.86, "cubic", color=GREEN, fontsize=13, ha="center",
             va="center")
    ax1.text(0.74, 1.19, "tuned quintic", color=BLUE, fontsize=13,
             ha="left", va="bottom")
    ax1.set_xlim(0, 1.2)
    ax1.set_ylim(0, 1.32)
    ax1.set_xticks([0, 0.5, 1.0])
    ax1.set_yticks([0, 0.5, 1.0])
    _fn_axes(ax1, r"singular value $\sigma$", r"after one step, $p(\sigma)$")

    # --- (b) five steps, log-spaced inputs ------------------------------- #
    s = np.logspace(-3, 0, 3000)
    ax2.axhline(1.0, color=LIGHT, lw=1.0, ls="--")
    ax2.plot(s, _ns_map(s, NS_CUBIC, 5), color=GREEN, lw=1.8)
    ax2.plot(s, _ns_map(s, NS_JORDAN, 5), color=BLUE, lw=1.6)
    ax2.plot(s, _ns_map(s, NS_POLAR, 5), color=ORANGE, lw=1.6)
    ax2.text(2.2e-2, 0.12, "cubic", color=GREEN, fontsize=13, ha="left",
             va="center")
    ax2.text(1.3e-3, 0.27, "tuned quintic", color=BLUE, fontsize=13,
             ha="left", va="center")
    ax2.text(1.3e-3, 0.40, "Polar Express", color=ORANGE, fontsize=13,
             ha="left", va="center")
    ax2.set_xscale("log")
    ax2.set_xlim(1e-3, 1)
    ax2.set_ylim(0, 1.32)
    ax2.set_yticks([0, 0.5, 1.0])
    _fn_axes(ax2, r"singular value $\sigma$ (log scale)",
             r"after five steps")

    fig.subplots_adjust(wspace=0.30)
    fl.save(fig, "mdl-opt-ns-maps")


# =========================================================================== #
# optimization-intro.md (Landscapes): risk vs empirical risk, the zero-       #
# gradient traps (local minima, a saddle), and a flat/saturated region.       #
# Every curve is the real analytic function evaluated on a grid, not a        #
# sketch.                                                                     #
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


def fig_tanh_flat():
    """f(x)=tanh(x): starting from x=4 the surface is nearly flat
    (f'(4)=1-tanh^2(4)~=0.0013), so gradient descent barely moves -- no
    critical point involved.  (A dashed tangent at x=4 used to lie on top of
    the curve and made that stretch look grey and dashed; the marker and the
    f'(4) label carry the point.)"""
    x = np.linspace(-2.0, 5.0, 500)
    y = np.tanh(x)
    x0 = 4.0
    y0 = np.tanh(x0)
    slope = 1 - y0 ** 2                        # 0.00134...
    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    ax.plot(x, y, color=BLUE, lw=2.6)
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
    # steeper than the x = -y zero-level diagonal, so the leader neither runs
    # along that grey line nor has it cut through the word
    axc.annotate("saddle", xy=(0, 0), xytext=(-0.45, -0.86), fontsize=12.5,
                 color="black", ha="center", va="center", bbox=_WBB,
                 arrowprops=dict(_ANN, shrinkB=5))
    axc.set_xlim(-1.0, 1.0)
    axc.set_ylim(-1.0, 1.0)
    axc.set_aspect("equal")
    axc.set_xlabel(r"$x$", fontsize=13, color="black")
    axc.set_ylabel(r"$y$", fontsize=13, color="black", rotation=0,
                   labelpad=10)
    axc.set_xticks([-1, 0, 1])
    axc.set_yticks([-1, 0, 1])
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
    # all four labels centred on the saddle, inside the parabola pockets,
    # clear of both curves
    axs.text(0.0, 0.62, r"$z=x^2$", color=BLUE, fontsize=13, ha="center")
    axs.text(0.0, -0.62, r"$z=-y^2$", color=ORANGE, fontsize=13, ha="center",
             va="top")
    axs.text(0.0, 0.16, "minimum", color=BLUE, fontsize=11.5, ha="center")
    axs.text(0.0, -0.16, "maximum", color=ORANGE, fontsize=11.5, ha="center",
             va="top")
    axs.set_xlim(-1.0, 1.0)
    axs.set_ylim(-1.1, 1.1)
    _fn_axes(axs, "distance from the saddle", r"$z$")
    axs.set_ylabel(r"$z$", fontsize=13, color="black", rotation=0,
                   labelpad=10)
    axs.set_xticks([-1, 0, 1])
    axs.set_yticks([-1, 0, 1])

    fig.subplots_adjust(wspace=0.28)
    fl.save(fig, "mdl-opt-saddle")

# =========================================================================== #
# index.md: the two obstacles -- ill-conditioning and gradient noise          #
# =========================================================================== #

def fig_two_obstacles():
    """Both panels run on the chapter's valley f = 0.1 x1^2 + 2 x2^2
    (lambda = 0.2, 4; kappa = 20) from the same start, in the same window.
    (a) gradient descent at eta = 0.45, just under 2/lambda_2 = 0.5: the
    steep coordinate zig-zags, the flat one creeps (factor 0.91 per step).
    (b) constant-step SGD with unit gradient noise at eta = 0.1: the iterates
    reach the minimum and then wander in a stationary cloud whose variance is
    eta sigma^2 / (2 lambda) per coordinate (dashed: two standard deviations;
    the exact values 0.2525 and 0.0156 are the ones sgd.md measures)."""
    lam = np.array([0.2, 4.0])
    start = np.array([-5.0, 1.5])
    lim_x, lim_y = (-5.6, 1.3), (-1.75, 1.75)
    xs = np.linspace(*lim_x, 300)
    ys = np.linspace(*lim_y, 200)
    X, Y = np.meshgrid(xs, ys)
    Z = 0.1 * X ** 2 + 2 * Y ** 2

    gd = [start.copy()]
    for _ in range(25):
        gd.append(gd[-1] - 0.45 * lam * gd[-1])
    gd = np.array(gd)

    rng = np.random.default_rng(3)
    eta, burn, keep = 0.1, 300, 1500
    p = start.copy()
    sgd = [p.copy()]
    for _ in range(burn + keep):
        p = p - eta * (lam * p + rng.normal(0.0, 1.0, 2))
        sgd.append(p.copy())
    sgd = np.array(sgd)
    var = eta / (2 * lam) / (1 - eta * lam / 2)  # exact stationary variance

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.0))
    box = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9)
    for ax in axes:
        ax.contour(X, Y, Z, levels=[0.3, 1.0, 2.0, 3.2, 4.6], colors=GRAY,
                   linewidths=0.8, alpha=0.5)
        ax.plot([0], [0], "+", color="black", ms=11, mew=1.6, zorder=6)
        ax.set_xlim(*lim_x)
        ax.set_ylim(*lim_y)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    ax = axes[0]
    ax.plot(gd[:, 0], gd[:, 1], color=BLUE, lw=1.8, marker="o", ms=3.5,
            mew=0, zorder=4)
    ax.plot(*start, "o", color="black", ms=6, zorder=5)
    ax.text(-0.9, 1.30, r"steep $\lambda_2 = 4$: zig-zag," "\n"
            r"stable only for $\eta < 0.5$", color="black", fontsize=13,
            ha="center", va="center", bbox=box)
    ax.text(-1.0, -1.38, r"flat $\lambda_1 = 0.2$: creeps at $\eta = 0.45$",
            color="black", fontsize=13, ha="center", va="center", bbox=box)

    ax = axes[1]
    ax.plot(sgd[:burn, 0], sgd[:burn, 1], color=ORANGE, lw=0.9, alpha=0.8,
            zorder=3)
    ax.plot(sgd[burn::3, 0], sgd[burn::3, 1], "o", color=ORANGE, ms=2.2,
            mew=0, alpha=0.55, zorder=3)
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(2 * np.sqrt(var[0]) * np.cos(th), 2 * np.sqrt(var[1]) * np.sin(th),
            color="black", lw=1.4, ls="--", zorder=5)
    ax.plot(*start, "o", color="black", ms=6, zorder=5)
    ax.text(-2.1, -1.25, "constant-step SGD never settles:\n"
            r"noise ball, variance $\eta\sigma^2/(2\lambda b)$",
            color="black", fontsize=13, ha="center", va="center", bbox=box)
    fig.subplots_adjust(wspace=0.06)
    fl.save(fig, "mdl-opt-two-obstacles")


# =========================================================================== #
# test-models.md: loss landmarks for TinyLM on WikiText-103 bytes            #
# =========================================================================== #

def fig_loss_landmarks():
    """Cross-entropy landmarks in nats per byte.  5.71 = ln 256 + s^2/2 with
    s^2 = 129/384 (the exercise in test-models.md); the unigram and add-one
    bigram tables, fit on 10^7 training bytes and scored on the validation
    split, give 3.17 and 2.46 (rounded to the prose's 3.2 and 2.5); the two
    TinyLM values are rounded to the precision that survives re-execution:
    about 2.1 for 2,000 steps of SGD, about 1.5 for the best tuned runs."""
    rows = [
        ("untrained TinyLM", 5.71, "5.71", GRAY),
        (r"uniform guess, $\ln 256$", 5.55, "5.55", GRAY),
        ("unigram byte table", 3.2, r"$\approx$3.2", BLUE),
        ("bigram byte table", 2.5, r"$\approx$2.5", BLUE),
        ("TinyLM, SGD, 2,000 steps", 2.1, r"$\approx$2.1", ORANGE),
        ("TinyLM, best tuned run", 1.5, r"$\approx$1.5", ORANGE),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    y = np.arange(len(rows))[::-1]
    for yi, (name, val, txt, color) in zip(y, rows):
        ax.barh(yi, val, height=0.62, color=to_rgba(color, 0.22), ec=color,
                lw=1.6)
        ax.text(val + 0.08, yi, txt, color="black", fontsize=13,
                ha="left", va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=13, color="black")
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 6.5)
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6])
    ax.set_xlabel("cross-entropy (nats per byte)", fontsize=13, color="black")
    ax.tick_params(axis="x", labelsize=11, colors="black")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("black")
    fl.save(fig, "mdl-opt-loss-landmarks")


# =========================================================================== #
# momentum.md: persistent gradients add up, alternating ones cancel          #
# =========================================================================== #

def fig_momentum_gain():
    """The velocity v_t = beta v_{t-1} + g_t at beta = 0.5 (the section's
    value) for a constant gradient g_t = 1 and an alternating one
    g_t = (-1)^(t+1).  It approaches 1/(1-beta) = 2 in the first case and
    oscillates with amplitude 1/(1+beta) = 2/3 in the second."""
    beta, T = 0.5, 10
    t = np.arange(1, T + 1)
    cases = [np.ones(T), (-1.0) ** (t + 1)]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.1), sharey=True)
    for ax, g, (amp, lab) in zip(
            axes, cases,
            [(1 / (1 - beta), r"$1/(1-\beta) = 2$"),
             (1 / (1 + beta), r"$\pm 1/(1+\beta) = \pm 2/3$")]):
        v = np.zeros(T)
        for k in range(T):
            v[k] = beta * (v[k - 1] if k else 0.0) + g[k]
        ax.axhline(0.0, color="black", lw=0.9)
        ax.vlines(t - 0.12, 0, g, color=GRAY, lw=3.0)
        ax.plot(t + 0.12, v, color=BLUE, lw=2.0, marker="o", ms=6, zorder=4)
        for sgn in ((1,) if g.min() > 0 else (1, -1)):
            ax.axhline(sgn * amp, color="black", lw=1.0, ls="--", zorder=2)
        ax.set_xlim(0.3, T + 0.7)
        ax.set_xticks([1, 5, 10])
        ax.set_xlabel(r"step $t$", fontsize=13, color="black")
        ax.tick_params(labelsize=11, colors="black")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("black")
    box = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9)
    ax = axes[0]
    ax.text(T + 0.6, 2.12, r"$1/(1-\beta) = 2$", fontsize=13, color="black",
            ha="right", va="bottom", bbox=box)
    ax.text(5.5, -0.55, r"gradient $g_t = 1$ (gray)", fontsize=13,
            color="black", ha="center", va="center")
    ax.text(4.3, 1.33, r"velocity $v_t$", fontsize=13, color=BLUE,
            ha="left", va="center")
    ax.set_ylabel("value", fontsize=13, color="black")
    ax = axes[1]
    ax.text(5.5, 2.12, r"gradient $g_t = (-1)^{t+1}$ (gray)", fontsize=13,
            color="black", ha="center", va="center")
    ax.text(5.5, 1.55, r"dashed: $\pm 1/(1+\beta) = \pm 2/3$", fontsize=13,
            color="black", ha="center", va="center")
    ax.set_ylim(-1.35, 2.45)
    ax.set_yticks([-1, 0, 1, 2])
    fig.subplots_adjust(wspace=0.12)
    fl.save(fig, "mdl-opt-momentum-gain")


# =========================================================================== #
# muon.md: orthogonalization flattens the spectrum                           #
# =========================================================================== #

def fig_muon_spectrum():
    """Singular values of a 64 x 32 matrix with decaying column scales (a
    stand-in for a gradient whose energy sits in a few directions), divided
    by sigma_1, and of its orthogonalization U V^T: every nonzero singular
    value becomes 1.  The first twelve are shown."""
    rng = np.random.default_rng(0)
    G = rng.normal(size=(64, 32)) / np.arange(1, 33) ** 1.2
    U, s, Vt = np.linalg.svd(G, full_matrices=False)
    so = np.linalg.svd(U @ Vt, compute_uv=False)
    k = 12
    idx = np.arange(1, k + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 2.9), sharey=True)
    for ax, vals, color, lab in [
            (axes[0], s[:k] / s[0], GRAY,
             r"$\mathbf{G} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^{\top}$"),
            (axes[1], so[:k], ORANGE,
             r"$\mathbf{U}\mathbf{V}^{\top}$")]:
        ax.bar(idx, vals, width=0.7, color=to_rgba(color, 0.22), ec=color,
               lw=1.5)
        ax.text(k + 0.5, 1.12, lab, fontsize=15, color="black", ha="right",
                va="bottom")
        ax.set_xlim(0.3, k + 0.7)
        ax.set_xticks([1, 4, 8, 12])
        ax.set_xlabel(r"index $i$", fontsize=13, color="black")
        ax.tick_params(labelsize=11, colors="black")
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            ax.spines[s_].set_color("black")
    axes[0].set_ylabel(r"$\sigma_i / \sigma_1$", fontsize=14, color="black")
    axes[0].set_ylim(0, 1.32)
    axes[0].set_yticks([0, 0.5, 1.0])
    fig.subplots_adjust(wspace=0.10)
    fl.save(fig, "mdl-opt-muon-spectrum")


if __name__ == "__main__":
    fig_critical_damping()
    fig_nesterov_lookahead()
    fig_river_valley()
    fig_norm_balls()
    fig_ns_maps()
    fig_tanh_flat()
    fig_saddle()
    fig_two_obstacles()
    fig_loss_landmarks()
    fig_momentum_gain()
    fig_muon_spectrum()
    for p in fl.WRITTEN:
        print("wrote", os.path.relpath(p))
