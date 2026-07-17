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


if __name__ == "__main__":
    fig_critical_damping()
    fig_river_valley()
    fig_norm_balls()
    for p in fl.WRITTEN:
        print("wrote", os.path.relpath(p))
