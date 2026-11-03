#!/usr/bin/env python3
"""Generate the new illustrative figures of "Flow Matching and Optimal
Transport" (``chapter_mdl-dynamics/mdl-flow-matching-transport.md``) in the
chapter's shared house style.

The shared style lives in ``gen_mdl_figures.py``; importing it applies the
``plt.rcParams`` (fixed ``svg.hashsalt`` and ``metadata={'Date': None}`` in
``save()`` make re-runs byte-for-byte identical) and exposes the palette.  The
section's older figures ``mdl-dyn-fm-paths`` and ``mdl-dyn-time-conventions``
stay in ``gen_mdl_dynamics_figures.py``.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_dynamics_flow_matching_transport_figures.py

Writes ``img/mdl-dyn-two-point-flow.svg`` and
``img/mdl-dyn-couplings-energy.svg``.  Every curve is computed: the two-point
trajectories integrate the closed-form marginal velocity with RK4, the
couplings use the exact deciles of the two Gaussians, and the rotation arcs are
exact quarter circles.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, INK = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.INK
PURPLE = fl.T.PURPLE.base
FAINT = fl.T.FAINT
BLUE_TINT = fl.T.BLUE.tint
WHITE_BOX = dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5)
save = fl.save

A = 2.0                                   # the two data points sit at -A, +A


def _u_two_point(x, t):
    """Marginal velocity of the straight path from N(0, 1) to the two points
    -A, +A under independent pairing: (A tanh(t A x / (1-t)^2) - x) / (1-t)."""
    return (A * np.tanh(t * A * x / (1 - t) ** 2) - x) / (1 - t)


def _rk4_two_point(x0, t1=0.995, n=4000):
    """RK4 trajectories of the two-point field from x0 at t = 0 to t1."""
    h = t1 / n
    xs = [np.asarray(x0, float)]
    x = xs[0].copy()
    for k in range(n):
        t = k * h
        k1 = _u_two_point(x, t)
        k2 = _u_two_point(x + 0.5 * h * k1, t + 0.5 * h)
        k3 = _u_two_point(x + 0.5 * h * k2, t + 0.5 * h)
        k4 = _u_two_point(x + h * k3, t + h)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        xs.append(x.copy())
    return np.linspace(0.0, t1, n + 1), np.array(xs)


def _spines(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("black")


def fig_two_point_flow():
    """(a) Exact marginal-flow trajectories for data at +-2 from Gaussian noise
    (independent pairing): everything first contracts toward 0, the origin
    stops attracting at t* = 3 - 2 sqrt 2, and trajectories near 0 commit to a
    mode late.  (b) Velocity profiles u_t(x) at four times."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.0, 4.0),
                                 gridspec_kw=dict(width_ratios=[1.3, 1.0],
                                                  wspace=0.45))

    # (a) trajectories ---------------------------------------------------------
    tb = np.linspace(0.0, 1.0, 400)
    for sgn in (-1, 1):
        ax.fill_between(tb, sgn * A * tb - (1 - tb), sgn * A * tb + (1 - tb),
                        color=BLUE_TINT, alpha=0.75, lw=0, zorder=1)
    starts = np.array([-2.0, -1.5, -1.0, -0.5, -0.12, 0.12, 0.5, 1.0, 1.5,
                       2.0])
    tt, xs = _rk4_two_point(starts)
    for k in range(len(starts)):
        ax.plot(tt, xs[:, k], color=BLUE, lw=2.0, zorder=3)
    ax.plot(np.zeros_like(starts), starts, "o", color=GRAY, ms=6, zorder=4)
    ax.plot([1, 1], [-A, A], "s", color="black", ms=7, zorder=4)
    t_star = 3 - 2 * np.sqrt(2)
    ax.axvline(t_star, color="black", lw=1.2, ls="--", zorder=2)
    ax.text(t_star + 0.02, 3.05, r"$t^{\ast}\!=0.17$: the origin" "\n"
            "stops attracting", fontsize=13, color=INK, ha="left",
            va="center", zorder=6)
    ax.text(1.03, A, "data $+2$", fontsize=13, color=INK, ha="left",
            va="center")
    ax.text(1.03, -A, "data $-2$", fontsize=13, color=INK, ha="left",
            va="center")
    i = np.searchsorted(tt, 0.5)
    ax.annotate("near $0$ the flow\ndecides late", xy=(tt[i], xs[i, 5]),
                xytext=(0.64, 0.0), fontsize=13, color=INK, ha="left",
                va="center", bbox=WHITE_BOX, zorder=7,
                arrowprops=dict(arrowstyle="->", color="black", lw=1.0,
                                shrinkA=2, shrinkB=3))
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-2.6, 3.6)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_yticks([-2, 0, 2])
    ax.set_xlabel(r"$t$ (noise $\rightarrow$ data)", fontsize=15)
    ax.set_ylabel("$x$", fontsize=15)
    _spines(ax)

    # (b) velocity profiles ----------------------------------------------------
    x = np.linspace(-3, 3, 801)
    bx.axhline(0, color="black", lw=0.9, zorder=1)
    bx.axvline(0, color="black", lw=0.9, zorder=1)
    cols = [GRAY, ORANGE, GREEN, PURPLE]
    times = (0.0, 0.2, 0.4, 0.6)
    for t, c in zip(times, cols):
        bx.plot(x, _u_two_point(x, t), color=c, lw=2.2, zorder=3)
    # labels at the right ends, outside the panel, nudged apart vertically
    # (green and orange end 0.2 apart); checked on the render
    for t, c, y in zip(times, (fl.T.SLATE.dark, fl.T.ORANGE.dark,
                               fl.T.GREEN.dark, fl.T.PURPLE.dark),
                       (-3.2, -1.0, -1.75, -2.5)):
        bx.text(3.12, y, f"$t={t:g}$", color=c, fontsize=14, ha="left",
                va="center", zorder=6, clip_on=False)
    bx.set_xlim(-3, 3)
    bx.set_ylim(-5, 5)
    bx.set_xticks([-2, 0, 2])
    bx.set_yticks([-4, 0, 4])
    bx.set_xlabel("$x$", fontsize=15)
    bx.set_ylabel("$u_t(x)$", fontsize=15)
    _spines(bx)
    save(fig, "mdl-dyn-two-point-flow")


def fig_couplings_energy():
    """(a) Independent pairing of the deciles of N(0, 1) with those of N(0, 4):
    segments cross.  (b) Sorted pairing x1 = 2 x0: no crossings, cost
    W_2^2 = 1.  (c) Quarter rotation of a standard Gaussian cloud: no
    crossings, endpoint law = start law, kinetic energy pi^2 / 2."""
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0),
                             gridspec_kw=dict(wspace=0.22))
    probs = (np.arange(10) + 0.5) / 10
    # standard normal quantiles via the inverse error function (no scipy)
    from math import erf, sqrt

    def _ppf(p):
        lo, hi = -8.0, 8.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if 0.5 * (1 + erf(mid / sqrt(2))) < p:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    q0 = np.array([_ppf(p) for p in probs])
    q1 = 2 * q0
    perm = np.array([6, 2, 9, 0, 7, 3, 1, 8, 4, 5])   # a fixed shuffle

    def _panel(ax, targets, color, ls, label):
        for x0, x1 in zip(q0, targets):
            ax.plot([0, 1], [x0, x1], color=color, lw=1.8, ls=ls, zorder=3)
        ax.plot(np.zeros(10), q0, "o", color=GRAY, ms=6, zorder=4)
        ax.plot(np.ones(10), q1, "s", color="black", ms=6, zorder=4)
        ax.text(0.5, 4.15, label, fontsize=14, color=INK, ha="center",
                va="center", zorder=6)
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(-3.7, 4.5)
        ax.set_xticks([0, 1])
        ax.set_yticks([-3, 0, 3])
        ax.set_xlabel("$t$", fontsize=15)
        ax.set_box_aspect(1)
        _spines(ax)

    _panel(axes[0], q1[perm], ORANGE, "--", "independent: cost $5$")
    axes[0].set_ylabel("$x$", fontsize=15)
    _panel(axes[1], q1, BLUE, "-", r"sorted: cost $1=W_2^2$")

    # (c) rotation ---------------------------------------------------------------
    ax = axes[2]
    rng = np.random.default_rng(7)
    cloud = rng.standard_normal((300, 2))
    cloud = cloud[np.abs(cloud).max(1) < 2.7]
    ax.plot(cloud[:, 0], cloud[:, 1], ".", color=FAINT, ms=3.5, zorder=1)
    s = np.linspace(0, np.pi / 2, 100)
    for r, phis in ((1.0, (0, 180)), (2.0, (90, 270))):
        for phi in phis:
            ang = np.deg2rad(phi) + s
            px, py = r * np.cos(ang), r * np.sin(ang)
            ax.plot(px[:-8], py[:-8], color=GREEN, lw=2.4, zorder=3)
            fl.arrow(ax, (px[-10], py[-10]), (px[-1], py[-1]), color=GREEN,
                     lw=0.0, mut=20)
            ax.plot([px[0]], [py[0]], "o", color=GREEN, ms=5, zorder=4)
    fl.axis_cross(ax, (-2.8, 2.8), (-2.8, 2.8), color="black", lw=0.9)
    ax.text(0.0, 3.25, r"rotation: energy $\pi^2/2$, $W_2^2=0$",
            fontsize=14, color=INK, ha="center", va="center", zorder=6)
    ax.set_xlim(-2.9, 2.9)
    ax.set_ylim(-2.9, 3.6)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "mdl-dyn-couplings-energy")


FIGURES = [fig_two_point_flow, fig_couplings_energy]


def main():
    for f in FIGURES:
        f()
    for p in fl.WRITTEN:
        print("wrote", os.path.relpath(p))


if __name__ == "__main__":
    main()
