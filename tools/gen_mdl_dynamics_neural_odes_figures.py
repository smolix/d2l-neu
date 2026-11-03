#!/usr/bin/env python3
"""Generate the new illustrative figures of "Neural ODEs, Adjoints, and
Continuous Flows" (``chapter_mdl-dynamics/mdl-neural-odes.md``) in the
chapter's shared house style.

The shared style lives in ``gen_mdl_figures.py``; importing it applies the
``plt.rcParams`` (fixed ``svg.hashsalt`` and ``metadata={'Date': None}`` in
``save()`` make re-runs byte-for-byte identical) and exposes the palette.  The
section's older figure ``mdl-dyn-resnet-as-euler`` stays in
``gen_mdl_dynamics_figures.py``.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_dynamics_neural_odes_figures.py

Writes ``img/mdl-dyn-flow-order.svg``, ``img/mdl-dyn-adjoint-timeline.svg`` and
``img/mdl-dyn-divergence-volume.svg`` (and the copies in ``img/auto/`` that the
slides read).  Every curve is computed from the exact
solution of the field it shows (closed forms and the eigendecomposition of
``e^{At}``); nothing is sketched.
"""

from __future__ import annotations

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, INK = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.INK
FAINT = fl.T.FAINT
BLUE_TINT = fl.T.BLUE.tint
WHITE_BOX = dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5)
save = fl.save

K = np.array([[0.0, -1.0], [1.0, 0.0]])      # rotation generator


def _copy_to_auto(name):
    """The slides read ``@fig`` figures from ``img/auto``: keep that copy in
    step with the book's ``img/<name>.svg``."""
    shutil.copyfile(os.path.join(fl.IMG_DIR, f"{name}.svg"),
                    os.path.join(fl.IMG_DIR, "auto", f"{name}.svg"))


def _expm(B, t):
    """e^{Bt} through the eigendecomposition (all fields here are diagonalizable)."""
    mu, V = np.linalg.eig(B)
    return (V @ np.diag(np.exp(mu * t)) @ np.linalg.inv(V)).real


def _spines(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("black")


def fig_flow_order():
    """(a) Trajectories of dx/dt = -x in the (t, x) plane never cross; two
    hypothetical trajectories that swap -1 and 1 would have to meet.  (b) The
    residual step x -> x - 2x swaps them in one step (arrows cross).  (c) In the
    plane the rotation by pi carries (x, 0) to (-x, 0) along half circles that
    pass without meeting."""
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.7),
                             gridspec_kw=dict(wspace=0.30))

    # (a) the line: order is preserved ----------------------------------------
    ax = axes[0]
    t = np.linspace(0, 1, 200)
    for x0 in (-1.5, -1.0, -0.5, 0.5, 1.0, 1.5):
        ax.plot(t, x0 * np.exp(-t), color=BLUE, lw=2.0, zorder=3)
    ax.plot(t, np.cos(np.pi * t), color=GRAY, lw=1.8, ls="--", zorder=2)
    ax.plot(t, -np.cos(np.pi * t), color=GRAY, lw=1.8, ls="--", zorder=2)
    ax.plot([0.5], [0.0], marker="x", color="black", ms=11, mew=2.4, zorder=5)
    ax.text(0.98, 1.47, "$\\times$: two solutions\nthrough one point",
            fontsize=13, color=INK, ha="right", va="center", zorder=6)
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.75, 1.75)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_yticks([-1, 0, 1])
    ax.set_xlabel(r"$t$", fontsize=15)
    ax.set_ylabel(r"$x$", fontsize=15, labelpad=-2)
    ax.set_box_aspect(1)
    _spines(ax)

    # (b) one residual step x -> x + f(x), f(x) = -2x ---------------------------
    ax = axes[1]
    pts = np.array([-1.0, -0.5, 0.5, 1.0])
    top, bot = 1.0, 0.0
    for y in (top, bot):
        ax.plot([-1.35, 1.35], [y, y], color="black", lw=1.0, zorder=1)
    cols = [BLUE, ORANGE, GREEN, fl.T.PURPLE.base]
    for p, c in zip(pts, cols):
        ax.plot([p], [top], "o", color=c, ms=7, zorder=4)
        ax.plot([-p], [bot], "o", color=c, ms=7, zorder=4)
        fl.arrow(ax, (p, top - 0.04), (-p, bot + 0.05), color=c, lw=1.8,
                 mut=13)
    for p in pts:
        ax.text(p, top + 0.11, f"${p:g}$", ha="center", va="bottom",
                fontsize=13, color=INK)
        ax.text(p, bot - 0.11, f"${-p:g}$", ha="center", va="top",
                fontsize=13, color=INK)
    ax.text(-1.42, top, r"$x$", ha="right", va="center", fontsize=15,
            color=INK)
    ax.text(-1.42, bot, r"$x+f(x)$", ha="right", va="center", fontsize=15,
            color=INK)
    ax.text(0.0, -0.55, r"$f(x)=-2x$:  $h=1$, $hL=2$", ha="center",
            va="center", fontsize=14, color=INK)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.75, 1.55)
    ax.set_box_aspect(1)
    ax.axis("off")

    # (c) the plane: rotation by pi -----------------------------------------
    ax = axes[2]
    s = np.linspace(0, 1, 200)
    # same colour per starting point as in (b)
    for p, c in zip(pts, cols):
        start = np.array([p, 0.0])
        path = np.array([_expm(np.pi * K, si) @ start for si in s])
        ax.plot(path[:, 0], path[:, 1], color=c, lw=2.0, zorder=3)
        i = len(s) // 2
        fl.arrow(ax, path[i - 4], path[i + 4], color=c, lw=0.0, mut=16)
        ax.plot([p], [0], "o", color=c, ms=6, zorder=4)
    ax.annotate("", xy=(1.38, 0), xytext=(-1.38, 0),
                arrowprops=dict(arrowstyle="->", color="black", lw=0.9))
    ax.text(0.0, 1.22, r"$\Phi_1=e^{\pi K}=-\mathbf{I}$", ha="center",
            va="bottom", fontsize=14, color=INK)
    # the outer points are labelled; the inner ones (+-0.5) share colours with (b)
    for p, ha in ((-1.0, "right"), (1.0, "left")):
        ax.text(1.07 * p, -0.05, f"${p:g}$", ha=ha, va="top", fontsize=13,
                color=INK, zorder=5)
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.25, 1.55)
    ax.set_aspect("equal")
    ax.axis("off")

    for ax, lab in zip(axes, "abc"):
        ax.text(0.0, 1.03, f"({lab})", transform=ax.transAxes, ha="left",
                va="bottom", fontsize=15, color=INK)
    save(fig, "mdl-dyn-flow-order")
    _copy_to_auto("mdl-dyn-flow-order")


def fig_adjoint_timeline():
    """dx/dt = theta x at theta = 1 with x(0) = 1, T = 1 and L = x(1)^2 / 2:
    the state grows forward, the adjoint a(t) = e^{2-t} is integrated backward
    from a(1) = x(1) = e, their product is the constant e^2, and the parameter
    gradient is the shaded area under it."""
    t = np.linspace(0, 1, 300)
    x = np.exp(t)
    a = np.exp(2 - t)
    e2 = np.e ** 2
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    ax.fill_between(t, 0, e2, color=BLUE_TINT, zorder=0)
    ax.plot(t, x, color=BLUE, lw=2.4, zorder=3)
    ax.plot(t, a, color=ORANGE, lw=2.4, zorder=3)
    ax.plot(t, np.full_like(t, e2), color="black", lw=2.0, zorder=3)
    # direction of integration: forward for x, backward for a
    fl.arrow(ax, (0.47, np.exp(0.47)), (0.55, np.exp(0.55)), color=BLUE,
             lw=0.0, mut=20)
    fl.arrow(ax, (0.55, np.exp(2 - 0.55)), (0.47, np.exp(2 - 0.47)),
             color=ORANGE, lw=0.0, mut=20)
    ax.text(0.60, 1.30, r"$x(t)=e^{t}$, forward", fontsize=14, color=INK,
            ha="left", va="center")
    ax.text(0.04, 3.55, "$a(t)=e^{2-t}$, backward\nfrom $a(1)=x(1)=e$",
            fontsize=14, color=INK, ha="left", va="center")
    ax.text(0.03, e2 + 0.32, r"$a(t)\,x(t)=e^2$", fontsize=14, color=INK,
            ha="left", va="center")
    ax.text(0.73, 5.75,
            r"$\dfrac{\partial\mathcal{L}}{\partial\theta}"
            r"=\int_0^1 a\,x\,dt=e^2$",
            fontsize=15, color=INK, ha="center", va="center", bbox=WHITE_BOX,
            zorder=6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 8.6)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_yticks([1, np.e, e2])
    ax.set_yticklabels([r"$1$", r"$e$", r"$e^2$"], fontsize=14)
    ax.set_xlabel(r"$t$", fontsize=15)
    _spines(ax)
    save(fig, "mdl-dyn-adjoint-timeline")
    _copy_to_auto("mdl-dyn-adjoint-timeline")


def fig_divergence_volume():
    """A disc of radius 0.4 centred at (1.2, 0), carried by the exact flow
    e^{At} of three linear fields to t = 0, 0.6, 1.2: the spiral
    (tr A = -1, area x e^{-t}), a saddle (tr A = 0, area constant) and an
    expanding spiral (tr A = +1, area x e^{t})."""
    I2 = np.eye(2)
    cases = [
        (-0.5 * I2 + K, r"$\mathrm{tr}\,A=-1$: area $\times\, e^{-t}$"),
        (np.diag([0.5, -0.5]), r"$\mathrm{tr}\,A=0$: area constant"),
        (0.5 * I2 + K, r"$\mathrm{tr}\,A=+1$: area $\times\, e^{t}$"),
    ]
    times = (0.0, 0.6, 1.2)
    shades = (fl.T.BLUE.tint, "#8DBBDC", BLUE)
    edges = ("#7FAFD3", "#2F84C0", fl.T.BLUE.dark)
    c0, r0 = np.array([1.2, 0.0]), 0.4
    phi = np.linspace(0, 2 * np.pi, 200)
    disc = c0[:, None] + r0 * np.stack([np.cos(phi), np.sin(phi)])
    xlim, ylim = (-0.6, 3.05), (-1.1, 2.95)
    gx = np.linspace(xlim[0] + 0.2, xlim[1] - 0.2, 10)
    gy = np.linspace(ylim[0] + 0.2, ylim[1] - 0.2, 11)
    GX, GY = np.meshgrid(gx, gy)
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.3),
                             gridspec_kw=dict(wspace=0.08))
    for ax, (A, title) in zip(axes, cases):
        U = A[0, 0] * GX + A[0, 1] * GY
        V = A[1, 0] * GX + A[1, 1] * GY
        nrm = np.hypot(U, V)
        nrm[nrm == 0] = 1.0
        ax.quiver(GX, GY, U / nrm, V / nrm, color=FAINT, alpha=0.55,
                  angles="xy", pivot="mid", width=0.005, headwidth=4,
                  headlength=5, scale=22, zorder=1)
        path = np.array([_expm(A, s) @ c0 for s in np.linspace(0, 1.2, 100)])
        ax.plot(path[:, 0], path[:, 1], color=INK, lw=1.0, ls=":", zorder=2)
        for t, fc, ec in zip(times, shades, edges):
            pts = _expm(A, t) @ disc
            ax.fill(pts[0], pts[1], facecolor=fc, edgecolor=ec, lw=1.8,
                    alpha=0.85, zorder=3)
        ax.plot([0], [0], "o", color="black", ms=4, zorder=4)
        ax.set_title(title, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        fl.clean_axes(ax, lim=(xlim, ylim), equal=True)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
    save(fig, "mdl-dyn-divergence-volume")
    _copy_to_auto("mdl-dyn-divergence-volume")


def main():
    fig_flow_order()
    fig_adjoint_timeline()
    fig_divergence_volume()
    for p in fl.WRITTEN:
        print("wrote", os.path.relpath(p))


if __name__ == "__main__":
    main()
