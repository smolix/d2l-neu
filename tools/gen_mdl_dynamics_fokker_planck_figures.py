#!/usr/bin/env python3
"""Generate the new illustrative figures of "The Fokker--Planck Equation and
Probability Flow" (``chapter_mdl-dynamics/mdl-fokker-planck-probability-flow.md``)
in the chapter's shared house style.

The shared style lives in ``gen_mdl_figures.py``; importing it applies the
``plt.rcParams`` (fixed ``svg.hashsalt`` and ``metadata={'Date': None}`` in
``save()`` make re-runs byte-for-byte identical) and exposes the palette.  The
section's older figures (``mdl-dyn-fokker-planck-flux``,
``mdl-dyn-forward-reverse``, ``mdl-dyn-score-field``,
``mdl-dyn-lambda-family``) stay in ``gen_mdl_dynamics_figures.py``.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_dynamics_fokker_planck_figures.py

Writes ``img/mdl-dyn-divergence-boxes.svg`` and
``img/mdl-dyn-diffusion-outward-flow.svg``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, INK = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.INK
FAINT = fl.T.FAINT
WHITE_BOX = dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5)
save = fl.save


def _gauss(x, mu, var):
    return np.exp(-0.5 * (x - mu) ** 2 / var) / np.sqrt(2 * np.pi * var)


def fig_divergence_boxes():
    """Two square panels at the same scale: (a) the expanding field
    u = (x1, x2), divergence 2, and (b) the rotation u = (-x2, x1), divergence
    0, each with the same small box centred at (0.6, 0.4).  At the midpoint of
    every side an arrow outside the box shows the normal component u.n
    (pointing away from the side for outflow, onto it for inflow), with length
    proportional to it.  In
    (a) outflow through the right and top exceeds inflow through the left and
    bottom by 2 x area; in (b) inflow and outflow balance."""
    lim = 1.5
    g = np.linspace(-1.25, 1.25, 6)
    X1, X2 = np.meshgrid(g, g)
    cx, cy, h = 0.6, 0.4, 0.25           # box centre and half-side
    fields = [
        (lambda x1, x2: (x1, x2),
         r"$\mathbf{u}=(x_1,x_2)$" "\n" r"$\nabla\cdot\mathbf{u}=2$: net outflow $2\times$area"),
        (lambda x1, x2: (-x2, x1),
         r"$\mathbf{u}=(-x_2,x_1)$" "\n" r"$\nabla\cdot\mathbf{u}=0$: inflow $=$ outflow"),
    ]
    sides = [((cx + h, cy), (1, 0)), ((cx - h, cy), (-1, 0)),
             ((cx, cy + h), (0, 1)), ((cx, cy - h), (0, -1))]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.3),
                             gridspec_kw=dict(wspace=0.12))
    for ax, (u, title) in zip(axes, fields):
        U1, U2 = u(X1, X2)
        keep = (np.abs(X1 - cx) > h + 0.2) | (np.abs(X2 - cy) > h + 0.2)
        ax.quiver(X1[keep], X2[keep], U1[keep], U2[keep], color=FAINT, angles="xy", scale_units="xy",
                  scale=4.0, width=0.006, zorder=1)
        fl.axis_cross(ax, (-lim, lim), (-lim, lim), color="black")
        ax.add_patch(plt.Rectangle((cx - h, cy - h), 2 * h, 2 * h,
                                   facecolor=ORANGE, alpha=0.18, lw=0,
                                   zorder=2))
        ax.add_patch(plt.Rectangle((cx - h, cy - h), 2 * h, 2 * h,
                                   facecolor="none", edgecolor=ORANGE,
                                   lw=2.0, zorder=2))
        for (px, py), (nx, ny) in sides:
            ux, uy = u(px, py)
            flux = ux * nx + uy * ny     # normal component u.n
            L = 0.7 * abs(flux)           # arrows stay outside the box
            out = (px + L * nx, py + L * ny)
            tail, tip = ((px, py), out) if flux > 0 else (out, (px, py))
            fl.arrow(ax, tail, tip, color=GREEN, lw=2.4, mut=13)
        ax.set_title(title, fontsize=14, color=INK, linespacing=1.3)
        fl.clean_axes(ax, lim=((-lim, lim), (-lim, lim)), hide=True)
    save(fig, "mdl-dyn-divergence-boxes")


def fig_diffusion_outward_flow():
    """The heat equation dp/dt = (1/2) g^2 p'' with g = 1 from p_0 = N(0, 1),
    solved as a flow.  (a) p_0 (solid) and p_3 = N(0, 4) (faint), with the
    probability-flow velocity u_0(x) = x / 2 drawn as arrows on the axis
    (outward, proportional to distance); the particle at x = 1 lands at x = 2
    by t = 3, and its density halves.  (b) The (t, x) plane: PF-ODE
    trajectories x(0) sqrt(1 + t), smooth and never crossing, against three
    Brownian paths from x = 1 (seeded; figures are fixed artwork) with the
    one-standard-deviation envelope 1 +- sqrt(t)."""
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.4, 3.9),
                                   gridspec_kw=dict(wspace=0.22))
    # (a) densities and the outward velocity field
    x = np.linspace(-3.6, 3.6, 600)
    p0, p3 = _gauss(x, 0, 1), _gauss(x, 0, 4)
    axA.plot(x, p3, color=BLUE, lw=1.6, alpha=0.45, zorder=2)
    axA.fill_between(x, 0, p3, color=BLUE, alpha=0.06, zorder=1)
    axA.plot(x, p0, color=BLUE, lw=2.4, zorder=3)
    for x0 in (-1.8, -1.2, -0.6, 0.6, 1.2, 1.8):   # length 0.8 * u_0(x0)
        fl.arrow(axA, (x0, 0.014), (x0 + 0.8 * x0 / 2, 0.014), color=GREEN,
                 lw=2.2, mut=11)
    axA.plot([1, 2], [_gauss(1, 0, 1), _gauss(2, 0, 4)], "o", color=ORANGE,
             ms=6, zorder=5)
    fl.arrow(axA, (1.06, _gauss(1, 0, 1) - 0.006), (1.93, _gauss(2, 0, 4) + 0.006),
             color=ORANGE, lw=1.6, ls=":", mut=12)
    axA.text(2.0, 0.20, r"$\times2$ by $t=3$", color=ORANGE, fontsize=13.5,
             ha="left", va="center")
    axA.text(-0.55, 0.415, r"$t=0$", color=BLUE, fontsize=13.5, ha="right",
             va="center")
    axA.text(-2.2, 0.165, r"$t=3$", color=BLUE, fontsize=13.5, ha="right",
             va="center", alpha=0.8)
    axA.text(3.55, 0.43, r"$u_t(x)=\dfrac{g^2}{2v_t}\,x$", color=GREEN,
             fontsize=14, ha="right", va="center")
    axA.text(3.55, 0.335, "outward,\n" + r"$\propto$ distance", color=GREEN,
             fontsize=13, ha="right", va="center", linespacing=1.15)
    axA.set_xlim(-3.6, 3.6)
    axA.set_ylim(0, 0.48)
    axA.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    axA.set_yticks([0, 0.2, 0.4])
    axA.set_xlabel("$x$")
    axA.set_ylabel("density")
    fl.clean_axes(axA, equal=False)

    # (b) trajectories in the (t, x) plane
    T, n = 3.0, 600
    t = np.linspace(0, T, n + 1)
    rng = np.random.default_rng(3)
    for _ in range(3):
        w = np.concatenate([[0.0], np.cumsum(np.sqrt(T / n) * rng.standard_normal(n))])
        axB.plot(t, 1 + w, color=GRAY, lw=0.9, alpha=0.9, zorder=2)
    axB.plot(t, 1 + np.sqrt(t), color=INK, lw=1.0, ls="--", zorder=3)
    axB.plot(t, 1 - np.sqrt(t), color=INK, lw=1.0, ls="--", zorder=3)
    for x0 in (-2, -1, -0.5, 0.5, 1, 2):
        axB.plot(t, x0 * np.sqrt(1 + t), color=BLUE, lw=2.0, zorder=4)
    axB.text(0.08, 3.95, r"ODE: $x(0)\sqrt{1+t}$", color=BLUE, fontsize=13.5,
             ha="left", va="center", bbox=WHITE_BOX, zorder=6)
    axB.text(T + 0.08, 1 + np.sqrt(T), r"$1+\sqrt{3}$", color=INK, fontsize=13,
             ha="left", va="center", clip_on=False)
    axB.text(T + 0.08, 1 - np.sqrt(T), r"$1-\sqrt{3}$", color=INK, fontsize=13,
             ha="left", va="center", clip_on=False)
    axB.text(T + 0.08, 1.0, "Brownian\nfrom 1", color=GRAY, fontsize=13,
             ha="left", va="center", clip_on=False, linespacing=1.1)
    axB.set_xlim(0, T)
    axB.set_ylim(-4.3, 4.6)
    axB.set_xticks([0, 1, 2, 3])
    axB.set_xlabel("$t$")
    axB.set_ylabel("$x$")
    fl.clean_axes(axB, equal=False)
    save(fig, "mdl-dyn-diffusion-outward-flow")


FIGURES = [fig_divergence_boxes, fig_diffusion_outward_flow]


def main():
    fl.WRITTEN.clear()
    for fn in FIGURES:
        fn()
    for path in fl.WRITTEN:
        print(f"wrote {path} ({os.path.getsize(path):,d} bytes)")


if __name__ == "__main__":
    main()
