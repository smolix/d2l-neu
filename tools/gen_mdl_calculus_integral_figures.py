#!/usr/bin/env python3
"""Generate the new illustrative figures of the section "Integral Calculus"
(``chapter_mdl-calculus/mdl-integral-calculus.md``) in the shared house style.

The section's other figures (Riemann sums, signed area, the area subtraction,
the substitution sliver, the bell surface, the grid reordering and the
Jacobian factor) live in ``gen_mdl_calculus_figures.py``.  This script adds:

* ``mdl-cal-ftc-sliver``: the area-so-far F(x) and the strip of width epsilon
  that moving to x + epsilon adds, the picture behind the fundamental theorem;
* ``mdl-cal-polar-cell``: one polar cell with sides dr and r dtheta, the
  picture behind the Jacobian factor r;
* ``mdl-cal-leibniz-boundary``: the two effects in the Leibniz rule for the
  uniform example f(x, theta) = x / theta on [0, theta].

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_calculus_integral_figures.py

The shared ``save()`` fixes ``svg.hashsalt`` and drops the date, so re-runs are
byte-identical.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

from matplotlib.patches import Polygon, Rectangle, Wedge

np, plt = fl.np, fl.plt
BLUE, ORANGE, GRAY, LIGHT, INK = fl.BLUE, fl.ORANGE, fl.GRAY, fl.LIGHT, fl.INK
FS = 14


def fig_ftc_sliver():
    """Area-so-far under f(y) = y/(1+y^2) from a = 0 to x = 1.2 (light), the
    strip [x, x + eps] with eps = 0.3 (darker), and the dashed rectangle of
    height f(x) that approximates the strip."""
    f = lambda t: t / (1 + t ** 2)
    a, x, eps = 0.0, 1.2, 0.3
    ys = np.linspace(0.0, 2.3, 400)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    left = ys[ys <= x]
    ax.fill_between(left, 0, f(left), color=BLUE, alpha=0.22, lw=0)
    strip = np.linspace(x, x + eps, 60)
    ax.fill_between(strip, 0, f(strip), color=ORANGE, alpha=0.45, lw=0)
    ax.plot([x, x, x + eps, x + eps], [0, f(x), f(x), 0], ls="--",
            color=INK, lw=1.2, zorder=4)
    ax.plot(ys, f(ys), color=BLUE, lw=2.4, zorder=5)
    fl.axis_cross(ax, (-0.05, 2.4), (-0.02, 0.72), color="black")
    ax.text(0.80, 0.13, "$F(x)$", ha="center", va="center", fontsize=FS + 2,
            color=INK)
    ax.text(x + eps / 2, f(x) + 0.05, r"$\approx\epsilon\,f(x)$", ha="center",
            va="bottom", fontsize=FS, color=INK)
    ax.text(2.05, f(2.05) + 0.05, "$f$", ha="center", va="bottom",
            fontsize=FS, color=INK)
    for t, lab in [(a, "$a$"), (x, "$x$"), (x + eps, r"$x+\epsilon$")]:
        ax.plot([t, t], [-0.015, 0.015], color="black", lw=1.0)
        ax.text(t, -0.035, lab, ha="center", va="top", fontsize=FS, color=INK)
    ax.set_xlim(-0.15, 2.45)
    ax.set_ylim(-0.13, 0.74)
    ax.axis("off")
    fl.save(fig, "mdl-cal-ftc-sliver")


def fig_polar_cell():
    """A faint polar grid and one filled annular sector at r = 1, theta = 40 deg,
    dr = 0.3, dtheta = 20 deg, with its side lengths dr and r dtheta."""
    r0, dr = 1.0, 0.3
    th0, dth = 40.0, 20.0
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    for r in [0.5, 1.0, 1.5, 2.0]:
        t = np.linspace(0, np.pi / 2, 200)
        ax.plot(r * np.cos(t), r * np.sin(t), color=LIGHT, lw=0.9, zorder=1)
    for deg in range(0, 91, 20):
        t = np.deg2rad(deg)
        ax.plot([0, 2.0 * np.cos(t)], [0, 2.0 * np.sin(t)], color=LIGHT,
                lw=0.9, zorder=1)
    ax.add_patch(Wedge((0, 0), r0 + dr, th0, th0 + dth, width=dr,
                       facecolor=ORANGE, alpha=0.45, edgecolor=ORANGE, lw=1.6,
                       zorder=3))
    fl.axis_cross(ax, (-0.08, 2.25), (-0.08, 2.25), color="black")
    ax.text(2.22, -0.06, "$x$", ha="center", va="top", fontsize=FS, color=INK)
    ax.text(-0.06, 2.22, "$y$", ha="right", va="center", fontsize=FS,
            color=INK)
    # dr on the lower radial side, offset below-right (outward normal of that side)
    t1 = np.deg2rad(th0)
    rm = r0 + dr / 2
    nrm = np.array([np.sin(t1), -np.cos(t1)])
    p = rm * np.array([np.cos(t1), np.sin(t1)]) + 0.11 * nrm
    ax.text(p[0], p[1], "$dr$", ha="center", va="center", fontsize=FS,
            color=INK)
    # r dtheta on the outer arc, offset radially outward
    tm = np.deg2rad(th0 + dth / 2)
    q = (r0 + dr + 0.20) * np.array([np.cos(tm), np.sin(tm)])
    ax.text(q[0], q[1], r"$r\,d\theta$", ha="center", va="center",
            fontsize=FS, color=INK)
    # the area, in the empty region below the grid's outer arc near the x-axis
    ax.text(1.55, 0.30, r"area $\approx r\,dr\,d\theta$", ha="center",
            va="center", fontsize=FS, color=INK,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    # the angle theta at the origin
    t = np.linspace(0, t1, 60)
    ax.plot(0.32 * np.cos(t), 0.32 * np.sin(t), color=INK, lw=1.0)
    ax.text(0.42 * np.cos(t1 / 2), 0.42 * np.sin(t1 / 2), r"$\theta$",
            ha="center", va="center", fontsize=FS, color=INK)
    ax.set_xlim(-0.2, 2.35)
    ax.set_ylim(-0.2, 2.35)
    ax.set_aspect("equal")
    ax.axis("off")
    fl.save(fig, "mdl-cal-polar-cell")


def fig_leibniz_boundary():
    """Leibniz rule for f(x, theta) = x/theta on [0, theta] at theta = 2 with
    d theta = 0.5.  The area under f(., theta) is shaded; the curve
    f(., theta + d theta) is dashed, the band between the curves (lost area,
    the swapped-integral term) is hatched; the strip [theta, theta + d theta] of
    height f(b, theta) = 1 is the boundary term."""
    th, dth = 2.0, 0.5
    f = lambda t, s: t / s
    xs = np.linspace(0, th, 200)
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.fill_between(xs, 0, f(xs, th + dth), color=BLUE, alpha=0.22, lw=0)
    ax.fill_between(xs, f(xs, th + dth), f(xs, th), facecolor="none",
                    edgecolor=GRAY, hatch="////", lw=0)
    ax.add_patch(Rectangle((th, 0), dth, f(th, th), facecolor=ORANGE,
                           alpha=0.45, edgecolor=ORANGE, lw=1.4, zorder=3))
    ax.plot(xs, f(xs, th), color=BLUE, lw=2.4, zorder=5)
    xs2 = np.linspace(0, th + dth, 200)
    ax.plot(xs2, f(xs2, th + dth), color=BLUE, lw=1.8, ls="--", zorder=5)
    fl.axis_cross(ax, (-0.05, 3.25), (-0.03, 1.32), color="black")
    ax.text(0.95, 0.60, r"$f(\cdot,\theta)$", ha="right", va="bottom",
            fontsize=FS, color=INK)
    ax.text(th + dth + 0.08, 1.0, r"$f(\cdot,\theta+d\theta)$", ha="left",
            va="center", fontsize=FS, color=INK)
    ax.text(th + dth + 0.08, 0.45, r"$f(b,\theta)\,b'\,d\theta$", ha="left",
            va="center", fontsize=FS, color=INK)
    # the hatched band (the swapped integral), labelled in the empty upper
    # left with a leader to the band's middle
    ax.annotate(r"$d\theta\;\int\;\partial_\theta f\,dx$", xy=(1.35, 0.60),
                xytext=(0.70, 1.12), fontsize=FS, color=INK, ha="center",
                va="center",
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.9,
                                shrinkA=4, shrinkB=1))
    for t, lab in [(0.0, "$0$"), (th, r"$\theta$"),
                   (th + dth, r"$\theta+d\theta$")]:
        ax.plot([t, t], [-0.02, 0.02], color="black", lw=1.0)
        ax.text(t, -0.05, lab, ha="center", va="top", fontsize=FS, color=INK)
    ax.set_xlim(-0.2, 3.6)
    ax.set_ylim(-0.2, 1.32)
    ax.axis("off")
    fl.save(fig, "mdl-cal-leibniz-boundary")


FIGURES = [fig_ftc_sliver, fig_polar_cell, fig_leibniz_boundary]


def main():
    start = len(fl.WRITTEN)
    for fn in FIGURES:
        fn()
    for p in fl.WRITTEN[start:]:
        print(f"  {os.path.basename(p):30s} {os.path.getsize(p):>8,d} bytes")


if __name__ == "__main__":
    main()
