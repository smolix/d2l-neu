#!/usr/bin/env python3
"""Generate the figures added to "Convexity"
(``chapter_mdl-optimization/mdl-convexity.md``) in the shared house style of
``gen_mdl_figures.py``.

* ``mdl-opt-convexity-hierarchy`` -- (a) convex, strictly convex, strongly
  convex: f(x) = x, x^4, x^2; (b) the sandwich of a mu-strongly convex,
  L-smooth function f(x) = x^2/2 + 3 log cosh x (f'' in [1, 4]) between two
  parabolas tangent at x0 = 1.
* ``mdl-opt-factored-model`` -- contours of the two-weight model (ab - 1)^2,
  its hyperbola of minimizers, two minimizers and their midpoint (the saddle).

Every curve is computed from its formula.  Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_optimization_convexity_figures.py

The script is byte-idempotent (the shared ``save()`` fixes the SVG hash salt
and drops the date).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY
LBL = 13  # the chapter's in-figure label size (gen_mdl_optimization_figures)
WHITE = dict(facecolor="white", edgecolor="none", pad=1.5)


def fig_convexity_hierarchy():
    """(a) x (chords on the graph), x^4 (strict, flat at 0), x^2 (curved
    everywhere), drawn without an axis cross.  (b) f = x^2/2 + 3 log cosh x between the parabolas of
    curvature mu = 1 and L = 4 tangent at x0 = 1, with the tangent line."""
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.0, 4.2))

    # ----- (a) three functions, three strengths of convexity -----
    xs = np.linspace(-1.25, 1.25, 400)
    axa.plot(xs, xs, color=GREEN, lw=2.2, zorder=3)
    axa.plot(xs, xs**4, color=ORANGE, lw=2.2, zorder=3)
    axa.plot(xs, xs**2, color=BLUE, lw=2.2, zorder=3)
    # a chord of x lies on its graph: its two end points are marked
    axa.plot([-0.9, 0.6], [-0.9, 0.6], "o", color=GREEN, ms=6, zorder=5)
    # labels beyond the ends of the curves (x^4 leaves the top at the left)
    axa.text(1.30, 1.25, "$x$", color=GREEN, fontsize=LBL + 1, ha="left",
             va="center")
    axa.text(1.30, 1.58, "$x^2$", color=BLUE, fontsize=LBL + 1, ha="left",
             va="center")
    axa.text(-1.13, 2.35, "$x^4$", color=ORANGE, fontsize=LBL + 1,
             ha="left", va="center")
    # no axis cross: x^4 hugs the horizontal axis near 0 and would vanish
    # into it; the three graphs carry the comparison
    fl.clean_axes(axa, lim=((-1.4, 1.65), (-1.3, 2.5)), hide=True,
                  equal=False)

    # ----- (b) the mu / L sandwich -----
    f = lambda t: 0.5 * t**2 + 3 * np.log(np.cosh(t))
    fp = lambda t: t + 3 * np.tanh(t)
    x0 = 1.0
    ts = np.linspace(-1.0, 3.0, 400)
    tan = f(x0) + fp(x0) * (ts - x0)
    lower = tan + 0.5 * (ts - x0)**2          # curvature mu = 1
    upper = tan + 2.0 * (ts - x0)**2          # curvature L = 4
    ytop, ybot = 12.0, -3.2
    for yy, col, ls in ((upper, ORANGE, "--"), (lower, GREEN, "--"),
                        (tan, GRAY, "-")):
        keep = (yy <= ytop) & (yy >= ybot)
        axb.plot(ts[keep], yy[keep], ls, color=col, lw=1.8, zorder=2)
    axb.plot(ts, f(ts), color=BLUE, lw=2.4, zorder=3)
    axb.plot(x0, f(x0), "o", color="black", ms=6, zorder=5)
    axb.text(x0 + 0.08, f(x0) - 0.55, "$x_0$", color="black",
             fontsize=LBL + 1, ha="left", va="top")
    # labels in the gaps between the curves, outward from the tangency point
    axb.text(2.02, 11.3, r"$+\frac{L}{2}(x-x_0)^2$", color=ORANGE,
             fontsize=LBL, ha="right", va="center", bbox=WHITE)
    axb.text(3.05, 10.3, r"$+\frac{\mu}{2}(x-x_0)^2$", color=GREEN,
             fontsize=LBL, ha="left", va="center")
    axb.text(3.05, 11.6, "$f$", color=BLUE, fontsize=LBL + 1, ha="left",
             va="center")
    axb.text(2.35, 5.0, "tangent", color=GRAY, fontsize=LBL, ha="left",
             va="center")
    fl.axis_cross(axb, (-1.1, 3.1), (ybot, ytop + 0.4), color="black")
    fl.clean_axes(axb, lim=((-1.1, 4.3), (ybot - 0.1, ytop + 0.5)), hide=True,
                  equal=False)

    fl.save(fig, "mdl-opt-convexity-hierarchy")


def fig_factored_model():
    """Contours of f(a, b) = (ab - 1)^2; the minimizer set ab = 1 (both
    branches); the minimizers (1, 1), (-1, -1) and their midpoint, the origin,
    where f = 1 and the Hessian has eigenvalues +-2 (a saddle)."""
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    lim = 2.2
    g = np.linspace(-lim, lim, 601)
    A, B = np.meshgrid(g, g)
    F = (A * B - 1.0)**2
    ax.contour(A, B, F, levels=[0.1, 0.5, 1.0, 2.0, 4.0], colors=[GRAY],
               linewidths=0.9, zorder=1)
    # the hyperbola ab = 1, both branches, drawn heavy
    for sgn in (1.0, -1.0):
        a = sgn * np.linspace(1 / lim, lim, 300)
        ax.plot(a, 1.0 / a, color=BLUE, lw=2.8, zorder=3)
    ax.plot([1, -1], [1, -1], "--", color=ORANGE, lw=1.8, zorder=4)
    ax.plot([1, -1], [1, -1], "o", color=BLUE, ms=8, zorder=5)
    ax.plot(0, 0, "o", color=ORANGE, ms=8, zorder=6)
    ax.text(1.12, 0.90, "$(1,1)$", color="black", fontsize=LBL, ha="left",
            va="top", bbox=WHITE)
    ax.text(-1.12, -0.90, "$(-1,-1)$", color="black", fontsize=LBL,
            ha="right", va="bottom", bbox=WHITE)
    # the two notes on the origin sit in the empty wedges next to it, clear
    # of the dashed segment (quadrants 1, 3) and of the contours ab = -0.41
    ax.text(0.12, -0.10, "saddle", color="black", fontsize=LBL, ha="left",
            va="top")
    ax.text(-0.12, 0.10, "$f=1$", color=ORANGE, fontsize=LBL, ha="right",
            va="bottom")
    ax.text(1.42, 0.98, "$ab=1$", color=BLUE, fontsize=LBL + 1, ha="left",
            va="bottom", bbox=WHITE)
    ax.text(lim - 0.02, -0.14, "$a$", color="black", fontsize=LBL + 1,
            ha="right", va="top")
    ax.text(-0.12, lim - 0.02, "$b$", color="black", fontsize=LBL + 1,
            ha="right", va="top")
    fl.axis_cross(ax, (-lim, lim), (-lim, lim), color="black")
    fl.clean_axes(ax, lim=((-lim, lim), (-lim, lim)), hide=True)
    fl.save(fig, "mdl-opt-factored-model")


FIGURES = [
    fig_convexity_hierarchy,
    fig_factored_model,
]


def main():
    start = len(fl.WRITTEN)
    for fn in FIGURES:
        fn()
    written = fl.WRITTEN[start:]
    print(f"\nWrote {len(written)} figures to {fl.IMG_DIR}:")
    for p in written:
        size = os.path.getsize(p)
        assert size > 0, f"empty: {p}"
        with open(p, "r", encoding="utf-8") as fh:
            assert "<svg" in fh.read(400), f"not valid SVG: {p}"
        print(f"  {os.path.basename(p):36s} {size:>8,d} bytes")


if __name__ == "__main__":
    main()
