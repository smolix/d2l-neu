#!/usr/bin/env python3
"""Generate the new illustrative figure of "Duality"
(``chapter_mdl-optimization/mdl-lagrangian-duality.md``) in the chapter's
shared house style.

The shared style lives in ``gen_mdl_figures.py``; importing it applies the
``plt.rcParams`` (fixed ``svg.hashsalt`` and ``metadata={'Date': None}`` in
``save()`` make re-runs byte-for-byte identical) and exposes the palette.  The
section's other two figures (``mdl-opt-primal-dual-gap`` and
``mdl-opt-water-filling``) stay in ``gen_mdl_optimization_figures.py``.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_optimization_lagrangian_duality_figures.py

Writes ``img/mdl-opt-conjugate-intercept.svg`` (and the copy in ``img/auto/``
that the slides read).
"""

from __future__ import annotations

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, INK = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.INK

LBL = 13          # the chapter's in-figure label size


def _copy_to_auto(name):
    shutil.copyfile(os.path.join(fl.IMG_DIR, f"{name}.svg"),
                    os.path.join(fl.IMG_DIR, "auto", f"{name}.svg"))


def fig_conjugate_intercept():
    """The convex conjugate as a supporting line.

    (a) f(x) = x^2/2 and slope y = 2: the dashed line t = 2x + 1/2 crosses the
        graph; lowering it until it touches gives the tangent t = 2x - 2 at
        (2, 2), whose intercept -2 is -f*(2), so f*(2) = 2.
    (b) f(x) = |x|: the line of slope 1/2 through the origin touches the V at
        0, so f*(1/2) = 0; a line of slope 2 runs above the graph for large x
        whatever its intercept, so f*(2) = +infinity.

    Both curves are the exact functions; the touching line in (a) is the exact
    tangent.  Both panels share one height (same figure row, equal=False with
    windows sized so the curves fill them).
    """
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.4, 4.3))

    # ======================= (a) f = x^2 / 2 ======================= #
    xr, yr = (-1.6, 3.4), (-2.6, 4.2)
    xs = np.linspace(-1.55, 2.9, 400)
    axa.plot(xs, 0.5 * xs ** 2, color=BLUE, lw=2.2, zorder=3)
    # the dashed parallel line before lowering: t = 2x + 1/2 (crosses the graph)
    xl = np.linspace(-1.2, 1.85, 2)
    axa.plot(xl, 2 * xl + 0.5, color=GRAY, lw=1.8, ls=(0, (5, 3)), zorder=2)
    # the touching line t = 2x - 2 (tangent at x = 2)
    xt = np.linspace(-0.3, 3.1, 2)
    axa.plot(xt, 2 * xt - 2, color=GREEN, lw=2.2, zorder=4)
    axa.plot(2, 2, "o", color="black", ms=6.5, zorder=6)
    axa.plot(0, -2, "o", color="black", ms=6.5, zorder=6)
    # "lower until it touches": arrow between the two parallel lines at x = 0.75
    xa = 0.75
    fl.arrow(axa, (xa, 2 * xa + 0.5 - 0.12), (xa, 2 * xa - 2 + 0.12),
             color=GRAY, lw=1.6, mut=12)
    # label in the empty wedge below-right of the touching line
    axa.text(1.2, -0.95, "lower until\nit touches", color="black",
             fontsize=LBL, ha="left", va="center")
    axa.text(0.22, -2.2, r"$-f^*(2)=-2$", color="black", fontsize=LBL,
             ha="left", va="center")
    axa.text(2.3, 1.62, r"touches at $x=2$", color="black", fontsize=LBL,
             ha="left", va="top")
    axa.text(-1.45, 3.3, r"$f(x)=\frac{1}{2}x^2$", color=BLUE,
             fontsize=LBL + 1, ha="left", va="center")
    axa.text(3.4, 2.05, r"slope $y=2$", color=GREEN, fontsize=LBL,
             ha="right", va="center")
    fl.axis_cross(axa, xr, yr, color="black")
    axa.text(xr[1], -0.12, r"$x$", color="black", fontsize=LBL + 1,
             ha="right", va="top")
    axa.text(0.1, yr[1], r"$t$", color="black", fontsize=LBL + 1,
             ha="left", va="top")
    fl.clean_axes(axa, lim=(xr, yr), hide=True, equal=False)

    # ========================== (b) f = |x| ========================== #
    xr, yr = (-2.5, 2.6), (-1.45, 3.2)
    xs = np.linspace(-2.4, 2.4, 3)
    axb.plot(xs, np.abs(xs), color=BLUE, lw=2.2, zorder=3)
    # slope 1/2 through the origin: a supporting line, touches at 0
    xh = np.linspace(-2.4, 2.4, 2)
    axb.plot(xh, 0.5 * xh, color=GREEN, lw=2.2, zorder=4)
    axb.plot(0, 0, "o", color="black", ms=6.5, zorder=6)
    # slope 2 through (0, -1): crosses the V at x = 1 and stays above for x > 1
    xq = np.linspace(-0.1, 2.05, 2)
    axb.plot(xq, 2 * xq - 1, color=GRAY, lw=1.8, ls=(0, (5, 3)), zorder=2)
    axb.text(-2.35, 2.95, r"$f(x)=|x|$", color=BLUE, fontsize=LBL + 1,
             ha="left", va="center")
    # in the wedge above the x-axis and below the left branch of the V
    axb.text(-2.4, 0.4, r"slope $\frac{1}{2}$: $f^*(\frac{1}{2})=0$",
             color=GREEN, fontsize=LBL, ha="left", va="center")
    # below the x-axis, right of the dashed line
    axb.text(2.55, -0.2, "slope $2$: no intercept\n"
             "keeps it below the\n"
             r"graph, $f^*(2)=+\infty$", color="black", fontsize=LBL,
             ha="right", va="top")
    fl.axis_cross(axb, xr, yr, color="black")
    axb.text(xr[1], 0.33, r"$x$", color="black", fontsize=LBL + 1,
             ha="right", va="bottom")
    axb.text(0.1, yr[1], r"$t$", color="black", fontsize=LBL + 1,
             ha="left", va="top")
    fl.clean_axes(axb, lim=(xr, yr), hide=True, equal=False)

    fig.subplots_adjust(wspace=0.12)
    fl.save(fig, "mdl-opt-conjugate-intercept")
    _copy_to_auto("mdl-opt-conjugate-intercept")


FIGURES = [fig_conjugate_intercept]


def main():
    fl.WRITTEN.clear()
    for fn in FIGURES:
        fn()
    for path in fl.WRITTEN:
        print(f"wrote {path} ({os.path.getsize(path):,d} bytes)")


if __name__ == "__main__":
    main()
