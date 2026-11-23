#!/usr/bin/env python3
"""Generate the new illustrative figure of "Random Variables"
(``chapter_mdl-probability-statistics/mdl-random-variables.md``) in the
chapter's shared house style.

The shared style lives in ``gen_mdl_figures.py``; importing it applies the
``plt.rcParams`` (fixed ``svg.hashsalt`` and ``metadata={'Date': None}`` in
``save()`` make re-runs byte-for-byte identical) and exposes the palette.  The
section's older figures (density, c.d.f., inverse transform, Chebyshev,
marginal, conditional slice, covariance, correlation) stay in
``gen_mdl_probability_figures.py``.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_probability_random_variables_figures.py

Writes ``img/mdl-prob-change-of-variables.svg`` (and the copy in
``img/auto/`` that the slides read), plus the book-only
``img/mdl-prob-cdf-types.svg``, ``img/mdl-prob-markov-step.svg`` and
``img/mdl-prob-dart-three-questions.svg`` (the last one belongs to the
joint-distributions section, as does the change-of-variables figure).
"""

from __future__ import annotations

import math
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, INK = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.INK
LIGHT = fl.LIGHT


def _normal_pdf(x):
    return np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)


def _Phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def fig_change_of_variables():
    """Change of variables for Y = e^X with X ~ N(0,1).  The main panel is the
    graph of g(x) = e^x; the bottom strip is p_X, the left strip is p_Y drawn
    sideways.  The interval [0.9, 1.3] under p_X maps to [e^0.9, e^1.3] =
    [2.46, 3.67] under p_Y; both shaded regions carry the same probability
    0.087, so where g stretches the interval, p_Y is lower by g'(x)."""
    x0, x1 = 0.9, 1.3
    y0, y1 = math.exp(x0), math.exp(x1)
    mass = _Phi(x1) - _Phi(x0)
    xs = np.linspace(-2.2, 1.8, 400)
    ys = np.linspace(1e-3, 6.0, 600)
    p_x = _normal_pdf(xs)
    p_y = _normal_pdf(np.log(ys)) / ys

    fig = plt.figure(figsize=(7.4, 5.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 3.0],
                          height_ratios=[3.0, 1.0], wspace=0.16, hspace=0.08)
    ax = fig.add_subplot(gs[0, 1])
    axb = fig.add_subplot(gs[1, 1], sharex=ax)
    axl = fig.add_subplot(gs[0, 0], sharey=ax)

    # main panel: the map g(x) = e^x and the guides of the interval
    ax.plot(xs, np.exp(xs), color=BLUE, lw=2.4, zorder=3)
    for xv, yv in ((x0, y0), (x1, y1)):
        ax.plot([xv, xv], [0, yv], "--", color=GRAY, lw=1.0, zorder=1)
        ax.plot([-2.2, xv], [yv, yv], "--", color=GRAY, lw=1.0, zorder=1)
    ax.text(1.25, 4.35, r"$y=g(x)=e^{x}$", color=BLUE, fontsize=15,
            ha="right", va="center")
    # the equality of the two shaded areas, stated once in the empty corner
    ax.text(-2.05, 5.85, "both shaded areas:\n" + rf"$P(0.9<X<1.3)={mass:.3f}$",
            ha="left", va="top", fontsize=13, color=INK, linespacing=1.5)
    ax.set_xlim(-2.2, 1.8)
    ax.set_ylim(0, 6.0)
    ax.set_yticks([0, 2, 4, 6])
    # the y name sits above the axis, clear of the sideways density
    ax.text(-2.2, 6.2, r"$y$", fontsize=14, ha="center", va="bottom",
            clip_on=False)
    ax.tick_params(labelbottom=False, bottom=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # bottom strip: p_X with the interval [x0, x1] shaded
    axb.plot(xs, p_x, color=ORANGE, lw=2.2)
    m = (xs >= x0) & (xs <= x1)
    axb.fill_between(xs[m], 0, p_x[m], color=ORANGE, alpha=0.35, lw=0)
    axb.annotate("", xy=(x1, 0.30), xytext=(x0, 0.30),
                 arrowprops=dict(arrowstyle="<->", color=INK, lw=1.2,
                                 shrinkA=0, shrinkB=0))
    axb.text((x0 + x1) / 2, 0.33, r"$\Delta x$", ha="center", va="bottom",
             fontsize=14, color=INK)
    axb.text(-1.25, 0.30, r"$p_X$", color=ORANGE, fontsize=15, ha="right",
             va="center")
    axb.set_ylim(0, 0.46)
    axb.set_yticks([])
    axb.set_xticks([-2, -1, 0, 1])
    axb.set_xlabel(r"$x$", fontsize=14)
    for s in ("top", "right", "left"):
        axb.spines[s].set_visible(False)

    # left strip: p_Y drawn sideways (density grows to the left)
    axl.plot(p_y, ys, color=ORANGE, lw=2.2)
    k = (ys >= y0) & (ys <= y1)
    axl.fill_betweenx(ys[k], 0, p_y[k], color=ORANGE, alpha=0.35, lw=0)
    axl.annotate("", xy=(0.27, y1), xytext=(0.27, y0),
                 arrowprops=dict(arrowstyle="<->", color=INK, lw=1.2,
                                 shrinkA=0, shrinkB=0))
    axl.text(0.30, (y0 + y1) / 2, r"$\Delta y$", ha="right", va="center",
             fontsize=14, color=INK)
    axl.text(0.45, 1.25, r"$p_Y$", color=ORANGE, fontsize=15, ha="center",
             va="center")
    axl.set_xlim(0.75, 0)                # inverted: density grows leftwards
    axl.set_xticks([])
    axl.tick_params(labelleft=False, left=False)
    for s in ("top", "left", "bottom"):
        axl.spines[s].set_visible(False)
    for a in (ax, axb, axl):
        for s in a.spines.values():
            s.set_color("black")
    fl.save(fig, "mdl-prob-change-of-variables")
    shutil.copyfile(
        os.path.join(fl.IMG_DIR, "mdl-prob-change-of-variables.svg"),
        os.path.join(fl.IMG_DIR, "auto", "mdl-prob-change-of-variables.svg"))


def _black_axes(ax, fs=13):
    """Left/bottom spines in black, no top/right spines."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("black")
    ax.tick_params(colors="black", labelsize=fs)


def _tag(ax, text):
    ax.text(-0.02, 1.04, text, transform=ax.transAxes, fontsize=14,
            color=INK, ha="right", va="bottom")


def _dot(ax, x, y, color, filled=True, z=5):
    ax.plot([x], [y], "o", ms=7, mec=color, mew=1.8,
            mfc=color if filled else "white", zorder=z)


def fig_cdf_types():
    """Three c.d.f.s.  (a) The dart's distance from the center, F(r) = r^2 on
    [0, 1]: continuous.  (b) A fair coin reporting 0 or 1: a staircase with
    jumps of 1/2 at 0 and 1; filled dots mark the value taken at each jump
    (the upper one: right-continuity), open dots the limit from the left.
    (c) ReLU(Z) with Z ~ N(0, 1): 0 for x < 0, a jump to 1/2 at 0 (the atom
    P(Z <= 0) = 1/2), then Phi(x) for x >= 0."""
    from scipy.stats import norm
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.1),
                             gridspec_kw=dict(wspace=0.32))
    ax, bx, cx = axes

    # (a) continuous
    r = np.linspace(0, 1, 200)
    ax.plot([-0.4, 0], [0, 0], color=BLUE, lw=2.4)
    ax.plot(r, r ** 2, color=BLUE, lw=2.4)
    ax.plot([1, 1.4], [1, 1], color=BLUE, lw=2.4)
    ax.text(0.42, 0.55, r"$F(r)=r^2$", color=BLUE, fontsize=15,
            ha="right", va="center")
    ax.set_xlim(-0.4, 1.4)
    ax.set_xticks([0, 1])
    ax.set_xlabel(r"$r$", fontsize=14, labelpad=1)

    # (b) staircase
    for (x0, x1, y) in ((-0.6, 0, 0.0), (0, 1, 0.5), (1, 1.6, 1.0)):
        bx.plot([x0, x1], [y, y], color=ORANGE, lw=2.4, solid_capstyle="butt")
    for xj, lo, hi in ((0, 0.0, 0.5), (1, 0.5, 1.0)):
        bx.plot([xj, xj], [lo, hi], ":", color=GRAY, lw=1.2, zorder=1)
        _dot(bx, xj, lo, ORANGE, filled=False)
        _dot(bx, xj, hi, ORANGE, filled=True)
    bx.set_xlim(-0.6, 1.6)
    bx.set_xticks([0, 1])
    bx.set_xlabel(r"$x$", fontsize=14, labelpad=1)

    # (c) mixed
    xs = np.linspace(0, 3, 300)
    cx.plot([-2.0, 0], [0, 0], color=BLUE, lw=2.4, solid_capstyle="butt")
    cx.plot(xs, norm.cdf(xs), color=BLUE, lw=2.4)
    cx.plot([0, 0], [0, 0.5], ":", color=GRAY, lw=1.2, zorder=1)
    _dot(cx, 0, 0, BLUE, filled=False)
    _dot(cx, 0, 0.5, BLUE, filled=True)
    cx.text(0.25, 0.25, "jump of " + r"$\frac{1}{2}$", color=INK,
            fontsize=13, ha="left", va="center")
    cx.text(1.45, 0.78, r"$\Phi(x)$", color=BLUE, fontsize=15,
            ha="left", va="top")
    cx.set_xlim(-2.0, 3.0)
    cx.set_xticks([-2, 0, 2])
    cx.set_xlabel(r"$x$", fontsize=14, labelpad=1)

    for a, tag in ((ax, "(a)"), (bx, "(b)"), (cx, "(c)")):
        a.set_ylim(-0.06, 1.1)
        a.set_yticks([0, 0.5, 1])
        a.set_yticklabels(["0", r"$\frac{1}{2}$", "1"])
        a.axhline(1, color=LIGHT, lw=0.9, ls="--", zorder=0)
        _black_axes(a)
        _tag(a, tag)
    fl.save(fig, "mdl-prob-cdf-types")


def fig_markov_step():
    """Markov's inequality: on [0, inf) the step t * 1{x >= t} (here t = 1.5)
    never rises above the line y = x.  Below t the step is 0 <= x; from t on
    it is t <= x.  The gap between the two (shaded) is what Markov throws
    away; averaging both sides over X gives t P(X >= t) <= E[X]."""
    t, xmax = 1.5, 3.2
    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    xs = np.array([0, xmax])
    ax.fill_between([0, t], [0, 0], [0, t], color=BLUE, alpha=0.10, lw=0)
    ax.fill_between([t, xmax], [t, t], [t, xmax], color=BLUE, alpha=0.10,
                    lw=0)
    ax.plot(xs, xs, color=BLUE, lw=2.4, zorder=3)
    ax.plot([0, t], [0, 0], color=ORANGE, lw=3.2, solid_capstyle="butt",
            zorder=4)
    ax.plot([t, xmax], [t, t], color=ORANGE, lw=3.2, solid_capstyle="butt",
            zorder=4)
    ax.plot([t, t], [0, t], ":", color=GRAY, lw=1.2, zorder=2)
    _dot(ax, t, 0, ORANGE, filled=False)
    _dot(ax, t, t, ORANGE, filled=True)
    ax.text(2.55, 2.95, r"$y=x$", color=BLUE, fontsize=15, ha="right",
            va="center")
    ax.text(2.35, t - 0.12, r"$t\,\mathbf{1}\{x\geq t\}$", color=ORANGE,
            fontsize=15, ha="center", va="top")
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.12, xmax)
    ax.set_xticks([0, t])
    ax.set_xticklabels(["0", r"$t$"], fontsize=14)
    ax.set_yticks([0, t])
    ax.set_yticklabels(["0", r"$t$"], fontsize=14)
    ax.set_xlabel(r"$x$", fontsize=14, labelpad=1)
    _black_axes(ax, fs=14)
    fl.save(fig, "mdl-prob-markov-step")


def fig_dart_three_questions():
    """The dart, uniform on the unit disk, and three questions about (X, Y).
    (a) Dependent: the chord at height Y = 0.9 confines |X| <= sqrt(1 - 0.81)
    = 0.44, while the chord at Y = -0.3 is more than twice as long; yet every chord is
    centred on x = 0, so E[X | Y] = 0 (dots).  (b) Uncorrelated: the product
    xy is positive in the first and third quadrants and negative in the other
    two, and the four quadrants carry equal mass, so Cov(X, Y) = E[XY] = 0."""
    th = np.linspace(0, 2 * np.pi, 400)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.0, 4.3),
                                 gridspec_kw=dict(wspace=0.15))
    for a in (ax, bx):
        a.fill(np.cos(th), np.sin(th), color=BLUE, alpha=0.10, lw=0)
        a.plot(np.cos(th), np.sin(th), color=BLUE, lw=2.0)
        fl.axis_cross(a, (-1.3, 1.35), (-1.15, 1.35), color=INK, lw=0.9)
        fl.clean_axes(a, lim=((-1.35, 1.45), (-1.45, 1.4)), hide=True)
    ax.text(1.42, -0.08, r"$x$", fontsize=14, ha="right", va="top")
    ax.text(0.06, 1.36, r"$y$", fontsize=14, ha="left", va="top")
    bx.text(1.42, -0.08, r"$x$", fontsize=14, ha="right", va="top")
    bx.text(0.06, 1.36, r"$y$", fontsize=14, ha="left", va="top")

    # (a) chords at y = 0.9 and y = -0.3, midpoints on the y-axis
    for y in (0.9, -0.3):
        h = math.sqrt(1 - y * y)
        ax.plot([-h, h], [y, y], color=ORANGE, lw=3.2,
                solid_capstyle="butt", zorder=4)
        _dot(ax, 0, y, GREEN, z=6)
    h9 = math.sqrt(1 - 0.81)
    for sx in (-1, 1):
        ax.text(sx * h9, 0.80, rf"${sx * h9:.2f}$",
                fontsize=12, color=fl.T.ORANGE.dark, ha="center", va="top")
    ax.text(0.62, 0.97, r"$Y=0.9$", fontsize=13, color=INK, ha="left",
            va="center")
    ax.text(0, -1.25, r"chord midpoints: $\mathbb{E}[X\mid Y]=0$",
            fontsize=14, color=fl.T.GREEN.dark, ha="center", va="top")

    # (b) sign of xy in each quadrant
    for sx in (-1, 1):
        for sy in (-1, 1):
            pos = sx * sy > 0
            bx.text(0.45 * sx, 0.45 * sy, r"$xy>0$" if pos else r"$xy<0$",
                    fontsize=14, color=fl.T.GREEN.dark if pos
                    else fl.T.RED.dark, ha="center", va="center")
    bx.text(0, -1.25, r"$\mathrm{Cov}(X,Y)=\mathbb{E}[XY]=0$", fontsize=14,
            color=INK, ha="center", va="top")
    for a, tag in ((ax, "(a)"), (bx, "(b)")):
        a.text(-1.35, 1.38, tag, fontsize=14, color=INK, ha="left", va="top")
    fl.save(fig, "mdl-prob-dart-three-questions")


FIGURES = [fig_change_of_variables, fig_cdf_types, fig_markov_step,
           fig_dart_three_questions]


def main():
    fl.WRITTEN.clear()
    for fn in FIGURES:
        fn()
    for path in fl.WRITTEN:
        print(f"wrote {path} ({os.path.getsize(path):,d} bytes)")


if __name__ == "__main__":
    main()
